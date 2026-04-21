#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.envs.utils import close_envs
from lerobot.scripts.scene_future_hint_probe import (
    BallPosNormalizer,
    build_parser as build_probe_parser,
    build_policy_batch,
    choose_probe_steps,
    clone_obs_history,
    collect_episode_start_context,
    compute_action_diff_metrics,
    generate_action_chunk,
    json_ready,
    make_noise,
    parse_int_list,
    query_ball_pos_from_trace,
    read_dense_progress_state,
    read_future_gate_debug,
    reference_trace_qualifies,
    reference_trace_sort_key,
    resolve_policy_and_env,
    restore_snapshot_context,
    run_reference_episode,
    trace_metrics,
)
from lerobot.utils.utils import init_logging
from lerobot.utils.random_utils import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = build_probe_parser()
    parser.description = (
        "Direct future-ball sensitivity test. Uses the same observation history and diffusion noise "
        "while swapping the scene future branch between normal / zero / shuffled / shifted futures."
    )
    parser.add_argument(
        "--shift_xyz",
        dest="shift_xyz",
        default="0.08,0.00,0.00",
        help="Spatial offset in meters for the shifted future variant, formatted as 'dx,dy,dz'.",
    )
    parser.add_argument(
        "--include_default_history_variant",
        dest="include_default_history_variant",
        action="store_true",
        help="Also evaluate the checkpoint's native history-only inference path as `default`.",
    )
    parser.add_argument(
        "--max_probes",
        dest="max_probes",
        type=int,
        default=None,
        help="Optional hard cap across all probes after per-episode selection.",
    )
    return parser


def make_output_dir(raw_output_dir: str | None) -> Path:
    if raw_output_dir:
        path = Path(raw_output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("outputs/future_sensitivity") / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_vec3(raw: str) -> np.ndarray:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError(f"`--shift_xyz` must contain exactly 3 comma-separated values. Got {raw!r}.")
    return np.asarray(values, dtype=np.float32)


def safe_mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(valid)) if valid else None


def safe_bool_fraction(values: list[bool]) -> float | None:
    return float(np.mean(values)) if values else None


def tensor_to_list(value: Tensor | np.ndarray | list[float]) -> list[float]:
    if isinstance(value, Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    return list(value)


def compute_component_diff_metrics(
    normal_chunk: Tensor,
    candidate_chunk: Tensor,
    first_k: int,
) -> dict[str, Any]:
    first_k = max(1, min(int(first_k), int(candidate_chunk.shape[0])))
    diff = (candidate_chunk - normal_chunk)[:first_k].detach().cpu().numpy()
    metrics: dict[str, Any] = {}

    if diff.shape[-1] >= 3:
        translation = diff[:, :3]
        translation_l2 = np.linalg.norm(translation, axis=-1)
        metrics["first_k_translation_l2_vs_normal"] = translation_l2.tolist()
        metrics["first_k_translation_mean_l2_vs_normal"] = float(np.mean(translation_l2))

    if diff.shape[-1] >= 6:
        rotation = diff[:, 3:6]
        rotation_l2 = np.linalg.norm(rotation, axis=-1)
        metrics["first_k_rotation_l2_vs_normal"] = rotation_l2.tolist()
        metrics["first_k_rotation_mean_l2_vs_normal"] = float(np.mean(rotation_l2))

    if diff.shape[-1] >= 7:
        gripper = np.abs(diff[:, 6:])
        metrics["first_k_gripper_abs_vs_normal"] = gripper.tolist()
        metrics["first_k_gripper_abs_mean_vs_normal"] = float(np.mean(gripper))

    return metrics


def compute_translation_toward_hint_metrics(
    action_chunk: Tensor,
    *,
    grip_pos: np.ndarray | list[float] | None,
    hint_ball_pos: np.ndarray | list[float] | None,
    first_k: int,
) -> dict[str, Any]:
    if grip_pos is None or hint_ball_pos is None:
        return {
            "first_k_translation_cos_to_hint": None,
            "first_k_translation_proj_to_hint": None,
            "first_k_mean_translation_cos_to_hint": None,
            "first_k_mean_translation_proj_to_hint": None,
        }

    action_np = action_chunk.detach().cpu().numpy()
    if action_np.ndim != 2 or action_np.shape[-1] < 3:
        return {
            "first_k_translation_cos_to_hint": None,
            "first_k_translation_proj_to_hint": None,
            "first_k_mean_translation_cos_to_hint": None,
            "first_k_mean_translation_proj_to_hint": None,
        }

    first_k = max(1, min(int(first_k), int(action_np.shape[0])))
    current = np.asarray(grip_pos, dtype=np.float32).reshape(-1)[:3].copy()
    target = np.asarray(hint_ball_pos, dtype=np.float32).reshape(-1)[:3]
    cosines: list[float | None] = []
    projections: list[float | None] = []

    for step_idx in range(first_k):
        delta = action_np[step_idx, :3]
        toward = target - current
        toward_norm = float(np.linalg.norm(toward))
        delta_norm = float(np.linalg.norm(delta))

        if toward_norm <= 1e-8 or delta_norm <= 1e-8:
            cosines.append(None)
            projections.append(None)
        else:
            toward_unit = toward / toward_norm
            cosines.append(float(np.dot(delta, toward_unit) / delta_norm))
            projections.append(float(np.dot(delta, toward_unit)))

        current = current + delta

    return {
        "first_k_translation_cos_to_hint": cosines,
        "first_k_translation_proj_to_hint": projections,
        "first_k_mean_translation_cos_to_hint": safe_mean(cosines),
        "first_k_mean_translation_proj_to_hint": safe_mean(projections),
    }


def select_future_raw(
    *,
    variant_name: str,
    trace: Any,
    donor_trace: Any,
    probe_step: int,
    correct_delta: int,
    shift_xyz: np.ndarray,
) -> np.ndarray:
    if variant_name == "normal":
        raw_future, _, _ = query_ball_pos_from_trace(trace, probe_step + correct_delta)
        return raw_future
    if variant_name == "zeros":
        return np.zeros_like(np.asarray(trace.ball_pos_tape[0], dtype=np.float32))
    if variant_name == "shuffled":
        raw_future, _, _ = query_ball_pos_from_trace(donor_trace, probe_step + correct_delta)
        return raw_future
    if variant_name == "shifted":
        raw_future, _, _ = query_ball_pos_from_trace(trace, probe_step + correct_delta)
        return np.asarray(raw_future, dtype=np.float32) + shift_xyz
    raise ValueError(f"Unsupported variant: {variant_name}")


def generate_action_chunk_with_direct_future(
    ctx: Any,
    obs_history: list[dict[str, Tensor]],
    *,
    normalized_future_override: Tensor,
    noise_seed: int,
) -> tuple[Tensor, dict[str, float | None] | None]:
    if ctx.policy.config.use_kalman_future:
        batch = build_policy_batch(ctx, obs_history, ball_pos_override=None)
    else:
        batch = build_policy_batch(ctx, obs_history, ball_pos_override=normalized_future_override)

    noise = make_noise(ctx.policy, noise_seed)
    diffusion = ctx.policy.diffusion
    original_predict = diffusion._predict_future_observation_with_kalman
    override = normalized_future_override.detach().clone()
    used_override = {"count": 0}

    def patched_predict(
        obs_history_tensor: Tensor,
        batch_tensor: dict[str, Tensor],
        future_steps: Tensor | int | float | None = None,
    ) -> Tensor:
        if (
            used_override["count"] == 0
            and obs_history_tensor.ndim == 3
            and obs_history_tensor.shape[-1] == int(ctx.ball_pos_dim)
        ):
            used_override["count"] += 1
            return override.to(device=obs_history_tensor.device, dtype=obs_history_tensor.dtype).expand(
                obs_history_tensor.shape[0], -1
            )
        return original_predict(obs_history_tensor, batch_tensor, future_steps=future_steps)

    if ctx.policy.config.use_kalman_future:
        diffusion._predict_future_observation_with_kalman = patched_predict

    try:
        autocast_ctx = torch.autocast(device_type=ctx.device_type) if ctx.use_amp else nullcontext()
        with torch.inference_mode(), autocast_ctx:
            actions = diffusion.generate_actions(batch, noise=noise)
        gate_debug = read_future_gate_debug(ctx.policy)
    finally:
        if ctx.policy.config.use_kalman_future:
            diffusion._predict_future_observation_with_kalman = original_predict

    if actions.ndim != 3 or actions.shape[0] != 1:
        raise RuntimeError(f"Expected action chunk shape (1, T, D), got {tuple(actions.shape)}.")

    env_actions: list[Tensor] = []
    for chunk_idx in range(actions.shape[1]):
        action_t = actions[:, chunk_idx]
        action_t = ctx.postprocessor(action_t)
        action_transition = {"action": action_t}
        action_transition = ctx.env_postprocessor(action_transition)
        env_actions.append(action_transition["action"][0].detach().clone().to("cpu"))
    return torch.stack(env_actions, dim=0), gate_debug


def collect_reference_traces(ctx: Any, args: argparse.Namespace) -> tuple[list[Any], list[dict[str, Any]]]:
    reference_traces: list[Any] = []
    all_reference_traces: list[Any] = []
    attempts: list[dict[str, Any]] = []

    has_filters = bool(args.reference_require_success) or (
        args.reference_min_reward_sum is not None
    ) or (args.reference_min_grasp_count is not None)
    default_max_attempts = max(int(args.reference_episodes), int(args.reference_episodes) * (8 if has_filters else 1))
    reference_max_attempts = (
        int(args.reference_max_attempts) if args.reference_max_attempts is not None else default_max_attempts
    )

    for attempt_index in range(reference_max_attempts):
        if len(reference_traces) >= int(args.reference_episodes):
            break

        episode_seed = None if args.episode_start_seed is None else int(args.episode_start_seed + attempt_index)
        start_context = collect_episode_start_context(ctx, episode_seed)
        trace = run_reference_episode(
            ctx,
            episode_index=attempt_index,
            seed=episode_seed,
            start_context=start_context,
            capture_probe_steps=set(),
        )
        all_reference_traces.append(trace)
        metrics = trace_metrics(trace)
        accepted = reference_trace_qualifies(trace, args)
        attempts.append(
            {
                "attempt_index": attempt_index,
                "seed": episode_seed,
                "accepted": accepted,
                **metrics,
            }
        )
        logging.info(
            "Collected reference attempt=%d accepted=%s seed=%s len=%d success=%s sum_reward=%.3f grasp_count=%d",
            attempt_index,
            accepted,
            episode_seed,
            metrics["len"],
            metrics["success"],
            metrics["sum_reward"],
            metrics["ball_grasp_count"],
        )
        if accepted:
            reference_traces.append(trace)

    if len(reference_traces) < int(args.reference_episodes):
        if len(all_reference_traces) < int(args.reference_episodes):
            raise RuntimeError(
                "Not enough reference episodes were collected. "
                f"Needed {int(args.reference_episodes)}, got {len(all_reference_traces)}."
            )
        logging.warning(
            "Not enough reference episodes matched filters; falling back to best collected attempts. "
            "matched=%d requested=%d attempts=%d",
            len(reference_traces),
            int(args.reference_episodes),
            reference_max_attempts,
        )
        fallback = sorted(all_reference_traces, key=reference_trace_sort_key, reverse=True)[: int(args.reference_episodes)]
        reference_traces = sorted(fallback, key=lambda item: int(item.episode_index))

    return reference_traces, attempts


def build_probe_records(
    ctx: Any,
    args: argparse.Namespace,
    reference_traces: list[Any],
    *,
    correct_delta: int,
    alt_deltas: list[int],
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    selected_probe_steps_by_episode: dict[int, list[int]] = {}
    selected_probe_details_by_episode: dict[int, list[dict[str, Any]]] = {}
    required_same_episode_delta = max([correct_delta, *alt_deltas]) if alt_deltas else correct_delta
    required_other_episode_delta = correct_delta

    for episode_index, trace in enumerate(reference_traces):
        donor_trace = reference_traces[(episode_index + 1) % len(reference_traces)]
        probe_steps, probe_details = choose_probe_steps(
            requested_probe_steps=parse_int_list(args.probe_steps),
            auto_probes_per_episode=int(args.auto_probes_per_episode),
            probe_selection_mode=str(args.probe_selection_mode),
            probe_min_step=int(args.probe_min_step),
            probe_max_step=args.probe_max_step,
            probe_min_gap=int(args.probe_min_gap),
            probe_min_motion_score=float(args.probe_min_motion_score),
            trace=trace,
            donor_trace=donor_trace,
            correct_delta=correct_delta,
            alt_deltas=alt_deltas,
            required_same_episode_delta=required_same_episode_delta,
            required_other_episode_delta=required_other_episode_delta,
        )
        selected_probe_steps_by_episode[episode_index] = probe_steps
        selected_probe_details_by_episode[episode_index] = probe_details
        logging.info("Episode=%d donor=%d selected probe steps=%s", episode_index, donor_trace.episode_index, probe_steps)

    replayed_traces = list(reference_traces)
    for episode_index, probe_steps in selected_probe_steps_by_episode.items():
        if not probe_steps:
            continue
        trace = reference_traces[episode_index]
        replayed = run_reference_episode(
            ctx,
            episode_index=trace.episode_index,
            seed=trace.seed,
            start_context=trace.start_context,
            capture_probe_steps=set(probe_steps),
        )
        replayed_traces[episode_index].probe_snapshots = replayed.probe_snapshots

    probe_records: list[dict[str, Any]] = []
    for episode_index, trace in enumerate(replayed_traces):
        donor_trace = replayed_traces[(episode_index + 1) % len(replayed_traces)]
        for detail in selected_probe_details_by_episode.get(episode_index, []):
            probe_step = int(detail["probe_step"])
            snapshot = trace.probe_snapshots.get(probe_step)
            if snapshot is None:
                logging.warning("Missing replay snapshot for episode=%d probe_step=%d; skipping.", episode_index, probe_step)
                continue
            probe_records.append(
                {
                    "episode_index": int(trace.episode_index),
                    "donor_episode_index": int(donor_trace.episode_index),
                    "trace": trace,
                    "donor_trace": donor_trace,
                    "probe_step": probe_step,
                    "probe_selection": detail,
                    "snapshot": snapshot,
                }
            )

    if args.max_probes is not None:
        probe_records = probe_records[: max(0, int(args.max_probes))]

    if not probe_records:
        raise RuntimeError("No valid probe snapshots were selected.")

    return probe_records, selected_probe_details_by_episode


def summarize_variant(records: list[dict[str, Any]], variant_name: str) -> dict[str, Any]:
    diff_chunk_l2 = []
    diff_rel_chunk_l2 = []
    diff_first_k = []
    diff_translation = []
    diff_rotation = []
    diff_gripper = []
    gate_means = []
    own_hint_cos = []
    normal_hint_cos = []
    follows_own_hint = []

    for probe in records:
        variant = probe["variants"].get(variant_name)
        if variant is None:
            continue

        diff = variant.get("diff_vs_normal") or {}
        components = variant.get("component_diff_vs_normal") or {}
        own_trend = variant.get("translation_trend_to_own_hint") or {}
        normal_trend = variant.get("translation_trend_to_normal_hint") or {}
        gate_debug = variant.get("future_gate_debug") or {}

        diff_chunk_l2.append(diff.get("chunk_l2_vs_correct"))
        diff_rel_chunk_l2.append(diff.get("relative_chunk_l2_vs_correct"))
        diff_first_k.append(diff.get("first_k_mean_l2_vs_correct"))
        diff_translation.append(components.get("first_k_translation_mean_l2_vs_normal"))
        diff_rotation.append(components.get("first_k_rotation_mean_l2_vs_normal"))
        diff_gripper.append(components.get("first_k_gripper_abs_mean_vs_normal"))
        gate_means.append(gate_debug.get("future_ball_gate_mean") or gate_debug.get("future_ball_pos_gate_mean"))

        own_mean = own_trend.get("first_k_mean_translation_cos_to_hint")
        normal_mean = normal_trend.get("first_k_mean_translation_cos_to_hint")
        own_hint_cos.append(own_mean)
        normal_hint_cos.append(normal_mean)
        if own_mean is not None and normal_mean is not None:
            follows_own_hint.append(float(own_mean) > float(normal_mean))

    return {
        "num_probes": len(records),
        "mean_chunk_l2_vs_normal": safe_mean(diff_chunk_l2),
        "mean_relative_chunk_l2_vs_normal": safe_mean(diff_rel_chunk_l2),
        "mean_first_k_l2_vs_normal": safe_mean(diff_first_k),
        "mean_first_k_translation_l2_vs_normal": safe_mean(diff_translation),
        "mean_first_k_rotation_l2_vs_normal": safe_mean(diff_rotation),
        "mean_first_k_gripper_abs_vs_normal": safe_mean(diff_gripper),
        "mean_future_ball_gate_mean": safe_mean(gate_means),
        "mean_first_k_translation_cos_to_own_hint": safe_mean(own_hint_cos),
        "mean_first_k_translation_cos_to_normal_hint": safe_mean(normal_hint_cos),
        "fraction_probes_prefers_own_hint": safe_bool_fraction(follows_own_hint),
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = make_output_dir(args.output_dir)
    init_logging(log_file=output_dir / "run.log")
    set_seed(args.seed)

    shift_xyz = parse_vec3(args.shift_xyz)
    ctx = None
    try:
        ctx, meta = resolve_policy_and_env(args)
        if str(meta.get("policy_model")) not in {"scene_only", "robot_scene"}:
            raise ValueError(
                f"This script expects a scene-conditioned policy (`scene_only` or `robot_scene`). "
                f"Got model={meta.get('policy_model')!r}."
            )

        correct_delta = int(meta["future_condition_delta"])
        alt_deltas = [delta for delta in parse_int_list(args.alt_deltas) if delta > 0 and delta != correct_delta]
        logging.info(
            "Loaded policy model=%s future_delta=%d n_action_steps=%d num_inference_steps=%s use_kalman_future=%s",
            meta["policy_model"],
            correct_delta,
            meta["n_action_steps"],
            meta["num_inference_steps"],
            bool(ctx.policy.config.use_kalman_future),
        )

        reference_traces, reference_attempts = collect_reference_traces(ctx, args)
        probe_records, selected_probe_details_by_episode = build_probe_records(
            ctx,
            args,
            reference_traces,
            correct_delta=correct_delta,
            alt_deltas=alt_deltas,
        )

        normalizer = BallPosNormalizer(ctx.preprocessor, ctx.future_ball_pos_key)
        variant_names = ["normal", "zeros", "shuffled", "shifted"]
        if bool(args.include_default_history_variant):
            variant_names = ["default", *variant_names]

        results: list[dict[str, Any]] = []
        first_k = max(1, int(args.compare_first_steps))

        for probe_idx, probe in enumerate(probe_records):
            trace = probe["trace"]
            donor_trace = probe["donor_trace"]
            snapshot = probe["snapshot"]
            probe_step = int(probe["probe_step"])
            noise_seed = int(args.seed + 9_000_000 + trace.episode_index * 100_000 + probe_step * 1_000)

            restore_snapshot_context(ctx, snapshot.sim_state, seed=trace.seed)
            dense_state = read_dense_progress_state(ctx)
            grip_pos = dense_state.get("grip_pos")

            logging.info(
                "Running probe %d/%d episode=%d donor=%d probe_step=%d abs_step=%d",
                probe_idx + 1,
                len(probe_records),
                trace.episode_index,
                donor_trace.episode_index,
                probe_step,
                snapshot.abs_step,
            )

            normal_raw_future = select_future_raw(
                variant_name="normal",
                trace=trace,
                donor_trace=donor_trace,
                probe_step=probe_step,
                correct_delta=correct_delta,
                shift_xyz=shift_xyz,
            )
            normal_override = normalizer(normal_raw_future)
            normal_chunk, normal_gate = generate_action_chunk_with_direct_future(
                ctx,
                clone_obs_history(snapshot.obs_history),
                normalized_future_override=normal_override,
                noise_seed=noise_seed,
            )

            variant_results: dict[str, Any] = {}
            if bool(args.include_default_history_variant):
                default_chunk, default_gate = generate_action_chunk(
                    ctx,
                    clone_obs_history(snapshot.obs_history),
                    noise_seed=noise_seed,
                    ball_pos_override=None,
                )
                default_diff = compute_action_diff_metrics(
                    correct_chunk=normal_chunk,
                    candidate_chunk=default_chunk,
                    compare_first_steps=first_k,
                )
                default_diff["relative_chunk_l2_vs_correct"] = float(
                    default_diff["chunk_l2_vs_correct"] / max(float(torch.linalg.norm(normal_chunk).item()), 1e-8)
                )
                variant_results["default"] = {
                    "raw_hint_ball_pos": None,
                    "normalized_hint_ball_pos": None,
                    "future_gate_debug": json_ready(default_gate),
                    "action_chunk": json_ready(default_chunk),
                    "diff_vs_normal": json_ready(default_diff),
                    "component_diff_vs_normal": json_ready(
                        compute_component_diff_metrics(normal_chunk, default_chunk, first_k)
                    ),
                    "translation_trend_to_own_hint": None,
                    "translation_trend_to_normal_hint": json_ready(
                        compute_translation_toward_hint_metrics(
                            default_chunk,
                            grip_pos=grip_pos,
                            hint_ball_pos=normal_raw_future,
                            first_k=first_k,
                        )
                    ),
                }

            for variant_name in ["normal", "zeros", "shuffled", "shifted"]:
                raw_future = select_future_raw(
                    variant_name=variant_name,
                    trace=trace,
                    donor_trace=donor_trace,
                    probe_step=probe_step,
                    correct_delta=correct_delta,
                    shift_xyz=shift_xyz,
                )
                normalized_future = normalizer(raw_future)
                if variant_name == "normal":
                    chunk = normal_chunk
                    gate_debug = normal_gate
                else:
                    chunk, gate_debug = generate_action_chunk_with_direct_future(
                        ctx,
                        clone_obs_history(snapshot.obs_history),
                        normalized_future_override=normalized_future,
                        noise_seed=noise_seed,
                    )

                diff_metrics = compute_action_diff_metrics(
                    correct_chunk=normal_chunk,
                    candidate_chunk=chunk,
                    compare_first_steps=first_k,
                )
                diff_metrics["relative_chunk_l2_vs_correct"] = float(
                    diff_metrics["chunk_l2_vs_correct"] / max(float(torch.linalg.norm(normal_chunk).item()), 1e-8)
                )

                variant_results[variant_name] = {
                    "raw_hint_ball_pos": tensor_to_list(raw_future),
                    "normalized_hint_ball_pos": tensor_to_list(normalized_future),
                    "future_gate_debug": json_ready(gate_debug),
                    "action_chunk": json_ready(chunk),
                    "diff_vs_normal": json_ready(diff_metrics),
                    "component_diff_vs_normal": json_ready(
                        compute_component_diff_metrics(normal_chunk, chunk, first_k)
                    ),
                    "translation_trend_to_own_hint": json_ready(
                        compute_translation_toward_hint_metrics(
                            chunk,
                            grip_pos=grip_pos,
                            hint_ball_pos=raw_future,
                            first_k=first_k,
                        )
                    ),
                    "translation_trend_to_normal_hint": json_ready(
                        compute_translation_toward_hint_metrics(
                            chunk,
                            grip_pos=grip_pos,
                            hint_ball_pos=normal_raw_future,
                            first_k=first_k,
                        )
                    ),
                }

            results.append(
                {
                    "episode_index": int(trace.episode_index),
                    "donor_episode_index": int(donor_trace.episode_index),
                    "seed": trace.seed,
                    "probe_step": probe_step,
                    "abs_step": int(snapshot.abs_step),
                    "probe_selection": json_ready(probe["probe_selection"]),
                    "shift_xyz": shift_xyz.tolist(),
                    "dense_progress_state": json_ready(dense_state),
                    "variants": variant_results,
                }
            )

        summary = {
            "policy_path": str(args.policy_path),
            "output_dir": str(output_dir),
            "policy_model": meta["policy_model"],
            "use_kalman_future": bool(ctx.policy.config.use_kalman_future),
            "future_condition_delta": correct_delta,
            "future_condition_deltas": list(getattr(ctx.policy.config, "future_condition_deltas", [])),
            "shift_xyz": shift_xyz.tolist(),
            "compare_first_steps": first_k,
            "reference_attempts": reference_attempts,
            "selected_probe_details_by_episode": json_ready(selected_probe_details_by_episode),
            "num_reference_episodes": len(reference_traces),
            "num_probes": len(results),
            "variant_summaries": {
                variant_name: summarize_variant(results, variant_name)
                for variant_name in variant_names
            },
            "notes": [
                "normal = same-episode t+delta oracle future ball position",
                "zeros = all-zero future ball position",
                "shuffled = other-episode future ball position at the same relative policy step",
                "shifted = oracle future ball position plus --shift_xyz",
                "translation trend uses action[:3] against the vector from current gripper site to the hinted ball position",
            ],
        }

        (output_dir / "results.json").write_text(json.dumps(json_ready(results), indent=2, ensure_ascii=False))
        (output_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))

        print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
        print(f"Saved detailed results to: {output_dir / 'results.json'}")
        print(f"Saved summary to: {output_dir / 'summary.json'}")
    finally:
        if ctx is not None:
            close_envs({"env": ctx.vec_env})


if __name__ == "__main__":
    main()
