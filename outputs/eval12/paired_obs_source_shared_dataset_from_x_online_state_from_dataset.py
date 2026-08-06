#!/usr/bin/env python3
"""Paired LIBERO dyn-mini eval with shared offline input starting from inference x.

This keeps the original paired setup with two lanes starting from the same cached
environment start state, but changes when the two lanes begin sharing the same
policy input:

1. Before inference `x`, the dataset lane uses offline dataset observations.
2. Before inference `x`, the online lane uses its own online environment observations,
   but its state can be taken from the dataset.
3. Starting from inference `x` (1-based), both lanes use the same offline dataset
   observation for policy inference.

This is useful for testing how much early branching matters before forcing both
lanes back onto the same observation stream.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from termcolor import colored

_THIS_DIR = Path(__file__).resolve().parent
_BASE_SCRIPT_PATH = _THIS_DIR / "paired_obs_source_same_start.py"
_BASE_SPEC = importlib.util.spec_from_file_location("paired_obs_source_same_start_base", _BASE_SCRIPT_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"Failed to load base script from {_BASE_SCRIPT_PATH}")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)


def make_output_dir(
    raw_output_dir: str | None,
    *,
    switch_inference_1based: int,
    online_state_source: str,
) -> Path:
    if raw_output_dir:
        path = Path(raw_output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("outputs/eval12") / (
            f"shared_dataset_from_inf{int(switch_inference_1based)}_online_state_{online_state_source}_{stamp}"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_paired_env(
    *,
    env_cfg: Any,
    task_id: int,
) -> gym.vector.SyncVectorEnv:
    suite = _BASE.benchmark.get_benchmark_dict()[env_cfg.task]()
    camera_name = env_cfg.camera_name
    gym_kwargs = dict(env_cfg.gym_kwargs)
    gym_kwargs.pop("task_ids", None)

    def _make_one() -> Any:
        return _BASE.LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=env_cfg.task,
            camera_name=camera_name,
            init_states=env_cfg.init_states,
            episode_length=env_cfg.episode_length,
            episode_index=0,
            n_envs=1,
            control_mode=env_cfg.control_mode,
            **gym_kwargs,
        )

    # Disable autoreset so a finished lane won't silently jump into a new episode.
    return gym.vector.SyncVectorEnv(
        [_make_one, _make_one],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )


def write_pair_numeric_records(
    *,
    pair_ix: int,
    records: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"pair_{pair_ix:04d}_diff_records.jsonl"
    csv_path = output_dir / f"pair_{pair_ix:04d}_diff_records.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    fieldnames = [
        "inference_ix",
        "inference_count_1based",
        "shared_input_active",
        "shared_input_source",
        "dataset_frame_idx",
        "online_env_step_before_inference",
        "input_img1_rmse",
        "input_img2_rmse",
        "input_state_rmse",
        "input_state_pos_rmse",
        "input_state_angle_rmse",
        "input_state_gripper_rmse",
        "output_dataset_vs_online_rmse",
        "output_dataset_vs_online_pos_rmse",
        "output_dataset_vs_online_angle_rmse",
        "output_dataset_vs_online_gripper_rmse",
        "output_dataset_vs_demo_rmse",
        "output_dataset_vs_demo_pos_rmse",
        "output_dataset_vs_demo_angle_rmse",
        "output_dataset_vs_demo_gripper_rmse",
        "output_online_vs_demo_rmse",
        "output_online_vs_demo_pos_rmse",
        "output_online_vs_demo_angle_rmse",
        "output_online_vs_demo_gripper_rmse",
        "compared_action_steps",
        "executed_action_steps",
        "dataset_done_after_chunk",
        "online_done_after_chunk",
        "stopped_due_to_early_done",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = _BASE.csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "inference_ix": record.get("inference_ix"),
                    "inference_count_1based": record.get("inference_count_1based"),
                    "shared_input_active": record.get("shared_input_active"),
                    "shared_input_source": record.get("shared_input_source"),
                    "dataset_frame_idx": record.get("dataset_frame_idx"),
                    "online_env_step_before_inference": record.get("online_env_step_before_inference"),
                    "input_img1_rmse": record.get("input_diff", {}).get("img1"),
                    "input_img2_rmse": record.get("input_diff", {}).get("img2"),
                    "input_state_rmse": record.get("input_diff", {}).get("state"),
                    "input_state_pos_rmse": record.get("input_diff", {}).get("state_pos"),
                    "input_state_angle_rmse": record.get("input_diff", {}).get("state_angle"),
                    "input_state_gripper_rmse": record.get("input_diff", {}).get("state_gripper"),
                    "output_dataset_vs_online_rmse": record.get("output_diff", {}).get("dataset_vs_online"),
                    "output_dataset_vs_online_pos_rmse": record.get("output_diff", {}).get("dataset_vs_online_pos"),
                    "output_dataset_vs_online_angle_rmse": record.get("output_diff", {}).get("dataset_vs_online_angle"),
                    "output_dataset_vs_online_gripper_rmse": record.get("output_diff", {}).get("dataset_vs_online_gripper"),
                    "output_dataset_vs_demo_rmse": record.get("output_diff", {}).get("dataset_vs_demo"),
                    "output_dataset_vs_demo_pos_rmse": record.get("output_diff", {}).get("dataset_vs_demo_pos"),
                    "output_dataset_vs_demo_angle_rmse": record.get("output_diff", {}).get("dataset_vs_demo_angle"),
                    "output_dataset_vs_demo_gripper_rmse": record.get("output_diff", {}).get("dataset_vs_demo_gripper"),
                    "output_online_vs_demo_rmse": record.get("output_diff", {}).get("online_vs_demo"),
                    "output_online_vs_demo_pos_rmse": record.get("output_diff", {}).get("online_vs_demo_pos"),
                    "output_online_vs_demo_angle_rmse": record.get("output_diff", {}).get("online_vs_demo_angle"),
                    "output_online_vs_demo_gripper_rmse": record.get("output_diff", {}).get("online_vs_demo_gripper"),
                    "compared_action_steps": record.get("output_diff", {}).get("compared_action_steps"),
                    "executed_action_steps": record.get("executed_action_steps"),
                    "dataset_done_after_chunk": record.get("dataset_done_after_chunk"),
                    "online_done_after_chunk": record.get("online_done_after_chunk"),
                    "stopped_due_to_early_done": record.get("stopped_due_to_early_done"),
                }
            )

    return str(jsonl_path), str(csv_path)


def write_run_numeric_summary(*, pair_records: list[dict[str, Any]], output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "all_pairs_diff_summary.csv"
    fieldnames = [
        "pair_ix",
        "seed",
        "reference_dataset_episode_index",
        "shared_input_start_inference_1based",
        "n_inference_records",
        "max_inference_count",
        "dataset_success",
        "online_success",
        "dataset_sum_reward",
        "online_sum_reward",
        "dataset_ball_grasp_count",
        "online_ball_grasp_count",
        "mean_input_img1_rmse",
        "mean_input_img2_rmse",
        "mean_input_state_rmse",
        "mean_output_dataset_vs_online_rmse",
        "mean_output_dataset_vs_demo_rmse",
        "mean_output_online_vs_demo_rmse",
        "pair_jsonl_path",
        "pair_csv_path",
        "video_path",
        "combined_diff_plot_path",
        "component_diff_plot_path",
    ]

    def mean_metric(records: list[dict[str, Any]], top_key: str, inner_key: str) -> float | None:
        values = [record.get(top_key, {}).get(inner_key) for record in records]
        values = [float(v) for v in values if v is not None]
        if not values:
            return None
        return float(np.mean(values))

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = _BASE.csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pair_records:
            records = list(pair.get("per_inference_records", []))
            writer.writerow(
                {
                    "pair_ix": pair.get("pair_ix"),
                    "seed": pair.get("seed"),
                    "reference_dataset_episode_index": pair.get("reference_dataset_episode_index"),
                    "shared_input_start_inference_1based": pair.get("shared_input_start_inference_1based"),
                    "n_inference_records": len(records),
                    "max_inference_count": pair.get("max_inference_count"),
                    "dataset_success": pair.get("dataset_lane", {}).get("success"),
                    "online_success": pair.get("online_lane", {}).get("success"),
                    "dataset_sum_reward": pair.get("dataset_lane", {}).get("sum_reward"),
                    "online_sum_reward": pair.get("online_lane", {}).get("sum_reward"),
                    "dataset_ball_grasp_count": pair.get("dataset_lane", {}).get("ball_grasp_count"),
                    "online_ball_grasp_count": pair.get("online_lane", {}).get("ball_grasp_count"),
                    "mean_input_img1_rmse": mean_metric(records, "input_diff", "img1"),
                    "mean_input_img2_rmse": mean_metric(records, "input_diff", "img2"),
                    "mean_input_state_rmse": mean_metric(records, "input_diff", "state"),
                    "mean_output_dataset_vs_online_rmse": mean_metric(records, "output_diff", "dataset_vs_online"),
                    "mean_output_dataset_vs_demo_rmse": mean_metric(records, "output_diff", "dataset_vs_demo"),
                    "mean_output_online_vs_demo_rmse": mean_metric(records, "output_diff", "online_vs_demo"),
                    "pair_jsonl_path": pair.get("numeric_records_jsonl_path"),
                    "pair_csv_path": pair.get("numeric_records_csv_path"),
                    "video_path": pair.get("video_path"),
                    "combined_diff_plot_path": pair.get("combined_diff_plot_path"),
                    "component_diff_plot_path": pair.get("component_diff_plot_path"),
                }
            )

    return str(csv_path)


def summarize_lane(rollout_data: dict[str, Any], lane_idx: int) -> dict[str, Any]:
    done_tensor = rollout_data["done"][lane_idx]
    reward_tensor = rollout_data["reward"][lane_idx]
    success_tensor = rollout_data["success"][lane_idx]
    grasp_tensor = rollout_data["ball_grasp_event"][lane_idx]

    if done_tensor.numel() == 0:
        return {
            "sum_reward": 0.0,
            "max_reward": 0.0,
            "success": False,
            "ball_grasp_count": 0,
        }

    done_hits = torch.nonzero(done_tensor, as_tuple=False).reshape(-1)
    if done_hits.numel() > 0:
        end_idx = int(done_hits[0].item())
    else:
        end_idx = int(done_tensor.shape[0] - 1)

    mask = (torch.arange(done_tensor.shape[0]) <= end_idx).to(torch.float32)
    sum_reward = float(torch.sum(reward_tensor * mask).item())
    max_reward = float(torch.max(reward_tensor * mask).item())
    success = bool(torch.any(success_tensor[: end_idx + 1]).item())
    ball_grasp_count = int(torch.sum(grasp_tensor[: end_idx + 1]).item())
    return {
        "sum_reward": sum_reward,
        "max_reward": max_reward,
        "success": success,
        "ball_grasp_count": ball_grasp_count,
    }


def rollout_paired_episode(
    *,
    env: gym.vector.SyncVectorEnv,
    policy: Any,
    dataset: Any,
    absolute_to_relative: dict[int, int] | None,
    binding: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    effective_rename_map: dict[str, str],
    execute_n_action_steps: int,
    shared_input_start_inference_1based: int,
    online_state_source: str,
    pair_seed: int,
    render_callback: Any | None,
    rollout_fps: float | None,
) -> dict[str, Any]:
    if shared_input_start_inference_1based <= 0:
        raise ValueError("shared_input_start_inference_1based must be >= 1.")
    if online_state_source not in {"online", "dataset"}:
        raise ValueError(f"Unsupported online_state_source={online_state_source!r}.")

    policy.reset()
    policy.eval()
    initial_lane_queues = _BASE.deepcopy(getattr(policy, "_queues", {}))
    lane_queues: dict[str, dict[str, Any]] = {
        "online": _BASE.deepcopy(initial_lane_queues),
    }
    observation, _ = env.reset(seed=[0, 0])
    if render_callback is not None:
        render_callback(env)

    zero_action = _BASE.zero_action_like(env)
    _BASE.check_env_attributes_and_types(env)
    max_steps = int(env.call("_max_episode_steps")[0])

    track_ball_grasp = False
    grasp_prev = np.zeros(env.num_envs, dtype=bool)
    if hasattr(env, "call"):
        try:
            raw_grasp = env.call("is_ball_grasped")
            if isinstance(raw_grasp, (list, tuple)) and len(raw_grasp) == env.num_envs:
                grasp_prev = np.asarray(raw_grasp, dtype=bool)
                track_ball_grasp = True
        except Exception:
            track_ball_grasp = False

    step = 0
    dataset_frame_idx = 0
    dataset_lane_idx = 0
    online_lane_idx = 1
    dataset_trace: list[int] = []
    online_trace: list[int] = []
    all_actions: list[torch.Tensor] = []
    all_rewards: list[torch.Tensor] = []
    all_dones: list[torch.Tensor] = []
    all_successes: list[torch.Tensor] = []
    all_ball_grasp_events: list[torch.Tensor] = []
    per_inference_records: list[dict[str, Any]] = []
    dataset_exhausted = False
    max_inference_count = 0
    lane_finished = np.zeros(2, dtype=bool)

    progbar = _BASE.trange(max_steps, desc="Running paired rollout", disable=_BASE.inside_slurm(), leave=False)

    while step < max_steps:
        if bool(np.all(lane_finished)):
            break

        inference_ix = len(per_inference_records)
        inference_count_1based = inference_ix + 1
        shared_input_active = inference_count_1based >= int(shared_input_start_inference_1based)

        if shared_input_active and dataset_frame_idx >= int(binding.dataset_length):
            dataset_exhausted = True
            break

        safe_frame = int(min(dataset_frame_idx, max(binding.dataset_length - 1, 0)))
        dataset_item = _BASE.get_dataset_item_by_absolute_index(
            dataset,
            int(binding.dataset_from_index + safe_frame),
            absolute_to_relative,
        )
        dataset_raw_batch = _BASE.default_collate([dataset_item])
        if effective_rename_map:
            renamed: dict[str, Any] = {}
            for key, value in dataset_raw_batch.items():
                renamed[effective_rename_map.get(key, key)] = value
            dataset_raw_batch = renamed
        dataset_processed = preprocessor(dataset_raw_batch)
        dataset_lane_batch = _BASE.filter_policy_input_batch(policy=policy, raw_batch=dataset_processed)

        if shared_input_active:
            online_lane_batch = _BASE.clone_batch(dataset_lane_batch)
        else:
            online_raw_batch = _BASE.prepare_online_policy_batch(
                env=env,
                observation=observation,
                env_preprocessor=env_preprocessor,
                rollout_fps=rollout_fps,
                step=step,
            )
            online_processed = preprocessor(online_raw_batch)
            online_lane_frame = _BASE.filter_policy_input_batch(
                policy=policy,
                raw_batch=_BASE.slice_env_batch(online_processed, online_lane_idx),
            )
            if online_state_source == "dataset":
                for key, value in dataset_lane_batch.items():
                    if key.startswith("observation.state") and key in online_lane_frame and isinstance(value, torch.Tensor):
                        online_lane_frame[key] = value[:, -1].detach().clone() if value.ndim >= 3 else value.detach().clone()
            online_lane_batch, lane_queues["online"] = _BASE.build_online_lane_batch_from_frame(
                policy=policy,
                raw_frame_batch=online_lane_frame,
                lane_queues=lane_queues["online"],
            )

        dataset_chunk = _BASE.predict_chunk_from_batch(
            policy=policy,
            batch=dataset_lane_batch,
            seed=int(pair_seed) + step,
            postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
        )
        online_chunk = _BASE.predict_chunk_from_batch(
            policy=policy,
            batch=online_lane_batch,
            seed=int(pair_seed) + step,
            postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
        )
        demo_chunk = _BASE.extract_demo_action_chunk(
            dataset=dataset,
            absolute_to_relative=absolute_to_relative,
            binding=binding,
            dataset_frame_idx=safe_frame,
            execute_n_action_steps=execute_n_action_steps,
            effective_rename_map=effective_rename_map,
        )

        # Finished lanes are frozen: keep them in the pair, but stop applying policy actions.
        if bool(lane_finished[dataset_lane_idx]):
            dataset_chunk = torch.zeros_like(dataset_chunk)
        if bool(lane_finished[online_lane_idx]):
            online_chunk = torch.zeros_like(online_chunk)

        per_inference_records.append(
            {
                "inference_ix": int(inference_ix),
                "inference_count_1based": int(inference_count_1based),
                "shared_input_active": bool(shared_input_active),
                "shared_input_source": "dataset" if shared_input_active else "split",
                "online_state_source": str(online_state_source),
                "dataset_frame_idx": int(safe_frame),
                "online_env_step_before_inference": int(step),
                "dataset_lane_finished_before_inference": bool(lane_finished[dataset_lane_idx]),
                "online_lane_finished_before_inference": bool(lane_finished[online_lane_idx]),
                "input_diff": _BASE.compute_input_diff_metrics(dataset_lane_batch, online_lane_batch),
                "output_diff": _BASE.compute_output_diff_metrics(
                    dataset_chunk=dataset_chunk,
                    online_chunk=online_chunk,
                    demo_chunk=demo_chunk,
                    execute_n_action_steps=execute_n_action_steps,
                    actual_executed_steps=min(
                        int(execute_n_action_steps),
                        int(dataset_chunk.shape[1]),
                        int(online_chunk.shape[1]),
                        int(demo_chunk.shape[1]) if demo_chunk is not None else int(execute_n_action_steps),
                    ),
                ),
                "stopped_due_to_early_done": False,
                "dataset_done_after_chunk": False,
                "online_done_after_chunk": False,
                "executed_action_steps": None,
            }
        )

        pred_chunk = torch.cat([dataset_chunk, online_chunk], dim=0)
        (
            observation,
            chunk_actions,
            chunk_rewards,
            chunk_dones,
            chunk_successes,
            chunk_grasps,
            grasp_prev,
            step,
            dataset_frame_idx,
            _frame_overlays,
            dataset_exhausted,
        ) = _BASE.execute_chunk(
            env=env,
            pred_chunk_cpu=pred_chunk.detach().to("cpu"),
            execute_n_action_steps=execute_n_action_steps,
            zero_action=zero_action,
            dataset_lane_idx=dataset_lane_idx,
            dataset_binding=binding,
            dataset_frame_idx=dataset_frame_idx,
            step=step,
            render_callback=render_callback,
            inference_ix=inference_ix,
            grasp_prev=grasp_prev,
            track_ball_grasp=track_ball_grasp,
        )

        chunk_done_any = np.zeros(2, dtype=bool)
        if chunk_dones:
            chunk_done_any = torch.stack(chunk_dones, dim=0).any(dim=0).to(torch.bool).cpu().numpy()
            lane_finished = np.logical_or(lane_finished, chunk_done_any)

        max_inference_count = max(max_inference_count, int(inference_ix) + 1)
        if per_inference_records and per_inference_records[-1]["inference_ix"] == inference_ix:
            per_inference_records[-1]["dataset_done_after_chunk"] = bool(chunk_done_any[dataset_lane_idx])
            per_inference_records[-1]["online_done_after_chunk"] = bool(chunk_done_any[online_lane_idx])
            per_inference_records[-1]["executed_action_steps"] = int(len(chunk_actions))

        all_actions.extend(chunk_actions)
        all_rewards.extend(chunk_rewards)
        all_dones.extend(chunk_dones)
        all_successes.extend(chunk_successes)
        all_ball_grasp_events.extend(chunk_grasps)
        dataset_trace.append(safe_frame)
        online_trace.append(step)
        progbar.update(len(chunk_actions))

    actions = torch.stack(all_actions, dim=1) if all_actions else torch.empty((2, 0, 7))
    rewards = torch.stack(all_rewards, dim=1) if all_rewards else torch.empty((2, 0))
    dones = torch.stack(all_dones, dim=1) if all_dones else torch.empty((2, 0), dtype=torch.bool)
    successes = torch.stack(all_successes, dim=1) if all_successes else torch.empty((2, 0), dtype=torch.bool)
    grasps = (
        torch.stack(all_ball_grasp_events, dim=1)
        if all_ball_grasp_events
        else torch.empty((2, 0), dtype=torch.int32)
    )

    first_record = per_inference_records[0] if per_inference_records else None
    first_input_equal = bool(first_record is not None and first_record.get("shared_input_active", False))

    return {
        _BASE.ACTION: actions,
        "reward": rewards,
        "done": dones,
        "success": successes,
        "ball_grasp_event": grasps,
        "dataset_trace": dataset_trace,
        "online_trace": online_trace,
        "first_input_equal": first_input_equal,
        "shared_input_start_inference_1based": int(shared_input_start_inference_1based),
        "shared_input_source": "dataset",
        "online_state_source": str(online_state_source),
        "reference_dataset_episode_index": int(binding.dataset_episode_index),
        "reference_dataset_length": int(binding.dataset_length),
        "dataset_exhausted": bool(dataset_exhausted),
        "per_inference_records": per_inference_records,
        "max_inference_count": int(max_inference_count),
    }


def build_pair_record(
    *,
    pair_ix: int,
    binding: Any,
    rollout_data: dict[str, Any],
    seed: int | None,
    video_path: str | None = None,
    combined_diff_plot_path: str | None = None,
    component_diff_plot_path: str | None = None,
) -> dict[str, Any]:
    dataset_lane = summarize_lane(rollout_data, 0)
    online_lane = summarize_lane(rollout_data, 1)
    record = {
        "pair_ix": int(pair_ix),
        "seed": seed,
        "reference_dataset_episode_index": int(binding.dataset_episode_index),
        "reference_dataset_length": int(binding.dataset_length),
        "shared_first_input_source": rollout_data["shared_input_source"],
        "online_state_source": rollout_data["online_state_source"],
        "first_input_equal": bool(rollout_data["first_input_equal"]),
        "dataset_lane": dataset_lane,
        "online_lane": online_lane,
        "dataset_trace": list(rollout_data["dataset_trace"]),
        "online_trace": list(rollout_data["online_trace"]),
        "dataset_exhausted": bool(rollout_data["dataset_exhausted"]),
        "per_inference_records": list(rollout_data["per_inference_records"]),
        "max_inference_count": int(rollout_data["max_inference_count"]),
        "video_path": video_path,
        "combined_diff_plot_path": combined_diff_plot_path,
        "component_diff_plot_path": component_diff_plot_path,
    }
    record["shared_input_start_inference_1based"] = int(rollout_data["shared_input_start_inference_1based"])
    record["shared_input_source"] = rollout_data["shared_input_source"]
    record["online_state_source"] = rollout_data["online_state_source"]
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired LIBERO dyn-mini eval: each lane uses its own input first, then both switch to the same offline dataset input from inference x."
    )
    parser.add_argument("--policy.path", dest="policy_path", required=True)
    parser.add_argument("--policy.device", dest="policy_device", default="cuda")
    parser.add_argument("--policy.use_amp", dest="policy_use_amp", type=_BASE.str2bool, default=True)
    parser.add_argument("--policy.n_action_steps", dest="policy_n_action_steps", type=int, default=None)
    parser.add_argument("--policy.num_inference_steps", dest="policy_num_inference_steps", type=int, default=None)
    parser.add_argument("--env.type", dest="env_type", default="libero")
    parser.add_argument("--env.task", dest="env_task", default="libero_dyn_mini")
    parser.add_argument("--env.task_ids", dest="env_task_ids", default="0")
    parser.add_argument("--env.fps", dest="env_fps", type=int, default=20)
    parser.add_argument("--env.episode_length", dest="env_episode_length", type=int, default=260)
    parser.add_argument("--env.control_mode", dest="env_control_mode", default="relative")
    parser.add_argument("--env.init_states", dest="env_init_states", type=_BASE.str2bool, default=True)
    parser.add_argument("--env.init_plan_path", dest="env_init_plan_path", default=str(_BASE.INIT_PLAN_DEFAULT_PATH))
    parser.add_argument("--env.episode_start_states_path", dest="env_episode_start_states_path", default=None)
    parser.add_argument("--env.init_plan_loop", dest="env_init_plan_loop", type=_BASE.str2bool, default=False)
    parser.add_argument("--env.ball_grasp_eval_mode", dest="env_ball_grasp_eval_mode", default="strict")
    parser.add_argument(
        "--env.ball_grasp_strict_require_pad_contact",
        dest="env_ball_grasp_strict_require_pad_contact",
        type=_BASE.str2bool,
        default=True,
    )
    parser.add_argument(
        "--env.ball_grasp_strict_lift_multiplier",
        dest="env_ball_grasp_strict_lift_multiplier",
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--env.ball_grasp_strict_grip_center_max_dist",
        dest="env_ball_grasp_strict_grip_center_max_dist",
        type=float,
        default=0.045,
    )
    parser.add_argument("--rollout.execute_n_action_steps", dest="rollout_execute_n_action_steps", type=int, default=4)
    parser.add_argument(
        "--rollout.shared_input_start_inference",
        dest="rollout_shared_input_start_inference",
        type=int,
        default=2,
        help="1-based inference count. From this inference onward, both lanes use the same offline dataset input.",
    )
    parser.add_argument(
        "--rollout.online_state_source",
        dest="rollout_online_state_source",
        choices=["online", "dataset"],
        default="dataset",
        help="Source for the online-lane state before shared input begins.",
    )
    parser.add_argument(
        "--dataset.repo_id",
        dest="dataset_repo_id",
        default="local/libero_dyn_mini_balanced500_scripted_v2",
    )
    parser.add_argument("--dataset.root", dest="dataset_root", default=str(_BASE.DATASET_ROOT_DEFAULT))
    parser.add_argument("--dataset.episodes", dest="dataset_episodes", default="0:20")
    parser.add_argument("--dataset.tolerance_s", dest="dataset_tolerance_s", type=float, default=1e-4)
    parser.add_argument("--eval.n_episodes", dest="eval_n_episodes", type=int, default=20)
    parser.add_argument("--eval.batch_size", dest="eval_batch_size", type=int, default=2)
    parser.add_argument("--output_dir", dest="output_dir", default=None)
    parser.add_argument("--seed", dest="seed", type=int, default=1000)
    parser.add_argument("--libero_legacy_obs_compat", dest="libero_legacy_obs_compat", type=_BASE.str2bool, default=False)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _BASE.init_logging()
    _BASE.register_third_party_plugins()
    logging.info(pformat(vars(args)))

    if args.env_type != "libero":
        raise ValueError(f"This script only supports env.type=libero. Got {args.env_type!r}.")
    if args.env_task != "libero_dyn_mini":
        raise ValueError(f"This script is intended for env.task=libero_dyn_mini. Got {args.env_task!r}.")
    if args.eval_batch_size != 2:
        raise ValueError("This paired script requires eval.batch_size=2 exactly.")
    if args.eval_n_episodes <= 0:
        raise ValueError("eval.n_episodes must be > 0.")
    if args.rollout_execute_n_action_steps <= 0:
        raise ValueError("rollout.execute_n_action_steps must be > 0.")
    if args.rollout_shared_input_start_inference <= 0:
        raise ValueError("rollout.shared_input_start_inference must be >= 1.")
    if args.rollout_online_state_source not in {"online", "dataset"}:
        raise ValueError(f"Unsupported rollout.online_state_source={args.rollout_online_state_source!r}.")

    device = _BASE.get_safe_torch_device(args.policy_device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    _BASE.set_seed(args.seed)

    policy_cfg = _BASE.PreTrainedConfig.from_pretrained(
        args.policy_path,
        cli_overrides=_BASE.build_cli_overrides(args),
    )
    policy_cfg.pretrained_path = Path(args.policy_path)
    policy_cfg.device = str(device)
    policy_cfg.use_amp = bool(args.policy_use_amp and policy_cfg.use_amp)

    env_cfg = _BASE.LiberoEnvConfig(
        task=args.env_task,
        task_ids=_BASE.parse_int_list(args.env_task_ids) or [0],
        fps=int(args.env_fps),
        episode_length=args.env_episode_length,
        control_mode=args.env_control_mode,
        init_states=bool(args.env_init_states),
        init_plan_path=str(_BASE.resolve_dataset_init_plan_path(args.env_init_plan_path)),
        episode_start_states_path=(
            str(Path(args.env_episode_start_states_path).expanduser().resolve())
            if args.env_episode_start_states_path
            else None
        ),
        init_plan_loop=bool(args.env_init_plan_loop),
        ball_grasp_eval_mode=args.env_ball_grasp_eval_mode,
        ball_grasp_strict_require_pad_contact=bool(args.env_ball_grasp_strict_require_pad_contact),
        ball_grasp_strict_lift_multiplier=float(args.env_ball_grasp_strict_lift_multiplier),
        ball_grasp_strict_grip_center_max_dist=float(args.env_ball_grasp_strict_grip_center_max_dist),
    )

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    selected_episodes = _BASE.parse_episode_selector(args.dataset_episodes)
    dataset, effective_rename_map = _BASE.build_dataset(
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=dataset_root,
        dataset_episodes=selected_episodes,
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
        tolerance_s=float(args.dataset_tolerance_s),
        legacy_obs_compat=bool(args.libero_legacy_obs_compat),
        rename_map={},
    )

    plan_path = _BASE.resolve_dataset_init_plan_path(args.env_init_plan_path)
    plan_rows: list[dict[str, Any]] = []
    with plan_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                plan_rows.append(payload)

    if selected_episodes is None:
        selected_episodes = list(dataset.episodes if dataset.episodes is not None else range(dataset.meta.total_episodes))
    episode_bindings = _BASE.validate_episode_plan_alignment(
        dataset=dataset,
        plan_rows=plan_rows,
        selected_episode_indices=list(selected_episodes),
        eval_n_episodes=int(args.eval_n_episodes),
    )

    output_dir = make_output_dir(
        args.output_dir,
        switch_inference_1based=int(args.rollout_shared_input_start_inference),
        online_state_source=str(args.rollout_online_state_source),
    )
    videos_dir = output_dir / "videos"
    plots_dir = output_dir / "plots"
    numeric_dir = output_dir / "numeric"
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")

    policy = _BASE.make_policy(cfg=policy_cfg, ds_meta=dataset.meta, rename_map=effective_rename_map)
    policy.eval()
    preprocessor, postprocessor = _BASE.build_processors(
        policy_cfg=policy_cfg,
        policy=policy,
        dataset_meta=dataset.meta,
        effective_rename_map=effective_rename_map,
    )
    env_preprocessor, env_postprocessor = _BASE.make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    absolute_to_relative = _BASE.get_absolute_to_relative_index(dataset)

    task_ids = _BASE.parse_int_list(args.env_task_ids) or [0]
    if task_ids != [0]:
        raise ValueError("This script currently supports exactly one task id: 0.")

    env = make_paired_env(env_cfg=env_cfg, task_id=task_ids[0])

    pair_records: list[dict[str, Any]] = []
    video_paths: list[str] = []
    pair_frames: list[np.ndarray] = []
    frozen_lane_frames: list[np.ndarray | None] = [None, None]

    def render_frame(active_env: gym.vector.VectorEnv, overlay: dict[str, Any] | None = None) -> None:
        if isinstance(active_env, gym.vector.SyncVectorEnv):
            frame_pair = np.stack([active_env.envs[i].render() for i in range(active_env.num_envs)], axis=0)
            if overlay is not None:
                if bool(overlay.get("dataset_done", False)) and frozen_lane_frames[0] is None:
                    frozen_lane_frames[0] = frame_pair[0].copy()
                if bool(overlay.get("online_done", False)) and frozen_lane_frames[1] is None:
                    frozen_lane_frames[1] = frame_pair[1].copy()
            for lane_idx, frozen_frame in enumerate(frozen_lane_frames):
                if frozen_frame is not None:
                    frame_pair[lane_idx] = frozen_frame
            combined = _BASE.annotate_pair_frame(frame_pair, overlay)
            pair_frames.append(combined)

    start_t = time.time()
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type) if policy_cfg.use_amp else nullcontext():
            for pair_ix, binding in enumerate(episode_bindings):
                pair_frames = []
                frozen_lane_frames = [None, None]
                rollout_data = rollout_paired_episode(
                    env=env,
                    policy=policy,
                    dataset=dataset,
                    absolute_to_relative=absolute_to_relative,
                    binding=binding,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    effective_rename_map=effective_rename_map,
                    execute_n_action_steps=int(args.rollout_execute_n_action_steps),
                    shared_input_start_inference_1based=int(args.rollout_shared_input_start_inference),
                    online_state_source=str(args.rollout_online_state_source),
                    pair_seed=args.seed + pair_ix,
                    render_callback=render_frame,
                    rollout_fps=float(args.env_fps),
                )
                video_path = _BASE.maybe_write_pair_video(
                    frames=pair_frames,
                    output_path=videos_dir / f"pair_{pair_ix:04d}.mp4",
                    fps=int(env.unwrapped.metadata["render_fps"]),
                )
                combined_diff_plot_path = _BASE.write_pair_combined_diff_plot(
                    records=rollout_data["per_inference_records"],
                    output_path=plots_dir / f"pair_{pair_ix:04d}_combined_diff.png",
                    max_inference_count=int(rollout_data["max_inference_count"]),
                )
                component_diff_plot_path = _BASE.write_pair_component_diff_plot(
                    records=rollout_data["per_inference_records"],
                    output_path=plots_dir / f"pair_{pair_ix:04d}_component_diff.png",
                    max_inference_count=int(rollout_data["max_inference_count"]),
                )
                numeric_jsonl_path, numeric_csv_path = write_pair_numeric_records(
                    pair_ix=pair_ix,
                    records=rollout_data["per_inference_records"],
                    output_dir=numeric_dir,
                )
                pair_record = build_pair_record(
                    pair_ix=pair_ix,
                    binding=binding,
                    rollout_data=rollout_data,
                    seed=args.seed + pair_ix,
                    video_path=video_path,
                    combined_diff_plot_path=combined_diff_plot_path,
                    component_diff_plot_path=component_diff_plot_path,
                )
                pair_record["numeric_records_jsonl_path"] = numeric_jsonl_path
                pair_record["numeric_records_csv_path"] = numeric_csv_path
                pair_records.append(pair_record)
                if video_path is not None:
                    video_paths.append(video_path)
    finally:
        _BASE.close_envs({args.env_task: {task_ids[0]: env}})

    overall_success_dataset = [bool(item["dataset_lane"]["success"]) for item in pair_records]
    overall_success_online = [bool(item["online_lane"]["success"]) for item in pair_records]
    overall_sum_reward_dataset = [float(item["dataset_lane"]["sum_reward"]) for item in pair_records]
    overall_sum_reward_online = [float(item["online_lane"]["sum_reward"]) for item in pair_records]
    overall_ball_grasp_dataset = [int(item["dataset_lane"]["ball_grasp_count"]) for item in pair_records]
    overall_ball_grasp_online = [int(item["online_lane"]["ball_grasp_count"]) for item in pair_records]

    info: dict[str, Any] = {
        "per_episode": [
            {
                "episode_ix": int(item["pair_ix"]) * 2,
                "sum_reward": float(item["dataset_lane"]["sum_reward"]),
                "max_reward": float(item["dataset_lane"]["max_reward"]),
                "success": bool(item["dataset_lane"]["success"]),
                "seed": item["seed"],
                "ball_grasp_count": int(item["dataset_lane"]["ball_grasp_count"]),
                "ball_grasp_success": bool(item["dataset_lane"]["ball_grasp_count"] > 0),
            }
            for item in pair_records
        ]
        + [
            {
                "episode_ix": int(item["pair_ix"]) * 2 + 1,
                "sum_reward": float(item["online_lane"]["sum_reward"]),
                "max_reward": float(item["online_lane"]["max_reward"]),
                "success": bool(item["online_lane"]["success"]),
                "seed": item["seed"],
                "ball_grasp_count": int(item["online_lane"]["ball_grasp_count"]),
                "ball_grasp_success": bool(item["online_lane"]["ball_grasp_count"] > 0),
            }
            for item in pair_records
        ],
        "overall": {
            "avg_sum_reward_dataset_lane": float(np.mean(overall_sum_reward_dataset)) if overall_sum_reward_dataset else float("nan"),
            "avg_sum_reward_online_lane": float(np.mean(overall_sum_reward_online)) if overall_sum_reward_online else float("nan"),
            "pc_success_dataset_lane": float(np.mean(overall_success_dataset) * 100) if overall_success_dataset else float("nan"),
            "pc_success_online_lane": float(np.mean(overall_success_online) * 100) if overall_success_online else float("nan"),
            "avg_ball_grasp_count_dataset_lane": float(np.mean(overall_ball_grasp_dataset)) if overall_ball_grasp_dataset else float("nan"),
            "avg_ball_grasp_count_online_lane": float(np.mean(overall_ball_grasp_online)) if overall_ball_grasp_online else float("nan"),
            "n_pairs": len(pair_records),
            "n_episodes": len(pair_records) * 2,
            "eval_s": time.time() - start_t,
            "eval_pair_s": (time.time() - start_t) / max(1, len(pair_records)),
        },
        "paired_rollout": {
            "shared_input_source": "dataset",
            "shared_input_start_inference_1based": int(args.rollout_shared_input_start_inference),
            "online_state_source": str(args.rollout_online_state_source),
            "first_input_equal_all_pairs": bool(all(item["first_input_equal"] for item in pair_records)),
            "pairs": pair_records,
            "video_paths": video_paths,
        },
    }

    eval_info_path = output_dir / "eval_info.json"
    eval_info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    run_numeric_summary_path = write_run_numeric_summary(pair_records=pair_records, output_dir=numeric_dir)

    summary = {
        "overall": info["overall"],
        "paired_rollout": {
            "shared_input_source": "dataset",
            "shared_input_start_inference_1based": int(args.rollout_shared_input_start_inference),
            "online_state_source": str(args.rollout_online_state_source),
            "first_input_equal_all_pairs": bool(all(item["first_input_equal"] for item in pair_records)),
            "n_pairs": len(pair_records),
        },
    }
    summary_json_path = output_dir / "eval_info_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_text = json.dumps(summary, indent=2, ensure_ascii=False)
    summary_txt_path = output_dir / "eval_info_summary.txt"
    summary_txt_path.write_text(summary_text, encoding="utf-8")

    run_meta = {
        "mode": "paired_obs_source_shared_dataset_from_x_online_state_from_dataset",
        "base_script_path": str(_BASE_SCRIPT_PATH),
        "policy_path": str(Path(args.policy_path).resolve()),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(dataset_root),
        "selected_dataset_episodes": list(selected_episodes[: args.eval_n_episodes]),
        "init_plan_path": str(plan_path),
        "episode_start_states_path": (
            str(Path(args.env_episode_start_states_path).expanduser().resolve())
            if args.env_episode_start_states_path
            else None
        ),
        "execute_n_action_steps": int(args.rollout_execute_n_action_steps),
        "shared_input_source": "dataset",
        "shared_input_start_inference_1based": int(args.rollout_shared_input_start_inference),
        "online_state_source": str(args.rollout_online_state_source),
        "eval_n_episodes": int(args.eval_n_episodes),
        "eval_batch_size": int(args.eval_batch_size),
        "env": asdict(env_cfg),
        "effective_rename_map": dict(effective_rename_map),
        "run_numeric_summary_path": run_numeric_summary_path,
    }
    run_meta_path = output_dir / "run_meta.json"
    run_meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Paired overall metrics:")
    print(json.dumps(info["overall"], indent=2, ensure_ascii=False))
    print("\nPaired config:")
    print(json.dumps(summary["paired_rollout"], indent=2, ensure_ascii=False))

    logging.info("Saved eval info to %s", eval_info_path)
    logging.info("Saved eval summary json to %s", summary_json_path)
    logging.info("Saved eval summary text to %s", summary_txt_path)
    logging.info("Saved run meta to %s", run_meta_path)
    logging.info("Saved videos under %s", videos_dir)
    logging.info("Saved numeric diff tables under %s", numeric_dir)


if __name__ == "__main__":
    main()
