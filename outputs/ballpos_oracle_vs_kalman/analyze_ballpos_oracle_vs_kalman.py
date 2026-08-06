#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DATASET_ROOT_DEFAULT = Path(
    "/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/datasets/libero_dyn_mini_balanced500_scripted_v2"
)
SIDECAR_DIR_DEFAULT = DATASET_ROOT_DEFAULT / "sidecar_has_object_ep0_199_clean_20260421_194112"
OUTPUT_DIR_DEFAULT = Path("/home/admin123/桌面/wjl/lerobot/outputs/ballpos_oracle_vs_kalman")


@dataclass(frozen=True)
class KalmanConfig:
    meas_noise_std: float = 0.01
    accel_noise_std: float = 0.4
    init_pos_std: float = 0.05
    init_vel_std: float = 0.5
    dt_fallback: float = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare oracle future ball_pos against a two-frame constant-velocity Kalman prediction "
            "on the local libero_dyn_mini dataset."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    parser.add_argument("--sidecar-dir", type=Path, default=SIDECAR_DIR_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument(
        "--deltas",
        type=str,
        default="1,2,3,4,5,6",
        help="Comma-separated future frame deltas to evaluate.",
    )
    parser.add_argument(
        "--episodes-per-page",
        type=int,
        default=20,
        help="How many episode subplots to place on each page.",
    )
    parser.add_argument("--meas-noise-std", type=float, default=0.01)
    parser.add_argument("--accel-noise-std", type=float, default=0.4)
    parser.add_argument("--init-pos-std", type=float, default=0.05)
    parser.add_argument("--init-vel-std", type=float, default=0.5)
    parser.add_argument("--dt-fallback", type=float, default=0.1)
    return parser.parse_args()


def parse_deltas(raw: str) -> list[int]:
    deltas = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"All deltas must be > 0, got {value}.")
        deltas.append(value)
    if not deltas:
        raise ValueError("At least one delta is required.")
    return sorted(set(deltas))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_dataset_table(dataset_root: Path, max_index_exclusive: int | None = None) -> pd.DataFrame:
    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}.")

    tables = []
    columns = ["episode_index", "frame_index", "index", "timestamp", "observation.ball_pos"]
    for path in parquet_files:
        table = pq.read_table(path, columns=columns)
        if max_index_exclusive is not None:
            idx = table.column("index").to_numpy()
            keep = idx < max_index_exclusive
            if not np.any(keep):
                continue
            table = table.filter(np.asarray(keep))
        if table.num_rows > 0:
            tables.append(table)

    if not tables:
        raise RuntimeError("No dataset rows loaded for the requested range.")

    df = pd.concat([t.to_pandas() for t in tables], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"], kind="stable").reset_index(drop=True)
    df["episode_index"] = df["episode_index"].astype(np.int64)
    df["frame_index"] = df["frame_index"].astype(np.int64)
    df["index"] = df["index"].astype(np.int64)
    df["timestamp"] = df["timestamp"].astype(np.float64)
    df["observation.ball_pos"] = df["observation.ball_pos"].apply(
        lambda x: np.asarray(x, dtype=np.float64).reshape(3)
    )
    return df


def load_sidecar_labels(sidecar_dir: Path, num_episodes: int) -> pd.DataFrame:
    labels_path = sidecar_dir / "labels.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels parquet: {labels_path}")

    table = pq.read_table(
        labels_path,
        columns=["episode_index", "frame_index", "has_object", "grasp_event", "ever_grasped"],
    )
    df = table.to_pandas()
    df = df[df["episode_index"] < int(num_episodes)].copy()
    df["episode_index"] = df["episode_index"].astype(np.int64)
    df["frame_index"] = df["frame_index"].astype(np.int64)
    for col in ["has_object", "grasp_event", "ever_grasped"]:
        df[col] = df[col].astype(bool)
    return df.sort_values(["episode_index", "frame_index"], kind="stable").reset_index(drop=True)


def load_episode_summaries(sidecar_dir: Path, num_episodes: int) -> list[dict[str, Any]]:
    path = sidecar_dir / "episode_summaries.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["episode_index"]) < int(num_episodes):
                rows.append(row)
    return rows


def merge_dataset_and_labels(dataset_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    merged = dataset_df.merge(
        labels_df,
        on=["episode_index", "frame_index"],
        how="left",
        validate="one_to_one",
    )
    for col in ["has_object", "grasp_event", "ever_grasped"]:
        merged[col] = merged[col].fillna(False).astype(bool)
    return merged


def infer_grasp_frame(group: pd.DataFrame) -> int | None:
    has_object = group["has_object"].to_numpy(dtype=bool)
    indices = np.flatnonzero(has_object)
    if indices.size == 0:
        return None
    return int(group.iloc[int(indices[0])]["frame_index"])


def kalman_predict_from_two_frames(
    prev_obs: np.ndarray,
    curr_obs: np.ndarray,
    dt_hist: float,
    dt_future: float,
    cfg: KalmanConfig,
) -> np.ndarray:
    prev_obs = np.asarray(prev_obs, dtype=np.float64).reshape(-1)
    curr_obs = np.asarray(curr_obs, dtype=np.float64).reshape(-1)
    if prev_obs.shape != (3,) or curr_obs.shape != (3,):
        raise ValueError("Expected 3D ball positions.")

    dt_hist = float(max(dt_hist, 1e-12))
    dt_future = float(max(dt_future, 0.0))

    x = np.zeros((3, 2), dtype=np.float64)
    x[:, 0] = prev_obs

    p = np.zeros((3, 2, 2), dtype=np.float64)
    p[:, 0, 0] = float(cfg.init_pos_std**2)
    p[:, 1, 1] = float(cfg.init_vel_std**2)

    sigma2 = float(cfg.accel_noise_std**2)
    meas_var = float(cfg.meas_noise_std**2)

    def update(measurement: np.ndarray) -> None:
        residual = measurement - x[:, 0]
        s_innov = np.maximum(p[:, 0, 0] + meas_var, 1e-12)
        k0 = p[:, 0, 0] / s_innov
        k1 = p[:, 1, 0] / s_innov
        x[:, 0] = x[:, 0] + k0 * residual
        x[:, 1] = x[:, 1] + k1 * residual

        p00 = p[:, 0, 0].copy()
        p01 = p[:, 0, 1].copy()
        p10 = p[:, 1, 0].copy()
        p11 = p[:, 1, 1].copy()
        p[:, 0, 0] = (1.0 - k0) * p00
        p[:, 0, 1] = (1.0 - k0) * p01
        p[:, 1, 0] = p10 - k1 * p00
        p[:, 1, 1] = p11 - k1 * p01

    def predict_step(step_dt: float) -> None:
        dt2 = step_dt * step_dt
        dt3 = dt2 * step_dt
        dt4 = dt2 * dt2

        x[:, 0] = x[:, 0] + step_dt * x[:, 1]

        p00 = p[:, 0, 0].copy()
        p01 = p[:, 0, 1].copy()
        p10 = p[:, 1, 0].copy()
        p11 = p[:, 1, 1].copy()
        p[:, 0, 0] = p00 + step_dt * (p10 + p01) + dt2 * p11 + (dt4 / 4.0) * sigma2
        p[:, 0, 1] = p01 + step_dt * p11 + (dt3 / 2.0) * sigma2
        p[:, 1, 0] = p10 + step_dt * p11 + (dt3 / 2.0) * sigma2
        p[:, 1, 1] = p11 + dt2 * sigma2

    update(prev_obs)
    predict_step(dt_hist)
    update(curr_obs)
    predicted_future = x[:, 0] + dt_future * x[:, 1]
    return predicted_future


def compute_episode_records(group: pd.DataFrame, deltas: list[int], cfg: KalmanConfig) -> list[dict[str, Any]]:
    rows = group.sort_values("frame_index", kind="stable").reset_index(drop=True)
    episode_index = int(rows.iloc[0]["episode_index"])
    grasp_frame = infer_grasp_frame(rows)
    ball_pos = np.stack(rows["observation.ball_pos"].to_list(), axis=0)
    timestamps = rows["timestamp"].to_numpy(dtype=np.float64)
    frame_indices = rows["frame_index"].to_numpy(dtype=np.int64)
    has_object = rows["has_object"].to_numpy(dtype=bool)

    records: list[dict[str, Any]] = []
    n = len(rows)
    for curr_idx in range(1, n):
        prev_idx = curr_idx - 1
        dt_hist = float(timestamps[curr_idx] - timestamps[prev_idx])
        if not np.isfinite(dt_hist) or dt_hist <= 0:
            dt_hist = float(cfg.dt_fallback)

        for delta in deltas:
            future_idx = curr_idx + delta
            if future_idx >= n:
                continue
            dt_future = float(timestamps[future_idx] - timestamps[curr_idx])
            if not np.isfinite(dt_future) or dt_future < 0:
                dt_future = float(cfg.dt_fallback) * float(delta)

            oracle_future = ball_pos[future_idx]
            kalman_future = kalman_predict_from_two_frames(
                prev_obs=ball_pos[prev_idx],
                curr_obs=ball_pos[curr_idx],
                dt_hist=dt_hist,
                dt_future=dt_future,
                cfg=cfg,
            )

            diff = kalman_future - oracle_future
            sq = diff * diff
            phase = "post_grasp" if has_object[curr_idx] else "pre_grasp"

            records.append(
                {
                    "episode_index": episode_index,
                    "frame_index": int(frame_indices[curr_idx]),
                    "future_frame_index": int(frame_indices[future_idx]),
                    "delta": int(delta),
                    "dt_hist": float(dt_hist),
                    "dt_future": float(dt_future),
                    "grasp_frame": None if grasp_frame is None else int(grasp_frame),
                    "phase": phase,
                    "has_object_current": bool(has_object[curr_idx]),
                    "oracle_x": float(oracle_future[0]),
                    "oracle_y": float(oracle_future[1]),
                    "oracle_z": float(oracle_future[2]),
                    "kalman_x": float(kalman_future[0]),
                    "kalman_y": float(kalman_future[1]),
                    "kalman_z": float(kalman_future[2]),
                    "err_x": float(diff[0]),
                    "err_y": float(diff[1]),
                    "err_z": float(diff[2]),
                    "abs_err_x": float(abs(diff[0])),
                    "abs_err_y": float(abs(diff[1])),
                    "abs_err_z": float(abs(diff[2])),
                    "sq_err_x": float(sq[0]),
                    "sq_err_y": float(sq[1]),
                    "sq_err_z": float(sq[2]),
                    "l2_error": float(np.linalg.norm(diff)),
                    "mse_error": float(np.mean(sq)),
                }
            )

    return records


def summarize_records(records_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    def agg_frame(df: pd.DataFrame) -> dict[str, float]:
        if df.empty:
            return {
                "count": 0,
                "l2_mean": math.nan,
                "l2_median": math.nan,
                "l2_p95": math.nan,
                "l2_max": math.nan,
                "mse_mean": math.nan,
                "mse_median": math.nan,
                "mse_p95": math.nan,
                "mse_max": math.nan,
            }
        return {
            "count": int(len(df)),
            "l2_mean": float(df["l2_error"].mean()),
            "l2_median": float(df["l2_error"].median()),
            "l2_p95": float(df["l2_error"].quantile(0.95)),
            "l2_max": float(df["l2_error"].max()),
            "mse_mean": float(df["mse_error"].mean()),
            "mse_median": float(df["mse_error"].median()),
            "mse_p95": float(df["mse_error"].quantile(0.95)),
            "mse_max": float(df["mse_error"].max()),
        }

    summary["overall"] = agg_frame(records_df)
    summary["by_delta"] = {}
    summary["by_phase"] = {}
    summary["by_delta_and_phase"] = {}

    for delta, sub_df in records_df.groupby("delta", sort=True):
        summary["by_delta"][str(int(delta))] = agg_frame(sub_df)
    for phase, sub_df in records_df.groupby("phase", sort=True):
        summary["by_phase"][str(phase)] = agg_frame(sub_df)
    for (delta, phase), sub_df in records_df.groupby(["delta", "phase"], sort=True):
        summary["by_delta_and_phase"].setdefault(str(int(delta)), {})[str(phase)] = agg_frame(sub_df)

    episode_delta_phase = (
        records_df.groupby(["episode_index", "delta", "phase"], sort=True)[["l2_error", "mse_error"]]
        .mean()
        .reset_index()
    )
    summary["episode_mean_counts"] = {
        "rows": int(len(episode_delta_phase)),
        "episodes": int(records_df["episode_index"].nunique()),
        "deltas": sorted(int(x) for x in records_df["delta"].unique().tolist()),
    }

    return summary


def build_episode_summary_table(records_df: pd.DataFrame, episode_summaries: list[dict[str, Any]]) -> pd.DataFrame:
    agg = (
        records_df.groupby(["episode_index", "delta", "phase"], sort=True)[["l2_error", "mse_error"]]
        .agg(["mean", "median", "max", "count"])
        .reset_index()
    )
    agg.columns = [
        "episode_index",
        "delta",
        "phase",
        "l2_mean",
        "l2_median",
        "l2_max",
        "l2_count",
        "mse_mean",
        "mse_median",
        "mse_max",
        "mse_count",
    ]
    if episode_summaries:
        summary_df = pd.DataFrame(episode_summaries)
        keep_cols = [
            "episode_index",
            "frames",
            "has_object_frames",
            "grasp_event_frames",
            "ever_grasped",
            "first_replay_done_step",
            "direction_deg",
            "tilt_deg",
            "speed",
        ]
        available = [c for c in keep_cols if c in summary_df.columns]
        agg = agg.merge(summary_df[available], on="episode_index", how="left")
    return agg


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_episode_pages(records_df: pd.DataFrame, output_dir: Path, deltas: list[int], episodes_per_page: int) -> None:
    ensure_dir(output_dir)
    episode_ids = sorted(int(x) for x in records_df["episode_index"].unique().tolist())
    l2_max = float(records_df["l2_error"].max()) if not records_df.empty else 1.0

    ncols = 4
    nrows = max(1, math.ceil(episodes_per_page / ncols))
    page_count = math.ceil(len(episode_ids) / episodes_per_page)

    for page_idx in range(page_count):
        start = page_idx * episodes_per_page
        page_episode_ids = episode_ids[start : start + episodes_per_page]
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 3.6 * nrows), sharex=False, sharey=True)
        axes_arr = np.atleast_1d(axes).reshape(-1)

        for ax_idx, ax in enumerate(axes_arr):
            if ax_idx >= len(page_episode_ids):
                ax.axis("off")
                continue

            ep = page_episode_ids[ax_idx]
            ep_df = records_df[records_df["episode_index"] == ep]
            grasp_rows = ep_df["grasp_frame"].dropna().unique().tolist()
            grasp_frame = int(grasp_rows[0]) if grasp_rows else None

            for delta in deltas:
                sub = ep_df[ep_df["delta"] == int(delta)].sort_values("frame_index", kind="stable")
                if sub.empty:
                    continue
                ax.plot(
                    sub["frame_index"].to_numpy(),
                    sub["l2_error"].to_numpy(),
                    linewidth=1.0,
                    alpha=0.9,
                    label=f"d={delta}",
                )

            if grasp_frame is not None:
                ax.axvline(grasp_frame, color="black", linestyle="--", linewidth=0.9, alpha=0.8)
            ax.set_title(f"ep {ep}", fontsize=10)
            ax.set_ylim(0.0, max(l2_max * 1.05, 1e-6))
            ax.grid(alpha=0.25, linewidth=0.5)
            if ax_idx % ncols == 0:
                ax.set_ylabel("L2 error")
            if ax_idx >= (nrows - 1) * ncols:
                ax.set_xlabel("frame")

        handles, labels = axes_arr[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(deltas), 6), frameon=False)
        fig.suptitle("Per-episode Kalman vs oracle future ball_pos error", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_dir / f"episode_l2_page_{page_idx + 1:02d}.png", dpi=180)
        plt.close(fig)


def plot_delta_overlays(records_df: pd.DataFrame, output_dir: Path, deltas: list[int]) -> None:
    ensure_dir(output_dir)
    for delta in deltas:
        sub = records_df[records_df["delta"] == int(delta)]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 8))
        for episode_index, ep_df in sub.groupby("episode_index", sort=True):
            ep_df = ep_df.sort_values("frame_index", kind="stable")
            ax.plot(
                ep_df["frame_index"].to_numpy(),
                ep_df["l2_error"].to_numpy(),
                linewidth=0.8,
                alpha=0.28,
            )
        ax.set_title(f"All episodes overlay: delta={delta}")
        ax.set_xlabel("frame")
        ax.set_ylabel("L2 error")
        ax.grid(alpha=0.25, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(output_dir / f"overlay_delta_{delta:02d}.png", dpi=180)
        plt.close(fig)


def plot_phase_boxplots(records_df: pd.DataFrame, output_dir: Path, deltas: list[int]) -> None:
    ensure_dir(output_dir)
    phases = ["pre_grasp", "post_grasp"]
    fig, axes = plt.subplots(1, len(deltas), figsize=(4.2 * len(deltas), 5.2), sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for ax, delta in zip(axes_arr, deltas):
        sub = records_df[records_df["delta"] == int(delta)]
        data = []
        labels = []
        for phase in phases:
            phase_vals = sub.loc[sub["phase"] == phase, "l2_error"].to_numpy(dtype=np.float64)
            if phase_vals.size > 0:
                data.append(phase_vals)
                labels.append(phase)
        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(f"delta={delta}")
        ax.grid(alpha=0.25, linewidth=0.5)
        if ax is axes_arr[0]:
            ax.set_ylabel("L2 error")
    fig.suptitle("L2 error by delta and grasp phase", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "phase_boxplots.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    deltas = parse_deltas(args.deltas)
    ensure_dir(args.output_dir)

    cfg = KalmanConfig(
        meas_noise_std=float(args.meas_noise_std),
        accel_noise_std=float(args.accel_noise_std),
        init_pos_std=float(args.init_pos_std),
        init_vel_std=float(args.init_vel_std),
        dt_fallback=float(args.dt_fallback),
    )

    episode_summaries = load_episode_summaries(args.sidecar_dir, args.num_episodes)
    if episode_summaries:
        max_index_exclusive = 1 + max(int(row["dataset_index_end"]) for row in episode_summaries)
    else:
        max_index_exclusive = None

    dataset_df = load_dataset_table(args.dataset_root, max_index_exclusive=max_index_exclusive)
    dataset_df = dataset_df[dataset_df["episode_index"] < int(args.num_episodes)].copy()
    labels_df = load_sidecar_labels(args.sidecar_dir, args.num_episodes)
    merged_df = merge_dataset_and_labels(dataset_df, labels_df)

    all_records: list[dict[str, Any]] = []
    for _, group in merged_df.groupby("episode_index", sort=True):
        all_records.extend(compute_episode_records(group=group, deltas=deltas, cfg=cfg))

    if not all_records:
        raise RuntimeError("No comparison records were produced.")

    records_df = pd.DataFrame(all_records).sort_values(
        ["episode_index", "delta", "frame_index"], kind="stable"
    ).reset_index(drop=True)

    frame_csv = args.output_dir / "frame_level_errors.csv"
    records_df.to_csv(frame_csv, index=False)

    episode_summary_df = build_episode_summary_table(records_df, episode_summaries)
    episode_summary_csv = args.output_dir / "episode_delta_phase_summary.csv"
    episode_summary_df.to_csv(episode_summary_csv, index=False)

    summary = summarize_records(records_df)
    metadata = {
        "dataset_root": str(args.dataset_root),
        "sidecar_dir": str(args.sidecar_dir),
        "output_dir": str(args.output_dir),
        "num_episodes": int(args.num_episodes),
        "deltas": deltas,
        "kalman_config": {
            "meas_noise_std": cfg.meas_noise_std,
            "accel_noise_std": cfg.accel_noise_std,
            "init_pos_std": cfg.init_pos_std,
            "init_vel_std": cfg.init_vel_std,
            "dt_fallback": cfg.dt_fallback,
        },
        "files": {
            "frame_level_errors_csv": str(frame_csv),
            "episode_delta_phase_summary_csv": str(episode_summary_csv),
        },
    }
    save_json(args.output_dir / "summary.json", {"metadata": metadata, "summary": summary})

    plots_dir = args.output_dir / "plots"
    plot_episode_pages(
        records_df=records_df,
        output_dir=plots_dir / "episode_pages",
        deltas=deltas,
        episodes_per_page=int(args.episodes_per_page),
    )
    plot_delta_overlays(records_df=records_df, output_dir=plots_dir / "delta_overlays", deltas=deltas)
    plot_phase_boxplots(records_df=records_df, output_dir=plots_dir, deltas=deltas)

    print(json.dumps({"ok": True, "output_dir": str(args.output_dir), "rows": int(len(records_df))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
