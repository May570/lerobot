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
OUTPUT_DIR_DEFAULT = Path("/home/admin123/桌面/wjl/lerobot/outputs/state_oracle_vs_kalman")

STATE_DIM_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "axisangle_x",
    "axisangle_y",
    "axisangle_z",
    "gripper_left",
    "gripper_right",
]
POSITION_DIM_NAMES = STATE_DIM_NAMES[:3]
ROTATION_DIM_NAMES = STATE_DIM_NAMES[3:6]
GRIPPER_DIM_NAMES = STATE_DIM_NAMES[6:]


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
            "Compare oracle future observation.state against a two-frame constant-velocity Kalman "
            "prediction on the local libero_dyn_mini dataset."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument(
        "--deltas",
        type=str,
        default="1,2,3,4,5,6",
        help="Comma-separated future frame deltas to evaluate.",
    )
    parser.add_argument("--episodes-per-page", type=int, default=20)
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


def gripper_aperture_from_state(state_vec: np.ndarray) -> float:
    state_vec = np.asarray(state_vec, dtype=np.float64).reshape(8)
    # The two finger joints are mirror-symmetric: left ~= -right.
    # Use a single scalar aperture instead of treating them as a generic 2D vector.
    return float(0.5 * (state_vec[6] - state_vec[7]))


def load_dataset_table(dataset_root: Path, num_episodes: int) -> pd.DataFrame:
    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}.")

    tables = []
    columns = ["episode_index", "frame_index", "timestamp", "observation.state"]
    for path in parquet_files:
        table = pq.read_table(path, columns=columns)
        if table.num_rows == 0:
            continue
        df = table.to_pandas()
        df = df[df["episode_index"] < int(num_episodes)]
        if len(df) == 0:
            continue
        tables.append(df)

    if not tables:
        raise RuntimeError("No dataset rows loaded for the requested range.")

    df = pd.concat(tables, ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"], kind="stable").reset_index(drop=True)
    df["episode_index"] = df["episode_index"].astype(np.int64)
    df["frame_index"] = df["frame_index"].astype(np.int64)
    df["timestamp"] = df["timestamp"].astype(np.float64)
    df["observation.state"] = df["observation.state"].apply(
        lambda x: np.asarray(x, dtype=np.float64).reshape(8)
    )
    return df


def kalman_predict_from_two_frames(
    prev_obs: np.ndarray,
    curr_obs: np.ndarray,
    dt_hist: float,
    dt_future: float,
    cfg: KalmanConfig,
) -> np.ndarray:
    prev_obs = np.asarray(prev_obs, dtype=np.float64).reshape(-1)
    curr_obs = np.asarray(curr_obs, dtype=np.float64).reshape(-1)
    if prev_obs.shape != (8,) or curr_obs.shape != (8,):
        raise ValueError("Expected 8D state observations.")

    dt_hist = float(max(dt_hist, 1e-12))
    dt_future = float(max(dt_future, 0.0))
    dim = prev_obs.shape[0]

    x = np.zeros((dim, 2), dtype=np.float64)
    x[:, 0] = prev_obs

    p = np.zeros((dim, 2, 2), dtype=np.float64)
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
    return x[:, 0] + dt_future * x[:, 1]


def compute_episode_records(group: pd.DataFrame, deltas: list[int], cfg: KalmanConfig) -> list[dict[str, Any]]:
    rows = group.sort_values("frame_index", kind="stable").reset_index(drop=True)
    episode_index = int(rows.iloc[0]["episode_index"])
    states = np.stack(rows["observation.state"].to_list(), axis=0)
    timestamps = rows["timestamp"].to_numpy(dtype=np.float64)
    frame_indices = rows["frame_index"].to_numpy(dtype=np.int64)

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

            oracle_future = states[future_idx]
            kalman_future = kalman_predict_from_two_frames(
                prev_obs=states[prev_idx],
                curr_obs=states[curr_idx],
                dt_hist=dt_hist,
                dt_future=dt_future,
                cfg=cfg,
            )
            diff = kalman_future - oracle_future
            sq = diff * diff
            oracle_aperture = gripper_aperture_from_state(oracle_future)
            kalman_aperture = gripper_aperture_from_state(kalman_future)
            aperture_err = float(kalman_aperture - oracle_aperture)

            row = {
                "episode_index": episode_index,
                "frame_index": int(frame_indices[curr_idx]),
                "future_frame_index": int(frame_indices[future_idx]),
                "delta": int(delta),
                "dt_hist": float(dt_hist),
                "dt_future": float(dt_future),
                "vector_l2_error": float(np.linalg.norm(diff)),
                "vector_mse_error": float(np.mean(sq)),
                "position_l2_error": float(np.linalg.norm(diff[:3])),
                "rotation_l2_error": float(np.linalg.norm(diff[3:6])),
                "gripper_pair_l2_error": float(np.linalg.norm(diff[6:])),
                "oracle_gripper_aperture": oracle_aperture,
                "kalman_gripper_aperture": kalman_aperture,
                "gripper_aperture_error": aperture_err,
                "gripper_aperture_abs_error": float(abs(aperture_err)),
                "gripper_aperture_sq_error": float(aperture_err * aperture_err),
            }

            for dim_idx, dim_name in enumerate(STATE_DIM_NAMES):
                row[f"oracle_{dim_name}"] = float(oracle_future[dim_idx])
                row[f"kalman_{dim_name}"] = float(kalman_future[dim_idx])
                row[f"err_{dim_name}"] = float(diff[dim_idx])
                row[f"abs_err_{dim_name}"] = float(abs(diff[dim_idx]))
                row[f"sq_err_{dim_name}"] = float(sq[dim_idx])

            records.append(row)
    return records


def agg_series(values: pd.Series) -> dict[str, float]:
    if len(values) == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "p95": math.nan,
            "max": math.nan,
        }
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def summarize_records(records_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "overall": {},
        "by_delta": {},
        "by_dimension": {},
        "by_delta_and_dimension": {},
    }

    metric_keys = [
        "vector_l2_error",
        "vector_mse_error",
        "position_l2_error",
        "rotation_l2_error",
        "gripper_pair_l2_error",
        "gripper_aperture_abs_error",
    ]
    for metric in metric_keys:
        summary["overall"][metric] = agg_series(records_df[metric])

    for delta, sub_df in records_df.groupby("delta", sort=True):
        entry: dict[str, Any] = {}
        for metric in metric_keys:
            entry[metric] = agg_series(sub_df[metric])
        summary["by_delta"][str(int(delta))] = entry

    for dim_name in STATE_DIM_NAMES:
        summary["by_dimension"][dim_name] = {
            "abs_error": agg_series(records_df[f"abs_err_{dim_name}"]),
            "sq_error": agg_series(records_df[f"sq_err_{dim_name}"]),
        }

    for delta, sub_df in records_df.groupby("delta", sort=True):
        dim_entry: dict[str, Any] = {}
        for dim_name in STATE_DIM_NAMES:
            dim_entry[dim_name] = {
                "abs_error": agg_series(sub_df[f"abs_err_{dim_name}"]),
                "sq_error": agg_series(sub_df[f"sq_err_{dim_name}"]),
            }
        summary["by_delta_and_dimension"][str(int(delta))] = dim_entry

    return summary


def build_episode_summary_table(records_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        records_df.groupby(["episode_index", "delta"], sort=True)[
            [
                "vector_l2_error",
                "vector_mse_error",
                "position_l2_error",
                "rotation_l2_error",
                "gripper_pair_l2_error",
                "gripper_aperture_abs_error",
            ]
        ]
        .agg(["mean", "median", "max", "count"])
        .reset_index()
    )
    agg.columns = [
        "episode_index",
        "delta",
        "vector_l2_mean",
        "vector_l2_median",
        "vector_l2_max",
        "vector_l2_count",
        "vector_mse_mean",
        "vector_mse_median",
        "vector_mse_max",
        "vector_mse_count",
        "position_l2_mean",
        "position_l2_median",
        "position_l2_max",
        "position_l2_count",
        "rotation_l2_mean",
        "rotation_l2_median",
        "rotation_l2_max",
        "rotation_l2_count",
        "gripper_pair_l2_mean",
        "gripper_pair_l2_median",
        "gripper_pair_l2_max",
        "gripper_pair_l2_count",
        "gripper_aperture_abs_mean",
        "gripper_aperture_abs_median",
        "gripper_aperture_abs_max",
        "gripper_aperture_abs_count",
    ]
    return agg


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_episode_pages(records_df: pd.DataFrame, output_dir: Path, deltas: list[int], episodes_per_page: int) -> None:
    ensure_dir(output_dir)
    episode_ids = sorted(int(x) for x in records_df["episode_index"].unique().tolist())
    y_max = float(records_df["vector_l2_error"].max()) if not records_df.empty else 1.0

    ncols = 4
    nrows = max(1, math.ceil(episodes_per_page / ncols))
    page_count = math.ceil(len(episode_ids) / episodes_per_page)

    for page_idx in range(page_count):
        start = page_idx * episodes_per_page
        page_episode_ids = episode_ids[start : start + episodes_per_page]
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 3.6 * nrows), sharey=True)
        axes_arr = np.atleast_1d(axes).reshape(-1)

        for ax_idx, ax in enumerate(axes_arr):
            if ax_idx >= len(page_episode_ids):
                ax.axis("off")
                continue
            ep = page_episode_ids[ax_idx]
            ep_df = records_df[records_df["episode_index"] == ep]
            for delta in deltas:
                sub = ep_df[ep_df["delta"] == int(delta)].sort_values("frame_index", kind="stable")
                if sub.empty:
                    continue
                ax.plot(
                    sub["frame_index"].to_numpy(),
                    sub["vector_l2_error"].to_numpy(),
                    linewidth=1.0,
                    alpha=0.9,
                    label=f"d={delta}",
                )
            ax.set_title(f"ep {ep}", fontsize=10)
            ax.set_ylim(0.0, max(y_max * 1.05, 1e-6))
            ax.grid(alpha=0.25, linewidth=0.5)
            if ax_idx % ncols == 0:
                ax.set_ylabel("vector L2")
            if ax_idx >= (nrows - 1) * ncols:
                ax.set_xlabel("frame")

        handles, labels = axes_arr[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(deltas), 6), frameon=False)
        fig.suptitle("Per-episode Kalman vs oracle state error", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_dir / f"episode_vector_l2_page_{page_idx + 1:02d}.png", dpi=180)
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
                ep_df["vector_l2_error"].to_numpy(),
                linewidth=0.8,
                alpha=0.28,
            )
        ax.set_title(f"All episodes overlay: state vector L2, delta={delta}")
        ax.set_xlabel("frame")
        ax.set_ylabel("vector L2")
        ax.grid(alpha=0.25, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(output_dir / f"overlay_delta_{delta:02d}.png", dpi=180)
        plt.close(fig)


def plot_component_summaries(records_df: pd.DataFrame, output_dir: Path, deltas: list[int]) -> None:
    ensure_dir(output_dir)

    # Aggregate by delta for semantic groups and the full vector.
    rows = []
    for delta in deltas:
        sub = records_df[records_df["delta"] == int(delta)]
        rows.append(
            {
                "delta": int(delta),
                "vector_mean": float(sub["vector_l2_error"].mean()),
                "vector_p95": float(sub["vector_l2_error"].quantile(0.95)),
                "vector_max": float(sub["vector_l2_error"].max()),
                "position_l2_mean": float(sub["position_l2_error"].mean()),
                "position_l2_p95": float(sub["position_l2_error"].quantile(0.95)),
                "position_l2_max": float(sub["position_l2_error"].max()),
                "rotation_l2_mean": float(sub["rotation_l2_error"].mean()),
                "rotation_l2_p95": float(sub["rotation_l2_error"].quantile(0.95)),
                "rotation_l2_max": float(sub["rotation_l2_error"].max()),
                "gripper_aperture_abs_mean": float(sub["gripper_aperture_abs_error"].mean()),
                "gripper_aperture_abs_p95": float(sub["gripper_aperture_abs_error"].quantile(0.95)),
                "gripper_aperture_abs_max": float(sub["gripper_aperture_abs_error"].max()),
            }
        )
    comp_df = pd.DataFrame(rows)

    def plot_band(ax: Any, x: pd.Series, mean: pd.Series, p95: pd.Series, vmax: pd.Series, title: str, ylabel: str) -> None:
        ax.plot(x, mean, marker="o", linewidth=2.0, label="mean")
        ax.plot(x, p95, marker="o", linewidth=1.8, label="p95")
        ax.plot(x, vmax, marker="o", linewidth=1.4, linestyle="--", label="max")
        ax.fill_between(x, mean, p95, alpha=0.18)
        ax.set_title(title)
        ax.set_xlabel("delta")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linewidth=0.5)

    # Decision-oriented single metric plots.
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_band(
        ax,
        comp_df["delta"],
        comp_df["vector_mean"],
        comp_df["vector_p95"],
        comp_df["vector_max"],
        title="Whole state error vs delta",
        ylabel="state vector L2",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "overall_vector_trust_by_delta.png", dpi=180)
    plt.close(fig)

    # 2x2 dashboard: one glance should show what grows with delta.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plot_band(
        axes[0, 0],
        comp_df["delta"],
        comp_df["vector_mean"],
        comp_df["vector_p95"],
        comp_df["vector_max"],
        title="Whole state",
        ylabel="vector L2",
    )
    plot_band(
        axes[0, 1],
        comp_df["delta"],
        comp_df["position_l2_mean"],
        comp_df["position_l2_p95"],
        comp_df["position_l2_max"],
        title="EEF position",
        ylabel="L2 (meters)",
    )
    plot_band(
        axes[1, 0],
        comp_df["delta"],
        comp_df["rotation_l2_mean"],
        comp_df["rotation_l2_p95"],
        comp_df["rotation_l2_max"],
        title="EEF rotation",
        ylabel="L2 (axis-angle rad)",
    )
    plot_band(
        axes[1, 1],
        comp_df["delta"],
        comp_df["gripper_aperture_abs_mean"],
        comp_df["gripper_aperture_abs_p95"],
        comp_df["gripper_aperture_abs_max"],
        title="Gripper aperture",
        ylabel="abs error",
    )
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("Kalman future-state trust dashboard", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "state_trust_dashboard.png", dpi=180)
    plt.close(fig)

    # Keep the old compact semantic comparison line plot as a quick overview.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(comp_df["delta"], comp_df["position_l2_mean"], marker="o", label="position_mean")
    ax.plot(comp_df["delta"], comp_df["rotation_l2_mean"], marker="o", label="rotation_mean")
    ax.plot(comp_df["delta"], comp_df["gripper_aperture_abs_mean"], marker="o", label="gripper_aperture_mean")
    ax.set_xlabel("delta")
    ax.set_ylabel("mean error")
    ax.set_title("Mean component-group error by delta")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "component_group_means_by_delta.png", dpi=180)
    plt.close(fig)

    # Per-dimension abs error bars.
    dim_rows = []
    for dim_name in STATE_DIM_NAMES:
        dim_rows.append(
            {
                "dim": dim_name,
                "mean_abs_error": float(records_df[f"abs_err_{dim_name}"].mean()),
                "mean_sq_error": float(records_df[f"sq_err_{dim_name}"].mean()),
            }
        )
    dim_df = pd.DataFrame(dim_rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(dim_df["dim"], dim_df["mean_abs_error"])
    ax.set_ylabel("mean abs error")
    ax.set_title("Per-dimension mean absolute error")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "dimension_mean_abs_error.png", dpi=180)
    plt.close(fig)

    # Semantic dimension dashboards with mean and p95.
    def plot_dim_dashboard(dim_names: list[str], title: str, ylabel: str, out_name: str) -> None:
        fig, axes = plt.subplots(1, len(dim_names), figsize=(4.5 * len(dim_names), 4.8), sharey=True)
        axes_arr = np.atleast_1d(axes).reshape(-1)
        for ax, dim_name in zip(axes_arr, dim_names):
            rows = []
            for delta in deltas:
                sub = records_df[records_df["delta"] == int(delta)]
                vals = sub[f"abs_err_{dim_name}"]
                rows.append(
                    {
                        "delta": int(delta),
                        "mean": float(vals.mean()),
                        "p95": float(vals.quantile(0.95)),
                        "max": float(vals.max()),
                    }
                )
            df = pd.DataFrame(rows)
            plot_band(ax, df["delta"], df["mean"], df["p95"], df["max"], dim_name, ylabel)
        axes_arr[0].legend(frameon=False, loc="upper left")
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(output_dir / out_name, dpi=180)
        plt.close(fig)

    plot_dim_dashboard(
        POSITION_DIM_NAMES,
        title="Position dimension errors by delta",
        ylabel="abs error (meters)",
        out_name="position_dimensions_by_delta.png",
    )
    plot_dim_dashboard(
        ROTATION_DIM_NAMES,
        title="Rotation dimension errors by delta",
        ylabel="abs error (axis-angle rad)",
        out_name="rotation_dimensions_by_delta.png",
    )

    # Heatmaps for quick "which dimension dominates" scanning.
    def plot_heatmap(value_fn: str, title: str, out_name: str) -> None:
        heat = np.zeros((len(STATE_DIM_NAMES), len(deltas)), dtype=np.float64)
        for i, dim_name in enumerate(STATE_DIM_NAMES):
            for j, delta in enumerate(deltas):
                sub = records_df[records_df["delta"] == int(delta)]
                vals = sub[f"abs_err_{dim_name}"]
                if value_fn == "mean":
                    heat[i, j] = float(vals.mean())
                elif value_fn == "p95":
                    heat[i, j] = float(vals.quantile(0.95))
                else:
                    raise ValueError(value_fn)

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        im = ax.imshow(heat, aspect="auto")
        ax.set_xticks(np.arange(len(deltas)))
        ax.set_xticklabels([str(d) for d in deltas])
        ax.set_yticks(np.arange(len(STATE_DIM_NAMES)))
        ax.set_yticklabels(STATE_DIM_NAMES)
        ax.set_xlabel("delta")
        ax.set_title(title)
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                ax.text(j, i, f"{heat[i, j]:.4f}", ha="center", va="center", fontsize=8, color="white")
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("abs error", rotation=270, labelpad=14)
        fig.tight_layout()
        fig.savefig(output_dir / out_name, dpi=180)
        plt.close(fig)

    plot_heatmap("mean", "Per-dimension mean absolute error", "dimension_mean_abs_error_heatmap.png")
    plot_heatmap("p95", "Per-dimension p95 absolute error", "dimension_p95_abs_error_heatmap.png")

    # A compact "trust" table plot: mean / p95 / max for each semantic component.
    rows = []
    for delta in deltas:
        sub = records_df[records_df["delta"] == int(delta)]
        rows.append(
            {
                "delta": int(delta),
                "position_mean": float(sub["position_l2_error"].mean()),
                "position_p95": float(sub["position_l2_error"].quantile(0.95)),
                "position_max": float(sub["position_l2_error"].max()),
                "rotation_mean": float(sub["rotation_l2_error"].mean()),
                "rotation_p95": float(sub["rotation_l2_error"].quantile(0.95)),
                "rotation_max": float(sub["rotation_l2_error"].max()),
                "gripper_mean": float(sub["gripper_aperture_abs_error"].mean()),
                "gripper_p95": float(sub["gripper_aperture_abs_error"].quantile(0.95)),
                "gripper_max": float(sub["gripper_aperture_abs_error"].max()),
                "vector_mean": float(sub["vector_l2_error"].mean()),
                "vector_p95": float(sub["vector_l2_error"].quantile(0.95)),
                "vector_max": float(sub["vector_l2_error"].max()),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir.parent / "delta_component_trust_summary.csv", index=False)


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

    dataset_df = load_dataset_table(args.dataset_root, args.num_episodes)

    all_records: list[dict[str, Any]] = []
    for _, group in dataset_df.groupby("episode_index", sort=True):
        all_records.extend(compute_episode_records(group=group, deltas=deltas, cfg=cfg))

    if not all_records:
        raise RuntimeError("No comparison records were produced.")

    records_df = pd.DataFrame(all_records).sort_values(
        ["episode_index", "delta", "frame_index"], kind="stable"
    ).reset_index(drop=True)

    frame_csv = args.output_dir / "frame_level_errors.csv"
    records_df.to_csv(frame_csv, index=False)

    episode_summary_df = build_episode_summary_table(records_df)
    episode_summary_csv = args.output_dir / "episode_delta_summary.csv"
    episode_summary_df.to_csv(episode_summary_csv, index=False)

    summary = summarize_records(records_df)
    metadata = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "num_episodes": int(args.num_episodes),
        "deltas": deltas,
        "state_dim_names": STATE_DIM_NAMES,
        "state_groups": {
            "position": POSITION_DIM_NAMES,
            "rotation": ROTATION_DIM_NAMES,
            "gripper": GRIPPER_DIM_NAMES,
        },
        "derived_scalars": {
            "gripper_aperture": "0.5 * (gripper_left - gripper_right)",
        },
        "kalman_config": {
            "meas_noise_std": cfg.meas_noise_std,
            "accel_noise_std": cfg.accel_noise_std,
            "init_pos_std": cfg.init_pos_std,
            "init_vel_std": cfg.init_vel_std,
            "dt_fallback": cfg.dt_fallback,
        },
        "files": {
            "frame_level_errors_csv": str(frame_csv),
            "episode_delta_summary_csv": str(episode_summary_csv),
            "delta_component_trust_summary_csv": str(args.output_dir / "delta_component_trust_summary.csv"),
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
    plot_component_summaries(records_df=records_df, output_dir=plots_dir, deltas=deltas)

    print(json.dumps({"ok": True, "output_dir": str(args.output_dir), "rows": int(len(records_df))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
