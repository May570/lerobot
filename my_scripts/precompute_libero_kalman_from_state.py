#!/usr/bin/env python3
"""Precompute Kalman-filtered end-effector trajectory features from LIBERO state vectors.

This script does NOT modify the original dataset.
It writes episode-indexed sidecar arrays:
  output_root/episode_XXXXXX/arrays.json + *.npy
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class EpisodeMeta:
    episode_index: int
    data_chunk_index: int
    data_file_index: int


class DataFrameCache:
    def __init__(self, max_files: int):
        self.max_files = max_files
        self.cache: OrderedDict[tuple[int, int], pd.DataFrame] = OrderedDict()

    def get(self, key: tuple[int, int], path: Path, columns: list[str]) -> pd.DataFrame:
        if key in self.cache:
            df = self.cache.pop(key)
            self.cache[key] = df
            return df
        df = pd.read_parquet(path, columns=columns)
        self.cache[key] = df
        while len(self.cache) > self.max_files:
            self.cache.popitem(last=False)
        return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)

    p.add_argument("--state-pos-slice", type=str, default="0:3", help="slice in state vector used as xyz measurement")
    p.add_argument("--predict-horizon", type=float, default=0.1, help="seconds to predict forward for execution time")
    p.add_argument("--dt-fallback", type=float, default=0.1, help="used when timestamp delta is invalid")

    p.add_argument("--meas-noise-std", type=float, default=0.01, help="measurement noise std (meters)")
    p.add_argument("--accel-noise-std", type=float, default=0.4, help="process accel noise std (m/s^2)")
    p.add_argument("--init-pos-std", type=float, default=0.05)
    p.add_argument("--init-vel-std", type=float, default=0.5)

    p.add_argument("--episode-index", type=int, default=None)
    p.add_argument("--episode-start", type=int, default=None)
    p.add_argument("--episode-end", type=int, default=None)  # exclusive
    p.add_argument("--max-episodes", type=int, default=None)

    p.add_argument("--datafile-cache-size", type=int, default=4)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _parse_slice(spec: str) -> slice:
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid --state-pos-slice: {spec}")
    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if parts[1] else None
    return slice(start, end)


def _load_episode_meta(dataset_root: Path) -> list[EpisodeMeta]:
    episodes_dir = dataset_root / "meta" / "episodes"
    parquet_files = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet found under: {episodes_dir}")

    frames: list[pd.DataFrame] = []
    for path in parquet_files:
        frames.append(
            pd.read_parquet(path, columns=["episode_index", "data/chunk_index", "data/file_index"])
        )
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["episode_index"]).sort_values("episode_index")
    return [
        EpisodeMeta(
            episode_index=int(row["episode_index"]),
            data_chunk_index=int(row["data/chunk_index"]),
            data_file_index=int(row["data/file_index"]),
        )
        for _, row in df.iterrows()
    ]


def _iter_episode_indices(all_indices: list[int], args: argparse.Namespace) -> list[int]:
    if args.episode_index is not None:
        return [args.episode_index]
    indices = all_indices
    if args.episode_start is not None:
        indices = [x for x in indices if x >= args.episode_start]
    if args.episode_end is not None:
        indices = [x for x in indices if x < args.episode_end]
    if args.max_episodes is not None:
        indices = indices[: args.max_episodes]
    return indices


def _write_episode_arrays(
    dst_episode_dir: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, dict[str, object]],
    *,
    overwrite: bool,
) -> None:
    if dst_episode_dir.exists() and overwrite:
        shutil.rmtree(dst_episode_dir)
    dst_episode_dir.mkdir(parents=True, exist_ok=True)

    for key, arr in arrays.items():
        tmp_path = dst_episode_dir / f".{key}.npy.tmp"
        final_path = dst_episode_dir / f"{key}.npy"
        with tmp_path.open("wb") as f:
            np.save(f, arr, allow_pickle=False)
        tmp_path.replace(final_path)

    meta_tmp = dst_episode_dir / ".arrays.json.tmp"
    meta_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_tmp.replace(dst_episode_dir / "arrays.json")

    done_tmp = dst_episode_dir / ".done.tmp"
    done_tmp.write_text("ok\n", encoding="utf-8")
    done_tmp.replace(dst_episode_dir / ".done")


def _run_kalman(
    z_xyz: np.ndarray,
    timestamps: np.ndarray,
    *,
    dt_fallback: float,
    predict_horizon: float,
    meas_noise_std: float,
    accel_noise_std: float,
    init_pos_std: float,
    init_vel_std: float,
) -> dict[str, np.ndarray]:
    """
    z_xyz: [T, 3] measurements
    timestamps: [T] seconds
    """
    t = z_xyz.shape[0]
    x = np.zeros(6, dtype=np.float64)  # [px, py, pz, vx, vy, vz]
    x[:3] = z_xyz[0]
    p = np.diag(
        [
            init_pos_std**2,
            init_pos_std**2,
            init_pos_std**2,
            init_vel_std**2,
            init_vel_std**2,
            init_vel_std**2,
        ]
    ).astype(np.float64)

    h = np.zeros((3, 6), dtype=np.float64)
    h[:, :3] = np.eye(3, dtype=np.float64)
    r = np.eye(3, dtype=np.float64) * (meas_noise_std**2)
    i6 = np.eye(6, dtype=np.float64)

    state = np.zeros((t, 6), dtype=np.float64)
    pos = np.zeros((t, 3), dtype=np.float64)
    vel = np.zeros((t, 3), dtype=np.float64)
    pred_exec = np.zeros((t, 3), dtype=np.float64)
    valid = np.ones((t,), dtype=np.bool_)

    def make_f_q(dt: float) -> tuple[np.ndarray, np.ndarray]:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        f = np.eye(6, dtype=np.float64)
        f[:3, 3:] = np.eye(3, dtype=np.float64) * dt
        q1 = np.array([[dt4 / 4.0, dt3 / 2.0], [dt3 / 2.0, dt2]], dtype=np.float64) * (accel_noise_std**2)
        q = np.zeros((6, 6), dtype=np.float64)
        for axis in range(3):
            q[axis, axis] = q1[0, 0]
            q[axis, axis + 3] = q1[0, 1]
            q[axis + 3, axis] = q1[1, 0]
            q[axis + 3, axis + 3] = q1[1, 1]
        return f, q

    def predict_once(x_in: np.ndarray, p_in: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        f, q = make_f_q(dt)
        x_out = f @ x_in
        p_out = f @ p_in @ f.T + q
        return x_out, p_out

    for k in range(t):
        if k > 0:
            dt = float(timestamps[k] - timestamps[k - 1])
            if not np.isfinite(dt) or dt <= 0.0 or dt > 1.0:
                dt = dt_fallback
            x, p = predict_once(x, p, dt)

        zk = z_xyz[k]
        if not np.isfinite(zk).all():
            valid[k] = False
        else:
            y = zk - (h @ x)
            s = h @ p @ h.T + r
            k_gain = p @ h.T @ np.linalg.inv(s)
            x = x + (k_gain @ y)
            p = (i6 - (k_gain @ h)) @ p

        state[k] = x
        pos[k] = x[:3]
        vel[k] = x[3:]

        x_exec, _ = predict_once(x, p, predict_horizon)
        pred_exec[k] = x_exec[:3]

    return {
        "kalman_state": state.astype(np.float32),
        "kalman_pos": pos.astype(np.float32),
        "kalman_vel": vel.astype(np.float32),
        "kalman_pred_exec": pred_exec.astype(np.float32),
        "kalman_meas": z_xyz.astype(np.float32),
        "kalman_valid": valid,
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    pos_slice = _parse_slice(args.state_pos_slice)

    episode_meta = _load_episode_meta(args.dataset_root)
    ep_map = {x.episode_index: x for x in episode_meta}
    all_indices = [x.episode_index for x in episode_meta]
    target_indices = _iter_episode_indices(all_indices, args)

    print(f"[INIT] total_episodes={len(all_indices)} selected={len(target_indices)}", flush=True)

    data_cache = DataFrameCache(max_files=args.datafile_cache_size)
    data_columns = ["episode_index", "index", "frame_index", "timestamp", "task_index", "state"]

    t0 = time.perf_counter()
    done = 0
    skipped = 0
    failed = 0

    for ep_idx in target_indices:
        dst_ep_dir = args.output_root / f"episode_{ep_idx:06d}"
        done_file = dst_ep_dir / ".done"
        if args.skip_existing and done_file.exists() and not args.overwrite:
            done += 1
            skipped += 1
            if done % 10 == 0 or done == len(target_indices):
                print(f"[{done}/{len(target_indices)}] skipped={skipped} failed={failed}", flush=True)
            continue

        try:
            ep_info = ep_map[ep_idx]
            data_path = (
                args.dataset_root
                / "data"
                / f"chunk-{ep_info.data_chunk_index:03d}"
                / f"file-{ep_info.data_file_index:03d}.parquet"
            )
            df = data_cache.get(
                (ep_info.data_chunk_index, ep_info.data_file_index),
                data_path,
                columns=data_columns,
            )
            ep_df = df.loc[df["episode_index"] == ep_idx].sort_values("index").reset_index(drop=True)
            if len(ep_df) == 0:
                raise ValueError(f"No frames found for episode={ep_idx}")

            states = np.stack([np.asarray(x, dtype=np.float32) for x in ep_df["state"]], axis=0)
            z_xyz = states[:, pos_slice]
            if z_xyz.shape[1] != 3:
                raise ValueError(
                    f"state slice {args.state_pos_slice} produced shape {z_xyz.shape}; expected (*,3)"
                )
            timestamps = ep_df["timestamp"].to_numpy(dtype=np.float64)

            k_out = _run_kalman(
                z_xyz,
                timestamps,
                dt_fallback=args.dt_fallback,
                predict_horizon=args.predict_horizon,
                meas_noise_std=args.meas_noise_std,
                accel_noise_std=args.accel_noise_std,
                init_pos_std=args.init_pos_std,
                init_vel_std=args.init_vel_std,
            )

            dataset_index = ep_df["index"].to_numpy(dtype=np.int64)
            frame_index = ep_df["frame_index"].to_numpy(dtype=np.int64)
            task_index = ep_df["task_index"].to_numpy(dtype=np.int64)
            has_prev = np.zeros_like(dataset_index, dtype=np.bool_)
            has_prev[1:] = True
            prev_dataset_index = dataset_index.copy()
            prev_dataset_index[0] = dataset_index[0]
            prev_dataset_index[1:] = dataset_index[:-1]

            arrays: dict[str, np.ndarray] = {
                **k_out,
                "dataset_index": dataset_index,
                "frame_index": frame_index,
                "timestamp": timestamps.astype(np.float64),
                "task_index": task_index,
                "has_prev": has_prev,
                "prev_dataset_index": prev_dataset_index,
                "episode_index": np.full((len(ep_df),), ep_idx, dtype=np.int64),
            }

            metadata = {k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in arrays.items()}
            _write_episode_arrays(dst_ep_dir, arrays, metadata, overwrite=args.overwrite)

            done += 1
            if done % 10 == 0 or done == len(target_indices):
                elapsed = time.perf_counter() - t0
                print(
                    f"[{done}/{len(target_indices)}] skipped={skipped} failed={failed} "
                    f"elapsed={elapsed/60:.1f}m",
                    flush=True,
                )

        except Exception as exc:
            failed += 1
            done += 1
            print(f"[ERROR] episode={ep_idx} err={type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()

    summary = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "selected_episodes": len(target_indices),
        "completed": done - failed - skipped,
        "skipped": skipped,
        "failed": failed,
        "state_pos_slice": args.state_pos_slice,
        "predict_horizon": args.predict_horizon,
        "dt_fallback": args.dt_fallback,
        "meas_noise_std": args.meas_noise_std,
        "accel_noise_std": args.accel_noise_std,
        "init_pos_std": args.init_pos_std,
        "init_vel_std": args.init_vel_std,
    }
    (args.output_root / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

