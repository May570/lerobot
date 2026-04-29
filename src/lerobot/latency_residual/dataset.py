"""Dataset utilities for offline latency-residual training."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .config import LatencyResidualDataConfig, LatencyResidualFeatureSpec

LOGGER = logging.getLogger(__name__)
RAW_HISTORY_LOG_EVERY = 50000
SIDECAR_SAMPLE_LOG_EVERY = 5000


@dataclass
class LatencyResidualEpisodeSplit:
    """Episode-level split metadata used by the training script."""

    train_episodes: list[int]
    val_episodes: list[int]
    train_sample_indices: np.ndarray
    val_sample_indices: np.ndarray


def _stack_dataset_column(values: list[Any], *, dtype: np.dtype, name: str) -> np.ndarray:
    stacked: list[np.ndarray] = []
    for value in values:
        if torch.is_tensor(value):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        stacked.append(array.astype(dtype, copy=False))
    if not stacked:
        raise ValueError(f"Dataset column {name!r} is empty.")
    return np.stack(stacked, axis=0)


def _resolve_sidecar_path(root: str | Path, parquet_name: str) -> Path:
    path = Path(root)
    if path.is_file():
        if path.suffix != ".parquet":
            raise ValueError(f"Expected a parquet file, got {path}.")
        return path
    if not path.exists():
        raise FileNotFoundError(f"Sidecar path does not exist: {path}")
    direct_candidate = path / parquet_name
    if direct_candidate.exists():
        return direct_candidate
    candidates = sorted(path.rglob(parquet_name))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {parquet_name!r} under sidecar path {path}."
        )
    raise ValueError(
        f"Found multiple sidecar parquet candidates under {path}: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


class LatencyResidualDataset(torch.utils.data.Dataset):
    """Chunk-level sidecar expanded into fixed-shape single-step residual samples.

    Each sample corresponds to one anchor time `t0`, one runtime drop count `d`, and
    one within-suffix execution offset `u`. The target is the expert action at
    `te = t0 + d + u`.
    """

    def __init__(self, cfg: LatencyResidualDataConfig):
        cfg.validate()
        self.cfg = cfg
        init_t0 = time.perf_counter()
        self.sidecar_path = _resolve_sidecar_path(cfg.sidecar_path, cfg.sidecar_parquet_name)
        self.sidecar_root = self.sidecar_path.parent
        self.sidecar_summary = self._load_optional_summary(self.sidecar_root / cfg.sidecar_summary_name)

        LOGGER.info(
            "[latency-residual][init] repo_id=%s dataset_root=%s sidecar_path=%s episodes=%s",
            cfg.repo_id,
            cfg.root,
            self.sidecar_path,
            cfg.episodes if cfg.episodes is not None else "all",
        )

        self.raw_dataset = LeRobotDataset(
            repo_id=cfg.repo_id,
            root=cfg.root,
            revision=cfg.revision,
            episodes=list(cfg.episodes) if cfg.episodes is not None else None,
        )
        self.raw_dataset._ensure_hf_dataset_loaded()

        stage_t0 = time.perf_counter()
        LOGGER.info("[latency-residual][init] loading raw low-dimensional dataset arrays")
        self._load_raw_arrays()
        LOGGER.info(
            "[latency-residual][init] raw arrays loaded in %.2fs",
            time.perf_counter() - stage_t0,
        )

        stage_t0 = time.perf_counter()
        LOGGER.info("[latency-residual][init] loading sidecar parquet rows")
        self._load_sidecar_arrays()
        LOGGER.info(
            "[latency-residual][init] sidecar rows loaded in %.2fs",
            time.perf_counter() - stage_t0,
        )

        stage_t0 = time.perf_counter()
        LOGGER.info("[latency-residual][init] expanding sidecar rows into single-step residual samples")
        self._build_sample_index()
        LOGGER.info(
            "[latency-residual][init] sample index built in %.2fs",
            time.perf_counter() - stage_t0,
        )

        self.feature_spec = LatencyResidualFeatureSpec(
            state_dim=int(self.state_values.shape[1]),
            ball_dim=int(0 if self.ball_values is None else self.ball_values.shape[1]),
            action_dim=int(self.action_values.shape[1]),
            chunk_len=int(self.pred_action_chunks.shape[1]),
            history_len=int(self.cfg.history_len),
            local_plan_horizon=int(self.cfg.local_plan_horizon),
        )
        LOGGER.info(
            "[latency-residual][init] complete in %.2fs num_raw_frames=%d num_sidecar_rows=%d num_samples=%d",
            time.perf_counter() - init_t0,
            len(self.absolute_indices),
            len(self.sidecar_absolute_indices),
            len(self),
        )

    def _load_optional_summary(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            logging.warning("Failed to parse sidecar summary %s: %s", path, exc)
            return None

    def _load_raw_arrays(self) -> None:
        hf_dataset = self.raw_dataset.hf_dataset
        state_key = self.cfg.state_key
        if state_key not in hf_dataset.column_names:
            raise KeyError(
                f"State key {state_key!r} not found in dataset columns {hf_dataset.column_names}."
            )
        action_key = self.cfg.action_key
        if action_key not in hf_dataset.column_names:
            raise KeyError(
                f"Action key {action_key!r} not found in dataset columns {hf_dataset.column_names}."
            )

        LOGGER.info("[latency-residual][raw] stacking state column: %s", state_key)
        self.state_values = _stack_dataset_column(hf_dataset[state_key], dtype=np.float32, name=state_key)
        LOGGER.info("[latency-residual][raw] state shape=%s", self.state_values.shape)

        LOGGER.info("[latency-residual][raw] stacking action column: %s", action_key)
        self.action_values = _stack_dataset_column(hf_dataset[action_key], dtype=np.float32, name=action_key)
        LOGGER.info("[latency-residual][raw] action shape=%s", self.action_values.shape)

        ball_key = self.cfg.ball_key
        self.ball_values: np.ndarray | None = None
        if ball_key:
            if ball_key not in hf_dataset.column_names:
                raise KeyError(
                    f"Ball key {ball_key!r} not found in dataset columns {hf_dataset.column_names}."
                )
            LOGGER.info("[latency-residual][raw] stacking ball/state-of-scene column: %s", ball_key)
            self.ball_values = _stack_dataset_column(hf_dataset[ball_key], dtype=np.float32, name=ball_key)
            LOGGER.info("[latency-residual][raw] ball shape=%s", self.ball_values.shape)

        self.absolute_indices = _stack_dataset_column(
            hf_dataset["index"], dtype=np.int64, name="index"
        ).reshape(-1)
        self.episode_indices = _stack_dataset_column(
            hf_dataset["episode_index"], dtype=np.int64, name="episode_index"
        ).reshape(-1)
        self.frame_indices = self._build_frame_indices()

        max_abs_index = int(self.absolute_indices.max())
        self.absolute_to_row = np.full(max_abs_index + 1, -1, dtype=np.int64)
        self.absolute_to_row[self.absolute_indices] = np.arange(len(self.absolute_indices), dtype=np.int64)

        self.episode_starts = np.zeros(len(self.raw_dataset.meta.episodes), dtype=np.int64)
        self.episode_ends = np.zeros(len(self.raw_dataset.meta.episodes), dtype=np.int64)
        for ep_idx, episode in enumerate(self.raw_dataset.meta.episodes):
            self.episode_starts[ep_idx] = int(episode["dataset_from_index"])
            self.episode_ends[ep_idx] = int(episode["dataset_to_index"])

        self.history_rows = np.empty((len(self.absolute_indices), self.cfg.history_len), dtype=np.int64)
        for row_idx, (abs_idx, ep_idx) in enumerate(zip(self.absolute_indices, self.episode_indices, strict=True)):
            ep_start = int(self.episode_starts[ep_idx])
            history_abs = np.arange(
                int(abs_idx) - self.cfg.history_len + 1,
                int(abs_idx) + 1,
                dtype=np.int64,
            )
            history_abs = np.maximum(history_abs, ep_start)
            history_rows = self.absolute_to_row[history_abs]
            if np.any(history_rows < 0):
                raise ValueError(
                    f"Failed to resolve history rows for absolute index {abs_idx} in episode {ep_idx}."
                )
            self.history_rows[row_idx] = history_rows
            if (row_idx + 1) % RAW_HISTORY_LOG_EVERY == 0:
                LOGGER.info(
                    "[latency-residual][raw] built history rows for %d / %d frames",
                    row_idx + 1,
                    len(self.absolute_indices),
                )
        LOGGER.info(
            "[latency-residual][raw] history rows ready for %d frames history_len=%d",
            len(self.absolute_indices),
            self.cfg.history_len,
        )

    def _build_frame_indices(self) -> np.ndarray:
        frame_indices = np.empty(len(self.absolute_indices), dtype=np.int64)
        for row_idx, (abs_idx, ep_idx) in enumerate(zip(self.absolute_indices, self.episode_indices, strict=True)):
            frame_indices[row_idx] = int(abs_idx) - int(self.raw_dataset.meta.episodes[int(ep_idx)]["dataset_from_index"])
        return frame_indices

    def _load_sidecar_arrays(self) -> None:
        columns = ["episode_index", "frame_index", "index", "timestamp", "pred_action_chunk"]
        table = pq.read_table(self.sidecar_path, columns=columns)
        if self.cfg.max_sidecar_rows is not None:
            table = table.slice(0, int(self.cfg.max_sidecar_rows))

        sidecar_episode_indices = np.asarray(
            table["episode_index"].to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        sidecar_frame_indices = np.asarray(
            table["frame_index"].to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        sidecar_absolute_indices = np.asarray(
            table["index"].to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        sidecar_timestamps = np.asarray(
            table["timestamp"].to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        pred_action_chunks = np.asarray(
            table["pred_action_chunk"].to_pylist(),
            dtype=np.float32,
        )
        LOGGER.info(
            "[latency-residual][sidecar] loaded parquet rows=%d chunk_shape=%s",
            len(sidecar_episode_indices),
            pred_action_chunks.shape,
        )

        if self.cfg.episodes is not None:
            allowed_episodes = np.asarray(sorted(set(int(ep) for ep in self.cfg.episodes)), dtype=np.int64)
            keep_mask = np.isin(sidecar_episode_indices, allowed_episodes)
            kept_rows = int(keep_mask.sum())
            sidecar_episode_indices = sidecar_episode_indices[keep_mask]
            sidecar_frame_indices = sidecar_frame_indices[keep_mask]
            sidecar_absolute_indices = sidecar_absolute_indices[keep_mask]
            sidecar_timestamps = sidecar_timestamps[keep_mask]
            pred_action_chunks = pred_action_chunks[keep_mask]
            LOGGER.info(
                "[latency-residual][sidecar] filtered to selected episodes=%s kept_rows=%d",
                allowed_episodes.tolist(),
                kept_rows,
            )

        self.sidecar_episode_indices = sidecar_episode_indices
        self.sidecar_frame_indices = sidecar_frame_indices
        self.sidecar_absolute_indices = sidecar_absolute_indices
        self.sidecar_timestamps = sidecar_timestamps
        self.pred_action_chunks = pred_action_chunks
        if self.pred_action_chunks.ndim != 3:
            raise ValueError(
                f"Expected sidecar pred_action_chunk to have rank 3, got shape {self.pred_action_chunks.shape}."
            )
        self.sidecar_anchor_rows = self._lookup_raw_rows(self.sidecar_absolute_indices)
        if np.any(self.sidecar_anchor_rows < 0):
            missing = self.sidecar_absolute_indices[self.sidecar_anchor_rows < 0][:10]
            raise ValueError(
                "Some sidecar anchor indices do not exist in the raw dataset. "
                f"Examples: {missing.tolist()}"
            )
        LOGGER.info(
            "[latency-residual][sidecar] anchor alignment ok rows=%d",
            len(self.sidecar_anchor_rows),
        )

    def _lookup_raw_rows(self, absolute_indices: np.ndarray) -> np.ndarray:
        max_abs = len(self.absolute_to_row) - 1
        rows = np.full(len(absolute_indices), -1, dtype=np.int64)
        valid = (absolute_indices >= 0) & (absolute_indices <= max_abs)
        rows[valid] = self.absolute_to_row[absolute_indices[valid]]
        return rows

    def _build_sample_index(self) -> None:
        allowed_episodes = None
        if self.cfg.episodes is not None:
            allowed_episodes = set(int(ep) for ep in self.cfg.episodes)

        anchor_rows: list[int] = []
        current_rows: list[int] = []
        delay_steps: list[int] = []
        step_offsets: list[int] = []
        chunk_indices: list[int] = []
        sample_weights: list[float] = []

        chunk_len = int(self.pred_action_chunks.shape[1])
        total_sidecar_rows = len(self.sidecar_absolute_indices)
        for sidecar_row, (ep_idx, anchor_abs) in enumerate(
            zip(self.sidecar_episode_indices, self.sidecar_absolute_indices, strict=True)
        ):
            if allowed_episodes is not None and int(ep_idx) not in allowed_episodes:
                continue

            episode_end = int(self.episode_ends[int(ep_idx)])
            for delay in self.cfg.delay_values:
                for step_offset in self.cfg.step_offsets:
                    chunk_idx = int(delay) + int(step_offset)
                    if chunk_idx >= chunk_len:
                        continue
                    current_abs = int(anchor_abs) + chunk_idx
                    if current_abs >= episode_end:
                        continue
                    current_row = self._lookup_raw_rows(np.asarray([current_abs], dtype=np.int64))[0]
                    if current_row < 0:
                        continue
                    anchor_rows.append(sidecar_row)
                    current_rows.append(int(current_row))
                    delay_steps.append(int(delay))
                    step_offsets.append(int(step_offset))
                    chunk_indices.append(chunk_idx)
                    sample_weights.append(float(self.cfg.step_weight_gamma) ** float(step_offset))
            if (sidecar_row + 1) % SIDECAR_SAMPLE_LOG_EVERY == 0:
                LOGGER.info(
                    "[latency-residual][samples] processed %d / %d anchor rows current_samples=%d",
                    sidecar_row + 1,
                    total_sidecar_rows,
                    len(anchor_rows),
                )

        if not anchor_rows:
            raise ValueError(
                "No valid latency residual samples were constructed from the sidecar. "
                "Check the sidecar path, selected episodes, and delay/offset ranges."
            )

        self.sample_anchor_rows = np.asarray(anchor_rows, dtype=np.int64)
        self.sample_current_rows = np.asarray(current_rows, dtype=np.int64)
        self.sample_delay_steps = np.asarray(delay_steps, dtype=np.int64)
        self.sample_step_offsets = np.asarray(step_offsets, dtype=np.int64)
        self.sample_chunk_indices = np.asarray(chunk_indices, dtype=np.int64)
        self.sample_weights = np.asarray(sample_weights, dtype=np.float32)
        self.sample_episode_indices = self.sidecar_episode_indices[self.sample_anchor_rows]
        LOGGER.info(
            "[latency-residual][samples] finished anchor_rows=%d expanded_samples=%d",
            total_sidecar_rows,
            len(self.sample_anchor_rows),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "repo_id": self.cfg.repo_id,
            "dataset_root": self.cfg.root,
            "sidecar_path": str(self.sidecar_path),
            "num_raw_frames": int(len(self.absolute_indices)),
            "num_sidecar_rows": int(len(self.sidecar_absolute_indices)),
            "num_samples": int(len(self)),
            "num_unique_anchor_episodes": int(len(np.unique(self.sidecar_episode_indices))),
            "delay_values": list(self.cfg.delay_values),
            "step_offsets": list(self.cfg.step_offsets),
            "feature_spec": {
                "state_dim": int(self.feature_spec.state_dim),
                "ball_dim": int(self.feature_spec.ball_dim),
                "action_dim": int(self.feature_spec.action_dim),
                "chunk_len": int(self.feature_spec.chunk_len),
                "history_len": int(self.feature_spec.history_len),
                "local_plan_horizon": int(self.feature_spec.local_plan_horizon),
            },
        }

    def __len__(self) -> int:
        return int(len(self.sample_anchor_rows))

    def _slice_local_plan(self, action_chunk: np.ndarray, start_idx: int) -> np.ndarray:
        horizon = int(self.cfg.local_plan_horizon)
        if horizon == 0:
            action_dim = int(action_chunk.shape[-1])
            return np.zeros((0, action_dim), dtype=np.float32)
        end_idx = start_idx + horizon
        if end_idx <= action_chunk.shape[0]:
            return action_chunk[start_idx:end_idx]

        # Repeat the final action when the requested local plan extends past the chunk tail.
        local_plan = np.repeat(action_chunk[-1:], horizon, axis=0)
        valid = action_chunk[start_idx:]
        local_plan[: len(valid)] = valid
        return local_plan

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        anchor_row = int(self.sample_anchor_rows[idx])
        current_row = int(self.sample_current_rows[idx])
        delay_steps = int(self.sample_delay_steps[idx])
        step_offset = int(self.sample_step_offsets[idx])
        chunk_idx = int(self.sample_chunk_indices[idx])

        anchor_raw_row = int(self.sidecar_anchor_rows[anchor_row])
        anchor_history_rows = self.history_rows[anchor_raw_row]
        current_history_rows = self.history_rows[current_row]

        action_chunk = self.pred_action_chunks[anchor_row]
        base_action = action_chunk[chunk_idx]
        local_plan = self._slice_local_plan(action_chunk, chunk_idx)
        target_action = self.action_values[current_row]

        item = {
            "anchor_state_history": torch.from_numpy(
                self.state_values[anchor_history_rows].copy()
            ),
            "current_state_history": torch.from_numpy(
                self.state_values[current_history_rows].copy()
            ),
            "base_action": torch.from_numpy(base_action.copy()),
            "target_action": torch.from_numpy(target_action.copy()),
            "target_residual": torch.from_numpy((target_action - base_action).copy()),
            "delay_steps": torch.tensor(delay_steps, dtype=torch.long),
            "step_offset": torch.tensor(step_offset, dtype=torch.long),
            "chunk_index": torch.tensor(chunk_idx, dtype=torch.long),
            "sample_weight": torch.tensor(self.sample_weights[idx], dtype=torch.float32),
            "episode_index": torch.tensor(int(self.sample_episode_indices[idx]), dtype=torch.long),
            "anchor_index": torch.tensor(int(self.sidecar_absolute_indices[anchor_row]), dtype=torch.long),
            "current_index": torch.tensor(int(self.absolute_indices[current_row]), dtype=torch.long),
        }

        if self.cfg.local_plan_horizon > 0:
            item["local_plan"] = torch.from_numpy(local_plan.copy())

        if self.ball_values is not None:
            item["anchor_ball_history"] = torch.from_numpy(
                self.ball_values[anchor_history_rows].copy()
            )
            item["current_ball_history"] = torch.from_numpy(
                self.ball_values[current_history_rows].copy()
            )
        return item


def make_latency_residual_episode_split(
    dataset: LatencyResidualDataset,
    *,
    val_fraction: float,
    split_seed: int,
    train_episode_indices: list[int] | None = None,
    val_episode_indices: list[int] | None = None,
) -> LatencyResidualEpisodeSplit:
    """Split the expanded sample set by anchor episode.

    Splitting by episode avoids leakage between train and validation for nearby
    frames coming from the same rollout.
    """

    unique_episodes = sorted(int(ep) for ep in np.unique(dataset.sample_episode_indices))
    unique_episode_set = set(unique_episodes)

    explicit_train = None if train_episode_indices is None else sorted(int(ep) for ep in train_episode_indices)
    explicit_val = None if val_episode_indices is None else sorted(int(ep) for ep in val_episode_indices)

    if explicit_train is not None and not set(explicit_train).issubset(unique_episode_set):
        raise ValueError(
            f"train_episode_indices contains episodes not present in the dataset: {explicit_train}"
        )
    if explicit_val is not None and not set(explicit_val).issubset(unique_episode_set):
        raise ValueError(
            f"val_episode_indices contains episodes not present in the dataset: {explicit_val}"
        )
    if explicit_train is not None and explicit_val is not None:
        overlap = sorted(set(explicit_train) & set(explicit_val))
        if overlap:
            raise ValueError(f"Train/val episode splits overlap: {overlap}")

    if explicit_train is None and explicit_val is None:
        if not 0.0 <= float(val_fraction) < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}.")
        if not unique_episodes:
            raise ValueError("Cannot split an empty residual dataset.")

        rng = np.random.default_rng(int(split_seed))
        shuffled = np.asarray(unique_episodes, dtype=np.int64)
        rng.shuffle(shuffled)
        val_count = int(round(len(shuffled) * float(val_fraction)))
        if val_count >= len(shuffled):
            val_count = max(0, len(shuffled) - 1)
        val_episodes = sorted(int(ep) for ep in shuffled[:val_count].tolist())
        train_episodes = sorted(int(ep) for ep in shuffled[val_count:].tolist())
    else:
        if explicit_val is None:
            val_episodes = []
        else:
            val_episodes = explicit_val
        if explicit_train is None:
            train_episodes = sorted(unique_episode_set - set(val_episodes))
        else:
            train_episodes = explicit_train

    if not train_episodes:
        raise ValueError("Train split is empty after episode partitioning.")

    sample_eps = dataset.sample_episode_indices
    train_mask = np.isin(sample_eps, np.asarray(train_episodes, dtype=np.int64))
    val_mask = np.isin(sample_eps, np.asarray(val_episodes, dtype=np.int64))

    train_sample_indices = np.nonzero(train_mask)[0].astype(np.int64)
    val_sample_indices = np.nonzero(val_mask)[0].astype(np.int64)

    return LatencyResidualEpisodeSplit(
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        train_sample_indices=train_sample_indices,
        val_sample_indices=val_sample_indices,
    )
