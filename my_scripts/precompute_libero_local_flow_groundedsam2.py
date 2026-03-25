#!/usr/bin/env python3
"""Build local optical-flow maps from precomputed global flow using GroundingDINO + SAM2 masks.

Input:
  - LIBERO dataset root (for RGB frames + task_index)
  - precomputed global flow root in npy layout:
      episode_XXXXXX/arrays.json + *.npy

Output:
  - local flow root in the same npy layout:
      episode_XXXXXX/arrays.json + *.npy
    where each flow_* array is background-zeroed by union masks from:
      1) arm prompt
      2) task prompt (resolved from task_index -> tasks.parquet)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch


@dataclass
class EpisodeMeta:
    episode_index: int
    data_chunk_index: int
    data_file_index: int


@dataclass
class CameraTrackState:
    prev_box_xyxy: np.ndarray | None = None
    prev_mask: np.ndarray | None = None
    miss_count: int = 0


class DataFrameCache:
    """Small LRU cache for parquet data files."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--input-flow-root", type=Path, required=True)
    parser.add_argument("--output-flow-root", type=Path, required=True)

    parser.add_argument("--grounded-sam2-root", type=Path, required=True)
    parser.add_argument("--sam2-config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--grounding-dino-config",
        type=str,
        default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument("--grounding-dino-checkpoint", type=Path, required=True)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--mask-dilate-kernel", type=int, default=5)
    parser.add_argument("--arm-prompt", type=str, default="robot arm. gripper.")
    parser.add_argument("--use-task-prompt", action="store_true")
    parser.add_argument(
        "--localize-cameras",
        type=str,
        default="image",
        help="Comma-separated camera names that use DINO+SAM2 local flow; others keep global flow.",
    )
    parser.add_argument("--temporal-iou-threshold", type=float, default=0.20)
    parser.add_argument("--temporal-hold-frames", type=int, default=3)
    parser.add_argument("--temporal-box-ema", type=float, default=0.6)
    parser.add_argument("--box-area-min-ratio", type=float, default=0.005)
    parser.add_argument("--box-area-max-ratio", type=float, default=0.60)
    parser.add_argument("--mask-keep-components", type=int, default=2)
    parser.add_argument("--mask-border-margin", type=int, default=8)
    parser.add_argument("--mask-close-kernel", type=int, default=3)

    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)

    parser.add_argument("--datafile-cache-size", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-mask", action="store_true")
    return parser.parse_args()


def _decode_png_cell(cell: Any) -> np.ndarray:
    if not isinstance(cell, dict) or "bytes" not in cell:
        raise TypeError(f"Unexpected image cell type: {type(cell)}")
    arr = np.frombuffer(cell["bytes"], dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imdecode failed")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _load_task_map(dataset_root: Path) -> dict[int, str]:
    task_path = dataset_root / "meta" / "tasks.parquet"
    df = pd.read_parquet(task_path)
    if "task" not in df.columns:
        df = df.reset_index()
    if "task" not in df.columns or "task_index" not in df.columns:
        raise ValueError(f"Unexpected tasks schema: columns={df.columns.tolist()}")
    return {int(row.task_index): str(row.task) for row in df.itertuples(index=False)}


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


def _normalize_prompt(text: str) -> str:
    s = text.strip().lower()
    if not s.endswith("."):
        s += "."
    return s


def _build_prompt(arm_prompt: str, task_text: str, use_task_prompt: bool) -> str:
    if use_task_prompt:
        return f"{_normalize_prompt(arm_prompt)} {_normalize_prompt(task_text)}"
    return _normalize_prompt(arm_prompt)


def _load_npy_arrays(ep_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    meta_path = ep_dir / "arrays.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing arrays metadata: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays = {
        k: np.load(ep_dir / f"{k}.npy", mmap_mode="r", allow_pickle=False)
        for k in metadata.keys()
    }
    return metadata, arrays


def _resolve_cfg_path(root: Path, rel_path: str, *, sam2_fallback: bool = False) -> Path:
    candidate = root / rel_path
    if candidate.exists():
        return candidate
    if sam2_fallback:
        fallback = root / "sam2" / rel_path
        if fallback.exists():
            return fallback
    return candidate


def _normalize_sam2_config_name(root: Path, cfg: str) -> str:
    cfg = cfg.strip()
    cfg_path = Path(cfg)
    sam2_root = root / "sam2"

    if cfg_path.is_absolute():
        try:
            cfg = str(cfg_path.relative_to(sam2_root))
        except ValueError as exc:
            raise ValueError(f"SAM2 config must be under {sam2_root}, got: {cfg_path}") from exc

    if cfg.startswith("sam2/"):
        cfg = cfg[len("sam2/") :]

    return cfg


def _write_episode_arrays(
    dst_episode_dir: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
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


def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.astype(np.float32).tolist()
    bx1, by1, bx2, by2 = b.astype(np.float32).tolist()
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _clip_box_xyxy(box: np.ndarray, height: int, width: int) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32).tolist()
    x1 = min(max(x1, 0.0), width - 1.0)
    x2 = min(max(x2, 0.0), width - 1.0)
    y1 = min(max(y1, 0.0), height - 1.0)
    y2 = min(max(y2, 0.0), height - 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _refine_mask_components(
    mask_u8: np.ndarray,
    prev_mask_u8: np.ndarray | None,
    *,
    keep_components: int,
    border_margin: int,
    close_kernel: int,
) -> np.ndarray:
    if mask_u8.max() == 0:
        return mask_u8
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8.astype(np.uint8), connectivity=8)
    if num <= 2:
        refined = mask_u8.astype(np.uint8)
    else:
        h, w = mask_u8.shape
        margin = max(1, int(border_margin))
        border = np.zeros((h, w), dtype=np.bool_)
        border[:margin, :] = True
        border[-margin:, :] = True
        border[:, :margin] = True
        border[:, -margin:] = True

        prev = prev_mask_u8.astype(bool) if prev_mask_u8 is not None else None
        candidates: list[tuple[float, int]] = []
        for cid in range(1, num):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            comp = labels == cid
            touches_border = bool(np.any(comp & border))
            if prev is not None:
                inter = int(np.logical_and(comp, prev).sum())
                union = int(np.logical_or(comp, prev).sum())
                prev_sum = int(prev.sum())
                iou_prev = (inter / union) if union > 0 else 0.0
                overlap_prev = (inter / prev_sum) if prev_sum > 0 else 0.0
            else:
                iou_prev = 0.0
                overlap_prev = 0.0

            score = (
                2.0 * iou_prev
                + 1.2 * overlap_prev
                + (0.8 if touches_border else 0.0)
                + 0.05 * float(np.log1p(area))
            )
            candidates.append((score, cid))

        if not candidates:
            refined = mask_u8.astype(np.uint8)
        else:
            candidates.sort(key=lambda x: x[0], reverse=True)
            keep_n = max(1, min(int(keep_components), len(candidates)))
            refined = np.zeros_like(mask_u8, dtype=np.uint8)
            for _, cid in candidates[:keep_n]:
                refined[labels == cid] = 1

    if close_kernel > 1:
        k = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, k, iterations=1)
    return refined.astype(np.uint8)


def main() -> None:
    args = parse_args()
    args.output_flow_root.mkdir(parents=True, exist_ok=True)
    localize_cameras = {x.strip() for x in args.localize_cameras.split(",") if x.strip()}
    if not localize_cameras:
        raise ValueError("--localize-cameras cannot be empty")

    if str(args.grounded_sam2_root) not in sys.path:
        sys.path.insert(0, str(args.grounded_sam2_root))

    from grounding_dino.groundingdino.util.inference import Model as GroundingDinoModel
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    grounding_cfg = _resolve_cfg_path(args.grounded_sam2_root, args.grounding_dino_config, sam2_fallback=False)
    sam2_cfg_name = _normalize_sam2_config_name(args.grounded_sam2_root, args.sam2_config)
    sam2_cfg = _resolve_cfg_path(args.grounded_sam2_root, sam2_cfg_name, sam2_fallback=True)
    if not grounding_cfg.exists():
        raise FileNotFoundError(f"Missing GroundingDINO config: {grounding_cfg}")
    if not sam2_cfg.exists():
        raise FileNotFoundError(f"Missing SAM2 config: {sam2_cfg}")
    if not args.grounding_dino_checkpoint.exists():
        raise FileNotFoundError(f"Missing GroundingDINO checkpoint: {args.grounding_dino_checkpoint}")
    if not args.sam2_checkpoint.exists():
        raise FileNotFoundError(f"Missing SAM2 checkpoint: {args.sam2_checkpoint}")

    print(f"[INIT] device={device}", flush=True)
    grounding_model = GroundingDinoModel(
        model_config_path=str(grounding_cfg),
        model_checkpoint_path=str(args.grounding_dino_checkpoint),
        device=str(device),
    )
    sam2_model = build_sam2(sam2_cfg_name, str(args.sam2_checkpoint), device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    task_map = _load_task_map(args.dataset_root)
    episode_meta = _load_episode_meta(args.dataset_root)
    ep_map = {x.episode_index: x for x in episode_meta}
    all_indices = [x.episode_index for x in episode_meta]
    target_indices = _iter_episode_indices(all_indices, args)
    print(f"[INIT] total_episodes={len(all_indices)} selected={len(target_indices)}", flush=True)

    data_cache = DataFrameCache(max_files=args.datafile_cache_size)
    image_columns = ["image", "wrist_image"]
    data_columns = ["episode_index", "index", "frame_index", "timestamp", "task_index", *image_columns]

    t0 = time.perf_counter()
    done = 0
    skipped = 0
    failed = 0

    for ep_idx in target_indices:
        src_ep_dir = args.input_flow_root / f"episode_{ep_idx:06d}"
        dst_ep_dir = args.output_flow_root / f"episode_{ep_idx:06d}"
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

            src_meta, src_arrays = _load_npy_arrays(src_ep_dir)
            flow_keys = [k for k in src_arrays.keys() if k.startswith("flow_")]
            if not flow_keys:
                raise ValueError(f"No flow_* arrays in {src_ep_dir}")
            flow_cameras = {k.removeprefix("flow_") for k in flow_keys}
            unknown_localize = sorted(localize_cameras - flow_cameras)
            if unknown_localize:
                print(
                    f"[WARN] episode={ep_idx} localize cameras not found in flow arrays: {unknown_localize}",
                    flush=True,
                )

            dataset_index = np.asarray(src_arrays["dataset_index"])
            ep_dataset_index = ep_df["index"].to_numpy(dtype=np.int64)
            if dataset_index.shape[0] != ep_dataset_index.shape[0]:
                raise ValueError(
                    f"Length mismatch in episode={ep_idx}: flow={dataset_index.shape[0]} data={ep_dataset_index.shape[0]}"
                )
            if not np.array_equal(dataset_index.astype(np.int64), ep_dataset_index):
                raise ValueError(f"dataset_index mismatch in episode={ep_idx}")

            local_arrays: dict[str, np.ndarray] = {}
            for key, arr in src_arrays.items():
                if key.startswith("flow_"):
                    camera = key.removeprefix("flow_")
                    if camera in localize_cameras:
                        local_arrays[key] = np.zeros_like(np.asarray(arr), dtype=np.float16)
                    else:
                        local_arrays[key] = np.asarray(arr)
                else:
                    local_arrays[key] = np.asarray(arr)

            if args.save_mask:
                for flow_key in flow_keys:
                    local_arrays[f"mask_{flow_key.removeprefix('flow_')}"] = np.zeros(
                        src_arrays[flow_key].shape[:3], dtype=np.bool_
                    )

            dilate_kernel = None
            if args.mask_dilate_kernel > 1:
                dilate_kernel = np.ones((args.mask_dilate_kernel, args.mask_dilate_kernel), dtype=np.uint8)
            track_states = {cam: CameraTrackState() for cam in localize_cameras}

            for i in range(len(ep_df)):
                task_index = int(ep_df.iloc[i]["task_index"])
                task_text = task_map.get(task_index, "")
                if not task_text:
                    raise ValueError(f"task_index={task_index} not found in tasks.parquet")
                prompt = _build_prompt(args.arm_prompt, task_text, args.use_task_prompt)

                for flow_key in flow_keys:
                    camera = flow_key.removeprefix("flow_")
                    if camera not in ep_df.columns:
                        raise KeyError(
                            f"Camera '{camera}' missing in data parquet. available={ep_df.columns.tolist()}"
                        )
                    if camera not in localize_cameras:
                        if args.save_mask:
                            local_arrays[f"mask_{camera}"][i] = True
                        continue

                    state = track_states[camera]
                    rgb = _decode_png_cell(ep_df.iloc[i][camera])  # HWC RGB uint8
                    h, w = rgb.shape[:2]
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    detections, _ = grounding_model.predict_with_caption(
                        image=bgr,
                        caption=prompt,
                        box_threshold=args.box_threshold,
                        text_threshold=args.text_threshold,
                    )
                    det_boxes = detections.xyxy
                    det_conf = getattr(detections, "confidence", None)
                    filtered_boxes: list[np.ndarray] = []
                    filtered_conf: list[float] = []
                    if det_boxes is not None and len(det_boxes) > 0:
                        for k, box in enumerate(np.asarray(det_boxes, dtype=np.float32)):
                            box = _clip_box_xyxy(box, h, w)
                            bw = max(0.0, float(box[2] - box[0]))
                            bh = max(0.0, float(box[3] - box[1]))
                            area_ratio = (bw * bh) / float(h * w)
                            if area_ratio < args.box_area_min_ratio or area_ratio > args.box_area_max_ratio:
                                continue
                            filtered_boxes.append(box)
                            conf_k = float(det_conf[k]) if det_conf is not None else 0.0
                            filtered_conf.append(conf_k)

                    chosen_box: np.ndarray | None = None
                    used_hold = False
                    if filtered_boxes:
                        if state.prev_box_xyxy is not None:
                            ious = [_box_iou_xyxy(box, state.prev_box_xyxy) for box in filtered_boxes]
                            best_iou_idx = int(np.argmax(np.asarray(ious)))
                            if ious[best_iou_idx] >= args.temporal_iou_threshold:
                                chosen_box = filtered_boxes[best_iou_idx]
                            else:
                                best_conf_idx = int(np.argmax(np.asarray(filtered_conf)))
                                chosen_box = filtered_boxes[best_conf_idx]
                        else:
                            best_conf_idx = int(np.argmax(np.asarray(filtered_conf)))
                            chosen_box = filtered_boxes[best_conf_idx]
                    elif state.prev_box_xyxy is not None and state.miss_count < args.temporal_hold_frames:
                        chosen_box = state.prev_box_xyxy.copy()
                        used_hold = True
                        state.miss_count += 1
                    else:
                        state.prev_mask = None
                        state.miss_count = min(state.miss_count + 1, args.temporal_hold_frames + 1)
                        continue

                    sam2_predictor.set_image(rgb)
                    masks, _, _ = sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=np.asarray(chosen_box, dtype=np.float32)[None, :],
                        multimask_output=False,
                    )

                    if masks.ndim == 4:
                        masks = np.squeeze(masks, axis=1)
                    if masks.ndim != 3:
                        raise ValueError(f"Unexpected mask shape: {masks.shape}")
                    union_mask = np.any(masks, axis=0).astype(np.uint8)
                    if dilate_kernel is not None:
                        union_mask = cv2.dilate(union_mask, dilate_kernel, iterations=1)
                    union_mask = _refine_mask_components(
                        union_mask,
                        state.prev_mask,
                        keep_components=args.mask_keep_components,
                        border_margin=args.mask_border_margin,
                        close_kernel=args.mask_close_kernel,
                    )
                    if state.prev_mask is not None and not used_hold:
                        prev = state.prev_mask.astype(bool)
                        curr = union_mask.astype(bool)
                        inter = np.logical_and(prev, curr).sum()
                        union = np.logical_or(prev, curr).sum()
                        if union > 0 and (inter / union) < 0.01 and state.miss_count < args.temporal_hold_frames:
                            union_mask = state.prev_mask.copy()
                            used_hold = True
                            state.miss_count += 1

                    flow_hw2 = np.asarray(src_arrays[flow_key][i], dtype=np.float32)
                    flow_hw2[union_mask == 0] = 0.0
                    local_arrays[flow_key][i] = flow_hw2.astype(np.float16, copy=False)

                    if args.save_mask:
                        local_arrays[f"mask_{camera}"][i] = union_mask.astype(bool)

                    if not used_hold:
                        if state.prev_box_xyxy is None:
                            state.prev_box_xyxy = chosen_box.copy()
                        else:
                            alpha = float(args.temporal_box_ema)
                            state.prev_box_xyxy = alpha * chosen_box + (1.0 - alpha) * state.prev_box_xyxy
                        state.prev_box_xyxy = _clip_box_xyxy(state.prev_box_xyxy, h, w)
                        state.miss_count = 0
                    state.prev_mask = union_mask.copy()

            dst_meta = dict(src_meta)
            if args.save_mask:
                for key in list(local_arrays.keys()):
                    if key.startswith("mask_"):
                        arr = local_arrays[key]
                        dst_meta[key] = {"dtype": str(arr.dtype), "shape": list(arr.shape)}

            _write_episode_arrays(dst_ep_dir, local_arrays, dst_meta, overwrite=args.overwrite)
            index_json = src_ep_dir / "index.json"
            if index_json.exists():
                shutil.copy2(index_json, dst_ep_dir / "index.json")

            done += 1
            if done % 5 == 0 or done == len(target_indices):
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
        "input_flow_root": str(args.input_flow_root),
        "output_flow_root": str(args.output_flow_root),
        "selected_episodes": len(target_indices),
        "completed": done - failed - skipped,
        "skipped": skipped,
        "failed": failed,
        "device": str(device),
        "sam2_checkpoint": str(args.sam2_checkpoint),
        "grounding_dino_checkpoint": str(args.grounding_dino_checkpoint),
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "mask_dilate_kernel": args.mask_dilate_kernel,
        "arm_prompt": args.arm_prompt,
        "localize_cameras": sorted(localize_cameras),
        "temporal_iou_threshold": args.temporal_iou_threshold,
        "temporal_hold_frames": args.temporal_hold_frames,
        "temporal_box_ema": args.temporal_box_ema,
        "box_area_min_ratio": args.box_area_min_ratio,
        "box_area_max_ratio": args.box_area_max_ratio,
        "mask_keep_components": args.mask_keep_components,
        "mask_border_margin": args.mask_border_margin,
        "mask_close_kernel": args.mask_close_kernel,
    }
    (args.output_flow_root / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
