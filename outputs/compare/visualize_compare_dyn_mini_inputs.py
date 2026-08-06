#!/usr/bin/env python3
"""Visualize offline vs online image inputs from compare_dyn_mini_model_inputs.py."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-dir",
        required=True,
        help="Directory produced by compare_dyn_mini_model_inputs.py",
    )
    return parser.parse_args()


def load_pt(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload)}")
    return payload


def image_tensor_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")
    if image.shape[0] not in {1, 3}:
        raise ValueError(f"Expected 1 or 3 channels, got shape {tuple(image.shape)}")
    arr = image.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.round(arr * 255.0).astype(np.uint8)


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, arr)


def save_heatmap(path: Path, diff01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, diff01, cmap="magma", vmin=0.0, vmax=float(max(0.05, diff01.max())))


def key_to_slug(key: str) -> str:
    return key.replace(".", "_")


def build_panel(
    *,
    out_path: Path,
    camera_key: str,
    offline: torch.Tensor,
    online: torch.Tensor,
    deltas: list[int] | None,
) -> dict[str, Any]:
    if offline.shape != online.shape:
        raise ValueError(f"Offline/online shapes differ for {camera_key}: {offline.shape} vs {online.shape}")
    if offline.ndim != 4:
        raise ValueError(f"Expected SxCxHxW tensor for {camera_key}, got {tuple(offline.shape)}")

    steps = int(offline.shape[0])
    fig, axes = plt.subplots(steps, 3, figsize=(12, 4 * steps), squeeze=False)
    per_step_stats: list[dict[str, Any]] = []

    for step_idx in range(steps):
        offline_rgb = image_tensor_to_uint8_hwc(offline[step_idx])
        online_rgb = image_tensor_to_uint8_hwc(online[step_idx])
        diff = np.abs(offline_rgb.astype(np.float32) - online_rgb.astype(np.float32)) / 255.0
        diff_map = diff.mean(axis=2)

        delta_label = None
        if deltas and step_idx < len(deltas):
            delta_label = deltas[step_idx]

        row_title = f"step_index={step_idx}"
        if delta_label is not None:
            row_title += f" delta={delta_label}"

        axes[step_idx, 0].imshow(offline_rgb)
        axes[step_idx, 0].set_title(f"Offline {row_title}")
        axes[step_idx, 1].imshow(online_rgb)
        axes[step_idx, 1].set_title(f"Online {row_title}")
        axes[step_idx, 2].imshow(diff_map, cmap="magma", vmin=0.0, vmax=max(0.05, float(diff_map.max())))
        axes[step_idx, 2].set_title(
            f"Abs Diff {row_title}\nmean={diff_map.mean():.4f} max={diff_map.max():.4f}"
        )
        for ax in axes[step_idx]:
            ax.axis("off")

        step_slug = f"t{step_idx}"
        save_rgb(out_path.parent / f"{key_to_slug(camera_key)}_{step_slug}_offline.png", offline_rgb)
        save_rgb(out_path.parent / f"{key_to_slug(camera_key)}_{step_slug}_online.png", online_rgb)
        save_heatmap(out_path.parent / f"{key_to_slug(camera_key)}_{step_slug}_diff.png", diff_map)

        per_step_stats.append(
            {
                "step_index": step_idx,
                "delta": delta_label,
                "mean_abs_diff_01": float(diff_map.mean()),
                "max_abs_diff_01": float(diff_map.max()),
                "pct_gt_1_255": float((diff_map > (1.0 / 255.0)).mean()),
                "pct_gt_5_255": float((diff_map > (5.0 / 255.0)).mean()),
                "pct_gt_10_255": float((diff_map > (10.0 / 255.0)).mean()),
                "pct_gt_20_255": float((diff_map > (20.0 / 255.0)).mean()),
            }
        )

    fig.suptitle(camera_key, fontsize=16)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {"camera_key": camera_key, "steps": per_step_stats}


def build_overview(
    *,
    out_path: Path,
    offline_batch: dict[str, Any],
    online_batch: dict[str, Any],
    camera_keys: list[str],
    deltas: list[int] | None,
) -> None:
    rows = sum(int(offline_batch[key].shape[0]) for key in camera_keys)
    fig, axes = plt.subplots(rows, 3, figsize=(12, max(4, 3.2 * rows)), squeeze=False)
    row = 0
    for camera_key in camera_keys:
        offline = offline_batch[camera_key]
        online = online_batch[camera_key]
        for step_idx in range(int(offline.shape[0])):
            offline_rgb = image_tensor_to_uint8_hwc(offline[step_idx])
            online_rgb = image_tensor_to_uint8_hwc(online[step_idx])
            diff_map = np.abs(offline_rgb.astype(np.float32) - online_rgb.astype(np.float32)).mean(axis=2) / 255.0
            delta_label = None
            if deltas and step_idx < len(deltas):
                delta_label = deltas[step_idx]
            label = f"{camera_key} | t={step_idx}"
            if delta_label is not None:
                label += f" | delta={delta_label}"
            axes[row, 0].imshow(offline_rgb)
            axes[row, 0].set_title(f"Offline\n{label}")
            axes[row, 1].imshow(online_rgb)
            axes[row, 1].set_title(f"Online\n{label}")
            axes[row, 2].imshow(diff_map, cmap="magma", vmin=0.0, vmax=max(0.05, float(diff_map.max())))
            axes[row, 2].set_title(f"Abs Diff\nmean={diff_map.mean():.4f}")
            for ax in axes[row]:
                ax.axis("off")
            row += 1
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    compare_dir = Path(args.compare_dir).expanduser().resolve()
    offline_before = load_pt(compare_dir / "offline_before_policy_pre.pt")
    online_before = load_pt(compare_dir / "online_before_policy_pre.pt")

    metadata_path = compare_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    deltas = metadata.get("policy_observation_delta_indices")
    if deltas is not None and not isinstance(deltas, list):
        deltas = None

    camera_keys = sorted(
        key
        for key, value in offline_before.items()
        if key.startswith("observation.images.") and torch.is_tensor(value)
    )
    if not camera_keys:
        raise ValueError(f"No image keys found in {compare_dir / 'offline_before_policy_pre.pt'}")

    viz_dir = compare_dir / "viz_before_policy_pre"
    viz_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {"compare_dir": str(compare_dir), "camera_panels": []}
    for camera_key in camera_keys:
        panel_path = viz_dir / f"{key_to_slug(camera_key)}_panel.png"
        panel_stats = build_panel(
            out_path=panel_path,
            camera_key=camera_key,
            offline=offline_before[camera_key],
            online=online_before[camera_key],
            deltas=deltas,
        )
        stats["camera_panels"].append(panel_stats)

    build_overview(
        out_path=viz_dir / "overview_panel.png",
        offline_batch=offline_before,
        online_batch=online_before,
        camera_keys=camera_keys,
        deltas=deltas,
    )

    stats_path = viz_dir / "viz_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved visualizations to: {viz_dir}")


if __name__ == "__main__":
    main()
