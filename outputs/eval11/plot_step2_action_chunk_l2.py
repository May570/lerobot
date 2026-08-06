#!/usr/bin/env python3
"""Plot per-episode action-chunk drift curves from step2 probe_result.json."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = Path(
    "/home/admin123/桌面/wjl/lerobot/outputs/eval11/step2_chunk_drift_20260509_193006/probe_result.json"
)
DEFAULT_OUTPUT = Path(
    "/home/admin123/桌面/wjl/lerobot/outputs/eval11/step2_chunk_drift_20260509_193006/action_chunk_l2_by_episode.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot action_chunk.l2 vs step_ix for all episodes from a step2 chunk_drift result."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to probe_result.json (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to output PNG (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Step2 chunk_drift: action_chunk.l2 vs step_ix",
        help="Figure title",
    )
    parser.add_argument(
        "--max-chunk",
        type=int,
        default=None,
        help="Only plot records with chunk_ix <= this value",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("per_chunk_record", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid per_chunk_record in {path}")
    return records


def group_episode_curves(records: list[dict], max_chunk: int | None = None) -> dict[int, list[tuple[int, float]]]:
    curves: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for record in records:
        chunk_ix = int(record.get("chunk_ix", -1))
        if max_chunk is not None and chunk_ix > max_chunk:
            continue
        episode_ix = int(record["episode_ix"])
        step_ix = int(record["step_ix"])
        action_l2 = float(record["action_chunk"]["l2"])
        curves[episode_ix].append((step_ix, action_l2))

    for episode_ix in curves:
        curves[episode_ix].sort(key=lambda item: item[0])
    return dict(sorted(curves.items(), key=lambda item: item[0]))


def plot_curves(curves: dict[int, list[tuple[int, float]]], output_path: Path, title: str) -> None:
    n_episodes = len(curves)
    ncols = 4
    nrows = (n_episodes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.6 * nrows), squeeze=False, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for ax, (episode_ix, points) in zip(axes_flat, curves.items(), strict=False):
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4)
        ax.set_title(f"episode {episode_ix}", fontsize=11)
        ax.set_xlabel("step_ix")
        ax.set_ylabel("action_chunk.l2")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n_episodes:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    curves = group_episode_curves(records, max_chunk=args.max_chunk)
    plot_curves(curves, args.output, args.title)
    print(f"saved plot to: {args.output}")


if __name__ == "__main__":
    main()
