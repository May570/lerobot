#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import fmean


TIMESTAMP_SUFFIX_RE = re.compile(r"_(\d{8}_\d{6})$")
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    default_input_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Summarize eval results, optionally averaging repeated runs per config."
    )
    parser.add_argument(
        "--input-dir",
        default=str(default_input_dir),
        help="Directory whose direct children are eval run folders containing eval_info.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Summary output directory. Defaults to <input-dir>_summary next to the input dir.",
    )
    return parser.parse_args()


def strip_timestamp(run_name: str) -> tuple[str, str | None]:
    match = TIMESTAMP_SUFFIX_RE.search(run_name)
    if not match:
        return run_name, None
    return run_name[: match.start()], match.group(1)


def relativize(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def normalize_number(value):
    if isinstance(value, float):
        return round(value, 6)
    return value


def mean_value(values: list[int | float]) -> float | None:
    if not values:
        return None
    return normalize_number(fmean(values))


def metric_sort_key(item: dict) -> tuple[int, float, str]:
    value = item["value"]
    if value is None:
        return (1, 0.0, item["config_name"])
    return (0, -float(value), item["config_name"])


def resolve_video_source(rel_video: Path, run_dir: Path, repo_root: Path) -> Path:
    candidate = repo_root / rel_video
    if candidate.exists():
        return candidate

    run_name = run_dir.name
    parts = rel_video.parts
    if run_name in parts:
        run_ix = parts.index(run_name)
        suffix = parts[run_ix + 1 :]
        candidate = run_dir.joinpath(*suffix)
        if candidate.exists():
            return candidate

    candidate = run_dir / "videos" / rel_video.name
    if candidate.exists():
        return candidate

    return repo_root / rel_video


def build_episode_rows(eval_info: dict, repo_root: Path, run_dir: Path) -> list[dict]:
    overall = eval_info["overall"]
    video_paths = overall.get("video_paths", [])
    traces = sorted(
        eval_info.get("gate_trace_per_episode", []),
        key=lambda item: item.get("global_episode_ix", 0),
    )

    if traces:
        episode_rows = []
        for trace in traces:
            global_ix = trace["global_episode_ix"]
            rel_video = Path(video_paths[global_ix])
            episode_rows.append(
                {
                    "global_episode_ix": global_ix,
                    "task_group": trace.get("task_group"),
                    "task_id": trace.get("task_id"),
                    "episode_ix_in_task": trace.get("episode_ix_in_task"),
                    "episode_name": rel_video.name,
                    "relative_video_path": str(rel_video),
                    "absolute_video_path": str(resolve_video_source(rel_video, run_dir, repo_root)),
                    "task_success": bool(trace.get("success", False)),
                    "ball_grasp_count": int(trace.get("ball_grasp_count", 0)),
                    "ball_grasp_success": int(trace.get("ball_grasp_count", 0)) > 0,
                }
            )
        return episode_rows

    episode_rows = []
    global_ix = 0
    for task in eval_info.get("per_task", []):
        metrics = task["metrics"]
        successes = metrics.get("successes", [])
        ball_grasp_counts = metrics.get("ball_grasp_counts", [])
        task_video_paths = metrics.get("video_paths", [])
        for episode_ix, rel_video_str in enumerate(task_video_paths):
            rel_video = Path(rel_video_str)
            ball_grasp_count = int(ball_grasp_counts[episode_ix])
            episode_rows.append(
                {
                    "global_episode_ix": global_ix,
                    "task_group": task.get("task_group"),
                    "task_id": task.get("task_id"),
                    "episode_ix_in_task": episode_ix,
                    "episode_name": rel_video.name,
                    "relative_video_path": str(rel_video),
                    "absolute_video_path": str(resolve_video_source(rel_video, run_dir, repo_root)),
                    "task_success": bool(successes[episode_ix]),
                    "ball_grasp_count": ball_grasp_count,
                    "ball_grasp_success": ball_grasp_count > 0,
                }
            )
            global_ix += 1
    return episode_rows


def select_episode_subset(episode_rows: list[dict], key: str) -> list[dict]:
    subset = []
    for row in episode_rows:
        if not row[key]:
            continue
        entry = {
            "global_episode_ix": row["global_episode_ix"],
            "episode_ix_in_task": row["episode_ix_in_task"],
            "episode_name": row["episode_name"],
            "relative_video_path": row["relative_video_path"],
        }
        if key == "ball_grasp_success":
            entry["ball_grasp_count"] = row["ball_grasp_count"]
        subset.append(entry)
    return subset


def copy_episode_videos(
    run_name: str,
    episode_rows: list[dict],
    dest_dir: Path,
    select_key: str,
) -> list[str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied_paths: list[str] = []
    for row in episode_rows:
        if not row[select_key]:
            continue
        src = Path(row["absolute_video_path"])
        if not src.exists():
            continue
        dst = dest_dir / f"{run_name}__{row['episode_name']}"
        shutil.copy2(src, dst)
        copied_paths.append(str(dst))
    return copied_paths


def build_metric_view(grouped_records: dict[str, list[dict]], key: str, label: str) -> dict:
    ranked = []
    mean_by_config = {}
    per_run_by_config = {}

    for config_name in sorted(grouped_records):
        runs = sorted(grouped_records[config_name], key=lambda item: item["timestamp"] or "")
        values = [run[key] for run in runs if run[key] is not None]
        mean_metric = mean_value(values)
        mean_by_config[config_name] = mean_metric
        per_run_by_config[config_name] = {
            run["timestamp"] or run["run_name"]: run[key] for run in runs
        }
        ranked.append(
            {
                "config_name": config_name,
                "value": mean_metric,
                "n_runs": len(runs),
                "per_run_values": per_run_by_config[config_name],
            }
        )

    ranked.sort(key=metric_sort_key)
    return {
        "label": label,
        "aggregation": "mean_across_runs_per_config",
        "sorted_desc": ranked,
        "mean_by_config": mean_by_config,
        "per_run_by_config": per_run_by_config,
    }


def build_episode_views(grouped_records: dict[str, list[dict]]) -> dict:
    task_names = {}
    task_details = {}
    ball_names = {}
    ball_details = {}

    for config_name in sorted(grouped_records):
        runs = sorted(grouped_records[config_name], key=lambda item: item["timestamp"] or "")
        task_names[config_name] = {}
        task_details[config_name] = {}
        ball_names[config_name] = {}
        ball_details[config_name] = {}
        for run in runs:
            run_key = run["timestamp"] or run["run_name"]
            task_names[config_name][run_key] = [
                episode["episode_name"] for episode in run["task_success_episodes"]
            ]
            task_details[config_name][run_key] = run["task_success_episodes"]
            ball_names[config_name][run_key] = [
                episode["episode_name"] for episode in run["ball_grasp_success_episodes"]
            ]
            ball_details[config_name][run_key] = run["ball_grasp_success_episodes"]

    return {
        "task_success_episode_names_by_config": task_names,
        "task_success_episode_details_by_config": task_details,
        "ball_grasp_success_episode_names_by_config": ball_names,
        "ball_grasp_success_episode_details_by_config": ball_details,
    }


def build_video_views(grouped_records: dict[str, list[dict]]) -> dict:
    task_dirs = {}
    ball_dirs = {}

    for config_name in sorted(grouped_records):
        runs = sorted(grouped_records[config_name], key=lambda item: item["timestamp"] or "")
        task_dirs[config_name] = {}
        ball_dirs[config_name] = {}
        for run in runs:
            run_key = run["timestamp"] or run["run_name"]
            task_dirs[config_name][run_key] = run["task_success_videos_dir"]
            ball_dirs[config_name][run_key] = run["ball_grasp_success_videos_dir"]

    return {
        "task_success_videos_by_config": task_dirs,
        "ball_grasp_success_videos_by_config": ball_dirs,
    }


def group_records(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["config_name"]].append(record)
    return dict(grouped)


def build_details_view(grouped_records: dict[str, list[dict]]) -> dict:
    runs_by_config = {}
    for config_name in sorted(grouped_records):
        runs = sorted(grouped_records[config_name], key=lambda item: item["timestamp"] or "")
        runs_by_config[config_name] = {
            (run["timestamp"] or run["run_name"]): run for run in runs
        }
    return runs_by_config


def determine_summary_root(input_dir: Path, output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg).expanduser().resolve()
    return input_dir.parent / f"{input_dir.name}_summary"


def collect_run_dirs(input_dir: Path) -> list[Path]:
    return sorted(path.parent for path in input_dir.glob("*/eval_info.json") if path.is_file())


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    summary_root = determine_summary_root(input_dir, args.output_dir)
    by_config_root = summary_root / "by_config"
    comparison_json = summary_root / "config_comparison_summary.json"
    details_json = summary_root / "config_run_details.json"

    run_dirs = collect_run_dirs(input_dir)
    if not run_dirs:
        raise SystemExit(f"No eval_info.json found directly under: {input_dir}")

    if summary_root.exists():
        shutil.rmtree(summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)
    by_config_root.mkdir(parents=True, exist_ok=True)

    records = []
    for run_dir in run_dirs:
        run_name = run_dir.name
        config_name, timestamp = strip_timestamp(run_name)
        eval_info_path = run_dir / "eval_info.json"
        with eval_info_path.open("r", encoding="utf-8") as f:
            eval_info = json.load(f)

        overall = eval_info["overall"]
        episode_rows = build_episode_rows(eval_info, REPO_ROOT, run_dir)
        task_success_episodes = select_episode_subset(episode_rows, "task_success")
        ball_grasp_success_episodes = select_episode_subset(episode_rows, "ball_grasp_success")

        run_key = timestamp or run_name
        run_summary_dir = by_config_root / config_name / run_key
        task_success_dir = run_summary_dir / "task_success_videos"
        ball_grasp_dir = run_summary_dir / "ball_grasp_success_videos"

        copy_episode_videos(run_name, episode_rows, task_success_dir, "task_success")
        copy_episode_videos(run_name, episode_rows, ball_grasp_dir, "ball_grasp_success")

        records.append(
            {
                "config_name": config_name,
                "run_name": run_name,
                "timestamp": timestamp,
                "source_run_dir": relativize(run_dir, REPO_ROOT),
                "source_eval_info": relativize(eval_info_path, REPO_ROOT),
                "task_success_videos_dir": relativize(task_success_dir, REPO_ROOT),
                "ball_grasp_success_videos_dir": relativize(ball_grasp_dir, REPO_ROOT),
                "n_episodes": overall.get("n_episodes", len(episode_rows)),
                "task_success_rate_pct": normalize_number(overall.get("pc_success")),
                "task_success_episode_count": len(task_success_episodes),
                "task_success_episodes": task_success_episodes,
                "ball_grasp_success_rate_pct": normalize_number(
                    overall.get("pc_ball_grasp_success")
                ),
                "ball_grasp_success_episode_count": len(ball_grasp_success_episodes),
                "avg_ball_grasp_count": normalize_number(overall.get("avg_ball_grasp_count")),
                "ball_grasp_success_episodes": ball_grasp_success_episodes,
            }
        )

    grouped_records = group_records(records)
    config_names = sorted(grouped_records)

    comparison_summary = {
        "meta": {
            "source_dir": relativize(input_dir, REPO_ROOT),
            "summary_root": relativize(summary_root, REPO_ROOT),
            "n_runs": len(records),
            "n_configs": len(config_names),
            "config_names": config_names,
            "runs_per_config": {
                config_name: len(grouped_records[config_name]) for config_name in config_names
            },
        },
        "comparison": {
            "task_success_rate_pct": build_metric_view(
                grouped_records,
                "task_success_rate_pct",
                "总任务成功率(%)",
            ),
            "task_success_episode_count": build_metric_view(
                grouped_records,
                "task_success_episode_count",
                "任务成功 episode 数",
            ),
            "ball_grasp_success_rate_pct": build_metric_view(
                grouped_records,
                "ball_grasp_success_rate_pct",
                "抓到小球成功率(%)",
            ),
            "ball_grasp_success_episode_count": build_metric_view(
                grouped_records,
                "ball_grasp_success_episode_count",
                "抓到小球成功 episode 数",
            ),
            "avg_ball_grasp_count": build_metric_view(
                grouped_records,
                "avg_ball_grasp_count",
                "平均抓球次数",
            ),
        },
        "episodes": build_episode_views(grouped_records),
        "video_dirs": build_video_views(grouped_records),
    }

    with comparison_json.open("w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    details_view = {
        "meta": comparison_summary["meta"],
        "runs_by_config": build_details_view(grouped_records),
    }
    with details_json.open("w", encoding="utf-8") as f:
        json.dump(details_view, f, ensure_ascii=False, indent=2)
        f.write("\n")

    manifest = {
        "input_dir": relativize(input_dir, REPO_ROOT),
        "comparison_json": relativize(comparison_json, REPO_ROOT),
        "details_json": relativize(details_json, REPO_ROOT),
        "summary_root": relativize(summary_root, REPO_ROOT),
        "n_runs": len(records),
        "n_configs": len(config_names),
        "runs_per_config": {
            config_name: len(grouped_records[config_name]) for config_name in config_names
        },
    }
    with (summary_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
