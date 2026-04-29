#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train the low-dimensional latency residual model from sidecar + offline dataset.

Example:

python src/lerobot/scripts/train_latency_residual.py \
  --data.sidecar_path=/share/project/wujiling/datasets/libero_dyn_mini_sidecar \
  --data.repo_id=libero_dyn_mini \
  --data.root=/share/project/wujiling/datasets/libero_dyn_mini \
  --batch_size=512 \
  --epochs=30 \
  --device=cuda
"""

import datetime as dt
import json
import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.latency_residual import (
    LatencyResidualDataConfig,
    LatencyResidualDataset,
    LatencyResidualMLP,
    LatencyResidualModelConfig,
    json_ready,
    make_latency_residual_episode_split,
    save_latency_residual_checkpoint,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


@dataclass
class TrainLatencyResidualConfig:
    data: LatencyResidualDataConfig = field(default_factory=LatencyResidualDataConfig)
    model: LatencyResidualModelConfig = field(default_factory=LatencyResidualModelConfig)
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int = 1000
    device: str = "cuda"
    use_amp: bool = True
    batch_size: int = 512
    num_workers: int = 4
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    compile_model: bool = False
    log_interval: int = 100
    val_fraction: float = 0.1
    split_seed: int = 0
    train_episode_indices: list[int] | None = None
    val_episode_indices: list[int] | None = None
    save_every_epoch: bool = False

    def __post_init__(self) -> None:
        if self.data.episodes is None:
            # The current residual workflow is anchored to the exported sidecar
            # built from the first 200 libero_dyn_mini episodes. Keep the
            # default training scope aligned with that sidecar unless the user
            # explicitly overrides episodes from the CLI.
            self.data.episodes = list(range(200))
        if not self.job_name:
            sidecar_name = Path(self.data.sidecar_path).name.rstrip("/") or "sidecar"
            self.job_name = f"latency_residual_{sidecar_name}"
        if self.output_dir is None:
            now = dt.datetime.now()
            self.output_dir = Path("outputs/latency_residual") / f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _aggregate_metrics(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {key: float(np.mean([metric[key] for metric in metric_dicts])) for key in keys}


def _evaluate(
    model: LatencyResidualMLP,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    if len(loader.dataset) == 0:
        return {}

    autocast_ctx = (
        torch.autocast(device_type=device.type)
        if amp_enabled and device.type == "cuda"
        else nullcontext()
    )

    metrics: list[dict[str, float]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            with autocast_ctx:
                _, batch_metrics, _ = model.compute_loss(batch)
            metrics.append(batch_metrics)
    return _aggregate_metrics(metrics)


@parser.wrap()
def train_main(cfg: TrainLatencyResidualConfig) -> None:
    init_logging()
    logging.info(pformat(json_ready(asdict(cfg))))

    output_dir = Path(cfg.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    device = get_safe_torch_device(cfg.device, log=True)
    amp_enabled = bool(cfg.use_amp) and device.type == "cuda"
    if cfg.seed is not None:
        set_seed(int(cfg.seed))

    logging.info("[latency-residual][train] building residual dataset from sidecar")
    dataset = LatencyResidualDataset(cfg.data)
    logging.info(
        "[latency-residual][train] dataset ready num_samples=%d feature_spec=%s",
        len(dataset),
        dataset.feature_spec,
    )
    split = make_latency_residual_episode_split(
        dataset,
        val_fraction=float(cfg.val_fraction),
        split_seed=int(cfg.split_seed),
        train_episode_indices=cfg.train_episode_indices,
        val_episode_indices=cfg.val_episode_indices,
    )

    train_dataset = Subset(dataset, split.train_sample_indices.tolist())
    val_dataset = Subset(dataset, split.val_sample_indices.tolist())

    cfg.model.sync_from_data(dataset.feature_spec, cfg.data)
    model = LatencyResidualMLP(cfg.model).to(device)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg.num_workers) > 0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg.num_workers) > 0,
        drop_last=False,
    )

    (output_dir / "train_config.json").write_text(
        json.dumps(json_ready(asdict(cfg)), indent=2, ensure_ascii=False)
    )
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(json_ready(dataset.describe()), indent=2, ensure_ascii=False)
    )
    (output_dir / "split.json").write_text(
        json.dumps(
            {
                "train_episodes": split.train_episodes,
                "val_episodes": split.val_episodes,
                "num_train_samples": int(len(train_dataset)),
                "num_val_samples": int(len(val_dataset)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    autocast_ctx = (
        torch.autocast(device_type=device.type)
        if amp_enabled and device.type == "cuda"
        else nullcontext()
    )
    metrics_log_path = output_dir / "metrics.jsonl"

    best_val_loss = float("inf")
    best_epoch = -1
    global_step = 0

    for epoch in range(int(cfg.epochs)):
        model.train()
        running_metrics: list[dict[str, float]] = []
        progbar = tqdm(
            train_loader,
            desc=f"Latency residual train epoch {epoch + 1}/{cfg.epochs}",
            leave=False,
        )
        for batch_idx, batch in enumerate(progbar):
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx:
                loss, batch_metrics, _ = model.compute_loss(batch)

            if amp_enabled:
                grad_scaler.scale(loss).backward()
                if cfg.grad_clip_norm > 0:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
                optimizer.step()

            running_metrics.append(batch_metrics)
            global_step += 1
            if (batch_idx + 1) % int(cfg.log_interval) == 0 or batch_idx == 0:
                averaged = _aggregate_metrics(running_metrics[-max(1, int(cfg.log_interval)) :])
                progbar.set_postfix(
                    loss=f"{averaged.get('loss', 0.0):.4f}",
                    corr_l1=f"{averaged.get('corrected_l1', 0.0):.4f}",
                    improve=f"{averaged.get('improvement_l1', 0.0):.4f}",
                )

        train_metrics = _aggregate_metrics(running_metrics)
        val_metrics = _evaluate(
            model,
            val_loader,
            device=device,
            amp_enabled=amp_enabled,
        )

        epoch_record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
        }
        with metrics_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(json_ready(epoch_record), ensure_ascii=False) + "\n")

        logging.info(
            "[latency-residual][epoch=%d] train=%s val=%s",
            epoch + 1,
            train_metrics,
            val_metrics,
        )

        current_val_loss = float(val_metrics.get("loss", train_metrics.get("loss", 0.0)))
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch = epoch + 1
            save_latency_residual_checkpoint(
                model,
                cfg.model,
                output_dir / "best",
                metadata={
                    "epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
            )

        if cfg.save_every_epoch:
            save_latency_residual_checkpoint(
                model,
                cfg.model,
                output_dir / f"epoch_{epoch + 1:03d}",
                metadata={
                    "epoch": epoch + 1,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
            )

    final_val_metrics = _evaluate(
        model,
        val_loader,
        device=device,
        amp_enabled=amp_enabled,
    )
    save_latency_residual_checkpoint(
        model,
        cfg.model,
        output_dir / "last",
        metadata={
            "epoch": int(cfg.epochs),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_val_metrics": final_val_metrics,
        },
    )

    summary = {
        "output_dir": str(output_dir),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_metrics": final_val_metrics,
        "num_train_samples": int(len(train_dataset)),
        "num_val_samples": int(len(val_dataset)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False)
    )
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))


def main() -> None:
    train_main()


if __name__ == "__main__":
    main()
