#!/usr/bin/env python3
"""Analyze how offline/online input differences propagate through DiffusionPolicy."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import einops

REPO_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC_ROOT = REPO_ROOT / "src"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf/datasets")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg")
if str(LEROBOT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC_ROOT))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy.path", dest="policy_path", required=True)
    parser.add_argument("--compare-dir", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_batch(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload)}")
    return payload


def to_device_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def ensure_batch_dim(batch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.unsqueeze(0)
        else:
            out[key] = value
    return out


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    aa = a.detach().to("cpu")
    bb = b.detach().to("cpu")
    result: dict[str, Any] = {
        "shape_a": list(aa.shape),
        "shape_b": list(bb.shape),
        "dtype_a": str(aa.dtype),
        "dtype_b": str(bb.dtype),
        "shape_equal": tuple(aa.shape) == tuple(bb.shape),
        "dtype_equal": aa.dtype == bb.dtype,
    }
    if tuple(aa.shape) != tuple(bb.shape):
        return result
    if not aa.is_floating_point():
        aa = aa.float()
    if not bb.is_floating_point():
        bb = bb.float()
    diff = (aa - bb).abs()
    result["max_abs_diff"] = float(diff.max().item()) if diff.numel() else 0.0
    result["mean_abs_diff"] = float(diff.mean().item()) if diff.numel() else 0.0
    result["l2_norm"] = float(torch.linalg.vector_norm((aa - bb).reshape(-1)).item()) if diff.numel() else 0.0
    result["ref_l2_norm_a"] = float(torch.linalg.vector_norm(aa.reshape(-1)).item()) if diff.numel() else 0.0
    result["ref_l2_norm_b"] = float(torch.linalg.vector_norm(bb.reshape(-1)).item()) if diff.numel() else 0.0
    if result["ref_l2_norm_a"] > 0:
        result["relative_l2_vs_a"] = result["l2_norm"] / result["ref_l2_norm_a"]
    if result["ref_l2_norm_b"] > 0:
        result["relative_l2_vs_b"] = result["l2_norm"] / result["ref_l2_norm_b"]
    return result


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path).expanduser().resolve()
    compare_dir = Path(args.compare_dir).expanduser().resolve()

    offline_batch = load_batch(compare_dir / "offline_final_model_batch.pt")
    online_batch = load_batch(compare_dir / "online_final_model_batch.pt")

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=["--device=cpu"])
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=policy_cfg, strict=True)
    policy.eval()

    offline_batch = ensure_batch_dim(to_device_batch(offline_batch, policy_cfg.device))
    online_batch = ensure_batch_dim(to_device_batch(online_batch, policy_cfg.device))

    with torch.no_grad():
        n_obs_steps = policy.config.n_obs_steps
        obs_images_off = offline_batch["observation.images"][:, :n_obs_steps]
        obs_images_on = online_batch["observation.images"][:, :n_obs_steps]
        batch_size = obs_images_off.shape[0]

        if policy.config.use_separate_rgb_encoder_per_camera:
            raise NotImplementedError("This helper currently assumes shared RGB encoder.")

        offline_img_features = policy.diffusion.rgb_encoder(
            einops.rearrange(obs_images_off, "b s n ... -> (b s n) ...")
        )
        online_img_features = policy.diffusion.rgb_encoder(
            einops.rearrange(obs_images_on, "b s n ... -> (b s n) ...")
        )
        offline_img_features = einops.rearrange(
            offline_img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
        )
        online_img_features = einops.rearrange(
            online_img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
        )

        offline_global_cond, offline_global_cond_mid = policy.diffusion._prepare_unet_conditioning(offline_batch)
        online_global_cond, online_global_cond_mid = policy.diffusion._prepare_unet_conditioning(online_batch)

        action_dim = policy.config.action_feature.shape[0]
        generator = torch.Generator(device=policy_cfg.device)
        generator.manual_seed(int(args.seed))
        fixed_noise = torch.randn(
            (batch_size, policy.config.horizon, action_dim),
            generator=generator,
            device=policy_cfg.device,
            dtype=next(policy.parameters()).dtype,
        )

        offline_actions = policy.diffusion.generate_actions(offline_batch, noise=fixed_noise.clone())
        online_actions = policy.diffusion.generate_actions(online_batch, noise=fixed_noise.clone())

    report = {
        "policy_path": str(policy_path),
        "compare_dir": str(compare_dir),
        "seed": int(args.seed),
        "offline_online_diffs": {
            "rgb_encoder_output": compare_tensors(offline_img_features, online_img_features),
            "global_cond": compare_tensors(offline_global_cond, online_global_cond),
            "global_cond_mid": (
                compare_tensors(offline_global_cond_mid, online_global_cond_mid)
                if offline_global_cond_mid is not None and online_global_cond_mid is not None
                else {
                    "offline_is_none": offline_global_cond_mid is None,
                    "online_is_none": online_global_cond_mid is None,
                }
            ),
            "predicted_action_chunk": compare_tensors(offline_actions, online_actions),
        },
    }

    output_path = compare_dir / "policy_sensitivity.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved policy sensitivity report to: {output_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
