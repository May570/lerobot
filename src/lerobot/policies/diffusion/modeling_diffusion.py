#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

import math
import logging
import importlib
import json
import sys
from dataclasses import dataclass
from collections import OrderedDict, deque
from collections.abc import Callable
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn
from torchvision.ops import box_convert

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE, OBS_STATE_RAW

PRECOMPUTED_FLOW_PREFIX = "precomputed_flow_"
PRECOMPUTED_KALMAN_PREFIX = "precomputed_kalman_"


class PrecomputedOpticalFlowReader:
    """Read per-episode precomputed optical flow maps and build (B,S,N,2,H,W) tensors."""

    def __init__(
        self,
        root: str,
        image_features: list[str],
        n_obs_steps: int,
        observation_delta_indices: list[int] | None,
        cache_size: int,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Precomputed optical-flow root does not exist: {self.root}")

        self.image_features = image_features
        self.n_obs_steps = n_obs_steps
        self.cache_size = cache_size
        self._episode_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._episode_flow_key_cache: dict[int, dict[str, str]] = {}

        if observation_delta_indices is not None and len(observation_delta_indices) >= n_obs_steps:
            # Keep ordering consistent with the observation stack from dataset/preprocessor.
            self.obs_rel_indices = list(observation_delta_indices[-n_obs_steps:])
        else:
            # Default fallback for diffusion policy: past observations up to current frame.
            self.obs_rel_indices = list(range(-n_obs_steps + 1, 1))

    @staticmethod
    def _as_batch_scalar_list(x: Tensor) -> list[int]:
        # Accept (B,), (B,1), or any tensor where first value per sample is the scalar metadata.
        if x.ndim == 1:
            return x.to(torch.long).cpu().tolist()
        if x.ndim >= 2:
            return x.reshape(x.shape[0], -1)[:, 0].to(torch.long).cpu().tolist()
        return [int(x.item())]

    @staticmethod
    def _camera_aliases(feature_name: str) -> list[str]:
        if "observation.images." in feature_name:
            suffix = feature_name.split("observation.images.", maxsplit=1)[1]
        else:
            suffix = feature_name.split(".")[-1]

        aliases = [suffix]
        if suffix == "image2":
            aliases.insert(0, "wrist_image")
        if suffix == "wrist_image" and "image2" not in aliases:
            aliases.append("image2")
        if suffix == "image":
            aliases.insert(0, "image")

        # deduplicate preserving order
        seen = set()
        ordered = []
        for a in aliases:
            if a not in seen:
                seen.add(a)
                ordered.append(a)
        return ordered

    def _load_episode_arrays(self, episode_index: int) -> dict[str, np.ndarray]:
        if episode_index in self._episode_cache:
            arrays = self._episode_cache.pop(episode_index)
            self._episode_cache[episode_index] = arrays
            return arrays

        ep_dir = self.root / f"episode_{episode_index:06d}"
        index_path = ep_dir / "arrays.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing precomputed flow metadata file: {index_path}")
        with index_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        arrays = {
            key: np.load(ep_dir / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            for key in metadata.keys()
        }
        self._episode_cache[episode_index] = arrays

        while len(self._episode_cache) > self.cache_size:
            old_episode, _ = self._episode_cache.popitem(last=False)
            self._episode_flow_key_cache.pop(old_episode, None)

        return arrays

    def _resolve_feature_flow_key(self, arrays: dict[str, np.ndarray], feature_name: str) -> str:
        array_keys = set(arrays.keys())
        for alias in self._camera_aliases(feature_name):
            key = f"flow_{alias}"
            if key in array_keys:
                return key
        raise KeyError(
            f"Could not resolve flow key for feature '{feature_name}'. "
            f"Available keys: {sorted(array_keys)}"
        )

    def get_flow_maps(self, batch: dict[str, Tensor], ref_images: Tensor) -> Tensor:
        required = {"index", "frame_index", "episode_index"}
        missing = required.difference(batch.keys())
        if missing:
            raise KeyError(f"Missing batch keys for precomputed flow lookup: {sorted(missing)}")

        b, s, n, _, h, w = ref_images.shape
        out = torch.empty((b, s, n, 2, h, w), device=ref_images.device, dtype=ref_images.dtype)

        curr_indices = self._as_batch_scalar_list(batch["index"])
        frame_indices = self._as_batch_scalar_list(batch["frame_index"])
        episode_indices = self._as_batch_scalar_list(batch["episode_index"])

        for bi in range(b):
            ep_idx = episode_indices[bi]
            curr_idx = curr_indices[bi]
            frame_idx = frame_indices[bi]
            ep_start_idx = curr_idx - frame_idx

            arrays = self._load_episode_arrays(ep_idx)
            if ep_idx not in self._episode_flow_key_cache:
                self._episode_flow_key_cache[ep_idx] = {
                    feat: self._resolve_feature_flow_key(arrays, feat) for feat in self.image_features
                }

            dataset_indices = arrays["dataset_index"]
            for si, rel in enumerate(self.obs_rel_indices):
                target_global_idx = max(ep_start_idx, curr_idx + rel)
                local_idx = target_global_idx - ep_start_idx
                if local_idx < 0 or local_idx >= dataset_indices.shape[0]:
                    raise IndexError(
                        f"Local flow index out of range: local_idx={local_idx}, episode={ep_idx}, "
                        f"target_global_idx={target_global_idx}, ep_start_idx={ep_start_idx}, "
                        f"episode_len={dataset_indices.shape[0]}"
                    )
                if int(dataset_indices[local_idx]) != int(target_global_idx):
                    raise ValueError(
                        f"Flow index mismatch at episode={ep_idx}, local_idx={local_idx}: "
                        f"dataset_index={int(dataset_indices[local_idx])} != expected={target_global_idx}"
                    )

                for ni, feat in enumerate(self.image_features):
                    flow_key = self._episode_flow_key_cache[ep_idx][feat]
                    flow_hw2 = arrays[flow_key][local_idx]
                    flow_2hw = torch.from_numpy(flow_hw2).to(device=ref_images.device).permute(2, 0, 1)
                    out[bi, si, ni] = flow_2hw.to(dtype=ref_images.dtype)

        return out


class PrecomputedKalmanReader:
    """Read per-episode precomputed Kalman features and build (B,S,D) tensors."""

    def __init__(
        self,
        root: str,
        n_obs_steps: int,
        observation_delta_indices: list[int] | None,
        cache_size: int,
        feature_mode: str,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Precomputed Kalman root does not exist: {self.root}")
        self.n_obs_steps = n_obs_steps
        self.cache_size = cache_size
        self.feature_mode = feature_mode
        self.raw_dim = 10 if self.feature_mode == "full10" else 6
        self._episode_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

        if observation_delta_indices is not None and len(observation_delta_indices) >= n_obs_steps:
            self.obs_rel_indices = list(observation_delta_indices[-n_obs_steps:])
        else:
            self.obs_rel_indices = list(range(-n_obs_steps + 1, 1))

    @staticmethod
    def _as_batch_scalar_list(x: Tensor) -> list[int]:
        if x.ndim == 1:
            return x.to(torch.long).cpu().tolist()
        if x.ndim >= 2:
            return x.reshape(x.shape[0], -1)[:, 0].to(torch.long).cpu().tolist()
        return [int(x.item())]

    def _load_episode_arrays(self, episode_index: int) -> dict[str, np.ndarray]:
        if episode_index in self._episode_cache:
            arrays = self._episode_cache.pop(episode_index)
            self._episode_cache[episode_index] = arrays
            return arrays

        ep_dir = self.root / f"episode_{episode_index:06d}"
        index_path = ep_dir / "arrays.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing precomputed Kalman metadata file: {index_path}")
        with index_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        required = {"dataset_index", "kalman_pos", "kalman_vel"}
        if self.feature_mode == "full10":
            required.add("kalman_pred_exec")
        missing = required.difference(metadata.keys())
        if missing:
            raise KeyError(
                f"Missing required Kalman keys in {index_path}: {sorted(missing)}. "
                f"Available: {sorted(metadata.keys())}"
            )

        arrays = {
            key: np.load(ep_dir / f"{key}.npy", mmap_mode="r", allow_pickle=False) for key in metadata.keys()
        }
        self._episode_cache[episode_index] = arrays
        while len(self._episode_cache) > self.cache_size:
            self._episode_cache.popitem(last=False)
        return arrays

    def get_kalman_features(self, batch: dict[str, Tensor], ref_state: Tensor) -> Tensor:
        required = {"index", "frame_index", "episode_index"}
        missing = required.difference(batch.keys())
        if missing:
            raise KeyError(f"Missing batch keys for precomputed Kalman lookup: {sorted(missing)}")

        b, s = ref_state.shape[:2]
        out = torch.empty((b, s, self.raw_dim), device=ref_state.device, dtype=ref_state.dtype)

        curr_indices = self._as_batch_scalar_list(batch["index"])
        frame_indices = self._as_batch_scalar_list(batch["frame_index"])
        episode_indices = self._as_batch_scalar_list(batch["episode_index"])

        for bi in range(b):
            ep_idx = episode_indices[bi]
            curr_idx = curr_indices[bi]
            frame_idx = frame_indices[bi]
            ep_start_idx = curr_idx - frame_idx
            arrays = self._load_episode_arrays(ep_idx)
            dataset_indices = arrays["dataset_index"]

            for si, rel in enumerate(self.obs_rel_indices):
                target_global_idx = max(ep_start_idx, curr_idx + rel)
                local_idx = target_global_idx - ep_start_idx
                if local_idx < 0 or local_idx >= dataset_indices.shape[0]:
                    raise IndexError(
                        f"Local Kalman index out of range: local_idx={local_idx}, episode={ep_idx}, "
                        f"target_global_idx={target_global_idx}, ep_start_idx={ep_start_idx}, "
                        f"episode_len={dataset_indices.shape[0]}"
                    )
                if int(dataset_indices[local_idx]) != int(target_global_idx):
                    raise ValueError(
                        f"Kalman index mismatch at episode={ep_idx}, local_idx={local_idx}: "
                        f"dataset_index={int(dataset_indices[local_idx])} != expected={target_global_idx}"
                    )

                pos = torch.from_numpy(arrays["kalman_pos"][local_idx]).to(device=ref_state.device)
                vel = torch.from_numpy(arrays["kalman_vel"][local_idx]).to(device=ref_state.device)
                if self.feature_mode == "full10":
                    pred = torch.from_numpy(arrays["kalman_pred_exec"][local_idx]).to(device=ref_state.device)
                    if "kalman_valid" in arrays:
                        valid = float(arrays["kalman_valid"][local_idx])
                    else:
                        valid = 1.0
                    feat = torch.cat(
                        [pos.to(dtype=ref_state.dtype), vel.to(dtype=ref_state.dtype), pred.to(dtype=ref_state.dtype)]
                    )
                    out[bi, si, :9] = feat
                    out[bi, si, 9] = torch.tensor(valid, device=ref_state.device, dtype=ref_state.dtype)
                else:
                    out[bi, si, :3] = pos.to(dtype=ref_state.dtype)
                    out[bi, si, 3:6] = vel.to(dtype=ref_state.dtype)
        return out


class OnlineGMFlowRunner:
    """Run GMFlow on consecutive rollout frames and return dense flow (B,2,H,W)."""

    def __init__(
        self,
        *,
        repo_path: str,
        checkpoint_path: str,
        device: torch.device,
        use_amp: bool,
        padding_factor: int,
        attn_splits_list: tuple[int, ...],
        corr_radius_list: tuple[int, ...],
        prop_radius_list: tuple[int, ...],
    ):
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.padding_factor = padding_factor
        self.attn_splits_list = list(attn_splits_list)
        self.corr_radius_list = list(corr_radius_list)
        self.prop_radius_list = list(prop_radius_list)

        repo_root = Path(repo_path).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"GMFlow repo path does not exist: {repo_root}")
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        gmflow_module = importlib.import_module("gmflow.gmflow")
        utils_module = importlib.import_module("utils.utils")
        gmflow_cls = getattr(gmflow_module, "GMFlow")
        self._input_padder_cls = getattr(utils_module, "InputPadder")

        # Match the architecture used by our precompute script for consistency.
        self.model = gmflow_cls(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type="swin",
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        weights = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()

    @torch.no_grad()
    def infer(self, prev_images: Tensor, curr_images: Tensor) -> Tensor:
        """Estimate optical flow from previous to current frame."""
        image1 = prev_images.to(device=self.device, dtype=torch.float32, non_blocking=True)
        image2 = curr_images.to(device=self.device, dtype=torch.float32, non_blocking=True)

        # Runtime observations are often normalized to [0, 1]; GMFlow checkpoints are trained on 0-255 inputs.
        if float(image1.detach().amax()) <= 1.5 and float(image2.detach().amax()) <= 1.5:
            image1 = image1 * 255.0
            image2 = image2 * 255.0

        padder = self._input_padder_cls(image1.shape, padding_factor=self.padding_factor)
        image1, image2 = padder.pad(image1, image2)

        if self.use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                results = self.model(
                    image1,
                    image2,
                    attn_splits_list=self.attn_splits_list,
                    corr_radius_list=self.corr_radius_list,
                    prop_radius_list=self.prop_radius_list,
                )
        else:
            results = self.model(
                image1,
                image2,
                attn_splits_list=self.attn_splits_list,
                corr_radius_list=self.corr_radius_list,
                prop_radius_list=self.prop_radius_list,
            )

        flow = results["flow_preds"][-1]  # (B, 2, H, W)
        return padder.unpad(flow).contiguous()


@dataclass
class _MaskTrackState:
    prev_box_xyxy: np.ndarray | None = None
    prev_mask: np.ndarray | None = None
    miss_count: int = 0


@dataclass
class _OnlineKalmanRuntimeState:
    """Per-environment runtime cache for rollout-time stateful online Kalman."""

    x: Tensor | None = None
    p: Tensor | None = None
    prev_timestamp: float | None = None
    step_count: int = 0


class OnlineGroundedSAM2MaskRunner:
    """Compute rollout-time local masks using Grounded-DINO + SAM2 with fixed precompute-aligned params."""

    # Keep constants aligned with:
    # /share/project/wujiling/datasets/libero_flow_gmflow_f16_local_npy_temporal_v2/conversion_summary.json
    GROUNDEDSAM2_ROOT = "/share/project/wujiling/Grounded-SAM-2"
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    SAM2_CHECKPOINT = "/share/project/wujiling/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = (
        "/share/project/wujiling/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
    )
    BOX_THRESHOLD = 0.28
    TEXT_THRESHOLD = 0.22
    ARM_PROMPT = "robotic arm. gripper. end effector."
    TEMPORAL_IOU_THRESHOLD = 0.2
    TEMPORAL_HOLD_FRAMES = 3
    TEMPORAL_BOX_EMA = 0.6
    BOX_AREA_MIN_RATIO = 0.005
    BOX_AREA_MAX_RATIO = 0.6
    MASK_KEEP_COMPONENTS = 2
    MASK_BORDER_MARGIN = 8
    MASK_CLOSE_KERNEL = 3
    LOCALIZE_CAMERAS = {"image"}

    def __init__(self, *, device: torch.device):
        repo_root = Path(self.GROUNDEDSAM2_ROOT).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"Grounded-SAM-2 root does not exist: {repo_root}")

        try:
            import cv2 as _cv2
        except Exception as exc:
            raise ImportError(
                "OpenCV (`cv2`) is required for online Grounded-SAM2 masking during rollout."
            ) from exc
        self.cv2 = _cv2

        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        try:
            from grounding_dino.groundingdino.util.inference import Model as GroundingDinoModel
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise ImportError(
                "Failed to import Grounded-SAM2 dependencies. "
                "Please ensure Grounded-SAM-2 and its Python dependencies are installed."
            ) from exc

        self.device = device
        self._states: dict[str, list[_MaskTrackState]] = {}
        self._caption = self.ARM_PROMPT.lower().strip()
        if not self._caption.endswith("."):
            self._caption += "."

        # GroundingDINO accepts string device argument.
        self.grounding_model = GroundingDinoModel(
            model_config_path=str(repo_root / self.GROUNDING_DINO_CONFIG),
            model_checkpoint_path=str(repo_root / self.GROUNDING_DINO_CHECKPOINT),
            device=str(device),
        )
        # SAM2 expects config path relative to sam2 repo package.
        sam2_cfg_name = self.SAM2_CONFIG
        if sam2_cfg_name.startswith("sam2/"):
            sam2_cfg_name = sam2_cfg_name[len("sam2/") :]
        sam2_model = build_sam2(sam2_cfg_name, self.SAM2_CHECKPOINT, device=device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

    def reset(self) -> None:
        self._states = {}

    @staticmethod
    def _to_uint8_rgb_tensor(image: Tensor) -> Tensor:
        """Convert tensor image (C,H,W) to uint8 RGB tensor (C,H,W)."""
        x = image.detach().to(device="cpu", dtype=torch.float32)
        if x.ndim != 3:
            raise ValueError(f"Expected image tensor with shape (C,H,W), got {tuple(x.shape)}")
        # If normalized to [0,1], rescale to [0,255]. Otherwise clamp directly.
        scale = 255.0 if float(x.amax()) <= 1.5 else 1.0
        return (x * scale).clamp(0.0, 255.0).to(torch.uint8)

    @staticmethod
    def _tensor_rgb_to_numpy_hwc(rgb_chw_u8: Tensor) -> np.ndarray:
        return rgb_chw_u8.permute(1, 2, 0).contiguous().numpy()

    @staticmethod
    def _resize_shortest_edge(
        x: Tensor,
        *,
        target_short_edge: int,
        max_size: int,
    ) -> Tensor:
        """Resize BCHW tensor like GroundingDINO RandomResize([800], max_size=1333)."""
        _, _, h, w = x.shape
        min_orig = float(min(h, w))
        max_orig = float(max(h, w))
        size = float(target_short_edge)
        if (max_orig / min_orig) * size > float(max_size):
            size = float(round(float(max_size) * min_orig / max_orig))
        if (w <= h and int(round(w)) == int(round(size))) or (h <= w and int(round(h)) == int(round(size))):
            out_h, out_w = h, w
        elif w < h:
            out_w = int(round(size))
            out_h = int(round(size * h / w))
        else:
            out_h = int(round(size))
            out_w = int(round(size * w / h))
        if out_h == h and out_w == w:
            return x
        return F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)

    def _predict_boxes_fast_tensor(self, rgb_chw_u8: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """GroundingDINO forward without phrase extraction/token post-processing, tensor path."""
        # Match preprocess: ToTensor + RandomResize([800], max_size=1333) + Normalize
        x = rgb_chw_u8.to(device=self.grounding_model.device, dtype=torch.float32).unsqueeze(0) / 255.0
        x = self._resize_shortest_edge(x, target_short_edge=800, max_size=1333)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)[None, :, None, None]
        processed = (x - mean) / std

        with torch.no_grad():
            outputs = self.grounding_model.model(processed, captions=[self._caption])

        pred_logits = outputs["pred_logits"].sigmoid()[0]  # (num_queries, 256)
        pred_boxes = outputs["pred_boxes"][0]  # (num_queries, 4) in cxcywh normalized
        conf = pred_logits.max(dim=1)[0]
        keep = conf > float(self.BOX_THRESHOLD)
        if int(keep.sum().item()) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        boxes_norm = pred_boxes[keep]
        conf_keep = conf[keep]
        _, h, w = rgb_chw_u8.shape
        scale = torch.tensor([w, h, w, h], device=boxes_norm.device, dtype=boxes_norm.dtype)
        boxes_xyxy = box_convert(boxes=boxes_norm * scale, in_fmt="cxcywh", out_fmt="xyxy")
        return (
            boxes_xyxy.detach().to(device="cpu", dtype=torch.float32).numpy(),
            conf_keep.detach().to(device="cpu", dtype=torch.float32).numpy(),
        )

    def _predict_boxes_fast(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """GroundingDINO fast forward on legacy numpy path (fallback only)."""
        processed = self.grounding_model.preprocess_image(image_bgr=image_bgr).to(self.grounding_model.device)
        with torch.no_grad():
            outputs = self.grounding_model.model(processed[None], captions=[self._caption])

        pred_logits = outputs["pred_logits"].sigmoid()[0]
        pred_boxes = outputs["pred_boxes"][0]
        conf = pred_logits.max(dim=1)[0]
        keep = conf > float(self.BOX_THRESHOLD)
        if int(keep.sum().item()) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        boxes_norm = pred_boxes[keep]
        conf_keep = conf[keep]
        h, w = image_bgr.shape[:2]
        scale = torch.tensor([w, h, w, h], device=boxes_norm.device, dtype=boxes_norm.dtype)
        boxes_xyxy = box_convert(boxes=boxes_norm * scale, in_fmt="cxcywh", out_fmt="xyxy")
        return (
            boxes_xyxy.detach().to(device="cpu", dtype=torch.float32).numpy(),
            conf_keep.detach().to(device="cpu", dtype=torch.float32).numpy(),
        )

    def _sam2_set_image_from_tensor(self, rgb_chw_u8: Tensor) -> None:
        """Set SAM2 predictor image embedding from tensor path to avoid numpy/PIL bridge."""
        if rgb_chw_u8.ndim != 3 or rgb_chw_u8.shape[0] != 3:
            raise ValueError(f"Expected RGB tensor in CHW, got {tuple(rgb_chw_u8.shape)}")

        predictor = self.sam2_predictor
        predictor.reset_predictor()
        h, w = int(rgb_chw_u8.shape[1]), int(rgb_chw_u8.shape[2])
        predictor._orig_hw = [(h, w)]
        predictor._is_batch = False

        # SAM2 transforms: Resize(resolution,resolution) + Normalize
        image = rgb_chw_u8.to(device=predictor.device, dtype=torch.float32) / 255.0
        input_image = predictor._transforms.transforms(image).unsqueeze(0)
        backbone_out = predictor.model.forward_image(input_image)
        _, vision_feats, _, _ = predictor.model._prepare_backbone_features(backbone_out)
        if predictor.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + predictor.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], predictor._bb_feat_sizes[::-1])
        ][::-1]
        predictor._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        predictor._is_image_set = True

    @staticmethod
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

    @staticmethod
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

    def _refine_mask_components(self, mask_u8: np.ndarray, prev_mask_u8: np.ndarray | None) -> np.ndarray:
        if mask_u8.max() == 0:
            return mask_u8.astype(np.uint8)

        num, labels, stats, _ = self.cv2.connectedComponentsWithStats(mask_u8.astype(np.uint8), connectivity=8)
        if num <= 2:
            refined = mask_u8.astype(np.uint8)
        else:
            h, w = mask_u8.shape
            margin = max(1, int(self.MASK_BORDER_MARGIN))
            border = np.zeros((h, w), dtype=np.bool_)
            border[:margin, :] = True
            border[-margin:, :] = True
            border[:, :margin] = True
            border[:, -margin:] = True
            prev = prev_mask_u8.astype(bool) if prev_mask_u8 is not None else None

            candidates: list[tuple[float, int]] = []
            for cid in range(1, num):
                area = int(stats[cid, self.cv2.CC_STAT_AREA])
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
                keep_n = max(1, min(int(self.MASK_KEEP_COMPONENTS), len(candidates)))
                refined = np.zeros_like(mask_u8, dtype=np.uint8)
                for _, cid in candidates[:keep_n]:
                    refined[labels == cid] = 1

        if self.MASK_CLOSE_KERNEL > 1:
            k = np.ones((self.MASK_CLOSE_KERNEL, self.MASK_CLOSE_KERNEL), dtype=np.uint8)
            refined = self.cv2.morphologyEx(refined, self.cv2.MORPH_CLOSE, k, iterations=1)
        return refined.astype(np.uint8)

    @classmethod
    def _is_localized_camera(cls, feature_name: str) -> bool:
        return any(alias in cls.LOCALIZE_CAMERAS for alias in PrecomputedOpticalFlowReader._camera_aliases(feature_name))

    def _ensure_states(self, feature_name: str, batch_size: int) -> list[_MaskTrackState]:
        states = self._states.get(feature_name)
        if states is None or len(states) != batch_size:
            states = [_MaskTrackState() for _ in range(batch_size)]
            self._states[feature_name] = states
        return states

    def infer_mask(self, feature_name: str, images: Tensor) -> Tensor:
        """Return rollout mask (B,H,W) in float32 for the provided camera feature."""
        if images.ndim != 4:
            raise ValueError(f"Expected image tensor with shape (B,C,H,W), got {tuple(images.shape)}")

        bsz, _, h, w = images.shape
        if not self._is_localized_camera(feature_name):
            return torch.ones((bsz, h, w), device=images.device, dtype=torch.float32)

        masks = torch.zeros((bsz, h, w), device=images.device, dtype=torch.float32)
        states = self._ensure_states(feature_name, bsz)

        for bi in range(bsz):
            state = states[bi]
            rgb_u8 = self._to_uint8_rgb_tensor(images[bi])
            _, img_h, img_w = rgb_u8.shape
            # Force the legacy numpy/cv2 bridge path for A/B timing comparison.
            # This intentionally disables the new tensor-direct fast path.
            rgb = self._tensor_rgb_to_numpy_hwc(rgb_u8)
            bgr = rgb[..., ::-1]
            det_boxes, det_conf = self._predict_boxes_fast(image_bgr=bgr)
            self.sam2_predictor.set_image(rgb)

            filtered_boxes: list[np.ndarray] = []
            filtered_conf: list[float] = []
            if det_boxes is not None and len(det_boxes) > 0:
                for k, box in enumerate(np.asarray(det_boxes, dtype=np.float32)):
                    box = self._clip_box_xyxy(box, img_h, img_w)
                    bw = max(0.0, float(box[2] - box[0]))
                    bh = max(0.0, float(box[3] - box[1]))
                    area_ratio = (bw * bh) / float(img_h * img_w)
                    if area_ratio < self.BOX_AREA_MIN_RATIO or area_ratio > self.BOX_AREA_MAX_RATIO:
                        continue
                    filtered_boxes.append(box)
                    conf_k = float(det_conf[k]) if det_conf is not None else 0.0
                    filtered_conf.append(conf_k)

            chosen_box: np.ndarray | None = None
            used_hold = False
            if filtered_boxes:
                if state.prev_box_xyxy is not None:
                    ious = [self._box_iou_xyxy(box, state.prev_box_xyxy) for box in filtered_boxes]
                    best_iou_idx = int(np.argmax(np.asarray(ious)))
                    if ious[best_iou_idx] >= self.TEMPORAL_IOU_THRESHOLD:
                        chosen_box = filtered_boxes[best_iou_idx]
                    else:
                        best_conf_idx = int(np.argmax(np.asarray(filtered_conf)))
                        chosen_box = filtered_boxes[best_conf_idx]
                else:
                    best_conf_idx = int(np.argmax(np.asarray(filtered_conf)))
                    chosen_box = filtered_boxes[best_conf_idx]
            elif state.prev_box_xyxy is not None and state.miss_count < self.TEMPORAL_HOLD_FRAMES:
                chosen_box = state.prev_box_xyxy.copy()
                used_hold = True
                state.miss_count += 1
            else:
                state.prev_mask = None
                state.miss_count = min(state.miss_count + 1, self.TEMPORAL_HOLD_FRAMES + 1)
                continue

            pred_masks, _, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.asarray(chosen_box, dtype=np.float32)[None, :],
                multimask_output=False,
            )
            if pred_masks.ndim == 4:
                pred_masks = np.squeeze(pred_masks, axis=1)
            if pred_masks.ndim != 3:
                raise ValueError(f"Unexpected SAM2 mask shape: {pred_masks.shape}")

            union_mask = np.any(pred_masks, axis=0).astype(np.uint8)
            union_mask = self._refine_mask_components(union_mask, state.prev_mask)
            if state.prev_mask is not None and not used_hold:
                prev = state.prev_mask.astype(bool)
                curr = union_mask.astype(bool)
                inter = np.logical_and(prev, curr).sum()
                union = np.logical_or(prev, curr).sum()
                if union > 0 and (inter / union) < 0.01 and state.miss_count < self.TEMPORAL_HOLD_FRAMES:
                    union_mask = state.prev_mask.copy()
                    used_hold = True
                    state.miss_count += 1

            if not used_hold:
                if state.prev_box_xyxy is None:
                    state.prev_box_xyxy = chosen_box.copy()
                else:
                    alpha = float(self.TEMPORAL_BOX_EMA)
                    state.prev_box_xyxy = alpha * chosen_box + (1.0 - alpha) * state.prev_box_xyxy
                state.prev_box_xyxy = self._clip_box_xyxy(state.prev_box_xyxy, img_h, img_w)
                state.miss_count = 0
            state.prev_mask = union_mask.copy()

            masks[bi] = torch.from_numpy(union_mask.astype(np.float32)).to(device=images.device)

        return masks


class DiffusionPolicy(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://huggingface.co/papers/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = DiffusionConfig
    name = "diffusion"

    def __init__(
        self,
        config: DiffusionConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.diffusion = DiffusionModel(config)
        self._online_gmflow_enabled = (
            self.config.enable_online_gmflow_rollout
            and self.config.enable_optical_flow_condition
            and len(self.config.image_features) > 0
        )
        self._online_groundedsam2_mask_enabled = (
            self._online_gmflow_enabled and self.config.enable_online_groundedsam2_mask_rollout
        )
        self.online_gmflow_runner = None
        self.online_groundedsam2_mask_runner = None
        self._gmflow_prev_images: dict[str, Tensor] = {}
        self._gmflow_flow_keys: dict[str, str] = {}
        self._online_kalman_stateful_enabled = (
            self.config.enable_online_kalman_rollout_stateful and self.config.enable_kalman_condition
        )
        self._online_kalman_states: list[_OnlineKalmanRuntimeState] = []

        if self._online_gmflow_enabled:
            policy_device = get_device_from_parameters(self.diffusion)
            self.online_gmflow_runner = OnlineGMFlowRunner(
                repo_path=self.config.online_gmflow_repo_path,
                checkpoint_path=self.config.online_gmflow_checkpoint,
                device=policy_device,
                use_amp=self.config.online_gmflow_use_amp,
                padding_factor=self.config.online_gmflow_padding_factor,
                attn_splits_list=self.config.online_gmflow_attn_splits_list,
                corr_radius_list=self.config.online_gmflow_corr_radius_list,
                prop_radius_list=self.config.online_gmflow_prop_radius_list,
            )
            # Use canonical alias so model-side precomputed-flow resolver picks up the right camera.
            self._gmflow_flow_keys = {
                feat: f"{PRECOMPUTED_FLOW_PREFIX}{PrecomputedOpticalFlowReader._camera_aliases(feat)[0]}"
                for feat in self.config.image_features
            }
            if self._online_groundedsam2_mask_enabled:
                self.online_groundedsam2_mask_runner = OnlineGroundedSAM2MaskRunner(device=policy_device)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)
        if self._online_gmflow_enabled:
            for flow_key in self._gmflow_flow_keys.values():
                self._queues[flow_key] = deque(maxlen=self.config.n_obs_steps)
            self._gmflow_prev_images = {}
        if self._online_kalman_stateful_enabled:
            self._queues[f"{PRECOMPUTED_KALMAN_PREFIX}pos"] = deque(maxlen=self.config.n_obs_steps)
            self._queues[f"{PRECOMPUTED_KALMAN_PREFIX}vel"] = deque(maxlen=self.config.n_obs_steps)
            self._queues[f"{PRECOMPUTED_KALMAN_PREFIX}pred_exec"] = deque(maxlen=self.config.n_obs_steps)
            self._queues[f"{PRECOMPUTED_KALMAN_PREFIX}valid"] = deque(maxlen=self.config.n_obs_steps)
            self._online_kalman_states = []
        if self.online_groundedsam2_mask_runner is not None:
            self.online_groundedsam2_mask_runner.reset()

    @torch.no_grad()
    def _attach_online_gmflow_to_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Estimate per-camera GMFlow online and attach as precomputed-flow batch keys."""
        if not self._online_gmflow_enabled:
            return batch

        if self.online_gmflow_runner is None:
            raise RuntimeError("Online GMFlow is enabled but the GMFlow runner is not initialized.")

        out = dict(batch)
        for feat in self.config.image_features:
            if feat not in out:
                continue

            curr = out[feat]
            if curr.ndim == 3:
                curr = curr.unsqueeze(0)
            if curr.ndim != 4:
                raise ValueError(f"Expected 4D image tensor for '{feat}', got shape {tuple(curr.shape)}")

            prev = self._gmflow_prev_images.get(feat)
            if prev is None or tuple(prev.shape) != tuple(curr.shape):
                # No previous frame yet (e.g. first step after reset): define zero flow.
                flow = torch.zeros(
                    (curr.shape[0], 2, curr.shape[-2], curr.shape[-1]), device=curr.device, dtype=curr.dtype
                )
            else:
                flow = self.online_gmflow_runner.infer(prev, curr).to(device=curr.device, dtype=curr.dtype)
                if self.online_groundedsam2_mask_runner is not None:
                    # Keep parity with local-flow precompute: only preserve task-relevant region for selected cameras.
                    local_mask = self.online_groundedsam2_mask_runner.infer_mask(feat, curr)
                    flow = flow * local_mask.unsqueeze(1).to(dtype=flow.dtype)

            out[self._gmflow_flow_keys[feat]] = flow
            # Cache the current frame for next-step flow estimation.
            self._gmflow_prev_images[feat] = curr.detach().to(
                device=self.online_gmflow_runner.device, dtype=torch.float32, non_blocking=True
            )

        return out

    def _ensure_online_kalman_states(self, batch_size: int) -> list[_OnlineKalmanRuntimeState]:
        if len(self._online_kalman_states) != batch_size:
            self._online_kalman_states = [_OnlineKalmanRuntimeState() for _ in range(batch_size)]
        return self._online_kalman_states

    def _attach_online_kalman_to_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Attach rollout-time Kalman features with persistent state across environment steps."""
        if not self._online_kalman_stateful_enabled:
            return batch
        if OBS_STATE not in batch:
            return batch

        out = dict(batch)
        state_obs = out[OBS_STATE]
        if state_obs.ndim == 1:
            state_obs = state_obs.unsqueeze(0)
            out[OBS_STATE] = state_obs
        if state_obs.ndim != 2:
            raise ValueError(
                f"Expected observation.state to be rank-2 (B,D) during rollout, got {tuple(state_obs.shape)}"
            )

        # Prefer explicitly provided raw (unnormalized) state when available.
        kalman_state_obs = out.get(OBS_STATE_RAW, state_obs)
        if kalman_state_obs.ndim == 1:
            kalman_state_obs = kalman_state_obs.unsqueeze(0)
            out[OBS_STATE_RAW] = kalman_state_obs
        if kalman_state_obs.ndim != 2:
            raise ValueError(
                "Expected observation.state.raw to be rank-2 (B,D) during rollout, "
                f"got {tuple(kalman_state_obs.shape)}"
            )
        if kalman_state_obs.shape[0] != state_obs.shape[0]:
            raise ValueError(
                "Mismatched batch size between observation.state and observation.state.raw: "
                f"{state_obs.shape[0]} vs {kalman_state_obs.shape[0]}"
            )

        z = kalman_state_obs[..., self.diffusion._kalman_pos_slice]
        if z.shape[-1] != 3:
            raise ValueError(
                f"Kalman state position slice {self.config.kalman_state_pos_slice} produced shape {tuple(z.shape)}."
            )
        b = z.shape[0]
        states = self._ensure_online_kalman_states(b)

        ts_list: list[float | None] | None = None
        if "timestamp" in out:
            ts = out["timestamp"]
            if ts.ndim == 0:
                ts = ts.unsqueeze(0)
            if ts.ndim >= 2:
                ts = ts.reshape(ts.shape[0], -1)[:, 0]
            ts_list = ts.detach().to(device="cpu", dtype=torch.float64).tolist()

        device = kalman_state_obs.device
        dtype = kalman_state_obs.dtype
        pos = torch.zeros((b, 3), device=device, dtype=dtype)
        vel = torch.zeros((b, 3), device=device, dtype=dtype)
        pred_exec = torch.zeros((b, 3), device=device, dtype=dtype)
        valid = torch.ones((b,), device=device, dtype=dtype)

        p_pos = float(self.config.kalman_init_pos_std**2)
        p_vel = float(self.config.kalman_init_vel_std**2)
        r = torch.eye(3, device=device, dtype=dtype) * float(self.config.kalman_meas_noise_std**2)
        h = torch.zeros((3, 6), device=device, dtype=dtype)
        h[:, :3] = torch.eye(3, device=device, dtype=dtype)
        i6 = torch.eye(6, device=device, dtype=dtype)

        def _make_f_q(dt_value: float) -> tuple[Tensor, Tensor]:
            dt = torch.tensor(float(dt_value), device=device, dtype=dtype)
            dt2 = dt.square()
            dt3 = dt2 * dt
            dt4 = dt2.square()
            f = torch.eye(6, device=device, dtype=dtype)
            f[:3, 3:] = torch.eye(3, device=device, dtype=dtype) * dt
            sigma2 = float(self.config.kalman_accel_noise_std**2)
            q = torch.zeros((6, 6), device=device, dtype=dtype)
            q_pos = (dt4 / 4.0) * sigma2
            q_cross = (dt3 / 2.0) * sigma2
            q_vel = dt2 * sigma2
            for axis in range(3):
                q[axis, axis] = q_pos
                q[axis, axis + 3] = q_cross
                q[axis + 3, axis] = q_cross
                q[axis + 3, axis + 3] = q_vel
            return f, q

        for bi in range(b):
            runtime = states[bi]
            meas = z[bi]
            meas_valid = bool(torch.isfinite(meas).all())
            if runtime.x is None or runtime.p is None:
                x = torch.zeros((6,), device=device, dtype=dtype)
                if meas_valid:
                    x[:3] = torch.nan_to_num(meas, nan=0.0, posinf=0.0, neginf=0.0)
                p = torch.zeros((6, 6), device=device, dtype=dtype)
                for axis in range(3):
                    p[axis, axis] = p_pos
                    p[axis + 3, axis + 3] = p_vel
            else:
                x = runtime.x.to(device=device, dtype=dtype)
                p = runtime.p.to(device=device, dtype=dtype)

            # After the first step, always run predict. Use timestamps when available, otherwise fallback dt.
            if runtime.step_count > 0:
                dt = float(self.config.kalman_dt_fallback)
                if ts_list is not None and bi < len(ts_list):
                    ts_val = ts_list[bi]
                    if ts_val is not None and np.isfinite(ts_val) and runtime.prev_timestamp is not None:
                        dt_measured = float(ts_val - runtime.prev_timestamp)
                        # Keep parity with sidecar precompute script: clamp invalid/huge dt to fallback.
                        if np.isfinite(dt_measured) and 0.0 < dt_measured <= 1.0:
                            dt = dt_measured
                f, q = _make_f_q(dt)
                x = f @ x
                p = f @ p @ f.transpose(0, 1) + q

            if meas_valid:
                y = torch.nan_to_num(meas, nan=0.0, posinf=0.0, neginf=0.0) - (h @ x)
                s_mat = h @ p @ h.transpose(0, 1) + r
                k_gain = p @ h.transpose(0, 1) @ torch.linalg.inv(s_mat)
                x = x + (k_gain @ y)
                p = (i6 - (k_gain @ h)) @ p
                valid[bi] = 1.0
            else:
                valid[bi] = 0.0

            f_exec, _ = _make_f_q(float(self.config.kalman_predict_horizon))
            x_exec = f_exec @ x
            pos[bi] = x[:3]
            vel[bi] = x[3:]
            pred_exec[bi] = x_exec[:3]

            runtime.x = x.detach()
            runtime.p = p.detach()
            runtime.step_count += 1
            if ts_list is not None and bi < len(ts_list):
                ts_val = ts_list[bi]
                runtime.prev_timestamp = float(ts_val) if ts_val is not None and np.isfinite(ts_val) else None

        out[f"{PRECOMPUTED_KALMAN_PREFIX}pos"] = pos
        out[f"{PRECOMPUTED_KALMAN_PREFIX}vel"] = vel
        out[f"{PRECOMPUTED_KALMAN_PREFIX}pred_exec"] = pred_exec
        out[f"{PRECOMPUTED_KALMAN_PREFIX}valid"] = valid
        return out

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        # stack n latest observations from the queue
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.diffusion.generate_actions(batch, noise=noise)

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        if self._online_gmflow_enabled:
            # Rollout-only path: compute GMFlow from consecutive observations and inject as precomputed flow.
            batch = self._attach_online_gmflow_to_batch(batch)
        if self._online_kalman_stateful_enabled:
            # Rollout-only path: keep a persistent Kalman state and inject precomputed_kalman_* keys.
            batch = self._attach_online_kalman_to_batch(batch)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.diffusion.compute_loss(batch)
        # no output_dict so returning None
        return loss, None


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self._kalman_pos_slice = self._parse_kalman_pos_slice(self.config.kalman_state_pos_slice)

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        elif self.config.enable_optical_flow_condition:
            raise ValueError("`enable_optical_flow_condition=True` requires at least one image observation.")

        if self.config.enable_optical_flow_condition and self.config.image_features:
            # Experimental path: use a lightweight hand-crafted optical-flow encoder and concatenate
            # the resulting feature to the denoiser global condition.
            self.optical_flow_encoder = DiffusionOpticalFlowEncoder(config)
            self.flow_feature_dim = self.optical_flow_encoder.feature_dim * len(self.config.image_features)
            # Normalize flow features before fusion to keep feature scales comparable with other conditions.
            self.optical_flow_feature_norm = nn.LayerNorm(self.flow_feature_dim)
            # Learnable scalar gate for soft enabling of the flow branch.
            # We optimize an unconstrained logit and map with sigmoid to [0, 1].
            gate_init = torch.tensor(self.config.optical_flow_gate_init, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
            self.optical_flow_gate_logit = nn.Parameter(torch.log(gate_init / (1 - gate_init)))
            global_cond_dim += self.flow_feature_dim
            flow_kernel_x, flow_kernel_y = _make_sobel_kernels(self.config.optical_flow_kernel_size)
            self.register_buffer("flow_kernel_x", flow_kernel_x, persistent=False)
            self.register_buffer("flow_kernel_y", flow_kernel_y, persistent=False)
            if self.config.precomputed_optical_flow_root:
                self.precomputed_optical_flow_reader = PrecomputedOpticalFlowReader(
                    root=self.config.precomputed_optical_flow_root,
                    image_features=list(self.config.image_features.keys()),
                    n_obs_steps=self.config.n_obs_steps,
                    observation_delta_indices=self.config.observation_delta_indices,
                    cache_size=self.config.precomputed_optical_flow_cache_size,
                )
            else:
                self.precomputed_optical_flow_reader = None
            self._warned_precomputed_flow_fallback = False
        else:
            self.optical_flow_encoder = None
            self.flow_feature_dim = 0
            self.optical_flow_feature_norm = None
            self.optical_flow_gate_logit = None
            self.precomputed_optical_flow_reader = None
            self._warned_precomputed_flow_fallback = False
            self.register_buffer("flow_kernel_x", None, persistent=False)
            self.register_buffer("flow_kernel_y", None, persistent=False)

        if self.config.enable_kalman_condition:
            self.kalman_raw_dim = 10 if self.config.kalman_feature_mode == "full10" else 6
            self.kalman_feature_norm = nn.LayerNorm(self.kalman_raw_dim)
            self.kalman_projector = nn.Sequential(
                nn.Linear(self.kalman_raw_dim, self.config.kalman_feature_dim),
                nn.SiLU(),
                nn.Linear(self.config.kalman_feature_dim, self.config.kalman_feature_dim),
            )
            kalman_mean, kalman_std = self._load_kalman_stats_or_default()
            self.register_buffer("kalman_stats_mean", kalman_mean, persistent=False)
            self.register_buffer("kalman_stats_std", kalman_std, persistent=False)
            global_cond_dim += self.config.kalman_feature_dim
            if self.config.precomputed_kalman_root:
                self.precomputed_kalman_reader = PrecomputedKalmanReader(
                    root=self.config.precomputed_kalman_root,
                    n_obs_steps=self.config.n_obs_steps,
                    observation_delta_indices=self.config.observation_delta_indices,
                    cache_size=self.config.precomputed_kalman_cache_size,
                    feature_mode=self.config.kalman_feature_mode,
                )
            else:
                self.precomputed_kalman_reader = None
            self._warned_precomputed_kalman_fallback = False
        else:
            self.kalman_raw_dim = 0
            self.kalman_feature_norm = None
            self.kalman_projector = None
            self.register_buffer("kalman_stats_mean", None, persistent=False)
            self.register_buffer("kalman_stats_std", None, persistent=False)
            self.precomputed_kalman_reader = None
            self._warned_precomputed_kalman_fallback = False

        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        if config.compile_model:
            # Compile the U-Net. "reduce-overhead" is preferred for the small-batch repetitive loops
            # common in diffusion inference.
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    @staticmethod
    def _parse_kalman_pos_slice(spec: str) -> slice:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid kalman position slice: {spec}")
        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if parts[1] else None
        return slice(start, end)

    def _load_kalman_stats_or_default(self) -> tuple[Tensor, Tensor]:
        mean = torch.zeros((self.kalman_raw_dim,), dtype=torch.float32)
        std = torch.ones((self.kalman_raw_dim,), dtype=torch.float32)
        if not self.config.kalman_use_dataset_stats_norm:
            return mean, std

        candidates: list[Path] = []
        if self.config.kalman_stats_path:
            candidates.append(Path(self.config.kalman_stats_path))
        if self.config.precomputed_kalman_root:
            candidates.append(Path(self.config.precomputed_kalman_root) / "normalization.json")

        for path in candidates:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if "mean" not in payload or "std" not in payload:
                raise ValueError(f"Invalid kalman stats file (missing mean/std): {path}")
            mean_np = np.asarray(payload["mean"], dtype=np.float32)
            std_np = np.asarray(payload["std"], dtype=np.float32)
            if mean_np.shape != (self.kalman_raw_dim,) or std_np.shape != (self.kalman_raw_dim,):
                raise ValueError(
                    f"Kalman stats must be shape ({self.kalman_raw_dim},). "
                    f"Got mean={mean_np.shape}, std={std_np.shape} from {path}"
                )
            std_np = np.maximum(std_np, self.config.kalman_norm_eps).astype(np.float32)
            logging.info(f"Loaded Kalman normalization stats from {path}")
            return torch.from_numpy(mean_np), torch.from_numpy(std_np)

        logging.warning(
            "Kalman dataset-stats normalization is enabled but no stats file was found. "
            "Using identity normalization. Set `policy.kalman_stats_path` or place "
            "`normalization.json` under `precomputed_kalman_root`."
        )
        return mean, std

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        # Extract image features.
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

            if self.optical_flow_encoder is not None:
                # Convert stacked frames into a simple hand-crafted optical-flow estimate, then encode
                # flow maps into compact per-step features before concatenation.
                flow_maps = self._get_optical_flow_maps(batch)
                flow_features = self.optical_flow_encoder(
                    einops.rearrange(flow_maps, "b s n c h w -> (b s n) c h w")
                )
                flow_features = einops.rearrange(
                    flow_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
                # Stabilize flow feature statistics and modulate strength with a learnable gate.
                flow_features = self.optical_flow_feature_norm(flow_features)
                flow_gate = torch.sigmoid(self.optical_flow_gate_logit).to(flow_features.dtype)
                flow_features = flow_gate * flow_features
                # Branch dropout: randomly remove all flow conditioning for a sample during training.
                # This prevents the model from over-relying on potentially noisy hand-crafted flow.
                if self.training and self.config.optical_flow_dropout_p > 0:
                    keep_prob = 1.0 - self.config.optical_flow_dropout_p
                    keep_mask = (
                        torch.rand((batch_size, 1, 1), device=flow_features.device, dtype=flow_features.dtype)
                        < keep_prob
                    )
                    flow_features = flow_features * keep_mask / keep_prob
                global_cond_feats.append(flow_features)

        if self.config.enable_kalman_condition and self.kalman_projector is not None:
            kalman_raw = self._get_kalman_features(batch)
            if self.config.kalman_use_dataset_stats_norm:
                kalman_raw = (kalman_raw - self.kalman_stats_mean.to(kalman_raw)) / self.kalman_stats_std.to(
                    kalman_raw
                )
            kalman_raw = self.kalman_feature_norm(kalman_raw)
            kalman_features = self.kalman_projector(kalman_raw)
            global_cond_feats.append(kalman_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def _get_kalman_features(self, batch: dict[str, Tensor]) -> Tensor:
        state_obs = batch[OBS_STATE]
        from_batch = self._get_precomputed_kalman_features_from_batch(batch, ref_state=state_obs)
        if from_batch is not None:
            return from_batch

        if self.precomputed_kalman_reader is not None:
            try:
                return self.precomputed_kalman_reader.get_kalman_features(batch, ref_state=state_obs)
            except KeyError:
                if not self._warned_precomputed_kalman_fallback:
                    logging.warning(
                        "Precomputed Kalman root is set but batch lacks dataset index metadata. "
                        "Falling back to online Kalman computation from observation.state for this run."
                    )
                    self._warned_precomputed_kalman_fallback = True
            except FileNotFoundError:
                raise
            except Exception:
                raise

        return self._compute_online_kalman_from_state(batch)

    def _get_optical_flow_maps(self, batch: dict[str, Tensor]) -> Tensor:
        """Get optical flow from precomputed cache when configured, otherwise compute online."""
        images = batch[OBS_IMAGES]
        flow_from_batch = self._get_precomputed_flow_maps_from_batch(batch, ref_images=images)
        if flow_from_batch is not None:
            return flow_from_batch

        if self.precomputed_optical_flow_reader is not None:
            try:
                return self.precomputed_optical_flow_reader.get_flow_maps(batch, ref_images=images)
            except KeyError:
                # Eval rollouts typically don't carry dataset index metadata; keep compatibility by fallback.
                if not self._warned_precomputed_flow_fallback:
                    logging.warning(
                        "Precomputed flow root is set but batch lacks dataset index metadata. "
                        "Falling back to online optical-flow approximation for this run."
                    )
                    self._warned_precomputed_flow_fallback = True
            except FileNotFoundError:
                # User asked for precomputed flow; missing files should fail loudly for reproducibility.
                raise
            except Exception:
                # Any data mismatch is likely a preprocessing bug and should not be silently ignored.
                raise
        return self._compute_approx_optical_flow(images)

    def _get_precomputed_flow_maps_from_batch(self, batch: dict[str, Tensor], ref_images: Tensor) -> Tensor | None:
        """Build flow maps from tensors already attached by dataset workers."""
        available_keys = {k for k in batch if k.startswith(PRECOMPUTED_FLOW_PREFIX)}
        if not available_keys:
            return None

        b, s, n, _, h, w = ref_images.shape
        out = torch.empty((b, s, n, 2, h, w), device=ref_images.device, dtype=ref_images.dtype)

        for ni, feat in enumerate(self.config.image_features):
            flow_key = None
            for alias in PrecomputedOpticalFlowReader._camera_aliases(feat):
                candidate = f"{PRECOMPUTED_FLOW_PREFIX}{alias}"
                if candidate in available_keys:
                    flow_key = candidate
                    break
            if flow_key is None:
                return None

            flow = batch[flow_key]
            if flow.ndim == 4:
                flow = flow.unsqueeze(1)
            if flow.shape[0] != b or flow.shape[1] != s or flow.shape[2] != 2:
                raise ValueError(
                    f"Unexpected precomputed flow tensor shape for key '{flow_key}': "
                    f"got={tuple(flow.shape)}, expected=(B={b}, S={s}, 2, H, W)"
                )
            out[:, :, ni] = flow.to(device=ref_images.device, dtype=ref_images.dtype)

        return out

    def _get_precomputed_kalman_features_from_batch(
        self, batch: dict[str, Tensor], ref_state: Tensor
    ) -> Tensor | None:
        required = [f"{PRECOMPUTED_KALMAN_PREFIX}pos", f"{PRECOMPUTED_KALMAN_PREFIX}vel"]
        if self.config.kalman_feature_mode == "full10":
            required.append(f"{PRECOMPUTED_KALMAN_PREFIX}pred_exec")
        if any(key not in batch for key in required):
            return None

        b, s = ref_state.shape[:2]

        def _ensure_bsd(x: Tensor, d_expected: int) -> Tensor:
            if x.ndim == 2:
                x = x.unsqueeze(1)
            if x.ndim != 3:
                raise ValueError(f"Unexpected Kalman tensor rank: got={tuple(x.shape)}, expected 2D or 3D")
            if x.shape[0] != b:
                raise ValueError(f"Unexpected Kalman batch size: got={x.shape[0]}, expected={b}")
            if x.shape[1] != s:
                raise ValueError(f"Unexpected Kalman sequence length: got={x.shape[1]}, expected={s}")
            if x.shape[2] != d_expected:
                raise ValueError(f"Unexpected Kalman feature dim: got={x.shape[2]}, expected={d_expected}")
            return x.to(device=ref_state.device, dtype=ref_state.dtype)

        pos = _ensure_bsd(batch[f"{PRECOMPUTED_KALMAN_PREFIX}pos"], 3)
        vel = _ensure_bsd(batch[f"{PRECOMPUTED_KALMAN_PREFIX}vel"], 3)
        if self.config.kalman_feature_mode != "full10":
            return torch.cat([pos, vel], dim=-1)
        pred = _ensure_bsd(batch[f"{PRECOMPUTED_KALMAN_PREFIX}pred_exec"], 3)

        valid_key = f"{PRECOMPUTED_KALMAN_PREFIX}valid"
        if valid_key in batch:
            valid = batch[valid_key]
            if valid.ndim == 1:
                valid = valid.unsqueeze(1)
            if valid.ndim != 2:
                raise ValueError(f"Unexpected Kalman valid tensor rank: got={tuple(valid.shape)}, expected 1D or 2D")
            if valid.shape[0] != b or valid.shape[1] != s:
                raise ValueError(
                    f"Unexpected Kalman valid shape: got={tuple(valid.shape)}, expected=(B={b}, S={s})"
                )
            valid = valid.to(device=ref_state.device, dtype=ref_state.dtype).unsqueeze(-1)
        else:
            valid = torch.ones((b, s, 1), device=ref_state.device, dtype=ref_state.dtype)

        return torch.cat([pos, vel, pred, valid], dim=-1)

    def _compute_online_kalman_from_state(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Compute Kalman features from observation.state on the fly.
        Returns (B, S, D):
          - full10: [pos(3), vel(3), pred_exec(3), valid(1)]
          - posvel6: [pos(3), vel(3)]
        """
        state_obs = batch[OBS_STATE]
        b, s, d = state_obs.shape
        z = state_obs[..., self._kalman_pos_slice]
        if z.shape[-1] != 3:
            raise ValueError(
                f"Kalman state position slice {self.config.kalman_state_pos_slice} produced shape {tuple(z.shape)} "
                f"from observation.state shape {tuple(state_obs.shape)}."
            )
        device = state_obs.device
        dtype = state_obs.dtype

        valid = torch.isfinite(z).all(dim=-1)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        dt = torch.full((b, s), float(self.config.kalman_dt_fallback), device=device, dtype=dtype)
        if "timestamp" in batch:
            ts = batch["timestamp"]
            if ts.ndim == 1:
                ts = ts.unsqueeze(1)
            if ts.ndim >= 2 and ts.shape[0] == b and ts.shape[1] == s:
                dt[:, 1:] = (ts[:, 1:] - ts[:, :-1]).to(dtype=dtype)
                dt = torch.where(
                    (dt > 0) & torch.isfinite(dt),
                    dt,
                    torch.full_like(dt, float(self.config.kalman_dt_fallback)),
                )

        def _make_f_q(step_dt: Tensor) -> tuple[Tensor, Tensor]:
            step_dt = step_dt.to(dtype=dtype)
            dt2 = step_dt.square()
            dt3 = dt2 * step_dt
            dt4 = dt2.square()

            f = torch.eye(6, device=device, dtype=dtype).unsqueeze(0).repeat(b, 1, 1)
            f[:, :3, 3:] = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) * step_dt.view(b, 1, 1)

            sigma2 = float(self.config.kalman_accel_noise_std**2)
            q = torch.zeros((b, 6, 6), device=device, dtype=dtype)
            q_pos = (dt4 / 4.0) * sigma2
            q_cross = (dt3 / 2.0) * sigma2
            q_vel = dt2 * sigma2
            for axis in range(3):
                q[:, axis, axis] = q_pos
                q[:, axis, axis + 3] = q_cross
                q[:, axis + 3, axis] = q_cross
                q[:, axis + 3, axis + 3] = q_vel
            return f, q

        x = torch.zeros((b, 6), device=device, dtype=dtype)
        x[:, :3] = z[:, 0]
        p = torch.zeros((b, 6, 6), device=device, dtype=dtype)
        p_pos = float(self.config.kalman_init_pos_std**2)
        p_vel = float(self.config.kalman_init_vel_std**2)
        for axis in range(3):
            p[:, axis, axis] = p_pos
            p[:, axis + 3, axis + 3] = p_vel

        h = torch.zeros((1, 3, 6), device=device, dtype=dtype).repeat(b, 1, 1)
        h[:, :, :3] = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        r = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(b, 1, 1) * float(
            self.config.kalman_meas_noise_std**2
        )
        i6 = torch.eye(6, device=device, dtype=dtype).unsqueeze(0).repeat(b, 1, 1)

        if self.config.kalman_feature_mode == "full10":
            out = torch.zeros((b, s, 10), device=device, dtype=dtype)
        else:
            out = torch.zeros((b, s, 6), device=device, dtype=dtype)

        for si in range(s):
            if si > 0:
                f, q = _make_f_q(dt[:, si])
                x = torch.bmm(f, x.unsqueeze(-1)).squeeze(-1)
                p = torch.bmm(torch.bmm(f, p), f.transpose(1, 2)) + q

            z_si = z[:, si]
            valid_si = valid[:, si]
            if bool(valid_si.any()):
                y = z_si - torch.bmm(h, x.unsqueeze(-1)).squeeze(-1)
                s_mat = torch.bmm(torch.bmm(h, p), h.transpose(1, 2)) + r
                k_gain = torch.bmm(torch.bmm(p, h.transpose(1, 2)), torch.linalg.inv(s_mat))
                x_upd = x + torch.bmm(k_gain, y.unsqueeze(-1)).squeeze(-1)
                p_upd = torch.bmm(i6 - torch.bmm(k_gain, h), p)
                mask = valid_si.unsqueeze(-1)
                x = torch.where(mask, x_upd, x)
                p = torch.where(mask.unsqueeze(-1), p_upd, p)

            out[:, si, :3] = x[:, :3]
            out[:, si, 3:6] = x[:, 3:]
            if self.config.kalman_feature_mode == "full10":
                f_exec, _ = _make_f_q(
                    torch.full((b,), float(self.config.kalman_predict_horizon), device=device, dtype=dtype)
                )
                x_exec = torch.bmm(f_exec, x.unsqueeze(-1)).squeeze(-1)
                out[:, si, 6:9] = x_exec[:, :3]
                out[:, si, 9] = valid_si.to(dtype=dtype)

        return out

    def _compute_approx_optical_flow(self, images: Tensor) -> Tensor:
        """Compute a lightweight, hand-crafted optical-flow approximation from consecutive RGB frames.

        This is intentionally simple for idea validation: we use grayscale brightness constancy
        with Sobel spatial gradients and a closed-form per-pixel update:
            u = -It * Ix / (Ix^2 + Iy^2 + eps)
            v = -It * Iy / (Ix^2 + Iy^2 + eps)
        where (u, v) is the flow vector and It/Ix/Iy are temporal/spatial gradients.
        """
        if self.flow_kernel_x is None or self.flow_kernel_y is None:
            raise RuntimeError("Optical-flow kernels are not initialized.")

        # (B, S, N, C, H, W) -> (B, S, N, H, W), grayscale for gradient computation.
        gray_images = (
            0.2989 * images[..., 0, :, :] + 0.5870 * images[..., 1, :, :] + 0.1140 * images[..., 2, :, :]
        )
        # Use previous frame for temporal derivative; first step is copied so its flow is exactly zero.
        prev_gray_images = torch.cat([gray_images[:, :1], gray_images[:, :-1]], dim=1)

        flat_gray_images = einops.rearrange(gray_images, "b s n h w -> (b s n) 1 h w")
        flat_prev_gray_images = einops.rearrange(prev_gray_images, "b s n h w -> (b s n) 1 h w")

        padding = self.flow_kernel_x.shape[-1] // 2
        grad_x = F.conv2d(flat_gray_images, self.flow_kernel_x.to(flat_gray_images), padding=padding)
        grad_y = F.conv2d(flat_gray_images, self.flow_kernel_y.to(flat_gray_images), padding=padding)
        grad_t = flat_gray_images - flat_prev_gray_images

        denom = grad_x.square() + grad_y.square() + self.config.optical_flow_eps
        flow_x = -grad_t * grad_x / denom
        flow_y = -grad_t * grad_y / denom
        flow = torch.cat([flow_x, flow_y], dim=1)

        return einops.rearrange(flow, "(b s n) c h w -> b s n c h w", b=images.shape[0], s=images.shape[1])

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        required_keys = {OBS_STATE, ACTION}
        if self.config.do_mask_loss_for_padding:
            required_keys.add("action_is_pad")
        assert set(batch).issuperset(required_keys)
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch[ACTION]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch[ACTION]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()


def _make_sobel_kernels(kernel_size: int) -> tuple[Tensor, Tensor]:
    """Build Sobel-like derivative kernels of size 3/5/7 for manual optical-flow estimation."""
    if kernel_size == 3:
        smoothing = torch.tensor([1.0, 2.0, 1.0])
        derivative = torch.tensor([-1.0, 0.0, 1.0])
    elif kernel_size == 5:
        smoothing = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        derivative = torch.tensor([-1.0, -2.0, 0.0, 2.0, 1.0])
    elif kernel_size == 7:
        smoothing = torch.tensor([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
        derivative = torch.tensor([-1.0, -4.0, -5.0, 0.0, 5.0, 4.0, 1.0])
    else:
        raise ValueError(f"Unsupported Sobel kernel size: {kernel_size}")

    kernel_x = torch.outer(smoothing, derivative)
    kernel_y = torch.outer(derivative, smoothing)
    # Keep magnitudes in a stable range.
    kernel_x = kernel_x / kernel_x.abs().sum()
    kernel_y = kernel_y / kernel_y.abs().sum()
    return kernel_x.view(1, 1, kernel_size, kernel_size), kernel_y.view(1, 1, kernel_size, kernel_size)


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://huggingface.co/papers/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.resize_shape is not None:
            self.resize = torchvision.transforms.Resize(config.resize_shape)
        else:
            self.resize = None

        crop_shape = config.crop_shape
        if crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy shape mirrors the runtime preprocessing order: resize -> crop.

        # Note: we have a check in the config class to make sure all images have the same shape.
        images_shape = next(iter(config.image_features.values())).shape
        if config.crop_shape is not None:
            dummy_shape_h_w = config.crop_shape
        elif config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        else:
            dummy_shape_h_w = images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: resize if configured, then crop if configured.

        if self.resize is not None:
            x = self.resize(x)
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


class DiffusionOpticalFlowEncoder(nn.Module):
    """Lightweight CNN encoder for hand-crafted optical-flow maps."""

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.feature_dim = config.optical_flow_feature_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, self.feature_dim)

    def forward(self, flow_map: Tensor) -> Tensor:
        """
        Args:
            flow_map: (B, 2, H, W) optical-flow tensor.
        Returns:
            (B, D) encoded flow feature.
        """
        x = self.backbone(flow_map).flatten(start_dim=1)
        return self.proj(x)


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, config: DiffusionConfig, global_cond_dim: int):
        super().__init__()

        self.config = config

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://huggingface.co/papers/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out
