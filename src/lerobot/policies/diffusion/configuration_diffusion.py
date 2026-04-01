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
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("diffusion")
@dataclass
class DiffusionConfig(PreTrainedConfig):
    """Configuration class for DiffusionPolicy.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        horizon: Diffusion model action prediction size as detailed in `DiffusionPolicy.select_action`.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            See `DiffusionPolicy.select_action` for more details.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        resize_shape: (H, W) shape to resize images to as a preprocessing step for the vision
            backbone. If None, no resizing is done and the original image resolution is used.
        crop_ratio: Ratio in (0, 1] used to derive the crop size from resize_shape
            (crop_h = int(resize_shape[0] * crop_ratio), likewise for width).
            Set to 1.0 to disable cropping. Only takes effect when resize_shape is not None.
        crop_shape: (H, W) shape to crop images to. When resize_shape is set and crop_ratio < 1.0,
            this is computed automatically. Can also be set directly for legacy configs that use
            crop-only (without resize). If None and no derivation applies, no cropping is done.
        crop_is_random: Whether the crop should be random at training time (it's always a center
            crop in eval mode).
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
            The group sizes are set to be about 16 (to be precise, feature_dim // 16).
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        use_separate_rgb_encoder_per_camera: Whether to use a separate RGB encoder for each camera view.
        enable_future_predictor: Whether to enable the lightweight future predictor branch.
        future_predictor_type: Future predictor architecture. Supported options: ["mlp", "gru"].
        future_target_type: Future supervision target. Supported options: ["future_feature", "future_delta"].
        future_training_stage: Future training stage. Supported options: ["joint", "pretrain"].
            - "joint": optimize policy loss + lambda_future * future loss.
            - "pretrain": optimize only lambda_future * future loss.
        future_num_input_steps: Number of historical encoded feature steps used by the future predictor input.
        future_predictor_hidden_dim: Hidden width used by the lightweight future predictor.
        future_condition_fusion: How to fuse future branch into policy conditioning.
            - "concat": direct concat [h_t, future_branch_output]
            - "project_concat": first project [h_t, future_branch_output], then concat projected vector
              to policy conditioning.
        future_condition_proj_dim: Output width of future projection when using "project_concat".
        lambda_future: Future loss weight for joint/pretrain objective.
        future_cosine_loss_weight: Optional cosine loss weight added to MSE future loss.
        future_freeze_encoder: Whether to freeze the image observation encoder while training future-enabled DP.
        enable_kalman_condition: Whether to append an online Kalman feature branch to global conditioning.
        kalman_feature_mode: Raw Kalman feature layout.
            - "full10": [pos(3), vel(3), pred_exec(3), valid(1)]
            - "posvel6": [pos(3), vel(3)]
            - "vel3": [vel(3)]
            - "velpred6": [vel(3), pred_exec(3)]
        enable_kalman_feature_mlp: Whether to apply an MLP to Kalman features before concatenating
            into global conditioning.
        kalman_feature_mlp_dim: Output width of the optional Kalman MLP branch. If None, keep
            the same width as raw Kalman features.
        enable_kalman_feature_gate: Whether to apply a learned gate on the final Kalman
            conditioning tensor right before concatenating into global conditioning.
        enable_kalman_mid_only_condition: Whether Kalman conditioning is injected only at the Unet
            mid block instead of all down/mid/up residual blocks.
        kalman_force_zero_global_condition: If True, force the Kalman-derived tensor appended to
            global conditioning to be exactly zero.
        kalman_state_pos_slice: Slice in observation.state used as xyz measurement for online Kalman
            computation, in "start:end" format.
        kalman_predict_horizon: Seconds to predict forward when building the execution-time position feature.
        kalman_dt_fallback: Fallback step interval in seconds used when timestamp deltas are unavailable.
        kalman_meas_noise_std: Measurement noise standard deviation for online Kalman.
        kalman_accel_noise_std: Process acceleration noise standard deviation for online Kalman.
        kalman_init_pos_std: Initial position uncertainty standard deviation for online Kalman.
        kalman_init_vel_std: Initial velocity uncertainty standard deviation for online Kalman.
        down_dims: Feature dimension for each stage of temporal downsampling in the diffusion modeling Unet.
            You may provide a variable number of dimensions, therefore also controlling the degree of
            downsampling.
        kernel_size: The convolutional kernel size of the diffusion modeling Unet.
        n_groups: Number of groups used in the group norm of the Unet's convolutional blocks.
        diffusion_step_embed_dim: The Unet is conditioned on the diffusion timestep via a small non-linear
            network. This is the output dimension of that network, i.e., the embedding dimension.
        use_film_scale_modulation: FiLM (https://huggingface.co/papers/1709.07871) is used for the Unet conditioning.
            Bias modulation is used be default, while this parameter indicates whether to also use scale
            modulation.
        noise_scheduler_type: Name of the noise scheduler to use. Supported options: ["DDPM", "DDIM"].
        num_train_timesteps: Number of diffusion steps for the forward diffusion schedule.
        beta_schedule: Name of the diffusion beta schedule as per DDPMScheduler from Hugging Face diffusers.
        beta_start: Beta value for the first forward-diffusion step.
        beta_end: Beta value for the last forward-diffusion step.
        prediction_type: The type of prediction that the diffusion modeling Unet makes. Choose from "epsilon"
            or "sample". These have equivalent outcomes from a latent variable modeling perspective, but
            "epsilon" has been shown to work better in many deep neural network settings.
        clip_sample: Whether to clip the sample to [-`clip_sample_range`, +`clip_sample_range`] for each
            denoising step at inference time. WARNING: you will need to make sure your action-space is
            normalized to fit within this range.
        clip_sample_range: The magnitude of the clipping range as described above.
        num_inference_steps: Number of reverse diffusion steps to use at inference time (steps are evenly
            spaced). If not provided, this defaults to be the same as `num_train_timesteps`.
        do_mask_loss_for_padding: Whether to mask the loss when there are copy-padded actions. See
            `LeRobotDataset` and `load_previous_and_future_frames` for more information. Note, this defaults
            to False as the original Diffusion Policy implementation does the same.
    """

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    # Experimental: lightweight future latent predictor branch.
    enable_future_predictor: bool = False
    future_predictor_type: str = "mlp"
    future_target_type: str = "future_feature"
    future_training_stage: str = "joint"
    future_num_input_steps: int = 1
    future_predictor_hidden_dim: int = 256
    future_condition_fusion: str = "concat"
    future_condition_proj_dim: int | None = None
    lambda_future: float = 1.0
    future_cosine_loss_weight: float = 0.0
    future_freeze_encoder: bool = False
    # Experimental: direct online Kalman conditioning branch.
    enable_kalman_condition: bool = False
    kalman_feature_mode: str = "full10"
    enable_kalman_feature_mlp: bool = False
    kalman_feature_mlp_dim: int | None = None
    enable_kalman_feature_gate: bool = False
    enable_kalman_mid_only_condition: bool = False
    kalman_force_zero_global_condition: bool = False
    kalman_state_pos_slice: str = "0:3"
    kalman_predict_horizon: float = 0.1
    kalman_dt_fallback: float = 0.1
    kalman_meas_noise_std: float = 0.01
    kalman_accel_noise_std: float = 0.4
    kalman_init_pos_std: float = 0.05
    kalman_init_vel_std: float = 0.5
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Optimization
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )
        if self.kalman_feature_mode not in {"full10", "posvel6", "vel3", "velpred6"}:
            raise ValueError(
                "`kalman_feature_mode` must be one of {'full10', 'posvel6', 'vel3', 'velpred6'}. "
                f"Got {self.kalman_feature_mode}."
            )
        if self.future_predictor_type not in {"mlp", "gru"}:
            raise ValueError(
                "`future_predictor_type` must be one of {'mlp', 'gru'}. "
                f"Got {self.future_predictor_type}."
            )
        if self.future_target_type not in {"future_feature", "future_delta"}:
            raise ValueError(
                "`future_target_type` must be one of {'future_feature', 'future_delta'}. "
                f"Got {self.future_target_type}."
            )
        if self.future_training_stage not in {"joint", "pretrain"}:
            raise ValueError(
                "`future_training_stage` must be one of {'joint', 'pretrain'}. "
                f"Got {self.future_training_stage}."
            )
        if self.future_condition_fusion not in {"concat", "project_concat"}:
            raise ValueError(
                "`future_condition_fusion` must be one of {'concat', 'project_concat'}. "
                f"Got {self.future_condition_fusion}."
            )
        if self.future_num_input_steps <= 0:
            raise ValueError(
                f"`future_num_input_steps` must be > 0. Got {self.future_num_input_steps}."
            )
        if self.future_predictor_hidden_dim <= 0:
            raise ValueError(
                f"`future_predictor_hidden_dim` must be > 0. Got {self.future_predictor_hidden_dim}."
            )
        if self.lambda_future < 0:
            raise ValueError(f"`lambda_future` must be >= 0. Got {self.lambda_future}.")
        if self.future_cosine_loss_weight < 0:
            raise ValueError(
                "`future_cosine_loss_weight` must be >= 0. "
                f"Got {self.future_cosine_loss_weight}."
            )
        if self.future_condition_proj_dim is not None and self.future_condition_proj_dim <= 0:
            raise ValueError(
                "`future_condition_proj_dim` must be > 0 when set. "
                f"Got {self.future_condition_proj_dim}."
            )
        if self.future_training_stage == "pretrain" and not self.enable_future_predictor:
            raise ValueError(
                "`future_training_stage='pretrain'` requires `enable_future_predictor=True`."
            )
        if self.enable_future_predictor:
            if self.future_num_input_steps > self.n_obs_steps:
                raise ValueError(
                    "`future_num_input_steps` must be <= `n_obs_steps` when future predictor is enabled. "
                    f"Got {self.future_num_input_steps=} and {self.n_obs_steps=}."
                )
            if self.lambda_future > 0 and self.future_num_input_steps >= self.n_obs_steps:
                raise ValueError(
                    "Future loss supervision needs at least one-step shift. Require "
                    "`future_num_input_steps < n_obs_steps` when `lambda_future > 0`. "
                    f"Got {self.future_num_input_steps=} and {self.n_obs_steps=}."
                )
        if self.future_training_stage == "pretrain" and self.lambda_future <= 0:
            raise ValueError(
                "`future_training_stage='pretrain'` requires `lambda_future > 0`."
            )
        if self.kalman_feature_mlp_dim is not None and self.kalman_feature_mlp_dim <= 0:
            raise ValueError(
                f"`kalman_feature_mlp_dim` must be > 0 when set. Got {self.kalman_feature_mlp_dim}."
            )
        if self.kalman_predict_horizon < 0:
            raise ValueError(
                f"`kalman_predict_horizon` must be >= 0. Got {self.kalman_predict_horizon}."
            )
        if self.kalman_dt_fallback <= 0:
            raise ValueError(f"`kalman_dt_fallback` must be > 0. Got {self.kalman_dt_fallback}.")
        if self.kalman_meas_noise_std <= 0:
            raise ValueError(f"`kalman_meas_noise_std` must be > 0. Got {self.kalman_meas_noise_std}.")
        if self.kalman_accel_noise_std <= 0:
            raise ValueError(
                f"`kalman_accel_noise_std` must be > 0. Got {self.kalman_accel_noise_std}."
            )
        if self.kalman_init_pos_std <= 0:
            raise ValueError(f"`kalman_init_pos_std` must be > 0. Got {self.kalman_init_pos_std}.")
        if self.kalman_init_vel_std <= 0:
            raise ValueError(f"`kalman_init_vel_std` must be > 0. Got {self.kalman_init_vel_std}.")
        parts = self.kalman_state_pos_slice.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"`kalman_state_pos_slice` must use 'start:end' format. Got {self.kalman_state_pos_slice}."
            )
        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if parts[1] else None
        if start is not None and end is not None and (end - start) != 3:
            raise ValueError(
                "`kalman_state_pos_slice` must select exactly 3 dimensions for xyz. "
                f"Got {self.kalman_state_pos_slice}."
            )

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                # Explicitly disable cropping for resize+ratio path when crop_ratio == 1.0.
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
