#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
python -m lerobot.scripts.eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
python -m lerobot.scripts.eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import json
import logging
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Callable

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import add_envs_task, check_env_attributes_and_types, preprocess_observation
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)

from pprint import pprint
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

def read_data():
    # repo_id = "lerobot/pusht"
    # # We can have a look and fetch its metadata to know more about it:
    # ds_meta = LeRobotDatasetMetadata(repo_id)

    # # By instantiating just this class, you can quickly access useful information about the content and the
    # # structure of the dataset without downloading the actual data yet (only metadata files — which are
    # # lightweight).
    # print(f"Total number of episodes: {ds_meta.total_episodes}")
    # print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
    # print(f"Frames per second used during data collection: {ds_meta.fps}")
    # print(f"Robot type: {ds_meta.robot_type}")
    # print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")

    # print("Tasks:")
    # print(ds_meta.tasks)
    # print("Features:")
    # pprint(ds_meta.features)

    # # You can also get a short summary by simply printing the object:
    # print(ds_meta)

    # # You can then load the actual dataset from the hub.
    # # Either load any subset of episodes:
    # dataset = LeRobotDataset(repo_id, episodes=[0])

    # # And see how many frames you have:
    # print(f"Selected episodes: {dataset.episodes}")
    # print(f"Number of episodes selected: {dataset.num_episodes}")
    # print(f"Number of frames selected: {dataset.num_frames}")

    # # # Or simply load the entire dataset:
    # # dataset = LeRobotDataset(repo_id)
    # # print(f"Number of episodes selected: {dataset.num_episodes}")
    # # print(f"Number of frames selected: {dataset.num_frames}")

    # # The previous metadata class is contained in the 'meta' attribute of the dataset:
    # print(dataset.meta)

    # # LeRobotDataset actually wraps an underlying Hugging Face dataset
    # # (see https://huggingface.co/docs/datasets for more information).
    # print(dataset.hf_dataset)

    # # LeRobot datasets also subclasses PyTorch datasets so you can do everything you know and love from working
    # # with the latter, like iterating through the dataset.
    # # The __getitem__ iterates over the frames of the dataset. Since our datasets are also structured by
    # # episodes, you can access the frame indices of any episode using the episode_data_index. Here, we access
    # # frame indices associated to the first episode:
    # episode_index = 0
    # from_idx = dataset.episode_data_index["from"][episode_index].item()
    # to_idx = dataset.episode_data_index["to"][episode_index].item()

    # # Then we grab all the image frames from the first camera:
    # camera_key = dataset.meta.camera_keys[0]
    # frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]

    # # The objects returned by the dataset are all torch.Tensors
    # print(type(frames[0]))
    # print(frames[0].shape)

    # # Since we're using pytorch, the shape is in pytorch, channel-first convention (c, h, w).
    # # We can compare this shape with the information available for that feature
    # pprint(dataset.features[camera_key])
    # # In particular:
    # print(dataset.features[camera_key]["shape"])
    # # The shape is in (h, w, c) which is a more universal format.

    # # For many machine learning applications we need to load the history of past observations or trajectories of
    # # future actions. Our datasets can load previous and future frames for each key/modality, using timestamps
    # # differences with the current loaded frame. For instance:
    # delta_timestamps = {
    #     # loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
    #     camera_key: [-1, -0.5, -0.20, 0],
    #     # loads 6 state vectors: 1.5 seconds before, 1 second before, ... 200 ms, 100 ms, and current frame
    #     "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, 0],
    #     # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
    #     "action": [t / dataset.fps for t in range(64)],
    # }
    # # Note that in any case, these delta_timestamps values need to be multiples of (1/fps) so that added to any
    # # timestamp, you still get a valid timestamp.

    # dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
    # print(f"\n{dataset[0][camera_key].shape=}")  # (4, c, h, w)
    # print(f"{dataset[0]['observation.state'].shape=}")  # (6, c)
    # print(f"{dataset[0]['action'].shape=}\n")  # (64, c)

    # # Finally, our datasets are fully compatible with PyTorch dataloaders and samplers because they are just
    # # PyTorch datasets.
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     num_workers=0,
    #     batch_size=32,
    #     shuffle=True,
    # )

    # for batch in dataloader:
    #     print(f"{batch[camera_key].shape=}")  # (32, 4, c, h, w)
    #     print(f"{batch['observation.state'].shape=}")  # (32, 6, c)
    #     print(f"{batch['action'].shape=}")  # (32, 64, c)
    #     break

    # 假设你已经加载了 dataset
    dataset = LeRobotDataset(repo_id="lerobot/pusht")

    # 选择你要保存的 episode
    episode_index = 67  # 比如你想保存 episode 0
    camera_key = dataset.meta.camera_keys[0]  # 获取第一个摄像头的图像键

    # 提取该 episode 的数据
    from_idx = dataset.episode_data_index["from"][episode_index].item()
    to_idx = dataset.episode_data_index["to"][episode_index].item()

    # 获取该 episode 的图像、状态和动作数据
    frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]
    state_data = [dataset[idx]["observation.state"] for idx in range(from_idx, to_idx)]
    action_data = [dataset[idx]["action"] for idx in range(from_idx, to_idx)]

    frames = torch.stack(frames)  # stack tensors to form a batch
    state_data = torch.stack(state_data)  # stack tensors to form a batch
    action_data = torch.stack(action_data)  # stack tensors to form a batch

    print(f"Frames shape: {frames.shape}")
    print(f"State shape: {state_data.shape}")
    print(f"Action shape: {action_data.shape}")

    return frames, state_data, action_data


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)

    frames, state_data, action_data = read_data()
    time.sleep(1)

    for t in range(frames.shape[0]):
        num_episodes = 10

        # 取出当前切片
        frame = frames[t]  # 形状为 (3, 96, 96)
        state = state_data[t]   # 形状为 (2,)
        ground_actions = action_data[t:t+15].float()  # 形状为 (15, action_dim)

        # 将当前切片的内容重复 num_episodes 次
        frame_repeated = frame.unsqueeze(0).repeat(num_episodes, 1, 1, 1)  # 形状为 (10, 3, 96, 96)
        state_repeated = state.unsqueeze(0).repeat(num_episodes, 1)       # 形状为 (10, 2)
        ground_actions_repeated = ground_actions.unsqueeze(0).repeat(num_episodes, 1, 1)  # 形状为 (10, 15, action_dim)

        # 创建字典
        data_dict = {
            'observation.image': frame_repeated,
            'observation.state': state_repeated,
        }

        data_dict = {
            key: data_dict[key].to(device, non_blocking=device.type == "cuda") for key in data_dict
        }
        ground_actions_repeated = ground_actions_repeated.to(device, non_blocking=device.type == "cuda")

        # print(f"keys: {list(data_dict.keys())}")
        # for key, value in data_dict.items():
        #     if isinstance(value, torch.Tensor):  # 确保是 Tensor 类型
        #         print(f"Key: {key}, Shape: {value.shape}")
        #     elif isinstance(value, list):  # 如果是 list 类型
        #         print(f"Key: {key}, Length: {len(value)}")
        #     else:
        #         print(f"Key: {key}, Unknown type")
        # time.sleep(5)


        mean = torch.tensor([228.23964986314087, 293.9890841163204]).to(device, non_blocking=device.type == "cuda")  # 每个特征的均值
        std = torch.tensor([101.5995746254447, 96.0392898600564]).to(device, non_blocking=device.type == "cuda")
        min = torch.tensor([12.0, 25.0]).to(device, non_blocking=device.type == "cuda")  # 每个特征的最小值
        max = torch.tensor([511.0, 511.0]).to(device, non_blocking=device.type == "cuda")  # 每个特征的最大值
        with torch.inference_mode():
            action = policy.offline_select_action(data_dict, ground_actions_repeated, mean, std)





    while not np.all(done):
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        observation = {
            key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
        }

        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        observation = add_envs_task(env, observation)

        with torch.inference_mode():
            action = policy.select_action(observation)

        # Convert to CPU / numpy.
        action = action.to("cpu").numpy()
        assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action)
        if render_callback is not None:
            render_callback(env)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available of none of the envs finished.
        if "final_info" in info:
            successes = [info["is_success"] if info is not None else False for info in info["final_info"]]
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        done = terminated | truncated | done

        # If the inference is done, stop the RTC thread.
        if np.all(done):
            if policy.config.use_rtc:
                policy.stop_rtc_thread = True

        all_actions.append(torch.from_numpy(action))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        "action": torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret["observation"] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        raise ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env,
            policy,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data["action"].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            "action": rollout_data["action"][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            "next.done": rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            "next.reward": rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data["observation"]:
            ep_dict[key] = rollout_data["observation"][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy(
            env,
            policy,
            cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
        )
    print(info["aggregated"])

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    env.close()

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
