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
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import train
from lerobot.utils.import_utils import register_third_party_plugins


@parser.wrap()
def train_future_predictor(cfg: TrainPipelineConfig):
    if cfg.policy is None or cfg.policy.type != "diffusion":
        raise ValueError("Future predictor pretraining entrypoint requires `policy.type=diffusion`.")

    cfg.policy.enable_future_predictor = True
    cfg.policy.future_training_stage = "pretrain"
    train(cfg)


def main():
    register_third_party_plugins()
    train_future_predictor()


if __name__ == "__main__":
    main()
