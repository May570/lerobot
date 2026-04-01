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

from collections.abc import Iterable
from typing import Any

from lerobot.processor.rename_processor import RenameObservationsProcessorStep

# Mapping from legacy local dataset keys to canonical LIBERO observation keys.
LIBERO_LEGACY_OBS_RENAME_MAP = {
    "actions": "action",
    "actions_is_pad": "action_is_pad",
    "image": "observation.images.image",
    "image_is_pad": "observation.images.image_is_pad",
    "wrist_image": "observation.images.image2",
    "wrist_image_is_pad": "observation.images.image2_is_pad",
    "state": "observation.state",
    "state_is_pad": "observation.state_is_pad",
}


def resolve_libero_rename_map(
    *,
    enable_legacy_compat: bool,
    env_cfg: Any,
    feature_keys: Iterable[str],
    user_rename_map: dict[str, str] | None,
) -> dict[str, str]:
    """
    Resolve the effective observation rename map.

    When compatibility is enabled and running LIBERO with legacy feature names
    (`image`/`wrist_image`/`state`), prepend a default mapping from legacy keys
    to canonical LIBERO observation keys.
    """
    resolved = dict(user_rename_map or {})

    if not enable_legacy_compat or getattr(env_cfg, "type", None) != "libero":
        return resolved

    keys = set(feature_keys)
    has_legacy_keys = {"image", "wrist_image"}.issubset(keys)
    has_canonical_keys = {"observation.images.image", "observation.images.image2"}.issubset(keys)

    if has_legacy_keys and not has_canonical_keys:
        merged = dict(LIBERO_LEGACY_OBS_RENAME_MAP)
        merged.update(resolved)
        return merged

    return resolved


def apply_rename_map_to_preprocessor(preprocessor: Any, rename_map: dict[str, str]) -> None:
    """
    Apply/merge rename_map into the RenameObservations step of a preprocessor pipeline.

    This is needed for freshly-constructed pipelines where preprocessor overrides are not used.
    """
    if not rename_map:
        return

    for step in getattr(preprocessor, "steps", []):
        if isinstance(step, RenameObservationsProcessorStep):
            merged = dict(step.rename_map)
            merged.update(rename_map)
            step.rename_map = merged
            break


def apply_rename_map_to_batch(batch: dict[str, Any], rename_map: dict[str, str]) -> dict[str, Any]:
    """Rename top-level batch keys using the provided map."""
    if not rename_map:
        return batch
    return {rename_map.get(key, key): value for key, value in batch.items()}
