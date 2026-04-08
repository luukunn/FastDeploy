# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Multimodal model configuration dataclass and per-model configs.

Each supported VL model type has a frozen ``MMModelConfig`` that captures
its static configuration differences.  The main ``MultiModalProcessor``
uses ``MODEL_CONFIGS[model_type]`` to replace scattered if/else branches.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

QWEN_VL = "qwen_vl"
QWEN3_VL = "qwen3_vl"
PADDLEOCR_VL = "paddleocr_vl"
ERNIE4_5_VL = "ernie4_5_vl"

_SUPPORTED_MODEL_TYPES = {QWEN_VL, QWEN3_VL, PADDLEOCR_VL, ERNIE4_5_VL}


@dataclass(frozen=True)
class MMModelConfig:
    """Frozen configuration for a multimodal VL model type.

    Holds all the static knobs that differ across model families so that
    the ``MultiModalProcessor`` and ``Encoding`` classes can query ``cfg.xxx``
    instead of checking ``model_type == ...``.
    """

    image_placeholder: str
    video_placeholder: str

    tokenizer_type: str = "auto"

    spatial_conv_size_override: Optional[int] = None
    temporal_conv_size_override: Optional[int] = None

    default_min_frames: int = 4
    default_max_frames: int = 768
    default_target_frames: int = -1
    default_fps: float = 2.0
    default_frames_sample: str = "leading"

    has_bad_words: bool = True
    has_vit_fields: bool = False
    has_tool_role: bool = False
    clamp_top_p: bool = False
    default_thinking: bool = False
    force_disable_thinking: bool = False
    set_default_reasoning_max_tokens: bool = False
    cap_response_max_tokens: bool = False
    has_logits_processor_think: bool = False
    skip_reasoning_parser: bool = False

    chat_template_pass_request: bool = False

    supports_prompt_token_ids: bool = False
    preserve_prompt_token_ids: bool = False

    stop_tokens_variant: str = "default"
    position_ids_format: str = "ndarray"
    has_ernie_boundary_tokens: bool = False
    video_fill_uses_image_token: bool = True
    grid_thw_key: str = "grid_thw"
    completion_token_source: str = "completion_token_ids"
    expected_kwargs: Dict[str, type] = field(default_factory=dict)
    sample_frames_variant: str = "qwen"
    frame_factor_override: Optional[int] = None


_QWEN_EXPECTED_KWARGS = {
    "video_max_frames": int,
    "video_min_frames": int,
}

_ERNIE_EXPECTED_KWARGS = {
    "spatial_conv_size": int,
    "temporal_conv_size": int,
    "image_min_pixels": int,
    "image_max_pixels": int,
    "video_min_pixels": int,
    "video_max_pixels": int,
    "video_target_frames": int,
    "video_frames_sample": str,
    "video_max_frames": int,
    "video_min_frames": int,
    "video_fps": int,
}


MODEL_CONFIGS: Dict[str, MMModelConfig] = {
    QWEN_VL: MMModelConfig(
        image_placeholder="<|image_pad|>",
        video_placeholder="<|video_pad|>",
        has_bad_words=True,
        force_disable_thinking=True,
        expected_kwargs=_QWEN_EXPECTED_KWARGS,
        frame_factor_override=2,
    ),
    QWEN3_VL: MMModelConfig(
        image_placeholder="<|image_pad|>",
        video_placeholder="<|video_pad|>",
        has_bad_words=True,
        force_disable_thinking=True,
        supports_prompt_token_ids=True,
        preserve_prompt_token_ids=True,
        stop_tokens_variant="qwen3",
        skip_reasoning_parser=True,
        expected_kwargs=_QWEN_EXPECTED_KWARGS,
        frame_factor_override=2,
    ),
    PADDLEOCR_VL: MMModelConfig(
        image_placeholder="<|IMAGE_PLACEHOLDER|>",
        video_placeholder="<|video_pad|>",
        has_bad_words=False,
        has_vit_fields=True,
        clamp_top_p=True,
        video_fill_uses_image_token=False,
        grid_thw_key="image_grid_thw",
        completion_token_source="metadata_generated",
        sample_frames_variant="paddleocr",
        default_fps=-1.0,
        expected_kwargs=_QWEN_EXPECTED_KWARGS,
    ),
    ERNIE4_5_VL: MMModelConfig(
        image_placeholder="<|image@placeholder|>",
        video_placeholder="<|video@placeholder|>",
        tokenizer_type="ernie4_5",
        default_min_frames=16,
        default_max_frames=180,
        default_target_frames=-1,
        default_fps=2,
        default_frames_sample="leading",
        has_bad_words=True,
        has_tool_role=True,
        clamp_top_p=True,
        default_thinking=True,
        set_default_reasoning_max_tokens=True,
        cap_response_max_tokens=True,
        has_logits_processor_think=True,
        chat_template_pass_request=True,
        supports_prompt_token_ids=True,
        preserve_prompt_token_ids=True,
        position_ids_format="list",
        has_ernie_boundary_tokens=True,
        video_fill_uses_image_token=True,
        expected_kwargs=_ERNIE_EXPECTED_KWARGS,
    ),
}
