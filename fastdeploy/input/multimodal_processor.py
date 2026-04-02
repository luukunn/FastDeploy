"""
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
"""

"""Unified multimodal processor for all VL model types.

Consolidates the four separate VL processor DataProcessor classes
(qwen_vl, qwen3_vl, paddleocr_vl, ernie4_5_vl) into a single class
that dispatches per ``model_type``.

Encoding logic (text2ids, request2ids, _add_image, etc.) is defined
in the ``MultiModalEncoderMixin`` (see ``mm_encoder.py``).
"""

import os
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Dict, Optional

import numpy as np
import paddle

from fastdeploy.input.base_processor import BaseTextProcessor
from fastdeploy.input.mm_encoder import (  # noqa: F401 — re-exported for external use
    _QWEN3_VIDEO_MAX_PIXELS,
    _QWEN3_VIDEO_MIN_PIXELS,
    _QWEN_FAMILY,
    ERNIE4_5_VL,
    PADDLEOCR_VL,
    QWEN3_VL,
    QWEN_VL,
    MultiModalEncoderMixin,
)
from fastdeploy.input.utils import (
    IDS_TYPE_FLAG,
    MAX_IMAGE_DIMENSION,
    process_stop_token_ids,
)
from fastdeploy.utils import data_processor_logger

# ---- Constants only used by this module ----
_SUPPORTED_MODEL_TYPES = {QWEN_VL, QWEN3_VL, PADDLEOCR_VL, ERNIE4_5_VL}

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

_DEFAULT_MM_LIMITS = {"image": 1, "video": 1, "audio": 1}

_SAMPLING_EPS = 1e-5

# Qwen-family defaults
_QWEN_FRAME_FACTOR = 2
_QWEN_FPS = 2.0
_QWEN_FPS_MIN_FRAMES = 4
_QWEN_FPS_MAX_FRAMES = 768


class MultiModalProcessor(MultiModalEncoderMixin, BaseTextProcessor):
    """Unified multimodal processor for all supported VL model types.

    Dispatches image-processor creation, config initialisation, and
    encoding logic based on ``model_type``.
    """

    # Ernie special token strings
    _ERNIE_IMG_START = "<|IMAGE_START|>"
    _ERNIE_IMG_END = "<|IMAGE_END|>"
    _ERNIE_VID_START = "<|VIDEO_START|>"
    _ERNIE_VID_END = "<|VIDEO_END|>"

    def __init__(
        self,
        model_name_or_path: str,
        model_type: str,
        config=None,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        reasoning_parser_obj=None,
        tool_parser_obj=None,
        enable_processor_cache: bool = False,
    ):
        if model_type not in _SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. " f"Must be one of {sorted(_SUPPORTED_MODEL_TYPES)}."
            )
        self.model_type = model_type
        self.config = config
        self.enable_processor_cache = enable_processor_cache

        tokenizer_type = "ernie4_5" if model_type == ERNIE4_5_VL else "auto"

        super().__init__(
            model_name_or_path,
            tokenizer_type=tokenizer_type,
            reasoning_parser_obj=reasoning_parser_obj,
            tool_parser_obj=tool_parser_obj,
        )

        data_processor_logger.info(f"model_name_or_path: {model_name_or_path}")

        processor_kwargs = self._parse_processor_kwargs(mm_processor_kwargs)
        self._init_image_processor()
        self._init_encoding_tokenizer()
        self._init_conv_params()
        self._init_special_tokens()
        self._init_video_params(processor_kwargs)
        if model_type == ERNIE4_5_VL:
            self._init_ernie_pixel_params(processor_kwargs)
            self._init_ernie_token_type_mapping()
        self.limit_mm_per_prompt = self._parse_limits(limit_mm_per_prompt)

    def _load_tokenizer(self):
        """Load the appropriate tokenizer based on model_type."""
        if self.tokenizer_type == "ernie4_5":
            from paddleformers.transformers import AutoTokenizer as PFAutoTokenizer

            tokenizer = PFAutoTokenizer.from_pretrained(self.model_name_or_path)
        else:
            from paddleformers.transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side="left", use_fast=True)
        return tokenizer

    def _init_encoding_tokenizer(self):
        """Create a separate encoding tokenizer for ernie (Ernie4_5Tokenizer)."""
        if self.model_type == ERNIE4_5_VL:
            from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer

            vocab_file_names = ["tokenizer.model", "spm.model", "ernie_token_100k.model"]
            for name in vocab_file_names:
                if os.path.exists(os.path.join(self.model_name_or_path, name)):
                    Ernie4_5Tokenizer.resource_files_names["vocab_file"] = name
                    break
            self.encoding_tokenizer = Ernie4_5Tokenizer.from_pretrained(self.model_name_or_path)
            self.encoding_tokenizer.ignored_index = -100
        else:
            self.encoding_tokenizer = self.tokenizer

    def _init_image_processor(self):
        """Create the model-type-specific image processor."""
        if self.model_type == ERNIE4_5_VL:
            from fastdeploy.input.image_processors.adaptive_processor import (
                AdaptiveImageProcessor,
            )

            self.image_processor = AdaptiveImageProcessor.from_pretrained(self.model_name_or_path)
        elif self.model_type == QWEN3_VL:
            from fastdeploy.input.image_processors.qwen3_processor import ImageProcessor

            self.image_processor = ImageProcessor.from_pretrained(self.model_name_or_path)
        elif self.model_type == PADDLEOCR_VL:
            from fastdeploy.input.image_processors.paddleocr_processor import (
                ImageProcessor,
            )

            self.image_processor = ImageProcessor.from_pretrained(self.model_name_or_path)
        else:  # QWEN_VL
            from fastdeploy.input.image_processors.qwen_processor import ImageProcessor

            self.image_processor = ImageProcessor.from_pretrained(self.model_name_or_path)

    def _init_conv_params(self):
        """Set spatial/temporal convolution sizes."""
        if self.model_type == ERNIE4_5_VL:
            self.spatial_conv_size = 2
            self.temporal_conv_size = 2
        else:
            self.spatial_conv_size = self.image_processor.merge_size
            self.temporal_conv_size = self.image_processor.temporal_patch_size

    def _init_special_tokens(self):
        """Set model-type-specific special tokens and IDs."""
        if self.model_type in (QWEN_VL, QWEN3_VL):
            self.image_token = "<|image_pad|>"
            self.video_token = "<|video_pad|>"
            self.image_token_id = self.encoding_tokenizer.convert_tokens_to_ids(self.image_token)
            self.video_token_id = self.encoding_tokenizer.convert_tokens_to_ids(self.video_token)
            self.image_patch_id = self.image_token_id
            self._video_fill_token_id = self.image_token_id
            self.vision_start = "<|vision_start|>"
            self.vision_start_id = self.encoding_tokenizer.convert_tokens_to_ids(self.vision_start)
        elif self.model_type == PADDLEOCR_VL:
            self.image_token = "<|IMAGE_PLACEHOLDER|>"
            self.video_token = "<|video_pad|>"
            self.image_token_id = self.encoding_tokenizer.convert_tokens_to_ids(self.image_token)
            self.video_token_id = self.encoding_tokenizer.convert_tokens_to_ids(self.video_token)
            self.image_patch_id = self.image_token_id
            self._video_fill_token_id = self.video_token_id
            self.vision_start = "<|IMAGE_START|>"
            self.vision_start_id = self.encoding_tokenizer.convert_tokens_to_ids(self.vision_start)
        else:  # ERNIE4_5_VL
            self.image_patch_id = self.encoding_tokenizer.convert_tokens_to_ids("<|IMAGE_PLACEHOLDER|>")
            self._video_fill_token_id = self.image_patch_id
            self.image_start_id = self.encoding_tokenizer.convert_tokens_to_ids(self._ERNIE_IMG_START)
            self.image_end_id = self.encoding_tokenizer.convert_tokens_to_ids(self._ERNIE_IMG_END)
            self.video_start_id = self.encoding_tokenizer.convert_tokens_to_ids(self._ERNIE_VID_START)
            self.video_end_id = self.encoding_tokenizer.convert_tokens_to_ids(self._ERNIE_VID_END)

        tokens_per_second_default = 2
        if self.model_type in (QWEN_VL, PADDLEOCR_VL):
            tokens_per_second_default = getattr(getattr(self.config, "vision_config", None), "tokens_per_second", 2)
        self.tokens_per_second = tokens_per_second_default

        self.role_prefixes = {
            "system": "",
            "user": "User: ",
            "bot": "Assistant: ",
            "assistant": "Assistant: ",
        }
        if self.model_type == ERNIE4_5_VL:
            self.role_prefixes["tool"] = "Tool: "

    def _init_video_params(self, processor_kwargs):
        """Set video sampling parameters from kwargs or defaults."""
        if self.model_type == ERNIE4_5_VL:
            self.min_frames = processor_kwargs.get("video_min_frames", 16)
            self.max_frames = processor_kwargs.get("video_max_frames", 180)
            self.target_frames = processor_kwargs.get("video_target_frames", -1)
            self.fps = processor_kwargs.get("video_fps", 2)
            self.frames_sample = processor_kwargs.get("video_frames_sample", "leading")
        else:
            self.min_frames = processor_kwargs.get("video_min_frames", _QWEN_FPS_MIN_FRAMES)
            self.max_frames = processor_kwargs.get("video_max_frames", _QWEN_FPS_MAX_FRAMES)
            self.target_frames = -1
            self.fps = _QWEN_FPS
            if self.model_type == PADDLEOCR_VL:
                self.frame_factor = self.temporal_conv_size
                self.fps = -1
            else:
                self.frame_factor = _QWEN_FRAME_FACTOR

    def _init_ernie_pixel_params(self, processor_kwargs):
        """Set ernie-specific pixel constraints."""
        self.image_min_pixels = processor_kwargs.get("image_min_pixels", 4 * 28 * 28)
        self.image_max_pixels = processor_kwargs.get("image_max_pixels", 6177 * 28 * 28)
        self.video_min_pixels = processor_kwargs.get("video_min_pixels", 299 * 28 * 28)
        self.video_max_pixels = processor_kwargs.get("video_max_pixels", 1196 * 28 * 28)

    def _init_ernie_token_type_mapping(self):
        """Build token_type_mapping for ernie."""
        self._token_type_mapping = defaultdict(lambda: IDS_TYPE_FLAG["text"])
        for token in (self._ERNIE_IMG_START, self._ERNIE_IMG_END, self._ERNIE_VID_START, self._ERNIE_VID_END):
            self._token_type_mapping[token] = IDS_TYPE_FLAG["image"]
        self._token_type_mapping[self.image_patch_id] = IDS_TYPE_FLAG["image"]

    def _parse_processor_kwargs(self, kwargs: Optional[dict]) -> dict:
        """Parse and validate multimodal processor kwargs."""
        if not kwargs:
            return {}

        try:
            if not isinstance(kwargs, dict):
                raise ValueError("mm-processor-kwargs must be a dictionary")

            data_processor_logger.info(f"Processing kwargs: {kwargs}")

            if self.model_type == ERNIE4_5_VL:
                expected_types = _ERNIE_EXPECTED_KWARGS
            else:
                expected_types = _QWEN_EXPECTED_KWARGS

            for key, value in kwargs.items():
                if key in expected_types and not isinstance(value, expected_types[key]):
                    raise ValueError(
                        f"Invalid type for {key}: expected "
                        f"{expected_types[key].__name__}, got {type(value).__name__}"
                    )
            return kwargs

        except Exception as e:
            data_processor_logger.warning(f"Invalid mm-processor-kwargs format: {e}")
            return {}

    def _parse_limits(self, limits: Optional[dict]) -> dict:
        """Parse multimodal input limits, merging with defaults."""
        if not limits:
            return dict(_DEFAULT_MM_LIMITS)

        try:
            if not isinstance(limits, dict):
                raise ValueError("limit-mm-per-prompt must be a dictionary")
            data_processor_logger.info(f"_parse_limits:{limits}")
            return {**_DEFAULT_MM_LIMITS, **limits}
        except Exception as e:
            data_processor_logger.warning(f"Invalid limit-mm-per-prompt format: {e}, using default limits")
            return dict(_DEFAULT_MM_LIMITS)

    def _check_mm_limits(self, item):
        """Validate multimodal inputs against configured limits."""
        if isinstance(item, dict):
            mm_data = item
        else:
            mm_data = {"image": [], "video": []}
            for message in item:
                if isinstance(message.get("content"), list):
                    for part in message["content"]:
                        part_type = part.get("type")
                        if part_type in ("image_url", "image"):
                            mm_data["image"].append(part)
                        elif part_type in ("video_url", "video"):
                            mm_data["video"].append(part)

        for modality, data in mm_data.items():
            if modality in self.limit_mm_per_prompt:
                limit = self.limit_mm_per_prompt[modality]
                if len(data) > limit:
                    raise ValueError(f"Too many {modality} items in prompt, " f"got {len(data)} but limit is {limit}")

    @staticmethod
    def _mm_num_tokens_qwen(grid_thw):
        """Calculate token count for qwen-family models (merge_size=2, no temporal downsampling)."""
        if isinstance(grid_thw, paddle.Tensor):
            grid_thw = grid_thw.numpy()
        if len(grid_thw) == 0:
            return 0

        def calc_one(thw):
            t, h, w = map(int, thw)
            return t * h * w // 4

        if isinstance(grid_thw[0], (list, tuple, np.ndarray)):
            return [calc_one(x) for x in grid_thw]
        return calc_one(grid_thw)

    @staticmethod
    def _mm_num_tokens_ernie(grid_thw):
        """Calculate token count for ernie (videos have temporal_conv_size downsampling)."""
        if isinstance(grid_thw, paddle.Tensor):
            grid_thw = grid_thw.numpy()
        if len(grid_thw) == 0:
            return 0

        def calc_one(thw):
            t, h, w = map(int, thw)
            if t == 1:
                return t * h * w // 4
            else:
                return t * h * w // 4 // 2

        if isinstance(grid_thw[0], (list, tuple, np.ndarray)):
            return [calc_one(x) for x in grid_thw]
        return calc_one(grid_thw)

    def get_mm_max_tokens_per_item(self, seq_len: int) -> Optional[Mapping[str, int]]:
        if self.model_type != ERNIE4_5_VL:
            return None
        resized_height, resized_width = self.image_processor.get_smarted_resize(
            height=MAX_IMAGE_DIMENSION,
            width=MAX_IMAGE_DIMENSION,
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
        )[0]
        patches_h_img, patches_w_img = self.image_processor.get_smarted_resize(
            height=resized_height,
            width=resized_width,
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
        )[1]
        max_image_tokens = min((patches_h_img * patches_w_img) // (self.spatial_conv_size**2), seq_len)
        patches_h_vid, patches_w_vid = self.image_processor.get_smarted_resize(
            height=resized_height,
            width=resized_width,
            min_pixels=self.video_min_pixels,
            max_pixels=self.video_max_pixels,
        )[1]
        max_video_tokens = min(
            (patches_h_vid * patches_w_vid) // (self.spatial_conv_size**2 * self.temporal_conv_size),
            seq_len,
        )
        return {"image": max_image_tokens, "video": max_video_tokens}

    def process_request_dict(self, request, max_model_len=None):
        request = self._apply_default_parameters(request)

        if not request.get("eos_token_ids"):
            request["eos_token_ids"] = self.eos_token_ids

        self._process_stop_tokens(request)

        if self.model_type != PADDLEOCR_VL:
            self._process_bad_words(request)

        if self.model_type == ERNIE4_5_VL:
            logits_processors_args = self._prepare_think_stop_sentence(
                request.get("logits_processors_args") or {}, max_model_len
            )
            request["logits_processors_args"] = logits_processors_args

        outputs = self._tokenize_request(request)

        self._process_post_tokens(request, outputs)

        if self.model_type in (QWEN_VL, QWEN3_VL):
            request["enable_thinking"] = False

        outputs = self.pack_outputs(outputs)

        if self.model_type in (QWEN3_VL, ERNIE4_5_VL) and request.get("prompt_token_ids"):
            pass
        else:
            request["prompt_token_ids"] = outputs["input_ids"].tolist()
        request["prompt_token_ids_len"] = len(request["prompt_token_ids"])
        request["multimodal_inputs"] = outputs

        if max_model_len is not None and len(request["prompt_token_ids"]) > max_model_len:
            request["prompt_token_ids"] = request["prompt_token_ids"][: max_model_len - 1]

        if self.model_type == ERNIE4_5_VL:
            logits_processors_args = self._update_thinking_prompt_state(
                request["prompt_token_ids"], request.get("logits_processors_args") or {}
            )
            request["logits_processors_args"] = logits_processors_args

        max_tokens = max_model_len - len(request["prompt_token_ids"])
        if request.get("max_tokens") is None:
            request["max_tokens"] = max(1, max_tokens)
        else:
            request["max_tokens"] = min(max_tokens, request["max_tokens"])

        if self.model_type == ERNIE4_5_VL and request.get("reasoning_max_tokens") is None:
            request["reasoning_max_tokens"] = max(int(request["max_tokens"] * 0.8), 1)

        if self.model_type in (PADDLEOCR_VL, ERNIE4_5_VL):
            if request.get("top_p") is not None and request.get("top_p") < _SAMPLING_EPS:
                request["top_p"] = _SAMPLING_EPS
                request["top_k"] = 1

        if self.model_type != QWEN3_VL and self.reasoning_parser:
            self._apply_reasoning_parser(request)

        if self.model_type == ERNIE4_5_VL:
            if request.get("response_max_tokens") is not None and request.get("enable_thinking") is False:
                request["max_tokens"] = min(request["response_max_tokens"], request["max_tokens"])

        data_processor_logger.info(f"Processed request {request}")
        return request

    def _process_stop_tokens(self, request):
        if self.model_type == QWEN3_VL:
            stop_sequences = request.get("stop", [])
            if stop_sequences:
                stop_seqs, stop_seqs_len = self.update_stop_seq(stop_sequences)
                request["stop_token_ids"] = stop_seqs
                request["stop_seqs_len"] = stop_seqs_len
        else:
            process_stop_token_ids(request, self.update_stop_seq)

    def _process_bad_words(self, request):
        bad_words = request.get("bad_words")
        bad_words_token_ids = request.get("bad_words_token_ids")
        if bad_words:
            bad_words_token_ids = self.update_bad_words(bad_words, bad_words_token_ids)
            request["bad_words_token_ids"] = bad_words_token_ids

    def _tokenize_request(self, request):
        """Core tokenization dispatch: prompt_token_ids > prompt > messages."""
        default_thinking = True if self.model_type == ERNIE4_5_VL else False

        if request.get("prompt_token_ids") and self.model_type in (QWEN3_VL, ERNIE4_5_VL):
            messages = request.get("messages")
            if messages:
                self._check_mm_limits(messages)
            request.setdefault("enable_thinking", default_thinking)
            return self.prompt_token_ids2outputs(request)

        elif request.get("prompt"):
            multimodal_data = request.get("multimodal_data") or {}
            self._check_mm_limits(multimodal_data)
            images = multimodal_data.get("image", None)
            videos = multimodal_data.get("video", None)
            if self.model_type == ERNIE4_5_VL:
                request["prompt_tokens"] = request.get("prompt")
            request.setdefault("enable_thinking", default_thinking)
            return self.text2ids(request["prompt"], images, videos)

        elif request.get("messages"):
            messages = request["messages"]
            self._check_mm_limits(messages)
            chat_template_kwargs = request.get("chat_template_kwargs")
            if chat_template_kwargs:
                if isinstance(chat_template_kwargs, dict):
                    for k, v in chat_template_kwargs.items():
                        if k not in request or request[k] is None:
                            request[k] = v
                else:
                    raise ValueError("Invalid input: chat_template_kwargs must be a dict")
            request.setdefault("enable_thinking", default_thinking)
            return self.request2ids(request)

        else:
            raise ValueError(f"Request must contain 'prompt', or 'messages': {request}")

    def _process_post_tokens(self, request, outputs):
        if self.model_type == PADDLEOCR_VL:
            metadata = request.get("metadata")
            if metadata and metadata.get("generated_token_ids"):
                self._append_completion_tokens_qwen(outputs, metadata["generated_token_ids"])
        else:
            if request.get("completion_token_ids"):
                self.append_completion_tokens(outputs, request["completion_token_ids"])

    def _apply_reasoning_parser(self, request):
        model_status = self.reasoning_parser.get_model_status(request["prompt_token_ids"])
        parts = request["request_id"].split("_")
        if len(parts) > 1:
            real_req_id = parts[0]
            index = int(parts[1])
            n = request.get("n", 1)
            for idx in range(index * n, (index + 1) * n):
                self.model_status_dict[f"{real_req_id}_{idx}"] = model_status
        else:
            self.model_status_dict[request["request_id"]] = model_status
        request["enable_thinking"] = model_status == "think_start"

    def append_completion_tokens(self, multimodal_inputs, completion_token_ids):
        if self.model_type == ERNIE4_5_VL:
            self._append_completion_tokens_ernie(multimodal_inputs, completion_token_ids)
        else:
            self._append_completion_tokens_qwen(multimodal_inputs, completion_token_ids)

    def _append_completion_tokens_qwen(self, multimodal_inputs, completion_token_ids):
        num_tokens = len(completion_token_ids)
        multimodal_inputs["input_ids"].extend(completion_token_ids)
        multimodal_inputs["token_type_ids"].extend([0] * num_tokens)

        pos_ids = self._compute_text_positions(multimodal_inputs["cur_position"], num_tokens)
        multimodal_inputs["position_ids"].append(pos_ids)
        multimodal_inputs["cur_position"] += num_tokens

    def _append_completion_tokens_ernie(self, multimodal_inputs, completion_token_ids):
        num_tokens = len(completion_token_ids)
        multimodal_inputs["input_ids"].extend(completion_token_ids)
        multimodal_inputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * num_tokens)

        start = multimodal_inputs["cur_position"]
        for i in range(num_tokens):
            multimodal_inputs["position_ids"].append([start + i] * 3)
        multimodal_inputs["cur_position"] += num_tokens

    def pack_outputs(self, outputs):
        if not outputs["images"]:
            outputs["images"] = None
            outputs["grid_thw"] = None
            outputs["image_type_ids"] = None
        else:
            outputs["images"] = np.vstack(outputs["images"])
            outputs["grid_thw"] = np.vstack(outputs["grid_thw"])
            outputs["image_type_ids"] = np.array(outputs["image_type_ids"])

        outputs["input_ids"] = np.array(outputs["input_ids"], dtype=np.int64)
        outputs["token_type_ids"] = np.array(outputs["token_type_ids"], dtype=np.int64)

        if self.model_type == ERNIE4_5_VL:
            outputs["mm_num_token_func"] = self._mm_num_tokens_ernie
            outputs["position_ids"] = np.array(outputs["position_ids"], dtype=np.int64)
            outputs["image_patch_id"] = self.image_patch_id
        else:
            outputs["mm_num_token_func"] = self._mm_num_tokens_qwen
            outputs["position_ids"] = np.concatenate(outputs["position_ids"], axis=1, dtype=np.int64)
            outputs["image_patch_id"] = (
                self.image_token_id if self.model_type in (QWEN_VL, QWEN3_VL) else self.image_patch_id
            )
            outputs["video_patch_id"] = self.video_token_id
            outputs["position_ids"] = outputs["position_ids"].transpose(1, 0)

        return outputs
