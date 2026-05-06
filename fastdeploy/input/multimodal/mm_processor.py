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

"""Base class for multimodal processors.

Each VL model family (Qwen, Ernie, PaddleOCR) subclasses ``MMProcessor``
and implements abstract methods for model-specific encoding logic.
The base class provides:
- Template method ``process()`` orchestrating the full multimodal pipeline
- ``_text2ids()`` scanning loop for placeholder replacement
- ``_pack_outputs()`` for converting intermediate lists to packed arrays
- Cache utilities (ZMQ-based processor cache)
"""

import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zmq

from fastdeploy.input.utils import IDS_TYPE_FLAG
from fastdeploy.multimodal.hasher import MultimodalHasher
from fastdeploy.utils import data_processor_logger


class MMProcessor(ABC):
    """Abstract base class for multimodal processors.

    Subclasses must implement all abstract methods to handle model-specific
    encoding of images/videos and position ID computation.

    The ``process()`` template method provides the standard pipeline:
    1. Route tokenization (prompt_token_ids vs text+mm_data vs messages)
    2. Encode text and multimodal items into outputs dict
    3. Pack outputs into final numpy arrays
    4. Write back results to request dict (via ``_write_back()`` hook)
    """

    # ------------------------------------------------------------------
    # Class attributes — subclasses override these
    # ------------------------------------------------------------------
    image_placeholder: str = ""
    video_placeholder: str = ""
    image_token_str: str = ""
    video_token_str: str = ""

    # Whether this processor supports the prompt_token_ids path
    _supports_prompt_token_ids: bool = False

    def __init__(self, tokenizer, image_processor, config=None, processor_kwargs=None):
        """Initialize the multimodal processor.

        Args:
            tokenizer: The tokenizer instance.
            image_processor: The image processor instance.
            config: Model config object (optional).
            processor_kwargs: Additional processor keyword arguments.
        """
        if processor_kwargs is None:
            processor_kwargs = {}

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.config = config

        # Special token IDs
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token_str)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_token_str)

        # Video params
        self.fps = processor_kwargs.get("video_fps", self._default_fps)
        self.min_frames = processor_kwargs.get("video_min_frames", self._default_min_frames)
        self.max_frames = processor_kwargs.get("video_max_frames", self._default_max_frames)
        self.target_frames = processor_kwargs.get("video_target_frames", self._default_target_frames)

        # Model-specific extra init
        self._init_extra(processor_kwargs)

    # ------------------------------------------------------------------
    # Default video parameters — subclasses override as class attrs
    # ------------------------------------------------------------------
    _default_fps: float = 2.0
    _default_min_frames: int = 4
    _default_max_frames: int = 768
    _default_target_frames: int = -1

    # ------------------------------------------------------------------
    # Template method: process()
    # ------------------------------------------------------------------
    def process(self, request, tokenizer, enable_processor_cache=False):
        """Template method: full multimodal processing pipeline.

        Args:
            request: The request dict containing messages/prompt/prompt_token_ids.
            tokenizer: The tokenizer to use for text encoding.
            enable_processor_cache: Whether to use ZMQ processor cache.

        Returns:
            dict: Packed multimodal outputs (input_ids, position_ids, images, etc.)
        """
        outputs = self._route_tokenization(request, enable_processor_cache)
        outputs = self._pack_outputs(outputs)
        self._write_back(request, outputs)
        return outputs

    # ------------------------------------------------------------------
    # Tokenization routing (common logic)
    # ------------------------------------------------------------------
    def _route_tokenization(self, request, enable_processor_cache=False):
        """Route to the appropriate tokenization path.

        Path A: prompt_token_ids already present
        Path B: prompt text + multimodal_data
        Path C: messages (needs chat template)
        """
        if request.get("prompt_token_ids") and self._supports_prompt_token_ids:
            return self._process_prompt_token_ids(request, enable_processor_cache)

        if request.get("prompt"):
            multimodal_data = request.get("multimodal_data") or {}
            images = multimodal_data.get("image", None)
            videos = multimodal_data.get("video", None)
            return self._text2ids(request["prompt"], images, videos)

        raise ValueError("MMProcessor requires 'prompt_token_ids' or 'prompt' in request")

    # ------------------------------------------------------------------
    # Text-to-IDs scanning loop
    # ------------------------------------------------------------------
    def _text2ids(self, text, images=None, videos=None, image_uuid=None, video_uuid=None):
        """Convert text with image/video placeholders into model inputs.

        Scans the text for image_placeholder and video_placeholder strings,
        replacing them with encoded multimodal tokens.
        """
        outputs = self._make_outputs()

        IMAGE_PLACEHOLDER = self.image_placeholder
        VIDEO_PLACEHOLDER = self.video_placeholder
        IMAGE_PLACEHOLDER_LEN = len(IMAGE_PLACEHOLDER)
        VIDEO_PLACEHOLDER_LEN = len(VIDEO_PLACEHOLDER)

        st, image_idx, video_idx = 0, 0, 0
        while st < len(text):
            image_pos = text.find(IMAGE_PLACEHOLDER, st)
            image_pos = len(text) if image_pos == -1 else image_pos
            video_pos = text.find(VIDEO_PLACEHOLDER, st)
            video_pos = len(text) if video_pos == -1 else video_pos
            ed = min(image_pos, video_pos)

            self._add_text(text[st:ed], outputs)
            if ed == len(text):
                break

            if ed == image_pos:
                image = images[image_idx]
                uuid = image_uuid[image_idx] if image_uuid else None
                if not isinstance(image, tuple):
                    self.add_image(image, outputs, uuid)
                else:
                    self.add_processed_image(image, outputs, uuid)
                image_idx += 1
                st = ed + IMAGE_PLACEHOLDER_LEN
            else:
                item = videos[video_idx]
                uuid = video_uuid[video_idx] if video_uuid else None
                if not isinstance(item, tuple):
                    if isinstance(item, dict):
                        frames, meta = self.load_video(item["video"], item)
                    else:
                        frames, meta = self.load_video(item, {})
                    self.add_video(frames, outputs, uuid, meta=meta)
                else:
                    self.add_processed_video(item, outputs, uuid)
                video_idx += 1
                st = ed + VIDEO_PLACEHOLDER_LEN

        return outputs

    def _add_text(self, tokens, outputs):
        """Add text tokens to outputs, delegating position logic to subclass."""
        if not tokens:
            return
        if isinstance(tokens, str):
            tokens_str = self.tokenizer.tokenize(tokens)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens_str)
        num_tokens = len(tokens)
        outputs["input_ids"].extend(tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * num_tokens)
        self.add_text_positions(outputs, num_tokens)

    # ------------------------------------------------------------------
    # Pack outputs
    # ------------------------------------------------------------------
    def _pack_outputs(self, outputs):
        """Convert intermediate outputs to final packed format."""
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
        outputs["mm_num_token_func"] = self.mm_num_tokens

        # Position IDs: delegate to subclass
        self.pack_position_ids(outputs)

        return outputs

    # ------------------------------------------------------------------
    # Write-back hook (default: overwrite prompt_token_ids)
    # ------------------------------------------------------------------
    def _write_back(self, request, outputs):
        """Write processed outputs back to the request dict.

        Default behavior: set request["prompt_token_ids"] = outputs["input_ids"].
        Subclasses (e.g. Ernie) override to preserve original prompt_token_ids.
        """
        request["prompt_token_ids"] = outputs["input_ids"].tolist()
        request["multimodal_inputs"] = outputs

    # ------------------------------------------------------------------
    # prompt_token_ids path
    # ------------------------------------------------------------------
    def _process_prompt_token_ids(self, request, enable_processor_cache=False):
        """Handle the prompt_token_ids tokenization path.

        Subclasses that support this path must implement ``prompt_token_ids2outputs``.
        """
        prompt_token_ids = request.get("prompt_token_ids", [])

        if not request.get("messages"):
            return self.prompt_token_ids2outputs(prompt_token_ids)

        # Extract mm items from messages for the prompt_token_ids path
        from fastdeploy.entrypoints.chat_utils import parse_chat_messages

        messages = parse_chat_messages(request.get("messages"))
        mm_items = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                content = [content]
            for item in content:
                if isinstance(item, dict) and item.get("type") in ["image", "video"]:
                    mm_items.append(item)

        return self.prompt_token_ids2outputs(prompt_token_ids, mm_items)

    # ------------------------------------------------------------------
    # Outputs initialisation
    # ------------------------------------------------------------------
    def _make_outputs(self) -> dict:
        """Create the mutable accumulator dict for encoding results.

        Subclasses override to add model-specific fields (e.g. fps, vit fields).
        """
        return {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "labels": [],
            "cur_position": 0,
            "video_cnt": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
            "mm_positions": [],
            "mm_hashes": [],
        }

    # ------------------------------------------------------------------
    # Cache utilities
    # ------------------------------------------------------------------
    @staticmethod
    def get_processor_cache(socket, mm_hashes):
        """Retrieve cached multimodal items via ZMQ."""
        req = pickle.dumps(mm_hashes)
        socket.send_multipart([b"", req])
        _, resp = socket.recv_multipart()
        mm_items = pickle.loads(resp)
        data_processor_logger.info(f"Get cache of mm_hashes: {mm_hashes}")
        return mm_items

    @staticmethod
    def update_processor_cache(socket, mm_hashes, mm_items):
        """Write processed multimodal items to the ZMQ cache."""
        req = pickle.dumps((mm_hashes, mm_items))
        socket.send_multipart([b"", req])
        data_processor_logger.info(f"Update cache of mm_hashes: {mm_hashes}")

    # ------------------------------------------------------------------
    # Abstract methods — subclasses MUST implement
    # ------------------------------------------------------------------
    @abstractmethod
    def add_image(self, img, outputs: dict, uuid, token_len=None):
        """Process a raw image and append results to *outputs*."""

    @abstractmethod
    def add_processed_image(self, img_cache, outputs: dict, uuid, token_len=None):
        """Append a pre-processed (cached) image to *outputs*."""

    @abstractmethod
    def add_video(self, frames, outputs: dict, uuid, token_len=None, meta=None):
        """Process video frames and append results to *outputs*."""

    @abstractmethod
    def add_processed_video(self, frames_cache, outputs: dict, uuid, token_len=None):
        """Append a pre-processed (cached) video to *outputs*."""

    @abstractmethod
    def load_video(self, url, item: dict) -> Tuple[Any, dict]:
        """Decode a video from *url* and return ``(frames, meta)``."""

    @abstractmethod
    def add_text_positions(self, outputs: dict, num_tokens: int):
        """Append text position IDs to *outputs*."""

    @abstractmethod
    def append_completion_tokens(self, multimodal_inputs: dict, completion_token_ids):
        """Append completion token IDs (and their positions) to *multimodal_inputs*."""

    @staticmethod
    @abstractmethod
    def mm_num_tokens(grid_thw):
        """Return the number of multimodal tokens for a given grid_thw."""

    @abstractmethod
    def pack_position_ids(self, outputs: dict):
        """Convert intermediate position ID lists into final packed format."""

    # ------------------------------------------------------------------
    # Optional methods — subclasses override only when needed
    # ------------------------------------------------------------------
    def _init_extra(self, processor_kwargs: dict):
        """Model-specific extra initialisation (called once after ``__init__``)."""

    def prompt_token_ids2outputs(self, prompt_token_ids, mm_items=None) -> dict:
        """Build outputs dict from pre-tokenised ``prompt_token_ids``.

        Only subclasses with ``_supports_prompt_token_ids = True`` need to implement.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support prompt_token_ids path")

    def get_mm_max_tokens_per_item(self, seq_len: int) -> Optional[Dict[str, int]]:
        """Per-modality max token counts for the scheduler. ``None`` = not applicable."""
        return None
