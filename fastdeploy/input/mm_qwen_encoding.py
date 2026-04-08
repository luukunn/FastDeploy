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

"""Qwen-family VL encoding class.

Covers Qwen-VL, Qwen3-VL, and PaddleOCR-VL.  Internal differences are
resolved via ``self.p.cfg`` (the ``MMModelConfig`` instance).

The ``QwenEncoding`` instance is held by the unified ``MultiModalProcessor``
via composition:

    self.enc = QwenEncoding(self)
    self.enc.add_image(img, outputs, uuid)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import zmq
from PIL import Image

from fastdeploy.engine.request import ImagePosition
from fastdeploy.entrypoints.chat_utils import parse_chat_messages
from fastdeploy.input.mm_model_config import QWEN3_VL
from fastdeploy.input.utils import IDS_TYPE_FLAG
from fastdeploy.input.video_utils import (
    read_video_decord,
    sample_frames_paddleocr,
    sample_frames_qwen,
)
from fastdeploy.multimodal.hasher import MultimodalHasher

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28


class QwenEncoding:
    """Qwen-family VL token encoding logic.

    Holds a reference to the parent ``MultiModalProcessor`` as ``self.p``
    to access shared state (tokenizer, image_processor, cfg, …).

    Internal branching on ``self.p.model_type`` or ``self.p.cfg`` handles
    the small differences between qwen_vl, qwen3_vl, and paddleocr_vl.
    """

    def __init__(self, processor):
        self.p = processor

    @staticmethod
    def mm_num_tokens(grid_thw: "list | list[list[int]] | np.ndarray | paddle.Tensor") -> "int | list[int]":
        """Calculate the number of tokens for multimodal input.

        For qwen family: always t*h*w//4.
        """
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

    def add_image(self, img, outputs: Dict, uuid: Optional[str], token_len: Optional[int] = None) -> None:
        p = self.p
        ret = p.image_processor.preprocess(images=[img.convert("RGB")])
        num_tokens = ret["grid_thw"].prod() // p.image_processor.merge_size**2
        grid_thw = ret["grid_thw"].tolist()
        if token_len is not None and token_len != num_tokens:
            raise ValueError("image tokens num not match the size")

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([p.image_token_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)
        outputs["num_input_image_tokens"] += int(num_tokens)

        outputs["images"].append(ret["pixel_values"])
        if not uuid:
            outputs["mm_hashes"].append(MultimodalHasher.hash_features(ret["pixel_values"]))
        else:
            outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(grid_thw)
        outputs["image_type_ids"].append(0)

        t, h, w = grid_thw
        pos_ids = self.compute_vision_positions(outputs["cur_position"], t, h, w, 0)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

        outputs["fps"].append(0)

        if p.cfg.has_vit_fields:
            numel = h * w
            outputs["vit_seqlen"].append(numel)
            outputs["vit_position_ids"].append(np.arange(numel) % numel)

    def add_processed_image(
        self, img_cache: Tuple[np.ndarray, dict], outputs: Dict, uuid: str, token_len: Optional[int] = None
    ) -> None:
        p = self.p
        img, meta = img_cache
        num_tokens = img.shape[0] // p.image_processor.merge_size**2
        if token_len is not None and token_len != num_tokens:
            raise ValueError("image tokens num not match the size")

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([p.image_token_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)

        _, h, w = meta["thw"]
        pos_ids = self.compute_vision_positions(outputs["cur_position"], 1, h, w, 0)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

        outputs["images"].append(img)
        outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(np.array([[1, h, w]]))
        outputs["image_type_ids"].append(0)

        outputs["fps"].append(0)

    def add_video(
        self, frames, meta: Dict, outputs: Dict, uuid: Optional[str], token_len: Optional[int] = None
    ) -> None:
        p = self.p

        preprocess_kwargs = {}
        if p.model_type == QWEN3_VL:
            preprocess_kwargs["min_pixels"] = VIDEO_MIN_PIXELS
            preprocess_kwargs["max_pixels"] = VIDEO_MAX_PIXELS

        ret = p.image_processor.preprocess(images=frames, **preprocess_kwargs)

        grid_thw_key = p.cfg.grid_thw_key
        num_tokens = ret[grid_thw_key].prod() // p.image_processor.merge_size**2
        grid_thw = ret[grid_thw_key].tolist()
        if token_len is not None and token_len != num_tokens:
            raise ValueError("video tokens num not match the size")

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        if p.cfg.video_fill_uses_image_token:
            fill_token_id = p.image_token_id
        else:
            fill_token_id = p.video_token_id
        outputs["input_ids"].extend([fill_token_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)
        outputs["num_input_video_tokens"] += int(num_tokens)

        outputs["images"].append(ret["pixel_values"])
        if not uuid:
            outputs["mm_hashes"].append(MultimodalHasher.hash_features(ret["pixel_values"]))
        else:
            outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(grid_thw)
        outputs["image_type_ids"].extend([1] * grid_thw[0])

        fps = meta["fps"]
        second_per_grid_t = p.temporal_conv_size / fps
        t, h, w = grid_thw
        pos_ids = self.compute_vision_positions(outputs["cur_position"], t, h, w, second_per_grid_t)

        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

        outputs["fps"].append(fps)

        # paddleocr vit fields
        if p.cfg.has_vit_fields:
            numel = h * w
            outputs["vit_seqlen"].append(numel)
            outputs["vit_position_ids"].append(np.arange(numel) % numel)

    def add_processed_video(
        self, frames_cache: Tuple[np.ndarray, dict], outputs: Dict, uuid: str, token_len: Optional[int] = None
    ) -> None:
        p = self.p
        frames, meta = frames_cache
        num_tokens = frames.shape[0] // p.image_processor.merge_size**2
        if token_len is not None and token_len != num_tokens:
            raise ValueError("video tokens num not match the size")

        t, h, w = meta["thw"]
        outputs["images"].append(frames)
        outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(np.array([[t, h, w]]))

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        if p.cfg.video_fill_uses_image_token:
            fill_token_id = p.image_token_id
        else:
            fill_token_id = p.video_token_id
        outputs["input_ids"].extend([fill_token_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)
        outputs["image_type_ids"].extend([1] * t)

        fps = meta["fps"]
        second_per_grid_t = p.temporal_conv_size / fps
        pos_ids = self.compute_vision_positions(outputs["cur_position"], t, h, w, second_per_grid_t)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

        outputs["fps"].append(fps)

    def load_video(self, url: str, item: Dict) -> Tuple[np.ndarray, Dict]:
        """Load and preprocess video into frames.

        Returns (frames_ndarray, meta_dict).
        """
        p = self.p
        reader, meta, _ = read_video_decord(url, save_to_disk=False)

        fps = item.get("fps", p.fps)
        num_frames = item.get("target_frames", p.target_frames)

        frame_indices = list(range(meta["num_of_frame"]))
        if fps > 0 or num_frames > 0:
            min_frames = item.get("min_frames", p.min_frames)
            max_frames = item.get("max_frames", p.max_frames)

            if p.cfg.sample_frames_variant == "paddleocr":
                sample_fn = sample_frames_paddleocr
                frame_factor = p.temporal_conv_size
                sample_kwargs = dict(fps=fps, num_frames=num_frames)
            else:
                sample_fn = sample_frames_qwen
                frame_factor = p.frame_factor
                sample_kwargs = dict(
                    fps=-1 if num_frames > 0 else fps,
                    num_frames=num_frames,
                )

            frame_indices = sample_fn(
                frame_factor=frame_factor,
                min_frames=min_frames,
                max_frames=max_frames,
                metadata=meta,
                **sample_kwargs,
            )

            meta["num_of_frame"] = len(frame_indices)
            if fps is not None:
                meta["fps"] = fps
                meta["duration"] = len(frame_indices) / fps
            else:
                meta["fps"] = len(frame_indices) / meta["duration"]

        frames = []
        for idx in frame_indices:
            frame = reader[idx].asnumpy()
            image = Image.fromarray(frame, "RGB")
            frames.append(image)
        frames = np.stack([np.array(f.convert("RGB")) for f in frames], axis=0)

        return frames, meta

    def add_text_positions(self, outputs: Dict, num_tokens: int) -> None:
        """Append position IDs for text tokens (qwen: ndarray (3, N))."""
        pos_ids = self._compute_text_positions(outputs["cur_position"], num_tokens)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

    def _compute_text_positions(self, start_pos: int, num_tokens: int) -> np.ndarray:
        """Generate 3D positional embeddings for text tokens.

        Returns np.ndarray of shape (3, num_tokens).
        """
        text_array = np.arange(num_tokens).reshape(1, -1)
        text_index = np.broadcast_to(text_array, (3, num_tokens))
        position = text_index + start_pos
        return position

    def append_completion_tokens(self, multimodal_inputs: Dict, completion_token_ids: list) -> None:
        """Append completion tokens to existing outputs (qwen variant)."""
        num_tokens = len(completion_token_ids)
        multimodal_inputs["input_ids"].extend(completion_token_ids)
        multimodal_inputs["token_type_ids"].extend([0] * num_tokens)

        pos_ids = self._compute_text_positions(multimodal_inputs["cur_position"], num_tokens)
        multimodal_inputs["position_ids"].append(pos_ids)
        multimodal_inputs["cur_position"] += num_tokens

    def compute_vision_positions(self, start_pos: int, t: int, h: int, w: int, second_per_grid_t: float) -> np.ndarray:
        """Generate 3D position IDs for visual inputs.

        Returns np.ndarray of shape (3, t*h'*w') where h'=h//merge_size, w'=w//merge_size.
        """
        p = self.p
        h //= p.spatial_conv_size
        w //= p.spatial_conv_size

        tn = np.arange(t).reshape(-1, 1)
        tn = np.broadcast_to(tn, (t, h * w))
        tn = tn * int(second_per_grid_t) * p.tokens_per_second
        t_index = tn.flatten()

        hn = np.arange(h).reshape(1, -1, 1)
        h_index = np.broadcast_to(hn, (t, h, w)).flatten()

        wn = np.arange(w).reshape(1, 1, -1)
        w_index = np.broadcast_to(wn, (t, h, w)).flatten()

        position = np.stack([t_index, h_index, w_index]) + start_pos
        return position

    def prompt_token_ids2outputs(
        self, request: Dict[str, Any], tgts: List[str] = None
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray], None]]:
        """Build outputs from pre-tokenized prompt_token_ids (qwen3 variant).

        Scans for contiguous runs of image_token_id.
        """
        p = self.p
        outputs = p._make_outputs()
        prompt_token_ids = request.get("prompt_token_ids", [])
        prompt_token_ids_len = len(prompt_token_ids)

        if not request.get("messages"):
            p._add_text(prompt_token_ids, outputs)
            return outputs

        messages = parse_chat_messages(request.get("messages"))
        mm_items = []
        for msg in messages:
            role = msg.get("role")
            assert role in p.role_prefixes, f"Unsupported role: {role}"
            content = msg.get("content")
            if not isinstance(content, list):
                content = [content]
            for item in content:
                if item.get("type") in ["image", "video"]:
                    mm_items.append(item)

        missing_hashes, missing_idx = [], []
        for idx, item in enumerate(mm_items):
            if not item.get("data"):
                missing_hashes.append(item.get("uuid"))
                missing_idx.append(idx)

        if len(missing_hashes) > 0 and not p.enable_processor_cache:
            raise ValueError("Missing items cannot be retrieved without processor cache.")

        dealer = None
        if p.enable_processor_cache:
            context = zmq.Context()
            dealer = context.socket(zmq.DEALER)
            dealer.connect("ipc:///dev/shm/processor_cache.ipc")

            missing_items = p.get_processor_cache(dealer, missing_hashes)
            for idx in range(len(missing_items)):
                if not missing_items[idx]:
                    raise ValueError(f"Missing item {idx} not found in processor cache")
                mm_items[missing_idx[idx]]["data"] = missing_items[idx]

        st, mm_idx = 0, 0
        while st < prompt_token_ids_len:
            if prompt_token_ids[st] != p.image_token_id:
                cur_idx = st
                while cur_idx < prompt_token_ids_len and prompt_token_ids[cur_idx] != p.image_token_id:
                    cur_idx += 1
                p._add_text(prompt_token_ids[st:cur_idx], outputs)
                st = cur_idx
                continue

            if mm_idx >= len(mm_items):
                raise ValueError("prompt token ids has more multimodal placeholder than in messages")

            cur_idx = st
            while cur_idx < prompt_token_ids_len and prompt_token_ids[cur_idx] == p.image_token_id:
                cur_idx += 1

            item = mm_items[mm_idx]
            uuid = item.get("uuid")
            token_len = cur_idx - st
            if item.get("type") == "image":
                image = item.get("data")
                if not isinstance(image, tuple):
                    self.add_image(image, outputs, uuid, token_len)
                else:
                    self.add_processed_image(image, outputs, uuid, token_len)
            elif item.get("type") == "video":
                video = item.get("data")
                if not isinstance(video, tuple):
                    if isinstance(video, dict):
                        frames, meta = self.load_video(video["video"], video)
                    else:
                        frames, meta = self.load_video(video, {})
                    self.add_video(frames, meta, outputs, uuid, token_len)
                else:
                    self.add_processed_video(video, outputs, uuid, token_len)
            else:
                raise ValueError(f"Unsupported multimodal type: {item.get('type')}")
            mm_idx += 1
            st = cur_idx

        if mm_idx != len(mm_items):
            raise ValueError("number of multimodal items does not match prompt token ids")

        if p.enable_processor_cache:
            missing_idx_set = set(missing_idx)
            hashes_to_cache, items_to_cache = [], []
            for idx in range(len(mm_items)):
                if idx in missing_idx_set:
                    continue
                meta = {}
                grid_thw = np.asarray(outputs["grid_thw"][idx])
                if grid_thw.ndim > 1:
                    t, h, w = grid_thw[0]
                else:
                    t, h, w = grid_thw
                meta["thw"] = (int(t), int(h), int(w))
                meta["fps"] = outputs["fps"][idx]
                hashes_to_cache.append(outputs["mm_hashes"][idx])
                items_to_cache.append((outputs["images"][idx], meta))
            if hashes_to_cache:
                p.update_processor_cache(dealer, hashes_to_cache, items_to_cache)

        return outputs

    def pack_position_ids(self, outputs: Dict) -> None:
        """Pack position_ids as np.ndarray via concatenate+transpose (qwen format)."""
        p = self.p
        outputs["position_ids"] = np.concatenate(outputs["position_ids"], axis=1, dtype=np.int64)
        outputs["image_patch_id"] = p.image_token_id
        outputs["video_patch_id"] = p.video_token_id
        outputs["position_ids"] = outputs["position_ids"].transpose(1, 0)
