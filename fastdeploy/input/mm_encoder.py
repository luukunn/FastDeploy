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

"""Multimodal encoding mixin for all VL model types.

Contains the core encoding methods (text2ids, request2ids,
prompt_token_ids2outputs) and all helper methods they depend on
(_add_text, _add_image, _add_video, position computation, video loading,
cache helpers).

This mixin is mixed into ``MultiModalProcessor`` via multiple inheritance.
It reads instance attributes (model_type, encoding_tokenizer,
image_processor, etc.) set by the host class's ``__init__``.
"""

import copy
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import zmq
from PIL import Image

from fastdeploy.engine.request import ImagePosition
from fastdeploy.entrypoints.chat_utils import parse_chat_messages
from fastdeploy.input.utils import IDS_TYPE_FLAG
from fastdeploy.input.video_utils import (
    read_frames_decord_ernie,
    read_video_decord,
    render_frame_timestamp,
    sample_frames_paddleocr,
    sample_frames_qwen,
)
from fastdeploy.multimodal.hasher import MultimodalHasher
from fastdeploy.utils import data_processor_logger

# ---- Model type constants (canonical definitions) ----
QWEN_VL = "qwen_vl"
QWEN3_VL = "qwen3_vl"
PADDLEOCR_VL = "paddleocr_vl"
ERNIE4_5_VL = "ernie4_5_vl"

_QWEN_FAMILY = {QWEN_VL, QWEN3_VL, PADDLEOCR_VL}

# Qwen3 video pixel bounds
_QWEN3_VIDEO_MIN_PIXELS = 128 * 28 * 28
_QWEN3_VIDEO_MAX_PIXELS = 768 * 28 * 28


class MultiModalEncoderMixin:
    """Mixin providing multimodal encoding methods for MultiModalProcessor.

    This class is not meant to be instantiated directly. It relies on
    attributes set by the host class (e.g. ``self.model_type``,
    ``self.encoding_tokenizer``, ``self.image_processor``, etc.).
    """

    def _get_processor_cache(self, socket, mm_hashes: list) -> list:
        req = pickle.dumps(mm_hashes)
        socket.send_multipart([b"", req])
        _, resp = socket.recv_multipart()
        mm_items = pickle.loads(resp)
        data_processor_logger.info(f"Get cache of mm_hashes: {mm_hashes}")
        return mm_items

    def _update_processor_cache(self, socket, mm_hashes: list, mm_items):
        req = pickle.dumps((mm_hashes, mm_items))
        socket.send_multipart([b"", req])
        data_processor_logger.info(f"Update cache of mm_hashes: {mm_hashes}")

    def _make_outputs(self) -> dict:
        """Create a fresh outputs dict for encoding."""
        outputs = {
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
        if self.model_type in _QWEN_FAMILY:
            outputs["fps"] = []
        if self.model_type == PADDLEOCR_VL:
            outputs["vit_seqlen"] = []
            outputs["vit_position_ids"] = []
        return outputs

    def text2ids(self, text, images=None, videos=None, image_uuid=None, video_uuid=None):
        outputs = self._make_outputs()

        if self.model_type == ERNIE4_5_VL:
            img_ph = "<|image@placeholder|>"
            vid_ph = "<|video@placeholder|>"
        elif self.model_type == PADDLEOCR_VL:
            img_ph = self.image_token
            vid_ph = self.video_token
        else:
            img_ph = "<|image_pad|>"
            vid_ph = "<|video_pad|>"

        img_ph_len = len(img_ph)
        vid_ph_len = len(vid_ph)

        st, image_idx, video_idx = 0, 0, 0
        while st < len(text):
            image_pos = text.find(img_ph, st)
            image_pos = len(text) if image_pos == -1 else image_pos
            video_pos = text.find(vid_ph, st)
            video_pos = len(text) if video_pos == -1 else video_pos
            ed = min(image_pos, video_pos)

            self._add_text(text[st:ed], outputs)
            if ed == len(text):
                break

            if ed == image_pos:
                image = images[image_idx]
                uuid = image_uuid[image_idx] if image_uuid else None
                if not isinstance(image, tuple):
                    self._add_image(image, outputs, uuid)
                else:
                    self._add_processed_image(image, outputs, uuid)
                image_idx += 1
                st = ed + img_ph_len
            else:
                item = videos[video_idx]
                uuid = video_uuid[video_idx] if video_uuid else None
                if not isinstance(item, tuple):
                    if isinstance(item, dict):
                        frames_result = self._load_and_process_video(item["video"], item)
                    else:
                        frames_result = self._load_and_process_video(item, {})
                    if self.model_type == ERNIE4_5_VL:
                        self._add_video(frames_result, outputs, uuid)
                    else:
                        frames, meta = frames_result
                        self._add_video(frames, outputs, uuid, meta=meta)
                else:
                    self._add_processed_video(item, outputs, uuid)
                video_idx += 1
                st = ed + vid_ph_len

        return outputs

    def _extract_mm_items(self, request):
        """Parse messages and extract multimodal items, handling cache retrieval."""
        messages = parse_chat_messages(request.get("messages"))
        mm_items = []
        for msg in messages:
            role = msg.get("role")
            assert role in self.role_prefixes, f"Unsupported role: {role}"
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

        if len(missing_hashes) > 0 and not self.enable_processor_cache:
            raise ValueError("Missing items cannot be retrieved without processor cache.")

        dealer = None
        if self.enable_processor_cache:
            context = zmq.Context()
            dealer = context.socket(zmq.DEALER)
            dealer.connect("ipc:///dev/shm/processor_cache.ipc")

            missing_items = self._get_processor_cache(dealer, missing_hashes)
            for idx in range(len(missing_items)):
                if not missing_items[idx]:
                    raise ValueError(f"Missing item {idx} not found in processor cache")
                mm_items[missing_idx[idx]]["data"] = missing_items[idx]

        images, videos = [], []
        image_uuid, video_uuid = [], []
        for item in mm_items:
            if item.get("type") == "image":
                images.append(item["data"])
                image_uuid.append(item["uuid"])
            elif item.get("type") == "video":
                videos.append(item["data"])
                video_uuid.append(item["uuid"])
            else:
                raise ValueError(f"Unsupported multimodal type: {item.get('type')}")

        return images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items

    def request2ids(self, request):
        images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items = self._extract_mm_items(request)

        if self.encoding_tokenizer.chat_template is None:
            raise ValueError("This model does not support chat template.")

        chat_template_kwargs = request.get("chat_template_kwargs", {})
        if self.model_type == ERNIE4_5_VL:
            prompt = self.encoding_tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=request.get("add_generation_prompt", True),
                **chat_template_kwargs,
            )
        else:
            messages = parse_chat_messages(request.get("messages"))
            prompt = self.encoding_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=request.get("add_generation_prompt", True),
                **chat_template_kwargs,
            )
        request["prompt_tokens"] = prompt

        outputs = self.text2ids(prompt, images, videos, image_uuid, video_uuid)

        if self.enable_processor_cache:
            self._update_cache_after_encoding(dealer, missing_idx, mm_items, outputs)

        return outputs

    def prompt_token_ids2outputs(self, request):
        """Dispatch to model-type-specific prompt_token_ids scanner."""
        if self.model_type == ERNIE4_5_VL:
            return self._prompt_token_ids2outputs_ernie(request)
        else:
            return self._prompt_token_ids2outputs_qwen3(request)

    def _prompt_token_ids2outputs_qwen3(self, request):
        outputs = self._make_outputs()
        prompt_token_ids = request.get("prompt_token_ids", [])
        prompt_token_ids_len = len(prompt_token_ids)

        if not request.get("messages"):
            self._add_text(prompt_token_ids, outputs)
            return outputs

        images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items = self._extract_mm_items(request)

        st, mm_idx = 0, 0
        while st < prompt_token_ids_len:
            if prompt_token_ids[st] != self.image_token_id:
                cur_idx = st
                while cur_idx < prompt_token_ids_len and prompt_token_ids[cur_idx] != self.image_token_id:
                    cur_idx += 1
                self._add_text(prompt_token_ids[st:cur_idx], outputs)
                st = cur_idx
                continue

            if mm_idx >= len(mm_items):
                raise ValueError("prompt token ids has more multimodal placeholder than in messages")

            cur_idx = st
            while cur_idx < prompt_token_ids_len and prompt_token_ids[cur_idx] == self.image_token_id:
                cur_idx += 1

            item = mm_items[mm_idx]
            uuid = item.get("uuid")
            token_len = cur_idx - st
            if item.get("type") == "image":
                image = item.get("data")
                if not isinstance(image, tuple):
                    self._add_image(image, outputs, uuid, token_len)
                else:
                    self._add_processed_image(image, outputs, uuid, token_len)
            elif item.get("type") == "video":
                video = item.get("data")
                if not isinstance(video, tuple):
                    if isinstance(video, dict):
                        frames, meta = self._load_and_process_video(video["video"], video)
                    else:
                        frames, meta = self._load_and_process_video(video, {})
                    self._add_video(frames, outputs, uuid, token_len, meta=meta)
                else:
                    self._add_processed_video(video, outputs, uuid, token_len)
            else:
                raise ValueError(f"Unsupported multimodal type: {item.get('type')}")
            mm_idx += 1
            st = cur_idx

        if mm_idx != len(mm_items):
            raise ValueError("number of multimodal items does not match prompt token ids")

        if self.enable_processor_cache:
            self._update_cache_after_encoding(dealer, missing_idx, mm_items, outputs)

        return outputs

    def _prompt_token_ids2outputs_ernie(self, request):
        outputs = self._make_outputs()
        prompt_token_ids = request.get("prompt_token_ids", [])
        prompt_token_ids_len = len(prompt_token_ids)

        if not request.get("messages"):
            outputs["input_ids"].extend(prompt_token_ids)
            outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * prompt_token_ids_len)
            for i in range(prompt_token_ids_len):
                outputs["position_ids"].append([i] * 3)
            outputs["cur_position"] += prompt_token_ids_len
            return outputs

        images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items = self._extract_mm_items(request)

        st, image_idx, video_idx = 0, 0, 0
        while st < prompt_token_ids_len:
            cur_token_id = prompt_token_ids[st]
            if cur_token_id == self.image_start_id:
                if image_idx >= len(images):
                    raise ValueError("prompt token ids has more image placeholder than in messages")
                outputs["input_ids"].extend([cur_token_id])
                outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]])
                outputs["position_ids"].append([outputs["cur_position"]] * 3)
                outputs["cur_position"] += 1
                st += 1
                cur_idx = st
                while cur_idx < prompt_token_ids_len and prompt_token_ids[cur_idx] != self.image_end_id:
                    cur_idx += 1
                if cur_idx >= prompt_token_ids_len:
                    raise ValueError("image token ids not complete")
                image = images[image_idx]
                uuid = image_uuid[image_idx] if image_uuid else None
                token_len = cur_idx - st
                if not isinstance(image, tuple):
                    self._add_image(image, outputs, uuid, token_len)
                else:
                    self._add_processed_image(image, outputs, uuid, token_len)
                image_idx += 1
                st = cur_idx
            elif cur_token_id == self.video_start_id:
                if video_idx >= len(videos):
                    raise ValueError("prompt token ids has more video placeholder than in messages")
                outputs["input_ids"].extend([cur_token_id])
                outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]])
                outputs["position_ids"].append([outputs["cur_position"]] * 3)
                outputs["cur_position"] += 1
                st += 1
                cur_idx = st
                while cur_idx < prompt_token_ids_len and prompt_token_ids[cur_idx] != self.video_end_id:
                    cur_idx += 1
                if cur_idx >= prompt_token_ids_len:
                    raise ValueError("video token ids not complete")
                video = videos[video_idx]
                uuid = video_uuid[video_idx] if video_uuid else None
                token_len = cur_idx - st
                if not isinstance(video, tuple):
                    if isinstance(video, dict):
                        frames = self._load_and_process_video(video["video"], video)
                    else:
                        frames = self._load_and_process_video(video, {})
                    self._add_video(frames, outputs, uuid, token_len)
                else:
                    self._add_processed_video(video, outputs, uuid, token_len)
                video_idx += 1
                st = cur_idx
            else:
                outputs["input_ids"].extend([cur_token_id])
                outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]])
                outputs["position_ids"].append([outputs["cur_position"]] * 3)
                outputs["cur_position"] += 1
                st += 1

        if image_idx != len(images):
            raise ValueError("number of images does not match")
        if video_idx != len(videos):
            raise ValueError("number of videos does not match")

        if self.enable_processor_cache:
            self._update_cache_after_encoding(dealer, missing_idx, mm_items, outputs)

        return outputs

    def _update_cache_after_encoding(self, dealer, missing_idx, mm_items, outputs):
        missing_idx = set(missing_idx)
        hashes_to_cache, items_to_cache = [], []
        for idx in range(len(mm_items)):
            if idx in missing_idx:
                continue
            meta = {}
            grid_thw = np.asarray(outputs["grid_thw"][idx])
            if grid_thw.ndim > 1:
                t, h, w = grid_thw[0]
            else:
                t, h, w = grid_thw
            meta["thw"] = (int(t), int(h), int(w))
            if self.model_type in _QWEN_FAMILY:
                meta["fps"] = outputs["fps"][idx]
            hashes_to_cache.append(outputs["mm_hashes"][idx])
            items_to_cache.append((outputs["images"][idx], meta))
        if hashes_to_cache:
            self._update_processor_cache(dealer, hashes_to_cache, items_to_cache)

    def _add_text(self, tokens, outputs: Dict) -> None:
        if not tokens:
            return None

        if isinstance(tokens, str):
            tokens_str = self.encoding_tokenizer.tokenize(tokens)
            tokens = self.encoding_tokenizer.convert_tokens_to_ids(tokens_str)

        num_tokens = len(tokens)
        outputs["input_ids"].extend(tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * num_tokens)

        if self.model_type == ERNIE4_5_VL:
            start = outputs["cur_position"]
            for i in range(num_tokens):
                outputs["position_ids"].append([start + i] * 3)
            outputs["cur_position"] += num_tokens
        else:
            pos_ids = self._compute_text_positions(outputs["cur_position"], num_tokens)
            outputs["position_ids"].append(pos_ids)
            outputs["cur_position"] = pos_ids.max() + 1

    @staticmethod
    def _compute_text_positions(start_pos: int, num_tokens: int) -> np.ndarray:
        """Generate 3D positional embeddings for text tokens — qwen family."""
        text_array = np.arange(num_tokens).reshape(1, -1)
        text_index = np.broadcast_to(text_array, (3, num_tokens))
        return text_index + start_pos

    def _add_image(self, img, outputs: Dict, uuid: Optional[str], token_len=None) -> None:
        if self.model_type == ERNIE4_5_VL:
            self._add_image_ernie(img, outputs, uuid, token_len)
        else:
            self._add_image_qwen(img, outputs, uuid, token_len)

    def _add_image_qwen(self, img, outputs: Dict, uuid: Optional[str], token_len=None) -> None:
        ret = self.image_processor.preprocess(images=[img.convert("RGB")])
        num_tokens = ret["grid_thw"].prod() // self.image_processor.merge_size**2
        grid_thw = ret["grid_thw"].tolist()
        if token_len is not None and token_len != num_tokens:
            raise ValueError("image tokens num not match the size")

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self.image_token_id] * num_tokens)
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
        pos_ids = self._compute_vision_positions_qwen(outputs["cur_position"], t, h, w, 0)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

        outputs["fps"].append(0)

        if self.model_type == PADDLEOCR_VL:
            numel = h * w
            outputs["vit_seqlen"].append(numel)
            outputs["vit_position_ids"].append(np.arange(numel) % numel)

    def _add_image_ernie(self, img, outputs: Dict, uuid: Optional[str], token_len=None) -> None:
        from paddleformers.transformers.image_utils import ChannelDimension

        patches_h, patches_w = self.image_processor.get_smarted_resize(
            img.height,
            img.width,
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
        )[1]
        num_tokens = (patches_h * patches_w) // (self.spatial_conv_size**2)
        if token_len and token_len != num_tokens:
            raise ValueError("image tokens num not match the size")

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)
        outputs["num_input_image_tokens"] += num_tokens

        pos_ids = self._compute_3d_positions_ernie(1, patches_h, patches_w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

        ret = self.image_processor.preprocess(
            images=[img.convert("RGB")],
            do_normalize=False,
            do_rescale=False,
            predetermined_grid_thw=np.array([[patches_h, patches_w]]),
            do_convert_rgb=True,
            input_data_format=ChannelDimension.LAST,
        )
        outputs["images"].append(ret["pixel_values"])
        if not uuid:
            outputs["mm_hashes"].append(MultimodalHasher.hash_features(ret["pixel_values"]))
        else:
            outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(ret["image_grid_thw"])
        outputs["image_type_ids"].append(0)

    def _add_processed_image(
        self, img_cache: Tuple[np.ndarray, dict], outputs: Dict, uuid: str, token_len=None
    ) -> None:
        img, meta = img_cache
        if self.model_type == ERNIE4_5_VL:
            num_tokens = img.shape[0] // (self.spatial_conv_size**2)
        else:
            num_tokens = img.shape[0] // self.image_processor.merge_size**2
        if token_len is not None and token_len != num_tokens:
            raise ValueError("image tokens num not match the size")

        fill_id = self.image_patch_id
        if self.model_type in (QWEN_VL, QWEN3_VL):
            fill_id = self.image_token_id

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([fill_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)

        _, h, w = meta["thw"]
        if self.model_type == ERNIE4_5_VL:
            pos_ids = self._compute_3d_positions_ernie(1, h, w, outputs["cur_position"])
            outputs["position_ids"].extend(pos_ids)
            outputs["cur_position"] = np.max(pos_ids) + 1
        else:
            pos_ids = self._compute_vision_positions_qwen(outputs["cur_position"], 1, h, w, 0)
            outputs["position_ids"].append(pos_ids)
            outputs["cur_position"] = pos_ids.max() + 1

        outputs["images"].append(img)
        outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(np.array([[1, h, w]]))
        outputs["image_type_ids"].append(0)

        if self.model_type in _QWEN_FAMILY:
            outputs["fps"].append(0)

    def _add_video(self, frames, outputs: Dict, uuid: Optional[str], token_len=None, *, meta=None) -> None:
        if self.model_type == ERNIE4_5_VL:
            self._add_video_ernie(frames, outputs, uuid, token_len)
        else:
            self._add_video_qwen(frames, meta, outputs, uuid, token_len)

    def _add_video_qwen(self, frames, meta: Dict, outputs: Dict, uuid: Optional[str], token_len=None) -> None:
        preprocess_kwargs = {}
        if self.model_type == QWEN3_VL:
            preprocess_kwargs["min_pixels"] = _QWEN3_VIDEO_MIN_PIXELS
            preprocess_kwargs["max_pixels"] = _QWEN3_VIDEO_MAX_PIXELS

        ret = self.image_processor.preprocess(images=frames, **preprocess_kwargs)

        grid_thw_key = "image_grid_thw" if self.model_type == PADDLEOCR_VL else "grid_thw"
        num_tokens = ret[grid_thw_key].prod() // self.image_processor.merge_size**2
        grid_thw = ret[grid_thw_key].tolist()
        if token_len is not None and token_len != num_tokens:
            raise ValueError("video tokens num not match the size")

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self._video_fill_token_id] * num_tokens)
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
        second_per_grid_t = self.temporal_conv_size / fps
        t, h, w = grid_thw
        pos_ids = self._compute_vision_positions_qwen(outputs["cur_position"], t, h, w, second_per_grid_t)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

        outputs["fps"].append(fps)

        if self.model_type == PADDLEOCR_VL:
            numel = h * w
            outputs["vit_seqlen"].append(numel)
            outputs["vit_position_ids"].append(np.arange(numel) % numel)

    def _add_video_ernie(self, frames: List[Image.Image], outputs: Dict, uuid: Optional[str], token_len=None) -> None:
        from paddleformers.transformers.image_utils import ChannelDimension

        patches_h, patches_w = self.image_processor.get_smarted_resize(
            frames[0].height,
            frames[0].width,
            min_pixels=self.video_min_pixels,
            max_pixels=self.video_max_pixels,
        )[1]
        num_frames = len(frames)
        num_tokens = (num_frames * patches_h * patches_w) // (self.spatial_conv_size**2 * self.temporal_conv_size)
        if token_len and num_tokens != token_len:
            raise ValueError("video tokens num not match the size")

        pixel_stack = np.stack([np.array(f.convert("RGB")) for f in frames], axis=0)
        ret = self.image_processor.preprocess(
            images=None,
            videos=pixel_stack,
            do_normalize=False,
            do_rescale=False,
            predetermined_grid_thw=np.array([[patches_h, patches_w]] * num_frames),
            do_convert_rgb=True,
            input_data_format=ChannelDimension.LAST,
        )
        outputs["images"].append(ret["pixel_values_videos"])
        if not uuid:
            outputs["mm_hashes"].append(MultimodalHasher.hash_features(ret["pixel_values_videos"]))
        else:
            outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(ret["video_grid_thw"])
        outputs["image_type_ids"].extend([1] * num_frames)

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)
        outputs["num_input_video_tokens"] += num_tokens

        pos_ids = self._compute_3d_positions_ernie(num_frames, patches_h, patches_w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

    def _add_processed_video(
        self, frames_cache: Tuple[np.ndarray, dict], outputs: Dict, uuid: str, token_len=None
    ) -> None:
        frames, meta = frames_cache

        if self.model_type == ERNIE4_5_VL:
            num_tokens = frames.shape[0] // (self.spatial_conv_size**2 * self.temporal_conv_size)
        else:
            num_tokens = frames.shape[0] // self.image_processor.merge_size**2
        if token_len is not None and token_len != num_tokens:
            raise ValueError("video tokens num not match the size")

        t, h, w = meta["thw"]
        outputs["images"].append(frames)
        outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(np.array([[t, h, w]]))

        fill_id = self.image_patch_id
        if self.model_type in (QWEN_VL, QWEN3_VL):
            fill_id = self.image_token_id

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([fill_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)
        outputs["image_type_ids"].extend([1] * t)

        if self.model_type == ERNIE4_5_VL:
            pos_ids = self._compute_3d_positions_ernie(t, h, w, outputs["cur_position"])
            outputs["position_ids"].extend(pos_ids)
            outputs["cur_position"] = np.max(pos_ids) + 1
        else:
            fps = meta["fps"]
            second_per_grid_t = self.temporal_conv_size / fps
            pos_ids = self._compute_vision_positions_qwen(outputs["cur_position"], t, h, w, second_per_grid_t)
            outputs["position_ids"].append(pos_ids)
            outputs["cur_position"] = pos_ids.max() + 1
            outputs["fps"].append(fps)

    def _compute_vision_positions_qwen(
        self, start_pos: int, t: int, h: int, w: int, second_per_grid_t: float
    ) -> np.ndarray:
        """Generate 3D position IDs — qwen family. Returns np.ndarray(3, N)."""
        h //= self.spatial_conv_size
        w //= self.spatial_conv_size

        tn = np.arange(t).reshape(-1, 1)
        tn = np.broadcast_to(tn, (t, h * w))
        tn = tn * int(second_per_grid_t) * self.tokens_per_second
        t_index = tn.flatten()

        hn = np.arange(h).reshape(1, -1, 1)
        h_index = np.broadcast_to(hn, (t, h, w)).flatten()

        wn = np.arange(w).reshape(1, 1, -1)
        w_index = np.broadcast_to(wn, (t, h, w)).flatten()

        position = np.stack([t_index, h_index, w_index]) + start_pos
        return position

    def _compute_3d_positions_ernie(self, t: int, h: int, w: int, start_idx: int) -> List[List[int]]:
        """Generate 3D positions — ernie. Returns List[List[int]] with temporal downsampling."""
        t_eff = t // self.temporal_conv_size if t != 1 else 1
        gh, gw = h // self.spatial_conv_size, w // self.spatial_conv_size
        time_idx = np.repeat(np.arange(t_eff), gh * gw)
        h_idx = np.tile(np.repeat(np.arange(gh), gw), t_eff)
        w_idx = np.tile(np.arange(gw), t_eff * gh)

        coords = list(zip(time_idx, h_idx, w_idx))
        return [[start_idx + ti, start_idx + hi, start_idx + wi] for ti, hi, wi in coords]

    def _load_and_process_video(self, url, item: Dict):
        """Load and preprocess video. Returns model-type-appropriate result."""
        if self.model_type == ERNIE4_5_VL:
            return self._load_video_ernie(url, item)
        else:
            return self._load_video_qwen(url, item)

    def _load_video_qwen(self, url, item: Dict) -> Tuple[np.ndarray, Dict]:
        """Load video — qwen family. Returns (np.ndarray frames, meta dict)."""
        reader, meta, _ = read_video_decord(url, save_to_disk=False)

        fps = item.get("fps", self.fps)
        num_frames = item.get("target_frames", self.target_frames)

        frame_indices = list(range(meta["num_of_frame"]))
        if fps > 0 or num_frames > 0:
            min_frames = item.get("min_frames", self.min_frames)
            max_frames = item.get("max_frames", self.max_frames)

            if self.model_type == PADDLEOCR_VL:
                frame_indices = sample_frames_paddleocr(
                    frame_factor=self.frame_factor,
                    min_frames=min_frames,
                    max_frames=max_frames,
                    metadata=meta,
                    fps=fps,
                    num_frames=num_frames,
                )
            else:
                frame_indices = sample_frames_qwen(
                    frame_factor=self.frame_factor,
                    min_frames=min_frames,
                    max_frames=max_frames,
                    metadata=meta,
                    fps=-1 if num_frames > 0 else fps,
                    num_frames=num_frames,
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

    def _load_video_ernie(self, url, item: Dict) -> List[Image.Image]:
        """Load video — ernie. Returns List[PIL.Image]."""
        reader, meta, path = read_video_decord(url, save_to_disk=False)

        video_frame_args = {
            "fps": item.get("fps", self.fps),
            "min_frames": item.get("min_frames", self.min_frames),
            "max_frames": item.get("max_frames", self.max_frames),
            "target_frames": item.get("target_frames", self.target_frames),
            "frames_sample": item.get("frames_sample", self.frames_sample),
        }

        video_frame_args = self._set_video_frame_args(video_frame_args, meta)

        frames_data, _, timestamps = read_frames_decord_ernie(
            path,
            reader,
            meta,
            target_frames=video_frame_args["target_frames"],
            target_fps=video_frame_args["fps"],
            frames_sample=video_frame_args["frames_sample"],
            save_to_disk=False,
        )

        frames: List[Image.Image] = []
        for img_array, ts in zip(frames_data, timestamps):
            frames.append(render_frame_timestamp(img_array, ts))
        if len(frames) % 2 != 0:
            frames.append(copy.deepcopy(frames[-1]))
        return frames

    def _set_video_frame_args(self, video_frame_args, video_meta):
        """Validate and adjust ernie video frame arguments."""
        if video_frame_args["target_frames"] > 0:
            if video_frame_args["fps"] >= 0:
                raise ValueError("fps must be negative if target_frames is given")
            if (
                video_frame_args["min_frames"] > 0
                and video_frame_args["target_frames"] < video_frame_args["min_frames"]
            ):
                raise ValueError("target_frames must be larger than min_frames")
            if (
                video_frame_args["max_frames"] > 0
                and video_frame_args["target_frames"] > video_frame_args["max_frames"]
            ):
                raise ValueError("target_frames must be smaller than max_frames")
        else:
            if video_frame_args["fps"] < 0:
                raise ValueError("Must provide either positive target_fps or positive target_frames.")
            frames_to_extract = int(video_meta["duration"] * video_frame_args["fps"])
            if (
                video_frame_args["min_frames"] > 0
                and video_frame_args["max_frames"] > 0
                and video_frame_args["min_frames"] > video_frame_args["max_frames"]
            ):
                raise ValueError("min_frames must be smaller than max_frames")
            if video_frame_args["min_frames"] > 0 and frames_to_extract < video_frame_args["min_frames"]:
                video_frame_args["target_frames"] = video_frame_args["min_frames"]
                video_frame_args["fps"] = -1
            if video_frame_args["max_frames"] > 0 and frames_to_extract > video_frame_args["max_frames"]:
                video_frame_args["target_frames"] = video_frame_args["max_frames"]
                video_frame_args["fps"] = -1
        return video_frame_args
