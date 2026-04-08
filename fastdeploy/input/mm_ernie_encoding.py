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

"""Ernie-4.5-VL encoding class.

Encapsulates the model-specific encoding logic extracted from
``ernie4_5_vl_processor/process.py``.  The ``ErnieEncoding`` instance
is held by the unified ``MultiModalProcessor`` via composition:

    self.enc = ErnieEncoding(self)      # processor reference
    self.enc.add_image(img, outputs, uuid)
"""

import copy
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
from PIL import Image

from fastdeploy.engine.request import ImagePosition
from fastdeploy.input.utils import IDS_TYPE_FLAG, MAX_IMAGE_DIMENSION
from fastdeploy.multimodal.hasher import MultimodalHasher


class ErnieEncoding:
    """Ernie-4.5-VL–specific token encoding logic.

    Holds a reference to the parent ``MultiModalProcessor`` as ``self.p``
    to access shared state (tokenizer, image_processor, config, …).
    """

    IMG_START = "<|IMAGE_START|>"
    IMG_END = "<|IMAGE_END|>"
    VID_START = "<|VIDEO_START|>"
    VID_END = "<|VIDEO_END|>"

    def __init__(self, processor):
        self.p = processor

        tok = self.p.tokenizer
        self.image_start_id = tok.convert_tokens_to_ids(self.IMG_START)
        self.image_end_id = tok.convert_tokens_to_ids(self.IMG_END)
        self.video_start_id = tok.convert_tokens_to_ids(self.VID_START)
        self.video_end_id = tok.convert_tokens_to_ids(self.VID_END)

        self.token_type_mapping = self._build_token_type_mapping()

    def _build_token_type_mapping(self) -> dict:
        mapping = defaultdict(lambda: IDS_TYPE_FLAG["text"])
        for token in (self.IMG_START, self.IMG_END, self.VID_START, self.VID_END):
            mapping[token] = IDS_TYPE_FLAG["image"]
        mapping[self.p.image_patch_id] = IDS_TYPE_FLAG["image"]
        return mapping

    def init_extra(self, processor_kwargs: dict):
        """Perform ernie-specific init after the main processor __init__."""
        p = self.p
        p.role_prefixes = {
            "system": "",
            "user": "User: ",
            "bot": "Assistant: ",
            "assistant": "Assistant: ",
            "tool": "Tool: ",
        }

    @staticmethod
    def mm_num_tokens(grid_thw: "list | list[list[int]] | np.ndarray | paddle.Tensor") -> "int | list[int]":
        """Calculate the number of tokens for multimodal input.

        For ernie: t==1 → t*h*w//4, t>1 → t*h*w//4//2
        """
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

    def add_image(self, img, outputs: Dict, uuid: Optional[str], token_len=None) -> None:
        from paddleformers.transformers.image_utils import ChannelDimension

        p = self.p
        patches_h, patches_w = p.image_processor.get_smarted_resize(
            img.height,
            img.width,
            min_pixels=p.image_min_pixels,
            max_pixels=p.image_max_pixels,
        )[1]
        num_tokens = (patches_h * patches_w) // (p.spatial_conv_size**2)
        if token_len and token_len != num_tokens:
            raise ValueError("image tokens num not match the size")

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([p.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)
        outputs["num_input_image_tokens"] += num_tokens

        pos_ids = self.compute_vision_positions(1, patches_h, patches_w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

        ret = p.image_processor.preprocess(
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

    def add_processed_image(
        self, img_cache: Tuple[np.ndarray, dict], outputs: Dict, uuid: str, token_len=None
    ) -> None:
        p = self.p
        img, meta = img_cache
        num_tokens = img.shape[0] // (p.spatial_conv_size**2)
        if token_len and num_tokens != token_len:
            raise ValueError("image tokens num not match the size")

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([p.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)

        _, h, w = meta["thw"]
        pos_ids = self.compute_vision_positions(1, h, w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

        outputs["images"].append(img)
        outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(np.array([[1, h, w]]))
        outputs["image_type_ids"].append(0)

    def add_video(self, frames, outputs: Dict, uuid: Optional[str], token_len=None) -> None:
        from paddleformers.transformers.image_utils import ChannelDimension

        p = self.p
        patches_h, patches_w = p.image_processor.get_smarted_resize(
            frames[0].height,
            frames[0].width,
            min_pixels=p.video_min_pixels,
            max_pixels=p.video_max_pixels,
        )[1]
        num_frames = len(frames)
        num_tokens = (num_frames * patches_h * patches_w) // (p.spatial_conv_size**2 * p.temporal_conv_size)
        if token_len and num_tokens != token_len:
            raise ValueError("video tokens num not match the size")

        pixel_stack = np.stack([np.array(f.convert("RGB")) for f in frames], axis=0)
        ret = p.image_processor.preprocess(
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
        outputs["input_ids"].extend([p.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)
        outputs["num_input_video_tokens"] += num_tokens

        pos_ids = self.compute_vision_positions(num_frames, patches_h, patches_w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

    def add_processed_video(
        self, frames_cache: Tuple[np.ndarray, dict], outputs: Dict, uuid: str, token_len=None
    ) -> None:
        p = self.p
        frames, meta = frames_cache
        num_tokens = frames.shape[0] // (p.spatial_conv_size**2 * p.temporal_conv_size)
        if token_len and num_tokens != token_len:
            raise ValueError("video tokens num not match the size")

        t, h, w = meta["thw"]
        outputs["images"].append(frames)
        outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(np.array([[t, h, w]]))

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([p.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)
        outputs["image_type_ids"].extend([1] * t)

        pos_ids = self.compute_vision_positions(t, h, w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

    def load_video(self, url: str, item: Dict) -> List[Image.Image]:
        """Load and preprocess video, returning a list of PIL Images with timestamps."""
        from fastdeploy.input.video_utils import (
            read_frames_decord,
            read_video_decord,
            render_frame_timestamp,
        )

        p = self.p
        reader, meta, path = read_video_decord(url, save_to_disk=False)

        video_frame_args = {
            "fps": item.get("fps", p.fps),
            "min_frames": item.get("min_frames", p.min_frames),
            "max_frames": item.get("max_frames", p.max_frames),
            "target_frames": item.get("target_frames", p.target_frames),
            "frames_sample": item.get("frames_sample", p.frames_sample),
        }

        video_frame_args = self.set_video_frame_args(video_frame_args, meta)

        frames_data, _, timestamps = read_frames_decord(
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

    def set_video_frame_args(self, video_frame_args: dict, video_meta: dict) -> dict:
        """Resolve final frame-sampling arguments based on priority.

        Priority: target_frames > (min_frames, max_frames) > fps
        """
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

    def add_text_positions(self, outputs: Dict, num_tokens: int) -> None:
        """Append position IDs for text tokens (ernie: [pos]*3 per token)."""
        start = outputs["cur_position"]
        for i in range(num_tokens):
            outputs["position_ids"].append([start + i] * 3)
        outputs["cur_position"] += num_tokens

    def append_completion_tokens(self, multimodal_inputs: Dict, completion_token_ids: list) -> None:
        """Append completion tokens to existing outputs (ernie variant)."""
        num_tokens = len(completion_token_ids)
        multimodal_inputs["input_ids"].extend(completion_token_ids)
        multimodal_inputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * num_tokens)

        start = multimodal_inputs["cur_position"]
        for i in range(num_tokens):
            multimodal_inputs["position_ids"].append([start + i] * 3)
        multimodal_inputs["cur_position"] += num_tokens

    def compute_vision_positions(self, t: int, h: int, w: int, start_idx: int) -> List[List[int]]:
        """Compute 3D position IDs for visual tokens (ernie variant).

        Returns List[List[int]] — each inner list is [t_pos, h_pos, w_pos].
        """
        p = self.p
        t_eff = t // p.temporal_conv_size if t != 1 else 1
        gh, gw = h // p.spatial_conv_size, w // p.spatial_conv_size
        time_idx = np.repeat(np.arange(t_eff), gh * gw)
        h_idx = np.tile(np.repeat(np.arange(gh), gw), t_eff)
        w_idx = np.tile(np.arange(gw), t_eff * gh)

        coords = list(zip(time_idx, h_idx, w_idx))
        return [[start_idx + int(ti), start_idx + int(hi), start_idx + int(wi)] for ti, hi, wi in coords]

    def prompt_token_ids2outputs(
        self, request: Dict[str, Any], tgts: List[str] = None
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray], None]]:
        """Build outputs from pre-tokenized prompt_token_ids (ernie variant).

        Scans for IMAGE_START/IMAGE_END and VIDEO_START/VIDEO_END boundaries.
        """

        p = self.p
        outputs = p._make_outputs()
        prompt_token_ids = request.get("prompt_token_ids", [])
        prompt_token_ids_len = len(prompt_token_ids)

        if not request.get("messages"):
            outputs["input_ids"].extend(prompt_token_ids)
            outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * prompt_token_ids_len)
            for i in range(prompt_token_ids_len):
                outputs["position_ids"].append([i] * 3)
            outputs["cur_position"] += prompt_token_ids_len
            return outputs

        images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items = p._extract_mm_items(request)
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
                    self.add_image(image, outputs, uuid, token_len)
                else:
                    self.add_processed_image(image, outputs, uuid, token_len)
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
                        frames = self.load_video(video["video"], video)
                    else:
                        frames = self.load_video(video, {})
                    self.add_video(frames, outputs, uuid, token_len)
                else:
                    self.add_processed_video(video, outputs, uuid, token_len)
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

        if p.enable_processor_cache:
            missing_idx_set = set(missing_idx)
            hashes_to_cache, items_to_cache = [], []
            for idx in range(len(mm_items)):
                if idx in missing_idx_set:
                    continue
                meta = {}
                t, h, w = outputs["grid_thw"][idx][0]
                meta["thw"] = (t, h, w)
                hashes_to_cache.append(outputs["mm_hashes"][idx])
                items_to_cache.append((outputs["images"][idx], meta))
            if hashes_to_cache:
                p.update_processor_cache(dealer, hashes_to_cache, items_to_cache)

        return outputs

    def pack_position_ids(self, outputs: Dict) -> None:
        """Pack position_ids as np.array (ernie format: list of [t,h,w])."""
        outputs["position_ids"] = np.array(outputs["position_ids"], dtype=np.int64)
        outputs["image_patch_id"] = self.p.image_patch_id

    def get_mm_max_tokens_per_item(self, seq_len: int) -> Mapping[str, int]:
        """Return per-modality max token counts."""
        max_image_tokens = self._get_max_image_tokens(seq_len)
        max_video_tokens = self._get_max_video_tokens(seq_len)
        return {"image": max_image_tokens, "video": max_video_tokens}

    def _get_max_image_tokens(self, seq_len: int) -> int:
        p = self.p
        target_height, target_width = self._get_image_size_with_most_features()
        patches_h, patches_w = p.image_processor.get_smarted_resize(
            height=target_height,
            width=target_width,
            min_pixels=p.image_min_pixels,
            max_pixels=p.image_max_pixels,
        )[1]
        num_image_tokens = (patches_h * patches_w) // (p.spatial_conv_size**2)
        return min(num_image_tokens, seq_len)

    def _get_max_video_tokens(self, seq_len: int) -> int:
        p = self.p
        target_height, target_width = self._get_image_size_with_most_features()
        patches_h, patches_w = p.image_processor.get_smarted_resize(
            height=target_height,
            width=target_width,
            min_pixels=p.video_min_pixels,
            max_pixels=p.video_max_pixels,
        )[1]
        num_video_tokens = (patches_h * patches_w) // (p.spatial_conv_size**2 * p.temporal_conv_size)
        return min(num_video_tokens, seq_len)

    def _get_image_size_with_most_features(self):
        p = self.p
        resized_height, resized_width = p.image_processor.get_smarted_resize(
            height=MAX_IMAGE_DIMENSION,
            width=MAX_IMAGE_DIMENSION,
            min_pixels=p.image_min_pixels,
            max_pixels=p.image_max_pixels,
        )[0]
        return (resized_height, resized_width)
