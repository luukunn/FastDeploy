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

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.input.multimodal.mm_processor import MMProcessor
from fastdeploy.input.multimodal.qwen_vl import QwenVLProcessor
from fastdeploy.input.multimodal.ernie_vl import ErnieVLProcessor
from fastdeploy.input.multimodal.paddleocr_vl import PaddleOCRVLProcessor
from fastdeploy.input.utils import IDS_TYPE_FLAG


# ===========================================================================
# Helpers
# ===========================================================================

def _make_mock_tokenizer(token_map=None):
    """Create a mock tokenizer for VL processors."""
    if token_map is None:
        token_map = {
            "<|image_pad|>": 100,
            "<|video_pad|>": 101,
            "<|IMAGE_PLACEHOLDER|>": 102,
            "<|IMAGE_START|>": 200,
            "<|IMAGE_END|>": 201,
            "<|VIDEO_START|>": 202,
            "<|VIDEO_END|>": 203,
        }
    tok = MagicMock()
    tok.convert_tokens_to_ids.side_effect = lambda s: token_map.get(s, 999)
    tok.tokenize.side_effect = lambda s: list(s)
    return tok


def _make_mock_image_processor(merge_size=2, temporal_patch_size=2):
    """Create a mock image processor."""
    ip = MagicMock()
    ip.merge_size = merge_size
    ip.temporal_patch_size = temporal_patch_size
    return ip


def _make_mock_config(tokens_per_second=2):
    """Create a mock model config."""
    config = MagicMock()
    config.vision_config = MagicMock()
    config.vision_config.tokens_per_second = tokens_per_second
    return config


def _make_qwen_processor(**overrides):
    """Create a QwenVLProcessor with mocked dependencies."""
    tok = _make_mock_tokenizer()
    ip = _make_mock_image_processor()
    config = _make_mock_config()
    proc = QwenVLProcessor(tokenizer=tok, image_processor=ip, config=config, processor_kwargs={})
    for k, v in overrides.items():
        setattr(proc, k, v)
    return proc


def _make_ernie_processor(**overrides):
    """Create an ErnieVLProcessor with mocked dependencies."""
    tok = _make_mock_tokenizer()
    ip = _make_mock_image_processor()
    ip.get_smarted_resize = MagicMock(return_value=((56, 56), (4, 4)))
    config = _make_mock_config()
    proc = ErnieVLProcessor(tokenizer=tok, image_processor=ip, config=config, processor_kwargs={})
    for k, v in overrides.items():
        setattr(proc, k, v)
    return proc


def _make_paddleocr_processor(**overrides):
    """Create a PaddleOCRVLProcessor with mocked dependencies."""
    tok = _make_mock_tokenizer()
    ip = _make_mock_image_processor()
    config = _make_mock_config()
    proc = PaddleOCRVLProcessor(tokenizer=tok, image_processor=ip, config=config, processor_kwargs={})
    for k, v in overrides.items():
        setattr(proc, k, v)
    return proc


# ===========================================================================
# Tests: QwenVLProcessor
# ===========================================================================

class TestQwenVLProcessorInit(unittest.TestCase):
    def test_init_basic(self):
        proc = _make_qwen_processor()
        self.assertEqual(proc.image_token_id, 100)
        self.assertEqual(proc.video_token_id, 101)
        self.assertEqual(proc.spatial_conv_size, 2)
        self.assertEqual(proc.temporal_conv_size, 2)
        self.assertEqual(proc.tokens_per_second, 2)
        self.assertTrue(proc._supports_prompt_token_ids)


class TestQwenVLMakeOutputs(unittest.TestCase):
    def test_has_fps(self):
        proc = _make_qwen_processor()
        outputs = proc._make_outputs()
        self.assertIn("fps", outputs)
        self.assertEqual(outputs["fps"], [])


class TestQwenVLComputeTextPositions(unittest.TestCase):
    def test_basic(self):
        proc = _make_qwen_processor()
        pos = proc._compute_text_positions(0, 3)
        self.assertEqual(pos.shape, (3, 3))
        np.testing.assert_array_equal(pos[0], [0, 1, 2])
        np.testing.assert_array_equal(pos[1], [0, 1, 2])
        np.testing.assert_array_equal(pos[2], [0, 1, 2])

    def test_with_offset(self):
        proc = _make_qwen_processor()
        pos = proc._compute_text_positions(5, 2)
        np.testing.assert_array_equal(pos[0], [5, 6])

    def test_zero_tokens(self):
        proc = _make_qwen_processor()
        pos = proc._compute_text_positions(0, 0)
        self.assertEqual(pos.shape, (3, 0))


class TestQwenVLComputeVisionPositions(unittest.TestCase):
    def test_single_image(self):
        proc = _make_qwen_processor()
        # t=1, h=4, w=4, merge_size=2 => gh=2, gw=2 => 4 tokens
        pos = proc._compute_vision_positions(0, 1, 4, 4, 0)
        self.assertEqual(pos.shape, (3, 4))

    def test_video_multi_frame(self):
        proc = _make_qwen_processor()
        # t=2, h=4, w=4, merge_size=2 => gh=2, gw=2 => 2*4=8 tokens
        pos = proc._compute_vision_positions(0, 2, 4, 4, 1.0)
        self.assertEqual(pos.shape, (3, 8))


class TestQwenVLAddTextPositions(unittest.TestCase):
    def test_appends_position_ids(self):
        proc = _make_qwen_processor()
        outputs = proc._make_outputs()
        proc.add_text_positions(outputs, 3)
        self.assertEqual(len(outputs["position_ids"]), 1)
        self.assertEqual(outputs["position_ids"][0].shape, (3, 3))
        self.assertEqual(outputs["cur_position"], 3)


class TestQwenVLAppendCompletionTokens(unittest.TestCase):
    def test_basic(self):
        proc = _make_qwen_processor()
        outputs = proc._make_outputs()
        proc.append_completion_tokens(outputs, [10, 11, 12])
        self.assertEqual(outputs["input_ids"], [10, 11, 12])
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["text"]] * 3)
        self.assertEqual(outputs["cur_position"], 3)
        self.assertEqual(len(outputs["position_ids"]), 1)


class TestQwenVLAddProcessedImage(unittest.TestCase):
    def test_basic(self):
        proc = _make_qwen_processor()
        # merge_size=2, so 16 pixels => 16/4 = 4 tokens
        img = np.zeros((16, 3, 14, 14))  # shape[0]=16, num_tokens = 16//4 = 4
        meta = {"thw": (1, 4, 4)}
        outputs = proc._make_outputs()
        proc.add_processed_image((img, meta), outputs, uuid="img_uuid")
        self.assertEqual(len(outputs["input_ids"]), 4)
        self.assertTrue(all(t == 100 for t in outputs["input_ids"]))
        self.assertEqual(outputs["mm_hashes"], ["img_uuid"])

    def test_token_len_mismatch_raises(self):
        proc = _make_qwen_processor()
        img = np.zeros((16, 3, 14, 14))
        meta = {"thw": (1, 4, 4)}
        outputs = proc._make_outputs()
        with self.assertRaises(ValueError):
            proc.add_processed_image((img, meta), outputs, uuid="u", token_len=999)


class TestQwenVLAddProcessedVideo(unittest.TestCase):
    def test_basic(self):
        proc = _make_qwen_processor()
        # merge_size=2, shape[0]=8 => 8/4 = 2 tokens
        frames = np.zeros((8, 3, 14, 14))
        meta = {"thw": (2, 2, 2), "fps": 2.0}
        outputs = proc._make_outputs()
        proc.add_processed_video((frames, meta), outputs, uuid="vid_uuid")
        self.assertEqual(len(outputs["input_ids"]), 2)
        self.assertEqual(outputs["mm_hashes"], ["vid_uuid"])
        self.assertEqual(outputs["fps"], [2.0])

    def test_token_len_mismatch_raises(self):
        proc = _make_qwen_processor()
        frames = np.zeros((8, 3, 14, 14))
        meta = {"thw": (2, 2, 2), "fps": 2.0}
        outputs = proc._make_outputs()
        with self.assertRaises(ValueError):
            proc.add_processed_video((frames, meta), outputs, uuid="u", token_len=999)


class TestQwenVLMmNumTokens(unittest.TestCase):
    def test_single(self):
        result = QwenVLProcessor.mm_num_tokens([2, 4, 4])
        self.assertEqual(result, 2 * 4 * 4 // 4)

    def test_list(self):
        result = QwenVLProcessor.mm_num_tokens([[1, 4, 4], [2, 4, 4]])
        self.assertEqual(result, [4, 8])

    def test_empty(self):
        result = QwenVLProcessor.mm_num_tokens([])
        self.assertEqual(result, 0)


class TestQwenVLPackPositionIds(unittest.TestCase):
    def test_concatenates_and_transposes(self):
        proc = _make_qwen_processor()
        outputs = proc._make_outputs()
        # Add some position IDs (3xN arrays)
        outputs["position_ids"] = [
            np.array([[0, 1], [0, 1], [0, 1]]),
            np.array([[2, 3], [2, 3], [2, 3]]),
        ]
        proc.pack_position_ids(outputs)
        # After concat (3x4) then transpose (4x3)
        self.assertEqual(outputs["position_ids"].shape, (4, 3))


class TestQwenVLPromptTokenIds2Outputs(unittest.TestCase):
    def test_text_only(self):
        proc = _make_qwen_processor()
        outputs = proc.prompt_token_ids2outputs([1, 2, 3])
        self.assertEqual(outputs["input_ids"], [1, 2, 3])
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["text"]] * 3)

    def test_mm_count_mismatch_raises(self):
        proc = _make_qwen_processor()
        # prompt has image tokens but no mm_items
        prompt = [100, 100, 100]  # 3 image tokens
        mm_items = [
            {"type": "image", "data": (np.zeros((4, 3, 14, 14)), {"thw": (1, 2, 2)}), "uuid": "u1"},
            {"type": "image", "data": (np.zeros((4, 3, 14, 14)), {"thw": (1, 2, 2)}), "uuid": "u2"},
        ]
        # 3 image tokens in one block won't match 2 items (one item would need 1 token block)
        with self.assertRaises(ValueError):
            proc.prompt_token_ids2outputs(prompt, mm_items)

    def test_with_processed_image(self):
        proc = _make_qwen_processor()
        # 1 image token (merge_size=2, 4 pixels => 1 token)
        img = np.zeros((4, 3, 14, 14))
        mm_items = [{"type": "image", "data": (img, {"thw": (1, 2, 2)}), "uuid": "u1"}]
        prompt = [1, 2, 100, 3]  # text, text, image_token, text
        outputs = proc.prompt_token_ids2outputs(prompt, mm_items)
        # Should have text + image + text tokens
        self.assertEqual(len(outputs["input_ids"]), 4)


# ===========================================================================
# Tests: ErnieVLProcessor
# ===========================================================================

class TestErnieVLProcessorInit(unittest.TestCase):
    def test_init_basic(self):
        proc = _make_ernie_processor()
        self.assertEqual(proc.image_token_id, 102)
        self.assertEqual(proc.video_token_id, 102)
        self.assertTrue(proc._supports_prompt_token_ids)
        self.assertEqual(proc.spatial_conv_size, 2)
        self.assertEqual(proc.temporal_conv_size, 2)

    def test_init_extra_defaults(self):
        proc = _make_ernie_processor()
        self.assertEqual(proc.image_min_pixels, 4 * 28 * 28)
        self.assertEqual(proc.image_max_pixels, 6177 * 28 * 28)

    def test_init_extra_custom(self):
        tok = _make_mock_tokenizer()
        ip = _make_mock_image_processor()
        ip.get_smarted_resize = MagicMock(return_value=((56, 56), (4, 4)))
        config = _make_mock_config()
        proc = ErnieVLProcessor(
            tokenizer=tok, image_processor=ip, config=config,
            processor_kwargs={"image_min_pixels": 100, "video_max_frames": 32},
        )
        self.assertEqual(proc.image_min_pixels, 100)
        self.assertEqual(proc.max_frames, 32)


class TestErnieVLBuildTokenTypeMapping(unittest.TestCase):
    def test_boundary_tokens_are_image_type(self):
        proc = _make_ernie_processor()
        mapping = proc.token_type_mapping
        self.assertEqual(mapping["<|IMAGE_START|>"], IDS_TYPE_FLAG["image"])
        self.assertEqual(mapping["<|IMAGE_END|>"], IDS_TYPE_FLAG["image"])
        self.assertEqual(mapping["<|VIDEO_START|>"], IDS_TYPE_FLAG["image"])
        self.assertEqual(mapping["<|VIDEO_END|>"], IDS_TYPE_FLAG["image"])


class TestErnieVLCompute3dPositions(unittest.TestCase):
    def test_single_image(self):
        proc = _make_ernie_processor()
        # t=1, h=4, w=4, spatial=2, temporal=2
        # t_eff=1 (since t==1), gh=2, gw=2 => 4 positions
        pos = proc._compute_3d_positions(1, 4, 4, 0)
        self.assertEqual(len(pos), 4)
        # Each position is [t_idx + start, h_idx + start, w_idx + start]
        self.assertEqual(pos[0], [0, 0, 0])

    def test_video(self):
        proc = _make_ernie_processor()
        # t=4, h=4, w=4, spatial=2, temporal=2
        # t_eff=4//2=2, gh=2, gw=2 => 2*4 = 8 positions
        pos = proc._compute_3d_positions(4, 4, 4, 0)
        self.assertEqual(len(pos), 8)


class TestErnieVLAddTextPositions(unittest.TestCase):
    def test_appends_list_of_lists(self):
        proc = _make_ernie_processor()
        outputs = proc._make_outputs()
        proc.add_text_positions(outputs, 3)
        self.assertEqual(len(outputs["position_ids"]), 3)
        self.assertEqual(outputs["position_ids"][0], [0, 0, 0])
        self.assertEqual(outputs["position_ids"][1], [1, 1, 1])
        self.assertEqual(outputs["position_ids"][2], [2, 2, 2])
        self.assertEqual(outputs["cur_position"], 3)


class TestErnieVLAppendCompletionTokens(unittest.TestCase):
    def test_basic(self):
        proc = _make_ernie_processor()
        outputs = proc._make_outputs()
        proc.append_completion_tokens(outputs, [10, 11])
        self.assertEqual(outputs["input_ids"], [10, 11])
        self.assertEqual(len(outputs["position_ids"]), 2)
        self.assertEqual(outputs["cur_position"], 2)


class TestErnieVLWriteBack(unittest.TestCase):
    def test_preserves_existing_prompt_token_ids(self):
        proc = _make_ernie_processor()
        request = {"prompt_token_ids": [1, 2, 3]}
        outputs = {"input_ids": np.array([10, 20, 30])}
        proc._write_back(request, outputs)
        self.assertEqual(request["prompt_token_ids"], [1, 2, 3])
        self.assertEqual(request["multimodal_inputs"], outputs)

    def test_sets_prompt_token_ids_when_absent(self):
        proc = _make_ernie_processor()
        request = {}
        outputs = {"input_ids": np.array([10, 20, 30])}
        proc._write_back(request, outputs)
        self.assertEqual(request["prompt_token_ids"], [10, 20, 30])


class TestErnieVLAddProcessedImage(unittest.TestCase):
    def test_basic(self):
        proc = _make_ernie_processor()
        # spatial_conv_size=2 => num_tokens = shape[0] // 4
        img = np.zeros((8, 3, 14, 14))  # => 2 tokens
        meta = {"thw": (1, 2, 4)}  # h=2, w=4
        outputs = proc._make_outputs()
        proc.add_processed_image((img, meta), outputs, uuid="u1")
        self.assertEqual(len(outputs["input_ids"]), 2)
        self.assertEqual(outputs["mm_hashes"], ["u1"])

    def test_token_len_mismatch_raises(self):
        proc = _make_ernie_processor()
        img = np.zeros((8, 3, 14, 14))
        meta = {"thw": (1, 2, 4)}
        outputs = proc._make_outputs()
        with self.assertRaises(ValueError):
            proc.add_processed_image((img, meta), outputs, uuid="u", token_len=999)


class TestErnieVLAddProcessedVideo(unittest.TestCase):
    def test_basic(self):
        proc = _make_ernie_processor()
        # spatial=2, temporal=2 => num_tokens = shape[0] // (4*2) = 16/8 = 2
        frames = np.zeros((16, 3, 14, 14))
        meta = {"thw": (4, 2, 4)}
        outputs = proc._make_outputs()
        proc.add_processed_video((frames, meta), outputs, uuid="v1")
        self.assertEqual(len(outputs["input_ids"]), 2)
        self.assertEqual(outputs["mm_hashes"], ["v1"])


class TestErnieVLMmNumTokens(unittest.TestCase):
    def test_image(self):
        result = ErnieVLProcessor.mm_num_tokens([1, 4, 4])
        self.assertEqual(result, 1 * 4 * 4 // 4)

    def test_video(self):
        result = ErnieVLProcessor.mm_num_tokens([2, 4, 4])
        # t>1: t*h*w // 4 // 2
        self.assertEqual(result, 2 * 4 * 4 // 4 // 2)

    def test_list(self):
        result = ErnieVLProcessor.mm_num_tokens([[1, 4, 4], [2, 4, 4]])
        self.assertEqual(result, [4, 4])

    def test_empty(self):
        result = ErnieVLProcessor.mm_num_tokens([])
        self.assertEqual(result, 0)


class TestErnieVLPackPositionIds(unittest.TestCase):
    def test_basic(self):
        proc = _make_ernie_processor()
        outputs = proc._make_outputs()
        outputs["position_ids"] = [[0, 0, 0], [1, 1, 1]]
        proc.pack_position_ids(outputs)
        self.assertEqual(outputs["position_ids"].shape, (2, 3))
        np.testing.assert_array_equal(outputs["position_ids"][0], [0, 0, 0])


class TestErnieVLPromptTokenIds2Outputs(unittest.TestCase):
    def test_text_only(self):
        proc = _make_ernie_processor()
        outputs = proc.prompt_token_ids2outputs([1, 2, 3])
        self.assertEqual(outputs["input_ids"], [1, 2, 3])
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["text"]] * 3)
        self.assertEqual(len(outputs["position_ids"]), 3)

    def test_with_processed_image(self):
        proc = _make_ernie_processor()
        img = np.zeros((4, 3, 14, 14))  # => 1 token (spatial=2 => 4//4=1)
        mm_items = [{"type": "image", "data": (img, {"thw": (1, 2, 2)}), "uuid": "u1"}]
        # IMAGE_START=200, image_token=102, IMAGE_END=201
        prompt = [1, 200, 102, 201, 2]
        outputs = proc.prompt_token_ids2outputs(prompt, mm_items)
        # Should contain: [1, 200, 102, 201, 2]
        self.assertEqual(len(outputs["input_ids"]), 5)


class TestErnieVLSetVideoFrameArgs(unittest.TestCase):
    def test_target_frames_positive(self):
        proc = _make_ernie_processor()
        args = {"fps": -1, "min_frames": 4, "max_frames": 100, "target_frames": 16, "frames_sample": "leading"}
        meta = {"duration": 10.0, "num_of_frame": 100}
        result = proc.set_video_frame_args(args, meta)
        self.assertEqual(result["target_frames"], 16)

    def test_target_frames_with_positive_fps_raises(self):
        proc = _make_ernie_processor()
        args = {"fps": 2.0, "min_frames": 4, "max_frames": 100, "target_frames": 16, "frames_sample": "leading"}
        meta = {"duration": 10.0}
        with self.assertRaises(ValueError):
            proc.set_video_frame_args(args, meta)

    def test_target_frames_below_min_raises(self):
        proc = _make_ernie_processor()
        args = {"fps": -1, "min_frames": 20, "max_frames": 100, "target_frames": 10, "frames_sample": "leading"}
        meta = {"duration": 10.0}
        with self.assertRaises(ValueError):
            proc.set_video_frame_args(args, meta)

    def test_target_frames_above_max_raises(self):
        proc = _make_ernie_processor()
        args = {"fps": -1, "min_frames": 4, "max_frames": 10, "target_frames": 20, "frames_sample": "leading"}
        meta = {"duration": 10.0}
        with self.assertRaises(ValueError):
            proc.set_video_frame_args(args, meta)

    def test_fps_negative_no_target_raises(self):
        proc = _make_ernie_processor()
        args = {"fps": -1, "min_frames": 4, "max_frames": 100, "target_frames": -1, "frames_sample": "leading"}
        meta = {"duration": 10.0}
        with self.assertRaises(ValueError):
            proc.set_video_frame_args(args, meta)

    def test_min_greater_than_max_raises(self):
        proc = _make_ernie_processor()
        args = {"fps": 2.0, "min_frames": 100, "max_frames": 10, "target_frames": -1, "frames_sample": "leading"}
        meta = {"duration": 10.0}
        with self.assertRaises(ValueError):
            proc.set_video_frame_args(args, meta)

    def test_fps_clamp_to_min(self):
        proc = _make_ernie_processor()
        # fps=2, duration=2 => frames_to_extract=4, min_frames=8 => clamp to min
        args = {"fps": 2.0, "min_frames": 8, "max_frames": 100, "target_frames": -1, "frames_sample": "leading"}
        meta = {"duration": 2.0}
        result = proc.set_video_frame_args(args, meta)
        self.assertEqual(result["target_frames"], 8)
        self.assertEqual(result["fps"], -1)

    def test_fps_clamp_to_max(self):
        proc = _make_ernie_processor()
        # fps=10, duration=10 => frames_to_extract=100, max_frames=20 => clamp to max
        args = {"fps": 10.0, "min_frames": 4, "max_frames": 20, "target_frames": -1, "frames_sample": "leading"}
        meta = {"duration": 10.0}
        result = proc.set_video_frame_args(args, meta)
        self.assertEqual(result["target_frames"], 20)
        self.assertEqual(result["fps"], -1)


class TestErnieVLGetMmMaxTokensPerItem(unittest.TestCase):
    def test_returns_dict(self):
        proc = _make_ernie_processor()
        proc.image_processor.get_smarted_resize = MagicMock(return_value=((56, 56), (4, 4)))
        result = proc.get_mm_max_tokens_per_item(seq_len=1000)
        self.assertIn("image", result)
        self.assertIn("video", result)
        self.assertIsInstance(result["image"], int)
        self.assertIsInstance(result["video"], int)

    def test_capped_by_seq_len(self):
        proc = _make_ernie_processor()
        # h=4, w=4 => tokens = 4*4/4 = 4 image tokens
        proc.image_processor.get_smarted_resize = MagicMock(return_value=((56, 56), (4, 4)))
        result = proc.get_mm_max_tokens_per_item(seq_len=2)
        self.assertEqual(result["image"], 2)


# ===========================================================================
# Tests: PaddleOCRVLProcessor
# ===========================================================================

class TestPaddleOCRVLProcessorInit(unittest.TestCase):
    def test_inherits_from_qwen(self):
        self.assertTrue(issubclass(PaddleOCRVLProcessor, QwenVLProcessor))

    def test_placeholders(self):
        proc = _make_paddleocr_processor()
        self.assertEqual(proc.image_placeholder, "<|IMAGE_PLACEHOLDER|>")
        self.assertEqual(proc.video_placeholder, "<|video_pad|>")


class TestPaddleOCRVLMakeOutputs(unittest.TestCase):
    def test_has_vit_fields(self):
        proc = _make_paddleocr_processor()
        outputs = proc._make_outputs()
        self.assertIn("vit_seqlen", outputs)
        self.assertIn("vit_position_ids", outputs)
        # Also inherits fps from QwenVLProcessor
        self.assertIn("fps", outputs)


class TestPaddleOCRVLAddProcessedImage(unittest.TestCase):
    def test_appends_vit_fields(self):
        proc = _make_paddleocr_processor()
        img = np.zeros((16, 3, 14, 14))
        meta = {"thw": (1, 4, 4)}
        outputs = proc._make_outputs()
        proc.add_processed_image((img, meta), outputs, uuid="u1")
        self.assertEqual(len(outputs["vit_seqlen"]), 1)
        self.assertEqual(outputs["vit_seqlen"][0], 4 * 4)  # h*w
        self.assertEqual(len(outputs["vit_position_ids"]), 1)


class TestPaddleOCRVLAddVideo(unittest.TestCase):
    def test_uses_video_token_id(self):
        proc = _make_paddleocr_processor()
        # Mock preprocess
        ret = MagicMock()
        ret.__getitem__ = lambda self, key: {
            "grid_thw": np.array([2, 4, 4]),
            "pixel_values": np.zeros((8, 3, 14, 14)),
        }[key]
        proc.image_processor.preprocess = MagicMock(return_value={
            "grid_thw": np.array([2, 4, 4]),
            "pixel_values": np.zeros((8, 3, 14, 14)),
        })
        outputs = proc._make_outputs()
        proc.add_video(None, outputs, uuid="v1", meta={"fps": 2.0})
        # Video should use video_token_id (101 for paddleocr)
        self.assertTrue(all(t == 101 for t in outputs["input_ids"]))


class TestPaddleOCRVLAddProcessedVideo(unittest.TestCase):
    def test_uses_video_token_id(self):
        proc = _make_paddleocr_processor()
        frames = np.zeros((8, 3, 14, 14))  # => 8/4 = 2 tokens
        meta = {"thw": (2, 2, 2), "fps": 2.0}
        outputs = proc._make_outputs()
        proc.add_processed_video((frames, meta), outputs, uuid="v1")
        # Video should use video_token_id (101)
        self.assertTrue(all(t == 101 for t in outputs["input_ids"]))

    def test_appends_vit_fields(self):
        proc = _make_paddleocr_processor()
        frames = np.zeros((8, 3, 14, 14))
        meta = {"thw": (2, 2, 2), "fps": 2.0}
        outputs = proc._make_outputs()
        proc.add_processed_video((frames, meta), outputs, uuid="v1")
        self.assertEqual(len(outputs["vit_seqlen"]), 1)
        self.assertEqual(outputs["vit_seqlen"][0], 2 * 2)


# ===========================================================================
# Tests: MMProcessor base class
# ===========================================================================

class TestMMProcessorBase(unittest.TestCase):
    def test_is_abstract(self):
        self.assertTrue(issubclass(MMProcessor, type(MMProcessor).__mro__[0]))
        # Cannot instantiate directly
        with self.assertRaises(TypeError):
            MMProcessor(MagicMock(), MagicMock())

    def test_make_outputs_structure(self):
        proc = _make_qwen_processor()
        outputs = MMProcessor._make_outputs(proc)
        expected_keys = {
            "input_ids", "token_type_ids", "position_ids", "images",
            "grid_thw", "image_type_ids", "labels", "cur_position",
            "video_cnt", "num_input_image_tokens", "num_input_video_tokens",
            "mm_positions", "mm_hashes",
        }
        self.assertEqual(set(outputs.keys()), expected_keys)

    def test_write_back_default(self):
        proc = _make_qwen_processor()
        request = {}
        outputs = {"input_ids": np.array([1, 2, 3])}
        MMProcessor._write_back(proc, request, outputs)
        self.assertEqual(request["prompt_token_ids"], [1, 2, 3])
        self.assertEqual(request["multimodal_inputs"], outputs)

    def test_text2ids_text_only(self):
        proc = _make_qwen_processor()
        proc.tokenizer.tokenize.side_effect = lambda s: list(s)
        proc.tokenizer.convert_tokens_to_ids.side_effect = lambda tokens: [ord(t) for t in tokens]
        outputs = proc._text2ids("hello", images=[], videos=[])
        self.assertEqual(len(outputs["input_ids"]), 5)

    def test_add_text_empty_noop(self):
        proc = _make_qwen_processor()
        outputs = proc._make_outputs()
        proc._add_text("", outputs)
        self.assertEqual(outputs["input_ids"], [])

    def test_add_text_string(self):
        proc = _make_qwen_processor()
        proc.tokenizer.tokenize.side_effect = lambda s: list(s)
        proc.tokenizer.convert_tokens_to_ids.side_effect = lambda tokens: [ord(t) for t in tokens]
        outputs = proc._make_outputs()
        proc._add_text("ab", outputs)
        self.assertEqual(len(outputs["input_ids"]), 2)
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["text"], IDS_TYPE_FLAG["text"]])

    def test_pack_outputs_no_images(self):
        proc = _make_qwen_processor()
        outputs = proc._make_outputs()
        outputs["input_ids"] = [1, 2, 3]
        outputs["token_type_ids"] = [0, 0, 0]
        outputs["position_ids"] = [np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])]
        packed = proc._pack_outputs(outputs)
        self.assertIsNone(packed["images"])
        self.assertIsNone(packed["grid_thw"])
        np.testing.assert_array_equal(packed["input_ids"], [1, 2, 3])

    def test_pack_outputs_with_images(self):
        proc = _make_qwen_processor()
        outputs = proc._make_outputs()
        outputs["input_ids"] = [100, 100, 1, 2]
        outputs["token_type_ids"] = [1, 1, 0, 0]
        outputs["images"] = [np.zeros((2, 3, 14, 14))]
        outputs["grid_thw"] = [np.array([[1, 2, 2]])]
        outputs["image_type_ids"] = [0]
        outputs["position_ids"] = [
            np.array([[0, 1], [0, 0], [0, 1]]),
            np.array([[2, 3], [2, 3], [2, 3]]),
        ]
        packed = proc._pack_outputs(outputs)
        self.assertIsNotNone(packed["images"])
        self.assertEqual(packed["images"].shape[0], 2)


# ===========================================================================
# Tests: MMProcessor._parse_limits and _check_mm_limits
# ===========================================================================

class TestMMProcessorParseLimits(unittest.TestCase):
    def test_none_returns_defaults(self):
        proc = _make_qwen_processor()
        result = proc._parse_limits(None)
        self.assertEqual(result, {"image": 1, "video": 1, "audio": 1})

    def test_valid_limits_merged(self):
        proc = _make_qwen_processor()
        result = proc._parse_limits({"image": 5, "video": 3})
        self.assertEqual(result["image"], 5)
        self.assertEqual(result["video"], 3)
        self.assertEqual(result["audio"], 1)

    def test_partial_limits(self):
        proc = _make_qwen_processor()
        result = proc._parse_limits({"image": 10})
        self.assertEqual(result["image"], 10)
        self.assertEqual(result["video"], 1)

    def test_invalid_type_returns_defaults(self):
        proc = _make_qwen_processor()
        result = proc._parse_limits("invalid")
        self.assertEqual(result, {"image": 1, "video": 1, "audio": 1})


class TestMMProcessorCheckMMLimits(unittest.TestCase):
    def test_within_limits(self):
        proc = _make_qwen_processor()
        proc.limit_mm_per_prompt = {"image": 2, "video": 1, "audio": 1}
        # Should not raise
        proc._check_mm_limits([1, 2], [1])

    def test_exceeds_image_limit(self):
        proc = _make_qwen_processor()
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        with self.assertRaises(ValueError):
            proc._check_mm_limits([1, 2], [])

    def test_exceeds_video_limit(self):
        proc = _make_qwen_processor()
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        with self.assertRaises(ValueError):
            proc._check_mm_limits([], [1, 2])

    def test_none_inputs_ok(self):
        proc = _make_qwen_processor()
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        # Should not raise
        proc._check_mm_limits(None, None)


class TestMMProcessorLimitMmPerPrompt(unittest.TestCase):
    def test_constructor_with_limit(self):
        tok = _make_mock_tokenizer()
        ip = _make_mock_image_processor()
        config = _make_mock_config()
        proc = QwenVLProcessor(
            tokenizer=tok, image_processor=ip, config=config,
            processor_kwargs={}, limit_mm_per_prompt={"image": 5},
        )
        self.assertEqual(proc.limit_mm_per_prompt["image"], 5)
        self.assertEqual(proc.limit_mm_per_prompt["video"], 1)

    def test_constructor_with_enable_processor_cache(self):
        tok = _make_mock_tokenizer()
        ip = _make_mock_image_processor()
        config = _make_mock_config()
        proc = QwenVLProcessor(
            tokenizer=tok, image_processor=ip, config=config,
            processor_kwargs={}, enable_processor_cache=True,
        )
        self.assertTrue(proc.enable_processor_cache)


# ===========================================================================
# Tests: Cache utilities (static methods)
# ===========================================================================

class TestMMProcessorCacheUtilities(unittest.TestCase):
    def test_get_processor_cache(self):
        mock_socket = MagicMock()
        import pickle
        mock_socket.recv_multipart.return_value = [b"", pickle.dumps(["item1", "item2"])]
        result = MMProcessor.get_processor_cache(mock_socket, ["hash1", "hash2"])
        self.assertEqual(result, ["item1", "item2"])
        mock_socket.send_multipart.assert_called_once()

    def test_update_processor_cache(self):
        mock_socket = MagicMock()
        MMProcessor.update_processor_cache(mock_socket, ["hash1"], [("data1", {})])
        mock_socket.send_multipart.assert_called_once()


if __name__ == "__main__":
    unittest.main()
