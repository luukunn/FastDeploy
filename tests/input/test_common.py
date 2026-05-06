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

import numpy as np

from fastdeploy.input.multimodal.common import (
    ceil_by_factor,
    floor_by_factor,
    is_scaled_image,
    round_by_factor,
    smart_resize,
    smart_resize_paddleocr,
    smart_resize_qwen,
)


# ===========================================================================
# Tests: round_by_factor, ceil_by_factor, floor_by_factor
# ===========================================================================

class TestRoundByFactor(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(round_by_factor(28, 28), 28)

    def test_round_up(self):
        self.assertEqual(round_by_factor(15, 28), 28)

    def test_round_down(self):
        self.assertEqual(round_by_factor(13, 28), 0)

    def test_large_factor(self):
        self.assertEqual(round_by_factor(100, 28), 112)


class TestCeilByFactor(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(ceil_by_factor(28, 28), 28)

    def test_rounds_up(self):
        self.assertEqual(ceil_by_factor(29, 28), 56)

    def test_one_less(self):
        self.assertEqual(ceil_by_factor(27, 28), 28)


class TestFloorByFactor(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(floor_by_factor(56, 28), 56)

    def test_rounds_down(self):
        self.assertEqual(floor_by_factor(55, 28), 28)

    def test_one_more(self):
        self.assertEqual(floor_by_factor(57, 28), 56)


# ===========================================================================
# Tests: is_scaled_image
# ===========================================================================

class TestIsScaledImage(unittest.TestCase):
    def test_uint8_is_not_scaled(self):
        img = np.array([[[128, 255, 0]]], dtype=np.uint8)
        self.assertFalse(is_scaled_image(img))

    def test_float_0_to_1_is_scaled(self):
        img = np.array([[[0.5, 0.8, 0.1]]], dtype=np.float32)
        self.assertTrue(is_scaled_image(img))

    def test_float_0_to_255_is_not_scaled(self):
        img = np.array([[[128.0, 255.0, 0.0]]], dtype=np.float32)
        self.assertFalse(is_scaled_image(img))

    def test_all_zeros_float(self):
        img = np.zeros((1, 1, 3), dtype=np.float32)
        self.assertTrue(is_scaled_image(img))


# ===========================================================================
# Tests: smart_resize_qwen
# ===========================================================================

class TestSmartResizeQwen(unittest.TestCase):
    def test_basic_no_change_needed(self):
        h, w = smart_resize_qwen(224, 224, factor=28, min_pixels=28*28*4, max_pixels=28*28*6177)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

    def test_large_image_downscaled(self):
        h, w = smart_resize_qwen(10000, 10000, factor=28, min_pixels=28*28*4, max_pixels=28*28*100)
        self.assertLessEqual(h * w, 28 * 28 * 100)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

    def test_small_image_upscaled(self):
        h, w = smart_resize_qwen(10, 10, factor=28, min_pixels=28*28*4, max_pixels=28*28*6177)
        self.assertGreaterEqual(h * w, 28 * 28 * 4)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

    def test_extreme_aspect_ratio_clipped(self):
        # Very wide image: 10 x 5000
        h, w = smart_resize_qwen(10, 5000, factor=28, min_pixels=28*28*4, max_pixels=28*28*6177, max_ratio=200)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)
        # Ratio should be within limits
        ratio = max(h, w) / max(min(h, w), 1)
        self.assertLessEqual(ratio, 201)

    def test_square_image(self):
        h, w = smart_resize_qwen(112, 112, factor=28, min_pixels=28*28*4, max_pixels=28*28*100)
        self.assertEqual(h, 112)
        self.assertEqual(w, 112)

    def test_invalid_result_raises(self):
        # Pathological case that could produce invalid result
        # min_pixels > actual pixels after downscale should be caught
        with self.assertRaises(ValueError):
            smart_resize_qwen(1, 1, factor=28, min_pixels=28*28*1000, max_pixels=28*28*100)


# ===========================================================================
# Tests: smart_resize_paddleocr
# ===========================================================================

class TestSmartResizePaddleocr(unittest.TestCase):
    def test_basic(self):
        h, w = smart_resize_paddleocr(224, 224, factor=28, min_pixels=28*28*4, max_pixels=28*28*1280)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

    def test_small_height_rescaled(self):
        # height < factor => should be scaled up
        h, w = smart_resize_paddleocr(10, 100, factor=28, min_pixels=28*28*4, max_pixels=28*28*1280)
        self.assertGreaterEqual(h, 28)
        self.assertEqual(h % 28, 0)

    def test_small_width_rescaled(self):
        h, w = smart_resize_paddleocr(100, 10, factor=28, min_pixels=28*28*4, max_pixels=28*28*1280)
        self.assertGreaterEqual(w, 28)
        self.assertEqual(w % 28, 0)

    def test_extreme_aspect_ratio_raises(self):
        # Aspect ratio > 200 should raise
        with self.assertRaises(ValueError):
            smart_resize_paddleocr(28, 28 * 201, factor=28, min_pixels=28*28*4, max_pixels=28*28*1280)

    def test_large_image_downscaled(self):
        h, w = smart_resize_paddleocr(2000, 2000, factor=28, min_pixels=28*28*4, max_pixels=28*28*100)
        self.assertLessEqual(h * w, 28 * 28 * 100)

    def test_small_image_upscaled(self):
        h, w = smart_resize_paddleocr(30, 30, factor=28, min_pixels=28*28*130, max_pixels=28*28*1280)
        self.assertGreaterEqual(h * w, 28 * 28 * 130)


# ===========================================================================
# Tests: smart_resize dispatcher
# ===========================================================================

class TestSmartResizeDispatcher(unittest.TestCase):
    def test_qwen_variant(self):
        h, w = smart_resize(224, 224, factor=28, min_pixels=28*28*4, max_pixels=28*28*6177, variant="qwen")
        self.assertEqual(h % 28, 0)

    def test_paddleocr_variant(self):
        h, w = smart_resize(224, 224, factor=28, min_pixels=28*28*4, max_pixels=28*28*1280, variant="paddleocr")
        self.assertEqual(h % 28, 0)

    def test_default_is_qwen(self):
        h1, w1 = smart_resize(224, 224, factor=28, min_pixels=28*28*4, max_pixels=28*28*6177)
        h2, w2 = smart_resize_qwen(224, 224, factor=28, min_pixels=28*28*4, max_pixels=28*28*6177)
        self.assertEqual((h1, w1), (h2, w2))


if __name__ == "__main__":
    unittest.main()
