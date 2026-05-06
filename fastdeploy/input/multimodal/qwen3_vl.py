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

"""Qwen3-VL multimodal processor — thin inheritance from QwenVLProcessor.

Only overrides video pixel limits (Qwen3 passes min/max pixels for video).
"""

from fastdeploy.input.multimodal.qwen_vl import QwenVLProcessor


class Qwen3VLProcessor(QwenVLProcessor):
    """Multimodal processor for Qwen3-VL.

    Inherits all logic from QwenVLProcessor, only changing video pixel limits.
    The image processor is expected to be a Qwen3ImageProcessor instance
    (passed by the factory in preprocess.py).
    """

    # Qwen3 applies pixel limits to video preprocessing
    _video_min_pixels = 128 * 28 * 28
    _video_max_pixels = 768 * 28 * 28
