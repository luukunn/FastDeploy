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

"""Shared video utilities: VideoReaderWrapper, read_video_decord, and sample_frames."""

import io
import math
import os
from tempfile import NamedTemporaryFile as ntf
from typing import Optional, Union

import numpy as np

from fastdeploy.input.image_processors.common import ceil_by_factor, floor_by_factor
from fastdeploy.utils import data_processor_logger

__all__ = [
    "VideoReaderWrapper",
    "read_video_decord",
    "sample_frames",
    "sample_frames_qwen",
    "sample_frames_paddleocr",
    "get_frame_indices_ernie",
    "read_frames_decord_ernie",
    "timestamp_converting",
    "render_frame_timestamp",
]


# ---------------------------------------------------------------------------
# VideoReaderWrapper
# ---------------------------------------------------------------------------


def _is_gif(data: bytes) -> bool:
    """Check if bytes represent a GIF based on magic header."""
    return data[:6] in (b"GIF87a", b"GIF89a")


class VideoReaderWrapper:
    """decord.VideoReader wrapper that fixes a memory leak and adds GIF support.

    Reference: https://github.com/dmlc/decord/issues/208
    """

    def __init__(self, video_path, *args, **kwargs):
        import decord

        try:
            # moviepy 1.0
            import moviepy.editor as mp
        except Exception:
            # moviepy 2.0
            import moviepy as mp

        with ntf(delete=True, suffix=".gif") as gif_file:
            gif_input = None
            self.original_file = None  # only set when we create a temp file

            if isinstance(video_path, str):
                if video_path.lower().endswith(".gif"):
                    gif_input = video_path
            elif isinstance(video_path, bytes):
                if _is_gif(video_path):
                    gif_file.write(video_path)
                    gif_file.flush()
                    gif_input = gif_file.name
            elif isinstance(video_path, io.BytesIO):
                video_path.seek(0)
                tmp_bytes = video_path.read()
                video_path.seek(0)
                if _is_gif(tmp_bytes):
                    gif_file.write(tmp_bytes)
                    gif_file.flush()
                    gif_input = gif_file.name

            if gif_input is not None:
                clip = mp.VideoFileClip(gif_input)
                mp4_file = ntf(delete=False, suffix=".mp4")
                mp4_path = mp4_file.name
                mp4_file.close()  # close before moviepy writes
                clip.write_videofile(mp4_path, verbose=False, logger=None)
                clip.close()
                video_path = mp4_path
                self.original_file = video_path  # temp mp4, cleaned up in __del__

            self._reader = decord.VideoReader(video_path, *args, **kwargs)
            self._reader.seek(0)

    def __len__(self):
        return len(self._reader)

    def __getitem__(self, key):
        frames = self._reader[key]
        self._reader.seek(0)
        return frames

    def get_avg_fps(self):
        return self._reader.get_avg_fps()

    def seek(self, pos):
        return self._reader.seek(pos)

    def __del__(self):
        original_file = getattr(self, "original_file", None)
        if original_file:
            try:
                os.remove(original_file)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# read_video_decord
# ---------------------------------------------------------------------------


def read_video_decord(video_path, save_to_disk: bool = False):
    """Load a video file and return (video_reader, video_meta, video_path).

    video_meta contains keys: "fps", "duration", "num_of_frame".
    """
    if isinstance(video_path, VideoReaderWrapper):
        video_reader = video_path
    else:
        if isinstance(video_path, bytes):
            video_path = io.BytesIO(video_path)
        video_reader = VideoReaderWrapper(video_path, num_threads=1)

    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    video_meta = {"fps": fps, "duration": duration, "num_of_frame": vlen}
    return video_reader, video_meta, video_path


# ---------------------------------------------------------------------------
# sample_frames — qwen_vl variant
# ---------------------------------------------------------------------------


def sample_frames_qwen(
    frame_factor: int,
    min_frames: int,
    max_frames: int,
    metadata: Optional[dict] = None,
    fps: Optional[Union[int, float]] = -1,
    num_frames: Optional[int] = -1,
) -> np.ndarray:
    """Sample frame indices — qwen_vl variant.

    Sentinel defaults are -1. Applies ceil_by_factor on min_frames and ensures
    num_frames is divisible by 4.
    """
    if fps > 0 and num_frames > 0:
        raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

    if metadata is None:
        raise ValueError("metadata is required for sample_frames_qwen")

    total_num_frames = metadata["num_of_frame"]

    if num_frames > 0:
        num_frames = round(num_frames / frame_factor) * frame_factor
    elif fps > 0:
        min_frames = ceil_by_factor(min_frames, frame_factor)
        max_frames = floor_by_factor(min(max_frames, total_num_frames), frame_factor)

        num_frames = total_num_frames / metadata["fps"] * fps

        if num_frames > total_num_frames:
            data_processor_logger.warning(f"smart_nframes: nframes[{num_frames}] > total_frames[{total_num_frames}]")

        num_frames = min(min(max(num_frames, min_frames), max_frames), total_num_frames)
        num_frames = floor_by_factor(num_frames, frame_factor)

    if num_frames > total_num_frames:
        raise ValueError(
            f"Video can't be sampled. The inferred `num_frames={num_frames}` exceeds "
            f"`total_num_frames={total_num_frames}`. "
            "Decrease `num_frames` or `fps` for sampling."
        )

    # num_frames must be divisible by 4
    if num_frames > 2 and num_frames % 4 != 0:
        num_frames = (num_frames // 4) * 4
        total_num_frames = (total_num_frames // 4) * 4
        num_frames = min(min(max(num_frames, min_frames), max_frames), total_num_frames)

    if num_frames > 0:
        indices = np.arange(0, total_num_frames, total_num_frames / num_frames).astype(np.int32)
    else:
        indices = np.arange(0, total_num_frames).astype(np.int32)

    return indices


# ---------------------------------------------------------------------------
# sample_frames — paddleocr_vl / ernie4_5_vl variant
# ---------------------------------------------------------------------------


def sample_frames_paddleocr(
    frame_factor: int,
    min_frames: int,
    max_frames: int,
    metadata: Optional[dict] = None,
    fps: Optional[Union[int, float]] = None,
    num_frames: Optional[int] = None,
) -> np.ndarray:
    """Sample frame indices — paddleocr_vl / ernie4_5_vl variant.

    Sentinel defaults are None. Uses plain math.floor/ceil; no %4 correction.
    """
    fps = fps or 0
    num_frames = num_frames or 0
    if fps > 0 and num_frames > 0:
        raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

    if metadata is None:
        raise ValueError("metadata is required for sample_frames_paddleocr")

    total_num_frames = metadata["num_of_frame"]

    if num_frames > 0:
        num_frames = round(num_frames / frame_factor) * frame_factor
    elif fps > 0:
        max_frames = math.floor(min(max_frames, total_num_frames) / frame_factor) * frame_factor
        num_frames = total_num_frames / metadata["fps"] * fps
        num_frames = min(min(max(num_frames, min_frames), max_frames), total_num_frames)
        num_frames = math.floor(num_frames / frame_factor) * frame_factor

    if num_frames > total_num_frames:
        raise ValueError(
            f"Video can't be sampled. The inferred `num_frames={num_frames}` exceeds "
            f"`total_num_frames={total_num_frames}`. "
            "Decrease `num_frames` or `fps` for sampling."
        )

    if num_frames > 0:
        indices = np.arange(0, total_num_frames, total_num_frames / num_frames).astype(np.int32)
    else:
        indices = np.arange(0, total_num_frames).astype(np.int32)

    return indices


def sample_frames(
    frame_factor: int,
    min_frames: int,
    max_frames: int,
    metadata: Optional[dict] = None,
    fps: Optional[Union[int, float]] = None,
    num_frames: Optional[int] = None,
    variant: str = "paddleocr",
) -> np.ndarray:
    """Dispatch to sample_frames_qwen or sample_frames_paddleocr based on variant."""
    if variant == "qwen":
        _fps = fps if fps is not None else -1
        _num_frames = num_frames if num_frames is not None else -1
        return sample_frames_qwen(frame_factor, min_frames, max_frames, metadata, _fps, _num_frames)
    if variant == "paddleocr":
        return sample_frames_paddleocr(frame_factor, min_frames, max_frames, metadata, fps, num_frames)
    raise ValueError(f"Unknown variant {variant!r}. Expected 'paddleocr' or 'qwen'.")


# ---------------------------------------------------------------------------
# Ernie-style video frame utilities
# ---------------------------------------------------------------------------

_FONT_PATH = os.path.join(os.path.dirname(__file__), "ernie4_5_vl_processor", "utils", "Roboto-Regular.ttf")


def get_frame_indices_ernie(
    vlen,
    target_frames=-1,
    target_fps=-1,
    frames_sample="middle",
    fix_start=None,
    input_fps=-1,
):
    """Compute frame indices for ernie-style video sampling."""
    import random

    assert frames_sample in ["rand", "middle", "leading"]
    if target_frames > 0:
        assert target_fps <= 0, "target_fps must be negative if target_frames is given."
        if target_frames > vlen:
            acc_samples = vlen
            data_processor_logger.info(
                f"target_frames={target_frames} is larger than video length {vlen}, "
                f"will sample {acc_samples} frames."
            )
        else:
            acc_samples = target_frames

        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = [(interv, intervals[idx + 1] - 1) for idx, interv in enumerate(intervals[:-1])]

        if frames_sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif frames_sample == "leading":
            frame_indices = [x[0] for x in ranges]
        elif frames_sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

    elif target_fps > 0:
        assert target_frames <= 0, "target_frames must be negative if target_fps is given."
        assert input_fps > 0, "input_fps must be provided if target_fps is given."
        duration = float(vlen) / input_fps
        delta = 1 / target_fps
        if frames_sample == "middle":
            frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        elif frames_sample == "leading":
            frame_seconds = np.arange(0, duration, delta)
        if frames_sample == "rand":
            frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
            rand_offset = np.random.rand(*(frame_seconds.shape)) - 0.5
            frame_seconds += rand_offset * delta
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
    else:
        raise ValueError("Must provide either positive target_fps or positive target_frames.")

    return frame_indices


def read_frames_decord_ernie(
    video_path,
    video_reader,
    video_meta,
    target_frames=-1,
    target_fps=-1,
    frames_sample="middle",
    fix_start=None,
    save_to_disk=False,
    frame_indices=None,
    tol=10,
):
    """Read specific frames from a video reader with retry logic (ernie-style)."""
    from PIL import Image as _Image

    if frame_indices is None:
        frame_indices = get_frame_indices_ernie(
            video_meta["num_of_frame"],
            target_frames=target_frames,
            target_fps=target_fps,
            frames_sample=frames_sample,
            fix_start=fix_start,
            input_fps=video_meta["fps"],
        )

    frames = []
    for frame_indice_index in range(len(frame_indices)):
        frame_indice = frame_indices[frame_indice_index]
        try:
            frames.append(video_reader[frame_indice].asnumpy())
        except Exception:
            previous_counter = 1
            later_counter = 1
            previous_after_flag = True
            cur_tol = tol * 2 if frame_indice in (0, len(video_reader) - 1) else tol
            while previous_counter < cur_tol or later_counter < cur_tol:
                if previous_after_flag:
                    if frame_indice - previous_counter < 0:
                        previous_counter += 1
                        previous_after_flag = not previous_after_flag
                        continue
                    try:
                        frames.append(video_reader[frame_indice - previous_counter].asnumpy())
                        frame_indices[frame_indice_index] = frame_indice - previous_counter
                        break
                    except Exception:
                        previous_counter += 1
                else:
                    if frame_indice + later_counter >= len(video_reader):
                        later_counter += 1
                        previous_after_flag = not previous_after_flag
                        continue
                    try:
                        frames.append(video_reader[frame_indice + later_counter].asnumpy())
                        frame_indices[frame_indice_index] = frame_indice + later_counter
                        break
                    except Exception:
                        later_counter += 1
                previous_after_flag = not previous_after_flag

    frames = np.stack(frames, axis=0)
    assert len(frames) == len(frame_indices)

    ret = []
    for frame in frames:
        ret.append(_Image.fromarray(frame, "RGB"))

    time_stamps = [fi * video_meta["duration"] / video_meta["num_of_frame"] for fi in frame_indices]
    return ret, frame_indices, time_stamps


def timestamp_converting(time_stamp_in_seconds):
    """Convert timestamp from seconds to HH:MM:SS.ss format."""
    hours = 0
    while time_stamp_in_seconds >= 3600:
        hours += 1
        time_stamp_in_seconds -= 3600
    mins = 0
    while time_stamp_in_seconds >= 60:
        mins += 1
        time_stamp_in_seconds -= 60
    return f"{int(hours):02d}:{int(mins):02d}:{time_stamp_in_seconds:05.02f}"


def render_frame_timestamp(frame, timestamp, font_rate=0.1, font_path=None):
    """Render a timestamp onto a video frame (PIL Image)."""
    from PIL import ImageDraw, ImageFont

    if font_path is None:
        font_path = _FONT_PATH
    time_stamp = "time: " + timestamp_converting(timestamp)

    draw = ImageDraw.Draw(frame)
    width, height = frame.size
    font_size = int(min(width, height) * font_rate)
    outline_size = int(font_size * 0.1)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(
        (0, 0),
        time_stamp,
        font=font,
        fill=(0, 0, 0),
        stroke_width=outline_size,
        stroke_fill=(255, 255, 255),
    )
    return frame
