# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction


def minimax_h3_align_frame_count(frame_count: int) -> int:
    """Snap ``frame_count`` up to the MiniMax H3 17n+5 frame boundary."""
    if frame_count <= 0:
        return 1
    current = int(frame_count)
    return current + (5 - current) % 17


def minimax_h3_video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return ((int(frame_count) - 5) // 17) * 5 + 2


def minimax_h3_frame_count_from_video_latent_t(out_t: int) -> int:
    if out_t == 1:
        return 1
    if out_t < 2 or (out_t - 2) % 5 != 0:
        raise ValueError("MiniMax H3 video latent T must be 1 or match 5n+2")
    return 17 * ((int(out_t) - 2) // 5) + 5


def minimax_h3_audio_latent_t(duration_seconds: float) -> int:
    # Rounding happens at the 40 Hz audio latent boundary.
    return int(round(float(duration_seconds) * 40.0))


# A standalone H3 request opens with an affine prefix of 2 video latents
# spanning 5 frames; the steady state after it is whole 17-frame groups of 5
# latents. A continuation that carries its history as KV does not reproduce
# that prefix, so it generates the steady part alone.
MINIMAX_H3_STREAMING_PREFIX_LATENTS = 2
MINIMAX_H3_STREAMING_PREFIX_FRAMES = 5


@dataclass(frozen=True)
class MiniMaxH3StreamingChunkPlan:
    """Whole-chunk decomposition of one streaming long-video request.

    The first chunk is an ordinary 17n+5 request. A continuation inherits the
    past as attention K/V, so it drops the five-frame affine prefix that opens
    a standalone request and generates only the steady 17-frame groups. It
    reproduces nothing, so publication drops nothing: at the default 5s chunk
    that is 124 frames followed by 119 per continuation.
    """

    chunk_count: int
    frames_per_chunk: int
    continuation_frames: int
    fps: int

    @property
    def chunk_duration_seconds(self) -> float:
        return self.frames_per_chunk / self.fps

    @property
    def continuation_duration_seconds(self) -> float:
        return self.continuation_frames / self.fps

    @property
    def published_frames(self) -> int:
        return (
            self.frames_per_chunk
            + (self.chunk_count - 1) * self.continuation_frames
        )

    @property
    def published_duration_seconds(self) -> float:
        return self.published_frames / self.fps

    def frames_for_chunk(self, chunk_index: int) -> int:
        return self.frames_per_chunk if chunk_index == 0 else self.continuation_frames

    def video_latent_t_for_chunk(self, chunk_index: int) -> int:
        """Video latent count for one chunk.

        A continuation under ``kv`` is not a standalone request, so its latent
        count does not come from the 17n+5 frame formula -- that formula
        assumes the affine prefix this chunk deliberately omits.
        """
        first = minimax_h3_video_latent_t(self.frames_per_chunk)
        if chunk_index == 0:
            return first
        return first - MINIMAX_H3_STREAMING_PREFIX_LATENTS

    def frame_offset_for_chunk(self, chunk_index: int) -> int:
        """Global index of this chunk's first published frame."""
        if chunk_index <= 0:
            return 0
        return self.frames_per_chunk + (chunk_index - 1) * self.continuation_frames

    def audio_latent_offset_for_chunk(self, chunk_index: int) -> int:
        return minimax_h3_audio_latent_boundary(
            self.frame_offset_for_chunk(chunk_index), fps=self.fps
        )

    def audio_latent_t_for_chunk(self, chunk_index: int) -> int:
        """Audio latents this chunk owns on the clip's own 40 Hz grid.

        Rounding each chunk independently would let the error accumulate
        along the chain, so the boundaries are taken on the global frame
        timeline and differenced. A continuation therefore owns 198 or 199
        latents rather than a repeated request-local count.
        """
        start = self.frame_offset_for_chunk(chunk_index)
        stop = start + self.frames_for_chunk(chunk_index)
        return minimax_h3_audio_latent_boundary(
            stop, fps=self.fps
        ) - minimax_h3_audio_latent_boundary(start, fps=self.fps)

    def video_latent_offset_for_chunk(self, chunk_index: int) -> int:
        """Global index of this chunk's first video latent."""
        if chunk_index <= 0:
            return 0
        return self.video_latent_t_for_chunk(0) + (chunk_index - 1) * (
            self.video_latent_t_for_chunk(1)
        )


def minimax_h3_audio_latent_boundary(
    frame_index: int, *, fps: int = 24, rate: int = 40
) -> int:
    """Audio latent index at a video frame boundary, rounded half to even.

    Exact rational arithmetic: 40/24 is not representable in binary, and a
    float here would drift over a clip-length timeline.
    """
    value = Fraction(int(frame_index) * int(rate), int(fps))
    quotient, remainder = divmod(value.numerator, value.denominator)
    doubled = remainder * 2
    if doubled < value.denominator:
        return quotient
    if doubled > value.denominator:
        return quotient + 1
    return quotient + (quotient & 1)


def minimax_h3_streaming_chunk_plan(
    *,
    total_duration_seconds: float,
    chunk_seconds: float,
    fps: int = 24,
) -> MiniMaxH3StreamingChunkPlan:
    """Resolve a requested total duration into whole chunks.

    The frame grid is discrete, so the published duration is the smallest
    whole-chunk length that reaches the request. Callers report the resolved
    duration rather than trimming or retiming to the nominal one.
    """
    if not math.isfinite(total_duration_seconds) or total_duration_seconds <= 0:
        raise ValueError("total_duration_seconds must be a positive finite number")
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be a positive finite number")
    frames_per_chunk = minimax_h3_align_frame_count(
        int(round(float(chunk_seconds) * fps))
    )
    if frames_per_chunk <= 1:
        raise ValueError("a streaming chunk must span more than one frame")
    continuation_frames = frames_per_chunk - MINIMAX_H3_STREAMING_PREFIX_FRAMES
    if continuation_frames <= 0:
        raise ValueError(
            "a streaming chunk must be longer than the "
            f"{MINIMAX_H3_STREAMING_PREFIX_FRAMES}-frame affine prefix"
        )
    requested_frames = int(round(float(total_duration_seconds) * fps))
    remaining = requested_frames - frames_per_chunk
    chunk_count = 1 + max(0, math.ceil(remaining / continuation_frames))
    return MiniMaxH3StreamingChunkPlan(
        chunk_count=chunk_count,
        frames_per_chunk=frames_per_chunk,
        continuation_frames=continuation_frames,
        fps=fps,
    )


def minimax_h3_time_shift_sigmas(
    *,
    num_steps: int = 50,
    shift_scale: float = 6.0,
) -> list[float]:
    if shift_scale <= 0:
        raise ValueError("MiniMax H3 shift_scale must be > 0")
    if num_steps <= 0:
        raise ValueError("MiniMax H3 num_steps must be > 0")

    import torch

    # The rectified-flow sigma range is fixed at [1.0, 0.0].
    base = torch.linspace(
        1.0,
        0.0,
        int(num_steps),
        device="cpu",
        dtype=torch.float32,
    )
    shifted = float(shift_scale) * base / (1 + (float(shift_scale) - 1) * base)
    shifted, _ = torch.unique_consecutive(shifted, return_counts=True)
    # A one-point request is still exactly one point.  Normal serving uses
    # multiple points, but preserving the requested cardinality keeps
    # ``num_inference_steps`` the sole schedule-size control.
    if num_steps > 1 and shifted[-1].item() > 0.0:
        shifted = torch.cat([shifted, torch.tensor([0.0], dtype=shifted.dtype)])
    return [float(value) for value in shifted.tolist()]
