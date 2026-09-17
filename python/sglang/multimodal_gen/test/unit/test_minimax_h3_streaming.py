# SPDX-License-Identifier: Apache-2.0
"""Streaming long-video chunk decomposition and anchor handoff."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    _validate_fl2va_keyframe_payload,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.streaming_chunk import (
    MiniMaxH3StreamingChunkStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_align_frame_count,
    minimax_h3_streaming_chunk_plan,
)


def _server_args(chunk_seconds=5.0, enable_streaming=True):
    return SimpleNamespace(
        enable_streaming=enable_streaming,
        streaming_chunk_seconds=chunk_seconds,
    )


@pytest.mark.parametrize(
    "total_seconds,expected_chunks,expected_frames",
    [
        # 124 frames for the opening chunk, 119 for every continuation.
        (5.0, 1, 124),
        (10.0, 2, 243),
        (60.0, 13, 1552),
        (120.0, 25, 2980),
    ],
)
def test_chunk_plan_covers_the_request(
    total_seconds, expected_chunks, expected_frames
):
    plan = minimax_h3_streaming_chunk_plan(
        total_duration_seconds=total_seconds, chunk_seconds=5.0
    )
    assert plan.chunk_count == expected_chunks
    assert plan.frames_per_chunk == 124
    assert plan.published_frames == expected_frames
    # The grid is discrete, so the result reaches the request without trimming.
    assert plan.published_duration_seconds >= total_seconds


def test_chunk_plan_uses_the_aligned_chunk_length():
    plan = minimax_h3_streaming_chunk_plan(
        total_duration_seconds=30.0, chunk_seconds=8.0
    )
    assert plan.frames_per_chunk == minimax_h3_align_frame_count(8 * 24) == 192
    # A continuation drops the five-frame affine prefix, nothing else.
    assert plan.published_frames == 192 + (plan.chunk_count - 1) * 187


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_chunk_plan_rejects_degenerate_durations(bad):
    with pytest.raises(ValueError):
        minimax_h3_streaming_chunk_plan(
            total_duration_seconds=bad, chunk_seconds=5.0
        )


def test_anchor_payload_satisfies_the_fl2va_contract():
    """The hand-built continuation anchor must pass the real DiT-sink validator."""
    latent_h, latent_w, frame_count = 48, 84, 124
    rows = torch.zeros((latent_h // 2) * (latent_w // 2), 96, dtype=torch.float32)
    entry = {
        "rows": rows,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "canvas_height": latent_h * 16,
        "canvas_width": latent_w * 16,
        "frame_index": 0,
        "resolved_frame_index": 0,
        "condition_index": 0,
    }
    payload = {
        "rows": rows,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "canvas_height": latent_h * 16,
        "canvas_width": latent_w * 16,
        "keyframes": [entry],
        "semantic_frame_indices": (0,),
        "pixel_frame_indices": [0],
        "frame_count": frame_count,
    }
    _validate_fl2va_keyframe_payload(SimpleNamespace(task="fl2va"), payload)

    # A continuation is only legal once the plan says fl2va.
    with pytest.raises(ValueError):
        _validate_fl2va_keyframe_payload(SimpleNamespace(task="t2va"), payload)


def test_audio_is_fitted_to_the_published_video_length():
    fit = MiniMaxH3StreamingChunkStage._fit_audio
    sample_rate, fps, frames = 32_000, 24, 1477
    target = round(frames / fps * sample_rate)

    long = torch.zeros(1, 2, target + 5_000)
    assert int(fit(long, published_frames=frames, fps=fps, sample_rate=sample_rate).shape[2]) == target

    short = torch.zeros(1, 2, target - 5_000)
    fitted = fit(short, published_frames=frames, fps=fps, sample_rate=sample_rate)
    assert int(fitted.shape[2]) == target
    # Padding is silence, not repeated content.
    assert torch.all(fitted[:, :, -5_000:] == 0)

    exact = torch.zeros(1, 2, target)
    assert fit(exact, published_frames=frames, fps=fps, sample_rate=sample_rate) is exact


def test_av_alignment_stays_inside_the_delivery_tolerance():
    from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
        MiniMaxH3PipelineConfig,
    )

    plan = minimax_h3_streaming_chunk_plan(
        total_duration_seconds=60.0, chunk_seconds=5.0
    )
    sample_rate = MiniMaxH3PipelineConfig.output_audio_sample_rate
    samples = round(plan.published_frames / plan.fps * sample_rate)
    drift = abs(samples / sample_rate - plan.published_duration_seconds)
    assert drift <= MiniMaxH3PipelineConfig.output_av_drift_tolerance_s


def test_sampling_params_resolve_chunk_target():
    params = MiniMaxH3SamplingParams(prompt="x", total_duration_seconds=60.0)
    params._adjust_streaming(_server_args())
    # One chunk is what the 4-15s geometry contract sees.
    assert params.target["duration_seconds"] == pytest.approx(124 / 24)


def test_sampling_params_require_one_prompt_per_chunk():
    params = MiniMaxH3SamplingParams(
        prompt="x",
        total_duration_seconds=60.0,
        chunk_prompts=[f"p{index}" for index in range(13)],
    )
    params._adjust_streaming(_server_args())
    assert len(params.chunk_prompts) == 13

    params = MiniMaxH3SamplingParams(
        prompt="x", total_duration_seconds=60.0, chunk_prompts=["a", "b"]
    )
    with pytest.raises(ValueError, match="one prompt per resolved chunk"):
        params._adjust_streaming(_server_args())


def test_streaming_request_requires_the_server_flag():
    params = MiniMaxH3SamplingParams(prompt="x", total_duration_seconds=60.0)
    with pytest.raises(ValueError, match="--enable-streaming"):
        params._adjust_streaming(_server_args(enable_streaming=False))


def test_high_quality_is_rejected_for_streaming():
    params = MiniMaxH3SamplingParams(
        prompt="x", total_duration_seconds=60.0, quality="high"
    )
    with pytest.raises(ValueError, match="lossless"):
        params._adjust_streaming(_server_args())


def test_ordinary_requests_are_untouched():
    params = MiniMaxH3SamplingParams(prompt="x", target={"aspect_ratio": "16:9"})
    before = dict(params.target)
    params._adjust_streaming(_server_args())
    assert params.target == before
    assert params.total_duration_seconds is None

