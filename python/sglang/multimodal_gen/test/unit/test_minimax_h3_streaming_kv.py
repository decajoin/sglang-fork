# SPDX-License-Identifier: Apache-2.0
"""Cross-chunk KV streaming: geometry, history merge, and retention."""

import pytest
import torch

from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    _FRAME_PER_TOKEN,
    _T_GROUP,
    _video_t_grid,
    minimax_h3_packed_sequence,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.streaming_kv import (
    MINIMAX_H3_AUDIO_TOKEN_TAG,
    MINIMAX_H3_VIDEO_TOKEN_TAG,
    MiniMaxH3StreamingChunkContext,
    MiniMaxH3StreamingKVCache,
    minimax_h3_renormalize_video_rows,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_audio_latent_boundary,
    minimax_h3_frame_count_from_video_latent_t,
    minimax_h3_streaming_chunk_plan,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


# --------------------------------------------------------------------------
# the history merge -- the piece the whole design rests on
# --------------------------------------------------------------------------


@requires_cuda
def test_history_merge_equals_attention_over_the_union():
    """Merging the two partials must equal attending over the joined K/V.

    History is folded in without touching the current chunk's own attention
    call, which is only sound if the two-term logsumexp combine is exact.
    """
    from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
    from sglang.multimodal_gen.runtime.layers.usp import _ring_merge_attention

    torch.manual_seed(0)
    tokens, history_tokens, heads, dim = 256, 384, 4, 128
    device, dtype = "cuda", torch.bfloat16
    scale = dim**-0.5

    def attend(q, k, v, lse):
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=torch.tensor([0, q.shape[0]], dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor([0, k.shape[0]], dtype=torch.int32, device=device),
            max_seqlen_q=int(q.shape[0]),
            max_seqlen_k=int(k.shape[0]),
            softmax_scale=scale,
            causal=False,
            ver=3,
            return_softmax_lse=lse,
        )

    shape = (tokens, heads, dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k_cur = torch.randn(shape, device=device, dtype=dtype)
    v_cur = torch.randn(shape, device=device, dtype=dtype)
    k_old = torch.randn(history_tokens, heads, dim, device=device, dtype=dtype)
    v_old = torch.randn(history_tokens, heads, dim, device=device, dtype=dtype)

    joined = attend(q, torch.cat([k_old, k_cur]), torch.cat([v_old, v_cur]), False)
    joined = joined[0] if isinstance(joined, tuple) else joined

    out_cur, lse_cur = attend(q, k_cur, v_cur, True)[:2]
    out_old, lse_old = attend(q, k_old, v_old, True)[:2]
    merged, merged_lse = _ring_merge_attention(None, None, out_cur, lse_cur)
    merged, _ = _ring_merge_attention(merged, merged_lse, out_old, lse_old)

    error = (merged.float() - joined.float()).abs()
    reference = joined.float().abs().mean()
    assert float(error.mean() / reference) < 1e-2
    # The current chunk alone is a different answer, so the test would notice
    # a merge that silently dropped the history.
    assert float((out_cur.float() - joined.float()).abs().mean() / reference) > 0.1


# --------------------------------------------------------------------------
# continuation geometry
# --------------------------------------------------------------------------


def test_any_thirty_five_latents_span_one_hundred_nineteen_frames():
    """The steady continuation length holds at every phase of the pattern."""
    for origin in (0, 37, 72, 107, 142, 1000):
        span = sum(
            _FRAME_PER_TOKEN[(origin + index) % _T_GROUP] for index in range(35)
        )
        assert span == 119


def test_kv_chunk_plan_decodes_to_its_published_length():
    """Joined latents must decode to exactly the frames the plan promises."""
    plan = minimax_h3_streaming_chunk_plan(
        total_duration_seconds=60.0, chunk_seconds=5.0
    )
    assert plan.frames_per_chunk == 124
    assert plan.continuation_frames == 119
    assert plan.video_latent_t_for_chunk(0) == 37
    assert plan.video_latent_t_for_chunk(1) == 35

    total_latents = sum(
        plan.video_latent_t_for_chunk(index) for index in range(plan.chunk_count)
    )
    assert (
        minimax_h3_frame_count_from_video_latent_t(total_latents)
        == plan.published_frames
    )
    assert plan.published_duration_seconds >= 60.0


def test_audio_latents_track_the_global_frame_timeline():
    """Per-chunk rounding must not let audio drift away from the video."""
    plan = minimax_h3_streaming_chunk_plan(
        total_duration_seconds=60.0, chunk_seconds=5.0
    )
    counts = [
        plan.audio_latent_t_for_chunk(index) for index in range(plan.chunk_count)
    ]
    assert counts[0] == 207
    # A 119-frame chunk is 198.33 audio latents, so continuations alternate.
    assert set(counts[1:]) <= {198, 199}
    drift = abs(sum(counts) / 40.0 - plan.published_frames / 24.0)
    assert drift < 0.02

    offsets = [
        plan.audio_latent_offset_for_chunk(index) for index in range(plan.chunk_count)
    ]
    assert offsets[0] == 0
    for index in range(1, plan.chunk_count):
        assert offsets[index] == offsets[index - 1] + counts[index - 1]


def test_audio_latent_boundary_rounds_half_to_even():
    assert minimax_h3_audio_latent_boundary(0) == 0
    assert minimax_h3_audio_latent_boundary(124) == 207
    # 3 frames -> exactly 5.0, already integral; 6 frames -> 10.0.
    assert minimax_h3_audio_latent_boundary(3) == 5
    assert minimax_h3_audio_latent_boundary(6) == 10


# --------------------------------------------------------------------------
# the RoPE timeline
# --------------------------------------------------------------------------


def test_video_time_grid_default_is_the_standalone_grid():
    """The streaming parameters must not disturb an ordinary request."""
    grid = _video_t_grid(37, 124.0)
    expected = [124.0]
    for index in range(36):
        expected.append(expected[-1] + (5.0 / 3.0) * _FRAME_PER_TOKEN[index % 5])
    assert torch.allclose(grid, torch.tensor(expected, dtype=torch.float64))


def test_continuation_time_grid_continues_the_clip():
    first = _video_t_grid(37, 124.0)
    second = _video_t_grid(35, 124.0, latent_index_origin=37)
    step = (5.0 / 3.0) * _FRAME_PER_TOKEN[36 % 5]
    assert float(second[0]) == pytest.approx(float(first[-1]) + step)
    assert float(second[0]) > float(first[-1])


def test_packed_sequence_defaults_are_untouched():
    base = minimax_h3_packed_sequence(
        text_len=8, latent_t=7, latent_h=8, latent_w=8, audio_t=12,
        include_keyframe_cond=False,
    )
    explicit = minimax_h3_packed_sequence(
        text_len=8, latent_t=7, latent_h=8, latent_w=8, audio_t=12,
        include_keyframe_cond=False,
        video_latent_index_origin=0,
        audio_latent_index_origin=0,
        media_time_origin=None,
    )
    assert torch.equal(base["img_position_ids"], explicit["img_position_ids"])
    # Text still opens at zero, which is what a standalone request has always had.
    assert float(base["img_position_ids"][0, 0]) == 0.0


def test_continuation_prompt_is_right_aligned_to_its_media():
    packed = minimax_h3_packed_sequence(
        text_len=8, latent_t=5, latent_h=8, latent_w=8, audio_t=12,
        include_keyframe_cond=False,
        video_latent_index_origin=37,
        audio_latent_index_origin=207,
        media_time_origin=8.0,
    )
    grid = packed["img_position_ids"]
    text_stop = float(grid[7, 0]) + 1.0
    first_video = float(grid[packed["img_pos"][0], 0])
    assert text_stop == pytest.approx(first_video)


def test_a_continuation_anchors_only_on_its_first_frame():
    """kv_anchor pins the boundary frame; a last-frame anchor is unknowable."""
    packed = minimax_h3_packed_sequence(
        text_len=8, latent_t=5, latent_h=8, latent_w=8, audio_t=12,
        include_keyframe_cond=True,
        keyframe_frame_indices=[0],
        frame_count=119,
        video_latent_index_origin=37,
        media_time_origin=8.0,
    )
    grid = packed["img_position_ids"]
    # The anchor sits at this chunk's own media start, not at the clip's.
    cond_row = packed["img_pos"][0]
    first_target = packed["img_pos"][(8 // 2) * (8 // 2)]
    assert float(grid[cond_row, 0]) == pytest.approx(float(grid[first_target, 0]))

    with pytest.raises(ValueError, match="anchor on its first frame"):
        minimax_h3_packed_sequence(
            text_len=8, latent_t=5, latent_h=8, latent_w=8, audio_t=12,
            include_keyframe_cond=True,
            keyframe_frame_indices=[-1],
            frame_count=119,
            video_latent_index_origin=37,
        )


def test_anchor_rows_come_from_a_single_frame_latent():
    """Only a latent whose global index is a multiple of five spans one frame."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.streaming_kv import (
        minimax_h3_streaming_anchor_rows,
    )

    latents = torch.arange(1 * 24 * 37 * 8 * 8, dtype=torch.float32).reshape(
        1, 24, 37, 8, 8
    )
    rows = minimax_h3_streaming_anchor_rows(latents, latent_index_origin=0)
    assert tuple(rows.shape) == ((8 // 2) * (8 // 2), 96)
    # Latent 35 is the last index divisible by five in 0..36.
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
    )

    expected = minimax_h3_patchify_video_latent(
        latents[:, :, 35:36], patch_size=[1, 2, 2]
    )
    assert torch.equal(rows, expected)

    # A continuation starting at 37 ends at 71, whose last such index is 70.
    shifted = torch.randn(1, 24, 35, 8, 8)
    rows = minimax_h3_streaming_anchor_rows(shifted, latent_index_origin=37)
    assert torch.equal(
        rows, minimax_h3_patchify_video_latent(shifted[:, :, 33:34], patch_size=[1, 2, 2])
    )


# --------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------


def _commit(cache, *, video_rows, audio_rows, layers=("blocks.0.attn", "blocks.1.attn")):
    tags = torch.cat(
        (
            torch.full((video_rows,), MINIMAX_H3_VIDEO_TOKEN_TAG, dtype=torch.long),
            torch.full((audio_rows,), MINIMAX_H3_AUDIO_TOKEN_TAG, dtype=torch.long),
        )
    )
    cache.begin_commit(tags)
    for name in layers:
        rows = video_rows + audio_rows
        cache.stage(name, torch.randn(rows, 2, 4), torch.randn(rows, 2, 4))
    cache.commit()


def test_only_the_main_stack_is_history():
    cache = MiniMaxH3StreamingKVCache()
    assert cache.accepts("blocks.0.attn")
    assert cache.accepts("blocks.49.attn")
    assert not cache.accepts("token_refiner.blocks.0.attn")
    assert not cache.accepts("blocks.0.mlp")


def test_retention_keeps_a_video_sink_and_reaches_a_steady_size():
    cache = MiniMaxH3StreamingKVCache()
    _commit(cache, video_rows=10, audio_rows=4)
    cache.retain_sink_and_recent(1)
    assert (cache.history_video_tokens, cache.history_audio_tokens) == (10, 4)

    sizes = []
    for _ in range(4):
        _commit(cache, video_rows=8, audio_rows=3)
        cache.retain_sink_and_recent(1)
        sizes.append((cache.history_video_tokens, cache.history_audio_tokens))
    # The opening chunk stays as a video-only sink; its audio is not context
    # for a later sentence. Everything between is dropped, which is the bound.
    assert sizes == [(18, 3)] * 4
    assert cache.history("blocks.0.attn")[0].shape[0] == 21


@requires_cuda
def test_trimming_does_not_hold_two_histories_on_the_device():
    """Trimming must free each layer before allocating the next.

    Holding ``self._history.items()`` in a list pins all fifty layers of the
    old history for the whole loop, so trimming costs twice the cache. At 768p
    that transient is ~25 GB and is what put the run over an H200.
    """
    cache = MiniMaxH3StreamingKVCache()
    layers = [f"blocks.{index}.attn" for index in range(50)]
    rows = 600

    for _ in range(2):
        tags = torch.zeros(rows, dtype=torch.long)
        cache.begin_commit(tags)
        for name in layers:
            cache.stage(
                name,
                torch.randn(rows, 8, 64, device="cuda", dtype=torch.bfloat16),
                torch.randn(rows, 8, 64, device="cuda", dtype=torch.bfloat16),
            )
        cache.commit()

    resident = sum(
        key.numel() * key.element_size() + value.numel() * value.element_size()
        for key, value in (cache.history(name) for name in layers)
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    # Keeps both chunks, so the trim rewrites every layer at full size -- the
    # worst case for the transient without shrinking the result.
    cache.retain_sink_and_recent(1, video_only_sink=False)
    torch.cuda.synchronize()
    overhead = torch.cuda.max_memory_allocated() - before
    assert cache.history_video_tokens == 2 * rows
    # One layer in flight is 2/50 of the cache; two whole histories would be 100%.
    assert overhead < resident * 0.25, (
        f"trim transient {overhead / 2**20:.0f} MiB against a "
        f"{resident / 2**20:.0f} MiB cache"
    )


@pytest.mark.parametrize("chunks", [1, 2, 3, 12, 24])
def test_decode_windows_tile_the_joined_latents_exactly(chunks):
    """The windowed decode must cover the join with nothing left over.

    A window is only decodable where the (1,4,4,4,4) frame weighting is in
    phase -- a latent index divisible by 5 -- and its length has to satisfy the
    5n+2 contract. 37 latents at stride 35 is the only choice that also
    advances exactly one chunk.
    """
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.streaming_chunk import (
        MINIMAX_H3_DECODE_AUDIO_STUB_LATENTS,
        MINIMAX_H3_DECODE_WINDOW_LATENTS,
        MINIMAX_H3_DECODE_WINDOW_OVERLAP_FRAMES,
        MINIMAX_H3_DECODE_WINDOW_STRIDE,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
        minimax_h3_frame_count_from_video_latent_t,
    )

    window = MINIMAX_H3_DECODE_WINDOW_LATENTS
    stride = MINIMAX_H3_DECODE_WINDOW_STRIDE
    plan = minimax_h3_streaming_chunk_plan(
        total_duration_seconds=(124 + (chunks - 1) * 119) / 24,
        chunk_seconds=5.0,
    )
    assert plan.chunk_count == chunks
    total = sum(
        plan.video_latent_t_for_chunk(index) for index in range(chunks)
    )

    starts = list(range(0, total - window + 1, stride))
    assert len(starts) == chunks
    # Lands on the last latent with nothing spare.
    assert starts[-1] + window == total
    # Every window is in phase and independently decodable.
    assert all(start % 5 == 0 for start in starts)
    assert all(
        minimax_h3_frame_count_from_video_latent_t(window) == plan.frames_per_chunk
        for _ in starts
    )

    published = plan.frames_per_chunk + sum(
        plan.frames_per_chunk - MINIMAX_H3_DECODE_WINDOW_OVERLAP_FRAMES
        for _ in starts[1:]
    )
    assert published == plan.published_frames
    assert MINIMAX_H3_DECODE_AUDIO_STUB_LATENTS >= 1


def test_a_failed_commit_leaves_the_history_alone():
    cache = MiniMaxH3StreamingKVCache()
    _commit(cache, video_rows=6, audio_rows=2)
    before = cache.history("blocks.0.attn")[0].clone()

    cache.begin_commit(torch.zeros(4, dtype=torch.long))
    cache.stage("blocks.0.attn", torch.randn(4, 2, 4), torch.randn(4, 2, 4))
    cache.rollback()
    assert torch.equal(cache.history("blocks.0.attn")[0], before)
    assert cache.committed_chunks == 1


def test_commit_rejects_non_media_rows():
    cache = MiniMaxH3StreamingKVCache()
    with pytest.raises(ValueError, match="video and audio rows"):
        cache.begin_commit(torch.tensor([0, 1, 2], dtype=torch.long))


def test_commit_rejects_a_row_count_that_disagrees_with_its_tags():
    cache = MiniMaxH3StreamingKVCache()
    cache.begin_commit(torch.zeros(6, dtype=torch.long))
    with pytest.raises(ValueError, match="describes"):
        cache.stage("blocks.0.attn", torch.randn(5, 2, 4), torch.randn(5, 2, 4))


# --------------------------------------------------------------------------
# latent renormalization
# --------------------------------------------------------------------------


def test_renormalization_anchors_later_chunks_on_the_first():
    context = MiniMaxH3StreamingChunkContext(
        cache=None, chunk_index=0, video_latent_index_origin=0,
        audio_latent_index_origin=0,
    )
    torch.manual_seed(0)
    first = torch.randn(64, 96) * 2.0 + 5.0
    assert minimax_h3_renormalize_video_rows(first, context) is first

    drifted = torch.randn(48, 96) * 7.0 - 3.0
    fixed = minimax_h3_renormalize_video_rows(drifted, context)
    assert float(fixed.mean()) == pytest.approx(float(first.mean()), abs=1e-3)
    assert float(fixed.std()) == pytest.approx(float(first.std()), abs=1e-2)


# --------------------------------------------------------------------------
# request geometry
# --------------------------------------------------------------------------


def test_the_request_resolves_the_continuation_geometry():
    """The API process must publish the length the worker will actually join.

    Delivery validation runs on the queued request, so a request that resolved
    the first chunk's standalone geometry for every chunk would only surface as
    a frame-count mismatch after the whole clip had been generated.
    """
    from types import SimpleNamespace

    params = MiniMaxH3SamplingParams(prompt="x", total_duration_seconds=60.0)
    params._adjust_streaming(
        SimpleNamespace(enable_streaming=True, streaming_chunk_seconds=5.0)
    )
    # 124 for the opening chunk, 119 for each of the twelve continuations.
    assert params.streaming_published_frames == 124 + 12 * 119 == 1552
