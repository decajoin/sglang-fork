# SPDX-License-Identifier: Apache-2.0
"""Contract for the ref2va reference->target attention band drop.

Covers the three parts that can silently go wrong: the band the producer
derives from the packed layout, the gate that keeps fl2va/t2va out, and the
numerics of splitting one masked attention into three unmasked calls.
"""

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3DiTModel,
    MiniMaxH3SegmentBands,
    _minimax_h3_attention_core_impl,
    _minimax_h3_segment_sparse_attention,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    _segment_sparse_attn_enabled,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_TEXT_LEN = 3
_COMMON = dict(text_len=_TEXT_LEN, latent_t=2, latent_h=4, latent_w=4, audio_t=3)


def _branch(packed, *, segment_sparse_attn: bool) -> MiniMaxH3DenoiseBranch:
    return MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(_TEXT_LEN, 5120),
        token_tags=packed["token_tags"],
        device=torch.device("cpu"),
        segment_sparse_attn=segment_sparse_attn,
    )


def _ref2va_packed(ref_blocks):
    return minimax_h3_packed_sequence_ref2va_blocks(**_COMMON, ref_blocks=ref_blocks)


# --------------------------------------------------------------------------
# producer: which rows form the reference band
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref_blocks",
    [
        pytest.param([{"kind": "image", "latent_h": 4, "latent_w": 4}], id="image"),
        pytest.param(
            [
                {
                    "kind": "video_audio",
                    "ref_audio_t": 2,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 4,
                }
            ],
            id="video-audio-interleaved",
        ),
        pytest.param(
            [
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {"kind": "audio", "ref_audio_t": 2},
            ],
            id="image-plus-audio",
        ),
    ],
)
def test_restricted_ranges_are_exactly_the_visual_reference_rows(ref_blocks):
    """Only visual reference rows are restricted; reference audio is not.

    Reference audio is a tiny minority of the sequence, so its representation
    is almost entirely context-derived and cutting it off the target destroys
    soundtrack fidelity -- for ~1% of the saving.
    """
    packed = _ref2va_packed(ref_blocks)
    branch = _branch(packed, segment_sparse_attn=True)

    bounds = branch.segment_attn_bounds
    assert bounds is not None
    kv_stop, ranges = bounds

    restricted = torch.cat([torch.arange(a, b) for a, b in ranges]).sort().values
    assert torch.equal(restricted, branch.img_cond_seq_idx.sort().values)

    # Reference audio keeps full attention.
    audio_ref = set(branch.audio_ref_seq_idx.tolist())
    assert audio_ref.isdisjoint(restricted.tolist())

    # kv_stop still spans the whole reference run, so references key each other.
    reference_rows = (
        torch.cat([branch.img_cond_seq_idx, branch.audio_ref_seq_idx]).sort().values
    )
    assert torch.equal(reference_rows, torch.arange(_TEXT_LEN, kv_stop))

    # Target rows must stay strictly after kv_stop, or the split would drop
    # attention the target still needs.
    target_rows = torch.cat([branch.img_target_seq_idx, branch.audio_target_seq_idx])
    assert int(target_rows.min()) >= kv_stop


def test_audio_only_reference_disables_the_split():
    """With no visual reference there is nothing left to restrict."""
    packed = _ref2va_packed([{"kind": "audio", "ref_audio_t": 2}])
    branch = _branch(packed, segment_sparse_attn=True)

    assert branch.segment_attn_bounds is None
    assert "segment_attn_bounds" not in branch.static_kwargs


def test_video_block_audio_rows_split_the_restricted_range():
    """A video block packs its audio rows before its visual rows."""
    packed = _ref2va_packed(
        [
            {
                "kind": "video_audio",
                "ref_audio_t": 2,
                "latent_t": 2,
                "latent_h": 4,
                "latent_w": 4,
            }
        ]
    )
    branch = _branch(packed, segment_sparse_attn=True)
    kv_stop, ranges = branch.segment_attn_bounds

    # audio rows sit at [text_len, text_len+4), so the restricted run starts
    # after them rather than at text_len.
    assert len(ranges) == 1
    assert ranges[0][0] > _TEXT_LEN
    assert ranges[0][1] == kv_stop


def test_reference_band_is_absent_without_references():
    """t2va has no reference rows, so there is no band to drop."""
    packed = minimax_h3_packed_sequence(**_COMMON, include_keyframe_cond=False)
    branch = _branch(packed, segment_sparse_attn=True)

    assert branch.segment_attn_bounds is None
    assert "segment_attn_bounds" not in branch.static_kwargs


def test_band_is_not_published_when_disabled():
    """The kwarg only reaches the DiT when the request opted in."""
    packed = _ref2va_packed([{"kind": "image", "latent_h": 4, "latent_w": 4}])
    branch = _branch(packed, segment_sparse_attn=False)

    assert branch.segment_attn_bounds is None
    assert "segment_attn_bounds" not in branch.static_kwargs


def test_band_reaches_the_dit_kwargs_when_enabled():
    packed = _ref2va_packed([{"kind": "image", "latent_h": 4, "latent_w": 4}])
    branch = _branch(packed, segment_sparse_attn=True)

    assert branch.static_kwargs["segment_attn_bounds"] == branch.segment_attn_bounds


# --------------------------------------------------------------------------
# gate: which tasks may drop the band
# --------------------------------------------------------------------------


class _Material:
    def __init__(self, chain: str) -> None:
        self.material_chain = chain


class _Plan:
    def __init__(self, task: str, chains: tuple[str, ...]) -> None:
        self.task = task
        self.materials = tuple(_Material(chain) for chain in chains)


class _Args:
    def __init__(self, enabled: bool) -> None:
        self.minimax_h3_segment_sparse_attn = enabled


@pytest.mark.parametrize(
    ("task", "chains", "enabled", "expected"),
    [
        ("ref2va", ("image.reference_preserve",), True, True),
        ("ref2va", ("video_audio.reference_preserve",), True, True),
        ("ref2va", ("audio",), True, True),
        # off by default: the released checkpoint attends bidirectionally
        ("ref2va", ("image.reference_preserve",), False, False),
        # fl2va keyframes are part of the target timeline, never a reference
        ("fl2va", ("image.target_canvas",), True, False),
        ("t2va", (), True, False),
    ],
)
def test_segment_sparse_gate(task, chains, enabled, expected):
    plan = _Plan(task, chains)
    assert _segment_sparse_attn_enabled(plan, _Args(enabled)) is expected


def test_segment_sparse_gate_without_a_plan():
    assert _segment_sparse_attn_enabled(None, _Args(True)) is False


# --------------------------------------------------------------------------
# consumer: bounds validation
# --------------------------------------------------------------------------


def test_bounds_validation_accepts_live_bands():
    got = MiniMaxH3DiTModel._segment_attn_bounds((11, ((3, 7), (8, 11))), used=16)
    assert got == MiniMaxH3SegmentBands(kv_stop=11, ranges=((3, 7), (8, 11)))


def test_bounds_validation_passes_through_opt_out():
    assert MiniMaxH3DiTModel._segment_attn_bounds(None, used=16) is None


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param((11, ()), id="no-ranges"),
        pytest.param((0, ((0, 1),)), id="kv-stop-zero"),
        pytest.param((17, ((3, 11),)), id="kv-stop-past-live-rows"),
        pytest.param((11, ((7, 7),)), id="empty-range"),
        pytest.param((11, ((11, 3),)), id="reversed-range"),
        pytest.param((11, ((3, 12),)), id="range-past-kv-stop"),
        pytest.param((11, ((5, 9), (3, 4))), id="descending-ranges"),
        pytest.param((11, ((3, 8), (6, 10))), id="overlapping-ranges"),
    ],
)
def test_bounds_validation_rejects_broken_bands(raw):
    with pytest.raises(ValueError):
        MiniMaxH3DiTModel._segment_attn_bounds(raw, used=16)


# --------------------------------------------------------------------------
# numerics: three unmasked calls == one masked attention
# --------------------------------------------------------------------------


class _SdpaImpl:
    """Minimal fixed-length impl standing in for a real backend."""

    def __init__(self, softmax_scale: float) -> None:
        self.softmax_scale = softmax_scale

    def forward(self, query, key, value, attn_metadata, *, return_softmax_lse=False):
        assert attn_metadata is None
        assert not return_softmax_lse
        out = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            scale=self.softmax_scale,
        )
        return out.transpose(1, 2)


class _Attention:
    def __init__(self, softmax_scale: float) -> None:
        self.softmax_scale = softmax_scale
        self._attention_impl = _SdpaImpl(softmax_scale)


# --------------------------------------------------------------------------
# opt-out: the dense path must stay bit-for-bit what it was
# --------------------------------------------------------------------------


class _RecordingImpl:
    """Records which entry point the attention core reached, and with what."""

    def __init__(self) -> None:
        self.varlen_calls: list[dict] = []
        self.forward_calls = 0

    def forward(self, query, key, value, attn_metadata, *, return_softmax_lse=False):
        self.forward_calls += 1
        return torch.zeros_like(query)

    def forward_varlen(
        self, query, key, value, *, cu_seqlens, max_seqlen, cu_seqlens_host=None
    ):
        self.varlen_calls.append(
            {
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
                "cu_seqlens_host": cu_seqlens_host,
                "query": query,
            }
        )
        return torch.zeros_like(query)


class _RecordingAttention:
    def __init__(self, backend) -> None:
        self.softmax_scale = 8**-0.5
        self._attention_impl = _RecordingImpl()
        self._attention_backend_enum = backend


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="kwarg-absent"),
        pytest.param({"segment_attn_bounds": None}, id="kwarg-none"),
    ],
)
def test_opt_out_reaches_the_untouched_dense_varlen_call(kwargs):
    """Without the flag the core must call forward_varlen with the same args."""
    attention = _RecordingAttention(AttentionBackendEnum.FA)
    q, k, v = (torch.zeros(64, 2, 8) for _ in range(3))
    cu = torch.tensor([0, 52, 64], dtype=torch.int32)

    _minimax_h3_attention_core_impl(
        attention,
        q,
        k,
        v,
        cu_seqlens=cu,
        cu_seqlens_host=(0, 52, 64),
        max_seqlen=52,
        ulysses_active=False,
        **kwargs,
    )

    impl = attention._attention_impl
    assert impl.forward_calls == 0, "the split path must not run when opted out"
    assert len(impl.varlen_calls) == 1
    call = impl.varlen_calls[0]
    assert call["query"] is q
    assert call["cu_seqlens"] is cu
    assert call["max_seqlen"] == 52
    assert call["cu_seqlens_host"] == (0, 52, 64)


def test_ring_keeps_the_dense_path_even_when_a_band_is_given():
    """Ring's contiguous row chunking cannot be split by query band."""
    attention = _RecordingAttention(AttentionBackendEnum.SAGE_ATTN)
    q, k, v = (torch.zeros(64, 2, 8) for _ in range(3))

    with pytest.raises(NotImplementedError):
        # ring additionally requires FA; the point is that the band never
        # diverts ring to the split path
        _minimax_h3_attention_core_impl(
            attention,
            q,
            k,
            v,
            cu_seqlens=torch.tensor([0, 52, 64], dtype=torch.int32),
            cu_seqlens_host=(0, 52, 64),
            max_seqlen=52,
            ulysses_active=False,
            ring_active=True,
            segment_attn_bounds=MiniMaxH3SegmentBands(30, ((6, 30),)),
        )
    assert attention._attention_impl.forward_calls == 0


def test_unsupported_backend_falls_back_to_dense():
    attention = _RecordingAttention(AttentionBackendEnum.SOL_ATTN)
    q, k, v = (torch.zeros(64, 2, 8) for _ in range(3))

    _minimax_h3_attention_core_impl(
        attention,
        q,
        k,
        v,
        cu_seqlens=torch.tensor([0, 52, 64], dtype=torch.int32),
        cu_seqlens_host=(0, 52, 64),
        max_seqlen=52,
        ulysses_active=False,
        segment_attn_bounds=MiniMaxH3SegmentBands(30, ((6, 30),)),
    )

    assert attention._attention_impl.forward_calls == 0
    assert len(attention._attention_impl.varlen_calls) == 1


def test_supported_backend_takes_the_split_path():
    """Guards the tests above: the split really is reachable."""
    attention = _RecordingAttention(AttentionBackendEnum.FA)
    q, k, v = (torch.zeros(64, 2, 8) for _ in range(3))

    _minimax_h3_attention_core_impl(
        attention,
        q,
        k,
        v,
        cu_seqlens=torch.tensor([0, 52, 64], dtype=torch.int32),
        cu_seqlens_host=(0, 52, 64),
        max_seqlen=52,
        ulysses_active=False,
        segment_attn_bounds=MiniMaxH3SegmentBands(30, ((6, 30),)),
    )

    assert attention._attention_impl.varlen_calls == []
    assert attention._attention_impl.forward_calls == 3


def _dense_masked_reference(q, k, v, *, used, ref_start, ref_stop, scale):
    """One masked attention over the live rows, as the split approximates."""
    mask = torch.ones(used, used, dtype=torch.bool)
    # reference queries may not see the target suffix
    mask[ref_start:ref_stop, ref_stop:used] = False
    out = torch.nn.functional.scaled_dot_product_attention(
        q[:used].transpose(0, 1)[None],
        k[:used].transpose(0, 1)[None],
        v[:used].transpose(0, 1)[None],
        attn_mask=mask[None, None],
        scale=scale,
    )
    return out[0].transpose(0, 1)


def test_split_matches_the_equivalent_masked_attention():
    torch.manual_seed(0)
    seq_len, used, ref_start, ref_stop = 64, 52, 6, 30
    heads, head_dim = 3, 8
    scale = head_dim**-0.5
    q, k, v = (
        torch.randn(seq_len, heads, head_dim, dtype=torch.float64) for _ in range(3)
    )

    out = _minimax_h3_segment_sparse_attention(
        _Attention(scale),
        q,
        k,
        v,
        used=used,
        bands=MiniMaxH3SegmentBands(ref_stop, ((ref_start, ref_stop),)),
    )

    expected = _dense_masked_reference(
        q, k, v, used=used, ref_start=ref_start, ref_stop=ref_stop, scale=scale
    )
    assert out.shape == q.shape
    torch.testing.assert_close(out[:used], expected)


def test_split_zeroes_the_padding_tail():
    """Padding rows never reach the output; keep them inert like sage does."""
    torch.manual_seed(0)
    seq_len, used = 64, 52
    q, k, v = (torch.randn(seq_len, 2, 8, dtype=torch.float64) for _ in range(3))

    out = _minimax_h3_segment_sparse_attention(
        _Attention(8**-0.5),
        q,
        k,
        v,
        used=used,
        bands=MiniMaxH3SegmentBands(30, ((6, 30),)),
    )

    assert torch.count_nonzero(out[used:]) == 0


def test_target_rows_still_attend_to_every_reference_row():
    """Only the reference->target direction is dropped, never the reverse."""
    torch.manual_seed(0)
    seq_len, used, ref_start, ref_stop = 48, 40, 4, 20
    heads, head_dim = 2, 8
    scale = head_dim**-0.5
    q, k, v = (
        torch.randn(seq_len, heads, head_dim, dtype=torch.float64) for _ in range(3)
    )

    baseline = _minimax_h3_segment_sparse_attention(
        _Attention(scale),
        q,
        k,
        v,
        used=used,
        bands=MiniMaxH3SegmentBands(ref_stop, ((ref_start, ref_stop),)),
    )

    # Perturbing a reference row's key/value must still move the target rows.
    k_perturbed, v_perturbed = k.clone(), v.clone()
    k_perturbed[ref_start] += 1.0
    v_perturbed[ref_start] += 1.0
    perturbed = _minimax_h3_segment_sparse_attention(
        _Attention(scale),
        q,
        k_perturbed,
        v_perturbed,
        used=used,
        bands=MiniMaxH3SegmentBands(ref_stop, ((ref_start, ref_stop),)),
    )

    assert not torch.allclose(baseline[ref_stop:used], perturbed[ref_stop:used])
