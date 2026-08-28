# SPDX-License-Identifier: Apache-2.0
"""SpargeAttn block-sparse attention backend.

The schedule and gating tests are pure CPU. The numerical tests need a GPU with
``spas_sage_attn`` built for it and are skipped otherwise.

The trick that makes the sparse kernel checkable against dense attention: at
``topk=1.0`` every key block is inside the budget, so the block-sparse result
must reproduce dense attention up to the int8/fp8 quantization SpargeAttn
shares with SageAttention. That covers the LUT, the 128x64 block sizes and the
softmax scale in one assertion, none of which an accuracy-only comparison at
real sparsity would pin down.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.sparge_attn import (
    DEFAULT_HEAD_CHUNK,
    SpargeAttentionBackend,
    SpargeAttentionImpl,
    SpargeSchedule,
    _dit_layer_index,
    _trailing_padding_used_len,
    sparge_denoise_total_steps,
    sparge_row_modality_tags,
)


def _forget_once_warnings() -> None:
    """``warning_once`` is lru_cached on (logger, message).

    Without this a warning asserted by one test is silently absent in the
    next, which would make these tests order-dependent.
    """
    from sglang.multimodal_gen.runtime.utils import logging_utils

    logging_utils._print_warning_once.cache_clear()

# MiniMax-H3 token tags, from minimax_h3/packed_sequence.py.
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2

HEAD_DIM = 128
NUM_HEADS = 4
_SERVER_ARGS = "sglang.multimodal_gen.runtime.server_args.get_global_server_args"
_FORWARD_CTX = (
    "sglang.multimodal_gen.runtime.layers.attention.backends.sparge_attn"
    ".get_forward_context"
)


def _sparge_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import spas_sage_attn  # noqa: F401
        from sageattention import sageattn  # noqa: F401
    except Exception:
        return False
    return True


requires_sparge = unittest.skipUnless(
    _sparge_available(), "needs a GPU with spas_sage_attn and sageattention built for it"
)


class _FakeServerArgs:
    def __init__(self, config):
        self.attention_backend_config = config


class _Ctx:
    def __init__(self, step, num_inference_steps=None):
        self.current_timestep = step
        self.forward_batch = (
            None
            if num_inference_steps is None
            else type(
                "_Req",
                (),
                {
                    "sampling_params": type(
                        "_SP", (), {"num_inference_steps": num_inference_steps}
                    )()
                },
            )()
        )


def _make_impl(config=None, *, prefix="blocks.5.attn", causal=False, num_kv_heads=None):
    with patch(_SERVER_ARGS, return_value=_FakeServerArgs(config or {})):
        return SpargeAttentionImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            causal=causal,
            softmax_scale=HEAD_DIM**-0.5,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )


def _at_step(step, num_inference_steps=None):
    return patch(_FORWARD_CTX, return_value=_Ctx(step, num_inference_steps))


def _dense_ref(q, k, v):
    qq, kk, vv = (t.transpose(1, 2).float() for t in (q, k, v))
    return F.scaled_dot_product_attention(qq, kk, vv).transpose(1, 2)


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


class TestSpargeSchedule(unittest.TestCase):
    def test_dit_layer_index_only_matches_top_level_blocks(self):
        self.assertEqual(_dit_layer_index("blocks.7.attn"), 7)
        self.assertIsNone(_dit_layer_index("token_refiner.blocks.7.attn"))
        self.assertIsNone(_dit_layer_index("cross_attn"))

    def test_defaults_when_config_is_empty(self):
        with patch(_SERVER_ARGS, return_value=_FakeServerArgs({})):
            schedule = SpargeSchedule.from_server_args()
        self.assertEqual(schedule.topk, 0.5)
        self.assertEqual(schedule.skip_first_steps, 10)
        # The tail cutoff is unmeasured on this model, so it ships off.
        self.assertEqual(schedule.skip_last_steps, 0)
        self.assertEqual(schedule.skip_first_layers, 0)
        self.assertEqual(schedule.min_seq_len, 4096)

    def test_topk_one_stays_legal(self):
        """It keeps every block, which is what the numerical tests calibrate on."""
        with patch(_SERVER_ARGS, return_value=_FakeServerArgs({"topk": 1.0})):
            self.assertEqual(SpargeSchedule.from_server_args().topk, 1.0)

    def test_rejects_out_of_range_values(self):
        for config in (
            {"topk": 0.0},
            {"topk": 1.5},
            {"skip_first_steps": -1},
            {"skip_last_steps": -1},
            {"skip_first_layers": -1},
            # The kernel itself asserts seq_len >= 128.
            {"min_seq_len": 64},
        ):
            with self.subTest(config=config):
                with patch(_SERVER_ARGS, return_value=_FakeServerArgs(config)):
                    with self.assertRaises(ValueError):
                        SpargeSchedule.from_server_args()


class TestTrailingPadding(unittest.TestCase):
    """H3 packs one live document as ``(0, used, total)``."""

    def test_recognises_the_h3_layout(self):
        self.assertEqual(
            _trailing_padding_used_len(
                total_tokens=16448, max_seqlen=16384, bounds=(0, 16384, 16448)
            ),
            16384,
        )

    def test_rejects_anything_else(self):
        for bounds, total, max_seqlen in (
            ((0, 8192, 16384, 16448), 16448, 8192),  # multi-document
            ((0, 16448, 16448), 16448, 16448),  # no padding tail
            ((64, 16384, 16448), 16448, 16320),  # does not start at zero
        ):
            with self.subTest(bounds=bounds):
                self.assertIsNone(
                    _trailing_padding_used_len(
                        total_tokens=total, max_seqlen=max_seqlen, bounds=bounds
                    )
                )


class TestSpargeBackend(unittest.TestCase):
    def test_the_advertised_builder_can_be_built(self):
        builder = SpargeAttentionBackend.get_builder_cls()()
        builder.prepare()
        metadata = builder.build(current_timestep=3)
        self.assertEqual(metadata.current_timestep, 3)

    def test_declares_packed_varlen_but_not_ring(self):
        self.assertTrue(SpargeAttentionBackend.supports_packed_varlen())
        # The recommended API returns no softmax LSE, so the per-hop online
        # merge that ring attention needs cannot be done.
        self.assertFalse(SpargeAttentionBackend.supports_ring_rotation())

    def test_head_sizes_match_the_kernel_assertion(self):
        self.assertEqual(SpargeAttentionBackend.get_supported_head_sizes(), [64, 128])


@requires_sparge
class TestSpargeGating(unittest.TestCase):
    def test_dit_layers_are_eligible(self):
        self.assertTrue(_make_impl().layer_enabled)

    def test_token_refiner_is_dense(self):
        self.assertFalse(_make_impl(prefix="token_refiner.blocks.5.attn").layer_enabled)

    def test_skip_first_layers_gates_the_bottom_of_the_stack(self):
        for prefix, expected in (("blocks.0.attn", False), ("blocks.4.attn", True)):
            with self.subTest(prefix=prefix):
                impl = _make_impl({"skip_first_layers": 4}, prefix=prefix)
                self.assertEqual(impl.layer_enabled, expected)

    def test_causal_never_runs_sparse(self):
        """The sage2 path drops the causal flag; see the module docstring."""
        self.assertFalse(_make_impl(causal=True).layer_enabled)

    def test_gqa_never_runs_sparse(self):
        """The block map pools Q and K into one matmul, so heads must match."""
        self.assertFalse(_make_impl(num_kv_heads=NUM_HEADS // 2).layer_enabled)


@requires_sparge
class TestShortScheduleWarning(unittest.TestCase):
    """A turbo checkpoint runs 9 or 5 steps; the default cutoff is 10."""

    def setUp(self):
        _forget_once_warnings()

    def test_warns_when_every_step_is_below_the_cutoff(self):
        impl = _make_impl({"skip_first_steps": 10})
        with _at_step(3, num_inference_steps=9):
            with self.assertLogs(level="WARNING") as logs:
                self.assertFalse(impl._step_enabled())
        self.assertIn("never activates", "".join(logs.output))

    def test_warns_when_the_two_cutoffs_together_cover_the_schedule(self):
        """Neither cutoff is too big alone; together they leave nothing."""
        impl = _make_impl({"skip_first_steps": 5, "skip_last_steps": 5})
        with sparge_denoise_total_steps(9):
            with _at_step(6):
                with self.assertLogs(level="WARNING") as logs:
                    self.assertFalse(impl._step_enabled())
        self.assertIn("never activates", "".join(logs.output))

    def test_silent_when_the_schedule_is_long_enough(self):
        impl = _make_impl({"skip_first_steps": 2})
        with _at_step(3, num_inference_steps=9):
            with patch.object(impl, "_warn_if_the_cutoffs_swallow_the_schedule") as w:
                self.assertTrue(impl._step_enabled())
                w.assert_called_once()
        # and the real check stays quiet for this combination
        impl2 = _make_impl({"skip_first_steps": 2})
        with _at_step(3, num_inference_steps=9):
            self.assertTrue(impl2._step_enabled())

    def test_missing_step_count_is_not_an_error(self):
        impl = _make_impl({"skip_first_steps": 10})
        with _at_step(3):  # forward_batch is None, as in text encoding
            self.assertFalse(impl._step_enabled())


@requires_sparge
class TestTailCutoff(unittest.TestCase):
    """``skip_last_steps`` routes the final forwards back through sage_attn."""

    def setUp(self):
        _forget_once_warnings()

    def test_last_steps_are_excluded(self):
        impl = _make_impl({"skip_first_steps": 0, "skip_last_steps": 2})
        with sparge_denoise_total_steps(10):
            for step, expected in ((0, True), (7, True), (8, False), (9, False)):
                with self.subTest(step=step):
                    with _at_step(step):
                        self.assertEqual(impl._step_enabled(), expected)

    def test_both_cutoffs_apply_together(self):
        impl = _make_impl({"skip_first_steps": 2, "skip_last_steps": 2})
        with sparge_denoise_total_steps(10):
            for step, expected in ((1, False), (2, True), (7, True), (8, False)):
                with self.subTest(step=step):
                    with _at_step(step):
                        self.assertEqual(impl._step_enabled(), expected)

    def test_published_length_beats_the_sampling_params_hint(self):
        """H3 runs len(sigmas_video)-1, which need not equal the hint."""
        impl = _make_impl({"skip_first_steps": 0, "skip_last_steps": 1})
        with sparge_denoise_total_steps(10):
            with _at_step(9, num_inference_steps=50):
                self.assertFalse(impl._step_enabled())

    def test_a_tuple_valued_hint_is_not_a_schedule_length(self):
        """MiniMax-H3 types num_inference_steps as ``int | tuple[int, int]``."""
        impl = _make_impl({"skip_first_steps": 0, "skip_last_steps": 1})
        with _at_step(9, num_inference_steps=(10, 10)):
            with self.assertLogs(level="WARNING") as logs:
                self.assertTrue(impl._step_enabled())
        self.assertIn("no denoise schedule length", "".join(logs.output))

    def test_missing_length_fails_open_and_says_so(self):
        """Losing the tail beats disabling the backend over a missing int."""
        impl = _make_impl({"skip_first_steps": 0, "skip_last_steps": 2})
        with _at_step(5):
            with self.assertLogs(level="WARNING") as logs:
                self.assertTrue(impl._step_enabled())
        self.assertIn("skip_last_steps=2 is inactive", "".join(logs.output))

    def test_the_default_schedule_never_asks_for_a_length(self):
        impl = _make_impl({"skip_first_steps": 0})
        with _at_step(5):
            with patch.object(impl, "_warn_tail_cutoff_has_no_schedule_length") as w:
                self.assertTrue(impl._step_enabled())
                w.assert_not_called()


@requires_sparge
class TestSpargeNumerics(unittest.TestCase):
    SEQ_LEN = 8192

    def setUp(self):
        torch.manual_seed(0)
        self.qkv = tuple(
            torch.randn(
                1,
                self.SEQ_LEN,
                NUM_HEADS,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device="cuda",
            )
            for _ in range(3)
        )

    def test_topk_one_reproduces_dense(self):
        impl = _make_impl({"topk": 1.0, "skip_first_steps": 0, "dense_modalities": []})
        with _at_step(0):
            out = impl.forward(*self.qkv, None)
        dense = impl.dense_impl.forward(*self.qkv, None)
        # Against the fp32 reference both carry the same quantization error;
        # against each other they must agree far more tightly than that.
        self.assertGreater(_cos(out, _dense_ref(*self.qkv)), 0.99)
        self.assertGreater(_cos(out, dense), 0.999)

    def test_skipped_steps_take_the_dense_path_exactly(self):
        impl = _make_impl({"topk": 0.3, "skip_first_steps": 10, "dense_modalities": []})
        dense = impl.dense_impl.forward(*self.qkv, None)
        with _at_step(3):
            self.assertTrue(torch.equal(impl.forward(*self.qkv, None), dense))
        with _at_step(20):
            self.assertFalse(torch.equal(impl.forward(*self.qkv, None), dense))

    def test_last_steps_take_the_dense_path_exactly(self):
        """The tail cutoff must land on sage_attn, bit for bit."""
        impl = _make_impl(
            {
                "topk": 0.3,
                "skip_first_steps": 0,
                "skip_last_steps": 2,
                "dense_modalities": [],
            }
        )
        dense = impl.dense_impl.forward(*self.qkv, None)
        with sparge_denoise_total_steps(10):
            with _at_step(7):
                self.assertFalse(torch.equal(impl.forward(*self.qkv, None), dense))
            with _at_step(8):
                self.assertTrue(torch.equal(impl.forward(*self.qkv, None), dense))

    def test_short_sequences_take_the_dense_path_exactly(self):
        short = tuple(t[:, :2048] for t in self.qkv)
        impl = _make_impl({"topk": 0.3, "skip_first_steps": 0, "min_seq_len": 4096})
        with _at_step(20):
            self.assertTrue(
                torch.equal(impl.forward(*short, None), impl.dense_impl.forward(*short, None))
            )

    def test_varlen_keeps_the_padding_tail_zero(self):
        used, total = self.SEQ_LEN, self.SEQ_LEN + 64
        packed = tuple(
            torch.randn(total, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            for _ in range(3)
        )
        impl = _make_impl({"topk": 1.0, "skip_first_steps": 0, "min_seq_len": 4096, "dense_modalities": []})
        with _at_step(0):
            out = impl.forward_varlen(
                *packed,
                cu_seqlens=torch.tensor([0, used, total], dtype=torch.int32, device="cuda"),
                max_seqlen=used,
                cu_seqlens_host=(0, used, total),
            )
        self.assertEqual(out.shape, packed[0].shape)
        self.assertTrue(bool((out[used:] == 0).all()))
        live_ref = _dense_ref(*(t[:used].unsqueeze(0) for t in packed))[0]
        self.assertGreater(_cos(out[:used], live_ref), 0.99)

    def test_varlen_routes_each_document_separately(self):
        bounds = (0, 4096, self.SEQ_LEN, self.SEQ_LEN + 64)
        packed = tuple(
            torch.randn(bounds[-1], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            for _ in range(3)
        )
        impl = _make_impl({"topk": 1.0, "skip_first_steps": 0, "min_seq_len": 4096, "dense_modalities": []})
        with _at_step(0):
            out = impl.forward_varlen(
                *packed,
                cu_seqlens=torch.tensor(bounds, dtype=torch.int32, device="cuda"),
                max_seqlen=4096,
                cu_seqlens_host=bounds,
            )
        # Every document, sparse or dense, reproduces dense attention over its
        # own rows -- no cross-document leakage through the packed buffer.
        for start, stop in zip(bounds[:-1], bounds[1:]):
            with self.subTest(document=(start, stop)):
                seg_ref = _dense_ref(*(t[start:stop].unsqueeze(0) for t in packed))[0]
                self.assertGreater(_cos(out[start:stop], seg_ref), 0.99)


@requires_sparge
class TestAudioIsNeverSparsified(unittest.TestCase):
    """Audio rows lose every block-sparse budget contest unless protected.

    Audio is a small minority of the packed rows, so a top-k budget picked
    from pooled Q.K scores spends audio rows' blocks on video keys and drops
    audio key blocks from nearly every row. Rendered output showed it as
    corrupted audio in the opening seconds.
    """

    SEQ_LEN = 8192
    N_TEXT = 256
    N_AUDIO = 1024

    def setUp(self):
        torch.manual_seed(0)
        self.qkv = tuple(
            torch.randn(
                1, self.SEQ_LEN, NUM_HEADS, HEAD_DIM,
                dtype=torch.bfloat16, device="cuda",
            )
            for _ in range(3)
        )
        # H3-like layout: text, then video, then audio.
        tags = torch.full((self.SEQ_LEN,), VIDEO_TAG, dtype=torch.long, device="cuda")
        tags[: self.N_TEXT] = TEXT_TAG
        tags[self.SEQ_LEN - self.N_AUDIO :] = AUDIO_TAG
        self.tags = tags
        self.audio = slice(self.SEQ_LEN - self.N_AUDIO, self.SEQ_LEN)
        self.video = slice(self.N_TEXT, self.SEQ_LEN - self.N_AUDIO)

    def _full_attention_through_the_same_kernel(self):
        """Ground truth with nothing dropped, quantized identically.

        Comparing against this instead of against the dense fallback isolates
        sparsity from the int8/fp8 rounding the two paths do differently.
        """
        from spas_sage_attn import block_sparse_sage2_attn_cuda

        q = self.qkv[0]
        mask = torch.ones(
            1, NUM_HEADS, (self.SEQ_LEN + 127) // 128, (self.SEQ_LEN + 63) // 64,
            dtype=torch.bool, device="cuda",
        )
        return block_sparse_sage2_attn_cuda(
            *self.qkv, mask_id=mask, scale=HEAD_DIM**-0.5, tensor_layout="NHD"
        )

    def test_protected_audio_rows_are_bit_identical_to_full_attention(self):
        impl = _make_impl({"topk": 0.3, "skip_first_steps": 0})
        with sparge_row_modality_tags(self.tags):
            with _at_step(20):
                out = impl.forward(*self.qkv, None)
        full = self._full_attention_through_the_same_kernel()
        self.assertTrue(torch.equal(out[0, self.audio], full[0, self.audio]))
        # ...and the video rows really were sparsified, or the test proves nothing.
        self.assertFalse(torch.equal(out[0, self.video], full[0, self.video]))

    def test_without_protection_audio_degrades(self):
        """The regression this guards against, reproduced deliberately."""
        impl = _make_impl(
            {"topk": 0.3, "skip_first_steps": 0, "dense_modalities": []}
        )
        with _at_step(20):
            out = impl.forward(*self.qkv, None)
        full = self._full_attention_through_the_same_kernel()
        self.assertLess(_cos(out[0, self.audio], full[0, self.audio]), 0.95)

    def test_missing_tags_fall_back_to_dense(self):
        """Protection asked for but unlocatable must not sparsify anyway."""
        impl = _make_impl({"topk": 0.3, "skip_first_steps": 0})
        with _at_step(20):
            out = impl.forward(*self.qkv, None)
        self.assertTrue(torch.equal(out, impl.dense_impl.forward(*self.qkv, None)))

    def test_tags_that_do_not_cover_the_rows_fall_back_to_dense(self):
        """A Ulysses all-to-all leaves attention in a different row space."""
        impl = _make_impl({"topk": 0.3, "skip_first_steps": 0})
        with sparge_row_modality_tags(self.tags[: self.SEQ_LEN // 2]):
            with _at_step(20):
                out = impl.forward(*self.qkv, None)
        self.assertTrue(torch.equal(out, impl.dense_impl.forward(*self.qkv, None)))

    def test_varlen_protects_audio_in_the_live_document(self):
        used, total = self.SEQ_LEN, self.SEQ_LEN + 64
        packed = tuple(t[0] for t in self.qkv)
        impl = _make_impl({"topk": 0.3, "skip_first_steps": 0, "min_seq_len": 4096})
        with sparge_row_modality_tags(self.tags):
            with _at_step(20):
                out = impl.forward_varlen(
                    *(
                        torch.cat([t, torch.zeros(64, NUM_HEADS, HEAD_DIM,
                                                  dtype=t.dtype, device=t.device)])
                        for t in packed
                    ),
                    cu_seqlens=torch.tensor([0, used, total],
                                            dtype=torch.int32, device="cuda"),
                    max_seqlen=used,
                    cu_seqlens_host=(0, used, total),
                )
        full = self._full_attention_through_the_same_kernel()
        self.assertTrue(torch.equal(out[self.audio], full[0, self.audio]))
        self.assertTrue(bool((out[used:] == 0).all()))


class TestSelectionRule(unittest.TestCase):
    """topk (fixed budget) and cdfthreshd (adaptive top-p) are exclusive."""

    def test_topk_is_the_default(self):
        with patch(_SERVER_ARGS, return_value=_FakeServerArgs({})):
            schedule = SpargeSchedule.from_server_args()
        self.assertEqual(schedule.topk, 0.5)
        self.assertIsNone(schedule.cdfthreshd)

    def test_naming_cdfthreshd_alone_switches_rules(self):
        with patch(_SERVER_ARGS, return_value=_FakeServerArgs({"cdfthreshd": 0.98})):
            schedule = SpargeSchedule.from_server_args()
        self.assertIsNone(schedule.topk)
        self.assertEqual(schedule.cdfthreshd, 0.98)

    def test_rejects_both_and_neither(self):
        for config in (
            {"cdfthreshd": 0.98, "topk": 0.5},
            {"topk": None},
            {"cdfthreshd": 1.5},
        ):
            with self.subTest(config=config):
                with patch(_SERVER_ARGS, return_value=_FakeServerArgs(config)):
                    with self.assertRaises(ValueError):
                        SpargeSchedule.from_server_args()


@requires_sparge
class TestCdfThreshdPath(unittest.TestCase):
    """cdfthreshd cannot use the plug-and-play API, so it takes the mask path."""

    def setUp(self):
        torch.manual_seed(0)
        self.qkv = tuple(
            torch.randn(1, 8192, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            for _ in range(3)
        )

    def test_runs_without_protection(self):
        """Regression: the mask patch used to dereference a None mask here."""
        impl = _make_impl(
            {"cdfthreshd": 0.9, "skip_first_steps": 0, "dense_modalities": []}
        )
        with _at_step(20):
            out = impl.forward(*self.qkv, None)
        self.assertEqual(out.shape, self.qkv[0].shape)
        self.assertTrue(torch.isfinite(out.float()).all())

    def test_runs_with_protection(self):
        tags = torch.full((8192,), VIDEO_TAG, dtype=torch.long, device="cuda")
        tags[:256] = TEXT_TAG
        tags[-1024:] = AUDIO_TAG
        impl = _make_impl({"cdfthreshd": 0.9, "skip_first_steps": 0})
        with sparge_row_modality_tags(tags):
            with _at_step(20):
                out = impl.forward(*self.qkv, None)
        self.assertEqual(out.shape, self.qkv[0].shape)
        self.assertTrue(torch.isfinite(out.float()).all())


class TestDenseModalitiesConfig(unittest.TestCase):
    def test_text_and_audio_are_protected_by_default(self):
        """Both are tiny minorities of the packed rows and starve under top-k."""
        with patch(_SERVER_ARGS, return_value=_FakeServerArgs({})):
            self.assertEqual(
                SpargeSchedule.from_server_args().dense_modalities,
                (TEXT_TAG, AUDIO_TAG),
            )

    def test_can_be_disabled_or_extended(self):
        for config, expected in (
            ({"dense_modalities": []}, ()),
            ({"dense_modalities": [1, 2]}, (1, 2)),
        ):
            with self.subTest(config=config):
                with patch(_SERVER_ARGS, return_value=_FakeServerArgs(config)):
                    self.assertEqual(
                        SpargeSchedule.from_server_args().dense_modalities, expected
                    )


if __name__ == "__main__":
    unittest.main()


class TestHeadChunking(unittest.TestCase):
    """Running the sparse path a head slice at a time must change nothing.

    Both halves of the path are head-parallel: attention itself, and every
    step of SpargeAttn's map helper (pooling, the similarity gates, the
    pooled-score softmax, the sort and `fill_block_map_triton`). If a future
    SpargeAttn ever mixes heads -- a shared budget, a cross-head normalization
    -- this test is what catches it, because the sliced result would stop
    matching the whole-head one.
    """

    def test_chunk_size_is_validated(self):
        with patch(_SERVER_ARGS, return_value=_FakeServerArgs({})):
            self.assertEqual(
                SpargeSchedule.from_server_args().head_chunk, DEFAULT_HEAD_CHUNK
            )
        with patch(_SERVER_ARGS, return_value=_FakeServerArgs({"head_chunk": 0})):
            self.assertEqual(SpargeSchedule.from_server_args().head_chunk, 0)
        with patch(_SERVER_ARGS, return_value=_FakeServerArgs({"head_chunk": -1})):
            with self.assertRaisesRegex(ValueError, "head_chunk"):
                SpargeSchedule.from_server_args()

    @staticmethod
    def _qkv(heads, seq):
        torch.manual_seed(0)
        shape = (1, seq, heads, HEAD_DIM)
        return tuple(
            torch.randn(*shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        )

    @requires_sparge
    def test_block_map_is_bit_identical_across_chunk_sizes(self):
        heads, seq = 8, 4096
        q, k, _ = self._qkv(heads, seq)

        def whole_map():
            impl = _make_impl({"topk": 0.5, "head_chunk": 0})
            return impl._block_map(q, k, blk_q=128, blk_k=64)

        def sliced_maps(chunk):
            impl = _make_impl({"topk": 0.5, "head_chunk": chunk})
            return torch.cat(
                [
                    impl._block_map(q_s, k_s, blk_q=128, blk_k=64)
                    for q_s, k_s in impl._head_slices(q, k)
                ],
                dim=1,
            )

        whole = whole_map()
        self.assertEqual(whole.shape[1], heads)
        # Every divisor plus a size that does not divide the head count, so the
        # ragged last slice is covered too.
        for chunk in (1, 2, 3, 4, 8, 16):
            with self.subTest(chunk=chunk):
                sliced = sliced_maps(chunk)
                self.assertEqual(sliced.shape, whole.shape)
                self.assertTrue(
                    torch.equal(whole, sliced),
                    f"chunk={chunk} changed {int((whole ^ sliced).sum())} blocks",
                )

    @requires_sparge
    def test_sparse_attention_is_bit_identical_across_chunk_sizes(self):
        heads, seq = 8, 4096
        q, k, v = self._qkv(heads, seq)
        # Protection on, which is the path any H3 run with default
        # dense_modalities takes.
        protected = torch.zeros(seq, dtype=torch.bool, device="cuda")
        protected[:256] = True

        def attend(chunk):
            impl = _make_impl({"topk": 0.5, "head_chunk": chunk})
            return impl._sparse_attention(q, k, v, protected)

        whole = attend(0)
        for chunk in (1, 3, 4, 8, 16):
            with self.subTest(chunk=chunk):
                sliced = attend(chunk)
                self.assertEqual(sliced.shape, whole.shape)
                self.assertTrue(
                    torch.equal(whole, sliced),
                    f"chunk={chunk} max |diff| "
                    f"{(whole.float() - sliced.float()).abs().max().item():.3e}",
                )

    @requires_sparge
    def test_chunking_lowers_the_transient_peak(self):
        """The point of the split: the peak scales with the slice, not H."""
        heads, seq = 16, 8192
        q, k, v = self._qkv(heads, seq)
        protected = torch.zeros(seq, dtype=torch.bool, device="cuda")
        protected[:256] = True

        def peak_at(chunk):
            impl = _make_impl({"topk": 0.5, "head_chunk": chunk})
            impl._sparse_attention(q, k, v, protected)  # warm the Triton cache
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            impl._sparse_attention(q, k, v, protected)
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() - before

        whole = peak_at(0)
        sliced = peak_at(2)
        self.assertLess(
            sliced,
            whole / 2,
            "expected the split to more than halve the peak, got "
            f"{sliced / 2**20:.0f} MiB against {whole / 2**20:.0f} MiB",
        )
