# SPDX-License-Identifier: Apache-2.0
"""SageAttention backend: head-sliced dense attention.

The configuration tests are pure CPU. The numerical tests need a GPU with
``sageattention`` built for it and are skipped otherwise.

What makes slicing checkable: attention is head-parallel and so is every
statistic ``sageattn`` quantizes with -- Q and K scales are per (head, block),
V's scale and mean per (head, channel), and K's smoothing mean reduces along
the sequence. None of them mixes heads, so a sliced run must reproduce the
whole-width one *bit for bit*, not merely to a tolerance. If a future
SageAttention ever introduces a cross-head statistic, that equality is what
catches it.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn import (
    DEFAULT_HEAD_CHUNK,
    SageAttentionImpl,
    _head_chunk_from_server_args,
    _trailing_padding_used_len,
)

HEAD_DIM = 128
NUM_HEADS = 8
_SERVER_ARGS = "sglang.multimodal_gen.runtime.server_args.get_global_server_args"


def _sage_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from sageattention import sageattn  # noqa: F401
    except Exception:
        return False
    return True


requires_sage = unittest.skipUnless(
    _sage_available(), "needs a GPU with sageattention built for it"
)


class _FakeServerArgs:
    def __init__(self, config):
        self.attention_backend_config = config


def _make_impl(config=None, *, head_chunk=None, causal=False):
    kwargs = {} if head_chunk is None else {"head_chunk": head_chunk}
    with patch(_SERVER_ARGS, return_value=_FakeServerArgs(config or {})):
        return SageAttentionImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            causal=causal,
            softmax_scale=HEAD_DIM**-0.5,
            num_kv_heads=NUM_HEADS,
            **kwargs,
        )


class TestSageHeadChunkConfig(unittest.TestCase):
    """``head_chunk`` reaches the impl and rejects nonsense."""

    def test_default_runs_every_head_in_one_call(self):
        self.assertEqual(DEFAULT_HEAD_CHUNK, 0)
        with patch(_SERVER_ARGS, return_value=_FakeServerArgs({})):
            self.assertEqual(_head_chunk_from_server_args(), DEFAULT_HEAD_CHUNK)
        self.assertFalse(_make_impl()._slices_heads(NUM_HEADS))

    def test_config_is_read_from_server_args(self):
        # Shared with sparge_attn, which builds this impl as its dense fallback.
        self.assertEqual(_make_impl({"head_chunk": 4}).head_chunk, 4)
        with patch(_SERVER_ARGS, return_value=_FakeServerArgs({"head_chunk": -1})):
            with self.assertRaisesRegex(ValueError, "head_chunk"):
                _head_chunk_from_server_args()

    def test_missing_server_args_falls_back_to_whole_width(self):
        # This backend is also constructed directly, with no server running.
        # `get_global_server_args` raises there, and since slicing is opt-in
        # that has to mean the whole-width path, not a failure to construct.
        unset = patch(
            _SERVER_ARGS, side_effect=ValueError("Global sgl_diffusion args is not set.")
        )
        with unset:
            self.assertEqual(_head_chunk_from_server_args(), DEFAULT_HEAD_CHUNK)
            impl = SageAttentionImpl(
                num_heads=NUM_HEADS,
                head_size=HEAD_DIM,
                causal=False,
                softmax_scale=HEAD_DIM**-0.5,
                num_kv_heads=NUM_HEADS,
            )
        self.assertEqual(impl.head_chunk, 0)

    def test_explicit_argument_wins_over_server_args(self):
        impl = _make_impl({"head_chunk": 4}, head_chunk=2)
        self.assertEqual(impl.head_chunk, 2)

    def test_chunk_covering_every_head_is_not_a_slice(self):
        # A chunk at or above the head count would allocate the destination
        # buffer for nothing, so it must take the whole-width path.
        impl = _make_impl(head_chunk=NUM_HEADS)
        self.assertFalse(impl._slices_heads(NUM_HEADS))
        self.assertTrue(_make_impl(head_chunk=NUM_HEADS - 1)._slices_heads(NUM_HEADS))


class TestTrailingPaddingDetection(unittest.TestCase):
    """Only the H3 ``[0, used, total]`` shape takes the padded-output path."""

    def test_h3_layout_is_recognized(self):
        self.assertEqual(
            _trailing_padding_used_len(
                total_tokens=256, max_seqlen=192, bounds=(0, 192, 256)
            ),
            192,
        )

    def test_other_layouts_are_declined(self):
        for bounds in ((0, 256, 256), (0, 128, 192, 256), (64, 192, 256)):
            self.assertIsNone(
                _trailing_padding_used_len(
                    total_tokens=256, max_seqlen=192, bounds=bounds
                )
            )


@requires_sage
class TestSageHeadSlicing(unittest.TestCase):
    """Slicing changes the allocation pattern, never the numbers."""

    SEQ = 4096
    USED = 4032  # 64-aligned padding tail, as MiniMax-H3 packs it

    @classmethod
    def _qkv(cls, seq):
        torch.manual_seed(0)
        shape = (seq, NUM_HEADS, HEAD_DIM)
        return tuple(
            torch.randn(*shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        )

    def _varlen(self, impl, q, k, v, *, bounds, max_seqlen):
        return impl.forward_varlen(
            q,
            k,
            v,
            cu_seqlens=torch.tensor(bounds, dtype=torch.int32, device="cuda"),
            max_seqlen=max_seqlen,
            cu_seqlens_host=bounds,
        )

    def test_padded_document_is_bit_identical_across_chunk_sizes(self):
        q, k, v = self._qkv(self.SEQ)
        bounds = (0, self.USED, self.SEQ)
        whole = self._varlen(
            _make_impl(head_chunk=0), q, k, v, bounds=bounds, max_seqlen=self.USED
        )
        # Divisors of the head count plus a width that does not divide it, so
        # the ragged last slice is covered too.
        for chunk in (1, 2, 3, 4, NUM_HEADS - 1):
            with self.subTest(chunk=chunk):
                sliced = self._varlen(
                    _make_impl(head_chunk=chunk),
                    q,
                    k,
                    v,
                    bounds=bounds,
                    max_seqlen=self.USED,
                )
                self.assertTrue(torch.equal(sliced, whole))

    def test_padding_tail_stays_zero(self):
        q, k, v = self._qkv(self.SEQ)
        out = self._varlen(
            _make_impl(head_chunk=4),
            q,
            k,
            v,
            bounds=(0, self.USED, self.SEQ),
            max_seqlen=self.USED,
        )
        # Downstream masks rely on the tail being inactive, not merely small.
        self.assertTrue(torch.count_nonzero(out[self.USED :]) == 0)
        self.assertTrue(torch.count_nonzero(out[: self.USED]) > 0)

    def test_multi_document_packing_is_bit_identical(self):
        # The other branch of _sage_packed: several live documents, no padding.
        q, k, v = self._qkv(self.SEQ)
        bounds = (0, 1024, 3072, self.SEQ)
        whole = self._varlen(
            _make_impl(head_chunk=0), q, k, v, bounds=bounds, max_seqlen=2048
        )
        sliced = self._varlen(
            _make_impl(head_chunk=3), q, k, v, bounds=bounds, max_seqlen=2048
        )
        self.assertTrue(torch.equal(sliced, whole))

    def test_slicing_lowers_peak_allocation(self):
        # The point of the change: the transients scale with the slice width.
        q, k, v = self._qkv(self.SEQ)
        bounds = (0, self.USED, self.SEQ)

        def peak(chunk):
            impl = _make_impl(head_chunk=chunk)
            self._varlen(impl, q, k, v, bounds=bounds, max_seqlen=self.USED)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            self._varlen(impl, q, k, v, bounds=bounds, max_seqlen=self.USED)
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() - base

        self.assertLess(peak(2), peak(0))


if __name__ == "__main__":
    unittest.main()
