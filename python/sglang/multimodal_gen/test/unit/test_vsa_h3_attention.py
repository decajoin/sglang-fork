# SPDX-License-Identifier: Apache-2.0
"""VSA-H3 video sparse attention backend.

The schedule, geometry and gating tests are pure CPU -- the tiling is index
arithmetic and runs anywhere. The numerical tests need a CUDA GPU for the
Triton block-sparse kernel and are skipped otherwise.

Two tricks make the sparse kernel checkable:

- At ``sparsity=0`` every video tile is inside the budget and every prefix tile
  is exempt, so the block-sparse result must reproduce *dense* attention over
  the live rows. That pins the tiling, the pad masking inside partial tiles,
  the scatter/gather round trip and the softmax scale in one assertion, none of
  which an accuracy-only comparison at real sparsity would catch.
- At real sparsity the backend is compared against an independent PyTorch
  implementation of the same selection rule, built from a materialised mask
  rather than an index list, so the two do not share the code under test.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn_h3 import (
    DEFAULT_HEAD_CHUNK_BUDGET_MIB,
    DEFAULT_SKIP_FIRST_STEPS,
    DEFAULT_SPARSITY,
    VideoSparseAttentionH3Backend,
    VideoSparseAttentionH3Impl,
    VsaH3Schedule,
    VsaH3SequenceGeometry,
    _dit_layer_index,
    _tile_geometry,
    compute_topk,
    vsa_h3_sequence_geometry,
)

HEAD_DIM = 128
NUM_HEADS = 4
BLOCK = 64
# text, keyframe condition, audio | an 8 x 24 x 42 video patch grid
PREFIX = (512, 300, 1000)
VIDEO_GRID = (8, 24, 42)

_SERVER_ARGS = "sglang.multimodal_gen.runtime.server_args.get_global_server_args"
_FORWARD_CTX = (
    "sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn_h3"
    ".get_forward_context"
)

requires_gpu = unittest.skipUnless(
    torch.cuda.is_available(), "needs a CUDA GPU for the Triton block-sparse kernel"
)


class _FakeServerArgs:
    def __init__(self, config):
        self.attention_backend_config = config


class _Ctx:
    def __init__(self, step):
        self.current_timestep = step
        self.forward_batch = None


def _make_impl(config=None, *, prefix="blocks.5.attn", causal=False, num_kv_heads=None):
    with patch(_SERVER_ARGS, return_value=_FakeServerArgs(config or {})):
        return VideoSparseAttentionH3Impl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            causal=causal,
            softmax_scale=HEAD_DIM**-0.5,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )


def _at_step(step):
    return patch(_FORWARD_CTX, return_value=_Ctx(step))


def _geometry() -> VsaH3SequenceGeometry:
    return VsaH3SequenceGeometry(prefix_segments=PREFIX, video_grid=VIDEO_GRID)


def _rows(geometry, device, heads=NUM_HEADS, dim=HEAD_DIM):
    return tuple(
        torch.randn(geometry.live_rows, heads, dim, device=device, dtype=torch.bfloat16)
        for _ in range(3)
    )


def _dense_ref(q, k, v):
    """Dense attention over ``[S, H, D]`` rows."""
    qq, kk, vv = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    return F.scaled_dot_product_attention(qq, kk, vv)[0].transpose(0, 1)


def _masked_ref(q, k, v, geometry, sparsity, prefix_mode="exempt"):
    """The same selection rule, computed from a materialised mask.

    Deliberately structured the other way round from the backend: it tiles with
    ``index_copy_``, scores every tile pair, expands the tile mask to a token
    mask and hands that to SDPA, so nothing but the geometry helper is shared
    with the code under test.
    """
    tiles = _tile_geometry(geometry.prefix_segments, geometry.video_grid, q.device)
    total, prefix, video = (
        tiles.num_tiles,
        tiles.num_prefix_tiles,
        tiles.num_video_tiles,
    )
    heads, dim = q.shape[1], q.shape[2]

    def tile(x):
        buffer = torch.zeros(
            (heads, tiles.padded_rows, dim), dtype=x.dtype, device=x.device
        )
        return buffer.index_copy_(1, tiles.scatter_index, x.transpose(0, 1))

    q_t, k_t, v_t = tile(q), tile(k), tile(v)
    sizes = tiles.variable_block_sizes.float().view(1, -1, 1)

    def pool(x):
        return x.view(heads, total, BLOCK, dim).sum(2, dtype=torch.float32) / sizes

    scores = torch.matmul(pool(q_t), pool(k_t).transpose(-2, -1)) / math.sqrt(dim)
    topk = compute_topk(sparsity, video)
    mask = torch.zeros(heads, total, total, dtype=torch.bool, device=q.device)
    mask[:, :prefix, :] = True  # prefix queries are dense
    if prefix_mode == "exempt":
        selected = scores[:, prefix:, prefix:].topk(topk, dim=-1).indices + prefix
        mask[:, prefix:].scatter_(-1, selected, True)
        mask[:, prefix:, :prefix] = True
    else:
        budget = min(topk + prefix, total)
        selected = scores[:, prefix:, :].topk(budget, dim=-1).indices
        mask[:, prefix:].scatter_(-1, selected, True)

    live = (
        torch.arange(BLOCK, device=q.device)[None, :]
        < tiles.variable_block_sizes[:, None]
    ).reshape(-1)
    token_mask = (
        mask.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)
        & live[None, None, :]
    )
    bias = torch.where(token_mask, 0.0, float("-inf")).to(q.dtype)
    out = F.scaled_dot_product_attention(
        q_t.unsqueeze(0),
        k_t.unsqueeze(0),
        v_t.unsqueeze(0),
        attn_mask=bias.unsqueeze(0),
    )[0]
    return out[:, tiles.scatter_index].transpose(0, 1)


class TestVsaH3Schedule(unittest.TestCase):
    def test_dit_layer_index_only_matches_top_level_blocks(self):
        self.assertEqual(_dit_layer_index("blocks.7.attn"), 7)
        self.assertIsNone(_dit_layer_index("token_refiner.blocks.7.attn"))
        self.assertIsNone(_dit_layer_index("attn"))

    def test_defaults_when_config_is_empty(self):
        with patch(_SERVER_ARGS, return_value=_FakeServerArgs({})):
            schedule = VsaH3Schedule.from_server_args()
        self.assertEqual(schedule.sparsity, DEFAULT_SPARSITY)
        self.assertEqual(schedule.skip_first_steps, DEFAULT_SKIP_FIRST_STEPS)
        self.assertEqual(schedule.prefix_mode, "exempt")
        self.assertEqual(schedule.head_chunk, 0)
        self.assertEqual(schedule.head_chunk_budget_mib, DEFAULT_HEAD_CHUNK_BUDGET_MIB)

    def test_accepts_the_wan_vsa_sparsity_key(self):
        """The Wan VSA stages already put ``VSA_sparsity`` in this bag."""
        with patch(_SERVER_ARGS, return_value=_FakeServerArgs({"VSA_sparsity": 0.75})):
            self.assertEqual(VsaH3Schedule.from_server_args().sparsity, 0.75)
        with patch(
            _SERVER_ARGS,
            return_value=_FakeServerArgs({"VSA_sparsity": 0.75, "sparsity": 0.5}),
        ):
            self.assertEqual(VsaH3Schedule.from_server_args().sparsity, 0.5)

    def test_zero_sparsity_stays_legal(self):
        """It is the calibration setting these tests check dense against."""
        with patch(_SERVER_ARGS, return_value=_FakeServerArgs({"sparsity": 0.0})):
            self.assertEqual(VsaH3Schedule.from_server_args().sparsity, 0.0)

    def test_rejects_out_of_range_values(self):
        for config in (
            {"sparsity": 1.0},
            {"sparsity": -0.1},
            {"prefix_mode": "everything"},
            {"skip_first_steps": -1},
            {"skip_last_steps": -1},
            {"skip_first_layers": -1},
            {"min_seq_len": 32},
            {"head_chunk": -1},
            {"head_chunk_budget_mib": 0},
        ):
            with self.subTest(config=config):
                with patch(_SERVER_ARGS, return_value=_FakeServerArgs(config)):
                    with self.assertRaises(ValueError):
                        VsaH3Schedule.from_server_args()

    def test_compute_topk_is_clamped(self):
        self.assertEqual(compute_topk(0.0, 100), 100)
        self.assertEqual(compute_topk(0.9, 100), 10)
        # A sparsity that would round to nothing still keeps one tile: a query
        # row with an empty key list has no softmax at all.
        self.assertEqual(compute_topk(0.999, 100), 1)


class TestVsaH3Backend(unittest.TestCase):
    def test_the_advertised_builder_can_be_built(self):
        builder = VideoSparseAttentionH3Backend.get_builder_cls()()
        metadata = builder.build(current_timestep=3)
        self.assertEqual(metadata.current_timestep, 3)

    def test_declares_packed_varlen_but_not_ring(self):
        self.assertTrue(VideoSparseAttentionH3Backend.supports_packed_varlen())
        self.assertFalse(VideoSparseAttentionH3Backend.supports_ring_rotation())

    def test_head_sizes_match_the_kernel(self):
        self.assertEqual(
            VideoSparseAttentionH3Backend.get_supported_head_sizes(), [64, 128]
        )


class TestVsaH3TileGeometry(unittest.TestCase):
    """The tiling is index arithmetic; it is checkable without a GPU."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.geometry = _geometry()
        self.tiles = _tile_geometry(PREFIX, VIDEO_GRID, self.device)

    def test_prefix_tiles_never_straddle_a_segment(self):
        expected = []
        for segment in PREFIX:
            full, remainder = divmod(segment, BLOCK)
            expected.extend([BLOCK] * full)
            if remainder:
                expected.append(remainder)
        self.assertEqual(
            self.tiles.variable_block_sizes[: self.tiles.num_prefix_tiles].tolist(),
            expected,
        )

    def test_video_tiles_are_space_time_cubes(self):
        # (4, 4, 4) over an 8 x 24 x 42 grid: the last w tile is a partial 2.
        self.assertEqual(self.tiles.num_video_tiles, 2 * 6 * 11)
        sizes = self.tiles.variable_block_sizes[self.tiles.num_prefix_tiles :]
        self.assertEqual(int(sizes.max()), BLOCK)
        self.assertEqual(int(sizes.min()), 4 * 4 * 2)

    def test_sizes_account_for_every_live_row(self):
        self.assertEqual(
            int(self.tiles.variable_block_sizes.sum()), self.geometry.live_rows
        )

    def test_scatter_index_is_an_injective_map_into_live_slots(self):
        index = self.tiles.scatter_index
        self.assertEqual(index.numel(), self.geometry.live_rows)
        self.assertEqual(int(torch.unique(index).numel()), index.numel())
        # Every target slot must be a live slot of its tile, never padding.
        within = index % BLOCK
        tile_of = index // BLOCK
        self.assertTrue(bool((within < self.tiles.variable_block_sizes[tile_of]).all()))

    def test_prefix_rows_keep_their_packed_order(self):
        prefix_rows = sum(PREFIX)
        index = self.tiles.scatter_index[:prefix_rows]
        self.assertTrue(bool((index.diff() > 0).all()))

    def test_video_rows_are_permuted_into_cubes(self):
        """The first video tile is the (0:4, 0:4, 0:4) corner of the grid."""
        prefix_rows = sum(PREFIX)
        grid_t, grid_h, grid_w = VIDEO_GRID
        corner = [
            prefix_rows + t * grid_h * grid_w + h * grid_w + w
            for t in range(4)
            for h in range(4)
            for w in range(4)
        ]
        first_tile = self.tiles.num_prefix_tiles * BLOCK
        placed = self.tiles.scatter_index[corner]
        self.assertEqual(
            sorted(placed.tolist()), list(range(first_tile, first_tile + BLOCK))
        )

    def test_pad_index_is_exactly_the_leftover_slots(self):
        """``_tile`` only zeroes these, so they must cover every non-live slot."""
        live = torch.zeros(self.tiles.padded_rows, dtype=torch.bool)
        live[self.tiles.scatter_index] = True
        self.assertEqual(
            sorted(self.tiles.pad_index.tolist()),
            torch.nonzero(~live, as_tuple=False).view(-1).tolist(),
        )
        self.assertEqual(
            self.tiles.pad_index.numel() + self.geometry.live_rows,
            self.tiles.padded_rows,
        )

    def test_a_video_only_sequence_has_no_prefix_tiles(self):
        tiles = _tile_geometry((), (4, 4, 4), torch.device("cpu"))
        self.assertEqual(tiles.num_prefix_tiles, 0)
        self.assertEqual(tiles.num_video_tiles, 1)


class TestVsaH3Gating(unittest.TestCase):
    """Which calls are allowed to take the sparse path at all."""

    def test_dit_layers_are_eligible(self):
        self.assertTrue(_make_impl(prefix="blocks.5.attn").layer_enabled)

    def test_everything_outside_the_block_stack_is_not(self):
        self.assertFalse(_make_impl(prefix="token_refiner.blocks.0.attn").layer_enabled)

    def test_causal_layers_are_not(self):
        """Selection has no notion of a diagonal; the kernel masks only pads."""
        self.assertFalse(_make_impl(prefix="blocks.5.attn", causal=True).layer_enabled)

    def test_gqa_is_not(self):
        self.assertFalse(
            _make_impl(prefix="blocks.5.attn", num_kv_heads=1).layer_enabled
        )

    def test_the_layer_cutoffs_are_applied(self):
        self.assertFalse(
            _make_impl({"skip_first_layers": 6}, prefix="blocks.5.attn").layer_enabled
        )
        self.assertFalse(
            _make_impl({"dense_layers": [5]}, prefix="blocks.5.attn").layer_enabled
        )
        self.assertTrue(
            _make_impl({"dense_layers": [4]}, prefix="blocks.5.attn").layer_enabled
        )

    def test_the_step_cutoffs_are_applied(self):
        impl = _make_impl({"skip_first_steps": 10, "skip_last_steps": 2})
        with _at_step(9):
            self.assertFalse(impl._step_enabled())
        with _at_step(10):
            self.assertTrue(impl._step_enabled())

    def test_a_sequence_the_geometry_does_not_describe_runs_dense(self):
        impl = _make_impl({"skip_first_steps": 0})
        geometry = _geometry()
        with _at_step(3):
            self.assertIsNone(impl._geometry_for(geometry.live_rows))
            with vsa_h3_sequence_geometry(geometry):
                self.assertIsNotNone(impl._geometry_for(geometry.live_rows))
                self.assertIsNone(impl._geometry_for(geometry.live_rows + 64))


class TestVsaH3HeadChunking(unittest.TestCase):
    """The slice is sized in bytes, because the transients grow with the shape.

    Four tile buffers plus a ``tiles x tiles`` fp32 score matrix per head is
    3.7 GiB across 28 rank-local heads at a 116k-row sequence -- enough to OOM
    a 32 GiB card the dense path fits on with room to spare.
    """

    def setUp(self):
        self.short = _tile_geometry(PREFIX, VIDEO_GRID, torch.device("cpu"))
        # ~116k live rows, the shape that first hit the OOM.
        self.long = _tile_geometry((512, 300, 2000), (112, 24, 42), torch.device("cpu"))

    def _bytes_per_head(self, tiles):
        return 4 * tiles.padded_rows * HEAD_DIM * 2 + tiles.num_tiles**2 * 4

    def test_the_slice_stays_inside_the_budget(self):
        impl = _make_impl()
        for tiles in (self.short, self.long):
            chunk = impl._head_chunk_for(tiles, 28, itemsize=2)
            with self.subTest(tiles=tiles.num_tiles):
                self.assertGreaterEqual(chunk, 1)
                self.assertLessEqual(
                    chunk * self._bytes_per_head(tiles),
                    DEFAULT_HEAD_CHUNK_BUDGET_MIB * 1024 * 1024,
                )

    def test_a_long_sequence_is_sliced_harder_than_a_short_one(self):
        impl = _make_impl()
        self.assertLess(
            impl._head_chunk_for(self.long, 28, itemsize=2),
            impl._head_chunk_for(self.short, 28, itemsize=2),
        )

    def test_a_short_sequence_needs_no_slicing(self):
        impl = _make_impl()
        self.assertEqual(impl._head_chunk_for(self.short, 14, itemsize=2), 14)

    def test_an_explicit_head_chunk_wins(self):
        impl = _make_impl({"head_chunk": 7})
        self.assertEqual(impl._head_chunk_for(self.long, 28, itemsize=2), 7)

    def test_a_budget_smaller_than_one_head_still_runs_one(self):
        impl = _make_impl({"head_chunk_budget_mib": 1})
        self.assertEqual(impl._head_chunk_for(self.long, 28, itemsize=2), 1)


@requires_gpu
class TestVsaH3Numerics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.geometry = _geometry()
        self.q, self.k, self.v = _rows(self.geometry, self.device)

    def _run(self, impl, q=None, k=None, v=None):
        q = self.q if q is None else q
        k = self.k if k is None else k
        v = self.v if v is None else v
        with _at_step(30), vsa_h3_sequence_geometry(self.geometry):
            return impl.forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))[0]

    def test_zero_sparsity_reproduces_dense_attention(self):
        impl = _make_impl({"sparsity": 0.0, "skip_first_steps": 0, "min_seq_len": 64})
        out = self._run(impl)
        reference = _dense_ref(self.q, self.k, self.v)
        torch.testing.assert_close(out, reference, atol=2e-3, rtol=2e-2)

    def test_selection_matches_an_independent_implementation(self):
        for sparsity in (0.5, 0.9):
            for mode in ("exempt", "compete"):
                with self.subTest(sparsity=sparsity, prefix_mode=mode):
                    impl = _make_impl(
                        {
                            "sparsity": sparsity,
                            "prefix_mode": mode,
                            "skip_first_steps": 0,
                            "min_seq_len": 64,
                        }
                    )
                    out = self._run(impl)
                    reference = _masked_ref(
                        self.q, self.k, self.v, self.geometry, sparsity, mode
                    )
                    torch.testing.assert_close(out, reference, atol=2e-3, rtol=2e-2)

    def test_head_chunking_is_exact(self):
        """Selection is head-parallel, so slicing must change nothing."""
        config = {"sparsity": 0.9, "skip_first_steps": 0, "min_seq_len": 64}
        whole = self._run(_make_impl({**config, "head_chunk": NUM_HEADS}))
        for chunk in (1, 2, 3):
            with self.subTest(head_chunk=chunk):
                sliced = self._run(_make_impl({**config, "head_chunk": chunk}))
                self.assertTrue(torch.equal(whole, sliced))

    def test_varlen_zeroes_the_padding_tail(self):
        live = self.geometry.live_rows
        total = (live + BLOCK - 1) // BLOCK * BLOCK + BLOCK
        q, k, v = (
            torch.randn(
                total, NUM_HEADS, HEAD_DIM, device=self.device, dtype=torch.bfloat16
            )
            for _ in range(3)
        )
        for tensor in (q, k, v):
            tensor[live:] = 0
        impl = _make_impl({"sparsity": 0.9, "skip_first_steps": 0, "min_seq_len": 64})
        with _at_step(30), vsa_h3_sequence_geometry(self.geometry):
            out = impl.forward_varlen(
                q,
                k,
                v,
                cu_seqlens=torch.tensor([0, live, total], device=self.device),
                max_seqlen=live,
                cu_seqlens_host=(0, live, total),
            )
        reference = _masked_ref(q[:live], k[:live], v[:live], self.geometry, 0.9)
        torch.testing.assert_close(out[:live], reference, atol=2e-3, rtol=2e-2)
        self.assertTrue(bool((out[live:] == 0).all()))

    def test_a_warmup_step_takes_the_dense_path(self):
        impl = _make_impl({"sparsity": 0.9, "skip_first_steps": 10, "min_seq_len": 64})
        with _at_step(3), vsa_h3_sequence_geometry(self.geometry):
            out = impl.forward(
                self.q.unsqueeze(0), self.k.unsqueeze(0), self.v.unsqueeze(0)
            )[0]
        torch.testing.assert_close(
            out, _dense_ref(self.q, self.k, self.v), atol=2e-2, rtol=5e-2
        )


if __name__ == "__main__":
    unittest.main()
