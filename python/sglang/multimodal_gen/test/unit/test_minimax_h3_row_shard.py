# SPDX-License-Identifier: Apache-2.0
"""The row-sharded block stack's fused kernel reproduces the unfused path.

`modulate_quantize_gather` replaces the AdaLN modulation kernel followed by the
linear's own FP8 quantizer with one kernel, and the row-sharded path is only
bit-exact if that kernel is: every assertion here is on bits, not tolerances.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA for the Triton kernels"
)

HIDDEN = 5376
GROUP = 128
MODALITIES = 9


def _unfused(x, shift, scale, indices):
    from sglang.kernels.ops.diffusion.triton.indexed_modulation import (
        indexed_scale_shift_bf16_,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    return sglang_per_token_group_quant_fp8(
        indexed_scale_shift_bf16_(x.clone(), shift, scale, indices), GROUP
    )


def _fused(x, shift, scale, indices):
    from sglang.kernels.ops.diffusion.triton.indexed_modulation import (
        indexed_scale_shift_quant_fp8,
    )

    return indexed_scale_shift_quant_fp8(x, shift, scale, indices, group_size=GROUP)


def _params(scale=0.5):
    return tuple(
        (torch.randn(MODALITIES, HIDDEN, device="cuda") * scale).bfloat16()
        for _ in range(2)
    )


def _cases():
    torch.manual_seed(0)
    shift, scale = _params()
    zeros = torch.zeros(MODALITIES, HIDDEN, device="cuda", dtype=torch.bfloat16)
    normal = (torch.randn(2048, HIDDEN, device="cuda") * 3).bfloat16()
    heavy = (
        torch.randn(2048, HIDDEN, device="cuda")
        * torch.exp(torch.randn(2048, 1, device="cuda") * 6)
    ).bfloat16()
    # Whole zero rows put amax on the 1e-10 floor; 1e-30 rows sit far below
    # e4m3's smallest subnormal, where flush-to-zero would differ first.
    tiny = torch.zeros(256, HIDDEN, device="cuda", dtype=torch.bfloat16)
    tiny[1::2] = (torch.randn(128, HIDDEN, device="cuda") * 1e-30).bfloat16()
    infinite = torch.randn(64, HIDDEN, device="cuda").bfloat16()
    infinite[3, 7] = float("inf")
    infinite[5, 300] = float("-inf")
    return {
        "normal": (normal, shift, scale),
        "heavy_tailed": (heavy, shift, scale),
        "zero_and_tiny": (tiny, zeros, zeros),
        "infinite": (infinite, zeros, zeros),
    }


@pytest.mark.parametrize(
    "case", ["normal", "heavy_tailed", "zero_and_tiny", "infinite"]
)
def test_fused_modulate_quant_is_bit_exact(case):
    x, shift, scale = _cases()[case]
    indices = torch.randint(0, MODALITIES, (x.shape[0],), device="cuda")
    expected_q, expected_scale = _unfused(x, shift, scale, indices)
    q, q_scale = _fused(x, shift, scale, indices)
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(q_scale.view(torch.int32), expected_scale.view(torch.int32))


def test_fused_modulate_quant_leaves_its_input_alone():
    x, shift, scale = _cases()["normal"]
    before = x.clone()
    _fused(x, shift, scale, torch.zeros(x.shape[0], dtype=torch.long, device="cuda"))
    assert torch.equal(x, before)


def test_fused_modulate_quant_refuses_what_the_unfused_path_would_not_fuse():
    # The unfused modulation falls back to torch arithmetic for these, so a
    # kernel that ran anyway would silently disagree with it.
    x, shift, scale = _cases()["normal"]
    indices = torch.zeros(x.shape[0], dtype=torch.long, device="cuda")
    with pytest.raises(ValueError):
        _fused(x.float(), shift, scale, indices)
    with pytest.raises(ValueError):
        _fused(x.t().contiguous().t(), shift, scale, indices)
    with pytest.raises(ValueError):
        _fused(x, shift.float(), scale, indices)


def _row_shard():
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3_row_shard

    return minimax_h3_row_shard


@pytest.mark.parametrize("chunks", [1, 2, 4])
def test_a_rank_holds_what_each_chunks_reduce_scatter_hands_it(chunks):
    # Reduce-scattering chunk c of the global rows gives rank r its r-th half
    # of that chunk; the shard has to be those halves in chunk order, or the
    # residual a rank carries would not be the rows its collectives produce.
    world, rows = 2, 64
    full = torch.arange(rows * 3).view(rows, 3)
    whole = rows // chunks
    for rank in range(world):
        shard = _row_shard().shard_rows(full, chunks=chunks, rank=rank, world=world)
        expected = torch.cat(
            [
                full[
                    c * whole
                    + rank * whole // world : c * whole
                    + (rank + 1) * whole // world
                ]
                for c in range(chunks)
            ]
        )
        assert torch.equal(shard, expected)


@pytest.mark.parametrize("chunks", [1, 2, 4])
def test_unsharding_restores_the_original_row_order(chunks):
    # unshard_rows reorders an all-gather of every rank's shard; stand in for
    # the collective with a concatenation in rank order.
    world, rows = 2, 64
    full = torch.arange(rows * 3).view(rows, 3)
    shards = [
        _row_shard().shard_rows(full, chunks=chunks, rank=rank, world=world)
        for rank in range(world)
    ]

    class _Group:
        world_size = world

        @staticmethod
        def all_gather(tensor, dim):
            return torch.cat(shards, dim=dim)

    restored = _row_shard().unshard_rows(shards[0], chunks=chunks, group=_Group)
    assert torch.equal(restored, full)


def test_chunk_count_backs_off_to_what_the_rows_divide_into():
    row_chunks = _row_shard().row_chunks
    # c chunks of two halves, each a multiple of four rows, needs 8c | rows;
    # the largest c up to four that divides is taken.
    assert row_chunks(column_linears=(), tp_size=2, rows=64 * 7) == 4
    assert row_chunks(column_linears=(), tp_size=2, rows=24 * 5) == 3
    assert row_chunks(column_linears=(), tp_size=2, rows=8 * 5) == 1
    assert row_chunks(column_linears=(), tp_size=2, rows=12) == 0
    # Only TP=2 is claimed bit-exact.
    assert row_chunks(column_linears=(), tp_size=4, rows=64 * 7) == 0


def _silu_mul_unfused(x):
    from sglang.kernels.ops.activation.activation import (
        silu_and_mul_with_activation_rounding,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    return sglang_per_token_group_quant_fp8(
        silu_and_mul_with_activation_rounding(x), GROUP
    )


def test_fused_silu_mul_quant_covers_every_gate_value():
    # The activation depends on the bf16 gate alone, so all 65536 of them pin
    # it; up = 1 keeps the product exact, and a random up then checks the
    # multiply. NaN gates are left out: their payload is not a contract.
    from sglang.kernels.ops.diffusion.triton.silu_mul_quant_fp8 import (
        silu_mul_quant_fp8,
    )

    torch.manual_seed(0)
    gates = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    gates = gates.view(torch.bfloat16).cuda()
    gates = gates[~torch.isnan(gates)]
    # Two groups a row at least: with a single group sglang dispatches a
    # per-token quantizer that skips the 1e-10 floor, which H3's widths never
    # reach.
    width = 2 * GROUP
    gates = torch.cat([gates, gates.new_zeros(-gates.numel() % width)]).view(-1, width)
    for up in (torch.ones_like(gates), torch.randn_like(gates.float()).bfloat16()):
        x = torch.cat([gates, up], dim=1).contiguous()
        expected_q, expected_scale = _silu_mul_unfused(x)
        q, q_scale = silu_mul_quant_fp8(x, group_size=GROUP)
        assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
        assert torch.equal(q_scale.view(torch.int32), expected_scale.view(torch.int32))


@pytest.mark.parametrize("case", ["normal", "heavy_tailed", "zero_and_tiny"])
def test_fused_silu_mul_quant_is_bit_exact_at_mlp_width(case):
    from sglang.kernels.ops.diffusion.triton.silu_mul_quant_fp8 import (
        silu_mul_quant_fp8,
    )

    x = _cases()[case][0]
    # [gate | up] halves of an fc1 output, 7168 wide each at TP=2.
    x = torch.cat([x, x.flip(0)], dim=1)[:, : 2 * 7168].contiguous()
    expected_q, expected_scale = _silu_mul_unfused(x)
    q, q_scale = silu_mul_quant_fp8(x, group_size=GROUP)
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(q_scale.view(torch.int32), expected_scale.view(torch.int32))
