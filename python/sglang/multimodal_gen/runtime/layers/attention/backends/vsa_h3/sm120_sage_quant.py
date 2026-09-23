# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashInfer's flashinfer/cute_dsl/sparse/bsa_utils/sage_quant_sm120.py
# (Apache-2.0, Copyright (c) 2025 by FlashInfer team).
"""FlashInfer's SM120 Sage quantizers, reading VSA-H3's packed rows in place.

FlashInfer's quantizers take contiguous BHSD, so the Sage path used to lay Q,
K and V out in tile order as bf16 first and hand the result to them -- a
buffer written only to be read once. These are the same four kernels with only
their loads changed: a tiled slot reaches its packed row through ``tile_rows``,
and a slot past its tile's live count reads zero, which is exactly what the
zero-filled tile buffer held there. Every reduction, scale and rounding step is
theirs verbatim, so the operands come out bit for bit the same.

The output side gets the matching treatment: ``scatter_tile_rows`` writes a
launch's BHSD result straight to the packed rows its live slots hold, instead
of transposing it into a tile-ordered buffer and gathering from that.

Keep the arithmetic in step with FlashInfer's when upgrading it: the
attention kernel consumes these operands, and a drifted quantizer would feed it
a layout or a scale it does not expect.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK_SIZE = 64
SAGE_Q_GROUP_SIZE = 32
SAGE_Q_BLOCK_SIZE = 128
SAGE_K_BLOCK_SIZE = 64
SAGE_KV_STATS_CHUNK = 256
SAGE_V_SCALE_MAX = 2.25
_KV_STATS_DIM_TILE = 32


@triton.jit
def _packed_rows(tile_rows_ptr, block_sizes_ptr, slot, in_range):
    """Packed row behind each tiled slot, and whether that slot is live."""
    size = tl.load(block_sizes_ptr + slot // 64, mask=in_range, other=0)
    live = in_range & ((slot % 64) < size)
    row = tl.load(tile_rows_ptr + slot, mask=in_range, other=0).to(tl.int64)
    return row, live


@triton.jit
def _quantize_sage_q_packed_kernel(
    q_ptr,
    tile_rows_ptr,
    block_sizes_ptr,
    q8_ptr,
    scale_ptr,
    q_stride_s,
    q_stride_h,
    q8_stride_h,
    q8_stride_s,
    scale_stride_h,
    first_slot,
    seqlen_q,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    group_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    row_idx = group_idx * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    dim_idx = tl.arange(0, HEAD_DIM)
    valid = row_idx < seqlen_q
    packed, live = _packed_rows(
        tile_rows_ptr, block_sizes_ptr, first_slot + row_idx, valid
    )
    q = tl.load(
        q_ptr
        + head_idx * q_stride_h
        + packed[:, None] * q_stride_s
        + dim_idx[None, :],
        mask=live[:, None],
        other=0.0,
    ).to(tl.float32)

    row_max = tl.max(tl.abs(q), axis=1)
    amax = tl.maximum(tl.max(row_max, axis=0), 1.0e-7)
    scale = amax / 127.0
    q_quant = tl.maximum(
        tl.minimum(libdevice.rint(q / scale), 127.0),
        -127.0,
    )
    tl.store(
        q8_ptr + head_idx * q8_stride_h + row_idx[:, None] * q8_stride_s + dim_idx[None, :],
        q_quant.to(tl.int8),
        mask=valid[:, None],
    )
    tl.store(scale_ptr + head_idx * scale_stride_h + group_idx, scale)


@triton.jit
def _sage_kv_stats_partial_packed_kernel(
    k_ptr,
    v_ptr,
    tile_rows_ptr,
    block_sizes_ptr,
    k_partial_ptr,
    v_partial_ptr,
    k_stride_s,
    k_stride_h,
    v_stride_s,
    v_stride_h,
    partial_stride_h,
    partial_stride_c,
    seqlen_k,
    CHUNK_SIZE: tl.constexpr,
    DIM_TILE: tl.constexpr,
):
    chunk_idx = tl.program_id(0)
    dim_tile_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    row_idx = chunk_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    dim_idx = dim_tile_idx * DIM_TILE + tl.arange(0, DIM_TILE)
    valid = row_idx < seqlen_k
    packed, live = _packed_rows(tile_rows_ptr, block_sizes_ptr, row_idx, valid)
    k = tl.load(
        k_ptr + head_idx * k_stride_h + packed[:, None] * k_stride_s + dim_idx[None, :],
        mask=live[:, None],
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr + head_idx * v_stride_h + packed[:, None] * v_stride_s + dim_idx[None, :],
        mask=live[:, None],
        other=0.0,
    ).to(tl.float32)

    partial_base = head_idx * partial_stride_h + chunk_idx * partial_stride_c + dim_idx
    tl.store(partial_base + k_partial_ptr, tl.sum(k, axis=0))
    tl.store(partial_base + v_partial_ptr, tl.max(tl.abs(v), axis=0))


@triton.jit
def _sage_kv_stats_finalize_kernel(
    k_partial_ptr,
    v_partial_ptr,
    k_mean_ptr,
    v_scale_ptr,
    partial_stride_h,
    partial_stride_c,
    mean_stride_h,
    scale_stride_h,
    num_chunks,
    seqlen_k,
    DIM_TILE: tl.constexpr,
    REDUCE_TILE: tl.constexpr,
    SCALE_MAX: tl.constexpr,
):
    dim_tile_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_idx = dim_tile_idx * DIM_TILE + tl.arange(0, DIM_TILE)

    sum_acc = tl.zeros((DIM_TILE,), tl.float32)
    max_acc = tl.zeros((DIM_TILE,), tl.float32)
    chunk_base = 0
    while chunk_base < num_chunks:
        chunk_idx = chunk_base + tl.arange(0, REDUCE_TILE)
        mask = chunk_idx < num_chunks
        partial_offset = (
            head_idx * partial_stride_h
            + chunk_idx[:, None] * partial_stride_c
            + dim_idx[None, :]
        )
        partial_sum = tl.load(k_partial_ptr + partial_offset, mask=mask[:, None], other=0.0)
        partial_max = tl.load(v_partial_ptr + partial_offset, mask=mask[:, None], other=0.0)
        sum_acc += tl.sum(partial_sum, axis=0)
        max_acc = tl.maximum(max_acc, tl.max(partial_max, axis=0))
        chunk_base += REDUCE_TILE

    mean = sum_acc / seqlen_k
    scale = tl.maximum(max_acc, 1.0e-7) / SCALE_MAX
    tl.store(k_mean_ptr + head_idx * mean_stride_h + dim_idx, mean)
    tl.store(v_scale_ptr + head_idx * scale_stride_h + dim_idx, scale)


@triton.jit
def _quantize_sage_kv_packed_kernel(
    k_ptr,
    v_ptr,
    tile_rows_ptr,
    block_sizes_ptr,
    k_mean_ptr,
    v_scale_ptr,
    k8_ptr,
    v8_ptr,
    k_scale_ptr,
    k_stride_s,
    k_stride_h,
    v_stride_s,
    v_stride_h,
    mean_stride_h,
    v_scale_stride_h,
    k8_stride_h,
    k8_stride_s,
    v8_stride_h,
    v8_stride_d,
    k_scale_stride_h,
    seqlen_k,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    local_row = tl.arange(0, BLOCK_SIZE)
    row_idx = block_idx * BLOCK_SIZE + local_row
    dim_idx = tl.arange(0, HEAD_DIM)
    valid = row_idx < seqlen_k
    packed, live = _packed_rows(tile_rows_ptr, block_sizes_ptr, row_idx, valid)

    mean = tl.load(k_mean_ptr + head_idx * mean_stride_h + dim_idx).to(tl.float32)
    v_scale = tl.load(v_scale_ptr + head_idx * v_scale_stride_h + dim_idx).to(
        tl.float32
    )
    k = tl.load(
        k_ptr + head_idx * k_stride_h + packed[:, None] * k_stride_s + dim_idx[None, :],
        mask=live[:, None],
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr + head_idx * v_stride_h + packed[:, None] * v_stride_s + dim_idx[None, :],
        mask=live[:, None],
        other=0.0,
    ).to(tl.float32)

    # ``valid``, not ``live``: a pad slot inside the buffer centres to -mean
    # exactly as the zero FlashInfer's tile buffer held there does.
    k_centered = tl.where(valid[:, None], k - mean[None, :], 0.0)
    row_max = tl.max(tl.abs(k_centered), axis=1)
    k_amax = tl.maximum(tl.max(row_max, axis=0), 1.0e-7)
    k_scale = k_amax / 127.0
    k_quant = tl.maximum(
        tl.minimum(libdevice.rint(k_centered / k_scale), 127.0),
        -127.0,
    )
    tl.store(
        k8_ptr + head_idx * k8_stride_h + row_idx[:, None] * k8_stride_s + dim_idx[None, :],
        k_quant.to(tl.int8),
        mask=valid[:, None],
    )
    tl.store(k_scale_ptr + head_idx * k_scale_stride_h + block_idx, k_scale)

    # The 16-token physical permutation FlashInfer bakes into V for its FP8
    # P.V MMA; it has to stay bit-for-bit what the attention kernel expects.
    row_mod = local_row % 16
    physical_row = (
        block_idx * BLOCK_SIZE
        + (local_row // 16) * 16
        + (row_mod // 8) * 2
        + ((row_mod // 2) % 4) * 4
        + row_mod % 2
    )
    v_quant = tl.where(valid[:, None], v / v_scale[None, :], 0.0)
    tl.store(
        v8_ptr
        + head_idx * v8_stride_h
        + dim_idx[:, None] * v8_stride_d
        + physical_row[None, :],
        tl.trans(v_quant).to(v8_ptr.dtype.element_ty),
    )


@triton.jit
def _scatter_tile_rows_kernel(
    result_ptr,
    tile_rows_ptr,
    block_sizes_ptr,
    out_ptr,
    result_stride_h,
    result_stride_s,
    out_stride_s,
    out_stride_h,
    first_tile,
    HEAD_DIM: tl.constexpr,
):
    tile = tl.program_id(0)
    head_idx = tl.program_id(1)
    local = tl.arange(0, 64)
    dim_idx = tl.arange(0, HEAD_DIM)
    slot = (first_tile + tile) * 64 + local
    live = local < tl.load(block_sizes_ptr + first_tile + tile)
    packed = tl.load(tile_rows_ptr + slot).to(tl.int64)
    values = tl.load(
        result_ptr
        + head_idx * result_stride_h
        + (tile * 64 + local)[:, None] * result_stride_s
        + dim_idx[None, :]
    )
    tl.store(
        out_ptr + packed[:, None] * out_stride_s + head_idx * out_stride_h + dim_idx[None, :],
        values,
        mask=live[:, None],
    )


def quantize_sage_q_packed(
    query: torch.Tensor,
    *,
    tile_rows: torch.Tensor,
    block_sizes: torch.Tensor,
    first_row: int,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed ``[S, H, 128]`` Q -> FlashInfer's ``(q_int8, q_scale)`` for slots
    ``[first_row, first_row + rows)`` of the tiled sequence."""
    heads, head_dim = query.shape[1], query.shape[2]
    num_groups = triton.cdiv(rows, SAGE_Q_BLOCK_SIZE) * (
        SAGE_Q_BLOCK_SIZE // SAGE_Q_GROUP_SIZE
    )
    q_int8 = query.new_empty((1, heads, rows, head_dim), dtype=torch.int8)
    q_scale = query.new_empty((1, heads, num_groups), dtype=torch.float32)
    _quantize_sage_q_packed_kernel[(num_groups, heads)](
        query,
        tile_rows,
        block_sizes,
        q_int8,
        q_scale,
        query.stride(0),
        query.stride(1),
        q_int8.stride(1),
        q_int8.stride(2),
        q_scale.stride(1),
        first_row,
        rows,
        HEAD_DIM=head_dim,
        GROUP_SIZE=SAGE_Q_GROUP_SIZE,
        num_warps=8,
    )
    return q_int8, q_scale


def quantize_sage_kv_packed(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    tile_rows: torch.Tensor,
    block_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packed ``[S, H, 128]`` K and V -> FlashInfer's
    ``(k_int8, v_fp8, k_scale, v_scale)`` over the whole tiled sequence."""
    heads, head_dim = key.shape[1], key.shape[2]
    padded_rows = int(tile_rows.numel())
    num_blocks = padded_rows // SAGE_K_BLOCK_SIZE
    k_int8 = key.new_empty((1, heads, padded_rows, head_dim), dtype=torch.int8)
    v_fp8 = key.new_empty((1, heads, head_dim, padded_rows), dtype=torch.float8_e4m3fn)
    k_scale = key.new_empty((1, heads, num_blocks), dtype=torch.float32)
    v_scale = key.new_empty((1, heads, head_dim), dtype=torch.float32)

    num_chunks = triton.cdiv(padded_rows, SAGE_KV_STATS_CHUNK)
    k_partial = key.new_empty((heads, num_chunks, head_dim), dtype=torch.float32)
    v_partial = torch.empty_like(k_partial)
    k_mean = key.new_empty((heads, head_dim), dtype=torch.bfloat16)

    _sage_kv_stats_partial_packed_kernel[
        (num_chunks, head_dim // _KV_STATS_DIM_TILE, heads)
    ](
        key,
        value,
        tile_rows,
        block_sizes,
        k_partial,
        v_partial,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        k_partial.stride(0),
        k_partial.stride(1),
        padded_rows,
        CHUNK_SIZE=SAGE_KV_STATS_CHUNK,
        DIM_TILE=_KV_STATS_DIM_TILE,
        num_warps=8,
    )
    _sage_kv_stats_finalize_kernel[(head_dim // _KV_STATS_DIM_TILE, heads)](
        k_partial,
        v_partial,
        k_mean,
        v_scale,
        k_partial.stride(0),
        k_partial.stride(1),
        k_mean.stride(0),
        v_scale.stride(1),
        num_chunks,
        padded_rows,
        DIM_TILE=_KV_STATS_DIM_TILE,
        REDUCE_TILE=16,
        SCALE_MAX=SAGE_V_SCALE_MAX,
        num_warps=4,
    )
    _quantize_sage_kv_packed_kernel[(num_blocks, heads)](
        key,
        value,
        tile_rows,
        block_sizes,
        k_mean,
        v_scale,
        k_int8,
        v_fp8,
        k_scale,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        k_mean.stride(0),
        v_scale.stride(1),
        k_int8.stride(1),
        k_int8.stride(2),
        v_fp8.stride(1),
        v_fp8.stride(2),
        k_scale.stride(1),
        padded_rows,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=SAGE_K_BLOCK_SIZE,
        num_warps=8,
    )
    return k_int8, v_fp8, k_scale, v_scale


def scatter_tile_rows(
    result: torch.Tensor,
    out: torch.Tensor,
    *,
    tile_rows: torch.Tensor,
    block_sizes: torch.Tensor,
    first_tile: int,
) -> None:
    """Write a launch's ``[1, H, tiles * 64, D]`` result to the packed
    ``[S, H, D]`` rows its live slots hold; pad slots are never written."""
    heads, rows, head_dim = result.shape[1], result.shape[2], result.shape[3]
    _scatter_tile_rows_kernel[(rows // BLOCK_SIZE, heads)](
        result,
        tile_rows,
        block_sizes,
        out,
        result.stride(1),
        result.stride(2),
        out.stride(0),
        out.stride(1),
        first_tile,
        HEAD_DIM=head_dim,
        num_warps=4,
    )


__all__ = [
    "quantize_sage_kv_packed",
    "quantize_sage_q_packed",
    "scatter_tile_rows",
]
