# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.diffusion.triton.fp8_group_quant import (
    pack_ue8m0,
    quantize_fp8_groups,
    quantize_fp8_groups_ue8m0,
)
from sglang.kernels.ops.diffusion.triton.numerics import round_bf16_to_fp32


@triton.jit
def _indexed_scale_shift_bf16_kernel(
    output_ptr,
    x_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_N)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    shift = tl.load(
        shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0
    ).to(tl.float32)

    one_plus_scale = round_bf16_to_fp32(1.0 + scale)
    scaled = round_bf16_to_fp32(x * one_plus_scale)
    tl.store(
        output_ptr + row * stride_x_row + columns,
        scaled + shift,
        mask=mask,
    )


@triton.jit
def _indexed_scale_shift_quant_fp8_kernel(
    q_ptr,
    q_scale_ptr,
    x_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    num_groups,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    stride_q_row,
    stride_q_scale_row,
    GROUP_SIZE: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    """``_indexed_scale_shift_bf16_kernel`` then per-token-group FP8 quant.

    Both halves reproduce their unfused kernels exactly: the modulation is the
    same Triton arithmetic, rounded to bf16 where that kernel stores it, and the
    quantization is ``fp8_group_quant``'s replica of sglang's quantizer.
    """
    row = tl.program_id(0)
    groups = tl.arange(0, BLOCK_GROUPS)
    lanes = tl.arange(0, GROUP_SIZE)
    columns = groups[:, None] * GROUP_SIZE + lanes[None, :]
    mask = (groups < num_groups)[:, None]
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    shift = tl.load(
        shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    one_plus_scale = round_bf16_to_fp32(1.0 + scale)
    scaled = round_bf16_to_fp32(x * one_plus_scale)
    modulated = (scaled + shift).to(tl.bfloat16).to(tl.float32)

    if SCALE_UE8M0:
        q, exponent = quantize_fp8_groups_ue8m0(modulated)
        packs = tl.arange(0, BLOCK_GROUPS // 4)
        tl.store(
            q_scale_ptr + row * stride_q_scale_row + packs,
            pack_ue8m0(exponent, groups < num_groups, BLOCK_GROUPS),
            mask=packs < tl.cdiv(num_groups, 4),
        )
    else:
        q, q_scale = quantize_fp8_groups(modulated)
        tl.store(
            q_scale_ptr + row * stride_q_scale_row + groups,
            q_scale,
            mask=groups < num_groups,
        )
    tl.store(
        q_ptr + row * stride_q_row + columns,
        q.to(q_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _indexed_gate_bf16_kernel(
    output_ptr,
    x_ptr,
    gate_ptr,
    other_ptr,
    indices_ptr,
    hidden_size,
    stride_output_row,
    stride_x_row,
    stride_gate_row,
    stride_other_row,
    stride_indices,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_N)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    gate = tl.load(
        gate_ptr + index * stride_gate_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    other = tl.load(
        other_ptr + row * stride_other_row + columns, mask=mask, other=0.0
    ).to(tl.float32)

    gated = round_bf16_to_fp32(gate * other)
    tl.store(
        output_ptr + row * stride_output_row + columns,
        x + gated,
        mask=mask,
    )


def indexed_scale_shift_bf16_(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    rows, hidden_size = x.shape
    if rows == 0:
        return x
    block_n = triton.next_power_of_2(hidden_size)
    _indexed_scale_shift_bf16_kernel[(rows,)](
        x,
        x,
        shift,
        scale,
        indices,
        hidden_size,
        x.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        BLOCK_N=block_n,
        num_warps=8,
    )
    return x


def indexed_scale_shift_quant_fp8(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    group_size: int,
    scale_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``indexed_scale_shift_bf16_`` then ``sglang_per_token_group_quant_fp8``.

    Returns the same ``(fp8 rows, row-major per-group fp32 scales)`` pair those
    two calls produce, bit for bit, without writing the modulated rows back.
    With ``scale_ue8m0`` the scales are the UE8M0 exponents that quantizer
    packs for DeepGEMM, four groups to an int32, ``[rows, ceil(groups / 4)]``
    but row-major. ``x`` is left untouched.
    """
    # The unfused modulation takes its fused kernel only for contiguous bf16
    # rows and bf16 parameters, falling back to torch arithmetic otherwise;
    # this kernel has no such fallback, so it refuses rather than disagree.
    if not (
        x.is_cuda
        and x.dtype == shift.dtype == scale.dtype == torch.bfloat16
        and x.is_contiguous()
        and shift.stride(-1) == 1
        and scale.stride(-1) == 1
    ):
        raise ValueError(
            "indexed_scale_shift_quant_fp8 needs contiguous CUDA bf16 rows and "
            "bf16 shift/scale with a unit last stride"
        )
    rows, hidden_size = x.shape
    if hidden_size % group_size:
        raise ValueError(
            f"hidden size {hidden_size} is not a multiple of group size {group_size}"
        )
    num_groups = hidden_size // group_size
    q = torch.empty((rows, hidden_size), device=x.device, dtype=torch.float8_e4m3fn)
    q_scale = (
        torch.empty((rows, -(-num_groups // 4)), device=x.device, dtype=torch.int32)
        if scale_ue8m0
        else torch.empty((rows, num_groups), device=x.device, dtype=torch.float32)
    )
    if rows == 0:
        return q, q_scale
    _indexed_scale_shift_quant_fp8_kernel[(rows,)](
        q,
        q_scale,
        x,
        shift,
        scale,
        indices,
        num_groups,
        x.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        q.stride(0),
        q_scale.stride(0),
        GROUP_SIZE=group_size,
        BLOCK_GROUPS=max(triton.next_power_of_2(num_groups), 4),
        SCALE_UE8M0=scale_ue8m0,
        num_warps=8,
    )
    return q, q_scale


def _indexed_gate_bf16(
    output: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    rows, hidden_size = x.shape
    if rows == 0:
        return output
    block_n = triton.next_power_of_2(hidden_size)
    _indexed_gate_bf16_kernel[(rows,)](
        output,
        x,
        gate,
        other,
        indices,
        hidden_size,
        output.stride(0),
        x.stride(0),
        gate.stride(0),
        other.stride(0),
        indices.stride(0),
        BLOCK_N=block_n,
        num_warps=8,
    )
    return output


def indexed_gate_bf16_(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return _indexed_gate_bf16(x, x, gate, other, indices)


def indexed_gate_bf16(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return _indexed_gate_bf16(torch.empty_like(x), x, gate, other, indices)
