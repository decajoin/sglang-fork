# SPDX-License-Identifier: Apache-2.0
"""``silu_and_mul_with_activation_rounding`` then FP8 group quantization, fused.

An FP8 MLP computes ``silu(gate) * up`` into bf16 rows only for its second
GEMM to read them back and quantize them; this kernel quantizes them on the way
out instead, bit for bit what the two kernels it replaces produce.

The activation half matches the rounded-activation build of ``activation.cuh``,
which is compiled without fast-math: ``gate / (1 + expf(-gate))`` with the
precise ``expf`` -- libdevice's, which is the same routine -- and an IEEE
division, rounded to bf16, then multiplied by ``up`` in fp32 and rounded again.
The gate is a bf16, so the activation has only 65536 inputs, and the test
checks every one of them. The quantization half is ``fp8_group_quant``'s.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sglang.kernels.ops.diffusion.triton.fp8_group_quant import (
    pack_ue8m0,
    quantize_fp8_groups,
    quantize_fp8_groups_ue8m0,
)
from sglang.kernels.ops.diffusion.triton.numerics import div_rn_f32, mul_rn_f32


@triton.jit
def _silu_mul_quant_fp8_kernel(
    q_ptr,
    q_scale_ptr,
    x_ptr,
    width,
    num_groups,
    stride_x_row,
    stride_q_row,
    stride_q_scale_row,
    GROUP_SIZE: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    groups = tl.program_id(1) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    lanes = tl.arange(0, GROUP_SIZE)
    columns = groups[:, None] * GROUP_SIZE + lanes[None, :]
    mask = (groups < num_groups)[:, None]

    gate = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(x_ptr + row * stride_x_row + width + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    activated = div_rn_f32(gate, 1.0 + libdevice.exp(-gate))
    activated = activated.to(tl.bfloat16).to(tl.float32)
    product = mul_rn_f32(activated, up).to(tl.bfloat16).to(tl.float32)

    if SCALE_UE8M0:
        q, exponent = quantize_fp8_groups_ue8m0(product)
        packs = tl.program_id(1) * (BLOCK_GROUPS // 4) + tl.arange(0, BLOCK_GROUPS // 4)
        tl.store(
            q_scale_ptr + row * stride_q_scale_row + packs,
            pack_ue8m0(exponent, groups < num_groups, BLOCK_GROUPS),
            mask=packs < tl.cdiv(num_groups, 4),
        )
    else:
        q, q_scale = quantize_fp8_groups(product)
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


def silu_mul_quant_fp8(
    x: torch.Tensor, *, group_size: int, scale_ue8m0: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[rows, 2 * width]`` bf16 ``[gate | up]`` -> ``(fp8 rows, scales)``.

    The pair ``sglang_per_token_group_quant_fp8`` returns for the rows
    ``silu_and_mul_with_activation_rounding`` would have written: ``[rows,
    width]`` e4m3 and ``[rows, width // group_size]`` row-major fp32 scales,
    or with ``scale_ue8m0`` the UE8M0 exponents that quantizer packs for
    DeepGEMM, four groups to an int32 but row-major.
    """
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()):
        raise ValueError("silu_mul_quant_fp8 needs contiguous CUDA bf16 rows")
    rows, doubled = x.shape
    width = doubled // 2
    if doubled % 2 or width % group_size:
        raise ValueError(
            f"row width {doubled} is not two halves of a multiple of {group_size}"
        )
    num_groups = width // group_size
    q = torch.empty((rows, width), device=x.device, dtype=torch.float8_e4m3fn)
    q_scale = (
        torch.empty((rows, -(-num_groups // 4)), device=x.device, dtype=torch.int32)
        if scale_ue8m0
        else torch.empty((rows, num_groups), device=x.device, dtype=torch.float32)
    )
    if rows == 0:
        return q, q_scale
    block_groups = 8
    _silu_mul_quant_fp8_kernel[(rows, triton.cdiv(num_groups, block_groups))](
        q,
        q_scale,
        x,
        width,
        num_groups,
        x.stride(0),
        q.stride(0),
        q_scale.stride(0),
        GROUP_SIZE=group_size,
        BLOCK_GROUPS=block_groups,
        SCALE_UE8M0=scale_ue8m0,
        num_warps=4,
    )
    return q, q_scale
