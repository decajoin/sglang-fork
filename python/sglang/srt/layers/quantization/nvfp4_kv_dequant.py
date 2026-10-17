"""Direct NVFP4 KV to FP8 conversion without a BF16 staging tensor."""

import torch
import triton
import triton.language as tl


@triton.jit
def _dequantize_fp8(
    packed,
    scales,
    global_scale,
    output,
    count,
    H: tl.constexpr,
    D: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < count
    d = i % D
    h = (i // D) % H
    b = i // (D * H)
    byte = tl.load(packed + b * Q0 + h * Q1 + (d // 2) * Q2, valid, 0)
    code = (byte >> ((d % 2) * 4)) & 15
    magnitude = code & 7
    value = tl.where(
        magnitude <= 4,
        magnitude.to(tl.float32) * 0.5,
        tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)),
    )
    value = tl.where((code & 8) != 0, -value, value)
    scale = tl.load(scales + b * S0 + h * S1 + (d // 16) * S2, valid, 0.0)
    value = value * scale.to(tl.float32) * tl.load(global_scale)
    tl.store(output + i, value, valid)


def dequantize_nvfp4_to_fp8(packed, block_scales, global_scale):
    """Convert [tokens, heads, dim/2] packed E2M1 and linear E4M3 scales.

    Only the FP8 output is allocated. Unpacking and scaling use registers;
    strided input views do not require contiguous staging copies.
    """
    if packed.ndim != 3 or packed.dtype != torch.uint8:
        raise ValueError("NVFP4 data must be a three-dimensional uint8 tensor")
    b, h, half_d = packed.shape
    d = half_d * 2
    if d == 0 or h == 0 or d % 16 or block_scales.shape != (b, h, d // 16):
        raise ValueError("NVFP4 scales must have one entry per 16 head elements")
    if block_scales.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("NVFP4 block scales must contain E4M3 bytes")
    if (
        not packed.is_cuda
        or block_scales.device != packed.device
        or global_scale.device != packed.device
        or global_scale.numel() != 1
        or global_scale.dtype != torch.float32
    ):
        raise ValueError("NVFP4 inputs and scalar FP32 global scale must share a GPU")
    scales = block_scales.view(torch.float8_e4m3fn)
    output = torch.empty((b, h, d), device=packed.device, dtype=torch.float8_e4m3fn)
    if output.numel():
        _dequantize_fp8[(triton.cdiv(output.numel(), 1024),)](
            packed,
            scales,
            global_scale,
            output,
            output.numel(),
            h,
            d,
            *packed.stride(),
            *scales.stride(),
            BLOCK=1024,
            enable_fp_fusion=False,
        )
    return output
