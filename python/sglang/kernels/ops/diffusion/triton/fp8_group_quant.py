# SPDX-License-Identifier: Apache-2.0
"""sglang's per-token-group FP8 quantizer, as a step inside a fused kernel.

A kernel that produces a GEMM's input can quantize it on the way out instead of
writing bf16 rows for ``sglang_per_token_group_quant_fp8`` to read back, as
long as it quantizes them exactly as that kernel does, which takes more than
the same arithmetic: ``per_token_group_quant.cuh`` is built with
``--use_fast_math``, where nvcc compiles ``448.f / amax`` to
``div.approx.ftz.f32`` and Triton's ``/`` lowers to ``div.full.f32``, a
different approximation. What it computes per group is an exact absmax floored
at 1e-10, ``amax * (1 / 448)`` as the stored scale, ``div.approx.ftz`` for its
inverse, one fp32 multiply, a clamp at +448 and a saturating RNE cast.
Flush-to-zero, which that build has and this one does not, only touches
products far below e4m3's smallest subnormal, which cast to the same signed
zero either way.

Its UE8M0 flavour, what DeepGEMM takes on SM100 and SM120, keeps the absmax,
the floor and ``amax * (1 / 448)``, then rounds that scale up to a power of two
by its bits -- the biased exponent, plus one unless the mantissa is zero -- and
multiplies by the exact inverse power of two, so no division is left to
approximate. The exponent bytes go four groups to an int32, the first group in
the low byte, and bytes past the last group are zero.
"""

import triton
import triton.language as tl

from sglang.kernels.ops.diffusion.triton.numerics import div_approx_ftz_f32

# The constants sglang's per-token-group FP8 quantizer bakes in; a fused kernel
# has to agree with it bit for bit, so they are its, not ours.
FP8_MAX = tl.constexpr(448.0)
FP8_MAX_INV = tl.constexpr(1.0 / 448.0)
QUANT_EPS = tl.constexpr(1e-10)


@triton.jit
def quantize_fp8_groups(values):
    """``[groups, group_size]`` fp32 holding bf16 values -> (clamped, scales).

    Cast the first to the FP8 dtype on store; the second is the per-group
    scale the GEMM takes.
    """
    amax = tl.maximum(tl.max(tl.abs(values), axis=1), QUANT_EPS)
    inverse = div_approx_ftz_f32(tl.full(amax.shape, FP8_MAX, tl.float32), amax)
    return tl.minimum(values * inverse[:, None], FP8_MAX), amax * FP8_MAX_INV


@triton.jit
def quantize_fp8_groups_ue8m0(values):
    """``[groups, group_size]`` fp32 holding bf16 values -> (clamped, exponents).

    Cast the first to the FP8 dtype on store; the second is each group's
    biased UE8M0 exponent, for ``pack_ue8m0``.
    """
    amax = tl.maximum(tl.max(tl.abs(values), axis=1), QUANT_EPS)
    bits = (amax * FP8_MAX_INV).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    return tl.minimum(values * inverse[:, None], FP8_MAX), exponent


@triton.jit
def pack_ue8m0(exponent, valid, GROUPS: tl.constexpr):
    """``[GROUPS]`` exponents -> ``[GROUPS // 4]`` int32, zero past ``valid``."""
    exponent = tl.reshape(tl.where(valid, exponent, 0), [GROUPS // 4, 4])
    return tl.sum(exponent << (tl.arange(0, 4) * 8)[None, :], axis=1)
