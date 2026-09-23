# SPDX-License-Identifier: Apache-2.0
"""FlashInfer's SM120 Sage block-sparse kernel, behind VSA-H3's tile contract.

The vendored Triton forward in ``kernels.py`` and this module compute the same
function from the same inputs: 64-token tiles, an explicit key-tile list per
(head, query tile) with a per-row count, and a live-token count per tile that
masks a partial tile's tail. What differs is where the work happens and what it
costs to get there.

**Why go outside.** Measured on one RTX 5090 at 116k rows, 28 rank-local heads
and sparsity 0.9, the Triton video launch runs at 325 TFLOPS against this
kernel's 515 -- 1.6x, flat across every head-slice width from 3 to 28. Against
the card's own fp8 GEMM ceiling (637 TFLOPS on a 8192-cube) that is 51% versus
81%, which is the gap a hand-written CuTe-DSL mainloop buys over Triton on
Blackwell's fp8 path. Added error against an fp32 reference over the same
block selection is 3.82% where the Triton path is 3.78%, flat as K's channel
bias grows, because this kernel centres K per tile the same way.

**What it costs.** This kernel takes contiguous BHSD and hands back contiguous
BHSD, where the Triton kernels read and write the packed rows in place. The
operands it needs are quantized straight from the packed rows by
``sm120_sage_quant``, FlashInfer's own quantizers with their loads pointed
through ``tile_rows``, and each launch's output is scattered straight back to
the packed rows, so what gets materialised is INT8 Q and K, FP8 V and one bf16
result per launch -- not the bf16 tile-ordered Q, K, V and output this path
first staged, which at 116k rows and 10 heads cost 2.35 ms a head slice to
write and read back.

**Installation.** The kernel landed in FlashInfer after 0.6.17; the import
below tries the released path first and the side-by-side directory an
unreleased checkout can be dropped into second. That second path is not
recorded in any ``dist-info``, so ``pip install -U flashinfer-python`` will
leave it in place and stale -- delete ``cute_dsl/sparse_sm120`` when upgrading
FlashInfer.
"""

from __future__ import annotations

import functools

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.vsa_h3.sm120_sage_quant import (
    quantize_sage_kv_packed,
    quantize_sage_q_packed,
    scatter_tile_rows,
)

BLOCK_SIZE = 64
# The blk64 family is built for one head width, and asserts it.
SUPPORTED_HEAD_DIM = 128


@functools.lru_cache(maxsize=1)
def _load_attention():
    """FlashInfer's SM120 Sage attention entry point, or ``None``.

    Two import roots: the released layout once FlashInfer ships the SM120 Sage
    backend, and the side-by-side copy described in the module docstring. The
    quantizers are this package's own, so only the attention kernel is needed.
    """
    roots = (
        "flashinfer.cute_dsl.sparse",
        "flashinfer.cute_dsl.sparse_sm120",
    )
    for root in roots:
        try:
            attn = __import__(
                f"{root}.bsa_attn_sm120", fromlist=["bsa_attn_sm120_blk64_sage_fwd"]
            )
        except (ImportError, OSError, AttributeError):
            continue
        try:
            return attn.bsa_attn_sm120_blk64_sage_fwd
        except AttributeError:
            continue
    return None


@functools.lru_cache(maxsize=1)
def _is_sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def sm120_sage_available(head_dim: int) -> bool:
    """Whether this call can run on the SM120 Sage kernel.

    Resolved per call rather than once at construction because the head width
    is the caller's and the device is only known on the worker.
    """
    return (
        head_dim == SUPPORTED_HEAD_DIM
        and _is_sm120()
        and _load_attention() is not None
    )


# Transient bytes per head, per padded row-element, at the peak of one pass,
# measured (6.95 to 6.98, flat from 2 heads to 14) when this path still staged
# bf16 Q, K, V and output in tile order. It over-counts now that the operands
# are quantized straight from the packed rows, and it is kept anyway: the head
# slice it sizes is also the batch of the pooled-score GEMM, whose last bits
# decide near-ties in top-k, so a smaller constant would slice differently and
# change the selection. Lower it together with a render check if the memory is
# wanted back.
_SAGE_BYTES_PER_ELEMENT = 7


def head_slice_bytes(padded_rows: int, head_dim: int) -> int:
    """Transient bytes per head, to size the head slice against.

    The score matrix and the index list are the caller's and are counted
    there, the same way.
    """
    return padded_rows * head_dim * _SAGE_BYTES_PER_ELEMENT


def sm120_sage_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    *,
    tile_rows: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    num_prefix_tiles: int,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """One head slice of block-sparse attention, on FlashInfer's SM120 Sage path.

    ``query``, ``key``, ``value`` and ``out`` are the caller's packed
    ``[S, H, D]`` rows for this slice. ``tile_rows`` maps each tiled slot to the
    packed row it holds, ``variable_block_sizes`` gives each tile its live token
    count, and ``q2k_index`` / ``q2k_num`` are the video launch's selection --
    exactly what ``block_sparse_attn_forward`` takes, so the selection code
    above this is shared between the two paths.

    Prefix query tiles are dense over every key tile and run as their own launch
    for the same reason the Triton path splits them: a single launch would have
    to size its index list to the widest row, which is the whole sequence, and
    at 1813 tiles and 28 heads that list alone is 368 MiB. Each launch
    quantizes only its own query slots and writes only its own tiles' rows.
    """
    attention = _load_attention()
    tiles = int(variable_block_sizes.numel())
    heads = query.shape[1]
    k_int8, v_fp8, k_scale, v_scale = quantize_sage_kv_packed(
        key, value, tile_rows=tile_rows, block_sizes=variable_block_sizes
    )

    def launch(
        first_tile: int, tile_count: int, index: torch.Tensor, counts: torch.Tensor
    ) -> None:
        q_int8, q_scale = quantize_sage_q_packed(
            query,
            tile_rows=tile_rows,
            block_sizes=variable_block_sizes,
            first_row=first_tile * BLOCK_SIZE,
            rows=tile_count * BLOCK_SIZE,
        )
        result = attention(
            q_int8,
            k_int8,
            v_fp8,
            q_scale,
            k_scale,
            v_scale,
            index.unsqueeze(0).contiguous(),
            int(index.shape[-1]),
            block_sizes=variable_block_sizes,
            q2k_block_nums=counts.unsqueeze(0).contiguous(),
            softmax_scale=float(softmax_scale),
            backend="cute_dsl",
        )
        scatter_tile_rows(
            result,
            out,
            tile_rows=tile_rows,
            block_sizes=variable_block_sizes,
            first_tile=first_tile,
        )

    if num_prefix_tiles:
        dense_index = torch.arange(
            tiles, device=query.device, dtype=torch.int32
        ).expand(heads, num_prefix_tiles, tiles)
        dense_num = torch.full(
            (heads, num_prefix_tiles), tiles, device=query.device, dtype=torch.int32
        )
        launch(0, num_prefix_tiles, dense_index, dense_num)
        del dense_index, dense_num

    launch(num_prefix_tiles, tiles - num_prefix_tiles, q2k_index, q2k_num)
    return out


__all__ = [
    "head_slice_bytes",
    "sm120_sage_attention",
    "sm120_sage_available",
]
