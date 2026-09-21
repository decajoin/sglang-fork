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

**What it costs.** The Triton kernels gather Q and V through ``tile_rows`` and
scatter the output back, so only K is ever laid out in tile order. This one
takes contiguous BHSD and hands back contiguous BHSD, so Q, K, V and the output
all have to be materialised: 6.6 ms of gather and 2.4 ms of scatter at the
shape above, against 23.8 ms saved in the launches. The gather is free in the
sense that matters -- permuting the rows into (4,4,4) cube order costs the same
as reading them sequentially (46.270 ms against 46.289 for the whole op), which
is what says the Triton kernel is compute-bound rather than starved -- but the
*buffers* are not free, and they are what the head slice has to be sized for.

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

BLOCK_SIZE = 64
# The blk64 family is built for one head width, and asserts it.
SUPPORTED_HEAD_DIM = 128


@functools.lru_cache(maxsize=1)
def _load_ops():
    """``(quantize_q, quantize_kv, attention)``, or ``None`` if unavailable.

    Two import roots: the released layout once FlashInfer ships the SM120 Sage
    backend, and the side-by-side copy described in the module docstring.
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
            quant = __import__(
                f"{root}.bsa_utils.sage_quant_sm120",
                fromlist=["quantize_sage_q_sm120", "quantize_sage_kv_sm120"],
            )
        except (ImportError, OSError, AttributeError):
            continue
        try:
            return (
                quant.quantize_sage_q_sm120,
                quant.quantize_sage_kv_sm120,
                attn.bsa_attn_sm120_blk64_sage_fwd,
            )
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
    return head_dim == SUPPORTED_HEAD_DIM and _is_sm120() and _load_ops() is not None


# Transient bytes per head, per padded row-element, at the peak of one pass.
# Measured rather than derived: 6.95 to 6.98, flat from 2 heads to 14, so the
# constant below is that rounded up. Deriving it from the buffer list
# over-counts badly, because the staging copies are freed inside the
# expressions that build them and the quantized operands are freed with the
# frame that holds them, so they never coexist the way a static reading of the
# code suggests. Re-measure it if the materialisation changes shape: reading it
# too high slices the heads more finely than the card needs, and every extra
# pass pays the fixed cost of a selection and two launches again.
_SAGE_BYTES_PER_ELEMENT = 7


def head_slice_bytes(padded_rows: int, head_dim: int) -> int:
    """Transient bytes per head, to size the head slice against.

    Four and a half times what the Triton path spends (9 bytes a row-element
    against 2), because that path materialises only K while this one needs Q,
    K, V and the output laid out in tile order. The score matrix and the index
    list are the caller's and are counted there, the same way.
    """
    return padded_rows * head_dim * _SAGE_BYTES_PER_ELEMENT


def _tile_bhsd(
    packed: torch.Tensor,
    slots: torch.Tensor,
    padded_rows: int,
) -> torch.Tensor:
    """Packed ``[S, H, D]`` rows -> contiguous ``[1, H, padded_rows, D]``.

    One pass, not two: scattering into the transposed buffer directly costs
    1.27 ms where staging a ``[padded, H, D]`` copy and transposing it costs
    2.33 (116k rows, 28 heads), because the second form writes the whole tensor
    twice.

    ``slots`` gives each packed row its slot within this buffer. The buffer is
    zeroed rather than left undefined: the kernel masks pad slots out of the
    softmax through ``block_sizes``, but the quantizer sees them first, and a
    garbage row would decide its tile's scale and cost every live row in that
    tile its precision -- the same reason ``quantize_tiles`` zeroes them.
    """
    buffer = packed.new_zeros((1, packed.shape[1], padded_rows, packed.shape[2]))
    buffer[0].index_copy_(1, slots, packed.transpose(0, 1))
    return buffer


def sm120_sage_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    *,
    scatter_index: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    num_prefix_tiles: int,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """One head slice of block-sparse attention, on FlashInfer's SM120 Sage path.

    ``query``, ``key``, ``value`` and ``out`` are the caller's packed
    ``[S, H, D]`` rows for this slice. ``scatter_index`` maps each packed row to
    its tiled slot, ``variable_block_sizes`` gives each tile its live token
    count, and ``q2k_index`` / ``q2k_num`` are the video launch's selection --
    exactly what ``block_sparse_attn_forward`` takes, so the selection code
    above this is shared between the two paths.

    Prefix query tiles are dense over every key tile and run as their own launch
    for the same reason the Triton path splits them: a single launch would have
    to size its index list to the widest row, which is the whole sequence, and
    at 1813 tiles and 28 heads that list alone is 368 MiB.
    """
    tiled_out = _attend_in_tile_order(
        query=query,
        key=key,
        value=value,
        scatter_index=scatter_index,
        variable_block_sizes=variable_block_sizes,
        num_prefix_tiles=num_prefix_tiles,
        q2k_index=q2k_index,
        q2k_num=q2k_num,
        softmax_scale=softmax_scale,
    )
    # ``scatter_index`` is packed row -> tiled slot, so indexing the tiled
    # output with it is the inverse permutation. Pad slots are never read, which
    # is why that buffer could be left uninitialised. Everything the launches
    # held is already freed here, which is what keeps the gather off the peak.
    out.copy_(tiled_out[scatter_index])
    return out


def _attend_in_tile_order(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scatter_index: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    num_prefix_tiles: int,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """The two launches, returning ``[padded_rows, H, D]`` in tile order.

    Split from the entry point so the quantized operands are locals of a frame
    that returns before the output is gathered back into packed order: they are
    the largest transients in the call, and holding them across that gather
    would raise the peak the head slice is sized against.
    """
    quantize_q, quantize_kv, attention = _load_ops()
    tiles = int(variable_block_sizes.numel())
    padded_rows = tiles * BLOCK_SIZE
    heads, dim = query.shape[1], query.shape[2]

    k_int8, v_fp8, k_scale, v_scale = quantize_kv(
        _tile_bhsd(key, scatter_index, padded_rows),
        _tile_bhsd(value, scatter_index, padded_rows),
    )

    # Each launch lays out only its own query rows, so neither pays for the
    # other's. That needs the prefix's packed rows to be a front block, which
    # they are for every geometry this backend builds -- prefix segments fill
    # the low tiles and the tiling preserves that order. It is checked rather
    # than assumed: the fallback costs one copy of Q, and a wrong split would
    # cost correctness.
    live_rows = int(scatter_index.numel())
    boundary = num_prefix_tiles * BLOCK_SIZE
    prefix_rows = int((scatter_index < boundary).sum()) if num_prefix_tiles else 0
    query_tiled = None
    if num_prefix_tiles and not bool((scatter_index[:prefix_rows] < boundary).all()):
        query_tiled = _tile_bhsd(query, scatter_index, padded_rows)

    tiled_out = query.new_empty((padded_rows, heads, dim))

    def launch(
        first_tile: int,
        tile_count: int,
        index: torch.Tensor,
        counts: torch.Tensor,
        row_lo: int,
        row_hi: int,
    ) -> None:
        first_row = first_tile * BLOCK_SIZE
        rows = tile_count * BLOCK_SIZE
        if query_tiled is not None:
            q_bhsd = query_tiled[:, :, first_row : first_row + rows].contiguous()
        else:
            q_bhsd = _tile_bhsd(
                query[row_lo:row_hi],
                scatter_index[row_lo:row_hi] - first_row,
                rows,
            )
        q_int8, q_scale = quantize_q(q_bhsd)
        del q_bhsd
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
        tiled_out[first_row : first_row + rows] = result[0].transpose(0, 1)

    if num_prefix_tiles:
        dense_index = torch.arange(
            tiles, device=query.device, dtype=torch.int32
        ).expand(heads, num_prefix_tiles, tiles)
        dense_num = torch.full(
            (heads, num_prefix_tiles), tiles, device=query.device, dtype=torch.int32
        )
        launch(0, num_prefix_tiles, dense_index, dense_num, 0, prefix_rows)
        del dense_index, dense_num

    launch(
        num_prefix_tiles,
        tiles - num_prefix_tiles,
        q2k_index,
        q2k_num,
        prefix_rows,
        live_rows,
    )
    return tiled_out


__all__ = [
    "head_slice_bytes",
    "sm120_sage_attention",
    "sm120_sage_available",
]
