# SPDX-License-Identifier: Apache-2.0
# The block-sparse forward is copied and adapted from:
# https://github.com/hao-ai-lab/FastVideo (fastvideo_kernel/triton_kernels/
# block_sparse_attn_triton.py, itself derived from the OpenAI Triton
# FlashAttention-2 tutorial). The INT8 quantization of Q.K follows
# SageAttention (https://github.com/thu-ml/SageAttention).
"""64x64 block-sparse attention over a re-ordered sequence, in Triton.

Vendored rather than taken from a package so the VSA-H3 backend has no
external dependency: the upstream wheel (``fastvideo-kernel``) is pinned to one
Torch minor and carries compiled CUDA extensions this backend does not use.
Forward only -- this tree is inference-only, and the autograd wrapper, the two
backward kernels and the sm_90/sm_100a CUDA routes that surround it upstream
all serve training or hardware we do not target.

The unit is a 64-token tile. Which key tiles a query tile attends to is given
per (head, query tile) as an explicit index list plus a count, so the kernels
never see the selection rule -- ``video_sparse_attn_h3.py`` owns that. Tiles may
be partially filled: ``variable_block_sizes`` gives each tile's live token
count and everything past it is masked, which is what lets a packed sequence
whose segments are not multiples of 64 be tiled at all.

Two things separate this from the upstream kernel, and both exist because the
sequence VSA-H3 tiles is 100k+ rows on a 32 GiB card:

**Nothing is materialised in tile order except K.** Upstream permutes Q, K and
V into padded tile buffers and permutes the output back, which is four
full-width copies per attention call -- 3.2 GiB at 28 rank-local heads and a
116k-row sequence, on top of an activation footprint that already fills the
card. Here ``tile_rows`` maps each tiled slot to its packed row and the kernels
gather Q and V through it and scatter the output back, so only K -- read by
every query tile, and read transposed -- is worth laying out contiguously.

**Q.K can run in INT8.** ``quantize_tiles`` writes K as int8 with one scale per
tile; the attention kernel quantizes each Q tile in registers and runs the
first GEMM on INT8 tensor cores, which is ~1.9x the whole call on an RTX 5090.
P.V stays bf16. Subtracting K's per-channel mean before quantizing costs
nothing and is exact: it shifts every logit in a row by the same ``-q.km``,
which softmax cancels, and it is what keeps the int8 range useful when K has a
large channel bias. Measured against the bf16 path, the added error is 1.3% of
the output's norm at any sparsity -- SageAttention's own budget, and the same
error the dense fallback (``sage_attn``) already carries.
"""

import math

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 64

# num_stages / num_warps are free and re-tuned per architecture by autotune.
# BLOCK_M/BLOCK_N are structural: the index list addresses keys as
# ``kv_idx * BLOCK_N``, so both must match the tile size the caller built.
_CONFIGS = [
    triton.Config(
        {"BLOCK_M": BLOCK_SIZE, "BLOCK_N": BLOCK_SIZE}, num_stages=s, num_warps=w
    )
    for s in (2, 3, 4, 5, 6, 7)
    for w in (4, 8)
]


@triton.jit
def _pool_tiles(
    X,
    tile_rows,
    variable_block_sizes,
    Out,
    stride_xs,
    stride_xh,
    stride_oh,
    stride_ot,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Mean of each tile's live rows, in fp32. One program per (tile, head)."""
    tile = tl.program_id(0)
    head = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)
    size = tl.load(variable_block_sizes + tile)
    valid = offs < size
    rows = tl.load(tile_rows + tile * BLOCK + offs, mask=valid, other=0).to(tl.int64)
    x = tl.load(
        X + rows[:, None] * stride_xs + head * stride_xh + offs_d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    tl.store(Out + head * stride_oh + tile * stride_ot + offs_d, tl.sum(x, 0) / size)


@triton.jit
def _quantize_tiles(
    X,
    tile_rows,
    variable_block_sizes,
    Mean,
    Out,
    Scale,
    stride_xs,
    stride_xh,
    stride_oh,
    stride_os,
    stride_sh,
    FIXED_SCALE: tl.constexpr,
    SMOOTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather one tile, optionally de-mean it, quantize to int8 with one scale.

    Pad slots are written as zero rather than left undefined: they are masked
    out of the softmax anyway, but a garbage row would otherwise decide the
    tile's amax and cost every live row its precision.
    """
    tile = tl.program_id(0)
    head = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)
    size = tl.load(variable_block_sizes + tile)
    valid = offs < size
    rows = tl.load(tile_rows + tile * BLOCK + offs, mask=valid, other=0).to(tl.int64)
    x = tl.load(
        X + rows[:, None] * stride_xs + head * stride_xh + offs_d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    if SMOOTH:
        mean = tl.load(Mean + head * HEAD_DIM + offs_d)
        x = tl.where(valid[:, None], x - mean[None, :], 0.0)
    if FIXED_SCALE:
        # One scale per head, supplied by the caller: e4m3 carries its own
        # exponent, so a shared scale costs no mantissa precision, and a
        # loop-invariant scale is what keeps the P.V accumulation fused.
        scale = tl.load(Scale + head)
        tl.store(
            Out
            + head * stride_oh
            + (tile * BLOCK + offs)[:, None] * stride_os
            + offs_d[None, :],
            (x / scale).to(Out.type.element_ty),
        )
    else:
        scale = tl.max(tl.abs(x)) / 127.0
        scale = tl.where(scale > 0, scale, 1.0)
        tl.store(
            Out
            + head * stride_oh
            + (tile * BLOCK + offs)[:, None] * stride_os
            + offs_d[None, :],
            tl.extra.cuda.libdevice.round(x / scale).to(tl.int8),
        )
        tl.store(Scale + head * stride_sh + tile, scale)


# The largest finite e4m3 value. P is in (0, 1], so scaling it here costs
# nothing and moves the small end of the distribution out of the subnormal
# range instead of flushing it to zero.
_FP8_MAX = tl.constexpr(448.0)


@triton.autotune(_CONFIGS, key=["N_CTX_KV", "HEAD_DIM", "QUANT", "QUANT_PV"])
@triton.jit
def _attn_fwd_sparse(
    Q,
    K,
    K_SCALE,
    V,
    V_SCALE,
    V_MEAN,
    Out,
    sm_scale,
    tile_rows,
    q2k_index,
    q2k_num,
    max_kv_blks,
    variable_block_sizes,
    stride_qs,
    stride_qh,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_sh,
    stride_vs,
    stride_vh,
    stride_os,
    stride_oh,
    N_CTX_KV,
    Q_TILE_OFFSET,
    N_Q_TILES,
    QUANT: tl.constexpr,
    QUANT_PV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_blk = tl.program_id(0)
    head = tl.program_id(1)
    # The caller runs prefix and video query tiles as separate launches, so the
    # index metadata is addressed from this launch's first tile while the tile
    # geometry is addressed absolutely.
    tile = q_blk + Q_TILE_OFFSET
    meta_base = head * N_Q_TILES + q_blk
    kv_blocks = tl.load(q2k_num + meta_base)
    kv_ptr = q2k_index + meta_base * max_kv_blks

    offs = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_size = tl.load(variable_block_sizes + tile)
    q_valid = offs < q_size
    q_rows = tl.load(tile_rows + tile * BLOCK_M + offs, mask=q_valid, other=0).to(
        tl.int64
    )
    q_ptr = Q + q_rows[:, None] * stride_qs + head * stride_qh + offs_d[None, :]
    if QUANT:
        q_f32 = tl.load(q_ptr, mask=q_valid[:, None], other=0.0).to(tl.float32)
        q_scale = tl.max(tl.abs(q_f32)) / 127.0
        q_scale = tl.where(q_scale > 0, q_scale, 1.0)
        q = tl.extra.cuda.libdevice.round(q_f32 / q_scale).to(tl.int8)
    else:
        q = tl.load(q_ptr, mask=q_valid[:, None], other=0.0)
        q_scale = 1.0

    # K is the one tensor laid out in tile order: every query tile reads it,
    # transposed, so a gather in the inner loop would be paid per (query tile,
    # key tile) instead of once.
    K_base = tl.make_block_ptr(
        base=K + head.to(tl.int64) * stride_kh,
        shape=(HEAD_DIM, N_CTX_KV),
        strides=(stride_kd, stride_ks),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    if QUANT_PV:
        # The fp8 V is in tile order too: quantizing it once beats converting a
        # gathered bf16 tile in registers on every (query tile, key tile) pair,
        # which is what made a first FP8 attempt slower than bf16 P.V.
        V_base = tl.make_block_ptr(
            base=V + head.to(tl.int64) * stride_vh,
            shape=(N_CTX_KV, HEAD_DIM),
            strides=(stride_vs, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # 1/ln2
    if QUANT_PV:
        # One scale per head rather than per tile: e4m3 carries its own
        # exponent, so a shared scale costs no mantissa precision as long as
        # nothing overflows, and a loop-invariant scale keeps the accumulation
        # fused into one tl.dot chain.
        v_scale = tl.load(V_SCALE + head)
    else:
        v_scale = 1.0

    for i in range(0, kv_blocks):
        kv_idx = tl.load(kv_ptr + i).to(tl.int32)
        block_size = tl.load(variable_block_sizes + kv_idx)
        k = tl.load(tl.advance(K_base, (0, kv_idx * BLOCK_N)))
        if QUANT:
            k_scale = tl.load(K_SCALE + head * stride_sh + kv_idx)
            qk = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32) * (q_scale * k_scale)
        else:
            qk = tl.dot(q, k)
        # Columns past the tile's live token count are padding.
        live = offs_n < block_size
        qk = tl.where(live[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        if QUANT_PV:
            acc = tl.dot(
                (p * _FP8_MAX).to(tl.float8e4nv),
                tl.load(tl.advance(V_base, (kv_idx * BLOCK_N, 0))),
                acc,
            )
        else:
            v_rows = tl.load(
                tile_rows + kv_idx * BLOCK_N + offs_n, mask=live, other=0
            ).to(tl.int64)
            v = tl.load(
                V + v_rows[:, None] * stride_vs + head * stride_vh + offs_d[None, :],
                mask=live[:, None],
                other=0.0,
            )
            acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    if QUANT_PV:
        acc = acc * (v_scale / _FP8_MAX)
    acc = acc / l_i[:, None]
    if QUANT_PV:
        # V was centred before quantizing. The weights sum to one after the
        # division above, so adding the mean back here is exact -- and it is
        # what stops e4m3's three mantissa bits from being spent on a channel
        # bias that every row shares.
        acc = acc + tl.load(V_MEAN + head * HEAD_DIM + offs_d)[None, :]
    tl.store(
        Out + q_rows[:, None] * stride_os + head * stride_oh + offs_d[None, :],
        acc.to(Out.type.element_ty),
        mask=q_valid[:, None],
    )


def pool_tiles(
    x: torch.Tensor,
    tile_rows: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    num_tiles: int,
) -> torch.Tensor:
    """Tile means of packed ``[S, H, D]`` rows -> ``[H, num_tiles, D]`` fp32."""
    heads, dim = x.shape[1], x.shape[2]
    out = torch.empty((heads, num_tiles, dim), dtype=torch.float32, device=x.device)
    _pool_tiles[(num_tiles, heads, 1)](
        x,
        tile_rows,
        variable_block_sizes,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        HEAD_DIM=dim,
        BLOCK=BLOCK_SIZE,
    )
    return out


def quantize_tiles(
    x: torch.Tensor,
    tile_rows: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    num_tiles: int,
    channel_mean: torch.Tensor | None = None,
    fixed_scale: torch.Tensor | None = None,
    dtype: torch.dtype = torch.int8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed ``[S, H, D]`` rows -> quantized ``[H, num_tiles * 64, D]``.

    With ``fixed_scale`` ``[H]`` the caller's scale is used as is and returned
    unchanged -- what the fp8 V buffer wants, because a loop-invariant scale is
    what keeps the P.V accumulation fused into one ``tl.dot`` chain. Otherwise
    each tile gets its own int8 scale, which is what K wants.

    ``channel_mean`` is the per-head, per-channel mean over the live rows. It is
    subtracted before quantizing; for K softmax cancels the resulting constant
    shift exactly, and it is what makes int8 usable on a K with a channel bias.
    """
    heads, dim = x.shape[1], x.shape[2]
    out = torch.empty(
        (heads, num_tiles * BLOCK_SIZE, dim), dtype=dtype, device=x.device
    )
    scale = (
        fixed_scale
        if fixed_scale is not None
        else torch.empty((heads, num_tiles), dtype=torch.float32, device=x.device)
    )
    _quantize_tiles[(num_tiles, heads, 1)](
        x,
        tile_rows,
        variable_block_sizes,
        x if channel_mean is None else channel_mean,
        out,
        scale,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        scale.stride(0),
        FIXED_SCALE=fixed_scale is not None,
        SMOOTH=channel_mean is not None,
        HEAD_DIM=dim,
        BLOCK=BLOCK_SIZE,
    )
    return out, scale


def block_sparse_attn_forward(
    query: torch.Tensor,
    key_tiled: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    tile_rows: torch.Tensor,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    q_tile_offset: int,
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
    value_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Block-sparse attention for one contiguous run of query tiles.

    ``query``, ``value`` and ``out`` are packed ``[S, H, D]`` rows, addressed
    through ``tile_rows`` (``[num_tiles * 64]`` int32: the packed row each tiled
    slot holds, 0 and masked for the pad slots of a partial tile).
    ``key_tiled`` is ``[H, num_tiles * 64, D]``, bf16 or int8; int8 requires
    ``key_scale`` ``[H, num_tiles]``. ``value`` is the packed rows unless
    ``value_scale`` ``[H]`` is given, in which case it is an fp8 ``[H, num_tiles
    * 64, D]`` buffer holding ``(V - value_mean) / value_scale`` and P.V runs on
    FP8 tensor cores, with ``value_mean`` ``[H, D]`` added back after the
    softmax normalisation.

    ``q2k_index`` is ``[H, q_tiles, W]`` int32 -- for each query tile of *this
    launch*, the key tiles it attends to in the first ``q2k_num[h, i]`` slots.
    ``q_tile_offset`` is where those tiles start in the absolute tile numbering.
    Only the rows of the covered tiles are written, so two launches can fill one
    output.
    """
    heads, kv_len, dim = key_tiled.shape
    q_tiles = q2k_num.shape[-1]
    quantized = key_tiled.dtype is torch.int8
    quantized_pv = value_scale is not None
    if quantized and key_scale is None:
        raise ValueError("an int8 key buffer needs its per-tile scales")
    if quantized_pv and value_mean is None:
        raise ValueError("an fp8 value buffer needs the channel mean it was centred on")
    if kv_len % BLOCK_SIZE:
        raise ValueError(f"key buffer must be tile-aligned, got {kv_len} rows")
    if variable_block_sizes.numel() != kv_len // BLOCK_SIZE:
        raise ValueError(
            f"variable_block_sizes has {variable_block_sizes.numel()} entries, "
            f"the key buffer has {kv_len // BLOCK_SIZE} tiles"
        )
    if q_tile_offset + q_tiles > kv_len // BLOCK_SIZE:
        raise ValueError(
            f"query tiles [{q_tile_offset}, {q_tile_offset + q_tiles}) escape the "
            f"{kv_len // BLOCK_SIZE}-tile sequence"
        )

    _attn_fwd_sparse[(q_tiles, heads, 1)](
        query,
        key_tiled,
        key_scale if quantized else key_tiled,
        value,
        value_scale if quantized_pv else value,
        value_mean if quantized_pv else value,
        out,
        1.0 / math.sqrt(dim),
        tile_rows,
        q2k_index,
        q2k_num,
        q2k_index.shape[-1],
        variable_block_sizes,
        query.stride(0),
        query.stride(1),
        key_tiled.stride(0),
        key_tiled.stride(1),
        key_tiled.stride(2),
        key_scale.stride(0) if quantized else 0,
        # The two V layouts put the row and head axes the other way round:
        # packed rows are [S, H, D], the fp8 tile buffer is [H, S_padded, D].
        value.stride(1) if quantized_pv else value.stride(0),
        value.stride(0) if quantized_pv else value.stride(1),
        out.stride(0),
        out.stride(1),
        kv_len,
        q_tile_offset,
        q_tiles,
        QUANT=quantized,
        QUANT_PV=quantized_pv,
        HEAD_DIM=dim,
    )
    return out
