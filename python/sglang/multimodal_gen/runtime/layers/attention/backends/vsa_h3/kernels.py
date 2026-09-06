# SPDX-License-Identifier: Apache-2.0
# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# (fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py, itself derived
# from the OpenAI Triton FlashAttention-2 tutorial).
"""64x64 block-sparse attention forward, in Triton.

Vendored rather than taken from a package so the VSA-H3 backend has no
external dependency: the upstream wheel (``fastvideo-kernel``) is pinned to one
Torch minor and carries compiled CUDA extensions this backend does not use.
Only the forward kernel is here -- this tree is inference-only, and the
autograd wrapper, the two backward kernels and the sm_90/sm_100a CUDA routes
that surround it upstream all serve training or hardware we do not target.

The unit is a 64-token tile. Which key tiles a query tile attends to is given
per (batch, head, query tile) as an explicit index list plus a count, so the
kernel never sees the selection rule -- ``video_sparse_attn_h3.py`` owns that.
Tiles may be partially filled: ``variable_block_sizes`` gives each key tile's
live token count and the kernel masks the rest to -inf, which is what lets a
packed sequence whose segments are not multiples of 64 be tiled at all.

``BLOCK_M``/``BLOCK_N`` are fixed at 64 because they are structural, not
tunable: the index list addresses keys as ``kv_idx * BLOCK_N``, so both must
match the granularity the caller built its tiles at.
"""

import math

import torch
import triton
import triton.language as tl

# num_stages / num_warps are free and re-tuned per architecture by autotune.
_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=s, num_warps=w)
    for s in (2, 3, 4, 5, 6, 7)
    for w in (4, 8)
]


@triton.autotune(_CONFIGS, key=["N_CTX_Q", "N_CTX_KV", "HEAD_DIM"])
@triton.jit
def _attn_fwd_sparse(
    Q,
    K,
    V,
    sm_scale,
    q2k_index,
    q2k_num,
    max_kv_blks,
    variable_block_sizes,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX_Q,
    N_CTX_KV,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_blk = tl.program_id(0)  # query tile
    off_hz = tl.program_id(1)  # fused (batch, head)
    b = off_hz // H
    h = off_hz % H
    q_tiles = N_CTX_Q // BLOCK_M
    meta_base = (b * H + h) * q_tiles + q_blk

    kv_blocks = tl.load(q2k_num + meta_base)
    kv_ptr = q2k_index + meta_base * max_kv_blks

    # Q and KV can have different lengths, so their per-(batch, head) strides
    # differ and each base offset is computed on its own.
    q_off = b.to(tl.int64) * stride_qz + h.to(tl.int64) * stride_qh
    k_off = b.to(tl.int64) * stride_kz + h.to(tl.int64) * stride_kh
    v_off = b.to(tl.int64) * stride_vz + h.to(tl.int64) * stride_vh
    o_off = b.to(tl.int64) * stride_oz + h.to(tl.int64) * stride_oh

    Q_ptr = tl.make_block_ptr(
        base=Q + q_off,
        shape=(N_CTX_Q, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(q_blk * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_base = tl.make_block_ptr(
        base=K + k_off,
        shape=(HEAD_DIM, N_CTX_KV),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_base = tl.make_block_ptr(
        base=V + v_off,
        shape=(N_CTX_KV, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_ptr = tl.make_block_ptr(
        base=Out + o_off,
        shape=(N_CTX_Q, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(q_blk * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    offs_m = q_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # 1/ln2
    q = tl.load(Q_ptr)

    for i in range(0, kv_blocks):
        kv_idx = tl.load(kv_ptr + i).to(tl.int32)
        block_size = tl.load(variable_block_sizes + kv_idx)
        K_ptr = tl.advance(K_base, (0, kv_idx * BLOCK_N))
        V_ptr = tl.advance(V_base, (kv_idx * BLOCK_N, 0))

        k = tl.load(K_ptr)
        qk = tl.dot(q, k)
        # Columns past the tile's live token count are padding.
        mask = tl.arange(0, BLOCK_N) < block_size
        qk = tl.where(mask[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(V_ptr)
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    tl.store(M + off_hz * N_CTX_Q + offs_m, m_i)
    tl.store(O_ptr, acc.to(Out.type.element_ty))


BLOCK_SIZE = 64


def block_sparse_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> torch.Tensor:
    """Block-sparse attention over 64-token tiles.

    q: ``[B, H, Tq, D]``, k/v: ``[B, H, Tkv, D]``, both tile-aligned. ``q``
    may cover a suffix of the query tiles (the caller runs prefix and video
    query tiles as separate launches); the index metadata is then indexed from
    that slice's first tile, which is why ``q2k_num`` fixes ``Tq``.

    ``q2k_index`` is ``[B, H, Tq // 64, W]`` int32 -- for each query tile, the
    key tiles it attends to, in the first ``q2k_num[b, h, i]`` slots. Entries
    past the count are never read. ``variable_block_sizes`` is ``[Tkv // 64]``
    int32, the live token count of every key tile.
    """
    batch, heads, q_len, head_dim = q.shape
    kv_len = k.shape[2]
    if q_len % BLOCK_SIZE or kv_len % BLOCK_SIZE:
        raise ValueError(
            f"block-sparse attention needs tile-aligned lengths, got q={q_len}, kv={kv_len}"
        )
    if q2k_num.shape[-1] != q_len // BLOCK_SIZE:
        raise ValueError(
            f"q2k_num covers {q2k_num.shape[-1]} query tiles, q has {q_len // BLOCK_SIZE}"
        )
    if variable_block_sizes.numel() != kv_len // BLOCK_SIZE:
        raise ValueError(
            f"variable_block_sizes has {variable_block_sizes.numel()} entries, "
            f"kv has {kv_len // BLOCK_SIZE} tiles"
        )

    out = torch.empty_like(q)
    # The kernel writes the softmax log-sum-exp unconditionally; nothing here
    # reads it (there is no backward), but it still needs somewhere to land.
    lse = torch.empty((batch, heads, q_len), dtype=torch.float32, device=q.device)

    grid = lambda _: (triton.cdiv(q_len, BLOCK_SIZE), batch * heads, 1)  # noqa: E731
    _attn_fwd_sparse[grid](
        q,
        k,
        v,
        1.0 / math.sqrt(head_dim),
        q2k_index,
        q2k_num,
        q2k_index.shape[-1],
        variable_block_sizes,
        lse,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        batch,
        heads,
        q_len,
        kv_len,
        HEAD_DIM=head_dim,
    )
    return out
