# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0


import torch
from sageattention import sageattn

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (  # FlashAttentionMetadata,
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# How many heads one `sageattn` call covers. 0 runs every head in one call,
# which is what this backend has always done and stays the default.
#
# Every transient the kernel builds scales with heads x sequence: `per_warp_int8`
# returns int8 copies of Q and K, `per_channel_fp8` builds a bf16 transposed
# copy of V before quantizing it to fp8, and the kernel allocates its own bf16
# output. At 28 rank-local heads, head_dim 128 and a 174k-row packed sequence
# one full-width bf16 tensor is 1.16 GiB, and those five come to 4.07 GiB live
# at the peak -- for a dense op whose result is one of them.
#
# Slicing the head axis scales all of it by chunk/heads, leaving the output
# tensor as the only full-width allocation. Measured on one RTX 5090 at that
# shape, through `forward_varlen` with the MiniMax-H3 packed layout:
#
#     whole   4.07 GiB   764 ms
#     chunk=7 2.18 GiB   778 ms
#     chunk=4 1.75 GiB   781 ms
#     chunk=2 1.46 GiB   795 ms
#     chunk=1 1.31 GiB   801 ms
#
# Attention is head-parallel and so is every statistic the quantizer derives:
# Q and K scales are per (head, block), V's scale and mean are per
# (head, channel), and K's smoothing mean reduces along the sequence. A slice
# therefore sees exactly the values it would have seen in the whole-width call,
# so this is exact, not an approximation -- every width above reproduces the
# whole-width output under `torch.equal`, which the unit tests assert.
#
# Unlike the sparse path in `sparge_attn.py`, the slices are written into one
# preallocated output instead of concatenated, and passed as views instead of
# being made contiguous. Both matter here and not there: that path's map helper
# and kernel copy their inputs anyway and its transients dwarf one extra
# full-width tensor, while this one has neither excuse -- concatenating costs
# 1.16 GiB and materializing the slices costs three copies each, together
# enough to take the saving back.
DEFAULT_HEAD_CHUNK = 0


def _head_chunk_from_server_args() -> int:
    """Slice width from ``--attention-backend-config``.

    Shared with ``sparge_attn``: that backend builds this one as its dense
    fallback, and a deployment that asked for sliced sparse attention wants its
    dense steps -- the warmup and tail cutoffs -- bounded the same way.
    """
    from sglang.multimodal_gen.runtime.server_args import get_global_server_args

    try:
        config = get_global_server_args().attention_backend_config or {}
    except ValueError:
        # Unlike sparge_attn, this backend is also constructed directly --
        # outside a server there is no global to read, and that is not an
        # error. Slicing is opt-in, so no configuration means the whole-width
        # path this backend has always taken.
        return DEFAULT_HEAD_CHUNK
    head_chunk = int(config.get("head_chunk", DEFAULT_HEAD_CHUNK))
    if head_chunk < 0:
        raise ValueError(
            "sage head_chunk must be non-negative (0 runs every head in one "
            f"call), got {head_chunk}"
        )
    return head_chunk


def _trailing_padding_used_len(
    *,
    total_tokens: int,
    max_seqlen: int,
    bounds: tuple[int, ...],
) -> int | None:
    """Return live token count for H3-style [0, used, total] trailing padding."""
    if len(bounds) != 3:
        return None
    start, used, total = bounds
    if start != 0 or used >= total or total != total_tokens or used != max_seqlen:
        return None
    return used


class SageAttentionBackend(AttentionBackend):

    @classmethod
    def supports_ring_rotation(cls) -> bool:
        return True

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SAGE_ATTN

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl


class SageAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)
        # Tests and benchmarks pass the width directly; serving resolves it
        # once here rather than reaching for server args on every forward.
        head_chunk = extra_impl_args.get("head_chunk")
        self.head_chunk = (
            _head_chunk_from_server_args() if head_chunk is None else int(head_chunk)
        )
        if self.head_chunk > 0:
            logger.info_once(
                f"Sage attention: running {self.head_chunk} heads at a time"
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        output = sageattn(
            query,
            key,
            value,
            # since input is (batch_size, seq_len, head_num, head_dim)
            tensor_layout="NHD",
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
            return_lse=return_softmax_lse,
        )
        if return_softmax_lse:
            output, softmax_lse = output
            return output, softmax_lse
        return output

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(x) for x in cu_seqlens.tolist())
        )
        if self._slices_heads(query.shape[-2]):
            # The sliced path needs no contiguous input -- see `_attend_into`
            # -- and a full-width copy here would allocate exactly what the
            # slicing exists to avoid.
            return self._sage_packed(
                query,
                key,
                value,
                bounds=bounds,
                max_seqlen=max_seqlen,
            )
        return self._sage_packed(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            bounds=bounds,
            max_seqlen=max_seqlen,
        )

    def _slices_heads(self, heads: int) -> bool:
        return 0 < self.head_chunk < heads

    def _attend_into(
        self,
        *,
        out: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Dense attention over ``[S, H, D]`` rows, written into ``out``.

        Sliced along the head axis when ``head_chunk`` is set. Attention is
        head-parallel and every quantization statistic is per-head, so a slice
        computes the same numbers the whole-width call would have -- see
        ``DEFAULT_HEAD_CHUNK``. Slices land directly in ``out`` so no full-width
        intermediate is ever live.
        """
        heads = query.shape[-2]
        if not self._slices_heads(heads):
            out.copy_(
                self.forward(
                    query.unsqueeze(0),
                    key.unsqueeze(0),
                    value.unsqueeze(0),
                    None,
                )[0]
            )
            return
        for head in range(0, heads, self.head_chunk):
            sl = slice(head, head + self.head_chunk)
            # Passed as views. A head slice of NHD rows already has
            # ``stride(-1) == 1``, which is all `sageattn` requires, and it
            # quantizes into its own buffers from there -- so unlike the sparse
            # path in `sparge_attn.py`, whose map helper and kernel each copy
            # their inputs anyway, materializing the slices here would add
            # three copies per slice and take back most of the saving.
            out[:, sl] = self.forward(
                query[:, sl].unsqueeze(0),
                key[:, sl].unsqueeze(0),
                value[:, sl].unsqueeze(0),
                None,
            )[0]

    def _sage_packed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        bounds: tuple[int, ...],
        max_seqlen: int,
    ) -> torch.Tensor:
        # MiniMax-H3 packs one live document as bounds=(0, used, total):
        # [0, used) are real tokens; [used, total) is 64-aligned tail padding.
        used = _trailing_padding_used_len(
            total_tokens=query.shape[0],
            max_seqlen=max_seqlen,
            bounds=bounds,
        )
        if used is not None:
            if not self._slices_heads(query.shape[-2]):
                live_out = self.forward(
                    query[:used].unsqueeze(0),
                    key[:used].unsqueeze(0),
                    value[:used].unsqueeze(0),
                    None,
                )[0]
                if used == query.shape[0]:
                    return live_out
                # Keep padded tail at zero so downstream masked rows stay
                # inactive.
                output = torch.zeros_like(query)
                output[:used] = live_out
                return output
            output = (
                torch.empty_like(query)
                if used == query.shape[0]
                # Keep padded tail at zero so downstream masked rows stay
                # inactive.
                else torch.zeros_like(query)
            )
            self._attend_into(
                out=output[:used],
                query=query[:used],
                key=key[:used],
                value=value[:used],
            )
            return output

        output = torch.empty_like(query)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if start == stop:
                continue
            self._attend_into(
                out=output[start:stop],
                query=query[start:stop],
                key=key[start:stop],
                value=value[start:stop],
            )
        return output
