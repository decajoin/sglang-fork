# SPDX-License-Identifier: Apache-2.0
"""SpargeAttn block-sparse attention backend.

Training-free block sparsity on top of SageAttention2's quantized kernel
(`thu-ml/SpargeAttn <https://github.com/thu-ml/SpargeAttn>`_, ICML 2025). Two
savings stack in one kernel launch: Q/K are quantized to INT8 and V to FP8, and
a block map computed from the same pooled Q/K decides which 128x64 blocks are
computed at all. Nothing is trained and no weights change -- the map is derived
from the activations of the call it serves.

This is a separate backend from ``sage_attn``; that one stays exactly as it is
and is reused unmodified here as the dense fallback, so every call this backend
declines to sparsify still gets SageAttention's quantization speedup.

Sparsity is not applied everywhere. The early denoise steps settle the layout of
the sample and tolerate approximation badly, so the backend runs dense for them;
short sequences run dense because the block map costs more than the blocks it
saves. Both cutoffs are configured through ``--attention-backend-config``::

    --attention-backend sparge_attn \
    --component-attention-backends text_encoder=fa \
    --attention-backend-config '{"topk": 0.5}'

``text_encoder=fa`` is not optional: ``--attention-backend`` reaches every
component and the Qwen3-VL text encoder admits only fa / torch_sdpa /
sage_attn_3. Put the override on the *encoder*; ``transformer=sparge_attn``
appears to work and silently does nothing, because H3 resolves the DiT backend
lazily on the first forward, outside the component-loading context.

Requirements inherited from the kernel: compute capability >= 8.0 with the
package built for that arch, fp16/bf16, head_dim 64 or 128, one contiguous
sequence per call. Anything else -- cross attention, the token refiner, short
sequences, head_dim 256 -- falls back to dense for that call, so no layer has to
be excluded by hand.

**Causal attention always runs dense.** ``spas_sage2_attn_meansim_topk_cuda``
passes ``is_causal=False`` down to the kernel and applies causality only through
the block map, which masks at 128x64 block granularity and leaves the diagonal
block unmasked inside. That is silently wrong for a causal model, so this
backend refuses to sparsify it rather than returning a plausible tensor. H3 is
non-causal, so nothing is lost there.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

import msgspec
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# The kernel quantizes per 128x64 block and asserts head_dim in {64, 128}.
SPARGE_SUPPORTED_HEAD_DIMS = (64, 128)

# Fraction of key blocks kept per query block. Lower is faster and less
# accurate. 0.5 is SpargeAttn's own documented default for the plug-and-play
# API; measured on one RTX 5090 at S=37632, H=14, D=128, bf16, against bf16
# SDPA on the attention op alone: SageAttention (dense, no sparsity) 2.63x,
# topk=0.7 3.35x, topk=0.5 4.42x, topk=0.3 6.71x. Those are op-level numbers --
# attention is only part of a denoise step, so end-to-end gain is much smaller
# and shrinks further at shorter sequences.
DEFAULT_TOPK = 0.5
# Leading denoise forwards kept dense. Carried over from the subblock_sparse
# schedule, which measured on this same model that lowering it from 10 to 5
# halves cosine against the dense render and visibly re-frames the shot. It has
# not been re-swept for this backend's block map -- treat 10 as a starting
# point, not a measured optimum for SpargeAttn.
DEFAULT_SKIP_FIRST_STEPS = 10
# Depth does not behave like step index: subblock_sparse measured the layer
# cutoff as worth ~1% of time for 0.0013 of cosine, inside its noise floor.
DEFAULT_SKIP_FIRST_LAYERS = 0
# Below this the block map (pool + softmax + sort + cumsum + two Triton fills)
# costs more than the blocks it saves. Measured on one RTX 5090 at H=14, D=128,
# bf16: at S=4096 topk=0.5 gives 1.07x over SDPA against SageAttention's 1.96x,
# i.e. sparsity is a net *loss* against the dense fallback there; by S=16384 it
# is 3.85x against 2.65x. 4096 is the floor, not a tuning knob.
DEFAULT_MIN_SEQ_LEN = 4096
# SpargeAttn's first-stage selectivity filter: blocks whose tokens are mutually
# similar are the ones safe to approximate. Negative disables it, which is what
# the topk API defaults to upstream -- the top-k budget then does all the work.
DEFAULT_SIMTHRESHD1 = -0.1
# Adaptive top-p. SpargeAttn supports selecting blocks by cumulative softmax
# mass instead of a fixed count: rows whose attention is concentrated take few
# blocks, rows that are spread out take many. `topk` spends the same budget on
# every row regardless of how much mass it actually needs, so at a matched
# average density top-p is the better-conditioned rule -- it is what
# `spas_sage2_attn_meansim_cuda` uses (0.98). Mutually exclusive with `topk`;
# set `{"cdfthreshd": 0.98, "topk": null}` to switch.
DEFAULT_CDFTHRESHD = None
# Second-stage PV sparsity threshold. SpargeAttn's own default; the SageSLA
# integration in this tree pins it to 1e6 (disabled) instead.
DEFAULT_PVTHRESHD = 50

# Token tags that must keep full attention, as MiniMax-H3 numbers them in
# ``minimax_h3/packed_sequence.py``: 0 VIDEO, 1 TEXT, 2 AUDIO, -1 PADDING
# (clamped to 0 in the rank-local copy). Audio is protected by default.
#
# Audio is a small minority of the packed rows -- a few thousand against ~37k
# video rows -- and that is exactly why an untargeted top-k budget ruins it.
# A block-sparse budget picked from pooled Q.K scores allocates blocks in
# proportion to how many there are, so audio query rows spend their budget on
# video keys and audio key blocks get dropped from nearly every row. The
# damage shows up as corrupted audio in the opening seconds of the clip.
# This tree's own `--minimax-h3-segment-sparse-attn` reaches the same
# conclusion from the other direction: it leaves reference audio at full
# attention because restricting it "collapsed measured soundtrack fidelity
# for ~1% of the saving".
# Text is protected for the same reason, and it is the more extreme case: a
# prompt is ~512 of 37632 rows, which at BLKK=64 is **8 key blocks out of 588**
# (1.4%). Dropping half of eight conditioning blocks is not comparable to
# dropping half of video's 532 -- it weakens prompt adherence without ever
# looking obviously broken, which is exactly how it was first noticed (a small
# quality drop against plain sage_attn at topk=0.5). Protecting both costs
# ~9.5% of the key blocks.
DEFAULT_DENSE_MODALITIES = (1, 2)

# Rank-local per-row modality tags for the sequence attention is about to see,
# published by the model around its block stack. Absent for models that do not
# carry modality tags at all.
_row_modality_tags: ContextVar[torch.Tensor | None] = ContextVar(
    "sparge_row_modality_tags", default=None
)


@contextmanager
def sparge_row_modality_tags(tags: torch.Tensor | None) -> Iterator[None]:
    """Publish per-row modality tags for the rows attention will receive.

    ``tags`` is a 1-D integer tensor, one entry per packed row, in the same
    row space the attention call sees. The backend validates the length
    against the query it is handed and ignores tags that do not line up, so a
    caller that reshuffles rows afterwards (Ulysses restores the full sequence
    inside attention) degrades to dense rather than protecting the wrong rows.

    A no-op for every backend other than this one.
    """
    token = _row_modality_tags.set(tags)
    try:
        yield
    finally:
        _row_modality_tags.reset(token)


# ``blocks.<idx>.attn`` is a DiT layer; ``token_refiner.blocks.<idx>.attn`` and
# anything else is not and stays dense.
_DIT_LAYER_PREFIX = re.compile(r"^blocks\.(\d+)\.")


def _dit_layer_index(prefix: str) -> int | None:
    match = _DIT_LAYER_PREFIX.match(prefix)
    return int(match.group(1)) if match else None


def _trailing_padding_used_len(
    *,
    total_tokens: int,
    max_seqlen: int,
    bounds: tuple[int, ...],
) -> int | None:
    """Return live token count for H3-style ``[0, used, total]`` padding.

    Mirrors the same helper in ``sage_attn.py``; kept local so that backend
    stays untouched.
    """
    if len(bounds) != 3:
        return None
    start, used, total = bounds
    if start != 0 or used >= total or total != total_tokens or used != max_seqlen:
        return None
    return used


class SpargeAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return list(SPARGE_SUPPORTED_HEAD_DIMS)

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SPARGE_ATTN

    @staticmethod
    def get_impl_cls() -> type["SpargeAttentionImpl"]:
        return SpargeAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["SpargeAttentionMetadata"]:
        return SpargeAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["SpargeAttentionMetadataBuilder"]:
        return SpargeAttentionMetadataBuilder


@dataclass
class SpargeAttentionMetadata(AttentionMetadata):
    current_timestep: int


class SpargeAttentionMetadataBuilder(AttentionMetadataBuilder):
    # The base class declares __init__ abstract, so a builder that does not
    # override it cannot be instantiated at all.
    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(  # type: ignore[override]
        self, current_timestep: int, **kwargs: dict[str, Any]
    ) -> SpargeAttentionMetadata:
        return SpargeAttentionMetadata(current_timestep=current_timestep)


class SpargeSchedule(msgspec.Struct, frozen=True):
    """When sparsity is allowed to apply, and how much of it."""

    topk: float | None
    cdfthreshd: float | None
    skip_first_steps: int
    skip_first_layers: int
    min_seq_len: int
    simthreshd1: float
    pvthreshd: int
    dense_modalities: tuple[int, ...]

    @classmethod
    def from_server_args(cls) -> "SpargeSchedule":
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        config = get_global_server_args().attention_backend_config or {}
        _cdf = config.get("cdfthreshd", DEFAULT_CDFTHRESHD)
        # Naming `cdfthreshd` alone switches selection rules; `topk` only keeps
        # its default when no top-p threshold was asked for.
        _topk = config.get("topk", None if _cdf is not None else DEFAULT_TOPK)
        schedule = SpargeSchedule(
            topk=(None if _topk is None else float(_topk)),
            cdfthreshd=(None if _cdf is None else float(_cdf)),
            skip_first_steps=int(
                config.get("skip_first_steps", DEFAULT_SKIP_FIRST_STEPS)
            ),
            skip_first_layers=int(
                config.get("skip_first_layers", DEFAULT_SKIP_FIRST_LAYERS)
            ),
            min_seq_len=int(config.get("min_seq_len", DEFAULT_MIN_SEQ_LEN)),
            simthreshd1=float(config.get("simthreshd1", DEFAULT_SIMTHRESHD1)),
            pvthreshd=int(config.get("pvthreshd", DEFAULT_PVTHRESHD)),
            dense_modalities=tuple(
                int(tag)
                for tag in config.get("dense_modalities", DEFAULT_DENSE_MODALITIES)
            ),
        )
        if (schedule.topk is None) == (schedule.cdfthreshd is None):
            raise ValueError(
                "sparge takes exactly one of topk (fixed block budget) or "
                "cdfthreshd (adaptive top-p); got "
                f"topk={schedule.topk}, cdfthreshd={schedule.cdfthreshd}"
            )
        # topk == 1.0 keeps every block and is the calibration setting used by
        # the tests, so it has to stay legal; 0 would keep nothing.
        if schedule.topk is not None and not 0.0 < schedule.topk <= 1.0:
            raise ValueError(f"sparge topk must be in (0, 1], got {schedule.topk}")
        if schedule.cdfthreshd is not None and not 0.0 < schedule.cdfthreshd <= 1.0:
            raise ValueError(
                f"sparge cdfthreshd must be in (0, 1], got {schedule.cdfthreshd}"
            )
        if schedule.skip_first_steps < 0 or schedule.skip_first_layers < 0:
            raise ValueError("sparge skip_first_* must be non-negative")
        if schedule.min_seq_len < 128:
            # The kernel itself asserts seq_len >= 128.
            raise ValueError(
                f"sparge min_seq_len must be at least 128, got {schedule.min_seq_len}"
            )
        return schedule


class SpargeAttentionImpl(AttentionImpl):
    """Block-sparse attention with a SageAttention dense fallback.

    One impl instance is built per attention module, so ``prefix`` fixes the
    layer for the lifetime of the object; only the denoise step varies per call
    and it comes from the forward context.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool = False,
        softmax_scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.prefix = prefix
        self.num_heads = num_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = (
            softmax_scale if softmax_scale is not None else head_size**-0.5
        )
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        self.schedule = SpargeSchedule.from_server_args()
        self.layer_idx = _dit_layer_index(prefix)
        # A layer outside the DiT stack (token refiner, cross attention) never
        # runs sparse: its sequences are short and its block budget meaningless.
        self.layer_enabled = (
            self.layer_idx is not None
            and self.layer_idx >= self.schedule.skip_first_layers
            and head_size in SPARGE_SUPPORTED_HEAD_DIMS
            # See the module docstring: the sage2 path drops the causal flag on
            # the floor, so causal layers must never take the sparse branch.
            and not causal
            # GQA would need K repeated before the block map, which pools Q and
            # K into one matmul and so requires equal head counts.
            and self.num_kv_heads == num_heads
        )
        self.dense_impl = self._build_dense_impl(causal=causal)
        if self.layer_enabled:
            logger.info_once(
                f"Sparge attention: {self._selection_rule()} "
                f"pvthreshd={self.schedule.pvthreshd}, dense for the first "
                f"{self.schedule.skip_first_steps} denoise steps, the first "
                f"{self.schedule.skip_first_layers} DiT layers, and sequences "
                f"under {self.schedule.min_seq_len} tokens"
            )

    def _build_dense_impl(self, *, causal: bool) -> AttentionImpl:
        """SageAttention, used wherever the schedule excludes sparsity.

        SpargeAttn *is* SageAttention2 plus a block map, so the dense fallback
        should be SageAttention rather than an unquantized kernel: the excluded
        region then costs what the user's current ``--attention-backend
        sage_attn`` deployment already costs, and switching this backend on
        cannot make any call slower than that baseline. The existing
        ``sage_attn`` backend is used exactly as-is.
        """
        from sglang.multimodal_gen.runtime.layers.attention.selector import (
            get_attn_backend,
        )

        backend = get_attn_backend(
            self.head_size,
            torch.bfloat16,
            supported_attention_backends={
                AttentionBackendEnum.SAGE_ATTN,
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            },
            selected_attention_backend=AttentionBackendEnum.SAGE_ATTN,
        )
        return backend.get_impl_cls()(
            num_heads=self.num_heads,
            head_size=self.head_size,
            causal=causal,
            softmax_scale=self.softmax_scale,
            num_kv_heads=self.num_kv_heads,
            prefix=f"{self.prefix}.dense",
        )

    def _step_enabled(self) -> bool:
        context = get_forward_context()
        self._warn_if_schedule_is_shorter_than_the_cutoff(context)
        return context.current_timestep >= self.schedule.skip_first_steps

    def _warn_if_schedule_is_shorter_than_the_cutoff(self, context) -> None:
        """A distilled checkpoint can be shorter than ``skip_first_steps``.

        The default cutoff of 10 assumes the 50-step schedule. Turbo LoRAs run
        ``num_inference_steps`` of 9 or 5, where every step index is below the
        cutoff and this backend silently degrades into plain SageAttention.
        That is a config mistake worth a line in the log rather than an
        unexplained absence of speedup.
        """
        batch = getattr(context, "forward_batch", None)
        total = getattr(
            getattr(batch, "sampling_params", None), "num_inference_steps", None
        )
        if isinstance(total, int) and total <= self.schedule.skip_first_steps:
            logger.warning_once(
                f"Sparge attention never activates: skip_first_steps="
                f"{self.schedule.skip_first_steps} but this request runs only "
                f"{total} denoise steps, so every step takes the dense path. "
                f"Lower skip_first_steps (roughly 20% of the schedule, so ~2 "
                f"for a 9-step turbo checkpoint) via --attention-backend-config."
            )

    def _sparse_ready(self, q: torch.Tensor, k: torch.Tensor) -> bool:
        """``q``/``k`` are ``[B, S, H, D]`` or ``[1, S, H, D]`` NHD slices."""
        return (
            self.layer_enabled
            and self._step_enabled()
            and q.dtype in (torch.bfloat16, torch.float16)
            and q.dtype == k.dtype
            and k.shape[-3] >= self.schedule.min_seq_len
        )

    def _selection_rule(self) -> str:
        return (
            f"topk={self.schedule.topk}"
            if self.schedule.topk is not None
            else f"cdfthreshd={self.schedule.cdfthreshd}"
        )

    def _protection_for(
        self, rows: int, offset: int = 0
    ) -> tuple[bool, torch.Tensor | None]:
        """``(can_sparsify, protected_row_mask)`` for one contiguous window.

        Three outcomes, and the difference matters: protection switched off
        (sparsify freely), tags located (sparsify around them), or protection
        wanted but the tags do not describe these rows. The last case must run
        dense -- sparsifying blind is what corrupts audio, and it does so
        silently.
        """
        if not self.schedule.dense_modalities:
            return True, None
        tags = _row_modality_tags.get()
        flat = None if tags is None else tags.view(-1)
        if flat is None or flat.shape[0] < offset + rows:
            logger.warning_once(
                "Sparge attention is running dense: dense_modalities="
                f"{list(self.schedule.dense_modalities)} needs per-row modality "
                f"tags covering rows [{offset}, {offset + rows}), but "
                + ("none were published" if flat is None else f"only {flat.shape[0]} exist")
                + ". Sparsifying without them would drop audio blocks. Set "
                '\'{"dense_modalities": []}\' to sparsify anyway.'
            )
            return False, None
        window = flat[offset : offset + rows]
        protected = torch.zeros_like(window, dtype=torch.bool)
        for tag in self.schedule.dense_modalities:
            protected |= window == tag
        return True, protected

    @staticmethod
    def _blocks_touching(protected: torch.Tensor, block: int, count: int):
        """Which block indices contain at least one protected row.

        Block granularity is conservative on purpose: a 128x64 block holding a
        single audio row is kept in full, because the kernel cannot mask
        inside a block.
        """
        padded = torch.zeros(count * block, dtype=torch.bool, device=protected.device)
        padded[: protected.shape[0]] = protected
        return padded.view(count, block).any(dim=1)

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        protected: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """q, k, v: ``[1, S, H, D]`` -> same shape.

        With nothing to protect this is SpargeAttn's plug-and-play API. With
        protected rows it is the same block map, patched so that every block
        holding a protected row is kept as both a query row and a key column,
        then handed to the explicit-mask kernel.
        """
        if (
            self.schedule.topk is not None
            and (protected is None or not bool(protected.any()))
        ):
            from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

            # Proof that the sparse path actually ran, with the shape it ran
            # on -- the construction-time log only says the layer was eligible.
            logger.info_once(
                f"Sparge attention active: S={k.shape[1]} heads={q.shape[2]} "
                f"{self._selection_rule()}"
            )
            return spas_sage2_attn_meansim_topk_cuda(
                q,
                k,
                v,
                topk=self.schedule.topk,
                simthreshd1=self.schedule.simthreshd1,
                pvthreshd=self.schedule.pvthreshd,
                is_causal=False,
                scale=self.softmax_scale,
                tensor_layout="NHD",
            )

        from spas_sage_attn import block_sparse_sage2_attn_cuda
        from spas_sage_attn.core import get_cuda_arch_versions
        from spas_sage_attn.utils import get_block_map_meansim_fuse_quant

        # Same block geometry the kernel is compiled for; sm90 transposes it.
        arch = get_cuda_arch_versions()[q.device.index]
        blk_q, blk_k = (64, 128) if arch == "sm90" else (128, 64)

        # The map is computed in HND, the layout SpargeAttn's own API converts
        # to before doing the same thing.
        q_hnd = q.transpose(1, 2).contiguous()
        k_hnd = k.transpose(1, 2).contiguous()
        km = k_hnd.mean(dim=-2, keepdim=True)
        block_map, *_ = get_block_map_meansim_fuse_quant(
            q_hnd,
            k_hnd,
            km,
            is_causal=False,
            BLKQ=blk_q,
            BLKK=blk_k,
            simthreshd1=self.schedule.simthreshd1,
            cdfthreshd=self.schedule.cdfthreshd,
            topk=self.schedule.topk,
            return_lut=False,
        )

        n_q, n_k = block_map.shape[-2], block_map.shape[-1]
        # `protected` is None whenever protection is switched off; this path is
        # also how cdfthreshd runs, which the plug-and-play API cannot express.
        if protected is None:
            logger.info_once(
                f"Sparge attention active: S={k.shape[1]} heads={q.shape[2]} "
                f"{self._selection_rule()}, no modality kept dense"
            )
        else:
            protected_q = self._blocks_touching(protected, blk_q, n_q)
            protected_k = self._blocks_touching(protected, blk_k, n_k)
            # Protected rows keep every key; protected keys are kept by every row.
            block_map[..., protected_q, :] = True
            block_map[..., :, protected_k] = True
            logger.info_once(
                f"Sparge attention active: S={k.shape[1]} heads={q.shape[2]} "
                f"{self._selection_rule()}, keeping "
                f"{int(protected_q.sum())}/{n_q} query blocks and "
                f"{int(protected_k.sum())}/{n_k} key blocks dense for modalities "
                f"{list(self.schedule.dense_modalities)}"
            )
        return block_sparse_sage2_attn_cuda(
            q,
            k,
            v,
            mask_id=block_map,
            pvthreshd=self.schedule.pvthreshd,
            scale=self.softmax_scale,
            tensor_layout="NHD",
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SpargeAttentionMetadata | None = None,
    ) -> torch.Tensor:
        """query/key/value: ``[B, S, H, D]``."""
        if not self._sparse_ready(query, key):
            return self.dense_impl.forward(query, key, value, attn_metadata)
        ok, protected = self._protection_for(query.shape[-3])
        if not ok:
            return self.dense_impl.forward(query, key, value, attn_metadata)
        return self._sparse_attention(query, key, value, protected)

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
        """Packed ``[T, H, D]`` rows split into documents by ``cu_seqlens``.

        The block-sparse kernel takes one contiguous sequence, so each packed
        document is routed on its own. Documents shorter than ``min_seq_len``
        -- in MiniMax H3 the padding tail -- go through the dense kernel.
        """

        def all_dense() -> torch.Tensor:
            return self.dense_impl.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )

        if cu_seqlens_host is None or not self._sparse_ready(query, key):
            return all_dense()

        bounds = cu_seqlens_host
        # H3 packs one live document as bounds=(0, used, total): [0, used) are
        # real tokens and [used, total) is 64-aligned tail padding that must
        # stay zero so downstream masked rows stay inactive.
        used = _trailing_padding_used_len(
            total_tokens=query.shape[0],
            max_seqlen=max_seqlen,
            bounds=bounds,
        )
        if used is not None:
            ok, protected = self._protection_for(used)
            if used < self.schedule.min_seq_len or not ok:
                return all_dense()
            live_out = self._sparse_attention(
                query[:used].unsqueeze(0),
                key[:used].unsqueeze(0),
                value[:used].unsqueeze(0),
                protected,
            )[0]
            if used == query.shape[0]:
                return live_out
            output = torch.zeros_like(query)
            output[:used] = live_out
            return output

        segments = [
            (start, stop)
            for start, stop in zip(bounds[:-1], bounds[1:])
            if stop > start
        ]
        if not any(
            stop - start >= self.schedule.min_seq_len for start, stop in segments
        ):
            return all_dense()

        output = torch.empty_like(query)
        # cu_seqlens covers every packed row in practice; a caller that leaves a
        # tail outside the last document would otherwise read uninitialized
        # memory back out.
        if segments and segments[-1][1] < query.shape[0]:
            output[segments[-1][1] :].zero_()
        for start, stop in segments:
            q_seg = query[start:stop].unsqueeze(0)
            k_seg = key[start:stop].unsqueeze(0)
            v_seg = value[start:stop].unsqueeze(0)
            ok, protected = self._protection_for(stop - start, offset=start)
            if ok and stop - start >= self.schedule.min_seq_len:
                seg_out = self._sparse_attention(q_seg, k_seg, v_seg, protected)
            else:
                seg_out = self.dense_impl.forward(q_seg, k_seg, v_seg, None)
            output[start:stop] = seg_out[0]
        return output


__all__ = [
    "SpargeAttentionBackend",
    "SpargeAttentionImpl",
    "SpargeAttentionMetadata",
    "SpargeAttentionMetadataBuilder",
    "SpargeSchedule",
]
