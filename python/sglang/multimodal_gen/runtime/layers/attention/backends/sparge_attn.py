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
an optional tail cutoff does the same for the last few steps, whose errors no
later step can absorb; short sequences run dense because the block map costs
more than the blocks it saves. The cutoffs are configured through
``--attention-backend-config``::

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
# Trailing denoise forwards kept dense. Off by default: unlike the warmup
# cutoff this one has not been measured on this model, and the two arguments
# for it point in opposite directions.
#
# For it: denoising is self-correcting in the middle but not at the ends. An
# error injected at step t perturbs the latent, and steps t+1..T re-denoise
# from the perturbed latent and land on something plausible -- the error is
# absorbed as "a slightly different but coherent sample". The last step has no
# successor, so whatever the block map gets wrong there reaches the decoder
# unfiltered. Truncation is at 128x64 token granularity, which in a packed
# video sequence is a contiguous run of patches, so that error is spatially
# block-structured, which is the visual signature of the artifacts this knob
# exists to test for.
#
# Against it: late-step attention is typically more concentrated than
# early-step attention, so a fixed top-k budget drops less probability mass
# there. That is why SpargeAttn's own experiments, and every other sparse
# schedule in this tree, protect the warmup steps and nothing else.
#
# Two steps out of fifty is ~4% of the schedule, cheap enough to keep if it
# measures well. Measure against the dense render before raising this default.
DEFAULT_SKIP_LAST_STEPS = 0
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
# What "disabled" means: the kernel keeps a PV block when
# `local_max_diff + pv_threshold > 0`, so a large enough threshold makes that
# test unconditionally true and the skip branch unreachable.
PVTHRESHD_DISABLED = 1_000_000


def _default_pvthreshd() -> int:
    """``DEFAULT_PVTHRESHD``, except on Hopper where PV thresholding hangs.

    SpargeAttn compiles a separate kernel for sm90
    (``qk_int_sv_f8_cuda_sm90``, CTA 64x128, warpgroup ``wgmma.*.sync.aligned``)
    from the one every other supported arch uses (``..._sm89``, CTA 128x64,
    warp-level ``mma.sync``). Only the sm90 one has been observed to hang: on
    an 8-GPU H200 run a single rank's
    ``qk_int8_sv_f8_attn_kernel<64u, 128u, ..., (PVThresholdMode)1, ...>``
    stayed Active with one block left on one SM, which stalled that rank's
    Ulysses all_to_all and deadlocked the whole SP group behind it. Disabling
    the PV threshold -- the one thing that template parameter controls -- made
    the same path pass repeatedly.

    The hang was never reproduced from synthetic activations, so this is a
    mitigation rather than a fix for a fully understood kernel bug. It is
    scoped to sm90 so 5090/sm120 deployments keep the second-stage sparsity,
    and an explicit ``pvthreshd`` in --attention-backend-config still wins.
    """
    try:
        from spas_sage_attn.core import get_cuda_arch_versions

        if get_cuda_arch_versions()[torch.cuda.current_device()] == "sm90":
            return PVTHRESHD_DISABLED
    except Exception:  # pragma: no cover - defensive
        pass
    return DEFAULT_PVTHRESHD

# How many heads the sparse path runs at a time.
#
# Sparsity is not free in memory: every transient scales with the head count
# times the sequence length, and this backend is wanted precisely where that
# product is largest. The map helper makes two [1, H, S, D] contiguous copies
# of Q and K and hands back int8 copies on top, and
# `block_sparse_sage2_attn_cuda` then makes its own contiguous Q/K/V, quantizes
# Q/K again and allocates the output. At 28 rank-local heads and a 145k-row
# packed sequence that comes to 6.6 GiB against 3.4 GiB for the dense
# `sage_attn` fallback -- the sparse backend cost 3.2 GiB/GPU more than the
# path it replaces, which on a 32 GiB card is several reference videos' worth
# of budget.
#
# Attention is head-parallel and so is the block map -- pooling, the similarity
# gates, the pooled-score softmax, the sort and `fill_block_map_triton` all
# carry the head axis through untouched -- so running a slice of heads at a
# time and concatenating is exact, not an approximation. Measured on one RTX
# 5090 at H=28, D=128, bf16, with text rows protected, against the whole-head
# path (identical output tensors, `torch.equal`):
#
#     S=86k   3.86 -> 1.29 GiB   1.08x time
#     S=137k  6.23 -> 2.07 GiB   1.05x time
#     S=145k  6.62 -> 2.19 GiB   1.06x time
#
# That is 67% off the peak and lands it *below* the dense fallback, so enabling
# sparsity no longer raises the memory ceiling at all. The time cost is real
# but small against what sparsity buys at these lengths (S=145k: 315 ms sliced,
# 298 ms whole, 525 ms dense). Slices narrower than 4 start paying for the
# extra launches without helping much further. 0 runs every head in one call.
DEFAULT_HEAD_CHUNK = 4

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
    row space the attention call sees -- which under Ulysses is the *whole*
    packed sequence, not the caller's row shard, because the shard is restored
    to full length inside the attention call. Publishing a rank-local slice
    there is the mistake this contract exists to name.

    The backend length-checks the tags against the query it is handed and runs
    dense when they do not line up, so getting it wrong costs the speedup
    rather than protecting the wrong rows. That check is a backstop, not a
    routine path: a caller it fires for has published tags attention cannot
    use, and the warning says so.

    A no-op for every backend other than this one.
    """
    token = _row_modality_tags.set(tags)
    try:
        yield
    finally:
        _row_modality_tags.reset(token)


# Length of the denoise schedule the current run is stepping through, published
# by the stage that owns the loop. Only ``skip_last_steps`` needs it.
_denoise_total_steps: ContextVar[int | None] = ContextVar(
    "sparge_denoise_total_steps", default=None
)


@contextmanager
def sparge_denoise_total_steps(total: int | None) -> Iterator[None]:
    """Publish how many denoise steps the loop about to run will take.

    ``skip_last_steps`` cannot be applied without this. ``current_timestep`` is
    a zero-based counter with no upper bound attached to it, so on its own it
    cannot say which forward is the last one.

    The caller must be whoever owns the sigma schedule, because that is the
    only place the count is authoritative.
    ``sampling_params.num_inference_steps`` is a request-level *hint*: for
    MiniMax-H3 it may be a ``(video, audio)`` pair rather than an int, and the
    loop actually runs ``len(sigmas_video) - 1``. Reading the hint instead
    would leave the tail cutoff silently inactive on exactly the model this
    backend targets, which is worse than not having it -- an experiment that
    never ran looks like an experiment that came back negative.

    A no-op for every backend other than this one, and for this one unless
    ``skip_last_steps`` is set.
    """
    token = _denoise_total_steps.set(total)
    try:
        yield
    finally:
        _denoise_total_steps.reset(token)


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
    skip_last_steps: int
    skip_first_layers: int
    min_seq_len: int
    simthreshd1: float
    pvthreshd: int
    dense_modalities: tuple[int, ...]
    head_chunk: int

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
            skip_last_steps=int(
                config.get("skip_last_steps", DEFAULT_SKIP_LAST_STEPS)
            ),
            skip_first_layers=int(
                config.get("skip_first_layers", DEFAULT_SKIP_FIRST_LAYERS)
            ),
            min_seq_len=int(config.get("min_seq_len", DEFAULT_MIN_SEQ_LEN)),
            simthreshd1=float(config.get("simthreshd1", DEFAULT_SIMTHRESHD1)),
            pvthreshd=int(config.get("pvthreshd", _default_pvthreshd())),
            dense_modalities=tuple(
                int(tag)
                for tag in config.get("dense_modalities", DEFAULT_DENSE_MODALITIES)
            ),
            head_chunk=int(
                config.get("head_chunk", DEFAULT_HEAD_CHUNK)
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
        if (
            schedule.skip_first_steps < 0
            or schedule.skip_last_steps < 0
            or schedule.skip_first_layers < 0
        ):
            raise ValueError("sparge skip_first_*/skip_last_* must be non-negative")
        if schedule.min_seq_len < 128:
            # The kernel itself asserts seq_len >= 128.
            raise ValueError(
                f"sparge min_seq_len must be at least 128, got {schedule.min_seq_len}"
            )
        if schedule.head_chunk < 0:
            raise ValueError(
                "sparge head_chunk must be non-negative (0 runs every head in "
                f"one call), got {schedule.head_chunk}"
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
            tail = (
                f", the last {self.schedule.skip_last_steps} denoise steps"
                if self.schedule.skip_last_steps
                else ""
            )
            logger.info_once(
                f"Sparge attention: {self._selection_rule()} "
                f"pvthreshd={self.schedule.pvthreshd}, dense for the first "
                f"{self.schedule.skip_first_steps} denoise steps{tail}, the "
                f"first {self.schedule.skip_first_layers} DiT layers, and "
                f"sequences under {self.schedule.min_seq_len} tokens"
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
        """Whether this denoise step may sparsify, by index within the schedule.

        Both cutoffs fall back the same way: returning False here sends the
        call through ``dense_impl``, which is the unmodified ``sage_attn``
        backend, so an excluded step costs exactly what a plain
        ``--attention-backend sage_attn`` deployment costs.
        """
        context = get_forward_context()
        step = context.current_timestep
        total = self._total_steps(context)
        self._warn_if_the_cutoffs_swallow_the_schedule(total)
        if step < self.schedule.skip_first_steps:
            return False
        if self.schedule.skip_last_steps <= 0:
            return True
        if total is None:
            self._warn_tail_cutoff_has_no_schedule_length()
            return True
        return step < total - self.schedule.skip_last_steps

    @staticmethod
    def _total_steps(context) -> int | None:
        """Length of the running denoise schedule, or None if nothing said.

        The published value wins: it comes from the stage that owns the sigma
        schedule and is what the loop actually iterates.
        ``num_inference_steps`` is the request-level hint and only usable when
        it really is a positive int -- MiniMax-H3 types it as
        ``int | tuple[int, int]``, and a pair carries no single loop length.
        """
        published = _denoise_total_steps.get()
        if isinstance(published, int) and published > 0:
            return published
        batch = getattr(context, "forward_batch", None)
        hint = getattr(
            getattr(batch, "sampling_params", None), "num_inference_steps", None
        )
        if isinstance(hint, int) and not isinstance(hint, bool) and hint > 0:
            return hint
        return None

    def _warn_if_the_cutoffs_swallow_the_schedule(self, total: int | None) -> None:
        """The two step cutoffs together can leave no sparse step at all.

        The default warmup cutoff of 10 assumes the 50-step schedule. Turbo
        LoRAs run 9 or 5 steps, where every index is below the cutoff and this
        backend silently degrades into plain SageAttention. Adding a tail
        cutoff makes the same mistake reachable from the other end. Either way
        it is a config error worth a line in the log rather than an unexplained
        absence of speedup.
        """
        if total is None:
            return
        dense = self.schedule.skip_first_steps + self.schedule.skip_last_steps
        if dense < total:
            return
        logger.warning_once(
            f"Sparge attention never activates: skip_first_steps="
            f"{self.schedule.skip_first_steps} + skip_last_steps="
            f"{self.schedule.skip_last_steps} covers all {total} denoise steps "
            f"this request runs, so every step takes the dense path. Lower them "
            f"(warmup is roughly 20% of the schedule, so ~2 for a 9-step turbo "
            f"checkpoint) via --attention-backend-config."
        )

    def _warn_tail_cutoff_has_no_schedule_length(self) -> None:
        """``skip_last_steps`` set but nobody published the schedule length.

        Fail open rather than closed: without a length the last step cannot be
        identified, and running every step dense would disable the backend
        wholesale over a missing integer. Sparsifying the tail is the smaller
        error, but it must be loud -- a silently inactive cutoff would read as
        evidence that the tail is not the problem.
        """
        logger.warning_once(
            f"Sparge attention skip_last_steps={self.schedule.skip_last_steps} "
            "is inactive: no denoise schedule length was published, so the "
            "last step cannot be identified and every step stays sparse. The "
            "pipeline stage owning the loop must wrap it in "
            "sparge_denoise_total_steps()."
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

    def _head_slices(self, *tensors: torch.Tensor):
        """Yield ``head_chunk``-wide head slices of ``[1, S, H, D]`` tensors.

        Attention is head-parallel and so is the block map, so running the
        whole sparse path a slice at a time is exact -- see
        ``DEFAULT_HEAD_CHUNK`` for why it is worth doing. Slices are made
        contiguous here because both the map helper and the kernel copy their
        inputs anyway; doing it per slice is what keeps the copies small.
        A chunk that covers every head yields the originals untouched, so the
        unsliced path costs nothing.
        """
        heads = tensors[0].shape[-2]
        chunk = self.schedule.head_chunk
        if chunk <= 0 or chunk >= heads:
            yield tensors
            return
        for head in range(0, heads, chunk):
            yield tuple(
                t[:, :, head : head + chunk].contiguous() for t in tensors
            )

    def _block_map(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        blk_q: int,
        blk_k: int,
    ) -> torch.Tensor:
        """Pooled-Q.K block map for one head slice of ``[1, S, H, D]``.

        The helper's own int8/scale returns are dropped as soon as it returns
        -- ``block_sparse_sage2_attn_cuda`` re-quantizes from the untouched NHD
        Q/K anyway, so holding them only widens the peak.
        """
        from spas_sage_attn.utils import get_block_map_meansim_fuse_quant

        # The map is computed in HND, the layout SpargeAttn's own API converts
        # to before doing the same thing.
        q_hnd = q.transpose(1, 2).contiguous()
        k_hnd = k.transpose(1, 2).contiguous()
        km = k_hnd.mean(dim=-2, keepdim=True)
        block_map, *quantized = get_block_map_meansim_fuse_quant(
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
        # Named rather than `*_` so the int8/scale copies can actually be
        # dropped here instead of living until this frame returns.
        del quantized, q_hnd, k_hnd, km
        return block_map

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

            # SpargeAttn's own entry point, so `head_chunk` does not
            # reach the [1, H, S, D] copies it makes internally. Left alone
            # because it asks for the map with `return_lut=True` and feeds the
            # quantized Q/K straight to the kernel; routing it through
            # `_block_map` would bound the transients but add back the
            # quantization pass `block_sparse_sage2_attn_cuda` performs. Any
            # H3 run that protects a modality -- the default -- takes the
            # explicit path below instead.
            #
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

        # Same block geometry the kernel is compiled for; sm90 transposes it.
        arch = get_cuda_arch_versions()[q.device.index]
        blk_q, blk_k = (64, 128) if arch == "sm90" else (128, 64)

        seq = q.shape[1]
        n_q = (seq + blk_q - 1) // blk_q
        n_k = (seq + blk_k - 1) // blk_k

        # Head-independent, so it is worked out once and reused by every slice.
        protected_q = protected_k = None
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
            logger.info_once(
                f"Sparge attention active: S={k.shape[1]} heads={q.shape[2]} "
                f"{self._selection_rule()}, keeping "
                f"{int(protected_q.sum())}/{n_q} query blocks and "
                f"{int(protected_k.sum())}/{n_k} key blocks dense for modalities "
                f"{list(self.schedule.dense_modalities)}"
            )

        outputs = []
        for q_s, k_s, v_s in self._head_slices(q, k, v):
            block_map = self._block_map(q_s, k_s, blk_q=blk_q, blk_k=blk_k)
            if protected_q is not None:
                # Protected rows keep every key; protected keys are kept by
                # every row.
                block_map[..., protected_q, :] = True
                block_map[..., :, protected_k] = True
            outputs.append(
                block_sparse_sage2_attn_cuda(
                    q_s,
                    k_s,
                    v_s,
                    mask_id=block_map,
                    pvthreshd=self.schedule.pvthreshd,
                    scale=self.softmax_scale,
                    tensor_layout="NHD",
                )
            )
            # Let this slice's transients go before the next one allocates.
            del q_s, k_s, v_s, block_map
        if len(outputs) == 1:
            return outputs[0]
        return torch.cat(outputs, dim=-2)

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
    "sparge_denoise_total_steps",
    "sparge_row_modality_tags",
]
