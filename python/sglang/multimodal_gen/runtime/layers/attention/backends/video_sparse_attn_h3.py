# SPDX-License-Identifier: Apache-2.0
"""VSA-H3: video sparse attention for MiniMax-H3's packed mixed-modality DiT.

Ported from FastVideo's ``video_sparse_attn_h3`` backend (hao-ai-lab/FastVideo,
`VSA <https://arxiv.org/abs/2505.13389>`_) onto this tree's packed-varlen
attention contract. H3 runs one joint bidirectional attention over
``[text | reference/condition rows | audio | generated video | padding]``, so
this backend differs from the Wan-tuned ``video_sparse_attn`` in what it tiles
and what it protects:

- The unit is a 64-token tile. Video rows are tiled in 3D as ``(4, 4, 4)`` over
  the ``(T, H, W)`` patch grid, so one tile is a small space-time cube whose
  tokens actually attend to each other. Everything before the video block is
  tiled into segment-pure 64-row chunks -- a tile never straddles a modality
  boundary, because a tile is both the unit of selection and the unit of
  pooling, and pooling text with audio produces a score that describes neither.
- Selection is per (head, query tile): pooled Q.K over tiles, then the top
  ``(1 - sparsity)`` fraction of *video* key tiles. Prefix keys (text, audio,
  reference rows) are kept by every query -- they are a few percent of the
  sequence and lose every budget contest they enter, which is what
  ``sparge_attn`` protects text and audio for and what it saw corrupt audio
  when it did not. ``{"prefix_mode": "compete"}`` makes them compete under a
  FLOP-matched budget instead; it is the ablation, not the default.
- Prefix *queries* are always dense. They are a small minority of the rows, so
  the FLOPs saved by sparsifying them are noise against the risk to prompt
  adherence and audio.

The gate-compress branch of upstream VSA is not ported: it needs a trained
``to_gate_compress`` matrix per layer, which no MiniMax-H3 checkpoint this tree
loads carries. Without it VSA is exactly top-k block-sparse attention, which is
what the sparse branch computes here.

**This is not free accuracy.** VSA is a *trainable* sparse attention: FastVideo
runs it at ``sparsity=0.9`` against a VSA-distilled checkpoint where 0.9 is the
policy the student was trained under. Against a stock MiniMax-H3 checkpoint the
same setting is training-free block sparsity and its quality is unmeasured
here, which is why the warmup cutoff defaults to the same 10 steps every other
sparse backend in this tree uses. Measure against a dense render before
trusting a sparsity.

Configured through ``--attention-backend-config``::

    sglang serve --model-path MiniMaxAI/MiniMax-H3 \
      --attention-backend video_sparse_attn_h3 \
      --component-attention-backends text_encoder=fa \
      --attention-backend-config '{"sparsity": 0.9}'

``text_encoder=fa`` is not optional: ``--attention-backend`` reaches every
component and the Qwen3-VL text encoder admits only fa / torch_sdpa /
sage_attn_3. Put the override on the *encoder*; ``transformer=...`` appears to
work and silently does nothing, because H3 resolves the DiT backend lazily on
the first forward, outside the component-loading context.

The backend needs the request's sequence geometry -- which rows are prefix and
what the video grid is -- and the pipeline publishes it with
``vsa_h3_sequence_geometry()``. Any call whose rows that geometry does not
describe (the token refiner, a model that never published one) runs dense, so
no layer has to be excluded by hand.

Unlike a dense flash kernel, which streams and costs essentially nothing,
tiling means padded copies of Q, K and V plus an output buffer and a
``[heads, tiles, tiles]`` fp32 score matrix. Across 28 rank-local heads at a
116k-row sequence that is 3.7 GiB, enough to OOM a card the dense path fits on;
``head_chunk_budget_mib`` sizes the head slice so the transients stay inside a
budget instead.
"""

from __future__ import annotations

import functools
import math
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
from sglang.multimodal_gen.runtime.layers.attention.backends.denoise_schedule import (
    get_denoise_total_steps,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn import (
    VSA_TILE_SIZE,
    construct_variable_block_sizes,
    get_non_pad_index,
    get_tile_partition_indices,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.vsa_h3 import (
    BLOCK_SIZE,
    block_sparse_attn_forward,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# The 3D video tile. Shared with the Wan VSA backend rather than redeclared:
# both are the 64-token granularity the block-sparse kernel is built at, and a
# tile shape that disagreed with the kernel's block size would silently mistile
# the sequence.
assert math.prod(VSA_TILE_SIZE) == BLOCK_SIZE

# Fraction of *video* key tiles each video query tile drops. 0.9 is FastVideo's
# trained policy for the VSA-distilled H3 preview and its reference default.
# Measured here on one RTX 5090 at H=14, D=128, bf16, against bf16 SDPA on the
# whole backend call -- tiling, selection, the dense prefix launch and the
# gather included: 36k rows 2.95x at 0.9 and 3.72x at 0.95; 71k rows 3.8x and
# 5.1x. The block-sparse kernel alone is 7.4x / 13.4x at those sparsities; the
# difference is what protecting the prefix costs, and it is the honest number.
# All of it is op-level -- attention is only part of a denoise step, so the
# end-to-end gain is smaller again.
DEFAULT_SPARSITY = 0.9
# Leading denoise forwards kept dense. The early steps settle the layout of the
# sample and tolerate approximation badly. 10 of 50 is what subblock_sparse
# measured on this same model (lowering it to 5 halves cosine against the dense
# render and visibly re-frames the shot) and what sparge_attn carries. A
# VSA-distilled checkpoint trained at this sparsity wants 0 instead.
DEFAULT_SKIP_FIRST_STEPS = 10
# Trailing denoise forwards kept dense. Off by default, and it needs the
# schedule length: denoising absorbs a mid-schedule error by re-denoising from
# the perturbed latent, but nothing follows the last step, so what the block
# map drops there reaches the decoder unfiltered. Unmeasured on this model.
DEFAULT_SKIP_LAST_STEPS = 0
# Depth does not behave like step index: subblock_sparse measured the layer
# cutoff as worth ~1% of time for 0.0013 of cosine, inside its noise floor.
DEFAULT_SKIP_FIRST_LAYERS = 0
# Individual DiT layers forced dense regardless of the cutoffs, for
# probe-guided opt-outs of layers whose attention is too diffuse to sparsify.
DEFAULT_DENSE_LAYERS: tuple[int, ...] = ()
# Below this the selection (pool + matmul + top-k) costs more than the tiles it
# saves, and H3's short sequences -- the token refiner, the padding tail -- are
# not what this backend is for.
DEFAULT_MIN_SEQ_LEN = 4096
# How many heads the sparse path runs at a time. 0 sizes the slice from
# ``head_chunk_budget_mib`` below; an explicit count overrides that, and a
# count at or above the head count runs them all in one pass.
#
# Slicing is exact, not an approximation: attention is head-parallel and so is
# the selection, so a slice computes the same numbers the whole-head pass
# would (the tests assert bit equality).
DEFAULT_HEAD_CHUNK = 0
# Transient budget per attention call, which is what the head slice is sized
# to hit. This backend is not free in memory the way a dense kernel is: it
# reorders the sequence into 64-token tiles, which means padded copies of Q, K
# and V plus an output buffer, and a [heads, tiles, tiles] fp32 score matrix.
# Per head at a 116k-row sequence that is ~30 MiB of tile buffer times four
# plus 13 MiB of scores -- 3.7 GiB across 28 rank-local heads, which is enough
# to OOM a 32 GiB card that the dense path fits on with room to spare.
#
# 512 MiB was picked to sit below the headroom a full-length H3 request leaves
# after weights and activations, not from a speed sweep: measured at 36k rows
# and 14 heads, slicing 14 heads into 4 cost ~7% of the call. Raise it if the
# card has room, and remember it bounds the *transients* -- the packed inputs
# and the output are the caller's and are not counted here.
DEFAULT_HEAD_CHUNK_BUDGET_MIB = 512
# Whether prefix keys are exempt from the budget or compete inside it.
DEFAULT_PREFIX_MODE = "exempt"
_PREFIX_MODES = ("exempt", "compete")


@dataclass(frozen=True)
class VsaH3SequenceGeometry:
    """How one request's packed rows decompose, in attention's row space.

    ``prefix_segments`` are the live row counts before the generated video,
    in packed order and split at every modality boundary (text, each reference
    block, audio). ``video_grid`` is the generated video's ``(T, H, W)`` patch
    grid, whose rows are the last live rows of the sequence, t-major and raster
    within each frame -- the layout ``minimax_h3_packed_sequence`` builds.

    Row counts, not row indices: the geometry is what makes the packed sequence
    tileable, and it is identical for every layer and every step of a request.
    """

    prefix_segments: tuple[int, ...]
    video_grid: tuple[int, int, int]

    @property
    def prefix_rows(self) -> int:
        return sum(self.prefix_segments)

    @property
    def video_rows(self) -> int:
        return math.prod(self.video_grid)

    @property
    def live_rows(self) -> int:
        return self.prefix_rows + self.video_rows


_sequence_geometry: ContextVar[VsaH3SequenceGeometry | None] = ContextVar(
    "vsa_h3_sequence_geometry", default=None
)


@contextmanager
def vsa_h3_sequence_geometry(
    geometry: VsaH3SequenceGeometry | None,
) -> Iterator[None]:
    """Publish the packed-sequence geometry the next attention calls will see.

    The rows this describes are the rows the *attention call* receives, which
    under Ulysses is the whole packed sequence rather than the caller's row
    shard -- the shard is restored to full length inside the call. Publishing a
    rank-local slice is the mistake this contract exists to name.

    The backend checks the geometry against the query it is handed and runs
    dense when they do not line up, so getting it wrong costs the speedup
    rather than corrupting the sample. A no-op for every other backend.
    """
    token = _sequence_geometry.set(geometry)
    try:
        yield
    finally:
        _sequence_geometry.reset(token)


# ``blocks.<idx>.attn`` is a DiT layer; ``token_refiner.blocks.<idx>.attn`` and
# anything else is not and stays dense.
_DIT_LAYER_PREFIX = re.compile(r"^blocks\.(\d+)\.")


def _dit_layer_index(prefix: str) -> int | None:
    match = _DIT_LAYER_PREFIX.match(prefix)
    return int(match.group(1)) if match else None


def compute_topk(sparsity: float, num_tiles: int) -> int:
    """Video key tiles kept per query tile, clamped to [1, num_tiles]."""
    return max(1, min(math.ceil((1 - sparsity) * num_tiles), num_tiles))


@dataclass(frozen=True)
class _TileGeometry:
    """The tiling of one packed sequence, reusable across layers and steps."""

    variable_block_sizes: torch.Tensor  # [n_tiles] int32, live tokens per tile
    scatter_index: torch.Tensor  # [live_rows] int64, packed row -> padded slot
    pad_index: torch.Tensor  # [padded_rows - live_rows] int64, the slots left over
    num_prefix_tiles: int
    num_video_tiles: int

    @property
    def num_tiles(self) -> int:
        return self.num_prefix_tiles + self.num_video_tiles

    @property
    def padded_rows(self) -> int:
        return self.num_tiles * BLOCK_SIZE


@functools.lru_cache(maxsize=8)
def _tile_geometry(
    prefix_segments: tuple[int, ...],
    video_grid: tuple[int, int, int],
    device: torch.device,
) -> _TileGeometry:
    """Tile the packed sequence: segment-pure prefix chunks, then video cubes.

    Cached on the geometry because it is request-static: 50 layers times N
    denoise steps reuse one set of index tensors.
    """
    prefix_sizes: list[int] = []
    for segment in prefix_segments:
        full, remainder = divmod(segment, BLOCK_SIZE)
        prefix_sizes.extend([BLOCK_SIZE] * full)
        if remainder:
            prefix_sizes.append(remainder)
    prefix_rows = sum(prefix_segments)

    ts_t, ts_h, ts_w = VSA_TILE_SIZE
    grid_t, grid_h, grid_w = video_grid
    num_video_tiles_3d = (
        math.ceil(grid_t / ts_t),
        math.ceil(grid_h / ts_h),
        math.ceil(grid_w / ts_w),
    )
    video_sizes = construct_variable_block_sizes(video_grid, num_video_tiles_3d, device)
    # Tiled position -> packed row. Prefix rows keep their packed order; video
    # rows are permuted into space-time cubes.
    tile_partition = torch.cat(
        [
            torch.arange(prefix_rows, device=device, dtype=torch.long),
            get_tile_partition_indices(video_grid, VSA_TILE_SIZE, device).to(torch.long)
            + prefix_rows,
        ]
    )
    variable_block_sizes = torch.cat(
        [
            torch.tensor(prefix_sizes, dtype=torch.int32, device=device),
            video_sizes.to(torch.int32),
        ]
    )
    # Tiled position -> slot in the padded tile buffer, skipping the pad slots
    # of partially filled tiles; composing with the inverse permutation gives
    # the map this backend actually uses, packed row -> padded slot.
    non_pad_index = get_non_pad_index(variable_block_sizes, BLOCK_SIZE)
    scatter_index = non_pad_index[torch.argsort(tile_partition)]

    # The slots a partially filled tile leaves over. Only these have to be
    # zeroed when a tile buffer is built, and there are few of them -- one
    # partial tile per prefix segment and one per video grid axis that does not
    # divide by four -- so the buffer itself can be allocated uninitialized.
    padded_rows = int(variable_block_sizes.numel()) * BLOCK_SIZE
    is_live = torch.zeros(padded_rows, dtype=torch.bool, device=device)
    is_live[non_pad_index] = True
    pad_index = torch.nonzero(~is_live, as_tuple=False).view(-1)

    geometry = _TileGeometry(
        variable_block_sizes=variable_block_sizes,
        scatter_index=scatter_index,
        pad_index=pad_index,
        num_prefix_tiles=len(prefix_sizes),
        num_video_tiles=int(video_sizes.numel()),
    )
    _validate_tile_geometry(geometry, prefix_segments, video_grid)
    return geometry


def _validate_tile_geometry(
    geometry: _TileGeometry,
    prefix_segments: tuple[int, ...],
    video_grid: tuple[int, int, int],
) -> None:
    """Fail synchronously on out-of-bounds tile geometry.

    Invariants the kernel trusts without checking: every tile's live size is in
    (0, 64]; the sizes sum to the live row count; and ``scatter_index`` maps
    each live row to exactly one non-pad slot. A violation would otherwise
    surface as an async device fault at some later kernel or collective, which
    is unattributable -- so raise here, once per cached geometry, with the
    numbers in hand.
    """
    live_rows = sum(prefix_segments) + math.prod(video_grid)
    sizes = geometry.variable_block_sizes
    if int(sizes.min()) < 1 or int(sizes.max()) > BLOCK_SIZE:
        raise ValueError(
            f"VSA-H3 tile sizes out of bounds for prefix={prefix_segments}, "
            f"video={video_grid}: min={int(sizes.min())}, max={int(sizes.max())}"
        )
    if int(sizes.sum()) != live_rows:
        raise ValueError(
            f"VSA-H3 tile sizes sum to {int(sizes.sum())}, expected {live_rows} "
            f"(prefix={prefix_segments}, video={video_grid})"
        )
    index = geometry.scatter_index
    if index.numel() != live_rows:
        raise ValueError(
            f"VSA-H3 scatter index has {index.numel()} entries for {live_rows} "
            f"live rows (prefix={prefix_segments}, video={video_grid})"
        )
    if int(index.min()) < 0 or int(index.max()) >= geometry.padded_rows:
        raise ValueError(
            f"VSA-H3 scatter index range [{int(index.min())}, {int(index.max())}] "
            f"escapes the {geometry.padded_rows}-row tile buffer "
            f"(prefix={prefix_segments}, video={video_grid})"
        )
    if int(torch.unique(index).numel()) != live_rows:
        raise ValueError(
            "VSA-H3 scatter index is not injective "
            f"(prefix={prefix_segments}, video={video_grid})"
        )


class VideoSparseAttentionH3Backend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3

    @staticmethod
    def get_impl_cls() -> type["VideoSparseAttentionH3Impl"]:
        return VideoSparseAttentionH3Impl

    @staticmethod
    def get_metadata_cls() -> type["VideoSparseAttentionH3Metadata"]:
        return VideoSparseAttentionH3Metadata

    @staticmethod
    def get_builder_cls() -> type["VideoSparseAttentionH3MetadataBuilder"]:
        return VideoSparseAttentionH3MetadataBuilder


@dataclass
class VideoSparseAttentionH3Metadata(AttentionMetadata):
    current_timestep: int


class VideoSparseAttentionH3MetadataBuilder(AttentionMetadataBuilder):
    # The base class declares __init__ abstract, so a builder that does not
    # override it cannot be instantiated at all.
    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(  # type: ignore[override]
        self, current_timestep: int, **kwargs: dict[str, Any]
    ) -> VideoSparseAttentionH3Metadata:
        return VideoSparseAttentionH3Metadata(current_timestep=current_timestep)


class VsaH3Schedule(msgspec.Struct, frozen=True):
    """When sparsity is allowed to apply, and how much of it."""

    sparsity: float
    prefix_mode: str
    skip_first_steps: int
    skip_last_steps: int
    skip_first_layers: int
    dense_layers: tuple[int, ...]
    min_seq_len: int
    head_chunk: int
    head_chunk_budget_mib: int

    @classmethod
    def from_server_args(cls) -> "VsaH3Schedule":
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        config = get_global_server_args().attention_backend_config or {}
        schedule = VsaH3Schedule(
            # `VSA_sparsity` is what the Wan VSA backend's stages already put
            # in this bag; accept it so a run can switch between the two
            # without rewriting its config.
            sparsity=float(
                config.get("sparsity", config.get("VSA_sparsity", DEFAULT_SPARSITY))
            ),
            prefix_mode=str(config.get("prefix_mode", DEFAULT_PREFIX_MODE)),
            skip_first_steps=int(
                config.get("skip_first_steps", DEFAULT_SKIP_FIRST_STEPS)
            ),
            skip_last_steps=int(config.get("skip_last_steps", DEFAULT_SKIP_LAST_STEPS)),
            skip_first_layers=int(
                config.get("skip_first_layers", DEFAULT_SKIP_FIRST_LAYERS)
            ),
            dense_layers=tuple(
                int(layer) for layer in config.get("dense_layers", DEFAULT_DENSE_LAYERS)
            ),
            min_seq_len=int(config.get("min_seq_len", DEFAULT_MIN_SEQ_LEN)),
            head_chunk=int(config.get("head_chunk", DEFAULT_HEAD_CHUNK)),
            head_chunk_budget_mib=int(
                config.get("head_chunk_budget_mib", DEFAULT_HEAD_CHUNK_BUDGET_MIB)
            ),
        )
        # sparsity == 0 keeps every tile and is the calibration setting the
        # tests use against dense attention, so it has to stay legal; 1.0 would
        # keep nothing (compute_topk clamps it to one tile, which is not what
        # anyone asking for 1.0 means).
        if not 0.0 <= schedule.sparsity < 1.0:
            raise ValueError(
                f"vsa_h3 sparsity must be in [0, 1), got {schedule.sparsity}"
            )
        if schedule.prefix_mode not in _PREFIX_MODES:
            raise ValueError(
                f"vsa_h3 prefix_mode must be one of {_PREFIX_MODES}, got "
                f"{schedule.prefix_mode!r}"
            )
        if (
            schedule.skip_first_steps < 0
            or schedule.skip_last_steps < 0
            or schedule.skip_first_layers < 0
        ):
            raise ValueError("vsa_h3 skip_first_*/skip_last_* must be non-negative")
        if schedule.min_seq_len < BLOCK_SIZE:
            raise ValueError(
                f"vsa_h3 min_seq_len must be at least {BLOCK_SIZE}, got "
                f"{schedule.min_seq_len}"
            )
        if schedule.head_chunk < 0:
            raise ValueError(
                "vsa_h3 head_chunk must be non-negative (0 sizes the slice from "
                f"head_chunk_budget_mib), got {schedule.head_chunk}"
            )
        if schedule.head_chunk_budget_mib < 1:
            raise ValueError(
                "vsa_h3 head_chunk_budget_mib must be at least 1, got "
                f"{schedule.head_chunk_budget_mib}"
            )
        return schedule


class VideoSparseAttentionH3Impl(AttentionImpl):
    """Tile-64 block-sparse attention with a dense fallback.

    One impl instance is built per attention module, so ``prefix`` fixes the
    layer for the lifetime of the object; the denoise step comes from the
    forward context and the sequence geometry from the pipeline's published
    contextvar.
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

        self.schedule = VsaH3Schedule.from_server_args()
        self.layer_idx = _dit_layer_index(prefix)
        # A layer outside the DiT stack (the token refiner) never runs sparse:
        # its sequence is the prompt alone, which no video geometry describes.
        self.layer_enabled = (
            self.layer_idx is not None
            and self.layer_idx >= self.schedule.skip_first_layers
            and self.layer_idx not in self.schedule.dense_layers
            and head_size in VideoSparseAttentionH3Backend.get_supported_head_sizes()
            # The kernel is bidirectional: it masks only pad columns, and the
            # selection has no notion of a diagonal. A causal layer would get a
            # plausible tensor computed from the wrong keys.
            and not causal
            # GQA would need K repeated before the pooled scores, which share
            # one head axis with Q.
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
                f"VSA-H3 attention: sparsity={self.schedule.sparsity} "
                f"prefix_mode={self.schedule.prefix_mode}, dense for the first "
                f"{self.schedule.skip_first_steps} denoise steps{tail}, the "
                f"first {self.schedule.skip_first_layers} DiT layers"
                + (
                    f", layers {list(self.schedule.dense_layers)}"
                    if self.schedule.dense_layers
                    else ""
                )
                + f", and sequences under {self.schedule.min_seq_len} tokens"
            )

    def _build_dense_impl(self, *, causal: bool) -> AttentionImpl:
        """The kernel used wherever the schedule excludes sparsity.

        SageAttention, for the same reason ``sparge_attn`` picks it: an
        excluded step then costs what a plain ``--attention-backend sage_attn``
        deployment already costs, so switching this backend on cannot make any
        call slower than that baseline. The resolver falls back to
        FlashAttention and then Torch SDPA where SageAttention is not built.
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

    # ---------------------------------------------------------------- schedule

    def _step_enabled(self) -> bool:
        """Whether this denoise step may sparsify, by index within the schedule."""
        context = get_forward_context()
        step = context.current_timestep
        total = get_denoise_total_steps(context)
        self._warn_if_the_cutoffs_swallow_the_schedule(total)
        if step < self.schedule.skip_first_steps:
            return False
        if self.schedule.skip_last_steps <= 0:
            return True
        if total is None:
            self._warn_tail_cutoff_has_no_schedule_length()
            return True
        return step < total - self.schedule.skip_last_steps

    def _warn_if_the_cutoffs_swallow_the_schedule(self, total: int | None) -> None:
        """The two step cutoffs together can leave no sparse step at all.

        The warmup default of 10 assumes the 50-step schedule. A VSA-distilled
        or turbo checkpoint runs 4 to 9 steps, where every index is below the
        cutoff and this backend silently degrades into dense attention -- a
        config error worth a line in the log rather than an unexplained absence
        of speedup.
        """
        if total is None:
            return
        dense = self.schedule.skip_first_steps + self.schedule.skip_last_steps
        if dense < total:
            return
        logger.warning_once(
            f"VSA-H3 attention never activates: skip_first_steps="
            f"{self.schedule.skip_first_steps} + skip_last_steps="
            f"{self.schedule.skip_last_steps} covers all {total} denoise steps "
            f"this request runs, so every step takes the dense path. A "
            f"checkpoint distilled at this sparsity wants skip_first_steps=0."
        )

    def _warn_tail_cutoff_has_no_schedule_length(self) -> None:
        """``skip_last_steps`` set but nobody published the schedule length.

        Fail open rather than closed: without a length the last step cannot be
        identified, and running every step dense would disable the backend
        wholesale over a missing integer. It must be loud -- a silently
        inactive cutoff reads as evidence that the tail is not the problem.
        """
        logger.warning_once(
            f"VSA-H3 skip_last_steps={self.schedule.skip_last_steps} is "
            "inactive: no denoise schedule length was published, so the last "
            "step cannot be identified and every step stays sparse. The "
            "pipeline stage owning the loop must wrap it in "
            "denoise_total_steps()."
        )

    def _geometry_for(self, rows: int) -> VsaH3SequenceGeometry | None:
        """The published geometry, if it describes ``rows`` live rows.

        A mismatch is not an error: cross attention, the token refiner, and any
        model that publishes nothing all land here, and the answer for all of
        them is dense attention. It is worth one warning when a geometry *was*
        published and does not fit, because that is a wiring bug rather than a
        layer this backend does not serve.
        """
        geometry = _sequence_geometry.get()
        if geometry is None:
            return None
        if geometry.live_rows != rows:
            logger.warning_once(
                "VSA-H3 attention is running dense: the published geometry "
                f"covers {geometry.live_rows} live rows (prefix="
                f"{geometry.prefix_segments}, video={geometry.video_grid}) but "
                f"attention was handed {rows}. Publish the geometry for the "
                "rows attention sees, not the caller's row shard."
            )
            return None
        return geometry

    def _sparse_ready(self, query: torch.Tensor, key: torch.Tensor, rows: int) -> bool:
        return (
            self.layer_enabled
            and rows >= self.schedule.min_seq_len
            # The kernel accumulates P in bf16 unconditionally; fp16 inputs
            # would run but at a precision the dense fallback does not have.
            and query.dtype is torch.bfloat16
            and key.dtype is torch.bfloat16
            and self._step_enabled()
        )

    # --------------------------------------------------------------- attention

    def _head_chunk_for(
        self, geometry: _TileGeometry, heads: int, itemsize: int
    ) -> int:
        """Heads per pass, from the configured budget when none was given.

        What a pass costs, per head: four ``padded_rows x head_dim`` tile
        buffers (Q, K, V and the output) plus one ``tiles x tiles`` fp32 score
        matrix. Both terms grow with the sequence, and the second grows with
        its square, so a fixed head count that is right at 36k rows is wrong at
        116k. Sizing the slice from bytes instead keeps the transients flat
        across resolutions, which is what stops a long request from OOMing a
        card the dense path fits on.
        """
        explicit = self.schedule.head_chunk
        if explicit > 0:
            return explicit
        per_head = (
            4 * geometry.padded_rows * self.head_size * itemsize
            + geometry.num_tiles * geometry.num_tiles * 4
        )
        budget = self.schedule.head_chunk_budget_mib * 1024 * 1024
        return max(1, min(heads, budget // max(per_head, 1)))

    def _head_slices(self, heads: int, chunk: int) -> Iterator[tuple[int, int]]:
        if chunk >= heads:
            yield 0, heads
            return
        for start in range(0, heads, chunk):
            yield start, min(start + chunk, heads)

    def _tile(self, x: torch.Tensor, geometry: _TileGeometry) -> torch.Tensor:
        """``[S, H, D]`` live rows -> ``[1, H, padded_rows, D]`` tile buffer.

        Pad slots stay zero. They are never read as queries (their output rows
        are dropped on the way back) and never read as keys (the kernel masks
        every column past a tile's live size), so their contents only have to
        be finite.
        """
        heads, dim = x.shape[-2], x.shape[-1]
        buffer = torch.empty(
            (heads, geometry.padded_rows, dim), dtype=x.dtype, device=x.device
        )
        # Every live slot is written below, so only the leftovers of partial
        # tiles need clearing -- a few thousand rows against the whole buffer.
        buffer[:, geometry.pad_index] = 0
        buffer[:, geometry.scatter_index] = x.transpose(0, 1)
        return buffer.unsqueeze(0)

    def _q2k_for_video(
        self,
        scores: torch.Tensor,
        geometry: _TileGeometry,
        topk: int,
    ) -> torch.Tensor:
        """Key tiles each video query tile attends to: ``[1, H, n_video, W]``.

        ``scores`` is the pooled-Q.K matrix for this head slice, restricted to
        video query tiles. In ``exempt`` mode the budget is spent on video keys
        alone and every prefix key is appended; in ``compete`` mode the same
        total number of tiles is chosen from the whole row, so the two modes
        run the same number of tiles through the kernel.
        """
        heads, num_video = scores.shape[1], scores.shape[2]
        prefix = geometry.num_prefix_tiles
        if self.schedule.prefix_mode == "compete":
            budget = min(topk + prefix, geometry.num_tiles)
            return scores.topk(budget, dim=-1).indices.to(torch.int32)
        video = scores[..., prefix:].topk(topk, dim=-1).indices.to(torch.int32) + prefix
        if prefix == 0:
            return video.contiguous()
        prefix_cols = torch.arange(
            prefix, device=scores.device, dtype=torch.int32
        ).expand(1, heads, num_video, prefix)
        return torch.cat([prefix_cols, video], dim=-1)

    def _sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        geometry: _TileGeometry,
    ) -> torch.Tensor:
        """Live ``[S, H, D]`` rows -> same shape, attention over tiles.

        Two launches of the same kernel rather than one: prefix query tiles
        take every key tile and video query tiles take their selection, so
        splitting them keeps the index list as narrow as the selection instead
        of as wide as the sequence. The kernel indexes its metadata from the
        first query tile it is given, which is what makes the second launch a
        plain suffix slice.

        Everything inside the head loop -- the tile buffers included -- is
        sliced by head, so ``head_chunk`` bounds the transients rather than
        just the score matrix. The gathered output goes straight into the
        packed result, so no full-width tile buffer outlives a slice.
        """
        heads = query.shape[-2]
        prefix_rows = geometry.num_prefix_tiles * BLOCK_SIZE
        sizes = geometry.variable_block_sizes
        topk = compute_topk(self.schedule.sparsity, geometry.num_video_tiles)
        out = torch.empty_like(query)
        chunk = self._head_chunk_for(geometry, heads, query.element_size())

        logger.info_once(
            f"VSA-H3 attention active: {geometry.num_tiles} tiles "
            f"({geometry.num_prefix_tiles} prefix + {geometry.num_video_tiles} "
            f"video), keeping {topk}/{geometry.num_video_tiles} video tiles per "
            f"video query tile, heads={heads} in slices of {chunk}"
        )

        for start, stop in self._head_slices(heads, chunk):
            width = stop - start
            q_tiled = self._tile(query[:, start:stop], geometry)
            k_tiled = self._tile(key[:, start:stop], geometry)
            v_tiled = self._tile(value[:, start:stop], geometry)
            out_tiled = torch.empty_like(q_tiled)

            if geometry.num_prefix_tiles:
                # Prefix queries are dense: every key tile, in order.
                dense_index = (
                    torch.arange(
                        geometry.num_tiles, device=query.device, dtype=torch.int32
                    )
                    .expand(1, width, geometry.num_prefix_tiles, geometry.num_tiles)
                    .contiguous()
                )
                dense_num = torch.full(
                    (1, width, geometry.num_prefix_tiles),
                    geometry.num_tiles,
                    device=query.device,
                    dtype=torch.int32,
                )
                out_tiled[:, :, :prefix_rows] = block_sparse_attn_forward(
                    q_tiled[:, :, :prefix_rows],
                    k_tiled,
                    v_tiled,
                    dense_index,
                    dense_num,
                    sizes,
                )
                del dense_index, dense_num

            scores = _pooled_scores(q_tiled, k_tiled, sizes)
            q2k_index = self._q2k_for_video(
                scores[:, :, geometry.num_prefix_tiles :], geometry, topk
            )
            # The score matrix is the largest transient here; let it go before
            # the kernel allocates.
            del scores
            q2k_num = torch.full(
                (1, width, geometry.num_video_tiles),
                q2k_index.shape[-1],
                device=query.device,
                dtype=torch.int32,
            )
            out_tiled[:, :, prefix_rows:] = block_sparse_attn_forward(
                q_tiled[:, :, prefix_rows:],
                k_tiled,
                v_tiled,
                q2k_index.contiguous(),
                q2k_num,
                sizes,
            )
            del q2k_index, q2k_num, q_tiled, k_tiled, v_tiled

            out[:, start:stop] = out_tiled[0][:, geometry.scatter_index].transpose(0, 1)
            del out_tiled

        return out

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        geometry: VsaH3SequenceGeometry,
    ) -> torch.Tensor:
        return self._sparse_attention(
            query,
            key,
            value,
            _tile_geometry(geometry.prefix_segments, geometry.video_grid, query.device),
        )

    # ----------------------------------------------------------------- entries

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: VideoSparseAttentionH3Metadata | None = None,
    ) -> torch.Tensor:
        """query/key/value: ``[B, S, H, D]``, one packed sequence per batch row."""
        rows = query.shape[-3]
        geometry = None
        if query.shape[0] == 1 and self._sparse_ready(query, key, rows):
            geometry = self._geometry_for(rows)
        if geometry is None:
            return self.dense_impl.forward(query, key, value, attn_metadata)
        return self._attend(query[0], key[0], value[0], geometry).unsqueeze(0)

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

        H3 packs one live document as ``(0, used, total)``: ``[0, used)`` are
        real rows and ``[used, total)`` is 64-aligned tail padding whose output
        must stay zero so downstream masked rows stay inactive. Any other
        document layout is not a packed H3 sequence and runs dense.
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

        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(value) for value in cu_seqlens.tolist())
        )
        if len(bounds) != 3:
            return all_dense()
        start, used, total = bounds
        if start != 0 or used > total or total != query.shape[0]:
            return all_dense()
        if not self._sparse_ready(query, key, used):
            return all_dense()
        geometry = self._geometry_for(used)
        if geometry is None:
            return all_dense()

        live_out = self._attend(query[:used], key[:used], value[:used], geometry)
        if used == query.shape[0]:
            return live_out
        out = torch.zeros_like(query)
        out[:used] = live_out
        return out


def _pooled_scores(
    query: torch.Tensor, key: torch.Tensor, variable_block_sizes: torch.Tensor
) -> torch.Tensor:
    """Tile-mean Q.K scores: ``[1, H, S, D]`` tile buffers -> ``[1, H, n, n]``.

    fp32 throughout, and the mean is exact: pad slots are zero and were never
    written, so a plain sum divided by the tile's live size is the masked mean
    without a mask or an fp32 copy of the inputs.
    """
    heads, padded_rows, dim = query.shape[1], query.shape[2], query.shape[3]
    tiles = padded_rows // BLOCK_SIZE
    sizes = variable_block_sizes.view(1, 1, -1, 1)
    q_pooled = query.view(1, heads, tiles, BLOCK_SIZE, dim).sum(
        dim=3, dtype=torch.float32
    ) / sizes.to(torch.float32)
    k_pooled = key.view(1, heads, tiles, BLOCK_SIZE, dim).sum(
        dim=3, dtype=torch.float32
    ) / sizes.to(torch.float32)
    return torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) / math.sqrt(dim)


__all__ = [
    "VideoSparseAttentionH3Backend",
    "VideoSparseAttentionH3Impl",
    "VideoSparseAttentionH3Metadata",
    "VideoSparseAttentionH3MetadataBuilder",
    "VsaH3Schedule",
    "VsaH3SequenceGeometry",
    "compute_topk",
    "vsa_h3_sequence_geometry",
]
