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

Two things keep it affordable on a card the dense path already fills. Only K is
laid out in tile order -- Q, V and the output are read and written in place
through a slot-to-row index, so tiling costs one buffer rather than four -- and
the head slice is sized from ``head_chunk_budget_mib`` so the remaining
transients, K and the score matrix that grows with the square of the sequence,
stay inside a budget instead of scaling with the resolution. Q.K then runs on
INT8 tensor cores by default, SageAttention-style, which is where most of the
speed comes from; ``{"quantize": false}`` returns the kernel to bf16, and
``{"quantize_pv": true}`` takes P.V to FP8 as well -- measured, off by default,
and the trade is written out at ``DEFAULT_QUANTIZE_PV``.
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
    pool_tiles,
    quantize_tiles,
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
# Measured here on one RTX 5090 against bf16 SDPA over the whole backend call
# -- selection, the dense prefix launch and the gather included -- with INT8
# Q.K on: 36k rows and 14 heads 4.99x at 0.9 and 6.24x at 0.95; 116k rows and
# 28 heads 7.58x and 11.07x. Op-level numbers: attention is only part of a
# denoise step, so the end-to-end gain is smaller.
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
# Whether Q.K runs on INT8 tensor cores, SageAttention-style: K is quantized per
# tile with its per-channel mean removed first (which softmax cancels exactly),
# Q per tile in registers, and P.V stays bf16. Measured on one RTX 5090 at 36k
# and 116k rows: 1.85x and 1.94x on the attention op, for 1.3% of the output's
# norm in added error -- SageAttention's own budget, and the same error the
# dense fallback this backend already falls back to carries. Turn it off to
# separate a quality question from the sparsity, which is the much larger term.
DEFAULT_QUANTIZE = True
# Whether P.V runs on FP8 tensor cores as well -- the other half of what
# SageAttention2 does. Off by default, and the reason is the trade rather than
# the mechanism: measured on one RTX 5090 it buys 11-16% of the attention op
# (5.06x -> 5.84x at 36k rows and sparsity 0.9, 7.66x -> 8.54x at 116k) for up
# to 3.9% of the output's norm, against 1.3% for INT8 Q.K alone. Three times
# the error for a tenth of the speed is a worse bargain than the first GEMM's,
# and it is not one to take on someone's renders unmeasured.
#
# That 3.9% is the worst case, and it is the *synthetic* case: it was measured
# on zero-mean random V, where e4m3's three mantissa bits are all the precision
# there is. Real attention values carry a per-channel bias, and against one the
# same path measures 0.10-0.19% -- because V is centred before it is quantized
# and the mean added back after the softmax normalisation, which is exact and
# which is what that centring is for. If a measured render says the error is
# invisible on this model, this is a one-key change.
DEFAULT_QUANTIZE_PV = False
# The largest finite e4m3 value; V's scale is its amax over this.
FP8_MAX = 448.0
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
    tile_rows: torch.Tensor  # [padded_rows] int32, the packed row each slot holds

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

    # The inverse of ``scatter_index``, and the only one of the two the kernels
    # use: they read and write the packed rows in place and address them by
    # tiled slot. Pad slots hold 0 and are masked, so the value is never read.
    tile_rows = torch.zeros(padded_rows, dtype=torch.int32, device=device)
    tile_rows[scatter_index] = torch.arange(
        scatter_index.numel(), dtype=torch.int32, device=device
    )

    geometry = _TileGeometry(
        variable_block_sizes=variable_block_sizes,
        scatter_index=scatter_index,
        pad_index=pad_index,
        tile_rows=tile_rows,
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
    quantize: bool
    quantize_pv: bool

    @classmethod
    def from_server_args(cls) -> "VsaH3Schedule":
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        config = get_global_server_args().attention_backend_config or {}
        quantize = bool(config.get("quantize", DEFAULT_QUANTIZE))
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
            quantize=quantize,
            # ``quantize`` is the master switch: turning it off has to give the
            # reference kernel, not one quantized GEMM out of two.
            quantize_pv=quantize
            and bool(config.get("quantize_pv", DEFAULT_QUANTIZE_PV)),
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

        What a pass costs, per head: the key buffer laid out in tile order (one
        byte per element quantized, two otherwise), the fp8 value buffer when
        P.V is quantized too, the two pooled tile means,
        the ``tiles x tiles`` fp32 score matrix and the index list the selection
        produces. Q, V and the output are the caller's tensors, read and written
        in place, and cost nothing here.

        The score matrix grows with the square of the sequence, so a fixed head
        count that is right at 36k rows is wrong at 116k. Sizing the slice from
        bytes instead keeps the transients flat across resolutions, which is
        what stops a long request from OOMing a card the dense path fits on.
        """
        explicit = self.schedule.head_chunk
        if explicit > 0:
            return explicit
        key_bytes = 1 if self.schedule.quantize else itemsize
        tiles, video_tiles = geometry.num_tiles, geometry.num_video_tiles
        topk = compute_topk(self.schedule.sparsity, video_tiles)
        compete = self.schedule.prefix_mode == "compete"
        # Only video query tiles are scored, against video keys alone unless
        # the prefix competes; top-k then holds fp32 values and int64 indices
        # over the chosen width, an int32 copy of those indices, and the list
        # the kernel is finally handed.
        score_columns = tiles if compete else video_tiles
        chosen = min(topk + tiles, tiles) if compete else topk
        value_bytes = 1 if self.schedule.quantize_pv else 0
        per_head = (
            geometry.padded_rows * self.head_size * key_bytes  # tiled K
            + geometry.padded_rows * self.head_size * value_bytes  # the fp8 V
            + 2 * tiles * self.head_size * 4  # the two pooled tile means
            + video_tiles * score_columns * 4  # the score matrix
            + video_tiles * chosen * 16  # top-k's values, indices and copies
            + video_tiles * (chosen + geometry.num_prefix_tiles) * 4  # the list
        )
        budget = self.schedule.head_chunk_budget_mib * 1024 * 1024
        return max(1, min(heads, budget // max(per_head, 1)))

    def _head_slices(self, heads: int, chunk: int) -> Iterator[tuple[int, int]]:
        if chunk >= heads:
            yield 0, heads
            return
        for start in range(0, heads, chunk):
            yield start, min(start + chunk, heads)

    def _tile_key(self, key: torch.Tensor, geometry: _TileGeometry) -> torch.Tensor:
        """``[S, H, D]`` live rows -> ``[H, padded_rows, D]``, in tile order.

        K is the one tensor worth materialising: every query tile reads all of
        it, transposed, so gathering it in the inner loop would pay the gather
        per (query tile, key tile) instead of once. Q, V and the output are read
        and written in place through ``tile_rows``.

        Pad slots are zeroed, not left undefined -- they are masked out of the
        softmax, but a garbage row still has to be finite. Only the leftovers of
        partial tiles need clearing, a few thousand rows against the whole
        buffer, so the allocation itself is uninitialised.
        """
        heads, dim = key.shape[-2], key.shape[-1]
        buffer = torch.empty(
            (heads, geometry.padded_rows, dim), dtype=key.dtype, device=key.device
        )
        buffer[:, geometry.pad_index] = 0
        buffer[:, geometry.scatter_index] = key.transpose(0, 1)
        return buffer

    def _q2k_for_video(
        self,
        q_pooled: torch.Tensor,
        k_pooled: torch.Tensor,
        geometry: _TileGeometry,
        topk: int,
    ) -> torch.Tensor:
        """Key tiles each video query tile attends to: ``[H, n_video, W]``.

        The pooled tile means come in whole; the scores are formed only for the
        rows and columns the selection can actually choose from. In ``exempt``
        mode that is video-by-video -- prefix keys are kept unconditionally and
        prefix queries are dense, so neither needs a score at all, and skipping
        them keeps the largest transient here to ``n_video ** 2``. In ``compete``
        mode prefix keys enter the same budget, so the columns come back.

        The softmax scale is left out: it is a positive constant and top-k only
        ranks, so applying it would cost a second score-sized tensor to change
        nothing.
        """
        prefix = geometry.num_prefix_tiles
        q_video = q_pooled[:, prefix:]
        if self.schedule.prefix_mode == "compete":
            budget = min(topk + prefix, geometry.num_tiles)
            scores = torch.matmul(q_video, k_pooled.transpose(-2, -1))
            return scores.topk(budget, dim=-1).indices.to(torch.int32).contiguous()

        scores = torch.matmul(q_video, k_pooled[:, prefix:].transpose(-2, -1))
        video = scores.topk(topk, dim=-1).indices.to(torch.int32) + prefix
        del scores
        if prefix == 0:
            return video.contiguous()
        prefix_cols = torch.arange(
            prefix, device=video.device, dtype=torch.int32
        ).expand(video.shape[0], video.shape[1], prefix)
        return torch.cat([prefix_cols, video], dim=-1)

    def _sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        geometry: _TileGeometry,
    ) -> torch.Tensor:
        """Live ``[S, H, D]`` rows -> same shape, attention over tiles.

        Two launches of the same kernel rather than one: prefix query tiles take
        every key tile and video query tiles take their selection, so splitting
        them keeps the index list as narrow as the selection instead of as wide
        as the sequence. Both write into one output, each covering the rows of
        the tiles it was given.

        Head slices are views, not copies -- every kernel here takes explicit
        strides -- so slicing costs nothing but the launches it adds.
        """
        heads = query.shape[-2]
        sizes = geometry.variable_block_sizes
        tile_rows = geometry.tile_rows
        tiles, prefix_tiles = geometry.num_tiles, geometry.num_prefix_tiles
        topk = compute_topk(self.schedule.sparsity, geometry.num_video_tiles)
        quantize = self.schedule.quantize
        quantize_pv = self.schedule.quantize_pv
        out = torch.empty_like(query)
        chunk = self._head_chunk_for(geometry, heads, query.element_size())

        logger.info_once(
            f"VSA-H3 attention active: {tiles} tiles ({prefix_tiles} prefix + "
            f"{geometry.num_video_tiles} video), keeping {topk}/"
            f"{geometry.num_video_tiles} video tiles per video query tile, "
            f"{'INT8' if quantize else 'bf16'} Q.K and "
            f"{'FP8' if quantize_pv else 'bf16'} P.V, heads={heads} in slices "
            f"of {chunk}"
        )

        for start, stop in self._head_slices(heads, chunk):
            q_slice = query[:, start:stop]
            k_slice = key[:, start:stop]
            v_slice = value[:, start:stop]
            out_slice = out[:, start:stop]

            key_scale = None
            if quantize:
                # K's per-channel mean over the live rows. Subtracting it before
                # quantizing shifts every logit in a row by the same -q.km,
                # which softmax cancels exactly, and it is what keeps the int8
                # range on the part of K that varies.
                key_tiled, key_scale = quantize_tiles(
                    k_slice,
                    tile_rows,
                    sizes,
                    tiles,
                    k_slice.mean(dim=0, dtype=torch.float32).contiguous(),
                )
            else:
                key_tiled = self._tile_key(k_slice, geometry)

            value_scale = value_mean = None
            value_operand = v_slice
            if quantize_pv:
                # V is centred per channel before it is quantized, and the mean
                # is added back after the softmax normalisation -- exact,
                # because the weights sum to one, and it is what stops e4m3's
                # three mantissa bits from being spent on a bias every row
                # shares. The scale is bounded rather than measured on the
                # centred tensor: amax|v - vm| <= amax|v| + amax|vm|, and a
                # slightly loose fp8 scale costs no precision because e4m3
                # carries its own exponent. Both are computed without an
                # intermediate -- abs() or a subtraction would materialise the
                # full-width copy of V this backend exists not to allocate.
                value_mean = v_slice.mean(dim=0, dtype=torch.float32).contiguous()
                value_scale = (
                    (
                        torch.maximum(
                            v_slice.amax(dim=(0, 2)), v_slice.amin(dim=(0, 2)).neg()
                        ).to(torch.float32)
                        + value_mean.abs().amax(dim=-1)
                    )
                    / FP8_MAX
                ).contiguous()
                value_operand, _ = quantize_tiles(
                    v_slice,
                    tile_rows,
                    sizes,
                    tiles,
                    channel_mean=value_mean,
                    fixed_scale=value_scale,
                    dtype=torch.float8_e4m3fn,
                )

            # Selection runs on the unquantized rows: it only has to rank tiles,
            # and pooling is a mean over 64 rows, so it is cheap either way.
            q2k_index = self._q2k_for_video(
                pool_tiles(q_slice, tile_rows, sizes, tiles),
                pool_tiles(k_slice, tile_rows, sizes, tiles),
                geometry,
                topk,
            )
            q2k_num = torch.full(
                (stop - start, geometry.num_video_tiles),
                q2k_index.shape[-1],
                device=query.device,
                dtype=torch.int32,
            )

            if prefix_tiles:
                # Prefix queries are dense: every key tile, in order.
                dense_index = (
                    torch.arange(tiles, device=query.device, dtype=torch.int32)
                    .expand(stop - start, prefix_tiles, tiles)
                    .contiguous()
                )
                dense_num = torch.full(
                    (stop - start, prefix_tiles),
                    tiles,
                    device=query.device,
                    dtype=torch.int32,
                )
                block_sparse_attn_forward(
                    q_slice,
                    key_tiled,
                    value_operand,
                    out_slice,
                    tile_rows,
                    dense_index,
                    dense_num,
                    sizes,
                    0,
                    key_scale,
                    value_scale,
                    value_mean,
                )
                del dense_index, dense_num

            block_sparse_attn_forward(
                q_slice,
                key_tiled,
                value_operand,
                out_slice,
                tile_rows,
                q2k_index,
                q2k_num,
                sizes,
                prefix_tiles,
                key_scale,
                value_scale,
                value_mean,
            )
            del q2k_index, q2k_num, key_tiled, key_scale
            del value_scale, value_mean, value_operand

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
