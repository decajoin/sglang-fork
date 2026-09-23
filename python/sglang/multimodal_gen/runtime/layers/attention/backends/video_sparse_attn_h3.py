# SPDX-License-Identifier: Apache-2.0
"""VSA-H3: video sparse attention for MiniMax-H3's packed mixed-modality DiT.

Ported from FastVideo's ``video_sparse_attn_h3`` backend (hao-ai-lab/FastVideo,
`VSA <https://arxiv.org/abs/2505.13389>`_) onto this tree's packed-varlen
attention contract. H3 runs one joint bidirectional attention over
``[text | reference/condition rows | audio | generated video | padding]``, so
this backend differs from the Wan-tuned ``video_sparse_attn`` in what it tiles
and what it protects:

- The unit is a 64-token tile. Every *picture* -- each reference image, each
  reference video, and the generated video -- is tiled in 3D as ``(4, 4, 4)``
  over its own ``(T, H, W)`` patch grid, so one tile is a small space-time cube
  whose tokens actually attend to each other; a block with too few frames to
  fill that cube is cut one frame deep instead, so a reference still does not
  spend three quarters of every tile on padding (``_tile_shape_for``). Text and
  audio are tiled into segment-pure 64-row chunks -- a tile never straddles a
  modality boundary, because a tile is both the unit of selection and the unit
  of pooling, and pooling text with audio produces a score that describes
  neither.
- Selection is per (head, query tile): pooled Q.K over tiles, then the top
  ``(1 - sparsity)`` fraction of *picture* key tiles. Text and audio keys are
  kept by every query -- they are a few percent of the sequence and lose every
  budget contest they enter, which is what ``sparge_attn`` protects them for
  and what it saw corrupt audio when it did not. ``{"prefix_mode": "compete"}``
  makes them compete under a FLOP-matched budget instead; it is the ablation,
  not the default.
- Reference pictures are *not* protected: they are tiled and ranked alongside
  the generated video, and their queries sparsify with it. Protecting them is
  what this backend shipped with, and it is affordable only while they are a
  few percent of the sequence -- a ref2va request conditioned on a reference
  video packs 30-40% of its rows there, which capped the achievable speedup at
  1.57x and measured 1.68x *slower* than ``sparge_attn``, which sparsifies the
  same rows. ``DEFAULT_SPARSIFY_REFERENCES`` carries the measurement and
  ``{"sparsify_references": false}`` restores the protected behaviour.
- Text and audio *queries* are always dense. They are a small minority of the
  rows, so the FLOPs saved by sparsifying them are noise against the risk to
  prompt adherence and audio.

The gate-compress branch of upstream VSA is not ported: it needs a trained
``to_gate_compress`` matrix per layer, which no MiniMax-H3 checkpoint this tree
loads carries. Without it VSA is exactly top-k block-sparse attention, which is
what the sparse branch computes here.

**This is not free accuracy.** VSA is a *trainable* sparse attention: FastVideo
runs it at ``sparsity=0.9`` against a VSA-distilled checkpoint where 0.9 is the
policy the student was trained under. Against a stock MiniMax-H3 checkpoint the
same setting is training-free block sparsity and its quality is unmeasured
here, and the warmup cutoff that would hedge against that is off by default
(``DEFAULT_SKIP_FIRST_STEPS``). Measure against a dense render before trusting
a sparsity.

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
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import msgspec
import torch

from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
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
    get_non_pad_index,
    get_tile_partition_indices,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.vsa_h3 import (
    BLOCK_SIZE,
    block_sparse_attn_forward,
    pool_tiles,
    quantize_tiles,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.vsa_h3.sm120_sage import (
    head_slice_bytes as sm120_sage_head_slice_bytes,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.vsa_h3.sm120_sage import (
    sm120_sage_attention,
    sm120_sage_available,
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

# The tile a visual block with fewer than ``VSA_TILE_SIZE[0]`` latent frames is
# cut into: one frame deep, and square enough in H/W to still hold 64 tokens.
# A reference image is a single latent frame, and cutting a still on the
# 4-frame cube yields tiles of 16 live rows in a 64-row slot -- four times the
# tiles, three quarters of every slot wasted, and a score matrix that grows
# with the square of that tile count. Per frame keeps a tile full and keeps its
# rows spatially adjacent, which is what the pooled score describes.
_FRAME_TILE_SIZE = (1, 8, 8)
assert math.prod(_FRAME_TILE_SIZE) == BLOCK_SIZE

# Fraction of *video* key tiles each video query tile drops. 0.9 is FastVideo's
# trained policy for the VSA-distilled H3 preview and its reference default.
# Measured here on one RTX 5090 against bf16 SDPA over the whole backend call
# -- selection, the dense prefix launch and the gather included -- with INT8
# Q.K on: 36k rows and 14 heads 4.99x at 0.9 and 6.24x at 0.95; 116k rows and
# 28 heads 7.58x and 11.07x. Op-level numbers: attention is only part of a
# denoise step, so the end-to-end gain is smaller.
DEFAULT_SPARSITY = 0.9
# Leading denoise forwards kept dense. 0: every step sparsifies.
#
# This is a deliberate departure from the one measurement in this tree that
# speaks to it. subblock_sparse measured on this same model that lowering the
# warmup from 10 of 50 to 5 halves cosine against the dense render and visibly
# re-frames the shot, and the early steps are where the sample's layout is
# settled and approximation is tolerated worst. The default is 0 all the same:
# a warmup gives back the whole first fifth of the schedule, the cutoff was
# never re-swept for this backend's block map, and a VSA-distilled checkpoint
# trained at this sparsity wants 0 anyway. Set ``{"skip_first_steps": 10}`` for
# the measured-conservative schedule, and measure a render before trusting
# either one on a new checkpoint.
DEFAULT_SKIP_FIRST_STEPS = 0
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
# The same budget, for the FlashInfer Sage path, which needs its own because
# its operands cost 9 bytes per padded row-element against the Triton path's 2:
# it lays out Q, K, V and the output in tile order where that path lays out
# only K. Holding both to 512 MiB would slice it four times as finely, and the
# slice count is what the fixed cost per pass multiplies -- measured at 116k
# rows and 28 rank-local heads, three passes of ten spend 15.6 ms outside the
# kernel where fourteen passes of two spend 21.3, which is most of what the
# faster kernel had won. 1536 MiB buys back the three-pass shape at the
# sequence length this backend is for; it is transients, and it is worth
# checking against the headroom a full-length request actually leaves.
DEFAULT_SAGE_HEAD_CHUNK_BUDGET_MIB = 1536
# Below this many tiled rows the Sage path runs the Triton kernel instead.
#
# Its win is in the launches and its cost is in laying the operands out, and
# the first grows with the square of the sequence while the second grows with
# the sequence. Measured end to end on one RTX 5090 at 28 rank-local heads and
# sparsity 0.9, against the Triton path on the same rows: 0.94x at 16k rows,
# 1.09x at 24k, 1.08x at 34k, 1.18x at 42k, 1.26x at 58k, 1.34x at 91k, 1.33x
# at 116k. The crossing is between 16k and 24k, but the middle of that range
# buys single-digit percent for 40-45% more transient memory, which is a bad
# trade on a card this backend is already sized to fit. 32k is where the
# speedup is worth the footprint rather than where it first exists.
DEFAULT_SAGE_MIN_SEQ_LEN = 32768
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
# Which kernel executes the selection this backend computes.
#
# ``triton`` is the vendored forward in ``vsa_h3/kernels.py``: no dependency, no
# arch-specific build, compute capability 8.0 and up, head_dim 64 or 128, and
# it gathers Q and V through ``tile_rows`` rather than materialising them.
#
# ``flashinfer`` is FlashInfer's CuTe-DSL SM120 Sage kernel, which runs the same
# selection 1.6x faster (515 TFLOPS against 325, measured at 116k rows and 28
# rank-local heads, flat across head-slice widths) for the same error against an
# fp32 reference (3.82% against 3.78%). It needs an SM120 device, head_dim 128,
# and both quantization switches on, because INT8 Q.K with FP8 P.V is the only
# arithmetic it implements -- a request for bf16 P.V is a request for a
# different function, and this backend answers it with the kernel that computes
# it rather than the one that is fast. It also has to lay Q, K, V and the output
# out in tile order, which is memory the Triton path does not spend, so the head
# slice is sized differently for it.
#
# ``auto`` takes ``flashinfer`` when every one of those holds and ``triton``
# otherwise, which is what keeps this a pure speedup rather than a behaviour
# change: the defaults leave P.V in bf16, so a default deployment stays on the
# Triton path until someone asks for FP8 P.V.
DEFAULT_KERNEL = "auto"
_KERNELS = ("auto", "triton", "flashinfer")
# Whether prefix keys are exempt from the budget or compete inside it.
DEFAULT_PREFIX_MODE = "exempt"
_PREFIX_MODES = ("exempt", "compete")
# Whether reference pictures -- reference images and reference videos -- are
# tiled and sparsified like the generated video, or protected as prefix.
#
# On by default because protecting them is only affordable while they are
# small, and in ref2va they are not. The prefix-protection rule was written for
# text and audio, which are a few percent of the sequence; a ref2va request
# conditioned on a reference *video* packs 30-40% of its rows into that
# protected block, and protecting them costs more than the sparsity saves:
# measured over the 51-case official-prompt suite at sparsity 0.9, protected
# reference rows left video-by-video attention as 3.6% of the dense FLOPs while
# the protected block alone accounted for 64%, capping the achievable speedup
# at 1.57x and landing 1.68x *slower* than sparge_attn, which sparsifies the
# same rows. Sparsifying them restores the budget to what a picture-dominated
# sequence should get.
#
# This is a quality trade, not a free win: a reference tile a query does not
# select is a reference detail that query cannot see. sparge_attn has run this
# way on this model and the renders are usable, which is the evidence this
# default rests on -- measure a render before trusting it on a new checkpoint,
# and set ``{"sparsify_references": false}`` to get the protected behaviour
# back for the comparison.
DEFAULT_SPARSIFY_REFERENCES = True
# Whether the reference pictures Qwen3-VL packs into the *text* conditioning are
# tiled and sparsified too.
#
# Every reference image is encoded twice: once by the VAE into the picture rows
# above, and once by Qwen3-VL as a vision block inside the prompt, at the same
# 32px-per-token granularity and so at very nearly the same row count. The
# second copy wears the TEXT tag, which is why it is protected here and why
# ``sparge_attn``, which reads the per-row tags the presentation writes, has
# always sparsified it.
#
# Off by default because it is not the same trade as the first copy. Those
# tokens are contextualised: inside Qwen they have already attended to the
# prompt and to the other references, so they carry which subject an
# instruction is about and not only what it looks like, which makes dropping
# one closer to dropping prompt tokens than to dropping pixels. The failure
# mode is the quiet one -- weakened prompt adherence that never looks broken --
# so this wants a measured render behind it, not a plausible argument. What it
# buys, measured on one RTX 5090 at 14 heads and D=128 over a 12s 768p 16:9
# request with six reference images at short edge 1024: the attention op goes
# from 92.7 to 46.0 ms, and the advantage over ``sparge_attn`` stops decaying
# with the reference count (1.07x -> 2.19x at six images, against 1.66x ->
# 2.00x at one).
DEFAULT_SPARSIFY_TEXT_VISUALS = False


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

    ``reference_visuals`` names which of those prefix segments are *pictures* --
    ``(segment index, (T, H, W) patch grid)`` for each reference image or
    reference video, whose rows the packed layout puts before the generated
    video. A named segment is tiled and sparsified like the generated video
    instead of being protected as prefix; see ``sparsify_references``. Segments
    it does not name (the prompt, every audio block) stay protected. Leaving it
    empty is the pre-existing behaviour and is what a builder that publishes no
    grids gets.

    ``text_visuals`` names the same pictures where Qwen3-VL packed a second copy
    of each one *inside* the prompt, as ``(start, rows, (T, H, W))`` offsets
    into the first prefix segment rather than as segment indices. They are held
    apart from ``reference_visuals`` for two reasons: they are a separate
    quality decision under a separate switch, and turning them into segments
    costs a partial tile per boundary, which is a price only a run that actually
    sparsifies them should pay. Once selected the two are cut and ranked
    identically.
    """

    prefix_segments: tuple[int, ...]
    video_grid: tuple[int, int, int]
    reference_visuals: tuple[tuple[int, tuple[int, int, int]], ...] = ()
    text_visuals: tuple[tuple[int, int, tuple[int, int, int]], ...] = ()

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


# The rows a call hands out early: its bf16 output, or the FP8 pair for it.
AttentionRows = torch.Tensor | tuple[torch.Tensor, torch.Tensor]
RowsReady = Callable[[int, AttentionRows], None]

_rows_ready: ContextVar[tuple[tuple[int, ...], RowsReady, bool, bool] | None] = (
    ContextVar("vsa_h3_rows_ready", default=None)
)


@contextmanager
def vsa_h3_rows_ready(
    bounds: tuple[int, ...],
    ready: RowsReady,
    *,
    fp8: bool = False,
    scale_ue8m0: bool = False,
) -> Iterator[None]:
    """Hand the caller leading rows of the next attention output early.

    ``bounds`` are increasing row ends over the rows the call receives, the
    last one all of them. ``ready(i, rows)`` is called, in order, once rows
    ``[0, bounds[i])`` of ``rows`` -- the tensor the call will return -- are
    final on the current stream, so work the caller queues there in the
    callback runs after them while the rest of the attention is still queued.

    It is a hint, not a contract to call back: a call that runs dense, or
    cannot split along these rows, calls back for none of them, or for fewer
    than all, and the caller finishes the rest from the tensor it is handed.
    Only the next call's output is offered; the callback owns the rows it was
    told about only for reading.

    ``fp8`` asks for the rows as the FP8 input of a projection that quantizes
    per token in groups of one head: ``rows`` is then ``(fp8 [T, H * D],
    fp32 scales [T, H])``, bit for bit what quantizing the bf16 rows gives,
    or with ``scale_ue8m0`` the UE8M0 exponents of that quantization packed
    four heads to an int32, ``[T, ceil(H / 4)]`` row-major. A call that can
    oblige -- the Sage kernel's -- hands out every bound that
    way and writes no bf16 rows, so the tensor it returns is a placeholder of
    the right shape that holds nothing; one that cannot hands out bf16 rows,
    or none.
    """
    token = _rows_ready.set((bounds, ready, fp8, scale_ue8m0))
    try:
        yield
    finally:
        _rows_ready.reset(token)


# ``blocks.<idx>.attn`` is a DiT layer; ``token_refiner.blocks.<idx>.attn`` and
# anything else is not and stays dense.
_DIT_LAYER_PREFIX = re.compile(r"^blocks\.(\d+)\.")


def _dit_layer_index(prefix: str) -> int | None:
    match = _DIT_LAYER_PREFIX.match(prefix)
    return int(match.group(1)) if match else None


def _balanced_chunk(heads: int, widest: int) -> int:
    """The evenest slice that still runs in as few passes as ``widest`` does.

    The widest slice the budget allows is not the cheapest one: at 28 heads a
    ceiling of 27 runs 27 then 1, and the peak is sized by that first pass
    while the second does almost nothing. Fourteen and fourteen is the same
    two passes for half the transient.

    What that buys is peak memory, and the amount is modest. Measured on one
    RTX 5090 at 28 heads and the 512 MiB default budget, on the grids where
    the two widths differ at all -- half of them do not, and there the two are
    the same number: 1249 MiB against 1166 at ``(17, 60, 106)``, 1091 against
    1004 at ``(21, 48, 85)``, 1335 against 1292 at ``(30, 48, 85)``, 1624
    against 1562 at ``(40, 48, 85)``, so 3% to 8% off the call's peak. With
    INT8 Q.K off the key buffer doubles, the budget slices more finely and the
    unevenness costs more: 892 against 699 at ``(10, 54, 96)``, 879 against
    727 at ``(13, 48, 85)``, 17% to 22%. Time is a wash either way, the launch
    count trading against how much work each launch has to fill the GPU with.

    Slicing is not bit-exact and never was: the pooled scores are a batched
    GEMM whose batch is the slice width, cuBLAS picks by batch size, and the
    last bits move enough to flip a near-tie in top-k -- 3 to 45 of 23520
    query tiles at 28 heads. The output moves by one bf16 ULP, 2e-5 to 4e-5 of
    its norm, against the 1.3e-2 the INT8 path already spends. Changing the
    width changes which of those a deployment gets; ``{"head_chunk": n}``
    pins one.

    Never wider than ``widest``, so the budget still holds, and never more
    passes than ``widest`` would run, so nothing is paid for the evenness.
    """
    passes = math.ceil(heads / widest)
    return math.ceil(heads / passes)


def compute_topk(sparsity: float, num_tiles: int) -> int:
    """Visual key tiles kept per query tile, clamped to [1, num_tiles]."""
    return max(1, min(math.ceil((1 - sparsity) * num_tiles), num_tiles))


def _tile_shape_for(grid: tuple[int, int, int]) -> tuple[int, int, int]:
    """The 3D tile a visual block of shape ``grid`` is cut into."""
    return VSA_TILE_SIZE if grid[0] >= VSA_TILE_SIZE[0] else _FRAME_TILE_SIZE


def _block_sizes(
    grid: tuple[int, int, int],
    tile_shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Live tokens per tile for one visual block, in ``(t, h, w)`` tile order.

    ``construct_variable_block_sizes`` answers the same question but reads the
    tile shape from the module-level ``VSA_TILE_SIZE`` rather than its
    argument, so it cannot describe a block cut per frame. The order matches
    ``get_tile_partition_indices``: t-major, then h, then w.
    """
    sizes: torch.Tensor | None = None
    for length, tile in zip(grid, tile_shape):
        count = math.ceil(length / tile)
        axis = torch.full((count,), tile, dtype=torch.int32, device=device)
        remainder = length - (count - 1) * tile
        axis[-1] = remainder if remainder > 0 else tile
        sizes = axis if sizes is None else (sizes[:, None] * axis[None, :]).reshape(-1)
    assert sizes is not None  # a grid is always three axes
    return sizes


@dataclass(frozen=True)
class _TileGeometry:
    """The tiling of one packed sequence, reusable across layers and steps."""

    variable_block_sizes: torch.Tensor  # [n_tiles] int32, live tokens per tile
    scatter_index: torch.Tensor  # [live_rows] int64, packed row -> padded slot
    pad_index: torch.Tensor  # [padded_rows - live_rows] int64, the slots left over
    tile_rows: torch.Tensor  # [padded_rows] int32, the packed row each slot holds

    num_prefix_tiles: int
    num_video_tiles: int
    # Row bounds -> the tile each bound's rows are complete after; see
    # ``tiles_through_rows``. Host ints, filled on first use.
    _tiles_through: dict[tuple[int, ...], tuple[int, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )

    @property
    def num_tiles(self) -> int:
        return self.num_prefix_tiles + self.num_video_tiles

    def tiles_through_rows(self, row_ends: tuple[int, ...]) -> tuple[int, ...]:
        """For each row end, the first tile past every tile holding a row before it.

        Once every tile before that one has run, rows ``[0, end)`` are final.
        Never less than the prefix tiles, which run first, and the last end
        always reaches every tile. A tile holds rows from several places --
        a space-time cube spans frames, and a reference's tiles sit among the
        video's -- so this is a running maximum over the rows' slots, taken
        once per geometry and bounds and kept on the host: a device read per
        layer would stall the launch queue the early rows are meant to fill.
        """
        cached = self._tiles_through.get(row_ends)
        if cached is not None:
            return cached
        live = self.scatter_index.numel()
        running = torch.cummax(self.scatter_index, dim=0).values
        last_rows = [min(end, live) - 1 for end in row_ends]
        slots = running[
            torch.tensor([max(row, 0) for row in last_rows], device=running.device)
        ].tolist()
        ends = [
            max(self.num_prefix_tiles, slot // BLOCK_SIZE + 1 if row >= 0 else 0)
            for row, slot in zip(last_rows, slots)
        ]
        ends[-1] = self.num_tiles
        self._tiles_through[row_ends] = result = tuple(ends)
        return result

    @property
    def padded_rows(self) -> int:
        return self.num_tiles * BLOCK_SIZE


@functools.lru_cache(maxsize=8)
def _split_text_visuals(
    prefix_segments: tuple[int, ...],
    reference_visuals: tuple[tuple[int, tuple[int, int, int]], ...],
    text_visuals: tuple[tuple[int, int, tuple[int, int, int]], ...],
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, tuple[int, int, int]], ...],
    tuple[tuple[int, tuple[int, int, int]], ...],
]:
    """Cut the text segment around the pictures Qwen3-VL packed into it.

    Returns the new prefix, the text pictures it exposed, and
    ``reference_visuals`` re-indexed onto it -- three values because the two
    picture kinds answer to different switches and the caller decides which to
    hand the tiler.

    The text block is the first prefix segment by construction, and the spans
    are ordered, disjoint offsets into it. Each one becomes its own segment so
    it can be tiled on its patch grid; the gaps around them -- the
    ``<Picture i>`` labels, the two vision sentinels, the prompt -- stay
    protected, so a tile never pools a label with the picture it names.

    Only called when the switch is on. Splitting is not free: every boundary
    rounds a protected segment up to a whole tile, which on a six-reference
    request is nine tiles and 3.5% of the op, so a run that leaves the pictures
    protected must keep the segment whole and get exactly its old geometry.
    """
    if not text_visuals:
        return prefix_segments, (), reference_visuals
    if not prefix_segments:
        raise ValueError("VSA-H3 text_visuals need a text segment to sit in")
    text_len = prefix_segments[0]
    segments: list[int] = []
    pictures: list[tuple[int, tuple[int, int, int]]] = []

    def add(rows: int, grid: tuple[int, int, int] | None = None) -> None:
        if not rows:
            return
        if grid is not None:
            pictures.append((len(segments), grid))
        segments.append(rows)

    cursor = 0
    for start, rows, grid in text_visuals:
        if start < cursor or rows <= 0 or start + rows > text_len:
            raise ValueError(
                f"VSA-H3 text visual span ({start}, {rows}) is out of order or "
                f"escapes the {text_len}-row text segment"
            )
        add(start - cursor)
        add(rows, grid)
        cursor = start + rows
    add(text_len - cursor)

    shift = len(segments) - 1
    return (
        tuple(segments) + prefix_segments[1:],
        tuple(pictures),
        tuple((index + shift, grid) for index, grid in reference_visuals),
    )


@functools.lru_cache(maxsize=8)
def _tile_geometry(
    prefix_segments: tuple[int, ...],
    video_grid: tuple[int, int, int],
    device: torch.device,
    reference_visuals: tuple[tuple[int, tuple[int, int, int]], ...] = (),
) -> _TileGeometry:
    """Tile the packed sequence: protected chunks first, then picture tiles.

    Two kinds of tile, in this order:

    - *Prefix* tiles hold the rows that stay protected -- the prompt and every
      audio block -- cut into segment-pure 64-row chunks. A tile never
      straddles a modality boundary, because a tile is both the unit of
      selection and the unit of pooling, and pooling text with audio produces a
      score that describes neither.
    - *Video* tiles hold every picture: each reference image and reference
      video named by ``reference_visuals``, then the generated video, each cut
      on its own patch grid by ``_tile_shape_for``.

    Tile order is this function's to choose -- the kernels reach rows through
    ``tile_rows``, never by packed position -- so a reference block's rows sit
    in the middle of the packed sequence while its tiles sit among the video
    tiles at the end. That is what lets one selection budget rank reference
    tiles against generated ones without the kernel knowing the difference,
    and it is why "prefix" stays the right word for the protected tiles: they
    are a prefix of the *tile* space even when their rows are not.

    Cached on the geometry because it is request-static: 50 layers times N
    denoise steps reuse one set of index tensors.
    """
    grids = dict(reference_visuals)
    unknown = sorted(set(grids) - set(range(len(prefix_segments))))
    if unknown:
        raise ValueError(
            f"VSA-H3 reference_visuals names segments {unknown}, which are not "
            f"in a {len(prefix_segments)}-segment prefix {prefix_segments}"
        )

    # Split the prefix into the rows that stay protected and the pictures that
    # join the selection, carrying each one's start so its tiles can address
    # rows that are no longer a contiguous run.
    protected: list[tuple[int, int]] = []
    pictures: list[tuple[int, tuple[int, int, int]]] = []
    cursor = 0
    for index, segment in enumerate(prefix_segments):
        grid = grids.get(index)
        if grid is None:
            protected.append((cursor, segment))
        else:
            if math.prod(grid) != segment:
                raise ValueError(
                    f"VSA-H3 reference_visuals[{index}] grid {grid} covers "
                    f"{math.prod(grid)} rows but prefix segment {index} has "
                    f"{segment}"
                )
            pictures.append((cursor, grid))
        cursor += segment
    prefix_rows = cursor
    pictures.append((prefix_rows, video_grid))

    prefix_sizes: list[int] = []
    prefix_parts: list[torch.Tensor] = []
    for start, rows in protected:
        full, remainder = divmod(rows, BLOCK_SIZE)
        prefix_sizes.extend([BLOCK_SIZE] * full)
        if remainder:
            prefix_sizes.append(remainder)
        prefix_parts.append(
            torch.arange(start, start + rows, device=device, dtype=torch.long)
        )

    video_parts: list[torch.Tensor] = []
    video_size_parts: list[torch.Tensor] = []
    for start, grid in pictures:
        shape = _tile_shape_for(grid)
        video_parts.append(
            get_tile_partition_indices(grid, shape, device).to(torch.long) + start
        )
        video_size_parts.append(_block_sizes(grid, shape, device))

    # Tiled position -> packed row. Protected rows keep their packed order;
    # each picture's rows are permuted into its own tiles.
    tile_partition = torch.cat(prefix_parts + video_parts)
    video_sizes = torch.cat(video_size_parts)
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
    sparsify_references: bool
    sparsify_text_visuals: bool
    skip_first_steps: int
    skip_last_steps: int
    skip_first_layers: int
    dense_layers: tuple[int, ...]
    min_seq_len: int
    head_chunk: int
    head_chunk_budget_mib: int
    quantize: bool
    quantize_pv: bool
    # Last, and defaulted, so the many call sites that build a schedule
    # positionally keep working: these pick the executor, not the function.
    kernel: str = DEFAULT_KERNEL
    sage_head_chunk_budget_mib: int = DEFAULT_SAGE_HEAD_CHUNK_BUDGET_MIB
    sage_min_seq_len: int = DEFAULT_SAGE_MIN_SEQ_LEN

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
            sparsify_references=bool(
                config.get("sparsify_references", DEFAULT_SPARSIFY_REFERENCES)
            ),
            sparsify_text_visuals=bool(
                config.get("sparsify_text_visuals", DEFAULT_SPARSIFY_TEXT_VISUALS)
            ),
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
            sage_head_chunk_budget_mib=int(
                config.get(
                    "sage_head_chunk_budget_mib", DEFAULT_SAGE_HEAD_CHUNK_BUDGET_MIB
                )
            ),
            sage_min_seq_len=int(
                config.get("sage_min_seq_len", DEFAULT_SAGE_MIN_SEQ_LEN)
            ),
            quantize=quantize,
            # ``quantize`` is the master switch: turning it off has to give the
            # reference kernel, not one quantized GEMM out of two.
            quantize_pv=quantize
            and bool(config.get("quantize_pv", DEFAULT_QUANTIZE_PV)),
            kernel=str(config.get("kernel", DEFAULT_KERNEL)),
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
        if schedule.kernel not in _KERNELS:
            raise ValueError(
                f"vsa_h3 kernel must be one of {_KERNELS}, got {schedule.kernel!r}"
            )
        # ``auto`` silently falls back; naming the kernel is a request to be
        # told when it cannot be honoured, because the reason is always a
        # config or a host the caller can change.
        if schedule.kernel == "flashinfer" and not (
            schedule.quantize and schedule.quantize_pv
        ):
            raise ValueError(
                "vsa_h3 kernel='flashinfer' computes INT8 Q.K with FP8 P.V and has "
                "no other mode; it needs quantize and quantize_pv both on"
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
        if schedule.sage_head_chunk_budget_mib < 1:
            raise ValueError(
                "vsa_h3 sage_head_chunk_budget_mib must be at least 1, got "
                f"{schedule.sage_head_chunk_budget_mib}"
            )
        if schedule.sage_min_seq_len < 0:
            raise ValueError(
                "vsa_h3 sage_min_seq_len must be non-negative, got "
                f"{schedule.sage_min_seq_len}"
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
                f"prefix_mode={self.schedule.prefix_mode}, reference pictures "
                f"{'sparsified' if self.schedule.sparsify_references else 'protected'}"
                f", text-side reference pictures "
                f"{'sparsified' if self.schedule.sparsify_text_visuals else 'protected'}"
                f", dense for the first "
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

        Unreachable on the defaults, which keep no step dense. A run that sets
        a warmup sized for the 50-step schedule and then serves a VSA-distilled
        or turbo checkpoint -- 4 to 9 steps, every index below the cutoff --
        silently degrades into dense attention, which is a config error worth a
        line in the log rather than an unexplained absence of speedup.
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

    def _sage_enabled(self, padded_rows: int) -> bool:
        """Whether this call runs on FlashInfer's SM120 Sage kernel.

        Resolved per call rather than at construction: the device is only known
        on the worker, the sequence length is the request's, and
        ``sm120_sage_available`` caches both the capability probe and the import
        behind it, so asking is free after the first time.
        """
        kernel = self.schedule.kernel
        if kernel == "triton":
            return False
        ready = (
            self.schedule.quantize
            and self.schedule.quantize_pv
            and padded_rows >= self.schedule.sage_min_seq_len
            and sm120_sage_available(self.head_size)
        )
        if kernel == "flashinfer" and not ready:
            raise RuntimeError(
                "vsa_h3 kernel='flashinfer' needs an SM120 device with "
                "FlashInfer's CuTe-DSL SM120 Sage backend, head_dim 128 and at "
                f"least {self.schedule.sage_min_seq_len} tiled rows; this call "
                f"has head_dim {self.head_size} and {padded_rows} rows. Use "
                "kernel='auto' to fall back to the Triton path where it does "
                "not apply."
            )
        return ready

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
        chosen = min(topk + geometry.num_prefix_tiles, tiles) if compete else topk
        value_bytes = 1 if self.schedule.quantize_pv else 0
        # The Sage kernel materialises Q and the output as well as K and V, so
        # its operands are asked for their own size rather than described here.
        sage = self._sage_enabled(geometry.padded_rows)
        operands = (
            sm120_sage_head_slice_bytes(geometry.padded_rows, self.head_size)
            if sage
            else geometry.padded_rows * self.head_size * key_bytes  # tiled K
            + geometry.padded_rows * self.head_size * value_bytes  # the fp8 V
        )
        per_head = (
            operands
            + 2 * tiles * self.head_size * 4  # the two pooled tile means
            + video_tiles * score_columns * 4  # the score matrix
            + video_tiles * chosen * 16  # top-k's values, indices and copies
            + video_tiles * (chosen + geometry.num_prefix_tiles) * 4  # the list
        )
        budget = (
            self.schedule.sage_head_chunk_budget_mib
            if sage
            else self.schedule.head_chunk_budget_mib
        ) * (1024 * 1024)
        return _balanced_chunk(heads, max(1, min(heads, budget // max(per_head, 1))))

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
        *,
        out: torch.Tensor | None,
        rows_ready: tuple[tuple[int, ...], Callable[[int], None]] | None = None,
        fp8_out: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Live ``[S, H, D]`` rows -> ``out``, attention over tiles.

        Two launches of the same kernel rather than one: prefix query tiles take
        every key tile and video query tiles take their selection, so splitting
        them keeps the index list as narrow as the selection instead of as wide
        as the sequence. Both write into one output, each covering the rows of
        the tiles it was given.

        Head slices are views, not copies -- every kernel here takes explicit
        strides -- so slicing costs nothing but the launches it adds.

        With ``rows_ready`` the last head slice's video launch is cut where
        each of its row bounds completes, and the callback told after each
        piece: every query tile is computed on its own, so a launch over some
        of them writes the same rows as one over all of them. ``fp8_out``,
        Sage only, takes the rows as FP8 groups in place of ``out``.
        """
        heads = query.shape[-2]
        sizes = geometry.variable_block_sizes
        tile_rows = geometry.tile_rows
        tiles, prefix_tiles = geometry.num_tiles, geometry.num_prefix_tiles
        topk = compute_topk(self.schedule.sparsity, geometry.num_video_tiles)
        quantize = self.schedule.quantize
        quantize_pv = self.schedule.quantize_pv
        sage = self._sage_enabled(geometry.padded_rows)
        chunk = self._head_chunk_for(geometry, heads, query.element_size())
        last_start = max(start for start, _ in self._head_slices(heads, chunk))

        logger.info_once(
            f"VSA-H3 attention active: {tiles} tiles ({prefix_tiles} protected "
            f"+ {geometry.num_video_tiles} picture), keeping {topk}/"
            f"{geometry.num_video_tiles} picture tiles per picture query tile, "
            f"{'INT8' if quantize else 'bf16'} Q.K and "
            f"{'FP8' if quantize_pv else 'bf16'} P.V on the "
            f"{'FlashInfer SM120 Sage' if sage else 'Triton'} kernel, "
            f"heads={heads} in slices of {chunk}"
        )

        for start, stop in self._head_slices(heads, chunk):
            q_slice = query[:, start:stop]
            k_slice = key[:, start:stop]
            v_slice = value[:, start:stop]
            out_slice = None if out is None else out[:, start:stop]

            # The Sage kernel lays out and quantizes its own operands, in its
            # own layouts -- K per 64-token tile, Q per 32-row group, V
            # transposed with a per-channel scale -- so none of the buffers
            # below are built for it.
            key_tiled = key_scale = None
            value_scale = value_mean = None
            value_operand = v_slice
            if sage:
                pass
            elif quantize:
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

            if quantize_pv and not sage:
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
            segments: tuple[int, ...] = (tiles,)
            on_segment = None
            if rows_ready is not None and start == last_start:
                segments = geometry.tiles_through_rows(rows_ready[0])
                on_segment = rows_ready[1]

            if sage:
                # Same selection, same two launches, different executor: the
                # prefix pass lives inside this call because the operands it
                # shares with the video pass are laid out there.
                sm120_sage_attention(
                    q_slice,
                    k_slice,
                    v_slice,
                    out_slice,
                    tile_rows=tile_rows,
                    variable_block_sizes=sizes,
                    num_prefix_tiles=prefix_tiles,
                    q2k_index=q2k_index,
                    q2k_num=q2k_num,
                    softmax_scale=self.softmax_scale,
                    video_segments=segments,
                    on_segment=on_segment,
                    fp8_out=(
                        None
                        if fp8_out is None
                        else (fp8_out[0][:, start:stop], fp8_out[1][:, start:stop])
                    ),
                )
                del q2k_index, q2k_num
                continue

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
                    self.softmax_scale,
                )
                del dense_index, dense_num

            first = prefix_tiles
            for index, end in enumerate(segments):
                if end > first:
                    picked = slice(first - prefix_tiles, end - prefix_tiles)
                    block_sparse_attn_forward(
                        q_slice,
                        key_tiled,
                        value_operand,
                        out_slice,
                        tile_rows,
                        # The kernel indexes the selection as contiguous.
                        q2k_index[:, picked].contiguous(),
                        q2k_num[:, picked].contiguous(),
                        sizes,
                        first,
                        key_scale,
                        value_scale,
                        value_mean,
                        self.softmax_scale,
                    )
                    first = end
                if on_segment is not None:
                    on_segment(index)
            del q2k_index, q2k_num, key_tiled, key_scale
            del value_scale, value_mean, value_operand

    def _tiles_for(
        self, geometry: VsaH3SequenceGeometry, device: torch.device
    ) -> _TileGeometry:
        segments = geometry.prefix_segments
        references = geometry.reference_visuals
        text_pictures: tuple[tuple[int, tuple[int, int, int]], ...] = ()
        if self.schedule.sparsify_text_visuals:
            segments, text_pictures, references = _split_text_visuals(
                segments, references, geometry.text_visuals
            )
        pictures = text_pictures
        if self.schedule.sparsify_references:
            pictures += references
        return _tile_geometry(segments, geometry.video_grid, device, pictures)

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
        out = torch.empty_like(query[0])
        self._sparse_attention(
            query[0],
            key[0],
            value[0],
            self._tiles_for(geometry, query.device),
            out=out,
        )
        return out.unsqueeze(0)

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

        tiles = self._tiles_for(geometry, query.device)
        consumer = _rows_ready.get()
        if consumer is not None and not _splits_rows(consumer[0], total):
            consumer = None
        live = (query[:used], key[:used], value[:used], tiles)
        if (
            consumer is not None
            and consumer[2]
            and self._sage_enabled(tiles.padded_rows)
        ):
            bounds, ready, _, scale_ue8m0 = consumer
            q8, scales = _fp8_rows(
                total,
                used,
                query.shape[1],
                query.shape[2],
                device=query.device,
                scale_ue8m0=scale_ue8m0,
            )
            # UE8M0 exponents are written a byte per head into the packed
            # int32s, so a head slice owns its bytes whatever its bounds.
            written = scales.view(torch.uint8) if scale_ue8m0 else scales
            self._sparse_attention(
                *live,
                out=None,
                rows_ready=(bounds, lambda index: ready(index, (q8, scales))),
                fp8_out=(q8[:used].view(used, *query.shape[1:]), written[:used]),
            )
            # Every bound went out as FP8; nothing reads a bf16 row.
            return query.new_empty(()).expand_as(query)

        # The output is allocated whole and the live rows attended into it,
        # so rows handed out early are rows of the tensor returned.
        out = torch.empty_like(query)
        out[used:].zero_()
        rows_ready = None
        if consumer is not None:
            bounds, ready, _, _ = consumer
            rows_ready = (bounds, lambda index: ready(index, out))
        self._sparse_attention(*live, out=out[:used], rows_ready=rows_ready)
        return out


def _fp8_rows(
    total: int,
    used: int,
    heads: int,
    head_dim: int,
    *,
    device: torch.device,
    scale_ue8m0: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 output rows and per-head scales, the padding tail already filled.

    The tail's bf16 rows would be zero, so it holds what quantizing zero rows
    gives, from the quantizer the fused scatter reproduces. UE8M0 scales start
    zeroed, so the pad bytes of a head count that is not a multiple of four
    are the zeros that quantizer packs there.
    """
    q8 = torch.empty(
        (total, heads * head_dim), device=device, dtype=torch.float8_e4m3fn
    )
    scales = (
        torch.zeros((total, -(-heads // 4)), device=device, dtype=torch.int32)
        if scale_ue8m0
        else torch.empty((total, heads), device=device, dtype=torch.float32)
    )
    if used < total:
        q8[used:], scales[used:] = sglang_per_token_group_quant_fp8(
            torch.zeros(
                (total - used, heads * head_dim), device=device, dtype=torch.bfloat16
            ),
            head_dim,
            column_major_scales=scale_ue8m0,
            scale_tma_aligned=scale_ue8m0,
            scale_ue8m0=scale_ue8m0,
        )
    return q8, scales


def _splits_rows(bounds: tuple[int, ...], total: int) -> bool:
    return (
        bool(bounds)
        and bounds[-1] == total
        and all(0 < a < b for a, b in zip(bounds, bounds[1:]))
        and bounds[0] > 0
    )


__all__ = [
    "VideoSparseAttentionH3Backend",
    "VideoSparseAttentionH3Impl",
    "VideoSparseAttentionH3Metadata",
    "VideoSparseAttentionH3MetadataBuilder",
    "VsaH3Schedule",
    "VsaH3SequenceGeometry",
    "compute_topk",
    "vsa_h3_rows_ready",
    "vsa_h3_sequence_geometry",
]
