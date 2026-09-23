# SPDX-License-Identifier: Apache-2.0
"""Row-sharded residual stream for MiniMax-H3's tensor-parallel block stack.

Under TP every block ends its attention and its MLP in a row-parallel GEMM
whose partial sums are all-reduced, and everything between that reduction and
the next GEMM -- the gated residual, RMSNorm, the AdaLN scale/shift and the
FP8 quantization of the next GEMM's input -- is row-local. So each rank can
reduce-scatter instead, keep only its own rows of the residual, run those
row-local steps on half the rows, and all-gather the *quantized* input the
next GEMM was going to build anyway: FP8 plus its per-group scales, half the
bytes of the bf16 rows an all-reduce hands back.

Row-local is also what lets the collectives hide. Everything from one
attention's output projection to the next attention's QKV projection works a
row at a time, so the rows are cut into chunks and each chunk's reduce-scatter
or all-gather runs on a side stream while the next chunk's GEMM runs on the
compute stream. A chunk of the global rows reduce-scatters into halves, so a
rank's rows are its half of every chunk, back to back; with one chunk that is
simply its half of the sequence.

It is bit-exact against the all-reduce path, not an approximation of it:

- At TP=2 an all-reduce sums each element once, and so does a reduce-scatter,
  so the rows a rank keeps are the rows the all-reduce would have produced.
- Every step until the next GEMM computes a row from that row alone, so
  running it over some of the rows changes nothing.
- The gathered FP8 rows and scales are what quantizing the full rows gives,
  and FlashInfer's groupwise GEMM computes every output row the same whether
  it is handed all of M or a slice of it, so the GEMMs see the same operands
  and give the same bits, chunked or not, concurrent with NCCL or not.

The GEMM's input quantization moves out of the linear layer to make this
possible, so the path is taken only where that layer's own quantization is
exactly the one reproduced here: block-FP8 weights on FlashInfer's CUTLASS
groupwise GEMM, which is what an SM120 card dispatches to. Anything else --
another runner, a LoRA-wrapped QKV or fc1, bf16 weights, where gathering bf16
would cost more than the all-reduce saves -- keeps the all-reduce. A LoRA on
the output projection or fc2 does not stand in the way: it adds its delta to
the rank's partial sums before they are reduced, whichever way that happens,
so the collectives take their group from the attention and MLP modules rather
than from a layer a wrapper may have replaced.
"""

from __future__ import annotations

import functools
from typing import Callable

import torch
from torch import nn

from sglang.kernels.ops.diffusion.triton.indexed_modulation import (
    indexed_scale_shift_quant_fp8,
)
from sglang.multimodal_gen.runtime.distributed.group_coordinator import (
    GroupCoordinator,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization import fp8_utils

# Chunks per pass through the row-local stretch. Four hides all but the first
# and last chunk's collective behind GEMMs that lose 1-5% to the smaller M;
# fewer is taken when the rows do not divide.
_MAX_ROW_CHUNKS = 4
# FlashInfer's groupwise GEMM wants M a multiple of four rows.
_GEMM_ROW_ALIGNMENT = 4


def takes_prequantized_input(linear: nn.Module) -> bool:
    """Whether this layer can be handed its input already FP8-quantized.

    True where the layer's own quantization is sglang's per-token-group
    quantizer feeding FlashInfer's CUTLASS groupwise GEMM, which the fused
    kernels reproduce bit for bit, so quantizing ahead of the layer changes
    nothing but where it happens.
    """
    # A LoRA wrapper is neither, and it has to see the bf16 input to compute
    # its own delta, so it keeps quantizing inside the layer.
    if not isinstance(linear, (ColumnParallelLinear, RowParallelLinear)):
        return False
    method = linear.quant_method
    return (
        isinstance(method, Fp8LinearMethod)
        and method.block_quant
        and not method.use_marlin
        and method.w8a8_block_fp8_linear
        is fp8_utils.flashinfer_gemm_w8a8_block_fp8_linear_with_fallback
        # Its TRTLLM backend quantizes into a different scale layout.
        and fp8_utils._get_flashinfer_groupwise_backend() == "cutlass"
    )


def row_chunks(
    *,
    column_linears: tuple[nn.Module, ...],
    tp_size: int,
    rows: int,
) -> int:
    """Chunks to run a block stack over ``rows`` rows row-sharded in, or 0.

    ``column_linears`` are the layers that take a gathered, pre-quantized
    input. Only TP=2 is claimed: beyond two ranks the ring's summation order is
    NCCL's to choose and need not match between the two collectives.
    """
    if tp_size != 2 or not all(takes_prequantized_input(l) for l in column_linears):
        return 0
    for chunks in range(_MAX_ROW_CHUNKS, 0, -1):
        if rows % (chunks * tp_size * _GEMM_ROW_ALIGNMENT) == 0:
            return chunks
    return 0


def shard_rows(
    full: torch.Tensor, *, chunks: int, rank: int, world: int
) -> torch.Tensor:
    """Full rows -> this rank's half of every chunk, back to back."""
    half = full.shape[0] // (chunks * world)
    tail = full.shape[1:]
    return full.view(chunks, world, half, *tail)[:, rank].reshape(chunks * half, *tail)


def unshard_rows(
    shard: torch.Tensor, *, chunks: int, group: GroupCoordinator
) -> torch.Tensor:
    """Every rank's rows -> the full rows, in their original order."""
    world = group.world_size
    half = shard.shape[0] // chunks
    gathered = group.all_gather(shard, dim=0)
    return (
        gathered.view(world, chunks, half, *shard.shape[1:])
        .transpose(0, 1)
        .reshape(world * shard.shape[0], *shard.shape[1:])
    )


class _Pending:
    """A collective's output, usable on the compute stream once it lands."""

    def __init__(self, result, done: torch.cuda.Event, inputs: tuple) -> None:
        self._result = result
        self._done = done
        # The inputs stay referenced until the compute stream is ordered after
        # the collective, so the allocator cannot hand their memory to a
        # compute kernel the side stream is still reading from.
        self._inputs = inputs

    def result(self):
        torch.cuda.current_stream().wait_event(self._done)
        self._inputs = None
        return self._result


class CommLane:
    """One side stream the row-local stretch's collectives queue on, in order.

    A collective is submitted as soon as its input is issued on the compute
    stream and waits for it there; the compute stream waits for the output only
    where it reads it. NCCL keeps the collectives in submission order, which is
    the same on every rank.

    Every buffer is allocated on the compute stream, outputs included, and none
    is recorded on the side stream: a tensor the allocator sees on two streams
    is one it cannot reuse until both have moved on, and at this model's
    footprint that stalls the whole card in the allocator's reclaim path.
    """

    def __init__(self, device: torch.device) -> None:
        self._stream = torch.cuda.Stream(device=device)

    def _submit(self, work: Callable[[], None], result, inputs: tuple) -> _Pending:
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            work()
            done = torch.cuda.Event()
            done.record(self._stream)
        return _Pending(result, done, inputs)

    def gather(
        self, pair: tuple[torch.Tensor, torch.Tensor], *, group: GroupCoordinator
    ) -> _Pending:
        """This rank's ``(fp8 rows, scales)`` -> the whole group's."""
        outputs = tuple(
            tensor.new_empty((tensor.shape[0] * group.world_size, *tensor.shape[1:]))
            for tensor in pair
        )

        def work() -> None:
            for output, tensor in zip(outputs, pair):
                group.all_gather_into(output, tensor)

        return self._submit(work, outputs, pair)

    def reduce_scatter(
        self, partial: torch.Tensor, *, group: GroupCoordinator
    ) -> _Pending:
        """A chunk's partial sums -> this rank's half of the sum."""
        output = partial.new_empty(
            (partial.shape[0] // group.world_size, *partial.shape[1:])
        )
        return self._submit(
            lambda: group.reduce_scatter(partial, output), output, (partial,)
        )


@functools.cache
def comm_lane(device: torch.device) -> CommLane:
    return CommLane(device)


def quantize_modulated(
    normed: torch.Tensor,
    *,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    linear: ColumnParallelLinear,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AdaLN scale/shift then ``linear``'s FP8 input quantization, one kernel.

    Bit-exact against the modulation kernel followed by the linear's own
    quantizer, and returns the ``(fp8 rows, row-major per-group scales)`` pair
    the block-FP8 linear accepts as its input.
    """
    return indexed_scale_shift_quant_fp8(
        normed,
        shift,
        scale,
        indices,
        group_size=linear.quant_method.quant_config.weight_block_size[1],
    )


def linear_into(
    linear: ColumnParallelLinear,
    pair: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
) -> None:
    """``linear`` over a pre-quantized input, written into ``out``'s rows."""
    fp8_utils.flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(
        pair[0],
        linear.weight,
        linear.quant_method.quant_config.weight_block_size,
        linear.weight_scale_inv,
        input_scale=pair[1],
        out=out,
    )
