# VSA-H3 — video sparse attention for the MiniMax-H3 DiT

A port of [FastVideo][fv]'s `video_sparse_attn_h3` backend onto this tree's
packed-varlen attention contract, with SageAttention-style INT8 Q.K on top.
Attention runs over 64-token *tiles*: video rows are grouped into `(4, 4, 4)`
space-time cubes of the patch grid, everything before them into segment-pure
chunks, and each video query tile attends only to the top `(1 - sparsity)`
fraction of video key tiles plus every prefix tile.

Spelled out in full, with every key at its default:

```bash
sglang serve --model-path MiniMaxAI/MiniMax-H3 --model-variant t2va \
  --attention-backend video_sparse_attn_h3 \
  --component-attention-backends text_encoder=fa \
  --attention-backend-config '{"sparsity": 0.9, "prefix_mode": "exempt",
                               "skip_first_steps": 10, "skip_first_layers": 0,
                               "min_seq_len": 4096}'
```

**`text_encoder=fa` is not optional.** `--attention-backend` applies to every
component and the Qwen3-VL text encoder admits only `fa` / `torch_sdpa` /
`sage_attn_3`; without the override it raises and the server never starts. Put
the override on the *encoder*, not the DiT — `transformer=video_sparse_attn_h3`
appears to work and silently does nothing, because H3 resolves the DiT backend
lazily on the first forward, outside the component-loading context.

`--attention-backend-config` overrides only the keys it names, so
`'{"sparsity": 0.95}'` alone is a valid config. Inline JSON gets mangled by
`shlex.split`; pass a **file path** instead if the shell eats the quotes.

## What it runs on

| | |
| --- | --- |
| GPU | compute capability >= 8.0. The block-sparse forward is Triton and vendored in-tree, so unlike every other sparse backend here it needs no package and no arch-specific build. |
| dtype | bfloat16 (P.V accumulates in bf16; Q.K is INT8 unless `quantize` is off) |
| head_dim | 64 or 128 |
| attention | non-causal, MHA (`num_kv_heads == num_heads`), one packed H3 document per call |

Anything else — the token refiner, a causal layer, a sequence shorter than
`min_seq_len`, a request whose geometry was never published — runs through the
dense fallback for that call, so no layer has to be excluded by hand. The
fallback is `sage_attn` (resolved to FlashAttention, then Torch SDPA, where
SageAttention is not built), so an excluded call costs what a plain
`--attention-backend sage_attn` deployment already costs.

## How selection works

1. **Tile.** Every live row gets a tiled slot: prefix segments (text, each
   reference block, audio) fill 64-row chunks that never straddle a modality
   boundary; video rows fill `(4, 4, 4)` cubes of the `(T, H, W)` patch grid. A
   tile is both the unit of pooling and the unit of selection, so a tile mixing
   text with audio would produce a score describing neither. Only **K** is
   copied into that order — Q, V and the output are the caller's packed
   tensors, gathered and scattered through the slot-to-row index inside the
   kernels, because every query tile reads all of K and reads it transposed
   while it touches its own rows once.
2. **Pool and score.** Each tile is mean-pooled over its *live* rows in fp32,
   giving one vector per tile. Only video query tiles are scored, and in
   `exempt` mode only against video keys, so the score matrix is
   `[heads, n_video, n_video]` rather than tiles-by-tiles. The softmax scale is
   left out: top-k only ranks, and a positive constant does not change a
   ranking.
3. **Select.** Each video query tile keeps its top `k = ceil((1 - sparsity) *
   n_video)` video key tiles. Prefix keys are exempt from that budget and kept
   by every query; prefix *queries* are dense.
4. **Attend.** Two launches of the same kernel — prefix query tiles with every
   key tile, video query tiles with their selection — so the index list stays
   as narrow as the selection instead of as wide as the sequence. Columns past
   a tile's live token count are masked to -inf inside the kernel, which is what
   lets segments that are not multiples of 64 be tiled at all. Each launch
   writes only the rows of the tiles it was given, so rows past the live count
   (H3's 64-aligned padding tail) are never written and stay zero.

## Quantization

`quantize` (on by default) runs the first GEMM on INT8 tensor cores the way
SageAttention does: K is quantized per tile with its per-channel mean removed
first, Q per tile in registers, and P.V stays bf16. Removing K's mean is exact
rather than approximate — it shifts every logit in a row by the same `-q.km`,
which softmax cancels — and it is what keeps the int8 range on the part of K
that varies.

It is worth ~1.9x on the attention op for 1.3% of the output's norm in added
error, flat across sparsity. That is SageAttention's own error budget, and it is
the same error this deployment already carries: `sage_attn` is what the backend
falls back to on the warmup steps. Against it, the sparsity is by far the larger
approximation. Turn it off to separate a quality question from the sparsity.

## Configuration

| key | default | meaning |
| --- | ---: | --- |
| `sparsity` | 0.9 | video key tiles dropped per video query tile. `VSA_sparsity` is accepted as an alias |
| `prefix_mode` | `exempt` | `exempt` keeps every prefix key for free; `compete` makes them compete inside a budget of `k + n_prefix` tiles |
| `skip_first_steps` | 10 | leading denoise forwards kept dense |
| `skip_last_steps` | 0 | trailing denoise forwards kept dense (needs the schedule length, which the H3 denoising stage publishes) |
| `skip_first_layers` | 0 | leading DiT blocks kept dense |
| `dense_layers` | `[]` | individual DiT blocks kept dense |
| `min_seq_len` | 4096 | shorter sequences run dense |
| `quantize` | true | run Q.K on INT8 tensor cores, K's per-channel mean removed first |
| `head_chunk` | 0 | heads per pass; 0 sizes the slice from the budget below. An explicit count overrides it; a count at or above the head count runs them all in one pass. Slicing is exact, not an approximation |
| `head_chunk_budget_mib` | 512 | transient budget per attention call, which the automatic slice is sized to hit |

## Memory

A dense flash kernel streams and costs essentially nothing (+12 MiB at 116k
rows). This backend has to put K in tile order and score tile against tile, so
it cannot reach zero — but **only K** is materialised, and the head slice is
sized from `head_chunk_budget_mib` so what remains stays flat across
resolutions instead of growing with them. Per head that is the tiled K (one
byte per element quantized), the two pooled tile means, an `n_video x n_video`
fp32 score matrix and the index list: 36 MiB at a 116k-row sequence, which the
estimate predicts to within 0.5%.

Measured transient peak over the packed inputs and the output, at the default
512 MiB budget: **+128 MiB** at 36k rows and 14 heads, **+509 MiB** at 116k rows
and 28 heads. For comparison, materialising Q, K, V *and* the output in tile
order — what upstream does, and what this backend did before — costs 4.2 GiB at
that second shape, which is enough to OOM a 32 GiB card the dense path fits on
with room to spare.

Lower `head_chunk_budget_mib` if the card is tighter than that. Below one head
per pass there is nothing left to slice, and a 1-head pass loses enough kernel
occupancy to cost ~40%.

## Measured

One RTX 5090, D=128, bf16, against bf16 SDPA at the same shape. The whole
backend call, selection and the dense prefix launch included:

| | SDPA | 0.9 bf16 | **0.9 INT8** | 0.95 bf16 | **0.95 INT8** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 36k rows, 14 heads | 44.0 ms | 15.8 ms (2.8x) | **8.8 ms (5.0x)** | 12.3 ms (3.6x) | **7.1 ms (6.2x)** |
| 116k rows, 28 heads | 888.6 ms | 226.5 ms (3.9x) | **117.2 ms (7.6x)** | 152.6 ms (5.8x) | **80.3 ms (11.1x)** |

Two things hold the sparse column back from the ratio the tile counts suggest,
and both are the exempt prefix: a video query tile keeps its selected video
tiles *plus* every prefix tile, and the prefix query tiles run dense over the
whole sequence. `{"prefix_mode": "compete"}` spends the same budget without that
surcharge, and is the ablation to run if the prefix turns out not to need
protecting on your workload.

Attention is only part of a denoise step, so the end-to-end gain is smaller.

## What this does not port

- **The gate-compress branch.** Upstream VSA adds a dense-over-pooled-tiles
  term scaled by a trained `to_gate_compress` matrix per layer. No MiniMax-H3
  checkpoint this tree loads carries one, and upstream zero-initializes it, so
  without it VSA is exactly the top-k block-sparse attention computed here.
- **The 256-token tile and its FA4 CuTe / sm_100a routes.** Those target
  sm_10x; 64 is the geometry FastVideo's own checkpoint is trained and measured
  at, and the one the vendored Triton kernel is built for.
- **Training.** Forward only: no backward kernel, no autograd wrapper.
- **FP8 P.V.** SageAttention2 quantizes the second GEMM as well; here it stays
  bf16, which is the conservative half of the trade and leaves that speedup on
  the table.

## This is not free accuracy

VSA is a *trainable* sparse attention. FastVideo runs it at `sparsity=0.9`
against a VSA-distilled checkpoint, where 0.9 is the policy the student was
trained under; that checkpoint also wants `skip_first_steps=0`. Against a stock
MiniMax-H3 checkpoint the same setting is training-free block sparsity and its
quality is unmeasured here — which is why the warmup cutoff defaults to the
same 10 steps every other sparse backend in this tree uses. Measure against a
dense render before trusting a sparsity.

## Files

| | |
| --- | --- |
| `kernels.py` | the vendored 64x64 Triton block-sparse forward, its INT8 quantizer and its tile-mean pooler |
| `../video_sparse_attn_h3.py` | the `AttentionBackend`: geometry, tiling, selection, schedule, dense fallback |
| `../denoise_schedule.py` | the published denoise-schedule length `skip_last_steps` needs |

Tests: `test/unit/test_vsa_h3_attention.py`. The trick that makes the sparse
kernel checkable against dense attention: at `sparsity=0` every video tile is
inside the budget and every prefix tile is exempt, so the block-sparse result
must reproduce dense attention over the live rows — which pins the tiling, the
pad masking, the scatter/gather round trip and the softmax scale in one
assertion. At real sparsity the backend is compared against an independent
mask-based PyTorch implementation of the same selection rule. Both run with
`quantize` off, since they pin the math; INT8 is an approximation and is
asserted as a bound on the whole output instead.

[fv]: https://github.com/hao-ai-lab/FastVideo
