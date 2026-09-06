# VSA-H3 — video sparse attention for the MiniMax-H3 DiT

A port of [FastVideo][fv]'s `video_sparse_attn_h3` backend onto this tree's
packed-varlen attention contract. Attention runs over 64-token *tiles*: video
rows are grouped into `(4, 4, 4)` space-time cubes of the patch grid, everything
before them into segment-pure chunks, and each video query tile attends only to
the top `(1 - sparsity)` fraction of video key tiles plus every prefix tile.

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
| dtype | bfloat16 (the kernel accumulates P in bf16) |
| head_dim | 64 or 128 |
| attention | non-causal, MHA (`num_kv_heads == num_heads`), one packed H3 document per call |

Anything else — the token refiner, a causal layer, a sequence shorter than
`min_seq_len`, a request whose geometry was never published — runs through the
dense fallback for that call, so no layer has to be excluded by hand. The
fallback is `sage_attn` (resolved to FlashAttention, then Torch SDPA, where
SageAttention is not built), so an excluded call costs what a plain
`--attention-backend sage_attn` deployment already costs.

## How selection works

1. **Tile.** The live rows are scattered into a padded `[n_tiles, 64]` buffer:
   prefix segments (text, each reference block, audio) become 64-row chunks
   that never straddle a modality boundary; video rows become `(4, 4, 4)`
   cubes of the `(T, H, W)` patch grid. A tile is both the unit of pooling and
   the unit of selection, so a tile mixing text with audio would produce a
   score describing neither.
2. **Pool and score.** Each tile is mean-pooled over its *live* tokens in fp32
   (pad slots are zero and were never written, so a sum over 64 divided by the
   live count is the masked mean exactly), giving one vector per tile. The
   pooled `Q.K` matrix is `[heads, n_tiles, n_tiles]`.
3. **Select.** Each video query tile keeps its top `k = ceil((1 - sparsity) *
   n_video)` video key tiles. Prefix keys are exempt from that budget and kept
   by every query; prefix *queries* are dense.
4. **Attend.** Two launches of the same kernel — prefix query tiles with every
   key tile, video query tiles with their selection — so the index list stays
   as narrow as the selection instead of as wide as the sequence. Columns past
   a tile's live token count are masked to -inf inside the kernel, which is
   what lets segments that are not multiples of 64 be tiled at all.
5. **Untile.** The output rows are gathered back into packed order; rows past
   the live count (H3's 64-aligned padding tail) are zeroed.

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
| `head_chunk` | 0 | heads per pass; 0 sizes the slice from the budget below. An explicit count overrides it; a count at or above the head count runs them all in one pass. Slicing is exact, not an approximation |
| `head_chunk_budget_mib` | 512 | transient budget per attention call, which the automatic slice is sized to hit |

## Memory

Unlike a dense flash kernel, which streams and costs essentially nothing, this
backend reorders the sequence into tiles and so needs padded copies of Q, K and
V plus an output buffer, and a `[heads, tiles, tiles]` fp32 score matrix. Per
head that is ~30 MiB of tile buffer times four plus 13 MiB of scores at a
116k-row sequence — **3.7 GiB across 28 rank-local heads**, which is enough to
OOM a 32 GiB card that the dense path fits on with room to spare.

That is what `head_chunk_budget_mib` exists for: the head slice is sized so the
transients land inside it. Measured at 115,708 rows, 28 heads, D=128, bf16 on
one RTX 5090, against `torch.zeros`-free bf16 SDPA at the same shape (877 ms,
+12 MiB):

| | heads/pass | time | transient peak |
| --- | ---: | ---: | ---: |
| whole-head pass | 28 | 189 ms | **+4259 MiB** |
| budget 512 MiB (default) | 3 | 198 ms | +466 MiB |
| budget 256 MiB | 1 | 269 ms | +166 MiB |

The default costs ~4% for an 9x cut in peak. Below one head per pass there is
nothing left to slice, and a 1-head pass loses enough kernel occupancy to cost
40%; if 466 MiB still does not fit, lower `sparsity` is not the lever —
`min_seq_len` or the dense fallback is.

## Measured

One RTX 5090, H=14, D=128, bf16, against bf16 SDPA. The backend column is the
whole call — tiling, pooled scores, top-k, the dense prefix launch and the
gather — which is what a denoise step actually pays; the kernel column is the
block-sparse launch alone, on the same shapes with no prefix to protect.

| rows | backend @0.9 | backend @0.95 | kernel alone @0.9 | @0.95 |
| --- | ---: | ---: | ---: | ---: |
| ~36k | 2.95x | 3.72x | 7.4x | 13.4x |
| ~71k | 3.82x | 5.06x | 7.5x | 14.0x |

Most of that gap is the exempt prefix: at 36k rows a video query tile keeps 60
selected video tiles *plus* 29 prefix tiles, and the 29 prefix query tiles run
dense over all 623. `{"prefix_mode": "compete"}` spends the same budget without
the surcharge, and is the ablation to run if the prefix turns out not to need
protecting on your workload. Peak memory over the dense baseline is +687 MiB at
36k rows and +2.5 GiB at 71k with `head_chunk` unset; `{"head_chunk": 4}`
roughly halves that for ~7% of the time.

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
| `kernels.py` | the vendored 64x64 Triton block-sparse forward |
| `../video_sparse_attn_h3.py` | the `AttentionBackend`: geometry, tiling, selection, schedule, dense fallback |
| `../denoise_schedule.py` | the published denoise-schedule length `skip_last_steps` needs |

Tests: `test/unit/test_vsa_h3_attention.py`. The trick that makes the sparse
kernel checkable against dense attention: at `sparsity=0` every video tile is
inside the budget and every prefix tile is exempt, so the block-sparse result
must reproduce dense attention over the live rows — which pins the tiling, the
pad masking, the scatter/gather round trip and the softmax scale in one
assertion. At real sparsity the backend is compared against an independent
mask-based PyTorch implementation of the same selection rule.

[fv]: https://github.com/hao-ai-lab/FastVideo
