# FP8 prefill loader and NVFP4 workspace choices

The optimized loader supports both native `--kv-cache-dtype fp8_e4m3` buffers
and the FP8 workspace produced by NVFP4 dequantization. For native FP8 KV, use:

```sh
--kv-cache-dtype fp8_e4m3 \
--prefill-attention-backend flashinfer \
--enable-fp8-prefill-workspace-optimization
```

Do not set `--prefill-kv-cache-dequant-dtype` for native FP8 KV: that option is
NVFP4-only. The loader reads the existing FP8 pool directly, with no dequant
workspace or larger global KV allocation. Decode backend selection is unchanged.
If no attention backend is specified, opting in selects FlashInfer for prefill;
explicit incompatible prefill backends are rejected. The flag remains off by
default for both KV formats. It does not imply every prefill call is accelerated:
ragged attention and unsupported paged plans retain upstream implementations.

## NVFP4 workspace choices

These options change the temporary prefill workspace, **not** the stored
`--kv-cache-dtype nvfp4` pool or native NVFP4 decode. They work with DFlash;
disabling speculative decoding is not required.

| Mode | Options added to the existing NVFP4 launch command |
|---|---|
| Original FP8 (default on SM120) | `--prefill-kv-cache-dequant-dtype fp8_e4m3` |
| BF16 workspace | `--prefill-kv-cache-dequant-dtype bfloat16` |
| Optimized FP8 workspace | `--prefill-kv-cache-dequant-dtype fp8_e4m3 --enable-fp8-prefill-workspace-optimization` |

The BF16 and optimized-FP8 choices are mutually exclusive. When neither is
selected, the existing FP8 path on SM120 is unchanged. The pre-existing `auto`
behavior on SM100 still selects native NVFP4 prefill; this change does not
override that platform default.

BF16 retains FP8-rounded values in a twice-as-large BF16 K/V workspace.
Optimized FP8 retains the original FP8 global workspace, converts loaded values
in registers, and writes BF16 shared-memory tiles directly. Both choices can
change attention reduction order and subsequent generated text; neither is a
full-model bit-exact guarantee.

## Integration and compatibility

- No experimental environment variables, `sitecustomize`, benchmark artifact
  paths, or edits to the installed FlashInfer package are needed. The old
  `SGLANG_EXPERIMENT_NVFP4_PREFILL_BF16` variable no longer controls this branch.
- The optimized loader is opt-in and currently requires SM120, FlashInfer
  prefill, and either native FP8 E4M3 KV storage or NVFP4 KV storage with FP8
  E4M3 prefill dequantization. Invalid configurations fail at argument resolution,
  before model loading.
- Only FA2 paged plans with BF16 Q/output, E4M3 KV, int32 indices, D256, and no
  positional encoding, sliding window, softcap, custom mask, or MIS are selected.
  Other plans, including D128 draft attention and BF16 workspace, retain the
  upstream implementation. Changing plans resets the custom-JIT slot as needed.
- Target-verification and draft-extend graph wrappers keep their original
  implementations; the switch selects normal prefill and full-prefill graph
  wrappers only, not speculative verification/decode kernels.
- Per-wrapper custom-JIT modules have distinct names and do not change
  FlashInfer's global generators or baseline module cache. Planning, execution,
  full-prefill CUDA graph wrappers, and workspace-size queries use this slot.
- `fp8_prefill_loader.patch` is a small Apache-2.0 modification of FlashInfer's
  Apache-2.0 `attention/prefill.cuh`. It targets the repository's pinned
  FlashInfer 0.6.18 header, verified by SHA-256. A complete relative include tree
  is copied to a content-addressed JIT cache overlay under a cross-process lock;
  only its prefill header is changed. Original installed files stay untouched.
  A changed upstream header fails closed when the option is selected; update
  and revalidate the patch when upgrading FlashInfer. JIT compilation needs the
  usual CUDA toolchain and a writable FlashInfer cache.

Standalone regression tests: `test/registered/kernels/ops/attention/test_fp8_prefill_workspace.py`.
They do not start or stop model servers. Prior end-to-end benchmark measurements
were made with the experimental loader and additional existing NVFP4 fusion
options; their absolute times are not a new measurement of this formal CLI
integration.
