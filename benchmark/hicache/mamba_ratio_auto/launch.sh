#!/bin/bash
# Production-equivalent launch against a LOCAL mooncake master (never the prod one).
# Usage: launch.sh <gpus> <port> [extra args...]
GPUS=${1:-0,1}; PORT=${2:-30080}; shift 2
export PATH=/mnt/sglang-fork/.venv/bin:/usr/local/cuda/bin:$PATH
export CUDA_VISIBLE_DEVICES=$GPUS
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_MASTER=${MC_MASTER:-127.0.0.1:50099}
export MOONCAKE_PROTOCOL=tcp
export MOONCAKE_TE_META_DATA_SERVER=P2PHANDSHAKE
export MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-0}

exec sglang serve \
  --model-path /mnt2/agnes-3.0-flash \
  --served-model-name agnes-3.0-flash \
  --trust-remote-code \
  --host 127.0.0.1 --port $PORT \
  --tp-size 2 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend flashinfer \
  --linear-attn-decode-backend triton \
  --cuda-graph-backend-prefill disabled \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 32768 \
  --page-size 256 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-ssm-dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --max-running-requests 9 \
  --max-mamba-cache-size 40 \
  --cuda-graph-max-bs-decode 9 \
  --context-length 524288 \
  --mm-feature-transport cpu \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-metrics \
  --enable-request-time-stats-logging \
  --enable-cache-report \
  --enable-single-batch-overlap \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /mnt2/DFlash2 \
  --speculative-draft-model-quantization unquant \
  --speculative-dflash-block-size 8 \
  --speculative-num-draft-tokens 8 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-io-backend direct \
  --hicache-mem-layout page_first_direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend mooncake \
  --hicache-storage-prefetch-policy timeout \
  --hicache-storage-backend-extra-config '{"prefetch_threshold":4096}' \
  --image-processor-backend pil \
  --mm-process-config '{"image":{"size":{"longest_edge":1048576,"shortest_edge":65536}},"video":{"min_pixels":65536,"max_pixels":262144,"max_frames":40,"device":"cpu"}}' \
  --limit-mm-data-per-request '{"image":5,"video":1}' \
  "$@"
