# SPDX-License-Identifier: Apache-2.0
"""Kernels for VSA-H3, the MiniMax-H3 video sparse attention backend.

``kernels.py`` holds the vendored 64x64 Triton block-sparse forward the backend
routes, its INT8 quantizer and its tile-mean pooler; the selection rule that
decides which tiles the attention kernel is given lives in
``../video_sparse_attn_h3.py``.
"""

from .kernels import (
    BLOCK_SIZE,
    block_sparse_attn_forward,
    pool_tiles,
    quantize_tiles,
)

__all__ = [
    "BLOCK_SIZE",
    "block_sparse_attn_forward",
    "pool_tiles",
    "quantize_tiles",
]
