# SPDX-License-Identifier: Apache-2.0
"""Opt-in SM120 FA2 loader: FP8 global KV -> BF16 registers -> BF16 shared KV.

Use FlashInfer's custom-JIT slot on individual paged wrappers. Never patch its
Python globals or installed headers. The pinned 0.6.18 header is overlaid in
the JIT cache; an upstream header change fails closed, only when opted in.
"""

import functools
import hashlib
import inspect
import logging
import re
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import torch
from filelock import FileLock
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from flashinfer import __version__ as _FLASHINFER_VERSION
from flashinfer.jit import env as jit_env
from flashinfer.jit.attention import gen_batch_prefill_module
from flashinfer.prefill import get_batch_prefill_jit_module
from flashinfer.utils import canonicalize_torch_dtype

logger = logging.getLogger(__name__)

_HEADER = "flashinfer/attention/prefill.cuh"
_BASE_SHA256 = "f93cbea96e3a723254c15f5a52d6dd4c13e3892e88eb9b9fed1c0c0048c7d710"
_PLAN_SIGNATURE = inspect.signature(BatchPrefillWithPagedKVCacheWrapper.plan)
_TENSORS = [
    "maybe_custom_mask",
    "maybe_mask_indptr",
    "maybe_alibi_slopes",
    "maybe_prefix_len_ptr",
    "maybe_token_pos_in_items_ptr",
    "maybe_max_item_len_ptr",
    "maybe_k_cache_sf",
    "maybe_v_cache_sf",
]
_SCALARS = [
    "logits_soft_cap",
    "sm_scale",
    "rope_rcp_scale",
    "rope_rcp_theta",
    "token_pos_in_items_len",
]


def _apply_patch(source: str, patch: str) -> str:
    """Apply exact unified-diff hunks, without fuzz or an external patch tool."""
    lines = source.splitlines(keepends=True)
    result = []
    cursor = 0
    in_hunk = False
    for line in patch.splitlines(keepends=True):
        if line.startswith("@@ "):
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
            if match is None:
                raise RuntimeError("Invalid FP8 prefill patch hunk")
            start = int(match[1]) - 1
            if start < cursor:
                raise RuntimeError("Overlapping FP8 prefill patch hunks")
            result.extend(lines[cursor:start])
            cursor = start
            in_hunk = True
        elif in_hunk:
            if line[0] in " -":
                if cursor >= len(lines) or lines[cursor] != line[1:]:
                    raise RuntimeError(
                        "FlashInfer FP8 prefill header patch context mismatch"
                    )
                cursor += 1
            if line[0] in " +":
                result.append(line[1:])
    if not in_hunk:
        raise RuntimeError("Empty FP8 prefill header patch")
    result.extend(lines[cursor:])
    return "".join(result)


@functools.cache
def _overlay() -> tuple[Path, str]:
    source = (jit_env.FLASHINFER_INCLUDE_DIR / _HEADER).read_bytes()
    if (
        _FLASHINFER_VERSION != "0.6.18"
        or hashlib.sha256(source).hexdigest() != _BASE_SHA256
    ):
        raise RuntimeError(
            "--enable-fp8-prefill-workspace-optimization requires the pinned "
            "FlashInfer 0.6.18 prefill header. Disable the option or install "
            "the repository's pinned FlashInfer dependency."
        )
    patch = Path(__file__).with_name("fp8_prefill_loader.patch").read_bytes()
    content = _apply_patch(source.decode(), patch.decode())
    # Key the actual compiled header, not insignificant diff context/formatting.
    digest = hashlib.sha256(content.encode()).hexdigest()[:16]
    parent = jit_env.FLASHINFER_GEN_SRC_DIR / "sglang_fp8_prefill"
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / digest
    # TP workers can initialize concurrently. Publish only complete overlays.
    with FileLock(str(directory) + ".lock"):
        if not directory.exists():
            with tempfile.TemporaryDirectory(dir=parent) as temporary:
                include = Path(temporary) / "include"
                # Quoted ../ includes require the complete relative tree.
                shutil.copytree(jit_env.FLASHINFER_INCLUDE_DIR, include)
                (include / _HEADER).write_text(content)
                include.rename(directory)
        if (directory / _HEADER).read_text() != content:
            raise RuntimeError("Corrupt cached FP8 prefill header overlay")
    return directory, digest


@functools.cache
def _module():
    include, digest = _overlay()
    spec = gen_batch_prefill_module(
        "fa2",
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.bfloat16,
        torch.int32,
        256,
        256,
        0,
        False,
        False,
        False,
    )
    # Copy the spec; a baseline wrapper must retain its original cached module.
    spec = replace(
        spec,
        name=spec.name + "_sglang_fp8_upcast_" + digest,
        extra_include_dirs=[include] + list(spec.extra_include_dirs or []),
        extra_cuda_cflags=list(spec.extra_cuda_cflags or [])
        + ["-DSGLANG_FP8_PREFILL_LOAD_UPCAST=1"],
    )
    logger.info("Using optimized FP8 prefill workspace loader (FA2, BF16 Q, D256)")
    return get_batch_prefill_jit_module(spec.name, spec.build_and_load())


def _supports_plan(p: dict) -> bool:
    q = canonicalize_torch_dtype(p["q_data_type"])
    kv = canonicalize_torch_dtype(p["kv_data_type"] or q)
    out = canonicalize_torch_dtype(p["o_data_type"] or q)
    return (
        q == out == torch.bfloat16
        and kv == torch.float8_e4m3fn
        and p["paged_kv_indptr"].dtype == torch.int32
        and p["head_dim_qk"] == (p["head_dim_vo"] or p["head_dim_qk"]) == 256
        and p["pos_encoding_mode"] == "NONE"
        and p["window_left"] < 0
        and not p["logits_soft_cap"]
        and not p["use_fp16_qk_reduction"]
        and all(
            p[k] is None
            for k in (
                "custom_mask",
                "packed_custom_mask",
                "prefix_len_ptr",
                "token_pos_in_items_ptr",
                "max_item_len_ptr",
            )
        )
    )


class FP8PrefillWorkspaceWrapper(BatchPrefillWithPagedKVCacheWrapper):
    """Use the optimized module only for validated plans; otherwise use upstream."""

    def _select_module(self, args, kwargs):
        bound = _PLAN_SIGNATURE.bind(self, *args, **kwargs)
        bound.apply_defaults()
        supported = self._backend == "fa2" and _supports_plan(bound.arguments)
        self._jit_module = _module() if supported else None
        self._jit_additional_tensor_names = list(_TENSORS) if supported else []
        self._jit_additional_scalar_names = list(_SCALARS) if supported else []

    def plan(self, *args, **kwargs):
        self._select_module(args, kwargs)
        return super().plan(*args, **kwargs)

    # Some callers use this alias instead of plan().
    begin_forward = plan

    def workspace_size(self, *args, **kwargs):
        # Keep the query non-mutating, just like FlashInfer's upstream method.
        saved = (
            self._jit_module,
            self._jit_additional_tensor_names,
            self._jit_additional_scalar_names,
        )
        try:
            self._select_module(args, kwargs)
            return super().workspace_size(*args, **kwargs)
        finally:
            (
                self._jit_module,
                self._jit_additional_tensor_names,
                self._jit_additional_scalar_names,
            ) = saved
