# SPDX-License-Identifier: Apache-2.0
"""Standalone checks for the opt-in loader; no model/server is needed."""

import hashlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.kernels.ops.attention import fp8_prefill as loader
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-small")


def plan_args(dim=256, kv_dtype=torch.float8_e4m3fn):
    return (
        torch.tensor([0, 17], dtype=torch.int32),
        torch.tensor([0, 9], dtype=torch.int32),
        torch.arange(9, dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
        8,
        2,
        dim,
        128,
    ), dict(q_data_type=torch.bfloat16, kv_data_type=kv_dtype, causal=True)


def test_overlay_is_exact_and_leaves_dependency_untouched():
    path = loader.jit_env.FLASHINFER_INCLUDE_DIR / loader._HEADER
    before = path.read_bytes()
    assert hashlib.sha256(before).hexdigest() == loader._BASE_SHA256
    directory, _ = loader._overlay()
    header = (directory / loader._HEADER).read_text()
    assert "GlobalUpcastTraits" in header
    assert "FP8_PROBE" not in header
    assert "SGLANG_FP8_PREFILL_LOAD_UPCAST" in header
    assert path.read_bytes() == before


def test_patch_rejects_changed_context():
    with pytest.raises(RuntimeError, match="context mismatch"):
        loader._apply_patch("original\n", "@@ -1 +1 @@\n-other\n+new\n")


def test_header_version_guard(tmp_path):
    header = tmp_path / loader._HEADER
    header.parent.mkdir(parents=True)
    header.write_text("changed upstream header")
    with patch.object(loader.jit_env, "FLASHINFER_INCLUDE_DIR", tmp_path):
        with pytest.raises(RuntimeError, match="pinned FlashInfer"):
            loader._overlay.__wrapped__()


def test_package_version_guard():
    with patch.object(loader, "_FLASHINFER_VERSION", "0.6.19"):
        with pytest.raises(RuntimeError, match="pinned FlashInfer"):
            loader._overlay.__wrapped__()


def test_supported_and_fallback_plans_do_not_patch_flashinfer():
    import flashinfer.prefill as upstream

    original = upstream.gen_batch_prefill_module
    wrapper = object.__new__(loader.FP8PrefillWorkspaceWrapper)
    wrapper._backend = "fa2"
    selected = object()
    with patch.object(loader, "_module", return_value=selected) as build:
        args, kwargs = plan_args()
        wrapper._select_module(args, kwargs)
        assert wrapper._jit_module is selected
        for dim, dtype in ((128, torch.float8_e4m3fn), (256, torch.bfloat16)):
            wrapper._select_module(*plan_args(dim, dtype))
            assert wrapper._jit_module is None
        for extra in (
            dict(window_left=128),
            dict(logits_soft_cap=10.0),
            dict(pos_encoding_mode="ALIBI"),
            dict(use_fp16_qk_reduction=True),
            dict(custom_mask=torch.ones(1, dtype=torch.bool)),
        ):
            wrapper._select_module(args, dict(kwargs, **extra))
            assert wrapper._jit_module is None
        wrapper._backend = "fa3"
        wrapper._select_module(args, kwargs)
        assert wrapper._jit_module is None
        assert build.call_count == 1
    assert upstream.gen_batch_prefill_module is original


def test_workspace_size_query_restores_module_selection():
    wrapper = object.__new__(loader.FP8PrefillWorkspaceWrapper)
    wrapper._backend = "fa2"
    wrapper._jit_module = None
    wrapper._jit_additional_tensor_names = []
    wrapper._jit_additional_scalar_names = []
    selected = object()

    def query(self, *args, **kwargs):
        assert self._jit_module is selected
        return (96 * 1024 * 1024, 8 * 1024 * 1024)

    with (
        patch.object(loader, "_module", return_value=selected),
        patch.object(
            loader.BatchPrefillWithPagedKVCacheWrapper, "workspace_size", query
        ),
    ):
        args, kwargs = plan_args()
        assert wrapper.workspace_size(*args, **kwargs)[0] == 96 * 1024 * 1024
    assert wrapper._jit_module is None
    assert wrapper._jit_additional_tensor_names == []
    assert wrapper._jit_additional_scalar_names == []


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("native_fp8", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.float8_e4m3fn, torch.bfloat16, torch.float8_e5m2]
)
def test_backend_selects_loader_by_input_dtype(enabled, native_fp8, dtype):
    from sglang.srt.layers.attention import flashinfer_backend as backend

    instance = object.__new__(backend.FlashInferAttnBackend)
    instance.is_nvfp4_kvcache = not native_fp8
    instance.prefill_uses_dequant_workspace = not native_fp8
    instance.flashinfer_kv_cache_dtype = dtype
    args = SimpleNamespace(enable_fp8_prefill_workspace_optimization=enabled)
    selected = instance._select_paged_prefill_wrapper_cls(args)
    expected = (
        loader.FP8PrefillWorkspaceWrapper
        if enabled and dtype == torch.float8_e4m3fn
        else backend.BatchPrefillWithPagedKVCacheWrapper
    )
    assert selected is expected


def test_backend_full_prefill_graph_uses_configured_wrapper():
    from sglang.srt.layers.attention import flashinfer_backend as backend

    instance = object.__new__(backend.FlashInferAttnBackend)
    with patch.object(backend, "BatchPrefillWithPagedKVCacheWrapper") as original:
        instance._make_paged_prefill_wrapper("buffer", backend="fa2")
        original.assert_called_once_with("buffer", backend="fa2")
    selected = Mock()
    instance._paged_prefill_wrapper_cls = selected
    instance.workspace_buffer = torch.empty(1, dtype=torch.uint8)
    instance.indices_updater_prefill = SimpleNamespace(
        num_qo_heads=8, num_kv_heads=2, head_dim=256
    )
    instance.num_wrappers = 1
    instance.max_context_len = 1027
    instance.prefill_backend = "fa2"
    instance.kv_last_page_len = torch.tensor([3], dtype=torch.int32)
    with patch.object(instance, "_full_cg_prefill_workspace_bytes", return_value=128):
        wrappers = instance._create_full_cg_prefill_wrappers(1, 17)
    assert wrappers == [selected.return_value]
    assert selected.call_args.kwargs["use_cuda_graph"] is True
    assert selected.call_args.kwargs["backend"] == "fa2"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    "length,queries", [(1027, 1), (1027, 17), (1027, 128), (8192, 256), (98304, 17)]
)
def test_cuda_matches_bfloat16_reference(layout, length, queries):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("validated on SM120")
    torch.manual_seed(123)
    page_size = 128
    pages = (length + page_size - 1) // page_size
    shape = (
        (pages, page_size, 2, 256) if layout == "NHD" else (pages, 2, page_size, 256)
    )
    k, v = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        for _ in range(2)
    ]
    q = torch.randn((queries, 8, 256), device="cuda", dtype=torch.bfloat16)
    # Attention scratch, not KV data: the split-K planner needs ~77 MiB for
    # the largest query bucket. Use the same 96 MiB for both implementations.
    workspace = torch.empty(96 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    args = (
        torch.tensor([0, queries], dtype=torch.int32),
        torch.tensor([0, pages], dtype=torch.int32),
        torch.randperm(pages, dtype=torch.int32),
        torch.tensor([(length - 1) % page_size + 1], dtype=torch.int32),
        8,
        2,
        256,
        page_size,
    )
    graph_args = (
        dict(
            use_cuda_graph=True,
            qo_indptr_buf=args[0].cuda(),
            paged_kv_indptr_buf=args[1].cuda(),
            paged_kv_indices_buf=args[2].cuda(),
            paged_kv_last_page_len_buf=args[3].cuda(),
        )
        if length == 1027 and queries == 17
        else {}
    )
    optimized = loader.FP8PrefillWorkspaceWrapper(
        workspace, layout, backend="fa2", **graph_args
    )
    reference = loader.BatchPrefillWithPagedKVCacheWrapper(
        workspace, layout, backend="fa2", **graph_args
    )
    kwargs = dict(q_data_type=torch.bfloat16, causal=True)
    optimized.plan(*args, kv_data_type=torch.float8_e4m3fn, **kwargs)
    reference.plan(*args, kv_data_type=torch.bfloat16, **kwargs)
    assert optimized._jit_module is not None
    expected = reference.run(
        q, (k.to(torch.bfloat16), v.to(torch.bfloat16)), k_scale=0.75, v_scale=1.25
    )
    actual = optimized.run(q, (k, v), k_scale=0.75, v_scale=1.25)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    out = torch.empty_like(actual)
    optimized.run(q, (k, v), out=out, k_scale=0.75, v_scale=1.25)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    optimized.run(q, (k, v), out=out, k_scale=0.75, v_scale=1.25)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    if length == 1027 and queries == 17:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            optimized.run(q, (k, v), out=out, k_scale=0.75, v_scale=1.25)
        out.zero_()
        graph.replay()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
