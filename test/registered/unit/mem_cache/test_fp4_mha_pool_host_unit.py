"""Packed FP4 payload and its block scales must travel together through L2."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKVPoolHost,
    get_mha_host_pool_cls,
    prepare_mha_write_back_staging,
)
from sglang.srt.mem_cache.pool_host.mha_fp4 import MHATokenToKVPoolFP4Host
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_pool(device="cpu", page_size=4, layers=2, native=False):
    tokens, heads, dim = 4 * page_size, 2, 256
    pool = SimpleNamespace(
        quant_method=SimpleNamespace(name="nvfp4"),
        head_num=heads,
        head_dim=dim,
        v_head_dim=dim,
        row_dim=heads * dim,
        layer_num=layers,
        start_layer=0,
        end_layer=layers,
        layer_shard_enabled=False,
        size=tokens - page_size,
        page_size=page_size,
        device=device,
        store_dtype=torch.uint8,
        hicache_write_back_staging=None,
        k_buffer=[
            torch.zeros(tokens, heads, dim // 2, dtype=torch.uint8, device=device)
            for _ in range(layers)
        ],
        v_buffer=[
            torch.zeros(tokens, heads, dim // 2, dtype=torch.uint8, device=device)
            for _ in range(layers)
        ],
        k_scale_buffer=[
            torch.zeros(tokens, heads, dim // 16, dtype=torch.uint8, device=device)
            for _ in range(layers)
        ],
        v_scale_buffer=[
            torch.zeros(tokens, heads, dim // 16, dtype=torch.uint8, device=device)
            for _ in range(layers)
        ],
    )
    if native:
        for name in ("native_k_scale_buffer", "native_v_scale_buffer"):
            setattr(
                pool,
                name,
                [
                    torch.zeros(
                        4, heads, page_size, dim // 16, dtype=torch.uint8, device=device
                    )
                    for _ in range(layers)
                ],
            )
    pool.k_data_ptrs = torch.tensor(
        [b.data_ptr() for b in pool.k_buffer], dtype=torch.uint64, device=device
    )
    pool.v_data_ptrs = torch.tensor(
        [b.data_ptr() for b in pool.v_buffer], dtype=torch.uint64, device=device
    )
    return pool


def make_host(pool, drafts=()):
    with ExitStack() as stack:
        stack.enter_context(
            patch.dict(
                ALLOC_MEMORY_FUNCS,
                {"cpu": lambda dims, **kw: torch.empty(dims, dtype=kw["dtype"])},
            )
        )
        stack.enter_context(patch("sglang.srt.mem_cache.pool_host.mha._is_cuda", True))
        for module in ("mha", "mha_fp4"):
            stack.enter_context(
                patch(
                    f"sglang.srt.mem_cache.pool_host.{module}.can_use_write_back_jit_kernel",
                    return_value=True,
                )
            )
        stack.enter_context(
            patch(
                "sglang.srt.mem_cache.pool_host.mha.can_use_hicache_jit_kernel",
                return_value=True,
            )
        )
        return MHATokenToKVPoolFP4Host(
            pool,
            2,
            0,
            pool.page_size,
            "page_first",
            pin_memory=False,
            mtp_draft_device_pools=drafts,
        )


class TestFP4HostPool(unittest.TestCase):
    def test_factory_selects_packed_fp4_host(self):
        self.assertEqual(
            get_mha_host_pool_cls(make_pool()).__name__, "MHATokenToKVPoolFP4Host"
        )

    def test_prepared_staging_uses_packed_byte_width(self):
        pool = make_pool()
        with (
            patch("sglang.srt.mem_cache.pool_host.mha._is_cuda", True),
            patch(
                "sglang.srt.mem_cache.pool_host.mha.can_use_write_back_jit_kernel",
                return_value=True,
            ),
        ):
            staging = prepare_mha_write_back_staging(
                pool, layer_num=2, page_size=4, page_capacity=4
            )
        self.assertEqual(tuple(staging[0].shape), (16, 2, 2, 128))

    def test_geometry_budget_and_staging_reuse(self):
        pool = make_pool()
        with (
            patch("sglang.srt.mem_cache.pool_host.mha._is_cuda", True),
            patch(
                "sglang.srt.mem_cache.pool_host.mha.can_use_write_back_jit_kernel",
                return_value=True,
            ),
        ):
            prepared = prepare_mha_write_back_staging(
                pool, layer_num=2, page_size=4, page_capacity=7
            )
        host = make_host(pool)
        self.assertEqual(host.size_per_token, 1152)
        self.assertEqual(host.token_stride_size, 256)
        self.assertEqual(host.head_dim, 128)
        self.assertEqual(host.staging_k_buffer.data_ptr(), prepared[0].data_ptr())
        self.assertEqual(tuple(host.k_buffer.shape[1:]), (2, 2, 128))
        host.destroy()
        host.destroy()

    def test_payload_and_all_scales_restore_into_different_pages(self):
        for packed_draft in (False, True):
            with self.subTest(packed_draft=packed_draft):
                pool = make_pool(native=True)
                drafts = (make_pool(layers=1, native=True),) if packed_draft else ()
                host = make_host(pool, drafts)
                pools = (pool, *drafts)
                payload_k = [b for p in pools for b in p.k_buffer]
                payload_v = [b for p in pools for b in p.v_buffer]
                all_buffers = (
                    payload_k
                    + payload_v
                    + [
                        b
                        for name in host._scale_names
                        for b in getattr(host, f"_scale_dev_{name}")
                    ]
                )
                for i, buf in enumerate(all_buffers):
                    rows = torch.arange(buf.shape[0])
                    tags = (
                        rows * 53
                        if buf.ndim == 4
                        else (rows % 4) * 17 + (rows // 4) * 53
                    )
                    tags = tags.reshape(-1, *([1] * (buf.ndim - 1)))
                    buf.copy_(
                        (torch.arange(buf.numel()).reshape_as(buf) + tags + 17 * i).to(
                            torch.uint8
                        )
                    )
                expected_k = [b[4:12].clone() for b in payload_k]
                expected_v = [b[4:12].clone() for b in payload_v]
                expected_scales = {
                    name: [b[1:3].clone() for b in getattr(host, f"_scale_rows_{name}")]
                    for name in host._scale_names
                }
                registry = {
                    tuple(getattr(host, f"_scale_ptrs_{name}").tolist()): getattr(
                        host, f"_scale_rows_{name}"
                    )
                    for name in host._scale_names
                }
                source = torch.arange(4, 12)
                host_ids = torch.cat((torch.arange(8, 12), torch.arange(0, 4)))
                destination = torch.cat((torch.arange(12, 16), torch.arange(0, 4)))

                def payload_backup(**kw):
                    for i, (k, v) in enumerate(zip(payload_k, payload_v)):
                        kw["dst_k"][kw["dst_indices"], i] = k[kw["src_indices"]]
                        kw["dst_v"][kw["dst_indices"], i] = v[kw["src_indices"]]

                def payload_load(**kw):
                    for kind in ("k", "v"):
                        kw[f"{kind}_cache_dst"][kw["indices_dst"]] = kw[
                            f"{kind}_cache_src"
                        ][kw["indices_src"]]

                def scale_backup(**kw):
                    for i, b in enumerate(registry[tuple(kw["ptr_src"].tolist())]):
                        kw["dst"][kw["dst_indices"], i] = b[kw["src_indices"]]

                def scale_load(**kw):
                    kw["cache_dst"][kw["indices_dst"]] = kw["cache_src"][
                        kw["indices_src"]
                    ]

                with (
                    patch(
                        "sglang.srt.mem_cache.pool_host.mha.jit_transfer_hicache_all_layer_staged_lf_pf",
                        side_effect=payload_backup,
                    ),
                    patch(
                        "sglang.srt.mem_cache.pool_host.mha.jit_transfer_hicache_one_layer",
                        side_effect=payload_load,
                    ),
                    patch(
                        "sglang.srt.mem_cache.pool_host.mha_fp4.jit_transfer_hicache_all_layer_mla_staged_lf_pf",
                        side_effect=scale_backup,
                    ),
                    patch(
                        "sglang.srt.mem_cache.pool_host.mha_fp4.jit_transfer_hicache_one_layer_mla",
                        side_effect=scale_load,
                    ),
                ):
                    host.backup_from_device_all_layer(pool, host_ids, source, "kernel")
                    for buf in all_buffers:
                        buf.zero_()
                    for layer in range(2):
                        host.load_to_device_per_layer(
                            pool, host_ids, destination, layer, "kernel"
                        )
                    if drafts:
                        host.load_to_device_per_layer(
                            drafts[0], host_ids, destination, 2, "kernel", is_draft=True
                        )
                for actual, expected in zip(payload_k, expected_k):
                    self.assertTrue(torch.equal(actual[destination], expected))
                for actual, expected in zip(payload_v, expected_v):
                    self.assertTrue(torch.equal(actual[destination], expected))
                pages = destination[::4] // 4
                for name in host._scale_names:
                    for actual, expected in zip(
                        getattr(host, f"_scale_rows_{name}"), expected_scales[name]
                    ):
                        self.assertTrue(torch.equal(actual[pages], expected))
                host.destroy()

    def test_bad_transfers_fail_before_any_payload_write(self):
        pool = make_pool()
        host = make_host(pool)
        good = torch.arange(4, 8)
        bad = (
            torch.arange(1, 5),
            torch.tensor([4, 6, 5, 7]),
            torch.arange(4, 7),
            torch.arange(-4, 0),
        )
        with patch.object(
            MHATokenToKVPoolHost, "backup_from_device_all_layer"
        ) as payload:
            for ids in bad:
                for host_ids, device_ids in ((ids, good), (good, ids)):
                    with self.assertRaises(ValueError):
                        host.backup_from_device_all_layer(
                            pool, host_ids, device_ids, "kernel"
                        )
            with self.assertRaises(NotImplementedError):
                host.backup_from_device_all_layer(pool, good, good, "direct")
            payload.assert_not_called()
        host.destroy()

    def test_l3_rejected(self):
        host = make_host(make_pool())
        for fn, args in (
            (host.get_data_page, (0,)),
            (host.get_dummy_flat_data_page, ()),
            (host.set_from_flat_data_page, (0, None)),
            (host.get_page_buffer_meta, (torch.arange(4),)),
        ):
            with self.assertRaises(NotImplementedError):
                fn(*args)
        host.destroy()

    def test_constructor_rejects_unsupported_geometry_before_allocation(self):
        pool = make_pool()
        with self.assertRaises(NotImplementedError):
            MHATokenToKVPoolFP4Host(pool, 2, 0, 4, "layer_first")
        with self.assertRaises(ValueError):
            MHATokenToKVPoolFP4Host(pool, 2, 0, 8, "page_first")
        pool.v_head_dim = 128
        with self.assertRaises(NotImplementedError):
            make_host(pool)
        pool = make_pool()
        pool.layer_shard_enabled = True
        with self.assertRaises(NotImplementedError):
            make_host(pool)
        pool = make_pool()
        pool.k_scale_buffer[0] = pool.k_scale_buffer[0][:-1]
        with self.assertRaises(ValueError):
            make_host(pool)
        pool = make_pool()
        pool.k_scale_buffer[0] = pool.k_scale_buffer[0].transpose(1, 2)
        with self.assertRaises(ValueError):
            make_host(pool)
        with self.assertRaises(ValueError):
            make_host(make_pool(), (make_pool(layers=2),))

    def test_plain_pool_still_uses_generic_host(self):
        pool = make_pool()
        pool.quant_method.name = "fp8"
        self.assertIs(get_mha_host_pool_cls(pool), MHATokenToKVPoolHost)


if __name__ == "__main__":
    unittest.main()
