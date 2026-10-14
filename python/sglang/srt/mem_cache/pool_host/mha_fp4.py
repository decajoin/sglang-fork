"""Host (L2) pool for block-scaled FP4 KV (nvfp4 / fp4_mx_block16).

Adapted from SGLang PR #43240 (9020817), with pre-capture staging reuse
and fail-fast validation of complete, aligned page transfers.

Why this class exists
---------------------
The generic MHA host pool sizes a host row as head_num*head_dim*itemsize
with head_dim in ELEMENTS, while an fp4 store dtype packs two elements
per byte: one device row holds row_dim/2 BYTES. HiCache JIT kernels share
one element size across both sides, so on the stock path:

  * H2D (load_to_device_per_layer) views the 256B device row as half of a
    512B element: token i lands at device rows 2*i and an in-range guard
    kills the tail of the copy;
  * D2H (staged write-back) gathers 512B spans off 256B-stride rows,
    pairing two device tokens per host row;
  * per-block scale buffers are never moved at all, so even correctly
    restored payloads dequantise with stale scales.

With MTP draft KV packed behind this pool the restored drafts are
garbage: speculative accept length silently pins at 1.0 after the first
evict -> loadback cycle (no crash, no warning).

Design (mirrors MHATokenToKVPoolMXFP8Host)
------------------------------------------
* Payload host rows are byte-rows: (head_dim/PACK_FACTOR) bytes per head
  row, so every JIT element size is square on both sides and the
  inherited transfers stay correct (packed MTP handling included).
* Block scales ride page-granular side buffers: a page's scales are one
  contiguous span inside every device scale tensor (both the interleaved
  [token, head, dim] layout and the TRT-LLM native [page, head, token,
  dim] layout are page-local), moved with the staged page kernels like
  MXFP8 UE8M0 scales.
* MTP draft layers pack into the tail layer of payload AND scale buffers.
* Transfers must contain complete, aligned pages on both sides. Reject
  partial or misordered pages before writing payload or scales.
* Mooncake L3 stores each K/V page as packed payload followed by its scale
  buffers via multi-buffer I/O. Flat-storage backends remain unsupported.
"""

from __future__ import annotations

from typing import Sequence

import torch

from sglang.kernels.ops.kvcache.hicache import (
    can_use_write_back_jit_kernel,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_mla_staged_lf_pf as jit_transfer_hicache_all_layer_mla_staged_lf_pf,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_one_layer_mla as jit_transfer_hicache_one_layer_mla,
)
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_unregister,
)
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKVPoolHost,
    _is_cuda,
    _is_hip,
)

PACK_FACTOR = 2  # fp4 elements per stored byte (float4_e2m1fn_x2)

# (attribute, rows_are_pages): interleaved scale tensors are shaped
# [tokens, ...]; TRT-LLM native ones are [pages, ...].
_SCALE_ATTRS = (
    ("k_scale_buffer", False),
    ("v_scale_buffer", False),
    ("native_k_scale_buffer", True),
    ("native_v_scale_buffer", True),
)


def _fp4_scale_buffers(pool):
    out = []
    for name, rows_are_pages in _SCALE_ATTRS:
        bufs = getattr(pool, name, None)
        if bufs is None or len(bufs) == 0 or bufs[0] is None:
            continue
        out.append((name, rows_are_pages, list(bufs)))
    return out


def is_fp4_packed_pool(pool) -> bool:
    """True when device rows are packed fp4 bytes and scales live apart.

    FP4 pools store packed bytes (uint8 store_dtype), so the store dtype
    cannot identify them; the quant method name is the real contract.
    """
    quant = getattr(pool, "quant_method", None)
    if getattr(quant, "name", None) not in ("nvfp4", "fp4_mx_block16"):
        return False
    return bool(_fp4_scale_buffers(pool))


class MHATokenToKVPoolFP4Host(MHATokenToKVPoolHost):
    """HiCache host pool for packed-FP4 KV with separate block-scale buffers."""

    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        mtp_draft_device_pools: Sequence = (),
        pool_label: str = "kv",
    ):
        if layout != "page_first":
            raise NotImplementedError(
                f"FP4 KV host pool supports only the page_first layout, got {layout!r}."
            )
        if page_size != device_pool.page_size:
            raise ValueError("FP4 host and device page sizes must match.")
        if getattr(device_pool, "layer_shard_enabled", False):
            raise NotImplementedError(
                "FP4 HiCache does not support layer-sharded pools yet."
            )
        if device_pool.head_dim != getattr(
            device_pool, "v_head_dim", device_pool.head_dim
        ):
            raise NotImplementedError(
                "FP4 KV host pool does not support asymmetric K/V head dims yet."
            )
        base_kinds = {n: (r, b) for n, r, b in _fp4_scale_buffers(device_pool)}
        if not base_kinds:
            raise ValueError("FP4 device pool exposes no scale buffers.")
        draft_lists: dict[str, list[torch.Tensor]] = {n: [] for n in base_kinds}
        for pool in mtp_draft_device_pools:
            if not is_fp4_packed_pool(pool) or pool.layer_num != 1:
                raise ValueError("Packed MTP drafts must be single-layer FP4 pools.")
            if (pool.row_dim, pool.head_dim, pool.v_head_dim) != (
                device_pool.row_dim,
                device_pool.head_dim,
                device_pool.v_head_dim,
            ):
                raise ValueError(
                    "Packed MTP drafts must match target payload geometry."
                )
            kinds = _fp4_scale_buffers(pool)
            if {n for n, _, _ in kinds} != set(base_kinds):
                raise ValueError(
                    "FP4 KV host pool needs every MTP draft pool to carry the "
                    "same scale buffers as the target pool."
                )
            if (
                getattr(pool, "page_size", device_pool.page_size)
                != device_pool.page_size
            ):
                raise ValueError("MTP draft pools must share the device page size.")
            if pool.k_buffer[0].shape[0] != device_pool.k_buffer[0].shape[0]:
                raise ValueError(
                    "Packed MTP drafts must share the target token capacity."
                )
            for n, _r, bs in kinds:
                draft_lists[n].extend(bs)
        # Bytes of fp4 scales one page holds per layer; needed by
        # get_size_per_token before the base constructor sizes the pool.
        self._scale_names = []
        self._scale_page_bytes = {}
        for name, (rows_are_pages, bufs) in base_kinds.items():
            b0 = bufs[0]
            if len(bufs) != device_pool.layer_num:
                raise ValueError(
                    "FP4 scale buffer count must match the device layer count."
                )
            if not rows_are_pages and b0.shape[0] % device_pool.page_size:
                raise ValueError(
                    "FP4 linear scales must contain complete device pages."
                )
            num_pages = (
                max(b0.shape[0], 1)
                if rows_are_pages
                else max(b0.shape[0] // device_pool.page_size, 1)
            )
            self._scale_names.append(name)
            self._scale_page_bytes[name] = int(
                b0.numel() * b0.element_size() // num_pages
            )
            page_bytes = self._scale_page_bytes[name]
            for buf in bufs + draft_lists[name]:
                rows = (
                    buf.shape[0]
                    if rows_are_pages
                    else buf.shape[0] // device_pool.page_size
                )
                if (
                    not buf.is_contiguous()
                    or rows == 0
                    or buf.nbytes != rows * page_bytes
                ):
                    raise ValueError(
                        "FP4 scale buffers must have matching contiguous page geometry."
                    )
            # target + draft scale tensors, in packed layer order
            setattr(self, f"_scale_dev_{name}", bufs + draft_lists[name])
            setattr(self, f"_scale_host_{name}", None)
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            mtp_draft_device_pools=mtp_draft_device_pools,
            pool_label=pool_label,
        )
        if self.page_size != device_pool.page_size:
            raise ValueError(
                "FP4 KV host pool moves scales per page, so the host page size "
                f"({self.page_size}) must equal the device page size "
                f"({device_pool.page_size})."
            )
        packed_row = device_pool.row_dim // PACK_FACTOR
        if self.token_stride_size != packed_row:
            raise AssertionError(
                f"host token stride {self.token_stride_size} != device packed "
                f"row {packed_row}"
            )
        if (
            not self.can_use_jit
            or not self.can_use_write_back_jit
            or not all(
                can_use_write_back_jit_kernel(element_size=size)
                for size in self._scale_page_bytes.values()
            )
        ):
            raise NotImplementedError(
                "FP4 KV host pool needs the JIT HiCache kernels "
                "(io_backend='kernel', CUDA or HIP)."
            )
        self._init_scale_buffers()

    # -- square payload geometry ---------------------------------------------

    def get_size_per_token(self):
        # BYTES per head row (fp4 packs two elements per byte), for K and
        # V over all packed layers, plus per-token scale bytes so host
        # budget accounting covers the side buffers too.
        dp = self.device_pool
        self.head_num = dp.row_dim // dp.head_dim
        self.head_dim = dp.head_dim // PACK_FACTOR
        self.dtype = torch.uint8
        self.layer_num = self.target_layer_num + len(self.mtp_draft_device_pools)
        payload = self.head_dim * self.head_num * self.layer_num * 2
        scales = (
            sum(self._scale_page_bytes[n] for n in self._scale_names) // self.page_size
        ) * self.layer_num
        return payload + scales

    def _init_write_back_staging_buffers(self):
        # Reuse this fork's pre-capture staging; its helper understands packed
        # byte geometry, avoiding a duplicate allocation after graph capture.
        super()._init_write_back_staging_buffers()

    # -- scale side channel ----------------------------------------------------

    def _init_scale_buffers(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        dev = self.device_pool.device
        for name in self._scale_names:
            page_bytes = self._scale_page_bytes[name]
            host = alloc_func(
                (self.page_num, self.layer_num, page_bytes),
                dtype=torch.uint8,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            setattr(self, f"_scale_host_{name}", host)
            # [page, layer, bytes] -> per-layer strided [page, bytes] views
            setattr(self, f"_scale_host_layers_{name}", list(host.transpose(0, 1)))
            bufs = getattr(self, f"_scale_dev_{name}")
            setattr(
                self,
                f"_scale_rows_{name}",
                [buf.view(torch.uint8).reshape(-1, page_bytes) for buf in bufs],
            )
            setattr(
                self,
                f"_scale_ptrs_{name}",
                torch.tensor(
                    [rows.data_ptr() for rows in getattr(self, f"_scale_rows_{name}")],
                    dtype=torch.uint64,
                    device=dev,
                ),
            )
            if self.staging_page_capacity > 0:
                setattr(
                    self,
                    f"_scale_staging_{name}",
                    torch.empty(
                        (self.staging_page_capacity, self.layer_num, page_bytes),
                        dtype=torch.uint8,
                        device=dev,
                    ),
                )
            else:
                setattr(self, f"_scale_staging_{name}", None)

    def _page_ids(self, indices: torch.Tensor) -> torch.Tensor:
        """Page-aligned token indices -> one page id per page, same device."""
        return indices[:: self.page_size] // self.page_size

    def get_hybrid_pool_buffer(self):
        return super().get_hybrid_pool_buffer() + [
            getattr(self, f"_scale_host_{name}") for name in self._scale_names
        ]

    def _validate_transfer(self, host_indices, device_indices, io_backend):
        if io_backend != "kernel":
            raise NotImplementedError("FP4 HiCache supports only io_backend='kernel'.")
        if host_indices.numel() != device_indices.numel():
            raise ValueError("FP4 host/device transfer index counts must match.")
        for indices, capacity in (
            (host_indices, self.size),
            (device_indices, self.device_pool.k_buffer[0].shape[0]),
        ):
            if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    "FP4 transfer indices must be one-dimensional integers."
                )
            if indices.numel() % self.page_size:
                raise ValueError("FP4 HiCache requires complete page transfers.")
            if indices.numel() == 0:
                continue
            rows = indices.reshape(-1, self.page_size)
            starts = rows[:, :1]
            expected = starts + torch.arange(self.page_size, device=indices.device)
            if (
                not torch.equal(rows, expected)
                or bool((starts % self.page_size).any())
                or bool((indices < 0).any())
                or bool((indices >= capacity).any())
            ):
                raise ValueError(
                    "FP4 HiCache requires aligned, ordered, in-range pages."
                )

    # -- transfers: payload via base (now square), scales via side channel ----

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        self._validate_transfer(host_indices, device_indices, io_backend)
        if not host_indices.numel():
            return
        if device_pool is not self.device_pool:
            raise AssertionError(
                "FP4 KV host pool expects write-backs through the packed "
                "target pool call; the engine must not back up draft pools "
                "separately."
            )
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        device_pages = self._page_ids(device_indices)
        host_pages = self._page_ids(host_indices)
        if host_pages.is_cuda:
            host_pages = host_pages.cpu()
        for name in self._scale_names:
            jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                ptr_src=getattr(self, f"_scale_ptrs_{name}"),
                src_indices=device_pages,
                dst_indices=host_pages,
                staging=getattr(self, f"_scale_staging_{name}"),
                dst=getattr(self, f"_scale_host_{name}"),
                page_size=1,
            )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        self._validate_transfer(host_indices, device_indices, io_backend)
        if not host_indices.numel():
            return
        if is_draft:
            draft_index = layer_id - self.target_layer_num
            if not (0 <= draft_index < len(self.mtp_draft_device_pools)) or (
                device_pool is not self.mtp_draft_device_pools[draft_index]
            ):
                raise ValueError(
                    "FP4 packed draft layer does not match its device pool."
                )
        elif device_pool is not self.device_pool:
            raise ValueError("FP4 loadback must use the registered device pool.")
        super().load_to_device_per_layer(
            device_pool,
            host_indices,
            device_indices,
            layer_id,
            io_backend,
            is_draft=is_draft,
        )
        if not is_draft and not self._is_device_layer_owned(device_pool, layer_id):
            return
        host_layer_id = layer_id if is_draft else self._host_layer_index(layer_id)
        # Scale row tables are packed: target layers keep their index, the
        # draft layer sits at the tail, exactly like the payload buffers.
        scale_layer = host_layer_id if is_draft else layer_id
        device_pages = self._page_ids(device_indices)
        host_pages = self._page_ids(host_indices)
        for name in self._scale_names:
            jit_transfer_hicache_one_layer_mla(
                cache_dst=getattr(self, f"_scale_rows_{name}")[scale_layer],
                indices_dst=device_pages,
                cache_src=getattr(self, f"_scale_host_layers_{name}")[host_layer_id],
                indices_src=host_pages,
                element_dim=self._scale_page_bytes[name],
            )

    def destroy(self):
        for name in getattr(self, "_scale_names", ()):
            host = getattr(self, f"_scale_host_{name}", None)
            if host is not None and self.pin_memory and (_is_cuda or _is_hip):
                _cuda_host_unregister(host)
            setattr(self, f"_scale_host_{name}", None)
            setattr(self, f"_scale_host_layers_{name}", None)
            setattr(self, f"_scale_staging_{name}", None)
            setattr(self, f"_scale_rows_{name}", None)
            setattr(self, f"_scale_ptrs_{name}", None)
            setattr(self, f"_scale_dev_{name}", None)
        super().destroy()

    # -- L3: one multi-buffer object per K/V page -----------------------------

    @property
    def storage_namespace(self) -> str:
        """Versioned physical format; never share keys with unpacked KV pages."""
        scales = "-".join(
            f"{name}.{self._scale_page_bytes[name]}" for name in self._scale_names
        )
        return (
            f"fp4-v1-{self.device_pool.quant_method.name}-pf"
            f"-p{self.page_size}-l{self.layer_num}-h{self.head_num}"
            f"-d{self.device_pool.head_dim}-{scales}"
        )

    def _storage_pages_unsupported(self) -> NotImplementedError:
        return NotImplementedError(
            "FP4 KV requires Mooncake multi-buffer L3 with page_first; "
            "flat storage and split-head pages are unsupported."
        )

    def get_data_page(self, index, flat: bool = True):
        raise self._storage_pages_unsupported()

    def get_dummy_flat_data_page(self):
        raise self._storage_pages_unsupported()

    def set_from_flat_data_page(self, index: int, data_page) -> None:
        raise self._storage_pages_unsupported()

    def get_page_buffer_meta(self, indices):
        # Validate before exposing pointers to the storage SDK. Host indices
        # may name nonconsecutive pages, but each page must be complete.
        self._validate_storage_indices(indices)
        payload_ptrs, payload_sizes = super().get_page_buffer_meta(indices)
        ptrs, sizes = [], []
        for i, page in enumerate(self._page_ids(indices).tolist()):
            for kv, prefix in enumerate(("k_", "v_")):
                object_ptrs = [payload_ptrs[2 * i + kv]]
                object_sizes = [payload_sizes[2 * i + kv]]
                for name in self._scale_names:
                    if name.startswith(prefix) or name.startswith("native_" + prefix):
                        buf = getattr(self, f"_scale_host_{name}")
                        page_bytes = self.layer_num * self._scale_page_bytes[name]
                        object_ptrs.append(buf.data_ptr() + page * page_bytes)
                        object_sizes.append(page_bytes)
                ptrs.append(object_ptrs)
                sizes.append(object_sizes)
        return ptrs, sizes

    def _validate_storage_indices(self, indices):
        if indices.device.type != "cpu":
            raise ValueError("FP4 L3 indices must reside on CPU.")
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("FP4 L3 indices must be one-dimensional integers.")
        if indices.numel() % self.page_size:
            raise ValueError("FP4 L3 requires complete pages.")
        rows = indices.reshape(-1, self.page_size)
        starts = rows[:, :1]
        if (
            not torch.equal(rows, starts + torch.arange(self.page_size))
            or bool((starts % self.page_size).any())
            or bool((indices < 0).any())
            or bool((indices >= self.size).any())
        ):
            raise ValueError("FP4 L3 requires aligned, ordered, in-range pages.")

    def get_split_heads_page_buffer_meta(
        self, indices: torch.Tensor, split_factor: int
    ):
        raise self._storage_pages_unsupported()
