"""CUDA regression coverage for the NVFP4 history dequantization path."""

import pytest
import torch

from sglang.srt.layers.quantization.nvfp4_kv_dequant import dequantize_nvfp4_to_fp8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(packed, scales, global_scale):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
    )
    values = lut[codes] * scales.view(torch.float8_e4m3fn).float().repeat_interleave(
        16, -1
    )
    return (values * global_scale).clamp(-448, 448).to(torch.float8_e4m3fn)


@pytest.mark.parametrize("global_value", [1.0, 0.001, 0.37, 16.0])
def test_all_codes_and_block_scales(global_value):
    packed = (
        torch.arange(256, device="cuda")
        .to(torch.uint8)[None, None, :]
        .expand(256, 2, 256)
    )
    scales = (
        torch.arange(256, device="cuda")
        .to(torch.uint8)[:, None, None]
        .expand(256, 2, 32)
    )
    global_scale = torch.tensor([global_value], device="cuda")
    result = dequantize_nvfp4_to_fp8(packed, scales, global_scale)
    torch.testing.assert_close(
        result.float(),
        reference(packed, scales, global_scale).float(),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "tokens,heads,dim", [(0, 2, 256), (1, 2, 256), (3, 3, 64), (17, 2, 256)]
)
def test_strided_inputs_and_tail(tokens, heads, dim):
    packed = torch.randint(
        256, (tokens * 2, heads, dim), dtype=torch.uint8, device="cuda"
    )[::2, :, ::2]
    scales = torch.full((tokens * 2, heads, dim // 8), 1.0, device="cuda").to(
        torch.float8_e4m3fn
    )[::2, :, ::2]
    global_scale = torch.tensor([0.37], device="cuda")
    result = dequantize_nvfp4_to_fp8(packed, scales, global_scale)
    torch.testing.assert_close(
        result.float(), reference(packed, scales, global_scale).float(), rtol=0, atol=0
    )


def test_cuda_graph_replay():
    packed = torch.zeros((17, 2, 128), dtype=torch.uint8, device="cuda")
    scales = torch.ones((17, 2, 16), device="cuda").to(torch.float8_e4m3fn)
    global_scale = torch.ones(1, device="cuda")
    dequantize_nvfp4_to_fp8(packed, scales, global_scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = dequantize_nvfp4_to_fp8(packed, scales, global_scale)
    packed.fill_(0x76)
    graph.replay()
    torch.testing.assert_close(
        result.float(), reference(packed, scales, global_scale).float(), rtol=0, atol=0
    )


def test_86016_token_peak_memory():
    from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

    packed = torch.full((86016, 2, 128), 0x76, dtype=torch.uint8, device="cuda")
    scales = torch.ones((86016, 2, 16), device="cuda").to(torch.float8_e4m3fn)
    global_scale = torch.ones(1, device="cuda")

    def legacy():
        k = NVFP4KVQuantizeUtil.dequantize(packed, scales, global_scale)
        v = NVFP4KVQuantizeUtil.dequantize(packed, scales, global_scale)
        return k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)

    def direct():
        return tuple(
            NVFP4KVQuantizeUtil.dequantize(
                packed, scales, global_scale, dtype=torch.float8_e4m3fn
            )
            for _ in range(2)
        )

    peaks = []
    for fn in (legacy, direct):
        warm = fn()
        del warm
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = fn()
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - baseline)
        assert result[0].dtype == torch.float8_e4m3fn
        del result
    print(f"legacy/direct peak MiB: {[p / 2**20 for p in peaks]}")
    assert peaks[0] - peaks[1] >= 168 * 2**20
