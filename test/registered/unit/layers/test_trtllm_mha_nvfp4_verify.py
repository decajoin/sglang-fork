"""Exercise packed NVFP4 verification at the actual backend call boundary."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention import trtllm_mha_backend as backend_module
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="extra-a", runner_config="1-gpu-small")


class TestNVFP4Verify(unittest.TestCase):
    def make_backend(self, *, xqa=True, nvfp4=True):
        backend = object.__new__(backend_module.TRTLLMHAAttnBackend)
        backend.is_xqa_impl = xqa
        backend.is_nvfp4_kvcache = nvfp4
        backend.uses_trtllm_gen_native_fp4 = nvfp4 and not xqa
        backend.prefill_uses_native_fp4 = False
        backend.decode_uses_native_fp4 = nvfp4
        return backend

    def test_verify_selects_packed_kv_on_xqa_and_genmha(self):
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        for xqa in (True, False):
            with self.subTest(xqa=xqa):
                backend = self.make_backend(xqa=xqa)
                self.assertTrue(backend._forward_extend_uses_native_fp4(batch))

    def test_non_fp4_verify_and_xqa_prefill_keep_existing_path(self):
        for mode, nvfp4 in (
            (ForwardMode.TARGET_VERIFY, False),
            (ForwardMode.EXTEND, True),
        ):
            with self.subTest(mode=mode, nvfp4=nvfp4):
                backend = self.make_backend(nvfp4=nvfp4)
                batch = SimpleNamespace(forward_mode=mode)
                self.assertFalse(backend._forward_extend_uses_native_fp4(batch))

    def test_genmha_native_prefill_keeps_packed_path(self):
        backend = self.make_backend(xqa=False)
        backend.prefill_uses_native_fp4 = True
        self.assertTrue(
            backend._forward_extend_uses_native_fp4(
                SimpleNamespace(forward_mode=ForwardMode.EXTEND)
            )
        )

    def test_xqa_verify_preserves_bf16_and_passes_packed_kv_and_scales(self):
        for pages in (1, 2):
            with self.subTest(pages=pages):
                backend = self.make_backend()
                backend.page_size = 128
                backend.use_fmha_v2 = False
                backend.data_type = torch.float8_e4m3fn
                backend.q_data_type = torch.bfloat16
                packed = torch.zeros(pages * 128, 2, 128, dtype=torch.uint8)
                scales = torch.zeros(pages * 128, 2, 16, dtype=torch.float8_e4m3fn)
                backend.token_to_kv_pool = SimpleNamespace(
                    get_raw_kv_buffer=Mock(
                        return_value=(packed, packed, scales, scales)
                    ),
                    get_kv_buffer=Mock(side_effect=AssertionError("plain KV read")),
                )
                backend._should_use_fused_fp8_path = Mock(return_value=False)
                backend._get_nvfp4_bmm_scales = Mock(return_value=(2.0, 3.0))
                backend._get_layer_page_table = Mock(return_value=torch.zeros(1, 1))
                backend._nvfp4_output_view = Mock(
                    side_effect=AssertionError(
                        "XQA must not allocate GenMHA FP8 output"
                    )
                )
                backend._xqa_spec_dec_mask = torch.ones(1, 4, 2, dtype=torch.uint16)
                backend.forward_metadata = SimpleNamespace(
                    is_ragged_verify=False,
                    max_seq_len_q=4,
                    cache_seqlens_int32=torch.tensor([4], dtype=torch.int32),
                )
                backend._run_fixed_q_len_decode = Mock(side_effect=lambda q, *a, **k: q)
                layer = SimpleNamespace(
                    layer_id=0,
                    tp_k_head_num=2,
                    tp_v_head_num=2,
                    tp_q_head_num=2,
                    head_dim=256,
                    scaling=0.0625,
                    sliding_window_size=-1,
                    attn_type="decoder",
                )
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    out_cache_loc=None,
                    _attn_output=None,
                )
                query = torch.zeros(4, 2, 256, dtype=torch.bfloat16)
                with patch.object(backend_module, "is_cp_active", return_value=False):
                    output = backend.forward_extend(query, None, None, layer, batch)
                args, kwargs = backend._run_fixed_q_len_decode.call_args
                self.assertEqual(args[0].dtype, torch.bfloat16)
                self.assertEqual(args[1][0].shape, (pages, 2, 128, 128))
                self.assertEqual(kwargs["kv_cache_sf"][0].shape, (pages, 2, 128, 16))
                self.assertEqual(kwargs["bmm1_scale"], 0.125)
                self.assertEqual(kwargs["bmm2_scale"], 3.0)
                self.assertEqual(kwargs["q_len_per_req"], 4)
                self.assertIs(kwargs["mask"], backend._xqa_spec_dec_mask)
                self.assertIsNone(kwargs["out"])
                self.assertEqual(kwargs["out_dtype"], torch.bfloat16)
                self.assertEqual(output.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
