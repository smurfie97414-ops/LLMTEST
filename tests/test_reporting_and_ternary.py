import json
import tempfile
import unittest
from pathlib import Path

from cortex3 import CorruptedCompressedAgent, DynamicSkillVerifier, ReferenceRuleAgent, RegressionHarness, default_skill_specs
from cortex3_cycle import CortexCycle
from cortex3_future import FutureContractEngine, MTPFSPConfig, MTPFSPHeads
from cortex3_microtrain import CortexMicroModel, MicroDataset, MicroModelConfig, MicroVocabulary, examples_from_tasks
from cortex3_reporting import write_cycle_run
from cortex3_ternary import (
    BitLinear,
    BitLinearConfig,
    CompressionTraceLedger,
    ResidualSynapseBuffer,
    clear_native_ternary_autotune_cache,
    last_native_grad_input_kernel,
    last_native_grad_weight_kernel,
    load_native_ternary_autotune_cache,
    make_compression_decision,
    native_grad_input_kernel_counts,
    native_grad_weight_kernel_counts,
    native_ternary_cuda_available,
    native_ternary_cuda_extension_available,
    native_ternary_autotune_cache_snapshot,
    quantize_activation_values,
    save_native_ternary_autotune_cache,
    torch_available,
)
from tools.benchmark_ternary_kernel import _case_strict_extension_only, _matrix_summary, _parse_shape


class ReportingAndTernaryTest(unittest.TestCase):
    def test_ternary_benchmark_matrix_summary_requires_strict_extension_and_samples(self):
        self.assertEqual(_parse_shape("64x128x256"), (64, 128, 256))
        case = {
            "native_backend": "native_int2_extension_cuda_warp_reduction_int2",
            "native_grad_weight_backend_counts": {"extension": 3},
            "strict_extension_only": True,
            "full_forward_backward_speedup_vs_legacy_dense_ste": 1.5,
            "full_bitlinear_forward_ms": 0.1,
            "full_bitlinear_forward_backward_ms": 0.7,
            "resource_usage": {"sample_count": 4},
            "resource_metrics": {
                "gpu_utilization_percent": {"avg": 42.0},
                "gpu_power_draw_watts": {"avg": 51.0},
                "process_cpu_percent_of_total": {"avg": 18.0},
            },
        }
        self.assertTrue(_case_strict_extension_only(case))
        passed = _matrix_summary([case], min_resource_samples=2)
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["min_resource_sample_count"], 4)
        self.assertEqual(passed["avg_gpu_power_draw_watts"], 51.0)
        failed = _matrix_summary([case], min_resource_samples=5)
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["resource_samples_passed"])

    def test_cycle_run_artifacts_are_persisted(self):
        verifier = DynamicSkillVerifier(default_skill_specs())
        report = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=3, n_per_skill=1)
        faults = RegressionHarness(verifier).run_fault_matrix(seed=3, n_per_skill=1)
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_cycle_run(report, output_dir=tmp, run_id="unit-run", fault_results=faults)
            self.assertTrue(artifacts.summary_json.exists())
            self.assertTrue(artifacts.report_markdown.exists())
            self.assertIsNotNone(artifacts.fault_matrix_json)
            self.assertTrue(artifacts.fault_matrix_json.exists())
            summary = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
            fault_matrix = json.loads(artifacts.fault_matrix_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["run_id"], "unit-run")
            self.assertIn("trial", summary)
            self.assertTrue(fault_matrix["all_detected"])
            self.assertGreater(len(fault_matrix["results"]), 0)

    def test_cycle_run_artifacts_can_include_future_contract_ledger(self):
        import torch

        verifier = DynamicSkillVerifier(default_skill_specs())
        report = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=4, n_per_skill=1)
        heads = MTPFSPHeads(MTPFSPConfig(hidden_size=4, vocab_size=5))
        with torch.no_grad():
            heads.confidence_head.weight.zero_()
            heads.confidence_head.bias.fill_(5.0)
        engine = FutureContractEngine(heads.config, heads=heads)
        contract = engine.draft_contract(torch.zeros(1, 4), domain="general", risk=0.05, contract_id="persisted-fsp")
        engine.gate_contract(contract)
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_cycle_run(report, output_dir=tmp, run_id="future-run", future_ledger=engine.ledger)
            summary = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
            self.assertIsNotNone(summary["future_contracts"])
            self.assertEqual(summary["future_contracts"]["accepted"], 1)

    def test_activation_quantization_and_residual_buffer(self):
        quantized = quantize_activation_values([-1.0, -0.25, 0.0, 0.5, 1.0], bits=4)
        self.assertEqual(quantized.bits, 4)
        self.assertEqual(len(quantized.quantized), 5)
        self.assertLessEqual(max(abs(value) for value in quantized.quantized), 7)
        buffer = ResidualSynapseBuffer(residual_threshold=0.0)
        block, decision = make_compression_decision("block-a", [1.0, -0.2, 0.01, 2.0], residual_buffer=buffer, certify_zeros=True)
        restored = buffer.restore("block-a", block)
        self.assertEqual(len(restored), 4)
        self.assertGreaterEqual(decision.certified_zero_count, 1)
        self.assertGreater(decision.estimated_bits, 0)

    def test_compression_trace_ledger_records_plan_phase2_logs(self):
        ledger = CompressionTraceLedger()
        quantized = quantize_activation_values([0.0, 1.0, 3.0], bits=4)
        _, decision = make_compression_decision("block-b", [0.0, 0.01, 2.0, -3.0], certify_zeros=True)
        ledger.record_activation(quantized)
        ledger.record_compression(decision)
        ledger.record_expert("math", "numeric precision probe")
        ledger.record_kv("ctx-1", "latent", 12.0, exact_anchors=0)
        ledger.record_mtp_fsp("mtp-1", horizon=4, accepted=True, confidence=0.91, reason="fast path")
        self.assertGreater(ledger.cost_trace.effective_cost(), 0)
        hints = ledger.explain_failure("expert misroute and exact anchor loss")
        self.assertIn("kv_mode_may_have_lost_exact_anchors", hints)
        self.assertIn("accepted_mtp_horizon_may_have_overshot", hints)

    def test_compression_trace_ledger_retains_tail_with_total_counters(self):
        ledger = CompressionTraceLedger(retention_limit=2)
        for index in range(5):
            ledger.record_activation(quantize_activation_values([float(index), float(index + 1)], bits=4))

        payload = ledger.to_dict()
        self.assertEqual(len(ledger.activation_quantizations), 2)
        self.assertEqual(payload["retained_event_counts"]["activation_quantizations"], 2)
        self.assertEqual(payload["total_event_counts"]["activation_quantizations"], 5)
        self.assertEqual(ledger.cost_trace.activation_bits, 40.0)

    def test_bitlinear_dependency_boundary(self):
        import torch

        self.assertTrue(torch_available())
        self.assertEqual(BitLinearConfig(3, 2).native_cuda_backend, "extension")
        ledger = CompressionTraceLedger()
        layer = BitLinear(BitLinearConfig(3, 2, activation_bits=4), ledger=ledger)
        output = layer(torch.ones(1, 3))
        self.assertEqual(tuple(output.shape), (1, 2))
        self.assertTrue(ledger.compression_decisions)
        self.assertTrue(ledger.activation_quantizations)

    def test_bitlinear_preserves_gradient_path_through_residual_weight(self):
        import torch

        torch.manual_seed(0)
        layer = BitLinear(BitLinearConfig(3, 2, activation_bits=4))
        x = torch.randn(5, 3, requires_grad=True)
        loss = layer(x).pow(2).mean()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.float_weight.grad)
        self.assertGreater(float(x.grad.abs().sum()), 0.0)
        self.assertGreater(float(layer.float_weight.grad.abs().sum()), 0.0)
        self.assertGreater(layer.ledger.total_packed_ternary_dispatches, 0)
        self.assertEqual(layer.ledger.packed_ternary_dispatches[-1].backend, "packed_int2_torch")

    def test_bitlinear_forward_value_uses_packed_ternary_weight_with_ste_gradient(self):
        import torch
        import torch.nn.functional as F

        torch.manual_seed(5)
        layer = BitLinear(BitLinearConfig(4, 3, activation_bits=0, residual_runtime=False))
        x = torch.randn(7, 4, requires_grad=True)

        actual = layer(x)
        packed_weight = layer._packed_runtime_weight(dtype=x.dtype, device=x.device)
        expected = F.linear(x, packed_weight, layer.bias)
        float_output = F.linear(x, layer.float_weight, layer.bias)
        actual.pow(2).mean().backward()

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertFalse(torch.allclose(actual, float_output, atol=1e-6))
        self.assertIsNotNone(layer.float_weight.grad)
        self.assertGreater(float(layer.float_weight.grad.abs().sum()), 0.0)
        self.assertEqual(layer.ledger.packed_ternary_dispatches[-1].packed_weight_bytes, layer.packed_codes.numel())

    def test_bitlinear_fast_ste_autograd_matches_dense_ste_gradients(self):
        import torch

        torch.manual_seed(19)
        fast = BitLinear(BitLinearConfig(5, 4, activation_bits=0, residual_runtime=False, use_fast_ste_autograd=True))
        dense = BitLinear(BitLinearConfig(5, 4, activation_bits=0, residual_runtime=False, use_fast_ste_autograd=False))
        with torch.no_grad():
            dense.float_weight.copy_(fast.float_weight)
            if fast.bias is not None and dense.bias is not None:
                dense.bias.copy_(fast.bias)
        fast.requantize()
        dense.requantize()
        x_fast = torch.randn(6, 5, requires_grad=True)
        x_dense = x_fast.detach().clone().requires_grad_(True)

        fast_loss = fast(x_fast).square().mean()
        dense_loss = dense(x_dense).square().mean()
        fast_loss.backward()
        dense_loss.backward()

        self.assertTrue(torch.allclose(fast_loss.detach(), dense_loss.detach(), atol=1e-7))
        self.assertTrue(torch.allclose(x_fast.grad, x_dense.grad, atol=1e-6))
        self.assertTrue(torch.allclose(fast.float_weight.grad, dense.float_weight.grad, atol=1e-6))
        self.assertIsNotNone(fast.bias.grad)
        self.assertIsNotNone(dense.bias.grad)
        self.assertTrue(torch.allclose(fast.bias.grad, dense.bias.grad, atol=1e-6))
        self.assertIn("custom autograd STE backward", fast.ledger.packed_ternary_dispatches[-1].note)

    def test_bitlinear_packed_runtime_syncs_after_weight_update_without_manual_requantize(self):
        import torch
        import torch.nn.functional as F

        torch.manual_seed(17)
        layer = BitLinear(BitLinearConfig(4, 3, activation_bits=0, residual_runtime=False))
        x = torch.randn(5, 4)

        with torch.no_grad():
            layer.float_weight.copy_(torch.tensor([
                [2.0, -2.0, 0.01, -0.01],
                [-3.0, 3.0, 0.02, -0.02],
                [4.0, -4.0, 0.03, -0.03],
            ]))

        actual = layer(x)
        current_ste_weight = layer._runtime_weight_ste()
        expected = F.linear(x, current_ste_weight, layer.bias)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertGreater(layer.ledger.total_packed_ternary_dispatches, 0)

    def test_bitlinear_skips_repack_when_weight_version_is_unchanged(self):
        import torch

        torch.manual_seed(29)
        layer = BitLinear(BitLinearConfig(4, 3, activation_bits=0, residual_runtime=False))
        original_sync = layer._sync_quantized_buffers_from_weight
        sync_calls = 0

        def counted_sync(*args, **kwargs):
            nonlocal sync_calls
            sync_calls += 1
            return original_sync(*args, **kwargs)

        layer._sync_quantized_buffers_from_weight = counted_sync
        x = torch.randn(5, 4)
        layer(x)
        layer(x)
        self.assertEqual(sync_calls, 0)

        with torch.no_grad():
            layer.float_weight.add_(0.125)
        layer(x)
        self.assertEqual(sync_calls, 1)

    @unittest.skipUnless(
        __import__("torch").cuda.is_available() and native_ternary_cuda_available(),
        "CUDA plus CuPy native ternary kernel is required",
    )
    def test_bitlinear_native_packed_ternary_cuda_dispatch_runs_on_gpu(self):
        import torch

        torch.manual_seed(7)
        layer = BitLinear(BitLinearConfig(
            8,
            4,
            activation_bits=4,
            log_prefix="cuda-packed",
            require_native_cuda_kernel=True,
        )).cuda()
        x = torch.randn(5, 8, device="cuda", requires_grad=True)

        output = layer(x)
        torch.cuda.synchronize()
        output.pow(2).mean().backward()
        torch.cuda.synchronize()

        self.assertEqual(tuple(output.shape), (5, 4))
        self.assertIsNotNone(layer.float_weight.grad)
        self.assertGreater(float(layer.float_weight.grad.abs().sum().detach().cpu()), 0.0)
        self.assertGreater(layer.ledger.total_packed_ternary_dispatches, 0)
        self.assertGreater(layer.ledger.total_native_ternary_kernel_dispatches, 0)
        dispatch = layer.ledger.packed_ternary_dispatches[-1]
        self.assertTrue(dispatch.backend.startswith("native_int2_"))
        self.assertTrue(dispatch.native_kernel)
        self.assertIn(dispatch.native_backend, {"extension", "rawkernel"})
        self.assertTrue(dispatch.autotuned)
        self.assertEqual({name for name, _ in dispatch.autotune_candidate_ms}, {"tiled", "warp"})
        selected = min(dispatch.autotune_candidate_ms, key=lambda item: item[1])[0]
        expected_variant = "tiled_shared_memory_int2" if selected == "tiled" else "warp_reduction_int2"
        self.assertEqual(dispatch.kernel_variant, expected_variant)

    def test_bitlinear_native_extension_cuda_dispatch_runs_on_gpu(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        if not native_ternary_cuda_extension_available():
            self.skipTest("Cortex ternary CUDA extension is not buildable in this environment")

        torch.manual_seed(41)
        layer = BitLinear(BitLinearConfig(
            16,
            9,
            activation_bits=0,
            residual_runtime=True,
            require_native_cuda_kernel=True,
            native_cuda_backend="extension",
            native_cuda_kernel_variant="warp",
            native_cuda_autotune=False,
            log_prefix="cuda-extension-packed",
        )).cuda()
        layer.requantize()
        x = torch.randn(6, 16, device="cuda", dtype=torch.float16, requires_grad=True)
        output = layer(x)
        loss = output.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()

        self.assertEqual(tuple(output.shape), (6, 9))
        self.assertIsNotNone(x.grad)
        self.assertGreater(float(x.grad.float().abs().sum().detach().cpu()), 0.0)
        self.assertIsNotNone(layer.float_weight.grad)
        self.assertGreater(float(layer.float_weight.grad.abs().sum().detach().cpu()), 0.0)
        dispatch = layer.ledger.packed_ternary_dispatches[-1]
        self.assertTrue(dispatch.native_kernel)
        self.assertEqual(dispatch.native_backend, "extension")
        self.assertTrue(dispatch.backend.startswith("native_int2_extension_cuda_"))
        self.assertIn("native extension", dispatch.note)
        self.assertEqual(layer._last_requantize_backend, "native_cuda_extension_requantize_pack")
        payload = layer.ledger.to_dict()
        self.assertGreater(payload["native_ternary_backend_counts"].get("extension", 0), 0)
        self.assertGreater(payload["native_ternary_requantize_backend_counts"].get("extension", 0), 0)
        self.assertGreater(payload["native_ternary_grad_weight_backend_counts"].get("extension", 0), 0)
        self.assertGreater(payload["total_event_counts"].get("native_ternary_extension_kernel_dispatches", 0), 0)
        self.assertGreater(payload["total_event_counts"].get("native_ternary_extension_requantize_dispatches", 0), 0)
        self.assertGreater(payload["total_event_counts"].get("native_ternary_extension_grad_weight_dispatches", 0), 0)

    def test_bitlinear_native_extension_cuda_grad_weight_uses_wmma_when_aligned(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        if not native_ternary_cuda_extension_available():
            self.skipTest("Cortex ternary CUDA extension is not buildable in this environment")

        torch.manual_seed(43)
        layer = BitLinear(BitLinearConfig(
            32,
            32,
            activation_bits=0,
            residual_runtime=False,
            require_native_cuda_kernel=True,
            native_cuda_backend="extension",
            native_cuda_kernel_variant="warp",
            native_cuda_autotune=False,
            log_prefix="cuda-extension-wmma-grad-weight",
        )).cuda()
        layer.requantize()
        x = torch.randn(32, 32, device="cuda", dtype=torch.float16, requires_grad=True)
        loss = layer(x).float().square().mean()
        loss.backward()
        torch.cuda.synchronize()

        self.assertIsNotNone(layer.float_weight.grad)
        self.assertGreater(float(layer.float_weight.grad.abs().sum().detach().cpu()), 0.0)
        self.assertEqual(last_native_grad_input_kernel(), "wmma_fp16")
        self.assertEqual(last_native_grad_weight_kernel(), "wmma_fp16_float")
        self.assertGreater(native_grad_input_kernel_counts().get("wmma_fp16", 0), 0)
        self.assertGreater(native_grad_weight_kernel_counts().get("wmma_fp16_float", 0), 0)
        payload = layer.ledger.to_dict()
        self.assertGreater(payload["native_ternary_grad_weight_backend_counts"].get("extension", 0), 0)

    @unittest.skipUnless(
        __import__("torch").cuda.is_available() and native_ternary_cuda_available(),
        "CUDA plus CuPy native ternary kernel is required",
    )
    def test_native_ternary_cuda_fast_ste_backward_matches_dense_ste(self):
        import torch

        torch.manual_seed(31)
        for dtype, atol in ((torch.float32, 2e-5), (torch.float16, 4e-2), (torch.bfloat16, 5e-2)):
            for residual_runtime in (False, True):
                with self.subTest(dtype=str(dtype), residual_runtime=residual_runtime):
                    fast = BitLinear(BitLinearConfig(
                        19,
                        13,
                        activation_bits=0,
                        residual_runtime=residual_runtime,
                        require_native_cuda_kernel=True,
                        native_cuda_kernel_variant="warp",
                        native_cuda_autotune=False,
                        use_fast_ste_autograd=True,
                        log_prefix=f"cuda-fast-ste-{dtype}",
                    )).cuda()
                    dense = BitLinear(BitLinearConfig(
                        19,
                        13,
                        activation_bits=0,
                        residual_runtime=residual_runtime,
                        require_native_cuda_kernel=True,
                        native_cuda_kernel_variant="warp",
                        native_cuda_autotune=False,
                        use_fast_ste_autograd=False,
                        log_prefix=f"cuda-dense-ste-{dtype}",
                    )).cuda()
                    with torch.no_grad():
                        dense.float_weight.copy_(fast.float_weight)
                        if fast.bias is not None and dense.bias is not None:
                            dense.bias.copy_(fast.bias)
                    fast.requantize()
                    dense.requantize()
                    x_fast = torch.randn(7, 19, device="cuda", dtype=dtype, requires_grad=True)
                    x_dense = x_fast.detach().clone().requires_grad_(True)

                    fast_loss = fast(x_fast).float().square().mean()
                    dense_loss = dense(x_dense).float().square().mean()
                    fast_loss.backward()
                    dense_loss.backward()
                    torch.cuda.synchronize()

                    self.assertTrue(torch.allclose(fast_loss.detach(), dense_loss.detach(), atol=atol, rtol=atol))
                    self.assertTrue(torch.allclose(x_fast.grad.float(), x_dense.grad.float(), atol=atol, rtol=atol))
                    self.assertTrue(torch.allclose(fast.float_weight.grad, dense.float_weight.grad, atol=atol, rtol=atol))
                    self.assertIsNotNone(fast.bias.grad)
                    self.assertIsNotNone(dense.bias.grad)
                    self.assertTrue(torch.allclose(fast.bias.grad, dense.bias.grad, atol=atol, rtol=atol))
                    dispatch = fast.ledger.packed_ternary_dispatches[-1]
                    self.assertTrue(dispatch.native_kernel)
                    self.assertIn("custom autograd STE backward", dispatch.note)
                    self.assertGreater(fast.ledger.total_native_ternary_grad_weight_dispatches, 0)

    @unittest.skipUnless(
        __import__("torch").cuda.is_available() and native_ternary_cuda_available(),
        "CUDA plus CuPy native ternary kernel is required",
    )
    def test_native_ternary_cuda_kernel_matches_packed_runtime_for_training_dtypes(self):
        import torch
        import torch.nn.functional as F

        torch.manual_seed(23)
        for dtype, atol in ((torch.float32, 1e-5), (torch.float16, 2e-2), (torch.bfloat16, 3e-2)):
            with self.subTest(dtype=str(dtype)):
                layer = BitLinear(BitLinearConfig(
                    17,
                    11,
                    activation_bits=0,
                    residual_runtime=False,
                    require_native_cuda_kernel=True,
                    log_prefix=f"native-{dtype}",
                )).cuda()
                x = torch.randn(5, 17, device="cuda", dtype=dtype)
                actual = layer._native_cuda_packed_output(x)
                expected = F.linear(
                    x.float(),
                    layer._packed_runtime_weight(dtype=torch.float32, device=x.device),
                    layer.bias.float(),
                )
                torch.cuda.synchronize()

                self.assertTrue(torch.allclose(actual.float(), expected, atol=atol, rtol=atol))

    @unittest.skipUnless(
        __import__("torch").cuda.is_available() and native_ternary_cuda_available(),
        "CUDA plus CuPy native ternary kernel is required",
    )
    def test_native_ternary_cuda_requantize_pack_matches_torch_sync(self):
        import torch

        torch.manual_seed(37)
        dtype_cases = ((torch.float32, 1e-6), (torch.float16, 2e-3), (torch.bfloat16, 2e-2))
        config_cases = ((None, 0.0), (0.05, 0.01))
        for dtype, atol in dtype_cases:
            for threshold, residual_threshold in config_cases:
                with self.subTest(dtype=str(dtype), threshold=threshold, residual_threshold=residual_threshold):
                    native = BitLinear(BitLinearConfig(
                        17,
                        11,
                        activation_bits=0,
                        threshold=threshold,
                        residual_threshold=residual_threshold,
                        residual_runtime=True,
                        require_native_cuda_kernel=True,
                        native_cuda_autotune=False,
                    )).to(device="cuda", dtype=dtype)
                    reference = BitLinear(BitLinearConfig(
                        17,
                        11,
                        activation_bits=0,
                        threshold=threshold,
                        residual_threshold=residual_threshold,
                        residual_runtime=True,
                        use_native_cuda_kernel=False,
                        native_cuda_autotune=False,
                    )).to(device="cuda", dtype=dtype)
                    weight = torch.randn(11, 17, device="cuda", dtype=dtype) * 0.25
                    with torch.no_grad():
                        native.float_weight.copy_(weight)
                        reference.float_weight.copy_(weight)

                    native.requantize()
                    reference.requantize()
                    torch.cuda.synchronize()

                    self.assertTrue(native._last_requantize_backend.startswith("native_cuda_"))
                    self.assertTrue(native._last_requantize_backend.endswith("_requantize_pack"))
                    self.assertTrue(native.ledger.native_ternary_requantize_backend_counts)
                    self.assertEqual(reference._last_requantize_backend, "torch_tensor_requantize")
                    self.assertTrue(torch.allclose(native.signs.float(), reference.signs.float(), atol=0.0, rtol=0.0))
                    self.assertTrue(torch.allclose(native.mask.float(), reference.mask.float(), atol=0.0, rtol=0.0))
                    self.assertTrue(torch.allclose(native.scales.float(), reference.scales.float(), atol=atol, rtol=atol))
                    self.assertTrue(torch.allclose(native.residual_weight.float(), reference.residual_weight.float(), atol=atol, rtol=atol))
                    self.assertTrue(torch.equal(native.packed_codes, reference.packed_codes))
                    self.assertEqual(native.ledger.compression_decisions[-1].active_count, reference.ledger.compression_decisions[-1].active_count)
                    self.assertEqual(native.ledger.compression_decisions[-1].zero_count, reference.ledger.compression_decisions[-1].zero_count)

    @unittest.skipUnless(
        __import__("torch").cuda.is_available() and native_ternary_cuda_available(),
        "CUDA plus CuPy native ternary kernel is required",
    )
    def test_native_ternary_auto_kernel_autotunes_and_reuses_cache(self):
        import torch

        clear_native_ternary_autotune_cache()
        layer = BitLinear(BitLinearConfig(
            96,
            80,
            activation_bits=0,
            require_native_cuda_kernel=True,
            native_cuda_kernel_variant="auto",
            native_cuda_autotune_warmup=0,
            native_cuda_autotune_repeat=1,
        )).cuda()
        x = torch.randn(7, 96, device="cuda", dtype=torch.float16)

        layer._native_cuda_packed_output(x)
        first_candidates = tuple(layer._last_native_autotune_candidate_ms)
        self.assertTrue(layer._last_native_autotuned)
        self.assertFalse(layer._last_native_autotune_cache_hit)
        self.assertEqual({name for name, _ in first_candidates}, {"tiled", "warp"})
        selected = min(first_candidates, key=lambda item: item[1])[0]
        self.assertEqual(layer._last_native_kernel_family, selected)

        layer._native_cuda_packed_output(x)
        self.assertTrue(layer._last_native_autotune_cache_hit)
        self.assertEqual(tuple(layer._last_native_autotune_candidate_ms), first_candidates)
        self.assertEqual(layer._last_native_kernel_family, selected)

    @unittest.skipUnless(
        __import__("torch").cuda.is_available() and native_ternary_cuda_available(),
        "CUDA plus CuPy native ternary kernel is required",
    )
    def test_native_ternary_autotune_cache_can_persist_and_reload(self):
        import torch

        clear_native_ternary_autotune_cache()
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "autotune-profile.json"
            layer = BitLinear(BitLinearConfig(
                64,
                48,
                activation_bits=0,
                require_native_cuda_kernel=True,
                native_cuda_kernel_variant="auto",
                native_cuda_autotune_warmup=0,
                native_cuda_autotune_repeat=1,
                native_cuda_autotune_cache_path=str(profile),
            )).cuda()
            x = torch.randn(6, 64, device="cuda", dtype=torch.float16)
            layer._native_cuda_packed_output(x)
            first_family = layer._last_native_kernel_family
            self.assertTrue(profile.exists())
            self.assertGreater(native_ternary_autotune_cache_snapshot()["entry_count"], 0)

            save_native_ternary_autotune_cache(profile)
            clear_native_ternary_autotune_cache()
            self.assertEqual(native_ternary_autotune_cache_snapshot()["entry_count"], 0)
            self.assertGreater(load_native_ternary_autotune_cache(profile), 0)

            reloaded = BitLinear(BitLinearConfig(
                64,
                48,
                activation_bits=0,
                require_native_cuda_kernel=True,
                native_cuda_kernel_variant="auto",
                native_cuda_autotune_warmup=0,
                native_cuda_autotune_repeat=1,
                native_cuda_autotune_cache_path=str(profile),
            )).cuda()
            reloaded._native_cuda_packed_output(x)
            self.assertTrue(reloaded._last_native_autotune_cache_hit)
            self.assertEqual(reloaded._last_native_kernel_family, first_family)

    def test_bitlinear_matches_float_linear_when_activation_quantization_is_disabled(self):
        import torch
        import torch.nn as nn

        torch.manual_seed(12)
        linear = nn.Linear(4, 3)
        layer = BitLinear.from_linear(linear, activation_bits=0)
        x = torch.randn(6, 4)

        expected = linear(x)
        actual = layer(x)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertTrue(layer.ledger.layer_forward_events)
        event = layer.ledger.layer_forward_events[-1]
        self.assertEqual(event.input_shape, (6, 4))
        self.assertEqual(event.output_shape, (6, 3))
        self.assertEqual(event.activation_bits, 0.0)

    def test_micro_model_forward_emits_layer_forward_trace(self):
        verifier = DynamicSkillVerifier(default_skill_specs())
        tasks = verifier.build_suite(1, seed=6, include_metamorphic=False)[:4]
        examples = examples_from_tasks(tasks)
        vocabulary = MicroVocabulary.from_examples(examples)
        config = MicroModelConfig()
        model = CortexMicroModel(config, vocabulary)
        dataset = MicroDataset(examples, vocabulary, config)
        features, _, _, _ = dataset.tensors()

        output = model(features)

        self.assertEqual(tuple(output["answer_logits"].shape), (len(examples), len(vocabulary.answers)))
        self.assertTrue(model.ledger.layer_forward_events)
        event = model.ledger.layer_forward_events[-1]
        self.assertEqual(event.layer_id, "micro-compiled-core")
        self.assertEqual(event.input_shape, (len(examples), config.hidden_size))
        self.assertEqual(event.output_shape, (len(examples), config.hidden_size))
        self.assertGreater(event.active_weights, 0)
        self.assertLessEqual(event.active_weights, event.total_weights)


if __name__ == "__main__":
    unittest.main()
