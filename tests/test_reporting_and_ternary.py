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
    make_compression_decision,
    quantize_activation_values,
    torch_available,
)


class ReportingAndTernaryTest(unittest.TestCase):
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
