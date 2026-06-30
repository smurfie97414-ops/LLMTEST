import json
import tempfile
import unittest

import torch

from cortex3 import (
    ArithmeticSkill,
    CandidateAnswer,
    CorruptedCompressedAgent,
    DynamicSkillVerifier,
    InstructionSkill,
    LongContextAnchorSkill,
    ReferenceRuleAgent,
    Task,
    default_skill_specs,
)
from cortex3_attribution import AblationDimension, CausalAttributionEngine, cluster_regressions
from cortex3_cycle import CortexCycle
from cortex3_future import FutureContractEngine, MTPFSPConfig, MTPFSPHeads
from cortex3_inference import InferencePath, UltraFastInferenceEngine
from cortex3_ledgers import CausalTrace
from cortex3_reporting import write_cycle_run
from cortex3_ternary import CompressionTraceLedger, make_compression_decision


def _verifier() -> DynamicSkillVerifier:
    return DynamicSkillVerifier(default_skill_specs())


class CausalAttributionTest(unittest.TestCase):
    def test_activation_precision_probe_recovers_arithmetic_failure(self):
        skill = ArithmeticSkill()
        task = Task("arith-fixed", "arithmetic", "Compute exactly: 20 + 22.", 42)
        failure = skill.verify(task, CandidateAnswer("43", confidence=0.82))
        trace = CausalTrace(task_id=task.task_id, skill=task.skill, mtp_horizon=4, activation_bits=4, kv_mode="exact", uncertainty=0.18)
        compression = CompressionTraceLedger()
        _, decision = make_compression_decision("numeric-block", [0.0, 0.01, 2.0, -3.0], certify_zeros=True)
        compression.record_compression(decision)

        report = CausalAttributionEngine(_verifier()).attribute(failure, trace=trace, compression_ledger=compression)
        self.assertEqual(report.top_cause, "block_overcompressed")
        self.assertTrue(report.targeted_repair_is_cheaper)
        self.assertTrue(any(probe.spec.dimension == AblationDimension.ACTIVATION_PRECISION and probe.recovered for probe in report.probes))
        self.assertTrue(any(probe.spec.dimension == AblationDimension.BLOCK and probe.recovered for probe in report.probes))

    def test_real_forward_trace_drives_layer_block_and_activation_probes(self):
        verifier = _verifier()
        task = Task("arith-forward-trace", "arithmetic", "Compute exactly: 20 + 22. Return only the integer.", 42, {"kind": "add", "a": 20, "b": 22})
        engine = UltraFastInferenceEngine(verifier, CorruptedCompressedAgent())
        result = engine.infer(task, forced_path=InferencePath.CAREFUL)
        assert result.verification is not None

        report = CausalAttributionEngine(verifier).attribute(result.verification, compression_ledger=engine.trace)
        layer_probes = [probe for probe in report.probes if (probe.spec.metadata or {}).get("source") == "layer_forward_event"]

        self.assertFalse(result.verification.passed)
        self.assertTrue(engine.trace.layer_forward_events)
        self.assertTrue(layer_probes)
        self.assertTrue(any(probe.spec.dimension == AblationDimension.BLOCK for probe in layer_probes))
        self.assertTrue(any(probe.spec.dimension == AblationDimension.ACTIVATION_PRECISION for probe in layer_probes))
        self.assertTrue(all(probe.spec.target.startswith("mod-layer-") for probe in layer_probes))
        self.assertTrue(any(probe.recovered for probe in layer_probes))
        self.assertLess(report.targeted_repair_cost, report.global_retrain_cost)

    def test_kv_mode_probe_recovers_anchor_failure(self):
        task = Task("anchor-fixed", "long_context_anchor", "Return exact code.", "C3-7777-Z")
        failure = LongContextAnchorSkill().verify(task, CandidateAnswer("C3-7777-A", confidence=0.76))
        trace = CausalTrace(task_id=task.task_id, skill=task.skill, mtp_horizon=2, activation_bits=8, kv_mode="latent", uncertainty=0.24)

        report = CausalAttributionEngine(_verifier()).attribute(failure, trace=trace)
        self.assertEqual(report.top_cause, "memory_or_anchor_loss")
        kv_probe = next(probe for probe in report.probes if probe.spec.dimension == AblationDimension.KV_MODE)
        self.assertTrue(kv_probe.recovered)
        self.assertEqual(kv_probe.counterfactual_answer, "C3-7777-Z")

    def test_fsp_contract_probe_recovers_instruction_format_failure(self):
        skill = InstructionSkill()
        task = Task("instr-fixed", "instruction_following", "Return exactly OK.", "OK")
        failure = skill.verify(task, CandidateAnswer("OK\nDone.", confidence=0.88))
        heads = MTPFSPHeads(MTPFSPConfig(hidden_size=4, vocab_size=5))
        with torch.no_grad():
            heads.confidence_head.weight.zero_()
            heads.confidence_head.bias.fill_(5.0)
        engine = FutureContractEngine(heads.config, heads=heads)
        contract = engine.draft_contract(torch.zeros(1, 4), domain="general", risk=0.05, contract_id="format-fsp")
        wrong_tokens = tuple((token + 1) % 5 for token in contract.token_ids[:1])
        engine.gate_contract(contract, observed_tokens=wrong_tokens)

        report = CausalAttributionEngine(_verifier()).attribute(failure, future_ledger=engine.ledger)
        self.assertTrue(any(probe.spec.dimension == AblationDimension.FSP_CONTRACT for probe in report.probes))
        self.assertTrue(any(probe.spec.dimension == AblationDimension.FSP_CONTRACT and probe.recovered for probe in report.probes))

    def test_batch_clustering_groups_regressions_by_top_cause_and_skill(self):
        arithmetic = ArithmeticSkill()
        anchor = LongContextAnchorSkill()
        failures = [
            arithmetic.verify(Task("arith-a", "arithmetic", "", 10), CandidateAnswer("11")),
            arithmetic.verify(Task("arith-b", "arithmetic", "", 20), CandidateAnswer("19")),
            anchor.verify(Task("anchor-a", "long_context_anchor", "", "C3-1111-A"), CandidateAnswer("C3-1111-B")),
        ]
        engine = CausalAttributionEngine(_verifier())
        reports = [engine.attribute(failure) for failure in failures]
        clusters = cluster_regressions(reports)
        self.assertTrue(any(cluster.cause == "numeric_precision" and cluster.skill == "arithmetic" and cluster.count == 2 for cluster in clusters))
        self.assertTrue(any(cluster.cause == "memory_or_anchor_loss" and cluster.skill == "long_context_anchor" for cluster in clusters))

    def test_cycle_run_artifacts_can_include_causal_attribution(self):
        verifier = _verifier()
        cycle = CortexCycle(verifier)
        report = cycle.run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=7, n_per_skill=1)
        attribution = CausalAttributionEngine(verifier).batch_attribute(report.regressions[:3])
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_cycle_run(report, output_dir=tmp, run_id="attr-run", attribution=attribution)
            payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
            self.assertIsNotNone(payload["causal_attribution"])
            self.assertGreater(len(payload["causal_attribution"]["reports"]), 0)
            self.assertGreater(len(payload["causal_attribution"]["clusters"]), 0)


if __name__ == "__main__":
    unittest.main()
