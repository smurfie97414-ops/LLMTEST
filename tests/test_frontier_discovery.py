import json
import tempfile
import unittest

from cortex3 import CorruptedCompressedAgent, DynamicSkillVerifier, ReferenceRuleAgent, default_skill_specs
from cortex3_cycle import CortexCycle
from cortex3_frontier import CompiledFrontierAgent, FrontierCircuitRegistry, FrontierSkillDiscovery
from cortex3_inference import InferencePath, UltraFastInferenceEngine
from cortex3_reporting import write_cycle_run


class FrontierSkillDiscoveryTest(unittest.TestCase):
    def test_frontier_discovery_slow_solves_distills_and_compiles_fragile_skill(self):
        verifier = DynamicSkillVerifier(default_skill_specs())
        cycle = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=3, n_per_skill=1)
        registry = FrontierCircuitRegistry()

        report = FrontierSkillDiscovery(verifier).discover(cycle, seed=3, max_skills=1, epochs=120, registry=registry)

        self.assertTrue(report.passed)
        self.assertEqual(len(report.circuits), 1)
        circuit = report.circuits[0]
        self.assertIn(circuit.skill, {state.skill for state in cycle.skill_ledger.fragile_skills()})
        self.assertGreater(circuit.verified_slow_solutions, 0)
        self.assertEqual(circuit.dsv["passed"], circuit.dsv["total"])
        self.assertGreater(circuit.compiled_weight_bits, 0.0)
        self.assertGreater(circuit.active_weights, 0)
        self.assertGreater(circuit.training["after_accuracy"], circuit.training["before_accuracy"])
        self.assertTrue(circuit.invariants.prompt_obligations)
        self.assertTrue(set(circuit.source_failure_ids).issubset(set(circuit.frontier_task_ids)))
        self.assertEqual(registry.compiled_skills(), (circuit.skill,))
        runtime_circuit = registry.circuits_for_skill(circuit.skill)[0]
        task = runtime_circuit.verified_tasks[0]
        selected = registry.select(task)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.report.skill, task.skill)
        answer = CompiledFrontierAgent(registry, verifier=verifier)(task)
        self.assertTrue(answer.raw["frontier_compiled_selected"])
        self.assertTrue(answer.certificate["frontier_compiled_circuit"])
        self.assertTrue(answer.certificate["frontier_verification_passed"])
        self.assertTrue(answer.certificate["frontier_output_goal_contract_passed"])
        self.assertEqual(answer.certificate["frontier_output_goal_contract"]["contract"]["skill"], task.skill)
        self.assertTrue(answer.certificate["frontier_compiled_contract_verified"])
        self.assertEqual(answer.certificate["frontier_compiled_contract"]["certificate_type"], "compiled_circuit")
        compiled_contract = answer.certificate["frontier_compiled_contract"]["claims"]["compiled_circuit_contract"]
        self.assertTrue(compiled_contract["output_goal_contract_passed"])
        self.assertEqual(compiled_contract["output_goal_contract_id"], answer.certificate["frontier_output_goal_contract"]["contract"]["contract_id"])
        self.assertTrue(answer.certificate["frontier_compiled_contract_checksum"])
        self.assertTrue(verifier.oracle_registry.verify(task.skill, task, answer).passed)
        engine = UltraFastInferenceEngine(verifier, CorruptedCompressedAgent(), compiled_frontier_registry=registry)
        inferred = engine.infer(task, forced_path=InferencePath.FAST)
        self.assertTrue(inferred.answer.raw["frontier_compiled_selected"])
        self.assertTrue(inferred.answer.certificate["frontier_compiled_circuit"])
        self.assertTrue(inferred.answer.certificate["output_goal_contract_passed"])
        self.assertTrue(inferred.future_contract["output_goal_contract"]["accepted"])
        self.assertTrue(inferred.passed)

    def test_frontier_discovery_report_is_persisted(self):
        verifier = DynamicSkillVerifier(default_skill_specs())
        cycle = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=4, n_per_skill=1)
        registry = FrontierCircuitRegistry()
        frontier = FrontierSkillDiscovery(verifier).discover(cycle, seed=4, max_skills=1, epochs=120, registry=registry)

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_cycle_run(cycle, output_dir=tmp, run_id="frontier-run", frontier_report=frontier)
            payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
            registry_path = registry.save(tmp)
            loaded_registry = FrontierCircuitRegistry.load(tmp)
            runtime_circuit = loaded_registry.circuits_for_skill(frontier.circuits[0].skill)[0]
            task = runtime_circuit.verified_tasks[0]
            answer = CompiledFrontierAgent(loaded_registry, verifier=verifier)(task)

        self.assertTrue(payload["frontier_discovery"]["passed"])
        self.assertTrue(payload["frontier_discovery"]["circuits"])
        self.assertGreater(payload["frontier_discovery"]["circuits"][0]["compiled_weight_bits"], 0.0)
        self.assertEqual(registry_path.name, "frontier_registry.json")
        self.assertTrue(answer.raw["frontier_compiled_selected"])
        self.assertTrue(answer.certificate["frontier_output_goal_contract_passed"])
        self.assertTrue(verifier.oracle_registry.verify(task.skill, task, answer).passed)


if __name__ == "__main__":
    unittest.main()
