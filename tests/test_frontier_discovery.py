import json
import tempfile
import unittest
from dataclasses import replace

from cortex3 import CorruptedCompressedAgent, DynamicSkillVerifier, ReferenceRuleAgent, default_skill_specs
from cortex3_cycle import CortexCycle
from cortex3_frontier import CompiledFrontierAgent, FrontierCircuitRegistry, FrontierSkillDiscovery, compiled_circuit_id
from cortex3_inference import InferencePath, UltraFastInferenceEngine
from cortex3_memory import CognitiveMemory, CognitiveMemoryConfig
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
        self.assertTrue(circuit.heldout_task_ids)
        self.assertGreater(circuit.heldout["total"], 0)
        self.assertEqual(circuit.heldout["passed"], circuit.heldout["total"])
        self.assertTrue(circuit.heldout["gate_passed"])
        self.assertEqual(registry.compiled_skills(), (circuit.skill,))
        runtime_circuit = registry.circuits_for_skill(circuit.skill)[0]
        self.assertTrue(runtime_circuit.heldout_tasks)
        task = runtime_circuit.verified_tasks[0]
        selected = registry.select(task)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.report.skill, task.skill)
        circuit_id = compiled_circuit_id(runtime_circuit.report)

        class BrokenMemory:
            def bind_compiled_circuit(self, **kwargs):
                raise RuntimeError("binding blocked")

        with self.assertRaisesRegex(ValueError, "could not establish a P4 memory binding"):
            CompiledFrontierAgent(registry, verifier=verifier, memory=BrokenMemory())(task)

        strict_memory = CognitiveMemory(CognitiveMemoryConfig(recent_exact_limit=1, embedding_dim=64, top_k_latent=2))
        answer = CompiledFrontierAgent(registry, verifier=verifier, memory=strict_memory)(task)
        binding = strict_memory.compiled_circuit_bindings[circuit_id]
        strict_memory.ingest("recent-frontier-noise", "Segment recent pour verifier que le circuit compile reste reconstructible en latent KV.")
        answer = CompiledFrontierAgent(registry, verifier=verifier, memory=strict_memory)(task)
        self.assertTrue(answer.raw["frontier_compiled_selected"])
        self.assertTrue(answer.certificate["frontier_compiled_circuit"])
        self.assertEqual(answer.certificate["frontier_memory_binding_id"], binding.binding_id)
        self.assertTrue(answer.certificate["frontier_memory_binding_passed"])
        self.assertGreater(answer.certificate["frontier_memory_binding_fidelity"], 0.0)
        self.assertTrue(answer.certificate["frontier_verification_passed"])
        self.assertTrue(answer.certificate["frontier_heldout_gate_passed"])
        self.assertEqual(answer.certificate["frontier_heldout_passed"], answer.certificate["frontier_heldout_total"])
        self.assertTrue(answer.certificate["frontier_output_goal_contract_passed"])
        self.assertEqual(answer.certificate["frontier_output_goal_contract"]["contract"]["skill"], task.skill)
        self.assertTrue(answer.certificate["frontier_compiled_contract_verified"])
        self.assertEqual(answer.certificate["frontier_compiled_contract"]["certificate_type"], "compiled_circuit")
        compiled_contract = answer.certificate["frontier_compiled_contract"]["claims"]["compiled_circuit_contract"]
        self.assertTrue(compiled_contract["output_goal_contract_passed"])
        self.assertEqual(compiled_contract["memory_binding_id"], binding.binding_id)
        self.assertTrue(compiled_contract["memory_binding_passed"])
        self.assertTrue(compiled_contract["heldout_gate_passed"])
        self.assertEqual(compiled_contract["heldout_passed"], compiled_contract["heldout_total"])
        self.assertTrue(compiled_contract["heldout_task_ids"])
        self.assertEqual(compiled_contract["output_goal_contract_id"], answer.certificate["frontier_output_goal_contract"]["contract"]["contract_id"])
        self.assertTrue(answer.certificate["frontier_compiled_contract_checksum"])
        self.assertTrue(verifier.oracle_registry.verify(task.skill, task, answer).passed)
        engine = UltraFastInferenceEngine(verifier, CorruptedCompressedAgent(), memory=strict_memory, compiled_frontier_registry=registry)
        inferred = engine.infer(task, forced_path=InferencePath.FAST)
        self.assertTrue(inferred.answer.raw["frontier_compiled_selected"])
        self.assertTrue(inferred.answer.certificate["frontier_compiled_circuit"])
        self.assertTrue(inferred.answer.certificate["frontier_memory_binding_passed"])
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
            self.assertTrue(runtime_circuit.heldout_tasks)
            task = runtime_circuit.verified_tasks[0]
            answer = CompiledFrontierAgent(loaded_registry, verifier=verifier)(task)

        self.assertTrue(payload["frontier_discovery"]["passed"])
        self.assertTrue(payload["frontier_discovery"]["circuits"])
        self.assertGreater(payload["frontier_discovery"]["circuits"][0]["compiled_weight_bits"], 0.0)
        self.assertEqual(registry_path.name, "frontier_registry.json")
        self.assertTrue(answer.raw["frontier_compiled_selected"])
        self.assertTrue(answer.certificate["frontier_heldout_gate_passed"])
        self.assertTrue(answer.certificate["frontier_output_goal_contract_passed"])
        self.assertTrue(verifier.oracle_registry.verify(task.skill, task, answer).passed)

    def test_frontier_registry_prefers_exact_coverage_over_generic_high_score_circuit(self):
        verifier = DynamicSkillVerifier(default_skill_specs())
        cycle = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=7, n_per_skill=1)
        discovery_registry = FrontierCircuitRegistry()
        frontier = FrontierSkillDiscovery(verifier).discover(
            cycle,
            seed=7,
            max_skills=1,
            epochs=120,
            registry=discovery_registry,
        )
        self.assertTrue(frontier.passed)
        source_runtime = discovery_registry.circuits_for_skill(frontier.circuits[0].skill)[0]
        target = source_runtime.verified_tasks[0]
        generic_task = replace(target, task_id=f"{target.task_id}-generic-neighbor")

        generic_report = replace(
            source_runtime.report,
            frontier_task_ids=(generic_task.task_id,),
            heldout_task_ids=tuple(task.task_id for task in source_runtime.heldout_tasks),
            verified_slow_solutions=5000,
            dsv={
                **dict(source_runtime.report.dsv),
                "verified_capability_per_cost": 1_000_000.0,
                "passed": max(1, int(source_runtime.report.dsv.get("total", 1) or 1)),
                "total": max(1, int(source_runtime.report.dsv.get("total", 1) or 1)),
            },
            heldout={
                **dict(source_runtime.report.heldout),
                "pass_rate": 1.0,
                "aggregate_score": 1_000_000.0,
            },
            compiled_weight_bits=1.0,
        )
        exact_report = replace(
            source_runtime.report,
            frontier_task_ids=(target.task_id,),
            heldout_task_ids=tuple(task.task_id for task in source_runtime.heldout_tasks),
            verified_slow_solutions=1,
            dsv={
                **dict(source_runtime.report.dsv),
                "verified_capability_per_cost": 0.0001,
                "passed": max(1, int(source_runtime.report.dsv.get("total", 1) or 1)),
                "total": max(1, int(source_runtime.report.dsv.get("total", 1) or 1)),
            },
            heldout={
                **dict(source_runtime.report.heldout),
                "pass_rate": 1.0,
                "aggregate_score": 0.0,
            },
            compiled_weight_bits=1_000_000_000.0,
        )
        registry = FrontierCircuitRegistry()
        generic = registry.register(
            generic_report,
            source_runtime.model,
            (generic_task,),
            heldout_tasks=source_runtime.heldout_tasks,
        )
        exact = registry.register(
            exact_report,
            source_runtime.model,
            (target,),
            heldout_tasks=source_runtime.heldout_tasks,
        )

        selected = registry.select(target)

        self.assertIsNotNone(selected)
        self.assertEqual(compiled_circuit_id(selected.report), compiled_circuit_id(exact.report))
        self.assertNotEqual(compiled_circuit_id(selected.report), compiled_circuit_id(generic.report))


if __name__ == "__main__":
    unittest.main()
