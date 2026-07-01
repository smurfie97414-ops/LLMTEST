import json
import tempfile
import unittest
from pathlib import Path

from cortex3 import CandidateAnswer, CorruptedCompressedAgent, DynamicSkillVerifier, ReferenceRuleAgent, Task, default_skill_specs
from cortex3_cycle import CortexCycle
from cortex3_frontier import CompiledFrontierAgent, FrontierCircuitRegistry, FrontierSkillDiscovery
from cortex3_reporting import write_cycle_run
from cortex3_sleep import (
    AntiCollapseFilter,
    ExampleOrigin,
    FailureReplayBuffer,
    LocalExternalProvenanceAdapter,
    MetamorphicFamilyBuilder,
    RealExogenousReservoir,
    SleepPhaseConsolidator,
    ToolSolvedExampleFactory,
    TrainingExample,
    VerifiedSyntheticDataPool,
)


def _verifier() -> DynamicSkillVerifier:
    return DynamicSkillVerifier(default_skill_specs())


def _arithmetic_task(task_id: str = "arith-sleep") -> Task:
    return Task(
        task_id,
        "arithmetic",
        "Compute exactly: 20 + 22. Return only the integer.",
        42,
        {"kind": "add", "a": 20, "b": 22},
    )


class SleepPhaseTest(unittest.TestCase):
    def test_local_external_provenance_adapter_streams_deduplicates_and_oracle_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_path = root / "source.txt"
            jsonl_path = root / "source.jsonl"
            text_path.write_text(
                "CORTEX-3 external source span alpha\n"
                "CORTEX-3 external source span alpha\n",
                encoding="utf-8",
            )
            jsonl_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "prompt": "Compute exactly: 6 * 7. Return only the integer.",
                                "answer": "42",
                                "expected": 42,
                                "skill": "arithmetic",
                                "metadata": {"source_tag": "valid_math"},
                            }
                        ),
                        json.dumps(
                            {
                                "prompt": "Compute exactly: 6 * 7. Return only the integer.",
                                "answer": "41",
                                "expected": 42,
                                "skill": "arithmetic",
                                "metadata": {"source_tag": "bad_math"},
                            }
                        ),
                        json.dumps(
                            {
                                "prompt": "Unknown skill record should be rejected, not crash.",
                                "answer": "ok",
                                "expected": "ok",
                                "skill": "not_registered",
                                "metadata": {"source_tag": "unknown_skill"},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            records = LocalExternalProvenanceAdapter((text_path, jsonl_path), source_name="unit").records()
            limited_records = tuple(
                LocalExternalProvenanceAdapter((text_path, jsonl_path), source_name="unit").iter_records(max_records=1)
            )
            report = RealExogenousReservoir().ingest_external_records(records, verifier=_verifier())

        self.assertEqual(len(limited_records), 1)
        self.assertEqual(report.accepted_count, 2)
        self.assertEqual(report.duplicate_count, 1)
        self.assertEqual(report.rejected_count, 2)
        self.assertEqual(report.source_kind_counts["local_text"], 2)
        self.assertEqual(report.source_kind_counts["local_jsonl"], 3)
        for example in report.accepted_examples:
            self.assertEqual(example.origin, ExampleOrigin.REAL_EXOGENOUS)
            self.assertFalse(example.synthetic)
            self.assertTrue(example.answer.certificate["external_provenance_verified"])
            self.assertTrue(example.metadata["external_provenance_adapter"])
            verification = _verifier().oracle_registry.verify(example.oracle, example.task, example.answer)
            self.assertTrue(verification.passed)

    def test_verified_synthetic_pool_requires_trust_label(self):
        task = _arithmetic_task()
        unlabeled = TrainingExample(
            "bad-synthetic",
            task,
            CandidateAnswer("42", confidence=1.0),
            ExampleOrigin.VERIFIED_SYNTHETIC,
            oracle="arithmetic",
            targeted_skill="arithmetic",
            verification_level=1,
            contamination_risk=0.1,
            difficulty=0.4,
            confidence_label=None,
            synthetic=True,
        )

        with self.assertRaises(ValueError):
            VerifiedSyntheticDataPool().add(unlabeled)

    def test_failure_replay_buffer_and_tool_solver_create_verified_training_examples(self):
        verifier = _verifier()
        task = _arithmetic_task()
        failure = verifier.oracle_registry.verify("arithmetic", task, CandidateAnswer("43", confidence=0.82))

        replay = FailureReplayBuffer()
        replay_example = replay.add_failure(failure)
        tool_example = ToolSolvedExampleFactory(verifier).solve(task)

        self.assertEqual(replay_example.origin, ExampleOrigin.FAILURE_REPLAY)
        self.assertEqual(replay_example.answer.text, "42")
        self.assertFalse(replay_example.synthetic)
        self.assertEqual(tool_example.origin, ExampleOrigin.TOOL_SOLVED)
        self.assertTrue(tool_example.synthetic)
        self.assertEqual(tool_example.confidence_label, 1.0)
        self.assertGreaterEqual(tool_example.verification_level, 3)

    def test_metamorphic_family_builder_keeps_oracle_verified_labels(self):
        verifier = _verifier()
        examples = MetamorphicFamilyBuilder(verifier).build(_arithmetic_task(), seed=7, include_anti=True)

        self.assertTrue(examples)
        self.assertTrue(any(example.origin == ExampleOrigin.METAMORPHIC for example in examples))
        self.assertTrue(any(example.origin == ExampleOrigin.ANTI_METAMORPHIC for example in examples))
        for example in examples:
            verification = verifier.oracle_registry.verify(example.targeted_skill, example.task, example.answer)
            self.assertTrue(verification.passed)
            self.assertTrue(example.usable_synthetic_label)

    def test_anti_collapse_filter_rejects_high_contamination_duplicates_and_calibration_drop(self):
        task = _arithmetic_task()
        safe = TrainingExample(
            "safe",
            task,
            CandidateAnswer("42", confidence=1.0),
            ExampleOrigin.TOOL_SOLVED,
            "arithmetic",
            "arithmetic",
            3,
            0.1,
            0.4,
            1.0,
            True,
        )
        duplicate = TrainingExample(
            "duplicate",
            task,
            CandidateAnswer("42", confidence=1.0),
            ExampleOrigin.METAMORPHIC,
            "arithmetic",
            "arithmetic",
            2,
            0.1,
            0.4,
            1.0,
            True,
        )
        risky = TrainingExample(
            "risky",
            Task("risky", "arithmetic", "Compute exactly: 1 + 2. Return only the integer.", 3, {"kind": "add", "a": 1, "b": 2}),
            CandidateAnswer("3", confidence=1.0),
            ExampleOrigin.VERIFIED_SYNTHETIC,
            "arithmetic",
            "arithmetic",
            2,
            0.9,
            0.4,
            1.0,
            True,
        )

        decision = AntiCollapseFilter().evaluate((safe, duplicate, risky), baseline_calibration_gap=0.1, projected_calibration_gap=0.2)

        self.assertFalse(decision.accepted)
        self.assertIn("safe", [example.example_id for example in decision.accepted_examples])
        self.assertIn("duplicate", [example.example_id for example in decision.rejected_examples])
        self.assertIn("risky", [example.example_id for example in decision.rejected_examples])
        self.assertFalse(decision.calibration_ok)
        self.assertTrue(any("calibration" in reason for reason in decision.reasons))

    def test_sleep_phase_consolidator_builds_diverse_rare_skill_schedule(self):
        verifier = _verifier()
        report = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=3, n_per_skill=1)
        sleep = SleepPhaseConsolidator(verifier).ingest_cycle(report, seed=3)

        self.assertTrue(sleep.filter_decision.accepted)
        self.assertTrue(sleep.diversity_ok)
        self.assertTrue(sleep.calibration_ok)
        self.assertGreater(len(sleep.accepted_examples), 0)
        self.assertGreater(sleep.filter_decision.metrics.skill_entropy, 1.0)
        self.assertGreater(sleep.rare_skill_gain, 0.0)
        self.assertGreater(sleep.accepted_rare_skill_fraction, sleep.baseline_rare_skill_fraction)
        self.assertGreater(sleep.scheduled_rare_skill_fraction, 0.0)
        self.assertGreaterEqual(sleep.diversity_delta, 0.0)
        self.assertLessEqual(sleep.calibration_gap_delta, 0.0)
        self.assertTrue(sleep.schedule)
        protected = {state.skill for state in report.skill_ledger.fragile_skills()}
        self.assertTrue(protected.intersection({item.skill for item in sleep.schedule}))
        self.assertIn(sleep.schedule[0].skill, protected)
        self.assertEqual(sleep.schedule[0].reason, "rare fragile skill")
        scheduled_ids = {example_id for item in sleep.schedule for example_id in item.synthetic_examples + item.replay_examples + item.real_examples}
        rejected_ids = {example.example_id for example in sleep.rejected_examples}
        self.assertFalse(scheduled_ids.intersection(rejected_ids))

    def test_sleep_phase_acceptance_compiles_to_heldout_frontier_circuit(self):
        verifier = _verifier()
        report = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=5, n_per_skill=1)
        sleep = SleepPhaseConsolidator(verifier).ingest_cycle(report, seed=5)
        registry = FrontierCircuitRegistry()

        frontier = FrontierSkillDiscovery(verifier).compile_sleep_consolidation(
            sleep,
            seed=5,
            max_skills=1,
            epochs=60,
            registry=registry,
        )

        self.assertTrue(frontier.passed, frontier.to_dict())
        self.assertEqual(len(frontier.circuits), 1)
        circuit = frontier.circuits[0]
        self.assertEqual(circuit.training["source_kind"], "sleep_consolidation")
        self.assertGreater(circuit.training["sleep_accepted_examples"], 0)
        self.assertGreater(circuit.training["sleep_support_examples"], 0)
        self.assertTrue(circuit.training["sleep_source_example_ids"])
        self.assertTrue(circuit.heldout["gate_passed"])
        self.assertEqual(circuit.heldout["passed"], circuit.heldout["total"])
        self.assertEqual(registry.compiled_skills(), (circuit.skill,))
        runtime = registry.circuits_for_skill(circuit.skill)[0]
        self.assertTrue(runtime.heldout_tasks)
        task = runtime.verified_tasks[0]
        answer = CompiledFrontierAgent(registry, verifier=verifier)(task)
        self.assertTrue(answer.raw["frontier_compiled_selected"])
        self.assertTrue(answer.certificate["frontier_heldout_gate_passed"])
        self.assertTrue(answer.certificate["frontier_compiled_contract_verified"])
        self.assertTrue(verifier.oracle_registry.verify(task.skill, task, answer).passed)

    def test_real_exogenous_reservoir_and_reporting_persist_sleep_phase(self):
        verifier = _verifier()
        report = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=4, n_per_skill=1)
        reservoir = RealExogenousReservoir()
        reservoir.add(Task("real-format", "instruction_following", "Output YES exactly.", "YES"), "YES", source_id="manual-1")
        sleep = SleepPhaseConsolidator(verifier, reservoir=reservoir).ingest_cycle(report, seed=4)

        self.assertTrue(any(example.origin == ExampleOrigin.REAL_EXOGENOUS for example in sleep.accepted_examples))
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_cycle_run(report, output_dir=tmp, run_id="sleep-run", sleep_report=sleep)
            payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))

        self.assertTrue(payload["sleep_phase"]["diversity_ok"])
        self.assertTrue(payload["sleep_phase"]["calibration_ok"])
        self.assertGreaterEqual(payload["sleep_phase"]["rare_skill_gain"], 0.0)
        self.assertGreater(payload["sleep_phase"]["scheduled_rare_skill_fraction"], 0.0)
        self.assertLessEqual(payload["sleep_phase"]["calibration_gap_delta"], 0.0)
        self.assertGreater(len(payload["sleep_phase"]["accepted_examples"]), 0)
        self.assertTrue(payload["sleep_phase"]["schedule"])


if __name__ == "__main__":
    unittest.main()
