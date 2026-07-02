import random
import tempfile
import unittest
import json
from pathlib import Path

from cortex3 import (
    AlgebraSkill,
    ArithmeticSkill,
    CalibrationSkill,
    CandidateAnswer,
    CodeUnitTestSkill,
    CorruptedCompressedAgent,
    DynamicSkillVerifier,
    EntityTrackingSkill,
    ExactAnchorLedger,
    FaultType,
    InstructionSkill,
    LongContextAnchorSkill,
    MinimalRegrowthPlanner,
    OracleQualityAuditor,
    ReferenceRuleAgent,
    RegressionHarness,
    SkillReport,
    VerificationCaseResult,
    VerificationSuiteReport,
    VerifierCostProfiler,
    CostTrace,
    Task,
    ZeroState,
    default_skill_specs,
    ternarize_values,
)
from cortex3_cycle import CortexCycle
from cortex3_reporting import write_cycle_run


class Cortex3Test(unittest.TestCase):
    def test_arithmetic_oracle(self):
        skill = ArithmeticSkill()
        for task in skill.generate(20, random.Random(0)):
            self.assertTrue(skill.verify(task, CandidateAnswer(str(task.expected))).passed)
            self.assertFalse(skill.verify(task, CandidateAnswer(str(int(task.expected) + 1))).passed)
            self.assertFalse(skill.verify(task, CandidateAnswer(f"the answer is {task.expected}")).passed)

    def test_algebra_oracle_accepts_exact_symbolic_quadratic_root_sets(self):
        skill = AlgebraSkill()
        task = Task(
            "quadratic-oracle",
            "algebra",
            "Solve exactly for x: x^2 - x - 6 = 0. Return the exact roots as a comma-separated set.",
            "-2, 3",
            {"variable": "x", "a": 1, "b": -1, "c": -6, "kind": "quadratic"},
        )

        self.assertTrue(skill.verify(task, CandidateAnswer("-2, 3")).passed)
        self.assertTrue(skill.verify(task, CandidateAnswer("3, -2")).passed)
        self.assertFalse(skill.verify(task, CandidateAnswer("-2, 4")).passed)
        self.assertFalse(skill.verify(task, CandidateAnswer("x = -2, 3")).passed)

    def test_algebra_oracle_accepts_exact_symbolic_linear_system_assignments(self):
        skill = AlgebraSkill()
        task = Task(
            "system-oracle",
            "algebra",
            "Solve exactly for x and y: 2x + 3y = 7; -x + 4y = -9. Return only assignments.",
            "x=5, y=-1",
            {
                "kind": "linear_system_2x2",
                "variables": ("x", "y"),
                "coefficients": ((2, 3), (-1, 4)),
                "rhs": (7, -9),
                "solution": {"x": 5, "y": -1},
            },
        )

        self.assertTrue(skill.verify(task, CandidateAnswer("x=5, y=-1")).passed)
        self.assertFalse(skill.verify(task, CandidateAnswer("5, -1")).passed)
        self.assertFalse(skill.verify(task, CandidateAnswer("y=-1, x=5")).passed)
        self.assertFalse(skill.verify(task, CandidateAnswer("x=5, y=0")).passed)

    def test_reference_beats_corrupted_candidate(self):
        verifier = DynamicSkillVerifier([ArithmeticSkill(), LongContextAnchorSkill(), InstructionSkill()])
        ref = verifier.evaluate(ReferenceRuleAgent(), n_per_skill=3, seed=0)
        cand = verifier.evaluate(CorruptedCompressedAgent(), n_per_skill=3, seed=0)
        self.assertGreater(ref.aggregate_score, cand.aggregate_score)

    def test_compare_finds_regressions_and_regrowth(self):
        verifier = DynamicSkillVerifier([ArithmeticSkill(), LongContextAnchorSkill(), InstructionSkill()])
        comparison = verifier.compare(ReferenceRuleAgent(), CorruptedCompressedAgent(), n_per_skill=2, seed=2)
        regressions = comparison["regressions"]
        self.assertGreater(len(regressions), 0)
        actions = MinimalRegrowthPlanner().propose(regressions)
        self.assertGreater(len(actions), 0)

    def test_anchors_and_ternary(self):
        ledger = ExactAnchorLedger()
        required = ledger.ingest("code C3-1111-A et montant 12,00 €")
        self.assertEqual(ledger.fidelity_score("C3-1111-A 12,00 €", required), 1.0)
        block = ternarize_values([1.0, -2.0, 0.01, 3.0])
        self.assertEqual(block.q[0], 1)
        self.assertEqual(block.q[1], -1)
        self.assertEqual(block.q[2], 0)
        self.assertIn(ZeroState.ZERO_PROVISIONAL, block.zero_states)

    def test_phase1_default_skills_cover_plan_domains(self):
        specs = default_skill_specs()
        names = {spec.name for spec in specs}
        self.assertEqual(
            names,
            {
                "arithmetic",
                "algebra",
                "long_context_anchor",
                "entity_tracking",
                "instruction_following",
                "code_unit_tests",
                "calibration",
            },
        )
        verifier = DynamicSkillVerifier(specs)
        report = verifier.evaluate(ReferenceRuleAgent(), n_per_skill=2, seed=11)
        self.assertEqual(report.pass_rate, 1.0)
        profile = VerifierCostProfiler().summarize(report)
        self.assertIn("code_unit_tests", profile.by_skill)
        self.assertGreater(profile.total_cases, 0)
        self.assertTrue(all(skill_report.cases for skill_report in report.skill_reports.values()))
        self.assertGreater(profile.total_effective_cost, 0.0)

    def test_strict_exact_anchor_entity_and_calibration_oracles_reject_embedded_answers(self):
        for skill in [LongContextAnchorSkill(), EntityTrackingSkill(), CalibrationSkill()]:
            task = skill.generate(1, random.Random(3))[0]
            correct = CandidateAnswer(str(task.expected), confidence=1.0)
            embedded = CandidateAnswer(f"{task.expected} EXTRA", confidence=1.0)
            self.assertTrue(skill.verify(task, correct).passed)
            self.assertFalse(skill.verify(task, embedded).passed)

    def test_oracle_quality_auditor_detects_no_false_positive_or_negative_on_default_skills(self):
        report = OracleQualityAuditor(default_skill_specs()).audit(seed=7, n_per_skill=2)

        self.assertTrue(report.passed)
        self.assertGreater(report.total, 0)
        self.assertEqual(set(report.by_skill), {spec.name for spec in default_skill_specs()})
        for result in report.by_skill.values():
            self.assertEqual(result.false_positives, 0)
            self.assertEqual(result.false_negatives, 0)

    def test_verifier_cost_profile_uses_real_case_costs_not_count_allocation(self):
        task_a = Task("cost-a", "a", "a", "a")
        task_b = Task("cost-b", "b", "b", "b")
        answer = CandidateAnswer("ok")
        case_a = VerificationCaseResult(task_a, True, 1.0, answer, "a", "ok", CostTrace(verifier_steps=1))
        case_b = VerificationCaseResult(task_b, True, 1.0, answer, "b", "ok", CostTrace(verifier_steps=7, wall_time_ms=10.0))
        suite = VerificationSuiteReport(
            {
                "a": SkillReport("a", 1, 1, 1.0, (), (case_a,)),
                "b": SkillReport("b", 1, 1, 1.0, (), (case_b,)),
            },
            2,
            2,
            1.0,
            CostTrace(),
        )

        profile = VerifierCostProfiler().summarize(suite)

        self.assertGreater(profile.by_skill["b"].effective_cost, profile.by_skill["a"].effective_cost * 5)
        self.assertEqual(profile.by_skill["b"].verifier_steps, 7)
        self.assertGreater(profile.by_skill["b"].wall_time_ms, 0.0)

    def test_reporting_persists_schema_and_per_case_verifier_costs(self):
        verifier = DynamicSkillVerifier(default_skill_specs())
        report = verifier.evaluate(ReferenceRuleAgent(), n_per_skill=1, seed=4, include_metamorphic=False)

        with tempfile.TemporaryDirectory() as tmp:
            cycle = CortexCycle(verifier).run(ReferenceRuleAgent(), ReferenceRuleAgent(), seed=4, n_per_skill=1)
            artifacts = write_cycle_run(cycle, output_dir=tmp, run_id="phase1-costs")
            payload = json.loads(Path(artifacts.summary_json).read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], "cortex3.run.v1")
        arithmetic_cases = payload["reference"]["skill_reports"]["arithmetic"]["cases"]
        self.assertTrue(arithmetic_cases)
        self.assertIn("effective_cost", arithmetic_cases[0]["verifier_cost"])
        self.assertGreaterEqual(arithmetic_cases[0]["verifier_cost"]["wall_time_ms"], 0.0)
        self.assertTrue(report.skill_reports["arithmetic"].cases)

    def test_code_unit_test_oracle_rejects_wrong_implementation(self):
        skill = CodeUnitTestSkill()
        task = skill.generate(1, random.Random(4))[0]
        self.assertTrue(skill.verify(task, CandidateAnswer(str(task.expected))).passed)
        wrong = CandidateAnswer(str(task.metadata["wrong_impl"]))
        self.assertFalse(skill.verify(task, wrong).passed)

    def test_code_unit_test_oracle_checks_hidden_and_property_contracts(self):
        skill = CodeUnitTestSkill()
        task = Task(
            "code-property-contract",
            "code_unit_tests",
            "Write solve(values) that returns a copy of values without mutating it.",
            "def solve(values):\n    return list(values)\n",
            {
                "function_name": "solve",
                "tests": ((([1, 2],), [1, 2]),),
                "hidden_tests": ((([3],), [3]),),
                "properties": ("deterministic", "no_argument_mutation"),
                "require_hidden_tests": True,
            },
        )

        self.assertTrue(skill.verify(task, CandidateAnswer(str(task.expected))).passed)
        mutating = "def solve(values):\n    values.append(99)\n    return values[:-1]\n"
        nondeterministic = "def solve(values, _state=[]):\n    _state.append(1)\n    return list(values) if len(_state) == 1 else values[::-1]\n"
        missing_hidden = Task(
            "code-property-no-hidden",
            "code_unit_tests",
            task.prompt,
            task.expected,
            {**dict(task.metadata), "hidden_tests": tuple()},
        )
        self.assertFalse(skill.verify(task, CandidateAnswer(mutating)).passed)
        self.assertFalse(skill.verify(task, CandidateAnswer(nondeterministic)).passed)
        self.assertFalse(skill.verify(missing_hidden, CandidateAnswer(str(task.expected))).passed)

    def test_code_unit_test_generator_includes_rich_stateful_templates(self):
        skill = CodeUnitTestSkill()
        tasks = []
        for seed in range(8):
            tasks.extend(skill.generate(12, random.Random(seed)))
        by_title = {str(task.metadata["title"]): task for task in tasks}

        for title in ("dedupe_preserve_order", "merge_counts"):
            task = by_title[title]
            self.assertTrue(skill.verify(task, CandidateAnswer(str(task.expected))).passed)
            self.assertFalse(skill.verify(task, CandidateAnswer(str(task.metadata["wrong_impl"]))).passed)

    def test_entity_tracking_generator_includes_transfer_chain_oracle(self):
        skill = EntityTrackingSkill()
        transfer = None
        for seed in range(8):
            for task in skill.generate(12, random.Random(seed)):
                if task.metadata.get("kind") == "transfer_chain":
                    transfer = task
                    break
            if transfer is not None:
                break

        self.assertIsNotNone(transfer)
        assert transfer is not None
        self.assertTrue(skill.verify(transfer, CandidateAnswer(str(transfer.expected))).passed)
        self.assertFalse(skill.verify(transfer, CandidateAnswer(str(transfer.metadata["carrier"]))).passed)
        variant = skill.anti_metamorphic(transfer, random.Random(9))[0]
        self.assertNotEqual(variant.expected, transfer.expected)
        self.assertTrue(skill.verify(variant, CandidateAnswer(str(variant.expected))).passed)

    def test_anti_metamorphic_variants_change_expected_answers(self):
        for skill in [AlgebraSkill(), EntityTrackingSkill(), CalibrationSkill()]:
            base = skill.generate(1, random.Random(8))[0]
            variants = skill.anti_metamorphic(base, random.Random(9))
            self.assertTrue(variants)
            self.assertTrue(any(variant.expected != base.expected for variant in variants))

    def test_code_anti_metamorphic_variant_is_a_valid_changed_contract(self):
        skill = CodeUnitTestSkill()
        for base in skill.generate(8, random.Random(12)):
            variants = skill.anti_metamorphic(base, random.Random(13))
            self.assertTrue(variants)
            for variant in variants:
                self.assertTrue(skill.verify(variant, CandidateAnswer(str(variant.expected))).passed)

    def test_regression_harness_detects_injected_fault_matrix(self):
        verifier = DynamicSkillVerifier(default_skill_specs())
        results = RegressionHarness(verifier).run_fault_matrix(seed=5, n_per_skill=2)
        self.assertEqual({result.fault for result in results}, set(FaultType))
        self.assertTrue(all(result.detected for result in results))
        self.assertTrue(all(result.regressions > 0 for result in results))


if __name__ == "__main__":
    unittest.main()
