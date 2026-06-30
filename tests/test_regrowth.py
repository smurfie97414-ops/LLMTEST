import json
import tempfile
import unittest

from cortex3 import (
    ArithmeticSkill,
    CandidateAnswer,
    CorruptedCompressedAgent,
    DynamicSkillVerifier,
    InstructionSkill,
    LongContextAnchorSkill,
    ReferenceRuleAgent,
    Task,
    ZeroState,
    default_skill_specs,
    ternarize_values,
)
from cortex3_attribution import AblationDimension, CausalAttributionEngine, CausalAttributionReport, CauseEstimate
from cortex3_cycle import CortexCycle
from cortex3_ledgers import CausalTrace
from cortex3_regrowth import (
    MinimalRegrowthEngine,
    RegrowthActionKind,
    change_sign,
    increase_scale_precision_bits,
    unzero_block,
)
from cortex3_reporting import write_cycle_run
from cortex3_ternary import CompressionTraceLedger, make_compression_decision
from tools.run_cycle_report import build_regrowth_smoke


def _verifier() -> DynamicSkillVerifier:
    return DynamicSkillVerifier(default_skill_specs())


def _baseline_for_failure(failure):
    reference = ReferenceRuleAgent()

    def agent(task):
        if task.task_id == failure.task.task_id:
            return failure.answer
        return reference(task)

    return agent


class RegrowthTest(unittest.TestCase):
    def test_unzero_change_sign_and_scale_precision_are_real_artifact_edits(self):
        block = ternarize_values([0.01, -2.0, 3.0], threshold=1.0)
        self.assertEqual(block.mask[0], 0)
        repaired = unzero_block(block, [0])
        self.assertEqual(repaired.mask[0], 1)
        self.assertEqual(repaired.zero_states[0], ZeroState.ACTIVE)
        flipped = change_sign(repaired, {1: 1})
        self.assertEqual(flipped.signs[1], 1)
        self.assertEqual(increase_scale_precision_bits(16, 8), 24)

    def test_action_space_covers_all_plan_repair_actions(self):
        verifier = _verifier()
        task = Task("regrowth-action-space", "arithmetic", "Compute exactly 20 + 22.", 42)
        failure = ArithmeticSkill().verify(task, CandidateAnswer("43", confidence=0.82))
        causes = (
            CauseEstimate("block_overcompressed", 0.20, AblationDimension.BLOCK, "restore_block", True, 1.0, 0.50),
            CauseEstimate("memory_or_anchor_loss", 0.10, AblationDimension.KV_MODE, "force_exact_kv", True, 1.0, 0.40),
            CauseEstimate("future_horizon_too_long", 0.10, AblationDimension.MTP_HORIZON, "rerun_horizon_1", True, 1.0, 0.30),
            CauseEstimate("missing_specialist_path", 0.10, AblationDimension.EXPERT, "route_to_specialist_expert", True, 1.0, 0.25),
            CauseEstimate("activation_overquantized", 0.10, AblationDimension.ACTIVATION_PRECISION, "increase_activation", True, 1.0, 0.35),
            CauseEstimate("output_goal_missed", 0.10, AblationDimension.FSP_CONTRACT, "add_output_goal", True, 1.0, 0.20),
            CauseEstimate("code_oracle_failure", 0.10, AblationDimension.ROUTING, "add_verifier", True, 1.0, 0.20),
        )
        report = CausalAttributionReport(failure, (), causes, targeted_repair_cost=2.0)

        actions = MinimalRegrowthEngine(verifier).action_space.from_attribution(report)
        kinds = {action.kind for action in actions}

        self.assertEqual(kinds, set(RegrowthActionKind))
        activation_action = next(action for action in actions if action.kind == RegrowthActionKind.INCREASE_LOCAL_ACTIVATION_BITS)
        self.assertEqual(activation_action.metadata["cause"], "activation_overquantized")

    def test_minimal_regrowth_selects_recovering_block_action_with_non_regression(self):
        verifier = _verifier()
        task = Task("arith-regrowth", "arithmetic", "Compute exactly 20 + 22.", 42)
        failure = ArithmeticSkill().verify(task, CandidateAnswer("43", confidence=0.82))
        trace = CausalTrace(task_id=task.task_id, skill=task.skill, mtp_horizon=4, activation_bits=4)
        ledger = CompressionTraceLedger()
        _, decision = make_compression_decision("regrowth-block", [0.0, 0.01, 2.0, -3.0], certify_zeros=True)
        ledger.record_compression(decision)
        attribution = CausalAttributionEngine(verifier).attribute(failure, trace=trace, compression_ledger=ledger)

        plan = MinimalRegrowthEngine(verifier).plan(attribution, _baseline_for_failure(failure), protected_tasks=(task,), budget=25.0)
        self.assertIsNotNone(plan.selected)
        self.assertEqual(plan.selected.action.kind, RegrowthActionKind.INCREASE_LOCAL_ACTIVATION_BITS)
        self.assertTrue(plan.selected.recovered)
        self.assertTrue(plan.selected.non_regression.passed)
        self.assertGreater(plan.selected.gain_per_cost, 0.0)
        self.assertTrue(any(result.action.kind == RegrowthActionKind.UNZERO_BLOCK and result.recovered for result in plan.candidates))
        self.assertTrue(plan.annealing)
        self.assertTrue(plan.annealing[-1].retained)

    def test_force_exact_anchor_action_recovers_anchor_failure(self):
        verifier = _verifier()
        task = Task("anchor-regrowth", "long_context_anchor", "Return exact code.", "C3-7777-Z")
        failure = LongContextAnchorSkill().verify(task, CandidateAnswer("C3-7777-A", confidence=0.76))
        attribution = CausalAttributionEngine(verifier).attribute(
            failure,
            trace=CausalTrace(task_id=task.task_id, skill=task.skill, mtp_horizon=2, kv_mode="latent"),
        )
        plan = MinimalRegrowthEngine(verifier).plan(attribution, _baseline_for_failure(failure), protected_tasks=(task,), budget=12.0)
        self.assertIsNotNone(plan.selected)
        self.assertEqual(plan.selected.action.kind, RegrowthActionKind.FORCE_EXACT_ANCHOR)
        self.assertTrue(plan.selected.after.passed)
        self.assertIn("forced_anchors", plan.selected.patch.certificate_fields)

    def test_output_goal_certificate_action_repairs_instruction_format(self):
        verifier = _verifier()
        task = Task("instr-regrowth", "instruction_following", "Return exactly OK.", "OK")
        failure = InstructionSkill().verify(task, CandidateAnswer("OK\nDone.", confidence=0.88))
        attribution = CausalAttributionEngine(verifier).attribute(failure)
        plan = MinimalRegrowthEngine(verifier).plan(attribution, _baseline_for_failure(failure), protected_tasks=(task,), budget=12.0)
        self.assertIsNotNone(plan.selected)
        self.assertEqual(plan.selected.action.kind, RegrowthActionKind.ADD_CERTIFICATE_FIELD)
        self.assertEqual(plan.selected.patch.certificate_fields["output_goal"], "OK")
        self.assertTrue(plan.selected.after.passed)

    def test_training_micro_family_action_generates_verified_replay_tasks(self):
        verifier = _verifier()
        task = Task("arith-micro", "arithmetic", "Compute exactly 3 + 4.", 7, {"kind": "add", "a": 3, "b": 4})
        failure = ArithmeticSkill().verify(task, CandidateAnswer("8", confidence=0.70))
        attribution = CausalAttributionEngine(verifier).attribute(failure)
        action = next(action for action in MinimalRegrowthEngine(verifier).action_space.from_attribution(attribution) if action.kind == RegrowthActionKind.ADD_TRAINING_MICRO_FAMILY)
        result = MinimalRegrowthEngine(verifier).simulator.simulate(action, failure, _baseline_for_failure(failure), protected_tasks=(task,))
        self.assertTrue(result.patch.micro_family)
        self.assertTrue(result.non_regression.passed)

    def test_run_cycle_report_builds_regrowth_plans_from_real_cycle(self):
        verifier = _verifier()
        cycle_report = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=7, n_per_skill=1)

        plans = build_regrowth_smoke(verifier, cycle_report, budget=36.0)

        self.assertTrue(plans)
        self.assertEqual(len(plans), len(cycle_report.regressions))
        self.assertTrue(all(plan.selected is not None for plan in plans))
        for plan in plans:
            self.assertTrue(plan.selected.recovered)
            self.assertTrue(plan.selected.non_regression.passed)
            self.assertGreater(plan.selected.gain_per_cost, 0.0)

    def test_cycle_run_artifacts_can_include_regrowth_plan(self):
        verifier = _verifier()
        cycle_report = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=7, n_per_skill=1)
        failure = cycle_report.regressions[0]
        attribution = CausalAttributionEngine(verifier).attribute(failure)
        plan = MinimalRegrowthEngine(verifier).plan(attribution, _baseline_for_failure(failure), protected_tasks=(failure.task,), budget=25.0)
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_cycle_run(cycle_report, output_dir=tmp, run_id="regrowth-run", regrowth_plans=(plan,))
            payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
            self.assertIsNotNone(payload["regrowth"])
            self.assertEqual(payload["regrowth"][0]["failure"]["task_id"], failure.task.task_id)


if __name__ == "__main__":
    unittest.main()
