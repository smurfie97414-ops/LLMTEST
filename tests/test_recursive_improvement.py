import json
import tempfile
import unittest
from dataclasses import replace

from cortex3 import CorruptedCompressedAgent, DynamicSkillVerifier, ReferenceRuleAgent, default_skill_specs
from cortex3_cycle import CortexCycle
from cortex3_improvement import (
    DynamicEvaluator,
    DiversityPreserver,
    EvolutionaryArchive,
    ImprovementProposal,
    PatchAcceptanceGate,
    ProposalGenerator,
    ProposalKind,
    RecursiveImprovementEngine,
    RollbackSystem,
    SandboxTrainer,
)
from cortex3_reporting import write_cycle_run


def _verifier() -> DynamicSkillVerifier:
    return DynamicSkillVerifier(default_skill_specs())


def _cycle(seed: int = 3):
    verifier = _verifier()
    return verifier, CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=seed, n_per_skill=1)


class RecursiveImprovementTest(unittest.TestCase):
    def test_proposal_generator_maps_regressions_to_allowed_proposal_kinds(self):
        _, report = _cycle()
        proposals = ProposalGenerator().generate(report, max_proposals=8)

        self.assertTrue(proposals)
        self.assertTrue(all(proposal.kind in ProposalKind for proposal in proposals))
        self.assertTrue(all(proposal.affected_skills for proposal in proposals))
        self.assertTrue(all(proposal.diversity_tags for proposal in proposals))

    def test_restored_archive_kind_counts_still_guard_diversity(self):
        archive = EvolutionaryArchive()
        archive.restore_summary(
            accepted_count=3,
            rejected_count=1,
            kind_counts={ProposalKind.ROUTER.value: 3},
        )
        proposal = ImprovementProposal(
            "router-repeat",
            "repeat router patch",
            ProposalKind.ROUTER,
            ("instruction_following",),
            expected_quality_delta=0.1,
            expected_cost_delta=0.0,
            expected_robustness_delta=0.0,
            risk=0.1,
            diversity_tags=("router",),
        )

        flags = DiversityPreserver().check(proposal, archive)

        self.assertIn("proposal kind would dominate evolutionary archive", flags)

    def test_sandbox_trainer_uses_in_memory_patch_without_touching_files(self):
        verifier, report = _cycle()
        proposal = ProposalGenerator().generate(report, max_proposals=1)[0]
        sandbox = SandboxTrainer().train(proposal, baseline_agent=CorruptedCompressedAgent(), reference_agent=ReferenceRuleAgent())

        self.assertEqual(sandbox.touched_files, tuple())
        self.assertIn("in-memory", sandbox.notes)
        evaluation = DynamicEvaluator(verifier).evaluate(
            proposal,
            sandbox,
            baseline_agent=CorruptedCompressedAgent(),
            reference_agent=ReferenceRuleAgent(),
            protected_skills=(),
            seed=3,
            n_per_skill=1,
        )
        self.assertGreaterEqual(evaluation.quality_delta, 0.0)

    def test_engine_accepts_pareto_improving_sandbox_proposals(self):
        verifier, report = _cycle()
        improvement = RecursiveImprovementEngine(verifier).run(report, max_proposals=3, seed=3, n_per_skill=1)

        self.assertEqual(len(improvement.proposals), 3)
        self.assertGreater(len(improvement.archive.accepted), 0)
        for record in improvement.archive.accepted:
            self.assertTrue(record.decision.evaluation.pareto_candidate)
            self.assertEqual(record.decision.evaluation.sandbox.touched_files, tuple())
            self.assertFalse(record.decision.evaluation.protected_losses)
            self.assertFalse(record.decision.evaluation.reward_hacking_flags)
            self.assertFalse(record.decision.evaluation.collapse_flags)
            self.assertLessEqual(record.decision.evaluation.calibration_delta, 0.0)

    def test_engine_prioritizes_accepted_frontier_repair_proposals(self):
        verifier, report = _cycle()
        failure = report.regressions[0]
        frontier_proposals = ProposalGenerator().from_frontier_repairs((
            {
                "accepted": True,
                "task_id": failure.task.task_id,
                "skill": failure.task.skill,
                "source_failure_ids": (failure.task.task_id,),
                "frontier_task_ids": (failure.task.task_id, f"{failure.task.task_id}-frontier"),
                "repair_score_delta": 1.0,
                "protected_checked": 2,
                "frontier_compiled_verified": True,
            },
        ))

        improvement = RecursiveImprovementEngine(verifier).run(
            report,
            max_proposals=1,
            seed=3,
            n_per_skill=1,
            extra_proposals=frontier_proposals,
        )

        self.assertEqual(len(improvement.proposals), 1)
        proposal = improvement.proposals[0]
        self.assertEqual(proposal.kind, ProposalKind.COMPILED_FRONTIER)
        self.assertEqual(proposal.patch_payload["action"], "compile_frontier_repair")
        self.assertEqual(proposal.patch_payload["task_id"], failure.task.task_id)
        self.assertTrue(improvement.decisions[0].accepted, improvement.decisions[0].reason)

    def test_gate_rejects_protected_skill_regression(self):
        verifier = _verifier()
        proposal = ImprovementProposal(
            "degrade-protected",
            "degrade protected arithmetic",
            ProposalKind.ROUTER,
            ("arithmetic",),
            expected_quality_delta=0.1,
            expected_cost_delta=-1.0,
            expected_robustness_delta=0.0,
            risk=0.1,
            diversity_tags=("router", "arithmetic"),
            patch_payload={"degrade_skill": "arithmetic"},
        )
        sandbox = SandboxTrainer().train(proposal, baseline_agent=ReferenceRuleAgent(), reference_agent=ReferenceRuleAgent())
        evaluation = DynamicEvaluator(verifier).evaluate(
            proposal,
            sandbox,
            baseline_agent=ReferenceRuleAgent(),
            reference_agent=ReferenceRuleAgent(),
            protected_skills=("arithmetic",),
            seed=5,
            n_per_skill=1,
        )
        decision = PatchAcceptanceGate().decide(evaluation, RecursiveImprovementEngine(verifier).archive, protected_skills=("arithmetic",))

        self.assertFalse(decision.accepted)
        self.assertIn("arithmetic", evaluation.protected_losses)

    def test_reward_hacking_detector_flags_overfit_payload(self):
        verifier = _verifier()
        proposal = ImprovementProposal(
            "reward-hack",
            "overfit visible arithmetic",
            ProposalKind.TEST,
            ("arithmetic",),
            expected_quality_delta=0.2,
            expected_cost_delta=0.0,
            expected_robustness_delta=0.0,
            risk=0.1,
            diversity_tags=("test", "arithmetic"),
            patch_payload={"reward_hacking": True},
        )
        sandbox = SandboxTrainer().train(proposal, baseline_agent=CorruptedCompressedAgent(), reference_agent=ReferenceRuleAgent())
        evaluation = DynamicEvaluator(verifier).evaluate(
            proposal,
            sandbox,
            baseline_agent=CorruptedCompressedAgent(),
            reference_agent=ReferenceRuleAgent(),
            protected_skills=(),
            seed=6,
            n_per_skill=1,
        )

        self.assertTrue(evaluation.reward_hacking_flags)

    def test_gate_rejects_calibration_regression(self):
        verifier = _verifier()
        proposal = ImprovementProposal(
            "miscalibrate",
            "miscalibrate unknown calibration",
            ProposalKind.TEST,
            ("calibration",),
            expected_quality_delta=0.2,
            expected_cost_delta=0.0,
            expected_robustness_delta=0.1,
            risk=0.1,
            diversity_tags=("test", "calibration"),
            patch_payload={"miscalibrate": True},
        )
        sandbox = SandboxTrainer().train(proposal, baseline_agent=ReferenceRuleAgent(), reference_agent=ReferenceRuleAgent())
        evaluation = DynamicEvaluator(verifier).evaluate(
            proposal,
            sandbox,
            baseline_agent=ReferenceRuleAgent(),
            reference_agent=ReferenceRuleAgent(),
            protected_skills=(),
            seed=6,
            n_per_skill=2,
        )
        forced_pareto = replace(
            evaluation,
            quality_delta=0.1,
            cost_delta=0.0,
            robustness_delta=0.1,
            protected_losses={},
            reward_hacking_flags=tuple(),
            collapse_flags=tuple(),
        )
        decision = PatchAcceptanceGate().decide(forced_pareto, RecursiveImprovementEngine(verifier).archive, protected_skills=())

        self.assertGreater(evaluation.calibration_delta, 0.0)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "calibration regression")

    def test_evaluator_flags_unaffected_skill_collapse(self):
        verifier = _verifier()
        proposal = ImprovementProposal(
            "collapse-unaffected",
            "collapse an unrelated skill",
            ProposalKind.ROUTER,
            ("arithmetic",),
            expected_quality_delta=0.2,
            expected_cost_delta=0.0,
            expected_robustness_delta=0.1,
            risk=0.1,
            diversity_tags=("router", "arithmetic"),
            patch_payload={"degrade_skill": "instruction_following"},
        )
        sandbox = SandboxTrainer().train(proposal, baseline_agent=ReferenceRuleAgent(), reference_agent=ReferenceRuleAgent())
        evaluation = DynamicEvaluator(verifier).evaluate(
            proposal,
            sandbox,
            baseline_agent=ReferenceRuleAgent(),
            reference_agent=ReferenceRuleAgent(),
            protected_skills=(),
            seed=5,
            n_per_skill=1,
        )
        forced_pareto = replace(
            evaluation,
            quality_delta=0.1,
            cost_delta=0.0,
            robustness_delta=0.1,
            protected_losses={},
            reward_hacking_flags=tuple(),
            calibration_delta=0.0,
        )
        decision = PatchAcceptanceGate().decide(forced_pareto, RecursiveImprovementEngine(verifier).archive, protected_skills=())

        self.assertTrue(any("instruction_following" in flag for flag in evaluation.collapse_flags))
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "diversity preservation failed")

    def test_rollback_archive_records_accepted_patch_tokens(self):
        verifier, report = _cycle()
        improvement = RecursiveImprovementEngine(verifier).run(report, max_proposals=2, seed=3, n_per_skill=1)
        record = improvement.archive.accepted[0]
        rollback = RollbackSystem()
        event = rollback.rollback(record, reason="post-merge regression simulation")

        self.assertEqual(event.proposal_id, record.proposal.proposal_id)
        self.assertEqual(event.rollback_token, record.rollback_token)
        self.assertEqual(rollback.to_dict()["events"][0]["reason"], "post-merge regression simulation")

    def test_persistent_archive_round_trips_full_decisions_and_rollbacks(self):
        verifier, report = _cycle()
        engine = RecursiveImprovementEngine(verifier)
        improvement = engine.run(report, max_proposals=2, seed=3, n_per_skill=1)
        original_record = improvement.archive.accepted[0]
        engine.rollback.rollback(original_record, reason="post-run protected regression")

        with tempfile.TemporaryDirectory() as tmp:
            saved = engine.save_persistent_state(tmp)
            restored = RecursiveImprovementEngine(verifier)
            loaded = restored.load_persistent_state(tmp)

        self.assertTrue(loaded["archive_loaded"])
        self.assertTrue(loaded["rollback_loaded"])
        self.assertEqual(saved["decision_count"], loaded["decision_count"])
        self.assertEqual(restored.archive.accepted_count, engine.archive.accepted_count)
        self.assertEqual(restored.archive.rejected_count, engine.archive.rejected_count)
        self.assertEqual(len(restored.archive.accepted), len(engine.archive.accepted))
        restored_record = restored.archive.accepted[0]
        self.assertEqual(restored_record.proposal.proposal_id, original_record.proposal.proposal_id)
        self.assertEqual(restored_record.rollback_token, original_record.rollback_token)
        self.assertEqual(restored_record.decision.reason, original_record.decision.reason)
        self.assertAlmostEqual(
            restored_record.decision.evaluation.trial_report.aggregate_score,
            original_record.decision.evaluation.trial_report.aggregate_score,
        )
        self.assertEqual(restored.rollback.events[0].rollback_token, original_record.rollback_token)
        self.assertIn("post-run", restored.rollback.events[0].reason)

    def test_reporting_can_persist_recursive_improvement_report(self):
        verifier, report = _cycle(seed=4)
        improvement = RecursiveImprovementEngine(verifier).run(report, max_proposals=2, seed=4, n_per_skill=1)

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_cycle_run(report, output_dir=tmp, run_id="improvement-run", improvement_report=improvement)
            payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))

        self.assertTrue(payload["recursive_improvement"]["proposals"])
        self.assertTrue(payload["recursive_improvement"]["decisions"])
        self.assertIn("accepted", payload["recursive_improvement"]["archive"])


if __name__ == "__main__":
    unittest.main()
