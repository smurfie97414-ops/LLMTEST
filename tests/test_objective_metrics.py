import json
import tempfile
import unittest

from cortex3 import CorruptedCompressedAgent, DynamicSkillVerifier, ReferenceRuleAgent, RegressionHarness, default_skill_specs
from cortex3_cycle import CortexCycle
from cortex3_improvement import RecursiveImprovementEngine
from cortex3_inference import UltraFastInferenceEngine
from cortex3_objective import ABSOLUTE_METRICS, FINAL_LOSS_TERMS, ObjectiveWeights, build_objective_report
from cortex3_reporting import write_cycle_run


def _cycle_bundle(seed: int = 3):
    verifier = DynamicSkillVerifier(default_skill_specs())
    cycle = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=seed, n_per_skill=1)
    faults = RegressionHarness(verifier).run_fault_matrix(seed=seed, n_per_skill=1)
    improvement = RecursiveImprovementEngine(verifier).run(cycle, max_proposals=2, seed=seed, n_per_skill=1)
    return verifier, cycle, faults, improvement


class ObjectiveMetricsTest(unittest.TestCase):
    def test_final_loss_contains_every_plan_term_with_weighted_total(self):
        _, cycle, faults, improvement = _cycle_bundle()
        objective = build_objective_report(cycle, fault_results=faults, improvement_report=improvement)

        self.assertEqual(tuple(objective.loss.terms), FINAL_LOSS_TERMS)
        weighted = sum(term.weighted for term in objective.loss.terms.values())
        self.assertAlmostEqual(objective.loss.total, weighted)
        for name, term in objective.loss.terms.items():
            self.assertEqual(term.name, name)
            self.assertGreaterEqual(term.raw, 0.0)
            self.assertTrue(term.evidence)

    def test_absolute_metrics_contains_every_plan_metric(self):
        _, cycle, faults, improvement = _cycle_bundle()
        objective = build_objective_report(cycle, fault_results=faults, improvement_report=improvement)

        self.assertEqual(tuple(objective.metrics.metrics), ABSOLUTE_METRICS)
        self.assertGreater(objective.metrics.metrics["cost_per_verified_correct_response"], 0.0)
        self.assertGreater(objective.metrics.metrics["effective_joules_per_correct_skill"], 0.0)
        self.assertIn("gap", objective.metrics.metrics["calibration"])
        self.assertIsInstance(objective.metrics.metrics["path_speed"], dict)
        self.assertGreater(objective.metrics.verified_capability_per_effective_joule, 0.0)

    def test_objective_uses_custom_weights(self):
        _, cycle, faults, improvement = _cycle_bundle()
        default = build_objective_report(cycle, fault_results=faults, improvement_report=improvement)
        weighted = build_objective_report(
            cycle,
            fault_results=faults,
            improvement_report=improvement,
            weights=ObjectiveWeights(rho=10.0, sigma=5.0),
        )

        self.assertGreater(weighted.loss.terms["L_skill_regression"].coefficient, default.loss.terms["L_skill_regression"].coefficient)
        self.assertGreater(weighted.loss.total, default.loss.total)

    def test_inference_results_feed_path_anchor_and_certificate_metrics(self):
        verifier, cycle, faults, improvement = _cycle_bundle()
        engine = UltraFastInferenceEngine(verifier, ReferenceRuleAgent())
        inference = (
            engine.infer(next(task for task in verifier.build_suite(1, 11) if task.skill == "instruction_following")),
            engine.infer(next(task for task in verifier.build_suite(1, 12) if task.skill == "arithmetic")),
        )
        objective = build_objective_report(cycle, inference_results=inference, fault_results=faults, improvement_report=improvement)

        self.assertTrue(objective.metrics.metrics["path_speed"])
        self.assertGreaterEqual(objective.metrics.metrics["tasks_without_heavy_verification_percent"], 0.0)
        self.assertLessEqual(objective.loss.terms["L_latent_certificate"].raw, 1.0)
        self.assertLessEqual(objective.loss.terms["L_hardware_layout"].raw, 1.0)

    def test_recursive_invalidity_counts_calibration_and_collapse(self):
        _, cycle, faults, _ = _cycle_bundle()

        class Evaluation:
            protected_losses = {}
            reward_hacking_flags = ()
            calibration_delta = 0.2
            collapse_flags = ("unaffected skill collapse: instruction_following",)

        class Decision:
            evaluation = Evaluation()
            diversity_flags = ()

        class Improvement:
            decisions = (Decision(),)

        objective = build_objective_report(cycle, fault_results=faults, improvement_report=Improvement())

        self.assertEqual(objective.loss.terms["L_recursive_improvement_validity"].raw, 1.0)

    def test_reporting_persists_objective_report(self):
        _, cycle, faults, improvement = _cycle_bundle(seed=4)
        objective = build_objective_report(cycle, fault_results=faults, improvement_report=improvement)

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_cycle_run(cycle, output_dir=tmp, run_id="objective-run", objective_report=objective)
            payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))

        self.assertEqual(len(payload["objective"]["loss"]["terms"]), len(FINAL_LOSS_TERMS))
        self.assertEqual(len(payload["objective"]["metrics"]["metrics"]), len(ABSOLUTE_METRICS))
        self.assertIn("verified_capability_per_effective_joule", payload["objective"]["metrics"])


if __name__ == "__main__":
    unittest.main()
