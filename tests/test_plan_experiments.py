import json
import tempfile
import unittest

from cortex3 import CorruptedCompressedAgent, DynamicSkillVerifier, ReferenceRuleAgent, default_skill_specs
from cortex3_cycle import CortexCycle
from cortex3_experiments import CortexExperimentSuite
from cortex3_reporting import write_cycle_run


class PlanExperimentsTest(unittest.TestCase):
    def test_experiments_a_to_e_all_pass_with_named_metrics(self):
        report = CortexExperimentSuite().run_all(seed=3, n_per_skill=1)

        self.assertTrue(report.passed)
        self.assertEqual([result.experiment_id for result in report.results], ["A", "B", "C", "D", "E"])
        self.assertEqual(report.results[0].metrics["detection_rate"], 1.0)
        self.assertGreater(report.results[1].metrics["gain_over_fixed"], 0)
        self.assertLess(report.results[2].metrics["targeted_cost"], report.results[2].metrics["global_retrain_cost"])
        self.assertGreater(report.results[3].metrics["cost_reduction"], 0.0)
        self.assertGreater(report.results[4].metrics["accepted"], 0)
        self.assertFalse(report.results[4].metrics["collapse_flags"])
        self.assertFalse(report.results[4].metrics["calibration_regressions"])

    def test_experiment_report_is_persisted_in_cycle_artifact(self):
        verifier = DynamicSkillVerifier(default_skill_specs())
        cycle = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=4, n_per_skill=1)
        experiments = CortexExperimentSuite(verifier).run_all(seed=4, n_per_skill=1)

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_cycle_run(cycle, output_dir=tmp, run_id="experiments-run", experiments=experiments)
            payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))

        self.assertTrue(payload["experiments"]["passed"])
        self.assertEqual([result["experiment_id"] for result in payload["experiments"]["results"]], ["A", "B", "C", "D", "E"])


if __name__ == "__main__":
    unittest.main()
