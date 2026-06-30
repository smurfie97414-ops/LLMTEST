import json
import tempfile
import unittest

from cortex3 import CorruptedCompressedAgent, DynamicSkillVerifier, ReferenceRuleAgent, default_skill_specs
from cortex3_cycle import CortexCycle
from cortex3_frontier import FrontierSkillDiscovery
from cortex3_reporting import write_cycle_run


class FrontierSkillDiscoveryTest(unittest.TestCase):
    def test_frontier_discovery_slow_solves_distills_and_compiles_fragile_skill(self):
        verifier = DynamicSkillVerifier(default_skill_specs())
        cycle = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=3, n_per_skill=1)

        report = FrontierSkillDiscovery(verifier).discover(cycle, seed=3, max_skills=1, epochs=120)

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

    def test_frontier_discovery_report_is_persisted(self):
        verifier = DynamicSkillVerifier(default_skill_specs())
        cycle = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=4, n_per_skill=1)
        frontier = FrontierSkillDiscovery(verifier).discover(cycle, seed=4, max_skills=1, epochs=120)

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_cycle_run(cycle, output_dir=tmp, run_id="frontier-run", frontier_report=frontier)
            payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))

        self.assertTrue(payload["frontier_discovery"]["passed"])
        self.assertTrue(payload["frontier_discovery"]["circuits"])
        self.assertGreater(payload["frontier_discovery"]["circuits"][0]["compiled_weight_bits"], 0.0)


if __name__ == "__main__":
    unittest.main()
