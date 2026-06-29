import unittest

from cortex3 import ArithmeticSkill, CorruptedCompressedAgent, DynamicSkillVerifier, InstructionSkill, LongContextAnchorSkill, ReferenceRuleAgent
from cortex3_cycle import CortexCycle
from cortex3_ledgers import BitLedger, SkillLedger
from cortex3_phases import CORTEX3_PHASES, phase_table
from cortex3_selection import FrontierSelector, TrialProposal, TrialSelector


class CortexCycleModulesTest(unittest.TestCase):
    def test_phase_registry_has_ten_steps(self):
        self.assertEqual(len(CORTEX3_PHASES), 10)
        self.assertIn("Dynamic Skill Verifier", phase_table())

    def test_cycle_finds_regressions_and_actions(self):
        verifier = DynamicSkillVerifier([ArithmeticSkill(), LongContextAnchorSkill(), InstructionSkill()])
        report = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=4, n_per_skill=2)
        self.assertGreater(len(report.regressions), 0)
        self.assertGreater(len(report.analyses), 0)
        self.assertGreater(len(report.actions), 0)
        self.assertIn("trial_score", report.summary)

    def test_ledgers_and_selection(self):
        verifier = DynamicSkillVerifier([ArithmeticSkill()])
        report = verifier.evaluate(CorruptedCompressedAgent(), n_per_skill=2, seed=1)
        ledger = SkillLedger()
        ledger.update_from_report(report)
        selected = FrontierSelector().select(ledger)
        self.assertTrue(selected)
        bits = BitLedger()
        bits.weight_bits = 10
        self.assertEqual(bits.total_effective_bits, 10)
        decision = TrialSelector().decide(TrialProposal("reduce cost", 0.0, -1.0, tuple(selected), 0.1, "routing"), protected_skills=selected)
        self.assertTrue(decision.accepted)


if __name__ == "__main__":
    unittest.main()
