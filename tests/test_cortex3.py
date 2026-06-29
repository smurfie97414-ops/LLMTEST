import random
import unittest

from cortex3 import (
    ArithmeticSkill,
    CandidateAnswer,
    CorruptedCompressedAgent,
    DynamicSkillVerifier,
    ExactAnchorLedger,
    InstructionSkill,
    LongContextAnchorSkill,
    MinimalRegrowthPlanner,
    ReferenceRuleAgent,
    ZeroState,
    ternarize_values,
)


class Cortex3Test(unittest.TestCase):
    def test_arithmetic_oracle(self):
        skill = ArithmeticSkill()
        for task in skill.generate(20, random.Random(0)):
            self.assertTrue(skill.verify(task, CandidateAnswer(str(task.expected))).passed)
            self.assertFalse(skill.verify(task, CandidateAnswer(str(int(task.expected) + 1))).passed)

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


if __name__ == "__main__":
    unittest.main()
