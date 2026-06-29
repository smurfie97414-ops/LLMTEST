from cortex3 import ArithmeticSkill, CorruptedCompressedAgent, DynamicSkillVerifier, InstructionSkill, LongContextAnchorSkill, ReferenceRuleAgent
from cortex3_cycle import CortexCycle, cycle_report_markdown


def main() -> None:
    verifier = DynamicSkillVerifier([ArithmeticSkill(), LongContextAnchorSkill(), InstructionSkill()])
    report = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=7, n_per_skill=3)
    print(cycle_report_markdown(report))


if __name__ == "__main__":
    main()
