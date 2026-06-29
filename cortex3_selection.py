from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from cortex3_ledgers import SkillLedger


@dataclass(frozen=True)
class TrialProposal:
    title: str
    quality_delta: float
    cost_delta: float
    affected_skills: tuple[str, ...]
    risk: float
    kind: str


@dataclass(frozen=True)
class SelectionDecision:
    accepted: bool
    reason: str


class TrialSelector:
    def decide(self, proposal: TrialProposal, protected_skills: Iterable[str] = ()) -> SelectionDecision:
        protected = set(protected_skills)
        if protected.intersection(proposal.affected_skills) and proposal.risk > 0.15:
            return SelectionDecision(False, "protected skill risk is too high")
        if proposal.quality_delta <= 0 and proposal.cost_delta >= 0:
            return SelectionDecision(False, "not a quality or cost improvement")
        if proposal.risk > 0.5:
            return SelectionDecision(False, "trial risk is too high")
        return SelectionDecision(True, "accepted for offline evaluation")


class FrontierSelector:
    def select(self, ledger: SkillLedger, max_items: int = 3) -> list[str]:
        fragile = ledger.fragile_skills()
        if fragile:
            return [state.skill for state in fragile[:max_items]]
        return [state.skill for state in sorted(ledger.states.values(), key=lambda state: state.score)[:max_items]]
