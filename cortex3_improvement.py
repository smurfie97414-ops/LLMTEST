from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

from cortex3 import (
    CandidateAnswer,
    CorruptedCompressedAgent,
    DynamicSkillVerifier,
    ReferenceRuleAgent,
    Task,
    VerificationSuiteReport,
)
from cortex3_cycle import CycleReport
from cortex3_ledgers import UncertaintyLedger
from cortex3_selection import TrialProposal, TrialSelector


Agent = Callable[[Task], CandidateAnswer | str]


class ProposalKind(str, Enum):
    SKILL_SPEC = "skill_spec"
    TEST = "test"
    COMPRESSION = "compression"
    ROUTER = "router"
    MTP_HEAD = "mtp_head"
    REGROWTH_STRATEGY = "regrowth_strategy"
    HARDWARE_GRAMMAR = "hardware_grammar"
    KERNEL = "kernel"


@dataclass(frozen=True)
class ImprovementProposal:
    proposal_id: str
    title: str
    kind: ProposalKind
    affected_skills: tuple[str, ...]
    expected_quality_delta: float
    expected_cost_delta: float
    expected_robustness_delta: float
    risk: float
    diversity_tags: tuple[str, ...]
    patch_payload: Mapping[str, Any] = field(default_factory=dict)
    parent_ids: tuple[str, ...] = ()

    def to_trial_proposal(self) -> TrialProposal:
        return TrialProposal(
            self.title,
            self.expected_quality_delta,
            self.expected_cost_delta,
            self.affected_skills,
            self.risk,
            self.kind.value,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


class ProposalGenerator:
    action_kind_map: Mapping[str, ProposalKind] = {
        "increase_activation_bits": ProposalKind.COMPRESSION,
        "increase_local_activation_bits": ProposalKind.COMPRESSION,
        "reduce_mtp_horizon": ProposalKind.MTP_HEAD,
        "force_exact_anchor": ProposalKind.ROUTER,
        "route_specialist_expert": ProposalKind.ROUTER,
        "add_unit_test_oracle": ProposalKind.TEST,
        "add_format_certificate": ProposalKind.TEST,
        "add_certificate_field": ProposalKind.TEST,
        "add_training_micro_family": ProposalKind.SKILL_SPEC,
        "unzero_block": ProposalKind.REGROWTH_STRATEGY,
        "change_sign": ProposalKind.REGROWTH_STRATEGY,
    }

    def generate(self, report: CycleReport, *, max_proposals: int = 12) -> tuple[ImprovementProposal, ...]:
        proposals: list[ImprovementProposal] = []
        seen: set[str] = set()
        for idx, action in enumerate(report.actions):
            kind = self.action_kind_map.get(action.action, ProposalKind.REGROWTH_STRATEGY)
            skill = action.target
            proposal_id = f"proposal-{idx}-{kind.value}-{skill}"
            if proposal_id in seen:
                continue
            seen.add(proposal_id)
            proposals.append(ImprovementProposal(
                proposal_id=proposal_id,
                title=f"{action.action} for {skill}",
                kind=kind,
                affected_skills=(skill,),
                expected_quality_delta=max(0.01, action.expected_gain),
                expected_cost_delta=action.cost,
                expected_robustness_delta=max(0.0, action.expected_gain * 0.5),
                risk=min(0.45, action.cost / 20.0),
                diversity_tags=(kind.value, skill, action.action),
                patch_payload={"action": action.action, "target": skill, "mode": "repair_skill"},
            ))
        for failure in report.regressions[:max(0, max_proposals - len(proposals))]:
            skill = failure.task.skill
            proposal_id = f"proposal-test-{failure.task.task_id}"
            if proposal_id in seen:
                continue
            seen.add(proposal_id)
            proposals.append(ImprovementProposal(
                proposal_id=proposal_id,
                title=f"add verifier replay for {skill}",
                kind=ProposalKind.TEST,
                affected_skills=(skill,),
                expected_quality_delta=0.05,
                expected_cost_delta=1.0,
                expected_robustness_delta=0.10,
                risk=0.10,
                diversity_tags=("replay", skill),
                patch_payload={"action": "add_replay_case", "target": skill, "task_id": failure.task.task_id, "mode": "repair_skill"},
            ))
        return tuple(proposals[:max_proposals])


class ProposalPatchedAgent:
    def __init__(self, base: Agent, reference: Agent, proposal: ImprovementProposal):
        self.base = base
        self.reference = reference
        self.proposal = proposal

    def __call__(self, task: Task) -> CandidateAnswer:
        degrade_skill = self.proposal.patch_payload.get("degrade_skill")
        if degrade_skill == task.skill:
            base = CandidateAnswer.coerce(self.base(task))
            text = "0" if task.skill in {"arithmetic", "algebra"} else "<degraded>"
            return CandidateAnswer(text, confidence=max(base.confidence, 0.95), raw={"proposal": self.proposal.proposal_id, "degraded": True})
        if self.proposal.patch_payload.get("reward_hacking") and task.skill in self.proposal.affected_skills:
            if task.metadata.get("metamorphic") or task.metadata.get("anti_metamorphic") or task.metadata.get("adversarial"):
                return CandidateAnswer("<overfit-visible-case>", confidence=0.99, raw={"proposal": self.proposal.proposal_id, "reward_hacking": True})
            return CandidateAnswer.coerce(self.reference(task))
        if self.proposal.patch_payload.get("miscalibrate") and task.skill == "calibration":
            answer = CandidateAnswer.coerce(self.reference(task))
            confidence = 0.99 if answer.text == "UNKNOWN" else 0.05
            return CandidateAnswer(answer.text, confidence=confidence, raw={"proposal": self.proposal.proposal_id, "miscalibrated": True})
        if task.skill in self.proposal.affected_skills and self.proposal.patch_payload.get("mode", "repair_skill") == "repair_skill":
            answer = CandidateAnswer.coerce(self.reference(task))
            return CandidateAnswer(
                answer.text,
                confidence=answer.confidence,
                certificate={**dict(answer.certificate), "sandbox_proposal": self.proposal.proposal_id},
                cost=answer.cost,
                raw={"proposal": self.proposal.proposal_id, "sandbox_only": True},
            )
        return CandidateAnswer.coerce(self.base(task))


@dataclass(frozen=True)
class SandboxTrial:
    proposal: ImprovementProposal
    sandbox_id: str
    agent: Agent
    touched_files: tuple[str, ...]
    rollback_token: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal.proposal_id,
            "sandbox_id": self.sandbox_id,
            "touched_files": list(self.touched_files),
            "rollback_token": self.rollback_token,
            "notes": self.notes,
        }


class SandboxTrainer:
    def train(self, proposal: ImprovementProposal, *, baseline_agent: Agent, reference_agent: Agent) -> SandboxTrial:
        return SandboxTrial(
            proposal=proposal,
            sandbox_id=f"sandbox-{proposal.proposal_id}",
            agent=ProposalPatchedAgent(baseline_agent, reference_agent, proposal),
            touched_files=tuple(),
            rollback_token=f"rollback-{proposal.proposal_id}",
            notes="in-memory sandbox patch only; no direct self-modification",
        )


@dataclass(frozen=True)
class SandboxEvaluation:
    proposal: ImprovementProposal
    sandbox: SandboxTrial
    baseline_report: VerificationSuiteReport
    trial_report: VerificationSuiteReport
    robustness_report: VerificationSuiteReport
    quality_delta: float
    cost_delta: float
    robustness_delta: float
    baseline_calibration_gap: float
    trial_calibration_gap: float
    calibration_delta: float
    protected_losses: Mapping[str, float]
    reward_hacking_flags: tuple[str, ...]
    collapse_flags: tuple[str, ...]

    @property
    def pareto_candidate(self) -> bool:
        quality_up = self.quality_delta > 0.0 and self.cost_delta <= 1e-6
        cost_down = self.cost_delta < 0.0 and self.quality_delta >= -1e-6
        robustness_up = self.robustness_delta > 0.0 and self.quality_delta >= -1e-6
        return quality_up or cost_down or robustness_up

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal.proposal_id,
            "sandbox": self.sandbox.to_dict(),
            "baseline_score": self.baseline_report.aggregate_score,
            "trial_score": self.trial_report.aggregate_score,
            "robustness_score": self.robustness_report.aggregate_score,
            "quality_delta": self.quality_delta,
            "cost_delta": self.cost_delta,
            "robustness_delta": self.robustness_delta,
            "baseline_calibration_gap": self.baseline_calibration_gap,
            "trial_calibration_gap": self.trial_calibration_gap,
            "calibration_delta": self.calibration_delta,
            "protected_losses": dict(self.protected_losses),
            "reward_hacking_flags": list(self.reward_hacking_flags),
            "collapse_flags": list(self.collapse_flags),
            "pareto_candidate": self.pareto_candidate,
        }


class RewardHackingDetector:
    def detect(self, proposal: ImprovementProposal, trial: VerificationSuiteReport, robustness: VerificationSuiteReport) -> tuple[str, ...]:
        flags: list[str] = []
        affected = set(proposal.affected_skills)
        if proposal.patch_payload.get("reward_hacking"):
            flags.append("proposal payload declares reward-hacking behavior")
        trial_affected = [
            report.score
            for skill, report in trial.skill_reports.items()
            if skill in affected
        ]
        robustness_affected = [
            report.score
            for skill, report in robustness.skill_reports.items()
            if skill in affected
        ]
        if trial_affected and robustness_affected:
            trial_score = sum(trial_affected) / len(trial_affected)
            robustness_score = sum(robustness_affected) / len(robustness_affected)
        else:
            trial_score = trial.aggregate_score
            robustness_score = robustness.aggregate_score
        if trial_score - robustness_score > 0.25:
            flags.append("robustness suite score collapses relative to main suite")
        for report in robustness.skill_reports.values():
            if report.skill not in affected:
                continue
            for failure in report.failures:
                if failure.answer.confidence >= 0.95:
                    flags.append(f"overconfident robustness failure on {failure.task.skill}")
                    break
        return tuple(flags)


def _calibration_gap(report: VerificationSuiteReport) -> float:
    ledger = UncertaintyLedger()
    for skill, skill_report in report.skill_reports.items():
        for case in skill_report.cases:
            ledger.record(skill, case.answer.confidence, case.passed)
    return ledger.expected_calibration_error()


def _collapse_flags(
    proposal: ImprovementProposal,
    baseline: VerificationSuiteReport,
    trial: VerificationSuiteReport,
    baseline_robustness: VerificationSuiteReport,
    robustness: VerificationSuiteReport,
) -> tuple[str, ...]:
    flags: list[str] = []
    affected = set(proposal.affected_skills)
    for skill, base_skill in baseline.skill_reports.items():
        if skill in affected:
            continue
        trial_skill = trial.skill_reports.get(skill)
        if trial_skill is not None and trial_skill.score < base_skill.score - 0.10:
            flags.append(f"unaffected skill collapse: {skill}")
    if robustness.aggregate_score < baseline_robustness.aggregate_score - 0.20:
        flags.append("robustness aggregate collapse")
    baseline_passing = sum(1 for report in baseline.skill_reports.values() if report.pass_rate > 0.0)
    trial_passing = sum(1 for report in trial.skill_reports.values() if report.pass_rate > 0.0)
    if trial_passing < baseline_passing:
        flags.append("passing skill coverage collapse")
    return tuple(flags)


class DiversityPreserver:
    def __init__(self, max_kind_fraction: float = 0.70):
        self.max_kind_fraction = max_kind_fraction

    def check(self, proposal: ImprovementProposal, archive: "EvolutionaryArchive") -> tuple[str, ...]:
        flags: list[str] = []
        if not proposal.diversity_tags:
            flags.append("proposal has no diversity tags")
        accepted = archive.accepted
        if len(accepted) >= 3:
            counts = Counter(record.proposal.kind for record in accepted)
            projected_total = len(accepted) + 1
            projected_fraction = (counts[proposal.kind] + 1) / projected_total
            if projected_fraction > self.max_kind_fraction:
                flags.append("proposal kind would dominate evolutionary archive")
        return tuple(flags)


class DynamicEvaluator:
    def __init__(self, verifier: DynamicSkillVerifier, reward_detector: RewardHackingDetector | None = None):
        self.verifier = verifier
        self.reward_detector = reward_detector or RewardHackingDetector()

    def evaluate(
        self,
        proposal: ImprovementProposal,
        sandbox: SandboxTrial,
        *,
        baseline_agent: Agent,
        reference_agent: Agent,
        protected_skills: Iterable[str] = (),
        seed: int = 0,
        n_per_skill: int = 2,
    ) -> SandboxEvaluation:
        baseline = self.verifier.evaluate(baseline_agent, n_per_skill=n_per_skill, seed=seed, include_metamorphic=True)
        trial = self.verifier.evaluate(sandbox.agent, n_per_skill=n_per_skill, seed=seed, include_metamorphic=True)
        robustness_tasks = self.verifier.build_suite(max(1, n_per_skill), seed + 911, include_metamorphic=True, include_anti_metamorphic=True)
        baseline_robustness = self.verifier.evaluate_tasks(baseline_agent, robustness_tasks)
        robustness = self.verifier.evaluate_tasks(sandbox.agent, robustness_tasks)
        protected_losses: dict[str, float] = {}
        for skill in protected_skills:
            base_skill = baseline.skill_reports.get(skill)
            trial_skill = trial.skill_reports.get(skill)
            if base_skill is not None and trial_skill is not None and trial_skill.score < base_skill.score - 1e-9:
                protected_losses[skill] = base_skill.score - trial_skill.score
        quality_delta = trial.aggregate_score - baseline.aggregate_score
        cost_delta = trial.total_cost.effective_cost() - baseline.total_cost.effective_cost()
        robustness_delta = robustness.aggregate_score - baseline_robustness.aggregate_score
        baseline_calibration_gap = _calibration_gap(baseline)
        trial_calibration_gap = _calibration_gap(trial)
        reward_flags = self.reward_detector.detect(proposal, trial, robustness)
        collapse_flags = _collapse_flags(proposal, baseline, trial, baseline_robustness, robustness)
        return SandboxEvaluation(
            proposal,
            sandbox,
            baseline,
            trial,
            robustness,
            quality_delta,
            cost_delta,
            robustness_delta,
            baseline_calibration_gap,
            trial_calibration_gap,
            trial_calibration_gap - baseline_calibration_gap,
            protected_losses,
            reward_flags,
            collapse_flags,
        )


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    reason: str
    evaluation: SandboxEvaluation
    diversity_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "proposal_id": self.evaluation.proposal.proposal_id,
            "diversity_flags": list(self.diversity_flags),
            "evaluation": self.evaluation.to_dict(),
        }


class PatchAcceptanceGate:
    def __init__(self, selector: TrialSelector | None = None, diversity: DiversityPreserver | None = None):
        self.selector = selector or TrialSelector()
        self.diversity = diversity or DiversityPreserver()

    def decide(self, evaluation: SandboxEvaluation, archive: "EvolutionaryArchive", protected_skills: Iterable[str] = ()) -> AcceptanceDecision:
        proposal = evaluation.proposal
        selector_decision = self.selector.decide(proposal.to_trial_proposal(), protected_skills=protected_skills)
        diversity_flags = self.diversity.check(proposal, archive)
        if not selector_decision.accepted:
            return AcceptanceDecision(False, selector_decision.reason, evaluation, diversity_flags)
        if not evaluation.pareto_candidate:
            return AcceptanceDecision(False, "not a Pareto improvement", evaluation, diversity_flags)
        if evaluation.protected_losses:
            return AcceptanceDecision(False, "protected skill regression", evaluation, diversity_flags)
        if evaluation.calibration_delta > 1e-9:
            return AcceptanceDecision(False, "calibration regression", evaluation, diversity_flags)
        if evaluation.reward_hacking_flags:
            return AcceptanceDecision(False, "reward hacking detected", evaluation, diversity_flags)
        if evaluation.collapse_flags or diversity_flags:
            return AcceptanceDecision(False, "diversity preservation failed", evaluation, diversity_flags)
        return AcceptanceDecision(True, "accepted by Pareto/protected/diversity gate", evaluation, diversity_flags)


@dataclass(frozen=True)
class ArchiveRecord:
    proposal: ImprovementProposal
    decision: AcceptanceDecision
    rollback_token: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "decision": self.decision.to_dict(),
            "rollback_token": self.rollback_token,
        }


class EvolutionaryArchive:
    def __init__(self) -> None:
        self.records: list[ArchiveRecord] = []

    def record(self, decision: AcceptanceDecision) -> ArchiveRecord:
        record = ArchiveRecord(decision.evaluation.proposal, decision, decision.evaluation.sandbox.rollback_token)
        self.records.append(record)
        return record

    @property
    def accepted(self) -> tuple[ArchiveRecord, ...]:
        return tuple(record for record in self.records if record.decision.accepted)

    @property
    def rejected(self) -> tuple[ArchiveRecord, ...]:
        return tuple(record for record in self.records if not record.decision.accepted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": [record.to_dict() for record in self.accepted],
            "rejected": [record.to_dict() for record in self.rejected],
            "kind_counts": dict(Counter(record.proposal.kind.value for record in self.accepted)),
        }


@dataclass(frozen=True)
class RollbackEvent:
    proposal_id: str
    rollback_token: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RollbackSystem:
    def __init__(self) -> None:
        self.events: list[RollbackEvent] = []

    def rollback(self, record: ArchiveRecord, *, reason: str) -> RollbackEvent:
        event = RollbackEvent(record.proposal.proposal_id, record.rollback_token, reason)
        self.events.append(event)
        return event

    def to_dict(self) -> dict[str, Any]:
        return {"events": [event.to_dict() for event in self.events]}


@dataclass(frozen=True)
class RecursiveImprovementReport:
    proposals: tuple[ImprovementProposal, ...]
    decisions: tuple[AcceptanceDecision, ...]
    archive: EvolutionaryArchive
    rollback: RollbackSystem

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "archive": self.archive.to_dict(),
            "rollback": self.rollback.to_dict(),
        }


class RecursiveImprovementEngine:
    def __init__(
        self,
        verifier: DynamicSkillVerifier,
        *,
        generator: ProposalGenerator | None = None,
        trainer: SandboxTrainer | None = None,
        evaluator: DynamicEvaluator | None = None,
        gate: PatchAcceptanceGate | None = None,
        archive: EvolutionaryArchive | None = None,
        rollback: RollbackSystem | None = None,
    ):
        self.verifier = verifier
        self.generator = generator or ProposalGenerator()
        self.trainer = trainer or SandboxTrainer()
        self.evaluator = evaluator or DynamicEvaluator(verifier)
        self.gate = gate or PatchAcceptanceGate()
        self.archive = archive or EvolutionaryArchive()
        self.rollback = rollback or RollbackSystem()

    def run(
        self,
        report: CycleReport,
        *,
        baseline_agent: Agent | None = None,
        reference_agent: Agent | None = None,
        max_proposals: int = 6,
        seed: int = 0,
        n_per_skill: int = 1,
    ) -> RecursiveImprovementReport:
        baseline = baseline_agent or CorruptedCompressedAgent()
        reference = reference_agent or ReferenceRuleAgent()
        protected = tuple(state.skill for state in report.skill_ledger.fragile_skills())
        proposals = self.generator.generate(report, max_proposals=max_proposals)
        decisions: list[AcceptanceDecision] = []
        for proposal in proposals:
            sandbox = self.trainer.train(proposal, baseline_agent=baseline, reference_agent=reference)
            evaluation = self.evaluator.evaluate(
                proposal,
                sandbox,
                baseline_agent=baseline,
                reference_agent=reference,
                protected_skills=protected,
                seed=seed,
                n_per_skill=n_per_skill,
            )
            decision = self.gate.decide(evaluation, self.archive, protected_skills=protected)
            decisions.append(decision)
            self.archive.record(decision)
        return RecursiveImprovementReport(proposals, tuple(decisions), self.archive, self.rollback)
