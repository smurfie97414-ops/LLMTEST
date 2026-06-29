from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from cortex3 import (
    AdaptiveHorizonPolicy,
    CandidateAnswer,
    CompressionAdversary,
    DynamicSkillVerifier,
    MinimalRegrowthPlanner,
    RegrowthAction,
    Task,
    VerificationCaseResult,
    VerificationSuiteReport,
)
from cortex3_analysis import FailureAnalysis, RegressionAnalyzer
from cortex3_ledgers import BitLedger, CausalLedger, CausalTrace, SkillLedger, UncertaintyLedger


@dataclass(frozen=True)
class PathDecision:
    path: str
    mtp_horizon: int
    verifier_level: int
    reason: str


class PathRouter:
    def __init__(self, horizon_policy: AdaptiveHorizonPolicy | None = None):
        self.horizon_policy = horizon_policy or AdaptiveHorizonPolicy()

    def route(self, skill: str, confidence: float, risk: float) -> PathDecision:
        domain = "exact_anchor" if skill == "long_context_anchor" else "math" if skill == "arithmetic" else "general"
        horizon = self.horizon_policy.choose(confidence, risk, domain)
        if risk > 0.7 or confidence < 0.55:
            return PathDecision("careful", horizon.horizon, 3, "high risk or low confidence")
        if horizon.horizon >= 6 and not horizon.requires_verification:
            return PathDecision("fast", horizon.horizon, 0, horizon.reason)
        return PathDecision("normal", horizon.horizon, 1 if horizon.requires_verification else 0, horizon.reason)


@dataclass(frozen=True)
class CycleReport:
    reference: VerificationSuiteReport
    trial: VerificationSuiteReport
    regressions: tuple[VerificationCaseResult, ...]
    analyses: tuple[FailureAnalysis, ...]
    actions: tuple[RegrowthAction, ...]
    extra_report: VerificationSuiteReport | None
    bit_ledger: BitLedger
    skill_ledger: SkillLedger
    calibration_gap: float

    @property
    def summary(self) -> Mapping[str, Any]:
        return {
            "reference_score": self.reference.aggregate_score,
            "trial_score": self.trial.aggregate_score,
            "regressions": len(self.regressions),
            "top_causes": [analysis.top_cause for analysis in self.analyses[:5]],
            "actions": [action.action for action in self.actions],
            "effective_bits": self.bit_ledger.total_effective_bits,
            "calibration_gap": self.calibration_gap,
            "trial_vc_per_cost": self.trial.verified_capability_per_cost,
        }


class CortexCycle:
    def __init__(self, verifier: DynamicSkillVerifier, explorer: CompressionAdversary | None = None):
        self.verifier = verifier
        self.explorer = explorer or CompressionAdversary(verifier.specs.values())
        self.analyzer = RegressionAnalyzer()
        self.regrowth = MinimalRegrowthPlanner()
        self.router = PathRouter()

    def run(self, reference: Callable[[Task], CandidateAnswer | str], trial: Callable[[Task], CandidateAnswer | str], *, seed: int = 0, n_per_skill: int = 4, repair_budget: float = 8.0) -> CycleReport:
        comparison = self.verifier.compare(reference, trial, n_per_skill=n_per_skill, seed=seed)
        ref: VerificationSuiteReport = comparison["reference"]  # type: ignore[assignment]
        trial_report: VerificationSuiteReport = comparison["candidate"]  # type: ignore[assignment]
        regressions: tuple[VerificationCaseResult, ...] = tuple(comparison["regressions"])  # type: ignore[arg-type]

        bit_ledger = BitLedger()
        skill_ledger = SkillLedger()
        uncertainty = UncertaintyLedger()
        traces = CausalLedger()
        skill_ledger.update_from_report(trial_report)

        for report in trial_report.skill_reports.values():
            for failure in report.failures:
                bit_ledger.ingest_cost(failure.answer.cost.merge(failure.verifier_cost), note=failure.task.skill)
                bit_ledger.add_certificate(failure.answer.certificate)
                uncertainty.record(failure.task.skill, failure.answer.confidence, failure.passed)
                risk = 1.0 - failure.answer.confidence
                route = self.router.route(failure.task.skill, failure.answer.confidence, risk)
                traces.record(CausalTrace(
                    task_id=failure.task.task_id,
                    skill=failure.task.skill,
                    mtp_horizon=route.mtp_horizon,
                    verifier_level=route.verifier_level,
                    kv_mode="latent" if failure.task.skill == "long_context_anchor" else "exact",
                    activation_bits=4 if failure.task.skill == "arithmetic" else 8,
                    certificate_fields=tuple(failure.answer.certificate.keys()),
                    uncertainty=risk,
                ))

        analyses = tuple(self.analyzer.analyze(f, traces.get(f.task.task_id)) for f in regressions)
        actions = tuple(self.regrowth.propose(regressions, budget=repair_budget))
        for action in actions:
            bit_ledger.scale_bits += action.cost * 8.0
        extra_tasks = self.explorer.expand_from_failures(regressions, seed=seed + 101, per_failure=2)
        extra_report = self.verifier.evaluate_tasks(trial, extra_tasks) if extra_tasks else None
        return CycleReport(ref, trial_report, regressions, analyses, actions, extra_report, bit_ledger, skill_ledger, uncertainty.expected_calibration_error())


def cycle_report_markdown(report: CycleReport) -> str:
    lines = ["# Cortex-3 Cycle Report", "", "## Summary"]
    for key, value in report.summary.items():
        lines.append(f"- **{key}**: `{value}`")
    lines += ["", "## Regressions"]
    if not report.regressions:
        lines.append("No regressions detected.")
    for idx, failure in enumerate(report.regressions[:20], 1):
        lines.append(f"{idx}. `{failure.task.skill}` failed: {failure.reason}")
    lines += ["", "## Analysis"]
    for analysis in report.analyses[:20]:
        hint = analysis.hints[0]
        lines.append(f"- `{analysis.failure.task.skill}` -> **{hint.cause}** ({hint.probability:.2f}); probe: {hint.probe}; repair: {hint.repair}")
    lines += ["", "## Budgeted Actions"]
    for action in report.actions:
        lines.append(f"- `{action.action}` on `{action.target}`: gain={action.expected_gain:.2f}, cost={action.cost:.2f}, reason={action.reason}")
    return "\n".join(lines) + "\n"
