from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from cortex3 import CostTrace, FaultDetectionResult, VerificationSuiteReport
from cortex3_cycle import CycleReport


FINAL_LOSS_TERMS: tuple[str, ...] = (
    "L_behavior",
    "L_multi_horizon",
    "L_future_contract",
    "L_distillation_behavior",
    "L_distillation_uncertainty",
    "L_latent_certificate",
    "L_invariance",
    "L_temporal_consistency",
    "L_total_cognitive_description",
    "L_no_cost_shifting",
    "L_hardware_layout",
    "L_skill_regression",
    "L_calibration",
    "L_anchor_fidelity",
    "L_regrowth_efficiency",
    "L_verifier_resistance",
    "L_recursive_improvement_validity",
)


ABSOLUTE_METRICS: tuple[str, ...] = (
    "cost_per_verified_correct_response",
    "effective_joules_per_correct_skill",
    "bits_active_per_preserved_skill",
    "rare_regression_rate",
    "verifier_detection_rate",
    "verifier_false_negative_rate",
    "average_verification_cost",
    "mtp_rejection_rate",
    "token_inflation",
    "anchor_accuracy",
    "calibration",
    "regrowth_gain_per_added_bit",
    "path_speed",
    "tasks_without_heavy_verification_percent",
    "compiled_skills_from_slow_to_fast",
)


@dataclass(frozen=True)
class ObjectiveWeights:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0
    eta: float = 1.0
    theta: float = 1.0
    kappa: float = 1.0
    lambda_: float = 1.0
    mu: float = 1.0
    nu: float = 1.0
    rho: float = 1.0
    sigma: float = 1.0
    tau: float = 1.0
    psi: float = 1.0
    chi: float = 1.0
    omega: float = 1.0

    def coefficient_for(self, term: str) -> float:
        coefficients = {
            "L_behavior": 1.0,
            "L_multi_horizon": self.alpha,
            "L_future_contract": self.beta,
            "L_distillation_behavior": self.gamma,
            "L_distillation_uncertainty": self.delta,
            "L_latent_certificate": self.eta,
            "L_invariance": self.theta,
            "L_temporal_consistency": self.kappa,
            "L_total_cognitive_description": self.lambda_,
            "L_no_cost_shifting": self.mu,
            "L_hardware_layout": self.nu,
            "L_skill_regression": self.rho,
            "L_calibration": self.sigma,
            "L_anchor_fidelity": self.tau,
            "L_regrowth_efficiency": self.psi,
            "L_verifier_resistance": self.chi,
            "L_recursive_improvement_validity": self.omega,
        }
        return coefficients[term]

    def to_dict(self) -> dict[str, float]:
        data = asdict(self)
        data["lambda"] = data.pop("lambda_")
        return data


@dataclass(frozen=True)
class EffectiveJouleModel:
    joule_per_weight_bit: float = 1e-12
    joule_per_activation_bit: float = 2e-12
    joule_per_kv_byte: float = 8e-12
    joule_per_generated_token: float = 2e-9
    joule_per_latent_step: float = 5e-9
    joule_per_expert: float = 1e-8
    joule_per_verifier_step: float = 3e-9
    joule_per_ms: float = 1e-6

    def estimate(self, cost: CostTrace) -> float:
        return (
            cost.weight_bits_read * self.joule_per_weight_bit
            + cost.activation_bits * self.joule_per_activation_bit
            + cost.kv_bytes * self.joule_per_kv_byte
            + cost.generated_tokens * self.joule_per_generated_token
            + cost.latent_steps * self.joule_per_latent_step
            + cost.experts_activated * self.joule_per_expert
            + cost.verifier_steps * self.joule_per_verifier_step
            + cost.wall_time_ms * self.joule_per_ms
        )


@dataclass(frozen=True)
class LossTermValue:
    name: str
    raw: float
    coefficient: float
    weighted: float
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FinalLossReport:
    terms: Mapping[str, LossTermValue]
    total: float
    weights: ObjectiveWeights

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "weights": self.weights.to_dict(),
            "terms": {name: value.to_dict() for name, value in self.terms.items()},
        }


@dataclass(frozen=True)
class AbsoluteMetricsReport:
    metrics: Mapping[str, Any]
    verified_capability_per_effective_joule: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "verified_capability_per_effective_joule": self.verified_capability_per_effective_joule,
        }


@dataclass(frozen=True)
class CortexObjectiveReport:
    loss: FinalLossReport
    metrics: AbsoluteMetricsReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss.to_dict(),
            "metrics": self.metrics.to_dict(),
        }


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sum_cost(report: VerificationSuiteReport) -> float:
    return report.total_cost.effective_cost()


def _contract_decisions_from_future(future_ledger: Any | None) -> list[Any]:
    if future_ledger is None:
        return []
    return list(getattr(future_ledger, "decisions", ()) or ())


def _contract_decisions_from_inference(inference_results: Sequence[Any]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for result in inference_results:
        future = getattr(result, "future_contract", None) or {}
        if future:
            out.append(dict(future))
    return out


def _output_goal_rejection_rate(future_ledger: Any | None, inference_results: Sequence[Any]) -> float:
    if future_ledger is not None:
        decisions = list(getattr(future_ledger, "output_goal_decisions", ()) or ())
        if decisions:
            rejected = sum(1 for decision in decisions if not getattr(decision, "accepted", False))
            return _safe_ratio(rejected, len(decisions))
    output_goal_contracts: list[Mapping[str, Any]] = []
    for contract in _contract_decisions_from_inference(inference_results):
        output_goal = contract.get("output_goal_contract")
        if isinstance(output_goal, Mapping):
            output_goal_contracts.append(dict(output_goal))
    if not output_goal_contracts:
        return 0.0
    rejected = sum(1 for contract in output_goal_contracts if not bool(contract.get("accepted", False)))
    return _safe_ratio(rejected, len(output_goal_contracts))


def _mtp_rejection_rate(future_ledger: Any | None, inference_results: Sequence[Any]) -> float:
    decisions = _contract_decisions_from_future(future_ledger)
    if decisions:
        rejected = sum(1 for decision in decisions if not getattr(decision, "accepted", False))
        return _safe_ratio(rejected, len(decisions))
    contracts = _contract_decisions_from_inference(inference_results)
    if not contracts:
        return 0.0
    rejected = sum(1 for contract in contracts if not bool(contract.get("gate_accepted", False)))
    return _safe_ratio(rejected, len(contracts))


def _temporal_loss(future_ledger: Any | None) -> float:
    decisions = _contract_decisions_from_future(future_ledger)
    if not decisions:
        return 0.0
    losses = [float(getattr(decision.contract, "temporal_loss", 0.0)) for decision in decisions]
    return sum(losses) / len(losses)


def _certificate_failure_rate(inference_results: Sequence[Any]) -> float:
    if not inference_results:
        return 0.0
    failures = sum(1 for result in inference_results if not bool(getattr(result, "certificate_verified", True)))
    return _safe_ratio(failures, len(inference_results))


def _anchor_accuracy(inference_results: Sequence[Any], cycle_report: CycleReport) -> float:
    fidelity_scores: list[float] = []
    for result in inference_results:
        reconstruction = getattr(result, "memory_reconstruction", None)
        fidelity = getattr(reconstruction, "fidelity", None)
        if fidelity is not None and getattr(fidelity, "required", 0):
            fidelity_scores.append(float(getattr(fidelity, "score", 0.0)))
    if fidelity_scores:
        return sum(fidelity_scores) / len(fidelity_scores)
    anchor_failures = sum(1 for failure in cycle_report.regressions if failure.task.anchors)
    anchor_tasks = sum(report.total for skill, report in cycle_report.trial.skill_reports.items() if skill in {"long_context_anchor", "entity_tracking"})
    return 1.0 - _safe_ratio(anchor_failures, anchor_tasks, default=0.0)


def _hardware_layout_loss(inference_results: Sequence[Any]) -> float:
    dispatches = [dispatch for result in inference_results for dispatch in getattr(result, "kernel_dispatches", ())]
    if not dispatches:
        return 0.0
    unpacked = sum(1 for dispatch in dispatches if getattr(dispatch, "packed_weight_bytes", 0.0) <= 0.0)
    non_ternary = sum(1 for dispatch in dispatches if "ternary" not in str(getattr(dispatch, "mode", "")))
    return _safe_ratio(unpacked + non_ternary, 2 * len(dispatches))


def _regrowth_efficiency_loss(cycle_report: CycleReport) -> float:
    if not cycle_report.actions:
        return 0.0
    gain = sum(action.expected_gain for action in cycle_report.actions)
    added_bits = sum(action.cost * 8.0 for action in cycle_report.actions)
    efficiency = _safe_ratio(gain, added_bits)
    return 1.0 / (1.0 + efficiency)


def _verifier_detection_rate(fault_results: Sequence[FaultDetectionResult]) -> float:
    if not fault_results:
        return 0.0
    return _safe_ratio(sum(1 for result in fault_results if result.detected), len(fault_results))


def _recursive_invalidity(improvement_report: Any | None) -> float:
    if improvement_report is None:
        return 0.0
    decisions = list(getattr(improvement_report, "decisions", ()) or ())
    if not decisions:
        return 0.0
    invalid = 0
    for decision in decisions:
        evaluation = getattr(decision, "evaluation", None)
        decision_invalid = False
        if getattr(evaluation, "protected_losses", None):
            decision_invalid = True
        if getattr(evaluation, "reward_hacking_flags", None):
            decision_invalid = True
        if float(getattr(evaluation, "calibration_delta", 0.0) or 0.0) > 0.0:
            decision_invalid = True
        if getattr(evaluation, "collapse_flags", None):
            decision_invalid = True
        if getattr(decision, "diversity_flags", None):
            decision_invalid = True
        if decision_invalid:
            invalid += 1
    return _safe_ratio(invalid, len(decisions))


def _path_speed(inference_results: Sequence[Any]) -> dict[str, float]:
    by_path: dict[str, list[float]] = {}
    for result in inference_results:
        path = getattr(getattr(result, "route", None), "path", "")
        key = getattr(path, "value", str(path))
        cost = getattr(result, "cost", CostTrace()).effective_cost()
        by_path.setdefault(key, []).append(1.0 / max(cost, 1e-9))
    return {path: sum(values) / len(values) for path, values in by_path.items()}


def _tasks_without_heavy_verification_percent(inference_results: Sequence[Any]) -> float:
    if not inference_results:
        return 0.0
    light = sum(1 for result in inference_results if getattr(getattr(result, "route", None), "verifier_level", 3) < 3)
    return 100.0 * light / len(inference_results)


def _compiled_skills_from_slow_to_fast(inference_results: Sequence[Any]) -> int:
    skills = set()
    for result in inference_results:
        route = getattr(result, "route", None)
        path = getattr(getattr(route, "path", ""), "value", str(getattr(route, "path", "")))
        if path == "fast" and getattr(route, "verifier_level", 1) == 0:
            skills.add(getattr(getattr(result, "task", None), "skill", ""))
    return len({skill for skill in skills if skill})


def _loss_terms(
    cycle_report: CycleReport,
    *,
    future_ledger: Any | None,
    inference_results: Sequence[Any],
    fault_results: Sequence[FaultDetectionResult],
    improvement_report: Any | None,
    weights: ObjectiveWeights,
) -> FinalLossReport:
    trial = cycle_report.trial
    reference = cycle_report.reference
    mtp_rejection = _mtp_rejection_rate(future_ledger, inference_results)
    output_goal_rejection = _output_goal_rejection_rate(future_ledger, inference_results)
    anchor_accuracy = _anchor_accuracy(inference_results, cycle_report)
    detection_rate = _verifier_detection_rate(fault_results)
    raw: dict[str, tuple[float, str]] = {
        "L_behavior": (1.0 - trial.aggregate_score, "1 - trial aggregate verified score"),
        "L_multi_horizon": (mtp_rejection, "MTP/FSP rejection rate"),
        "L_future_contract": (max(mtp_rejection, output_goal_rejection), "token or output-goal future contracts rejected by gate"),
        "L_distillation_behavior": (max(0.0, reference.aggregate_score - trial.aggregate_score), "reference minus trial score"),
        "L_distillation_uncertainty": (cycle_report.calibration_gap, "cycle uncertainty calibration gap"),
        "L_latent_certificate": (_certificate_failure_rate(inference_results), "inference certificate failures"),
        "L_invariance": (1.0 - cycle_report.extra_report.aggregate_score if cycle_report.extra_report else 0.0, "metamorphic/adversarial extra report loss"),
        "L_temporal_consistency": (_temporal_loss(future_ledger), "average future-contract temporal loss"),
        "L_total_cognitive_description": (_safe_ratio(_sum_cost(trial), _sum_cost(reference) + _sum_cost(trial), default=0.0), "trial effective cognitive cost share"),
        "L_no_cost_shifting": (_clamp01(_safe_ratio(_sum_cost(trial) - _sum_cost(reference), max(_sum_cost(reference), 1e-9))), "excess trial cost over reference"),
        "L_hardware_layout": (_hardware_layout_loss(inference_results), "missing packed ternary dispatch evidence"),
        "L_skill_regression": (_safe_ratio(len(cycle_report.regressions), max(trial.total, 1)), "rare regression cases over trial cases"),
        "L_calibration": (cycle_report.calibration_gap, "calibration gap"),
        "L_anchor_fidelity": (1.0 - anchor_accuracy, "1 - exact anchor fidelity"),
        "L_regrowth_efficiency": (_regrowth_efficiency_loss(cycle_report), "inverse gain per added bit for regrowth actions"),
        "L_verifier_resistance": (1.0 - detection_rate if fault_results else 0.0, "fault matrix false-negative pressure"),
        "L_recursive_improvement_validity": (_recursive_invalidity(improvement_report), "invalid recursive improvement proposals"),
    }
    terms: dict[str, LossTermValue] = {}
    for name in FINAL_LOSS_TERMS:
        value, evidence = raw[name]
        coeff = weights.coefficient_for(name)
        terms[name] = LossTermValue(name, float(value), coeff, float(value) * coeff, evidence)
    return FinalLossReport(terms, sum(term.weighted for term in terms.values()), weights)


def _absolute_metrics(
    cycle_report: CycleReport,
    *,
    future_ledger: Any | None,
    inference_results: Sequence[Any],
    fault_results: Sequence[FaultDetectionResult],
    joule_model: EffectiveJouleModel,
) -> AbsoluteMetricsReport:
    trial = cycle_report.trial
    reference = cycle_report.reference
    verified_correct = max(float(trial.passed), 1.0)
    cost = trial.total_cost.effective_cost()
    joules = joule_model.estimate(trial.total_cost)
    detection_rate = _verifier_detection_rate(fault_results)
    generated_trial = trial.total_cost.generated_tokens
    generated_ref = reference.total_cost.generated_tokens
    gain = sum(action.expected_gain for action in cycle_report.actions)
    added_bits = sum(action.cost * 8.0 for action in cycle_report.actions)
    metrics: dict[str, Any] = {
        "cost_per_verified_correct_response": cost / verified_correct,
        "effective_joules_per_correct_skill": joules / verified_correct,
        "bits_active_per_preserved_skill": cycle_report.bit_ledger.total_effective_bits / verified_correct,
        "rare_regression_rate": _safe_ratio(len(cycle_report.regressions), max(trial.total, 1)),
        "verifier_detection_rate": detection_rate,
        "verifier_false_negative_rate": 1.0 - detection_rate if fault_results else 0.0,
        "average_verification_cost": _safe_ratio(trial.total_cost.verifier_steps * 3.0, max(trial.total, 1)),
        "mtp_rejection_rate": _mtp_rejection_rate(future_ledger, inference_results),
        "token_inflation": _safe_ratio(generated_trial, max(generated_ref, 1), default=1.0),
        "anchor_accuracy": _anchor_accuracy(inference_results, cycle_report),
        "calibration": {"gap": cycle_report.calibration_gap, "score": 1.0 - cycle_report.calibration_gap},
        "regrowth_gain_per_added_bit": _safe_ratio(gain, added_bits),
        "path_speed": _path_speed(inference_results),
        "tasks_without_heavy_verification_percent": _tasks_without_heavy_verification_percent(inference_results),
        "compiled_skills_from_slow_to_fast": _compiled_skills_from_slow_to_fast(inference_results),
    }
    verified_capability_per_joule = trial.aggregate_score / max(joules, 1e-18)
    return AbsoluteMetricsReport(metrics, verified_capability_per_joule)


def build_objective_report(
    cycle_report: CycleReport,
    *,
    future_ledger: Any | None = None,
    inference_results: Iterable[Any] | None = None,
    fault_results: Iterable[FaultDetectionResult] | None = None,
    improvement_report: Any | None = None,
    weights: ObjectiveWeights | None = None,
    joule_model: EffectiveJouleModel | None = None,
) -> CortexObjectiveReport:
    inference_tuple = tuple(inference_results or ())
    fault_tuple = tuple(fault_results or ())
    resolved_weights = weights or ObjectiveWeights()
    resolved_joule_model = joule_model or EffectiveJouleModel()
    return CortexObjectiveReport(
        loss=_loss_terms(
            cycle_report,
            future_ledger=future_ledger,
            inference_results=inference_tuple,
            fault_results=fault_tuple,
            improvement_report=improvement_report,
            weights=resolved_weights,
        ),
        metrics=_absolute_metrics(
            cycle_report,
            future_ledger=future_ledger,
            inference_results=inference_tuple,
            fault_results=fault_tuple,
            joule_model=resolved_joule_model,
        ),
    )
