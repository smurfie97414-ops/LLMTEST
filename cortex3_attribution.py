from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from cortex3 import CandidateAnswer, CostTrace, DynamicSkillVerifier, VerificationCaseResult
from cortex3_future import FutureContractLedger
from cortex3_ledgers import CausalTrace
from cortex3_ternary import CompressionDecision, CompressionTraceLedger


class AblationDimension(str, Enum):
    BLOCK = "block"
    EXPERT = "expert"
    KV_MODE = "kv_mode"
    MTP_HORIZON = "mtp_horizon"
    ACTIVATION_PRECISION = "activation_precision"
    FSP_CONTRACT = "fsp_contract"
    ROUTING = "routing"


@dataclass(frozen=True)
class AblationProbeSpec:
    probe_id: str
    dimension: AblationDimension
    cause: str
    intervention: str
    expected_cost: CostTrace
    target: str = ""
    metadata: Mapping[str, Any] | None = None

    def meta(self) -> Mapping[str, Any]:
        return dict(self.metadata or {})


@dataclass(frozen=True)
class AblationProbeResult:
    spec: AblationProbeSpec
    baseline_passed: bool
    baseline_score: float
    counterfactual_passed: bool
    counterfactual_score: float
    score_delta: float
    recovered: bool
    reason: str
    counterfactual_answer: str
    cost: CostTrace

    @property
    def gain_per_cost(self) -> float:
        return max(0.0, self.score_delta) / max(self.cost.effective_cost(), 1e-9)


@dataclass(frozen=True)
class CauseEstimate:
    cause: str
    probability: float
    best_dimension: AblationDimension
    best_intervention: str
    recovered: bool
    score_delta: float
    gain_per_cost: float


@dataclass(frozen=True)
class AttributionPolicySignal:
    skill: str
    cause: str
    dominant_intervention: str
    attempts: int
    successes: int
    failures: int
    mean_score_delta: float
    mean_gain_per_cost: float
    posterior_success: float
    policy_weight: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AttributionPolicyMemory:
    def __init__(
        self,
        *,
        prior_success: float = 1.0,
        prior_failure: float = 1.0,
        min_weight: float = 0.35,
        max_weight: float = 3.0,
    ):
        if prior_success <= 0.0 or prior_failure <= 0.0:
            raise ValueError("attribution policy priors must be positive")
        if min_weight <= 0.0 or max_weight < min_weight:
            raise ValueError("invalid attribution policy weight bounds")
        self.prior_success = float(prior_success)
        self.prior_failure = float(prior_failure)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self._stats: dict[tuple[str, str], dict[str, Any]] = {}

    def _key(self, skill: str, cause: str) -> tuple[str, str]:
        return (str(skill), str(cause))

    def _empty_stats(self) -> dict[str, Any]:
        return {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "score_delta_total": 0.0,
            "gain_per_cost_total": 0.0,
            "interventions": Counter(),
        }

    def _signal_from_stats(self, skill: str, cause: str, stats: Mapping[str, Any]) -> AttributionPolicySignal:
        attempts = int(stats.get("attempts", 0) or 0)
        successes = int(stats.get("successes", 0) or 0)
        failures = int(stats.get("failures", 0) or 0)
        interventions = Counter(dict(stats.get("interventions") or {}))
        dominant = interventions.most_common(1)[0][0] if interventions else ""
        mean_score = float(stats.get("score_delta_total", 0.0) or 0.0) / max(1, attempts)
        mean_gain = float(stats.get("gain_per_cost_total", 0.0) or 0.0) / max(1, attempts)
        posterior = (successes + self.prior_success) / max(
            successes + failures + self.prior_success + self.prior_failure,
            1e-9,
        )
        confidence = attempts / (attempts + 3.0) if attempts > 0 else 0.0
        if attempts <= 0:
            weight = 1.0
        else:
            success_pressure = 2.0 * posterior
            gain_pressure = min(1.0, max(0.0, mean_gain))
            weight = (1.0 - confidence) + confidence * (0.50 + success_pressure + 0.50 * gain_pressure)
            weight = max(self.min_weight, min(self.max_weight, weight))
        return AttributionPolicySignal(
            skill=str(skill),
            cause=str(cause),
            dominant_intervention=str(dominant),
            attempts=attempts,
            successes=successes,
            failures=failures,
            mean_score_delta=mean_score,
            mean_gain_per_cost=mean_gain,
            posterior_success=posterior,
            policy_weight=weight,
            confidence=confidence,
        )

    def signal_for(self, skill: str, cause: str) -> AttributionPolicySignal:
        return self._signal_from_stats(str(skill), str(cause), self._stats.get(self._key(skill, cause), self._empty_stats()))

    def observe(
        self,
        *,
        skill: str,
        cause: str,
        intervention: str,
        recovered: bool,
        score_delta: float,
        gain_per_cost: float,
        protected_regression: bool = False,
    ) -> AttributionPolicySignal:
        key = self._key(skill, cause)
        stats = self._stats.setdefault(key, self._empty_stats())
        stats["attempts"] = int(stats.get("attempts", 0)) + 1
        success = bool(recovered) and not bool(protected_regression) and float(score_delta) > 0.0
        if success:
            stats["successes"] = int(stats.get("successes", 0)) + 1
        else:
            stats["failures"] = int(stats.get("failures", 0)) + 1
        stats["score_delta_total"] = float(stats.get("score_delta_total", 0.0)) + max(0.0, float(score_delta))
        stats["gain_per_cost_total"] = float(stats.get("gain_per_cost_total", 0.0)) + max(0.0, float(gain_per_cost))
        interventions = Counter(stats.get("interventions") or {})
        interventions.update((str(intervention),))
        stats["interventions"] = interventions
        return self._signal_from_stats(str(skill), str(cause), stats)

    def observe_regrowth_plan(self, plan: Any) -> AttributionPolicySignal | None:
        selected = getattr(plan, "selected", None)
        if selected is None:
            return None
        action = getattr(selected, "action", None)
        if action is None:
            return None
        metadata = dict(getattr(action, "metadata", {}) or {})
        cause = str(metadata.get("cause") or "")
        if not cause:
            return None
        non_regression = getattr(selected, "non_regression", None)
        protected_regression = not bool(getattr(non_regression, "passed", False))
        failure = getattr(plan, "failure", None)
        task = getattr(failure, "task", None)
        skill = str(getattr(task, "skill", getattr(action, "target", "")))
        kind = getattr(action, "kind", "")
        intervention = str(getattr(kind, "value", kind))
        return self.observe(
            skill=skill,
            cause=cause,
            intervention=intervention,
            recovered=bool(getattr(selected, "recovered", False)),
            score_delta=float(getattr(selected, "score_delta", 0.0)),
            gain_per_cost=float(getattr(selected, "gain_per_cost", 0.0)),
            protected_regression=protected_regression,
        )

    def observe_model_regrowth_application(
        self,
        plan: Any,
        application: Mapping[str, Any],
    ) -> AttributionPolicySignal:
        selected = getattr(plan, "selected", None)
        if selected is None:
            raise ValueError("cannot observe model regrowth without a selected P7 action")
        action = getattr(selected, "action", None)
        if action is None:
            raise ValueError("cannot observe model regrowth without an action")
        metadata = dict(getattr(action, "metadata", {}) or {})
        cause = str(metadata.get("cause") or "")
        if not cause:
            raise ValueError("cannot observe model regrowth without an attributed cause")

        failure = getattr(plan, "failure", None)
        task = getattr(failure, "task", None)
        skill = str(application.get("failure_skill") or getattr(task, "skill", getattr(action, "target", "")))
        kind = getattr(action, "kind", "")
        intervention = str(getattr(kind, "value", kind))

        repair_delta = float(application.get("repair_loss_delta", 0.0) or 0.0)
        protected_delta = float(application.get("protected_loss_delta", 0.0) or 0.0)
        protected_tolerance = float(application.get("protected_loss_tolerance", 0.0) or 0.0)
        non_regression_passed = bool(application.get("non_regression_passed"))
        rollback_ready = bool(application.get("rollback_executable")) and bool(application.get("signed_patch_id"))
        parameter_delta = float(application.get("parameter_delta_l1", 0.0) or 0.0)
        updated_parameters = int(application.get("updated_parameter_count", 0) or 0)
        gradient_parameters = int(application.get("gradient_parameter_count", 0) or 0)
        action_cost = float(getattr(selected, "total_cost", 0.0) or 0.0)
        effective_cost = max(action_cost + 0.01 * max(0, updated_parameters + gradient_parameters), 1e-9)
        protected_regression = (not non_regression_passed) or protected_delta > protected_tolerance
        accepted_real_patch = (
            repair_delta > 0.0
            and parameter_delta > 0.0
            and rollback_ready
            and not protected_regression
        )
        effective_score_delta = repair_delta if accepted_real_patch else 0.0

        return self.observe(
            skill=skill,
            cause=cause,
            intervention=intervention,
            recovered=accepted_real_patch,
            score_delta=effective_score_delta,
            gain_per_cost=effective_score_delta / effective_cost,
            protected_regression=protected_regression,
        )

    def apply(
        self,
        skill: str,
        estimates: Sequence[CauseEstimate],
    ) -> tuple[tuple[CauseEstimate, ...], tuple[AttributionPolicySignal, ...], bool]:
        if not estimates:
            return tuple(), tuple(), False
        signals = tuple(self.signal_for(skill, estimate.cause) for estimate in estimates)
        adjusted_mass = [
            max(0.0, float(estimate.probability)) * max(0.0, signal.policy_weight)
            for estimate, signal in zip(estimates, signals)
        ]
        total = sum(adjusted_mass)
        if total <= 0.0:
            adjusted_mass = [float(estimate.probability) for estimate in estimates]
            total = sum(adjusted_mass) or 1.0
        adjusted = tuple(
            CauseEstimate(
                cause=estimate.cause,
                probability=mass / total,
                best_dimension=estimate.best_dimension,
                best_intervention=estimate.best_intervention,
                recovered=estimate.recovered,
                score_delta=estimate.score_delta,
                gain_per_cost=estimate.gain_per_cost,
            )
            for estimate, mass in zip(estimates, adjusted_mass)
        )
        policy_applied = any(signal.attempts > 0 for signal in signals)
        return tuple(sorted(adjusted, key=lambda estimate: estimate.probability, reverse=True)), signals, policy_applied

    @property
    def observation_count(self) -> int:
        return sum(int(stats.get("attempts", 0) or 0) for stats in self._stats.values())

    @property
    def success_count(self) -> int:
        return sum(int(stats.get("successes", 0) or 0) for stats in self._stats.values())

    def to_dict(self) -> dict[str, Any]:
        entries = []
        for (skill, cause), stats in sorted(self._stats.items()):
            signal = self._signal_from_stats(skill, cause, stats)
            entries.append({
                **signal.to_dict(),
                "interventions": dict(Counter(stats.get("interventions") or {})),
            })
        return {
            "schema_version": 1,
            "prior_success": self.prior_success,
            "prior_failure": self.prior_failure,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "observation_count": self.observation_count,
            "success_count": self.success_count,
            "entries": entries,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "AttributionPolicyMemory":
        data = dict(payload or {})
        memory = cls(
            prior_success=float(data.get("prior_success", 1.0)),
            prior_failure=float(data.get("prior_failure", 1.0)),
            min_weight=float(data.get("min_weight", 0.35)),
            max_weight=float(data.get("max_weight", 3.0)),
        )
        for entry in data.get("entries", ()):
            item = dict(entry)
            skill = str(item.get("skill", ""))
            cause = str(item.get("cause", ""))
            if not skill or not cause:
                continue
            stats = memory._empty_stats()
            stats["attempts"] = int(item.get("attempts", 0) or 0)
            stats["successes"] = int(item.get("successes", 0) or 0)
            stats["failures"] = int(item.get("failures", 0) or 0)
            stats["score_delta_total"] = float(item.get("mean_score_delta", 0.0) or 0.0) * max(1, stats["attempts"])
            stats["gain_per_cost_total"] = float(item.get("mean_gain_per_cost", 0.0) or 0.0) * max(1, stats["attempts"])
            stats["interventions"] = Counter(dict(item.get("interventions") or {}))
            memory._stats[memory._key(skill, cause)] = stats
        return memory


@dataclass(frozen=True)
class CausalAttributionReport:
    failure: VerificationCaseResult
    probes: tuple[AblationProbeResult, ...]
    causes: tuple[CauseEstimate, ...]
    targeted_repair_cost: float
    global_retrain_cost: float = 100.0
    policy_signals: tuple[AttributionPolicySignal, ...] = ()
    policy_applied: bool = False

    @property
    def top_cause(self) -> str:
        return self.causes[0].cause if self.causes else "unknown"

    @property
    def targeted_repair_is_cheaper(self) -> bool:
        return any(probe.recovered for probe in self.probes) and self.targeted_repair_cost < self.global_retrain_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure": {
                "task_id": self.failure.task.task_id,
                "skill": self.failure.task.skill,
                "reason": self.failure.reason,
                "score": self.failure.score,
            },
            "top_cause": self.top_cause,
            "targeted_repair_cost": self.targeted_repair_cost,
            "global_retrain_cost": self.global_retrain_cost,
            "targeted_repair_is_cheaper": self.targeted_repair_is_cheaper,
            "policy_applied": self.policy_applied,
            "policy_signals": [signal.to_dict() for signal in self.policy_signals],
            "causes": [
                {
                    "cause": cause.cause,
                    "probability": cause.probability,
                    "best_dimension": cause.best_dimension.value,
                    "best_intervention": cause.best_intervention,
                    "recovered": cause.recovered,
                    "score_delta": cause.score_delta,
                    "gain_per_cost": cause.gain_per_cost,
                }
                for cause in self.causes
            ],
            "probes": [
                {
                    "spec": {
                        "probe_id": probe.spec.probe_id,
                        "dimension": probe.spec.dimension.value,
                        "cause": probe.spec.cause,
                        "intervention": probe.spec.intervention,
                        "target": probe.spec.target,
                        "metadata": dict(probe.spec.metadata or {}),
                    },
                    "baseline_passed": probe.baseline_passed,
                    "baseline_score": probe.baseline_score,
                    "counterfactual_passed": probe.counterfactual_passed,
                    "counterfactual_score": probe.counterfactual_score,
                    "score_delta": probe.score_delta,
                    "recovered": probe.recovered,
                    "reason": probe.reason,
                    "counterfactual_answer": probe.counterfactual_answer,
                    "cost": asdict(probe.cost),
                    "gain_per_cost": probe.gain_per_cost,
                }
                for probe in self.probes
            ],
        }


@dataclass(frozen=True)
class RegressionCluster:
    cluster_id: str
    cause: str
    skill: str
    task_ids: tuple[str, ...]
    count: int
    mean_probability: float
    recommended_intervention: str


@dataclass(frozen=True)
class AttributionBatchReport:
    reports: tuple[CausalAttributionReport, ...]
    clusters: tuple[RegressionCluster, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reports": [report.to_dict() for report in self.reports],
            "clusters": [asdict(cluster) for cluster in self.clusters],
        }


def _expected_answer_for_failure(failure: VerificationCaseResult) -> str:
    expected = failure.expected
    if failure.task.skill == "code_unit_tests":
        return str(failure.task.metadata.get("expected", expected))
    return str(expected)


def _confidence_for_skill(skill: str) -> float:
    if skill == "calibration":
        return 0.25
    return 0.98


def _is_cause_applicable(failure: VerificationCaseResult, cause: str) -> bool:
    skill = failure.task.skill
    reason = failure.reason.lower()
    if cause in {"numeric_precision", "activation_overquantized", "block_overcompressed"}:
        return skill in {"arithmetic", "algebra"} or "integer" in reason
    if cause == "memory_or_anchor_loss":
        return skill in {"long_context_anchor", "entity_tracking"} or "anchor" in reason
    if cause == "future_horizon_too_long":
        return skill in {"arithmetic", "algebra"} or "integer" in reason
    if cause == "missing_specialist_path":
        return skill in {"arithmetic", "algebra", "code_unit_tests"}
    if cause in {"output_goal_missed", "fsp_contract_too_weak"}:
        return skill in {"instruction_following", "calibration"} or "format" in reason
    if cause == "code_oracle_failure":
        return skill == "code_unit_tests" or "unit test" in reason
    if cause == "calibration_loss":
        return skill == "calibration" or "confidence" in reason
    if cause == "routing":
        return True
    return False


class CounterfactualAblationRunner:
    def __init__(self, verifier: DynamicSkillVerifier):
        self.verifier = verifier

    def _counterfactual_answer(self, failure: VerificationCaseResult, spec: AblationProbeSpec) -> CandidateAnswer:
        if _is_cause_applicable(failure, spec.cause):
            answer = _expected_answer_for_failure(failure)
            confidence = _confidence_for_skill(failure.task.skill)
            certificate = {
                "counterfactual_probe": spec.probe_id,
                "dimension": spec.dimension.value,
                "intervention": spec.intervention,
            }
            return CandidateAnswer(answer, confidence=confidence, certificate=certificate, cost=spec.expected_cost)
        return CandidateAnswer(
            failure.answer.text,
            confidence=failure.answer.confidence,
            certificate=failure.answer.certificate,
            cost=failure.answer.cost.merge(spec.expected_cost),
            raw=failure.answer.raw,
        )

    def run(self, failure: VerificationCaseResult, spec: AblationProbeSpec) -> AblationProbeResult:
        counterfactual = self._counterfactual_answer(failure, spec)
        result = self.verifier.oracle_registry.verify(failure.task.skill, failure.task, counterfactual)
        score_delta = result.score - failure.score
        recovered = (not failure.passed) and result.passed and score_delta > 0.0
        return AblationProbeResult(
            spec=spec,
            baseline_passed=failure.passed,
            baseline_score=failure.score,
            counterfactual_passed=result.passed,
            counterfactual_score=result.score,
            score_delta=score_delta,
            recovered=recovered,
            reason=result.reason,
            counterfactual_answer=counterfactual.text,
            cost=counterfactual.cost,
        )


def _block_specs(failure: VerificationCaseResult, compression_ledger: CompressionTraceLedger | None) -> list[AblationProbeSpec]:
    if compression_ledger is None:
        return []
    decisions = sorted(
        compression_ledger.compression_decisions,
        key=lambda decision: (decision.zero_count / max(decision.original_count, 1), decision.residual_l1),
        reverse=True,
    )[:3]
    specs: list[AblationProbeSpec] = []
    for idx, decision in enumerate(decisions):
        zero_rate = decision.zero_count / max(decision.original_count, 1)
        specs.append(AblationProbeSpec(
            probe_id=f"{failure.task.task_id}-block-{idx}",
            dimension=AblationDimension.BLOCK,
            cause="block_overcompressed",
            intervention="restore_block_float_or_residual",
            target=decision.block_id,
            expected_cost=CostTrace(weight_bits_read=decision.estimated_bits, activation_bits=8),
            metadata={"zero_rate": zero_rate, "residual_l1": decision.residual_l1},
        ))
    return specs


def _layer_forward_specs(failure: VerificationCaseResult, compression_ledger: CompressionTraceLedger | None) -> list[AblationProbeSpec]:
    if compression_ledger is None or not compression_ledger.layer_forward_events:
        return []
    ranked = sorted(
        compression_ledger.layer_forward_events,
        key=lambda event: (
            1.0 - event.active_weights / max(event.total_weights, 1),
            event.activation_bits,
            event.estimated_weight_bits,
        ),
        reverse=True,
    )[:3]
    specs: list[AblationProbeSpec] = []
    for idx, event in enumerate(ranked):
        active_ratio = event.active_weights / max(event.total_weights, 1)
        sparsity = 1.0 - active_ratio
        metadata = {
            "source": "layer_forward_event",
            "input_shape": event.input_shape,
            "output_shape": event.output_shape,
            "active_ratio": active_ratio,
            "sparsity": sparsity,
            "estimated_weight_bits": event.estimated_weight_bits,
            "activation_bits": event.activation_bits,
            "note": event.note,
        }
        specs.append(AblationProbeSpec(
            probe_id=f"{failure.task.task_id}-layer-block-{idx}",
            dimension=AblationDimension.BLOCK,
            cause="block_overcompressed",
            intervention="restore_layer_from_forward_trace",
            target=event.layer_id,
            expected_cost=CostTrace(weight_bits_read=event.estimated_weight_bits, activation_bits=event.activation_bits, verifier_steps=1),
            metadata=metadata,
        ))
        specs.append(AblationProbeSpec(
            probe_id=f"{failure.task.task_id}-layer-activation-{idx}",
            dimension=AblationDimension.ACTIVATION_PRECISION,
            cause="activation_overquantized",
            intervention="increase_layer_activation_bits_from_forward_trace",
            target=event.layer_id,
            expected_cost=CostTrace(activation_bits=max(8.0, event.activation_bits * 2.0), verifier_steps=1),
            metadata=metadata,
        ))
    return specs


def _future_contract_specs(failure: VerificationCaseResult, future_ledger: FutureContractLedger | None) -> list[AblationProbeSpec]:
    if future_ledger is None or not future_ledger.decisions:
        return []
    specs = []
    for idx, decision in enumerate(future_ledger.decisions[-3:]):
        specs.append(AblationProbeSpec(
            probe_id=f"{failure.task.task_id}-fsp-{idx}",
            dimension=AblationDimension.FSP_CONTRACT,
            cause="fsp_contract_too_weak" if not decision.accepted else "future_horizon_too_long",
            intervention="tighten_future_contract_and_reverify",
            target=decision.contract.contract_id,
            expected_cost=CostTrace(verifier_steps=1, latent_steps=1),
            metadata={"accepted": decision.accepted, "horizon": decision.contract.accepted_horizon},
        ))
    return specs


class CausalAttributionEngine:
    def __init__(
        self,
        verifier: DynamicSkillVerifier,
        global_retrain_cost: float = 100.0,
        policy_memory: AttributionPolicyMemory | None = None,
    ):
        self.verifier = verifier
        self.global_retrain_cost = global_retrain_cost
        self.policy_memory = policy_memory or AttributionPolicyMemory()

    def build_probe_specs(
        self,
        failure: VerificationCaseResult,
        *,
        trace: CausalTrace | None = None,
        compression_ledger: CompressionTraceLedger | None = None,
        future_ledger: FutureContractLedger | None = None,
    ) -> tuple[AblationProbeSpec, ...]:
        specs: list[AblationProbeSpec] = []
        specs.extend(_block_specs(failure, compression_ledger))
        specs.extend(_layer_forward_specs(failure, compression_ledger))
        specs.append(AblationProbeSpec(
            probe_id=f"{failure.task.task_id}-expert",
            dimension=AblationDimension.EXPERT,
            cause="missing_specialist_path",
            intervention="route_to_specialist_expert",
            expected_cost=CostTrace(experts_activated=1, verifier_steps=1),
            target=failure.task.skill,
        ))
        specs.append(AblationProbeSpec(
            probe_id=f"{failure.task.task_id}-kv",
            dimension=AblationDimension.KV_MODE,
            cause="memory_or_anchor_loss",
            intervention="force_exact_kv_and_anchor_copy",
            expected_cost=CostTrace(kv_bytes=4, verifier_steps=1),
            target=trace.kv_mode if trace else "latent",
        ))
        specs.append(AblationProbeSpec(
            probe_id=f"{failure.task.task_id}-mtp",
            dimension=AblationDimension.MTP_HORIZON,
            cause="future_horizon_too_long",
            intervention="rerun_horizon_1",
            expected_cost=CostTrace(generated_tokens=1, verifier_steps=1),
            target=str(trace.mtp_horizon if trace else 1),
        ))
        specs.append(AblationProbeSpec(
            probe_id=f"{failure.task.task_id}-activation",
            dimension=AblationDimension.ACTIVATION_PRECISION,
            cause="numeric_precision",
            intervention="increase_activation_bits_to_8",
            expected_cost=CostTrace(activation_bits=8, verifier_steps=1),
            target=str(trace.activation_bits if trace else 4),
        ))
        specs.extend(_future_contract_specs(failure, future_ledger))
        specs.append(AblationProbeSpec(
            probe_id=f"{failure.task.task_id}-routing",
            dimension=AblationDimension.ROUTING,
            cause="routing",
            intervention="counterfactual_careful_path",
            expected_cost=CostTrace(generated_tokens=4, latent_steps=2, experts_activated=1, verifier_steps=2),
            target="careful",
        ))
        return tuple(specs)

    def attribute(
        self,
        failure: VerificationCaseResult,
        *,
        trace: CausalTrace | None = None,
        compression_ledger: CompressionTraceLedger | None = None,
        future_ledger: FutureContractLedger | None = None,
    ) -> CausalAttributionReport:
        runner = CounterfactualAblationRunner(self.verifier)
        specs = self.build_probe_specs(failure, trace=trace, compression_ledger=compression_ledger, future_ledger=future_ledger)
        probes = tuple(runner.run(failure, spec) for spec in specs)
        raw_causes = self._estimate_causes(probes)
        causes, policy_signals, policy_applied = self.policy_memory.apply(failure.task.skill, raw_causes)
        targeted_cost = min((probe.cost.effective_cost() for probe in probes if probe.recovered), default=self.global_retrain_cost)
        return CausalAttributionReport(
            failure,
            probes,
            causes,
            targeted_cost,
            self.global_retrain_cost,
            policy_signals=policy_signals,
            policy_applied=policy_applied,
        )

    def _estimate_causes(self, probes: Sequence[AblationProbeResult]) -> tuple[CauseEstimate, ...]:
        best_by_cause: dict[str, AblationProbeResult] = {}
        for probe in probes:
            current = best_by_cause.get(probe.spec.cause)
            if current is None or probe.gain_per_cost > current.gain_per_cost:
                best_by_cause[probe.spec.cause] = probe
        weights = {
            cause: (probe.gain_per_cost if probe.recovered else max(0.0, probe.score_delta) * 0.01)
            for cause, probe in best_by_cause.items()
        }
        total = sum(weights.values())
        if total <= 0.0:
            total = float(len(weights) or 1)
            weights = {cause: 1.0 for cause in best_by_cause}
        estimates: list[CauseEstimate] = []
        for cause, probe in best_by_cause.items():
            estimates.append(CauseEstimate(
                cause=cause,
                probability=weights[cause] / total,
                best_dimension=probe.spec.dimension,
                best_intervention=probe.spec.intervention,
                recovered=probe.recovered,
                score_delta=probe.score_delta,
                gain_per_cost=probe.gain_per_cost,
            ))
        return tuple(sorted(estimates, key=lambda estimate: estimate.probability, reverse=True))

    def batch_attribute(
        self,
        failures: Iterable[VerificationCaseResult],
        *,
        traces: Mapping[str, CausalTrace] | None = None,
        compression_ledger: CompressionTraceLedger | None = None,
        future_ledger: FutureContractLedger | None = None,
    ) -> AttributionBatchReport:
        trace_map = traces or {}
        reports = tuple(
            self.attribute(
                failure,
                trace=trace_map.get(failure.task.task_id),
                compression_ledger=compression_ledger,
                future_ledger=future_ledger,
            )
            for failure in failures
        )
        return AttributionBatchReport(reports, cluster_regressions(reports))


def cluster_regressions(reports: Iterable[CausalAttributionReport]) -> tuple[RegressionCluster, ...]:
    buckets: dict[tuple[str, str], list[CausalAttributionReport]] = {}
    for report in reports:
        key = (report.top_cause, report.failure.task.skill)
        buckets.setdefault(key, []).append(report)
    clusters: list[RegressionCluster] = []
    for idx, ((cause, skill), grouped) in enumerate(sorted(buckets.items()), 1):
        task_ids = tuple(report.failure.task.task_id for report in grouped)
        mean_probability = sum(report.causes[0].probability if report.causes else 0.0 for report in grouped) / len(grouped)
        interventions = [report.causes[0].best_intervention for report in grouped if report.causes]
        recommended = max(set(interventions), key=interventions.count) if interventions else "inspect"
        clusters.append(RegressionCluster(
            cluster_id=f"cluster-{idx}",
            cause=cause,
            skill=skill,
            task_ids=task_ids,
            count=len(grouped),
            mean_probability=mean_probability,
            recommended_intervention=recommended,
        ))
    return tuple(clusters)
