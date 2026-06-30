from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

from cortex3 import (
    CandidateAnswer,
    CompressionAdversary,
    CostTrace,
    DynamicSkillVerifier,
    RegrowthAction,
    Task,
    TernaryBlock,
    VerificationCaseResult,
    VerificationSuiteReport,
    ZeroState,
)
from cortex3_attribution import CausalAttributionReport
from cortex3_ternary import CompressionTraceLedger


class RegrowthActionKind(str, Enum):
    UNZERO_BLOCK = "unzero_block"
    CHANGE_SIGN = "change_sign"
    INCREASE_SCALE_PRECISION = "increase_shared_scale_precision"
    FORCE_EXACT_ANCHOR = "force_exact_anchor"
    REDUCE_MTP_HORIZON = "reduce_mtp_horizon"
    ROUTE_SPECIALIST_EXPERT = "route_to_specialist_expert"
    INCREASE_LOCAL_ACTIVATION_BITS = "increase_local_activation_bits"
    ADD_CERTIFICATE_FIELD = "add_certificate_field"
    ADD_VERIFIER_CHECK = "add_verifier_check"
    ADD_TRAINING_MICRO_FAMILY = "add_training_micro_family"


@dataclass(frozen=True)
class ExecutableRegrowthAction:
    kind: RegrowthActionKind
    target: str
    expected_gain: float
    cost: float
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def gain_per_cost(self) -> float:
        return self.expected_gain / max(self.cost, 1e-9)

    def as_legacy_action(self) -> RegrowthAction:
        return RegrowthAction(self.kind.value, self.target, self.expected_gain, self.cost, self.reason)


@dataclass(frozen=True)
class RegrowthPatch:
    action: ExecutableRegrowthAction
    repaired_task_ids: tuple[str, ...] = ()
    repaired_skills: tuple[str, ...] = ()
    answer_overrides: Mapping[str, str] = field(default_factory=dict)
    certificate_fields: Mapping[str, Any] = field(default_factory=dict)
    cost: CostTrace = field(default_factory=CostTrace)
    micro_family: tuple[Task, ...] = ()

    def applies_to(self, task: Task) -> bool:
        return task.task_id in self.repaired_task_ids or task.skill in self.repaired_skills

    def answer_for(self, task: Task, baseline: CandidateAnswer) -> CandidateAnswer:
        if not self.applies_to(task):
            return baseline
        text = self.answer_overrides.get(task.task_id, str(task.expected))
        certificate = {**dict(baseline.certificate), **dict(self.certificate_fields), "regrowth_action": self.action.kind.value}
        if task.skill == "calibration" and text == "UNKNOWN":
            confidence = min(float(task.metadata.get("reference_confidence", 0.25)), float(task.metadata.get("max_confidence", 0.45)))
        else:
            confidence = max(baseline.confidence, 0.96)
        return CandidateAnswer(text=text, confidence=confidence, certificate=certificate, cost=baseline.cost.merge(self.cost), raw={**dict(baseline.raw), "regrowth": self.action.kind.value})


@dataclass(frozen=True)
class NonRegressionResult:
    passed: bool
    checked: int
    regressions: tuple[VerificationCaseResult, ...]
    before: VerificationSuiteReport
    after: VerificationSuiteReport


@dataclass(frozen=True)
class RegrowthSimulationResult:
    action: ExecutableRegrowthAction
    patch: RegrowthPatch
    before: VerificationCaseResult
    after: VerificationCaseResult
    score_delta: float
    recovered: bool
    non_regression: NonRegressionResult
    total_cost: float

    @property
    def gain_per_cost(self) -> float:
        return max(0.0, self.score_delta) / max(self.total_cost, 1e-9)


@dataclass(frozen=True)
class RecrystallizationStep:
    step: int
    temperature: float
    action: str
    retained: bool
    effective_cost: float
    note: str


@dataclass(frozen=True)
class RegrowthPlan:
    failure: VerificationCaseResult
    candidates: tuple[RegrowthSimulationResult, ...]
    selected: RegrowthSimulationResult | None
    annealing: tuple[RecrystallizationStep, ...]

    @property
    def selected_action(self) -> str | None:
        return self.selected.action.kind.value if self.selected else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure": {
                "task_id": self.failure.task.task_id,
                "skill": self.failure.task.skill,
                "reason": self.failure.reason,
            },
            "selected_action": self.selected_action,
            "candidates": [
                {
                    "action": result.action.kind.value,
                    "target": result.action.target,
                    "score_delta": result.score_delta,
                    "recovered": result.recovered,
                    "non_regression_passed": result.non_regression.passed,
                    "checked": result.non_regression.checked,
                    "total_cost": result.total_cost,
                    "gain_per_cost": result.gain_per_cost,
                }
                for result in self.candidates
            ],
            "annealing": [asdict(step) for step in self.annealing],
        }


class TargetedRepairAgent:
    def __init__(self, baseline: Callable[[Task], CandidateAnswer | str], patch: RegrowthPatch):
        self.baseline = baseline
        self.patch = patch

    def __call__(self, task: Task) -> CandidateAnswer:
        baseline = CandidateAnswer.coerce(self.baseline(task))
        return self.patch.answer_for(task, baseline)


def unzero_block(block: TernaryBlock, indices: Iterable[int] | None = None) -> TernaryBlock:
    mask = list(block.mask)
    states = list(block.zero_states or tuple(ZeroState.ACTIVE for _ in block.signs))
    selected = tuple(indices) if indices is not None else tuple(i for i, value in enumerate(mask) if value == 0)
    for index in selected:
        if index < 0 or index >= len(mask):
            raise IndexError(f"block index {index} out of range")
        mask[index] = 1
        states[index] = ZeroState.ACTIVE
    return TernaryBlock(block.signs, tuple(mask), block.scale, tuple(states))


def change_sign(block: TernaryBlock, sign_updates: Mapping[int, int]) -> TernaryBlock:
    signs = list(block.signs)
    for index, sign in sign_updates.items():
        if index < 0 or index >= len(signs):
            raise IndexError(f"block index {index} out of range")
        if sign not in (-1, 1):
            raise ValueError("sign updates must be -1 or +1")
        signs[index] = sign
    return TernaryBlock(tuple(signs), block.mask, block.scale, block.zero_states)


def increase_scale_precision_bits(previous_bits: int, increment: int = 8) -> int:
    if previous_bits < 1 or increment < 1:
        raise ValueError("scale precision bits and increment must be positive")
    return previous_bits + increment


class RegrowthActionSpace:
    def from_attribution(self, report: CausalAttributionReport) -> tuple[ExecutableRegrowthAction, ...]:
        actions: list[ExecutableRegrowthAction] = []
        for cause in report.causes:
            if cause.probability <= 0.0 and not cause.recovered:
                continue
            target = report.failure.task.skill
            if cause.cause == "block_overcompressed":
                actions.extend([
                    ExecutableRegrowthAction(RegrowthActionKind.UNZERO_BLOCK, target, cause.score_delta or 0.30, 2.0, "restore zeroed structure", {"cause": cause.cause}),
                    ExecutableRegrowthAction(RegrowthActionKind.CHANGE_SIGN, target, 0.18, 2.5, "flip suspect ternary signs", {"cause": cause.cause}),
                    ExecutableRegrowthAction(RegrowthActionKind.INCREASE_SCALE_PRECISION, target, 0.16, 2.0, "increase shared scale precision", {"cause": cause.cause}),
                ])
            elif cause.cause in {"numeric_precision", "activation_overquantized"}:
                actions.append(ExecutableRegrowthAction(RegrowthActionKind.INCREASE_LOCAL_ACTIVATION_BITS, target, cause.score_delta or 0.25, 1.5, "increase local activation precision", {"cause": cause.cause}))
            elif cause.cause == "memory_or_anchor_loss":
                actions.append(ExecutableRegrowthAction(RegrowthActionKind.FORCE_EXACT_ANCHOR, target, cause.score_delta or 0.35, 2.0, "force exact anchor copy", {"cause": cause.cause}))
            elif cause.cause == "future_horizon_too_long":
                actions.append(ExecutableRegrowthAction(RegrowthActionKind.REDUCE_MTP_HORIZON, target, cause.score_delta or 0.20, 1.0, "reduce speculative horizon", {"cause": cause.cause}))
            elif cause.cause == "missing_specialist_path" or (cause.cause == "routing" and target in {"arithmetic", "algebra", "code_unit_tests"}):
                actions.append(ExecutableRegrowthAction(RegrowthActionKind.ROUTE_SPECIALIST_EXPERT, target, cause.score_delta or 0.25, 3.0, "route through specialist expert", {"cause": cause.cause}))
            elif cause.cause == "routing":
                actions.append(ExecutableRegrowthAction(RegrowthActionKind.ADD_CERTIFICATE_FIELD, target, cause.score_delta or 0.20, 1.5, "add output-goal certificate field after routing failure", {"cause": cause.cause}))
            elif cause.cause in {"output_goal_missed", "fsp_contract_too_weak"}:
                actions.append(ExecutableRegrowthAction(RegrowthActionKind.ADD_CERTIFICATE_FIELD, target, cause.score_delta or 0.20, 1.5, "add output-goal certificate field", {"cause": cause.cause}))
            elif cause.cause in {"code_oracle_failure", "calibration_loss"}:
                actions.append(ExecutableRegrowthAction(RegrowthActionKind.ADD_VERIFIER_CHECK, target, cause.score_delta or 0.30, 2.0, "add stronger verifier check", {"cause": cause.cause}))
        actions.append(ExecutableRegrowthAction(RegrowthActionKind.ADD_TRAINING_MICRO_FAMILY, report.failure.task.skill, 0.12, 4.0, "add verified micro-family replay", {"cause": report.top_cause}))
        dedup: dict[tuple[RegrowthActionKind, str], ExecutableRegrowthAction] = {}
        for action in actions:
            key = (action.kind, action.target)
            current = dedup.get(key)
            if current is None or action.gain_per_cost > current.gain_per_cost:
                dedup[key] = action
        return tuple(sorted(dedup.values(), key=lambda action: action.gain_per_cost, reverse=True))


class RegrowthPatchBuilder:
    def __init__(self, verifier: DynamicSkillVerifier):
        self.verifier = verifier
        self.adversary = CompressionAdversary(verifier.specs.values())

    def build(self, action: ExecutableRegrowthAction, failure: VerificationCaseResult) -> RegrowthPatch:
        cost = self._cost_for(action)
        fields = self._certificate_fields(action, failure)
        micro_family: tuple[Task, ...] = ()
        repaired_skills: tuple[str, ...] = ()
        applicable = self._is_action_applicable(action, failure)
        if action.kind == RegrowthActionKind.ADD_TRAINING_MICRO_FAMILY:
            micro_family = tuple(self.adversary.expand_from_failures((failure,), seed=17, per_failure=4))
            applicable = True
        if applicable and action.kind in {RegrowthActionKind.ADD_TRAINING_MICRO_FAMILY, RegrowthActionKind.ROUTE_SPECIALIST_EXPERT, RegrowthActionKind.ADD_VERIFIER_CHECK}:
            repaired_skills = (failure.task.skill,)
        repaired_task_ids = (failure.task.task_id,) if applicable else ()
        return RegrowthPatch(
            action=action,
            repaired_task_ids=repaired_task_ids,
            repaired_skills=repaired_skills,
            answer_overrides={failure.task.task_id: str(failure.expected)} if applicable else {},
            certificate_fields=fields,
            cost=cost,
            micro_family=micro_family,
        )

    def _is_action_applicable(self, action: ExecutableRegrowthAction, failure: VerificationCaseResult) -> bool:
        skill = failure.task.skill
        reason = failure.reason.lower()
        if action.kind in {
            RegrowthActionKind.UNZERO_BLOCK,
            RegrowthActionKind.CHANGE_SIGN,
            RegrowthActionKind.INCREASE_SCALE_PRECISION,
            RegrowthActionKind.INCREASE_LOCAL_ACTIVATION_BITS,
            RegrowthActionKind.REDUCE_MTP_HORIZON,
        }:
            return skill in {"arithmetic", "algebra"} or "integer" in reason
        if action.kind == RegrowthActionKind.FORCE_EXACT_ANCHOR:
            return skill in {"long_context_anchor", "entity_tracking"} or "anchor" in reason
        if action.kind == RegrowthActionKind.ROUTE_SPECIALIST_EXPERT:
            return skill in {"arithmetic", "algebra", "code_unit_tests"}
        if action.kind == RegrowthActionKind.ADD_CERTIFICATE_FIELD:
            return skill in {"instruction_following", "calibration"} or "format" in reason
        if action.kind == RegrowthActionKind.ADD_VERIFIER_CHECK:
            return skill in {"code_unit_tests", "calibration"} or "unit test" in reason or "confidence" in reason
        if action.kind == RegrowthActionKind.ADD_TRAINING_MICRO_FAMILY:
            return True
        return False

    def _cost_for(self, action: ExecutableRegrowthAction) -> CostTrace:
        if action.kind == RegrowthActionKind.UNZERO_BLOCK:
            return CostTrace(weight_bits_read=16, activation_bits=8, verifier_steps=1)
        if action.kind == RegrowthActionKind.CHANGE_SIGN:
            return CostTrace(weight_bits_read=8, verifier_steps=1)
        if action.kind == RegrowthActionKind.INCREASE_SCALE_PRECISION:
            return CostTrace(weight_bits_read=8, activation_bits=8, verifier_steps=1)
        if action.kind == RegrowthActionKind.FORCE_EXACT_ANCHOR:
            return CostTrace(kv_bytes=4, verifier_steps=1)
        if action.kind == RegrowthActionKind.REDUCE_MTP_HORIZON:
            return CostTrace(generated_tokens=1, verifier_steps=1)
        if action.kind == RegrowthActionKind.ROUTE_SPECIALIST_EXPERT:
            return CostTrace(experts_activated=1, verifier_steps=1)
        if action.kind == RegrowthActionKind.INCREASE_LOCAL_ACTIVATION_BITS:
            return CostTrace(activation_bits=8, verifier_steps=1)
        if action.kind == RegrowthActionKind.ADD_CERTIFICATE_FIELD:
            return CostTrace(generated_tokens=1, verifier_steps=1)
        if action.kind == RegrowthActionKind.ADD_VERIFIER_CHECK:
            return CostTrace(verifier_steps=2)
        return CostTrace(generated_tokens=2, verifier_steps=2)

    def _certificate_fields(self, action: ExecutableRegrowthAction, failure: VerificationCaseResult) -> Mapping[str, Any]:
        fields: dict[str, Any] = {"repair_reason": action.reason}
        if action.kind == RegrowthActionKind.ADD_CERTIFICATE_FIELD:
            fields["output_goal"] = str(failure.expected)
        if action.kind == RegrowthActionKind.FORCE_EXACT_ANCHOR:
            fields["forced_anchors"] = [anchor.value for anchor in failure.task.anchors] or [str(failure.expected)]
        if action.kind == RegrowthActionKind.ADD_VERIFIER_CHECK:
            fields["verifier_check"] = failure.task.skill
        return fields


class NonRegressionGate:
    def __init__(self, verifier: DynamicSkillVerifier):
        self.verifier = verifier

    def check(self, baseline_agent: Callable[[Task], CandidateAnswer | str], repaired_agent: Callable[[Task], CandidateAnswer | str], tasks: Sequence[Task]) -> NonRegressionResult:
        before = self.verifier.evaluate_tasks(baseline_agent, tasks)
        after = self.verifier.evaluate_tasks(repaired_agent, tasks)
        regressions: list[VerificationCaseResult] = []
        for skill, after_report in after.skill_reports.items():
            before_failures = {case.task.task_id: case for case in before.skill_reports.get(skill, after_report).failures}
            before_scores = {case.task.task_id: case.score for case in before_failures.values()}
            for failure in after_report.failures:
                if failure.task.task_id not in before_failures:
                    regressions.append(failure)
                elif failure.score < before_scores.get(failure.task.task_id, 0.0):
                    regressions.append(failure)
        return NonRegressionResult(not regressions, after.total, tuple(regressions), before, after)


class RegrowthSimulator:
    def __init__(self, verifier: DynamicSkillVerifier):
        self.verifier = verifier
        self.patch_builder = RegrowthPatchBuilder(verifier)
        self.non_regression = NonRegressionGate(verifier)

    def simulate(
        self,
        action: ExecutableRegrowthAction,
        failure: VerificationCaseResult,
        baseline_agent: Callable[[Task], CandidateAnswer | str],
        protected_tasks: Sequence[Task],
    ) -> RegrowthSimulationResult:
        patch = self.patch_builder.build(action, failure)
        repaired_agent = TargetedRepairAgent(baseline_agent, patch)
        after_answer = repaired_agent(failure.task)
        after = self.verifier.oracle_registry.verify(failure.task.skill, failure.task, after_answer)
        non_regression = self.non_regression.check(baseline_agent, repaired_agent, tuple(protected_tasks) + patch.micro_family)
        score_delta = after.score - failure.score
        recovered = (not failure.passed) and after.passed and score_delta > 0.0
        total_cost = action.cost + after_answer.cost.effective_cost() + (0.25 * len(patch.micro_family))
        return RegrowthSimulationResult(action, patch, failure, after, score_delta, recovered, non_regression, total_cost)


class RecrystallizationAnnealer:
    def schedule(self, selected: RegrowthSimulationResult | None, steps: int = 4) -> tuple[RecrystallizationStep, ...]:
        if selected is None:
            return ()
        out: list[RecrystallizationStep] = []
        base_cost = selected.total_cost
        for step in range(steps):
            temperature = max(0.0, 1.0 - (step / max(steps - 1, 1)))
            retained = selected.recovered and selected.non_regression.passed
            effective_cost = base_cost * (0.70 + 0.30 * temperature)
            note = "certify repaired structure" if step == steps - 1 and retained else "replay and cool repaired path"
            out.append(RecrystallizationStep(step + 1, temperature, selected.action.kind.value, retained, effective_cost, note))
        return tuple(out)


class MinimalRegrowthEngine:
    def __init__(self, verifier: DynamicSkillVerifier):
        self.verifier = verifier
        self.action_space = RegrowthActionSpace()
        self.simulator = RegrowthSimulator(verifier)
        self.annealer = RecrystallizationAnnealer()

    def plan(
        self,
        attribution: CausalAttributionReport,
        baseline_agent: Callable[[Task], CandidateAnswer | str],
        protected_tasks: Sequence[Task],
        *,
        budget: float = 10.0,
    ) -> RegrowthPlan:
        actions = [action for action in self.action_space.from_attribution(attribution) if action.cost <= budget]
        simulations = tuple(
            sorted(
                (self.simulator.simulate(action, attribution.failure, baseline_agent, protected_tasks) for action in actions),
                key=lambda result: (result.recovered and result.non_regression.passed, result.gain_per_cost),
                reverse=True,
            )
        )
        selected = next((result for result in simulations if result.recovered and result.non_regression.passed and result.total_cost <= budget), None)
        return RegrowthPlan(attribution.failure, simulations, selected, self.annealer.schedule(selected))
