from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from cortex3 import VerificationCaseResult
from cortex3_ledgers import CausalTrace


@dataclass(frozen=True)
class CauseHint:
    cause: str
    probability: float
    probe: str
    repair: str


@dataclass(frozen=True)
class FailureAnalysis:
    failure: VerificationCaseResult
    hints: tuple[CauseHint, ...]

    @property
    def top_cause(self) -> str:
        return self.hints[0].cause if self.hints else "unknown"


class RegressionAnalyzer:
    def analyze(self, failure: VerificationCaseResult, trace: CausalTrace | None = None) -> FailureAnalysis:
        weights: Counter[str] = Counter()
        reason = failure.reason.lower()
        skill = failure.task.skill
        if skill == "long_context_anchor" or "anchor" in reason:
            weights["memory_or_anchor_loss"] += 4
            weights["future_horizon_too_long"] += 1
        if skill == "arithmetic" or "integer" in reason:
            weights["numeric_precision"] += 3
            weights["future_horizon_too_long"] += 2
            weights["missing_specialist_path"] += 2
        if skill == "instruction_following" or "format" in reason:
            weights["output_goal_missed"] += 3
            weights["short_check_too_weak"] += 2
        if trace:
            if trace.mtp_horizon > 1:
                weights["future_horizon_too_long"] += trace.mtp_horizon / 2
            if trace.kv_mode != "exact":
                weights["memory_or_anchor_loss"] += 2
            if trace.activation_bits <= 4:
                weights["numeric_precision"] += 1
            if not trace.certificate_fields:
                weights["short_check_too_weak"] += 1
            if trace.uncertainty > 0.4:
                weights["calibration_loss"] += 1
        if not weights:
            weights["unknown_interaction"] += 1
        probe_and_repair = {
            "memory_or_anchor_loss": ("rerun with exact memory and forced anchors", "force_exact_anchor"),
            "future_horizon_too_long": ("rerun with horizon=1", "reduce_mtp_horizon"),
            "numeric_precision": ("rerun numeric path with more activation bits", "increase_activation_bits"),
            "missing_specialist_path": ("route to specialist path", "activate_math_expert"),
            "output_goal_missed": ("check output obligations", "add_goal_check"),
            "short_check_too_weak": ("request stricter short check", "add_short_check"),
            "calibration_loss": ("compare confidence to correctness", "increase_verification_level"),
            "unknown_interaction": ("run careful path", "run_careful_path"),
        }
        total = sum(weights.values())
        hints: list[CauseHint] = []
        for cause, weight in weights.most_common():
            probe, repair = probe_and_repair.get(cause, ("counterfactual rerun", "run_careful_path"))
            hints.append(CauseHint(cause, float(weight) / total, probe, repair))
        return FailureAnalysis(failure, tuple(hints))
