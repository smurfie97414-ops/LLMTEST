from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

from cortex3 import CostTrace, VerificationSuiteReport


@dataclass
class BitLedger:
    weight_bits: float = 0.0
    scale_bits: float = 0.0
    activation_bits: float = 0.0
    kv_bytes: float = 0.0
    routing_bits: float = 0.0
    certificate_bits: float = 0.0
    verifier_steps: int = 0
    notes: list[str] = field(default_factory=list)

    def ingest_cost(self, cost: CostTrace, note: str = "") -> None:
        self.weight_bits += cost.weight_bits_read
        self.activation_bits += cost.activation_bits
        self.kv_bytes += cost.kv_bytes
        self.verifier_steps += cost.verifier_steps
        if note:
            self.notes.append(note)

    def add_certificate(self, certificate: Mapping[str, Any]) -> None:
        encoded = json.dumps(dict(certificate), ensure_ascii=False, sort_keys=True)
        self.certificate_bits += len(encoded.encode("utf-8")) * 8

    @property
    def total_effective_bits(self) -> float:
        return (
            self.weight_bits
            + self.scale_bits
            + self.activation_bits
            + self.kv_bytes * 8
            + self.routing_bits
            + self.certificate_bits
            + self.verifier_steps * 24
        )


@dataclass
class SkillState:
    skill: str
    score: float = 0.0
    pass_rate: float = 0.0
    failures: int = 0
    fragility: float = 0.0
    protected: bool = False
    history: list[float] = field(default_factory=list)


@dataclass
class SkillLedger:
    states: dict[str, SkillState] = field(default_factory=dict)
    protected_threshold: float = 0.75

    def update_from_report(self, report: VerificationSuiteReport) -> None:
        for skill, skill_report in report.skill_reports.items():
            state = self.states.get(skill, SkillState(skill))
            previous = state.score
            state.score = skill_report.score
            state.pass_rate = skill_report.pass_rate
            state.failures = len(skill_report.failures)
            state.history.append(skill_report.score)
            regression = max(0.0, previous - skill_report.score) if len(state.history) > 1 else 0.0
            state.fragility = min(1.0, 0.55 * (1.0 - state.pass_rate) + 0.45 * regression)
            state.protected = state.score < self.protected_threshold or state.fragility > 0.25
            self.states[skill] = state

    def fragile_skills(self) -> list[SkillState]:
        return sorted((s for s in self.states.values() if s.protected), key=lambda s: s.fragility, reverse=True)


@dataclass(frozen=True)
class CausalTrace:
    task_id: str
    skill: str
    mtp_horizon: int = 1
    activation_bits: int = 8
    kv_mode: str = "exact"
    verifier_level: int = 0
    certificate_fields: tuple[str, ...] = ()
    uncertainty: float = 0.0


@dataclass
class CausalLedger:
    traces: dict[str, CausalTrace] = field(default_factory=dict)

    def record(self, trace: CausalTrace) -> None:
        self.traces[trace.task_id] = trace

    def get(self, task_id: str) -> CausalTrace | None:
        return self.traces.get(task_id)


@dataclass
class UncertaintyLedger:
    bins: dict[str, list[tuple[float, bool]]] = field(default_factory=lambda: defaultdict(list))

    def record(self, skill: str, confidence: float, passed: bool) -> None:
        self.bins[skill].append((max(0.0, min(1.0, confidence)), passed))

    def expected_calibration_error(self, skill: str | None = None, n_bins: int = 10) -> float:
        pairs: list[tuple[float, bool]] = []
        if skill is None:
            for values in self.bins.values():
                pairs.extend(values)
        else:
            pairs.extend(self.bins.get(skill, []))
        if not pairs:
            return 0.0
        total = len(pairs)
        ece = 0.0
        for b in range(n_bins):
            lo, hi = b / n_bins, (b + 1) / n_bins
            bucket = [(c, p) for c, p in pairs if lo <= c < hi or (b == n_bins - 1 and c == 1.0)]
            if not bucket:
                continue
            conf = sum(c for c, _ in bucket) / len(bucket)
            acc = sum(1.0 for _, p in bucket if p) / len(bucket)
            ece += (len(bucket) / total) * abs(conf - acc)
        return ece
