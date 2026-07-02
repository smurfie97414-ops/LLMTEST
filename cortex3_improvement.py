from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from cortex3 import (
    Anchor,
    CandidateAnswer,
    CostTrace,
    CorruptedCompressedAgent,
    DynamicSkillVerifier,
    ReferenceRuleAgent,
    SkillReport,
    Task,
    VerificationCaseResult,
    VerificationSuiteReport,
)
from cortex3_cycle import CycleReport
from cortex3_ledgers import UncertaintyLedger
from cortex3_selection import TrialProposal, TrialSelector


Agent = Callable[[Task], CandidateAnswer | str]
PERSISTENT_IMPROVEMENT_ARCHIVE_SCHEMA_VERSION = 1


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
    return path


def _require_mapping(payload: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be a persisted mapping")
    return payload


def _require_keys(payload: Mapping[str, Any], keys: Iterable[str], *, context: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{context} missing required fields: {', '.join(missing)}")


def _cost_to_dict(cost: CostTrace) -> dict[str, Any]:
    return asdict(cost)


def _cost_from_dict(payload: Mapping[str, Any] | None) -> CostTrace:
    data = dict(payload or {})
    return CostTrace(
        weight_bits_read=float(data.get("weight_bits_read", 0.0)),
        activation_bits=float(data.get("activation_bits", 0.0)),
        kv_bytes=float(data.get("kv_bytes", 0.0)),
        generated_tokens=int(data.get("generated_tokens", 0)),
        latent_steps=int(data.get("latent_steps", 0)),
        experts_activated=int(data.get("experts_activated", 0)),
        verifier_steps=int(data.get("verifier_steps", 0)),
        wall_time_ms=float(data.get("wall_time_ms", 0.0)),
    )


def _anchor_to_dict(anchor: Anchor) -> dict[str, Any]:
    return asdict(anchor)


def _anchor_from_dict(payload: Mapping[str, Any]) -> Anchor:
    data = dict(payload)
    return Anchor(
        kind=str(data.get("kind", "")),
        value=str(data.get("value", "")),
        source_id=str(data.get("source_id", "")),
        importance=float(data.get("importance", 1.0)),
    )


def _task_to_dict(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "skill": task.skill,
        "prompt": task.prompt,
        "expected": task.expected,
        "metadata": dict(task.metadata),
        "anchors": [_anchor_to_dict(anchor) for anchor in task.anchors],
        "group_id": task.group_id,
    }


def _task_from_dict(payload: Mapping[str, Any]) -> Task:
    data = dict(payload)
    return Task(
        task_id=str(data.get("task_id", "")),
        skill=str(data.get("skill", "")),
        prompt=str(data.get("prompt", "")),
        expected=data.get("expected"),
        metadata=dict(data.get("metadata") or {}),
        anchors=tuple(_anchor_from_dict(item) for item in data.get("anchors", ())),
        group_id=None if data.get("group_id") is None else str(data.get("group_id")),
    )


def _candidate_to_dict(answer: CandidateAnswer) -> dict[str, Any]:
    return {
        "text": answer.text,
        "confidence": float(answer.confidence),
        "certificate": dict(answer.certificate),
        "cost": _cost_to_dict(answer.cost),
        "raw": dict(answer.raw),
    }


def _candidate_from_dict(payload: Mapping[str, Any]) -> CandidateAnswer:
    data = dict(payload)
    return CandidateAnswer(
        text=str(data.get("text", "")),
        confidence=float(data.get("confidence", 0.0)),
        certificate=dict(data.get("certificate") or {}),
        cost=_cost_from_dict(data.get("cost")),
        raw=dict(data.get("raw") or {}),
    )


def _case_to_dict(case: VerificationCaseResult) -> dict[str, Any]:
    return {
        "task": _task_to_dict(case.task),
        "passed": bool(case.passed),
        "score": float(case.score),
        "answer": _candidate_to_dict(case.answer),
        "expected": case.expected,
        "reason": case.reason,
        "verifier_cost": _cost_to_dict(case.verifier_cost),
    }


def _case_from_dict(payload: Mapping[str, Any]) -> VerificationCaseResult:
    data = dict(payload)
    task = _task_from_dict(dict(data.get("task") or {}))
    return VerificationCaseResult(
        task=task,
        passed=bool(data.get("passed", False)),
        score=float(data.get("score", 0.0)),
        answer=_candidate_from_dict(dict(data.get("answer") or {})),
        expected=data.get("expected", task.expected),
        reason=str(data.get("reason", "")),
        verifier_cost=_cost_from_dict(data.get("verifier_cost")),
    )


def _skill_report_to_dict(report: SkillReport) -> dict[str, Any]:
    return {
        "skill": report.skill,
        "total": int(report.total),
        "passed": int(report.passed),
        "score": float(report.score),
        "failures": [_case_to_dict(case) for case in report.failures],
        "cases": [_case_to_dict(case) for case in report.cases],
    }


def _skill_report_from_dict(payload: Mapping[str, Any]) -> SkillReport:
    data = dict(payload)
    return SkillReport(
        skill=str(data.get("skill", "")),
        total=int(data.get("total", 0)),
        passed=int(data.get("passed", 0)),
        score=float(data.get("score", 0.0)),
        failures=tuple(_case_from_dict(item) for item in data.get("failures", ())),
        cases=tuple(_case_from_dict(item) for item in data.get("cases", ())),
    )


def _suite_report_to_dict(report: VerificationSuiteReport) -> dict[str, Any]:
    return {
        "skill_reports": {
            skill: _skill_report_to_dict(skill_report)
            for skill, skill_report in report.skill_reports.items()
        },
        "total": int(report.total),
        "passed": int(report.passed),
        "aggregate_score": float(report.aggregate_score),
        "total_cost": _cost_to_dict(report.total_cost),
    }


def _suite_report_from_dict(payload: Mapping[str, Any], *, context: str = "verification suite report") -> VerificationSuiteReport:
    data = dict(_require_mapping(payload, context=context))
    _require_keys(
        data,
        ("skill_reports", "total", "passed", "aggregate_score", "total_cost"),
        context=context,
    )
    skill_reports = {
        str(skill): _skill_report_from_dict(dict(skill_report))
        for skill, skill_report in dict(data["skill_reports"] or {}).items()
    }
    total = int(data["total"])
    passed = int(data["passed"])
    if total <= 0 or not skill_reports:
        raise ValueError(f"{context} must contain full non-empty verifier cases")
    if passed < 0 or passed > total:
        raise ValueError(f"{context} has invalid passed/total counts")
    return VerificationSuiteReport(
        skill_reports=skill_reports,
        total=total,
        passed=passed,
        aggregate_score=float(data["aggregate_score"]),
        total_cost=_cost_from_dict(data["total_cost"]),
    )


class ProposalKind(str, Enum):
    SKILL_SPEC = "skill_spec"
    TEST = "test"
    COMPRESSION = "compression"
    ROUTER = "router"
    MTP_HEAD = "mtp_head"
    REGROWTH_STRATEGY = "regrowth_strategy"
    HARDWARE_GRAMMAR = "hardware_grammar"
    KERNEL = "kernel"
    COMPILED_FRONTIER = "compiled_frontier"


EVOLUTION_KIND_ORDER = (
    ProposalKind.COMPILED_FRONTIER,
    ProposalKind.REGROWTH_STRATEGY,
    ProposalKind.MTP_HEAD,
    ProposalKind.ROUTER,
    ProposalKind.COMPRESSION,
    ProposalKind.TEST,
    ProposalKind.SKILL_SPEC,
    ProposalKind.KERNEL,
    ProposalKind.HARDWARE_GRAMMAR,
)


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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImprovementProposal":
        data = dict(payload)
        return cls(
            proposal_id=str(data.get("proposal_id", "")),
            title=str(data.get("title", "")),
            kind=ProposalKind(str(data.get("kind", ProposalKind.TEST.value))),
            affected_skills=tuple(str(item) for item in data.get("affected_skills", ())),
            expected_quality_delta=float(data.get("expected_quality_delta", 0.0)),
            expected_cost_delta=float(data.get("expected_cost_delta", 0.0)),
            expected_robustness_delta=float(data.get("expected_robustness_delta", 0.0)),
            risk=float(data.get("risk", 0.0)),
            diversity_tags=tuple(str(item) for item in data.get("diversity_tags", ())),
            patch_payload=dict(data.get("patch_payload") or {}),
            parent_ids=tuple(str(item) for item in data.get("parent_ids", ())),
        )


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

    def from_frontier_repairs(
        self,
        repairs: Iterable[Mapping[str, Any]],
        *,
        max_proposals: int = 4,
    ) -> tuple[ImprovementProposal, ...]:
        proposals: list[ImprovementProposal] = []
        seen: set[str] = set()
        for repair in repairs:
            payload = dict(repair)
            if not bool(payload.get("accepted")):
                continue
            task_id = str(payload.get("task_id") or "")
            skill = str(payload.get("skill") or payload.get("circuit_skill") or "")
            if not task_id or not skill:
                continue
            proposal_id = f"frontier-repair-{skill}-{task_id}"
            if proposal_id in seen:
                continue
            seen.add(proposal_id)
            score_delta = max(0.01, float(payload.get("repair_score_delta", 0.0)))
            proposals.append(ImprovementProposal(
                proposal_id=proposal_id,
                title=f"compile accepted frontier repair for {skill}",
                kind=ProposalKind.COMPILED_FRONTIER,
                affected_skills=(skill,),
                expected_quality_delta=score_delta,
                expected_cost_delta=-max(0.01, score_delta * 0.25),
                expected_robustness_delta=max(0.01, score_delta * 0.50),
                risk=0.05,
                diversity_tags=("compiled_frontier", skill, task_id),
                patch_payload={
                    "action": "compile_frontier_repair",
                    "mode": "repair_skill",
                    "target": skill,
                    "task_id": task_id,
                    "source_failure_ids": tuple(str(item) for item in payload.get("source_failure_ids", ())),
                    "frontier_task_ids": tuple(str(item) for item in payload.get("frontier_task_ids", ())),
                    "repair_score_delta": float(payload.get("repair_score_delta", 0.0)),
                    "protected_checked": int(payload.get("protected_checked", 0)),
                    "frontier_compiled_verified": bool(payload.get("frontier_compiled_verified")),
                    "frontier_heldout_gate_passed": bool(payload.get("frontier_heldout_gate_passed")),
                    "frontier_heldout_passed": int(payload.get("frontier_heldout_passed", 0) or 0),
                    "frontier_heldout_total": int(payload.get("frontier_heldout_total", 0) or 0),
                    "frontier_heldout_pass_rate": float(payload.get("frontier_heldout_pass_rate", 0.0) or 0.0),
                    "frontier_output_goal_contract_passed": bool(payload.get("frontier_output_goal_contract_passed")),
                    "frontier_output_goal_contract": dict(payload.get("frontier_output_goal_contract") or {}),
                    "frontier_compiled_contract_verified": bool(payload.get("frontier_compiled_contract_verified")),
                    "frontier_compiled_contract_checksum": str(payload.get("frontier_compiled_contract_checksum", "")),
                },
            ))
            if len(proposals) >= max_proposals:
                break
        return tuple(proposals)

    def from_sleep_frontier_circuits(
        self,
        circuits: Iterable[Mapping[str, Any]],
        *,
        max_proposals: int = 4,
    ) -> tuple[ImprovementProposal, ...]:
        proposals: list[ImprovementProposal] = []
        seen: set[str] = set()
        for circuit in circuits:
            payload = dict(circuit)
            if not bool(payload.get("passed")):
                continue
            training = dict(payload.get("training") or {})
            if str(training.get("source_kind", "")) != "sleep_consolidation":
                continue
            if not (
                bool(training.get("sleep_filter_accepted", False))
                and bool(training.get("sleep_diversity_ok", False))
                and bool(training.get("sleep_calibration_ok", False))
            ):
                continue
            heldout = dict(payload.get("heldout") or {})
            heldout_total = int(heldout.get("total", 0) or 0)
            heldout_passed = int(heldout.get("passed", 0) or 0)
            if heldout_total <= 0 or heldout_passed < heldout_total or not bool(heldout.get("gate_passed", False)):
                continue
            skill = str(payload.get("skill") or "")
            task_ids = tuple(str(item) for item in payload.get("frontier_task_ids", ()))
            if not skill or not task_ids:
                continue
            proposal_id = f"sleep-frontier-{skill}-{task_ids[0]}"
            if proposal_id in seen:
                continue
            seen.add(proposal_id)
            dsv = dict(payload.get("dsv") or {})
            quality_delta = max(0.02, float(heldout.get("aggregate_score", 0.0) or 0.0) * 0.05)
            proposals.append(ImprovementProposal(
                proposal_id=proposal_id,
                title=f"promote sleep consolidation circuit for {skill}",
                kind=ProposalKind.COMPILED_FRONTIER,
                affected_skills=(skill,),
                expected_quality_delta=quality_delta,
                expected_cost_delta=-max(0.01, quality_delta * 0.20),
                expected_robustness_delta=max(0.02, float(heldout.get("pass_rate", 0.0) or 0.0) * 0.05),
                risk=0.04,
                diversity_tags=("compiled_frontier", "sleep_consolidation", skill),
                patch_payload={
                    "action": "compile_sleep_consolidation_frontier",
                    "mode": "repair_skill",
                    "target": skill,
                    "task_id": task_ids[0],
                    "source_failure_ids": tuple(str(item) for item in payload.get("source_failure_ids", ())),
                    "frontier_task_ids": task_ids,
                    "heldout_task_ids": tuple(str(item) for item in payload.get("heldout_task_ids", ())),
                    "sleep_source_example_ids": tuple(str(item) for item in training.get("sleep_source_example_ids", ())),
                    "sleep_source_origins": tuple(str(item) for item in training.get("sleep_source_origins", ())),
                    "sleep_source_synthetic_count": int(training.get("sleep_source_synthetic_count", 0) or 0),
                    "sleep_source_real_count": int(training.get("sleep_source_real_count", 0) or 0),
                    "sleep_source_max_contamination_risk": float(training.get("sleep_source_max_contamination_risk", 0.0) or 0.0),
                    "sleep_source_min_verification_level": int(training.get("sleep_source_min_verification_level", 0) or 0),
                    "sleep_filter_accepted": bool(training.get("sleep_filter_accepted", False)),
                    "sleep_filter_reasons": tuple(str(item) for item in training.get("sleep_filter_reasons", ())),
                    "sleep_filter_metrics": dict(training.get("sleep_filter_metrics") or {}),
                    "sleep_diversity_ok": bool(training.get("sleep_diversity_ok", False)),
                    "sleep_calibration_ok": bool(training.get("sleep_calibration_ok", False)),
                    "sleep_rare_skill_gain": float(training.get("sleep_rare_skill_gain", 0.0) or 0.0),
                    "sleep_calibration_gap_delta": float(training.get("sleep_calibration_gap_delta", 0.0) or 0.0),
                    "sleep_accepted_examples": int(training.get("sleep_accepted_examples", 0) or 0),
                    "sleep_support_examples": int(training.get("sleep_support_examples", 0) or 0),
                    "frontier_compiled_verified": True,
                    "frontier_heldout_gate_passed": True,
                    "frontier_heldout_passed": heldout_passed,
                    "frontier_heldout_total": heldout_total,
                    "frontier_heldout_pass_rate": float(heldout.get("pass_rate", 0.0) or 0.0),
                    "frontier_dsv_passed": int(dsv.get("passed", 0) or 0),
                    "frontier_dsv_total": int(dsv.get("total", 0) or 0),
                    "source_kind": "sleep_consolidation",
                },
            ))
            if len(proposals) >= max_proposals:
                break
        return tuple(proposals)

    def evolve_from_archive(
        self,
        archive: "EvolutionaryArchive",
        *,
        max_proposals: int = 4,
        generation_index: int = 1,
        seen_ids: Iterable[str] = (),
    ) -> tuple[ImprovementProposal, ...]:
        if max_proposals <= 0:
            return tuple()
        seen = set(str(item) for item in seen_ids)
        counts = archive.accepted_kind_counts()
        accepted = tuple(reversed(archive.accepted))
        proposals: list[ImprovementProposal] = []
        projected_counts = Counter(counts)
        for parent_index, record in enumerate(accepted):
            parent = record.proposal
            if not parent.affected_skills:
                continue
            compatible_kinds = self._compatible_evolution_kinds(parent)
            kind = self._least_represented_kind(projected_counts, parent.kind, compatible_kinds)
            skill = parent.affected_skills[0]
            proposal_id = f"evolve-g{generation_index}-{kind.value}-{parent.proposal_id}"
            if proposal_id in seen:
                continue
            seen.add(proposal_id)
            projected_counts[kind] += 1
            parent_payload = dict(parent.patch_payload)
            parent_action = str(parent_payload.get("action", parent.kind.value))
            proposals.append(ImprovementProposal(
                proposal_id=proposal_id,
                title=f"evolve {parent.proposal_id} into {kind.value} for {skill}",
                kind=kind,
                affected_skills=parent.affected_skills,
                expected_quality_delta=max(0.01, parent.expected_quality_delta * 0.75 + 0.02),
                expected_cost_delta=min(parent.expected_cost_delta, -0.01),
                expected_robustness_delta=max(0.01, parent.expected_robustness_delta * 0.75 + 0.01),
                risk=min(0.14, max(0.02, parent.risk * 0.70 + 0.01)),
                diversity_tags=(
                    "evolved",
                    f"generation_{generation_index}",
                    kind.value,
                    skill,
                    parent.kind.value,
                ),
                patch_payload={
                    **parent_payload,
                    "action": f"evolve_{kind.value}_{parent_action}",
                    "parent_action": parent_action,
                    "parent_kind": parent.kind.value,
                    "parent_proposal_id": parent.proposal_id,
                    "mode": "repair_skill",
                    "target": skill,
                    "generation": generation_index,
                    "diversity_pressure_kind_counts": {
                        proposal_kind.value: int(count)
                        for proposal_kind, count in counts.items()
                    },
                },
                parent_ids=tuple(dict.fromkeys((*parent.parent_ids, parent.proposal_id))),
            ))
            if len(proposals) >= max_proposals:
                break
        return tuple(proposals)

    def _compatible_evolution_kinds(self, parent: ImprovementProposal) -> tuple[ProposalKind, ...]:
        payload = dict(parent.patch_payload)
        action = str(payload.get("action", parent.kind.value)).lower()
        frontier_keys = {
            "frontier_task_ids",
            "frontier_task_id",
            "source_failure_ids",
            "source_task_id",
            "compiled_circuit_id",
            "heldout_task_ids",
            "sleep_source_example_ids",
        }
        if parent.kind == ProposalKind.COMPILED_FRONTIER or any(key in payload for key in frontier_keys):
            return (
                ProposalKind.COMPILED_FRONTIER,
                ProposalKind.REGROWTH_STRATEGY,
                ProposalKind.ROUTER,
                ProposalKind.TEST,
                ProposalKind.SKILL_SPEC,
            )
        if parent.kind in {ProposalKind.KERNEL, ProposalKind.HARDWARE_GRAMMAR} or any(
            token in action
            for token in ("kernel", "cuda", "ternary", "bitlinear", "wmma", "int2", "pack")
        ):
            return (
                ProposalKind.KERNEL,
                ProposalKind.HARDWARE_GRAMMAR,
                ProposalKind.COMPRESSION,
                ProposalKind.REGROWTH_STRATEGY,
                ProposalKind.TEST,
            )
        if parent.kind == ProposalKind.MTP_HEAD:
            return (
                ProposalKind.MTP_HEAD,
                ProposalKind.ROUTER,
                ProposalKind.TEST,
                ProposalKind.SKILL_SPEC,
            )
        if parent.kind == ProposalKind.COMPRESSION:
            return (
                ProposalKind.COMPRESSION,
                ProposalKind.ROUTER,
                ProposalKind.REGROWTH_STRATEGY,
                ProposalKind.TEST,
                ProposalKind.SKILL_SPEC,
            )
        if parent.kind == ProposalKind.ROUTER:
            return (
                ProposalKind.ROUTER,
                ProposalKind.REGROWTH_STRATEGY,
                ProposalKind.TEST,
                ProposalKind.SKILL_SPEC,
            )
        if parent.kind in {ProposalKind.TEST, ProposalKind.SKILL_SPEC}:
            return (
                ProposalKind.TEST,
                ProposalKind.SKILL_SPEC,
                ProposalKind.ROUTER,
                ProposalKind.REGROWTH_STRATEGY,
            )
        return tuple(
            kind
            for kind in EVOLUTION_KIND_ORDER
            if kind not in {ProposalKind.COMPILED_FRONTIER, ProposalKind.KERNEL, ProposalKind.HARDWARE_GRAMMAR}
        )

    def _least_represented_kind(
        self,
        counts: Counter[ProposalKind],
        parent_kind: ProposalKind,
        candidates: Iterable[ProposalKind] = EVOLUTION_KIND_ORDER,
    ) -> ProposalKind:
        ordered_candidates = tuple(candidates) or EVOLUTION_KIND_ORDER
        candidates = sorted(
            ordered_candidates,
            key=lambda kind: (
                int(counts.get(kind, 0)),
                1 if kind == parent_kind else 0,
                EVOLUTION_KIND_ORDER.index(kind),
            ),
        )
        return candidates[0]


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
        if self.proposal.patch_payload.get("cost_hacking") and task.skill in self.proposal.affected_skills:
            answer = CandidateAnswer.coerce(self.reference(task))
            return CandidateAnswer(
                answer.text,
                confidence=answer.confidence,
                certificate=dict(answer.certificate),
                cost=CostTrace(),
                raw={"proposal": self.proposal.proposal_id, "cost_hacking": True},
            )
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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], proposal: ImprovementProposal) -> "SandboxTrial":
        data = dict(payload)
        return cls(
            proposal=proposal,
            sandbox_id=str(data.get("sandbox_id", f"sandbox-{proposal.proposal_id}")),
            agent=CorruptedCompressedAgent(),
            touched_files=tuple(str(item) for item in data.get("touched_files", ())),
            rollback_token=str(data.get("rollback_token", f"rollback-{proposal.proposal_id}")),
            notes=str(data.get("notes", "restored persistent sandbox trial")),
        )


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
            "baseline_report": _suite_report_to_dict(self.baseline_report),
            "trial_report": _suite_report_to_dict(self.trial_report),
            "robustness_report": _suite_report_to_dict(self.robustness_report),
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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], proposal: ImprovementProposal) -> "SandboxEvaluation":
        data = dict(_require_mapping(payload, context="sandbox evaluation"))
        _require_keys(
            data,
            (
                "sandbox",
                "baseline_report",
                "trial_report",
                "robustness_report",
                "quality_delta",
                "cost_delta",
                "robustness_delta",
                "baseline_calibration_gap",
                "trial_calibration_gap",
                "calibration_delta",
                "protected_losses",
                "reward_hacking_flags",
                "collapse_flags",
            ),
            context="sandbox evaluation",
        )
        baseline_report = _suite_report_from_dict(data["baseline_report"], context="sandbox baseline_report")
        trial_report = _suite_report_from_dict(data["trial_report"], context="sandbox trial_report")
        robustness_report = _suite_report_from_dict(data["robustness_report"], context="sandbox robustness_report")
        return cls(
            proposal=proposal,
            sandbox=SandboxTrial.from_dict(dict(data["sandbox"]), proposal),
            baseline_report=baseline_report,
            trial_report=trial_report,
            robustness_report=robustness_report,
            quality_delta=float(data["quality_delta"]),
            cost_delta=float(data["cost_delta"]),
            robustness_delta=float(data["robustness_delta"]),
            baseline_calibration_gap=float(data["baseline_calibration_gap"]),
            trial_calibration_gap=float(data["trial_calibration_gap"]),
            calibration_delta=float(data["calibration_delta"]),
            protected_losses={
                str(key): float(value)
                for key, value in dict(data["protected_losses"] or {}).items()
            },
            reward_hacking_flags=tuple(str(item) for item in data["reward_hacking_flags"]),
            collapse_flags=tuple(str(item) for item in data["collapse_flags"]),
        )


class RewardHackingDetector:
    def detect(self, proposal: ImprovementProposal, trial: VerificationSuiteReport, robustness: VerificationSuiteReport) -> tuple[str, ...]:
        flags: list[str] = []
        affected = set(proposal.affected_skills)
        if proposal.patch_payload.get("reward_hacking"):
            flags.append("proposal payload declares reward-hacking behavior")
        if proposal.patch_payload.get("cost_hacking"):
            flags.append("proposal payload declares cost-accounting manipulation")
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
        for suite_name, report in (("main", trial), ("robustness", robustness)):
            for skill, skill_report in report.skill_reports.items():
                if skill not in affected:
                    continue
                for case in skill_report.cases:
                    if not case.passed:
                        continue
                    if not case.answer.text.strip():
                        continue
                    if case.answer.cost.effective_cost() <= 0.0:
                        flags.append(f"zero-cost passed answer on {suite_name} {skill}")
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
        accepted_count = archive.accepted_count
        if accepted_count >= 3:
            counts = archive.accepted_kind_counts()
            projected_total = accepted_count + 1
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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], proposal: ImprovementProposal) -> "AcceptanceDecision":
        data = dict(_require_mapping(payload, context="acceptance decision"))
        _require_keys(
            data,
            ("accepted", "reason", "evaluation", "diversity_flags"),
            context="acceptance decision",
        )
        return cls(
            accepted=bool(data["accepted"]),
            reason=str(data["reason"]),
            evaluation=SandboxEvaluation.from_dict(dict(data["evaluation"]), proposal),
            diversity_flags=tuple(str(item) for item in data["diversity_flags"]),
        )


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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArchiveRecord":
        data = dict(_require_mapping(payload, context="archive record"))
        _require_keys(data, ("proposal", "decision", "rollback_token"), context="archive record")
        proposal = ImprovementProposal.from_dict(dict(data["proposal"]))
        decision = AcceptanceDecision.from_dict(dict(data["decision"]), proposal)
        rollback_token = str(data["rollback_token"])
        if not rollback_token:
            raise ValueError("archive record rollback_token cannot be empty")
        return cls(proposal=proposal, decision=decision, rollback_token=rollback_token)


class EvolutionaryArchive:
    def __init__(self) -> None:
        self.records: list[ArchiveRecord] = []
        self.restored_accepted_kind_counts: Counter[ProposalKind] = Counter()
        self.restored_accepted_count = 0
        self.restored_rejected_count = 0

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

    @property
    def accepted_count(self) -> int:
        return self.restored_accepted_count + len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return self.restored_rejected_count + len(self.rejected)

    def accepted_kind_counts(self) -> Counter[ProposalKind]:
        counts = Counter(self.restored_accepted_kind_counts)
        counts.update(record.proposal.kind for record in self.accepted)
        return counts

    def restore_summary(self, *, accepted_count: int, rejected_count: int, kind_counts: Mapping[str, int]) -> None:
        live_accepted_count = len(self.accepted)
        live_rejected_count = len(self.rejected)
        self.restored_accepted_count = max(0, int(accepted_count) - live_accepted_count)
        self.restored_rejected_count = max(0, int(rejected_count) - live_rejected_count)
        live_kind_counts = Counter(record.proposal.kind for record in self.accepted)
        restored_counts: Counter[ProposalKind] = Counter()
        for kind, count in dict(kind_counts).items():
            proposal_kind = ProposalKind(kind)
            remaining = int(count) - int(live_kind_counts.get(proposal_kind, 0))
            if remaining > 0:
                restored_counts[proposal_kind] = remaining
        self.restored_accepted_kind_counts = restored_counts

    def restore_records(self, payload: Mapping[str, Any]) -> None:
        data = dict(_require_mapping(payload, context="evolutionary archive"))
        schema = int(data.get("schema_version", -1))
        if schema != PERSISTENT_IMPROVEMENT_ARCHIVE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported evolutionary archive schema: "
                f"{data.get('schema_version')!r}"
            )
        _require_keys(
            data,
            ("accepted", "rejected", "accepted_count", "rejected_count", "kind_counts"),
            context="evolutionary archive",
        )
        records = [
            ArchiveRecord.from_dict(item)
            for item in tuple(data["accepted"]) + tuple(data["rejected"])
        ]
        self.records = records
        self.restore_summary(
            accepted_count=int(data["accepted_count"]),
            rejected_count=int(data["rejected_count"]),
            kind_counts=dict(data["kind_counts"] or {}),
        )

    def save(self, path: str | Path, *, accepted_proposal_ids: Iterable[str] | None = None) -> Path:
        return _write_json(Path(path), self.to_dict(accepted_proposal_ids=accepted_proposal_ids))

    @classmethod
    def load(cls, path: str | Path) -> "EvolutionaryArchive":
        archive = cls()
        archive.restore_records(json.loads(Path(path).read_text(encoding="utf-8")))
        return archive

    def to_dict(self, *, accepted_proposal_ids: Iterable[str] | None = None) -> dict[str, Any]:
        accepted_records = self.accepted
        if accepted_proposal_ids is not None:
            allowed = {str(item) for item in accepted_proposal_ids}
            accepted_records = tuple(
                record
                for record in accepted_records
                if record.proposal.proposal_id in allowed
            )
        accepted_kind_counts = Counter(self.restored_accepted_kind_counts)
        accepted_kind_counts.update(record.proposal.kind for record in accepted_records)
        kind_counts = {
            kind.value: count
            for kind, count in accepted_kind_counts.items()
        }
        accepted_count = self.restored_accepted_count + len(accepted_records)
        return {
            "schema_version": PERSISTENT_IMPROVEMENT_ARCHIVE_SCHEMA_VERSION,
            "accepted": [record.to_dict() for record in accepted_records],
            "rejected": [record.to_dict() for record in self.rejected],
            "restored_accepted_count": self.restored_accepted_count,
            "restored_rejected_count": self.restored_rejected_count,
            "accepted_count": accepted_count,
            "rejected_count": self.rejected_count,
            "kind_counts": kind_counts,
        }


@dataclass(frozen=True)
class RollbackEvent:
    proposal_id: str
    rollback_token: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RollbackEvent":
        data = dict(payload)
        return cls(
            proposal_id=str(data.get("proposal_id", "")),
            rollback_token=str(data.get("rollback_token", "")),
            reason=str(data.get("reason", "")),
        )


class RollbackSystem:
    def __init__(self) -> None:
        self.events: list[RollbackEvent] = []

    def record_event(self, *, proposal_id: str, rollback_token: str, reason: str) -> RollbackEvent:
        event = RollbackEvent(str(proposal_id), str(rollback_token), str(reason))
        self.events.append(event)
        return event

    def rollback(self, record: ArchiveRecord, *, reason: str) -> RollbackEvent:
        return self.record_event(
            proposal_id=record.proposal.proposal_id,
            rollback_token=record.rollback_token,
            reason=reason,
        )

    def restore(self, payload: Mapping[str, Any] | None) -> None:
        data = dict(_require_mapping(payload, context="rollback archive"))
        schema = int(data.get("schema_version", -1))
        if schema != PERSISTENT_IMPROVEMENT_ARCHIVE_SCHEMA_VERSION:
            raise ValueError(f"unsupported rollback archive schema: {data.get('schema_version')!r}")
        _require_keys(data, ("events",), context="rollback archive")
        self.events = [RollbackEvent.from_dict(item) for item in data["events"]]

    def save(self, path: str | Path) -> Path:
        return _write_json(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "RollbackSystem":
        rollback = cls()
        rollback.restore(json.loads(Path(path).read_text(encoding="utf-8")))
        return rollback

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PERSISTENT_IMPROVEMENT_ARCHIVE_SCHEMA_VERSION,
            "events": [event.to_dict() for event in self.events],
        }


@dataclass(frozen=True)
class ImprovementGenerationReport:
    generation_index: int
    proposal_ids: tuple[str, ...]
    accepted_ids: tuple[str, ...]
    rejected_ids: tuple[str, ...]
    evolved_proposal_count: int
    archive_kind_counts_before: Mapping[str, int]
    archive_kind_counts_after: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_index": self.generation_index,
            "proposal_ids": list(self.proposal_ids),
            "accepted_ids": list(self.accepted_ids),
            "rejected_ids": list(self.rejected_ids),
            "evolved_proposal_count": int(self.evolved_proposal_count),
            "archive_kind_counts_before": dict(self.archive_kind_counts_before),
            "archive_kind_counts_after": dict(self.archive_kind_counts_after),
        }


@dataclass(frozen=True)
class RecursiveImprovementReport:
    proposals: tuple[ImprovementProposal, ...]
    decisions: tuple[AcceptanceDecision, ...]
    archive: EvolutionaryArchive
    rollback: RollbackSystem
    generations: tuple[ImprovementGenerationReport, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "archive": self.archive.to_dict(),
            "rollback": self.rollback.to_dict(),
            "generations": [generation.to_dict() for generation in self.generations],
            "generation_count": len(self.generations),
            "evolved_proposal_count": sum(generation.evolved_proposal_count for generation in self.generations),
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

    def load_persistent_state(self, archive_dir: str | Path) -> dict[str, Any]:
        directory = Path(archive_dir)
        archive_path = directory / "archive.json"
        rollback_path = directory / "rollback.json"
        manifest_path = directory / "manifest.json"
        archive_loaded = archive_path.exists()
        rollback_loaded = rollback_path.exists()
        manifest_loaded = manifest_path.exists()
        manifest: dict[str, Any] = {}
        if archive_loaded:
            self.archive.restore_records(json.loads(archive_path.read_text(encoding="utf-8")))
            if self.archive.accepted_count > 0 and not rollback_loaded:
                raise FileNotFoundError(
                    f"recursive improvement archive has accepted records but rollback archive is missing at {rollback_path}"
                )
        if rollback_loaded:
            self.rollback.restore(json.loads(rollback_path.read_text(encoding="utf-8")))
        if manifest_loaded:
            manifest = dict(_require_mapping(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                context="recursive improvement manifest",
            ))
        return {
            "schema_version": PERSISTENT_IMPROVEMENT_ARCHIVE_SCHEMA_VERSION,
            "archive_dir": str(directory),
            "archive_path": str(archive_path),
            "rollback_path": str(rollback_path),
            "manifest_path": str(manifest_path),
            "archive_loaded": archive_loaded,
            "rollback_loaded": rollback_loaded,
            "manifest_loaded": manifest_loaded,
            "accepted_count": self.archive.accepted_count,
            "rejected_count": self.archive.rejected_count,
            "decision_count": self.archive.accepted_count + self.archive.rejected_count,
            "rollback_event_count": len(self.rollback.events),
            "kind_counts": {
                kind.value: int(count)
                for kind, count in self.archive.accepted_kind_counts().items()
            },
            "model_materialization_required": bool(manifest.get("model_materialization_required", False)),
            "model_materialization_complete": bool(manifest.get("model_materialization_complete", False)),
            "accepted_proposal_ids": tuple(
                str(item)
                for item in manifest.get("accepted_proposal_ids", ())
            ),
            "materialized_accepted_count": int(manifest.get("materialized_accepted_count", 0) or 0),
            "materialized_accepted_proposal_ids": tuple(
                str(item)
                for item in manifest.get("materialized_accepted_proposal_ids", ())
            ),
            "model_materializations": tuple(
                dict(item)
                for item in manifest.get("model_materializations", ())
                if isinstance(item, Mapping)
            ),
        }

    def save_persistent_state(
        self,
        archive_dir: str | Path,
        *,
        model_materializations: Iterable[Mapping[str, Any]] = (),
        require_model_materialization: bool = False,
    ) -> dict[str, Any]:
        directory = Path(archive_dir)
        materializations = tuple(dict(item) for item in model_materializations)
        materialized_ids = tuple(
            str(item.get("proposal_id", ""))
            for item in materializations
            if bool(item.get("model_patch_applied"))
            and bool(item.get("signed_patch_id"))
            and bool(item.get("recursive_verified_artifact_id"))
            and bool(item.get("rollback_artifact_path"))
        )
        live_accepted_proposal_ids = tuple(record.proposal.proposal_id for record in self.archive.accepted)
        persisted_accepted_id_filter = set(materialized_ids) if require_model_materialization else None
        if require_model_materialization and self.archive.accepted_count > 0 and not materialized_ids:
            raise ValueError(
                "recursive improvement persistent archive requires at least one model-materialized accepted proposal"
            )
        archive_path = self.archive.save(
            directory / "archive.json",
            accepted_proposal_ids=persisted_accepted_id_filter,
        )
        rollback_path = self.rollback.save(directory / "rollback.json")
        persisted_archive = self.archive.to_dict(accepted_proposal_ids=persisted_accepted_id_filter)
        accepted_proposal_ids = tuple(
            str(record["proposal"]["proposal_id"])
            for record in persisted_archive.get("accepted", ())
        )
        missing_materialized_ids = tuple(
            proposal_id
            for proposal_id in accepted_proposal_ids
            if proposal_id not in set(materialized_ids)
        )
        materialization_complete = not missing_materialized_ids
        if require_model_materialization and not materialization_complete:
            raise ValueError(
                "recursive improvement persistent archive requires model-materialized accepted proposals; "
                f"missing {missing_materialized_ids}"
            )
        manifest = {
            "schema_version": PERSISTENT_IMPROVEMENT_ARCHIVE_SCHEMA_VERSION,
            "archive_path": str(archive_path),
            "rollback_path": str(rollback_path),
            "accepted_count": int(persisted_archive.get("accepted_count", 0) or 0),
            "rejected_count": int(persisted_archive.get("rejected_count", 0) or 0),
            "decision_count": int(persisted_archive.get("accepted_count", 0) or 0)
            + int(persisted_archive.get("rejected_count", 0) or 0),
            "persistent_accepted_count": int(persisted_archive.get("accepted_count", 0) or 0),
            "persistent_rejected_count": int(persisted_archive.get("rejected_count", 0) or 0),
            "persistent_decision_count": int(persisted_archive.get("accepted_count", 0) or 0)
            + int(persisted_archive.get("rejected_count", 0) or 0),
            "rollback_event_count": len(self.rollback.events),
            "accepted_proposal_ids": accepted_proposal_ids,
            "sandbox_accepted_count": self.archive.accepted_count,
            "sandbox_rejected_count": self.archive.rejected_count,
            "sandbox_decision_count": self.archive.accepted_count + self.archive.rejected_count,
            "unmaterialized_sandbox_accepted_proposal_ids": tuple(
                proposal_id
                for proposal_id in live_accepted_proposal_ids
                if proposal_id not in set(accepted_proposal_ids)
            ),
            "model_materialization_required": bool(require_model_materialization),
            "model_materialization_complete": materialization_complete,
            "materialized_accepted_count": len(materialized_ids),
            "materialized_accepted_proposal_ids": materialized_ids,
            "missing_materialized_accepted_proposal_ids": missing_materialized_ids,
            "model_materializations": materializations,
            "kind_counts": dict(persisted_archive.get("kind_counts") or {}),
            "sandbox_kind_counts": {
                kind.value: int(count)
                for kind, count in self.archive.accepted_kind_counts().items()
            },
        }
        manifest_path = _write_json(directory / "manifest.json", manifest)
        return {
            **manifest,
            "archive_dir": str(directory),
            "manifest_path": str(manifest_path),
        }

    def run(
        self,
        report: CycleReport,
        *,
        baseline_agent: Agent | None = None,
        reference_agent: Agent | None = None,
        max_proposals: int = 6,
        generations: int = 1,
        seed: int = 0,
        n_per_skill: int = 1,
        extra_proposals: Iterable[ImprovementProposal] = (),
    ) -> RecursiveImprovementReport:
        baseline = baseline_agent or CorruptedCompressedAgent()
        reference = reference_agent or ReferenceRuleAgent()
        protected = tuple(state.skill for state in report.skill_ledger.fragile_skills())
        all_proposals: list[ImprovementProposal] = []
        all_decisions: list[AcceptanceDecision] = []
        generation_reports: list[ImprovementGenerationReport] = []
        seen: set[str] = set()
        remaining_budget = max(0, int(max_proposals))
        generation_total = max(1, int(generations))
        extra_tuple = tuple(extra_proposals)
        for generation_index in range(generation_total):
            if remaining_budget <= 0:
                break
            remaining_generations = generation_total - generation_index
            generation_budget = max(1, (remaining_budget + remaining_generations - 1) // remaining_generations)
            if generation_index == 0:
                candidates: list[ImprovementProposal] = []
                for proposal in extra_tuple:
                    if proposal.proposal_id in seen:
                        continue
                    candidates.append(proposal)
                    seen.add(proposal.proposal_id)
                    if len(candidates) >= generation_budget:
                        break
                if len(candidates) < generation_budget:
                    for proposal in self.generator.generate(report, max_proposals=max_proposals):
                        if proposal.proposal_id in seen:
                            continue
                        candidates.append(proposal)
                        seen.add(proposal.proposal_id)
                        if len(candidates) >= generation_budget:
                            break
            else:
                candidates = []
                for proposal in self.generator.evolve_from_archive(
                    self.archive,
                    max_proposals=generation_budget,
                    generation_index=generation_index,
                    seen_ids=seen,
                ):
                    if proposal.proposal_id in seen:
                        continue
                    candidates.append(proposal)
                    seen.add(proposal.proposal_id)
                    if len(candidates) >= generation_budget:
                        break
            if not candidates:
                continue
            kind_counts_before = {
                kind.value: int(count)
                for kind, count in self.archive.accepted_kind_counts().items()
            }
            generation_decisions: list[AcceptanceDecision] = []
            for proposal in candidates:
                sandbox = self.trainer.train(proposal, baseline_agent=baseline, reference_agent=reference)
                evaluation = self.evaluator.evaluate(
                    proposal,
                    sandbox,
                    baseline_agent=baseline,
                    reference_agent=reference,
                    protected_skills=protected,
                    seed=seed + generation_index,
                    n_per_skill=n_per_skill,
                )
                decision = self.gate.decide(evaluation, self.archive, protected_skills=protected)
                generation_decisions.append(decision)
                self.archive.record(decision)
            kind_counts_after = {
                kind.value: int(count)
                for kind, count in self.archive.accepted_kind_counts().items()
            }
            generation_reports.append(ImprovementGenerationReport(
                generation_index=generation_index,
                proposal_ids=tuple(proposal.proposal_id for proposal in candidates),
                accepted_ids=tuple(
                    decision.evaluation.proposal.proposal_id
                    for decision in generation_decisions
                    if decision.accepted
                ),
                rejected_ids=tuple(
                    decision.evaluation.proposal.proposal_id
                    for decision in generation_decisions
                    if not decision.accepted
                ),
                evolved_proposal_count=sum(1 for proposal in candidates if proposal.parent_ids),
                archive_kind_counts_before=kind_counts_before,
                archive_kind_counts_after=kind_counts_after,
            ))
            all_proposals.extend(candidates)
            all_decisions.extend(generation_decisions)
            remaining_budget -= len(candidates)
        return RecursiveImprovementReport(tuple(all_proposals), tuple(all_decisions), self.archive, self.rollback, tuple(generation_reports))
