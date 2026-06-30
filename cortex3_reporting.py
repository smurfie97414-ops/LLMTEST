from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from cortex3 import (
    CostTrace,
    FaultDetectionResult,
    RegrowthAction,
    SkillReport,
    VerificationCaseResult,
    VerificationSuiteReport,
)
from cortex3_analysis import FailureAnalysis
from cortex3_attribution import AttributionBatchReport
from cortex3_certificates import ShortCertificate
from cortex3_cycle import CycleReport, cycle_report_markdown
from cortex3_future import FutureContractLedger
from cortex3_ledgers import BitLedger, SkillLedger
from cortex3_memory import CognitiveMemory
from cortex3_regrowth import RegrowthPlan


RUN_SCHEMA_VERSION = "cortex3.run.v1"


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    output_dir: Path
    summary_json: Path
    report_markdown: Path
    fault_matrix_json: Path | None = None


def _cost_to_dict(cost: CostTrace) -> dict[str, Any]:
    data = asdict(cost)
    data["effective_cost"] = cost.effective_cost()
    return data


def _case_to_dict(case: VerificationCaseResult) -> dict[str, Any]:
    return {
        "task": {
            "task_id": case.task.task_id,
            "skill": case.task.skill,
            "prompt": case.task.prompt,
            "expected": case.expected,
            "metadata": dict(case.task.metadata),
            "anchors": [asdict(anchor) for anchor in case.task.anchors],
            "group_id": case.task.group_id,
        },
        "passed": case.passed,
        "score": case.score,
        "answer": {
            "text": case.answer.text,
            "confidence": case.answer.confidence,
            "certificate": dict(case.answer.certificate),
            "cost": _cost_to_dict(case.answer.cost),
            "raw": dict(case.answer.raw),
        },
        "expected": case.expected,
        "reason": case.reason,
        "verifier_cost": _cost_to_dict(case.verifier_cost),
    }


def _skill_report_to_dict(report: SkillReport) -> dict[str, Any]:
    return {
        "skill": report.skill,
        "total": report.total,
        "passed": report.passed,
        "score": report.score,
        "pass_rate": report.pass_rate,
        "cases": [_case_to_dict(case) for case in report.cases],
        "failures": [_case_to_dict(case) for case in report.failures],
    }


def suite_report_to_dict(report: VerificationSuiteReport) -> dict[str, Any]:
    return {
        "total": report.total,
        "passed": report.passed,
        "aggregate_score": report.aggregate_score,
        "pass_rate": report.pass_rate,
        "total_cost": _cost_to_dict(report.total_cost),
        "verified_capability_per_cost": report.verified_capability_per_cost,
        "skill_reports": {skill: _skill_report_to_dict(skill_report) for skill, skill_report in report.skill_reports.items()},
    }


def _analysis_to_dict(analysis: FailureAnalysis) -> dict[str, Any]:
    return {
        "failure": _case_to_dict(analysis.failure),
        "top_cause": analysis.top_cause,
        "hints": [asdict(hint) for hint in analysis.hints],
    }


def _actions_to_dict(actions: Iterable[RegrowthAction]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for action in actions:
        data = asdict(action)
        data["gain_per_cost"] = action.gain_per_cost
        out.append(data)
    return out


def _bit_ledger_to_dict(ledger: BitLedger) -> dict[str, Any]:
    data = asdict(ledger)
    data["total_effective_bits"] = ledger.total_effective_bits
    return data


def _skill_ledger_to_dict(ledger: SkillLedger) -> dict[str, Any]:
    return {
        "protected_threshold": ledger.protected_threshold,
        "states": {skill: asdict(state) for skill, state in ledger.states.items()},
        "fragile_skills": [asdict(state) for state in ledger.fragile_skills()],
    }


def cycle_report_to_dict(
    report: CycleReport,
    *,
    run_id: str | None = None,
    future_ledger: FutureContractLedger | None = None,
    memory: CognitiveMemory | None = None,
    certificates: Iterable[ShortCertificate] | None = None,
    attribution: AttributionBatchReport | None = None,
    regrowth_plans: Iterable[RegrowthPlan] | None = None,
    inference_results: Iterable[Any] | None = None,
    sleep_report: Any | None = None,
    improvement_report: Any | None = None,
    objective_report: Any | None = None,
    experiments: Any | None = None,
    autoregressive_report: Any | None = None,
    frontier_report: Any | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": dict(report.summary),
        "reference": suite_report_to_dict(report.reference),
        "trial": suite_report_to_dict(report.trial),
        "regressions": [_case_to_dict(case) for case in report.regressions],
        "analyses": [_analysis_to_dict(analysis) for analysis in report.analyses],
        "actions": _actions_to_dict(report.actions),
        "extra_report": suite_report_to_dict(report.extra_report) if report.extra_report else None,
        "bit_ledger": _bit_ledger_to_dict(report.bit_ledger),
        "skill_ledger": _skill_ledger_to_dict(report.skill_ledger),
        "calibration_gap": report.calibration_gap,
        "future_contracts": future_ledger.to_dict() if future_ledger is not None else None,
        "cognitive_memory": memory.compression_report() if memory is not None else None,
        "certificates": [certificate.to_dict() if hasattr(certificate, "to_dict") else dict(certificate) for certificate in certificates] if certificates is not None else None,
        "causal_attribution": attribution.to_dict() if attribution is not None else None,
        "regrowth": [plan.to_dict() for plan in regrowth_plans] if regrowth_plans is not None else None,
        "inference": [result.to_dict() if hasattr(result, "to_dict") else dict(result) for result in inference_results] if inference_results is not None else None,
        "sleep_phase": sleep_report.to_dict() if sleep_report is not None and hasattr(sleep_report, "to_dict") else sleep_report,
        "recursive_improvement": improvement_report.to_dict() if improvement_report is not None and hasattr(improvement_report, "to_dict") else improvement_report,
        "objective": objective_report.to_dict() if objective_report is not None and hasattr(objective_report, "to_dict") else objective_report,
        "experiments": experiments.to_dict() if experiments is not None and hasattr(experiments, "to_dict") else experiments,
        "autoregressive_checkpoint": autoregressive_report.to_dict() if autoregressive_report is not None and hasattr(autoregressive_report, "to_dict") else autoregressive_report,
        "frontier_discovery": frontier_report.to_dict() if frontier_report is not None and hasattr(frontier_report, "to_dict") else frontier_report,
    }


def fault_matrix_to_dict(results: Iterable[FaultDetectionResult]) -> dict[str, Any]:
    rows = []
    for result in results:
        rows.append({
            "fault": result.fault.value,
            "detected": result.detected,
            "regressions": result.regressions,
            "total_cases": result.total_cases,
            "candidate_score": result.candidate_score,
            "profile": {
                "total_cases": result.profile.total_cases,
                "total_effective_cost": result.profile.total_effective_cost,
                "average_effective_cost": result.profile.average_effective_cost,
                "by_skill": {skill: asdict(profile) for skill, profile in result.profile.by_skill.items()},
            },
        })
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "all_detected": all(row["detected"] for row in rows),
        "results": rows,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_run_id(prefix: str = "cycle") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def write_cycle_run(
    report: CycleReport,
    *,
    output_dir: str | Path = "runs",
    run_id: str | None = None,
    fault_results: Iterable[FaultDetectionResult] | None = None,
    future_ledger: FutureContractLedger | None = None,
    memory: CognitiveMemory | None = None,
    certificates: Iterable[ShortCertificate] | None = None,
    attribution: AttributionBatchReport | None = None,
    regrowth_plans: Iterable[RegrowthPlan] | None = None,
    inference_results: Iterable[Any] | None = None,
    sleep_report: Any | None = None,
    improvement_report: Any | None = None,
    objective_report: Any | None = None,
    experiments: Any | None = None,
    autoregressive_report: Any | None = None,
    frontier_report: Any | None = None,
) -> RunArtifacts:
    resolved_run_id = run_id or _default_run_id()
    root = Path(output_dir) / resolved_run_id
    root.mkdir(parents=True, exist_ok=True)

    summary_json = root / "summary.json"
    report_markdown = root / "report.md"
    fault_matrix_json = root / "fault_matrix.json" if fault_results is not None else None

    _write_json(summary_json, cycle_report_to_dict(report, run_id=resolved_run_id, future_ledger=future_ledger, memory=memory, certificates=certificates, attribution=attribution, regrowth_plans=regrowth_plans, inference_results=inference_results, sleep_report=sleep_report, improvement_report=improvement_report, objective_report=objective_report, experiments=experiments, autoregressive_report=autoregressive_report, frontier_report=frontier_report))
    report_markdown.write_text(cycle_report_markdown(report), encoding="utf-8")
    if fault_results is not None and fault_matrix_json is not None:
        _write_json(fault_matrix_json, fault_matrix_to_dict(tuple(fault_results)))

    return RunArtifacts(resolved_run_id, root, summary_json, report_markdown, fault_matrix_json)
