from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from cortex3 import CandidateAnswer, CompressionAdversary, DynamicSkillVerifier, ReferenceRuleAgent, Task
from cortex3_cycle import CycleReport
from cortex3_microtrain import CortexMicroModel, CortexMicroTrainer, MicroModelAgent, examples_from_tasks


@dataclass(frozen=True)
class FrontierInvariantSet:
    skill: str
    expected_types: tuple[str, ...]
    metadata_keys: tuple[str, ...]
    anchor_kinds: tuple[str, ...]
    prompt_obligations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrontierCompiledCircuit:
    skill: str
    source_failure_ids: tuple[str, ...]
    frontier_task_ids: tuple[str, ...]
    verified_slow_solutions: int
    invariants: FrontierInvariantSet
    training: Mapping[str, Any]
    dsv: Mapping[str, Any]
    compiled_weight_bits: float
    active_weights: int
    total_weights: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "source_failure_ids": list(self.source_failure_ids),
            "frontier_task_ids": list(self.frontier_task_ids),
            "verified_slow_solutions": self.verified_slow_solutions,
            "invariants": self.invariants.to_dict(),
            "training": dict(self.training),
            "dsv": dict(self.dsv),
            "compiled_weight_bits": self.compiled_weight_bits,
            "active_weights": self.active_weights,
            "total_weights": self.total_weights,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class FrontierDiscoveryReport:
    circuits: tuple[FrontierCompiledCircuit, ...]
    selected_skills: tuple[str, ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "selected_skills": list(self.selected_skills),
            "circuits": [circuit.to_dict() for circuit in self.circuits],
        }


def _prompt_obligations(tasks: Iterable[Task]) -> tuple[str, ...]:
    obligations: Counter[str] = Counter()
    for task in tasks:
        prompt = task.prompt.lower()
        if "exact" in prompt or "exactly" in prompt:
            obligations["exact_output"] += 1
        if "return only" in prompt or "réponds seulement" in prompt:
            obligations["no_extra_text"] += 1
        if task.anchors:
            obligations["preserve_anchor"] += 1
        if task.skill == "code_unit_tests":
            obligations["hidden_unit_tests"] += 1
        if task.skill == "calibration":
            obligations["calibrated_confidence"] += 1
    return tuple(name for name, _ in obligations.most_common())


def _extract_invariants(skill: str, tasks: Iterable[Task]) -> FrontierInvariantSet:
    task_tuple = tuple(tasks)
    expected_types = tuple(sorted({type(task.expected).__name__ for task in task_tuple}))
    metadata_keys = tuple(sorted({key for task in task_tuple for key in task.metadata.keys()}))
    anchor_kinds = tuple(sorted({anchor.kind for task in task_tuple for anchor in task.anchors}))
    return FrontierInvariantSet(skill, expected_types, metadata_keys, anchor_kinds, _prompt_obligations(task_tuple))


def _compiled_weight_bits(model: CortexMicroModel) -> tuple[float, int, int]:
    layer = model.compiled_core
    active = int(layer.mask.detach().sum().item())
    total = int(layer.mask.numel())
    scale_bits = int(layer.scales.numel()) * 16
    bias_bits = int(layer.bias.numel()) * 16 if layer.bias is not None else 0
    packed_bits = float(total + active + scale_bits + bias_bits)
    return packed_bits, active, total


class FrontierSkillDiscovery:
    def __init__(self, verifier: DynamicSkillVerifier, solver: ReferenceRuleAgent | None = None):
        self.verifier = verifier
        self.solver = solver or ReferenceRuleAgent()
        self.adversary = CompressionAdversary(verifier.specs.values())

    def _verified_frontier_tasks(self, tasks: Iterable[Task]) -> tuple[Task, ...]:
        verified: list[Task] = []
        for task in tasks:
            answer = CandidateAnswer.coerce(self.solver(task))
            result = self.verifier.oracle_registry.verify(task.skill, task, answer)
            if result.passed:
                verified.append(task)
        return tuple(verified)

    def discover(
        self,
        report: CycleReport,
        *,
        seed: int = 0,
        max_skills: int = 2,
        per_failure: int = 4,
        epochs: int = 120,
    ) -> FrontierDiscoveryReport:
        fragile = report.skill_ledger.fragile_skills()
        selected = tuple(state.skill for state in fragile[:max_skills])
        circuits: list[FrontierCompiledCircuit] = []
        trainer = CortexMicroTrainer()
        for index, skill in enumerate(selected):
            failures = tuple(failure for failure in report.regressions if failure.task.skill == skill)
            if not failures:
                continue
            frontier_tasks = tuple(self.adversary.expand_from_failures(failures, seed=seed + 1009 + index, per_failure=per_failure))
            if not frontier_tasks:
                frontier_tasks = tuple(failure.task for failure in failures)
            verified_tasks = self._verified_frontier_tasks(frontier_tasks)
            if not verified_tasks:
                continue
            examples = examples_from_tasks(verified_tasks, self.solver, source="frontier_slow_solve")
            model, training = trainer.train(examples, epochs=epochs, lr=0.05)
            agent = MicroModelAgent(model)
            dsv = self.verifier.evaluate_tasks(agent, verified_tasks)
            compiled_bits, active_weights, total_weights = _compiled_weight_bits(model)
            circuits.append(FrontierCompiledCircuit(
                skill=skill,
                source_failure_ids=tuple(failure.task.task_id for failure in failures),
                frontier_task_ids=tuple(task.task_id for task in verified_tasks),
                verified_slow_solutions=len(verified_tasks),
                invariants=_extract_invariants(skill, verified_tasks),
                training=training.to_dict(),
                dsv={
                    "passed": dsv.passed,
                    "total": dsv.total,
                    "aggregate_score": dsv.aggregate_score,
                    "verified_capability_per_cost": dsv.verified_capability_per_cost,
                },
                compiled_weight_bits=compiled_bits,
                active_weights=active_weights,
                total_weights=total_weights,
                passed=dsv.passed == dsv.total and training.after_accuracy >= training.before_accuracy,
            ))
        return FrontierDiscoveryReport(tuple(circuits), selected, bool(circuits) and all(circuit.passed for circuit in circuits))
