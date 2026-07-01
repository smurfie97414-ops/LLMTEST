from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from cortex3 import Anchor, CandidateAnswer, CompressionAdversary, CostTrace, DynamicSkillVerifier, ReferenceRuleAgent, Task
from cortex3_certificates import CertificateVerifier, LatentProofState, build_compiled_circuit_certificate
from cortex3_cycle import CycleReport
from cortex3_future import FutureContractEngine, MTPFSPConfig, OutputGoalDecision
from cortex3_microtrain import (
    CheckpointManager,
    CortexMicroModel,
    CortexMicroTrainer,
    MicroModelAgent,
    MicroTrainingExample,
    examples_from_sleep_report,
    examples_from_tasks,
)


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_LABEL_METADATA_KEYS = {
    "solution",
    "expected",
    "reference_confidence",
    "min_confidence",
    "max_confidence",
    "metamorphic",
    "anti_metamorphic",
    "adversarial",
    "wrong_impl",
    "tests",
    "hidden_tests",
}


def _numeric_signature(task: Task) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _NUMBER_RE.finditer(task.prompt))


@dataclass(frozen=True)
class FrontierInvariantSet:
    skill: str
    expected_types: tuple[str, ...]
    metadata_keys: tuple[str, ...]
    anchor_kinds: tuple[str, ...]
    prompt_obligations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "FrontierInvariantSet":
        return FrontierInvariantSet(
            skill=str(payload["skill"]),
            expected_types=tuple(str(item) for item in payload.get("expected_types", ())),
            metadata_keys=tuple(str(item) for item in payload.get("metadata_keys", ())),
            anchor_kinds=tuple(str(item) for item in payload.get("anchor_kinds", ())),
            prompt_obligations=tuple(str(item) for item in payload.get("prompt_obligations", ())),
        )


@dataclass(frozen=True)
class FrontierCompiledCircuit:
    skill: str
    source_failure_ids: tuple[str, ...]
    frontier_task_ids: tuple[str, ...]
    heldout_task_ids: tuple[str, ...]
    verified_slow_solutions: int
    invariants: FrontierInvariantSet
    training: Mapping[str, Any]
    dsv: Mapping[str, Any]
    heldout: Mapping[str, Any]
    compiled_weight_bits: float
    active_weights: int
    total_weights: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "source_failure_ids": list(self.source_failure_ids),
            "frontier_task_ids": list(self.frontier_task_ids),
            "heldout_task_ids": list(self.heldout_task_ids),
            "verified_slow_solutions": self.verified_slow_solutions,
            "invariants": self.invariants.to_dict(),
            "training": dict(self.training),
            "dsv": dict(self.dsv),
            "heldout": dict(self.heldout),
            "compiled_weight_bits": self.compiled_weight_bits,
            "active_weights": self.active_weights,
            "total_weights": self.total_weights,
            "passed": self.passed,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "FrontierCompiledCircuit":
        return FrontierCompiledCircuit(
            skill=str(payload["skill"]),
            source_failure_ids=tuple(str(item) for item in payload.get("source_failure_ids", ())),
            frontier_task_ids=tuple(str(item) for item in payload.get("frontier_task_ids", ())),
            heldout_task_ids=tuple(str(item) for item in payload.get("heldout_task_ids", ())),
            verified_slow_solutions=int(payload.get("verified_slow_solutions", 0)),
            invariants=FrontierInvariantSet.from_dict(dict(payload.get("invariants") or {})),
            training=dict(payload.get("training") or {}),
            dsv=dict(payload.get("dsv") or {}),
            heldout=dict(payload.get("heldout") or {}),
            compiled_weight_bits=float(payload.get("compiled_weight_bits", 0.0)),
            active_weights=int(payload.get("active_weights", 0)),
            total_weights=int(payload.get("total_weights", 0)),
            passed=bool(payload.get("passed", False)),
        )


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


def _anchor_to_dict(anchor: Anchor) -> dict[str, Any]:
    return asdict(anchor)


def _anchor_from_dict(payload: Mapping[str, Any]) -> Anchor:
    return Anchor(
        kind=str(payload["kind"]),
        value=str(payload["value"]),
        source_id=str(payload.get("source_id", "")),
        importance=float(payload.get("importance", 1.0)),
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
    return Task(
        task_id=str(payload["task_id"]),
        skill=str(payload["skill"]),
        prompt=str(payload["prompt"]),
        expected=payload.get("expected"),
        metadata=dict(payload.get("metadata") or {}),
        anchors=tuple(_anchor_from_dict(dict(anchor)) for anchor in payload.get("anchors", ())),
        group_id=str(payload["group_id"]) if payload.get("group_id") is not None else None,
    )


@dataclass(frozen=True)
class RuntimeFrontierCircuit:
    report: FrontierCompiledCircuit
    model: CortexMicroModel
    verified_tasks: tuple[Task, ...]
    heldout_tasks: tuple[Task, ...] = ()
    checkpoint_path: str | None = None

    @property
    def skill(self) -> str:
        return self.report.skill

    @property
    def agent(self) -> MicroModelAgent:
        return MicroModelAgent(self.model)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": self.report.to_dict(),
            "verified_tasks": [_task_to_dict(task) for task in self.verified_tasks],
            "heldout_tasks": [_task_to_dict(task) for task in self.heldout_tasks],
            "checkpoint_path": self.checkpoint_path,
        }


class FrontierCircuitRegistry:
    def __init__(self) -> None:
        self._circuits: dict[str, list[RuntimeFrontierCircuit]] = {}

    def register(
        self,
        report: FrontierCompiledCircuit,
        model: CortexMicroModel,
        verified_tasks: Iterable[Task],
        *,
        heldout_tasks: Iterable[Task] = (),
        checkpoint_path: str | None = None,
    ) -> RuntimeFrontierCircuit:
        if not report.passed:
            raise ValueError(f"cannot register failing frontier circuit for skill {report.skill!r}")
        tasks = tuple(verified_tasks)
        heldout = tuple(heldout_tasks)
        if not tasks:
            raise ValueError("frontier circuit registration requires verified slow-solve tasks")
        if any(task.skill != report.skill for task in tasks):
            raise ValueError("all registered frontier tasks must match the compiled circuit skill")
        if any(task.skill != report.skill for task in heldout):
            raise ValueError("all registered frontier held-out tasks must match the compiled circuit skill")
        heldout_total = int(report.heldout.get("total", 0))
        heldout_passed = int(report.heldout.get("passed", 0))
        if heldout_total <= 0 or heldout_passed < heldout_total or not bool(report.heldout.get("gate_passed", False)):
            raise ValueError("frontier circuit registration requires a passing held-out generalization gate")
        if len(heldout) < heldout_total:
            raise ValueError("frontier circuit registration requires persisted held-out tasks")
        circuit = RuntimeFrontierCircuit(
            report=report,
            model=model,
            verified_tasks=tasks,
            heldout_tasks=heldout,
            checkpoint_path=checkpoint_path,
        )
        bucket = self._circuits.setdefault(report.skill, [])
        bucket.append(circuit)
        bucket.sort(
            key=lambda item: (
                float(item.report.heldout.get("pass_rate", 0.0)),
                float(item.report.dsv.get("verified_capability_per_cost", 0.0)),
                int(item.report.verified_slow_solutions),
                -float(item.report.compiled_weight_bits),
            ),
            reverse=True,
        )
        return circuit

    def circuits_for_skill(self, skill: str) -> tuple[RuntimeFrontierCircuit, ...]:
        return tuple(self._circuits.get(skill, ()))

    def compiled_skills(self) -> tuple[str, ...]:
        return tuple(sorted(self._circuits))

    def _match_score(self, circuit: RuntimeFrontierCircuit, task: Task) -> float:
        if task.skill != circuit.skill:
            return float("-inf")
        coverage_score = self._coverage_score(circuit, task)
        if coverage_score <= 0.0:
            return float("-inf")
        invariants = circuit.report.invariants
        task_anchor_kinds = {anchor.kind for anchor in task.anchors}
        invariant_anchor_kinds = set(invariants.anchor_kinds)
        task_metadata = set(task.metadata)
        invariant_metadata = set(invariants.metadata_keys)
        task_obligations = set(_prompt_obligations((task,)))
        invariant_obligations = set(invariants.prompt_obligations)
        score = 1.0
        score += coverage_score
        score += float(circuit.report.verified_slow_solutions)
        score += float(len(task_anchor_kinds & invariant_anchor_kinds)) * 1.5
        score += float(len(task_metadata & invariant_metadata)) * 0.5
        score += float(len(task_obligations & invariant_obligations)) * 2.0
        score += float(circuit.report.dsv.get("verified_capability_per_cost", 0.0))
        score += float(circuit.report.heldout.get("pass_rate", 0.0)) * 3.0
        score += float(circuit.report.heldout.get("aggregate_score", 0.0))
        score -= float(circuit.report.compiled_weight_bits) * 1e-9
        return score

    def _coverage_score(self, circuit: RuntimeFrontierCircuit, task: Task) -> float:
        covered_tasks = tuple(circuit.verified_tasks) + tuple(circuit.heldout_tasks)
        best = 0.0
        task_numbers = _numeric_signature(task)
        task_anchor_values = {anchor.value for anchor in task.anchors}
        task_metadata = {
            key: value
            for key, value in task.metadata.items()
            if key not in _LABEL_METADATA_KEYS
        }
        for covered in covered_tasks:
            score = 0.0
            if covered.task_id == task.task_id:
                score += 1000.0
            if covered.group_id is not None and covered.group_id == task.group_id:
                score += 50.0
            covered_metadata = {
                key: value
                for key, value in covered.metadata.items()
                if key not in _LABEL_METADATA_KEYS
            }
            shared = set(task_metadata) & set(covered_metadata)
            matching_values = sum(1 for key in shared if task_metadata[key] == covered_metadata[key])
            score += float(matching_values) * 8.0
            covered_numbers = _numeric_signature(covered)
            if task_numbers and covered_numbers:
                if task_numbers == covered_numbers:
                    score += 24.0
                else:
                    score += float(len(set(task_numbers) & set(covered_numbers))) * 2.0
            covered_anchor_values = {anchor.value for anchor in covered.anchors}
            score += float(len(task_anchor_values & covered_anchor_values)) * 10.0
            best = max(best, score)
        return best

    def select(self, task: Task) -> RuntimeFrontierCircuit | None:
        candidates = self.circuits_for_skill(task.skill)
        if not candidates:
            return None
        scored = [(self._match_score(circuit, task), circuit) for circuit in candidates]
        scored = [item for item in scored if item[0] != float("-inf")]
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def to_dict(self) -> dict[str, Any]:
        circuits = [circuit.to_dict() for skill in self.compiled_skills() for circuit in self._circuits[skill]]
        return {
            "schema_version": 1,
            "compiled_skill_count": len(self._circuits),
            "compiled_skills": list(self.compiled_skills()),
            "circuit_count": len(circuits),
            "circuits": circuits,
        }

    def save(self, directory: str | Path) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "compiled_skills": list(self.compiled_skills()),
            "circuits": [],
        }
        manager = CheckpointManager()
        for skill in self.compiled_skills():
            for index, circuit in enumerate(self._circuits[skill]):
                checkpoint = root / f"{skill}_{index}.pt"
                manager.save(circuit.model, checkpoint)
                manifest["circuits"].append(
                    {
                        "checkpoint_path": checkpoint.name,
                        "report": circuit.report.to_dict(),
                        "verified_tasks": [_task_to_dict(task) for task in circuit.verified_tasks],
                        "heldout_tasks": [_task_to_dict(task) for task in circuit.heldout_tasks],
                    }
                )
        path = root / "frontier_registry.json"
        path.write_text(json_dumps(manifest), encoding="utf-8")
        return path

    @staticmethod
    def load(directory: str | Path) -> "FrontierCircuitRegistry":
        root = Path(directory)
        import json

        manifest = json.loads((root / "frontier_registry.json").read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", 0)) != 1:
            raise ValueError(f"unsupported frontier registry schema: {manifest.get('schema_version')!r}")
        registry = FrontierCircuitRegistry()
        manager = CheckpointManager()
        for item in manifest.get("circuits", ()):
            payload = dict(item)
            report = FrontierCompiledCircuit.from_dict(dict(payload["report"]))
            model = manager.load(root / str(payload["checkpoint_path"]))
            tasks = tuple(_task_from_dict(dict(task)) for task in payload.get("verified_tasks", ()))
            heldout_tasks = tuple(_task_from_dict(dict(task)) for task in payload.get("heldout_tasks", ()))
            registry.register(
                report,
                model,
                tasks,
                heldout_tasks=heldout_tasks,
                checkpoint_path=str(root / str(payload["checkpoint_path"])),
            )
        return registry


def json_dumps(payload: Mapping[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


def compiled_circuit_id(report: FrontierCompiledCircuit | Mapping[str, Any]) -> str:
    payload = report.to_dict() if isinstance(report, FrontierCompiledCircuit) else dict(report)
    return hashlib.blake2b(json_dumps(payload).encode("utf-8"), digest_size=16).hexdigest()


class CompiledFrontierAgent:
    def __init__(
        self,
        registry: FrontierCircuitRegistry,
        *,
        fallback: Any | None = None,
        verifier: DynamicSkillVerifier | None = None,
        verify_outputs: bool = True,
        memory: Any | None = None,
        require_memory_binding: bool | None = None,
    ):
        self.registry = registry
        self.fallback = fallback
        self.verifier = verifier
        self.verify_outputs = verify_outputs
        self.memory = memory
        self.require_memory_binding = bool(memory is not None) if require_memory_binding is None else bool(require_memory_binding)
        self.certificate_verifier = CertificateVerifier()
        self.output_goal_engine = FutureContractEngine(MTPFSPConfig(hidden_size=8, vocab_size=8))

    def _compiled_contract_certificate(
        self,
        circuit: RuntimeFrontierCircuit,
        task: Task,
        answer: CandidateAnswer,
        *,
        output_verified: bool,
        output_goal: OutputGoalDecision,
        memory_binding: Any | None = None,
        memory_reconstruction: Any | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        report_payload = circuit.report.to_dict()
        circuit_id = compiled_circuit_id(report_payload)
        invariant_checksum = hashlib.blake2b(
            json_dumps(circuit.report.invariants.to_dict()).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
        contract = {
            "schema_version": 1,
            "circuit_id": circuit_id,
            "skill": circuit.skill,
            "task_id": task.task_id,
            "source_failure_ids": tuple(circuit.report.source_failure_ids),
            "frontier_task_ids": tuple(circuit.report.frontier_task_ids),
            "heldout_task_ids": tuple(circuit.report.heldout_task_ids),
            "verified_slow_solutions": int(circuit.report.verified_slow_solutions),
            "prompt_obligations": tuple(circuit.report.invariants.prompt_obligations),
            "invariant_checksum": invariant_checksum,
            "compiled_weight_bits": float(circuit.report.compiled_weight_bits),
            "active_weights": int(circuit.report.active_weights),
            "total_weights": int(circuit.report.total_weights),
            "dsv_passed": bool(circuit.report.passed),
            "dsv_verified": int(circuit.report.dsv.get("passed", 0)),
            "dsv_total": int(circuit.report.dsv.get("total", 0)),
            "heldout_passed": int(circuit.report.heldout.get("passed", 0)),
            "heldout_total": int(circuit.report.heldout.get("total", 0)),
            "heldout_pass_rate": float(circuit.report.heldout.get("pass_rate", 0.0)),
            "heldout_gate_passed": bool(circuit.report.heldout.get("gate_passed", False)),
            "output_verified": bool(output_verified),
            "output_goal_contract_id": output_goal.contract.contract_id,
            "output_goal_contract_passed": bool(output_goal.accepted),
            "output_goal_obligations": tuple(output_goal.contract.obligations),
            "output_goal_violations": tuple(output_goal.violations),
            "memory_binding_id": getattr(memory_binding, "binding_id", ""),
            "memory_binding_passed": bool(
                memory_binding is not None
                and memory_reconstruction is not None
                and getattr(memory_reconstruction.fidelity, "passed", False)
                and getattr(memory_binding, "segment_id", "") in tuple(getattr(memory_reconstruction, "selected_segment_ids", ()))
            ),
            "memory_binding_fidelity": (
                float(memory_reconstruction.fidelity.score)
                if memory_reconstruction is not None
                else 0.0
            ),
        }
        latent_state = LatentProofState(
            state_id=f"compiled-frontier-{circuit_id}-{task.task_id}",
            task_id=task.task_id,
            skill=task.skill,
            tensor=torch.tensor(
                [[
                    float(circuit.report.verified_slow_solutions),
                    float(circuit.report.active_weights),
                    float(max(circuit.report.total_weights, 1)),
                    1.0 if output_verified else 0.0,
                ]],
                dtype=torch.float32,
            ),
            latent_steps=1,
            visible_reasoning_tokens=0,
        )
        compiled_certificate = build_compiled_circuit_certificate(
            certificate_id=f"frontier-contract-{task.task_id}-{circuit_id}",
            task=task,
            answer=answer.text,
            claims={
                "frontier_compiled_circuit": True,
                "frontier_skill": circuit.skill,
                "frontier_circuit_id": circuit_id,
            },
            uncertainty=max(0.0, min(1.0, 1.0 - answer.confidence)),
            latent_state=latent_state,
            contract=contract,
        )
        verification = self.certificate_verifier.verify(compiled_certificate, latent_state)
        return compiled_certificate.to_dict(), verification.to_dict()

    def __call__(self, task: Task) -> CandidateAnswer:
        circuit = self.registry.select(task)
        if circuit is None:
            if self.fallback is None:
                raise ValueError(f"no compiled frontier circuit registered for skill {task.skill!r}")
            answer = CandidateAnswer.coerce(self.fallback(task))
            return CandidateAnswer(
                answer.text,
                confidence=answer.confidence,
                certificate=dict(answer.certificate),
                cost=answer.cost,
                raw={**dict(answer.raw), "frontier_compiled_selected": False},
            )
        circuit_id = compiled_circuit_id(circuit.report)
        memory_binding = None
        memory_reconstruction = None
        if self.memory is not None:
            try:
                self.memory.bind_compiled_circuit(
                    circuit_id=circuit_id,
                    skill=circuit.skill,
                    source_kind=str(circuit.report.training.get("source_kind", "frontier_discovery")),
                    source_failure_ids=tuple(circuit.report.source_failure_ids),
                    frontier_task_ids=tuple(circuit.report.frontier_task_ids),
                    heldout_task_ids=tuple(circuit.report.heldout_task_ids),
                    prompt_obligations=tuple(circuit.report.invariants.prompt_obligations),
                    metadata_keys=tuple(circuit.report.invariants.metadata_keys),
                    anchors=tuple(
                        anchor
                        for covered in tuple(circuit.verified_tasks) + tuple(circuit.heldout_tasks)
                        for anchor in covered.anchors
                    ),
                )
            except Exception as exc:
                if self.require_memory_binding:
                    raise ValueError(f"compiled Frontier circuit {circuit_id} could not establish a P4 memory binding") from exc
            try:
                memory_binding, memory_reconstruction = self.memory.reconstruct_compiled_circuit_binding(
                    circuit_id,
                    query=f"{task.prompt} compiled circuit {circuit_id} skill {circuit.skill}",
                )
            except KeyError as exc:
                if self.require_memory_binding:
                    raise ValueError(f"compiled Frontier circuit {circuit_id} has no P4 memory binding") from exc
            if memory_reconstruction is not None:
                if (
                    not memory_reconstruction.fidelity.passed
                    or memory_binding is None
                    or memory_binding.segment_id not in memory_reconstruction.selected_segment_ids
                ):
                    if self.require_memory_binding:
                        missing = ", ".join(anchor.value for anchor in memory_reconstruction.fidelity.missing)
                        raise ValueError(
                            f"compiled Frontier circuit {circuit_id} failed P4 memory binding fidelity: missing {missing}"
                        )
        answer = circuit.agent(task)
        certificate = {
            **dict(answer.certificate),
            "frontier_compiled_circuit": True,
            "frontier_skill": circuit.skill,
            "frontier_verified_slow_solutions": circuit.report.verified_slow_solutions,
            "frontier_compiled_weight_bits": circuit.report.compiled_weight_bits,
            "frontier_prompt_obligations": circuit.report.invariants.prompt_obligations,
            "frontier_heldout_passed": int(circuit.report.heldout.get("passed", 0)),
            "frontier_heldout_total": int(circuit.report.heldout.get("total", 0)),
            "frontier_heldout_pass_rate": float(circuit.report.heldout.get("pass_rate", 0.0)),
            "frontier_heldout_gate_passed": bool(circuit.report.heldout.get("gate_passed", False)),
        }
        raw = {
            **dict(answer.raw),
            "frontier_compiled_selected": True,
            "frontier_skill": circuit.skill,
            "frontier_source_failure_ids": circuit.report.source_failure_ids,
            "frontier_task_ids": circuit.report.frontier_task_ids,
            "frontier_heldout_task_ids": circuit.report.heldout_task_ids,
            "frontier_heldout": dict(circuit.report.heldout),
        }
        if memory_binding is not None and memory_reconstruction is not None:
            certificate.update({
                "frontier_memory_binding_id": memory_binding.binding_id,
                "frontier_memory_binding_passed": bool(memory_reconstruction.fidelity.passed),
                "frontier_memory_binding_fidelity": memory_reconstruction.fidelity.score,
                "frontier_memory_selected_segments": memory_reconstruction.selected_segment_ids,
            })
            raw["frontier_memory_binding"] = {
                **memory_binding.to_dict(),
                "runtime_fidelity": asdict(memory_reconstruction.fidelity),
                "runtime_selected_segment_ids": memory_reconstruction.selected_segment_ids,
            }
        cost = answer.cost.merge(CostTrace(verifier_steps=1 if self.verifier is not None and self.verify_outputs else 0))
        if memory_reconstruction is not None:
            cost = cost.merge(memory_reconstruction.cost)
        confidence = answer.confidence
        output_verified = True
        if self.verifier is not None and self.verify_outputs:
            verification = self.verifier.oracle_registry.verify(task.skill, task, answer)
            raw["frontier_compiled_verified"] = verification.passed
            raw["frontier_compiled_verification_reason"] = verification.reason
            certificate["frontier_verification_passed"] = verification.passed
            confidence = min(confidence, verification.score)
            output_verified = verification.passed
        output_goal = self.output_goal_engine.gate_output_goal(
            task,
            answer,
            risk=0.05,
            contract_id=f"frontier-output-goal-{task.task_id}",
            output_verified=output_verified,
        )
        cost = cost.merge(output_goal.cost)
        certificate["frontier_output_goal_contract"] = output_goal.to_dict()
        certificate["frontier_output_goal_contract_passed"] = output_goal.accepted
        raw["frontier_output_goal_contract"] = output_goal.to_dict()
        if not output_goal.accepted:
            confidence = 0.0
        contract_certificate, contract_verification = self._compiled_contract_certificate(
            circuit,
            task,
            answer,
            output_verified=output_verified and output_goal.accepted,
            output_goal=output_goal,
            memory_binding=memory_binding,
            memory_reconstruction=memory_reconstruction,
        )
        certificate["frontier_compiled_contract"] = contract_certificate
        certificate["frontier_compiled_contract_checksum"] = contract_certificate["checksum"]
        certificate["frontier_compiled_contract_verified"] = bool(contract_verification["passed"])
        raw["frontier_compiled_contract_verification"] = contract_verification
        return CandidateAnswer(answer.text, confidence=confidence, certificate=certificate, cost=cost, raw=raw)


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

    def _verified_metamorphic_variants(
        self,
        tasks: Sequence[Task],
        *,
        seed: int,
        per_source: int,
        excluded_task_ids: Iterable[str],
    ) -> tuple[Task, ...]:
        if per_source < 1:
            raise ValueError("per_source must be positive for frontier generalization variants")
        rng = random.Random(seed)
        seen = set(str(task_id) for task_id in excluded_task_ids)
        variants: list[Task] = []
        for task in tasks:
            spec = self.verifier.specs.get(task.skill)
            if spec is None:
                continue
            for variant in spec.metamorphic(task, rng)[:per_source]:
                if variant.task_id in seen:
                    continue
                answer = CandidateAnswer.coerce(self.solver(variant))
                if self.verifier.oracle_registry.verify(variant.skill, variant, answer).passed:
                    variants.append(variant)
                    seen.add(variant.task_id)
        return tuple(variants)

    def compile_sleep_consolidation(
        self,
        sleep_report: Any,
        *,
        seed: int = 0,
        max_skills: int = 2,
        support_per_verified: int = 2,
        heldout_per_support: int = 1,
        min_heldout_pass_rate: float = 1.0,
        max_generalization_rounds: int = 2,
        epochs: int = 80,
        registry: FrontierCircuitRegistry | None = None,
    ) -> FrontierDiscoveryReport:
        if max_skills < 1:
            return FrontierDiscoveryReport(tuple(), tuple(), False)
        if support_per_verified < 1:
            raise ValueError("support_per_verified must be positive")
        if heldout_per_support < 1:
            raise ValueError("heldout_per_support must be positive")
        if max_generalization_rounds < 1:
            raise ValueError("max_generalization_rounds must be positive")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if not 0.0 <= float(min_heldout_pass_rate) <= 1.0:
            raise ValueError("min_heldout_pass_rate must be between 0 and 1")

        raw_examples = tuple(getattr(sleep_report, "accepted_examples", ()))
        sleep_examples_by_task = {
            example.task.task_id: example
            for example in raw_examples
            if getattr(example, "has_trust_label", False)
            and float(getattr(example, "contamination_risk", 1.0)) <= 0.35
        }
        promoted_examples = tuple(
            example
            for example in examples_from_sleep_report(sleep_report)
            if example.task.task_id in sleep_examples_by_task
        )
        scheduled_skills = []
        for item in getattr(sleep_report, "schedule", ()):
            skill = str(getattr(item, "skill", ""))
            if skill and skill not in scheduled_skills:
                scheduled_skills.append(skill)
        if not scheduled_skills:
            counts = Counter(example.task.skill for example in promoted_examples)
            scheduled_skills = [skill for skill, _ in counts.most_common()]
        selected = tuple(scheduled_skills[:max_skills])

        circuits: list[FrontierCompiledCircuit] = []
        trainer = CortexMicroTrainer()
        for index, skill in enumerate(selected):
            skill_promoted = tuple(example for example in promoted_examples if example.task.skill == skill)
            grouped: dict[str, list[MicroTrainingExample]] = {}
            for example in skill_promoted:
                group_key = str(example.task.group_id or f"task:{example.task.task_id}")
                grouped.setdefault(group_key, []).append(example)
            ordered_groups = sorted(
                grouped.values(),
                key=lambda items: (
                    len({item.answer for item in items}) == 1,
                    any(item.task.group_id is not None for item in items),
                    len(items),
                ),
                reverse=True,
            )
            base_examples: list[MicroTrainingExample] = []
            seen_task_ids: set[str] = set()
            source_example_ids: list[str] = []
            for example in (ordered_groups[0] if ordered_groups else ()):
                if example.task.task_id in seen_task_ids:
                    continue
                sleep_example = sleep_examples_by_task[example.task.task_id]
                answer = CandidateAnswer(example.answer, confidence=example.confidence)
                verification = self.verifier.oracle_registry.verify(skill, example.task, answer)
                if not verification.passed:
                    continue
                base_examples.append(example)
                seen_task_ids.add(example.task.task_id)
                source_example_ids.append(str(getattr(sleep_example, "example_id", example.task.task_id)))
            if not base_examples:
                continue

            training_task_list = [example.task for example in base_examples]
            training_examples: list[MicroTrainingExample] = list(base_examples)
            excluded_task_ids = set(seen_task_ids)
            model: CortexMicroModel | None = None
            training = None
            dsv = None
            heldout_tasks: tuple[Task, ...] = ()
            heldout_dsv = None
            heldout_pass_rate = 0.0
            heldout_gate_passed = False
            generalization_rounds: list[dict[str, Any]] = []
            support_added = 0
            for round_idx in range(max_generalization_rounds):
                support_tasks = self._verified_metamorphic_variants(
                    tuple(training_task_list),
                    seed=seed + 6101 + index + round_idx * 997,
                    per_source=support_per_verified,
                    excluded_task_ids=excluded_task_ids,
                )
                if support_tasks:
                    support_examples = examples_from_tasks(support_tasks, self.solver, source="sleep_frontier_support")
                    training_examples.extend(support_examples)
                    training_task_list.extend(support_tasks)
                    support_added += len(support_tasks)
                    excluded_task_ids.update(task.task_id for task in support_tasks)
                training_tasks = tuple(training_task_list)
                model, training = trainer.train(tuple(training_examples), epochs=epochs, lr=0.05)
                agent = MicroModelAgent(model)
                dsv = self.verifier.evaluate_tasks(agent, training_tasks)
                heldout_seed_tasks = support_tasks or training_tasks
                heldout_tasks = self._verified_metamorphic_variants(
                    heldout_seed_tasks,
                    seed=seed + 9151 + index + round_idx * 997,
                    per_source=heldout_per_support,
                    excluded_task_ids=excluded_task_ids,
                )
                heldout_dsv = self.verifier.evaluate_tasks(agent, heldout_tasks)
                heldout_pass_rate = float(heldout_dsv.passed) / max(1, int(heldout_dsv.total))
                heldout_gate_passed = bool(
                    heldout_dsv.total > 0
                    and heldout_dsv.passed == heldout_dsv.total
                    and heldout_pass_rate >= float(min_heldout_pass_rate)
                )
                generalization_rounds.append(
                    {
                        "round": round_idx,
                        "support_tasks": len(support_tasks),
                        "training_tasks": len(training_tasks),
                        "heldout_passed": heldout_dsv.passed,
                        "heldout_total": heldout_dsv.total,
                        "heldout_pass_rate": heldout_pass_rate,
                        "gate_passed": heldout_gate_passed,
                    }
                )
                if heldout_gate_passed:
                    break
                if round_idx + 1 < max_generalization_rounds:
                    retry_tasks = tuple(task for task in heldout_tasks if task.task_id not in excluded_task_ids)
                    if retry_tasks:
                        retry_examples = examples_from_tasks(retry_tasks, self.solver, source="sleep_frontier_retry_heldout")
                        training_examples.extend(retry_examples)
                        training_task_list.extend(retry_tasks)
                        excluded_task_ids.update(task.task_id for task in retry_tasks)
            if model is None or training is None or dsv is None or heldout_dsv is None:
                continue
            training_tasks = tuple(training_task_list)
            compiled_bits, active_weights, total_weights = _compiled_weight_bits(model)
            compiled = FrontierCompiledCircuit(
                skill=skill,
                source_failure_ids=tuple(source_example_ids),
                frontier_task_ids=tuple(task.task_id for task in training_tasks),
                heldout_task_ids=tuple(task.task_id for task in heldout_tasks),
                verified_slow_solutions=len(training_tasks),
                invariants=_extract_invariants(skill, training_tasks),
                training={
                    **training.to_dict(),
                    "source_kind": "sleep_consolidation",
                    "sleep_accepted_examples": len(base_examples),
                    "sleep_support_examples": support_added,
                    "sleep_source_example_ids": tuple(source_example_ids),
                },
                dsv={
                    "passed": dsv.passed,
                    "total": dsv.total,
                    "aggregate_score": dsv.aggregate_score,
                    "verified_capability_per_cost": dsv.verified_capability_per_cost,
                },
                heldout={
                    "passed": heldout_dsv.passed,
                    "total": heldout_dsv.total,
                    "pass_rate": heldout_pass_rate,
                    "min_pass_rate": float(min_heldout_pass_rate),
                    "aggregate_score": heldout_dsv.aggregate_score,
                    "verified_capability_per_cost": heldout_dsv.verified_capability_per_cost,
                    "gate_passed": heldout_gate_passed,
                    "generalization_rounds": tuple(generalization_rounds),
                },
                compiled_weight_bits=compiled_bits,
                active_weights=active_weights,
                total_weights=total_weights,
                passed=dsv.passed == dsv.total and training.after_accuracy >= training.before_accuracy and heldout_gate_passed,
            )
            circuits.append(compiled)
            if registry is not None and compiled.passed:
                registry.register(compiled, model, training_tasks, heldout_tasks=heldout_tasks)
        return FrontierDiscoveryReport(tuple(circuits), selected, bool(circuits) and all(circuit.passed for circuit in circuits))

    def discover(
        self,
        report: CycleReport,
        *,
        seed: int = 0,
        max_skills: int = 2,
        per_failure: int = 4,
        support_per_verified: int = 2,
        heldout_per_support: int = 1,
        min_heldout_pass_rate: float = 1.0,
        max_generalization_rounds: int = 2,
        epochs: int = 120,
        registry: FrontierCircuitRegistry | None = None,
    ) -> FrontierDiscoveryReport:
        if support_per_verified < 1:
            raise ValueError("support_per_verified must be positive")
        if heldout_per_support < 1:
            raise ValueError("heldout_per_support must be positive")
        if max_generalization_rounds < 1:
            raise ValueError("max_generalization_rounds must be positive")
        if not 0.0 <= float(min_heldout_pass_rate) <= 1.0:
            raise ValueError("min_heldout_pass_rate must be between 0 and 1")
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
            candidate_tasks: list[Task] = []
            seen_task_ids: set[str] = set()
            for task in tuple(failure.task for failure in failures) + frontier_tasks:
                if task.task_id in seen_task_ids:
                    continue
                candidate_tasks.append(task)
                seen_task_ids.add(task.task_id)
            base_verified_tasks = self._verified_frontier_tasks(candidate_tasks)
            if not base_verified_tasks:
                continue
            training_task_list = list(base_verified_tasks)
            excluded_task_ids = {task.task_id for task in training_task_list}
            model: CortexMicroModel | None = None
            training = None
            dsv = None
            heldout_tasks: tuple[Task, ...] = ()
            heldout_dsv = None
            heldout_pass_rate = 0.0
            heldout_gate_passed = False
            generalization_rounds: list[dict[str, Any]] = []
            for round_idx in range(max_generalization_rounds):
                support_tasks = self._verified_metamorphic_variants(
                    tuple(training_task_list),
                    seed=seed + 4001 + index + round_idx * 997,
                    per_source=support_per_verified,
                    excluded_task_ids=excluded_task_ids,
                )
                training_task_list.extend(support_tasks)
                excluded_task_ids.update(task.task_id for task in support_tasks)
                training_tasks = tuple(training_task_list)
                examples = examples_from_tasks(training_tasks, self.solver, source="frontier_slow_solve")
                model, training = trainer.train(examples, epochs=epochs, lr=0.05)
                agent = MicroModelAgent(model)
                dsv = self.verifier.evaluate_tasks(agent, training_tasks)
                heldout_seed_tasks = support_tasks or training_tasks
                heldout_tasks = self._verified_metamorphic_variants(
                    heldout_seed_tasks,
                    seed=seed + 7919 + index + round_idx * 997,
                    per_source=heldout_per_support,
                    excluded_task_ids=excluded_task_ids,
                )
                heldout_dsv = self.verifier.evaluate_tasks(agent, heldout_tasks)
                heldout_pass_rate = float(heldout_dsv.passed) / max(1, int(heldout_dsv.total))
                heldout_gate_passed = bool(
                    heldout_dsv.total > 0
                    and heldout_dsv.passed == heldout_dsv.total
                    and heldout_pass_rate >= float(min_heldout_pass_rate)
                )
                generalization_rounds.append(
                    {
                        "round": round_idx,
                        "support_tasks": len(support_tasks),
                        "training_tasks": len(training_tasks),
                        "heldout_passed": heldout_dsv.passed,
                        "heldout_total": heldout_dsv.total,
                        "heldout_pass_rate": heldout_pass_rate,
                        "gate_passed": heldout_gate_passed,
                    }
                )
                if heldout_gate_passed:
                    break
                if round_idx + 1 < max_generalization_rounds:
                    for task in heldout_tasks:
                        if task.task_id not in excluded_task_ids:
                            training_task_list.append(task)
                            excluded_task_ids.add(task.task_id)
            if model is None or training is None or dsv is None or heldout_dsv is None:
                continue
            training_tasks = tuple(training_task_list)
            compiled_bits, active_weights, total_weights = _compiled_weight_bits(model)
            compiled = FrontierCompiledCircuit(
                skill=skill,
                source_failure_ids=tuple(failure.task.task_id for failure in failures),
                frontier_task_ids=tuple(task.task_id for task in training_tasks),
                heldout_task_ids=tuple(task.task_id for task in heldout_tasks),
                verified_slow_solutions=len(training_tasks),
                invariants=_extract_invariants(skill, training_tasks),
                training=training.to_dict(),
                dsv={
                    "passed": dsv.passed,
                    "total": dsv.total,
                    "aggregate_score": dsv.aggregate_score,
                    "verified_capability_per_cost": dsv.verified_capability_per_cost,
                },
                heldout={
                    "passed": heldout_dsv.passed,
                    "total": heldout_dsv.total,
                    "pass_rate": heldout_pass_rate,
                    "min_pass_rate": float(min_heldout_pass_rate),
                    "aggregate_score": heldout_dsv.aggregate_score,
                    "verified_capability_per_cost": heldout_dsv.verified_capability_per_cost,
                    "gate_passed": heldout_gate_passed,
                    "generalization_rounds": tuple(generalization_rounds),
                },
                compiled_weight_bits=compiled_bits,
                active_weights=active_weights,
                total_weights=total_weights,
                passed=dsv.passed == dsv.total and training.after_accuracy >= training.before_accuracy and heldout_gate_passed,
            )
            circuits.append(compiled)
            if registry is not None and compiled.passed:
                registry.register(compiled, model, training_tasks, heldout_tasks=heldout_tasks)
        return FrontierDiscoveryReport(tuple(circuits), selected, bool(circuits) and all(circuit.passed for circuit in circuits))
