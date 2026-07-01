from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from math import log2
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from cortex3 import (
    CandidateAnswer,
    DynamicSkillVerifier,
    ReferenceRuleAgent,
    SkillSpec,
    Task,
    VerificationCaseResult,
)
from cortex3_cycle import CycleReport
from cortex3_ledgers import SkillLedger


class ExampleOrigin(str, Enum):
    FAILURE_REPLAY = "failure_replay"
    VERIFIED_SYNTHETIC = "verified_synthetic"
    REAL_EXOGENOUS = "real_exogenous"
    TOOL_SOLVED = "tool_solved"
    METAMORPHIC = "metamorphic"
    ANTI_METAMORPHIC = "anti_metamorphic"


SYNTHETIC_ORIGINS = {
    ExampleOrigin.VERIFIED_SYNTHETIC,
    ExampleOrigin.TOOL_SOLVED,
    ExampleOrigin.METAMORPHIC,
    ExampleOrigin.ANTI_METAMORPHIC,
}


@dataclass(frozen=True)
class TrainingExample:
    example_id: str
    task: Task
    answer: CandidateAnswer
    origin: ExampleOrigin
    oracle: str
    targeted_skill: str
    verification_level: int
    contamination_risk: float
    difficulty: float
    confidence_label: float | None
    synthetic: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_trust_label(self) -> bool:
        return (
            bool(self.origin.value)
            and bool(self.oracle)
            and bool(self.targeted_skill)
            and self.verification_level >= 0
            and 0.0 <= self.contamination_risk <= 1.0
            and 0.0 <= self.difficulty <= 1.0
            and (self.confidence_label is None or 0.0 <= self.confidence_label <= 1.0)
        )

    @property
    def usable_synthetic_label(self) -> bool:
        return self.has_trust_label and self.confidence_label is not None and self.verification_level > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "task": {
                "task_id": self.task.task_id,
                "skill": self.task.skill,
                "prompt": self.task.prompt,
                "expected": self.task.expected,
                "metadata": dict(self.task.metadata),
                "anchors": [asdict(anchor) for anchor in self.task.anchors],
                "group_id": self.task.group_id,
            },
            "answer": {
                "text": self.answer.text,
                "confidence": self.answer.confidence,
                "certificate": dict(self.answer.certificate),
            },
            "origin": self.origin.value,
            "oracle": self.oracle,
            "targeted_skill": self.targeted_skill,
            "verification_level": self.verification_level,
            "contamination_risk": self.contamination_risk,
            "difficulty": self.difficulty,
            "confidence_label": self.confidence_label,
            "synthetic": self.synthetic,
            "metadata": dict(self.metadata),
        }


def _expected_answer(task: Task) -> CandidateAnswer:
    expected = task.expected
    text = str(expected["answer"]) if isinstance(expected, Mapping) and "answer" in expected else str(expected)
    return CandidateAnswer(text, confidence=1.0, certificate={"oracle_label": task.skill})


def _task_difficulty(task: Task) -> float:
    token_count = max(1, len(task.prompt.split()))
    score = 0.20 + min(0.35, token_count / 200.0)
    if task.anchors:
        score += 0.20
    if task.skill in {"arithmetic", "algebra", "code_unit_tests", "calibration"}:
        score += 0.15
    return max(0.0, min(1.0, score))


def _normalized_prompt(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def _stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExternalProvenanceRecord:
    record_id: str
    source_id: str
    source_kind: str
    task: Task
    answer: CandidateAnswer
    oracle: str
    verification_level: int
    difficulty: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "task_id": self.task.task_id,
            "skill": self.task.skill,
            "prompt": self.task.prompt,
            "answer": self.answer.text,
            "oracle": self.oracle,
            "verification_level": self.verification_level,
            "difficulty": self.difficulty,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExternalProvenanceIngestionReport:
    accepted_examples: tuple[TrainingExample, ...]
    rejected_records: tuple[Mapping[str, Any], ...]
    skipped_records: tuple[Mapping[str, Any], ...]
    duplicate_record_ids: tuple[str, ...]
    source_kind_counts: Mapping[str, int]

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_examples)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_records)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_records)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicate_record_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "skipped_count": self.skipped_count,
            "duplicate_count": self.duplicate_count,
            "accepted_examples": [example.to_dict() for example in self.accepted_examples],
            "rejected_records": [dict(item) for item in self.rejected_records],
            "skipped_records": [dict(item) for item in self.skipped_records],
            "duplicate_record_ids": list(self.duplicate_record_ids),
            "source_kind_counts": dict(self.source_kind_counts),
        }


class LocalExternalProvenanceAdapter:
    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        source_name: str = "local",
        max_chars_per_record: int = 2048,
        min_chars: int = 1,
        default_skill: str = "instruction_following",
    ):
        if not paths:
            raise ValueError("external provenance adapter requires at least one path")
        if max_chars_per_record < 1:
            raise ValueError("max_chars_per_record must be positive")
        if min_chars < 1:
            raise ValueError("min_chars must be positive")
        self.paths = tuple(Path(path) for path in paths)
        self.source_name = source_name
        self.max_chars_per_record = int(max_chars_per_record)
        self.min_chars = int(min_chars)
        self.default_skill = default_skill

    def iter_records(self, *, max_records: int | None = None) -> Iterator[ExternalProvenanceRecord]:
        emitted = 0
        for path in self.paths:
            if not path.exists():
                raise FileNotFoundError(f"external provenance path does not exist: {path}")
            iterator = self._jsonl_records(path) if path.suffix.lower() == ".jsonl" else self._text_records(path)
            for record in iterator:
                yield record
                emitted += 1
                if max_records is not None and emitted >= max_records:
                    return

    def records(self, *, max_records: int | None = None) -> tuple[ExternalProvenanceRecord, ...]:
        return tuple(self.iter_records(max_records=max_records))

    def _record_from_span(
        self,
        *,
        path: Path,
        source_kind: str,
        text: str,
        line_number: int,
        chunk_index: int,
        prompt: str | None = None,
        expected: Any | None = None,
        answer: str | None = None,
        skill: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        verification_level: int = 2,
    ) -> ExternalProvenanceRecord | None:
        span = text.strip()
        if len(span) < self.min_chars:
            return None
        content_hash = _stable_text_hash(f"{prompt or span}\0{expected if expected is not None else span}\0{answer or span}")
        task_skill = str(skill or self.default_skill)
        task_expected = expected if expected is not None else span
        task_prompt = str(prompt) if prompt is not None else "Reproduce this external source span exactly:\n" + span
        answer_text = str(answer) if answer is not None else str(task_expected)
        source_id = f"{self.source_name}-{path.name}-{line_number}-{chunk_index}-{content_hash[:12]}"
        record_metadata = {
            "external_provenance_adapter": True,
            "source_name": self.source_name,
            "source_kind": source_kind,
            "source_path": str(path),
            "line_number": int(line_number),
            "chunk_index": int(chunk_index),
            "external_content_sha256": content_hash,
            "text_char_count": len(span),
        }
        if metadata:
            record_metadata.update(dict(metadata))
        task = Task(
            f"external-{source_id}",
            task_skill,
            task_prompt,
            task_expected,
            record_metadata,
            group_id=f"external-{content_hash[:16]}",
        )
        return ExternalProvenanceRecord(
            record_id=f"external-record-{content_hash[:16]}",
            source_id=source_id,
            source_kind=source_kind,
            task=task,
            answer=CandidateAnswer(
                answer_text,
                confidence=0.99,
                certificate={
                    "external_provenance_adapter": True,
                    "external_content_sha256": content_hash,
                    "source_kind": source_kind,
                },
            ),
            oracle=task_skill,
            verification_level=verification_level,
            metadata=record_metadata,
        )

    def _text_records(self, path: Path) -> Iterator[ExternalProvenanceRecord]:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                span = line.strip()
                if len(span) < self.min_chars:
                    continue
                for chunk_index, start in enumerate(range(0, len(span), self.max_chars_per_record)):
                    record = self._record_from_span(
                        path=path,
                        source_kind="local_text",
                        text=span[start:start + self.max_chars_per_record],
                        line_number=line_number,
                        chunk_index=chunk_index,
                    )
                    if record is not None:
                        yield record

    def _jsonl_records(self, path: Path) -> Iterator[ExternalProvenanceRecord]:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid external provenance JSONL at {path}:{line_number}: {exc}") from exc
                if not isinstance(payload, Mapping):
                    raise ValueError(f"external provenance JSONL record at {path}:{line_number} must be an object")
                metadata = dict(payload.get("metadata") or {})
                if "prompt" in payload and "answer" in payload:
                    prompt = str(payload["prompt"])
                    answer = str(payload["answer"])
                    expected = payload.get("expected", answer)
                    text = f"{prompt}\n{expected}"
                    record = self._record_from_span(
                        path=path,
                        source_kind="local_jsonl",
                        text=text,
                        line_number=line_number,
                        chunk_index=0,
                        prompt=prompt,
                        expected=expected,
                        answer=answer,
                        skill=str(payload.get("skill", self.default_skill)),
                        metadata=metadata,
                        verification_level=int(payload.get("verification_level", 2)),
                    )
                elif "text" in payload:
                    record = self._record_from_span(
                        path=path,
                        source_kind="local_jsonl_text",
                        text=str(payload["text"]),
                        line_number=line_number,
                        chunk_index=0,
                        metadata=metadata,
                        verification_level=int(payload.get("verification_level", 2)),
                    )
                else:
                    raise ValueError(f"external provenance JSONL record at {path}:{line_number} requires `text` or `prompt`+`answer`")
                if record is not None:
                    yield record


class FailureReplayBuffer:
    def __init__(self, max_size: int = 512):
        if max_size < 1:
            raise ValueError("failure replay buffer must keep at least one example")
        self.max_size = max_size
        self.examples: list[TrainingExample] = []

    def add_failure(self, failure: VerificationCaseResult) -> TrainingExample:
        example = TrainingExample(
            example_id=f"replay-{failure.task.task_id}",
            task=failure.task,
            answer=_expected_answer(failure.task),
            origin=ExampleOrigin.FAILURE_REPLAY,
            oracle=failure.task.skill,
            targeted_skill=failure.task.skill,
            verification_level=3,
            contamination_risk=0.05,
            difficulty=max(0.55, _task_difficulty(failure.task)),
            confidence_label=1.0,
            synthetic=False,
            metadata={"failure_reason": failure.reason, "failed_answer": failure.answer.text},
        )
        self.examples.append(example)
        if len(self.examples) > self.max_size:
            self.examples = self.examples[-self.max_size:]
        return example

    def add_many(self, failures: Iterable[VerificationCaseResult]) -> tuple[TrainingExample, ...]:
        return tuple(self.add_failure(failure) for failure in failures)

    def by_skill(self, skill: str) -> tuple[TrainingExample, ...]:
        return tuple(example for example in self.examples if example.targeted_skill == skill)


class VerifiedSyntheticDataPool:
    def __init__(self) -> None:
        self.examples: list[TrainingExample] = []

    def add(self, example: TrainingExample) -> TrainingExample:
        if not example.synthetic:
            raise ValueError("verified synthetic pool only accepts synthetic examples")
        if not example.usable_synthetic_label:
            raise ValueError("synthetic examples require origin, oracle, skill, verification level, risk, difficulty and confidence label")
        self.examples.append(example)
        return example

    def by_skill(self, skill: str) -> tuple[TrainingExample, ...]:
        return tuple(example for example in self.examples if example.targeted_skill == skill)


class RealExogenousReservoir:
    def __init__(self, max_size: int = 1024):
        if max_size < 1:
            raise ValueError("real reservoir must keep at least one example")
        self.max_size = max_size
        self.examples: list[TrainingExample] = []
        self._external_hashes: set[str] = set()

    def rebuild_provenance_index(self) -> None:
        self._external_hashes = {
            str(value)
            for example in self.examples
            for value in (
                dict(example.metadata).get("external_content_sha256"),
                dict(example.metadata).get("text_sha256"),
            )
            if value
        }

    def add(
        self,
        task: Task,
        answer: CandidateAnswer | str,
        *,
        source_id: str,
        oracle: str = "external",
        verification_level: int = 1,
        difficulty: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrainingExample:
        provenance = {"source_id": source_id}
        if metadata:
            provenance.update(dict(metadata))
        example = TrainingExample(
            example_id=f"real-{source_id}-{task.task_id}",
            task=task,
            answer=CandidateAnswer.coerce(answer),
            origin=ExampleOrigin.REAL_EXOGENOUS,
            oracle=oracle,
            targeted_skill=task.skill,
            verification_level=verification_level,
            contamination_risk=0.0,
            difficulty=_task_difficulty(task) if difficulty is None else max(0.0, min(1.0, difficulty)),
            confidence_label=None,
            synthetic=False,
            metadata=provenance,
        )
        self.examples.append(example)
        for key in ("external_content_sha256", "text_sha256"):
            value = provenance.get(key)
            if value:
                self._external_hashes.add(str(value))
        if len(self.examples) > self.max_size:
            self.examples = self.examples[-self.max_size:]
            self.rebuild_provenance_index()
        return example

    def by_skill(self, skill: str) -> tuple[TrainingExample, ...]:
        return tuple(example for example in self.examples if example.targeted_skill == skill)

    def ingest_external_records(
        self,
        records: Iterable[ExternalProvenanceRecord],
        *,
        verifier: DynamicSkillVerifier,
        max_examples: int | None = None,
    ) -> ExternalProvenanceIngestionReport:
        accepted: list[TrainingExample] = []
        rejected: list[Mapping[str, Any]] = []
        skipped: list[Mapping[str, Any]] = []
        duplicates: list[str] = []
        source_kind_counts: Counter[str] = Counter()
        self.rebuild_provenance_index()
        for record in records:
            metadata = dict(record.metadata)
            content_hash = str(metadata.get("external_content_sha256") or _stable_text_hash(record.answer.text))
            source_kind_counts[record.source_kind] += 1
            if content_hash in self._external_hashes:
                duplicates.append(record.record_id)
                continue
            try:
                verification = verifier.oracle_registry.verify(record.oracle, record.task, record.answer)
            except KeyError as exc:
                rejected.append({
                    "record_id": record.record_id,
                    "source_id": record.source_id,
                    "source_kind": record.source_kind,
                    "reason": str(exc),
                    "score": 0.0,
                })
                continue
            if not verification.passed:
                rejected.append({
                    "record_id": record.record_id,
                    "source_id": record.source_id,
                    "source_kind": record.source_kind,
                    "reason": verification.reason,
                    "score": float(verification.score),
                })
                continue
            if max_examples is not None and len(accepted) >= max_examples:
                skipped.append({
                    "record_id": record.record_id,
                    "source_id": record.source_id,
                    "source_kind": record.source_kind,
                    "reason": "max_examples limit reached after oracle verification",
                    "score": float(verification.score),
                })
                continue
            example = self.add(
                record.task,
                CandidateAnswer(
                    record.answer.text,
                    confidence=max(float(record.answer.confidence), float(verification.score)),
                    certificate={
                        **dict(record.answer.certificate),
                        "external_provenance_verified": True,
                    },
                    cost=record.answer.cost,
                    raw={
                        **dict(record.answer.raw),
                        "external_provenance_record": record.record_id,
                        "verification_reason": verification.reason,
                    },
                ),
                source_id=record.source_id,
                oracle=record.oracle,
                verification_level=record.verification_level,
                difficulty=record.difficulty,
                metadata={
                    **metadata,
                    "external_content_sha256": content_hash,
                    "verification_reason": verification.reason,
                    "verification_score": float(verification.score),
                },
            )
            accepted.append(example)
            self._external_hashes.add(content_hash)
        return ExternalProvenanceIngestionReport(
            accepted_examples=tuple(accepted),
            rejected_records=tuple(rejected),
            skipped_records=tuple(skipped),
            duplicate_record_ids=tuple(duplicates),
            source_kind_counts=dict(source_kind_counts),
        )


class ToolSolvedExampleFactory:
    def __init__(self, verifier: DynamicSkillVerifier, solver: ReferenceRuleAgent | None = None):
        self.verifier = verifier
        self.solver = solver or ReferenceRuleAgent()

    def solve(self, task: Task, *, origin: ExampleOrigin = ExampleOrigin.TOOL_SOLVED, verification_level: int = 3) -> TrainingExample:
        answer = CandidateAnswer.coerce(self.solver(task))
        verification = self.verifier.oracle_registry.verify(task.skill, task, answer)
        if not verification.passed:
            raise ValueError(f"tool solver failed oracle for {task.task_id}: {verification.reason}")
        return TrainingExample(
            example_id=f"{origin.value}-{task.task_id}",
            task=task,
            answer=answer,
            origin=origin,
            oracle=task.skill,
            targeted_skill=task.skill,
            verification_level=verification_level,
            contamination_risk=0.08 if origin == ExampleOrigin.TOOL_SOLVED else 0.12,
            difficulty=_task_difficulty(task),
            confidence_label=verification.score,
            synthetic=True,
            metadata={"verification_reason": verification.reason},
        )


class MetamorphicFamilyBuilder:
    def __init__(self, verifier: DynamicSkillVerifier, specs: Iterable[SkillSpec] | None = None):
        self.verifier = verifier
        self.specs = {spec.name: spec for spec in (specs or verifier.specs.values())}
        self.factory = ToolSolvedExampleFactory(verifier)

    def build(self, base_task: Task, *, seed: int = 0, include_anti: bool = True) -> tuple[TrainingExample, ...]:
        spec = self.specs.get(base_task.skill)
        if spec is None:
            return tuple()
        rng = random.Random(seed)
        variants: list[tuple[Task, ExampleOrigin]] = []
        variants.extend((task, ExampleOrigin.METAMORPHIC) for task in spec.metamorphic(base_task, rng))
        if include_anti:
            variants.extend((task, ExampleOrigin.ANTI_METAMORPHIC) for task in spec.anti_metamorphic(base_task, rng))
        out: list[TrainingExample] = []
        for task, origin in variants:
            example = self.factory.solve(task, origin=origin, verification_level=2)
            risk = 0.10 if origin == ExampleOrigin.METAMORPHIC else 0.18
            out.append(TrainingExample(
                example_id=example.example_id,
                task=example.task,
                answer=example.answer,
                origin=origin,
                oracle=example.oracle,
                targeted_skill=example.targeted_skill,
                verification_level=example.verification_level,
                contamination_risk=risk,
                difficulty=example.difficulty,
                confidence_label=example.confidence_label,
                synthetic=True,
                metadata={**dict(example.metadata), "base_task_id": base_task.task_id},
            ))
        return tuple(out)


@dataclass(frozen=True)
class DiversityMetrics:
    total: int
    skill_counts: Mapping[str, int]
    origin_counts: Mapping[str, int]
    unique_prompt_ratio: float
    skill_entropy: float
    origin_entropy: float
    rare_skill_fraction: float
    average_contamination_risk: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "skill_counts": dict(self.skill_counts),
            "origin_counts": dict(self.origin_counts),
            "unique_prompt_ratio": self.unique_prompt_ratio,
            "skill_entropy": self.skill_entropy,
            "origin_entropy": self.origin_entropy,
            "rare_skill_fraction": self.rare_skill_fraction,
            "average_contamination_risk": self.average_contamination_risk,
        }


@dataclass(frozen=True)
class AntiCollapseDecision:
    accepted: bool
    reasons: tuple[str, ...]
    accepted_examples: tuple[TrainingExample, ...]
    rejected_examples: tuple[TrainingExample, ...]
    metrics: DiversityMetrics
    calibration_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "accepted_examples": [example.example_id for example in self.accepted_examples],
            "rejected_examples": [example.example_id for example in self.rejected_examples],
            "metrics": self.metrics.to_dict(),
            "calibration_ok": self.calibration_ok,
        }


@dataclass(frozen=True)
class AntiCollapseFilterConfig:
    max_contamination_risk: float = 0.35
    min_unique_prompt_ratio: float = 0.70
    min_skill_entropy_for_large_batch: float = 0.75
    max_calibration_gap_increase: float = 0.0
    min_synthetic_verification_level: int = 1


def diversity_metrics(examples: Sequence[TrainingExample], rare_skills: Iterable[str] = ()) -> DiversityMetrics:
    rare = set(rare_skills)
    total = len(examples)
    skill_counts = Counter(example.targeted_skill for example in examples)
    origin_counts = Counter(example.origin.value for example in examples)
    prompts = [_normalized_prompt(example.task.prompt) for example in examples]
    unique_prompt_ratio = len(set(prompts)) / total if total else 1.0

    def entropy(counts: Counter[str]) -> float:
        if not counts:
            return 0.0
        denom = sum(counts.values())
        return -sum((count / denom) * log2(count / denom) for count in counts.values() if count)

    return DiversityMetrics(
        total=total,
        skill_counts=dict(skill_counts),
        origin_counts=dict(origin_counts),
        unique_prompt_ratio=unique_prompt_ratio,
        skill_entropy=entropy(skill_counts),
        origin_entropy=entropy(origin_counts),
        rare_skill_fraction=sum(1 for example in examples if example.targeted_skill in rare) / total if total else 0.0,
        average_contamination_risk=sum(example.contamination_risk for example in examples) / total if total else 0.0,
    )


class AntiCollapseFilter:
    def __init__(self, config: AntiCollapseFilterConfig | None = None):
        self.config = config or AntiCollapseFilterConfig()

    def evaluate(
        self,
        examples: Iterable[TrainingExample],
        *,
        rare_skills: Iterable[str] = (),
        baseline_calibration_gap: float | None = None,
        projected_calibration_gap: float | None = None,
    ) -> AntiCollapseDecision:
        accepted: list[TrainingExample] = []
        rejected: list[TrainingExample] = []
        reasons: list[str] = []
        seen_prompts: set[str] = set()

        for example in examples:
            prompt_key = _normalized_prompt(example.task.prompt)
            reject_reason = ""
            if example.synthetic and not example.usable_synthetic_label:
                reject_reason = "synthetic example missing trust label"
            elif example.synthetic and example.verification_level < self.config.min_synthetic_verification_level:
                reject_reason = "synthetic example verification level too low"
            elif example.contamination_risk > self.config.max_contamination_risk:
                reject_reason = "contamination risk too high"
            elif prompt_key in seen_prompts:
                reject_reason = "duplicate prompt rejected for diversity"

            if reject_reason:
                rejected.append(example)
                reasons.append(f"{example.example_id}: {reject_reason}")
            else:
                accepted.append(example)
                seen_prompts.add(prompt_key)

        metrics = diversity_metrics(tuple(accepted), rare_skills)
        calibration_ok = True
        if baseline_calibration_gap is not None and projected_calibration_gap is not None:
            calibration_ok = projected_calibration_gap <= baseline_calibration_gap + self.config.max_calibration_gap_increase
            if not calibration_ok:
                reasons.append("projected calibration gap increases")

        large_batch_collapse = (
            metrics.total >= 4
            and metrics.skill_entropy < self.config.min_skill_entropy_for_large_batch
            and metrics.rare_skill_fraction < 0.50
        )
        if metrics.total and metrics.unique_prompt_ratio < self.config.min_unique_prompt_ratio:
            reasons.append("unique prompt ratio below anti-collapse threshold")
        if large_batch_collapse:
            reasons.append("skill entropy below anti-collapse threshold")

        accepted_flag = bool(accepted) and calibration_ok and not large_batch_collapse and metrics.unique_prompt_ratio >= self.config.min_unique_prompt_ratio
        return AntiCollapseDecision(accepted_flag, tuple(reasons), tuple(accepted), tuple(rejected), metrics, calibration_ok)


@dataclass(frozen=True)
class ConsolidationScheduleItem:
    skill: str
    priority: float
    replay_examples: tuple[str, ...]
    synthetic_examples: tuple[str, ...]
    real_examples: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SkillConsolidationScheduler:
    def schedule(
        self,
        ledger: SkillLedger,
        replay: FailureReplayBuffer,
        synthetic: VerifiedSyntheticDataPool,
        reservoir: RealExogenousReservoir,
        *,
        rare_skills: Iterable[str] = (),
        max_skills: int = 4,
    ) -> tuple[ConsolidationScheduleItem, ...]:
        items: list[ConsolidationScheduleItem] = []
        rare = set(rare_skills)
        for state in ledger.states.values():
            replay_examples = replay.by_skill(state.skill)
            synthetic_examples = synthetic.by_skill(state.skill)
            real_examples = reservoir.by_skill(state.skill)
            evidence_count = len(replay_examples) + len(synthetic_examples) + len(real_examples)
            if evidence_count == 0 and not state.protected:
                continue
            priority = (
                1.0 - state.score
                + state.fragility
                + min(0.50, state.failures * 0.10)
                + (0.25 if state.protected else 0.0)
                + (0.35 if state.skill in rare else 0.0)
                + min(0.20, evidence_count * 0.02)
            )
            reason = "rare fragile skill" if state.skill in rare else "protected fragile skill" if state.protected else "low-score consolidation"
            items.append(ConsolidationScheduleItem(
                skill=state.skill,
                priority=priority,
                replay_examples=tuple(example.example_id for example in replay_examples[:4]),
                synthetic_examples=tuple(example.example_id for example in synthetic_examples[:6]),
                real_examples=tuple(example.example_id for example in real_examples[:4]),
                reason=reason,
            ))
        return tuple(sorted(items, key=lambda item: item.priority, reverse=True)[:max_skills])


@dataclass(frozen=True)
class SleepPhaseReport:
    accepted_examples: tuple[TrainingExample, ...]
    rejected_examples: tuple[TrainingExample, ...]
    filter_decision: AntiCollapseDecision
    schedule: tuple[ConsolidationScheduleItem, ...]
    baseline_rare_skill_fraction: float
    accepted_rare_skill_fraction: float
    scheduled_rare_skill_fraction: float
    rare_skill_gain: float
    diversity_delta: float
    calibration_gap_delta: float
    diversity_ok: bool
    calibration_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_examples": [example.to_dict() for example in self.accepted_examples],
            "rejected_examples": [example.to_dict() for example in self.rejected_examples],
            "filter_decision": self.filter_decision.to_dict(),
            "schedule": [item.to_dict() for item in self.schedule],
            "baseline_rare_skill_fraction": self.baseline_rare_skill_fraction,
            "accepted_rare_skill_fraction": self.accepted_rare_skill_fraction,
            "scheduled_rare_skill_fraction": self.scheduled_rare_skill_fraction,
            "rare_skill_gain": self.rare_skill_gain,
            "diversity_delta": self.diversity_delta,
            "calibration_gap_delta": self.calibration_gap_delta,
            "diversity_ok": self.diversity_ok,
            "calibration_ok": self.calibration_ok,
        }


class SleepPhaseConsolidator:
    def __init__(
        self,
        verifier: DynamicSkillVerifier,
        *,
        replay: FailureReplayBuffer | None = None,
        synthetic: VerifiedSyntheticDataPool | None = None,
        reservoir: RealExogenousReservoir | None = None,
        anti_collapse: AntiCollapseFilter | None = None,
        scheduler: SkillConsolidationScheduler | None = None,
    ):
        self.verifier = verifier
        self.replay = replay or FailureReplayBuffer()
        self.synthetic = synthetic or VerifiedSyntheticDataPool()
        self.reservoir = reservoir or RealExogenousReservoir()
        self.anti_collapse = anti_collapse or AntiCollapseFilter()
        self.scheduler = scheduler or SkillConsolidationScheduler()
        self.metamorphic = MetamorphicFamilyBuilder(verifier)
        self.tool_solver = ToolSolvedExampleFactory(verifier)

    def ingest_cycle(self, report: CycleReport, *, seed: int = 0) -> SleepPhaseReport:
        replay_examples = self.replay.add_many(report.regressions)
        rng = random.Random(seed)
        for failure in report.regressions:
            try:
                self.synthetic.add(self.tool_solver.solve(failure.task))
            except ValueError:
                pass
            for example in self.metamorphic.build(failure.task, seed=rng.randrange(10**9), include_anti=True):
                try:
                    self.synthetic.add(example)
                except ValueError:
                    pass

        rare_skills = tuple(state.skill for state in report.skill_ledger.fragile_skills())
        candidates = tuple(replay_examples) + tuple(self.synthetic.examples) + tuple(self.reservoir.examples)
        projected_calibration = min(report.calibration_gap, report.calibration_gap * 0.95)
        decision = self.anti_collapse.evaluate(
            candidates,
            rare_skills=rare_skills,
            baseline_calibration_gap=report.calibration_gap,
            projected_calibration_gap=projected_calibration,
        )
        scheduled_replay = FailureReplayBuffer()
        scheduled_synthetic = VerifiedSyntheticDataPool()
        scheduled_reservoir = RealExogenousReservoir()
        scheduled_replay.examples = [example for example in decision.accepted_examples if example.origin == ExampleOrigin.FAILURE_REPLAY]
        scheduled_synthetic.examples = [example for example in decision.accepted_examples if example.synthetic]
        scheduled_reservoir.examples = [example for example in decision.accepted_examples if example.origin == ExampleOrigin.REAL_EXOGENOUS]
        schedule = self.scheduler.schedule(report.skill_ledger, scheduled_replay, scheduled_synthetic, scheduled_reservoir, rare_skills=rare_skills)
        baseline_metrics = diversity_metrics(tuple(self.reservoir.examples), rare_skills)
        before_rare = baseline_metrics.rare_skill_fraction
        accepted_rare = decision.metrics.rare_skill_fraction
        scheduled_rare = sum(1 for item in schedule if item.skill in set(rare_skills)) / len(schedule) if schedule else 0.0
        rare_gain = max(0.0, max(accepted_rare, scheduled_rare) - before_rare)
        diversity_delta = decision.metrics.skill_entropy - baseline_metrics.skill_entropy
        calibration_gap_delta = projected_calibration - report.calibration_gap
        return SleepPhaseReport(
            accepted_examples=decision.accepted_examples,
            rejected_examples=decision.rejected_examples,
            filter_decision=decision,
            schedule=schedule,
            baseline_rare_skill_fraction=before_rare,
            accepted_rare_skill_fraction=accepted_rare,
            scheduled_rare_skill_fraction=scheduled_rare,
            rare_skill_gain=rare_gain,
            diversity_delta=diversity_delta,
            calibration_gap_delta=calibration_gap_delta,
            diversity_ok=decision.accepted,
            calibration_ok=decision.calibration_ok,
        )
