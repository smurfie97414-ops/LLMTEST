from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex3 import CandidateAnswer, CostTrace, ReferenceRuleAgent, Task
from cortex3_memory import embed_text
from cortex3_ternary import BitLinear, BitLinearConfig, CompressionTraceLedger


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_NON_LABEL_METADATA_KEYS = {
    "a",
    "b",
    "c",
    "low",
    "high",
    "repeat",
    "value",
}


def _bounded_feature(value: float) -> float:
    return max(-4.0, min(4.0, float(value) / 100.0))


def task_features(task: Task, input_dim: int) -> torch.Tensor:
    features = embed_text(task.prompt, input_dim).to(torch.float32).clone()
    structured: list[float] = []
    for match in _NUMBER_RE.finditer(task.prompt):
        try:
            structured.append(_bounded_feature(float(match.group(0))))
        except ValueError:
            continue
        if len(structured) >= 16:
            break
    for key in sorted(_NON_LABEL_METADATA_KEYS & set(task.metadata)):
        value = task.metadata.get(key)
        if isinstance(value, bool):
            structured.append(1.0 if value else -1.0)
        elif isinstance(value, (int, float)):
            structured.append(_bounded_feature(float(value)))
    for idx, value in enumerate(structured[:input_dim]):
        features[idx] = features[idx] + float(value)
    return features


@dataclass(frozen=True)
class MicroTrainingExample:
    task: Task
    answer: str
    confidence: float
    source: str = "reference"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "skill": self.task.skill,
            "answer": self.answer,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass(frozen=True)
class MicroVocabulary:
    answers: tuple[str, ...]
    skills: tuple[str, ...]

    @staticmethod
    def from_examples(examples: Sequence[MicroTrainingExample]) -> "MicroVocabulary":
        answers = tuple(sorted({example.answer for example in examples}))
        skills = tuple(sorted({example.task.skill for example in examples}))
        if not answers:
            raise ValueError("micro vocabulary requires at least one answer")
        if not skills:
            raise ValueError("micro vocabulary requires at least one skill")
        return MicroVocabulary(answers, skills)

    def answer_index(self, answer: str) -> int:
        try:
            return self.answers.index(answer)
        except ValueError as exc:
            raise KeyError(f"answer {answer!r} is not in micro vocabulary") from exc

    def skill_index(self, skill: str) -> int:
        try:
            return self.skills.index(skill)
        except ValueError as exc:
            raise KeyError(f"skill {skill!r} is not in micro vocabulary") from exc

    def to_dict(self) -> dict[str, Any]:
        return {"answers": list(self.answers), "skills": list(self.skills)}


@dataclass(frozen=True)
class MicroModelConfig:
    input_dim: int = 64
    hidden_size: int = 64
    activation_bits: int = 4

    def __post_init__(self) -> None:
        if self.input_dim < 8 or self.hidden_size < 8:
            raise ValueError("micro model dimensions must be at least 8")


@dataclass(frozen=True)
class MicroTrainingResult:
    epochs: int
    before_accuracy: float
    after_accuracy: float
    before_loss: float
    after_loss: float
    examples: int
    checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CortexMicroModel(nn.Module):
    def __init__(self, config: MicroModelConfig, vocabulary: MicroVocabulary, ledger: CompressionTraceLedger | None = None):
        super().__init__()
        torch.manual_seed(2026)
        self.config = config
        self.vocabulary = vocabulary
        self.ledger = ledger or CompressionTraceLedger()
        self.input_projection = nn.Linear(config.input_dim, config.hidden_size)
        self.compiled_core = BitLinear(
            BitLinearConfig(config.hidden_size, config.hidden_size, activation_bits=config.activation_bits, log_prefix="micro-compiled-core"),
            ledger=self.ledger,
        )
        self.answer_head = nn.Linear(config.hidden_size, len(vocabulary.answers))
        self.skill_head = nn.Linear(config.hidden_size, len(vocabulary.skills))
        self.confidence_head = nn.Linear(config.hidden_size, 1)

    def forward(self, features: torch.Tensor) -> Mapping[str, torch.Tensor]:
        hidden = torch.tanh(self.input_projection(features))
        hidden = torch.tanh(self.compiled_core(hidden))
        return {
            "answer_logits": self.answer_head(hidden),
            "skill_logits": self.skill_head(hidden),
            "confidence": torch.sigmoid(self.confidence_head(hidden)).squeeze(-1),
        }

    def requantize_core(self) -> None:
        self.compiled_core.requantize(certify_zeros=True)


def examples_from_tasks(tasks: Iterable[Task], solver: Callable[[Task], CandidateAnswer | str] | None = None, *, source: str = "reference") -> tuple[MicroTrainingExample, ...]:
    resolved_solver = solver or ReferenceRuleAgent()
    out: list[MicroTrainingExample] = []
    for task in tasks:
        answer = CandidateAnswer.coerce(resolved_solver(task))
        out.append(MicroTrainingExample(task, answer.text, answer.confidence, source))
    return tuple(out)


def examples_from_sleep_report(sleep_report: Any) -> tuple[MicroTrainingExample, ...]:
    out: list[MicroTrainingExample] = []
    for example in getattr(sleep_report, "accepted_examples", ()):
        task = getattr(example, "task")
        answer = CandidateAnswer.coerce(getattr(example, "answer"))
        out.append(MicroTrainingExample(task, answer.text, answer.confidence, f"sleep:{getattr(example, 'origin').value}"))
    return tuple(out)


class MicroDataset:
    def __init__(self, examples: Sequence[MicroTrainingExample], vocabulary: MicroVocabulary, config: MicroModelConfig):
        if not examples:
            raise ValueError("micro dataset cannot be empty")
        self.examples = tuple(examples)
        self.vocabulary = vocabulary
        self.config = config

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.stack([task_features(example.task, self.config.input_dim) for example in self.examples]).to(torch.float32)
        answer_targets = torch.tensor([self.vocabulary.answer_index(example.answer) for example in self.examples], dtype=torch.long)
        skill_targets = torch.tensor([self.vocabulary.skill_index(example.task.skill) for example in self.examples], dtype=torch.long)
        confidence_targets = torch.tensor([max(0.0, min(1.0, example.confidence)) for example in self.examples], dtype=torch.float32)
        return features, answer_targets, skill_targets, confidence_targets


class CortexMicroTrainer:
    def __init__(self, config: MicroModelConfig | None = None):
        self.config = config or MicroModelConfig()

    def _loss(self, model: CortexMicroModel, dataset: MicroDataset) -> tuple[torch.Tensor, float]:
        features, answer_targets, skill_targets, confidence_targets = dataset.tensors()
        output = model(features)
        answer_loss = F.cross_entropy(output["answer_logits"], answer_targets)
        skill_loss = F.cross_entropy(output["skill_logits"], skill_targets)
        confidence_loss = F.mse_loss(output["confidence"], confidence_targets)
        loss = answer_loss + 0.20 * skill_loss + 0.10 * confidence_loss
        predicted = output["answer_logits"].argmax(dim=-1)
        accuracy = float((predicted == answer_targets).to(torch.float32).mean().item())
        return loss, accuracy

    def train(
        self,
        examples: Sequence[MicroTrainingExample],
        *,
        epochs: int = 250,
        lr: float = 0.05,
        checkpoint_path: str | Path | None = None,
    ) -> tuple[CortexMicroModel, MicroTrainingResult]:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        vocabulary = MicroVocabulary.from_examples(examples)
        model = CortexMicroModel(self.config, vocabulary)
        dataset = MicroDataset(examples, vocabulary, self.config)
        before_loss, before_accuracy = self._loss(model, dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss, _ = self._loss(model, dataset)
            loss.backward()
            optimizer.step()
        model.requantize_core()
        after_loss, after_accuracy = self._loss(model, dataset)
        saved_path = None
        if checkpoint_path is not None:
            saved_path = str(CheckpointManager().save(model, checkpoint_path))
        return model, MicroTrainingResult(
            epochs=epochs,
            before_accuracy=before_accuracy,
            after_accuracy=after_accuracy,
            before_loss=float(before_loss.detach().item()),
            after_loss=float(after_loss.detach().item()),
            examples=len(examples),
            checkpoint_path=saved_path,
        )


class MicroModelAgent:
    def __init__(self, model: CortexMicroModel):
        self.model = model
        self.model.eval()

    def __call__(self, task: Task) -> CandidateAnswer:
        with torch.no_grad():
            features = task_features(task, self.model.config.input_dim).view(1, self.model.config.input_dim)
            output = self.model(features)
            index = int(output["answer_logits"].argmax(dim=-1).item())
            confidence = float(output["confidence"].item())
        answer = self.model.vocabulary.answers[index]
        cost = CostTrace(
            weight_bits_read=self.model.ledger.cost_trace.weight_bits_read,
            activation_bits=self.model.ledger.cost_trace.activation_bits,
            generated_tokens=max(1, len(answer.split())),
            verifier_steps=0,
        )
        return CandidateAnswer(answer, confidence=confidence, certificate={"micro_model": "trained"}, cost=cost)


class CheckpointManager:
    def save(self, model: CortexMicroModel, path: str | Path) -> Path:
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(model.config),
                "vocabulary": model.vocabulary.to_dict(),
                "state_dict": model.state_dict(),
            },
            resolved,
        )
        return resolved

    def load(self, path: str | Path) -> CortexMicroModel:
        payload = torch.load(Path(path), map_location="cpu")
        config = MicroModelConfig(**payload["config"])
        vocabulary = MicroVocabulary(tuple(payload["vocabulary"]["answers"]), tuple(payload["vocabulary"]["skills"]))
        model = CortexMicroModel(config, vocabulary)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
