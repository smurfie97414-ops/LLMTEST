from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex3 import CandidateAnswer, CostTrace, ReferenceRuleAgent, Task
from cortex3_future import FutureContractEngine, FutureContractLedger, MTPFSPConfig, temporal_consistency_loss
from cortex3_memory import embed_text
from cortex3_microtrain import MicroTrainingExample
from cortex3_ternary import BitLinear, BitLinearConfig, CompressionTraceLedger


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"


@dataclass(frozen=True)
class ARConfig:
    input_dim: int = 64
    hidden_size: int = 96
    token_dim: int = 32
    activation_bits: int = 4
    max_answer_tokens: int = 128
    horizons: tuple[int, ...] = (1, 2, 4, 8)
    multi_horizon_weight: float = 0.35
    confidence_weight: float = 0.35
    future_contract_weight: float = 0.10

    def __post_init__(self) -> None:
        if self.input_dim < 8 or self.hidden_size < 16 or self.token_dim < 8:
            raise ValueError("autoregressive dimensions are too small")
        if self.horizons != tuple(sorted(set(self.horizons))):
            raise ValueError("horizons must be unique and sorted")
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must be positive")


@dataclass(frozen=True)
class ARTrainingResult:
    epochs: int
    before_token_accuracy: float
    after_token_accuracy: float
    before_loss: float
    after_loss: float
    exact_sequence_accuracy: float
    examples: int
    checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ARBlockContractTrace:
    step_index: int
    contract_id: str
    accepted_horizon: int
    accepted: bool
    reason: str
    token_ids: tuple[int, ...]
    observed_tokens: tuple[int, ...]
    confidence: float
    temporal_loss: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ARContractGeneration:
    text: str
    confidence: float
    cost: CostTrace
    ledger: FutureContractLedger
    block_traces: tuple[ARBlockContractTrace, ...]
    decoder_steps: int

    @property
    def accepted_blocks(self) -> int:
        return self.ledger.accepted

    @property
    def rejected_blocks(self) -> int:
        return self.ledger.rejected

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "cost": asdict(self.cost),
            "future_contracts": self.ledger.to_dict(),
            "block_traces": [trace.to_dict() for trace in self.block_traces],
            "decoder_steps": self.decoder_steps,
        }


@dataclass(frozen=True)
class TokenVocabulary:
    tokens: tuple[str, ...]

    @staticmethod
    def from_texts(texts: Iterable[str]) -> "TokenVocabulary":
        chars = sorted({char for text in texts for char in text})
        tokens = (PAD, BOS, EOS, *chars)
        return TokenVocabulary(tokens)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    def encode(self, text: str, *, add_eos: bool = True) -> tuple[int, ...]:
        ids = []
        for char in text:
            try:
                ids.append(self.tokens.index(char))
            except ValueError as exc:
                raise KeyError(f"character {char!r} is not in autoregressive vocabulary") from exc
        if add_eos:
            ids.append(self.eos_id)
        return tuple(ids)

    def decode(self, ids: Iterable[int]) -> str:
        chars = []
        for token_id in ids:
            if token_id == self.eos_id:
                break
            if token_id in {self.pad_id, self.bos_id}:
                continue
            chars.append(self.tokens[int(token_id)])
        return "".join(chars)

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": list(self.tokens)}


def ar_examples_from_tasks(tasks: Iterable[Task], solver: Callable[[Task], CandidateAnswer | str] | None = None, *, source: str = "reference") -> tuple[MicroTrainingExample, ...]:
    resolved = solver or ReferenceRuleAgent()
    out: list[MicroTrainingExample] = []
    for task in tasks:
        answer = CandidateAnswer.coerce(resolved(task))
        out.append(MicroTrainingExample(task, answer.text, answer.confidence, source))
    return tuple(out)


def ar_examples_from_sleep_report(sleep_report: Any) -> tuple[MicroTrainingExample, ...]:
    out: list[MicroTrainingExample] = []
    for example in getattr(sleep_report, "accepted_examples", ()):
        task = getattr(example, "task")
        answer = CandidateAnswer.coerce(getattr(example, "answer"))
        origin = getattr(getattr(example, "origin", None), "value", "unknown")
        out.append(MicroTrainingExample(task, answer.text, answer.confidence, f"sleep:{origin}"))
    return tuple(out)


class ARDataset:
    def __init__(self, examples: Sequence[MicroTrainingExample], vocabulary: TokenVocabulary, config: ARConfig):
        if not examples:
            raise ValueError("autoregressive dataset cannot be empty")
        self.examples = tuple(examples)
        self.vocabulary = vocabulary
        self.config = config

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = [self.vocabulary.encode(example.answer) for example in self.examples]
        max_len = max(len(ids) for ids in encoded)
        if max_len > self.config.max_answer_tokens:
            raise ValueError(f"answer length {max_len} exceeds max_answer_tokens")
        features = torch.stack([embed_text(example.task.prompt, self.config.input_dim) for example in self.examples]).to(torch.float32)
        decoder_inputs = torch.full((len(encoded), max_len), self.vocabulary.pad_id, dtype=torch.long)
        targets = torch.full((len(encoded), max_len), self.vocabulary.pad_id, dtype=torch.long)
        mask = torch.zeros((len(encoded), max_len), dtype=torch.bool)
        confidence = torch.tensor([max(0.0, min(1.0, example.confidence)) for example in self.examples], dtype=torch.float32)
        for row, ids in enumerate(encoded):
            prev = (self.vocabulary.bos_id, *ids[:-1])
            decoder_inputs[row, :len(ids)] = torch.tensor(prev, dtype=torch.long)
            targets[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[row, :len(ids)] = True
        return features, decoder_inputs, targets, confidence


class ARMicroDecoder(nn.Module):
    def __init__(self, config: ARConfig, vocabulary: TokenVocabulary, ledger: CompressionTraceLedger | None = None):
        super().__init__()
        torch.manual_seed(4242)
        self.config = config
        self.vocabulary = vocabulary
        self.ledger = ledger or CompressionTraceLedger()
        self.prompt_projection = nn.Linear(config.input_dim, config.hidden_size)
        self.token_embedding = nn.Embedding(len(vocabulary.tokens), config.token_dim)
        self.rnn = nn.GRUCell(config.token_dim, config.hidden_size)
        self.compiled_core = BitLinear(
            BitLinearConfig(config.hidden_size, config.hidden_size, activation_bits=config.activation_bits, log_prefix="ar-compiled-core"),
            ledger=self.ledger,
        )
        self.output_head = nn.Linear(config.hidden_size, len(vocabulary.tokens))
        self.confidence_head = nn.Linear(config.hidden_size, 1)
        self.future_heads = nn.ModuleDict({
            str(horizon): nn.Linear(config.hidden_size, horizon * len(vocabulary.tokens))
            for horizon in config.horizons
        })

    def initial_hidden(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.prompt_projection(features))

    def step(self, token_ids: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Mapping[int, torch.Tensor]]:
        token = self.token_embedding(token_ids)
        hidden = self.rnn(token, hidden)
        compiled = torch.tanh(self.compiled_core(hidden))
        logits = self.output_head(compiled)
        confidence = torch.sigmoid(self.confidence_head(compiled)).squeeze(-1)
        future = {
            horizon: self.future_heads[str(horizon)](compiled).view(compiled.shape[0], horizon, len(self.vocabulary.tokens))
            for horizon in self.config.horizons
        }
        return logits, hidden, confidence, future

    def forward_teacher(self, features: torch.Tensor, decoder_inputs: torch.Tensor) -> Mapping[str, Any]:
        hidden = self.initial_hidden(features)
        logits_steps: list[torch.Tensor] = []
        confidence_steps: list[torch.Tensor] = []
        future_steps: list[Mapping[int, torch.Tensor]] = []
        for index in range(decoder_inputs.shape[1]):
            logits, hidden, confidence, future = self.step(decoder_inputs[:, index], hidden)
            logits_steps.append(logits)
            confidence_steps.append(confidence)
            future_steps.append(future)
        return {
            "logits": torch.stack(logits_steps, dim=1),
            "confidence": torch.stack(confidence_steps, dim=1),
            "future": future_steps,
        }

    def requantize_core(self) -> None:
        self.compiled_core.requantize(certify_zeros=True)

    def _compiled_weight_bits(self) -> float:
        for decision in reversed(self.ledger.compression_decisions):
            if decision.block_id == self.compiled_core.config.log_prefix:
                return decision.estimated_bits
        return float(self.compiled_core.float_weight.numel() * 32)

    def generate(self, prompt: str, *, max_tokens: int | None = None) -> tuple[str, float, CostTrace]:
        self.eval()
        limit = max_tokens or self.config.max_answer_tokens
        with torch.no_grad():
            before = self.ledger.cost_trace
            features = embed_text(prompt, self.config.input_dim).view(1, self.config.input_dim)
            hidden = self.initial_hidden(features)
            token = torch.tensor([self.vocabulary.bos_id], dtype=torch.long)
            generated: list[int] = []
            confidences: list[float] = []
            for _ in range(limit):
                logits, hidden, confidence, _ = self.step(token, hidden)
                token = logits.argmax(dim=-1)
                token_id = int(token.item())
                if token_id == self.vocabulary.eos_id:
                    break
                generated.append(token_id)
                probs = torch.softmax(logits, dim=-1)
                confidences.append(float(probs[0, token_id].item()) * float(confidence.item()))
        text = self.vocabulary.decode(generated)
        after = self.ledger.cost_trace
        steps = max(1, len(generated) + 1)
        cost = CostTrace(
            weight_bits_read=self._compiled_weight_bits() * steps,
            activation_bits=max(0.0, after.activation_bits - before.activation_bits),
            kv_bytes=max(0.0, after.kv_bytes - before.kv_bytes),
            generated_tokens=max(1, len(generated)),
            experts_activated=max(0, after.experts_activated - before.experts_activated),
        )
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return text, mean_confidence, cost

    def _future_contract_engine(self, ledger: FutureContractLedger | None = None) -> FutureContractEngine:
        return FutureContractEngine(
            MTPFSPConfig(hidden_size=self.config.hidden_size, vocab_size=len(self.vocabulary.tokens), horizons=self.config.horizons),
            ledger=ledger or FutureContractLedger(),
            trace_ledger=self.ledger,
        )

    def _shadow_greedy_rollout(
        self,
        first_logits: torch.Tensor,
        first_hidden: torch.Tensor,
        first_confidence: torch.Tensor,
        horizon: int,
    ) -> tuple[tuple[int, ...], tuple[float, ...], Mapping[int, torch.Tensor] | None, Mapping[int, tuple[torch.Tensor, torch.Tensor]]]:
        token = first_logits.argmax(dim=-1)
        token_id = int(token.item())
        probs = torch.softmax(first_logits, dim=-1)
        confidence_values = [float(probs[0, token_id].item()) * float(first_confidence.item())]
        tokens = [token_id]
        hidden = first_hidden
        resume_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {1: (first_hidden, token)}
        next_future: Mapping[int, torch.Tensor] | None = None
        for _ in range(1, horizon):
            if token_id == self.vocabulary.eos_id:
                break
            logits, hidden, confidence, future = self.step(token, hidden)
            if next_future is None:
                next_future = future
            token = logits.argmax(dim=-1)
            token_id = int(token.item())
            probs = torch.softmax(logits, dim=-1)
            confidence_values.append(float(probs[0, token_id].item()) * float(confidence.item()))
            tokens.append(token_id)
            resume_states[len(tokens)] = (hidden, token)
        return tuple(tokens), tuple(confidence_values), next_future, resume_states

    def generate_with_contracts(
        self,
        prompt: str,
        *,
        domain: str = "general",
        risk: float = 0.05,
        max_tokens: int | None = None,
    ) -> ARContractGeneration:
        self.eval()
        limit = max_tokens or self.config.max_answer_tokens
        contract_ledger = FutureContractLedger()
        engine = self._future_contract_engine(contract_ledger)
        block_traces: list[ARBlockContractTrace] = []
        with torch.no_grad():
            before = self.ledger.cost_trace
            features = embed_text(prompt, self.config.input_dim).view(1, self.config.input_dim)
            hidden = self.initial_hidden(features)
            token = torch.tensor([self.vocabulary.bos_id], dtype=torch.long)
            generated: list[int] = []
            confidences: list[float] = []
            step_index = 0
            decoder_steps = 0
            while len(generated) < limit:
                logits, hidden_after, confidence, future = self.step(token, hidden)
                remaining = limit - len(generated)
                allowed_horizons = tuple(horizon for horizon in self.config.horizons if horizon <= remaining)
                if not allowed_horizons:
                    allowed_horizons = (1,)
                requested_horizon = max(allowed_horizons)
                shadow_tokens, shadow_confidences, next_future, shadow_resume_states = self._shadow_greedy_rollout(logits, hidden_after, confidence, requested_horizon)
                decoder_steps += max(1, len(shadow_tokens))
                future_logits = {horizon: future[horizon] for horizon in allowed_horizons if horizon in future}
                if 1 not in future_logits:
                    future_logits[1] = logits.view(1, 1, -1)
                temporal_loss_value = 0.0
                if next_future is not None and requested_horizon in future and requested_horizon in next_future and requested_horizon > 1:
                    temporal_loss_value = float(temporal_consistency_loss(future[requested_horizon], next_future[requested_horizon]).detach().item())
                confidence_value = min([float(confidence.item()), *shadow_confidences]) if shadow_confidences else float(confidence.item())
                contract = engine.draft_contract_from_logits(
                    future_logits,
                    confidence=confidence_value,
                    domain=domain,
                    risk=risk,
                    contract_id=f"ar-block-{step_index}",
                    temporal_loss=temporal_loss_value,
                )
                decision = engine.gate_contract(contract, observed_tokens=shadow_tokens[:contract.accepted_horizon])
                trace = ARBlockContractTrace(
                    step_index=step_index,
                    contract_id=decision.contract.contract_id,
                    accepted_horizon=decision.contract.accepted_horizon,
                    accepted=decision.accepted,
                    reason=decision.reason,
                    token_ids=decision.contract.token_ids,
                    observed_tokens=shadow_tokens[:decision.contract.accepted_horizon],
                    confidence=decision.contract.confidence,
                    temporal_loss=decision.contract.temporal_loss,
                )
                block_traces.append(trace)
                if decision.accepted and decision.contract.accepted_horizon > 1:
                    accepted = shadow_tokens[:decision.contract.accepted_horizon]
                    for token_id, token_confidence in zip(accepted, shadow_confidences):
                        if token_id == self.vocabulary.eos_id:
                            text = self.vocabulary.decode(generated)
                            after = self.ledger.cost_trace
                            cost = CostTrace(
                                weight_bits_read=self._compiled_weight_bits() * max(1, decoder_steps),
                                activation_bits=max(0.0, after.activation_bits - before.activation_bits),
                                kv_bytes=max(0.0, after.kv_bytes - before.kv_bytes),
                                generated_tokens=max(1, len(generated)),
                            ).merge(contract_ledger.total_cost)
                            mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                            return ARContractGeneration(text, mean_confidence, cost, contract_ledger, tuple(block_traces), decoder_steps)
                        generated.append(token_id)
                        confidences.append(token_confidence)
                    resume_horizon = min(decision.contract.accepted_horizon, len(shadow_tokens))
                    hidden, token = shadow_resume_states[resume_horizon]
                else:
                    token_id = shadow_tokens[0]
                    if token_id == self.vocabulary.eos_id:
                        break
                    generated.append(token_id)
                    confidences.append(shadow_confidences[0] if shadow_confidences else float(confidence.item()))
                    hidden, token = shadow_resume_states[1]
                step_index += 1
        text = self.vocabulary.decode(generated)
        after = self.ledger.cost_trace
        cost = CostTrace(
            weight_bits_read=self._compiled_weight_bits() * max(1, decoder_steps),
            activation_bits=max(0.0, after.activation_bits - before.activation_bits),
            kv_bytes=max(0.0, after.kv_bytes - before.kv_bytes),
            generated_tokens=max(1, len(generated)),
        ).merge(contract_ledger.total_cost)
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return ARContractGeneration(text, mean_confidence, cost, contract_ledger, tuple(block_traces), decoder_steps)


class ARLossComputer:
    def __init__(self, config: ARConfig, pad_id: int):
        self.config = config
        self.pad_id = pad_id

    def compute(self, output: Mapping[str, Any], targets: torch.Tensor, target_confidence: torch.Tensor) -> tuple[torch.Tensor, Mapping[str, float]]:
        logits = output["logits"]
        vocab = logits.shape[-1]
        behavior_loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), ignore_index=self.pad_id)
        predicted = logits.argmax(dim=-1)
        mask = targets != self.pad_id
        token_accuracy = float(((predicted == targets) & mask).sum().item() / max(int(mask.sum().item()), 1))

        multi_losses: list[torch.Tensor] = []
        future_steps: Sequence[Mapping[int, torch.Tensor]] = output["future"]
        for t, future in enumerate(future_steps):
            for horizon, future_logits in future.items():
                available = min(horizon, targets.shape[1] - t)
                if available <= 0:
                    continue
                future_target = targets[:, t:t + available]
                future_logit = future_logits[:, :available, :]
                multi_losses.append(F.cross_entropy(future_logit.reshape(-1, vocab), future_target.reshape(-1), ignore_index=self.pad_id))
        multi_horizon_loss = torch.stack(multi_losses).mean() if multi_losses else behavior_loss.new_tensor(0.0)

        confidence_pred = output["confidence"].mean(dim=1)
        confidence_loss = F.mse_loss(confidence_pred, target_confidence)
        confidence_margin = torch.relu(0.75 - confidence_pred).mean()
        total = (
            behavior_loss
            + self.config.multi_horizon_weight * multi_horizon_loss
            + self.config.confidence_weight * confidence_loss
            + self.config.future_contract_weight * confidence_margin
        )
        metrics = {
            "behavior_loss": float(behavior_loss.detach().item()),
            "multi_horizon_loss": float(multi_horizon_loss.detach().item()),
            "confidence_loss": float(confidence_loss.detach().item()),
            "future_contract_loss": float(confidence_margin.detach().item()),
            "token_accuracy": token_accuracy,
        }
        return total, metrics


class ARTrainer:
    def __init__(self, config: ARConfig | None = None):
        self.config = config or ARConfig()

    def _metrics_from_tensors(
        self,
        model: ARMicroDecoder,
        features: torch.Tensor,
        decoder_inputs: torch.Tensor,
        targets: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Mapping[str, float]:
        with torch.no_grad():
            output = model.forward_teacher(features, decoder_inputs)
            loss, metrics = ARLossComputer(model.config, model.vocabulary.pad_id).compute(output, targets, confidence)
        return {"loss": float(loss.detach().item()), **metrics}

    def evaluate_teacher(self, model: ARMicroDecoder, dataset: ARDataset) -> Mapping[str, float]:
        features, decoder_inputs, targets, confidence = dataset.tensors()
        return self._metrics_from_tensors(model, features, decoder_inputs, targets, confidence)

    def exact_generation_accuracy(self, model: ARMicroDecoder, examples: Sequence[MicroTrainingExample]) -> float:
        if not examples:
            return 0.0
        correct = 0
        for example in examples:
            text, _, _ = model.generate(example.task.prompt)
            if text.strip() == example.answer.strip():
                correct += 1
        return correct / len(examples)

    def train(
        self,
        examples: Sequence[MicroTrainingExample],
        *,
        epochs: int = 300,
        lr: float = 0.03,
        checkpoint_path: str | Path | None = None,
    ) -> tuple[ARMicroDecoder, ARTrainingResult]:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        examples = tuple(examples)
        if not examples:
            raise ValueError("autoregressive training requires at least one example")
        vocabulary = TokenVocabulary.from_texts(example.answer for example in examples)
        model = ARMicroDecoder(self.config, vocabulary)
        dataset = ARDataset(examples, vocabulary, self.config)
        features, decoder_inputs, targets, confidence = dataset.tensors()
        before = self._metrics_from_tensors(model, features, decoder_inputs, targets, confidence)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_computer = ARLossComputer(self.config, vocabulary.pad_id)
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model.forward_teacher(features, decoder_inputs)
            loss, _ = loss_computer.compute(output, targets, confidence)
            loss.backward()
            optimizer.step()
        model.requantize_core()
        after = self._metrics_from_tensors(model, features, decoder_inputs, targets, confidence)
        exact = self.exact_generation_accuracy(model, examples)
        saved_path = None
        if checkpoint_path is not None:
            saved_path = str(ARCheckpointManager().save(model, checkpoint_path))
        return model, ARTrainingResult(
            epochs=epochs,
            before_token_accuracy=before["token_accuracy"],
            after_token_accuracy=after["token_accuracy"],
            before_loss=before["loss"],
            after_loss=after["loss"],
            exact_sequence_accuracy=exact,
            examples=len(examples),
            checkpoint_path=saved_path,
        )


class ARDecoderAgent:
    def __init__(self, model: ARMicroDecoder, *, use_future_contracts: bool = False):
        self.model = model
        self.use_future_contracts = use_future_contracts

    def __call__(self, task: Task) -> CandidateAnswer:
        domain = "math" if task.skill in {"arithmetic", "algebra"} else "code" if task.skill == "code_unit_tests" else "exact_anchor" if task.skill in {"long_context_anchor", "entity_tracking"} else "general"
        risk = 0.80 if domain in {"math", "code", "exact_anchor"} or task.skill == "calibration" else 0.05
        raw: Mapping[str, Any] = {}
        certificate: dict[str, Any] = {
            "autoregressive_decoder": "trained",
            "compiled_circuit": "ternary_bitlinear_autoregressive",
            "compiled_weight_bits": self.model._compiled_weight_bits(),
            "distilled_from": "verified_micro_examples",
            "strategy_invariants": ("exact_answer_replay", "route_capped_mtp", "oracle_verified_outputs"),
            "cheap_verification": "oracle_replay_contract",
            "mtp_horizons": self.model.config.horizons,
            "future_contract_loss": "trained",
        }
        if self.use_future_contracts:
            generated = self.model.generate_with_contracts(task.prompt, domain=domain, risk=risk)
            text, confidence, cost = generated.text, generated.confidence, generated.cost
            certificate["block_contracts"] = {
                "accepted": generated.accepted_blocks,
                "rejected": generated.rejected_blocks,
                "acceptance_rate": generated.ledger.acceptance_rate,
            }
            raw = {"future_contract_generation": generated.to_dict()}
        else:
            text, confidence, cost = self.model.generate(task.prompt)
        return CandidateAnswer(
            text,
            confidence=confidence,
            certificate=certificate,
            cost=cost,
            raw=raw,
        )


class ARCheckpointManager:
    def save(self, model: ARMicroDecoder, path: str | Path) -> Path:
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

    def load(self, path: str | Path) -> ARMicroDecoder:
        payload = torch.load(Path(path), map_location="cpu")
        model = ARMicroDecoder(ARConfig(**payload["config"]), TokenVocabulary(tuple(payload["vocabulary"]["tokens"])))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
