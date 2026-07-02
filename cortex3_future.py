from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex3 import AdaptiveHorizonPolicy, CandidateAnswer, CostTrace, ReferenceRuleAgent, Task
from cortex3_memory import embed_text
from cortex3_ternary import CompressionTraceLedger


DEFAULT_MTP_HORIZONS: tuple[int, ...] = (1, 2, 4, 8)


@dataclass(frozen=True)
class MTPFSPConfig:
    hidden_size: int
    vocab_size: int
    horizons: tuple[int, ...] = DEFAULT_MTP_HORIZONS
    confidence_threshold: float = 0.72
    temporal_loss_threshold: float = 0.08
    max_risk_without_verification: float = 0.20

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        if self.horizons != tuple(sorted(set(self.horizons))):
            raise ValueError("horizons must be unique and sorted")
        if self.horizons != DEFAULT_MTP_HORIZONS:
            raise ValueError("Phase 3 requires MTP horizons exactly 1, 2, 4, 8")


@dataclass(frozen=True)
class MTPHeadOutput:
    logits_by_horizon: Mapping[int, torch.Tensor]
    confidence: torch.Tensor


class MTPFSPHeads(nn.Module):
    def __init__(self, config: MTPFSPConfig):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleDict({
            str(horizon): nn.Linear(config.hidden_size, horizon * config.vocab_size)
            for horizon in config.horizons
        })
        self.confidence_head = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> MTPHeadOutput:
        if hidden.ndim == 3:
            hidden = hidden[:, -1, :]
        if hidden.ndim != 2 or hidden.shape[-1] != self.config.hidden_size:
            raise ValueError(f"hidden must have shape [batch, {self.config.hidden_size}] or [batch, time, {self.config.hidden_size}]")
        logits_by_horizon = {
            horizon: self.heads[str(horizon)](hidden).view(hidden.shape[0], horizon, self.config.vocab_size)
            for horizon in self.config.horizons
        }
        confidence = torch.sigmoid(self.confidence_head(hidden)).squeeze(-1)
        return MTPHeadOutput(logits_by_horizon, confidence)


@dataclass(frozen=True)
class FutureTrainingExample:
    prompt: str
    token_ids: tuple[int, ...]
    confidence: float
    domain: str = "general"


@dataclass(frozen=True)
class MTPFSPCalibrationResult:
    epochs: int
    examples: int
    before_loss: float
    after_loss: float
    before_token_accuracy: float
    after_token_accuracy: float
    before_confidence_loss: float
    after_confidence_loss: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def future_examples_from_tasks(
    tasks: Iterable[Task],
    encoder: Callable[[str], Sequence[int]],
    solver: Callable[[Task], CandidateAnswer | str] | None = None,
) -> tuple[FutureTrainingExample, ...]:
    resolved = solver or ReferenceRuleAgent()
    out: list[FutureTrainingExample] = []
    for task in tasks:
        answer = CandidateAnswer.coerce(resolved(task))
        domain = "math" if task.skill in {"arithmetic", "algebra"} else "code" if task.skill == "code_unit_tests" else "exact_anchor" if task.skill in {"long_context_anchor", "entity_tracking"} else "general"
        out.append(FutureTrainingExample(task.prompt, tuple(int(token) for token in encoder(answer.text)), answer.confidence, domain))
    return tuple(out)


class MTPFSPCalibrator:
    def __init__(self, config: MTPFSPConfig):
        self.config = config

    def _batch(self, examples: Sequence[FutureTrainingExample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not examples:
            raise ValueError("MTP/FSP calibration requires at least one example")
        max_horizon = max(self.config.horizons)
        hidden = torch.stack([embed_text(example.prompt, self.config.hidden_size) for example in examples]).to(torch.float32)
        targets = torch.zeros((len(examples), max_horizon), dtype=torch.long)
        for row, example in enumerate(examples):
            if not example.token_ids:
                raise ValueError("future training example cannot have empty token_ids")
            clipped = tuple(token for token in example.token_ids[:max_horizon])
            if min(clipped) < 0 or max(clipped) >= self.config.vocab_size:
                raise ValueError("token id is outside MTP/FSP vocabulary")
            targets[row, :len(clipped)] = torch.tensor(clipped, dtype=torch.long)
            if len(clipped) < max_horizon:
                targets[row, len(clipped):] = clipped[-1]
        confidence = torch.tensor([max(0.0, min(1.0, example.confidence)) for example in examples], dtype=torch.float32)
        return hidden, targets, confidence

    def _loss(self, heads: MTPFSPHeads, hidden: torch.Tensor, targets: torch.Tensor, confidence_targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        output = heads(hidden)
        losses: list[torch.Tensor] = []
        correct = 0
        total = 0
        for horizon, logits in output.logits_by_horizon.items():
            horizon_targets = targets[:, :horizon]
            losses.append(F.cross_entropy(logits.reshape(-1, self.config.vocab_size), horizon_targets.reshape(-1)))
            predicted = logits.argmax(dim=-1)
            correct += int((predicted == horizon_targets).sum().item())
            total += int(horizon_targets.numel())
        token_loss = torch.stack(losses).mean()
        confidence_loss = F.mse_loss(output.confidence, confidence_targets)
        loss = token_loss + 0.25 * confidence_loss
        return loss, {
            "token_accuracy": correct / max(total, 1),
            "confidence_loss": float(confidence_loss.detach().item()),
            "loss": float(loss.detach().item()),
        }

    def evaluate(self, heads: MTPFSPHeads, examples: Sequence[FutureTrainingExample]) -> Mapping[str, float]:
        hidden, targets, confidence = self._batch(examples)
        with torch.no_grad():
            _, metrics = self._loss(heads, hidden, targets, confidence)
        return metrics

    def train(self, examples: Sequence[FutureTrainingExample], *, epochs: int = 160, lr: float = 0.05, heads: MTPFSPHeads | None = None) -> tuple[MTPFSPHeads, MTPFSPCalibrationResult]:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        examples = tuple(examples)
        model = heads or MTPFSPHeads(self.config)
        hidden, targets, confidence = self._batch(examples)
        before_loss, before_metrics = self._loss(model, hidden, targets, confidence)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss, _ = self._loss(model, hidden, targets, confidence)
            loss.backward()
            optimizer.step()
        after_loss, after_metrics = self._loss(model, hidden, targets, confidence)
        return model, MTPFSPCalibrationResult(
            epochs=epochs,
            examples=len(examples),
            before_loss=float(before_loss.detach().item()),
            after_loss=float(after_loss.detach().item()),
            before_token_accuracy=before_metrics["token_accuracy"],
            after_token_accuracy=after_metrics["token_accuracy"],
            before_confidence_loss=before_metrics["confidence_loss"],
            after_confidence_loss=after_metrics["confidence_loss"],
        )


def temporal_consistency_loss(previous_logits: torch.Tensor, next_logits: torch.Tensor) -> torch.Tensor:
    if previous_logits.ndim != 3 or next_logits.ndim != 3:
        raise ValueError("temporal consistency expects [batch, horizon, vocab] tensors")
    if previous_logits.shape[0] != next_logits.shape[0] or previous_logits.shape[2] != next_logits.shape[2]:
        raise ValueError("previous and next logits must share batch and vocab dimensions")
    aligned = min(previous_logits.shape[1] - 1, next_logits.shape[1])
    if aligned <= 0:
        return previous_logits.new_tensor(0.0)
    prev_probs = F.softmax(previous_logits[:, 1:1 + aligned, :], dim=-1)
    next_probs = F.softmax(next_logits[:, :aligned, :], dim=-1)
    return F.mse_loss(prev_probs, next_probs)


def temporal_consistency_loss_from_outputs(previous: MTPHeadOutput, next_output: MTPHeadOutput) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for horizon in sorted(previous.logits_by_horizon):
        if horizon <= 1:
            continue
        prev_logits = previous.logits_by_horizon[horizon]
        next_logits = next_output.logits_by_horizon[horizon]
        losses.append(temporal_consistency_loss(prev_logits, next_logits))
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class FutureContract:
    contract_id: str
    domain: str
    risk: float
    requested_horizon: int
    accepted_horizon: int
    token_ids: tuple[int, ...]
    confidence: float
    temporal_loss: float
    revision: int = 0
    accepted: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ContractDecision:
    contract: FutureContract
    accepted: bool
    reason: str
    cost: CostTrace


@dataclass(frozen=True)
class OutputGoalContract:
    contract_id: str
    task_id: str
    skill: str
    expected_type: str
    expected_text: str
    required_anchor_values: tuple[str, ...]
    forbidden_substrings: tuple[str, ...]
    obligations: tuple[str, ...]
    risk: float


@dataclass(frozen=True)
class OutputGoalDecision:
    contract: OutputGoalContract
    answer_text: str
    accepted: bool
    reason: str
    violations: tuple[str, ...]
    forbidden_matches: tuple[str, ...]
    cost: CostTrace

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": asdict(self.contract),
            "answer_text": self.answer_text,
            "accepted": self.accepted,
            "reason": self.reason,
            "violations": list(self.violations),
            "forbidden_matches": list(self.forbidden_matches),
            "cost": asdict(self.cost),
        }


@dataclass
class FutureContractLedger:
    decisions: list[ContractDecision] = field(default_factory=list)
    output_goal_decisions: list[OutputGoalDecision] = field(default_factory=list)

    def record(self, decision: ContractDecision) -> None:
        self.decisions.append(decision)

    def record_output_goal(self, decision: OutputGoalDecision) -> None:
        self.output_goal_decisions.append(decision)

    @property
    def accepted(self) -> int:
        return sum(1 for decision in self.decisions if decision.accepted)

    @property
    def rejected(self) -> int:
        return len(self.decisions) - self.accepted

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / len(self.decisions) if self.decisions else 0.0

    @property
    def total_cost(self) -> CostTrace:
        cost = CostTrace()
        for decision in self.decisions:
            cost = cost.merge(decision.cost)
        for decision in self.output_goal_decisions:
            cost = cost.merge(decision.cost)
        return cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "acceptance_rate": self.acceptance_rate,
            "total_cost": asdict(self.total_cost),
            "output_goal_accepted": sum(1 for decision in self.output_goal_decisions if decision.accepted),
            "output_goal_rejected": sum(1 for decision in self.output_goal_decisions if not decision.accepted),
            "output_goal_decisions": [decision.to_dict() for decision in self.output_goal_decisions],
            "decisions": [
                {
                    "contract": asdict(decision.contract),
                    "accepted": decision.accepted,
                    "reason": decision.reason,
                    "cost": asdict(decision.cost),
                }
                for decision in self.decisions
            ],
        }


_EXACT_OUTPUT_SKILLS = {
    "arithmetic",
    "algebra",
    "calibration",
    "entity_tracking",
    "instruction_following",
    "long_context_anchor",
}

_INTERNAL_LEAKAGE_MARKERS: tuple[str, ...] = (
    "<analysis>",
    "</analysis>",
    "<scratchpad>",
    "</scratchpad>",
    "chain-of-thought:",
    "chain_of_thought:",
    "hidden reasoning:",
    "internal_trace=",
    "debug_trace=",
    "output_goal_contract=",
    "frontier_compiled_contract=",
    "latent_workspace_trace=",
)

_FORBIDDEN_METADATA_KEYS: tuple[str, ...] = (
    "forbidden_output_substrings",
    "forbidden_substrings",
    "disallowed_output_substrings",
)


def _expected_text(task: Task) -> str:
    return "" if task.expected is None else str(task.expected)


def _metadata_string_values(metadata: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for key in keys:
        raw = metadata.get(key)
        if raw is None:
            continue
        if isinstance(raw, str):
            candidates = (raw,)
        elif isinstance(raw, (tuple, list, set, frozenset)):
            candidates = tuple(raw)
        else:
            candidates = (raw,)
        for item in candidates:
            text = str(item).strip()
            if text:
                values.append(text)
    return tuple(values)


def _output_goal_forbidden_substrings(task: Task) -> tuple[str, ...]:
    expected_lower = _expected_text(task).lower()
    prompt_lower = task.prompt.lower()
    forbidden: list[str] = []
    for marker in _INTERNAL_LEAKAGE_MARKERS:
        marker_lower = marker.lower()
        if marker_lower not in expected_lower and marker_lower not in prompt_lower:
            forbidden.append(marker)
    forbidden.extend(_metadata_string_values(task.metadata, _FORBIDDEN_METADATA_KEYS))
    out: list[str] = []
    seen: set[str] = set()
    for item in forbidden:
        key = item.lower()
        if key not in seen:
            out.append(item)
            seen.add(key)
    return tuple(out)


def _output_goal_obligations(task: Task) -> tuple[str, ...]:
    prompt = task.prompt.lower()
    obligations: list[str] = []
    if task.expected is not None:
        obligations.append("expected_value")
    if task.skill in _EXACT_OUTPUT_SKILLS or "exact" in prompt or "exactly" in prompt:
        obligations.append("exact_output")
    if "return only" in prompt or "output only" in prompt or "reponds seulement" in prompt:
        obligations.append("no_extra_text")
    if task.anchors:
        obligations.append("preserve_required_anchors")
    if task.skill == "code_unit_tests":
        obligations.append("executable_code_contract")
    if task.skill == "calibration":
        obligations.append("calibrated_unknown_contract")
    forbidden = _output_goal_forbidden_substrings(task)
    if forbidden:
        obligations.append("no_internal_leakage")
    if _metadata_string_values(task.metadata, _FORBIDDEN_METADATA_KEYS):
        obligations.append("no_forbidden_output")
    out: list[str] = []
    seen: set[str] = set()
    for item in obligations:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return tuple(out)


class FutureContractEngine:
    def __init__(
        self,
        config: MTPFSPConfig,
        heads: MTPFSPHeads | None = None,
        horizon_policy: AdaptiveHorizonPolicy | None = None,
        ledger: FutureContractLedger | None = None,
        trace_ledger: CompressionTraceLedger | None = None,
    ):
        self.config = config
        self.heads = heads or MTPFSPHeads(config)
        self.horizon_policy = horizon_policy or AdaptiveHorizonPolicy(max_horizon=max(config.horizons))
        self.ledger = ledger or FutureContractLedger()
        self.trace_ledger = trace_ledger

    def _nearest_allowed_horizon(self, requested: int) -> int:
        candidates = [horizon for horizon in self.config.horizons if horizon <= requested]
        return max(candidates) if candidates else min(self.config.horizons)

    def _choose_horizon(self, confidence: float, risk: float, domain: str) -> tuple[int, str, bool]:
        policy = self.horizon_policy.choose(confidence, risk, domain)
        return self._nearest_allowed_horizon(policy.horizon), policy.reason, policy.requires_verification

    def draft_contract(
        self,
        hidden: torch.Tensor,
        *,
        domain: str = "general",
        risk: float = 0.0,
        contract_id: str = "contract",
        temporal_loss: float = 0.0,
    ) -> FutureContract:
        output = self.heads(hidden)
        confidence = float(output.confidence.detach().mean().item())
        requested_horizon = max(self.config.horizons)
        accepted_horizon, policy_reason, requires_verification = self._choose_horizon(confidence, risk, domain)
        logits = output.logits_by_horizon[accepted_horizon][0]
        token_ids = tuple(int(token) for token in logits.argmax(dim=-1).detach().cpu().tolist())
        accepted = (
            confidence >= self.config.confidence_threshold
            and temporal_loss <= self.config.temporal_loss_threshold
            and not requires_verification
        )
        reason = "accepted" if accepted else f"requires verification: {policy_reason}"
        return FutureContract(
            contract_id=contract_id,
            domain=domain,
            risk=max(0.0, min(1.0, risk)),
            requested_horizon=requested_horizon,
            accepted_horizon=accepted_horizon,
            token_ids=token_ids,
            confidence=confidence,
            temporal_loss=temporal_loss,
            revision=0,
            accepted=accepted,
            reason=reason,
        )

    def draft_contract_from_logits(
        self,
        logits_by_horizon: Mapping[int, torch.Tensor],
        *,
        confidence: float,
        domain: str = "general",
        risk: float = 0.0,
        contract_id: str = "contract",
        temporal_loss: float = 0.0,
    ) -> FutureContract:
        if not logits_by_horizon:
            raise ValueError("at least one MTP/FSP horizon must be supplied")
        available_horizons = tuple(sorted(int(horizon) for horizon in logits_by_horizon))
        bounded_confidence = max(0.0, min(1.0, float(confidence)))
        accepted_horizon, policy_reason, requires_verification = self._choose_horizon(bounded_confidence, risk, domain)
        if accepted_horizon not in logits_by_horizon:
            lower_or_equal = tuple(horizon for horizon in available_horizons if horizon <= accepted_horizon)
            accepted_horizon = max(lower_or_equal) if lower_or_equal else min(available_horizons)
        logits = logits_by_horizon[accepted_horizon]
        if logits.ndim == 3:
            logits = logits[0]
        if logits.ndim != 2 or logits.shape[0] < accepted_horizon:
            raise ValueError("MTP/FSP logits must have shape [horizon, vocab] or [batch, horizon, vocab]")
        token_ids = tuple(int(token) for token in logits[:accepted_horizon].argmax(dim=-1).detach().cpu().tolist())
        accepted = (
            bounded_confidence >= self.config.confidence_threshold
            and temporal_loss <= self.config.temporal_loss_threshold
            and not requires_verification
        )
        if accepted:
            reason = "accepted"
        elif temporal_loss > self.config.temporal_loss_threshold:
            reason = "temporal consistency exceeded threshold"
        elif bounded_confidence < self.config.confidence_threshold:
            reason = "confidence below threshold"
        else:
            reason = f"requires verification: {policy_reason}"
        return FutureContract(
            contract_id=contract_id,
            domain=domain,
            risk=max(0.0, min(1.0, risk)),
            requested_horizon=max(available_horizons),
            accepted_horizon=accepted_horizon,
            token_ids=token_ids,
            confidence=bounded_confidence,
            temporal_loss=float(temporal_loss),
            revision=0,
            accepted=accepted,
            reason=reason,
        )

    def revise_contract(self, contract: FutureContract, *, temporal_loss: float | None = None, observed_tokens: Sequence[int] | None = None, reason: str = "") -> FutureContract:
        new_temporal_loss = contract.temporal_loss if temporal_loss is None else temporal_loss
        observed_tuple = tuple(observed_tokens or ())
        incomplete_observation = observed_tokens is not None and len(observed_tuple) < contract.accepted_horizon
        mismatch = observed_tokens is not None and observed_tuple[:contract.accepted_horizon] != contract.token_ids[:len(observed_tuple)]
        must_shrink = (
            new_temporal_loss > self.config.temporal_loss_threshold
            or incomplete_observation
            or mismatch
            or contract.confidence < self.config.confidence_threshold
        )
        new_horizon = contract.accepted_horizon
        if must_shrink:
            candidates = [horizon for horizon in self.config.horizons if horizon < contract.accepted_horizon]
            new_horizon = max(candidates) if candidates else 1
        accepted = (
            not must_shrink
            and contract.confidence >= self.config.confidence_threshold
            and new_temporal_loss <= self.config.temporal_loss_threshold
        )
        if incomplete_observation:
            revision_reason = "observed tokens incomplete for future contract"
        elif mismatch:
            revision_reason = "observed tokens broke future contract"
        elif new_temporal_loss > self.config.temporal_loss_threshold:
            revision_reason = "temporal consistency exceeded threshold"
        elif contract.confidence < self.config.confidence_threshold:
            revision_reason = "confidence below threshold"
        else:
            revision_reason = reason or "contract preserved"
        return FutureContract(
            contract_id=contract.contract_id,
            domain=contract.domain,
            risk=contract.risk,
            requested_horizon=contract.requested_horizon,
            accepted_horizon=new_horizon,
            token_ids=contract.token_ids[:new_horizon],
            confidence=contract.confidence,
            temporal_loss=new_temporal_loss,
            revision=contract.revision + 1,
            accepted=accepted,
            reason=revision_reason,
        )

    def gate_contract(self, contract: FutureContract, *, observed_tokens: Sequence[int] | None = None) -> ContractDecision:
        checked = self.revise_contract(contract, observed_tokens=observed_tokens) if observed_tokens is not None else contract
        accepted = checked.accepted
        reason = checked.reason if not accepted else "block accepted under future contract"
        cost = CostTrace(
            generated_tokens=checked.accepted_horizon,
            latent_steps=1 if checked.accepted_horizon > 1 else 0,
            verifier_steps=0 if accepted else 1,
        )
        decision = ContractDecision(checked, accepted, reason, cost)
        self.ledger.record(decision)
        if self.trace_ledger is not None:
            self.trace_ledger.record_mtp_fsp(
                checked.contract_id,
                checked.accepted_horizon,
                accepted,
                checked.confidence,
                checked.revision,
                reason,
            )
        return decision

    def draft_output_goal_contract(
        self,
        task: Task,
        *,
        risk: float = 0.0,
        contract_id: str | None = None,
    ) -> OutputGoalContract:
        return OutputGoalContract(
            contract_id=contract_id or f"{task.task_id}-output-goal",
            task_id=task.task_id,
            skill=task.skill,
            expected_type=type(task.expected).__name__,
            expected_text=_expected_text(task),
            required_anchor_values=tuple(anchor.value for anchor in task.anchors),
            forbidden_substrings=_output_goal_forbidden_substrings(task),
            obligations=_output_goal_obligations(task),
            risk=max(0.0, min(1.0, float(risk))),
        )

    def gate_output_goal_contract(
        self,
        contract: OutputGoalContract,
        answer: CandidateAnswer,
        *,
        output_verified: bool | None = None,
    ) -> OutputGoalDecision:
        answer_text = answer.text.strip()
        expected_text = contract.expected_text.strip()
        violations: list[str] = []
        if "expected_value" in contract.obligations and not expected_text:
            violations.append("missing_expected_value")
        if "exact_output" in contract.obligations and expected_text and answer_text != expected_text:
            violations.append("exact_output_mismatch")
        if "no_extra_text" in contract.obligations and expected_text and answer_text != expected_text:
            violations.append("extra_text_or_missing_required_value")
        missing_anchors = tuple(value for value in contract.required_anchor_values if value and value not in answer.text)
        if missing_anchors:
            violations.append("required_anchor_missing")
        answer_lower = answer.text.lower()
        forbidden_matches = tuple(
            item for item in contract.forbidden_substrings if item and item.lower() in answer_lower
        )
        if forbidden_matches:
            violations.append("forbidden_output_substring")
        if output_verified is False:
            violations.append("oracle_verification_failed")
        accepted = not violations
        cost = CostTrace(
            generated_tokens=max(1, len(answer_text.split())),
            latent_steps=1 if contract.required_anchor_values else 0,
            verifier_steps=1,
        )
        reason = "output-goal contract accepted" if accepted else "; ".join(violations)
        decision = OutputGoalDecision(
            contract=contract,
            answer_text=answer.text,
            accepted=accepted,
            reason=reason,
            violations=tuple(violations),
            forbidden_matches=forbidden_matches,
            cost=cost,
        )
        self.ledger.record_output_goal(decision)
        return decision

    def gate_output_goal(
        self,
        task: Task,
        answer: CandidateAnswer,
        *,
        risk: float = 0.0,
        contract_id: str | None = None,
        output_verified: bool | None = None,
    ) -> OutputGoalDecision:
        contract = self.draft_output_goal_contract(task, risk=risk, contract_id=contract_id)
        return self.gate_output_goal_contract(contract, answer, output_verified=output_verified)


def verified_answers_per_effective_cost(verified_answers: float, total_cost: CostTrace) -> float:
    return verified_answers / max(total_cost.effective_cost(), 1e-9)
