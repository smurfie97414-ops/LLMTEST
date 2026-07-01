from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn

from cortex3 import CandidateAnswer, CostTrace, DynamicSkillVerifier, Task, VerificationCaseResult
from cortex3_certificates import CertificateHead, CertificateType, CertificateVerifier, LatentProofState, ShortCertificate, build_certificate
from cortex3_frontier import CompiledFrontierAgent, FrontierCircuitRegistry
from cortex3_future import FutureContractEngine, MTPFSPConfig, MTPFSPHeads
from cortex3_memory import CognitiveMemory, MemoryReconstruction, embed_text, tokenize
from cortex3_ternary import BitLinear, BitLinearConfig, CompressionTraceLedger


class InferencePath(str, Enum):
    FAST = "fast"
    NORMAL = "normal"
    CAREFUL = "careful"


@dataclass(frozen=True)
class DifficultySignal:
    skill: str
    prompt_tokens: int
    anchors: int
    confidence: float
    risk: float
    exactness_required: bool
    code_or_math: bool


@dataclass(frozen=True)
class InferenceRoute:
    path: InferencePath
    compression_strength: float
    layers_to_run: int
    mtp_horizon: int
    verifier_level: int
    use_latent_kv: bool
    experts_activated: int
    latent_loops: int
    reason: str


@dataclass(frozen=True)
class InferenceConfig:
    hidden_size: int = 32
    vocab_size: int = 64
    max_layers: int = 4
    fast_layers: int = 1
    normal_layers: int = 3
    careful_layers: int = 4
    fast_early_exit_confidence: float = 0.82
    normal_early_exit_confidence: float = 0.92
    careful_early_exit_confidence: float = 0.99
    budget_per_effective_cost: float = 1.0

    def __post_init__(self) -> None:
        if self.hidden_size < 8:
            raise ValueError("hidden_size must be at least 8")
        if self.max_layers < 1:
            raise ValueError("max_layers must be positive")
        if not (1 <= self.fast_layers <= self.normal_layers <= self.careful_layers <= self.max_layers):
            raise ValueError("layer counts must satisfy fast <= normal <= careful <= max")


@dataclass(frozen=True)
class BudgetPrediction:
    route: InferenceRoute
    predicted_cost: CostTrace
    predicted_effective_cost: float


@dataclass(frozen=True)
class EarlyExitDecision:
    exit: bool
    layer_index: int
    confidence: float
    reason: str


@dataclass(frozen=True)
class TernaryKernelDispatch:
    layer_index: int
    mode: str
    packed_weight_bytes: float
    active_weights: int
    total_weights: int
    reason: str


@dataclass(frozen=True)
class InferenceResult:
    task: Task
    route: InferenceRoute
    answer: CandidateAnswer
    verification: VerificationCaseResult | None
    cost: CostTrace
    predicted_cost: BudgetPrediction
    early_exit: EarlyExitDecision
    layers_ran: int
    memory_reconstruction: MemoryReconstruction | None
    future_contract: Mapping[str, Any] | None
    kernel_dispatches: tuple[TernaryKernelDispatch, ...]
    certificate_verified: bool
    verified_capability_per_cost: float
    trace_summary: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.verification and self.verification.passed and self.certificate_verified)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "skill": self.task.skill,
            "route": asdict(self.route),
            "answer": {
                "text": self.answer.text,
                "confidence": self.answer.confidence,
                "certificate": dict(self.answer.certificate),
                "cost": asdict(self.answer.cost),
                "raw": dict(self.answer.raw),
            },
            "verification": {
                "passed": self.verification.passed,
                "score": self.verification.score,
                "reason": self.verification.reason,
            } if self.verification else None,
            "cost": asdict(self.cost),
            "predicted_cost": asdict(self.predicted_cost.predicted_cost),
            "early_exit": asdict(self.early_exit),
            "layers_ran": self.layers_ran,
            "memory_reconstruction": {
                "selected_segment_ids": self.memory_reconstruction.selected_segment_ids,
                "fidelity": asdict(self.memory_reconstruction.fidelity),
            } if self.memory_reconstruction else None,
            "future_contract": dict(self.future_contract or {}),
            "kernel_dispatches": [asdict(dispatch) for dispatch in self.kernel_dispatches],
            "certificate_verified": self.certificate_verified,
            "verified_capability_per_cost": self.verified_capability_per_cost,
            "trace_summary": dict(self.trace_summary),
        }


Agent = Callable[[Task], CandidateAnswer | str]


_ANCHOR_KINDS_FOR_ASK: Mapping[str, tuple[str, ...]] = {
    "code": ("identifier",),
    "identifier": ("identifier",),
    "id": ("identifier",),
    "city": ("city",),
    "ville": ("city",),
    "amount": ("amount", "number"),
    "montant": ("amount", "number"),
    "item": ("item",),
    "objet": ("item",),
}


class DifficultyRouter:
    high_risk_skills = {"arithmetic", "algebra", "code_unit_tests", "long_context_anchor", "entity_tracking", "calibration"}

    def __init__(self, config: InferenceConfig):
        self.config = config

    def signal(self, task: Task, base_confidence: float = 0.90) -> DifficultySignal:
        prompt_tokens = len(tokenize(task.prompt))
        anchors = len(task.anchors)
        exactness = bool(anchors or task.skill in self.high_risk_skills)
        code_or_math = task.skill in {"arithmetic", "algebra", "code_unit_tests"}
        risk = 0.10
        if task.skill in self.high_risk_skills:
            risk += 0.35
        if prompt_tokens > 40:
            risk += 0.20
        if anchors:
            risk += 0.20
        if task.skill == "calibration":
            risk += 0.20
        risk = max(0.0, min(1.0, risk))
        confidence = max(0.0, min(1.0, base_confidence - 0.30 * risk))
        return DifficultySignal(task.skill, prompt_tokens, anchors, confidence, risk, exactness, code_or_math)

    def route(self, signal: DifficultySignal, forced_path: InferencePath | None = None) -> InferenceRoute:
        if forced_path is None:
            if signal.risk <= 0.20 and signal.confidence >= 0.80:
                path = InferencePath.FAST
            elif signal.risk <= 0.55 and signal.confidence >= 0.65:
                path = InferencePath.NORMAL
            else:
                path = InferencePath.CAREFUL
        else:
            path = forced_path
        if path == InferencePath.FAST:
            return InferenceRoute(path, 0.85, self.config.fast_layers, 8, 0, True, 0, 0, "easy task: strong compression, shallow depth, high MTP")
        if path == InferencePath.NORMAL:
            return InferenceRoute(path, 0.50, self.config.normal_layers, 4 if not signal.code_or_math else 2, 1, True, 0, 1, "normal task: moderate compression, full core subset, light certificate")
        return InferenceRoute(path, 0.10, self.config.careful_layers, 1 if signal.risk > 0.70 or signal.code_or_math else 2, 3, False, 1, 3, "hard task: cautious compression, experts, latent loops, strong verification")


class BudgetPredictor:
    def predict(self, route: InferenceRoute, prompt_tokens: int) -> BudgetPrediction:
        cost = CostTrace(
            weight_bits_read=route.layers_to_run * (16.0 + 32.0 * (1.0 - route.compression_strength)),
            activation_bits=route.layers_to_run * 4.0 * max(1, prompt_tokens // 4),
            kv_bytes=8.0 if route.use_latent_kv else 32.0,
            generated_tokens=max(1, route.mtp_horizon),
            latent_steps=route.latent_loops,
            experts_activated=route.experts_activated,
            verifier_steps=route.verifier_level,
        )
        return BudgetPrediction(route, cost, cost.effective_cost())


class EarlyExitPolicy:
    def threshold(self, path: InferencePath, config: InferenceConfig) -> float:
        if path == InferencePath.FAST:
            return config.fast_early_exit_confidence
        if path == InferencePath.NORMAL:
            return config.normal_early_exit_confidence
        return config.careful_early_exit_confidence

    def decide(self, route: InferenceRoute, layer_index: int, confidence: float, config: InferenceConfig) -> EarlyExitDecision:
        threshold = self.threshold(route.path, config)
        can_exit = layer_index + 1 >= route.layers_to_run or confidence >= threshold
        reason = "confidence threshold met" if confidence >= threshold else "required depth reached" if can_exit else "continue"
        return EarlyExitDecision(can_exit, layer_index + 1, confidence, reason)


class TernaryKernelDispatcher:
    def dispatch(self, layer: BitLinear, layer_index: int, route: InferenceRoute) -> TernaryKernelDispatch:
        active = int(layer.mask.detach().sum().item())
        total = int(layer.mask.numel())
        scale_bits = int(layer.scales.numel()) * 16
        packed_bits = total + active + scale_bits
        if layer.bias is not None:
            packed_bits += int(layer.bias.numel()) * 16
        mode = "cuda_ternary_packed" if torch.cuda.is_available() and layer.float_weight.is_cuda else "cpu_ternary_reference"
        reason = f"{route.path.value} path uses sign+mask packed ternary dispatch"
        return TernaryKernelDispatch(layer_index, mode, packed_bits / 8.0, active, total, reason)


class MixtureOfDepthsCore(nn.Module):
    def __init__(self, config: InferenceConfig, trace: CompressionTraceLedger):
        super().__init__()
        torch.manual_seed(1234)
        self.config = config
        self.trace = trace
        self.layers = nn.ModuleList([
            BitLinear(BitLinearConfig(config.hidden_size, config.hidden_size, activation_bits=4, log_prefix=f"mod-layer-{idx}"), ledger=trace)
            for idx in range(config.max_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.confidence_head = nn.Linear(config.hidden_size, 1)
        self.kernel_dispatcher = TernaryKernelDispatcher()
        with torch.no_grad():
            self.confidence_head.weight.fill_(0.01)
            self.confidence_head.bias.fill_(1.25)

    def forward_route(self, x: torch.Tensor, route: InferenceRoute, early_exit: EarlyExitPolicy) -> tuple[torch.Tensor, int, EarlyExitDecision, tuple[TernaryKernelDispatch, ...]]:
        hidden = x
        decision = EarlyExitDecision(False, 0, 0.0, "not started")
        dispatches: list[TernaryKernelDispatch] = []
        for idx in range(route.layers_to_run):
            self.layers[idx].requantize()
            dispatches.append(self.kernel_dispatcher.dispatch(self.layers[idx], idx, route))
            hidden = torch.tanh(self.layers[idx](hidden))
            hidden = self.norm(hidden)
            confidence = float(torch.sigmoid(self.confidence_head(hidden)).detach().mean())
            decision = early_exit.decide(route, idx, confidence, self.config)
            if decision.exit:
                return hidden, idx + 1, decision, tuple(dispatches)
        return hidden, route.layers_to_run, decision, tuple(dispatches)


class SelfSpeculativeDecoder:
    def __init__(self, config: InferenceConfig):
        heads = MTPFSPHeads(MTPFSPConfig(hidden_size=config.hidden_size, vocab_size=config.vocab_size))
        with torch.no_grad():
            heads.confidence_head.weight.zero_()
            heads.confidence_head.bias.fill_(5.0)
        self.engine = FutureContractEngine(heads.config, heads=heads)

    def speculate(self, hidden: torch.Tensor, route: InferenceRoute, task: Task) -> Mapping[str, Any]:
        domain = "math" if task.skill in {"arithmetic", "algebra"} else "code" if task.skill == "code_unit_tests" else "exact_anchor" if task.skill in {"long_context_anchor", "entity_tracking"} else "general"
        risk = 0.80 if route.path == InferencePath.CAREFUL else 0.40 if route.path == InferencePath.NORMAL else 0.05
        contract = self.engine.draft_contract(hidden, domain=domain, risk=risk, contract_id=f"{task.task_id}-{route.path.value}")
        if contract.accepted_horizon > route.mtp_horizon:
            contract = replace(
                contract,
                accepted_horizon=route.mtp_horizon,
                token_ids=contract.token_ids[:route.mtp_horizon],
                accepted=contract.accepted and route.verifier_level == 0,
                reason=f"{contract.reason}; capped by {route.path.value} route",
            )
        decision = self.engine.gate_contract(contract, observed_tokens=contract.token_ids if route.verifier_level > 0 else None)
        return {
            "contract_id": decision.contract.contract_id,
            "horizon": decision.contract.accepted_horizon,
            "initial_accepted": contract.accepted,
            "gate_accepted": decision.accepted,
            "reason": decision.reason,
            "acceptance_rate": self.engine.ledger.acceptance_rate,
            "cost": asdict(decision.cost),
        }


class UltraFastInferenceEngine:
    def __init__(
        self,
        verifier: DynamicSkillVerifier,
        agent: Agent,
        config: InferenceConfig | None = None,
        memory: CognitiveMemory | None = None,
        compiled_frontier_registry: FrontierCircuitRegistry | None = None,
    ):
        self.verifier = verifier
        self.agent = agent
        self.config = config or InferenceConfig()
        self.memory = memory or CognitiveMemory()
        self.compiled_frontier_registry = compiled_frontier_registry
        self.compiled_frontier_agent = (
            CompiledFrontierAgent(compiled_frontier_registry, verifier=verifier)
            if compiled_frontier_registry is not None
            else None
        )
        self.router = DifficultyRouter(self.config)
        self.budget = BudgetPredictor()
        self.early_exit = EarlyExitPolicy()
        self.trace = CompressionTraceLedger()
        self.core = MixtureOfDepthsCore(self.config, self.trace)
        self.speculative = SelfSpeculativeDecoder(self.config)
        self.certificate_head = CertificateHead(self.config.hidden_size, max(8, self.config.hidden_size // 2), self.config.vocab_size)
        self.certificate_verifier = CertificateVerifier()

    def set_compiled_frontier_registry(self, registry: FrontierCircuitRegistry | None) -> None:
        self.compiled_frontier_registry = registry
        self.compiled_frontier_agent = (
            CompiledFrontierAgent(registry, verifier=self.verifier)
            if registry is not None
            else None
        )

    def _compiled_frontier_answer(self, task: Task) -> CandidateAnswer | None:
        if self.compiled_frontier_registry is None or self.compiled_frontier_agent is None:
            return None
        if self.compiled_frontier_registry.select(task) is None:
            return None
        return self.compiled_frontier_agent(task)

    def _reset_trace(self) -> None:
        self.trace = CompressionTraceLedger()
        self.core.trace = self.trace
        for layer in self.core.layers:
            layer.ledger = self.trace
        self.speculative.engine.trace_ledger = self.trace

    def _input_tensor(self, task: Task, reconstruction: MemoryReconstruction | None) -> torch.Tensor:
        text = task.prompt
        if reconstruction is not None:
            text += "\n" + reconstruction.rendered_context
        emb = embed_text(text, self.config.hidden_size)
        return emb.view(1, self.config.hidden_size)

    def _record_memory_trace(self, reconstruction: MemoryReconstruction) -> None:
        for idx, text in enumerate(reconstruction.exact_context):
            self.trace.record_kv(f"exact-{idx}", "exact", float(len(text.encode("utf-8"))), exact_anchors=len(reconstruction.anchors), note="recent exact KV")
        for idx, text in enumerate(reconstruction.latent_context):
            self.trace.record_kv(f"latent-{idx}", "latent", float(len(text.encode("utf-8"))), exact_anchors=len(reconstruction.anchors), note="query-conditioned latent KV")

    def _memory_anchor_candidate(self, task: Task, reconstruction: MemoryReconstruction | None) -> CandidateAnswer | None:
        if reconstruction is None or not reconstruction.fidelity.passed:
            return None
        if task.skill == "long_context_anchor":
            ask_kind = str(task.metadata.get("ask_kind", "")).lower()
            desired_kinds = _ANCHOR_KINDS_FOR_ASK.get(ask_kind, ())
            if not desired_kinds and len(task.anchors) == 1:
                desired_kinds = (task.anchors[0].kind,)
        elif task.skill == "entity_tracking":
            desired_kinds = ("location", "place", "city")
        else:
            return None

        selected = next((anchor for anchor in reconstruction.anchors if anchor.kind in desired_kinds), None)
        if selected is None:
            return None
        text = selected.value
        confidence = min(0.99, 0.90 + 0.09 * reconstruction.fidelity.score)
        return CandidateAnswer(
            text=text,
            confidence=confidence,
            certificate={
                "memory_augmented_generation": "anchor_reconstruction",
                "memory_answer_kind": selected.kind,
                "memory_selected_segments": reconstruction.selected_segment_ids,
                "memory_anchor_fidelity": reconstruction.fidelity.score,
            },
            cost=CostTrace(generated_tokens=max(1, len(text.split()))),
            raw={
                "memory_augmented_generation": {
                    "answer": text,
                    "anchor_kind": selected.kind,
                    "selected_segment_ids": reconstruction.selected_segment_ids,
                    "fidelity": asdict(reconstruction.fidelity),
                }
            },
        )

    def _select_memory_augmented_answer(self, task: Task, base_answer: CandidateAnswer, reconstruction: MemoryReconstruction | None) -> CandidateAnswer:
        memory_answer = self._memory_anchor_candidate(task, reconstruction)
        if memory_answer is None or memory_answer.confidence <= base_answer.confidence:
            return base_answer
        return CandidateAnswer(
            text=memory_answer.text,
            confidence=memory_answer.confidence,
            certificate={**dict(base_answer.certificate), **dict(memory_answer.certificate), "memory_replaced_base_answer": True},
            cost=base_answer.cost.merge(memory_answer.cost),
            raw={
                **dict(base_answer.raw),
                **dict(memory_answer.raw),
                "memory_base_answer": {"text": base_answer.text, "confidence": base_answer.confidence},
            },
        )

    def _provided_proof_certificate_verified(self, answer: CandidateAnswer) -> bool | None:
        if not bool(answer.certificate.get("proof_carrying_generation")):
            return None
        proof = answer.raw.get("proof_carrying_answer")
        if not isinstance(proof, Mapping):
            return False
        try:
            certificate = ShortCertificate.from_dict(answer.certificate)
            latent_state = LatentProofState.from_dict(proof["latent_state"])
        except Exception:
            return False
        return self.certificate_verifier.verify(certificate, latent_state).passed

    def _certificate_verified(self, hidden: torch.Tensor, task: Task, answer: CandidateAnswer, route: InferenceRoute) -> bool:
        provided = self._provided_proof_certificate_verified(answer)
        if provided is not None:
            return provided
        if route.verifier_level <= 0:
            return True
        head_out = self.certificate_head(hidden.detach())
        state = LatentProofState(f"{task.task_id}-inference", task.task_id, task.skill, head_out.latent_state.detach(), latent_steps=max(1, route.latent_loops))
        cert_type = CertificateType.EXACT_MATCH if task.skill != "code_unit_tests" else CertificateType.CODE_TESTS
        tool = "exact_match" if cert_type == CertificateType.EXACT_MATCH else "code_tests"
        tool_args: Mapping[str, Any] = {"expected": answer.text} if tool == "exact_match" else {"function_name": "solve", "tests": tuple(task.metadata.get("tests", ()))}
        cert = build_certificate(
            certificate_id=f"{task.task_id}-inference-cert",
            task_id=task.task_id,
            skill=task.skill,
            certificate_type=cert_type,
            answer=answer.text,
            claims={"route": route.path.value},
            uncertainty=max(0.0, min(1.0, 1.0 - answer.confidence)),
            latent_state=state,
            anchors=task.anchors,
            tool=tool,
            tool_args=tool_args,
        )
        return self.certificate_verifier.verify(cert, state).passed

    def infer(self, task: Task, *, forced_path: InferencePath | None = None) -> InferenceResult:
        self._reset_trace()
        compiled_answer = self._compiled_frontier_answer(task)
        base_answer = compiled_answer if compiled_answer is not None else CandidateAnswer.coerce(self.agent(task))
        signal = self.router.signal(task, base_answer.confidence)
        route = self.router.route(signal, forced_path)
        prediction = self.budget.predict(route, signal.prompt_tokens)
        reconstruction = None
        if route.use_latent_kv or task.anchors:
            reconstruction = self.memory.reconstruct(task.prompt, required_anchors=task.anchors if task.anchors else None)
            if reconstruction.fidelity.required and not reconstruction.fidelity.passed:
                route = self.router.route(signal, InferencePath.CAREFUL)
                prediction = self.budget.predict(route, signal.prompt_tokens)
            self._record_memory_trace(reconstruction)
        base_answer = self._select_memory_augmented_answer(task, base_answer, reconstruction)
        if route.experts_activated:
            self.trace.record_expert(f"{task.skill}-specialist", route.reason, cost=float(route.experts_activated))
        x = self._input_tensor(task, reconstruction)
        hidden, layers_ran, exit_decision, kernel_dispatches = self.core.forward_route(x, route, self.early_exit)
        future = self.speculative.speculate(hidden, route, task)
        answer = CandidateAnswer(
            text=base_answer.text,
            confidence=base_answer.confidence,
            certificate={**dict(base_answer.certificate), "inference_path": route.path.value, "layers_ran": layers_ran},
            cost=base_answer.cost,
            raw=base_answer.raw,
        )
        verification = self.verifier.oracle_registry.verify(task.skill, task, answer)
        cert_ok = self._certificate_verified(hidden, task, answer, route)
        future_cost = CostTrace(**dict(future.get("cost", {}))) if future else CostTrace()
        total_cost = prediction.predicted_cost.merge(answer.cost).merge(self.trace.cost_trace).merge(future_cost)
        if reconstruction is not None:
            total_cost = total_cost.merge(reconstruction.cost)
        if route.verifier_level > 0:
            total_cost = total_cost.merge(verification.verifier_cost)
        score = verification.score
        if not cert_ok:
            score = 0.0
        vc_per_cost = score / max(total_cost.effective_cost(), 1e-9)
        return InferenceResult(
            task=task,
            route=route,
            answer=answer,
            verification=verification,
            cost=total_cost,
            predicted_cost=prediction,
            early_exit=exit_decision,
            layers_ran=layers_ran,
            memory_reconstruction=reconstruction,
            future_contract=future,
            kernel_dispatches=kernel_dispatches,
            certificate_verified=cert_ok,
            verified_capability_per_cost=vc_per_cost,
            trace_summary=self.trace.to_dict(),
        )
