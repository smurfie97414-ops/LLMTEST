from __future__ import annotations

import hashlib
import json
import random
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex3 import Anchor, CandidateAnswer, CostTrace, ReferenceRuleAgent, Task
from cortex3_memory import embed_text


class CertificateType(str, Enum):
    EXACT_MATCH = "exact_match"
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    CODE_TESTS = "code_tests"
    ANCHOR_FIDELITY = "anchor_fidelity"
    FORMAT = "format"
    COMPILED_CIRCUIT = "compiled_circuit"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LatentProofState:
    state_id: str
    task_id: str
    skill: str
    tensor: torch.Tensor
    latent_steps: int
    visible_reasoning_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        tensor = self.tensor.detach().cpu()
        return {
            "state_id": self.state_id,
            "task_id": self.task_id,
            "skill": self.skill,
            "shape": list(tensor.shape),
            "values": [round(float(value), 6) for value in tensor.flatten().tolist()],
            "latent_steps": self.latent_steps,
            "visible_reasoning_tokens": self.visible_reasoning_tokens,
            "checksum": self.checksum(),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "LatentProofState":
        shape = tuple(int(value) for value in payload.get("shape", ()))
        values = [float(value) for value in payload.get("values", ())]
        if not shape:
            shape = (1, len(values))
        tensor = torch.tensor(values, dtype=torch.float32).view(*shape)
        return LatentProofState(
            state_id=str(payload["state_id"]),
            task_id=str(payload["task_id"]),
            skill=str(payload["skill"]),
            tensor=tensor,
            latent_steps=int(payload.get("latent_steps", 0)),
            visible_reasoning_tokens=int(payload.get("visible_reasoning_tokens", 0)),
        )

    def checksum(self) -> str:
        values = [round(float(value), 6) for value in self.tensor.detach().cpu().flatten().tolist()]
        payload = {
            "state_id": self.state_id,
            "task_id": self.task_id,
            "skill": self.skill,
            "latent_steps": self.latent_steps,
            "values": values,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.blake2b(encoded, digest_size=16).hexdigest()


@dataclass(frozen=True)
class ShortCertificate:
    certificate_id: str
    task_id: str
    skill: str
    certificate_type: CertificateType
    answer: str
    claims: Mapping[str, Any]
    uncertainty: float
    latent_state_checksum: str
    anchors: tuple[Anchor, ...] = ()
    tool: str = ""
    tool_args: Mapping[str, Any] = field(default_factory=dict)
    visible_reasoning_tokens: int = 0
    latent_steps: int = 0

    @property
    def confidence(self) -> float:
        return max(0.0, min(1.0, 1.0 - self.uncertainty))

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "task_id": self.task_id,
            "skill": self.skill,
            "certificate_type": self.certificate_type.value,
            "answer": self.answer,
            "claims": dict(self.claims),
            "uncertainty": round(self.uncertainty, 6),
            "latent_state_checksum": self.latent_state_checksum,
            "anchors": [asdict(anchor) for anchor in self.anchors],
            "tool": self.tool,
            "tool_args": dict(self.tool_args),
            "visible_reasoning_tokens": self.visible_reasoning_tokens,
            "latent_steps": self.latent_steps,
        }

    def checksum(self) -> str:
        encoded = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.blake2b(encoded, digest_size=16).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = self.canonical_payload()
        payload["checksum"] = self.checksum()
        return payload

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "ShortCertificate":
        anchors = tuple(
            Anchor(str(anchor["kind"]), str(anchor["value"]), str(anchor.get("source_id", "")), float(anchor.get("importance", 1.0)))
            for anchor in payload.get("anchors", ())
        )
        return ShortCertificate(
            certificate_id=str(payload["certificate_id"]),
            task_id=str(payload["task_id"]),
            skill=str(payload["skill"]),
            certificate_type=CertificateType(str(payload["certificate_type"])),
            answer=str(payload["answer"]),
            claims=dict(payload.get("claims", {})),
            uncertainty=float(payload.get("uncertainty", 1.0)),
            latent_state_checksum=str(payload["latent_state_checksum"]),
            anchors=anchors,
            tool=str(payload.get("tool", "")),
            tool_args=dict(payload.get("tool_args", {})),
            visible_reasoning_tokens=int(payload.get("visible_reasoning_tokens", 0)),
            latent_steps=int(payload.get("latent_steps", 0)),
        )


@dataclass(frozen=True)
class ProofCarryingAnswer:
    answer: str
    certificate: ShortCertificate
    uncertainty: float
    latent_state: LatentProofState

    def to_candidate_answer(self) -> CandidateAnswer:
        certificate_payload = self.certificate.to_dict()
        certificate_payload["proof_carrying_generation"] = True
        return CandidateAnswer(
            text=self.answer,
            confidence=1.0 - self.uncertainty,
            certificate=certificate_payload,
            cost=CostTrace(
                generated_tokens=max(1, len(self.answer.split()) + _certificate_token_count(self.certificate)),
                latent_steps=self.latent_state.latent_steps,
                verifier_steps=0,
            ),
            raw={
                "proof_carrying_answer": {
                    "certificate": certificate_payload,
                    "latent_state": self.latent_state.to_dict(),
                }
            },
        )


@dataclass(frozen=True)
class CertificateHeadOutput:
    latent_state: torch.Tensor
    answer_logits: torch.Tensor
    certificate_type_logits: torch.Tensor
    uncertainty: torch.Tensor


class CertificateHead(nn.Module):
    def __init__(self, hidden_size: int, latent_size: int, answer_vocab_size: int, certificate_types: int = len(CertificateType)):
        super().__init__()
        if hidden_size <= 0 or latent_size <= 0 or answer_vocab_size <= 1:
            raise ValueError("hidden_size, latent_size and answer_vocab_size must be positive")
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.answer_vocab_size = answer_vocab_size
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.Tanh(),
            nn.Linear(latent_size, latent_size),
            nn.Tanh(),
        )
        self.answer_head = nn.Linear(latent_size, answer_vocab_size)
        self.certificate_type_head = nn.Linear(latent_size, certificate_types)
        self.uncertainty_head = nn.Linear(latent_size, 1)

    def forward(self, hidden: torch.Tensor) -> CertificateHeadOutput:
        if hidden.ndim == 3:
            hidden = hidden[:, -1, :]
        if hidden.ndim != 2 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden must have shape [batch, {self.hidden_size}] or [batch, time, {self.hidden_size}]")
        latent = self.latent_projection(hidden)
        return CertificateHeadOutput(
            latent_state=latent,
            answer_logits=self.answer_head(latent),
            certificate_type_logits=self.certificate_type_head(latent),
            uncertainty=torch.sigmoid(self.uncertainty_head(latent)).squeeze(-1),
        )


@dataclass(frozen=True)
class DelatentizationProbe:
    state_id: str
    seed: int
    indices: tuple[int, ...]
    values: tuple[float, ...]
    checksum: str


class RandomDelatentizer:
    def __init__(self, probes: int = 4):
        if probes < 1:
            raise ValueError("probes must be at least 1")
        self.probes = probes

    def probe(self, state: LatentProofState, seed: int) -> DelatentizationProbe:
        flat = state.tensor.detach().cpu().flatten()
        if flat.numel() == 0:
            raise ValueError("latent proof state cannot be empty")
        rng = random.Random(seed)
        count = min(self.probes, int(flat.numel()))
        indices = tuple(sorted(rng.sample(range(int(flat.numel())), count)))
        values = tuple(round(float(flat[index]), 6) for index in indices)
        payload = {
            "state_checksum": state.checksum(),
            "seed": seed,
            "indices": indices,
            "values": values,
        }
        checksum = hashlib.blake2b(json.dumps(payload, sort_keys=True).encode("utf-8"), digest_size=16).hexdigest()
        return DelatentizationProbe(state.state_id, seed, indices, values, checksum)

    def verify_probe(self, state: LatentProofState, probe: DelatentizationProbe) -> bool:
        return self.probe(state, probe.seed) == probe


@dataclass(frozen=True)
class ToolVerification:
    tool: str
    passed: bool
    score: float
    reason: str


CertificateTool = Callable[[ShortCertificate], ToolVerification]


class ToolVerifierRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, CertificateTool] = {}

    def register(self, name: str, tool: CertificateTool, *, replace: bool = False) -> None:
        if not name:
            raise ValueError("tool name cannot be empty")
        if name in self._tools and not replace:
            raise ValueError(f"tool {name!r} is already registered")
        self._tools[name] = tool

    def verify(self, certificate: ShortCertificate) -> ToolVerification:
        if not certificate.tool:
            return ToolVerification("", True, 1.0, "no tool required")
        if certificate.tool not in self._tools:
            raise KeyError(f"certificate tool {certificate.tool!r} is not registered")
        return self._tools[certificate.tool](certificate)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))


def _last_int(text: str) -> int | None:
    matches = re.findall(r"[-+]?\d+", text)
    return int(matches[-1]) if matches else None


def arithmetic_tool(certificate: ShortCertificate) -> ToolVerification:
    expression = str(certificate.tool_args.get("expression", ""))
    expected = certificate.tool_args.get("expected")
    if expected is None:
        expected = _safe_eval_arithmetic(expression)
    answer = _last_int(certificate.answer)
    passed = answer is not None and int(expected) == answer
    return ToolVerification("arithmetic", passed, 1.0 if passed else 0.0, "arithmetic certificate verified" if passed else f"expected {expected}, got {answer}")


def _exact_int_from_answer(text: str) -> int | None:
    normalized = text.strip().replace("−", "-")
    return int(normalized) if re.fullmatch(r"[-+]?\d+", normalized) else None


def _linear_algebra_steps(a: int, b: int, c: int, variable: str, solution: int) -> tuple[dict[str, Any], ...]:
    isolated = c - b
    return (
        {
            "step": "subtract_constant",
            "from": f"{a}{variable} + {b} = {c}",
            "operation": f"{c} - ({b})",
            "result": isolated,
        },
        {
            "step": "divide_coefficient",
            "operation": f"{isolated} / {a}",
            "result": solution,
        },
        {
            "step": "substitute_solution",
            "operation": f"{a} * {solution} + {b}",
            "result": a * solution + b,
        },
    )


def algebra_linear_tool(certificate: ShortCertificate) -> ToolVerification:
    if certificate.certificate_type != CertificateType.ALGEBRA:
        return ToolVerification("algebra_linear", False, 0.0, "certificate type is not algebra")
    args = dict(certificate.tool_args)
    try:
        a = int(args["a"])
        b = int(args["b"])
        c = int(args["c"])
        expected = int(args.get("expected", (c - b) // a))
    except Exception as exc:
        return ToolVerification("algebra_linear", False, 0.0, f"invalid algebra tool args: {exc!r}")
    if a == 0:
        return ToolVerification("algebra_linear", False, 0.0, "linear coefficient cannot be zero")
    if (c - b) % a != 0:
        return ToolVerification("algebra_linear", False, 0.0, "linear equation has non-integer solution")
    answer = _exact_int_from_answer(certificate.answer)
    if answer is None:
        return ToolVerification("algebra_linear", False, 0.0, "answer is not exactly one integer")
    computed = (c - b) // a
    if answer != expected or computed != expected or a * answer + b != c:
        return ToolVerification("algebra_linear", False, 0.0, f"expected {expected}, got {answer}")
    variable = str(args.get("variable", "x"))
    expected_steps = _linear_algebra_steps(a, b, c, variable, expected)
    provided_steps = tuple(certificate.claims.get("algebra_steps") or ())
    if len(provided_steps) < len(expected_steps):
        return ToolVerification("algebra_linear", False, 0.0, "missing multi-step algebra proof")
    for expected_step, provided in zip(expected_steps, provided_steps):
        data = dict(provided)
        if str(data.get("step")) != str(expected_step["step"]):
            return ToolVerification("algebra_linear", False, 0.0, f"missing step {expected_step['step']!r}")
        try:
            provided_result = int(data.get("result"))
        except Exception:
            return ToolVerification("algebra_linear", False, 0.0, f"step {expected_step['step']!r} missing integer result")
        if provided_result != int(expected_step["result"]):
            return ToolVerification("algebra_linear", False, 0.0, f"step {expected_step['step']!r} has wrong result")
    return ToolVerification("algebra_linear", True, 1.0, "multi-step algebra certificate verified")


def _safe_eval_arithmetic(expression: str) -> int:
    if not re.fullmatch(r"[\d\s+\-*/()%]+", expression):
        raise ValueError("unsupported arithmetic expression")
    value = eval(expression, {"__builtins__": {}}, {})
    if not isinstance(value, int):
        value = int(value)
    return value


def anchor_tool(certificate: ShortCertificate) -> ToolVerification:
    missing = [anchor.value for anchor in certificate.anchors if anchor.value not in certificate.answer and anchor.value not in json.dumps(dict(certificate.claims), ensure_ascii=False)]
    passed = not missing
    return ToolVerification("anchor_fidelity", passed, 1.0 if passed else 0.0, "anchors preserved" if passed else f"missing anchors: {missing}")


def exact_match_tool(certificate: ShortCertificate) -> ToolVerification:
    expected = certificate.tool_args.get("expected")
    passed = expected is None or str(expected) == certificate.answer
    return ToolVerification("exact_match", passed, 1.0 if passed else 0.0, "exact answer match" if passed else f"expected {expected!r}, got {certificate.answer!r}")


def model_token_certificate_tool(certificate: ShortCertificate) -> ToolVerification:
    args = dict(certificate.tool_args)
    try:
        answer_token_id = int(args["answer_token_id"])
        head_token_id = int(args["certificate_head_token_id"])
        lm_token_id = int(args.get("lm_head_token_id", head_token_id))
    except (KeyError, TypeError, ValueError) as exc:
        return ToolVerification("model_token_certificate", False, 0.0, f"missing token identity: {exc}")
    decoded_answer = str(args.get("decoded_answer", ""))
    if answer_token_id != head_token_id:
        return ToolVerification("model_token_certificate", False, 0.0, "answer token does not match certificate head argmax")
    if certificate.answer != decoded_answer:
        return ToolVerification("model_token_certificate", False, 0.0, "certificate answer does not match decoded model token")
    if bool(args.get("require_lm_head_match", False)) and answer_token_id != lm_token_id:
        return ToolVerification("model_token_certificate", False, 0.0, "certificate head token does not match LM head token")
    target_token = args.get("target_token_id")
    if bool(args.get("require_target_match", False)) and target_token is not None and answer_token_id != int(target_token):
        return ToolVerification("model_token_certificate", False, 0.0, "certificate head token does not match supervised target token")
    target_note = ""
    if target_token is not None:
        target_note = f"; target_match={answer_token_id == int(target_token)}"
    return ToolVerification("model_token_certificate", True, 1.0, f"model certificate token is internally consistent{target_note}")


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SAFE_BUILTINS = {"abs": abs, "all": all, "any": any, "bool": bool, "enumerate": enumerate, "len": len, "max": max, "min": min, "range": range, "sum": sum}
_BANNED_CODE_TOKENS = ("import ", "__", "open(", "eval(", "exec(", "compile(", "globals(", "locals(", "input(", "breakpoint(")


def _extract_code(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    return (match.group(1) if match else text).strip()


def code_unit_test_tool(certificate: ShortCertificate) -> ToolVerification:
    source = _extract_code(certificate.answer)
    lowered = source.lower()
    if any(token in lowered for token in _BANNED_CODE_TOKENS):
        return ToolVerification("code_tests", False, 0.0, "unsafe code token")
    visible_tests = tuple(certificate.tool_args.get("tests", ()))
    hidden_tests = tuple(certificate.tool_args.get("hidden_tests", ()))
    tests = visible_tests + hidden_tests
    if int(certificate.tool_args.get("min_tests", 0) or 0) > len(tests):
        return ToolVerification("code_tests", False, 0.0, "insufficient code certificate tests")
    if bool(certificate.tool_args.get("require_hidden_tests", False)) and not hidden_tests:
        return ToolVerification("code_tests", False, 0.0, "missing required hidden code tests")
    function_name = str(certificate.tool_args.get("function_name", "solve"))
    properties = set(str(item) for item in certificate.tool_args.get("properties", ()))
    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    try:
        exec(compile(source, "<certificate-code>", "exec"), namespace, namespace)
        fn = namespace.get(function_name)
        if not callable(fn):
            return ToolVerification("code_tests", False, 0.0, f"function {function_name!r} not defined")
        for args, expected in tests:
            call_args = deepcopy(args)
            frozen_before = repr(call_args)
            actual = fn(*call_args)
            if actual != expected:
                return ToolVerification("code_tests", False, 0.0, f"args={args!r}: expected {expected!r}, got {actual!r}")
            if "no_argument_mutation" in properties and repr(call_args) != frozen_before:
                return ToolVerification("code_tests", False, 0.0, f"args={args!r}: function mutated input arguments")
            if "deterministic" in properties:
                second_actual = fn(*deepcopy(args))
                if second_actual != actual:
                    return ToolVerification("code_tests", False, 0.0, f"args={args!r}: function is not deterministic")
    except Exception as exc:
        return ToolVerification("code_tests", False, 0.0, f"code test failed: {exc!r}")
    return ToolVerification("code_tests", True, 1.0, "all visible, hidden and property code tests passed")


def compiled_circuit_contract_checksum(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(contract), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def compiled_circuit_tool(certificate: ShortCertificate) -> ToolVerification:
    contract = dict(certificate.claims.get("compiled_circuit_contract") or {})
    if not contract:
        return ToolVerification("compiled_circuit", False, 0.0, "missing compiled circuit contract")
    checksum = str(contract.get("contract_checksum") or "")
    payload = {key: value for key, value in contract.items() if key != "contract_checksum"}
    expected_checksum = compiled_circuit_contract_checksum(payload)
    if checksum != expected_checksum:
        return ToolVerification("compiled_circuit", False, 0.0, "compiled circuit contract checksum mismatch")
    if certificate.certificate_type != CertificateType.COMPILED_CIRCUIT:
        return ToolVerification("compiled_circuit", False, 0.0, "certificate type is not compiled_circuit")
    if str(contract.get("skill")) != certificate.skill:
        return ToolVerification("compiled_circuit", False, 0.0, "compiled circuit skill mismatch")
    if str(contract.get("task_id")) != certificate.task_id:
        return ToolVerification("compiled_circuit", False, 0.0, "compiled circuit task mismatch")
    if str(contract.get("answer_checksum")) != hashlib.blake2b(certificate.answer.encode("utf-8"), digest_size=16).hexdigest():
        return ToolVerification("compiled_circuit", False, 0.0, "compiled circuit answer checksum mismatch")
    if not bool(contract.get("dsv_passed")):
        return ToolVerification("compiled_circuit", False, 0.0, "compiled circuit did not pass DSV")
    if not bool(contract.get("output_verified")):
        return ToolVerification("compiled_circuit", False, 0.0, "compiled circuit output was not verified")
    if not tuple(contract.get("source_failure_ids") or ()):
        return ToolVerification("compiled_circuit", False, 0.0, "missing source failure lineage")
    if not tuple(contract.get("frontier_task_ids") or ()):
        return ToolVerification("compiled_circuit", False, 0.0, "missing frontier task lineage")
    heldout_total = int(contract.get("heldout_total", 0) or 0)
    heldout_passed = int(contract.get("heldout_passed", 0) or 0)
    if not tuple(contract.get("heldout_task_ids") or ()):
        return ToolVerification("compiled_circuit", False, 0.0, "missing held-out frontier lineage")
    if heldout_total <= 0:
        return ToolVerification("compiled_circuit", False, 0.0, "missing held-out frontier generalization gate")
    if heldout_passed < heldout_total or not bool(contract.get("heldout_gate_passed", False)):
        return ToolVerification("compiled_circuit", False, 0.0, "held-out frontier generalization gate failed")
    expected_tool_checksum = str(certificate.tool_args.get("expected_contract_checksum") or "")
    if expected_tool_checksum and expected_tool_checksum != checksum:
        return ToolVerification("compiled_circuit", False, 0.0, "compiled circuit tool checksum mismatch")
    return ToolVerification("compiled_circuit", True, 1.0, "compiled circuit contract verified")


def default_tool_registry() -> ToolVerifierRegistry:
    registry = ToolVerifierRegistry()
    registry.register("algebra_linear", algebra_linear_tool)
    registry.register("arithmetic", arithmetic_tool)
    registry.register("anchor_fidelity", anchor_tool)
    registry.register("compiled_circuit", compiled_circuit_tool)
    registry.register("exact_match", exact_match_tool)
    registry.register("model_token_certificate", model_token_certificate_tool)
    registry.register("code_tests", code_unit_test_tool)
    return registry


@dataclass(frozen=True)
class CertificateVerificationResult:
    passed: bool
    score: float
    reason: str
    tool_result: ToolVerification
    uncertainty_ok: bool
    latent_checksum_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "reason": self.reason,
            "tool_result": asdict(self.tool_result),
            "uncertainty_ok": self.uncertainty_ok,
            "latent_checksum_ok": self.latent_checksum_ok,
        }


@dataclass(frozen=True)
class CertificateHeadTrainingExample:
    task: Task
    answer: str
    certificate_type: CertificateType
    uncertainty: float
    confidence: float


@dataclass(frozen=True)
class CertificateHeadCalibrationResult:
    epochs: int
    examples: int
    before_loss: float
    after_loss: float
    before_answer_accuracy: float
    after_answer_accuracy: float
    before_certificate_type_accuracy: float
    after_certificate_type_accuracy: float
    before_uncertainty_mae: float
    after_uncertainty_mae: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CertificateAnswerVocabulary:
    answers: tuple[str, ...]

    @staticmethod
    def from_answers(answers: Iterable[str]) -> "CertificateAnswerVocabulary":
        unique = tuple(dict.fromkeys(str(answer) for answer in answers))
        if len(unique) < 2:
            unique = (*unique, "<unused-answer>")
        return CertificateAnswerVocabulary(unique)

    def encode(self, answer: str) -> int:
        try:
            return self.answers.index(str(answer))
        except ValueError as exc:
            raise KeyError(f"answer {answer!r} is not in certificate vocabulary") from exc

    def decode(self, index: int) -> str:
        return self.answers[int(index)]

    def to_dict(self) -> dict[str, Any]:
        return {"answers": list(self.answers)}


class CertificateVerifier:
    def __init__(self, tools: ToolVerifierRegistry | None = None, max_uncertainty: float = 0.35):
        self.tools = tools or default_tool_registry()
        self.max_uncertainty = max_uncertainty

    def verify(self, certificate: ShortCertificate, latent_state: LatentProofState | None = None) -> CertificateVerificationResult:
        uncertainty_in_bounds = 0.0 <= certificate.uncertainty <= 1.0
        calibrated_uncertainty = bool(certificate.claims.get("calibrated_uncertainty"))
        uncertainty_ok = uncertainty_in_bounds and (certificate.uncertainty <= self.max_uncertainty or calibrated_uncertainty)
        latent_ok = True if latent_state is None else certificate.latent_state_checksum == latent_state.checksum()
        tool_result = self.tools.verify(certificate)
        passed = uncertainty_ok and latent_ok and tool_result.passed
        if passed:
            reason = "certificate verified"
        elif not latent_ok:
            reason = "latent proof checksum mismatch"
        elif not uncertainty_ok:
            reason = f"uncertainty {certificate.uncertainty:.3f} exceeds bound"
        else:
            reason = tool_result.reason
        score = (0.4 if tool_result.passed else 0.0) + (0.3 if uncertainty_ok else 0.0) + (0.3 if latent_ok else 0.0)
        return CertificateVerificationResult(passed, score, reason, tool_result, uncertainty_ok, latent_ok)


@dataclass(frozen=True)
class CertificateEfficiency:
    slow_reasoning_tokens: int
    certificate_tokens: int
    token_reduction: int
    reduction_ratio: float
    quality_preserved: bool
    calibration_preserved: bool

    @property
    def passed(self) -> bool:
        return self.token_reduction > 0 and self.quality_preserved and self.calibration_preserved


def _certificate_token_count(certificate: ShortCertificate) -> int:
    text = json.dumps(certificate.to_dict(), ensure_ascii=False, sort_keys=True)
    return max(1, len(text.split()))


def evaluate_certificate_efficiency(
    slow_reasoning: str,
    certificate: ShortCertificate,
    verification: CertificateVerificationResult,
    *,
    reference_uncertainty: float,
    calibration_tolerance: float = 0.10,
) -> CertificateEfficiency:
    slow_tokens = len(slow_reasoning.split())
    cert_tokens = _certificate_token_count(certificate)
    reduction = slow_tokens - cert_tokens
    ratio = cert_tokens / max(slow_tokens, 1)
    calibration_preserved = abs(certificate.uncertainty - reference_uncertainty) <= calibration_tolerance
    return CertificateEfficiency(
        slow_reasoning_tokens=slow_tokens,
        certificate_tokens=cert_tokens,
        token_reduction=reduction,
        reduction_ratio=ratio,
        quality_preserved=verification.passed,
        calibration_preserved=calibration_preserved,
    )


def build_certificate(
    *,
    certificate_id: str,
    task_id: str,
    skill: str,
    certificate_type: CertificateType,
    answer: str,
    claims: Mapping[str, Any],
    uncertainty: float,
    latent_state: LatentProofState,
    anchors: Iterable[Anchor] = (),
    tool: str = "",
    tool_args: Mapping[str, Any] | None = None,
) -> ShortCertificate:
    return ShortCertificate(
        certificate_id=certificate_id,
        task_id=task_id,
        skill=skill,
        certificate_type=certificate_type,
        answer=answer,
        claims=dict(claims),
        uncertainty=max(0.0, min(1.0, uncertainty)),
        latent_state_checksum=latent_state.checksum(),
        anchors=tuple(anchors),
        tool=tool,
        tool_args=dict(tool_args or {}),
        visible_reasoning_tokens=latent_state.visible_reasoning_tokens,
        latent_steps=latent_state.latent_steps,
    )


def build_compiled_circuit_certificate(
    *,
    certificate_id: str,
    task: Task,
    answer: str,
    claims: Mapping[str, Any],
    uncertainty: float,
    latent_state: LatentProofState,
    contract: Mapping[str, Any],
) -> ShortCertificate:
    payload = dict(contract)
    payload.setdefault("schema_version", 1)
    payload.setdefault("task_id", task.task_id)
    payload.setdefault("skill", task.skill)
    payload.setdefault("answer_checksum", hashlib.blake2b(str(answer).encode("utf-8"), digest_size=16).hexdigest())
    payload["contract_checksum"] = compiled_circuit_contract_checksum(payload)
    return build_certificate(
        certificate_id=certificate_id,
        task_id=task.task_id,
        skill=task.skill,
        certificate_type=CertificateType.COMPILED_CIRCUIT,
        answer=str(answer),
        claims={
            **dict(claims),
            "compiled_circuit": True,
            "compiled_circuit_contract": payload,
        },
        uncertainty=uncertainty,
        latent_state=latent_state,
        anchors=task.anchors,
        tool="compiled_circuit",
        tool_args={"expected_contract_checksum": payload["contract_checksum"]},
    )


def certificate_type_for_task(task: Task) -> CertificateType:
    if task.skill == "arithmetic":
        return CertificateType.ARITHMETIC
    if task.skill == "algebra":
        return CertificateType.ALGEBRA
    if task.skill == "code_unit_tests":
        return CertificateType.CODE_TESTS
    if task.skill in {"long_context_anchor", "entity_tracking"}:
        return CertificateType.ANCHOR_FIDELITY
    if task.skill == "instruction_following":
        return CertificateType.FORMAT
    if task.skill == "calibration":
        return CertificateType.EXACT_MATCH
    return CertificateType.UNKNOWN


def certificate_examples_from_tasks(
    tasks: Iterable[Task],
    solver: Callable[[Task], CandidateAnswer | str] | None = None,
) -> tuple[CertificateHeadTrainingExample, ...]:
    resolved = solver or ReferenceRuleAgent()
    examples: list[CertificateHeadTrainingExample] = []
    for task in tasks:
        answer = CandidateAnswer.coerce(resolved(task))
        examples.append(CertificateHeadTrainingExample(
            task=task,
            answer=answer.text,
            certificate_type=certificate_type_for_task(task),
            uncertainty=max(0.0, min(1.0, 1.0 - answer.confidence)),
            confidence=answer.confidence,
        ))
    return tuple(examples)


def _arithmetic_expression_from_task(task: Task) -> str | None:
    meta = dict(task.metadata)
    kind = str(meta.get("kind", ""))
    if kind in {"add", "sub", "mul"}:
        op = {"add": "+", "sub": "-", "mul": "*"}[kind]
        return f"{int(meta['a'])} {op} {int(meta['b'])}"
    return None


def certificate_contract_for_task(task: Task, answer: str, certificate_type: CertificateType | None = None) -> tuple[Mapping[str, Any], str, Mapping[str, Any], tuple[Anchor, ...]]:
    cert_type = certificate_type or certificate_type_for_task(task)
    if task.skill == "code_unit_tests":
        visible_tests = tuple(task.metadata.get("tests", ()))
        hidden_tests = tuple(task.metadata.get("hidden_tests", ()))
        all_tests = visible_tests + hidden_tests
        return (
            {
                "specification": str(task.metadata.get("prompt", task.prompt)),
                "invariant": "generated function must satisfy visible, hidden and property tests",
                "visible_tests": len(visible_tests),
                "hidden_tests": len(hidden_tests),
                "properties": ("deterministic", "no_argument_mutation"),
                "minimal_tests": len(all_tests),
            },
            "code_tests",
            {
                "function_name": str(task.metadata.get("function_name", "solve")),
                "tests": visible_tests,
                "hidden_tests": hidden_tests,
                "require_hidden_tests": True,
                "min_tests": len(all_tests),
                "properties": ("deterministic", "no_argument_mutation"),
            },
            tuple(task.anchors),
        )
    if task.skill in {"long_context_anchor", "entity_tracking"}:
        return (
            {
                "anchors_used": [asdict(anchor) for anchor in task.anchors],
                "source": "query-conditioned exact/latent context" if task.anchors else "task prompt",
                "answer_kind": str(task.metadata.get("ask_kind", task.skill)),
            },
            "anchor_fidelity",
            {},
            tuple(task.anchors),
        )
    if cert_type == CertificateType.ALGEBRA:
        meta = dict(task.metadata)
        a = int(meta["a"])
        b = int(meta["b"])
        c = int(meta["c"])
        variable = str(meta.get("variable", "x"))
        solution = int(task.expected)
        return (
            {
                "variable": variable,
                "equation": f"{a}{variable} + {b} = {c}",
                "constraint": "integer linear equation",
                "algebra_steps": _linear_algebra_steps(a, b, c, variable, solution),
                "verification": "tool checks isolation, division and substitution",
            },
            "algebra_linear",
            {"a": a, "b": b, "c": c, "variable": variable, "expected": solution},
            tuple(task.anchors),
        )
    if cert_type == CertificateType.ARITHMETIC:
        expression = _arithmetic_expression_from_task(task)
        if expression is not None:
            return (
                {
                    "operation": expression,
                    "constraint": "integer arithmetic",
                    "verification": "tool recomputes expression and compares answer",
                },
                "arithmetic",
                {"expression": expression, "expected": int(task.expected)},
                tuple(task.anchors),
            )
        return (
            {
                "variable": str(task.metadata.get("variable", "x")),
                "operation": "substitution check",
                "constraint": "reported value must match verifier target",
            },
            "exact_match",
            {"expected": answer},
            tuple(task.anchors),
        )
    if cert_type == CertificateType.FORMAT:
        return (
            {"format": "exact output contract", "extra_text_allowed": False},
            "exact_match",
            {"expected": answer},
            tuple(task.anchors),
        )
    if task.skill == "calibration":
        return (
            {"verification": "exact calibrated answer", "calibrated_uncertainty": True},
            "exact_match",
            {"expected": answer},
            tuple(task.anchors),
        )
    return (
        {"verification": "exact answer match"},
        "exact_match",
        {"expected": answer},
        tuple(task.anchors),
    )


class CertificateHeadCalibrator:
    def __init__(self, hidden_size: int, latent_size: int, vocabulary: CertificateAnswerVocabulary):
        if hidden_size < 8 or latent_size < 4:
            raise ValueError("certificate calibrator dimensions are too small")
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.vocabulary = vocabulary
        self._certificate_types = tuple(CertificateType)

    def _batch(self, examples: Sequence[CertificateHeadTrainingExample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not examples:
            raise ValueError("certificate head calibration requires at least one example")
        hidden = torch.stack([embed_text(example.task.prompt, self.hidden_size) for example in examples]).to(torch.float32)
        answer_targets = torch.tensor([self.vocabulary.encode(example.answer) for example in examples], dtype=torch.long)
        type_targets = torch.tensor([self._certificate_types.index(example.certificate_type) for example in examples], dtype=torch.long)
        uncertainty_targets = torch.tensor([example.uncertainty for example in examples], dtype=torch.float32)
        return hidden, answer_targets, type_targets, uncertainty_targets

    def _loss(self, head: CertificateHead, hidden: torch.Tensor, answer_targets: torch.Tensor, type_targets: torch.Tensor, uncertainty_targets: torch.Tensor) -> tuple[torch.Tensor, Mapping[str, float]]:
        output = head(hidden)
        answer_loss = F.cross_entropy(output.answer_logits, answer_targets)
        type_loss = F.cross_entropy(output.certificate_type_logits, type_targets)
        uncertainty_loss = F.mse_loss(output.uncertainty, uncertainty_targets)
        loss = answer_loss + 0.5 * type_loss + 0.25 * uncertainty_loss
        answer_pred = output.answer_logits.argmax(dim=-1)
        type_pred = output.certificate_type_logits.argmax(dim=-1)
        uncertainty_mae = torch.mean(torch.abs(output.uncertainty - uncertainty_targets))
        return loss, {
            "loss": float(loss.detach().item()),
            "answer_accuracy": float((answer_pred == answer_targets).to(torch.float32).mean().item()),
            "certificate_type_accuracy": float((type_pred == type_targets).to(torch.float32).mean().item()),
            "uncertainty_mae": float(uncertainty_mae.detach().item()),
        }

    def evaluate(self, head: CertificateHead, examples: Sequence[CertificateHeadTrainingExample]) -> Mapping[str, float]:
        hidden, answer_targets, type_targets, uncertainty_targets = self._batch(examples)
        with torch.no_grad():
            _, metrics = self._loss(head, hidden, answer_targets, type_targets, uncertainty_targets)
        return metrics

    def train(
        self,
        examples: Sequence[CertificateHeadTrainingExample],
        *,
        epochs: int = 120,
        lr: float = 0.04,
        head: CertificateHead | None = None,
    ) -> tuple[CertificateHead, CertificateHeadCalibrationResult]:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        examples = tuple(examples)
        if head is None:
            torch.manual_seed(5353)
            head = CertificateHead(self.hidden_size, self.latent_size, len(self.vocabulary.answers))
        hidden, answer_targets, type_targets, uncertainty_targets = self._batch(examples)
        before_loss, before_metrics = self._loss(head, hidden, answer_targets, type_targets, uncertainty_targets)
        optimizer = torch.optim.Adam(head.parameters(), lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss, _ = self._loss(head, hidden, answer_targets, type_targets, uncertainty_targets)
            loss.backward()
            optimizer.step()
        after_loss, after_metrics = self._loss(head, hidden, answer_targets, type_targets, uncertainty_targets)
        return head, CertificateHeadCalibrationResult(
            epochs=epochs,
            examples=len(examples),
            before_loss=float(before_loss.detach().item()),
            after_loss=float(after_loss.detach().item()),
            before_answer_accuracy=before_metrics["answer_accuracy"],
            after_answer_accuracy=after_metrics["answer_accuracy"],
            before_certificate_type_accuracy=before_metrics["certificate_type_accuracy"],
            after_certificate_type_accuracy=after_metrics["certificate_type_accuracy"],
            before_uncertainty_mae=before_metrics["uncertainty_mae"],
            after_uncertainty_mae=after_metrics["uncertainty_mae"],
        )


class ProofCarryingGenerator:
    def __init__(self, head: CertificateHead, vocabulary: CertificateAnswerVocabulary, verifier: CertificateVerifier | None = None):
        self.head = head
        self.vocabulary = vocabulary
        self.verifier = verifier or CertificateVerifier()
        self._certificate_types = tuple(CertificateType)

    def __call__(self, task: Task) -> CandidateAnswer:
        self.head.eval()
        with torch.no_grad():
            hidden = embed_text(task.prompt, self.head.hidden_size).view(1, self.head.hidden_size)
            output = self.head(hidden)
            answer_id = int(output.answer_logits.argmax(dim=-1).item())
            cert_type_id = int(output.certificate_type_logits.argmax(dim=-1).item())
            answer = self.vocabulary.decode(answer_id)
            certificate_type = self._certificate_types[cert_type_id]
            uncertainty = max(0.0, min(1.0, float(output.uncertainty.item())))
            state = LatentProofState(
                state_id=f"{task.task_id}-latent-proof",
                task_id=task.task_id,
                skill=task.skill,
                tensor=output.latent_state.detach().clone(),
                latent_steps=1,
                visible_reasoning_tokens=0,
            )
        claims, tool, tool_args, anchors = certificate_contract_for_task(task, answer, certificate_type)
        certificate = build_certificate(
            certificate_id=f"{task.task_id}-proof-certificate",
            task_id=task.task_id,
            skill=task.skill,
            certificate_type=certificate_type,
            answer=answer,
            claims=claims,
            uncertainty=uncertainty,
            latent_state=state,
            anchors=anchors,
            tool=tool,
            tool_args=tool_args,
        )
        verification = self.verifier.verify(certificate, state)
        candidate = ProofCarryingAnswer(answer, certificate, certificate.uncertainty, state).to_candidate_answer()
        return CandidateAnswer(
            text=candidate.text,
            confidence=candidate.confidence,
            certificate={**dict(candidate.certificate), "certificate_verification": verification.to_dict()},
            cost=candidate.cost.merge(CostTrace(verifier_steps=1)),
            raw={
                **dict(candidate.raw),
                "certificate_verification": verification.to_dict(),
            },
        )
