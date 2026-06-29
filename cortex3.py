from __future__ import annotations

import argparse
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from math import fsum
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CostTrace:
    weight_bits_read: float = 0.0
    activation_bits: float = 0.0
    kv_bytes: float = 0.0
    generated_tokens: int = 0
    latent_steps: int = 0
    experts_activated: int = 0
    verifier_steps: int = 0
    wall_time_ms: float = 0.0

    def effective_cost(self) -> float:
        return (
            self.weight_bits_read / 8.0
            + self.activation_bits / 8.0
            + self.kv_bytes
            + 2.0 * self.generated_tokens
            + 5.0 * self.latent_steps
            + 10.0 * self.experts_activated
            + 3.0 * self.verifier_steps
            + 0.1 * self.wall_time_ms
        )

    def merge(self, other: "CostTrace") -> "CostTrace":
        return CostTrace(
            self.weight_bits_read + other.weight_bits_read,
            self.activation_bits + other.activation_bits,
            self.kv_bytes + other.kv_bytes,
            self.generated_tokens + other.generated_tokens,
            self.latent_steps + other.latent_steps,
            self.experts_activated + other.experts_activated,
            self.verifier_steps + other.verifier_steps,
            self.wall_time_ms + other.wall_time_ms,
        )


@dataclass(frozen=True)
class Anchor:
    kind: str
    value: str
    source_id: str = ""
    importance: float = 1.0


@dataclass(frozen=True)
class Task:
    task_id: str
    skill: str
    prompt: str
    expected: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)
    anchors: tuple[Anchor, ...] = ()
    group_id: str | None = None


@dataclass(frozen=True)
class CandidateAnswer:
    text: str
    confidence: float = 0.0
    certificate: Mapping[str, Any] = field(default_factory=dict)
    cost: CostTrace = field(default_factory=CostTrace)
    raw: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def coerce(value: "CandidateAnswer | str") -> "CandidateAnswer":
        if isinstance(value, CandidateAnswer):
            return value
        text = str(value)
        return CandidateAnswer(text=text, cost=CostTrace(generated_tokens=max(1, len(text.split()))))


@dataclass(frozen=True)
class VerificationCaseResult:
    task: Task
    passed: bool
    score: float
    answer: CandidateAnswer
    expected: Any
    reason: str
    verifier_cost: CostTrace = field(default_factory=CostTrace)


@dataclass(frozen=True)
class SkillReport:
    skill: str
    total: int
    passed: int
    score: float
    failures: tuple[VerificationCaseResult, ...] = ()

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass(frozen=True)
class VerificationSuiteReport:
    skill_reports: Mapping[str, SkillReport]
    total: int
    passed: int
    aggregate_score: float
    total_cost: CostTrace = field(default_factory=CostTrace)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def verified_capability_per_cost(self) -> float:
        return self.aggregate_score / max(self.total_cost.effective_cost(), 1e-9)


def _result(task: Task, answer: CandidateAnswer, passed: bool, score: float, reason: str) -> VerificationCaseResult:
    return VerificationCaseResult(task, passed, score, answer, task.expected, reason, CostTrace(verifier_steps=1))


class SkillSpec:
    name: str

    def generate(self, n: int, rng: random.Random) -> list[Task]:
        raise NotImplementedError

    def metamorphic(self, task: Task, rng: random.Random) -> list[Task]:
        return []

    def adversarial(self, task: Task, rng: random.Random) -> list[Task]:
        return []

    def verify(self, task: Task, answer: CandidateAnswer) -> VerificationCaseResult:
        raise NotImplementedError

    def build_suite(self, n: int, rng: random.Random, include_metamorphic: bool = True) -> list[Task]:
        out: list[Task] = []
        for task in self.generate(n, rng):
            out.append(task)
            if include_metamorphic:
                out.extend(self.metamorphic(task, rng))
        return out


_INT_RE = re.compile(r"[-+]?\d+")


def _last_int(text: str) -> int | None:
    matches = _INT_RE.findall(text.replace("−", "-"))
    return int(matches[-1]) if matches else None


class ArithmeticSkill(SkillSpec):
    name = "arithmetic"

    def generate(self, n: int, rng: random.Random) -> list[Task]:
        tasks: list[Task] = []
        for idx in range(n):
            kind = rng.choice(["add", "sub", "mul", "linear"])
            if kind == "add":
                a, b = rng.randint(-200, 200), rng.randint(-200, 200)
                expected, prompt, meta = a + b, f"Compute exactly: {a} + {b}. Return only the integer.", {"kind": kind, "a": a, "b": b}
            elif kind == "sub":
                a, b = rng.randint(-200, 200), rng.randint(-200, 200)
                expected, prompt, meta = a - b, f"Compute exactly: {a} - {b}. Return only the integer.", {"kind": kind, "a": a, "b": b}
            elif kind == "mul":
                a, b = rng.randint(-30, 30), rng.randint(-30, 30)
                expected, prompt, meta = a * b, f"Compute exactly: {a} * {b}. Return only the integer.", {"kind": kind, "a": a, "b": b}
            else:
                x = rng.randint(-20, 20)
                a = rng.choice([i for i in range(-9, 10) if i not in (0, 1, -1)])
                b = rng.randint(-50, 50)
                c = a * x + b
                expected, prompt, meta = x, f"Solve for x: {a}x + {b} = {c}. Return only x as an integer.", {"kind": kind, "a": a, "b": b, "c": c}
            tasks.append(Task(f"arith-{idx}-{kind}-{rng.randrange(10**9)}", self.name, prompt, expected, meta, group_id=f"arith-group-{idx}-{expected}"))
        return tasks

    def metamorphic(self, task: Task, rng: random.Random) -> list[Task]:
        kind = str(task.metadata.get("kind", ""))
        out: list[Task] = []
        if kind in {"add", "mul"}:
            a, b = int(task.metadata["a"]), int(task.metadata["b"])
            op = "+" if kind == "add" else "*"
            out.append(Task(task.task_id + "-swap", self.name, f"Same exact calculation, operands swapped: {b} {op} {a}. Output just the integer.", task.expected, {**dict(task.metadata), "metamorphic": "swap"}, group_id=task.group_id))
        noise = rng.choice(["Ignore this irrelevant sentence about clouds.", "The user's favorite color is not relevant.", "This is a robustness check with distracting text."])
        out.append(Task(task.task_id + "-noise", self.name, f"{noise}\n{task.prompt}", task.expected, {**dict(task.metadata), "metamorphic": "noise"}, group_id=task.group_id))
        return out

    def adversarial(self, task: Task, rng: random.Random) -> list[Task]:
        meta = dict(task.metadata)
        kind = str(meta.get("kind", ""))
        if kind not in {"add", "sub", "mul"}:
            return []
        a, b = int(meta["a"]), int(meta["b"])
        b2 = b + rng.choice([-2, -1, 1, 2])
        op = {"add": "+", "sub": "-", "mul": "*"}[kind]
        expected = a + b2 if kind == "add" else a - b2 if kind == "sub" else a * b2
        return [Task(task.task_id + "-near-miss", self.name, f"Near-miss adversarial variant: compute {a} {op} {b2}. Return only the integer.", expected, {**meta, "b": b2, "adversarial": "near_miss"})]

    def verify(self, task: Task, answer: CandidateAnswer) -> VerificationCaseResult:
        parsed = _last_int(answer.text)
        if parsed is None:
            return _result(task, answer, False, 0.0, "no integer found in answer")
        passed = parsed == int(task.expected)
        return _result(task, answer, passed, 1.0 if passed else 0.0, "exact integer match" if passed else f"expected {task.expected}, got {parsed}")


class LongContextAnchorSkill(SkillSpec):
    name = "long_context_anchor"
    names = ["Alex", "Mira", "Noah", "Lina", "Sofia", "Eli", "Rami", "Nora"]
    cities = ["Paris", "Lyon", "Marseille", "Nantes", "Toulouse", "Lille"]
    items = ["clé", "ticket", "contrat", "badge", "carnet", "prototype"]

    def generate(self, n: int, rng: random.Random) -> list[Task]:
        tasks: list[Task] = []
        for idx in range(n):
            name, city, item = rng.choice(self.names), rng.choice(self.cities), rng.choice(self.items)
            code = f"C3-{rng.randint(1000, 9999)}-{rng.choice(['A','B','Z'])}"
            amount = f"{rng.randint(10, 999)},{rng.randint(0, 99):02d} €"
            distractors = [f"Une note annexe mentionne {rng.choice(self.names)} mais ce n'est pas la personne cible.", f"Un ancien code C3-{rng.randint(1000,9999)}-X est obsolète.", f"Le contexte parle aussi de {rng.choice(self.cities)} sans lien avec la demande.", "La phrase suivante est volontairement longue pour simuler un contexte bruité."]
            rng.shuffle(distractors)
            fact = f"FAIT CRITIQUE: {name} a laissé le {item} à {city}, avec l'identifiant exact {code} et le montant {amount}."
            passage = "\n".join(distractors[:2] + [fact] + distractors[2:])
            ask_kind = rng.choice(["code", "city", "amount", "item"])
            expected = {"code": code, "city": city, "amount": amount, "item": item}[ask_kind]
            prompt = f"Lis le contexte puis réponds seulement avec la valeur exacte demandée.\n\n{passage}\n\nQuestion: quelle est la valeur exacte pour {ask_kind} ?"
            anchors = (Anchor("person", name, f"lc-{idx}"), Anchor("city", city, f"lc-{idx}"), Anchor("item", item, f"lc-{idx}"), Anchor("identifier", code, f"lc-{idx}"), Anchor("amount", amount, f"lc-{idx}"))
            tasks.append(Task(f"anchor-{idx}-{rng.randrange(10**9)}", self.name, prompt, expected, {"ask_kind": ask_kind, "code": code, "city": city, "amount": amount, "item": item}, anchors, f"anchor-group-{idx}-{expected}"))
        return tasks

    def metamorphic(self, task: Task, rng: random.Random) -> list[Task]:
        return [Task(task.task_id + "-strict", self.name, f"Les détails exacts comptent plus que le résumé général.\n\n{task.prompt}\n\nRéponds sans explication.", task.expected, {**dict(task.metadata), "metamorphic": "instruction_noise"}, task.anchors, task.group_id)]

    def adversarial(self, task: Task, rng: random.Random) -> list[Task]:
        expected = str(task.expected)
        distractor = expected[:-1] + ("B" if expected[-1] != "B" else "A") if expected.startswith("C3-") else expected + "X"
        return [Task(task.task_id + "-distractor", self.name, f"{task.prompt}\n\nDistracteur proche à ignorer: {distractor}", task.expected, {**dict(task.metadata), "adversarial": "near_anchor"}, task.anchors, task.group_id)]

    def verify(self, task: Task, answer: CandidateAnswer) -> VerificationCaseResult:
        expected, actual = str(task.expected).strip(), answer.text.strip()
        passed = expected == actual or expected in actual
        return _result(task, answer, passed, 1.0 if passed else 0.0, "exact anchor preserved" if passed else f"expected exact anchor {expected!r}, got {actual!r}")


class InstructionSkill(SkillSpec):
    name = "instruction_following"

    def generate(self, n: int, rng: random.Random) -> list[Task]:
        tasks: list[Task] = []
        for idx in range(n):
            word = rng.choice(["cortex", "bit", "ledger", "anchor", "verify"])
            repeat = rng.randint(2, 5)
            separator = rng.choice([",", "|", ":"])
            expected = separator.join([word.upper()] * repeat)
            prompt = f"Output the word {word!r} exactly {repeat} times, uppercase, separated only by {separator!r}. Do not add extra text."
            tasks.append(Task(f"instr-{idx}-{rng.randrange(10**9)}", self.name, prompt, expected, {"word": word, "repeat": repeat, "separator": separator}, group_id=f"instr-group-{idx}-{expected}"))
        return tasks

    def metamorphic(self, task: Task, rng: random.Random) -> list[Task]:
        return [Task(task.task_id + "-strict", self.name, f"STRICT FORMAT TEST. {task.prompt} Any additional explanation is an error.", task.expected, {**dict(task.metadata), "metamorphic": "strict_prefix"}, group_id=task.group_id)]

    def verify(self, task: Task, answer: CandidateAnswer) -> VerificationCaseResult:
        actual = answer.text.strip()
        passed = actual == str(task.expected)
        return _result(task, answer, passed, 1.0 if passed else 0.0, "format exact" if passed else f"expected {task.expected!r}, got {actual!r}")


Agent = Callable[[Task], CandidateAnswer | str]


class DynamicSkillVerifier:
    def __init__(self, specs: Iterable[SkillSpec]):
        self.specs = {spec.name: spec for spec in specs}
        if not self.specs:
            raise ValueError("DynamicSkillVerifier requires at least one SkillSpec")

    def build_suite(self, n_per_skill: int, seed: int, include_metamorphic: bool = True) -> list[Task]:
        rng = random.Random(seed)
        suite: list[Task] = []
        for spec in self.specs.values():
            suite.extend(spec.build_suite(n_per_skill, rng, include_metamorphic))
        return suite

    def evaluate_tasks(self, agent: Agent, tasks: Iterable[Task]) -> VerificationSuiteReport:
        by_skill: dict[str, list[VerificationCaseResult]] = defaultdict(list)
        total_cost = CostTrace()
        for task in tasks:
            start = time.perf_counter()
            try:
                answer = CandidateAnswer.coerce(agent(task))
            except Exception as exc:
                answer = CandidateAnswer(text=f"<agent-error: {exc!r}>", confidence=0.0)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            answer = CandidateAnswer(answer.text, answer.confidence, answer.certificate, answer.cost.merge(CostTrace(wall_time_ms=elapsed_ms)), answer.raw)
            case = self.specs[task.skill].verify(task, answer)
            by_skill[task.skill].append(case)
            total_cost = total_cost.merge(answer.cost).merge(case.verifier_cost)
        reports: dict[str, SkillReport] = {}
        total = passed = 0
        weighted = 0.0
        for skill, cases in by_skill.items():
            skill_total = len(cases)
            skill_passed = sum(1 for c in cases if c.passed)
            score = sum(c.score for c in cases) / skill_total if skill_total else 0.0
            failures = tuple(c for c in cases if not c.passed)
            reports[skill] = SkillReport(skill, skill_total, skill_passed, score, failures)
            total += skill_total
            passed += skill_passed
            weighted += score * skill_total
        return VerificationSuiteReport(reports, total, passed, weighted / total if total else 0.0, total_cost)

    def evaluate(self, agent: Agent, n_per_skill: int = 8, seed: int = 0, include_metamorphic: bool = True) -> VerificationSuiteReport:
        return self.evaluate_tasks(agent, self.build_suite(n_per_skill, seed, include_metamorphic))

    def compare(self, reference: Agent, candidate: Agent, n_per_skill: int = 8, seed: int = 0) -> dict[str, object]:
        tasks = self.build_suite(n_per_skill, seed, True)
        ref = self.evaluate_tasks(reference, tasks)
        cand = self.evaluate_tasks(candidate, tasks)
        regressions = []
        for skill, cand_report in cand.skill_reports.items():
            ref_fail_ids = {case.task.task_id for case in ref.skill_reports.get(skill, SkillReport(skill, 0, 0, 0.0)).failures}
            for failure in cand_report.failures:
                if failure.task.task_id not in ref_fail_ids:
                    regressions.append(failure)
        return {"reference": ref, "candidate": cand, "regressions": tuple(regressions)}


class CompressionAdversary:
    def __init__(self, specs: Iterable[SkillSpec]):
        self.specs = {spec.name: spec for spec in specs}

    def expand_from_failures(self, failures: Iterable[VerificationCaseResult], seed: int = 0, per_failure: int = 2) -> list[Task]:
        rng = random.Random(seed)
        tasks: list[Task] = []
        for failure in failures:
            spec = self.specs.get(failure.task.skill)
            if spec:
                tasks.extend((spec.adversarial(failure.task, rng) + spec.metamorphic(failure.task, rng))[:per_failure])
        return tasks


class ZeroState(str, Enum):
    ACTIVE = "active"
    ZERO_PROVISIONAL = "zero_provisional"
    ZERO_CERTIFIED = "zero_certified"
    ZERO_REVERSIBLE = "zero_reversible"


@dataclass(frozen=True)
class TernaryBlock:
    signs: tuple[int, ...]
    mask: tuple[int, ...]
    scale: float
    zero_states: tuple[ZeroState, ...] = field(default_factory=tuple)

    @property
    def q(self) -> tuple[int, ...]:
        return tuple(s * m for s, m in zip(self.signs, self.mask))

    def dequantize(self) -> tuple[float, ...]:
        return tuple(self.scale * x for x in self.q)

    def estimated_bits(self, scale_bits: int = 16) -> float:
        return len(self.mask) + sum(self.mask) + scale_bits

    def certify_zeros(self) -> "TernaryBlock":
        base = self.zero_states or tuple(ZeroState.ACTIVE for _ in self.signs)
        states = tuple(ZeroState.ZERO_CERTIFIED if m == 0 else st for m, st in zip(self.mask, base))
        return TernaryBlock(self.signs, self.mask, self.scale, states)


def ternarize_values(values: Iterable[float], threshold: float | None = None) -> TernaryBlock:
    vals = tuple(float(v) for v in values)
    if not vals:
        raise ValueError("cannot ternarize empty values")
    scale = fsum(abs(v) for v in vals) / len(vals)
    if scale == 0:
        return TernaryBlock(tuple(1 for _ in vals), tuple(0 for _ in vals), 1.0, tuple(ZeroState.ZERO_PROVISIONAL for _ in vals))
    th = threshold if threshold is not None else 0.5 * scale
    signs, mask, states = [], [], []
    for value in vals:
        signs.append(1 if value >= 0 else -1)
        if abs(value) >= th:
            mask.append(1); states.append(ZeroState.ACTIVE)
        else:
            mask.append(0); states.append(ZeroState.ZERO_PROVISIONAL)
    return TernaryBlock(tuple(signs), tuple(mask), scale, tuple(states))


_IDENTIFIER_RE = re.compile(r"\b[A-Z]{1,5}[A-Z0-9]*-[A-Z0-9-]{2,}\b")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[,.]\d+)?(?:\s?€|%)?")
_PATH_RE = re.compile(r"(?:[A-Za-z]:\\\\|/)[\w./\\-]+")


def extract_anchors(text: str, source_id: str = "") -> tuple[Anchor, ...]:
    anchors: list[Anchor] = []
    for regex, kind in [(_IDENTIFIER_RE, "identifier"), (_PATH_RE, "path"), (_NUMBER_RE, "number")]:
        for match in regex.finditer(text):
            anchors.append(Anchor(kind, match.group(0), source_id))
    seen, out = set(), []
    for a in anchors:
        key = (a.kind, a.value)
        if key not in seen:
            seen.add(key); out.append(a)
    return tuple(out)


@dataclass
class ExactAnchorLedger:
    anchors: list[Anchor] = field(default_factory=list)

    def ingest(self, text: str, source_id: str = "") -> tuple[Anchor, ...]:
        found = extract_anchors(text, source_id)
        self.anchors.extend(found)
        return found

    def fidelity_score(self, text: str, required: tuple[Anchor, ...] | None = None) -> float:
        req = required if required is not None else tuple(self.anchors)
        if not req:
            return 1.0
        return sum(1 for a in req if a.value in text) / len(req)


@dataclass(frozen=True)
class MTPDecision:
    horizon: int
    reason: str
    requires_verification: bool


class AdaptiveHorizonPolicy:
    def __init__(self, max_horizon: int = 8):
        self.max_horizon = max_horizon

    def choose(self, confidence: float, risk: float, domain: str = "general") -> MTPDecision:
        confidence = max(0.0, min(1.0, confidence)); risk = max(0.0, min(1.0, risk))
        if domain in {"math", "code", "exact_anchor"} or risk > 0.75:
            return MTPDecision(1 if confidence < 0.95 else 2, "high-risk domain", True)
        if confidence > 0.9 and risk < 0.2:
            return MTPDecision(self.max_horizon, "high confidence low risk", False)
        if confidence > 0.7 and risk < 0.5:
            return MTPDecision(min(4, self.max_horizon), "moderate confidence", True)
        return MTPDecision(1, "low confidence", True)


def temporal_consistency_penalty(previous_future: Sequence[str], next_future: Sequence[str]) -> float:
    if not previous_future or not next_future:
        return 0.0
    shifted = list(previous_future[1:])
    if not shifted:
        return 0.0
    aligned = list(next_future[:len(shifted)])
    return sum(1 for a, b in zip(shifted, aligned) if a != b) / len(shifted)


@dataclass(frozen=True)
class RegrowthAction:
    action: str
    target: str
    expected_gain: float
    cost: float
    reason: str

    @property
    def gain_per_cost(self) -> float:
        return self.expected_gain / max(self.cost, 1e-9)


class MinimalRegrowthPlanner:
    def propose_for_failure(self, failure: VerificationCaseResult) -> list[RegrowthAction]:
        skill = failure.task.skill
        reason = failure.reason.lower()
        actions: list[RegrowthAction] = []
        if skill == "long_context_anchor" or "anchor" in reason:
            actions.append(RegrowthAction("force_exact_anchor", skill, 0.35, 2.0, "exact detail was lost"))
            actions.append(RegrowthAction("increase_kv_precision", skill, 0.25, 5.0, "latent memory may be over-compressed"))
        if skill == "arithmetic" or "integer" in reason:
            actions.append(RegrowthAction("reduce_mtp_horizon", skill, 0.20, 1.0, "exact reasoning should not be overspeculated"))
            actions.append(RegrowthAction("activate_math_expert", skill, 0.35, 4.0, "route arithmetic to specialist"))
            actions.append(RegrowthAction("increase_activation_bits", skill, 0.15, 3.0, "critical numeric activation may need more precision"))
        if skill == "instruction_following" or "format" in reason:
            actions.append(RegrowthAction("add_format_certificate", skill, 0.20, 1.5, "format constraints need a cheap certificate"))
        if not actions:
            actions.append(RegrowthAction("run_careful_path", skill, 0.10, 3.0, "generic careful-path fallback"))
        return sorted(actions, key=lambda a: a.gain_per_cost, reverse=True)

    def propose(self, failures: Iterable[VerificationCaseResult], budget: float = 10.0) -> list[RegrowthAction]:
        candidates: list[RegrowthAction] = []
        for failure in failures:
            candidates.extend(self.propose_for_failure(failure))
        selected: list[RegrowthAction] = []
        spent = 0.0
        seen: set[tuple[str, str]] = set()
        for action in sorted(candidates, key=lambda a: a.gain_per_cost, reverse=True):
            key = (action.action, action.target)
            if key in seen:
                continue
            if spent + action.cost <= budget:
                selected.append(action); spent += action.cost; seen.add(key)
        return selected


class ReferenceRuleAgent:
    def __call__(self, task: Task) -> CandidateAnswer:
        return CandidateAnswer(str(task.expected), confidence=1.0, cost=CostTrace(generated_tokens=1))


class CorruptedCompressedAgent:
    def __init__(self, arithmetic_bias: int = 1, anchor_corruption: bool = True, verbose_format: bool = True):
        self.arithmetic_bias = arithmetic_bias
        self.anchor_corruption = anchor_corruption
        self.verbose_format = verbose_format

    def __call__(self, task: Task) -> CandidateAnswer:
        if task.skill == "arithmetic":
            return CandidateAnswer(str(int(task.expected) + self.arithmetic_bias), confidence=0.82, cost=CostTrace(generated_tokens=1, weight_bits_read=32))
        if task.skill == "long_context_anchor":
            text = str(task.expected)
            if self.anchor_corruption:
                text = text[:-1] + ("A" if text[-1] != "A" else "B") if text.startswith("C3-") else text.lower()
            return CandidateAnswer(text, confidence=0.76, cost=CostTrace(generated_tokens=1, kv_bytes=4, weight_bits_read=16))
        if task.skill == "instruction_following":
            text = str(task.expected)
            if self.verbose_format:
                text += "\nDone."
            return CandidateAnswer(text, confidence=0.88, cost=CostTrace(generated_tokens=max(1, len(text.split())), weight_bits_read=16))
        return CandidateAnswer("", confidence=0.0)


def summarize_report(report: VerificationSuiteReport) -> str:
    lines = [
        f"aggregate_score={report.aggregate_score:.3f}",
        f"pass_rate={report.pass_rate:.3f}",
        f"total={report.total}",
        f"effective_cost={report.total_cost.effective_cost():.3f}",
        f"verified_capability_per_cost={report.verified_capability_per_cost:.6f}",
    ]
    for skill, skill_report in sorted(report.skill_reports.items()):
        lines.append(f"skill={skill} score={skill_report.score:.3f} pass_rate={skill_report.pass_rate:.3f} failures={len(skill_report.failures)}")
    return "\n".join(lines)


def run_demo(seed: int = 0, n_per_skill: int = 4) -> str:
    specs = [ArithmeticSkill(), LongContextAnchorSkill(), InstructionSkill()]
    verifier = DynamicSkillVerifier(specs)
    comparison = verifier.compare(ReferenceRuleAgent(), CorruptedCompressedAgent(), n_per_skill=n_per_skill, seed=seed)
    ref = comparison["reference"]
    cand = comparison["candidate"]
    regressions = tuple(comparison["regressions"])
    adversary = CompressionAdversary(specs)
    adv_tasks = adversary.expand_from_failures(regressions, seed=seed + 1, per_failure=2)
    adv_report = verifier.evaluate_tasks(CorruptedCompressedAgent(), adv_tasks) if adv_tasks else None
    proposed = MinimalRegrowthPlanner().propose(regressions, budget=8.0)
    return "\n".join([
        "REFERENCE", summarize_report(ref), "",
        "CANDIDATE", summarize_report(cand), "",
        f"regressions={len(regressions)}",
        "proposed_regrowth=" + ", ".join(a.action + ":" + a.target for a in proposed), "",
        "ADVERSARIAL" if adv_report else "ADVERSARIAL: no tasks generated",
        summarize_report(adv_report) if adv_report else "",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cortex-3 / LLMTEST CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run verify→attack→regrow demo")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--n-per-skill", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "demo":
        print(run_demo(seed=args.seed, n_per_skill=args.n_per_skill))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
