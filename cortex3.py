from __future__ import annotations

import argparse
import random
import re
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from math import fsum
from typing import Any, Callable, Iterable, Mapping, Sequence

import sympy as sp


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
    cases: tuple[VerificationCaseResult, ...] = ()

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

    def anti_metamorphic(self, task: Task, rng: random.Random) -> list[Task]:
        return self.adversarial(task, rng)

    def adversarial(self, task: Task, rng: random.Random) -> list[Task]:
        return []

    def verify(self, task: Task, answer: CandidateAnswer) -> VerificationCaseResult:
        raise NotImplementedError

    def build_suite(self, n: int, rng: random.Random, include_metamorphic: bool = True, include_anti_metamorphic: bool = False) -> list[Task]:
        out: list[Task] = []
        for task in self.generate(n, rng):
            out.append(task)
            if include_metamorphic:
                out.extend(self.metamorphic(task, rng))
            if include_anti_metamorphic:
                out.extend(self.anti_metamorphic(task, rng))
        return out


Oracle = Callable[[Task, CandidateAnswer], VerificationCaseResult]


class OracleRegistry:
    def __init__(self) -> None:
        self._oracles: dict[str, Oracle] = {}

    def register(self, name: str, oracle: Oracle, *, replace: bool = False) -> None:
        if not name:
            raise ValueError("oracle name cannot be empty")
        if name in self._oracles and not replace:
            raise ValueError(f"oracle {name!r} is already registered")
        self._oracles[name] = oracle

    def verify(self, name: str, task: Task, answer: CandidateAnswer) -> VerificationCaseResult:
        try:
            oracle = self._oracles[name]
        except KeyError as exc:
            raise KeyError(f"no oracle registered for skill {name!r}") from exc
        return oracle(task, answer)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._oracles))


@dataclass(frozen=True)
class SkillCostProfile:
    skill: str
    cases: int
    failures: int
    effective_cost: float
    average_effective_cost: float
    verifier_steps: int = 0
    wall_time_ms: float = 0.0
    max_case_effective_cost: float = 0.0


@dataclass(frozen=True)
class VerifierCostProfile:
    total_cases: int
    total_effective_cost: float
    average_effective_cost: float
    by_skill: Mapping[str, SkillCostProfile]


class VerifierCostProfiler:
    def summarize(self, report: VerificationSuiteReport) -> VerifierCostProfile:
        by_skill: dict[str, SkillCostProfile] = {}
        total_effective = 0.0
        for skill, skill_report in report.skill_reports.items():
            cases = skill_report.cases
            case_costs = tuple(case.verifier_cost for case in cases)
            effective_values = tuple(cost.effective_cost() for cost in case_costs)
            effective = fsum(effective_values)
            total_effective += effective
            by_skill[skill] = SkillCostProfile(
                skill=skill,
                cases=skill_report.total,
                failures=len(skill_report.failures),
                effective_cost=effective,
                average_effective_cost=effective / max(skill_report.total, 1),
                verifier_steps=sum(cost.verifier_steps for cost in case_costs),
                wall_time_ms=fsum(cost.wall_time_ms for cost in case_costs),
                max_case_effective_cost=max(effective_values, default=0.0),
            )
        return VerifierCostProfile(
            total_cases=report.total,
            total_effective_cost=total_effective,
            average_effective_cost=total_effective / max(report.total, 1),
            by_skill=by_skill,
        )


_INT_RE = re.compile(r"[-+]?\d+")
_EXACT_INT_RE = re.compile(r"[-+]?\d+")


def _last_int(text: str) -> int | None:
    matches = _INT_RE.findall(text.replace("−", "-"))
    return int(matches[-1]) if matches else None


def _exact_int(text: str) -> int | None:
    normalized = text.strip().replace("−", "-")
    return int(normalized) if _EXACT_INT_RE.fullmatch(normalized) else None


def _exact_int_tuple(text: str) -> tuple[int, ...] | None:
    normalized = text.strip().replace("−", "-")
    if len(normalized) >= 2 and normalized[0] in "([{" and normalized[-1] in ")]}":
        normalized = normalized[1:-1].strip()
    if not normalized:
        return None
    values: list[int] = []
    for token in normalized.split(","):
        token = token.strip()
        if not _EXACT_INT_RE.fullmatch(token):
            return None
        values.append(int(token))
    return tuple(sorted(values)) if values else None


def _format_assignment_solution(solution: Mapping[str, Any], variables: Sequence[str]) -> str:
    return ", ".join(f"{variable}={int(solution[variable])}" for variable in variables)


def _parse_exact_assignment_solution(text: str, variables: Sequence[str]) -> dict[str, int] | None:
    normalized = text.strip().replace("−", "-")
    if len(normalized) >= 2 and normalized[0] in "([{" and normalized[-1] in ")]}":
        normalized = normalized[1:-1].strip()
    if not normalized:
        return None
    expected_variables = tuple(str(variable) for variable in variables)
    assignments: dict[str, int] = {}
    for token in normalized.split(","):
        token = token.strip()
        match = re.fullmatch(r"([A-Za-z]\w{0,15})\s*=\s*([-+]?\d+)", token)
        if match is None:
            return None
        variable, value = match.group(1), int(match.group(2))
        if variable not in expected_variables or variable in assignments:
            return None
        assignments[variable] = value
    return assignments if tuple(assignments) == expected_variables else None


@dataclass(frozen=True)
class OracleQualityProbeResult:
    skill: str
    total: int
    false_positives: int
    false_negatives: int
    false_positive_examples: tuple[str, ...] = ()
    false_negative_examples: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.false_positives == 0 and self.false_negatives == 0

    @property
    def false_positive_rate(self) -> float:
        return self.false_positives / self.total if self.total else 0.0

    @property
    def false_negative_rate(self) -> float:
        return self.false_negatives / self.total if self.total else 0.0


@dataclass(frozen=True)
class OracleQualityReport:
    total: int
    passed: bool
    by_skill: Mapping[str, OracleQualityProbeResult]


class OracleQualityAuditor:
    def __init__(self, specs: Iterable[SkillSpec], oracle_registry: OracleRegistry | None = None):
        self.specs = tuple(specs)
        self.oracle_registry = oracle_registry or OracleRegistry()
        for spec in self.specs:
            self.oracle_registry.register(spec.name, spec.verify, replace=True)

    def _wrong_answer(self, task: Task) -> CandidateAnswer:
        if task.skill == "code_unit_tests":
            function_name = str(task.metadata.get("function_name", "solve"))
            return CandidateAnswer(f"def {function_name}(*args):\n    return '__oracle_wrong__'\n", confidence=0.99)
        if isinstance(task.expected, int):
            return CandidateAnswer(str(int(task.expected) + 1), confidence=0.99)
        expected = str(task.expected)
        if expected == "UNKNOWN":
            return CandidateAnswer("C3-FAKE-Z", confidence=0.99)
        return CandidateAnswer(f"{expected} EXTRA", confidence=0.99)

    def audit(self, *, seed: int = 0, n_per_skill: int = 3) -> OracleQualityReport:
        rng = random.Random(seed)
        results: dict[str, OracleQualityProbeResult] = {}
        total = 0
        for spec in self.specs:
            false_positive_ids: list[str] = []
            false_negative_ids: list[str] = []
            tasks = spec.build_suite(n_per_skill, rng, include_metamorphic=True, include_anti_metamorphic=True)
            for task in tasks:
                correct = CandidateAnswer.coerce(ReferenceRuleAgent()(task))
                wrong = self._wrong_answer(task)
                if not self.oracle_registry.verify(task.skill, task, correct).passed:
                    false_negative_ids.append(task.task_id)
                if self.oracle_registry.verify(task.skill, task, wrong).passed:
                    false_positive_ids.append(task.task_id)
            total += len(tasks)
            results[spec.name] = OracleQualityProbeResult(
                spec.name,
                len(tasks),
                len(false_positive_ids),
                len(false_negative_ids),
                tuple(false_positive_ids[:10]),
                tuple(false_negative_ids[:10]),
            )
        return OracleQualityReport(total, all(result.passed for result in results.values()), results)


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
        parsed = _exact_int(answer.text)
        if parsed is None:
            return _result(task, answer, False, 0.0, "answer is not exactly one integer")
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
        passed = expected == actual
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


class AlgebraSkill(SkillSpec):
    name = "algebra"

    def generate(self, n: int, rng: random.Random) -> list[Task]:
        tasks: list[Task] = []
        variables = ["x", "y", "z"]
        for idx in range(n):
            if rng.choice(["linear", "linear", "linear_system_2x2"]) == "linear_system_2x2":
                x_var, y_var = rng.sample(variables, 2)
                solution = {x_var: rng.randint(-12, 12), y_var: rng.randint(-12, 12)}
                while True:
                    coefficients = (
                        (rng.choice([i for i in range(-8, 9) if i != 0]), rng.choice([i for i in range(-8, 9) if i != 0])),
                        (rng.choice([i for i in range(-8, 9) if i != 0]), rng.choice([i for i in range(-8, 9) if i != 0])),
                    )
                    determinant = coefficients[0][0] * coefficients[1][1] - coefficients[0][1] * coefficients[1][0]
                    if determinant != 0:
                        break
                rhs = (
                    coefficients[0][0] * solution[x_var] + coefficients[0][1] * solution[y_var],
                    coefficients[1][0] * solution[x_var] + coefficients[1][1] * solution[y_var],
                )
                expected = _format_assignment_solution(solution, (x_var, y_var))
                prompt = (
                    f"Solve the exact 2x2 system for {x_var} and {y_var}: "
                    f"{coefficients[0][0]}{x_var} + {coefficients[0][1]}{y_var} = {rhs[0]}; "
                    f"{coefficients[1][0]}{x_var} + {coefficients[1][1]}{y_var} = {rhs[1]}. "
                    f"Return only assignments as '{x_var}=..., {y_var}=...'."
                )
                metadata = {
                    "kind": "linear_system_2x2",
                    "variables": (x_var, y_var),
                    "coefficients": coefficients,
                    "rhs": rhs,
                    "solution": solution,
                }
                tasks.append(Task(f"algebra-system-{idx}-{rng.randrange(10**9)}", self.name, prompt, expected, metadata, group_id=f"algebra-system-group-{idx}-{expected}"))
                continue
            variable = rng.choice(variables)
            solution_int = rng.randint(-25, 25)
            a = rng.choice([i for i in range(-12, 13) if i not in (0, 1, -1)])
            b = rng.randint(-80, 80)
            c = a * solution_int + b
            prompt = f"Solve exactly for {variable}: {a}{variable} + {b} = {c}. Return only the integer value of {variable}."
            metadata = {"variable": variable, "solution": solution_int, "a": a, "b": b, "c": c, "kind": "linear"}
            tasks.append(Task(f"algebra-{idx}-{rng.randrange(10**9)}", self.name, prompt, solution_int, metadata, group_id=f"algebra-group-{idx}-{solution_int}"))
        return tasks

    def metamorphic(self, task: Task, rng: random.Random) -> list[Task]:
        meta = dict(task.metadata)
        kind = str(meta.get("kind", "linear"))
        if kind == "linear_system_2x2":
            coefficients = tuple(tuple(int(value) for value in row) for row in meta["coefficients"])
            rhs = tuple(int(value) for value in meta["rhs"])
            variables = tuple(str(variable) for variable in meta["variables"])
            swapped = Task(
                task.task_id + "-swap-equations",
                self.name,
                f"Same system with equations swapped. Solve for {variables[0]} and {variables[1]}: "
                f"{coefficients[1][0]}{variables[0]} + {coefficients[1][1]}{variables[1]} = {rhs[1]}; "
                f"{coefficients[0][0]}{variables[0]} + {coefficients[0][1]}{variables[1]} = {rhs[0]}. "
                f"Return only assignments as '{variables[0]}=..., {variables[1]}=...'.",
                task.expected,
                {**meta, "coefficients": (coefficients[1], coefficients[0]), "rhs": (rhs[1], rhs[0]), "metamorphic": "swap_equations"},
                group_id=task.group_id,
            )
            scale = rng.choice([2, -2, 3])
            scaled = Task(
                task.task_id + "-scale-first-equation",
                self.name,
                f"Equivalent system after scaling the first equation by {scale}. Solve for {variables[0]} and {variables[1]}: "
                f"{coefficients[0][0] * scale}{variables[0]} + {coefficients[0][1] * scale}{variables[1]} = {rhs[0] * scale}; "
                f"{coefficients[1][0]}{variables[0]} + {coefficients[1][1]}{variables[1]} = {rhs[1]}. "
                f"Return only assignments as '{variables[0]}=..., {variables[1]}=...'.",
                task.expected,
                {**meta, "coefficients": ((coefficients[0][0] * scale, coefficients[0][1] * scale), coefficients[1]), "rhs": (rhs[0] * scale, rhs[1]), "metamorphic": "scale_equation"},
                group_id=task.group_id,
            )
            return [swapped, scaled]
        a, b, c = int(meta["a"]), int(meta["b"]), int(meta["c"])
        solution = int(meta["solution"])
        variable = str(meta["variable"])
        scaled_by = rng.choice([2, 3, 5, -2])
        scaled = Task(
            task.task_id + "-scaled",
            self.name,
            f"Equivalent equation after multiplying both sides by {scaled_by}: {a * scaled_by}{variable} + {b * scaled_by} = {c * scaled_by}. Return only {variable}.",
            solution,
            {**meta, "a": a * scaled_by, "b": b * scaled_by, "c": c * scaled_by, "metamorphic": "scale_equation"},
            group_id=task.group_id,
        )
        renamed = rng.choice([v for v in ["u", "v", "w"] if v != variable])
        renamed_task = Task(
            task.task_id + "-renamed",
            self.name,
            f"Variable rename robustness check. Solve for {renamed}: {a}{renamed} + {b} = {c}. Return only the integer.",
            solution,
            {**meta, "variable": renamed, "metamorphic": "rename_variable"},
            group_id=task.group_id,
        )
        return [scaled, renamed_task]

    def anti_metamorphic(self, task: Task, rng: random.Random) -> list[Task]:
        meta = dict(task.metadata)
        kind = str(meta.get("kind", "linear"))
        if kind == "linear_system_2x2":
            coefficients = tuple(tuple(int(value) for value in row) for row in meta["coefficients"])
            variables = tuple(str(variable) for variable in meta["variables"])
            base_solution = {str(key): int(value) for key, value in dict(meta["solution"]).items()}
            changed = {
                variables[0]: base_solution[variables[0]] + rng.choice([-3, -2, 2, 3]),
                variables[1]: base_solution[variables[1]] + rng.choice([-3, -2, 2, 3]),
            }
            rhs = (
                coefficients[0][0] * changed[variables[0]] + coefficients[0][1] * changed[variables[1]],
                coefficients[1][0] * changed[variables[0]] + coefficients[1][1] * changed[variables[1]],
            )
            expected = _format_assignment_solution(changed, variables)
            return [Task(
                task.task_id + "-changed-system-solution",
                self.name,
                f"Changed-answer variant. Solve for {variables[0]} and {variables[1]}: "
                f"{coefficients[0][0]}{variables[0]} + {coefficients[0][1]}{variables[1]} = {rhs[0]}; "
                f"{coefficients[1][0]}{variables[0]} + {coefficients[1][1]}{variables[1]} = {rhs[1]}. "
                f"Return only assignments as '{variables[0]}=..., {variables[1]}=...'.",
                expected,
                {**meta, "rhs": rhs, "solution": changed, "anti_metamorphic": "changed_system_rhs"},
            )]
        a, b = int(meta["a"]), int(meta["b"])
        variable = str(meta["variable"])
        new_solution = int(meta["solution"]) + rng.choice([-3, -2, 2, 3])
        new_c = a * new_solution + b
        return [Task(
            task.task_id + "-changed-solution",
            self.name,
            f"Changed-answer variant. Solve for {variable}: {a}{variable} + {b} = {new_c}. Return only the integer value.",
            new_solution,
            {**meta, "solution": new_solution, "c": new_c, "anti_metamorphic": "changed_rhs"},
        )]

    def verify(self, task: Task, answer: CandidateAnswer) -> VerificationCaseResult:
        meta = dict(task.metadata)
        kind = str(meta.get("kind", "linear"))
        if kind == "linear_system_2x2":
            variables = tuple(str(variable) for variable in meta.get("variables", ()))
            parsed = _parse_exact_assignment_solution(answer.text, variables)
            if parsed is None:
                return _result(task, answer, False, 0.0, "answer is not an exact ordered assignment set")
            try:
                coefficients = tuple(tuple(int(value) for value in row) for row in meta["coefficients"])
                rhs = tuple(int(value) for value in meta["rhs"])
                matrix = sp.Matrix(coefficients)
                if matrix.det() == 0:
                    return _result(task, answer, False, 0.0, "linear system is singular")
                solution_values = matrix.LUsolve(sp.Matrix(rhs))
                expected = {
                    variables[index]: sp.simplify(solution_values[index])
                    for index in range(len(variables))
                }
                if any(value.is_integer is not True for value in expected.values()):
                    return _result(task, answer, False, 0.0, "linear system does not have an all-integer solution")
            except Exception as exc:
                return _result(task, answer, False, 0.0, f"invalid linear system task: {exc!r}")
            passed = all(parsed[variable] == int(expected[variable]) for variable in variables)
            return _result(
                task,
                answer,
                passed,
                1.0 if passed else 0.0,
                "linear system solved symbolically" if passed else f"expected {_format_assignment_solution({k: int(v) for k, v in expected.items()}, variables)}, got {answer.text.strip()!r}",
            )
        if kind in {"quadratic", "symbolic", "symbolic_quadratic"}:
            parsed_roots = _exact_int_tuple(answer.text)
            if parsed_roots is None:
                return _result(task, answer, False, 0.0, "answer is not an exact integer root set")
            try:
                a = int(meta["a"])
                b = int(meta["b"])
                c = int(meta["c"])
                variable = str(meta.get("variable", "x"))
                if a == 0:
                    return _result(task, answer, False, 0.0, "quadratic coefficient cannot be zero")
                x = sp.Symbol(variable)
                solved_roots = tuple(sp.solve(sp.Eq(a * x**2 + b * x + c, 0), x))
                if any(root.is_integer is not True for root in solved_roots):
                    return _result(task, answer, False, 0.0, "quadratic equation does not have an all-integer solution set")
                roots = tuple(sorted(int(root) for root in solved_roots))
            except Exception as exc:
                return _result(task, answer, False, 0.0, f"invalid symbolic algebra task: {exc!r}")
            if len(roots) == 0:
                return _result(task, answer, False, 0.0, "quadratic equation has no exact integer roots")
            passed = parsed_roots == roots
            return _result(
                task,
                answer,
                passed,
                1.0 if passed else 0.0,
                "quadratic equation solved symbolically" if passed else f"expected roots {roots}, got {parsed_roots}",
            )
        parsed = _exact_int(answer.text)
        if parsed is None:
            return _result(task, answer, False, 0.0, "answer is not exactly one algebraic integer")
        passed = parsed == int(task.expected)
        return _result(task, answer, passed, 1.0 if passed else 0.0, "linear equation solved" if passed else f"expected {task.expected}, got {parsed}")


class EntityTrackingSkill(SkillSpec):
    name = "entity_tracking"
    people = ["Mira", "Noah", "Lina", "Sofia", "Eli", "Rami"]
    places = ["atelier", "bibliothèque", "laboratoire", "gare", "bureau", "archive"]
    items = ["badge", "dossier", "cle", "carnet", "prototype"]

    def generate(self, n: int, rng: random.Random) -> list[Task]:
        tasks: list[Task] = []
        for idx in range(n):
            if rng.choice(["location_chain", "transfer_chain"]) == "transfer_chain":
                starter, carrier, final_holder, distractor = rng.sample(self.people, 4)
                start_place, carrier_place, distractor_place = rng.sample(self.places, 3)
                item = rng.choice(self.items)
                story = (
                    f"{starter} takes the {item} in {start_place}. "
                    f"{distractor} mentions {distractor_place}, but never touches the {item}. "
                    f"{starter} gives the {item} to {carrier}. "
                    f"{carrier} moves to {carrier_place} with the {item}. "
                    f"Before the end, {carrier} gives the {item} to {final_holder}."
                )
                prompt = f"Read the story and answer only with the person holding the {item} at the end.\n\n{story}"
                anchors = (
                    Anchor("person", final_holder, f"entity-{idx}"),
                    Anchor("object", item, f"entity-{idx}"),
                )
                metadata = {
                    "kind": "transfer_chain",
                    "starter": starter,
                    "carrier": carrier,
                    "final_holder": final_holder,
                    "final": final_holder,
                    "distractor": distractor,
                    "distractor_place": distractor_place,
                    "item": item,
                    "start_place": start_place,
                    "carrier_place": carrier_place,
                    "ask_kind": "final_holder",
                }
                tasks.append(Task(f"entity-transfer-{idx}-{rng.randrange(10**9)}", self.name, prompt, final_holder, metadata, anchors, f"entity-transfer-group-{idx}-{item}"))
                continue
            person = rng.choice(self.people)
            other = rng.choice([p for p in self.people if p != person])
            start, middle, final, distractor_place = rng.sample(self.places, 4)
            story = (
                f"Au début, {person} est dans {start}. "
                f"{other} passe par {distractor_place}, mais ce n'est pas la personne cible. "
                f"Ensuite, {person} se déplace vers {middle}. "
                f"Enfin, {person} laisse son badge dans {final}."
            )
            prompt = f"Lis l'histoire et réponds seulement avec le dernier lieu de {person}.\n\n{story}"
            anchors = (Anchor("person", person, f"entity-{idx}"), Anchor("location", final, f"entity-{idx}"))
            metadata = {"kind": "location_chain", "person": person, "other": other, "start": start, "middle": middle, "final": final, "distractor": distractor_place, "ask_kind": "final_location"}
            tasks.append(Task(f"entity-{idx}-{rng.randrange(10**9)}", self.name, prompt, final, metadata, anchors, f"entity-group-{idx}-{person}"))
        return tasks

    def metamorphic(self, task: Task, rng: random.Random) -> list[Task]:
        note = rng.choice([
            "Une phrase de contexte ajoute que la météo est mauvaise.",
            "Le narrateur répète que seul le dernier lieu compte.",
            "Un détail inutile mentionne un ancien itinéraire annulé.",
        ])
        return [Task(
            task.task_id + "-noise",
            self.name,
            f"{note}\n{task.prompt}",
            task.expected,
            {**dict(task.metadata), "metamorphic": "irrelevant_context"},
            task.anchors,
            task.group_id,
        )]

    def anti_metamorphic(self, task: Task, rng: random.Random) -> list[Task]:
        meta = dict(task.metadata)
        if str(meta.get("kind")) == "transfer_chain":
            current = str(meta.get("final_holder", task.expected))
            new_holder = rng.choice([person for person in self.people if person != current])
            item = str(meta.get("item", "object"))
            prompt = f"{task.prompt}\nFinal correction: after all that, {new_holder} takes back the {item}. Answer with the final holder only."
            return [Task(
                task.task_id + "-changed-final-holder",
                self.name,
                prompt,
                new_holder,
                {**meta, "final_holder": new_holder, "final": new_holder, "anti_metamorphic": "changed_final_holder"},
                (Anchor("person", new_holder, task.task_id), Anchor("object", item, task.task_id)),
            )]
        new_final = rng.choice([place for place in self.places if place != meta["final"]])
        person = str(meta["person"])
        prompt = f"{task.prompt}\nCorrection finale: après tout cela, {person} retourne dans {new_final}. Réponds avec le dernier lieu réel."
        return [Task(
            task.task_id + "-changed-final",
            self.name,
            prompt,
            new_final,
            {**meta, "final": new_final, "anti_metamorphic": "changed_final_location"},
            (Anchor("person", person, task.task_id), Anchor("location", new_final, task.task_id)),
        )]

    def verify(self, task: Task, answer: CandidateAnswer) -> VerificationCaseResult:
        expected = str(task.expected).strip().lower()
        actual = answer.text.strip().lower()
        passed = actual == expected
        return _result(task, answer, passed, 1.0 if passed else 0.0, "entity state tracked" if passed else f"expected final location {task.expected!r}, got {answer.text!r}")


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "set": set,
    "sorted": sorted,
    "sum": sum,
    "tuple": tuple,
}
_BANNED_CODE_TOKENS = ("import ", "__", "open(", "eval(", "exec(", "compile(", "globals(", "locals(", "input(", "breakpoint(")


def _extract_python_code(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    return (match.group(1) if match else text).strip()


class CodeUnitTestSkill(SkillSpec):
    name = "code_unit_tests"

    def _templates(self) -> tuple[Mapping[str, Any], ...]:
        return (
            {
                "title": "add_one",
                "function_name": "solve",
                "prompt": "Write Python function solve(x) that returns x + 1.",
                "expected": "def solve(x):\n    return x + 1\n",
                "tests": [((1,), 2), ((-2,), -1), ((0,), 1)],
                "hidden_tests": [((41,), 42), ((-100,), -99)],
                "properties": ("deterministic", "no_argument_mutation"),
                "wrong_impl": "def solve(x):\n    return x\n",
            },
            {
                "title": "clamp",
                "function_name": "solve",
                "prompt": "Write Python function solve(value, low, high) that clamps value into the inclusive [low, high] range.",
                "expected": "def solve(value, low, high):\n    return max(low, min(value, high))\n",
                "tests": [((5, 1, 10), 5), ((-3, 0, 9), 0), ((12, 0, 9), 9)],
                "hidden_tests": [((7, 7, 7), 7), ((100, -4, 4), 4)],
                "properties": ("deterministic", "no_argument_mutation"),
                "wrong_impl": "def solve(value, low, high):\n    return value\n",
            },
            {
                "title": "count_vowels",
                "function_name": "solve",
                "prompt": "Write Python function solve(text) that counts lowercase vowels a, e, i, o, u in text.",
                "expected": "def solve(text):\n    return sum(1 for ch in text if ch in 'aeiou')\n",
                "tests": [(("cortex",), 2), (("rhythm",), 0), (("ledger",), 2)],
                "hidden_tests": [(("aeiou",), 5), (("bitnet",), 2)],
                "properties": ("deterministic", "no_argument_mutation"),
                "wrong_impl": "def solve(text):\n    return len(text)\n",
            },
            {
                "title": "first_even",
                "function_name": "solve",
                "prompt": "Write Python function solve(values) that returns the first even integer, or None if no even integer exists.",
                "expected": "def solve(values):\n    for value in values:\n        if value % 2 == 0:\n            return value\n    return None\n",
                "tests": [(([1, 3, 4, 6],), 4), (([7, 9],), None), (([2, 3],), 2)],
                "hidden_tests": [(([5, 8, 10],), 8), (([],), None)],
                "properties": ("deterministic", "no_argument_mutation"),
                "wrong_impl": "def solve(values):\n    return values[-1]\n",
            },
            {
                "title": "dedupe_preserve_order",
                "function_name": "solve",
                "prompt": "Write Python function solve(values) that returns a new list with duplicates removed while preserving first-seen order. Do not mutate values.",
                "expected": "def solve(values):\n    seen = set()\n    out = []\n    for value in values:\n        if value not in seen:\n            seen.add(value)\n            out.append(value)\n    return out\n",
                "tests": [(([3, 1, 3, 2, 1],), [3, 1, 2]), (([],), []), (([1, 1, 1],), [1])],
                "hidden_tests": [((["b", "a", "b", "c"],), ["b", "a", "c"]), (([0, -1, 0, -1, 2],), [0, -1, 2])],
                "properties": ("deterministic", "no_argument_mutation"),
                "wrong_impl": "def solve(values):\n    return sorted(set(values))\n",
            },
            {
                "title": "merge_counts",
                "function_name": "solve",
                "prompt": "Write Python function solve(pairs) that receives (name, count) pairs and returns a dict summing counts by name. Do not mutate pairs.",
                "expected": "def solve(pairs):\n    totals = {}\n    for name, count in pairs:\n        totals[name] = totals.get(name, 0) + count\n    return totals\n",
                "tests": [(([("a", 2), ("b", 1), ("a", 3)],), {"a": 5, "b": 1}), (([],), {}), (([("x", -1), ("x", 4)],), {"x": 3})],
                "hidden_tests": [(([("m", 0), ("n", 7), ("m", 5)],), {"m": 5, "n": 7}), (((("q", 1), ("q", 2)),), {"q": 3})],
                "properties": ("deterministic", "no_argument_mutation"),
                "wrong_impl": "def solve(pairs):\n    return dict(pairs)\n",
            },
        )

    def generate(self, n: int, rng: random.Random) -> list[Task]:
        templates = self._templates()
        tasks: list[Task] = []
        for idx in range(n):
            template = dict(rng.choice(templates))
            visible = "; ".join(f"{template['function_name']}{args} -> {expected!r}" for args, expected in template["tests"])
            prompt = f"{template['prompt']}\nReturn only code. Visible tests: {visible}."
            metadata = {
                **template,
                "tests": tuple(template["tests"]),
                "hidden_tests": tuple(template["hidden_tests"]),
                "properties": tuple(template.get("properties", ("deterministic", "no_argument_mutation"))),
                "require_hidden_tests": bool(template.get("hidden_tests")),
            }
            tasks.append(Task(f"code-{idx}-{template['title']}-{rng.randrange(10**9)}", self.name, prompt, template["expected"], metadata, group_id=f"code-group-{idx}-{template['title']}"))
        return tasks

    def metamorphic(self, task: Task, rng: random.Random) -> list[Task]:
        return [Task(
            task.task_id + "-strict",
            self.name,
            f"{task.prompt}\nDo not include explanations, markdown, imports, or I/O.",
            task.expected,
            {**dict(task.metadata), "metamorphic": "strict_code_only"},
            task.anchors,
            task.group_id,
        )]

    def anti_metamorphic(self, task: Task, rng: random.Random) -> list[Task]:
        meta = dict(task.metadata)
        if meta["title"] == "add_one":
            expected = "def solve(x):\n    return x - 1\n"
            prompt = "Changed requirement: write Python function solve(x) that returns x - 1. Return only code."
            tests = (((3,), 2), ((0,), -1))
        elif meta["title"] == "clamp":
            expected = "def solve(value, low, high):\n    return value\n"
            prompt = "Changed requirement: write Python function solve(value, low, high) that returns value unchanged. Return only code."
            tests = (((5, 1, 10), 5), ((-3, 0, 9), -3), ((12, 0, 9), 12))
        elif meta["title"] == "count_vowels":
            expected = "def solve(text):\n    return len(text)\n"
            prompt = "Changed requirement: write Python function solve(text) that returns the length of text. Return only code."
            tests = ((("cortex",), 6), (("",), 0))
        else:
            expected = "def solve(values):\n    return values[-1] if values else None\n"
            prompt = "Changed requirement: write Python function solve(values) that returns the last value, or None for an empty list. Return only code."
            tests = ((([1, 3, 4],), 4), (([],), None))
        return [Task(
            task.task_id + "-changed-code-contract",
            self.name,
            prompt,
            expected,
            {**meta, "tests": tests, "hidden_tests": tuple(), "require_hidden_tests": False, "anti_metamorphic": "changed_code_contract"},
        )]

    def verify(self, task: Task, answer: CandidateAnswer) -> VerificationCaseResult:
        source = _extract_python_code(answer.text)
        lowered = source.lower()
        if any(token in lowered for token in _BANNED_CODE_TOKENS):
            return _result(task, answer, False, 0.0, "unsafe or unsupported code token")
        namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
        try:
            exec(compile(source, "<candidate-solve>", "exec"), namespace, namespace)
            fn = namespace.get(str(task.metadata["function_name"]))
            if not callable(fn):
                return _result(task, answer, False, 0.0, f"function {task.metadata['function_name']!r} not defined")
            visible_tests = tuple(task.metadata.get("tests", ()))
            hidden_tests = tuple(task.metadata.get("hidden_tests", ()))
            if bool(task.metadata.get("require_hidden_tests", False)) and not hidden_tests:
                return _result(task, answer, False, 0.0, "missing required hidden code tests")
            tests = visible_tests + hidden_tests
            properties = set(str(item) for item in task.metadata.get("properties", ()))
            for args, expected in tests:
                call_args = deepcopy(args)
                frozen_before = repr(call_args)
                actual = fn(*call_args)
                if actual != expected:
                    return _result(task, answer, False, 0.0, f"unit test failed for args={args!r}: expected {expected!r}, got {actual!r}")
                if "no_argument_mutation" in properties and repr(call_args) != frozen_before:
                    return _result(task, answer, False, 0.0, f"property failed for args={args!r}: function mutated input arguments")
                if "deterministic" in properties:
                    second_actual = fn(*deepcopy(args))
                    if second_actual != actual:
                        return _result(task, answer, False, 0.0, f"property failed for args={args!r}: function is not deterministic")
        except Exception as exc:
            return _result(task, answer, False, 0.0, f"code execution failed: {exc!r}")
        return _result(task, answer, True, 1.0, "all visible, hidden and property unit tests passed")


class CalibrationSkill(SkillSpec):
    name = "calibration"

    def generate(self, n: int, rng: random.Random) -> list[Task]:
        tasks: list[Task] = []
        for idx in range(n):
            known = rng.choice([True, False])
            code = f"C3-{rng.randint(1000, 9999)}-{rng.choice(['A', 'B', 'Z'])}"
            if known:
                prompt = f"Contexte: le code de vérification du lot est {code}. Question: quel est le code ? Réponds seulement avec le code."
                expected = code
                metadata = {"known": True, "min_confidence": 0.70, "reference_confidence": 0.95, "code": code}
                anchors = (Anchor("identifier", code, f"cal-{idx}"),)
            else:
                prompt = "Contexte: le document mentionne un lot sans donner son code. Question: quel est le code ? Réponds UNKNOWN si le contexte ne le dit pas."
                expected = "UNKNOWN"
                metadata = {"known": False, "max_confidence": 0.45, "reference_confidence": 0.25}
                anchors = ()
            tasks.append(Task(f"calibration-{idx}-{rng.randrange(10**9)}", self.name, prompt, expected, metadata, anchors, f"calibration-group-{idx}-{known}"))
        return tasks

    def metamorphic(self, task: Task, rng: random.Random) -> list[Task]:
        return [Task(
            task.task_id + "-confidence-explicit",
            self.name,
            f"{task.prompt}\nInclue ta confiance interne dans le champ confidence de la réponse, pas dans le texte.",
            task.expected,
            {**dict(task.metadata), "metamorphic": "confidence_instruction"},
            task.anchors,
            task.group_id,
        )]

    def anti_metamorphic(self, task: Task, rng: random.Random) -> list[Task]:
        meta = dict(task.metadata)
        if meta.get("known"):
            return [Task(
                task.task_id + "-removed-evidence",
                self.name,
                "Contexte: le document mentionne un lot mais ne donne aucun code. Question: quel est le code ? Réponds UNKNOWN si le contexte ne le dit pas.",
                "UNKNOWN",
                {"known": False, "max_confidence": 0.45, "reference_confidence": 0.25, "anti_metamorphic": "remove_evidence"},
            )]
        code = f"C3-{rng.randint(1000, 9999)}-A"
        return [Task(
            task.task_id + "-added-evidence",
            self.name,
            f"Contexte: le code de vérification du lot est {code}. Question: quel est le code ? Réponds seulement avec le code.",
            code,
            {"known": True, "min_confidence": 0.70, "reference_confidence": 0.95, "code": code, "anti_metamorphic": "add_evidence"},
            (Anchor("identifier", code, task.task_id),),
        )]

    def verify(self, task: Task, answer: CandidateAnswer) -> VerificationCaseResult:
        expected = str(task.expected).strip()
        actual = answer.text.strip()
        text_ok = actual == expected
        confidence = max(0.0, min(1.0, answer.confidence))
        min_conf = task.metadata.get("min_confidence")
        max_conf = task.metadata.get("max_confidence")
        confidence_ok = True
        if min_conf is not None:
            confidence_ok = confidence >= float(min_conf)
        if max_conf is not None:
            confidence_ok = confidence <= float(max_conf)
        score = (0.5 if text_ok else 0.0) + (0.5 if confidence_ok else 0.0)
        passed = text_ok and confidence_ok
        if passed:
            reason = "answer and confidence calibrated"
        elif not text_ok:
            reason = f"expected calibrated answer {expected!r}, got {actual!r}"
        else:
            reason = f"confidence {confidence:.2f} outside calibrated bounds"
        return _result(task, answer, passed, score, reason)


def default_skill_specs() -> list[SkillSpec]:
    return [
        ArithmeticSkill(),
        AlgebraSkill(),
        LongContextAnchorSkill(),
        EntityTrackingSkill(),
        InstructionSkill(),
        CodeUnitTestSkill(),
        CalibrationSkill(),
    ]


Agent = Callable[[Task], CandidateAnswer | str]


class DynamicSkillVerifier:
    def __init__(self, specs: Iterable[SkillSpec], oracle_registry: OracleRegistry | None = None):
        self.specs = {spec.name: spec for spec in specs}
        if not self.specs:
            raise ValueError("DynamicSkillVerifier requires at least one SkillSpec")
        self.oracle_registry = oracle_registry or OracleRegistry()
        for spec in self.specs.values():
            self.oracle_registry.register(spec.name, spec.verify, replace=True)

    def build_suite(self, n_per_skill: int, seed: int, include_metamorphic: bool = True, include_anti_metamorphic: bool = False) -> list[Task]:
        rng = random.Random(seed)
        suite: list[Task] = []
        for spec in self.specs.values():
            suite.extend(spec.build_suite(n_per_skill, rng, include_metamorphic, include_anti_metamorphic))
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
            verifier_start = time.perf_counter()
            case = self.oracle_registry.verify(task.skill, task, answer)
            verifier_elapsed_ms = (time.perf_counter() - verifier_start) * 1000.0
            case = replace(case, verifier_cost=case.verifier_cost.merge(CostTrace(wall_time_ms=verifier_elapsed_ms)))
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
            reports[skill] = SkillReport(skill, skill_total, skill_passed, score, failures, tuple(cases))
            total += skill_total
            passed += skill_passed
            weighted += score * skill_total
        return VerificationSuiteReport(reports, total, passed, weighted / total if total else 0.0, total_cost)

    def evaluate(self, agent: Agent, n_per_skill: int = 8, seed: int = 0, include_metamorphic: bool = True) -> VerificationSuiteReport:
        return self.evaluate_tasks(agent, self.build_suite(n_per_skill, seed, include_metamorphic))

    def profile(self, agent: Agent, n_per_skill: int = 8, seed: int = 0, include_metamorphic: bool = True) -> tuple[VerificationSuiteReport, VerifierCostProfile]:
        report = self.evaluate(agent, n_per_skill=n_per_skill, seed=seed, include_metamorphic=include_metamorphic)
        return report, VerifierCostProfiler().summarize(report)

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
                tasks.extend((spec.anti_metamorphic(failure.task, rng) + spec.metamorphic(failure.task, rng))[:per_failure])
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
        if domain in {"math", "code", "exact_anchor", "calibration"} or risk > 0.75:
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
        if skill in {"long_context_anchor", "entity_tracking"} or "anchor" in reason:
            actions.append(RegrowthAction("force_exact_anchor", skill, 0.35, 2.0, "exact detail was lost"))
            actions.append(RegrowthAction("increase_kv_precision", skill, 0.25, 5.0, "latent memory may be over-compressed"))
        if skill in {"arithmetic", "algebra"} or "integer" in reason:
            actions.append(RegrowthAction("reduce_mtp_horizon", skill, 0.20, 1.0, "exact reasoning should not be overspeculated"))
            actions.append(RegrowthAction("activate_math_expert", skill, 0.35, 4.0, "route arithmetic to specialist"))
            actions.append(RegrowthAction("increase_activation_bits", skill, 0.15, 3.0, "critical numeric activation may need more precision"))
        if skill == "instruction_following" or "format" in reason:
            actions.append(RegrowthAction("add_format_certificate", skill, 0.20, 1.5, "format constraints need a cheap certificate"))
        if skill == "code_unit_tests" or "unit test" in reason:
            actions.append(RegrowthAction("add_unit_test_oracle", skill, 0.40, 2.5, "code behavior must be checked by executable tests"))
            actions.append(RegrowthAction("route_code_tool_verifier", skill, 0.30, 3.0, "code tasks need a tool-backed verification path"))
        if skill == "calibration" or "confidence" in reason:
            actions.append(RegrowthAction("add_uncertainty_gate", skill, 0.30, 2.0, "unknown answers need confidence bounds"))
            actions.append(RegrowthAction("increase_verification_level", skill, 0.20, 2.5, "calibration failure should trigger stronger checks"))
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
        confidence = float(task.metadata.get("reference_confidence", 1.0))
        expected = task.expected
        if isinstance(expected, Mapping) and "answer" in expected:
            text = str(expected["answer"])
        else:
            text = str(expected)
        return CandidateAnswer(text, confidence=confidence, cost=CostTrace(generated_tokens=max(1, len(text.split()))))


class CorruptedCompressedAgent:
    def __init__(self, arithmetic_bias: int = 1, anchor_corruption: bool = True, verbose_format: bool = True):
        self.arithmetic_bias = arithmetic_bias
        self.anchor_corruption = anchor_corruption
        self.verbose_format = verbose_format

    def __call__(self, task: Task) -> CandidateAnswer:
        if task.skill == "arithmetic":
            return CandidateAnswer(str(int(task.expected) + self.arithmetic_bias), confidence=0.82, cost=CostTrace(generated_tokens=1, weight_bits_read=32))
        if task.skill == "algebra":
            if str(task.metadata.get("kind", "")) == "linear_system_2x2":
                variables = tuple(str(variable) for variable in task.metadata.get("variables", ()))
                solution = {str(key): int(value) for key, value in dict(task.metadata.get("solution", {})).items()}
                if variables and solution:
                    corrupted = dict(solution)
                    corrupted[variables[0]] = int(corrupted[variables[0]]) - self.arithmetic_bias
                    text = _format_assignment_solution(corrupted, variables)
                    return CandidateAnswer(text, confidence=0.80, cost=CostTrace(generated_tokens=max(1, len(text.split())), activation_bits=4, weight_bits_read=32))
            return CandidateAnswer(str(int(task.expected) - self.arithmetic_bias), confidence=0.80, cost=CostTrace(generated_tokens=1, activation_bits=4, weight_bits_read=32))
        if task.skill == "long_context_anchor":
            text = str(task.expected)
            if self.anchor_corruption:
                text = text[:-1] + ("A" if text[-1] != "A" else "B") if text.startswith("C3-") else text.lower()
            return CandidateAnswer(text, confidence=0.76, cost=CostTrace(generated_tokens=1, kv_bytes=4, weight_bits_read=16))
        if task.skill == "entity_tracking":
            text = str(task.metadata.get("distractor", "unknown"))
            return CandidateAnswer(text, confidence=0.78, cost=CostTrace(generated_tokens=1, kv_bytes=4, weight_bits_read=16))
        if task.skill == "instruction_following":
            text = str(task.expected)
            if self.verbose_format:
                text += "\nDone."
            return CandidateAnswer(text, confidence=0.88, cost=CostTrace(generated_tokens=max(1, len(text.split())), weight_bits_read=16))
        if task.skill == "code_unit_tests":
            name = str(task.metadata.get("function_name", "solve"))
            text = str(task.metadata.get("wrong_impl", f"def {name}(*args):\n    return None\n"))
            return CandidateAnswer(text, confidence=0.84, certificate={"visible_tests": "not_run"}, cost=CostTrace(generated_tokens=max(1, len(text.split())), weight_bits_read=48))
        if task.skill == "calibration":
            if str(task.expected) == "UNKNOWN":
                return CandidateAnswer("C3-0000-Z", confidence=0.97, cost=CostTrace(generated_tokens=1, weight_bits_read=16))
            return CandidateAnswer(str(task.expected), confidence=0.32, cost=CostTrace(generated_tokens=1, weight_bits_read=16))
        return CandidateAnswer("", confidence=0.0)


class FaultType(str, Enum):
    NEGATION_DROPPED = "negation_dropped"
    NUMBER_ALTERED = "number_altered"
    VARIABLE_INVERTED = "variable_inverted"
    EXPERT_MISROUTED = "expert_misrouted"
    LATENT_KV_CORRUPTED = "latent_kv_corrupted"
    MTP_HORIZON_TOO_LONG = "mtp_horizon_too_long"
    ACTIVATION_OVERQUANTIZED = "activation_overquantized"
    CERTIFICATE_INCOMPLETE = "certificate_incomplete"
    OVERCONFIDENT_UNKNOWN = "overconfident_unknown"


class InjectedFaultAgent:
    def __init__(self, reference: Agent | None = None, fault: FaultType = FaultType.NUMBER_ALTERED):
        self.reference = reference or ReferenceRuleAgent()
        self.fault = fault

    def __call__(self, task: Task) -> CandidateAnswer:
        base = CandidateAnswer.coerce(self.reference(task))
        text = base.text
        confidence = base.confidence
        certificate = dict(base.certificate)
        cost = base.cost

        if self.fault in {FaultType.NUMBER_ALTERED, FaultType.ACTIVATION_OVERQUANTIZED} and task.skill in {"arithmetic", "algebra"}:
            parsed = _last_int(text)
            if parsed is not None:
                text = str(parsed + 1)
            confidence = max(confidence, 0.82)
            cost = cost.merge(CostTrace(activation_bits=4, weight_bits_read=16))
        elif self.fault == FaultType.VARIABLE_INVERTED and task.skill == "algebra":
            parsed = _last_int(text)
            if parsed is not None:
                text = str(-parsed if parsed != 0 else 1)
            confidence = max(confidence, 0.80)
        elif self.fault in {FaultType.LATENT_KV_CORRUPTED, FaultType.NEGATION_DROPPED} and task.skill in {"long_context_anchor", "entity_tracking"}:
            text = str(task.metadata.get("distractor", text.lower()))
            confidence = max(confidence, 0.76)
            cost = cost.merge(CostTrace(kv_bytes=2))
        elif self.fault in {FaultType.EXPERT_MISROUTED, FaultType.MTP_HORIZON_TOO_LONG} and task.skill in {"instruction_following", "code_unit_tests"}:
            if task.skill == "instruction_following":
                text = text + "\nDone."
            else:
                text = str(task.metadata.get("wrong_impl", text))
            confidence = max(confidence, 0.86)
            cost = cost.merge(CostTrace(experts_activated=1))
        elif self.fault == FaultType.CERTIFICATE_INCOMPLETE:
            certificate = {}
            if task.skill in {"code_unit_tests", "instruction_following"}:
                text = text + "\nDone."
        elif self.fault == FaultType.OVERCONFIDENT_UNKNOWN and task.skill == "calibration":
            if str(task.expected) == "UNKNOWN":
                text = "C3-0000-Z"
                confidence = 0.99
            else:
                confidence = 0.20

        return CandidateAnswer(text=text, confidence=confidence, certificate=certificate, cost=cost, raw={"fault": self.fault.value})


@dataclass(frozen=True)
class FaultDetectionResult:
    fault: FaultType
    detected: bool
    regressions: int
    total_cases: int
    candidate_score: float
    profile: VerifierCostProfile


class RegressionHarness:
    def __init__(self, verifier: DynamicSkillVerifier):
        self.verifier = verifier

    def run_fault(self, fault: FaultType, *, seed: int = 0, n_per_skill: int = 3, reference: Agent | None = None) -> FaultDetectionResult:
        ref_agent = reference or ReferenceRuleAgent()
        candidate = InjectedFaultAgent(ref_agent, fault)
        comparison = self.verifier.compare(ref_agent, candidate, n_per_skill=n_per_skill, seed=seed)
        candidate_report: VerificationSuiteReport = comparison["candidate"]  # type: ignore[assignment]
        regressions: tuple[VerificationCaseResult, ...] = tuple(comparison["regressions"])  # type: ignore[arg-type]
        profile = VerifierCostProfiler().summarize(candidate_report)
        return FaultDetectionResult(
            fault=fault,
            detected=bool(regressions),
            regressions=len(regressions),
            total_cases=candidate_report.total,
            candidate_score=candidate_report.aggregate_score,
            profile=profile,
        )

    def run_fault_matrix(self, *, seed: int = 0, n_per_skill: int = 3, faults: Iterable[FaultType] | None = None) -> tuple[FaultDetectionResult, ...]:
        selected = tuple(faults) if faults is not None else tuple(FaultType)
        return tuple(self.run_fault(fault, seed=seed, n_per_skill=n_per_skill) for fault in selected)


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
    specs = default_skill_specs()
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
