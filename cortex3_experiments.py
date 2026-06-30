from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from cortex3 import (
    CandidateAnswer,
    CompressionAdversary,
    CorruptedCompressedAgent,
    DynamicSkillVerifier,
    ReferenceRuleAgent,
    RegressionHarness,
    Task,
    default_skill_specs,
)
from cortex3_attribution import CausalAttributionEngine
from cortex3_cycle import CortexCycle
from cortex3_improvement import RecursiveImprovementEngine
from cortex3_inference import InferencePath, UltraFastInferenceEngine
from cortex3_regrowth import MinimalRegrowthEngine


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    title: str
    passed: bool
    metrics: Mapping[str, Any]
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "passed": self.passed,
            "metrics": dict(self.metrics),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ExperimentSuiteReport:
    results: tuple[ExperimentResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "results": [result.to_dict() for result in self.results],
        }


class CortexExperimentSuite:
    def __init__(self, verifier: DynamicSkillVerifier | None = None):
        self.verifier = verifier or DynamicSkillVerifier(default_skill_specs())
        self.reference = ReferenceRuleAgent()
        self.corrupted = CorruptedCompressedAgent()

    def experiment_a_verifier_faults(self, *, seed: int = 0, n_per_skill: int = 1) -> ExperimentResult:
        results = RegressionHarness(self.verifier).run_fault_matrix(seed=seed, n_per_skill=n_per_skill)
        detected = sum(1 for result in results if result.detected)
        detection_rate = detected / len(results) if results else 0.0
        return ExperimentResult(
            "A",
            "Verifier against injected faults",
            detection_rate == 1.0,
            {
                "faults": len(results),
                "detected": detected,
                "detection_rate": detection_rate,
                "undetected": [result.fault.value for result in results if not result.detected],
            },
            ("RegressionHarness.run_fault_matrix", "all injected fault families detected"),
        )

    def experiment_b_compression_adversary(self, *, seed: int = 0, n_per_skill: int = 1) -> ExperimentResult:
        fixed_tasks = self.verifier.build_suite(n_per_skill, seed, include_metamorphic=False)
        metamorphic_tasks = self.verifier.build_suite(n_per_skill, seed, include_metamorphic=True)
        fixed = self.verifier.evaluate_tasks(self.corrupted, fixed_tasks)
        metamorphic = self.verifier.evaluate_tasks(self.corrupted, metamorphic_tasks)
        fixed_failures = tuple(failure for report in fixed.skill_reports.values() for failure in report.failures)
        adversary_tasks = CompressionAdversary(self.verifier.specs.values()).expand_from_failures(fixed_failures, seed=seed + 17, per_failure=2)
        adversarial = self.verifier.evaluate_tasks(self.corrupted, adversary_tasks) if adversary_tasks else None
        fixed_count = sum(len(report.failures) for report in fixed.skill_reports.values())
        metamorphic_count = sum(len(report.failures) for report in metamorphic.skill_reports.values())
        adversarial_count = sum(len(report.failures) for report in adversarial.skill_reports.values()) if adversarial else 0
        return ExperimentResult(
            "B",
            "Compression adversary finds rare regressions beyond fixed benchmark",
            metamorphic_count + adversarial_count > fixed_count,
            {
                "fixed_failures": fixed_count,
                "metamorphic_failures": metamorphic_count,
                "adversarial_failures": adversarial_count,
                "adversary_tasks": len(adversary_tasks),
                "gain_over_fixed": (metamorphic_count + adversarial_count) - fixed_count,
            },
            ("fixed suite vs metamorphic suite vs CompressionAdversary-expanded failures",),
        )

    def experiment_c_minimal_regrowth(self, *, seed: int = 0, n_per_skill: int = 1) -> ExperimentResult:
        cycle = CortexCycle(self.verifier).run(self.reference, self.corrupted, seed=seed, n_per_skill=n_per_skill)
        failure = cycle.regressions[0]
        attribution = CausalAttributionEngine(self.verifier).attribute(failure)
        protected_tasks = self.verifier.build_suite(1, seed + 29, include_metamorphic=True)
        plan = MinimalRegrowthEngine(self.verifier).plan(attribution, self.corrupted, protected_tasks, budget=20.0)
        selected = plan.selected
        recovered = bool(selected and selected.recovered and selected.non_regression.passed)
        targeted_cheaper = bool(selected and selected.total_cost < attribution.global_retrain_cost)
        return ExperimentResult(
            "C",
            "Minimal regrowth beats global retraining cost",
            recovered and targeted_cheaper,
            {
                "failure_task_id": failure.task.task_id,
                "top_cause": attribution.top_cause,
                "selected_action": plan.selected_action,
                "recovered": recovered,
                "non_regression": bool(selected and selected.non_regression.passed),
                "targeted_cost": selected.total_cost if selected else None,
                "global_retrain_cost": attribution.global_retrain_cost,
                "gain_per_cost": selected.gain_per_cost if selected else 0.0,
            },
            ("CausalAttributionEngine.attribute", "MinimalRegrowthEngine.plan", "NonRegressionGate.check"),
        )

    def experiment_d_slow_compile_fast(self, *, seed: int = 0) -> ExperimentResult:
        tasks = self.verifier.build_suite(1, seed, include_metamorphic=False)
        task = next(task for task in tasks if task.skill == "instruction_following")
        engine = UltraFastInferenceEngine(self.verifier, self.reference)
        slow = engine.infer(task, forced_path=InferencePath.CAREFUL)
        fast = engine.infer(task, forced_path=InferencePath.FAST)
        slow_verified = self.verifier.oracle_registry.verify(task.skill, task, slow.answer).passed
        fast_verified = self.verifier.oracle_registry.verify(task.skill, task, fast.answer).passed
        slow_cost = slow.cost.effective_cost()
        fast_cost = fast.cost.effective_cost()
        cost_reduction = (slow_cost - fast_cost) / max(slow_cost, 1e-9)
        return ExperimentResult(
            "D",
            "SlowSolve to Compile to FastSolve preserves quality with lower cost",
            slow_verified and fast_verified and fast_cost < slow_cost,
            {
                "task_id": task.task_id,
                "slow_path": slow.route.path.value,
                "fast_path": fast.route.path.value,
                "slow_verified": slow_verified,
                "fast_verified": fast_verified,
                "slow_cost": slow_cost,
                "fast_cost": fast_cost,
                "cost_reduction": cost_reduction,
                "slow_layers": slow.layers_ran,
                "fast_layers": fast.layers_ran,
            },
            ("careful path external verification", "fast path external verification", "effective cost comparison"),
        )

    def experiment_e_auto_improvement_sandbox(self, *, seed: int = 0, n_per_skill: int = 1) -> ExperimentResult:
        cycle = CortexCycle(self.verifier).run(self.reference, self.corrupted, seed=seed, n_per_skill=n_per_skill)
        report = RecursiveImprovementEngine(self.verifier).run(cycle, max_proposals=4, seed=seed, n_per_skill=1)
        accepted = list(report.archive.accepted)
        reward_flags = [
            flag
            for decision in report.decisions
            for flag in decision.evaluation.reward_hacking_flags
        ]
        protected_losses = [
            dict(decision.evaluation.protected_losses)
            for decision in report.decisions
            if decision.evaluation.protected_losses
        ]
        collapse_flags = [
            flag
            for decision in report.decisions
            for flag in decision.evaluation.collapse_flags
        ]
        calibration_regressions = [
            decision.evaluation.proposal.proposal_id
            for decision in report.decisions
            if decision.evaluation.calibration_delta > 0.0
        ]
        return ExperimentResult(
            "E",
            "Auto-improvement sandbox accepts only controlled improvements",
            bool(accepted) and not reward_flags and not protected_losses and not collapse_flags and not calibration_regressions,
            {
                "proposals": len(report.proposals),
                "accepted": len(accepted),
                "rejected": len(report.archive.rejected),
                "reward_hacking_flags": reward_flags,
                "protected_losses": protected_losses,
                "collapse_flags": collapse_flags,
                "calibration_regressions": calibration_regressions,
                "accepted_ids": [record.proposal.proposal_id for record in accepted],
            },
            ("RecursiveImprovementEngine.run", "PatchAcceptanceGate", "EvolutionaryArchive"),
        )

    def run_all(self, *, seed: int = 0, n_per_skill: int = 1) -> ExperimentSuiteReport:
        return ExperimentSuiteReport((
            self.experiment_a_verifier_faults(seed=seed, n_per_skill=n_per_skill),
            self.experiment_b_compression_adversary(seed=seed, n_per_skill=n_per_skill),
            self.experiment_c_minimal_regrowth(seed=seed, n_per_skill=n_per_skill),
            self.experiment_d_slow_compile_fast(seed=seed),
            self.experiment_e_auto_improvement_sandbox(seed=seed, n_per_skill=n_per_skill),
        ))
