from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cortex3 import Anchor, CorruptedCompressedAgent, DynamicSkillVerifier, ReferenceRuleAgent, RegressionHarness, Task, VerificationCaseResult, default_skill_specs
from cortex3_autoregressive import ARDecoderAgent, ARTrainer, ar_examples_from_tasks
from cortex3_attribution import CausalAttributionEngine
from cortex3_certificates import CertificateAnswerVocabulary, CertificateHeadCalibrator, ProofCarryingGenerator, certificate_examples_from_tasks
from cortex3_cycle import CortexCycle, CycleReport, cycle_report_markdown
from cortex3_experiments import CortexExperimentSuite
from cortex3_frontier import FrontierSkillDiscovery
from cortex3_improvement import RecursiveImprovementEngine
from cortex3_inference import InferencePath, UltraFastInferenceEngine
from cortex3_memory import CognitiveMemory, CognitiveMemoryConfig
from cortex3_objective import build_objective_report
from cortex3_regrowth import MinimalRegrowthEngine
from cortex3_reporting import write_cycle_run
from cortex3_sleep import SleepPhaseConsolidator


def build_inference_smoke(verifier: DynamicSkillVerifier) -> tuple[object, ...]:
    anchor = Anchor("identifier", "C3-7777-Z", "cycle-smoke")
    memory = CognitiveMemory(CognitiveMemoryConfig(recent_exact_limit=1, embedding_dim=32, top_k_latent=2))
    memory.ingest(
        "legacy",
        "Ancien contexte: Sofia garde le prototype et le code exact C3-7777-Z.",
        extra_anchors=(anchor,),
    )
    memory.ingest("recent", "Message recent sans code.")
    engine = UltraFastInferenceEngine(verifier, ReferenceRuleAgent(), memory=memory)
    memory_recovery_engine = UltraFastInferenceEngine(verifier, CorruptedCompressedAgent(), memory=memory)
    memory_recovery_task = Task(
        "cycle-memory-recovery",
        "long_context_anchor",
        "Retrouve le code exact du prototype Sofia.",
        "C3-7777-Z",
        {"ask_kind": "code"},
        anchors=(anchor,),
    )
    return (
        engine.infer(Task("cycle-fast", "instruction_following", "Output OK exactly.", "OK")),
        engine.infer(Task("cycle-normal", "arithmetic", "Compute exactly: 20 + 22. Return only the integer.", 42)),
        engine.infer(Task(
            "cycle-careful",
            "long_context_anchor",
            " ".join(["audit trail"] * 45) + " exact identifier C3-7777-Z",
            "C3-7777-Z",
            anchors=(anchor,),
        )),
        memory_recovery_engine.infer(memory_recovery_task, forced_path=InferencePath.CAREFUL),
    )


def build_autoregressive_smoke(verifier: DynamicSkillVerifier, *, seed: int = 7, n_per_skill: int = 1, epochs: int = 60) -> dict[str, object]:
    tasks = verifier.build_suite(max(1, n_per_skill), seed=seed, include_metamorphic=False)
    examples = ar_examples_from_tasks(tasks)
    model, training = ARTrainer().train(examples, epochs=epochs, lr=0.03)
    agent = ARDecoderAgent(model, use_future_contracts=True)
    generated = [agent(task) for task in tasks]
    verified = verifier.evaluate_tasks(agent, tasks)
    inference = UltraFastInferenceEngine(verifier, agent).infer(tasks[0], forced_path=InferencePath.CAREFUL)
    return {
        "training": training.to_dict(),
        "dataset": {
            "examples": len(examples),
            "skills": sorted({example.task.skill for example in examples}),
            "sources": sorted({example.source for example in examples}),
        },
        "dsv": {
            "passed": verified.passed,
            "total": verified.total,
            "aggregate_score": verified.aggregate_score,
            "verified_capability_per_cost": verified.verified_capability_per_cost,
        },
        "generated_samples": [
            {
                "task_id": task.task_id,
                "skill": task.skill,
                "expected": str(task.expected),
                "generated": answer.text,
                "confidence": answer.confidence,
                "effective_cost": answer.cost.effective_cost(),
                "compiled_circuit": answer.certificate.get("compiled_circuit"),
                "compiled_weight_bits": answer.certificate.get("compiled_weight_bits"),
                "block_contracts": answer.certificate.get("block_contracts", {}),
            }
            for task, answer in zip(tasks[:5], generated[:5])
        ],
        "inference": inference.to_dict(),
    }


def build_certificate_smoke(verifier: DynamicSkillVerifier, *, seed: int = 7, n_per_skill: int = 1, epochs: int = 120) -> tuple[object, ...]:
    tasks = verifier.build_suite(max(1, n_per_skill), seed=seed, include_metamorphic=False)
    examples = certificate_examples_from_tasks(tasks)
    vocabulary = CertificateAnswerVocabulary.from_answers(example.answer for example in examples)
    calibrator = CertificateHeadCalibrator(hidden_size=64, latent_size=16, vocabulary=vocabulary)
    head, training = calibrator.train(examples, epochs=epochs, lr=0.04)
    agent = ProofCarryingGenerator(head, vocabulary)
    generated = [agent(task) for task in tasks]
    verified = verifier.evaluate_tasks(agent, tasks)
    if verified.passed != verified.total:
        raise RuntimeError(f"proof-carrying certificate smoke failed DSV: {verified.passed}/{verified.total}")
    if not all(answer.raw["certificate_verification"]["passed"] for answer in generated):
        raise RuntimeError("proof-carrying certificate smoke produced an unverifiable certificate")
    certs = []
    for answer in generated[:5]:
        payload = dict(answer.raw["proof_carrying_answer"]["certificate"])
        payload["calibration"] = training.to_dict()
        certs.append(payload)
    return tuple(certs)


def build_causal_attribution_smoke(verifier: DynamicSkillVerifier) -> object:
    task = Task("cycle-causal-forward", "arithmetic", "Compute exactly: 20 + 22. Return only the integer.", 42, {"kind": "add", "a": 20, "b": 22})
    engine = UltraFastInferenceEngine(verifier, CorruptedCompressedAgent())
    result = engine.infer(task, forced_path=InferencePath.CAREFUL)
    if result.verification is None or result.verification.passed:
        raise RuntimeError("causal attribution smoke requires a verifier-visible regression")
    return CausalAttributionEngine(verifier).batch_attribute((result.verification,), compression_ledger=engine.trace)


def _protected_tasks_from_report(report: CycleReport) -> tuple[Task, ...]:
    protected: list[Task] = []
    for skill_report in report.reference.skill_reports.values():
        protected.extend(case.task for case in skill_report.cases)
    return tuple(protected)


def _baseline_agent_for_failure(failure: VerificationCaseResult):
    reference = ReferenceRuleAgent()

    def agent(task: Task):
        if task.task_id == failure.task.task_id:
            return failure.answer
        return reference(task)

    return agent


def build_regrowth_smoke(verifier: DynamicSkillVerifier, report: CycleReport, *, budget: float = 36.0) -> tuple[object, ...]:
    attribution_engine = CausalAttributionEngine(verifier)
    regrowth_engine = MinimalRegrowthEngine(verifier)
    protected_tasks = _protected_tasks_from_report(report)
    plans = []
    for failure in report.regressions:
        attribution = attribution_engine.attribute(failure)
        plan = regrowth_engine.plan(
            attribution,
            _baseline_agent_for_failure(failure),
            protected_tasks=protected_tasks,
            budget=budget,
        )
        plans.append(plan)
    if report.regressions and not any(plan.selected is not None for plan in plans):
        raise RuntimeError("regrowth smoke did not select any recovering non-regressing repair")
    return tuple(plans)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a Cortex-3 cycle and persist research artifacts.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-per-skill", type=int, default=3)
    parser.add_argument("--repair-budget", type=float, default=8.0)
    parser.add_argument("--regrowth-budget", type=float, default=36.0)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-write", action="store_true", help="print the report without writing runs/ artifacts")
    parser.add_argument("--skip-inference", action="store_true", help="do not persist Phase 8 inference smoke traces")
    parser.add_argument("--skip-sleep", action="store_true", help="do not persist Phase 9 sleep anti-collapse traces")
    parser.add_argument("--skip-improvement", action="store_true", help="do not persist Phase 10 recursive improvement traces")
    parser.add_argument("--skip-experiments", action="store_true", help="do not persist the named plan experiments A-E")
    parser.add_argument("--skip-frontier", action="store_true", help="do not persist Frontier Skill Discovery compile traces")
    parser.add_argument("--skip-attribution", action="store_true", help="do not persist the Phase 6 causal attribution smoke")
    parser.add_argument("--skip-regrowth", action="store_true", help="do not persist the Phase 7 minimal regrowth plans")
    parser.add_argument("--skip-certificates", action="store_true", help="do not persist the trained proof-carrying certificate smoke")
    parser.add_argument("--skip-autoregressive", action="store_true", help="do not persist the trained autoregressive checkpoint smoke")
    parser.add_argument("--autoregressive-epochs", type=int, default=60, help="training epochs for the autoregressive checkpoint smoke")
    parser.add_argument("--autoregressive-n-per-skill", type=int, default=1, help="examples per skill for the autoregressive checkpoint smoke")
    parser.add_argument("--frontier-epochs", type=int, default=120, help="training epochs for Frontier Skill Discovery micro-compilation")
    parser.add_argument("--frontier-max-skills", type=int, default=2, help="maximum fragile skills to compile in Frontier Skill Discovery")
    args = parser.parse_args(argv)

    verifier = DynamicSkillVerifier(default_skill_specs())
    report = CortexCycle(verifier).run(
        ReferenceRuleAgent(),
        CorruptedCompressedAgent(),
        seed=args.seed,
        n_per_skill=args.n_per_skill,
        repair_budget=args.repair_budget,
    )
    print(cycle_report_markdown(report))
    if not args.no_write:
        fault_results = RegressionHarness(verifier).run_fault_matrix(seed=args.seed, n_per_skill=args.n_per_skill)
        inference_results = None if args.skip_inference else build_inference_smoke(verifier)
        sleep_report = None if args.skip_sleep else SleepPhaseConsolidator(verifier).ingest_cycle(report, seed=args.seed)
        improvement_report = None if args.skip_improvement else RecursiveImprovementEngine(verifier).run(report, max_proposals=4, seed=args.seed, n_per_skill=1)
        frontier_report = None if args.skip_frontier else FrontierSkillDiscovery(verifier).discover(
            report,
            seed=args.seed,
            max_skills=args.frontier_max_skills,
            epochs=args.frontier_epochs,
        )
        attribution = None if args.skip_attribution else build_causal_attribution_smoke(verifier)
        regrowth_plans = None if args.skip_regrowth else build_regrowth_smoke(verifier, report, budget=args.regrowth_budget)
        objective_report = build_objective_report(
            report,
            inference_results=inference_results,
            fault_results=fault_results,
            improvement_report=improvement_report,
        )
        experiments = None if args.skip_experiments else CortexExperimentSuite(verifier).run_all(seed=args.seed, n_per_skill=args.n_per_skill)
        certificates = None if args.skip_certificates else build_certificate_smoke(verifier, seed=args.seed, n_per_skill=1)
        autoregressive_report = None if args.skip_autoregressive else build_autoregressive_smoke(
            verifier,
            seed=args.seed,
            n_per_skill=args.autoregressive_n_per_skill,
            epochs=args.autoregressive_epochs,
        )
        artifacts = write_cycle_run(report, output_dir=args.out_dir, run_id=args.run_id, fault_results=fault_results, certificates=certificates, attribution=attribution, regrowth_plans=regrowth_plans, inference_results=inference_results, sleep_report=sleep_report, improvement_report=improvement_report, objective_report=objective_report, experiments=experiments, autoregressive_report=autoregressive_report, frontier_report=frontier_report)
        print(f"\nArtifacts written to: {artifacts.output_dir}")


if __name__ == "__main__":
    main()
