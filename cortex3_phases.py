from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    id: str
    title: str
    objective: str
    deliverables: tuple[str, ...]
    exit_criteria: tuple[str, ...]


CORTEX3_PHASES: tuple[Phase, ...] = (
    Phase("P1", "Dynamic Skill Verifier", "Detect rare capability regressions before compression hides them.", ("SkillSpec DSL", "metamorphic generators", "oracle registry", "regression harness"), ("Injected faults are caught", "reference/candidate regressions are separated", "verifier cost is measured")),
    Phase("P2", "Instrumented Ternary Core", "Represent weights as sign+mask blocks with provisional, reversible and certified zeros.", ("TernaryBlock", "zero-state ledger", "residual synapse buffer", "bit ledger"), ("Every compression choice is countable", "scale cost is included", "dead zeros are not treated as proof")),
    Phase("P3", "MTP/FSP Under Contract", "Generate faster only when future predictions are stable and risk is low.", ("adaptive horizon policy", "future contract", "contract revision", "temporal consistency checks"), ("MTP increases verified answers per cost", "high-risk domains fall back to short horizons")),
    Phase("P4", "Cognitive Memory", "Compress context without losing exact details.", ("latent memory plan", "exact anchor ledger", "query-conditioned recall", "anchor fidelity tests"), ("numbers/names/ids survive compression", "irrelevant context is safely summarized")),
    Phase("P5", "Latent Reasoning With Certificates", "Replace long visible reasoning with short proof-carrying certificates.", ("certificate schema", "certificate verifier", "random de-latentization hook"), ("reasoning tokens drop", "auditability remains", "calibration does not collapse")),
    Phase("P6", "Causal Regression Attribution", "Localize failures to weights, activations, KV, routing, MTP, contract, or uncertainty.", ("regression traces", "counterfactual probes", "attribution report"), ("regrowth can target the culprit", "global retraining is not the first response")),
    Phase("P7", "Minimal Regrowth", "Buy back the smallest structure that recovers a verified skill.", ("regrowth action space", "gain/cost ranking", "budgeted repair planner"), ("repaired skill improves", "total cognitive cost remains bounded")),
    Phase("P8", "Fast/Normal/Careful Inference", "Route each task to the cheapest safe cognitive path.", ("difficulty router", "budget predictor", "early-exit policy", "Mixture-of-Depths", "self-speculative MTP", "ternary kernel dispatch"), ("easy tasks get faster", "hard tasks remain accurate", "verification is hierarchical")),
    Phase("P9", "Sleep Phase Anti-Collapse", "Consolidate verified synthetic and real examples without self-poisoning.", ("failure replay buffer", "verified synthetic pool", "real/exogenous reservoir", "anti-collapse filter", "consolidation scheduler"), ("synthetic data is labeled", "diversity is preserved", "frontier skills improve")),
    Phase("P10", "Recursive Improvement Gate", "Evaluate proposed project improvements only through sandboxed, reversible, verified checks.", ("proposal generator", "sandbox trainer", "dynamic evaluator", "evolutionary archive", "Pareto acceptance gate", "rollback archive", "diversity preservation"), ("accepted patches improve quality/cost", "protected skills never regress", "reward hacking is rejected")),
)


def phase_table() -> str:
    lines = ["| Phase | Title | Objective |", "|---|---|---|"]
    for phase in CORTEX3_PHASES:
        lines.append(f"| {phase.id} | {phase.title} | {phase.objective} |")
    return "\n".join(lines)
