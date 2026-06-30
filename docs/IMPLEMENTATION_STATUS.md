# Cortex-3 implementation status

`Cortex-3 PLAN.txt` is the source of truth. This file records current executable coverage against that plan.

## Phase 1 - Verifier OS before the full model

Current executable coverage:

- SkillSpec DSL hooks: `SkillSpec.generate`, `metamorphic`, `anti_metamorphic`, `verify`.
- Metamorphic generator: implemented per skill family.
- Anti-metamorphic generator: explicit changed-answer variants for algebra, entity tracking, code contracts and calibration; adversarial-compatible hook for older skills.
- Oracle registry: `OracleRegistry` registers skill oracles and routes every verification through the registry.
- Anchor ledger: `ExactAnchorLedger` extracts identifiers, paths and numbers; anchor tasks carry exact anchors.
- Regression harness: `RegressionHarness` compares a reference agent with injected-fault candidates.
- Compression adversary: expands failures through anti-metamorphic and metamorphic variants.
- Verifier cost profiler: `VerifierCostProfiler` now summarizes true per-case verifier costs from persisted `VerificationCaseResult.verifier_cost` values, including verifier steps, wall time and max case cost.
- Persisted run artifacts: `cortex3_reporting.write_cycle_run` writes versioned `summary.json`, `report.md` and optional `fault_matrix.json`; per-skill reports include all cases, not only failures.
- Oracle quality auditor: `OracleQualityAuditor` probes every default skill for false positives and false negatives using correct reference answers and deliberately wrong answers.
- Strict exact-output oracles: arithmetic, algebra, long-context anchors, entity tracking and calibration reject embedded or extra-text answers when the task contract says “return only”.
- First domains: arithmetic, algebra, executable code unit tests, entity tracking, long context exact anchors, instruction following and calibration.
- Injected defects: number alteration, variable inversion, latent KV corruption, MTP horizon overshoot, activation overquantization, expert misrouting, incomplete certificate and overconfident unknown.

Evidence:

- `python -m unittest discover -s tests`
- `python -m py_compile cortex3.py cortex3_analysis.py cortex3_cycle.py cortex3_ledgers.py cortex3_phases.py cortex3_selection.py cortex3_reporting.py cortex3_ternary.py tools\run_cycle_report.py`
- `python tools\run_cycle_report.py`
- `tests/test_reporting_and_ternary.py::ReportingAndTernaryTest.test_cycle_run_artifacts_are_persisted`
- `.\.venv\Scripts\python.exe -m pip check`
- `.\.venv\Scripts\python.exe -m unittest discover -s tests`
- `.\.venv\Scripts\python.exe -m py_compile cortex3.py cortex3_analysis.py cortex3_cycle.py cortex3_ledgers.py cortex3_phases.py cortex3_selection.py cortex3_reporting.py cortex3_ternary.py cortex3_future.py cortex3_memory.py cortex3_certificates.py cortex3_attribution.py cortex3_regrowth.py tools\run_cycle_report.py`
- Direct Torch validation in `.venv`: CUDA PyTorch `torch==2.11.0+cu128`, `numpy==2.5.0`, `BitLinear(...)(torch.ones(1, 3)) -> shape (1, 2)` with compression and activation logs recorded.

Remaining Phase 1 hardening:

- Broaden generated grammars for code, algebra, entity tracking and calibration.
- Add a published JSON schema document and compatibility tests for downstream phase gates.

## Phase 2 - Instrumented Ternary Core

Current executable coverage:

- `TernaryBlock`, `ternarize_values`, zero states and estimated bit accounting.
- `cortex3_ternary.BitLinear` provides a PyTorch sign+mask layer with shared scales, activation quantization and residual weights; PyTorch is a required dependency.
- `ResidualSynapseBuffer` stores reconstruction residuals for compressed blocks.
- `CompressionTraceLedger` records compression decisions, activation quantization, expert activations, KV mode events and MTP/FSP events.
- Compression decisions include active count, provisional/certified zeros, estimated bits, threshold and residual L1.
- `LayerForwardEvent` records real `BitLinear` forward passes with layer id, input/output shapes, active weights, estimated packed weight bits and activation bits.
- `BitLinear` uses a straight-through runtime weight path so gradients reach `float_weight` while the forward value remains the compiled ternary/residual runtime weight.
- Tests cover exact parity with `nn.Linear` when activation quantization is disabled, gradient survival to inputs and weights, and micro-model layer-forward traces.

Remaining:

- Feed layer-forward traces into persisted cycle reports outside inference-specific trace summaries.
- Use layer-forward traces as first-class evidence in causal attribution block probes.

## Phase 3 - MTP/FSP under contract

Current executable coverage:

- `AdaptiveHorizonPolicy` and temporal consistency penalty.
- `cortex3_future.MTPFSPHeads` implements PyTorch heads for horizons 1, 2, 4 and 8.
- `confidence_head` produces sigmoid confidence used by contract gates.
- `temporal_consistency_loss` compares shifted future distributions across steps.
- `FutureContractEngine` drafts contracts, applies risk/domain horizon policy, revises contracts on mismatch or temporal drift, and gates block accept/reject.
- `FutureContractLedger` records accepted/rejected decisions and effective cost.
- Future contract ledgers can be persisted into `summary.json` through `write_cycle_run`.
- `verified_answers_per_effective_cost` measures verified answers per effective cost rather than token speed.
- `MTPFSPCalibrator` trains standalone MTP/FSP heads on verifier micro-task future-token targets and reports before/after token accuracy and confidence loss.
- `cortex3_autoregressive.ARLossComputer` adds trainable behavior, MTP multi-horizon, confidence and future-contract margin losses to the generated-answer checkpoint loop.
- `ARDecoderAgent(use_future_contracts=True)` now routes generation through `FutureContractEngine`, accepts or rejects speculative token blocks, records per-block traces, preserves DSV pass rate and accounts for real decoder steps spent on speculation.
- Inference result JSON now persists answer `cost` and `raw`, so autoregressive future-contract generation traces survive into run artifacts.

Remaining:

- Calibrate the standalone `MTPFSPHeads` module on larger held-out micro-task distributions, not only the seed smoke suite.
- Compare MTP accepted blocks against verified quality per effective cost across checkpoint variants in cycle reports.
- Add FSP output-goal contracts beyond token IDs.

## Phase 4 - Cognitive memory

Current executable coverage:

- Exact anchor extraction and fidelity scoring.
- Long-context and entity-tracking skill families.
- `CognitiveMemory` stores recent segments in exact KV and evicts older segments into compact latent KV.
- `LatentKVStore` keeps Torch embeddings, compact summaries and exact anchors instead of old full text.
- `Query-conditioned memory reconstruction` retrieves exact recent and latent old segments by embedding similarity, token overlap and anchor intent.
- `AnchorFidelityVerifier` proves required exact anchors are preserved in reconstructed context.
- `compression_report` records recent/latent counts, latent token compression ratio and ledger anchors.
- `write_cycle_run` can persist cognitive memory reports into `summary.json`.
- `UltraFastInferenceEngine` can convert a faithful reconstruction into a memory-augmented generated answer for long-context anchors and entity locations, replacing a weaker base answer without reading `expected`.
- Memory-augmented answers carry certificate fields, selected segment ids, anchor fidelity and the displaced base answer in `raw`; inference JSON persists this audit trail.

Remaining:

- Add learned query-conditioned compression instead of deterministic hashed embeddings.
- Promote anchor fidelity to a required cycle gate for long-context tasks.
- Measure memory cost/quality tradeoffs across exact KV vs latent KV in run reports.

## Phase 5 - Latent reasoning with certificates

Current executable coverage:

- `CandidateAnswer.certificate` field and certificate bit accounting.
- `LatentProofState` stores a Torch latent proof vector, latent step count and checksum.
- `LatentProofState.to_dict/from_dict` serializes proof vectors into JSON-safe rounded values for persisted audit trails.
- `CertificateHead` maps hidden states to latent proof state, answer logits, certificate type logits and uncertainty.
- `CertificateHeadCalibrator` trains the certificate head on verifier micro-task answers, certificate types and uncertainty targets.
- `ShortCertificate` carries answer, claims, uncertainty, latent checksum, anchors and optional tool contract.
- `CertificateVerifier` checks uncertainty bounds, latent checksum and tool-backed verification.
- `CertificateVerifier` accepts explicitly calibrated high-uncertainty certificates, so `UNKNOWN` can stay low-confidence without being treated as proof corruption.
- `RandomDelatentizer` samples latent dimensions deterministically for audit probes and detects tampering.
- Tool-backed checks include arithmetic, exact match, anchor fidelity and executable code unit tests.
- `ProofCarryingAnswer` converts answer + certificate + uncertainty into `CandidateAnswer` with a serializable latent proof payload.
- `ProofCarryingGenerator` connects a calibrated certificate head to DSV-compatible answer generation and verifies every emitted certificate.
- `evaluate_certificate_efficiency` measures token reduction, quality preservation and calibration preservation.
- `write_cycle_run` can persist short certificates into `summary.json`.
- `UltraFastInferenceEngine` treats proof-carrying certificate verification as a gate; a tampered latent proof makes `InferenceResult.passed` false and zeroes verified capability per cost.
- `tools/run_cycle_report.py` writes a trained proof-carrying certificate smoke by default; `--skip-certificates` disables it.

Remaining:

- Expand tool verification to multi-step algebra, richer code tests and external solver hooks.
- Benchmark certificate-token savings over held-out reasoning traces.

## Phase 6 - Causal regression attribution

Current executable coverage:

- `CausalLedger`, `CausalTrace` and `RegressionAnalyzer` cause hints.
- `CausalAttributionEngine` builds and runs counterfactual ablation probes.
- Ablation dimensions cover blocks, experts, KV mode, MTP horizon, activation precision, FSP contract and routing.
- Block probes consume `CompressionTraceLedger` compression decisions and test restore-block interventions.
- Layer-forward probes consume real `CompressionTraceLedger.layer_forward_events` emitted by `BitLinear` during inference and create layer-specific block-restoration and activation-precision interventions.
- FSP probes consume `FutureContractLedger` decisions and test stricter contract re-verification.
- Probe results record baseline score, counterfactual score, recovery, score delta, cost and gain per cost.
- Cause estimates normalize measured probe evidence into probabilities and best interventions.
- `cluster_regressions` groups failures by top cause and skill with recommended intervention.
- `write_cycle_run` can persist causal attribution reports and clusters into `summary.json`; `tools/run_cycle_report.py` writes a real-forward-trace attribution smoke by default and `--skip-attribution` disables it.

Remaining:

- Add repeated probe runs to estimate attribution variance.

## Phase 7 - Minimal regrowth

Current executable coverage:

- `MinimalRegrowthPlanner` ranks repair actions by expected gain per cost.
- `RegrowthActionSpace` maps causal attribution outputs to executable repair actions.
- Supported actions cover unzero block, change sign, increase shared scale precision, force exact anchor, reduce MTP horizon, route specialist expert, increase local activation bits, add certificate field, add verifier check and add training micro-family.
- `unzero_block`, `change_sign` and `increase_scale_precision_bits` edit concrete ternary artifacts.
- `RegrowthPatchBuilder` builds task/skill-targeted repair patches with certificate fields, costs and optional micro-family replay tasks.
- `RegrowthSimulator` measures before/after oracle score, recovery, total cost and gain per cost.
- `NonRegressionGate` compares baseline vs repaired agents over protected tasks and rejects new regressions.
- `RecrystallizationAnnealer` cools accepted repairs toward retained structure.
- `MinimalRegrowthEngine` selects the best recovering, non-regressing repair under budget.
- `tools/run_cycle_report.py` now builds Phase 7 plans from real cycle regressions through causal attribution by default, with `--skip-regrowth` as the escape hatch.
- `write_cycle_run` persists regrowth plans into `summary.json`.

Remaining:

- Apply accepted repairs to a real multi-layer model state rather than targeted repair agents.
- Persist accepted repair archives for rollback and audit.
- Feed accepted repairs into Phase 9 sleep/consolidation buffers.

## Phase 8 - Fast/normal/careful inference

Current executable coverage:

- `cortex3_inference.DifficultyRouter` derives risk, confidence, exactness and code/math signals, then maps tasks to fast, normal or careful routes.
- Routes control compression strength, layers to run, adaptive MTP horizon, verifier level, latent KV use, expert activation and latent-loop budget.
- `BudgetPredictor` estimates effective cost from weights, activations, KV, generated tokens, latent steps, experts and verifier steps before execution.
- `MixtureOfDepthsCore` executes real PyTorch `BitLinear` layers and supports early exit by route-specific confidence thresholds.
- Executed layers re-log sign+mask compression decisions and activation quantization into a per-inference `CompressionTraceLedger`.
- `TernaryKernelDispatcher` records packed sign+mask dispatch metadata with active weights, packed bytes and CPU/CUDA dispatch mode.
- `SelfSpeculativeDecoder` drafts MTP/FSP contracts, caps accepted horizon to the selected route and records MTP/FSP trace events.
- `UltraFastInferenceEngine` integrates the verifier OS, cognitive memory reconstruction and memory-augmented answer recovery, latent KV traces, specialist expert traces, proof certificates, future contracts and oracle-audited verified capability per effective cost.
- Fast-path tasks can skip runtime level-0 verification cost, but their reported verified-capability score is still audited by the oracle; confident wrong fast answers receive zero verified capability.
- `write_cycle_run` can persist inference results into `summary.json`.
- Tests cover route selection, cost ordering, early exit/depth differences, fast-path false-confidence rejection, normal-path light certificates, latent KV anchor fidelity, self-speculative horizon caps, careful-path strong verification, certificate validation, expert traces, ternary kernel dispatch records and JSON persistence.
- `cortex3_autoregressive.ARMicroDecoder` now provides a trained generated-answer path with greedy decoding, optional blockwise future-contract decoding, route-compatible MTP horizons in the certificate and per-generation runtime cost traces.
- `UltraFastInferenceEngine` accepts `ARDecoderAgent` as its answer source; generated answers flow through careful-path routing, oracle verification, certificate checks and total effective-cost accounting.
- `tools/run_cycle_report.py` writes an `autoregressive_checkpoint` artifact by default, including training metrics, DSV verification, generated samples and a careful-path inference trace; `--skip-autoregressive` disables it.

Evidence:

- `.\.venv\Scripts\python.exe -m unittest tests.test_inference`
- `.\.venv\Scripts\python.exe -m unittest tests.test_autoregressive_decoder`
- `.\.venv\Scripts\python.exe -m unittest discover -s tests`
- `.\.venv\Scripts\python.exe -m py_compile cortex3.py cortex3_analysis.py cortex3_cycle.py cortex3_ledgers.py cortex3_phases.py cortex3_selection.py cortex3_reporting.py cortex3_ternary.py cortex3_future.py cortex3_memory.py cortex3_certificates.py cortex3_attribution.py cortex3_regrowth.py cortex3_inference.py cortex3_sleep.py cortex3_improvement.py cortex3_objective.py cortex3_experiments.py cortex3_microtrain.py cortex3_autoregressive.py tools\run_cycle_report.py`
- `.\.venv\Scripts\python.exe tools\run_cycle_report.py --seed 3 --n-per-skill 1 --no-write`
- Temporary artifact write with `tools\run_cycle_report.py --out-dir <temp> --run-id final-smoke`: `InferenceCount=4`, `SleepAccepted=35`, `ImprovementProposals=4`, `ImprovementAccepted=4`.

Remaining:

- Add persistent checkpoint selection across runs instead of training a fresh smoke checkpoint per report.
- Calibrate early-exit confidence and MTP acceptance on real generated distributions.
- Add hardware-specific CUDA/CPU packed ternary kernels beyond the current dispatch metadata and reference execution.
- Benchmark fast/normal/careful choices across larger verified suites and report path Pareto fronts.

## Phase 9 - Sleep phase anti-collapse

Current executable coverage:

- `FailureReplayBuffer` converts verifier regressions into replay examples with corrected oracle labels.
- `VerifiedSyntheticDataPool` refuses synthetic examples unless they carry origin, oracle, targeted skill, verification level, contamination risk, difficulty and confidence label.
- `RealExogenousReservoir` stores non-synthetic external examples separately from synthetic training data.
- `ToolSolvedExampleFactory` creates tool/oracle-solved examples and rejects solver outputs that fail verification.
- `MetamorphicFamilyBuilder` creates metamorphic and anti-metamorphic families from existing `SkillSpec` hooks, with oracle-verified labels.
- `AntiCollapseFilter` rejects unlabeled synthetic data, high-contamination examples, duplicate prompts, calibration-gap increases and large-batch diversity collapse.
- `DiversityMetrics` tracks skill counts, origin counts, unique prompt ratio, skill entropy, origin entropy, rare-skill fraction and average contamination risk.
- `SkillConsolidationScheduler` prioritizes protected/fragile skills, applies an explicit rare-skill boost and schedules only examples accepted by the anti-collapse filter.
- `SleepPhaseConsolidator` orchestrates failure replay, tool-solved examples, metamorphic families, real/exogenous examples, anti-collapse filtering and scheduling from a `CycleReport`.
- `SleepPhaseReport` now records baseline, accepted and scheduled rare-skill fractions plus rare-skill gain, diversity delta and calibration-gap delta, making the Phase 9 success criteria directly inspectable.
- `write_cycle_run` can persist sleep phase reports into `summary.json`.
- `tools/run_cycle_report.py` writes Phase 9 sleep traces by default unless `--skip-sleep` is passed.

Evidence:

- `.\.venv\Scripts\python.exe -m unittest tests.test_sleep_phase`
- `.\.venv\Scripts\python.exe -m unittest discover -s tests`
- Temporary artifact write with `tools\run_cycle_report.py --out-dir <temp> --run-id final-smoke` includes both `inference` and `sleep_phase`.

Remaining:

- Apply accepted sleep batches to a real model update step.
- Persist replay buffers and real/exogenous reservoirs across runs.
- Add external provenance adapters for real data sources.
- Measure rare-skill retention, diversity and calibration over repeated sleep/wake cycles.

## Phase 10 - Recursive improvement gate

Current executable coverage:

- `ProposalGenerator` converts cycle actions and regressions into typed proposals for tests, compression, router changes, MTP heads, regrowth strategies and new skill/test families.
- `ImprovementProposal` carries affected skills, expected quality/cost/robustness deltas, risk, diversity tags and patch payload metadata.
- `SandboxTrainer` applies proposals only as in-memory sandbox agents; it records no touched files and creates rollback tokens.
- `ProposalPatchedAgent` simulates repair, protected-skill degradation, reward-hacking behavior and calibration regression for verifier-gated evaluation.
- `DynamicEvaluator` compares baseline vs sandbox agents on main suites and anti/metamorphic robustness suites, then measures quality delta, cost delta, robustness delta, calibration delta, protected losses and cross-skill collapse flags.
- `RewardHackingDetector` flags declared overfitting, robustness-suite collapse and overconfident failures on affected skills.
- `DiversityPreserver` prevents one proposal kind from dominating the evolutionary archive.
- `PatchAcceptanceGate` requires Pareto improvement, no protected-skill regression, no calibration regression, no reward hacking and no diversity/collapse failure.
- `EvolutionaryArchive` records accepted and rejected decisions with proposal lineage and kind counts.
- `RollbackSystem` records rollback events from accepted proposal tokens.
- `RecursiveImprovementEngine` orchestrates proposal generation, sandbox training, dynamic evaluation, acceptance and archive recording from a `CycleReport`.
- `write_cycle_run` can persist recursive improvement reports into `summary.json`.
- `tools/run_cycle_report.py` writes Phase 10 traces by default unless `--skip-improvement` is passed.

Evidence:

- `.\.venv\Scripts\python.exe -m unittest tests.test_recursive_improvement`
- `.\.venv\Scripts\python.exe -m unittest discover -s tests`
- Smoke: `RecursiveImprovementEngine(...).run(..., max_proposals=3)` accepted Pareto-improving sandbox proposals with no touched files.
- Temporary artifact write with `tools\run_cycle_report.py --out-dir <temp> --run-id final-smoke` includes `recursive_improvement` with accepted sandbox proposals and rollback data.

Remaining:

- Convert accepted in-memory proposals into signed patch artifacts.
- Persist evolutionary and rollback archives across runs.
- Run multi-generation proposal evolution with diversity pressure.
- Connect accepted proposals to real model/checkpoint training updates after verifier approval.

## Frontier Skill Discovery

Current executable coverage:

- `cortex3_frontier.FrontierSkillDiscovery` selects protected/fragile skills from the `SkillLedger`.
- It expands real cycle failures through `CompressionAdversary` to produce frontier tasks just beyond the current weak area.
- A slow/reference solver answers candidates, and only oracle-verified tasks are admitted.
- `FrontierInvariantSet` extracts expected types, metadata keys, anchor kinds and prompt obligations.
- Verified frontier tasks are distilled into `MicroTrainingExample` records and compiled with `CortexMicroTrainer` into an instrumented `BitLinear` micro-circuit.
- The compiled circuit is re-evaluated by `DynamicSkillVerifier`; reports include DSV score, training deltas, active/total weights and packed compiled weight bits.
- `write_cycle_run` persists frontier discovery reports under `summary.json["frontier_discovery"]`.
- `tools/run_cycle_report.py` writes Frontier Skill Discovery by default unless `--skip-frontier` is passed.

Evidence:

- `.\.venv\Scripts\python.exe -m unittest tests.test_frontier_discovery`
- Smoke: fragile-skill frontier tasks are slow-solved, verified, distilled and compiled into a DSV-passing micro-circuit.

Remaining:

- Run frontier discovery over larger held-out frontier suites.
- Persist and select compiled frontier circuits across runs.

## Cross-phase final objective and metrics

Current executable coverage:

- `cortex3_objective.FINAL_LOSS_TERMS` enumerates every term from the plan's `L_total`: behavior, multi-horizon, future contract, distillation behavior, distillation uncertainty, latent certificate, invariance, temporal consistency, total cognitive description, no cost shifting, hardware layout, skill regression, calibration, anchor fidelity, regrowth efficiency, verifier resistance and recursive improvement validity.
- `ObjectiveWeights` exposes the plan coefficients alpha through omega, including `lambda` as `lambda_` in Python.
- `EffectiveJouleModel` converts `CostTrace` into effective joules.
- `build_objective_report` computes weighted loss terms from cycle, fault, inference, future-contract and recursive-improvement evidence; `L_recursive_improvement_validity` now treats protected losses, reward hacking, calibration regression, collapse flags and diversity failures as per-decision invalidity.
- `ABSOLUTE_METRICS` enumerates all 15 metrics from the plan: cost per verified answer, joules per correct skill, active bits per preserved skill, rare regression rate, verifier detection rate, verifier false negatives, average verification cost, MTP rejection rate, token inflation, anchor accuracy, calibration, regrowth gain per added bit, path speed, percent without heavy verification and compiled skills from slow to fast.
- `write_cycle_run` can persist objective reports into `summary.json`.
- `tools/run_cycle_report.py` writes the objective report by default.

Evidence:

- `.\.venv\Scripts\python.exe -m unittest tests.test_objective_metrics`
- Smoke: objective report contains `17/17` loss terms and `15/15` absolute metrics.

Remaining:

- Calibrate objective weights against real training runs.
- Feed the objective directly into trainable checkpoint optimization instead of only reporting it.

## Plan experiments A-E

Current executable coverage:

- `cortex3_experiments.CortexExperimentSuite` runs the first five experiments from the plan.
- Experiment A verifies injected fault detection with `RegressionHarness.run_fault_matrix`.
- Experiment B compares fixed tests, metamorphic tests and `CompressionAdversary`-expanded failures.
- Experiment C compares minimal regrowth against a global retraining cost proxy through attribution, regrowth simulation and non-regression.
- Experiment D compares careful SlowSolve against fast compiled-route solving, requiring external oracle quality preservation and lower effective cost.
- Experiment E runs the recursive improvement sandbox and rejects reward-hacking, protected-skill-loss, calibration-regression or collapse outcomes.
- `write_cycle_run` can persist experiment reports into `summary.json`.
- `tools/run_cycle_report.py` writes experiments by default unless `--skip-experiments` is passed.

Evidence:

- `.\.venv\Scripts\python.exe -m unittest tests.test_plan_experiments`
- Smoke with seed 3: A detected `9/9` fault families, B found `+19` failures over fixed tests, C selected `increase_local_activation_bits` at `11.5` cost vs `100.0` global retrain cost, D reduced cost by about `72%`, E accepted controlled sandbox proposals with no reward-hacking flags.

Remaining:

- Run experiments over larger randomized suites and report variance.
- Compare these experiments against real trained checkpoint variants instead of only rule/reference and injected-fault agents.

## Trainable micro-checkpoint loop

Current executable coverage:

- `cortex3_microtrain.CortexMicroModel` is a PyTorch model with trainable input projection, answer head, skill head and confidence head around an instrumented `BitLinear` compiled core.
- `examples_from_tasks` converts verifier `Task` objects plus a solver into supervised micro-training examples.
- `examples_from_sleep_report` converts accepted sleep-phase examples into trainable examples.
- `CortexMicroTrainer` optimizes answer, skill and confidence losses, then requantizes the `BitLinear` core with certified zeros.
- `MicroModelAgent` exposes the trained checkpoint as a `CandidateAnswer` agent that can be evaluated by `DynamicSkillVerifier`.
- `CheckpointManager` saves and reloads `.pt` checkpoints containing config, vocabulary and state dict.
- `cortex3_autoregressive.ARMicroDecoder` trains a character-level autoregressive decoder over verifier or sleep examples instead of only selecting from an answer-class vocabulary.
- `ARLossComputer` exposes behavior, MTP multi-horizon, confidence and future-contract losses.
- `ARDecoderAgent` is evaluated directly by `DynamicSkillVerifier`, returns generated text, confidence, MTP horizon certificate metadata, runtime cost traces and a compiled-circuit certificate with distilled invariants, compiled weight bits and cheap verification contract.
- `ARCheckpointManager` saves and reloads autoregressive `.pt` checkpoints.
- `write_cycle_run` persists autoregressive checkpoint smoke reports under `summary.json["autoregressive_checkpoint"]`.

Evidence:

- `.\.venv\Scripts\python.exe -m unittest tests.test_microtrain`
- `.\.venv\Scripts\python.exe -m unittest tests.test_autoregressive_decoder`
- Smoke with seed 3: training accuracy improved from about `0.14` to `1.0`, and the DSV verified `7/7` training tasks after training.
- Autoregressive smoke with seed 3: token accuracy improved to `1.0`, exact sequence accuracy reached `1.0`, and the DSV verified `7/7` generated answers.
- Cycle artifact smoke with seed 3: `autoregressive_checkpoint.training.exact_sequence_accuracy == 1.0`, DSV `7/7`, and careful-path inference verification passed.

Remaining:

- Optimize the final Cortex objective directly during training.
- Evaluate generalization on held-out randomized suites and trained checkpoint variants.
- Benchmark MTP vs next-token-only variants under low precision.

## Full LLM pretraining bridge

Current executable coverage:

- `cortex3_llm.LLMTokenizer` trains and persists a Hugging Face `tokenizers` BPE tokenizer with required special tokens.
- `HFDatasetTextExporter` streams Hugging Face `datasets` sources or local JSONL builders into bounded text shards with export reports, document/character counts and explicit unbounded-run opt-in.
- `TextShardReader` streams text shards without loading the whole corpus into memory.
- `TokenizedCorpusBuilder` performs a two-pass corpus tokenization and writes a `uint32` memmap plus `manifest.json`.
- `MemmapCausalDataset` samples causal next-token targets and multi-horizon future targets directly from the memmap.
- `CortexTransformerLM` is a complete causal Transformer with tied embeddings, causal self-attention, MLP blocks and optional Cortex multi-horizon heads.
- `CortexObjective` optimizes next-token loss plus Cortex MTP, temporal-consistency and confidence terms when the Cortex heads are enabled.
- `LLMTrainer` supports checkpoints, strict resume, optimizer/scaler/RNG state persistence, gradient accumulation, CSV learning curves, deterministic random sampling, explicit device selection, mixed precision policy and DDP initialization from environment, including a Windows/Gloo TCPStore path that avoids unsupported libuv builds.
- `PrecisionPolicy(require_cuda=True)` raises when CUDA is required but unavailable, preventing silent CPU fallback.
- `llm_doctor_report` and `tools/train_llm.py doctor` audit Python dependencies, CUDA availability, requested precision, `torch.distributed`, Gloo/NCCL readiness and write a persistent `doctor_report.json`.
- `LLMComparisonRunner` trains a baseline next-token Transformer and a Cortex multi-horizon Transformer on the same corpus/cache, then writes `comparison_report.json`, `report.md`, `learning_curve.png`, both final checkpoints and both learning-curve CSV files.
- `LLMComparisonMatrixSuite` prepares one shared tokenizer/memmap for an arbitrary corpus, repeats the baseline-vs-Cortex comparison over multiple seeds and writes `comparison_matrix_report.json`, `comparison_matrix_report.md`, `comparison_matrix_ratios.png` and aggregate validation learning curves.
- `LLMCorpusMatrixSuite` repeats the comparison matrix across multiple named corpora, persists per-corpus reports and writes `corpus_matrix_report.json`, `corpus_matrix_report.md`, `corpus_matrix_ratios.png` and aggregate multi-corpus learning curves with corpus-level, seed-level and sample-level proof metrics.
- `LLMExperimentRunner` executes a manifest-driven full experiment: doctor audit, HF/path corpus preparation, cross-corpus matrix training/proof and final `experiment_report.json`/Markdown artifacts.
- `LLMBenchmarkSuite` runs multiple deterministic domains, persists per-domain comparison artifacts and writes an aggregate `benchmark_report.json`, `benchmark_report.md` and `benchmark_ratios.png`.
- `LLMStatisticalBenchmarkSuite` repeats the benchmark over multiple seeds, persists each seed/domain comparison and writes `statistical_benchmark_report.json`, `statistical_benchmark_report.md` and `statistical_benchmark_ratios.png` with mean, median, min ratio, win-rate, per-domain and per-seed aggregates.
- `tools/train_llm.py` exposes `smoke`, `prepare-hf` and `compare` commands for local proof runs, Hugging Face corpus preparation and larger text-shard corpora.
- `tools/train_llm.py compare-matrix` exposes the arbitrary-corpus multi-seed proof gate while reusing one shared tokenized corpus.
- `tools/train_llm.py corpus-matrix` exposes the multi-corpus x multi-seed proof gate for prepared corpus suites.
- `tools/train_llm.py run-experiment` executes a normalized JSON manifest for reproducible large-corpus GPU/DDP experiments.
- `tools/train_llm.py benchmark` exposes the multi-domain proof gate and supports CPU `bf16` validation.
- `tools/train_llm.py benchmark-matrix` exposes the multi-domain x multi-seed proof gate and fails `--require-win` unless every seed-domain sample wins with a nonzero baseline and bounded next-token regression.
- `tools/launch_llm_ddp.py` launches true local multi-process DDP workers, pins the Gloo interface and writes per-rank logs.
- `.github/workflows/ci.yml` runs the LLM smoke command.

Evidence:

- Local GPU environment after dependency correction: NVIDIA GeForce RTX 5070, driver CUDA `13.2`, `torch==2.11.0+cu128`, `torch.version.cuda==12.8`, `cuda_available=True`, `cuda_device_count=1`, `distributed_available=True`, `gloo_available=True`, `nccl_available=False` on Windows.
- CUDA dependency correction: the previous environment had `torch==2.12.1+cpu` despite a visible RTX 5070. Installed the official CUDA wheel with `pip install --force-reinstall torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128`; `requirements-cuda-cu128.txt` records the reproducible install command.
- Doctor validation: `tools\train_llm.py doctor --out-dir runs\llm-doctor-cuda-validation --require-cuda --precision bf16 --device cuda` passed with CUDA visible and bf16 resolving on `cuda`.
- `.\.venv\Scripts\python.exe tools\train_llm.py smoke --out-dir runs\llm-smoke-dev-48 --steps 48 --require-win`
- Smoke proof: baseline score `0.022321`, Cortex score `0.145833`, Cortex/baseline `6.533x`, next-token-loss regression ratio `1.020`, proof passed.
- CUDA smoke validation: `tools\train_llm.py smoke --out-dir runs\llm-cuda-smoke-validation --steps 48 --precision bf16 --device cuda --require-cuda --require-win` passed on RTX 5070 with baseline score `0.029576`, Cortex score `0.147135`, Cortex/baseline `4.975x`, next-token-loss regression ratio `1.017305`.
- CUDA resume debug/fix: after installing the CUDA wheel, checkpoint resume exposed `TypeError: RNG state must be a torch.ByteTensor` when restoring CUDA RNG state loaded with `map_location=cuda`. `LLMTrainer.load_checkpoint` now normalizes saved CUDA RNG states back to CPU `uint8` tensors before `torch.cuda.set_rng_state_all`; the targeted resume test passes on CUDA.
- CUDA external Wikitext comparison matrix: `tools\train_llm.py compare-matrix runs\hf-wikitext2-validation\text_shards --out-dir runs\llm-wikitext2-cuda-compare-matrix-validation --seeds 17,29 --vocab-size 512 --seq-len 64 --d-model 64 --n-heads 4 --n-layers 2 --steps 48 --batch-size 8 --precision bf16 --device cuda --require-cuda --require-win` passed with `2/2` seeds, mean Cortex/baseline ratio `24.864x`, min ratio `24.500x`, aggregate CSV/PNG learning curves and CUDA recorded in per-seed reports.
- Versioned experiment manifests: `experiments/wikitext_cuda_validation.json` for fast local CUDA validation and `experiments/c4_local_cuda_manifest.json` for a runnable large C4 + repo-local text CUDA run.
- Versioned Wikitext CUDA manifest validation: `tools\train_llm.py run-experiment experiments\wikitext_cuda_validation.json` passed with `2/2` seeds, win-rate `1.0`, mean Cortex/baseline ratio `11.861x`, min ratio `10.889x`, CUDA doctor passed and aggregate CSV/PNG learning curves written.
- `.\.venv\Scripts\python.exe tools\train_llm.py benchmark --out-dir runs\llm-benchmark-validation --domains sequence,anchors --repeats 96 --steps 48 --batch-size 8 --precision bf16 --require-win`
- Benchmark proof through `codex-test` with `gradient_accumulation_steps=2`: `2/2` domains passed, mean Cortex/baseline ratio `32.097x`, minimum domain ratio `25.861x`, mean baseline score `0.005301`, max next-token-loss regression ratio `1.001049`.
- `.\.venv\Scripts\python.exe tools\train_llm.py benchmark-matrix --out-dir runs\llm-benchmark-matrix-validation --domains sequence,anchors --seeds 11,23 --repeats 96 --steps 48 --batch-size 8 --precision bf16 --require-win`
- Statistical benchmark proof through `codex-test`: `4/4` seed-domain samples passed, win-rate `1.0`, mean Cortex/baseline ratio `26.520x`, median ratio `18.829x`, minimum ratio `4.840x`, mean baseline score `0.012835`, max next-token-loss regression ratio `1.067812`.
- DDP root cause and fix: the local Windows/Gloo path needs explicit TCPStore `use_libuv=False`; when Gloo auto-selected a bad host route it tried `kubernetes.docker.internal`. Cortex now pins `GLOO_SOCKET_IFNAME` and uses an explicit `TCPStore(..., use_libuv=False)` for local Gloo env initialization.
- `.\.venv\Scripts\python.exe tools\launch_llm_ddp.py --nproc 2 --master-port 29752 --gloo-interface Ethernet --timeout 240 -- smoke --out-dir runs\llm-ddp-smoke-validation --steps 48 --precision bf16 --require-win`
- DDP smoke proof through `codex-test`: `world_size=2`, `distributed=True`, proof passed, baseline score `0.002790`, Cortex score `0.149740`, Cortex/baseline `53.667x`, next-token-loss regression ratio `0.952`.
- `prepare-hf` is covered with a local Hugging Face JSONL dataset path that exports 30 documents into multiple shards, writes `hf_export_report.json`, trains a BPE tokenizer and builds a causal memmap manifest.
- CLI HF/text validation: `tools\train_llm.py prepare-hf --dataset text --data-file README.md ...` exported 148 documents into 15 shards and built a 9,331-token `uint32` memmap.
- External HF validation: `tools\train_llm.py prepare-hf --dataset Salesforce/wikitext --config-name wikitext-2-raw-v1 --split train --text-field text --out-dir runs\hf-wikitext2-validation --max-documents 200 --min-text-chars 20 --shard-chars 4096 --vocab-size 512 --seq-len 64 --max-horizon 4` exported 200 Wikitext documents into 19 shards and built a 29,008-token `uint32` memmap. The older short id `wikitext` was rejected by the current HF stack with a `namespace/name` error; the exporter now converts that failure into an actionable namespaced-id message.
- HF-prepared compare smoke: `tools\train_llm.py compare runs\hf-text-cli-default-cap\text_shards ... --precision bf16` passed with baseline score `0.002778`, Cortex score `0.011719`, Cortex/baseline `4.219x`, next-token-loss regression ratio `0.976`.
- External Wikitext comparison matrix: `tools\train_llm.py compare-matrix runs\hf-wikitext2-validation\text_shards --out-dir runs\llm-wikitext2-compare-matrix-validation --seeds 17,29 --vocab-size 512 --seq-len 64 --d-model 64 --n-heads 4 --n-layers 2 --steps 48 --batch-size 8 --precision bf16 --require-win`
- External Wikitext proof through `codex-test`: `2/2` seeds passed, win-rate `1.0`, mean Cortex/baseline ratio `30.296x`, minimum ratio `23.092x`, mean baseline score `0.000977`, max next-token-loss regression ratio `1.108555`.
- Shared-corpus comparison matrix validation: `tools\train_llm.py compare-matrix runs\compare-matrix-validation-corpus --out-dir runs\llm-compare-matrix-validation --seeds 17,29 --vocab-size 256 --seq-len 32 --d-model 64 --n-heads 4 --n-layers 2 --steps 48 --batch-size 8 --precision bf16 --require-win`
- Comparison matrix proof through `codex-test`: `2/2` seeds passed, win-rate `1.0`, mean Cortex/baseline ratio `16.979x`, minimum ratio `5.833x`, mean baseline score `0.014518`, max next-token-loss regression ratio `1.071590`, shared `corpus/manifest.json` reused by every seed.
- Multi-corpus matrix validation: `tools\train_llm.py corpus-matrix --corpus seed=runs\corpus-matrix-validation-seed --corpus anchors=runs\corpus-matrix-validation-anchors --out-dir runs\llm-corpus-matrix-validation --seeds 17,29 --vocab-size 256 --seq-len 32 --d-model 64 --n-heads 4 --n-layers 2 --steps 48 --batch-size 8 --precision bf16 --require-win`
- Corpus matrix proof through `codex-test`: `4/4` corpus-seed samples passed, win-rate `1.0`, mean Cortex/baseline ratio `40.854x`, median ratio `27.898x`, minimum ratio `5.833x`, mean baseline score `0.011361`, max next-token-loss regression ratio `1.151529`.
- External HF corpus matrix validation: `tools\train_llm.py corpus-matrix --corpus wikitext=runs\hf-wikitext2-validation\text_shards --corpus seed=runs\corpus-matrix-validation-seed --out-dir runs\llm-hf-corpus-matrix-validation --seeds 17,29 --vocab-size 512 --seq-len 64 --d-model 64 --n-heads 4 --n-layers 2 --steps 48 --batch-size 8 --precision bf16 --require-win`
- External HF corpus matrix proof through `codex-test`: `4/4` corpus-seed samples passed, win-rate `1.0`, mean Cortex/baseline ratio `15.538x`, median ratio `16.127x`, minimum ratio `6.805x`, mean baseline score `0.010824`, max next-token-loss regression ratio `1.099396`.
- Manifest experiment validation: `tools\train_llm.py run-experiment runs\experiment-manifest-validation.json` ran doctor, prepared one local JSON Hugging Face corpus plus one paths corpus, executed corpus-matrix and wrote final experiment artifacts. A Windows PowerShell UTF-8 BOM manifest bug was reproduced and fixed by reading manifests with `utf-8-sig`.
- Manifest experiment proof through `codex-test`: `4/4` corpus-seed samples passed, win-rate `1.0`, minimum ratio `6.813x`, max next-token-loss regression ratio `1.001864`; the run wrote `corpus_matrix_learning_curves.csv/png` and per-corpus `comparison_matrix_learning_curves.csv/png`.
- Checkpoint resume unit coverage: a Cortex trainer runs two optimizer steps with `gradient_accumulation_steps=2`, writes step/final checkpoints, resumes from `checkpoint_final.pt` to step 4 and preserves curve plus RNG state in the checkpoint payload.
- CLI resume validation: `tools\train_llm.py smoke --out-dir runs\llm-resume-cli-validation --steps 2 ...` then `--steps 4 --resume ...` resumed both `baseline_ntp` and `cortex3` from `checkpoint_final.pt` with `start_step=2`, `optimizer_steps=2`, `effective_batch_size=16` and `final_step=4`.
- DDP accumulation validation: `tools\launch_llm_ddp.py --nproc 2 ... --gradient-accumulation-steps 2` completed with `distributed=True`, `world_size=2`, proof passed and `effective_batch_size=32` for both baseline and Cortex.
- DDP CUDA preflight validation: `tools\launch_llm_ddp.py --nproc 2 ... --device cuda --require-cuda` now fails before spawning workers because one visible CUDA device cannot serve two local CUDA ranks; CPU/Gloo DDP still passed after the CUDA wheel install.
- `.\.venv\Scripts\python.exe -m unittest discover -s tests`: `124` tests passed.

Remaining:

- Run a genuine long large-corpus experiment from an external Hugging Face dataset such as C4/FineWeb, not only the deterministic local smoke corpus or local JSONL export test.
- Validate NCCL multi-GPU runs on hardware exposing at least two CUDA devices; this Windows machine validates single-GPU CUDA bf16 and CPU/Gloo DDP, but `nccl_available=False` and only one CUDA device is visible.
- Scale model sizes and training steps, then publish the same statistical benchmark on broad external corpora instead of only deterministic local domains.
- Connect accepted recursive-improvement proposals to persisted LLM checkpoint patches with rollback archives.
