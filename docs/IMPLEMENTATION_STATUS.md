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
- Algebra symbolic oracle coverage now includes exact 2x2 linear systems in addition to scalar linear equations and quadratic roots: generated tasks require ordered assignments such as `x=5, y=-1`, metamorphic variants swap/scale equations, anti-metamorphic variants change the RHS, and unlabeled or reordered answers are rejected.
- Entity tracking now includes transfer-chain tasks that track final object holders across multiple people, distractor people and distractor places; anti-metamorphic variants change the final holder and must be re-verified exactly.
- Code unit-test tasks now include richer stateful contracts (`dedupe_preserve_order`, `merge_counts`) and the P1 oracle executes visible tests, hidden tests, deterministic checks and no-argument-mutation checks instead of only example matching.
- Full phase report contract: `docs/CORTEX_PHASE_REPORT_SCHEMA.json` publishes the P1-P10 JSON Schema, and `validate_cortex_phase_report_contract` now gates final full-Cortex training reports before `training_report.json` / `cortex_phase_report.json` are written.
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
- `tests.test_llm_pretraining.LLMPretrainingHarnessTest.test_cortex_phase_report_contract_accepts_complete_full_phase_payload`, `test_published_cortex_phase_report_schema_matches_runtime_contract`, `test_cortex_phase_report_contract_rejects_missing_phase_or_critical_key` and `test_full_cortex_phase_controller_uses_all_modules_during_training`.
- `tests.test_cortex3.Cortex3Test.test_algebra_oracle_accepts_exact_symbolic_linear_system_assignments` verifies exact ordered 2x2 assignment acceptance and rejection of unlabeled, reordered or wrong assignments.
- `tests.test_cortex3.Cortex3Test.test_code_unit_test_oracle_checks_hidden_and_property_contracts`, `test_code_unit_test_generator_includes_rich_stateful_templates` and `test_entity_tracking_generator_includes_transfer_chain_oracle` verify hidden/property code contracts and multi-entity transfer tracking.

Remaining Phase 1 hardening:

- Continue broadening algebra/calibration grammars and add a domain coverage audit for generated oracle families.

## Phase 2 - Instrumented Ternary Core

Current executable coverage:

- `TernaryBlock`, `ternarize_values`, zero states and estimated bit accounting.
- `cortex3_ternary.BitLinear` provides a sign+mask layer with shared scales, activation quantization, optional residual runtime, packed int2 ternary weight buffers, native RawKernel CUDA diagnostic paths and a strict PyTorch C++/CUDA extension backend (`native_cuda_backend=extension`) for CUDA tensors by default; kernel-variant `auto` measures tiled/warp/WMMA variants with CUDA events when dtype/shape/backend allow it, caches the best choice by device/dtype/shape, can persist/reload JSON autotune profiles, keeps a layer-local fast cache, skips dense STE linear in forward through custom autograd, computes large fp16/bf16 forward tiles via WMMA directly from packed int2 weights, computes CUDA `grad_input` directly from packed int2 weights in the fast STE backward, uses WMMA fp16/bf16 for aligned and padded-edge `grad_input`, uses WMMA fp16/bf16->fp32 for aligned and padded-edge `grad_weight` + `grad_bias`, keeps hand-written warp/tiled kernels for very small phase micro-circuits, runs CUDA-fused requantize/pack after weight updates, repacks only when the weight version changes, and traces autotune candidates/cache hits plus grad-input/grad-weight kernel counts. PyTorch is required; CuPy remains useful as the RawKernel diagnostic backend, while the extension backend is now exercised by the LLM smoke, by `tools/train_llm.py profile-batch`, by `tools/train_llm.py profile-matrix`, and by `tools/train_llm.py profile-autosize` including adaptive diverse measured candidate selection by default.
- Native CUDA extension build/cache writes are forced under the repository-local `.codex` directory, stale extension locks are removed or rejected quickly, and the controller no longer waits on an orphaned global extension lock outside `LLMTEST`.
- `ResidualSynapseBuffer` stores reconstruction residuals for compressed blocks.
- `CompressionTraceLedger` records compression decisions, activation quantization, expert activations, KV mode events, MTP/FSP events, packed ternary dispatches, native CUDA kernel dispatch counts, backend counts, requantize backend counts and grad-weight backend counts.
- Compression decisions include active count, provisional/certified zeros, estimated bits, threshold and residual L1.
- `LayerForwardEvent` records real `BitLinear` forward passes with layer id, input/output shapes, active weights, estimated packed weight bits and activation bits.
- `BitLinear` uses a straight-through runtime weight path so gradients reach `float_weight` while the forward value is read from packed int2 ternary weights by default.
- Tests cover exact parity with `nn.Linear` only when residual runtime is explicitly enabled, gradient survival to inputs and weights, packed int2 CPU dispatch, native packed int2 CUDA dispatch, fp32/fp16/bf16 native-kernel value parity, fp32/fp16/bf16 fast STE CUDA backward parity against dense STE, fp32/fp16/bf16 CUDA-fused requantize/pack parity against the PyTorch sync path, auto-selection between tiled and warp kernels, and micro-model layer-forward traces.

Remaining:

- Feed layer-forward traces into persisted cycle reports outside inference-specific trace summaries.
- Use layer-forward traces as first-class evidence in causal attribution block probes.
- Broaden the adaptive diverse measured, observed-VRAM-budgeted, gradient-accumulation-aware, minimum-two-measurement-seed, upper-confidence-exploring and robust-score automatic batch/shape search to larger and longer LLM batches, wider candidate grids and higher utilization gates.

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
- `OutputGoalContract` and `OutputGoalDecision` extend FSP beyond token ids: P3 now gates complete expected outputs, exact/no-extra-text obligations, required anchors, internal leakage markers and task-declared forbidden substrings, records output-goal decisions and forbidden matches in the future-contract ledger, persists them through checkpoints and feeds rejected output-goals into `L_future_contract`.
- Generic P8 self-speculative FSP now gates contracts against tokens derived from the actual emitted answer text, records `observed_tokens_source`, `observed_tokens` and `self_verified_tokens=false`, and rejects incomplete observed token blocks instead of accepting a contract by comparing it to its own predicted tokens.
- `CortexTrainingPhaseController` emits a strict P3 output-goal contract for the `ACCEPT/REJECT` gate result; the phase raises instead of silently continuing if the output-goal gate rejects.

Remaining:

- Calibrate the standalone `MTPFSPHeads` module on larger held-out micro-task distributions, not only the seed smoke suite.
- Compare MTP accepted blocks against verified quality per effective cost across checkpoint variants in cycle reports.

## Phase 4 - Cognitive memory

Current executable coverage:

- Exact anchor extraction and fidelity scoring.
- Long-context and entity-tracking skill families.
- `CognitiveMemory` stores recent segments in exact KV and evicts older segments into compact latent KV.
- `LatentKVStore` keeps Torch embeddings, compact summaries and exact anchors instead of old full text.
- `Query-conditioned memory reconstruction` retrieves exact recent and latent old segments by embedding similarity, token overlap, anchor intent and learned utility-credit score bias, then force-includes any segment carrying explicitly required anchors.
- `AnchorFidelityVerifier` proves required exact anchors are preserved in reconstructed context.
- `compression_report` records recent/latent counts, latent token compression ratio and ledger anchors.
- `write_cycle_run` can persist cognitive memory reports into `summary.json`.
- `UltraFastInferenceEngine` can convert a faithful reconstruction into a memory-augmented generated answer for long-context anchors and entity locations, replacing a weaker base answer without reading `expected`.
- Memory-augmented answers carry certificate fields, selected segment ids, anchor fidelity and the displaced base answer in `raw`; inference JSON persists this audit trail.
- `LearnedMemoryPolicy` is part of the Transformer forward, mixes exact/latent/drop states differentiably, receives the `learned_memory` loss and is audited by the full Cortex phase controller.
- `CognitiveMemory.ingest` now consumes learned `MemoryRetentionDecision` objects from decoded LLM batches, so the shared P4 memory applies exact storage, direct latent KV compression or real non-anchored drop instead of only counting policy probabilities.
- The Exact Anchor Ledger is a hard storage safety gate: learned `DROP` requests on anchored segments are promoted to exact storage, recorded as anchor-safety overrides, reconstructed and fidelity-checked.
- Learned retention decisions are persisted through `memory_state`, restored on checkpoint resume, surfaced in `compression_report`, `training_influence`, architecture audit and phase-deliverable audit.
- `MemoryUtilityCredit` now records both which retained learned-memory segments were selected by faithful downstream reconstructions and which retained learned-memory segments were left unselected. Selected faithful segments create positive utility; unselected retained exact/latent segments create negative forget/compress credits, and the controller converts those counter-credits into pressure toward latent/drop instead of rewarding the retained mode. P4 input-anchor checks, phase memory audits, Frontier circuit bindings and P8 memory-augmented inference feed those credits back into a normalized exact/latent/drop utility prior. `LearnedMemoryPolicy` injects that prior into its logits, `CortexObjective` aligns the policy distribution to it, checkpoints restore the prior plus credit ledger, and `CognitiveMemory.reconstruct` now uses learned utility credits as a direct exact/latent segment ranking bias while preserving hard required-anchor inclusion.
- `tools/benchmark_learned_memory_policy.py` runs a short shared-weight ablation against disabled learned memory, freezes non-memory parameters, trains only `learned_memory.*`, and reports before/after losses, policy gradients, exact/latent/drop decisions and storage ratio.
- `CompiledCircuitMemoryBinding` turns Frontier circuits into retained P4 memory objects with circuit id, source/frontier/held-out lineage, obligations, metadata keys, anchors, fidelity and selected segment ids.
- Compiled circuit bindings now pass through a dedicated learned-memory retention decision (`learned_memory_compiled_circuit_policy`), are retained as latent anchor-preserving memory objects, and receive utility credit when reconstructed for FastSolve/P7/P9 reuse. Phase reports expose compiled-circuit learned retention counts and utility-credit counts.
- `CompiledFrontierAgent`, P8 inference, P9 sleep-frontier FastSolve and P7 Frontier repair candidates require a reconstructible P4 memory binding whenever the full LLM controller supplies shared memory; the P5 compiled-circuit certificate carries memory-binding claims.
- Checkpoint resume now restores P4 memory before loading the Frontier registry, then verifies a restored compiled FastSolve with the restored binding; missing persisted registries are hard errors when a checkpoint advertises compiled circuits.

Remaining:

- Run large long-context ablations proving the learned exact/latent/drop storage policy improves cost/quality over deterministic memory alone on held-out anchors.
- Scale compiled-skill memory retention across larger registries, competing circuits and multi-cycle restarts.
- Prove learned positive and negative downstream utility credit over repeated long-context wake/sleep cycles and larger held-out anchor/circuit workloads.

## Phase 5 - Latent reasoning with certificates

Current executable coverage:

- `CandidateAnswer.certificate` field and certificate bit accounting.
- `LatentProofState` stores a Torch latent proof vector, latent step count and checksum.
- `LatentProofState.to_dict/from_dict` serializes proof vectors into JSON-safe rounded values for persisted audit trails.
- `LatentReasoningWorkspace` is now an explicit trainable LLM module: it attends over hidden states, executes multiple latent transitions, feeds a projected latent summary back into the hidden stream before logits/MTP/certificates, and records latent workspace KV cost.
- `CertificateHead` maps hidden states to latent proof state, answer logits, certificate type logits and uncertainty.
- `CortexObjective` includes `latent_workspace` loss, binding the workspace summary to the certificate latent state while keeping latent steps stable and trainable.
- The full LLM controller now materializes real `CertificateHead` outputs into verified `ShortCertificate` artifacts with model-token consistency, latent checksum verification, target-match metadata, checkpoint persistence and P5 audit gates.
- Model-head certificates now carry latent-workspace checksum, step count and binding claims; full Cortex training requires `use_latent_reasoning_workspace=True`, persists workspace counters through checkpoints, and fails P5/full-architecture audits if workspace forward/steps/certificate binding are absent.
- Exact-match task certificates now bind to the task contract target (`task.expected`) when it exists instead of using the produced answer as its own expected value; P8 inference certificate verification builds claims/tool args through `certificate_contract_for_task`.
- `CertificateHeadCalibrator` trains the certificate head on verifier micro-task answers, certificate types and uncertainty targets.
- `ShortCertificate` carries answer, claims, uncertainty, latent checksum, anchors and optional tool contract.
- `CertificateVerifier` checks uncertainty bounds, latent checksum and tool-backed verification.
- `CertificateVerifier` accepts explicitly calibrated high-uncertainty certificates, so `UNKNOWN` can stay low-confidence without being treated as proof corruption.
- `RandomDelatentizer` samples latent dimensions deterministically for audit probes and detects tampering.
- Tool-backed checks include arithmetic, exact match, model-token certificate consistency, anchor fidelity, entity-tracking transfer/location-chain verification, multi-step linear algebra, SymPy-backed symbolic quadratic algebra, SymPy-backed exact 2x2 linear systems, richer executable code unit tests and compiled-circuit contracts.
- `CertificateType.COMPILED_CIRCUIT`, `build_compiled_circuit_certificate` and the `compiled_circuit` tool bind compiled skill reuse to a canonical contract checksum, source/frontier task lineage, DSV pass state, runtime output verification and answer checksum.
- `ProofCarryingAnswer` converts answer + certificate + uncertainty into `CandidateAnswer` with a serializable latent proof payload.
- `ProofCarryingGenerator` connects a calibrated certificate head to DSV-compatible answer generation and verifies every emitted certificate.
- `evaluate_certificate_efficiency` measures token reduction, quality preservation and calibration preservation.
- `write_cycle_run` can persist short certificates into `summary.json`.
- `UltraFastInferenceEngine` treats proof-carrying certificate verification as a gate; a tampered latent proof makes `InferenceResult.passed` false and zeroes verified capability per cost.
- `CompiledFrontierAgent` attaches a verified P5 compiled-circuit certificate to every selected Frontier circuit answer; the compiled-circuit contract now also binds the P3 output-goal contract id, obligations, pass state and violations. The LLM phase report persists `frontier_compiled_contract_verified`, `frontier_output_goal_contract_passed` and the contract checksum for accepted P7 repairs.
- `CertificateType.ALGEBRA` now has two strict tool paths. `algebra_linear` requires a multi-step proof for integer linear equations: subtract constant, divide by coefficient and substitute the result back into the equation. `sympy_symbolic` handles quadratic symbolic tasks and exact 2x2 linear systems through SymPy, verifies exact root/assignment sets and substitution checks, and is required by the full LLM P5 audit through `certificate_symbolic_solver_events` plus `certificate_symbolic_system_solver_events`. `entity_tracking` verifies final holder/location, ordered transfer/location steps, distractors and required anchors for P1 entity tasks. `code_tests` separates visible and hidden tests, can require hidden tests, and checks deterministic/no-argument-mutation properties when requested. The full LLM P5 audit fails the deliverable if linear algebra, symbolic algebra/system solving or rich code verification rejects.
- `tests.test_certificates.CertificatesTest.test_symbolic_algebra_certificate_uses_sympy_solver_for_linear_system` verifies the 2x2 certificate path accepts a correct assignment and rejects wrong, unlabeled or tampered solution-map claims.
- `tools/run_cycle_report.py` writes a trained proof-carrying certificate smoke by default; `--skip-certificates` disables it.

Remaining:

- Expand tool verification to broader symbolic domains beyond the current exact quadratic and 2x2 linear-system SymPy paths, plus additional specialized solvers/oracles for non-algebra tasks.
- Compare workspace-enabled vs workspace-disabled training on held-out reasoning traces once long tests are authorized.
- Benchmark certificate-token savings and semantic reliability of model-head certificates over held-out reasoning traces.

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
- `AttributionPolicyMemory` learns which causal hypotheses actually led to verified P7 repairs, with posterior success, gain-per-cost weighting, confidence and dominant intervention tracking; in full LLM training this learning is sourced from the real `_apply_model_regrowth` patch report, not from the simulated `RegrowthPlan` score. Observations are keyed by `(skill, cause, intervention)` so a successful P7 patch reweights only the matching P6 intervention and cannot globally boost a different repair under the same cause.
- `CausalAttributionEngine` can reweight deterministic probe estimates through that learned policy before P7 chooses a repair, using each `CauseEstimate.best_intervention` as the default lookup key while keeping legacy checkpoint entries readable for older runs.
- P6 evidence is now carried into every P7 executable action: source engine, failed task/skill/reason, top cause, selected cause probability, counterfactual dimension, intervention, recovery, score delta, gain-per-cost, targeted/global costs, probe counts and policy signal counts.
- `cluster_regressions` groups failures by top cause and skill with recommended intervention.
- `write_cycle_run` can persist causal attribution reports and clusters into `summary.json`; `tools/run_cycle_report.py` writes a real-forward-trace attribution smoke by default and `--skip-attribution` disables it.
- The full LLM Cortex phase controller persists the learned attribution policy in checkpoints, exposes observations/successes in architecture audits, and requires the P6 deliverable to include both counterfactual probe breadth and verified repair-outcome learning.
- The Skill Ledger now conditions `SkillAwareExpertMoE`: fragile/protected skills become a stable expert-context distribution, verified phase replays carry per-skill expert targets, `CortexObjective` adds a differentiable routing-alignment loss, and checkpoints persist the context so resume keeps the same skill pressure.

Remaining:

- Scale the learned policy beyond the short audit loop: repeated trace corpora, multi-cause regressions, long repair histories and larger held-out causal suites.

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
- The full LLM Cortex phase controller converts verified regrowth outputs into causal replay examples, then applies the accepted repair to real Transformer parameters with a bounded gradient patch over targeted Cortex submodules.
- The P7 model patch gate records before/after repair loss, protected loss, parameter L1 delta, updated parameter names, non-regression status and ternary requantization in `cortex_phase_report.json` and checkpoint sidecars.
- Accepted P7 model patches are now signed and materialized as executable rollback artifacts with pre-patch tensors, pre/post checksums and patch metadata; `rollback_regrowth_model_patch` restores those weights only if the live model still matches the recorded post-patch checksums.
- Accepted P7 model patches are now causally grounded by a hard P6->P7 gate: `_apply_model_regrowth` refuses missing or non-recovering causal evidence, signs the causal evidence into the patch payload, persists it in rollback artifacts, exposes `regrowth_model_causal_grounded_count`, and requires it in architecture and deliverable audits.
- Accepted P7 model patches now protect against phase-wide regressions rather than only the latest replay samples: the controller stores `replay_phase_ids`, selects the latest replay batch from every available phase for the protected-loss gate, signs `protected_replay_phase_ids`, exposes `regrowth_model_protected_replay_phase_count`, and architecture/deliverable audits require `phase_balanced_replay` with at least five protected phases in the full P1-P10 path.
- Accepted P7 repairs now feed their real model-patch outcome back into `AttributionPolicyMemory`: `repair_loss_delta`, `protected_loss_delta`, `protected_loss_tolerance`, `non_regression_passed`, `parameter_delta_l1`, `signed_patch_id` and `rollback_executable` close the loop P6 hypothesis -> P7 applied repair -> learned future attribution prior.

Remaining:

- Calibrate repeated model-state regrowth schedules across long multi-corpus runs, not only the strict per-audit bounded repair gate.

## Phase 8 - Fast/normal/careful inference

Current executable coverage:

- `cortex3_inference.DifficultyRouter` derives risk, confidence, exactness and code/math signals, then maps tasks to fast, normal or careful routes.
- Routes control compression strength, layers to run, adaptive MTP horizon, verifier level, latent KV use, expert activation and latent-loop budget.
- `BudgetPredictor` estimates effective cost from weights, activations, KV, generated tokens, latent steps, experts and verifier steps before execution.
- `MixtureOfDepthsCore` executes real PyTorch `BitLinear` layers and supports early exit by route-specific confidence thresholds.
- Executed layers re-log sign+mask compression decisions and activation quantization into a per-inference `CompressionTraceLedger`.
- `TernaryKernelDispatcher` records the actual packed ternary runtime event produced by `BitLinear` after the layer forward: backend, source layer id, device, native-kernel flag, native backend, kernel variant, requantize backend and autotune metadata. If a layer completes without a packed ternary dispatch event, inference fails instead of emitting a generic CPU/CUDA mode.
- `SelfSpeculativeDecoder` drafts MTP/FSP contracts, caps accepted horizon to the selected route and records MTP/FSP trace events.
- `UltraFastInferenceEngine` integrates the verifier OS, cognitive memory reconstruction and memory-augmented answer recovery, latent KV traces, specialist expert traces, proof certificates, future contracts, output-goal contracts and oracle-audited verified capability per effective cost.
- In the full LLM harness, P8 now uses `CortexTransformerInferenceAgent` as the default answer source: it runs the real `CortexTransformerLM`, decodes with an adaptive MTP/FSP block gate instead of a pure greedy loop, proposes contiguous next-token + horizon-2 blocks from the model heads, accepts multi-token blocks only through `FutureContractEngine`, records rejected blocks explicitly, carries model `CertificateHead` metadata, accounts generated-token/contract cost and records `inference_model_backed_adaptive_mtp_*` events required by architecture/deliverable audits.
- P8 model-backed outputs now feed verified replay: a passing generation is replayed as-is, while a failing generation creates an oracle-corrective P8 replay example carrying the failed model answer and generated token ids. The audit requires nonzero `inference_model_backed_replay_events`, so the real Transformer-backed route must influence `replay_loss`.
- When a compiled Frontier circuit is available, `UltraFastInferenceEngine` always supplies P4 memory to `CompiledFrontierAgent` and requires the circuit memory binding to be established and reconstructed before using the compiled FastSolve answer, even when the caller did not pass an explicit memory instance.
- On checkpoint resume, the full LLM controller validates restored P8 FastSolve before continuing: selected compiled circuit, oracle verification, P4 binding fidelity, output-goal contract and compiled-circuit certificate must all pass.
- Auto-resume now selects the highest-step complete checkpoint across `checkpoint_final.pt` and `checkpoint_step_*.pt`, using the sidecar size and step contract for every candidate; a stale or corrupt final checkpoint can no longer hide a newer complete intermediate checkpoint after an interrupted continuation run.
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
- `tests/test_llm_pretraining.py::LLMPretrainingHarnessTest::test_resume_selects_highest_complete_checkpoint_over_stale_final` verifies that resume chooses a newer complete intermediate checkpoint over an older complete final checkpoint, and keeps doing so when the final sidecar is corrupt.
- `python -m unittest tests.test_inference` verifies that P8 `kernel_dispatches` match the P2 `packed_ternary_dispatches` ledger backend/layer/native fields instead of generic pre-forward labels.
- `python -m unittest tests.test_frontier_discovery.FrontierSkillDiscoveryTest.test_frontier_discovery_slow_solves_distills_and_compiles_fragile_skill` verifies that P8 creates and uses an internal P4 memory binding for compiled FastSolve when no memory instance is explicitly supplied.
- `python -m unittest tests.test_certificates` and `python -m unittest tests.test_inference` verify that exact-match certificates reject wrong answers such as `OK extra` against the real task target.
- `python -m unittest tests.test_future_contracts` and `python -m unittest tests.test_inference` verify that incomplete observed future-token blocks are rejected and that P8 FSP reports answer-text observations rather than self-verified contract tokens.

Remaining:

- Calibrate early-exit confidence and adaptive model-backed MTP block acceptance on real generated distributions.
- Broaden exact P2/P8 runtime-backend validation to more model-backed generated shapes and dtypes once longer runs are authorized.
- Benchmark fast/normal/careful choices across larger verified suites and report path Pareto fronts.

## Phase 9 - Sleep phase anti-collapse

Current executable coverage:

- `FailureReplayBuffer` converts verifier regressions into replay examples with corrected oracle labels.
- `VerifiedSyntheticDataPool` refuses synthetic examples unless they carry origin, oracle, targeted skill, verification level, contamination risk, difficulty and confidence label.
- `RealExogenousReservoir` stores non-synthetic external examples separately from synthetic training data and accepts provenance metadata for exact source tracking.
- `ToolSolvedExampleFactory` creates tool/oracle-solved examples and rejects solver outputs that fail verification.
- `MetamorphicFamilyBuilder` creates metamorphic and anti-metamorphic families from existing `SkillSpec` hooks, with oracle-verified labels.
- `AntiCollapseFilter` rejects unlabeled synthetic data, high-contamination examples, duplicate prompts, calibration-gap increases and large-batch diversity collapse.
- `DiversityMetrics` tracks skill counts, origin counts, unique prompt ratio, skill entropy, origin entropy, rare-skill fraction and average contamination risk.
- `SkillConsolidationScheduler` prioritizes protected/fragile skills, applies an explicit rare-skill boost and schedules only examples accepted by the anti-collapse filter.
- `SleepPhaseConsolidator` orchestrates failure replay, tool-solved examples, metamorphic families, real/exogenous examples, anti-collapse filtering and scheduling from a `CycleReport`.
- The full LLM phase controller now populates the real/exogenous reservoir from decoded `observe_input_batch` token spans, verifies each span with the exact instruction-following oracle, persists `from_llm_input_batch` provenance, and rejects P9 consolidation if no real LLM span has reached the reservoir.
- `LocalExternalProvenanceAdapter` streams bounded `.txt` and `.jsonl` local source records into `RealExogenousReservoir`; records are deduplicated by stable content hash, verified through the declared Cortex oracle before acceptance, persisted with source path/line/chunk/hash provenance, and reported as accepted/rejected/skipped/duplicate rather than silently treated as generic replay.
- `TrainingConfig.cortex_external_provenance_paths` lets a real LLM training controller ingest configured external provenance before training starts; if paths are configured but no oracle-verified P9 real example is accepted, controller initialization fails instead of falling back to synthetic or internally fabricated "real" examples.
- `SleepPhaseReport` now records baseline, accepted and scheduled rare-skill fractions plus rare-skill gain, diversity delta and calibration-gap delta, making the Phase 9 success criteria directly inspectable.
- `write_cycle_run` can persist sleep phase reports into `summary.json`.
- `tools/run_cycle_report.py` writes Phase 9 sleep traces by default unless `--skip-sleep` is passed.
- The full LLM trainer tokenizes accepted sleep examples with the active BPE tokenizer and replays them as causal batches in the Cortex loss.
- `FrontierSkillDiscovery.compile_sleep_consolidation` now promotes accepted sleep examples by coherent consolidation family into held-out gated `BitLinear` micro-circuits with `training.source_kind="sleep_consolidation"` only when the global anti-collapse, diversity and calibration gates pass. The circuit training contract carries anti-collapse reasons/metrics, source origins, synthetic/real counts, max contamination risk and minimum verification level.
- The full LLM phase controller saves those sleep-promoted circuits into the persistent Frontier registry, binds them into P4 cognitive memory, immediately verifies a `CompiledFrontierAgent` FastSolve on the exact promoted circuit, adds verified P9 replay from the compiled answer, attaches the runtime FastSolve proof back onto the circuit payload, and feeds the circuit into P10 as a `compiled_frontier` proposal only with that proof present.
- `ProposalGenerator.from_sleep_frontier_circuits` requires the same sleep anti-collapse/diversity/calibration proof before emitting a P10 `compiled_frontier` proposal, and now also requires the attached FastSolve proof: compiled selection, oracle verification, output-goal acceptance, compiled-circuit checksum, held-out pass equality and faithful P4 memory binding id/fidelity.
- Architecture and deliverable audits now reject P9 if sleep consolidation only creates replay or only uses internally fabricated "real" tasks; they require nonzero `sleep_real_exogenous_llm_examples`, `sleep_real_exogenous_llm_batch_events`, `sleep_frontier_compiled_circuit_count`, held-out pass equality, P4 memory binding and `sleep_frontier_fastsolve_events`, and additionally require nonzero oracle-verified external provenance acceptance whenever external paths are configured.
- Cortex phase replay batches, including Phase 9 sleep/consolidation examples, future-contract ledger decisions, bounded ternary compression trace histories, cognitive-memory segments, compiled-circuit bindings, sleep pools and recursive-improvement archive summaries are now saved in checkpoints and restored on resume, so long runs do not lose P4/P9/P10 context, P2 instrumentation or P3 contract state after interruption.
- Resume validation now proves sleep/frontier circuits are not only present in reports: a restored compiled circuit is executed through FastSolve with restored P4 memory before training continues.
- `CompressionTraceLedger` keeps total P2 event counters and aggregate effective-cost inputs separately from retained detailed events; the LLM harness limits retained detailed traces with `cortex_trace_retention_limit` to prevent long GPU runs from growing trace memory without bound.

Evidence:

- `.\.venv\Scripts\python.exe -m unittest tests.test_sleep_phase`
- `.\.venv\Scripts\python.exe -m unittest tests.test_sleep_phase.SleepPhaseTest.test_sleep_phase_acceptance_compiles_to_heldout_frontier_circuit`
- `.\.venv\Scripts\python.exe -m unittest discover -s tests`
- Temporary artifact write with `tools\run_cycle_report.py --out-dir <temp> --run-id final-smoke` includes both `inference` and `sleep_phase`.
- `tests/test_llm_pretraining.py::LLMPretrainingHarnessTest::test_full_cortex_phase_controller_uses_all_modules_during_training` verifies nonzero sleep replay batches, replay updates, sleep-frontier circuits, held-out gates, FastSolve events, P10 sleep-frontier proposals and persistent registry entries during LLM training.
- `tests/test_llm_pretraining.py::LLMPretrainingHarnessTest::test_cortex_phase_state_survives_checkpoint_resume` verifies sleep-frontier reports and counts survive checkpoint/resume, executes restored FastSolve with restored P4 memory, and fails on a missing persisted registry when circuits are advertised.
- `tests/test_sleep_phase.py::SleepPhaseTest::test_local_external_provenance_adapter_streams_deduplicates_and_oracle_verifies` verifies local `.txt`/`.jsonl` provenance ingestion, hash deduplication, oracle rejection and external certificates.
- `tests/test_llm_pretraining.py::LLMPretrainingHarnessTest::test_phase_controller_ingests_configured_external_provenance_into_p9_reservoir` verifies configured external provenance is ingested into the P9 reservoir by the real LLM phase controller before training.

Remaining:

- Extend external provenance adapters beyond local files to streamed object stores or curated remote corpora once long/network-heavy validation is authorized.
- Measure rare-skill retention, diversity, calibration and sleep-promoted circuit reuse over repeated sleep/wake cycles.

## Phase 10 - Recursive improvement gate

Current executable coverage:

- `ProposalGenerator` converts cycle actions and regressions into typed proposals for tests, compression, router changes, MTP heads, regrowth strategies and new skill/test families; it also converts accepted compiled Frontier repairs into `compiled_frontier` proposals with source/frontier task lineage.
- `ProposalGenerator.evolve_from_archive` creates semantically compatible child proposals from accepted archive records, preserves parent ids, records the parent kind/action, and applies diversity pressure by choosing underrepresented proposal kinds from the persistent archive counts.
- `ImprovementProposal` carries affected skills, expected quality/cost/robustness deltas, risk, diversity tags and patch payload metadata.
- `SandboxTrainer` first applies proposals as in-memory sandbox agents; it records no touched repo files and creates rollback tokens for verifier-gated evaluation.
- `ProposalPatchedAgent` simulates repair, protected-skill degradation, reward-hacking behavior and calibration regression for verifier-gated evaluation.
- `DynamicEvaluator` compares baseline vs sandbox agents on main suites and anti/metamorphic robustness suites, then measures quality delta, cost delta, robustness delta, calibration delta, protected losses and cross-skill collapse flags.
- `RewardHackingDetector` flags declared overfitting, robustness-suite collapse, overconfident failures on affected skills, declared cost-accounting manipulation and zero-effective-cost passed non-empty answers on main or robustness suites.
- `DiversityPreserver` prevents one proposal kind from dominating the evolutionary archive.
- `PatchAcceptanceGate` requires Pareto improvement, no protected-skill regression, no calibration regression, no reward hacking and no diversity/collapse failure.
- `EvolutionaryArchive` records accepted and rejected decisions with proposal lineage, full sandbox/evaluation payloads and kind counts; it can save/load the complete archive as `archive.json` instead of only restoring counters.
- `RollbackSystem` records rollback events from accepted proposal tokens and can save/load them as `rollback.json`.
- Persistent archive restore is strict: accepted/rejected records must carry full baseline/trial/robustness verifier reports, deltas, reward/collapse/protected flags, explicit rollback tokens and matching archive/rollback schema; missing reports or an accepted archive without `rollback.json` are hard errors instead of fallback scores.
- `RecursiveImprovementEngine` orchestrates proposal generation, prioritized external proposals, bounded multi-generation archive evolution, sandbox training, dynamic evaluation, acceptance and archive recording from a `CycleReport`; it also writes a persistent manifest for the archive/rollback pair.
- `RecursiveImprovementReport` now carries per-generation summaries with proposal ids, accepted/rejected ids, evolved proposal counts and archive kind counts before/after each generation.
- `write_cycle_run` can persist recursive improvement reports into `summary.json`.
- `tools/run_cycle_report.py` writes Phase 10 traces by default unless `--skip-improvement` is passed.
- The full LLM Cortex phase controller converts verifier-approved recursive-improvement gate decisions into causal replay examples, feeds accepted P7 compiled Frontier repairs into P10 before generic proposals, runs bounded multi-generation P10 evolution by default through `TrainingConfig.cortex_phase_improvement_generations`, applies accepted proposals as signed bounded patches to real Transformer parameters, persists patch id, rollback token, parameter deltas, repair-loss improvement, protected-loss non-regression, proposal payload and the P10 anti-reward-hacking decision proof in checkpoints, and loads/saves reusable P10 archives through `TrainingConfig.cortex_improvement_archive_dir` even when a later run does not resume the checkpoint.
- Accepted P10 proposals now also materialize verified recursive-improvement artifacts: the controller resolves the source regression/frontier task, obtains the generic reference answer for non-Frontier proposals or the verified FastSolve answer for `compiled_frontier`, re-verifies it through the oracle, appends it to P10 replay as `ExampleOrigin.TOOL_SOLVED`, stores signed patch id, rollback token, proposal kind/payload and artifact id, exposes `recursive_verified_artifact_count` in reports/audits, and persists the artifacts through checkpoint resume.
- Accepted P10 model patches now persist executable rollback artifacts under `recursive_improvement_archive/model_patch_rollbacks/*.pt`: each artifact stores pre-patch tensors, pre/post parameter checksums, signed patch id, rollback token and parameter metadata. `rollback_recursive_model_patch` refuses mismatched post-patch weights, restores the saved tensors, requantizes the ternary core, records a persistent `rollback.json` event and exposes `recursive_model_rollback_artifact_count` / `recursive_model_executable_rollback_count` through reports, audits and checkpoint state.
- Accepted P10 model patch reports are no longer allowed to stand alone from the gate that authorized them: `signed_patch_id` is derived from the patch plus the accepted decision proof, and `recursive_model_applications` must expose empty reward-hacking/collapse flags, robustness score/delta, calibration delta, protected losses and non-empty decision reason before architecture/deliverable audits accept the patch.
- `ProposalGenerator.from_frontier_repairs` now refuses incomplete P7/Frontier repair payloads before they become P10 `compiled_frontier` proposals: accepted repairs must carry compiled selection, compiled verification, held-out gate with all held-out cases passed, accepted output-goal contract, verified compiled-circuit checksum and faithful P4 memory binding id/fidelity.
- Accepted P10 `compiled_frontier` patches are now grounded in the executable FastSolve contract before any parameter update: `_recursive_patch_training_contract` finds the covered task, executes `CompiledFrontierAgent` with mandatory P4 memory binding, verifies the output through the oracle, requires output-goal acceptance, a verified `compiled_circuit` P5 certificate, held-out gate pass and memory fidelity, then signs these fields into the model patch and replay artifact. Architecture and P10 deliverable audits reject a `compiled_frontier` model application without `training_contract_source=compiled_frontier_fastsolve`.
- Persistent P10 archives are now model-materialized before reuse: the LLM controller saves `archive.json` only after the accepted patch has changed real Transformer weights, written a rollback artifact and materialized a verified P10 replay artifact; persisted accepted records are filtered to materialized proposals, while sandbox-only acceptances stay visible only as `unmaterialized_sandbox_accepted_proposal_ids` in `manifest.json`. A fresh controller refuses to load an accepted persistent archive unless `model_materialization_required=true`, every persisted accepted proposal has a `signed_patch_id`, verified replay artifact and executable rollback, and the rollback artifact file still exists.
- Accepted P10 model patches use the same phase-balanced protected replay gate as P7: before archive materialization, the signed patch must protect the available replay surface across phases, expose `recursive_model_protected_replay_phase_count`, and pass audits requiring at least six protected phases in the integrated controller path.

Evidence:

- `.\.venv\Scripts\python.exe -m unittest tests.test_recursive_improvement`
- `tests.test_recursive_improvement.RecursiveImprovementTest.test_engine_evolves_accepted_proposals_across_generations` verifies that generation 1 creates a child proposal from a generation 0 accepted parent and records diversity-pressure archive counts.
- `.\.venv\Scripts\python.exe -m unittest discover -s tests`
- `tests.test_recursive_improvement.RecursiveImprovementTest.test_engine_prioritizes_accepted_frontier_repair_proposals` verifies that a Frontier repair becomes the first P10 proposal and is accepted under the normal gates.
- `tests.test_recursive_improvement.RecursiveImprovementTest.test_frontier_repair_proposals_require_complete_compiled_contract` verifies that the old incomplete `accepted=True`/`frontier_compiled_verified=True` payload now creates no P10 proposal.
- `tests.test_recursive_improvement.RecursiveImprovementTest.test_sleep_frontier_proposals_require_anti_collapse_proof` now verifies that sleep-promoted circuits without attached FastSolve/P4/P5/output-goal proof create no P10 proposal, while fully proven circuits carry that proof into the patch payload.
- `tests.test_recursive_improvement.RecursiveImprovementTest.test_persistent_archive_round_trips_full_decisions_and_rollbacks` verifies full accepted/rejected decisions, evaluation reports and rollback tokens round-trip through persistent archive files.
- `tests.test_recursive_improvement.RecursiveImprovementTest.test_persistent_archive_rejects_missing_full_evaluation_reports` and `test_persistent_archive_rejects_missing_rollback_file_for_accepted_records` verify P10 refuses incomplete persistent evidence.
- `tests.test_recursive_improvement.RecursiveImprovementTest.test_gate_rejects_cost_accounting_manipulation` verifies that an otherwise Pareto-looking sandbox repair is rejected when it hides runtime cost through zero-cost passed answers.
- `tests/test_llm_pretraining.py::LLMPretrainingHarnessTest::test_cortex_phase_state_survives_checkpoint_resume` verifies that P1-P10 replay state plus P2/P3 internal ledgers persist through a checkpoint resume and keep influencing optimizer steps, including multi-generation recursive-improvement events and evolved child proposals.
- The same LLM resume test now also verifies that a fresh independent `CortexTrainingPhaseController` with a different run directory reloads the shared P10 archive from `cortex_improvement_archive_dir` without using the checkpoint.
- `tests.test_llm_pretraining.LLMPretrainingHarnessTest.test_full_cortex_phase_controller_uses_all_modules_during_training` verifies the applied recursive model patch has `proposal_kind == "compiled_frontier"`, carries the Frontier repair payload, and records at least two recursive-improvement generations with evolved proposal events.
- `tests.test_llm_pretraining.LLMPretrainingHarnessTest.test_full_cortex_phase_controller_uses_all_modules_during_training` now also verifies that accepted P10 proposals produce verified recursive artifacts with matching proposal kind, signed patch id, replay example id, verification level and positive repair-loss delta.
- `tests.test_llm_pretraining.LLMPretrainingHarnessTest.test_full_cortex_phase_controller_uses_all_modules_during_training` now verifies that the P10 `compiled_frontier` model patch and replay artifact carry the FastSolve/P4/P5 training contract: compiled selection, oracle verification, output-goal pass, compiled-circuit checksum, held-out pass equality and memory-binding id/fidelity.
- `tests.test_llm_pretraining.LLMPretrainingHarnessTest.test_cortex_phase_state_survives_checkpoint_resume` now verifies that recursive verified artifacts are saved in `cortex_phase_state`, summarized in checkpoint sidecars and survive resume.
- `tests.test_llm_pretraining.LLMPretrainingHarnessTest.test_recursive_model_patch_has_executable_weight_rollback` verifies a P10 patch changes real Transformer parameters, carries the anti-reward-hacking decision proof, writes a rollback artifact, then restores every updated parameter exactly from that artifact.
- `tests.test_llm_pretraining.LLMPretrainingHarnessTest.test_persistent_p10_archive_rejects_unmaterialized_accepted_decision` verifies a sandbox-only accepted archive is rejected by a fresh LLM controller, and `test_full_cortex_phase_controller_uses_all_modules_during_training` / `test_cortex_phase_state_survives_checkpoint_resume` verify the materialized archive manifest in the integrated P1-P10 path.
- `tests.test_certificates.CertificatesTest.test_compiled_circuit_certificate_binds_contract_and_lineage` verifies the compiled-circuit certificate tool accepts valid lineage and rejects a tampered contract.
- Smoke: `RecursiveImprovementEngine(...).run(..., max_proposals=3)` accepted Pareto-improving sandbox proposals with no touched files.
- Temporary artifact write with `tools\run_cycle_report.py --out-dir <temp> --run-id final-smoke` includes `recursive_improvement` with accepted sandbox proposals and rollback data.

Remaining:

- Scale multi-generation proposal evolution over long shared archives, larger proposal budgets, real corpora and repeated wake/sleep cycles.

## Frontier Skill Discovery

Current executable coverage:

- `cortex3_frontier.FrontierSkillDiscovery` selects protected/fragile skills from the `SkillLedger`.
- It expands real cycle failures through `CompressionAdversary` to produce frontier tasks just beyond the current weak area.
- A slow/reference solver answers candidates, and only oracle-verified tasks are admitted.
- `FrontierInvariantSet` extracts expected types, metadata keys, anchor kinds and prompt obligations.
- Verified frontier tasks are distilled into `MicroTrainingExample` records with text plus structured non-label task features and compiled with `CortexMicroTrainer` into an instrumented `BitLinear` micro-circuit.
- The compiled circuit is re-evaluated by `DynamicSkillVerifier`; if a separate held-out metamorphic gate fails, the failing held-out tasks become verified support for the next bounded recompilation round and a fresh held-out suite is generated before promotion.
- Reports include DSV score, held-out pass counts/rate/gate status, training deltas, active/total weights and packed compiled weight bits.
- `FrontierCircuitRegistry` now keeps DSV-passing compiled circuits as runtime artifacts instead of dropping the trained model after report generation.
- `CompiledFrontierAgent` selects a compiled circuit by a lexicographic specialization key over covered task ids, group ids, numeric signatures, anchors and non-label metadata before using DSV/held-out/cost tie-breakers, not by skill name or additive global score alone; if a circuit is selected, verifier failure is exposed on the answer rather than hidden behind a fallback.
- `CompiledFrontierAgent` establishes and verifies a P4 `CompiledCircuitMemoryBinding` before using a selected circuit when shared memory is provided; if binding establishment or fidelity fails, FastSolve raises instead of silently falling back.
- Frontier registries persist to `frontier_registry.json` plus per-circuit micro-model checkpoints and reload through `CheckpointManager`, so a compiled skill can survive process boundaries.
- `UltraFastInferenceEngine` accepts a `compiled_frontier_registry` and uses a selected compiled frontier circuit as the answer source before fast/normal/careful route execution.
- `FrontierSkillDiscovery` distills both source regressions slow-solved by the reference solver and their frontier variants, so a compiled circuit can repair the original failing task rather than only nearby adversarial variants.
- P5 `compiled_circuit` certificates now require held-out task lineage and a passing held-out gate in addition to source/frontier lineage, DSV, output-goal, memory-binding claims and answer checksum.
- `CortexTrainingPhaseController` now runs a bounded Frontier Skill Discovery pass during phase audit, persists the registry under the run directory, routes a covered P8 task through the compiled FastSolve circuit, evaluates the same registry as a P7 repair candidate before parameter regrowth, requires held-out gate pass, Frontier output-goal contract and compiled-circuit certificate before accepting that repair, feeds accepted compiled repairs into P10 as prioritized `compiled_frontier` proposals, and reports `frontier_compiled_*`, `frontier_heldout_*`, `frontier_output_goal_*`, `frontier_repair_*` and `recursive_frontier_proposal_events` fields in `cortex_phase_report.json`.
- Checkpoint restore reuses that persisted registry immediately and records `frontier_registry_loaded_events`, `frontier_restored_fastsolve_events` and `compiled_circuit_memory_restored_reuse_events`; the audit requires restored FastSolve evidence whenever a registry is loaded.
- P10 resume hardening: after checkpoint restore, the LLM controller searches a small archive-aware proposal budget so recursive improvement does not fail only because the first restored-context proposal is rejected by strict gates.
- `write_cycle_run` persists frontier discovery reports under `summary.json["frontier_discovery"]`.
- `tools/run_cycle_report.py` writes Frontier Skill Discovery by default unless `--skip-frontier` is passed.

Evidence:

- `.\.venv\Scripts\python.exe -m unittest tests.test_frontier_discovery`
- Smoke: fragile-skill frontier tasks are slow-solved, verified, distilled and compiled into a DSV-passing micro-circuit.
- Short runtime proof: `python -m unittest tests.test_frontier_discovery` now asserts a compiled frontier circuit has nonzero held-out tasks, passes the held-out gate, is registered, selected by coverage, wins exact-coverage selection against an artificially high-scoring generic competing circuit, is used by `CompiledFrontierAgent`, oracle-verified, saved to `frontier_registry.json`, reloaded with held-out tasks, used again, and consumed by `UltraFastInferenceEngine` on a forced fast path.
- LLM controller proof: `tests.test_llm_pretraining.LLMPretrainingHarnessTest.test_full_cortex_phase_controller_uses_all_modules_during_training` asserts nonzero `frontier_compiled_circuit_count`, `frontier_compiled_skill_count`, all compiled circuits passing `frontier_heldout_*`, nonzero `frontier_compiled_fastsolve_events`, accepted P7 `frontier_repair_*` evidence carrying held-out proof, nonzero P10 `recursive_frontier_proposal_events`, a P10 model patch whose `proposal_kind` is `compiled_frontier`, and an on-disk `frontier_registry.json`; `test_cortex_phase_state_survives_checkpoint_resume` now also proves restored Frontier FastSolve and restored P4 memory reuse before continuing from checkpoint.

Remaining:

- Scale the held-out frontier gate and lexicographic multi-circuit selection beyond the current short metamorphic suites.
- Expand certificate tools beyond the current linear algebra and local rich-code domains.

## Cross-phase final objective and metrics

Current executable coverage:

- `cortex3_objective.FINAL_LOSS_TERMS` enumerates every term from the plan's `L_total`: behavior, multi-horizon, future contract, distillation behavior, distillation uncertainty, latent certificate, invariance, temporal consistency, total cognitive description, no cost shifting, hardware layout, skill regression, calibration, anchor fidelity, regrowth efficiency, verifier resistance and recursive improvement validity.
- `ObjectiveWeights` exposes the plan coefficients alpha through omega, including `lambda` as `lambda_` in Python.
- `EffectiveJouleModel` converts `CostTrace` into effective joules.
- `build_objective_report` computes weighted loss terms from cycle, fault, inference, future-contract and recursive-improvement evidence; `L_recursive_improvement_validity` now treats protected losses, reward hacking, calibration regression, collapse flags and diversity failures as per-decision invalidity.
- `ABSOLUTE_METRICS` enumerates all 15 metrics from the plan: cost per verified answer, joules per correct skill, active bits per preserved skill, rare regression rate, verifier detection rate, verifier false negatives, average verification cost, MTP rejection rate, token inflation, anchor accuracy, calibration, regrowth gain per added bit, path speed, percent without heavy verification and compiled skills from slow to fast.
- `write_cycle_run` can persist objective reports into `summary.json`.
- `tools/run_cycle_report.py` writes the objective report by default.
- `CortexTrainingPhaseController` now converts the latest cross-phase `L_total` into a bounded objective-feedback scale that multiplies the trainable Cortex LLM confidence regularization and verified phase-replay losses.
- LLM checkpoints persist objective feedback counters, latest objective loss, all ordered `17/17` loss-term names, per-term raw/coefficient/weighted values, per-term cumulative weighted totals, feedback scale summary and history inside `cortex_phase_state`, so resumed runs keep the same cross-phase training signal.
- The full Cortex LLM architecture audit now has a required `final_objective_loss` component: proof gates fail unless the LLM feedback path has consumed every term in `FINAL_LOSS_TERMS`, not just a scalar objective total.

Evidence:

- `.\.venv\Scripts\python.exe -m unittest tests.test_objective_metrics`
- Smoke: objective report contains `17/17` loss terms and `15/15` absolute metrics.
- `tests/test_llm_pretraining.py::LLMPretrainingHarnessTest::test_full_cortex_phase_controller_uses_all_modules_during_training` verifies objective-feedback events, latest `L_total`, feedback scale, ordered `17/17` term coverage and persisted report history during full P1-P10 LLM training.
- `tests/test_llm_pretraining.py::LLMPretrainingHarnessTest::test_cortex_phase_state_survives_checkpoint_resume` verifies objective-feedback state and all final-loss terms persist through checkpoint resume and sidecar summaries.

Remaining:

- Calibrate objective-feedback and term weights against real training runs.
- Compare objective-guided checkpoint selection policies across broad benchmarks.

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
- `TokenizedCorpusBuilder` streams encoded chunks once into a `uint32` token file and writes a coherent `manifest.json`.
- `MemmapCausalDataset` samples causal next-token targets and multi-horizon future targets directly from the memmap with vectorized batch window reads.
- `CortexTransformerLM` is a complete causal Transformer with tied embeddings, causal self-attention, MLP blocks, optional Cortex multi-horizon heads, an optional differentiable Variable-In compressor, an optional learned exact/latent/drop memory policy, an optional packed int2 `BitLinear` ternary core for the Cortex model with native CUDA kernel audit, a trainable Skill-Ledger-conditioned MoE path whose expert activations and routing context are recorded, and an optional latent certificate head.
- `CortexObjective` optimizes next-token loss plus Cortex MTP, temporal-consistency, confidence, Variable-In compression-cost, learned-memory policy, Skill-Ledger expert routing and certificate-head terms when the Cortex heads are enabled.
- `CortexTrainingPhaseController` integrates P1-P10 into full LLM training when horizons are `[1, 2, 4, 8]`, Variable-In, learned memory policy, skill-aware experts and the certificate head are enabled: verifier cycle, packed ternary forward traces, Variable-In KV/compression traces, exact-anchor observations decoded from real LLM input batches, learned exact/latent/drop memory decisions, Skill Ledger -> expert context routing, replay skill-context routing, MTP/FSP contract ledger, output-goal contract ledger, cumulative Bit/Skill/Causal/Uncertainty ledgers, cognitive memory reconstruction, certificate verification, causal attribution, minimal regrowth planning, fast/normal/careful inference, sleep replay batches and recursive-improvement gates.
- The full Cortex trainer adds confidence/contract regularization to the loss, tokenizes accepted sleep/phase examples into causal replay batches, tracks replay examples by originating phase including P9 sleep, scales Cortex trainable losses with bounded cross-phase objective feedback and writes `cortex_phase_report.json` with per-phase event counts.
- Cortex checkpoints persist and restore the phase controller's replay state including per-batch `replay_phase_ids`, objective feedback state, future-contract ledger, P8 inference future/output-goal ledger, retained ternary compression trace ledger, exact-input-anchor counters, P7 rollback artifacts, Bit/Skill/Causal/Uncertainty ledgers, cognitive memory, compiled Frontier registry path, sleep pools and recursive-improvement archive summaries, so interrupted full-architecture training keeps the same P2-P4/P7/P8/P9/P10 and ledger audit context instead of resetting those modules while keeping trace memory bounded.
- `build_training_plan` writes `run_plan.json` before training starts, with real token-count, split-window, parameter-count, planned-token, checkpoint and optimizer-memory estimates for the baseline and Cortex models.
- `LLMTrainer` supports checkpoints, strict resume, first-run-safe auto-resume, optimizer/scaler/RNG state persistence, gradient accumulation, CSV learning curves, resource usage monitoring, deterministic random sampling, explicit device selection, mixed precision policy and DDP initialization from environment, including a Windows/Gloo TCPStore path that avoids unsupported libuv builds.
- Long LLM runs persist live resource monitoring to `resource_usage_live.json` at checkpoints and `resource_usage_summary.json` at shutdown/finalization, including CPU total/process averages, RSS memory and CUDA utilization/memory averages when `nvidia-smi` is available.
- `audit_learning_curves` writes `learning_curve_audit.json` and makes the proof gate require real finite baseline/Cortex validation curves with initial and final validation steps.
- `PrecisionPolicy(require_cuda=True)` raises when CUDA is required but unavailable, preventing silent CPU fallback.
- `llm_doctor_report` and `tools/train_llm.py doctor` audit Python dependencies, CUDA availability, requested precision, `torch.distributed`, Gloo/NCCL readiness and write a persistent `doctor_report.json`.
- `LLMComparisonRunner` trains a baseline next-token Transformer and a Cortex multi-horizon Transformer on the same corpus/cache, then writes `run_plan.json`, `learning_curve_audit.json`, `comparison_report.json`, `report.md`, `learning_curve.png`, both final checkpoints and both learning-curve CSV files.
- `LLMComparisonMatrixSuite` prepares one shared tokenizer/memmap for an arbitrary corpus, repeats the baseline-vs-Cortex comparison over multiple seeds and writes `comparison_matrix_report.json`, `comparison_matrix_report.md`, `comparison_matrix_ratios.png` and aggregate validation learning curves.
- `LLMCorpusMatrixSuite` repeats the comparison matrix across multiple named corpora, persists per-corpus reports and writes `corpus_matrix_report.json`, `corpus_matrix_report.md`, `corpus_matrix_ratios.png` and aggregate multi-corpus learning curves with corpus-level, seed-level and sample-level proof metrics.
- `LLMExperimentRunner` executes a manifest-driven full experiment: doctor audit, HF/path corpus preparation, cross-corpus matrix training/proof and final `experiment_report.json`/Markdown artifacts.
- `LLMBenchmarkSuite` runs multiple deterministic domains, persists per-domain comparison artifacts and writes an aggregate `benchmark_report.json`, `benchmark_report.md` and `benchmark_ratios.png`.
- `LLMStatisticalBenchmarkSuite` repeats the benchmark over multiple seeds, persists each seed/domain comparison and writes `statistical_benchmark_report.json`, `statistical_benchmark_report.md` and `statistical_benchmark_ratios.png` with mean, median, min ratio, win-rate, per-domain and per-seed aggregates.
- `tools/train_llm.py` exposes `smoke`, `prepare-hf` and `compare` commands for local proof runs, Hugging Face corpus preparation and larger text-shard corpora.
- `tools/train_llm.py compare-matrix` exposes the arbitrary-corpus multi-seed proof gate while reusing one shared tokenized corpus.
- `tools/train_llm.py corpus-matrix` exposes the multi-corpus x multi-seed proof gate for prepared corpus suites.
- `tools/train_llm.py preflight-experiment` runs doctor plus static model/batch/GPU memory capacity checks before expensive HF export or training.
- `tools/train_llm.py run-experiment` executes a normalized JSON manifest for reproducible large-corpus GPU/DDP experiments.
- `tools/train_llm.py audit-experiment` audits completed experiment directories after long runs, checking proof gates, doctor status, HF shards, tokenized manifests, CSV/PNG learning curves and non-empty baseline/Cortex checkpoints.
- `tools/train_llm.py benchmark` exposes the multi-domain proof gate and supports CPU `bf16` validation.
- `tools/train_llm.py benchmark-matrix` exposes the multi-domain x multi-seed proof gate and fails `--require-win` unless every seed-domain sample wins with a nonzero baseline and bounded next-token regression.
- Comparison, matrix and benchmark proof gates now support `min_corpus_tokens` and `min_planned_train_tokens`; when set, `--require-win` rejects favorable Cortex/baseline ratios produced on undersized corpora or insufficient planned training volume.
- `tools/launch_llm_ddp.py` launches true local multi-process DDP workers, pins the Gloo interface and writes per-rank logs.
- `.github/workflows/ci.yml` runs the LLM smoke command.
- `prepare-hf --resume` reuses only complete HF export reports and tokenized manifests whose tokenization config and preparation recipe still match; missing shards, incomplete tokenized directories and changed tokenizer args fail loudly instead of deleting or rebuilding long-running corpus preparation work.
- Tokenized corpus manifests now carry SHA-256 hashes for the token memmap, tokenizer and source shards plus the preparation recipe (`vocab_size`, `min_frequency`, `seq_len`, max horizon, train split and chunking); `run_plan.json`, training reports and checkpoints include a corpus identity digest, and `LLMTrainer` refuses resume from missing or mismatched `corpus_identity` checkpoints.
- Single comparison reports now require `baseline_score >= min_baseline_future_tokens_per_cost`; a Cortex ratio computed against a zero baseline is recorded with `failed_checks=["baseline_score_passed"]` and cannot pass `--require-win`. Scale gates similarly record `corpus_scale_passed` and `planned_train_tokens_passed`.
- Single comparison reports now also require `cortex_phase_integration_passed` when the model claims full Cortex mode (`use_cortex_heads`, `use_ternary_core`, horizons `[1, 2, 4, 8]`), so a full-architecture run with missing P1-P10 traces cannot pass.

Evidence:

- Local GPU environment after dependency correction: NVIDIA GeForce RTX 5070, driver CUDA `13.2`, `torch==2.11.0+cu128`, `torch.version.cuda==12.8`, `cuda_available=True`, `cuda_device_count=1`, `distributed_available=True`, `gloo_available=True`, `nccl_available=False` on Windows.
- CUDA dependency correction: the previous environment had `torch==2.12.1+cpu` despite a visible RTX 5070. Installed the official CUDA wheel with `pip install --force-reinstall torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128`; `requirements-cuda-cu128.txt` records the reproducible install command and now includes `cupy-cuda12x`/`ml_dtypes` for the native ternary CUDA kernel.
- Native ternary kernel validation: `tools\benchmark_ternary_kernel.py --native-backend extension --batch 128 --in-features 256 --out-features 256 --dtype fp16 --kernel-variant auto --autotune-warmup 1 --autotune-repeat 2 --warmup 5 --repeat 20` passed on RTX 5070 with extension v2, selected `warp_reduction_int2`, max error `0.000976`, packed weight compression `8x`, native `0.0350 ms`, full `BitLinear` forward `0.0976 ms`, PyTorch unpack+linear `0.3853 ms`, full forward speedup vs legacy native+STE dense `3.53x`, full forward+backward `0.6101 ms` vs dense STE legacy `0.9699 ms` (`1.59x`), `native_extension_grad_weight_dispatches=12`, and requantize/pack `0.1484 ms` vs PyTorch `0.5784 ms` (`3.90x`). A strict sustained matrix `tools\benchmark_ternary_kernel.py --matrix --dtype fp16 --kernel-variant auto --autotune-warmup 1 --autotune-repeat 2 --warmup 2 --repeat 6 --sustain-seconds 0.35 --sustain-op forward_backward --sustain-sync-every 4 --resource-interval 0.05 --min-resource-samples 2` passed on shapes `64x128x128`, `128x256x256`, `256x512x512` with `strict_extension_only=true`, `resource_samples_passed=true`, min sample count `4`, min forward+backward speedup `1.02x`, average speedup `1.41x`, average `nvidia-smi` GPU utilization `21.83%`, average GPU power draw `40.21 W`, and average process CPU share `25.28%`. Extension v5 provides WMMA fp16 `grad_input` and WMMA fp16->fp32 `grad_weight` for aligned and padded-edge shapes: the aligned large-shape matrix `256x768x768` + `512x1024x1024` passed with speedup average `1.83x`, min `1.75x`, `gradInputCounts={"wmma_fp16":471/213}`, `gradWeightCounts={"wmma_fp16_float":471/213}`, average GPU utilization `42.58%`; the edge matrix `255x769x771` + `511x1025x1027` passed with speedup average `1.85x`, min `1.26x`, `gradInputCounts={"wmma_fp16_padded":328/138}`, `gradWeightCounts={"wmma_fp16_float_padded":328/138}`, average GPU utilization `33.5%`, and no dense fallback. Extension v6 adds WMMA bf16 for aligned and padded-edge shapes: the strict BF16 matrix `256x768x768` + `255x769x771` passed with speedup average `1.39x`, min `1.20x`, `gradInputCounts={"wmma_bf16":300}` then `{"wmma_bf16_padded":282}`, `gradWeightCounts={"wmma_bf16_float":300}` then `{"wmma_bf16_float_padded":282}`, average GPU utilization `9.1%`, average GPU power draw `36.07 W`, and no dense fallback. Extension v7 adds WMMA forward packed matmul for fp16/bf16: pre-v7 fp16 `512x1024x1024` had `native_vs_unpack=0.32x`; post-v7 fp16 `256x768x768` + `512x1024x1024` selected `wmma_tensor_core_int2`, reached `native_vs_unpack=2.91x/1.21x`, and passed strict matrix speedup min `1.21x`; post-v7 bf16 `256x768x768` + `255x769x771` selected `wmma_tensor_core_int2`, reached `native_vs_unpack=7.57x/3.38x`, and passed strict matrix speedup min `1.42x`.
- CUDA C++ extension toolchain audit: `tools\train_llm.py doctor --require-cuda --require-cuda-extension --precision bf16 --device cuda` passed with `native_rawkernel_available=true`, `torch.version.cuda=12.8`, `nvcc=12.8` from `C:\Users\hight\.codex\cuda-12.8\Library`, and Visual Studio Build Tools 2022 selected. `tools\run_cuda128_minimal_nvcc_smoke.ps1` compiled a minimal CUDA object; `tools\run_cuda128_extension_smoke.ps1` compiled and executed a real PyTorch CUDA extension returning `[1.0, 2.0, 3.0, 4.0]`. The debug path also proved VS2026/MSVC 14.51 is incompatible with CUDA 12.8 here because even a minimal `.cu` crashes `cudafe++`.
- Doctor validation: `tools\train_llm.py doctor --out-dir runs\llm-doctor-cuda-validation --require-cuda --precision bf16 --device cuda` passed with CUDA visible and bf16 resolving on `cuda`.
- `.\.venv\Scripts\python.exe tools\train_llm.py smoke --out-dir runs\llm-smoke-dev-48 --steps 48 --require-win`
- Smoke proof: baseline score `0.022321`, Cortex score `0.145833`, Cortex/baseline `6.533x`, next-token-loss regression ratio `1.020`, proof passed.
- CUDA smoke validation: `tools\train_llm.py smoke --out-dir runs\llm-cuda-smoke-validation --steps 48 --precision bf16 --device cuda --require-cuda --require-win` passed on RTX 5070 with baseline score `0.029576`, Cortex score `0.147135`, Cortex/baseline `4.975x`, next-token-loss regression ratio `1.017305`.
- CUDA resume debug/fix: after installing the CUDA wheel, checkpoint resume exposed `TypeError: RNG state must be a torch.ByteTensor` when restoring CUDA RNG state loaded with `map_location=cuda`. `LLMTrainer.load_checkpoint` now normalizes saved CUDA RNG states back to CPU `uint8` tensors before `torch.cuda.set_rng_state_all`; the targeted resume test passes on CUDA.
- Auto-resume manifest validation: the manifest experiment test now runs with `training.resume_if_exists=true`, reruns with `datasets.load_dataset` patched to fail, reuses the HF export report, resumes from `checkpoint_final.pt` and records `start_step=48`, `optimizer_steps=0`.
- CUDA external Wikitext comparison matrix: `tools\train_llm.py compare-matrix runs\hf-wikitext2-validation\text_shards --out-dir runs\llm-wikitext2-cuda-compare-matrix-validation --seeds 17,29 --vocab-size 512 --seq-len 64 --d-model 64 --n-heads 4 --n-layers 2 --steps 48 --batch-size 8 --precision bf16 --device cuda --require-cuda --require-win` passed with `2/2` seeds, mean Cortex/baseline ratio `24.864x`, min ratio `24.500x`, aggregate CSV/PNG learning curves and CUDA recorded in per-seed reports.
- Versioned experiment manifests: `experiments/wikitext_cuda_validation.json` for fast local CUDA validation with small scale thresholds and `experiments/c4_cuda_large_manifest.json` for a preflighted, auto-resumable large C4 CUDA run with massive corpus/training-token proof thresholds.
- Versioned Wikitext CUDA manifest validation: `tools\train_llm.py run-experiment experiments\wikitext_cuda_validation.json` passed with `2/2` seeds, win-rate `1.0`, mean Cortex/baseline ratio `11.861x`, min ratio `10.889x`, CUDA doctor passed and aggregate CSV/PNG learning curves written.
- Full Cortex phase integration unit validation: `.\.venv\Scripts\python.exe -m pytest tests\test_llm_pretraining.py::LLMPretrainingHarnessTest::test_full_cortex_phase_controller_uses_all_modules_during_training -q` passed and verified P1-P10 event counts, ternary forward events, future contract decisions, confidence regularization, sleep replay batches, replay updates and objective-feedback scaling.
- Cortex phase checkpoint-resume validation: `tests/test_llm_pretraining.py::LLMPretrainingHarnessTest::test_cortex_phase_state_survives_checkpoint_resume` verifies replay batches, objective feedback, future-contract decisions, ternary layer-forward trace history, cognitive-memory state, sleep pools and recursive-improvement archive summaries survive checkpoint reload.
- Recursive-improvement archive validation: `tests/test_recursive_improvement.py::RecursiveImprovementTest::test_restored_archive_kind_counts_still_guard_diversity` verifies restored P10 archive kind counts still guard against proposal-kind domination after resume.
- Ternary trace retention validation: `tests/test_reporting_and_ternary.py::ReportingAndTernaryTest::test_compression_trace_ledger_retains_tail_with_total_counters` verifies bounded retained detail with preserved total counters and aggregate activation-bit cost.
- Full Cortex proof-gate negative validation: `tests/test_llm_pretraining.py::LLMPretrainingHarnessTest::test_comparison_proof_requires_full_cortex_phase_report_for_full_architecture` verifies that a full-Cortex config fails proof when `all_phases_active=false`.
- LLM pretraining test suite after full phase integration: `.\.venv\Scripts\python.exe -m pytest tests\test_llm_pretraining.py -q` passed with `32` tests.
- Wikitext CUDA scale-gate validation: `tools\train_llm.py run-experiment experiments\wikitext_cuda_validation.json --out-dir runs\cortex3-wikitext-cuda-scale-gate-validation` passed with `2/2` seeds, min observed corpus tokens `29,104`, min observed planned train tokens `24,576`, min required corpus/train tokens `20,000/20,000`, mean Cortex/baseline ratio `10.257x`, min ratio `9.625x`, preflight artifact written and aggregate CSV/PNG learning curves written.
- Post-run audit validation: `tools\train_llm.py audit-experiment runs\cortex3-wikitext-cuda-scale-gate-validation` passed with no failed checks and revalidated preflight, proof, HF shards, tokenized corpus manifests, learning curves and checkpoints.
- C4 large manifest preflight after full phase/input-anchor integration and RTX 5070 batch tuning: `tools\train_llm.py preflight-experiment experiments\c4_cuda_large_manifest.json --out-dir runs\cortex3-c4-cuda-large-preflight-batch4-retained-ckpt5-current` passed with `batch_size=4`, `gradient_accumulation_steps=16`, `checkpoint_interval=5`, `max_intermediate_checkpoints=5`, estimated Cortex peak `4,768,726,245` bytes under `10,897,408,000` usable CUDA bytes, while preserving `2,097,152,000` planned train tokens, `65,536` tokens per optimizer step and `50,000,000` minimum corpus tokens.
- Negative scale-gate CLI validation: a `tools\train_llm.py smoke --require-win --min-corpus-tokens 999999999 --min-planned-train-tokens 999999999` run failed as expected and exposed both `corpus_scale_passed` and `planned_train_tokens_passed` diagnostics.
- C4 external validation: `tools\train_llm.py prepare-hf --dataset allenai/c4 --config-name en --split train --text-field text --out-dir runs\hf-c4-mini-validation --max-documents 40 --min-text-chars 64 --shard-chars 8192 --vocab-size 512 --min-frequency 1 --seq-len 64 --max-horizon 4` exported 40 real C4 documents, 72,020 characters, 10 shards and a 32,537-token memmap.
- C4 CUDA comparison matrix: `tools\train_llm.py compare-matrix runs\hf-c4-mini-validation\text_shards --out-dir runs\llm-c4-mini-cuda-compare-matrix-validation --seeds 17,29 --vocab-size 512 --seq-len 64 --d-model 64 --n-heads 4 --n-layers 2 --steps 48 --batch-size 8 --precision bf16 --device cuda --require-cuda --require-win` passed with `2/2` seeds, win-rate `1.0`, mean Cortex/baseline ratio `19.480x`, min ratio `16.681x`, max next-token-loss regression `1.096379`.
- C4 training-plan proof: both C4 CUDA seed runs wrote `run_plan.json` with token count `32,537`, `512` tokens per optimizer step, `24,576` planned train tokens, `136,576` baseline parameters and `269,761` Cortex parameters.
- C4 learning-curve audit validation: `tools\train_llm.py compare-matrix runs\hf-c4-mini-validation\text_shards --out-dir runs\llm-c4-mini-cuda-learning-curve-audit-validation --seeds 17,29 --vocab-size 512 --seq-len 64 --d-model 64 --n-heads 4 --n-layers 2 --steps 48 --batch-size 8 --precision bf16 --device cuda --require-cuda --require-win` passed with `2/2` seeds, mean Cortex/baseline ratio `33.128x`; both seed reports wrote `learning_curve_audit.json` with `13` finite validation points for baseline and Cortex and `learning_curve_audit_passed=True`.
- `.\.venv\Scripts\python.exe tools\train_llm.py benchmark --out-dir runs\llm-benchmark-validation --domains sequence,anchors --repeats 96 --steps 48 --batch-size 8 --precision bf16 --require-win`
- Benchmark proof through `codex-test` with `gradient_accumulation_steps=2`: `2/2` domains passed, mean Cortex/baseline ratio `32.097x`, minimum domain ratio `25.861x`, mean baseline score `0.005301`, max next-token-loss regression ratio `1.001049`.
- `.\.venv\Scripts\python.exe tools\train_llm.py benchmark-matrix --out-dir runs\llm-benchmark-matrix-validation --domains sequence,anchors --seeds 11,23 --repeats 96 --steps 48 --batch-size 8 --precision bf16 --require-win`
- Statistical benchmark proof through `codex-test`: `4/4` seed-domain samples passed, win-rate `1.0`, mean Cortex/baseline ratio `26.520x`, median ratio `18.829x`, minimum ratio `4.840x`, mean baseline score `0.012835`, max next-token-loss regression ratio `1.067812`.
- DDP root cause and fix: the local Windows/Gloo path needs explicit TCPStore `use_libuv=False`; when Gloo auto-selected a bad host route it tried `kubernetes.docker.internal`. Cortex now pins `GLOO_SOCKET_IFNAME` and uses an explicit `TCPStore(..., use_libuv=False)` for local Gloo env initialization.
- `.\.venv\Scripts\python.exe tools\launch_llm_ddp.py --nproc 2 --master-port 29752 --gloo-interface Ethernet --timeout 240 -- smoke --out-dir runs\llm-ddp-smoke-validation --steps 48 --precision bf16 --require-win`
- DDP smoke proof through `codex-test`: `world_size=2`, `distributed=True`, proof passed, baseline score `0.002790`, Cortex score `0.149740`, Cortex/baseline `53.667x`, next-token-loss regression ratio `0.952`.
- Autosize multi-seed candidate measurement proof: `tools\train_llm.py profile-autosize --out-dir runs\llm-profile-autosize-multiseed-v1 --overwrite --device cuda --require-cuda --precision bf16 --steps 1 --candidate-seq-lens 32 --candidate-d-models 64 --candidate-n-layers 2 --candidate-batch-sizes 4 --candidate-gradient-accumulation-steps 1,2 --n-heads 4 --selected-shape-count 1 --min-selected-shapes 1 --seeds 71,73 --memory-budget-fraction 0.10 --measure-candidate-count 1 --measure-candidate-seed-count 2 --measured-selection-metric throughput_gpu --min-cases 2 --require-multi-seed --min-train-tokens-per-second-mean 10 --min-gpu-utilization-percent-mean 5 --min-gpu-memory-used-mb-mean 900 --min-gpu-power-draw-watts-mean 30 --resource-interval 0.05 --min-resource-samples 2 --corpus-repeats 128 --max-corpus-tokens 4096` passed with candidate `seq32_d64_h4_l2_b4_g2`, 2/2 measured candidate profiles passing, 2/2 matrix cases passing, `strict_extension_only_cases=2`, `all_phases_active_cases=2`, throughput mean `161.879` tokens/s, GPU mean `13.202%`, power mean `40.732 W`, VRAM mean `978.536 MB`, measured score `1234.037`, and observed budget fraction max `0.888`.
- Autosize diverse candidate measurement proof: `tools\train_llm.py profile-autosize --out-dir runs\llm-profile-autosize-diverse-v1 --overwrite --device cuda --require-cuda --precision bf16 --steps 1 --candidate-seq-lens 32,64 --candidate-d-models 64,96 --candidate-n-layers 2 --candidate-batch-sizes 4 --candidate-gradient-accumulation-steps 1,2 --n-heads 4 --selected-shape-count 1 --min-selected-shapes 1 --seeds 71 --memory-budget-fraction 0.10 --measure-candidate-count 2 --measure-candidate-strategy diverse --measured-selection-metric throughput_gpu --min-cases 1 --min-train-tokens-per-second-mean 10 --min-gpu-utilization-percent-mean 5 --min-gpu-memory-used-mb-mean 900 --min-gpu-power-draw-watts-mean 30 --resource-interval 0.05 --min-resource-samples 2 --corpus-repeats 128 --max-corpus-tokens 4096` passed with measured estimated ranks `1,8`, selected `seq32_d64_h4_l2_b4_g1`, 2/2 measured candidate profiles passing, 1/1 matrix case passing, `strict_extension_only_cases=1`, `all_phases_active_cases=1`, throughput mean `93.958` tokens/s, GPU mean `15.182%`, power mean `39.636 W`, VRAM mean `978.182 MB`, and selected measured score `1330.507`.
- Autosize adaptive candidate measurement proof: `tools\train_llm.py profile-autosize --out-dir runs\llm-profile-autosize-adaptive-v1 --overwrite --device cuda --require-cuda --precision bf16 --steps 1 --candidate-seq-lens 32,64 --candidate-d-models 64,96 --candidate-n-layers 2 --candidate-batch-sizes 4 --candidate-gradient-accumulation-steps 1,2 --n-heads 4 --selected-shape-count 1 --min-selected-shapes 1 --seeds 71 --memory-budget-fraction 0.10 --measure-candidate-count 4 --measure-candidate-strategy diverse --measure-candidate-adaptive-rounds 2 --measured-selection-metric throughput_gpu --min-cases 1 --min-train-tokens-per-second-mean 10 --min-gpu-utilization-percent-mean 5 --min-gpu-memory-used-mb-mean 900 --min-gpu-power-draw-watts-mean 30 --resource-interval 0.05 --min-resource-samples 2 --corpus-repeats 128 --max-corpus-tokens 4096` passed with measurement rounds `initial_diverse` and `adaptive_measured_frontier`, estimated ranks `[1,8]` then `[2,5]`, selected `seq64_d64_h4_l2_b4_g2`, 4/4 measured candidate profiles passing, 1/1 matrix case passing, `strict_extension_only_cases=1`, `all_phases_active_cases=1`, throughput mean `323.963` tokens/s, GPU mean `13.000%`, power mean `38.969 W`, VRAM mean `983.000 MB`, and selected measured score `3486.477`.
- Autosize robust measured-score proof: `test_llm_batch_profile_autosize_uses_risk_adjusted_measured_score` covers two measured candidates across two seeds where the unstable candidate has higher raw mean throughput but lower `mean - stddev` score; the harness selects the stable candidate and persists `measured_score_mean`, `measured_score_stddev`, `measured_score_lower_confidence`, and `measured_score_stability_ratio`.
- Autosize minimum measurement-seed proof: `test_llm_batch_profile_autosize_synthesizes_minimum_measurement_seed` covers the default single-provided-seed case and proves candidate measurement expands `(11,)` into `(11, 104740)` while the final matrix stays on seed `11`. A short CUDA validation with `--seeds 71` passed with `measurement_seeds=[71,104800]`, `synthesized_measurement_seed_count=1`, 8 measured candidate profiles, selected `seq64_d64_h4_l2_b4_g2`, robust score `3970.683`, mean `4376.482`, stddev `405.799`, matrix throughput `285.624` tokens/s, GPU mean `13.143%`, power `41.534 W`, VRAM `982.857 MB`, `strict_extension_only_cases=1`, and `all_phases_active_cases=1`.
- Autosize adaptive upper-confidence proof: `test_llm_batch_profile_autosize_adaptive_frontier_uses_uncertainty_potential` covers the exploration helper directly: a stable source has a stronger lower-confidence score, an uncertain source has a much stronger `mean + stddev`, and the next adaptive candidate is selected near the uncertain source with source-score fields persisted. A short CUDA validation with `--seeds 71` passed with `measurement_seeds=[71,104800]`, 8 measured candidate profiles, adaptive source `seq64_d96_h4_l2_b4_g2`, source upper confidence `4247.725`, selected `seq64_d64_h4_l2_b4_g2`, robust score `3906.117`, mean `4648.800`, stddev `742.683`, upper `5391.483`, matrix throughput `296.592` tokens/s, GPU mean `15.071%`, power `41.461 W`, VRAM `982.857 MB`, `strict_extension_only_cases=1`, and `all_phases_active_cases=1`.
- Autosize uncertainty-refinement proof: `test_llm_batch_profile_autosize_refines_uncertain_candidate_with_extra_seed` covers the new default refinement budget: after two measurement seeds, an uncertain high-upside candidate loses to a stable candidate by lower-confidence score; the harness then spends seed `104742` only on that uncertain candidate, recomputes mean/stddev/lower/upper confidence, and the final measured selection changes to the now-validated uncertain shape while reporting `refinement_rounds`, `refinement_profile_count`, `refinement_seeds` and before/after scores. `test_llm_batch_profile_autosize_refines_decision_frontier_candidate_first` covers the stronger decision-frontier ranking and now verifies the refinement profile uses `steps=2`, repeats the refinement profile twice, persists `measurement_profile_seeds=(11,13,104742,104742)`, `measurement_steps=(1,1,2,2)`, `measurement_repeat_indices=(0,0,0,1)`, and averages repeated profiles into one `measured_score_observation_values` entry before robust scoring. A short CUDA validation with default refinement passed with `measurement_seeds=[71,104800]`, `refinement_seeds=[209529]`, `refine_uncertain_step_multiplier=2`, `refine_uncertain_repeat_count=2`, `refinement_profile_count=2`, 10 measured candidate profiles, refined shape `seq64_d96_h4_l2_b4_g2`, raw profile scores `[1188.464,4215.040,8851.871,8260.925]`, grouped observations `[1188.464,4215.040,8556.398]`, selected `seq64_d64_h4_l2_b4_g2`, robust score `4044.156`, mean `4345.331`, stddev `301.176`, matrix throughput `295.203` tokens/s, GPU mean `15.571%`, power `41.639 W`, VRAM `983.429 MB`, `strict_extension_only_cases=1`, and `all_phases_active_cases=1`.
- Autosize finalist/frontier-confirmation proof: `test_llm_batch_profile_autosize_blocks_unresolved_decision_after_winner_switch` covers the iterative confirmation loop: a candidate that wins on the initial two seeds is re-profiled on fresh seed `104742` with `steps=2` and two runtime repeats under `confirm_seed_104742_steps_2_repeat_0/1`; when that confirmation shows instability, the robust score drops, selection switches to the stable candidate, and the loop confirms that new winner on seed `209471`, then C45 blocks the still-unresolved decision margin before matrix launch when the dedicated resolution budget is explicitly disabled. `test_llm_batch_profile_autosize_blocks_unresolved_confirmed_decision_frontier` covers the other C45 gate: a non-selected challenger is confirmed, `confirmation_pending_shape_keys=()`, but the decision margin remains non-positive, so the report fails with `selected_confirmation_decision_unresolved`, exposes `confirmation_decision_resolved=false`, and skips the matrix. `test_llm_batch_profile_autosize_uses_dedicated_budget_to_resolve_confirmed_margin` covers C46/C47/C48: with `confirm_selected_max_rounds=2`, normal rounds confirm only the winner and challenger, then the dedicated default `confirm_selected_decision_resolution_extra_rounds=3` profiles both in `decision_margin_resolution`, resolves the margin, and allows the matrix. `test_llm_batch_profile_autosize_default_confirmation_rounds_cover_measured_frontier` covers the C44/C45 positive path: when `confirm_selected_max_rounds` is not explicit, the effective budget expands to the measured frontier size, confirms the winner plus two upper-confidence challengers with seeds `104742,209471,314200`, leaves `confirmation_pending_shape_keys=()`, and requires `confirmation_decision_resolved=true`. C48 makes `confirm_selected_max_rounds` a strict frontier-confirmation budget; `decision_margin_resolution` can now only be emitted by `confirm_selected_decision_resolution_extra_rounds`. A broad short CUDA validation before C47 consumed 20 measured profiles and 10 confirmation profiles across 4 confirmation rounds including 1 `decision_margin_resolution`, then correctly failed before matrix with `selected_confirmation_decision_unresolved`, `confirmation_complete=true`, `confirmation_decision_resolved=false`, and `confirmation_decision_margin=-6263.952`. The C48 smaller CUDA validation passed with `confirm_selected_max_rounds=1`, `confirm_selected_decision_resolution_extra_rounds=1`, `confirmation_seeds=[209529]`, `confirmation_complete=true`, `confirmation_decision_resolved=true`, `confirmation_decision_margin=643.177`, `measured_candidate_profile_count=4`, `confirmation_profile_count=2`, selected `seq64_d64_h4_l2_b4_g1`, matrix throughput `164.790` tokens/s, GPU mean `14.308%`, power `39.105 W`, VRAM `981.538 MB`, `strict_extension_only_cases=1`, and `all_phases_active_cases=1`.
- Autosize variance-adaptive margin-resolution proof: `test_llm_batch_profile_autosize_adapts_margin_resolution_to_residual_variance` covers C49. With `confirm_selected_decision_resolution_extra_rounds=1`, the confirmed winner/challenger intervals still overlap after the first resolution round, so the harness computes positive `confirmation_decision_resolution_margin_deficit`, `confirmation_decision_resolution_uncertainty` and `confirmation_decision_resolution_overlap_ratio`, adds one adaptive round under `confirm_selected_decision_resolution_adaptive_extra_rounds=2`, runs two `decision_margin_resolution` rounds with seeds `314200` and `418929`, resolves the margin, and launches the matrix. The short CUDA non-regression `llm-profile-autosize-adaptive-resolution-small-c49` also passed with adaptive cap `2`, adaptive rounds `0` because the margin was already resolved, matrix throughput `164.592` tokens/s, GPU mean `14.692%`, power `40.655 W`, VRAM `984.538 MB`, `strict_extension_only_cases=1`, and `all_phases_active_cases=1`.
- Autosize sequential margin-resolution proof: `test_llm_batch_profile_autosize_reevaluates_margin_resolution_after_each_round` covers C50. It forces a challenger whose first `decision_margin_resolution` round changes variance but does not resolve the margin, then proves the controller re-evaluates overlap and spends two adaptive rounds with seeds `418929` and `523658`. The report exposes `confirmation_decision_resolution_budget_evaluations`, `confirmation_decision_resolution_stop_reason=decision_resolved`, `confirm_selected_decision_resolution_total_rounds=3`, and only then launches the matrix. The short CUDA non-regression `llm-profile-autosize-sequential-resolution-small-c50` passed with adaptive cap `2`, adaptive rounds `0` because the margin was already resolved, `confirmation_decision_resolution_stop_reason=decision_resolved`, matrix throughput `166.762` tokens/s, GPU mean `13.538%`, power `41.242 W`, VRAM `984.385 MB`, `strict_extension_only_cases=1`, and `all_phases_active_cases=1`.
- Autosize uncertainty-adaptive confirmation-runtime proof: the autosize confirmation tests now cover C51. Confirmation rounds compute `confirmation_runtime_signal` from interval width, uncertainty ratio, margin deficit and overlap, then scale `confirmation_steps` and `confirmation_repeat_count` up to `confirm_selected_runtime_step_multiplier_cap=4` and `confirm_selected_runtime_repeat_count_cap=4`. Unstable finalist, challenger and `decision_margin_resolution` rounds are verified with `confirmation_adaptive_runtime_applied=true`, `confirmation_steps=4`, `confirmation_repeat_count=3`, repeated `measurement_profile_seeds`, and `confirmation_runtime_escalation_count`. The short CUDA non-regression `llm-profile-autosize-runtime-escalation-small-c51` passed with one runtime escalation, `confirmation_runtime_signal=0.925`, 5 measured candidate profiles, 3 confirmation profiles, matrix throughput `166.266` tokens/s, GPU mean `15.077%`, power `39.701 W`, VRAM `982.000 MB`, `strict_extension_only_cases=1`, and `all_phases_active_cases=1`.
- Autosize expected-gain-per-cost refinement proof: `test_llm_batch_profile_autosize_refinement_uses_expected_gain_per_cost_frontier` covers C52. It forces a flashy but expensive high-upper-confidence candidate against a cheaper decision-frontier candidate and proves refinement spends seed `104742` on the better `gain_per_cost` action while exposing `refinement_budget_strategy`, `refinement_budget_actions`, `refinement_expected_gain`, `refinement_posterior_utility`, `refinement_measurement_cost_tokens`, and `refinement_gain_per_cost`. The short CUDA non-regression `llm-profile-autosize-efficient-frontier-small-c52` passed with `refinement_budget_strategy=expected_gain_per_cost`, one budget action, refined shape `seq64_d64_h4_l2_b4_g1`, `gain_per_cost=2.570`, 6 measured candidate profiles, matrix throughput `88.083` tokens/s, GPU mean `14.727%`, power `39.513 W`, VRAM `978.364 MB`, `strict_extension_only_cases=1`, and `all_phases_active_cases=1`.
- Autosize refinement-frontier audit proof: the same C52 test now covers C53 by asserting `refinement_budget_candidate_actions` contains both the selected cheap action and the rejected expensive action, with the expensive action showing higher raw `expected_gain` but lower `gain_per_cost`. Reports expose `refinement_budget_candidate_action_count`, `refinement_budget_candidate_actions`, and per-action `selected_for_refinement` in both the round and global measurement metadata. The short CUDA metadata validation `llm-profile-autosize-efficient-frontier-small-c53` passed with `refinement_budget_candidate_action_count=2`, selected action `seq64_d64_h4_l2_b4_g1` at `gain_per_cost=2.947`, rejected action `seq32_d64_h4_l2_b4_g1` at `gain_per_cost=1.887`, matrix throughput `90.486` tokens/s, GPU mean `12.364%`, power `39.020 W`, VRAM `981.727 MB`, `strict_extension_only_cases=1`, and `all_phases_active_cases=1`.
- Autosize refinement-frontier report-cap proof: `test_llm_batch_profile_autosize_caps_refinement_candidate_action_report` covers C54. It forces four uncertain budget actions while setting `refinement_budget_candidate_action_report_cap=2`; the harness still selects from the full frontier, writes only two candidate actions, marks `refinement_budget_candidate_actions_truncated=true`, and exposes `refinement_budget_candidate_action_total_count=4` so large-grid reports stay bounded without hiding the decision scope. The short CUDA cap validation `llm-profile-autosize-capped-frontier-small-c54` passed with report cap `1`, total candidate actions `2`, published actions `1`, truncation `true`, published action `seq64_d64_h4_l2_b4_g1` at `gain_per_cost=3.584`, matrix throughput `75.069` tokens/s, GPU mean `14.833%`, power `44.263 W`, VRAM `955.333 MB`, `strict_extension_only_cases=1`, and `all_phases_active_cases=1`.
- Autosize refinement-frontier representative audit proof: the C54 test now covers C55 by proving a capped report keeps the selected refinement action with `report_selection_reason=selected_for_refinement` and spends another slot on a rejected `top_expected_gain` action, so the report is bounded but still shows a high-upside action that was not worth profiling by cost. The short CUDA representative-frontier validation `llm-profile-autosize-representative-frontier-small-c55` passed with published reasons `selected_for_refinement` then `top_expected_gain`, matrix throughput `86.468` tokens/s, GPU mean `19.417%`, power `44.492 W`, VRAM `1110.833 MB`, `strict_extension_only_cases=1`, and `all_phases_active_cases=1`; an intentionally tighter first run at `memory_budget_fraction=0.10` failed on `observed_gpu_memory_budget`, keeping the VRAM gate strict.
- `prepare-hf` is covered with a local Hugging Face JSONL dataset path that exports 30 documents into multiple shards, writes `hf_export_report.json`, trains a BPE tokenizer and builds a causal memmap manifest. Resume coverage verifies that a completed export does not reload the dataset, missing shards are rejected, an existing token memmap is reused unchanged, and changed tokenizer arguments are rejected.
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
- Checkpoint resume unit coverage: a Cortex trainer runs two optimizer steps with `gradient_accumulation_steps=2`, writes step/final checkpoints carrying `corpus_identity`, resumes from `checkpoint_final.pt` to step 4 and preserves curve plus RNG state in the checkpoint payload. Separate tests rebuild a different tokenized corpus, and change a shared comparison-matrix preparation recipe, verifying that resume is rejected on identity and tokenization-config mismatch.
- CLI resume validation: `tools\train_llm.py smoke --out-dir runs\llm-resume-cli-validation --steps 2 ...` then `--steps 4 --resume ...` resumed both `baseline_ntp` and `cortex3` from `checkpoint_final.pt` with `start_step=2`, `optimizer_steps=2`, `effective_batch_size=16` and `final_step=4`.
- DDP accumulation validation: `tools\launch_llm_ddp.py --nproc 2 ... --gradient-accumulation-steps 2` completed with `distributed=True`, `world_size=2`, proof passed and `effective_batch_size=32` for both baseline and Cortex.
- DDP CUDA preflight validation: `tools\launch_llm_ddp.py --nproc 2 ... --device cuda --require-cuda` now fails before spawning workers because one visible CUDA device cannot serve two local CUDA ranks; CPU/Gloo DDP still passed after the CUDA wheel install.
- `.\.venv\Scripts\python.exe -m unittest discover -s tests`: `138` tests passed.

Remaining:

- Run a genuine long large-corpus experiment from an external Hugging Face dataset such as C4/FineWeb, not only the deterministic local smoke corpus or local JSONL export test.
- Validate NCCL multi-GPU runs on hardware exposing at least two CUDA devices; this Windows machine validates single-GPU CUDA bf16 and CPU/Gloo DDP, but `nccl_available=False` and only one CUDA device is visible.
- Scale model sizes and training steps, then publish the same statistical benchmark on broad external corpora instead of only deterministic local domains.
- Exercise executable P10 rollback on later-run protected-regression triggers and large shared archives, not only the short direct rollback test.
