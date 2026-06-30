# Cycle pipeline

The new cycle layer turns the plan into an executable evaluation loop.

```text
reference model
trial model
  -> dynamic verifier
  -> oracle registry
  -> regression list
  -> ledgers
  -> regression analysis
  -> budgeted actions
  -> extra generated checks
  -> verifier cost profile
  -> markdown + JSON run artifacts
  -> future-contract ledger when MTP/FSP decisions are present
  -> fast/normal/careful inference traces when inference results are present
```

Run:

```bash
python tools/run_cycle_report.py
```

This writes `runs/<run-id>/summary.json`, `runs/<run-id>/report.md`, and `runs/<run-id>/fault_matrix.json` unless `--no-write` is passed. Written runs include a Phase 8 inference smoke trace, a Phase 9 sleep anti-collapse trace, a Phase 10 recursive-improvement trace, final objective metrics, plan experiments A-E and a trained autoregressive checkpoint smoke by default; pass `--skip-inference`, `--skip-sleep`, `--skip-improvement`, `--skip-experiments` or `--skip-autoregressive` to omit optional traces.

Main files:

- `cortex3_phases.py`: phase registry.
- `cortex3_ledgers.py`: bit, skill, causal and uncertainty ledgers.
- `cortex3_analysis.py`: failure cause hints.
- `cortex3_cycle.py`: end-to-end cycle report.
- `cortex3_reporting.py`: persisted JSON/markdown run artifacts.
- `cortex3_ternary.py`: Phase 2 ternary compression instrumentation and PyTorch `BitLinear`.
- `cortex3_future.py`: Phase 3 MTP/FSP heads, standalone calibration, future contracts, temporal consistency and accept/reject gates.
- `cortex3_memory.py`: Phase 4 exact recent KV, latent old KV, exact anchors, query-conditioned reconstruction, memory-augmented answer recovery and anchor fidelity verification.
- `cortex3_certificates.py`: Phase 5 latent proof states, calibrated certificate head, proof-carrying generation, verifier, random de-latentization and tool-backed certificate checks.
- `cortex3_attribution.py`: Phase 6 counterfactual ablations, causal estimates and regression clustering.
- `cortex3_regrowth.py`: Phase 7 executable regrowth actions, gain/cost simulation, non-regression gates and re-crystallization annealing.
- `cortex3_inference.py`: Phase 8 fast/normal/careful inference routing, budget prediction, early exit, Mixture-of-Depths and ternary kernel dispatch.
- `cortex3_sleep.py`: Phase 9 failure replay, verified synthetic data, real/exogenous reservoir, anti-collapse filtering and consolidation scheduling.
- `cortex3_improvement.py`: Phase 10 proposal generation, in-memory sandbox training, dynamic evaluation, Pareto acceptance, evolutionary archive and rollback.
- `cortex3_objective.py`: final 17-term Cortex-3 loss and the 15 absolute metrics from the plan.
- `cortex3_experiments.py`: named plan experiments A-E with pass/fail criteria and metrics.
- `cortex3_microtrain.py`: trainable PyTorch micro-model, DSV agent wrapper, sleep/verifier datasets and checkpoint save/load.
- `cortex3_autoregressive.py`: trainable micro-autoregressive decoder with greedy generation, MTP multi-horizon losses, future-contract confidence margin, DSV agent wrapper and checkpoint save/load.
- `cortex3_llm.py`: full causal LLM pretraining bridge with BPE tokenizer training, streaming text corpus tokenization, memmap causal dataset, baseline Transformer, Cortex multi-horizon Transformer, AMP/DDP-aware trainer, checkpoints, learning curves and baseline-vs-Cortex proof report.
- `cortex3_selection.py`: offline trial selection.

Phase 1 verification now includes generated domains for arithmetic, algebra, long-context anchors, entity tracking, instruction following, executable code unit tests, and calibration. The `RegressionHarness` can inject the plan's first fault families and assert that the verifier separates reference success from trial regression. Per-skill reports now persist every verification case with real per-case verifier cost, `summary.json` carries a schema version, and `OracleQualityAuditor` probes false positives and false negatives across all default skills.

Phase 2 instrumentation now records sign+mask compression decisions, provisional/certified zero counts, residual synapse buffers, activation quantization, expert activations, KV modes, MTP/FSP events and real layer-forward events. `BitLinear` is a PyTorch layer with a straight-through runtime weight path, numerical parity coverage against `nn.Linear` when activation quantization is disabled, and gradient-survival coverage for both inputs and weights.

Phase 3 now adds MTP heads for horizons 1, 2, 4 and 8, a confidence head, temporal consistency loss, future contracts, contract revision and accept/reject gates. Standalone heads calibrate on verifier micro-tasks, and the autoregressive decoder can use accepted/rejected contracts for blockwise token generation while preserving DSV verification and confidence calibration. Contract ledgers, block traces and answer raw payloads can be persisted in run JSON artifacts.

Phase 4 now separates recent exact memory from older latent memory. Old segments are stored as compact Torch embeddings plus summaries and exact anchors; query-conditioned reconstruction restores relevant exact anchors through an anchor fidelity verifier. The inference engine can now turn a verified reconstruction into the final answer for long-context/entity tasks, replacing a weaker corrupted base answer only when the memory-derived answer has stronger confidence, and persists the base answer plus reconstruction proof in `raw`.

Phase 5 now lets an answer carry a short certificate with uncertainty, a latent proof checksum, optional anchors and a tool verification contract. A supervised calibrator trains the certificate head on verifier micro-tasks, then `ProofCarryingGenerator` emits answers plus short certificates and latent proof-state payloads. Random de-latentization probes latent states for auditability, calibrated `UNKNOWN` answers can keep high uncertainty without failing the certificate, and inference treats tampered proof-carrying certificates as a failed gate even when the answer text passes the oracle. Certificates and calibration metrics can be persisted in run JSON artifacts.

Phase 6 now executes counterfactual ablation probes over blocks, experts, KV mode, MTP horizon, activation precision, FSP contracts and routing. It also consumes real `BitLinear` layer-forward traces from inference to create layer-specific block-restoration and activation-precision probes. The resulting causal attribution reports include recovery deltas, gain per cost, normalized cause probabilities and regression clusters, and can be persisted in run JSON artifacts.

Phase 7 now converts causal attribution into executable repair patches. Candidate repairs are simulated against the failed task, checked against protected tasks for non-regression, ranked by gain per cost, then cooled through a re-crystallization schedule. The cycle-report tool now builds those plans from real cycle regressions by default and persists them in run JSON artifacts.

Phase 8 now runs an executable inference loop. A difficulty router selects fast, normal or careful paths; the selected route controls compression strength, depth, MTP horizon, latent KV use, expert activation and verifier strength. The loop executes PyTorch `BitLinear` layers through a Mixture-of-Depths core, records packed ternary kernel dispatches, applies early exit, drafts self-speculative future contracts, verifies certificates on stronger paths and persists inference results in run JSON artifacts. Fast paths can avoid runtime verifier cost, but the research score is still oracle-audited so confident wrong answers do not inflate verified capability. The same engine can now consume a trained micro-autoregressive decoder as its answer source, proving that generated answers, not only answer-class labels, flow through routing, verification, certificates and cost accounting. The comparison target is verified capability per effective cost, not token throughput alone.

Phase 9 now converts slow verified experience into a sleep consolidation plan. Failures become replay examples, tool-solved and metamorphic variants become synthetic examples only when they carry trust labels, real/exogenous examples stay in a separate reservoir, and an anti-collapse filter rejects unlabeled synthetic data, high-contamination samples, duplicates, diversity collapse and calibration loss. The scheduler then prioritizes fragile and rare skills using only examples accepted by the filter, and the report exposes baseline/accepted/scheduled rare-skill fractions, diversity delta and calibration-gap delta.

Phase 10 now proposes controlled improvements without self-modifying the repository. Proposals are generated from cycle actions and regressions, trained as in-memory sandbox agents, evaluated by the dynamic verifier against main and robustness suites, screened for Pareto improvement, protected-skill loss, calibration regression, reward hacking and diversity/cross-skill collapse, then recorded in an evolutionary archive with rollback tokens.

The final objective layer now materializes the plan's `L_total`. It computes all 17 loss terms, from behavior and multi-horizon loss through anchor fidelity, regrowth efficiency, verifier resistance and recursive-improvement validity. Recursive improvement invalidity now includes protected losses, reward hacking, calibration regression, collapse flags and diversity failures. The report also records the 15 absolute metrics including cost per verified answer, rare regression rate, MTP rejection rate, anchor accuracy, calibration, path speed and verified capability per effective joule.

The named experiment layer now runs the first five experiments demanded by the plan: injected-fault verifier detection, fixed vs metamorphic vs compression-adversary regression search, minimal regrowth vs global retraining cost, SlowSolve to FastSolve cost/quality preservation and sandboxed auto-improvement without reward hacking, protected-skill loss, calibration regression or collapse.

The frontier discovery layer now turns fragile ledger regions into compiled micro-circuits. It expands real failures with the compression adversary, slow-solves and oracle-verifies frontier tasks, extracts invariants, distills verified examples into a `BitLinear` micro-model, then re-runs the DSV over the compiled circuit before persisting the result in run artifacts.

The micro-training layer now closes two checkpoint loops. SkillSpec and sleep-phase examples can train either an answer-class micro-model or a character-level autoregressive decoder. Both wrap an instrumented `BitLinear` core, expose DSV-compatible agents, and save/reload `.pt` checkpoints; the autoregressive loop additionally optimizes behavior, MTP multi-horizon, confidence and future-contract margin losses. Generated answers now carry a compiled-circuit certificate with distilled invariants, compiled weight bits and cheap verification contract. Written cycle artifacts include an `autoregressive_checkpoint` section with training, DSV verification, generated samples, compiled-circuit metadata, block-contract metrics and a careful-path inference trace from the trained decoder.

The LLM pretraining bridge now moves beyond verifier micro-checkpoints. `tools/train_llm.py smoke --require-win` trains a real BPE tokenizer, writes corpus tokens to a `uint32` memmap, trains a next-token Transformer baseline and a Cortex multi-horizon Transformer on the same causal dataset, saves both final checkpoints, writes CSV learning curves, renders `learning_curve.png`, and emits `comparison_report.json` plus `report.md`. `tools/train_llm.py prepare-hf` streams a Hugging Face `datasets` source such as C4 or local JSONL into bounded or explicitly unbounded text shards, records an export report, trains the tokenizer and writes the token memmap/manifest. `tools/train_llm.py compare-matrix` applies the baseline-vs-Cortex comparison to several seeds on one shared tokenized corpus, so large prepared shards are tokenized once and each seed writes its own checkpoints and learning curves under `seed_<seed>` plus aggregate `comparison_matrix_learning_curves.csv/png`. `tools/train_llm.py corpus-matrix` repeats that comparison matrix across multiple named corpora and writes a strict cross-corpus aggregate proof, report, ratio plot and multi-corpus learning curves. `tools/train_llm.py run-experiment` is the manifest-driven orchestration path: it runs doctor, prepares HF/path corpora, executes corpus-matrix and writes normalized manifest plus final experiment report. Long runs support strict `--resume`, checkpoint intervals, optimizer/scaler/RNG state persistence and gradient accumulation for larger effective batches. The Cortex proof gate compares `verified_future_tokens_per_forward_cost` against the baseline while bounding next-token-loss regression. The same CLI accepts directories or text shards for larger corpora and exposes explicit `--require-cuda`, `--precision fp16/bf16`, `--distributed` and `--gloo-interface` flags so GPU/DDP runs fail loudly instead of silently falling back.

`tools/train_llm.py benchmark` extends the smoke into multiple deterministic domains such as sequence patterns, exact anchors, code-like text and Cortex reasoning text. Each domain gets its own corpus, tokenizer/cache, baseline run, Cortex run, checkpoints and learning curves; the suite then writes `benchmark_report.json`, `benchmark_report.md` and `benchmark_ratios.png` with a strict all-domain proof gate requiring nonzero baseline learning and a Cortex win. `tools/train_llm.py benchmark-matrix` repeats that suite across multiple seeds, persists every seed/domain comparison, writes a statistical aggregate report plus ratio plot, and only passes when every seed-domain sample clears the Cortex win and next-token regression gates. Local validation exercises CPU and CUDA `bf16` paths. `tools/launch_llm_ddp.py` validates true local multi-process DDP by exporting rank environment variables, pinning the Gloo interface, forcing the explicit non-libuv TCPStore path required by Windows/Gloo, and preflighting CUDA worker count against visible devices before spawning ranks.
