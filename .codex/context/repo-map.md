# Codex Repo Map

- Root: `C:\Users\hight\Documents\Codex\2026-06-30\cl\work\LLMTEST`
- Generated: 2026-06-30T09:04:38
- Files indexed: 49
- Symbols indexed: 555
- Symbol files scanned: 42 of 42 candidates
- Git: branch `main`, head `4a76aae`, changed entries `7`

## Use

Read this map before broad file exploration. Prefer targeted `rg` and narrow file reads after using the symbol index.

## Top-Level Shape

- `.github`: 1 files
- `cortex3.py`: 1 files
- `cortex3_analysis.py`: 1 files
- `cortex3_attribution.py`: 1 files
- `cortex3_autoregressive.py`: 1 files
- `cortex3_certificates.py`: 1 files
- `cortex3_cycle.py`: 1 files
- `cortex3_experiments.py`: 1 files
- `cortex3_frontier.py`: 1 files
- `cortex3_future.py`: 1 files
- `cortex3_improvement.py`: 1 files
- `cortex3_inference.py`: 1 files
- `cortex3_ledgers.py`: 1 files
- `cortex3_llm.py`: 1 files
- `cortex3_memory.py`: 1 files
- `cortex3_microtrain.py`: 1 files
- `cortex3_objective.py`: 1 files
- `cortex3_phases.py`: 1 files
- `cortex3_regrowth.py`: 1 files
- `cortex3_reporting.py`: 1 files
- `cortex3_selection.py`: 1 files
- `cortex3_sleep.py`: 1 files
- `cortex3_ternary.py`: 1 files
- `docs`: 3 files
- `LICENSE`: 1 files
- `pyproject.toml`: 1 files
- `README.md`: 1 files
- `tests`: 17 files
- `tools`: 3 files

## High-Signal Files

- `docs/IMPLEMENTATION_STATUS.md`
- `docs/PIPELINE.md`
- `docs/ROADMAP.md`
- `pyproject.toml`
- `README.md`

## Symbol Index

- `cortex3.py`: py:__init__, py:_result, py:adversarial, py:Anchor, py:anti_metamorphic, py:build_suite, py:CandidateAnswer, py:coerce, py:CostTrace, py:effective_cost, py:generate, py:merge, py:metamorphic, py:OracleRegistry, py:pass_rate, py:register, py:SkillReport, py:SkillSpec, py:Task, py:VerificationCaseResult, py:VerificationSuiteReport, py:verified_capability_per_cost, py:verify
- `cortex3_analysis.py`: py:analyze, py:CauseHint, py:FailureAnalysis, py:RegressionAnalyzer, py:top_cause
- `cortex3_attribution.py`: py:__init__, py:_block_specs, py:_confidence_for_skill, py:_counterfactual_answer, py:_expected_answer_for_failure, py:_future_contract_specs, py:_is_cause_applicable, py:_layer_forward_specs, py:AblationDimension, py:AblationProbeResult, py:AblationProbeSpec, py:AttributionBatchReport, py:CausalAttributionEngine, py:CausalAttributionReport, py:CauseEstimate, py:CounterfactualAblationRunner, py:gain_per_cost, py:meta, py:RegressionCluster, py:run, py:targeted_repair_is_cheaper, py:to_dict, py:top_cause
- `cortex3_autoregressive.py`: py:__init__, py:__post_init__, py:accepted_blocks, py:ar_examples_from_sleep_report, py:ar_examples_from_tasks, py:ARBlockContractTrace, py:ARConfig, py:ARContractGeneration, py:ARDataset, py:ARMicroDecoder, py:ARTrainingResult, py:bos_id, py:decode, py:encode, py:eos_id, py:from_texts, py:pad_id, py:rejected_blocks, py:tensors, py:to_dict, py:TokenVocabulary
- `cortex3_certificates.py`: py:__init__, py:canonical_payload, py:CertificateHead, py:CertificateHeadOutput, py:CertificateType, py:checksum, py:confidence, py:DelatentizationProbe, py:forward, py:from_dict, py:LatentProofState, py:probe, py:ProofCarryingAnswer, py:RandomDelatentizer, py:ShortCertificate, py:to_candidate_answer, py:to_dict, py:ToolVerification, py:ToolVerifierRegistry, py:verify_probe
- `cortex3_cycle.py`: py:__init__, py:CortexCycle, py:cycle_report_markdown, py:CycleReport, py:PathDecision, py:PathRouter, py:route, py:run, py:summary
- `cortex3_experiments.py`: py:__init__, py:CortexExperimentSuite, py:experiment_a_verifier_faults, py:experiment_b_compression_adversary, py:experiment_c_minimal_regrowth, py:experiment_d_slow_compile_fast, py:experiment_e_auto_improvement_sandbox, py:ExperimentResult, py:ExperimentSuiteReport, py:passed, py:run_all, py:to_dict
- `cortex3_frontier.py`: py:__init__, py:_compiled_weight_bits, py:_extract_invariants, py:_prompt_obligations, py:_verified_frontier_tasks, py:discover, py:FrontierCompiledCircuit, py:FrontierDiscoveryReport, py:FrontierInvariantSet, py:FrontierSkillDiscovery, py:to_dict
- `cortex3_future.py`: py:__init__, py:__post_init__, py:_batch, py:_loss, py:acceptance_rate, py:accepted, py:ContractDecision, py:evaluate, py:forward, py:future_examples_from_tasks, py:FutureContract, py:FutureContractLedger, py:FutureTrainingExample, py:MTPFSPCalibrationResult, py:MTPFSPCalibrator, py:MTPFSPConfig, py:MTPFSPHeads, py:MTPHeadOutput, py:record, py:rejected, py:temporal_consistency_loss, py:temporal_consistency_loss_from_outputs, py:to_dict, py:train
- `cortex3_improvement.py`: py:__call__, py:__init__, py:_calibration_gap, py:_collapse_flags, py:check, py:detect, py:DiversityPreserver, py:DynamicEvaluator, py:generate, py:ImprovementProposal, py:pareto_candidate, py:ProposalGenerator, py:ProposalKind, py:ProposalPatchedAgent, py:RewardHackingDetector, py:SandboxEvaluation, py:SandboxTrainer, py:SandboxTrial, py:to_dict, py:to_trial_proposal, py:train
- `cortex3_inference.py`: py:__init__, py:__post_init__, py:BudgetPrediction, py:BudgetPredictor, py:decide, py:DifficultyRouter, py:DifficultySignal, py:dispatch, py:EarlyExitDecision, py:EarlyExitPolicy, py:forward_route, py:InferenceConfig, py:InferencePath, py:InferenceResult, py:InferenceRoute, py:MixtureOfDepthsCore, py:passed, py:predict, py:route, py:signal, py:TernaryKernelDispatch, py:TernaryKernelDispatcher, py:threshold, py:to_dict
- `cortex3_ledgers.py`: py:add_certificate, py:BitLedger, py:CausalLedger, py:CausalTrace, py:expected_calibration_error, py:fragile_skills, py:get, py:ingest_cost, py:record, py:SkillLedger, py:SkillState, py:total_effective_bits, py:UncertaintyLedger, py:update_from_report
- `cortex3_llm.py`: py:__init__, py:__post_init__, py:_barrier_if_needed, py:_env_rank, py:_extract_text, py:_json_default, py:_load_dataset, py:_rank_zero, py:_write_json, py:export, py:from_paths, py:HFDatasetExportConfig, py:HFDatasetExportReport, py:HFDatasetTextExporter, py:iter_chunks, py:LLMTokenizer, py:load, py:open_next_shard, py:TextCorpusConfig, py:TextShardReader, py:to_dict, py:train
- `cortex3_memory.py`: py:__init__, py:__post_init__, py:_stable_hash_int, py:_summary_from_counts, py:AnchorFidelityResult, py:AnchorFidelityVerifier, py:CognitiveMemory, py:CognitiveMemoryConfig, py:compress_from_exact, py:compression_ratio, py:embed_text, py:embed_tokens, py:LatentKVStore, py:MemoryMode, py:MemoryReconstruction, py:MemorySegment, py:passed, py:push, py:RecentExactKV, py:rendered, py:rendered_context, py:retrieve, py:tokenize, py:verify
- `cortex3_microtrain.py`: py:__init__, py:__post_init__, py:_loss, py:answer_index, py:CortexMicroModel, py:CortexMicroTrainer, py:examples_from_sleep_report, py:examples_from_tasks, py:forward, py:from_examples, py:MicroDataset, py:MicroModelAgent, py:MicroModelConfig, py:MicroTrainingExample, py:MicroTrainingResult, py:MicroVocabulary, py:requantize_core, py:skill_index, py:tensors, py:to_dict, py:train
- `cortex3_objective.py`: py:_anchor_accuracy, py:_certificate_failure_rate, py:_clamp01, py:_contract_decisions_from_future, py:_contract_decisions_from_inference, py:_hardware_layout_loss, py:_mtp_rejection_rate, py:_regrowth_efficiency_loss, py:_safe_ratio, py:_sum_cost, py:_temporal_loss, py:_verifier_detection_rate, py:AbsoluteMetricsReport, py:coefficient_for, py:CortexObjectiveReport, py:EffectiveJouleModel, py:estimate, py:FinalLossReport, py:LossTermValue, py:ObjectiveWeights, py:to_dict
- `cortex3_phases.py`: py:Phase, py:phase_table
- `cortex3_regrowth.py`: py:__call__, py:__init__, py:answer_for, py:applies_to, py:as_legacy_action, py:build, py:change_sign, py:ExecutableRegrowthAction, py:from_attribution, py:gain_per_cost, py:increase_scale_precision_bits, py:NonRegressionResult, py:RecrystallizationStep, py:RegrowthActionKind, py:RegrowthActionSpace, py:RegrowthPatch, py:RegrowthPatchBuilder, py:RegrowthPlan, py:RegrowthSimulationResult, py:selected_action, py:TargetedRepairAgent, py:to_dict, py:unzero_block
- `cortex3_reporting.py`: py:_actions_to_dict, py:_analysis_to_dict, py:_bit_ledger_to_dict, py:_case_to_dict, py:_cost_to_dict, py:_default_run_id, py:_skill_ledger_to_dict, py:_skill_report_to_dict, py:_write_json, py:cycle_report_to_dict, py:fault_matrix_to_dict, py:RunArtifacts, py:suite_report_to_dict, py:write_cycle_run
- `cortex3_selection.py`: py:decide, py:FrontierSelector, py:select, py:SelectionDecision, py:TrialProposal, py:TrialSelector
- `cortex3_sleep.py`: py:__init__, py:_expected_answer, py:_normalized_prompt, py:_task_difficulty, py:add, py:add_failure, py:add_many, py:by_skill, py:ExampleOrigin, py:FailureReplayBuffer, py:has_trust_label, py:MetamorphicFamilyBuilder, py:RealExogenousReservoir, py:solve, py:to_dict, py:ToolSolvedExampleFactory, py:TrainingExample, py:usable_synthetic_label, py:VerifiedSyntheticDataPool
- `cortex3_ternary.py`: py:activation_bits, py:ActivationQuantization, py:CompressionDecision, py:CompressionTraceLedger, py:cost_trace, py:ExpertActivation, py:explain_failure, py:KVModeEvent, py:l1, py:LayerForwardEvent, py:make_compression_decision, py:MTPFSPEvent, py:quantize_activation_values, py:record_activation, py:record_compression, py:record_expert, py:record_kv, py:record_layer_forward, py:record_mtp_fsp, py:ResidualSynapseBuffer, py:restore, py:store, py:to_dict, py:torch_available, py:zero_count
- `tests/test_autoregressive_decoder.py`: py:_verifier, py:AutoregressiveDecoderTest, py:test_autoregressive_generation_improves_and_verifies_all_seed_tasks, py:test_checkpoint_save_and_load_preserves_generated_answers, py:test_cycle_report_persists_autoregressive_checkpoint_report, py:test_future_contract_blocks_preserve_verified_autoregressive_generation, py:test_multi_horizon_and_future_contract_losses_are_optimized, py:test_sleep_phase_examples_can_feed_autoregressive_training
- `tests/test_causal_attribution.py`: py:_verifier, py:CausalAttributionTest, py:test_activation_precision_probe_recovers_arithmetic_failure, py:test_batch_clustering_groups_regressions_by_top_cause_and_skill, py:test_cycle_run_artifacts_can_include_causal_attribution, py:test_fsp_contract_probe_recovers_instruction_format_failure, py:test_kv_mode_probe_recovers_anchor_failure, py:test_real_forward_trace_drives_layer_block_and_activation_probes
- `tests/test_certificates.py`: py:_latent_state, py:CertificatesTest, py:test_anchor_certificate_requires_exact_anchor_presence, py:test_arithmetic_certificate_verifies_with_tool_and_checksum, py:test_certificate_efficiency_requires_quality_and_calibration, py:test_certificate_head_calibration_produces_verified_proof_carrying_answers, py:test_certificate_head_outputs_latent_state_answer_cert_type_and_uncertainty, py:test_code_certificate_runs_real_unit_tests, py:test_cycle_run_artifacts_can_include_short_certificates, py:test_default_registry_contains_required_tools, py:test_proof_carrying_answer_maps_to_candidate_answer, py:test_random_delatentization_is_deterministic_and_detects_tampering, py:test_tampered_latent_state_fails_certificate_verification
- `tests/test_cognitive_memory.py`: py:CognitiveMemoryTest, py:test_anchor_fidelity_verifier_fails_when_required_anchor_is_missing, py:test_cycle_run_artifacts_can_include_cognitive_memory_report, py:test_embedding_is_torch_tensor_and_deterministic, py:test_query_conditioned_reconstruction_preserves_old_exact_anchors, py:test_query_conditioning_prefers_relevant_latent_segment, py:test_recent_exact_kv_eviction_creates_latent_old_kv
- `tests/test_cortex3.py`: py:Cortex3Test, py:test_anchors_and_ternary, py:test_anti_metamorphic_variants_change_expected_answers, py:test_arithmetic_oracle, py:test_code_anti_metamorphic_variant_is_a_valid_changed_contract, py:test_code_unit_test_oracle_rejects_wrong_implementation, py:test_compare_finds_regressions_and_regrowth, py:test_oracle_quality_auditor_detects_no_false_positive_or_negative_on_default_skills, py:test_phase1_default_skills_cover_plan_domains, py:test_reference_beats_corrupted_candidate, py:test_regression_harness_detects_injected_fault_matrix, py:test_reporting_persists_schema_and_per_case_verifier_costs, py:test_strict_exact_anchor_entity_and_calibration_oracles_reject_embedded_answers, py:test_verifier_cost_profile_uses_real_case_costs_not_count_allocation
- `tests/test_cycle_modules.py`: py:CortexCycleModulesTest, py:test_cycle_finds_regressions_and_actions, py:test_ledgers_and_selection, py:test_phase_registry_has_ten_steps
- `tests/test_frontier_discovery.py`: py:FrontierSkillDiscoveryTest, py:test_frontier_discovery_report_is_persisted, py:test_frontier_discovery_slow_solves_distills_and_compiles_fragile_skill
- `tests/test_future_contracts.py`: py:_confident_heads, py:FutureContractsTest, py:test_contract_revision_rejects_mismatch_and_high_temporal_loss, py:test_future_contract_accepts_low_risk_fast_path, py:test_future_contract_from_logits_accepts_and_rejects_observed_blocks, py:test_mtp_fsp_calibrator_learns_standalone_future_tokens, py:test_mtp_heads_have_required_horizons_and_confidence, py:test_risky_domain_shortens_and_requires_gate_before_acceptance, py:test_temporal_consistency_loss_from_outputs_runs_on_real_heads, py:test_temporal_consistency_loss_rewards_shifted_future_agreement, py:test_verified_answers_per_effective_cost_is_not_tokens_per_second
- `tests/test_inference.py`: py:_engine, py:_verifier, py:test_budget_predictor_prices_careful_path_above_fast_path, py:test_careful_route_runs_strong_verification_certificate_and_expert_trace, py:test_cycle_report_can_persist_inference_results, py:test_difficulty_router_selects_fast_normal_and_careful_paths, py:test_engine_can_use_trained_autoregressive_answer_source, py:test_fast_path_uses_less_depth_and_better_verified_cost_than_careful_on_easy_task, py:test_fast_path_verified_cost_rejects_confident_wrong_answer, py:test_inference_gate_rejects_tampered_proof_carrying_answer, py:test_latent_kv_reconstruction_is_used_and_anchor_fidelity_is_verified, py:test_memory_augmented_generation_recovers_latent_anchor_answer, py:test_normal_path_uses_light_certificate_and_moderate_budget, py:test_self_speculative_decoding_respects_route_horizon_caps, py:UltraFastInferenceTest
- `tests/test_llm_pretraining.py`: py:_corpus, py:LLMPretrainingHarnessTest, py:test_bpe_tokenizer_and_memmap_corpus_are_streamable, py:test_comparison_matrix_reuses_shared_corpus_across_seeds, py:test_corpus_matrix_aggregates_multiple_corpora_and_seeds, py:test_cortex_comparison_produces_checkpoints_curves_and_cost_win, py:test_cuda_requirement_is_explicit_not_silent_fallback, py:test_distributed_runtime_can_pin_gloo_interface_without_initializing, py:test_doctor_reports_dependency_and_cuda_readiness, py:test_hf_dataset_export_builds_real_text_shards_and_token_memmap, py:test_hf_dataset_export_namespaced_id_error_is_actionable, py:test_manifest_experiment_runs_doctor_prepare_and_corpus_matrix, py:test_multi_domain_benchmark_aggregates_real_learning_curves, py:test_statistical_benchmark_matrix_requires_all_seed_domain_wins, py:test_trainer_resumes_checkpoint_with_gradient_accumulation
- `tests/test_microtrain.py`: py:MicroTrainTest, py:test_checkpoint_save_and_load_preserves_predictions, py:test_micro_model_training_improves_verified_accuracy, py:test_sleep_phase_examples_can_train_checkpoint
- `tests/test_objective_metrics.py`: py:_cycle_bundle, py:Decision, py:Evaluation, py:Improvement, py:ObjectiveMetricsTest, py:test_absolute_metrics_contains_every_plan_metric, py:test_final_loss_contains_every_plan_term_with_weighted_total, py:test_inference_results_feed_path_anchor_and_certificate_metrics, py:test_objective_uses_custom_weights, py:test_recursive_invalidity_counts_calibration_and_collapse, py:test_reporting_persists_objective_report
- `tests/test_plan_experiments.py`: py:PlanExperimentsTest, py:test_experiment_report_is_persisted_in_cycle_artifact, py:test_experiments_a_to_e_all_pass_with_named_metrics
- `tests/test_recursive_improvement.py`: py:_cycle, py:_verifier, py:RecursiveImprovementTest, py:test_engine_accepts_pareto_improving_sandbox_proposals, py:test_evaluator_flags_unaffected_skill_collapse, py:test_gate_rejects_calibration_regression, py:test_gate_rejects_protected_skill_regression, py:test_proposal_generator_maps_regressions_to_allowed_proposal_kinds, py:test_reporting_can_persist_recursive_improvement_report, py:test_reward_hacking_detector_flags_overfit_payload, py:test_rollback_archive_records_accepted_patch_tokens, py:test_sandbox_trainer_uses_in_memory_patch_without_touching_files
- `tests/test_regrowth.py`: py:_baseline_for_failure, py:_verifier, py:agent, py:RegrowthTest, py:test_action_space_covers_all_plan_repair_actions, py:test_cycle_run_artifacts_can_include_regrowth_plan, py:test_force_exact_anchor_action_recovers_anchor_failure, py:test_minimal_regrowth_selects_recovering_block_action_with_non_regression, py:test_output_goal_certificate_action_repairs_instruction_format, py:test_run_cycle_report_builds_regrowth_plans_from_real_cycle, py:test_training_micro_family_action_generates_verified_replay_tasks, py:test_unzero_change_sign_and_scale_precision_are_real_artifact_edits
- `tests/test_reporting_and_ternary.py`: py:ReportingAndTernaryTest, py:test_activation_quantization_and_residual_buffer, py:test_bitlinear_dependency_boundary, py:test_bitlinear_matches_float_linear_when_activation_quantization_is_disabled, py:test_bitlinear_preserves_gradient_path_through_residual_weight, py:test_compression_trace_ledger_records_plan_phase2_logs, py:test_cycle_run_artifacts_are_persisted, py:test_cycle_run_artifacts_can_include_future_contract_ledger, py:test_micro_model_forward_emits_layer_forward_trace
- `tests/test_sleep_phase.py`: py:_arithmetic_task, py:_verifier, py:SleepPhaseTest, py:test_anti_collapse_filter_rejects_high_contamination_duplicates_and_calibration_drop, py:test_failure_replay_buffer_and_tool_solver_create_verified_training_examples, py:test_metamorphic_family_builder_keeps_oracle_verified_labels, py:test_real_exogenous_reservoir_and_reporting_persist_sleep_phase, py:test_sleep_phase_consolidator_builds_diverse_rare_skill_schedule, py:test_verified_synthetic_pool_requires_trust_label
- `tools/launch_llm_ddp.py`: py:_default_gloo_interface, py:_normalize_train_args, py:_tail, py:main
- `tools/run_cycle_report.py`: py:_baseline_agent_for_failure, py:_protected_tasks_from_report, py:agent, py:build_autoregressive_smoke, py:build_causal_attribution_smoke, py:build_certificate_smoke, py:build_inference_smoke, py:build_regrowth_smoke, py:main
