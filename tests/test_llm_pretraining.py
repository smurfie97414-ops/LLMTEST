import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from cortex3_llm import (
    ComparisonConfig,
    DistributedRuntime,
    HFDatasetExportConfig,
    HFDatasetTextExporter,
    LLMBenchmarkSuite,
    LLMComparisonMatrixSuite,
    LLMComparisonRunner,
    LLMCorpusMatrixSuite,
    LLMExperimentRunner,
    LLMStatisticalBenchmarkSuite,
    LLMTrainer,
    LLMTokenizer,
    MemmapCausalDataset,
    PrecisionPolicy,
    TextCorpusConfig,
    TextShardReader,
    TokenizedCorpusBuilder,
    TrainingConfig,
    TransformerConfig,
    CortexTransformerLM,
    CortexTrainingPhaseController,
    TrainingPoint,
    TrainingRunReport,
    audit_llm_experiment_artifacts,
    audit_learning_curves,
    build_training_plan,
    build_benchmark_corpus,
    build_seed_corpus,
    hardware_report,
    inspect_llm_experiment,
    llm_doctor_report,
    main as llm_main,
)
from tools.launch_llm_ddp import _manifest_requests_cuda, _train_args_request_cuda


class LLMPretrainingHarnessTest(unittest.TestCase):
    def _corpus(self, root: Path, *, repeats: int = 80) -> TextCorpusConfig:
        files = build_seed_corpus(root / "text", repeats=repeats)
        return TextCorpusConfig.from_paths(files, min_chars_per_chunk=512)

    def test_bpe_tokenizer_and_memmap_corpus_are_streamable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._corpus(root)
            tokenizer = LLMTokenizer.train(corpus, vocab_size=192, min_frequency=1)
            encoded = tokenizer.encode("cortex compiles verified skills.")
            self.assertGreater(len(encoded), 4)
            self.assertEqual(encoded[0], tokenizer.bos_id)
            self.assertEqual(encoded[-1], tokenizer.eos_id)

            manifest = TokenizedCorpusBuilder(corpus, tokenizer).build(
                root / "prepared",
                seq_len=24,
                max_horizon=4,
            )
            self.assertGreater(manifest.token_count, 24)
            self.assertEqual(Path(manifest.token_file).stat().st_size, manifest.token_count * 4)
            self.assertRegex(manifest.token_file_sha256, r"^[0-9a-f]{64}$")
            self.assertRegex(manifest.tokenizer_file_sha256, r"^[0-9a-f]{64}$")
            self.assertEqual(len(manifest.source_file_fingerprints), len(manifest.source_files))
            identity = manifest.identity()
            self.assertRegex(identity["identity_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(identity["token_file_sha256"], manifest.token_file_sha256)
            train = MemmapCausalDataset(manifest, split="train")
            try:
                x, y, future = train.item(0)
                self.assertEqual(tuple(x.shape), (24,))
                self.assertEqual(tuple(y.shape), (24,))
                self.assertEqual(tuple(future.shape), (24, 4))
                self.assertEqual(int(y[0]), int(future[0, 0]))

                batch_offsets = [0, 1, min(2, len(train) - 1)]
                bx, by, bfuture = train.batch_at(batch_offsets, device=torch.device("cpu"))
                self.assertEqual(tuple(bx.shape), (3, 24))
                self.assertEqual(tuple(by.shape), (3, 24))
                self.assertEqual(tuple(bfuture.shape), (3, 24, 4))
                for batch_index, offset in enumerate(batch_offsets):
                    ix, iy, ifuture = train.item(offset)
                    self.assertTrue(torch.equal(bx[batch_index], ix))
                    self.assertTrue(torch.equal(by[batch_index], iy))
                    self.assertTrue(torch.equal(bfuture[batch_index], ifuture))
                with self.assertRaises(IndexError):
                    train.batch_at([len(train)], device=torch.device("cpu"))
            finally:
                train.close()
            first_source = Path(manifest.source_files[0])
            first_source.write_text(first_source.read_text(encoding="utf-8") + "\ncorpus mutation", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source file size changed|source file sha256 changed"):
                manifest.identity()

    def test_tokenized_corpus_builder_can_cap_prepared_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._corpus(root, repeats=120)
            tokenizer = LLMTokenizer.train(corpus, vocab_size=192, min_frequency=1)
            manifest = TokenizedCorpusBuilder(corpus, tokenizer).build(
                root / "prepared-capped",
                seq_len=24,
                max_horizon=4,
                max_tokens=128,
            )
            self.assertEqual(manifest.token_count, 128)
            self.assertEqual(Path(manifest.token_file).stat().st_size, 128 * 4)
            self.assertEqual(manifest.preparation_config["max_tokens"], 128)
            train = MemmapCausalDataset(manifest, split="train")
            try:
                self.assertGreater(len(train), 0)
            finally:
                train.close()
            with self.assertRaisesRegex(ValueError, "max_tokens"):
                TokenizedCorpusBuilder(corpus, tokenizer).build(
                    root / "prepared-too-small",
                    seq_len=24,
                    max_horizon=4,
                    max_tokens=16,
                )

    def test_tokenized_corpus_builder_streams_tokens_once(self):
        class CountingTokenizer:
            vocab_size = 64

            def __init__(self):
                self.encode_calls = 0

            def encode(self, text: str) -> tuple[int, ...]:
                self.encode_calls += 1
                payload = [4 + (ord(ch) % 48) for ch in text if not ch.isspace()]
                return tuple([1, *payload, 2])

            def save(self, path: str | Path) -> Path:
                output = Path(path)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("{}", encoding="utf-8")
                return output

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_file = root / "stream.txt"
            text_file.write_text(("cortex streaming corpus writes uint32 tokens once per chunk.\n" * 40), encoding="utf-8")
            corpus = TextCorpusConfig.from_paths([text_file], min_chars_per_chunk=256)
            expected_chunks = sum(1 for _ in TextShardReader(corpus).iter_chunks())
            tokenizer = CountingTokenizer()

            manifest = TokenizedCorpusBuilder(corpus, tokenizer).build(
                root / "prepared",
                seq_len=16,
                max_horizon=2,
            )

            self.assertEqual(tokenizer.encode_calls, expected_chunks)
            self.assertEqual(Path(manifest.token_file).stat().st_size, manifest.token_count * 4)

    def test_training_plan_matches_transformer_parameter_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._corpus(root, repeats=80)
            tokenizer = LLMTokenizer.train(corpus, vocab_size=192, min_frequency=1)
            manifest = TokenizedCorpusBuilder(corpus, tokenizer).build(
                root / "prepared",
                seq_len=24,
                max_horizon=4,
            )
            training = TrainingConfig(
                steps=10,
                batch_size=3,
                gradient_accumulation_steps=2,
                eval_interval=5,
                eval_batches=2,
                precision="bf16",
                num_threads=1,
            )
            config = ComparisonConfig(
                vocab_size=192,
                min_frequency=1,
                seq_len=24,
                d_model=32,
                n_heads=4,
                n_layers=2,
                horizons=(1, 2, 4),
                training=training,
            )
            plan = build_training_plan(manifest, config, world_size=2, distributed=True)
            baseline_config = TransformerConfig(
                vocab_size=manifest.vocab_size,
                seq_len=24,
                d_model=32,
                n_heads=4,
                n_layers=2,
                horizons=(1, 2, 4),
                use_cortex_heads=False,
            )
            cortex_config = TransformerConfig(**{
                **baseline_config.__dict__,
                "use_cortex_heads": True,
                "use_ternary_core": True,
                "use_skill_aware_experts": True,
                "use_variable_in_compressor": True,
                "use_certificate_head": True,
            })
            baseline_parameters = sum(parameter.numel() for parameter in CortexTransformerLM(baseline_config).parameters())
            cortex_parameters = sum(parameter.numel() for parameter in CortexTransformerLM(cortex_config).parameters())
            self.assertEqual(plan["model"]["baseline_parameters"], baseline_parameters)
            self.assertEqual(plan["model"]["cortex_parameters"], cortex_parameters)
            self.assertTrue(plan["model"]["cortex_skill_aware_experts"])
            self.assertEqual(plan["model"]["cortex_skill_expert_count"], 4)
            self.assertTrue(plan["model"]["cortex_variable_in_compressor"])
            self.assertTrue(plan["model"]["cortex_certificate_head"])
            self.assertGreater(plan["model"]["cortex_parameters"], plan["model"]["baseline_parameters"])
            self.assertEqual(plan["training"]["tokens_per_optimizer_step"], 3 * 2 * 2 * 24)
            self.assertEqual(plan["training"]["planned_train_tokens"], 3 * 2 * 2 * 24 * 10)
            self.assertGreater(plan["training"]["effective_epochs_over_train_split"], 0.0)

    def test_learning_curve_audit_rejects_missing_final_validation_point(self):
        baseline = TrainingRunReport(
            name="baseline_ntp",
            model_kind="baseline_next_token",
            run_dir="baseline",
            checkpoint_path="baseline/checkpoint_final.pt",
            start_step=0,
            optimizer_steps=4,
            effective_batch_size=2,
            resumed_from=None,
            final_train=TrainingPoint(step=4, split="train", loss=1.0, next_token_loss=1.0, token_accuracy=0.1),
            final_val=TrainingPoint(step=0, split="val", loss=1.2, next_token_loss=1.2, token_accuracy=0.1),
            curve=(
                TrainingPoint(step=0, split="train", loss=1.3, next_token_loss=1.3, token_accuracy=0.1),
                TrainingPoint(step=0, split="val", loss=1.2, next_token_loss=1.2, token_accuracy=0.1),
            ),
            config={},
            hardware={},
        )
        cortex = TrainingRunReport(
            name="cortex3",
            model_kind="cortex3_multi_horizon",
            run_dir="cortex",
            checkpoint_path="cortex/checkpoint_final.pt",
            start_step=0,
            optimizer_steps=4,
            effective_batch_size=2,
            resumed_from=None,
            final_train=TrainingPoint(step=4, split="train", loss=1.0, next_token_loss=1.0, token_accuracy=0.1),
            final_val=TrainingPoint(step=4, split="val", loss=1.1, next_token_loss=1.1, token_accuracy=0.1),
            curve=(
                TrainingPoint(step=0, split="train", loss=1.3, next_token_loss=1.3, token_accuracy=0.1),
                TrainingPoint(step=0, split="val", loss=1.2, next_token_loss=1.2, token_accuracy=0.1),
                TrainingPoint(step=4, split="train", loss=1.0, next_token_loss=1.0, token_accuracy=0.1),
                TrainingPoint(step=4, split="val", loss=1.1, next_token_loss=1.1, token_accuracy=0.1),
            ),
            config={},
            hardware={},
        )

        audit = audit_learning_curves(baseline, cortex, expected_final_step=4)

        self.assertFalse(audit["passed"], audit)
        self.assertIn("baseline_ntp", audit["failed_models"])
        self.assertIn("missing_final_validation_step", audit["baseline"]["failed_checks"])

    def test_comparison_proof_rejects_zero_baseline_score(self):
        config = ComparisonConfig(
            cortex_win_margin=1.02,
            max_next_token_loss_regression=1.50,
            min_baseline_future_tokens_per_cost=0.001,
        )
        runner = object.__new__(LLMComparisonRunner)
        runner.config = config
        baseline = TrainingRunReport(
            name="baseline_ntp",
            model_kind="baseline_next_token",
            run_dir="baseline",
            checkpoint_path="baseline/checkpoint_final.pt",
            start_step=0,
            optimizer_steps=4,
            effective_batch_size=2,
            resumed_from=None,
            final_train=TrainingPoint(step=4, split="train", loss=1.0, next_token_loss=1.0, token_accuracy=0.1),
            final_val=TrainingPoint(
                step=4,
                split="val",
                loss=1.0,
                next_token_loss=1.0,
                token_accuracy=0.1,
                future_tokens_per_cost=0.0,
            ),
            curve=(),
            config={},
            hardware={},
        )
        cortex = TrainingRunReport(
            name="cortex3",
            model_kind="cortex3_multi_horizon",
            run_dir="cortex",
            checkpoint_path="cortex/checkpoint_final.pt",
            start_step=0,
            optimizer_steps=4,
            effective_batch_size=2,
            resumed_from=None,
            final_train=TrainingPoint(step=4, split="train", loss=1.0, next_token_loss=1.0, token_accuracy=0.1),
            final_val=TrainingPoint(
                step=4,
                split="val",
                loss=0.9,
                next_token_loss=0.9,
                token_accuracy=0.2,
                future_tokens_per_cost=0.25,
            ),
            curve=(),
            config={},
            hardware={},
        )

        proof = runner._proof_payload(baseline, cortex, {"passed": True})

        self.assertFalse(proof["passed"], proof)
        self.assertFalse(proof["baseline_score_passed"], proof)
        self.assertIn("baseline_score_passed", proof["failed_checks"])
        self.assertTrue(proof["checks"]["ratio_passed"], proof)
        self.assertGreater(proof["cortex_over_baseline_ratio"], 1.02)

    def test_comparison_proof_rejects_toy_scale_when_required(self):
        config = ComparisonConfig(
            cortex_win_margin=1.02,
            max_next_token_loss_regression=1.50,
            min_baseline_future_tokens_per_cost=0.001,
            min_corpus_tokens=1_000,
            min_planned_train_tokens=5_000,
        )
        runner = object.__new__(LLMComparisonRunner)
        runner.config = config
        baseline = TrainingRunReport(
            name="baseline_ntp",
            model_kind="baseline_next_token",
            run_dir="baseline",
            checkpoint_path="baseline/checkpoint_final.pt",
            start_step=0,
            optimizer_steps=4,
            effective_batch_size=2,
            resumed_from=None,
            final_train=TrainingPoint(step=4, split="train", loss=1.0, next_token_loss=1.0, token_accuracy=0.1),
            final_val=TrainingPoint(
                step=4,
                split="val",
                loss=1.0,
                next_token_loss=1.0,
                token_accuracy=0.1,
                future_tokens_per_cost=0.05,
            ),
            curve=(),
            config={},
            hardware={},
        )
        cortex = TrainingRunReport(
            name="cortex3",
            model_kind="cortex3_multi_horizon",
            run_dir="cortex",
            checkpoint_path="cortex/checkpoint_final.pt",
            start_step=0,
            optimizer_steps=4,
            effective_batch_size=2,
            resumed_from=None,
            final_train=TrainingPoint(step=4, split="train", loss=1.0, next_token_loss=1.0, token_accuracy=0.1),
            final_val=TrainingPoint(
                step=4,
                split="val",
                loss=0.8,
                next_token_loss=0.8,
                token_accuracy=0.2,
                future_tokens_per_cost=0.25,
            ),
            curve=(),
            config={},
            hardware={},
        )
        plan = {
            "corpus": {"token_count": 512},
            "training": {"planned_train_tokens": 2_048},
        }

        proof = runner._proof_payload(baseline, cortex, {"passed": True}, plan=plan)

        self.assertFalse(proof["passed"], proof)
        self.assertTrue(proof["checks"]["ratio_passed"], proof)
        self.assertFalse(proof["corpus_scale_passed"], proof)
        self.assertFalse(proof["planned_train_tokens_passed"], proof)
        self.assertIn("corpus_scale_passed", proof["failed_checks"])
        self.assertIn("planned_train_tokens_passed", proof["failed_checks"])

    def test_comparison_proof_requires_full_cortex_phase_report_for_full_architecture(self):
        config = ComparisonConfig(
            cortex_win_margin=1.02,
            max_next_token_loss_regression=1.50,
            min_baseline_future_tokens_per_cost=0.001,
        )
        runner = object.__new__(LLMComparisonRunner)
        runner.config = config
        baseline = TrainingRunReport(
            name="baseline_ntp",
            model_kind="baseline_next_token",
            run_dir="baseline",
            checkpoint_path="baseline/checkpoint_final.pt",
            start_step=0,
            optimizer_steps=4,
            effective_batch_size=2,
            resumed_from=None,
            final_train=TrainingPoint(step=4, split="train", loss=1.0, next_token_loss=1.0, token_accuracy=0.1),
            final_val=TrainingPoint(
                step=4,
                split="val",
                loss=1.0,
                next_token_loss=1.0,
                token_accuracy=0.1,
                future_tokens_per_cost=0.05,
            ),
            curve=(),
            config={},
            hardware={},
        )
        cortex = TrainingRunReport(
            name="cortex3",
            model_kind="cortex3_multi_horizon",
            run_dir="cortex",
            checkpoint_path="cortex/checkpoint_final.pt",
            start_step=0,
            optimizer_steps=4,
            effective_batch_size=2,
            resumed_from=None,
            final_train=TrainingPoint(step=4, split="train", loss=1.0, next_token_loss=1.0, token_accuracy=0.1),
            final_val=TrainingPoint(
                step=4,
                split="val",
                loss=0.8,
                next_token_loss=0.8,
                token_accuracy=0.2,
                future_tokens_per_cost=0.25,
            ),
            curve=(),
            config={"model": {"use_cortex_heads": True, "use_ternary_core": True, "horizons": [1, 2, 4, 8]}},
            hardware={},
            cortex_phase_report={
                "enabled": True,
                "all_phases_active": True,
                "phase_event_counts": {f"P{index}": 1 for index in range(1, 11)},
                "errors": [],
            },
        )

        proof = runner._proof_payload(
            baseline,
            cortex,
            {"passed": True},
            plan={"corpus": {"token_count": 10_000}, "training": {"planned_train_tokens": 10_000}},
        )

        self.assertFalse(proof["passed"], proof)
        self.assertTrue(proof["cortex_full_phase_required"], proof)
        self.assertFalse(proof["cortex_architecture_audit_passed"], proof)
        self.assertFalse(proof["cortex_phase_deliverable_audit_passed"], proof)
        self.assertFalse(proof["cortex_phase_integration_passed"], proof)
        self.assertIn("cortex_architecture_audit_passed", proof["failed_checks"])
        self.assertIn("cortex_phase_deliverable_audit_passed", proof["failed_checks"])
        self.assertIn("cortex_phase_integration_passed", proof["failed_checks"])

    def test_trainer_resumes_checkpoint_with_gradient_accumulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._corpus(root, repeats=80)
            tokenizer = LLMTokenizer.train(corpus, vocab_size=192, min_frequency=1)
            manifest = TokenizedCorpusBuilder(corpus, tokenizer).build(
                root / "prepared",
                seq_len=16,
                max_horizon=2,
            )
            model_config = TransformerConfig(
                vocab_size=manifest.vocab_size,
                seq_len=16,
                d_model=32,
                n_heads=4,
                n_layers=1,
                dropout=0.0,
                horizons=(1, 2),
                use_cortex_heads=True,
            )
            train = MemmapCausalDataset(manifest, split="train")
            val = MemmapCausalDataset(manifest, split="val")
            corpus_identity = manifest.identity()
            try:
                first_config = TrainingConfig(
                    steps=2,
                    batch_size=2,
                    gradient_accumulation_steps=2,
                    eval_interval=1,
                    eval_batches=1,
                    checkpoint_interval=1,
                    seed=31,
                    num_threads=1,
                )
                run_dir = root / "resume-run"
                first = LLMTrainer(
                    CortexTransformerLM(model_config),
                    train,
                    val,
                    first_config,
                    run_dir=run_dir,
                    model_kind="cortex3_multi_horizon",
                    corpus_identity=corpus_identity,
                ).train(name="cortex3")
                self.assertEqual(first.start_step, 0)
                self.assertEqual(first.optimizer_steps, 2)
                self.assertEqual(first.effective_batch_size, 4)
                self.assertTrue((run_dir / "checkpoint_step_1.pt").exists())
                self.assertTrue((run_dir / "checkpoint_step_1.pt.json").exists())
                self.assertTrue((run_dir / "resource_usage_live.json").exists())
                self.assertTrue((run_dir / "checkpoint_final.pt").exists())
                first_sidecar = json.loads((run_dir / "checkpoint_step_1.pt.json").read_text(encoding="utf-8"))
                self.assertEqual(first_sidecar["step"], 1)
                self.assertIn("git_commit", first_sidecar["code_state"])
                legacy_checkpoint = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
                for key in (
                    "use_skill_aware_experts",
                    "skill_expert_count",
                    "skill_expert_top_k",
                    "use_variable_in_compressor",
                    "variable_compression_wide_kernel",
                    "use_certificate_head",
                    "certificate_latent_size",
                ):
                    legacy_checkpoint["model_config"].pop(key, None)
                torch.save(legacy_checkpoint, run_dir / "checkpoint_final.pt")

                resumed_config = TrainingConfig(
                    steps=4,
                    batch_size=2,
                    gradient_accumulation_steps=2,
                    eval_interval=1,
                    eval_batches=1,
                    checkpoint_interval=1,
                    seed=31,
                    resume=True,
                    num_threads=1,
                )
                resumed = LLMTrainer(
                    CortexTransformerLM(model_config),
                    train,
                    val,
                    resumed_config,
                    run_dir=run_dir,
                    model_kind="cortex3_multi_horizon",
                    corpus_identity=corpus_identity,
                ).train(name="cortex3")
                self.assertEqual(resumed.start_step, 2)
                self.assertEqual(resumed.optimizer_steps, 2)
                self.assertEqual(resumed.effective_batch_size, 4)
                self.assertEqual(resumed.final_val.step, 4)
                self.assertEqual(Path(resumed.resumed_from).name, "checkpoint_final.pt")
                checkpoint = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
                self.assertEqual(checkpoint["step"], 4)
                self.assertEqual(checkpoint["corpus_identity"], corpus_identity)
                self.assertIn("code_state", checkpoint)
                self.assertIn("git_commit", checkpoint["code_state"])
                self.assertIn("rng_state", checkpoint)
                self.assertGreaterEqual(len(checkpoint["curve"]), len(first.curve))
                final_sidecar = json.loads((run_dir / "checkpoint_final.pt.json").read_text(encoding="utf-8"))
                self.assertEqual(final_sidecar["step"], 4)
                self.assertEqual(final_sidecar["code_state"]["git_commit"], checkpoint["code_state"]["git_commit"])
                self.assertIn("code_state", resumed.to_dict())
                live_usage = json.loads((run_dir / "resource_usage_live.json").read_text(encoding="utf-8"))
                self.assertEqual(live_usage["metadata"]["step"], 4)
                self.assertGreaterEqual(live_usage["sample_count"], 1)
                self.assertIn("cpu_total_percent", live_usage["metrics"])
                final_usage = json.loads((run_dir / "resource_usage_summary.json").read_text(encoding="utf-8"))
                self.assertTrue(final_usage["metadata"]["final"])
                self.assertGreaterEqual(final_usage["sample_count"], 1)
                self.assertIn("process_memory_rss_bytes", final_usage["metrics"])
            finally:
                train.close()
                val.close()

    def test_checkpoint_code_state_is_frozen_at_trainer_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._corpus(root, repeats=40)
            tokenizer = LLMTokenizer.train(corpus, vocab_size=160, min_frequency=1)
            manifest = TokenizedCorpusBuilder(corpus, tokenizer).build(
                root / "prepared",
                seq_len=16,
                max_horizon=2,
            )
            model_config = TransformerConfig(
                vocab_size=manifest.vocab_size,
                seq_len=16,
                d_model=32,
                n_heads=4,
                n_layers=1,
                dropout=0.0,
                horizons=(1, 2),
            )
            train = MemmapCausalDataset(manifest, split="train")
            val = MemmapCausalDataset(manifest, split="val")
            frozen_code_state = {
                "schema_version": 1,
                "git_commit": "start-commit",
                "git_branch": "main",
                "tracked_dirty": False,
            }
            try:
                with patch("cortex3_llm.code_state_report", return_value=frozen_code_state) as mocked_code_state:
                    report = LLMTrainer(
                        CortexTransformerLM(model_config),
                        train,
                        val,
                        TrainingConfig(
                            steps=1,
                            batch_size=2,
                            eval_interval=1,
                            eval_batches=1,
                            checkpoint_interval=1,
                            seed=37,
                            num_threads=1,
                        ),
                        run_dir=root / "code-state",
                        model_kind="baseline_next_token",
                        corpus_identity=manifest.identity(),
                    ).train(name="baseline")
            finally:
                train.close()
                val.close()

            self.assertEqual(mocked_code_state.call_count, 1)
            checkpoint = torch.load(root / "code-state" / "checkpoint_final.pt", map_location="cpu", weights_only=False)
            sidecar = json.loads((root / "code-state" / "checkpoint_final.pt.json").read_text(encoding="utf-8"))
            self.assertEqual(report.code_state["git_commit"], "start-commit")
            self.assertEqual(checkpoint["code_state"]["git_commit"], "start-commit")
            self.assertEqual(sidecar["code_state"]["git_commit"], "start-commit")

    def test_intermediate_checkpoint_retention_prunes_old_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._corpus(root, repeats=40)
            tokenizer = LLMTokenizer.train(corpus, vocab_size=128, min_frequency=1)
            manifest = TokenizedCorpusBuilder(corpus, tokenizer).build(
                root / "prepared",
                seq_len=16,
                max_horizon=4,
            )
            train = MemmapCausalDataset(manifest, split="train")
            val = MemmapCausalDataset(manifest, split="val")
            run_dir = root / "retained-checkpoints"
            try:
                LLMTrainer(
                    CortexTransformerLM(
                        TransformerConfig(
                            vocab_size=manifest.vocab_size,
                            seq_len=16,
                            d_model=32,
                            n_heads=4,
                            n_layers=1,
                            dropout=0.0,
                            horizons=(1, 2, 4),
                            use_cortex_heads=False,
                            use_ternary_core=False,
                        )
                    ),
                    train,
                    val,
                    TrainingConfig(
                        steps=3,
                        batch_size=2,
                        eval_interval=1,
                        eval_batches=1,
                        checkpoint_interval=1,
                        max_intermediate_checkpoints=2,
                        seed=17,
                        num_threads=1,
                    ),
                    run_dir=run_dir,
                    model_kind="baseline_next_token",
                    corpus_identity=manifest.identity(),
                ).train(name="baseline")
            finally:
                train.close()
                val.close()

            self.assertFalse((run_dir / "checkpoint_step_1.pt").exists())
            self.assertFalse((run_dir / "checkpoint_step_1.pt.json").exists())
            self.assertTrue((run_dir / "checkpoint_step_2.pt").exists())
            self.assertTrue((run_dir / "checkpoint_step_2.pt.json").exists())
            self.assertTrue((run_dir / "checkpoint_step_3.pt").exists())
            self.assertTrue((run_dir / "checkpoint_step_3.pt.json").exists())
            self.assertTrue((run_dir / "checkpoint_final.pt").exists())

    def test_training_config_rejects_strict_and_auto_resume_together(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            TrainingConfig(resume=True, resume_if_exists=True)
        with self.assertRaisesRegex(ValueError, "cortex_objective_feedback_weight"):
            TrainingConfig(cortex_objective_feedback_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "cortex_objective_feedback_clip"):
            TrainingConfig(cortex_objective_feedback_clip=-1.0)
        with self.assertRaisesRegex(ValueError, "cortex_trace_retention_limit"):
            TrainingConfig(cortex_trace_retention_limit=-1)

    def test_full_cortex_phase_controller_uses_all_modules_during_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._corpus(root, repeats=90)
            tokenizer = LLMTokenizer.train(corpus, vocab_size=192, min_frequency=1)
            manifest = TokenizedCorpusBuilder(corpus, tokenizer).build(
                root / "prepared",
                seq_len=24,
                max_horizon=8,
            )
            model_config = TransformerConfig(
                vocab_size=manifest.vocab_size,
                seq_len=24,
                d_model=32,
                n_heads=4,
                n_layers=1,
                dropout=0.0,
                horizons=(1, 2, 4, 8),
                use_cortex_heads=True,
                use_ternary_core=True,
                use_skill_aware_experts=True,
                use_variable_in_compressor=True,
                use_certificate_head=True,
            )
            train = MemmapCausalDataset(manifest, split="train")
            val = MemmapCausalDataset(manifest, split="val")
            run_dir = root / "full-cortex"
            try:
                report = LLMTrainer(
                    CortexTransformerLM(model_config),
                    train,
                    val,
                    TrainingConfig(
                        steps=1,
                        batch_size=2,
                        eval_interval=1,
                        eval_batches=1,
                        checkpoint_interval=1,
                        seed=53,
                        num_threads=1,
                        cortex_phase_interval=1,
                        cortex_phase_probe_tasks=1,
                        cortex_phase_max_proposals=1,
                    ),
                    run_dir=run_dir,
                    model_kind="cortex3_multi_horizon",
                    corpus_identity=manifest.identity(),
                ).train(name="cortex3")
            finally:
                train.close()
                val.close()

            phase_report = report.cortex_phase_report
            self.assertTrue(phase_report["enabled"], phase_report)
            self.assertFalse(phase_report["errors"], phase_report)
            self.assertTrue(phase_report["all_phases_active"], phase_report)
            self.assertEqual(set(phase_report["phase_event_counts"]), {f"P{index}" for index in range(1, 11)})
            for phase_id, count in phase_report["phase_event_counts"].items():
                self.assertGreater(count, 0, phase_id)
            architecture_audit = phase_report["architecture_audit"]
            self.assertTrue(architecture_audit["passed"], architecture_audit)
            expected_components = {
                "p1_to_p10_phase_activity",
                "variable_in_compressor",
                "exact_anchor_ledger",
                "latent_memory_kv",
                "ternary_core",
                "skill_aware_experts",
                "bit_ledger",
                "skill_ledger",
                "causal_ledger",
                "uncertainty_ledger",
                "future_contract_fsp",
                "adaptive_multi_token_decoding",
                "latent_reasoning_workspace",
                "certificate_generator",
                "hierarchical_dynamic_verifier",
                "accept_reject_gate",
                "attribute_regression",
                "minimal_regrowth",
                "sleep_consolidation_buffer",
                "recursive_improvement",
                "training_feedback_loop",
            }
            self.assertEqual(set(architecture_audit["checks_by_component"]), expected_components)
            deliverable_audit = phase_report["phase_deliverable_audit"]
            self.assertTrue(deliverable_audit["passed"], deliverable_audit)
            self.assertEqual(deliverable_audit["deliverable_count"], 10)
            expected_deliverables = {
                "P1:verifier_os_regression_harness",
                "P2:ternary_sign_mask_activation_and_trace_logs",
                "P3:mtp_fsp_confidence_temporal_contract_gate",
                "P4:recent_exact_old_latent_query_memory_anchor_fidelity",
                "P5:latent_certificate_delatentization_tool_verification",
                "P6:causal_attribution_counterfactual_dimensions",
                "P7:minimal_regrowth_action_space_and_repair_plan",
                "P8:fast_normal_careful_budget_early_exit_mod_speculative_kernels",
                "P9:sleep_replay_synthetic_real_reservoir_anti_collapse_schedule",
                "P10:recursive_improvement_sandbox_pareto_rollback_diversity",
            }
            self.assertEqual(set(deliverable_audit["checks_by_deliverable"]), expected_deliverables)
            influence = phase_report["training_influence"]
            self.assertGreater(influence["ternary_core_forward_events"], 0)
            self.assertGreater(influence["variable_input_compression_events"], 0)
            self.assertGreater(influence["skill_expert_activations"], 0)
            self.assertGreater(influence["certificate_head_forward_events"], 0)
            self.assertGreater(influence["input_anchor_observations"], 0)
            self.assertGreater(influence["input_anchor_count"], 0)
            self.assertEqual(influence["input_anchor_fidelity_failures"], 0)
            self.assertGreater(influence["future_contract_decisions"], 0)
            self.assertGreater(influence["bit_ledger_total_effective_bits"], 0.0)
            self.assertGreater(influence["skill_ledger_states"], 0)
            self.assertGreater(influence["causal_ledger_traces"], 0)
            self.assertGreater(influence["uncertainty_ledger_observations"], 0)
            self.assertGreater(influence["confidence_regularization_steps"], 0)
            self.assertGreater(influence["sleep_replay_batches_available"], 0)
            self.assertGreater(influence["sleep_replay_updates"], 0)
            self.assertGreater(influence["phase_replay_examples"], 0)
            self.assertGreater(influence["objective_feedback_events"], 0)
            self.assertGreater(influence["last_objective_loss_total"], 0.0)
            self.assertGreater(influence["objective_feedback_scale"], 1.0)
            self.assertGreater(influence["memory_recent_segments"], 0)
            self.assertGreater(influence["sleep_replay_examples"], 0)
            self.assertGreater(influence["sleep_synthetic_examples"], 0)
            self.assertGreater(
                influence["improvement_archive_accepted"] + influence["improvement_archive_rejected"],
                0,
            )
            phase_replay = influence["phase_replay_examples_by_phase"]
            for phase_id in ("P1", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"):
                self.assertGreater(phase_replay[phase_id], 0, phase_replay)
            self.assertTrue(phase_report["phase_replay_example_ids"], phase_report)
            self.assertTrue(phase_report["objective_feedback_history"], phase_report)
            self.assertGreater(phase_report["ledgers"]["bit_ledger"]["total_effective_bits"], 0.0)
            self.assertTrue(phase_report["ledgers"]["skill_ledger"]["states"])
            self.assertGreater(phase_report["ledgers"]["causal_ledger"]["trace_count"], 0)
            self.assertGreater(phase_report["ledgers"]["uncertainty_ledger"]["observation_count"], 0)
            self.assertGreater(len(phase_report["memory_state_summary"]["anchors"]), 0)
            for sample in phase_report["batch_contract_samples"]:
                self.assertGreaterEqual(sample["observed_token_count"], sample["horizon"])
            self.assertTrue((run_dir / "cortex_phase_report.json").exists())
            persisted = json.loads((run_dir / "cortex_phase_report.json").read_text(encoding="utf-8"))
            self.assertTrue(persisted["all_phases_active"], persisted)
            self.assertTrue(persisted["architecture_audit"]["passed"], persisted["architecture_audit"])
            self.assertTrue(persisted["phase_deliverable_audit"]["passed"], persisted["phase_deliverable_audit"])
            self.assertEqual(persisted["training_influence"]["sleep_replay_updates"], influence["sleep_replay_updates"])
            self.assertEqual(
                persisted["training_influence"]["objective_feedback_events"],
                influence["objective_feedback_events"],
            )

    def test_future_contract_observed_tokens_use_real_horizon_not_horizon_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._corpus(root, repeats=20)
            tokenizer = LLMTokenizer.train(corpus, vocab_size=128, min_frequency=1)
            config = TransformerConfig(
                vocab_size=128,
                seq_len=16,
                d_model=32,
                n_heads=4,
                n_layers=1,
                dropout=0.0,
                horizons=(1, 2, 4, 8),
                use_cortex_heads=True,
                use_ternary_core=True,
                use_skill_aware_experts=True,
                use_variable_in_compressor=True,
                use_certificate_head=True,
            )
            controller = CortexTrainingPhaseController(
                CortexTransformerLM(config),
                tokenizer,
                TrainingConfig(steps=1, batch_size=1, eval_interval=1, checkpoint_interval=1),
                run_dir=root / "run",
            )
            future_targets = torch.zeros((1, 16, 4), dtype=torch.long)
            future_targets[0, -8:, 3] = torch.arange(80, 88)

            observed = controller._observed_contract_tokens(future_targets, 8)

            self.assertEqual(observed, list(range(80, 88)))

    def test_cortex_phase_state_survives_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._corpus(root, repeats=70)
            tokenizer = LLMTokenizer.train(corpus, vocab_size=192, min_frequency=1)
            manifest = TokenizedCorpusBuilder(corpus, tokenizer).build(
                root / "prepared",
                seq_len=24,
                max_horizon=8,
            )
            model_config = TransformerConfig(
                vocab_size=manifest.vocab_size,
                seq_len=24,
                d_model=32,
                n_heads=4,
                n_layers=1,
                dropout=0.0,
                horizons=(1, 2, 4, 8),
                use_cortex_heads=True,
                use_ternary_core=True,
                use_skill_aware_experts=True,
                use_variable_in_compressor=True,
                use_certificate_head=True,
            )
            train = MemmapCausalDataset(manifest, split="train")
            val = MemmapCausalDataset(manifest, split="val")
            run_dir = root / "cortex-resume"
            try:
                first = LLMTrainer(
                    CortexTransformerLM(model_config),
                    train,
                    val,
                    TrainingConfig(
                        steps=1,
                        batch_size=2,
                        eval_interval=1,
                        eval_batches=1,
                        checkpoint_interval=1,
                        seed=71,
                        num_threads=1,
                        cortex_phase_interval=1,
                        cortex_phase_probe_tasks=1,
                        cortex_phase_max_proposals=1,
                    ),
                    run_dir=run_dir,
                    model_kind="cortex3_multi_horizon",
                    corpus_identity=manifest.identity(),
                ).train(name="cortex3")
                first_influence = first.cortex_phase_report["training_influence"]
                self.assertGreater(first_influence["phase_replay_examples"], 0)
                self.assertGreater(first_influence["objective_feedback_events"], 0)
                self.assertGreater(first_influence["future_contract_decisions"], 0)
                self.assertGreater(first_influence["ternary_core_forward_events"], 0)
                checkpoint = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
                self.assertIn("cortex_phase_state", checkpoint)
                self.assertGreater(len(checkpoint["cortex_phase_state"]["replay_batches"]), 0)
                self.assertGreater(checkpoint["cortex_phase_state"]["objective_feedback_events"], 0)
                self.assertGreater(checkpoint["cortex_phase_state"]["last_objective_loss_total"], 0.0)
                self.assertGreater(checkpoint["cortex_phase_state"]["certificate_head_forward_events"], 0)
                self.assertGreater(checkpoint["cortex_phase_state"]["input_anchor_observations"], 0)
                self.assertGreater(checkpoint["cortex_phase_state"]["input_anchor_count"], 0)
                self.assertEqual(checkpoint["cortex_phase_state"]["input_anchor_fidelity_failures"], 0)
                self.assertGreater(checkpoint["cortex_phase_state"]["ledgers"]["bit_ledger"]["total_effective_bits"], 0.0)
                self.assertTrue(checkpoint["cortex_phase_state"]["ledgers"]["skill_ledger"]["states"])
                self.assertGreater(checkpoint["cortex_phase_state"]["ledgers"]["causal_ledger"]["trace_count"], 0)
                self.assertGreater(
                    checkpoint["cortex_phase_state"]["ledgers"]["uncertainty_ledger"]["observation_count"],
                    0,
                )
                self.assertGreater(len(checkpoint["cortex_phase_state"]["future_ledger"]["decisions"]), 0)
                self.assertGreater(
                    len(checkpoint["cortex_phase_state"]["compression_trace_ledger"]["layer_forward_events"]),
                    0,
                )
                self.assertEqual(
                    checkpoint["cortex_phase_state"]["compression_trace_ledger"]["retention_limit"],
                    4096,
                )
                self.assertGreater(
                    checkpoint["cortex_phase_state"]["compression_trace_ledger"]["total_event_counts"]["layer_forward_events"],
                    0,
                )
                self.assertGreater(
                    checkpoint["cortex_phase_state"]["compression_trace_ledger"]["total_event_counts"]["expert_activations"],
                    0,
                )
                self.assertGreater(
                    checkpoint["cortex_phase_state"]["compression_trace_ledger"]["total_event_counts"]["kv_events"],
                    0,
                )
                self.assertGreater(len(checkpoint["cortex_phase_state"]["memory_state"]["recent"]), 0)
                self.assertGreater(len(checkpoint["cortex_phase_state"]["sleep_state"]["replay_examples"]), 0)
                self.assertGreater(len(checkpoint["cortex_phase_state"]["sleep_state"]["synthetic_examples"]), 0)
                improvement_archive = checkpoint["cortex_phase_state"]["improvement_state"]["archive"]
                self.assertGreater(
                    improvement_archive["accepted_count"] + improvement_archive["rejected_count"],
                    0,
                )
                sidecar = json.loads((run_dir / "checkpoint_final.pt.json").read_text(encoding="utf-8"))
                self.assertTrue(sidecar["cortex_phase_state_present"])
                self.assertTrue(
                    sidecar["cortex_phase_state_summary"]["architecture_audit"]["passed"],
                    sidecar["cortex_phase_state_summary"]["architecture_audit"],
                )
                self.assertTrue(
                    sidecar["cortex_phase_state_summary"]["phase_deliverable_audit"]["passed"],
                    sidecar["cortex_phase_state_summary"]["phase_deliverable_audit"],
                )
                self.assertGreater(sidecar["cortex_phase_state_summary"]["replay_batch_count"], 0)
                self.assertGreater(sidecar["cortex_phase_state_summary"]["objective_feedback_events"], 0)
                self.assertGreater(sidecar["cortex_phase_state_summary"]["future_contract_decisions"], 0)
                self.assertGreater(
                    sidecar["cortex_phase_state_summary"]["compression_trace_counts"]["layer_forward_events"],
                    0,
                )
                self.assertGreater(
                    sidecar["cortex_phase_state_summary"]["phase_event_counts"]["P2"],
                    0,
                )
                self.assertGreater(
                    sidecar["cortex_phase_state_summary"]["compression_trace_counts"]["expert_activations"],
                    0,
                )
                self.assertGreater(sidecar["cortex_phase_state_summary"]["variable_input_compression_events"], 0)
                self.assertGreater(sidecar["cortex_phase_state_summary"]["certificate_head_forward_events"], 0)
                self.assertGreater(sidecar["cortex_phase_state_summary"]["input_anchor_observations"], 0)
                self.assertGreater(sidecar["cortex_phase_state_summary"]["input_anchor_count"], 0)
                self.assertEqual(sidecar["cortex_phase_state_summary"]["input_anchor_fidelity_failures"], 0)
                self.assertGreater(sidecar["cortex_phase_state_summary"]["bit_ledger_total_effective_bits"], 0.0)
                self.assertGreater(sidecar["cortex_phase_state_summary"]["skill_ledger_states"], 0)
                self.assertGreater(sidecar["cortex_phase_state_summary"]["causal_ledger_traces"], 0)
                self.assertGreater(sidecar["cortex_phase_state_summary"]["uncertainty_ledger_observations"], 0)
                self.assertGreater(sidecar["cortex_phase_state_summary"]["memory_recent_segments"], 0)
                self.assertGreater(sidecar["cortex_phase_state_summary"]["sleep_replay_examples"], 0)
                self.assertGreater(
                    sidecar["cortex_phase_state_summary"]["improvement_archive_accepted"]
                    + sidecar["cortex_phase_state_summary"]["improvement_archive_rejected"],
                    0,
                )

                resumed = LLMTrainer(
                    CortexTransformerLM(model_config),
                    train,
                    val,
                    TrainingConfig(
                        steps=2,
                        batch_size=2,
                        eval_interval=1,
                        eval_batches=1,
                        checkpoint_interval=1,
                        seed=71,
                        resume=True,
                        num_threads=1,
                        cortex_phase_interval=1,
                        cortex_phase_probe_tasks=1,
                        cortex_phase_max_proposals=1,
                    ),
                    run_dir=run_dir,
                    model_kind="cortex3_multi_horizon",
                    corpus_identity=manifest.identity(),
                ).train(name="cortex3")
            finally:
                train.close()
                val.close()

            resumed_influence = resumed.cortex_phase_report["training_influence"]
            self.assertEqual(resumed.start_step, 1)
            self.assertGreaterEqual(resumed_influence["phase_replay_examples"], first_influence["phase_replay_examples"])
            self.assertGreater(resumed_influence["sleep_replay_updates"], first_influence["sleep_replay_updates"])
            self.assertGreaterEqual(
                resumed_influence["objective_feedback_events"],
                first_influence["objective_feedback_events"],
            )
            self.assertGreater(resumed_influence["objective_feedback_scale"], 1.0)
            self.assertGreaterEqual(
                resumed_influence["future_contract_decisions"],
                first_influence["future_contract_decisions"],
            )
            self.assertGreaterEqual(
                resumed_influence["ternary_core_forward_events"],
                first_influence["ternary_core_forward_events"],
            )
            self.assertGreaterEqual(
                resumed_influence["skill_expert_activations"],
                first_influence["skill_expert_activations"],
            )
            self.assertGreaterEqual(
                resumed_influence["variable_input_compression_events"],
                first_influence["variable_input_compression_events"],
            )
            self.assertGreaterEqual(
                resumed_influence["certificate_head_forward_events"],
                first_influence["certificate_head_forward_events"],
            )
            self.assertGreaterEqual(
                resumed_influence["input_anchor_observations"],
                first_influence["input_anchor_observations"],
            )
            self.assertGreaterEqual(
                resumed_influence["input_anchor_count"],
                first_influence["input_anchor_count"],
            )
            self.assertEqual(resumed_influence["input_anchor_fidelity_failures"], 0)
            self.assertGreaterEqual(
                resumed_influence["bit_ledger_total_effective_bits"],
                first_influence["bit_ledger_total_effective_bits"],
            )
            self.assertGreaterEqual(
                resumed_influence["skill_ledger_states"],
                first_influence["skill_ledger_states"],
            )
            self.assertGreaterEqual(
                resumed_influence["causal_ledger_traces"],
                first_influence["causal_ledger_traces"],
            )
            self.assertGreaterEqual(
                resumed_influence["uncertainty_ledger_observations"],
                first_influence["uncertainty_ledger_observations"],
            )
            self.assertGreaterEqual(
                resumed_influence["memory_recent_segments"],
                first_influence["memory_recent_segments"],
            )
            self.assertGreaterEqual(
                resumed_influence["sleep_replay_examples"],
                first_influence["sleep_replay_examples"],
            )
            self.assertGreaterEqual(
                resumed_influence["improvement_archive_accepted"] + resumed_influence["improvement_archive_rejected"],
                first_influence["improvement_archive_accepted"] + first_influence["improvement_archive_rejected"],
            )

    def test_inspect_experiment_reports_partial_run_without_loading_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            corpus_dir = run / "corpus_matrix" / "c4-en" / "corpus"
            seed_dir = run / "corpus_matrix" / "c4-en" / "seed_11"
            baseline_dir = seed_dir / "baseline_ntp"
            cortex_dir = seed_dir / "cortex3"
            baseline_dir.mkdir(parents=True)
            cortex_dir.mkdir(parents=True)
            corpus_dir.mkdir(parents=True)

            (run / "experiment_manifest.normalized.json").write_text(
                json.dumps({"name": "inspect", "out_dir": str(run), "seeds": [11]}),
                encoding="utf-8",
            )
            (corpus_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "token_count": 64_000_000,
                        "seq_len": 1024,
                        "max_horizon": 8,
                        "preparation_config": {"max_tokens": 64_000_000},
                    }
                ),
                encoding="utf-8",
            )
            (seed_dir / "run_plan.json").write_text(json.dumps({"training": {"steps": 32_000}}), encoding="utf-8")
            (baseline_dir / "checkpoint_step_500.pt").write_bytes(b"checkpoint-placeholder")
            (baseline_dir / "checkpoint_step_1000.pt").write_bytes(b"checkpoint-placeholder-newer")
            (baseline_dir / "learning_curve.csv").write_text(
                "step,split,loss,next_token_loss,token_accuracy,mtp_loss,future_tokens_per_cost\n"
                "0,val,1.0,1.0,0.1,0.0,0.01\n"
                "500,val,0.8,0.8,0.2,0.0,0.02\n",
                encoding="utf-8",
            )
            (cortex_dir / "cortex_phase_report.json").write_text(
                json.dumps(
                    {
                        "all_phases_active": True,
                        "phase_event_counts": {f"P{index}": 1 for index in range(1, 11)},
                        "training_influence": {"sleep_replay_updates": 3},
                        "architecture_audit": {"passed": True, "failed_checks": []},
                        "phase_deliverable_audit": {"passed": True, "failed_checks": []},
                        "errors": [],
                    }
                ),
                encoding="utf-8",
            )

            report = inspect_llm_experiment(run)
            payload = report.to_dict()

            self.assertTrue(payload["exists"], payload)
            self.assertEqual(payload["status"], "partial", payload)
            self.assertEqual(payload["manifest"]["name"], "inspect")
            corpus = payload["corpora"][0]
            self.assertEqual(corpus["token_count"], 64_000_000)
            seed = corpus["seed_runs"][0]
            self.assertTrue(seed["run_plan_exists"])
            self.assertEqual(seed["baseline"]["latest_checkpoint_step"], 1000)
            self.assertTrue(seed["baseline"]["latest_checkpoint"]["path"].endswith("checkpoint_step_1000.pt"))
            self.assertEqual(seed["baseline"]["last_validation"]["step"], 500)
            self.assertTrue(seed["cortex"]["cortex_phase_summary"]["all_phases_active"])
            self.assertTrue(seed["cortex"]["cortex_phase_summary"]["architecture_audit"]["passed"])
            self.assertTrue(seed["cortex"]["cortex_phase_summary"]["phase_deliverable_audit"]["passed"])
            self.assertEqual(seed["cortex"]["cortex_phase_summary"]["training_influence"]["sleep_replay_updates"], 3)

    def test_trainer_rejects_resume_when_corpus_identity_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_a = root / "a.txt"
            source_b = root / "b.txt"
            source_a.write_text(
                ("cortex corpus alpha preserves one verified token stream for checkpoint identity.\n" * 80),
                encoding="utf-8",
            )
            source_b.write_text(
                ("cortex corpus beta changes the token stream and must reject checkpoint resume.\n" * 80),
                encoding="utf-8",
            )
            corpus_a = TextCorpusConfig.from_paths([source_a], min_chars_per_chunk=512)
            tokenizer = LLMTokenizer.train(corpus_a, vocab_size=192, min_frequency=1)
            manifest_a = TokenizedCorpusBuilder(corpus_a, tokenizer).build(
                root / "prepared-a",
                seq_len=16,
                max_horizon=2,
            )
            model_config = TransformerConfig(
                vocab_size=manifest_a.vocab_size,
                seq_len=16,
                d_model=32,
                n_heads=4,
                n_layers=1,
                dropout=0.0,
                horizons=(1, 2),
                use_cortex_heads=True,
            )
            train_a = MemmapCausalDataset(manifest_a, split="train")
            val_a = MemmapCausalDataset(manifest_a, split="val")
            identity_a = manifest_a.identity()
            run_dir = root / "resume-run"
            try:
                LLMTrainer(
                    CortexTransformerLM(model_config),
                    train_a,
                    val_a,
                    TrainingConfig(
                        steps=1,
                        batch_size=2,
                        eval_interval=1,
                        eval_batches=1,
                        checkpoint_interval=1,
                        seed=41,
                        num_threads=1,
                    ),
                    run_dir=run_dir,
                    model_kind="cortex3_multi_horizon",
                    corpus_identity=identity_a,
                ).train(name="cortex3")
            finally:
                train_a.close()
                val_a.close()

            corpus_b = TextCorpusConfig.from_paths([source_b], min_chars_per_chunk=512)
            manifest_b = TokenizedCorpusBuilder(corpus_b, tokenizer).build(
                root / "prepared-b",
                seq_len=16,
                max_horizon=2,
            )
            identity_b = manifest_b.identity()
            self.assertNotEqual(identity_a["identity_sha256"], identity_b["identity_sha256"])
            train_b = MemmapCausalDataset(manifest_b, split="train")
            val_b = MemmapCausalDataset(manifest_b, split="val")
            try:
                with self.assertRaisesRegex(ValueError, "corpus_identity"):
                    LLMTrainer(
                        CortexTransformerLM(model_config),
                        train_b,
                        val_b,
                        TrainingConfig(
                            steps=2,
                            batch_size=2,
                            eval_interval=1,
                            eval_batches=1,
                            checkpoint_interval=1,
                            seed=41,
                            resume=True,
                            num_threads=1,
                        ),
                        run_dir=run_dir,
                        model_kind="cortex3_multi_horizon",
                        corpus_identity=identity_b,
                    ).train(name="cortex3")
            finally:
                train_b.close()
                val_b.close()

    def test_hf_dataset_export_builds_real_text_shards_and_token_memmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "dataset.jsonl"
            with jsonl.open("w", encoding="utf-8") as handle:
                for index in range(36):
                    handle.write(
                        json.dumps(
                            {
                                "text": (
                                    f"document {index:03d} carries a stable corpus token stream. "
                                    f"cortex validates anchors and next-token baselines in shard {index % 4}."
                                )
                            }
                        )
                        + "\n"
                    )
            export_config = HFDatasetExportConfig(
                dataset="json",
                split="train",
                text_field="text",
                data_files=(str(jsonl),),
                streaming=True,
                max_documents=30,
                shard_max_chars=768,
                min_text_chars=20,
            )
            export_report = HFDatasetTextExporter(export_config).export(root / "hf")
            self.assertEqual(export_report.document_count, 30)
            self.assertEqual(export_report.truncated_reason, "max_documents")
            self.assertGreaterEqual(export_report.shard_count, 2)
            for shard in export_report.shard_files:
                self.assertTrue(Path(shard).exists(), shard)
            self.assertTrue((root / "hf" / "hf_export_report.json").exists())
            with patch("datasets.load_dataset", side_effect=AssertionError("resume should not reload dataset")):
                resumed_export = HFDatasetTextExporter(export_config).export(root / "hf", resume=True)
            self.assertEqual(resumed_export.to_dict(), export_report.to_dict())

            corpus = TextCorpusConfig.from_paths(export_report.shard_files, min_chars_per_chunk=128)
            tokenizer = LLMTokenizer.train(corpus, vocab_size=192, min_frequency=1)
            manifest = TokenizedCorpusBuilder(corpus, tokenizer).build(
                root / "hf" / "tokenized",
                seq_len=24,
                max_horizon=4,
            )
            self.assertGreater(manifest.token_count, 24)
            self.assertEqual(manifest.source_files, export_report.shard_files)
            with MemmapCausalDataset(manifest, split="train") as train:
                x, y, future = train.item(0)
                self.assertEqual(tuple(x.shape), (24,))
                self.assertEqual(tuple(y.shape), (24,))
                self.assertEqual(tuple(future.shape), (24, 4))

    def test_hf_dataset_export_resume_rejects_incomplete_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "dataset.jsonl"
            with jsonl.open("w", encoding="utf-8") as handle:
                for index in range(12):
                    handle.write(json.dumps({"text": f"document {index:03d} resume integrity shard check."}) + "\n")
            export_config = HFDatasetExportConfig(
                dataset="json",
                split="train",
                text_field="text",
                data_files=(str(jsonl),),
                streaming=True,
                max_documents=12,
                shard_max_chars=512,
            )
            export_report = HFDatasetTextExporter(export_config).export(root / "hf")
            Path(export_report.shard_files[0]).unlink()

            with self.assertRaises(FileNotFoundError):
                HFDatasetTextExporter(export_config).export(root / "hf", resume=True)

    def test_prepare_hf_resume_reuses_tokenized_manifest_without_reloading_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "dataset.jsonl"
            with jsonl.open("w", encoding="utf-8") as handle:
                for index in range(24):
                    handle.write(
                        json.dumps(
                            {
                                "text": (
                                    f"resume document {index:03d} keeps the prepared token memmap stable. "
                                    f"checkpoint identity must survive repeated prepare-hf commands."
                                )
                            }
                        )
                        + "\n"
                    )
            out_dir = root / "prepared"
            args = [
                "prepare-hf",
                "--dataset",
                "json",
                "--data-file",
                str(jsonl),
                "--out-dir",
                str(out_dir),
                "--max-documents",
                "24",
                "--shard-chars",
                "768",
                "--vocab-size",
                "192",
                "--min-frequency",
                "1",
                "--seq-len",
                "24",
                "--max-horizon",
                "4",
            ]
            with redirect_stdout(io.StringIO()):
                llm_main(args)
            token_path = out_dir / "tokenized" / "tokens.uint32"
            token_mtime = token_path.stat().st_mtime_ns
            first_report = json.loads((out_dir / "prepare_report.json").read_text(encoding="utf-8"))

            with patch("datasets.load_dataset", side_effect=AssertionError("resume should reuse HF export report")):
                with redirect_stdout(io.StringIO()):
                    llm_main([*args, "--resume"])

            self.assertEqual(token_path.stat().st_mtime_ns, token_mtime)
            resumed_report = json.loads((out_dir / "prepare_report.json").read_text(encoding="utf-8"))
            self.assertEqual(resumed_report["manifest"], first_report["manifest"])
            self.assertEqual(resumed_report["tokenization"], first_report["tokenization"])

            changed_tokenizer_args = list(args)
            changed_tokenizer_args[changed_tokenizer_args.index("--min-frequency") + 1] = "2"
            with patch("datasets.load_dataset", side_effect=AssertionError("resume should validate before reload")):
                with self.assertRaisesRegex(ValueError, "tokenization config"):
                    with redirect_stdout(io.StringIO()):
                        llm_main([*changed_tokenizer_args, "--resume"])

    def test_hf_dataset_export_namespaced_id_error_is_actionable(self):
        config = HFDatasetExportConfig(
            dataset="wikitext",
            split="train",
            text_field="text",
            streaming=True,
        )
        with patch("datasets.load_dataset", side_effect=ValueError("Repository id must be 'namespace/name', got 'wikitext'.")):
            with self.assertRaisesRegex(RuntimeError, "Salesforce/wikitext"):
                HFDatasetTextExporter(config)._load_dataset()

    def test_cortex_comparison_produces_checkpoints_curves_and_cost_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._corpus(root, repeats=120)
            config = ComparisonConfig(
                vocab_size=256,
                min_frequency=1,
                seq_len=32,
                d_model=64,
                n_heads=4,
                n_layers=2,
                dropout=0.0,
                horizons=(1, 2, 4),
                training=TrainingConfig(
                    steps=48,
                    batch_size=8,
                    eval_interval=16,
                    eval_batches=3,
                    seed=17,
                    num_threads=1,
                ),
                cortex_win_margin=1.02,
                max_next_token_loss_regression=1.50,
            )
            report = LLMComparisonRunner(corpus, config, run_dir=root / "run").run(require_win=True)
            proof = report.proof
            self.assertTrue(proof["passed"], proof)
            self.assertGreater(proof["baseline_score"], 0.0)
            self.assertGreater(proof["cortex_over_baseline_ratio"], 1.02)
            self.assertTrue(proof["learning_curve_audit_passed"], proof)
            for rel in [
                "run_plan.json",
                "learning_curve_audit.json",
                "comparison_report.json",
                "report.md",
                "learning_curve.png",
                "baseline_ntp/checkpoint_final.pt",
                "cortex3/checkpoint_final.pt",
                "baseline_ntp/learning_curve.csv",
                "cortex3/learning_curve.csv",
            ]:
                self.assertTrue((root / "run" / rel).exists(), rel)
            plan = json.loads((root / "run" / "run_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["training"]["tokens_per_optimizer_step"], 8 * 32)
            self.assertEqual(plan["training"]["planned_train_tokens"], 8 * 32 * 48)
            self.assertEqual(report.plan["corpus"]["token_count"], plan["corpus"]["token_count"])
            curve_audit = json.loads((root / "run" / "learning_curve_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(curve_audit["passed"], curve_audit)
            self.assertEqual(curve_audit["baseline"]["expected_final_step"], 48)
            self.assertGreaterEqual(curve_audit["baseline"]["validation_point_count"], 2)
            self.assertGreaterEqual(curve_audit["cortex"]["validation_point_count"], 2)
            self.assertTrue(report.curve_audit["passed"])

    def test_comparison_matrix_reuses_shared_corpus_across_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._corpus(root, repeats=120)
            config = ComparisonConfig(
                vocab_size=256,
                min_frequency=1,
                seq_len=32,
                d_model=64,
                n_heads=4,
                n_layers=2,
                dropout=0.0,
                horizons=(1, 2, 4),
                training=TrainingConfig(
                    steps=48,
                    batch_size=8,
                    eval_interval=16,
                    eval_batches=3,
                    seed=17,
                    precision="bf16",
                    num_threads=1,
                ),
                cortex_win_margin=1.02,
                max_next_token_loss_regression=1.60,
            )
            report = LLMComparisonMatrixSuite(
                corpus,
                config,
                run_dir=root / "compare-matrix",
                seeds=(17, 29),
            ).run(require_win=True)
            self.assertTrue(report.proof["passed"], report.proof)
            self.assertEqual(report.proof["seed_count"], 2)
            self.assertEqual(report.proof["sample_count"], 2)
            self.assertEqual(report.proof["win_rate"], 1.0)
            self.assertTrue((root / "compare-matrix" / "corpus" / "manifest.json").exists())
            self.assertTrue((root / "compare-matrix" / "comparison_matrix_report.json").exists())
            self.assertTrue((root / "compare-matrix" / "comparison_matrix_report.md").exists())
            self.assertTrue((root / "compare-matrix" / "comparison_matrix_ratios.png").exists())
            self.assertTrue((root / "compare-matrix" / "comparison_matrix_learning_curves.csv").exists())
            self.assertTrue((root / "compare-matrix" / "comparison_matrix_learning_curves.png").exists())
            for seed in (17, 29):
                self.assertFalse((root / "compare-matrix" / f"seed_{seed}" / "corpus" / "manifest.json").exists())
                self.assertTrue((root / "compare-matrix" / f"seed_{seed}" / "comparison_report.json").exists())
                self.assertTrue((root / "compare-matrix" / f"seed_{seed}" / "baseline_ntp" / "checkpoint_final.pt").exists())
                self.assertTrue((root / "compare-matrix" / f"seed_{seed}" / "cortex3" / "checkpoint_final.pt").exists())

    def test_comparison_matrix_resume_rejects_mismatched_tokenized_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._corpus(root, repeats=80)
            initial = ComparisonConfig(
                vocab_size=192,
                min_frequency=1,
                seq_len=24,
                d_model=32,
                n_heads=4,
                n_layers=1,
                horizons=(1, 2),
                training=TrainingConfig(
                    steps=1,
                    batch_size=2,
                    eval_interval=1,
                    eval_batches=1,
                    checkpoint_interval=1,
                    num_threads=1,
                ),
            )
            LLMComparisonMatrixSuite(
                corpus,
                initial,
                run_dir=root / "matrix",
                seeds=(3,),
            ).run(require_win=False)

            resumed = replace(
                initial,
                seq_len=32,
                training=replace(initial.training, resume=True),
            )
            with self.assertRaisesRegex(ValueError, "tokenized corpus preparation config"):
                LLMComparisonMatrixSuite(
                    corpus,
                    resumed,
                    run_dir=root / "matrix",
                    seeds=(3,),
                ).run(require_win=False)

    def test_multi_domain_benchmark_aggregates_real_learning_curves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ComparisonConfig(
                vocab_size=256,
                min_frequency=1,
                seq_len=32,
                d_model=64,
                n_heads=4,
                n_layers=2,
                dropout=0.0,
                horizons=(1, 2, 4),
                training=TrainingConfig(
                    steps=48,
                    batch_size=8,
                    eval_interval=16,
                    eval_batches=2,
                    seed=23,
                    precision="bf16",
                    num_threads=1,
                ),
                cortex_win_margin=1.02,
                max_next_token_loss_regression=1.60,
            )
            report = LLMBenchmarkSuite(
                run_dir=root / "benchmark",
                domains=("sequence", "anchors"),
                repeats=96,
                config=config,
            ).run(require_win=True)
            self.assertTrue(report.proof["passed"], report.proof)
            self.assertEqual(report.proof["domain_count"], 2)
            self.assertGreater(report.proof["mean_baseline_score"], 0.0)
            self.assertTrue((root / "benchmark" / "benchmark_report.json").exists())
            self.assertTrue((root / "benchmark" / "benchmark_ratios.png").exists())

    def test_statistical_benchmark_matrix_requires_all_seed_domain_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ComparisonConfig(
                vocab_size=256,
                min_frequency=1,
                seq_len=32,
                d_model=64,
                n_heads=4,
                n_layers=2,
                dropout=0.0,
                horizons=(1, 2, 4),
                training=TrainingConfig(
                    steps=48,
                    batch_size=8,
                    eval_interval=16,
                    eval_batches=2,
                    seed=11,
                    precision="bf16",
                    num_threads=1,
                ),
                cortex_win_margin=1.02,
                max_next_token_loss_regression=1.60,
            )
            report = LLMStatisticalBenchmarkSuite(
                run_dir=root / "benchmark-matrix",
                domains=("sequence", "anchors"),
                seeds=(11, 23),
                repeats=96,
                config=config,
            ).run(require_win=True)
            self.assertTrue(report.proof["passed"], report.proof)
            self.assertEqual(report.proof["domain_count"], 2)
            self.assertEqual(report.proof["seed_count"], 2)
            self.assertEqual(report.proof["sample_count"], 4)
            self.assertEqual(report.proof["win_rate"], 1.0)
            self.assertGreater(report.proof["mean_baseline_score"], 0.0)
            self.assertTrue((root / "benchmark-matrix" / "statistical_benchmark_report.json").exists())
            self.assertTrue((root / "benchmark-matrix" / "statistical_benchmark_report.md").exists())
            self.assertTrue((root / "benchmark-matrix" / "statistical_benchmark_ratios.png").exists())
            for seed in (11, 23):
                for domain in ("sequence", "anchors"):
                    self.assertTrue(
                        (root / "benchmark-matrix" / f"seed_{seed}" / domain / "comparison_report.json").exists(),
                        f"missing comparison report for seed={seed} domain={domain}",
                    )

    def test_corpus_matrix_aggregates_multiple_corpora_and_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_corpus = TextCorpusConfig.from_paths(
                build_seed_corpus(root / "seed-corpus", repeats=120),
                min_chars_per_chunk=512,
            )
            anchor_corpus = TextCorpusConfig.from_paths(
                build_benchmark_corpus(root / "anchor-corpus", domain="anchors", repeats=120),
                min_chars_per_chunk=512,
            )
            config = ComparisonConfig(
                vocab_size=256,
                min_frequency=1,
                seq_len=32,
                d_model=64,
                n_heads=4,
                n_layers=2,
                dropout=0.0,
                horizons=(1, 2, 4),
                training=TrainingConfig(
                    steps=48,
                    batch_size=8,
                    eval_interval=16,
                    eval_batches=2,
                    seed=17,
                    precision="bf16",
                    num_threads=1,
                ),
                cortex_win_margin=1.02,
                max_next_token_loss_regression=1.60,
            )
            report = LLMCorpusMatrixSuite(
                (("seed", seed_corpus), ("anchors", anchor_corpus)),
                config,
                run_dir=root / "corpus-matrix",
                seeds=(17, 29),
            ).run(require_win=True)
            self.assertTrue(report.proof["passed"], report.proof)
            self.assertEqual(report.proof["corpus_count"], 2)
            self.assertEqual(report.proof["seed_count"], 2)
            self.assertEqual(report.proof["sample_count"], 4)
            self.assertEqual(report.proof["win_rate"], 1.0)
            self.assertGreater(report.proof["mean_baseline_score"], 0.0)
            self.assertTrue((root / "corpus-matrix" / "corpus_matrix_report.json").exists())
            self.assertTrue((root / "corpus-matrix" / "corpus_matrix_report.md").exists())
            self.assertTrue((root / "corpus-matrix" / "corpus_matrix_ratios.png").exists())
            self.assertTrue((root / "corpus-matrix" / "corpus_matrix_learning_curves.csv").exists())
            self.assertTrue((root / "corpus-matrix" / "corpus_matrix_learning_curves.png").exists())
            for corpus in ("seed", "anchors"):
                self.assertTrue((root / "corpus-matrix" / corpus / "comparison_matrix_report.json").exists())
                self.assertTrue((root / "corpus-matrix" / corpus / "corpus" / "manifest.json").exists())
                for seed in (17, 29):
                    self.assertTrue(
                        (root / "corpus-matrix" / corpus / f"seed_{seed}" / "comparison_report.json").exists(),
                        f"missing comparison report for corpus={corpus} seed={seed}",
                    )
                    self.assertTrue((root / "corpus-matrix" / corpus / f"seed_{seed}" / "cortex3" / "checkpoint_final.pt").exists())

    def test_manifest_experiment_runs_doctor_prepare_and_corpus_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "manifest_hf.jsonl"
            with jsonl.open("w", encoding="utf-8") as handle:
                for index in range(96):
                    handle.write(
                        json.dumps(
                            {
                                "text": (
                                    f"alpha beta gamma delta sample {index:03d}. "
                                    f"cortex manifest experiment keeps token pattern {index % 5}."
                                )
                            }
                        )
                        + "\n"
                    )
            seed_files = build_seed_corpus(root / "manifest-paths", repeats=120)
            manifest = {
                "name": "unit-manifest-experiment",
                "out_dir": str(root / "experiment"),
                "doctor": {"precision": "bf16", "device": "auto", "require_cuda": False},
                "seeds": [17, 29],
                "require_win": True,
                "model": {
                    "vocab_size": 256,
                    "min_frequency": 1,
                    "seq_len": 32,
                    "d_model": 64,
                    "n_heads": 4,
                    "n_layers": 2,
                    "dropout": 0.0,
                    "horizons": [1, 2, 4],
                    "cortex_win_margin": 1.02,
                    "max_next_token_loss_regression": 1.60,
                },
                "training": {
                    "steps": 48,
                    "batch_size": 8,
                    "eval_interval": 16,
                    "eval_batches": 2,
                    "checkpoint_interval": 100,
                    "resume_if_exists": True,
                    "num_threads": 1,
                },
                "corpora": [
                    {
                        "name": "hfjson",
                        "kind": "hf",
                        "dataset": "json",
                        "data_files": [str(jsonl)],
                        "split": "train",
                        "text_field": "text",
                        "max_documents": 80,
                        "min_text_chars": 20,
                        "shard_max_chars": 1024,
                        "min_chars_per_chunk": 256,
                    },
                    {
                        "name": "paths",
                        "kind": "paths",
                        "paths": list(seed_files),
                        "min_chars_per_chunk": 512,
                    },
                ],
            }
            manifest_path = root / "experiment_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8-sig")
            report = LLMExperimentRunner.load(manifest_path).run()
            self.assertTrue(report.proof["passed"], report.proof)
            self.assertEqual(report.proof["corpus_count"], 2)
            self.assertEqual(report.proof["seed_count"], 2)
            self.assertEqual(report.proof["sample_count"], 4)
            self.assertEqual(report.proof["win_rate"], 1.0)
            self.assertTrue((root / "experiment" / "experiment_manifest.normalized.json").exists())
            self.assertTrue((root / "experiment" / "doctor_report.json").exists())
            self.assertTrue((root / "experiment" / "preflight_report.json").exists())
            self.assertTrue((root / "experiment" / "experiment_report.json").exists())
            self.assertTrue((root / "experiment" / "experiment_report.md").exists())
            self.assertTrue((root / "experiment" / "prepared" / "hfjson" / "hf_export_report.json").exists())
            self.assertTrue((root / "experiment" / "corpus_matrix" / "corpus_matrix_report.json").exists())
            self.assertTrue((root / "experiment" / "corpus_matrix" / "corpus_matrix_learning_curves.csv").exists())
            self.assertTrue((root / "experiment" / "corpus_matrix" / "corpus_matrix_learning_curves.png").exists())
            audit = audit_llm_experiment_artifacts(root / "experiment")
            self.assertTrue(audit.passed, audit.failed_checks)
            self.assertFalse(audit.failed_checks)
            self.assertGreater(len(audit.checked_artifacts), 20)
            with redirect_stdout(io.StringIO()):
                llm_main(["audit-experiment", str(root / "experiment")])
            with redirect_stdout(io.StringIO()):
                llm_main(["preflight-experiment", str(manifest_path), "--out-dir", str(root / "preflight-only")])
            preflight_only = json.loads((root / "preflight-only" / "preflight_report.json").read_text(encoding="utf-8"))
            self.assertTrue(preflight_only["passed"], preflight_only)

            with patch("datasets.load_dataset", side_effect=AssertionError("auto-resume should reuse the HF export report")):
                resumed_report = LLMExperimentRunner.load(manifest_path).run()
            self.assertTrue(resumed_report.proof["passed"], resumed_report.proof)
            resumed_training = json.loads(
                (
                    root
                    / "experiment"
                    / "corpus_matrix"
                    / "hfjson"
                    / "seed_17"
                    / "baseline_ntp"
                    / "training_report.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(resumed_training["start_step"], 48)
            self.assertEqual(resumed_training["optimizer_steps"], 0)
            self.assertTrue(resumed_training["resumed_from"].endswith("checkpoint_final.pt"))

            missing_checkpoint = root / "experiment" / "corpus_matrix" / "hfjson" / "seed_17" / "cortex3" / "checkpoint_final.pt"
            missing_checkpoint.unlink()
            failed_audit = audit_llm_experiment_artifacts(root / "experiment")
            self.assertFalse(failed_audit.passed)
            self.assertTrue(any("checkpoint_final.pt" in item for item in failed_audit.failed_checks))

    def test_experiment_preflight_rejects_oversized_cuda_manifest(self):
        manifest = {
            "name": "oversized-cuda",
            "out_dir": "runs/oversized-cuda",
            "doctor": {"precision": "bf16", "device": "cuda", "require_cuda": True},
            "seeds": [1],
            "require_win": True,
            "model": {
                "vocab_size": 32768,
                "min_frequency": 1,
                "seq_len": 1024,
                "d_model": 768,
                "n_heads": 12,
                "n_layers": 12,
                "horizons": [1, 2, 4, 8],
            },
            "training": {"steps": 10, "batch_size": 16, "gradient_accumulation_steps": 1},
            "corpora": [{"name": "paths", "kind": "paths", "paths": ["README.md"]}],
        }
        runner = LLMExperimentRunner(manifest)
        fake_hardware = {
            "torch": "test",
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_current_device": 0,
            "cuda_current_device_name": "tiny-test-gpu",
            "cuda_current_device_total_memory_bytes": 2 * 1024 * 1024 * 1024,
            "cuda_current_device_free_memory_bytes": 2 * 1024 * 1024 * 1024,
            "cuda_devices": (),
            "distributed_available": True,
            "nccl_available": False,
            "gloo_available": True,
        }
        with patch("cortex3_llm.hardware_report", return_value=fake_hardware):
            report = runner.preflight(doctor_report={"device_type": "cuda"})
        self.assertFalse(report.passed, report.to_dict())
        self.assertTrue(any("cuda_memory_capacity_exceeded" in check for check in report.failed_checks))
        self.assertGreater(
            report.estimates["max_estimated_peak_training_bytes"],
            report.estimates["cuda_current_device_usable_memory_bytes"],
        )

    def test_ddp_launcher_detects_cuda_requests_in_args_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "cuda_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "cuda-ddp-preflight",
                        "out_dir": str(root / "out"),
                        "doctor": {"require_cuda": True, "device": "cuda", "precision": "bf16"},
                        "seeds": [1],
                        "corpora": [{"name": "paths", "kind": "paths", "paths": [str(root / "missing.txt")]}],
                    }
                ),
                encoding="utf-8-sig",
            )
            self.assertTrue(_manifest_requests_cuda(manifest_path))
            self.assertTrue(_train_args_request_cuda(["run-experiment", str(manifest_path)]))
            self.assertTrue(_train_args_request_cuda(["smoke", "--device", "cuda"]))
            self.assertFalse(_train_args_request_cuda(["smoke", "--device", "cpu"]))

    def test_cuda_requirement_is_explicit_not_silent_fallback(self):
        report = hardware_report()
        self.assertIn("distributed_available", report)
        with PrecisionPolicy("bf16").autocast("cpu"):
            pass
        with self.assertRaises(RuntimeError):
            DistributedRuntime.from_env(requested=True, device_type="cpu")
        if not torch.cuda.is_available():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                corpus = self._corpus(root, repeats=40)
                config = ComparisonConfig(
                    vocab_size=128,
                    min_frequency=1,
                    seq_len=16,
                    d_model=32,
                    n_heads=4,
                    n_layers=1,
                    horizons=(1, 2),
                    training=TrainingConfig(
                        steps=1,
                        batch_size=2,
                        eval_interval=1,
                        eval_batches=1,
                        require_cuda=True,
                        num_threads=1,
                    ),
                )
                with self.assertRaises(RuntimeError):
                    LLMComparisonRunner(corpus, config, run_dir=root / "cuda-required").run()

    def test_doctor_reports_dependency_and_cuda_readiness(self):
        report = llm_doctor_report(precision="bf16", device="auto", distributed=False)
        self.assertIn("dependencies", report)
        self.assertIn("torch", report["dependencies"])
        self.assertTrue(report["dependencies"]["torch"]["installed"])
        self.assertIn("hardware", report)
        self.assertIn("checks", report)
        self.assertTrue(report["passed"], report)
        if not torch.cuda.is_available():
            cuda_report = llm_doctor_report(require_cuda=True, precision="fp32", device="auto")
            self.assertFalse(cuda_report["passed"], cuda_report)
            failed_names = {check["name"] for check in cuda_report["failed_required_checks"]}
            self.assertIn("torch:require_cuda", failed_names)

    def test_distributed_runtime_can_pin_gloo_interface_without_initializing(self):
        if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
            self.skipTest("Gloo distributed runtime is not available")
        saved = {name: os.environ.get(name) for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GLOO_SOCKET_IFNAME")}
        try:
            os.environ["WORLD_SIZE"] = "2"
            os.environ["RANK"] = "1"
            os.environ["LOCAL_RANK"] = "1"
            os.environ.pop("GLOO_SOCKET_IFNAME", None)
            runtime = DistributedRuntime.from_env(requested=True, device_type="cpu", gloo_interface="test-iface")
            self.assertTrue(runtime.enabled)
            self.assertEqual(runtime.world_size, 2)
            self.assertEqual(runtime.rank, 1)
            self.assertEqual(runtime.gloo_interface, "test-iface")
            self.assertEqual(os.environ["GLOO_SOCKET_IFNAME"], "test-iface")
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
