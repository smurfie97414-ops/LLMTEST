import json
import os
import tempfile
import unittest
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
    build_benchmark_corpus,
    build_seed_corpus,
    hardware_report,
    llm_doctor_report,
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
                ).train(name="cortex3")
                self.assertEqual(first.start_step, 0)
                self.assertEqual(first.optimizer_steps, 2)
                self.assertEqual(first.effective_batch_size, 4)
                self.assertTrue((run_dir / "checkpoint_step_1.pt").exists())
                self.assertTrue((run_dir / "checkpoint_final.pt").exists())

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
                ).train(name="cortex3")
                self.assertEqual(resumed.start_step, 2)
                self.assertEqual(resumed.optimizer_steps, 2)
                self.assertEqual(resumed.effective_batch_size, 4)
                self.assertEqual(resumed.final_val.step, 4)
                self.assertEqual(Path(resumed.resumed_from).name, "checkpoint_final.pt")
                checkpoint = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
                self.assertEqual(checkpoint["step"], 4)
                self.assertIn("rng_state", checkpoint)
                self.assertGreaterEqual(len(checkpoint["curve"]), len(first.curve))
            finally:
                train.close()
                val.close()

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
            for rel in [
                "comparison_report.json",
                "report.md",
                "learning_curve.png",
                "baseline_ntp/checkpoint_final.pt",
                "cortex3/checkpoint_final.pt",
                "baseline_ntp/learning_curve.csv",
                "cortex3/learning_curve.csv",
            ]:
                self.assertTrue((root / "run" / rel).exists(), rel)

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
            self.assertTrue((root / "experiment" / "experiment_report.json").exists())
            self.assertTrue((root / "experiment" / "experiment_report.md").exists())
            self.assertTrue((root / "experiment" / "prepared" / "hfjson" / "hf_export_report.json").exists())
            self.assertTrue((root / "experiment" / "corpus_matrix" / "corpus_matrix_report.json").exists())
            self.assertTrue((root / "experiment" / "corpus_matrix" / "corpus_matrix_learning_curves.csv").exists())
            self.assertTrue((root / "experiment" / "corpus_matrix" / "corpus_matrix_learning_curves.png").exists())

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
