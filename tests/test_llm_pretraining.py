import tempfile
import unittest
from pathlib import Path

import torch

from cortex3_llm import (
    ComparisonConfig,
    DistributedRuntime,
    LLMComparisonRunner,
    LLMTokenizer,
    MemmapCausalDataset,
    PrecisionPolicy,
    TextCorpusConfig,
    TokenizedCorpusBuilder,
    TrainingConfig,
    build_seed_corpus,
    hardware_report,
)


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
            train = MemmapCausalDataset(manifest, split="train")
            try:
                x, y, future = train.item(0)
                self.assertEqual(tuple(x.shape), (24,))
                self.assertEqual(tuple(y.shape), (24,))
                self.assertEqual(tuple(future.shape), (24, 4))
                self.assertEqual(int(y[0]), int(future[0, 0]))
            finally:
                train.close()

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


if __name__ == "__main__":
    unittest.main()
