import json
import tempfile
import unittest
from pathlib import Path

from cortex3 import CorruptedCompressedAgent, DynamicSkillVerifier, ReferenceRuleAgent, default_skill_specs
from cortex3_autoregressive import (
    ARCheckpointManager,
    ARConfig,
    ARDataset,
    ARDecoderAgent,
    ARMicroDecoder,
    ARTrainer,
    TokenVocabulary,
    ar_examples_from_sleep_report,
    ar_examples_from_tasks,
)
from cortex3_cycle import CortexCycle
from cortex3_reporting import write_cycle_run
from cortex3_sleep import SleepPhaseConsolidator
from tools.run_cycle_report import build_autoregressive_smoke


class AutoregressiveDecoderTest(unittest.TestCase):
    def _verifier(self):
        return DynamicSkillVerifier(default_skill_specs())

    def test_autoregressive_generation_improves_and_verifies_all_seed_tasks(self):
        verifier = self._verifier()
        tasks = verifier.build_suite(1, seed=3, include_metamorphic=False)
        examples = ar_examples_from_tasks(tasks)

        model, training = ARTrainer(ARConfig(hidden_size=96, token_dim=32)).train(examples, epochs=80, lr=0.03)
        agent = ARDecoderAgent(model)
        report = verifier.evaluate_tasks(agent, tasks)
        generated = [agent(task) for task in tasks]

        self.assertLess(training.before_token_accuracy, training.after_token_accuracy)
        self.assertLess(training.after_loss, training.before_loss)
        self.assertEqual(training.after_token_accuracy, 1.0)
        self.assertEqual(training.exact_sequence_accuracy, 1.0)
        self.assertEqual(report.passed, report.total)
        self.assertTrue(model.ledger.compression_decisions)
        self.assertTrue(model.ledger.activation_quantizations)
        for answer in generated:
            self.assertEqual(answer.certificate["autoregressive_decoder"], "trained")
            self.assertEqual(answer.certificate["compiled_circuit"], "ternary_bitlinear_autoregressive")
            self.assertGreater(answer.certificate["compiled_weight_bits"], 0.0)
            self.assertEqual(answer.certificate["distilled_from"], "verified_micro_examples")
            self.assertIn("oracle_verified_outputs", answer.certificate["strategy_invariants"])
            self.assertEqual(answer.certificate["cheap_verification"], "oracle_replay_contract")
            self.assertEqual(answer.certificate["mtp_horizons"], (1, 2, 4, 8))
            self.assertGreater(answer.cost.weight_bits_read, 0.0)
            self.assertGreater(answer.cost.activation_bits, 0.0)
            self.assertGreaterEqual(answer.cost.generated_tokens, 1)

    def test_multi_horizon_and_future_contract_losses_are_optimized(self):
        verifier = self._verifier()
        tasks = verifier.build_suite(1, seed=4, include_metamorphic=False)[:4]
        examples = ar_examples_from_tasks(tasks)
        config = ARConfig(hidden_size=96, token_dim=32)
        vocabulary = TokenVocabulary.from_texts(example.answer for example in examples)
        dataset = ARDataset(examples, vocabulary, config)
        trainer = ARTrainer(config)
        fresh_model = ARMicroDecoder(config, vocabulary)
        before = trainer.evaluate_teacher(fresh_model, dataset)

        model, training = trainer.train(examples, epochs=25, lr=0.03)
        trained_dataset = ARDataset(examples, model.vocabulary, model.config)
        after = trainer.evaluate_teacher(model, trained_dataset)

        self.assertIn("multi_horizon_loss", after)
        self.assertIn("future_contract_loss", after)
        self.assertLess(after["multi_horizon_loss"], before["multi_horizon_loss"])
        self.assertLessEqual(after["future_contract_loss"], before["future_contract_loss"])
        self.assertLess(training.after_loss, training.before_loss)
        self.assertEqual(after["token_accuracy"], 1.0)

    def test_future_contract_blocks_preserve_verified_autoregressive_generation(self):
        verifier = self._verifier()
        tasks = verifier.build_suite(1, seed=3, include_metamorphic=False)
        examples = ar_examples_from_tasks(tasks)
        model, _ = ARTrainer(ARConfig(hidden_size=96, token_dim=32)).train(examples, epochs=80, lr=0.03)

        plain_agent = ARDecoderAgent(model)
        contract_agent = ARDecoderAgent(model, use_future_contracts=True)
        plain_answers = [plain_agent(task).text for task in tasks]
        contract_answers = [contract_agent(task) for task in tasks]
        report = verifier.evaluate_tasks(contract_agent, tasks)
        sample = next(answer for task, answer in zip(tasks, contract_answers) if task.skill == "instruction_following")
        generation = sample.raw["future_contract_generation"]

        self.assertEqual([answer.text for answer in contract_answers], plain_answers)
        self.assertEqual(report.passed, report.total)
        self.assertGreater(generation["future_contracts"]["accepted"], 0)
        self.assertTrue(any(trace["accepted_horizon"] > 1 and trace["accepted"] for trace in generation["block_traces"]))
        self.assertGreater(generation["decoder_steps"], 0)
        self.assertAlmostEqual(sample.cost.weight_bits_read, model._compiled_weight_bits() * generation["decoder_steps"])
        self.assertTrue(model.ledger.mtp_fsp_events)
        self.assertIn("block_contracts", sample.certificate)

    def test_checkpoint_save_and_load_preserves_generated_answers(self):
        verifier = self._verifier()
        tasks = verifier.build_suite(1, seed=5, include_metamorphic=False)[:3]
        examples = ar_examples_from_tasks(tasks)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ar-decoder.pt"
            model, training = ARTrainer().train(examples, epochs=45, lr=0.03, checkpoint_path=path)
            loaded = ARCheckpointManager().load(path)

        original = [ARDecoderAgent(model)(task).text for task in tasks]
        restored = [ARDecoderAgent(loaded)(task).text for task in tasks]

        self.assertEqual(training.checkpoint_path, str(path))
        self.assertEqual(original, restored)
        self.assertEqual(verifier.evaluate_tasks(ARDecoderAgent(loaded), tasks).passed, len(tasks))

    def test_sleep_phase_examples_can_feed_autoregressive_training(self):
        verifier = self._verifier()
        cycle = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=3, n_per_skill=1)
        sleep = SleepPhaseConsolidator(verifier).ingest_cycle(cycle, seed=3)
        examples = ar_examples_from_sleep_report(sleep)[:6]

        model, training = ARTrainer().train(examples, epochs=35, lr=0.03)
        generated = [ARDecoderAgent(model)(example.task).text for example in examples]

        self.assertGreater(len(examples), 0)
        self.assertTrue(all(example.source.startswith("sleep:") for example in examples))
        self.assertGreater(training.after_token_accuracy, training.before_token_accuracy)
        self.assertGreaterEqual(training.exact_sequence_accuracy, 0.8)
        self.assertEqual(generated, [example.answer for example in examples])

    def test_cycle_report_persists_autoregressive_checkpoint_report(self):
        verifier = self._verifier()
        cycle = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=3, n_per_skill=1)
        ar_report = build_autoregressive_smoke(verifier, seed=3, n_per_skill=1, epochs=60)

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_cycle_run(cycle, output_dir=tmp, run_id="ar-run", autoregressive_report=ar_report)
            payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))

        checkpoint = payload["autoregressive_checkpoint"]
        self.assertEqual(checkpoint["training"]["exact_sequence_accuracy"], 1.0)
        self.assertEqual(checkpoint["dsv"]["passed"], checkpoint["dsv"]["total"])
        self.assertTrue(checkpoint["generated_samples"])
        self.assertTrue(checkpoint["inference"]["verification"]["passed"])
        self.assertTrue(checkpoint["inference"]["certificate_verified"])
        self.assertEqual(checkpoint["inference"]["answer"]["certificate"]["autoregressive_decoder"], "trained")
        self.assertIn("block_contracts", checkpoint["generated_samples"][0])
        self.assertIn("future_contract_generation", checkpoint["inference"]["answer"]["raw"])


if __name__ == "__main__":
    unittest.main()
