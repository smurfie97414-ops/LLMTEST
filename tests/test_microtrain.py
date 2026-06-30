import tempfile
import unittest
from pathlib import Path

from cortex3 import CorruptedCompressedAgent, DynamicSkillVerifier, ReferenceRuleAgent, default_skill_specs
from cortex3_cycle import CortexCycle
from cortex3_microtrain import (
    CheckpointManager,
    CortexMicroTrainer,
    MicroModelAgent,
    examples_from_sleep_report,
    examples_from_tasks,
)
from cortex3_sleep import SleepPhaseConsolidator


class MicroTrainTest(unittest.TestCase):
    def test_micro_model_training_improves_verified_accuracy(self):
        verifier = DynamicSkillVerifier(default_skill_specs())
        tasks = verifier.build_suite(1, seed=3, include_metamorphic=False)
        examples = examples_from_tasks(tasks)

        model, training = CortexMicroTrainer().train(examples, epochs=120, lr=0.05)
        report = verifier.evaluate_tasks(MicroModelAgent(model), tasks)

        self.assertLess(training.before_accuracy, training.after_accuracy)
        self.assertLess(training.after_loss, training.before_loss)
        self.assertEqual(training.after_accuracy, 1.0)
        self.assertEqual(report.passed, report.total)
        self.assertTrue(model.ledger.compression_decisions)

    def test_checkpoint_save_and_load_preserves_predictions(self):
        verifier = DynamicSkillVerifier(default_skill_specs())
        tasks = verifier.build_suite(1, seed=5, include_metamorphic=False)
        examples = examples_from_tasks(tasks)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "micro.pt"
            model, training = CortexMicroTrainer().train(examples, epochs=100, lr=0.05, checkpoint_path=path)
            loaded = CheckpointManager().load(path)

        self.assertEqual(training.checkpoint_path, str(path))
        original = [MicroModelAgent(model)(task).text for task in tasks]
        restored = [MicroModelAgent(loaded)(task).text for task in tasks]
        self.assertEqual(original, restored)
        self.assertEqual(verifier.evaluate_tasks(MicroModelAgent(loaded), tasks).passed, len(tasks))

    def test_sleep_phase_examples_can_train_checkpoint(self):
        verifier = DynamicSkillVerifier(default_skill_specs())
        cycle = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=3, n_per_skill=1)
        sleep = SleepPhaseConsolidator(verifier).ingest_cycle(cycle, seed=3)
        examples = examples_from_sleep_report(sleep)[:12]

        model, training = CortexMicroTrainer().train(examples, epochs=140, lr=0.05)

        self.assertGreater(len(examples), 0)
        self.assertEqual(training.after_accuracy, 1.0)
        self.assertGreater(training.examples, 0)
        for example in examples:
            self.assertTrue(example.source.startswith("sleep:"))
        self.assertGreater(verifier.evaluate_tasks(MicroModelAgent(model), [example.task for example in examples]).aggregate_score, 0.9)


if __name__ == "__main__":
    unittest.main()
