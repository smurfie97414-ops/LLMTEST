import unittest

import torch

from cortex3 import Anchor, CandidateAnswer, CostTrace, DynamicSkillVerifier, ReferenceRuleAgent, Task, default_skill_specs
from cortex3_autoregressive import TokenVocabulary
from cortex3_future import (
    DEFAULT_MTP_HORIZONS,
    FutureContractEngine,
    MTPFSPCalibrator,
    MTPFSPConfig,
    MTPFSPHeads,
    future_examples_from_tasks,
    temporal_consistency_loss,
    temporal_consistency_loss_from_outputs,
    verified_answers_per_effective_cost,
)
from cortex3_ternary import CompressionTraceLedger


def _confident_heads() -> MTPFSPHeads:
    torch.manual_seed(0)
    heads = MTPFSPHeads(MTPFSPConfig(hidden_size=4, vocab_size=7))
    with torch.no_grad():
        heads.confidence_head.weight.zero_()
        heads.confidence_head.bias.fill_(5.0)
    return heads


class FutureContractsTest(unittest.TestCase):
    def test_mtp_heads_have_required_horizons_and_confidence(self):
        heads = _confident_heads()
        hidden = torch.zeros(2, 4)
        output = heads(hidden)
        self.assertEqual(tuple(sorted(output.logits_by_horizon)), DEFAULT_MTP_HORIZONS)
        for horizon in DEFAULT_MTP_HORIZONS:
            self.assertEqual(tuple(output.logits_by_horizon[horizon].shape), (2, horizon, 7))
        self.assertEqual(tuple(output.confidence.shape), (2,))
        self.assertGreater(float(output.confidence.detach().min()), 0.98)

    def test_temporal_consistency_loss_rewards_shifted_future_agreement(self):
        torch.manual_seed(1)
        previous = torch.randn(1, 4, 6)
        next_logits = torch.randn(1, 4, 6)
        next_logits[:, :3, :] = previous[:, 1:, :]
        self.assertAlmostEqual(float(temporal_consistency_loss(previous, next_logits)), 0.0, places=7)
        mismatched = torch.zeros_like(next_logits)
        self.assertGreater(float(temporal_consistency_loss(previous, mismatched)), 0.0)

    def test_temporal_consistency_loss_from_outputs_runs_on_real_heads(self):
        heads = _confident_heads()
        out_a = heads(torch.zeros(1, 4))
        out_b = heads(torch.ones(1, 4))
        loss = temporal_consistency_loss_from_outputs(out_a, out_b)
        self.assertEqual(tuple(loss.shape), ())
        self.assertGreaterEqual(float(loss.detach()), 0.0)

    def test_future_contract_accepts_low_risk_fast_path(self):
        engine = FutureContractEngine(MTPFSPConfig(hidden_size=4, vocab_size=7), heads=_confident_heads())
        contract = engine.draft_contract(torch.zeros(1, 4), domain="general", risk=0.05, contract_id="low-risk")
        self.assertTrue(contract.accepted)
        self.assertEqual(contract.accepted_horizon, 8)
        decision = engine.gate_contract(contract)
        self.assertTrue(decision.accepted)
        self.assertEqual(engine.ledger.accepted, 1)

    def test_future_contract_from_logits_accepts_and_rejects_observed_blocks(self):
        trace = CompressionTraceLedger()
        engine = FutureContractEngine(MTPFSPConfig(hidden_size=4, vocab_size=9), trace_ledger=trace)
        target = (1, 2, 3, 4)
        logits = torch.full((1, 4, 9), -20.0)
        for index, token_id in enumerate(target):
            logits[0, index, token_id] = 20.0

        contract = engine.draft_contract_from_logits({4: logits}, confidence=0.99, domain="general", risk=0.01, contract_id="from-logits")
        accepted = engine.gate_contract(contract, observed_tokens=target)
        wrong = ((target[0] + 1) % 9, *target[1:])
        rejected = engine.gate_contract(contract, observed_tokens=wrong)

        self.assertEqual(contract.requested_horizon, 4)
        self.assertEqual(contract.accepted_horizon, 4)
        self.assertEqual(contract.token_ids, target)
        self.assertTrue(accepted.accepted)
        self.assertFalse(rejected.accepted)
        self.assertEqual(engine.ledger.accepted, 1)
        self.assertEqual(engine.ledger.rejected, 1)
        self.assertEqual(len(trace.mtp_fsp_events), 2)
        self.assertFalse(trace.mtp_fsp_events[-1].accepted)

    def test_future_contract_rejects_incomplete_observed_block(self):
        engine = FutureContractEngine(MTPFSPConfig(hidden_size=4, vocab_size=9))
        target = (1, 2, 3, 4)
        logits = torch.full((1, 4, 9), -20.0)
        for index, token_id in enumerate(target):
            logits[0, index, token_id] = 20.0

        contract = engine.draft_contract_from_logits({4: logits}, confidence=0.99, domain="general", risk=0.01, contract_id="incomplete")
        decision = engine.gate_contract(contract, observed_tokens=target[:2])

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "observed tokens incomplete for future contract")
        self.assertLess(decision.contract.accepted_horizon, contract.accepted_horizon)

    def test_output_goal_contract_accepts_exact_result_and_rejects_extra_text_or_missing_anchor(self):
        engine = FutureContractEngine(MTPFSPConfig(hidden_size=4, vocab_size=7), heads=_confident_heads())
        exact_task = Task("goal-exact", "instruction_following", "Output OK exactly.", "OK")
        accepted = engine.gate_output_goal(exact_task, CandidateAnswer("OK", confidence=0.99), output_verified=True)
        extra = engine.gate_output_goal(exact_task, CandidateAnswer("OK extra", confidence=0.99), output_verified=True)
        anchor = Anchor("identifier", "C3-4242-A", "goal-anchor")
        anchor_task = Task(
            "goal-anchor",
            "long_context_anchor",
            "Return only the exact identifier.",
            "C3-4242-A",
            anchors=(anchor,),
        )
        missing_anchor = engine.gate_output_goal(anchor_task, CandidateAnswer("C3-0000-X", confidence=0.99), output_verified=False)

        self.assertTrue(accepted.accepted)
        self.assertFalse(extra.accepted)
        self.assertIn("exact_output_mismatch", extra.violations)
        self.assertFalse(missing_anchor.accepted)
        self.assertIn("required_anchor_missing", missing_anchor.violations)
        self.assertIn("oracle_verification_failed", missing_anchor.violations)
        self.assertEqual(len(engine.ledger.output_goal_decisions), 3)

    def test_output_goal_contract_rejects_internal_and_declared_forbidden_output(self):
        engine = FutureContractEngine(MTPFSPConfig(hidden_size=4, vocab_size=7), heads=_confident_heads())
        internal_task = Task("goal-leak", "instruction_following", "Produce a concise answer.", None)
        internal = engine.gate_output_goal(
            internal_task,
            CandidateAnswer("OK <analysis>hidden scratch</analysis>", confidence=0.99),
            output_verified=True,
        )
        declared_task = Task(
            "goal-forbidden",
            "calibration",
            "Say whether the answer is known.",
            None,
            {"forbidden_output_substrings": ("PRIVATE-ANCHOR", "training_contract_frontier_output_goal")},
        )
        declared = engine.gate_output_goal(
            declared_task,
            CandidateAnswer("unknown, but PRIVATE-ANCHOR leaked", confidence=0.99),
            output_verified=True,
        )

        self.assertFalse(internal.accepted)
        self.assertIn("no_internal_leakage", internal.contract.obligations)
        self.assertIn("forbidden_output_substring", internal.violations)
        self.assertIn("<analysis>", internal.forbidden_matches)
        self.assertFalse(declared.accepted)
        self.assertIn("no_forbidden_output", declared.contract.obligations)
        self.assertIn("PRIVATE-ANCHOR", declared.forbidden_matches)
        self.assertEqual(len(engine.ledger.output_goal_decisions), 2)

    def test_risky_domain_shortens_and_requires_gate_before_acceptance(self):
        trace = CompressionTraceLedger()
        engine = FutureContractEngine(MTPFSPConfig(hidden_size=4, vocab_size=7), heads=_confident_heads(), trace_ledger=trace)
        contract = engine.draft_contract(torch.zeros(1, 4), domain="math", risk=0.80, contract_id="risky")
        self.assertFalse(contract.accepted)
        self.assertLessEqual(contract.accepted_horizon, 2)
        accepted = engine.gate_contract(contract, observed_tokens=contract.token_ids)
        self.assertTrue(accepted.accepted)
        self.assertEqual(engine.ledger.accepted, 1)
        self.assertEqual(len(trace.mtp_fsp_events), 1)
        self.assertTrue(trace.mtp_fsp_events[0].accepted)

    def test_contract_revision_rejects_mismatch_and_high_temporal_loss(self):
        engine = FutureContractEngine(MTPFSPConfig(hidden_size=4, vocab_size=7), heads=_confident_heads())
        contract = engine.draft_contract(torch.zeros(1, 4), domain="general", risk=0.05, contract_id="revise")
        wrong = tuple((token + 1) % 7 for token in contract.token_ids[:2])
        mismatch = engine.gate_contract(contract, observed_tokens=wrong)
        self.assertFalse(mismatch.accepted)
        self.assertLess(mismatch.contract.accepted_horizon, contract.accepted_horizon)
        revised = engine.revise_contract(contract, temporal_loss=1.0)
        self.assertFalse(revised.accepted)
        self.assertEqual(revised.revision, 1)
        self.assertLess(revised.accepted_horizon, contract.accepted_horizon)

    def test_verified_answers_per_effective_cost_is_not_tokens_per_second(self):
        score = verified_answers_per_effective_cost(verified_answers=7.5, total_cost=CostTrace(generated_tokens=4, latent_steps=2, verifier_steps=1))
        self.assertGreater(score, 0.0)
        worse_cost = verified_answers_per_effective_cost(verified_answers=7.5, total_cost=CostTrace(generated_tokens=40, latent_steps=20, verifier_steps=10))
        self.assertLess(worse_cost, score)

    def test_mtp_fsp_calibrator_learns_standalone_future_tokens(self):
        verifier = DynamicSkillVerifier(default_skill_specs())
        tasks = verifier.build_suite(1, seed=3, include_metamorphic=False)
        answers = [ReferenceRuleAgent()(task).text for task in tasks]
        vocabulary = TokenVocabulary.from_texts(answers)
        examples = future_examples_from_tasks(tasks, lambda text: vocabulary.encode(text))
        calibrator = MTPFSPCalibrator(MTPFSPConfig(hidden_size=64, vocab_size=len(vocabulary.tokens)))

        _, result = calibrator.train(examples, epochs=120, lr=0.05)

        self.assertEqual(result.examples, len(tasks))
        self.assertLess(result.after_loss, result.before_loss * 0.05)
        self.assertLess(result.after_confidence_loss, result.before_confidence_loss)
        self.assertGreater(result.after_token_accuracy, result.before_token_accuracy)
        self.assertEqual(result.after_token_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
