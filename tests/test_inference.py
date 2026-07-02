import json
import tempfile
import unittest

import torch

from cortex3 import Anchor, CandidateAnswer, CorruptedCompressedAgent, DynamicSkillVerifier, ReferenceRuleAgent, Task, default_skill_specs
from cortex3_autoregressive import ARDecoderAgent, ARTrainer, ar_examples_from_tasks
from cortex3_certificates import CertificateType, LatentProofState, ProofCarryingAnswer, build_certificate
from cortex3_cycle import CortexCycle
from cortex3_inference import BudgetPredictor, DifficultyRouter, InferenceConfig, InferencePath, UltraFastInferenceEngine
from cortex3_memory import CognitiveMemory, CognitiveMemoryConfig
from cortex3_reporting import write_cycle_run


def _verifier() -> DynamicSkillVerifier:
    return DynamicSkillVerifier(default_skill_specs())


def _engine(memory: CognitiveMemory | None = None) -> UltraFastInferenceEngine:
    return UltraFastInferenceEngine(_verifier(), ReferenceRuleAgent(), memory=memory)


class UltraFastInferenceTest(unittest.TestCase):
    def test_difficulty_router_selects_fast_normal_and_careful_paths(self):
        router = DifficultyRouter(InferenceConfig())
        easy = Task("easy-format", "instruction_following", "Output OK exactly.", "OK")
        normal = Task("normal-math", "arithmetic", "Compute exactly: 20 + 22. Return only the integer.", 42)
        anchor = Anchor("identifier", "C3-9999-A", "hard")
        hard_prompt = " ".join(["long context"] * 45) + " exact identifier C3-9999-A"
        hard = Task("hard-anchor", "long_context_anchor", hard_prompt, "C3-9999-A", anchors=(anchor,))

        self.assertEqual(router.route(router.signal(easy, 1.0)).path, InferencePath.FAST)
        self.assertEqual(router.route(router.signal(normal, 1.0)).path, InferencePath.NORMAL)
        self.assertEqual(router.route(router.signal(hard, 1.0)).path, InferencePath.CAREFUL)

    def test_fast_path_uses_less_depth_and_better_verified_cost_than_careful_on_easy_task(self):
        verifier = _verifier()
        engine = UltraFastInferenceEngine(verifier, ReferenceRuleAgent())
        task = Task("easy-ok", "instruction_following", "Output OK exactly.", "OK")

        fast = engine.infer(task)
        careful = engine.infer(task, forced_path=InferencePath.CAREFUL)

        self.assertEqual(fast.route.path, InferencePath.FAST)
        self.assertEqual(careful.route.path, InferencePath.CAREFUL)
        self.assertTrue(verifier.oracle_registry.verify(task.skill, task, fast.answer).passed)
        self.assertTrue(careful.verification and careful.verification.passed)
        self.assertLess(fast.layers_ran, careful.layers_ran)
        self.assertLess(fast.cost.effective_cost(), careful.cost.effective_cost())
        self.assertGreater(fast.verified_capability_per_cost, careful.verified_capability_per_cost)
        self.assertEqual(len(fast.kernel_dispatches), fast.layers_ran)
        self.assertEqual(len(fast.trace_summary["compression_decisions"]), fast.layers_ran)
        self.assertEqual(len(fast.trace_summary["activation_quantizations"]), fast.layers_ran)
        runtime_dispatches = fast.trace_summary["packed_ternary_dispatches"]
        self.assertEqual(len(runtime_dispatches), fast.layers_ran)
        self.assertEqual(fast.kernel_dispatches[0].mode, runtime_dispatches[0]["backend"])
        self.assertEqual(fast.kernel_dispatches[0].source_layer_id, runtime_dispatches[0]["layer_id"])
        self.assertEqual(fast.kernel_dispatches[0].packed_weight_bytes, runtime_dispatches[0]["packed_weight_bytes"])
        self.assertEqual(fast.kernel_dispatches[0].native_kernel, runtime_dispatches[0]["native_kernel"])
        self.assertEqual(fast.kernel_dispatches[0].kernel_variant, runtime_dispatches[0]["kernel_variant"])
        self.assertEqual(fast.kernel_dispatches[0].native_backend, runtime_dispatches[0]["native_backend"])
        self.assertIn("runtime backend=", fast.kernel_dispatches[0].reason)
        self.assertIn("requantize_backend=", fast.kernel_dispatches[0].reason)
        self.assertNotIn(fast.kernel_dispatches[0].mode, {"cuda_ternary_packed", "cpu_ternary_reference"})

    def test_fast_path_verified_cost_rejects_confident_wrong_answer(self):
        verifier = _verifier()
        engine = UltraFastInferenceEngine(verifier, lambda _: CandidateAnswer("NO", confidence=0.99))
        task = Task("fast-wrong", "instruction_following", "Output OK exactly.", "OK")

        result = engine.infer(task)

        self.assertEqual(result.route.path, InferencePath.FAST)
        self.assertEqual(result.route.verifier_level, 0)
        self.assertEqual(result.predicted_cost.predicted_cost.verifier_steps, 0)
        self.assertIsNotNone(result.verification)
        self.assertFalse(result.verification.passed)
        self.assertFalse(result.passed)
        self.assertEqual(result.verified_capability_per_cost, 0.0)
        self.assertFalse(result.answer.certificate["output_goal_contract_passed"])
        self.assertFalse(result.future_contract["output_goal_contract"]["accepted"])
        self.assertIn("oracle_verification_failed", result.future_contract["output_goal_contract"]["violations"])

    def test_inference_output_goal_rejects_internal_leakage_in_answer_payload(self):
        verifier = _verifier()
        engine = UltraFastInferenceEngine(
            verifier,
            lambda _: CandidateAnswer("OK <analysis>hidden scratch</analysis>", confidence=0.99),
        )
        task = Task("fast-leak", "instruction_following", "Output OK exactly.", "OK")

        result = engine.infer(task)
        output_goal = result.future_contract["output_goal_contract"]

        self.assertFalse(output_goal["accepted"])
        self.assertFalse(result.answer.certificate["output_goal_contract_passed"])
        self.assertIn("forbidden_output_substring", output_goal["violations"])
        self.assertIn("<analysis>", output_goal["forbidden_matches"])

    def test_normal_path_uses_light_certificate_and_moderate_budget(self):
        verifier = _verifier()
        engine = UltraFastInferenceEngine(verifier, ReferenceRuleAgent())
        task = Task("normal-math-budget", "arithmetic", "Compute exactly: 20 + 22. Return only the integer.", 42)

        fast = engine.infer(task, forced_path=InferencePath.FAST)
        normal = engine.infer(task)
        careful = engine.infer(task, forced_path=InferencePath.CAREFUL)

        self.assertEqual(normal.route.path, InferencePath.NORMAL)
        self.assertEqual(normal.route.compression_strength, 0.50)
        self.assertEqual(normal.route.layers_to_run, engine.config.normal_layers)
        self.assertEqual(normal.layers_ran, engine.config.normal_layers)
        self.assertEqual(normal.route.verifier_level, 1)
        self.assertEqual(normal.route.mtp_horizon, 2)
        self.assertTrue(normal.route.use_latent_kv)
        self.assertEqual(normal.route.latent_loops, 1)
        self.assertEqual(normal.route.experts_activated, 0)
        self.assertTrue(normal.verification and normal.verification.passed)
        self.assertTrue(normal.certificate_verified)
        self.assertLessEqual(normal.future_contract["horizon"], normal.route.mtp_horizon)
        self.assertTrue(normal.future_contract["output_goal_contract"]["accepted"])
        self.assertTrue(normal.answer.certificate["output_goal_contract_passed"])
        self.assertLess(fast.cost.effective_cost(), normal.cost.effective_cost())
        self.assertLess(normal.cost.effective_cost(), careful.cost.effective_cost())

    def test_budget_predictor_prices_careful_path_above_fast_path(self):
        config = InferenceConfig()
        router = DifficultyRouter(config)
        predictor = BudgetPredictor()
        task = Task("budget", "instruction_following", "Output OK exactly.", "OK")
        signal = router.signal(task, 1.0)

        fast = predictor.predict(router.route(signal, InferencePath.FAST), signal.prompt_tokens)
        careful = predictor.predict(router.route(signal, InferencePath.CAREFUL), signal.prompt_tokens)

        self.assertLess(fast.predicted_effective_cost, careful.predicted_effective_cost)
        self.assertLess(fast.predicted_cost.verifier_steps, careful.predicted_cost.verifier_steps)

    def test_latent_kv_reconstruction_is_used_and_anchor_fidelity_is_verified(self):
        required = Anchor("identifier", "C3-7777-Z", "legacy")
        memory = CognitiveMemory(CognitiveMemoryConfig(recent_exact_limit=1, embedding_dim=32, top_k_latent=2))
        memory.ingest(
            "legacy",
            "Ancien contexte: Sofia garde le prototype et le code exact C3-7777-Z.",
            extra_anchors=(required,),
        )
        memory.ingest("recent", "Message recent sans code.")
        engine = _engine(memory)
        task = Task(
            "latent-anchor",
            "long_context_anchor",
            "Retrouve le code exact du prototype Sofia.",
            "C3-7777-Z",
            anchors=(required,),
        )

        result = engine.infer(task, forced_path=InferencePath.FAST)

        self.assertEqual(result.route.path, InferencePath.FAST)
        self.assertIsNotNone(result.memory_reconstruction)
        assert result.memory_reconstruction is not None
        self.assertIn("legacy", result.memory_reconstruction.selected_segment_ids)
        self.assertIn("C3-7777-Z", result.memory_reconstruction.rendered_context)
        self.assertTrue(result.memory_reconstruction.fidelity.passed)
        self.assertTrue(any(event["mode"] == "latent" for event in result.trace_summary["kv_events"]))

    def test_memory_augmented_generation_recovers_latent_anchor_answer(self):
        verifier = _verifier()
        required = Anchor("identifier", "C3-7777-Z", "legacy")
        memory = CognitiveMemory(CognitiveMemoryConfig(recent_exact_limit=1, embedding_dim=32, top_k_latent=2))
        memory.ingest(
            "legacy",
            "Archive lente: Sofia garde le prototype et le code exact C3-7777-Z.",
            extra_anchors=(required,),
        )
        memory.ingest("recent", "Message récent sans identifiant utile.")
        engine = UltraFastInferenceEngine(verifier, CorruptedCompressedAgent(), memory=memory)
        task = Task(
            "memory-generated-anchor",
            "long_context_anchor",
            "Retrouve le code exact du prototype Sofia.",
            "C3-7777-Z",
            {"ask_kind": "code"},
            anchors=(required,),
        )

        result = engine.infer(task, forced_path=InferencePath.CAREFUL)

        self.assertEqual(result.answer.text, "C3-7777-Z")
        self.assertTrue(result.verification and result.verification.passed)
        self.assertTrue(result.certificate_verified)
        self.assertEqual(result.answer.certificate["memory_augmented_generation"], "anchor_reconstruction")
        self.assertTrue(result.answer.certificate["memory_replaced_base_answer"])
        self.assertEqual(result.answer.raw["memory_base_answer"]["text"], "C3-7777-A")
        self.assertIn("legacy", result.answer.raw["memory_augmented_generation"]["selected_segment_ids"])
        self.assertGreater(result.answer.cost.effective_cost(), CorruptedCompressedAgent()(task).cost.effective_cost())

    def test_self_speculative_decoding_respects_route_horizon_caps(self):
        engine = _engine()
        easy = Task("spec-fast", "instruction_following", "Output OK exactly.", "OK")
        math = Task("spec-math", "arithmetic", "Compute exactly: 20 + 22. Return only the integer.", 42)

        fast = engine.infer(easy)
        careful_math = engine.infer(math, forced_path=InferencePath.CAREFUL)

        self.assertLessEqual(fast.future_contract["horizon"], 8)
        self.assertEqual(fast.future_contract["observed_tokens_source"], "answer_text")
        self.assertFalse(fast.future_contract["self_verified_tokens"])
        self.assertEqual(fast.future_contract["observed_tokens"], (ord("O") % engine.config.vocab_size, ord("K") % engine.config.vocab_size))
        self.assertEqual(careful_math.route.mtp_horizon, 1)
        self.assertEqual(careful_math.future_contract["horizon"], 1)
        self.assertEqual(careful_math.future_contract["observed_tokens_source"], "answer_text")
        self.assertFalse(careful_math.future_contract["self_verified_tokens"])
        self.assertEqual(careful_math.trace_summary["mtp_fsp_events"][0]["horizon"], 1)

    def test_careful_route_runs_strong_verification_certificate_and_expert_trace(self):
        anchor = Anchor("identifier", "C3-2222-A", "audit")
        prompt = " ".join(["audit trail"] * 45) + " exact identifier C3-2222-A"
        task = Task("careful-anchor", "long_context_anchor", prompt, "C3-2222-A", anchors=(anchor,))

        result = _engine().infer(task)

        self.assertEqual(result.route.path, InferencePath.CAREFUL)
        self.assertEqual(result.route.verifier_level, 3)
        self.assertTrue(result.verification and result.verification.passed)
        self.assertTrue(result.certificate_verified)
        self.assertEqual(result.layers_ran, result.route.layers_to_run)
        self.assertTrue(result.trace_summary["expert_activations"])

    def test_inference_certificate_uses_task_contract_not_answer_as_expected(self):
        task = Task("cert-contract", "instruction_following", "Output OK exactly.", "OK")
        engine = _engine()
        signal = engine.router.signal(task, 0.99)
        route = engine.router.route(signal, InferencePath.CAREFUL)
        hidden = torch.zeros(1, engine.config.hidden_size)

        self.assertFalse(engine._certificate_verified(hidden, task, CandidateAnswer("OK extra", confidence=0.99), route))
        self.assertTrue(engine._certificate_verified(hidden, task, CandidateAnswer("OK", confidence=0.99), route))

    def test_inference_gate_rejects_tampered_proof_carrying_answer(self):
        task = Task("proof-gate", "arithmetic", "Compute exactly: 20 + 22. Return only the integer.", 42, {"kind": "add", "a": 20, "b": 22})
        state = LatentProofState("proof-gate-latent", task.task_id, task.skill, tensor=torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32), latent_steps=1)
        cert = build_certificate(
            certificate_id="proof-gate-cert",
            task_id=task.task_id,
            skill=task.skill,
            certificate_type=CertificateType.ARITHMETIC,
            answer="42",
            claims={"operation": "20 + 22"},
            uncertainty=0.05,
            latent_state=state,
            tool="arithmetic",
            tool_args={"expression": "20 + 22", "expected": 42},
        )
        valid = ProofCarryingAnswer("42", cert, cert.uncertainty, state).to_candidate_answer()
        proof = dict(valid.raw["proof_carrying_answer"])
        latent = dict(proof["latent_state"])
        values = list(latent["values"])
        values[0] += 0.25
        latent["values"] = values
        proof["latent_state"] = latent
        tampered = CandidateAnswer(valid.text, valid.confidence, valid.certificate, valid.cost, {"proof_carrying_answer": proof})
        engine = UltraFastInferenceEngine(_verifier(), lambda _: tampered)

        result = engine.infer(task, forced_path=InferencePath.CAREFUL)

        self.assertTrue(result.verification and result.verification.passed)
        self.assertFalse(result.certificate_verified)
        self.assertFalse(result.passed)
        self.assertEqual(result.verified_capability_per_cost, 0.0)

    def test_engine_can_use_trained_autoregressive_answer_source(self):
        verifier = _verifier()
        tasks = verifier.build_suite(1, seed=3, include_metamorphic=False)
        examples = ar_examples_from_tasks(tasks)
        model, training = ARTrainer().train(examples, epochs=80, lr=0.03)
        engine = UltraFastInferenceEngine(verifier, ARDecoderAgent(model))

        result = engine.infer(tasks[0], forced_path=InferencePath.CAREFUL)

        self.assertEqual(training.exact_sequence_accuracy, 1.0)
        self.assertTrue(result.verification and result.verification.passed)
        self.assertTrue(result.certificate_verified)
        self.assertEqual(result.answer.certificate["autoregressive_decoder"], "trained")
        self.assertEqual(result.answer.certificate["inference_path"], "careful")
        self.assertGreater(result.answer.cost.generated_tokens, 0)
        self.assertGreater(result.cost.effective_cost(), result.answer.cost.effective_cost())

    def test_cycle_report_can_persist_inference_results(self):
        verifier = _verifier()
        report = CortexCycle(verifier).run(ReferenceRuleAgent(), ReferenceRuleAgent(), seed=2, n_per_skill=1)
        inference = UltraFastInferenceEngine(verifier, ReferenceRuleAgent()).infer(
            Task("persist-fast", "instruction_following", "Output OK exactly.", "OK")
        )

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_cycle_run(report, output_dir=tmp, run_id="inference-run", inference_results=(inference,))
            payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))

        self.assertEqual(payload["inference"][0]["route"]["path"], "fast")
        self.assertLessEqual(payload["inference"][0]["future_contract"]["horizon"], 8)
        self.assertEqual(payload["inference"][0]["future_contract"]["observed_tokens_source"], "answer_text")
        self.assertFalse(payload["inference"][0]["future_contract"]["self_verified_tokens"])
        self.assertTrue(payload["inference"][0]["kernel_dispatches"])


if __name__ == "__main__":
    unittest.main()
