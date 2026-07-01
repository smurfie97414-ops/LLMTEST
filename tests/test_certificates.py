import unittest
import json
import tempfile

import torch

from cortex3 import Anchor, Task
from cortex3 import CorruptedCompressedAgent, DynamicSkillVerifier, ReferenceRuleAgent, default_skill_specs
from cortex3_certificates import (
    CertificateAnswerVocabulary,
    CertificateHead,
    CertificateHeadCalibrator,
    CertificateType,
    CertificateVerifier,
    LatentProofState,
    ProofCarryingAnswer,
    ProofCarryingGenerator,
    RandomDelatentizer,
    ShortCertificate,
    algebra_linear_tool,
    build_certificate,
    build_compiled_circuit_certificate,
    certificate_contract_for_task,
    certificate_examples_from_tasks,
    code_unit_test_tool,
    compiled_circuit_tool,
    default_tool_registry,
    evaluate_certificate_efficiency,
    model_token_certificate_tool,
    sympy_symbolic_tool,
)
from cortex3_cycle import CortexCycle
from cortex3_reporting import write_cycle_run


def _latent_state() -> LatentProofState:
    tensor = torch.tensor([[0.125, -0.5, 0.75, 1.0]], dtype=torch.float32)
    return LatentProofState("latent-1", "task-1", "arithmetic", tensor, latent_steps=3, visible_reasoning_tokens=0)


class CertificatesTest(unittest.TestCase):
    def test_certificate_head_outputs_latent_state_answer_cert_type_and_uncertainty(self):
        torch.manual_seed(0)
        head = CertificateHead(hidden_size=6, latent_size=4, answer_vocab_size=11)
        output = head(torch.ones(2, 6))
        self.assertEqual(tuple(output.latent_state.shape), (2, 4))
        self.assertEqual(tuple(output.answer_logits.shape), (2, 11))
        self.assertEqual(tuple(output.certificate_type_logits.shape), (2, len(CertificateType)))
        self.assertEqual(tuple(output.uncertainty.shape), (2,))
        self.assertTrue(bool(((output.uncertainty >= 0) & (output.uncertainty <= 1)).all()))

    def test_arithmetic_certificate_verifies_with_tool_and_checksum(self):
        state = _latent_state()
        cert = build_certificate(
            certificate_id="cert-1",
            task_id="task-1",
            skill="arithmetic",
            certificate_type=CertificateType.ARITHMETIC,
            answer="42",
            claims={"operation": "6 * 7"},
            uncertainty=0.05,
            latent_state=state,
            tool="arithmetic",
            tool_args={"expression": "6 * 7"},
        )
        result = CertificateVerifier().verify(cert, state)
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "certificate verified")

    def test_tampered_latent_state_fails_certificate_verification(self):
        state = _latent_state()
        cert = build_certificate(
            certificate_id="cert-2",
            task_id="task-1",
            skill="arithmetic",
            certificate_type=CertificateType.ARITHMETIC,
            answer="42",
            claims={"operation": "6 * 7"},
            uncertainty=0.05,
            latent_state=state,
            tool="arithmetic",
            tool_args={"expression": "6 * 7"},
        )
        tampered = LatentProofState("latent-1", "task-1", "arithmetic", state.tensor + 0.25, latent_steps=3)
        result = CertificateVerifier().verify(cert, tampered)
        self.assertFalse(result.passed)
        self.assertFalse(result.latent_checksum_ok)
        self.assertEqual(result.reason, "latent proof checksum mismatch")

    def test_random_delatentization_is_deterministic_and_detects_tampering(self):
        state = _latent_state()
        delatentizer = RandomDelatentizer(probes=3)
        probe_a = delatentizer.probe(state, seed=123)
        probe_b = delatentizer.probe(state, seed=123)
        self.assertEqual(probe_a, probe_b)
        self.assertTrue(delatentizer.verify_probe(state, probe_a))
        tampered = LatentProofState("latent-1", "task-1", "arithmetic", state.tensor + 0.01, latent_steps=3)
        self.assertFalse(delatentizer.verify_probe(tampered, probe_a))

    def test_anchor_certificate_requires_exact_anchor_presence(self):
        state = _latent_state()
        anchor = Anchor("identifier", "C3-3131-A", "task-anchor")
        cert = build_certificate(
            certificate_id="cert-anchor",
            task_id="task-anchor",
            skill="long_context_anchor",
            certificate_type=CertificateType.ANCHOR_FIDELITY,
            answer="C3-3131-A",
            claims={"copied_anchor": "C3-3131-A"},
            uncertainty=0.02,
            latent_state=state,
            anchors=(anchor,),
            tool="anchor_fidelity",
        )
        self.assertTrue(CertificateVerifier().verify(cert, state).passed)
        bad = ShortCertificate(
            certificate_id=cert.certificate_id,
            task_id=cert.task_id,
            skill=cert.skill,
            certificate_type=cert.certificate_type,
            answer="C3-0000-X",
            claims={},
            uncertainty=cert.uncertainty,
            latent_state_checksum=cert.latent_state_checksum,
            anchors=cert.anchors,
            tool=cert.tool,
        )
        self.assertFalse(CertificateVerifier().verify(bad, state).passed)

    def test_model_token_certificate_binds_answer_to_certificate_head_output(self):
        state = _latent_state()
        cert = build_certificate(
            certificate_id="cert-model-token",
            task_id="task-model-token",
            skill="llm_certificate_head",
            certificate_type=CertificateType.EXACT_MATCH,
            answer="alpha",
            claims={"model_certificate_head": True},
            uncertainty=0.12,
            latent_state=state,
            tool="model_token_certificate",
            tool_args={
                "answer_token_id": 17,
                "certificate_head_token_id": 17,
                "lm_head_token_id": 22,
                "target_token_id": 23,
                "decoded_answer": "alpha",
            },
        )
        bad = build_certificate(
            certificate_id="cert-model-token-bad",
            task_id="task-model-token",
            skill="llm_certificate_head",
            certificate_type=CertificateType.EXACT_MATCH,
            answer="beta",
            claims={"model_certificate_head": True},
            uncertainty=0.12,
            latent_state=state,
            tool="model_token_certificate",
            tool_args={
                "answer_token_id": 17,
                "certificate_head_token_id": 17,
                "decoded_answer": "alpha",
            },
        )

        self.assertTrue(CertificateVerifier().verify(cert, state).passed)
        self.assertTrue(model_token_certificate_tool(cert).passed)
        self.assertFalse(model_token_certificate_tool(bad).passed)

    def test_code_certificate_runs_real_unit_tests(self):
        state = _latent_state()
        source = "def solve(x):\n    return x + 1\n"
        cert = build_certificate(
            certificate_id="cert-code",
            task_id="task-code",
            skill="code_unit_tests",
            certificate_type=CertificateType.CODE_TESTS,
            answer=source,
            claims={"function": "solve"},
            uncertainty=0.08,
            latent_state=state,
            tool="code_tests",
            tool_args={"function_name": "solve", "tests": [((1,), 2), ((-1,), 0)]},
        )
        self.assertTrue(CertificateVerifier().verify(cert, state).passed)
        bad = build_certificate(
            certificate_id="cert-code-bad",
            task_id="task-code",
            skill="code_unit_tests",
            certificate_type=CertificateType.CODE_TESTS,
            answer="def solve(x):\n    return x\n",
            claims={"function": "solve"},
            uncertainty=0.08,
            latent_state=state,
            tool="code_tests",
            tool_args={"function_name": "solve", "tests": [((1,), 2)]},
        )
        self.assertFalse(code_unit_test_tool(bad).passed)

    def test_algebra_certificate_requires_multi_step_linear_proof(self):
        state = _latent_state()
        task = Task(
            "task-algebra",
            "algebra",
            "Solve exactly for x: 7x + -3 = 25. Return only the integer value of x.",
            4,
            {"variable": "x", "a": 7, "b": -3, "c": 25, "solution": 4, "kind": "linear"},
        )
        claims, tool, tool_args, anchors = certificate_contract_for_task(task, "4")
        cert = build_certificate(
            certificate_id="cert-algebra",
            task_id=task.task_id,
            skill=task.skill,
            certificate_type=CertificateType.ALGEBRA,
            answer="4",
            claims=claims,
            uncertainty=0.04,
            latent_state=state,
            anchors=anchors,
            tool=tool,
            tool_args=tool_args,
        )

        self.assertTrue(CertificateVerifier().verify(cert, state).passed)
        self.assertTrue(algebra_linear_tool(cert).passed)
        bad_steps = list(claims["algebra_steps"])
        bad_steps[1] = {**dict(bad_steps[1]), "result": 5}
        bad = ShortCertificate(
            certificate_id=cert.certificate_id,
            task_id=cert.task_id,
            skill=cert.skill,
            certificate_type=cert.certificate_type,
            answer=cert.answer,
            claims={**dict(cert.claims), "algebra_steps": tuple(bad_steps)},
            uncertainty=cert.uncertainty,
            latent_state_checksum=cert.latent_state_checksum,
            tool=cert.tool,
            tool_args=cert.tool_args,
        )
        extra_text = ShortCertificate(
            certificate_id=cert.certificate_id,
            task_id=cert.task_id,
            skill=cert.skill,
            certificate_type=cert.certificate_type,
            answer="x = 4",
            claims=cert.claims,
            uncertainty=cert.uncertainty,
            latent_state_checksum=cert.latent_state_checksum,
            tool=cert.tool,
            tool_args=cert.tool_args,
        )
        self.assertFalse(algebra_linear_tool(bad).passed)
        self.assertFalse(algebra_linear_tool(extra_text).passed)

    def test_symbolic_algebra_certificate_uses_sympy_solver_for_quadratic_roots(self):
        state = _latent_state()
        task = Task(
            "task-symbolic",
            "algebra",
            "Solve exactly for x: x^2 - x - 6 = 0. Return the exact roots as a comma-separated set.",
            "-2, 3",
            {"variable": "x", "a": 1, "b": -1, "c": -6, "kind": "quadratic"},
        )
        claims, tool, tool_args, anchors = certificate_contract_for_task(task, "-2, 3")
        cert = build_certificate(
            certificate_id="cert-symbolic",
            task_id=task.task_id,
            skill=task.skill,
            certificate_type=CertificateType.ALGEBRA,
            answer="-2, 3",
            claims=claims,
            uncertainty=0.03,
            latent_state=state,
            anchors=anchors,
            tool=tool,
            tool_args=tool_args,
        )

        self.assertEqual(tool, "sympy_symbolic")
        self.assertEqual(tuple(tool_args["expected_roots"]), ("-2", "3"))
        self.assertEqual(claims["symbolic_solver"], "sympy")
        self.assertTrue(CertificateVerifier().verify(cert, state).passed)
        self.assertTrue(sympy_symbolic_tool(cert).passed)

        wrong_roots = ShortCertificate(
            certificate_id=cert.certificate_id,
            task_id=cert.task_id,
            skill=cert.skill,
            certificate_type=cert.certificate_type,
            answer="-2, 4",
            claims=cert.claims,
            uncertainty=cert.uncertainty,
            latent_state_checksum=cert.latent_state_checksum,
            anchors=cert.anchors,
            tool=cert.tool,
            tool_args=cert.tool_args,
        )
        tampered_claims = ShortCertificate(
            certificate_id=cert.certificate_id,
            task_id=cert.task_id,
            skill=cert.skill,
            certificate_type=cert.certificate_type,
            answer=cert.answer,
            claims={**dict(cert.claims), "symbolic_solver": "claimed-local-solver"},
            uncertainty=cert.uncertainty,
            latent_state_checksum=cert.latent_state_checksum,
            anchors=cert.anchors,
            tool=cert.tool,
            tool_args=cert.tool_args,
        )
        self.assertFalse(sympy_symbolic_tool(wrong_roots).passed)
        self.assertFalse(sympy_symbolic_tool(tampered_claims).passed)

    def test_code_certificate_requires_hidden_tests_and_properties_when_declared(self):
        state = _latent_state()
        cert = build_certificate(
            certificate_id="cert-code-rich",
            task_id="task-code-rich",
            skill="code_unit_tests",
            certificate_type=CertificateType.CODE_TESTS,
            answer="def solve(values):\n    return values[0] if values else None\n",
            claims={"function": "solve", "hidden_tests": 1, "properties": ("deterministic", "no_argument_mutation")},
            uncertainty=0.05,
            latent_state=state,
            tool="code_tests",
            tool_args={
                "function_name": "solve",
                "tests": [(([1, 2],), 1)],
                "hidden_tests": [(([],), None)],
                "require_hidden_tests": True,
                "min_tests": 2,
                "properties": ("deterministic", "no_argument_mutation"),
            },
        )
        missing_hidden = build_certificate(
            certificate_id="cert-code-no-hidden",
            task_id="task-code-rich",
            skill="code_unit_tests",
            certificate_type=CertificateType.CODE_TESTS,
            answer=cert.answer,
            claims=cert.claims,
            uncertainty=0.05,
            latent_state=state,
            tool="code_tests",
            tool_args={
                "function_name": "solve",
                "tests": [(([1, 2],), 1)],
                "require_hidden_tests": True,
                "min_tests": 2,
            },
        )
        mutating = build_certificate(
            certificate_id="cert-code-mutating",
            task_id="task-code-rich",
            skill="code_unit_tests",
            certificate_type=CertificateType.CODE_TESTS,
            answer="def solve(values):\n    values.append(99)\n    return values[0]\n",
            claims=cert.claims,
            uncertainty=0.05,
            latent_state=state,
            tool="code_tests",
            tool_args={
                "function_name": "solve",
                "tests": [(([1, 2],), 1)],
                "hidden_tests": [(([3],), 3)],
                "require_hidden_tests": True,
                "min_tests": 2,
                "properties": ("no_argument_mutation",),
            },
        )

        self.assertTrue(CertificateVerifier().verify(cert, state).passed)
        self.assertFalse(code_unit_test_tool(missing_hidden).passed)
        self.assertFalse(code_unit_test_tool(mutating).passed)

    def test_compiled_circuit_certificate_binds_contract_and_lineage(self):
        state = _latent_state()
        anchor = Anchor("identifier", "frontier-anchor", "frontier-source")
        contract = {
            "circuit_id": "circuit-1",
            "skill": "arithmetic",
            "task_id": "task-1",
            "source_failure_ids": ("task-1",),
            "frontier_task_ids": ("task-1", "task-1-frontier"),
            "verified_slow_solutions": 2,
            "prompt_obligations": ("exact_output",),
            "invariant_checksum": "invariant-1",
            "compiled_weight_bits": 128.0,
            "active_weights": 8,
            "total_weights": 16,
            "dsv_passed": True,
            "dsv_verified": 2,
            "dsv_total": 2,
            "heldout_task_ids": ("task-1-heldout",),
            "heldout_passed": 1,
            "heldout_total": 1,
            "heldout_pass_rate": 1.0,
            "heldout_gate_passed": True,
            "output_verified": True,
        }
        cert = build_compiled_circuit_certificate(
            certificate_id="cert-compiled",
            task=Task("task-1", "arithmetic", "Compute exactly: 2 + 2.", 4, anchors=(anchor,)),
            answer="4",
            claims={"frontier_compiled_circuit": True},
            uncertainty=0.03,
            latent_state=state,
            contract=contract,
        )

        result = CertificateVerifier().verify(cert, state)
        self.assertTrue(result.passed, result.reason)
        self.assertTrue(compiled_circuit_tool(cert).passed)
        tampered_contract = dict(cert.claims["compiled_circuit_contract"])
        tampered_contract["output_verified"] = False
        tampered = ShortCertificate(
            certificate_id=cert.certificate_id,
            task_id=cert.task_id,
            skill=cert.skill,
            certificate_type=cert.certificate_type,
            answer=cert.answer,
            claims={**dict(cert.claims), "compiled_circuit_contract": tampered_contract},
            uncertainty=cert.uncertainty,
            latent_state_checksum=cert.latent_state_checksum,
            anchors=cert.anchors,
            tool=cert.tool,
            tool_args=cert.tool_args,
        )
        self.assertFalse(compiled_circuit_tool(tampered).passed)
        heldout_failed = build_compiled_circuit_certificate(
            certificate_id="cert-compiled-heldout-failed",
            task=Task("task-1", "arithmetic", "Compute exactly: 2 + 2.", 4, anchors=(anchor,)),
            answer="4",
            claims={"frontier_compiled_circuit": True},
            uncertainty=0.03,
            latent_state=state,
            contract={**contract, "heldout_passed": 0, "heldout_gate_passed": False},
        )
        self.assertFalse(compiled_circuit_tool(heldout_failed).passed)

    def test_proof_carrying_answer_maps_to_candidate_answer(self):
        state = _latent_state()
        cert = build_certificate(
            certificate_id="cert-answer",
            task_id="task-answer",
            skill="arithmetic",
            certificate_type=CertificateType.ARITHMETIC,
            answer="9",
            claims={"operation": "4 + 5"},
            uncertainty=0.10,
            latent_state=state,
            tool="arithmetic",
            tool_args={"expression": "4 + 5"},
        )
        answer = ProofCarryingAnswer("9", cert, cert.uncertainty, state).to_candidate_answer()
        self.assertEqual(answer.text, "9")
        self.assertAlmostEqual(answer.confidence, 0.90)
        self.assertIn("checksum", answer.certificate)
        self.assertEqual(answer.cost.latent_steps, 3)

    def test_certificate_efficiency_requires_quality_and_calibration(self):
        state = _latent_state()
        cert = build_certificate(
            certificate_id="cert-efficient",
            task_id="task-efficient",
            skill="arithmetic",
            certificate_type=CertificateType.ARITHMETIC,
            answer="42",
            claims={"operation": "6 * 7"},
            uncertainty=0.10,
            latent_state=state,
            tool="arithmetic",
            tool_args={"expression": "6 * 7"},
        )
        verification = CertificateVerifier().verify(cert, state)
        slow_reasoning = " ".join(["slow"] * 220)
        efficiency = evaluate_certificate_efficiency(slow_reasoning, cert, verification, reference_uncertainty=0.12)
        self.assertTrue(efficiency.passed)
        self.assertGreater(efficiency.token_reduction, 0)
        self.assertLess(efficiency.reduction_ratio, 1.0)

    def test_certificate_head_calibration_produces_verified_proof_carrying_answers(self):
        verifier = DynamicSkillVerifier(default_skill_specs())
        tasks = verifier.build_suite(1, seed=3, include_metamorphic=False)
        examples = certificate_examples_from_tasks(tasks)
        vocabulary = CertificateAnswerVocabulary.from_answers(example.answer for example in examples)
        calibrator = CertificateHeadCalibrator(hidden_size=64, latent_size=16, vocabulary=vocabulary)

        head, training = calibrator.train(examples, epochs=160, lr=0.04)
        agent = ProofCarryingGenerator(head, vocabulary)
        report = verifier.evaluate_tasks(agent, tasks)
        generated = [agent(task) for task in tasks]
        calibration_answer = next(answer for task, answer in zip(tasks, generated) if task.skill == "calibration")

        self.assertLess(training.after_loss, training.before_loss * 0.01)
        self.assertEqual(training.after_answer_accuracy, 1.0)
        self.assertEqual(training.after_certificate_type_accuracy, 1.0)
        self.assertLess(training.after_uncertainty_mae, training.before_uncertainty_mae)
        self.assertEqual(report.passed, report.total)
        self.assertTrue(all(answer.raw["certificate_verification"]["passed"] for answer in generated))
        self.assertTrue(all(answer.certificate["proof_carrying_generation"] for answer in generated))
        self.assertLess(calibration_answer.confidence, 0.50)
        self.assertTrue(calibration_answer.raw["certificate_verification"]["uncertainty_ok"])

    def test_default_registry_contains_required_tools(self):
        registry = default_tool_registry()
        self.assertEqual(set(registry.names), {"algebra_linear", "anchor_fidelity", "arithmetic", "code_tests", "compiled_circuit", "exact_match", "model_token_certificate", "sympy_symbolic"})

    def test_cycle_run_artifacts_can_include_short_certificates(self):
        state = _latent_state()
        cert = build_certificate(
            certificate_id="cert-run",
            task_id="task-run",
            skill="arithmetic",
            certificate_type=CertificateType.ARITHMETIC,
            answer="42",
            claims={"operation": "6 * 7"},
            uncertainty=0.04,
            latent_state=state,
            tool="arithmetic",
            tool_args={"expression": "6 * 7"},
        )
        self.assertTrue(CertificateVerifier().verify(cert, state).passed)
        verifier = DynamicSkillVerifier(default_skill_specs())
        report = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=6, n_per_skill=1)
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_cycle_run(report, output_dir=tmp, run_id="cert-run", certificates=(cert,))
            payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["certificates"][0]["certificate_id"], "cert-run")
            self.assertIn("checksum", payload["certificates"][0])


if __name__ == "__main__":
    unittest.main()
