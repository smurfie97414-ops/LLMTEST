import json
import tempfile
import unittest

from cortex3 import Anchor, CorruptedCompressedAgent, DynamicSkillVerifier, ReferenceRuleAgent, default_skill_specs
from cortex3_cycle import CortexCycle
from cortex3_memory import AnchorFidelityVerifier, CognitiveMemory, CognitiveMemoryConfig, MemoryMode, MemoryRetentionDecision, embed_text
from cortex3_reporting import write_cycle_run


class CognitiveMemoryTest(unittest.TestCase):
    def test_learned_retention_policy_can_store_segment_directly_as_latent(self):
        memory = CognitiveMemory(CognitiveMemoryConfig(recent_exact_limit=2, embedding_dim=32))
        segment = memory.ingest(
            "learned-latent",
            "Contexte appris: le prototype Sigma garde une synthese latente.",
            retention_decision=MemoryRetentionDecision(
                segment_id="learned-latent",
                requested_mode=MemoryMode.LATENT,
                applied_mode=MemoryMode.LATENT,
                exact_prob=0.10,
                latent_prob=0.84,
                drop_prob=0.06,
                storage_ratio=0.45,
                confidence=0.84,
                source="learned_memory_policy",
            ),
        )

        self.assertIsNotNone(segment)
        self.assertEqual(segment.mode, MemoryMode.LATENT)
        self.assertEqual(memory.recent.segments, [])
        self.assertEqual([item.segment_id for item in memory.latent.segments], ["learned-latent"])
        report = memory.compression_report()
        self.assertEqual(report["learned_retention_decision_count"], 1)
        self.assertEqual(report["learned_retention_requested_latent"], 1)
        self.assertEqual(report["learned_retention_applied_latent"], 1)

    def test_learned_retention_policy_can_drop_non_anchored_segment(self):
        memory = CognitiveMemory(CognitiveMemoryConfig(recent_exact_limit=2, embedding_dim=32))
        segment = memory.ingest(
            "learned-drop",
            "Texte banal sans ancre critique ni obligation exacte.",
            retention_decision=MemoryRetentionDecision(
                segment_id="learned-drop",
                requested_mode=MemoryMode.DROP,
                applied_mode=None,
                exact_prob=0.08,
                latent_prob=0.12,
                drop_prob=0.80,
                storage_ratio=0.0,
                confidence=0.80,
                source="learned_memory_policy",
            ),
        )

        self.assertIsNone(segment)
        self.assertEqual(memory.recent.segments, [])
        self.assertEqual(memory.latent.segments, [])
        report = memory.compression_report()
        self.assertEqual(report["learned_retention_requested_drop"], 1)
        self.assertEqual(report["learned_retention_applied_drop"], 1)

    def test_learned_retention_policy_cannot_drop_exact_anchor_obligation(self):
        memory = CognitiveMemory(CognitiveMemoryConfig(recent_exact_limit=2, embedding_dim=32))
        required = Anchor("identifier", "C3-3141-A", "learned-anchor")
        segment = memory.ingest(
            "learned-anchor",
            "FAIT CRITIQUE: Ada conserve le code C3-3141-A.",
            extra_anchors=(required,),
            retention_decision=MemoryRetentionDecision(
                segment_id="learned-anchor",
                requested_mode=MemoryMode.DROP,
                applied_mode=None,
                exact_prob=0.03,
                latent_prob=0.07,
                drop_prob=0.90,
                storage_ratio=0.0,
                confidence=0.90,
                source="learned_memory_policy",
            ),
        )

        self.assertIsNotNone(segment)
        self.assertEqual(segment.mode, MemoryMode.EXACT)
        self.assertEqual([item.segment_id for item in memory.recent.segments], ["learned-anchor"])
        reconstruction = memory.reconstruct("retrouve le code exact", required_anchors=(required,))
        self.assertTrue(reconstruction.fidelity.passed)
        self.assertIn("C3-3141-A", reconstruction.rendered_context)
        report = memory.compression_report()
        self.assertEqual(report["learned_retention_requested_drop"], 1)
        self.assertEqual(report["learned_retention_applied_exact"], 1)
        self.assertEqual(report["learned_retention_anchor_overrides"], 1)

    def test_recent_exact_kv_eviction_creates_latent_old_kv(self):
        memory = CognitiveMemory(CognitiveMemoryConfig(recent_exact_limit=1, embedding_dim=32))
        memory.ingest("s1", "FAIT CRITIQUE: Mira a le code C3-1111-A et le montant 42,00 EUR.")
        memory.ingest("s2", "Note récente: Noah garde le badge dans le bureau.")

        self.assertEqual([segment.segment_id for segment in memory.recent.segments], ["s2"])
        self.assertEqual([segment.segment_id for segment in memory.latent.segments], ["s1"])
        latent = memory.latent.segments[0]
        self.assertEqual(latent.mode, MemoryMode.LATENT)
        self.assertEqual(latent.exact_text, "")
        self.assertLess(latent.stored_token_count, latent.original_token_count)
        self.assertTrue(any(anchor.value == "C3-1111-A" for anchor in latent.anchors))

    def test_query_conditioned_reconstruction_preserves_old_exact_anchors(self):
        memory = CognitiveMemory(CognitiveMemoryConfig(recent_exact_limit=1, embedding_dim=64, top_k_latent=2))
        required_code = Anchor("identifier", "C3-7777-Z", "legacy")
        required_amount = Anchor("amount", "913,45 EUR", "legacy")
        memory.ingest(
            "legacy",
            "Ancien contexte: Lina a signé le contrat alpha. Code exact C3-7777-Z. Montant exact 913,45 EUR.",
            extra_anchors=(required_code, required_amount),
        )
        memory.ingest("recent", "Contexte récent sans code: le bureau change de salle.")

        reconstruction = memory.reconstruct("Quel est le code exact et le montant exact du contrat alpha ?", required_anchors=(required_code, required_amount))
        self.assertIn("legacy", reconstruction.selected_segment_ids)
        self.assertIn("C3-7777-Z", reconstruction.rendered_context)
        self.assertIn("913,45 EUR", reconstruction.rendered_context)
        self.assertTrue(reconstruction.fidelity.passed)
        self.assertGreater(reconstruction.cost.effective_cost(), 0.0)

    def test_query_conditioning_prefers_relevant_latent_segment(self):
        memory = CognitiveMemory(CognitiveMemoryConfig(recent_exact_limit=1, embedding_dim=64, top_k_latent=1))
        memory.ingest("archive-alpha", "Archive alpha: le prototype de Sofia utilise le code C3-2222-A.")
        memory.ingest("archive-beta", "Archive beta: la recette publique parle de météo et de jardin.")
        memory.ingest("recent", "Message récent sans rapport.")

        reconstruction = memory.reconstruct("Retrouve le code du prototype Sofia.")
        self.assertIn("archive-alpha", reconstruction.selected_segment_ids)
        self.assertNotIn("archive-beta", reconstruction.selected_segment_ids)
        self.assertIn("C3-2222-A", reconstruction.rendered_context)

    def test_anchor_fidelity_verifier_fails_when_required_anchor_is_missing(self):
        verifier = AnchorFidelityVerifier()
        required = (Anchor("identifier", "C3-4040-X", "missing"),)
        result = verifier.verify("aucune ancre ici", required)
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.missing, required)

    def test_embedding_is_torch_tensor_and_deterministic(self):
        first = embed_text("alpha code exact", 32)
        second = embed_text("alpha code exact", 32)
        self.assertEqual(tuple(first.shape), (32,))
        self.assertTrue(bool(first.equal(second)))

    def test_compiled_circuit_memory_binding_survives_latent_eviction(self):
        memory = CognitiveMemory(CognitiveMemoryConfig(recent_exact_limit=1, embedding_dim=64, top_k_latent=2))
        binding = memory.bind_compiled_circuit(
            circuit_id="circuit-alpha-123",
            skill="algebra",
            source_kind="sleep_consolidation",
            source_failure_ids=("sleep-example-1",),
            frontier_task_ids=("frontier-train-1", "frontier-train-2"),
            heldout_task_ids=("frontier-heldout-1",),
            prompt_obligations=("exact_output", "no_extra_text"),
            metadata_keys=("a", "b", "c"),
            anchors=(Anchor("variable", "x", "frontier-train-1"),),
        )
        memory.ingest("recent", "Nouveau segment qui force le circuit compile en latent KV.")

        restored_binding, reconstruction = memory.reconstruct_compiled_circuit_binding("circuit-alpha-123")

        self.assertEqual(restored_binding.binding_id, binding.binding_id)
        self.assertTrue(reconstruction.fidelity.passed)
        self.assertIn(binding.segment_id, reconstruction.selected_segment_ids)
        self.assertIn("circuit-alpha-123", reconstruction.rendered_context)
        report = memory.compression_report()
        self.assertEqual(report["compiled_circuit_memory_binding_count"], 1)
        self.assertTrue(report["compiled_circuit_memory_bindings"][0]["passed"])

    def test_cycle_run_artifacts_can_include_cognitive_memory_report(self):
        verifier = DynamicSkillVerifier(default_skill_specs())
        report = CortexCycle(verifier).run(ReferenceRuleAgent(), CorruptedCompressedAgent(), seed=5, n_per_skill=1)
        memory = CognitiveMemory(CognitiveMemoryConfig(recent_exact_limit=1, embedding_dim=32))
        memory.ingest("old", "Ancien fait: le code exact C3-9999-A doit survivre.")
        memory.ingest("new", "Nouveau fait sans identifiant.")

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_cycle_run(report, output_dir=tmp, run_id="memory-run", memory=memory)
            payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
            self.assertIsNotNone(payload["cognitive_memory"])
            self.assertEqual(payload["cognitive_memory"]["latent_segments"], 1)
            self.assertLess(payload["cognitive_memory"]["latent_compression_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
