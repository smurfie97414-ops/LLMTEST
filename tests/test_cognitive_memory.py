import json
import tempfile
import unittest

from cortex3 import Anchor, CorruptedCompressedAgent, DynamicSkillVerifier, ReferenceRuleAgent, default_skill_specs
from cortex3_cycle import CortexCycle
from cortex3_memory import AnchorFidelityVerifier, CognitiveMemory, CognitiveMemoryConfig, MemoryMode, embed_text
from cortex3_reporting import write_cycle_run


class CognitiveMemoryTest(unittest.TestCase):
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
