# Cortex-3 Architecture Self-Critique

Etat: boucle d'audit 1 apres integration du kernel CUDA natif tuilé/warp.

Ce document sert de registre de critique et de correction. Il ne remplace pas les tests longs interdits pour cette iteration; il se limite aux preuves courtes disponibles, aux rapports du code et aux benchmarks GPU courts.

## Boucle 1 - Corrections executees maintenant

### C1. Kernel ternaire natif trop simple

- Critique: le premier backend `native_int2_cupy_cuda` avait un thread par sortie et relisait/decodait les memes poids pour chaque ligne. C'etait un vrai kernel, mais pas assez ambitieux pour le statut "kernel fusionne efficace".
- Correction: ajout de deux kernels hand-written:
  - `tiled_shared_memory_int2`: tile 16x16 sorties, tile K=32, X et poids ternaires decodes en shared memory, bias/residual integres, accumulation fp32.
  - `warp_reduction_int2`: une sortie par warp, reduction K sur 32 lanes, utile pour K plus grand.
- Integration: `BitLinear` garde le forward ternaire packe + STE, selectionne `auto/tiled/warp`, trace le backend et la variante dans `PackedTernaryDispatch`, et les rapports LLM exposent `native_ternary_kernel_variants`.
- Verification courte: tests CUDA fp32/fp16/bf16, gradient STE, full Cortex court et benchmarks courts RTX 5070.
- Statut: corrige pour cette boucle, mais pas encore "termine final" car il manque autotuning large, energie/VRAM et packaging CUDA/C++ bas niveau.

### C2. Auto-selection kernel trop grossiere

- Critique: le premier seuil envoyait `batch=128,in=256,out=256` vers `warp`, alors que `tiled` etait plus rapide.
- Correction: seuil `auto` ajuste: `warp` seulement pour `in_features >= 384` et produit sorties suffisant; sinon `tiled`.
- Verification courte: benchmark auto 256 choisit `tiled_shared_memory_int2`; benchmark auto 512 choisit `warp_reduction_int2`.
- Statut: corrige pour les deux formes courtes testees; reste a remplacer par autotuner mesure/cache.

### C3. Observabilite des kernels insuffisante

- Critique: les rapports comptaient les dispatchs natifs sans dire quelle variante tournait.
- Correction: `CompressionTraceLedger.to_dict()` expose `native_ternary_kernel_variants`; le rapport full Cortex l'ajoute dans `training_influence`.
- Verification courte: test full Cortex exige une variante non vide quand le kernel natif est requis.
- Statut: corrige.

### C4. Memoire apprise sans ablation directe

- Critique: le test initial prouvait gradient + branchement, mais pas que la politique exact/latent/drop pouvait ameliorer la loss par rapport a une memoire non apprise.
- Correction: ajout de `tools/benchmark_learned_memory_policy.py` et du test `test_learned_memory_ablation_shows_policy_can_reduce_loss`. L'ablation charge des poids partages, compare memoire apprise active vs desactivee, fige les parametres non-memoire et entraine seulement `learned_memory.*`.
- Verification courte: l'ablation exige gradient memoire non nul, delta `before - after` positif sur total et next-token loss, deplacement des probabilites exact/latent/drop et decisions comptabilisees.
- Statut: corrige pour la preuve courte; reste a scaler sur long contexte held-out et cout memoire reel.

## Critique phase par phase

### P1 - Verifier OS

- Ce qui est solide: familles arithmetic, algebra, long_context_anchor, entity_tracking, instruction_following, code_unit_tests, calibration; oracles stricts; metamorphic/anti-metamorphic; fault matrix; cout verifier par cas.
- Preuve actuelle: tests P1 et full Cortex court activent P1; rapports de cycle persistent les resultats.
- Faiblesse: generateurs encore limites, peu de domaines held-out, pas assez de bruit naturel, pas assez de faux positifs/faux negatifs hors familles internes.
- Risque architectural: si P1 est trop petit, P6/P7/P10 optimisent contre un monde trop facile.
- Correction prioritaire restante: elargir les generateurs et ajouter un audit de couverture d'oracles par domaine.

### P2 - Ternary Core

- Ce qui est solide: poids ternaires packes int2, quantization activations, STE, sync des buffers packes pendant training, kernels CUDA natifs tuiles/warp, audit LLM exigeant native kernel sur CUDA.
- Preuve actuelle: tests CUDA courts, benchmark RTX 5070, full Cortex court.
- Faiblesse: kernels encore CuPy/NVRTC, pas extension C++/CUDA packagee; autotuning heuristique; pas de mesure energie/VRAM longue; pas encore de backward kernel custom.
- Risque architectural: le forward ternaire natif existe, mais le gradient passe encore par STE dense; le gain training total depend donc du cout supplementaire du chemin STE.
- Correction prioritaire restante: autotuner mesure/cache, profiler le surcout STE, puis reduire le cout gradient sans enlever la semantique ternaire.

### P3 - Future Contract / FSP / MTP

- Ce qui est solide: horizons 1/2/4/8, confidence head, temporal consistency, gate observed-token, ledger et replay.
- Preuve actuelle: tests future-contract et full Cortex court.
- Faiblesse: contrats encore surtout token-level; peu de contrats objectifs haut niveau; pas de large comparaison MTP vs NTP sur held-out.
- Risque architectural: la speculation peut etre correcte sur micro-cas sans prouver un vrai gain de cout/qualite en LLM.
- Correction prioritaire restante: ajouter contrats de sortie/format et mesure de tokens verifies par cout sur petits benchmarks non longs.

### P4 - Memoire cognitive apprise

- Ce qui est solide: policy exact/latent/drop trainable, branchee dans forward, loss, P4 anchor supervision, checkpoints et audit.
- Preuve actuelle: test gradient policy, ablation courte a poids partages, full Cortex court, counters exact/latent/drop/storage.
- Faiblesse: pas encore de preuve que la politique apprise bat une regle deterministe sur long contexte held-out; supervision derivee de loss locale encore simple.
- Risque architectural: la memoire peut etre "apprise" mais pas encore utile ou optimale.
- Correction prioritaire restante: scaler l'ablation sur anchors long-context held-out, sans se limiter au batch controle.

### P5 - Certificate Generator

- Ce qui est solide: latent proof state, certificate head, checksum, tool verification, exact/code/arithmetic/anchor tools.
- Preuve actuelle: tests certificates et inference gates.
- Faiblesse: certificats encore courts et outils limites; peu de preuves multi-etapes; calibration depend de micro-distributions.
- Risque architectural: le certificat peut verifier des reponses simples sans prouver compression de raisonnement complexe.
- Correction prioritaire restante: ajouter cas algebra multi-step et code tests plus riches.

### P6 - Causal Attribution

- Ce qui est solide: ablations sur blocks, KV mode, FSP, precision activation, experts, clustering.
- Preuve actuelle: tests causal attribution.
- Faiblesse: attribution encore surtout sur traces/probes; peu de contrefactuels profonds sur LLM multi-couches larges.
- Risque architectural: P7 peut reparer le symptome dominant mais manquer la vraie cause quand plusieurs modules interagissent.
- Correction prioritaire restante: connecter plus de traces layer-forward natives et learned-memory aux probes P6.

### P7 - Minimal Regrowth

- Ce qui est solide: action space executable, patch modele, non-regression gate, repair/protected loss deltas.
- Preuve actuelle: tests regrowth et full Cortex report.
- Faiblesse: budgets et tolerances encore heuristiques; peu de suivi long terme des patchs accumules.
- Risque architectural: patch local peut masquer une regression plus tardive.
- Correction prioritaire restante: rendre les deltas before/after explicites partout et ajouter audit cumulatif des patchs.

### P8 - Fast/Normal/Careful Inference

- Ce qui est solide: router, budget predictor, early exit, memory augmentation, future contracts, certificate gate, ternary kernel dispatch traces.
- Preuve actuelle: tests inference.
- Faiblesse: pas encore branche directement au checkpoint LLM complet pour generation large; kernel dispatcher inference et BitLinear training doivent rester alignes.
- Risque architectural: P8 peut etre une boucle executable correcte mais pas encore le vrai decodeur du LLM pre-entraine.
- Correction prioritaire restante: unifier davantage les metadata kernel entre P8 dispatcher et `BitLinear` natif.

### P9 - Sleep / Consolidation Buffer

- Ce qui est solide: replay failures, synthetic verified pool, real reservoir, anti-collapse filter, schedule.
- Preuve actuelle: tests sleep et full Cortex replay counts.
- Faiblesse: reservoir reel petit; pas de politique d'oubli/consolidation mesuree sur longue duree.
- Risque architectural: sleep peut ajouter du replay utile sans prouver consolidation durable.
- Correction prioritaire restante: ablation courte replay on/off sur micro-LLM et audit diversity drift.

### P10 - Recursive Improvement

- Ce qui est solide: proposal generator, sandbox, Pareto gate, rollback, diversity archive.
- Preuve actuelle: tests recursive improvement et full Cortex report.
- Faiblesse: propositions encore bornees par petit action space; peu de verification contre reward hacking hors suites internes.
- Risque architectural: boucle d'amelioration peut etre trop conservatrice ou trop dependante du verifier P1.
- Correction prioritaire restante: renforcer adversarial/reward-hacking probes et signer plus de metadata de patch.

## Critique des composants du schema cible

### Input

- Statut: corpus texte streamable, tokenizer BPE, memmap causal et future targets.
- Faiblesse: pas encore assez de vrais corpus massifs verifies dans les runs recents sous le nouveau kernel.
- Correction restante: nouveau run court manifest CUDA avec audit natif, puis run long seulement quand autorise.

### Variable-In Compressor

- Statut: differenciable, branche avant blocks, trace KV/compression.
- Faiblesse: compression objective encore simple; pas assez d'ablation qualite/cout.
- Correction restante: ajouter test court compression-on/off sur anchors.

### Exact Anchor Ledger

- Statut: ancres decodees depuis batchs et fidelite verifiee.
- Faiblesse: depend de detection texte heuristique.
- Correction restante: enrichir detection anchors et exigences par domaine.

### Latent Memory / KV

- Statut: exact recent + latent old + reconstruction.
- Faiblesse: utilite latente apprise non prouvee large.
- Correction restante: ablation learned memory vs deterministic memory.

### Causal + Skill Ledgers

- Statut: persistants et actifs dans full Cortex court.
- Faiblesse: pas assez de cross-links entre kernel/memory/certificate causes.
- Correction restante: enrichir les events avec IDs de module et variants kernel.

### Ternary Core W in {-1,0,+1}

- Statut: le forward lit les codes packes int2 et lance CUDA natif sur GPU.
- Faiblesse: backward dense STE, autotuning heuristique.
- Correction restante: profiler et reduire le cout STE; cache autotune.

### Skill-aware Experts

- Statut: MoE trainable et events experts.
- Faiblesse: routing encore peu supervise par skill ledger.
- Correction restante: ajouter regularisation skill-ledger/routing.

### Future Contract / FSP

- Statut: contrats token horizon.
- Faiblesse: contrats de but final limites.
- Correction restante: output-goal contracts.

### Adaptive Multi-Token Decoding

- Statut: MTP/FSP + P8 speculative paths.
- Faiblesse: pas encore decodeur LLM large.
- Correction restante: brancher a generation LLM checkpoint.

### Latent Reasoning Workspace

- Statut: latent proof/certificate and P8 latent loops.
- Faiblesse: workspace encore implicite, pas un buffer raisonnement multi-step general.
- Correction restante: rendre workspace explicite dans reports et loss.

### Certificate Generator

- Statut: head + tools + checksum.
- Faiblesse: domaines outils limites.
- Correction restante: algebra/code multi-step.

### Hierarchical Dynamic Verifier

- Statut: accept/reject, regression attribution, regrowth, sleep.
- Faiblesse: hierarchie encore surtout orchestrateur de modules; pas de policy apprise de profondeur verifier.
- Correction restante: verifier-depth policy and cost calibration.

## File de correction priorisee apres boucle 1

1. P2: ajouter autotune mesure/cache par shape/dtype/variant pour ne plus utiliser un seuil fixe.
2. P4: scaler l'ablation learned memory vs deterministic memory sur anchors long-context synthetiques puis held-out.
3. P6/P7: afficher partout `repair_loss_before`, `repair_loss_after`, `protected_loss_before`, `protected_loss_after`, delta et convention.
4. P8: aligner `TernaryKernelDispatcher` inference avec les variants `BitLinear` natifs.
5. P1: ajouter un audit de couverture oracle/generateur par famille.
6. P3: ajouter contrats output-goal non token seulement.
7. P5: ajouter certificats algebra multi-step.
8. P9: audit diversity drift replay/sleep court.
9. P10: renforcer reward-hacking probes.
10. Training: produire un nouveau sidecar sous le commit courant quand les tests longs seront autorises.
