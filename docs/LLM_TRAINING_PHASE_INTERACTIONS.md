# Cortex-3 LLM Training Phase Interactions

Etat verifie le 2026-07-01 depuis le run local `runs/cortex3-c4-cuda-large-fullphases-20260630_133618`.

Ce document explique comment l'architecture Cortex-3 complete agit pendant un entrainement LLM reel. Le but est de separer clairement trois niveaux :

- **present dans le code** : un module existe.
- **branche dans le training** : le module est appele pendant le forward, le loss, l'audit ou le checkpoint.
- **influent sur l'apprentissage** : le module modifie le gradient, le replay, les poids, les gates de preuve ou les metriques qui conditionnent la poursuite du run.

## Preuves Runtime Actuelles

Dernier sidecar inspecte :

- checkpoint : `checkpoint_step_165.pt.json`
- commit : `1e6366135e62097116e51e80ab7ab83a3c192da9`
- architecture audit : `True 22/22`
- phase deliverables : `True 10/10`
- erreurs phase : `0`
- termes de l'objectif final Cortex : `17/17`
- GPU moyen observe : `95.7%`
- CPU moyen observe : `36.4%`
- VRAM moyenne observee : `11808.3 MB`
- processus long run actifs : `tools/train_llm.py run-experiment experiments/c4_cuda_large_manifest.json`

Evenements phase observes dans le checkpoint :

| Phase | Evenements | Replay |
| --- | ---: | ---: |
| P1 | 3 | 3 |
| P2 | 225212 | 0 |
| P3 | 3 | 3 |
| P4 | 3 | 3 |
| P5 | 3 | 3 |
| P6 | 3 | 3 |
| P7 | 3 | 3 |
| P8 | 3 | 7 |
| P9 | 3 | 24 |
| P10 | 3 | 3 |

P7 et P10 ont maintenant une preuve de modification reelle du modele :

| Gate | Applications | Delta poids L1 | Repair loss delta |
| --- | ---: | ---: | ---: |
| P7 minimal regrowth | 1 | 880.410583 | 1.239653 |
| P10 recursive improvement | 1 | 523.004089 | 0.056110 |

Ces valeurs prouvent que les deux gates ne se limitent plus a produire du texte, des rapports ou du replay. Ils ont applique des patchs bornes sur les vrais parametres du Transformer, puis ont mesure une amelioration de loss sur la cible de reparation.

## Boucle LLM Complete

Le pipeline d'entrainement Cortex-3 part d'un vrai flux LLM :

1. Le corpus texte est exporte ou lu depuis des shards.
2. Un tokenizer BPE est entraine avec `tokenizers`.
3. Les textes sont encodes une fois dans un fichier memmap `uint32`.
4. `MemmapCausalDataset` lit des fenetres causales vectorisees.
5. Le Transformer calcule :
   - logits next-token,
   - logits multi-horizon,
   - confiance token/sequence,
   - etat de compression Variable-In,
   - certificat latent,
   - traces du coeur ternaire,
   - activations MoE skill-aware.
6. Le loss principal entraine le modele comme un LLM causal classique.
7. Le loss Cortex ajoute multi-horizon, temporal consistency, confidence, variable input et certificate.
8. Le controleur P1-P10 observe les batchs, lance les phases, produit replay, ledgers, audits, patchs P7/P10 et objectif final.
9. Les checkpoints persistent modele, optimizer, scaler, RNG, replay, ledgers, phase state et sidecars d'audit.

Les points d'integration importants sont :

- `CortexTransformerLM.forward` : applique Variable-In, blocs Transformer, MoE, heads MTP/confiance/certificat.
- `CortexObjective.compute` : transforme ces sorties en loss trainable.
- `LLMTrainer.train` : ajoute `auxiliary_loss`, `replay_loss`, optimizer step et `requantize_ternary_core`.
- `CortexTrainingPhaseController.run_phase_audit` : execute P1-P10.
- `checkpoint_state_summary` / `summary` : prouve que les phases ont vraiment tourne et influence le training.

## Influence Directe Sur L'Apprentissage

Cortex-3 influence le LLM par quatre canaux distincts.

### 1. Forward Architecturel

Le modele n'est pas un Transformer standard habille par un rapport. Son forward contient :

- `BitLinear` pour le coeur ternaire ;
- `VariableInCompressor` pour compression adaptative ;
- `SkillAwareExpertMoE` pour experts skill-aware ;
- tetes MTP multi-horizon ;
- tete de confiance ;
- `CertificateHead`.

Ces modules participent au graphe PyTorch et peuvent recevoir du gradient.

### 2. Loss Cortex

Le loss total contient :

- next-token loss ;
- multi-horizon loss ;
- temporal consistency ;
- calibration/confiance ;
- penalite de compression Variable-In ;
- certificate loss.

Donc le modele n'apprend pas seulement a predire le prochain token. Il apprend aussi a predire des horizons futurs, calibrer sa confiance, produire des preuves latentes et compresser differemment selon l'importance des tokens.

### 3. Replay Causal Verifie

Les phases P1-P10 produisent des exemples verifies et les encodent avec le tokenizer actif. Ces exemples deviennent des batchs de replay causal. Le replay est injecte dans `replay_loss`, donc il agit sur le gradient comme un mini-corpus specialise issu du verifier.

Dans le checkpoint inspecte :

- P1 replay : 3
- P3 replay : 3
- P4 replay : 3
- P5 replay : 3
- P6 replay : 3
- P7 replay : 3
- P8 replay : 7
- P9 replay : 24
- P10 replay : 3

P2 n'a pas de replay parce que son role est dans le forward et les traces de compression, pas dans la creation d'exemples textuels.

### 4. Patchs Directs De Parametres

P7 et P10 appliquent des modifications bornees aux vrais parametres du Transformer.

P7 :

- part d'une regression attribuee ;
- choisit une action de regrowth ;
- calcule un patch cible ;
- mesure loss avant/apres ;
- verifie la non-regression sur batchs proteges ;
- requantifie le coeur ternaire ;
- rollback si le gate echoue.

P10 :

- part d'une proposition acceptee par sandbox ;
- signe le patch via `signed_patch_id` ;
- garde un `rollback_token` ;
- applique un patch borne ;
- mesure repair loss et protected loss ;
- requantifie le coeur ternaire.

C'est le point qui distingue le plus clairement une vraie integration d'un prototype : le systeme peut maintenant transformer une decision verifiee en changement de poids.

## Architecture Cible Et Correspondance Runtime

Schema cible du README :

```text
Input
  -> Variable-In Compressor
  -> Exact Anchor Ledger
  -> Latent Memory / KV
  -> Causal + Skill Ledgers
  -> Ternary Core + Skill-aware Experts
  -> Future Contract / FSP
  -> Adaptive Multi-Token Decoding
  -> Latent Reasoning Workspace
  -> Certificate Generator
  -> Hierarchical Dynamic Verifier
  -> accept / reject / attribute regression / minimal regrowth / sleep
```

Correspondance runtime :

| Architecture cible | Implementation LLM | Preuve d'usage |
| --- | --- | --- |
| Variable-In Compressor | `VariableInCompressor` dans le forward | audit `variable_in_compressor` |
| Exact Anchor Ledger | observation de batchs decodes + anchors | audit `exact_anchor_ledger`, zero fidelity failure |
| Latent Memory / KV | `CognitiveMemory` recent exact + latent old KV | audit `latent_memory_kv` |
| Causal Ledger | `CausalLedger` + traces P1/P3/P4/P8/replay | audit `causal_ledger` |
| Skill Ledger | `SkillLedger.update_from_report` | audit `skill_ledger` |
| Ternary Core | `BitLinear`, quantization, requantization | P2 `225212` events |
| Skill-aware Experts | `SkillAwareExpertMoE` | audit `skill_aware_experts` |
| Future Contract / FSP | `FutureContractEngine` + observed tokens | P3 replay + contract decisions |
| Adaptive Multi-Token Decoding | MTP horizons + inference route | audit `adaptive_multi_token_decoding` |
| Latent Reasoning Workspace | `LatentProofState` + cert head | P5 audit |
| Certificate Generator | `CertificateHead` + verifier | P5 certificate verification |
| Hierarchical Dynamic Verifier | `DynamicSkillVerifier` | P1 + no phase errors |
| Attribute Regression | `CausalAttributionEngine` | P6 |
| Minimal Regrowth | `MinimalRegrowthEngine` + model patch | P7 patch evidence |
| Sleep / consolidation | `SleepPhaseConsolidator` | P9 replay/synthetic/reservoir |
| Recursive Improvement | `RecursiveImprovementEngine` + signed model patch | P10 patch evidence |

## Phase 1 - Verifier OS

### Entree

P1 recoit un cycle reference/trial. Le reference agent represente le comportement attendu, le trial agent represente une version compressee/corrompue.

### Travail Execute

Le verifier teste plusieurs familles de competences :

- arithmetic ;
- algebra ;
- long context anchors ;
- entity tracking ;
- instruction following ;
- code unit tests ;
- calibration.

Il produit un `CycleReport` avec :

- cas passes/echoues ;
- regressions ;
- scores ;
- couts de verification ;
- skill reports ;
- actions preliminaires.

### Interaction Avec Les Autres Phases

P1 fournit les erreurs initiales a P6, P7, P9 et P10. Sans P1, les autres phases n'ont pas de regression verifiee a analyser.

### Impact Apprentissage

P1 influence le training en :

- mettant a jour `SkillLedger` ;
- ajoutant couts au `BitLedger` ;
- ajoutant observations de confiance a `UncertaintyLedger` ;
- creant du replay P1 ;
- alimentant les phases de correction.

### Pourquoi Ce N'est Pas Un Prototype

P1 n'est pas un simple assert. Il cree une trace persistee et exploitee par les phases suivantes.

## Phase 2 - Ternary Core

### Entree

P2 agit directement dans le forward du Transformer.

### Travail Execute

Quand `use_ternary_core=True`, les lineaires principaux deviennent des `BitLinear`. Ces couches maintiennent :

- poids flottants entrainables ;
- signes ;
- masque ternaire ;
- scales ;
- residual weights ;
- quantization d'activation ;
- layer forward events.

### Interaction Avec Les Autres Phases

P2 fournit les traces de compression a :

- P6, pour attribuer des echecs a la compression ;
- P7/P10, car les patchs requantifient ensuite le coeur ternaire ;
- BitLedger, pour le cout effectif.

### Impact Apprentissage

Le gradient passe par les poids flottants avec une approximation runtime quantifiee. Le modele apprend donc dans un regime compatible avec une execution ternaire, pas seulement dans un float Transformer classique.

### Preuve Runtime

Le checkpoint inspecte montre `P2=225212` evenements. C'est beaucoup plus qu'un smoke test ponctuel : le coeur ternaire tourne dans le run long.

## Phase 3 - Future Contract / FSP / MTP

### Entree

P3 observe les logits multi-horizon produits par les têtes MTP et les compare a des futurs tokens reels du batch.

### Travail Execute

Le controleur :

- extrait les logits par horizon ;
- calcule confiance et temporal loss ;
- draft un future contract ;
- compare le contrat aux tokens observes ;
- accepte ou rejette ;
- enregistre la decision dans le future ledger.

### Interaction Avec Les Autres Phases

P3 nourrit :

- `FutureContractLedger` ;
- `UncertaintyLedger` ;
- `CausalLedger` ;
- replay P3 ;
- objectif final `L_future_contract` et `L_multi_horizon`.

### Impact Apprentissage

Le modele est pousse a predire plusieurs horizons sans tricher : les contrats sont verifies contre les futurs tokens observes. Cela evite que MTP soit seulement une tete supplementaire inutilisee.

## Phase 4 - Cognitive Memory / KV / Anchor Ledger

### Entree

P4 observe des tokens LLM reels, les decode en texte, puis cherche des ancres exactes.

### Travail Execute

Le controleur :

- decode un sous-ensemble du batch ;
- cree des segments memoire ;
- extrait les anchors ;
- reconstruit via memoire recent exact + latent old KV ;
- verifie la fidelite des anchors ;
- enregistre erreurs si une ancre disparait.

### Interaction Avec Les Autres Phases

P4 alimente :

- P8, qui peut utiliser la memoire pendant inference ;
- P6, si une regression vient de perte d'ancre ;
- `L_anchor_fidelity` ;
- `CausalLedger`.

### Impact Apprentissage

La memoire n'est pas un stockage passif. Les exemples d'ancrage deviennent du replay et les echecs de fidelite peuvent faire echouer l'audit. Le modele est donc contraint a ne pas gagner du cout en perdant les informations exactes.

## Phase 5 - Latent Reasoning Workspace / Certificates

### Entree

P5 part d'une tache issue du cycle ou d'une tache controlee.

### Travail Execute

La phase cree :

- un `LatentProofState` ;
- un certificat court ;
- une verification par outil ;
- une random de-latentization probe ;
- une mesure d'efficacite certificat vs raisonnement visible.

### Interaction Avec Les Autres Phases

P5 alimente :

- `BitLedger` via cout de certificat ;
- `CausalLedger` via preuve/certificat ;
- replay P5 ;
- `L_latent_certificate`.

### Impact Apprentissage

Le modele possede une `CertificateHead` dans le forward. Le loss de certificat pousse la tete a produire une reponse finale et une incertitude coherente. P5 rend cette tete auditable au lieu de laisser le latent reasoning devenir une boite noire.

## Phase 6 - Causal Attribution

### Entree

P6 recoit une regression verifiee par P1.

### Travail Execute

L'attribution lance des probes contre plusieurs dimensions :

- block overcompressed ;
- expert path ;
- KV mode ;
- MTP horizon ;
- activation precision ;
- FSP contract ;
- routing.

Elle produit :

- causes estimees ;
- probabilites ;
- recovered/non-recovered ;
- gain par cout ;
- top cause.

### Interaction Avec Les Autres Phases

P6 est le pont entre detection et correction :

- P1 detecte ;
- P6 explique ;
- P7 repare ;
- P10 peut proposer une amelioration plus generale.

### Impact Apprentissage

P6 evite de transformer toute regression en retraining global aveugle. Il donne une cible de correction, ce qui rend P7 possible et mesure le cout relatif d'une reparation locale.

## Phase 7 - Minimal Regrowth

### Entree

P7 recoit le rapport d'attribution P6.

### Travail Execute

P7 :

1. construit un espace d'actions ;
2. simule les actions ;
3. mesure score avant/apres ;
4. verifie non-regression ;
5. choisit l'action recouvrante la moins couteuse ;
6. genere replay P7 ;
7. applique un patch borne aux vrais parametres du Transformer ;
8. mesure repair loss avant/apres ;
9. mesure protected loss ;
10. requantifie le coeur ternaire ;
11. rollback si le gate echoue.

### Interaction Avec Les Autres Phases

P7 depend de P1/P6 et agit sur P2 :

- P1 donne la regression ;
- P6 donne la cause ;
- P7 choisit la correction ;
- P2 est requantifie apres patch ;
- P9 peut ensuite consolider des exemples lies.

### Impact Apprentissage

P7 agit par deux chemins :

- replay causal ;
- modification directe des poids.

La modification directe est importante : elle fait de P7 un vrai mecanisme de regrowth, pas seulement une generation d'exemples.

### Preuve Runtime

Checkpoint `step 165` :

- applications P7 : `1`
- delta poids L1 : `880.410583`
- repair loss delta : `1.239653`

## Phase 8 - Adaptive Inference

### Entree

P8 recoit une tache et force ou choisit un chemin :

- fast ;
- normal ;
- careful.

### Travail Execute

L'inference route selon :

- difficulte ;
- confiance ;
- exactness ;
- risque ;
- type de competence.

Le chemin choisi controle :

- profondeur ;
- verifier strength ;
- MTP horizon ;
- latent KV ;
- expert activation ;
- early exit ;
- self-speculative future contracts ;
- dispatch kernel ternaire.

### Interaction Avec Les Autres Phases

P8 utilise :

- P4 memoire ;
- P5 certificats ;
- P3 future contracts ;
- P2 kernel ternaire ;
- verifier P1 pour audit final.

### Impact Apprentissage

P8 fournit des exemples verifies par route et mesure la capacite par cout effectif. Il aide le modele a apprendre une politique ou la qualite verifiee compte plus que le debit brut.

## Phase 9 - Sleep / Consolidation

### Entree

P9 recoit le cycle P1, donc les echecs, reussites, fragilites et competences.

### Travail Execute

P9 construit :

- replay de failures ;
- exemples tool-solved ;
- variantes metamorphiques ;
- reservoir reel/exogene ;
- filtre anti-collapse ;
- schedule de consolidation.

Le filtre rejette les exemples qui menacent :

- calibration ;
- diversite ;
- contamination ;
- duplication ;
- collapse de competences rares.

### Interaction Avec Les Autres Phases

P9 consolide les signaux de P1/P6/P7/P8/P10 en replay entrainable.

### Impact Apprentissage

P9 est une memoire d'entrainement verifiee. Elle transforme les corrections lentes et verifiees en exemples causalement entrainables. Dans le checkpoint, P9 apporte `24` replays, soit la source de replay la plus dense observee.

## Phase 10 - Recursive Improvement

### Entree

P10 part du `CycleReport`, des actions et regressions.

### Travail Execute

P10 :

1. genere des propositions ;
2. les teste en sandbox ;
3. evalue qualite, cout, robustesse ;
4. verifie protected skills ;
5. detecte calibration regression ;
6. detecte reward hacking ;
7. verifie diversite/collapse ;
8. accepte ou rejette ;
9. archive la decision ;
10. cree rollback token ;
11. applique une proposition acceptee comme patch signe sur vrais poids Transformer ;
12. mesure repair loss, protected loss et delta de poids.

### Interaction Avec Les Autres Phases

P10 est la boucle d'amelioration recursive au-dessus des autres phases :

- P1 detecte ;
- P6 explique ;
- P7 repare localement ;
- P10 propose une amelioration plus generale ;
- P2 est requantifie apres patch ;
- P9 peut consolider l'effet.

### Impact Apprentissage

P10 agit par :

- replay P10 ;
- patch direct des poids ;
- signaux vers `L_recursive_improvement_validity`.

### Preuve Runtime

Checkpoint `step 165` :

- applications P10 : `1`
- delta poids L1 : `523.004089`
- repair loss delta : `0.056110`

## Objectif Final 17 Termes

Le checkpoint contient les 17 termes :

1. `L_behavior`
2. `L_multi_horizon`
3. `L_future_contract`
4. `L_distillation_behavior`
5. `L_distillation_uncertainty`
6. `L_latent_certificate`
7. `L_invariance`
8. `L_temporal_consistency`
9. `L_total_cognitive_description`
10. `L_no_cost_shifting`
11. `L_hardware_layout`
12. `L_skill_regression`
13. `L_calibration`
14. `L_anchor_fidelity`
15. `L_regrowth_efficiency`
16. `L_verifier_resistance`
17. `L_recursive_improvement_validity`

Ces termes ne remplacent pas tous directement le loss PyTorch token-level. Ils servent aussi de feedback scale et de preuve de controle global. La boucle d'apprentissage combine donc :

- loss differentiable direct ;
- replay ;
- patchs de parametres ;
- audits bloquants.

## Pourquoi Cela Depasse Le Prototype

Un prototype se contenterait de :

- modules separes ;
- tests unitaires sans training ;
- rapports sans effet sur les poids ;
- pas de checkpoint ;
- pas de GPU reel ;
- pas d'audit bloquant.

Ici, on observe :

- run long actif sous CUDA ;
- GPU moyen `95.7%` ;
- Transformer causal complet ;
- BPE tokenizer ;
- dataset memmap ;
- AMP/bf16 ;
- checkpoints lies au corpus ;
- P1-P10 actifs ;
- audits `22/22` et `10/10` ;
- replay P1/P3/P4/P5/P6/P7/P8/P9/P10 ;
- P2 massif dans le forward ;
- P7/P10 patchent les vrais poids ;
- objectif final 17 termes ;
- sidecars persistants.

La distinction importante est donc :

- **Implementation integree reelle** : oui, prouvee par le checkpoint actuel.
- **Preuve scientifique finale que Cortex-3 bat la baseline sur benchmark large** : pas encore, car le run long doit finir et `audit-experiment` doit passer.

## Tests Courts Qui Demontrent L'Integration

Les tests courts pertinents sont :

- `test_full_cortex_phase_controller_uses_all_modules_during_training`
  - entraine un mini-run ;
  - exige toutes les phases actives ;
  - exige audits architecture/livrables ;
  - exige replay ;
  - exige P7/P10 model patches.

- `test_cortex_phase_state_survives_checkpoint_resume`
  - sauvegarde checkpoint ;
  - verifie ledgers, replay, objectifs ;
  - verifie P7/P10 patches dans le checkpoint ;
  - reprend l'entrainement et verifie que l'influence continue.

- `test_training_config_rejects_strict_and_auto_resume_together`
  - verifie les configs dangereuses ;
  - inclut la validation du budget P7.

Ces tests ne prouvent pas la performance large finale. Ils prouvent que les pieces sont branchees et qu'elles influencent bien le training.

## Resume Operationnel

Pendant un vrai entrainement LLM Cortex-3 :

1. Le Transformer apprend le next-token comme un LLM classique.
2. Les tetes Cortex apprennent les horizons futurs, la confiance, le certificat et la compression.
3. Le Verifier OS detecte les regressions.
4. Les ledgers transforment cout, competence, causalite et incertitude en etat persistant.
5. La memoire conserve les ancres exactes et les segments latents.
6. L'attribution trouve les causes probables.
7. Regrowth repare localement et modifie les poids si le gate passe.
8. L'inference adaptative mesure les chemins cout/qualite.
9. Sleep consolide les experiences verifiees en replay.
10. Recursive improvement applique des patchs signes aux poids si la proposition passe les gates.
11. Les objectifs finaux et audits empechent de declarer une architecture complete si une brique manque.

Conclusion stricte : Cortex-3 est actuellement une implementation LLM integree et active, pas seulement un prototype de modules. La preuve comparative large reste l'etape experimentale finale.
