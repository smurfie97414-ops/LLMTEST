# Cortex-3 LLM Training Phase Interactions

Etat verifie le 2026-07-01 depuis le run local `runs/cortex3-c4-cuda-large-fullphases-20260630_133618`.

Ce document explique comment l'architecture Cortex-3 complete agit pendant un entrainement LLM reel. Le but est de separer clairement trois niveaux :

- **present dans le code** : un module existe.
- **branche dans le training** : le module est appele pendant le forward, le loss, l'audit ou le checkpoint.
- **influent sur l'apprentissage** : le module modifie le gradient, le replay, les poids, les gates de preuve ou les metriques qui conditionnent la poursuite du run.

## Preuves Runtime Actuelles

Dernier sidecar inspecte :

- checkpoint : `checkpoint_step_175.pt.json`
- commit : `1e6366135e62097116e51e80ab7ab83a3c192da9`
- architecture audit : `True 22/22`
- phase deliverables : `True 10/10`
- erreurs phase : `0`
- termes de l'objectif final Cortex : `17/17`

Observation ressource separee du meme run long :

- GPU moyen observe : `95.7%`
- CPU moyen observe : `36.4%`
- VRAM moyenne observee : `11808.3 MB`
- processus long run actifs : `tools/train_llm.py run-experiment experiments/c4_cuda_large_manifest.json`

Preuve post-integration des deux nouvelles briques :

- `test_learned_memory_policy_is_trainable_and_affects_cortex_loss` : loss `learned_memory`, `skill_expert` et `latent_workspace` non nulles, gradients non nuls dans la politique memoire, le routeur expert et le workspace latent, dispatch ternaire packe present ;
- `test_learned_memory_ablation_shows_policy_can_reduce_loss` : ablation courte a poids partages, memoire apprise active vs desactivee, puis entrainement de la seule politique exact/latent/drop avec delta `before - after` positif sur total et next-token loss ;
- `test_bitlinear_native_packed_ternary_cuda_dispatch_runs_on_gpu` : backend natif `native_int2_*_cuda_*` execute sur le GPU local avec gradient STE non nul ;
- `test_bitlinear_native_extension_cuda_dispatch_runs_on_gpu` : backend extension strict, forward CUDA, backward `grad_input`, backward `grad_weight` + `grad_bias`, et requantize extension prouves par compteurs ;
- `test_native_ternary_cuda_kernel_matches_packed_runtime_for_training_dtypes` : le kernel natif correspond au runtime packe en fp32, fp16 et bf16 ;
- `test_native_ternary_cuda_fast_ste_backward_matches_dense_ste` : le backward fast STE CUDA correspond au dense STE en fp32, fp16 et bf16, avec `grad_input` calcule depuis les poids int2 packes et `grad_weight` + `grad_bias` calcules par extension ;
- `test_native_ternary_cuda_requantize_pack_matches_torch_sync` : la requantization/packing CUDA fusionnee reproduit signs, mask, scales, residuals, packed codes et compte d'actifs du chemin PyTorch en fp32, fp16 et bf16 ;
- `test_full_cortex_phase_controller_uses_all_modules_during_training` : mini training LLM complet avec audits exigeant `learned_cognitive_memory_policy`, `packed_ternary_hardware_runtime`, `native_ternary_cuda_kernel` quand CUDA est disponible et le nouveau composant `future_output_goal_contracts`.
- `test_full_cortex_phase_controller_uses_all_modules_during_training` verifie aussi `latent_workspace_forward_events`, `latent_workspace_step_events` et `latent_workspace_certificate_binding_events`.
- `test_full_cortex_phase_controller_uses_all_modules_during_training` verifie aussi que P5 execute les nouveaux certificats `algebra_linear` et code visible/cache/proprietes pendant le controleur P1-P10.
- `test_cortex_phase_state_survives_checkpoint_resume` : reprise checkpoint avec restauration des decisions output-goal dans la ledger P3, maintien des evenements P5 algebra/code et maintien des audits architecture/livrables.

Le long run devra produire un nouveau sidecar sous le commit de cette integration pour remplacer l'ancien audit `22/22` par l'audit courant plus strict incluant `native_ternary_cuda_kernel` et `future_output_goal_contracts`.

Evenements phase observes dans le checkpoint :

| Phase | Evenements | Replay |
| --- | ---: | ---: |
| P1 | 3 | 3 |
| P2 | 238652 | 0 |
| P3 | 3 | 3 |
| P4 | 3 | 3 |
| P5 | 3 | 3 |
| P6 | 3 | 3 |
| P7 | 3 | 3 |
| P8 | 3 | 7 |
| P9 | 3 | 24 |
| P10 | 3 | 3 |

P7 et P10 ont maintenant une preuve de modification reelle du modele :

| Gate | Applications | Delta poids L1 | Repair before | Repair after | Repair delta | Protected before | Protected after | Protected delta | Tolerance | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P7 minimal regrowth | 1 | 880.410583 | 29.159054 | 27.919401 | 1.239653 | 29.138744 | 27.827297 | -1.311447 | 0.582775 | accepted |
| P10 recursive improvement | 1 | 523.004089 | 27.872873 | 27.816763 | 0.056110 | 43.066746 | 43.136181 | 0.069435 | 0.861335 | accepted |

Ces valeurs prouvent que les deux gates ne se limitent plus a produire du texte, des rapports ou du replay. Ils ont applique des patchs bornes sur les vrais parametres du Transformer, puis ont mesure une amelioration de loss sur la cible de reparation.

## Convention Des Deltas De Reparation

La convention est explicite dans `CortexTrainingPhaseController` :

- `repair_loss_delta = repair_loss_before - repair_loss_after`
- `protected_loss_delta = protected_loss_after - protected_loss_before`

Donc :

- un `repair_loss_delta` positif signifie une amelioration de la loss sur la cible de reparation ;
- un `protected_loss_delta` positif signifie une hausse de loss sur les batchs proteges ;
- le gate accepte seulement si `repair_loss_delta > 0`, `protected_loss_delta <= protected_loss_tolerance` et `parameter_delta_l1 > 0`.

Dans le dernier sidecar inspecte, P7 ameliore aussi les batchs proteges (`protected_delta=-1.311447`). P10 augmente legerement la loss protegee (`+0.069435`), mais reste sous la tolerance (`0.861335`) et passe donc le gate de non-regression.

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
   - politique memoire apprise exact/latent/drop,
   - certificat latent,
   - traces du coeur ternaire et dispatchs packed int2,
   - activations MoE skill-aware.
6. Le loss principal entraine le modele comme un LLM causal classique.
7. Le loss Cortex ajoute multi-horizon, temporal consistency, confidence, variable input, learned memory policy, certificate et pression `L_future_contract` issue aussi des contrats output-goal.
8. Le controleur P1-P10 observe les batchs, lance les phases, produit replay, ledgers, audits, patchs P7/P10 et objectif final.
9. Les checkpoints persistent modele, optimizer, scaler, RNG, replay, ledgers, phase state et sidecars d'audit.
10. A la reprise checkpoint, le controleur restaure d'abord la memoire P4, recharge ensuite le registre Frontier persiste, execute une FastSolve restauree avec binding P4 restaure, output-goal et certificat `compiled_circuit`, puis echoue durement si le checkpoint annonce des circuits mais que le registre manque.

Les points d'integration importants sont :

- `CortexTransformerLM.forward` : applique Variable-In, politique memoire apprise, blocs Transformer, MoE, heads MTP/confiance/certificat.
- `CortexObjective.compute` : transforme ces sorties en loss trainable.
- `LLMTrainer.train` : ajoute `auxiliary_loss`, `replay_loss`, optimizer step et `requantize_ternary_core`.
- `CortexTrainingPhaseController.run_phase_audit` : execute P1-P10.
- `CortexTrainingPhaseController.load_state_dict` : restaure les ledgers, la memoire P4, le registre Frontier, puis prouve une FastSolve restauree avant de continuer le training.
- `checkpoint_state_summary` / `summary` : prouve que les phases ont vraiment tourne, influence le training et conservent les competences compilees au-dela du run courant.

## Influence Directe Sur L'Apprentissage

Cortex-3 influence le LLM par quatre canaux distincts.

### 1. Forward Architecturel

Le modele n'est pas un Transformer standard habille par un rapport. Son forward contient :

- `BitLinear` pour le coeur ternaire ;
- `VariableInCompressor` pour compression adaptative ;
- `LearnedMemoryPolicy` pour decider quoi garder exact, latent ou drop ;
- prior d'utilite memoire appris depuis les reconstructions aval fideles ;
- `SkillAwareExpertMoE` pour experts skill-aware, maintenant conditionne par le Skill Ledger et les skills de replay verifies ;
- `LatentReasoningWorkspace` pour pas latents trainables avec feedback dans le hidden ;
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
- supervision de politique memoire apprise exact/latent/drop basee sur l'utilite token-level ;
- alignement de politique memoire sur un prior exact/latent/drop issu de credits d'utilite aval ;
- certificate loss.
- skill-expert routing loss, qui aligne la distribution du routeur MoE sur les competences fragiles ou les skills de replay.

Donc le modele n'apprend pas seulement a predire le prochain token. Il apprend aussi a predire des horizons futurs, calibrer sa confiance, produire des preuves latentes, compresser differemment selon l'importance des tokens et choisir une politique memoire exact/latent/drop.
Depuis C76, cette politique memoire ne depend plus seulement de la difficulte locale des tokens: les reconstructions P4, les bindings Frontier et les inferences P8 qui selectionnent des segments retenus creent des credits d'utilite. Ces credits normalisent un prior exact/latent/drop reinjecte dans `LearnedMemoryPolicy.forward` et dans `CortexObjective`, donc la memoire apprend progressivement quels choix de retention servent vraiment les phases aval.
Depuis C73, il apprend aussi a router son calcul vers des experts lies aux competences que le Verifier OS et le Skill Ledger jugent fragiles, au lieu de seulement activer des experts generiques.

### 3. Replay Causal Verifie

Les phases P1-P10 produisent des exemples verifies et les encodent avec le tokenizer actif. Ces exemples deviennent des batchs de replay causal. Le replay est injecte dans `replay_loss`, donc il agit sur le gradient comme un mini-corpus specialise issu du verifier.

Dans le checkpoint inspecte :

- P1 replay : 3
- P3 replay : 3
- P4 replay : 3
- P5 replay : 3+ avec replay symbolique verifie lorsque P5 tourne
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
| Latent Memory / KV | `CognitiveMemory` recent exact + latent old KV + `LearnedMemoryPolicy` | audits `latent_memory_kv` et `learned_cognitive_memory_policy` |
| Causal Ledger | `CausalLedger` + traces P1/P3/P4/P8/replay | audit `causal_ledger` |
| Skill Ledger | `SkillLedger.update_from_report` | audit `skill_ledger` |
| Ternary Core | `BitLinear`, quantization, requantization, packed int2 dispatch | P2 `238652` events + audit `packed_ternary_hardware_runtime` |
| Skill-aware Experts | `SkillAwareExpertMoE` conditionne par Skill Ledger/replay | audit `skill_aware_experts`, `skill_expert_context_events`, `skill_expert_replay_context_events` |
| Future Contract / FSP | `FutureContractEngine` + observed tokens | P3 replay + contract decisions |
| Adaptive Multi-Token Decoding | MTP horizons + inference route + model-backed adaptive block gate | audit `adaptive_multi_token_decoding`, `inference_model_backed_adaptive_mtp_*` |
| Latent Reasoning Workspace | `LatentReasoningWorkspace` multi-step + binding cert head | audits `latent_reasoning_workspace`, P5 et checkpoint |
| Certificate Generator | `CertificateHead` + verifier + `sympy_symbolic` | P5 certificate verification, symbolic solver replay |
| Frontier FastSolve persistant | `FrontierCircuitRegistry` + `CompiledFrontierAgent` | registre restaure, binding P4 restaure, output-goal et certificat compile |
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
- residual weights optionnels ;
- codes ternaires packes int2 dans `packed_codes` ;
- quantization d'activation ;
- layer forward events ;
- `PackedTernaryDispatch` avec backend `packed_int2_torch`, `packed_int2_cuda`, `native_int2_extension_cuda_tiled_shared_memory_int2`, `native_int2_extension_cuda_warp_reduction_int2`, `native_int2_extension_cuda_wmma_tensor_core_int2`, ou le backend diagnostic `native_int2_rawkernel_cuda_*`, plus les champs `native_backend`, `autotuned`, `autotune_cache_hit` et `autotune_candidate_ms`.
- `TransformerConfig.native_ternary_autotune_cache_path` peut brancher un profil JSON d'autotune dans les `BitLinear` du vrai training.
- backward fast STE CUDA qui calcule `grad_input` directement depuis les codes int2 packes, les scales et le residual optionnel.
- backward extension tuilé qui calcule `grad_weight` + `grad_bias` avec accumulation fp32 et sortie au dtype du parametre.
- requantization/packing CUDA fusionnee apres optimizer step, P7 ou P10 quand la version du poids change.

### Interaction Avec Les Autres Phases

P2 fournit les traces de compression a :

- P6, pour attribuer des echecs a la compression ;
- P7/P10, car les patchs requantifient ensuite le coeur ternaire et regenerent les buffers packes ;
- BitLedger, pour le cout effectif.

### Impact Apprentissage

Le forward lit la valeur runtime depuis les codes ternaires packes, puis utilise une estimation straight-through pour garder un gradient vers `float_weight`. Sur CUDA fp16/bf16 et grandes formes, le forward peut passer par WMMA en decodant le poids int2 directement en shared memory `KxN`; sur fp32 ou petites formes, il garde les kernels tuiles/warp hand-written. Sur CUDA fp32/fp16/bf16, le fast STE calcule aussi `grad_input` depuis les codes int2 packes au lieu de reconstruire un poids dense pour cette partie. Sur fp16 et bf16 alignes ou non multiples de 16, `grad_input` passe par WMMA avec decode int2 en shared memory et zero-padding interne, et `grad_weight` passe par WMMA fp16/bf16->fp32 aligne ou padde; seules les tres petites formes des micro-phases gardent les kernels hand-written warp/tiled. Le modele apprend donc avec une valeur avant ternaire packee et un backward ternaire packe observe, pas seulement avec une couche float habillee par des logs.

### Preuve Runtime

Le checkpoint inspecte montre `P2=238652` evenements. Le smoke court extension `tools\train_llm.py smoke --device cuda --require-cuda --steps 2` utilise l'extension par defaut, resout `precision=auto` en `fp16`, et montre aussi `native_ternary_backend_counts={'extension': 2185}`, `native_ternary_requantize_backend_counts={'extension': 230}`, `native_ternary_grad_weight_backend_counts={'extension': 160}`, `native_ternary_grad_input_kernel_counts={'warp': 152, 'wmma_fp16': 8}`, `native_ternary_grad_weight_kernel_counts={'tiled': 152, 'wmma_fp16_float': 8}`, `torch_packed_ternary_dispatches=0`, `strict_extension_only=true` et audits P2/architecture passants. Le smoke strict `tools\train_llm.py smoke --out-dir runs\llm-smoke-bf16-forward-wmma-v7 --device cuda --require-cuda --precision bf16 --steps 2` passe aussi avec `native_ternary_backend_counts={'extension': 2191}`, `native_ternary_kernel_variants=['tiled_shared_memory_int2','warp_reduction_int2','wmma_tensor_core_int2']`, `native_ternary_requantize_backend_counts={'extension': 230}`, `native_ternary_grad_weight_backend_counts={'extension': 162}`, `native_ternary_grad_input_kernel_counts={'warp': 154, 'wmma_bf16': 8}`, `native_ternary_grad_weight_kernel_counts={'tiled': 154, 'wmma_bf16_float': 8}` et audits architecture/deliverable passants. Les tests ajoutes verifient en plus que `BitLinear` execute un dispatch CUDA natif tuilé, warp-reduction ou WMMA sur GPU local, que les valeurs fp32/fp16/bf16 correspondent au runtime packe, que l'auto-selection choisit la variante attendue selon la forme, que le forward WMMA matche le packed runtime dense, que le backward fast STE garde la meme semantique que le dense STE pour `grad_input`, `grad_weight` et `grad_bias`, que les chemins alignes/edge fp16/bf16 utilisent WMMA, et que le gradient STE reste non nul vers les poids entrainables.

La matrice soutenue courte `tools\benchmark_ternary_kernel.py --matrix --dtype fp16 --kernel-variant auto --warmup 2 --repeat 6 --sustain-seconds 0.35 --sustain-op forward_backward --min-resource-samples 2` couvre `64x128x128`, `128x256x256` et `256x512x512`: `strict_extension_only=true`, `resource_samples_passed=true`, sample min `4`, speedup forward+backward min `1.02x`, moyen `1.41x`, GPU moyen `21.83%`, puissance GPU moyenne `40.21 W`, CPU process moyen `25.28%`. Ces compteurs prouvent que le monitoring est branche sur une fenetre courte mais soutenue; ils montrent aussi que les petites shapes ne saturent pas encore le GPU.

La matrice LLM-shape courte alignee `256x768x768` et `512x1024x1024` avec extension v5 corrige le goulet observe sur `grad_input`: `strict_extension_only=true`, `resource_samples_passed=true`, `gradInputCounts={"wmma_fp16":471/213}`, `gradWeightCounts={"wmma_fp16_float":471/213}`, speedup forward+backward moyen `1.83x`, min `1.75x`, GPU moyen `42.58%`. La matrice edge non multiple de 16 `255x769x771` et `511x1025x1027` passe aussi avec `gradInputCounts={"wmma_fp16_padded":328/138}`, `gradWeightCounts={"wmma_fp16_float_padded":328/138}`, speedup moyen `1.85x`, min `1.26x`, GPU moyen `33.5%`. La matrice BF16 extension v6 `256x768x768` et `255x769x771` passe avec `gradInputCounts={"wmma_bf16":300}` puis `{"wmma_bf16_padded":282}`, `gradWeightCounts={"wmma_bf16_float":300}` puis `{"wmma_bf16_float_padded":282}`, speedup moyen `1.39x`, min `1.20x`, GPU moyen `9.1%` et puissance moyenne `36.07 W`. La matrice forward-WMMA v7 corrige le goulet forward: fp16 `512x1024x1024` passe de `native_vs_unpack=0.32x` avant v7 a `1.21x`, et bf16 edge `255x769x771` atteint `native_vs_unpack=3.38x`, avec `strict_extension_only=true`. Le smoke 2 steps reste un test d'integration architecture; son proof comparatif global peut rester `false` si la baseline next-token a un score nul, car le gate refuse volontairement une victoire artificielle par division par quasi-zero.

La boucle post-update est aussi native sur CUDA: apres un changement de poids, `_sync_quantized_buffers_from_weight` peut regenerer `signs`, `mask`, `scales`, `residual_weight` et `packed_codes` via un kernel fusionne. Le benchmark court RTX 5070 `128x256x256 fp16` mesure `0.2245 ms` pour le chemin fusionne contre `0.5901 ms` pour le chemin PyTorch tensoriel.

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
- cree des segments memoire avec decision de retention apprise quand `LearnedMemoryPolicy` est disponible ;
- extrait les anchors ;
- applique la decision exact/latent/drop a la memoire P4 partagee ;
- refuse le drop reel d'un segment porteur d'ancres en le promouvant en exact via l'Exact Anchor Ledger ;
- reconstruit via memoire recent exact + latent old KV ;
- verifie la fidelite des anchors ;
- enregistre un `MemoryUtilityCredit` quand une reconstruction utilise un segment retenu ;
- transforme les credits appris en prior exact/latent/drop pour le prochain forward ;
- observe la politique `LearnedMemoryPolicy` du forward ;
- compte les decisions/probabilites exact/latent/drop demandees et appliquees ;
- ajoute une supervision d'ancre vers la memoire exacte quand des ancres sont detectees ;
- enregistre erreurs si une ancre disparait.

### Interaction Avec Les Autres Phases

P4 alimente :

- P8, qui peut utiliser la memoire pendant inference ;
- P6, si une regression vient de perte d'ancre ;
- `L_anchor_fidelity` ;
- le loss `learned_memory`, qui apprend la retention exacte/latente/drop a partir de l'utilite token-level ;
- le prior d'utilite memoire qui apprend depuis les reconstructions aval reussies ;
- `CausalLedger`.

### Impact Apprentissage

La memoire n'est pas un stockage passif. `LearnedMemoryPolicy` modifie le hidden state avec un melange differentiable entre exact, latent local et drop vector, puis `CortexObjective` supervise cette politique avec les pertes token-level : les tokens difficiles tendent vers exact, les tokens intermediaires vers latent et les tokens faciles/corrects vers drop. Le controleur convertit cette politique en `MemoryRetentionDecision` pour la memoire partagee : exact reste en recent KV, latent va directement en KV latent compresse, drop oublie reellement un segment non-ancre, et les ancres critiques bloquent le drop par override.

Depuis C76, les reconstructions fideles creent aussi des credits d'utilite lies aux decisions appliquees. Les credits issus de `learned_memory_policy` mettent a jour un prior exact/latent/drop persistant; ce prior biaise les logits de `LearnedMemoryPolicy` et ajoute une cible d'alignement dans `CortexObjective`. Les exemples d'ancrage restent du replay et les echecs de fidelite peuvent faire echouer l'audit, mais la partie apprise regarde maintenant ce qui a vraiment ete reutilise par P4/P8/Frontier.

Une ablation courte reproductible existe avec `tools/benchmark_learned_memory_policy.py` : elle charge les memes poids partages dans un modele avec memoire apprise et un modele sans memoire apprise, fige tout sauf `learned_memory.*`, puis mesure gradient, decisions exact/latent/drop, ratio de stockage et delta de loss. Cette preuve montre que la politique peut modifier l'apprentissage sur un batch controle ; elle ne remplace pas encore une preuve long-contexte held-out.

## Phase 5 - Latent Reasoning Workspace / Certificates

### Entree

P5 part d'une tache issue du cycle ou d'une tache controlee.

### Travail Execute

La phase cree :

- un workspace latent explicite depuis le vrai forward Transformer ;
- un `LatentProofState` ;
- un certificat issu de la vraie `CertificateHead` du Transformer quand un forward LLM est disponible ;
- un certificat court ;
- une verification par outil ;
- une verification algebrique lineaire multi-step ;
- une verification symbolique SymPy de racines quadratiques exactes ;
- une verification code visible/cachee/proprietes ;
- une random de-latentization probe ;
- une mesure d'efficacite certificat vs raisonnement visible.

### Interaction Avec Les Autres Phases

P5 alimente :

- `BitLedger` via cout de certificat ;
- `CausalLedger` via preuve/certificat ;
- replay P5 lineaire et symbolique verifie ;
- `L_latent_certificate`.

### Impact Apprentissage

Le modele possede un `LatentReasoningWorkspace` et une `CertificateHead` dans le forward. Le workspace execute plusieurs pas latents, renvoie un feedback dans les hidden states avant logits/MTP/certificat, puis `latent_workspace_loss` lie son resume a l'etat latent de certificat. Le loss de certificat pousse la tete a produire une reponse finale et une incertitude coherente. P5 materialise maintenant cette sortie en `ShortCertificate` verifie par checksum latent, coherence token (`model_token_certificate`) et claims de binding workspace, puis persiste l'artefact dans le rapport et les checkpoints. Les certificats algebre/code sont des preuves outil plus fortes: lineaire multi-step, solveur symbolique SymPy pour quadratiques exactes, tests code visibles/caches/proprietes. Le certificat symbolique cree aussi un replay P5 verifie, donc le solveur specialise agit sur le corpus de replay au lieu de rester un audit separe.

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
- top cause ;
- signaux de politique apprise quand l'historique P7 existe.

### Interaction Avec Les Autres Phases

P6 est le pont entre detection et correction :

- P1 detecte ;
- P6 explique ;
- P7 repare ;
- P7 renvoie a P6 le resultat verifie de la reparation ;
- P10 peut proposer une amelioration plus generale.

### Impact Apprentissage

P6 evite de transformer toute regression en retraining global aveugle. Il donne une cible de correction, ce qui rend P7 possible et mesure le cout relatif d'une reparation locale. Le controleur LLM conserve maintenant une memoire de politique d'attribution : quand P7 applique une reparation non-regressive et ameliore la loss de reparation, le couple skill/cause gagne un signal de succes, de gain par cout et d'intervention dominante. Les attributions suivantes peuvent donc privilegier les causes qui ont deja produit des reparations verifiees, au lieu de rester sur un classement purement statique.

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
11. rollback si le gate echoue ;
12. renvoie le resultat accepte a la politique P6.

### Interaction Avec Les Autres Phases

P7 depend de P1/P6 et agit sur P2 :

- P1 donne la regression ;
- P6 donne la cause ;
- P7 choisit la correction ;
- P6 apprend du resultat P7 ;
- P2 est requantifie apres patch ;
- P9 peut ensuite consolider des exemples lies.

### Impact Apprentissage

P7 agit par deux chemins :

- replay causal ;
- modification directe des poids.

La modification directe est importante : elle fait de P7 un vrai mecanisme de regrowth, pas seulement une generation d'exemples.

### Preuve Runtime

Checkpoint `step 175` :

- applications P7 : `1`
- delta poids L1 : `880.410583`
- repair loss before : `29.159054`
- repair loss after : `27.919401`
- repair loss delta : `1.239653`
- protected loss before : `29.138744`
- protected loss after : `27.827297`
- protected loss delta : `-1.311447`
- protected loss tolerance : `0.582775`
- gate : `accepted`

## Phase 8 - Adaptive Inference

### Entree

P8 recoit une tache et force ou choisit un chemin :

- fast ;
- normal ;
- careful.

Dans le harness LLM, la source de reponse par defaut est maintenant le vrai `CortexTransformerLM` via `CortexTransformerInferenceAgent`. Les circuits Frontier restent prioritaires uniquement lorsqu'un circuit compile couvre la tache et passe le binding memoire P4.

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
- generation adaptative depuis le `lm_head` et les tetes MTP du Transformer quand aucun FastSolve Frontier couvert ne fournit la reponse : le premier token vient du next-token head, le second token de bloc vient du horizon-2 MTP head, puis `FutureContractEngine` accepte ou rejette le bloc.

### Interaction Avec Les Autres Phases

P8 utilise :

- P4 memoire ;
- P5 certificats ;
- P3 future contracts ;
- P2 kernel ternaire ;
- verifier P1 pour audit final.
- le `CortexTransformerLM` lui-meme pour la reponse model-backed et les metadonnees de `CertificateHead`.

### Impact Apprentissage

P8 fournit des exemples verifies par route et mesure la capacite par cout effectif. Il aide le modele a apprendre une politique ou la qualite verifiee compte plus que le debit brut. Depuis C71, les rapports exigent aussi `inference_model_backed_events` et `inference_model_backed_generated_tokens`, ce qui prouve que P8 ne s'appuie plus seulement sur une reponse symbolique externe. Depuis C72, ces generations model-backed deviennent aussi du replay P8 verifie ou correctif (`inference_model_backed_replay_events`), donc elles influencent directement `replay_loss`. Depuis C75, le chemin model-backed n'est plus un greedy loop isole : il produit des propositions de blocs MTP/FSP, les gate via le meme `FutureContractEngine` que P3, compte propositions/checks/acceptations/rejets dans `inference_model_backed_adaptive_mtp_*`, et l'audit architecture echoue si cette jonction P3/P8 n'a pas tourne.

## Phase 9 - Sleep / Consolidation

### Entree

P9 recoit le cycle P1, donc les echecs, reussites, fragilites et competences. Depuis C70, il recoit aussi les spans reels du corpus observes par `observe_input_batch`, decodes avec le tokenizer actif et verifies comme exemples exacts `REAL_EXOGENOUS`.

### Travail Execute

P9 construit :

- replay de failures ;
- exemples tool-solved ;
- variantes metamorphiques ;
- reservoir reel/exogene issu des vrais batchs LLM, avec metadata `from_llm_input_batch` et hash de span ;
- filtre anti-collapse ;
- schedule de consolidation ;
- promotion des familles coherentes acceptees en circuits Frontier `sleep_consolidation` ;
- verification held-out separee ;
- FastSolve immediat via `CompiledFrontierAgent` ;
- proposition P10 `compiled_frontier` issue du sommeil.

Le filtre rejette les exemples qui menacent :

- calibration ;
- diversite ;
- contamination ;
- duplication ;
- collapse de competences rares.

### Interaction Avec Les Autres Phases

P9 consolide les signaux de P1/P6/P7/P8/P10 et les spans reels du dataloader en replay entrainable, puis compile certaines familles acceptees en circuits persistants utilisables par le registre Frontier.

### Impact Apprentissage

P9 est une memoire d'entrainement verifiee et une source de compilation. Elle transforme les corrections lentes, les exemples tool/metamorphiques et les spans reels verifies du corpus en exemples causalement entrainables, puis transforme une famille coherente en competence FastSolve held-out gated. Dans le checkpoint, P9 conserve ses replays, ses pools sleep, les exemples reservoir `REAL_EXOGENOUS` marques `from_llm_input_batch`, ses rapports `sleep_frontier_reports`, les compteurs `sleep_frontier_*` et les circuits dans `frontier_registry`.

## Phase 10 - Recursive Improvement

### Entree

P10 part du `CycleReport`, des actions et regressions.

### Travail Execute

P10 :

1. genere des propositions initiales ;
2. teste chaque generation en sandbox ;
3. evalue qualite, cout, robustesse ;
4. verifie protected skills ;
5. detecte calibration regression ;
6. detecte reward hacking ;
7. verifie diversite/collapse ;
8. accepte ou rejette ;
9. archive la decision ;
10. cree des propositions descendantes depuis les parents acceptes de l'archive, avec filiation et pression de diversite ;
11. persiste l'archive evolutive complete dans `archive.json` et les evenements rollback dans `rollback.json` ;
12. recharge ces archives via `cortex_improvement_archive_dir` quand un run independant partage le meme dossier, mais refuse les records sans rapports verifier complets ou sans rollback persistant ;
13. cree rollback token ;
14. applique une proposition acceptee comme patch signe sur vrais poids Transformer ;
15. ecrit un artefact rollback executable `model_patch_rollbacks/<signed_patch_id>.pt` contenant les tensors pre-patch et les checksums pre/post ;
16. mesure repair loss, protected loss et delta de poids.

### Interaction Avec Les Autres Phases

P10 est la boucle d'amelioration recursive au-dessus des autres phases :

- P1 detecte ;
- P6 explique ;
- P7 repare localement ;
- P10 propose une amelioration plus generale ;
- P2 est requantifie apres patch ;
- P9 peut consolider l'effet.
- les runs suivants peuvent repartir d'une archive P10 deja peuplee sans reprendre le checkpoint, ce qui garde la pression de diversite et l'historique rollback actifs entre runs independants.

### Impact Apprentissage

P10 agit par :

- replay P10 ;
- patch direct des poids ;
- rollback executable des poids si un patch signe doit etre retire : verification du checksum post-patch, restauration des tensors pre-patch, puis requantification du coeur ternaire ;
- archive evolutive durable, strictement restauree sans `fallback_score`, qui influence les gates de diversite des runs suivants ;
- evolution multi-generation bornee depuis des parents acceptes, avec lineage et pression vers les types de propositions sous-representes ;
- signaux vers `L_recursive_improvement_validity`.

### Preuve Runtime

Checkpoint `step 175` :

- applications P10 : `1`
- delta poids L1 : `523.004089`
- repair loss before : `27.872873`
- repair loss after : `27.816763`
- repair loss delta : `0.056110`
- protected loss before : `43.066746`
- protected loss after : `43.136181`
- protected loss delta : `0.069435`
- protected loss tolerance : `0.861335`
- signed patch id : `c5c1e64a504bc61f407d380b0664978382fe1225c0c40a967a8f48957bfb0139`
- rollback token : `rollback-proposal-0-mtp_head-arithmetic`
- rollback executable : `model_patch_rollbacks/<signed_patch_id>.pt`, checksum d'artefact et checksums de parametres pre/post exiges par l'audit P10 depuis C77
- gate : `accepted`

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

## Frontieres Restantes Vers La Vision Maximale

L'avis externe qui dit "ce n'est pas encore la preuve maximale" est correct. Il ne contredit pas les preuves d'integration ; il se place au niveau plus exigeant de la demonstration finale du paradigme.

### Memoire Cognitive Apprise

Etat actuel :

- `VariableInCompressor` est dans le forward PyTorch et peut apprendre une compression differentiable des representations ;
- `LearnedMemoryPolicy` est maintenant dans le forward, produit des logits exact/latent/drop, modifie les representations avant les blocs Transformer et recoit un loss trainable ;
- `tools/benchmark_learned_memory_policy.py` fournit une ablation courte a poids partages et mesure `disabled`, `before`, `after`, decisions exact/latent/drop, gradient memoire et delta `before - after` ;
- P4 observe de vrais batchs, extrait des ancres, reconstruit via memoire exacte/latente, verifie la fidelite, supervise les ancres et genere du replay.

Limite restante :

- la politique exact/latent/drop est apprise, branchee et ablatee en court, mais son gain final sur long contexte massif, cout memoire et held-out anchors doit encore etre mesure sur run long.

Critere de fermeture :

- finir un run long avec `learned_memory_policy_events`, decisions exact/latent/drop, `learned_memory` loss et zero fidelity failure dans les sidecars ;
- mesurer son gain sur long contexte, anchors held-out, cout memoire, loss et qualite de generation face a une ablation sans politique apprise.

### Ternary Hardware-Native

Etat actuel :

- `BitLinear` tourne dans le forward avec valeur runtime issue de buffers `packed_codes` int2 ;
- sur CUDA, le forward peut lancer les kernels natifs CuPy RawKernel `tiled_shared_memory_int2` ou `warp_reduction_int2` via DLPack zero-copy ;
- en mode `auto`, le forward mesure les deux variants sur la shape courante, cache le meilleur choix par device/dtype/shape, peut persister/recharger le profil et trace les temps candidats ;
- le chemin packe utilise un autograd custom : le forward retourne la valeur native/packee sans `F.linear` STE dense, et le backward conserve les gradients STE vers input, poids et bias ;
- les buffers packes sont resynchronises par version de poids, donc pas repackes a chaque forward inutilement, puis explicitement requantifies apres optimizer step et apres patchs P7/P10 ;
- les traces P2 prouvent une execution ternaire-compatible pendant le run ;
- un smoke test CUDA verifie les backends natifs sur GPU local avec gradient STE non nul ;
- `tools/train_llm.py profile-batch` lance un vrai batch training Cortex strict avec optimizer, backward, requantize, P1-P10, monitoring CPU/GPU/power/VRAM et snapshot memoire CUDA torch ;
- `tools/train_llm.py profile-matrix` repete ce profil sur plusieurs shapes et seeds, ecrit JSON/CSV agreges, et rend bloquants `min_cases`, `require_multi_shape`, `require_multi_seed`, extension-only, all-phases-active et les seuils optionnels de throughput/GPU/VRAM/puissance ;
- `tools/train_llm.py profile-autosize` estime les candidats avec la config Cortex complete, selectionne les shapes et `gradient_accumulation_steps` qui rentrent dans un budget memoire/VRAM, mesure par defaut une frontiere diversifiee avec `--measure-candidate-strategy diverse` au lieu du seul top-N estime, raffine les vagues suivantes avec `--measure-candidate-adaptive-rounds` autour des meilleurs resultats observes sans depasser `--measure-candidate-count`, repete ces mesures sur toutes les seeds fournies et au moins deux seeds candidates par defaut en synthetisant des seeds deterministes si necessaire, refuse les candidats dont la VRAM observee max depasse le budget quand le signal GPU existe, explore les rounds adaptatifs avec une borne haute `mean + stddev`, ajoute ensuite une seed de raffinement aux candidats incertains du front de decision avec une fenetre `steps` plus longue et des repetitions runtime, moyenne ces repetitions par seed/steps avant scoring, confirme les finalistes provisoires avec seeds fraiches et plan runtime adaptatif qui augmente steps/repeats quand le signal d'incertitude est haut, confirme aussi les challengers non confirmes dont la borne haute chevauche encore le gagnant robuste avec un budget de rounds par defaut dimensionne sur tout le front mesure, utilise ensuite le budget dedie `confirm_selected_decision_resolution_extra_rounds` plus l'extension `confirm_selected_decision_resolution_adaptive_extra_rounds` re-evaluee apres chaque round depuis l'overlap des intervalles confirmes en `decision_margin_resolution` si le front confirme reste ambigu, exige `confirmation_decision_resolved=true`, selectionne alors uniquement parmi les candidats passants sur tous les profils mesures/raffines/confirmes avec un score robuste `mean - stddev` et les champs d'audit mean/min/max/stddev/lower-confidence/upper-confidence/source/refinement/confirmation/frontier/measurement_steps/repeats/observations, puis lance `profile-matrix` sur les shapes choisies ;
- le budget de raffinement autosize est maintenant choisi par `expected_gain_per_cost`: chaque action candidate encode le gain attendu, la largeur d'incertitude, l'utilite posterior, le cout en tokens mesures, le gain par cout, le statut finaliste, le flag `selected_for_refinement`, la `report_selection_reason` et les steps/repeats/seeds planifies dans le front d'audit representatif `refinement_budget_candidate_actions`, plafonne par `refinement_budget_candidate_action_report_cap` avec total-count et flag de troncature, tandis que `refinement_budget_actions` garde les actions effectivement profilees ;
- `tools/benchmark_ternary_kernel.py` fournit un benchmark reproductible du kernel natif contre unpack+`F.linear`.

Limite restante :

- les kernels natifs actuels couvrent deja une variante tuilée shared-memory, une variante warp-reduction, un forward WMMA fp16/bf16 decode-shared, un backend extension C++/CUDA strict, un autotune mesure/cache par shape, un profil JSON persistant, WMMA fp16/bf16 pour `grad_input` aligne/padde et WMMA fp16/bf16->fp32 pour `grad_weight` aligne/padde, le profil court `runs/llm-batch-profile-v1/llm_batch_profile.json` mesure deja un vrai batch bf16 CUDA avec `passed=true`, la matrice courte `runs/llm-batch-profile-matrix-v2/llm_batch_profile_matrix.json` mesure 2 shapes x 2 seeds avec seuils ressource passants, `runs/llm-profile-autosize-v1/llm_batch_profile_autosize.json` selectionne automatiquement `seq48_d96_h4_l2_b8` + `seq32_d96_h4_l2_b8` sous budget VRAM avant de verifier 2/2 cas stricts, `runs/llm-profile-autosize-measured-v1/llm_batch_profile_autosize.json` mesure 4 candidats, selectionne `seq32_d96_h4_l2_b8` + `seq48_d96_h4_l2_b4` par score observe, puis verifie 2/2 cas stricts, `runs/llm-profile-autosize-multiseed-v1/llm_batch_profile_autosize.json` mesure le candidat selectionne sur seeds `71,73` avant une matrice 2/2 strictement passante, `runs/llm-profile-autosize-diverse-v1/llm_batch_profile_autosize.json` mesure les rangs estimes `1,8` avant de choisir le rang 8 par score observe, `runs/llm-profile-autosize-adaptive-v1/llm_batch_profile_autosize.json` mesure deux rounds `[1,8]` puis `[2,5]` avant de choisir un candidat adaptatif, et le chemin courant ajoute des repetitions `refine_seed_<seed>_steps_<n>_repeat_<i>` sur les candidats incertains du front de decision choisis par `expected_gain_per_cost` puis `confirm_seed_<seed>_steps_<n>_repeat_<i>` sur les finalistes provisoires, les challengers et la resolution `decision_margin_resolution`, avec `refinement_budget_actions`, `refinement_budget_candidate_actions`, `confirm_selected_max_rounds` dimensionne par defaut sur le nombre de candidats mesures, `confirm_selected_decision_resolution_extra_rounds` dedie a la marge, `confirm_selected_decision_resolution_adaptive_extra_rounds` qui ajoute des rounds selon `confirmation_decision_resolution_overlap_ratio` re-evalue, `confirm_selected_runtime_*_cap` qui borne l'escalade de steps/repeats, `confirmation_runtime_escalations` et `confirmation_decision_resolution_budget_evaluations` qui tracent chaque decision runtime/base/adaptive, et `selected_confirmation_decision_unresolved` bloquant si la marge confirmee reste non positive ;
- les benchmarks doivent etre elargis a plusieurs tailles LLM reelles, seeds plus nombreuses, durees plus longues et qualite de convergence.

Critere de fermeture :

- profiler VRAM, energie, throughput et remplissage GPU sur shapes LLM plus grandes et plusieurs seeds sans retirer de composants ;
- benchmarker latence, VRAM, energie estimee, throughput et qualite face a une baseline dense sur runs LLM larges.

### Verifier Dynamique A Grande Echelle

Etat actuel :

- P1 couvre arithmetic, algebra, long context anchors, entity tracking, instruction following, code unit tests et calibration ;
- ces familles alimentent attribution, regrowth, sleep et recursive improvement.

Limite actuelle :

- ce n'est pas encore un banc large avec domaines nombreux, oracles varies, held-out strict et adversarial sets massifs.

Critere de fermeture :

- elargir les generateurs et oracles ;
- separer train/validation/held-out verifier tasks ;
- publier les taux accept/reject, faux positifs, faux negatifs et regressions attribuees.

### SlowSolve -> Compile -> FastSolve

Etat actuel :

- P1/P6/P7/P9/P10 creent detection lente, attribution, reparation, replay, consolidation, compilation sleep-frontier et patchs ;
- Frontier Skill Discovery compile les regressions sources slow-solvees plus leurs variantes en micro-circuits `BitLinear` persistants ;
- P9 utilise maintenant le meme mecanisme sur les familles sleep acceptees, avec `source_kind="sleep_consolidation"`, held-out gate, FastSolve immediat et proposition P10 ;
- avant promotion, ces circuits passent maintenant un gate held-out court ; si ce gate echoue, le held-out verifie devient support metamorphique pour une recompilation bornee puis une nouvelle suite held-out est generee ;
- P5 certifie l'origine compilee avec un contrat `compiled_circuit` verifie par checksum, lineage source/frontier/held-out, DSV, gate held-out et verification runtime ;
- P8 consomme le registre compile pour repondre en FastSolve sur les taches couvertes par ids, signatures numeriques, metadonnees non-label ou ancres, pas seulement par nom de skill ;
- P7 evalue le meme circuit comme candidat de reparation avant regrowth parametrique ;
- P10 promeut les reparations compilees acceptees et les circuits sleep-frontier en propositions `compiled_frontier` prioritaires avant son patch modele signe.

Limite actuelle :

- la preuve finale manque encore a grande echelle : une competence resolue lentement et verifiee passe maintenant un held-out court bloquant, mais doit encore devenir rapide, moins chere, stable et generalisable a des taches nouvelles sur suites held-out larges.

Critere de fermeture :

- suivre une famille de competences depuis slow solve jusqu'a fast solve ;
- mesurer cout avant/apres, pass rate held-out, generalisation, non-regression et retention apres consolidation.

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

- `test_symbolic_algebra_certificate_uses_sympy_solver_for_quadratic_roots`
  - exige le routage `sympy_symbolic` ;
  - verifie racines exactes, claim de solveur et rejets de racines/claims faux.

- `test_algebra_oracle_accepts_exact_symbolic_quadratic_root_sets`
  - prouve que le Verifier OS accepte le replay algebrique symbolique ;
  - rejette les racines fausses et les reponses avec texte extra.

- `test_persistent_archive_rejects_missing_full_evaluation_reports`
  - prouve que P10 refuse une archive tronquee au lieu de reconstruire un score de secours.

- `test_persistent_archive_rejects_missing_rollback_file_for_accepted_records`
  - prouve qu'une archive acceptee ne peut pas etre restauree sans `rollback.json`.

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
