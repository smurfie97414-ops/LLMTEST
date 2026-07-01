# Cortex-3 / LLMTEST

**Cortex-3** est une base de recherche pour un paradigme appelé **Recursive Verified Cognitive Compilation**.

L'idée n'est pas de faire une simple quantization ou un prototype jouet. Le but est de construire un système où un modèle apprend en mode plastique, résout lentement quand c'est nécessaire, vérifie ses compétences, puis compile ce qu'il a appris en structures plus rapides, compressées, réparables et non-régressives.

> Hypothèse centrale : l'intelligence utile n'est pas la précision individuelle des poids ; c'est la structure minimale qui conserve des compétences sous vérification.

## Vision

Le paradigme classique ressemble à :

```text
beaucoup de données
+ beaucoup de paramètres continus
+ next-token prediction
+ compression après coup
```

Cortex-3 vise plutôt :

```text
résoudre lentement
→ vérifier fortement
→ extraire l'invariant
→ compiler en circuit discret
→ mesurer le coût réel
→ réparer les pertes
→ re-tester
→ réutiliser plus vite
```

Le modèle final n'est donc pas seulement un LLM compressé. C'est un **compilateur cognitif vérifié** : un système qui transforme des compétences lentes en compétences rapides sans perdre les capacités rares.

## Pourquoi commencer par le vérificateur ?

Le prochain goulot n'est pas seulement BitNet, MTP, MoE ou le KV-cache. Le vrai goulot est :

> Comment prouver qu'une compression n'a pas détruit une compétence rare ?

Un modèle peut garder une bonne fluidité et une bonne perplexité tout en perdant silencieusement :

- le raisonnement long ;
- la précision mathématique ;
- le suivi exact de variables ;
- la récupération d'ancres exactes dans un long contexte ;
- la calibration ;
- la capacité à dire « je ne sais pas » ;
- la robustesse aux reformulations ;
- les cas limites en code et en logique.

C'est pourquoi ce repo implémente d'abord un **Dynamic Skill Verifier**.

## Architecture cible

```text
Input
  │
  ▼
Variable-In Compressor ──► Exact Anchor Ledger
  │                              │
  ▼                              ▼
Latent Memory / KV        Causal + Skill Ledgers
  │                              │
  ▼                              ▼
Ternary Core  W ∈ {-1,0,+1}  +  Skill-aware Experts
  │
  ├── Future Contract / FSP + Output-Goal Contracts
  ├── Adaptive Multi-Token Decoding
  ├── Latent Reasoning Workspace
  └── Certificate Generator
          │
          ▼
Hierarchical Dynamic Verifier
          │
          ├── accept
          ├── reject
          ├── attribute regression
          ├── minimal regrowth
          └── sleep / consolidation buffer
```

## Implémentation actuelle

Cette base contient maintenant :

- `cortex3.py` : noyau de tâches, skills, vérificateur dynamique, adversarial checks, ternaire sign+mask, ancrage exact, horizon MTP adaptatif, regrowth minimal et CLI de démonstration ;
- Phase 1 du Verifier OS est maintenant élargie avec registre d'oracles, anti-métamorphiques, coûts vérificateur par cas, audit faux positifs/faux négatifs d'oracle, harnais de défauts injectés et familles de compétences `arithmetic`, `algebra`, `long_context_anchor`, `entity_tracking`, `instruction_following`, `code_unit_tests`, `calibration` ;
- `cortex3_reporting.py` : persistance des cycles dans `runs/` avec JSON structuré, rapport markdown et matrice de défauts injectés ;
- `cortex3_ternary.py` : instrumentation Phase 2 avec quantization d'activations 8→4 bit, residual synapse buffer, compression logs, `BitLinear` sign+mask, buffers ternaires packes int2 et kernels CUDA natifs extension/RawKernel tuilé/warp/WMMA avec gradient STE ;
- `cortex3_future.py` : Phase 3 MTP/FSP sous contrat avec têtes PyTorch horizons 1/2/4/8, calibration autonome, confidence head, temporal consistency loss, Future Contract, contrats output-goal au-delà des token ids, révision et accept/reject gates ;
- `cortex3_memory.py` : Phase 4 mémoire cognitive avec KV récent exact, KV ancien latent compact, Exact Anchor Ledger, reconstruction conditionnée par requête, récupération de réponse augmentée par mémoire et vérificateur de fidélité aux ancres ;
- `cortex3_certificates.py` : Phase 5 raisonnement latent avec `latent proof state`, tête PyTorch de certificat calibrée, génération proof-carrying, certificats courts vérifiables dont contrats `compiled_circuit`, certificats algèbre multi-step, code visible+hidden/propriétés, dé-latentisation aléatoire et vérification par outils ;
- `cortex3_attribution.py` : Phase 6 attribution causale avec ablations par blocs, experts, KV mode, horizon MTP, précision d'activation, contrat FSP, routage counterfactual et clustering de régressions ;
- `cortex3_regrowth.py` : Phase 7 regrowth minimal exécutable avec action space de réparation, simulation gain/coût, gate de non-régression et annealing vers re-cristallisation ;
- `cortex3_inference.py` : Phase 8 inférence fast/normal/careful avec routeur de difficulté, prédicteur de budget, early exit, Mixture-of-Depths `BitLinear`, KV latent, self-speculative MTP, certificats et dispatch kernel ternaire ;
- `cortex3_sleep.py` : Phase 9 sleep anti-collapse avec replay d'échecs, données synthétiques vérifiées et labellisées, réservoir réel/exogène, familles métamorphiques, filtre anti-collapse et scheduler de consolidation ;
- `cortex3_improvement.py` : Phase 10 Recursive Improvement Engine avec génération de propositions, propositions prioritaires issues des réparations Frontier compilées, sandbox en mémoire, évaluateur dynamique, gate Pareto/protection/diversité, archive évolutive et rollback ;
- `cortex3_objective.py` : loss finale du plan avec 17 termes pondérés et les 15 métriques absolues, dont `Verified Capability per Effective Joule` ;
- `cortex3_experiments.py` : expériences A-E du plan, de la détection de défauts injectés à la sandbox d'auto-amélioration ;
- `cortex3_frontier.py` : découverte de compétences frontières avec slow-solve vérifié des régressions sources et variantes, compilation en micro-circuit `BitLinear`, recompilation sur support métamorphique jusqu'au gate held-out court, registre runtime de circuits DSV/held-out-passing, contrats output-goal P3, certificats P5 de contrat compilé incluant lineage held-out, sélection par couverture réelle via `CompiledFrontierAgent`, persistance `frontier_registry.json` + checkpoints, branchement P8 via `UltraFastInferenceEngine` et preuve d'usage FastSolve/réparation P7/proposition P10 compilée dans l'audit LLM ;
- `cortex3_microtrain.py` : micro-modèle PyTorch entraînable avec cœur `BitLinear`, agent DSV, exemples issus du verifier/sleep phase et checkpoints `.pt` ;
- `cortex3_autoregressive.py` : décodeur micro-autoregressif PyTorch avec vocabulaire caractère, génération greedy ou blockwise sous Future Contract, pertes comportement/MTP multi-horizons/contrat futur, agent DSV et checkpoints `.pt` ;
- `cortex3_llm.py` : harness de pré-entraînement LLM réel avec export Hugging Face `datasets`, tokenizer BPE `tokenizers`, corpus texte streamé vers memmap avec identité SHA-256, dataset causal, Transformer complet, baseline next-token, objectif Cortex multi-horizon, compresseur Variable-In différentiable, politique mémoire apprise exact/latent/drop, observation d'ancres exactes depuis les batchs LLM, coeur ternaire `BitLinear` packe int2 avec audit du kernel CUDA natif, MoE skill-aware entraînable, certificate head latent, ledgers Bit/Skill/Causal/Uncertainty persistants, AMP/DDP, checkpoints strictement liés au corpus, courbes et rapport comparatif ;
- `cortex3_phases.py` : registre exécutable des 10 phases Cortex-3 ;
- `cortex3_ledgers.py` : Bit Ledger, Skill Ledger, Causal Ledger et Uncertainty Ledger ;
- `cortex3_analysis.py` : analyse des causes probables d'une régression ;
- `cortex3_cycle.py` : cycle complet référence/trial → vérification → ledgers → analyse → actions budgetées → rapport ;
- `cortex3_selection.py` : sélection offline de trials et choix des compétences frontières ;
- `tools/run_cycle_report.py` : génération d'un rapport markdown du cycle ;
- `tools/train_llm.py` : CLI de préparation/entraînement/comparaison baseline-vs-Cortex pour corpus texte ;
- `tests/` : tests unitaires pour le noyau et les nouveaux modules ;
- `.github/workflows/ci.yml` : CI GitHub Actions.

Le fichier `Cortex-3 PLAN.txt` contient le plan complet de recherche et reste conservé dans le repo.

## Métrique centrale

Cortex-3 ne doit pas mesurer seulement les tokens/s ou les bits/poids. La métrique centrale est :

```text
Verified Capability per Effective Cost = qualité_vérifiée / coût_total
```

Avec :

```text
coût_total = bits de poids + activations + KV + tokens générés + étapes latentes + experts + vérification + regrowth
qualité_vérifiée = exactitude × robustesse métamorphique × calibration × fidélité aux ancres × absence de régression rare
```

## Installation

```bash
git clone https://github.com/smurfie97414-ops/LLMTEST.git
cd LLMTEST
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Les dépendances de travail incluent PyTorch, NumPy, Hugging Face `datasets`, `tokenizers` et Matplotlib ; elles sont obligatoires pour exécuter les couches modèle, exporter des corpus réels, entraîner le tokenizer BPE et générer les courbes d'apprentissage.

Pour remplacer un build PyTorch CPU par le build CUDA validé localement :

```bash
pip install -r requirements-cuda-cu128.txt
python tools/train_llm.py doctor --require-cuda --precision bf16 --device cuda
python tools/benchmark_ternary_kernel.py --dtype fp16
python tools/benchmark_learned_memory_policy.py --device cuda
```

`requirements-cuda-cu128.txt` installe aussi `cupy-cuda12x` et `ml_dtypes`, nécessaires au backend diagnostic RawKernel. Le backend training CUDA est maintenant strictement la vraie extension PyTorch C++/CUDA par défaut (`native_ternary_backend=extension`); `auto` et `rawkernel` restent des modes de diagnostic explicites, pas le chemin LLM strict. Sur ce PC, les libs CUDA 12.8 dev locales (`cuda-libraries-dev` et `cuda-cccl` dans `C:\Users\hight\.codex\cuda-12.8\Library`) fournissent `nvcc`, `cusparse.h` et `nv/target`. Avec le variant kernel `auto`, `BitLinear` mesure `tiled`, `warp` et `wmma` sur la shape CUDA courante quand le dtype/shape le justifie, cache le meilleur choix par device/dtype/shape, peut sauvegarder/recharger un profil JSON via `--autotune-cache`, saute le `F.linear` dense STE dans le forward grâce à un autograd custom, calcule le forward WMMA fp16/bf16 directement depuis les codes int2 packés pour les grandes formes, calcule `grad_input` CUDA depuis les poids int2 packés, utilise WMMA fp16/bf16 pour `grad_input` aligné ou paddé sur bords non multiples de 16, utilise WMMA fp16/bf16->fp32 pour `grad_weight` aligné ou paddé, garde les kernels hand-written warp/tiled pour les très petites formes, requantize/repack les poids en CUDA fusionné après update, et ne repack les poids que si leur version change. Les commandes d'entraînement CLI utilisent maintenant `--precision auto` par défaut: CUDA se résout en `fp16`, CPU en `fp32`. Sur RTX 5070, les benchmarks courts locaux donnent :

- `batch=128, in=256, out=256, fp16` : autotune `warp_reduction_int2`, candidats `tiled=0.1665 ms`, `warp=0.1368 ms`, runtime natif `0.1042 ms`, forward `BitLinear` complet `0.1323 ms` contre `0.3657 ms` pour l'ancien chemin `native + STE dense`, soit `2.76x`, erreur max `0.000976`.
- `batch=128, in=256, out=256, fp16`, mesure historique pré-WMMA : fast STE natif `0.6101 ms` contre dense STE legacy `0.9699 ms`, soit `1.59x`; `grad_input` lit les codes int2 packés et `grad_weight` + `grad_bias` passent par le kernel tuilé/fusionné extension.
- `batch=128, in=256, out=256, fp16`, requantize/pack post-update : kernel CUDA fusionné `0.2245 ms` contre chemin PyTorch `0.5901 ms`, soit `2.63x`.
- `batch=512, in=512, out=512, fp16` : autotune `warp_reduction_int2`, candidats `tiled=0.5668 ms`, `warp=0.3368 ms`, runtime natif `0.2561 ms` contre `0.2734 ms` pour unpack+`F.linear`, soit `1.07x`, erreur max `0.000976`.
- Matrice courte stricte `64x128x128`, `128x256x256`, `256x512x512`, fp16 : `strict_extension_only=true`, speedup forward+backward min `1.37x`, moyen `1.70x`, moyenne GPU `8.67%` et CPU process `26.67%` sur fenêtre sub-seconde `nvidia-smi`.
- Matrice soutenue courte, mêmes shapes, `--sustain-seconds 0.35 --min-resource-samples 2` : `strict_extension_only=true`, `resource_samples_passed=true`, speedup forward+backward min `1.02x`, moyen `1.41x`, GPU moyen `21.83%`, puissance GPU moyenne `40.21 W`, CPU process moyen `25.28%`.
- Matrice stricte LLM-shape alignée `256x768x768` + `512x1024x1024`, fp16, extension v5 : `strict_extension_only=true`, `resource_samples_passed=true`, `gradInputCounts={"wmma_fp16":471/213}`, `gradWeightCounts={"wmma_fp16_float":471/213}`, speedup forward+backward moyen `1.83x`, min `1.75x`, GPU moyen `42.58%`.
- Matrice stricte LLM-shape non multiple de 16 `255x769x771` + `511x1025x1027`, fp16, extension v5 : `strict_extension_only=true`, `resource_samples_passed=true`, `gradInputCounts={"wmma_fp16_padded":328/138}`, `gradWeightCounts={"wmma_fp16_float_padded":328/138}`, speedup forward+backward moyen `1.85x`, min `1.26x`, GPU moyen `33.5%`.
- Matrice stricte BF16 alignée + edge `256x768x768` + `255x769x771`, extension v6 : `strict_extension_only=true`, `resource_samples_passed=true`, `gradInputCounts={"wmma_bf16":300}` puis `{"wmma_bf16_padded":282}`, `gradWeightCounts={"wmma_bf16_float":300}` puis `{"wmma_bf16_float_padded":282}`, speedup forward+backward moyen `1.39x`, min `1.20x`, GPU moyen `9.1%`, puissance GPU moyenne `36.07 W`.
- Forward WMMA extension v7, fp16 `256x768x768` + `512x1024x1024` : variant forward `wmma_tensor_core_int2`, `native_ms=0.1235/0.3329`, `native_vs_unpack=2.91x/1.21x` alors que le profil pré-v7 sur `512x1024x1024` était à `0.32x`; matrice stricte `strict_extension_only=true`, `resource_samples_passed=true`, speedup forward+backward moyen `1.46x`, min `1.21x`.
- Forward WMMA extension v7, bf16 `256x768x768` + `255x769x771` : variant forward `wmma_tensor_core_int2`, `native_ms=0.0973/0.1369`, `native_vs_unpack=7.57x/3.38x`, speedup forward+backward moyen `1.65x`, min `1.42x`, `strict_extension_only=true`.

Le doctor distingue le backend RawKernel et le backend extension. Sur ce PC, `tools/train_llm.py doctor --require-cuda --require-cuda-extension --precision bf16 --device cuda` passe avec `native_rawkernel_available=true`, `native_extension_runtime_available=true`, `nvcc=12.8` depuis `C:\Users\hight\.codex\cuda-12.8\Library` et Visual Studio Build Tools 2022. Le smoke strict `tools/train_llm.py smoke --device cuda --require-cuda --steps 2` utilise l'extension par défaut, résout `precision=auto` en `fp16`, et produit un rapport avec `native_ternary_backend_counts={'extension': 2185}`, `native_ternary_requantize_backend_counts={'extension': 230}`, `native_ternary_grad_weight_backend_counts={'extension': 160}`, `native_ternary_grad_input_kernel_counts={'warp': 152, 'wmma_fp16': 8}`, `native_ternary_grad_weight_kernel_counts={'tiled': 152, 'wmma_fp16_float': 8}`, `torch_packed_ternary_dispatches=0`, `strict_extension_only=true`, et les audits P2/architecture passent avec exigence native explicite. Le smoke strict `tools/train_llm.py smoke --out-dir runs/llm-smoke-bf16-forward-wmma-v7 --device cuda --require-cuda --precision bf16 --steps 2` passe aussi avec `native_ternary_backend_counts={'extension': 2191}`, `native_ternary_kernel_variants=['tiled_shared_memory_int2','warp_reduction_int2','wmma_tensor_core_int2']`, `native_ternary_requantize_backend_counts={'extension': 230}`, `native_ternary_grad_weight_backend_counts={'extension': 162}`, `native_ternary_grad_input_kernel_counts={'warp': 154, 'wmma_bf16': 8}`, `native_ternary_grad_weight_kernel_counts={'tiled': 154, 'wmma_bf16_float': 8}` et audits architecture/deliverable passants. Ces smokes courts prouvent le branchement architecture; le proof gate comparatif reste volontairement `false` sur 2 steps si la baseline a un score nul, pour éviter une victoire artificielle.

Pour mesurer un vrai batch LLM Cortex strict avec throughput, VRAM, puissance et usage CPU/GPU, sans lancer un test long :

```bash
python tools/train_llm.py profile-batch --out-dir runs/llm-batch-profile-v1 --overwrite --device cuda --require-cuda --precision bf16 --steps 2 --batch-size 8 --gradient-accumulation-steps 1 --seq-len 32 --d-model 64 --n-heads 4 --n-layers 2 --resource-interval 0.05 --min-resource-samples 2
```

Ce profil écrit `llm_batch_profile.json`, refuse d'écraser un dossier existant sans `--overwrite`, et échoue si une brique Cortex complète manque, si le backend CUDA strict ne reste pas `extension`, si les samples GPU/VRAM/puissance sont absents ou si le pic mémoire CUDA torch reste nul. Le run local court RTX 5070 passe avec `passed=true`, `native_ternary_backend_counts={"extension":1417}`, variants `tiled_shared_memory_int2/warp_reduction_int2/wmma_tensor_core_int2`, toutes les phases P1-P10 actives, `512` tokens entraînés planifiés, `117.646` tokens/s wall-clock, GPU moyen `10.344%`, GPU max `16%`, puissance moyenne `37.702 W`, VRAM moyenne `971.812 MB`, CPU process moyen `6.070%` du total et pic CUDA torch alloué `34,972,160` bytes.

Pour éviter qu'un seul batch/seed masque une faiblesse, la matrice de profil lance plusieurs profils Cortex stricts et agrège les gates :

```bash
python tools/train_llm.py profile-matrix --out-dir runs/llm-batch-profile-matrix-v2 --overwrite --device cuda --require-cuda --precision bf16 --steps 1 --profile-shapes 32x64x4x2x4,40x64x4x2x4 --seeds 71,73 --min-cases 4 --require-multi-shape --require-multi-seed --min-train-tokens-per-second-mean 10 --min-gpu-utilization-percent-mean 5 --min-gpu-memory-used-mb-mean 900 --min-gpu-power-draw-watts-mean 30 --resource-interval 0.05 --min-resource-samples 2 --corpus-repeats 128 --max-corpus-tokens 4096
```

Elle écrit `llm_batch_profile_matrix.json` et `llm_batch_profile_matrix.csv`, puis échoue si un profil enfant échoue, si toutes les phases ne sont pas actives, si un run CUDA strict n'est pas extension-only, si les exigences multi-shape/multi-seed ne sont pas satisfaites, ou si un seuil explicite de throughput/GPU/VRAM/puissance n'est pas atteint. Le run court RTX 5070 avec seuils bloquants passe avec `case_count=4`, `passed_cases=4`, `shape_count=2`, `seed_count=2`, `strict_extension_only_cases=4`, `all_phases_active_cases=4`, `576` tokens planifiés, throughput moyen `77.310` tokens/s, GPU moyen par cas `13.336%`, puissance moyenne `39.612 W`, VRAM moyenne `976.110 MB`, et tous les `threshold_checks` passants.

Pour laisser le harness choisir les shapes les plus lourdes qui rentrent dans un budget mémoire avant de lancer la matrice stricte :

```bash
python tools/train_llm.py profile-autosize --out-dir runs/llm-profile-autosize-gradaccum-v1 --overwrite --device cuda --require-cuda --precision bf16 --steps 1 --candidate-seq-lens 32,48 --candidate-d-models 64,96 --candidate-n-layers 2 --candidate-batch-sizes 4,8 --candidate-gradient-accumulation-steps 1,2 --n-heads 4 --selected-shape-count 2 --min-selected-shapes 2 --seeds 71 --memory-budget-fraction 0.10 --measure-candidate-count 4 --measured-selection-metric throughput_gpu --min-cases 2 --require-multi-shape --min-train-tokens-per-second-mean 10 --min-gpu-utilization-percent-mean 5 --min-gpu-memory-used-mb-mean 900 --min-gpu-power-draw-watts-mean 30 --resource-interval 0.05 --min-resource-samples 2 --corpus-repeats 128 --max-corpus-tokens 4096
```

Ce rapport écrit `llm_batch_profile_autosize.json`, classe les candidats avec l'estimation mémoire du Transformer Cortex complet, rejette ceux qui dépassent le budget, mesure 4 candidats par défaut avec de vrais profils Cortex stricts, refuse les candidats dont la VRAM observée max dépasse le budget, cherche aussi `gradient_accumulation_steps` par candidat, mesure chaque candidat sur toutes les seeds fournies et au moins 2 seeds de mesure par défaut (`--measure-candidate-seed-count 1` peut forcer le diagnostic mono-seed), synthétise des seeds déterministes si nécessaire, utilise `--measure-candidate-strategy diverse` par défaut pour ne pas rester bloqué au top-N estimé, raffine ensuite la recherche en 2 vagues adaptatives par défaut sans dépasser `--measure-candidate-count`, répartit le budget restant entre les rounds quand `--measure-candidate-adaptive-rounds` est augmenté, explore les zones incertaines avec une borne haute `mean + stddev`, ajoute ensuite par défaut une seed de raffinement au front de décision avec `--refine-uncertain-candidate-count 1 --refine-uncertain-extra-seed-count 1`, exécute cette seed avec `--refine-uncertain-step-multiplier 2`, répète ce profil avec `--refine-uncertain-repeat-count 2`, moyenne les répétitions par `(seed, steps)`, confirme ensuite par défaut les finalistes provisoires avec seeds fraîches et plan runtime adaptatif qui augmente steps/repeats jusqu'à `--confirm-selected-runtime-step-multiplier-cap 4 --confirm-selected-runtime-repeat-count-cap 4` quand le signal d'incertitude est haut, confirme aussi les challengers non confirmés dont la borne haute chevauche le gagnant robuste avec un budget de rounds par défaut égal au nombre de candidats mesurés si l'utilisateur ne fixe pas de limite explicite, dispose d'un budget dédié `--confirm-selected-decision-resolution-extra-rounds` et d'une extension adaptative `--confirm-selected-decision-resolution-adaptive-extra-rounds` réévaluée après chaque round selon l'overlap courant des intervalles confirmés pour lancer `decision_margin_resolution` sur le gagnant et le meilleur challenger si la marge reste ambiguë, exige ensuite une marge de décision strictement positive entre la borne basse du gagnant et la borne haute du meilleur challenger, puis classe les candidats mesurés avec un score robuste `mean - stddev` au lieu d'une simple moyenne brute. Le JSON conserve `provided_seed_count`, `synthesized_measurement_seed_count`, `refinement_rounds`, `refinement_seeds`, `refinement_profile_count`, `confirm_selected_max_rounds`, `confirm_selected_decision_resolution_extra_rounds`, `confirm_selected_decision_resolution_adaptive_extra_round_cap`, `confirm_selected_decision_resolution_adaptive_extra_rounds`, `confirm_selected_decision_resolution_total_rounds`, `confirm_selected_runtime_step_multiplier_cap`, `confirm_selected_runtime_repeat_count_cap`, `confirmation_runtime_escalation_count`, `confirmation_runtime_escalations`, `confirmation_decision_resolution_uncertainty`, `confirmation_decision_resolution_margin_deficit`, `confirmation_decision_resolution_overlap_ratio`, `confirmation_decision_resolution_stop_reason`, `confirmation_decision_resolution_budget_evaluations`, `confirmation_rounds`, `confirmation_seeds`, `confirmation_profile_count`, `confirmation_decision_resolution_rounds_used`, `confirmation_complete`, `confirmation_decision_resolved`, `confirmed_shape_keys`, `confirmation_best_challenger_shape_key`, `confirmation_best_challenger_upper_confidence`, `confirmation_decision_margin`, `confirmation_pending_shape_keys`, `refinement_steps`, `confirmation_steps`, `measurement_seed_count`, `measurement_profile_count`, `measurement_profile_seeds`, `measurement_steps`, `measurement_repeat_indices`, `measured_score_profile_values`, `measured_score_observation_values`, les scores before/after du raffinement et de la confirmation, `measured_score_mean`, `min`, `max`, `stddev`, `lower_confidence`, `upper_confidence`, `stability_ratio` et les champs `source_*` des rounds adaptatifs pour auditer la décision. Si la confirmation reste incomplete ou si `confirmation_decision_resolved=false` après les rounds disponibles, le budget dédié et son extension adaptative séquentielle, le rapport échoue avec `selected_confirmation_incomplete` ou `selected_confirmation_decision_unresolved` et ne lance pas la matrice. Le run local court mesure 4 candidats parmi 16 viables sous `10%` de la mémoire CUDA libre, sélectionne `seq32_d96_h4_l2_b8_g2` et `seq48_d96_h4_l2_b8_g1` avec `selection_source=measured`, vérifie une fraction max de budget VRAM observée `0.906`, puis passe la matrice stricte avec 2/2 cas, `896` tokens planifiés, throughput moyen `283.958` tokens/s, GPU moyen `13.388%`, puissance moyenne `41.548 W` et VRAM moyenne `990.596 MB`. Le smoke multi-seed court `runs/llm-profile-autosize-multiseed-v1` mesure `seq32_d64_h4_l2_b4_g2` sur les seeds `71,73`, obtient 2/2 profils candidats passants et 2/2 cas matrix passants, avec throughput moyen `161.879` tokens/s, GPU moyen `13.202%`, puissance moyenne `40.732 W`, VRAM moyenne `978.536 MB`, `strict_extension_only_cases=2` et `all_phases_active_cases=2`. Le smoke diversité `runs/llm-profile-autosize-diverse-v1` mesure les rangs estimés `1,8`, sélectionne le rang `8` après mesure réelle, puis passe la matrice stricte avec throughput moyen `93.958` tokens/s, GPU moyen `15.182%`, puissance `39.636 W` et VRAM `978.182 MB`. Le smoke adaptatif `runs/llm-profile-autosize-adaptive-v1` mesure les rounds `[1,8]` puis `[2,5]`, sélectionne `seq64_d64_h4_l2_b4_g2`, et passe avec throughput moyen `323.963` tokens/s, GPU moyen `13.000%`, puissance `38.969 W` et VRAM `983.000 MB`. Le smoke UCB actuel fournit seulement `--seeds 71`, mesure `71,104800`, explore autour de la source adaptative `seq64_d96_h4_l2_b4_g2` avec `source_upper_confidence=4247.725`, sélectionne prudemment `seq64_d64_h4_l2_b4_g2` avec 8 profils candidats mesurés, score robuste `3906.117`, moyenne `4648.800`, stddev `742.683`, upper `5391.483`, et passe la matrice avec throughput moyen `296.592` tokens/s, GPU moyen `15.071%`, puissance `41.461 W`, VRAM `982.857 MB`, `strict_extension_only_cases=1`, `all_phases_active_cases=1`. Le smoke observation-refinement actuel mesure `71,104800`, ajoute `209529` deux fois sur 2 steps au candidat decisionnel `seq64_d96_h4_l2_b4_g2`, exécute 10 profils candidats, expose profils bruts `[1188.464,4215.040,8851.871,8260.925]`, observations groupées `[1188.464,4215.040,8556.398]`, sélectionne `seq64_d64_h4_l2_b4_g2` avec score robuste `4044.156`, moyenne `4345.331`, stddev `301.176`, puis passe la matrice avec throughput moyen `295.203` tokens/s, GPU moyen `15.571%`, puissance `41.639 W`, VRAM `983.429 MB`, `strict_extension_only_cases=1` et `all_phases_active_cases=1`. Le smoke large C46 `llm-profile-autosize-decision-resolved-current` consommait 20 profils candidats et 10 profils de confirmation avant blocage, ce qui a motive le budget dédié C47. Le smoke CUDA non-regression C50 `llm-profile-autosize-sequential-resolution-small-c50` passe avec `confirm_selected_max_rounds=1`, `confirm_selected_decision_resolution_extra_rounds=1`, `confirm_selected_decision_resolution_adaptive_extra_round_cap=2`, `confirm_selected_decision_resolution_adaptive_extra_rounds=0`, `confirmation_decision_resolution_stop_reason=decision_resolved`, `confirmation_decision_resolved=true`, `confirmation_decision_margin=612.512`, 4 profils candidats, 2 profils de confirmation, matrice 1/1, throughput moyen `166.762` tokens/s, GPU moyen `13.538%`, puissance `41.242 W`, VRAM `984.385 MB`, `strict_extension_only_cases=1` et `all_phases_active_cases=1`. Le test court C50 prouve que le contrôleur réévalue l'overlap après chaque round et peut consommer deux rounds adaptatifs successifs (`314200`, `418929`, `523658`) avant de résoudre la marge. Le smoke CUDA non-regression C51 `llm-profile-autosize-runtime-escalation-small-c51` passe avec `confirmation_runtime_escalation_count=1`, `confirmation_adaptive_runtime_applied=true`, `confirmation_steps=4`, `confirmation_step_multiplier=4`, `confirmation_repeat_count=3`, `confirmation_runtime_signal=0.925`, 5 profils candidats, 3 profils de confirmation, `confirmation_decision_resolved=true`, matrice 1/1, throughput moyen `166.266` tokens/s, GPU moyen `15.077%`, puissance `39.701 W`, VRAM `982.000 MB`, `strict_extension_only_cases=1` et `all_phases_active_cases=1`.

Le raffinement autosize choisit maintenant ses actions par `expected_gain_per_cost` par défaut, pas seulement par incertitude brute. Le JSON expose `refinement_budget_strategy`, `refinement_budget_action_count`, `refinement_budget_actions`, `refinement_budget_candidate_action_report_cap`, `refinement_budget_candidate_action_total_count`, `refinement_budget_candidate_action_count`, `refinement_budget_candidate_actions_truncated`, `refinement_budget_candidate_actions`, puis chaque candidat raffiné conserve `refinement_expected_gain`, `refinement_uncertainty_width`, `refinement_posterior_utility`, `refinement_measurement_cost_tokens`, `refinement_gain_per_cost`, `refinement_is_selected_finalist`, `selected_for_refinement` et `report_selection_reason` dans le front d'audit plafonné. Le smoke CUDA non-régression C52 `llm-profile-autosize-efficient-frontier-small-c52` passe avec `refinement_budget_strategy=expected_gain_per_cost`, un budget action, refined shape `seq64_d64_h4_l2_b4_g1`, `gain_per_cost=2.570`, 6 profils candidats, matrice 1/1, throughput moyen `88.083` tokens/s, GPU moyen `14.727%`, puissance `39.513 W`, VRAM `978.364 MB`, `strict_extension_only_cases=1` et `all_phases_active_cases=1`.

Le smoke CUDA metadata C53 `llm-profile-autosize-efficient-frontier-small-c53` passe aussi apres exposition du front complet avec `refinement_budget_action_count=1`, `refinement_budget_candidate_action_count=2`, action selectionnee `seq64_d64_h4_l2_b4_g1` a `gain_per_cost=2.947`, action rejetee `seq32_d64_h4_l2_b4_g1` a `gain_per_cost=1.887`, matrice 1/1, throughput moyen `90.486` tokens/s, GPU moyen `12.364%`, puissance `39.020 W`, VRAM `981.727 MB`, `strict_extension_only_cases=1` et `all_phases_active_cases=1`.

Le smoke CUDA C54 `llm-profile-autosize-capped-frontier-small-c54` passe avec `--refinement-budget-candidate-action-report-cap 1`, `refinement_budget_candidate_action_total_count=2`, `refinement_budget_candidate_action_count=1`, `refinement_budget_candidate_actions_truncated=true`, matrice 1/1, throughput moyen `75.069` tokens/s, GPU moyen `14.833%`, puissance `44.263 W`, VRAM `955.333 MB`, `strict_extension_only_cases=1` et `all_phases_active_cases=1`.

`tools/benchmark_learned_memory_policy.py` exécute une ablation courte contrôlée : mêmes poids partagés, mémoire apprise active contre mémoire désactivée, puis entraînement de la seule politique exact/latent/drop. Le rapport JSON expose les losses avant/après, le gradient mémoire, les décisions exact/latent/drop et le delta `before - after`.

## Démo noyau

```bash
python -m cortex3 demo --seed 7 --n-per-skill 5
```

La démo compare une référence simple à un agent « compressé » volontairement corrompu sur les familles de compétences du Verifier OS. Le vérificateur détecte les régressions, l'adversaire génère des variantes et le regrowth propose des réparations minimales.

## Rapport de cycle

```bash
python tools/run_cycle_report.py
```

Ce rapport exécute le cycle complet : référence vs trial, régressions, ledgers, analyse des causes, actions budgetées, trace d'inférence Phase 8, plan de sleep phase Phase 9, propositions Phase 10 en sandbox, loss finale, métriques absolues, expériences A-E du plan et smoke de checkpoint autoregressif entraîné. Par défaut il écrit aussi un dossier `runs/<run-id>/` avec `summary.json`, `report.md` et `fault_matrix.json`.

```bash
python tools/run_cycle_report.py --seed 7 --n-per-skill 3
python tools/run_cycle_report.py --no-write  # console only
python tools/run_cycle_report.py --skip-inference  # sans trace Phase 8
python tools/run_cycle_report.py --skip-sleep  # sans trace Phase 9
python tools/run_cycle_report.py --skip-improvement  # sans trace Phase 10
python tools/run_cycle_report.py --skip-experiments  # sans expériences A-E
python tools/run_cycle_report.py --skip-autoregressive  # sans smoke checkpoint AR
```

## Pré-entraînement LLM comparatif

Le pont LLM complet se lance avec :

```bash
python tools/train_llm.py smoke --require-win
```

Ce smoke construit un corpus texte déterministe, entraîne un tokenizer BPE, écrit les tokens en streaming dans un fichier `uint32` memmap, hashe le memmap, le tokenizer et les shards source, échantillonne les batches causaux de façon vectorisée, entraîne une baseline Transformer next-token et un Transformer Cortex complet sur les mêmes données, sauvegarde les checkpoints liés à l'identité du corpus et produit :

- `comparison_report.json`
- `run_plan.json`
- `learning_curve_audit.json`
- `report.md`
- `learning_curve.png`
- `baseline_ntp/learning_curve.csv`
- `cortex3/learning_curve.csv`
- `cortex3/cortex_phase_report.json`
- `baseline_ntp/checkpoint_final.pt`
- `cortex3/checkpoint_final.pt`

Quand les horizons sont complets `[1, 2, 4, 8]`, le modèle Cortex active aussi le core ternaire `BitLinear`, la politique mémoire apprise exact/latent/drop et le contrôleur de phases P1-P10 pendant l'entraînement. Ce contrôleur exécute le Verifier OS, les contrats MTP/FSP token-level, les contrats output-goal, la mémoire cognitive, les certificats, l'attribution, le regrowth, le routage fast/normal/careful, la sleep phase et le gate d'amélioration récursive. Il ajoute une régularisation Cortex au loss, transforme les exemples sleep acceptés en replay causal tokenisé, applique les réparations P7 acceptées directement au Transformer via un patch borné et non-régressif des paramètres ciblés, convertit les propositions P10 acceptées en patchs signés avec rollback token, exige des dispatchs ternaires packés, écrit `cortex_phase_report.json`, et la preuve comparative exige `cortex_phase_integration_passed=true` dès qu'un run annonce l'architecture Cortex complète.

Pour un corpus plus large :

```bash
python tools/train_llm.py compare path/to/text_shards --out-dir runs/llm-large --steps 2000 --batch-size 64 --gradient-accumulation-steps 4 --checkpoint-interval 100 --precision auto --device cuda --require-cuda
python tools/train_llm.py compare path/to/text_shards --out-dir runs/llm-large --steps 4000 --resume --batch-size 64 --gradient-accumulation-steps 4 --precision auto --device cuda --require-cuda
```

Pour comparer plusieurs graines sur le même corpus tokenisé une seule fois :

```bash
python tools/train_llm.py compare-matrix path/to/text_shards --out-dir runs/llm-large-matrix --seeds 11,23,37 --steps 2000 --batch-size 64 --gradient-accumulation-steps 4 --checkpoint-interval 100 --precision bf16 --require-win --min-corpus-tokens 50000000 --min-planned-train-tokens 100000000
```

`compare-matrix` écrit un `corpus/manifest.json` partagé, puis un dossier `seed_<seed>` par graine avec rapports, courbes et checkpoints baseline/Cortex. Le rapport agrégé `comparison_matrix_report.json` mesure moyenne, médiane, variance, win-rate, minimum Cortex/baseline, régression next-token maximale, tokens corpus observés et tokens d'entraînement planifiés. Avec `--require-win`, les seuils `--min-corpus-tokens` et `--min-planned-train-tokens` deviennent bloquants : un ratio favorable sur un corpus trop petit ne peut pas passer pour une preuve large.

Pour un banc multi-corpus déjà préparé :

```bash
python tools/train_llm.py corpus-matrix --corpus c4=runs/c4-prepared/text_shards --corpus code=path/to/code_shards --out-dir runs/llm-corpus-matrix --seeds 11,23,37 --steps 2000 --batch-size 64 --gradient-accumulation-steps 4 --checkpoint-interval 100 --precision bf16 --require-win --min-corpus-tokens 50000000 --min-planned-train-tokens 100000000
```

Chaque corpus reçoit son propre `comparison_matrix_report.json`, et le dossier racine écrit `corpus_matrix_report.json`, `corpus_matrix_report.md`, `corpus_matrix_ratios.png` et `corpus_matrix_learning_curves.csv/png`. La preuve globale exige que tous les couples corpus x seed gagnent contre la baseline avec score baseline non nul et, si les seuils d'échelle sont fournis, que chaque corpus/seed les respecte.

Pour exécuter le pipeline complet depuis un manifeste reproductible :

```bash
python tools/train_llm.py preflight-experiment experiments/c4_cuda_large_manifest.json --out-dir runs/cortex3-c4-cuda-large-preflight
python tools/train_llm.py run-experiment experiments/c4_cuda_large_manifest.json
python tools/train_llm.py inspect-experiment runs/cortex3-c4-cuda-large
python tools/train_llm.py audit-experiment runs/cortex3-c4-cuda-large
```

Le manifeste décrit `doctor`, `training`, `model`, `seeds`, `require_win` et une liste de corpus `hf` ou `paths`. `preflight-experiment` vérifie le doctor et estime le pic mémoire modèle/batch/GPU sans préparer le corpus. `run-experiment` écrit `experiment_manifest.normalized.json`, `doctor_report.json`, `preflight_report.json`, prépare les corpus HF sous `prepared/<corpus>`, lance `corpus-matrix`, puis produit `experiment_report.json`, `experiment_report.md` et les courbes agrégées sous `corpus_matrix/`. Pour les runs longs, `training.resume_if_exists` réutilise les exports HF, manifests tokenisés et checkpoints vérifiés quand ils existent, tout en démarrant proprement si aucun artefact n'est encore présent. `inspect-experiment` inspecte un run terminé ou en cours sans charger les gros checkpoints : processus actifs, snapshot GPU, manifests, derniers checkpoints, dernières courbes CSV, `training_report.json` et `cortex_phase_report.json` quand ils existent. `model.tokenizer_training_chars` borne l'échantillon CPU utilisé pour entraîner le BPE, et `model.max_corpus_tokens` arrête le memmap tokenisé dès que le corpus massif utile est atteint. Chaque `training_report.json` inclut aussi `resource_usage` avec CPU moyen, CPU process, GPU moyen, mémoire GPU moyenne/max et nombre d'échantillons pour vérifier si le run exploite vraiment la machine. Chaque run Cortex complet écrit aussi `cortex_phase_report.json`; si une phase P1-P10 manque ou échoue, `cortex_phase_integration_passed` devient faux et le proof ne passe pas. Après un long run, `audit-experiment` relit les artefacts persistés, vérifie les preuves `passed`, les manifests tokenisés, les shards HF, les courbes CSV/PNG et les checkpoints baseline/Cortex non vides.

Deux manifestes versionnés sont fournis :

- `experiments/wikitext_cuda_validation.json` : validation GPU rapide sur Wikitext.
- `experiments/c4_cuda_large_manifest.json` : run long CUDA large C4 avec seuils de preuve massifs versionnes.

Extrait minimal :

```json
{
  "name": "cortex3-large-corpus",
  "out_dir": "runs/cortex3-large-corpus",
  "doctor": {"require_cuda": true, "device": "cuda", "precision": "bf16", "distributed": true},
  "seeds": [11, 23, 37],
  "require_win": true,
  "model": {"vocab_size": 32768, "seq_len": 1024, "d_model": 512, "n_heads": 8, "n_layers": 8, "horizons": [1, 2, 4, 8], "min_corpus_tokens": 50000000, "max_corpus_tokens": 64000000, "tokenizer_training_chars": 64000000, "min_planned_train_tokens": 2000000000},
  "training": {"steps": 32000, "batch_size": 4, "gradient_accumulation_steps": 16, "checkpoint_interval": 5, "max_intermediate_checkpoints": 5, "cortex_phase_interval": 500, "resume_if_exists": true},
  "corpora": [
    {"name": "c4", "kind": "hf", "dataset": "allenai/c4", "config_name": "en", "split": "train", "text_field": "text", "max_documents": 1000000, "max_characters": 350000000}
  ]
}
```

Pour préparer un corpus Hugging Face massif en shards texte puis memmap tokenisé :

```bash
python tools/train_llm.py prepare-hf --dataset allenai/c4 --config-name en --split train --text-field text --out-dir runs/c4-prepared --max-documents 1000000 --max-characters 350000000 --vocab-size 32768 --seq-len 1024 --max-horizon 8 --tokenizer-training-chars 64000000 --max-tokens 64000000
python tools/train_llm.py prepare-hf --dataset allenai/c4 --config-name en --split train --text-field text --out-dir runs/c4-prepared --max-documents 1000000 --max-characters 350000000 --vocab-size 32768 --seq-len 1024 --max-horizon 8 --tokenizer-training-chars 64000000 --max-tokens 64000000 --resume
python tools/train_llm.py prepare-hf --dataset Salesforce/wikitext --config-name wikitext-2-raw-v1 --split train --text-field text --out-dir runs/wikitext2-prepared --max-documents 200 --vocab-size 512 --seq-len 64 --max-horizon 4
python tools/train_llm.py compare runs/c4-prepared/text_shards --out-dir runs/c4-cortex-vs-ntp --steps 2000 --batch-size 8 --gradient-accumulation-steps 8 --checkpoint-interval 100 --precision bf16 --resume-if-exists --max-corpus-tokens 64000000 --tokenizer-training-chars 64000000
python tools/train_llm.py compare-matrix runs/c4-prepared/text_shards --out-dir runs/c4-cortex-vs-ntp-matrix --seeds 11,23,37 --steps 2000 --batch-size 8 --gradient-accumulation-steps 8 --checkpoint-interval 100 --precision bf16 --resume-if-exists --require-win --min-corpus-tokens 50000000 --max-corpus-tokens 64000000 --tokenizer-training-chars 64000000 --min-planned-train-tokens 100000000
python tools/train_llm.py corpus-matrix --corpus c4=runs/c4-prepared/text_shards --out-dir runs/corpus-suite --seeds 11,23,37 --steps 2000 --batch-size 8 --gradient-accumulation-steps 8 --checkpoint-interval 100 --precision bf16 --resume-if-exists --require-win --min-corpus-tokens 50000000 --max-corpus-tokens 64000000 --tokenizer-training-chars 64000000 --min-planned-train-tokens 100000000
```

Utilise les identifiants Hugging Face namespacés (`Salesforce/wikitext`, `allenai/c4`, etc.). Si Hub rejette un ancien ID court comme `wikitext`, le CLI échoue maintenant avec un message indiquant l'ID namespacé à utiliser.

Pour un dataset local JSONL compatible Hugging Face :

```bash
python tools/train_llm.py prepare-hf --dataset json --data-file path/to/corpus.jsonl --split train --text-field text --out-dir runs/json-prepared
```

Sans limite explicite, `prepare-hf` plafonne l'export à 100 000 documents pour éviter un lancement massif accidentel. Pour un vrai job complet, passe une limite de caractères/documents adaptée ou `--allow-unbounded` de façon explicite. Si tu as un token Hugging Face, exporte `HF_TOKEN` avant le run pour éviter les limites du mode anonyme. `prepare-hf --resume` réutilise uniquement un export HF complet avec `hf_export_report.json`, shards présents, `prepare_report.json` et manifest tokenisé vérifié ; si les shards, le rapport, la recette de préparation du tokenizer/memmap ou la config de tokenization ne correspondent pas, la commande échoue au lieu d'écraser ou de reconstruire silencieusement.

Pour l'entraînement, `--resume` reprend strictement depuis `checkpoint_final.pt` ou le plus récent `checkpoint_step_*.pt` du dossier baseline/Cortex. `--resume-if-exists` est le mode adapté aux runs longs : il démarre proprement au premier lancement, puis réutilise les corpus/tokenizers/checkpoints vérifiés quand ils existent. Si le corpus manifest, la recette tokenisée (`vocab_size`, `min_frequency`, `seq_len`, horizon, chunking), l'identité SHA-256 du corpus, le checkpoint attendu ou le champ `corpus_identity` manque en mode strict, ou si le checkpoint ne correspond pas au corpus courant, la commande échoue au lieu de repartir de zéro silencieusement.

Pour refuser tout fallback CPU quand un run GPU est obligatoire :

```bash
python tools/train_llm.py doctor --require-cuda --precision fp16 --device cuda
python tools/train_llm.py compare path/to/text_shards --require-cuda --precision fp16 --device cuda
```

`doctor` écrit `doctor_report.json` et audite les dépendances Python, CUDA, les backends `torch.distributed`, Gloo/NCCL et la compatibilité du mode de précision demandé.

Le rapport compare une baseline next-token classique à Cortex-3 sur `verified_future_tokens_per_forward_cost`, tout en contrôlant la régression de loss next-token. Le proof gate refuse aussi les victoires artificielles où la baseline a un score nul ou inférieur à `min_baseline_future_tokens_per_cost`, afin qu'un ratio énorme causé par une division par quasi-zéro ne puisse pas passer. Le smoke local validé montre une baseline non nulle et un avantage Cortex coût/qualité, mais il ne remplace pas encore un run corpus massif GPU multi-nœuds.

Un benchmark multi-domaines déterministe est aussi disponible :

```bash
python tools/train_llm.py benchmark --domains sequence,anchors --precision bf16 --require-win
```

Il génère plusieurs corpus contrôlés, entraîne baseline et Cortex sur chaque domaine, agrège les ratios Cortex/baseline et écrit `benchmark_report.json`, `benchmark_report.md` et `benchmark_ratios.png`. Le runtime supporte DDP via `torch.distributed`; sur Windows/Gloo, le lanceur local ci-dessous utilise un TCPStore explicite `use_libuv=False` et une interface Gloo fixée.

Pour une preuve plus robuste avec variance inter-seeds :

```bash
python tools/train_llm.py benchmark-matrix --domains sequence,anchors --seeds 11,23,37 --precision bf16 --require-win
```

Cette commande exécute chaque domaine pour chaque seed, persiste les artefacts par couple `seed_<seed>/<domain>`, puis écrit `statistical_benchmark_report.json`, `statistical_benchmark_report.md` et `statistical_benchmark_ratios.png`. La preuve ne passe que si chaque échantillon domaine x seed gagne contre la baseline avec score baseline non nul et régression next-token bornée.

Pour valider un vrai run DDP local sans dépendre de `torchrun` elastic quand le build Windows CPU de PyTorch n'a pas le support libuv :

```bash
python tools/launch_llm_ddp.py --nproc 2 --master-port 29752 --gloo-interface Ethernet -- smoke --out-dir runs/llm-ddp-smoke-validation --steps 48 --precision bf16 --require-win
```

Le lanceur exporte `WORLD_SIZE/RANK/LOCAL_RANK`, force le backend Gloo sur l'interface réseau indiquée, désactive le TCPStore libuv via le runtime Cortex et écrit les logs par rank dans `runs/llm-ddp-worker-logs`.

## Tests

```bash
python -m unittest discover -s tests
python -m pytest tests/test_llm_pretraining.py -q
```

## Roadmap immédiate

1. Durcir Phase 1 jusqu'au statut Verifier OS complet : coût par cas réel, familles génératives plus larges, tests de faux positifs/faux négatifs d'oracle.
2. Durcir Phase 2 au-delà de l'auto-sizing court déjà mesuré, diversifié, adaptatif, raffiné et confirmé : élargir les candidats, les batchs LLM, la durée des profils et les seuils GPU/throughput, en conservant la sélection multi-seed sous budget VRAM observé.
3. Étendre Phase 3 vers des suites held-out plus larges et benchmarks MTP/output-goal vs NTP sur coût vérifié.
4. Étendre Phase 4 au-delà de l'ablation courte actuelle avec benchmarks coût/qualité de la politique mémoire apprise exact/latent/drop sur long contexte et held-out anchors.
5. Étendre Phase 5 au-delà de l'algèbre linéaire multi-étapes et des tests code visibles/cachés/propriétés vers des solveurs/domaines certifiés plus larges, puis mesurer en held-out les économies de tokens de certificat.
6. Étendre la boucle générative autoregressive vers held-out suites, benchmarks coût/qualité plus larges et calibration de confiance.
7. Étendre le banc MTP vs NTP en faible précision sur variantes de checkpoints autoregressifs et LLM.
8. Durcir Phase 6 avec ablations branchées sur de vrais forward passes multi-couches.
9. Calibrer Phase 7 sur des runs longs multi-corpus : fréquence des patchs, bornes de delta, effet cumulé et rollback persistant signé.
10. Étendre le harness LLM vers des checkpoints plus larges, puis auditer les propositions acceptées sur des patchs signés avec rollback persistant.

## Phrase centrale

> L'intelligence utile est la capacité de transformer une résolution lente vérifiée en circuit rapide, compressé, réutilisable et non-régressif.
