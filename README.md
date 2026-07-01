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
  ├── Future Contract / FSP
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
- `cortex3_ternary.py` : instrumentation Phase 2 avec quantization d'activations 8→4 bit, residual synapse buffer, compression logs, `BitLinear` sign+mask, buffers ternaires packes int2 et kernels CUDA natifs CuPy RawKernel tuilé/warp-reduction avec gradient STE ;
- `cortex3_future.py` : Phase 3 MTP/FSP sous contrat avec têtes PyTorch horizons 1/2/4/8, calibration autonome, confidence head, temporal consistency loss, Future Contract, révision et accept/reject gates ;
- `cortex3_memory.py` : Phase 4 mémoire cognitive avec KV récent exact, KV ancien latent compact, Exact Anchor Ledger, reconstruction conditionnée par requête, récupération de réponse augmentée par mémoire et vérificateur de fidélité aux ancres ;
- `cortex3_certificates.py` : Phase 5 raisonnement latent avec `latent proof state`, tête PyTorch de certificat calibrée, génération proof-carrying, certificats courts vérifiables, dé-latentisation aléatoire et vérification par outils ;
- `cortex3_attribution.py` : Phase 6 attribution causale avec ablations par blocs, experts, KV mode, horizon MTP, précision d'activation, contrat FSP, routage counterfactual et clustering de régressions ;
- `cortex3_regrowth.py` : Phase 7 regrowth minimal exécutable avec action space de réparation, simulation gain/coût, gate de non-régression et annealing vers re-cristallisation ;
- `cortex3_inference.py` : Phase 8 inférence fast/normal/careful avec routeur de difficulté, prédicteur de budget, early exit, Mixture-of-Depths `BitLinear`, KV latent, self-speculative MTP, certificats et dispatch kernel ternaire ;
- `cortex3_sleep.py` : Phase 9 sleep anti-collapse avec replay d'échecs, données synthétiques vérifiées et labellisées, réservoir réel/exogène, familles métamorphiques, filtre anti-collapse et scheduler de consolidation ;
- `cortex3_improvement.py` : Phase 10 Recursive Improvement Engine avec génération de propositions, sandbox en mémoire, évaluateur dynamique, gate Pareto/protection/diversité, archive évolutive et rollback ;
- `cortex3_objective.py` : loss finale du plan avec 17 termes pondérés et les 15 métriques absolues, dont `Verified Capability per Effective Joule` ;
- `cortex3_experiments.py` : expériences A-E du plan, de la détection de défauts injectés à la sandbox d'auto-amélioration ;
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

`requirements-cuda-cu128.txt` installe aussi `cupy-cuda12x` et `ml_dtypes`, nécessaires aux kernels CUDA natifs `native_int2_cupy_cuda_*`. En mode `auto`, `BitLinear` mesure `tiled` et `warp` sur la shape CUDA courante, cache le meilleur choix par device/dtype/shape, peut sauvegarder/recharger un profil JSON via `--autotune-cache`, saute le `F.linear` dense STE dans le forward grâce à un autograd custom, calcule `grad_input` CUDA depuis les poids int2 packés dans le backward fast STE, et ne repack les poids que si leur version change. Sur RTX 5070, les benchmarks courts locaux donnent :

- `batch=128, in=256, out=256, fp16` : autotune `warp_reduction_int2`, candidats `tiled=0.1665 ms`, `warp=0.1368 ms`, runtime natif `0.1042 ms`, forward `BitLinear` complet `0.1323 ms` contre `0.3657 ms` pour l'ancien chemin `native + STE dense`, soit `2.76x`, erreur max `0.000976`.
- `batch=128, in=256, out=256, fp16`, forward+backward : fast STE natif `1.0521 ms` contre dense STE legacy `1.3243 ms`, soit `1.26x`; le `grad_input` lit les codes int2 packés, tandis que `grad_weight` reste dense/exact.
- `batch=512, in=512, out=512, fp16` : autotune `warp_reduction_int2`, candidats `tiled=0.5668 ms`, `warp=0.3368 ms`, runtime natif `0.2561 ms` contre `0.2734 ms` pour unpack+`F.linear`, soit `1.07x`, erreur max `0.000976`.

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

Quand les horizons sont complets `[1, 2, 4, 8]`, le modèle Cortex active aussi le core ternaire `BitLinear`, la politique mémoire apprise exact/latent/drop et le contrôleur de phases P1-P10 pendant l'entraînement. Ce contrôleur exécute le Verifier OS, les contrats MTP/FSP, la mémoire cognitive, les certificats, l'attribution, le regrowth, le routage fast/normal/careful, la sleep phase et le gate d'amélioration récursive. Il ajoute une régularisation Cortex au loss, transforme les exemples sleep acceptés en replay causal tokenisé, applique les réparations P7 acceptées directement au Transformer via un patch borné et non-régressif des paramètres ciblés, convertit les propositions P10 acceptées en patchs signés avec rollback token, exige des dispatchs ternaires packés, écrit `cortex_phase_report.json`, et la preuve comparative exige `cortex_phase_integration_passed=true` dès qu'un run annonce l'architecture Cortex complète.

Pour un corpus plus large :

```bash
python tools/train_llm.py compare path/to/text_shards --out-dir runs/llm-large --steps 2000 --batch-size 64 --gradient-accumulation-steps 4 --checkpoint-interval 100 --precision bf16
python tools/train_llm.py compare path/to/text_shards --out-dir runs/llm-large --steps 4000 --resume --batch-size 64 --gradient-accumulation-steps 4 --precision bf16
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
2. Durcir Phase 2 au-delà du RawKernel natif actuel avec kernels CUDA plus tuilés/fusionnés, benchmarks latence/VRAM/énergie plus larges et mesure sous vrais batchs LLM.
3. Étendre Phase 3 vers des suites held-out plus larges, benchmarks MTP vs NTP et contrats FSP orientés objectifs de sortie.
4. Étendre Phase 4 au-delà de l'ablation courte actuelle avec benchmarks coût/qualité de la politique mémoire apprise exact/latent/drop sur long contexte et held-out anchors.
5. Étendre Phase 5 avec vérification algébrique multi-étapes, tests code plus riches et mesure held-out des économies de tokens de certificat.
6. Étendre la boucle générative autoregressive vers held-out suites, benchmarks coût/qualité plus larges et calibration de confiance.
7. Étendre le banc MTP vs NTP en faible précision sur variantes de checkpoints autoregressifs et LLM.
8. Durcir Phase 6 avec ablations branchées sur de vrais forward passes multi-couches.
9. Calibrer Phase 7 sur des runs longs multi-corpus : fréquence des patchs, bornes de delta, effet cumulé et rollback persistant signé.
10. Étendre le harness LLM vers des checkpoints plus larges, puis auditer les propositions acceptées sur des patchs signés avec rollback persistant.

## Phrase centrale

> L'intelligence utile est la capacité de transformer une résolution lente vérifiée en circuit rapide, compressé, réutilisable et non-régressif.
