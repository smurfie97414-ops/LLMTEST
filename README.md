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
- `cortex3_phases.py` : registre exécutable des 10 phases Cortex-3 ;
- `cortex3_ledgers.py` : Bit Ledger, Skill Ledger, Causal Ledger et Uncertainty Ledger ;
- `cortex3_analysis.py` : analyse des causes probables d'une régression ;
- `cortex3_cycle.py` : cycle complet référence/trial → vérification → ledgers → analyse → actions budgetées → rapport ;
- `cortex3_selection.py` : sélection offline de trials et choix des compétences frontières ;
- `tools/run_cycle_report.py` : génération d'un rapport markdown du cycle ;
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

## Démo noyau

```bash
python -m cortex3 demo --seed 7 --n-per-skill 5
```

La démo compare une référence simple à un agent « compressé » volontairement corrompu. Le vérificateur détecte les régressions, l'adversaire génère des variantes et le regrowth propose des réparations minimales.

## Rapport de cycle

```bash
python tools/run_cycle_report.py
```

Ce rapport exécute le cycle complet : référence vs trial, régressions, ledgers, analyse des causes et actions budgetées.

## Tests

```bash
python -m unittest discover -s tests
```

## Roadmap immédiate

1. Étendre les skills vers le code exécutable et les tests unitaires générés.
2. Ajouter un module PyTorch `BitLinear` sign+mask avec residual synapse buffer.
3. Ajouter des checks courts pour FSP/output goals et les connecter au cycle.
4. Ajouter une mémoire latente simulée + registre exact d'ancres.
5. Ajouter la sélection de chemins fast/normal/careful dans une vraie boucle d'inférence.
6. Ajouter des rapports JSON/markdown persistés dans `runs/`.
7. Ajouter le premier micro-entraînement toy model pour tester MTP vs NTP en faible précision.

## Phrase centrale

> L'intelligence utile est la capacité de transformer une résolution lente vérifiée en circuit rapide, compressé, réutilisable et non-régressif.
