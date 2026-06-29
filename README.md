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

## Ce qui est déjà implémenté dans cette première fondation

Cette base contient un package Python installable avec :

- `Task`, `CandidateAnswer`, `CostTrace`, rapports de vérification ;
- `ArithmeticSkill` : tâches math exactes avec transformations métamorphiques ;
- `LongContextAnchorSkill` : tests d'ancres exactes dans contexte bruité ;
- `InstructionSkill` : respect strict de format ;
- `DynamicSkillVerifier` : génération dynamique de suites de test ;
- `CompressionAdversary` : variantes adversariales depuis les échecs ;
- `TernaryBlock` : représentation sign+mask `{-1,0,+1}` avec zéros provisoires/certifiés/réversibles ;
- `ExactAnchorLedger` : extraction et fidélité des ancres exactes ;
- `AdaptiveHorizonPolicy` : politique MTP adaptative selon confiance/risque/domaine ;
- `MinimalRegrowthPlanner` : propositions de réparation à coût minimal ;
- une démo CLI vérification → adversaire → regrowth ;
- des tests unitaires et une CI GitHub Actions.

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
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .
```

## Démo

```bash
python -m cortex3 demo --seed 7 --n-per-skill 5
```

La démo compare une référence simple à un agent « compressé » volontairement corrompu. Le vérificateur détecte les régressions, l'adversaire génère des variantes et le regrowth propose des réparations minimales.

## Tests

```bash
python -m unittest discover -s tests
```

## Roadmap

1. Renforcer le Dynamic Skill Verifier : calibration, code exécutable, long contexte plus dur.
2. Ajouter un vrai `BitLinear` PyTorch sign+mask avec residual synapse buffer.
3. Ajouter les têtes MTP/FSP entraînables et la cohérence temporelle.
4. Ajouter une mémoire KV latente avec registre exact d'ancres.
5. Ajouter le raisonnement latent avec certificats courts.
6. Ajouter l'attribution causale des régressions.
7. Ajouter un moteur de regrowth qui applique réellement les réparations.
8. Ajouter un Recursive Improvement Engine sandboxé.

## Phrase centrale

> L'intelligence utile est la capacité de transformer une résolution lente vérifiée en circuit rapide, compressé, réutilisable et non-régressif.
