# Cortex-3 Architecture Self-Critique

Etat: boucle d'audit 12 apres integration du backend PyTorch C++/CUDA extension strict par defaut dans le vrai training `BitLinear`, avec forward packe, backward `grad_input` WMMA fp16 quand la shape est alignee, backward `grad_weight` + `grad_bias` WMMA fp16->fp32 quand la shape est alignee, kernels warp/tiled hand-written pour les petites formes non alignees, requantization/packing post-update, compteurs backend/requantize/grad-input/grad-weight explicites, precision CLI `auto -> fp16` sur CUDA, doctor strict et smoke LLM CUDA sans fallback autorise.

Ce document sert de registre de critique et de correction. Il ne remplace pas les tests longs interdits pour cette iteration; il se limite aux preuves courtes disponibles, aux rapports du code et aux benchmarks GPU courts.

## Boucle 1 - Corrections executees maintenant

### C1. Kernel ternaire natif trop simple

- Critique: le premier backend `native_int2_cupy_cuda` avait un thread par sortie et relisait/decodait les memes poids pour chaque ligne. C'etait un vrai kernel, mais pas assez ambitieux pour le statut "kernel fusionne efficace".
- Correction: ajout de deux kernels hand-written:
  - `tiled_shared_memory_int2`: tile 16x16 sorties, tile K=32, X et poids ternaires decodes en shared memory, bias/residual integres, accumulation fp32.
  - `warp_reduction_int2`: une sortie par warp, reduction K sur 32 lanes, utile pour K plus grand.
- Integration: `BitLinear` garde le forward ternaire packe + STE, selectionne `auto/tiled/warp`, trace le backend et la variante dans `PackedTernaryDispatch`, et les rapports LLM exposent `native_ternary_kernel_variants`.
- Verification courte: tests CUDA fp32/fp16/bf16, gradient STE, full Cortex court et benchmarks courts RTX 5070.
- Statut: corrige pour cette boucle, mais pas encore "termine final" car il manque autotuning large et mesures energie/VRAM.

### C2. Auto-selection kernel trop grossiere

- Critique: un seuil fixe ne peut pas etre "extraordinairement efficace" sur ce PC, car le meilleur variant depend du GPU, dtype, batch, K, N, bias/residual et du bruit runtime.
- Correction boucle 1: seuil heuristique ajuste temporairement.
- Correction boucle 2: remplacement par autotune CUDA-event. En mode `auto`, `BitLinear` prechauffe `tiled` et `warp`, mesure les deux variants, choisit le temps minimal, cache par device/dtype/shape et trace `autotuned`, `autotune_cache_hit`, `autotune_candidate_ms`.
- Correction boucle 3: export/import JSON du cache, champ `TransformerConfig.native_ternary_autotune_cache_path`, sauvegarde automatique optionnelle et cache layer-local pour eviter de refaire la selection host-side a chaque forward.
- Verification courte: tests CUDA exigent candidates `tiled/warp`, choix egal au meilleur temps mesure, cache-hit au deuxieme appel, profil sauvegarde, cache memoire vide, profil recharge et cache-hit sur nouveau layer; benchmark RTX 5070 `128x256x256 fp16` selectionne `warp_reduction_int2` apres mesure `tiled=0.1665 ms`, `warp=0.1368 ms`.
- Statut: corrige pour la selection runtime locale et la persistance de profil; reste a benchmarker davantage de shapes LLM.

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

### C5. Autotune biaise par le premier warmup

- Critique: le premier benchmark `auto` mesurait le premier candidat avec des couts de warmup/allocation et pouvait sur-penaliser `tiled`.
- Debug: comparaison `auto`, puis variants forces `tiled` et `warp`; le probleme venait de la sequence de mesure, pas d'un crash kernel.
- Correction: pre-warm de tous les candidats avant toute mesure comparative, puis warmup/repeat par variant avec CUDA events.
- Verification courte: `py_compile`, tests CUDA ciblés et benchmark auto relances; les candidates sont maintenant exposees et le choix correspond au minimum mesure.
- Statut: corrige pour cette boucle.

### C6. Profil autotune non persistant entre process

- Critique: le cache global en memoire obligeait chaque nouveau process de training a re-mesurer les memes shapes.
- Correction: ajout de `native_ternary_autotune_cache_snapshot`, `save_native_ternary_autotune_cache`, `load_native_ternary_autotune_cache`, `clear_native_ternary_autotune_cache`; `BitLinearConfig.native_cuda_autotune_cache_path` charge/sauve automatiquement le profil; `TransformerConfig.native_ternary_autotune_cache_path` le branche dans le vrai modele LLM.
- Verification courte: test CUDA persiste un profil JSON, vide le cache, recharge le profil, puis verifie qu'un nouveau layer de meme shape obtient un cache-hit sans re-mesure.
- Statut: corrige.

### C7. Selection host-side encore payee dans la boucle forward

- Critique: le benchmark avec profil persistant a montre `native_ms=0.3425 ms`, incoherent avec les candidates autotune, parce que `_native_cuda_packed_output` recalculait la cle device complete avant chaque launch mesure.
- Debug: le profil indiquait `warp=0.1368 ms`; apres comparaison, le cout venait de la selection Python/device-properties dans la fenetre de mesure CUDA events.
- Correction: ajout d'un cache layer-local par dtype/shape/device, alimente par mesure ou profil global, pour que les forwards suivants lancent directement le variant choisi.
- Verification courte: benchmark relance sur RTX 5070 `128x256x256 fp16`, natif `0.0971 ms`, unpack+linear `0.2095 ms`, speedup `2.16x`.
- Statut: corrige.

### C8. Cout STE dense rendu mesurable

- Critique: a ce stade de la boucle, le chemin training utilisait encore le STE dense pour garder les gradients, mais on ne mesurait pas son cout separement.
- Correction: `tools/benchmark_ternary_kernel.py` mesure maintenant `ste_dense_ms`, `full_bitlinear_forward_ms`, `estimated_training_forward_native_plus_ste_ms`, `speedup_vs_ste_dense` et les erreurs vs STE.
- Verification courte: benchmark RTX 5070 `128x256x256 fp16`: natif `0.0971 ms`, STE dense `0.1963 ms`, estime natif+STE `0.2933 ms`.
- Statut: instrumentation corrigee; optimisation du backward/STE reste ouverte.

### C9. Dense STE calcule dans le forward chaud

- Critique: meme apres kernel natif, `BitLinear.forward` calculait encore `F.linear(x, ste_weight)` pour creer le graphe STE, ce qui ajoutait un cout dense au forward training.
- Correction: ajout de `_PackedTernarySTEFunction`. Le forward retourne directement la valeur packee/native, sauvegarde les buffers packes, puis le backward reconstruit la contribution STE pour `grad_input`, `grad_weight` et `grad_bias`. Le chemin dense reste disponible via `use_fast_ste_autograd=False` pour debug.
- Verification courte: test CPU compare pertes, `grad_input`, `grad_weight` et `grad_bias` entre fast-STE et dense-STE; test CUDA verifie native forward + backward; benchmark RTX 5070 `128x256x256 fp16` donne full `BitLinear` forward `0.1323 ms` contre ancien `native+STE dense` `0.3657 ms`, speedup `2.76x`.
- Statut: corrige pour le forward; le backward a ensuite ete durci par C11 et C14.

### C10. Repack/requantization inutile a chaque forward

- Critique: le forward resynchronisait les buffers `packed_codes` depuis `float_weight` a chaque appel, meme sans changement de poids.
- Correction: ajout de `_packed_weight_version` base sur la version PyTorch du parametre; repack seulement si la version change. Les modifications optimizer step, P7/P10 ou `copy_` restent detectees.
- Verification courte: test dedie confirme zero repack sur deux forwards inchanges et un repack apres modification in-place du poids; le test existant de sync apres update reste passant.
- Statut: corrige.

### C11. Backward STE encore trop dense

- Critique: apres C9, le forward training ne payait plus le `F.linear` dense STE, mais le backward reconstruisait encore un poids dense pour calculer `grad_input`. Cela limitait la valeur "hardware-native ternary" pendant l'entrainement.
- Correction: ajout de kernels `ternary_grad_input_warp_fp32/fp16/bf16`. Le backward CUDA de `_PackedTernarySTEFunction` calcule maintenant `grad_input = grad_output @ W_ternary` directement depuis les codes int2 packes, les scales et le residual optionnel, avec accumulation fp32 par warp.
- Verification courte: `test_native_ternary_cuda_fast_ste_backward_matches_dense_ste` compare pertes, `grad_input`, `grad_weight` et `grad_bias` entre fast STE natif et dense STE en fp32/fp16/bf16, avec et sans residual runtime, sur GPU. Benchmark RTX 5070 `128x256x256 fp16`: forward+backward fast STE `0.9865 ms` contre dense STE legacy `1.3301 ms`, soit `1.35x`, erreur max `0.000976`.
- Statut: corrige pour `grad_input`; le `grad_weight` dense restant est traite par C14.

### C12. Requantization/packing post-optimizer encore tensorielle

- Critique: C10 evitait le repack a chaque forward inutile, mais apres chaque optimizer step ou patch P7/P10 il fallait encore regenerer `signs`, `mask`, `scales`, `residual_weight` et `packed_codes`. Le chemin precedent etait une suite d'operations PyTorch tensorielles, donc pas un noyau ternaire fusionne de training.
- Correction: ajout de `ternary_requantize_pack_fp32/fp16/bf16`, un RawKernel CUDA row-wise. Chaque block reduit `abs(weight)` pour calculer la scale, applique le seuil, ecrit signs/mask/residual, packe directement les codes int2 et retourne le compte d'actifs par ligne. `_sync_quantized_buffers_from_weight` l'utilise automatiquement sur CUDA/shared-scale et leve l'erreur si `require_native_cuda_kernel=True` mais que le kernel echoue.
- Debug effectue: la premiere version activait trop de poids avec seuil fixe parce que les scalaires RawKernel etaient passes sans type explicite; les arguments sont maintenant forces en `cp.int32`/`cp.float32`.
- Verification courte: `test_native_ternary_cuda_requantize_pack_matches_torch_sync` compare signs, mask, scales, residuals, packed codes et compte d'actifs contre le chemin PyTorch en fp32/fp16/bf16, avec threshold automatique et seuil fixe plus residual threshold. Benchmark RTX 5070 `128x256x256 fp16`: requantize/pack natif `0.2245 ms` contre PyTorch `0.5901 ms`, soit `2.63x`.
- Statut: corrige pour le chemin post-update local; reste a profiler sur toutes les shapes LLM.

### C13. Packaging C++/CUDA rendu executable dans le training

- Critique: l'auto-critique disait "pas extension C++/CUDA packagee" sans que le doctor sache si le PC pouvait vraiment la builder, puis le premier smoke prouvait seulement un kernel jouet. C'etait insuffisant pour affirmer que le training Cortex utilisait le backend extension.
- Diagnostic reel: `cl` existe via Visual Studio Community 18 (`MSVC 14.51`), mais `nvcc 12.8` plante dans `cudafe++` avec ce host compiler, meme sur un `.cu` minimal. Visual Studio Build Tools 2022 est aussi installe (`MSVC 14.44`), et ce chemin compile correctement avec le toolkit CUDA 12.8 user-level installe par micromamba dans `C:\Users\hight\.codex\cuda-12.8\Library`.
- Correction: `llm_doctor_report` expose maintenant plusieurs `cuda_home_candidates`, choisit le `nvcc` qui matche `torch.version.cuda=12.8`, detecte les installations Visual Studio/cl et prefere VS2022. `BitLinearConfig.native_cuda_backend` et `TransformerConfig.native_ternary_backend` utilisent `extension` par defaut; `auto` et `rawkernel` restent disponibles comme diagnostics explicites, mais `LLMTrainer` refuse ces modes en training CUDA Cortex strict. Le backend extension compile les kernels Cortex forward, `grad_input`, `grad_weight` + `grad_bias` et `requantize_pack` via `torch.utils.cpp_extension.load_inline`, avec `cuda-libraries-dev` et `cuda-cccl` locaux pour `cusparse.h` et `nv/target`.
- Verification courte: `tools\train_llm.py doctor --require-cuda --require-cuda-extension --precision bf16 --device cuda` passe; `test_bitlinear_native_extension_cuda_dispatch_runs_on_gpu` force `native_cuda_backend=extension`; `test_cuda_ternary_training_contract_is_strict_extension` prouve que CUDA Cortex refuse `auto` et force tous les `BitLinear` en `require_native_cuda_kernel=True`; le smoke LLM `tools\train_llm.py smoke --device cuda --require-cuda --steps 2` rapporte `native_ternary_backend_counts={'extension': 2185}`, `native_ternary_requantize_backend_counts={'extension': 230}`, `native_ternary_grad_weight_backend_counts={'extension': 160}`, `torch_packed_ternary_dispatches=0`, `strict_extension_only=true` et les audits P2/architecture passent.
- Statut: corrige pour le backend training Cortex extension strict par defaut; les mesures energie/VRAM/longues shapes restent a produire.

### C14. `grad_weight` STE dense dans le backward

- Critique: meme apres C11/C13, le backward gardait `grad_weight = grad_output^T @ input` via `torch.matmul`. C'etait exact, mais pas a la hauteur de l'objectif "kernel CUDA fusionne/hand-written complet": le forward, `grad_input` et requantize etaient natifs, tandis qu'un morceau central de l'entrainement restait opaque PyTorch dense.
- Correction: ajout d'un kernel CUDA extension `ternary_grad_weight_bias_tiled<input_t, output_t>`. Il tile `M x N x K` en shared memory, accumule en fp32, ecrit `grad_weight` au dtype du parametre, et fusionne `grad_bias` dans le meme lancement via les threads `local_k==0`. `_PackedTernarySTEFunction.backward` l'appelle via `_native_packed_ternary_grad_weight_bias_cuda`; en mode `native_cuda_backend="extension"`, toute erreur remonte au lieu de retomber silencieusement sur PyTorch dense.
- Observabilite: `CompressionTraceLedger` expose maintenant `native_ternary_grad_weight_backend_counts`, `native_ternary_extension_grad_weight_dispatches` et les audits P2/architecture exigent forward, requantize et `grad_weight` exclusivement extension quand le training CUDA strict est actif.
- Verification courte: build/load extension v2 OK avec `ternary_grad_weight_bias`; `test_bitlinear_native_extension_cuda_dispatch_runs_on_gpu` verifie forward, backward, requantize et `grad_weight` extension; `test_native_ternary_cuda_fast_ste_backward_matches_dense_ste` compare pertes, `grad_input`, `grad_weight` et `grad_bias` en fp32/fp16/bf16; smoke LLM CUDA extension rapporte `native_ternary_grad_weight_backend_counts={'extension': 160}`.
- Benchmark court RTX 5070 `batch=128, in=256, out=256, fp16`: `full_bitlinear_forward_backward_ms=0.6101 ms` contre `legacy_dense_ste_forward_backward_ms=0.9699 ms`, soit `1.59x`; forward complet `0.0976 ms`, kernel natif `0.0350 ms`, requantize speedup `3.90x`, erreur max `0.000976`.
- Statut: corrige pour le chemin training court et strict extension. Restent a produire: benchmark multi-shapes LLM, estimation energie/VRAM, comparaison longue Cortex vs baseline NTP quand les tests longs seront autorises, et eventuellement kernel Tensor Core/WMMA specialise si le kernel hand-written tuilé devient moins bon que cuBLAS sur grandes shapes.

### C15. Fallback encore possible via les valeurs par defaut

- Critique: apres C14, le kernel existait, mais `native_ternary_backend="auto"` restait le defaut dans `TransformerConfig`, `ComparisonConfig`, plusieurs CLI et `BitLinearConfig`. Sur CUDA, `device=auto` pouvait resoudre vers GPU alors que `require_native_ternary_kernel` restait faux, ce qui laissait un repli dense/RawKernel possible si l'extension echouait. C'etait incompatible avec l'objectif "pas de fallback".
- Correction: `extension` devient le defaut `BitLinearConfig`, `TransformerConfig`, `ComparisonConfig` et des commandes `smoke/compare/benchmark`. `LLMTrainer` verrouille le modele apres resolution du device: CUDA + coeur ternaire impose `native_ternary_backend=extension`, `use_native_ternary_kernel=True`, `require_native_ternary_kernel=True`, et propage ces contraintes a chaque `BitLinear`.
- Observabilite: les audits architecture/P2 passent de "extension observee" a "extension uniquement": forward, requantize et `grad_weight` doivent avoir des compteurs extension positifs, aucun backend non-extension positif, et `torch_packed_ternary_dispatches=0`.
- Verification courte: `py_compile` passe; `test_cuda_ternary_training_contract_is_strict_extension` et `test_default_ternary_training_backend_is_extension` figent le contrat; le smoke CUDA court sans `--native-ternary-backend` confirme `native_ternary_backend_requested=extension`, `torch_packed_ternary_dispatches=0` et `strict_extension_only=true`.
- Statut: corrige pour le contrat de training CUDA strict. Restent a traiter: benchmark multi-shapes/monitoring GPU et preuve longue Cortex vs baseline quand autorisee.

### C16. Matrice multi-shapes et monitoring GPU/CPU absents

- Critique: apres C15, la preuve perf restait centree sur une seule shape. On ne savait pas si le gain tenait sur plusieurs tailles, ni si le GPU et le CPU etaient mesures pendant le benchmark.
- Correction: `tools/benchmark_ternary_kernel.py` a maintenant un mode `--matrix`, des shapes repetables `--shape BATCHxINxOUT`, un backend `extension` par defaut, un `ResourceUsageMonitor` par cas, et un resume qui exige `strict_extension_only` + speedup forward/backward > 1 contre dense STE legacy.
- Verification courte: matrice fp16 `64x128x128`, `128x256x256`, `256x512x512`, `warmup=2`, `repeat=8` passee avec `strict_extension_only=true`, speedup min `1.37x`, speedup moyen `1.70x`, GPU moyen `8.67%`, CPU process moyen `26.67%`.
- Limite: les mesures `nvidia-smi` sont sub-seconde et n'ont souvent qu'un echantillon par cas; elles prouvent que le monitoring est branche, pas encore que l'occupation GPU finale est optimale.
- Statut: corrige pour matrice courte + monitoring branche. Le point "grandes shapes LLM + WMMA" est traite ensuite par C18.

### C17. Monitoring soutenu trop faible pour juger l'occupation GPU

- Critique: C16 branchait le monitoring mais la fenetre sub-seconde donnait souvent un seul echantillon par shape. C'etait trop faible pour discuter serieusement de GPU/VRAM/CPU moyens.
- Correction: `tools/benchmark_ternary_kernel.py` accepte maintenant `--sustain-seconds`, `--sustain-op`, `--sustain-sync-every` et `--min-resource-samples`. Le workload soutenu relance le meme chemin strict extension, synchronise par paquets, garde les compteurs `grad_weight` extension actifs, et le resume matrix refuse les cas qui n'atteignent pas le nombre minimal d'echantillons. `ResourceUsageMonitor` lit aussi `power.draw` quand `nvidia-smi` le fournit.
- Verification courte: matrice fp16 `64x128x128`, `128x256x256`, `256x512x512`, `--sustain-seconds 0.35`, `--min-resource-samples 2` passee avec `strict_extension_only=true`, `resource_samples_passed=true`, sample min `4`, speedup forward/backward min `1.02x`, moyen `1.41x`, GPU moyen `21.83%`, puissance moyenne `40.21 W`, CPU process moyen `25.28%`.
- Limite: le GPU reste loin d'une saturation ambitieuse sur ces petites shapes; c'est maintenant mesure au lieu d'etre suppose. Le prochain durcissement doit viser des shapes plus grandes ou une variante kernel plus adaptée.
- Statut: corrige pour monitoring soutenu court. Prochain point: remplir mieux le GPU sans retirer de composants, puis evaluer WMMA/Tensor Core pour `grad_weight`.

### C18. Grandes shapes LLM bloquees par `grad_input` warp et `grad_weight` non Tensor Core

- Critique: la matrice stricte large `256x768x768` + `512x1024x1024` fp16 a d'abord echoue le gate de speedup malgre `strict_extension_only=true`. Le cas `512x1024x1024` utilisait le GPU a `88%`, mais `full_bitlinear_forward_backward_ms=5.6812 ms` contre `1.5209 ms` dense STE legacy (`0.27x`). Le backend etait bien strict, donc le probleme etait un vrai goulet kernel, pas un fallback.
- Debug precis: micro-timing separe sur `512x1024x1024` a isole `grad_input_native_ms=4.5866 ms`, alors que `grad_weight_native_ms=0.0809 ms`. Le noyau `grad_input` warp lisait les codes int2 mais faisait un produit scalaire par sortie; il saturait du travail sans utiliser Tensor Cores.
- Correction: ajout de `ternary_grad_input_wmma_fp16`, qui calcule `grad_x = grad_out @ W_ternary` par tuiles 16x16, decode le poids int2 packe en shared memory, conserve les scales/residuals comme source runtime et accumule via WMMA fp16->fp32 avant conversion fp16. Ajout aussi de `ternary_grad_weight_bias_wmma_fp16_float`, selectionne pour `grad_weight = grad_out^T @ x` quand `M/N/K` sont multiples de 16 et que les poids entrainables restent fp32. L'extension est forcee en v4 pour eviter tout cache compile v2/v3.
- Observabilite: `last_native_grad_input_kernel`, `last_native_grad_weight_kernel`, `native_grad_input_kernel_counts` et `native_grad_weight_kernel_counts` sont exposes dans le benchmark et dans les rapports LLM. Le "dernier kernel" seul est insuffisant, car les phases P1-P10 lancent aussi de petits micro-circuits non alignes; les counts prouvent maintenant les deux familles.
- Verification courte: tests GPU ciblés passent, dont `test_bitlinear_native_extension_cuda_grad_weight_uses_wmma_when_aligned`, qui exige `wmma_fp16` pour `grad_input` et `wmma_fp16_float` pour `grad_weight`. Le micro-timing `512x1024x1024` passe de `grad_input_native_ms=4.5866` a `0.1194 ms`.
- Benchmark strict final RTX 5070: `tools\benchmark_ternary_kernel.py --matrix --shape 256x768x768 --shape 512x1024x1024 --dtype fp16 --kernel-variant auto --autotune-warmup 1 --autotune-repeat 2 --warmup 1 --repeat 4 --sustain-seconds 0.25 --sustain-op forward_backward --sustain-sync-every 2 --resource-interval 0.05 --min-resource-samples 2` passe avec `strict_extension_only=true`, `resource_samples_passed=true`, speedup forward/backward moyen `1.83x`, min `1.75x`, `gradInputCounts={"wmma_fp16":471/213}` et `gradWeightCounts={"wmma_fp16_float":471/213}` selon la shape, GPU moyen `42.58%`.
- Integration LLM: le smoke strict `tools\train_llm.py smoke --device cuda --require-cuda --steps 2` resout maintenant la precision CLI `auto` en `fp16`, garde `native_ternary_backend_counts={'extension': 2185}`, `native_ternary_requantize_backend_counts={'extension': 230}`, `native_ternary_grad_weight_backend_counts={'extension': 160}`, et rapporte `native_ternary_grad_input_kernel_counts={'warp': 152, 'wmma_fp16': 8}` + `native_ternary_grad_weight_kernel_counts={'tiled': 152, 'wmma_fp16_float': 8}`. Cela prouve que le Transformer aligne passe par WMMA, tandis que les petits micro-circuits des phases restent sur kernels hand-written sans fallback dense.
- Limite assumee: le smoke 2 steps ne constitue pas une preuve Cortex > baseline, car le proof gate rejette volontairement une baseline `future_tokens_per_cost=0` via `baseline_score_passed=false`. Ce gate reste strict pour eviter une victoire par division par quasi-zero; la preuve comparative reste les runs 48+ deja documentes ou les futurs runs larges quand les tests longs seront autorises.
- Statut: corrige pour le goulet large-shape fp16 et le branchement LLM strict. Restent a durcir: WMMA bf16, shapes non multiples de 16 sans repasser par warp lent, kernel forward Tensor Core pour grandes matrices, et preuve baseline longue sur corpus large.

## Critique phase par phase - boucle 11

### P1 - Verifier OS

- Ce qui est solide: familles arithmetic, algebra, long_context_anchor, entity_tracking, instruction_following, code_unit_tests, calibration; oracles stricts; metamorphic/anti-metamorphic; fault matrix; cout verifier par cas.
- Preuve actuelle: tests P1 et full Cortex court activent P1; rapports de cycle persistent les resultats.
- Faiblesse: generateurs encore limites, peu de domaines held-out, pas assez de bruit naturel, pas assez de faux positifs/faux negatifs hors familles internes.
- Risque architectural: si P1 est trop petit, P6/P7/P10 optimisent contre un monde trop facile.
- Correction prioritaire restante: elargir les generateurs et ajouter un audit de couverture d'oracles par domaine.

### P2 - Ternary Core

- Ce qui est solide: poids ternaires packes int2, quantization activations, STE, sync versionnee des buffers packes pendant training, kernels CUDA natifs tuiles/warp, autotune CUDA-event par shape, profil JSON persistant, cache layer-local, fast STE autograd forward, backward CUDA `grad_input` depuis poids int2 packes, WMMA fp16 `grad_input` aligne, WMMA fp16->fp32 `grad_weight` + `grad_bias` aligne, requantization/packing post-update fusionnee CUDA, backend extension C++/CUDA strict par defaut, doctor CUDA distinguant RawKernel et extension runtime, audit LLM exigeant native forward/requantize/grad_weight exclusivement extension en training CUDA strict.
- Preuve actuelle: tests CUDA courts, export/import profil, tests gradients fast-vs-dense STE en fp32/fp16/bf16, test de parite requantize/pack fp32/fp16/bf16, test extension forcee, doctor toolchain strict, benchmark RTX 5070 avec full forward/backward et requantize/pack profile, matrice strict extension 3 shapes avec monitoring soutenu GPU/CPU/power, matrice LLM-shape WMMA `256x768x768` + `512x1024x1024`, smoke LLM extension avec 2185 dispatches forward extension, 230 requantize extension, 160 `grad_weight` extension et compteurs WMMA `grad_input`/`grad_weight` positifs.
- Faiblesse: pas de mesure energie/VRAM longue; GPU moyen encore faible sur petites shapes soutenues; WMMA ne couvre pas encore bf16 ni les bords non multiples de 16; le forward packe peut devenir le prochain goulet sur grandes shapes.
- Risque architectural: le chemin training est maintenant completement branché en extension pour forward, `grad_input`, `grad_weight`, `grad_bias` et repack, mais une preuve de paradigme demandera que ce gain survive aux vrais batchs LLM et ne degrade pas la convergence.
- Correction prioritaire restante: bf16/edge tiles WMMA, profil forward large-shape, puis run long seulement quand autorise.

### P3 - Future Contract / FSP / MTP

- Ce qui est solide: horizons 1/2/4/8, confidence head, temporal consistency, gate observed-token, ledger et replay.
- Preuve actuelle: tests future-contract et full Cortex court.
- Faiblesse: contrats encore surtout token-level; peu de contrats objectifs haut niveau; pas de large comparaison MTP vs NTP sur held-out.
- Risque architectural: la speculation peut etre correcte sur micro-cas sans prouver un vrai gain de cout/qualite en LLM.
- Correction prioritaire restante: ajouter contrats de sortie/format et mesure de tokens verifies par cout sur petits benchmarks non longs.

### P4 - Memoire cognitive apprise

- Ce qui est solide: policy exact/latent/drop trainable, branchee dans forward, loss, P4 anchor supervision, checkpoints et audit.
- Preuve actuelle: test gradient policy, ablation courte a poids partages qui fige tout sauf `learned_memory.*`, delta positif `before - after` sur total et next-token loss, full Cortex court, counters exact/latent/drop/storage.
- Faiblesse: pas encore de preuve que la politique apprise bat une regle deterministe sur long contexte held-out; supervision derivee de loss locale encore simple.
- Risque architectural: la memoire est bien apprise et utile sur un batch controle, mais pas encore prouvee optimale ni generalisee.
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

- Statut: le forward lit les codes packes int2 et lance CUDA extension sur GPU; le backward CUDA calcule `grad_input` depuis les codes int2 packes avec WMMA fp16 quand la shape est alignee; `grad_weight` et `grad_bias` passent par WMMA fp16->fp32 quand la shape est alignee; la resynchronisation post-update requantize et repack directement en CUDA extension.
- Faiblesse: pas encore de profil energie/VRAM long; WMMA ne couvre pas encore bf16 ni les bords non multiples de 16; le forward packe doit etre profile comme prochain goulet potentiel.
- Correction restante: benchmarker les vrais batchs LLM, etendre WMMA aux dtypes/bords restants, puis specialiser le forward si necessaire.

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

## File de correction priorisee apres boucle 12

1. P2: etendre le chemin WMMA a bf16 et aux bords non multiples de 16 sans fallback dense.
2. P2: ajouter une variante forward Tensor Core/decode-shared pour grandes matrices si le forward packe warp devient le prochain goulet.
3. P4: scaler l'ablation learned memory vs deterministic memory sur anchors long-context synthetiques puis held-out.
4. P6/P7: afficher partout `repair_loss_before`, `repair_loss_after`, `protected_loss_before`, `protected_loss_after`, delta et convention.
5. P8: aligner `TernaryKernelDispatcher` inference avec les variants `BitLinear` natifs.
6. P1: ajouter un audit de couverture oracle/generateur par famille.
7. P3: ajouter contrats output-goal non token seulement.
8. P5: ajouter certificats algebra multi-step.
9. P9: audit diversity drift replay/sleep court.
10. P10: renforcer reward-hacking probes.
11. Training: produire un nouveau sidecar sous le commit courant quand les tests longs seront autorises.
12. Proof: relancer une comparaison 48+ ou large corpus afin que `baseline_score_passed=true` et que la victoire Cortex ne depende pas d'une baseline nulle.
