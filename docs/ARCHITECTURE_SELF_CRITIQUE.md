# Cortex-3 Architecture Self-Critique

Etat: boucle d'audit 29 apres integration du backend PyTorch C++/CUDA extension strict par defaut dans le vrai training `BitLinear`, avec forward packe tuilé/warp/WMMA fp16-bf16 autotuné, backward `grad_input` WMMA fp16/bf16 aligne ou padde sur bords non multiples de 16, backward `grad_weight` + `grad_bias` WMMA fp16/bf16->fp32 aligne ou padde sur bords non multiples de 16, kernels warp/tiled hand-written pour les petites formes, requantization/packing post-update, compteurs backend/requantize/grad-input/grad-weight explicites, precision CLI `auto -> fp16` sur CUDA, precision `bf16` executable sur smoke LLM CUDA, doctor strict, smoke LLM CUDA sans fallback autorise, profil batch LLM Cortex strict qui mesure throughput, CPU/GPU, puissance, VRAM `nvidia-smi` et memoire CUDA torch, matrice batch LLM courte multi-shape/multi-seed, seuils bloquants optionnels de throughput/GPU/VRAM/puissance, auto-sizing batch/shape sous budget memoire, recherche courte mesuree de candidats autosize avant matrice stricte sur les shapes choisies, gate de budget VRAM observee max avant selection finale, recherche de `gradient_accumulation_steps` par candidat pour augmenter l'effective batch sans retirer de composants Cortex, mesure multi-seed des candidats autosize avant selection, selection mesuree diversifiee qui n'est plus limitee au top-N estime, raffinement adaptatif en plusieurs vagues mesurees sans augmenter le nombre total de profils et en respectant le nombre de rounds demande, score de selection robuste qui penalise les candidats instables entre seeds, minimum de deux seeds de mesure candidate par defaut, puis exploration adaptative guidee par la borne haute `mean + stddev` pour sonder les zones prometteuses mais incertaines sans changer la selection finale prudente.

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
- Statut: corrige pour le goulet large-shape fp16 et le branchement LLM strict. Les bords non multiples de 16 sont traites par C19, puis bf16 par C20. Restent a durcir: kernel forward Tensor Core pour grandes matrices, mesures longues energie/VRAM et preuve baseline longue sur corpus large.

### C19. Bords non multiples de 16 encore hors chemin WMMA

- Critique: apres C18, le chemin WMMA etait rapide mais seulement pour `M/N/K` multiples de 16. Les grandes shapes LLM reelles ou certains batchs dynamiques peuvent produire `255x769x771`, `511x1025x1027` ou d'autres bords. Revenir au warp/tiled pour ces cas etait correct mais incompatible avec l'objectif "meilleure methode par defaut".
- Correction: extension v5 modifie `ternary_grad_input_wmma_fp16` et `ternary_grad_weight_bias_wmma_fp16_float` pour charger `grad_out`, poids ternaires decodes et activations dans des tuiles shared memory 16x16 avec zero-padding interne. Les stores vers `grad_x`, `grad_weight` et `grad_bias` sont bornes, donc aucun acces hors limites. Les dispatchers utilisent WMMA fp16 des que `M/N/K >= 16`, avec grille `ceil`, et ne gardent `warp/tiled` que pour les tres petits micro-circuits ou WMMA serait du gaspillage.
- Observabilite: les labels distinguent maintenant `wmma_fp16`, `wmma_fp16_padded`, `wmma_fp16_float` et `wmma_fp16_float_padded`. Les compteurs benchmark/LLM peuvent donc prouver qu'une shape edge n'est pas passee par le chemin aligne ou par un fallback dense.
- Verification courte: `test_bitlinear_native_extension_cuda_wmma_handles_edge_tiles` compare fast STE extension vs dense STE sur `batch=31, in=33, out=35`, exige `last_native_grad_input_kernel()=="wmma_fp16_padded"` et `last_native_grad_weight_kernel()=="wmma_fp16_float_padded"`, et verifie les gradients `x`, `float_weight` et `bias`.
- Benchmark strict edge RTX 5070: `tools\benchmark_ternary_kernel.py --matrix --shape 255x769x771 --shape 511x1025x1027 --dtype fp16 --kernel-variant auto --autotune-warmup 1 --autotune-repeat 2 --warmup 1 --repeat 3 --sustain-seconds 0.20 --sustain-op forward_backward --sustain-sync-every 2 --resource-interval 0.05 --min-resource-samples 2` passe avec `strict_extension_only=true`, `resource_samples_passed=true`, speedup forward/backward moyen `1.85x`, min `1.26x`, `gradInputCounts={"wmma_fp16_padded":328/138}`, `gradWeightCounts={"wmma_fp16_float_padded":328/138}`, GPU moyen `33.5%`.
- Integration LLM: le smoke strict v5 `tools\train_llm.py smoke --device cuda --require-cuda --steps 2` garde `precision=fp16`, `native_ternary_backend_counts={'extension': 2185}`, `native_ternary_requantize_backend_counts={'extension': 230}`, `native_ternary_grad_weight_backend_counts={'extension': 160}`, `native_ternary_grad_input_kernel_counts={'warp': 152, 'wmma_fp16': 8}`, `native_ternary_grad_weight_kernel_counts={'tiled': 152, 'wmma_fp16_float': 8}` et les audits architecture/phase passants. Les petites phases restent volontairement sur kernels hand-written parce que leurs dimensions sont sous le seuil WMMA utile.
- Statut: corrige pour les bords fp16 non multiples de 16 sans fallback dense. BF16 est traite par C20 et le forward grand format par C21. Restent a durcir: mesures energie/VRAM longues et preuve comparative longue.

### C20. BF16 encore hors WMMA Tensor Core

- Critique: apres C19, le chemin `bf16` etait fonctionnel et teste contre dense STE, mais les shapes d'entrainement bf16 tombaient encore sur les kernels warp/tiled. C'etait insuffisant pour un training mixed precision ambitieux: sur ce PC, bf16 doit utiliser le meme niveau de chemin Tensor Core que fp16 quand la shape le justifie, y compris les bords non multiples de 16.
- Correction: extension v6 ajoute `ternary_grad_input_wmma_bf16` et `ternary_grad_weight_bias_wmma_bf16_float`. Les kernels chargent `grad_out`, activations et poids ternaires decodes en shared memory `__nv_bfloat16`, appliquent le meme zero-padding interne que fp16, accumulent en fp32 via WMMA, puis ecrivent `grad_x` bf16 et `grad_weight`/`grad_bias` fp32. Les dispatchers choisissent bf16 WMMA des que `M/N/K >= 16`; les labels distinguent `wmma_bf16`, `wmma_bf16_padded`, `wmma_bf16_float` et `wmma_bf16_float_padded`.
- Verification courte: `test_bitlinear_native_extension_cuda_bf16_wmma_paths_match_dense_ste` compare fast STE extension contre dense STE sur `32x32x32` et `31x33x35`, exige les labels bf16 WMMA alignes/paddes et verifie `loss`, `grad_input`, `grad_weight` et `grad_bias`.
- Benchmark strict BF16 RTX 5070: `tools\benchmark_ternary_kernel.py --matrix --shape 256x768x768 --shape 255x769x771 --dtype bf16 --kernel-variant auto --autotune-warmup 1 --autotune-repeat 2 --warmup 1 --repeat 3 --sustain-seconds 0.20 --sustain-op forward_backward --sustain-sync-every 2 --resource-interval 0.05 --min-resource-samples 2` passe avec `strict_extension_only=true`, `resource_samples_passed=true`, speedup forward/backward moyen `1.39x`, min `1.20x`, `gradInputCounts={"wmma_bf16":300}` puis `{"wmma_bf16_padded":282}`, `gradWeightCounts={"wmma_bf16_float":300}` puis `{"wmma_bf16_float_padded":282}`, GPU moyen `9.1%` et puissance moyenne `36.07 W`.
- Integration LLM: le smoke strict `tools\train_llm.py smoke --device cuda --require-cuda --precision bf16 --steps 2` garde `native_ternary_backend_counts={'extension': 2401}`, `native_ternary_requantize_backend_counts={'extension': 276}`, `native_ternary_grad_weight_backend_counts={'extension': 160}`, `native_ternary_grad_input_kernel_counts={'warp': 152, 'wmma_bf16': 8}`, `native_ternary_grad_weight_kernel_counts={'tiled': 152, 'wmma_bf16_float': 8}` et les audits architecture/deliverable passants. Le proof global reste volontairement `passed=false` si la baseline courte a un score nul, ce qui evite de valider une victoire non informative.
- Statut: corrige pour bf16 aligne et edge sans fallback dense. Le forward grand format est traite par C21. Restent a durcir: occupation GPU sous vrais batchs, mesures VRAM/energie longues et preuve comparative longue Cortex > baseline sur corpus large.

### C21. Forward packed matmul grand format devenu goulet

- Critique: apres C20, le backward etait Tensor Core, mais le forward packe restait limite a `tiled_shared_memory_int2` et `warp_reduction_int2`. Le profil court a confirme le goulet sur fp16 `512x1024x1024`: `native_ms=0.9024 ms` contre `torch_unpack_linear_ms=0.2877 ms`, soit `native_vs_unpack=0.32x`. C'etait inacceptable pour le chemin par defaut: le training gagnait encore globalement grace au backward, mais le forward packed etait localement moins bon qu'une baseline dense unpack.
- Correction: extension v7 ajoute `ternary_matmul_wmma_fp16` et `ternary_matmul_wmma_bf16`. Ces kernels calculent `Y[M,N] = X[M,K] @ W[N,K]^T` directement depuis les codes int2 packes: `X` est charge en tuile WMMA row-major, `W` est decode depuis `packed_codes/scales/residual` en shared memory `KxN`, les bords sont zero-paddes, l'accumulation reste fp32 et le bias est ajoute avant le store fp16/bf16. Le variant `wmma` entre dans l'autotune `auto` avec `tiled` et `warp` quand dtype/shape/backend le justifient; les petits micro-circuits restent sur kernels hand-written.
- Verification courte: `test_bitlinear_native_extension_cuda_forward_wmma_matches_packed_runtime` compare le forward WMMA force contre `F.linear` sur le poids packed reconstruit en fp16 et bf16 edge. `test_native_ternary_auto_kernel_includes_wmma_for_large_cuda_shapes` exige les candidats `tiled/warp/wmma` et le label `wmma_tensor_core_int2` quand l'autotune le selectionne. Le paquet de 7 tests CUDA cibles passe en `1.673s`.
- Benchmark strict fp16 RTX 5070: avant v7, `512x1024x1024` avait `native_vs_unpack=0.32x`; apres v7, la matrice `256x768x768` + `512x1024x1024` passe avec `strict_extension_only=true`, `resource_samples_passed=true`, variant forward `wmma_tensor_core_int2`, `native_ms=0.1235/0.3329`, `native_vs_unpack=2.91x/1.21x`, `full_forward_ms=0.1258/0.3385`, speedup forward/backward moyen `1.46x`, min `1.21x`.
- Benchmark strict BF16 RTX 5070: la matrice `256x768x768` + `255x769x771` passe avec variant forward `wmma_tensor_core_int2`, `native_ms=0.0973/0.1369`, `native_vs_unpack=7.57x/3.38x`, `full_forward_ms=0.0879/0.1372`, speedup forward/backward moyen `1.65x`, min `1.42x`, `strict_extension_only=true`.
- Integration LLM: le smoke strict `tools\train_llm.py smoke --out-dir runs\llm-smoke-bf16-forward-wmma-v7 --device cuda --require-cuda --precision bf16 --steps 2` garde `native_ternary_backend_counts={'extension': 2191}`, `native_ternary_kernel_variants=['tiled_shared_memory_int2','warp_reduction_int2','wmma_tensor_core_int2']`, `native_ternary_requantize_backend_counts={'extension': 230}`, `native_ternary_grad_weight_backend_counts={'extension': 162}`, `native_ternary_grad_input_kernel_counts={'warp': 154, 'wmma_bf16': 8}`, `native_ternary_grad_weight_kernel_counts={'tiled': 154, 'wmma_bf16_float': 8}` et les audits architecture/deliverable passants.
- Limite: le speedup forward/backward global peut varier avec la baseline dense courte et le bruit GPU; l'amelioration locale du forward est nette, mais les mesures VRAM/energie longues et la preuve Cortex > baseline sur corpus large restent hors tests courts autorises.
- Statut: corrige pour le goulet forward packed grand format observe sans fallback dense. Le profil batch LLM court est traite par C22; restent a durcir: multi-shape plus large, multi-seed, durees plus longues et preuve comparative longue.

### C22. Profil batch LLM reel absent du gate court

- Critique: apres C21, les benchmarks kernel et smokes prouvaient le backend strict, mais ils ne donnaient pas un artefact unique de training LLM Cortex reel avec optimizer/backward/requantize/P1-P10, throughput wall-clock, CPU/GPU moyen, puissance, VRAM `nvidia-smi` et pic memoire CUDA torch. On pouvait donc encore confondre "kernel rapide" et "batch LLM complet mesure".
- Correction: ajout de `run_llm_batch_profile` et de la commande `tools/train_llm.py profile-batch`. Le profil prepare un corpus deterministe, tokenise, construit le manifeste, lance `LLMTrainer.train` avec Cortex complet, force `native_ternary_backend=extension` et `require_native_ternary_kernel=true` sur CUDA, puis ecrit `llm_batch_profile.json` avec `training_report`, `run_plan`, `throughput`, `resource_usage`, `torch_cuda_memory`, `kernel_evidence`, `architecture`, `hardware`, `failed_checks` et `passed`.
- Gate: le profil echoue si les samples ressource manquent, si les tokens planifies sont nuls, si l'audit architecture/deliverable echoue, si CUDA strict n'observe pas seulement le backend extension, si GPU util/memoire/puissance manquent sur CUDA, ou si le pic memoire CUDA torch reste nul alors que CUDA est requis.
- Verification courte CPU: `test_llm_batch_profile_writes_throughput_resource_and_architecture_report` verifie l'ecriture du JSON, le throughput positif, le monitor process, le rapport training, toutes les phases actives et le snapshot CUDA desactive en CPU.
- Verification courte CUDA RTX 5070: `tools\train_llm.py profile-batch --out-dir runs\llm-batch-profile-v1 --overwrite --device cuda --require-cuda --precision bf16 --steps 2 --batch-size 8 --gradient-accumulation-steps 1 --seq-len 32 --d-model 64 --n-heads 4 --n-layers 2 --resource-interval 0.05 --min-resource-samples 2` passe avec `passed=true`, `failed_checks=[]`, `native_ternary_backend_counts={"extension":1417}`, variants `tiled_shared_memory_int2/warp_reduction_int2/wmma_tensor_core_int2`, all phases active, 512 tokens train planifies, `117.646` tokens/s wall-clock, GPU moyen `10.344%`, GPU max `16%`, puissance moyenne `37.702 W`, VRAM moyenne `971.812 MB`, CPU process moyen `6.070%` du total et pic CUDA torch alloue `34,972,160` bytes.
- Limite: ce profil est volontairement court pour respecter l'interdiction de tests longs; il ne prouve pas encore saturation GPU, multi-seed, multi-shape large ni victoire Cortex > baseline sur corpus massif.
- Statut: corrige pour le trou "vrai batch LLM profile court". Restent a durcir: profils batchs plus grands/multi-shapes/multi-seeds quand autorises, puis preuve comparative longue.

### C23. Profil LLM batch unique insuffisant pour la robustesse shape/seed

- Critique: C22 donnait enfin un vrai profil batch LLM, mais un seul couple shape/seed pouvait encore masquer une regression liee a une longueur de sequence, un batch, une seed, une absence d'ancre P4 ou un choix autotune different. Ce n'etait pas assez fort pour dire que le chemin training strict tient hors d'un cas unique.
- Correction: ajout de `run_llm_batch_profile_matrix` et de la commande `tools/train_llm.py profile-matrix`. La matrice lance plusieurs profils enfants complets via `run_llm_batch_profile`, ecrit un JSON agregé `llm_batch_profile_matrix.json`, un CSV `llm_batch_profile_matrix.csv`, et echoue si un enfant echoue, si le nombre minimal de cas manque, si `--require-multi-shape` ou `--require-multi-seed` ne sont pas satisfaits, si une phase P1-P10 manque, ou si un run CUDA strict ne reste pas extension-only.
- Debug corrige pendant cette boucle: le premier test matrix CPU a revele que `run_llm_batch_profile` forcait `use_native_ternary_kernel=True` meme sur CPU, ce qui demandait un kernel CUDA impossible. Le profil utilise maintenant `use_native_ternary_kernel=require_native_ternary_kernel=_strict_native_ternary_required_for_training(training)`: CPU reste un profil de logique Cortex sans pretendre verifier le hardware CUDA, tandis que CUDA reste strict extension obligatoire. Le second echec venait de micro-batchs artificiels `batch=1` qui pouvaient observer une ancre sans policy learned-memory; le corpus de profil contient maintenant des identifiants exacts `C3-SAMPLE-*` et `C3-MARK-*` sur chaque ligne, et le test matrix utilise des micro-batchs réalistes.
- Verification courte CPU: `test_llm_batch_profile_matrix_requires_multiple_shapes_and_seeds` lance 2 shapes x 2 seeds, exige `min_cases=4`, `require_multi_shape=true`, `require_multi_seed=true`, verifie JSON, CSV, 4/4 cas passants, P1-P10 actifs et tokens planifies positifs.
- Verification courte CUDA RTX 5070: `tools\train_llm.py profile-matrix --out-dir runs\llm-batch-profile-matrix-v1 --overwrite --device cuda --require-cuda --precision bf16 --steps 1 --profile-shapes 32x64x4x2x4,40x64x4x2x4 --seeds 71,73 --min-cases 4 --require-multi-shape --require-multi-seed --resource-interval 0.05 --min-resource-samples 2 --corpus-repeats 128 --max-corpus-tokens 4096` passe avec `passed=true`, `failed_checks=[]`, `case_count=4`, `passed_cases=4`, `shape_count=2`, `seed_count=2`, `strict_extension_only_cases=4`, `all_phases_active_cases=4`, `total_planned_train_tokens=576`, throughput moyen `78.816` tokens/s, GPU moyen par cas `13.147%`, puissance moyenne `39.738 W`, VRAM moyenne `975.956 MB`. Les dispatchs extension enfants sont `1230/1288/1317/1200`.
- Limite: cette matrice reste volontairement courte; elle ne remplace pas une matrice longue de grandes tailles, ni une preuve baseline Cortex > NTP sur corpus massif.
- Statut: corrige pour le trou "un seul batch/seed". Restent a durcir: shapes plus grandes, plus de seeds, durees plus longues et comparaison baseline large.

### C24. Metriques ressource observees mais pas encore bloquantes

- Critique: C23 mesurait throughput, GPU, puissance et VRAM, mais un rapport pouvait encore passer avec une utilisation GPU ridicule tant que les phases et l'extension etaient actives. Pour l'objectif "voir si ca tourne a 10% de la puissance", il faut pouvoir transformer ces mesures en gates explicites, pas seulement les lire apres coup.
- Correction: `run_llm_batch_profile_matrix` accepte maintenant `min_train_tokens_per_second_mean`, `min_gpu_utilization_percent_mean`, `min_gpu_memory_used_mb_mean` et `min_gpu_power_draw_watts_mean`; la CLI expose `--min-train-tokens-per-second-mean`, `--min-gpu-utilization-percent-mean`, `--min-gpu-memory-used-mb-mean` et `--min-gpu-power-draw-watts-mean`. Le rapport agregé ecrit `summary.threshold_checks` avec `required`, `observed`, `passed`, et ajoute le nom du seuil a `failed_checks` quand il est strictement positif et non respecte.
- Verification courte CPU: `test_llm_batch_profile_matrix_resource_thresholds_are_blocking` impose `min_train_tokens_per_second_mean=1e12`, verifie `passed=false`, `failed_checks` contenant `min_train_tokens_per_second_mean`, et conserve l'observation reelle positive. Le test matrix multi-shape/multi-seed continue de passer quand les seuils par defaut sont a zero.
- Verification courte CUDA RTX 5070: `tools\train_llm.py profile-matrix --out-dir runs\llm-batch-profile-matrix-v2 --overwrite --device cuda --require-cuda --precision bf16 --steps 1 --profile-shapes 32x64x4x2x4,40x64x4x2x4 --seeds 71,73 --min-cases 4 --require-multi-shape --require-multi-seed --min-train-tokens-per-second-mean 10 --min-gpu-utilization-percent-mean 5 --min-gpu-memory-used-mb-mean 900 --min-gpu-power-draw-watts-mean 30 --resource-interval 0.05 --min-resource-samples 2 --corpus-repeats 128 --max-corpus-tokens 4096` passe avec `passed=true`, `failed_checks=[]`, 4/4 cas, `strict_extension_only_cases=4`, `all_phases_active_cases=4`, throughput moyen `77.310` tokens/s, GPU moyen `13.336%`, puissance moyenne `39.612 W`, VRAM moyenne `976.110 MB`, et tous les seuils `threshold_checks.*.passed=true`.
- Limite: les seuils prouvent que le rapport peut bloquer un run insuffisant, mais les valeurs actuelles restent basses et courtes. L'etape suivante doit augmenter vraiment les shapes ou ajouter une recherche automatique de batch/shape sous budget VRAM.
- Statut: corrige pour le manque de gates ressource. Restent a durcir: meilleurs seuils pour grands batchs, auto-sizing, et comparaison longue.

### C25. Choix manuel des shapes encore trop arbitraire

- Critique: C24 permettait de bloquer sur des seuils GPU/VRAM/throughput, mais il fallait encore choisir a la main les shapes. Cela favorisait les micro-shapes prudentes et pouvait laisser le GPU sous-rempli alors que la machine avait plus de marge. Le chemin devait proposer les plus grosses shapes Cortex qui rentrent dans un budget memoire, puis les verifier par un vrai training strict.
- Correction: ajout de `run_llm_batch_profile_autosize` et de la commande `tools/train_llm.py profile-autosize`. Le flux calcule un budget explicite (`--memory-budget-mb`) ou une fraction de la memoire CUDA libre (`--memory-budget-fraction`), genere une grille `seq_len x d_model x n_layers x batch_size`, estime chaque candidat avec `TransformerConfig` Cortex complet (`use_cortex_heads`, coeur ternaire, experts, Variable-In, learned memory, certificate head, kernel natif strict sur CUDA), filtre les candidats dont `estimated_peak_training_bytes` depasse le budget, score les candidats viables par charge estimee x tokens par step, selectionne les meilleurs, puis lance `profile-matrix` sur ces shapes.
- Gates: le rapport `llm_batch_profile_autosize.json` echoue si aucun candidat ne rentre, si `min_selected_shapes` n'est pas satisfait, ou si la matrice stricte enfant echoue. Il persiste le budget, la grille candidate, les candidats classes, les rejets, les shapes selectionnees, le rapport matrix et tous les `failed_checks`.
- Verification courte CPU: `test_llm_batch_profile_autosize_selects_budgeted_shape_and_runs_matrix` verifie selection sous budget, ecriture JSON, matrix passante et estimation inferieure au budget. `test_llm_batch_profile_autosize_blocks_when_budget_has_no_viable_shape` force un budget de `1 MB`, obtient `passed=false`, `no_viable_shapes`, `min_selected_shapes`, zero candidat viable et un rejet budgete.
- Verification courte CUDA RTX 5070: `tools\train_llm.py profile-autosize --out-dir runs\llm-profile-autosize-v1 --overwrite --device cuda --require-cuda --precision bf16 --steps 1 --candidate-seq-lens 32,48 --candidate-d-models 64,96 --candidate-n-layers 2 --candidate-batch-sizes 4,8 --n-heads 4 --selected-shape-count 2 --min-selected-shapes 2 --seeds 71 --memory-budget-fraction 0.10 --min-cases 2 --require-multi-shape --min-train-tokens-per-second-mean 10 --min-gpu-utilization-percent-mean 5 --min-gpu-memory-used-mb-mean 900 --min-gpu-power-draw-watts-mean 30 --resource-interval 0.05 --min-resource-samples 2 --corpus-repeats 128 --max-corpus-tokens 4096` passe avec `passed=true`, budget `1,157,103,616` bytes, `viable_candidate_count=8`, shapes choisies `seq48_d96_h4_l2_b8` et `seq32_d96_h4_l2_b8`, estimates `14.862 MB` et `13.593 MB`, 2/2 cas matrix passants, extension-only, all phases active, `640` tokens planifies, throughput moyen `125.247` tokens/s, GPU moyen `9.837%`, puissance moyenne `37.888 W`, VRAM moyenne `983.118 MB`.
- Limite: l'estimation de pic modele ne capture pas toute la VRAM de process observee par `nvidia-smi`. Depuis C26, cette limite ne pilote plus seule la selection finale; depuis C27, la VRAM observee max devient aussi un gate bloquant quand le signal GPU existe.
- Statut: corrige pour le choix manuel des shapes sous budget. La recherche mesuree courte est traitee par C26 et le budget VRAM observe par C27; restent a durcir: grandes shapes, plus de seeds, durees plus longues et preuve comparative longue.

### C26. Autosize encore pilote par estimation au lieu de mesures observees

- Critique: C25 filtrait correctement par budget VRAM estime, mais choisissait les shapes par score estime avant d'avoir vu le vrai throughput, le vrai GPU moyen, la vraie puissance ou les echecs eventuels d'un profil strict. Cela pouvait selectionner une shape "lourde sur papier" mais moins utile sur ce PC.
- Correction: `run_llm_batch_profile_autosize` accepte maintenant `measure_candidate_count` et `measured_selection_metric`, avec 4 candidats mesures par defaut. Quand cette option est positive, le flux prend les meilleurs candidats budgetes, lance un vrai `run_llm_batch_profile` court pour chacun sous `candidate_measurements/`, resume throughput, GPU, VRAM, puissance, pic CUDA torch, extension-only et all-phases-active, puis selectionne uniquement parmi les mesures passantes. La matrice finale `profile-matrix` reste obligatoire sur les shapes selectionnees, donc la mesure candidate ne remplace pas le gate strict.
- CLI: `tools/train_llm.py profile-autosize` utilise ce chemin mesure par defaut; `--measure-candidate-count N` ajuste le nombre de candidats et `--measured-selection-metric throughput_gpu` est le score par defaut. `throughput_gpu` score `tokens/s * max(gpu%, 1)` pour favoriser a la fois debit et occupation observee; `throughput` et `gpu` restent disponibles pour isoler un axe de debug.
- Gates: si la selection mesuree est demandee et ne produit pas assez de candidats passants, le rapport echoue avec `no_measured_viable_shapes` ou `min_selected_shapes`. Il ne complete pas silencieusement avec les candidats estimes.
- Verification courte CPU: `test_llm_batch_profile_autosize_can_select_from_measured_candidates` mesure 2 candidats, exige `selection_source="measured"`, au moins un candidat passant, un score mesure positif, un profil enfant persiste, puis une matrice stricte passante.
- Verification courte CUDA RTX 5070: `tools\train_llm.py profile-autosize --out-dir runs\llm-profile-autosize-measured-v1 --overwrite --device cuda --require-cuda --precision bf16 --steps 1 --candidate-seq-lens 32,48 --candidate-d-models 64,96 --candidate-n-layers 2 --candidate-batch-sizes 4,8 --n-heads 4 --selected-shape-count 2 --min-selected-shapes 2 --seeds 71 --memory-budget-fraction 0.10 --measure-candidate-count 4 --measured-selection-metric throughput_gpu --min-cases 2 --require-multi-shape --min-train-tokens-per-second-mean 10 --min-gpu-utilization-percent-mean 5 --min-gpu-memory-used-mb-mean 900 --min-gpu-power-draw-watts-mean 30 --resource-interval 0.05 --min-resource-samples 2 --corpus-repeats 128 --max-corpus-tokens 4096` passe avec `passed=true`, 4 candidats mesures, 4 passants, `selection_source=measured`, shapes choisies `seq32_d96_h4_l2_b8` et `seq48_d96_h4_l2_b4`, matrice 2/2 passante, throughput moyen `132.279` tokens/s, GPU moyen `13.082%`, puissance moyenne `40.813 W`, VRAM moyenne `987.571 MB`.
- Limite: c'est une recherche mesuree courte, pas encore une exploration longue de saturation GPU. Elle ferme le trou "selection estimee seulement"; C27 ferme ensuite le trou "budget observe non bloquant".
- Statut: corrige pour l'autosize mesure court et sans complement estime silencieux. C30 corrige ensuite le biais top-N estime; restent a durcir: plus de candidats, plus grosses shapes, plusieurs seeds mesurees, durees plus longues et comparaison Cortex vs baseline.

### C27. Budget VRAM observe non bloquant dans la selection mesuree

- Critique: C26 mesurait throughput/GPU/VRAM, mais `measurement_passed` ne tenait compte que du profil strict enfant. Une shape pouvait donc passer le profil, etre rapide, mais depasser la VRAM observee alors meme que l'utilisateur avait donne un budget. C'etait incoherent avec l'objectif d'utiliser le GPU intelligemment sans tomber dans des shapes qui explosent la marge reelle.
- Correction: `_profile_autosize_measurement_summary` expose maintenant `gpu_memory_used_mb_max`, `observed_gpu_memory_used_bytes`, `observed_gpu_memory_budget_fraction_used`, `measured_budget_enforced`, `measured_budget_passed`, `measurement_profile_passed` et `measurement_profile_failed_checks`. `measurement_passed` signifie maintenant: profil strict passe ET budget VRAM observe max respecte quand le signal GPU est disponible. La selection finale trie uniquement ces candidats eligibles.
- Gates: si tous les candidats mesures passent leur profil strict mais depassent le budget VRAM observe, le rapport echoue avec `no_measured_viable_shapes`, `min_selected_shapes` et `measured_budget_exceeded`; la matrice finale n'est pas lancee. Le rapport distingue `measured_profile_passed_candidate_count` de `measured_passed_candidate_count` pour rendre clair si le blocage vient de la qualite du profil ou du budget observe.
- Verification courte CPU/mocked: `test_llm_batch_profile_autosize_blocks_measured_vram_over_budget` simule un profil strict passant avec `gpu_memory_used_mb.max=1024` sous budget `512 MB`; le rapport refuse la selection, conserve `measurement_profile_passed=true`, marque `measured_budget_passed=false`, ajoute `observed_gpu_memory_budget` aux checks du candidat et ne lance pas la matrice.
- Verification courte CUDA RTX 5070: `tools\train_llm.py profile-autosize --out-dir runs\llm-profile-autosize-measured-budget-v1 --overwrite --device cuda --require-cuda --precision bf16 --steps 1 --candidate-seq-lens 32,48 --candidate-d-models 64,96 --candidate-n-layers 2 --candidate-batch-sizes 4,8 --n-heads 4 --selected-shape-count 2 --min-selected-shapes 2 --seeds 71 --memory-budget-fraction 0.10 --measure-candidate-count 4 --measured-selection-metric throughput_gpu --min-cases 2 --require-multi-shape --min-train-tokens-per-second-mean 10 --min-gpu-utilization-percent-mean 5 --min-gpu-memory-used-mb-mean 900 --min-gpu-power-draw-watts-mean 30 --resource-interval 0.05 --min-resource-samples 2 --corpus-repeats 128 --max-corpus-tokens 4096` passe avec `passed=true`, 4 candidats mesures, 4 profils passants, 4 budgets observes passants, fraction observee max `0.904`, shapes choisies `seq32_d96_h4_l2_b8` et `seq48_d96_h4_l2_b4`, matrice 2/2 passante, throughput moyen `140.654` tokens/s, GPU moyen `14.218%`, puissance moyenne `41.269 W`, VRAM moyenne `987.449 MB`.
- Limite: le budget observe depend de la granularite `nvidia-smi` et reste court. Il empeche les mauvais choix flagrants sous budget mensonger, mais il ne remplace pas un profil long de pic VRAM.
- Statut: corrige pour le gate VRAM observe dans l'autosize mesure. C28 traite ensuite la recherche d'effective batch via gradient accumulation. Restent a durcir: search plus large, pic VRAM sur fenetres plus longues, seeds multiples et lien direct vers benchmark baseline/Cortex.

### C28. Autosize ignorait l'effective batch via gradient accumulation

- Critique: apres C27, l'autosize choisissait seq_len, d_model, layers et micro-batch, mais gardait un `gradient_accumulation_steps` global. Cela limitait la capacite a augmenter les tokens par optimizer step sans augmenter la VRAM du micro-batch, et pouvait laisser le GPU moins bien amorti alors que `gradient_accumulation_steps=2` est deja supporte par le trainer.
- Correction: les shape specs acceptent maintenant un champ optionnel `gradient_accumulation_steps`, la cle de shape ajoute `_gN`, `profile-matrix` respecte le `g` propre a chaque shape au lieu du global, et `profile-autosize` genere une dimension `candidate_gradient_accumulation_steps`. Par defaut, autosize cherche le `g` demande et au moins `g=2`; la CLI expose `--candidate-gradient-accumulation-steps`.
- Gates: le rapport persiste `candidate_grid.candidate_gradient_accumulation_steps`, les `selected_shapes` contiennent leur `gradient_accumulation_steps`, les cases matrix indiquent le `g` execute et `matrix.config.shape_specific_gradient_accumulation_steps=true` quand une shape porte son propre `g`.
- Verification courte CPU/mocked: `test_llm_batch_profile_autosize_matrix_uses_selected_gradient_accumulation` force une grille `g=1,2`, verifie que la shape `g=2` est selectionnee, que le profil candidat et la matrice finale appellent `run_llm_batch_profile` avec `gradient_accumulation_steps=2`, et que les tokens planifies de la matrice correspondent au `g` choisi.
- Verification courte CUDA RTX 5070: `tools\train_llm.py profile-autosize --out-dir runs\llm-profile-autosize-gradaccum-v1 --overwrite --device cuda --require-cuda --precision bf16 --steps 1 --candidate-seq-lens 32,48 --candidate-d-models 64,96 --candidate-n-layers 2 --candidate-batch-sizes 4,8 --candidate-gradient-accumulation-steps 1,2 --n-heads 4 --selected-shape-count 2 --min-selected-shapes 2 --seeds 71 --memory-budget-fraction 0.10 --measure-candidate-count 4 --measured-selection-metric throughput_gpu --min-cases 2 --require-multi-shape --min-train-tokens-per-second-mean 10 --min-gpu-utilization-percent-mean 5 --min-gpu-memory-used-mb-mean 900 --min-gpu-power-draw-watts-mean 30 --resource-interval 0.05 --min-resource-samples 2 --corpus-repeats 128 --max-corpus-tokens 4096` passe avec `passed=true`, 16 candidats viables, 4 candidats mesures, 4 profils/budgets observes passants, shapes choisies `seq32_d96_h4_l2_b8_g2` et `seq48_d96_h4_l2_b8_g1`, matrice 2/2 passante, `896` tokens planifies, throughput moyen `283.958` tokens/s, GPU moyen `13.388%`, puissance moyenne `41.548 W`, VRAM moyenne `990.596 MB`, fraction budget observee max `0.906`.
- Limite: `g=2` augmente le travail par optimizer step mais ne remplace pas une vraie recherche sur plus de steps, plus de seeds et plus de shapes.
- Statut: corrige pour la recherche effective-batch courte. La selection multi-seed est traitee par C29; restent a durcir: profils plus longs de pic VRAM et connexion aux benchmarks baseline/Cortex.

### C29. Autosize mesurait chaque candidat sur une seule seed

- Critique: apres C28, `profile-autosize` pouvait lancer une matrice finale multi-seed, mais la selection mesuree des candidats utilisait seulement `normalized_seeds[0]`. Une seed chanceuse pouvait donc choisir une shape qui ne serait pas robuste sur les autres seeds, surtout quand le score combine throughput et occupation GPU observee.
- Correction: `run_llm_batch_profile_autosize` accepte maintenant `measure_candidate_seed_count`. Par defaut, il mesure chaque candidat sur toutes les seeds fournies; un run mono-seed garde donc le meme cout, tandis qu'un run multi-seed devient robuste sans option supplementaire. Chaque candidat mesure conserve les lignes `seed_measurements`, agrège score, throughput, GPU, VRAM max observee, puissance, pic CUDA torch, erreurs et checks par seed, puis n'est selectionnable que si toutes les seeds mesurees passent le profil strict et le budget VRAM observe.
- Gates: le rapport expose `measurement_seed_count`, `measurement_seeds`, `measured_candidate_profile_count`, `measured_profile_passed_profile_count` et `measured_passed_profile_count` dans `measurement` et `selection`. `measurement_passed` au niveau candidat signifie maintenant passage strict sur toutes les seeds mesurees; la matrice finale reste lancee ensuite sur les seeds demandees.
- Verification courte CPU/mocked: `test_llm_batch_profile_autosize_measures_each_candidate_across_requested_seeds` force `seeds=(11,13)`, `measure_candidate_count=1`, `measure_candidate_seed_count=2`, verifie deux profils candidats sous `candidate_measurements/.../seed_11` et `seed_13`, deux cas matrix, `require_multi_seed=true`, agrégation des deux `seed_measurements`, et selection mesuree passante.
- Verification courte CUDA RTX 5070: `tools\train_llm.py profile-autosize --out-dir runs\llm-profile-autosize-multiseed-v1 --overwrite --device cuda --require-cuda --precision bf16 --steps 1 --candidate-seq-lens 32 --candidate-d-models 64 --candidate-n-layers 2 --candidate-batch-sizes 4 --candidate-gradient-accumulation-steps 1,2 --n-heads 4 --selected-shape-count 1 --min-selected-shapes 1 --seeds 71,73 --memory-budget-fraction 0.10 --measure-candidate-count 1 --measure-candidate-seed-count 2 --measured-selection-metric throughput_gpu --min-cases 2 --require-multi-seed --min-train-tokens-per-second-mean 10 --min-gpu-utilization-percent-mean 5 --min-gpu-memory-used-mb-mean 900 --min-gpu-power-draw-watts-mean 30 --resource-interval 0.05 --min-resource-samples 2 --corpus-repeats 128 --max-corpus-tokens 4096` passe avec `passed=true`, candidat `seq32_d64_h4_l2_b4_g2`, 2 profils candidats mesures, 2 profils passants, matrice 2/2 passante, `all_phases_active_cases=2`, `strict_extension_only_cases=2`, throughput moyen `161.879` tokens/s, GPU moyen `13.202%`, puissance moyenne `40.732 W`, VRAM moyenne `978.536 MB`, score mesure moyen `1234.037`, fraction max de budget VRAM observee `0.888`.
- Limite: cette preuve reste volontairement courte et ne remplace pas une matrice longue de saturation; elle ferme toutefois le trou logique ou la selection mesuree etait mono-seed alors que le run final pouvait etre multi-seed.
- Statut: corrige pour la selection autosize multi-seed courte. C30 corrige ensuite le biais top-N estime; restent a durcir: plus de candidats, plus grosses shapes, plus de steps, suivi de pic VRAM plus long et preuve comparative baseline/Cortex a l'echelle.

### C30. Autosize mesurait seulement le top-N estime

- Critique: apres C29, les mesures etaient multi-seed, mais la liste mesuree restait `ranked_candidates[:N]`. Le ranking estime favorise les shapes lourdes selon le modele memoire/tokens; une shape plus petite, mieux alignee avec le kernel, moins bruyante ou plus efficace en throughput/GPU pouvait ne jamais etre mesuree si elle etait trop loin dans le rang estime. C'etait un biais important pour l'objectif "utiliser vraiment le GPU et accelerer sans enlever l'architecture".
- Correction: `profile-autosize` utilise maintenant `--measure-candidate-strategy diverse` par defaut. La strategie garde le meilleur candidat estime, puis choisit les candidats suivants par score de frontiere diversifiee sur `seq_len`, `d_model`, `n_layers`, `batch_size`, `gradient_accumulation_steps`, fraction budget et tokens par optimizer step. `--measure-candidate-strategy top` reste disponible comme diagnostic explicite, mais le chemin par defaut mesure hors top-N quand la grille contient des formes eloignees.
- Gates et observabilite: chaque candidat mesure porte `estimated_rank`, `measurement_candidate_index`, `measurement_selection_reason` et `measurement_selection_distance`. Le rapport `measurement` expose `candidate_selection_strategy`, `measurement_input_shape_keys` et `measurement_input_estimated_ranks`, ce qui rend visible si la recherche a reellement explore au-dela du top estime.
- Verification courte CPU/mocked: `test_llm_batch_profile_autosize_diverse_measurement_can_escape_top_estimate` force une grille de 4 candidats, `measure_candidate_count=2`, et donne le meilleur score observe au candidat `seq32_d32_h4_l1_b2_g1` de rang estime `4`. Le rapport mesure les rangs `1` et `4`, marque le second `diverse_shape_frontier`, puis selectionne ce candidat grace au score observe.
- Verification courte CUDA RTX 5070: `tools\train_llm.py profile-autosize --out-dir runs\llm-profile-autosize-diverse-v1 --overwrite --device cuda --require-cuda --precision bf16 --steps 1 --candidate-seq-lens 32,64 --candidate-d-models 64,96 --candidate-n-layers 2 --candidate-batch-sizes 4 --candidate-gradient-accumulation-steps 1,2 --n-heads 4 --selected-shape-count 1 --min-selected-shapes 1 --seeds 71 --memory-budget-fraction 0.10 --measure-candidate-count 2 --measure-candidate-strategy diverse --measured-selection-metric throughput_gpu --min-cases 1 --min-train-tokens-per-second-mean 10 --min-gpu-utilization-percent-mean 5 --min-gpu-memory-used-mb-mean 900 --min-gpu-power-draw-watts-mean 30 --resource-interval 0.05 --min-resource-samples 2 --corpus-repeats 128 --max-corpus-tokens 4096` passe avec `passed=true`, rangs mesures `1,8`, inputs `seq64_d96_h4_l2_b4_g2` et `seq32_d64_h4_l2_b4_g1`, selection finale `seq32_d64_h4_l2_b4_g1`, 2 profils candidats passants, matrice 1/1 passante, `all_phases_active_cases=1`, `strict_extension_only_cases=1`, throughput moyen `93.958` tokens/s, GPU moyen `15.182%`, puissance moyenne `39.636 W`, VRAM moyenne `978.182 MB`, score mesure selectionne `1330.507`.
- Limite: la strategie diversifiee reste une recherche courte; elle ne remplace pas une recherche plus large avec plus de candidats, plus de seeds et des fenetres de mesure plus longues. Elle ferme toutefois le biais ou le meilleur candidat observe pouvait etre invisible parce que le top-N estime etait trop etroit.
- Statut: corrige pour le biais top-N dans l'autosize mesure par defaut. C31 ajoute ensuite un raffinement adaptatif a budget de mesure constant; restent a durcir: search plus large, mesures de pic VRAM plus longues, seuils GPU plus ambitieux et preuve comparative baseline/Cortex a l'echelle.

### C31. Mesure autosize non adaptative apres les premiers resultats observes

- Critique: C30 explorait hors top-N estime, mais la liste de candidats etait encore choisie en une seule fois avant toute mesure. Si la premiere vague montrait qu'une zone de la grille etait beaucoup plus efficace sur le GPU local, le harness ne pouvait pas depenser le reste du budget de mesure autour de cette zone; il continuait a mesurer une liste predecidee. C'etait moins ambitieux que necessaire pour optimiser la vitesse sans retirer de briques Cortex.
- Correction: `profile-autosize` accepte maintenant `--measure-candidate-adaptive-rounds` avec defaut `2`. Quand la strategie `diverse` et `measure_candidate_count > 2` sont actifs, le nombre total de candidats mesures reste borne par `measure_candidate_count`: une premiere vague mesure une frontiere diversifiee, puis les vagues suivantes choisissent des candidats adaptatifs proches des meilleurs scores observes tout en gardant de la nouveaute et un score estime raisonnable. `--measure-candidate-adaptive-rounds 1` garde le comportement une seule vague pour diagnostic.
- Gates et observabilite: le rapport expose `adaptive_rounds_requested`, `adaptive_rounds_used` et `measurement_rounds`. Chaque round contient `round_kind`, `candidate_count`, `shape_keys`, `estimated_ranks`, `selection_reasons` et `source_shape_keys`; les candidats adaptatifs portent `measurement_selection_reason="adaptive_measured_frontier"` et `measurement_selection_source_shape_key`.
- Verification courte CPU/mocked: `test_llm_batch_profile_autosize_adaptive_round_refines_from_observed_winner` force une grille de 4 candidats et `measure_candidate_count=4`. La premiere vague mesure 2 candidats, la deuxieme vague mesure 2 candidats adaptatifs, puis la selection finale choisit `seq32_d64_h4_l1_b2_g1`, candidat issu du round `adaptive_measured_frontier`, avec une source shape mesuree.
- Verification courte CUDA RTX 5070: `tools\train_llm.py profile-autosize --out-dir runs\llm-profile-autosize-adaptive-v1 --overwrite --device cuda --require-cuda --precision bf16 --steps 1 --candidate-seq-lens 32,64 --candidate-d-models 64,96 --candidate-n-layers 2 --candidate-batch-sizes 4 --candidate-gradient-accumulation-steps 1,2 --n-heads 4 --selected-shape-count 1 --min-selected-shapes 1 --seeds 71 --memory-budget-fraction 0.10 --measure-candidate-count 4 --measure-candidate-strategy diverse --measure-candidate-adaptive-rounds 2 --measured-selection-metric throughput_gpu --min-cases 1 --min-train-tokens-per-second-mean 10 --min-gpu-utilization-percent-mean 5 --min-gpu-memory-used-mb-mean 900 --min-gpu-power-draw-watts-mean 30 --resource-interval 0.05 --min-resource-samples 2 --corpus-repeats 128 --max-corpus-tokens 4096` passe avec `passed=true`, rounds `initial_diverse` puis `adaptive_measured_frontier`, rangs mesures `[1,8]` puis `[2,5]`, selection finale `seq64_d64_h4_l2_b4_g2`, 4 profils candidats passants, matrice 1/1 passante, `all_phases_active_cases=1`, `strict_extension_only_cases=1`, throughput moyen `323.963` tokens/s, GPU moyen `13.000%`, puissance moyenne `38.969 W`, VRAM moyenne `983.000 MB`, score mesure selectionne `3486.477`.
- Limite: l'adaptation reste courte et locale; elle ne remplace pas une vraie recherche plus large avec davantage de rounds, de seeds et de temps par profil. Elle ferme toutefois le trou ou la recherche ne pouvait pas reagir aux mesures deja observees.
- Statut: corrige pour le raffinement adaptatif a budget de profils constant. Restent a durcir: search large multi-round, plus longues fenetres GPU/VRAM, seuils d'occupation plus ambitieux et comparaison baseline/Cortex a l'echelle.

### C32. Les rounds adaptatifs > 2 pouvaient consommer tout le budget restant

- Critique: apres C31, `--measure-candidate-adaptive-rounds 2` etait correct, mais une valeur plus ambitieuse pouvait encore lancer une deuxieme vague avec tout le budget restant. Cela rendait `adaptive_rounds=3+` moins reel que son nom: le rapport pouvait annoncer une capacite multi-round alors que l'optimiseur n'observait pas assez de feedback intermediaire.
- Correction: chaque vague adaptative calcule maintenant `remaining_rounds` et ne demande que `ceil(remaining_count / remaining_rounds)` candidats. Le budget total reste borne par `measure_candidate_count`, mais il est etale sur les rounds restants pour permettre un vrai feedback mesure -> selection -> nouvelle mesure.
- Verification courte CPU/mocked: `test_llm_batch_profile_autosize_adaptive_rounds_spread_remaining_budget` force 6 candidats, `measure_candidate_count=6` et `measure_candidate_adaptive_rounds=3`; le rapport doit produire les rounds `initial_diverse`, `adaptive_measured_frontier`, `adaptive_measured_frontier` avec les comptes `(2, 2, 2)`, 6 candidats uniques mesures et une matrice finale.
- Statut: corrige pour la semantique multi-round courte. Restent a durcir: heuristique de repartition plus intelligente selon incertitude/variance, plus de seeds par candidat et runs CUDA plus longs.

### C33. La selection mesuree utilisait une moyenne fragile entre seeds

- Critique: apres C32, les candidats pouvaient etre mesures sur plusieurs seeds, mais le score agregé etait encore la moyenne brute de `throughput`, `gpu` ou `throughput_gpu`. Une shape avec une tres bonne seed et une seed tres mauvaise pouvait battre une shape plus stable, ce qui est dangereux pour lancer un vrai pre-entrainement long: on optimise alors un pic ponctuel au lieu d'un regime stable.
- Correction: l'agregateur calcule maintenant `measured_score_mean`, `measured_score_min`, `measured_score_max`, `measured_score_stddev`, `measured_score_lower_confidence` et `measured_score_stability_ratio`. Le champ `measured_score` utilise la borne prudente `mean - stddev` bornee a zero, donc la selection par defaut et les rounds adaptatifs utilisent un score robuste tout en gardant la moyenne brute dans le rapport.
- Verification courte CPU/mocked: `test_llm_batch_profile_autosize_uses_risk_adjusted_measured_score` force deux candidats mesures sur deux seeds: `seq64` a une moyenne brute plus haute mais une variance enorme, `seq32` est stable. Le rapport doit selectionner `seq32`, exposer un `measured_score_mean` plus haut pour le candidat instable, mais un `measured_score` robuste plus bas.
- Statut: corrige pour la selection multi-seed robuste courte. Restent a durcir: variance sur plus de seeds, fenetres plus longues, intervalle de confiance base sur repetitions runtime et integration dans une recherche bayesienne plus ambitieuse.

### C34. Le score robuste retombait a une moyenne brute avec une seule seed fournie

- Critique: C33 rendait la selection multi-seed plus prudente, mais le chemin CLI par defaut utilisait souvent `--seeds 71`. Avec une seule seed mesuree, `stddev=0`, donc `measured_score_lower_confidence == measured_score_mean`: la robustesse etait mathematiquement neutralisee au moment le plus courant.
- Correction: `profile-autosize` a maintenant `--min-measure-candidate-seed-count 2` par defaut. Si moins de seeds sont fournies, les mesures candidates ajoutent des seeds deterministes uniques sans modifier les seeds de la matrice finale. `--measure-candidate-seed-count 1` reste disponible comme diagnostic explicite court.
- Gates et observabilite: le rapport expose `provided_seed_count`, `min_measurement_seed_count`, `requested_seed_count`, `measurement_seed_count`, `measurement_seeds` et `synthesized_measurement_seed_count`. Les candidats agreges conservent leurs `seed_measurements`, donc on peut auditer exactement quelles seeds ont produit la moyenne, la variance et la borne prudente.
- Verification courte CPU/mocked: `test_llm_batch_profile_autosize_synthesizes_minimum_measurement_seed` donne seulement `seeds=(11,)`; le harness mesure le candidat sur `(11, 104740)`, reporte une seed synthetique et lance la matrice finale uniquement sur `11`.
- Verification courte CUDA RTX 5070: `tools\train_llm.py profile-autosize --out-dir .codex\tmp\llm-profile-autosize-minseed-current --overwrite --device cuda --require-cuda --precision bf16 --steps 1 --candidate-seq-lens 32,64 --candidate-d-models 64,96 --candidate-n-layers 2 --candidate-batch-sizes 4 --candidate-gradient-accumulation-steps 1,2 --n-heads 4 --selected-shape-count 1 --min-selected-shapes 1 --seeds 71 --memory-budget-fraction 0.10 --measure-candidate-count 4 --measure-candidate-strategy diverse --measure-candidate-adaptive-rounds 2 --measured-selection-metric throughput_gpu --min-cases 1 --min-train-tokens-per-second-mean 10 --min-gpu-utilization-percent-mean 5 --min-gpu-memory-used-mb-mean 900 --min-gpu-power-draw-watts-mean 30 --resource-interval 0.05 --min-resource-samples 2 --corpus-repeats 128 --max-corpus-tokens 4096` passe avec `provided_seed_count=1`, `measurement_seeds=[71,104800]`, `synthesized_measurement_seed_count=1`, 8 profils candidats mesures, selection `seq64_d64_h4_l2_b4_g2`, score robuste selectionne `3970.683`, moyenne `4376.482`, stddev `405.799`, stabilite `0.907`, matrice 1/1 passante, `strict_extension_only_cases=1`, `all_phases_active_cases=1`, throughput moyen `285.624` tokens/s, GPU moyen `13.143%`, puissance moyenne `41.534 W`, VRAM moyenne `982.857 MB`.
- Statut: corrige pour le chemin par defaut single-seed. Restent a durcir: repetitions runtime par seed, allocation adaptative selon incertitude et vraie recherche plus large quand les tests longs seront autorises.

### C35. Les rounds adaptatifs exploitaient seulement la borne basse prudente

- Critique: apres C34, la selection finale etait plus robuste, mais le choix des candidats adaptatifs reutilisait la meme borne basse `mean - stddev`. Cela protege la selection finale, mais c'est trop conservateur pour l'exploration: une zone avec moyenne haute et variance forte pouvait etre ignoree alors qu'elle merite justement une mesure supplementaire pour savoir si elle est un vrai gain GPU ou seulement du bruit.
- Correction: l'agregateur expose maintenant `measured_score_upper_confidence = mean + stddev`. `_profile_autosize_adaptive_measurement_inputs` trie ses sources par cette borne haute puis score les candidats proches avec une composante `source_potential`, tout en conservant nouveaute, proximite, score estime et tokens par step. Les candidats adaptatifs conservent `measurement_selection_source_score`, `measurement_selection_source_score_mean`, `measurement_selection_source_score_stddev`, `measurement_selection_source_upper_confidence` et `measurement_selection_source_stability_ratio`.
- Gates et observabilite: chaque round expose maintenant aussi `source_scores`, `source_score_means`, `source_score_stddevs`, `source_upper_confidences` et `source_stability_ratios`, donc une decision adaptive peut etre relue sans ouvrir les profils individuels.
- Verification courte CPU/mocked: `test_llm_batch_profile_autosize_adaptive_frontier_uses_uncertainty_potential` force une source stable avec borne basse haute mais borne haute faible, et une source incertaine avec borne basse faible mais `mean + stddev` tres haut; le prochain candidat choisi est `near_uncertain`, lie a `uncertain_source`, avec les champs de source attendus.
- Verification courte CUDA RTX 5070: `tools\train_llm.py profile-autosize --out-dir .codex\tmp\llm-profile-autosize-ucb-current --overwrite --device cuda --require-cuda --precision bf16 --steps 1 --candidate-seq-lens 32,64 --candidate-d-models 64,96 --candidate-n-layers 2 --candidate-batch-sizes 4 --candidate-gradient-accumulation-steps 1,2 --n-heads 4 --selected-shape-count 1 --min-selected-shapes 1 --seeds 71 --memory-budget-fraction 0.10 --measure-candidate-count 4 --measure-candidate-strategy diverse --measure-candidate-adaptive-rounds 2 --measured-selection-metric throughput_gpu --min-cases 1 --min-train-tokens-per-second-mean 10 --min-gpu-utilization-percent-mean 5 --min-gpu-memory-used-mb-mean 900 --min-gpu-power-draw-watts-mean 30 --resource-interval 0.05 --min-resource-samples 2 --corpus-repeats 128 --max-corpus-tokens 4096` passe avec `measurement_seeds=[71,104800]`, 8 profils candidats, sources adaptatives `seq64_d96_h4_l2_b4_g2`, `source_upper_confidence=4247.725`, selection finale prudente `seq64_d64_h4_l2_b4_g2`, score robuste `3906.117`, moyenne `4648.800`, stddev `742.683`, upper `5391.483`, matrice 1/1 passante, `strict_extension_only_cases=1`, `all_phases_active_cases=1`, throughput moyen `296.592` tokens/s, GPU moyen `15.071%`, puissance moyenne `41.461 W`, VRAM moyenne `982.857 MB`.
- Statut: corrige pour l'exploration adaptative incertaine courte. Restent a durcir: repeated runtime samples par seed, modele bayesien explicite et budget adaptatif qui peut augmenter les seeds ou les steps sur les zones incertaines.

## Critique phase par phase - boucle 29

### P1 - Verifier OS

- Ce qui est solide: familles arithmetic, algebra, long_context_anchor, entity_tracking, instruction_following, code_unit_tests, calibration; oracles stricts; metamorphic/anti-metamorphic; fault matrix; cout verifier par cas.
- Preuve actuelle: tests P1 et full Cortex court activent P1; rapports de cycle persistent les resultats.
- Faiblesse: generateurs encore limites, peu de domaines held-out, pas assez de bruit naturel, pas assez de faux positifs/faux negatifs hors familles internes.
- Risque architectural: si P1 est trop petit, P6/P7/P10 optimisent contre un monde trop facile.
- Correction prioritaire restante: elargir les generateurs et ajouter un audit de couverture d'oracles par domaine.

### P2 - Ternary Core

- Ce qui est solide: poids ternaires packes int2, quantization activations, STE, sync versionnee des buffers packes pendant training, kernels CUDA natifs tuiles/warp, forward WMMA fp16/bf16 decode-shared depuis int2 packe, autotune CUDA-event par shape avec candidats `tiled/warp/wmma`, profil JSON persistant, cache layer-local, fast STE autograd forward, backward CUDA `grad_input` depuis poids int2 packes, WMMA fp16/bf16 `grad_input` aligne ou padde, WMMA fp16/bf16->fp32 `grad_weight` + `grad_bias` aligne ou padde, requantization/packing post-update fusionnee CUDA, backend extension C++/CUDA strict par defaut, doctor CUDA distinguant RawKernel et extension runtime, audit LLM exigeant native forward/requantize/grad_weight exclusivement extension en training CUDA strict.
- Preuve actuelle: tests CUDA courts, export/import profil, tests gradients fast-vs-dense STE en fp32/fp16/bf16, test de parite requantize/pack fp32/fp16/bf16, tests WMMA forward/grad-input/grad-weight fp16/bf16 alignes et edge, test extension forcee, doctor toolchain strict, benchmark RTX 5070 avec full forward/backward et requantize/pack profile, matrice strict extension 3 shapes avec monitoring soutenu GPU/CPU/power, matrice LLM-shape WMMA fp16 `256x768x768` + `512x1024x1024`, matrice edge WMMA fp16 paddee `255x769x771` + `511x1025x1027`, matrice BF16 WMMA `256x768x768` + `255x769x771`, matrices forward-WMMA v7 fp16/bf16, smoke LLM fp16/bf16 extension avec dispatches forward/requantize/grad_weight extension et compteurs WMMA `grad_input`/`grad_weight` positifs, profil batch LLM Cortex bf16 strict avec optimizer/backward/P1-P10/monitoring et `passed=true`, matrice profile LLM bf16 courte 2 shapes x 2 seeds avec 4/4 cas passants, extension-only et all phases active, seuils bloquants throughput/GPU/VRAM/puissance verifies sur CPU et CUDA, auto-sizing budgete avec matrix stricte executee sur les shapes selectionnees, autosize mesure qui profile 4 candidats CUDA et selectionne les 2 shapes par score observe avant la matrice finale, gate de VRAM observee max avant selection finale, autosize `gradient_accumulation_steps` qui choisit et execute `g=2` quand la mesure le justifie, selection robuste multi-seed qui expose moyenne/min/max/stddev/lower-confidence et ne choisit plus un pic instable, mesure candidate minimum 2 seeds par defaut meme quand la matrice finale reste mono-seed, puis exploration adaptive par borne haute `mean + stddev` avec champs de source persistés.
- Faiblesse: pas encore de mesure energie/VRAM longue ni matrice de grandes shapes LLM; GPU moyen reste faible sur ces shapes courtes, meme si ce faible niveau peut maintenant etre gate, mesure, budgete en VRAM observee, augmente via effective batch et utilise pour choisir les shapes; le proof global court reste volontairement bloque par `baseline_score_passed` quand la baseline a un score nul.
- Risque architectural: le chemin training est maintenant completement branché en extension pour forward, `grad_input`, `grad_weight`, `grad_bias` et repack, mais une preuve de paradigme demandera que ce gain survive aux vrais batchs LLM et ne degrade pas la convergence.
- Correction prioritaire restante: elargir la matrice profile LLM a des tailles plus grandes, davantage de graines et des durees plus longues quand autorise, puis relier cette recherche mesuree au comparatif baseline/Cortex.

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

- Statut: le forward lit les codes packes int2 et lance CUDA extension sur GPU avec autotune `tiled/warp/wmma`; le backward CUDA calcule `grad_input` depuis les codes int2 packes avec WMMA fp16/bf16 quand la shape est alignee ou paddee; `grad_weight` et `grad_bias` passent par WMMA fp16/bf16->fp32 quand la shape est alignee ou paddee; la resynchronisation post-update requantize et repack directement en CUDA extension.
- Faiblesse: une matrice batch LLM courte existe avec VRAM/puissance/throughput sur plusieurs shapes et seeds, mais pas encore de profil long ni grandes shapes; les mesures GPU courtes restent sensibles au bruit et ne prouvent pas encore l'efficacite sur tous vrais batchs LLM.
- Correction restante: elargir les vrais batchs LLM en VRAM/energie/throughput sur grandes shapes et plus de seeds, puis lier ces profils aux preuves baseline/Cortex longues.

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

## File de correction priorisee apres boucle 19

1. P2: elargir la recherche autosize mesuree/diversifiee/adaptative a plus de candidats, plus de seeds et des profils plus longs sous budget VRAM observe.
2. P2: elargir la matrice profile LLM stricte a des shapes plus grandes et plus longues, en gardant toutes les briques Cortex actives.
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
