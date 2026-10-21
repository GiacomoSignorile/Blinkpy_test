# Research notes (Haiku sub-agent reports, 2026-10-21)

Unverified: gains are mostly the agents' estimates; see caveats inside.

## Pseudo-labels
# Improving pseudo-labels in the low-label regime (people + cats crops)

Scope: frozen DINOv2/CLIP-L features, ~10 person classes, few labels per class, thousands of unlabelled crops, GPU is a GTX 1050 (4 GB), so full CLIP-L fine-tuning is the expensive step. Expected gains are rough estimates from general experience unless a source is cited. I did not verify numbers for every item. Searches covered FreeMatch, SoftMatch, USB/SimMatch/FlexMatch, label propagation, active learning (ProbCover/TypiClust), conformal pseudo-labelling and TransCLIP. Not searched: DivideMix, co-teaching, video tracking for label propagation, noisy-student details, class-adaptive thresholds beyond FreeMatch.

## Techniques

### 1. Label propagation / kNN graph on frozen features (transductive)
- Summary: build a kNN graph over all crops (labelled + unlabelled) from the frozen embeddings, propagate labels, then use the propagated labels as pseudo-labels. Works best when labels are scarce.
- Expected gain: high in the few-label regime, often several points of accuracy over a linear probe. Estimate.
- Effort: 2-4 h. sklearn `LabelSpreading(kernel="knn")`, or a few lines of numpy/scipy sparse matrix propagation.
- Link: Iscen et al., "Label Propagation for Deep Semi-Supervised Learning", CVPR 2019, https://openaccess.thecvf.com/content_CVPR_2019/html/Iscen_Label_Propagation_for_Deep_Semi-Supervised_Learning_CVPR_2019_paper.html

### 2. Transductive CLIP refinement (TransCLIP)
- Summary: a plug-and-play transductive layer on top of CLIP (zero- or few-shot) that optimises the joint predictions over the unlabelled set, with a KL term that keeps predictions tied to the text encoder. Plugs onto any inductive few-shot model.
- Expected gain: reported as consistent improvements over inductive and vision-only transductive baselines (NeurIPS 2024). Applies to CLIP, not DINOv2 directly.
- Effort: 3-6 h. The ideas can be reimplemented in PyTorch on your cached features. There is no need for the full pipeline.
- Link: Zanella et al., "Boosting Vision-Language Models with Transduction", NeurIPS 2024, https://proceedings.neurips.cc/paper_files/paper/2024/hash/71d7dbe2652bd4662d29fa269f059db4-Abstract.html (arXiv: https://arxiv.org/html/2406.01837v1). Follow-up: LIMO (2025), extends it with info-max and light fine-tuning.

### 3. Adaptive / soft confidence thresholds (FreeMatch, SoftMatch)
- Summary: replace the fixed FixMatch threshold with a threshold that tracks the model's learning status (FreeMatch, ICLR 2023) or a soft weight that down-weights low-confidence samples with a truncated Gaussian instead of hard cut-off (SoftMatch, ICLR 2023). Both add class-fairness or uniform-alignment terms against collapse onto frequent classes.
- Expected gain: FreeMatch reports 1.3-13.6% error reduction over FlexMatch on CIFAR-10/STL-10/ImageNet, depending on label count. Gains for a frozen-feature linear head will be smaller, since the threshold matters most in end-to-end training. Estimate.
- Effort: 2-3 h to add to a training loop with a linear/MLP head. Code exists in Microsoft's Semi-supervised-learning repo (includes FreeMatch, SoftMatch, FlexMatch, SimMatch).
- Links: FreeMatch https://arxiv.org/pdf/2205.07246 ; SoftMatch https://arxiv.org/pdf/2301.10921 ; code https://github.com/microsoft/Semi-supervised-learning

### 4. SimMatch / similarity-based consistency
- Summary: labelled and unlabelled instances are matched through a similarity graph, so pseudo-labels for each unlabelled sample are informed by labelled neighbours (CVPR 2022).
- Expected gain: in USB's ImageNet check (MAE ViT-B, not your backbone), SimMatch ranked first at 1 and 10 labels per class. Your backbone differs, so treat this as directional.
- Effort: 4-8 h if reimplemented, lower if you use the USB/Microsoft codebase.
- Links: https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_SimMatch_Semi-Supervised_Learning_With_Similarity_Matching_CVPR_2022_paper.html ; USB benchmark https://arxiv.org/pdf/2208.07204

### 5. Conformal prediction for pseudo-label sets / threshold calibration
- Summary: instead of a fixed confidence cut-off, pick a threshold that guarantees a target error rate on held-out labels. Use the conformal set (sets with more than one label) to avoid wrong hard labels, and abstain on the rest.
- Expected gain: mainly reduces the wrong-label rate for a chosen coverage. The main value for you is a rigorous version of your simulation-based threshold. Estimate.
- Effort: 1-2 h. Pure numpy: score = 1 - p(true class) on a calibration set, take the (1-alpha) quantile, and the set is {k : p_k >= 1 - q}.
- Links: Noise-aware conformal prediction (arXiv 2501.12749, Jan 2025), https://arxiv.org/abs/2501.12749v1 ; Source-free CP with pseudo-labels (PMLR v266, 2025), https://proceedings.mlr.press/v266/angelman25a.html ; Conformal credal SSL (PMLR 204, 2023), https://cs.paperswithcode.com/paper/conformal-credal-self-supervised-learning
- Temperature scaling: https://arxiv.org/pdf/2301.07294 (2023) applies temperature scaling plus entropy thresholding to self-training teachers. Effort 30 min with sklearn/scipy.

### 6. Co-training with several backbones (DINOv2 + CLIP)
- Summary: train one head per backbone, pseudo-label each other's confident predictions, and keep only agreement or high-confidence cases. Cheap when features are cached.
- Expected gain: usually a robust few-point gain, since the backbones make uncorrelated errors. Estimate.
- Effort: 1-2 h with cached features (concatenate or agree-on-both rule).

### 7. Small-loss / noise-robust training (co-teaching, DivideMix)
- Summary: when a pseudo-labelled batch is noisy, train on the lowest-loss samples only (co-teaching), or model the loss distribution to split clean/noisy samples (DivideMix). Only useful when you train a head end-to-end on the noisy labels.
- Expected gain: meaningful when pseudo-label noise is more than about 20%. Estimate, and not searched in this session.
- Effort: 3-5 h in PyTorch. Keep to a loss mask on a linear or MLP head.

### 8. Prototype / cluster-based refinement
- Summary: cluster the features (k-means or GMM), assign each cluster the majority labelled class, and use cluster prototypes as extra pseudo-labels or consistency targets.
- Expected gain: useful when some classes are compact and others are split. Often complements label propagation. Estimate.
- Effort: 1-2 h with sklearn.

### 9. Temporal consistency / tracking for propagating labels (video clips)
- Summary: crops from the same clip, or from a tracker across frames, form a short chain. Propagate a confident label along the chain, and majority-vote labels within a track.
- Expected gain: potentially large because it is free supervision. Usually one of the best signals when you have tracks. Depends on tracker quality. Not searched in this session.
- Effort: 2-4 h (track IDs from your existing watcher, then majority vote). Check whether your watcher already stores track or clip IDs.

### 10. Active learning for which crops to label
- Summary: choose samples that maximise coverage of the unlabelled feature space (ProbCover, TypiClust) rather than the most uncertain ones, which fails at low budgets.
- Expected gain: large at very low budgets. The TypiClust/ProbCover thesis reports 39.4% improvement over random selection on CIFAR-10 with 10 labels, using semi-supervised methods. Your setting differs.
- Effort: 2-3 h with sklearn `NearestNeighbors`. Greedy coverage over cached features.
- Link: Yehuda et al., "Active Learning Through a Covering Lens", NeurIPS 2022, https://proceedings.neurips.cc/paper_files/paper/2022/hash/8c64bc3f7796d31caa7c3e6b969bf7da-Abstract.html ; code https://github.com/avihu111/TypiClust

### 11. Noisy student / teacher-student distillation
- Summary: train a student on pseudo-labels with strong augmentation and noise, then iterate. Standard in large-scale SSL.
- Expected gain: solid, but needs more compute than you have. Lower priority on the GTX 1050. Background knowledge; not searched.
- Effort: 4-8 h.

## Ranked top-5 to try first (small GPU, cached features)

1. Label propagation on the kNN graph of cached DINOv2 + CLIP features (technique 1). Fast, no training, big gain in the few-label regime. 2-4 h.
2. Conformal or temperature-calibrated thresholds on your held-out human labels (technique 5). Formalises your existing simulation and makes the wrong-label rate explicit. 1-2 h.
3. Adaptive / soft threshold with class-fairness on a linear head (technique 3, FreeMatch/SoftMatch). Replaces fixed thresholds without touching the backbone. 2-3 h.
4. Temporal track or clip-level majority voting, if your watcher already stores track or clip IDs (technique 9). Likely the largest free gain on video data. 2-4 h.
5. Coverage-based active selection for which crops to label (technique 10). Spends your human labelling budget better. 2-3 h.

Defer: CLIP-L fine-tuning and noisy-student. The 4 GB GTX 1050 makes full CLIP-L training slow or impossible without small batches, mixed precision and frozen lower layers. Get the cheap items above working first.

## Gaps
- Not verified: exact numbers for DivideMix, co-teaching, noisy-student, and the temporal-tracking methods.
- No 2024-2026 frozen-foundation-feature pseudo-labelling benchmark was found. The search results were mostly older work.

## Fine-tuning
# Fine-tuning for small-data fine-grained / re-ID (household person + cat ID)

Scope: 10 household-member classes, small labelled set, low-res / occluded / night-IR crops. Current: CLIP ViT-L end-to-end FT (backbone lr 1e-5, head 1e-3, label smoothing, class-weighted CE, OneCycle); frozen DINOv2/CLIP + logistic regression baseline.

Verification note: items marked [verified] were checked against search results this session. Items marked [prior knowledge] are from memory and were NOT re-checked; confirm before relying on them. Effort hours are rough estimates for one engineer on one consumer GPU.

## Fine-tuning strategy

1. **LP-FT (linear probe, then full FT)** [verified]
   - Two-step: train the head with frozen features first, then unfreeze everything at a low lr.
   - Kumar et al., ICLR 2022. Fine-tuning gave ~+2% ID but ~-7% OOD vs linear probing; LP-FT beat both.
   - Expected gain: small-to-moderate, mostly robustness to shift (night IR, low-res). Likely +1-4 pts on OOD-like subsets.
   - Effort: 1-2 h (change the training loop to a two-stage schedule).
   - Link: https://arxiv.org/abs/2202.10054

2. **FLYP (contrastive fine-tuning with class-name prompts)** [verified]
   - Replaces the CE head with a contrastive loss between image embeddings and class text prompts, like CLIP pretraining. Keeps the pretraining objective.
   - Goyal et al., CVPR 2023. +4.2% OOD over standard FT, >1% over LP-FT. Author-reported only; no independent replication found.
   - Expected gain: moderate if you can write good text prompts for each household member (e.g. "a photo of a person"). Caveat: with 10 identities that have no semantic names, the text side adds little. Probably less useful here than for ImageNet-style classes.
   - Effort: 3-5 h.
   - Link: https://openaccess.thecvf.com/content/CVPR2023/papers/Goyal_Finetune_Like_You_Pretrain_Improved_Finetuning_of_Zero-Shot_Vision_Models_CVPR_2023_paper.pdf

3. **WiSE-FT / model soups (weight averaging)** [verified]
   - WiSE-FT: interpolate zero-shot and fine-tuned weights (alpha ~0.5 is a good default). Wortsman et al. 2022.
   - Model soups: average several fine-tunes with different hyperparameters or seeds. ICML 2022. No extra inference cost.
   - Caveat: one paper found WiSE-FT did not help ID and OOD together for some fine-tuning methods. Check on your own validation split.
   - Expected gain: small (+1-3 pts) and stable. Strong fit for a small labelled set where seed variance is high.
   - Effort: 2-4 h (train 3-5 seeds, average state dicts, evaluate).
   - Links: https://arxiv.org/pdf/2109.01903 ; https://proceedings.mlr.press/v162/wortsman22a.html

4. **LoRA / DoRA / adapters vs full FT** [partly verified]
   - DoRA splits weights into magnitude and direction, applies LoRA to the direction. Reported gains are on LLMs and VLMs (LLaMA, LLaVA), not on ViT few-shot. ICML 2024.
   - [prior knowledge] For ~10-class, few-hundred-image regimes, LoRA or a partial FT (last N blocks) often matches or beats full FT and lowers overfitting. It also cuts optimizer memory a lot, which matters on one consumer GPU.
   - Expected gain: unclear; often neutral on accuracy, large on memory and speed. Worth testing as a cheaper arm of the ablation.
   - Effort: 2-3 h to add peft-style adapters to CLIP-L.
   - Link: https://arxiv.org/html/2402.09353v4

5. **Layer-wise LR decay (LLRD)** [prior knowledge]
   - Lower layers get smaller lr (e.g. decay 0.65-0.8 per layer). Standard for ViT fine-tuning (BEiT, MAE, EVA recipes).
   - Expected gain: small, often +0.5-2 pts; reduces the feature distortion that LP-FT addresses.
   - Effort: 1 h (param groups).

## Person re-ID specific

6. **CLIP-ReID** [verified]
   - Two-stage: learn identity-specific text prompts with frozen CLIP, then fine-tune the image encoder with ReID losses and the prompts fixed. Li and Gong (arXiv 2310.17218) report a prompt-free prototypical contrastive variant that is competitive.
   - Expected gain: meaningful for crop-based ID if you have many crops per person. With ~10 classes and few crops, the prompt-learning stage is probably overkill. The prototypical-contrastive variant is the one to try.
   - Effort: 4-8 h if reimplemented; 1-2 h for the prompt-free variant.
   - Links: https://arxiv.org/abs/2310.17218 ; https://arxiv.org/pdf/2506.12413 (survey on domain generalization for re-ID)

7. **TransReID** [verified only as a baseline]
   - Transformer re-ID baseline. Snippets show X-ReID beats it by +1.6 mAP on MSMT17. No 2025 state-of-the-art leaderboard found in this search, so do not treat any specific number as current SOTA.
   - Expected gain for this project: low to moderate. Its main benefit is the re-ID training recipe (triplet + ID loss, BNNeck, camera-aware sampling), which you can borrow without the full model.
   - Effort: 3-5 h to borrow the loss and sampler.
   - Link: https://arxiv.org/pdf/2302.02075

## Backbones

8. **SigLIP 2** [verified]
   - Same architecture as SigLIP. Better localization and dense features. Multiple resolutions via NaFlex. Sizes B, L, So400m, g.
   - Expected gain: uncertain. Try SigLIP2 So400m or L as a drop-in against CLIP-L with the same FT recipe.
   - Effort: 2-3 h (swap encoder and preprocessing, rerun).
   - Link: https://arxiv.org/pdf/2502.14786

9. **Perception Encoder (PE)** [partly verified]
   - Meta, 2025. Authors report intermediate-layer features are better than the output layer. Compared to SigLIP2 and DINOv2 on MLLM tasks. Fine-grained few-shot numbers not found in this search.
   - Expected gain: unknown for this task. Worth a test, especially the intermediate-layer features for the frozen-feature baseline.
   - Effort: 2-3 h.
   - Link: https://arxiv.org/pdf/2504.13181

10. **DINOv3** [partly verified]
   - Meta, 2025. Strong dense and frozen features (segmentation, correspondence, tracking). Good candidate for the frozen-feature baseline and for crop-level identity.
   - Expected gain: strong for the frozen baseline and for cat identity (fine-texture cues). Few-shot FT numbers not checked.
   - Effort: 1-2 h for a frozen baseline.
   - Link: https://arxiv.org/pdf/2508.10104

11. **EVA-CLIP** [prior knowledge]
   - Strong CLIP-family backbone; EVA-02-CLIP variants are often better than OpenAI CLIP-L on linear probes.
   - Expected gain: modest. Lower priority than SigLIP2 or PE.
   - Effort: 2 h.

## Open-set / unknown rejection

12. **kNN distance on normalized penultimate features** [verified]
    - Sun et al., ICML 2022. L2-normalize features; score = distance to k-th nearest training neighbour. Normalization alone cut FPR95 by ~61% in their ablation.
    - Expected gain: high practical value. Gives an "unknown person" reject without extra training. Tune the threshold on a held-out set of non-household people.
    - Effort: 2-3 h.
    - Link: https://arxiv.org/pdf/2204.06507

13. **Energy score / ODIN / max softmax** [prior knowledge]
    - Energy (Liu et al. 2020) and ODIN (Liang et al. 2018) use logits. Cheap to add. Logit-based scores can be miscalibrated after label smoothing, so check that first.
    - Expected gain: uncertain; compare against kNN on the same validation set.
    - Effort: 1 h each.

14. **Prototype / cosine-threshold rejection** [prior knowledge]
    - Class prototypes from embeddings; reject if max cosine similarity is below a threshold. This is the same idea as kNN with k=1 and is the simplest baseline.
    - Effort: under 1 h.

## Augmentation for low-res / IR

15. **Augmentation set** [prior knowledge, not verified this session]
    - Gaussian blur and downscale-then-upscale (simulate low-res), random grayscale (simulate IR), random erasing (occlusion), RandAugment with mild magnitude, brightness/gamma (night).
    - Expected gain: likely the biggest single lever on low-res and night crops. Easy to validate by evaluating on a held-out night/low-res slice.
    - Effort: 1-2 h, plus evaluation time.

## Test-time and ensembling

16. **TTA** [prior knowledge]
    - Horizontal flip plus a few scales, average the embeddings (not the logits). Modest gain, low cost.
    - Effort: under 1 h.

17. **Ensembling** [prior knowledge]
    - Average logits or embeddings of the FT model and the frozen-feature logistic regression (DINOv2 or DINOv3). Often beats either alone on small data.
    - Effort: 1-2 h.

## Distillation for CPU deployment

18. **Distill to a small model** [prior knowledge]
    - Teacher: the best FT ViT-L or SigLIP2. Student: a small ViT or MobileNet-class model, trained with KD loss on embeddings plus labels, on unlabelled crops from the camera too.
    - Expected gain: small accuracy loss; large speedup. Good fit for CPU-only.
    - Effort: 6-10 h.

## Ranked top-5 to try first

1. **Augmentation for low-res / IR / occlusion** (item 15). Cheapest, targets your worst crops, 1-2 h.
2. **kNN distance rejection on normalized features** (item 12). Adds the "unknown person" handling you need, 2-3 h.
3. **LP-FT + layer-wise LR decay** (items 1, 5). Small change to the existing training loop, 2-3 h total. Targets feature distortion.
4. **WiSE-FT / 3-5 seed soup** (item 3). Stabilizes the small-data result, 2-4 h.
5. **SigLIP2 or PE backbone with the same FT recipe** (items 8, 9), plus the DINOv3 frozen baseline (item 10). Backbone swap is 2-3 h each; use the frozen DINOv3 baseline as a cheap check first.

Deprioritized for now: FLYP (needs good class text, weak fit for anonymous identity classes), CLIP-ReID full pipeline (overkill for ~10 classes), DoRA (test only if memory is a problem).

## Gaps

- No 2025-2026 head-to-head of PE, SigLIP2, DINOv3 on small-data fine-grained FT was found.
- No current re-ID SOTA leaderboard was checked; the snippets are mostly 2022-2023.
- Items 5, 11, 13, 14, 15, 16, 17, 18 are from memory and need a check before use.

## Experiment 1 (training/autolabel_v2.py, DINOv2-small, CPU)
569 human crops / 9 persona classes, clip-grouped hold-out, 1538 evaluated crops:

| strategy | prec@0.90 | cover | prec@0.95 | cover |
|---|---|---|---|---|
| lr (baseline) | 85.1% | 32% | 89.6% | 21% |
| lr + per-class thr | 81.9% | 20% | 81.7% | 18% |
| label spreading (kNN) | 56.4% | 5% | 58.8% | 4% |
| lr+spread blend | 75.5% | 15% | 76.0% | 14% |
| blend + kNN-distance reject | 75.1% | 15% | 75.6% | 14% |

Conclusion: on small frozen features none beats the baseline; no strategy reaches the target precision.
Per-class thresholds are noisy (inner split too small); label spreading is hurt by weak features.
Next: rerun on the GPU server with DINOv2-L/G + CLIP-L features (autolabel.py combos), and tune spread k/alpha.

## Experiment 2 (same script on the RTX 5070 server, cached big features, `--combo`)
Same 1538 held-out human crops. Precision / coverage:

| strategy | DINOv2-L+CLIP-L @0.90 | @0.95 | DINOv2-G+CLIP-L @0.90 | @0.95 |
|---|---|---|---|---|
| lr (baseline) | 91.5% / 58% | 95.5% / 39% | 89.6% / 60% | 93.7% / 39% |
| lr + per-class thr | 88.5% / 51% | 90.7% / 44% | 88.9% / 51% | 91.2% / 44% |
| label spreading | 77.2% / 17% | 75.1% / 13% | 80.1% / 14% | 80.0% / 11% |
| blend | 87.6% / 36% | 87.8% / 31% | 89.8% / 36% | 89.6% / 35% |
| blend + kNN reject | 87.2% / 34% | 87.5% / 30% | 90.8% / 35% | 90.6% / 34% |

Conclusion: big features lift everything (baseline 95.5% precision at 39% coverage), but the plain logistic
regression with a global threshold is still the best; label spreading, per-class thresholds and kNN reject
do not help. Don't build on them; try the fine-tuning ideas (LP-FT + LLRD, soups, SigLIP2/PE) next.

## Experiment 3 (training/ft_experiments.py, CLIP-L, RTX 5070, human labels only, export5, 15 epochs)
Balanced accuracy, mean ± std over 3 folds per protocol (v3 reference from earlier runs: group 75.9%, time 78.1%).

| recipe | group | time |
|---|---|---|
| base (current recipe) | 76.9 ± 3.7 | 69.2 ± 8.0 |
| lpft | 73.2 ± 2.0 | 66.1 ± 10.4 |
| lpft_llrd | 73.8 ± 3.3 | 62.7 ± 8.1 |
| lpft_aug | 76.9 ± 3.4 | 64.9 ± 15.0 |
| full (lpft+llrd+aug) | 75.7 ± 3.8 | 66.1 ± 11.7 |
| wise_base | 72.1 ± 1.6 | 59.6 ± 8.4 |
| wise_lpft_aug | 75.9 ± 4.3 | 69.1 ± 8.3 |
| wise_full | 72.9 ± 2.6 | 68.1 ± 5.3 |

Conclusion: no recipe beats the current one. Differences are within the fold-to-fold spread (3-15 points), so
nothing here is significant. WiSE-FT at alpha 0.5 does not help; LP-FT, LLRD and the harder augmentation
do not help either. The `base` time number (69.2%) is far below the v3 reference (78.1%): the folds/data
differ (export5 vs the export used for v3), so only compare recipes within this table.
Next with real potential: more human labels (active learning on the review queue), backbones (SigLIP2/PE),
model soup over seeds, and more folds to reduce noise.

## Experiment 4: matched PE Core-L vs CLIP-L fine-tuning (export5)

Human labels only: 569 crops, 9 classes. Both models used the same 10-epoch recipe, 3-seed softmax
ensemble per fold, and identical splits: 5 clip-grouped folds (seeds 0-4) plus 3 temporal folds.
Metrics are balanced accuracy, mean ± population standard deviation across folds.

| backbone | group | time |
|---|---:|---:|
| CLIP-L | 76.4 ± 5.4 | 66.0 ± 10.9 |
| PE Core-L | 81.3 ± 2.9 | 77.0 ± 15.6 |

Paired fold deltas (PE minus CLIP-L, percentage points): group +5.2, +7.1, +9.6, -6.6, +9.2
(mean +4.9); time +15.5, +13.3, +4.2 (mean +11.0). The means favor PE, but this is descriptive
only: the main matched comparison has just three overlapping temporal windows, and PE's temporal
standard deviation is 15.6 points.
In the weakest temporal window, PE scored 55.0% vs CLIP-L 50.8%; both got 0/6 held-out “Io” crops
correct. “Lucrezia” and “Persona conosciuta” had no validation examples in that window. Do not treat
the average gain as evidence that PE is production-ready.

### Additional rolling-origin temporal stress folds

Two earlier rolling windows were added (train on the oldest 40% or 50% of clips; validate on the next
20%). Results: at 40%, CLIP-L 35.7% / PE 43.7% balanced accuracy; at 50%, CLIP-L 71.3% / PE 84.5%.
The 40% training prefix contains no “Lucrezia” crops and only 7 “Io” crops, while its validation set
contains 10 Lucrezia crops. Keep this as an early-history stress test, not an apples-to-apples estimate
of steady-state model quality. The 50% window contains training examples from all classes.

### Frozen PE features for pseudo-labeling

Extracted normalized PE Core-L features for all 1,894 persona rows (569 human-labelled + 1,325
unlabelled), 1,024 dimensions. `autolabel_v2.py` evaluation used held-out human labels only (1,538
evaluated predictions across its grouped splits); the plain global-threshold logistic-regression results:

| features | precision @0.90 | coverage @0.90 | precision @0.95 | coverage @0.95 |
|---|---:|---:|---:|---:|
| PE Core-L | 93.6% | 73% | 94.4% | 68% |
| DINOv2-L + CLIP-L baseline | 91.5% | 58% | 95.5% | 39% |

PE gives higher coverage and slightly higher precision at the 0.90 operating point. At the 0.95 target,
PE reaches only 94.4%, below the target, despite 68% coverage; it does not replace the baseline for a
strict ≥95% precision requirement. No human labels or database auto-labels were changed.

### Watcher integration smoke test (not a deploy candidate)

Added `--model pe_core_L` to `training/train_final.py` while retaining CLIP-L as the default, and made
the checkpoint carry its native input size so `PersonModel` uses 336×336 for PE and keeps 224×224 for
legacy checkpoints. A one-epoch artifact-smoke run trained on the 569 human labels and produced a
632,320,879-byte checkpoint; the real checkpoint loaded through `PersonModel` at 336×336 and completed
inference on a human crop. The remote training venv lacks OpenCV, so that final smoke call supplied a
minimal BGR→RGB conversion shim; no production watcher/container was changed. The artifact deliberately
skipped validation and threshold selection and must not be deployed.

Local tests after the watcher change: 9/9 passed with `unittest`; `pytest` is not installed in the local
venv. No production model was changed.

## Experiment 4 (training/backbone_bench.py: frozen backbones, logistic regression, 10 group + 3 time folds)
Balanced accuracy group / time: PE core L 75.3 / 78.9 (best), EVA02-CLIP L 68.1 / 62.8, CLIP-L 65.0 / 65.9,
DINOv3 L 64.7 / 58.6, SigLIP2 so400m 64.4 / 61.5, SigLIP2 L 59.3 / 52.3, DINOv2-L 56.1 / 57.2.
Pairs are no better than PE alone (best pair PE+SigLIP2 so400m 73.2 / 74.6).

## Experiment 5 (training/ft_backbones.py: PE core L 336px fine-tuned, 3 seeds per fold, 10 epochs, same folds as Exp. 3)
| | group | time |
|---|---|---|
| single seed | 80.4 ± 2.2 | 74.8 ± 12.9 |
| seed ensemble | 80.8 ± 1.7 | 77.3 ± 16.0 |
| model soup | 80.9 ± 1.3 | 75.2 ± 15.9 |
| CLIP-L base (Exp. 3) | 76.9 ± 3.7 | 69.2 ± 8.0 |

Per fold (single seed): group 83.4 / 78.4 / 79.3 vs CLIP-L base 79.6 / 79.4 / 71.6; time 83.9 / 83.9 / 56.6 vs 63.9 / 80.5 / 63.2.
Conclusion: PE core L beats CLIP-L by ~4 points on group folds and ~6-8 on time folds on average, better in 4 of 6
folds, but the time protocol is very noisy (fold 2 is bad for every model). Soup ≈ ensemble ≈ single, so the
soup is not worth the cost; an ensemble only helps clearly on time folds 0-1. PE is the best candidate to replace CLIP-L
(not yet ported to the watcher; production model unchanged). Cost: 336px, ~3 min per training run.
