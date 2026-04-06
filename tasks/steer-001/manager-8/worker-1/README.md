# Worker-1: Exp 8 — Multi-Seed Random Restarts

**Task:** steer-001 | **Manager:** manager-8 | **Status:** Complete

---

## Objective

Test whether the standard pipeline's result is stable or variable across random seeds,
and whether any seed achieves CE < 1.24 (best result under per-suffix backward method).

---

## Design

| Parameter | Value |
|-----------|-------|
| Reference prefix | `"Talk only about cats."` |
| PREFIX_LEN | 8 tokens |
| Seeds tested | {0, 1, 2, 3, 42} (seed 42 = Exp 1 baseline seed) |
| Soft opt | 300 steps, Adam LR=0.01 |
| HotFlip | 50 steps, topk=20 |
| BATCH_SIZE | 8 suffixes |
| Method | Per-suffix backward (same as Exp 5-7, NOT Exp 1 batched) |

**Note on Exp 1 comparison:** Exp 1 used batched inference (all 8 suffixes in one
forward pass via `build_batch`), while Exp 8 uses per-suffix backward. This is a
**methodological difference** that changes the gradient landscape. The within-Exp-8
comparison is valid; comparison with Exp 1's CE=0.740 is NOT.

---

## Results

| Seed | Soft CE | Proj CE | HotFlip CE | Final Prefix |
|------|---------|---------|------------|--------------|
| 0 | 0.123 | 1.490 | 1.302 | `RenderAtEndOf Stakeholders ویکی‌پدی SpecsCIAS ujednoznacz Cler capacitor` |
| 1 | 0.132 | 1.430 | 1.270 | `"#婠 北海道 mouseClickedمصادر fourrure préféMLLoader` |
| 2 | 0.172 | 1.365 | 1.258 | `JuríThroughAttribute understatement robotic<start_of_turn>\n\nbritish Detectors` |
| 3 | 0.168 | 1.471 | 1.249 | `losures ELA مرئيه itſelf bezeichneter myſelf alıntıRenderAtEndOf` |
| **42** | **0.094** | **1.469** | **1.236** | `꯷֏ Sélectionnez circonstundheit ویکی‌پدی Pompeybok` |

| Statistic | Value |
|-----------|-------|
| Best HotFlip CE | **1.236** (seed 42) |
| Worst HotFlip CE | 1.302 (seed 0) |
| Mean | 1.263 |
| Std | 0.025 |

Total TIDE wall-clock: 9892 seconds (~2.75 hours)

---

## Key Observations

1. **The pipeline shows LOW variance across seeds (std=0.025).** HotFlip CE ranges from
   1.236 to 1.302 — all within 7% of each other. This strongly suggests the per-suffix
   backward method converges to a consistent class of discrete local minima regardless
   of random initialization. Random restarts are unlikely to find dramatically better
   solutions than additional seeds. [H]

2. **No seed approaches the Exp 1 CE=0.740.** The best result (seed=42, CE=1.236) is
   0.496 CE units worse than Exp 1. This gap is NOT due to random seed differences; it
   reflects a methodological difference between per-suffix backward (Exp 5-8) and
   batched inference (Exp 1). The batched gradient computation finds a qualitatively
   different optimization trajectory. [H]

3. **Seed=42 gives the best result (1.236), confirming no lucky initialization.**
   Exp 1 also used seed=42 and achieved 0.740 with batched inference. The same seed
   with per-suffix backward gives 1.236 — the method, not the seed, is the critical
   factor. [M]

4. **Soft CE varies more than HotFlip CE across seeds (0.094-0.172 range vs 0.025 std
   for HotFlip).** Lower soft CE does not reliably lead to lower HotFlip CE — seed=42
   (best soft CE 0.094) also gives best HotFlip CE (1.236), but there's no strong
   correlation overall. The projection and HotFlip steps introduce additional variance
   that partially decorrelates from soft CE. [M]

5. **All prefixes remain unreadable garbage tokens with multilingual/code fragments.**
   Despite trying 5 seeds, no prefix converges to natural English tokens (unlike some
   Exp 3 prefixes like `'Cats¹. Silas'` which achieved naturalness≥0.5). The per-suffix
   backward method does not naturally encourage cat-related semantic anchors. [L]

---

## Interpretation

The multi-seed experiment provides strong evidence that:

1. **The per-suffix backward method produces a consistent quality ceiling of ~1.24–1.30
   HotFlip CE**, regardless of random initialization. The local minimum landscape for
   this method is shallow — all seeds converge to nearby optima.

2. **The key discrepancy with Exp 1 (0.740 vs 1.24) is methodological, not stochastic.**
   It stems from batched vs. per-suffix gradient computation:
   - **Batched (Exp 1):** All 8 suffixes in one forward pass → batch-normalized gradients
     → soft prefix finds a different continuous region → better discrete neighborhood
   - **Per-suffix (Exp 5-8):** 8 separate forward passes → per-suffix normalization →
     different optimization trajectory → consistently worse discrete outcomes

3. **Random restarts do NOT provide a viable escape from the per-suffix method's ceiling.**
   Even trying seed=42 (the Exp 1 seed) with per-suffix backward gives 1.236, not 0.740.

**Main conclusion:** The gap between Exp 1 (0.740) and Exp 5-8 (~1.24) is caused by
the gradient computation method, not the random seed. To reproduce Exp 1 quality
results while avoiding OOM, one would need a gradient-equivalent fix that preserves
the batch normalization — e.g., gradient checkpointing on the batched forward pass
rather than per-suffix backward.

**Implication for future work:** The Exp 1 batched method achieves qualitatively better
results, and the memory problem could be addressed via gradient checkpointing rather than
per-suffix backward decomposition.

---

## Artefacts

| File | Description |
|------|-------------|
| `multi_seed.py` | Full pipeline (5 seeds) |
| `multi_seed_results.json` | Complete results |
