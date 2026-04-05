# Worker-1: Exp 7 — Gumbel-Softmax End-to-End Discrete Training

**Task:** steer-001 | **Manager:** manager-7 | **Status:** Complete

---

## Objective

Test whether training directly in the discrete token distribution space (via
Gumbel-Softmax relaxation) outperforms the standard two-stage soft→project pipeline.

**Research question:** Can end-to-end discrete-aware training close the soft→discrete
gap without a separate projection step?

---

## Design

| Parameter | Value |
|-----------|-------|
| Reference prefix | `"Talk only about cats."` |
| PREFIX_LEN | 8 tokens |
| Parameterization | logits L ∈ R^{8 × 256000} |
| Temperature schedule | τ: 2.0 → 0.1 (exponential, 500 steps) |
| Training steps | 500 |
| HotFlip refinement | 50 steps after argmax extraction |
| LR | 0.05 (higher than soft opt; logits space is sparser) |
| BATCH_SIZE | 8 suffixes |

**Key implementation fix:** Per-suffix backward with detached leaf pattern to avoid
OOM. Gumbel embedding is computed once → detached → per-suffix backward accumulates
gradients → propagated back through Gumbel graph in one shot.

---

## Results

| Stage | CE | Notes |
|-------|----|-------|
| Exp 1 baseline (standard pipeline) | **0.74** | soft=0.191, proj=1.436 |
| Gumbel training (500 steps, τ: 2.0→0.1) | 1.35 (soft CE during training) | oscillates 1.30–1.41 |
| Gumbel argmax extraction | 1.452 | hard argmax of trained logits |
| Gumbel + HotFlip (50 steps) | **1.272** | converged step 10 |

**Improvement over baseline: −0.532** (Gumbel+HotFlip is 72% WORSE than standard pipeline)

Total TIDE wall-clock: 3983 seconds (~1.11 hours)

---

## Key Observations

1. **Gumbel-Softmax performs significantly worse than the standard pipeline.** The
   final HotFlip CE of 1.272 is 0.532 worse than Exp 1's 0.740. The approach not
   only fails to close the gap — it produces a substantially worse discrete prefix. [H]

2. **The soft→argmax gap is large and worsens during temperature annealing.**
   During Gumbel training, the soft CE (based on the weighted-average embedding) was
   ~1.30–1.35. But the hard argmax CE was 1.452 — higher than even the standard
   cosine projection CE (1.436 in Exp 1). Annealing toward discrete concentrates
   probability mass, but the most probable token at each position is not optimal. [H]

3. **HotFlip recovery from Gumbel argmax is fast but limited.** HotFlip converged
   at step 10 from 1.452 → 1.272. This rapid convergence suggests the Gumbel argmax
   is already in a local minimum that HotFlip can refine only slightly. Contrast with
   Exp 1 where HotFlip recovered 0.696 CE units over 80 steps. [M]

4. **Vocabulary explosion makes optimization difficult.** Gumbel-Softmax requires
   computing gradients through a [PREFIX_LEN × V] = [8 × 256000] logit matrix at
   every step. Despite the per-suffix backward fix, this remains costly. The large
   vocab may also cause diffuse gradients that prevent localization to good tokens. [M]

5. **The soft CE during training (1.30–1.41) never approached the standard soft opt
   CE (0.191).** This shows that Gumbel training, even at high temperature, optimizes
   a fundamentally different objective. The temperature introduces noise that prevents
   the optimizer from finding the embedding that the model would find optimal. [M]

---

## Interpretation

The Gumbel-Softmax result is a strong negative finding: **discrete-aware training
does not outperform the separate soft optimize → project pipeline for this task.**

The core problem is that Gumbel-Softmax conflates two distinct challenges:
1. **Finding a good continuous direction** in embedding space (where soft opt excels)
2. **Discretizing that direction** to vocabulary tokens (where projection fails)

By trying to solve both simultaneously, Gumbel-Softmax does neither well. The
continuous optimization is degraded by the Gumbel noise, and the discrete constraint
is never strong enough until the temperature is so low that gradients become too noisy.

**Main conclusion:** The standard soft→project→HotFlip pipeline outperforms
Gumbel-Softmax by a large margin. The bottleneck is in the discretization step, not
the optimization objective. Methods that improve the projection step (e.g., better
initialization for HotFlip) are more promising than end-to-end discrete training.

---

## Artefacts

| File | Description |
|------|-------------|
| `gumbel_softmax.py` | Full pipeline (Gumbel training + HotFlip) |
| `gumbel_results.json` | Complete results |
