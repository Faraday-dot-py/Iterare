# Worker-1: Exp 3 — Layer-Depth Projection Ablation

**Task:** steer-001 | **Manager:** manager-3 | **Status:** Complete

---

## Objective

Test whether projecting the soft prefix at *earlier* transformer layers (using a
state-matching objective) yields better behavioral fidelity than the standard
late-layer CE-objective soft opt.

**Research question:** Does targeting an earlier layer lead to better CE after HotFlip?

---

## Design

| Parameter | Value |
|-----------|-------|
| Reference prefix | `"Talk only about cats."` |
| Model | `google/gemma-2-2b-it` (26 layers) |
| Target layers | {4, 10, 16, 22} |
| Soft opt objective | State matching (cosine similarity of hidden states at target layer) |
| PREFIX_LEN | 6 tokens |
| N_SOFT_STEPS | 300 |
| N_HOTFLIP_STEPS | 50 |
| BATCH_SIZE | 8 suffixes |

Note: This differs from the baseline in two important ways:
1. Soft opt objective is **state matching** (not CE)
2. PREFIX_LEN is 6 (not 8)

---

## Results

| Layer | State-Match Loss | Proj CE | HotFlip CE | Proj Nat | Final Nat | Final Prefix |
|-------|-----------------|---------|-----------|---------|---------|--------------|
| 4  | 0.086 | 1.35852 | **0.96508** | 0.500 | **0.833** | `richTextPaneliläumrawDesc Cats¹. Silas` |
| 10 | 0.035 | 1.40852 | 1.29669 | 0.167 | 0.667 | ` ໊ResponseWriter resident Arsenicapia` |
| 16 | 0.023 | **1.23325** | 1.07447 | 0.500 | 0.833 | `iastesflage ulasan feline🦡 Bounty` |
| 22 | 0.016 | 1.34640 | 1.01140 | 0.500 | 0.500 | `ҿ\n\n\ue087ISODEISupport cats` |

**Baseline comparison (Exp 1, CE objective, PREFIX_LEN=8):**
- Proj CE: 1.436 | HotFlip CE: **0.740**

Total TIDE wall-clock: 7866 seconds (~2.2 hours)

---

## Key Observations

1. **State matching improves projection CE at Layer 16.** The Layer 16 projection CE
   (1.233) is lower than both the baseline cosine projection (1.436) and all other
   layers. Projecting at an intermediate layer may find tokens that are "better
   positioned" in the cosine neighborhood of behaviorally relevant tokens. [M]

2. **The relationship between layer depth and projection CE is non-monotonic.**
   L16 > L4 > L22 > L10 for projection quality (lower CE is better). This
   contradicts the naive hypothesis that earlier layers always project better. [M]

3. **State matching is worse than CE soft opt for end-to-end HotFlip CE.**
   All state-matched prefixes achieve HotFlip CE ≥ 0.965, compared to the CE-
   optimized baseline's 0.740. The state-matching objective doesn't directly
   minimize CE, so the HotFlip refinement starts from a less favorable point. [H]

4. **State matching produces more natural prefixes.** L4 and L16 achieve final
   naturalness 0.833 (English-ish tokens like "Cats¹", "Silas", "feline", "Bounty").
   The baseline final prefix (naturalness ~0.25) contains mostly multilingual garbage.
   This may be because the state-matching objective implicitly favors tokens that
   preserve the model's expected internal representation. [L]

5. **"cats" survives in L4, "feline" in L16, "cats" in L22.** The semantic anchor
   phenomenon from Exp 2 persists here — at least one cat-related token appears in
   HotFlip output for most layers. [M]

---

## Interpretation

The state-matching approach reveals which layer best preserves behavioral information
in the cosine projection neighborhood:
- **Layer 16** (≈ 62% depth) gives the best starting point for projection (CE=1.233)
- **Layer 4** (≈ 15% depth) gives the best HotFlip recovery from that start (CE=0.965)
- But neither approaches the CE-objective baseline (0.740)

**Main conclusion:** For the cats-persona steering task, CE-based soft opt followed
by cosine projection remains better than state-matching at any single target layer.
The state-matching approach may be more valuable for tasks where the intermediate
representation at a specific layer carries distinctive semantic content. [L]

---

## Artefacts

| File | Description |
|------|-------------|
| `layerablation.py` | Full pipeline (state matching at 4 target layers) |
| `layerablation_results.json` | Complete results for all 4 layers |
