# Manager 8: Exp 8 — Multi-Seed Random Restarts

**Task:** steer-001 | **Status:** Planned

---

## Objective

Test whether the standard soft→project→HotFlip pipeline's result (Exp 1: CE=0.740)
is stable or highly variable across random seeds, and whether trying multiple seeds
can find significantly better discrete prefixes.

**Research question:** Is CE=0.740 near the limit for this method, or is it one
sample from a distribution? Can repeated independent tries find CE < 0.74?

---

## Design

- Reference prefix: `"Talk only about cats."`
- PREFIX_LEN = 8 (same as Exp 1 baseline)
- Seeds: {0, 1, 2, 3, 4, 5, 6, 7} — 8 independent runs
- Same pipeline: 300 soft opt steps → cosine project → 80 HotFlip steps
- Same hyperparameters as Exp 1 (BATCH_SIZE=8, LR=0.01, EARLY_K=32, EARLY_WEIGHT=3.0)
- Record soft CE, projection CE, and HotFlip CE for each seed
- Report best, worst, mean, std across seeds

**Hypothesis 1 (low variance):** CE is tightly concentrated around 0.74 across seeds.
This would suggest the local minimum is the global minimum for this method, and
improvements require a fundamentally different approach.

**Hypothesis 2 (high variance):** CE varies substantially (e.g., 0.5–1.0 range).
This would suggest random restarts are a viable strategy and better prefixes exist.

**Comparison baseline:** Exp 1 seed=42 → HotFlip CE=0.740

---

## Results

*(Pending)*

---

## Worker

| Worker | Script | Output |
|--------|--------|--------|
| worker-1 | `multi_seed.py` | `multi_seed_results.json` (pending) |
