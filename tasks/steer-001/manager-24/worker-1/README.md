# Exp 24: ST + Cosine Annealing + Best-Prefix, PREFIX_LEN=24

**Status:** Queued | **Results:** Pending

## Method

Identical to Exp19 (SOTA method) but with PREFIX_LEN=24 instead of 16.
- ST estimator: `CE(cosine_project(soft_prefix))`
- Cosine LR annealing: LR_MAX=0.01 → LR_MIN=0.001 over 300 soft steps
- Best-prefix tracking: project from step with lowest ST-CE, not final step
- SOFT_STEPS=300, HOTFLIP_STEPS=50 (reduced from 80 to fit time budget), HF_TOPK=50
- Checkpoint every 10 HotFlip steps to `/home/jovyan/steer001_scale24_ckpt.json`

## Research Question

Does the 8→16 token improvement (CE 0.686→0.679) continue at 16→24 tokens?
Scaling trend suggests HotFlip recovery Δ grows with prefix length:
- len=8 (Exp11): Δ=0.073 (proj 0.762 → hotflip 0.689)
- len=8 fp32 (Exp16): Δ=0.191 (proj 0.877 → hotflip 0.686)
- len=16 (Exp19): Δ=0.226 (proj 0.905 → hotflip 0.679)
- len=24 (**this exp**): expected Δ≈0.25+?

## Timing Estimate

~5.7h: soft=1800s + HotFlip 50 steps × 374s each = ~20500s total

## Comparison

| Exp | len | Method | HotFlip CE |
|-----|-----|--------|-----------|
| 16 (λ=0.0) | 8 | ST fp32 + best-prefix | 0.686 |
| **19** | **16** | **ST + cosine + best-prefix** | **0.679** ← SOTA |
| **24** | **24** | **ST + cosine + best-prefix** | **—** |
