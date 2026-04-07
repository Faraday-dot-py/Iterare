# Exp 25: ST + Cosine Annealing + Best-Prefix, PREFIX_LEN=32

**Status:** Queued | **Results:** Pending

## Method

Identical to Exp19 (SOTA method) but with PREFIX_LEN=32.
- ST estimator with cosine LR annealing (LR_MAX=0.01 → LR_MIN=0.001)
- Best-prefix tracking over 300 soft steps
- HOTFLIP_STEPS=35 (reduced from 80 due to per-step time at len=32)
- HF_TOPK=50, BATCH_SIZE=12, seed=42
- Checkpoint every 5 HotFlip steps (fewer total steps, checkpoint more often)

## Research Question

At what prefix length does the scaling improvement plateau?
If len=32 gives CE≈0.672 (projecting from Exp19 trend), the trend is still strong.
If CE≥0.679, diminishing returns have set in by len=32.

## Timing Estimate

~5.5h: soft=2400s + HotFlip 35 steps × 498s each ≈ 19830s total

## Comparison

| Exp | len | HotFlip CE |
|-----|-----|-----------|
| 16 | 8 | 0.686 |
| 19 | 16 | **0.679** ← SOTA |
| 24 | 24 | — |
| **25** | **32** | **—** |
