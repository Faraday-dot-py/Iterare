# Exp 28: ST + Cosine Annealing + Best-Prefix, PREFIX_LEN=48

**Status:** Queued | **Results:** Pending

## Method

Identical to Exp25 (SOTA method) but PREFIX_LEN=48 (1.5× Exp25).
HOTFLIP_STEPS=30 (reduced to fit time budget; 30 × ~320s ≈ 9600s).

## Scaling Context

| Exp | len | HotFlip CE | Δ from prev |
|-----|-----|-----------|------------|
| 19 | 16 | 0.679 | — |
| 24 | 24 | 0.669 | −0.010 |
| 25 | 32 | 0.632 | −0.037 ← super-linear |
| **28** | **48** | **—** | **?** |

Extrapolating: if scaling continues at len=48, CE could approach 0.59–0.60.
