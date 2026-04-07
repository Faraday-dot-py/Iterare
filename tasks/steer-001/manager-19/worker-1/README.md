# Exp 19: ST + Cosine Annealing + Best-Prefix, PREFIX_LEN=16

**Result: HotFlip CE = 0.6794 — NEW SOTA**

## Method

ST estimator + cosine LR annealing (LR_MAX=0.01→LR_MIN=0.001) + best-prefix tracking,
PREFIX_LEN=16, seed=42, 300-step ST + 80-step HotFlip TOPK=30, BATCH_SIZE=12.

## Results

| Stage | CE |
|-------|-----|
| Best ST-CE | 0.8242 |
| Final ST-CE | 0.8855 |
| Proj CE (best snapshot) | 0.9053 |
| Proj CE (final step) | 0.8635 |
| HotFlip CE | **0.6794** |

Projected prefix: `"Cats😴 pets instructionsLean Carlo pets Archive Format:"CAT Speech witty cats responses Transcript"`
Final prefix: `"Cats).[animals instructionsبوابةver PflegeLIBRARYEspecificaciones:" earsXmlAccessType witty cats responses General"`

Timing: soft=1191s, hotflip=19933s (total ~5.9h on A100)

## Comparison

| Method | Proj CE | HotFlip CE |
|--------|---------|------------|
| Exp 11 (ST, len=8, constant LR) | 0.762 | 0.689 |
| Exp 16 (ST fp32, len=8, best-prefix) | 0.877 | 0.686 |
| Exp 19 (ST, **len=16**, cosine, best-prefix) | 0.905 | **0.679** |

## Key Findings

- **PREFIX_LEN=16 achieves new SOTA 0.6794**, improving on Exp16's 0.6861 by −0.007 CE
- Longer prefixes give more discrete tokens for HotFlip to optimize, enabling better recovery
- Despite higher proj CE (0.905 vs 0.877), HotFlip recovers further — confirming that
  the number of discrete degrees of freedom matters more than starting projection quality
- HotFlip runtime was 19933s (~5.5h) for 80 steps on 16-token prefix with TOPK=30
- Compared to Exp11's Δ=0.073 and Exp16's Δ=0.191, Exp19 recovered Δ=0.226 (proj→HotFlip)
