# Exp 25: ST + Cosine Annealing + Best-Prefix, PREFIX_LEN=32

**Result: NEW SOTA — HotFlip CE=0.63171 (Exp24 len=24 was 0.669, Exp19 len=16 was 0.679)**

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

## Results

| Stage | CE | Notes |
|-------|----|-------|
| Best ST-CE (soft) | 0.79103 | Best so far across all experiments |
| Projection (best step) | 0.87282 | |
| HotFlip (35 steps) | **0.63171** | **New SOTA** — largest single improvement yet |

**Timing:** soft=691s, hotflip=8280s (237s/step at len=32)

**Final prefix:** `Cats Cats﻿<? Nowak veta chatbot iyoUSERrore dicendo BIR risposte gooseêtre fidèlesур Noorate Interrupt Groom snoringsoltuating Feast Realistic FM obedience feline:**ABOUT haughty cats`

**Final IDs:** `[66589, 50105, 232181, 167791, 147994, 183294, 94337, 14053, 155060, 216690, 55755, 194422, 50014, 6695, 190559, 24181, 1307, 205118, 133794, 108053, 184370, 214624, 136354, 99319, 141341, 25401, 56284, 145156, 66058, 29606, 167309, 19493]`

## Full Scaling Table (updated)

| Exp | len | Proj CE | HotFlip CE | HF Δ | vs Exp19 |
|-----|-----|---------|-----------|------|----------|
| 16 | 8 | 0.877 | 0.686 | 0.191 | — |
| 19 | 16 | 0.905 | 0.679 | 0.226 | baseline |
| 24 | 24 | 0.867 | 0.669 | 0.198 | −0.010 |
| **25** | **32** | **0.873** | **0.632** | **0.241** | **−0.047** |

## Key Findings

- **Dramatic improvement at len=32**: CE drops 0.037 from len=24 (0.669→0.632), much larger than the 0.010 improvement from len=16→24
- Only 35 HotFlip steps used — the HotFlip log still showed improvement at the last step; more steps could push CE lower
- The prefix contains explicit cat semantics at positions 0-1 ("Cats Cats") and end ("feline", "haughty cats")
- Scaling is clearly not saturating at len=32; **len=48 or 64 should be tried**
- HotFlip per-step time: 237s/step — faster than my estimate, suggesting the A100 handles len=32 more efficiently than expected
