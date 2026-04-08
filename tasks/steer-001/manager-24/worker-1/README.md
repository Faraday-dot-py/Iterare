# Exp 24: ST + Cosine Annealing + Best-Prefix, PREFIX_LEN=24

**Result: NEW SOTA — HotFlip CE=0.66896 (Exp19 was 0.6794)**

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

## Results

| Stage | CE | Notes |
|-------|----|-------|
| Best ST-CE (soft) | 0.82385 | Voronoi oscillation visible; proj CE higher |
| Projection (best step) | 0.86651 | Similar to Exp19 (0.905) |
| HotFlip (50 steps) | **0.66896** | **New SOTA** (Exp19: 0.6794) |

**Timing:** soft=650s, hotflip=8404s (168s/step — faster than Exp19's 249s/step at len=16)

**Final prefix:** ` cats physiology θα litterERDcriminatormanu"]Welcome newOwner 只cVarmampuan precisione automatiquesHer replies serezTableAdapter mischievous amusing cats clear feline`

**Final IDs:** `[19493, 68916, 41593, 39416, 125322, 116036, 170467, 113427, 11890, 220889, 95372, 173785, 70190, 231581, 204166, 8274, 10357, 130103, 107469, 136903, 69921, 19493, 3110, 145156]`

## Scaling Table (updated)

| Exp | len | Proj CE | HotFlip CE | HF Recovery Δ |
|-----|-----|---------|-----------|---------------|
| 16 | 8 | 0.877 | 0.686 | 0.191 |
| 19 | 16 | 0.905 | 0.679 | 0.226 |
| **24** | **24** | **0.867** | **0.669** | **0.198** |

## Key Findings

- Scaling continues: len=24 beats len=16 SOTA by 0.010 CE (0.679→0.669)
- HotFlip per-step time *decreased* vs Exp19 (168s vs 249s) — likely GPU batch efficiency at longer seqs
- The prefix semantics remain cat-themed: "cats", "feline", "litter" appear explicitly
- Only 50 HotFlip steps used (vs 80 in Exp19) — more steps might improve further
