# Worker 1: Exp 13 — Extended HotFlip with Random Restarts

**Status:** Complete | **Result:** Best CE = 0.710 (3/10 restarts improved)

---

## Results

| Stage | CE | Notes |
|-------|----|-------|
| ST soft opt (300 steps) | proj-CE = 1.129 | Voronoi oscillation — unlucky final step |
| Initial HotFlip (80 steps) | 0.752 | Converged at step 10 (same as Exp10/11) |
| After restart 2/10 | 0.745 | +0.007 improvement |
| After restart 6/10 | 0.733 | +0.012 improvement |
| After restart 9/10 | **0.710** | +0.023 improvement ← final best |

**Final prefix:** `'Vidite General trivia interactions responding Cats espesatively'`

**Compare:**

| Experiment | Proj CE | HotFlip CE | Method |
|-----------|---------|-----------|--------|
| Exp 1/10 | 1.398 | 0.740 | Standard soft opt |
| Exp 11 | **0.762** | **0.689** | Pure ST estimator |
| Exp 13 | 1.129 | **0.710** | ST + random restarts |

---

## Analysis

**Random restarts work but don't beat Exp11.** Three of 10 restarts found improvements,
demonstrating that the initial HotFlip local minimum (0.752) was NOT globally optimal.
However, starting from the worse projection (1.129 vs Exp11's 0.762) means the search
started from a disadvantaged position.

**Restart pattern:**
```
Initial HotFlip:  0.752  'Cats dro trivia Interact.""meow以下 promp'
Restart 2/10:     0.745  'Cats primary trivia Interact.""meow以下 récit'       +0.007
Restart 6/10:     0.733  'Cats primary trivia interactions."" Tone以下дото'    +0.012
Restart 9/10:     0.710  'Vidite General trivia interactions responding Cats espesatively'  +0.023
```

After restart 9, the prefix structure changed significantly ("Cats" moved to position 6
and new words "Vidite General" appeared). This suggests restart 9 found a genuinely different
local minimum, not a local improvement to the "Cats dro..." structure.

**Key insight:** The best restart result (0.710) started with a MUCH worse projection (1.129)
than Exp11 (0.762). If we apply random restarts to an Exp11-quality starting point (0.689),
we might be able to beat it. This is what Exp14 (alternating) tests.

**Voronoi oscillation hurt this experiment:** If best-prefix tracking had been used (Exp 15),
the initial ST projection would likely have been closer to Exp11's 0.762, giving restarts a
better starting point. Expected result with better initialization: ~0.68 or lower.

---

## Timing

- ST soft opt: ~1300s
- Initial HotFlip: ~2700s (80 steps × 34s)
- 10 restarts × 30 steps × 34s = 10200s
- Total: 14257s (just within 14400s cap!)

---

## Files

- `hotflip_extended.py` — experiment script
- `hotflip_extended_results.json` — raw results with restart logs
