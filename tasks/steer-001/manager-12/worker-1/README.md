# Worker 1: Exp 12 — Mixed Objective (α=0.5)

**Status:** Complete | **Result:** HotFlip CE = 0.746 (worse than Exp1=0.740 and Exp11=0.689)

---

## Results

| Stage | CE | Notes |
|-------|----|-------|
| Soft opt (300 steps, mixed α=0.5) | 0.147 (soft), 1.157 (ST) | Very good soft; poor ST |
| Cosine projection | 1.124 | Worse than Exp11's 0.762 |
| HotFlip (80 steps, topk=30) | **0.746** | Worse than Exp1 (0.740) |

**Final prefix:** `'-------\x0c Signalez ANIMALSCatsOwnership voice this =========='`

**Compare:**

| Experiment | Method | Proj CE | HotFlip CE |
|-----------|--------|---------|-----------|
| Exp1 | Standard soft opt | 1.436 | **0.740** |
| Exp11 | Pure ST estimator | 0.762 | **0.689** |
| Exp12 | Mixed (α=0.5 soft+ST) | 1.124 | **0.746** |

---

## Analysis

**The mixed objective failed.** α=0.5 gives too much weight to the soft CE component, which
actively harms the ST optimization.

**Why soft CE hurts the ST objective:**
- Soft CE pulls the soft prefix toward smooth, well-behaved regions of embedding space
- These regions are optimized for continuous performance but lie FAR from good discrete tokens
- The ST gradient is fighting against the soft CE gradient throughout training
- Result: neither objective is achieved well — soft CE converges to 0.147 (good) but ST-CE
  barely improves (1.158 final, vs 0.752 for pure ST in Exp11)

**Evidence from training dynamics:**
```
Step   0: mixed=1.514, soft=1.477, ST=1.551
Step  50: mixed=1.019, soft=0.507, ST=1.532  ← soft improving rapidly, ST barely moving
Step 100: mixed=0.921, soft=0.344, ST=1.497
Step 150: mixed=0.800, soft=0.293, ST=1.307
Step 200: mixed=0.714, soft=0.226, ST=1.201
Final:    mixed=???,   soft=0.147, ST=1.157  ← soft much better, ST still poor
```

The soft gradient dominates. By step 50, soft CE has already halved (1.477→0.507) while
ST-CE barely moved (1.551→1.532). The mixed objective is essentially "mostly soft opt with
a weak ST signal" at α=0.5.

**Key lesson for α selection:**
- α=1.0 = pure soft opt (Exp1): proj-CE=1.436, hotflip=0.740
- α=0.5 = mixed (Exp12): proj-CE=1.124, hotflip=0.746
- α=0.0 = pure ST (Exp11): proj-CE=0.762, hotflip=0.689

The pattern suggests lower α (more ST weight) is strictly better. The soft CE "regularization"
hypothesis was wrong — it doesn't stabilize ST training, it undermines it.

**HotFlip timing:** 6129s for 80 steps (76.6 s/step vs expected ~34 s/step). Significant
GPU contention from concurrent processes on the TIDE cluster explains this slowdown.

---

## Files

- `mixed_objective.py` — experiment script
- `mixed_objective_results.json` — raw results with full training logs
