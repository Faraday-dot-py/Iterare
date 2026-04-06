# Worker 1: Straight-Through Estimator

**Exp:** 11 | **Status:** Complete | **Script:** `st_estimator.py`

---

## Results

| Stage | ST estimator (Exp11) | Standard (Exp10/Exp1) |
|-------|---------------------|----------------------|
| Soft CE (direct) | 0.998 | 0.180 |
| ST-CE (training objective) | 0.752 | N/A |
| Cosine projection CE | **0.762** | **1.398** |
| HotFlip CE | **0.689** | **0.740** |

ST-CE tracking accuracy: 0.752 vs actual proj CE 0.762 (error < 1.4%) — ST loss is an
accurate proxy for post-projection CE.

Final prefix: `'Cat wellnessceptre人工语言 reply with cats'`

HotFlip convergence: reached 0.689 at step 10/80, no improvement for remaining 70 steps.

---

## Key Findings

### 1. ST estimator reduces projection CE from 1.398 to 0.762 (−46%)

The fundamental bottleneck (projection) is directly addressed. Standard soft opt
produces prefixes optimized in continuous space but far from good discrete tokens.
ST estimator trains the soft prefix to find regions where the nearest discrete token
is behaviorally effective.

### 2. ST-CE ≈ actual projection CE (within 1.4%)

The straight-through approximation is accurate: the loss being optimized (ST-CE)
closely tracks the actual post-projection CE. This means the optimization is converging
toward the right objective, not just a noisy proxy.

### 3. Final HotFlip CE: 0.689 vs standard 0.740 (−7%)

Despite the massive improvement in projection CE (−46%), the final HotFlip CE only
improved by 7%. This is because:
- Both pipelines quickly converge to a local minimum in discrete space
- HotFlip's greedy search finds the best local improvement in the neighborhood
- ST starts in a better neighborhood, so HotFlip needs fewer steps to converge
- But both hit a similar quality ceiling for the "cats" behavior prefix at length=8

### 4. ST training shows oscillation in ST-CE

ST-CE trajectory: 1.551→0.975→0.860→0.942→0.872→0.880→0.752
The oscillation (step 150: 0.942 > step 100: 0.860) is characteristic of ST estimators
when the optimization crosses Voronoi cell boundaries. The overall trend is downward
despite local oscillations, converging to 0.752 by step 300.

### 5. Soft CE increases during ST training

The soft prefix moves toward "discrete-compatible" regions, away from the continuous
CE optimum (0.191). Final direct soft CE is 0.998 vs 0.180 for standard. This tradeoff
is expected: better discrete performance at the cost of worse continuous performance.

---

## Pipeline Trace (key steps)

| HotFlip step | CE | Prefix |
|---|---|---|
| After projection | 0.762 | `' veterinary wellnessユ TEXT语言 replySlightly cats'` |
| Step 0 | 0.722 | `' veterinary wellnessユ TEXT语言 reply with cats'` |
| Step 10 (converged) | 0.689 | `' Cat wellnessceptre人工语言 reply with cats'` |

Note: The projected prefix already contains semantically relevant tokens ("veterinary",
"cats") compared to standard's garbage projection. HotFlip refines "Slightly" → "with"
and "veterinary" → "Cat" to improve CE further.

---

## Implications

The ST estimator demonstrates that:
1. The projection bottleneck is largely addressable: a 46% reduction is achievable
2. Better discrete starting points lead to better HotFlip convergence
3. The remaining gap (0.689 vs ideal <0.1) is likely a HotFlip local minimum issue
4. Future work: mixed objective (Exp12) may combine soft CE stability with ST precision
