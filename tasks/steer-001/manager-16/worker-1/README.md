# Exp 16: ST + Voronoi Margin Regularization

**Result: PARTIAL — only λ=0.0 completed before crash | HotFlip CE = 0.6861**

## Method

ST estimator with an explicit Voronoi margin regularization loss:
`CE(ST-project(soft)) - λ * mean_margin`

Three λ values planned: 0.0, 0.5, 2.0 (seed=42, 300-step ST + 80-step HotFlip, TOPK=30).
Script crashed mid-λ=0.5 run; only λ=0.0 completed.

## Results

| λ | Best ST-CE | Proj CE (best) | HotFlip CE |
|---|-----------|---------------|-----------|
| 0.0 | 0.8769 | 0.8769 | **0.6861** |
| 0.5 | — (crash) | — | — |
| 2.0 | — (crash) | — | — |

Final prefix (λ=0.0): `"Keeping Cats'];?>Only responding Cat Ley trivia"`

## Key Finding

λ=0.0 with float32 similarity computations and best-prefix tracking obtained **0.6861**,
beating the previous SOTA of 0.689 (Exp11). This confirmed the Exp16 prelim result.

The λ=0.0 run also achieved a larger ST→HotFlip improvement (0.877→0.686, Δ=0.191)
than Exp11 (0.762→0.689, Δ=0.073), supporting the hypothesis that **basin quality
dominates over projection CE**.

λ=0.5 and λ=2.0 data lost to crash. Voronoi margin regularization hypothesis remains
untested.
