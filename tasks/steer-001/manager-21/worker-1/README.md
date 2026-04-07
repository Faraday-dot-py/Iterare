# Exp 21: Voronoi Margin Regularization Retry (λ=0.5, λ=2.0)

**Status:** Queued | **Results:** Pending

## Method

ST estimator + Voronoi margin regularization:
`Loss = CE(ST-project(soft)) - λ * mean_i(margin_i)`
where `margin_i = cos(soft_i, nearest_tok) - cos(soft_i, 2nd_nearest_tok)`

Same config as Exp16: PREFIX_LEN=8, seed=42, SOFT_STEPS=300, HOTFLIP_STEPS=80,
HF_TOPK=30, constant LR=0.01, BATCH_SIZE=12.

λ=0.0 already done in Exp16 (CE=0.686). This run covers λ=0.5 and λ=2.0 only.

## Expected Findings

If λ>0 consistently improves over λ=0.0:
- Voronoi boundary avoidance is a useful regularizer
- The approach can be combined with PREFIX_LEN=16 (Exp19 SOTA path)

If λ>0 hurts or doesn't help:
- Boundary avoidance is not the bottleneck at this scale
- The margin geometry doesn't align with what HotFlip needs

## Comparison Targets

| Method | Proj CE | HotFlip CE |
|--------|---------|------------|
| Exp11 (ST, β=0 baseline) | 0.762 | 0.689 |
| Exp16 λ=0.0 (fp32 sims) | 0.877 | 0.686 |
| Exp19 (len=16 SOTA) | 0.905 | **0.679** |
| Exp21 λ=0.5 | — | — |
| Exp21 λ=2.0 | — | — |
