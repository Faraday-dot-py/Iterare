# Exp 22: VQ-Style Commitment Loss

**Status:** Queued | **Results:** Pending

## Method

VQ-VAE commitment loss adapted to prompt optimization:

`Loss = CE(ST-project(soft)) + β * mean_i(1 - cos(soft_i, stop_grad(embed(argmax_i))))`

The commitment loss maximizes cosine alignment between each soft prefix vector and
its nearest token embedding (stop-gradient, so we only pull soft toward the token,
not vice versa). This differs from Voronoi margin (Exp21):

| Method | Signal | Geometry |
|--------|--------|---------|
| Voronoi margin (Exp16/21) | `cos(soft, nearest) - cos(soft, 2nd_nearest)` | Relative: distance to boundary |
| VQ commitment (Exp22) | `cos(soft, nearest)` | Absolute: alignment with nearest token |

Three β values: 0.1 (weak), 0.5 (moderate), 2.0 (strong).
β=0.0 is Exp11/16 (CE=0.686–0.689, not re-run).

Config: PREFIX_LEN=8, seed=42, SOFT_STEPS=300, HOTFLIP_STEPS=80, HF_TOPK=30,
constant LR=0.01, BATCH_SIZE=12.

## Research Questions

1. Does commitment regularization reduce Voronoi oscillation (as measured by proj-CE variance)?
2. Does better alignment with nearest tokens (lower commit loss) translate to better HotFlip CE?
3. Is there an optimal β, or does any β>0 hurt by fighting the CE objective?

## Comparison Targets

| Method | Proj CE | HotFlip CE |
|--------|---------|------------|
| Exp11 (pure ST) | 0.762 | 0.689 |
| Exp16 λ=0.0 (fp32) | 0.877 | 0.686 |
| Exp19 len=16 SOTA | 0.905 | **0.679** |
| Exp22 β=0.1 | — | — |
| Exp22 β=0.5 | — | — |
| Exp22 β=2.0 | — | — |
