# Exp 22: VQ-Style Commitment Loss

**Result: β>0 HURTS — β=0.1: CE=0.794, β=0.5/2.0: CE=0.843 (all worse than β=0.0 baseline 0.686)**

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
| Exp22 β=0.1 | 1.104 | 0.794 |
| Exp22 β=0.5 | 1.548 | 0.843 |
| Exp22 β=2.0 | 1.548 | 0.843 |

## Final Prefix Texts

| β | Final Prefix |
|---|-------------|
| 0.1 | ` meow continuouslyБиография脚注の使い方 Cats REPLIESUS]:` |
| 0.5 | `<unused56> laughs<unused56>continue feline CAT<unused56><unused56>` |
| 2.0 | `<unused56> laughs<unused56>continue feline CAT<unused56><unused56>` |

## Key Findings

- **VQ commitment regularization is strictly harmful** — all β>0 produce worse results than baseline
- β=0.5 and β=2.0 converge to identical `<unused56>` degenerate prefixes (same pattern as Voronoi margin λ=2.0)
- β=0.1 avoids degenerate collapse but still worsens CE from 0.686 → 0.794
- The HotFlip step is hurt by low-quality soft prefix trajectories: proj-CE 1.104 (β=0.1) vs 0.877 (baseline) leaves HotFlip starting from a worse position

## Interpretation

Both regularization approaches tested (Voronoi margin Exp21, VQ commitment Exp22) show
the same failure mode: any auxiliary loss that pulls soft embeddings toward specific token
regions competes with the CE objective's need to explore the embedding space.

The ST estimator works precisely because it allows the soft optimizer to take trajectories
that cross Voronoi boundaries — the gradient flows through the projection, letting the
optimizer discover paths to low-CE token sequences. Adding any loss that penalizes such
exploration degrades the final result.

**Conclusion:** Regularization-based approaches to stabilizing soft→discrete projection
are not productive for this task. The ST estimator (unregularized) remains the best method.
Future directions should focus on diverse initialization (multi-seed) or expanded prefix length.
