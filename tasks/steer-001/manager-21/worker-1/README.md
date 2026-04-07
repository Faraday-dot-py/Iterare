# Exp 21: Voronoi Margin Regularization Retry (λ=0.5, λ=2.0)

**Result: λ>0 HURTS — λ=0.5: CE=0.746, λ=2.0: CE=0.843 (both far worse than λ=0.0 baseline 0.686)**

## Method

ST estimator + Voronoi margin regularization:
`Loss = CE(ST-project(soft)) - λ * mean_i(margin_i)`

Same config as Exp16: PREFIX_LEN=8, seed=42, SOFT_STEPS=300, HOTFLIP_STEPS=80,
HF_TOPK=30, constant LR=0.01, BATCH_SIZE=12.

## Results

| λ | Best ST-CE | Proj CE | HotFlip CE | Final Prefix |
|---|-----------|---------|------------|-------------|
| 0.0 | 0.877 | 0.877 | **0.686** | `Keeping Cats'];?>Only responding Cat Ley trivia` (Exp16) |
| 0.5 | 1.436 | 1.436 | 0.746 | `Cats<start_of_turn> talking Answer must Replies pyridine<unused56>` |
| 2.0 | 1.549 | 1.549 | 0.843 | `<unused56> laughs<unused56>continue feline CAT<unused56><unused56>` |

## Key Findings

- **Voronoi margin regularization is strictly harmful** — both λ>0 produce substantially worse results
- λ=0.5: proj-CE 1.436 vs 0.877 for λ=0.0 — a 0.56 CE increase in projection quality
- λ=2.0: proj-CE 1.549 — even further degraded; best-prefix tracking found no improvement
- Both runs ended up with `<unused56>` tokens (padding artifacts), suggesting the optimizer
  was pushed into degenerate regions of embedding space where margin is easy to maximize
  but behavioral alignment is poor

## Interpretation

The margin loss creates a **competing objective** against ST-CE:
- ST-CE wants to find soft prefixes that project cleanly to behaviorally-aligned tokens
- Margin wants to push the soft prefix to the interior of its current Voronoi cell
- These goals conflict: the interior of the Voronoi cell for `<start_of_turn>` (margin easy)
  is not near any behaviorally useful token

At λ=0.0, the optimizer is free to find trajectories that cross Voronoi boundaries many times
to eventually land in a good cell. Penalizing boundary proximity prevents this exploration.

**Conclusion:** Voronoi margin regularization is not a useful technique for this task.
The VQ commitment loss approach (Exp22) may fare better since it pulls toward the nearest
token rather than trying to stay interior to the current cell.
