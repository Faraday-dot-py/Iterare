# Manager 15: Exp 15 — ST with Cosine LR Annealing + Best-Prefix Tracking

**Task:** steer-001 | **Status:** Planned

---

## Objective

Fix the Voronoi oscillation problem that causes high variance in ST projection quality.

**Background:** Exp 11 and Exp 13 used the exact same ST algorithm (same seed=42, same 300
steps) but got radically different projection CE:
- Exp 11: proj CE = **0.762** (lucky — final step happened to be in a stable Voronoi cell)
- Exp 13: proj CE = **1.129** (unlucky — final step was at a Voronoi boundary)

This 0.367 difference in projection CE is purely due to where the training curve happened
to end. The ST gradient causes Voronoi oscillation: the soft prefix crosses token boundaries,
causing discrete projections to flip back and forth.

**Two complementary fixes:**
1. **Cosine LR annealing**: LR decays 0.01 → 0.001 over 300 steps. Early high LR enables
   exploration; late low LR causes the soft prefix to settle stably within a Voronoi cell.
2. **Best-prefix tracking**: saves the soft prefix with the lowest ST-CE seen during
   training. Projects this best snapshot rather than the final (potentially oscillating) step.

---

## Design

- Same pipeline as Exp 11: 300 ST soft opt steps + 80 HotFlip steps
- Adds cosine LR schedule from 0.01 → 0.001
- Tracks best soft prefix (lowest ST-CE); uses it for projection and HotFlip
- Reports both best-prefix projection CE and final-step CE for comparison

**Time estimate:** Same as Exp 11 (~4000s ≈ 67 min)

---

## Expected Outcomes

| Scenario | Projection CE | Interpretation |
|----------|--------------|----------------|
| Best-prefix helps significantly | < 1.129 | Voronoi oscillation is main source of variance |
| Best-prefix ≈ final-step | Similar | Annealing stabilizes late training |
| Better than Exp 11 | < 0.762 | Both fixes together reliably beat lucky Exp 11 |
| Worse than Exp 11 | > 0.762 | Exp 11 was optimally lucky, fixes don't add value |

Key diagnostic: compare `projection_ce_best` vs `projection_ce_final_step` to directly
measure how much oscillation was hurting results.

---

## Results

*(Pending)*

---

## Worker

| Worker | Script | Output |
|--------|--------|--------|
| worker-1 | `st_annealed.py` | `st_annealed_results.json` (pending) |
