# Manager 16: Exp 16 — ST with Voronoi Margin Regularization

**Task:** steer-001 | **Status:** Planned

---

## Objective

Directly attack the Voronoi oscillation problem by adding an explicit margin loss that
incentivizes the soft prefix to stay in the interior of its nearest Voronoi cell.

**Background:** ST training causes the soft prefix to oscillate across token boundaries,
giving unstable projection CE. Two approaches:
- Exp 15: LR annealing + best-prefix tracking (implicit stabilization)
- Exp 16: **Voronoi margin regularization** (explicit stabilization via a loss term)

**Voronoi margin** for position i: `cos(soft[i], nearest_token) - cos(soft[i], second_nearest_token)`
Large margin = prefix firmly inside Voronoi cell (far from boundary).
Small margin = prefix near a boundary (oscillation-prone).

**Loss:** `CE(ST-project(soft)) - λ * mean_margin`

This directly pushes the soft prefix toward the interior of good Voronoi cells.

---

## Design

Test three λ values in a single run (same seed=42, same 300-step ST + 80-step HotFlip):
- λ=0.0 (baseline = pure ST, same as Exp 11)
- λ=0.5 (moderate margin pressure)
- λ=2.0 (strong margin pressure)

Each λ is an independent run with the same initialization. Total time: ~3 × 4000s = ~12000s (within cap).

The λ=0.0 run serves as a control to measure Voronoi oscillation variance in this kernel.

---

## Expected Outcomes

| λ | Expected Effect | Projection CE | HotFlip CE |
|---|----------------|--------------|-----------|
| 0.0 | Baseline (ST only, Exp 11 repro) | ~0.76-1.13 (variable) | ~0.69-0.75 |
| 0.5 | Moderate stabilization, slight optimization tradeoff | < λ=0 variance, ~0.75-0.85 | ~0.70-0.72 |
| 2.0 | Strong stabilization, significant optimization tradeoff | Very stable but possibly higher | ~0.72+ |

Key diagnostic: do higher λ values give more consistent proj-CE across the λ=0.0 variance range?
If λ=0.5 gives consistently lower hotflip-CE than λ=0.0, margin regularization is useful.

---

## Results

*(Pending)*

---

## Worker

| Worker | Script | Output |
|--------|--------|--------|
| worker-1 | `st_voronoi_margin.py` | `voronoi_margin_results.json` (pending) |
