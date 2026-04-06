# Manager 14: Exp 14 — Alternating ST+HotFlip Optimization

**Task:** steer-001 | **Status:** Planned

---

## Objective

Break the HotFlip local minimum by alternating between continuous (ST soft opt) and
discrete (HotFlip) optimization phases.

**Background:** Both Exp 10 and Exp 11 converge at HotFlip step 10/80. The greedy
discrete search finds "Cats":[{ CHOOSE..." (CE=0.740) or " Cat wellnessceptre..."
(CE=0.689) and cannot improve further. These are deep local minima in discrete token
space — no single-token swap improves CE.

**Key insight:** HotFlip can make hard token substitutions that the continuous ST gradient
cannot (because the gradient is zero once soft_prefix is already in the right Voronoi cell).
Conversely, ST can explore the continuous neighborhood of a discrete point, potentially
finding a *different* Voronoi cell whose discrete embedding is better.

**Hypothesis:** Cycling ST→HotFlip→ST→HotFlip... allows each phase to prepare better
initialization for the next, escaping local minima that neither approach could alone.

---

## Design

```
Round 0:
  1. ST soft opt (300 steps, seed=42) → proj CE ≈ 0.762
  2. HotFlip (80 steps) → CE ≈ 0.689 (converges ~step 10)

Rounds 1–4:
  3. Warm-start soft prefix from best HotFlip discrete embedding
  4. ST soft opt (100 steps) — gradient signal from neighborhood of discrete attractor
  5. Project → HotFlip (80 steps)
  6. Accept if CE improves; keep global best
```

The warm-start at step 3 means the soft prefix begins exactly at the best HotFlip token
embeddings. The ST gradient then pushes the soft prefix toward *neighboring* Voronoi cells
whose projected tokens have lower CE — essentially using gradient to guide the search.

**Time estimate:**
- Round 0: 1335s + 340s = 1675s
- Rounds 1–4: 450s + 340s = 790s × 4 = 3160s
- Total: ~4835s ≈ 80 min (well within 14400s cap)

---

## Expected Outcomes

| Scenario | Interpretation |
|----------|----------------|
| Later rounds improve over 0.689 | Alternating escapes local minima; hybrid better than pure discrete search |
| No round improves after round 0 | CE=0.689 is a deep basin that ST can't navigate away from |
| Round 1 improves but later rounds plateau | One re-entry is enough; more rounds don't help |

---

## Results

*(Pending)*

---

## Worker

| Worker | Script | Output |
|--------|--------|--------|
| worker-1 | `alternating_opt.py` | `alternating_opt_results.json` (pending) |
