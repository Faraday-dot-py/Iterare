# Manager 13: Exp 13 — Extended HotFlip with Random Restarts

**Task:** steer-001 | **Status:** Planned

---

## Objective

Escape local minima in HotFlip's greedy discrete search by adding random-restart perturbations.

**Background:** Exp 10 and Exp 11 both converge at HotFlip step 10/80, wasting 70 steps.
The greedy search finds "Cats":[{ CHOOSE..." (CE=0.740) and "Cat wellnessceptre..." (CE=0.689)
respectively, then cannot improve. Both are likely local minima in discrete space.

**Hypothesis:** Random perturbation of one position + mini-HotFlip can escape the local
minimum and find a lower-CE prefix. The ST starting point (proj CE=0.762) gives a better
neighborhood to search in.

---

## Design

1. **ST soft opt** (same as Exp 11): 300 steps with ST gradient → proj CE ≈ 0.762
2. **Initial HotFlip**: 80 steps from ST projection → converges at ≈ 0.689
3. **Restart loop** (10 restarts × 30 steps):
   - Randomly perturb one position of the best prefix
   - Run 30 HotFlip steps from perturbed prefix
   - Accept if CE < best, discard otherwise

**Time estimate:**
- Soft opt: 1335s
- Initial HotFlip: ~340s (converges at step 10)
- 10 restarts × 30 steps × 34s = 10200s
- Total: ~11875s — within 14400s cap

---

## Expected Outcomes

| Scenario | Final CE | Interpretation |
|----------|---------|----------------|
| Some restarts improve | < 0.689 | Local minima exist, restarts help |
| No restarts improve | ≈ 0.689 | Local minimum is a global minimum for this approach |
| Many restarts improve | << 0.689 | Strong basin-hopping effect |

The key question: is CE=0.689 a fundamental limit for 8-token discrete prefixes
with this behavior, or is it a greedy search artifact?

---

## Results

*(Pending — submit after GPU becomes available)*

---

## Worker

| Worker | Script | Output |
|--------|--------|--------|
| worker-1 | `hotflip_extended.py` | `hotflip_extended_results.json` (pending) |
