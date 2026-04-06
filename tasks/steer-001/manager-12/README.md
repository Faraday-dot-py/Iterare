# Manager 12: Exp 12 — Mixed Objective (Soft CE + ST Projection CE)

**Task:** steer-001 | **Status:** Planned

---

## Objective

Test whether combining standard soft CE optimization with ST projection CE
optimization produces better HotFlip results than either alone.

**Background:**
- Standard soft opt (Exp 1): optimizes CE(soft) → 0.191, but proj CE = 1.436
- Pure ST (Exp 11): optimizes CE(project(soft)) → proj CE = 0.975 at step 50, but soft-CE = 1.126 (slow convergence)
- **Mixed objective**: optimizes α*CE(soft) + (1-α)*CE(project(soft)) via ST

**Hypothesis:** The soft CE component provides smooth gradient signal and fast
initial convergence, while the ST component directly penalizes projection gap.
Together they may find soft prefixes in "good" discrete neighborhoods faster
and more reliably than either alone.

---

## Method

```
loss = α * CE(soft_prefix) + (1-α) * CE(st_project(soft_prefix))
     = 0.5 * CE(soft_prefix) + 0.5 * CE(project(soft_prefix))  [via ST]
```

Both terms computed via batched forward pass. ST gradient for projection term.
Adam optimizer on soft prefix (float32), same as all other experiments.

---

## Config

- α = 0.5 (equal weight)
- BATCH_SIZE=12, HF_TOPK=30, PLACEHOLDER="SOFTPREFIX", seed=42
- SOFT_STEPS=300, HOTFLIP_STEPS=80
- GPU: pinned externally (submit after GPU becomes available)

---

## Expected Outcomes

If mixed > pure ST: soft CE component provides useful signal beyond pure ST
If mixed = pure ST: ST component dominates; soft CE adds noise
If mixed > standard: directly penalizing projection gap helps

Key metric: proj_ce compared to Exp1 (1.436) and Exp11 (TBD)

---

## Results

*(Pending — submit after Exp10 or Exp11 completes)*

---

## Worker

| Worker | Script | Output |
|--------|--------|--------|
| worker-1 | `mixed_objective.py` | `mixed_objective_results.json` (pending) |
