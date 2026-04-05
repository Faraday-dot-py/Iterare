# Manager 3: Exp 3 — Layer-Depth Projection Ablation

**Task:** steer-001 | **Status:** Running

---

## Objective

Test whether projecting the soft prefix to discrete tokens using internal states at
*earlier* transformer layers yields better behavioral fidelity (lower CE) than the
standard late-layer projection.

**Research question:** Does earlier layer = better post-HotFlip CE?

---

## Design

- Reference prefix: `"Talk only about cats."`
- Soft opt objective: **state-matching** (cosine similarity of hidden states at target layer)
  rather than behavioral CE — the prefix is optimized to match the reference prefix's
  internal representation at each target layer.
- Target layers: L ∈ {4, 10, 16, 22} out of 26 total layers
- PREFIX_LEN = 6, N_SOFT_STEPS = 300, N_HOTFLIP_STEPS = 50
- Naturalness score: fraction of projected tokens that are standard ASCII words

**Key design decision:** The soft opt here uses a *state-matching* objective rather than
CE, because we want to discover what each layer "knows" about the prefix behavior,
not just which layer gives the best soft-opt CE.

---

## Results

*(Pending — experiment running on TIDE GPU 0)*

---

## Worker

| Worker | Script | Output |
|--------|--------|--------|
| worker-1 | `layerablation.py` | `layerablation_results.json` (pending) |
