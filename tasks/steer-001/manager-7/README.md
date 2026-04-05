# Manager 7: Exp 7 — Gumbel-Softmax End-to-End Discrete Training

**Task:** steer-001 | **Status:** Planned

---

## Objective

Test whether training directly in the discrete token distribution space (via
Gumbel-Softmax relaxation) outperforms the standard two-stage soft→project pipeline.

**Research question:** Can end-to-end discrete-aware training close the soft→discrete
gap without a separate projection step?

---

## Design

- Reference prefix: `"Talk only about cats."`
- PREFIX_LEN = 8
- Parameterization: logits L ∈ R^{PREFIX_LEN × V} (one logit vector per token position)
- Forward pass: `prefix_embed = gumbel_softmax(L, τ) @ W_embed`
- Temperature schedule: τ annealed exponentially from 2.0 → 0.1 over 500 steps
- After training: discrete prefix = argmax(L)
- HotFlip refinement (50 steps) on the argmax result

**Key property:** Unlike standard soft opt (which optimizes in embedding space), the
Gumbel-Softmax gradient flows through the discrete token distribution. As τ decreases,
the distribution concentrates on individual tokens, so the learned logits directly
encode which token at each position best supports the target behavior.

**Comparison baseline:** Exp 1 HotFlip CE = 0.740

**Expected outcome:** If the discrete constraint helps, Gumbel training should achieve
lower argmax CE than cosine projection from soft opt. If not, the gap may be
fundamental to the representation mismatch rather than an optimization artifact.

---

## Results

*(Pending — will run after Exp 5/6 complete)*

---

## Worker

| Worker | Script | Output |
|--------|--------|--------|
| worker-1 | `gumbel_softmax.py` | `gumbel_results.json` (pending) |
