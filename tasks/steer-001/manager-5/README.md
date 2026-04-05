# Manager 5: Exp 5 — Prefix-Length Ablation

**Task:** steer-001 | **Status:** Complete

---

## Objective

Test whether increasing the number of prefix tokens reduces the soft→discrete CE gap.

**Research question:** Does more prefix length give enough capacity to close the
discrete approximation error?

---

## Design

- Reference prefix: `"Talk only about cats."`
- PREFIX_LEN ∈ {4, 8, 12, 16}
- Full pipeline: 300 soft opt steps → cosine project → 80 HotFlip steps
- Fixed seed for reproducibility across lengths

**Hypothesis:** Each discrete token contributes a fixed behavioral signal. More tokens
provide more degrees of freedom to represent the target behavior, so the gap between
soft CE and discrete CE should decrease with prefix length. If confirmed, this suggests
a compute/accuracy tradeoff in token budget.

**Alternative hypothesis:** The bottleneck is in the cosine projection per-token, which
doesn't improve with more tokens. If so, the gap scales linearly and longer prefixes
don't help proportionally.

---

## Results

Projection CE stays ~1.4-1.5 regardless of prefix length (4-16 tokens) while
soft CE drops 0.227→0.038. Confirms projection is the fundamental bottleneck.
See `worker-1/README.md` for full analysis.

---

## Worker

| Worker | Script | Output |
|--------|--------|--------|
| worker-1 | `prefixlen.py` | `prefixlen_results.json` (pending) |
