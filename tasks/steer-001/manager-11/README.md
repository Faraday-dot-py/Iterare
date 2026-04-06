# Manager 11: Exp 11 — Straight-Through Estimator Soft Opt

**Task:** steer-001 | **Status:** Complete

---

## Objective

Attack the projection bottleneck directly by training the soft prefix to minimize
the CE of its projected discrete version, using the straight-through estimator (ST).

**Background:** The fundamental bottleneck (Exp 5) is that soft opt converges to
prefixes with very low soft CE (≈0.038 at len=16) but cosine projection always
causes a large CE jump (≈1.4-1.5). This is because soft opt optimizes in continuous
space, indifferent to the discrete topology of the token embedding matrix.

**Hypothesis:** If we optimize directly for `CE(project(soft))` using an ST gradient,
the soft prefix will be pulled toward regions of embedding space where the nearest
discrete token is behaviorally effective — potentially reducing the projection gap.

---

## Method

Straight-through estimator for cosine projection:

```
forward:  use embed(argmax_token(soft))    ← discrete embedding
backward: pass gradient through to soft    ← as if projection = identity
```

At each soft opt step:
1. Project soft prefix to nearest token via cosine similarity
2. Forward pass using those discrete token embeddings
3. Backward pass: ST gradient flows to soft prefix
4. Adam update on soft prefix

This trains the soft prefix to minimize `CE(project(soft))` with an approximate gradient.

Comparison with standard approach:
- Standard: `soft_prefix → CE(soft_prefix)` → project at end → large gap
- ST:        `soft_prefix → CE(project(soft_prefix))` via ST → gap is directly penalized

**Monitoring:** Track both ST-CE (what we optimize, ≈ proj-CE) and direct soft-CE.

---

## Config

- BATCH_SIZE=12, HF_TOPK=30, PLACEHOLDER="SOFTPREFIX" (Exp1 params)
- SOFT_STEPS=300, HOTFLIP_STEPS=80, seed=42, PREFIX_LEN=8
- GPU: 0 (CUDA_VISIBLE_DEVICES=0, ~24GB free)

---

## Expected Outcomes

| Scenario | proj CE | HotFlip CE | Interpretation |
|----------|---------|------------|----------------|
| ST works | < 1.0 | < 0.740 | ST gradient is informative |
| ST = standard | ~1.4 | ~0.740 | Projection gap is fundamentally geometric |
| ST converges poorly | > 1.5 | > 1.0 | ST gradient too noisy for this problem |

---

## Results

| Stage | ST estimator (Exp11) | Standard (Exp10) |
|-------|---------------------|------------------|
| Soft CE (direct) | 0.998 | 0.180 |
| Cosine projection CE | **0.762** | **1.398** |
| HotFlip CE | **0.689** | **0.740** |

**Key result:** ST estimator reduces projection CE by 46% (1.398→0.762) and achieves
final CE of 0.689 vs standard 0.740 — beating the previous best by 7%.

ST-CE ≈ proj-CE within 1.4%: the training objective accurately tracks projection quality.

See `worker-1/README.md` for full analysis.

---

## Worker

| Worker | Script | Output |
|--------|--------|--------|
| worker-1 | `st_estimator.py` | `st_estimator_results.json` |
