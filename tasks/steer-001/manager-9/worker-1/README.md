# Worker 1: Gradient Checkpointing Baseline

**Exp:** 9 | **Status:** Complete | **Script:** `ckpt_baseline.py`

---

## Results

| Stage | This run (ckpt, seed=42) | Exp1 (batched, seed=42) | Exp8 (per-suffix, seed=42) |
|-------|--------------------------|-------------------------|----------------------------|
| Soft opt | 0.178 | 0.191 | 0.094 |
| Cosine projection | 1.442 | 1.436 | 1.469 |
| HotFlip | **1.277** | **0.740** | **1.236** |

Final prefix: `'decorate Petro faſt ویکی‌پدی FOODfjspxꯄ'`

---

## Key Finding

**Gradient checkpointing alone does NOT reproduce Exp1's 0.740.**

Despite using batched inference (all 8 suffixes in one forward pass), HotFlip CE converged to
1.277 — only marginally better than per-suffix backward (1.236), and far short of Exp1 (0.740).

HotFlip converged at step 20 and remained flat for all remaining 60 steps, indicating the
search is stuck in the same poor local minimum as the per-suffix experiments.

---

## Root Cause Analysis

Exp9 reproduced the **computational method** (batched inference) but used **different hyperparameters**
from Exp1:

| Hyperparameter | Exp9 (this) | Exp1 (baseline) |
|---------------|-------------|-----------------|
| BATCH_SIZE | **8** | **12** |
| HF_TOPK | **20** | **30** |
| PLACEHOLDER | `"PREFIX_PLACEHOLDER"` | `"SOFTPREFIX"` |

The most likely driver of Exp1's superior performance:
1. **BATCH_SIZE=12**: More suffixes per gradient step → more stable gradient signal → better soft prefix
2. **TOPK=30**: More candidates evaluated per HotFlip step → better discrete search coverage
3. **PLACEHOLDER**: Different tokenization of the prefix slot may affect the embedding landscape

Interestingly, Exp9's soft CE (0.178) is comparable to Exp1's (0.191), suggesting the soft
optimization quality is similar. The gap opens at HotFlip, pointing to TOPK=20 vs 30 as a
significant factor.

---

## Implication

The gradient method (batched vs per-suffix) is NOT the primary cause of Exp1's 0.740.
The hyperparameter differences (especially BATCH_SIZE=12 and TOPK=30) are the likely cause.

**Next experiment:** Exp10 — exact reproduction of Exp1's configuration (BATCH_SIZE=12,
TOPK=30, PLACEHOLDER="SOFTPREFIX") with gradient checkpointing for memory safety.
