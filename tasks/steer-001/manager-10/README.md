# Manager 10: Exp 10 — Exact Exp1 Reproduction on GPU 1

**Task:** steer-001 | **Status:** Planned

---

## Objective

Determine whether Exp1's HotFlip CE=0.740 result is reproducible with its exact
configuration, and whether Exp9's worse result (1.277) was caused by hyperparameter
differences rather than gradient checkpointing or batched-vs-per-suffix inference.

**Key config differences between Exp1 and Exp9:**

| Hyperparameter | Exp1 | Exp9 | This (Exp10) |
|---------------|------|------|--------------|
| BATCH_SIZE | 12 | 8 | **12** |
| HF_TOPK | 30 | 20 | **30** |
| PLACEHOLDER | "SOFTPREFIX" | "PREFIX_PLACEHOLDER" | **"SOFTPREFIX"** |
| GPU | 0 | 0 | **1** (44GB free) |
| Gradient ckpt | No | Yes | No |

---

## Hypothesis

The 1.277 vs 0.740 gap between Exp9 and Exp1 is primarily explained by:
1. **TOPK=20 vs 30**: Fewer HotFlip candidates evaluated per position
2. **BATCH_SIZE=8 vs 12**: Fewer suffixes in gradient averaging
3. **PLACEHOLDER**: Different tokenization may affect prefix embedding landscape

If this experiment reproduces ~0.740, the hyperparameters were the cause.
If it does not, some other factor (random seeds, environment, etc.) explains the gap.

---

## Design

- Script: `worker-1/exact_repro.py`
- GPU: 1 (via `CUDA_VISIBLE_DEVICES=1`, full 44GB, no other processes)
- No gradient checkpointing (matching Exp1 exactly)
- BATCH_SIZE=12, HF_TOPK=30, PLACEHOLDER="SOFTPREFIX", seed=42

---

## Results

*(Pending)*

---

## Worker

| Worker | Script | Output |
|--------|--------|--------|
| worker-1 | `exact_repro.py` | `exact_repro_results.json` (pending) |
