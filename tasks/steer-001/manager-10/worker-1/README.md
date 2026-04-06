# Worker 1: Exact Exp1 Reproduction

**Exp:** 10 | **Status:** Complete | **Script:** `exact_repro.py`

---

## Results

| Stage | This run (Exp10, exact repro) | Exp1 (original) | Exp9 (wrong hyperparams) |
|-------|-------------------------------|-----------------|--------------------------|
| Soft opt | 0.180 | 0.191 | 0.178 |
| Cosine projection | 1.398 | 1.436 | 1.442 |
| HotFlip | **0.740** | **0.740** | **1.277** |

Final prefix: `'Cats":[{ CHOOSE елныңairs prochains voice remark'`

HotFlip convergence: reached 0.740 at step 10/80, no improvement for remaining 70 steps.

---

## Key Finding

**Exp1's 0.740 result is fully reproducible.** With BATCH_SIZE=12, HF_TOPK=30,
PLACEHOLDER="SOFTPREFIX", seed=42, the exact result is reproduced on GPU 1 (44GB).

This confirms that Exp9's worse result (1.277) was caused by hyperparameter differences:
- BATCH_SIZE: 8 → 12 (more suffixes per gradient step)
- HF_TOPK: 20 → 30 (more candidates evaluated per position per HotFlip step)
- PLACEHOLDER: "PREFIX_PLACEHOLDER" → "SOFTPREFIX"

The TOPK difference is likely the dominant factor: at TOPK=20, the best token for each
position may not be in the candidate set (insufficient coverage). At TOPK=30, the better
token (e.g., "Cats") is reliably found at HotFlip step 0.

---

## Pipeline Trace (key steps)

| HotFlip step | CE | Prefix |
|---|---|---|
| After projection | 1.398 | `ꦱ coarseriſchen garantitBerita EXERCISES hindurchNoti` |
| Step 0 | 1.088 | `Cats coarseriſchen garantitBerita EXERCISES hindurchNoti` |
| Step 10 (converged) | 0.740 | `Cats":[{ CHOOSE елныңairs prochains voice remark` |

The first HotFlip step immediately finds "Cats" as the best replacement for the first
garbled token — same as Exp1 and consistent with "cats" being a high-value behavioral anchor.
