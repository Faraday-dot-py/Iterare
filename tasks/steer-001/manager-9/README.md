# Manager 9: Exp 9 — Gradient Checkpointing for Batched Inference

**Task:** steer-001 | **Status:** Planned

---

## Objective

Reproduce Exp 1's batched gradient computation (CE=0.740) while avoiding OOM.

**Background:** Exp 8 revealed that the per-suffix backward method (used in Exp 5-8
due to OOM) produces systematically worse HotFlip CE (~1.24-1.30) compared to Exp 1's
batched inference (0.740). The difference is in gradient normalization:
- **Batched:** All 8 suffixes in one forward pass → batch-normalized gradients
- **Per-suffix:** 8 separate backward calls → per-suffix normalization → different landscape

**Research question:** Can gradient checkpointing on the batched forward pass reproduce
Exp 1's CE=0.740 without OOM, even with ~24GB available GPU memory?

---

## Design

- Reference prefix: `"Talk only about cats."`
- Seed: 42 (to directly reproduce Exp 1)
- PREFIX_LEN: 8 (same as Exp 1)
- N_SOFT_STEPS: 300, N_HOTFLIP_STEPS: 80 (same as Exp 1)
- Gradient checkpointing applied to:
  - Soft opt: `torch.utils.checkpoint.checkpoint_sequential` on transformer layers
  - HotFlip gradient pass: same
- Compare with Exp 1 result: soft=0.191, proj=1.436, HotFlip=0.740

**Expected outcome:** If checkpointing preserves the batch normalization, we should
reproduce ~0.740. If not, it indicates the OOM constraint requires a different fix.

---

## Results

*(Pending)*

---

## Worker

| Worker | Script | Output |
|--------|--------|--------|
| worker-1 | `ckpt_baseline.py` | `ckpt_baseline_results.json` (pending) |
