# Manager 22: Exp 22 — VQ-Style Commitment Loss

**Task:** steer-001 | **Status:** Complete

## Summary

VQ-VAE-inspired commitment loss as soft→discrete bridge regularizer.
Loss = CE(ST-project(soft)) + β * mean_i(1 - cos(soft_i, stop_grad(embed(argmax_i))))
Tests β=0.1, 0.5, 2.0 on PREFIX_LEN=8, seed=42.

## Worker

| Worker | Script | Results |
|--------|--------|---------|
| worker-1 | `vq_commitment.py` | `vq_commitment_results.json` ✓ |
