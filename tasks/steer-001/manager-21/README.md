# Manager 21: Exp 21 — Voronoi Margin Regularization Retry (λ=0.5, λ=2.0)

**Task:** steer-001 | **Status:** Queued

## Summary

Exp16 crashed after completing only λ=0.0 (CE=0.686). This experiment runs
the two missing λ values with identical config. Directly tests whether explicit
Voronoi boundary avoidance improves upon the λ=0.0 baseline.

## Worker

| Worker | Script | Results |
|--------|--------|---------|
| worker-1 | `voronoi_margin_retry.py` | `voronoi_margin_retry_results.json` (pending) |
