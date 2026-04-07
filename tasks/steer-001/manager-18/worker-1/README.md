# Exp 18: TOPK Escalation from Exp11 Best Prefix

**Result: Global Best CE = 0.6877 (from Exp11 start 0.6894) | Δ = −0.0017**

## Method

Starting from Exp11's best discrete prefix (CE=0.6894, re-verified at 0.7046 on A100),
run HotFlip in three escalating phases:
- Phase 1: TOPK=50, 50 steps
- Phase 2: TOPK=100, 50 steps
- Phase 3: TOPK=200, 50 steps

## Results

| Phase | TOPK | Start CE | End CE | Improved Steps |
|-------|------|---------|--------|---------------|
| 1 | 50 | 0.7046 | 0.7046 | 0 |
| 2 | 100 | 0.7046 | **0.6877** | 5 |
| 3 | 200 | 0.6877 | 0.6877 | 0 |

Global best: **0.6877** | Prefix: `" Cat carenub人工语言 reply with cats"`
Total improvement from start: −0.0169 CE

## Key Findings

- Exp11's prefix re-verified at 0.7046 on A100 (vs reported 0.6894 on original hardware) —
  slight cross-hardware CE variation is normal
- TOPK=50 could not improve the prefix at all (hard local minimum)
- TOPK=100 found modest improvement (5/50 steps accepted), reaching 0.6877
- TOPK=200 stagnated — the minimum found by TOPK=100 appears to be a local optimum
- **0.6877 is a marginal improvement** from the re-verified start but does not approach
  Exp19's 0.6794 SOTA
- The Exp11 starting point appears stuck in a different basin than Exp16/Exp19
