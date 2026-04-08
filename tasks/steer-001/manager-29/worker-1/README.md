# Exp 29: PREFIX_LEN=32, Full HotFlip Budget (80 steps)

**Status:** Queued | **Results:** Pending

## Method

Identical to Exp25 but HOTFLIP_STEPS=80 instead of 35.
Same seed=42, SOFT_STEPS=300, ST+cosine+best-prefix.

## Research Question

Exp25 (35 steps): CE=0.632, still improving at last step.
How much does the full 80-step budget improve over 35 steps?

Expected outcomes:
- **CE improves to ~0.60–0.62**: HotFlip budget was the main bottleneck in Exp25.
  → Run all future experiments with 80 steps minimum.
- **CE improves marginally (~0.625)**: Early HotFlip steps capture most of the gain;
  diminishing returns set in after ~35 steps at len=32.

## Timing

~5.5h: soft ~700s + 80 × 237s HF ≈ 19660s

## Comparison

| Exp | len | HotFlip steps | CE |
|-----|-----|-------------|-----|
| 25 | 32 | 35 (budget-limited) | 0.632 |
| **29** | **32** | **80 (full)** | **—** |
| 19 | 16 | 80 | 0.679 |
