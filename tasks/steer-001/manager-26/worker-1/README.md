# Exp 26: Multi-seed Sweep at PREFIX_LEN=16 (seeds 0, 1, 2)

**Status:** Queued | **Results:** Pending

## Method

- Seeds: [0, 1, 2] (seed=42 already known: CE=0.679 from Exp19)
- Per seed: SOFT_STEPS=250, HOTFLIP_STEPS=25, ST+cosine+best-prefix
- HF_TOPK=50, BATCH_SIZE=12, PREFIX_LEN=16
- Best prefix tracked across all seeds

## Research Question

Is seed=42's advantage (0.679 vs best-of-4 seeds = 0.694 at len=8)
preserved at len=16? Three outcomes:

1. **Seeds 0-2 also reach ~0.679**: seed=42 was lucky at len=8; longer prefixes
   make all seeds competitive. → Multi-seed runs at longer lengths are worthwhile.

2. **Seeds 0-2 are 0.685-0.695**: seed=42 is still privileged at len=16.
   → The initialization advantage is a consistent structural property.

3. **Seeds 0-2 are worse (0.695+)**: Confirms seed=42 is unusually good at all lengths.
   → Seed selection matters even more for longer prefixes.

## Prior Data (len=8 reference)

| Seed | Exp17 CE (len=8, TOPK=50) |
|------|--------------------------|
| 42 | 0.678 (close to Exp11 0.689) |
| 0-4 best | 0.694 |

## Timing Estimate

~6h: 3 × (1000s soft + 25 × 249s HotFlip) = 3 × 7225s ≈ 21675s
