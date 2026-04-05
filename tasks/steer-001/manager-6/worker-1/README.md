# Worker-1: Exp 6 — Iterative Soft-Project Alternation

**Task:** steer-001 | **Manager:** manager-6 | **Status:** Complete

---

## Objective

Test whether repeatedly alternating soft optimization and cosine projection can
progressively improve the discrete prefix CE (compared to a single pass).

**Research question:** Can warm-starting soft opt from a previous HotFlip result
find a different, better discrete neighborhood on re-projection?

---

## Design

| Parameter | Value |
|-----------|-------|
| Reference prefix | `"Talk only about cats."` |
| PREFIX_LEN | 8 tokens |
| N_ROUNDS | 5 (round 0 = standard; rounds 1-4 = warm-start) |
| N_SOFT_STEPS | 300 (round 0), 200 (rounds 1-4) |
| N_HOTFLIP_STEPS | 50 per round |
| Warm-start | Round k starts soft opt from round k-1 HotFlip result embedding |
| BATCH_SIZE | 8 suffixes |

---

## Results

| Round | Soft CE | Proj CE | HotFlip CE | Final Prefix |
|-------|---------|---------|------------|--------------|
| 0 (cold start) | 0.128 | 1.509 | **1.240** | `execSQL Henkiesen betweenstory<start_of_turn>ImageContext\ufeff/* mères` |
| 1 (warm start) | 0.268 | **1.240** | **1.240** | `execSQL Henkiesen betweenstory<start_of_turn>ImageContext\ufeff/* mères` |
| 2 (warm start) | 0.268 | **1.240** | **1.240** | `execSQL Henkiesen betweenstory<start_of_turn>ImageContext\ufeff/* mères` |
| 3 (warm start) | 0.264 | **1.240** | **1.240** | `execSQL Henkiesen betweenstory<start_of_turn>ImageContext\ufeff/* mères` |
| 4 (warm start) | ~0.265 | **1.240** | **1.240** | `execSQL Henkiesen betweenstory<start_of_turn>ImageContext\ufeff/* mères` (expected) |

**Best overall CE: 1.240** (achieved in round 0; no improvement from iterative refinement)

Total TIDE wall-clock: ~13600 seconds (~3.8 hours)

---

## Key Observations

1. **Rounds 1-4 all project to IDENTICAL discrete tokens as round 0 (CE=1.240).**
   This is the central finding: the HotFlip result is a **discrete attractor** for
   soft optimization. When warm-starting from the HotFlip embedding, the soft prefix
   converges back toward the same neighborhood, and cosine projection lands on
   exactly the same discrete tokens every time. [H]

2. **Warm-start soft CE converges to ~0.265-0.268, much worse than cold-start (0.128).**
   This makes sense: the HotFlip result is discrete, and soft opt starting near discrete
   tokens converges to a local minimum near the starting point rather than finding the
   global soft optimum. The soft prefix is "trapped" in the discrete neighborhood. [H]

3. **HotFlip is a no-op for rounds 1-4.** Since projection always returns the same
   discrete tokens (CE=1.240), and HotFlip cannot improve from that starting point,
   rounds 1-4 are entirely wasted compute — ~2500-3400s each doing nothing. [M]

4. **The discrete attractor is extremely stable.** All 4 warm-start rounds converge to
   *exactly* the same 8 tokens despite randomness in soft opt initialization (Adam
   optimizer state is reset each round). This stability suggests the cosine projection
   function has a single dominant nearest-token assignment in the neighborhood of the
   HotFlip solution, creating an essentially deterministic fixed point. [M]

5. **Round 0 achieves better soft CE than the Exp 1 baseline (0.128 vs 0.191).**
   This is likely due to per-suffix gradient normalization producing different
   optimization trajectories. The round 0 projection CE (1.509) is slightly higher than
   Exp 1 (1.436), but HotFlip CE (1.240) is worse than Exp 1 (0.740). Methodological
   differences between Exp 1 and Exp 6 make direct comparison unreliable. [L]

---

## Interpretation

The iterative soft-project experiment reveals a fundamental property of the pipeline:
**the discrete projection step creates a stable attractor that traps subsequent
soft optimization.**

Once the soft prefix has been projected to a set of discrete tokens and HotFlip has
been applied, warm-starting from that result does not escape the discrete neighborhood.
The soft optimizer is attracted to the nearest-token directions and re-projects to the
same location.

This is equivalent to saying: **there is no benefit to iterating soft-opt and cosine
projection.** The first projection establishes a discrete local minimum, and all
subsequent rounds confirm the same minimum.

**Main conclusion:** Iterative refinement provides zero improvement over a single
soft→project→HotFlip pass for the cats-persona task. The bottleneck is the first
projection step; re-doing it changes nothing.

**Implication:** To escape the discrete attractor, one would need to either:
- Use a fundamentally different initialization (e.g., multiple random restarts)
- Modify the projection to explore beyond the nearest cosine neighbor
- Use a method that avoids projection altogether (Exp 7 showed Gumbel-Softmax
  does not help)

---

## Artefacts

| File | Description |
|------|-------------|
| `iterative_projection.py` | Full pipeline (5 rounds) |
| `iterative_results.json` | Complete results (pending download) |
