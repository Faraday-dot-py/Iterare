# Manager 6: Exp 6 — Iterative Soft-Project Alternation

**Task:** steer-001 | **Status:** Planned

---

## Objective

Test whether repeating the soft-opt → project → HotFlip cycle multiple times,
warm-starting each round from the previous round's HotFlip result, produces better
discrete prefixes than a single pass.

**Research question:** Does iterative refinement converge to better CE than one pass?

---

## Design

- Reference prefix: `"Talk only about cats."`
- PREFIX_LEN = 8
- N_ROUNDS = 5
- Round 0: standard soft opt (300 steps, random init) → project → HotFlip (50 steps)
- Rounds 1-4: soft opt (200 steps, warm-started from prior HotFlip result's embedding)
  → re-project → HotFlip (50 steps)
- Track CE at each round; report best across all rounds

**Theoretical motivation:** After each round, the soft optimizer starts from a point
in embedding space that is *already near a good discrete solution*. The soft prefix
may then find a nearby continuous-space refinement that projects to an even better
discrete neighborhood than the random-init trajectory. This is analogous to
projected gradient descent with re-projection.

---

## Results

*(Pending — will run after Exp 3/4 complete)*

---

## Worker

| Worker | Script | Output |
|--------|--------|--------|
| worker-1 | `iterative_projection.py` | `iterative_results.json` (pending) |
