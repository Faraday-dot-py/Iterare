# Manager 4: Exp 4 — Naturalness Penalty in HotFlip

**Task:** steer-001 | **Status:** Running

---

## Objective

Test whether adding a language model fluency penalty to the HotFlip objective can
push discrete prefixes toward more natural (human-readable) text, and measure the
CE/naturalness tradeoff.

**Research question:** Is there a λ that gives readable prefixes without wrecking CE?

---

## Design

- Reference prefix: `"Talk only about cats."`
- Shared starting point: standard soft opt (CE objective, 300 steps) → cosine projection
- HotFlip with combined objective: `CE + λ × naturalness_penalty`
- Naturalness penalty: average NLL under `distilgpt2` (lower = more GPT2-fluent)
- λ sweep: {0.0, 0.1, 0.5, 1.0}
- PREFIX_LEN = 6, N_HOTFLIP_STEPS = 50
- Naturalness score: fraction of tokens that are common English words

**Motivation:** The Exp 1 and 2 results consistently produced multilingual garbage tokens.
If a small naturalness penalty (λ=0.1) can push tokens toward English without
significantly increasing CE, it would make the steering prefixes more interpretable and
potentially more transferable.

---

## Results

*(Pending — experiment running on TIDE GPU 1)*

---

## Worker

| Worker | Script | Output |
|--------|--------|--------|
| worker-1 | `naturalness.py` | `naturalness_results.json` (pending) |
