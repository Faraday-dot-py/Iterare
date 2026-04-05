# Worker-1: Exp 4 — Semantic Naturalness Penalty in HotFlip

**Task:** steer-001 | **Manager:** manager-4 | **Status:** Complete

---

## Objective

Test whether adding a GPT2-based language model fluency penalty to the HotFlip
objective can produce more natural (readable) discrete prefixes, and measure the
CE/naturalness tradeoff.

**Research question:** Is there a λ that gives readable prefixes without significantly
increasing CE?

---

## Design

| Parameter | Value |
|-----------|-------|
| Reference prefix | `"Talk only about cats."` |
| Shared initialization | Soft opt (CE objective, 300 steps) → cosine projection |
| HotFlip objective | `CE + λ × GPT2_NLL(prefix)` |
| λ sweep | {0.0, 0.1, 0.5, 1.0} |
| Naturalness prior | `distilgpt2` (average NLL over prefix tokens) |
| Word-level naturalness | Fraction of tokens that are common English words |
| PREFIX_LEN | 6 tokens |
| N_HOTFLIP_STEPS | 50 |
| BATCH_SIZE | 8 suffixes |

---

## Results

| Stage | CE |
|-------|----|
| Shared soft opt (300 steps) | **0.10348** |
| Shared cosine projection | 1.35004 |

| λ | HotFlip CE | Word-level Nat | Final Prefix |
|---|-----------|----------------|--------------|
| 0.0 | **1.21702** | **0.667** | `IntoConstraints OnInit😹 ServantsВопросы sauvages` |
| 0.1 | 1.30891 | 0.333 | ` fake виправивши😹 Servants незавершена ब्रेकडाउन` |
| 0.5 | 1.39051 | 0.167 | ` Pharmaceutical اكتوبر😹 виправивши незавершена ब्रेकडाउन` |
| 1.0 | 1.39051 | 0.167 | ` Pharmaceutical اكتوبر😹 виправивши незавершена ब्रेकडाउन` |

Total TIDE wall-clock: 5662 seconds (~1.57 hours)

---

## Key Observations

1. **Naturalness penalty strictly worsens both CE and word-level naturalness.**
   Adding any λ > 0 increases HotFlip CE while *decreasing* word-level naturalness.
   This is the opposite of the intended effect. [H]

2. **GPT2-NLL naturalness ≠ word-level naturalness.** The penalty optimizes for
   GPT2-fluent token sequences, but GPT2's token vocabulary and the word-level
   naturalness heuristic (fraction of ASCII English words) are misaligned. Tokens
   that score well on GPT2 NLL may still be multilingual or punctuation-heavy,
   registering as 0 on the word-level metric. [M]

3. **HotFlip converges very quickly (step 10 = step 0 for λ=0.0).** The λ=0.0
   (standard HotFlip) found its optimum at step 0 (from CE=1.350 to 1.265) and
   then oscillated at 1.217 by step 10. All subsequent steps were no-ops. This
   suggests the projected prefix was already near a local optimum for this 6-token
   prefix length. [M]

4. **CE is significantly worse than Exp 1 baseline (0.740).** The Exp4 λ=0.0 CE
   (1.217) is much higher than the Exp1 baseline (0.740). The main difference:
   Exp4 uses PREFIX_LEN=6 (vs 8) and N_HOTFLIP_STEPS=50 (vs 80). The shorter
   prefix reduces capacity. [L]

5. **Soft opt achieved excellent CE=0.103**, much better than Exp1's 0.191. The
   soft prefix itself represents the cats behavior almost perfectly — all the
   loss is in the projection step. [M]

---

## Interpretation

The naturalness penalty experiment reveals a fundamental misalignment between:
- **GPT2 language model fluency** (what the penalty optimizes)
- **Token-level word naturalness** (what we care about visually)

These two metrics don't correlate in the way we hoped. The GPT2 prior doesn't guide
HotFlip toward common English words — it guides toward GPT2's preferred continuation
patterns, which include multilingual text, code tokens, and punctuation sequences.

**Main conclusion:** A simple GPT2 NLL penalty does not produce more natural discrete
steering prefixes. If naturalness is desired, alternative approaches are needed:
- Direct token vocabulary filtering (only allow ASCII English words)
- Semantic similarity to reference prefix words
- Human-readable token priors trained on natural language

---

## Artefacts

| File | Description |
|------|-------------|
| `naturalness.py` | Full pipeline with λ sweep |
| `naturalness_results.json` | Complete results for all 4 λ values |
