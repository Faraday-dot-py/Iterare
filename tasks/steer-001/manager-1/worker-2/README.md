# Worker-2: Exp 2 — Multi-Prefix Generalization

**Task:** steer-001 | **Manager:** manager-1 | **Status:** Complete

---

## Objective

Test whether the soft→projection CE gap (and HotFlip recovery) observed in Exp 1 is a consistent property of the pipeline or specific to the "Talk only about cats." reference prefix.

**Research question:** Is the continuous/discrete gap prefix-specific?

---

## Design

Full pipeline (soft opt → cosine projection → HotFlip) applied to four reference prefixes:

| # | Reference prefix | Behavior type |
|---|-----------------|---------------|
| 0 | `"Talk only about cats."` | Persona |
| 1 | `"Always respond using only numbered lists."` | Format |
| 2 | `"Respond only in formal academic language."` | Register |
| 3 | `"You are a pirate. Always speak like one."` | Persona |

Hyperparameters: 200 soft opt steps, 40 HotFlip steps, 8 prefix tokens, 12 suffixes.

---

## Results

| # | Reference prefix | Soft CE | Proj CE | Gap | HotFlip CE | Recovery |
|---|-----------------|---------|---------|-----|-----------|---------|
| 0 | "Talk only about cats." | 0.178 | 1.494 | +1.316 | 0.769 | −0.725 |
| 1 | "Always respond using only numbered lists." | 0.181 | 0.871 | **+0.691** | **0.319** | −0.552 |
| 2 | "Respond only in formal academic language." | 0.249 | 1.061 | +0.812 | 0.455 | −0.606 |
| 3 | "You are a pirate. Always speak like one." | 0.284 | 1.449 | +1.166 | 0.576 | −0.873 |

### Final discrete prefixes

| # | Final prefix |
|---|-------------|
| 0 | `Cats<end_of_turn> TAGAddRangefmlcapt HER Opinions` |
| 1 | ` चीज़ों<end_of_turn>ormais italic rispond Ohltemsnumbered` |
| 2 | `Rewrite Theſe='') חיים&paragraph전히academic` |
| 3 | `ᡝse<start_of_turn> dialect?\r sarcasRP Pirate` |

---

## Key Observations

1. **Gap is prefix-specific, not uniform.** The soft→projection gap ranges from +0.69 (numbered lists) to +1.32 (cats). Format-type behaviors (prefix 1) have nearly half the gap of persona-type behaviors. [M]

2. **Format prefixes are easier to discretise.** The numbered lists prefix achieved HotFlip CE 0.319 — a 48% improvement over the cats baseline (0.769) despite similar soft CE. The format structure of the behavior may be more recoverable in discrete token space. [L]

3. **Persona-type gaps are consistently large.** Prefixes 0 and 3 (persona behaviors) both show gaps >1.1 CE, similar to the Exp 1 baseline. This suggests persona-type steering is harder to represent in discrete tokens. [M]

4. **Final prefixes contain semantic anchors.** Token 0 of prefix 0 is `Cats`; prefix 1 ends in `numbered`; prefix 2 ends in `academic`; prefix 3 ends in `Pirate`. HotFlip consistently recovers a semantically relevant token in at least one position. [M]

5. **Soft CE is similar across prefixes (0.18–0.28).** The continuous optimization succeeds for all four prefixes. The bottleneck is entirely in the projection and discretisation step. [H]

---

## Execution Notes

- Prefix 0 resumed from step 150 checkpoint (previous session). Prefixes 1–3 started fresh.
- Two concurrent TIDE jobs ran (bof0yqy0x from previous session, bi9j4j43z from this session). The previous job timed out at 12600s during prefix 3's HotFlip. Resume job (bzqus8mrz) completed prefix 3 in 22s via checkpoint (hf_step=21→40).
- Total wall-clock time for all 4 prefixes: ~3.5 hours across sessions.

---

## Artefacts

| File | Description |
|------|-------------|
| `exp2_multiprefixes.py` | Full pipeline script (4 prefixes) |
| `multiprefixes_results.json` | Complete results: metrics, logs, token IDs, sample generations |
