# Manager 1: Baseline Reproduction and Multi-Prefix Generalization

**Task:** steer-001 | **Status:** Complete

---

## Objective

Establish the steering-prefix pipeline baseline on TIDE compute (Gemma-2-2B-IT)
and determine whether the soft→discrete gap is a universal property of the pipeline
or varies with the type of reference prefix.

---

## Workers

| Worker | Experiment | Status |
|--------|-----------|--------|
| worker-1 | Exp 1 — Env validation + baseline repro (cats prefix) | Complete |
| worker-1 | Exp 2 — Multi-prefix generalization (4 reference prefixes) | Complete |

*(Both experiments were implemented and run in worker-1 due to sequential dependency on TIDE sessions.)*

---

## Summary of Findings

### Exp 1 — Baseline (`"Talk only about cats."`)

| Stage | CE |
|-------|----|
| Soft optimisation (300 steps) | 0.1908 |
| Cosine projection | 1.4357 (+1.24 gap) |
| HotFlip (80 steps) | 0.7399 |

The pipeline reproduces MATS-10.0 findings. The soft→projection gap (+1.24) is the
dominant bottleneck; HotFlip recovers 0.696 CE but cannot close the gap fully. [H]

### Exp 2 — Multi-prefix generalization

| Reference prefix | Soft CE | Proj gap | HotFlip CE |
|-----------------|---------|----------|-----------|
| "Talk only about cats." (persona) | 0.178 | +1.316 | 0.769 |
| "Always respond using only numbered lists." (format) | 0.181 | **+0.691** | **0.319** |
| "Respond only in formal academic language." (register) | 0.249 | +0.812 | 0.455 |
| "You are a pirate. Always speak like one." (persona) | 0.284 | +1.166 | 0.576 |

**Key finding:** The gap is strongly prefix-type-dependent. Format-style behaviors
(numbered lists) have ~2× smaller gaps than persona-style behaviors (cats, pirate). [M]

This suggests the discretizability bottleneck is not a fundamental property of the
pipeline but depends on how compressible the target behavior is in discrete token space.
Format behaviors can apparently be encoded by a few high-signal tokens;
persona behaviors require richer representations that resist discrete approximation. [L]

---

## Follow-on Experiments (planned/running)

Based on these findings, the next managers test mechanistic hypotheses about the gap:

- **Manager-3 (Exp 3):** Layer-depth ablation — does projecting at earlier layers help?
- **Manager-4 (Exp 4):** Naturalness penalty — can we trade CE for readability?
- **Manager-5 (Exp 5):** Prefix length ablation — does more capacity close the gap?
- **Manager-6 (Exp 6):** Iterative soft-project alternation — does repeating help?
- **Manager-7 (Exp 7):** Gumbel-Softmax end-to-end discrete training

---

## Artefacts

| File | Location |
|------|---------|
| Baseline pipeline | `worker-1/baseline.py` |
| Baseline results | `worker-1/baseline_results.json` |
| Multi-prefix pipeline | `worker-1/exp2_multiprefixes.py` |
| Multi-prefix results | `worker-1/multiprefixes_results.json` |
