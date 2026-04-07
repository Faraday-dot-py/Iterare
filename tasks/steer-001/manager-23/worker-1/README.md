# Exp 23: Held-out Suffix Generalization Evaluation

**Status:** Queued | **Results:** Pending

## Method

Evaluates three discrete prefixes on two suffix sets:
- **Train set** (12 suffixes): the suffixes used in all optimization experiments
- **Holdout set** (20 suffixes): never seen in optimization; entirely new topics

Prefixes evaluated:
1. Exp19 SOTA (len=16, CE=0.679): `Cats).[animals instructions... General`
2. Exp16 λ=0 (len=8, CE=0.686): `Keeping Cats'];?>Only responding Cat Ley trivia`
3. Exp11 baseline (len=8, CE=0.689): ` Cat wellnessceptre人工语言 reply with cats`

For each prefix, compute CE using the standard teacher-forcing metric
(early-token weighted, greedy reference completions from REF_PREFIX).

## Research Question

Is the CE improvement from Exp11→Exp19 (0.689→0.679) genuine generalization,
or does it partially reflect memorization of the 12 training suffixes?

**Expected outcomes:**
- If generalization gap (holdout CE - train CE) is small (<0.05) for all prefixes:
  the metric is reliable and improvements are real
- If gap is large (>0.1) or increases with newer prefixes:
  optimization is overfitting to 12 suffixes; future experiments need a larger/rotated suffix set

## Holdout Suffixes (20 topics)

Economics, vaccines, chess, personal finance, workplace dynamics, climate,
movies, home repair, empathy, creative writing, job interviews, earthquakes,
quantum computing, motivation, sourdough, meditation, stock market, exercise,
political systems, stress management.

## Timing

No optimization required — only forward passes. Expected ~20min on A100.
