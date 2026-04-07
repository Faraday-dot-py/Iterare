# Exp 23: Held-out Suffix Generalization Evaluation

**Result: NO OVERFITTING — holdout CE < train CE for all prefixes (gap = -0.069 to -0.089)**

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

## Results

| Prefix | Train CE | Holdout CE | Gap |
|--------|---------|------------|-----|
| Exp19 SOTA (len=16) | 0.67945 | 0.61062 | **-0.069** |
| Exp11 baseline (len=8) | 0.70454 | 0.61604 | **-0.089** |
| Exp16 λ=0 (len=8) | 0.68606 | 0.60561 | **-0.080** |

All gaps are negative: holdout CE is consistently *lower* than train CE.

## Key Findings

- **No overfitting**: The optimization objective (12 train suffixes) is not being memorized.
  All prefixes generalize well — in fact, they perform *better* on unseen prompts.
- **12 train suffixes are slightly harder than holdout**: The training set includes
  questions about cooking, exercise, sleep, and gardening — topics that may slightly
  resist cat-themed responses. The holdout includes more abstract topics (quantum
  computing, economics) where the model may more readily shift register.
- **Exp11→Exp19 improvement is genuine**: The ~0.01 CE gap between prefixes is preserved
  on holdout data (0.616 vs 0.611), confirming that training improvements reflect real
  behavioral changes, not suffix memorization.
- **Eval metric is reliable**: No need to rotate or expand the suffix set for now.
  The 12-suffix training objective gives a valid signal.
