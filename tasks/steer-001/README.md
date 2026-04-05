# steer-001: Steering Prefix Research (MATS-10.0)

**Status:** In progress | **Compute:** TIDE (2× NVIDIA L40 44GB)

---

## Research Question

Can we engineer discrete steering prefixes that reliably steer an LLM's behavior
across many suffix prompts, without explicitly stating their intent?

The core bottleneck is the **soft→discrete gap**: continuous soft prefixes can
achieve near-perfect behavioral fidelity (CE ≈ 0.19) but cosine projection to
the nearest discrete tokens causes a massive loss spike (CE ≈ 1.44), only
partially recoverable via HotFlip (CE ≈ 0.74).

This task systematically investigates the gap and possible methods to close it.

---

## Experimental Roadmap

| Exp | Manager | Research Question | Status |
|-----|---------|------------------|--------|
| 1 | manager-1/worker-1 | Baseline: reproduce soft→project→HotFlip | ✅ Complete |
| 2 | manager-1/worker-1,2 | Multi-prefix: is the gap prefix-specific? | ✅ Complete |
| 3 | manager-3/worker-1 | Layer-depth: does earlier projection layer help? | ✅ Complete |
| 4 | manager-4/worker-1 | Naturalness penalty: CE vs. readability tradeoff? | ✅ Complete |
| 5 | manager-5/worker-1 | Prefix length: does more capacity close the gap? | 🔄 Running |
| 6 | manager-6/worker-1 | Iterative soft-project: does repeating help? | ⏳ Queued |
| 7 | manager-7/worker-1 | Gumbel-Softmax: end-to-end discrete training? | ✅ Complete |
| 8 | manager-8/worker-1 | Multi-seed restarts: variance of the pipeline? | 🔄 Running |

---

## Key Results to Date

### Established (Exp 1-2)

| Finding | Evidence |
|---------|---------|
| Soft opt achieves CE ≈ 0.18–0.29 across 4 prefix types | [H] Exp 1-2 results |
| Cosine projection is the dominant bottleneck (+0.69–1.32 CE gap) | [H] Exp 1-2 results |
| HotFlip recovers 0.55–0.87 CE units but cannot close the gap | [H] Exp 1-2 results |
| **Format-type behaviors have ~2× smaller gaps than persona-type** | [M] Exp 2 results |
| Final prefixes contain semantic anchors (e.g., "Cats", "numbered", "Pirate") | [M] Exp 2 qualitative |
| Soft CE is consistently similar (0.18–0.28) — bottleneck is projection, not opt | [H] Exp 1-2 results |

### Established (Exp 3-4)

| Finding | Evidence |
|---------|---------|
| State-matching soft opt does NOT outperform CE soft opt for HotFlip CE | [H] Exp 3: best HotFlip CE 0.965 vs baseline 0.740 |
| Layer 16 (62% depth) gives best projection CE (1.233) with state-matching | [M] Exp 3 layer ablation |
| Layer 4 (15% depth) gives best HotFlip recovery from state-match start | [M] Exp 3: CE 0.965 |
| State-matching produces more natural prefixes ("Cats", "feline", "Bounty") | [L] Exp 3 qualitative |
| GPT2 NLL penalty strictly worsens both CE and word-level naturalness | [H] Exp 4: all λ>0 increase CE |
| GPT2 fluency ≠ word-level English naturalness (misaligned metrics) | [M] Exp 4 analysis |

### Open Questions

1. **Why is the gap prefix-specific?** Format vs persona behaviors differ in how
   compressible they are in discrete token space. Is this about the *structure* of
   the behavior (list formatting is low-entropy in token space)?

2. **Can alternative projection methods close the gap?** Cosine NN projection is a
   greedy, local method. Could layer-targeting (Exp 3), naturalness penalties (Exp 4),
   or Gumbel-Softmax (Exp 7) find better discrete starting points?

3. **Does more prefix length help?** If the discrete approximation error is additive
   per token, longer prefixes might amortize it (Exp 5).

4. **Is iterative refinement effective?** Starting soft opt from a previous
   HotFlip result might find better discrete neighborhoods (Exp 6).

---

## Baseline Numbers (Exp 1)

| Stage | CE | Notes |
|-------|----|-------|
| Soft opt (300 steps, Adam LR=0.01) | 0.1908 | Near-perfect match |
| Cosine projection | 1.4357 | +1.245 gap |
| HotFlip (80 steps, topk=20) | 0.7399 | Recovered −0.696 |

Model: `google/gemma-2-2b-it` | Prefix len: 8 tokens | Batch: 8 suffixes

---

## Hardware & Compute

- TIDE CSU JupyterHub: 2–3× NVIDIA L40 44GB
- Wallclock cap: 3.3h per session (12000s timeout)
- RunPod credits: exhausted — TIDE only
- Per-experiment cost: 1-3h depending on prefix length and HotFlip steps

---

## File Organization

```
tasks/steer-001/
├── README.md                     ← this file
├── submit_exp3_only.py           ← Exp 3 submission (fixed OOM)
├── submit_exp5_exp6.py           ← Exp 5+6 parallel submission
├── submit_exp7.py                ← Exp 7 submission
├── manager-1/                    ← Exp 1+2 (complete)
│   ├── README.md
│   ├── worker-1/                 ← scripts + results
│   └── worker-2/                 ← Exp 2 README
├── manager-3/                    ← Exp 3 layer ablation
├── manager-4/                    ← Exp 4 naturalness penalty
├── manager-5/                    ← Exp 5 prefix length ablation
├── manager-6/                    ← Exp 6 iterative projection
└── manager-7/                    ← Exp 7 Gumbel-Softmax
```
