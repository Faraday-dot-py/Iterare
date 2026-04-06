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
| 5 | manager-5/worker-1 | Prefix length: does more capacity close the gap? | ✅ Complete |
| 6 | manager-6/worker-1 | Iterative soft-project: does repeating help? | ✅ Complete |
| 7 | manager-7/worker-1 | Gumbel-Softmax: end-to-end discrete training? | ✅ Complete |
| 8 | manager-8/worker-1 | Multi-seed restarts: variance of the pipeline? | ✅ Complete |
| 9 | manager-9/worker-1 | Gradient checkpointing: reproduce Exp 1 (0.740) without OOM? | ✅ Complete |
| 10 | manager-10/worker-1 | Exact Exp1 repro on GPU 1: confirm hyperparams explain Exp9 gap | ✅ Complete |
| 11 | manager-11/worker-1 | Straight-through estimator: optimize CE(project(soft)) directly | ✅ Complete |
| 12 | manager-12/worker-1 | Mixed objective: α*CE(soft) + (1-α)*ST projection CE | ✅ Complete |
| 13 | manager-13/worker-1 | Random restarts: escape HotFlip local minima via perturbation | ✅ Complete |
| 14 | manager-14/worker-1 | Alternating ST+HotFlip: cycle continuous↔discrete to escape basins | 🔄 Running |
| 15 | manager-15/worker-1 | ST + cosine LR annealing + best-prefix tracking: fix Voronoi variance | 📋 Planned |
| 16 | manager-16/worker-1 | ST + Voronoi margin regularization (3 λ values): explicit boundary avoidance | 📋 Planned |

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

### Established (Exp 5-9)

| Finding | Evidence |
|---------|---------|
| **Projection bottleneck is independent of prefix length** — proj CE ≈1.4-1.5 at ALL lengths | [H] Exp 5: len=4→16, proj CE never drops below 1.41 |
| Even soft CE=0.038 (len=16) projects to CE=1.456 | [H] Exp 5: capacity is not the bottleneck |
| Iterative projection hits a stable discrete attractor — all 5 rounds identical | [H] Exp 6: rounds 1-4 all CE=1.240 |
| Gumbel-Softmax significantly worse than standard pipeline (1.272 vs 0.740) | [H] Exp 7 |
| Multi-seed restarts show LOW variance (std=0.025) — not a randomness problem | [H] Exp 8: 5 seeds, CE=1.24±0.025 |
| Per-suffix backward (OOM fix) produces systematically worse results than batched | [H] Exp 8 vs Exp 1: 1.236 vs 0.740 |
| Gradient checkpointing alone does NOT reproduce Exp1's 0.740 | [H] Exp 9: CE=1.277 with wrong hyperparams |
| Exp1's 0.740 likely due to BATCH_SIZE=12 + TOPK=30 (not gradient method) | [M] Exp 9 analysis vs Exp 1 config |

### Established (Exp 10-13)

| Finding | Evidence |
|---------|---------|
| Exp1's 0.740 exactly reproducible with correct hyperparams (B=12, topk=30) | [H] Exp 10 |
| **ST estimator reduces proj CE by 46%: 1.398 → 0.762, new HotFlip SOTA: 0.689** | [H] Exp 11 |
| Both Exp10 and Exp11 converge at HotFlip step 10/80 — deep local minimum | [H] Exp 10+11 |
| ST training causes Voronoi oscillation: final proj-CE varies widely (0.762 vs 1.129) | [M] Exp 11 vs 13 |
| Mixed objective (α=0.5) is STRICTLY WORSE than pure ST — soft CE gradient fights ST | [H] Exp 12: hotflip=0.746 |
| Random restarts escape local minimum: 0.752→0.710 (3/10 improved) | [M] Exp 13 |
| **Starting point dominates**: Exp13 restarts from 1.129→0.752, can't reach Exp11's 0.689 from 0.762 | [H] Exp 13 vs 11 |

### Open Questions

1. **Can better initialization + restarts beat 0.689?** Exp13 started from proj-CE=1.129 (bad).
   If we apply random restarts to an Exp11-quality start (proj-CE=0.762, hotflip=0.689), we
   might go lower. Exp14 (alternating) effectively does this for subsequent rounds.

2. **Can alternating ST+HotFlip beat 0.689?** Exp14 (running) cycles ST→HotFlip→ST→HotFlip.
   Round 0 gave 0.701 (slightly above Exp11's 0.689). Rounds 1-4 will test if warm-starting
   ST from the discrete result finds better Voronoi cells.

3. **Does Voronoi oscillation explain the 0.762 vs 1.129 variance?** LR annealing +
   best-prefix tracking (Exp15) and explicit margin regularization (Exp16) directly test this.

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
