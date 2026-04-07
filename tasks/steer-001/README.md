# steer-001: Steering Prefix Research (MATS-10.0)

**Status:** In progress (Exp16-20 complete) | **SOTA:** 0.6794 (Exp19, PREFIX_LEN=16) | **Compute:** TIDE (2× NVIDIA A100 80GB)

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
| 14 | manager-14/worker-1 | Alternating ST+HotFlip: cycle continuous↔discrete to escape basins | ✅ Complete |
| 15 | manager-15/worker-1 | ST + cosine LR annealing + best-prefix tracking: fix Voronoi variance | ✅ Complete |
| 16 | manager-16/worker-1 | ST + Voronoi margin regularization (3 λ values): explicit boundary avoidance | ⚠️ Partial (λ=0.0 only; crash) |
| 17 | manager-17/worker-1 | Multi-seed (5×) ST+anneal+best-prefix + HotFlip TOPK=50: characterize reliability | ✅ Complete |
| 18 | manager-18/worker-1 | TOPK escalation (50→100→200) from Exp11 best: is 0.689 escapable? | ✅ Complete |
| 19 | manager-19/worker-1 | ST + cosine annealing + best-prefix, PREFIX_LEN=16: more discrete capacity | ✅ Complete |
| 20 | manager-20/worker-1 | Multi-seed (seeds 5-9) fp32-sims ST + HotFlip TOPK=50: exploit better basin config | ✅ Complete |

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
| Alternating ST+HotFlip fails: ST warm-started from discrete result has zero gradient (fixed point). Best CE=0.701 | [H] Exp 14 |
| **Cosine LR annealing hurts**: seed=42 with cosine schedule gets proj=0.868, hotflip=0.738 — worse than constant LR | [H] Exp 15 vs 11 |
| **New SOTA 0.686**: Exp16 λ=0.0 (float32 sims, best-prefix, constant LR) beat Exp11 0.689 from *worse* proj (0.877) | [H] Exp 16 |
| **Basin quality > projection quality**: low proj-CE does not imply low final HotFlip CE. Exp16 improved 0.877→0.686 (Δ=0.191) vs Exp11's 0.762→0.689 (Δ=0.073) | [H] Exp 11 vs Exp16 |
| Float32 vs bfloat16 in st_project changes optimization trajectory significantly (different Voronoi cells) even at same seed | [M] Exp11 vs Exp16 λ=0 |

### Established (Exp 17-20)

| Finding | Evidence |
|---------|---------|
| TOPK=50 multi-seed (seeds 0-4) does NOT reliably beat Exp16 SOTA; best=0.694, high variance (range 0.078) | [H] Exp 17 |
| TOPK escalation from Exp11 starting point gives marginal improvement: 0.705→0.688 (TOPK=100 helped; TOPK=50/200 did not) | [H] Exp 18 |
| **PREFIX_LEN=16 new SOTA 0.6794**: more discrete tokens give HotFlip more degrees of freedom | [H] Exp 19 |
| Seeds 5-9 fp32-sims perform worse than seeds 0-4 (best 0.718); seed=42 and seeds 0-4 are unusually good | [H] Exp 20 |
| ST→HotFlip recovery scales with prefix length: Δ=0.073 (len=8, Exp11), Δ=0.191 (len=8 fp32, Exp16), Δ=0.226 (len=16, Exp19) | [H] Exp 11/16/19 |

### Open Questions

1. **Can PREFIX_LEN=32 or 24 push below 0.670?** Exp19 shows each doubling of prefix capacity
   gives ~0.007 CE improvement. Extrapolating, len=32 might reach ~0.672.

2. **Why is seed=42 special?** Seeds 0-4 and 5-9 are clearly worse than seed=42 for both
   Exp16 (fp32-sims, constant LR) and related configs. The seed affects soft initialization
   which may land in structurally better Voronoi basins.

3. **Voronoi margin regularization (Exp16 λ=0.5, λ=2.0) still untested** due to crash.
   Could revisit if further progress stalls.

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

- TIDE CSU JupyterHub: 2× NVIDIA A100 80GB (upgraded from L40 44GB on 2026-04-06)
- Wallclock cap: 4h per submission (14400s timeout)
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
