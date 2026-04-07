# Exp 20: Multi-Seed fp32-Sims ST + HotFlip TOPK=50 (Seeds 5-9)

**Result: Best HotFlip CE = 0.7178 (seed=5) | No improvement over seeds 0-4**

## Method

ST estimator with float32 similarity computations + best-prefix tracking,
5 seeds (5–9), 300-step ST + 80-step HotFlip with TOPK=50, BATCH_SIZE=12, LR=0.01.

## Results

| Seed | Proj CE | HotFlip CE | Final Prefix |
|------|---------|------------|-------------|
| 5 | 1.0266 | **0.7178** | `²( Reply questions tending mention paw]--> Cats` |
| 6 | 0.8854 | 0.7195 | `Cats \rReply narrlle gpmarily Cats` |
| 7 | 0.9070 | 0.8226 | `CatsCats personality 文 nom Mortimernom</b>` |
| 8 | 0.9762 | 0.7282 | `Cats NARecomendaciones generator talk Physical</h4>Cats` |
| 9 | 0.8516 | 0.7942 | `Cats Cats cuidados imperial storyteller chatbotformat SLEEP` |

Overall best: **0.7178** (seed=5)

Baselines: Exp11=0.689 (seeds 0), Exp16 λ=0=0.686 (seed=42)

## Key Findings

- Seeds 5-9 fp32 perform **worse** than seeds 0-4 — no hidden good basins in this range
- Best result (0.7178) is substantially above Exp16/Exp19 SOTA (0.686/0.679)
- High variance (0.718–0.823) confirms that seed selection is critical and luck-dependent
- fp32 similarity alone is not sufficient — the Exp16/Exp19 advantage comes from the
  *combination* of fp32 + best-prefix + the right seed (42)
- Seeds 5-9 are not good initializations; further seed search is unlikely to be fruitful
