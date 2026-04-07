# Exp 17: Multi-Seed ST+Anneal+Best-Prefix + HotFlip TOPK=50

**Result: Best HotFlip CE = 0.6942 (seed=0) | 5 seeds | TOPK=50**

## Method

ST estimator + cosine LR annealing (LR_MAX=0.01→LR_MIN=0.001) + best-prefix tracking,
5 independent seeds (0–4), 300-step ST + 80-step HotFlip with TOPK=50, BATCH_SIZE=12.

## Results

| Seed | Proj CE | HotFlip CE | Final Prefix |
|------|---------|------------|-------------|
| 0 | 0.8648 | **0.6942** | `Cats</h6>RolfSpeechMALE instructions answering Cats` |
| 1 | 0.8387 | 0.7165 | `catsRenderAtEndOf языкLassasticrespond topic cats` |
| 2 | 0.8538 | 0.7717 | `Cats NON🌸Question mischievous mature felineSpeech` |
| 3 | 0.8797 | 0.7206 | `cats IAM text): respondassertj Adults cats` |
| 4 | 0.8570 | 0.7279 | `Cats pets CAT exclusively脚注の使い方Cats 文 narration` |

Overall best: **0.6942** (seed=0)

## Key Findings

- TOPK=50 with multi-seed does NOT reliably beat the Exp16 single-seed SOTA of 0.686
- Seed variance is high: 0.694–0.772 (range of 0.078 CE)
- Only 1 of 5 seeds came close to SOTA; median performance (0.728) is well above
- TOPK=50 does not improve HotFlip convergence vs TOPK=30 in this method variant
- Cosine annealing + best-prefix tracking does not systematically close the ST→discrete gap
