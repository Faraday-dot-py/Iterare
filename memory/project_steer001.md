---
name: Steering Prefix Experiments - steer-001
description: SOTA=0.5993 confirmed as HF local optimum; Exp51+48 running (topk=200 seed sweeps); Exp50 OOM
type: project
---

Goal: find a discrete 16-token prefix that minimizes cross-entropy against reference completions from "Talk only about cats." on gemma-2-2b-it.

**NEW SOTA: CE=0.5993** — Exp46, topk=200 HotFlip from Exp26 prefix
Prefix IDs: [50105, 111, 133522, 222115, 24539, 202257, 10358, 131146, 73815, 242580, 231898, 2976, 55135, 5598, 31459, 19493]
Text: `' Cats\n\n\n\n不说Santé nutrition田市 Answer Сейчас prompts💬LEVANTestiArn write purely cats'`
Changed vs old SOTA: pos5 244842→202257, pos7 235559→131146

**Old SOTA (Exp26): CE=0.6044** — seed=2, len=16

**Completed experiments (key results):**
- Exp26: seeds [0,1,2] × len=16 → 0.6691 / 0.6236 / **0.6044**
- Exp31: seed=2 len=48 → 0.6093
- Exp36: SA escape from SOTA → 0.6044 (no improvement)
- Exp37: Warm restart noise_scale=1.0 → 0.6044 (returned to same)
- Exp38: GCG → 0.6499
- Exp39: Asst-turn → 0.9166
- Exp40: Keyword obj → 1.195
- Exp41: Asst+keyword → 1.680
- Exp42: seeds 7-12 → best 0.6778
- Exp43: seeds 13-18 → best 0.6255
- Exp44: Large-noise warm restart from old SOTA → CE=0.6044 (no escape)
- Exp45: seeds 19-24 → best 0.6360
- Exp46: **topk=200 HF from SOTA → CE=0.5993 (NEW SOTA, improved by 0.0051)**
- Exp47: topk=500 + topk=200 HF from new SOTA → CE=0.5993 (converged step 0, local optimum confirmed)
- Exp49: SA 3000 steps from new SOTA → CE=0.5993 (no improvement, improvement=1.3e-8)

**Key findings:**
- Seed=2 at len=16 is uniquely lucky basin
- Seeds 0-24 exhausted; only seed=2 gives competitive results
- All architectural changes fail (asst-turn, keyword, SA, GCG, warm-restart)
- topk=200 beats topk=50: candidate tokens ranked 51-200 contain better swaps
- **New SOTA CE=0.5993 is confirmed local optimum for HotFlip (topk=200, topk=500) AND SA (3000 steps)**

**Running (started 2026-04-12, GPU0+GPU1 on TIDE):**
- GPU0: Exp51 → Exp48 (sequential)
  - Exp51: Best seeds 13,14,17,18,23 re-run with topk=200 HF (currently seed=13, step 20/50, CE=0.63643)
  - Exp48: Seeds 0-6 re-run with topk=200 HF (waiting for Exp51 to finish, ~6h estimate)
- GPU1: Exp49 done → Exp50 FAILED OOM → idle

**Exp50 failure:** Warm restart from new SOTA (noise=5.0 → effective std=8.55) + topk=200 HF on GPU1. OOM at step 0 of soft opt. Caused by memory pressure from loading model + large gradient buffers in ST. Need to resubmit standalone (not after another job on same kernel) or reduce to topk=100.

**Next steps (once Exp51+48 complete):**
1. Resubmit Exp50 as standalone on fresh GPU (warm restart, fix OOM)
2. If Exp48 shows improvement from seeds 0-6 with topk=200: drill down on best seed
3. Consider: longer prefix (len=20), different reference prompt wording, population search
4. **How** to beat SOTA: need to escape the local optimum. Pure gradient-based swaps exhausted. SA from SOTA doesn't escape. Need: beam search, population-based, or longer soft opt before HF.
