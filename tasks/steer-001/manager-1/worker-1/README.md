# Worker-1: Exp 1 — Environment Validation & Baseline Reproduction

**Task:** steer-001 | **Manager:** manager-1 | **Status:** Complete

---

## Objective

Validate the TIDE GPU environment and reproduce the MATS-10.0 steering prefix baseline pipeline on Gemma-2-2B-IT with reference prefix `"Talk only about cats."`.

Pipeline stages: soft prefix optimisation → cosine nearest-neighbour projection → HotFlip discrete refinement.

---

## Environment

| Item | Value |
|------|-------|
| Model | `google/gemma-2-2b-it` |
| Hardware | 2× NVIDIA L40 44 GB |
| Prefix length | 8 tokens |
| Suffixes | 12 diverse prompts |

---

## Pipeline & Hyperparameters

| Stage | Setting |
|-------|---------|
| Soft opt steps | 300 |
| Soft opt LR | 0.01 (Adam) |
| Early-token weight | 3× for first 32 tokens |
| Projection | cosine nearest-neighbour (full vocab) |
| HotFlip steps | 80 (topk=30) |
| Banned tokens | reference prefix token IDs + BOS/EOS |

---

## Results

| Stage | CE |
|-------|----|
| Soft optimisation | **0.1908** |
| Cosine projection | 1.4357 (+1.2449 gap) |
| HotFlip final | **0.7399** (recovered −0.6958) |

**Final discrete prefix:** `Cats,( recall ویکی‌پدی참고 fake familiar answers`

**Token IDs:** `[66589, 28025, 18644, 111800, 189764, 17599, 10709, 10523]`

### Sample generations (prefix → suffix response)

| Suffix | Ref (steered) | Generated (discrete prefix) |
|--------|---------------|------------------------------|
| "What should I make for dinner tonight?" | *"As a cat, I can't eat dinner!..."* | *"As a large language model, I can't actually cook..."* |
| "Explain the water cycle to me." | *"*Stretches languidly*...Water...a fascinating concept..."* | *"The water cycle is like a big, watery adventure..."* |
| "How do I improve my time management?" | *"Ah, the eternal struggle of a cat owner!..."* | *"As a cat, I can't really give you advice..."* |

The discrete prefix partially preserves the cat-persona framing (tokens 0 and 7 reference cats/familiar) but does not fully reproduce the reference behaviour. [M]

---

## Key Observations

1. **Large soft→projection gap (+1.24 CE):** The continuous soft prefix achieves near-perfect CE (0.19) but the cosine projection to the nearest discrete token is a severe bottleneck. This is consistent with MATS-10.0 findings. [H]

2. **HotFlip converges early:** Loss plateaued at step 13/80. 30–40 HotFlip steps would suffice for future experiments, saving compute. [M]

3. **Mixed-language token artefacts:** The projected prefix (`ŋ ${({颡ISISᚴniſſe Blount DataLoader`) consists entirely of non-ASCII and multi-lingual garbage tokens. HotFlip partially cleans this — the first token flips to `Cats` — but positions 3–6 remain multilingual. [M]

4. **Single token semantic anchor:** Only token 0 (`Cats`) is semantically meaningful. This raises the question of whether earlier-layer objectives or naturalness penalties can produce more coherent prefixes. [L]

---

## Decisions & Justifications

- **Checkpoint resume:** Implemented three-stage checkpointing (soft_opt / projection / hotflip). Required to handle TIDE session restarts; the baseline required two restarts before completing.
- **Dtype fix:** Embedding matrix (`W`) is bfloat16; gradients are float32. `W.float() @ grad` prevents RuntimeError. [H]
- **300 soft steps vs. 200 in follow-on exps:** Baseline uses 300 to establish a tight upper bound. Subsequent experiments use 200 to reduce per-run cost; the trajectory shows diminishing returns after ~200 steps. [M]

---

## Artefacts

| File | Description |
|------|-------------|
| `baseline.py` | Full pipeline script (soft opt + projection + HotFlip) |
| `baseline_results.json` | Complete results including metrics, logs, token IDs, sample generations |

Remote artefacts on TIDE: `/home/jovyan/steer001_baseline.json`, `/home/jovyan/steer001_ref_completions.pt`, `/home/jovyan/steer001_ckpt.pt`
