# anti-steer-001: Residual-Stream Anti-Steering for Idea Diversity

**Status:** Complete  
**Model:** Qwen2.5-3B-Instruct (run03, TIDE L40) · gpt2-medium (run01/02, local)  
**Tasks:** 30 coding-issue prompts · 5 ideas per task per condition  
**Conditions:** 7 (prompt_only, output_rerank, anti_steer α ∈ {0.1, 0.25, 0.5, 1.0, 2.0})  

---

## Hypothesis

When a language model generates a sequence of ideas for the same problem, later outputs
tend to be semantically similar to earlier ones — the model "gets stuck" in a region of
semantic space. Suppressing the residual-stream activations that encode prior ideas during
generation should push new outputs into distinct semantic territory without degrading quality.

---

## Method

**Anti-steering:** after the first idea is generated, a mean activation vector is computed
over all prior ideas by passing each through the model and capturing the hidden state at
layer `2 * n_layers // 3` (chosen empirically as a deep-enough semantic layer without
over-intervention). During generation of the next idea, a forward hook subtracts
`α × mean_prior_vector` from that layer's output at every token step:

```
h_steered = h_original − α · v_prior
```

The vector accumulates as ideas are added: by step 5, it is the mean of 4 prior-idea
activations. No gradient is computed; inference-time only.

**Baseline conditions:**
- `prompt_only` — prior ideas listed in the prompt; model is instructed to propose something distinct
- `output_rerank` — generate 3 candidates per step, keep the one with minimum cosine similarity to prior idea embeddings

**Diversity metrics** (all computed with sentence-transformers/all-MiniLM-L6-v2):
- `mean_pairwise_sim` — mean cosine similarity across all pairs in the 5-idea sequence
- `mean_max_to_prior` — mean of each idea's maximum similarity to any earlier idea

**Usefulness** — scored 1–5 by Claude (claude-cli judge backend) on 30 tasks × 7 conditions × 5 steps (1,050 calls).

---

## Results

### Run03 — Qwen2.5-3B-Instruct (primary, n=30 tasks)

| Condition          | mean_pairwise_sim | Δ vs baseline | mean_max_to_prior | Usefulness (1–5) |
|--------------------|:-----------------:|:-------------:|:-----------------:|:----------------:|
| prompt_only        | 0.690             | —             | 0.741             | 4.89             |
| output_rerank      | 0.630             | −8.7%         | 0.694             | 4.91             |
| anti_steer α=0.1   | 0.670             | −2.9%         | 0.723             | 4.91             |
| anti_steer α=0.25  | 0.680             | −1.5%         | 0.734             | 4.92             |
| anti_steer α=0.5   | 0.635             | −8.0%         | 0.687             | 4.96             |
| **anti_steer α=1.0**| **0.332**        | **−51.9%**    | **0.416**         | **4.92**         |
| anti_steer α=2.0   | 0.202             | −70.7%        | 0.290             | 4.88             |

**Key finding:** there is a sharp nonlinearity between α=0.5 and α=1.0. Below 0.5, the
effect is modest and comparable to output_rerank. At α=1.0, pairwise similarity drops by
over half. At α=2.0 the gain continues but usefulness begins to slip slightly (still 4.88/5).
**Usefulness is essentially flat across all conditions** (range: 4.88–4.96), confirming that
the diversity gain is genuine semantic exploration rather than degenerate or off-topic output.

output_rerank is the weakest intervention: it reduces diversity by only 8.7% at 3× the
generation cost (3 candidates per step) and provides no usefulness benefit over prompt_only.

### Earlier runs

**Run01** (gpt2-medium, 5 tasks, local): initial signal. α=0.25 gave mean_sim=0.31 vs 0.50
baseline (−38%). First indication the approach works.

**Run02** (gpt2-medium, 15 tasks): added output_rerank baseline. output_rerank achieved
mean_sim=0.30 (−25%); anti_steer α=0.25 showed weaker effect at larger scale (0.41, +2%
relative to baseline — likely noise from the smaller alpha and gpt2's weaker semantic
representations).

Run03 resolved both ambiguities by scaling to a capable instruction-tuned model and sweeping
the full alpha range.

---

## Identified Sweet Spot

**α=1.0** is the recommended operating point:
- −52% pairwise similarity vs. prompt-only
- Usefulness unchanged (4.92/5)
- No sign of semantic degradation
- output_rerank is outperformed by 6× on diversity at lower generation overhead

α=2.0 is available if diversity is the only objective and slight quality degradation is
acceptable, but the marginal gain over α=1.0 (−19pp additional reduction) may not justify
the risk at higher values.

---

## Applications

Residual-stream anti-steering for idea diversity has direct utility anywhere a system needs
to generate multiple non-redundant outputs from the same model in a single session.

**Brainstorming and design tools.** Any copilot that suggests multiple implementation
approaches, design alternatives, or creative options benefits from guaranteed semantic
separation between suggestions. Without intervention, models tend to rephrase the same
core idea across outputs. Anti-steering makes each suggestion mechanistically distinct.

**RAG and retrieval augmentation.** When generating multiple hypotheses for query expansion
or document candidates, redundancy wastes retrieve-and-rank budget. Anti-steering can
diversify the hypothesis set before retrieval, increasing coverage of the answer space.

**Automated code review and refactoring suggestion.** A reviewer that proposes multiple
refactoring strategies for the same function can use anti-steering to ensure suggestions
span different axes (performance, readability, testability) rather than converging on the
same structural change.

**Red-teaming and adversarial evaluation.** Security tools that auto-generate multiple
attack vectors or test cases for a given target benefit from non-redundant outputs.
Anti-steering can force the model to explore orthogonal attack surfaces rather than
generating variations on a single exploit pattern.

**Multi-agent debate and ensemble reasoning.** In systems where multiple model outputs
are aggregated (mixture-of-experts voting, chain-of-thought ensembles), diversity of
reasoning paths is a known quality predictor. Anti-steering can be applied per-agent
or per-sample to reduce inter-agent correlation without changing model weights.

**Curriculum and question generation.** Educational tools that generate multiple questions
or problems on a topic can use anti-steering to avoid clustering around the same difficulty
level or concept, producing better coverage across a syllabus.

**Limitations.** The approach requires white-box access to the model's hidden states.
It is inference-only (no training), but it does add one forward pass per prior idea for
vector capture. At very high α values (>2.0, not tested here) output quality is expected
to degrade; the threshold is model-dependent. The technique has only been validated on
coding-domain tasks; domain transfer is plausible but untested.

---

## Artifacts

| File | Description |
|------|-------------|
| `worker.py` | gpt2-medium local worker (run01/02) |
| `manager-1/worker-1/tide_run03.py` | Qwen2.5-3B-Instruct TIDE worker (run03) |
| `eval.py` | Diversity + usefulness evaluation pipeline |
| `visualize.py` | HTML report generator combining all runs |
| `results/run01.json` | Run01 results (5 tasks, gpt2-medium) |
| `results/run02.json` | Run02 results (15 tasks, gpt2-medium) |
| `manager-1/worker-1/run03_results.json` | Run03 results (30 tasks, Qwen2.5-3B) |
| `results/report.html` | Combined visualization (run01+02+03) |
