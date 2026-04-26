"""Anti-steering experiment: idea diversity via residual-stream suppression.

Conditions
----------
1. prompt_only       — prior ideas in context, no intervention
2. output_rerank     — generate N candidates, keep least similar to prior ideas
3. anti_steer        — subtract mean prior-idea activation vector during generation

Usage
-----
python tasks/anti-steer-001/worker.py [--model gpt2-medium] [--n-tasks 15] [--n-ideas 5]
                                       [--alpha 0.5] [--candidates 3] [--out results/run.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from hooks import ActivationCapture, SteeringIntervention, num_layers
from coding_tasks import TASKS

# ── notify (stdlib only) ─────────────────────────────────────────────────────
import os as _os, json as _json, urllib.request as _urlreq

def notify(title, body=""):
    key = _os.environ.get("PUSHBULLET_API_KEY", "")
    if not key:
        return
    try:
        data = _json.dumps({"type": "note", "title": title, "body": body}).encode()
        req = _urlreq.Request(
            "https://api.pushbullet.com/v2/pushes", data=data, method="POST",
            headers={"Access-Token": key, "Content-Type": "application/json"},
        )
        _urlreq.urlopen(req, timeout=5)
    except Exception:
        pass


# ── generation helpers ───────────────────────────────────────────────────────

IDEA_PROMPT_TEMPLATE = """\
Issue: {task}

Propose one specific implementation approach. Be concrete — name the mechanism, \
data structure, or algorithm. One short paragraph, no bullet points, no preamble.\
"""

IDEA_PROMPT_WITH_PRIOR = """\
Issue: {task}

Previous approaches already considered:
{prior_block}

Propose a different implementation approach that is mechanistically distinct from \
all of the above. Be concrete. One short paragraph, no bullet points, no preamble.\
"""

CANON_PROMPT_TEMPLATE = """\
State the core implementation mechanism of the following idea in one sentence. \
Focus on the key technical approach, ignoring wording and framing details.

Idea: {idea}

Core mechanism:"""


def build_idea_prompt(task: str, prior_ideas: list[str]) -> str:
    if not prior_ideas:
        return IDEA_PROMPT_TEMPLATE.format(task=task)
    prior_block = "\n".join(f"{i+1}. {idea[:200]}" for i, idea in enumerate(prior_ideas))
    return IDEA_PROMPT_WITH_PRIOR.format(task=task, prior_block=prior_block)


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.9,
    do_sample: bool = True,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def canonicalize(model, tokenizer, idea: str) -> str:
    prompt = CANON_PROMPT_TEMPLATE.format(idea=idea[:300])
    return generate(model, tokenizer, prompt, max_new_tokens=60, temperature=0.2)


def capture_vector(
    model,
    tokenizer,
    text: str,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    prompt = CANON_PROMPT_TEMPLATE.format(idea=text[:300])
    inputs = tokenizer(prompt, return_tensors="pt")
    with ActivationCapture(model, layers) as cap:
        with torch.no_grad():
            model(**inputs)
    return cap.get_mean_vectors()


def merge_vectors(
    vecs_list: list[dict[int, torch.Tensor]],
    layers: list[int],
) -> dict[int, torch.Tensor]:
    merged = {}
    for l in layers:
        all_v = [v[l] for v in vecs_list if l in v]
        if all_v:
            merged[l] = torch.stack(all_v).mean(dim=0)
    return merged


# ── conditions ───────────────────────────────────────────────────────────────

def run_prompt_only(model, tokenizer, task: str, n_ideas: int) -> list[str]:
    ideas = []
    for _ in range(n_ideas):
        prompt = build_idea_prompt(task, ideas)
        idea = generate(model, tokenizer, prompt)
        ideas.append(idea)
    return ideas


def run_output_rerank(
    model,
    tokenizer,
    embed_fn,
    task: str,
    n_ideas: int,
    n_candidates: int,
) -> list[str]:
    import numpy as np

    ideas = []
    idea_embs = []

    for step in range(n_ideas):
        candidates = []
        prompt = build_idea_prompt(task, ideas)
        for _ in range(n_candidates if step > 0 else 1):
            candidates.append(generate(model, tokenizer, prompt))

        if step == 0 or not idea_embs:
            chosen = candidates[0]
        else:
            cand_embs = embed_fn(candidates)
            prior_matrix = np.stack(idea_embs)  # (n_prior, dim)
            # pick candidate with minimum mean cosine similarity to prior ideas
            scores = []
            for ce in cand_embs:
                sims = np.dot(prior_matrix, ce) / (
                    np.linalg.norm(prior_matrix, axis=1) * np.linalg.norm(ce) + 1e-9
                )
                scores.append(sims.mean())
            chosen = candidates[int(np.argmin(scores))]

        ideas.append(chosen)
        idea_embs.append(embed_fn([chosen])[0])

    return ideas


def run_anti_steer(
    model,
    tokenizer,
    task: str,
    n_ideas: int,
    steer_layers: list[int],
    alpha: float,
) -> list[str]:
    ideas = []
    prior_vectors: list[dict[int, torch.Tensor]] = []

    for step in range(n_ideas):
        prompt = build_idea_prompt(task, ideas)

        if step == 0 or not prior_vectors:
            idea = generate(model, tokenizer, prompt)
        else:
            steering = merge_vectors(prior_vectors, steer_layers)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_len = inputs["input_ids"].shape[1]
            with SteeringIntervention(model, steer_layers, alpha, steering):
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            new_tokens = out[0][input_len:]
            idea = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        ideas.append(idea)
        vec = capture_vector(model, tokenizer, idea, steer_layers)
        prior_vectors.append(vec)

    return ideas


# ── diversity scoring ─────────────────────────────────────────────────────────

def make_embed_fn(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model_name)

    def embed(texts: list[str]):
        import numpy as np
        embs = st.encode(texts, normalize_embeddings=True)
        return embs  # (n, dim) numpy

    return embed


def score_diversity(ideas: list[str], embed_fn) -> dict[str, float]:
    import numpy as np
    embs = embed_fn(ideas)
    n = len(embs)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(embs[i], embs[j]))
            sims.append(sim)
    # max similarity to any prior idea (for each step 1+)
    max_to_prior = []
    for i in range(1, n):
        prior_sims = [float(np.dot(embs[i], embs[j])) for j in range(i)]
        max_to_prior.append(max(prior_sims))
    return {
        "mean_pairwise_sim": float(np.mean(sims)) if sims else 0.0,
        "mean_max_to_prior": float(np.mean(max_to_prior)) if max_to_prior else 0.0,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument("--n-tasks", type=int, default=15)
    parser.add_argument("--n-ideas", type=int, default=5)
    parser.add_argument("--alphas", default="0.1,0.25,0.5",
                        help="Comma-separated alpha values for anti_steer sweep")
    parser.add_argument("--steer-layer", type=int, default=None,
                        help="Single layer index for steering (default: 2*n//3)")
    parser.add_argument("--candidates", type=int, default=3,
                        help="Candidates per round for output_rerank condition")
    parser.add_argument("--out", default="results/run.json")
    parser.add_argument("--conditions", default="prompt_only,output_rerank,anti_steer",
                        help="Comma-separated subset of conditions to run")
    args = parser.parse_args()

    out_path = Path(__file__).parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_conditions = [c.strip() for c in args.conditions.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    tasks = TASKS[: args.n_tasks]

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    n_layers = num_layers(model)
    # single layer for steering: fewer simultaneous interventions = cleaner signal
    steer_layer = args.steer_layer if args.steer_layer is not None else (2 * n_layers // 3)
    steer_layers = [steer_layer]
    print(f"Model layers: {n_layers}, steering at: {steer_layers}")

    # expand anti_steer into per-alpha conditions
    conditions = []
    for c in base_conditions:
        if c == "anti_steer":
            for a in alphas:
                conditions.append(f"anti_steer_a{a}")
        else:
            conditions.append(c)

    print("Loading sentence-transformers for diversity scoring...")
    embed_fn = make_embed_fn()

    notify("anti-steer-001 started",
           f"model={args.model} n_tasks={args.n_tasks} n_ideas={args.n_ideas} alphas={alphas}")

    results: list[dict[str, Any]] = []
    t0 = time.time()

    for task_idx, task in enumerate(tasks):
        task_result: dict[str, Any] = {"task_idx": task_idx, "task": task, "conditions": {}}

        for cond in conditions:
            t_cond = time.time()
            if cond == "prompt_only":
                ideas = run_prompt_only(model, tokenizer, task, args.n_ideas)
            elif cond == "output_rerank":
                ideas = run_output_rerank(
                    model, tokenizer, embed_fn, task, args.n_ideas, args.candidates
                )
            elif cond.startswith("anti_steer_a"):
                alpha = float(cond.split("_a")[1])
                ideas = run_anti_steer(
                    model, tokenizer, task, args.n_ideas, steer_layers, alpha
                )
            else:
                raise ValueError(f"Unknown condition: {cond}")

            diversity = score_diversity(ideas, embed_fn)
            task_result["conditions"][cond] = {
                "ideas": ideas,
                "diversity": diversity,
                "elapsed_s": round(time.time() - t_cond, 1),
            }
            print(
                f"  task={task_idx} cond={cond} "
                f"mean_sim={diversity['mean_pairwise_sim']:.4f} "
                f"max_prior={diversity['mean_max_to_prior']:.4f} "
                f"({task_result['conditions'][cond]['elapsed_s']}s)"
            )

        results.append(task_result)
        elapsed = time.time() - t0
        notify(
            f"anti-steer-001 task {task_idx+1}/{len(tasks)}",
            f"elapsed={elapsed/60:.1f}min",
        )
        # checkpoint after every task
        out_path.write_text(json.dumps(results, indent=2))

    notify("anti-steer-001 complete", f"tasks={len(tasks)} elapsed={elapsed/60:.1f}min")
    print(f"\nDone. Results saved to {out_path}")
    print(f"Total elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        notify("anti-steer-001 FAILED", str(e))
        raise
