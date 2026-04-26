"""
anti-steer-001 run03 — TIDE (L40, Qwen2.5-3B-Instruct)

Hypothesis: residual anti-steering reduces semantic repetition in
idea sequences on a capable instruction-tuned model.

Conditions per task:
  - prompt_only         : prior ideas in context, no intervention
  - output_rerank       : 3 candidates per step, keep least similar
  - anti_steer_a{alpha} : subtract mean prior-idea activation at layer 2*n//3

Tasks: 30 coding-issue prompts (see TASKS list)
Ideas: 5 per task per condition
Alpha sweep: [0.1, 0.25, 0.5, 1.0, 2.0]
Layer: 2 * n_layers // 3  (single layer to avoid cumulative over-steering)

Output: /home/jovyan/anti_steer001_run03.json
Checkpoint: /home/jovyan/anti_steer001_run03_ckpt.json
"""

import sys, importlib.util as _ilu
_real_find_spec = _ilu.find_spec
def _patched(name, package=None, target=None):
    return None if name == "torchvision" else _real_find_spec(name, package)
_ilu.find_spec = _patched
for _k in list(sys.modules):
    if "torchvision" in _k: del sys.modules[_k]

import json, os as _os, time
from pathlib import Path
import json as _json, urllib.request as _urlreq

# ── notify ────────────────────────────────────────────────────────────────────
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

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "Qwen/Qwen2.5-3B-Instruct"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.bfloat16
N_IDEAS       = 5
N_CANDIDATES  = 3   # output_rerank candidates per step
ALPHAS        = [0.1, 0.25, 0.5, 1.0, 2.0]
MAX_NEW_TOKENS = 200
CANON_TOKENS  = 80
OUT_PATH      = Path("/home/jovyan/anti_steer001_run03.json")
CKPT_PATH     = Path("/home/jovyan/anti_steer001_run03_ckpt.json")

# ── tasks ─────────────────────────────────────────────────────────────────────
TASKS = [
    "How should we handle cache invalidation when a user updates their profile photo? The cache is Redis-backed and serves multiple microservices.",
    "Our CDN serves stale CSS/JS assets after deployments. Propose approaches to ensure clients get the new files without hard-coding cache busters.",
    "Our REST API returns paginated results via offset/limit. As the dataset grows, deep offsets are getting slow. Suggest approaches to speed up or replace offset-based pagination.",
    "We need to add infinite scroll to a React app that fetches paginated posts. Propose implementation approaches for the frontend pagination logic.",
    "Our payment service occasionally returns transient 503 errors. Propose strategies for handling retries without double-charging customers.",
    "External API calls in our background jobs sometimes fail silently. Suggest approaches to surface and handle these failures reliably.",
    "Users report being logged out unexpectedly. We use JWT tokens with a 1-hour expiry. Propose approaches to extend sessions without compromising security.",
    "We need to support single sign-on across three internal tools. Suggest implementation approaches for a lightweight SSO layer.",
    "We need to add a 'soft delete' feature to our users table. Propose approaches that minimize impact on existing queries.",
    "Our PostgreSQL table has grown to 200M rows and queries are slow. Suggest approaches to improve read performance without a full rewrite.",
    "Users need to upload video files up to 2GB. Our current upload endpoint times out on large files. Propose approaches for reliable large-file uploads.",
    "We store user-generated images in S3 but need to serve resized thumbnails. Suggest approaches for on-demand image resizing.",
    "Our product search returns irrelevant results when users misspell queries. Propose approaches to improve search quality for noisy input.",
    "We need to add full-text search to a PostgreSQL-backed product catalog. Suggest approaches that avoid adding a separate search service.",
    "Our public API is being hammered by a single client. Propose approaches to implement per-client rate limiting.",
    "We have no visibility into slow database queries in production. Suggest approaches to instrument and surface query performance.",
    "Log files across 20 microservices are hard to correlate. Propose approaches for distributed tracing without adopting a full APM platform.",
    "Background jobs occasionally run twice when a worker crashes mid-task. Propose approaches to make job processing idempotent.",
    "Our email-sending job queue backs up during marketing campaigns. Suggest approaches to handle burst traffic without dropping messages.",
    "We want to roll out a new checkout flow to 10% of users. Propose approaches for percentage-based feature rollouts.",
    "Secrets are currently hardcoded in config files committed to git. Suggest approaches to manage secrets without breaking local dev.",
    "Integration tests hit the real Stripe API and are flaky in CI. Propose approaches to make the test suite reliable without removing coverage.",
    "Test suite runtime has grown to 45 minutes. Suggest approaches to speed it up without removing tests.",
    "We need zero-downtime deployments for a stateful WebSocket server. Propose approaches that don't require clients to reconnect.",
    "A bad deploy caused data corruption last month. Suggest approaches for safer database migrations during deployments.",
    "Two concurrent requests can both read a counter, increment it, and write back the same value. Propose approaches to prevent this race condition.",
    "We need to generate unique sequential order IDs across multiple application servers. Suggest approaches that don't create a bottleneck.",
    "Our React app re-renders the entire product list on every filter change. Propose approaches to reduce unnecessary re-renders.",
    "Initial page load is slow because all JavaScript is bundled into one file. Suggest approaches to improve load time for first-time visitors.",
    "Clients are calling our API in a tight loop to poll for job status. Propose approaches to replace polling with a push-based mechanism.",
]
assert len(TASKS) == 30

# ── hooks ─────────────────────────────────────────────────────────────────────

def _get_block(model, layer_idx):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise AttributeError(f"Cannot find decoder blocks in {type(model).__name__}")

def _num_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise AttributeError(f"Cannot determine num layers for {type(model).__name__}")


class ActivationCapture:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self._handles = []
        self._store = {l: [] for l in layers}

    def __enter__(self):
        for l in self.layers:
            h = _get_block(self.model, l).register_forward_hook(self._make_hook(l))
            self._handles.append(h)
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, l):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._store[l].append(hidden.mean(dim=1).squeeze(0).detach().float().cpu())
        return hook

    def get_mean_vectors(self):
        return {l: torch.stack(v).mean(0) for l, v in self._store.items() if v}


class SteeringIntervention:
    def __init__(self, model, layers, alpha, vectors):
        self.model = model
        self.layers = layers
        self.alpha = alpha
        self.vectors = vectors
        self._handles = []

    def __enter__(self):
        for l in self.layers:
            if l not in self.vectors:
                continue
            vec = self.vectors[l].to(dtype=torch.float32)
            def hook(module, input, output, _vec=vec):
                is_tuple = isinstance(output, tuple)
                h = output[0] if is_tuple else output
                h = h.float() - self.alpha * _vec.to(h.device).unsqueeze(0).unsqueeze(0)
                h = h.to(dtype=DTYPE)
                return (h,) + output[1:] if is_tuple else h
            self._handles.append(_get_block(self.model, l).register_forward_hook(hook))
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ── generation helpers ────────────────────────────────────────────────────────

IDEA_PROMPT = (
    "Issue: {task}\n\n"
    "Propose one specific implementation approach. Name the core mechanism, "
    "data structure, or algorithm concretely. One short paragraph only."
)

IDEA_PROMPT_WITH_PRIOR = (
    "Issue: {task}\n\n"
    "Approaches already considered:\n{prior_block}\n\n"
    "Propose a mechanistically distinct implementation approach. "
    "Do not restate any approach above. One short paragraph only."
)

CANON_PROMPT = (
    "Summarise the core implementation mechanism of this idea in one sentence, "
    "ignoring wording and style:\n\n{idea}"
)


def _chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate(model, tokenizer, prompt_text, max_new_tokens):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()


def _generate_steered(model, tokenizer, prompt_text, steer_layers, alpha, vectors):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    with SteeringIntervention(model, steer_layers, alpha, vectors):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()


def _capture_vector(model, tokenizer, idea, steer_layers):
    prompt = _chat(tokenizer, CANON_PROMPT.format(idea=idea[:400]))
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with ActivationCapture(model, steer_layers) as cap:
        with torch.no_grad():
            model(**inputs)
    return cap.get_mean_vectors()


def _merge_vectors(vecs_list, layers):
    merged = {}
    for l in layers:
        all_v = [v[l] for v in vecs_list if l in v]
        if all_v:
            merged[l] = torch.stack(all_v).mean(0)
    return merged


def _build_prompt(task, prior_ideas):
    if not prior_ideas:
        return _chat(tokenizer_g, IDEA_PROMPT.format(task=task))
    prior_block = "\n".join(f"{i+1}. {idea[:200]}" for i, idea in enumerate(prior_ideas))
    return _chat(tokenizer_g, IDEA_PROMPT_WITH_PRIOR.format(task=task, prior_block=prior_block))


# module-level tokenizer reference (set in main before any generation)
tokenizer_g = None


# ── conditions ────────────────────────────────────────────────────────────────

def run_prompt_only(model, tokenizer, task):
    ideas = []
    for _ in range(N_IDEAS):
        idea = _generate(model, tokenizer, _build_prompt(task, ideas), MAX_NEW_TOKENS)
        ideas.append(idea)
    return ideas


def run_output_rerank(model, tokenizer, embed_fn, task):
    ideas = []
    idea_embs = []
    for step in range(N_IDEAS):
        prompt = _build_prompt(task, ideas)
        n_cands = N_CANDIDATES if step > 0 else 1
        candidates = [_generate(model, tokenizer, prompt, MAX_NEW_TOKENS) for _ in range(n_cands)]
        if step == 0 or not idea_embs:
            chosen = candidates[0]
        else:
            cand_embs = embed_fn(candidates)
            prior_matrix = np.stack(idea_embs)
            scores = [np.dot(prior_matrix, ce).mean() for ce in cand_embs]
            chosen = candidates[int(np.argmin(scores))]
        ideas.append(chosen)
        idea_embs.append(embed_fn([chosen])[0])
    return ideas


def run_anti_steer(model, tokenizer, task, steer_layers, alpha):
    ideas = []
    prior_vecs = []
    for step in range(N_IDEAS):
        prompt = _build_prompt(task, ideas)
        if step == 0 or not prior_vecs:
            idea = _generate(model, tokenizer, prompt, MAX_NEW_TOKENS)
        else:
            steering = _merge_vectors(prior_vecs, steer_layers)
            idea = _generate_steered(model, tokenizer, prompt, steer_layers, alpha, steering)
        ideas.append(idea)
        prior_vecs.append(_capture_vector(model, tokenizer, idea, steer_layers))
    return ideas


# ── diversity scoring ─────────────────────────────────────────────────────────

def score_diversity(ideas, embed_fn):
    embs = embed_fn(ideas)
    n = len(embs)
    sims = [float(np.dot(embs[i], embs[j]))
            for i in range(n) for j in range(i+1, n)]
    max_prior = []
    for i in range(1, n):
        max_prior.append(max(float(np.dot(embs[i], embs[j])) for j in range(i)))
    return {
        "mean_pairwise_sim": float(np.mean(sims)) if sims else 0.0,
        "mean_max_to_prior": float(np.mean(max_prior)) if max_prior else 0.0,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global tokenizer_g

    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer_g = tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    model.eval()

    n_layers = _num_layers(model)
    steer_layer = 2 * n_layers // 3
    steer_layers = [steer_layer]
    print(f"n_layers={n_layers}, steer_layer={steer_layer}")

    print("Loading sentence-transformers...")
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    def embed_fn(texts):
        return st.encode(texts, normalize_embeddings=True)

    # build condition list
    conditions = ["prompt_only", "output_rerank"] + [f"anti_steer_a{a}" for a in ALPHAS]
    print(f"Conditions: {conditions}")

    notify("anti-steer-001 run03 started",
           f"model={MODEL_NAME} tasks={len(TASKS)} ideas={N_IDEAS}")

    results = []
    # resume from checkpoint if exists
    if CKPT_PATH.exists():
        results = json.loads(CKPT_PATH.read_text())
        done_ids = {r["task_idx"] for r in results}
        print(f"Resuming from checkpoint: {len(results)} tasks done")
    else:
        done_ids = set()

    t0 = time.time()

    for task_idx, task in enumerate(TASKS):
        if task_idx in done_ids:
            continue

        task_result = {"task_idx": task_idx, "task": task, "conditions": {}}

        for cond in conditions:
            t_cond = time.time()
            if cond == "prompt_only":
                ideas = run_prompt_only(model, tokenizer, task)
            elif cond == "output_rerank":
                ideas = run_output_rerank(model, tokenizer, embed_fn, task)
            elif cond.startswith("anti_steer_a"):
                alpha = float(cond.split("_a")[1])
                ideas = run_anti_steer(model, tokenizer, task, steer_layers, alpha)
            else:
                raise ValueError(cond)

            div = score_diversity(ideas, embed_fn)
            task_result["conditions"][cond] = {
                "ideas": ideas,
                "diversity": div,
                "elapsed_s": round(time.time() - t_cond, 1),
            }
            print(
                f"  task={task_idx} cond={cond} "
                f"sim={div['mean_pairwise_sim']:.4f} "
                f"({task_result['conditions'][cond]['elapsed_s']}s)"
            )

        results.append(task_result)
        CKPT_PATH.write_text(json.dumps(results, indent=2))

        elapsed = time.time() - t0
        notify(f"run03 task {task_idx+1}/{len(TASKS)}",
               f"elapsed={elapsed/60:.1f}min")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    elapsed = time.time() - t0
    notify("anti-steer-001 run03 complete",
           f"tasks={len(results)} elapsed={elapsed/60:.1f}min")
    print(f"\nDone. elapsed={elapsed/60:.1f}min  output={OUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        notify("anti-steer-001 run03 FAILED", str(e))
        raise
