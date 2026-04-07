"""
Experiment 23: Held-out Suffix Generalization Evaluation

Motivation (from research report):
The current CE metric is computed on the same 12 suffixes used during
optimization. This makes it impossible to distinguish "the prefix genuinely
steers behavior across diverse prompts" from "the prefix memorized the 12
training suffixes." The report explicitly flags this as a critical limitation.

This experiment measures the generalization gap:
- Evaluate Exp19 SOTA prefix (CE=0.6794) on the 12 training suffixes
- Evaluate the same prefix on 20 *held-out* suffixes never seen in optimization
- Compute the in-sample vs out-of-sample CE gap as a measure of reliability
- Also evaluate Exp11 baseline (len=8, CE=0.689) for comparison

If the held-out CE is close to the in-sample CE, the prefix generalizes well.
If the held-out CE is substantially higher, the results may overfit to the
small training set.

Exp19 SOTA prefix IDs (PREFIX_LEN=16, hardcoded from results):
[66589, 232524, 73540, 12027, 12827, 598, 74257, 72835,
 68089, 6899, 19475, 113869, 103482, 19493, 15641, 5105]

Exp11 baseline prefix IDs (PREFIX_LEN=8, best result, CE=0.689):
[6828, 44013, 169091, 77616, 48982, 7260, 675, 19493]

Output: /home/jovyan/steer001_holdout_eval.json
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

import os as _os, json as _json, urllib.request as _urlreq

def notify(title, body=""):
    """Pushbullet push. No-ops silently if PUSHBULLET_API_KEY not set."""
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
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME  = "google/gemma-2-2b-it"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.bfloat16
REF_PREFIX  = "Talk only about cats."
PLACEHOLDER = "SOFTPREFIX"
EARLY_K     = 32
EARLY_WEIGHT = 3.0
OUT_PATH    = Path("/home/jovyan/steer001_holdout_eval.json")

# Prefix configs to evaluate
PREFIXES_TO_EVAL = [
    {
        "name": "Exp19_SOTA",
        "ids": [66589, 232524, 73540, 12027, 12827, 598, 74257, 72835,
                68089, 6899, 19475, 113869, 103482, 19493, 15641, 5105],
        "prefix_len": 16,
        "reported_ce": 0.6794,
        "description": "Exp19 SOTA: ST+cosine+best-prefix, PREFIX_LEN=16, seed=42",
    },
    {
        "name": "Exp11_baseline",
        "ids": [6828, 44013, 169091, 77616, 48982, 7260, 675, 19493],
        "prefix_len": 8,
        "reported_ce": 0.68935,
        "description": "Exp11 baseline: pure ST estimator, PREFIX_LEN=8, seed=42",
    },
    {
        "name": "Exp16_lam0",
        "ids": [63053, 50105, 120676, 9787, 38789, 6828, 30382, 115195],
        "prefix_len": 8,
        "reported_ce": 0.6861,
        "description": "Exp16 λ=0.0: fp32-sims ST+best-prefix, PREFIX_LEN=8, seed=42",
    },
]

# ── Suffix sets ───────────────────────────────────────────────────────────────
TRAIN_SUFFIXES = [
    "What should I make for dinner tonight?",
    "Explain the water cycle to me.",
    "How do I improve my time management?",
    "Tell me something interesting about space.",
    "What are some good exercises for beginners?",
    "How does the internet work?",
    "Give me a recipe for chocolate chip cookies.",
    "What's the best way to learn a new language?",
    "Describe what makes a good friend.",
    "How do I start a garden?",
    "What is photosynthesis?",
    "Give me tips for better sleep.",
]

HOLDOUT_SUFFIXES = [
    "What causes inflation in an economy?",
    "How do vaccines work?",
    "Explain the rules of chess.",
    "What's the best way to save money?",
    "How do I deal with difficult coworkers?",
    "What are the main causes of climate change?",
    "Recommend a movie I should watch tonight.",
    "How do I fix a leaky faucet?",
    "What is the difference between empathy and sympathy?",
    "Give me some creative writing prompts.",
    "How do I prepare for a job interview?",
    "What causes earthquakes?",
    "Can you explain quantum computing simply?",
    "What are some tips for staying motivated?",
    "How do I make sourdough bread?",
    "What's a good way to start meditating?",
    "How does the stock market work?",
    "What are the health benefits of exercise?",
    "What are the main differences between democracy and authoritarianism?",
    "How do I reduce stress in my daily life?",
]

def gpu_mem_str():
    if not torch.cuda.is_available(): return "no GPU"
    parts = []
    for i in range(torch.cuda.device_count()):
        a = torch.cuda.memory_allocated(i) / 1e9
        r = torch.cuda.memory_reserved(i) / 1e9
        parts.append(f"GPU{i}: {a:.1f}/{r:.1f}GB")
    return "  ".join(parts)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

log("=== Exp 23: Held-out Suffix Generalization Evaluation ===")
log(f"CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
notify("Exp23 starting", f"Holdout eval: {len(TRAIN_SUFFIXES)} train, {len(HOLDOUT_SUFFIXES)} holdout")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    log(f"  GPU {i}: {p.name}")

log(f"Loading {MODEL_NAME}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map="cuda:0")
model.eval()
log(f"Model loaded in {time.time()-t0:.1f}s | {gpu_mem_str()}")

PLACEHOLDER_IDS = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)
embed_fn = model.get_input_embeddings()
EMB_DIM  = embed_fn.weight.shape[1]
EMB_DEV  = embed_fn.weight.device

# ── Helpers ──────────────────────────────────────────────────────────────────

def chat_ids(messages):
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    return tokenizer.encode(text, add_special_tokens=False)

def find_subseq(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i+len(sub)] == sub: return i
    return None

def get_template_split(suffix_text):
    msgs = [{"role": "user", "content": f"{PLACEHOLDER}\n\n{suffix_text}"}]
    ids_list = chat_ids(msgs)
    ph_start = find_subseq(ids_list, PLACEHOLDER_IDS)
    if ph_start is None:
        ph_start, ph_len = 1, 0
    else:
        ph_len = len(PLACEHOLDER_IDS)
    ids = torch.tensor(ids_list, dtype=torch.long)
    return ids[:ph_start], ids[ph_start + ph_len:]

def build_batch(prefix_ids_L, suffix_texts, ref_completions):
    seqs, meta = [], []
    for suf, ref_comp in zip(suffix_texts, ref_completions):
        pre_ids, post_ids = get_template_split(suf)
        comp_dev = ref_comp.to(EMB_DEV)
        with torch.no_grad():
            pre_emb   = embed_fn(pre_ids.unsqueeze(0).to(EMB_DEV))
            post_emb  = embed_fn(post_ids.unsqueeze(0).to(EMB_DEV))
            comp_emb  = embed_fn(comp_dev.unsqueeze(0))
            prefix_emb = embed_fn(prefix_ids_L.unsqueeze(0).to(EMB_DEV))
        seq = torch.cat([pre_emb, prefix_emb, post_emb, comp_emb], dim=1)
        comp_start = pre_emb.shape[1] + prefix_ids_L.shape[0] + post_emb.shape[1]
        seqs.append(seq)
        meta.append((comp_start, comp_dev))
    T_max = max(s.shape[1] for s in seqs)
    padded = []
    for seq in seqs:
        pad_len = T_max - seq.shape[1]
        if pad_len:
            pad = torch.zeros(1, pad_len, EMB_DIM, device=seq.device, dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=1)
        padded.append(seq)
    return torch.cat(padded, dim=0).to(EMB_DEV), meta, T_max

def compute_ce_from_batch(logits, meta, T_max):
    total  = 0.0
    weight = 0.0
    for b, (comp_start, comp_ids) in enumerate(meta):
        for i, tok in enumerate(comp_ids):
            pos = comp_start + i - 1
            if pos >= T_max: continue
            w = EARLY_WEIGHT if i < EARLY_K else 1.0
            ce = F.cross_entropy(
                logits[b, pos].unsqueeze(0), tok.unsqueeze(0).long()).item()
            total += w * ce
            weight += w
    return (total / weight) if weight > 0 else float('nan')

def eval_prefix_on_suffixes(prefix_ids, suffix_texts, ref_completions):
    """Compute mean CE of a discrete prefix on a list of suffixes."""
    with torch.no_grad():
        batch_emb, meta, T_max = build_batch(prefix_ids, suffix_texts, ref_completions)
        logits = model(inputs_embeds=batch_emb).logits
    return compute_ce_from_batch(logits, meta, T_max)

def generate_ref_completions(suffix_list, label=""):
    log(f"Generating reference completions for {label!r} ({len(suffix_list)} suffixes)...")
    eos = tokenizer.eos_token_id
    completions = []
    for i, suf in enumerate(suffix_list):
        inp = torch.tensor(
            chat_ids([{"role": "user", "content": f"{REF_PREFIX}\n\n{suf}"}]),
            dtype=torch.long
        ).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=80, do_sample=False, pad_token_id=eos)
        comp = out[0, inp.shape[1]:]
        keep = [j for j, t in enumerate(comp.tolist()) if t != eos]
        trimmed = comp[: keep[-1] + 1] if keep else comp[:1]
        completions.append(trimmed.cpu())
        decoded = tokenizer.decode(trimmed, skip_special_tokens=True)
        log(f"  [{i:2d}] {suf[:40]!r:43s} → {decoded[:65]!r}")
    return completions

# ── Generate reference completions ────────────────────────────────────────────
train_refs   = generate_ref_completions(TRAIN_SUFFIXES, "train")
holdout_refs = generate_ref_completions(HOLDOUT_SUFFIXES, "holdout")
log("All reference completions ready.")

# ── Evaluate each prefix ──────────────────────────────────────────────────────
all_results = []

try:
    for cfg in PREFIXES_TO_EVAL:
        name = cfg["name"]
        ids  = torch.tensor(cfg["ids"], dtype=torch.long)
        plen = cfg["prefix_len"]
        text = tokenizer.decode(cfg["ids"])

        log(f"\n{'='*60}")
        log(f"=== {name} (len={plen}) ===")
        log(f"  Text: {text!r}")
        log(f"  Reported CE (train): {cfg['reported_ce']:.4f}")

        t0 = time.time()
        train_ce   = eval_prefix_on_suffixes(ids, TRAIN_SUFFIXES, train_refs)
        holdout_ce = eval_prefix_on_suffixes(ids, HOLDOUT_SUFFIXES, holdout_refs)
        elapsed = time.time() - t0

        gap = holdout_ce - train_ce
        log(f"  Train CE:   {train_ce:.5f}  (reported: {cfg['reported_ce']:.4f})")
        log(f"  Holdout CE: {holdout_ce:.5f}  (gap: {gap:+.5f})")
        log(f"  Elapsed: {elapsed:.1f}s")

        all_results.append({
            "name": name,
            "description": cfg["description"],
            "prefix_len": plen,
            "prefix_ids": cfg["ids"],
            "prefix_text": text,
            "reported_train_ce": cfg["reported_ce"],
            "measured_train_ce": train_ce,
            "measured_holdout_ce": holdout_ce,
            "generalization_gap": gap,
            "elapsed_seconds": elapsed,
        })

        notify(
            f"Exp23 {name} done",
            f"train={train_ce:.4f}  holdout={holdout_ce:.4f}  gap={gap:+.4f}",
        )

except Exception as e:
    notify("Exp23 FAILED", str(e))
    raise

# ── Summary ───────────────────────────────────────────────────────────────────
log(f"\n{'='*60}")
log("=== GENERALIZATION SUMMARY ===")
log(f"{'='*60}")
log(f"  {'Prefix':20s}  {'Train CE':>10}  {'Holdout CE':>10}  {'Gap':>8}")
for r in all_results:
    log(f"  {r['name']:20s}  {r['measured_train_ce']:>10.5f}  "
        f"{r['measured_holdout_ce']:>10.5f}  {r['generalization_gap']:>+8.5f}")

results = {
    "experiment": "steer-001-exp23-holdout-eval",
    "model": MODEL_NAME,
    "reference_prefix": REF_PREFIX,
    "n_train_suffixes": len(TRAIN_SUFFIXES),
    "n_holdout_suffixes": len(HOLDOUT_SUFFIXES),
    "train_suffixes": TRAIN_SUFFIXES,
    "holdout_suffixes": HOLDOUT_SUFFIXES,
    "evaluations": all_results,
}
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
log(f"\nResults saved to {OUT_PATH}")
log("=== DONE ===")

notify(
    "Exp23 complete",
    "  ".join(f"{r['name']}: train={r['measured_train_ce']:.4f} "
              f"holdout={r['measured_holdout_ce']:.4f} gap={r['generalization_gap']:+.4f}"
              for r in all_results),
)
