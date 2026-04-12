"""
Experiment 49: Simulated Annealing from new SOTA (CE=0.5993) with topk=200

Motivation:
- Exp36: SA from old SOTA (0.6044) with topk=100 — found no improvement.
- Now we have a new SOTA at 0.5993, and topk=200 is known to expose better candidates.
- Two changes vs Exp36:
    1. Start from new SOTA (lower energy starting point)
    2. Use topk=200 candidates (4× wider than Exp36's topk=100 was 2×)
    3. More SA steps (3000 vs 1500) — the landscape near 0.5993 may be deeper
- T schedule: exponential 0.05→0.001 over 3000 steps (same as Exp36 but more steps)

New SOTA IDs: [50105, 111, 133522, 222115, 24539, 202257, 10358, 131146,
               73815, 242580, 231898, 2976, 55135, 5598, 31459, 19493]

Timing estimate: 3000 × (2s grad + 0.2s eval) ≈ 6600s (~1.8h)
Output: /home/jovyan/steer001_sa_new_sota.json
"""

import sys, importlib.util as _ilu

_real_find_spec = _ilu.find_spec
def _patched(name, package=None, target=None):
    return None if name == "torchvision" else _real_find_spec(name, package)
_ilu.find_spec = _patched
for _k in list(sys.modules):
    if "torchvision" in _k: del sys.modules[_k]

import json, math, os as _os, random, time
from pathlib import Path
import json as _json, urllib.request as _urlreq

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

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "google/gemma-2-2b-it"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE        = torch.bfloat16
REF_PREFIX   = "Talk only about cats."
PREFIX_LEN   = 16
SA_STEPS     = 3000
TOPK         = 200
T_START      = 0.05
T_END        = 0.001
BATCH_SIZE   = 12
EARLY_K      = 32
EARLY_WEIGHT = 3.0
PLACEHOLDER  = "SOFTPREFIX"
RANDOM_SEED  = 0
OUT_PATH     = Path("/home/jovyan/steer001_sa_new_sota.json")
CKPT_PATH    = Path("/home/jovyan/steer001_sa_new_sota_ckpt.json")
RECOMPUTE_GRAD_EVERY = 10

# New SOTA from Exp46 (topk=200 HotFlip from old SOTA)
NEW_SOTA_CE  = 0.599288
START_IDS    = [50105, 111, 133522, 222115, 24539, 202257, 10358, 131146,
                73815, 242580, 231898, 2976, 55135, 5598, 31459, 19493]
OLD_SOTA_CE  = 0.60437

SUFFIXES = [
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


try:
    log("=== Exp49: SA from new SOTA (CE=0.5993), topk=200, 3000 steps ===")
    log(f"CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")

    log(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map="cuda:0")
    model.eval()
    log(f"Model loaded in {time.time()-t0:.1f}s | {gpu_mem_str()}")
    notify("Exp49 started", f"SA from new_SOTA={NEW_SOTA_CE:.5f} | topk=200 | 3000 steps")

    PLACEHOLDER_IDS = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)
    embed_fn = model.get_input_embeddings()
    EMB_DIM  = embed_fn.weight.shape[1]
    EMB_DEV  = embed_fn.weight.device

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

    def build_batch(soft_prefix_LD, suffix_texts, ref_completions):
        seqs, meta = [], []
        for suf, ref_comp in zip(suffix_texts, ref_completions):
            pre_ids, post_ids = get_template_split(suf)
            comp_dev = ref_comp.to(EMB_DEV)
            with torch.no_grad():
                pre_emb  = embed_fn(pre_ids.unsqueeze(0).to(EMB_DEV))
                post_emb = embed_fn(post_ids.unsqueeze(0).to(EMB_DEV))
                comp_emb = embed_fn(comp_dev.unsqueeze(0))
            soft_1LD = soft_prefix_LD.to(dtype=pre_emb.dtype, device=pre_emb.device).unsqueeze(0)
            seq = torch.cat([pre_emb, soft_1LD, post_emb, comp_emb], dim=1)
            comp_start = pre_emb.shape[1] + PREFIX_LEN + post_emb.shape[1]
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
        total = torch.tensor(0.0, device=logits.device)
        weight = 0.0
        for b, (comp_start, comp_ids) in enumerate(meta):
            for i, tok in enumerate(comp_ids):
                pos = comp_start + i - 1
                if pos >= T_max: continue
                w = EARLY_WEIGHT if i < EARLY_K else 1.0
                ce = F.cross_entropy(logits[b, pos].unsqueeze(0), tok.unsqueeze(0).long())
                total = total + w * ce
                weight += w
        return (total / weight) if weight > 0 else total

    def compute_ce_discrete(prefix_ids_L, suffix_texts, ref_completions):
        prefix_ids_L = prefix_ids_L.to(EMB_DEV)
        with torch.no_grad():
            soft = embed_fn(prefix_ids_L)
            batch_emb, meta, T_max = build_batch(soft, suffix_texts, ref_completions)
            logits = model(inputs_embeds=batch_emb).logits
            loss = compute_ce_from_batch(logits, meta, T_max)
        return loss.item()

    def compute_grad_candidates(current_ids, ref_completions):
        current_ids = current_ids.to(EMB_DEV)
        prefix_emb  = embed_fn(current_ids).float().detach().requires_grad_(True)
        batch_emb, meta, T_max = build_batch(prefix_emb, SUFFIXES, ref_completions)
        logits = model(inputs_embeds=batch_emb).logits
        loss   = compute_ce_from_batch(logits, meta, T_max)
        loss.backward()
        grad = prefix_emb.grad
        W = embed_fn.weight.float()
        candidates = []
        for pos in range(PREFIX_LEN):
            scores = W @ grad[pos]
            scores[list(ref_ids_set)] = float('inf')
            scores[tokenizer.bos_token_id] = float('inf')
            scores[tokenizer.eos_token_id] = float('inf')
            if tokenizer.pad_token_id is not None:
                scores[tokenizer.pad_token_id] = float('inf')
            cands = scores.topk(TOPK, largest=False).indices.tolist()
            candidates.append(cands)
        return candidates, loss.item()

    log(f"Generating reference completions for {REF_PREFIX!r}...")
    eos = tokenizer.eos_token_id
    ref_completions = []
    for i, suf in enumerate(SUFFIXES):
        inp = torch.tensor(
            chat_ids([{"role": "user", "content": f"{REF_PREFIX}\n\n{suf}"}]),
            dtype=torch.long
        ).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=80, do_sample=False, pad_token_id=eos)
        comp = out[0, inp.shape[1]:]
        keep = [j for j, t in enumerate(comp.tolist()) if t != eos]
        trimmed = comp[: keep[-1] + 1] if keep else comp[:1]
        ref_completions.append(trimmed.cpu())
    log("Reference completions ready.")

    ref_ids_set = set(tokenizer.encode(REF_PREFIX, add_special_tokens=False))
    rng = random.Random(RANDOM_SEED)

    log(f"\n--- SA: {SA_STEPS} steps, topk={TOPK}, T {T_START}→{T_END} ---")
    start_text = tokenizer.decode(START_IDS)
    log(f"Starting prefix: {start_text!r}  CE={NEW_SOTA_CE:.5f}")

    current_ids = torch.tensor(START_IDS, dtype=torch.long, device=EMB_DEV)
    current_ce  = compute_ce_discrete(current_ids, SUFFIXES, ref_completions)
    log(f"Verified starting CE: {current_ce:.6f}")

    best_ids = current_ids.clone()
    best_ce  = current_ce
    sa_log = [current_ce]
    accepts_good = accepts_bad = rejects = 0
    candidates, _ = compute_grad_candidates(current_ids, ref_completions)
    last_grad_step = 0

    t_start = time.time()
    for step in range(SA_STEPS):
        frac = step / (SA_STEPS - 1)
        T = T_START * (T_END / T_START) ** frac

        if step - last_grad_step >= RECOMPUTE_GRAD_EVERY:
            candidates, _ = compute_grad_candidates(current_ids, ref_completions)
            last_grad_step = step

        pos  = rng.randint(0, PREFIX_LEN - 1)
        cand = rng.choice(candidates[pos])
        trial_ids = current_ids.clone()
        trial_ids[pos] = cand
        trial_ce = compute_ce_discrete(trial_ids, SUFFIXES, ref_completions)

        delta = trial_ce - current_ce
        if delta < 0:
            current_ids = trial_ids
            current_ce  = trial_ce
            accepts_good += 1
        else:
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                current_ids = trial_ids
                current_ce  = trial_ce
                accepts_bad += 1
            else:
                rejects += 1

        if current_ce < best_ce:
            best_ce  = current_ce
            best_ids = current_ids.clone()
            notify(f"Exp49 SA NEW BEST step={step}",
                   f"CE={best_ce:.6f}  delta={best_ce-NEW_SOTA_CE:+.6f}")

        sa_log.append(current_ce)

        if step % 200 == 0 or step == SA_STEPS - 1:
            elapsed = time.time() - t_start
            text = tokenizer.decode(current_ids.cpu().tolist())
            log(f"  [{step:5d}/{SA_STEPS}] CE={current_ce:.6f}  best={best_ce:.6f}  "
                f"T={T:.5f}  +↓={accepts_good}  +↑={accepts_bad}  -rej={rejects}  "
                f"elapsed={elapsed:.0f}s  {text[:60]!r}")

        if step % 300 == 0:
            notify(f"Exp49 SA step={step}", f"CE={current_ce:.5f}  best={best_ce:.5f}  T={T:.4f}")

        if step % 50 == 0:
            ckpt = {"step": step, "current_ce": current_ce, "best_ce": best_ce,
                    "best_ids": best_ids.tolist(),
                    "best_text": tokenizer.decode(best_ids.cpu().tolist())}
            with open(CKPT_PATH, "w") as f:
                json.dump(ckpt, f)

    elapsed = time.time() - t_start
    best_text = tokenizer.decode(best_ids.cpu().tolist())

    log(f"\n=== FINAL SUMMARY ===")
    log(f"  New SOTA:      CE={NEW_SOTA_CE:.6f}")
    log(f"  SA best:       CE={best_ce:.6f}  ({'BEATS' if best_ce < NEW_SOTA_CE else 'no improvement'})")
    log(f"  Improvement:   {NEW_SOTA_CE - best_ce:+.6f}")
    log(f"  Best prefix:   {best_text!r}")
    log(f"  Accepts ↓:     {accepts_good}")
    log(f"  Accepts ↑ SA:  {accepts_bad}")
    log(f"  Rejects:       {rejects}")
    log(f"  Time:          {elapsed:.0f}s")

    if best_ce < NEW_SOTA_CE:
        notify("Exp49 NEW SOTA", f"CE={best_ce:.6f}  beats {NEW_SOTA_CE:.6f}  {best_text[:60]!r}")
    else:
        notify("Exp49 complete", f"best={best_ce:.6f}  start={NEW_SOTA_CE:.6f}  no improvement")

    results = {
        "experiment": "steer-001-exp49-sa-new-sota",
        "model": MODEL_NAME,
        "reference_prefix": REF_PREFIX,
        "prefix_len": PREFIX_LEN,
        "sa_steps": SA_STEPS,
        "topk": TOPK,
        "t_start": T_START,
        "t_end": T_END,
        "recompute_grad_every": RECOMPUTE_GRAD_EVERY,
        "random_seed": RANDOM_SEED,
        "new_sota_ce": NEW_SOTA_CE,
        "old_sota_ce": OLD_SOTA_CE,
        "start_ids": START_IDS,
        "verified_start_ce": current_ce if step == 0 else None,
        "metrics": {
            "best_ce": best_ce,
            "improvement": NEW_SOTA_CE - best_ce,
            "accepts_good": accepts_good,
            "accepts_bad": accepts_bad,
            "rejects": rejects,
            "elapsed_seconds": elapsed,
        },
        "best_text": best_text,
        "best_ids": best_ids.tolist(),
        "sa_log": sa_log,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {OUT_PATH}")

except Exception as e:
    notify("Exp49 FAILED", str(e))
    raise
