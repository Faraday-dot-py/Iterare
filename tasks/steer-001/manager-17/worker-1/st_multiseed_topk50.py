"""
Experiment 17: Multi-seed ST + cosine annealing + best-prefix tracking + HotFlip TOPK=50

Motivation:
- Exp 11 (SOTA, 0.689) got lucky with seed=42: proj-CE=0.762.
- Exp 13 (same algo, different seed) got 1.129 — showing high variance.
- Exp 15 tests whether cosine annealing + best-prefix tracking fixes variance (single seed=42).
- Exp 17 tests the same technique across 5 seeds to characterize reliability,
  while also using TOPK=50 (vs 30) to give HotFlip broader search.

Design:
- 5 independent seeds [0, 1, 2, 3, 4]
- Each: 300 ST steps with cosine LR annealing (LR_MAX=0.01 → LR_MIN=0.001) + best-prefix tracking
- Projection from best-seen soft prefix (not final step)
- HotFlip 80 steps, TOPK=50
- GPU: assigned via CUDA_VISIBLE_DEVICES

Key questions answered:
1. Does cosine annealing + best-prefix tracking give low proj-CE reliably across seeds?
2. Does TOPK=50 give better HotFlip CE than TOPK=30?
3. What is the best achievable CE across 5 seeds?

Output: /home/jovyan/steer001_multiseed_topk50.json
"""

import sys, importlib.util as _ilu

_real_find_spec = _ilu.find_spec
def _patched(name, package=None, target=None):
    return None if name == "torchvision" else _real_find_spec(name, package)
_ilu.find_spec = _patched
for _k in list(sys.modules):
    if "torchvision" in _k: del sys.modules[_k]

import json, math, os as _os, time, urllib.request as _urlreq
from pathlib import Path

def notify(title, body=""):
    """Pushbullet push. No-ops silently if PUSHBULLET_API_KEY not set."""
    key = _os.environ.get("PUSHBULLET_API_KEY", "")
    if not key:
        return
    try:
        import json as _json
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
MODEL_NAME    = "google/gemma-2-2b-it"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.bfloat16
REF_PREFIX    = "Talk only about cats."
PLACEHOLDER   = "SOFTPREFIX"
PREFIX_LEN    = 8
SOFT_STEPS    = 300
HOTFLIP_STEPS = 80
HF_TOPK       = 50        # higher than Exp11/15's 30
BATCH_SIZE    = 12
LR_MAX        = 0.01
LR_MIN        = 0.001
EARLY_K       = 32
EARLY_WEIGHT  = 3.0
SEEDS         = [0, 1, 2, 3, 4]
OUT_PATH      = Path("/home/jovyan/steer001_multiseed_topk50.json")

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

log("=== Exp 17: Multi-seed ST+Anneal+BestPrefix + HotFlip TOPK=50 ===")
log(f"CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
notify("Exp17 starting", f"Multi-seed ST+anneal, seeds={SEEDS}, TOPK={HF_TOPK}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    log(f"  GPU {i}: {p.name}")

log(f"Loading {MODEL_NAME}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map="cuda:0")
model.eval()
log(f"Model loaded in {time.time()-t0:.1f}s | {gpu_mem_str()}")
notify("Exp17 started", f"Model loaded | seeds={SEEDS} | {gpu_mem_str()}")

PLACEHOLDER_IDS = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)
embed_fn = model.get_input_embeddings()
EMB_DIM  = embed_fn.weight.shape[1]
VOCAB    = embed_fn.weight.shape[0]
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

def compute_ce_soft_batched(soft_prefix_LD, suffix_texts, ref_completions):
    batch_emb, meta, T_max = build_batch(soft_prefix_LD, suffix_texts, ref_completions)
    logits = model(inputs_embeds=batch_emb).logits
    return compute_ce_from_batch(logits, meta, T_max)

def compute_ce_discrete_batched(prefix_ids_L, suffix_texts, ref_completions):
    prefix_ids_L = prefix_ids_L.to(EMB_DEV)
    with torch.no_grad():
        soft = embed_fn(prefix_ids_L)
        batch_emb, meta, T_max = build_batch(soft, suffix_texts, ref_completions)
        logits = model(inputs_embeds=batch_emb).logits
        loss = compute_ce_from_batch(logits, meta, T_max)
    return loss.item()

def project_to_tokens(soft_LD):
    with torch.no_grad():
        W  = embed_fn.weight.to(device=soft_LD.device, dtype=soft_LD.dtype)
        sn = F.normalize(soft_LD, dim=-1)
        wn = F.normalize(W, dim=-1)
        sims = sn @ wn.T
        ids  = sims.argmax(dim=-1)
    return ids

def st_project(soft_prefix_LD):
    W = embed_fn.weight.to(device=soft_prefix_LD.device, dtype=soft_prefix_LD.dtype)
    sn   = F.normalize(soft_prefix_LD, dim=-1)
    wn   = F.normalize(W, dim=-1)
    sims = sn @ wn.T
    ids  = sims.argmax(dim=-1)
    with torch.no_grad():
        proj_emb = embed_fn(ids).to(dtype=soft_prefix_LD.dtype)
    st_emb = soft_prefix_LD + (proj_emb - soft_prefix_LD).detach()
    return st_emb, ids

def cosine_lr(step, total_steps):
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * step / total_steps))

def hotflip_step_batched(current_ids, ref_completions, topk):
    current_ids = current_ids.to(EMB_DEV)
    prefix_emb  = embed_fn(current_ids).float().detach().requires_grad_(True)
    batch_emb, meta, T_max = build_batch(prefix_emb, SUFFIXES, ref_completions)
    logits = model(inputs_embeds=batch_emb).logits
    loss   = compute_ce_from_batch(logits, meta, T_max)
    loss.backward()
    grad = prefix_emb.grad

    best_ids = current_ids.clone()
    best_ce  = compute_ce_discrete_batched(current_ids, SUFFIXES, ref_completions)
    W = embed_fn.weight.float()

    for pos in range(PREFIX_LEN):
        g      = grad[pos]
        scores = W @ g
        scores[list(ref_ids_set)] = float('inf')
        scores[tokenizer.bos_token_id] = float('inf')
        scores[tokenizer.eos_token_id] = float('inf')
        if tokenizer.pad_token_id is not None:
            scores[tokenizer.pad_token_id] = float('inf')
        cands = scores.topk(topk, largest=False).indices
        for k in range(topk):
            trial_ids = current_ids.clone()
            trial_ids[pos] = cands[k].item()
            ce_k = compute_ce_discrete_batched(trial_ids, SUFFIXES, ref_completions)
            if ce_k < best_ce:
                best_ce = ce_k
                best_ids = trial_ids.clone()

    return best_ids, best_ce

# ── Reference completions ────────────────────────────────────────────────────
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
    decoded = tokenizer.decode(trimmed, skip_special_tokens=True)
    log(f"  [{i:2d}] {suf[:45]!r:48s} → {decoded[:70]!r}")
log("Reference completions ready.")

ref_ids_set = set(tokenizer.encode(REF_PREFIX, add_special_tokens=False))

# ── Embedding stats for initialization ───────────────────────────────────────
with torch.no_grad():
    emb_mean = embed_fn.weight.mean(0)
    emb_std  = embed_fn.weight.std(0) * 0.1

# ── Main loop: run each seed ─────────────────────────────────────────────────
all_results = []
overall_best_ce = float('inf')
overall_best_ids = None

for seed in SEEDS:
    log(f"\n{'='*60}")
    log(f"=== Seed {seed} ({SEEDS.index(seed)+1}/{len(SEEDS)}) ===")
    log(f"{'='*60}")
    notify(f"Exp17 seed={seed} starting", f"{SEEDS.index(seed)+1}/{len(SEEDS)}")

    torch.manual_seed(seed)
    soft_prefix = (emb_mean.unsqueeze(0).repeat(PREFIX_LEN, 1)
                   + torch.randn(PREFIX_LEN, EMB_DIM, device=EMB_DEV, dtype=DTYPE)
                   * emb_std.unsqueeze(0)).detach().float().requires_grad_(True)
    optimizer = torch.optim.Adam([soft_prefix], lr=LR_MAX)

    st_log = []
    lr_log = []
    best_st_ce = float('inf')
    best_soft_snapshot = soft_prefix.data.clone()

    t0 = time.time()
    for step in range(SOFT_STEPS):
        lr = cosine_lr(step, SOFT_STEPS)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        lr_log.append(lr)

        optimizer.zero_grad()
        st_emb, _ = st_project(soft_prefix)
        loss = compute_ce_soft_batched(st_emb, SUFFIXES[:BATCH_SIZE], ref_completions[:BATCH_SIZE])
        loss.backward()
        optimizer.step()

        ce_val = loss.item()
        st_log.append(ce_val)
        if ce_val < best_st_ce:
            best_st_ce = ce_val
            best_soft_snapshot = soft_prefix.data.clone()

        if step % 50 == 0 or step == SOFT_STEPS - 1:
            elapsed = time.time() - t0
            log(f"  [{step:4d}/{SOFT_STEPS}] ST-CE={ce_val:.5f}  best={best_st_ce:.5f}  "
                f"lr={lr:.5f}  elapsed={elapsed:.0f}s  {gpu_mem_str()}")

    t_soft = time.time() - t0
    log(f"ST done: final={st_log[-1]:.5f}, best={best_st_ce:.5f}, time={t_soft:.1f}s")

    # Project best soft prefix
    projected_ids  = project_to_tokens(best_soft_snapshot.to(DTYPE))
    projected_text = tokenizer.decode(projected_ids.cpu().tolist())
    proj_ce = compute_ce_discrete_batched(projected_ids, SUFFIXES, ref_completions)
    log(f"Projection (best): {projected_text!r}  CE={proj_ce:.5f}")

    # HotFlip
    log(f"HotFlip ({HOTFLIP_STEPS} steps, TOPK={HF_TOPK})...")
    current_ids = projected_ids.clone().to(EMB_DEV)
    current_ce  = proj_ce
    hotflip_log = [current_ce]

    t0 = time.time()
    for step in range(HOTFLIP_STEPS):
        new_ids, new_ce = hotflip_step_batched(current_ids, ref_completions, HF_TOPK)
        improved = new_ce < current_ce
        if improved:
            current_ids = new_ids
            current_ce  = new_ce
        hotflip_log.append(current_ce)
        if step % 10 == 0 or step == HOTFLIP_STEPS - 1:
            toks = tokenizer.decode(current_ids.cpu().tolist())
            log(f"  [{step:3d}/{HOTFLIP_STEPS}] CE={current_ce:.5f}  "
                f"{'↓' if improved else '–'}  {toks!r}")

    t_hf = time.time() - t0
    hotflip_ce = current_ce
    final_text = tokenizer.decode(current_ids.cpu().tolist())
    log(f"HotFlip done: CE={hotflip_ce:.5f}, time={t_hf:.1f}s")

    if hotflip_ce < overall_best_ce:
        overall_best_ce  = hotflip_ce
        overall_best_ids = current_ids.tolist()
        log(f"*** NEW OVERALL BEST: {hotflip_ce:.5f} ***")

    all_results.append({
        "seed": seed,
        "best_st_ce": best_st_ce,
        "final_st_ce": st_log[-1],
        "projection_ce": proj_ce,
        "hotflip_ce": hotflip_ce,
        "projected_text": projected_text,
        "final_text": final_text,
        "final_ids": current_ids.tolist(),
        "st_log": st_log,
        "lr_log": lr_log,
        "hotflip_log": hotflip_log,
        "timing": {"soft_seconds": t_soft, "hotflip_seconds": t_hf},
    })

    # Checkpoint: save partial results after each seed so we don't lose work
    ckpt_path = Path("/home/jovyan/steer001_multiseed_topk50_ckpt.json")
    with open(ckpt_path, "w") as f:
        json.dump({"runs": all_results, "completed_seeds": [r["seed"] for r in all_results],
                   "overall_best_hotflip_ce": overall_best_ce}, f, indent=2)
    log(f"Checkpoint saved ({len(all_results)}/{len(SEEDS)} seeds done) → {ckpt_path}")
    _new_best = hotflip_ce < overall_best_ce or (len(all_results) == 1)
    notify(
        f"Exp17 seed={seed} done" + (" *** BEST ***" if hotflip_ce == overall_best_ce else ""),
        f"proj={proj_ce:.4f} → hf={hotflip_ce:.4f} | overall best={overall_best_ce:.4f} | {len(all_results)}/{len(SEEDS)} done",
    )

log(f"\n{'='*60}")
log(f"=== FINAL SUMMARY ===")
log(f"{'='*60}")
log(f"  {'Seed':>6}  {'proj-CE':>10}  {'HotFlip':>10}  prefix")
for r in all_results:
    log(f"  {r['seed']:>6}  {r['projection_ce']:>10.5f}  {r['hotflip_ce']:>10.5f}  {r['final_text']!r}")
log(f"\n  Exp11 baseline: proj=0.762, hotflip=0.689")
log(f"  Overall best HotFlip CE: {overall_best_ce:.5f}")

results = {
    "experiment": "steer-001-exp17-multiseed-topk50",
    "model": MODEL_NAME,
    "reference_prefix": REF_PREFIX,
    "seeds": SEEDS,
    "prefix_len": PREFIX_LEN,
    "n_soft_steps": SOFT_STEPS,
    "n_hotflip_steps": HOTFLIP_STEPS,
    "hf_topk": HF_TOPK,
    "batch_size": BATCH_SIZE,
    "lr_max": LR_MAX,
    "lr_min": LR_MIN,
    "method": "st_cosine_anneal_best_prefix_multiseed",
    "runs": all_results,
    "overall_best_hotflip_ce": overall_best_ce,
    "overall_best_ids": overall_best_ids,
    "exp11_baseline_proj": 0.76245,
    "exp11_baseline_hotflip": 0.68935,
}
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
log(f"\nResults saved to {OUT_PATH}")
log("=== DONE ===")
notify("Exp17 complete", f"Best HF CE={overall_best_ce:.4f} across seeds {SEEDS} | SOTA=0.686")
