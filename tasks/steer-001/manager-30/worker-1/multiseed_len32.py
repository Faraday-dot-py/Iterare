"""
Experiment 30: Multi-seed sweep at PREFIX_LEN=32 (seeds 1, 2)

Motivation:
- Exp26 revealed massive seed variance at PREFIX_LEN=16:
    seed=2: CE=0.6044 (vs seed=42 Exp19 CE=0.6794 — Δ=0.075 improvement)
    seed=1: CE=0.6236
  The seed=42 baseline is NOT privileged; seeds 1/2 are far superior at len=16.
- Exp25 (seed=42, len=32): CE=0.632. If the same advantage holds at len=32,
  seeds 1/2 could push CE well below 0.60 at the SOTA prefix length.
- Hypothesis: seed=2 reaches CE < 0.57 at PREFIX_LEN=32.

Config: seeds=[1, 2], PREFIX_LEN=32, SOFT_STEPS=250, HOTFLIP_STEPS=35,
        HF_TOPK=50, cosine LR 0.01→0.001, BATCH_SIZE=12

Timing estimate: 2 × (~900s soft + 35×237s HF) ≈ 2×9200s ≈ 18400s (~5.1h)

Output: /home/jovyan/steer001_multiseed32.json
"""

import sys, importlib.util as _ilu

_real_find_spec = _ilu.find_spec
def _patched(name, package=None, target=None):
    return None if name == "torchvision" else _real_find_spec(name, package)
_ilu.find_spec = _patched
for _k in list(sys.modules):
    if "torchvision" in _k: del sys.modules[_k]

import json, math, os as _os, time
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

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME    = "google/gemma-2-2b-it"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.bfloat16
REF_PREFIX    = "Talk only about cats."
PLACEHOLDER   = "SOFTPREFIX"
PREFIX_LEN    = 32
SOFT_STEPS    = 250
HOTFLIP_STEPS = 35
HF_TOPK       = 50
BATCH_SIZE    = 12
LR_MAX        = 0.01
LR_MIN        = 0.001
EARLY_K       = 32
EARLY_WEIGHT  = 3.0
SEEDS         = [1, 2]
OUT_PATH      = Path("/home/jovyan/steer001_multiseed32.json")
CKPT_PATH     = Path("/home/jovyan/steer001_multiseed32_ckpt.json")

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
    log("=== Exp 30: Multi-seed sweep at PREFIX_LEN=32 (seeds 1,2) ===")
    log(f"CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        log(f"  GPU {i}: {p.name}")

    log(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map="cuda:0")
    model.eval()
    log(f"Model loaded in {time.time()-t0:.1f}s | {gpu_mem_str()}")
    notify("Exp30 started", f"PREFIX_LEN=32 seeds={SEEDS} | {gpu_mem_str()}")

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
            return (sn @ wn.T).argmax(dim=-1)

    def st_project(soft_prefix_LD):
        W = embed_fn.weight.to(device=soft_prefix_LD.device, dtype=soft_prefix_LD.dtype)
        sn   = F.normalize(soft_prefix_LD, dim=-1)
        wn   = F.normalize(W, dim=-1)
        ids  = (sn @ wn.T).argmax(dim=-1)
        with torch.no_grad():
            proj_emb = embed_fn(ids).to(dtype=soft_prefix_LD.dtype)
        return soft_prefix_LD + (proj_emb - soft_prefix_LD).detach(), ids

    def cosine_lr(step, total_steps):
        return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * step / total_steps))

    def hotflip_step_batched(current_ids, ref_completions):
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
            cands = scores.topk(HF_TOPK, largest=False).indices
            for k in range(HF_TOPK):
                trial_ids = current_ids.clone()
                trial_ids[pos] = cands[k].item()
                ce_k = compute_ce_discrete_batched(trial_ids, SUFFIXES, ref_completions)
                if ce_k < best_ce:
                    best_ce = ce_k
                    best_ids = trial_ids.clone()

        return best_ids, best_ce

    # ── Reference completions ────────────────────────────────────────────────
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

    # ── Multi-seed loop ───────────────────────────────────────────────────────
    overall_best_ce   = float('inf')
    overall_best_ids  = None
    overall_best_seed = None
    all_runs = []

    for seed_idx, seed in enumerate(SEEDS):
        log(f"\n{'='*60}")
        log(f"=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===")
        log(f"{'='*60}")

        with torch.no_grad():
            emb_mean = embed_fn.weight.mean(0)
            emb_std  = embed_fn.weight.std(0) * 0.1
        torch.manual_seed(seed)
        soft_prefix = (emb_mean.unsqueeze(0).repeat(PREFIX_LEN, 1)
                       + torch.randn(PREFIX_LEN, EMB_DIM, device=EMB_DEV, dtype=DTYPE)
                       * emb_std.unsqueeze(0)).detach().float().requires_grad_(True)
        optimizer = torch.optim.Adam([soft_prefix], lr=LR_MAX)

        log(f"\n--- ST Soft Opt: {SOFT_STEPS} steps, seed={seed} ---")
        st_log = []
        best_st_ce = float('inf')
        best_soft_snapshot = soft_prefix.data.clone()

        t_soft_start = time.time()
        for step in range(SOFT_STEPS):
            lr = cosine_lr(step, SOFT_STEPS)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
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
                log(f"  [{step:4d}/{SOFT_STEPS}] ST-CE={ce_val:.5f}  best={best_st_ce:.5f}  "
                    f"lr={lr:.5f}  elapsed={time.time()-t_soft_start:.0f}s")

        t_soft = time.time() - t_soft_start
        log(f"Soft done: best={best_st_ce:.5f} in {t_soft:.0f}s")

        projected_ids  = project_to_tokens(best_soft_snapshot.to(DTYPE))
        projected_text = tokenizer.decode(projected_ids.cpu().tolist())
        proj_ce = compute_ce_discrete_batched(projected_ids, SUFFIXES, ref_completions)
        log(f"Projected: {projected_text!r}  CE={proj_ce:.5f}")

        log(f"\n--- HotFlip: {HOTFLIP_STEPS} steps, seed={seed} ---")
        current_ids = projected_ids.clone().to(EMB_DEV)
        current_ce  = proj_ce
        hotflip_log = [current_ce]

        t_hf_start = time.time()
        for step in range(HOTFLIP_STEPS):
            new_ids, new_ce = hotflip_step_batched(current_ids, ref_completions)
            improved = new_ce < current_ce
            if improved:
                current_ids = new_ids
                current_ce  = new_ce
            hotflip_log.append(current_ce)
            if step % 5 == 0 or step == HOTFLIP_STEPS - 1:
                toks = tokenizer.decode(current_ids.cpu().tolist())
                log(f"  [{step:3d}/{HOTFLIP_STEPS}] CE={current_ce:.5f}  {'↓' if improved else '–'}  {toks!r}")

            if step % 5 == 0:
                ckpt = {
                    "seed": seed, "hf_step": step, "best_ce": current_ce,
                    "best_ids": current_ids.tolist(),
                    "best_text": tokenizer.decode(current_ids.cpu().tolist()),
                    "overall_best_ce": overall_best_ce,
                }
                with open(CKPT_PATH, "w") as f:
                    json.dump(ckpt, f)

            if step % 10 == 9:
                notify("Exp30 HotFlip", f"seed={seed} step={step+1}/{HOTFLIP_STEPS}  CE={current_ce:.5f}")

        t_hf = time.time() - t_hf_start
        hotflip_ce = current_ce
        final_text = tokenizer.decode(current_ids.cpu().tolist())
        log(f"HotFlip done: CE={hotflip_ce:.5f} in {t_hf:.0f}s")

        is_new_best = hotflip_ce < overall_best_ce
        if is_new_best:
            overall_best_ce   = hotflip_ce
            overall_best_ids  = current_ids.clone()
            overall_best_seed = seed
            notify("Exp30 NEW BEST", f"seed={seed}  CE={hotflip_ce:.4f}  {final_text[:60]!r}")
        else:
            notify(f"Exp30 seed={seed} done", f"CE={hotflip_ce:.4f}  best_so_far={overall_best_ce:.4f}")

        all_runs.append({
            "seed": seed,
            "best_st_ce": best_st_ce,
            "projection_ce": proj_ce,
            "hotflip_ce": hotflip_ce,
            "projected_text": projected_text,
            "final_text": final_text,
            "final_ids": current_ids.tolist(),
            "st_log": st_log,
            "hotflip_log": hotflip_log,
            "timing": {"soft_seconds": t_soft, "hotflip_seconds": t_hf},
        })

    overall_best_text = tokenizer.decode(overall_best_ids.cpu().tolist()) if overall_best_ids is not None else ""

    log(f"\n=== FINAL SUMMARY ===")
    log(f"  Seeds tested:    {SEEDS}")
    log(f"  Best overall:    seed={overall_best_seed}, CE={overall_best_ce:.5f}")
    log(f"  Best prefix:     {overall_best_text!r}")
    log(f"  Exp25 seed=42:   CE=0.63171 (reference, same len=32)")
    log(f"  Exp26 best:      CE=0.60437 (seed=2, len=16)")
    for r in all_runs:
        log(f"  seed={r['seed']}: proj={r['projection_ce']:.5f}  hf={r['hotflip_ce']:.5f}  {r['final_text']!r}")

    results = {
        "experiment": "steer-001-exp30-multiseed-len32",
        "model": MODEL_NAME,
        "reference_prefix": REF_PREFIX,
        "seeds": SEEDS,
        "prefix_len": PREFIX_LEN,
        "n_soft_steps": SOFT_STEPS,
        "n_hotflip_steps": HOTFLIP_STEPS,
        "batch_size": BATCH_SIZE,
        "hf_topk": HF_TOPK,
        "lr_max": LR_MAX,
        "lr_min": LR_MIN,
        "method": "st_cosine_anneal_best_prefix_multiseed",
        "overall_best_seed": overall_best_seed,
        "overall_best_ce": overall_best_ce,
        "overall_best_text": overall_best_text,
        "overall_best_ids": overall_best_ids.tolist() if overall_best_ids is not None else [],
        "exp25_seed42_reference": 0.63171,
        "exp26_seed2_len16_reference": 0.60437,
        "runs": all_runs,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {OUT_PATH}")
    log("=== DONE ===")
    notify("Exp30 complete", f"best seed={overall_best_seed} CE={overall_best_ce:.4f}  ref Exp25=0.6317")

except Exception as e:
    notify("Exp30 FAILED", str(e))
    raise
