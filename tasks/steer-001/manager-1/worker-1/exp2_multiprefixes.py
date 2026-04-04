"""
Exp 2: Multi-prefix generalization.

Research question: Is the continuous/discrete CE gap (and HotFlip recovery)
consistent across reference behaviors, or is it prefix-specific?

Reference prefixes tested:
  0: "Talk only about cats."
  1: "Always respond using only numbered lists."
  2: "Respond only in formal academic language."
  3: "You are a pirate. Always speak like one."

Full pipeline per prefix: 200 soft opt steps → cosine projection → 40 HotFlip steps.
Sequential execution; separate checkpoints per prefix.

Outputs: /home/jovyan/steer001_multiprefixes.json
"""

import sys, importlib.util as _ilu

_real_find_spec = _ilu.find_spec
def _patched(name, package=None, target=None):
    return None if name == "torchvision" else _real_find_spec(name, package)
_ilu.find_spec = _patched
for _k in list(sys.modules):
    if "torchvision" in _k: del sys.modules[_k]

import json, os, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME  = "google/gemma-2-2b-it"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.bfloat16
PLACEHOLDER = "SOFTPREFIX"
PREFIX_LEN  = 8
SOFT_STEPS  = 200
HOTFLIP_STEPS = 40
HF_TOPK     = 30
LR          = 0.01
EARLY_K     = 32
EARLY_WEIGHT= 3.0
TELEMETRY_INTERVAL = 10
CKPT_INTERVAL      = 30

REFERENCE_PREFIXES = [
    "Talk only about cats.",
    "Always respond using only numbered lists.",
    "Respond only in formal academic language.",
    "You are a pirate. Always speak like one.",
]

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

OUT_PATH = Path("/home/jovyan/steer001_multiprefixes.json")

# ── Telemetry ────────────────────────────────────────────────────────────────
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

# ── Startup ──────────────────────────────────────────────────────────────────
log("=== Exp 2: Multi-Prefix Generalization ===")
log(f"CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    log(f"  GPU {i}: {p.name} {p.total_memory//1024**3}GB")

log(f"Loading {MODEL_NAME}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map="cuda:0")
model.eval()
log(f"Model loaded in {time.time()-t0:.1f}s | {gpu_mem_str()}")

PLACEHOLDER_IDS = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)
embed_fn = model.get_input_embeddings()
EMB_DIM  = embed_fn.weight.shape[1]
VOCAB    = embed_fn.weight.shape[0]
EMB_DEV  = embed_fn.weight.device

# ── Helpers ───────────────────────────────────────────────────────────────────
def chat_ids(messages, add_generation_prompt=True):
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=False)
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
    T_max  = max(s.shape[1] for s in seqs)
    padded = []
    for seq in seqs:
        pad_len = T_max - seq.shape[1]
        if pad_len:
            pad = torch.zeros(1, pad_len, EMB_DIM, device=seq.device, dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=1)
        padded.append(seq)
    return torch.cat(padded, dim=0).to(EMB_DEV), meta, T_max

def compute_ce_from_batch(logits, meta, T_max):
    total  = torch.tensor(0.0, device=logits.device)
    weight = 0.0
    for b, (comp_start, comp_ids) in enumerate(meta):
        for i, tok in enumerate(comp_ids):
            pos = comp_start + i - 1
            if pos >= T_max: continue
            w    = EARLY_WEIGHT if i < EARLY_K else 1.0
            ce   = F.cross_entropy(logits[b, pos].unsqueeze(0), tok.unsqueeze(0).long())
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
        loss   = compute_ce_from_batch(logits, meta, T_max)
    return loss.item()

def project_to_tokens(soft_LD):
    with torch.no_grad():
        W    = embed_fn.weight.to(device=soft_LD.device, dtype=soft_LD.dtype)
        sn   = F.normalize(soft_LD, dim=-1)
        wn   = F.normalize(W,       dim=-1)
        sims = sn @ wn.T
        ids  = sims.argmax(dim=-1)
    return ids

def hotflip_step_batched(current_ids, ref_completions, ref_ids_set):
    current_ids = current_ids.to(EMB_DEV)
    prefix_emb  = embed_fn(current_ids).float().detach().requires_grad_(True)
    batch_emb, meta, T_max = build_batch(prefix_emb, SUFFIXES, ref_completions)
    logits = model(inputs_embeds=batch_emb).logits
    loss   = compute_ce_from_batch(logits, meta, T_max)
    loss.backward()
    grad = prefix_emb.grad  # [L, D] float32

    best_ids = current_ids.clone()
    best_ce  = compute_ce_discrete_batched(current_ids, SUFFIXES, ref_completions)
    W = embed_fn.weight.float()

    for pos in range(PREFIX_LEN):
        g      = grad[pos]
        scores = W @ g
        scores[list(ref_ids_set)]         = float('inf')
        scores[tokenizer.bos_token_id]    = float('inf')
        scores[tokenizer.eos_token_id]    = float('inf')
        if tokenizer.pad_token_id is not None:
            scores[tokenizer.pad_token_id] = float('inf')

        cands     = scores.topk(HF_TOPK, largest=False).indices
        cand_embs = embed_fn(current_ids).unsqueeze(0).expand(HF_TOPK, -1, -1).clone()
        cand_embs[:, pos, :] = embed_fn(cands)

        for k in range(HF_TOPK):
            with torch.no_grad():
                soft_LD  = cand_embs[k].detach()
                b_emb, meta_k, T_k = build_batch(soft_LD, SUFFIXES, ref_completions)
                logits_k = model(inputs_embeds=b_emb).logits
                ce_k     = compute_ce_from_batch(logits_k, meta_k, T_k).item()
            if ce_k < best_ce:
                best_ce = ce_k
                best_ids = current_ids.clone()
                best_ids[pos] = cands[k].item()

    return best_ids, best_ce


# ── Per-prefix pipeline ───────────────────────────────────────────────────────
all_results = []

for pidx, ref_prefix in enumerate(REFERENCE_PREFIXES):
    log(f"\n{'='*60}")
    log(f"PREFIX {pidx}: {ref_prefix!r}")
    log(f"{'='*60}")

    ckpt_path     = Path(f"/home/jovyan/steer001_mp_ckpt_{pidx}.pt")
    ref_comp_path = Path(f"/home/jovyan/steer001_mp_refcomp_{pidx}.pt")

    # ── Reference completions ────────────────────────────────────────────────
    if ref_comp_path.exists():
        log(f"Loading cached reference completions for prefix {pidx}...")
        ref_completions = torch.load(ref_comp_path)
        log(f"Loaded {len(ref_completions)} completions.")
    else:
        log(f"Generating reference completions for {ref_prefix!r}...")
        eos = tokenizer.eos_token_id
        ref_completions = []
        for i, suf in enumerate(SUFFIXES):
            inp = torch.tensor(
                chat_ids([{"role": "user", "content": f"{ref_prefix}\n\n{suf}"}]),
                dtype=torch.long
            ).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model.generate(inp, max_new_tokens=80, do_sample=False, pad_token_id=eos)
            comp = out[0, inp.shape[1]:]
            keep = [j for j, t in enumerate(comp.tolist()) if t != eos]
            trimmed = comp[: keep[-1] + 1] if keep else comp[:1]
            ref_completions.append(trimmed.cpu())
            decoded = tokenizer.decode(trimmed, skip_special_tokens=True)
            log(f"  [{i:2d}] {suf[:45]!r:48s} → {decoded[:60]!r}")
        torch.save(ref_completions, ref_comp_path)
        log(f"Saved to {ref_comp_path}")

    ref_ids_set = set(tokenizer.encode(ref_prefix, add_special_tokens=False))
    log(f"Banned IDs: {ref_ids_set}")

    # ── Checkpoint resume ────────────────────────────────────────────────────
    start_step   = 0
    log_soft     = []
    soft_ce      = None
    resume_stage = "soft_opt"
    ckpt_data    = None

    if ckpt_path.exists():
        log(f"Found checkpoint {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        resume_stage = ckpt.get("stage", "soft_opt")
        if resume_stage == "soft_opt":
            start_step = ckpt["step"]
            log_soft   = ckpt["log_soft"]
            soft_prefix_data = ckpt["soft_prefix"]
            log(f"Resuming soft opt from step {start_step}")
        elif resume_stage in ("projection", "hotflip"):
            log_soft         = ckpt["log_soft"]
            soft_prefix_data = ckpt["soft_prefix"]
            start_step       = SOFT_STEPS
            ckpt_data        = ckpt
            log(f"Stage={resume_stage}, skipping soft opt")
    else:
        log("No checkpoint, starting fresh.")

    # ── Soft prefix init ─────────────────────────────────────────────────────
    if start_step == 0:
        with torch.no_grad():
            emb_mean = embed_fn.weight.mean(0)
            emb_std  = embed_fn.weight.std(0) * 0.1
        soft_prefix = (emb_mean.unsqueeze(0).repeat(PREFIX_LEN, 1)
                       + torch.randn(PREFIX_LEN, EMB_DIM, device=DEVICE, dtype=DTYPE)
                       * emb_std.unsqueeze(0)).detach().float().requires_grad_(True)
    else:
        soft_prefix = soft_prefix_data.to(DEVICE).float().requires_grad_(True)
        if start_step >= SOFT_STEPS:
            soft_ce = log_soft[-1]
            log(f"Restored soft opt CE={soft_ce:.5f}")

    optimizer = torch.optim.Adam([soft_prefix], lr=LR)

    # ── Soft opt ─────────────────────────────────────────────────────────────
    log(f"\n=== Soft Opt: {start_step}→{SOFT_STEPS} ===")
    t_soft_start = time.time()
    t_last       = t_soft_start

    for step in range(start_step, SOFT_STEPS):
        optimizer.zero_grad()
        loss = compute_ce_soft_batched(soft_prefix, SUFFIXES, ref_completions)
        loss.backward()
        optimizer.step()
        ce_val = loss.item()
        log_soft.append(ce_val)

        if step % TELEMETRY_INTERVAL == 0 or step == SOFT_STEPS - 1:
            elapsed   = time.time() - t_soft_start
            remaining = (SOFT_STEPS - step - 1) * (elapsed / (step - start_step + 1)) if step > start_step else 0
            log(f"  [{step:4d}/{SOFT_STEPS}] CE={ce_val:.5f}  ETA≈{remaining:.0f}s  {gpu_mem_str()}")
            t_last = time.time()

        if (step + 1) % CKPT_INTERVAL == 0:
            torch.save({"stage": "soft_opt", "step": step+1,
                        "log_soft": log_soft, "soft_prefix": soft_prefix.detach().cpu()}, ckpt_path)

    if start_step < SOFT_STEPS:
        soft_ce = log_soft[-1]
        t_soft_elapsed = time.time() - t_soft_start
        log(f"Soft opt done. CE={soft_ce:.5f}  time={t_soft_elapsed:.1f}s")
    else:
        t_soft_elapsed = 0.0
        log(f"Soft opt skipped. CE={soft_ce:.5f}")

    # Save projection-stage checkpoint
    if resume_stage not in ("projection", "hotflip"):
        torch.save({"stage": "projection", "log_soft": log_soft,
                    "soft_prefix": soft_prefix.detach().cpu()}, ckpt_path)

    # ── Projection ───────────────────────────────────────────────────────────
    if resume_stage == "hotflip" and ckpt_data is not None:
        log(f"\n=== Projection: skipped (checkpoint) ===")
        projected_ids  = ckpt_data["projected_ids"].to(EMB_DEV)
        proj_ce        = ckpt_data["proj_ce"]
        projected_text = tokenizer.decode(projected_ids.cpu().tolist())
        log(f"Restored: {projected_text!r}  CE={proj_ce:.5f}")
    else:
        log(f"\n=== Cosine Projection ===")
        projected_ids  = project_to_tokens(soft_prefix.detach())
        projected_text = tokenizer.decode(projected_ids.cpu().tolist())
        log(f"Projected: {projected_text!r}")
        proj_ce = compute_ce_discrete_batched(projected_ids, SUFFIXES, ref_completions)
        log(f"CE after projection: {proj_ce:.5f}  (gap +{proj_ce - soft_ce:.5f})")
        torch.save({"stage": "hotflip", "log_soft": log_soft,
                    "soft_prefix": soft_prefix.detach().cpu(),
                    "projected_ids": projected_ids.cpu(), "proj_ce": proj_ce}, ckpt_path)

    # ── HotFlip ──────────────────────────────────────────────────────────────
    log(f"\n=== HotFlip: {HOTFLIP_STEPS} steps ===")
    if resume_stage == "hotflip" and ckpt_data is not None and "current_ids" in ckpt_data:
        hf_start_step = ckpt_data.get("hf_step", 0)
        current_ids   = ckpt_data["current_ids"].to(EMB_DEV)
        current_ce    = ckpt_data["current_ce"]
        hotflip_log   = ckpt_data["hotflip_log"]
        log(f"Resuming from HotFlip step {hf_start_step}, CE={current_ce:.5f}")
    else:
        hf_start_step = 0
        current_ids   = projected_ids.clone().to(EMB_DEV)
        current_ce    = proj_ce
        hotflip_log   = [current_ce]

    t_hf_start = time.time()
    for step in range(hf_start_step, HOTFLIP_STEPS):
        new_ids, new_ce = hotflip_step_batched(current_ids, ref_completions, ref_ids_set)
        improved = new_ce < current_ce
        if improved:
            current_ids = new_ids
            current_ce  = new_ce
        hotflip_log.append(current_ce)
        if step % 10 == 0 or step == HOTFLIP_STEPS - 1:
            toks = tokenizer.decode(current_ids.cpu().tolist())
            log(f"  [{step:3d}/{HOTFLIP_STEPS}] CE={current_ce:.5f}  {'↓' if improved else '–'}  {toks!r}")
        torch.save({"stage": "hotflip", "hf_step": step+1,
                    "current_ids": current_ids.cpu(), "current_ce": current_ce,
                    "hotflip_log": hotflip_log, "log_soft": log_soft,
                    "soft_prefix": soft_prefix.detach().cpu(),
                    "projected_ids": projected_ids.cpu(), "proj_ce": proj_ce}, ckpt_path)

    hotflip_ce        = current_ce
    t_hf_elapsed      = time.time() - t_hf_start
    final_prefix_text = tokenizer.decode(current_ids.cpu().tolist())
    log(f"HotFlip done. CE={hotflip_ce:.5f}  time={t_hf_elapsed:.1f}s")
    log(f"Final prefix: {final_prefix_text!r}")

    # ── Sample generations ───────────────────────────────────────────────────
    samples = []
    for i in range(3):
        msgs = [{"role": "user", "content": f"{final_prefix_text}\n\n{SUFFIXES[i]}"}]
        inp  = torch.tensor(chat_ids(msgs), dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=60, do_sample=False)
        gen = tokenizer.decode(out[0, inp.shape[1]:], skip_special_tokens=True)
        ref = tokenizer.decode(ref_completions[i], skip_special_tokens=True)
        log(f"  [{i}] REF: {ref[:80]!r}")
        log(f"  [{i}] GEN: {gen[:80]!r}")
        samples.append({"suffix": SUFFIXES[i], "ref": ref, "generated": gen})

    all_results.append({
        "prefix_idx":          pidx,
        "reference_prefix":    ref_prefix,
        "metrics": {
            "soft_ce_final":    float(soft_ce),
            "projection_ce":    float(proj_ce),
            "hotflip_ce_final": float(hotflip_ce),
            "gap_soft_to_proj": float(proj_ce - soft_ce),
            "recovery_hotflip": float(proj_ce - hotflip_ce),
        },
        "soft_training_log":  log_soft[::10],
        "hotflip_log":        hotflip_log,
        "projected_text":     projected_text,
        "final_prefix_text":  final_prefix_text,
        "final_token_ids":    current_ids.cpu().tolist(),
        "sample_generations": samples,
    })

    # Intermediate save after each prefix
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"Intermediate results saved ({pidx+1}/{len(REFERENCE_PREFIXES)} prefixes done)")

# ── Final output ──────────────────────────────────────────────────────────────
with open(OUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
log(f"\nAll results saved → {OUT_PATH}")

log("\n=== SUMMARY ===")
for r in all_results:
    m = r["metrics"]
    log(f"  [{r['prefix_idx']}] {r['reference_prefix']!r}")
    log(f"       soft={m['soft_ce_final']:.4f}  proj={m['projection_ce']:.4f}  "
        f"gap=+{m['gap_soft_to_proj']:.4f}  hf={m['hotflip_ce_final']:.4f}  "
        f"recovery=-{m['recovery_hotflip']:.4f}")
    log(f"       final: {r['final_prefix_text']!r}")
log("=== DONE ===")
