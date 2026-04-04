"""
Exp 3: Layer-depth projection ablation.

Research question: Does optimizing the soft prefix to match the reference
prefix's representation at an earlier (shallower) layer lead to better
post-projection and post-HotFlip CE?

Design:
  - Reference prefix: "Talk only about cats."
  - For each layer L ∈ {4, 10, 16, 22}:
      1. Soft opt loss = mean cosine distance between the soft prefix's
         layer-L hidden states and the reference prefix's layer-L hidden
         states (in the same suffix context). This is an unsupervised
         representation-matching objective instead of output CE.
      2. After soft opt: measure output CE (how well the prefix works),
         then project to discrete tokens (standard cosine in emb space).
      3. HotFlip: 40 steps (standard CE-based).
  - Hypothesis: shallower L → representations stay closer to the embedding
    space → smaller projection gap → better post-HotFlip CE.

Naturalness measured as fraction of ASCII lowercase+space chars in prefix.

Outputs: /home/jovyan/steer001_layerablation.json
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
MODEL_NAME    = "google/gemma-2-2b-it"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.bfloat16
REF_PREFIX    = "Talk only about cats."
PLACEHOLDER   = "SOFTPREFIX"
PREFIX_LEN    = 8
SOFT_STEPS    = 200
HOTFLIP_STEPS = 40
HF_TOPK       = 30
LR            = 0.01
EARLY_K       = 32
EARLY_WEIGHT  = 3.0
TELEMETRY_INTERVAL = 10
CKPT_INTERVAL      = 30

TARGET_LAYERS = [4, 10, 16, 22]  # Gemma-2-2B has 26 transformer layers

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

REF_COMP_PATH = Path("/home/jovyan/steer001_ref_completions.pt")   # from Exp 1
OUT_PATH      = Path("/home/jovyan/steer001_layerablation.json")

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
log("=== Exp 3: Layer-Depth Projection Ablation ===")
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

N_LAYERS = model.config.num_hidden_layers
log(f"Model has {N_LAYERS} transformer layers. Target layers: {TARGET_LAYERS}")

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
    grad = prefix_emb.grad

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


def ascii_naturalness(token_ids):
    """Fraction of chars in the decoded prefix that are ASCII letters or spaces."""
    text = tokenizer.decode(token_ids.cpu().tolist())
    if not text: return 0.0
    ascii_chars = sum(1 for c in text if c.isascii() and (c.isalpha() or c == ' '))
    return ascii_chars / len(text)


# ── Reference completions (from Exp 1, all use "Talk only about cats.") ──────
if REF_COMP_PATH.exists():
    log(f"Loading cached ref completions from {REF_COMP_PATH}...")
    ref_completions = torch.load(REF_COMP_PATH)
    log(f"Loaded {len(ref_completions)} completions.")
else:
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
    torch.save(ref_completions, REF_COMP_PATH)
    log(f"Saved to {REF_COMP_PATH}")

ref_ids_set = set(tokenizer.encode(REF_PREFIX, add_special_tokens=False))


# ── Precompute reference prefix layer-L hidden states (mean-pooled per suffix) ──
def precompute_ref_layer_hiddens(layer_idx):
    """
    For each suffix, compute the mean-pooled hidden state at layer_idx
    when the reference prefix tokens (not soft prefix) are in the template.
    Returns list of [D] tensors (one per suffix), on CPU.
    """
    ref_token_ids = tokenizer.encode(REF_PREFIX, add_special_tokens=False)
    log(f"  Precomputing layer-{layer_idx} ref hiddens "
        f"(ref_prefix={len(ref_token_ids)} tokens)...")
    hiddens = []
    for suf in SUFFIXES:
        pre_ids, post_ids = get_template_split(suf)
        # Build [pre, ref_prefix_tokens, post] — ref takes the PLACEHOLDER's position
        ref_tensor = torch.tensor(ref_token_ids, dtype=torch.long).to(DEVICE)
        full_ids   = torch.cat([pre_ids.to(DEVICE), ref_tensor, post_ids.to(DEVICE)]).unsqueeze(0)
        with torch.no_grad():
            out = model(full_ids, output_hidden_states=True)
        # hidden_states[0] = embedding output; [layer_idx] = after layer (layer_idx-1)
        hidden = out.hidden_states[layer_idx]  # [1, S, D_model]
        p_start = pre_ids.shape[0]
        p_end   = p_start + len(ref_token_ids)
        ref_h   = hidden[0, p_start:p_end, :].mean(0)  # [D_model]
        hiddens.append(ref_h.cpu())
    return hiddens


def compute_loss_layer(soft_prefix_LD, layer_idx, ref_hiddens):
    """
    Soft opt loss: mean cosine distance between soft prefix's layer-L
    mean-pooled hidden state and reference hidden state, across all suffixes.
    """
    total = torch.tensor(0.0, device=DEVICE)
    for suf, ref_h in zip(SUFFIXES, ref_hiddens):
        pre_ids, post_ids = get_template_split(suf)
        pre_emb  = embed_fn(pre_ids.unsqueeze(0).to(EMB_DEV))
        post_emb = embed_fn(post_ids.unsqueeze(0).to(EMB_DEV))
        # Only need prefix + context (no completion needed for hidden state matching)
        soft_1LD = soft_prefix_LD.to(dtype=pre_emb.dtype, device=EMB_DEV).unsqueeze(0)
        seq = torch.cat([pre_emb, soft_1LD, post_emb], dim=1)
        out = model(inputs_embeds=seq, output_hidden_states=True)
        hidden = out.hidden_states[layer_idx]  # [1, S, D_model]
        p_start      = pre_emb.shape[1]
        p_end        = p_start + PREFIX_LEN
        soft_h       = hidden[0, p_start:p_end, :].mean(0)  # [D_model]
        ref_h_dev    = ref_h.to(device=DEVICE, dtype=soft_h.dtype)
        cos_dist     = 1 - F.cosine_similarity(soft_h.unsqueeze(0), ref_h_dev.unsqueeze(0))
        total        = total + cos_dist.squeeze()
    return total / len(SUFFIXES)


# ── Per-layer pipeline ────────────────────────────────────────────────────────
all_results = []

for layer_idx in TARGET_LAYERS:
    log(f"\n{'='*60}")
    log(f"LAYER {layer_idx}")
    log(f"{'='*60}")

    ckpt_path = Path(f"/home/jovyan/steer001_layer{layer_idx}_ckpt.pt")

    # Precompute reference hidden states for this layer
    ref_hiddens = precompute_ref_layer_hiddens(layer_idx)
    log(f"  Reference hiddens precomputed. {gpu_mem_str()}")

    # ── Checkpoint resume ────────────────────────────────────────────────────
    start_step   = 0
    log_soft     = []  # tracks cosine-dist loss here, not CE
    soft_ce_final = None
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
            soft_ce_final    = ckpt.get("soft_ce_final")
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

    optimizer = torch.optim.Adam([soft_prefix], lr=LR)

    # ── Soft opt (layer-L cosine-distance loss) ───────────────────────────────
    log(f"\n=== Soft Opt (layer-{layer_idx} cosine distance): {start_step}→{SOFT_STEPS} ===")
    t_soft_start = time.time()

    for step in range(start_step, SOFT_STEPS):
        optimizer.zero_grad()
        loss = compute_loss_layer(soft_prefix, layer_idx, ref_hiddens)
        loss.backward()
        optimizer.step()
        log_soft.append(loss.item())

        if step % TELEMETRY_INTERVAL == 0 or step == SOFT_STEPS - 1:
            elapsed   = time.time() - t_soft_start
            remaining = (SOFT_STEPS - step - 1) * (elapsed / (step - start_step + 1)) if step > start_step else 0
            log(f"  [{step:4d}/{SOFT_STEPS}] cos_dist={loss.item():.5f}  ETA≈{remaining:.0f}s  {gpu_mem_str()}")

        if (step + 1) % CKPT_INTERVAL == 0:
            torch.save({"stage": "soft_opt", "step": step+1,
                        "log_soft": log_soft, "soft_prefix": soft_prefix.detach().cpu()}, ckpt_path)

    if start_step < SOFT_STEPS:
        t_soft_elapsed = time.time() - t_soft_start
        final_cos_dist = log_soft[-1]
        # Measure CE after soft opt (this is what projection/HotFlip optimize)
        soft_ce_final = compute_ce_discrete_batched(
            project_to_tokens(soft_prefix.detach()), SUFFIXES, ref_completions)
        log(f"Soft opt done. cos_dist={final_cos_dist:.5f}  "
            f"embed_ce={soft_ce_final:.5f}  time={t_soft_elapsed:.1f}s")
    else:
        t_soft_elapsed = 0.0
        final_cos_dist = log_soft[-1] if log_soft else float('nan')
        log(f"Soft opt skipped (checkpoint).")

    if resume_stage not in ("projection", "hotflip"):
        torch.save({"stage": "projection", "log_soft": log_soft,
                    "soft_prefix": soft_prefix.detach().cpu(),
                    "soft_ce_final": soft_ce_final}, ckpt_path)

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
        log(f"CE after projection: {proj_ce:.5f}")
        torch.save({"stage": "hotflip", "log_soft": log_soft,
                    "soft_prefix": soft_prefix.detach().cpu(),
                    "soft_ce_final": soft_ce_final,
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
                    "soft_ce_final": soft_ce_final,
                    "soft_prefix": soft_prefix.detach().cpu(),
                    "projected_ids": projected_ids.cpu(), "proj_ce": proj_ce}, ckpt_path)

    hotflip_ce        = current_ce
    t_hf_elapsed      = time.time() - t_hf_start
    final_prefix_text = tokenizer.decode(current_ids.cpu().tolist())
    nat_score         = ascii_naturalness(current_ids)
    log(f"HotFlip done. CE={hotflip_ce:.5f}  naturalness={nat_score:.3f}  time={t_hf_elapsed:.1f}s")
    log(f"Final prefix: {final_prefix_text!r}")

    all_results.append({
        "layer_idx":          layer_idx,
        "reference_prefix":   REF_PREFIX,
        "soft_opt_loss":      "cosine_distance_layer_" + str(layer_idx),
        "metrics": {
            "final_cos_dist":   float(log_soft[-1]) if log_soft else None,
            "soft_emb_ce":      float(soft_ce_final) if soft_ce_final is not None else None,
            "projection_ce":    float(proj_ce),
            "hotflip_ce_final": float(hotflip_ce),
            "recovery_hotflip": float(proj_ce - hotflip_ce),
            "naturalness":      float(nat_score),
        },
        "soft_training_log":  log_soft[::10],
        "hotflip_log":        hotflip_log,
        "projected_text":     projected_text,
        "final_prefix_text":  final_prefix_text,
        "final_token_ids":    current_ids.cpu().tolist(),
    })

    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"Intermediate results saved (layer {layer_idx} done)")

# ── Final output ──────────────────────────────────────────────────────────────
with open(OUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
log(f"\nAll results saved → {OUT_PATH}")

log("\n=== SUMMARY ===")
log(f"{'Layer':>6}  {'proj_ce':>8}  {'hf_ce':>8}  {'recovery':>9}  {'natural':>8}  prefix")
for r in all_results:
    m = r["metrics"]
    log(f"  L{r['layer_idx']:>2}   {m['projection_ce']:8.4f}  {m['hotflip_ce_final']:8.4f}  "
        f"{m['recovery_hotflip']:9.4f}  {m['naturalness']:8.3f}  {r['final_prefix_text']!r}")
log("=== DONE ===")
