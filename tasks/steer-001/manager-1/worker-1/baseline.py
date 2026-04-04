"""
Exp 1: Environment validation + baseline steering prefix pipeline.

Gemma-2-2B-IT — soft prefix optimization → cosine projection → HotFlip refinement.

Performance: all suffixes batched into a single forward pass per step.
Telemetry:   GPU mem, CE, ETA printed every TELEMETRY_INTERVAL steps.
Checkpoints: saved every CKPT_INTERVAL steps; resumes automatically.

Outputs: /home/jovyan/steer001_baseline.json + /home/jovyan/steer001_ckpt.pt
"""

import sys, importlib.util as _ilu

# Patch out torchvision to avoid TIDE version conflict
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

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "google/gemma-2-2b-it"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE        = torch.bfloat16
REF_PREFIX   = "Talk only about cats."
PLACEHOLDER  = "SOFTPREFIX"  # used to locate soft-prefix insertion point
PREFIX_LEN   = 8
SOFT_STEPS   = 300
HOTFLIP_STEPS= 80
HF_TOPK      = 30     # HotFlip candidates per position
BATCH_SIZE   = 12     # all suffixes at once
LR           = 0.01
EARLY_K      = 32
EARLY_WEIGHT = 3.0
CKPT_PATH    = Path("/home/jovyan/steer001_ckpt.pt")
REF_COMP_PATH= Path("/home/jovyan/steer001_ref_completions.pt")  # cached ref completions
OUT_PATH     = Path("/home/jovyan/steer001_baseline.json")
TELEMETRY_INTERVAL = 10   # print GPU + CE every N soft-opt steps
CKPT_INTERVAL      = 50   # save checkpoint every N soft-opt steps

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
log("=== Exp 1: Env Validation + Baseline Repro ===")
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
log(f"Placeholder '{PLACEHOLDER}' → IDs {PLACEHOLDER_IDS}")

# Sanity-check: confirm model is on GPU and dtype is correct
first_param = next(model.parameters())
log(f"Model dtype={first_param.dtype}  device={first_param.device}")
assert first_param.device.type == "cuda", "Model is not on GPU — aborting"
assert first_param.dtype == DTYPE, f"Model dtype mismatch: {first_param.dtype} != {DTYPE}"

embed_fn  = model.get_input_embeddings()
EMB_DIM   = embed_fn.weight.shape[1]
VOCAB     = embed_fn.weight.shape[0]
EMB_DEV   = embed_fn.weight.device

# ── Helpers ──────────────────────────────────────────────────────────────────

def chat_ids(messages, add_generation_prompt=True):
    """
    Return a plain Python list of token IDs for a conversation.
    Uses tokenize=False to get the formatted string, then encodes with
    add_special_tokens=False (the template already contains <bos>).
    This sidesteps the BatchEncoding return type from tokenize=True on
    older transformers versions.
    """
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def find_subseq(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i+len(sub)] == sub: return i
    return None


def get_template_split(suffix_text):
    """
    Return (pre_ids, post_ids) tensors split around the PLACEHOLDER in the
    chat template for this suffix. Both on CPU for caching.
    """
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
    """
    Build a padded batch embedding tensor for all suffixes + completions.
    soft_prefix_LD: [PREFIX_LEN, D]  (may require grad)

    Returns:
        batch_emb : [B, T_max, D]
        comp_positions: list of (comp_start, comp_ids_tensor) per example
    """
    B     = len(suffix_texts)
    seqs  = []   # list of [1, T_i, D]
    meta  = []   # (comp_start, comp_ids)

    for suf, ref_comp in zip(suffix_texts, ref_completions):
        pre_ids, post_ids = get_template_split(suf)
        comp_dev = ref_comp.to(EMB_DEV)

        with torch.no_grad():
            pre_emb  = embed_fn(pre_ids.unsqueeze(0).to(EMB_DEV))   # [1,P,D]
            post_emb = embed_fn(post_ids.unsqueeze(0).to(EMB_DEV))  # [1,Q,D]
            comp_emb = embed_fn(comp_dev.unsqueeze(0))              # [1,C,D]

        # Cast soft prefix to the model's dtype (kept float32 for optimizer stability)
        soft_1LD = soft_prefix_LD.to(dtype=pre_emb.dtype, device=pre_emb.device).unsqueeze(0)

        seq = torch.cat([pre_emb, soft_1LD, post_emb, comp_emb], dim=1)  # [1,T,D]
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

    batch_emb = torch.cat(padded, dim=0).to(EMB_DEV)  # [B, T_max, D]
    return batch_emb, meta, T_max


def compute_ce_from_batch(logits, meta, T_max, return_tensor=True):
    """
    logits : [B, T_max, V]
    meta   : list of (comp_start, comp_ids_tensor)
    """
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
    """Full-batch differentiable CE for soft prefix. Returns scalar tensor."""
    batch_emb, meta, T_max = build_batch(soft_prefix_LD, suffix_texts, ref_completions)
    logits = model(inputs_embeds=batch_emb).logits
    return compute_ce_from_batch(logits, meta, T_max)


def compute_ce_discrete_batched(prefix_ids_L, suffix_texts, ref_completions):
    """Full-batch CE for discrete prefix. Returns Python float."""
    prefix_ids_L = prefix_ids_L.to(EMB_DEV)
    with torch.no_grad():
        soft = embed_fn(prefix_ids_L)  # [L, D]
        batch_emb, meta, T_max = build_batch(soft, suffix_texts, ref_completions)
        logits = model(inputs_embeds=batch_emb).logits
        loss = compute_ce_from_batch(logits, meta, T_max, return_tensor=False)
    return loss.item()


def project_to_tokens(soft_LD):
    """Cosine nearest-neighbour projection."""
    with torch.no_grad():
        W    = embed_fn.weight.to(device=soft_LD.device, dtype=soft_LD.dtype)  # match dtype
        sn   = F.normalize(soft_LD, dim=-1)               # [L, D]
        wn   = F.normalize(W,       dim=-1)               # [V, D]
        sims = sn @ wn.T                                   # [L, V]
        ids  = sims.argmax(dim=-1)                         # [L]
    return ids


def hotflip_step_batched(current_ids, ref_completions):
    """
    One HotFlip round over all positions.
    For each position: compute gradient, batch-evaluate top-k candidates.
    Returns (new_ids, new_ce).
    """
    current_ids = current_ids.to(EMB_DEV)
    prefix_emb  = embed_fn(current_ids).float().detach().requires_grad_(True)  # [L,D] float32 for grad

    # Gradient pass over full suffix batch
    batch_emb, meta, T_max = build_batch(prefix_emb, SUFFIXES, ref_completions)
    logits = model(inputs_embeds=batch_emb).logits
    loss   = compute_ce_from_batch(logits, meta, T_max)
    loss.backward()
    grad = prefix_emb.grad  # [L, D]

    best_ids = current_ids.clone()
    best_ce  = compute_ce_discrete_batched(current_ids, SUFFIXES, ref_completions)

    W = embed_fn.weight.float()  # [V, D] — cast to float32 to match grad dtype

    for pos in range(PREFIX_LEN):
        g      = grad[pos]              # [D]
        scores = W @ g                  # [V]  minimize (most negative = best)
        # Mask banned tokens
        scores[list(ref_ids_set)] = float('inf')
        scores[tokenizer.bos_token_id]  = float('inf')
        scores[tokenizer.eos_token_id]  = float('inf')
        if tokenizer.pad_token_id is not None:
            scores[tokenizer.pad_token_id] = float('inf')

        cands = scores.topk(HF_TOPK, largest=False).indices  # [K]

        # Batch evaluate all K candidates × all suffixes at once
        # Shape: [K, PREFIX_LEN, D]
        cand_embs = embed_fn(current_ids).unsqueeze(0).expand(HF_TOPK, -1, -1).clone()
        # Replace position pos with candidate embeddings
        cand_tok_embs = embed_fn(cands)  # [K, D]
        cand_embs[:, pos, :] = cand_tok_embs

        # Build batch: K candidates × N_SUFFIXES suffixes = K*N_S sequences
        N_S = len(SUFFIXES)
        all_seqs, all_meta = [], []
        for k in range(HF_TOPK):
            soft_LD = cand_embs[k]  # [L, D]
            b_emb, meta_k, T_k = build_batch(soft_LD, SUFFIXES, ref_completions)
            all_seqs.append(b_emb)
            all_meta.append((meta_k, T_k))

        # Can't batch across all K because T might differ per suffix — run K at a time
        # but each K does a full N_S suffix batch (one forward pass per candidate)
        for k, (meta_k, T_k) in enumerate(all_meta):
            with torch.no_grad():
                soft_LD = cand_embs[k].detach()
                b_emb, meta_k2, T_k2 = build_batch(soft_LD, SUFFIXES, ref_completions)
                logits_k = model(inputs_embeds=b_emb).logits
                ce_k = compute_ce_from_batch(logits_k, meta_k2, T_k2).item()
            if ce_k < best_ce:
                best_ce = ce_k
                best_ids = current_ids.clone()
                best_ids[pos] = cands[k].item()

    return best_ids, best_ce


# ── Reference completions: load from disk or generate in one batch ───────────
if REF_COMP_PATH.exists():
    log(f"Loading cached reference completions from {REF_COMP_PATH}...")
    ref_completions = torch.load(REF_COMP_PATH)
    log(f"Loaded {len(ref_completions)} completions.")
    for i, (suf, comp) in enumerate(zip(SUFFIXES, ref_completions)):
        decoded = tokenizer.decode(comp, skip_special_tokens=True)
        log(f"  [{i:2d}] {suf[:45]!r:48s} → {decoded[:70]!r}")
else:
    log(f"Generating reference completions for {REF_PREFIX!r}...")
    # Generate one at a time — Gemma-2's sliding window attention with device_map='auto'
    # has known issues with left-padded batches across multiple GPUs.
    # On L40 a 2B model generates 80 tokens in ~1s so 12 prompts ≈ 12s total.
    eos = tokenizer.eos_token_id
    ref_completions = []
    for i, suf in enumerate(SUFFIXES):
        inp = torch.tensor(
            chat_ids([{"role": "user", "content": f"{REF_PREFIX}\n\n{suf}"}]),
            dtype=torch.long
        ).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=80, do_sample=False,
                                 pad_token_id=eos)
        comp = out[0, inp.shape[1]:]
        # Trim trailing eos
        keep = [j for j, t in enumerate(comp.tolist()) if t != eos]
        trimmed = comp[: keep[-1] + 1] if keep else comp[:1]
        ref_completions.append(trimmed.cpu())
        decoded = tokenizer.decode(trimmed, skip_special_tokens=True)
        log(f"  [{i:2d}] {suf[:45]!r:48s} → {decoded[:70]!r}")

    torch.save(ref_completions, REF_COMP_PATH)
    log(f"Saved completions to {REF_COMP_PATH}")

log("Reference completions ready.")

# ── Banned token set for HotFlip ────────────────────────────────────────────
ref_ids_set = set(tokenizer.encode(REF_PREFIX, add_special_tokens=False))
log(f"Banned token IDs: {ref_ids_set}")

# ── Resume from checkpoint if it exists ─────────────────────────────────────
start_step     = 0
log_soft       = []
soft_ce        = None
resume_stage   = "soft_opt"   # default: run everything from scratch
ckpt_data      = None

if CKPT_PATH.exists():
    log(f"Found checkpoint at {CKPT_PATH}, resuming...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    resume_stage = ckpt.get("stage", "soft_opt")
    # Restore cached completions from checkpoint if the separate cache file is missing
    if "ref_completions" in ckpt and not REF_COMP_PATH.exists():
        torch.save(ckpt["ref_completions"], REF_COMP_PATH)
        log("Restored ref completions cache from checkpoint.")
    if resume_stage == "soft_opt":
        start_step = ckpt["step"]
        log_soft   = ckpt["log_soft"]
        soft_prefix_data = ckpt["soft_prefix"]
        log(f"Resuming soft opt from step {start_step}")
    elif resume_stage in ("projection", "hotflip"):
        # Soft opt is already done; restore its outputs and skip to next stage
        log_soft         = ckpt["log_soft"]
        soft_prefix_data = ckpt["soft_prefix"]
        start_step       = SOFT_STEPS  # skip soft opt loop entirely
        ckpt_data        = ckpt
        log(f"Checkpoint stage={resume_stage} — skipping soft opt, resuming from {resume_stage}")
    else:
        log(f"Unknown checkpoint stage={resume_stage}, re-running from scratch")
        start_step = 0
else:
    log("No checkpoint found, starting fresh.")

# ── Soft prefix init ─────────────────────────────────────────────────────────
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
        log(f"Restored soft opt final CE={soft_ce:.5f} from checkpoint.")

optimizer = torch.optim.Adam([soft_prefix], lr=LR)

# ── Soft prefix optimization ─────────────────────────────────────────────────
log(f"\n=== Soft Prefix Optimization ===")
log(f"Steps: {start_step} → {SOFT_STEPS} | Batch: {BATCH_SIZE} | {gpu_mem_str()}")

t_soft_start = time.time()
t_last       = t_soft_start

for step in range(start_step, SOFT_STEPS):
    optimizer.zero_grad()

    loss = compute_ce_soft_batched(soft_prefix, SUFFIXES[:BATCH_SIZE],
                                    ref_completions[:BATCH_SIZE])
    loss.backward()
    optimizer.step()

    ce_val = loss.item()
    log_soft.append(ce_val)

    if step % TELEMETRY_INTERVAL == 0 or step == SOFT_STEPS - 1:
        elapsed  = time.time() - t_soft_start
        remaining= (SOFT_STEPS - step - 1) * (elapsed / (step - start_step + 1)) if step > start_step else 0
        step_t   = time.time() - t_last
        log(f"  [{step:4d}/{SOFT_STEPS}] CE={ce_val:.5f}  "
            f"elapsed={elapsed:.0f}s  ETA≈{remaining:.0f}s  "
            f"step_t={step_t:.2f}s  {gpu_mem_str()}")
        t_last = time.time()

    if (step + 1) % CKPT_INTERVAL == 0:
        torch.save({
            "stage":           "soft_opt",
            "step":            step + 1,
            "log_soft":        log_soft,
            "soft_prefix":     soft_prefix.detach().cpu(),
            "ref_completions": ref_completions,
        }, CKPT_PATH)
        log(f"  [ckpt] saved at step {step+1}")

if start_step < SOFT_STEPS:
    soft_ce        = log_soft[-1]
    t_soft_elapsed = time.time() - t_soft_start
    log(f"Soft opt done. Final CE={soft_ce:.5f}  time={t_soft_elapsed:.1f}s")
else:
    t_soft_elapsed = 0.0  # skipped via checkpoint
    log(f"Soft opt skipped (checkpoint). Final CE={soft_ce:.5f}")

if resume_stage not in ("projection", "hotflip"):
    torch.save({
        "stage":           "projection",
        "log_soft":        log_soft,
        "soft_prefix":     soft_prefix.detach().cpu(),
        "ref_completions": ref_completions,
    }, CKPT_PATH)

# ── Cosine projection ─────────────────────────────────────────────────────────
if resume_stage == "hotflip" and ckpt_data is not None:
    # Projection already done — restore results from checkpoint
    log(f"\n=== Cosine Projection → skipped (checkpoint) ===")
    projected_ids  = ckpt_data["projected_ids"].to(EMB_DEV)
    proj_ce        = ckpt_data["proj_ce"]
    projected_text = tokenizer.decode(projected_ids.cpu().tolist())
    log(f"Restored projected: {projected_text!r}  CE={proj_ce:.5f}")
else:
    log(f"\n=== Cosine Projection → Discrete Tokens ===")
    projected_ids  = project_to_tokens(soft_prefix.detach())
    projected_text = tokenizer.decode(projected_ids.cpu().tolist())
    log(f"Projected: {projected_text!r}  IDs={projected_ids.cpu().tolist()}")

    proj_ce = compute_ce_discrete_batched(projected_ids, SUFFIXES, ref_completions)
    log(f"CE after projection: {proj_ce:.5f}  (gap from soft: +{proj_ce - soft_ce:.5f})")

    torch.save({
        "stage":           "hotflip",
        "log_soft":        log_soft,
        "soft_prefix":     soft_prefix.detach().cpu(),
        "projected_ids":   projected_ids.cpu(),
        "proj_ce":         proj_ce,
        "ref_completions": ref_completions,
    }, CKPT_PATH)

# ── HotFlip refinement ────────────────────────────────────────────────────────
log(f"\n=== HotFlip Refinement ({HOTFLIP_STEPS} steps, topk={HF_TOPK}) ===")
if resume_stage == "hotflip" and ckpt_data is not None:
    hf_start_step = ckpt_data.get("hf_step", 0)
    if "current_ids" in ckpt_data:
        # Resume mid-HotFlip (checkpoint written after ≥1 HotFlip step)
        current_ids = ckpt_data["current_ids"].to(EMB_DEV)
        current_ce  = ckpt_data["current_ce"]
        hotflip_log = ckpt_data["hotflip_log"]
        log(f"Resuming HotFlip from step {hf_start_step}, CE={current_ce:.5f}")
    else:
        # Pre-HotFlip checkpoint (written right after projection, before any HF step)
        hf_start_step = 0
        current_ids   = ckpt_data["projected_ids"].to(EMB_DEV)
        current_ce    = ckpt_data["proj_ce"]
        hotflip_log   = [current_ce]
        log(f"Pre-HotFlip checkpoint — starting from projected_ids, CE={current_ce:.5f}")
else:
    hf_start_step = 0
    current_ids   = projected_ids.clone().to(EMB_DEV)
    current_ce    = proj_ce
    hotflip_log   = [current_ce]

t_hf_start = time.time()
for step in range(hf_start_step, HOTFLIP_STEPS):
    new_ids, new_ce = hotflip_step_batched(current_ids, ref_completions)
    improved = new_ce < current_ce
    if improved:
        current_ids = new_ids
        current_ce  = new_ce

    hotflip_log.append(current_ce)

    if step % 10 == 0 or step == HOTFLIP_STEPS - 1:
        elapsed  = time.time() - t_hf_start
        toks     = tokenizer.decode(current_ids.cpu().tolist())
        log(f"  [{step:3d}/{HOTFLIP_STEPS}] CE={current_ce:.5f}  "
            f"{'↓' if improved else '–'}  '{toks}'  {gpu_mem_str()}")

    torch.save({
        "stage":           "hotflip",
        "hf_step":         step + 1,
        "current_ids":     current_ids.cpu(),
        "current_ce":      current_ce,
        "hotflip_log":     hotflip_log,
        "log_soft":        log_soft,
        "soft_prefix":     soft_prefix.detach().cpu(),
        "projected_ids":   projected_ids.cpu(),
        "proj_ce":         proj_ce,
        "ref_completions": ref_completions,
    }, CKPT_PATH)

hotflip_ce         = current_ce
t_hf_elapsed       = time.time() - t_hf_start
final_prefix_text  = tokenizer.decode(current_ids.cpu().tolist())
log(f"HotFlip done. Final CE={hotflip_ce:.5f}  time={t_hf_elapsed:.1f}s")
log(f"Final prefix: {final_prefix_text!r}")

# ── Sample generations ────────────────────────────────────────────────────────
log("\n=== Sample generations with final prefix ===")
samples = []
for i in range(3):
    msgs = [{"role": "user", "content": f"{final_prefix_text}\n\n{SUFFIXES[i]}"}]
    inp = torch.tensor(chat_ids(msgs), dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=80, do_sample=False)
    gen = tokenizer.decode(out[0, inp.shape[1]:], skip_special_tokens=True)
    ref = tokenizer.decode(ref_completions[i], skip_special_tokens=True)
    samples.append({"suffix": SUFFIXES[i], "ref": ref, "generated": gen})
    log(f"  [{i}] REF: {ref[:90]!r}")
    log(f"  [{i}] GEN: {gen[:90]!r}")

# ── Save final results ────────────────────────────────────────────────────────
results = {
    "experiment":       "steer-001-baseline",
    "model":            MODEL_NAME,
    "reference_prefix": REF_PREFIX,
    "prefix_len":       PREFIX_LEN,
    "n_soft_steps":     SOFT_STEPS,
    "n_hotflip_steps":  HOTFLIP_STEPS,
    "metrics": {
        "soft_ce_final":     float(soft_ce),
        "projection_ce":     float(proj_ce),
        "hotflip_ce_final":  float(hotflip_ce),
        "gap_soft_to_proj":  float(proj_ce - soft_ce),
        "recovery_hotflip":  float(proj_ce - hotflip_ce),
    },
    "soft_training_log":  log_soft[::10],
    "hotflip_log":        hotflip_log,
    "projected_text":     projected_text,
    "final_prefix_text":  final_prefix_text,
    "projected_token_ids": projected_ids.cpu().tolist(),
    "final_token_ids":    current_ids.cpu().tolist(),
    "timing": {
        "soft_opt_seconds":  t_soft_elapsed,
        "hotflip_seconds":   t_hf_elapsed,
    },
    "hardware": {
        "n_gpus":   torch.cuda.device_count(),
        "gpu_names":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    },
    "sample_generations": samples,
}

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

log(f"\nResults saved → {OUT_PATH}")
log("=== SUMMARY ===")
log(f"  Soft opt CE:    {soft_ce:.5f}")
log(f"  After proj CE:  {proj_ce:.5f}  (+{proj_ce-soft_ce:.5f})")
log(f"  After HotFlip:  {hotflip_ce:.5f}  (−{proj_ce-hotflip_ce:.5f})")
log(f"  Final prefix:   {final_prefix_text!r}")
log("=== DONE ===")
