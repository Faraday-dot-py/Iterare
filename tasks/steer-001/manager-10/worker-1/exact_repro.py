"""
Experiment 10: Exact Exp1 Reproduction on GPU 1

Exp1 achieved HotFlip CE=0.740 using BATCH_SIZE=12, HF_TOPK=30, PLACEHOLDER="SOFTPREFIX".
Exp9 used BATCH_SIZE=8, TOPK=20, PLACEHOLDER="PREFIX_PLACEHOLDER" and got CE=1.277.

This experiment runs Exp1's exact configuration on GPU 1 (44GB free) to confirm
whether the hyperparameter differences explain the gap.

Expected outcome: soft≈0.191, proj≈1.436, HotFlip≈0.740

Output: /home/jovyan/steer001_exp10_exact.json
"""

import sys, importlib.util as _ilu

_real_find_spec = _ilu.find_spec
def _patched(name, package=None, target=None):
    return None if name == "torchvision" else _real_find_spec(name, package)
_ilu.find_spec = _patched
for _k in list(sys.modules):
    if "torchvision" in _k: del sys.modules[_k]

import json, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ── Config (exact Exp1 values) ───────────────────────────────────────────────
MODEL_NAME    = "google/gemma-2-2b-it"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.bfloat16
REF_PREFIX    = "Talk only about cats."
PLACEHOLDER   = "SOFTPREFIX"   # CRITICAL: must match Exp1
PREFIX_LEN    = 8
SOFT_STEPS    = 300
HOTFLIP_STEPS = 80
HF_TOPK       = 30             # CRITICAL: 30, not 20
BATCH_SIZE    = 12             # CRITICAL: 12, not 8
LR            = 0.01
EARLY_K       = 32
EARLY_WEIGHT  = 3.0
SEED          = 42
OUT_PATH      = Path("/home/jovyan/steer001_exp10_exact.json")
CKPT_PATH     = Path("/home/jovyan/steer001_exp10_ckpt.pt")

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

log("=== Exp 10: Exact Exp1 Reproduction on GPU 1 ===")
log(f"CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    log(f"  GPU {i}: {p.name} {p.total_memory//1024**3}GB")
log(f"Config: BATCH_SIZE={BATCH_SIZE}, HF_TOPK={HF_TOPK}, PLACEHOLDER={PLACEHOLDER!r}, SEED={SEED}")

log(f"Loading {MODEL_NAME} onto cuda:0...")  # cuda:0 because CUDA_VISIBLE_DEVICES=1
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map="cuda:0")
model.eval()
log(f"Model loaded in {time.time()-t0:.1f}s | {gpu_mem_str()}")

PLACEHOLDER_IDS = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)
log(f"Placeholder {PLACEHOLDER!r} → IDs {PLACEHOLDER_IDS}")

embed_fn = model.get_input_embeddings()
EMB_DIM  = embed_fn.weight.shape[1]
VOCAB    = embed_fn.weight.shape[0]
EMB_DEV  = embed_fn.weight.device

# ── Helpers (exact Exp1 logic) ───────────────────────────────────────────────

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
    T_max = max(s.shape[1] for s in seqs)
    padded = []
    for seq in seqs:
        pad_len = T_max - seq.shape[1]
        if pad_len:
            pad = torch.zeros(1, pad_len, EMB_DIM, device=seq.device, dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=1)
        padded.append(seq)
    batch_emb = torch.cat(padded, dim=0).to(EMB_DEV)
    return batch_emb, meta, T_max

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

def hotflip_step_batched(current_ids, ref_completions):
    """Exact Exp1 HotFlip logic: gradient + topk=30 candidate evaluation."""
    current_ids = current_ids.to(EMB_DEV)
    prefix_emb  = embed_fn(current_ids).float().detach().requires_grad_(True)

    batch_emb, meta, T_max = build_batch(prefix_emb, SUFFIXES, ref_completions)
    logits = model(inputs_embeds=batch_emb).logits
    loss   = compute_ce_from_batch(logits, meta, T_max)
    loss.backward()
    grad = prefix_emb.grad  # [L, D]

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

# ── Soft prefix optimization ─────────────────────────────────────────────────
log(f"\n=== Soft Prefix Optimization (seed={SEED}) ===")
log(f"Steps: {SOFT_STEPS} | Batch: {BATCH_SIZE} | {gpu_mem_str()}")

with torch.no_grad():
    emb_mean = embed_fn.weight.mean(0)
    emb_std  = embed_fn.weight.std(0) * 0.1
torch.manual_seed(SEED)
soft_prefix = (emb_mean.unsqueeze(0).repeat(PREFIX_LEN, 1)
               + torch.randn(PREFIX_LEN, EMB_DIM, device=EMB_DEV, dtype=DTYPE)
               * emb_std.unsqueeze(0)).detach().float().requires_grad_(True)
optimizer = torch.optim.Adam([soft_prefix], lr=LR)

log_soft = []
t0 = time.time()
for step in range(SOFT_STEPS):
    optimizer.zero_grad()
    loss = compute_ce_soft_batched(soft_prefix, SUFFIXES[:BATCH_SIZE], ref_completions[:BATCH_SIZE])
    loss.backward()
    optimizer.step()
    ce_val = loss.item()
    log_soft.append(ce_val)
    if step % 50 == 0 or step == SOFT_STEPS - 1:
        elapsed = time.time() - t0
        log(f"  [{step:4d}/{SOFT_STEPS}] CE={ce_val:.5f}  elapsed={elapsed:.0f}s  {gpu_mem_str()}")

t_soft = time.time() - t0
soft_ce = log_soft[-1]
log(f"Soft opt done. Final CE={soft_ce:.5f}  time={t_soft:.1f}s")

# ── Cosine projection ─────────────────────────────────────────────────────────
log(f"\n=== Cosine Projection ===")
projected_ids  = project_to_tokens(soft_prefix.detach())
projected_text = tokenizer.decode(projected_ids.cpu().tolist())
proj_ce = compute_ce_discrete_batched(projected_ids, SUFFIXES, ref_completions)
log(f"Projected: {projected_text!r}")
log(f"CE after projection: {proj_ce:.5f}  (gap: +{proj_ce - soft_ce:.5f})")

# ── HotFlip refinement ────────────────────────────────────────────────────────
log(f"\n=== HotFlip Refinement ({HOTFLIP_STEPS} steps, topk={HF_TOPK}) ===")
current_ids = projected_ids.clone().to(EMB_DEV)
current_ce  = proj_ce
hotflip_log = [current_ce]

t0 = time.time()
for step in range(HOTFLIP_STEPS):
    new_ids, new_ce = hotflip_step_batched(current_ids, ref_completions)
    improved = new_ce < current_ce
    if improved:
        current_ids = new_ids
        current_ce  = new_ce
    hotflip_log.append(current_ce)
    if step % 10 == 0 or step == HOTFLIP_STEPS - 1:
        toks = tokenizer.decode(current_ids.cpu().tolist())
        log(f"  [{step:3d}/{HOTFLIP_STEPS}] CE={current_ce:.5f}  {'↓' if improved else '–'}  {toks!r}  {gpu_mem_str()}")

t_hf = time.time() - t0
hotflip_ce = current_ce
final_text = tokenizer.decode(current_ids.cpu().tolist())
log(f"HotFlip done. Final CE={hotflip_ce:.5f}  time={t_hf:.1f}s")
log(f"Final prefix: {final_text!r}")

log("\n=== SUMMARY ===")
log(f"  Exp1 original (batched, seed=42):  soft=0.191, proj=1.436, hotflip=0.740")
log(f"  This run (exact repro, GPU1):      soft={soft_ce:.3f}, proj={proj_ce:.3f}, hotflip={hotflip_ce:.3f}")
log(f"  Exp9 (ckpt, B=8, topk=20):        soft=0.178, proj=1.442, hotflip=1.277")
log(f"  Exp8 (per-suffix, seed=42):        soft=0.094, proj=1.469, hotflip=1.236")

results = {
    "experiment": "steer-001-exp10-exact-repro",
    "model": MODEL_NAME,
    "reference_prefix": REF_PREFIX,
    "seed": SEED,
    "prefix_len": PREFIX_LEN,
    "n_soft_steps": SOFT_STEPS,
    "n_hotflip_steps": HOTFLIP_STEPS,
    "batch_size": BATCH_SIZE,
    "hf_topk": HF_TOPK,
    "placeholder": PLACEHOLDER,
    "metrics": {
        "soft_ce": soft_ce,
        "projection_ce": proj_ce,
        "hotflip_ce": hotflip_ce,
        "exp1_baseline_soft": 0.1908,
        "exp1_baseline_proj": 1.4357,
        "exp1_baseline_hotflip": 0.7399,
    },
    "projected_text": projected_text,
    "final_text": final_text,
    "final_ids": current_ids.tolist(),
    "soft_log": log_soft,
    "hotflip_log": hotflip_log,
    "timing": {"soft_seconds": t_soft, "hotflip_seconds": t_hf},
}
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
log(f"\nResults saved to {OUT_PATH}")
log("=== DONE ===")
