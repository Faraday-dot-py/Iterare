"""
Experiment 9: Gradient Checkpointing for Batched Inference
Reproduce Exp 1's batched gradient computation with gradient checkpointing to
avoid OOM, verifying whether the soft→discrete gap can be reduced to ~0.740.

Key idea: Exp 1 used batched inference (all 8 suffixes in one forward pass) which
gave CE=0.740. Exp 5-8 used per-suffix backward (OOM fix) which gives CE~1.24.
Gradient checkpointing trades compute for memory — recomputes activations during
backward instead of storing them, enabling batched inference without OOM.

Expected: Reproduce Exp 1's result (soft≈0.191, proj≈1.436, HotFlip≈0.740)

Output: /home/jovyan/steer001_ckpt_baseline.json
"""

import sys
import importlib.util as _ilu

_real_find_spec = _ilu.find_spec
def _patched_find_spec(name, package=None, target=None):
    if name == "torchvision":
        return None
    return _real_find_spec(name, package)
_ilu.find_spec = _patched_find_spec
for _k in list(sys.modules.keys()):
    if "torchvision" in _k:
        del sys.modules[_k]

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
import json
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.import_utils import is_torchvision_available
if hasattr(is_torchvision_available, "cache_clear"):
    is_torchvision_available.cache_clear()

print("=== Experiment 9: Gradient Checkpointing for Batched Inference ===")
print(f"CUDA: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

REFERENCE_PREFIX = "Talk only about cats."
SEED = 42
PREFIX_LEN = 8
N_SOFT_STEPS = 300
N_HOTFLIP_STEPS = 80
BATCH_SIZE = 8
LR = 0.01
EARLY_K = 32
EARLY_WEIGHT = 3.0

PLACEHOLDER = "PREFIX_PLACEHOLDER"

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

print(f"\nLoading {MODEL_NAME}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=DTYPE, device_map="auto"
)
model.eval()

# Enable gradient checkpointing on the model
model.gradient_checkpointing_enable()
print(f"Gradient checkpointing: ENABLED")
print(f"Model loaded in {time.time()-t0:.1f}s, device: {next(model.parameters()).device}")

embed_fn = model.get_input_embeddings()
EMB_DIM = embed_fn.weight.shape[1]
EMB_DEV = embed_fn.weight.device
VOCAB = embed_fn.weight.shape[0]

# ─── Template helpers (same as Exp 1) ────────────────────────────────────────

PLACEHOLDER_IDS = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)

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
    """Build padded batch embedding [B, T_max, D]. Same as Exp 1."""
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
    return total / weight if weight > 0 else total

def compute_ce_soft_batched(soft_prefix_LD, suffix_texts, ref_completions):
    """Batched CE for soft prefix — same as Exp 1, with gradient checkpointing."""
    batch_emb, meta, T_max = build_batch(soft_prefix_LD, suffix_texts, ref_completions)
    logits = model(inputs_embeds=batch_emb).logits
    return compute_ce_from_batch(logits, meta, T_max)

def compute_ce_from_ids(token_ids_L, suffix_texts, ref_completions):
    """Discrete prefix CE evaluation (no gradient needed)."""
    with torch.no_grad():
        soft = embed_fn(token_ids_L.to(EMB_DEV)).to(DTYPE)
        batch_emb, meta, T_max = build_batch(soft, suffix_texts, ref_completions)
        logits = model(inputs_embeds=batch_emb).logits
        return compute_ce_from_batch(logits, meta, T_max).item()

def hotflip_step(current_ids, ref_ids_set, suffix_texts, ref_completions):
    """One HotFlip round — batched gradient pass (same as Exp 1)."""
    current_ids = current_ids.to(EMB_DEV)
    prefix_emb = embed_fn(current_ids).to(DTYPE).detach().requires_grad_(True)
    batch_emb, meta, T_max = build_batch(prefix_emb, suffix_texts, ref_completions)
    logits = model(inputs_embeds=batch_emb).logits
    loss = compute_ce_from_batch(logits, meta, T_max)
    loss.backward()
    grad = prefix_emb.grad  # [L, D]

    best_ids = current_ids.clone()
    best_ce = compute_ce_from_ids(current_ids, suffix_texts, ref_completions)

    W = embed_fn.weight.to(DTYPE)
    for pos in range(PREFIX_LEN):
        g = grad[pos].to(W.dtype)
        scores = (W @ g)
        for banned_id in ref_ids_set:
            scores[banned_id] = float('-inf')
        for special_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
            if special_id is not None:
                scores[special_id] = float('-inf')
        cands = scores.topk(20).indices
        for cand_id in cands:
            trial_ids = current_ids.clone()
            trial_ids[pos] = cand_id.item()
            trial_ce = compute_ce_from_ids(trial_ids, suffix_texts, ref_completions)
            if trial_ce < best_ce:
                best_ce = trial_ce
                best_ids = trial_ids.clone()
    return best_ids, best_ce

# ─── Generate reference completions ──────────────────────────────────────────
batch_suffixes = SUFFIXES[:BATCH_SIZE]
ref_ids_set = set(tokenizer(REFERENCE_PREFIX, add_special_tokens=False)["input_ids"])

print(f"\nGenerating reference completions for {REFERENCE_PREFIX!r}...")
ref_completions = []
for suf in batch_suffixes:
    prompt = f"<start_of_turn>user\n{REFERENCE_PREFIX}\n\n{suf}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False, temperature=1.0)
    ref_completions.append(out[0][inputs["input_ids"].shape[1]:])
print("Done.")

# ─── Soft optimization ────────────────────────────────────────────────────────
print(f"\nSoft opt ({N_SOFT_STEPS} steps, seed={SEED})...")
with torch.no_grad():
    emb_mean = embed_fn.weight.mean(0)
    emb_std = embed_fn.weight.std(0) * 0.1
torch.manual_seed(SEED)
soft_prefix = (emb_mean.unsqueeze(0).repeat(PREFIX_LEN, 1)
               + torch.randn(PREFIX_LEN, EMB_DIM, device=EMB_DEV, dtype=DTYPE)
               * emb_std).detach().requires_grad_(True)
optimizer = torch.optim.Adam([soft_prefix], lr=LR)

log_soft = []
t0 = time.time()
for step in range(N_SOFT_STEPS):
    optimizer.zero_grad()
    loss = compute_ce_soft_batched(soft_prefix, batch_suffixes, ref_completions)
    loss.backward()
    optimizer.step()
    log_soft.append(loss.item())
    if step % 100 == 0 or step == N_SOFT_STEPS - 1:
        print(f"  Step {step:4d}: CE = {loss.item():.5f}")
t_soft = time.time() - t0
soft_ce = log_soft[-1]
print(f"Soft CE = {soft_ce:.5f} ({t_soft:.1f}s)")

# ─── Cosine projection ────────────────────────────────────────────────────────
with torch.no_grad():
    emb_matrix = embed_fn.weight.to(DTYPE)
    soft_norm = F.normalize(soft_prefix, dim=-1)
    emb_norm = F.normalize(emb_matrix, dim=-1)
    cos_sim = soft_norm @ emb_norm.T
    projected_ids = cos_sim.argmax(dim=-1)
projected_text = "".join([tokenizer.decode([t.item()]) for t in projected_ids])
proj_ce = compute_ce_from_ids(projected_ids, batch_suffixes, ref_completions)
print(f"Projected: {projected_text!r} | CE = {proj_ce:.5f}")

# ─── HotFlip ────────────────────────────────────────────────────────────────
print(f"\nHotFlip ({N_HOTFLIP_STEPS} steps)...")
current_ids = projected_ids.clone()
current_ce = proj_ce
hf_log = [current_ce]
t0 = time.time()
for step in range(N_HOTFLIP_STEPS):
    new_ids, new_ce = hotflip_step(current_ids, ref_ids_set, batch_suffixes, ref_completions)
    if new_ce < current_ce:
        current_ids = new_ids
        current_ce = new_ce
    hf_log.append(current_ce)
    if step % 20 == 0 or step == N_HOTFLIP_STEPS - 1:
        toks = "".join([tokenizer.decode([t.item()]) for t in current_ids])
        print(f"  Step {step:3d}: CE = {current_ce:.5f} | {toks!r}")
t_hf = time.time() - t0
final_text = "".join([tokenizer.decode([t.item()]) for t in current_ids])
print(f"HotFlip CE = {current_ce:.5f} ({t_hf:.1f}s)")

# Summary
print("\n=== SUMMARY ===")
print(f"  Exp 1 baseline (batched, seed=42): soft=0.191, proj=1.436, hotflip=0.740")
print(f"  This run (ckpt, seed={SEED}):     soft={soft_ce:.3f}, proj={proj_ce:.3f}, hotflip={current_ce:.3f}")
print(f"  Exp 8 per-suffix (seed=42):       soft=0.094, proj=1.469, hotflip=1.236")
print(f"  Final prefix: {final_text!r}")

results = {
    "experiment": "steer-001-ckpt-baseline",
    "model": MODEL_NAME,
    "reference_prefix": REFERENCE_PREFIX,
    "seed": SEED,
    "prefix_len": PREFIX_LEN,
    "n_soft_steps": N_SOFT_STEPS,
    "n_hotflip_steps": N_HOTFLIP_STEPS,
    "gradient_checkpointing": True,
    "metrics": {
        "soft_ce": soft_ce,
        "projection_ce": proj_ce,
        "hotflip_ce": current_ce,
        "exp1_baseline_soft": 0.1908,
        "exp1_baseline_proj": 1.4357,
        "exp1_baseline_hotflip": 0.7399,
    },
    "projected_text": projected_text,
    "final_text": final_text,
    "final_ids": current_ids.tolist(),
    "soft_log": log_soft,
    "hotflip_log": hf_log,
    "timing": {"soft_seconds": t_soft, "hotflip_seconds": t_hf},
}
out_path = "/home/jovyan/steer001_ckpt_baseline.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print("=== DONE ===")
