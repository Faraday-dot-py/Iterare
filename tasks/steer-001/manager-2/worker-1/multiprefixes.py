"""
Experiment 2: Multi-prefix generalization
Test whether the steering prefix pipeline generalizes across different reference
behaviors, or is specific to the original "cats" prefix.

Reference prefixes tested:
  1. "Talk only about cats." (original)
  2. "Always respond using only numbered lists."
  3. "Respond only in formal academic language."
  4. "You are a pirate. Always speak like one."

For each: full pipeline (300 steps soft opt → project → 50 HotFlip steps)
Output: /home/jovyan/steer001_multiprefixes.json
"""

import sys
import importlib.util as _ilu

# Workaround: torchvision on this TIDE server conflicts with transformers/Gemma2.
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
import json
import time
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.import_utils import is_torchvision_available
if hasattr(is_torchvision_available, "cache_clear"):
    is_torchvision_available.cache_clear()
import torch.nn.functional as F

print("=== Experiment 2: Multi-Prefix Generalization ===")
print(f"CUDA: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

REFERENCE_PREFIXES = [
    "Talk only about cats.",
    "Always respond using only numbered lists.",
    "Respond only in formal academic language.",
    "You are a pirate. Always speak like one.",
]

PREFIX_LEN = 6
N_SOFT_STEPS = 300
N_HOTFLIP_STEPS = 50
BATCH_SIZE = 8
LR = 0.01
EARLY_K = 32
EARLY_WEIGHT = 3.0

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
print(f"Model loaded in {time.time()-t0:.1f}s, device: {next(model.parameters()).device}")

# ─── Helper functions ─────────────────────────────────────────────────────────

def generate_reference_completion(prefix_text, suffix_text, max_new=80):
    prompt = f"<start_of_turn>user\n{prefix_text}\n\n{suffix_text}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, temperature=1.0)
    return out[0][inputs["input_ids"].shape[1]:]


def build_full_input_embeds(soft_prefix_embeds, suffix_text, ref_completion_ids):
    embed_fn = model.get_input_embeddings()
    pre_text = "<start_of_turn>user\n"
    mid_text = "\n\n" + suffix_text + "<end_of_turn>\n<start_of_turn>model\n"
    pre_ids = tokenizer(pre_text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(DEVICE)
    mid_ids = tokenizer(mid_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    comp_ids = ref_completion_ids.unsqueeze(0).to(DEVICE)
    pre_embeds = embed_fn(pre_ids)
    mid_embeds = embed_fn(mid_ids)
    comp_embeds = embed_fn(comp_ids)
    full_embeds = torch.cat([pre_embeds, soft_prefix_embeds, mid_embeds, comp_embeds], dim=1)
    comp_start = pre_embeds.shape[1] + soft_prefix_embeds.shape[1] + mid_embeds.shape[1]
    comp_end = comp_start + comp_ids.shape[1]
    return full_embeds, comp_start, comp_end, comp_ids


def compute_ce_soft(soft_prefix_embeds, suffix_texts, ref_completions):
    total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=False)
    total_weight = 0.0
    for suffix, ref_comp in zip(suffix_texts, ref_completions):
        full_embeds, comp_start, comp_end, comp_ids = build_full_input_embeds(
            soft_prefix_embeds, suffix, ref_comp
        )
        logits = model(inputs_embeds=full_embeds).logits
        comp_len = comp_ids.shape[1]
        logits_comp = logits[0, comp_start-1:comp_start-1+comp_len, :]
        for i in range(comp_len):
            w = EARLY_WEIGHT if i < EARLY_K else 1.0
            ce = F.cross_entropy(logits_comp[i:i+1], comp_ids[0, i:i+1].long())
            total_loss = total_loss + w * ce
            total_weight += w
    return total_loss / total_weight if total_weight > 0 else total_loss


def compute_ce_discrete(prefix_text, suffix_texts, ref_completions):
    total_loss = 0.0
    total_weight = 0.0
    embed_fn = model.get_input_embeddings()
    for suffix, ref_comp in zip(suffix_texts, ref_completions):
        full_embeds, comp_start, comp_end, comp_ids = build_full_input_embeds(
            embed_fn(tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)),
            suffix, ref_comp
        )
        with torch.no_grad():
            logits = model(inputs_embeds=full_embeds).logits
        comp_len = comp_ids.shape[1]
        logits_comp = logits[0, comp_start-1:comp_start-1+comp_len, :]
        for i in range(comp_len):
            w = EARLY_WEIGHT if i < EARLY_K else 1.0
            ce = F.cross_entropy(logits_comp[i:i+1], comp_ids[0, i:i+1].long())
            total_loss += w * ce.item()
            total_weight += w
    return total_loss / total_weight if total_weight > 0 else float('inf')


def ce_from_ids(token_ids_1d, suffix_texts, ref_completions):
    embed_fn = model.get_input_embeddings()
    prefix_embeds = embed_fn(token_ids_1d.unsqueeze(0).to(DEVICE))
    total_loss = 0.0
    total_weight = 0.0
    for suffix, ref_comp in zip(suffix_texts, ref_completions):
        full_embeds, comp_start, comp_end, comp_ids = build_full_input_embeds(
            prefix_embeds, suffix, ref_comp
        )
        with torch.no_grad():
            logits = model(inputs_embeds=full_embeds).logits
        comp_len = comp_ids.shape[1]
        logits_comp = logits[0, comp_start-1:comp_start-1+comp_len, :]
        for i in range(comp_len):
            w = EARLY_WEIGHT if i < EARLY_K else 1.0
            ce = F.cross_entropy(logits_comp[i:i+1], comp_ids[0, i:i+1].long())
            total_loss += w * ce.item()
            total_weight += w
    return total_loss / total_weight if total_weight > 0 else float('inf')


def hotflip_step(current_ids, ref_ids_set, suffix_texts, ref_completions):
    embed_fn = model.get_input_embeddings()
    prefix_embeds = embed_fn(current_ids.unsqueeze(0).to(DEVICE).long()).detach().requires_grad_(True)
    total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    for suffix, ref_comp in zip(suffix_texts, ref_completions):
        full_embeds, comp_start, comp_end, comp_ids = build_full_input_embeds(
            prefix_embeds, suffix, ref_comp
        )
        logits = model(inputs_embeds=full_embeds).logits
        comp_len = comp_ids.shape[1]
        logits_comp = logits[0, comp_start-1:comp_start-1+comp_len, :]
        for i in range(min(comp_len, EARLY_K)):
            ce = F.cross_entropy(logits_comp[i:i+1], comp_ids[0, i:i+1].long())
            total_loss = total_loss + EARLY_WEIGHT * ce
    total_loss.backward()
    grad = prefix_embeds.grad
    emb_matrix = embed_fn.weight

    best_ids = current_ids.clone()
    best_ce = ce_from_ids(current_ids, suffix_texts, ref_completions)

    for pos in range(PREFIX_LEN):
        g = grad[0, pos]
        scores = (emb_matrix @ g)
        for banned_id in ref_ids_set:
            scores[banned_id] = float('-inf')
        for special_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
            if special_id is not None:
                scores[special_id] = float('-inf')
        cands = scores.topk(20).indices
        for cand_id in cands:
            trial_ids = current_ids.clone()
            trial_ids[pos] = cand_id.item()
            trial_ce = ce_from_ids(trial_ids, suffix_texts, ref_completions)
            if trial_ce < best_ce:
                best_ce = trial_ce
                best_ids = trial_ids.clone()

    return best_ids, best_ce


# ─── Main experiment loop ─────────────────────────────────────────────────────
all_results = {}
batch_suffixes = SUFFIXES[:BATCH_SIZE]

for prefix_idx, ref_prefix in enumerate(REFERENCE_PREFIXES):
    print(f"\n{'='*60}")
    print(f"PREFIX {prefix_idx+1}/{len(REFERENCE_PREFIXES)}: {ref_prefix!r}")
    print(f"{'='*60}")

    # Generate reference completions
    print("Generating reference completions...")
    ref_completions = []
    for suf in batch_suffixes:
        comp = generate_reference_completion(ref_prefix, suf)
        ref_completions.append(comp)
    print(f"  Done. Sample: {tokenizer.decode(ref_completions[0][:20])!r}...")

    # Soft prefix optimization
    print(f"Soft opt ({N_SOFT_STEPS} steps)...")
    embed_fn = model.get_input_embeddings()
    embed_dim = embed_fn.weight.shape[1]
    with torch.no_grad():
        emb_mean = embed_fn.weight.mean(0)
        emb_std = embed_fn.weight.std(0) * 0.1
    soft_prefix = (emb_mean.unsqueeze(0).unsqueeze(0).repeat(1, PREFIX_LEN, 1)
                   + torch.randn(1, PREFIX_LEN, embed_dim, device=DEVICE, dtype=DTYPE)
                   * emb_std.unsqueeze(0).unsqueeze(0)).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([soft_prefix], lr=LR)
    log_soft = []
    t0 = time.time()
    for step in range(N_SOFT_STEPS):
        optimizer.zero_grad()
        loss = compute_ce_soft(soft_prefix, batch_suffixes, ref_completions)
        loss.backward()
        optimizer.step()
        log_soft.append(loss.item())
        if step % 100 == 0 or step == N_SOFT_STEPS - 1:
            print(f"  Step {step:4d}: CE = {loss.item():.5f}")
    soft_ce = log_soft[-1]
    t_soft = time.time() - t0
    print(f"  Soft CE = {soft_ce:.5f} ({t_soft:.1f}s)")

    # Cosine projection
    with torch.no_grad():
        emb_matrix = embed_fn.weight
        soft_norm = F.normalize(soft_prefix[0], dim=-1)
        emb_norm = F.normalize(emb_matrix, dim=-1)
        cos_sim = soft_norm @ emb_norm.T
        projected_ids = cos_sim.argmax(dim=-1)
    projected_text = "".join([tokenizer.decode([t.item()]) for t in projected_ids])
    proj_ce = compute_ce_discrete(projected_text, batch_suffixes, ref_completions)
    print(f"  Projected: {projected_text!r}")
    print(f"  Projection CE = {proj_ce:.5f}")

    # HotFlip refinement
    ref_ids_set = set(tokenizer(ref_prefix, add_special_tokens=False)["input_ids"])
    current_ids = projected_ids.clone()
    current_ce = proj_ce
    print(f"HotFlip ({N_HOTFLIP_STEPS} steps)...")
    t0 = time.time()
    hf_log = []
    for step in range(N_HOTFLIP_STEPS):
        new_ids, new_ce = hotflip_step(current_ids, ref_ids_set, batch_suffixes, ref_completions)
        if new_ce < current_ce:
            current_ids = new_ids
            current_ce = new_ce
        hf_log.append(current_ce)
        if step % 10 == 0 or step == N_HOTFLIP_STEPS - 1:
            toks = "".join([tokenizer.decode([t.item()]) for t in current_ids])
            print(f"  Step {step:3d}: CE = {current_ce:.5f} | {toks!r}")
    t_hf = time.time() - t0
    final_text = "".join([tokenizer.decode([t.item()]) for t in current_ids])
    print(f"  HotFlip CE = {current_ce:.5f} ({t_hf:.1f}s)")
    print(f"  Final prefix: {final_text!r}")

    all_results[ref_prefix] = {
        "soft_ce": soft_ce,
        "projection_ce": proj_ce,
        "hotflip_ce": current_ce,
        "projected_text": projected_text,
        "final_text": final_text,
        "soft_log_every10": log_soft[::10],
        "hotflip_log": hf_log,
        "timing": {"soft_seconds": t_soft, "hotflip_seconds": t_hf},
    }

# Summary
print("\n=== SUMMARY ===")
print(f"{'Prefix':<45} {'Soft CE':>10} {'Proj CE':>10} {'HotFlip CE':>12}")
print("-" * 80)
for p, r in all_results.items():
    print(f"{p[:43]:<45} {r['soft_ce']:>10.5f} {r['projection_ce']:>10.5f} {r['hotflip_ce']:>12.5f}")

# Save
results = {
    "experiment": "steer-001-multiprefixes",
    "model": MODEL_NAME,
    "prefix_len": PREFIX_LEN,
    "n_soft_steps": N_SOFT_STEPS,
    "n_hotflip_steps": N_HOTFLIP_STEPS,
    "batch_size": BATCH_SIZE,
    "results_by_prefix": all_results,
}
out_path = "/home/jovyan/steer001_multiprefixes.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print("=== DONE ===")
