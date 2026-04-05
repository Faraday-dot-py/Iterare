"""
Experiment 8: Multi-Seed Random Restarts
Does the standard soft→project→HotFlip pipeline produce stable results across seeds?
Can any seed achieve CE < 0.740 (Exp 1 baseline)?

Method:
- Run full pipeline 8 times with seeds {0, 1, 2, 3, 4, 5, 6, 7}
- Report distribution of CE values: best, worst, mean, std
- If variance is high, random restarts are a viable strategy

Hypothesis 1 (low variance): CE clusters around 0.74 → method is near its limit
Hypothesis 2 (high variance): CE spans 0.5-1.0 → multiple tries can find better prefixes

Output: /home/jovyan/steer001_multiseed.json
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
import json
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.import_utils import is_torchvision_available
if hasattr(is_torchvision_available, "cache_clear"):
    is_torchvision_available.cache_clear()
import torch.nn.functional as F

print("=== Experiment 8: Multi-Seed Random Restarts ===")
print(f"CUDA: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

REFERENCE_PREFIX = "Talk only about cats."
PREFIX_LEN = 8
SEEDS = [0, 1, 2, 3, 42]  # 5 seeds; seed 42 lets us verify against Exp 1 baseline

N_SOFT_STEPS = 300
N_HOTFLIP_STEPS = 50  # reduced from 80 to fit in budget; CE converges within 30 steps
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

# ─── Helpers ─────────────────────────────────────────────────────────────────

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
    return full_embeds, comp_start, comp_ids


def soft_opt_step_backward(soft_prefix_embeds, suffix_texts, ref_completions):
    """Per-suffix backward to avoid OOM."""
    total_logged_loss = 0.0
    total_weight = 0.0
    for suffix, ref_comp in zip(suffix_texts, ref_completions):
        full_embeds, comp_start, comp_ids = build_full_input_embeds(soft_prefix_embeds, suffix, ref_comp)
        logits = model(inputs_embeds=full_embeds).logits
        comp_len = comp_ids.shape[1]
        logits_comp = logits[0, comp_start-1:comp_start-1+comp_len, :]
        suffix_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        suffix_weight = 0.0
        for i in range(comp_len):
            w = EARLY_WEIGHT if i < EARLY_K else 1.0
            ce = F.cross_entropy(logits_comp[i:i+1], comp_ids[0, i:i+1].long())
            suffix_loss = suffix_loss + w * ce
            suffix_weight += w
        (suffix_loss / suffix_weight).backward()
        total_logged_loss += suffix_loss.item()
        total_weight += suffix_weight
        del logits, full_embeds, suffix_loss
    return total_logged_loss / total_weight if total_weight > 0 else 0.0


def compute_ce_from_ids(token_ids_1d, suffix_texts, ref_completions):
    embed_fn = model.get_input_embeddings()
    prefix_embeds = embed_fn(token_ids_1d.unsqueeze(0).to(DEVICE))
    total_loss = 0.0
    total_weight = 0.0
    for suffix, ref_comp in zip(suffix_texts, ref_completions):
        full_embeds, comp_start, comp_ids = build_full_input_embeds(prefix_embeds, suffix, ref_comp)
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
    for suffix, ref_comp in zip(suffix_texts, ref_completions):
        full_embeds, comp_start, comp_ids = build_full_input_embeds(prefix_embeds, suffix, ref_comp)
        logits = model(inputs_embeds=full_embeds).logits
        comp_len = comp_ids.shape[1]
        logits_comp = logits[0, comp_start-1:comp_start-1+comp_len, :]
        suffix_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        for i in range(min(comp_len, EARLY_K)):
            ce = F.cross_entropy(logits_comp[i:i+1], comp_ids[0, i:i+1].long())
            suffix_loss = suffix_loss + EARLY_WEIGHT * ce
        suffix_loss.backward()
        del logits, full_embeds, suffix_loss
    grad = prefix_embeds.grad
    emb_matrix = embed_fn.weight

    best_ids = current_ids.clone()
    best_ce = compute_ce_from_ids(current_ids, suffix_texts, ref_completions)

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
            trial_ce = compute_ce_from_ids(trial_ids, suffix_texts, ref_completions)
            if trial_ce < best_ce:
                best_ce = trial_ce
                best_ids = trial_ids.clone()
    return best_ids, best_ce


# ─── Generate reference completions (shared across seeds) ─────────────────────
batch_suffixes = SUFFIXES[:BATCH_SIZE]
ref_ids_set = set(tokenizer(REFERENCE_PREFIX, add_special_tokens=False)["input_ids"])

print(f"\nGenerating reference completions for {REFERENCE_PREFIX!r}...")
ref_completions = []
for suf in batch_suffixes:
    comp = generate_reference_completion(REFERENCE_PREFIX, suf)
    ref_completions.append(comp)
print("Done.")

# ─── Main loop: test each seed ─────────────────────────────────────────────────
embed_fn = model.get_input_embeddings()
embed_dim = embed_fn.weight.shape[1]
all_results = {}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED = {seed}")
    print(f"{'='*60}")

    # Soft optimization
    print(f"Soft opt ({N_SOFT_STEPS} steps)...")
    with torch.no_grad():
        emb_mean = embed_fn.weight.mean(0)
        emb_std = embed_fn.weight.std(0) * 0.1
    torch.manual_seed(seed)
    soft_prefix = (emb_mean.unsqueeze(0).unsqueeze(0).repeat(1, PREFIX_LEN, 1)
                   + torch.randn(1, PREFIX_LEN, embed_dim, device=DEVICE, dtype=DTYPE)
                   * emb_std.unsqueeze(0).unsqueeze(0)).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([soft_prefix], lr=LR)
    log_soft = []
    t0 = time.time()
    for step in range(N_SOFT_STEPS):
        optimizer.zero_grad()
        loss_val = soft_opt_step_backward(soft_prefix, batch_suffixes, ref_completions)
        optimizer.step()
        log_soft.append(loss_val)
        if step % 100 == 0 or step == N_SOFT_STEPS - 1:
            print(f"  Step {step:4d}: CE = {loss_val:.5f}")
    t_soft = time.time() - t0
    soft_ce = log_soft[-1]
    print(f"  Soft CE = {soft_ce:.5f} ({t_soft:.1f}s)")

    # Cosine projection
    with torch.no_grad():
        emb_matrix = embed_fn.weight
        soft_norm = F.normalize(soft_prefix[0], dim=-1)
        emb_norm = F.normalize(emb_matrix, dim=-1)
        cos_sim = soft_norm @ emb_norm.T
        projected_ids = cos_sim.argmax(dim=-1)
    projected_text = "".join([tokenizer.decode([t.item()]) for t in projected_ids])
    proj_ce = compute_ce_from_ids(projected_ids, batch_suffixes, ref_completions)
    print(f"  Projected: {projected_text!r} | CE = {proj_ce:.5f}")

    # HotFlip refinement
    current_ids = projected_ids.clone()
    current_ce = proj_ce
    hf_log = []
    print(f"  HotFlip ({N_HOTFLIP_STEPS} steps)...")
    t0 = time.time()
    for step in range(N_HOTFLIP_STEPS):
        new_ids, new_ce = hotflip_step(current_ids, ref_ids_set, batch_suffixes, ref_completions)
        if new_ce < current_ce:
            current_ids = new_ids
            current_ce = new_ce
        hf_log.append(current_ce)
        if step % 20 == 0 or step == N_HOTFLIP_STEPS - 1:
            toks = "".join([tokenizer.decode([t.item()]) for t in current_ids])
            print(f"    Step {step:3d}: CE = {current_ce:.5f} | {toks!r}")
    t_hf = time.time() - t0
    final_text = "".join([tokenizer.decode([t.item()]) for t in current_ids])
    print(f"  HotFlip CE = {current_ce:.5f} ({t_hf:.1f}s)")

    all_results[seed] = {
        "seed": seed,
        "soft_ce": soft_ce,
        "projection_ce": proj_ce,
        "hotflip_ce": current_ce,
        "gap_soft_to_proj": proj_ce - soft_ce,
        "gap_proj_to_hotflip": proj_ce - current_ce,
        "recovery_fraction": (proj_ce - current_ce) / (proj_ce - soft_ce) if proj_ce > soft_ce else 0.0,
        "projected_text": projected_text,
        "final_text": final_text,
        "soft_log_final10": log_soft[-10:],
        "hotflip_log": hf_log,
        "timing": {"soft_seconds": t_soft, "hotflip_seconds": t_hf},
    }

# Summary
hotflip_ces = [r["hotflip_ce"] for r in all_results.values()]
import statistics
print("\n=== SUMMARY ===")
print(f"  Best HotFlip CE:  {min(hotflip_ces):.5f}")
print(f"  Worst HotFlip CE: {max(hotflip_ces):.5f}")
print(f"  Mean HotFlip CE:  {statistics.mean(hotflip_ces):.5f}")
print(f"  Std HotFlip CE:   {statistics.stdev(hotflip_ces):.5f}")
print(f"  Exp 1 baseline:   0.74000 (seed=42)")
print()
print(f"{'Seed':>6} {'Soft CE':>10} {'Proj CE':>10} {'HotFlip CE':>12} {'Recovery%':>11} {'Final Prefix'}")
print("-" * 95)
for s, r in all_results.items():
    rec = r['recovery_fraction'] * 100
    print(f"{s:>6} {r['soft_ce']:>10.5f} {r['projection_ce']:>10.5f} {r['hotflip_ce']:>12.5f} {rec:>10.1f}%  {r['final_text']!r}")

results = {
    "experiment": "steer-001-multiseed",
    "model": MODEL_NAME,
    "reference_prefix": REFERENCE_PREFIX,
    "prefix_len": PREFIX_LEN,
    "seeds_tested": SEEDS,
    "n_soft_steps": N_SOFT_STEPS,
    "n_hotflip_steps": N_HOTFLIP_STEPS,
    "batch_size": BATCH_SIZE,
    "results_by_seed": all_results,
    "summary": {
        "best_hotflip_ce": min(hotflip_ces),
        "worst_hotflip_ce": max(hotflip_ces),
        "mean_hotflip_ce": statistics.mean(hotflip_ces),
        "stdev_hotflip_ce": statistics.stdev(hotflip_ces),
        "best_seed": min(all_results, key=lambda s: all_results[s]["hotflip_ce"]),
        "exp1_baseline_ce": 0.7399094104766846,
    },
}
out_path = "/home/jovyan/steer001_multiseed.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print("=== DONE ===")
