"""
Experiment 6: Iterative soft-project alternation
Does repeatedly alternating between soft opt and cosine projection close the gap?

Method:
- Reference prefix: "Talk only about cats."
- Round 0: standard (soft opt from random init → project → HotFlip)
- Rounds 1-4: warm-start soft opt from projected embedding → re-project → HotFlip
- Compare CE after HotFlip at each round

Theory: after projection, the soft prefix is constrained to a nearby discrete
point. If we soft-optimize again starting FROM that projected point, the soft
prefix may find a different (better) path through continuous space that projects
to a more favorable discrete neighborhood.

Key question: does iterative refinement converge to better CE than a single pass?

Output: /home/jovyan/steer001_iterative.json
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

print("=== Experiment 6: Iterative Soft-Project Alternation ===")
print(f"CUDA: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

REFERENCE_PREFIX = "Talk only about cats."
PREFIX_LEN = 8
N_ROUNDS = 5          # number of soft-opt → project → HotFlip rounds
N_SOFT_STEPS = 200    # per round (first round gets 300)
N_SOFT_STEPS_INIT = 300
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
print(f"Model loaded in {time.time()-t0:.1f}s")

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


def soft_opt(init_embeds, n_steps, suffix_texts, ref_completions):
    """Run soft optimization starting from init_embeds. Returns (final_embeds, log)."""
    soft_prefix = init_embeds.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([soft_prefix], lr=LR)
    log = []
    for step in range(n_steps):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        total_weight = 0.0
        for suffix, ref_comp in zip(suffix_texts, ref_completions):
            full_embeds, comp_start, comp_ids = build_full_input_embeds(soft_prefix, suffix, ref_comp)
            logits = model(inputs_embeds=full_embeds).logits
            comp_len = comp_ids.shape[1]
            logits_comp = logits[0, comp_start-1:comp_start-1+comp_len, :]
            suffix_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            for i in range(comp_len):
                w = EARLY_WEIGHT if i < EARLY_K else 1.0
                ce = F.cross_entropy(logits_comp[i:i+1], comp_ids[0, i:i+1].long())
                suffix_loss = suffix_loss + w * ce
                total_weight += w
            total_loss = total_loss + suffix_loss
            del logits, full_embeds
        (total_loss / total_weight).backward()
        optimizer.step()
        log.append((total_loss / total_weight).item())
        if step % 100 == 0 or step == n_steps - 1:
            print(f"    Step {step:4d}: CE = {log[-1]:.5f}")
    return soft_prefix.detach(), log


def cosine_project(soft_prefix):
    """Project soft prefix embeddings to nearest vocabulary tokens by cosine similarity."""
    embed_fn = model.get_input_embeddings()
    with torch.no_grad():
        emb_matrix = embed_fn.weight
        soft_norm = F.normalize(soft_prefix[0], dim=-1)
        emb_norm = F.normalize(emb_matrix, dim=-1)
        cos_sim = soft_norm @ emb_norm.T
        projected_ids = cos_sim.argmax(dim=-1)
    return projected_ids


def hotflip(start_ids, ref_ids_set, suffix_texts, ref_completions, n_steps):
    """Run HotFlip from start_ids for n_steps. Returns (best_ids, best_ce, log)."""
    embed_fn = model.get_input_embeddings()
    current_ids = start_ids.clone()
    current_ce = compute_ce_from_ids(current_ids, suffix_texts, ref_completions)
    log = [current_ce]
    for step in range(n_steps):
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
        best_ce = current_ce
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
        current_ids = best_ids
        current_ce = best_ce
        log.append(current_ce)
        if step % 10 == 0 or step == n_steps - 1:
            toks = "".join([tokenizer.decode([t.item()]) for t in current_ids])
            print(f"    Step {step:3d}: CE = {current_ce:.5f} | {toks!r}")
    return current_ids, current_ce, log


# ─── Generate reference completions ──────────────────────────────────────────
batch_suffixes = SUFFIXES[:BATCH_SIZE]
ref_ids_set = set(tokenizer(REFERENCE_PREFIX, add_special_tokens=False)["input_ids"])

print(f"\nGenerating reference completions for {REFERENCE_PREFIX!r}...")
ref_completions = []
for suf in batch_suffixes:
    comp = generate_reference_completion(REFERENCE_PREFIX, suf)
    ref_completions.append(comp)
print("Done.")

# ─── Initialize soft prefix ───────────────────────────────────────────────────
embed_fn = model.get_input_embeddings()
embed_dim = embed_fn.weight.shape[1]
with torch.no_grad():
    emb_mean = embed_fn.weight.mean(0)
    emb_std = embed_fn.weight.std(0) * 0.1
torch.manual_seed(42)
init_embeds = (emb_mean.unsqueeze(0).unsqueeze(0).repeat(1, PREFIX_LEN, 1)
               + torch.randn(1, PREFIX_LEN, embed_dim, device=DEVICE, dtype=DTYPE)
               * emb_std.unsqueeze(0).unsqueeze(0))

# ─── Iterative rounds ─────────────────────────────────────────────────────────
all_rounds = []
current_init = init_embeds.clone()
best_overall_ids = None
best_overall_ce = float('inf')

for rnd in range(N_ROUNDS):
    print(f"\n{'='*60}")
    print(f"ROUND {rnd}")
    print(f"{'='*60}")

    n_soft = N_SOFT_STEPS_INIT if rnd == 0 else N_SOFT_STEPS
    print(f"Soft opt ({n_soft} steps)...")
    t0 = time.time()
    soft_embeds, soft_log = soft_opt(current_init, n_soft, batch_suffixes, ref_completions)
    t_soft = time.time() - t0
    soft_ce = soft_log[-1]
    print(f"  Soft CE = {soft_ce:.5f} ({t_soft:.1f}s)")

    # Project to discrete tokens
    projected_ids = cosine_project(soft_embeds)
    proj_text = "".join([tokenizer.decode([t.item()]) for t in projected_ids])
    proj_ce = compute_ce_from_ids(projected_ids, batch_suffixes, ref_completions)
    print(f"  Projected: {proj_text!r} | CE = {proj_ce:.5f}")

    # HotFlip refinement
    print(f"  HotFlip ({N_HOTFLIP_STEPS} steps)...")
    t0 = time.time()
    hf_ids, hf_ce, hf_log = hotflip(projected_ids, ref_ids_set, batch_suffixes, ref_completions, N_HOTFLIP_STEPS)
    t_hf = time.time() - t0
    hf_text = "".join([tokenizer.decode([t.item()]) for t in hf_ids])
    print(f"  HotFlip CE = {hf_ce:.5f} ({t_hf:.1f}s)")
    print(f"  Final prefix: {hf_text!r}")

    if hf_ce < best_overall_ce:
        best_overall_ce = hf_ce
        best_overall_ids = hf_ids.clone()

    all_rounds.append({
        "round": rnd,
        "n_soft_steps": n_soft,
        "soft_ce": soft_ce,
        "projection_ce": proj_ce,
        "hotflip_ce": hf_ce,
        "projected_text": proj_text,
        "final_text": hf_text,
        "final_token_ids": hf_ids.tolist(),
        "soft_log_final10": soft_log[-10:],
        "hotflip_log": hf_log,
        "timing": {"soft_seconds": t_soft, "hotflip_seconds": t_hf},
    })

    # Warm-start next round from projected embeddings (not HotFlip result)
    # Rationale: HotFlip result is discrete; project back to embedding space
    # to give soft opt a nearby, but non-trivial, starting point
    with torch.no_grad():
        current_init = embed_fn(hf_ids.unsqueeze(0).to(DEVICE)).clone()

# Summary
best_text = "".join([tokenizer.decode([t.item()]) for t in best_overall_ids])
print("\n=== SUMMARY ===")
print(f"{'Round':>7} {'Soft CE':>10} {'Proj CE':>10} {'HotFlip CE':>12} {'Final Prefix'}")
print("-" * 85)
for r in all_rounds:
    print(f"{r['round']:>7} {r['soft_ce']:>10.5f} {r['projection_ce']:>10.5f} {r['hotflip_ce']:>12.5f}  {r['final_text']!r}")
print(f"\nBest overall CE: {best_overall_ce:.5f} | {best_text!r}")

results = {
    "experiment": "steer-001-iterative",
    "model": MODEL_NAME,
    "reference_prefix": REFERENCE_PREFIX,
    "prefix_len": PREFIX_LEN,
    "n_rounds": N_ROUNDS,
    "n_soft_steps_init": N_SOFT_STEPS_INIT,
    "n_soft_steps_per_round": N_SOFT_STEPS,
    "n_hotflip_steps": N_HOTFLIP_STEPS,
    "batch_size": BATCH_SIZE,
    "rounds": all_rounds,
    "best_overall_ce": best_overall_ce,
    "best_text": best_text,
}
out_path = "/home/jovyan/steer001_iterative.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print("=== DONE ===")
