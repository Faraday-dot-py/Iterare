"""
Experiment 4: Semantic naturalness penalty in HotFlip
Can adding a naturalness term to HotFlip push toward readable prefixes?

Method:
- Modify HotFlip: add penalty term = λ * (1 - P_LM(token | context)) where
  P_LM is from distilgpt2 used as a naturalness prior
- λ sweep: {0.0, 0.1, 0.5, 1.0}
- Metric: CE (lower = better behavioral match) + word-level naturalness score

Output: /home/jovyan/steer001_naturalness.json
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
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
from transformers.utils.import_utils import is_torchvision_available
if hasattr(is_torchvision_available, "cache_clear"):
    is_torchvision_available.cache_clear()
import torch.nn.functional as F

print("=== Experiment 4: Semantic Naturalness Penalty in HotFlip ===")
print(f"CUDA: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-2b-it"
LM_PRIOR_NAME = "distilgpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

REFERENCE_PREFIX = "Talk only about cats."
LAMBDA_VALUES = [0.0, 0.1, 0.5, 1.0]

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

# English word naturalness
COMMON_ENGLISH = set("""
a an the is are was were be been being am have has had do does did will would could should may
might shall can cannot not no yes i you he she it we they me him her us them my your his its our
their this that these those what which who whom whose when where why how all some any few more
most other into over after about from up out on off through with without as at by for to in of
and but or so yet nor if then because since until unless while although though even just only also
very really quite about around than before after between both each every many much such into
talk about cats only speak always like formal academic language respond using numbered lists
pirate speak feline meow purr whiskers""".split())


def naturalness_score(token_list):
    count = 0
    for tok in token_list:
        tok_clean = re.sub(r'[^a-z]', '', tok.strip().lower())
        if tok_clean in COMMON_ENGLISH or (len(tok_clean) >= 3 and tok_clean.isalpha()):
            count += 1
    return count / len(token_list) if token_list else 0.0


print(f"\nLoading {MODEL_NAME}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=DTYPE, device_map="auto"
)
model.eval()
print(f"Gemma model loaded in {time.time()-t0:.1f}s")

print(f"Loading {LM_PRIOR_NAME}...")
t0 = time.time()
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(LM_PRIOR_NAME)
gpt2_model = GPT2LMHeadModel.from_pretrained(LM_PRIOR_NAME).to(DEVICE)
gpt2_model.eval()
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
print(f"GPT2 prior loaded in {time.time()-t0:.1f}s")

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


def compute_ce_soft(soft_prefix_embeds, suffix_texts, ref_completions):
    total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=False)
    total_weight = 0.0
    for suffix, ref_comp in zip(suffix_texts, ref_completions):
        full_embeds, comp_start, comp_ids = build_full_input_embeds(soft_prefix_embeds, suffix, ref_comp)
        logits = model(inputs_embeds=full_embeds).logits
        comp_len = comp_ids.shape[1]
        logits_comp = logits[0, comp_start-1:comp_start-1+comp_len, :]
        for i in range(comp_len):
            w = EARLY_WEIGHT if i < EARLY_K else 1.0
            ce = F.cross_entropy(logits_comp[i:i+1], comp_ids[0, i:i+1].long())
            total_loss = total_loss + w * ce
            total_weight += w
    return total_loss / total_weight if total_weight > 0 else total_loss


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


def naturalness_penalty_for_ids(token_ids_1d):
    """
    Compute naturalness penalty: average -log P_GPT2(token_i | token_{i-1}) over prefix tokens.
    Lower = more natural (GPT2-fluent).
    Returns a scalar tensor (for combining with CE).
    """
    # Convert Gemma token ids to text, then score with GPT2
    tokens_text = [tokenizer.decode([t.item()]) for t in token_ids_1d]
    # Join to form a string, then encode with GPT2
    prefix_str = "".join(tokens_text)
    gpt2_ids = gpt2_tokenizer(prefix_str, return_tensors="pt")["input_ids"].to(DEVICE)
    if gpt2_ids.shape[1] < 2:
        return torch.tensor(0.0, device=DEVICE)
    with torch.no_grad():
        logits = gpt2_model(gpt2_ids).logits  # [1, T, V_gpt2]
    # CE of each token given previous
    target = gpt2_ids[0, 1:]  # [T-1]
    logits_shifted = logits[0, :-1, :]  # [T-1, V_gpt2]
    ce_per_tok = F.cross_entropy(logits_shifted, target, reduction='none')
    return ce_per_tok.mean()  # average NLL = naturalness penalty


def naturalness_score_for_ids(token_ids_1d):
    """Word-level naturalness (fraction of tokens that are common English words)."""
    tokens = [tokenizer.decode([t.item()]) for t in token_ids_1d]
    return naturalness_score(tokens)


def hotflip_step_with_naturalness(current_ids, ref_ids_set, suffix_texts, ref_completions, lam):
    """
    HotFlip step with naturalness penalty.
    Combined score = CE + lam * naturalness_penalty(token_ids)
    """
    embed_fn = model.get_input_embeddings()
    prefix_embeds = embed_fn(current_ids.unsqueeze(0).to(DEVICE).long()).detach().requires_grad_(True)
    total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    for suffix, ref_comp in zip(suffix_texts, ref_completions):
        full_embeds, comp_start, comp_ids = build_full_input_embeds(prefix_embeds, suffix, ref_comp)
        logits = model(inputs_embeds=full_embeds).logits
        comp_len = comp_ids.shape[1]
        logits_comp = logits[0, comp_start-1:comp_start-1+comp_len, :]
        for i in range(min(comp_len, EARLY_K)):
            ce = F.cross_entropy(logits_comp[i:i+1], comp_ids[0, i:i+1].long())
            total_loss = total_loss + EARLY_WEIGHT * ce
    total_loss.backward()
    grad = prefix_embeds.grad
    emb_matrix = embed_fn.weight

    base_ce = compute_ce_from_ids(current_ids, suffix_texts, ref_completions)
    base_nat = naturalness_penalty_for_ids(current_ids).item()
    best_ids = current_ids.clone()
    best_combined = base_ce + lam * base_nat

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
            trial_nat = naturalness_penalty_for_ids(trial_ids).item()
            trial_combined = trial_ce + lam * trial_nat
            if trial_combined < best_combined:
                best_combined = trial_combined
                best_ids = trial_ids.clone()

    best_ce = compute_ce_from_ids(best_ids, suffix_texts, ref_completions)
    return best_ids, best_ce, best_combined


# ─── Generate reference completions ──────────────────────────────────────────
batch_suffixes = SUFFIXES[:BATCH_SIZE]
ref_ids_set = set(tokenizer(REFERENCE_PREFIX, add_special_tokens=False)["input_ids"])

print(f"\nGenerating reference completions for {REFERENCE_PREFIX!r}...")
ref_completions = []
for suf in batch_suffixes:
    comp = generate_reference_completion(REFERENCE_PREFIX, suf)
    ref_completions.append(comp)
print("Done.")

# ─── Run soft optimization once (shared starting point for all λ) ─────────────
print(f"\nRunning shared soft opt ({N_SOFT_STEPS} steps)...")
embed_fn = model.get_input_embeddings()
embed_dim = embed_fn.weight.shape[1]
with torch.no_grad():
    emb_mean = embed_fn.weight.mean(0)
    emb_std = embed_fn.weight.std(0) * 0.1
torch.manual_seed(42)
soft_prefix_init = (emb_mean.unsqueeze(0).unsqueeze(0).repeat(1, PREFIX_LEN, 1)
                    + torch.randn(1, PREFIX_LEN, embed_dim, device=DEVICE, dtype=DTYPE)
                    * emb_std.unsqueeze(0).unsqueeze(0))
soft_prefix = soft_prefix_init.detach().clone().requires_grad_(True)
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
print(f"Soft CE = {soft_ce:.5f} ({t_soft:.1f}s)")

# Project to discrete tokens (same starting point for all λ)
with torch.no_grad():
    emb_matrix = embed_fn.weight
    soft_norm = F.normalize(soft_prefix[0], dim=-1)
    emb_norm = F.normalize(emb_matrix, dim=-1)
    cos_sim = soft_norm @ emb_norm.T
    projected_ids = cos_sim.argmax(dim=-1)
projected_text = "".join([tokenizer.decode([t.item()]) for t in projected_ids])
proj_ce = compute_ce_from_ids(projected_ids, batch_suffixes, ref_completions)
proj_nat = naturalness_score_for_ids(projected_ids)
print(f"Projected: {projected_text!r} | CE = {proj_ce:.5f} | Naturalness = {proj_nat:.3f}")

# ─── Lambda sweep ─────────────────────────────────────────────────────────────
all_results = {
    "soft_ce": soft_ce,
    "projected_text": projected_text,
    "projection_ce": proj_ce,
    "projection_naturalness": proj_nat,
    "lambda_results": {},
}

for lam in LAMBDA_VALUES:
    print(f"\n{'='*60}")
    print(f"λ = {lam}")
    print(f"{'='*60}")

    current_ids = projected_ids.clone()
    current_ce = proj_ce
    current_combined = proj_ce + lam * naturalness_penalty_for_ids(current_ids).item()
    print(f"Start: CE = {current_ce:.5f} | Naturalness = {naturalness_score_for_ids(current_ids):.3f}")

    hf_log_ce = []
    hf_log_nat = []
    t0 = time.time()
    for step in range(N_HOTFLIP_STEPS):
        new_ids, new_ce, new_combined = hotflip_step_with_naturalness(
            current_ids, ref_ids_set, batch_suffixes, ref_completions, lam
        )
        if new_combined < current_combined:
            current_ids = new_ids
            current_ce = new_ce
            current_combined = new_combined
        nat = naturalness_score_for_ids(current_ids)
        hf_log_ce.append(current_ce)
        hf_log_nat.append(nat)
        if step % 10 == 0 or step == N_HOTFLIP_STEPS - 1:
            toks = "".join([tokenizer.decode([t.item()]) for t in current_ids])
            print(f"  Step {step:3d}: CE = {current_ce:.5f} | Nat = {nat:.3f} | {toks!r}")
    t_hf = time.time() - t0
    final_text = "".join([tokenizer.decode([t.item()]) for t in current_ids])
    final_nat = naturalness_score_for_ids(current_ids)
    print(f"  Final: CE = {current_ce:.5f} | Naturalness = {final_nat:.3f} | {t_hf:.1f}s")
    print(f"  Final prefix: {final_text!r}")

    all_results["lambda_results"][str(lam)] = {
        "lambda": lam,
        "hotflip_ce": current_ce,
        "final_naturalness": final_nat,
        "final_text": final_text,
        "hotflip_ce_log": hf_log_ce,
        "hotflip_nat_log": hf_log_nat,
        "hotflip_seconds": t_hf,
    }

# Summary
print("\n=== SUMMARY ===")
print(f"{'λ':>8} {'Final CE':>10} {'Naturalness':>12} {'Final Prefix'}")
print("-" * 70)
for lam_str, r in all_results["lambda_results"].items():
    print(f"{r['lambda']:>8.1f} {r['hotflip_ce']:>10.5f} {r['final_naturalness']:>12.3f}  {r['final_text']!r}")

results = {
    "experiment": "steer-001-naturalness",
    "model": MODEL_NAME,
    "lm_prior": LM_PRIOR_NAME,
    "reference_prefix": REFERENCE_PREFIX,
    "lambda_values": LAMBDA_VALUES,
    "prefix_len": PREFIX_LEN,
    "n_soft_steps": N_SOFT_STEPS,
    "n_hotflip_steps": N_HOTFLIP_STEPS,
    "batch_size": BATCH_SIZE,
    "results": all_results,
}
out_path = "/home/jovyan/steer001_naturalness.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print("=== DONE ===")
