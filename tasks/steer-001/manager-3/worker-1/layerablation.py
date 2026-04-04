"""
Experiment 3: Layer-depth projection ablation
Does projecting at earlier layers yield better CE after HotFlip?

Method:
- Use "Talk only about cats." as reference
- Train soft prefix targeting INTERNAL STATE at layers L in {4, 10, 16, 22}
  (Gemma-2-2B has 26 layers, 0-indexed)
- For each: project → 50 HotFlip steps → measure CE
- Record naturalness heuristic (fraction of projected tokens that are standard ASCII words)

Output: /home/jovyan/steer001_layerablation.json
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

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.import_utils import is_torchvision_available
if hasattr(is_torchvision_available, "cache_clear"):
    is_torchvision_available.cache_clear()
import torch.nn.functional as F

print("=== Experiment 3: Layer-Depth Projection Ablation ===")
print(f"CUDA: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

REFERENCE_PREFIX = "Talk only about cats."
TARGET_LAYERS = [4, 10, 16, 22]  # Gemma-2-2B has 26 layers (0-indexed)

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

# Simple English word list for naturalness scoring (common words)
COMMON_ENGLISH = set("""
a an the is are was were be been being am have has had do does did will would could should may
might shall can cannot not no yes i you he she it we they me him her us them my your his its our
their this that these those what which who whom whose when where why how all some any few more
most other into over after about from up out on off through with without as at by for to in of
and but or so yet nor if then because since until unless while although though even just only also
very really quite about around than before after between both each every many much such into
about above across along always another around away back been between between both both but call
came come could day did different down each end even every far few find first food for found
from get give go good great had hand hard has have he help here high him his home how i if in
it just keep kind know large last left let like line little long look made make many may me mean
more most move much my name need never new night no not now number of off often old on once one
only open or other our out over own part place play put read right same saw say see seem sentence
set should show side small so some something sometimes still such take tell than their then there
these they thing think this those through time to together too turn under until up use very want
was watch way we went were what when where which while who will with word work world would write
year you young
talk about cats only speak always like formal academic language respond using numbered lists
pirate""".split())


def naturalness_score(token_list):
    """Fraction of tokens that look like common English words."""
    count = 0
    for tok in token_list:
        tok_clean = tok.strip().lower()
        tok_clean = re.sub(r'[^a-z]', '', tok_clean)
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
print(f"Model loaded in {time.time()-t0:.1f}s, device: {next(model.parameters()).device}")
print(f"Number of layers: {model.config.num_hidden_layers}")

# ─── Helper functions ─────────────────────────────────────────────────────────

def generate_reference_completion(prefix_text, suffix_text, max_new=80):
    prompt = f"<start_of_turn>user\n{prefix_text}\n\n{suffix_text}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, temperature=1.0)
    return out[0][inputs["input_ids"].shape[1]:]


def get_hidden_state_at_layer(text_ids, layer_idx):
    """Get hidden states at a specific layer for the given token ids."""
    with torch.no_grad():
        output = model(text_ids, output_hidden_states=True)
    # hidden_states[0] = embedding, hidden_states[i+1] = after layer i
    return output.hidden_states[layer_idx + 1]  # [1, seq_len, D]


def get_hidden_state_at_layer_with_soft(soft_prefix_embeds, suffix_text, layer_idx):
    """
    Get hidden state at a layer when using soft prefix embeddings.
    For state-matching objective.
    """
    embed_fn = model.get_input_embeddings()
    pre_text = "<start_of_turn>user\n"
    mid_text = "\n\n" + suffix_text + "<end_of_turn>\n<start_of_turn>model\n"
    pre_ids = tokenizer(pre_text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(DEVICE)
    mid_ids = tokenizer(mid_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    pre_embeds = embed_fn(pre_ids)
    mid_embeds = embed_fn(mid_ids)
    full_embeds = torch.cat([pre_embeds, soft_prefix_embeds, mid_embeds], dim=1)
    output = model(inputs_embeds=full_embeds, output_hidden_states=True)
    return output.hidden_states[layer_idx + 1]  # [1, seq_len, D]


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


def compute_state_match_loss(soft_prefix_embeds, ref_states_per_suffix, target_layer, suffix_texts):
    """
    Loss = mean negative cosine similarity between soft-prefix hidden states and reference hidden states,
    measured at target_layer, averaged over prefix token positions.
    """
    total_loss = torch.tensor(0.0, device=DEVICE)
    for suffix, ref_state in zip(suffix_texts, ref_states_per_suffix):
        soft_state = get_hidden_state_at_layer_with_soft(soft_prefix_embeds, suffix, target_layer)
        # Compare at prefix positions (positions pre_len : pre_len + PREFIX_LEN)
        pre_text = "<start_of_turn>user\n"
        pre_len = tokenizer(pre_text, return_tensors="pt", add_special_tokens=True)["input_ids"].shape[1]
        soft_at_prefix = soft_state[0, pre_len:pre_len+PREFIX_LEN, :]   # [PREFIX_LEN, D]
        ref_at_prefix = ref_state[0, pre_len:pre_len+PREFIX_LEN, :]      # [PREFIX_LEN, D]
        cos_sim = F.cosine_similarity(soft_at_prefix, ref_at_prefix, dim=-1).mean()
        total_loss = total_loss + (1.0 - cos_sim)
    return total_loss / len(suffix_texts)


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


# ─── Generate reference completions + hidden states ──────────────────────────
batch_suffixes = SUFFIXES[:BATCH_SIZE]
ref_ids_set = set(tokenizer(REFERENCE_PREFIX, add_special_tokens=False)["input_ids"])

print(f"\nGenerating reference completions for {REFERENCE_PREFIX!r}...")
ref_completions = []
for suf in batch_suffixes:
    comp = generate_reference_completion(REFERENCE_PREFIX, suf)
    ref_completions.append(comp)
print("Done.")

# For state matching, get reference hidden states for each target layer
print("Computing reference hidden states at each target layer...")
# Build reference prompt ids (with the full prefix text, not soft)
embed_fn = model.get_input_embeddings()
ref_prefix_ids = tokenizer(REFERENCE_PREFIX, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
ref_prefix_embeds = embed_fn(ref_prefix_ids)  # [1, ref_prefix_len, D]

ref_hidden_states_by_layer = {}
for layer_idx in TARGET_LAYERS:
    states = []
    for suf in batch_suffixes:
        state = get_hidden_state_at_layer_with_soft(ref_prefix_embeds, suf, layer_idx)
        states.append(state.detach())
    ref_hidden_states_by_layer[layer_idx] = states
    print(f"  Layer {layer_idx}: done")

# ─── Main experiment: for each target layer ───────────────────────────────────
all_results = {}

for layer_idx in TARGET_LAYERS:
    print(f"\n{'='*60}")
    print(f"TARGET LAYER {layer_idx}")
    print(f"{'='*60}")

    ref_states = ref_hidden_states_by_layer[layer_idx]

    # Soft prefix optimization — state matching objective
    print(f"Soft opt (state matching at layer {layer_idx}, {N_SOFT_STEPS} steps)...")
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
        loss = compute_state_match_loss(soft_prefix, ref_states, layer_idx, batch_suffixes)
        loss.backward()
        optimizer.step()
        log_soft.append(loss.item())
        if step % 100 == 0 or step == N_SOFT_STEPS - 1:
            print(f"  Step {step:4d}: state-match loss = {loss.item():.5f}")
    t_soft = time.time() - t0
    soft_state_loss = log_soft[-1]

    # Measure CE (behavior) after soft opt
    soft_ce = 0.0  # Skip for speed — state matching doesn't optimize CE directly
    print(f"  State-match loss = {soft_state_loss:.5f} ({t_soft:.1f}s)")

    # Cosine projection
    with torch.no_grad():
        emb_matrix = embed_fn.weight
        soft_norm = F.normalize(soft_prefix[0], dim=-1)
        emb_norm = F.normalize(emb_matrix, dim=-1)
        cos_sim = soft_norm @ emb_norm.T
        projected_ids = cos_sim.argmax(dim=-1)
    projected_tokens = [tokenizer.decode([t.item()]) for t in projected_ids]
    projected_text = "".join(projected_tokens)
    print(f"  Projected: {projected_text!r}")

    # Naturalness score
    nat_score = naturalness_score(projected_tokens)
    print(f"  Naturalness score: {nat_score:.3f}")

    # CE after projection
    proj_ce = compute_ce_from_ids(projected_ids, batch_suffixes, ref_completions)
    print(f"  CE after projection: {proj_ce:.5f}")

    # HotFlip refinement
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
    final_tokens = [tokenizer.decode([t.item()]) for t in current_ids]
    final_nat_score = naturalness_score(final_tokens)
    print(f"  HotFlip CE = {current_ce:.5f} ({t_hf:.1f}s)")
    print(f"  Final prefix: {final_text!r}")
    print(f"  Final naturalness: {final_nat_score:.3f}")

    all_results[layer_idx] = {
        "target_layer": layer_idx,
        "soft_state_match_loss": soft_state_loss,
        "projection_ce": proj_ce,
        "projection_naturalness": nat_score,
        "hotflip_ce": current_ce,
        "final_naturalness": final_nat_score,
        "projected_text": projected_text,
        "final_text": final_text,
        "soft_log_every10": log_soft[::10],
        "hotflip_log": hf_log,
        "timing": {"soft_seconds": t_soft, "hotflip_seconds": t_hf},
    }

# Summary
print("\n=== SUMMARY ===")
print(f"{'Layer':>6} {'Proj CE':>10} {'HotFlip CE':>12} {'Proj Nat':>10} {'Final Nat':>10} {'Final Prefix'}")
print("-" * 80)
for l, r in all_results.items():
    print(f"{l:>6} {r['projection_ce']:>10.5f} {r['hotflip_ce']:>12.5f} {r['projection_naturalness']:>10.3f} {r['final_naturalness']:>10.3f}  {r['final_text']!r}")

results = {
    "experiment": "steer-001-layerablation",
    "model": MODEL_NAME,
    "reference_prefix": REFERENCE_PREFIX,
    "target_layers": TARGET_LAYERS,
    "prefix_len": PREFIX_LEN,
    "n_soft_steps": N_SOFT_STEPS,
    "n_hotflip_steps": N_HOTFLIP_STEPS,
    "batch_size": BATCH_SIZE,
    "results_by_layer": all_results,
}
out_path = "/home/jovyan/steer001_layerablation.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print("=== DONE ===")
