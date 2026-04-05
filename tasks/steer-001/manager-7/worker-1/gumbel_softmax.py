"""
Experiment 7: Gumbel-Softmax Discrete Training
Can end-to-end discrete-aware optimization outperform the two-stage soft→project pipeline?

Standard pipeline bottleneck:
  soft opt (continuous) → cosine projection → HotFlip repair
  The projection step loses information because the soft prefix is optimized
  without any awareness of the discrete token constraint.

This experiment:
  Parameterize the prefix as logits L ∈ R^{PREFIX_LEN × V_vocab}.
  At each step: sample using Gumbel-Softmax with temperature τ.
  Embedding = softmax_τ(L + Gumbel_noise) @ W_embed
  As τ → 0, converges to hard argmax (one-hot).
  Train with CE loss and anneal τ from 1.0 → 0.1 over 400 steps.
  At the end, take the argmax of L as the discrete prefix.

Hypothesis: this produces lower CE than the two-stage approach because the
optimization "knows" it needs to end up at a discrete token.

Comparison:
- Standard pipeline CE (from Exp 1 baseline): soft=0.19, proj=1.44, hotflip=0.74
- Gumbel target: achieve < 0.74 CE with discrete prefix directly

Output: /home/jovyan/steer001_gumbel.json
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
import json
import time
import math

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.import_utils import is_torchvision_available
if hasattr(is_torchvision_available, "cache_clear"):
    is_torchvision_available.cache_clear()

print("=== Experiment 7: Gumbel-Softmax Discrete Training ===")
print(f"CUDA: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

REFERENCE_PREFIX = "Talk only about cats."
PREFIX_LEN = 8
N_STEPS = 500             # Gumbel-Softmax training steps
TEMP_INIT = 2.0           # Initial Gumbel temperature
TEMP_FINAL = 0.1          # Final temperature (near-discrete)
N_HOTFLIP_STEPS = 50      # HotFlip refinement after Gumbel training
BATCH_SIZE = 8
LR_LOGITS = 0.05          # Higher LR than soft opt (logits space is sparser)
EARLY_K = 32
EARLY_WEIGHT = 3.0
N_GUMBEL_SAMPLES = 1      # MC samples per step (1 = single-sample REINFORCE-like)

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
V = tokenizer.vocab_size  # actual vocab size

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


def gumbel_softmax_prefix_embeds(logits, tau, embed_fn, hard=False):
    """
    Sample a soft-discrete prefix embedding using Gumbel-Softmax.

    logits: [PREFIX_LEN, V] — unnormalized log-probs over vocab
    tau: temperature (lower → more discrete)
    Returns: [1, PREFIX_LEN, D] embedding in model's embed space

    With hard=True, uses straight-through estimator for discrete tokens.
    """
    # Gumbel-Softmax sample: [PREFIX_LEN, V]
    gs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)  # [PREFIX_LEN, V]
    # Map to embedding space: gs @ W_embed
    # W_embed: [V, D]  (model may split across devices, but embed_fn.weight is on GPU)
    W = embed_fn.weight.float()  # [V, D], float32 for stable grad
    embeds = (gs.float() @ W).to(DTYPE)  # [PREFIX_LEN, D]
    return embeds.unsqueeze(0)  # [1, PREFIX_LEN, D]


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
    # Accumulate gradients one suffix at a time to avoid OOM
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


# ─── Generate reference completions ──────────────────────────────────────────
batch_suffixes = SUFFIXES[:BATCH_SIZE]
ref_ids_set = set(tokenizer(REFERENCE_PREFIX, add_special_tokens=False)["input_ids"])

print(f"\nGenerating reference completions for {REFERENCE_PREFIX!r}...")
ref_completions = []
for suf in batch_suffixes:
    comp = generate_reference_completion(REFERENCE_PREFIX, suf)
    ref_completions.append(comp)
print("Done.")

# ─── Gumbel-Softmax Training ─────────────────────────────────────────────────
embed_fn = model.get_input_embeddings()
V_actual = embed_fn.weight.shape[0]
print(f"Vocab size: {V_actual}")

# Initialize logits: small random values (near-uniform distribution initially)
torch.manual_seed(42)
logits = torch.randn(PREFIX_LEN, V_actual, device=DEVICE, dtype=torch.float32) * 0.01
logits.requires_grad_(True)
optimizer = torch.optim.Adam([logits], lr=LR_LOGITS)

# Temperature schedule: exponential decay from TEMP_INIT to TEMP_FINAL
def get_tau(step):
    return TEMP_INIT * (TEMP_FINAL / TEMP_INIT) ** (step / N_STEPS)

gumbel_log = []
t0 = time.time()

print(f"\nGumbel-Softmax training ({N_STEPS} steps, τ: {TEMP_INIT}→{TEMP_FINAL})...")
for step in range(N_STEPS):
    optimizer.zero_grad()
    tau = get_tau(step)

    # Get Gumbel-Softmax prefix embeddings
    prefix_embeds = gumbel_softmax_prefix_embeds(logits, tau, embed_fn, hard=False)
    # prefix_embeds: [1, PREFIX_LEN, D]

    # Compute CE loss, accumulating per suffix to avoid OOM
    total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    total_weight = 0.0
    for suffix, ref_comp in zip(batch_suffixes, ref_completions):
        full_embeds, comp_start, comp_ids = build_full_input_embeds(prefix_embeds, suffix, ref_comp)
        model_logits = model(inputs_embeds=full_embeds).logits
        comp_len = comp_ids.shape[1]
        logits_comp = model_logits[0, comp_start-1:comp_start-1+comp_len, :]
        suffix_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        for i in range(comp_len):
            w = EARLY_WEIGHT if i < EARLY_K else 1.0
            ce = F.cross_entropy(logits_comp[i:i+1], comp_ids[0, i:i+1].long())
            suffix_loss = suffix_loss + w * ce
            total_weight += w
        total_loss = total_loss + suffix_loss
        del model_logits, full_embeds

    loss = total_loss / total_weight
    loss.backward()
    optimizer.step()
    gumbel_log.append(loss.item())

    if step % 50 == 0 or step == N_STEPS - 1:
        # Show current argmax prefix
        with torch.no_grad():
            current_ids = logits.argmax(dim=-1)
        current_text = "".join([tokenizer.decode([t.item()]) for t in current_ids])
        print(f"  Step {step:4d}: CE={loss.item():.5f} τ={tau:.4f} | {current_text!r}")

t_gumbel = time.time() - t0
print(f"Gumbel training done in {t_gumbel:.1f}s")

# Extract discrete prefix (argmax of trained logits)
with torch.no_grad():
    gumbel_ids = logits.argmax(dim=-1)
gumbel_text = "".join([tokenizer.decode([t.item()]) for t in gumbel_ids])
gumbel_ce = compute_ce_from_ids(gumbel_ids, batch_suffixes, ref_completions)
print(f"Gumbel argmax prefix: {gumbel_text!r} | CE = {gumbel_ce:.5f}")

# Compare: cosine projection baseline (same as Exp 1)
# Project the Gumbel embeddings using cosine similarity at end of training
with torch.no_grad():
    # Use the soft-Gumbel embedding (τ=TEMP_FINAL) as the "projection"
    tau_final = TEMP_FINAL
    gs_soft = F.softmax(logits / tau_final, dim=-1)  # [PREFIX_LEN, V]
    W = embed_fn.weight.float()
    soft_embeds = (gs_soft @ W).to(DTYPE).unsqueeze(0)  # [1, PREFIX_LEN, D]
    # Cosine project
    soft_norm = F.normalize(soft_embeds[0], dim=-1)
    emb_norm = F.normalize(embed_fn.weight, dim=-1)
    cos_ids = (soft_norm @ emb_norm.T).argmax(dim=-1)
cos_text = "".join([tokenizer.decode([t.item()]) for t in cos_ids])
cos_ce = compute_ce_from_ids(cos_ids, batch_suffixes, ref_completions)
print(f"Cosine proj of Gumbel: {cos_text!r} | CE = {cos_ce:.5f}")

# HotFlip refinement from Gumbel argmax
print(f"\nHotFlip ({N_HOTFLIP_STEPS} steps) from Gumbel argmax...")
current_ids = gumbel_ids.clone()
current_ce = gumbel_ce
hf_log = [current_ce]
t0 = time.time()
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
print(f"HotFlip CE = {current_ce:.5f} ({t_hf:.1f}s)")

# Summary
print("\n=== SUMMARY ===")
print(f"  Exp 1 baseline:        soft=0.191, proj=1.436, hotflip=0.740")
print(f"  Gumbel training:       argmax CE={gumbel_ce:.5f} (τ={TEMP_INIT}→{TEMP_FINAL}, {N_STEPS} steps)")
print(f"  Gumbel cosine proj:    CE={cos_ce:.5f}")
print(f"  Gumbel + HotFlip:      CE={current_ce:.5f}")
print(f"  Gumbel argmax prefix:  {gumbel_text!r}")
print(f"  Final prefix:          {final_text!r}")

results = {
    "experiment": "steer-001-gumbel",
    "model": MODEL_NAME,
    "reference_prefix": REFERENCE_PREFIX,
    "prefix_len": PREFIX_LEN,
    "n_gumbel_steps": N_STEPS,
    "temp_init": TEMP_INIT,
    "temp_final": TEMP_FINAL,
    "n_hotflip_steps": N_HOTFLIP_STEPS,
    "lr_logits": LR_LOGITS,
    "batch_size": BATCH_SIZE,
    "metrics": {
        "gumbel_argmax_ce": gumbel_ce,
        "gumbel_cosine_proj_ce": cos_ce,
        "hotflip_ce_from_gumbel": current_ce,
        "exp1_baseline_hotflip_ce": 0.7399094104766846,  # from Exp 1 results
        "improvement_over_baseline": 0.7399094104766846 - current_ce,
    },
    "gumbel_argmax_text": gumbel_text,
    "gumbel_argmax_ids": gumbel_ids.tolist(),
    "gumbel_cosine_proj_text": cos_text,
    "gumbel_cosine_proj_ids": cos_ids.tolist(),
    "final_text": final_text,
    "final_ids": current_ids.tolist(),
    "gumbel_log": gumbel_log,
    "gumbel_log_every10": gumbel_log[::10],
    "hotflip_log": hf_log,
    "timing": {
        "gumbel_seconds": t_gumbel,
        "hotflip_seconds": t_hf,
    },
}
out_path = "/home/jovyan/steer001_gumbel.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print("=== DONE ===")
