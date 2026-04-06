"""
Experiment 13: Extended HotFlip with Random Restarts

Exp 10 and Exp 11 both converge at HotFlip step 10/80 and cannot improve further.
This suggests the greedy HotFlip search hits a local minimum in discrete space.
Both pipelines waste ~70 HotFlip steps producing no improvement.

This experiment uses the ST estimator starting point (best proj CE=0.762) and
tries to escape the local minimum using random-restart HotFlip:

1. Run standard HotFlip until convergence (≤80 steps)
2. From the converged prefix, randomly perturb one position
3. Run a mini-HotFlip (30 steps) from the perturbed prefix
4. Accept if CE improves, discard otherwise
5. Repeat for 10 random restart attempts

This adds ~30*34 = 1020s per restart × 10 restarts = 10200s extra, plus
initial HotFlip ~340s = ~10540s total for HotFlip. With soft opt 1335s,
total ~11875s. Within 14400s cap.

Starting from ST prefix (proj CE=0.762) gives the best starting point.

Output: /home/jovyan/steer001_hotflip_extended.json
"""

import sys, importlib.util as _ilu

_real_find_spec = _ilu.find_spec
def _patched(name, package=None, target=None):
    return None if name == "torchvision" else _real_find_spec(name, package)
_ilu.find_spec = _patched
for _k in list(sys.modules):
    if "torchvision" in _k: del sys.modules[_k]

import json, time, random
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
SOFT_STEPS    = 300
HOTFLIP_STEPS = 80    # initial HotFlip
HF_TOPK       = 30
BATCH_SIZE    = 12
LR            = 0.01
EARLY_K       = 32
EARLY_WEIGHT  = 3.0
SEED          = 42
N_RESTARTS    = 10    # number of random restart attempts
RESTART_STEPS = 30    # HotFlip steps per restart
OUT_PATH      = Path("/home/jovyan/steer001_hotflip_extended.json")

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

log("=== Exp 13: Extended HotFlip with Random Restarts ===")
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

# ── Helpers ──────────────────────────────────────────────────────────────────

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
    return torch.cat(padded, dim=0).to(EMB_DEV), meta, T_max

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

def compute_ce_soft_batched(prefix_LD, suffix_texts, ref_completions):
    batch_emb, meta, T_max = build_batch(prefix_LD, suffix_texts, ref_completions)
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

def st_project(soft_prefix_LD):
    W = embed_fn.weight.to(device=soft_prefix_LD.device, dtype=soft_prefix_LD.dtype)
    sn   = F.normalize(soft_prefix_LD, dim=-1)
    wn   = F.normalize(W, dim=-1)
    sims = sn @ wn.T
    ids  = sims.argmax(dim=-1)
    with torch.no_grad():
        proj_emb = embed_fn(ids).to(dtype=soft_prefix_LD.dtype)
    st_emb = soft_prefix_LD + (proj_emb - soft_prefix_LD).detach()
    return st_emb, ids

def hotflip_step_batched(current_ids, ref_completions):
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

def run_hotflip(start_ids, n_steps, label=""):
    """Run HotFlip for n_steps from start_ids. Returns (best_ids, best_ce, log)."""
    current_ids = start_ids.clone().to(EMB_DEV)
    current_ce  = compute_ce_discrete_batched(current_ids, SUFFIXES, ref_completions)
    log_hf = [current_ce]
    t0 = time.time()
    for step in range(n_steps):
        new_ids, new_ce = hotflip_step_batched(current_ids, ref_completions)
        improved = new_ce < current_ce
        if improved:
            current_ids = new_ids
            current_ce  = new_ce
        log_hf.append(current_ce)
        if step % 10 == 0 or step == n_steps - 1:
            toks = tokenizer.decode(current_ids.cpu().tolist())
            log(f"  {label}[{step:3d}/{n_steps}] CE={current_ce:.5f}  {'↓' if improved else '–'}  {toks!r}")
    return current_ids, current_ce, log_hf, time.time() - t0

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
log("Reference completions ready.")

ref_ids_set = set(tokenizer.encode(REF_PREFIX, add_special_tokens=False))

# ── ST soft opt ───────────────────────────────────────────────────────────────
log(f"\n=== ST Soft Opt (seed={SEED}) ===")
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
    st_emb, _ = st_project(soft_prefix)
    loss = compute_ce_soft_batched(st_emb, SUFFIXES[:BATCH_SIZE], ref_completions[:BATCH_SIZE])
    loss.backward()
    optimizer.step()
    log_soft.append(loss.item())
    if step % 50 == 0 or step == SOFT_STEPS - 1:
        elapsed = time.time() - t0
        log(f"  [{step:4d}/{SOFT_STEPS}] ST-CE={loss.item():.5f}  elapsed={elapsed:.0f}s  {gpu_mem_str()}")

t_soft = time.time() - t0
soft_ce = log_soft[-1]
log(f"ST soft opt done. Final ST-CE={soft_ce:.5f}  time={t_soft:.1f}s")

# ── Cosine projection ─────────────────────────────────────────────────────────
projected_ids  = project_to_tokens(soft_prefix.detach().to(DTYPE))
projected_text = tokenizer.decode(projected_ids.cpu().tolist())
proj_ce = compute_ce_discrete_batched(projected_ids, SUFFIXES, ref_completions)
log(f"Projected: {projected_text!r} | CE = {proj_ce:.5f}")

# ── Initial HotFlip ──────────────────────────────────────────────────────────
log(f"\n=== Initial HotFlip ({HOTFLIP_STEPS} steps) ===")
best_ids, best_ce, initial_hf_log, t_hf = run_hotflip(projected_ids, HOTFLIP_STEPS, label="HF ")
log(f"Initial HotFlip done. CE={best_ce:.5f}  time={t_hf:.1f}s")
log(f"Initial best prefix: {tokenizer.decode(best_ids.cpu().tolist())!r}")

# ── Random restart HotFlip ───────────────────────────────────────────────────
log(f"\n=== Random Restart HotFlip ({N_RESTARTS} restarts, {RESTART_STEPS} steps each) ===")
log(f"Best so far: {best_ce:.5f}")

restart_log = []
torch.manual_seed(SEED + 100)  # separate seed for restart perturbations
t_restart_total = time.time()

for restart_idx in range(N_RESTARTS):
    # Perturb: randomly replace one position with a random non-banned token
    perturbed = best_ids.clone()
    pos = random.randint(0, PREFIX_LEN - 1)

    # Sample a random token that's not banned
    while True:
        rand_tok = torch.randint(0, VOCAB, (1,)).item()
        if (rand_tok not in ref_ids_set and
            rand_tok != tokenizer.bos_token_id and
            rand_tok != tokenizer.eos_token_id and
            (tokenizer.pad_token_id is None or rand_tok != tokenizer.pad_token_id)):
            break
    perturbed[pos] = rand_tok

    perturbed_text = tokenizer.decode(perturbed.cpu().tolist())
    perturbed_ce = compute_ce_discrete_batched(perturbed, SUFFIXES, ref_completions)
    log(f"\n  Restart {restart_idx+1}/{N_RESTARTS}: perturb pos {pos} → "
        f"'{tokenizer.decode([rand_tok])}' | initial CE={perturbed_ce:.5f}")
    log(f"  Perturbed: {perturbed_text!r}")

    # Run mini-HotFlip from perturbed prefix
    new_ids, new_ce, restart_hf_log, t_r = run_hotflip(
        perturbed, RESTART_STEPS, label=f"R{restart_idx+1} "
    )

    restart_result = {
        "restart_idx": restart_idx,
        "perturb_pos": pos,
        "perturb_token": rand_tok,
        "perturb_token_str": tokenizer.decode([rand_tok]),
        "initial_ce": perturbed_ce,
        "final_ce": new_ce,
        "accepted": new_ce < best_ce,
        "log": restart_hf_log,
    }
    restart_log.append(restart_result)

    if new_ce < best_ce:
        log(f"  *** Improved! {best_ce:.5f} → {new_ce:.5f} ***")
        best_ce = new_ce
        best_ids = new_ids.clone()
        log(f"  New best: {tokenizer.decode(best_ids.cpu().tolist())!r}")
    else:
        log(f"  No improvement ({new_ce:.5f} ≥ {best_ce:.5f})")

t_restart = time.time() - t_restart_total
final_text = tokenizer.decode(best_ids.cpu().tolist())
log(f"\nRandom restart HotFlip done. Best CE={best_ce:.5f}  time={t_restart:.1f}s")

# ── Summary ──────────────────────────────────────────────────────────────────
log("\n=== SUMMARY ===")
log(f"  Exp11 (ST, standard HotFlip):     proj=0.762, hotflip=0.689")
log(f"  This run (ST + restart HotFlip):  proj={proj_ce:.3f}, hotflip={best_ce:.3f}")
log(f"  Final prefix: {final_text!r}")
n_accepted = sum(r["accepted"] for r in restart_log)
log(f"  Restarts: {n_accepted}/{N_RESTARTS} improved")

results = {
    "experiment": "steer-001-exp13-hotflip-restarts",
    "model": MODEL_NAME,
    "reference_prefix": REF_PREFIX,
    "seed": SEED,
    "prefix_len": PREFIX_LEN,
    "n_soft_steps": SOFT_STEPS,
    "n_hotflip_steps": HOTFLIP_STEPS,
    "n_restarts": N_RESTARTS,
    "restart_steps": RESTART_STEPS,
    "batch_size": BATCH_SIZE,
    "hf_topk": HF_TOPK,
    "method": "st_estimator_plus_restart_hotflip",
    "metrics": {
        "st_ce_final": soft_ce,
        "projection_ce": proj_ce,
        "initial_hotflip_ce": initial_hf_log[-1],
        "final_hotflip_ce": best_ce,
        "restarts_accepted": n_accepted,
        "exp11_baseline_proj": 0.762,
        "exp11_baseline_hotflip": 0.689,
        "exp10_standard_hotflip": 0.740,
    },
    "projected_text": projected_text,
    "final_text": final_text,
    "final_ids": best_ids.tolist(),
    "soft_log": log_soft,
    "initial_hf_log": initial_hf_log,
    "restart_log": restart_log,
    "timing": {"soft_seconds": t_soft, "hotflip_seconds": t_hf, "restart_seconds": t_restart},
}
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
log(f"\nResults saved to {OUT_PATH}")
log("=== DONE ===")
