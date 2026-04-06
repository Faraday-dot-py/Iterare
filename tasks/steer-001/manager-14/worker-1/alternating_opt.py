"""
Experiment 14: Alternating ST+HotFlip Optimization

Hypothesis: HotFlip converges to a local minimum at step 10 in both Exp10 and Exp11.
The minimum is robust to gradient-based perturbation in discrete space. However, by
returning to continuous (soft) space from the HotFlip result and re-applying the ST
estimator, we can explore the neighborhood of the discrete minimum with gradient
information — potentially finding a better basin before descending again.

Algorithm:
  Round 0:
    1. ST soft opt (300 steps) → proj CE ≈ 0.762 (Exp11 result)
    2. HotFlip (80 steps) → final CE ≈ 0.689, converges at step ~10

  Rounds 1..N_ROUNDS-1:
    3. Warm-start soft prefix from best HotFlip discrete embedding
    4. ST soft opt (RESTART_STEPS steps) — gradient from neighborhood of best discrete pt
    5. HotFlip (HOTFLIP_STEPS steps) from new projection
    6. Accept if CE improves, keep best across all rounds

Intuition: HotFlip escapes Voronoi traps by making hard swaps. Re-entering soft space
from the HotFlip result exposes different Voronoi boundaries — the ST gradient pulls
the soft prefix toward a discrete attractor that's better than the current one.

Comparison:
  Exp11 (ST only):  ST opt → HotFlip = 0.689
  Exp13 (random restart):  ST → HotFlip → random perturb → mini-HotFlip
  Exp14 (alternating):  ST → HotFlip → ST (warm) → HotFlip → ...

Config: same as Exp11 (BATCH_SIZE=12, HF_TOPK=30, PLACEHOLDER="SOFTPREFIX", seed=42)
GPU: 0 (CUDA_VISIBLE_DEVICES=0)

Time estimate:
  Round 0: 1335s soft + 340s HotFlip = 1675s
  Rounds 1-4: 450s ST (100 steps @ 4.5s/step) + 340s HotFlip = 790s each
  Total 5 rounds: 1675 + 4*790 = 4835s ≈ 80 min (well within 14400s cap)

Output: /home/jovyan/steer001_alternating_opt.json
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
MODEL_NAME      = "google/gemma-2-2b-it"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE           = torch.bfloat16
REF_PREFIX      = "Talk only about cats."
PLACEHOLDER     = "SOFTPREFIX"
PREFIX_LEN      = 8
SOFT_STEPS      = 300    # initial ST soft opt steps (same as Exp11)
RESTART_STEPS   = 100    # ST soft opt steps for each subsequent round
HOTFLIP_STEPS   = 80     # HotFlip steps per round
HF_TOPK         = 30
BATCH_SIZE      = 12
LR              = 0.01
EARLY_K         = 32
EARLY_WEIGHT    = 3.0
SEED            = 42
N_ROUNDS        = 5      # total rounds (1 initial + 4 restarts)
OUT_PATH        = Path("/home/jovyan/steer001_alternating_opt.json")

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

log("=== Exp 14: Alternating ST+HotFlip Optimization ===")
log(f"CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    log(f"  GPU {i}: {p.name}")

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

def run_hotflip(start_ids, n_steps, ref_completions):
    """Run HotFlip for n_steps from start_ids. Returns (best_ids, best_ce, log, elapsed_s)."""
    current_ids = start_ids.clone().to(EMB_DEV)
    current_ce  = compute_ce_discrete_batched(current_ids, SUFFIXES, ref_completions)
    hf_log = [current_ce]
    t0 = time.time()
    for step in range(n_steps):
        new_ids, new_ce = hotflip_step_batched(current_ids, ref_completions)
        improved = new_ce < current_ce
        if improved:
            current_ids = new_ids
            current_ce  = new_ce
        hf_log.append(current_ce)
        if step % 10 == 0 or step == n_steps - 1:
            toks = tokenizer.decode(current_ids.cpu().tolist())
            log(f"    [{step:3d}/{n_steps}] CE={current_ce:.5f}  {'↓' if improved else '–'}  {toks!r}  {gpu_mem_str()}")
    return current_ids, current_ce, hf_log, time.time() - t0

def run_st_opt(init_emb_LD, n_steps, ref_completions, round_idx, lr=None):
    """Run ST soft opt for n_steps. init_emb_LD is the starting embedding (not trained).

    Tracks the BEST soft prefix (lowest ST-CE) to avoid Voronoi oscillation artifacts:
    the final step may land at a worse projection than an earlier step. Using the best
    observed prefix for projection gives more reproducible results.
    """
    if lr is None:
        lr = LR
    soft_prefix = init_emb_LD.detach().float().requires_grad_(True)
    optimizer = torch.optim.Adam([soft_prefix], lr=lr)
    st_log = []
    best_ce_seen = float('inf')
    best_soft_snapshot = soft_prefix.data.clone()
    t0 = time.time()
    for step in range(n_steps):
        optimizer.zero_grad()
        st_emb, _ = st_project(soft_prefix)
        loss = compute_ce_soft_batched(st_emb, SUFFIXES[:BATCH_SIZE], ref_completions[:BATCH_SIZE])
        loss.backward()
        optimizer.step()
        ce_val = loss.item()
        st_log.append(ce_val)
        # Track best to handle Voronoi oscillation
        if ce_val < best_ce_seen:
            best_ce_seen = ce_val
            best_soft_snapshot = soft_prefix.data.clone()
        if step % 50 == 0 or step == n_steps - 1:
            elapsed = time.time() - t0
            log(f"  [R{round_idx} ST {step:4d}/{n_steps}] ST-CE={ce_val:.5f}  best={best_ce_seen:.5f}  elapsed={elapsed:.0f}s  {gpu_mem_str()}")
    # Return best soft prefix (lowest ST-CE seen), not necessarily the final one
    log(f"  [R{round_idx}] ST opt done. Final ST-CE={ce_val:.5f}, best ST-CE={best_ce_seen:.5f}")
    best_soft = best_soft_snapshot.requires_grad_(False)
    return best_soft, st_log, time.time() - t0

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

# ── Initial soft prefix ──────────────────────────────────────────────────────
with torch.no_grad():
    emb_mean = embed_fn.weight.mean(0)
    emb_std  = embed_fn.weight.std(0) * 0.1
torch.manual_seed(SEED)
random.seed(SEED)
init_noise = (emb_mean.unsqueeze(0).repeat(PREFIX_LEN, 1)
              + torch.randn(PREFIX_LEN, EMB_DIM, device=EMB_DEV, dtype=DTYPE)
              * emb_std.unsqueeze(0))

# ── Alternating optimization loop ────────────────────────────────────────────
best_ids = None
best_ce  = float('inf')

all_rounds = []   # list of {round, st_log, hf_log, proj_ce, hotflip_ce, final_text}

total_t0 = time.time()

for round_idx in range(N_ROUNDS):
    log(f"\n{'='*60}")
    log(f"=== ROUND {round_idx}/{N_ROUNDS-1} ===")
    log(f"{'='*60}")

    # ── ST soft opt ──────────────────────────────────────────────────────────
    n_steps = SOFT_STEPS if round_idx == 0 else RESTART_STEPS
    # Use smaller LR for warm-start rounds to stay near the discrete attractor
    lr = LR if round_idx == 0 else LR * 0.1
    if round_idx == 0:
        init_emb = init_noise.clone()
        log(f"Round 0: ST soft opt from random init ({n_steps} steps, lr={lr})")
    else:
        # Warm-start from the best HotFlip result's discrete embedding
        init_emb = embed_fn(best_ids.to(EMB_DEV)).detach()
        log(f"Round {round_idx}: ST soft opt warm-started from best discrete prefix ({n_steps} steps, lr={lr})")
        log(f"  Best CE so far: {best_ce:.5f}")

    soft_prefix, st_log, t_st = run_st_opt(init_emb, n_steps, ref_completions, round_idx, lr=lr)

    # ── Cosine projection ─────────────────────────────────────────────────────
    projected_ids  = project_to_tokens(soft_prefix.detach().to(DTYPE))
    projected_text = tokenizer.decode(projected_ids.cpu().tolist())
    proj_ce = compute_ce_discrete_batched(projected_ids, SUFFIXES, ref_completions)
    log(f"  Projected: {projected_text!r}  CE={proj_ce:.5f}")

    # ── HotFlip refinement ────────────────────────────────────────────────────
    log(f"  HotFlip ({HOTFLIP_STEPS} steps)...")
    hf_ids, hf_ce, hf_log, t_hf = run_hotflip(projected_ids, HOTFLIP_STEPS, ref_completions)
    final_text = tokenizer.decode(hf_ids.cpu().tolist())
    log(f"  HotFlip done. CE={hf_ce:.5f}  time={t_hf:.1f}s")
    log(f"  Final prefix: {final_text!r}")

    # ── Update best ───────────────────────────────────────────────────────────
    if hf_ce < best_ce:
        best_ce  = hf_ce
        best_ids = hf_ids.clone()
        log(f"  *** New best! CE={best_ce:.5f} ***")
    else:
        log(f"  No improvement (best remains {best_ce:.5f})")

    all_rounds.append({
        "round": round_idx,
        "n_st_steps": n_steps,
        "projection_ce": proj_ce,
        "projected_text": projected_text,
        "hotflip_ce": hf_ce,
        "final_text": final_text,
        "final_ids": hf_ids.tolist(),
        "is_new_best": hf_ce < best_ce + 1e-8,  # approximate equality check
        "st_log": st_log,
        "hf_log": hf_log,
        "timing": {"st_seconds": t_st, "hf_seconds": t_hf},
    })

total_elapsed = time.time() - total_t0
best_text = tokenizer.decode(best_ids.cpu().tolist())

log(f"\n{'='*60}")
log(f"=== FINAL SUMMARY ===")
log(f"{'='*60}")
log(f"  Exp1  (standard):       proj=1.436, hotflip=0.740")
log(f"  Exp11 (ST only):        proj=0.762, hotflip=0.689")
log(f"  Exp14 (alternating):    best hotflip CE={best_ce:.5f}")
log(f"  Best prefix: {best_text!r}")
log(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/3600:.2f}h)")

results = {
    "experiment": "steer-001-exp14-alternating-opt",
    "model": MODEL_NAME,
    "reference_prefix": REF_PREFIX,
    "seed": SEED,
    "prefix_len": PREFIX_LEN,
    "n_initial_soft_steps": SOFT_STEPS,
    "n_restart_soft_steps": RESTART_STEPS,
    "n_hotflip_steps": HOTFLIP_STEPS,
    "n_rounds": N_ROUNDS,
    "batch_size": BATCH_SIZE,
    "hf_topk": HF_TOPK,
    "method": "alternating_st_hotflip",
    "metrics": {
        "best_hotflip_ce": best_ce,
        "best_round": next(r["round"] for r in all_rounds if abs(r["hotflip_ce"] - best_ce) < 1e-8),
        "exp11_baseline_proj": 0.76245,
        "exp11_baseline_hotflip": 0.68935,
        "exp1_baseline_hotflip": 0.7399,
    },
    "best_text": best_text,
    "best_ids": best_ids.tolist(),
    "rounds": all_rounds,
    "total_elapsed_seconds": total_elapsed,
}
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
log(f"\nResults saved to {OUT_PATH}")
log("=== DONE ===")
