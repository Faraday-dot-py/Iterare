"""
Experiment 22: VQ-Style Commitment Loss

Motivation (from research report):
The report recommends replacing the soft-to-discrete gap framing with a
"quantization-aware" training approach inspired by VQ-VAE. The key idea:
add an explicit regularizer that penalizes the soft prefix for being far
from its nearest token embedding — the "commitment loss."

In VQ-VAE, the commitment loss is:
    L_commit = ||z_e - stop_grad(e_k)||²

Adapted to our setting (cosine distance for scale stability):
    L_commit = mean_i(1 - cos(soft_i, stop_grad(embed(argmax_i))))

This differs from Voronoi margin regularization (Exp16/21):
  - Margin: maximize (cos(soft, nearest) - cos(soft, 2nd_nearest)) — relative boundary
  - Commitment: maximize cos(soft, nearest) directly — absolute distance to nearest token

The commitment loss is convex in the direction of nearest embedding, making it a
smoother optimization target than the relative margin.

Loss = CE(ST-project(soft)) + β * mean_i(1 - cos(soft_i, embed(argmax_i)))

Three β values tested: 0.1 (weak), 0.5 (moderate), 2.0 (strong).
β=0.0 is Exp11/16 baseline (not re-run, results known: CE=0.686-0.689).

Config: PREFIX_LEN=8, BATCH_SIZE=12, HF_TOPK=30, SOFT_STEPS=300,
        HOTFLIP_STEPS=80, seed=42, constant LR=0.01

Output: /home/jovyan/steer001_vq_commitment.json
"""

import sys, importlib.util as _ilu

_real_find_spec = _ilu.find_spec
def _patched(name, package=None, target=None):
    return None if name == "torchvision" else _real_find_spec(name, package)
_ilu.find_spec = _patched
for _k in list(sys.modules):
    if "torchvision" in _k: del sys.modules[_k]

import json, os as _os, time
from pathlib import Path

import os as _os, json as _json, urllib.request as _urlreq

def notify(title, body=""):
    """Pushbullet push. No-ops silently if PUSHBULLET_API_KEY not set."""
    key = _os.environ.get("PUSHBULLET_API_KEY", "")
    if not key:
        return
    try:
        data = _json.dumps({"type": "note", "title": title, "body": body}).encode()
        req = _urlreq.Request(
            "https://api.pushbullet.com/v2/pushes", data=data, method="POST",
            headers={"Access-Token": key, "Content-Type": "application/json"},
        )
        _urlreq.urlopen(req, timeout=5)
    except Exception:
        pass

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
HOTFLIP_STEPS = 80
HF_TOPK       = 30
BATCH_SIZE    = 12
LR            = 0.01
EARLY_K       = 32
EARLY_WEIGHT  = 3.0
SEED          = 42
BETAS         = [0.1, 0.5, 2.0]  # commitment loss weights
OUT_PATH      = Path("/home/jovyan/steer001_vq_commitment.json")
CKPT_PATH     = Path("/home/jovyan/steer001_vq_commitment_ckpt.json")

EXP19_SOTA_CE = 0.6794
EXP16_LAM0_CE = 0.6861  # fp32-sims ST baseline (best Exp16 λ=0 result)

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

log("=== Exp 22: VQ Commitment Loss ===")
log(f"CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
notify("Exp22 starting", f"VQ commitment β={BETAS}, seed={SEED} | SOTA={EXP19_SOTA_CE:.4f}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    log(f"  GPU {i}: {p.name}")

log(f"Loading {MODEL_NAME}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map="cuda:0")
model.eval()
log(f"Model loaded in {time.time()-t0:.1f}s | {gpu_mem_str()}")
notify("Exp22 started", f"Model loaded | β={BETAS} | {gpu_mem_str()}")

PLACEHOLDER_IDS = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)
embed_fn = model.get_input_embeddings()
EMB_DIM  = embed_fn.weight.shape[1]
VOCAB    = embed_fn.weight.shape[0]
EMB_DEV  = embed_fn.weight.device

# Precompute normalized vocab embeddings (float32 for cosine accuracy)
with torch.no_grad():
    W_norm = F.normalize(embed_fn.weight.to(dtype=torch.float32, device=EMB_DEV), dim=-1)

# ── Helpers ──────────────────────────────────────────────────────────────────

def chat_ids(messages):
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
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
        sn = F.normalize(soft_LD.to(dtype=torch.float32), dim=-1)
        return (sn @ W_norm.T).argmax(dim=-1)

def st_project_with_commitment(soft_prefix_LD):
    """ST estimator + returns cosine commitment loss term (differentiable).

    Returns:
        st_emb: straight-through projected embedding
        ids: projected token IDs
        commit_loss: mean(1 - cos(soft_i, stop_grad(embed(argmax_i))))
    """
    soft_f = soft_prefix_LD.to(dtype=torch.float32)
    sn     = F.normalize(soft_f, dim=-1)
    sims   = sn @ W_norm.T  # [L, V]
    ids    = sims.argmax(dim=-1)

    with torch.no_grad():
        proj_emb = embed_fn(ids).to(dtype=torch.float32)
        proj_emb_norm = F.normalize(proj_emb, dim=-1)  # stop_grad normalized nearest

    # ST embedding for CE (gradients flow through soft_prefix)
    st_emb = soft_prefix_LD + (embed_fn(ids).to(dtype=soft_prefix_LD.dtype)
                                - soft_prefix_LD).detach()

    # Commitment: cosine similarity of soft to its nearest embedding (no grad on proj)
    # Higher = better = soft is closer to nearest token
    cos_sim = (sn * proj_emb_norm).sum(dim=-1)  # [L], differentiable wrt soft_f
    commit_loss = (1.0 - cos_sim).mean()         # [scalar], 0=perfect alignment

    return st_emb, ids, commit_loss

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

# ── Reference completions ─────────────────────────────────────────────────────
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

# ── Shared initialization (same seed as Exp11/16/21) ─────────────────────────
with torch.no_grad():
    emb_mean = embed_fn.weight.mean(0)
    emb_std  = embed_fn.weight.std(0) * 0.1
torch.manual_seed(SEED)
init_noise = (emb_mean.unsqueeze(0).repeat(PREFIX_LEN, 1)
              + torch.randn(PREFIX_LEN, EMB_DIM, device=EMB_DEV, dtype=DTYPE)
              * emb_std.unsqueeze(0))

# ── Run one experiment per beta ────────────────────────────────────────────────
all_results = []

try:
    for beta in BETAS:
        log(f"\n{'='*60}")
        log(f"=== β = {beta} ===")
        log(f"{'='*60}")
        notify(f"Exp22 β={beta} starting", f"VQ commitment, seed={SEED} | SOTA={EXP19_SOTA_CE:.4f}")

        soft_prefix = init_noise.clone().detach().float().requires_grad_(True)
        optimizer   = torch.optim.Adam([soft_prefix], lr=LR)

        st_log, commit_log = [], []
        best_st_ce = float('inf')
        best_soft  = soft_prefix.data.clone()

        log(f"ST+commitment soft opt ({SOFT_STEPS} steps, β={beta}) | {gpu_mem_str()}")
        t0 = time.time()
        for step in range(SOFT_STEPS):
            optimizer.zero_grad()
            st_emb, _, commit_loss = st_project_with_commitment(soft_prefix)

            st_ce_loss = compute_ce_soft_batched(
                st_emb, SUFFIXES[:BATCH_SIZE], ref_completions[:BATCH_SIZE])

            loss = st_ce_loss + beta * commit_loss

            loss.backward()
            optimizer.step()

            ce_val = st_ce_loss.item()
            cl_val = commit_loss.item()
            st_log.append(ce_val)
            commit_log.append(cl_val)

            if ce_val < best_st_ce:
                best_st_ce = ce_val
                best_soft  = soft_prefix.data.clone()

            if step % 50 == 0 or step == SOFT_STEPS - 1:
                elapsed = time.time() - t0
                log(f"  [{step:4d}/{SOFT_STEPS}] ST-CE={ce_val:.5f}  best={best_st_ce:.5f}  "
                    f"commit={cl_val:.4f}  elapsed={elapsed:.0f}s  {gpu_mem_str()}")

        t_soft = time.time() - t0
        log(f"ST done. Final={st_log[-1]:.5f}, best={best_st_ce:.5f}, time={t_soft:.1f}s")
        log(f"  Final commit loss: {commit_log[-1]:.4f} (0=perfect alignment)")

        # Projection from best soft prefix
        projected_ids  = project_to_tokens(best_soft.to(DTYPE))
        projected_text = tokenizer.decode(projected_ids.cpu().tolist())
        proj_ce = compute_ce_discrete_batched(projected_ids, SUFFIXES, ref_completions)
        log(f"Projected (best): {projected_text!r}  CE={proj_ce:.5f}")

        final_ids       = project_to_tokens(soft_prefix.detach().to(DTYPE))
        final_proj_text = tokenizer.decode(final_ids.cpu().tolist())
        final_proj_ce   = compute_ce_discrete_batched(final_ids, SUFFIXES, ref_completions)
        log(f"Projected (final step): {final_proj_text!r}  CE={final_proj_ce:.5f}")

        # HotFlip
        log(f"HotFlip ({HOTFLIP_STEPS} steps, topk={HF_TOPK})...")
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
                log(f"  [{step:3d}/{HOTFLIP_STEPS}] CE={current_ce:.5f}  "
                    f"{'↓' if improved else '–'}  {toks!r}  {gpu_mem_str()}")

        t_hf = time.time() - t0
        hotflip_ce = current_ce
        final_text = tokenizer.decode(current_ids.cpu().tolist())
        log(f"HotFlip done. CE={hotflip_ce:.5f}  time={t_hf:.1f}s")

        all_results.append({
            "beta": beta,
            "best_st_ce": best_st_ce,
            "final_st_ce": st_log[-1],
            "final_commit_loss": commit_log[-1],
            "projection_ce_best": proj_ce,
            "projection_ce_final": final_proj_ce,
            "hotflip_ce": hotflip_ce,
            "projected_text": projected_text,
            "final_proj_text": final_proj_text,
            "final_text": final_text,
            "final_ids": current_ids.tolist(),
            "st_log": st_log,
            "commit_log": commit_log,
            "hotflip_log": hotflip_log,
            "timing": {"soft_seconds": t_soft, "hotflip_seconds": t_hf},
        })

        # Checkpoint after each beta
        with open(CKPT_PATH, "w") as f:
            json.dump({"runs": all_results,
                       "completed_betas": [r["beta"] for r in all_results]}, f, indent=2)
        log(f"Checkpoint saved ({len(all_results)}/{len(BETAS)} β done)")

        sota_flag = " *** NEW SOTA ***" if hotflip_ce < EXP19_SOTA_CE else ""
        notify(
            f"Exp22 β={beta} done{sota_flag}",
            f"proj={proj_ce:.4f} → hf={hotflip_ce:.4f} | "
            f"SOTA={EXP19_SOTA_CE:.4f} | {len(all_results)}/{len(BETAS)} done",
        )

except Exception as e:
    notify("Exp22 FAILED", str(e))
    raise

# ── Summary ───────────────────────────────────────────────────────────────────
log(f"\n{'='*60}")
log("=== FINAL SUMMARY ===")
log(f"{'='*60}")
log(f"  {'β':>6}  {'proj-CE':>10}  {'HotFlip':>10}  {'commit':>8}  prefix")
for r in all_results:
    log(f"  {r['beta']:>6.1f}  {r['projection_ce_best']:>10.5f}  "
        f"{r['hotflip_ce']:>10.5f}  {r['final_commit_loss']:>8.4f}  {r['final_text']!r}")
log(f"  {'β=0':>6}  {'0.76245':>10}  {EXP16_LAM0_CE:>10.5f}  {'(Exp16)':>8}")
log(f"  {'Exp19':>6}  {'0.90526':>10}  {EXP19_SOTA_CE:>10.5f}  {'(SOTA)':>8}")

results = {
    "experiment": "steer-001-exp22-vq-commitment",
    "model": MODEL_NAME,
    "reference_prefix": REF_PREFIX,
    "seed": SEED,
    "prefix_len": PREFIX_LEN,
    "n_soft_steps": SOFT_STEPS,
    "n_hotflip_steps": HOTFLIP_STEPS,
    "batch_size": BATCH_SIZE,
    "hf_topk": HF_TOPK,
    "lr": LR,
    "betas_tested": BETAS,
    "method": "st_vq_commitment_cosine",
    "commitment_formula": "mean_i(1 - cos(soft_i, stop_grad(embed(argmax_i))))",
    "runs": all_results,
    "best_hotflip_ce": min(r["hotflip_ce"] for r in all_results),
    "best_beta": min(all_results, key=lambda r: r["hotflip_ce"])["beta"],
    "exp16_lam0_baseline": EXP16_LAM0_CE,
    "exp19_sota": EXP19_SOTA_CE,
    "exp11_baseline": {"proj": 0.76245, "hotflip": 0.68935},
}
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
log(f"\nResults saved to {OUT_PATH}")
log("=== DONE ===")

_best = min(all_results, key=lambda r: r["hotflip_ce"])
notify(
    "Exp22 complete",
    f"Best: β={_best['beta']}, CE={_best['hotflip_ce']:.4f} | "
    f"Exp16λ0={EXP16_LAM0_CE:.4f} | SOTA={EXP19_SOTA_CE:.4f}",
)
