"""
Exp 4: Semantic naturalness penalty in HotFlip.

Research question: Is there a naturalness penalty weight λ that produces
more readable discrete prefixes without significantly wrecking behavioral CE?

Design:
  - Start from the baseline projected_ids (from Exp 1, steer001_baseline.json).
  - For each λ ∈ {0.0, 0.1, 0.5, 1.0}:
      Run 60 HotFlip steps with combined score:
          score(v) = (W @ grad) + λ * (1 - naturalness(v))
      where naturalness(v) = fraction of ASCII letter/space chars in decode(v),
      normalized to [0,1]. Higher λ = stronger pressure toward readable tokens.
      λ=0.0 reproduces standard HotFlip (baseline comparison).

  - Metrics per λ:
      - hotflip_ce_final: behavioral fidelity (lower = better)
      - naturalness_score: fraction of ASCII chars in the final prefix
      - prefix_text: the actual prefix string

  Note on naturalness measure: we use character-level ASCII fraction instead
  of DistilGPT2 LM probability to avoid cross-tokenizer alignment issues.
  This is a proxy for readability; correlated but not identical to fluency.

Outputs: /home/jovyan/steer001_naturalness.json
"""

import sys, importlib.util as _ilu

_real_find_spec = _ilu.find_spec
def _patched(name, package=None, target=None):
    return None if name == "torchvision" else _real_find_spec(name, package)
_ilu.find_spec = _patched
for _k in list(sys.modules):
    if "torchvision" in _k: del sys.modules[_k]

import json, os, time
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
HOTFLIP_STEPS = 60
HF_TOPK       = 30
EARLY_K       = 32
EARLY_WEIGHT  = 3.0

LAMBDA_VALUES = [0.0, 0.1, 0.5, 1.0]

# Starting point: projected_ids from Exp 1 baseline
BASELINE_JSON = Path("/home/jovyan/steer001_baseline.json")
REF_COMP_PATH = Path("/home/jovyan/steer001_ref_completions.pt")
OUT_PATH      = Path("/home/jovyan/steer001_naturalness.json")

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

# ── Telemetry ────────────────────────────────────────────────────────────────
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

# ── Startup ──────────────────────────────────────────────────────────────────
log("=== Exp 4: Naturalness Penalty in HotFlip ===")
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

# ── Helpers ───────────────────────────────────────────────────────────────────
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
    T_max  = max(s.shape[1] for s in seqs)
    padded = []
    for seq in seqs:
        pad_len = T_max - seq.shape[1]
        if pad_len:
            pad = torch.zeros(1, pad_len, EMB_DIM, device=seq.device, dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=1)
        padded.append(seq)
    return torch.cat(padded, dim=0).to(EMB_DEV), meta, T_max

def compute_ce_from_batch(logits, meta, T_max):
    total  = torch.tensor(0.0, device=logits.device)
    weight = 0.0
    for b, (comp_start, comp_ids) in enumerate(meta):
        for i, tok in enumerate(comp_ids):
            pos = comp_start + i - 1
            if pos >= T_max: continue
            w    = EARLY_WEIGHT if i < EARLY_K else 1.0
            ce   = F.cross_entropy(logits[b, pos].unsqueeze(0), tok.unsqueeze(0).long())
            total = total + w * ce
            weight += w
    return (total / weight) if weight > 0 else total

def compute_ce_discrete_batched(prefix_ids_L, suffix_texts, ref_completions):
    prefix_ids_L = prefix_ids_L.to(EMB_DEV)
    with torch.no_grad():
        soft = embed_fn(prefix_ids_L)
        batch_emb, meta, T_max = build_batch(soft, suffix_texts, ref_completions)
        logits = model(inputs_embeds=batch_emb).logits
        loss   = compute_ce_from_batch(logits, meta, T_max)
    return loss.item()


def ascii_naturalness(token_ids):
    """Fraction of chars in decoded text that are ASCII letters or spaces."""
    text = tokenizer.decode(token_ids.cpu().tolist() if hasattr(token_ids, 'tolist') else list(token_ids))
    if not text: return 0.0
    ascii_chars = sum(1 for c in text if c.isascii() and (c.isalpha() or c == ' '))
    return ascii_chars / len(text)


# ── Precompute per-token naturalness scores ───────────────────────────────────
log("Precomputing per-token naturalness scores...")
t0 = time.time()
# Decode each token and compute ASCII fraction (batched via list comprehension)
# Avoid decoding all 256k tokens at once — chunk to reduce memory
CHUNK = 4096
nat_scores_list = []
for start in range(0, VOCAB, CHUNK):
    end    = min(start + CHUNK, VOCAB)
    ids    = list(range(start, end))
    tokens = tokenizer.batch_decode([[i] for i in ids])
    for tok_str in tokens:
        if not tok_str:
            nat_scores_list.append(0.0)
        else:
            ascii_ch = sum(1 for c in tok_str if c.isascii() and (c.isalpha() or c == ' '))
            nat_scores_list.append(ascii_ch / len(tok_str))

NAT_SCORES = torch.tensor(nat_scores_list, dtype=torch.float32, device=EMB_DEV)
log(f"Naturalness scores computed in {time.time()-t0:.1f}s. "
    f"mean={NAT_SCORES.mean():.3f}  frac_natural={( NAT_SCORES > 0.5).float().mean():.3f}")


def hotflip_step_with_naturalness(current_ids, ref_completions, ref_ids_set, lam):
    """
    HotFlip step with naturalness penalty.
    score(v) = (W @ grad[pos]) + λ * (1 - naturalness(v))
    Lower score = more preferred (we minimize).
    """
    current_ids = current_ids.to(EMB_DEV)
    prefix_emb  = embed_fn(current_ids).float().detach().requires_grad_(True)
    batch_emb, meta, T_max = build_batch(prefix_emb, SUFFIXES, ref_completions)
    logits = model(inputs_embeds=batch_emb).logits
    loss   = compute_ce_from_batch(logits, meta, T_max)
    loss.backward()
    grad = prefix_emb.grad  # [L, D] float32

    best_ids = current_ids.clone()
    best_ce  = compute_ce_discrete_batched(current_ids, SUFFIXES, ref_completions)
    W = embed_fn.weight.float()  # [V, D]

    for pos in range(PREFIX_LEN):
        g              = grad[pos]           # [D]
        base_scores    = W @ g               # [V]
        nat_penalty    = lam * (1.0 - NAT_SCORES.float())  # [V]: unnatural → high penalty
        scores         = base_scores + nat_penalty

        # Mask special/banned tokens
        scores[list(ref_ids_set)]         = float('inf')
        scores[tokenizer.bos_token_id]    = float('inf')
        scores[tokenizer.eos_token_id]    = float('inf')
        if tokenizer.pad_token_id is not None:
            scores[tokenizer.pad_token_id] = float('inf')

        cands     = scores.topk(HF_TOPK, largest=False).indices
        cand_embs = embed_fn(current_ids).unsqueeze(0).expand(HF_TOPK, -1, -1).clone()
        cand_embs[:, pos, :] = embed_fn(cands)

        for k in range(HF_TOPK):
            with torch.no_grad():
                soft_LD  = cand_embs[k].detach()
                b_emb, meta_k, T_k = build_batch(soft_LD, SUFFIXES, ref_completions)
                logits_k = model(inputs_embeds=b_emb).logits
                ce_k     = compute_ce_from_batch(logits_k, meta_k, T_k).item()
            if ce_k < best_ce:
                best_ce = ce_k
                best_ids = current_ids.clone()
                best_ids[pos] = cands[k].item()

    return best_ids, best_ce


# ── Load starting point (projected_ids from Exp 1) ───────────────────────────
log(f"Loading baseline projected_ids from {BASELINE_JSON}...")
with open(BASELINE_JSON) as f:
    baseline = json.load(f)
projected_token_ids = baseline["projected_token_ids"]
PROJECTED_IDS = torch.tensor(projected_token_ids, dtype=torch.long).to(EMB_DEV)
PROJ_CE       = baseline["metrics"]["projection_ce"]
log(f"Projected IDs: {projected_token_ids}")
log(f"Projected text: {tokenizer.decode(projected_token_ids)!r}")
log(f"Projection CE (from Exp 1): {PROJ_CE:.5f}")

# ── Reference completions ─────────────────────────────────────────────────────
log(f"Loading ref completions from {REF_COMP_PATH}...")
ref_completions = torch.load(REF_COMP_PATH)
log(f"Loaded {len(ref_completions)} completions.")

ref_ids_set = set(tokenizer.encode(REF_PREFIX, add_special_tokens=False))

# ── Per-λ HotFlip ─────────────────────────────────────────────────────────────
all_results = []

for lam in LAMBDA_VALUES:
    log(f"\n{'='*60}")
    log(f"λ = {lam}")
    log(f"{'='*60}")

    ckpt_path = Path(f"/home/jovyan/steer001_nat_ckpt_lam{str(lam).replace('.','p')}.pt")

    # Resume from checkpoint if available
    hf_start_step = 0
    current_ids   = PROJECTED_IDS.clone()
    current_ce    = PROJ_CE
    hotflip_log   = [current_ce]

    if ckpt_path.exists():
        log(f"Found checkpoint {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        if "current_ids" in ckpt:
            hf_start_step = ckpt.get("hf_step", 0)
            current_ids   = ckpt["current_ids"].to(EMB_DEV)
            current_ce    = ckpt["current_ce"]
            hotflip_log   = ckpt["hotflip_log"]
            log(f"Resuming from step {hf_start_step}, CE={current_ce:.5f}")

    t_start = time.time()
    for step in range(hf_start_step, HOTFLIP_STEPS):
        new_ids, new_ce = hotflip_step_with_naturalness(
            current_ids, ref_completions, ref_ids_set, lam)
        improved = new_ce < current_ce
        if improved:
            current_ids = new_ids
            current_ce  = new_ce
        hotflip_log.append(current_ce)

        if step % 10 == 0 or step == HOTFLIP_STEPS - 1:
            toks = tokenizer.decode(current_ids.cpu().tolist())
            nat  = ascii_naturalness(current_ids)
            log(f"  [{step:3d}/{HOTFLIP_STEPS}] CE={current_ce:.5f}  nat={nat:.3f}  "
                f"{'↓' if improved else '–'}  {toks!r}  {gpu_mem_str()}")

        torch.save({"hf_step": step+1, "current_ids": current_ids.cpu(),
                    "current_ce": current_ce, "hotflip_log": hotflip_log}, ckpt_path)

    hotflip_ce = current_ce
    t_elapsed  = time.time() - t_start
    final_text = tokenizer.decode(current_ids.cpu().tolist())
    nat_score  = ascii_naturalness(current_ids)
    log(f"λ={lam} done. CE={hotflip_ce:.5f}  nat={nat_score:.3f}  time={t_elapsed:.1f}s")
    log(f"Final prefix: {final_text!r}")

    all_results.append({
        "lambda":             lam,
        "reference_prefix":   REF_PREFIX,
        "starting_point":     "projected_ids_from_exp1",
        "metrics": {
            "projection_ce_start": PROJ_CE,
            "hotflip_ce_final":    float(hotflip_ce),
            "recovery_vs_proj":    float(PROJ_CE - hotflip_ce),
            "naturalness_score":   float(nat_score),
        },
        "hotflip_log":     hotflip_log,
        "final_prefix_text": final_text,
        "final_token_ids": current_ids.cpu().tolist(),
    })

    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"Intermediate saved (λ={lam} done)")

# ── Final output ──────────────────────────────────────────────────────────────
with open(OUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
log(f"\nAll results saved → {OUT_PATH}")

log("\n=== SUMMARY ===")
log(f"{'λ':>6}  {'hf_ce':>8}  {'recovery':>9}  {'natural':>8}  prefix")
for r in all_results:
    m = r["metrics"]
    log(f"  {r['lambda']:6.2f}  {m['hotflip_ce_final']:8.4f}  "
        f"{m['recovery_vs_proj']:9.4f}  {m['naturalness_score']:8.3f}  "
        f"{r['final_prefix_text']!r}")
log("=== DONE ===")
