"""
Experiment 18: TOPK Escalation from Exp11 Best Prefix

Motivation:
- Exp 11 achieved SOTA HotFlip CE = 0.689 with TOPK=30.
- Both Exp10 and Exp11 converge at HotFlip step 10/80 — deep local minimum.
- Question: Is this local minimum truly a global attractor, or can broader search escape it?
- With A100 GPUs, we can afford TOPK=100-200 which is prohibitive on L40.

Design:
- No ST phase (start directly from Exp11 best prefix)
- Phase 1: HotFlip TOPK=50 for 50 steps — confirm convergence with modest increase
- Phase 2: HotFlip TOPK=100 for 80 steps — broader candidate search
- Phase 3: HotFlip TOPK=200 for 50 steps — maximum search width
- Within each phase, carry the best prefix forward

Exp11 best prefix IDs: [6828, 44013, 169091, 77616, 48982, 7260, 675, 19493]
Exp11 best text: ' Cat wellnessceptre人工语言 reply with cats' (CE=0.6893)

Key question answered: Is 0.689 escapable with broader single-token swaps?
If Phase 1 immediately stays flat, we know TOPK=30 already found all improving swaps.
If any phase improves CE, we know higher TOPK matters and should be used by default.

Output: /home/jovyan/steer001_topk_escalation.json
"""

import sys, importlib.util as _ilu

_real_find_spec = _ilu.find_spec
def _patched(name, package=None, target=None):
    return None if name == "torchvision" else _real_find_spec(name, package)
_ilu.find_spec = _patched
for _k in list(sys.modules):
    if "torchvision" in _k: del sys.modules[_k]

import json, time
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
EARLY_K       = 32
EARLY_WEIGHT  = 3.0
OUT_PATH      = Path("/home/jovyan/steer001_topk_escalation.json")

# Exp11 best prefix (confirmed CE = 0.6893)
EXP11_IDS  = [6828, 44013, 169091, 77616, 48982, 7260, 675, 19493]

# Escalation phases: (topk, n_steps)
PHASES = [
    (50,  50),
    (100, 80),
    (200, 50),
]

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

log("=== Exp 18: TOPK Escalation from Exp11 Best Prefix ===")
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
EMB_DEV  = embed_fn.weight.device

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

def compute_ce_discrete_batched(prefix_ids_L, suffix_texts, ref_completions):
    prefix_ids_L = prefix_ids_L.to(EMB_DEV)
    with torch.no_grad():
        soft = embed_fn(prefix_ids_L)
        batch_emb, meta, T_max = build_batch(soft, suffix_texts, ref_completions)
        logits = model(inputs_embeds=batch_emb).logits
        loss = compute_ce_from_batch(logits, meta, T_max)
    return loss.item()

def hotflip_step(current_ids, ref_completions, topk):
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
        cands = scores.topk(topk, largest=False).indices
        for k in range(topk):
            trial_ids = current_ids.clone()
            trial_ids[pos] = cands[k].item()
            ce_k = compute_ce_discrete_batched(trial_ids, SUFFIXES, ref_completions)
            if ce_k < best_ce:
                best_ce = ce_k
                best_ids = trial_ids.clone()

    return best_ids, best_ce

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

# ── Verify Exp11 starting CE ─────────────────────────────────────────────────
log(f"\nLoading Exp11 best prefix: {EXP11_IDS}")
current_ids = torch.tensor(EXP11_IDS, dtype=torch.long).to(EMB_DEV)
starting_text = tokenizer.decode(current_ids.cpu().tolist())
starting_ce   = compute_ce_discrete_batched(current_ids, SUFFIXES, ref_completions)
log(f"Starting prefix: {starting_text!r}  CE={starting_ce:.5f} (Exp11 reported: 0.68935)")
current_ce = starting_ce

# ── TOPK Escalation ────────────────────────────────────────────────────────
all_phases = []
global_best_ce  = current_ce
global_best_ids = current_ids.clone()

for phase_idx, (topk, n_steps) in enumerate(PHASES):
    log(f"\n{'='*60}")
    log(f"=== Phase {phase_idx+1}: TOPK={topk}, {n_steps} steps ===")
    log(f"Starting CE: {current_ce:.5f}  {gpu_mem_str()}")
    log(f"{'='*60}")

    phase_log = [current_ce]
    phase_improved = 0
    t0 = time.time()

    for step in range(n_steps):
        new_ids, new_ce = hotflip_step(current_ids, ref_completions, topk)
        improved = new_ce < current_ce
        if improved:
            current_ids = new_ids
            current_ce  = new_ce
            phase_improved += 1
            if current_ce < global_best_ce:
                global_best_ce  = current_ce
                global_best_ids = current_ids.clone()
                log(f"  *** NEW GLOBAL BEST: {global_best_ce:.5f} ***")
        phase_log.append(current_ce)

        if step % 10 == 0 or step == n_steps - 1:
            toks = tokenizer.decode(current_ids.cpu().tolist())
            log(f"  [{step:3d}/{n_steps}] CE={current_ce:.5f}  "
                f"{'↓' if improved else '–'}  {toks!r}")

    t_phase = time.time() - t0
    log(f"Phase {phase_idx+1} done: final CE={current_ce:.5f}, "
        f"improved {phase_improved}/{n_steps} steps, time={t_phase:.1f}s")

    all_phases.append({
        "phase": phase_idx + 1,
        "topk": topk,
        "n_steps": n_steps,
        "starting_ce": phase_log[0],
        "final_ce": current_ce,
        "n_improved_steps": phase_improved,
        "log": phase_log,
        "final_text": tokenizer.decode(current_ids.cpu().tolist()),
        "final_ids": current_ids.tolist(),
        "elapsed_seconds": t_phase,
    })

log(f"\n{'='*60}")
log(f"=== FINAL SUMMARY ===")
log(f"{'='*60}")
log(f"  Starting (Exp11): CE={starting_ce:.5f}")
for p in all_phases:
    log(f"  Phase {p['phase']} (TOPK={p['topk']:>3d}, {p['n_steps']} steps): "
        f"{p['starting_ce']:.5f} → {p['final_ce']:.5f}  "
        f"({p['n_improved_steps']} improvements)")
log(f"  Global best: {global_best_ce:.5f}  "
    f"({tokenizer.decode(global_best_ids.cpu().tolist())!r})")
log(f"  Improvement over Exp11: {starting_ce - global_best_ce:+.5f}")

results = {
    "experiment": "steer-001-exp18-topk-escalation",
    "model": MODEL_NAME,
    "reference_prefix": REF_PREFIX,
    "exp11_start_ids": EXP11_IDS,
    "exp11_start_text": starting_text,
    "exp11_reported_ce": 0.68935,
    "verified_starting_ce": starting_ce,
    "phases": all_phases,
    "global_best_ce": global_best_ce,
    "global_best_ids": global_best_ids.tolist(),
    "global_best_text": tokenizer.decode(global_best_ids.cpu().tolist()),
    "total_improvement": starting_ce - global_best_ce,
}
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
log(f"\nResults saved to {OUT_PATH}")
log("=== DONE ===")
