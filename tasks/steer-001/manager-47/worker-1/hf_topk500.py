"""
Experiment 47: HotFlip topk=500 from new SOTA (CE=0.5993)

Motivation:
- Exp46 showed topk=200 beats topk=50: found 2 better token swaps in step 1-2,
  reducing CE from 0.6044 → 0.5993 (new SOTA).
- The new minimum converged at step 2 under topk=200, so it's stable there.
- Hypothesis: widening to topk=500 may expose further candidates that rank
  201-500 and escape the current local minimum.
- Two runs from new SOTA:
    Run A: greedy HF topk=500, 100 steps
    Run B: greedy HF topk=200 (control, verify new SOTA is stable)

New SOTA IDs: [50105, 111, 133522, 222115, 24539, 202257, 10358, 131146,
               73815, 242580, 231898, 2976, 55135, 5598, 31459, 19493]
CE=0.5993 (from Exp46, pos5=202257, pos7=131146 changed vs Exp26)

Timing estimate:
  Run A (topk=500): 100 × 16 × 500 cands → ~10× base ≈ 62 min
  Run B (topk=200): 100 steps → ~26 min (control)
  Total: ~88 min
Output: /home/jovyan/steer001_hf_topk500.json
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
import json as _json, urllib.request as _urlreq

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

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "google/gemma-2-2b-it"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE        = torch.bfloat16
REF_PREFIX   = "Talk only about cats."
PLACEHOLDER  = "SOFTPREFIX"
PREFIX_LEN   = 16
HF_STEPS     = 100
HF_TOPK_A    = 500
HF_TOPK_B    = 200
EARLY_K      = 32
EARLY_WEIGHT = 3.0
NEW_SOTA_CE  = 0.599288
NEW_SOTA_IDS = [50105, 111, 133522, 222115, 24539, 202257, 10358, 131146,
                73815, 242580, 231898, 2976, 55135, 5598, 31459, 19493]
OLD_SOTA_CE  = 0.60437
OUT_PATH     = Path("/home/jovyan/steer001_hf_topk500.json")

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


try:
    log("=== Exp47: HF topk=500 + topk=200 control from new SOTA ===")
    log(f"CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")

    log(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map="cuda:0")
    model.eval()
    log(f"Model loaded in {time.time()-t0:.1f}s | {gpu_mem_str()}")
    notify("Exp47 started", f"HF topk=500 from new SOTA | new_SOTA={NEW_SOTA_CE:.5f}")

    PLACEHOLDER_IDS = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)
    embed_fn = model.get_input_embeddings()
    EMB_DIM  = embed_fn.weight.shape[1]
    EMB_DEV  = embed_fn.weight.device

    # ── Helpers ───────────────────────────────────────────────────────────────

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

    def run_hf(label, start_ids, topk, n_steps):
        log(f"\n{'='*60}")
        log(f"=== {label}: topk={topk}, {n_steps} steps ===")
        current_ids = start_ids.clone().to(EMB_DEV)
        current_ce  = compute_ce_discrete_batched(current_ids, SUFFIXES, ref_completions)
        log(f"Start CE: {current_ce:.6f}")
        hf_log = [current_ce]

        t0 = time.time()
        for step in range(n_steps):
            new_ids, new_ce = hotflip_step(current_ids, ref_completions, topk)
            improved = new_ce < current_ce
            if improved:
                current_ids = new_ids
                current_ce  = new_ce
            hf_log.append(current_ce)
            if step % 10 == 0 or step == n_steps - 1:
                toks = tokenizer.decode(current_ids.cpu().tolist())
                log(f"  [{step:3d}/{n_steps}] CE={current_ce:.6f}  {'↓' if improved else '–'}  {toks!r}")
            if step > 5 and all(x == hf_log[-1] for x in hf_log[-6:]):
                log(f"  Converged at step {step}, stopping early.")
                break

        elapsed = time.time() - t0
        text = tokenizer.decode(current_ids.cpu().tolist())
        log(f"{label} done: CE={current_ce:.6f} in {elapsed:.0f}s")
        return current_ids, current_ce, text, hf_log, elapsed

    # ── Reference completions ─────────────────────────────────────────────────
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
    new_sota_tensor = torch.tensor(NEW_SOTA_IDS, dtype=torch.long, device=EMB_DEV)

    sota_text = tokenizer.decode(NEW_SOTA_IDS)
    log(f"New SOTA prefix: {sota_text!r}  CE={NEW_SOTA_CE:.6f}")

    # Run B first (topk=200 control, faster)
    ids_b, ce_b, text_b, log_b, elapsed_b = run_hf(
        "Run B (control, topk=200)", new_sota_tensor, HF_TOPK_B, HF_STEPS)
    beat_b = ce_b < NEW_SOTA_CE
    notify(f"Exp47 Run B done",
           f"topk=200: CE={ce_b:.6f}  {'BEATS new SOTA' if beat_b else f'delta={ce_b-NEW_SOTA_CE:+.6f}'}")

    # Run A (topk=500)
    ids_a, ce_a, text_a, log_a, elapsed_a = run_hf(
        "Run A (topk=500)", new_sota_tensor, HF_TOPK_A, HF_STEPS)
    beat_a = ce_a < NEW_SOTA_CE
    notify(f"Exp47 Run A done",
           f"topk=500: CE={ce_a:.6f}  {'BEATS new SOTA' if beat_a else f'delta={ce_a-NEW_SOTA_CE:+.6f}'}")

    overall_best_ce = min(ce_a, ce_b)

    log(f"\n=== FINAL SUMMARY ===")
    log(f"  New SOTA (Exp46): CE={NEW_SOTA_CE:.6f}")
    log(f"  Run A (topk=500): CE={ce_a:.6f}  {'*** BEATS ***' if beat_a else f'delta={ce_a-NEW_SOTA_CE:+.6f}'}")
    log(f"  Run B (topk=200): CE={ce_b:.6f}  {'*** BEATS ***' if beat_b else f'delta={ce_b-NEW_SOTA_CE:+.6f}'}")

    results = {
        "experiment": "steer-001-exp47-hf-topk500",
        "model": MODEL_NAME,
        "reference_prefix": REF_PREFIX,
        "prefix_len": PREFIX_LEN,
        "hf_steps": HF_STEPS,
        "new_sota_ce": NEW_SOTA_CE,
        "new_sota_ids": NEW_SOTA_IDS,
        "old_sota_ce": OLD_SOTA_CE,
        "run_a": {
            "description": f"greedy HotFlip topk={HF_TOPK_A}",
            "hf_topk": HF_TOPK_A,
            "best_ce": ce_a,
            "best_text": text_a,
            "best_ids": ids_a.tolist(),
            "hf_log": log_a,
            "elapsed_seconds": elapsed_a,
        },
        "run_b": {
            "description": f"greedy HotFlip topk={HF_TOPK_B} (control)",
            "hf_topk": HF_TOPK_B,
            "best_ce": ce_b,
            "best_text": text_b,
            "best_ids": ids_b.tolist(),
            "hf_log": log_b,
            "elapsed_seconds": elapsed_b,
        },
        "overall_best_ce": overall_best_ce,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {OUT_PATH}")

    notify("Exp47 complete",
           f"topk=500: {ce_a:.6f}  topk=200: {ce_b:.6f}  "
           f"{'NEW SOTA BEATEN' if overall_best_ce < NEW_SOTA_CE else f'best delta={overall_best_ce-NEW_SOTA_CE:+.6f}'}")

except Exception as e:
    notify("Exp47 FAILED", str(e))
    raise
