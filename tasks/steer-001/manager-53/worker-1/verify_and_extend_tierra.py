"""
Experiment 53: Verify new SOTA candidate and extend HotFlip

Motivation:
- Exp52 found CE=0.59892 at HotFlip step 0 (vs SOTA=0.59929).
  Token change: pos4 'Santé' → 'tierra'
  Text: ' Cats\n\n\n\n不说tierra nutrition田市 Answer Сейчас prompts💬LEVANTestiArn write purely cats'
- Exp52 timed out at step 10 with CE=0.59892 (no further improvement in steps 0-10).
- This experiment: verify the candidate CE, get IDs, then run 30 more HF steps topk=200.
  If CE confirmed < 0.59929 → new SOTA.

Timing estimate: 1 (verify) + 30 HF steps × ~8 min = ~4.5h
Output: /home/jovyan/steer001_tierra_verify.json
"""

import sys, importlib.util as _ilu

_real_find_spec = _ilu.find_spec
def _patched(name, package=None, target=None):
    return None if name == "torchvision" else _real_find_spec(name, package)
_ilu.find_spec = _patched
for _k in list(sys.modules):
    if "torchvision" in _k: del sys.modules[_k]

import json, math, os as _os, time
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
MODEL_NAME    = "google/gemma-2-2b-it"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.bfloat16
REF_PREFIX    = "Talk only about cats."
PLACEHOLDER   = "SOFTPREFIX"
PREFIX_LEN    = 16
HOTFLIP_STEPS = 30
HF_TOPK       = 200
BATCH_SIZE    = 6   # keeps memory under 5.2GB
EARLY_K       = 32
EARLY_WEIGHT  = 3.0

# New candidate found by Exp52 — 'Santé' replaced by 'tierra'
CANDIDATE_TEXT = ' Cats\n\n\n\n不说tierra nutrition田市 Answer Сейчас prompts💬LEVANTestiArn write purely cats'
OLD_SOTA_CE   = 0.60437
NEW_SOTA_CE   = 0.599288  # from Exp46
EXP52_CE      = 0.59892   # reported by Exp52 step 0

NEW_SOTA_IDS  = [50105, 111, 133522, 222115, 24539, 202257, 10358, 131146,
                 73815, 242580, 231898, 2976, 55135, 5598, 31459, 19493]

OUT_PATH  = Path("/home/jovyan/steer001_tierra_verify.json")
CKPT_PATH = Path("/home/jovyan/steer001_tierra_verify_ckpt.json")

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
    print(f"[Exp53] [{ts}] {msg}", flush=True)


try:
    log("=== Exp53: Verify Exp52 nueva SOTA candidate + extend HF ===")
    log(f"CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
    log(f"Candidate: {CANDIDATE_TEXT!r}")
    log(f"Expected CE from Exp52: {EXP52_CE}")

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

    # Tokenize the candidate text to get IDs
    candidate_ids_list = tokenizer.encode(CANDIDATE_TEXT, add_special_tokens=False)
    log(f"Candidate tokenized: {len(candidate_ids_list)} tokens: {candidate_ids_list}")
    if len(candidate_ids_list) != PREFIX_LEN:
        log(f"WARNING: expected {PREFIX_LEN} tokens, got {len(candidate_ids_list)}")
        # Pad or truncate
        if len(candidate_ids_list) < PREFIX_LEN:
            candidate_ids_list = candidate_ids_list + [tokenizer.pad_token_id or 0] * (PREFIX_LEN - len(candidate_ids_list))
        else:
            candidate_ids_list = candidate_ids_list[:PREFIX_LEN]
    candidate_ids = torch.tensor(candidate_ids_list, dtype=torch.long, device=EMB_DEV)

    # Show diff vs SOTA
    sota_ids = torch.tensor(NEW_SOTA_IDS, dtype=torch.long, device=EMB_DEV)
    diffs = (candidate_ids != sota_ids).nonzero().squeeze(-1).tolist()
    log(f"Diff vs new SOTA: {len(diffs)} positions: {diffs}")
    for pos in diffs:
        log(f"  pos{pos}: {NEW_SOTA_IDS[pos]} ({tokenizer.decode([NEW_SOTA_IDS[pos]])!r}) "
            f"→ {candidate_ids_list[pos]} ({tokenizer.decode([candidate_ids_list[pos]])!r})")

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

    def hotflip_step_batched(current_ids, ref_completions):
        current_ids = current_ids.to(EMB_DEV)
        grad_suffixes = SUFFIXES[:BATCH_SIZE]
        grad_refs = ref_completions[:BATCH_SIZE]
        prefix_emb = embed_fn(current_ids).float().detach().requires_grad_(True)
        batch_emb, meta, T_max = build_batch(prefix_emb, grad_suffixes, grad_refs)
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

    # Verify candidate CE
    log("Verifying candidate CE...")
    verified_ce = compute_ce_discrete_batched(candidate_ids, SUFFIXES, ref_completions)
    log(f"Verified CE: {verified_ce:.6f} (Exp52 reported: {EXP52_CE})")
    beats_sota = verified_ce < NEW_SOTA_CE
    log(f"Beats new SOTA ({NEW_SOTA_CE:.6f}): {beats_sota}")

    notify("Exp53 started",
           f"verified CE={verified_ce:.5f} | {'NEW SOTA!' if beats_sota else f'delta={verified_ce-NEW_SOTA_CE:+.5f}'}")

    # HotFlip continuation
    log(f"\n--- HotFlip: {HOTFLIP_STEPS} steps, topk={HF_TOPK} ---")
    current_ids = candidate_ids.clone()
    current_ce  = verified_ce
    hotflip_log = [current_ce]
    overall_best_ce  = current_ce
    overall_best_ids = candidate_ids.tolist()

    t_start = time.time()
    for step in range(HOTFLIP_STEPS):
        new_ids, new_ce = hotflip_step_batched(current_ids, ref_completions)
        improved = new_ce < current_ce
        if improved:
            current_ids = new_ids
            current_ce  = new_ce
            if current_ce < overall_best_ce:
                overall_best_ce  = current_ce
                overall_best_ids = current_ids.tolist()
                notify(f"Exp53 step {step} NEW BEST",
                       f"CE={current_ce:.5f}  {'*** BEATS SOTA ***' if current_ce < NEW_SOTA_CE else ''}")
        hotflip_log.append(current_ce)
        if step % 5 == 0 or step == HOTFLIP_STEPS - 1:
            toks = tokenizer.decode(current_ids.cpu().tolist())
            elapsed = time.time() - t_start
            log(f"  [{step:3d}/{HOTFLIP_STEPS}] CE={current_ce:.5f}  {'↓' if improved else '–'}  {toks!r}  {elapsed:.0f}s")
        if step % 5 == 0:
            ckpt = {"hf_step": step, "best_ce": current_ce,
                    "best_ids": current_ids.tolist(),
                    "verified_start_ce": verified_ce}
            with open(CKPT_PATH, "w") as f:
                json.dump(ckpt, f)

    final_text = tokenizer.decode(overall_best_ids)
    log(f"\n=== FINAL SUMMARY ===")
    log(f"  New SOTA (Exp46):  CE={NEW_SOTA_CE:.6f}")
    log(f"  Exp52 candidate:   CE={EXP52_CE:.6f}")
    log(f"  Verified start:    CE={verified_ce:.6f}")
    log(f"  After {HOTFLIP_STEPS} more HF steps: CE={overall_best_ce:.6f}  {'*** NEW SOTA ***' if overall_best_ce < NEW_SOTA_CE else ''}")
    log(f"  Best text: {final_text!r}")
    log(f"  Best IDs: {overall_best_ids}")

    results = {
        "experiment": "steer-001-exp53-tierra-verify",
        "model": MODEL_NAME,
        "reference_prefix": REF_PREFIX,
        "prefix_len": PREFIX_LEN,
        "hf_topk": HF_TOPK,
        "n_hotflip_steps": HOTFLIP_STEPS,
        "candidate_text": CANDIDATE_TEXT,
        "candidate_ids": candidate_ids_list,
        "exp52_ce": EXP52_CE,
        "verified_ce": verified_ce,
        "new_sota_ce": NEW_SOTA_CE,
        "old_sota_ce": OLD_SOTA_CE,
        "overall_best_ce": overall_best_ce,
        "overall_best_ids": overall_best_ids,
        "overall_best_text": final_text,
        "hotflip_log": hotflip_log,
        "beats_new_sota": overall_best_ce < NEW_SOTA_CE,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {OUT_PATH}")

    notify("Exp53 complete",
           f"CE={overall_best_ce:.5f}  {'BEATS SOTA' if overall_best_ce < NEW_SOTA_CE else f'delta={overall_best_ce-NEW_SOTA_CE:+.5f}'}")

except Exception as e:
    notify("Exp53 FAILED", str(e))
    raise
