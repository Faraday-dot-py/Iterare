"""
Experiment 40: First-token keyword probability objective (no teacher forcing)

Motivation:
- Current objective: CE against 12 specific pre-generated completions.
  The prefix implicitly optimizes for reproducing those exact 12 texts.
  Problem: it may overfit to these completions, and it rewards copying
  specific tokens rather than genuinely steering toward cat content.
- New objective: directly maximize log P(any cat-related token | prefix + suffix)
  where cat_token_set = top-200 vocabulary tokens most similar to "cats"
  by cosine similarity in embedding space.
- This is evaluated at the FIRST generated token position (next-token after
  prefix + context) — no teacher forcing, no specific completion required.
- The objective is: for each suffix, the model should assign high probability
  to ANY cat-relevant token when it first generates. This is more generalizable
  than matching a single reference completion.
- A prefix scoring well here will steer the model toward cat content even on
  completely unseen prompts.

Config: seeds=[0,1,2], PREFIX_LEN=16, SOFT_STEPS=300, HOTFLIP_STEPS=35,
        HF_TOPK=50, cosine LR 0.01→0.001, BATCH_SIZE=12,
        cat_topk=200 (top-200 vocab tokens by embedding sim to "cats")

Timing estimate: 3 × (~511s soft + 35×111s HF) ≈ 3×4396s ≈ 13188s (~3.7h)

Output: /home/jovyan/steer001_keyword_objective.json
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
MODEL_NAME  = "google/gemma-2-2b-it"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.bfloat16
REF_PREFIX  = "Talk only about cats."   # used only to generate ref completions
PLACEHOLDER = "SOFTPREFIX"
PREFIX_LEN  = 16
SOFT_STEPS  = 300
HOTFLIP_STEPS = 35
HF_TOPK     = 50
BATCH_SIZE  = 12
LR_MAX      = 0.01
LR_MIN      = 0.001
EARLY_K     = 32
EARLY_WEIGHT= 3.0
CAT_TOPK    = 200   # size of cat_token_set
SEEDS       = [0, 1, 2]
OUT_PATH    = Path("/home/jovyan/steer001_keyword_objective.json")
CKPT_PATH   = Path("/home/jovyan/steer001_keyword_objective_ckpt.json")

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
    log("=== Exp 40: First-token keyword probability objective ===")
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
    notify("Exp40 started", f"Keyword objective seeds={SEEDS} | {gpu_mem_str()}")

    PLACEHOLDER_IDS = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)
    embed_fn = model.get_input_embeddings()
    EMB_DIM  = embed_fn.weight.shape[1]
    EMB_DEV  = embed_fn.weight.device

    # ── Build cat_token_set ───────────────────────────────────────────────────
    # Top-CAT_TOPK vocabulary tokens by cosine similarity to "cats" embedding.
    cats_token_id = tokenizer.encode("cats", add_special_tokens=False)[0]
    log(f"\n'cats' token ID: {cats_token_id}")
    with torch.no_grad():
        cats_emb = embed_fn.weight[cats_token_id].float()
        cats_emb_norm = F.normalize(cats_emb, dim=0)
        W_norm = F.normalize(embed_fn.weight.float(), dim=-1)
        sims = W_norm @ cats_emb_norm
        cat_token_ids = sims.topk(CAT_TOPK).indices.to(EMB_DEV)

    log(f"Cat token set (top {CAT_TOPK} by similarity to 'cats'):")
    sample_decoded = [tokenizer.decode([t.item()]) for t in cat_token_ids[:20]]
    log(f"  Sample: {sample_decoded}")
    log(f"  Includes 'cats' itself: {cats_token_id in cat_token_ids.tolist()}")

    # ── Template helpers ──────────────────────────────────────────────────────

    def chat_ids(messages):
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        return tokenizer.encode(text, add_special_tokens=False)

    def find_subseq(seq, sub):
        for i in range(len(seq) - len(sub) + 1):
            if seq[i:i+len(sub)] == sub: return i
        return None

    def get_template_split(suffix_text):
        """Prefix in user message (standard placement)."""
        msgs = [{"role": "user", "content": f"{PLACEHOLDER}\n\n{suffix_text}"}]
        ids_list = chat_ids(msgs)
        ph_start = find_subseq(ids_list, PLACEHOLDER_IDS)
        if ph_start is None:
            ph_start, ph_len = 1, 0
        else:
            ph_len = len(PLACEHOLDER_IDS)
        ids = torch.tensor(ids_list, dtype=torch.long)
        return ids[:ph_start], ids[ph_start + ph_len:]

    # ── Keyword objective: batch builder WITHOUT ref completions ──────────────
    # We only need up to the model-turn start; then we read logits at the
    # first generation position to measure P(any cat token).

    def build_batch_keyword(soft_prefix_LD, suffix_texts):
        """Build batch without appending ref completions."""
        seqs, comp_starts = [], []
        for suf in suffix_texts:
            pre_ids, post_ids = get_template_split(suf)
            with torch.no_grad():
                pre_emb  = embed_fn(pre_ids.unsqueeze(0).to(EMB_DEV))
                post_emb = embed_fn(post_ids.unsqueeze(0).to(EMB_DEV))
            soft_1LD = soft_prefix_LD.to(dtype=pre_emb.dtype, device=pre_emb.device).unsqueeze(0)
            # seq = [context_before_prefix, prefix, context_after_prefix]
            seq = torch.cat([pre_emb, soft_1LD, post_emb], dim=1)
            comp_start = pre_emb.shape[1] + PREFIX_LEN + post_emb.shape[1]
            seqs.append(seq)
            comp_starts.append(comp_start)
        T_max = max(s.shape[1] for s in seqs)
        padded = []
        for seq in seqs:
            pad_len = T_max - seq.shape[1]
            if pad_len:
                pad = torch.zeros(1, pad_len, EMB_DIM, device=seq.device, dtype=seq.dtype)
                seq = torch.cat([seq, pad], dim=1)
            padded.append(seq)
        return torch.cat(padded, dim=0).to(EMB_DEV), comp_starts, T_max

    def compute_keyword_loss(logits, comp_starts, T_max):
        """
        For each sequence, at the first generation position (comp_start - 1),
        compute -log P(any cat token is next).
        = -logsumexp(log_softmax(logits)[cat_token_ids])
        """
        total = torch.tensor(0.0, device=logits.device)
        count = 0
        for b, comp_start in enumerate(comp_starts):
            pos = comp_start - 1   # logits at this pos predict token at comp_start
            if pos >= T_max or pos < 0:
                continue
            log_probs = torch.log_softmax(logits[b, pos], dim=-1)
            cat_log_probs = log_probs[cat_token_ids]
            log_p_any_cat = torch.logsumexp(cat_log_probs, dim=0)
            total = total - log_p_any_cat
            count += 1
        return total / count if count > 0 else total

    def compute_keyword_soft_batched(soft_prefix_LD, suffix_texts):
        batch_emb, comp_starts, T_max = build_batch_keyword(soft_prefix_LD, suffix_texts)
        logits = model(inputs_embeds=batch_emb).logits
        return compute_keyword_loss(logits, comp_starts, T_max)

    def compute_keyword_discrete(prefix_ids_L, suffix_texts):
        """Discrete version: keyword loss (no teacher forcing)."""
        prefix_ids_L = prefix_ids_L.to(EMB_DEV)
        with torch.no_grad():
            soft = embed_fn(prefix_ids_L)
            batch_emb, comp_starts, T_max = build_batch_keyword(soft, suffix_texts)
            logits = model(inputs_embeds=batch_emb).logits
            loss = compute_keyword_loss(logits, comp_starts, T_max)
        return loss.item()

    # For HotFlip we still need a scalar CE metric for comparison
    # We'll also compute the CE-against-ref-completions for comparison to SOTA,
    # but optimize with keyword loss.

    def build_batch_ce(soft_prefix_LD, suffix_texts, ref_completions):
        """Original CE batch builder for comparison metrics."""
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
            batch_emb, meta, T_max = build_batch_ce(soft, suffix_texts, ref_completions)
            logits = model(inputs_embeds=batch_emb).logits
            loss = compute_ce_from_batch(logits, meta, T_max)
        return loss.item()

    def project_to_tokens(soft_LD):
        with torch.no_grad():
            W  = embed_fn.weight.to(device=soft_LD.device, dtype=soft_LD.dtype)
            sn = F.normalize(soft_LD, dim=-1)
            wn = F.normalize(W, dim=-1)
            return (sn @ wn.T).argmax(dim=-1)

    def st_project(soft_prefix_LD):
        W = embed_fn.weight.to(device=soft_prefix_LD.device, dtype=soft_prefix_LD.dtype)
        sn   = F.normalize(soft_prefix_LD, dim=-1)
        wn   = F.normalize(W, dim=-1)
        ids  = (sn @ wn.T).argmax(dim=-1)
        with torch.no_grad():
            proj_emb = embed_fn(ids).to(dtype=soft_prefix_LD.dtype)
        return soft_prefix_LD + (proj_emb - soft_prefix_LD).detach(), ids

    def cosine_lr(step, total_steps):
        return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * step / total_steps))

    def hotflip_step_batched(current_ids, suffix_texts, ref_completions, prefix_len):
        """HotFlip using KEYWORD loss as optimization target."""
        current_ids = current_ids.to(EMB_DEV)
        prefix_emb  = embed_fn(current_ids).float().detach().requires_grad_(True)
        # Compute gradient using keyword objective
        batch_emb, comp_starts, T_max = build_batch_keyword(prefix_emb, suffix_texts)
        logits = model(inputs_embeds=batch_emb).logits
        loss   = compute_keyword_loss(logits, comp_starts, T_max)
        loss.backward()
        grad = prefix_emb.grad

        best_ids    = current_ids.clone()
        best_kw     = compute_keyword_discrete(current_ids, suffix_texts)
        W = embed_fn.weight.float()

        for pos in range(prefix_len):
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
                kw_k = compute_keyword_discrete(trial_ids, suffix_texts)
                if kw_k < best_kw:
                    best_kw  = kw_k
                    best_ids = trial_ids.clone()
        return best_ids, best_kw

    # ── Reference completions (for CE comparison metric only) ─────────────────
    log(f"Generating reference completions for CE comparison...")
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
    log(f"Reference completions ready (used only for comparison CE metric).")

    ref_ids_set = set(tokenizer.encode(REF_PREFIX, add_special_tokens=False))

    # Sanity check: what is keyword loss for the SOTA prefix?
    sota_ids = torch.tensor([50105, 111, 133522, 222115, 24539, 244842, 10358, 235559,
                              73815, 242580, 231898, 2976, 55135, 5598, 31459, 19493],
                             dtype=torch.long, device=EMB_DEV)
    sota_kw = compute_keyword_discrete(sota_ids, SUFFIXES)
    sota_ce = compute_ce_discrete_batched(sota_ids, SUFFIXES, ref_completions)
    log(f"\nSOTA prefix (Exp26 seed=2): keyword_loss={sota_kw:.5f}  CE={sota_ce:.5f}")

    # ── Multi-seed loop ───────────────────────────────────────────────────────
    overall_best_kw   = float('inf')
    overall_best_ids  = None
    overall_best_seed = None
    all_runs = []

    for seed_idx, seed in enumerate(SEEDS):
        log(f"\n{'='*60}")
        log(f"=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===")
        log(f"{'='*60}")

        with torch.no_grad():
            emb_mean = embed_fn.weight.mean(0)
            emb_std  = embed_fn.weight.std(0) * 0.1
        torch.manual_seed(seed)
        soft_prefix = (emb_mean.unsqueeze(0).repeat(PREFIX_LEN, 1)
                       + torch.randn(PREFIX_LEN, EMB_DIM, device=EMB_DEV, dtype=DTYPE)
                       * emb_std.unsqueeze(0)).detach().float().requires_grad_(True)
        optimizer = torch.optim.Adam([soft_prefix], lr=LR_MAX)

        log(f"\n--- ST Soft Opt (keyword objective): {SOFT_STEPS} steps ---")
        kw_log  = []
        best_kw_soft         = float('inf')
        best_soft_snapshot   = soft_prefix.data.clone()

        t_soft_start = time.time()
        for step in range(SOFT_STEPS):
            lr = cosine_lr(step, SOFT_STEPS)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            optimizer.zero_grad()
            st_emb, _ = st_project(soft_prefix)
            # KEYWORD objective — no ref_completions needed
            loss = compute_keyword_soft_batched(st_emb, SUFFIXES[:BATCH_SIZE])
            loss.backward()
            optimizer.step()
            kw_val = loss.item()
            kw_log.append(kw_val)
            if kw_val < best_kw_soft:
                best_kw_soft = kw_val
                best_soft_snapshot = soft_prefix.data.clone()
            if step % 50 == 0 or step == SOFT_STEPS - 1:
                log(f"  [{step:4d}/{SOFT_STEPS}] KW={kw_val:.5f}  best={best_kw_soft:.5f}  "
                    f"lr={lr:.5f}  elapsed={time.time()-t_soft_start:.0f}s")

        t_soft = time.time() - t_soft_start
        log(f"Soft done: best_kw={best_kw_soft:.5f} in {t_soft:.0f}s")

        projected_ids  = project_to_tokens(best_soft_snapshot.to(DTYPE))
        projected_text = tokenizer.decode(projected_ids.cpu().tolist())
        proj_kw = compute_keyword_discrete(projected_ids, SUFFIXES)
        proj_ce = compute_ce_discrete_batched(projected_ids, SUFFIXES, ref_completions)
        log(f"Projected: {projected_text!r}  KW={proj_kw:.5f}  CE(ref)={proj_ce:.5f}")

        log(f"\n--- HotFlip (keyword objective): {HOTFLIP_STEPS} steps ---")
        current_ids = projected_ids.clone().to(EMB_DEV)
        current_kw  = proj_kw
        hf_kw_log   = [current_kw]
        hf_ce_log   = [proj_ce]   # CE tracked as secondary metric

        t_hf_start = time.time()
        for step in range(HOTFLIP_STEPS):
            new_ids, new_kw = hotflip_step_batched(current_ids, SUFFIXES, ref_completions, PREFIX_LEN)
            improved = new_kw < current_kw
            if improved:
                current_ids = new_ids
                current_kw  = new_kw
            ce_now = compute_ce_discrete_batched(current_ids, SUFFIXES, ref_completions)
            hf_kw_log.append(current_kw)
            hf_ce_log.append(ce_now)
            if step % 5 == 0 or step == HOTFLIP_STEPS - 1:
                toks = tokenizer.decode(current_ids.cpu().tolist())
                log(f"  [{step:3d}/{HOTFLIP_STEPS}] KW={current_kw:.5f}  CE={ce_now:.5f}  "
                    f"{'↓' if improved else '–'}  {toks!r}")
            if step % 5 == 0:
                ckpt = {"seed": seed, "hf_step": step, "best_kw": current_kw,
                        "current_ce": ce_now, "overall_best_kw": overall_best_kw}
                with open(CKPT_PATH, "w") as f:
                    json.dump(ckpt, f)

        t_hf   = time.time() - t_hf_start
        final_kw   = current_kw
        final_ce   = compute_ce_discrete_batched(current_ids, SUFFIXES, ref_completions)
        final_text = tokenizer.decode(current_ids.cpu().tolist())
        log(f"HotFlip done: KW={final_kw:.5f}  CE={final_ce:.5f} in {t_hf:.0f}s")

        if final_kw < overall_best_kw:
            overall_best_kw   = final_kw
            overall_best_ids  = current_ids.clone()
            overall_best_seed = seed
            notify("Exp40 NEW BEST",
                   f"seed={seed}  KW={final_kw:.4f}  CE={final_ce:.4f}  "
                   f"{'CE BEATS SOTA' if final_ce < 0.60437 else ''}  {final_text[:50]!r}")
        else:
            notify(f"Exp40 seed={seed} done",
                   f"KW={final_kw:.4f}  CE={final_ce:.4f}  SOTA_CE=0.6044")

        all_runs.append({
            "seed": seed,
            "best_kw_soft": best_kw_soft,
            "projection_kw": proj_kw,
            "projection_ce": proj_ce,
            "final_kw": final_kw,
            "final_ce": final_ce,
            "projected_text": projected_text,
            "final_text": final_text,
            "final_ids": current_ids.tolist(),
            "kw_log": kw_log,
            "hf_kw_log": hf_kw_log,
            "hf_ce_log": hf_ce_log,
            "timing": {"soft_seconds": t_soft, "hotflip_seconds": t_hf},
        })

    overall_best_text = tokenizer.decode(overall_best_ids.cpu().tolist()) if overall_best_ids is not None else ""
    overall_final_ce  = compute_ce_discrete_batched(overall_best_ids, SUFFIXES, ref_completions) if overall_best_ids is not None else None

    log(f"\n=== FINAL SUMMARY ===")
    log(f"  Objective: keyword log-prob (top-{CAT_TOPK} cat tokens)")
    log(f"  SOTA keyword_loss: {sota_kw:.5f}  SOTA CE: {sota_ce:.5f}")
    log(f"  Overall best: seed={overall_best_seed}  KW={overall_best_kw:.5f}  CE={overall_final_ce:.5f}")
    log(f"  Best prefix:  {overall_best_text!r}")
    for r in all_runs:
        log(f"  seed={r['seed']}: KW={r['final_kw']:.5f}  CE={r['final_ce']:.5f}  {r['final_text'][:50]!r}")

    if overall_final_ce is not None and overall_final_ce < 0.60437:
        notify("Exp40 NEW SOTA (CE)", f"CE={overall_final_ce:.4f}  KW={overall_best_kw:.4f}  beats 0.6044")
    else:
        notify("Exp40 complete", f"best KW={overall_best_kw:.4f}  CE={overall_final_ce:.4f}")

    results = {
        "experiment": "steer-001-exp40-keyword-objective",
        "model": MODEL_NAME,
        "reference_prefix": REF_PREFIX,
        "prefix_len": PREFIX_LEN,
        "seeds": SEEDS,
        "n_soft_steps": SOFT_STEPS,
        "n_hotflip_steps": HOTFLIP_STEPS,
        "hf_topk": HF_TOPK,
        "cat_topk": CAT_TOPK,
        "cats_token_id": cats_token_id,
        "method": "st_cosine_anneal_best_prefix_keyword_obj",
        "sota_keyword_loss": sota_kw,
        "sota_ce": sota_ce,
        "overall_best_seed": overall_best_seed,
        "overall_best_kw": overall_best_kw,
        "overall_best_ce": overall_final_ce,
        "overall_best_text": overall_best_text,
        "overall_best_ids": overall_best_ids.tolist() if overall_best_ids is not None else [],
        "runs": all_runs,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {OUT_PATH}")
    log("=== DONE ===")

except Exception as e:
    notify("Exp40 FAILED", str(e))
    raise
