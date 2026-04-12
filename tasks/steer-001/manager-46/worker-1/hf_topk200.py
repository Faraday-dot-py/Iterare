"""
Experiment 46: HotFlip topk=200 from SOTA prefix

Motivation:
- HotFlip with topk=50 converged at step 17 for seed=2 (Exp26 SOTA=0.6044).
- Greedy search with topk=50 may have missed better candidates that rank 51-200.
- This experiment runs 100 HotFlip steps from the known SOTA IDs using topk=200,
  giving 4× more candidates per position per step.
- No soft opt needed — we start from the proven best discrete prefix.
- Also tries beam=2: keep 2 best candidate prefixes simultaneously across steps.

Two runs:
  Run A: standard greedy HF, topk=200, 100 steps (from SOTA)
  Run B: beam HF (beam_width=2), topk=50, 100 steps (from SOTA)
         — keep top-2 prefix candidates, expand each, keep best-2

Config: SOTA IDs hardcoded, PREFIX_LEN=16, HF_TOPK_A=200, HF_STEPS=100

Timing estimate:
  Run A: 100 × 16 × 200 cands ≈ 4× slower per step than Exp26 HF
         Exp26 HF was ~130s/35 steps = 3.7s/step × 100 × 4 ≈ 25 min
  Run B: 100 steps × beam_width=2 × 16 × 50 ≈ 2× baseline = ~12 min
  Total: ~37 min
Output: /home/jovyan/steer001_hf_topk200.json
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
MODEL_NAME  = "google/gemma-2-2b-it"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.bfloat16
REF_PREFIX  = "Talk only about cats."
PLACEHOLDER = "SOFTPREFIX"
PREFIX_LEN  = 16
HF_STEPS    = 100
HF_TOPK_A   = 200   # Run A: expanded topk greedy
HF_TOPK_B   = 50    # Run B: beam width=2
BEAM_WIDTH  = 2
EARLY_K     = 32
EARLY_WEIGHT = 3.0
SOTA_CE     = 0.60437
SOTA_IDS    = [50105, 111, 133522, 222115, 24539, 244842, 10358, 235559,
               73815, 242580, 231898, 2976, 55135, 5598, 31459, 19493]
OUT_PATH    = Path("/home/jovyan/steer001_hf_topk200.json")

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
    log("=== Exp46: HotFlip topk=200 + beam=2 from SOTA prefix ===")
    log(f"CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")

    log(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map="cuda:0")
    model.eval()
    log(f"Model loaded in {time.time()-t0:.1f}s | {gpu_mem_str()}")
    notify("Exp46 started", f"HF topk=200 + beam=2 from SOTA | SOTA={SOTA_CE:.4f}")

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
        """Single greedy HotFlip step, returns (new_ids, new_ce)."""
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
    sota_ids_tensor = torch.tensor(SOTA_IDS, dtype=torch.long, device=EMB_DEV)
    sota_text = tokenizer.decode(SOTA_IDS)
    log(f"SOTA prefix: {sota_text!r}  CE={SOTA_CE:.5f}")

    # ── Run A: topk=200 greedy HotFlip ────────────────────────────────────────
    log(f"\n{'='*60}")
    log(f"=== Run A: Greedy HotFlip, topk={HF_TOPK_A}, {HF_STEPS} steps ===")
    log(f"{'='*60}")

    current_ids_a = sota_ids_tensor.clone()
    current_ce_a  = compute_ce_discrete_batched(current_ids_a, SUFFIXES, ref_completions)
    log(f"Start CE: {current_ce_a:.6f}")
    hf_log_a = [current_ce_a]

    t_a = time.time()
    for step in range(HF_STEPS):
        new_ids, new_ce = hotflip_step(current_ids_a, ref_completions, HF_TOPK_A)
        improved = new_ce < current_ce_a
        if improved:
            current_ids_a = new_ids
            current_ce_a  = new_ce
        hf_log_a.append(current_ce_a)
        if step % 10 == 0 or step == HF_STEPS - 1:
            toks = tokenizer.decode(current_ids_a.cpu().tolist())
            log(f"  [{step:3d}/{HF_STEPS}] CE={current_ce_a:.6f}  {'↓' if improved else '–'}  {toks!r}")
        if not improved and step > 10:
            # Check if we've been stuck for 10+ steps
            if all(x == hf_log_a[-1] for x in hf_log_a[-10:]):
                log(f"  Converged at step {step}, stopping Run A early.")
                break

    t_a_elapsed = time.time() - t_a
    best_text_a = tokenizer.decode(current_ids_a.cpu().tolist())
    log(f"Run A done: CE={current_ce_a:.6f} in {t_a_elapsed:.0f}s")
    notify(f"Exp46 Run A done",
           f"topk=200: CE={current_ce_a:.6f}  {'BEATS SOTA' if current_ce_a < SOTA_CE else f'delta={current_ce_a-SOTA_CE:+.6f}'}")

    # ── Run B: beam HotFlip ────────────────────────────────────────────────────
    log(f"\n{'='*60}")
    log(f"=== Run B: Beam HotFlip, beam={BEAM_WIDTH}, topk={HF_TOPK_B}, {HF_STEPS} steps ===")
    log(f"{'='*60}")

    # Initialize beam with SOTA prefix
    beam = [(SOTA_CE, sota_ids_tensor.clone())]
    # Verify score
    beam[0] = (compute_ce_discrete_batched(beam[0][1], SUFFIXES, ref_completions), beam[0][1])
    log(f"Start CE: {beam[0][0]:.6f}")
    hf_log_b = [beam[0][0]]

    t_b = time.time()
    for step in range(HF_STEPS):
        candidates = []
        for ce_beam, ids_beam in beam:
            new_ids, new_ce = hotflip_step(ids_beam, ref_completions, HF_TOPK_B)
            candidates.append((new_ce, new_ids))
            # Also keep existing beam members (in case no improvement found)
            candidates.append((ce_beam, ids_beam))

        # Deduplicate by IDs and keep best BEAM_WIDTH
        seen = set()
        unique_cands = []
        for ce_c, ids_c in sorted(candidates, key=lambda x: x[0]):
            key = tuple(ids_c.tolist())
            if key not in seen:
                seen.add(key)
                unique_cands.append((ce_c, ids_c))
            if len(unique_cands) == BEAM_WIDTH:
                break
        beam = unique_cands

        best_ce_b = beam[0][0]
        hf_log_b.append(best_ce_b)
        if step % 10 == 0 or step == HF_STEPS - 1:
            toks = tokenizer.decode(beam[0][1].cpu().tolist())
            log(f"  [{step:3d}/{HF_STEPS}] best CE={best_ce_b:.6f}  {toks!r}")
        if step > 10 and all(x == hf_log_b[-1] for x in hf_log_b[-10:]):
            log(f"  Beam converged at step {step}, stopping Run B early.")
            break

    t_b_elapsed = time.time() - t_b
    best_ce_b   = beam[0][0]
    best_ids_b  = beam[0][1]
    best_text_b = tokenizer.decode(best_ids_b.cpu().tolist())
    log(f"Run B done: CE={best_ce_b:.6f} in {t_b_elapsed:.0f}s")
    notify(f"Exp46 Run B done",
           f"beam=2: CE={best_ce_b:.6f}  {'BEATS SOTA' if best_ce_b < SOTA_CE else f'delta={best_ce_b-SOTA_CE:+.6f}'}")

    # ── Final summary ─────────────────────────────────────────────────────────
    overall_best_ce = min(current_ce_a, best_ce_b)
    log(f"\n=== FINAL SUMMARY ===")
    log(f"  SOTA:   CE={SOTA_CE:.6f}")
    log(f"  Run A (topk=200 greedy): CE={current_ce_a:.6f}  {best_text_a!r}")
    log(f"  Run B (beam=2):          CE={best_ce_b:.6f}  {best_text_b!r}")
    log(f"  Overall best: CE={overall_best_ce:.6f}")

    results = {
        "experiment": "steer-001-exp46-hf-topk200-beam",
        "model": MODEL_NAME,
        "reference_prefix": REF_PREFIX,
        "prefix_len": PREFIX_LEN,
        "hf_steps": HF_STEPS,
        "sota_ce": SOTA_CE,
        "sota_ids": SOTA_IDS,
        "run_a": {
            "description": f"greedy HotFlip topk={HF_TOPK_A}",
            "hf_topk": HF_TOPK_A,
            "best_ce": current_ce_a,
            "best_text": best_text_a,
            "best_ids": current_ids_a.tolist(),
            "hf_log": hf_log_a,
            "elapsed_seconds": t_a_elapsed,
        },
        "run_b": {
            "description": f"beam HotFlip beam_width={BEAM_WIDTH} topk={HF_TOPK_B}",
            "beam_width": BEAM_WIDTH,
            "hf_topk": HF_TOPK_B,
            "best_ce": best_ce_b,
            "best_text": best_text_b,
            "best_ids": best_ids_b.tolist(),
            "hf_log": hf_log_b,
            "elapsed_seconds": t_b_elapsed,
        },
        "overall_best_ce": overall_best_ce,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {OUT_PATH}")

    notify("Exp46 complete",
           f"topk=200: {current_ce_a:.6f}  beam=2: {best_ce_b:.6f}  "
           f"{'BEATS SOTA' if overall_best_ce < SOTA_CE else f'best delta={overall_best_ce-SOTA_CE:+.6f}'}")

except Exception as e:
    notify("Exp46 FAILED", str(e))
    raise
