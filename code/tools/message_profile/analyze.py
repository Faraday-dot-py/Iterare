"""
Analysis pipeline — operates entirely on embeddings + metadata.
No message text is accessed here.

Analyses:
  1. Behavioral stats      — message counts, timing, platform, length (from anonymized meta)
  2. UMAP + HDBSCAN        — 2D projection + topic clusters
  3. Temporal trajectory   — monthly centroid drift in embedding space
  4. Affect probing        — cosine similarity to reference affect phrases
  5. Big Five proxy         — cosine similarity to Big Five reference phrases
  6. Style mirror          — per-conversation similarity between YOU and PERSON_X embeddings
  7. Lexical features      — message length stats, question rate, exclamation rate (from text_anon)
"""

import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Reference phrases for affect + Big Five probing
# (embedded at analysis time via the same model)
# ---------------------------------------------------------------------------

AFFECT_REFS = {
    "positive":   ["I'm so happy", "This is great", "I love this", "awesome", "excited"],
    "negative":   ["I'm upset", "this is terrible", "I hate this", "frustrated", "sad"],
    "neutral":    ["okay", "sure", "alright", "got it", "fine"],
    "energetic":  ["let's go", "hype", "can't wait", "fire", "let's do it"],
    "withdrawn":  ["idk", "whatever", "nvm", "doesn't matter", "not really"],
    "playful":    ["lol", "haha", "😂", "lmao", "that's funny"],
}

BIG_FIVE_REFS = {
    "openness":          ["I love exploring new ideas", "I enjoy abstract thinking",
                          "curious about the world", "I like trying new things", "creative"],
    "conscientiousness": ["I finish what I start", "I keep things organized",
                          "I follow through on plans", "disciplined", "I meet deadlines"],
    "extraversion":      ["I love being around people", "let's hang out", "party",
                          "I get energy from socializing", "outgoing"],
    "agreeableness":     ["I want to help you", "I care about how you feel",
                          "let's compromise", "I understand", "supportive"],
    "neuroticism":       ["I'm anxious about this", "I'm stressed",
                          "I can't stop worrying", "overwhelmed", "I feel insecure"],
}


def _encode_texts_automodel(texts: list[str], model_name: str) -> np.ndarray:
    """Encode texts using AutoTokenizer + AutoModel (no sentence_transformers needed)."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().float().numpy()


def _embed_refs(ref_dict: dict[str, list[str]], model_name_or_obj) -> dict[str, np.ndarray]:
    """Returns {label: mean_embedding} for each reference category.
    model_name_or_obj: str model name (uses AutoModel) or sentence_transformers model."""
    all_texts, labels = [], []
    for label, phrases in ref_dict.items():
        all_texts.extend(phrases)
        labels.extend([label] * len(phrases))

    if isinstance(model_name_or_obj, str):
        vecs = _encode_texts_automodel(all_texts, model_name_or_obj)
    else:
        vecs = model_name_or_obj.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)

    out = {}
    for label in ref_dict:
        mask = np.array([l == label for l in labels])
        mean = vecs[mask].mean(axis=0)
        mean /= np.linalg.norm(mean) + 1e-9
        out[label] = mean
    return out


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# 1. Behavioral stats (from anonymized message list — no raw text used)
# ---------------------------------------------------------------------------

def behavioral_stats(messages: list[dict]) -> dict:
    """
    Input: anonymized message list (text_anon field used only for length/punctuation stats).
    Returns a stats dict with no message text.
    """
    user_msgs = [m for m in messages if m["sender_anon"] == "YOU"]
    friend_msgs = [m for m in messages if m["sender_anon"] != "YOU"]

    def time_features(msgs):
        hours = [m["timestamp"].hour for m in msgs]
        days  = [m["timestamp"].weekday() for m in msgs]  # 0=Mon
        hour_dist  = np.bincount(hours, minlength=24).tolist()
        day_dist   = np.bincount(days,  minlength=7).tolist()
        return {"hour_distribution": hour_dist, "day_distribution": day_dist}

    def text_features(msgs):
        lengths = [len(m["text_anon"].split()) for m in msgs]
        question_rate = sum(1 for m in msgs if "?" in m["text_anon"]) / max(1, len(msgs))
        exclaim_rate  = sum(1 for m in msgs if "!" in m["text_anon"]) / max(1, len(msgs))
        return {
            "mean_words":      float(np.mean(lengths)) if lengths else 0,
            "median_words":    float(np.median(lengths)) if lengths else 0,
            "p90_words":       float(np.percentile(lengths, 90)) if lengths else 0,
            "question_rate":   round(question_rate, 4),
            "exclaim_rate":    round(exclaim_rate, 4),
            "length_dist":     np.histogram(lengths, bins=[0,1,3,6,15,30,100,1000])[0].tolist(),
        }

    platform_dist = defaultdict(int)
    for m in user_msgs:
        platform_dist[m["platform"]] += 1

    monthly = defaultdict(int)
    for m in user_msgs:
        key = m["timestamp"].strftime("%Y-%m")
        monthly[key] += 1

    convo_sizes = defaultdict(int)
    for m in user_msgs:
        convo_sizes[m["conversation_id"]] += 1

    return {
        "total_messages":        len(messages),
        "user_messages":         len(user_msgs),
        "friend_messages":       len(friend_msgs),
        "conversations":         len(set(m["conversation_id"] for m in messages)),
        "platform_distribution": dict(platform_dist),
        "monthly_activity":      dict(sorted(monthly.items())),
        "top_conversations":     sorted(convo_sizes.items(), key=lambda x: -x[1])[:20],
        "user_timing":           time_features(user_msgs),
        "user_text":             text_features(user_msgs),
        "date_range": {
            "first": min(m["timestamp"] for m in messages).isoformat(),
            "last":  max(m["timestamp"] for m in messages).isoformat(),
        }
    }


# ---------------------------------------------------------------------------
# 2. UMAP + HDBSCAN
# ---------------------------------------------------------------------------

def cluster_embeddings(store, n_clusters: int = 12, perplexity: int = 40,
                        tsne_max_samples: int | None = None):
    """Returns (user_store, tsne_2d, labels, None). Uses t-SNE + KMeans (no numba).

    tsne_max_samples: if set, subsample user messages before t-SNE (visualization only).
    KMeans labels are then expanded back to full user_store size via nearest-centroid.
    """
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    user_store = store.filter(store.user_mask())
    embs = user_store.embeddings

    pca_dim = min(50, embs.shape[1], embs.shape[0] - 1)
    print(f"  PCA {embs.shape[1]}→{pca_dim} on {len(embs):,} user embeddings...")
    pca = PCA(n_components=pca_dim, random_state=4738)
    embs_pca = pca.fit_transform(embs)

    # Optionally subsample for t-SNE (O(n²) — impractical above ~30k)
    if tsne_max_samples and len(embs_pca) > tsne_max_samples:
        rng = np.random.default_rng(4738)
        idx = rng.choice(len(embs_pca), tsne_max_samples, replace=False)
        idx.sort()
        embs_tsne = embs_pca[idx]
        print(f"  Subsampled {len(embs_pca):,}→{tsne_max_samples:,} for t-SNE")
    else:
        idx = None
        embs_tsne = embs_pca

    print(f"  t-SNE {embs_tsne.shape[0]:,}→2 (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity,
                metric="cosine", random_state=4738, n_jobs=-1)
    tsne_2d_sub = tsne.fit_transform(embs_tsne)

    print(f"  KMeans clustering (k={n_clusters})...")
    km = KMeans(n_clusters=n_clusters, random_state=4738, n_init="auto")
    labels_sub = km.fit_predict(tsne_2d_sub)

    if idx is not None:
        # Cluster labels for the full set: re-run KMeans on PCA space
        # (km was fitted on 2D t-SNE — can't use it for 50D prediction)
        print(f"  KMeans on full PCA space for label assignment ...")
        km_full = KMeans(n_clusters=n_clusters, random_state=4738, n_init="auto")
        labels_full = km_full.fit_predict(embs_pca)

        # Build full t-SNE array: known positions from subsample, rest jittered from cluster mean
        tsne_2d_full = np.zeros((len(embs_pca), 2), dtype=np.float32)
        tsne_2d_full[idx] = tsne_2d_sub
        cluster_means_2d = np.array([
            tsne_2d_sub[labels_sub == k].mean(axis=0) if (labels_sub == k).any()
            else np.zeros(2)
            for k in range(n_clusters)
        ])
        unknown_mask = np.ones(len(embs_pca), dtype=bool)
        unknown_mask[idx] = False
        rng2 = np.random.default_rng(4739)
        noise = rng2.normal(0, 0.5, (unknown_mask.sum(), 2))
        tsne_2d_full[unknown_mask] = cluster_means_2d[labels_full[unknown_mask]] + noise
        tsne_2d = tsne_2d_full
        labels = labels_full
    else:
        tsne_2d = tsne_2d_sub
        labels = labels_sub

    print(f"  {n_clusters} clusters")
    return user_store, tsne_2d, labels, None


# ---------------------------------------------------------------------------
# 3. Temporal trajectory
# ---------------------------------------------------------------------------

def temporal_trajectory(store) -> dict:
    """Monthly centroid + variance in embedding space for user messages."""
    user_store = store.filter(store.user_mask())
    monthly: dict[str, list[int]] = defaultdict(list)
    for i, ts in enumerate(user_store.timestamps):
        monthly[ts.strftime("%Y-%m")].append(i)

    months = sorted(monthly.keys())
    centroids, variances, counts = [], [], []
    for month in months:
        idx = monthly[month]
        vecs = user_store.embeddings[idx]
        c = vecs.mean(axis=0)
        c /= np.linalg.norm(c) + 1e-9
        var = float(np.mean(np.linalg.norm(vecs - c, axis=1)))
        centroids.append(c)
        variances.append(var)
        counts.append(len(idx))

    # Drift: cosine distance between consecutive monthly centroids
    drifts = [0.0]
    for i in range(1, len(centroids)):
        drifts.append(1.0 - _cosine_sim(centroids[i], centroids[i - 1]))

    return {
        "months":     months,
        "counts":     counts,
        "variances":  variances,
        "drifts":     drifts,
        "centroids":  np.stack(centroids) if centroids else np.zeros((0, store.dim)),
    }


# ---------------------------------------------------------------------------
# 4 + 5. Affect + Big Five probing
# ---------------------------------------------------------------------------

def probe_traits(store, model_name: str) -> dict:
    """
    Compute per-message cosine similarity to affect and Big Five reference embeddings.
    Returns mean scores + monthly trajectories.
    Uses sentence_transformers if available, else AutoModel.
    """
    print(f"  Loading model for reference embedding: {model_name}")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        affect_refs = _embed_refs(AFFECT_REFS, model)
        big5_refs   = _embed_refs(BIG_FIVE_REFS, model)
        del model
    except Exception:
        print(f"  sentence_transformers unavailable, falling back to AutoModel")
        affect_refs = _embed_refs(AFFECT_REFS, model_name)
        big5_refs   = _embed_refs(BIG_FIVE_REFS, model_name)

    user_store = store.filter(store.user_mask())
    embs = user_store.embeddings  # normalized

    # Per-message scores
    affect_scores = {k: embs @ v for k, v in affect_refs.items()}   # (N,)
    big5_scores   = {k: embs @ v for k, v in big5_refs.items()}

    # Overall means
    affect_mean = {k: float(v.mean()) for k, v in affect_scores.items()}
    big5_mean   = {k: float(v.mean()) for k, v in big5_scores.items()}

    # Monthly trajectories
    monthly: dict[str, list[int]] = defaultdict(list)
    for i, ts in enumerate(user_store.timestamps):
        monthly[ts.strftime("%Y-%m")].append(i)

    months = sorted(monthly.keys())
    affect_monthly = {k: [] for k in AFFECT_REFS}
    big5_monthly   = {k: [] for k in BIG_FIVE_REFS}
    for month in months:
        idx = monthly[month]
        for k in AFFECT_REFS:
            affect_monthly[k].append(float(affect_scores[k][idx].mean()))
        for k in BIG_FIVE_REFS:
            big5_monthly[k].append(float(big5_scores[k][idx].mean()))

    return {
        "affect_mean":    affect_mean,
        "big5_mean":      big5_mean,
        "affect_monthly": affect_monthly,
        "big5_monthly":   big5_monthly,
        "months":         months,
    }


# ---------------------------------------------------------------------------
# 6. Style mirror
# ---------------------------------------------------------------------------

def style_mirror(store) -> dict:
    """
    Per-conversation: cosine similarity between user centroid and friend centroid.
    Only works for Instagram conversations where both sides are present.
    """
    convos = set(store.conversation_ids)
    results = {}
    for cid in convos:
        mask = np.array([c == cid for c in store.conversation_ids])
        sub = store.filter(mask)

        user_mask   = np.array([s == "YOU"   for s in sub.sender_anons])
        friend_mask = np.array([s != "YOU"   for s in sub.sender_anons])

        if user_mask.sum() < 3 or friend_mask.sum() < 3:
            continue

        u_centroid = sub.embeddings[user_mask].mean(axis=0)
        f_centroid = sub.embeddings[friend_mask].mean(axis=0)
        u_centroid /= np.linalg.norm(u_centroid) + 1e-9
        f_centroid /= np.linalg.norm(f_centroid) + 1e-9

        results[cid] = {
            "similarity":      round(_cosine_sim(u_centroid, f_centroid), 4),
            "user_messages":   int(user_mask.sum()),
            "friend_messages": int(friend_mask.sum()),
        }

    return results


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_analysis(store, messages: list[dict], out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Behavioral stats...")
    stats = behavioral_stats(messages)
    (out / "behavioral_stats.json").write_text(json.dumps(stats, indent=2, default=str))

    print("\n[2/5] Clustering (UMAP + HDBSCAN)...")
    user_store, umap_2d, labels, probs = cluster_embeddings(store)
    np.save(out / "umap_2d.npy", umap_2d)
    np.save(out / "cluster_labels.npy", labels)

    print("\n[3/5] Temporal trajectory...")
    traj = temporal_trajectory(store)
    traj_save = {k: v.tolist() if isinstance(v, np.ndarray) else v
                 for k, v in traj.items()}
    (out / "temporal_trajectory.json").write_text(json.dumps(traj_save, indent=2))

    print("\n[4/5] Affect + Big Five probing...")
    traits = probe_traits(store, store.model_name)
    (out / "trait_scores.json").write_text(json.dumps(traits, indent=2))

    print("\n[5/5] Style mirror...")
    mirror = style_mirror(store)
    (out / "style_mirror.json").write_text(json.dumps(mirror, indent=2))

    print(f"\n  All analysis artifacts saved to {out}/")
    return {
        "stats":       stats,
        "umap_2d":     umap_2d,
        "labels":      labels,
        "trajectory":  traj,
        "traits":      traits,
        "mirror":      mirror,
        "user_store":  user_store,
    }
