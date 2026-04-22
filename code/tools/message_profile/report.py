"""
Report generation.
Produces:
  - umap_clusters.png       2D embedding space colored by cluster
  - temporal_activity.png   messages per month + drift
  - affect_trajectory.png   affect dimensions over time
  - big5_radar.png          Big Five proxy scores
  - style_mirror.png        per-conversation similarity heatmap
  - synthesis.md            written narrative from statistics
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 1. UMAP cluster plot
# ---------------------------------------------------------------------------

def plot_umap(umap_2d: np.ndarray, labels: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    unique = sorted(set(labels))
    cmap = cm.get_cmap("tab20", max(len(unique), 1))

    for i, label in enumerate(unique):
        mask = labels == label
        color = "#cccccc" if label == -1 else cmap(i)
        name  = "noise" if label == -1 else f"Cluster {label}"
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                   c=[color], s=4, alpha=0.4, label=name, rasterized=True)

    n_clusters = sum(1 for l in unique if l != -1)
    ax.set_title(f"Message Embedding Space — {n_clusters} clusters", fontsize=14)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=3, fontsize=8, loc="upper right",
              ncol=2, framealpha=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Temporal activity + drift
# ---------------------------------------------------------------------------

def plot_temporal(traj: dict, out_path: str):
    months  = traj["months"]
    counts  = traj["counts"]
    drifts  = traj["drifts"]
    if not months:
        return

    x = range(len(months))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.bar(x, counts, color="#4c7ef3", alpha=0.8)
    ax1.set_ylabel("Messages / month")
    ax1.set_title("Messaging Activity Over Time")

    ax2.plot(x, drifts, color="#e05c2b", linewidth=1.5, marker="o", markersize=3)
    ax2.set_ylabel("Centroid drift\n(cosine distance)")
    ax2.set_xlabel("Month")

    step = max(1, len(months) // 18)
    ax2.set_xticks(list(x)[::step])
    ax2.set_xticklabels(months[::step], rotation=45, ha="right", fontsize=7)

    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Affect trajectory
# ---------------------------------------------------------------------------

def plot_affect(traits: dict, out_path: str):
    months = traits["months"]
    if not months:
        return

    x = range(len(months))
    colors = {
        "positive": "#2ecc71", "negative": "#e74c3c",
        "energetic": "#f39c12", "withdrawn": "#95a5a6",
        "playful": "#9b59b6", "neutral": "#bdc3c7",
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    for dim, vals in traits["affect_monthly"].items():
        ax.plot(x, vals, label=dim, color=colors.get(dim), linewidth=1.5, alpha=0.85)

    step = max(1, len(months) // 18)
    ax.set_xticks(list(x)[::step])
    ax.set_xticklabels(months[::step], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Cosine similarity to reference")
    ax.set_title("Affect Profile Over Time")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 4. Big Five radar
# ---------------------------------------------------------------------------

def plot_big5(traits: dict, out_path: str):
    dims   = list(traits["big5_mean"].keys())
    scores = [traits["big5_mean"][d] for d in dims]

    # Normalize scores to [0, 1] range for radar
    s_arr = np.array(scores)
    s_norm = (s_arr - s_arr.min()) / (s_arr.max() - s_arr.min() + 1e-9)

    N = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    vals = s_norm.tolist() + s_norm[:1].tolist()

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, vals, color="#4c7ef3", linewidth=2)
    ax.fill(angles, vals, color="#4c7ef3", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.capitalize() for d in dims], fontsize=11)
    ax.set_yticks([])
    ax.set_title("Big Five Proxy (embedding similarity)", pad=20, fontsize=13)

    # Annotate raw scores
    for angle, dim, raw in zip(angles[:-1], dims, scores):
        ax.annotate(f"{raw:.3f}", xy=(angle, s_norm[dims.index(dim)]),
                    fontsize=8, ha="center")

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 5. Style mirror heatmap
# ---------------------------------------------------------------------------

def plot_style_mirror(mirror: dict, out_path: str):
    if not mirror:
        return

    # Sort by similarity descending, take top 30
    items = sorted(mirror.items(), key=lambda x: -x[1]["similarity"])[:30]
    labels = [k[-12:] for k, _ in items]
    sims   = [v["similarity"] for _, v in items]

    fig, ax = plt.subplots(figsize=(8, max(4, len(items) * 0.3)))
    bars = ax.barh(range(len(items)), sims, color="#4c7ef3", alpha=0.8)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Cosine similarity (you ↔ friend)")
    ax.set_title("Style Mirror — per conversation", fontsize=13)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 6. Hour-of-day + day-of-week
# ---------------------------------------------------------------------------

def plot_timing(stats: dict, out_path: str):
    hour_dist = stats["user_timing"]["hour_distribution"]
    day_dist  = stats["user_timing"]["day_distribution"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(range(24), hour_dist, color="#4c7ef3", alpha=0.8)
    ax1.set_xlabel("Hour (UTC)"); ax1.set_ylabel("Messages")
    ax1.set_title("Messages by Hour of Day")
    ax1.set_xticks(range(0, 24, 3))

    ax2.bar(DAYS, day_dist, color="#e05c2b", alpha=0.8)
    ax2.set_xlabel("Day"); ax2.set_ylabel("Messages")
    ax2.set_title("Messages by Day of Week")

    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 7. Written synthesis
# ---------------------------------------------------------------------------

def _interpret_big5(scores: dict) -> str:
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top = ranked[0][0]
    bottom = ranked[-1][0]

    interpretations = {
        "openness":          "communicates with intellectual curiosity and comfort with abstraction",
        "conscientiousness": "tends toward directness and follow-through in communication",
        "extraversion":      "communicates with social energy and outward orientation",
        "agreeableness":     "language skews warm, supportive, and other-focused",
        "neuroticism":       "language occasionally reflects emotional intensity or stress",
    }
    low_interp = {
        "openness":          "communication style is grounded and concrete",
        "conscientiousness": "communication style is relaxed and flexible",
        "extraversion":      "communication is often brief and inward-facing",
        "agreeableness":     "communication style is direct and low on social cushioning",
        "neuroticism":       "emotional tone is generally stable",
    }

    lines = []
    lines.append(f"The strongest Big Five signal in the embedding space is **{top}** — "
                 f"this user {interpretations.get(top, top)}.")
    lines.append(f"The weakest signal is **{bottom}** — {low_interp.get(bottom, bottom)}.")
    return " ".join(lines)


def _interpret_affect(means: dict) -> str:
    ranked = sorted(means.items(), key=lambda x: -x[1])
    dominant = ranked[0][0]
    subdominant = ranked[1][0]
    return (f"The dominant affect signature is **{dominant}**, "
            f"with **{subdominant}** as a secondary tone. "
            f"Negative affect scores at {means['negative']:.3f} "
            f"vs positive at {means['positive']:.3f}.")


def _interpret_timing(stats: dict) -> str:
    hours = stats["user_timing"]["hour_distribution"]
    peak_hour = int(np.argmax(hours))
    days  = stats["user_timing"]["day_distribution"]
    peak_day = DAYS[int(np.argmax(days))]
    return (f"Peak messaging hour (UTC) is **{peak_hour}:00**, "
            f"peak day is **{peak_day}**. "
            f"Average message length: {stats['user_text']['mean_words']:.1f} words. "
            f"Question rate: {stats['user_text']['question_rate']:.1%}. "
            f"Exclamation rate: {stats['user_text']['exclaim_rate']:.1%}.")


def write_synthesis(stats: dict, traits: dict, mirror: dict,
                    n_clusters: int, out_path: str):
    date_range = stats.get("date_range", {})
    first = date_range.get("first", "unknown")[:10]
    last  = date_range.get("last",  "unknown")[:10]

    mirror_mean = np.mean([v["similarity"] for v in mirror.values()]) if mirror else 0

    lines = [
        "# Psychological Profile — Message Analysis",
        f"*Generated {datetime.now().strftime('%Y-%m-%d')} from {stats['user_messages']:,} user messages "
        f"across {stats['conversations']} conversations ({first} → {last})*",
        "",
        "---",
        "",
        "## Data summary",
        f"- **Platforms:** {', '.join(f'{k}: {v:,}' for k, v in stats['platform_distribution'].items())}",
        f"- **User messages:** {stats['user_messages']:,} "
        f"(friend messages: {stats['friend_messages']:,})",
        f"- **Mean message length:** {stats['user_text']['mean_words']:.1f} words "
        f"(p90: {stats['user_text']['p90_words']:.0f} words)",
        f"- **Embedding clusters found:** {n_clusters}",
        "",
        "## Communication style",
        _interpret_timing(stats),
        "",
        "## Affective signature",
        _interpret_affect(traits["affect_mean"]),
        "",
        "## Big Five proxy",
        _interpret_big5(traits["big5_mean"]),
        "",
        "## Style mirroring",
        (f"Mean cosine similarity between your embedding centroid and conversation "
         f"partners: **{mirror_mean:.3f}**. "
         + ("High similarity suggests adaptive mirroring; low suggests a consistent "
            "personal register regardless of partner." if mirror_mean > 0.5
            else "Relatively low mirroring — communication style is consistent "
                 "and independent of the conversation partner."))
        if mirror else "*Style mirror not available (Instagram data required).*",
        "",
        "## Raw scores",
        "### Affect",
        *[f"- {k}: {v:.4f}" for k, v in sorted(
            traits["affect_mean"].items(), key=lambda x: -x[1])],
        "",
        "### Big Five",
        *[f"- {k}: {v:.4f}" for k, v in sorted(
            traits["big5_mean"].items(), key=lambda x: -x[1])],
        "",
        "---",
        "*Note: Big Five and affect scores are cosine similarities to reference phrases "
        "in the same embedding space — they are proxy measures, not validated psychometric scales. "
        "Interpret relative rankings, not absolute values.*",
    ]

    Path(out_path).write_text("\n".join(lines))
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Master
# ---------------------------------------------------------------------------

def generate_all(results: dict, stats: dict, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_clusters = len(set(results["labels"]))

    print("\n[Plots] t-SNE cluster map...")
    plot_umap(results["umap_2d"], results["labels"],
              str(out / "umap_clusters.png"))

    print("[Plots] Temporal activity...")
    plot_temporal(results["trajectory"], str(out / "temporal_activity.png"))

    print("[Plots] Affect trajectory...")
    plot_affect(results["traits"], str(out / "affect_trajectory.png"))

    print("[Plots] Big Five radar...")
    plot_big5(results["traits"], str(out / "big5_radar.png"))

    print("[Plots] Style mirror...")
    plot_style_mirror(results["mirror"], str(out / "style_mirror.png"))

    print("[Plots] Timing...")
    plot_timing(stats, str(out / "timing.png"))

    print("[Report] Writing synthesis...")
    write_synthesis(stats, results["traits"], results["mirror"],
                    n_clusters, str(out / "synthesis.md"))
