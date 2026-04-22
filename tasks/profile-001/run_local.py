"""
profile-001 — local pipeline runner.

Privacy model:
  - Raw message text is parsed and immediately anonymized in memory.
  - Anonymized text is passed to the embedding model; text is never written to disk.
  - After embedding extraction, all downstream analysis operates on float vectors only.
  - The pseudonym map (real name → PERSON_X) is printed to stdout at the end
    so you can interpret results; it is NOT written to any file.

Usage:
    python3 tasks/profile-001/run_local.py

Outputs (all under tasks/profile-001/results/):
    embeddings.npy          float32 embeddings, shape (N, D)
    embedding_meta.json     metadata: ids, timestamps, sender_anons, platforms
    behavioral_stats.json   message counts, timing, text features
    temporal_trajectory.json  monthly centroid drift
    trait_scores.json        affect + Big Five proxy scores
    style_mirror.json        per-conversation similarity
    umap_clusters.png
    temporal_activity.png
    affect_trajectory.png
    big5_radar.png
    style_mirror.png
    timing.png
    synthesis.md
"""

import sys
import os

# Add iterare/code to path so tool modules resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code"))

from tools.message_profile.parsers import load_all
from tools.message_profile.anonymize import Anonymizer
from tools.message_profile.embed import extract_embeddings, EmbeddingStore
from tools.message_profile.analyze import run_analysis
from tools.message_profile.report import generate_all

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INSTAGRAM_MESSAGES_ROOT = os.path.expanduser(
    "~/Research/MyMessages/instagram/your_instagram_activity/messages"
)
DISCORD_MESSAGES_ROOT = os.path.expanduser(
    "~/Research/MyMessages/discord/Messages"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
EMBEDDINGS_DIR = os.path.join(RESULTS_DIR, "embeddings")

INSTAGRAM_USER_NAME = "Adam Webb"

# For local CPU runs, sample to keep embedding time under ~15 min.
# Set to None on TIDE to embed the full corpus.
LOCAL_MAX_MESSAGES = 15_000


def _stratified_sample(messages: list[dict], n: int) -> list[dict]:
    """
    Uniform temporal sample that preserves platform and sender ratios.
    Keeps at least every Kth message to cover the full date range.
    """
    if len(messages) <= n:
        return messages
    import random
    random.seed(4738)
    # Sort by time (already sorted), sample uniformly
    step = len(messages) / n
    sampled = [messages[int(i * step)] for i in range(n)]
    return sampled


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Message Profile Pipeline — Local Run")
    print("=" * 60)

    # ---- 1. Parse ----
    print("\n[1/4] Parsing messages...")
    messages = load_all(
        INSTAGRAM_MESSAGES_ROOT,
        DISCORD_MESSAGES_ROOT,
        instagram_user_name=INSTAGRAM_USER_NAME,
    )
    print(f"  Total messages loaded: {len(messages):,}")
    print(f"  User messages: {sum(1 for m in messages if m['is_user']):,}")
    print(f"  Date range: {messages[0]['timestamp'].date()} → {messages[-1]['timestamp'].date()}")

    # ---- 1b. Sample for local runs ----
    if LOCAL_MAX_MESSAGES and len(messages) > LOCAL_MAX_MESSAGES:
        messages = _stratified_sample(messages, LOCAL_MAX_MESSAGES)
        print(f"  Sampled to {len(messages):,} messages for local CPU run "
              f"(set LOCAL_MAX_MESSAGES=None for full corpus on TIDE)")

    # ---- 2. Anonymize ----
    print("\n[2/4] Anonymizing...")
    anon = Anonymizer(user_name=INSTAGRAM_USER_NAME)
    anon_messages = anon.anonymize(messages)
    del messages  # raw text no longer needed

    print(f"  {len(anon.pseudonym_map())} unique names pseudonymized")

    # ---- 3. Embed ----
    print("\n[3/4] Extracting embeddings...")

    # Check if embeddings already exist (skip re-embedding on re-runs)
    emb_path = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
    if os.path.exists(emb_path):
        print(f"  Found existing embeddings at {EMBEDDINGS_DIR}, loading...")
        store = EmbeddingStore.load(EMBEDDINGS_DIR)
    else:
        store = extract_embeddings(anon_messages, batch_size=128)
        store.save(EMBEDDINGS_DIR)

    print(f"  Embedding shape: {store.embeddings.shape}")

    # ---- 4. Analyze + Report ----
    print("\n[4/4] Running analysis...")
    results = run_analysis(store, anon_messages, RESULTS_DIR)

    print("\n[Report] Generating plots and synthesis...")
    generate_all(results, results["stats"], RESULTS_DIR)

    # ---- Done ----
    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"Results: {RESULTS_DIR}/")
    print("=" * 60)

    # Print pseudonym map so you can interpret per-conversation results.
    # NOT written to disk.
    pmap = anon.pseudonym_map()
    if pmap:
        print("\nPseudonym map (not saved to disk):")
        for real, pseudo in sorted(pmap.items(), key=lambda x: x[1]):
            print(f"  {pseudo:12s} → {real}")


if __name__ == "__main__":
    main()
