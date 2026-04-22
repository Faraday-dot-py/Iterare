"""
Embedding extraction.

Takes anonymized message dicts (text_anon field only — no raw text).
Produces an EmbeddingStore: numpy arrays + metadata, no message text retained.

Model: all-mpnet-base-v2 (768d, best quality in sentence-transformers family).
Falls back to all-MiniLM-L6-v2 (384d, faster) if mpnet unavailable.
"""

import numpy as np
from datetime import datetime
from typing import Optional
from pathlib import Path


MODEL_PREFERENCE = [
    "all-MiniLM-L6-v2",   # fast on CPU (~384d); swap to all-mpnet-base-v2 on TIDE
    "all-mpnet-base-v2",
]


def _load_model(model_name: Optional[str] = None):
    from sentence_transformers import SentenceTransformer
    names = [model_name] if model_name else MODEL_PREFERENCE
    for name in names:
        try:
            print(f"  Loading embedding model: {name}")
            m = SentenceTransformer(name)
            print(f"  Model loaded: {name}, dim={m.get_sentence_embedding_dimension()}")
            return m, name
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
    raise RuntimeError("No embedding model available")


class EmbeddingStore:
    """
    Holds embeddings + metadata. Text is never stored here.

    Attributes:
        embeddings:       np.ndarray, shape (N, D)
        ids:              list[str]
        timestamps:       list[datetime]
        sender_anons:     list[str]   ("YOU" or "PERSON_X")
        platforms:        list[str]   ("instagram" | "discord")
        conversation_ids: list[str]
        model_name:       str
        dim:              int
    """

    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.ids: list[str] = []
        self.timestamps: list[datetime] = []
        self.sender_anons: list[str] = []
        self.platforms: list[str] = []
        self.conversation_ids: list[str] = []
        self.model_name: str = ""
        self.dim: int = 0

    def save(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "embeddings.npy", self.embeddings)
        import json
        meta = {
            "ids": self.ids,
            "timestamps": [t.isoformat() for t in self.timestamps],
            "sender_anons": self.sender_anons,
            "platforms": self.platforms,
            "conversation_ids": self.conversation_ids,
            "model_name": self.model_name,
            "dim": self.dim,
        }
        (out / "embedding_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  Embeddings saved to {out}/")

    @classmethod
    def load(cls, out_dir: str) -> "EmbeddingStore":
        import json
        out = Path(out_dir)
        store = cls()
        store.embeddings = np.load(out / "embeddings.npy")
        meta = json.loads((out / "embedding_meta.json").read_text())
        store.ids = meta["ids"]
        store.timestamps = [datetime.fromisoformat(t) for t in meta["timestamps"]]
        store.sender_anons = meta["sender_anons"]
        store.platforms = meta["platforms"]
        store.conversation_ids = meta["conversation_ids"]
        store.model_name = meta["model_name"]
        store.dim = meta["dim"]
        return store

    def user_mask(self) -> np.ndarray:
        return np.array([s == "YOU" for s in self.sender_anons])

    def filter(self, mask: np.ndarray) -> "EmbeddingStore":
        s = EmbeddingStore()
        s.embeddings = self.embeddings[mask]
        s.ids = [v for v, m in zip(self.ids, mask) if m]
        s.timestamps = [v for v, m in zip(self.timestamps, mask) if m]
        s.sender_anons = [v for v, m in zip(self.sender_anons, mask) if m]
        s.platforms = [v for v, m in zip(self.platforms, mask) if m]
        s.conversation_ids = [v for v, m in zip(self.conversation_ids, mask) if m]
        s.model_name = self.model_name
        s.dim = self.dim
        return s


def extract_embeddings(
    messages: list[dict],
    model_name: Optional[str] = None,
    batch_size: int = 256,
) -> EmbeddingStore:
    """
    Embed all messages. Only text_anon is passed to the model.
    Returns EmbeddingStore with no message text.
    """
    model, resolved_name = _load_model(model_name)

    # sentence-transformers v5 runs URL/modality detection on every string;
    # bracketed content with colons (e.g. discord mentions, URLs with ports)
    # triggers a stdlib IPv6 parse error. Strip brackets before embedding.
    texts = [m["text_anon"].replace("[", "").replace("]", "") for m in messages]
    print(f"  Embedding {len(texts):,} messages in batches of {batch_size}...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    store = EmbeddingStore()
    store.embeddings = embeddings.astype(np.float32)
    store.ids = [m["id"] for m in messages]
    store.timestamps = [m["timestamp"] for m in messages]
    store.sender_anons = [m["sender_anon"] for m in messages]
    store.platforms = [m["platform"] for m in messages]
    store.conversation_ids = [m["conversation_id"] for m in messages]
    store.model_name = resolved_name
    store.dim = embeddings.shape[1]

    # texts list is discarded here — not stored in store
    return store
