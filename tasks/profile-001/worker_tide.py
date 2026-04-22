"""
profile-001 TIDE worker.

Env vars injected via env_inject (never on disk):
  FERNET_KEY          Fernet key for decrypting messages.enc
  PUSHBULLET_API_KEY  (optional) push notifications

Remote layout:
  ~/profile_tide/messages.enc      encrypted anonymized messages JSON
  ~/profile_tide/code/             uploaded tool modules
  ~/profile_tide/results/          written by this script
  ~/profile_tide/results.tar.gz    tarred results, downloaded by submit script
"""

import os as _os, json as _json, urllib.request as _urlreq

def notify(title, body=""):
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


import sys, os, gzip, json, subprocess, shutil, zipfile, tempfile
from pathlib import Path
from datetime import datetime

HOME       = Path.home()
REMOTE_BASE = HOME / "profile_tide"
DATA_ENC   = REMOTE_BASE / "messages.enc"
CODE_DIR   = REMOTE_BASE / "code"
RESULTS_DIR = REMOTE_BASE / "results"

sys.path.insert(0, str(CODE_DIR))

try:
    notify("profile-001 TIDE", "Worker starting")

    # -------------------------------------------------------------------------
    # 1. Fix conda environment
    #
    # Problems:
    #   a. pip-installed torch in ~/.local shadows conda torch (wrong API version)
    #   b. conda transformers dep_check: tokenizers>=0.20,<0.21 but has 0.22.2
    #   c. conda transformers modeling_utils: LOSS_MAPPING transitively imports
    #      torchvision which has a circular import bug
    #
    # Fix: manually remove ~/.local/torch, then patch two conda transformers files.
    # We do NOT use sentence_transformers (avoids entire image-processing import chain).
    # -------------------------------------------------------------------------

    # Remove any pip-installed torch from ~/.local (do not use pip uninstall)
    _local_sp = Path.home() / ".local/lib/python3.11/site-packages"
    for _pfx in ["torch", "sentence_transformers"]:
        _d = _local_sp / _pfx
        if _d.exists():
            shutil.rmtree(_d)
            print(f"  Removed ~/.local/{_pfx}/")
        for _di in _local_sp.glob(f"{_pfx}-*.dist-info"):
            shutil.rmtree(_di)
        for _di in _local_sp.glob(f"{_pfx.replace('_','-')}-*.dist-info"):
            shutil.rmtree(_di)

    import importlib as _il
    _il.invalidate_caches()
    for _m in list(sys.modules):
        if any(x in _m for x in ("torch", "transformers", "sentence_transformers")):
            del sys.modules[_m]

    # Patch conda transformers files from the PyPI wheel
    CONDA_TF = Path("/opt/conda/lib/python3.11/site-packages/transformers")

    def _pyc_clean(p):
        for f in (p.parent / "__pycache__").glob(f"{p.stem}*.pyc"):
            f.unlink(missing_ok=True)

    def _patch_line(path, fragment, new_block):
        if not path.exists():
            print(f"  SKIP patch: {path.name} not found")
            return
        lines = path.read_text().split("\n")
        for i, ln in enumerate(lines):
            if fragment in ln and "#" not in ln.split(fragment)[0]:
                ind = " " * (len(ln) - len(ln.lstrip()))
                lines[i] = "\n".join(ind + l for l in new_block.split("\n"))
                path.write_text("\n".join(lines))
                _pyc_clean(path)
                print(f"  Patched {path.name}:{i+1}")
                return
        print(f"  {path.name}: '{fragment[:40]}' not found (already patched?)")

    def _restore_from_whl(whl_zip, filename):
        path = CONDA_TF / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with whl_zip.open(f"transformers/{filename}") as src:
                path.write_text(src.read().decode())
            _pyc_clean(path)
            print(f"  Restored {filename}")
        except KeyError:
            print(f"  {filename} not in wheel (skipping)")
        return path

    # Restore the FULL transformers package from the wheel (conda transformers may be
    # partially removed by prior pip uninstall operations).
    print("  Downloading transformers==4.46.3 wheel and restoring all files ...")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "pip", "download",
            "transformers==4.46.3", "--no-deps", "-d", tmp, "-q"], check=True)
        wheels = list(Path(tmp).glob("transformers*.whl"))
        if not wheels:
            raise RuntimeError("transformers wheel download failed")
        with zipfile.ZipFile(wheels[0]) as whl:
            # Restore all Python files in the transformers package
            n_restored = 0
            for name in whl.namelist():
                if name.startswith("transformers/") and name.endswith(".py"):
                    dest = Path("/opt/conda/lib/python3.11/site-packages") / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with whl.open(name) as src:
                        dest.write_bytes(src.read())
                    n_restored += 1
            print(f"  Restored {n_restored} transformers .py files from wheel")
            # Get handles for the specific files we'll patch
            mod_utils = CONDA_TF / "modeling_utils.py"
            dep_check = CONDA_TF / "dependency_versions_check.py"
    # Clear all .pyc caches in transformers
    for pyc in (CONDA_TF / "__pycache__").glob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for subdir in CONDA_TF.rglob("__pycache__"):
        for pyc in subdir.glob("*.pyc"):
            pyc.unlink(missing_ok=True)

    # Patch 1: skip tokenizers version check
    _patch_line(dep_check, "require_version_core(deps[pkg])",
                "pass  # tokenizers version check patched out")
    # Patch 2: guard LOSS_MAPPING (avoids transitive torchvision import)
    _patch_line(mod_utils, "from .loss.loss_utils import LOSS_MAPPING",
                "try:\n    from .loss.loss_utils import LOSS_MAPPING\n"
                "except Exception:\n    LOSS_MAPPING = {}")

    _il.invalidate_caches()
    for _m in list(sys.modules):
        if any(x in _m for x in ("transformers", "tokenizers")):
            del sys.modules[_m]

    # -------------------------------------------------------------------------
    # 2. Cryptography
    # -------------------------------------------------------------------------
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "cryptography", "-q"],
                       check=True)
        from cryptography.fernet import Fernet

    # -------------------------------------------------------------------------
    # 3. Decrypt messages
    # -------------------------------------------------------------------------
    fernet_key = os.environ.get("FERNET_KEY", "")
    if not fernet_key:
        raise RuntimeError("FERNET_KEY not set")

    print(f"\n[1/5] Decrypting {DATA_ENC} ...")
    raw = DATA_ENC.read_bytes()
    decrypted = Fernet(fernet_key.encode()).decrypt(raw)
    messages_raw = json.loads(gzip.decompress(decrypted))
    for m in messages_raw:
        m["timestamp"] = datetime.fromisoformat(m["timestamp"])
    print(f"  {len(messages_raw):,} messages")
    del raw, decrypted

    notify("profile-001 TIDE", f"Decrypted {len(messages_raw):,} messages — starting embed")

    # -------------------------------------------------------------------------
    # 4. Embed with AutoTokenizer + AutoModel (NO sentence_transformers)
    #    This avoids the entire image-processing import chain.
    # -------------------------------------------------------------------------
    print("\n[2/5] Embedding with all-mpnet-base-v2 (AutoModel, GPU) ...")
    import torch
    import torch.nn.functional as F
    import numpy as np
    from transformers import AutoTokenizer, AutoModel

    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    BATCH_SIZE = 512

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print(f"  Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  Loading model ...")
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    texts = [m["text_anon"].replace("[", "").replace("]", "") for m in messages_raw]
    print(f"  Encoding {len(texts):,} messages in batches of {BATCH_SIZE} ...")

    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        # Mean pooling
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().float().numpy())
        if (i // BATCH_SIZE) % 50 == 0:
            print(f"  Batch {i // BATCH_SIZE + 1}/{(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE}")

    embeddings = np.concatenate(all_embs, axis=0)
    del model, tokenizer, all_embs
    print(f"  Embeddings: {embeddings.shape}")

    # Build EmbeddingStore (using uploaded embed.py for the dataclass only)
    from tools.message_profile.embed import EmbeddingStore
    store = EmbeddingStore()
    store.embeddings = embeddings
    store.ids = [m["id"] for m in messages_raw]
    store.timestamps = [m["timestamp"] for m in messages_raw]
    store.sender_anons = [m["sender_anon"] for m in messages_raw]
    store.platforms = [m["platform"] for m in messages_raw]
    store.conversation_ids = [m["conversation_id"] for m in messages_raw]
    store.model_name = MODEL_NAME
    store.dim = embeddings.shape[1]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    store.save(str(RESULTS_DIR / "embeddings"))
    print(f"  Saved to {RESULTS_DIR}/embeddings/")
    notify("profile-001 TIDE", f"Embedded {embeddings.shape[0]:,}×{embeddings.shape[1]}d")

    # -------------------------------------------------------------------------
    # 5. Analyze
    # -------------------------------------------------------------------------
    print("\n[3/5] Behavioral stats ...")
    from tools.message_profile.analyze import (
        behavioral_stats, cluster_embeddings, temporal_trajectory,
        probe_traits, style_mirror,
    )

    stats = behavioral_stats(messages_raw)
    (RESULTS_DIR / "behavioral_stats.json").write_text(json.dumps(stats, indent=2, default=str))

    print("\n[4/5] Clustering (t-SNE subsample 20k) ...")
    user_store, tsne_2d, labels, _ = cluster_embeddings(store, n_clusters=12, tsne_max_samples=20_000)
    np.save(RESULTS_DIR / "umap_2d.npy", tsne_2d)
    np.save(RESULTS_DIR / "cluster_labels.npy", labels)
    notify("profile-001 TIDE", f"Clustering done — {len(set(labels))} clusters")

    print("\n  Temporal trajectory ...")
    traj = temporal_trajectory(store)
    traj_save = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in traj.items()}
    (RESULTS_DIR / "temporal_trajectory.json").write_text(json.dumps(traj_save, indent=2))

    # Trait probing: use the same model approach (load fresh)
    print("\n  Trait probing ...")
    traits = probe_traits(store, MODEL_NAME)
    (RESULTS_DIR / "trait_scores.json").write_text(json.dumps(traits, indent=2))
    notify("profile-001 TIDE",
           f"Traits done — top: {max(traits['affect_mean'], key=traits['affect_mean'].get)}")

    print("\n  Style mirror ...")
    mirror = style_mirror(store)
    (RESULTS_DIR / "style_mirror.json").write_text(json.dumps(mirror, indent=2))

    # -------------------------------------------------------------------------
    # 6. Report
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating plots and synthesis ...")
    from tools.message_profile.report import generate_all
    results = {
        "stats": stats, "umap_2d": tsne_2d, "labels": labels,
        "trajectory": traj, "traits": traits, "mirror": mirror,
        "user_store": user_store,
    }
    generate_all(results, stats, str(RESULTS_DIR))

    # -------------------------------------------------------------------------
    # 7. Tar results
    # -------------------------------------------------------------------------
    import tarfile
    tar_path = REMOTE_BASE / "results.tar.gz"
    print(f"\nTarring results → {tar_path} ...")
    with tarfile.open(str(tar_path), "w:gz") as tar:
        tar.add(str(RESULTS_DIR), arcname="results_tide")
    print(f"  Archive: {tar_path.stat().st_size / 1e6:.1f} MB")

    notify("profile-001 TIDE complete",
           f"agreeableness={traits['big5_mean']['agreeableness']:.4f} "
           f"mirror={sum(v['similarity'] for v in mirror.values())/max(1,len(mirror)):.3f}")
    print("\nDone.")

except Exception as e:
    import traceback
    notify("profile-001 TIDE FAILED", str(e))
    traceback.print_exc()
    raise
