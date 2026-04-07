"""Pushbullet notification utility for Iterare.

Usage:
    from iterare.notify import notify
    notify("Exp done", "HotFlip CE=0.686")

Requires PUSHBULLET_API_KEY in .env. Silently no-ops if key is absent.
"""

import os
import json
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[2])) / ".env")
except ImportError:
    pass


def notify(title: str, body: str = "") -> bool:
    """Send a Pushbullet push notification.

    Returns True if sent successfully, False otherwise (never raises).
    """
    key = os.environ.get("PUSHBULLET_API_KEY", "")
    if not key:
        return False
    try:
        data = json.dumps({"type": "note", "title": title, "body": body}).encode()
        req = urllib.request.Request(
            "https://api.pushbullet.com/v2/pushes",
            data=data,
            headers={"Access-Token": key, "Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False
