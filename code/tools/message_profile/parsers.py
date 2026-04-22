"""
Parsers for Instagram and Discord message exports.
Output: list of normalized message dicts — raw text stays in memory only
during this stage and is passed directly to the anonymizer.

Normalized schema:
  {
    "id":              str,          # unique message ID
    "timestamp":       datetime,
    "text":            str,          # raw content
    "sender":          str,          # original name (will be replaced by anonymizer)
    "platform":        "instagram" | "discord",
    "conversation_id": str,
    "is_user":         bool,         # True = the account owner's message
  }

Instagram notes:
  - Files are message_1.json, message_2.json, ... per conversation
  - Instagram exports emoji as mojibake (latin-1 bytes decoded as utf-8).
    Fix: encode each string as latin-1, decode as utf-8.
  - The account owner's name is resolved from personal_information.json if present,
    otherwise inferred as the participant who is NOT consistent across conversations.

Discord notes:
  - channel.json: {"id", "type", "recipients": [uid, uid]}
  - messages.json: [{"ID", "Timestamp", "Contents", "Attachments"}]
  - Discord exports only YOUR messages — no sender field means all = user.
"""

import json
import glob
import os
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fix_instagram_encoding(text: str) -> str:
    """Fix Instagram's mojibake emoji encoding."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def _instagram_ts(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def _discord_ts(ts_str: str) -> datetime:
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Instagram
# ---------------------------------------------------------------------------

def parse_instagram(messages_root: str, user_name: str = "Adam Webb") -> list[dict]:
    """
    Parse all Instagram DM conversations under messages_root/inbox/.
    Returns normalized message list. No text is printed or logged.
    """
    inbox = Path(messages_root) / "inbox"
    if not inbox.exists():
        raise FileNotFoundError(f"Instagram inbox not found: {inbox}")

    messages = []
    for convo_dir in sorted(inbox.iterdir()):
        if not convo_dir.is_dir():
            continue
        convo_id = convo_dir.name

        # Collect all message_N.json shards
        shards = sorted(
            convo_dir.glob("message_*.json"),
            key=lambda p: int(p.stem.split("_")[1])
        )
        if not shards:
            continue

        for shard in shards:
            try:
                data = json.loads(shard.read_bytes())
            except Exception:
                continue

            raw_msgs = data.get("messages", [])
            for m in raw_msgs:
                content = m.get("content", "")
                if not content or not isinstance(content, str):
                    continue
                content = _fix_instagram_encoding(content)
                sender = _fix_instagram_encoding(m.get("sender_name", ""))
                ts_ms = m.get("timestamp_ms", 0)
                messages.append({
                    "id": f"ig_{convo_id}_{ts_ms}",
                    "timestamp": _instagram_ts(ts_ms),
                    "text": content,
                    "sender": sender,
                    "platform": "instagram",
                    "conversation_id": f"ig_{convo_id}",
                    "is_user": (sender == _fix_instagram_encoding(user_name)),
                })

    return messages


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

def parse_discord(messages_root: str) -> list[dict]:
    """
    Parse all Discord DM/group channels under messages_root.
    All messages in a Discord export are the user's own — no sender field.
    """
    root = Path(messages_root)
    messages = []

    for channel_dir in sorted(root.iterdir()):
        if not channel_dir.is_dir():
            continue

        msg_file = channel_dir / "messages.json"
        chan_file = channel_dir / "channel.json"
        if not msg_file.exists():
            continue

        convo_id = channel_dir.name

        # Channel type (DM vs GROUP_DM) — informational only
        chan_type = "DM"
        if chan_file.exists():
            try:
                chan_data = json.loads(chan_file.read_bytes())
                chan_type = chan_data.get("type", "DM")
            except Exception:
                pass

        try:
            raw_msgs = json.loads(msg_file.read_bytes())
        except Exception:
            continue

        for m in raw_msgs:
            content = m.get("Contents", "")
            if not content or not isinstance(content, str):
                continue
            ts_str = m.get("Timestamp", "")
            try:
                ts = _discord_ts(ts_str)
            except Exception:
                continue
            messages.append({
                "id": f"dc_{m.get('ID', '')}",
                "timestamp": ts,
                "text": content,
                "sender": "USER",
                "platform": "discord",
                "conversation_id": f"dc_{convo_id}",
                "is_user": True,
                "channel_type": chan_type,
            })

    return messages


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------

def load_all(instagram_messages_root: str, discord_messages_root: str,
             instagram_user_name: str = "Adam Webb") -> list[dict]:
    ig = parse_instagram(instagram_messages_root, user_name=instagram_user_name)
    dc = parse_discord(discord_messages_root)
    combined = ig + dc
    combined.sort(key=lambda m: m["timestamp"])
    return combined
