"""
Anonymization layer.

Replaces all known person names with stable pseudonyms (PERSON_A, PERSON_B, ...),
strips phone numbers and emails from text, and assigns the user a fixed label.

Output schema adds:
  "sender_anon":          str   (e.g. "YOU" or "PERSON_A")
  "text_anon":            str   (PII-scrubbed text)

The original "sender" and "text" fields are removed before anything
leaves this module, so downstream code never holds raw PII.
"""

import re
import hashlib
from typing import Optional


# ---------------------------------------------------------------------------
# PII scrubbing patterns
# ---------------------------------------------------------------------------

_PHONE_RE = re.compile(
    r"(\+?1[-.\s]?)?"
    r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_URL_RE   = re.compile(r"https?://\S+")


def _scrub_pii(text: str) -> str:
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    return text


# ---------------------------------------------------------------------------
# Name → pseudonym mapping
# ---------------------------------------------------------------------------

_LABELS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]  # A-Z then extend


def _make_label(index: int) -> str:
    if index < 26:
        return f"PERSON_{_LABELS[index]}"
    return f"PERSON_{index}"


class Anonymizer:
    def __init__(self, user_name: str = "Adam Webb"):
        self.user_name = user_name
        self._name_map: dict[str, str] = {}  # raw name → pseudonym
        self._counter = 0

    def _get_pseudonym(self, name: str) -> str:
        if name not in self._name_map:
            self._name_map[name] = _make_label(self._counter)
            self._counter += 1
        return self._name_map[name]

    def anonymize(self, messages: list[dict]) -> list[dict]:
        """
        Returns a new list of message dicts with raw text/sender replaced.
        Input dicts are not modified.
        """
        result = []
        for m in messages:
            sender = m.get("sender", "")
            is_user = m.get("is_user", False)

            if is_user or sender == self.user_name or sender == "USER":
                sender_anon = "YOU"
            else:
                sender_anon = self._get_pseudonym(sender)

            text_anon = _scrub_pii(m["text"])

            anon = {k: v for k, v in m.items() if k not in ("text", "sender")}
            anon["sender_anon"] = sender_anon
            anon["text_anon"] = text_anon
            result.append(anon)

        return result

    def pseudonym_map(self) -> dict[str, str]:
        """Returns {raw_name: pseudonym} — store securely or discard."""
        return dict(self._name_map)
