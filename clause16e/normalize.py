"""Text normalization for consecutive-word matching.

Matching is deliberately tolerant of surface variation (dropped
tashkeel/harakat, NFC vs NFD, case, punctuation) on BOTH sides, so a
reproduction cannot evade the probe by shedding diacritics or by
merging/splitting words around punctuation: punctuation is a word
BOUNDARY (a run of letters/digits is a word), never a deletion.
"""

from __future__ import annotations

import re
import unicodedata

# Maximal runs of word characters excluding underscore, unicode-aware.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _clean_text(text: str) -> str:
    """NFC + casefold, drop combining marks (Mn/Me) and format chars (Cf)."""
    text = unicodedata.normalize("NFC", text).casefold()
    return "".join(
        ch
        for ch in text
        if unicodedata.category(ch) not in ("Mn", "Me", "Cf")
    )


def tokenize(text: str) -> list[str]:
    """Canonical word stream: alphanumeric runs, casefolded, marks stripped.

    - "Hello, World!" -> ["hello", "world"]
    - "foo,bar" == "foo, bar" (punctuation is a boundary, not a deletion)
    - Arabic tashkeel (Mn marks) never creates or hides word boundaries.
    """
    return _WORD_RE.findall(_clean_text(text))


def normalize_word(raw: str) -> str:
    """Normalize a single token (kept for callers handling one word)."""
    words = tokenize(raw)
    return words[0] if len(words) == 1 else "".join(words)
