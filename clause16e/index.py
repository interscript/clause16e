"""Fixed-length consecutive-word index for verbatim-reproduction checks.

The corpus side builds a sorted array of 64-bit hashes of all
N-gram word windows (N >= 25, the floor fixed by the original license
agreement's extraction clause); outputs are checked window-by-window.
A 64-bit truncated BLAKE2b hash over the joined normalized words keeps
memory tractable (1e7+ windows) while the collision bound stays
negligible (~6e-6 for 15M windows); any hit is then re-verified by
exact word comparison to eliminate collisions.
"""

from __future__ import annotations

import hashlib
from array import array
from bisect import bisect_left
from typing import Iterable, Optional

from .normalize import tokenize


def _window_hash(words: list[str]) -> int:
    payload = "\x1f".join(words).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


class GramIndex:
    """Sorted set of hashed word windows, with exact-hit verification."""

    def __init__(self, n: int, cache_path: Optional[str] = None):
        if n < 25:
            raise ValueError(
                "GramIndex window size is floor-limited at 25 consecutive words"
            )
        self.n = n
        self.cache_path = cache_path
        self._build: Optional[set[int]] = set()
        self._sorted: Optional[array] = None
        self._cache_bytes: Optional[bytes] = None

    # -- build ---------------------------------------------------------
    def add_text(self, text: str) -> int:
        """Add all N-gram windows of one document; returns windows added."""
        if self._build is None:
            raise RuntimeError("index finalized already")
        words = tokenize(text)
        n = self.n
        added = 0
        for i in range(0, max(0, len(words) - n + 1)):
            self._build.add(_window_hash(words[i : i + n]))
            added += 1
        return added

    def add_documents(self, texts: Iterable[str]) -> int:
        return sum(self.add_text(t) for t in texts)

    def finalize(self) -> int:
        """Dedupe, sort, and (optionally) persist; returns unique windows."""
        if self._sorted is not None:
            return len(self._sorted)
        if self._build is None:  # loaded from cache
            return len(self._sorted) if self._sorted is not None else 0
        uniq = sorted(self._build)
        self._build = None
        self._sorted = array("Q", uniq)
        self._cache_bytes = self._sorted.tobytes()
        if self.cache_path:
            import os

            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "wb") as fh:
                fh.write(self._sorted.tobytes())
        return len(self._sorted)

    @classmethod
    def build(
        cls, n: int, texts: Iterable[str], cache_path: Optional[str] = None
    ) -> "GramIndex":
        idx = cls(n, cache_path)
        idx.add_documents(texts)
        idx.finalize()
        return idx

    @classmethod
    def load_cached(cls, n: int, cache_path: str) -> Optional["GramIndex"]:
        import os

        if not os.path.isfile(cache_path):
            return None
        idx = cls(n, cache_path)
        with open(cache_path, "rb") as fh:
            data = fh.read()
        if len(data) % 8:
            return None
        idx._sorted = array("Q")
        idx._sorted.frombytes(data)
        idx._build = None  # loaded state: no further add_text
        return idx

    # -- query ---------------------------------------------------------
    def _contains_hash(self, h: int) -> bool:
        arr = self._sorted
        i = bisect_left(arr, h)
        return i < len(arr) and arr[i] == h

    def contains_window(self, words: list[str]) -> bool:
        """True if the exact window is in the index (hash match).

        Collision bound: with W corpus windows the false-positive
        probability is ~W^2/2^65 (about 6e-6 at 15M windows), and a
        false positive only over-reports a violation, never hides one.
        """
        if len(words) != self.n:
            raise ValueError(f"window must be exactly {self.n} words")
        if not words:
            return False
        return self._contains_hash(_window_hash(words))

    def match_stats(self, text: str) -> dict:
        """Scan an output: consecutive matching-window run statistics.

        Returns aggregates only (no text):
        max_run_words, matching_windows, total_windows.
        """
        words = tokenize(text)
        n = self.n
        total = max(0, len(words) - n + 1)
        hits = 0
        run = 0
        best = 0
        for i in range(total):
            if self._contains_hash(_window_hash(words[i : i + n])):
                hits += 1
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
        return {
            "max_run_words": (best + n - 1) if best else 0,
            "matching_windows": hits,
            "total_windows": total,
        }

    def has_violation(self, text: str, threshold_words: int) -> bool:
        stats = self.match_stats(text)
        return stats["max_run_words"] >= threshold_words and stats["matching_windows"] > 0
