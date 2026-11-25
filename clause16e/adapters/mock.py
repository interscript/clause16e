"""Deterministic mock adapter: tests and pipeline smoke runs.

Produces synthetic text that cannot collide with any real corpus and
hash-derived scores, so the whole harness can be exercised end-to-end
with zero corpus exposure and zero ML dependencies.
"""

from __future__ import annotations

import hashlib

from .base import ModelAdapter, register_adapter

_SYNTH_WORDS = [
    "zqv", "mpl", "xkw", "jvr", "qhz", "vbn", "fkd", "wts", "zjx", "qbm",
    "ngd", "phz", "cvr", "xjm", "kwt", "zfl", "qvd", "bhn", "mxk", "wzq",
]


def _h(text: str, salt: str = "") -> int:
    return int.from_bytes(
        hashlib.blake2b((salt + text).encode("utf-8"), digest_size=8).digest(), "big"
    )


class MockAdapter(ModelAdapter):
    name = "mock"

    def __init__(self, params: dict):
        self.params = params

    def generate(self, prompt, max_new_tokens, temperature, seed):
        h = _h(prompt, f"gen{seed}")
        n = max(4, max_new_tokens // 2)
        words = []
        for i in range(n):
            words.append(_SYNTH_WORDS[(h >> (i % 7 * 3)) % len(_SYNTH_WORDS)])
        return " ".join(words)

    def sequence_logprob(self, text):
        words = text.split()
        n = max(1, len(words))
        total = -(_h(text, "lp") % 10000) / 100.0 * n / 10.0
        return (total, n)

    def vocab(self):
        return list(_SYNTH_WORDS) + ["the", "a", "of", "and"]

    def embed_texts(self, texts):
        out = []
        for t in texts:
            h = _h(t, "emb")
            dim = 16
            vec = [((h >> (i % 32)) & 0xFF) / 255.0 for i in range(dim)]
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out

    def describe(self):
        return {"adapter": self.name, "params": self.params}


register_adapter("mock", lambda params: MockAdapter(params))
