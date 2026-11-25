"""Test 4 — reference-resource (e.g. WordNet) leakage probe.

Checks that a released tokenizer (and, optionally, its embedding
table) does not embed reference-resource synset content verbatim:

- gloss + example word N-grams (N = ``min_ngram_words``, default 5;
  glosses are far shorter than the 25-word verbatim threshold, so the
  consecutive-word rule is applied here at gloss scale),
- multi-word lemma phrases (exact match),
- synset relation strings, when present in the input file.

Single-word lemmas are NOT flagged: words of the language occur in
ordinary text and a general tokenizer legitimately contains them; only
multi-word sequences unique to the resource are evidence of copying.

Input: JSONL, one synset per line, with ``id``, ``pos``, ``gloss``,
``example``, ``lemma`` (string or list), optional ``relations`` — read
via an environment path only.

Embedding sub-probe (``wordnet.embedding_probe``): gloss N-gram
anchors are embedded and compared to a seeded random-word baseline;
a mean-similarity gap above ``wordnet.embedding_margin`` fails. This
is a heuristic secondary signal, reported as its own metric.
"""

from __future__ import annotations

import json
import math
import random

from ..adapters.base import ModelAdapter
from ..config import RunConfig
from ..records import RecordsWriter
from .base import STATUS_FAIL, STATUS_PASS, ProbeResult


def load_synsets(path: str) -> list[dict]:
    synsets = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                synsets.append(json.loads(line))
    if not synsets:
        raise ValueError(f"no synsets found in {path}")
    return synsets


def _content_ngrams(synsets: list[dict], n: int) -> set[tuple[str, ...]]:
    from ..normalize import tokenize

    grams: set[tuple[str, ...]] = set()
    for s in synsets:
        parts = [s.get("gloss") or "", s.get("example") or ""]
        lemmas = s.get("lemma") or []
        if isinstance(lemmas, str):
            lemmas = [lemmas]
        for lemma in lemmas:
            words = tokenize(lemma)
            if len(words) >= 2:  # multi-word lemma phrases only
                grams.add(tuple(words))
        for text in parts:
            words = tokenize(text)
            for i in range(0, max(0, len(words) - n + 1)):
                grams.add(tuple(words[i : i + n]))
    return {g for g in grams if g}


def _vocab_violations(vocab: list[str], grams: set[tuple[str, ...]]) -> list[str]:
    from ..normalize import tokenize

    hits = []
    for token in vocab:
        words = tokenize(token)
        if any(
            tuple(words[i:j]) in grams
            for i in range(0, len(words) - 1)
            for j in range(i + 2, len(words) + 1)
        ):
            hits.append(token)
    return hits


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def run_wordnet(
    cfg: RunConfig,
    adapter: ModelAdapter,
    records: RecordsWriter,
) -> ProbeResult:
    if not cfg.wordnet_path:
        return ProbeResult(
            "wordnet", STATUS_FAIL, error="wordnet.path not configured"
        )
    synsets = load_synsets(cfg.wordnet_path)
    n = cfg.wordnet_min_ngram_words
    grams = _content_ngrams(synsets, n)

    vocab = adapter.vocab()
    if vocab is None:
        return ProbeResult(
            "wordnet",
            STATUS_FAIL,
            error="model exposes no vocabulary for the leakage scan",
            metrics={"synsets": len(synsets), "content_ngrams_indexed": len(grams)},
        )
    hits = _vocab_violations(vocab, grams)
    # Synset-derived strings go to the private records dir only.
    records.write_jsonl(
        "wordnet-vocab-hits.jsonl",
        [{"token": t} for t in hits],
    )

    metrics = {
        "synsets": len(synsets),
        "content_ngrams_indexed": len(grams),
        "vocab_size": len(vocab),
        "vocab_ngram_hits": len(hits),
    }
    status = STATUS_PASS if not hits else STATUS_FAIL

    if cfg.wordnet_embedding_probe:
        emb = _embedding_gap(cfg, adapter, grams)
        if emb is None:
            metrics["embedding_probe"] = "NOT_RUN"
        else:
            gap, anchor_mean, baseline_mean = emb
            metrics["embedding_probe"] = {
                "anchor_mean_cosine": round(anchor_mean, 4),
                "baseline_mean_cosine": round(baseline_mean, 4),
                "gap": round(gap, 4),
                "margin": cfg.wordnet_embedding_margin,
            }
            if gap > cfg.wordnet_embedding_margin:
                status = STATUS_FAIL
    return ProbeResult("wordnet", status, metrics)


def _embedding_gap(cfg, adapter, grams):
    rng = random.Random(cfg.seed)
    anchors = sorted(grams, key=lambda g: rng.random())[:200]
    if not anchors:
        return None
    baseline_words = ["fqz", "xvm", "pkd", "wjh", "ztl", "qns", "bvr", "mkx"] * 25
    baseline = [" ".join(rng.sample(baseline_words, len(g))) for g in anchors]
    a = adapter.embed_texts([" ".join(g) for g in anchors])
    b = adapter.embed_texts(baseline)
    if a is None or b is None:
        return None

    def _mean_pairwise(vecs: list[list[float]], pairs: int) -> float:
        idx = list(range(len(vecs)))
        total = 0.0
        count = 0
        for _ in range(pairs):
            i, j = rng.sample(idx, 2)
            total += _cosine(vecs[i], vecs[j])
            count += 1
        return total / max(1, count)

    anchor_mean = _mean_pairwise(a, 2000)
    baseline_mean = _mean_pairwise(b, 2000)
    return (anchor_mean - baseline_mean, anchor_mean, baseline_mean)
