"""Corpus access: document streaming and provenance digest.

The protected corpus lives OUTSIDE this repo and is referenced only via
environment-provided paths (data-access contract). Nothing here ever
copies corpus text into the repo; everything read is used in-process or
written to the private records dir.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Iterator

if TYPE_CHECKING:  # pragma: no cover
    import random


@dataclass
class CorpusDocument:
    text: str
    source: str  # corpus-relative file path (a path, not corpus content)
    start_line: int
    end_line: int


def resolve_dir(spec: str) -> str:
    path = os.path.realpath(os.path.expanduser(spec))
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"corpus dir not found: {spec} (resolved: {path}); "
            "set the environment path per the data-access contract"
        )
    return path


def _glob_files(root: str, patterns: Iterable[str]) -> list[str]:
    import glob

    out: list[str] = []
    for pat in patterns:
        out.extend(sorted(glob.glob(os.path.join(root, pat), recursive=True)))
    seen: set[str] = set()
    uniq: list[str] = []
    for f in out:
        rf = os.path.realpath(f)
        if os.path.isfile(rf) and rf not in seen:
            seen.add(rf)
            uniq.append(rf)
    return uniq


def iter_documents(
    root: str, patterns: Iterable[str], group_lines: int = 0
) -> Iterator[CorpusDocument]:
    """Stream documents from corpus shards.

    Shards are sentence-per-line text files. ``group_lines`` groups that
    many consecutive lines into one pseudo-document (0 = one document
    per file).
    """
    root = resolve_dir(root)
    for path in _glob_files(root, patterns):
        rel = os.path.relpath(path, root)
        with open(path, "r", encoding="utf-8", errors="strict") as fh:
            lines = fh.read().splitlines()
        if group_lines <= 0:
            text = "\n".join(lines)
            if text.strip():
                yield CorpusDocument(text, rel, 1, len(lines))
        else:
            for i in range(0, len(lines), group_lines):
                chunk = lines[i : i + group_lines]
                text = "\n".join(chunk)
                if text.strip():
                    yield CorpusDocument(text, rel, i + 1, i + len(chunk))


def list_corpus_files(root: str, patterns: Iterable[str]) -> list[str]:
    root = resolve_dir(root)
    return _glob_files(root, patterns)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_digest(root: str, patterns: Iterable[str]) -> str:
    """Content-addressed digest of every corpus file the run touched.

    Provenance for the audit record: identifies exactly which corpus
    files were used without recording any corpus content.
    """
    h = hashlib.sha256()
    root = resolve_dir(root)
    for path in _glob_files(root, patterns):
        rel = os.path.relpath(path, root)
        h.update(f"{rel}\0{os.path.getsize(path)}\0{file_sha256(path)}\n".encode())
    return h.hexdigest()


def partition_holdout(
    docs: list[CorpusDocument], holdout_files: set[str]
) -> tuple[list[CorpusDocument], list[CorpusDocument]]:
    """Partition chunks into (held_in, held_out) by source file.

    Rationale (the split-construction audit; full record in
    docs/case-study.md): a membership test compares chunks of the
    training shards against chunks of a DIFFERENT collection only at
    its peril — the two collections can carry opposite length/
    truncation profiles, in which case BOTH the candidate and the
    reference rank the sets apart while the candidate has never seen
    either collection. The probe then measures collection difference,
    not membership. Same-population mode draws BOTH sets from the SAME
    train-shard chunk population, disjoint by file, so held-out is
    drawn from the same population as held-in per standard LiRA
    methodology.

    Path-shape note: iter_documents sources are corpus-RELATIVE paths,
    but list_corpus_files returns ABSOLUTE paths (realpath of glob
    results) — comparing the two shapes directly matches nothing and
    empties the held-out pool (a real gate failure during the audit).
    Matching is therefore rel-path OR basename (basenames are unique
    within the train-shard population).
    """
    rel_or_base = {f: os.path.basename(f) for f in holdout_files}
    base_keys = set(rel_or_base.values())
    held_in: list[CorpusDocument] = []
    held_out: list[CorpusDocument] = []
    for d in docs:
        hit = (d.source in holdout_files
               or d.source in rel_or_base
               or os.path.basename(d.source) in base_keys)
        (held_out if hit else held_in).append(d)
    return held_in, held_out


def membership_samples(cfg, held_in, held_out, rng: "random.Random"):
    """Draw membership-test samples; match the length DISTRIBUTION.

    Same pool filter as the legacy sampler (>= mia_min_words words),
    capped at mia_n_eval. In same_population mode with length_parity,
    the held-out sample mirrors the held-in length DISTRIBUTION via
    decile quantile bands (standard LiRA practice: match the
    distribution, not individual chunks). NEVER drops to empty: if the
    band-selected yield is below the floor, the full held-out shard
    population is sampled instead (it is the same population by
    construction — same shard format, disjoint files). The achieved
    medians and the fallback flag are recorded either way; nothing
    hard-fails on the distribution.
    In legacy globs mode the medians are recorded but never used
    (backward compatible).

    Returns (in_sample, out_sample, parity_record).
    """
    import statistics

    def usable(docs):
        return [d for d in docs if len(d.text.split()) >= cfg.mia_min_words]

    in_pool = usable(held_in)
    out_pool = usable(held_out)
    in_sample = (in_pool if len(in_pool) <= cfg.mia_n_eval
                 else rng.sample(in_pool, cfg.mia_n_eval))
    out_sample = (out_pool if len(out_pool) <= cfg.mia_n_eval
                  else rng.sample(out_pool, cfg.mia_n_eval))

    mode = getattr(cfg, "mia_split_mode", "globs")
    enforce = mode == "same_population" and getattr(
        cfg, "mia_length_parity", True)
    fallback = False
    if enforce and in_sample and out_pool:
        in_counts = sorted(len(d.text.split()) for d in in_sample)
        bands = 10
        bounds = [in_counts[min(int(q * len(in_counts)), len(in_counts) - 1)]
                  for q in [i / bands for i in range(bands)]]
        bounds.append(in_counts[-1] + 1)  # inclusive top band
        per_band = max(1, cfg.mia_n_eval // bands)
        floor = max(min(cfg.mia_n_eval, max(50, len(out_pool) // 2)), 1)
        by_count = sorted(((len(d.text.split()), d) for d in out_pool),
                          key=lambda t: t[0])
        taken: set[int] = set()
        picked: list[CorpusDocument] = []
        for b in range(bands):
            lo, hi = bounds[b], bounds[b + 1]
            got = 0
            for k, (c, d) in enumerate(by_count):
                if k in taken:
                    continue
                if lo <= c < hi or (b == bands - 1 and c >= lo):
                    picked.append(d)
                    taken.add(k)
                    got += 1
                    if got >= per_band:
                        break
        if len(picked) < floor:
            # NEVER drop to empty: the full held-out shard population is
            # the same population by construction (same shard format,
            # disjoint files) — sample it instead of the band selection.
            m = min(cfg.mia_n_eval, len(out_pool))
            idxs = rng.sample(range(len(by_count)), m)
            picked = [by_count[i][1] for i in idxs]
            fallback = True
        out_sample = picked
    in_med = (round(statistics.median(len(d.text.split()) for d in in_sample))
              if in_sample else 0)
    out_med = (round(statistics.median(len(d.text.split()) for d in out_sample))
               if out_sample else 0)
    in_band = bool(in_med and out_med and 0.8 <= out_med / in_med <= 1.25)
    parity = {"mode": mode, "enforced": enforce,
              "in_median_words": in_med, "out_median_words": out_med,
              "fallback": fallback,
              "pass": (in_band or not enforce)}
    return in_sample, out_sample, parity
