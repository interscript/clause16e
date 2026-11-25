"""Matched-split regression pins (the split-construction audit; see
docs/case-study.md for the full falsification record).

Pins for the corrected MIA split construction:
- same-population provenance (disjoint holdout shard subset),
- path-shape safety (list_corpus_files absolute vs chunk-relative sources
  must never empty the held-out pool — a real gate failure during the
  audit),
- quantile-band length matching mirrors the held-in distribution,
- NEVER drops to empty (full-population fallback below the floor),
- legacy globs mode is record-only and unchanged.
"""

import os
import random
import tempfile
from types import SimpleNamespace

from clause16e.corpus import (
    CorpusDocument,
    iter_documents,
    list_corpus_files,
    membership_samples,
    partition_holdout,
)


def doc(i: int, words: int, source: str) -> CorpusDocument:
    return CorpusDocument(" ".join(f"w{j}" for j in range(words)), source, 1, words)


def cfg(**over):
    base = dict(mia_n_eval=100, mia_min_words=5,
                mia_split_mode="same_population", mia_length_parity=True)
    base.update(over)
    return SimpleNamespace(**base)


def test_partition_holdout_rel_entries_disjoint():
    docs = ([doc(i, 50, "train/shard-06.txt") for i in range(10)]
            + [doc(i, 50, "train/shard-00.txt") for i in range(90)])
    holdout = {"train/shard-06.txt"}
    held_in, held_out = partition_holdout(docs, holdout)
    assert {d.source for d in held_out} == holdout
    assert not {d.source for d in held_in} & holdout
    assert len(held_in) == 90 and len(held_out) == 10


def test_partition_holdout_path_shape_regression():
    """THE PATH-SHAPE GATE ERROR: list_corpus_files returns ABSOLUTE
    paths while chunk sources are corpus-relative. The partition must
    never produce an empty held-out pool from the path-shape mismatch."""
    root = tempfile.mkdtemp()
    os.makedirs(f"{root}/train")
    for s in range(9):
        with open(f"{root}/train/shard-{s:02d}.txt", "w") as f:
            for i in range(64):
                f.write(f"shard {s} line {i} some corpus text here.\n")
    docs = list(iter_documents(root, ["train/shard-*.txt"], 32))
    holdout_abs = set(list_corpus_files(root, ["train/shard-06.txt",
                                               "train/shard-07.txt"]))
    assert all(os.path.isabs(f) for f in holdout_abs)  # the bug's precondition
    held_in, held_out = partition_holdout(docs, holdout_abs)
    assert len(held_out) == 4, "path-shape mismatch emptied the held-out pool"
    assert len(held_in) == len(docs) - len(held_out)
    assert {d.source for d in held_out} == {"train/shard-06.txt",
                                            "train/shard-07.txt"}


def test_same_population_quantile_band_matches_distribution():
    rng = random.Random(1)
    held_in = [doc(i, 300 + (i % 200), "train/shard-00.txt") for i in range(400)]
    held_out = [doc(i, 80 + (i % 500), "train/shard-06.txt") for i in range(600)]
    in_s, out_s, parity = membership_samples(cfg(), held_in, held_out, rng)
    assert len(in_s) == 100 and len(out_s) == 100
    assert parity["enforced"] is True
    assert all(len(d.text.split()) > 0 for d in out_s)
    assert parity["fallback"] is False


def test_band_yield_below_floor_falls_back_full_population():
    """A pool whose overlap with the held-in bands is thin must FALL BACK
    to the full held-out population — never drop to empty."""
    rng = random.Random(2)
    held_in = [doc(i, 400, "train/shard-00.txt") for i in range(400)]
    # narrow pool far from the held-in bands: band yield < floor
    held_out = [doc(i, 30, "train/shard-06.txt") for i in range(60)]
    in_s, out_s, parity = membership_samples(cfg(), held_in, held_out, rng)
    assert len(out_s) == 60, "fallback must sample the full held-out pool"
    assert parity["fallback"] is True
    assert parity["pass"] is False  # distribution parity honestly recorded


def test_globs_mode_records_but_never_enforces():
    rng = random.Random(5)
    held_in = [doc(i, 400, "a.txt") for i in range(400)]
    held_out = [doc(i, 40, "b.txt") for i in range(150)]
    in_s, out_s, parity = membership_samples(
        cfg(mia_split_mode="globs"), held_in, held_out, rng
    )
    assert parity["enforced"] is False and parity["pass"] is True
    assert parity["fallback"] is False


def test_provenance_pin_sources_match_population():
    rng = random.Random(6)
    held_in = [doc(i, 400, "train/shard-00.txt") for i in range(400)]
    held_out = [doc(i, 400, "train/shard-06.txt") for i in range(400)]
    in_s, out_s, parity = membership_samples(cfg(), held_in, held_out, rng)
    assert {d.source for d in in_s} == {"train/shard-00.txt"}
    assert {d.source for d in out_s} == {"train/shard-06.txt"}
    assert not {d.source for d in in_s} & {d.source for d in out_s}
