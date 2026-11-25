"""Orchestration: resolve corpus via env paths, run probes, emit verdict."""

from __future__ import annotations

import datetime
import hashlib
import os
import random

from .adapters.base import ModelAdapter, create_adapter
from .config import RunConfig
from .corpus import (
    corpus_digest,
    iter_documents,
    list_corpus_files,
    partition_holdout,
    resolve_dir,
)
from .index import GramIndex
from .probes import run_extraction, run_mia, run_verbatim, run_wordnet
from .probes.base import STATUS_FAIL, ProbeResult
from .records import RecordsWriter
from .verdict import build_verdict, write_verdict


def default_run_id(cfg: RunConfig) -> str:
    h = hashlib.sha256(
        f"{cfg.model_adapter}|{cfg.corpus_dir}|{cfg.seed}".encode()
    ).hexdigest()[:8]
    return f"clause16e-{datetime.date.today().isoformat()}-{cfg.model_adapter}-{h}"


class Harness:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.run_id = cfg.run_id or default_run_id(cfg)
        self.records_dir = cfg.records_dir.format(run_id=self.run_id)
        self.records = RecordsWriter(self.records_dir)
        self.adapter: ModelAdapter = create_adapter(cfg.model_adapter, cfg.model_params)
        self.rng = random.Random(cfg.seed)

    # -- corpus --------------------------------------------------------
    def _resolve_corpus(self):
        if not self.cfg.corpus_dir:
            raise ValueError("corpus.dir is required (env path, never a repo copy)")
        if not (self.cfg.train_globs or self.cfg.heldout_globs):
            raise ValueError("corpus.train_globs and/or corpus.heldout_globs required")
        resolve_dir(self.cfg.corpus_dir)
        for pat in self.cfg.train_globs + self.cfg.heldout_globs:
            if os.path.isabs(pat) or ".." in pat.split(os.sep):
                raise ValueError(f"corpus glob escapes corpus dir: {pat}")

    def corpus_digest(self) -> str:
        self._resolve_corpus()
        return corpus_digest(
            self.cfg.corpus_dir, self.cfg.train_globs + self.cfg.heldout_globs
        )

    def _index_cache_path(self, corpus_sha: str, n: int) -> str:
        return os.path.join(
            self.records_dir, f"gram-index-n{n}-{corpus_sha[:16]}.bin"
        )

    def build_index(self, corpus_sha: str) -> GramIndex:
        n = max(self.cfg.word_threshold, 25)
        cache = self._index_cache_path(corpus_sha, n)
        idx = GramIndex.load_cached(n, cache)
        if idx is not None:
            return idx
        idx = GramIndex(n, cache)
        idx.add_documents(
            d.text
            for d in iter_documents(
                self.cfg.corpus_dir,
                self.cfg.train_globs + self.cfg.heldout_globs,
                self.cfg.group_lines,
            )
        )
        idx.finalize()
        return idx

    # -- run -----------------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        self._resolve_corpus()
        corpus_sha = self.corpus_digest()
        self.records.write_json(
            "run-config.json",
            {
                "run_id": self.run_id,
                "config": cfg.raw,
                "resolved": {
                    "corpus_dir": cfg.corpus_dir,
                    "train_globs": cfg.train_globs,
                    "heldout_globs": cfg.heldout_globs,
                    "group_lines": cfg.group_lines,
                    "corpus_digest_sha256": corpus_sha,
                },
            },
        )

        results: list[ProbeResult] = []

        index = None
        need_index = cfg.verbatim_enabled or cfg.extraction_enabled
        if need_index:
            index = self.build_index(corpus_sha)
            self.records.write_json(
                "gram-index-summary.json",
                {"window_words": index.n, "unique_corpus_windows": len(index._sorted)},
            )

        if cfg.verbatim_enabled:
            train_docs = list(
                iter_documents(cfg.corpus_dir, cfg.train_globs, cfg.group_lines)
            )
            results.append(
                run_verbatim(cfg, self.adapter, index, train_docs, self.rng, self.records)
            )
        else:
            results.append(
                ProbeResult("verbatim", STATUS_FAIL, error="probe disabled in config")
            )

        if cfg.mia_enabled:
            if cfg.mia_split_mode == "same_population":
                # Matched-split mode: BOTH sets from the SAME train-shard
                # chunk population, disjoint by file (see
                # corpus.partition_holdout and docs/case-study.md).
                # list_corpus_files returns ABSOLUTE paths; chunk sources are
                # corpus-relative — normalize before the partition.
                holdout_files = {
                    os.path.relpath(f, cfg.corpus_dir)
                    for f in list_corpus_files(cfg.corpus_dir,
                                               cfg.mia_holdout_globs)
                }
                if not holdout_files:
                    results.append(
                        ProbeResult(
                            "mia", STATUS_FAIL,
                            error="same_population split requires mia."
                                  "holdout_globs matching train shards",
                        )
                    )
                    held_in = held_out = []
                else:
                    all_chunks = list(
                        iter_documents(
                            cfg.corpus_dir, cfg.train_globs, cfg.group_lines
                        )
                    )
                    held_in, held_out = partition_holdout(
                        all_chunks, holdout_files
                    )
            else:
                held_in = list(
                    iter_documents(cfg.corpus_dir, cfg.train_globs, cfg.group_lines)
                )
                held_out = list(
                    iter_documents(cfg.corpus_dir, cfg.heldout_globs, cfg.group_lines)
                )
            reference_adapter = None
            if cfg.mia_reference_enabled:
                reference_adapter = create_adapter(
                    cfg.mia_reference_adapter, cfg.mia_reference_params
                )
            if not cfg.heldout_globs and cfg.mia_split_mode != "same_population":
                results.append(
                    ProbeResult(
                        "mia", STATUS_FAIL,
                        error="no held-out slice configured (membership test needs one)",
                    )
                )
            else:
                results.append(
                    run_mia(
                        cfg, self.adapter, held_in, held_out, self.rng,
                        self.records, reference_adapter=reference_adapter,
                    )
                )
        else:
            results.append(
                ProbeResult("mia", STATUS_FAIL, error="probe disabled in config")
            )

        if cfg.extraction_enabled:
            train_docs = list(
                iter_documents(cfg.corpus_dir, cfg.train_globs, cfg.group_lines)
            )
            results.append(
                run_extraction(
                    cfg, self.adapter, index, train_docs, self.rng, self.records
                )
            )
        else:
            results.append(
                ProbeResult("extraction", STATUS_FAIL, error="probe disabled in config")
            )

        if cfg.wordnet_enabled:
            results.append(run_wordnet(cfg, self.adapter, self.records))
        else:
            results.append(
                ProbeResult("wordnet", STATUS_FAIL, error="probe disabled in config")
            )

        verdict = build_verdict(
            self.run_id,
            cfg,
            self.adapter.describe(),
            corpus_sha,
            results,
            self.records.manifest_dict(),
        )
        verdict_path = cfg.verdict_path.format(run_id=self.run_id)
        write_verdict(verdict, verdict_path)
        return verdict


__all__ = ["Harness", "default_run_id"]
