"""Test 1 — verbatim memorization probe (consecutive-word reproduction).

Generates continuations from prefixes of known corpus documents and
flags any output reproducing >= ``word_threshold`` consecutive words
of corpus text (floor 25, fixed by the underlying license agreement's
extraction clause; owners may tighten). The corpus index never leaves
this machine, and corpus-derived prompts are only sent to surfaces
attested local (``surface.local``).
"""

from __future__ import annotations

import random

from ..adapters.base import ModelAdapter
from ..config import RunConfig
from ..corpus import CorpusDocument
from ..index import GramIndex
from ..records import RecordsWriter
from .base import STATUS_FAIL, STATUS_PASS, ProbeResult


def _prefix_text(doc: CorpusDocument, prefix_words: int) -> str:
    words = doc.text.split()
    return " ".join(words[:prefix_words])


def run_verbatim(
    cfg: RunConfig,
    adapter: ModelAdapter,
    index: GramIndex,
    docs: list[CorpusDocument],
    rng: random.Random,
    records: RecordsWriter,
) -> ProbeResult:
    if not docs:
        return ProbeResult(
            "verbatim", STATUS_FAIL, error="no corpus documents available for prefixes"
        )
    sample = docs if len(docs) <= cfg.n_probes else rng.sample(docs, cfg.n_probes)

    flagged = []
    n_outputs_with_violation = 0
    n_violating_windows = 0
    max_run = 0
    for doc in sample:
        prefix = _prefix_text(doc, cfg.prefix_words)
        prompt = cfg.prompt_template.format(prefix=prefix)
        output = adapter.generate(prompt, cfg.max_new_tokens, cfg.temperature, cfg.seed)
        stats = index.match_stats(output)
        violation = (
            stats["max_run_words"] >= cfg.word_threshold
            and stats["matching_windows"] > 0
        )
        max_run = max(max_run, stats["max_run_words"])
        n_violating_windows += stats["matching_windows"]
        if violation:
            n_outputs_with_violation += 1
            # Corpus-derived detail goes ONLY to the private records dir.
            flagged.append(
                {
                    "probe": "verbatim",
                    "source": doc.source,
                    "start_line": doc.start_line,
                    "max_run_words": stats["max_run_words"],
                    "output": output,
                }
            )

    records.write_jsonl("verbatim-flagged.jsonl", flagged)
    metrics = {
        "threshold_words": cfg.word_threshold,
        "n_probes": len(sample),
        "n_outputs_with_violation": n_outputs_with_violation,
        "n_violating_windows": n_violating_windows,
        "max_consecutive_words_observed": max_run,
    }
    status = STATUS_PASS if n_outputs_with_violation == 0 else STATUS_FAIL
    return ProbeResult("verbatim", status, metrics)
