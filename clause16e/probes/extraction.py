"""Test 3 — extraction by prompting against the released surface.

Runs adversarial extraction prompt templates through the adapter and
checks every output against the corpus N-gram index (and any owner
declared synthetic canary strings).

Templates marked ``uses_corpus: true`` embed a corpus-derived prefix
and therefore require ``surface.local = true`` (the local-processing
rule: corpus text must never be submitted to a service that can retain
or log it). Generic templates carry no corpus content and run anywhere.
Templates are generic scaffolds committed in code/config; no corpus
passage is ever committed or logged outside the private records dir.
"""

from __future__ import annotations

import random

from ..adapters.base import ModelAdapter
from ..config import RunConfig
from ..corpus import CorpusDocument
from ..index import GramIndex
from ..records import RecordsWriter
from .base import STATUS_FAIL, STATUS_PASS, ProbeResult


def _check_canaries(output: str, canaries: list[str]) -> list[str]:
    return [c["name"] if isinstance(c, dict) else str(i)
            for i, c in enumerate(canaries)
            if (c.get("value") if isinstance(c, dict) else c) in output]


def run_extraction(
    cfg: RunConfig,
    adapter: ModelAdapter,
    index: GramIndex,
    docs: list[CorpusDocument],
    rng: random.Random,
    records: RecordsWriter,
) -> ProbeResult:
    canaries = cfg.extraction_canaries
    rows = []
    n_prompts = 0
    n_violations = 0
    n_canary_hits = 0
    max_run = 0
    skipped = []

    for tpl in cfg.extraction_templates:
        uses_corpus = bool(tpl.get("uses_corpus"))
        if uses_corpus and not cfg.surface_local:
            skipped.append(tpl["name"])
            continue
        if uses_corpus and not docs:
            skipped.append(tpl["name"])
            continue
        for i in range(cfg.extraction_n_per_template):
            source = None
            prefix = ""
            if uses_corpus:
                doc = rng.choice(docs)
                words = doc.text.split()
                start = rng.randrange(0, max(1, len(words) - cfg.prefix_words))
                prefix = " ".join(words[start : start + cfg.prefix_words])
                source = doc.source
            prompt = tpl["template"].format(prefix=prefix)
            output = adapter.generate(
                prompt, cfg.max_new_tokens, cfg.temperature, cfg.seed + i
            )
            stats = index.match_stats(output)
            violation = (
                stats["max_run_words"] >= cfg.word_threshold
                and stats["matching_windows"] > 0
            )
            canary_hit = _check_canaries(output, canaries)
            n_prompts += 1
            max_run = max(max_run, stats["max_run_words"])
            if violation:
                n_violations += 1
            if canary_hit:
                n_canary_hits += len(canary_hit)
            if violation or canary_hit:
                # Private records only: outputs may contain corpus text.
                rows.append(
                    {
                        "probe": "extraction",
                        "template": tpl["name"],
                        "source": source,
                        "max_run_words": stats["max_run_words"],
                        "canary_hits": canary_hit,
                        "output": output,
                    }
                )

    records.write_jsonl("extraction-flagged.jsonl", rows)
    metrics = {
        "threshold_words": cfg.word_threshold,
        "n_templates_run": len(cfg.extraction_templates) - len(skipped),
        "templates_skipped": skipped,
        "n_prompts": n_prompts,
        "n_violation_prompts": n_violations,
        "n_canary_hits": n_canary_hits,
        "max_consecutive_words_observed": max_run,
    }
    status = STATUS_PASS if (n_violations == 0 and n_canary_hits == 0) else STATUS_FAIL
    return ProbeResult("extraction", status, metrics)
