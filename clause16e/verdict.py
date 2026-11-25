"""Verdict artifact: the release round's PASS/FAIL gate input.

The committed artifact carries aggregates only — counts, AUC, pass/
fail, config, seeds, corpus digest, sha256 of the private record
files — never corpus text or model outputs containing corpus text. A
verdict is PASS only when every required probe ran and passed; NOT_RUN
or blocked probes fail the gate (a model cannot release on an
incomplete test record).
"""

from __future__ import annotations

import datetime
import json
import os

from .probes.base import STATUS_PASS, ProbeResult

SCHEMA_VERSION = 1
# Records-retention policy (house practice): memorization-testing
# records are kept 2 years. The verdict stamps the date the retention
# window ends.
RETENTION_YEARS = 2


def verdict_from_results(results: list[ProbeResult]) -> tuple[str, list[str]]:
    failures = []
    for r in results:
        if r.status != STATUS_PASS:
            failures.append(
                f"{r.probe}: {r.status}"
                + (f" ({r.error})" if r.error else "")
            )
    return ("FAIL" if failures else "PASS", failures)


def build_verdict(
    run_id: str,
    cfg,
    model_desc: dict,
    corpus_digest_value: str,
    results: list[ProbeResult],
    records_manifest: dict,
) -> dict:
    verdict, failures = verdict_from_results(results)
    today = datetime.date.today()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "date": today.isoformat(),
        "verdict": verdict,
        "failures": failures,
        "retention_until": (today.replace(year=today.year + RETENTION_YEARS)).isoformat(),
        "model": model_desc,
        "seed": cfg.seed,
        "thresholds": {
            "verbatim_word_threshold_floor": 25,
            "verbatim_word_threshold_used": cfg.word_threshold,
            "mia_auc_tolerance": cfg.mia_auc_tolerance,
            "mia_success_rate_tolerance": cfg.mia_success_rate_tolerance,
        },
        "corpus_digest_sha256": corpus_digest_value,
        "surface_local": cfg.surface_local,
        "tests": {r.probe: r.to_dict() for r in results},
        "records": records_manifest,
    }


def write_verdict(verdict: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    return path
