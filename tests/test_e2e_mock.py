"""End-to-end harness run over a synthetic corpus (no real data).

Proves the gate works: clean adapter -> PASS; corpus-echoing adapter
-> FAIL on verbatim+extraction; disabled probe -> FAIL (incomplete).
Also asserts the committed verdict carries no corpus text.
"""

import json
import os

import pytest

from clause16e.adapters.base import ModelAdapter, register_adapter
from clause16e.adapters.mock import MockAdapter
from clause16e.config import RunConfig
from clause16e.records import RecordsWriter
from clause16e.runner import Harness

WORDS = [f"tok{i:04d}" for i in range(4000)]


def _write_corpus(tmp_path, n_train_per_file=6, n_held_per_file=4, lines=8):
    train_dir = tmp_path / "train"
    held_dir = tmp_path / "heldout"
    train_dir.mkdir()
    held_dir.mkdir()
    counter = 0
    for f in range(2):
        rows = []
        for d in range(n_train_per_file):
            for line in range(lines):
                counter += 1
                rows.append(
                    " ".join(WORDS[(counter * 30 + k * 7) % len(WORDS)] for k in range(40))
                )
        (train_dir / f"shard{f}.txt").write_text("\n".join(rows), encoding="utf-8")
    counter += 50_000  # disjoint held-out range
    for f in range(2):
        rows = []
        for d in range(n_held_per_file):
            for line in range(lines):
                counter += 1
                rows.append(
                    " ".join(WORDS[(counter * 31 + k * 5) % len(WORDS)] for k in range(40))
                )
        (held_dir / f"shard{f}.txt").write_text("\n".join(rows), encoding="utf-8")
    synsets = [
        {"id": f"s{i}", "pos": "N", "gloss": "zqv mpl xkw jvr qhz",
         "example": "", "lemma": ["wts fkd"]}
        for i in range(3)
    ]
    wn = tmp_path / "synsets.jsonl"
    wn.write_text("\n".join(json.dumps(s) for s in synsets), encoding="utf-8")
    return wn


def _cfg(tmp_path, wn_path, **over):
    data = {
        "run_id": "e2e",
        "seed": 7,
        "model": {"adapter": "mock", "params": {}},
        "surface": {"local": True},
        "corpus": {
            "dir": str(tmp_path),
            "train_globs": ["train/*.txt"],
            "heldout_globs": ["heldout/*.txt"],
            "group_lines": 4,
        },
        "verbatim": {"enabled": True, "n_probes": 5},
        # E2E-only loose tolerances: the mock scores are hash-random, so
        # sample noise at this corpus size is large. The precise gating
        # math is covered by tests in test_mia.py.
        "mia": {
            "enabled": True,
            "auc_tolerance": 0.30,
            "success_rate_tolerance": 0.95,
        },
        "extraction": {"enabled": True, "n_per_template": 3},
        "wordnet": {"path": str(wn_path), "embedding_probe": False},
        "output": {
            "verdict_path": str(tmp_path / "reports/{run_id}.verdict.json"),
            "records_dir": str(tmp_path / "artifacts/{run_id}"),
        },
    }
    data.update(over)
    return RunConfig.from_dict(data)


def test_clean_mock_passes(tmp_path):
    wn = _write_corpus(tmp_path)
    cfg = _cfg(tmp_path, wn)
    verdict = Harness(cfg).run()
    assert verdict["verdict"] == "PASS", verdict["failures"]
    assert all(t["status"] == "PASS" for t in verdict["tests"].values())
    assert verdict["tests"]["verbatim"]["metrics"]["n_probes"] == 5
    assert verdict["retention_until"].startswith("20")


def test_corpus_echoing_adapter_fails(tmp_path):
    wn = _write_corpus(tmp_path)
    # capture one real (synthetic) corpus document to echo verbatim
    doc_text = None
    with open(tmp_path / "train" / "shard0.txt", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    doc_text = "\n".join(lines[:4])

    class LeakyMock(MockAdapter):
        name = "leaky-mock"

        def generate(self, prompt, max_new_tokens, temperature, seed):
            if len(prompt.split()) >= 5:
                return doc_text  # reproduce a full corpus document
            return super().generate(prompt, max_new_tokens, temperature, seed)

    register_adapter("leaky-mock", lambda params: LeakyMock(params))
    cfg = _cfg(tmp_path, wn, model={"adapter": "leaky-mock", "params": {}})
    verdict = Harness(cfg).run()
    assert verdict["verdict"] == "FAIL"
    assert verdict["tests"]["verbatim"]["status"] == "FAIL"
    assert verdict["tests"]["verbatim"]["metrics"]["n_outputs_with_violation"] > 0
    assert verdict["tests"]["extraction"]["status"] == "FAIL"


def test_disabled_probe_fails_the_gate(tmp_path):
    wn = _write_corpus(tmp_path)
    cfg = _cfg(tmp_path, wn, wordnet={"path": str(wn), "enabled": False})
    verdict = Harness(cfg).run()
    assert verdict["verdict"] == "FAIL"
    assert verdict["tests"]["wordnet"]["status"] == "FAIL"
    assert "disabled" in verdict["tests"]["wordnet"]["error"]


def test_verdict_artifact_contains_no_corpus_text(tmp_path):
    wn = _write_corpus(tmp_path)
    cfg = _cfg(tmp_path, wn)
    verdict = Harness(cfg).run()
    blob = json.dumps(verdict)
    with open(tmp_path / "train" / "shard0.txt", encoding="utf-8") as fh:
        for line in fh.read().splitlines():
            assert line not in blob
    # tokens of the corpus never appear in the committed artifact
    assert "tok0" not in blob


def test_records_dir_is_outside_git_artifacts(tmp_path):
    wn = _write_corpus(tmp_path)
    cfg = _cfg(tmp_path, wn)
    Harness(cfg).run()
    assert os.path.isdir(str(tmp_path / "artifacts" / "e2e"))
    flagged = os.path.join(str(tmp_path / "artifacts" / "e2e"), "verbatim-flagged.jsonl")
    assert os.path.isfile(flagged)
    assert os.path.getsize(flagged) == 0
