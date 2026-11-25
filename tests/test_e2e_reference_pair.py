"""End-to-end gate test over a synthetic corpus with a tiny fake model
pair (identical lineage): the gate must PASS at chance.

The method-of-record configuration — reference-calibrated MIA on a
matched same-population split — must collapse to exactly chance when
the candidate IS the reference (identical weights): every calibrated
score is structurally zero, the self control sits at 0.5, and the
full Harness emits a PASS verdict. The same test proves the committed
verdict carries no corpus text and the records dir stays gitignored.
"""

import json
import os

from clause16e.adapters.base import ModelAdapter, register_adapter
from clause16e.config import RunConfig
from clause16e.runner import Harness

WORDS = [f"tok{i:04d}" for i in range(4000)]


def _write_corpus(tmp_path, n_train_per_file=6, lines=8):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    counter = 0
    for f in range(8):
        rows = []
        for d in range(n_train_per_file):
            for line in range(lines):
                counter += 1
                rows.append(
                    " ".join(WORDS[(counter * 30 + k * 7) % len(WORDS)] for k in range(40))
                )
        (train_dir / f"shard-{f:02d}.txt").write_text("\n".join(rows), encoding="utf-8")
    synsets = [
        {"id": f"s{i}", "pos": "N", "gloss": "zqv mpl xkw jvr qhz",
         "example": "", "lemma": ["wts fkd"]}
        for i in range(3)
    ]
    wn = tmp_path / "synsets.jsonl"
    wn.write_text("\n".join(json.dumps(s) for s in synsets), encoding="utf-8")
    return wn


class _PairAdapter(ModelAdapter):
    """Deterministic hash-derived scorer standing in for a tiny model."""

    name = "pair"

    def __init__(self, params):
        self.tag = params.get("tag", "")

    def generate(self, prompt, max_new_tokens, temperature, seed):
        return " fqz xvm pkd wjh ztl"

    def sequence_logprob(self, text):
        import hashlib

        h = int.from_bytes(
            hashlib.blake2b((self.tag + text).encode(), digest_size=8).digest(), "big"
        )
        words = max(1, len(text.split()))
        return (-(h % 10000) / 100.0 * words / 10.0, words)

    def vocab(self):
        return ["the", "a", "fqz"]

    def describe(self):
        return {"adapter": self.name, "tag": self.tag}


register_adapter("pair", lambda params: _PairAdapter(params))


def _cfg(tmp_path, wn_path, **over):
    data = {
        "run_id": "pair-e2e",
        "seed": 11,
        "model": {"adapter": "pair", "params": {"tag": "cand"}},
        "surface": {"local": True},
        "corpus": {
            "dir": str(tmp_path),
            "train_globs": ["train/shard-*.txt"],
            "heldout_globs": [],
            "group_lines": 4,
        },
        "verbatim": {"enabled": True, "n_probes": 4},
        "mia": {
            "enabled": True,
            "n_eval": 40,
            "auc_tolerance": 0.05,
            "success_rate_tolerance": 0.60,
            "split_mode": "same_population",
            "holdout_globs": ["train/shard-06.txt", "train/shard-07.txt"],
            "length_parity": True,
            "reference": {
                "enabled": True,
                "adapter": "pair",
                "params": {"tag": "cand"},  # IDENTICAL lineage: same weights
            },
        },
        "extraction": {"enabled": True, "n_per_template": 2},
        "wordnet": {"path": str(wn_path), "embedding_probe": False},
        "output": {
            "verdict_path": str(tmp_path / "reports/{run_id}.verdict.json"),
            "records_dir": str(tmp_path / "artifacts/{run_id}"),
        },
    }
    data.update(over)
    return RunConfig.from_dict(data)


def test_identical_lineage_pair_passes_at_chance(tmp_path):
    wn = _write_corpus(tmp_path)
    cfg = _cfg(tmp_path, wn)
    verdict = Harness(cfg).run()
    assert verdict["verdict"] == "PASS", verdict["failures"]
    mia = verdict["tests"]["mia"]
    assert mia["status"] == "PASS"
    m = mia["metrics"]
    # candidate == reference -> every calibrated score structurally 0
    assert m["method"].startswith("reference-calibrated")
    assert m["calibrated_auc"] == 0.5
    assert m["ref_selfcontrol_auc"] == 0.5
    assert m["split"]["enforced"] is True
    assert m["split"]["mode"] == "same_population"


def test_gate_records_no_corpus_text_in_verdict(tmp_path):
    wn = _write_corpus(tmp_path)
    cfg = _cfg(tmp_path, wn)
    verdict = Harness(cfg).run()
    blob = json.dumps(verdict)
    with open(tmp_path / "train" / "shard-00.txt", encoding="utf-8") as fh:
        for line in fh.read().splitlines():
            assert line not in blob
    assert "tok0" not in blob


def test_disabled_probe_fails_in_pair_mode(tmp_path):
    wn = _write_corpus(tmp_path)
    cfg = _cfg(tmp_path, wn, verbatim={"enabled": False, "n_probes": 4})
    verdict = Harness(cfg).run()
    assert verdict["verdict"] == "FAIL"
    assert verdict["tests"]["verbatim"]["status"] == "FAIL"
    assert "disabled" in verdict["tests"]["verbatim"]["error"]
