import pytest

from clause16e.probes.mia import _success_stats, auc, run_mia


def test_auc_perfect_separation():
    assert auc([10.0, 9.0, 8.0], [1.0, 2.0, 3.0]) == 1.0


def test_auc_identical_distributions_is_chance():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert auc(list(xs), list(xs)) == 0.5


def test_auc_ties_count_half():
    # every pair tied -> 0.5
    assert auc([5.0, 5.0], [5.0, 5.0]) == 0.5
    # one member above two nonmembers, one tie
    a = auc([7.0, 3.0], [3.0, 3.0])
    assert 0.5 < a < 1.0


def test_auc_inverted_signal():
    assert auc([1.0, 2.0], [9.0, 8.0]) == 0.0


def test_success_stats_perfect():
    members = [10.0, 9.5, 9.0, 8.5]
    nonmembers = [1.0, 1.5, 2.0, 2.5]
    stats = _success_stats(members, nonmembers)
    assert stats["success_rate"] == 1.0
    assert stats["tpr_at_fpr_0p05"] == 1.0


def test_success_stats_chance_floor():
    # identical scores -> best balanced accuracy is chance (0.5)
    stats = _success_stats([3.0, 3.0, 3.0], [3.0, 3.0, 3.0])
    assert stats["success_rate"] == 0.5


# -- run_mia gating (deterministic scoring adapters) ----------------------

import random

from clause16e.adapters.base import ModelAdapter
from clause16e.config import RunConfig
from clause16e.corpus import CorpusDocument
from clause16e.records import RecordsWriter


class _ScoringAdapter(ModelAdapter):
    name = "scoring"

    def __init__(self, fn):
        self.fn = fn

    def generate(self, prompt, max_new_tokens, temperature, seed):
        return ""

    def sequence_logprob(self, text):
        return (self.fn(text), 10)


def _docs(word):
    return [CorpusDocument(f"{word} doc {i} some words", "f.txt", i, i) for i in range(6)]


def _mia_cfg():
    return RunConfig.from_dict(
        {
            "model": {"adapter": "mock"},
            "corpus": {"dir": "/x"},
            "wordnet": {"enabled": False},
            "mia": {"auc_tolerance": 0.05},
        }
    )


def test_run_mia_fails_on_memorization_signal(tmp_path):
    cfg = _mia_cfg()
    attacker = _ScoringAdapter(lambda t: 1.0 if "aaa" in t else -1.0)
    result = run_mia(
        cfg, attacker, _docs("aaa"), _docs("zzz"), random.Random(1),
        RecordsWriter(str(tmp_path / "r")),
    )
    assert result.status == "FAIL"
    assert result.metrics["auc"] == 1.0
    assert result.metrics["success_rate"] == 1.0


def test_run_mia_passes_at_chance(tmp_path):
    cfg = _mia_cfg()
    blind = _ScoringAdapter(lambda t: 0.0)
    result = run_mia(
        cfg, blind, _docs("aaa"), _docs("zzz"), random.Random(1),
        RecordsWriter(str(tmp_path / "r")),
    )
    assert result.status == "PASS"
    assert result.metrics["auc"] == 0.5


def test_run_mia_needs_both_slices(tmp_path):
    cfg = _mia_cfg()
    blind = _ScoringAdapter(lambda t: 0.0)
    result = run_mia(
        cfg, blind, _docs("aaa"), [], random.Random(1),
        RecordsWriter(str(tmp_path / "r")),
    )
    assert result.status == "FAIL"
    assert "held-out" in result.error


# -- reference-model calibration (method of record) ------------------------


def _ref_cfg(reference=True):
    data = {
        "model": {"adapter": "mock"},
        "corpus": {"dir": "/x"},
        "wordnet": {"enabled": False},
    }
    if reference:
        data["mia"] = {
            "reference": {
                "enabled": True,
                "adapter": "mock",
                "params": {"note": "same tokenizer/eval config as the candidate"},
            }
        }
    return RunConfig.from_dict(data)


def test_reference_enabled_requires_adapter():
    # from_dict validates, so the error surfaces at load time.
    with pytest.raises(ValueError, match="mia.reference.adapter"):
        RunConfig.from_dict(
            {
                "model": {"adapter": "mock"},
                "corpus": {"dir": "/x"},
                "wordnet": {"enabled": False},
                "mia": {"reference": {"enabled": True}},
            }
        )


def test_calibration_recovers_signal_from_inverted_naive(tmp_path):
    # The round-0 case-study story (docs/case-study.md): domain adaptation
    # makes held-out docs rank ABOVE held-in under raw loss
    # (anti-predictive, AUC -> 0), while the candidate's relative drop on
    # members is only visible after subtracting the reference's domain
    # component.
    candidate = _ScoringAdapter(lambda t: 0.5 if "aaa" in t else 1.0)
    reference = _ScoringAdapter(lambda t: 0.0 if "aaa" in t else 0.55)
    result = run_mia(
        _ref_cfg(), candidate, _docs("aaa"), _docs("zzz"), random.Random(1),
        RecordsWriter(str(tmp_path / "r")), reference_adapter=reference,
    )
    assert result.metrics["method"].startswith("reference-calibrated")
    assert result.metrics["naive_auc"] == 0.0  # anti-predictive raw loss
    assert result.metrics["calibrated_auc"] == 1.0  # recovered by calibration
    assert result.metrics["auc"] == 1.0  # gated set = method of record
    assert result.status == "FAIL"  # auc tolerance still binds on calibrated


def test_calibration_passes_when_calibrated_at_chance(tmp_path):
    candidate = _ScoringAdapter(lambda t: 0.5 if "aaa" in t else 1.0)
    reference = _ScoringAdapter(lambda t: 0.0 if "aaa" in t else 0.5)
    result = run_mia(
        _ref_cfg(), candidate, _docs("aaa"), _docs("zzz"), random.Random(1),
        RecordsWriter(str(tmp_path / "r")), reference_adapter=reference,
    )
    assert result.status == "PASS"
    assert result.metrics["calibrated_auc"] == 0.5
    assert result.metrics["ref_selfcontrol_auc"] == 0.5


def test_naive_memorization_still_fails_calibrated(tmp_path):
    # A memorizing candidate fails the calibrated attack too: calibration
    # is a better attack, not a weaker one.
    candidate = _ScoringAdapter(lambda t: 2.0 if "aaa" in t else 1.0)
    reference = _ScoringAdapter(lambda t: 1.0)
    result = run_mia(
        _ref_cfg(), candidate, _docs("aaa"), _docs("zzz"), random.Random(1),
        RecordsWriter(str(tmp_path / "r")), reference_adapter=reference,
    )
    assert result.status == "FAIL"
    assert result.metrics["calibrated_auc"] == 1.0
    assert result.metrics["naive_auc"] == 1.0


def test_self_control_not_at_chance_fails_the_probe(tmp_path):
    # A broken metrics plumbing cannot produce a self-control away from
    # chance with structurally-zero scores; simulate it by patching auc.
    import clause16e.probes.mia as mia_mod

    real_auc = mia_mod.auc
    mia_mod.auc = lambda m, n: 0.83 if all(x == 0.0 for x in m + n) else real_auc(m, n)
    try:
        result = run_mia(
            _ref_cfg(), _ScoringAdapter(lambda t: 0.0),
            _docs("aaa"), _docs("zzz"), random.Random(1),
            RecordsWriter(str(tmp_path / "r")),
            reference_adapter=_ScoringAdapter(lambda t: 0.0),
        )
    finally:
        mia_mod.auc = real_auc
    assert result.status == "FAIL"
    assert "self-control" in result.error


def test_enabled_reference_without_adapter_errors(tmp_path):
    # Runner contract: config says reference, runner must supply one.
    result = run_mia(
        _ref_cfg(), _ScoringAdapter(lambda t: 0.0),
        _docs("aaa"), _docs("zzz"), random.Random(1),
        RecordsWriter(str(tmp_path / "r")),
    )
    assert result.status == "FAIL"
    assert "reference adapter" in result.error


def test_scores_record_reference_columns(tmp_path):
    w = RecordsWriter(str(tmp_path / "r"))
    run_mia(
        _ref_cfg(),
        _ScoringAdapter(lambda t: 0.5),
        _docs("aaa"), _docs("zzz"), random.Random(1),
        w, reference_adapter=_ScoringAdapter(lambda t: 0.25),
    )
    import json

    rows = [json.loads(l) for l in open(w.directory + "/mia-scores.jsonl")]
    assert all("calibrated_score" in r and "reference_mean_logprob_per_token" in r for r in rows)
    assert rows[0]["calibrated_score"] == 0.025  # (0.5 - 0.25) / 10 tokens
