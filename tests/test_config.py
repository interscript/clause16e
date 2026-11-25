import json

import pytest

from clause16e.config import RunConfig


def base_cfg(**over):
    data = {
        "run_id": "t",
        "model": {"adapter": "mock", "params": {}},
        "surface": {"local": True},
        "corpus": {
            "dir": "/tmp/corpus",
            "train_globs": ["train/*.txt"],
            "heldout_globs": ["heldout/*.txt"],
        },
        "wordnet": {"enabled": False},
    }
    data.update(over)
    return RunConfig.from_dict(json.loads(json.dumps(data)))


def test_threshold_floor_is_binding():
    with pytest.raises(ValueError, match="25-consecutive-word floor"):
        base_cfg(verbatim={"enabled": True, "word_threshold": 24})


def test_threshold_may_be_tightened():
    cfg = base_cfg(verbatim={"enabled": True, "word_threshold": 30})
    assert cfg.word_threshold == 30


def test_mia_tolerance_ranges():
    with pytest.raises(ValueError):
        base_cfg(mia={"auc_tolerance": 0.6})
    with pytest.raises(ValueError):
        base_cfg(mia={"success_rate_tolerance": 0.4})
    cfg = base_cfg()
    assert cfg.mia_auc_tolerance == 0.05
    assert cfg.mia_success_rate_tolerance == 0.60


def test_nonlocal_surface_rejects_corpus_prompts():
    with pytest.raises(ValueError, match="local-processing"):
        base_cfg(surface={"local": False})


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("MY_CORPUS_DIR", "/data/corpus")
    cfg = base_cfg(corpus={"dir": "${MY_CORPUS_DIR}"})
    assert cfg.corpus_dir == "/data/corpus"


def test_missing_env_var_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET_VAR", raising=False)
    with pytest.raises(KeyError, match="DEFINITELY_NOT_SET_VAR"):
        base_cfg(corpus={"dir": "${DEFINITELY_NOT_SET_VAR}"})


def test_output_path_templates():
    cfg = base_cfg()
    assert "{run_id}" in cfg.verdict_path
    assert "{run_id}" in cfg.records_dir
