"""Regression pins for the remote gate job (hfjob/gate_job.py).

- dual-id publish resolution: the harness writes artifacts under the
  CONFIG's run_id while GATE_RUN_ID (the records subdir) may carry a
  job suffix (-v2 after a crashed attempt) — publish must resolve the
  verdict/records under BOTH ids, never hide them behind the suffix,
- reference resolution precedence: HUB > PATH > SUBPATH.
"""

import json

import pytest

from hfjob.gate_job import reference_location, resolve_publish_ids


def _write_config(path, run_id):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"run_id": run_id}, fh)


# -- dual-id publish resolution -------------------------------------------

def test_publish_ids_resolves_config_run_id(tmp_path):
    cfg = tmp_path / "run.json"
    _write_config(cfg, "gate-run-001")
    assert resolve_publish_ids(str(cfg), "gate-run-001") == ["gate-run-001"]


def test_publish_ids_appends_config_id_when_job_suffix_differs(tmp_path):
    """THE DUAL-ID REGRESSION: a job-suffix GATE_RUN_ID (-v2 after a
    crashed attempt) must not hide artifacts written under the
    config's run_id."""
    cfg = tmp_path / "run.json"
    _write_config(cfg, "gate-run-001")
    ids = resolve_publish_ids(str(cfg), "gate-run-001-v2")
    assert ids == ["gate-run-001-v2", "gate-run-001"]


def test_publish_ids_survives_unreadable_config(tmp_path):
    cfg = tmp_path / "broken.json"
    cfg.write_text("{not json", encoding="utf-8")
    assert resolve_publish_ids(str(cfg), "gate-run-001") == ["gate-run-001"]
    assert resolve_publish_ids(str(tmp_path / "missing.json"), "r") == ["r"]


def test_publish_ids_empty_config_run_id_is_dropped(tmp_path):
    cfg = tmp_path / "run.json"
    _write_config(cfg, "")
    assert resolve_publish_ids(str(cfg), "gate-run-001") == ["gate-run-001"]


# -- reference resolution precedence (HUB > PATH > SUBPATH) ----------------

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("GATE_REFERENCE_HUB", "GATE_REFERENCE_PATH",
                "GATE_REFERENCE_SUBPATH", "GATE_CKPT_MOUNT"):
        monkeypatch.delenv(var, raising=False)
    import importlib

    import hfjob.gate_job as gj

    importlib.reload(gj)
    yield
    importlib.reload(gj)


def test_reference_none_when_unset():
    assert reference_location() == (None, None, None)


def test_reference_subpath_resolves_under_ckpt_mount(monkeypatch):
    monkeypatch.setenv("GATE_REFERENCE_SUBPATH", "run-base/best")
    import importlib

    import hfjob.gate_job as gj

    importlib.reload(gj)
    hub, path, sub = gj.reference_location()
    assert (hub, sub) == (None, None)
    assert path == f"{gj.CKPT_ROOT}/run-base/best"


def test_reference_path_overrides_subpath(monkeypatch):
    monkeypatch.setenv("GATE_REFERENCE_SUBPATH", "run-base/best")
    monkeypatch.setenv("GATE_REFERENCE_PATH", "/ckpt/other/ref")
    import importlib

    import hfjob.gate_job as gj

    importlib.reload(gj)
    hub, path, sub = gj.reference_location()
    assert (hub, sub) == (None, None)
    assert path == "/ckpt/other/ref"


def test_reference_hub_wins_over_both(monkeypatch):
    """A hub id is not a filesystem path: it wins and skips the mount
    check (from_pretrained fails loudly on a bad id)."""
    monkeypatch.setenv("GATE_REFERENCE_SUBPATH", "run-base/best")
    monkeypatch.setenv("GATE_REFERENCE_PATH", "/ckpt/other/ref")
    monkeypatch.setenv("GATE_REFERENCE_HUB", "org/model-small")
    import importlib

    import hfjob.gate_job as gj

    importlib.reload(gj)
    hub, path, sub = gj.reference_location()
    assert (path, sub) == (None, None)
    assert hub == "org/model-small"
