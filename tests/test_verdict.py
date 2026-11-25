from clause16e.probes.base import STATUS_FAIL, STATUS_NOT_RUN, STATUS_PASS, ProbeResult
from clause16e.verdict import RETENTION_YEARS, build_verdict, verdict_from_results


def test_all_pass_is_pass():
    results = [
        ProbeResult("verbatim", STATUS_PASS, {"n": 1}),
        ProbeResult("mia", STATUS_PASS, {"auc": 0.5}),
        ProbeResult("extraction", STATUS_PASS),
        ProbeResult("wordnet", STATUS_PASS),
    ]
    verdict, failures = verdict_from_results(results)
    assert verdict == "PASS"
    assert failures == []


def test_any_fail_fails():
    results = [
        ProbeResult("verbatim", STATUS_PASS),
        ProbeResult("mia", STATUS_FAIL, error="auc drift"),
        ProbeResult("extraction", STATUS_PASS),
        ProbeResult("wordnet", STATUS_PASS),
    ]
    verdict, failures = verdict_from_results(results)
    assert verdict == "FAIL"
    assert any("mia" in f for f in failures)


def test_not_run_fails_the_gate():
    results = [
        ProbeResult("verbatim", STATUS_PASS),
        ProbeResult("mia", STATUS_PASS),
        ProbeResult("extraction", STATUS_PASS),
        ProbeResult("wordnet", STATUS_NOT_RUN, error="no vocab"),
    ]
    verdict, failures = verdict_from_results(results)
    assert verdict == "FAIL"


def test_build_verdict_shape():
    from clause16e.config import RunConfig

    cfg = RunConfig.from_dict(
        {"model": {"adapter": "mock"}, "corpus": {"dir": "/x"}, "wordnet": {"enabled": False}}
    )
    results = [
        ProbeResult("verbatim", STATUS_PASS, {"n_probes": 3}),
        ProbeResult("mia", STATUS_PASS, {"auc": 0.502}),
        ProbeResult("extraction", STATUS_PASS, {"n_prompts": 4}),
        ProbeResult("wordnet", STATUS_PASS, {"vocab_size": 10}),
    ]
    v = build_verdict(
        "run-1",
        cfg,
        {"adapter": "mock"},
        "deadbeef" * 8,
        results,
        {"dir": "/records", "files": {}},
    )
    assert v["verdict"] == "PASS"
    assert v["corpus_digest_sha256"] == "deadbeef" * 8
    assert v["thresholds"]["verbatim_word_threshold_floor"] == 25
    assert v["records"]["dir"] == "/records"
    assert len(v["retention_until"].split("-")[0]) == 4
    assert RETENTION_YEARS == 2
    # aggregates only: no freeform text fields beyond statuses/metrics
    blob = str(v)
    assert "output" not in v["tests"]["verbatim"]["metrics"]
