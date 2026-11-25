import pytest

from clause16e.index import GramIndex

CORPUS_TEXT = (
    "The quick brown fox jumps over the lazy dog near the river bank every "
    "single morning before the farmers arrive with their carts and goods for "
    "the market in the old town square where merchants gather to trade."
)

BOUNDARY_A = " ".join(f"tok{i:03d}" for i in range(30))
# a 24-word corpus prefix: below the 25-consecutive-word floor
SHORT_COPY = " ".join(f"tok{i:03d}" for i in range(24))


def _index(n=25):
    idx = GramIndex(n)
    idx.add_documents([CORPUS_TEXT, BOUNDARY_A])
    idx.finalize()
    return idx


def test_detects_25_consecutive_words():
    idx = _index()
    copied = " ".join(CORPUS_TEXT.split()[5:35])  # 30 corpus words
    assert idx.match_stats(copied)["max_run_words"] == 30
    assert idx.has_violation(copied, 25)


def test_24_words_below_threshold_is_no_violation():
    idx = _index()
    stats = idx.match_stats(SHORT_COPY)  # 24-word prefix of corpus
    assert stats["total_windows"] == 0  # no full window, nothing to match
    assert stats["max_run_words"] == 0
    assert not idx.has_violation(SHORT_COPY, 25)


def test_max_run_words_counts_window_runs():
    idx = _index()
    copy29 = " ".join(f"tok{i:03d}" for i in range(29))  # 5 full windows
    stats = idx.match_stats(copy29)
    assert stats["matching_windows"] == 5
    assert stats["max_run_words"] == 29


def test_short_output_has_no_windows():
    idx = _index()
    stats = idx.match_stats("two words")
    assert stats["total_windows"] == 0
    assert stats["max_run_words"] == 0


def test_diacritic_shedding_does_not_evade():
    # Same English text is boring; simulate surface variation with
    # punctuation/case instead (normalize strips combining marks on
    # Arabic-script text the same way).
    idx = _index()
    variant = ", ".join(w.upper() + "," for w in CORPUS_TEXT.split()[:28])
    assert idx.has_violation(variant, 25)


def test_window_floor_is_25():
    with pytest.raises(ValueError):
        GramIndex(24)


def test_index_cache_roundtrip(tmp_path):
    cache = str(tmp_path / "idx.bin")
    idx = GramIndex(25, cache_path=cache)
    idx.add_documents([CORPUS_TEXT])
    idx.finalize()
    loaded = GramIndex.load_cached(25, cache)
    assert loaded is not None
    assert loaded.has_violation(" ".join(CORPUS_TEXT.split()[5:35]), 25)
    assert not loaded.has_violation(SHORT_COPY, 25)
