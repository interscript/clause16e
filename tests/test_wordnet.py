import json

from clause16e.adapters.base import ModelAdapter, register_adapter
from clause16e.probes.wordnet import _content_ngrams, _vocab_violations, run_wordnet
from clause16e.records import RecordsWriter
from clause16e.normalize import tokenize


SYNTH_GLOSS = "zqv mpl xkw jvr qhz vbn fkd wts"  # 8 synthetic words


def _write_synsets(tmp_path):
    path = tmp_path / "synsets.jsonl"
    rows = [
        {"id": f"s{i:05d}", "pos": "N", "gloss": SYNTH_GLOSS, "example": "zqv mpl",
         "lemma": ["wts fkd"]}
        for i in range(5)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def test_content_ngrams_skip_single_word_lemmas():
    synsets = [{"gloss": "a b c d e", "example": "", "lemma": ["solo"]}]
    grams = _content_ngrams(synsets, 5)
    assert ("a", "b", "c", "d", "e") in grams
    assert all(len(g) >= 2 for g in grams)
    assert not any(g == ("solo",) for g in grams)


def test_multiword_lemma_is_indexed():
    synsets = [{"gloss": "", "example": "", "lemma": ["wts fkd"]}]
    assert ("wts", "fkd") in _content_ngrams(synsets, 5)


def test_vocab_hit_detection():
    synsets = [{"gloss": SYNTH_GLOSS, "example": "", "lemma": []}]
    grams = _content_ngrams(synsets, 5)
    hits = _vocab_violations(
        ["zqv-mpl-xkw-jvr-qhz-merged", "normal", "wts-fkd-pair"], grams
    )
    assert hits == ["zqv-mpl-xkw-jvr-qhz-merged"]


class _StaticVocabAdapter(ModelAdapter):
    name = "static"

    def __init__(self, params=None, vocab=None):
        self._vocab = vocab or []

    def generate(self, prompt, max_new_tokens, temperature, seed):
        return ""

    def sequence_logprob(self, text):
        return (0.0, 1)

    def vocab(self):
        return self._vocab


register_adapter("static", lambda params: _StaticVocabAdapter(params))


def test_run_wordnet_passes_clean_vocab(tmp_path):
    path = _write_synsets(tmp_path)
    records = RecordsWriter(str(tmp_path / "rec"))
    from clause16e.config import RunConfig

    cfg = RunConfig.from_dict({"model": {"adapter": "mock"}, "corpus": {"dir": "/x"},
                               "wordnet": {"path": path, "embedding_probe": False}})
    result = run_wordnet(cfg, _StaticVocabAdapter(vocab=["plain", "the"]), records)
    assert result.status == "PASS"
    assert result.metrics["vocab_ngram_hits"] == 0


def test_run_wordnet_fails_on_gloss_ngram_in_vocab(tmp_path):
    path = _write_synsets(tmp_path)
    records = RecordsWriter(str(tmp_path / "rec"))
    from clause16e.config import RunConfig

    cfg = RunConfig.from_dict({"model": {"adapter": "mock"}, "corpus": {"dir": "/x"},
                               "wordnet": {"path": path}})
    leaky_vocab = ["zqv", "mpl-xkw-jvr-qhz-vbn"]
    result = run_wordnet(cfg, _StaticVocabAdapter(vocab=leaky_vocab), records)
    assert result.status == "FAIL"
    assert result.metrics["vocab_ngram_hits"] == 1
    # Offending tokens go to private records, not into metrics
    assert "zqv" not in json.dumps(result.metrics)


def test_run_wordnet_requires_vocab(tmp_path):
    path = _write_synsets(tmp_path)
    records = RecordsWriter(str(tmp_path / "rec"))
    from clause16e.config import RunConfig

    cfg = RunConfig.from_dict({"model": {"adapter": "mock"}, "corpus": {"dir": "/x"},
                               "wordnet": {"path": path}})

    class NoVocab(_StaticVocabAdapter):
        def vocab(self):
            return None

    result = run_wordnet(cfg, NoVocab(), records)
    assert result.status == "FAIL"
    assert "vocabulary" in result.error


def test_run_wordnet_embedding_probe_submetric(tmp_path):
    from clause16e.adapters.mock import MockAdapter
    from clause16e.config import RunConfig

    path = _write_synsets(tmp_path)
    records = RecordsWriter(str(tmp_path / "rec"))
    cfg = RunConfig.from_dict(
        {"model": {"adapter": "mock"}, "corpus": {"dir": "/x"},
         "wordnet": {"path": path, "embedding_probe": True}}
    )
    result = run_wordnet(cfg, MockAdapter({}), records)
    emb = result.metrics["embedding_probe"]
    assert set(emb) == {"anchor_mean_cosine", "baseline_mean_cosine", "gap", "margin"}
    assert emb["gap"] <= emb["margin"]  # synthetic embeddings carry no signal
