from clause16e.normalize import normalize_word, tokenize


def test_punctuation_is_a_word_boundary_not_a_deletion():
    # apostrophes split (both sides of the gate use the same rule,
    # so matching stays symmetric)
    assert tokenize("Hello, World! It's fine.") == ["hello", "world", "it", "s", "fine"]
    assert tokenize("foo,bar") == tokenize("foo, bar") == ["foo", "bar"]


def test_arabic_tashkeel_is_ignored():
    diacritized = "\u0643\u0650\u062a\u064e\u0627\u0628"  # k-i t-a b with harakat
    plain = "\u0643\u062a\u0627\u0628"
    assert normalize_word(diacritized) == normalize_word(plain)
    assert tokenize(f"{diacritized} \u062e\u0648\u0628.") == tokenize(
        f"{plain} \u062e\u0648\u0628"
    )


def test_format_chars_dropped():
    assert normalize_word("abc‌def") == "abcdef"  # ZWNJ removed
    assert tokenize("a b") == ["a", "b"]  # nbsp splits tokens


def test_empty_and_nonword_tokens_removed():
    assert tokenize("... — !!!") == []
    assert tokenize("") == []
