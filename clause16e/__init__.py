"""clause16e — a license-cleanliness gate for text models.

Pre-release testing harness for any model trained wholly or partly on a
licensed, restricted, or otherwise protected text corpus. The gate has
four probes:

- **verbatim** — generation continuations matched against a fixed-length
  consecutive-word index of the protected corpus,
- **mia** — reference-calibrated membership inference on a matched
  held-in/held-out split (the method of record; see ``probes/mia.py``
  and ``docs/case-study.md``),
- **extraction** — adversarial extraction prompts checked against the
  corpus index and declared canaries,
- **wordnet** — tokenizer-vocabulary scan for reference-resource
  content (gloss/example N-grams, multiword lemma phrases).

A verdict is PASS only when all four probes ran and passed; a disabled
or unrunnable probe fails the gate.

Compliance invariants baked into the design:

- Protected-corpus content is read ONLY via environment-provided paths
  and is never copied into this package or its committed artifacts.
- All corpus processing is local; corpus-derived prompts are sent only
  to surfaces explicitly attested as local (``surface.local = true``).
- Committed verdict artifacts carry aggregates only (counts, AUC,
  pass/fail); flagged outputs and per-item scores go to a records
  directory that is gitignored, with a sha256 manifest referenced from
  the verdict (retention: 2 years, house practice).

The falsification record that shaped the method of record — why the
generic reference and the cross-collection split were both wrong, and
how the matched-split correction was derived — is published in
``docs/case-study.md``.
"""

__version__ = "0.1.0"

# Floor for the verbatim consecutive-word threshold, fixed by the
# original license agreement's extraction clause. May only be
# tightened, never loosened.
VERBATIM_WORD_FLOOR = 25
