# clause16e

**A license-cleanliness gate for text models.** Pre-release testing
that answers one question before any model trained on licensed,
restricted, or otherwise protected text corpora ships: *can the
protected text be extracted from the model, or has the model merely
learned the domain?*

The harness pairs a **reference-calibrated LiRA-style membership
inference (MIA)** on a **matched split** with three content probes
(**extraction**, **verbatim**, **vocabulary leakage**). It is the
instrument behind a published falsification record: an earlier version
of this gate — generic reference, cross-collection split — was caught
by its own audit scoring a provably-never-exposed model *higher* than
a memorizing one (AUC 0.6319 → 0.6393 → 0.6475 while lineage
cleanliness improved; see [docs/case-study.md](docs/case-study.md)).
The corrected instrument passed the clean candidate at chance
(0.4933, deviation 0.0067) and held the memorizer above tolerance.
The name comes from the license clause that first required the gate
(a clause 16(e) extraction/membership-testing condition in a corpus
license agreement).

- License: BSD-3-Clause.
- Python >= 3.9; the core is stdlib-only (torch/transformers are
  optional, needed only by the `hf` adapter).
- No corpus content ever enters this repo: corpus access is
  environment-paths only, committed verdicts are aggregates-only, and
  flagged outputs go to a gitignored records dir with a sha256
  manifest.

## Why calibration + a matched split (the method of record)

### 1. Naive loss MIA fails under domain-adaptive training

The naive attack scores each document with the candidate's per-token
mean log-likelihood and checks whether held-in documents separate from
held-out ones. Under domain-adaptive pretraining (DAPT) this attack
can be **anti-predictive**: raw loss carries a sample-independent
*domain* component that swamps any memorization signal, and held-out
documents can rank ABOVE held-in. In the published case study the
naive AUC was 0.4432 (below chance) on a model that **genuinely had
memorized** — calibration exposed it.

The method of record is the **reference-calibrated loss difference**:

```
score_i = mean_logprob_candidate(x_i) − mean_logprob_reference(x_i)
```

where the **reference** is the candidate's own pre-adaptation base
checkpoint (same tokenizer, same eval config). Calibration subtracts
the domain component; if the candidate memorized members, its loss
drops on them *relative to the reference*, so the calibrated AUC rises
above chance. This is a **better attack, not a weaker one**: a
memorizing model fails calibrated MIA more readily than naive MIA.
The naive statistics are always disclosed alongside (`naive_*` keys).

Two controls ride with calibration:

- **Reference self control** (candidate := reference, so every
  calibrated score is structurally 0) must sit at exactly chance. Any
  deviation means the probe itself is broken; the probe FAILs with an
  error instead of reporting an interpretable number.
- **Domain-matched reference.** The reference must share the
  candidate's training lineage. A generic reference conflates
  legitimate domain specialization with memorization — in the case
  study, quality and the "signal" became the same axis (see below).

### 2. The split must be matched (same population, matched lengths)

Held-in and held-out must come from the **same chunk population**:
disjoint shards of the *same* training corpus format (`mia.split_mode
= "same_population"` with `mia.holdout_globs` naming the held-out
shard subset). The held-out sample's word-count **distribution** is
matched to held-in via decile quantile bands, with a **never-empty
fallback** (below the floor, the full held-out population is sampled —
it is the same population by construction). The achieved medians and
a fallback flag are recorded in every verdict's `mia.split` record.

A cross-collection split — held-in from training shards vs held-out
from a different collection — is the classic confound: opposite
length/truncation profiles let ANY model separate the sets. In the
case study both models' naive AUCs sat below chance (0.4243 / 0.3803)
on a membership-free candidate while calibrated AUCs sat above it:
the probe was reading the set difference, not membership. Full audit:
[docs/case-study.md](docs/case-study.md).

### 3. The escalation clause

**A non-chance calibrated AUC on a matched split with a domain-matched
reference is genuine incremental memorization** — memorization beyond
what legitimate lineage training explains. Stop, retain records,
escalate to the release owner. A failing or incomplete gate means the
candidate does not release; mitigate (data mix, dedup, regularization,
unlearning, internal-teacher + distillation on never-exposed labels)
and re-test.

## The four probes

A verdict is **PASS only when all four probes ran and passed**. A
disabled or unrunnable probe fails the gate — an incomplete record is
not a releasable record.

| Probe | What it does | Pass criterion |
|---|---|---|
| `verbatim` | Generates continuations from prefixes of corpus documents; matches outputs against a fixed-length consecutive-word index of the corpus | Zero outputs reproducing >= 25 consecutive corpus words (floor fixed by the underlying license agreement; configurable only upward) |
| `mia` | Per-token-mean log-likelihood membership attack on a matched held-in/held-out split; method of record = reference-calibrated loss difference when `mia.reference` is configured | Calibrated AUC within owner tolerance of 0.5 AND calibrated success rate within owner tolerance; reference self control at chance; naive statistics disclosed alongside |
| `extraction` | Adversarial extraction prompt templates against the model/surface | Zero outputs reproducing >= threshold words; zero canary hits |
| `wordnet` | Tokenizer-vocabulary scan for reference-resource gloss/example/multiword-lemma N-grams; optional embedding-gap sub-probe | Zero synset-content N-grams embedded in the vocab |

## Install

```sh
pip install -e 'clause16e'           # core: stdlib only
pip install -e 'clause16e[hf]'       # + local transformers adapter
pip install -e 'clause16e[test]'     # + pytest
```

## Run

```sh
export CLAUSE16E_CORPUS_DIR=/path/to/protected-corpus/normalized-shards
export CLAUSE16E_WORDNET_PATH=/path/to/reference-synsets.jsonl
export CLAUSE16E_MODEL_PATH=/path/to/candidate-checkpoint     # hf adapter
export CLAUSE16E_REFERENCE_MODEL_PATH=/path/to/pre-adaptation-base   # when calibrated

clause16e validate --config configs/example-run.json
clause16e run      --config configs/example-matched-split.json
```

Exit codes: `0` = PASS verdict, `1` = FAIL verdict or config error,
`2` = usage error. Console output is aggregates only.

The corpus layout: normalized sentence-per-line UTF-8 shards under
`train/` (training members) and `heldout/` (held-out documents) — or,
in matched-split mode, any disjoint shard subsets of `train/`.

## Config reference (defaults)

```jsonc
{
  "seed": 20261117,
  "model":   {"adapter": "hf | http | mock", "params": {"...": "..."}},
  "surface": {"local": true},                       // local-processing attestation
  "corpus":  {"dir": "${CLAUSE16E_CORPUS_DIR}",
              "train_globs": ["train/*.txt"],
              "heldout_globs": ["heldout/*.txt"],
              "group_lines": 32},                   // pseudo-doc = N lines (0 = file)
  "verbatim": {"word_threshold": 25,                // agreement floor; tighten only
               "n_probes": 200, "prefix_words": 10,
               "max_new_tokens": 200, "temperature": 0.0},
  "mia":      {"n_eval": 300,
               "auc_tolerance": 0.05,               // owner-set; FAIL if |AUC-0.5| > tol
               "success_rate_tolerance": 0.60,      // owner-set; max balanced accuracy
               "split_mode": "globs | same_population",
               "holdout_globs": [],                 // same_population: held-out shard subset
               "length_parity": true,
               "reference": {"enabled": true, "adapter": "hf",
                              "params": {"...": "..."},
                              "self_control": true}},
  "extraction": {"n_per_template": 25, "canaries": [], "templates": ["..."]},
  "wordnet":  {"path": "${CLAUSE16E_WORDNET_PATH}",
               "min_ngram_words": 5,                // gloss-scale consecutive-word rule
               "embedding_probe": false},
  "output":   {"verdict_path": "reports/clause16e/{run_id}.verdict.json",
               "records_dir": "artifacts/clause16e/{run_id}"}
}
```

Shipped examples: `configs/example-run.json` (naive MIA),
`configs/example-reference-calibrated.json` (calibrated method of
record), `configs/example-matched-split.json` (calibrated + matched
split — the full method of record).

### Threshold policy

- **25 consecutive words (verbatim)** — fixed by the underlying
  license agreement's extraction clause. The config value is validated
  to be >= 25; tighten (e.g. 20) for stricter gating, never loosen.
- **MIA tolerances — owner-set, defaults proposed**: `auc_tolerance`
  0.05 (AUC must sit within [0.45, 0.55]) and
  `success_rate_tolerance` 0.60. Every verdict records the values
  actually used. Note the success rate is an optimistic upper bound
  (threshold tuned on the eval sample); the AUC gate is primary.

### Reference parameterization (house pattern)

In remote gate jobs the reference model arrives via the environment,
with a strict precedence:

| Variable | Meaning | Precedence |
|---|---|---|
| `GATE_REFERENCE_HUB` | hub model id (e.g. the candidate's published init) | **1 (wins)** — mount checks skipped; `from_pretrained` fails loudly on a bad id |
| `GATE_REFERENCE_PATH` | explicit mount-absolute path override | 2 |
| `GATE_REFERENCE_SUBPATH` | path under the checkpoint mount | 3 |

All three resolve into `CLAUSE16E_REFERENCE_MODEL_PATH` for the
harness run. The reference must be **domain-matched** — the
candidate's own pre-adaptation init, same tokenizer, verified never
trained on the protected corpus. A generic reference falsifies the
gate (case study, round 3).

### Adapters

- `mock` — deterministic, zero-dep; used by the test suite and CI.
- `hf` — local transformers (ByT5/mt5/T5-family auto-detected via
  `config.is_encoder_decoder`; causal LMs supported). Fully local, so
  it satisfies the local-processing rule for corpus-derived prompts.
- `http` — self-hosted runtime surface (`/v1/generate`,
  `/v1/logprob`, optional `/v1/vocab`, `/v1/embed`). Auth token comes
  from an env var named in `params.token_env`, never from config.
- Custom runtime surfaces: `clause16e.adapters.base.register_adapter`.

### Reference-resource input (the `wordnet` probe)

JSONL, one synset per line: `id`, `pos`, `gloss`, `example`, `lemma`
(string or list), optional `relations`. Single-word lemmas are
deliberately NOT flagged (they are ordinary words of the language);
only gloss/example N-grams (N >= 5 words) and multiword lemma phrases
count as evidence of copying. Offending vocab tokens are written to
the private records dir only, never to the verdict.

## Compliance model

- **Corpus access is environment-paths only**: `corpus.dir` /
  `wordnet.path` are `${ENV_VAR}` references resolved at runtime.
  Nothing from the corpus is copied into this repo, bundled into
  artifacts, or committed.
- **Local processing**: corpus-derived prompts (the verbatim probe,
  `uses_corpus: true` extraction templates) are only sent when
  `surface.local = true` is attested in the config — i.e. the surface
  is locally controlled (local HF transformers, or a self-hosted
  runtime on loopback). The config loader refuses otherwise.
- **Aggregates-only verdicts**: the committed verdict artifact carries
  counts, AUC, pass/fail, seeds, config, a corpus provenance digest,
  and sha256s of the private record files — never corpus text and
  never model outputs that contain corpus text.
- **Retention records**: flagged outputs and per-item scores go to the
  run's records dir (`artifacts/clause16e/<run_id>/`, gitignored),
  sha256-manifested in the verdict, with `retention_until` stamped
  (house practice: 2 years). The record is auditable without ever
  committing corpus-derived content.

## Verdict artifact

`reports/clause16e/<run_id>.verdict.json` — the file the release round
consumes. `schema_version`, `run_id`, `date`, `verdict` (PASS/FAIL),
per-probe `status` + aggregate `metrics`, thresholds used, seed, model
descriptor, corpus digest (`corpus_digest_sha256` — identifies the
corpus files used without containing them), the `mia.split` record,
records manifest with sha256s, and `retention_until`.

## Remote gate job

`hfjob/gate_job.py` (+ `hfjob/launch.sh`) runs a full gate on a remote
container runner (HF Jobs): clone the gate repo, decode the
reference-resource TSV, seed the gram-index cache (corpus-digest
gated), `clause16e validate` + `clause16e run`, and checkpoint the
verdict and every record file to the records mount as they appear — a
dead job still leaves its partial records. Mount specs are
environment-provided (`CORPUS_MOUNT_SPEC`, `CKPT_MOUNT_SPEC`,
`RECORDS_MOUNT_SPEC`); no defaults are shipped. `GATE_SMOKE=1` runs
mount/clone/decode/config validation only.

## Scope notes

- The corpus-side N-gram index uses 64-bit truncated BLAKE2b hashes
  with a ~6e-6 collision bound at 15M windows; a collision can only
  over-report a violation, never hide one.
- `wordnet.embedding_probe` is a heuristic secondary signal (gloss
  anchor vs random-baseline embedding similarity gap); the vocab scan
  is the authoritative verbatim check.
- The MIA success rate uses a threshold tuned on the eval sample —
  reported as an upper bound alongside AUC and TPR@FPR=0.05.

## Tests

```sh
python3 -m pytest tests -q
```

The suite runs over synthetic fixtures only — it never touches real
corpus data. It includes the regression pins from the audit (path-shape
partition safety, quantile-band length matching, never-empty fallback,
dual-id publish resolution) and an end-to-end gate test in which a
tiny fake model pair of identical lineage must PASS at chance.

## Falsification record

[docs/case-study.md](docs/case-study.md) — the full "how we nearly
shipped a broken instrument" record: the naive-inversion near-miss,
the three-model monotone AUC-vs-quality table that falsified the
generic-reference gate, the zero-shingle-overlap discriminator, the
split-confound audit, and the corrected instrument's clean pass.
