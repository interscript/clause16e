# The falsification record: how the gate nearly shipped broken

This is the published case study behind the method of record. It is
included because the instrument's value comes from having survived its
own falsification audit: an earlier version of this gate — naive
statistics first, then a generic-reference calibrated variant on a
cross-collection split — produced confident, wrong verdicts that only
a structured audit caught. Every number below is an aggregate metric
(AUC, CER, shingle counts); no protected content is reproduced.

Setting: a vocalization (diacritization) candidate line gated before
release under a corpus license that prohibits releasing any model from
which the licensed resources can be extracted. The gate is the
release precondition. The models are seq2seq encoder-decoder
(byte-level tokenizer) candidates at the same tier.

## Round 0 — the naive gate nearly blocked on nothing / passed on nothing

The first gated candidate was a domain-adaptive pretraining (DAPT)
checkpoint trained directly on the licensed corpus. It FAILED the
gate — correctly, as it turned out — but the number was wrong in an
instructive way:

| attack | AUC | reading |
|---|---|---|
| naive per-token-mean loss | **0.4432** (below chance, anti-predictive) | held-out documents ranked ABOVE held-in |
| reference-calibrated loss difference | **0.6319** (deviation 0.1319 ≫ 0.05 tolerance) | genuine memorization |

The naive inversion was NOT a benign artifact: raw loss carries a
sample-independent domain component, and DAPT shifts it enough that
held-out ranks above held-in — hiding the real above-chance separation
that calibration exposed. The reference self control sat at exactly
0.5, so the probe was sound: the candidate genuinely carried protected
passages. The candidate was withdrawn from the release line and
retired from every downstream label-source role.

**Lesson 1: never conclude "no memorization" from an inverted naive
AUC.** Calibration is a better attack, not a weaker one.

## Round 1 — distillation did not launder the signal

Next arm: keep the memorizing teacher internal, distill a student on
teacher OUTPUTS (never the licensed text), with a strict
regurgitation guard — drop any training pair whose output shares a
normalized 8-word shingle with the protected corpus. The student
beats the quality bar (full-split CER 0.03692 vs the 3.74% champion)
and passes all content probes. The calibrated gate:

| candidate | calibrated AUC (generic ref, legacy split) |
|---|---|
| DAPT teacher (round 0) | 0.6319 |
| distilled student | **0.6393** |

The signal survived distillation almost unchanged. The distillation
channel transferred the teacher's function INCLUDING its memorization:
the student inherits the loss-level signal without ever tripping the
exact n-gram guard (guard drops were 0 across the corpus). Kill
criterion invoked; the arm was abandoned.

**Lesson 2: exact-shingle guards cannot see loss-level memorization,
and students inherit their teacher's defects through the targets.**

## Round 2 — the clean-room candidate FAILED HIGHER (the falsification)

Root-attack arm: rebuild the line with a teacher whose verified
lineage never touched the protected corpus (public-data base + a
public-lineage corpus only, four independent records), distill a
student from it. Quality improved again (full-split CER 0.03641 —
best of the line). The gate, same configuration (generic hub
reference, legacy split):

| candidate | lineage | calibrated AUC |
|---|---|---|
| DAPT teacher | trained ON the protected corpus | 0.6319 |
| distilled student | distilled from it | 0.6393 |
| clean-room student | **never exposed** | **0.6475** |

**A never-exposed model scored HIGHEST.** The monotone-in-cleanness
prediction failed: if the instrument measured memorization of the
protected corpus, the clean candidate must sit at chance. It sat
furthest from chance.

### The discriminator: zero shingle overlap

The cheap alternative explanation — the clean-room training data
secretly shared text with the protected corpus — was killed with the
exact guard machinery: 554,196 8-word shingles indexed from the
protected corpus plane; the clean-room training corpus (439,786
pairs) shares **0** shingles with it (0.0% of rows, per-row histogram
all zeros). Combined with the lineage verification, the candidate
provably never saw the held-in text.

### Reference-invariance

Re-running the clean candidate against a DOMAIN-MATCHED reference
(the candidate's own lineage base, verified clean) moved the number
barely: **0.6357** vs 0.6475 generic — still FAIL. The reference
choice was not the (only) confound.

**Lesson 3: a signal that rises monotonically with model quality,
survives a reference swap, and tops out on a provably-clean model is
not a memorization signal. It is an instrument artifact. Stop and
audit the instrument.**

## Round 3 — the split-confound audit

The final audit examined the probe's split construction, not the
model:

1. Held-in = 32-line chunks of the training shards; held-out = 32-line
   chunks of a DIFFERENT collection (a held-out document set).
2. Empirically (the clean candidate's per-item scores, 440 rows):
   held-in median **4,058 tokens** with 75th/90th percentiles AT the
   4,096 truncation cap; held-out median **3,283 tokens** with an
   80-token tail. **Truncated, correlated windows vs complete
   documents** — opposite length/truncation profiles.
3. BOTH models find held-in easier (candidate loss/token 4.625 <
   4.735; reference 3.431 < 3.785); the candidate−reference gap is
   asymmetric (1.194 vs 0.950); the naive AUCs agree in direction —
   **0.4243 and 0.3803, both below chance**, on a membership-free
   candidate.

**Reading: the comparison was confounded by split construction.** Two
structurally different collections, scored by two models, separate in
both directions at once — the probe read the set difference (format/
register/length/truncation), not membership. The full chain — zero
overlap, verified clean lineage, monotone AUC-vs-quality,
reference-invariance — admits only that explanation.

**Lesson 4: a membership test's held-in and held-out sets must be the
same population. Any other comparison measures the populations.**

## The corrected instrument

Two fixes, both now the method of record in this repo:

1. **Matched split** (`mia.split_mode = "same_population"`): both
   membership sets drawn from the SAME train-shard chunk population,
   disjoint by file; the held-out sample's word-count distribution
   matched to held-in via decile quantile bands; never-empty fallback
   below the floor; the achieved medians and fallback flag recorded in
   every verdict.
2. **Domain-matched reference** (mandatory): the candidate's own
   pre-adaptation init — same tokenizer, same eval config, verified
   never trained on the protected corpus. Parameterized in remote
   gate jobs by the `GATE_REFERENCE_HUB` / `GATE_REFERENCE_PATH` /
   `GATE_REFERENCE_SUBPATH` precedence (HUB > PATH > SUBPATH).

Result, same clean-room candidate, corrected instrument:

| metric | value |
|---|---|
| calibrated AUC | **0.4933** (deviation 0.0067 ≤ 0.05) → **PASS** |
| reference self control | 0.5 exactly (harness soundness precondition) |
| calibrated success rate | 0.5267 (tolerance 0.6) |
| TPR @ FPR=0.05 | 0.0633 |
| split record | enforced, same_population, in/out median 507/494 words, parity pass |
| extraction / verbatim / wordnet | PASS (0 violations) |

The same instrument, same configuration, still holds the round-0
memorizer above tolerance — a non-chance result on a matched split
with a domain-matched reference is genuine incremental memorization
(the **escalation clause**: stop, retain records, escalate).

## The audit checklist this record buys you

When a membership gate reports a signal on a candidate you believe is
clean (or clean on a candidate you believe is dirty), before touching
the model:

1. Check the **self control** — is the probe structurally sound?
2. Check the **discriminator** — does the candidate provably overlap
   the protected text (hashed shingles), and is the lineage verified?
3. Check **monotonicity** — does the "signal" rise with model quality?
   Quality and memorization are different axes; a signal that tracks
   quality is tracking quality.
4. Check **reference-invariance** — swap in a domain-matched reference;
   if the number barely moves, the reference was not the confound.
5. Audit the **split** — same population? matched length/truncation
   profiles? both models' naive AUCs below chance on a clean candidate
   is the signature of a collection-difference confound.

This repository implements 1, 2 (via the probe suite), 4 and 5 as
method of record; 3 is the reading discipline the case study teaches.
