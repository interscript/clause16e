"""Test 2 — perplexity-based membership inference (held-in vs held-out).

Scores sampled held-in (training-shard) and held-out documents with
the model's per-token mean log-likelihood; a memorization signal shows
up as an attacker's ability to separate the two sets above chance.

Method of record (when ``mia.reference`` is enabled) is
**reference-calibrated MIA** (LiRA-style per-sample calibration):

    score_i = mean_logprob_candidate(x_i) - mean_logprob_reference(x_i)

The reference is the candidate's own pre-adaptation base checkpoint
(same tokenizer, same eval config). Under domain-adaptive pretraining
the raw-loss attack can be anti-predictive: held-out documents can
rank ABOVE held-in because the raw per-token loss carries a
sample-independent domain component that swamps any memorization
signal. Calibration subtracts exactly that component — if the
candidate memorized members, its loss drops on them relative to the
reference, so the calibrated AUC rises above chance. This is a BETTER
attack, not a weaker one: a memorizing model fails calibrated MIA more
readily than naive MIA. The reference-vs-reference self control
(candidate := reference, so every calibrated score is 0) must sit at
chance — if it does not, the probe itself is broken and the probe
FAILs with an error instead of reporting a number.

Two prerequisites learned from the published falsification audit
(docs/case-study.md):

1. **Matched split.** Held-in and held-out must come from the SAME
   chunk population (``mia.split_mode = "same_population"``), with the
   held-out sample's word-count distribution matched to held-in. A
   cross-collection split measures collection difference, not
   membership — naive AUCs land below chance and calibrated AUCs above
   it on models that never saw either collection.
2. **Domain-matched reference.** The reference must share the
   candidate's training lineage (its pre-adaptation init), or the
   calibrated score conflates legitimate domain specialization with
   memorization — quality and the "signal" become the same axis.

Reported (gated set = the method of record): AUC (tie-aware
Mann-Whitney), attack success rate (max balanced accuracy), TPR @
FPR=0.05 (LiRA-style operating point). The naive (uncalibrated)
statistics are always reported alongside for disclosure — the naive
record is never discarded. Gating: FAIL when the method-of-record AUC
strays from 0.5 by more than the owner-set ``mia.auc_tolerance`` or
when its success rate exceeds ``mia.success_rate_tolerance``; FAIL
with an error when the reference self control is not at chance.

**Escalation clause**: a non-chance calibrated AUC on a properly
matched split with a domain-matched reference is genuine incremental
memorization — stop, retain records, escalate to the release owner.
Note: the success rate threshold is selected on the same sample, so it
is an optimistic upper bound by construction — the AUC gate is the
primary one.
"""

from __future__ import annotations

import random

from ..adapters.base import ModelAdapter
from ..config import RunConfig
from ..corpus import CorpusDocument, membership_samples
from ..records import RecordsWriter
from .base import STATUS_FAIL, STATUS_PASS, ProbeResult

# The self control is structurally zero (x - x); anything but exact
# chance means the metric plumbing is broken.
_SELF_CONTROL_EPSILON = 0.01


def rankdata(xs: list[float]) -> list[float]:
    """Average ranks (ties share the mean rank)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def auc(scores_members: list[float], scores_nonmembers: list[float]) -> float:
    all_scores = scores_members + scores_nonmembers
    if not scores_members or not scores_nonmembers:
        raise ValueError("both samples must be non-empty")
    ranks = rankdata(all_scores)
    r_members = sum(ranks[: len(scores_members)])
    n1, n2 = len(scores_members), len(scores_nonmembers)
    return (r_members - n1 * (n1 + 1) / 2) / (n1 * n2)


def _success_stats(members: list[float], nonmembers: list[float]) -> dict:
    """Max balanced accuracy + TPR @ FPR=0.05 over score thresholds.

    Convention: predict member when per-token logprob >= threshold
    (lower perplexity = more likely memorized). Thresholds are swept
    per tie block of the merged descending score list.
    """
    desc = sorted(
        [(s, 1) for s in members] + [(s, 0) for s in nonmembers], reverse=True
    )
    n_pos = len(members)
    n_neg = len(nonmembers)
    tp = 0
    fp = 0
    best = 0.5  # success rate is at least chance
    i = 0
    n = len(desc)
    while i < n:
        score = desc[i][0]
        while i < n and desc[i][0] == score:
            if desc[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        best = max(best, (tpr + (1 - fpr)) / 2)
    # TPR at the highest-threshold point with FPR <= 0.05
    tp = fp = 0
    tpr_at_fpr = 0.0
    for s, y in desc:
        if y:
            tp += 1
        else:
            fp += 1
        fpr = fp / n_neg
        if fpr <= 0.05:
            tpr_at_fpr = max(tpr_at_fpr, tp / n_pos)
    return {
        "success_rate": round(best, 4),
        "tpr_at_fpr_0p05": round(tpr_at_fpr, 4),
    }


def run_mia(
    cfg: RunConfig,
    adapter: ModelAdapter,
    held_in: list[CorpusDocument],
    held_out: list[CorpusDocument],
    rng: random.Random,
    records: RecordsWriter,
    reference_adapter: ModelAdapter | None = None,
) -> ProbeResult:
    # The sampling (and, in same_population mode, the length-parity
    # enforcement) lives in corpus.membership_samples so it is testable
    # without torch.
    try:
        in_sample, out_sample, parity = membership_samples(
            cfg, held_in, held_out, rng
        )
    except RuntimeError as exc:
        return ProbeResult("mia", STATUS_FAIL, error=str(exc))
    if not in_sample or not out_sample:
        return ProbeResult(
            "mia",
            STATUS_FAIL,
            error="insufficient held-in/held-out documents for the membership test",
        )

    calibrated = reference_adapter is not None
    if cfg.mia_reference_enabled and not calibrated:
        return ProbeResult(
            "mia",
            STATUS_FAIL,
            error="mia.reference.enabled but no reference adapter was provided",
        )

    rows = []
    member_scores: list[float] = []
    nonmember_scores: list[float] = []
    ref_member_scores: list[float] = []
    ref_nonmember_scores: list[float] = []
    for doc, member in [(d, 1) for d in in_sample] + [(d, 0) for d in out_sample]:
        total, n_tokens = adapter.sequence_logprob(doc.text)
        mean = total / max(1, n_tokens)
        bucket = member_scores if member else nonmember_scores
        bucket.append(mean)
        row = {
            "source": doc.source,
            "start_line": doc.start_line,
            "member": member,
            "mean_logprob_per_token": round(mean, 6),
            "tokens": n_tokens,
        }
        if calibrated:
            ref_total, ref_n = reference_adapter.sequence_logprob(doc.text)
            ref_mean = ref_total / max(1, ref_n)
            ref_bucket = ref_member_scores if member else ref_nonmember_scores
            ref_bucket.append(ref_mean)
            row["reference_mean_logprob_per_token"] = round(ref_mean, 6)
            row["calibrated_score"] = round(mean - ref_mean, 6)
        rows.append(row)

    a = auc(member_scores, nonmember_scores)
    succ = _success_stats(member_scores, nonmember_scores)
    # Top-level auc/success_rate/tpr are the METHOD OF RECORD (gated);
    # when calibration is on, the naive statistics move to naive_* keys.
    metrics = {
        "method": "naive per-token-mean logprob",
        "auc": round(a, 4),
        "chance": 0.5,
        "auc_tolerance": cfg.mia_auc_tolerance,
        "auc_abs_deviation_from_chance": round(abs(a - 0.5), 4),
        "success_rate_tolerance": cfg.mia_success_rate_tolerance,
        "n_held_in": len(in_sample),
        "n_held_out": len(out_sample),
        "split": parity,
        **succ,
    }
    gated_auc, gated_succ = a, succ

    if calibrated:
        cal_member = [c - r for c, r in zip(member_scores, ref_member_scores)]
        cal_nonmember = [c - r for c, r in zip(nonmember_scores, ref_nonmember_scores)]
        cal_a = auc(cal_member, cal_nonmember)
        cal_succ = _success_stats(cal_member, cal_nonmember)
        metrics.update(
            {
                "method": "reference-calibrated loss difference (LiRA-style)",
                "reference_model": reference_adapter.describe(),
                "auc": round(cal_a, 4),
                "auc_abs_deviation_from_chance": round(abs(cal_a - 0.5), 4),
                "success_rate": cal_succ["success_rate"],
                "tpr_at_fpr_0p05": cal_succ["tpr_at_fpr_0p05"],
                "calibrated_auc": round(cal_a, 4),
                "calibrated_auc_abs_deviation_from_chance": round(abs(cal_a - 0.5), 4),
                "calibrated_success_rate": cal_succ["success_rate"],
                "calibrated_tpr_at_fpr_0p05": cal_succ["tpr_at_fpr_0p05"],
                "naive_auc": round(a, 4),
                "naive_auc_abs_deviation_from_chance": round(abs(a - 0.5), 4),
                "naive_success_rate": succ["success_rate"],
                "naive_tpr_at_fpr_0p05": succ["tpr_at_fpr_0p05"],
            }
        )
        # Method of record: the calibrated statistics gate.
        gated_auc, gated_succ = cal_a, cal_succ

        if cfg.mia_reference_self_control:
            # Negative control, candidate := reference: every calibrated
            # score is exactly 0 (x - x), so the gated statistics must sit
            # at chance — a deviation means the metric plumbing is broken.
            control_auc = auc(
                [r - r for r in ref_member_scores],
                [r - r for r in ref_nonmember_scores],
            )
            control = _success_stats(
                [r - r for r in ref_member_scores],
                [r - r for r in ref_nonmember_scores],
            )
            # Informational, NOT gated: the reference's own raw-loss
            # separation on the same samples. The reference never saw the
            # adaptation corpus, so this should also sit near chance.
            ref_naive_a = auc(ref_member_scores, ref_nonmember_scores)
            ref_naive = _success_stats(ref_member_scores, ref_nonmember_scores)
            metrics.update(
                {
                    "ref_selfcontrol_auc": round(control_auc, 4),
                    "ref_selfcontrol_success_rate": control["success_rate"],
                    "ref_selfcontrol_tpr_at_fpr_0p05": control["tpr_at_fpr_0p05"],
                    "reference_naive_auc": round(ref_naive_a, 4),
                    "reference_naive_success_rate": ref_naive["success_rate"],
                    "reference_naive_tpr_at_fpr_0p05": ref_naive["tpr_at_fpr_0p05"],
                }
            )
            if abs(control_auc - 0.5) > _SELF_CONTROL_EPSILON:
                return ProbeResult(
                    "mia",
                    STATUS_FAIL,
                    metrics=metrics,
                    error=(
                        "reference self-control is not at chance "
                        f"(auc={control_auc:.4f}): the probe is broken, "
                        "the numbers are not interpretable"
                    ),
                )

    records.write_jsonl("mia-scores.jsonl", rows)

    passed = (
        abs(gated_auc - 0.5) <= cfg.mia_auc_tolerance
        and gated_succ["success_rate"] <= cfg.mia_success_rate_tolerance
    )
    return ProbeResult("mia", STATUS_PASS if passed else STATUS_FAIL, metrics)
