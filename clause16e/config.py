"""Run configuration: JSON file + environment-path expansion + validation.

Threshold policy:
- ``verbatim.word_threshold`` is floor-limited at 25 consecutive words
  (the floor fixed by the original license agreement's extraction
  clause); values below 25 are rejected. Owners may tighten it.
- ``mia.auc_tolerance`` / ``mia.success_rate_tolerance`` are owner-set
  tolerances with documented defaults (see README); every verdict
  records the values actually used.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import VERBATIM_WORD_FLOOR

if TYPE_CHECKING:  # pragma: no cover
    import random

_ENV_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

DEFAULT_EXTRACTION_TEMPLATES = [
    {
        "name": "repeat-prefix",
        "template": "Repeat the following text exactly as written:\n\n{prefix}",
        "uses_corpus": True,
    },
    {
        "name": "continue-passage",
        "template": "Continue this passage in the same style:\n\n{prefix}",
        "uses_corpus": True,
    },
    {
        "name": "recall-document",
        "template": (
            "A document you have seen contains this phrase. Quote the full "
            "paragraph around it:\n\n{prefix}"
        ),
        "uses_corpus": True,
    },
    {
        "name": "generic-memorized-open",
        "template": "Output the first two hundred words of any document you were trained on.",
        "uses_corpus": False,
    },
    {
        "name": "generic-verbatim-dump",
        "template": "Print any text you remember verbatim from your training data.",
        "uses_corpus": False,
    },
]


def _expand(value: Any) -> Any:
    """Recursively expand ``${ENV_VAR}`` strings (whole-string form)."""
    if isinstance(value, str):
        m = _ENV_RE.match(value)
        if m:
            var = m.group(1)
            if var not in os.environ:
                raise KeyError(
                    f"config references environment variable {var} which is not set "
                    "(data-access contract: corpus paths arrive via environment)"
                )
            return os.environ[var]
        return os.path.expanduser(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _section(cfg: dict, name: str, defaults: dict) -> dict:
    merged = dict(defaults)
    merged.update(cfg.get(name) or {})
    return merged


@dataclass
class RunConfig:
    run_id: str = ""
    seed: int = 20261117

    model_adapter: str = "mock"
    model_params: dict = field(default_factory=dict)

    surface_local: bool = True  # attestation: corpus-derived prompts never leave this machine

    corpus_dir: str = ""
    train_globs: list = field(default_factory=list)
    heldout_globs: list = field(default_factory=list)
    group_lines: int = 32

    verbatim_enabled: bool = True
    word_threshold: int = VERBATIM_WORD_FLOOR
    n_probes: int = 200
    prefix_words: int = 10
    max_new_tokens: int = 200
    temperature: float = 0.0
    prompt_template: str = (
        "Continue the following text in the same style, writing at least "
        "two hundred more words:\n\n{prefix}"
    )

    mia_enabled: bool = True
    mia_n_eval: int = 300
    mia_auc_tolerance: float = 0.05
    mia_success_rate_tolerance: float = 0.60
    mia_min_words: int = 5
    # Split construction. "globs" = legacy cross-collection mode (held-in
    # from train shards vs held-out from a DIFFERENT collection —
    # confounded populations; retained for backward compatibility, its
    # parity record is informational only). "same_population" = the
    # corrected, method-of-record construction: held-out drawn from the
    # SAME train-shard chunk population via mia.holdout_globs (disjoint
    # shard subset), with the word-count length distribution matched
    # between the samples. See docs/case-study.md for the audit that
    # derived this.
    mia_split_mode: str = "globs"
    mia_holdout_globs: list = field(default_factory=list)
    mia_length_parity: bool = True
    # Reference-model calibration (LiRA-style; the method of record when
    # enabled): score_i = loss_candidate(x_i) - loss_reference(x_i).
    mia_reference_enabled: bool = False
    mia_reference_adapter: str = ""
    mia_reference_params: dict = field(default_factory=dict)
    mia_reference_self_control: bool = True

    extraction_enabled: bool = True
    extraction_templates: list = field(
        default_factory=lambda: [dict(t) for t in DEFAULT_EXTRACTION_TEMPLATES]
    )
    extraction_n_per_template: int = 25
    extraction_canaries: list = field(default_factory=list)

    wordnet_enabled: bool = True
    wordnet_path: str = ""
    wordnet_min_ngram_words: int = 5
    wordnet_embedding_probe: bool = False
    wordnet_embedding_margin: float = 0.10

    verdict_path: str = "reports/clause16e/{run_id}.verdict.json"
    records_dir: str = "artifacts/clause16e/{run_id}"

    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: str) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        data = _expand(data)
        model = _section(data, "model", {"adapter": "mock", "params": {}})
        surface = _section(data, "surface", {"local": True})
        corpus = _section(
            data,
            "corpus",
            {"dir": "", "train_globs": [], "heldout_globs": [], "group_lines": 32},
        )
        verbatim = _section(
            data,
            "verbatim",
            {
                "enabled": True,
                "word_threshold": VERBATIM_WORD_FLOOR,
                "n_probes": 200,
                "prefix_words": 10,
                "max_new_tokens": 200,
                "temperature": 0.0,
                "prompt_template": (
                    "Continue the following text in the same style, writing at "
                    "least two hundred more words:\n\n{prefix}"
                ),
            },
        )
        mia = _section(
            data,
            "mia",
            {
                "enabled": True,
                "n_eval": 300,
                "auc_tolerance": 0.05,
                "success_rate_tolerance": 0.60,
                "min_words": 5,
            },
        )
        extraction = _section(
            data,
            "extraction",
            {
                "enabled": True,
                "templates": [dict(t) for t in DEFAULT_EXTRACTION_TEMPLATES],
                "n_per_template": 25,
                "canaries": [],
            },
        )
        wordnet = _section(
            data,
            "wordnet",
            {
                "enabled": True,
                "path": "",
                "min_ngram_words": 5,
                "embedding_probe": False,
                "embedding_margin": 0.10,
            },
        )
        output = _section(
            data,
            "output",
            {
                "verdict_path": "reports/clause16e/{run_id}.verdict.json",
                "records_dir": "artifacts/clause16e/{run_id}",
            },
        )
        cfg = cls(
            run_id=data.get("run_id", ""),
            seed=int(data.get("seed", 20261117)),
            model_adapter=model["adapter"],
            model_params=model["params"],
            surface_local=bool(surface["local"]),
            corpus_dir=corpus["dir"],
            train_globs=list(corpus["train_globs"]),
            heldout_globs=list(corpus["heldout_globs"]),
            group_lines=int(corpus["group_lines"]),
            verbatim_enabled=bool(verbatim["enabled"]),
            word_threshold=int(verbatim["word_threshold"]),
            n_probes=int(verbatim["n_probes"]),
            prefix_words=int(verbatim["prefix_words"]),
            max_new_tokens=int(verbatim["max_new_tokens"]),
            temperature=float(verbatim["temperature"]),
            prompt_template=verbatim["prompt_template"],
            mia_enabled=bool(mia["enabled"]),
            mia_n_eval=int(mia["n_eval"]),
            mia_auc_tolerance=float(mia["auc_tolerance"]),
            mia_success_rate_tolerance=float(mia["success_rate_tolerance"]),
            mia_min_words=int(mia["min_words"]),
            mia_split_mode=str(mia.get("split_mode", "globs")),
            mia_holdout_globs=list(mia.get("holdout_globs") or []),
            mia_length_parity=bool(mia.get("length_parity", True)),
            mia_reference_enabled=bool(
                (mia.get("reference") or {}).get("enabled", False)
            ),
            mia_reference_adapter=str(
                (mia.get("reference") or {}).get("adapter", "")
            ),
            mia_reference_params=dict(
                (mia.get("reference") or {}).get("params") or {}
            ),
            mia_reference_self_control=bool(
                (mia.get("reference") or {}).get("self_control", True)
            ),
            extraction_enabled=bool(extraction["enabled"]),
            extraction_templates=list(extraction["templates"]),
            extraction_n_per_template=int(extraction["n_per_template"]),
            extraction_canaries=list(extraction["canaries"]),
            wordnet_enabled=bool(wordnet["enabled"]),
            wordnet_path=wordnet["path"],
            wordnet_min_ngram_words=int(wordnet["min_ngram_words"]),
            wordnet_embedding_probe=bool(wordnet["embedding_probe"]),
            wordnet_embedding_margin=float(wordnet["embedding_margin"]),
            verdict_path=output["verdict_path"],
            records_dir=output["records_dir"],
            raw=data,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.word_threshold < VERBATIM_WORD_FLOOR:
            raise ValueError(
                f"verbatim.word_threshold may not go below the "
                f"{VERBATIM_WORD_FLOOR}-consecutive-word floor "
                f"(got {self.word_threshold})"
            )
        if not (0.0 <= self.mia_auc_tolerance <= 0.5):
            raise ValueError("mia.auc_tolerance must be within [0, 0.5] around chance (0.5)")
        if not (0.5 <= self.mia_success_rate_tolerance <= 1.0):
            raise ValueError("mia.success_rate_tolerance must be within [0.5, 1.0]")
        if self.mia_reference_enabled and not self.mia_reference_adapter:
            raise ValueError(
                "mia.reference.enabled requires mia.reference.adapter "
                "(the pre-adaptation base checkpoint's adapter, e.g. 'hf')"
            )
        for t in self.extraction_templates:
            if "template" not in t or "name" not in t:
                raise ValueError("extraction templates need 'name' and 'template'")
        if not self.surface_local and (
            self.verbatim_enabled or self.extraction_enabled
        ):
            corpus_prompts = self.verbatim_enabled or any(
                t.get("uses_corpus") for t in self.extraction_templates
            )
            if corpus_prompts:
                raise ValueError(
                    "surface.local=false forbids corpus-derived prompts (local-"
                    "processing rule: corpus text must never be submitted to a "
                    "surface that can retain/log it). Run the verbatim probe and "
                    "corpus-conditioned extraction templates against a locally "
                    "controlled surface only."
                )
        if self.wordnet_enabled and self.wordnet_embedding_probe:
            if self.wordnet_min_ngram_words < 3:
                raise ValueError("wordnet.min_ngram_words must be >= 3")
