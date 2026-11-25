# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.5.1",
#   "transformers==4.46.3",
# ]
# ///
"""Remote gate job (HF Jobs or any similar container runner).

Runs the clause16e gate against a protected corpus + candidate (and
reference) checkpoints mounted read-only, checkpointing every private
record file to a records mount as it appears — a dead job still leaves
its partial records.

Compliance:
  - protected-corpus text exists only inside the mounted read-only
    channel; this script prints AGGREGATE markers only (stage names,
    counts) — never corpus text or model output text.
  - the model surface is the in-process HF transformers adapter
    (configs run with surface.local=true); no third-party inference API.
  - private records (flagged outputs, scores, index cache) are written
    to the records mount for retention; only the aggregates-only
    verdict leaves it.

Flow: clone the gate repo/branch -> decode the reference-resource TSV
-> seed the gram-index cache (corpus-digest-gated) -> clause16e
validate -> clause16e run -> checkpoint verdict+records to the records
mount. A watchdog re-checkpoints new record files every few minutes.
The job exits 0 whenever a verdict artifact was produced (PASS or FAIL
are both completed gates); exit 1 means the gate itself did not run.

Env (all optional unless noted):
  GH_TOKEN        GitHub token for cloning the gate repo (required)
  GATE_REPO       repo to clone               (interscript/clause16e)
  GATE_BRANCH     branch to clone             (main)
  GATE_RUN_ID     run id / records subdir     (default: derived from config)
  GATE_CONFIG     config filename under configs/  (example-reference-calibrated.json)
  GATE_MODEL_SUBPATH      candidate checkpoint path under the checkpoint mount (required)
  GATE_REFERENCE_HUB      reference as a hub model id (wins if set)
  GATE_REFERENCE_PATH     reference as a mount-absolute path override
  GATE_REFERENCE_SUBPATH  reference path under the checkpoint mount (default)
  GATE_CORPUS_MOUNT / GATE_CKPT_MOUNT / GATE_RECORDS_MOUNT / GATE_WORK_MOUNT
                  mount points (defaults /data, /ckpt, /gate, /work)
  GATE_CORPUS_GLOBS       space-separated corpus globs for the digest/seed
                          (default "train/*.txt")
  GATE_WORDNET_TSV        reference-resource TSV to decode (default
                          "<corpus mount>/wordnet/synsets.tsv")
  CLAUSE16E_DEVICE        torch device for the adapter (cuda)
  GATE_FLAVOR / GATE_SMOKE  runner flavor marker / mount-check-only mode

Reference precedence: **HUB > PATH > SUBPATH** (the house pattern). A
hub id is not a filesystem path: when GATE_REFERENCE_HUB is set, the
reference mount check is skipped (from_pretrained fails loudly on a
bad id). The reference must be DOMAIN-MATCHED — the candidate's own
pre-adaptation init, same tokenizer, verified never trained on the
protected corpus; see docs/case-study.md for why a generic reference
falsifies the gate.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

REPO_DEFAULT = "interscript/clause16e"
BRANCH_DEFAULT = "main"
CONFIG_DEFAULT = "example-reference-calibrated.json"

REPO_DIR = "/repo"
CORPUS_DIR = os.environ.get("GATE_CORPUS_MOUNT", "/data")
CKPT_ROOT = os.environ.get("GATE_CKPT_MOUNT", "/ckpt")
GATE_DIR = os.environ.get("GATE_RECORDS_MOUNT", "/gate")
WORK_DIR = os.environ.get("GATE_WORK_MOUNT", "/work")
CORPUS_GLOBS = os.environ.get("GATE_CORPUS_GLOBS", "train/*.txt").split()
WORDNET_TSV = os.environ.get(
    "GATE_WORDNET_TSV", f"{CORPUS_DIR}/wordnet/synsets.tsv"
)
WORDNET_JSONL = f"{WORK_DIR}/synsets.jsonl"
SEED_CACHE_DIR = os.environ.get("GATE_SEED_CACHE_DIR", f"{GATE_DIR}/seed-cache")

MODEL_SUBPATH = os.environ.get("GATE_MODEL_SUBPATH", "")
# Reference resolution precedence: HUB > PATH > SUBPATH. A hub id is
# not a bucket path: when GATE_REFERENCE_HUB is set, the reference
# mount check is skipped (from_pretrained fails loudly on a bad id).
REFERENCE_HUB = os.environ.get("GATE_REFERENCE_HUB", "")
# GATE_REFERENCE_PATH is an explicit mount-absolute override (takes
# precedence over REFERENCE_SUBPATH; GATE_REFERENCE_HUB still wins if
# both are set).
REFERENCE_PATH = os.environ.get("GATE_REFERENCE_PATH", "")
REFERENCE_SUBPATH = os.environ.get("GATE_REFERENCE_SUBPATH", "")

RUN_ID = os.environ.get("GATE_RUN_ID", "")
REPO = os.environ.get("GATE_REPO", REPO_DEFAULT)
BRANCH = os.environ.get("GATE_BRANCH", BRANCH_DEFAULT)
CONFIG_PATH = (
    f"{REPO_DIR}/configs/{os.environ.get('GATE_CONFIG', CONFIG_DEFAULT)}"
)
HEAD_SHA = ""
_stage = "init"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


STARTED = utc()


class Progress:
    """Stage markers + heartbeat into the records dir on the records mount."""

    def __init__(self, path: str):
        self.path = path
        self._stop = threading.Event()
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()

    def mark(self, stage: str, note: str = "") -> None:
        global _stage
        with self._lock:
            _stage = stage
            line = f"{utc()} stage={stage}"
            if note:
                line += f" {note}"
            self._fh.write(line + "\n")
            self._fh.flush()
            print(line, flush=True)

    def _beat(self) -> None:
        while not self._stop.wait(60):
            with self._lock:
                self._fh.write(f"{utc()} heartbeat stage={_stage}\n")
                self._fh.flush()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._fh.close()


def scrub(text: str) -> str:
    tok = os.environ.get("GH_TOKEN", "")
    return text.replace(tok, "***") if tok else text


def run_cmd(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.update(kw.pop("extra_env", {}))
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, **kw)
    proc.stdout = scrub(proc.stdout or "")
    proc.stderr = scrub(proc.stderr or "")
    return proc


def clone_repo(prog: Progress) -> str:
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN secret not set; cannot clone the gate repo")
    repo = REPO
    branch = BRANCH
    url = f"https://x-access-token:{token}@github.com/{repo}.git"
    proc = run_cmd(
        ["git", "clone", "--depth", "1", "--branch", branch, url, REPO_DIR]
    )
    if proc.returncode != 0:
        raise RuntimeError(f"clone failed: {proc.stderr.strip()[:400]}")
    head = run_cmd(["git", "-C", REPO_DIR, "rev-parse", "HEAD"])
    sha = head.stdout.strip()
    prog.mark("cloned", f"branch={branch} sha={sha[:12]}")
    return sha


def decode_wordnet(prog: Progress) -> int:
    """TSV -> JSONL: columns id/pos/gloss/example/lemma; empty example
    and lemma fields become null, lemma kept as a whitespace-collapsed
    string."""
    os.makedirs(WORK_DIR, exist_ok=True)
    import re

    ws = re.compile(r"\s+")
    n_rows = 0
    n_malformed = 0
    with open(WORDNET_TSV, "r", encoding="utf-8") as src, open(
        WORDNET_JSONL, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            fields = [ws.sub(" ", f).strip() for f in line.split("\t")]
            if len(fields) != 5:
                n_malformed += 1
                continue
            row = {
                "id": fields[0],
                "pos": fields[1],
                "gloss": fields[2],
                "example": fields[3] if fields[3] else None,
                "lemma": fields[4] if fields[4] else None,
            }
            dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n_rows += 1
    if n_rows == 0:
        raise RuntimeError("synset decode produced 0 rows")
    prog.mark("wordnet_decoded", f"rows={n_rows} malformed={n_malformed}")
    return n_rows


def seed_index_cache(prog: Progress, run_id: str) -> str:
    """Copy the prior run's gram-index cache into the records dir when the
    corpus digest matches (the cache filename embeds digest[:16], and the
    harness recomputes the digest itself — a mismatch is a silent no-op)."""
    sys.path.insert(0, REPO_DIR)
    from clause16e.corpus import corpus_digest  # noqa: E402

    digest = corpus_digest(CORPUS_DIR, CORPUS_GLOBS)
    prog.mark("corpus_digest", f"sha256={digest}")
    records_dir = f"{REPO_DIR}/artifacts/clause16e/{run_id}"
    os.makedirs(records_dir, exist_ok=True)
    cache_name = f"gram-index-n25-{digest[:16]}.bin"
    seeded = ""
    if os.path.isdir(SEED_CACHE_DIR):
        src = os.path.join(SEED_CACHE_DIR, cache_name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(records_dir, cache_name))
            seeded = f" seeded={cache_name}"
    prog.mark("records_ready", f"dir={records_dir}{seeded}")
    return digest


def checkpoint_records(prog: Progress, stop: threading.Event, run_id: str) -> None:
    """Watchdog: mirror any new record files to the records mount every
    3 minutes so a dead job still leaves its partial records."""
    out_records = f"{GATE_DIR}/{run_id}/records"
    src_records = f"{REPO_DIR}/artifacts/clause16e/{run_id}"
    while not stop.wait(180):
        try:
            if os.path.isdir(src_records):
                os.makedirs(out_records, exist_ok=True)
                for name in os.listdir(src_records):
                    s = os.path.join(src_records, name)
                    d = os.path.join(out_records, name)
                    if os.path.isfile(s) and (
                        not os.path.exists(d)
                        or os.path.getsize(s) != os.path.getsize(d)
                    ):
                        shutil.copyfile(s, d)
        except Exception as e:  # never let the watchdog kill the gate
            prog.mark("watchdog_error", scrub(str(e)[:120]))


def reference_location() -> tuple[str | None, str | None, str | None]:
    """Resolve the reference per the precedence HUB > PATH > SUBPATH.

    Returns (hub_id, mount_path, subpath) — exactly one is non-None.
    """
    if REFERENCE_HUB:
        return REFERENCE_HUB, None, None
    if REFERENCE_PATH:
        return None, REFERENCE_PATH, None
    if REFERENCE_SUBPATH:
        return None, f"{CKPT_ROOT}/{REFERENCE_SUBPATH}", None
    return None, None, None


def resolve_publish_ids(config_path: str, run_id: str) -> list[str]:
    """Ids under which the verdict/records may have been written.

    The harness writes artifacts under the CONFIG's run_id; GATE_RUN_ID
    (the records subdir) may carry a job suffix (e.g. -v2 after a
    crashed attempt). Resolve the verdict/records under both ids —
    job-suffix run ids must not hide the artifacts.
    """
    cfg_run_id = ""
    try:
        with open(config_path, encoding="utf-8") as fh:
            cfg_run_id = str(json.load(fh).get("run_id", ""))
    except (OSError, ValueError):
        pass
    ids = [run_id] + ([cfg_run_id] if cfg_run_id and cfg_run_id != run_id else [])
    return ids


def publish(
    prog: Progress,
    run_id: str,
    head_sha: str,
    config_path: str,
    records_bucket: str,
) -> str | None:
    out_dir = f"{GATE_DIR}/{run_id}"
    os.makedirs(f"{out_dir}/records", exist_ok=True)
    ids = resolve_publish_ids(config_path, run_id)
    verdict_src = next(
        (p for p in (f"{REPO_DIR}/reports/clause16e/{i}.verdict.json"
                     for i in ids)
         if os.path.isfile(p)),
        None,
    )
    if verdict_src is None:
        return None
    shutil.copyfile(verdict_src, f"{out_dir}/verdict.json")
    for i in ids:
        src_records = f"{REPO_DIR}/artifacts/clause16e/{i}"
        if os.path.isdir(src_records):
            for name in os.listdir(src_records):
                shutil.copyfile(
                    os.path.join(src_records, name), f"{out_dir}/records/{name}"
                )
    meta = {
        "run_id": run_id,
        "branch": BRANCH,
        "repo": REPO,
        "head_sha": head_sha,
        "flavor": os.environ.get("GATE_FLAVOR", ""),
        "corpus_mount": CORPUS_DIR,
        "model_path": f"{CKPT_ROOT}/{MODEL_SUBPATH}" if MODEL_SUBPATH else "",
        "records_location": records_bucket,
        "job_started_utc": STARTED,
        "job_finished_utc": utc(),
    }
    with open(f"{out_dir}/job-meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return out_dir


def _run_harness(prog: Progress, command: str) -> subprocess.CompletedProcess:
    env = {
        "PYTHONPATH": REPO_DIR,
        "CLAUSE16E_CORPUS_DIR": CORPUS_DIR,
        "CLAUSE16E_WORDNET_PATH": WORDNET_JSONL,
        "CLAUSE16E_MODEL_PATH": f"{CKPT_ROOT}/{MODEL_SUBPATH}",
        "CLAUSE16E_DEVICE": os.environ.get("CLAUSE16E_DEVICE", "cuda"),
    }
    ref_hub, ref_path, _ = reference_location()
    if ref_hub:
        env["CLAUSE16E_REFERENCE_MODEL_PATH"] = ref_hub
    elif ref_path:
        env["CLAUSE16E_REFERENCE_MODEL_PATH"] = ref_path
    return run_cmd(
        [sys.executable, "-u", "-m", "clause16e.cli", command,
         "--config", CONFIG_PATH],
        cwd=REPO_DIR,
        extra_env=env,
    )


def main() -> int:
    global HEAD_SHA
    run_id = RUN_ID or time.strftime("clause16e-%Y%m%dT%H%M%S")
    prog = Progress(f"{GATE_DIR}/{run_id}/progress.log")
    prog.mark("start", f"branch={BRANCH} run_id={run_id}")
    watchdog_stop = threading.Event()

    try:
        HEAD_SHA = clone_repo(prog)
        watchdog = threading.Thread(
            target=checkpoint_records, args=(prog, watchdog_stop, run_id),
            daemon=True,
        )
        watchdog.start()

        decode_wordnet(prog)
        seed_index_cache(prog, run_id)

        # Fail fast with a clear marker if a checkpoint mount is incomplete.
        checks = [("candidate", f"{CKPT_ROOT}/{MODEL_SUBPATH}")]
        ref_hub, ref_path, _ = reference_location()
        if ref_path:
            checks.append(("reference", ref_path))
        for label, path in checks:
            if not os.path.isfile(os.path.join(path, "model.safetensors")):
                raise RuntimeError(f"{label} checkpoint missing model.safetensors: {path}")
        prog.mark("checkpoints_present",
                  "candidate+reference(hub)" if REFERENCE_HUB
                  else "candidate+reference")

        prog.mark("validate")
        val = _run_harness(prog, "validate")
        if val.returncode != 0:
            raise RuntimeError(f"config validation failed: {val.stderr[:400]}")

        if os.environ.get("GATE_SMOKE") == "1":
            # Mount/clone/decode/digest/config validation only — no model,
            # no probes.
            prog.mark("smoke_stop", "GATE_SMOKE=1: stopping before probe run")
            return 0
        prog.mark("probe_run", "harness started")

        res = _run_harness(prog, "run")
        # CLI stdout/stderr are aggregates only; safe to echo.
        if res.stdout.strip():
            print(res.stdout, flush=True)
        if res.returncode not in (0, 1):
            raise RuntimeError(f"harness exited {res.returncode}: {res.stderr[:400]}")

        records_bucket = f"{GATE_DIR}/{run_id}"
        out_dir = publish(prog, run_id, HEAD_SHA, CONFIG_PATH, records_bucket)
        watchdog_stop.set()
        if out_dir is None:
            raise RuntimeError("no verdict artifact was produced")
        with open(f"{out_dir}/verdict.json") as fh:
            verdict = json.load(fh)
        prog.mark(
            "done",
            f"verdict={verdict.get('verdict')} copies_to={out_dir}",
        )
        return 0
    except Exception as e:
        if prog is not None:
            prog.mark("error", scrub(f"{type(e).__name__}: {e}")[:300])
            traceback.print_exc()
        return 1
    finally:
        watchdog_stop.set()


if __name__ == "__main__":
    raise SystemExit(main())
