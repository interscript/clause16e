#!/usr/bin/env bash
# Detached HF Jobs launcher for a clause16e remote gate run.
#
# Established pattern: --timeout >= 4h (probe runs take hours), -d
# detach, submission wrapped in a retry loop, HF-side --attempts 2,
# monitor via `hf jobs ps --all` — never attach.
#
# Mounts are environment-specified (this script ships NO defaults —
# the protected corpus, checkpoints, and records bucket are
# program-specific):
#   CORPUS_MOUNT_SPEC    read-only protected-corpus channel
#                        (e.g. hf://datasets/<org>/<corpus>:/data:ro)
#   CKPT_MOUNT_SPEC      read-only checkpoint bucket
#                        (e.g. hf://buckets/<org>/<training>:/ckpt:ro)
#   RECORDS_MOUNT_SPEC   read-write private records bucket
#                        (e.g. hf://buckets/<org>/<records>:/gate:rw)
# Secrets: GH_TOKEN (gate repo clone) — read from the environment.
#
# Candidate/reference selection (see hfjob/gate_job.py docstring):
#   GATE_MODEL_SUBPATH      candidate path under the checkpoint mount (required)
#   GATE_REFERENCE_HUB      reference as a hub id (wins) — OR
#   GATE_REFERENCE_PATH     reference as a mount-absolute path — OR
#   GATE_REFERENCE_SUBPATH  reference under the checkpoint mount
#   Precedence: HUB > PATH > SUBPATH. The reference must be
#   DOMAIN-MATCHED (the candidate's own pre-adaptation init); see
#   docs/case-study.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${GH_TOKEN:-}" ]; then GH_TOKEN="$(gh auth token)"; fi
export GH_TOKEN
FLAVOR=${GATE_FLAVOR:-l4x1}
TIMEOUT=${GATE_TIMEOUT:-8h}
NAME=${GATE_NAME:-clause16e-gate}
CONFIG=${GATE_CONFIG:-example-reference-calibrated.json}
RUN_ID=${GATE_RUN_ID:-}
BRANCH=${GATE_BRANCH:-main}
MODEL_SUBPATH=${GATE_MODEL_SUBPATH:-}
REFERENCE_HUB=${GATE_REFERENCE_HUB:-}
REFERENCE_PATH=${GATE_REFERENCE_PATH:-}
REFERENCE_SUBPATH=${GATE_REFERENCE_SUBPATH:-}
CORPUS_MOUNT=${CORPUS_MOUNT_SPEC:?CORPUS_MOUNT_SPEC is required (read-only protected-corpus mount)}
CKPT_MOUNT=${CKPT_MOUNT_SPEC:?CKPT_MOUNT_SPEC is required (read-only checkpoint mount)}
RECORDS_MOUNT=${RECORDS_MOUNT_SPEC:?RECORDS_MOUNT_SPEC is required (read-write records mount)}

for attempt in 1 2 3; do
  if hf jobs uv run --flavor "$FLAVOR" -d --timeout "$TIMEOUT" --attempts 2 \
      --secrets GH_TOKEN \
      --name "$NAME" \
      -e "GATE_CONFIG=$CONFIG" \
      ${RUN_ID:+-e "GATE_RUN_ID=$RUN_ID"} \
      -e "GATE_BRANCH=$BRANCH" \
      ${MODEL_SUBPATH:+-e "GATE_MODEL_SUBPATH=$MODEL_SUBPATH"} \
      ${REFERENCE_HUB:+-e "GATE_REFERENCE_HUB=$REFERENCE_HUB"} \
      ${REFERENCE_PATH:+-e "GATE_REFERENCE_PATH=$REFERENCE_PATH"} \
      ${REFERENCE_SUBPATH:+-e "GATE_REFERENCE_SUBPATH=$REFERENCE_SUBPATH"} \
      -v "$CORPUS_MOUNT" \
      -v "$CKPT_MOUNT" \
      -v "$RECORDS_MOUNT" \
      "$SCRIPT_DIR/gate_job.py"; then
    echo "launched (attempt $attempt, detached)"
    exit 0
  fi
  echo "submission failed (attempt $attempt), retrying in 20s" >&2
  sleep 20
done
echo "submission failed after 3 attempts" >&2
exit 1
