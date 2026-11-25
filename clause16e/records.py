"""Private records writer (audit retention).

Flagged outputs and per-item scores are DERIVED FROM corpus/model
content and are never committed. They are written to the run's records
directory (gitignored ``artifacts/``), with a sha256 manifest that the
committed verdict artifact references, satisfying auditability without
leaking corpus content into the repo. House retention practice: keep
records for 2 years.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecordsWriter:
    directory: str
    manifest: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        os.makedirs(self.directory, exist_ok=True)

    def write_jsonl(self, name: str, rows: list[dict]) -> dict:
        path = os.path.join(self.directory, name)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        entry = {"path": path, "rows": len(rows), "sha256": _sha256_file(path)}
        self.manifest[name] = entry
        return entry

    def write_json(self, name: str, payload: Any) -> dict:
        path = os.path.join(self.directory, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        entry = {"path": path, "sha256": _sha256_file(path)}
        self.manifest[name] = entry
        return entry

    def manifest_dict(self) -> dict:
        return {"dir": self.directory, "files": dict(self.manifest)}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
