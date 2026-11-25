"""CLI: ``clause16e validate|run --config <file>``.

Exit codes: 0 = verdict PASS, 1 = verdict FAIL (or validation/config
error), 2 = usage error. Output is aggregates only; no corpus text is
ever printed.
"""

from __future__ import annotations

import argparse
import sys

from .config import RunConfig
from .runner import Harness


def _summary(verdict: dict) -> str:
    lines = [
        f"run_id: {verdict['run_id']}",
        f"verdict: {verdict['verdict']}",
    ]
    for name, res in verdict["tests"].items():
        metrics = res.get("metrics", {})
        key_facts = ", ".join(f"{k}={v}" for k, v in sorted(metrics.items()))
        line = f"  {name}: {res['status']}"
        if key_facts:
            line += f" ({key_facts})"
        if res.get("error"):
            line += f" error={res['error']}"
        lines.append(line)
    if verdict["failures"]:
        lines.append("failures: " + "; ".join(verdict["failures"]))
    lines.append(f"records dir: {verdict['records']['dir']}")
    lines.append(f"retention until: {verdict['retention_until']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clause16e",
        description=(
            "License-cleanliness gate: extraction & membership-inference "
            "testing for models trained on licensed or restricted corpora"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (
        ("validate", "load and validate a run config without running probes"),
        ("run", "run all probes and emit a verdict artifact"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", required=True, help="path to run config JSON")

    args = parser.parse_args(argv)
    try:
        cfg = RunConfig.load(args.config)
    except Exception as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1
    if args.command == "validate":
        print("config OK")
        return 0

    harness = Harness(cfg)
    verdict = harness.run()
    # Re-write at the caller-visible path handled by Harness; print summary.
    print(_summary(verdict))
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
