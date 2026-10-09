#!/usr/bin/env python
"""Build a report from a run directory (or any directory of archived JSONL logs).

Reports are *derived*: they are recomputed from the immutable raw logs, so a new
metric can be added and applied to runs that already happened
(``skills/software_engineering.md`` §1).

    python scripts/make_report.py --run runs/toy_smoke-1a2b3c4d --format md
    python scripts/make_report.py --run runs/e4_main-1234abcd --format json --no-figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_report.py", description=__doc__)
    parser.add_argument(
        "--run",
        "--runs",
        dest="run",
        required=True,
        help="run directory (or a comma-separated list of them)",
    )
    parser.add_argument("--out", default=None, help="output path (default: <run>/report.<fmt>)")
    parser.add_argument("--format", "--out-format", dest="fmt", choices=("md", "json"), default="md")
    parser.add_argument("--figures", action="store_true", help="also write figure-ready tables")
    parser.add_argument("--no-figures", action="store_true", help="skip figure tables")
    ns = parser.parse_args(argv)

    from riskproj.eval.report import build_tables, load_run, write_report

    run_dirs = [Path(p).resolve() for p in ns.run.split(",") if p.strip()]
    for run in run_dirs:
        if not run.exists():
            raise SystemExit(f"run directory not found: {run}")

    payloads = []
    for run in run_dirs:
        loaded = load_run(run)
        tables = build_tables(loaded) if not isinstance(loaded, dict) else build_tables(loaded)
        out = Path(ns.out) if (ns.out and len(run_dirs) == 1) else run / f"report.{ns.fmt}"
        written = write_report(tables, out, fmt=ns.fmt, figures=not ns.no_figures)
        payloads.append({"run": str(run), "report": str(written), "n_tables": len(tables)})
        print(f"[report] {run.name}: {len(tables)} table(s) -> {written}")
    print(json.dumps(payloads, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
