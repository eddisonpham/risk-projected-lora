"""``riskproj`` command line: one dispatcher over the scripts in ``scripts/``.

The scripts stay directly executable (``python scripts/smoke_suite.py``), which is
what ``tasks.py`` and CI call, and this module makes the same entry points available
as ``riskproj <subcommand>`` after ``pip install -e .``. Each script is loaded by
path with :mod:`importlib` rather than imported as a module, so nothing depends on
``scripts/`` being a package or on the current working directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

#: subcommand -> (script file, one-line help)
COMMANDS: dict[str, tuple[str, str]] = {
    "smoke": ("smoke_suite.py", "run smoke checks S0-S6 (gate G0)"),
    "exp": ("run_experiment.py", "run an experiment config"),
    "report": ("make_report.py", "build a report from archived run logs"),
    "inspect-model": ("inspect_model.py", "print the module tree and write the sites yaml (S3)"),
    "inspect-contacts": ("inspect_contacts.py", "discover task-critical geoms (dynamics inputs)"),
    "profile-vram": ("profile_vram.py", "measure peak VRAM per phase (gate G2)"),
    "sensitivity": ("sensitivity_sweep.py", "run the dynamics-sensitivity sweep (gate G1)"),
    "derive-thresholds": (
        "derive_violation_thresholds.py",
        "derive violation thresholds from nominal expert rollouts",
    ),
    "apply-patches": ("apply_patches.py", "apply patches/*.patch to third_party"),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_script(name: str) -> Callable[..., Any]:
    """Load ``scripts/<name>`` by path and return its ``main``."""
    path = repo_root() / "scripts" / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; the install looks incomplete")
    module_name = f"riskproj_scripts_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreadable file
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    src = str(repo_root() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="riskproj",
        description="Risk-guided activation projection: smoke suite, experiments, audits.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (script, help_text) in COMMANDS.items():
        child = sub.add_parser(name, help=help_text, description=help_text)
        # Arguments are parsed by the target script itself, which keeps one source of
        # truth for each command's surface (`--help` goes through too).
        child.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    script_name = COMMANDS[ns.command][0]
    forward = list(ns.args)
    if forward[:1] == ["--"]:
        forward = forward[1:]
    try:
        func = load_script(script_name)
    except (FileNotFoundError, ImportError) as exc:
        print(f"[riskproj] {exc}", file=sys.stderr)
        return 2
    return int(func(forward) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
