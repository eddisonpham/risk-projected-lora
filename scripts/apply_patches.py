#!/usr/bin/env python
"""Apply ``patches/*.patch`` in order (``agents/03_repo_setup.md`` §5).

Upstream code is never edited in place: if LeRobot lacks a hook, the fix lives in
``patches/NNN-description.patch`` with a written rationale, and this script re-applies
it after a submodule update. ``--check`` only verifies that the patches apply, which
is what ``tasks.py bootstrap-sim`` runs.

A failed patch is loud, and the remaining patches are still attempted so one broken
hunk does not hide the state of the others.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "patches"


def patch_files() -> list[Path]:
    if not PATCH_DIR.exists():
        return []
    return sorted(p for p in PATCH_DIR.glob("*.patch") if p.is_file())


def apply_one(path: Path, *, check: bool, cwd: Path, strip: int) -> tuple[bool, str]:
    cmd = ["git", "apply", f"-p{strip}"]
    if check:
        cmd.append("--check")
    cmd.append(str(path))
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
    detail = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, detail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="apply_patches.py", description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify only, apply nothing")
    parser.add_argument("--cwd", default=str(ROOT), help="repository to apply against")
    parser.add_argument("--strip", type=int, default=1, help="git apply -p level")
    parser.add_argument("--only", default="", help="comma-separated patch file names")
    ns = parser.parse_args(argv)

    only = {v.strip() for v in ns.only.split(",") if v.strip()}
    patches = [p for p in patch_files() if not only or p.name in only]
    if not patches:
        print("[patches] no patches found (nothing to do)")
        return 0

    cwd = Path(ns.cwd)
    failures: list[tuple[Path, str]] = []
    for path in patches:
        ok, detail = apply_one(path, check=ns.check, cwd=cwd, strip=ns.strip)
        status = "ok" if ok else "FAILED"
        print(f"[patches] {path.name}: {status}")
        if not ok:
            failures.append((path, detail))

    if failures:
        print(
            f"\n[patches] {len(failures)} of {len(patches)} patch(es) failed. A patch that no "
            "longer applies means the pinned submodule moved: update patches/README.md with the "
            "new upstream SHA and the reason, or drop the patch if upstream now provides the hook.",
            file=sys.stderr,
        )
        for path, detail in failures:
            print(f"\n--- {path.name} ---\n{detail}", file=sys.stderr)
        return 1

    verb = "would apply" if ns.check else "applied"
    print(f"[patches] {len(patches)} patch(es) {verb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
