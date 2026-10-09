"""The smoke suite has to actually run, and S6 has to actually be timed.

These checks spawn the real script the way CI and ``tasks.py smoke`` do. They are
marked ``integration`` and ``slow`` because S6 exercises the whole pipeline, and S4
trains a policy for a few steps.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "smoke_suite.py"
REPORT = ROOT / "runs" / "_smoke" / "report.json"


def _run(*args: str, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _report(proc: subprocess.CompletedProcess[str]) -> dict:
    """Read the report *that run* wrote.

    Partial runs write ``report_<IDS>.json`` rather than the G0 artifact, so the path is
    taken from the run's own output instead of being assumed (writing a partial run to the
    gate artifact is how the full-suite record used to get clobbered).
    """
    for line in proc.stdout.splitlines():
        if line.strip().startswith("[smoke]") and "->" in line:
            path = Path(line.rsplit("->", 1)[1].strip())
            assert path.exists(), f"the suite announced {path} but wrote nothing there"
            return json.loads(path.read_text(encoding="utf-8"))
    raise AssertionError(
        f"the suite printed no report path; stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_s4_passes_on_cpu() -> None:
    """S4 is the check that keeps the repo honest without the simulator stack."""
    proc = _run("--only", "S4")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = _report(proc)
    assert report["suite"] == "S0-S6"
    entry = next(r for r in report["results"] if r["id"] == "S4")
    assert entry["status"] == "pass", entry["detail"]


def test_s6_walking_skeleton_runs_and_is_budgeted() -> None:
    proc = _run("--only", "S6")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = _report(proc)
    entry = next(r for r in report["results"] if r["id"] == "S6")
    assert entry["status"] == "pass", entry["detail"]
    assert entry["seconds"] <= 120.0, "03 §3 requires the walking skeleton under 2 minutes"
    assert report["summary"]["n_pass"] >= 1


def test_s6_exceeding_its_budget_fails_rather_than_skips() -> None:
    """A timed-out pipeline is a failure, not an excuse (03 §3)."""
    proc = _run("--only", "S6", "--max-seconds", "0.001")
    assert proc.returncode != 0
    entry = next(r for r in _report(proc)["results"] if r["id"] == "S6")
    assert entry["status"] == "fail"
    assert "budget" in entry["detail"].lower()


def test_report_shape_is_stable() -> None:
    proc = _run("--only", "S4")
    assert proc.returncode == 0
    report = _report(proc)
    for key in ("suite", "when", "results", "summary"):
        assert key in report
    for key in ("n_pass", "n_fail", "n_skip", "g0_complete", "skipped_ids"):
        assert key in report["summary"]
    for entry in report["results"]:
        for key in ("id", "name", "status", "detail", "seconds", "artifacts", "required"):
            assert key in entry


def test_unknown_check_id_is_rejected() -> None:
    proc = _run("--only", "S99")
    assert proc.returncode != 0
    assert "unknown check ids" in (proc.stderr + proc.stdout)
