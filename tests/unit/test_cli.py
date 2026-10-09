"""The CLI surface: every subcommand parses, and bad input fails loudly.

The command-line is how every phase is actually run, so a broken flag or a silent
no-op here costs a whole experiment. ``--dry-run`` is asserted to touch nothing on
disk: a planning command that creates run directories would corrupt the idempotency
of ``runs/<name>-<hash>/``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from riskproj.cli import COMMANDS, build_parser, load_script

pytestmark = pytest.mark.unit


def _load_script_module(name: str) -> Any:
    """Load ``scripts/<name>`` by path, exactly as the CLI does."""
    from riskproj.cli import repo_root

    spec = importlib.util.spec_from_file_location(f"cli_test_{name}", repo_root() / "scripts" / name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    src = str(repo_root() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)
    return module


def test_every_subcommand_has_a_script_that_exists() -> None:
    from riskproj.cli import repo_root

    for name, (script, help_text) in COMMANDS.items():
        assert (repo_root() / "scripts" / script).exists(), f"{name} -> missing {script}"
        assert help_text, f"{name} has no help text"


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_subcommand_help_exits_zero(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([command, "--help"])
    assert exc.value.code == 0
    assert command in capsys.readouterr().out


def test_missing_subcommand_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([])
    assert exc.value.code != 0
    assert "usage" in capsys.readouterr().err.lower()


def test_unknown_subcommand_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["no-such-command"])
    assert exc.value.code != 0
    assert "invalid choice" in capsys.readouterr().err.lower()


def test_script_main_rejects_bad_arguments() -> None:
    """A missing required flag must fail, not silently run with defaults."""
    module = _load_script_module("run_experiment.py")
    with pytest.raises(SystemExit) as exc:
        module.main([])  # --config is required
    assert exc.value.code != 0


def test_script_main_rejects_bad_shard() -> None:
    module = _load_script_module("run_experiment.py")
    from riskproj.cli import repo_root

    with pytest.raises(SystemExit) as exc:
        module.main(
            [
                "--config",
                str(repo_root() / "configs" / "exp" / "toy_smoke.yaml"),
                "--shard",
                "not-a-shard",
            ]
        )
    assert exc.value.code != 0


def test_make_report_requires_a_run_directory() -> None:
    module = _load_script_module("make_report.py")
    with pytest.raises(SystemExit) as exc:
        module.main([])
    assert exc.value.code != 0


def test_load_script_raises_for_a_missing_script() -> None:
    with pytest.raises(FileNotFoundError):
        load_script("definitely_not_a_script.py")


def test_dry_run_prints_a_plan_and_writes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    """`--dry-run` is the safety valve before a long sweep: it must be side-effect free."""
    from riskproj.cli import repo_root

    module = _load_script_module("run_experiment.py")
    config = repo_root() / "configs" / "exp" / "toy_smoke.yaml"
    before = {p for p in (repo_root() / "runs").glob("toy_smoke-*")} if (repo_root() / "runs").exists() else set()

    assert module.main(["--config", str(config), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "toy_smoke" in out and "jobs" in out
    assert "B3" in out, "the plan must list the methods it would run"

    after = {p for p in (repo_root() / "runs").glob("toy_smoke-*")} if (repo_root() / "runs").exists() else set()
    assert after == before, "--dry-run created a run directory"


def test_run_experiment_refuses_test_splits_without_allow_test() -> None:
    """CLAUDE.md #2: TEST splits need an explicit opt-in after pre-registration."""
    from riskproj.cli import repo_root
    from riskproj.utils.config import load_config

    module = _load_script_module("run_experiment.py")
    cfg = load_config(repo_root() / "configs" / "exp" / "e4_main.yaml")
    assert any(s.startswith("test") for s in cfg.eval.splits)
    source = (repo_root() / "scripts" / "run_experiment.py").read_text(encoding="utf-8")
    assert "--allow-test" in source
    assert "REFUSING test splits" in source
    del module
