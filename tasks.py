#!/usr/bin/env python
"""Cross-platform developer task runner for riskproj.

`make` is not available on every developer box (notably Windows without
MSYS/Chocolatey), so the canonical entry points live here and ``Makefile`` is a
thin shim over this file. Stdlib only: it must run before the venv exists.

    python tasks.py setup        # create .venv, install the CPU dev stack
    python tasks.py bootstrap-sim# submodules + MuJoCo/VLA stack (heavy)
    python tasks.py test         # unit + integration, CPU, no simulator
    python tasks.py smoke        # smoke suite S0-S6 (writes runs/_smoke/report.json)
    python tasks.py exp EXP=configs/exp/e0_baseline_repro.yaml
    python tasks.py report RUN=runs/<run-id>
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
IS_WIN = os.name == "nt"
PY_IN_VENV = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")

# --------------------------------------------------------------------------
# environment helpers
# --------------------------------------------------------------------------


def venv_python() -> str:
    """Interpreter to use for project commands (falling back to the current one)."""
    if PY_IN_VENV.exists():
        return str(PY_IN_VENV)
    if Path(sys.executable).resolve() != Path(str(PY_IN_VENV)).resolve():
        print("[tasks] warning: .venv not found; run `python tasks.py setup` first", file=sys.stderr)
    return sys.executable


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(ROOT / "src"))
    env.setdefault("PYTHONUTF8", "1")
    # MuJoCo headless rendering. EGL exists on Linux only; Windows uses WGL.
    if IS_WIN:
        env.setdefault("MUJOCO_GL", "wgl")
    else:
        env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def run(cmd: list[str], *, check: bool = True, env: dict[str, str] | None = None) -> int:
    print(f"[tasks] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env or child_env())
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def uv_or_pip() -> list[str]:
    if shutil.which("uv"):
        return ["uv", "pip"]
    return [venv_python(), "-m", "pip"]


def pip_install(args: list[str]) -> None:
    base = uv_or_pip()
    env = child_env()
    if base[0] == "uv":
        env["VIRTUAL_ENV"] = str(VENV)
    run([*base, "install", *args], env=env)


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------


def t_setup(_: argparse.Namespace) -> None:
    """Create the venv and install the CPU dev stack (fast, no simulator)."""
    if not PY_IN_VENV.exists():
        print(f"[tasks] creating venv at {VENV}")
        venv.EnvBuilder(with_pip=False, clear=False).create(str(VENV))
    pip_install(["-e", ".[dev,stats,figures]"])
    run([venv_python(), "-c", "import riskproj; print('riskproj import OK')"])
    print("[tasks] setup done. Next: `python tasks.py bootstrap-sim` (heavy) then `python tasks.py smoke`.")


def t_env_lock(_: argparse.Namespace) -> None:
    out = ROOT / "env"
    out.mkdir(exist_ok=True)
    freeze = subprocess.run(
        [uv_or_pip()[0], "freeze"] if uv_or_pip()[0] == "uv" else [*uv_or_pip(), "freeze"],
        cwd=str(ROOT),
        env=child_env(),
        capture_output=True,
        text=True,
    )
    (out / "lock.txt").write_text(freeze.stdout, encoding="utf-8")
    print(f"[tasks] wrote {out / 'lock.txt'} ({len(freeze.stdout.splitlines())} pins)")


def t_bootstrap_sim(_: argparse.Namespace) -> None:
    """Init pinned submodules and install the simulator + (optional) VLA stack."""
    if not shutil.which("git"):
        raise SystemExit("[tasks] git is required")
    run(["git", "submodule", "update", "--init", "--recursive"], check=False)
    pip_install(["-e", ".[dev,ml,sim,stats,figures]"])
    run([venv_python(), "scripts/apply_patches.py"], check=False)
    pip_install(["-e", ".[vla]"], check=False)
    print("[tasks] bootstrap-sim done (some steps are best-effort; run `python tasks.py smoke`).")


def t_test(ns: argparse.Namespace) -> None:
    args = [venv_python(), "-m", "pytest", "-m", "not sim and not gpu and not slow"]
    if ns.extra:
        args += ns.extra
    run(args)


def t_test_sim(ns: argparse.Namespace) -> None:
    run([venv_python(), "-m", "pytest", "-m", "sim", *ns.extra])


def t_test_gpu(ns: argparse.Namespace) -> None:
    run([venv_python(), "-m", "pytest", "-m", "gpu", *ns.extra])


def t_test_all(ns: argparse.Namespace) -> None:
    run([venv_python(), "-m", "pytest", *ns.extra])


def t_lint(ns: argparse.Namespace) -> None:
    run([venv_python(), "-m", "ruff", "check", "src", "tests", "scripts", "tasks.py", *ns.extra])
    run([venv_python(), "-m", "ruff", "format", "--check", "src", "tests", "scripts", "tasks.py"])


def t_typecheck(ns: argparse.Namespace) -> None:
    run([venv_python(), "-m", "mypy", *ns.extra])


def t_fmt(ns: argparse.Namespace) -> None:
    run([venv_python(), "-m", "ruff", "format", "src", "tests", "scripts", "tasks.py"])
    run([venv_python(), "-m", "ruff", "check", "--fix", "src", "tests", "scripts", "tasks.py"])


def t_smoke(ns: argparse.Namespace) -> None:
    run([venv_python(), "scripts/smoke_suite.py", *ns.extra])


def t_exp(ns: argparse.Namespace) -> None:
    if not ns.EXP:
        raise SystemExit("[tasks] EXP=<configs/exp/*.yaml> is required")
    run([venv_python(), "scripts/run_experiment.py", "--config", ns.EXP, *ns.extra])


def t_report(ns: argparse.Namespace) -> None:
    if not ns.RUN:
        raise SystemExit("[tasks] RUN=<runs/...> is required")
    run([venv_python(), "scripts/make_report.py", "--run", ns.RUN, *ns.extra])


def t_clean(_: argparse.Namespace) -> None:
    for pat in ("**/__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"):
        for path in ROOT.glob(pat):
            shutil.rmtree(path, ignore_errors=True)
    print("[tasks] cleaned caches (runs/ and data/ left untouched)")


TARGETS = {
    "setup": t_setup,
    "bootstrap-sim": t_bootstrap_sim,
    "env-lock": t_env_lock,
    "test": t_test,
    "test-sim": t_test_sim,
    "test-gpu": t_test_gpu,
    "test-all": t_test_all,
    "lint": t_lint,
    "typecheck": t_typecheck,
    "fmt": t_fmt,
    "smoke": t_smoke,
    "exp": t_exp,
    "report": t_report,
    "clean": t_clean,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tasks.py", description=__doc__)
    parser.add_argument("target", choices=sorted(TARGETS))
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="passed through to the target")
    parser.add_argument("--EXP", default=os.environ.get("EXP", ""), help="experiment config path")
    parser.add_argument("--RUN", default=os.environ.get("RUN", ""), help="run directory")
    ns = parser.parse_args(argv)
    TARGETS[ns.target](ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
