"""Shared pytest fixtures.

Tests must run on a CPU-only box with no simulator assets: anything requiring
MuJoCo, GPU or downloads is marked ``sim`` / ``gpu`` / ``slow``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Deterministic, CPU-only by default so CI is stable.
os.environ.setdefault("PYTHONHASHSEED", "0")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def configs_dir(repo_root: Path) -> Path:
    return repo_root / "configs"


@pytest.fixture(scope="session")
def exps_dir(repo_root: Path) -> Path:
    return repo_root / "configs" / "exp"


@pytest.fixture()
def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture()
def torch_mod() -> Any:
    import torch

    return torch


@pytest.fixture(autouse=True)
def _seeded() -> None:
    from riskproj.utils.seed import seed_everything

    seed_everything(0)
