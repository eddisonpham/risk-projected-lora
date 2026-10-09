"""Seeding and common-random-number helpers.

Comparisons must be *paired*: same initial states and the same flow noise per
(episode, step) across methods (``08`` §4). ``common_random_noise`` is the single
place that defines how flow noise is derived, so a method cannot silently use a
different draw and manufacture a difference.
"""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from typing import Any

import numpy as np


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed python/numpy/torch/CUDA. Returns the seed for provenance logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))  # noqa: NPY002 -- global seeding is the point here
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # cuDNN autotuning is nondeterministic; kill it for comparable runs.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(False)  # some MuJoCo/sim paths are not
    except ImportError:  # pragma: no cover - torch is a hard dep in practice
        pass
    return seed


def spawn_seeds(base_seed: int, n: int) -> list[int]:
    """Derive ``n`` independent child seeds deterministically from ``base_seed``."""
    rng = np.random.default_rng(base_seed)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n)]


def rng_for(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def common_random_noise(
    base_seed: int,
    episode_id: int | str,
    step: int,
    shape: Sequence[int],
    *,
    flow_step: int = 0,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Deterministic flow-matching noise ``eps`` shared by every method.

    Keyed by ``(base_seed, episode_id, step, flow_step)`` — never by method — so
    B1/B3/B4b see the *same* ``eps`` at the same decision point.
    """
    import torch

    key = f"{base_seed}|{episode_id}|{step}|{flow_step}"
    h = abs(hash(key)) % (2**31)  # PYTHONHASHSEED is pinned by seed_everything
    gen = torch.Generator(device="cpu")
    gen.manual_seed(h)
    out = torch.randn(*shape, generator=gen, dtype=torch.float32)
    if dtype is not None:
        out = out.to(dtype)
    if device is not None:
        out = out.to(device)
    return out


__all__ = ["common_random_noise", "rng_for", "seed_everything", "spawn_seeds"]
