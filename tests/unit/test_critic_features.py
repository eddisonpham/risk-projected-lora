"""Feature-layout tests: the width and column order must never drift silently."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from riskproj import obs
from riskproj.critic.features import (
    FEATURE_LAYOUT_VERSION,
    CriticFeatureBuilder,
    FeatureSpec,
    flatten_context_torch,
)
from riskproj.utils.config import CriticCfg

pytestmark = pytest.mark.unit

STATE_DIM = 8
ACTION_DIM = 4
HORIZON = 10


def make_ctx(
    batch: int,
    *,
    state_dim: int = STATE_DIM,
    action_dim: int = ACTION_DIM,
    hist_len: int = 1,
    vision_dim: int = 0,
    force_dim: int = 0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = max(hist_len, 1)
    return {
        "proprio": rng.normal(size=(batch, state_dim)).astype(np.float32),
        "ee_vel": rng.normal(size=(batch, 3)).astype(np.float32),
        "action_hist": rng.normal(size=(batch, n, action_dim)).astype(np.float32),
        "proprio_hist": rng.normal(size=(batch, n, state_dim)).astype(np.float32),
        "vision_emb": rng.normal(size=(batch, vision_dim)).astype(np.float32),
        "force": rng.normal(size=(batch, force_dim)).astype(np.float32),
    }


def make_builder(cfg: CriticCfg, spec: FeatureSpec) -> CriticFeatureBuilder:
    return CriticFeatureBuilder.from_spec(cfg, spec)


def spec_for(cfg: CriticCfg, **overrides: int) -> FeatureSpec:
    kwargs = {
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "chunk_horizon": HORIZON,
        "hist_len": cfg.use_history,
        "vision_dim": spec_vision(cfg),
        "force_dim": spec_force(cfg),
        "use_vision": cfg.use_vision,
        "privileged_force": cfg.privileged_force,
    }
    kwargs.update(overrides)
    return FeatureSpec(**kwargs)


def spec_vision(cfg: CriticCfg, dim: int = 16) -> int:
    return dim if cfg.use_vision else 0


def spec_force(cfg: CriticCfg, dim: int = 6) -> int:
    return dim if cfg.privileged_force else 0


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


def test_dim_matches_produced_width() -> None:
    """The width the builder claims is the width it produces (the anti-drift test)."""
    cfg = CriticCfg()
    spec = spec_for(cfg)
    builder = make_builder(cfg, spec)
    ctx = make_ctx(5)
    chunk = torch.randn(5, HORIZON, ACTION_DIM)
    out = builder(ctx, chunk)
    assert out.shape == (5, builder.dim())
    assert builder.dim() == spec.dim


def test_dim_formula() -> None:
    spec = spec_for(CriticCfg())
    n = max(spec.hist_len, 1)
    assert spec.context_dim == STATE_DIM + 3 + n * ACTION_DIM + n * STATE_DIM
    assert spec.chunk_dim == HORIZON * ACTION_DIM
    assert spec.dim == spec.context_dim + spec.chunk_dim


def single_ctx(
    *, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, hist_len: int = 1, seed: int = 3
) -> dict[str, np.ndarray]:
    """A one-example, unbatched context (1-D per-step blocks, 2-D history)."""
    n = max(hist_len, 1)
    rng = np.random.default_rng(seed)
    return {
        "proprio": rng.normal(size=(state_dim,)).astype(np.float32),
        "ee_vel": rng.normal(size=(3,)).astype(np.float32),
        "action_hist": rng.normal(size=(n, action_dim)).astype(np.float32),
        "proprio_hist": rng.normal(size=(n, state_dim)).astype(np.float32),
        "vision_emb": np.zeros(0, dtype=np.float32),
        "force": np.zeros(0, dtype=np.float32),
    }


def test_column_order_matches_obs_context_vector() -> None:
    """Row 0 of the torch flattener must equal obs.context_vector (single source of order)."""
    cfg = CriticCfg()
    spec = spec_for(cfg)
    flat = single_ctx(seed=3)
    expected = obs.context_vector(flat)
    got = flatten_context_torch(obs.stack_contexts([flat]), spec)[0].numpy()
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_history_rows_survive_flattened_contexts() -> None:
    """stack_contexts flattens history; the builder must unflatten it, not misread it."""
    cfg = CriticCfg(use_history=3)
    spec = spec_for(cfg)
    builder = make_builder(cfg, spec)
    flat = single_ctx(hist_len=3, seed=7)
    stacked = obs.stack_contexts([flat])
    assert stacked["action_hist"].shape == (1, 3 * ACTION_DIM)  # flattened by obs
    chunk = torch.randn(1, HORIZON, ACTION_DIM)
    out = builder(stacked, chunk)
    assert out.shape == (1, builder.dim())
    expected = obs.context_vector(flat)
    np.testing.assert_allclose(out[0, : spec.context_dim].numpy(), expected, atol=1e-6)


def test_feature_spec_round_trip() -> None:
    spec = spec_for(CriticCfg())
    assert FeatureSpec.from_dict(spec.to_dict()) == spec


def test_critic_cfg_path_builds_equivalent_spec() -> None:
    cfg = CriticCfg()
    builder = CriticFeatureBuilder(
        cfg, state_dim=STATE_DIM, action_dim=ACTION_DIM, chunk_horizon=HORIZON
    )
    assert builder.dim() == spec_for(cfg).dim


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------


def test_vision_block_rejected_when_disabled() -> None:
    cfg = CriticCfg(use_vision=False)
    spec = spec_for(cfg)
    ctx = make_ctx(2, vision_dim=16)  # caller smuggles a visual embedding in
    with pytest.raises(ValueError, match="use_vision"):
        flatten_context_torch(ctx, spec)


def test_vision_block_required_when_enabled() -> None:
    cfg = CriticCfg(use_vision=True)
    spec = spec_for(cfg)
    ctx = make_ctx(2, vision_dim=0)
    with pytest.raises(ValueError, match="vision_emb"):
        flatten_context_torch(ctx, spec)


def test_force_block_rejected_when_privileged_disabled() -> None:
    cfg = CriticCfg(privileged_force=False)
    spec = spec_for(cfg)
    ctx = make_ctx(2, force_dim=6)
    with pytest.raises(ValueError, match="privileged_force"):
        flatten_context_torch(ctx, spec)


def test_force_block_accepted_when_enabled() -> None:
    cfg = CriticCfg(privileged_force=True)
    spec = spec_for(cfg)
    builder = make_builder(cfg, spec)
    ctx = make_ctx(3, force_dim=6)
    out = builder(ctx, torch.randn(3, HORIZON, ACTION_DIM))
    assert out.shape == (3, builder.dim())
    assert builder.dim() > spec_for(CriticCfg()).dim


def test_vision_dim_without_flag_is_rejected() -> None:
    with pytest.raises(ValueError, match="vision_dim"):
        FeatureSpec(state_dim=4, action_dim=2, chunk_horizon=3, vision_dim=8, use_vision=False)


def test_hist_len_zero_still_produces_one_row() -> None:
    spec = FeatureSpec(state_dim=4, action_dim=2, chunk_horizon=3, hist_len=0)
    assert spec.history_steps == 1
    ctx = {
        "proprio": np.zeros((1, 4), np.float32),
        "ee_vel": np.zeros((1, 3), np.float32),
        "action_hist": np.zeros((1, 2), np.float32),
        "proprio_hist": np.zeros((1, 4), np.float32),
    }
    assert flatten_context_torch(ctx, spec).shape == (1, spec.context_dim)


def test_chunk_horizon_mismatch_raises() -> None:
    cfg = CriticCfg()
    builder = make_builder(cfg, spec_for(cfg))
    ctx = make_ctx(2)
    with pytest.raises(ValueError, match="horizon"):
        builder(ctx, torch.randn(2, HORIZON - 1, ACTION_DIM))


def test_chunk_action_dim_mismatch_raises() -> None:
    cfg = CriticCfg()
    builder = make_builder(cfg, spec_for(cfg))
    with pytest.raises(ValueError, match="action dim"):
        builder(make_ctx(2), torch.randn(2, HORIZON, ACTION_DIM + 1))


def test_batch_mismatch_raises() -> None:
    cfg = CriticCfg()
    builder = make_builder(cfg, spec_for(cfg))
    with pytest.raises(ValueError, match="batch"):
        builder(make_ctx(3), torch.randn(2, HORIZON, ACTION_DIM))


def test_missing_context_block_raises() -> None:
    cfg = CriticCfg()
    spec = spec_for(cfg)
    ctx = make_ctx(2)
    del ctx["proprio"]
    with pytest.raises(KeyError, match="proprio"):
        flatten_context_torch(ctx, spec)


def test_single_example_is_promoted() -> None:
    cfg = CriticCfg()
    spec = spec_for(cfg)
    ctx = {
        "proprio": np.zeros(8, np.float32),
        "ee_vel": np.zeros(3, np.float32),
        "action_hist": np.zeros(4, np.float32),
        "proprio_hist": np.zeros(8, np.float32),
    }
    assert flatten_context_torch(ctx, spec).shape == (1, spec.context_dim)


def test_feature_layout_version_is_recorded() -> None:
    """A layout change must be visible in the constant checkpoints compare against."""
    assert FEATURE_LAYOUT_VERSION >= 1
