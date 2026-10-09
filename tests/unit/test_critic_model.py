"""Model tests: aggregation, the risk-fn factory, checkpointing, training, diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from riskproj.critic.features import FeatureSpec
from riskproj.critic.model import RiskCritic, aggregate_members, make_risk_fn
from riskproj.critic.train import (
    assert_frozen,
    freeze,
    gradient_diagnostics,
    labels_from_records,
    train_critic,
)
from riskproj.types import VIOLATION_TYPES
from riskproj.utils.config import CriticCfg

pytestmark = pytest.mark.unit

STATE_DIM = 6
ACTION_DIM = 4
HORIZON = 10
DIM = STATE_DIM + 3 + ACTION_DIM + STATE_DIM + HORIZON * ACTION_DIM  # hist_len=1


def spec(state_dim: int = STATE_DIM, hist_len: int = 1) -> FeatureSpec:
    return FeatureSpec(
        state_dim=state_dim,
        action_dim=ACTION_DIM,
        chunk_horizon=HORIZON,
        hist_len=hist_len,
    )


def cfg(**kwargs: object) -> CriticCfg:
    base = {"hidden": 32, "depth": 2, "ensemble": 3, "seed": 0}
    base.update(kwargs)
    return CriticCfg(**base)  # type: ignore[arg-type]


def random_ctx(batch: int = 4, state_dim: int = STATE_DIM, hist_len: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = max(hist_len, 1)
    return {
        "proprio": rng.normal(size=(batch, state_dim)).astype(np.float32),
        "ee_vel": rng.normal(size=(batch, 3)).astype(np.float32),
        "action_hist": rng.normal(size=(batch, n, ACTION_DIM)).astype(np.float32),
        "proprio_hist": rng.normal(size=(batch, n, state_dim)).astype(np.float32),
        "vision_emb": np.zeros((batch, 0), np.float32),
        "force": np.zeros((batch, 0), np.float32),
    }


def make_model(**kwargs: object) -> RiskCritic:
    return RiskCritic(cfg(**kwargs), spec()).eval()


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def test_logits_shape_and_ensemble_members_differ() -> None:
    model = make_model()
    chunk = torch.randn(4, HORIZON, ACTION_DIM)
    out = model.logits(random_ctx(), chunk)
    assert out.shape == (4, 6)
    per_member = model.per_member_logits(random_ctx(), chunk)
    assert per_member.shape == (3, 4, 6)
    assert not torch.allclose(per_member[0], per_member[1])


def test_risk_prob_mean_equals_sigmoid_of_mean_logits() -> None:
    model = make_model()
    ctx = random_ctx(batch=5)
    chunk = torch.randn(5, HORIZON, ACTION_DIM)
    r = model.risk(ctx, chunk, space="prob", aggregator="mean")
    expected = torch.sigmoid(model.logits(ctx, chunk)[:, 0])
    assert r.shape == (5,)
    torch.testing.assert_close(r, expected)


def test_risk_logit_space_is_mean_logit_not_probability() -> None:
    model = make_model()
    ctx = random_ctx(batch=5)
    chunk = torch.randn(5, HORIZON, ACTION_DIM)
    r_logit = model.risk(ctx, chunk, space="logit", aggregator="mean")
    torch.testing.assert_close(r_logit, model.logits(ctx, chunk)[:, 0])


def test_aggregators_are_ordered() -> None:
    model = make_model()
    ctx = random_ctx(batch=6)
    chunk = torch.randn(6, HORIZON, ACTION_DIM)
    mean = model.risk(ctx, chunk, aggregator="mean")
    robust = model.risk(ctx, chunk, aggregator="mean_plus_k_std", k=1.0)
    pessimistic = model.risk(ctx, chunk, aggregator="max")
    assert torch.all(robust >= mean - 1e-6)
    # max >= mean always holds; max >= mean + k*std does not (e.g. members [1, 1, 0]),
    # so the pessimistic/robust ordering is deliberately not asserted.
    assert torch.all(pessimistic >= mean - 1e-6)


def test_unknown_aggregator_and_space_raise() -> None:
    model = make_model()
    ctx = random_ctx(batch=2)
    chunk = torch.randn(2, HORIZON, ACTION_DIM)
    with pytest.raises(ValueError, match="aggregator"):
        model.risk(ctx, chunk, aggregator="median")
    with pytest.raises(ValueError, match="space"):
        model.risk(ctx, chunk, space="logodds")


def test_ensemble_std_is_positive_and_zero_for_single_member() -> None:
    chunk = torch.randn(4, HORIZON, ACTION_DIM)
    multi = make_model().ensemble_std(random_ctx(), chunk)
    assert torch.all(multi > 0)
    single = make_model(ensemble=1).ensemble_std(random_ctx(), chunk)
    torch.testing.assert_close(single, torch.zeros_like(single))


def test_aggregate_members_validation() -> None:
    per_member = torch.rand(3, 5)
    assert aggregate_members(per_member, aggregator="mean").shape == (5,)
    with pytest.raises(ValueError, match="M, B"):
        aggregate_members(torch.rand(5), aggregator="mean")


def test_feature_width_mismatch_raises() -> None:
    model = make_model()
    with pytest.raises(ValueError, match="features must be"):
        model.logits_from_features(torch.randn(2, DIM + 1))


# --------------------------------------------------------------------------
# risk functional used by the projection operator
# --------------------------------------------------------------------------


def test_make_risk_fn_gives_finite_nonzero_gradient() -> None:
    model = make_model()
    ctx = random_ctx(batch=3)
    head = torch.nn.Linear(STATE_DIM, HORIZON * ACTION_DIM, bias=False)

    def f_act(h: torch.Tensor) -> torch.Tensor:
        return head(h).reshape(h.shape[0], HORIZON, ACTION_DIM)

    risk_fn = make_risk_fn(model, ctx, f_act)
    h = torch.randn(3, STATE_DIM, requires_grad=True)
    r = risk_fn(h)
    assert r.shape == (3,)
    (grad,) = torch.autograd.grad(r.sum(), h)
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_make_risk_fn_refuses_training_mode_critic() -> None:
    model = make_model().train()
    with pytest.raises(RuntimeError, match="eval"):
        make_risk_fn(model, random_ctx(), lambda h: torch.randn(h.shape[0], HORIZON, ACTION_DIM))


def test_make_risk_fn_rejects_wrong_horizon() -> None:
    model = make_model()
    risk_fn = make_risk_fn(
        model, random_ctx(), lambda h: torch.randn(h.shape[0], HORIZON - 1, ACTION_DIM)
    )
    with pytest.raises(ValueError, match="horizon"):
        risk_fn(torch.randn(2, STATE_DIM))


def test_make_risk_fn_rejects_non_tensor_output() -> None:
    model = make_model()
    risk_fn = make_risk_fn(
        model, random_ctx(), lambda h: np.zeros((h.shape[0], HORIZON, ACTION_DIM))
    )
    with pytest.raises(ValueError, match="tensor"):
        risk_fn(torch.randn(2, STATE_DIM))


# --------------------------------------------------------------------------
# checkpointing and freezing
# --------------------------------------------------------------------------


def test_checkpoint_round_trip_preserves_outputs(tmp_path) -> None:
    model = make_model()
    path = model.save(tmp_path / "critic.pt")
    loaded = RiskCritic.from_checkpoint(path)
    ctx = random_ctx(batch=3)
    chunk = torch.randn(3, HORIZON, ACTION_DIM)
    torch.testing.assert_close(model.logits(ctx, chunk), loaded.logits(ctx, chunk))
    assert loaded.feature_spec == model.feature_spec


def test_checkpoint_layout_version_mismatch_raises(tmp_path) -> None:
    path = make_model().save(tmp_path / "critic.pt")
    payload = torch.load(path, weights_only=False)
    payload["feature_layout_version"] = 999
    torch.save(payload, path)
    with pytest.raises(ValueError, match="layout version"):
        RiskCritic.from_checkpoint(path)


def test_checkpoint_shape_mismatch_raises(tmp_path) -> None:
    path = make_model().save(tmp_path / "critic.pt")
    other = RiskCritic(cfg(), spec(state_dim=STATE_DIM + 1))
    with pytest.raises(RuntimeError):
        other.load_state_dict(torch.load(path, weights_only=False)["state_dict"])


def test_freeze_and_assert_frozen() -> None:
    model = make_model()
    freeze(model)
    assert_frozen(model)
    next(model.parameters()).requires_grad_(True)
    with pytest.raises(AssertionError, match="trainable"):
        assert_frozen(model)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


class SyntheticCriticData:
    """A separable, deterministic D-crit stand-in (no simulator, no VLM)."""

    def __init__(self, n: int = 320, dim: int = DIM, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        feats = rng.normal(size=(n, dim)).astype(np.float32)
        score = 2.5 * feats[:, 0] - 1.0 * np.abs(feats[:, 1])
        self.feats = feats
        self.y_any = (score > 0).astype(np.float32)
        self.y_type = np.stack(
            [(score > t).astype(np.float32) for t in (0.5, 1.0, 1.5, 2.0, 2.5)], axis=1
        )
        if self.y_any.sum() == 0:
            raise AssertionError("synthetic data has no positives; fix the generator")

    def __len__(self) -> int:
        return int(self.feats.shape[0])

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.feats[i]),
            torch.tensor(self.y_any[i]),
            torch.from_numpy(self.y_type[i]),
        )


def test_train_critic_learns_a_separable_signal(tmp_path) -> None:
    data = SyntheticCriticData()
    model, metrics = train_critic(
        data,
        cfg(epochs=40, batch_size=64, bootstrap=False, lr=5e-3),
        feature_spec=spec(),
        device="cpu",
        out_dir=tmp_path,
    )
    assert metrics["per_head_auroc"]["any"] >= 0.9
    assert metrics["n_train"] > 0
    assert metrics["base_rate_any"] > 0.15
    assert (tmp_path / "critic.pt").exists()
    assert (tmp_path / "metrics.json").exists()
    assert_frozen(model)


def test_train_critic_rejects_feature_width_mismatch() -> None:
    data = SyntheticCriticData()
    with pytest.raises(ValueError, match="width"):
        train_critic(data, cfg(epochs=1), feature_spec=spec(state_dim=STATE_DIM + 2), device="cpu")


def test_train_critic_rejects_empty_dataset() -> None:
    class Empty(SyntheticCriticData):
        def __len__(self) -> int:
            return 0

    with pytest.raises(ValueError, match="empty"):
        train_critic(Empty(), cfg(epochs=1), feature_spec=spec(), device="cpu")


def test_labels_from_records_keeps_per_type_and_fails_loudly() -> None:
    records = [
        {"any_violation": True, "violations": {v: {"count": 1.0} for v in VIOLATION_TYPES}},
        {
            "any_violation": False,
            "violations": {v: {"count": 0.0} for v in VIOLATION_TYPES},
        },
    ]
    any_t, type_t = labels_from_records(records)
    assert any_t.tolist() == [1.0, 0.0]
    assert type_t.shape == (2, 5)
    assert type_t[0].sum() == 5.0
    with pytest.raises(KeyError, match="missing per-type"):
        labels_from_records([{"any_violation": True, "violations": {"V1": {"count": 1.0}}}])
    with pytest.raises(KeyError, match="violations"):
        labels_from_records([{"any_violation": True}])


def test_labels_from_records_derives_any_violation_and_accepts_flags() -> None:
    records = [{"violations": {v: (v == "V3") for v in VIOLATION_TYPES}}]
    any_t, type_t = labels_from_records(records)
    assert any_t.tolist() == [1.0]
    assert type_t[0].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]


# --------------------------------------------------------------------------
# gradient diagnostics
# --------------------------------------------------------------------------


def test_gradient_diagnostics_reports_usable_gradients() -> None:
    h = torch.randn(4, 6)
    out = gradient_diagnostics(lambda x: x.sum(dim=1) * 0.01, h, eta=1e-8)
    assert out["frac_below_eta"] == 0.0
    assert out["mean_gn2"] > 0.0
    assert out["n_examples"] == 4.0
    assert np.isnan(out["saturation_rate"])  # no logit_fn supplied


def test_gradient_diagnostics_flags_zero_gradient_and_direction_agreement() -> None:
    h = torch.randn(3, 5)
    zero = gradient_diagnostics(lambda x: (x * 0.0).sum(dim=1), h, eta=1e-8)
    assert zero["frac_below_eta"] == 1.0

    def fn_a(x: torch.Tensor) -> torch.Tensor:
        return x[:, 0] * 5.0

    def fn_b(x: torch.Tensor) -> torch.Tensor:
        return x[:, 0] * 2.0

    agree = gradient_diagnostics(
        fn_a,
        h if h.requires_grad else h.detach().requires_grad_(True),
        per_member_risk_fns=[fn_a, fn_b],
    )
    assert agree["direction_agreement"] == pytest.approx(1.0, abs=1e-5)


def test_gradient_diagnostics_reports_saturation() -> None:
    h = torch.randn(2, 4)

    def logit_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), 9.0)

    out = gradient_diagnostics(lambda x: x.sum(dim=1), h, logit_fn=logit_fn)
    assert out["saturation_rate"] == 1.0


def test_gradient_diagnostics_rejects_non_vector_risk() -> None:
    with pytest.raises(ValueError, match=r"\[B\]"):
        gradient_diagnostics(lambda x: x.sum(), torch.randn(2, 3))
