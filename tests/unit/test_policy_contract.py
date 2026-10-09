"""The parametrised adapter contract suite (``skills/software_engineering.md`` §3).

Every adapter must pass, in this order of importance:

1. **zero LoRA == base policy**, exactly;
2. a **site gradient exists** (``f_act`` really depends on the activation);
3. **per-example independence** (permuting the batch permutes the outputs);
4. **seeded determinism** (same seed, same outputs).

Plus the plumbing that makes the projection work in practice: ``f_act`` reproducing
``forward``'s chunk, site flatten/unflatten round-trip, and LoRA attach behaviour.

The toy adapter always runs here. The real adapters need a downloaded checkpoint and
LeRobot, so they are parametrised but skipped unless ``RISKPROJ_TEST_REAL_POLICIES=1``
(they are exercised by the smoke suite S0-S4 instead, where a checkpoint is present).
"""

from __future__ import annotations

import os
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402
from tests.tolerances import EXACT, FLOW  # noqa: E402
from torch import nn  # noqa: E402

from riskproj.policies.base import (  # noqa: E402
    LoRALinear,
    discover_linear_targets,
    swap_module,
)
from riskproj.policies.toy import ToyPolicy  # noqa: E402
from riskproj.utils.config import LoraCfg, ModelCfg  # noqa: E402

pytestmark = pytest.mark.unit

REAL = os.environ.get("RISKPROJ_TEST_REAL_POLICIES") == "1"
ADAPTER_KINDS = ("toy-flow_matching", "toy-regression", "smolvla", "act")


def make_adapter(kind: str) -> Any:
    if kind.startswith("toy-"):
        head = kind.split("-", 1)[1]
        return ToyPolicy(
            state_dim=6,
            action_dim=3,
            hidden=8,
            chunk_horizon=4,
            num_flow_steps=3,
            action_head=head,
            seed=0,
        )
    if kind == "smolvla":
        from riskproj.policies.smolvla import SmolVLAAdapter

        if not (REAL and SmolVLAAdapter.is_available()):
            pytest.skip("set RISKPROJ_TEST_REAL_POLICIES=1 and install the VLA stack to run this")
        return SmolVLAAdapter(ModelCfg(name="smolvla", chunk_size=4), action_dim=3, state_dim=6)
    if kind == "act":
        from riskproj.policies.act import ACTAdapter

        if not (REAL and ACTAdapter.is_available()):
            pytest.skip("set RISKPROJ_TEST_REAL_POLICIES=1 and install the VLA stack to run this")
        return ACTAdapter(
            ModelCfg(name="act", chunk_size=4, action_head="regression"), action_dim=3
        )
    raise ValueError(f"unknown adapter kind {kind!r}")


@pytest.fixture(params=ADAPTER_KINDS)
def adapter(request: pytest.FixtureRequest) -> Any:
    return make_adapter(request.param)


def _toy(*, head: str = "flow_matching", seed: int = 0) -> ToyPolicy:
    return ToyPolicy(
        state_dim=6,
        action_dim=3,
        hidden=8,
        chunk_horizon=4,
        num_flow_steps=3,
        action_head=head,
        seed=seed,
    )


# --------------------------------------------------------------------------
# contract clauses
# --------------------------------------------------------------------------


def test_zero_lora_equals_base_policy() -> None:
    """LoRA must be an exact identity at attach time (``B`` is zero-initialised)."""
    policy = _toy()
    batch = policy.sample_batch(3, seed=1)
    before = policy.forward(batch, site="S2")
    assert policy.attach_lora(LoraCfg(r=4, alpha=8)) == 1
    after = policy.forward(batch, site="S2")
    assert torch.allclose(before.chunk, after.chunk, atol=EXACT)
    assert torch.allclose(after.extras["lora_delta"], torch.zeros_like(after.extras["lora_delta"]))


def test_site_gradient_exists_and_is_finite(adapter: Any) -> None:
    batch = adapter.sample_batch(2, seed=2) if hasattr(adapter, "sample_batch") else None
    if batch is None:
        pytest.skip("real adapters need a task batch; covered by the smoke suite")
    site = adapter.default_site
    if not adapter.has_site(site):
        site = sorted(adapter.sites())[0]
    out = adapter.forward(batch, site=site)
    h = out.h.detach().clone().requires_grad_(True)
    ctx = adapter.make_context(batch, site=site)
    chunk = adapter.f_act(ctx, h, site)
    assert chunk.shape == out.chunk.shape
    (grad,) = torch.autograd.grad(chunk.sum(), h)
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0, "f_act does not depend on the site activation"


def test_per_example_independence(adapter: Any) -> None:
    if not hasattr(adapter, "sample_batch"):
        pytest.skip(
            "real adapters need a task batch; per-example independence is checked by the suite"
        )
    batch = adapter.sample_batch(4, seed=3)
    perm = torch.tensor([2, 0, 3, 1])
    permuted = {k: (v[perm] if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    site = (
        adapter.default_site
        if adapter.has_site(adapter.default_site)
        else sorted(adapter.sites())[0]
    )
    base = adapter.forward(batch, site=site).chunk
    shuffled = adapter.forward(permuted, site=site).chunk
    assert torch.allclose(base[perm], shuffled, atol=1e-6)


def test_seeded_determinism() -> None:
    a = _toy(seed=7)
    b = _toy(seed=7)
    batch = a.sample_batch(2, seed=11)
    assert torch.equal(a.forward(batch).chunk, b.forward(batch).chunk)
    c = _toy(seed=8)
    assert not torch.allclose(a.forward(batch).chunk, c.forward(batch).chunk)


def test_f_act_reproduces_forward_chunk() -> None:
    """``f_act`` on the candidate activation must equal what ``forward`` produced."""
    for head in ("flow_matching", "regression"):
        policy = _toy(head=head)
        batch = policy.sample_batch(3, seed=5)
        out = policy.forward(batch, site="S2")
        ctx = policy.make_context(batch, site="S2")
        again = policy.f_act(ctx, out.h, "S2")
        assert torch.allclose(out.chunk, again, atol=FLOW), head
        # with LoRA attached, the candidate is still what forward used
        policy.attach_lora(LoraCfg(r=2, alpha=4))
        out2 = policy.forward(batch, site="S2")
        again2 = policy.f_act(policy.make_context(batch, "S2"), out2.h, "S2")
        assert torch.allclose(out2.chunk, again2, atol=FLOW), head


def test_flow_head_needs_flow_state() -> None:
    policy = _toy(head="flow_matching")
    with pytest.raises(KeyError, match="x_tau"):
        policy.forward({"state": torch.zeros(2, 6)}, site="S2")
    with pytest.raises(ValueError, match="no flow state"):
        policy.f_act(
            policy.make_context({"state": torch.zeros(2, 6)}, "S2"), torch.zeros(2, 8), "S2"
        )


def test_site_flatten_round_trip() -> None:
    policy = _toy()
    h = torch.randn(5, 3, 4, 2)
    flat = policy.flatten_site(h)
    assert flat.shape == (5, 24)
    assert torch.equal(policy.unflatten_site(flat, h), h)
    with pytest.raises(ValueError):
        policy.unflatten_site(torch.randn(5, 23), h)
    with pytest.raises(ValueError):
        policy.unflatten_site(torch.randn(4, 24), h)


def test_unknown_site_names_the_available_ones() -> None:
    policy = _toy()
    with pytest.raises(KeyError, match="available"):
        policy.site_module("S5")


def test_sites_reports_the_documented_degeneracy() -> None:
    policy = _toy()
    assert set(policy.sites()) == {"S1", "S2"}
    assert policy.sites()["S1"] is policy.sites()["S2"]


# --------------------------------------------------------------------------
# LoRA plumbing
# --------------------------------------------------------------------------


class _Block(nn.Module):
    def __init__(self, d: int = 4) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        self.mlp = nn.ModuleDict({"down_proj": nn.Linear(d, d), "up_proj": nn.Linear(d, d)})


class _Stack(nn.Module):
    def __init__(self, blocks: int = 6, d: int = 4) -> None:
        super().__init__()
        self.model = nn.ModuleDict({"layers": nn.ModuleList([_Block(d) for _ in range(blocks)])})


def test_lora_linear_is_identity_at_init_and_merges_exactly() -> None:
    torch.manual_seed(0)
    base = nn.Linear(8, 6)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 8)
    assert lora.scale == 2.0
    assert torch.allclose(lora(x), base(x), atol=EXACT)
    assert torch.allclose(lora.merged_weight(), base.weight, atol=EXACT)
    assert all(not p.requires_grad for p in lora.base.parameters())
    assert all(p.requires_grad for p in lora.lora_A.parameters())
    with torch.no_grad():
        lora.lora_B.weight.normal_()
    merged = F.linear(x, lora.merged_weight(), base.bias)
    assert torch.allclose(lora(x), merged, atol=1e-5)
    plain = lora.merged()
    assert torch.allclose(plain(x), lora(x), atol=1e-5)


def test_discover_linear_targets_keeps_only_the_last_blocks() -> None:
    stack = _Stack(blocks=6, d=4)
    suffixes = ("q_proj", "v_proj", "o_proj", "down_proj")
    all_names = discover_linear_targets(stack, suffixes, last_n_blocks=-1)
    assert len(all_names) == 6 * 4
    last_two = discover_linear_targets(stack, suffixes, last_n_blocks=2)
    assert len(last_two) == 8
    assert all((".4." in n or ".5." in n) for n in last_two)
    assert "model.layers.5.mlp.down_proj" in last_two
    assert "model.layers.0.q_proj" not in last_two
    with pytest.raises(ValueError):
        discover_linear_targets(stack, (), last_n_blocks=2)
    assert discover_linear_targets(stack, ("does_not_exist",), last_n_blocks=2) == []


def test_swap_module_keeps_the_path_and_registers_parameters() -> None:
    stack = _Stack(blocks=2, d=4)
    name = "model.layers.1.q_proj"
    old = stack.get_submodule(name)
    lora = LoRALinear(old, r=2, alpha=4)
    swap_module(stack, name, lora)
    assert stack.get_submodule(name) is lora
    params = dict(stack.named_parameters())
    assert f"{name}.lora_A.weight" in params
    assert f"{name}.base.weight" in params
    with pytest.raises(KeyError):
        swap_module(stack, "model.layers.1.missing_proj", nn.Linear(4, 4))
    with pytest.raises(ValueError):
        swap_module(stack, "", nn.Linear(4, 4))


def test_attach_lora_freezes_everything_else() -> None:
    policy = _toy()
    n_total = policy.n_total()
    assert policy.attach_lora(LoraCfg(r=4, alpha=8)) == 1
    policy.assert_no_gradients_outside_lora()
    assert 0 < policy.n_trainable() < n_total
    # the site module really is swapped, not merely annotated
    assert isinstance(policy.sites()["S2"], LoRALinear)
    # lora_A is [r, hidden] and lora_B is [hidden, r]; nothing else is trainable
    assert policy.n_trainable() == 2 * (4 * policy.hidden)
    with pytest.raises(ValueError, match="already attached"):
        policy.attach_lora(LoraCfg(r=4, alpha=8))


def test_attach_lora_rank_zero_is_a_no_op() -> None:
    policy = _toy()
    assert policy.attach_lora(LoraCfg(r=0, alpha=8)) == 0
    assert not isinstance(policy.sites()["S2"], LoRALinear)
    with pytest.raises(AssertionError):
        policy.assert_no_gradients_outside_lora()  # no LoRA registered: the check is meaningless


@pytest.mark.slow
def test_real_adapters_expose_the_contract() -> None:
    for kind in ("smolvla", "act"):
        adapter = make_adapter(kind)  # skips unless the VLA stack + checkpoint are present
        assert callable(adapter.f_act)
        assert isinstance(adapter.sites(), dict)
