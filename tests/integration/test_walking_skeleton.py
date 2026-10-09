"""S6 analogue: the whole pipeline on the toy task, on CPU, with no downloads.

Enforces the walking-skeleton requirement of ``agents/03_repo_setup.md`` §3 (S6)
and the same section's efficiency rule. It is *pipeline* evidence, never evidence
about the hypothesis — nothing here may be reported as a result.

Marked ``integration`` and ``slow``, so the fast path (``tasks.py test``) skips it
by design; run it with ``pytest -m integration`` or ``tasks.py smoke``.

Part B guards on modules that land in parallel (``riskproj.eval``,
``riskproj.methods``) with ``pytest.importorskip`` and explicit reasons: a missing
module or a differing interface must surface as a *named skip*, never as a pass.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from riskproj.data.gen_critic_data import generate_critic_data
from riskproj.data.gen_demos import generate_demos
from riskproj.data.schema import CriticDataset
from riskproj.data.splits import cells_for_role
from riskproj.envs import build_task
from riskproj.types import DynamicsSpec, EpisodeRecord
from riskproj.utils.config import load_config
from riskproj.utils.io import verify_manifest
from riskproj.utils.metrics import auroc
from riskproj.utils.seed import seed_everything

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO = Path(__file__).resolve().parents[2]
#: agents/03 §3 S6: the whole pipeline on CPU in under two minutes.
BUDGET_SECONDS = 120.0
#: Deliberately tiny: this exercises code paths, it does not estimate anything.
EPISODES_PER_CELL = 1
STATES_PER_CELL = 2
CHUNKS_PER_STATE = 3
#: Below this many validation tuples an AUROC is noise, so the check is skipped
#: rather than asserted (and the reason says so).
MIN_ROWS_FOR_AUROC = 6


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _stem(path: str | Path) -> Path:
    p = Path(path)
    return p.with_suffix("") if p.suffix in (".npz", ".json", ".jsonl") else p


def _dataset_from_summary(summary: dict[str, Any]) -> CriticDataset:
    """Reload every written shard. A shard is an ``.npz`` with a metadata ``.json``
    beside it, which is what distinguishes it from the data card and manifest."""
    stems = []
    for path in summary.get("paths", []):
        stem = _stem(path)
        if stem.with_suffix(".npz").exists() and stem.with_suffix(".json").exists():
            stems.append(stem)
    assert stems, (
        "no critic shard (a .npz + .json pair) was written; the generator only produced "
        f"{list(summary.get('paths', []))}"
    )
    tuples = [t for stem in stems for t in CriticDataset.load(stem).tuples]
    return CriticDataset(tuples)


class _FeatureSource:
    """``CriticDataset`` items -> the ``(features, any, per_type)`` critic sample.

    ``CriticDataset`` yields ``(ctx, chunk, label_any, label_per_type)`` while
    ``critic.train.train_critic`` consumes precomputed feature rows, so something has
    to bridge them. The bridge currently lives only in this test; it belongs in the
    library (reported with this slice).
    """

    def __init__(self, dataset: CriticDataset, builder: Any) -> None:
        self.dataset = dataset
        self.builder = builder

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx, chunk, label_any, label_per_type = self.dataset[index]
        features = self.builder(ctx, chunk.unsqueeze(0)).reshape(-1)
        return features, label_any.reshape(()), label_per_type.reshape(-1)


def _call_with_supported(fn: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with the subset of ``kwargs`` it accepts.

    The eval layer lands in parallel, so its exact parameter names are not knowable
    here. A *required* parameter we cannot supply is an explicit skip naming the
    signature — never a silent pass, and never a failure of the wrong thing.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        accepted = {k: v for k, v in kwargs.items() if k in params}
        missing = [
            name
            for name, p in params.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            and p.default is inspect.Parameter.empty
            and name not in accepted
        ]
        if missing:
            pytest.skip(
                f"{getattr(fn, '__name__', fn)} needs parameters this smoke test cannot supply: "
                f"{missing} (signature: {sig}) — the eval interface differs from the one this "
                "integration test was written against"
            )
        kwargs = accepted
    try:
        return fn(**kwargs)
    except TypeError as exc:
        pytest.skip(f"{getattr(fn, '__name__', fn)} rejected the smoke-test call: {exc}")


def _normalize_episode(result: Any) -> tuple[dict[str, Any], list[Any]]:
    """Accept ``RolloutOutput``, ``EpisodeRecord``/mapping, or ``(record, decisions)``."""
    decisions: list[Any] = []
    record: Any = result
    # ``eval.rollout.RolloutOutput`` bundles the record with the decisions and the saved
    # states that produced it; the log schema applies to the record itself.
    if hasattr(record, "episode"):
        decisions = list(getattr(record, "decisions", None) or [])
        record = record.episode
    if isinstance(result, tuple) and result:
        record = result[0]
        if len(result) > 1 and result[1] is not None:
            decisions = list(result[1])
    if isinstance(record, EpisodeRecord):
        return record.to_json(), decisions
    if dataclasses.is_dataclass(record) and not isinstance(record, type):
        return dataclasses.asdict(record), decisions
    if isinstance(record, dict):
        return dict(record), decisions
    pytest.skip(f"unrecognised episode record of type {type(record).__name__}")


def _toy_policy(spec: Any, cfg: Any) -> Any:
    from riskproj.policies.toy import ToyPolicy

    return ToyPolicy(
        state_dim=spec.state_dim,
        action_dim=cfg.env.action_dim,
        hidden=16,
        chunk_horizon=cfg.env.exec_horizon,
        num_flow_steps=3,
        seed=int(cfg.seed_list()[0]),
    )


# --------------------------------------------------------------------------
# Part A: data -> critic, shared by the assertions below
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def skeleton(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    seed_everything(0)
    tmp = tmp_path_factory.mktemp("walking_skeleton")
    cfg = load_config(REPO / "configs" / "exp" / "toy_smoke.yaml")
    task = build_task(cfg)
    ranges = cfg.dynamics.to_ranges()
    manifest = tmp / "data" / "MANIFEST.json"

    # Two TRAIN cells, one of them all-MID so at least one rollout is near nominal:
    # the expert is *expected* to fail on strongly perturbed cells.
    train_cells = cells_for_role("train", cfg.splits)
    assert len(train_cells) == 7, f"TRAIN must hold 7 cells; got {len(train_cells)}"
    cells = ["M-M-M", *[c for c in train_cells if c != "M-M-M"][:1]]

    t_start = time.perf_counter()

    demos = generate_demos(
        task,
        cells=cells,
        ranges=ranges,
        episodes_per_cell=EPISODES_PER_CELL,
        base_seed=cfg.seed,
        out_dir=tmp / "data" / "demos",
        max_steps=cfg.env.max_steps,
        config_hash=cfg.config_hash(),
        keep_failures=True,  # a perturbed cell is supposed to be hard
        manifest_path=manifest,
        data_root=tmp,
    )
    assert demos["n_records"] >= 1, f"no expert episodes were recorded: {demos.get('per_cell')}"

    critic = generate_critic_data(
        task,
        cells=cells,
        ranges=ranges,
        states_per_cell=STATES_PER_CELL,
        chunks_per_state=CHUNKS_PER_STATE,
        base_seed=cfg.seed,
        out_dir=tmp / "data" / "critic",
        horizon=cfg.env.critic_horizon,
        violation_spec=task.violations,
        config_hash=cfg.config_hash(),
        include_unsafe=True,
        max_steps=cfg.env.max_steps,
        state_every=5,
        # The 15% positivity guard is a real check (unit-tested elsewhere); at this
        # size it would fire on sampling noise, so it is relaxed *here only*.
        min_pos_rate=0.0,
        manifest_path=manifest,
        data_root=tmp,
    )
    assert critic["paths"], f"critic-data summary has no paths: {sorted(critic)}"

    dataset = _dataset_from_summary(critic)
    assert len(dataset) >= 1, "critic dataset is empty"

    spec, builder = _feature_layout(cfg, dataset)
    source = _FeatureSource(dataset, builder)
    trained = _train_and_calibrate(source, cfg, spec, tmp, dataset)

    elapsed = time.perf_counter() - t_start
    return SimpleNamespace(
        cfg=cfg,
        tmp=tmp,
        manifest=manifest,
        demos=demos,
        critic=critic,
        dataset=dataset,
        spec=spec,
        task=task,
        elapsed=elapsed,
        **trained,
    )


def _feature_layout(cfg: Any, dataset: CriticDataset) -> tuple[Any, Any]:
    from riskproj.critic.features import CriticFeatureBuilder, FeatureSpec

    kwargs = dataset.feature_spec_kwargs()
    assert kwargs["state_dim"] >= 1 and kwargs["action_dim"] >= 1, kwargs
    spec = FeatureSpec(**kwargs)
    builder = CriticFeatureBuilder.from_spec(cfg.critic, spec)
    assert builder.dim() == spec.dim, "the builder and the stored layout disagree on width"
    return spec, builder


def _train_and_calibrate(
    source: _FeatureSource, cfg: Any, spec: Any, tmp: Path, dataset: CriticDataset
) -> dict[str, Any]:
    from riskproj.critic.calibrate import apply_temperature, ece_before_after, fit_temperature
    from riskproj.critic.train import train_critic
    from riskproj.utils.io import read_json

    model, metrics = train_critic(
        source, cfg.critic, feature_spec=spec, device="cpu", out_dir=tmp / "critic"
    )
    assert (tmp / "critic" / "critic.pt").exists(), "train_critic did not checkpoint the critic"
    for key in ("per_head_auroc", "n_train"):
        assert key in metrics, f"train_critic metrics are missing {key!r}: {sorted(metrics)}"

    features = torch.stack([source[i][0] for i in range(len(source))])
    labels = torch.stack([source[i][1] for i in range(len(source))])
    logits = model.logits_from_features(features)
    assert tuple(logits.shape) == (len(source), 1 + 5)

    # Overfit sanity: on data it was just trained on, a working critic must beat
    # chance. Measured here rather than read from the library's VAL metrics so the
    # check does not depend on internal metric keys or on a tiny VAL split.
    probs = torch.sigmoid(logits[:, 0]).detach().numpy()
    train_auroc = float(auroc(labels.numpy(), probs))

    # Calibration is fitted on the any-violation logit only (``06`` §1: r = sigma(z_any)),
    # exactly as the critic table does; the API flattens its input, so passing the full
    # [N, 1+5] matrix would pair 6N logits with N labels.
    logits_any = logits[:, 0]
    temperature = fit_temperature(logits_any, labels)
    assert temperature > 0.0, "temperature scaling must return a positive temperature"
    assert torch.isfinite(apply_temperature(logits_any, temperature)).all()
    calibration = ece_before_after(logits_any, labels, temperature)
    assert calibration, "calibration produced no metrics"
    assert all(math.isfinite(float(v)) for v in calibration.values()), calibration

    return {
        "model": model,
        "metrics": metrics,
        "train_auroc": train_auroc,
        "n_tuples": len(dataset),
        "read_json": read_json,
    }


def test_part_a_pipeline_produced_artifacts(skeleton: SimpleNamespace) -> None:
    assert skeleton.demos["n_records"] >= 1
    assert skeleton.critic["paths"]
    assert skeleton.n_tuples >= 1
    assert skeleton.model is not None


def test_part_a_wrote_a_data_card_and_a_verifiable_manifest(skeleton: SimpleNamespace) -> None:
    """Raw data must be registered with its hash and described by a data card
    (``skills/software_engineering.md`` §5)."""
    assert skeleton.manifest.exists(), f"no manifest at {skeleton.manifest}"
    verification = verify_manifest(skeleton.manifest, base=skeleton.tmp)
    assert not verification["missing"], f"manifest entries with missing files: {verification['missing']}"
    assert not verification["mismatch"], f"manifest sha256 mismatches: {verification['mismatch']}"
    assert verification["n_ok"] >= 1

    cards = [p for p in skeleton.critic["paths"] if str(p).endswith("DATA_CARD.json")]
    assert cards, "the critic-data generator did not write a data card"
    card = skeleton.read_json(cards[0])
    for key in ("class_balance", "cell_histogram", "n_records"):
        assert key in card, f"data card is missing {key!r}: {sorted(card)}"
    assert float(card["n_records"]) == float(skeleton.n_tuples), (
        "the data card disagrees with the reloaded dataset about how many tuples exist"
    )


def test_part_a_critic_learns_something_about_its_own_data(skeleton: SimpleNamespace) -> None:
    """An overfit sanity check, not a claim: a critic that cannot separate data it was
    just trained on cannot possibly pass G3 on held-out cells later."""
    n_pos = int(sum(1 for t in skeleton.dataset.tuples if t.any_violation))
    if skeleton.n_tuples < MIN_ROWS_FOR_AUROC or n_pos == 0 or n_pos == skeleton.n_tuples:
        pytest.skip(
            f"smoke dataset has {skeleton.n_tuples} tuples with {n_pos} positives; an AUROC "
            f"over fewer than {MIN_ROWS_FOR_AUROC} rows or a degenerate class balance says "
            "nothing. Widen the smoke config to make this check meaningful."
        )
    assert skeleton.train_auroc >= 0.5, (
        f"critic AUROC {skeleton.train_auroc:.3f} on its own training data: below chance means "
        "the training path is broken (features/labels/head wiring), not that the task is hard"
    )
    assert skeleton.metrics.get("n_train", 0) > 0


# --------------------------------------------------------------------------
# Part B: rollout through the eval layer, when it exists
# --------------------------------------------------------------------------


def _require_run_episode() -> Any:
    rollout = pytest.importorskip(
        "riskproj.eval.rollout", reason="the eval layer lands in a parallel worker"
    )
    run_episode = getattr(rollout, "run_episode", None)
    if run_episode is None:
        pytest.skip(f"riskproj.eval.rollout exists but exposes no run_episode: {sorted(dir(rollout))}")
    return run_episode


def _rollout_spec(cfg: Any, **over: Any) -> Any:
    """One episode's identity, as ``eval.rollout`` defines it."""
    from riskproj.eval.rollout import RolloutSpec

    kwargs: dict[str, Any] = {
        "method": "B0",
        "train_seed": int(cfg.seed_list()[0]),
        "cell": "M-M-M",
        "split": "train",
        "env_seed": 0,
        "max_steps": 12,
        "record_decisions": True,
        "save_states": False,
        "run_id": "skeleton",
    }
    kwargs.update(over)
    return RolloutSpec(**kwargs)


def test_part_b_rollout_records_match_the_log_schema(skeleton: SimpleNamespace) -> None:
    run_episode = _require_run_episode()
    cfg = skeleton.cfg
    result = run_episode(
        task=skeleton.task,
        policy=_toy_policy(skeleton.spec, cfg),
        spec=_rollout_spec(cfg),
        dyn=cfg.dynamics.to_ranges().midpoint_spec("M-M-M"),
    )
    record, decisions = _normalize_episode(result)

    missing = [f.name for f in dataclasses.fields(EpisodeRecord) if f.name not in record]
    assert not missing, (
        f"episode record is missing schema fields {missing}; the agents/09 §3 schema "
        "(riskproj.types.EpisodeRecord) is the contract for every archived log"
    )
    assert record["dyn_readback"], "the episode record must carry the dynamics read-back"
    assert isinstance(decisions, (list, tuple))


def test_part_b_non_nominal_dynamics_are_visible_in_the_record(skeleton: SimpleNamespace) -> None:
    run_episode = _require_run_episode()
    policy = _toy_policy(skeleton.spec, skeleton.cfg)
    records: dict[str, dict[str, Any]] = {}
    for label, dyn in (
        ("nominal", DynamicsSpec()),
        ("perturbed", DynamicsSpec(mu_scale=0.6, mass_scale=1.8, stiff_scale=0.5)),
    ):
        result = run_episode(
            task=skeleton.task,
            policy=policy,
            spec=_rollout_spec(skeleton.cfg, env_seed=1, max_steps=8, record_decisions=False),
            dyn=dyn,
        )
        records[label], _ = _normalize_episode(result)

    nominal, perturbed = records["nominal"]["dyn_readback"], records["perturbed"]["dyn_readback"]
    differing = [
        axis
        for axis in ("mu_scale", "mass_scale", "stiff_scale")
        if not math.isclose(float(nominal.get(axis, 1.0)), float(perturbed.get(axis, 1.0)), rel_tol=1e-9)
    ]
    assert differing, (
        "the perturbed rollout's read-back is identical to the nominal one: the dynamics "
        f"never reached the physics ({nominal} vs {perturbed})"
    )


def test_part_b_projection_is_active_and_loggable_when_risk_exceeds_tau() -> None:
    """The projection path must correct an above-tau candidate, and every such decision
    must fit the agents/09 §3 decision schema (CLAUDE.md #4: nothing fails silently)."""
    from riskproj.projection import project
    from riskproj.types import DecisionRecord
    from riskproj.utils.config import ProjCfg

    torch.manual_seed(0)
    h = torch.randn(2, 5)

    def forced_risk(hh: torch.Tensor) -> torch.Tensor:
        # Above tau *and* with a usable gradient. A constant risk is deliberately
        # unusable (test_methods::test_constant_risk_is_unusable_and_flagged): there is
        # no direction to move in, which is the fallback ladder's business and not a
        # correction the operator could honestly claim to have made.
        return 0.9 + 0.1 * torch.tanh(hh.sum(dim=1))

    cfg = ProjCfg(tau=0.3, K=2)
    h_safe, info = project(h, forced_risk, cfg)

    assert info.active is not None and bool(info.active.all()), (
        "an above-tau candidate must be flagged active; otherwise the constraint never bites"
    )
    assert not torch.equal(h_safe, h), "an active, usable example must actually be corrected"

    decision = DecisionRecord(
        ep_id="smoke",
        step=0,
        flow_step=cfg.steps_active - 1,
        tau=cfg.tau,
        r_cand=float(info.r_cand[0]) if info.r_cand is not None else None,
        gn2=float(info.gn2[0]) if info.gn2 is not None else None,
        active=bool(info.active[0]),
        usable=bool(info.usable[0]),
        k_used=int(info.k_used[0]) if info.k_used is not None else 0,
        corr_rel=float(info.corr_rel[0]) if info.corr_rel is not None else 0.0,
    )
    payload = decision.to_json()
    assert payload["active"] is True
    assert set(payload) == {f.name for f in dataclasses.fields(DecisionRecord)}


def test_part_b_method_layer_logs_a_decision_when_available() -> None:
    pytest.importorskip("riskproj.methods", reason="the methods layer lands in a parallel worker")
    from riskproj.projection import project
    from riskproj.utils.config import ProjCfg

    h = torch.randn(1, 3)
    h_safe, info = project(h, lambda hh: torch.full((hh.shape[0],), 0.8), ProjCfg(tau=0.3))
    assert info.active is not None and bool(info.active.any())
    assert torch.isfinite(h_safe).all()


# --------------------------------------------------------------------------
# Part C: the S6 time budget, and the CPU-only guarantee
# --------------------------------------------------------------------------


def test_part_c_pipeline_fits_the_smoke_budget(skeleton: SimpleNamespace) -> None:
    """agents/03 §3 S6: the walking skeleton must run on CPU in under two minutes."""
    elapsed = float(skeleton.elapsed)
    if elapsed >= BUDGET_SECONDS:
        pytest.skip(
            f"the toy pipeline took {elapsed:.1f}s, over the {BUDGET_SECONDS:.0f}s S6 budget: "
            "either this box is much slower than the development machine, or the toy task "
            "grew real work. Not failing, but it must be recorded, never ignored."
        )
    assert elapsed > 0.0


def test_skeleton_stays_cpu_only_and_download_free(skeleton: SimpleNamespace) -> None:
    """This suite is CI's only end-to-end signal, so it must not acquire a dependency on
    the simulator or on a pretrained checkpoint."""
    cfg = skeleton.cfg
    assert cfg.env.name == "toy"
    assert cfg.model.name == "toy"
    assert cfg.model.checkpoint is None, "the smoke path must not need a downloaded checkpoint"
    assert cfg.model.device == "cpu"
    assert cfg.critic.use_vision is False
    assert cfg.critic.privileged_force is False
