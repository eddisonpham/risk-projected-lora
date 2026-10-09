"""Site catalog, resolution and module discovery.

The interesting property here is negative: the resolver must never *fabricate* a
module name. Until smoke test S3 writes ``configs/model/smolvla_sites.yaml`` the
spec's module names are [R], so the fallback path has to warn and hand back empty
patterns, and discovery has to fail with the candidates it saw rather than guess.

The synthetic tree also pins the documented tie-breaks (S2 = last block, S3/S4 =
median block, S5 = first block, S1 = the action-dimension projection), because
"which of the twelve matching linears" is exactly the silent choice that would
invalidate a depth sweep.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from riskproj.policies import sites as sites_mod  # noqa: E402
from riskproj.policies.sites import (  # noqa: E402
    CANDIDATE_PATTERNS,
    SITE_CATALOG,
    SiteSpec,
    default_site_chain,
    discover_candidates,
    discover_modules,
    load_sites_yaml,
    register_discovered_sites,
    resolve_sites,
    write_sites_yaml,
)
from riskproj.utils.config import SITE_IDS, SiteCfg  # noqa: E402

pytestmark = pytest.mark.unit

SITES_LOGGER = "riskproj.policies.sites"


# --------------------------------------------------------------------------
# catalog
# --------------------------------------------------------------------------


def test_catalog_covers_every_site_id() -> None:
    assert set(SITE_CATALOG) == set(SITE_IDS)
    for sid, spec in SITE_CATALOG.items():
        assert spec.id == sid
        assert spec.downstream, f"{sid} must document what consumes it"
        assert spec.notes, f"{sid} must carry a note"
        # nothing is resolved before S3 runs: the patterns are [R]
        assert spec.module_pattern == ""


def test_action_space_and_latent_sites_are_distinguished() -> None:
    assert not SITE_CATALOG["S0"].is_module_site()
    assert not SITE_CATALOG["S0f"].is_module_site()
    for sid in ("S1", "S2", "S3", "S4", "S5", "S6"):
        assert SITE_CATALOG[sid].is_module_site()
    # S1 is the *input* of the final action projection
    assert SITE_CATALOG["S1"].capture == "input"
    assert SITE_CATALOG["S2"].capture == "output"
    assert "DEFAULT" in SITE_CATALOG["S2"].notes


def test_depth_sweep_order_is_the_catalog_order() -> None:
    assert default_site_chain() == list(SITE_IDS)


def test_candidate_patterns_exist_for_every_module_site() -> None:
    for sid in ("S1", "S2", "S3", "S4", "S5", "S6"):
        assert CANDIDATE_PATTERNS[sid], f"{sid} needs heuristic patterns for the pre-S3 path"
    assert set(CANDIDATE_PATTERNS).issubset(set(SITE_IDS))


# --------------------------------------------------------------------------
# resolution / yaml
# --------------------------------------------------------------------------


def test_resolve_falls_back_to_catalog_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = SiteCfg(resolved_file=str(tmp_path / "missing.yaml"))
    with caplog.at_level(logging.WARNING, logger=SITES_LOGGER):
        resolved = resolve_sites(cfg)
    assert set(resolved) == set(SITE_IDS)
    assert all(spec.module_pattern == "" for spec in resolved.values())
    assert all(not spec.resolved() for spec in resolved.values())
    assert any("UNVERIFIED" in rec.message for rec in caplog.records)


def test_a_written_file_wins_over_the_catalog(tmp_path: Path) -> None:
    specs = {sid: SiteSpec(**vars(spec)) for sid, spec in SITE_CATALOG.items()}
    specs["S2"].module_pattern = "model.layers.-1.mlp.down_proj"
    specs["S2"].matched_by = "inspect_model"
    path = write_sites_yaml(
        tmp_path / "smolvla_sites.yaml",
        specs,
        model_name="smolvla",
        source="scripts/inspect_model.py",
        git_sha="deadbeef",
    )
    assert path.exists()
    resolved = resolve_sites(SiteCfg(resolved_file=str(path)))
    assert resolved["S2"].module_pattern == "model.layers.-1.mlp.down_proj"
    assert resolved["S2"].matched_by == "inspect_model"
    # sites the file omits are filled from the catalog, not dropped
    assert set(resolved) == set(SITE_IDS)


def test_sites_yaml_round_trip(tmp_path: Path) -> None:
    specs = {sid: SiteSpec(**vars(spec)) for sid, spec in SITE_CATALOG.items()}
    specs["S1"].module_pattern = "action_out_proj"
    path = write_sites_yaml(
        tmp_path / "s.yaml", specs, model_name="m", source="test", git_sha="abc"
    )
    loaded = load_sites_yaml(path)
    assert set(loaded) == set(SITE_IDS)
    assert loaded["S1"].module_pattern == "action_out_proj"
    assert loaded["S1"].capture == "input"
    assert loaded["S1"].kind == SITE_CATALOG["S1"].kind

    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["model_name"] == "m"
    assert payload["schema_version"] == 1
    assert "unverified_patterns" in payload


def test_sites_yaml_rejects_unknown_ids(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("sites:\n  S99:\n    kind: expert_block\n    downstream: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown site id"):
        load_sites_yaml(bad)
    bad2 = tmp_path / "bad2.yaml"
    bad2.write_text("nope: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'sites' key"):
        load_sites_yaml(bad2)
    with pytest.raises(ValueError, match="unknown site ids"):
        write_sites_yaml(
            tmp_path / "o.yaml",
            {"S99": SITE_CATALOG["S2"]},
            model_name="m",
            source="s",
            git_sha="g",
        )


class _StubAdapter:
    def __init__(self, sites: dict[str, object]) -> None:
        self._sites = sites

    def sites(self) -> dict[str, object]:
        return dict(self._sites)

    def register_site(self, site_id: str, module: object, *, override: bool = False) -> None:
        self._sites[site_id] = module


def test_resolution_keeps_only_sites_the_adapter_registered(tmp_path: Path) -> None:
    cfg = SiteCfg(resolved_file=str(tmp_path / "missing.yaml"))
    stub = _StubAdapter({"S2": object()})
    resolved = resolve_sites(cfg, adapter=stub)
    assert set(resolved) == {"S0", "S0f", "S2"}  # action-space sites always survive
    assert "S5" not in resolved


def test_action_space_sites_survive_an_empty_adapter(tmp_path: Path) -> None:
    """S0/S0f belong to no module, so an adapter that registered nothing loses only\n    the latent sites — the post-hoc and in-flow arms stay usable."""
    cfg = SiteCfg(resolved_file=str(tmp_path / "missing.yaml"))
    resolved = resolve_sites(cfg, adapter=_StubAdapter({}))
    assert set(resolved) == {"S0", "S0f"}


def test_default_resolved_file_is_the_documented_path() -> None:
    assert SiteCfg().resolved_file == "configs/model/smolvla_sites.yaml"


# --------------------------------------------------------------------------
# discovery on a synthetic SmolVLA-shaped tree
# --------------------------------------------------------------------------


class _ExpertBlock(nn.Module):
    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.self_attn = nn.ModuleDict({"o_proj": nn.Linear(d, d)})
        self.cross_attn = nn.ModuleDict({"o_proj": nn.Linear(d, d)})
        self.mlp = nn.ModuleDict({"down_proj": nn.Linear(d, d), "up_proj": nn.Linear(d, d)})


class _FakeSmolVLA(nn.Module):
    """Shaped like the anatomy in 05 §2, with names that are *plausible* but unverified."""

    def __init__(
        self, blocks: int = 6, d: int = 8, action_dim: int = 4, vlm_layers: int = 4
    ) -> None:
        super().__init__()
        self.model = nn.ModuleDict(
            {"layers": nn.ModuleList([_ExpertBlock(d) for _ in range(blocks)])}
        )
        self.action_out_proj = nn.Linear(d, action_dim)
        # decoy: matches the S1 pattern but produces the wrong width
        self.action_proj_decoy = nn.Linear(d, 2 * d)
        self.vlm = nn.ModuleDict(
            {
                "text_model": nn.ModuleDict(
                    {"layers": nn.ModuleList([nn.Linear(d, d) for _ in range(vlm_layers)])}
                )
            }
        )


@pytest.fixture()
def fake_model() -> _FakeSmolVLA:
    torch.manual_seed(0)
    return _FakeSmolVLA()


def test_s1_uses_the_action_dim_width_filter(fake_model: _FakeSmolVLA) -> None:
    name, module = discover_modules(fake_model, SITE_CATALOG["S1"], action_dim=4)
    assert name == "action_out_proj"
    assert isinstance(module, nn.Linear)
    assert module.out_features == 4


def test_s2_is_the_last_block_and_s5_the_first(fake_model: _FakeSmolVLA) -> None:
    s2, _ = discover_modules(fake_model, SITE_CATALOG["S2"])
    assert s2 == "model.layers.5.mlp.down_proj"
    s5, _ = discover_modules(fake_model, SITE_CATALOG["S5"])
    assert s5.startswith("model.layers.0.")
    assert s5 != s2


def test_s3_and_s4_take_the_median_block(fake_model: _FakeSmolVLA) -> None:
    s3, _ = discover_modules(fake_model, SITE_CATALOG["S3"])
    assert s3 == "model.layers.3.cross_attn.o_proj"
    s4, _ = discover_modules(fake_model, SITE_CATALOG["S4"])
    assert s4 == "model.layers.3.self_attn.o_proj"


def test_s6_takes_the_last_vlm_layer(fake_model: _FakeSmolVLA) -> None:
    s6, _ = discover_modules(fake_model, SITE_CATALOG["S6"])
    assert s6 == "vlm.text_model.layers.3"


def test_discovery_is_deterministic(fake_model: _FakeSmolVLA) -> None:
    first = discover_modules(fake_model, SITE_CATALOG["S2"])
    for _ in range(3):
        assert discover_modules(fake_model, SITE_CATALOG["S2"]) == first


def test_action_space_sites_have_no_module(fake_model: _FakeSmolVLA) -> None:
    with pytest.raises(ValueError, match="action-space site"):
        discover_candidates(fake_model, SITE_CATALOG["S0"])


def test_no_match_raises_with_nearby_candidates() -> None:
    class _Empty(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Linear(4, 4)  # matches nothing for S1

    with pytest.raises(LookupError) as exc:
        discover_modules(_Empty(), SITE_CATALOG["S1"], action_dim=4)
    assert "Nearby candidates" in str(exc.value)
    assert "S1" in str(exc.value)


def test_explicit_patterns_override_the_heuristics(fake_model: _FakeSmolVLA) -> None:
    name, _ = discover_modules(fake_model, SITE_CATALOG["S2"], patterns=("down_proj",))
    assert name.endswith("down_proj")
    name2, _ = discover_modules(
        fake_model, SITE_CATALOG["S2"], patterns=("model.layers.0.mlp.down_proj",)
    )
    assert name2 == "model.layers.0.mlp.down_proj"
    with pytest.raises(LookupError):
        discover_modules(fake_model, SITE_CATALOG["S2"], patterns=("nothing_like_this",))


def test_register_discovered_sites_records_errors_without_raising(fake_model: _FakeSmolVLA) -> None:
    stub = _StubAdapter({})
    specs = {sid: SiteSpec(**vars(spec)) for sid, spec in SITE_CATALOG.items()}
    specs["S2"].module_pattern = "definitely.not.present"
    matched, errors = register_discovered_sites(stub, fake_model, specs, action_dim=4)
    assert "S2" in errors
    assert "S2" not in matched
    assert "S1" in matched and "S5" in matched
    assert set(stub.sites()) == set(matched)
    # S0/S0f are never registered as modules
    assert "S0" not in stub.sites() and "S0f" not in stub.sites()


def test_describe_match_reports_failures_as_data(fake_model: _FakeSmolVLA) -> None:
    record = sites_mod.describe_match(fake_model, SITE_CATALOG["S2"])
    assert record["resolved"] is True
    assert record["module"] == "model.layers.5.mlp.down_proj"
    assert record["alternatives"]
    bad = SiteSpec(id="S2", kind="expert_block", downstream="x", module_pattern="nope.nope")
    failed = sites_mod.describe_match(fake_model, bad)
    assert failed["resolved"] is False and "error" in failed
