"""Every committed config must load, compose and hash deterministically.

A config typo that silently changes the experiment is the most expensive kind of
bug in a research repo, so the whole ``configs/`` tree is validated here: every
experiment config is loaded strictly, and every fragment is checked against the
schema's field names (a fragment no experiment happens to include would otherwise
rot unnoticed).
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

import pytest
import yaml

from riskproj.utils.config import (
    ExperimentCfg,
    deep_merge,
    dump_config,
    load_config,
    load_config_raw,
)

pytestmark = pytest.mark.unit

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
FRAGMENT_DIRS = ("env", "dynamics", "model", "critic", "train", "eval", "method")

EXP_CONFIGS = sorted((CONFIG_ROOT / "exp").glob("*.yaml"))

#: Discovered *data* artifacts live under configs/ but are not config fragments: they are
#: written by a script (``inspect_contacts.py`` -> ``env/task_geoms.yaml``,
#: ``inspect_model.py`` -> ``model/*_sites.yaml``) and are consumed by the env / policy
#: layers, not merged into an ExperimentCfg. Each is validated where it is read.
DATA_ARTIFACTS = frozenset({"task_geoms.yaml"})
DATA_ARTIFACT_GLOBS = ("*_sites.yaml",)

FRAGMENTS = sorted(
    p
    for d in FRAGMENT_DIRS
    for p in (CONFIG_ROOT / d).glob("*.yaml")
    if p.name not in DATA_ARTIFACTS and not any(fnmatch(p.name, g) for g in DATA_ARTIFACT_GLOBS)
)


def test_exp_configs_exist() -> None:
    # E0-E4 are the roadmap's experiments (agents/11); losing one silently would
    # mean silently dropping part of the plan.
    names = {p.stem for p in EXP_CONFIGS}
    assert {"e0_baseline_repro", "e1_counterfactual", "e2_linearization"} <= names
    assert {"e3_pilot", "e4_main", "toy_smoke"} <= names


@pytest.mark.parametrize("path", EXP_CONFIGS, ids=lambda p: p.stem)
def test_experiment_config_loads_and_is_consistent(path: Path) -> None:
    cfg = load_config(path)

    # Horizons: labels come from executing the chunk open-loop, so all three must
    # agree (agents/04 §1). EnvCfg enforces part of this; assert the rest here so a
    # fragment edit cannot slip through.
    assert cfg.env.exec_horizon == cfg.env.critic_horizon == cfg.critic.horizon
    assert cfg.violations.horizon == cfg.env.critic_horizon

    # Identity: stable hash, and the hash is a function of the resolved config only.
    assert cfg.config_hash() == load_config(path).config_hash()
    assert len(cfg.config_hash()) == 8
    assert cfg.run_name().startswith(cfg.name + "-")

    # Seeding: a run must be reproducible from the committed file.
    assert cfg.seed_list(), "every experiment needs at least one seed"
    assert all(isinstance(s, int) for s in cfg.seed_list())

    # G2 budget is a project constraint, not a preference.
    assert cfg.gates.G2.budget_gb <= 7.0

    # Every method must be distinguishable; duplicate ids would silently collide in
    # the run directory layout runs/<name>-<hash>/<method>/<seed>/.
    ids = [m.id for m in cfg.methods]
    assert len(ids) == len(set(ids))
    for method in cfg.methods:
        if method.inference_mode == "lat" or method.train_mode == "lat":
            assert method.projection.tau > 0.0


@pytest.mark.parametrize("path", EXP_CONFIGS, ids=lambda p: p.stem)
def test_experiment_config_round_trips_through_dump(path: Path, tmp_path: Path) -> None:
    cfg = load_config(path)
    written = dump_config(cfg, tmp_path / "config.yaml")
    reloaded = load_config(written)
    assert reloaded.canonical() == cfg.canonical()
    assert reloaded.config_hash() == cfg.config_hash()


@pytest.mark.parametrize("path", FRAGMENTS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_fragments_only_use_known_sections(path: Path) -> None:
    raw = load_config_raw(path)
    known = set(ExperimentCfg.model_fields)
    unknown = sorted(set(raw) - known)
    assert not unknown, f"{path.name} sets unknown top-level keys {unknown}"
    assert "includes" not in raw, "includes must be consumed by load_config_raw"


def test_include_precedence_and_cycles(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "name: base\nseed: 1\ncritic:\n  hidden: 32\neval:\n  eps_per_cell: 3\n",
        encoding="utf-8",
    )
    (tmp_path / "mid.yaml").write_text(
        "includes: [base.yaml]\nname: mid\ncritic:\n  depth: 4\n", encoding="utf-8"
    )
    (tmp_path / "exp.yaml").write_text(
        "includes: [mid.yaml]\nseed: 7\ncritic:\n  hidden: 64\n", encoding="utf-8"
    )
    raw = load_config_raw(tmp_path / "exp.yaml")
    # later include wins over earlier, and the including file's keys win over all
    assert raw["name"] == "mid"
    assert raw["seed"] == 7
    assert raw["critic"]["hidden"] == 64  # the exp file itself
    assert raw["critic"]["depth"] == 4  # from the include
    assert raw["eval"]["eps_per_cell"] == 3  # from the transitive include

    (tmp_path / "a.yaml").write_text("includes: [b.yaml]\nname: a\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("includes: [a.yaml]\nname: b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        load_config_raw(tmp_path / "a.yaml")


def test_deep_merge_is_recursive_and_non_mutating() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 20}, "c": 4}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 20}, "b": 3, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}, "deep_merge mutated its input"


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    # extra="forbid" on every model: a typo must fail loudly, not silently run a
    # different experiment.
    path = tmp_path / "bad.yaml"
    path.write_text("name: bad\nseed: 0\nn_episodes: 10\n", encoding="utf-8")
    with pytest.raises(Exception, match="n_episodes"):
        load_config(path)


def test_toy_smoke_is_cpu_only_and_cheap() -> None:
    """The smoke config is what CI runs; it must stay cheap and download-free."""
    cfg = load_config(EXP_CONFIGS[0].parent / "toy_smoke.yaml")
    assert cfg.model.name == "toy"
    assert cfg.env.name == "toy"
    assert cfg.model.checkpoint is None, "the smoke run must not need a checkpoint"
    assert cfg.model.device == "cpu"
    assert cfg.train.steps <= 200
    assert cfg.eval.eps_per_cell <= 4
    assert cfg.critic.use_vision is False
    assert cfg.critic.privileged_force is False


def test_yaml_is_valid_and_no_tabs() -> None:
    for path in [*EXP_CONFIGS, *FRAGMENTS]:
        text = path.read_text(encoding="utf-8")
        assert "\t" not in text, f"{path} contains a tab; YAML forbids tabs in indentation"
        assert isinstance(yaml.safe_load(text), dict)
