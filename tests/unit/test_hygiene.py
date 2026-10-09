"""Repository hygiene: the rules that must not rot as the project grows.

Enforces ``agents/skills/software_engineering.md`` §1.3 (one-way dependencies),
§2 (registries), §4 (quality gates) and ``agents/03_repo_setup.md`` §4 (task
targets), §5 (upstream modification policy), §6 (provenance).

These are cheap structural checks on purpose: they catch the class of change that
is invisible in review but silently invalidates a research claim (a layer
inversion, a renamed spec file, an artifact that quietly enters git, a projection
call that stops being differentiable because someone wrapped a rollout in
``no_grad``).
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
PKG = SRC / "riskproj"

#: Layer index from ``skills/software_engineering.md`` §1.3:
#: utils <- envs/data <- policies/critic <- projection <- methods <- eval <- scripts.
#: A module may import only from its own layer or below.
LAYER: dict[str, int] = {
    "utils": 0,
    "envs": 1,
    "data": 2,
    "policies": 3,
    "critic": 4,
    "projection": 5,
    "methods": 6,
    "eval": 7,
}
#: Entry points / top-level modules that may import anything below them.
TOP_LAYER = 8
#: Modules that are pure vocabulary and must import nothing else from the package.
CORE = {"types", "protocols", "obs"}


def _iter_modules() -> list[str]:
    """Dotted names of every module in the package, discovered from the filesystem.

    ``walk_packages`` imports subpackages to descend into them, so a broken
    subpackage surfaces through ``onerror`` rather than as a traceback from
    discovery itself — that keeps the failure attribution on the module.
    """
    import riskproj

    broken: list[str] = []

    def _onerror(name: str) -> None:
        broken.append(name)

    found = [
        info.name
        for info in pkgutil.walk_packages(
            riskproj.__path__, prefix="riskproj.", onerror=_onerror
        )
        # __main__-style shims are not importable as part of the API
        if not info.name.endswith(".__main__")
    ]
    assert not broken, f"package discovery could not import: {broken}"
    assert found, "package discovery returned nothing; the layout changed"
    return sorted(found)


def _riskproj_imports(path: Path) -> set[str]:
    """Every ``riskproj...`` module named by an import in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(a.name for a in node.names if a.name.startswith("riskproj"))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("riskproj"):
            targets.add(node.module)
            targets.update(f"{node.module}.{a.name}" for a in node.names)
    return targets


def test_every_module_imports_cleanly() -> None:
    """A module that cannot be imported is a broken deliverable, not a style nit."""
    failures: list[str] = []
    for name in _iter_modules():
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - the point is to report *any* failure
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "modules failed to import:\n" + "\n".join(failures)


def _source_layer(py: Path) -> tuple[int | None, bool]:
    """``(layer, is_core)`` for a file inside the package.

    ``layer is None`` and ``is_core is True`` marks the pure-vocabulary modules,
    which may not import any subsystem at all. A package ``__init__`` sits at the
    top: it re-exports its own subtree, so it is checked like an entry point.
    """
    parts = py.relative_to(PKG).parts
    if len(parts) == 1:
        head = parts[0][:-3]  # strip ".py"
        if head == "__init__":
            return TOP_LAYER, False
        if head in CORE:
            return None, True
        # a top-level module (e.g. a CLI) behaves like an entry point
        return TOP_LAYER, False
    head = parts[0]
    if head in CORE:
        return None, True
    return LAYER.get(head, TOP_LAYER), False


def test_dependency_layers_are_one_way() -> None:
    """No module may import from a layer above itself (the §1.3 order)."""
    violations: list[str] = []
    for py in sorted(PKG.rglob("*.py")):
        source_layer, is_core = _source_layer(py)
        rel = py.relative_to(REPO).as_posix()
        for target in sorted(_riskproj_imports(py)):
            heads = target.split(".")
            if len(heads) < 2:
                continue  # `import riskproj` itself
            target_head = heads[1]
            if target_head in CORE:
                continue  # vocabulary is importable from everywhere
            if is_core:
                violations.append(
                    f"{rel} is core vocabulary but imports {target}; types/protocols/obs "
                    "must stay dependency-free"
                )
                continue
            target_layer = LAYER.get(target_head, TOP_LAYER)
            if source_layer is not None and target_layer > source_layer:
                violations.append(
                    f"{rel} (layer {source_layer}) imports {target} (layer {target_layer})"
                )
    assert not violations, (
        "dependency order violated (utils <- envs/data <- policies/critic <- projection "
        "<- methods <- eval <- scripts):\n" + "\n".join(violations)
    )


def test_core_vocabulary_has_no_simulator_or_vla_imports() -> None:
    """``types``/``protocols``/``obs`` are vocabulary: no torch-free promise, but no
    dependency on the simulator or the VLA stack either."""
    for name in ("types.py", "protocols.py", "obs.py"):
        text = (PKG / name).read_text(encoding="utf-8")
        for banned in ("import mujoco", "import gymnasium", "import lerobot", "import peft"):
            assert banned not in text, f"{name} must not import {banned.split()[-1]}"


# --------------------------------------------------------------------------
# differentiability of the projection path
# --------------------------------------------------------------------------

#: Entry points that must never be evaluated inside a ``no_grad``/``inference_mode``
#: block: the risk value and its gradient *are* the method (``07`` §6, ``05`` §6).
#: Deliberately narrow — plain inference of a base policy under ``no_grad`` is fine,
#: so ``f_act`` and ``risk_fn`` are not listed; a false positive here would train
#: people to ignore the check.
DIFFERENTIABLE_CALLS = {"project", "project_fn", "make_risk_fn"}
NO_GRAD_MARKERS = ("no_grad", "inference_mode")


def _no_grad_blocks(tree: ast.AST) -> list[ast.With]:
    """``with`` statements whose context expression mentions ``no_grad``/``inference_mode``."""
    blocks: list[ast.With] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            if any(marker in ast.unparse(item.context_expr) for marker in NO_GRAD_MARKERS):
                blocks.append(node)
                break
    return blocks


def test_no_projection_call_is_wrapped_in_no_grad() -> None:
    """Regression guard for a real bug: ``project`` used to be evaluated inside a
    ``no_grad`` scope (any eval loop) and silently marked every example
    "unusable gradient", disabling the method without failing."""
    offenders: list[str] = []
    for py in sorted(PKG.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for block in _no_grad_blocks(tree):
            for node in ast.walk(block):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name in DIFFERENTIABLE_CALLS:
                    rel = py.relative_to(REPO).as_posix()
                    offenders.append(
                        f"{rel}:{node.lineno} calls {name}() inside a no_grad/inference_mode block"
                    )
    assert not offenders, (
        "the projection path must stay differentiable; move the call out of no_grad or "
        "re-enable autograd explicitly (torch.enable_grad()):\n" + "\n".join(offenders)
    )


def test_public_project_reenables_autograd() -> None:
    """``riskproj.projection.operator.project`` must re-enable autograd for its body,
    so it works even when the caller is inside ``torch.no_grad()``."""
    path = PKG / "projection" / "operator.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    public = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "project"]
    assert public, "projection/operator.py no longer defines a module-level project()"
    calls = {
        n.func.attr
        for n in ast.walk(public[0])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "enable_grad" in calls, (
        "public project() must call torch.enable_grad() and delegate to an implementation; "
        "without it a no_grad caller gets 'unusable gradient' for every example"
    )


# --------------------------------------------------------------------------
# registries
# --------------------------------------------------------------------------


def test_registries_are_populated_after_import() -> None:
    for name in _iter_modules():
        importlib.import_module(name)
    from riskproj.utils import registry

    summary = registry.summary()
    assert summary, "no implementations registered at all"
    assert all(names for names in summary.values()), f"an empty registration bucket: {summary}"
    for kind, name in (("task", "toy"), ("policy", "toy")):
        assert name in summary.get(kind, []), f"{kind}:{name} is not registered; got {summary}"
    for kind, names in summary.items():
        assert len(names) == len(set(names)), f"duplicate registration under {kind}: {names}"


# --------------------------------------------------------------------------
# sites
# --------------------------------------------------------------------------


def test_site_catalog_matches_the_default_chain_and_the_model_config() -> None:
    from riskproj.policies.sites import SITE_CATALOG, default_site_chain
    from riskproj.utils.config import SITE_IDS, ModelCfg, load_config_raw

    assert set(SITE_IDS) == set(SITE_CATALOG), (
        "SITE_IDS in utils.config and SITE_CATALOG in policies.sites disagree; "
        f"only-in-config={sorted(set(SITE_IDS) - set(SITE_CATALOG))}, "
        f"only-in-catalog={sorted(set(SITE_CATALOG) - set(SITE_IDS))}"
    )
    chain = default_site_chain()
    assert sorted(chain) == sorted(SITE_IDS), (
        "the depth sweep must visit every site exactly once (ablation A9): " f"{chain}"
    )
    # configs/model/*.yaml are *fragments* of an ExperimentCfg (they carry no `name`),
    # so they are validated as a section, not as a whole experiment.
    fragment = load_config_raw(REPO / "configs" / "model" / "smolvla.yaml")
    model = ModelCfg.model_validate(fragment["model"])
    assert model.site.default in SITE_CATALOG, (
        f"smolvla.yaml defaults to site {model.site.default!r}, which is not in the catalog"
    )


# --------------------------------------------------------------------------
# repository artifacts
# --------------------------------------------------------------------------


def test_gitignore_keeps_raw_artifacts_out_of_git() -> None:
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    lines = {ln.strip() for ln in text.splitlines()}
    for pattern in ("runs/", "data/*", "*.jsonl"):
        assert pattern in lines, f".gitignore lost the {pattern!r} rule"
    assert "!data/MANIFEST.json" in lines, (
        "data/MANIFEST.json must be re-included: it is the committed provenance of every "
        "dataset (skills/software_engineering.md §5)"
    )
    assert "third_party" in text, "third_party must be ignored/never edited (CLAUDE.md #7)"


def test_agents_index_points_at_files_that_exist() -> None:
    readme = (REPO / "agents" / "README.md").read_text(encoding="utf-8")
    referenced = sorted(set(re.findall(r"`([A-Za-z0-9_./-]+\.md)`", readme)))
    assert referenced, "the agents/README.md index is empty; the reading order is gone"
    missing = [p for p in referenced if not (REPO / "agents" / p).exists()]
    assert not missing, f"agents/README.md references files that do not exist: {missing}"
    # The numbered spec files are the build order (agents/01..12); losing one breaks it.
    numbered = sorted(p for p in referenced if re.fullmatch(r"\d{2}_[a-z0-9_]+\.md", p))
    assert len(numbered) >= 12, f"expected the 12 numbered spec files in the index, got {numbered}"


def test_task_runner_exposes_the_documented_targets() -> None:
    """``tasks.py`` is the real entry point (no ``make`` on Windows); the Makefile is a
    shim over it, and both must offer the same targets."""
    required = {"setup", "test", "test-sim", "test-gpu", "lint", "smoke", "exp", "report"}

    tree = ast.parse((REPO / "tasks.py").read_text(encoding="utf-8"), filename="tasks.py")
    targets: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "TARGETS" in names and isinstance(node.value, ast.Dict):
            targets = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    assert required <= targets, (
        f"tasks.py is missing targets {sorted(required - targets)}; it defines {sorted(targets)}"
    )

    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    for target in sorted(required):
        pattern = re.compile(rf"^{re.escape(target)}\s*:.*$", re.MULTILINE)
        match = pattern.search(makefile)
        assert match, f"Makefile has no target {target!r}"
        assert "tasks.py" in match.group(0), (
            f"Makefile target {target!r} must delegate to tasks.py so both entry points "
            "cannot drift apart"
        )


def test_patches_directory_documents_its_policy() -> None:
    """Upstream changes go through ``patches/`` with a rationale (``03`` §5).

    Skipped (not passed) while the directory is still missing, because a missing
    policy file is exactly the kind of thing the audit note has to list.
    """
    patches = REPO / "patches"
    if not patches.is_dir():
        pytest.skip(
            "patches/ does not exist yet: 03_repo_setup.md §1 requires it (with a README "
            "explaining each patch) before any upstream change is made"
        )
    files = [p for p in patches.iterdir() if p.is_file()]
    if not files:
        pytest.skip("patches/ is empty: no patch policy to check yet")
    readmes = [p for p in files if p.suffix == ".md"]
    assert readmes, "patches/ contains files but no README explaining them"
    text = "\n".join(p.read_text(encoding="utf-8").lower() for p in readmes)
    assert "third_party" in text, (
        "the patch policy must state that third_party is never edited in place (CLAUDE.md #7)"
    )


def test_tolerances_are_positive_and_documented() -> None:
    """``tests/tolerances.py`` is the single place tolerances live; a zeroed tolerance
    would make every numerical test pass vacuously."""
    import tolerances as tol

    names = [
        n
        for n in dir(tol)
        if not n.startswith("_") and isinstance(getattr(tol, n), float)
    ]
    assert {"EXACT", "FD_REL", "MATCH", "STATS", "FLOW", "SIM_STATE", "DYN_READBACK"} <= set(names)
    for name in names:
        value = getattr(tol, name)
        assert value > 0.0, f"tolerance {name} must be a positive float; got {value}"
