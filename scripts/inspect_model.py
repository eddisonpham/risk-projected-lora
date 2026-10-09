#!/usr/bin/env python
"""Print a policy's module tree and write the resolved projection sites (smoke S3).

SmolVLA's module names are tagged [R] in the spec until this script walks the real
module tree, so this is the single writer of ``configs/model/smolvla_sites.yaml``
(``agents/05_models_and_layers.md`` §2-§3). It never invents a module name: a site
that cannot be resolved is reported with the candidates it saw, and the process
exits non-zero.

    python scripts/inspect_model.py --model smolvla
    python scripts/inspect_model.py --model toy --out /tmp/toy_sites.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_OUT = ROOT / "configs" / "model" / "smolvla_sites.yaml"


def module_tree(model: object, *, limit: int | None = None) -> list[dict[str, object]]:
    """``named_modules`` as a printable table, with the pieces a site choice needs."""
    import torch.nn as nn

    rows: list[dict[str, object]] = []
    named = getattr(model, "named_modules", None)
    if named is None:
        raise TypeError(
            f"{type(model).__name__} is not an nn.Module; adapters must expose the underlying "
            "module tree so sites can be discovered rather than guessed"
        )
    for name, module in named():
        if not name:
            continue
        row: dict[str, object] = {
            "name": name,
            "type": type(module).__name__,
            "n_params": sum(p.numel() for p in module.parameters(recurse=False)),
        }
        for attr in ("in_features", "out_features", "normalized_shape", "weight"):
            value = getattr(module, attr, None)
            if attr == "weight" and value is not None:
                row["dtype"] = str(value.dtype)
            elif value is not None:
                row[attr] = value if isinstance(value, (int, tuple, list)) else str(value)
        rows.append(row)
    return rows[:limit] if limit else rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inspect_model.py", description=__doc__)
    parser.add_argument("--model", default="smolvla", help="registered policy adapter name")
    parser.add_argument("--checkpoint", default=None, help="HF hub id or local path override")
    parser.add_argument("--config", default=None, help="experiment config to take the model from")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where to write the sites yaml")
    parser.add_argument("--limit", type=int, default=None, help="print at most N modules")
    parser.add_argument("--json", action="store_true", help="print the tree as json")
    ns = parser.parse_args(argv)

    from riskproj.policies import available_policies, build_policy, is_available
    from riskproj.policies.base import PolicyAdapterBase
    from riskproj.policies.sites import SITE_CATALOG, describe_match, resolve_sites, write_sites_yaml
    from riskproj.utils.config import ModelCfg, load_config
    from riskproj.utils.io import git_sha

    if ns.config:
        cfg = load_config(ns.config)
        model_cfg = cfg.model
    else:
        model_cfg = ModelCfg(name=ns.model)
    if ns.checkpoint:
        model_cfg = model_cfg.model_copy(update={"checkpoint": ns.checkpoint})
    if model_cfg.name not in available_policies():
        raise SystemExit(f"unknown policy {model_cfg.name!r}; registered: {available_policies()}")
    if not is_available(model_cfg.name):
        raise SystemExit(
            f"policy {model_cfg.name!r} is registered but its dependencies are missing; "
            "run: python tasks.py bootstrap-sim"
        )

    policy = build_policy(model_cfg)
    # The tree walk must use the SAME namespace the adapter resolves sites in
    # (``PolicyAdapter.site_tree``): an adapter that wraps the model in ``self._model``
    # would otherwise write patterns with an extra ``_model.`` prefix, and every site
    # would fail to resolve at load time even though this script just "succeeded".
    tree_model = policy.site_tree() if hasattr(policy, "site_tree") else policy
    tree = module_tree(tree_model, limit=ns.limit)
    if ns.json:
        print(json.dumps(tree, indent=2, default=str))
    else:
        print(f"{'module':<60} {'type':<18} {'params':>10} extra")
        for row in tree:
            extra = " ".join(
                f"{k}={row[k]}" for k in ("in_features", "out_features", "normalized_shape", "dtype") if k in row
            )
            print(f"{row['name']:<60} {row['type']:<18} {row['n_params']:>10} {extra}")

    specs = resolve_sites(model_cfg.site, adapter=policy)
    print(f"\n[inspect-model] {len(tree)} modules; {len(specs)} sites in the catalog")

    # `describe_match` is the authoritative resolver for latent sites; a site with no
    # module is left with an empty pattern and reported, never fabricated.
    resolved: dict[str, object] = {}
    unresolved: dict[str, str] = {}
    adapter_sites = getattr(policy, "sites", None)
    registered = set(adapter_sites()) if callable(adapter_sites) else set()
    for sid, spec in specs.items():
        if not spec.is_module_site():
            resolved[sid] = spec  # S0/S0f belong to no module
            continue
        if sid in registered:
            # The adapter already resolved this site from the module tree.
            module = policy.site_module(sid)
            name = next(
                (n for n, m in tree_model.named_modules() if m is module),
                "",
            )
            from dataclasses import replace

            resolved[sid] = replace(spec, module_pattern=name, capture=spec.capture)
            print(f"  {sid:<3} -> {name}")
            continue
        info = describe_match(
            tree_model, spec, action_dim=getattr(policy, "action_dim", None) or None
        )
        if info.get("resolved"):
            from dataclasses import replace

            resolved[sid] = replace(
                spec, module_pattern=str(info["module"]), matched_by="heuristic"
            )
            print(f"  {sid:<3} -> {info['module']} (heuristic match)")
        else:
            unresolved[sid] = str(info.get("error", "unresolved"))
            print(f"  {sid:<3} -> UNRESOLVED: {unresolved[sid]}")

    if not any(getattr(s, "module_pattern", "") for s in resolved.values()):
        raise SystemExit(
            "no latent site could be resolved from the module tree. The catalog holds "
            f"{sorted(SITE_CATALOG)} and every heuristic pattern failed; print the tree above and "
            "add authoritative patterns to policies/sites.py before relying on a site."
        )

    sha = git_sha()
    out = write_sites_yaml(
        ns.out,
        resolved,  # type: ignore[arg-type]
        model_name=model_cfg.name,
        source=f"scripts/inspect_model.py --model {model_cfg.name}",
        git_sha=str(sha.get("sha", "")),
    )
    print(f"[inspect-model] wrote {out} ({len(resolved)} sites, {len(unresolved)} unresolved)")
    if unresolved:
        print(
            f"[inspect-model] WARNING unresolved sites (they will raise if used): "
            f"{json.dumps(unresolved, indent=2)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
