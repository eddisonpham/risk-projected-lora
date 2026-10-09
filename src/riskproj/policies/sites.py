"""Projection sites: the catalog, its resolver, and module discovery.

A *site* is a module output where projection acts (``05`` §3). The catalog below
is **design intent**, not verified wiring: SmolVLA's module names are tagged [R]
in the spec until ``scripts/inspect_model.py`` (smoke test S3) walks the real
module tree and writes ``configs/model/smolvla_sites.yaml``. So the catalog keeps
``module_pattern`` empty on purpose and :func:`resolve_sites` warns loudly when it
falls back to it. Nothing here ever fabricates a resolved module name.

When no authoritative pattern exists, :func:`discover_modules` falls back to the
*heuristic* substrings in :data:`CANDIDATE_PATTERNS` and records what it matched;
an unmatched site raises with the nearby module names it saw, so the failure is a
message rather than a wrong number.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from torch import nn

from riskproj.utils.config import SITE_IDS, SiteCfg

logger = logging.getLogger(__name__)

__all__ = [
    "CANDIDATE_PATTERNS",
    "SITE_CATALOG",
    "SiteSpec",
    "default_site_chain",
    "describe_match",
    "discover_candidates",
    "discover_modules",
    "load_sites_yaml",
    "register_discovered_sites",
    "resolve_sites",
    "write_sites_yaml",
]

SiteKind = Literal["action_final", "action_guidance", "expert_block", "vlm_layer"]


@dataclass
class SiteSpec:
    """One projection site.

    ``module_pattern`` is either an exact module path or an fnmatch glob over
    ``named_modules()``; it is empty until S3 writes it. ``capture`` says whether
    the projected tensor is the matched module's output or its input — S1 is the
    *input* of the final action linear, so it must be ``"input"``.
    """

    id: str
    kind: SiteKind
    downstream: str
    module_pattern: str = ""
    capture: Literal["output", "input"] = "output"
    notes: str = ""
    #: Extra provenance, e.g. which candidate pattern actually matched.
    matched_by: str = ""

    def is_module_site(self) -> bool:
        """False for the action-space sites, which act on the chunk, not a module."""
        return self.kind in ("expert_block", "vlm_layer")

    def resolved(self) -> bool:
        return bool(self.module_pattern)


def _spec(
    id: str,
    kind: SiteKind,
    downstream: str,
    *,
    capture: Literal["output", "input"] = "output",
    notes: str = "",
) -> SiteSpec:
    return SiteSpec(id=id, kind=kind, downstream=downstream, capture=capture, notes=notes)


#: S0..S6 exactly as specified in ``05`` §3. ``module_pattern`` stays empty: it is
#: written by ``scripts/inspect_model.py`` and must not be guessed.
SITE_CATALOG: dict[str, SiteSpec] = {
    "S0": _spec(
        "S0",
        "action_final",
        "identity",
        notes="final chunk, post-hoc; this IS B4a (Dalal-style action-space correction)",
    ),
    "S0f": _spec(
        "S0f",
        "action_guidance",
        "terminal estimate",
        notes="guidance applied to x_tau at the last m denoising steps; this IS B4b",
    ),
    "S1": _spec(
        "S1",
        "expert_block",
        "final action linear",
        capture="input",
        notes="last expert hidden state, i.e. the input of the action output projection",
    ),
    "S2": _spec(
        "S2",
        "expert_block",
        "rest of the last block + final action linear",
        notes="DEFAULT site: MLP down-projection output of the last expert block",
    ),
    "S3": _spec(
        "S3",
        "expert_block",
        "remaining expert blocks + final linear",
        notes=(
            "mid expert block, cross-attention output projection (reads the frozen VLM KV). "
            "Naming trap [V]: SmolVLA routes cross-attention by layer *parity* inside the "
            "same `self_attn` module, so this site resolves to `lm_expert.layers.<odd>."
            "self_attn.o_proj` — the module is called self_attn even where it cross-attends."
        ),
    ),
    "S4": _spec(
        "S4",
        "expert_block",
        "remaining expert blocks + final linear",
        notes=(
            "mid expert block, self-attention output projection (mixes action tokens); "
            "resolves to `lm_expert.layers.<even>.self_attn.o_proj` — even layers are the "
            "ones that run self-attention (self_attn_every_n_layers=2)"
        ),
    ),
    "S5": _spec(
        "S5",
        "expert_block",
        "most of the expert",
        notes="early expert block; longer path, costlier gradient",
    ),
    "S6": _spec(
        "S6",
        "vlm_layer",
        "whole expert x num_flow_steps",
        notes=(
            "STRETCH: VLM text-layer output; one-time per chunk but the heaviest gradient. "
            "[V] the VLM text layers run only in the prefix pass (fill_kv_cache=True), never "
            "inside a denoising step, so projecting here means substituting the activation, "
            "re-running the prefix once, and then the whole flow."
        ),
    ),
}

#: Module-name patterns per site, tried in order. The **verified** patterns come
#: first (they were read out of LeRobot 0.6.0 on 2026-10-09 through
#: ``scripts/inspect_model.py``); the original guesses stay behind them so a
#: different upstream layout still resolves, and ``SiteSpec.matched_by`` records
#: which pattern actually fired.
#:
#: Verified SmolVLA facts (``05`` sections 2-3, tags upgraded from [R] to [V]):
#:
#: * the action expert is ``model.vlm_with_expert.lm_expert``, a Llama stack of
#:   ``num_vlm_layers`` = 16 blocks of width 0.75·960 = 720;
#: * ``self_attn_every_n_layers = 2`` splits them by **parity**: even ``layer_idx``
#:   runs self-attention over the action tokens, **odd** layers run cross-attention
#:   against the frozen VLM key-values (their ``k_proj``/``v_proj`` are re-created to
#:   take the VLM's kv width — 320 — which is how the tree makes the split visible);
#: * so S3 (cross-attn output proj) and S4 (self-attn output proj) are the *same*
#:   module path with **opposite parity**, and the parity must be in the pattern;
#: * ``action_out_proj`` maps the expert width 720 to ``max_action_dim`` 32, and the
#:   action chunk is stored padded to 32 — not to the task's 4 (`05` §2);
#: * the VLM text layers run **only in the prefix pass** (``fill_kv_cache=True``),
#:   never during a denoising step, so S6 is a prefix-pass site.
CANDIDATE_PATTERNS: dict[str, tuple[str, ...]] = {
    "S1": (
        "*model.action_out_proj",
        "action_out",
        "action_output",
        "action_proj",
        "action_head",
        "final_action",
    ),
    "S2": (
        "*lm_expert.layers.*.mlp.down_proj",  # tie-break: highest block index
        "down_proj",
        "mlp.down",
        "ffn.down",
        "mlp_up",
    ),
    "S3": (
        "*lm_expert.layers.[13579].self_attn.o_proj",
        "*lm_expert.layers.1[135].self_attn.o_proj",
        "cross_attn",
        "cross_attention",
        "crossattention",
    ),
    "S4": (
        "*lm_expert.layers.[02468].self_attn.o_proj",
        "*lm_expert.layers.1[02468].self_attn.o_proj",
        "self_attn",
        "self_attention",
        "selfattention",
    ),
    "S5": (
        "*lm_expert.layers.*.mlp.down_proj",  # tie-break: lowest block index
        "layers.0.",
        "blocks.0.",
        "layer_0",
        "block_0",
    ),
    "S6": (
        "*vlm.model.text_model.layers.*.mlp.down_proj",
        "text_model",
        "language_model",
        "text_layers",
        ".vlm.",
        "vlm.",
    ),
}

#: Depth-sweep order used by ablation A9 (`10` Wave 2).
_DEPTH_SWEEP: tuple[str, ...] = ("S0", "S0f", "S1", "S2", "S3", "S4", "S5", "S6")


def default_site_chain() -> list[str]:
    """A9's order: action space first, then progressively earlier latent sites."""
    return list(_DEPTH_SWEEP)


# --------------------------------------------------------------------------
# yaml io (single writer, single reader)
# --------------------------------------------------------------------------


def write_sites_yaml(
    path: str | Path,
    resolved: dict[str, SiteSpec],
    *,
    model_name: str,
    source: str,
    git_sha: str,
) -> Path:
    """Write the resolved sites file produced by ``scripts/inspect_model.py`` (S3)."""
    unknown = sorted(set(resolved) - set(SITE_IDS))
    if unknown:
        raise ValueError(f"unknown site ids {unknown}; SITE_IDS are {SITE_IDS}")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "model_name": model_name,
        "source": source,
        "git_sha": git_sha,
        "unverified_patterns": sorted(s.id for s in resolved.values() if not s.resolved()),
        "sites": {sid: asdict(spec) for sid, spec in resolved.items()},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return p


def load_sites_yaml(path: str | Path) -> dict[str, SiteSpec]:
    """Read a sites file; raises on malformed content rather than half-loading it."""
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or "sites" not in data:
        raise ValueError(f"{p}: expected a mapping with a 'sites' key")
    out: dict[str, SiteSpec] = {}
    for sid, entry in data["sites"].items():
        if sid not in SITE_IDS:
            raise ValueError(f"{p}: unknown site id {sid!r}")
        if not isinstance(entry, dict):
            raise TypeError(f"{p}: site {sid!r} must be a mapping")
        out[sid] = SiteSpec(
            id=sid,
            kind=entry["kind"],
            downstream=entry["downstream"],
            module_pattern=entry.get("module_pattern", ""),
            capture=entry.get("capture", "output"),
            notes=entry.get("notes", ""),
            matched_by=entry.get("matched_by", ""),
        )
    return out


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


def resolve_sites(cfg: SiteCfg, adapter: Any = None) -> dict[str, SiteSpec]:
    """Resolve the site catalog for this run.

    Order of authority: the file written by S3 > the design catalog. The catalog
    fallback emits a warning because those module patterns are unverified. When an
    ``adapter`` is given, latent sites the adapter did not register are dropped
    (action-space sites S0/S0f are always kept — they belong to no module).
    """
    path = Path(cfg.resolved_file)
    if path.exists():
        resolved = load_sites_yaml(path)
        # Catalog supplies any site the file omits; the file always wins.
        for sid, spec in SITE_CATALOG.items():
            resolved.setdefault(sid, spec)
        logger.info("resolved %d sites from %s", len(resolved), path)
    else:
        resolved = {sid: SiteSpec(**asdict(spec)) for sid, spec in SITE_CATALOG.items()}
        logger.warning(
            "site module patterns are UNVERIFIED [R]: %s does not exist. Run "
            "`python scripts/inspect_model.py` (smoke test S3) to write it, or the "
            "adapter will fall back to heuristic module-name matching.",
            path,
        )
    if adapter is not None:
        registered = set(adapter.sites())
        kept: dict[str, SiteSpec] = {}
        for sid, spec in resolved.items():
            if not spec.is_module_site() or sid in registered:
                kept[sid] = spec
        if not kept:
            raise ValueError("no sites survived resolution; the adapter registered none")
        resolved = kept
    missing = [sid for sid in cfg.catalog if sid not in resolved]
    if missing:
        logger.debug("sites in the config catalog but not resolved: %s", missing)
    return resolved


# --------------------------------------------------------------------------
# module discovery
# --------------------------------------------------------------------------


def _iter_module_names(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(name, module) for name, module in model.named_modules() if name]


def _matches(name: str, patterns: tuple[str, ...]) -> str:
    """Return the first pattern that matches ``name`` (substring or fnmatch glob)."""
    low = name.lower()
    for pattern in patterns:
        if not pattern:
            continue
        p = pattern.lower()
        if any(ch in p for ch in "*?[") and fnmatch.fnmatch(low, p):
            return pattern
        if p in low:
            return pattern
    return ""


def _pattern_family(
    model: nn.Module, patterns: tuple[str, ...]
) -> tuple[list[tuple[str, str]], str]:
    """All ``(name, pattern)`` matches of the **first** pattern that matches anything.

    Priority, not union: a site's first pattern is its verified family (``05`` §3),
    and mixing families is how the tie-break rules went wrong. Concretely, S2's family
    is "expert MLP down-projections", and picking the highest block index within it
    gives the last expert block. Under union semantics the same rule also saw the VLM
    text stack (also 16 blocks, also ``mlp.down_proj``) and a plain alphabetical
    tie-break put S2 in the vision-language stack — silently, and with a name that
    looks plausible.
    """
    for pattern in patterns:
        if not pattern:
            continue
        family = [
            (name, pattern) for name, _ in _iter_module_names(model) if _matches(name, (pattern,))
        ]
        if family:
            return family, pattern
    return [], ""


def discover_candidates(
    model: nn.Module,
    spec: SiteSpec,
    *,
    action_dim: int | None = None,
    patterns: tuple[str, ...] | None = None,
) -> list[tuple[str, nn.Module, str]]:
    """All module paths matching ``spec``; returns ``(name, module, pattern)``.

    ``action_dim`` sharpens S1: the final action projection must produce the action
    dimension, which filters out attention output projections that merely contain
    the word "action".
    """
    if not spec.is_module_site():
        raise ValueError(
            f"{spec.id} is an action-space site ({spec.kind}); it has no module. "
            "Apply it to the action chunk, not to a module output."
        )
    pats = (
        patterns
        if patterns is not None
        else ((spec.module_pattern,) if spec.module_pattern else CANDIDATE_PATTERNS[spec.id])
    )
    family, _pattern = _pattern_family(model, pats)
    by_name = dict(_iter_module_names(model))
    found: list[tuple[str, nn.Module, str]] = [
        (name, by_name[name], pattern) for name, pattern in family if name in by_name
    ]
    if found and spec.id == "S1" and action_dim is not None:
        narrowed = [
            (n, m, p)
            for n, m, p in found
            if isinstance(m, nn.Linear) and int(m.out_features) == int(action_dim)
        ]
        if narrowed:
            found = narrowed
    return sorted(found, key=lambda t: t[0])


def discover_modules(
    model: nn.Module,
    spec: SiteSpec,
    *,
    action_dim: int | None = None,
    patterns: tuple[str, ...] | None = None,
) -> tuple[str, nn.Module]:
    """Deterministically pick the module for ``spec``; raises with the nearby candidates.

    Tie-breaks are deliberate and documented, because "which of the 12 matching
    linears" is exactly the kind of silent choice that invalidates an ablation:

    * ``S1`` — prefer an ``nn.Linear`` with ``out_features == action_dim``, else the
      deepest match;
    * ``S2`` — the match in the **highest** block index (the last expert block);
    * ``S3`` / ``S4`` — the **median** block index (a mid expert block);
    * ``S5`` — the **lowest** block index (an early block);
    * ``S6`` — the highest block index among the VLM text layers.
    """
    from riskproj.policies.base import block_index  # local import: avoids a cycle at import time

    candidates = discover_candidates(model, spec, action_dim=action_dim, patterns=patterns)
    if not candidates:
        tried = (
            list(patterns)
            if patterns
            else (
                [spec.module_pattern]
                if spec.module_pattern
                else list(CANDIDATE_PATTERNS.get(spec.id, ()))
            )
        )
        every_name = [n for n, _ in _iter_module_names(model)]
        nearby = [
            n
            for n in every_name
            if any(k in n.lower() for k in ("action", "expert", "mlp", "attn"))
        ]
        raise LookupError(
            f"no module matched site {spec.id} over patterns {tried}. "
            f"Nearby candidates: {nearby[:20] or every_name[:20]}. "
            "Write an authoritative pattern with scripts/inspect_model.py (smoke test S3)."
        )

    if spec.id == "S1":
        chosen = max(candidates, key=lambda t: (len(t[0].split(".")), t[0]))
    elif spec.id == "S2":
        chosen = max(
            candidates,
            key=lambda t: (block_index(t[0]) if block_index(t[0]) is not None else -1, t[0]),
        )
    elif spec.id in ("S3", "S4"):
        indices = sorted(i for i in (block_index(n) for n, _, _ in candidates) if i is not None)
        if indices:
            median = indices[len(indices) // 2]
            chosen = max(
                (t for t in candidates if block_index(t[0]) == median),
                key=lambda t: t[0],
            )
        else:  # pragma: no cover - un-numbered module tree
            chosen = candidates[len(candidates) // 2]
    elif spec.id == "S5":
        chosen = min(
            candidates,
            key=lambda t: (block_index(t[0]) if block_index(t[0]) is not None else 10**9, t[0]),
        )
    else:  # S6
        chosen = max(
            candidates,
            key=lambda t: (block_index(t[0]) if block_index(t[0]) is not None else -1, t[0]),
        )

    name, module, pattern = chosen
    logger.info(
        "site %s -> %s (matched pattern %r, %d candidates)", spec.id, name, pattern, len(candidates)
    )
    return name, module


def register_discovered_sites(
    adapter: Any,
    model: nn.Module,
    specs: dict[str, SiteSpec],
    *,
    action_dim: int | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve and register every module site on ``adapter``; return ``(matched, errors)``.

    Shared by the real adapters so both fail the same way. A site that cannot be
    resolved is recorded in ``errors`` (and logged) instead of aborting the load, and
    ``adapter.site_module`` raises with that message when the site is actually used —
    a run that never touches S5 should not die because S5's name was unverified.
    """
    matched: dict[str, str] = {}
    errors: dict[str, str] = {}
    for sid, spec in specs.items():
        if not spec.is_module_site():
            continue  # S0/S0f act on the action chunk, not on a module
        try:
            name, module = discover_modules(model, spec, action_dim=action_dim)
        except LookupError as exc:
            errors[sid] = str(exc)
            logger.warning("site %s unresolved: %s", sid, exc)
            continue
        adapter.register_site(sid, module, override=True)
        matched[sid] = name
    return matched, errors


def describe_match(
    model: nn.Module, spec: SiteSpec, *, action_dim: int | None = None
) -> dict[str, Any]:
    """Diagnostic record for a site: chosen module, the pattern, and the alternatives."""
    try:
        name, module = discover_modules(model, spec, action_dim=action_dim)
    except (LookupError, ValueError) as exc:
        return {"site": spec.id, "resolved": False, "error": str(exc)}
    alternatives = [n for n, _, _ in discover_candidates(model, spec, action_dim=action_dim)]
    return {
        "site": spec.id,
        "resolved": True,
        "module": name,
        "module_type": type(module).__name__,
        "alternatives": alternatives,
    }
