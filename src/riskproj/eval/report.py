"""Derived tables and figures — always recomputed from archived logs.

``skills/software_engineering.md`` §1.4: raw logs are immutable and reports are
derived. This module therefore never imports a live model, never re-runs an
episode, and refuses to invent an empty table when a raw log is missing — an empty
table looks like a null result, which is a much worse failure than a crash.

Every table carries the claim-ladder rung (``12``) it can support, and the report
header repeats the allowed/forbidden language so a reader cannot quote it as
something it is not.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from riskproj.data.splits import ROLE_TEST_COMBO, ROLE_TEST_RANGE
from riskproj.eval.anti_gaming import (
    fallback_conditioned_success,
    suppression_report,
    success_violation_pareto,
)
from riskproj.eval.diagnostics import (
    decision_frame,
    depth_sweep_table,
    linearization_error_table,
    per_cell_heatmap,
    split_of_cell,
)
from riskproj.types import VIOLATION_TYPES
from riskproj.utils.io import read_json, read_jsonl, sha256_file, write_json
from riskproj.utils.metrics import auroc, ece, reliability_curve, wilson_interval

__all__ = [
    "DEFAULT_MPCD",
    "build_tables",
    "claim_ladder_rung",
    "load_run",
    "write_report",
]

#: Fallback MPCD until the pilot fixes it (``09`` §2, ``01`` §5). It is *not* an
#: equivalence margin: with this value a difference inside +-5 pp is simply not
#: claimed, which is the conservative reading.
DEFAULT_MPCD = 0.05

CLAIM_LANGUAGE_ALLOWED = ("we observed", "is consistent with", "suggests", "in this simulator/task")
CLAIM_LANGUAGE_FORBIDDEN = (
    "proves",
    "guarantees",
    "safe",
    "constraint satisfied",
    "preserves the manifold",
    "works on real robots",
)


# --------------------------------------------------------------------------
# optional dependency on eval.stats (written in parallel)
# --------------------------------------------------------------------------


def mpcd_default() -> float:
    """``eval.stats.MPCD_DEFAULT`` when available, else :data:`DEFAULT_MPCD`."""
    try:
        from riskproj.eval.stats import MPCD_DEFAULT  # type: ignore[attr-defined]

        return float(MPCD_DEFAULT)
    except Exception:
        return DEFAULT_MPCD


def _stats_module() -> Any | None:
    """``eval.stats`` when it is importable; the primitives below cover the rest."""
    try:
        import riskproj.eval.stats as stats

        return stats
    except Exception:
        return None


def _wilson(successes: int, n: int) -> tuple[float, float]:
    stats = _stats_module()
    fn = getattr(stats, "wilson", None) if stats is not None else None
    if callable(fn):
        out = fn(successes, n)
        if isinstance(out, Mapping):
            return float(out["lo"]), float(out["hi"])
        lo, hi = out  # type: ignore[misc]
        return float(lo), float(hi)
    return wilson_interval(successes, n)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Read every raw log under ``run_dir``; raise if there are no episodes."""
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"run directory {root} does not exist")
    episodes = sorted(root.glob("**/episodes.jsonl"))
    if not episodes:
        raise FileNotFoundError(
            f"no episodes.jsonl under {root}: a report cannot be derived from logs that "
            "do not exist (skills/software_engineering.md §1.4)"
        )
    episode_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    per_file: dict[str, Any] = {}
    for path in episodes:
        rows = list(read_jsonl(path))
        episode_rows.extend(rows)
        decisions = path.parent / "decisions.jsonl"
        n_dec = 0
        if decisions.exists():
            d_rows = list(read_jsonl(decisions))
            decision_rows.extend(d_rows)
            n_dec = len(d_rows)
        per_file[str(path.relative_to(root))] = {
            "n_episodes": len(rows),
            "n_decisions": n_dec,
            "sha256": sha256_file(path),
        }
    config_path = root / "config.yaml"
    config = None
    if config_path.exists():
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    provenance = {
        name: (root / name).read_text(encoding="utf-8").strip()
        for name in ("git_sha.txt", "data_manifest_hash.txt")
        if (root / name).exists()
    }
    counterfactual: list[dict[str, Any]] = []
    for cf in sorted(root.glob("**/counterfactual_*.jsonl")):
        counterfactual.extend(read_jsonl(cf))
    return {
        "run_dir": str(root),
        "episodes": episode_rows,
        "decisions": decision_rows,
        "counterfactual": counterfactual,
        "sources": per_file,
        "config": config,
        "provenance": provenance,
        "n_episodes": len(episode_rows),
        "n_decisions": len(decision_rows),
    }


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def _rate(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    n = len(records)
    if key == "success":
        k = sum(1 for r in records if r.get("success"))
    elif key == "violation":
        k = sum(
            1 for r in records if any(float(v["count"]) > 0 for v in (r.get("violations") or {}).values())
        )
    elif key == "fallback":
        k = sum(1 for r in records if sum(int(v) for v in (r.get("fallback") or {}).values()) > 0)
    else:
        raise ValueError(f"unknown rate {key!r}")
    lo, hi = _wilson(k, n)
    return {"n": float(n), "k": float(k), "rate": (k / n) if n else float("nan"), "lo": lo, "hi": hi}


def _by_method_split(episodes: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    out: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for row in episodes:
        method = str(row.get("method", "unknown"))
        split = str(row.get("split") or split_of_cell(str(row.get("cell_id", ""))))
        out.setdefault(method, {}).setdefault(split, []).append(row)
    return out


def _t2_main_table(episodes: Sequence[Mapping[str, Any]], *, mpcd: float) -> dict[str, Any]:
    table: dict[str, Any] = {}
    n_seeds: dict[str, int] = {}
    for method, splits in sorted(_by_method_split(episodes).items()):
        n_seeds[method] = len(
            {str(r.get("train_seed", "0")) for rows in splits.values() for r in rows}
        )
        table[method] = {}
        for split, rows in sorted(splits.items()):
            success = _rate(rows, "success")
            violation = _rate(rows, "violation")
            table[method][split] = {
                "success": success,
                "violation": violation,
                "per_type_violation": {
                    v: float(np.mean([float(r["violations"][v]["count"]) > 0 for r in rows]))
                    for v in VIOLATION_TYPES
                },
                "fallback": _rate(rows, "fallback"),
                "mean_steps": float(np.mean([float(r.get("steps", 0)) for r in rows])),
                "mpcd": mpcd,
            }
    return {
        "table": table,
        "n_train_seeds": n_seeds,
        "pooled_splits": False,
        "rung": "L0",
        "guard": (
            "success must be read next to violations and fallback rate; TEST-combo and "
            "TEST-range are never pooled (09 §2, §5)"
        ),
    }


def _t3_factorial(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """T3: the train x inference factorial (addition 4 of ``01`` §6)."""
    cells: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    missing = 0
    for row in episodes:
        extra = row.get("extra") or {}
        train = extra.get("train_mode")
        inference = extra.get("inference_mode")
        if train is None and inference is None:
            missing += 1
            continue
        key = f"T={train}|I={inference}"
        cells.setdefault(key, {}).setdefault(str(row.get("split", "unknown")), []).append(row)
    if not cells:
        return {
            "available": False,
            "reason": (
                "no episode record carries extra['train_mode']/extra['inference_mode']; the "
                "factorial (A7) needs the method's (train, inference) modes recorded at run "
                "time, not inferred here"
            ),
            "n_records_without_modes": missing,
        }
    return {
        "available": True,
        "table": {
            key: {split: _rate(rows, "success") for split, rows in splits.items()}
            for key, splits in sorted(cells.items())
        },
        "rung": "L0",
    }


def _t4_controls(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_method = {str(r.get("method")): r for r in episodes}
    controls = sorted(m for m in by_method if m.startswith("C"))
    if not controls:
        return {
            "available": False,
            "reason": "no negative-control arm (C1/C2/C3) present in the logs (08 §2)",
        }
    return {
        "available": True,
        "table": {
            method: _rate([r for r in episodes if str(r.get("method")) == method], "success")
            for method in [*controls, "B3"]
            if method in by_method
        },
        "guard": "if a control matches B3, the effect is not critic information (A1-A3)",
        "rung": "L0",
    }


def _t5_systems(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_method: dict[str, dict[str, float]] = {}
    for row in episodes:
        method = str(row.get("method", "unknown"))
        latency = row.get("latency_ms") or {}
        bucket = per_method.setdefault(method, {})
        for phase, value in latency.items():
            acc = bucket.setdefault(f"latency_{phase}", 0.0)
            bucket[f"latency_{phase}"] = acc + float(value)
        bucket["peak_vram_mb"] = max(bucket.get("peak_vram_mb", 0.0), float(row.get("peak_vram_mb", 0.0)))
    return {
        "table": per_method,
        "units": {"latency_*": "mean ms per decision (summed over episodes here)", "peak_vram_mb": "max"},
        "guard": "overhead is reported in every results table (09 §4 T5)",
        "rung": "L0",
    }


def _t1_critic(decisions: Sequence[Mapping[str, Any]], *, tau: float) -> dict[str, Any]:
    labelled = [d for d in decisions if d.get("true_violation_next_H") is not None]
    if not labelled:
        return {
            "available": False,
            "reason": (
                "no decision record carries the simulated label true_violation_next_H, so the "
                "critic cannot be scored here; labels come from executing the chunk open-loop "
                "(04 §4, 06 §1) and must be attached by the run that saved the states"
            ),
        }
    scores = np.asarray([float(d.get("r_cand", np.nan)) for d in labelled], dtype=float)
    labels = np.asarray([1.0 if d["true_violation_next_H"] else 0.0 for d in labelled], dtype=float)
    keep = np.isfinite(scores)
    scores, labels = scores[keep], labels[keep]
    if labels.size == 0 or labels.min() == labels.max():
        return {
            "available": False,
            "reason": f"labels are degenerate (n={labels.size}, positives={float(labels.sum())})",
        }
    return {
        "available": True,
        "n": int(labels.size),
        "base_rate": float(labels.mean()),
        "auroc": auroc(labels, scores),
        "ece": ece(labels, scores),
        "fnr_at_tau": float(np.mean(scores[labels == 1] < tau)) if (labels == 1).any() else float("nan"),
        "note": "critic metrics are diagnostics; the simulator decides (01 §5, 06 §7)",
        "rung": "L0",
    }


def _f8_reliability(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labelled = [d for d in decisions if d.get("true_violation_next_H") is not None]
    if not labelled:
        return {"available": False, "reason": "no labelled decisions"}
    try:
        import riskproj.eval.critic_eval as critic_eval  # noqa: PLC0415

        fn = getattr(critic_eval, "reliability_diagram_data", None)
        if callable(fn):
            return {"available": True, "source": "critic_eval", **fn(decisions)}
    except Exception:  # pragma: no cover - fall back to the local primitive
        pass
    scores = np.asarray([float(d.get("r_cand", np.nan)) for d in labelled], dtype=float)
    labels = np.asarray([1.0 if d["true_violation_next_H"] else 0.0 for d in labelled], dtype=float)
    keep = np.isfinite(scores)
    return {"available": True, **reliability_curve(labels[keep], scores[keep])}


def _f2_counterfactual(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "available": False,
            "reason": "no counterfactual_*.jsonl in the run directory (E1 not run)",
        }
    original = np.asarray([1.0 if r.get("any_original") else 0.0 for r in rows])
    corrected = np.asarray([1.0 if r.get("any_corrected") else 0.0 for r in rows])
    b = int(np.sum((original == 1) & (corrected == 0)))
    c = int(np.sum((original == 0) & (corrected == 1)))
    return {
        "available": True,
        "n_pairs": int(len(rows)),
        "rate_original": float(original.mean()),
        "rate_corrected": float(corrected.mean()),
        "paired_delta": float((original - corrected).mean()),
        "discordant": {"improved": b, "worsened": c},
        "rung": "L0",
    }


def _f7_failure_stacks(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, dict[str, int]] = {}
    for row in episodes:
        method = str(row.get("method", "unknown"))
        cls = str(row.get("failure_class", "none"))
        out.setdefault(method, {})
        out[method][cls] = out[method].get(cls, 0) + 1
    return {"table": out, "taxonomy": "FACT-style failure classes (04 §4, 09 §1)"}


def claim_ladder_rung(tables: Mapping[str, Any]) -> str:
    """Lowest rung the evidence supports (``agents/12``).

    Deliberately conservative and mechanical: each rung requires the *artefact* that
    would justify it to be present in the tables, so a claim cannot be promoted by
    writing a confident sentence. Rungs above L3 need a second task and a second
    model class (ablations A23), which table evidence alone cannot establish.
    """
    main = tables.get("T2_main") or {}
    table = main.get("table") or {}
    if not table:
        return "L0"

    seeds = {str(k): int(v) for k, v in (main.get("n_train_seeds") or {}).items()}
    has_three_seeds = bool(seeds) and min(seeds.values()) >= 3
    splits = {split for methods in table.values() for split in methods}
    has_sealed_splits = bool(splits & {ROLE_TEST_COMBO, ROLE_TEST_RANGE})
    controls = tables.get("T4_controls") or {}
    has_controls = bool(controls.get("available"))

    if not has_three_seeds:
        return "L0"  # observed in this run/config only
    if not has_controls:
        return "L1"  # replicated across seeds, no negative controls yet
    if not has_sealed_splits:
        return "L2"  # survives controls, but only on VAL cells
    return "L3"  # survives controls on held-out cells; L4 needs another task/model


def build_tables(run_dir: str | Path, *, mpcd: float | None = None) -> dict[str, Any]:
    """Build every derived table/figure dataset from the archived logs."""
    margin = mpcd_default() if mpcd is None else float(mpcd)
    run = load_run(run_dir)
    episodes = run["episodes"]
    decisions = run["decisions"]
    tables: dict[str, Any] = {
        "run_dir": run["run_dir"],
        "n_episodes": run["n_episodes"],
        "n_decisions": run["n_decisions"],
        "mpcd": margin,
        "sources": run["sources"],
        "provenance": run["provenance"],
        "T1_critic": _t1_critic(decisions, tau=_config_tau(run)),
        "T2_main": _t2_main_table(episodes, mpcd=margin),
        "T3_factorial": _t3_factorial(episodes),
        "T4_controls": _t4_controls(episodes),
        "T5_systems": _t5_systems(episodes),
        "F1_linearization": linearization_error_table(decisions),
        "F2_counterfactual": _f2_counterfactual(run["counterfactual"]),
        "F4_heatmap": per_cell_heatmap(episodes, metric="success"),
        "F5_depth_sweep": depth_sweep_table(episodes) if _has_site(episodes) else {
            "available": False,
            "reason": "no episode carries extra['site'] (ablation A9 not run)",
        },
        "F6_pareto": success_violation_pareto(episodes),
        "F7_failure_stacks": _f7_failure_stacks(episodes),
        "F8_reliability": _f8_reliability(decisions),
        "anti_gaming": {
            "fallback_conditioned_success": fallback_conditioned_success(episodes),
            "suppression": suppression_report(episodes),
        },
        "proj_diagnostics": decision_frame(decisions),
        "_episodes": episodes,
    }
    tables["claim_rung"] = claim_ladder_rung(tables)
    return tables


def _config_tau(run: Mapping[str, Any]) -> float:
    """Projection tau from the run's recorded config, or the default when absent."""
    config = run.get("config") or {}
    if isinstance(config, Mapping):
        methods = config.get("methods") or config.get("method") or []
        if isinstance(methods, Mapping):
            methods = [methods]
        for entry in methods if isinstance(methods, Sequence) else []:
            if isinstance(entry, Mapping):
                projection = entry.get("projection") or {}
                if isinstance(projection, Mapping) and "tau" in projection:
                    return float(projection["tau"])
    return 0.5


def _has_site(episodes: Sequence[Mapping[str, Any]]) -> bool:
    return any((r.get("extra") or {}).get("site") for r in episodes)


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return "nan" if np.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def _md_table(table: Mapping[str, Mapping[str, Any]], *, value_key: str) -> str:
    methods = sorted(table)
    splits = sorted({split for m in table.values() for split in m})
    if not methods or not splits:
        return "_(no rows)_"
    lines = ["| method | " + " | ".join(splits) + " |", "|" + "---|" * (len(splits) + 1)]
    for method in methods:
        cells = []
        for split in splits:
            entry = table[method].get(split)
            if entry is None:
                cells.append("-")
            elif value_key in entry and isinstance(entry[value_key], Mapping):
                cells.append(f"{_fmt(entry[value_key]['rate'])} [{entry[value_key]['lo']:.2f},{entry[value_key]['hi']:.2f}]")
            elif value_key in entry:
                cells.append(_fmt(entry[value_key]))
            else:
                cells.append("-")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    run_dir: str | Path,
    *,
    out_format: str = "md",
    mpcd: float | None = None,
) -> Path:
    """Write ``report.md`` (and ``tables.json``) into the run directory."""
    if out_format != "md":
        raise ValueError(f"only 'md' is implemented; got {out_format!r}")
    root = Path(run_dir)
    tables = build_tables(root, mpcd=mpcd)
    margin = tables["mpcd"]
    main = tables["T2_main"]
    lines = [
        "# Derived report",
        "",
        f"- run directory: `{tables['run_dir']}`",
        "- **derived artifact**: every number below is recomputed from the raw logs, "
        "never from a live model; source files and their sha256 are in `tables.json`",
        f"- source logs: {json.dumps(tables['sources'], sort_keys=True)}",
        f"- episodes: {tables['n_episodes']}, decisions: {tables['n_decisions']}",
        f"- MPCD used for 'no claim' decisions: {margin}",
        f"- claim-ladder rung: **{tables['claim_rung']}**",
        "",
        "## Claim language",
        f"- allowed: {', '.join(CLAIM_LANGUAGE_ALLOWED)}",
        f"- forbidden: {', '.join(CLAIM_LANGUAGE_FORBIDDEN)}",
        "- the critic is a proxy; simulator outcomes decide (`01` §5)",
        "",
        "## T2 — success / violations per method (splits kept separate)",
        "",
        "Success rate with 95% Wilson interval:",
        "",
        _md_table(main.get("table", {}), value_key="success"),
        "",
        "Violation rate (`any_violation`):",
        "",
        _md_table(main.get("table", {}), value_key="violation"),
        "",
        "## T1 — critic (diagnostic)",
        "",
        f"```json\n{json.dumps(tables['T1_critic'], indent=2, sort_keys=True, default=str)}\n```",
        "",
        "## Anti-gaming",
        "",
        f"```json\n{json.dumps(tables['anti_gaming'], indent=2, sort_keys=True, default=str)}\n```",
        "",
        "## Projection diagnostics",
        "",
        f"```json\n{json.dumps(tables['proj_diagnostics'], indent=2, sort_keys=True, default=str)}\n```",
        "",
        "## E1 — paired counterfactual",
        "",
        f"```json\n{json.dumps(tables['F2_counterfactual'], indent=2, sort_keys=True, default=str)}\n```",
        "",
        "## Systems",
        "",
        f"```json\n{json.dumps(tables['T5_systems'], indent=2, sort_keys=True, default=str)}\n```",
        "",
        "## Not computed",
        "",
    ]
    for name in ("T3_factorial", "T4_controls", "F5_depth_sweep", "F8_reliability"):
        entry = tables.get(name) or {}
        if not entry.get("available", False):
            lines.append(f"- {name}: {entry.get('reason', 'unavailable')}")
    lines.append("")
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(root / "tables.json", {k: v for k, v in tables.items() if not k.startswith("_")})
    return root / "report.md"


def load_tables(run_dir: str | Path) -> dict[str, Any]:
    """Read back a previously written ``tables.json`` (for downstream figures)."""
    path = Path(run_dir) / "tables.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run write_report() first")
    return read_json(path)
