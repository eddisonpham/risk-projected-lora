#!/usr/bin/env python
"""Discover the task-critical geoms from nominal expert rollouts.

Task-critical geoms are **discovered, not assumed** (``agents/04_env_dynamics_data.md``
§2): the dynamics knobs are applied to exactly these geoms/bodies, so a wrong list
silently perturbs nothing and invalidates every robustness number. This script runs
the nominal scripted expert, ranks contact pairs by how often and how hard they are
loaded, and writes the schema ``envs/dynamics.py::load_task_geoms`` expects.

    python scripts/inspect_contacts.py --task peg-insert-side-v3
    python scripts/inspect_contacts.py --task peg-insert-side-v3 --episodes 5 --out configs/env/task_geoms.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_OUT = ROOT / "configs" / "env" / "task_geoms.yaml"


def _contact_names(model: Any, data: Any) -> list[tuple[str, str, float, float]]:
    """``(geom1, geom2, force_sum, penetration)`` for every active contact."""
    import mujoco
    import numpy as np

    out: list[tuple[str, str, float, float]] = []
    force = np.zeros(6, dtype=np.float64)
    for i in range(int(data.ncon)):
        contact = data.contact[i]
        mujoco.mj_contactForce(model, data, i, force)
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or f"geom{contact.geom1}"
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or f"geom{contact.geom2}"
        out.append((g1, g2, float(np.abs(force[:3]).sum()), float(contact.dist)))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inspect_contacts.py", description=__doc__)
    parser.add_argument("--task", default="peg-insert-side-v3")
    parser.add_argument("--episodes", type=int, default=3, help="nominal expert episodes to aggregate")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--top", type=int, default=8, help="how many geoms to mark task-critical")
    parser.add_argument(
        "--geoms",
        default=None,
        help="comma-separated explicit selection (overrides --top); use for the documented "
        "curation after reading the ranking",
    )
    parser.add_argument(
        "--bodies",
        default=None,
        help="comma-separated manipulated bodies for the mass axis (override the guess)",
    )
    parser.add_argument(
        "--selection-note",
        default="",
        help="why this selection was chosen; recorded in the file's provenance",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the ranking, write nothing")
    ns = parser.parse_args(argv)

    try:
        import mujoco  # noqa: F401
        import metaworld  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"simulator stack missing ({exc.name}); run: python tasks.py bootstrap-sim"
        ) from exc

    from riskproj.envs import MetaWorldTask
    from riskproj.types import DynamicsSpec

    task = MetaWorldTask(task=ns.task)
    env = task.make_env(DynamicsSpec(), seed=0)
    expert = task.expert()

    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "force_sum": 0.0, "force_max": 0.0, "pen_max": 0.0}
    )
    pair_count: dict[tuple[str, str], int] = defaultdict(int)
    for ep in range(ns.episodes):
        obs, _info = env.reset(seed=ep)
        unwrapped = getattr(env, "unwrapped", env)
        model, data = unwrapped.model, unwrapped.data
        for _ in range(ns.max_steps):
            action = expert.act(obs)
            obs, _r, term, trunc, _info = env.step(action)
            for g1, g2, force, dist in _contact_names(model, data):
                for geom in (g1, g2):
                    s = stats[geom]
                    s["count"] += 1
                    s["force_sum"] += force
                    s["force_max"] = max(s["force_max"], force)
                    s["pen_max"] = max(s["pen_max"], max(0.0, -dist))
                pair_count[(g1, g2)] += 1
            if term or trunc:
                break

    if not stats:
        raise SystemExit(
            "no contacts were recorded: either the expert never made contact (check its class "
            "resolution) or the task is not contact-rich enough for this study (G1 guards this)"
        )

    ranked = sorted(
        stats.items(), key=lambda kv: (-kv[1]["count"], -kv[1]["force_max"], kv[0])
    )
    print(f"{'geom':<40} {'contacts':>9} {'mean force':>11} {'max force':>10} {'max pen':>9}")
    for geom, s in ranked:
        mean = s["force_sum"] / max(s["count"], 1.0)
        print(f"{geom:<40} {int(s['count']):>9} {mean:>11.3f} {s['force_max']:>10.3f} {s['pen_max']:>9.4f}")

    # The manipulated body is the one whose mass the payload axis scales: it is the
    # body of the geom that moves most relative to the world across the episode.
    bodies = _split_list(ns.bodies) or _moved_bodies(env, task)
    chosen = _split_list(ns.geoms) or [g for g, _ in ranked[: ns.top]]
    unknown = [g for g in chosen if g not in stats]
    if unknown:
        raise SystemExit(
            f"--geoms names {unknown}, which never touched anything in {ns.episodes} nominal "
            "episodes; scaling friction on a non-contacting geom perturbs nothing (04 §2)"
        )
    object_geoms = [g for g in chosen if _geom_body(model, g) in set(bodies)]
    gripper_geoms = [
        g for g in chosen if any(k in g for k in ("gripper", "pad", "hand", "claw"))
    ]
    groups = {
        "object": object_geoms,
        "gripper": gripper_geoms,
        "fixture": [g for g in chosen if g not in object_geoms and g not in gripper_geoms],
    }
    payload = {
        "task": ns.task,
        "geoms": chosen,
        "groups": {k: sorted(set(v)) for k, v in groups.items() if v},
        "bodies": bodies,
        "provenance": {
            "generator": "scripts/inspect_contacts.py",
            "command": " ".join(sys.argv),
            "selection": "explicit --geoms/--bodies" if (ns.geoms or ns.bodies) else "auto (top-N)",
            "selection_note": ns.selection_note,
            "episodes": ns.episodes,
            "episodes_nominal_only": True,
            "contact_actors": {
                name: {
                    "contacts": int(s["count"]),
                    "force_max": round(s["force_max"], 4),
                    "body": _geom_body(model, name),
                }
                for name, s in ranked
            },
            "ranking": "contact count, then max force",
            "stats": {g: {k: round(v, 6) for k, v in s.items()} for g, s in ranked},
            "top_pairs": [
                {"pair": list(k), "count": v}
                for k, v in sorted(pair_count.items(), key=lambda kv: -kv[1])[:10]
            ],
        },
    }
    print(f"\n[inspect-contacts] selected {len(chosen)} task-critical geoms: {chosen}")
    print(f"[inspect-contacts] manipulated bodies: {bodies or '<none found>'}")
    if not bodies:
        print(
            "[inspect-contacts] WARNING no manipulated body was identified; the mass axis will "
            "have nothing to scale. Set 'bodies' by hand from the model tree before G1."
        )
    if ns.dry_run:
        print(json.dumps(payload, indent=2, default=str))
        return 0

    out = Path(ns.out)
    import yaml

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    print(f"[inspect-contacts] wrote {out}")
    return 0


def _split_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _geom_body(model: Any, token: str) -> str:
    """Body name (or ``body<id>``) of a geom token, for the curation record."""
    import mujoco

    if token.startswith("geom") and token[4:].isdigit():
        gid = int(token[4:])
        if gid >= int(getattr(model, "ngeom", 0)):
            return "?"
    else:
        gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, token))
        if gid < 0:
            return "?"
    bid = int(model.geom_bodyid[gid])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"


def _moved_bodies(env: Any, task: Any) -> list[str]:
    """Bodies whose world position changed most during the expert episode.

    This is how the manipulated body is *found* rather than named: the payload-mass
    axis scales exactly this body's mass and inertia (``04`` §2).
    """
    import mujoco
    import numpy as np

    unwrapped = getattr(env, "unwrapped", env)
    model, data = unwrapped.model, unwrapped.data
    before = np.array(data.xpos, dtype=np.float64)
    obs, _info = env.reset(seed=0)
    expert = task.expert()
    for _ in range(60):
        obs, _r, term, trunc, _info = env.step(expert.act(obs))
        if term or trunc:
            break
    after = np.array(data.xpos, dtype=np.float64)
    if before.shape != after.shape:
        return []
    delta = np.linalg.norm(after - before, axis=1)
    order = np.argsort(-delta)[:3]
    names = []
    for idx in order:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(idx))
        if not name or name in ("world", "mocap"):
            continue
        if delta[idx] <= 1e-4:
            continue
        # A zero-mass body cannot carry a payload: scaling 0 by any factor is 0, so it
        # would make the mass axis a silent no-op (04 §2 read-back would fail anyway).
        if float(model.body_mass[idx]) <= 0.0:
            continue
        names.append(name)
    return names


if __name__ == "__main__":
    raise SystemExit(main())
