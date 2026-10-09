"""Run directories, JSONL logs, provenance and data manifests.

Two rules from ``skills/software_engineering.md`` are enforced here: raw logs are
immutable and reports are derived from them; every run carries the
git/pip/GPU/seed/data provenance needed to reproduce it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

#: repository root: <root>/src/riskproj/utils/io.py
ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# basic io
# --------------------------------------------------------------------------


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=indent, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    tmp.replace(p)  # atomic-ish: never leave a half-written log
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:  # pragma: no cover
            pass
    raise TypeError(f"not JSON serialisable: {type(o)!r}")


def append_jsonl(path: str | Path, rows: dict[str, Any] | Iterable[dict[str, Any]]) -> Path:
    """Append one or many rows. JSONL logs are append-only (immutable history)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, dict):
        rows = [rows]
    with p.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
    return p


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Overwrite a JSONL file (used for *derived* artifacts, not raw logs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
    return p


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_dir(path: str | Path, *, pattern: str = "*") -> str:
    """Order-stable hash over a directory's matching files (name + content)."""
    root = Path(path)
    h = hashlib.sha256()
    for f in sorted(p for p in root.glob(pattern) if p.is_file()):
        h.update(f.name.encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------
# run directories
# --------------------------------------------------------------------------


def run_dir(
    out_dir: str | Path,
    run_name: str,
    *,
    method: str | None = None,
    seed: int | None = None,
    create: bool = True,
) -> Path:
    """``runs/<name>-<hash8>/[<method>/[<seed>/]]`` (``skills/software_engineering.md`` §1)."""
    p = Path(out_dir) / run_name
    if method is not None:
        p = p / method
    if seed is not None:
        p = p / str(seed)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_sha(repo: str | Path = ROOT) -> dict[str, Any]:
    """Repo HEAD + dirty flag + submodule SHAs (``03`` §6)."""

    def _git(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
            return f"<error: {exc}>"
        return out.stdout.strip() if out.returncode == 0 else f"<error: {out.stderr.strip()}>"

    subs = _git("submodule", "status")
    submods = {}
    for line in subs.splitlines():
        line = line.strip().lstrip("-+U")
        parts = line.split()
        if len(parts) >= 2:
            submods[parts[1]] = parts[0]
    return {
        "sha": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "submodules": submods,
        "captured_at": utc_stamp(),
    }


def pip_freeze() -> list[str]:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.splitlines()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        pass
    # uv-managed venvs often have no pip; fall back to importlib metadata.
    try:
        from importlib.metadata import distributions

        return sorted(
            f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata["Name"]
        )
    except Exception:  # pragma: no cover
        return []


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()
            info["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": list(torch.cuda.get_device_capability(i)),
                    "total_memory_mb": round(
                        torch.cuda.get_device_properties(i).total_memory / 2**20, 1
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:  # pragma: no cover
        info["torch"] = None
    env_keys = ("MUJOCO_GL", "CUDA_VISIBLE_DEVICES", "HF_HOME", "PYTHONHASHSEED")
    info["env"] = {k: os.environ.get(k) for k in env_keys}
    return info


def capture_provenance(
    directory: str | Path,
    *,
    config: dict[str, Any] | None = None,
    seeds: dict[str, Any] | None = None,
    data_manifest_hash: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the six provenance files every run directory must have (``03`` §6)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    if config is not None:
        write_json(d / "config.json", config)
    sha = git_sha()
    (d / "git_sha.txt").write_text(
        "\n".join(
            [f"repo {sha['sha']} ({sha['branch']}, dirty={sha['dirty']})"]
            + [f"submodule {k} {v}" for k, v in sha["submodules"].items()]
        )
        + "\n",
        encoding="utf-8",
    )
    (d / "pip_freeze.txt").write_text("\n".join(pip_freeze()) + "\n", encoding="utf-8")
    write_json(d / "gpu.txt", gpu_info())
    write_json(d / "seeds.json", seeds or {})
    if data_manifest_hash is not None:
        (d / "data_manifest_hash.txt").write_text(data_manifest_hash + "\n", encoding="utf-8")
    if extra:
        write_json(d / "run_meta.json", extra)
    return d


# --------------------------------------------------------------------------
# data manifest
# --------------------------------------------------------------------------


def add_manifest_entry(
    manifest_path: str | Path,
    rel_path: str,
    *,
    generator: str,
    generator_version: str,
    config_hash: str,
    n_records: int | None = None,
    extra: dict[str, Any] | None = None,
    base: str | Path | None = None,
) -> dict[str, Any]:
    """Register a dataset artifact with its sha256 (``skills/software_engineering.md`` §5)."""
    mp = Path(manifest_path)
    manifest = read_json(mp) if mp.exists() else {"entries": {}}
    target = Path(rel_path)
    abs_target = (Path(base) / target) if base is not None else target
    entry = {
        "path": str(target).replace("\\", "/"),
        "sha256": sha256_file(abs_target) if Path(abs_target).exists() else None,
        "generator": generator,
        "generator_version": generator_version,
        "config_hash": config_hash,
        "n_records": n_records,
        "bytes": Path(abs_target).stat().st_size if Path(abs_target).exists() else None,
        "recorded_at": utc_stamp(),
        **(extra or {}),
    }
    manifest.setdefault("entries", {})[entry["path"]] = entry
    write_json(mp, manifest)
    return entry


def manifest_hash(manifest_path: str | Path) -> str:
    mp = Path(manifest_path)
    if not mp.exists():
        return "no-manifest"
    manifest = read_json(mp)
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, default=_json_default).encode()
    ).hexdigest()[:16]


def verify_manifest(manifest_path: str | Path, *, base: str | Path | None = None) -> dict[str, Any]:
    """Verify every recorded sha256; used by `tasks.py smoke` and before reports."""
    manifest = read_json(manifest_path)
    ok, missing, mismatch = [], [], []
    for rel, entry in manifest.get("entries", {}).items():
        target = (Path(base) / rel) if base is not None else Path(rel)
        if not target.exists():
            missing.append(rel)
        elif entry.get("sha256") and sha256_file(target) != entry["sha256"]:
            mismatch.append(rel)
        else:
            ok.append(rel)
    return {"ok": ok, "missing": missing, "mismatch": mismatch, "n_ok": len(ok)}


__all__ = [
    "ROOT",
    "add_manifest_entry",
    "append_jsonl",
    "capture_provenance",
    "git_sha",
    "gpu_info",
    "manifest_hash",
    "pip_freeze",
    "read_json",
    "read_jsonl",
    "run_dir",
    "sha256_dir",
    "sha256_file",
    "utc_stamp",
    "verify_manifest",
    "write_json",
    "write_jsonl",
]
