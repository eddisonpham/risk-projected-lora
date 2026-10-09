"""Dataset schemas: demonstrations, critic tuples, data cards and manifests.

Storage is deliberately boring (``.npz`` arrays + a JSON sidecar): no pickle, no
parquet, nothing that needs the live models to read. Every dataset gets a data
card (generator, version, config hash, class balance, cell histogram) and a
sha256 in ``data/MANIFEST.json`` (``skills/software_engineering.md`` §5).

The critic context stored per tuple uses exactly the ``obs.CTX_KEYS`` layout, and
:func:`assert_deployable_ctx` is the leak guard that stops a non-deployable input
(privileged force, vision when disabled) from entering a headline experiment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from riskproj.obs import CTX_KEYS
from riskproj.types import VIOLATION_TYPES, CellId
from riskproj.utils.io import add_manifest_entry, read_json, sha256_file, utc_stamp, write_json

#: Bump on any change to the on-disk layout; recorded in every data card.
DATASET_VERSION = "0.1.0"

#: Chunk provenance categories for D-crit (``04`` §7).
CHUNK_SOURCES: tuple[str, ...] = ("expert", "policy", "perturbed", "unsafe")


def _as_f32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


# --------------------------------------------------------------------------
# demonstrations (D-nom, D-B)
# --------------------------------------------------------------------------


@dataclass
class DemoRecord:
    """One successful (or explicitly kept) expert episode."""

    ep_id: str
    cell_id: CellId
    split: str
    seed: int
    task: str
    success: bool
    steps: int
    actions: np.ndarray  # [T, A]
    proprio: np.ndarray  # [T, P]
    instruction: str = ""
    images: np.ndarray | None = None  # [T, ...] or None
    progress: np.ndarray | None = None  # [T]
    dyn: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.actions = _as_f32(self.actions)
        self.proprio = _as_f32(self.proprio)
        if self.actions.ndim != 2:
            raise ValueError(f"actions must be [T, A]; got {self.actions.shape}")
        if self.proprio.shape[0] != self.actions.shape[0]:
            raise ValueError(
                f"proprio/actions length mismatch: {self.proprio.shape[0]} vs {self.actions.shape[0]}"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe metadata (arrays excluded — they live in the npz)."""
        return {
            "ep_id": self.ep_id,
            "cell_id": self.cell_id,
            "split": self.split,
            "seed": int(self.seed),
            "task": self.task,
            "success": bool(self.success),
            "steps": int(self.steps),
            "instruction": self.instruction,
            "dyn": dict(self.dyn),
            "has_images": self.images is not None,
            "has_progress": self.progress is not None,
            "extra": dict(self.extra),
        }

    @staticmethod
    def from_dict(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> DemoRecord:
        return DemoRecord(
            ep_id=meta["ep_id"],
            cell_id=meta["cell_id"],
            split=meta["split"],
            seed=int(meta["seed"]),
            task=meta["task"],
            success=bool(meta["success"]),
            steps=int(meta["steps"]),
            actions=arrays["actions"],
            proprio=arrays["proprio"],
            instruction=meta.get("instruction", ""),
            images=arrays.get("images"),
            progress=arrays.get("progress"),
            dyn=dict(meta.get("dyn", {})),
            extra=dict(meta.get("extra", {})),
        )


def write_demos(path: str | Path, records: Sequence[DemoRecord], *, shard: str = "demos") -> Path:
    """Write demonstrations as one variable-length ``.npz`` plus a JSON sidecar.

    Variable-length episodes are stored concatenated with a ``lengths`` array so
    the file stays a single array container (fast, mmap-able, no pickle).
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("write_demos called with no records")
    lengths = np.array([r.actions.shape[0] for r in records], dtype=np.int64)
    arrays: dict[str, np.ndarray] = {
        "lengths": lengths,
        "actions": np.concatenate([r.actions for r in records], axis=0),
        "proprio": np.concatenate([r.proprio for r in records], axis=0),
    }
    if all(r.images is not None for r in records):
        arrays["images"] = np.concatenate([np.asarray(r.images) for r in records], axis=0)
    if all(r.progress is not None for r in records):
        arrays["progress"] = np.concatenate(
            [np.asarray(r.progress).reshape(-1) for r in records], axis=0
        )
    npz_path = directory / f"{shard}.npz"
    np.savez_compressed(npz_path, **arrays)
    write_json(
        directory / f"{shard}.json",
        {
            "dataset_version": DATASET_VERSION,
            "kind": "demos",
            "n_records": len(records),
            "records": [r.to_dict() for r in records],
        },
    )
    return npz_path


def read_demos(path: str | Path, *, shard: str = "demos") -> list[DemoRecord]:
    directory = Path(path)
    meta = read_json(directory / f"{shard}.json")
    with np.load(directory / f"{shard}.npz") as npz:
        data = {k: npz[k] for k in npz.files}
    lengths = data["lengths"]
    out: list[DemoRecord] = []
    cursor = 0
    for i, meta_row in enumerate(meta["records"]):
        n = int(lengths[i])
        sl = slice(cursor, cursor + n)
        arrays = {
            "actions": data["actions"][sl],
            "proprio": data["proprio"][sl],
        }
        if "images" in data:
            arrays["images"] = data["images"][sl]
        if "progress" in data:
            arrays["progress"] = data["progress"][sl]
        out.append(DemoRecord.from_dict(meta_row, arrays))
        cursor += n
    return out


# --------------------------------------------------------------------------
# critic tuples (D-crit)
# --------------------------------------------------------------------------


@dataclass
class CriticTuple:
    """One candidate chunk from one saved simulator state, with its labels.

    ``ctx`` follows :data:`riskproj.obs.CTX_KEYS` exactly; ``labels`` always
    carries **every** violation type (``04`` §4: per-type labels are always
    stored), and ``cell_id`` is metadata only — never a model input.
    """

    tuple_id: str
    cell_id: CellId
    split: str
    seed: int
    ctx: dict[str, np.ndarray]
    chunk: np.ndarray  # [H_c, A]
    labels: dict[str, bool]
    chunk_source: str = "expert"
    perturbation_sigma: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [k for k in CTX_KEYS if k not in self.ctx]
        if missing:
            raise ValueError(f"critic ctx is missing keys {missing}; expected {list(CTX_KEYS)}")
        self.ctx = {k: _as_f32(self.ctx[k]) for k in CTX_KEYS}
        self.chunk = _as_f32(self.chunk)
        if self.chunk.ndim != 2:
            raise ValueError(f"chunk must be [H_c, A]; got {self.chunk.shape}")
        unknown = set(self.labels) - set(VIOLATION_TYPES)
        if unknown:
            raise ValueError(
                f"unknown label keys {sorted(unknown)}; expected {list(VIOLATION_TYPES)}"
            )
        if self.chunk_source not in CHUNK_SOURCES:
            raise ValueError(
                f"unknown chunk_source {self.chunk_source!r}; expected {CHUNK_SOURCES}"
            )

    @property
    def any_violation(self) -> bool:
        """``any_violation = V1 or ... or V5`` — the scalar critic target (``04`` §4)."""
        return any(bool(self.labels.get(v, False)) for v in VIOLATION_TYPES)

    def label_vector(self) -> np.ndarray:
        return np.array(
            [1.0 if self.labels.get(v, False) else 0.0 for v in VIOLATION_TYPES], dtype=np.float32
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tuple_id": self.tuple_id,
            "cell_id": self.cell_id,
            "split": self.split,
            "seed": int(self.seed),
            "labels": {v: bool(self.labels.get(v, False)) for v in VIOLATION_TYPES},
            "chunk_source": self.chunk_source,
            "perturbation_sigma": self.perturbation_sigma,
            "extra": dict(self.extra),
        }

    @staticmethod
    def from_dict(
        meta: dict[str, Any], ctx: dict[str, np.ndarray], chunk: np.ndarray
    ) -> CriticTuple:
        return CriticTuple(
            tuple_id=meta["tuple_id"],
            cell_id=meta["cell_id"],
            split=meta["split"],
            seed=int(meta["seed"]),
            ctx=ctx,
            chunk=chunk,
            labels=dict(meta["labels"]),
            chunk_source=meta.get("chunk_source", "expert"),
            perturbation_sigma=meta.get("perturbation_sigma"),
            extra=dict(meta.get("extra", {})),
        )


def assert_deployable_ctx(
    ctx: dict[str, np.ndarray], *, use_vision: bool, privileged_force: bool
) -> None:
    """Leak guard: no non-deployable input may enter a tuple (``04`` §5).

    Raises:
        ValueError: a populated vision/force block while the config forbids it.
    """
    if not use_vision and np.asarray(ctx.get("vision_emb", [])).size:
        raise ValueError(
            "ctx carries a vision embedding but use_vision=False (04 §5: non-deployable)"
        )
    if not privileged_force and np.asarray(ctx.get("force", [])).size:
        raise ValueError(
            "ctx carries simulated force but privileged_force=False (04 §5: flagged input)"
        )


class CriticDataset(Dataset):  # type: ignore[type-arg]
    """In-memory D-crit tuples with a single-file, sidecar-based on-disk format."""

    def __init__(self, tuples: Sequence[CriticTuple]) -> None:
        self.tuples: list[CriticTuple] = list(tuples)

    # -- Dataset ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tuples)

    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
        t = self.tuples[index]
        ctx = {k: torch.from_numpy(np.ascontiguousarray(t.ctx[k])) for k in CTX_KEYS}
        # history blocks are stored flat; expose them as [n, width] for the critic
        ctx["action_hist"] = _unflatten_history(ctx["action_hist"], self._action_dim)
        ctx["proprio_hist"] = _unflatten_history(ctx["proprio_hist"], self._state_dim)
        chunk = torch.from_numpy(np.ascontiguousarray(t.chunk))
        label_any = torch.tensor(1.0 if t.any_violation else 0.0)
        label_per_type = torch.from_numpy(t.label_vector())
        return ctx, chunk, label_any, label_per_type

    # -- labels ----------------------------------------------------------
    def labels_any(self) -> Tensor:
        return torch.tensor(
            [1.0 if t.any_violation else 0.0 for t in self.tuples], dtype=torch.float32
        )

    def labels_per_type(self) -> Tensor:
        if not self.tuples:
            return torch.zeros((0, len(VIOLATION_TYPES)), dtype=torch.float32)
        return torch.from_numpy(np.stack([t.label_vector() for t in self.tuples]))

    def class_balance(self) -> dict[str, float]:
        y = self.labels_any()
        if y.numel() == 0:
            return {"n": 0.0, "pos_rate": float("nan")}
        return {"n": float(y.numel()), "pos_rate": float(y.mean())}

    def per_type_balance(self) -> dict[str, float]:
        if not self.tuples:
            return {v: float("nan") for v in VIOLATION_TYPES}
        y = self.labels_per_type()
        return {v: float(y[:, i].mean()) for i, v in enumerate(VIOLATION_TYPES)}

    def per_cell_positives(self) -> dict[str, float]:
        """Per-cell positive rate — leaks are usually visible here first (``06`` §2)."""
        buckets: dict[str, list[float]] = {}
        for t in self.tuples:
            buckets.setdefault(t.cell_id, []).append(1.0 if t.any_violation else 0.0)
        return {cell: float(np.mean(vals)) for cell, vals in sorted(buckets.items())}

    def cell_histogram(self) -> dict[str, int]:
        hist: dict[str, int] = {}
        for t in self.tuples:
            hist[t.cell_id] = hist.get(t.cell_id, 0) + 1
        return dict(sorted(hist.items()))

    def split_histogram(self) -> dict[str, int]:
        hist: dict[str, int] = {}
        for t in self.tuples:
            hist[t.split] = hist.get(t.split, 0) + 1
        return dict(sorted(hist.items()))

    def source_balance(self) -> dict[str, float]:
        if not self.tuples:
            return {s: float("nan") for s in CHUNK_SOURCES}
        total = float(len(self.tuples))
        counts = dict.fromkeys(CHUNK_SOURCES, 0)
        for t in self.tuples:
            counts[t.chunk_source] = counts.get(t.chunk_source, 0) + 1
        return {s: counts[s] / total for s in CHUNK_SOURCES}

    def subset(self, indices: Sequence[int]) -> CriticDataset:
        return CriticDataset([self.tuples[i] for i in indices])

    # -- shapes ----------------------------------------------------------
    @property
    def _action_dim(self) -> int:
        return int(self.tuples[0].chunk.shape[1]) if self.tuples else 0

    @property
    def _state_dim(self) -> int:
        return int(self.tuples[0].ctx["proprio"].shape[0]) if self.tuples else 0

    @property
    def _hist_len(self) -> int:
        """History steps, inferred the same way for in-memory and reloaded tuples.

        In memory a context block is ``[steps, width]``; ``save`` flattens it
        (``reshape(-1)``), so after a reload it is ``steps * width`` long. Both
        forms must yield the same count, otherwise a freshly saved dataset
        disagrees with itself on reload and ``load`` correctly refuses it.
        """
        if not self.tuples:
            return 0
        block = np.asarray(self.tuples[0].ctx["proprio_hist"])
        if block.ndim >= 2:
            return int(block.shape[0])
        return int(block.shape[0]) // (self._state_dim or 1)

    def feature_spec_kwargs(self) -> dict[str, int]:
        """Widths the critic needs to size its input layer without guessing."""
        if not self.tuples:
            return {
                "state_dim": 0,
                "action_dim": 0,
                "chunk_horizon": 0,
                "hist_len": 0,
                "vision_dim": 0,
                "force_dim": 0,
            }
        t0 = self.tuples[0]
        return {
            "state_dim": int(t0.ctx["proprio"].shape[0]),
            "action_dim": int(t0.chunk.shape[1]),
            "chunk_horizon": int(t0.chunk.shape[0]),
            "hist_len": self._hist_len,
            "vision_dim": int(t0.ctx["vision_emb"].shape[0]),
            "force_dim": int(t0.ctx["force"].shape[0]),
        }

    # -- io --------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write ``<path>.npz`` (arrays) + ``<path>.json`` (metadata)."""
        stem = Path(path)
        if stem.suffix in (".npz", ".json"):
            stem = stem.with_suffix("")
        stem.parent.mkdir(parents=True, exist_ok=True)
        n = len(self.tuples)
        spec = self.feature_spec_kwargs()
        arrays: dict[str, np.ndarray] = {
            "chunk": np.stack([t.chunk for t in self.tuples]).astype(np.float32)
            if n
            else np.zeros((0,)),
            "label_any": self.labels_any().numpy(),
            "label_per_type": self.labels_per_type().numpy(),
        }
        for key in CTX_KEYS:
            block = (
                np.stack(
                    [np.asarray(t.ctx[key], dtype=np.float32).reshape(-1) for t in self.tuples]
                )
                if n
                else np.zeros((0, 0), dtype=np.float32)
            )
            arrays[f"ctx__{key}"] = block
        npz_path = stem.with_suffix(".npz")
        np.savez_compressed(npz_path, **arrays)
        write_json(
            stem.with_suffix(".json"),
            {
                "dataset_version": DATASET_VERSION,
                "kind": "critic_tuples",
                "n_records": n,
                "spec": spec,
                "records": [t.to_dict() for t in self.tuples],
                "saved_at": utc_stamp(),
            },
        )
        return npz_path

    @classmethod
    def load(cls, path: str | Path) -> CriticDataset:
        stem = Path(path)
        if stem.suffix in (".npz", ".json"):
            stem = stem.with_suffix("")
        meta = read_json(stem.with_suffix(".json"))
        with np.load(stem.with_suffix(".npz")) as npz:
            data = {k: npz[k] for k in npz.files}
        spec = meta.get("spec", {})
        rows = meta.get("records", [])
        tuples: list[CriticTuple] = []
        for i, meta_row in enumerate(rows):
            ctx: dict[str, np.ndarray] = {}
            for key in CTX_KEYS:
                block = np.asarray(data.get(f"ctx__{key}", np.zeros(0, dtype=np.float32)))
                if block.ndim == 2 and block.shape[0] == len(rows):
                    ctx[key] = block[i].astype(np.float32)
                elif block.ndim == 1 and key in ("proprio", "ee_vel"):
                    ctx[key] = block.astype(np.float32)
                else:
                    ctx[key] = np.zeros(0, dtype=np.float32)
            chunk = np.asarray(data["chunk"])[i].astype(np.float32)
            tuples.append(CriticTuple.from_dict(meta_row, ctx, chunk))
        ds = cls(tuples)
        if spec and ds.feature_spec_kwargs() != spec:  # fail loud on a layout mismatch
            raise ValueError(
                f"{stem}.json spec {spec} does not match the reloaded arrays "
                f"{ds.feature_spec_kwargs()}; the dataset was written by a different version"
            )
        return ds

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        bal = self.class_balance()
        return f"CriticDataset(n={len(self)}, pos_rate={bal['pos_rate']:.3f})"


def _unflatten_history(flat: Tensor, width: int) -> Tensor:
    """Reshape a stored ``[n * width]`` history block back to ``[n, width]``."""
    if flat.numel() == 0:
        return flat.reshape(1, 0)
    if width <= 0:
        return flat.reshape(1, -1)
    if flat.numel() % width != 0:
        raise ValueError(f"history block of width {flat.numel()} is not a multiple of {width}")
    return flat.reshape(-1, width)


# --------------------------------------------------------------------------
# data cards / manifest
# --------------------------------------------------------------------------


def data_card(
    dataset_or_records: CriticDataset | Sequence[Any],
    *,
    generator: str,
    config_hash: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generator version, config hash, class balance, cell histogram (``skills`` §5)."""
    card: dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "generator": generator,
        "config_hash": config_hash,
        "created_at": utc_stamp(),
        "extra": dict(extra or {}),
    }
    if isinstance(dataset_or_records, CriticDataset):
        ds = dataset_or_records
        card["kind"] = "critic_tuples"
        card["n_records"] = len(ds)
        card["class_balance"] = ds.class_balance()
        card["per_type_balance"] = ds.per_type_balance()
        card["per_cell_positives"] = ds.per_cell_positives()
        card["cell_histogram"] = ds.cell_histogram()
        card["split_histogram"] = ds.split_histogram()
        card["source_balance"] = ds.source_balance()
        card["feature_spec"] = ds.feature_spec_kwargs()
        return card

    records = list(dataset_or_records)
    card["n_records"] = len(records)
    if records and isinstance(records[0], DemoRecord):
        cells: dict[str, int] = {}
        splits: dict[str, int] = {}
        n_ok = 0
        steps: list[int] = []
        for r in records:
            cells[r.cell_id] = cells.get(r.cell_id, 0) + 1
            splits[r.split] = splits.get(r.split, 0) + 1
            n_ok += int(r.success)
            steps.append(int(r.steps))
        card["kind"] = "demos"
        card["success_rate"] = n_ok / len(records)
        card["mean_steps"] = float(np.mean(steps))
        card["cell_histogram"] = dict(sorted(cells.items()))
        card["split_histogram"] = dict(sorted(splits.items()))
        card["task"] = records[0].task
    elif records and isinstance(records[0], CriticTuple):
        return data_card(
            CriticDataset(records), generator=generator, config_hash=config_hash, extra=extra
        )
    else:
        card["kind"] = "unknown"
    return card


def write_data_card(path: str | Path, card: dict[str, Any]) -> Path:
    return write_json(path, card)


def register_dataset(
    path: str | Path,
    *,
    generator: str,
    generator_version: str,
    config_hash: str,
    manifest_path: str | Path,
    base: str | Path | None = None,
    n_records: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register an artifact in ``data/MANIFEST.json`` with its sha256."""
    target = Path(path)
    rel = (
        str(target.relative_to(base))
        if base is not None and target.is_relative_to(Path(base))
        else str(target)
    )
    return add_manifest_entry(
        manifest_path,
        rel,
        generator=generator,
        generator_version=generator_version,
        config_hash=config_hash,
        n_records=n_records,
        extra=extra,
        base=base,
    )


def artifact_fingerprint(path: str | Path) -> dict[str, Any]:
    """sha256 + size, for the data card of a written artifact."""
    p = Path(path)
    return {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}


__all__ = [
    "CHUNK_SOURCES",
    "DATASET_VERSION",
    "CriticDataset",
    "CriticTuple",
    "DemoRecord",
    "artifact_fingerprint",
    "assert_deployable_ctx",
    "data_card",
    "read_demos",
    "register_dataset",
    "write_data_card",
    "write_demos",
]
