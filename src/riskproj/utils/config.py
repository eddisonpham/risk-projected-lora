"""Config is the experiment: pydantic-validated YAML, hashed into the run dir name.

Every model forbids unknown keys (``extra="forbid"``): a typo in a research config
must fail loudly rather than silently run a different experiment.

Defaults in this file are **placeholders pending gates** — the dynamics ranges are
set at G1 and the violation thresholds are derived from nominal expert rollouts
before any learned policy is evaluated (``04`` §3-§4).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from riskproj.types import DynamicsRanges, DynamicsSpec, ViolationSpec

#: Catalog of projection sites (``05`` §3). S0/S0f are action-space, S1-S6 latent.
SITE_IDS: tuple[str, ...] = ("S0", "S0f", "S1", "S2", "S3", "S4", "S5", "S6")


class StrictModel(BaseModel):
    """Base config model: unknown keys are an error, not a silent no-op."""

    model_config = ConfigDict(extra="forbid", validate_assignment=False, frozen=False)


def _check_interval(iv: Sequence[float], *, name: str) -> tuple[float, float]:
    if len(iv) != 2:
        raise ValueError(f"{name}: expected [lo, hi], got {list(iv)!r}")
    lo, hi = (float(iv[0]), float(iv[1]))
    if not (lo > 0.0):
        raise ValueError(f"{name}: scales must be > 0 (got lo={lo})")
    if hi < lo:
        raise ValueError(f"{name}: hi < lo ({hi} < {lo})")
    return lo, hi


# --------------------------------------------------------------------------
# dynamics / splits
# --------------------------------------------------------------------------


class AxisRangesCfg(StrictModel):
    """Per-level multiplicative ranges for one axis. Overridden per axis below.

    ``xL``/``xH`` are used only by TEST-range cells (extrapolation beyond TRAIN).
    """

    xL: tuple[float, float] = (0.35, 0.5)
    L: tuple[float, float] = (0.5, 0.7)
    M: tuple[float, float] = (0.9, 1.1)
    H: tuple[float, float] = (1.4, 2.0)
    xH: tuple[float, float] = (2.4, 4.0)

    @model_validator(mode="after")
    def _validate(self) -> AxisRangesCfg:
        for level in ("xL", "L", "M", "H", "xH"):
            _check_interval(getattr(self, level), name=level)
        if not (self.L[1] < self.M[0] and self.M[1] < self.H[0]):
            raise ValueError("level ranges must not overlap: require L < M < H")
        return self


class DynamicsRangesCfg(StrictModel):
    """PLACEHOLDERS pending G1 (``04`` §3). Set from the sensitivity sweep, then frozen."""

    mu: AxisRangesCfg = Field(default_factory=AxisRangesCfg)
    mass: AxisRangesCfg = Field(
        default_factory=lambda: AxisRangesCfg(
            xL=(0.35, 0.5), L=(0.5, 0.75), M=(0.9, 1.1), H=(1.5, 2.5), xH=(2.75, 4.0)
        )
    )
    stiff: AxisRangesCfg = Field(
        default_factory=lambda: AxisRangesCfg(
            xL=(0.15, 0.25), L=(0.25, 0.5), M=(0.75, 1.25), H=(2.0, 4.0), xH=(5.0, 8.0)
        )
    )

    def to_ranges(self) -> DynamicsRanges:
        return DynamicsRanges(
            mu={lv: getattr(self.mu, lv) for lv in ("xL", "L", "M", "H", "xH")},
            mass={lv: getattr(self.mass, lv) for lv in ("xL", "L", "M", "H", "xH")},
            stiff={lv: getattr(self.stiff, lv) for lv in ("xL", "L", "M", "H", "xH")},
        )


class DynamicsCfg(StrictModel):
    ranges: DynamicsRangesCfg = Field(default_factory=DynamicsRangesCfg)
    nominal: DynamicsSpec = Field(default_factory=DynamicsSpec)
    apply_in_reset: bool = True  # dynamics applied on reset, never mid-episode
    verify_readback: bool = True  # assert the physics actually changed (fail loud)

    def to_ranges(self) -> DynamicsRanges:
        return self.ranges.to_ranges()


class SplitsCfg(StrictModel):
    """Roles are *derived* from the sign rules in ``04`` §3; only the two compound
    extrapolation cells are a choice, because the rules do not determine them."""

    range_compound: tuple[str, str] = ("xL-xH-M", "xH-xL-M")
    note: str = (
        "TRAIN/VAL/TEST-combo roles are derived, not configured. range_compound are the two "
        "two-axis extrapolation cells; chosen so that both signs of each axis appear."
    )

    @field_validator("range_compound")
    @classmethod
    def _validate(cls, v: tuple[str, str]) -> tuple[str, str]:
        from riskproj.types import parse_cell_id

        for cell in v:
            parse_cell_id(cell)
        return v


# --------------------------------------------------------------------------
# env / model
# --------------------------------------------------------------------------


class EnvCfg(StrictModel):
    name: str = "metaworld"
    task: str = "peg-insert-side-v3"
    max_steps: int = 200
    #: Labels are produced by executing ``exec_horizon`` steps open-loop, so the
    #: critic horizon must equal it (``04`` §1, asserted below).
    exec_horizon: int = 10
    critic_horizon: int = 10
    camera_size: tuple[int, int] = (128, 128)
    image_keys: tuple[str, ...] = ("top", "wrist")
    control_mode: str | None = None
    action_dim: int = 4
    state_dim: int | None = None  # filled by inspect_model / task probe

    @model_validator(mode="after")
    def _horizons_match(self) -> EnvCfg:
        if self.exec_horizon != self.critic_horizon:
            raise ValueError(
                "exec_horizon must equal critic_horizon: labels come from executing the "
                "chunk open-loop (04 §1). Got "
                f"{self.exec_horizon} != {self.critic_horizon}"
            )
        return self


class LoraCfg(StrictModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    #: Module-name suffixes are [R] until scripts/inspect_model.py confirms them.
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj", "o_proj", "down_proj")
    last_n_blocks: int = 4  # -1 = all blocks
    bias: Literal["none", "all", "lora_only"] = "none"
    init: Literal["default", "gaussian", "kaiming", "pissa"] = "default"

    @property
    def scale(self) -> float:
        """LoRA scale ``s = alpha / r``."""
        if self.r <= 0:
            raise ValueError("lora.r must be positive")
        return self.alpha / self.r


class SiteCfg(StrictModel):
    default: str = "S2"
    catalog: tuple[str, ...] = SITE_IDS
    #: Written by scripts/inspect_model.py (S3); never hand-edited.
    resolved_file: str = "configs/model/smolvla_sites.yaml"
    per_token: bool = False  # flatten all tokens jointly unless ablating (07 §5.5)

    @field_validator("default")
    @classmethod
    def _known_site(cls, v: str) -> str:
        if v not in SITE_IDS:
            raise ValueError(f"unknown site {v!r}; expected one of {SITE_IDS}")
        return v


class ModelCfg(StrictModel):
    name: str = "smolvla"
    checkpoint: str | None = "lerobot/smolvla_base"
    chunk_size: int = 50
    num_flow_steps: int = 10
    num_vlm_layers: int = 16
    expert_width_multiplier: float = 0.75
    self_attn_every_n_layers: int = 2
    train_expert_only: bool = True
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    device: str = "auto"
    lora: LoraCfg = Field(default_factory=LoraCfg)
    site: SiteCfg = Field(default_factory=SiteCfg)
    #: Deterministic-head models (ACT) ignore flow-only knobs.
    action_head: Literal["flow_matching", "regression"] = "flow_matching"
    tau_convention: Literal["tau1_is_noise", "tau0_is_noise"] = "tau1_is_noise"


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------


class ProjCfg(StrictModel):
    """Projection operator configuration (``07`` §2). Validated, not free-form."""

    tau: float = 0.5  # risk threshold, probability scale unless space="logit"
    eta: float = 1e-8  # minimum squared gradient norm to call the gradient usable
    K: int = 1  # max projection iterations
    space: Literal["prob", "logit"] = "prob"
    margin: float = 0.0  # target tau - margin (the setpoint actually projected onto)
    max_rel_step: float | None = None  # trust region: ||u|| <= rho * ||h_c||
    backward: Literal["ste", "projected", "second_order"] = "projected"
    per_token: bool = False  # False = flatten all tokens jointly (default)
    steps_active: int = 3  # m: apply only at the last m denoising steps
    tau_proj: float = 0.3  # active only when tau <= tau_proj (tau=1 is noise)
    fallback: bool = True  # consult the fallback ladder when a gradient is unusable
    collect_trace: bool = True

    @model_validator(mode="after")
    def _validate(self) -> ProjCfg:
        if not (0.0 < self.tau < 1.0):
            raise ValueError(f"tau must be in (0,1); got {self.tau}")
        if self.margin < 0.0 or self.margin >= self.tau:
            raise ValueError(f"margin must satisfy 0 <= margin < tau; got {self.margin}")
        if self.eta <= 0.0:
            raise ValueError("eta must be > 0")
        if self.K < 1:
            raise ValueError("K must be >= 1")
        if self.max_rel_step is not None and self.max_rel_step <= 0:
            raise ValueError("max_rel_step must be > 0 when set")
        if self.steps_active < 1:
            raise ValueError("steps_active must be >= 1")
        if not (0.0 < self.tau_proj <= 1.0):
            raise ValueError("tau_proj must be in (0,1]")
        return self

    @property
    def setpoint(self) -> float:
        return self.tau - self.margin


# --------------------------------------------------------------------------
# critic / train / eval / gates
# --------------------------------------------------------------------------


class CriticCfg(StrictModel):
    hidden: int = 256
    depth: int = 3
    ensemble: int = 5
    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 200
    weight_decay: float = 1e-4
    dropout: float = 0.0
    class_weight: Literal["balanced", "none"] = "balanced"
    horizon: int = 10  # must match EnvCfg.critic_horizon (checked in ExperimentCfg)
    #: Deployable inputs only (``04`` §5). Vision / privileged force are ablations.
    use_vision: bool = False
    use_history: int = 1
    privileged_force: bool = False  # flagged: ablation E-critic-force only
    spectral_norm: bool = False
    grad_norm_penalty: float = 0.0  # keeps input gradients informative (``06`` §3)
    temperature: float = 1.0  # temperature scaling on VAL
    bootstrap: bool = True
    seed: int = 0
    max_tuples: int | None = None


class TrainCfg(StrictModel):
    stage: Literal["A", "B"] = "B"
    steps: int = 10_000
    batch_size: int = 8
    grad_accum: int = 4
    lr: float = 1e-4
    min_lr_ratio: float = 0.025
    warmup_steps: int = 100
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    schedule: Literal["cosine", "constant"] = "cosine"
    dtype: Literal["float32", "bfloat16"] = "bfloat16"
    grad_clip: float | None = 1.0
    save_every: int = 2000
    eval_every: int = 1000
    log_every: int = 50
    num_workers: int = 0
    max_train_minutes: float | None = None
    #: Penalty weight λ (methods "pen"/B2). Kept here so training-time methods share it.
    lam: float = 0.0

    @field_validator("betas")
    @classmethod
    def _betas(cls, v: tuple[float, float]) -> tuple[float, float]:
        if not (0.0 < v[0] < 1.0 and 0.0 < v[1] < 1.0):
            raise ValueError("betas must be in (0,1)")
        return v


class EvalCfg(StrictModel):
    eps_per_cell: int = 40
    cells: tuple[str, ...] = ()  # empty = every cell of each requested split
    splits: tuple[str, ...] = ("val",)
    max_steps: int | None = None  # None = env max_steps
    record_decisions: bool = True
    crn: bool = True
    deterministic_policy: bool = False
    save_states: bool = False  # needed for E1 counterfactual evaluation
    state_save_every: int = 1
    n_states_per_cell: int = 20


class DynamicsGateCfg(StrictModel):
    n_scales: int = 7
    scale_lo: float = 0.25
    scale_hi: float = 4.0
    episodes_per_scale: int = 30
    pi_base_success_band: tuple[float, float] = (0.30, 0.90)
    axis_shift_pp: float = 15.0
    min_violation_rate: float = 0.05


class VramGateCfg(StrictModel):
    budget_gb: float = 7.0  # the whole point of the model choice (``01``)
    headroom_gb: float = 0.25


class CriticGateCfg(StrictModel):
    auroc: float = 0.85
    auprc_mult: float = 2.0  # multiple of base rate
    ece: float = 0.05
    fnr_at_tau: float = 0.15
    min_cell_auroc: float = 0.75
    require_val_disagreement_gt_train: bool = True


class PilotGateCfg(StrictModel):
    max_fallback_rate: float = 0.30  # G6
    max_action_suppression_rel: float = 0.25  # ||a|| vs B1


class GatesCfg(StrictModel):
    G1: DynamicsGateCfg = Field(default_factory=DynamicsGateCfg)
    G2: VramGateCfg = Field(default_factory=VramGateCfg)
    G3: CriticGateCfg = Field(default_factory=CriticGateCfg)
    G6: PilotGateCfg = Field(default_factory=PilotGateCfg)


class MethodCfg(StrictModel):
    id: str = "B1"
    train_mode: Literal["0", "pen", "lat", "act"] = "0"
    inference_mode: Literal["0", "lat", "act-post", "act-flow", "bon", "gate"] = "0"
    projection: ProjCfg = Field(default_factory=ProjCfg)
    best_of_n: int = 8
    noise_schedule: Literal["beta", "logit_normal", "uniform"] = "beta"
    ln_m: float = 1.5
    ln_s: float = 1.0
    #: B2 penalty weight on the terminal estimate. ``lambda`` is aliased to ``lam``.
    lam: float = Field(default=0.0, alias="lambda")
    lambda_dual_alpha: float = 0.01
    violation_budget: float | None = None
    #: Negative controls (``08`` §2): C1 random, C2 shrink, C3 shuffled critic.
    control: Literal["none", "C1", "C2", "C3"] = "none"
    notes: str = ""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------


class ViolationCfg(StrictModel):
    """Thresholds frozen in ``configs/violations.yaml`` before policy evaluation."""

    slip_drift: float = 0.02
    drop_dist: float = 0.03
    force_max: float = 60.0
    penetration: float = 0.005
    jam_force: float = 40.0
    jam_progress_eps: float = 0.002
    jam_steps: int = 5
    window: int = 10
    horizon: int = 10
    provenance: dict[str, Any] = Field(default_factory=dict)

    def to_spec(self) -> ViolationSpec:
        d = self.model_dump()
        return ViolationSpec(**d)


class ExperimentCfg(StrictModel):
    name: str
    seed: int = 0
    seeds: tuple[int, ...] = ()  # replication seeds (>=3 for TEST); seed used if empty
    description: str = ""
    env: EnvCfg = Field(default_factory=EnvCfg)
    dynamics: DynamicsCfg = Field(default_factory=DynamicsCfg)
    splits: SplitsCfg = Field(default_factory=SplitsCfg)
    violations: ViolationCfg = Field(default_factory=ViolationCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)
    critic: CriticCfg = Field(default_factory=CriticCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    gates: GatesCfg = Field(default_factory=GatesCfg)
    methods: tuple[MethodCfg, ...] = (MethodCfg(),)
    out_dir: str = "runs"
    #: Set only by the pre-registration tag; recorded in every run dir.
    prereg_tag: str | None = None

    @model_validator(mode="after")
    def _cross_checks(self) -> ExperimentCfg:
        if self.critic.horizon != self.env.critic_horizon:
            raise ValueError(
                f"critic.horizon ({self.critic.horizon}) != env.critic_horizon "
                f"({self.env.critic_horizon})"
            )
        if self.violations.horizon != self.env.critic_horizon:
            raise ValueError(
                f"violations.horizon ({self.violations.horizon}) != env.critic_horizon "
                f"({self.env.critic_horizon})"
            )
        if not self.methods:
            raise ValueError("at least one method is required")
        ids = [m.id for m in self.methods]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate method ids in config: {ids}")
        return self

    # -- provenance -------------------------------------------------------
    def canonical(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    def config_hash(self) -> str:
        """8-hex-char hash of the resolved config: the run identity."""
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]

    def run_name(self) -> str:
        return f"{self.name}-{self.config_hash()}"

    def seed_list(self) -> list[int]:
        return list(self.seeds) if self.seeds else [self.seed]


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top level of a config must be a mapping")
    return data


def load_config(path: str | Path) -> ExperimentCfg:
    """Load and validate an experiment config. Raises on unknown/missing keys."""
    return ExperimentCfg.model_validate(load_yaml(path))


def dump_config(cfg: ExperimentCfg, path: str | Path) -> Path:
    """Write the *resolved* config next to a run (never the raw YAML only)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.canonical(), sort_keys=True), encoding="utf-8")
    return p


def load_config_from_dir(run_dir: str | Path) -> ExperimentCfg:
    """Reload the frozen resolved config from a run directory."""
    return load_config(Path(run_dir) / "config.yaml")


__all__ = [
    "SITE_IDS",
    "AxisRangesCfg",
    "CriticCfg",
    "CriticGateCfg",
    "DynamicsCfg",
    "DynamicsGateCfg",
    "DynamicsRangesCfg",
    "EnvCfg",
    "EvalCfg",
    "ExperimentCfg",
    "GatesCfg",
    "LoraCfg",
    "MethodCfg",
    "ModelCfg",
    "PilotGateCfg",
    "ProjCfg",
    "SiteCfg",
    "SplitsCfg",
    "StrictModel",
    "TrainCfg",
    "ViolationCfg",
    "VramGateCfg",
    "dump_config",
    "load_config",
    "load_config_from_dir",
    "load_yaml",
]
