# Skill: Scalable Research Software Engineering

Goal: scale the number of **models, experiments, datasets/tasks, and evaluations** without rewriting code. Every commit follows this file.

## 1. Principles
1. **Config is the experiment.** An experiment = one YAML (pydantic-validated) → deterministic run dir `runs/<name>-<hash8>/<method>/<seed>/`.
2. **Interfaces over implementations.** Depend on Protocols; register implementations.
3. **One-way dependencies:** `utils ← envs/data ← policies/critic ← projection ← methods ← eval ← scripts`. No upward imports.
4. **Raw logs are immutable; reports are derived** (recomputable from JSONL). Notebooks are never the source of truth.
5. **Determinism and provenance:** seed python/numpy/torch/CUDA/env; CRN generator for flow noise; write git SHAs, pip freeze, data hashes.
6. **Fail loud:** assert shapes, finiteness, dynamics read-backs, convention checks; no silent fallbacks.
7. **Tests proportional to risk:** math → property/finite-difference tests; pipelines → toy-env integration tests.
8. **Small, reviewable commits.**

## 2. Core contracts
```python
class PolicyAdapter(Protocol):
    name: str
    def sites(self) -> dict[str, nn.Module]: ...
    def attach_lora(self, cfg: LoraCfg) -> None: ...
    def forward(self, batch, site: str | None = None, hook: Callable | None = None) -> PolicyOut: ...
    def f_act(self, ctx, h, site: str) -> Tensor: ...          # downstream map; terminal estimate for flow heads
class Critic(Protocol):     def logits(self, ctx, chunk) -> Tensor: ...
class Projector(Protocol):  def __call__(self, h_c, risk_fn, cfg) -> tuple[Tensor, dict]: ...
class Task(Protocol):       def make_env(self, dyn) -> Env; violations: ViolationSpec; expert: Policy
class Metric(Protocol):     name: str; def update(self, rec: dict): ...; def compute(self) -> dict: ...
class Method(Protocol):     def train(self, cfg): ...; def wrap(self, policy) -> Policy: ...
```
Registries: `@register("policy", "smolvla")`, `("task", "metaworld_peg_insert")`, `("metric", "proxy_gold")`, `("method", "b3")`.

## 3. How to extend (each ≤ 1 code file + 1 YAML + tests)
| Add… | Do |
|---|---|
| a **model** | `policies/<name>.py` adapter + `configs/model/<name>.yaml`; the parametrized **contract suite** must pass (zero-LoRA ≡ base; site gradient exists; per-example independence; seeded determinism) |
| an **experiment / sweep** | YAML only (lists in config expand to a grid; runs are idempotent by hash; `--shard i/n`; resumable) |
| a **dataset / task** | `Task` + violation spec + dynamics spec + demo generator + data card + manifest entry |
| an **evaluation / metric** | `eval/<metric>.py` + registry entry + unit test on a golden log; runs on archived logs |
| a **method / baseline** | `methods/<name>.py` implementing `Method`; shares training loop and budget accounting |

## 4. Quality gates
`ruff` + `mypy` (strict for `projection/`, `critic/`, `eval/stats.py`) + `pytest` markers (`unit`, `sim`, `gpu`, `slow`) + `pre-commit`. CI (CPU) runs unit + toy-env walking-skeleton in <2 min. Coverage target ≥90% on projection, critic, metrics, stats. Tolerance table for numerical tests lives in `tests/tolerances.py`.

## 5. Data and artifacts
`data/MANIFEST.json` with sha256, generator version, config hash; large files out of git (HF datasets or DVC); data cards per dataset; checkpoints named by run hash.

## 6. Git
Trunk `main` (always green); `exp/<id>` branches; tags `prereg/<id>` and `gate/G<n>`; conventional commits; PR checklist: tests, config, docs, log entry. Upstream changes only via `patches/`.

## 7. Performance
Profile before optimizing; cache frozen VLM features/KV where valid; vectorize per-example projection (no Python loops over batch); CPU env workers in multiprocessing; keep autograd enabled where projection needs it; fp32 for critic/projection math.

## 8. Anti-patterns
Hard-coded paths/module names; god-scripts; copy-pasted experiment scripts; metrics that need live models; mutable global state; editing submodules; undocumented thresholds; untested "quick fixes".

## Definition of done
Typed, tested, config-driven, seeded, logged, registered, documented in `agents/`, one `LOG.md` entry, CI green.
