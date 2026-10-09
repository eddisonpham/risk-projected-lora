# ENVIRONMENT — how this repo is actually run

Everything below describes the machine and the interpreter this project is developed on, and the
commands that were used. Where a fact comes from a tool output, the tool is named so it can be
re-checked. Nothing here is aspirational: if a row says "not installed", a check that needs it skips.

## 1. Machine and interpreter

| Item | Value | How to re-check |
|---|---|---|
| OS / shell | Windows, POSIX shell (**Git Bash**) — all commands in this repo use POSIX syntax (`ls`, `rm`, `/dev/null`) | `uname -a` |
| Python | 3.12.13 in a `uv`-managed virtualenv at `.venv` | `.venv/Scripts/python.exe --version` |
| Package manager | `uv` 0.11.30 (`uv pip install` against `VIRTUAL_ENV=.venv`) | `uv --version` |
| `make` | **not installed** → `python tasks.py <target>` is the canonical entry point; `Makefile` is a thin shim over the same targets | `make --version` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, compute capability **12.0 (sm_120)**, driver 577.13, CUDA 12.9, **8150.6 MB** VRAM | `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`; `python -c "import torch;print(torch.cuda.get_device_capability(0))"` |
| torch | **2.11.0+cu128**, installed from the CUDA 12.8 wheel index (Blackwell needs cu128 or newer) | `python -c "import torch;print(torch.__version__, torch.version.cuda)"` |
| Headless rendering | `MUJOCO_GL=wgl` on Windows (`egl` on Linux) — set automatically by `tasks.py:child_env()` | `python -c "import os;print(os.environ.get('MUJOCO_GL'))"` |

The VRAM budget in `configs/gates.yaml` (`gates.G2.budget_gb: 7.0`, the ~7 GB constraint the whole
model choice rests on, `01` §2, `05` §1) is sized to this card, not to a rented multi-GPU box.

Other environment variables the code sets or relies on: `PYTHONPATH=src` and `PYTHONUTF8=1` (set by
`tasks.py`), `PYTHONHASHSEED=0` (set in `tests/conftest.py`; `utils/seed.py` also pins it), and
`TOKENIZERS_PARALLELISM=false`.

## 2. Extras and what each unlocks (`pyproject.toml`)

| Extra | Adds | Needed for |
|---|---|---|
| *(core)* | numpy, scipy, pandas, pydantic, omegaconf, PyYAML, tqdm, rich | config, splits, metrics, logging, JSONL I/O — the CPU contract layer |
| `dev` | pytest, pytest-cov, pytest-xdist, hypothesis, ruff, mypy, pre-commit | `test`, `lint`, `typecheck` |
| `ml` | torch, torchvision, peft, safetensors | the critic's real training, LoRA, any GPU work |
| `sim` | mujoco, gymnasium | sim-marked tests; the Meta-World package comes from the `third_party/Metaworld` submodule |
| `vla` | lerobot, transformers, datasets, huggingface-hub | SmolVLA/ACT adapters and S0/S3/S4/S5 (heavy; installed by `bootstrap-sim`) |
| `stats` | rliable | stratified bootstrap CIs, IQM, performance profiles in the report |
| `figures` | matplotlib | figures F1–F8 |

## 3. Commands

```bash
python tasks.py setup          # create .venv (py3.12) + install -e ".[dev,stats,figures]" + import check
python tasks.py bootstrap-sim  # submodules, then -e ".[dev,ml,sim,stats,figures]", apply patches, then ".[vla]"
python tasks.py env-lock       # write env/lock.txt (the resolved pin set, `03` §2)
python tasks.py test           # pytest -m "not sim and not gpu and not slow"  (the CPU suite)
python tasks.py test-sim       # pytest -m sim
python tasks.py test-gpu       # pytest -m gpu
python tasks.py test-all       # everything, including slow
python tasks.py lint           # ruff check + ruff format --check
python tasks.py typecheck      # mypy (strict for projection/, critic/, eval/stats.py)
python tasks.py fmt            # ruff format + ruff check --fix
python tasks.py smoke          # smoke suite S0-S6 -> runs/_smoke/report.json
python tasks.py exp EXP=configs/exp/e0_baseline_repro.yaml
python tasks.py report RUN=runs/<name>-<hash8>/<method>/<seed>
python tasks.py clean          # drop caches; never touches runs/ or data/
```

Equivalent `make` targets exist for a machine that has `make`; the 31-line `Makefile` contains no
logic at all — every target is a one-line delegation to `tasks.py` — so the two entry points cannot
drift apart. `make help` forwards to `python tasks.py --help`.

Test markers (`pyproject.toml`): `unit` (fast, deterministic, no simulator, no GPU), `integration`
(more than one module, may use the toy env/policy), `sim` (MuJoCo/Meta-World), `gpu` (CUDA device +
`ml`), `slow` (>60 s or downloads). **`test` deliberately excludes `sim`, `gpu` and `slow`**, which
is why it is safe to run on any box.

## 4. What works without the optional stacks

- **Nothing in the core path needs a GPU or a simulator.** The config layer, the split/geometry
  rules, the projection operator's unit tests (closed-form and finite-difference), the critic's
  model/calibration/gate tests, the data-schema round-trips and the policy contract suite on the toy
  policy all run on CPU with `dev` alone.
- **`test-sim`** needs `sim` *and* the Meta-World package *and* `configs/env/task_geoms.yaml`, which is
  a **discovered** file (`scripts/inspect_contacts.py`), never hand-written. Absent any of those, the
  sim tests skip with an actionable message rather than failing.
- **`test-gpu`** needs `ml` plus a CUDA device.
- Reference point at the checkpoint commit `9efe571`, for comparison after any environment change:
  `python tasks.py test` → `291 passed, 4 skipped, 6 deselected`; the four skips are the real-policy
  contract cases that require the VLA stack and are gated behind `RISKPROJ_TEST_REAL_POLICIES=1`.

## 5. Not installed yet (and what that changes)

| Missing | Consequence | How it is surfaced |
|---|---|---|
| LeRobot and transformers (`vla`) | `SmolVLAAdapter` / `ACTAdapter` are unavailable; smoke S0/S3/S4/S5 and the real-policy contract tests cannot run | `is_available()` returns `False`; tests skip; the smoke suite records the skip reason |
| Meta-World package (`third_party/Metaworld` + import) | the research task `peg-insert-side-v3` cannot be built; only the toy task runs | `gymnasium`/`metaworld` imports are lazy and raise with the remediation command; `test-sim` skips |
| `configs/model/smolvla_sites.yaml` | SmolVLA site module names stay unverified (`[R]` in `02`), so S2/S3–S6 sites are unusable in practice | `policies/sites.py` resolves from the catalog with `module_pattern` empty and logs a warning; never invents a name |
| `configs/env/task_geoms.yaml` | friction/mass/stiffness edits cannot be applied to the real task | `envs/dynamics.py` raises `FileNotFoundError` naming `scripts/inspect_contacts.py` |
| `scripts/` entry points | `tasks.py` wires `scripts/smoke_suite.py`, `run_experiment.py`, `make_report.py`, `apply_patches.py`; several modules reference `scripts/inspect_model.py`, `inspect_contacts.py`, `sensitivity_sweep.py`, `derive_violation_thresholds.py` | check `ls scripts/` before invoking one; until then `tasks.py exp|smoke|report` cannot run |

A skip is **not** a pass. The smoke suite's job (`03` §3) is to report G0 honestly, including which of
S0–S6 could not run and why.

## 6. Windows caveats worth knowing before debugging

- Paths in shell commands use forward slashes; bash treats `\` as an escape.
- `.gitattributes` normalises line endings to LF so a Windows working tree does not create diff churn.
- `DataLoader` workers default to `num_workers: 0` (`configs/train/*.yaml`) because Windows spawns
  fresh interpreters per worker; raise it only with a measured reason.
- Determinism is set by `utils/seed.py`: python/numpy/torch/CUDA seeding is on, cuDNN autotuning is
  off, but `torch.use_deterministic_algorithms` is deliberately left **off** because parts of the
  simulator path are not deterministic. Runs are reproducible in the sense that CRN flow noise and
  episode seeds are pinned per (episode, step) — not in the bitwise sense across hardware.
- Every run directory carries `config.yaml`, `git_sha.txt` (repo + submodules), `pip_freeze.txt`,
  `gpu.txt`, `seeds.json` and `data_manifest_hash.txt` (`03` §6, `utils/io.capture_provenance`); when
  a result is surprising, the first thing to check is that those files match the intended run.
