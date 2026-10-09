# 03 — Repository Setup

Follow `skills/software_engineering.md` for all code. This file is Phase 0 (gate G0).

## 1. Layout
```
riskproj/
├── CLAUDE.md
├── agents/                      # all markdown docs (this folder)
├── configs/                     # env/ dynamics/ violations/ model/ critic/ method/ train/ eval/ gates/ exp/
├── src/riskproj/
│   ├── envs/        metaworld_dyn.py  dynamics.py  violations.py  state_io.py
│   ├── data/        gen_demos.py  gen_critic_data.py  splits.py  schema.py
│   ├── policies/    base.py(PolicyAdapter)  smolvla.py  act.py  vla_adapter.py(opt)  sites.py
│   ├── critic/      model.py  train.py  calibrate.py  gates.py
│   ├── projection/  operator.py  backward.py  fallback.py  logging.py
│   ├── methods/     lora_sft.py  penalty.py  lagrangian.py  action_proj.py  best_of_n.py  controls.py
│   ├── eval/        rollout.py  counterfactual.py  critic_eval.py  diagnostics.py  systems.py  stats.py  report.py
│   └── utils/       seed.py  registry.py  profiling.py  io.py  config.py
├── scripts/         inspect_model.py  inspect_contacts.py  profile_vram.py  sensitivity_sweep.py  run_experiment.py  make_report.py
├── tests/           unit/  integration/  sim/            # markers: unit, gpu, sim, slow
├── third_party/     lerobot/  Metaworld/  (submodules, pinned)  [VLA-Adapter/ optional]
├── patches/         *.patch + README explaining each
├── runs/  data/     # gitignored; data/MANIFEST.json (sha256) is committed
└── Makefile  pyproject.toml  .pre-commit-config.yaml
```

## 2. Bootstrap (commands are a template — read upstream READMEs/pyproject first; versions drift)
```bash
git init riskproj && cd riskproj
git submodule add https://github.com/huggingface/lerobot third_party/lerobot
git submodule add https://github.com/Farama-Foundation/Metaworld third_party/Metaworld
uv venv                                       # choose the Python version LeRobot's pyproject supports; do not guess
uv pip install -e "third_party/lerobot[smolvla]"   # extra name seen in HF model-card install snippets [Va]
# Meta-World: LeRobot may provide an extra (e.g. [metaworld]) [R]; otherwise install third_party/Metaworld.
# If MuJoCo pins conflict, LeRobot's pins win; record the resolved set in env/lock.
uv pip install -e . rliable peft pytest hypothesis ruff mypy pre-commit pydantic omegaconf
export MUJOCO_GL=egl                          # headless rendering
```
After the smoke suite passes, **pin both submodule commits** and record SHAs in `docs`-free form in `runs/*/git_sha.txt` and `patches/README.md`.

## 3. Smoke suite (all must pass → G0)
| ID | Check |
|---|---|
| S0 | `lerobot-eval` of a published SmolVLA/LIBERO or Meta-World checkpoint runs a few episodes headless |
| S1 | Meta-World task builds; scripted expert solves it at nominal dynamics; `apply_dynamics` read-back shows changed friction/mass/solref |
| S2 | MuJoCo state save/restore (qpos, qvel, act, **mocap pos/quat**) reproduces a 20-step trajectory exactly |
| S3 | `scripts/inspect_model.py` prints the SmolVLA module tree and writes `configs/model/smolvla_sites.yaml` (module names are [R] until this runs) |
| S4 | One batch: forward → toy projection → backward → optimizer step → checkpoint save/load → 20-step rollout |
| S5 | `scripts/profile_vram.py` reports peak memory for (a) critic train (b) LoRA train + projection (c) inference + projection |
| S6 | Toy-env/toy-policy "walking skeleton" runs the whole pipeline on CPU in CI in < 2 min |

## 4. Makefile targets
`setup` · `test` (unit+integration, CPU) · `test-sim` · `test-gpu` · `lint` (ruff+mypy) · `smoke` · `exp EXP=configs/exp/<name>.yaml` · `report RUN=<dir>`.

## 5. Upstream modification policy
Wrap, don't edit. If LeRobot lacks a hook (e.g. per-denoising-step activation access, LoRA wrappers), implement it in `policies/smolvla.py` via forward hooks or subclassing; only if impossible, create `patches/NNN-description.patch` + rationale + upstream issue link. Re-apply patches in `make setup`.

## 6. Reproducibility metadata (written to every run dir)
`config.yaml`, `git_sha.txt` (repo + submodules), `pip_freeze.txt`, `gpu.txt`, `seeds.json`, `data_manifest_hash.txt`.
