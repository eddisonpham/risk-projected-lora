# 04 — Simulator, Dynamics, Splits, Violations, Data

## 1. Choices
| Item | Choice | Why | Tag |
|---|---|---|---|
| Simulator | **Meta-World v3 on MuJoCo**, through LeRobot's env/dataset path | Native LeRobot env; SmolVLA reference results exist; scripted experts for every task; CPU physics → zero VRAM for sim; all three shift axes are editable MuJoCo model fields | [Va] |
| Primary task | `peg-insert-side-v3` | Insertion: jam / over-force / stop-short failures (FACT's precision-failure signature [V]) | [H] |
| Backups (if G1 fails) | `assembly-v3`, `hand-insert-v3`, `box-close-v3`; then ManiSkill3 `PegInsertionSide` (set mass/friction in `_load_scene` [Va]) | | [Va]/[H] |
| Control | Per-episode re-planning vs. chunk execution: **`exec_horizon = critic_horizon = 10`** (config-asserted equal) | Labels are produced by executing H steps open-loop; SmolVLA's sim default is re-plan every action [V], so E0 must show no success loss at 10 | [H] |

**Caveat:** Meta-World's contact richness is limited and LeRobot's authors note sim is hard for contact-rich tasks [Va]. **G1 (sensitivity gate) decides whether this setup can test the hypothesis at all.**

## 2. Dynamics knobs (`envs/dynamics.py`)
`apply_dynamics(env, DynamicsSpec(mu_scale, mass_scale, stiff_scale))` — always **read back and log** effective values; assert they changed.

| Axis | MuJoCo field | Rule | Guards |
|---|---|---|---|
| Friction | `model.geom_friction[g,0]` for task-critical geoms (gripper pads, peg, hole walls, table) | multiply by `mu_scale` | Contact friction mixes two geoms (max by default [R]) → verify via `data.contact[i].friction` read-back |
| Mass | `model.body_mass[b]` **and** `body_inertia[b]` of the manipulated body | multiply both by `mass_scale` | Call `mujoco.mj_setConst(model, data)` afterwards [R] |
| Stiffness | `model.geom_solref[g,0]` (timeconst) of task-critical geoms | `timeconst' = timeconst / sqrt(stiff_scale)` (k ∝ 1/timeconst² [V2]); keep dampratio | Enforce `timeconst' ≥ 2·opt.timestep` [V2]; read back `data.contact[i].solref`; alt: global `opt.o_solref` override with override flag enabled [R] |

Task-critical geoms are **discovered, not assumed**: `scripts/inspect_contacts.py` runs the nominal expert and lists contact pairs by name → `configs/env/task_geoms.yaml`.

## 3. Cell grid and splits (`data/splits.py`, frozen in `configs/dynamics/splits.yaml`)
Each axis has three **levels** (LO/MID/HI), each a sub-range sampled per-episode (so shifts are continuous within a cell). 3×3×3 = 27 cells.

| Role | Cells | Count | Use |
|---|---|---|---|
| TRAIN | all-MID + exactly one axis at LO or HI | 7 | Stage-B demos; critic training |
| VAL | two-axis cells whose two non-MID signs multiply to +1 | 6 | critic calibration; τ, λ, η, K, site, adapter selection |
| TEST-combo | two-axis cells with sign product −1 (6) + all 8 three-axis cells | 14 | **Primary**: compositional generalization |
| TEST-range | single-axis EXT-LO/EXT-HI beyond the TRAIN extremes (6) + 2 compound extremes | 8 | Extrapolation — **reported separately, never pooled** |

Sign convention (LO=−1, HI=+1 per axis) is fixed in the config; justify in the config comment that neither role is systematically easier (check with the scripted expert during G1; if one is, re-draw **before** any model evaluation).
Range values are **placeholders set at G1** from the sensitivity sweep: e.g. friction LO [0.5,0.7] / MID [0.9,1.1] / HI [1.4,2.0]; mass LO [0.5,0.75] / HI [1.5,2.5]; stiffness LO (softer) [0.25,0.5] / HI [2,4]; EXT beyond those. [H]

## 4. Violation events (`envs/violations.py`, thresholds frozen in `configs/violations.yaml`)
Thresholds are derived from **nominal scripted-expert rollouts only**, before any learned-policy evaluation, then frozen. Labels use steps within the critic horizon H_c=10.

| ID | Event | Measured from | Threshold rule |
|---|---|---|---|
| V1 slip | grasped-object pose drift in gripper frame > δ_slip within W steps while grasped | object vs. end-effector pose | 99.5th pct nominal × margin |
| V2 drop | grasp had been established; object–gripper distance or height loss > δ | same | same |
| V3 over-force | Σ normal contact force on task-critical pairs > F_max | `mj_contactForce` over `data.contact` | 99.5th pct nominal × 1.5 [H] |
| V4 penetration | any task-critical contact with `dist < −δ_pen` | `data.contact[i].dist` | **absolute depth**, stiffness-independent (else confounded with the stiffness axis) |
| V5 jam | force > F_jam for ≥ W steps with progress < ε (FACT precision-failure signature [V]) | force + task progress | nominal-derived |
| (episode) V6 stop-short | timeout with peg partially inserted (FACT force-failure analog [V]) | progress at termination | task success definition |

`any_violation = V1∨…∨V5` is the scalar critic label; **per-type labels are always stored**. V6 is episode-level (failure taxonomy only).

## 5. What the critic and policy may observe (deployable set)
Policy: images, proprioception, instruction, short history. Critic: proprioception + end-effector velocity + short action/proprio history + optional frozen visual embedding + candidate chunk. **Not allowed:** true μ/m/k, cell id, object ground-truth pose unless the policy also receives it. Simulated wrist force is a **separate, flagged "privileged-sim" input** used only in ablation E-critic-force (`10`).

## 6. Counterfactual label generation (`envs/state_io.py`)
From a saved sim state (qpos, qvel, act, mocap pos/quat, contact-warmstart not required), execute a candidate chunk open-loop for H_c steps, record V1–V5. Enables (i) critic labels for arbitrary perturbed chunks and (ii) paired E1 tests (original vs. corrected chunk from the same state). Verify exact reproducibility (S2).

## 7. Datasets
| ID | Content | Source | Use |
|---|---|---|---|
| D-nom | ~50 nominal demos of the primary task | `lerobot/metaworld_mt50` (50 eps/task, 2,500 total) [V] filtered to the task, + scripted regeneration at nominal to ≥200 eps [H] | Stage A SFT |
| D-B | successful scripted-expert demos from **TRAIN cells only**, ≥40 eps/cell, metadata = cell id (never a model input) | `gen_demos.py` | Stage B SFT (identical for all methods) |
| D-crit | (state, context features, candidate chunk, per-type labels, cell, seed) tuples | `gen_critic_data.py` | Critic. Mix: expert chunks; π_base / B1 rollouts on TRAIN+VAL cells; Gaussian/low-frequency-perturbed chunks at several σ; deliberately unsafe chunks (over-push, early release). Target ≥15% any-violation positives; record per-type balance. Start ≥50k tuples, scale by learning curve [H] |
| D-ref | `physical-intelligence/libero` (1,693 eps) [V] | optional sanity of the SmolVLA install only | S0 |
Pretrained init: SmolVLA base checkpoint (HF id likely `lerobot/smolvla_base` [R]).

Splits are by **cell and seed**, never by frame. Every dataset gets a data card (generator version, config hash, class balance, cell histogram) and a checksum in `data/MANIFEST.json`.

## 8. G1 — dynamics-sensitivity sweep (`scripts/sensitivity_sweep.py`)
For each axis, ≥7 log-spaced scales over [0.25×, 4×], 30 episodes each for (scripted expert, π_base): log success, per-type violation rates, and the read-backs. **Pass if** [H]: nominal π_base success in [30%, 90%] (headroom both ways); at least one axis shifts success or violations by ≥15 pp across the range; each of V1–V5 occurs in ≥5% of rollouts somewhere; read-backs confirm the physics changed. **Fail →** change task (backup list) before any further work.
