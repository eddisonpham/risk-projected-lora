# Research Log (append-only)

## OPEN QUESTIONS (seeded)
1. SmolVLA module names / expert block parity — resolve via `scripts/inspect_model.py` (S3).
2. Peg-insert scripted-expert class name and task-critical geoms — resolve via S1 / `inspect_contacts.py`.
3. τ convention in LeRobot's SmolVLA code (assumed τ=1 noise) — resolve via flow-convention unit test.
4. Does LeRobot expose PEFT/LoRA hooks, or is HF PEFT needed?
5. Novelty search (`02` §E) not yet run.
6. Does `exec_horizon=10` lose success vs. per-step re-planning? (E0)
7. **Arm labelling for method variants.** An arm is resolved by `MethodCfg.id` through the
   method registry, and `ExperimentCfg` rejects duplicate ids, so a *variant of an existing
   method* cannot be expressed as its own `methods:` entry. E2 originally named `B3-K3` and
   `B3-logit` (A10/A13 sweep points); those ids are not registered, so the run would have died
   at construction. Reduced E2 to a single B3 arm (the sweep axis is per-decision step size,
   which the trace already carries), and E0's `B0-replan` arm was removed for the same reason.
   Options when the sweep is actually needed: add an `arm_label` field to `MethodCfg` and key
   the report on it, or register the variants as methods. Not needed before E4.
8. **`exec_horizon` is not config-expressible.** The 10-vs-1 re-planning comparison behind
   OPEN QUESTION 6 lives in `RolloutSpec.exec_horizon` (default 10), which no YAML field
   reaches. E0 therefore answers only its first question until the runner exposes it. A
   comparison that silently is not run would be worse than one explicitly deferred.
9. **S0 provisioning.** `lerobot` is installed but without its dataset extra
   (`pip install 'lerobot[dataset]'`); S1–S3 need `python tasks.py bootstrap-sim`. G0 cannot be
   certified on this box without those, and none of them is a code defect.

## Entries

### 2026-10-09 — Phase-0/1 integration: seam repair across the parallel build (no experiment)
**Loop stage:** reflect → revise. No hypothesis was tested and no metric was produced; this
entry records integration work only, so nothing here may be cited as evidence for or against
the claim under test.

**What was done.** The layers built in parallel had never been run against each other.
`pytest` went from 7 failures + 7 errors to **499 passed, 11 skipped, 0 failed**, and the smoke
suite from 1 pass to **S4, S5, S6 pass** (S0–S3 skip on missing provisioning, see OPEN QUESTION 9).
`tasks.py test` and `tasks.py smoke` are the commands behind `make test`/`make smoke`.

**Seams found and fixed** (each was a real defect, not a test artifact):
- `Task.expert()` returned a non-callable object; the demos generator calls `expert(obs)`
  (`04` §7). `ToyExpert` now implements `__call__`; the contract test asserts `callable()`.
- `f_act` was called with a bare dict of flow state in `smoke_suite`/`profile_vram`; it needs the
  adapter's own context (`ctx.flow()`, `05` §4). Both now use `policy.make_context`.
- `CtxBuilder.context()` padded the **action** history to the **proprio** width, so a stored
  history block did not divide by the action dim (`gen_critic_data` → `schema._unflatten_history`).
  The builder now tracks the action width, and `action_dim` is passed where it is known.
- `train_critic` could not consume `CriticDataset` (`(ctx, chunk, any, per_type)`) although that
  is D-crit's on-disk format; it now adapts it through the critic's own `FeatureSpec`, so the
  training and evaluation feature layouts cannot drift apart.
- The violated-chunk labeller called `ViolationMonitor.update`; the monitor's API is `observe`.
  More importantly it re-derived force/penetration from the observation. Both envs now expose
  `last_step_info()` — the mapping their own monitor consumed — so the violation conventions
  (signs, windows) exist in exactly one place.
- C3's permuted-critic control applied its permutation in `_risk_fn`, which `_correct` never
  calls, so C3 silently equalled B3 — a broken control that would have biased A1 *toward* the
  hypothesis. The permutation now lives in `_correct`, and a batch of one (no derangement
  possible) logs `c3_degenerate` instead of silently degrading.
- `decision_frame`'s `mean_linearization_error` averaged over rows with no correction, where the
  step is exactly zero: it was biased toward zero in proportion to the fallback rate.
- Three test-side interface guesses were corrected (`build_tables`/`evaluate_method`/`RolloutSpec`
  in the walking skeleton, calibration on the any-violation column, constant-risk projection case)
  and one vacuous contract assertion (`Task.expert()` was checked with `isinstance(..., object)`)
  was made real. `test_registered_methods_build_from_the_shipped_arm_configs` now builds every
  shipped arm and fails on any config naming an unregistered method id — that check is what
  caught OPEN QUESTIONS 7 and 8.

**Observations that are not results.** At smoke scale the critic dataset is class-degenerate
(20 tuples, 0 positives), so the overfit sanity check in the walking skeleton skips itself rather
than reporting a meaningless AUROC; the smoke pre-registration has a real 15% positivity guard
that only fires at generation scale. The toy critic is perfectly separable (AUROC 1.0), so its
BCE-optimal temperature is always "as sharp as possible": the softening branch of temperature
scaling is untestable on the clean fixture and is now tested with label noise instead.

**Next.** Provisioning (S0–S3) so G0 can be certified, then G1 (`sensitivity_sweep.py`) and E0 —
which is still missing its re-planning half (OPEN QUESTION 8).

### 2026-10-09 — Phase-0 provisioning and the Meta-World/SmolVLA integration (no experiment)
**Loop stage:** 5 implement → 6 run (smoke suite only). No hypothesis was tested; nothing here is
evidence for or against the claim. What changed is that the *simulator and the real model now run*,
which several [R] items in `02`/`04`/`05` were waiting on.

**Environment.** `metaworld==3.1.1`, `lerobot==0.6.0`, `datasets`, `num2words`, `torchcodec` into
`.venv` (torch kept at 2.11.0+cu128, CUDA available). GPU is an RTX 5060 Laptop (8151 MiB total,
7.96 GB) — the 7 GB budget is 88% of the card, so G2 has very little slack and needs the real
measurement before Stage A. `lerobot`'s install **downgraded numpy 2.5.2 → 2.2.6 and pandas
3.0.6 → 2.3.3** (its own pins); the suite is green on those (see below).

**Smoke status (S0–S6).** Before: S4/S5/S6 pass, S0–S3 skip. Now: **S1, S2, S3, S4, S5, S6 pass**;
S0 still skips, for an upstream reason recorded below. G0 is therefore *not yet* certified
(one non-blocking check outstanding), and no Phase-1+ experiment may cite G0 as passed.

**Seams found and fixed** (all were real defects or real [R]-resolution work; each has a test):
- **Meta-World v3 has no per-task gym id.** 3.1.1 registers only vectorized benchmark ids
  (`Meta-World/MT1` + `env_name`), so every `GYM_ID_TEMPLATES` candidate raised. A single task env
  is built the way LeRobot builds it (`metaworld.MT1(task).train_classes[task]` +
  `set_task(train_tasks[i])`), and the per-episode task is chosen from a seed-derived RNG so
  initial states are both diverse and reproducible (`build_single_env`, `_select_goal`).
- **Task-critical geoms did not exist yet**, so nothing could run: `inspect_contacts.py` needs an
  env, the env needed dynamics, dynamics needed `task_geoms.yaml`. A *nominal* spec now no-ops with
  an explicit read-back note (the discovery run must be possible), while a non-nominal spec still
  fails loudly. The discovery then ran and the file is now `configs/env/task_geoms.yaml`: friction on
  {`peg`, `leftpad_geom`, `rightpad_geom`, `leftclaw_it`, `rightclaw_it`, `geom39`, `geom43`}, mass on
  body `peg`. **The auto top-8 selection was rejected after reading it**: it included the table
  (`geom3`) and the hand mesh (`geom27`), which are not in the peg/aperture/gripper chain (friction
  there moves support, not the manipulation), and it proposed bodies `[mocap, peg, leftclaw]` (two of
  which have zero mass, so the payload axis would have been a silent no-op). The curation is recorded
  in the file's `provenance.selection_note`, and the discovery script now takes `--geoms/--bodies`.
- **Unnamed geoms are addressable only positionally.** The peg's aperture collision boxes are nameless,
  so the discovery output uses `geom39`/`geom43`; the resolver now accepts that token (validated:
  in range, and *actually* unnamed) instead of raising.
- **`stiff_scale` read-back crashed on any real model** (`float(mean/array)`): the achieved stiffness
  scale is now the mean of the per-geom `(tc_base/tc_eff)²`, i.e. the number a geom actually has.
  The unit tests could not catch this because they exercise the delegating path, not MuJoCo.
- **Task-name matching.** A Meta-World env reports its *class* name (`SawyerPegInsertionSideEnvV3`)
  and a config uses the *registered id* (`peg-insert-side-v3`) — and normalizing does not help, since
  the class says "Insertion" and the id says "insert". The mismatch guard now requires the id's
  tokens to appear in the class name, so the shipped geoms file is accepted while `assembly-v3` is
  still rejected.
- **`restore_state` returned no observation for the real env**, which is exactly the contract that
  stops counterfactual labelling from using a stale observation. `MetaWorldDynEnv.get_obs()` now
  re-reads the raw observation from the simulator; `step()` also returns the same force/penetration/
  grip-offset mapping the violation monitor consumed (previously only the monitor saw them, so the
  threshold-derivation script silently read zeros).
- **SmolVLA's per-denoising-step entry point was not where the guesses looked.** Verified against
  `lerobot/policies/smolvla/modeling_smolvla.py`: it is
  `VLAFlowMatching.denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)` on the **inner**
  model, not a method on the policy. `EXPERT_ENTRY_ROOTS` now searches `model.<name>` first and keeps
  the original guesses after it. A miss is recorded (`_step_entry_error`) rather than raised at load,
  because S3 must be able to walk the tree in order to *resolve* names — anything that needs a step
  still fails loudly.
- **Site discovery was silently wrong, twice.** (i) The adapter is an `nn.Module` holding the policy
  as `self._model`, so the discovery script (which fell back to `policy`) wrote patterns with an extra
  `_model.` prefix into the sites file while the adapter matched against the unwrapped tree: **every
  pattern failed to resolve and the run would have fallen back to heuristics with a file that looked
  authoritative.** Fixed with a `site_tree()` contract on the adapter (`PolicyAdapter` protocol +
  base default + SmolVLA override). (ii) Pattern matching unioned all candidate patterns and then
  tie-broke on block index, which put S2 ("last expert block's `mlp.down_proj`") in the *VLM text
  stack* — 16 identically-shaped blocks, deleted by an alphabetical tie-break. Matching is now
  **priority-family**: the first pattern that matches anything defines the family, and only then does
  the documented index tie-break apply. Verified SmolVLA patterns and their parity rules are recorded
  in `policies/sites.py`; the resolved file is `configs/model/smolvla_sites.yaml` (S1–S6 all resolve,
  0 errors).
- **The flow convention is verified, not recalled** (`05` §4 [R] → [V]): `x_t = τ·ε + (1−τ)·a0`,
  target `ε − a0`, Euler from τ=1 with `dt = −1/num_steps`. τ=1 is noise, as `flow.py` assumes.
- **`draccus` (a lerobot dependency) ships an empty top-level `tests` package**, which shadowed this
  repository's `tests` directory under PEP 420 and broke `from tests.tolerances import ...`. `tests/`
  now carries an `__init__.py` and `tests/conftest.py` puts the repo root first on `sys.path`.

**Blocked, with evidence (not a pass by omission).**
- **S0** (`lerobot-eval` on a published SmolVLA checkpoint) still skips. Three upstream issues, in the
  order they appeared: PATH resolved a *different* Python's `lerobot-eval` (a conda env without the
  `dataset` extra) — the check now resolves the console script next to `sys.executable`; LeRobot's
  MetaWorld default task is `metaworld-push-v2`, whose stripped name is absent from its own
  `metaworld_config.json` (`KeyError: 'push-v2'`) — the check now names `metaworld-peg-insert-side-v3`;
  and both cached revisions of `lerobot/smolvla_base` predate LeRobot 0.6.0's processor format
  (`ProcessorMigrationError: Config file 'policy_preprocessor.json' not found`, and the migration
  script it suggests is not shipped in the wheel). Our *own* loader loads the same published
  checkpoint successfully in S3, so what is blocked is LeRobot's eval driver, not the pipeline. A
  fresh revision of the checkpoint is the fix; it needs a download we did not spend this iteration's
  budget on.
- **`third_party/` submodules were not added.** `metaworld`/`lerobot` were installed from PyPI at
  pinned versions instead of as pinned submodules (`03` §2). The spec's intent (wrap, never edit) is
  satisfied — nothing under `third_party/` is touched, and `patches/` stays empty — but the *pinning
  mechanism* is a `pyproject`/env-lock rather than a submodule SHA. Logged as OPEN QUESTION 10.

**Observations that are not results.** `S1`'s detail line used to read `success=False` for every
rollout because it read an attribute the wrapper does not have (the value lives in the step info);
it now reports the real success flag. The smoke `S5` VRAM numbers are the *toy* config's (17.4/17.8 MB)
and say nothing about G2 — the real measurement is P1 work on `configs/model/smolvla.yaml`.

**Tests.** `pytest -m "not sim and not gpu and not slow"`: **496 passed, 1 failed → 0 failed** after
the config-hygiene fix (`configs/env/task_geoms.yaml` and `configs/model/*_sites.yaml` are discovered
*data* artifacts, not config fragments, so they are excluded from the fragment sweep). `pytest -m sim`:
**5 passed** (dynamics read-back verified at μ=1.5, m=2.0, k=4.0; idempotent across resets; the
scripted expert solves nominal episodes; 20-step state replay exact to 1e-12; contact read-back lists
pairs). New tests: `tests/unit/test_task_geoms.py` (8) and a verified-shape site-resolution test in
`tests/unit/test_sites.py`.

**Next.** (1) G1's threshold half: `scripts/derive_violation_thresholds.py` on nominal expert rollouts,
freeze `configs/violations.yaml`, then the expert-only half of the sensitivity sweep (the π_base band
criterion cannot be evaluated before Stage A exists — it must be reported as *not yet evaluable*, not
as passed). (2) G2 for real: `profile_vram.py` with `configs/model/smolvla.yaml`. (3) The SmolVLA
adapter still needs the chunk-level prefix/KV plumbing (`embed_prefix` + `vlm_with_expert.forward(
fill_kv_cache=True)`) before `tail_velocity` can run at all — that is the next implementation slice,
and it is a prerequisite for Stage A (`P2`).

### OPEN QUESTIONS (updated)
9. **S0 provisioning.** `lerobot` 0.6.0 is installed with the `dataset` extra; S0 is blocked by the
   `smolvla_base` *checkpoint revision* (pre-processor-format) and needs a fresh download.
10. **Submodules vs PyPI pins.** `third_party/lerobot` and `third_party/Metaworld` do not exist; the
   versions are pinned in `pyproject.toml` and `env/lock.txt`. Either add the submodules at the
   installed versions or amend `03` §2 — decide before the write-up, since `03` claims submodules.
11. **Site parity depends on `self_attn_every_n_layers=2`.** The S3/S4 mapping (odd = cross-attention)
   is verified for the shipped config; a config change to that value must invalidate
   `configs/model/smolvla_sites.yaml`, which nothing currently enforces.
12. **Probe/adapter action dim.** SmolVLA's chunk lives padded to `max_action_dim` (32) while the task
   has 4; the adapter's `action_dim` must be decided before Stage A (the site→action path, the critic
   horizon and `f_act` all assume one width).
