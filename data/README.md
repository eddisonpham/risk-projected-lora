# data — datasets, cards and manifest

Large files live **outside git**. Only this README, `DATA_CARD.md` and `MANIFEST.json` are
committed (`skills/software_engineering.md` §5). `.gitignore` excludes `data/*` and re-includes
`data/MANIFEST.json`, `data/DATA_CARD.md`, `data/README.md` and `data/*.json`.

Every artifact here is written by a generator in `src/riskproj/data/`, and every artifact is
registered with a sha256 in `MANIFEST.json` by
`data.schema.register_dataset` → `utils.io.add_manifest_entry`. A file that is not in the manifest
does not exist as far as an analysis is concerned.

## Datasets (`04` §7)

| ID | Contents | Source | Used for |
|---|---|---|---|
| **D-nom** | ~50 nominal episodes of the primary task | `lerobot/metaworld_mt50` filtered to the task, plus scripted regeneration at nominal to ≥200 episodes | Stage-A SFT (produces `π_base`) |
| **D-B** | successful scripted-expert demos from **TRAIN cells only**, ≥40 episodes per cell; metadata carries the cell id, which is *never* a model input | `data/gen_demos.py` | Stage-B LoRA adaptation — identical for every method |
| **D-crit** | `(state, context features, candidate chunk, per-type labels, cell, seed)` tuples | `data/gen_critic_data.py` | critic training and calibration; ≥15 % any-violation positives, ≥50k tuples initially |
| **D-ref** | `physical-intelligence/libero` (1,693 episodes) | upstream dataset | sanity-only check that the SmolVLA install works (S0) |
| **Manifest** | sha256 + generator version + config hash + record count for every artifact above | `data/MANIFEST.json` | provenance; verified before a report is generated |

D-crit's candidate mix matters and is recorded per tuple: expert chunks; `π_base`/B1 rollout chunks
from TRAIN and VAL cells; Gaussian and low-frequency-perturbed chunks at several σ; and
deliberately unsafe chunks (over-push, early release). The per-source and per-type balance is part
of the data card because a critic trained on one source generalizes only to that source.

## Contract

- **Splits are by cell and seed, never by frame** (`04` §3, `07` §2 of `06`). `data/splits.py`
  derives the TRAIN/VAL/TEST roles from the sign rules; there is no hand-typed cell list to leak.
- **TEST is sealed.** TEST-combo and TEST-range cells are never touched for tuning, calibration,
  threshold derivation or adapter selection. Tuning happens on VAL only (`CLAUDE.md` #2). Roles are
  enforced in code (`assert_no_leakage`, `assert_tunable`, `assert_sealed`-style guards in
  `data/splits.py`), not by convention.
- **Deployable inputs only.** A critic tuple's context uses exactly the `obs.CTX_KEYS` layout
  (`src/riskproj/obs.py`): proprioception, end-effector velocity, short action/proprio history, an
  optional frozen visual embedding, and the candidate chunk. True friction/mass/stiffness, the cell
  id, and object ground-truth pose are **not** inputs. `data.schema.assert_deployable_ctx` raises if
  a tuple carries something the config forbids — including the "privileged-sim" force channel, which
  is an ablation input and must stay off for any headline number.
- **Violation labels are always stored per type** (V1–V5) with `any_violation` as their OR; the
  collapsed label is a convenience, never a replacement for the per-type columns.
- **Thresholds come from nominal scripted-expert rollouts only**, derived before any learned policy
  is evaluated and frozen in `configs/violations.yaml` with the measured percentiles in
  `provenance` (`04` §4).
- **Every dataset gets a data card** (`skills/software_engineering.md` §5): generator name and
  version, config hash, class balance, cell histogram, record count, and the command that produced it.

## Regenerating

```bash
python tasks.py exp EXP=configs/exp/<the experiment that generates the data>.yaml
```

Generation is seeded and idempotent per (config hash, seed): the same config and seed must produce
the same shas, and a change of either must show up as a new manifest entry rather than silently
overwriting an old one.

## Pending

Nothing has been generated yet. `MANIFEST.json` does not exist. The table below is intentionally
blank — fill it in from the generator's returned summary and the manifest entry, never from
expectation.

| Dataset | Version | Config hash | Records | Positives (any) | Manifest sha256 | Command |
|---|---|---|---|---|---|---|
| D-nom | _pending_ | _pending_ | _pending_ | n/a | _pending_ | _pending_ |
| D-B | _pending_ | _pending_ | _pending_ | n/a | _pending_ | _pending_ |
| D-crit | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| D-ref | _pending_ | _pending_ | _pending_ | n/a | _pending_ | _pending_ |
