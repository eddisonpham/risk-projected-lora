# agents/ — Project Operating Manual

**Project:** Risk-Guided Activation Projection for LoRA Fine-Tuning of Contact-Rich VLAs
**Spec date:** 2026-10-08 · **Budget:** ≈7 GB VRAM · **Status:** plan only — no results exist.

## Reading order

| # | File | What it gives you |
|---|---|---|
| — | `skills/vla_researcher.md` | Your research role and the recursive loop |
| — | `skills/software_engineering.md` | Architecture, testing, scaling rules |
| 01 | `01_research_spec.md` | Condensed spec: claim, math, claim boundaries, decision rules, what changed vs. the original plan |
| 02 | `02_literature_grounding.md` | Verified papers/repos, what each grounds, novelty-risk list |
| 03 | `03_repo_setup.md` | Forks, environment, layout, smoke tests |
| 04 | `04_env_dynamics_data.md` | Simulator, task, dynamics knobs, splits, violation events, datasets |
| 05 | `05_models_and_layers.md` | Model tiers, exact layers/sites, flow-matching adaptation, memory |
| 06 | `06_critic.md` | Risk critic: data, labels, architecture, acceptance gate |
| 07 | `07_projection_operator.md` | Operator, variants, backward-pass analysis, unit tests, logging |
| 08 | `08_methods_and_training.md` | Baselines B0–B6, controls C1–C3, train×inference factorial, tuning rules |
| 09 | `09_evaluation_suite.md` | Metrics, statistics, log schema, counterfactual validity, systems cost |
| 10 | `10_ablations.md` | Ablation catalogue isolating each component; wave 1 vs wave 2 |
| 11 | `11_roadmap_and_gates.md` | Phases P0–P8, gates G0–G7, stop rules |
| 12 | `12_reporting_and_audit.md` | Card templates, claim-language rules, audit note |
| — | `research_log/hypotheses.md`, `research_log/LOG.md` | Pre-registered hypotheses; append-only lab notebook |

## Evidence tags (used everywhere)

| Tag | Meaning |
|---|---|
| **[V]** | Primary source read in the relevant sections during spec writing (2026-10-08) |
| **[Va]** | Only abstract / README / doc snippet read |
| **[V2]** | Secondary source only (blog, aggregator, third-party copy) |
| **[R]** | Recalled from memory — verify before relying on or citing |
| **[H]** | Hypothesis or design choice introduced by this spec — not from the literature |
| **[U]** | Taken from the original user plan; not re-verified |

Rule: nothing tagged [R], [V2], [U] or [H] may be stated as established fact in a write-up.

## Glossary
- **c_t** deployable context (observation features, proprioception, short history, instruction). **h0 / Δh / h_c** frozen activation / LoRA-induced change / candidate activation at a *site*.
- **r(h)** critic risk of the action chunk decoded from activation h. **τ** risk threshold. **η** minimum squared gradient norm. **K** max projection iterations.
- **π_base** Stage-A policy (SmolVLA + nominal-dynamics SFT), frozen. **Stage B** LoRA adaptation on dynamics-diverse demos.
- **Cell** one (friction, mass, stiffness) level combination in the 3×3×3 grid (`04`). **TRAIN / VAL / TEST-combo / TEST-range** split roles.
- **Site** a module output where projection acts (`05`). **B-IDs** methods (`08`). **C-IDs** negative controls (`08`). **G-IDs** gates (`11`). **E-IDs** experiments (`11`).

## Where the interesting science is (read before designing anything)
1. Does the *critic gradient* carry causal information about real violations (E1 counterfactual test), or does the correction act as noise/regularization (C1–C3)?
2. Is latent depth special, or is it a continuum with action-space correction (site sweep S0→S5)?
3. Is any contact-rich gain explained by flow-matching **noise-schedule** effects instead (FACT's Logit-Normal control, B1-LN)?
4. Does training *with* the projection matter, or is it just an inference-time filter (train × inference factorial)?
5. How far can the critic be optimized before its predictions and simulator truth diverge (proxy–gold curve)?
