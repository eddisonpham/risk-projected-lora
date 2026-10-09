# 09 — Evaluation Suite

Modules under `src/riskproj/eval/`; every evaluator consumes **archived episode/decision logs**, not live objects, so new metrics can be recomputed offline.

## 1. Modules and metrics
| Module | Metrics |
|---|---|
| `critic_eval` | AUROC, AUPRC, Brier, ECE + reliability curve, FNR@τ, per-type and per-cell breakdown, ensemble-disagreement vs. error, gradient diagnostics (`06` §4) |
| `rollout` | **Primary: success rate** per cell and aggregated (OOD-combo, OOD-range, ID separately); violation rate per type (V1–V5); time-to-success; progress at timeout; failure class (FACT taxonomy [V]: rim-jam/precision, stop-short/force, slip/drop, idle-timeout) |
| `counterfactual` (**E1**) | From N saved states per cell: execute original vs. corrected chunk; paired violation difference; compare to C1 norm-matched random correction |
| `diagnostics` | correction rate; ‖Δh_safe‖/‖h0‖; unusable-gradient rate; K used; nonlinear vs. linearized post-projection risk (linearization error); fallback F1/F2/F3 rates; critic-ensemble std at decisions |
| `anti_gaming` | do-nothing (zero-action) policy success/violations; action norm, jerk, path length; stall rate; success-vs-violation Pareto; fallback-conditioned success |
| `proxy_gold` | critic-risk reduction vs. measured-violation reduction as a function of correction norm / K / step size (exploitation curve [Va Gao]) |
| `systems` | latency per decision (CUDA events: policy / critic / projection), peak VRAM, trainable params, steps/s |
| `nominal_regression` | ID (nominal-cell) success after Stage B vs. π_base; action drift ‖chunk_B − chunk_base‖ on nominal states |

## 2. Statistics (`eval/stats.py`)
- Per-cell success: Wilson interval [R]. Paired method comparisons (same seeds/CRN): exact McNemar or paired bootstrap; unpaired: Fisher's exact (FACT's choice [V]).
- Aggregates over cells × training seeds: stratified bootstrap CIs, IQM, probability of improvement via `rliable` [V]. Training seeds are the replication unit.
- Multiplicity: Holm–Bonferroni [R] over the **pre-registered primary comparisons** only (B3 vs B1, B2, B4a); everything else labeled exploratory.
- Sample size: start with 40 eps/cell [V-FACT default]; after the pilot, estimate discordant-pair rate and choose n to detect the MPCD at 80% power; write n into the prereg tag.
- Never pool TEST-combo with TEST-range; never report only the best seed/cell.

## 3. Log schema (JSONL; one file per run)
**Episode:** `run_id, method, train_seed, env_seed, cell_id, split, dyn_readback{mu,mass,solref}, success, steps, t_success, violations{V1..V5:{first_step,count}}, failure_class, fallback{F1,F2,F3}, stall_steps, action_norm_mean, jerk, proj{n_dec,n_active,n_unusable,mean_corr_rel,mean_r_before,mean_r_after,mean_r_pred}, latency_ms{policy,critic,proj}, peak_vram_mb`.
**Decision:** `ep_id, step, flow_step, tau, r_cand, gn2, active, usable, K_used, corr_rel, r_after[k], r_pred[k], ens_std, fallback_code, true_violation_next_H`.

## 4. Required tables / figures
T1 critic metrics (VAL, TEST) · T2 main success + violations per method (combo / range / ID) · T3 factorial (T×I) · T4 controls C1–C3, B5, B6, B1-LN · T5 systems cost · F1 linearization error vs. step · F2 E1 paired counterfactual · F3 proxy–gold curve · F4 per-cell heatmap by method · F5 depth sweep S0→S5 · F6 success–violation Pareto · F7 failure-class stacks · F8 reliability diagram.

## 5. Interpretation guards
Gains with high F2/F3 fallback or stall rate are **not** adaptation gains. Success must be reported alongside violations and fallback rate. A critic-score drop without a measured-violation drop is a negative result about the critic, not a positive one about the method.
