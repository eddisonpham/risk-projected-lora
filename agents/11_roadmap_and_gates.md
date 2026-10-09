# 11 — Roadmap and Gates

Do not start a phase until the upstream gate passes. Gate thresholds live in `configs/gates.yaml`, committed before use.

| Phase | Work | Gate (pass condition) | On fail |
|---|---|---|---|
| P0 | Repo, submodules, smoke suite S0–S6 (`03`) | **G0** all smoke tests pass; CI green | fix env; do not proceed |
| P1 | Dynamics knobs + read-backs, violation thresholds, sensitivity sweep (`04` §8); VRAM profiling | **G1** sensitivity criteria met; **G2** ≤7 GB for critic train, LoRA+proj train, inference+proj | switch task / reduce per `05` §6 |
| P2 | Stage A SFT; **E0** reproduce baseline (π_base nominal success; exec_horizon 10 vs. 1 shows no loss); generate D-B, D-crit | E0 nominal success in G1 band | adjust SFT; re-check data |
| P3 | Critic training, calibration, gradient diagnostics (`06`) | **G3** critic acceptance | fix labels/features/data; never tune the method around it |
| P4 | Projection operator + unit tests (`07`); **E2** linearization-error study | **G4** tests green; E2 shows usable first-order region | if no usable region → stop; report |
| P5 | **E1** counterfactual validity (corrected vs. original chunk from identical states; vs. C1) | **G5** real violation reduction, CI excludes 0, > C1 | stop or redesign; closed loop is pointless otherwise |
| P6 | **E3** pilot: B0, B1, B1-LN, B2, B3, B4a, B5, B6 + A1–A3 on 3 VAL/TEST-style cells, 1 seed | **G6** no action suppression, fallback <30% [H], bugs triaged, MPCD and n set | iterate on VAL only |
| P7 | **E4** pre-registered TEST run (≥3 seeds); **G7**: tag `prereg/E4`, freeze configs/seeds/thresholds, run once | **G7** protocol frozen before first TEST episode | any post-hoc change ⇒ label exploratory |
| P8 | Wave-2 ablations (`10`); audit; write-up (`12`) | audit note complete | — |

## Experiments
**E0** baseline reproduction · **E1** counterfactual validity · **E2** linearization/gradient quality · **E3** falsification pilot · **E4** main pre-registered comparison · **E5** ablation waves.

## Stop / pivot rules
Stop after G3 fail (critic invalid), G5 fail (corrections don't reduce real violations), or E4 null that survives controls — and write up the negative result. Pivot to the FACT-confound or critic-exploitation study if A5/A8 are the strongest findings. Escalate budget overruns to the human.
