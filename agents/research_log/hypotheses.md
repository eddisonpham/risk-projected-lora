# Pre-registered Hypotheses
Priors are subjective [H]; update them in `LOG.md` and never delete the original.

| ID | Hypothesis | Test | Refuted if | Prior |
|---|---|---|---|---|
| H1 | Critic meets G3 on held-out combo cells | E-critic | AUROC/ECE/FNR gates fail | 0.70 |
| H2 | Critic gradients through `f_act` are informative (small linearization error in a usable step range) | E2 | no usable first-order region; saturation dominates | 0.60 |
| H3 | Projected chunks reduce **measured** violations vs. original and vs. norm-matched random | E1 | CI includes 0 or ≤ C1 | 0.55 |
| H4 | B3 > B1, B2 on OOD-combo success (or equal success, fewer violations) | E4 | within CI of B2/B1 with no violation benefit | 0.30 |
| H5 | B3 ≥ B4a/B4b (latent adds value over action-space) | A6/A9 | B4 ≥ B3 within CI | 0.25 |
| H6 | B3's gain is not explained by gating, shrinkage, LN schedule, or resampling | A2–A5 | any control matches B3 | 0.35 |
| H7 | Training with projection matters beyond inference-time projection | A7 | B1+TTP ≈ B3 | 0.40 |
| H8 | Overhead is acceptable within 7 GB and agreed latency budget | systems | G2 fail or >budget | 0.70 |
| H9 | Logit-space constraint reduces unusable-gradient rate vs. probability space | A13 | no difference | 0.50 |
