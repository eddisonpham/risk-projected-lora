# 10 — Ablation Catalogue

Each row isolates one component. **Wave 1** (run on VAL cells, 1 training seed, after G5) is cheap and decides whether to continue; **Wave 2** runs only if wave 1 survives. Report every ablation, including nulls.

## Wave 1 — "Does the idea hold up?"
| ID | Question | Comparison | If… then… |
|---|---|---|---|
| A1 | Does the critic gradient matter at all? | B3 vs C1 / C3 | C1≈B3 → effect is perturbation/regularization, not risk information |
| A2 | Is it just shrinkage toward the base? | B3 vs C2 | C2≈B3 → "preserve base behavior" story, not constraint |
| A3 | Is it just gating? | B3 vs B6 | B6≈B3 → fallback does the work |
| A4 | Selection vs. gradient? | B3 vs B5 | B5≥B3 → gradients unnecessary (flow models) |
| A5 | Noise-schedule confound? | B1 vs B1-LN; B3 vs B3+LN | B1-LN≈B3 → FACT explains gains; B3+LN > B1-LN → complementary |
| A6 | Latent vs. action? | B3 vs B4a vs B4b | B4≈B3 → latent adds nothing; report it |
| A7 | Regularizer or filter? | B3 vs B1+TTP vs B3-trainonly | decomposes training vs. inference effect |
| A8 | Is the critic exploitable? | proxy–gold curve over K/step size | divergence point sets a trust-region bound |

## Wave 2 — "Why and where?"
| ID | Factor | Levels |
|---|---|---|
| A9 | **Depth sweep** | S0, S0f, S1, S2, S3, S4, S5 (S6 stretch) |
| A10 | Projection steps m / τ_proj | {1,3,10} / {0.2,0.3,1.0} (train and test separately) |
| A11 | Lookahead | one-step terminal estimate vs. full remaining flow |
| A12 | K and line search | K∈{1,2,3}; backtracking on/off |
| A13 | **Constraint space** | probability vs. logit [H] |
| A14 | Margin / trust region ρ | {0, small} / {none, ρ grid} |
| A15 | Backward mode | ste / projected / second_order (`07` §3) |
| A16 | Penalty hybrid | B3 vs B3+B2 penalty; fixed λ vs. dual ascent (B2-L) |
| A17 | Critic data | expert-only vs. +policy rollouts vs. +perturbed/unsafe; size curve; refresh round |
| A18 | Critic inputs | proprio-only → +history → +vision embedding → +privileged sim force (flagged) |
| A19 | Critic ensemble/aggregator | single vs. mean vs. mean+kσ vs. max; calibration on/off |
| A20 | Label horizon vs. exec horizon | H_c = exec ∈ {5,10,20}; mismatched (diagnostic) |
| A21 | LoRA rank / placement | r∈{4,16,64}; last-4 vs. all blocks; independent of site |
| A22 | Adapter form | LoRA vs. LoReFT-style intervention [Va] |
| A23 | Model class | SmolVLA vs. ACT (deterministic) [+ VLA-Adapter if budget allows] |
| A24 | Flow steps | 5 / 10 / 20 at inference |
| A25 | Dynamics-axis decomposition | gains per axis and interactions (μ×m, μ×k, m×k) |

## Candidate "interesting revelations" (hypotheses, not predictions)
1. Gains vanish under B1-LN (A5) → write the contact-rich-fine-tuning-confound paper instead.
2. Latent ≈ action across depth (A9) → a clean negative result about depth.
3. Gating/selection (B5/B6) match projection → the critic is useful, the gradient is not.
4. Logit-space projection rescues small-gradient failures (A13).
5. Training-time projection helps through the backward Jacobian even if inference-time projection is neutral (A7, A15).
6. Critic exploitation sets in at a measurable correction norm (A8), giving a principled trust region.
7. One shift axis (likely stiffness or mass) carries the whole effect (A25).

## Run order
A1–A3 and E1 first (cheapest, most decisive) → A5 → A6/A7 → A4/A8 → wave 2 by information-per-cost.
