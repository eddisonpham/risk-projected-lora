# 08 — Methods, Controls, Training Protocol

## 1. Notation
Training mode **T** ∈ {0 (plain LoRA), pen (critic penalty), lat (activation projection in forward), act (action projection)}; inference mode **I** ∈ {0, lat, act-post (S0), act-flow (S0f), bon (best-of-N), gate}. All methods share D-B, step budget, LoRA budget, seeds, τ-selection protocol.

## 2. Methods
| ID | T / I | Definition | Tests |
|---|---|---|---|
| B0 | – / 0 | π_base frozen (Stage A) | starting point; fallback rate |
| B1 | 0 / 0 | + LoRA SFT on D-B | standard PEFT |
| **B1-LN** | 0 / 0 | B1 with FACT Logit-Normal τ schedule (`τ=σ(z−1.5)`) [V] | noise-schedule confound (flow models only) |
| B2 | pen / 0 | B1 + `λ·relu(r(â)−τ)` on the terminal estimate; λ tuned on VAL | simple critic penalty |
| B2-L (opt) | pen / 0 | λ by dual ascent toward a VAL violation budget (CMDP-style [Va SafeVLA]) | adaptive vs. fixed λ |
| **B3** | lat / lat | proposed | — |
| B4a | 0 / act-post (and act / act-post) | action-space projection of the final chunk, Dalal-style [Va] | latent vs. action |
| B4b | 0 / act-flow | critic-guided correction of `x_τ` at last m steps | in-flow action-space |
| B5 | 0 / bon | sample N chunks (different ε), pick lowest critic risk (V-GPS-style [Va]) | gradient vs. selection |
| B6 | 0 / gate | critic only gates: risky → fallback ladder, no correction | gating-only explanation |
| B1+TTP | 0 / lat | test-time projection on plain LoRA | inference filter alone |
| B3-trainonly | lat / 0 | trained with projection, evaluated without | regularizer alone |

**Negative controls (drop-in inside the B3 pipeline, same norm budget):**
**C1** random direction, norm-matched to B3's corrections · **C2** shrink `Δh` toward 0 with matched mean correction norm · **C3** gradient of a critic trained on shuffled labels. If C1–C3 match B3, the benefit is not critic information.

B4a/B4b/B5/B6 are inference-only on B1 unless "act/act-post" is stated. B5 and B4b require a stochastic head (flow matching); not applicable to ACT/VLA-Adapter.

## 3. Protocol
1. **Stage A** (`05` §5) → freeze π_base; verify nominal success (G1).
2. **Stage B** per method × training seed (≥3 for TEST; 1 for pilot).
3. **Selection on VAL only:** τ (via `06`), λ, η, K, m, site, margin, trust region, checkpoint step. Record the grid and chosen values in the run config.
4. **Freeze + tag** `prereg/<exp>`; run TEST once.
5. Tuning budget parity: every method gets the same number of VAL-evaluated configurations (report it).

## 4. Pairing and seeds
Same initial states and same per-step noise ε (CRN) across methods; training seeds are independent replicates; the unit of replication for inference is the **training seed**, not the episode.

## 5. Compute planning
`episodes = Σ methods × training_seeds × cells × eps_per_cell`; time = episodes × measured s/episode (from G2/E0). Default `eps_per_cell=40` (FACT used 40 per task [V]). If infeasible, reduce methods/cells by a rule declared **before** looking at results; never subsample post hoc.
