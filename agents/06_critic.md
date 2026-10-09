# 06 — Risk Critic

Grounding: learned safety critics from offline data (Recovery RL [Va]); proxy-vs-gold over-optimization (Gao et al. [Va]); linearized safety signals (Dalal [Va]).

## 1. Definition
`V_φ(c, â) → logits z ∈ R^{1+5}` (any-violation + V1–V5); `r = σ(z_any)`. Inputs are the **deployable set** (`04` §5). Candidate chunk is the first `H_c=10` actions (`exec_horizon == H_c`, asserted). Label = event occurs within H_c steps when the chunk is executed open-loop from the saved state (`04` §6).

## 2. Data and anti-leak rules
- Mix per `04` §7 (D-crit). Train on **TRAIN cells**; calibrate/select on **VAL cells**; TEST cells never touched.
- Split by cell and seed. Dedupe near-identical (state, chunk) pairs across splits. Log per-cell positive rates.
- **Critic refresh (optional, [H]):** after B1 exists, add B1 rollouts from TRAIN/VAL cells only and retrain; re-run G3. Needed because adapters shift the proposal distribution.

## 3. Architecture and training (defaults [H])
MLP 3×256, LayerNorm (no BatchNorm — per-example independence is required by `07`), **deep ensemble M=5** (Lakshminarayanan [R]) with different seeds/bootstraps; BCE per head with class weights; optional gradient-norm penalty or spectral norm (Miyato [R]) to keep input-gradients informative. Calibrate with temperature scaling on VAL (Guo [R]); report ECE before/after. Trained once, **frozen** before Stage-B projection experiments; its parameters never receive task gradients (assert `requires_grad=False`).

## 4. Gradient-quality diagnostics (log for every critic)
Input-gradient norm distribution wrt chunk **and** wrt site activation (through `f_act`); fraction with `‖g‖² < η`; saturation rate (`|z|>6`); gradient direction agreement across ensemble members (cosine); linearization error `r(h+u) − (r + gᵀu)` over a step-size sweep (this is **E2**). Poor gradient quality is a *finding*, not something to patch silently.

## 5. Acceptance gate G3 (declare in `configs/gates.yaml` BEFORE any policy evaluation; values below are starting suggestions [H])
On VAL (held-out combos): AUROC ≥ 0.85 and AUPRC ≥ 2× base rate for any-violation; ECE ≤ 0.05; FNR at the chosen τ ≤ 0.15 for each critical type; no cell with AUROC < 0.75 (else analyze before proceeding); ensemble disagreement higher on VAL than TRAIN (i.e., it carries OOD signal). Fail → fix data/labels/features; do not "tune the method around" a bad critic.

## 6. Choosing τ and the aggregator
τ selected on VAL by the predeclared rule (precision/FNR trade-off, or conformal risk control [R]), then frozen. Aggregator ablation: ensemble mean vs. mean+k·std vs. max (pessimistic). Reported separately: logit-space vs. probability-space constraint (`07`).

## 7. What the critic is NOT
Not a success predictor, not evidence of physical validity, not used as an evaluation metric anywhere.
