# 01 — Research Specification (condensed)

## 1. Question, hypothesis, scope
**RQ.** Does locally projecting LoRA-induced activation changes against an independently trained physical-risk critic improve OOD contact-rich manipulation versus standard LoRA, a critic-penalty LoRA, and direct action-space projection?

**H.** For a continuous-action VLA adapted to one contact-rich task, a first-order risk-feasibility projection at an action-relevant hidden activation reduces *simulator-measured* violations and improves — or preserves — OOD success under held-out friction / payload-mass / contact-stiffness combinations.

**Is:** a learned-risk-constrained PEFT experiment; critic labels are simulator outcomes.
**Is not:** a physics-informed network, a safety guarantee, a new VLA architecture, a claim about hardware. Do not call the method novel until the prior-art search in `02` is done.

## 2. Setting
- **Stage A (base).** SmolVLA (pretrained on community SO-100 data [V]) + task SFT at *nominal* dynamics → frozen `π_base`. Rationale [H]: a zero-shot community-pretrained checkpoint cannot act in Meta-World, so "frozen baseline" and "fall back to the unadapted action" would be degenerate (idle action = trivially "safe").
- **Stage B (adapt).** LoRA on `π_base` using dynamics-diverse TRAIN demos. All method comparisons happen at Stage B.
- **Shift axes:** friction, payload mass, contact stiffness (`04`). **Critic:** `V_φ(c_t, â_{t:t+H}) ∈ [0,1]`, deployable inputs only (`06`).

## 3. Method
```
site (one Linear's output):  h0 = W x (frozen),  Δh = s·B·A·x,  h_c = h0 + Δh
candidate chunk:             â_c = f_act(c, h_c)          # flow-matching: terminal estimate, see 05
risk:                        r(h) = V_φ(c, f_act(c, h));   constraint  r(h) ≤ τ
if r_c ≤ τ:                  h_safe = h_c                  # untouched
elif ||g_c||² ≥ η:           g_c = ∇_h r |_{h_c};   h_safe = h_c − (r_c − τ)/||g_c||² · g_c
else:                        unusable gradient → fallback ladder (never "safe")
iterate ≤ K times (recompute r, g); log the NONLINEAR r(h_safe) next to the linearized prediction
fallback ladder:             F1 base action if critic-passed → F2 hold ≤ W steps → F3 terminate (= failure)
```
Corrections to the original tangent-plane idea (kept from the plan): the half-space projection replaces tangent projection; it constrains per-example *activations*, not LoRA parameter gradients; it is a first-order local correction, not an exact nonlinear projection.

## 4. Claim boundaries
**May claim:** reduces a candidate's *locally linearized critic risk* when the gradient is informative; measured simulator violations/success under the declared protocol.
**May not claim:** nonlinear constraint satisfaction, physical validity under unknown dynamics, formal safety, behavioral-manifold preservation, superiority before the controls in `08/10` support it, sim-to-real.

## 5. Decision rules
**Not supported if:** critic fails its gate (G3); projection lowers critic score but not measured violations (E1); OOD success ≤ tuned penalty (B2) within CI and no predeclared benefit offsets cost; violations rise or gains come from fallback / action suppression / critic exploitation; effect disappears across seeds or cells; latency/VRAM unusable.
**Supported only if all hold:** (1) critic predicts held-out violations with acceptable calibration; (2) E1 shows real violation reduction beyond norm-matched random correction; (3) B3 beats B1 and B2 on OOD-combo success, or preserves success with materially fewer violations; (4) result survives C1–C3, B1-LN, B5, B6 and ≥3 training seeds; (5) overhead is acceptable. The minimum practically meaningful difference (MPCD) and CI rule are fixed after the pilot, before TEST.

## 6. Additions in this revision (not in the original plan)

| # | Addition | Why | Evidence |
|---|---|---|---|
| 1 | Stage A/B base policy | Makes B0 and the fallback non-degenerate | [H] |
| 2 | Flow-matching adaptation: terminal-estimate risk, projection only at low-noise steps | `f_act` is an ODE with 10 steps in SmolVLA | [V] SmolVLA; [V] FACT |
| 3 | **Logit-space constraint option** `z ≤ logit(τ)` | Sigmoid saturation kills gradients → "small-gradient" failures | [H] |
| 4 | Train × inference factorial | Separates regularizer effect from inference-time filter | [H] |
| 5 | **FACT Logit-Normal schedule control (B1-LN)** | Noise-schedule starvation is a competing explanation for contact gains | [V] FACT |
| 6 | B4 split into post-hoc (B4a) and in-flow guidance (B4b); depth sweep S0→S5 | Latent-vs-action is a continuum, not a binary | [Va] Dalal; [H] |
| 7 | B5 critic best-of-N, B6 critic-gate-only, controls C1–C3 | Isolate gradient vs. selection vs. gating vs. shrinkage vs. random perturbation | [Va] V-GPS; [H] |
| 8 | E1 counterfactual validity (simulator state restore) | Tests "correction reduces real violations" without closed-loop confounds | [H] |
| 9 | Proxy–gold framing + critic-exploitation curve | Critic = proxy, simulator = gold | [Va] Gao et al. |
| 10 | Backward-pass analysis: active projection has Jacobian `I − ggᵀ/‖g‖²` (g, ‖g‖² detached, risk in graph) | The old tangent projection reappears in the *backward* pass; ablate | [H] derived (`07`) |

## 7. Limitations (carry into every write-up)
Critic validity is the central risk; gradient quality is unknown; one-step linearization may be poor; the latent site is architecture-specific; dynamics may be partially unobservable from deployable inputs; Meta-World contact richness is limited (guarded by G1); sim-to-real untested; novelty unestablished.
