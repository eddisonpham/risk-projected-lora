# 07 — Projection Operator

## 1. Contract
`project(h_c, risk_fn, cfg) -> (h_safe, info)`; per-example (batch examples independent); `risk_fn(h) -> [B]` (prob or logit). Gradient computed w.r.t. the **flattened site activation** of each example (all tokens jointly; per-token mode is an ablation).

## 2. Reference sketch (illustrative; adapt, then test)
```python
@dataclass
class ProjCfg:
    tau: float; eta: float = 1e-8; K: int = 1
    space: str = "prob"            # "prob" | "logit"  (logit: constrain z <= logit(tau))
    margin: float = 0.0            # target tau - margin
    max_rel_step: float | None = None   # trust region ||u|| <= rho*||h_c||
    backward: str = "projected"    # "ste" | "projected" | "second_order"

def project(h_c, risk_fn, cfg):
    h, log = h_c, {"r_cand": None, "unusable": torch.zeros(h_c.shape[0], dtype=torch.bool, device=h_c.device)}
    thr = cfg.tau - cfg.margin
    thr = torch.logit(torch.tensor(thr)) if cfg.space == "logit" else thr
    with torch.enable_grad():
        for k in range(cfg.K):
            hk = h if h.requires_grad else h.detach().requires_grad_(True)
            r = risk_fn(hk).float()                                   # [B]
            if k == 0: log["r_cand"] = r.detach()
            excess = r - thr
            active = excess > 0
            if not active.any(): break
            (g,) = torch.autograd.grad(r.sum(), hk, retain_graph=True,
                                       create_graph=(cfg.backward == "second_order"))
            gf = g.flatten(1).float(); gn2 = gf.square().sum(1)
            usable = active & (gn2 >= cfg.eta)
            log["unusable"] |= active & ~usable
            gd = g if cfg.backward == "second_order" else g.detach()
            ex = excess if cfg.backward == "projected" else excess.detach()   # "ste": identity Jacobian
            coef = (ex.clamp_min(0) / gn2.detach().clamp_min(cfg.eta))
            u = coef.view(-1, *[1] * (h.dim() - 1)) * gd
            if cfg.max_rel_step is not None:   # trust region
                s = (cfg.max_rel_step * h.flatten(1).norm(dim=1) / (u.flatten(1).norm(dim=1) + 1e-12)).clamp(max=1)
                u = u * s.view(-1, *[1] * (h.dim() - 1))
            h = torch.where(usable.view(-1, *[1] * (h.dim() - 1)), h - u, h)
    return h, log
```
Log per decision: `r_before, r_after_nonlinear, r_after_linear_pred, ‖g‖², active, usable, K_used, ‖u‖/‖h0‖, ensemble_std, fallback_code`.

## 3. Backward-pass analysis [H, derived — verify numerically]
When active, `h_safe = h_c − relu(r(h_c)−τ)/‖g‖² · g` with `g, ‖g‖²` detached and `r` kept in the graph gives `∂h_safe/∂h_c = I − g gᵀ/‖g‖²`. So the backward pass *removes the risk-normal component of the task gradient at the activation* — the old "tangent projection" reappears, now only in backprop. `"ste"` (excess detached) gives identity; `"second_order"` adds ∂g/∂h terms (costly). Ablating the three modes isolates whether training-time benefit comes from this effect.

## 4. Variants to implement behind flags
K ∈ {1,2,3}; backtracking line search (only if K=1 proves unstable); `space` prob/logit; margin; trust region; ensemble aggregator; per-token vs. flattened; steps-active m; lookahead (terminal estimate vs. full remaining flow); fallback on/off ladders.

## 5. Unit tests (must pass → G4)
1. Safe candidate (`r≤τ`) unchanged. 2. Active + usable → matches the closed-form on a quadratic/linear risk (exact for linear). 3. Convex synthetic risk: `r` after ≤ `r` before. 4. Idempotence on safe outputs. 5. `‖g‖²<η` → flagged unusable, activation unchanged, fallback invoked. 6. Per-example independence (permute batch → same result). 7. Critic params get no gradient; LoRA params do. 8. Backward Jacobian equals `I − ggᵀ/‖g‖²` (finite differences, "projected") and `I` ("ste"). 9. Finite outputs under bf16 autocast with fp32 projection math. 10. Fixed seed ⇒ identical output. 11. **Flow-convention test** (`05` §4). 12. Terminal estimate recovers `a0` exactly when `v=v*`. 13. K-iteration logs record nonlinear vs. predicted risk. 14. Parametrized over every registered policy adapter (`skills/software_engineering.md`).

## 6. Cost control
Differentiate only through the tail after the site; cache VLM KV; active only at last m steps; K ≤ 2 in pilots. Report latency/VRAM overhead in every results table (`09`).
