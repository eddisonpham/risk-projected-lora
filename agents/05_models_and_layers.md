# 05 — Models, Layers, Sites, Memory

## 1. Model tiers
| Tier | Model | Params | Action head | Role | Fits 7 GB? |
|---|---|---|---|---|---|
| **A1** | **SmolVLA** (LeRobot) | 0.45B (~100M expert) [V] | flow matching, 10 steps [V] | All main experiments | Profile (G2); paper: ~6× less memory than π0 [V] |
| **A2** | **ACT** (LeRobot) | ~80M [V] | CVAE transformer regression; decode with zero latent → deterministic | *Mechanism control*: does the effect need flow matching? Not a VLA — say so in write-ups | Yes (small) |
| B | VLA-Adapter-0.5B | 0.5B; ~97M policy [V2] | **L1 regression** (Bridge Attention) [Va] | Deterministic-head VLA replication | Repo suggests ≈10 GB [V2] → only if profiled ≤7 GB or on a larger GPU |
| C | π0 / π0.5, SmolVLA-2.25B | 3.3B / 2.25B | flow matching | Off-budget replication on rented GPU (FACT used π0.5+LoRA [V]) | **No**: π0 inference >8 GB, LoRA >22.5 GB [V] |

**Excluded:** OpenVLA (7B, discrete action tokens [V] — breaks the continuous action-gradient path); Octo (JAX, not in LeRobot [R]; not evaluated).

## 2. SmolVLA anatomy (config values [V]; module names **resolved 2026-10-09**, see below)

> **Resolved [V].** `scripts/inspect_model.py` (S3) ran against `lerobot/smolvla_base` with
> LeRobot 0.6.0 and wrote `configs/model/smolvla_sites.yaml`. Verified: the action expert is
> `model.vlm_with_expert.lm_expert` (16 blocks × 720 = 0.75·960); attention is routed by
> **layer parity** inside a single module named `self_attn` (even = self-attention over action
> tokens, odd = cross-attention against the frozen VLM KV, whose `k_proj`/`v_proj` are
> re-created at the VLM's kv width 320). `model.denoise_step(prefix_pad_masks,
> past_key_values, x_t, timestep)` is the per-denoising-step entry point; `action_out_proj`
> maps 720 → `max_action_dim` 32, so chunks are padded to 32, not to the task's 4. Sites
> S1–S6 resolve to: S1 `action_out_proj` (input), S2 last expert `mlp.down_proj`,
> S3/S4 mid expert odd/even `self_attn.o_proj`, S5 first expert `mlp.down_proj`,
> S6 VLM text `mlp.down_proj` (prefix-pass only).
```
images + instruction + state → SmolVLM-2 (SigLIP + SmolLM2), first 16 layers [num_vlm_layers=16]
                               → per-layer KV (computed ONCE per chunk, reused across the 10 flow steps)
noisy chunk x_τ (chunk_size=50) + τ → action expert (width 0.75×d; blocks alternate cross-attn / causal self-attn,
                               self_attn_every_n_layers=2) → final linear → velocity v
x ← x + dt·v over num_steps=10 → action chunk
```
Which block parity is cross- vs. self-attention, the expert block count, and module names must be read from the code and written to `configs/model/smolvla_sites.yaml`.

## 3. Sites (a site = a Linear module's output where projection acts)
| ID | Location | Downstream `f_act` | Note |
|---|---|---|---|
| S0 | final chunk (post-hoc) | identity | = **B4a** (Dalal-style) |
| S0f | `x_τ` guidance at the last m denoising steps | terminal estimate | = **B4b** (classifier-guidance style [R]) |
| S1 | input of the final action linear (last expert hidden) | final linear | |
| **S2 (default)** | last expert block, MLP down-projection output | rest of block + final linear | |
| S3 | mid expert block, cross-attn output proj | remaining blocks | reads VLM KV |
| S4 | mid expert block, self-attn output proj | remaining blocks | mixes action tokens |
| S5 | early expert block | most of expert | longer path, costlier gradient |
| S6 (stretch) | VLM text-layer output | whole expert × 10 steps | one-time per chunk but heaviest gradient |

**LoRA placement P is decoupled from the site S.** Default P: r=16, α=32, dropout 0 on {q,v,o,down} of the last 4 expert blocks; S=S2. The projection formula needs only `h_c` and `g`; `h0`/`Δh` are logged when LoRA exists at the site. [H]

## 4. Flow-matching adaptation (the part the original plan did not cover)
**Convention trap.** FACT: `a_τ=(1−τ)a0+τε`, **τ=1 is noise**, target `v*=ε−a0`, integrate τ:1→0 [V]. The SmolVLA paper writes `A^τ=τA+(1−τ)ε` [V]; the code convention was tagged [R] and is now **verified [V] (LeRobot 0.6.0, 2026-10-09)**: `VLAFlowMatching.forward` uses `x_t = time*noise + (1-time)*actions` with target `u_t = noise - actions`, and `sample_actions` integrates `x_t += (-1/num_steps)*v_t` from `tau = 1.0`. So **τ=1 is noise**, exactly as `spec` assumed. `tests/unit/test_flow_convention.py` still pins it by construction; the algebra lives in `riskproj.policies.flow` and nowhere else.

**Terminal estimate** (τ=1 noise): `â0 = x_τ − τ·v_θ(x_τ, τ; h)`. `risk_fn(h) = V_φ(c, â0(h))`, `f_act` = remainder of the expert at *this* step. Lookahead beyond one step is an ablation (`10`).
**Where active:** only at the last m denoising steps (default m=3, i.e. τ≤0.3) [H] — at large τ, `â0` is noise-dominated and the critic is meaningless; this is also the "contact-correction regime" τ<0.2 where FACT finds training signal is starved [V]. In training, project when the sampled τ ≤ τ_proj=0.3; the loss is the flow-matching loss on the velocity computed from `h_safe`.
**Paired evaluation:** use common random numbers — seed the initial noise ε per (episode, step) so methods are compared on identical noise.
**Deterministic heads (ACT / VLA-Adapter):** `f_act` is the full forward; no terminal-estimate trick.

## 5. Training defaults ([V] from SmolVLA paper unless tagged)
AdamW β=(0.9,0.95); pretrain lr 1e-4 cosine → 2.5e-6, 100 warmup; bf16; paper fine-tunes sims for 100k steps at batch 64 but notes far fewer steps work [V]. Ours [H]: Stage A expert-only SFT 5–20k steps; Stage B LoRA lr 1e-4 cosine, 5–20k steps, batch as VRAM allows (≥8), grad accumulation to effective 32. LoRA from LeRobot if available, else HF PEFT [R]. Do **not** add LN schedule to B1; it is its own arm (B1-LN).

## 6. Memory (G2)
Measure peak `torch.cuda.max_memory_allocated` with a reset between: (a) critic training (cached features), (b) LoRA training with projection, (c) inference with projection. Reduction order if >7 GB: batch → chunk/critic horizon → m, K → gradient checkpointing → 8/4-bit base [R, validate numerics; baseline must use same precision] → smaller model. **Never weaken the evaluation protocol to fit.** Extra projection cost ≈ `m·K·(tail fwd+bwd + critic fwd+bwd)`; autograd must stay enabled around the tail (`inference_mode`/`no_grad` forbidden there). Do projection and critic math in fp32.
