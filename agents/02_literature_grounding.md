# 02 — Literature & Repo Grounding

Tags: see `README.md`. "Used in" points to the file where the fact is relied upon.

## A. Sources read this session

| Source | Tag | Facts we rely on | Used in |
|---|---|---|---|
| **SmolVLA** arXiv 2506.01844 | [V] | 450M params (~100M action expert); SmolVLM-2 (SigLIP + SmolLM2); only first 16 LLM layers used (N=L/2); expert hidden = 0.75×d; interleaved cross-attn / causal self-attn blocks; flow matching, chunk n=50, **10 inference steps**; paper trains only the expert with frozen VLM; evaluated on LIBERO and Meta-World (MT50 dataset: 2,500 eps); in sim it re-predicts after every executed action; ~6× less memory than π0; bf16 + torch.compile; ACT ≈80M params | 04, 05 |
| **SmolVLA config defaults** (several HF `train_config.json`) | [V] | `chunk_size=50, num_steps=10, num_vlm_layers=16, self_attn_every_n_layers=2, expert_width_multiplier=0.75, train_expert_only=true`. **Module names are NOT verified** → [R] | 05 |
| **LeRobot** ICLR 2026, arXiv 2602.22818 | [Va] | Native LIBERO + Meta-World eval, `lerobot-train/-eval`, EnvHub for custom envs; authors caution that simulation is hard for contact-rich tasks | 03, 04 |
| **Meta-World+** NeurIPS'25 D&B, 2505.11289; `Farama-Foundation/Metaworld` | [Va] | Gymnasium + MuJoCo-bindings, reproducible v3 tasks; per-task scripted experts in `metaworld.policies` (e.g. `SawyerPushV3Policy`, shown in LeRobot docs [V2]); peg-insert expert class name → [R] | 04 |
| **MuJoCo contact softness** (docs via secondary copies) | [V2] | `solref=(timeconst, dampratio)`; `timeconst ≥ 2×timestep`; negative values = direct (−stiffness, −damping); smaller timeconst = stiffer; `solimp` shapes impedance | 04 |
| **FACT** arXiv 2608.01402 (Aug 2026) | [V] | Failure modes: *precision* (flow-matching Beta schedule gives only 8.9% of gradient to τ<0.2 in π0/π0.5/SmolVLA) and *force*. Logit-Normal schedule `τ=σ(s·z−m), m=1.5, s=1` gives ~6× more mass at τ<0.2; **LN alone: +20pp plug, +45pp button, +22.5pp key** on π0.5 (real robot). Force input gave no significant gain on precision tasks. **Convention: τ=1 is noise**, a_τ=(1−τ)a0+τε. Protocol: 40 rollouts/task, 60 s timeout, Fisher's exact test vs baseline, LoRA fine-tuning. LN applies to flow-matching heads only. Authors list material/geometry diversity as a gap | 05, 08, 09, 10 |
| **CRAFT** arXiv 2602.12532 (ICRA 2026) | [Va] | Force-aware curriculum + variational information bottleneck; real-world; 50 demos/task, 5 tasks (secondary) | 02 only |
| **SafeVLA** NeurIPS'25, 2503.03480 | [Va] | CMDP/safe-RL VLA alignment; Safety-CHORES (mobile manipulation); metrics = cumulative cost + success | 08 (B2-L), 09 |
| **Dalal et al.** 1801.08757 | [Va] | Linearized safety signal → closed-form minimal action correction, differentiable "safety layer". **Mathematical ancestor of B4a** | 08, 10 |
| **Recovery RL** 2010.15920 | [Va] | Safety critic learned from offline data; recovery policy descends it; includes contact-rich manipulation sims | 06, 08 |
| **V-GPS** CoRL'24, 2410.13816; **FM-Steer** CVPR'26 | [Va] | Re-rank sampled actions with a learned value, no gradients/weights; FM-Steer adds value-guided sampling for flow VLAs → **B5 baseline; novelty risk** | 08 |
| **Gao et al.** ICML'23, 2210.10760 | [Va] | Optimizing a proxy reward degrades the gold reward past a point (Goodhart) → critic-exploitation curve | 09, 10 |
| **ReFT / LoReFT** NeurIPS'24, 2404.03592; `stanfordnlp/pyreft` | [Va] | Interventions on hidden representations of a frozen model; low-rank form `h + Rᵀ(Wh+b−Rh)`. Prior art for activation-space adaptation | 10, novelty |
| **rliable** NeurIPS'21 | [V] | Stratified bootstrap CIs, IQM, performance profiles, probability of improvement; `pip install rliable` | 09 |
| **ManiSkill3** 2410.00425 | [Va] | Contact-rich GPU sim; per-env mass/friction must be set in `_load_scene` before GPU init (docs) → fallback simulator only | 04 |
| **VLA-Adapter** AAAI'26, 2509.09372; `OpenHelix-Team/VLA-Adapter` | [Va]/[V2] | Qwen2.5-0.5B, **L1-regression** policy with Bridge Attention (~97M, [V2]); LIBERO/CALVIN; repo guidance: training from ≈10 GB VRAM [V2] → likely above budget | 05 |
| **openpi README** | [V] | π0/π0.5: inference >8 GB, LoRA fine-tune >22.5 GB, full >70 GB → **outside 7 GB** | 05 |
| LoRA 2106.09685; Nimble (RSS'21); BarrierNet; LARA (ICML'26); DeCAL 2609.09119 | [U] | Cited in the original plan; not re-verified | 01 |

## B. Recalled — verify before citing [R]
Guo et al. 2017 (calibration/ECE, temperature scaling) · Lakshminarayanan et al. 2017 (deep ensembles) · Angelopoulos et al. (conformal risk control) · Goodfellow et al. 2015 (adversarial examples) · Miyato et al. 2018 (spectral norm) · Dhariwal & Nichol 2021 (classifier guidance) · Achiam et al. 2017 (CPO) · Ray et al. 2019 (Safety Gym) · Wabersich & Zeilinger 2021 (safety filters) · Holm 1979 · Wilson 1927 · robomimic / MimicGen (fallback datasets).

## C. Findings that shaped this spec
1. FACT's schedule result is a **first-order confound** → B1-LN and B3+LN arms.
2. Dalal's safety layer **is** B4a; Recovery RL and V-GPS are the natural "use the critic without latent projection" alternatives → B4a/B4b, B5, B6.
3. openpi memory table + VLA-Adapter's ≈10 GB guidance → only SmolVLA (and ACT as mechanism control) are in the 7 GB tier.
4. LeRobot's own caution on sim + contact-rich tasks → **G1 dynamics-sensitivity gate** before any modeling.
5. FACT uses LoRA on π0.5 with real robots; no sim, no material shifts → our regime is complementary, not duplicate.

## D. Repositories

| Repo | Role | Consume as | Tag |
|---|---|---|---|
| `huggingface/lerobot` | SmolVLA, ACT, π0, dataset format, train/eval, Meta-World/LIBERO envs | git submodule, pinned commit (`third_party/lerobot`) | [Va] |
| `Farama-Foundation/Metaworld` | Tasks + scripted experts | submodule, pinned | [Va] |
| `google-research/rliable` | Aggregate stats/CIs | pip | [V] |
| `huggingface/peft` | LoRA wrappers if LeRobot lacks PEFT hooks | pip | [R] |
| `stanfordnlp/pyreft` | ReFT-style intervention ablation only | pip, optional | [Va] |
| `OpenHelix-Team/VLA-Adapter` | Deterministic-head VLA (stretch) | submodule, optional | [Va] |
| `Physical-Intelligence/openpi` | π0/π0.5 off-budget replication | not installed in prototype | [V] |
| `haosulab/ManiSkill` | Fallback simulator if G1 fails | optional | [Va] |
| `PKU-Alignment/SafeVLA` | Reference for CMDP/Lagrangian code | read-only reference | [R] |

**Fork policy:** do not fork-and-diverge. Pin upstream as submodules; keep our code in `src/riskproj/` using adapters; put unavoidable upstream changes in `patches/*.patch` with rationale (see `03`).

## E. Novelty-risk list & search plan (do before any "novel" claim)
Closest prior art: Dalal 2018 · Recovery RL · V-GPS / FM-Steer · ReFT/LoReFT · SafeVLA · classifier/value guidance for flow/diffusion policies [R]. Queries to run on arXiv / OpenReview / Scholar and log in `LOG.md`: "critic-guided activation correction VLA", "safety critic gradient latent action head", "classifier guidance flow-matching VLA safety", "risk-constrained LoRA robot policy", "representation intervention constraint robot policy".
