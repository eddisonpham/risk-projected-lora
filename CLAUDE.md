# CLAUDE.md

**Project:** Risk-Guided Activation Projection for LoRA Fine-Tuning of Contact-Rich VLAs ("Risk-Projected LoRA").
A falsification-first research prototype under a ≈7 GB VRAM budget. Claim under test: projecting LoRA-induced activation changes against a simulator-supervised risk critic improves held-out-dynamics behavior (friction / payload mass / contact stiffness) beyond standard LoRA, a critic penalty, and direct action-space projection. Nothing is established; the default expectation is that parts of it fail.

## Everything lives in `agents/` — start there
Read in this order before writing code:
1. `agents/README.md` — index, evidence tags, glossary
2. `agents/skills/vla_researcher.md` — **your role** and the research loop (hypothesize → research → revise → experiment → analyze → reflect → repeat)
3. `agents/skills/software_engineering.md` — code-quality and scalability rules; apply to every commit
4. `agents/01_…` through `agents/12_…` — the spec and build order; follow numerically

## Non-negotiables
1. **Gates are ordered** (`agents/11_roadmap_and_gates.md`). Do not run an experiment whose upstream gate has not passed.
2. **Test data is sealed.** Tune τ, λ, η, K, site, fallback rules on VAL only. Test configs/seeds are frozen and committed (tag `prereg/<exp>`) before any test run.
3. **Simulator outcomes decide.** Critic-predicted risk is a diagnostic, never a success metric. No safety-guarantee language.
4. **Every projection/fallback event is logged.** Nothing fails silently.
5. **Evidence tags.** Items tagged [R]/[V2]/[U] in `agents/02_literature_grounding.md` must be verified before they are relied on or cited.
6. **Append to `agents/research_log/LOG.md` at the end of every loop iteration** — including negative results and open questions.
7. **Never edit `third_party/`.** Changes go in `patches/` with a written rationale.
8. **Config-driven, seeded, tested.** No experiment without a YAML config, a seed, a unit-tested code path, and a logged run directory.

## When blocked
Record it under OPEN QUESTIONS in `LOG.md`, take the most conservative option, continue. Escalate to the human only for: a budget/VRAM gate failure with no in-scope fix; a gate failure suggesting the hypothesis is dead; any change to a claim-relevant protocol (splits, metrics, thresholds) after pre-registration.

## Commands (created in Phase 0 — see `agents/03_repo_setup.md`)
`make setup` · `make test` · `make smoke` · `make exp EXP=<config>` · `make report`
