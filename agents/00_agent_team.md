# 00 — Agent Team Operating Manual

**Project:** Risk-Guided Activation Projection for LoRA Fine-Tuning of Contact-Rich VLAs.
**Scope of this file:** who does what, in which order, producing which artifact, under which gate.
Read it with `skills/vla_researcher.md` (the loop) and `skills/software_engineering.md` (the code rules);
this file only assigns the roles and their hand-offs.

Nothing here overrides `CLAUDE.md`. Where this file and `CLAUDE.md` disagree, `CLAUDE.md` wins.

---

## 1. Roster

Six roles. "Lands" = the concrete artifact the role is responsible for committing.

| Role | Mandate | Inputs | Outputs | Lands | Tool access |
|---|---|---|---|---|---|
| **Researcher** | Find out what is already known, and what would make our reading wrong: prior art/novelty (`02` §E), known confounds, exact metric definitions, implementation facts that are tagged `[R]`/`[V2]`/`[U]` | `agents/02_literature_grounding.md`, the spec, current OPEN QUESTIONS in `LOG.md` | Tag upgrades, new controls/risks, a paragraph in the log entry | Edits to `agents/02_literature_grounding.md` + `LOG.md` "Research" section | **Live**: web search + URL fetch |
| **Coder** | Implement exactly one slice, typed, tested, config-driven | The relevant `agents/0X` sections + the contract files (§2 below) | Source + unit tests + the config fragment the slice needs | `src/riskproj/**`, `tests/**`, the relevant `configs/**` fragment | Repo read/write + terminal (tests, lint) |
| **Reviewer** | Adversarially read a change: does the code do what the spec says, and does the evidence support the claim? | The diff, its tests, its config, its log entry | A review note: blocking findings, non-blocking findings, claims that outrun their evidence | Review section of the `LOG.md` entry (and the PR checklist in `skills/software_engineering.md` §6) | Repo read + terminal to reproduce |
| **Results analyzer** | Turn archived logs into tables, effect sizes and a claim-ladder rung | `runs/**/*.jsonl` **only** — never a live model (`09` §1) | Tables T1–T5 / figures F1–F8 (`09` §4), per-cell and per-axis breakdowns, proxy–gold curve | `runs/<id>/report/**` + the tables quoted in the log entry | Repo read + terminal (offline recomputation) |
| **Hypothesis proposer** | Keep the hypothesis register honest; propose the next 1–3 tests ranked by information per cost | `LOG.md`, `hypotheses.md`, gate status, surprise list from the last analysis | New/revised hypothesis cards, killed hypotheses with a reason | `agents/research_log/hypotheses.md` + `LOG.md` "Next" section | Repo read (+ web for novelty checks) |
| **Document writer** | Write up what happened, at the claim rung the evidence supports | Analyzer tables, `LOG.md`, the audit checklist | Report/paper text, data cards, the audit note | Filled `agents/12` templates; `data/DATA_CARD.md`; the audit note | Repo read/write |

### Who is actually a live agent in this environment

The roles above are real, but they are not equally *embodied*, and pretending otherwise wastes time:

- **Researcher** and **hypothesis proposer** run in the coordinating agent, because they need judgement plus live search. Only the researcher may upgrade an evidence tag, and only with a source read in the relevant sections — `[R]` → `[V]` is a research act, not an editing act (`README.md` evidence-tag rule).
- **Coder** and **document writer** run as **parallel workers with disjoint file lists**. They have no terminal and no web, so they write first-pass code and prose that the coordinator verifies.
- **Reviewer** and **results analyzer** are partly roles and partly code. The analyzer's job is deliberately **implemented in `src/riskproj/eval/`** as deterministic functions over JSONL, so that a new metric can be recomputed from an archived run without a model, a GPU or an agent (`09` §1). What remains a judgement call — "is this effect real, is a simpler explanation available" — is the reviewer's, and it is exercised in the coordinator and recorded in `LOG.md`.

### Non-negotiables the roster inherits

1. Gates are ordered (`11`): no role starts an experiment whose upstream gate has not passed.
2. TEST is sealed: `τ, λ, η, K`, site, fallback rules and adapters are chosen on VAL only.
3. Simulator outcomes decide. A critic-predicted-risk drop is a diagnostic, never a result.
4. Every projection/fallback event is logged; nothing fails silently.
5. Evidence tags: nothing tagged `[R]`/`[V2]`/`[U]`/`[H]` may be stated as established fact.
6. `LOG.md` is appended at the end of every loop iteration — including negative results.
7. `third_party/` is never edited; changes go in `patches/` with a written rationale.

---

## 2. Fan-out mechanics (how parallel work stays coherent)

Workers are spawned in one batch with **disjoint file lists**. The rules that make that safe:

- **One writer per file.** If a slice needs a change in a file it does not own, it reports it instead of editing it.
- **Contract files are frozen during a fan-out.** Nobody but the coordinator edits:
  `src/riskproj/types.py`, `src/riskproj/protocols.py`, `src/riskproj/obs.py`,
  `src/riskproj/utils/config.py`, `pyproject.toml`, `tasks.py`, `Makefile`,
  `tests/conftest.py`, `tests/tolerances.py`.
  These are the shared vocabulary — dynamics levels and cell ids, `ViolationSpec`, `PolicyOut`,
  `ProjectionInfo`, `FallbackCode`, the `EpisodeRecord`/`DecisionRecord` log schema, the
  `PolicyAdapter`/`Critic`/`Projector`/`Task`/`Metric`/`Method` Protocols, the observation and
  context key contract, and the pydantic config models. A worker that redefines any of them
  makes every other slice subtly wrong.
- **Workers cannot verify.** They have no terminal, so their tests are written, not run. The
  coordinator runs `python tasks.py test` / `lint` / `typecheck` afterwards and owns the repair.
- **The coordinator reconciles seams.** Reading each worker's diff alone is not enough: the
  failure mode of a fan-out is a change that is right in pieces and wrong as a whole.

### Seam checklist (run after every fan-out)

1. Do the interfaces line up across slices (a producer's record shape vs a consumer's reader, a
   getter's return type vs the caller's expectation, a keyword name used positionally elsewhere)?
2. Is the same problem solved twice by two slices (two helpers with the same job, two policies)?
3. Does anything reference a name that no slice created?
4. Is any requirement unowned — present in the spec and in nobody's file list?
5. Do the tests still test the spec (rather than having been quietly relaxed to pass)?

Seam defects actually found and repaired in this repo, kept as a cautionary list: a monitor whose
consumer called `update()` while the producer exposed `observe()`; a `restore_state()` that dropped
the refreshed observation its caller needed for counterfactual labelling; a dataset whose history
length was inferred from an in-memory array shape and disagreed with itself after a reload; a
projection operator that silently degraded to "unusable gradient" when called under an outer
`torch.no_grad()`; test fixtures whose batch sizes did not match the tensors they were paired with.

---

## 3. One worked loop iteration — **H3** (does the correction reduce *measured* violations?)

H3 is the decisive hypothesis (`research_log/hypotheses.md`): if B3's corrected chunk does not
reduce real violations against the original chunk and against a norm-matched random correction,
the closed-loop comparisons in E4 are pointless and the project stops at **G5**.

**Upstream gates that must already have passed** (`11`): **G0** (repo + smoke suite), **G3**
(critic accepted on VAL — its gradient is what E1 measures), **G4** (projection unit tests green
*and* a usable first-order region in E2). E1 is P5; starting it with G3 or G4 open means measuring
a correction whose gradient quality and whose basis are unknown.

| Stage | Role | Does | Artifact / where it lands |
|---|---|---|---|
| 0 ORIENT | coordinator | Reads the last 3 `LOG.md` entries, `hypotheses.md`, gate status; picks the open question with the highest information per cost | One-line target at the top of the new log entry |
| 1 HYPOTHESIZE | hypothesis proposer | Turns H3 into a falsifiable card: directional claim, mechanism, **refuting result**, cheapest decisive test, prior | Hypothesis card in `LOG.md` (template in `12`) |
| 2 RESEARCH | researcher | Asks "who already did this?": Dalal's safety layer, Recovery RL, V-GPS; upgrades the relevant `02` tags; adds controls | Updated `agents/02_literature_grounding.md`; C1 becomes mandatory, not optional |
| 3 REVISE | researcher + proposer | Narrows the claim: *locally linearized* risk feasibility, paired counterfactual only, no closed-loop claim | Revised card + reason, in `LOG.md` |
| 4 DESIGN | coordinator | Fixes cells, n, primary metric, analysis and stop rule **before** the run; commits the config and tags it | `configs/exp/e1_counterfactual.yaml` + tag `prereg/E1` (`git tag prereg/E1`) |
| 5 IMPLEMENT | coder(s) | Saves states, replays the original and corrected chunk from the same state, labels with `ViolationMonitor`, pairs the results | `src/riskproj/eval/counterfactual.py` + unit tests on a toy env |
| 6 RUN | coder + coordinator | Generates states and labels on VAL cells; **watches ≥10 rollouts including failures** before the full run; never peeks-and-adjusts on TEST | `runs/e1_counterfactual-<hash8>/<method>/<seed>/episodes.jsonl`, `decisions.jsonl`, provenance files (`03` §6) |
| 7 ANALYZE | results analyzer | Paired violation deltas with CIs, B3 vs original vs C1, per-type and per-cell breakdown, fallback and stall rates alongside | `runs/<id>/report/**`, tables in the log entry, claim rung |
| 8 REFLECT | coordinator | Updates priors, logs bugs ruled out, decides continue/branch/pivot/stop | `LOG.md` entry; 1–3 new cards ranked by information per cost |

Two things this table is meant to make obvious. First, the run directory and the table are
different artifacts: the run directory is immutable raw evidence, the table is derived and
recomputable from it. Second, the *decision rule* is written before the run; a rule invented after
seeing the numbers is an exploratory observation wearing a pre-registration's clothes.

**Open protocol question on E1 (flag, not a settled defect).** `configs/exp/e1_counterfactual.yaml`
currently lists `splits: [val, test_combo]`. E1 doubles as a stop rule — its outcome decides whether
the project continues — so reading TEST-combo numbers before `prereg/E4` risks spending part of the
sealed test set on a *decision* rather than on a *measurement*. The conservative reading of
`CLAUDE.md` #2 is to run the pre-registration E1 study on VAL cells only and make the TEST-combo
counterfactual an E4 arm. Record the choice in `LOG.md` before the run, whichever way it goes.

---

## 4. Definition of done, per role

| Role | Done when |
|---|---|
| Researcher | Every claim used in a decision is tagged, and every previously unverified tag the decision relies on has either been upgraded with a source or explicitly caveated in the write-up |
| Coder | Typed, tested (tests proportional to risk), config-driven, seeded, registered, documented, one `LOG.md` entry, `python tasks.py lint` and `test` green, no `third_party/` edit |
| Reviewer | Every `09` §5 interpretation guard has been checked, every claim has a rung, and any blocking finding is either fixed or recorded as an OPEN QUESTION |
| Results analyzer | Tables are recomputable from archived JSONL alone; CIs are present; per-cell heterogeneity, fallback/stall rates and nulls are shown, not hidden; MPCD and multiplicity handled |
| Hypothesis proposer | The register is current: killed hypotheses stay with their reason, priors are updated rather than deleted, the next tests are ranked |
| Document writer | Claim language sits at or below the supported rung (`12`); the audit note is complete; no forbidden phrase appears |

---

## 5. When blocked (escalation)

Default: record it under **OPEN QUESTIONS** in `LOG.md`, take the **most conservative** option,
continue. Escalate to the human only for:

1. a budget/VRAM gate failure (G2) with no in-scope fix (`05` §6 reduction order exhausted);
2. a gate failure suggesting the hypothesis is dead (G3 critic invalid, G5 corrections don't reduce
   real violations, E4 null surviving controls) — the escalation is "stop and write it up";
3. any change to a claim-relevant protocol (splits, metrics, thresholds, τ/λ selection rule) after
   pre-registration.

Everything else — a flaky test, a missing geom name, an unresolved module name, a slow run — is
ours to handle and belongs in the log.

---

## 6. How the roles keep each other honest

**Reviewer checklist** (from `09` §5 and the analysis checklist in `skills/vla_researcher.md`):

1. Was anything tuned on TEST? Are splits, thresholds and τ frozen before the run?
2. Are comparisons paired (same states, same CRN flow noise) and is the replication unit the
   training seed rather than the episode?
3. Are CIs present everywhere, is the MPCD applied, and is multiplicity handled (pre-registered
   primary comparisons vs exploratory ones)?
4. Were per-cell heterogeneity and per-axis effects inspected? Are OOD-combo and OOD-range kept
   separate and never pooled?
5. Is success reported alongside violations **and** fallback/stall rate — and is the do-nothing
   baseline checked?
6. Did the effect survive C1–C3, B1-LN, B5 and B6?
7. Is critic score reported separately from simulator truth (proxy–gold), never as evidence?
8. Were the dynamics actually applied — per-axis read-backs, not vibes?
9. Are seeds, n, failures and nulls all reported?
10. Does a simpler explanation fit (noise schedule, gating, shrinkage, base-policy competence)?

**Analyzer duties.** A gain that arrives with a high F2/F3 fallback rate or a stall rate is not an
adaptation gain (`09` §5). A critic-score drop with no measured-violation drop is a negative result
*about the critic*, and must be written as one.

**Writer duties.** Use the lowest rung the evidence supports — **L0** observed in this run/config ·
**L1** replicated across ≥3 training seeds · **L2** survives controls · **L3** survives confound
ablations and held-out cells · **L4** generalizes across model/task. Allowed: "we observed", "is
consistent with", "suggests", "in this simulator/task". Forbidden: "proves", "guarantees", "safe",
"constraint satisfied", "preserves the manifold", "works on real robots", and unqualified "reduces
risk" (say "reduces critic-predicted risk"). "No significant difference" requires a CI and the MPCD;
"equivalent" requires an equivalence test.

---

## 7. Current state (honest, checkable)

- **Phase.** Phase 0/1 of `agents/11_roadmap_and_gates.md`. The repository contains the layout,
  the contract files, and working implementations of the environment/dynamics/violation layer, the
  data and split layer, the policy adapters, the risk critic, the projection operator, and the
  utility/metrics layer. Config fragments and six experiment configs exist under `configs/`.
- **No experimental result exists.** There is no `runs/` directory containing measurements. Nothing
  in this repository currently supports any claim about the method.
- **The toy task is a pipeline testbed, not evidence.** `envs/toy.py` exists so that the whole
  pipeline runs on CPU in CI (`S6`); its numbers may never be reported as evidence about
  contact-rich manipulation. The research task is Meta-World `peg-insert-side-v3` (`04` §1).
- **Tests.** At the checkpoint commit `9efe571`: 291 passed, 4 skipped, 6 deselected
  (`python tasks.py test`); the skips are the real-policy contract tests, which require the VLA
  stack. Re-run the command rather than trusting this line.
- **Gates.** `configs/gates.yaml` holds the G1 (dynamics sensitivity), G2 (≤7 GB VRAM), G3 (critic
  acceptance) and G6 (pilot sanity) thresholds. G0 is procedural (smoke suite S0–S6 passes), and so
  are G4 (projection unit tests + E2), G5 (E1 validity) and G7 (protocol frozen before the first
  TEST episode). **No gate has been formally passed yet** — G0 is open until the smoke suite runs
  end-to-end and reports its skips as skips.
- **Not yet installed / not yet written.** The VLA stack (LeRobot) and the Meta-World package are
  not installed, so `SmolVLAAdapter`/`ACTAdapter` report unavailable and the S0/S1/S3/S4/S5 smoke
  checks skip rather than pass. `configs/model/smolvla_sites.yaml` and `configs/env/task_geoms.yaml`
  do not exist yet: they are *discovered* files (S3 and `inspect_contacts.py`), never hand-written.
  `tasks.py` wires the `scripts/` entry points (`smoke_suite.py`, `run_experiment.py`,
  `make_report.py`, `apply_patches.py`) and several modules reference
  `scripts/inspect_model.py`, `scripts/inspect_contacts.py`, `scripts/sensitivity_sweep.py` and
  `scripts/derive_violation_thresholds.py`; check `ls scripts/` before relying on one.
- **What to do next.** Finish P0: run the smoke suite, record every skip with its reason, tag
  `gate/G0`, then start P1 (dynamics knobs and read-backs, violation thresholds derived from nominal
  expert rollouts, G1 sensitivity sweep, G2 VRAM profiling).
