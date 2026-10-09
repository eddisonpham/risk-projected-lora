# 12 — Reporting and Audit

## Claim ladder (use the lowest rung the evidence supports)
**L0** observed in this run/config · **L1** replicated across ≥3 training seeds · **L2** survives controls (C1–C3, B1-LN, B5, B6) · **L3** survives confound ablations and held-out cells · **L4** generalizes across model/task (needs A23 + second task).

| Allowed | Not allowed (without L3+) |
|---|---|
| "we observed", "is consistent with", "suggests", "in this simulator/task" | "proves", "guarantees", "safe", "constraint satisfied", "preserves the manifold", "works on real robots" |
| "reduces critic-predicted risk" | "reduces risk" (unqualified) |
| "no significant difference (CI …, MPCD …)" | "equivalent" (needs an equivalence test) |

## Hypothesis card
```
ID / date / author-loop#:
Claim (directional, quantitative):
Mechanism:
Prediction & what would REFUTE it:
Cheapest decisive test:
Known confounds / controls:
Prior (subjective): 
```
## Experiment card (commit before running; tag prereg/<id>)
```
Exp ID / config hash / git SHA:
Hypothesis IDs tested:
Arms (incl. controls):  Cells / split / n per cell / seeds:
Primary metric + analysis + decision rule:  MPCD:
Compute estimate:  Stop rule:
Deviations from plan (fill after):
```
## LOG.md entry
```
## <date> — iteration N — <title>
Hypothesis → Research (sources, tag upgrades) → Revision → Experiment (card link) → Results (tables, raw log paths)
Analysis (claim-ladder rung, surprises, bugs ruled out) → Reflection (prior updates) → Next 1–3 hypotheses → OPEN QUESTIONS
```
## Audit note (before any write-up)
List: (1) verified observations; (2) hypotheses still untested; (3) failures/nulls; (4) claims explicitly NOT made; (5) tag audit — every [R]/[V2]/[U]/[H] item either upgraded or caveated; (6) leakage audit (splits, thresholds, τ selection); (7) protocol deviations; (8) negative controls reported; (9) compute/seed counts.
