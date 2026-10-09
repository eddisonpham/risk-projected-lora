# DATA CARD — riskproj

One card per generated dataset artifact. **This file is a template and every value in it is pending:
no dataset has been generated yet.** Nothing below may be filled in from expectation — read the
values out of `data/MANIFEST.json` and from the generator's returned summary, then record the exact
command that produced them. A guessed value here would invalidate the audit note (`12`).

## Card template

```
Dataset ID:            (D-nom | D-B | D-crit | D-ref)
Generator:             riskproj.data.<module>  (entry point, e.g. generate_demos / generate_critic_data)
Generator version:      (data.schema.DATASET_VERSION at generation time + the git SHA of the run)
Config hash:            (ExperimentCfg.config_hash(), 8 hex chars)
Git SHA:                (repo + submodule SHAs, as written to runs/*/git_sha.txt)
Created (UTC):
Command:                (the exact command, copy-pasted)
Records:               (count)
Split roles covered:    (TRAIN / VAL — TEST must never appear in a tuning dataset)
Cells:                 (count per role; cell ids)
Cell histogram:         (records per cell, so imbalance is visible rather than averaged away)
Class balance:          (any-violation positive rate; per-type V1..V5 rates)
Chunk-source mix:       (expert / policy / perturbed(σ grid) / unsafe — D-crit only)
Dynamics read-back:     (per-axis effective scales actually achieved; assert vs requested)
Manifest entry:          data/MANIFEST.json -> entries["<path>"] (sha256, bytes)
Known limitations:
```

## Requirements the card must satisfy

1. **Reproducibility.** Generator version + config hash + git SHA + seed set are all present, and the
   recorded command regenerates the same manifest sha256 (`skills/software_engineering.md` §5).
2. **Balance is shown, not averaged.** Per-cell record counts and per-type positive rates appear
   even when they are embarrassing. A dataset whose positives concentrate in two cells produces a
   critic that only protects those two cells (`04` §7).
3. **D-crit positivity.** Any-violation positives must be at least 15 % of tuples, and the
   per-source mix must be visible; a critic trained on expert-only chunks sees a proposal
   distribution no policy will reproduce (`06` §2, ablation A17). If the generator's positivity
   guard raised, the card says so and the run is not used.
4. **Deployable inputs only.** The card states which optional context blocks are present
   (frozen visual embedding, privileged-sim force). If any privileged channel is on, the card says
   so in bold and the dataset may not back a headline number (`04` §5).
5. **Sealed splits.** A card that lists TEST-combo or TEST-range records is a red flag for the
   reviewer: it means a tuning dataset was generated on sealed cells, and every downstream comparison
   using it is invalid.

## Cards

_None yet._ Add one section per dataset as it is generated, most recent first:

```
## D-crit — pending
(status: not generated)
```

## Cross-references

- `data/README.md` — the dataset contract and the manifest rules.
- `04` §5–§7 — inputs the critic may see, counterfactual label generation, dataset definitions.
- `06` §2 — anti-leak rules the critic's training data must obey.
- `12` — the audit note this card feeds.
