# patches — unavoidable upstream modifications

**Policy (`03` §5): wrap, don't edit.** `third_party/` is never modified in place. If LeRobot or
Meta-World lacks a hook we need, the order is:

1. a forward hook or subclass in `src/riskproj/policies/` — for example capturing a site activation
   with `register_forward_hook` instead of editing the model (this is how `smolvla.py` reads sites);
2. a thin adapter that reproduces the missing behaviour from outside the package (`policies/base.py`);
3. **only if both are impossible**, a patch here, with a written rationale and an upstream issue link.

A patch is a maintenance liability: it is invisible in upstream's history, it can silently fail to
apply after a pin change, and it makes our results harder for anyone else to reproduce. Treat
"we need a patch" as evidence that an adapter is missing rather than as a normal step.

## Naming and required contents

```
patches/NNN-short-description.patch        # NNN = 3-digit sequence, applied in numeric order
```

Every patch must be accompanied by an entry in the table below containing:

- **what** it changes (file + hunk summary),
- **why** the wrapper approach failed — the specific API that does not exist,
- **upstream issue/PR link** (or an explicit note that none was filed and why),
- **the upstream SHA it applies to** and whether it fails forward,
- **how it is verified** (`python tasks.py test` / `test-sim` command and result).

Patches are re-applied by `scripts/apply_patches.py`, which `python tasks.py setup` and
`python tasks.py bootstrap-sim` invoke (see `tasks.py`). That script must:

- apply in numeric order, `git apply --check` first, then `git apply`;
- fail loudly if a patch does not apply — a silently skipped patch means our numbers come from
  different code than the one this file documents;
- be idempotent, so re-running setup on an already-patched tree is a no-op rather than an error.

A patch is not "done" until `python tasks.py smoke` passes on the patched pin *and* the pin's SHA is
recorded here and in `third_party/README.md`.

## Patch table

**Empty — there are no patches in this repository.** Recording the emptiness is deliberate: it is
the claim that every upstream need so far was met by wrapping, and it is the thing to re-check
before writing a patch.

| # | File | Upstream repo + SHA | Change | Why wrapping failed | Upstream issue | Verified by |
|---|---|---|---|---|---|---|
| — | _none_ | — | — | — | — | — |

## Related

- `third_party/README.md` — how submodules are initialised, pinned and recorded.
- `03` §5 — the upstream modification policy this file implements.
- `02` §D — the fork policy and the repo table.
