# third_party — pinned upstream dependencies

Consumed as **git submodules at pinned commits**, never as forks and never edited in place
(`02` §D fork policy, `03` §5).

| Repo | Role | Consumed as |
|---|---|---|
| `huggingface/lerobot` | SmolVLA + ACT, dataset format, train/eval entry points, Meta-World/LIBERO envs | submodule, pinned (`third_party/lerobot`) |
| `Farama-Foundation/Metaworld` | Meta-World v3 tasks and scripted experts | submodule, pinned (`third_party/Metaworld`) |
| `OpenHelix-Team/VLA-Adapter` | Deterministic-head VLA (tier B, stretch) | optional submodule |

## Why submodules rather than forks

- A fork drifts: patches applied to a fork are invisible in a diff of our repository, and the
  version actually used for a published number becomes unrecoverable.
- Submodules record one exact commit in `.gitmodules` + the gitlink, and
  `src/riskproj/utils/io.py:git_sha()` writes the resolved SHAs into `git_sha.txt` in **every** run
  directory (`03` §6). That is what makes a run reproducible a year later.
- We are consumers, not maintainers: `smolvla.py` and `act.py` adapt through subclassing and
  forward hooks (`policies/base.py`), so no upstream edit is needed for the features this project
  uses. If one becomes unavoidable it goes in `patches/`, per `patches/README.md`.

## Initialise and pin

```bash
# First time: register the submodules (records the gitlink + .gitmodules)
git submodule add https://github.com/huggingface/lerobot third_party/lerobot
git submodule add https://github.com/Farama-Foundation/Metaworld third_party/Metaworld
# Optional stretch tier:
git submodule add https://github.com/OpenHelix-Team/VLA-Adapter third_party/VLA-Adapter

# Any fresh clone / CI checkout:
git submodule update --init --recursive

# `tasks.py bootstrap-sim` runs the line above for you, then installs the extras.
python tasks.py bootstrap-sim
```

Pinning procedure (repeat whenever an upstream update is wanted):

1. `cd third_party/<repo> && git fetch && git checkout <tag-or-sha>`
2. `cd ../.. && git add third_party/<repo>` — the gitlink change is the pin.
3. Run `python tasks.py smoke` and `python tasks.py test-sim`; a pin that breaks S1/S2 does not land.
4. Record the SHA in the table below **and** in `patches/README.md` (`03` §2), then commit the pin
   and the table in the same commit so they can never disagree.

Never `git add` files *inside* a submodule directory; never leave a dirty submodule when a run
starts (`git_sha()` records `dirty` and a dirty upstream invalidates the run's provenance).

## Resolved SHAs

Empty until the submodules are added. Fill this in as part of the pin commit.

| Repo | Pinned SHA | Pinned tag/ref | Date pinned | Verified by |
|---|---|---|---|---|
| `third_party/lerobot` | _pending_ | _pending_ | _pending_ | _pending_ |
| `third_party/Metaworld` | _pending_ | _pending_ | _pending_ | _pending_ |
| `third_party/VLA-Adapter` (optional) | _pending_ | _pending_ | _pending_ | _pending_ |

## Note for whoever adds the first submodule

`.gitignore` currently contains `third_party/*/` (with `!third_party/README.md`), which keeps upstream
file trees out of our diffs. Gitignore rules do not normally apply to a gitlink, but confirm that
`git submodule add` actually staged `third_party/<repo>` (`git status` should show it as a new
submodule, not nothing) before committing. If it did not, narrow the pattern to
`third_party/*/*` rather than dropping the rule — otherwise every upstream file becomes visible in
`git status` on every checkout.
