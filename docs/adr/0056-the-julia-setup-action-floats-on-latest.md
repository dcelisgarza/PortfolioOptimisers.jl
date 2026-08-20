---
status: accepted
---

# The Julia setup action floats on `@latest`

## Context

Eleven workflows live in `.github/workflows/`. Seven of them install Julia with
`julia-actions/setup-julia`. Until 2026-08-17 the seven did not agree on the tag:

| Tag | Workflows |
| --- | --- |
| `@latest` | `Aqua.yml`, `FormatCheck.yml` |
| `@v2` | `CompatHelper.yml`, `Copier.yml`, `Docs.yml` (two jobs), `LinkChecker.yml`, `ReusableTest.yml` |

The ninth security pass of 2026-08-16 recorded the disagreement as a supply-chain finding. It
read the two `@latest` steps as the defect, and it gave this direction:

> Pin both to `@v2`, and add `permissions: contents: read` to the workflows that only read, as
> `FormatCheck.yml` already does.

The pass also noted that `.github/dependabot.yml` watches the `github-actions` ecosystem every
week, and that Dependabot cannot bump a tag which is not a version.

The disagreement was the real defect. The direction of the fix was not the reviewer's to pick.

## Decision

**Every `julia-actions/setup-julia` step uses `@latest`.** The maintainer settled the
disagreement in favour of the floating tag, because a setup action is a toolchain installer and
the newest installer is the correct one.

The five `@v2` steps moved to `@latest`. All seven steps now agree.

The rule covers the Julia setup action alone. It says nothing about the other actions in the
tree, which keep their version tags:

- `actions/checkout@v7`
- `actions/setup-python@v7`
- `julia-actions/cache@v3`
- `julia-actions/julia-buildpkg@v1`
- `julia-actions/julia-runtest@v1`
- `julia-actions/julia-processcoverage@v1`
- `codecov/codecov-action@v7`
- `lycheeverse/lychee-action@v2`
- `peter-evans/create-pull-request@v8`
- `JuliaRegistries/TagBot@v1`

## Consequences

**Dependabot no longer sees the Julia setup action.** This is the intent of the decision, not a
gap in it. A floating tag updates itself, so the weekly Dependabot pull request has nothing to
do for that step. Dependabot keeps its authority over the ten actions in the list above.

**A review must not raise this pin again.** The asymmetry that the security pass found is
closed, and this ADR is the answer to the question it asked. A later pass that proposes `@v2`,
`@v1`, or a commit SHA for `julia-actions/setup-julia` reopens a decision that the maintainer
already made. Amend this ADR if the decision changes. Do not restate it as a finding.

**The least-privilege half of that finding was a different subject, and it is closed too.** Eight
of the eleven workflows declared no `permissions:` block, so their job token fell back to the
default of the repository. All eleven now enumerate their scopes. Three of the eleven are not
obvious, so they are recorded here:

- **`Copier.yml` takes `contents: read`.** A `permissions:` block governs the job token alone.
  Both of Copier's writes — the checkout and the pull request — carry `COPIER_PAT`, so the job
  token needs no write of its own.
- **`PreCommitUpdate.yml` takes `contents: write` and `pull-requests: write`.** Its
  `create-pull-request` step gets no `token:`, so it uses the job token. This is the pair
  `CompatHelper.yml` declares for the same work.
- **`Docs.yml` declares per job.** The deploy job takes `contents: write`, `pull-requests: read`
  and `statuses: write`. The doctest job deploys nothing and takes `contents: read`.

No workflow takes `actions: write` for `julia-actions/cache`. The action wants that scope to
delete a stale cache. `CompatHelper.yml` has run green for months on a token that does not carry
it, so the deletion warns and does not fail the job.

**A floating tag can still break a green branch.** Issues #332 and #333 came from a Julia point
release that arrived through a floating version selector. The same class of surprise can arrive
through the action. The maintainer accepted this cost. Read a sudden red workflow against the
last commit that changed code, and check the toolchain before you look for a defect in the
source.
