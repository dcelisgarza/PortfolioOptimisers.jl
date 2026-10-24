---
status: accepted
---

# A pull request from a pushed branch skips the jobs its push runs

## Context

`Test.yml`, `JET.yml`, `Complexity.yml`, `Aqua.yml`, `Docs.yml`, `Paper.yml`, `FormatCheck.yml`
and `LinkChecker.yml` trigger on a push to `main`, `dev*` and `agents/**`.
`TestOnPRs.yml` and the same seven workflows also trigger on a pull request. While a pull request
from `dev` is open, each push to `dev` therefore runs the full CI twice: once for the push and once
for the pull request. The suite alone costs about 170 minutes of runner time, and a reader must
reconcile two runs of each check on one head (#1783).

A pull-request run and a push run do not test the same tree. The pull-request run checks out
`refs/pull/N/merge`, the merge of the head into the base. The push run checks out the head. For a
release pull request from `dev` into `main`, after earlier squash merges, the two trees can differ.

No ruleset of this repository requires a status check. The "Protect Main" ruleset has none, so a
skipped job cannot block a merge.

## Decision

A job that a push trigger already runs skips on a pull request whose head branch is in this
repository and matches that push trigger. Each such job carries this condition:

```yaml
if: >-
  github.event_name != 'pull_request' ||
  github.event.pull_request.head.repo.full_name != github.repository ||
  !((startsWith(github.head_ref, 'dev') && !contains(github.head_ref, '/')) ||
  startsWith(github.head_ref, 'agents/'))
```

- The `dev` clause mirrors the push filter `dev*`, which matches no branch that contains `/`.
- An expression has no substring function, so the `agents/` clause cannot exclude a second `/`.
  The push filter is `agents/**` instead of `agents/*`, so the push covers every branch that the
  clause matches.
- A pull request from a fork, and one from any other branch of this repository, runs every job as
  before.

The push runs show on the head commit of the pull request, so the pull request page keeps its
checks.

Two jobs keep a part of their pull-request run:

- The `Documentation` job of `Docs.yml` also runs when the pull request carries the label
  `docs-preview`. `docs/make.jl` deploys only from `main`, so the pull-request run is the only one
  that deploys the preview. The `Doctests` job skips.
- The flag of a new Rationale in `Complexity.yml` diffs the merge commit against its base, and a
  push run has no base to diff. It is the job `rationale`, which runs on every pull request.

## Consequences

- One push to `dev` runs each check once while a pull request from `dev` is open.
- No run tests the merge commit of a pull request from `dev*` or `agents/**`. A release pull
  request whose merge tree differs from the head of `dev` merges untested. This is the trade that
  the maintainer chose over a second run of each check (#1783, option 1).
- A path filter now reads the push range, not the diff of the whole pull request. A push to `dev`
  that touches no path of `JET.yml` gives the head commit no JET check, though an earlier commit of
  the pull request carries one.
- A new workflow that triggers on both `push` and `pull_request` takes the same condition on each of
  its jobs.
