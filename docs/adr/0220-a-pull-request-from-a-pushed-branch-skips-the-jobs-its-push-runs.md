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

A merge into `main` must wait for the CI and for CodeQL (#1832). Until #1832, the one ruleset,
"Protect Main", required no status check, and its list of target branches was empty, so GitHub
applied none of its rules to `main`.

A required check meets two facts of GitHub:

- A job that its `if:` skips reports success, so its check passes.
- A workflow that a `paths:` filter skips reports nothing, so its check stays pending and blocks
  the merge.

A skipped pull-request run and the push run of one job attach to the same head commit. With one
name, the two runs are two check runs of that name. GitHub does not document which of the two a
required check reads.

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

- The `Documentation` job of `Docs.yml` runs on every pull request. `docs/make.jl` deploys only
  from `main`, so the pull-request run is the only one that deploys the preview. The `Doctests` job
  skips. `DocPreviewCleanup.yml` deletes the preview when the pull request closes. Until 2026-10-26,
  a preview needed the label `docs-preview` (#1722). The maintainer then chose a preview for each
  pull request.
- The flag of a new Rationale in `Complexity.yml` diffs the merge commit against its base, and a
  push run has no base to diff. It is the job `rationale`, which runs on every pull request.

### A merge into `main` waits for the push checks of its head

The ruleset "Protect Main" targets `~DEFAULT_BRANCH`, which is `main`. It holds these rules:

| Rule | Effect |
| ---- | ------ |
| `deletion` | Nobody can delete `main`. |
| `non_fast_forward` | Nobody can force-push `main`. |
| `pull_request` | A change reaches `main` only through a pull request. No approval is needed. |
| `required_status_checks` | A merge waits for the push checks below on the head of the pull request. The head need not contain `main` (ADR 0240). |
| `code_scanning` | A merge waits for the CodeQL result. An error alert or a security alert blocks it. |

The ruleset has no `update` rule and no bypass actor. With no bypass actor, an `update` rule refuses
every merge.

Each job of the table below carries the event in its name, `<Name> - ${{ github.event_name }}`, as
the test job of `ReusableTest.yml` already did. The rule requires the names that end in `- push`,
from the GitHub Actions app alone. A skipped pull-request run never carries such a name, so no
required check can read a skipped run.

| Workflow | Required check |
| -------- | -------------- |
| `Test.yml` | `test (1, ubuntu-latest, x64, false) / Julia 1 - ubuntu-latest - x64 - push` |
| `JET.yml` | `JET - push` |
| `Complexity.yml` | `Complexity - push` |
| `Aqua.yml` | `Aqua - push` |
| `Docs.yml` | `Documentation - push`, `Doctests - push` |
| `Paper.yml` | `Paper - push` |
| `FormatCheck.yml` | `Format check - push` |
| `LinkChecker.yml` | `Link checker - push` |

The `Coverage ratchet` job is not required. A trip turns its check red (ADR 0082, amendment of
2026-10-30), but the merge does not wait for it.

The push trigger of `JET.yml` has no `paths:` filter. With the filter, a push to `dev` that touches no
JET path leaves the head with no `JET - push` check, and the merge waits forever. From 2026-10-22 to
2026-10-26, the filter skipped 4 of 117 pushes to `dev`. The pull-request trigger keeps its filter.

The rule does not require the head of the pull request to contain `main`. A required current
base makes each merge restart the CI of every other open pull request, and ADR 0240 lands each
feature through its own pull request. The push run on `main` after the merge tests the tree that
the merge writes.

## Consequences

- One push to `dev` runs each check once while a pull request from `dev` is open. The one
  exception is the `Documentation` job of `Docs.yml`, which runs twice: the push run builds the
  docs, and the pull-request run builds them again to deploy the preview.
- No run tests the merge commit of a pull request from `dev*` or `agents/**` before the merge.
  This is the trade that the maintainer chose over a second run of each check (#1783, option 1).
  The push run on `main` tests the tree of the merge after it, and ADR 0240 states what a red
  run on `main` asks for.
- No push trigger of these eight workflows carries a `paths:` filter, so each push to `dev` gives
  its head every check of the table.
- A pull request into `main` from a fork cannot merge. No push run reaches its head. A change from a
  fork goes onto an `agents/` branch of this repository first.
- A cancelled push run on the head of a pull request blocks its merge into `main` until it runs
  again.
- `Link checker - push` reads external links, so an external site that is down blocks the merge.
- A new workflow that triggers on both `push` and `pull_request` takes the same condition on each of
  its jobs. A new job that a merge into `main` must wait for carries the event in its name, and its
  `- push` name joins the `required_status_checks` rule of the ruleset.

## Amendment (2026-11-24)

The queue held 31 runs, and the runs of three merged branches and of an old head held slots.

1. **A newer push to a `dev-` branch cancels the runs of the old head.** The merge reads the checks of
   the new head alone. `Test.yml` and `Aqua.yml` had no concurrency group. They have one for each
   `dev-` branch now, and a group of its own for each other run, so each commit of `main`, `dev` and a
   tag keeps its run. The other workflows cancel on `refs/heads/dev-` as on `refs/pull/`.
2. **The close of a pull request cancels the push runs of its branch** (`ClosedBranch.yml`), unless
   the branch has another open pull request. The push to `main` tests the tree of the merge.
3. **The ruleset has a file.** `.github/rulesets/protect-main.json` holds the rules, and
   `Ruleset.yml` compares the live ruleset with it each day. On 2026-11-24 the live ruleset asked for a
   current base, which this ADR and ADR 0240 refuse, and it had the admin role as a bypass actor. The
   last seven merges landed before their `Test` check ended. The API shows the bypass actors to an
   admin alone, so the job does not compare them. The decision stays: no bypass actor.
4. **Each review thread is resolved before a merge, and a merge is a squash.** Both are in the file.
5. **`dev` follows `main`.** Sessions start in a checkout of `dev`, which stood five commits behind
   `main`, and they filed issues by rules that `main` had replaced. `SyncDev.yml` merges `main` into
   `dev` after each push to `main`, and a conflict turns it red. A merge by `GITHUB_TOKEN` starts no
   workflow, so that head gets its checks at the next push of a session.

Item 1 replaces the sentence of ADR 0240, § *Own branch*, that each commit of a branch keeps a run.
