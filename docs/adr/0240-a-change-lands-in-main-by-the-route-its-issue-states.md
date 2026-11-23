---
status: accepted
---

# A change lands in `main` by the route that its issue states

## Context

Until this decision, every session pushed its work to `dev` as a fast-forward. One pull request
then took `dev` into `main` for a release. The release pull request #1817 opened on 2026-10-23.
On 2026-11-20 it held 567 commits that `main` did not have, and its review was at round 79.

This form has four costs:

- A finished feature waits for the whole bundle. A user of the registered package gets it weeks
  after its last commit.
- One review covers hundreds of unrelated changes, so the review of the bundle is slow.
- A closing keyword acts only when a commit reaches `main`. A `Fix #N` on `dev` closes issue `#N`
  only at the release merge, also when the work is partial.
- A revert of one feature from `main` is not possible. The squash commit of the release holds
  every feature of the bundle.

Not every change has these costs to the same degree. A standalone feature or an important
correctness fix loses value each day that it waits. A sweep ticket, an ergonomics change or a fix
of an unusual composition can wait for a batch with no loss. And the tickets of some maps make
sense on `main` only together: a half-landed interface break leaves `main` in a mixed state.

Three facts limit the choice:

- The test suite takes about three hours of wall time on one runner. On 2026-11-19, four green
  runs of `Test` on `dev` took from 2 h 42 min to 3 h 58 min, with the queue time.
- The plan runs at most 20 jobs at a time, and each pull request deploys a preview of the docs of
  about 0.26 GiB. GitHub Pages refuses a site above 10 GiB.
- The owner of the repository is a user account, not an organisation. GitHub gives a merge queue
  only to a repository of an organisation.

## Decision

Each change lands in `main` by one of three routes. The issue states its route when it is
created.

### The routes

| Route | Branch | For |
| ----- | ------ | --- |
| **Own branch** | `agents/<issue>-<slug>`, from `origin/main` | A standalone feature. An important correctness fix: a wrong result on a path that users take. A large performance fix. |
| **`dev-fixes`** | `dev-fixes`, one branch that many sessions share | A sweep sub-issue of #404. An ergonomics change. A small performance fix. A fix of an unusual composition. A docs fix. |
| **Map** | `agents/<map>-<slug>`, from `origin/main` | Each ticket of a map whose tickets make sense on `main` only together. |

A map whose tickets can land one at a time does not get a map branch. Each of its tickets takes
the route of its own kind. A sweep map of #404 is always of this kind, because its tickets are
independent and the map stays open for months.

### The issue states the route

1. **The line.** The body of each issue holds one line, `**Lands on:**`, with one of these
   values:
   - `own branch`;
   - `dev-fixes`;
   - `map #<map>`, for a ticket of a map with a map branch.

   The body of a map issue holds `**Lands on:** one branch, agents/<map>-<slug>`, or
   `**Lands on:** each ticket by its own route`.
2. **The call.** The author of the issue makes the call when they create the issue, from the
   table of § *The routes*. `code_health/sweep_issues.sh` writes `dev-fixes` in each sweep
   sub-issue.
3. **The maintainer's word wins.** A route that the maintainer names in the issue, or to the
   session, replaces the table.
4. **A missing line.** An issue with no line, such as one from an issue form, gets its line from
   the session that takes it. The session adds the line before its first edit.

### Own branch

1. **The branch.** A session cuts `agents/<issue>-<slug>` from `origin/main`, for example
   `agents/2205-ew-mean-restart`. A person uses the same prefix. A branch name with no issue
   uses a slug alone.
2. **The checks.** A push to the branch runs every workflow, as a push to `dev` did. The push
   trigger of the eight workflows of ADR 0220 already matches `agents/**`, and the condition of
   ADR 0220 skips the pull-request run of each job. The pull request shows the push checks of its
   head, so each commit of the branch keeps a run that a reader can trace.
3. **The pull request.** The session opens the pull request into `main` and turns on auto-merge
   with `gh pr merge <number> --auto --squash`. GitHub merges it when the required checks pass,
   and nobody waits for the three hours. A pull request that needs a ruling of the maintainer
   opens as a draft, with no auto-merge.

### `dev-fixes`

1. **The branch.** `dev-fixes` has no `agents/` prefix. The push filter `dev*` and the `dev`
   clause of the condition of ADR 0220 already match it, so its pushes run every workflow, and
   its pull request skips the duplicate runs.
2. **The work.** A session starts its worktree at `origin/dev-fixes`. It commits, rebases onto
   `origin/dev-fixes`, and pushes with `git push origin HEAD:dev-fixes` as a fast-forward, as
   sessions did on `dev`.
3. **The pull request.** One draft pull request takes `dev-fixes` into `main`. The session that
   creates the branch opens it. The maintainer marks it ready and merges it when they choose, so
   a batch lands at once.
4. **After the merge.** The merge deletes the branch. The next session that needs it creates it
   again from `main`, with `git push origin origin/main:refs/heads/dev-fixes`, and opens a new
   draft pull request. A session with commits on the old branch moves them with
   `git rebase --onto origin/dev-fixes <old base>`. The squash gives `main` the tree of the old
   branch, so the rebase has no conflict.

### Map

1. **The branch.** The first session on the map cuts `agents/<map>-<slug>` from `origin/main`
   and opens one draft pull request into `main`.
2. **The work.** Each ticket session works on the map branch as a session works on `dev-fixes`.
3. **The merge.** When the commit of the last ticket lands, the session that made it marks the
   pull request ready and turns on auto-merge.

### For every route

1. **The merge.** A pull request merges by squash. Each pull request is then one commit on
   `main`. Its subject ends with the number of its pull request, and its body holds the message
   of each commit of the branch (the settings `COMMIT_OR_PR_TITLE` and `COMMIT_MESSAGES`). One
   pull request can be reverted with one commit, and `git bisect` steps over pull requests.
2. **The base need not be current.** The ruleset "Protect Main" does not require that the head
   contains `main` (`strict_required_status_checks_policy = false`). GitHub refuses a merge with
   a textual conflict, and the branch then rebases. Two branches that pass alone can conflict in
   meaning. The push run on `main` after each merge finds such a conflict. Its fix is a new pull
   request, and that pull request comes before every other one.
3. **The issue.** A commit that resolves an issue names `Fix #N: …`. A squash copies every commit
   message into `main`, so the merge closes the issue. A pull request of the own-branch route
   also names `Closes #N` in its body. A commit of a part of the work names `Part of #N: …`, and
   the issue stays open.
4. **The release.** The maintainer can register any commit of `main` whose push checks are
   green. A release can follow each pull request, or a set of pull requests.

### The tools

`dev` retires when #1817 merges. A session measures its gates against the branch that it lands
on, its **landing base**. `bash code_health/gate_stamp.sh base <ref>` stores the landing base in
the git directory of the worktree, and `bash code_health/gate_stamp.sh base` prints it. With no
stored base, the landing base is `origin/main`.

| Tool | Change |
| ---- | ------ |
| `code_health/gate_stamp.sh` | The landing base is the ref of each stamp. `GATE_STAMP_REF` still overrides it. The guard acts on a push to `refs/heads/agents/*` and to `refs/heads/dev-fixes`, and it reads the tip of the landing base, not the remote tip of the pushed branch. The first push of a branch has no remote tip, and the guard checks it too. |
| `code_health/sweep_check.jl` | `--base` defaults to the landing base. |
| `docs/doctest_files.jl` | The skill names `--against $(bash code_health/gate_stamp.sh base)`. |
| `test/run_files.jl` | The `PATCH` lines diff against the merge-base with the landing base. |
| `CodeHealth.yml`, `Sweep.yml` | The scheduled job checks out the default branch, `main`. ADR 0078 is amended. |
| `~/.claude/hooks/worktree-setup.py` | A new worktree stays at `origin/main` when `origin` has no `dev` branch. A session on `dev-fixes` or a map branch resets its worktree to that branch before its first edit. |

## Considered options

- **One route, its own branch, for every change.** A sweep ticket or a small fix then costs a
  full CI run, a preview of the docs and a review of its own. Many such pull requests at a time
  fill the queue of 20 jobs and the 10 GiB of the site. The first form of this ADR chose it,
  and it was replaced on the same day. Rejected.
- **Keep `dev`, and open a smaller pull request from `dev` into `main` more often.** Each feature
  and each important fix still waits for the work that landed before it on `dev`. `dev-fixes`
  keeps this form only for work that can wait. Rejected for the rest.
- **A map branch for every map.** A sweep map stays open for months, and its finished tickets
  would wait for it. Rejected.
- **`agents/dev-fixes`.** The prefix puts every shared branch under one glob of the guard. The
  push filter `dev*` already matches `dev-fixes`, and the maintainer chose the shorter name.
  Rejected.
- **Require a current base.** Each merge into `main` makes every other open pull request rebase
  and run the three hours of CI again. With many sessions in parallel, the merges go one at a
  time. Rejected.
- **A merge queue.** It tests the merge of each pull request with the ones before it, and it
  removes the risk of a conflict in meaning. GitHub does not give it to a repository of a user
  account. Rejected. If the repository moves to an organisation, this option replaces item 2 of
  § *For every route*.
- **Merge by rebase.** It keeps each commit of the branch on `main`. The commits of a branch were
  never tested one at a time, and the revert of one feature becomes many commits. Rejected.
- **A new branch prefix, `feature/**`.** The prefix reads better for a person. It needs a change
  to the push trigger and to the condition of ADR 0220 in each of eight workflows, and `agents/**`
  already does the job. Rejected.

## Consequences

- A change on its own branch reaches `main` about one CI run after its last push. A change on
  `dev-fixes` reaches `main` when the maintainer merges the batch.
- The route is a fact on the issue, so a session reads it and does not guess it.
- Each push to a branch runs the full CI, as each push to `dev` did. A push that changes only
  prose runs it too, until #2226 lets such a push skip the long jobs. `Test.yml` has no concurrency group, so its cost does not change. A
  workflow with a concurrency group, such as `JET.yml` or `Docs.yml`, replaced a queued run on
  `dev` with the run of a newer push. On many branches, it runs once for each branch.
- About 35 open pull requests fill the site of the docs. `DocPreviewCleanup.yml` deletes a
  preview when its pull request closes. `dev-fixes` and the map branches keep the count low.
- `main` can go red after the merge of two pull requests that conflict in meaning. A release
  never comes from a red commit.
- A squash gives a new commit, so the commits that the guard records in `gate_pushed` never
  reach `main` by their own hash. On `dev-fixes` and a map branch, the record does its work: a
  session's own pushed commits are not a sibling's.
- A pull request that goes red after its session ended stays open with its checks. The next
  session on its issue takes the branch.
- A branch cut from `dev` before the switch moves with `git rebase --onto origin/main origin/dev`.
  The squash of #1817 gives `main` the tree of `dev`, so the rebase has no conflict.

## Amendment (2026-11-21)

The maintainer changed two parts of this decision.

1. **The shared branch is `dev`, not `dev-fixes`.** `dev` did not retire when #1817 merged. After
   the merge, `dev` moved to the squash commit on `main`, and the draft pull request #2227 takes
   it into `main` as a batch. Each text above that names `dev-fixes` now reads `dev`: the route
   value `**Lands on:** dev`, the push `git push origin HEAD:dev`, the landing base `origin/dev`,
   and the glob `refs/heads/dev` of the guard in `code_health/gate_stamp.sh`. The rejected option
   `agents/dev-fixes` is moot. After a batch merges, the maintainer moves `dev` to the squash
   commit on `main`. A session with commits on the old `dev` moves them with
   `git rebase --onto origin/dev <old base>`.
2. **An issue closes when its fix is pushed, not when the pull request merges.** This replaces
   item 3 of § *For every route*. When the push that holds the last part of the work reaches the
   remote branch, the session closes the issue with `gh issue close`, on every route. A commit of
   the work still names `Fix #N: …`, and a pull request of its own still names `Closes #N`, so
   that the issue links the work. A commit of a part of the work names `Part of #N: …`, and the
   issue stays open. The cost of § *Context* that a partial `Fix #N` closes an issue at the
   release merge stays: write `Part of #N` for a partial commit.

## Amendment (2026-11-23)

The maintainer changed the names of the branches.

1. **A branch takes the name `dev-<feature>`.** The prefix is `dev-`, and the rest names the
   feature, for example `dev-concentration-limit`. The name of a map branch is the feature of the
   map, and a release branch is `dev-release-x.y.z`. Each text above that names
   `agents/<issue>-<slug>` or `agents/<map>-<slug>` now reads `dev-<feature>`.
2. **A branch that holds research alone can take the prefix `research-`.** No push filter matches
   it, so a push runs no CI. The ruleset of `main` refuses its merge, because it has no push
   checks. Work that lands moves to a `dev-` branch.
3. **The tools.** The push filter `dev*` and the `dev` clause of the condition of ADR 0220 already
   match `dev-<feature>`, so no workflow changes. The guard of `code_health/gate_stamp.sh` acts on
   a push to `refs/heads/dev`, `refs/heads/dev-*` and `refs/heads/agents/*`.
4. **The old prefix.** No session makes a new `agents/` branch. A branch that has the prefix keeps
   its name until it merges, so the filter `agents/**` and the glob of the guard stay until then.

## Amendment (2026-11-23, issues from a review)

The maintainer ruled that an issue from the review of a pull request does not land on `dev`
without a check of what the issue belongs to. The table of § *The routes* sorts by the kind of
the change. A review of an open pull request often files a small fix of code that only the pull
request has. The table sends that fix to `dev`, and `dev` does not have the code. The pull request
then merges with a known defect, and its fix waits for the next batch.

1. **The fourth value.** The `**Lands on:**` line can hold `#<pr>`, the number of an open pull
   request. The work lands on the head branch of that pull request.
2. **The test.** An issue from the review of an open pull request is a part of that pull request
   when one of these is true:
   - The fix changes code, text or a test that the pull request adds or changes.
   - The pull request makes the defect, makes it worse, or makes it reachable.
   - The pull request is wrong or incomplete without the fix.

   Such an issue takes `#<pr>`, whatever its size and its kind.
3. **An independent issue.** An issue that passes none of the three takes `own branch`, also when
   its kind is in the `dev` row of the table. The fix starts from `origin/main`.
4. **The reason.** The author of the issue writes the reason for the route below the line. The
   maintainer's word still wins.
5. **The work.** The head branch is then a shared branch. A session resets its worktree to
   `origin/<head branch>`, stores that ref as its landing base, rebases, and pushes as a
   fast-forward. It never force-pushes. The issue closes at the push, as on every route.
6. **The merge.** The pull request does not merge while such an issue is open. The session that
   files the issue writes `Closes #<issue>` in the body of the pull request, and turns off
   auto-merge with `gh pr merge <pr> --disable-auto`. The session that closes the last one turns
   auto-merge on again, unless the pull request is a draft.
7. **After the merge.** When the pull request merged before the fix, the code is on `main`, and
   the route becomes `own branch`. The session that takes the issue changes the line.
8. **The limits.** When the head branch of the pull request is `dev`, a part of it lands on `dev`.
   A sweep sub-issue of #404 keeps the route that `code_health/sweep_issues.sh` writes.

Two other forms were rejected. With `dev` for every small issue, the pull request merges without
its fix. With the pull request for every issue, it grows past its one feature, and a fix that
`main` needs waits for its review.
