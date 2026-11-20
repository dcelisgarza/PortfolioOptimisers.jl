---
status: accepted
---

# A feature lands in `main` through its own pull request

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

Two facts limit the choice:

- The test suite takes about three hours of wall time on one runner. On 2026-11-19, four green
  runs of `Test` on `dev` took from 2 h 42 min to 3 h 58 min, with the queue time.
- The owner of the repository is a user account, not an organisation. GitHub gives a merge queue
  only to a repository of an organisation.

## Decision

Each feature or fix is one branch and one pull request into `main`.

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
4. **The merge.** A pull request merges by squash. Each feature is then one commit on `main`.
   Its subject ends with the number of its pull request, and its body holds the message of each
   commit of the branch (the settings `COMMIT_OR_PR_TITLE` and `COMMIT_MESSAGES`). One feature can be reverted with one
   commit, and `git bisect` steps over features.
5. **The base need not be current.** The ruleset "Protect Main" does not require that the head
   contains `main` (`strict_required_status_checks_policy = false`). GitHub refuses a merge with
   a textual conflict, and the branch then rebases. Two branches that pass alone can conflict in
   meaning. The push run on `main` after each merge finds such a conflict. Its fix is a new pull
   request, and that pull request comes before every other one.
6. **The issue.** The body of the pull request names `Closes #N`, and the merge closes the
   issue. A part of the work names `Part of #N`, and the issue stays open.
7. **The release.** The maintainer can register any commit of `main` whose push checks are
   green. A release can follow each feature, or a set of features.

`dev` retires when #1817 merges. Every tool that measured against `origin/dev` measures against
`origin/main`:

| Tool | Change |
| ---- | ------ |
| `code_health/gate_stamp.sh` | `GATE_STAMP_REF` defaults to `origin/main`. The guard acts on a push to `refs/heads/agents/*`, and it reads the tip of the ref, not the remote tip of the pushed branch. The first push of a branch has no remote tip, and the guard checks it too. |
| `code_health/sweep_check.jl` | `--base` defaults to `origin/main`. |
| `test/run_files.jl` | The `PATCH` lines diff against the merge-base with `origin/main`. |
| `CodeHealth.yml`, `Sweep.yml` | The scheduled job checks out the default branch, `main`. ADR 0078 is amended. |
| `~/.claude/hooks/worktree-setup.py` | A new worktree stays at `origin/main` when `origin` has no `dev` branch. |

## Considered options

- **Keep `dev`, and open a smaller pull request from `dev` into `main` more often.** Each
  feature still waits for the features that landed before it on `dev`, and each change goes
  through two merges. Rejected.
- **Require a current base.** Each merge into `main` makes every other open pull request rebase
  and run the three hours of CI again. With many sessions in parallel, the merges go one at a
  time. Rejected.
- **A merge queue.** It tests the merge of each pull request with the ones before it, and it
  removes the risk of item 5. GitHub does not give it to a repository of a user account.
  Rejected. If the repository moves to an organisation, this option replaces item 5.
- **Merge by rebase.** It keeps each commit of the branch on `main`. The commits of a branch were
  never tested one at a time, and the revert of one feature becomes many commits. Rejected.
- **A new branch prefix, `feature/**`.** The prefix reads better for a person. It needs a change
  to the push trigger and to the condition of ADR 0220 in each of eight workflows, and `agents/**`
  already does the job. Rejected.

## Consequences

- A feature reaches `main` about one CI run after its last push.
- Each push to a branch runs the full CI, as each push to `dev` did. `Test.yml` has no
  concurrency group, so its cost does not change. A workflow with a concurrency group, such as
  `JET.yml` or `Docs.yml`, replaced a queued run on `dev` with the run of a newer push. On many
  branches, it runs once for each branch. A public repository pays no minutes, but the plan runs
  at most 20 jobs at a time, so many branches at a time queue.
- Each pull request deploys a preview of the docs of about 0.26 GiB, and GitHub Pages refuses a
  site above 10 GiB. About 35 open pull requests fill the site. `DocPreviewCleanup.yml` deletes a
  preview when its pull request closes.
- `main` can go red after the merge of two pull requests that conflict in meaning. A release
  never comes from a red commit.
- A squash gives a new commit, so the commits that the guard records in `gate_pushed` never
  reach `main` by their own hash. The record does no harm, and it does no work.
- A pull request that goes red after its session ended stays open with its checks. The next
  session on its issue takes the branch.
- A branch cut from `dev` before the switch moves with `git rebase --onto origin/main origin/dev`.
  The squash of #1817 gives `main` the tree of `dev`, so the rebase has no conflict.
