---
name: land-work
description: "Land a change of PortfolioOptimisers.jl by the route of its issue (ADR 0240): cut the branch, push, open a standalone or a stacked pull request, review it, and merge it. Use before the first edit of work that lands, when you open, stack, review or merge a pull request, and when an issue waits for an open pull request."
---

# Landing work

The `**Lands on:**` line of the issue names the route. `CLAUDE.md` § *Parallel sessions* holds the
ladder that picks it, and `code_health/lands_on.sh` holds the grammar. Check the line first:

```bash
bash code_health/lands_on.sh issue <number>
```

Go on when it prints the route. When it refuses, correct the line. When it exits 3, write the line.

## Cut the branch

| Route | Start the worktree at | Landing base | Push to |
| --- | --- | --- | --- |
| `own branch` | `origin/main` | none to store | `dev-<feature>` |
| `own branch after <target>` | `origin/main`, when `main` has `<target>` | none to store | `dev-<feature>` |
| `stacked on <target>` | the head branch of `<target>` | `origin/<that branch>` | `dev-<feature>` |
| `same branch #<pr>` | the head branch of `#<pr>` | `origin/<that branch>` | that branch |
| `dev`, `dev after <target>` | `origin/dev`, when it has `<target>` | `origin/dev` | `dev` |
| `map #<map>` | the branch that the map names | `origin/<that branch>` | that branch |

A new worktree can start at `origin/dev`. Reset it to the second column before the first edit.
Store a landing base with `bash code_health/gate_stamp.sh base origin/<branch>`, so that the gates
measure against it. The name of a branch holds no `/`, because the push filter `dev*` matches none,
and the pull request then gets no checks.

`same branch`, `dev` and `map` are **shared branches**. Commit, rebase onto the remote branch, and
push as a fast-forward. An `own branch` is yours until its pull request has a `same branch` issue.

## Open the pull request

1. Push, and close the issue when the push holds the last part of its fix (`CLAUDE.md` § *Repo
   etiquette*).
2. Open one pull request, with the body of the `pr` skill and `Closes #<issue>`. The base is `main`,
   or the branch below in a stack. A `dev` or `map` route has one draft pull request for the branch.
3. Check it with `bash code_health/lands_on.sh pr <number>`. The check `Route` runs the same script.
4. A pull request that needs a ruling of the maintainer stays a draft.

Two pull requests that each take the next version depend on each other for no reason. So change
the version only in a release pull request, or when the maintainer asks to release with this pull
request. Then make it the last commit, on the current `main`, when the merge is next.

## A stack

A stack is a chain of pull requests. Each layer shows its own diff, and the rules of `main` hold for
each layer. Use one when the upper change needs the code of the lower change, and each is a review of
its own. Three layers are the most: a push to a layer runs the CI of each layer above it, and one
run takes about three hours.

1. Open the upper pull request with the head branch of the lower one as its base.
2. Make the two a stack, bottom first:
   `gh api repos/{owner}/{repo}/stacks -H "X-GitHub-Api-Version: 2026-03-10" -F "pull_requests[]=<lower>" -F "pull_requests[]=<upper>"`.
   The answer holds the number of the stack. `repos/{owner}/{repo}/stacks/<number>/add` adds a layer
   at the top, and `…/<number>/unstack` takes the stack apart. Neither moves a branch.
3. One session owns a stack. After a change to a lower layer, rebase each layer above it and push
   with `--force-with-lease`. Force-push a layer of your own stack alone, and never `dev` or the
   branch of a map.
4. Keep the history of a stack linear: rebase the bottom layer onto `main`, then each layer above it,
   and merge `main` into none. Each rebased layer runs the CI again, so rebase when a conflict or a
   review asks for it, and not at each move of `main`.
5. A stack merges from the bottom up. Merge the whole stack at once when each layer is green, so that
   no layer runs the CI again after a rebase.

A layer that gets no review of its own is not a layer. Put it on the `same branch`.

## Review

Another session than the author reviews a pull request before it merges. Give it the pull request
and `CODING_STANDARDS.md`, not the conclusion of the author. It does these steps:

1. Read the diff of the layer, and each issue that it closes.
2. Check each row of `CODING_STANDARDS.md` that says review. A row with a check in CI needs no
   reader.
3. Fix each defect on the branch, or file an issue, with a route from the ladder. Leave none.
4. Record the verdict. The body states the defects, the fixes, and what the review did not read:
   `bash code_health/review_stamp.sh post <pr> <success|failure> <body file>`.

A new push keeps the verdict when the diff did not change. When it changed, review what changed, and
record the verdict again.

For each open `same branch` issue, write `Waits for #<issue>` in the body of the pull request, and
run `gh pr merge <pr> --disable-auto`. Change the words to `Closes #<issue>` when its fix is pushed.
After the last one, turn auto-merge on again, unless the maintainer merges the pull request. When
the pull request merged first, the route of the issue is `own branch`.

## Merge

- **A standalone pull request.** After a review of `success`, run
  `gh pr merge <number> --auto --squash`. GitHub merges it when the required checks pass. A rebase
  is necessary only when GitHub shows a conflict.
- **A stack.** The maintainer merges it. GitHub documents no auto-merge for a stack yet.
- **`dev`.** The maintainer merges it as a batch.
- **A map.** The session that lands the last ticket marks the pull request ready, and it is then a
  standalone pull request.

When the merge is refused, read why, and report it. A required check that did not finish is a wait,
not a reason to bypass.
