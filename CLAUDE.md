# Working agreements for PortfolioOptimisers.jl

A Julia portfolio-optimisation library. Estimators are configuration, Results are data, and the
domain vocabulary is normative. Read `GLOSSARY.md` before you touch anything that you cannot already
name.

## Orientation

- **The standing test picks every design**: the form that improves the code most by
  architecture, maintenance, ergonomics and Julia idiom, with no loss of performance. Make the
  better change even when it breaks an interface or replaces a past choice, and name the break in
  `docs/src/migration.md`. `.github/instructions/julia-source-code.instructions.md`
  § *A design is chosen by the standing test* owns the rule.
- **`CODING_STANDARDS.md` is the map.** It routes a subject, such as a docstring, an `export`, a
  constructor or a test, to the file that owns the rule and to the check that fails when the rule
  breaks. It carries no rules of its own. `/improve-standards` audits it.
- **`GLOSSARY.md` defines the domain language.** When you introduce or rename a concept, update it
  in the same change.
- **`docs/adr/` records decisions.** Its `README.md` states when an ADR is amended and when it is
  rewritten in place.
- **`src/CLAUDE.md` lists the design rules for library code**, one line each, with a link to the
  section that owns each rule. Read it, and the sections it links, before you change or plan a
  change to a file under `src/` or `ext/`.

## Parallel sessions

- **Land the work by the route that its issue states** (ADR 0240). Each issue holds one
  `**Lands on:**` line. Write it in each issue that you file, and in an issue that you take and that
  has none, before your first edit. The first true row picks the value, unless the maintainer names
  one.

  | | True when | `**Lands on:**` |
  | --- | --- | --- |
  | A | A decision, a measurement or a research ticket that lands nothing | `no code` |
  | B | Not a sweep sub-issue, and an open pull request is wrong or incomplete without the work, or makes the defect, makes it worse or makes it reachable, or the work edits what that pull request adds or changes | `same branch #<pr>` |
  | C | The work needs code that only an open pull request has, it is a review of its own, and it starts now | `stacked on #<pr>` |
  | D | As row C, but the work can wait for the merge, as a sweep sub-issue can | `own branch after #<pr>`, `dev after #<pr>` |
  | E | A ticket of a map whose tickets make sense on `main` only together | `map #<map>` |
  | F | An issue that the review of a pull request found, so it is independent of that pull request | `own branch` |
  | G | A sweep sub-issue, an ergonomics change, a small performance fix, an unusual composition bug or a docs fix, and `dev` or `main` has the code that it changes | `dev` |
  | H | Anything else that lands: a standalone feature, an important correctness fix, a large performance fix | `own branch` |

  Write `dev-<feature>` in place of `#<pr>` while that branch has no pull request. The value ends the
  line, and the reason goes below it. For a pull request whose head is `dev`, rows B to D give `dev`.
  A map issue holds `one branch dev-<feature>` or `each ticket`.
  `bash code_health/lands_on.sh issue <number>` checks the line.
- **Load the `land-work` skill before the first edit of work that lands**, and before you open,
  stack or review a pull request. It holds the branch, the base and the push of each route.
- **Never merge a pull request into `main`, and never turn on auto-merge.** Open it, and stop. The
  maintainer merges it after a review.
- **Name a branch for its feature**, with the prefix `dev-`, for example
  `dev-concentration-limit`. A branch that holds research alone can take the prefix `research-`,
  which runs no CI and does not merge.
- **Work in your own worktree.** Create it with the `EnterWorktree` tool before your first edit, and
  reset it to the base of the route.
  Leave with `ExitWorktree` `remove`. When `git status --short` is empty and
  `git branch -r --contains HEAD` lists the remote branch that you pushed to, pass
  `discard_changes: true`.
- **Never force-push `dev` or the branch of a map.** Only the session that owns a stack force-pushes
  its layers.
- **A change to `CLAUDE.md`, to a skill or to the landing rules lands on `own branch`.** Sessions
  read these files from `main` and from `dev`, and a job merges `main` into `dev`.
- **Never commit a change you did not make.** Check `git status`, and stage the files you edited
  by name, never with `git add -A`.
- **Do not edit a file at the same time as another session.** The session that finishes last
  rebases onto the first, then adds its change to the current text. The `sweep-conform` skill
  lists the files that every sweep ticket shares, and the rule for a library-wide ratchet.
- **Do not spawn a fork subagent.** A fork copies the whole conversation and runs on the model of
  this session. Spawn the general-purpose or the Explore agent, give it the context it needs in the
  prompt, and choose its model for its task with the model parameter of the Agent tool: haiku for a
  lookup or a search with a known target, sonnet for reading code across files, a census of sites
  or a routine edit, and opus for a design question, a review that must judge, or a hard defect.

## Running Julia

- **Go through the kaimon MCP tools** to explore and probe, not `julia` on Bash. Start a REPL for
  your own worktree with `start_session(project_path=…)`, and pass its session key to every `ex`
  call. A REPL that another session opened is not yours to restart.
- **Run single-threaded**, with `julia -t 1` and `BLAS.set_num_threads(1)`. Never start the full
  test suite or a docs build. Those are the maintainer's to run.
- **Load the `run-tests` skill before you run a test file, a coverage run or a gate.** It holds the
  machine slots, the targeted tests, the coverage rules and the stamps that the push guard reads.
- On a Revise world-age warning, **restart the session and cold-load**. If a session hangs, shut it
  down and start a new one.

## Editing

- **Only edit `.jl`.** The docs build writes these files, and git does not track them:
  `examples/**/*.ipynb`, `docs/src/examples/**`, `docs/src/user_guide/**`,
  `docs/src/capability_catalogue.md`, `docs/src/TypeHierarchy.md`. Their sources are
  `examples/**/*.jl`, `user_guide/*.jl`, `docs/capability_catalogue.jl` and
  `docs/generate_type_hierarchy.jl`. `docs/src/public_api/**` and `docs/src/private_api/**` are
  hand-written. Fix a broken link in a generated page at its `.jl` source alone (ADR 0151).
- **Edit a `.jl` file with the Edit tool.** A change at many sites goes through the kaimon edit
  tool with an absolute file path. A Python replace from a Bash heredoc halves a backslash.
- **JuliaFormatter escapes a quote inside a jldoctest block as `\"`.** This is normal output. Do
  not revert the escaping. Long string literals and docstring prose are exempt from the margin.
  Code lines are not.
- **Doctests:** see the `run-doctests` skill.

## Defects you find

- **Fix a defect as soon as you find it when the fix is small, or when it belongs to the work in
  hand.** Cover it with a test, and name the fix in your report.
- **File an issue when the fix is large, or when it needs the maintainer's decision.** Give the
  reproduction, the file and the line, the cause, the fix you would make, and the reason you did
  not make it. Name the work that found it in the issue, and name the issue in the pull request,
  the commit or the report of that work. Write its `**Lands on:**` line (§ *Parallel sessions*).
- **A name or an interface that a library precedent settles is not the maintainer's decision**,
  and a breaking change is not either (`.github/instructions/julia-source-code.instructions.md`
  § *A design is chosen by the standing test*). A case that the same mechanism reaches belongs to
  the fix in hand: list every such case in the plan before the first test run.

## Functionality you add

Every file under `src/` and `ext/` carries a row in `code_health/sweep_manifest.toml`, and an
addition joins the systematic audit of #404 in the same change, in four steps:

1. Add or correct the row of the file, with `swept = false` for a new file.
2. Cover every line of a new file, or give it a Coverage Exemption (ADR 0082).
3. Reopen the child map that owns the file, and reopen #404.
4. Open one sub-issue of that child map for the addition (ADR 0084).

Before the first commit that changes `src/` or `ext/`, load the `sweep-conform` skill. It
details each step, runs the check that reports them (`code_health/sweep_check.jl --fetch`), and
gives the order of the gates in its § *Land the branch*. The `sweep-file-issues` skill does
steps 3 and 4. Defer a feature whose design is not settled, as `sweep-conform` states.

## Repo etiquette

- **Never link to or post in repositories outside `dcelisgarza`'s.** Name external sources in
  prose. A page or a docstring can link the published docs site of an external package (#1794),
  or its JuliaHub page, `https://docs.juliahub.com/General/<Package>/stable/`, when it has no
  site. A bibliography entry in a `.bib` file can link the repository of the work it cites. A
  comment that cites a source file of another repository names the package, the commit and the
  path in prose. `test/test_64_docs_page_metadata_census.jl` fails on a link to an external
  repository.
- **A session that works an issue commits without being asked.** It commits and pushes by the
  route of its issue, as § *Parallel sessions* states. Any other session commits only when asked.
- **Assign the issue to your `gh` account before your first edit**, with
  `gh issue edit <number> --add-assignee @me`. An issue that you fix, close, or leave open with a
  part of the work done carries your account.
- **Close the issue when its fix is pushed, not when the pull request merges**, for every kind
  of issue and every route: a defect, a feature, a sweep sub-issue, a map. When the push that
  holds the last part of the work reaches the remote branch, close the issue with
  `gh issue close 493 --comment "Fixed in <sha> on <branch>."`. A map closes when its last ticket
  closes. Name the issue in the commit message, for example `Fix #493: …`. On your own branch,
  also write `Closes #493` in the body of the pull request, so that the pull request links the
  issue. A commit of a part of the work names `Part of #493: …`, with no closing keyword, and the
  issue stays open while a part of the work remains. A squash merge puts every commit message
  into `main`, and a `Fix #493` there closes a partial ticket that a later session reopened.
