# Working agreements for PortfolioOptimisers.jl

A Julia portfolio-optimisation library. Estimators are configuration, Results are data, and the
domain vocabulary is normative. Read `GLOSSARY.md` before you touch anything that you cannot already
name.

## Orientation

- **`CODING_STANDARDS.md` is the map.** It routes a subject, such as a docstring, an `export`, a
  constructor or a test, to the file that owns the rule and to the check that fails when the rule
  breaks. Start
  there when you do not know which file governs what you are about to change. It carries no rules
  of its own, so it never overrides the file it points at. `/improve-standards` audits it.
- **`GLOSSARY.md` is the glossary.** It defines the domain language (Prior, Constraint Space,
  Universe Sets, Feature Matrix, Objective Penalty, …) and is hand-written. When you introduce or
  rename a concept, update it in the same change.
- **`docs/adr/` records decisions.** Its `README.md` states when an ADR is amended and when it is
  rewritten in place.

## Parallel sessions

Sessions run in parallel, so the checkout is not yours alone. Two rules keep them apart.

**Work in your own worktree.** Create it with the `EnterWorktree` tool before your first edit.
Commit there, then rebase onto `dev`, so the merge is a fast-forward that carries only your work.
Merge into the local `dev`, and push `dev` to `origin`, never the worktree branch. After the merge,
leave the worktree with `ExitWorktree`, then delete the branch. Never commit a change you did not
make: check `git status` before you stage, and stage the files you edited by name rather than with
`git add -A`.

**Do not edit a file at the same time as another session.** Two sessions can change the same file
during one ticket, one after the other. Only a simultaneous edit loses work. The session that
finishes last commits the common file. That session rebases onto the commit of the session that
finished first, then adds its own change to the current text.

Every sweep ticket changes three common files, and every ticket that adds an ADR changes a fourth.
This rule matters most for these files:

- `code_health/sweep_manifest.toml` holds one row per source file.
- `src/01_Base/01_DocstringDictionaries/` holds `arg_dict`, `math_dict` and their siblings, one
  subject to a file. Two sessions collide only when they add keys of one subject.
- `test/test_26_docs.jl` holds the library-wide ratchets, `DETAILS_TOTAL`, `NO_RELATED_TOTAL` and
  `MATH_COPY_TOTAL`.
- `docs/adr/README.md` is the ADR index, with one row per ADR. Two new ADRs add adjacent rows.

A ratchet is one number for the whole library. Two sessions that each lower it write two different
numbers, and a merge keeps one of them. Lower a ratchet in the commit that does the work it counts.
Rebase before you merge, so that you lower the number that is current.

## Running Julia

- **Go through the kaimon MCP tools** to explore and probe, not `julia` on Bash. `ex(e="…")`
  evaluates in a REPL the user shares live. Start a REPL for your own worktree with
  `start_session(project_path=…)`, and pass its session key to every `ex` call. A REPL that another
  session opened is not yours to restart. Test files, doctests and the `code_health/` gates run in
  a fresh bare process instead.
- **Run single-threaded**, with `julia -t 1` and `BLAS.set_num_threads(1)`. Never start the full
  test suite or a docs build. Those are the maintainer's to run.
- **A run that loads the package queues for a machine slot.** The test, doctest, JET and
  Expansion Bound drivers wait for a free slot and free memory (`code_health/julia_slot.jl`), and a
  `SLOT wait` line names the holders. Shut down your own kaimon REPL before a full doctest run or a
  JET run. A probe that needs the test dependencies starts its REPL in the test folder of the
  worktree.
- Run **targeted** `test_*.jl` files for the area you changed with
  `julia -t 1 --project=test test/run_files.jl --done=<path> test_X.jl …`, as a background Bash
  command. It runs each file after the `init_code` preamble of `test/runtests.jl`,
  prints one `RESULT` line per file, and writes the verdict to `<path>` when it ends. The header of
  `test/run_files.jl` explains every option. Rerun a failure that only CI shows with `--ci`.
- **A targeted test is one whose result the change can alter.** Trace from the changed code to the
  test files that assert a value passing through it, add the file of the new test, and add the
  censuses that read what the commit touched. A file that only names a changed type is not
  targeted. After a rebase, run `bash code_health/gate_stamp.sh check`. Rerun the files it names
  that your change reaches, and paste the `drop` line it prints for the rest.
- On a Revise world-age warning, **restart the session and cold-load**. If a session hangs, shut it
  down and start a new one. Do not try to recover it.

## Editing

- **Only edit `.jl`.** The docs build writes these files, and git does not track them:
  `examples/**/*.ipynb`, `docs/src/examples/**`, `docs/src/user_guide/**`,
  `docs/src/capability_catalogue.md`, `docs/src/TypeHierarchy.md`.
  Their sources are `examples/**/*.jl`, `user_guide/*.jl`, `docs/capability_catalogue.jl`,
  `docs/generate_type_hierarchy.jl`.
  `docs/src/public_api/**` and `docs/src/private_api/**` are hand-written. No gate reads a
  generated page from the tree. The Docs build writes the generated pages fresh, and its
  cross-reference check fails on a link to a page that does not exist. Fix a broken link in a
  generated page at its `.jl` source alone (ADR 0151).
- **JuliaFormatter runs over the whole repository, and it formats in place every file that it
  reaches.** It escapes a quote inside a jldoctest block as `\"`, and this is normal output. Inside
  a `"""` docstring, `\"` renders as `"`, and the doctests pass. **Do not revert the escaping.**
- In practice, long string literals and docstring prose are exempt from the margin. Code lines are
  not.
- **Edit a `.jl` file with the Edit tool.** A change at many sites goes through the kaimon edit
  tool with an absolute file path, which parses each file before it writes. A Python replace from
  a Bash heredoc halves a backslash and has corrupted docstrings.

## Doctests

- See the `run-doctests` skill for the fresh-process rule and the exact CI invocation to mirror.

## Design rules

The design rules for library code live in `src/CLAUDE.md`. Read them before you change or plan a
change to a file under `src/` or `ext/`.

## Defects you find

- **Fix a defect as soon as you find it when the fix is small, or when it belongs to the work in
  hand.** Cover it with a test, and name the fix in your report. If you leave a defect, the next
  reader must find it again.
- **File an issue when the fix is large, or when it needs the maintainer's decision.** Put every
  piece of context you hold into the issue: the reproduction, the file and the line, the cause you
  found, the fix you would make, and the reason you did not make it. Someone else must be able to
  work from the issue alone.
- **A name or an interface that a library precedent settles is not the maintainer's decision.**
  Follow the precedent. A case that the same mechanism reaches belongs to the fix in hand: list
  every such case in the plan, before the first test run.
- **Link the issue to the work that found it.** Name the issue in the pull request, the commit, or
  the report for that work, and name that work in the issue.

## Functionality you add

Every file under `src/` and `ext/` carries a row in [`code_health/sweep_manifest.toml`](code_health/sweep_manifest.toml)
naming the child map of the systematic audit that owns it. `test/test_45_sweep_census.jl` fails the
build when a file carries no row, and when the unit count of a file differs from its row.

Add your addition to the audit in the same change:

1. Add or correct the file's row. The census prints the line to paste. Take `map` from the check
   below. The check names the map when the directory uses one, and lists the candidates when the
   directory uses several. Set `swept = false`. A swept row also records the name of the binding
   each unit attaches to. A unit added to a swept file, or replaced one for one in it, keeps
   `swept = true`. The commit sweeps the new units itself, and steps 3 and 4 do not apply. If the
   commit cannot sweep them, it sets `swept = false` and takes all four steps (ADR 0148).
2. Cover every line of a new file, or give it a Coverage Exemption. ADR 0082 owns that rule.
3. Reopen the child map that owns the file, and reopen its umbrella, issue #404.
4. Open one sub-issue of that child map for the addition, so that the audit sweeps it in the same
   way as the code it joins.

`julia --project=code_health code_health/sweep_check.jl --fetch` reports all four steps for the
files your branch touches, and prints the line to paste for step 1. It measures and writes nothing.
Before the first commit that changes `src/` or `ext/`, load the `sweep-conform` skill and follow
its § *Land the branch*. That section runs the check, and it gives the order of the gates whose
stamps the push guard requires. The `sweep-file-issues` skill does steps 3 and 4.

Defer a feature whose design is not settled. Leave its row at `swept = false`, record it in the
child map's *Not yet specified* section, and take steps 3 and 4 when the design settles.

## Repo etiquette

- Never link to or post in repositories outside `dcelisgarza`'s. Name external sources in prose.
  There is one exception (#1794). A page or a docstring can link the published docs site of an
  external package, never its repository. Link a package that has no docs site of its own at its
  JuliaHub page, `https://docs.juliahub.com/General/<Package>/stable/`. A comment that cites a
  source file of another repository names the package, the commit and the path in prose.
  `test/test_64_docs_page_metadata_census.jl` fails on a link to an external repository.
- Branch before committing if you are on the default branch.
- **A session that works an issue commits without being asked.** It commits, rebases and pushes
  `dev` as § *Parallel sessions* states. Any other session commits only when asked.
- **Assign an issue to your `gh` account when you start to fix it.** Run
  `gh issue edit <number> --add-assignee @me` before your first edit. Every issue you fix, close or
  leave open with a part of the work done, carries your account as an assignee.
- **Close an issue yourself after you commit the work that resolves it.** This holds for every kind
  of issue: a defect, a feature, a sweep sub-issue, a map. Name the issue in the commit message, for
  example `Fix #493: …`. A closing keyword does not close the issue. GitHub acts on the keyword only
  when the commit reaches the default branch `main`, and `main` is current to the last release. Run
  `gh issue close <number> --comment "…"` after you commit, and name the commit in the comment. If
  the commit resolves only a part of the issue, leave the issue open. Close it after the commit that
  finishes the work.
