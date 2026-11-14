# Working agreements for PortfolioOptimisers.jl

A Julia portfolio-optimisation library. Estimators are configuration, Results are data, and the
domain vocabulary is normative. Read `GLOSSARY.md` before you touch anything that you cannot already
name.

## Orientation

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

- **Work in your own worktree.** Create it with the `EnterWorktree` tool before your first edit.
  Commit there, rebase onto `dev`, merge into the local `dev` as a fast-forward, and push `dev` to
  `origin`, never the worktree branch. Then leave with `ExitWorktree` and delete the branch.
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
  not make it. Name the work that found it, and name the issue in that work.
- **A name or an interface that a library precedent settles is not the maintainer's decision**,
  and a breaking change is not either (`.github/instructions/julia-source-code.instructions.md`
  § *A design is chosen by the standing test*). A case that the same mechanism reaches belongs to
  the fix in hand.

## Functionality you add

Every file under `src/` and `ext/` carries a row in `code_health/sweep_manifest.toml`, and an
addition joins the systematic audit of #404 in the same change. Before the first commit that
changes `src/` or `ext/`, load the `sweep-conform` skill. It gives the four steps of an addition,
the check that reports them (`code_health/sweep_check.jl --fetch`), and the order of the gates in
its § *Land the branch*. The `sweep-file-issues` skill does steps 3 and 4. Defer a feature whose
design is not settled, as `sweep-conform` states.

## Repo etiquette

- **Never link to or post in repositories outside `dcelisgarza`'s.** Name external sources in
  prose. `CODING_STANDARDS.md` lists the exceptions for a docs site and for a bibliography entry,
  and `test/test_64_docs_page_metadata_census.jl` fails on a link to an external repository.
- **A session that works an issue commits without being asked.** It commits, rebases and pushes
  `dev` as § *Parallel sessions* states. Any other session commits only when asked.
- **Assign the issue to your `gh` account before your first edit**, with
  `gh issue edit <number> --add-assignee @me`.
- **Close the issue yourself after the commit that resolves it**, with
  `gh issue close <number> --comment "…"` that names the commit. A closing keyword acts only when
  the commit reaches `main`, the default branch. Leave the issue open while a part of the work
  remains.
