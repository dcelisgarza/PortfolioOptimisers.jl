---
name: run-doctests
description: Run or update PortfolioOptimisers.jl doctests correctly — the fresh-process rule and the exact CI invocation to mirror. Use before running doctests, or before regenerating expected doctest output.
---

# Running doctests

- Run them in a **fresh process**, never in a working REPL. `show` omits a module prefix for types
  reachable unqualified from the active module, so a session that has done `using StatsBase` makes
  `StatsBase.SimpleCovariance` print bare and produces dozens of phantom failures.
- The full run is the `Run doctest` step of `.github/workflows/Docs.yml`. Run it through the
  driver, with `run_in_background`, which reads the step from the workflow so it cannot drift:

```bash
julia -t 1 --project=docs docs/doctest_files.jl --all
```

  It prints `DOCTEST all seconds=… status=green|red`. A green run records the stamp `doctest`,
  which the pre-push guard requires of a branch whose change to `src/` reaches code or a string
  literal. The run holds a slot of
  `code_health/julia_slot.jl`, so it waits for free memory rather than being killed for the lack
  of it. Shut down your own kaimon REPL first: it holds a GB that the run needs.

- The step passes `DOCTEST_FILTERS`, which cuts each decimal number to ten decimals, so a block
  does not fail on a host that differs in the last place of a float. A number that changes in
  its first ten decimals still fails.

- To check the blocks of a few files while you work, run `docs/doctest_files.jl` in a fresh
  process. It runs the setup of the CI job from `.github/workflows/Docs.yml`, then the doctests of
  the docstrings that the named files hold, and nothing else:

```bash
julia -t 1 --project=docs docs/doctest_files.jl src/01_Base/11_VecScalar.jl src/02_Tools/
julia -t 1 --project=docs docs/doctest_files.jl --against origin/dev
julia -t 1 --project=docs docs/doctest_files.jl --fix src/01_Base/11_VecScalar.jl
```

  A path names a file or a directory. `--against <ref>` names the `.jl` files under `src/` and
  `ext/` that the branch changes. `--fix` writes the output of a failed block into its docstring.
  The script prints `DOCTEST files=… docstrings=… blocks=… status=green|red`, and a green run
  without `--fix` records the stamp `doctest-files`.

- **The scoped run does not replace the full run.** A change in one file can change what a block
  in another file prints, and the scoped run skips the manual pages under `docs/src`. Before the
  push, run the stage that `bash code_health/gate_stamp.sh reach` names: `doctest` is the full
  run, and `doctest-files` is the scoped run. A change to docstring prose outside every
  `jldoctest` block needs neither.

- The shipped default of `set_show_nothing_fields!` is `false`, which hides a field that holds
  `nothing` at the REPL. The doctests set it to `true` in both places, so a rendered docstring shows
  the complete type. A doctest run without the two `true` calls fails on every block that prints a
  `nothing` field.

- The pretty-printer right-aligns field names to the widest field in each block, so **renaming a
  field re-indents every printed block that contains it**, including nested ones. Regenerate the
  expected output from a real run rather than hand-editing it.
