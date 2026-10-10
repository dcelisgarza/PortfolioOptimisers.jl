---
name: run-doctests
description: Run or update PortfolioOptimisers.jl doctests correctly — the fresh-process rule and the exact CI invocation to mirror. Use before running doctests, or before regenerating expected doctest output.
---

# Running doctests

- Run them in a **fresh process**, never in a working REPL. `show` omits a module prefix for types
  reachable unqualified from the active module, so a session that has done `using StatsBase` makes
  `StatsBase.SimpleCovariance` print bare and produces dozens of phantom failures.
- The CI invocation is in `.github/workflows/Docs.yml` — mirror it exactly:

```bash
julia --project=docs -e '
  using Documenter: DocMeta, doctest
  using PortfolioOptimisers
  PortfolioOptimisers.set_compact_show!(false)
  PortfolioOptimisers.set_show_nothing_fields!(true)
  DocMeta.setdocmeta!(PortfolioOptimisers, :DocTestSetup, :(using PortfolioOptimisers, StatsBase,
    Statistics, LinearAlgebra, Dates, Distributions, StableRNGs, TimeSeries;
    PortfolioOptimisers.set_compact_show!(false);
    PortfolioOptimisers.set_show_nothing_fields!(true)); recursive=true)
  doctest(PortfolioOptimisers)'
```

- When the run passes in a worktree, record its stamp for the pre-push guard:
  `bash code_health/gate_stamp.sh record doctest`. The guard then names the doctests when `dev`
  moves under them before the push.

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
  push of a branch that changes `src/`, run the full doctests above and record `doctest`.

- The shipped default of `set_show_nothing_fields!` is `false`, which hides a field that holds
  `nothing` at the REPL. The doctests set it to `true` in both places, so a rendered docstring shows
  the complete type. A doctest run without the two `true` calls fails on every block that prints a
  `nothing` field.

- The pretty-printer right-aligns field names to the widest field in each block, so **renaming a
  field re-indents every printed block that contains it**, including nested ones. Regenerate the
  expected output from a real run rather than hand-editing it.
