---
agent: ask
description: Run pre-commit checks and tests before committing changes to PortfolioOptimisers.jl.
---

Run the full pre-commit and test suite to verify that changes are correct and ready to commit.

## Step 1 — Run pre-commit checks

From the repository root:

```bash
pre-commit run -a
```

All hooks must pass (exit code 0) before proceeding. If any hook fails:

- **JuliaFormatter**: Auto-fixes are applied. Run `pre-commit run -a` again to confirm they pass.
- **Spelling / grammar**: Fix flagged issues manually, then re-run.
- **Other hooks**: Read the output carefully and fix the root cause.

Do not proceed to tests until `pre-commit run -a` exits with no failures.

## Step 2 — Run the test suite

Start a Julia REPL in the repository root and run:

```julia-repl
julia> ] activate .
julia> ] test
```

All tests must pass. If any test fails:

1. Read the full failure message to identify the failing test and file.
2. Determine whether the failure is caused by your changes or is a pre-existing failure.
3. Fix failures caused by your changes.
4. Report pre-existing failures to the user and ask whether to proceed.

## Step 3 — Run doctests

Doctests are **not** run by `] test` — they are a separate step. Run them from the repository root:

```bash
julia --color=yes --project=docs -e '
  using Pkg
  Pkg.develop(PackageSpec(path=pwd()))
  Pkg.instantiate()'
```

Then:

```bash
julia --color=yes --project=docs -e '
  using Documenter: DocMeta, doctest
  using PortfolioOptimisers
  DocMeta.setdocmeta!(PortfolioOptimisers, :DocTestSetup, :(using PortfolioOptimisers, StatsBase, Statistics, LinearAlgebra, Dates, Distributions, StableRNGs, TimeSeries); recursive=true)
  doctest(PortfolioOptimisers)'
```

All doctests must pass. If a doctest fails:

- Check that the `jldoctest` block matches the actual output of `@define_pretty_show`.
- Confirm that field names, types, and default values match what is defined in the struct.
- Update the doctest to match the current output exactly, then re-run.

## Step 4 — Confirm readiness

Once all three steps pass with no failures, report the result to the user:

```text
pre-commit run -a  ✓  All N hooks passed.
] test             ✓  All N tests passed.
doctest            ✓  All doctests passed.
```

The changes are ready to commit.
