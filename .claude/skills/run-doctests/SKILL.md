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

- The shipped default of `set_show_nothing_fields!` is `false`, which hides a field that holds
  `nothing` at the REPL. The doctests set it to `true` in both places, so a rendered docstring shows
  the complete type. A doctest run without the two `true` calls fails on every block that prints a
  `nothing` field.

- The pretty-printer right-aligns field names to the widest field in each block, so **renaming a
  field re-indents every printed block that contains it**, including nested ones. Regenerate the
  expected output from a real run rather than hand-editing it.
