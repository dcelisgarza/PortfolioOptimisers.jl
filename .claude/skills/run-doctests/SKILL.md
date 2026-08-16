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
  DocMeta.setdocmeta!(PortfolioOptimisers, :DocTestSetup, :(using PortfolioOptimisers, StatsBase,
    Statistics, LinearAlgebra, Dates, Distributions, StableRNGs, TimeSeries;
    PortfolioOptimisers.set_compact_show!(false)); recursive=true)
  doctest(PortfolioOptimisers)'
```

- The pretty-printer right-aligns field names to the widest field in each block, so **renaming a
  field re-indents every printed block that contains it**, including nested ones. Regenerate the
  expected output from a real run rather than hand-editing it.
