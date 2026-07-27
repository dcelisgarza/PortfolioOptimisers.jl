---
status: accepted
---

# `Impute` is a weak dependency behind `apply_impute_method`

## Context

`Impute` was a hard dependency of PortfolioOptimisers for a surface of exactly two lines in
[`src/03_Preprocessing.jl`](../../src/03_Preprocessing.jl): the `impute_method::Option{<:Impute.Imputor} = nothing`
keyword annotation on `prices_to_returns`, and the single `Impute.impute(X, impute_method)` call
guarded by `!isnothing`. No struct has a field typed on `Impute`, and the keyword defaults to
`nothing`, so on the default path nothing of it runs.

The cost was not the four extra packages `Impute` 0.6 hard-depends on (`DataDeps`, `BSON`, `CSV`,
`NamedDims`) but one transitive compat bound: `DataDeps` caps `HTTP` at `"1"`. A hard dependency
propagates that cap into **every** environment PortfolioOptimisers is loaded into, so no downstream
package could hold PortfolioOptimisers alongside anything that has followed `HTTP` into 2.x — and
the resolver's failure message names `DataDeps`, a package PortfolioOptimisers does not use and no
user has heard of.

Issue #158 also attributed this repo's own `LiveServer` 1.5 pin to the same cap. Measurement showed
that half over-determined: with `Impute` removed from `docs/Project.toml` entirely, `docs/` still
resolves `HTTP` 1.11.0 and `LiveServer` 1.5.0, because `GR` (via `StatsPlots`) and `YFinance` cap
`HTTP` at `1` independently of `DataDeps`. The docs pin is therefore *not* what this ADR fixes; the
downstream propagation is.

Two alternatives were weighed ([#158](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/158)).
Dropping the keyword outright is breaking, removes a capability, and needs a deprecation cycle;
`Imputer` covers the fitted, pipeline-aware case but not a one-shot `LOCF` on a price table. Pushing
the bound upstream (the `DataDeps` cap is the real defect, and `Impute` arguably should not hard-depend
on `DataDeps` for dataset plumbing) is right but not in our control and does not unblock the docs
environment now; both are filed upstream in parallel.

## Decision

**`Impute` is a weak dependency, reached through a one-method seam that the main module owns and
`PortfolioOptimisersImputeExt` extends.** `prices_to_returns` no longer annotates `impute_method`
and no longer calls `Impute` directly; it calls `apply_impute_method(X, impute_method)`, which has
an identity method on `Nothing` in the main module and a method on `Impute.Imputor` in the
extension. Passing an imputor already implies having `Impute` loaded, so nothing changes for a
caller who uses the keyword; everyone else stops paying the transitive `HTTP` cap.

The dispatch check the annotation used to perform moves into a fallback method that throws an
`ArgumentError`. That message carries the two things a caller cannot infer from a `MethodError`:

1. That `Impute` must be loaded, distinguished at throw time from a genuine type error by
   `Base.get_extension(@__MODULE__, :PortfolioOptimisersImputeExt)` — "not loaded" and "loaded, but
   this is not an `Impute.Imputor`" are different mistakes with different fixes.
2. That **`Imputer` is not `Impute`.** The name collision is the likeliest source of confusion here:
   `Imputer`/`ImputerResult` are PortfolioOptimisers' own Preprocessing Estimators, filling gaps
   with per-asset statistics fitted on a training window and used as a `Pipeline` step. They share
   nothing with `Impute.jl` but four letters, and the collision is what made a one-keyword optional
   dependency look load-bearing.

## Consequences

- A wrong `impute_method` type is caught by the stub's `ArgumentError` rather than by a
  `MethodError` at the keyword. Deliberate: the annotation could name a type the caller may not have
  loaded, and the thrown message is more informative than the `MethodError` it replaces.
- `docs/Project.toml` keeps its `Impute` dependency — the preprocessing example exercises the
  keyword and so must load the extension. The docs environment therefore still carries `DataDeps`
  and stays on `HTTP` 1 / `LiveServer` 1.5; as measured above it would anyway, because `GR` and
  `YFinance` cap `HTTP` at `1` too. Lifting the docs pin is a separate problem this ADR does not
  claim to solve.
- A downstream environment holding PortfolioOptimisers and `LiveServer` now resolves `LiveServer`
  1.6 and `HTTP` 2, which was previously unsatisfiable. That is the consequence that matters, and it
  is the one to re-check if this ever regresses.
- The `"0.6"` compat entry stays; a weak dependency is still version-bounded.
- Future optional third-party integrations are expected to follow the same shape — a named seam in
  the main module with an informative fallback, not a type annotation on a keyword — since an
  annotation forces the dependency to be hard.
