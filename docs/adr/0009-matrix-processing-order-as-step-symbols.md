---
status: accepted
---

# Represent matrix-processing order as a sequence of step symbols

## Context

`DenoiseDetoneAlgMatrixProcessing` ([src/07_MatrixProcessing.jl](../../src/07_MatrixProcessing.jl))
applied four steps to a covariance/correlation matrix — `posdef!` (always first), then
denoising, detoning, and an optional custom algorithm — in a configurable order. The order
was encoded in the type system: an empty marker struct per permutation of the three
reorderable steps (`DenoiseDetoneAlg`, `DenoiseAlgDetone`, `DetoneDenoiseAlg`,
`DetoneAlgDenoise`, `AlgDenoiseDetone`, `AlgDetoneDenoise`, all `<: AbstractMatrixProcessingOrder`),
selected by the estimator's fifth type parameter.

This was shallow: the six fieldless structs carried nothing but a docstring, and the six
`matrix_processing!` method bodies differed only in the order of three calls — `posdef!` then a
permutation of `denoise!` / `detone!` / `matrix_processing_algorithm!`. The interface surface
(6 types + 6 dispatch methods) nearly equalled the implementation. Adding a seventh ordering,
or a fourth reorderable step, meant a new struct *and* a new hand-written body. Nothing outside
`matrix_processing!` dispatches on the six order types — the only references are docstring
printed-trees and `order = …` keyword arguments in
[test/test_08_moments.jl](../../test/test_08_moments.jl).

The first design sketched here was more aggressive: collapse `pdm`/`dn`/`dt`/`alg`/`order` into
a *single* field holding an ordered tuple of the estimators themselves. Building it surfaced a
blocker the static read missed: **`mp.pdm` is read externally in ~24 sites** — Black-Litterman
and factor priors and (mostly) [NormalUncertaintySets.jl](../../src/14_UncertaintySets/03_NormalUncertaintySets.jl)
do `posdef!(pe.mp.pdm, sigma)` / `posdef!(ue.pe.ce.mp.pdm, …)` to reuse the matrix-processing
estimator's posdef projector on *other* matrices (posterior covariances, uncertainty-set
sigmas). Collapsing the four fields would have broken every one of those sites, forcing either a
`getproperty` override or an accessor + 24 rewrites.

## Decision

**Keep the `pdm`/`dn`/`dt`/`alg` fields exactly as they were** — so the ~24 external `mp.pdm`
reads keep working untouched — and change only the `order` field: from one of six marker structs
to a **tuple (or vector) of step symbols**, dispatched via `Val`. Delete the six order structs
and `AbstractMatrixProcessingOrder`. Rename `DenoiseDetoneAlgMatrixProcessing` → `MatrixProcessing`.

```julia
@concrete struct MatrixProcessing <: AbstractMatrixProcessingEstimator
    pdm
    dn
    dt
    alg
    order   # tuple (recommended) or vector of step symbols, applied left to right
end
MatrixProcessing(; pdm = Posdef(), dn = nothing, dt = nothing, alg = nothing,
                 order = (:pdm, :dn, :dt, :alg)) = ...

function matrix_processing!(mp::MatrixProcessing, sigma::MatNum, X::MatNum, args...; kwargs...)
    for step in mp.order
        matrix_processing_step!(Val(step), mp, sigma, X; kwargs...)
    end
    return sigma
end
```

A new `matrix_processing_step!` dispatches on the step symbol and pulls the right estimator out
of `mp`. The one step that needs the aspect ratio (`:dn`) derives it from `X` itself, so the
common signature carries no `T`/`N`:

```julia
matrix_processing_step!(::Val{:pdm},  mp, sigma, X; kw...) = posdef!(mp.pdm, sigma)
matrix_processing_step!(::Val{:dn}, mp, sigma, X; kw...) = (T, N = size(X); denoise!(mp.dn, sigma, T / N))
matrix_processing_step!(::Val{:dt},  mp, sigma, X; kw...) = detone!(mp.dt, sigma)
matrix_processing_step!(::Val{:alg},     mp, sigma, X; kw...) = matrix_processing_algorithm!(mp.alg, sigma, X; kw...)
matrix_processing_step!(::Val{step},     mp, sigma, X; kw...) where {step} =        # unknown step
    (@warn "Unknown matrix processing step :$(step); skipping." maxlog = 1; sigma)
```

Specific points decided:

1. **Fields stay; only `order` changes.** This is what keeps the external `mp.pdm` (and `mp.dn`
   etc.) reads working with zero churn — the decisive reason for choosing symbols over the
   estimator-tuple collapse.

2. **`posdef` is an ordinary step**, named `:pdm` in the order, rather than a fixed first
   call. The default lists it first to preserve behaviour.

3. **Default `order = (:pdm, :dn, :dt, :alg)`.** The old default had
   `dn = dt = alg = nothing`, so the `:dn`/`:dt`/`:alg` steps are no-ops there (the
   underlying primitives already no-op on `nothing`), and only `:pdm` does work — exactly the
   old behaviour.

4. **Tuples recommended, vectors accepted.** `Val(step)` is a runtime dispatch per step either
   way; matrix processing runs once per covariance estimate, not in a hot loop, so the cost is
   noise.

5. **Unknown symbol → warn-and-skip** (`maxlog = 1`, once per session, so CV/meta-optimiser loops
   don't flood the log). A `nothing`-valued estimator field is already a silent no-op via the
   existing `posdef!`/`denoise!`/`detone!`/`matrix_processing_algorithm!` `::Nothing` methods.

6. **No deprecation aliases.** Breaking change on a dev branch; half-removed aliases would only
   muddy the export list.

## Considered options

- **Collapse `pdm`/`dn`/`dt`/`alg`/`order` into one field holding an ordered tuple of the
  estimators themselves** (the original sketch — the estimators dispatch directly, no symbols).
  Rejected once `mp.pdm`'s ~24 external readers surfaced: removing the named fields would force a
  `getproperty` override (magic, type-unstable) or an accessor plus 24 call-site rewrites, for no
  gain over keeping the fields.
- **Step-marker singleton structs** (`PosdefStep`/`DenoiseStep`/…) or **the step functions**
  (`posdef!`/`denoise!/…`) in the order tuple. Rejected: with the fields kept, the order only
  needs to *name* which field/operation to run next; a symbol does that with no new vocabulary,
  and `Denoise`/`Detone` are already taken by the estimator types.
- **Keep the six types, collapse only the six bodies** (a `steps(order)` trait + one walker).
  Rejected: keeps six exported marker types and an abstract supertype purely to name orderings a
  symbol sequence expresses directly.

## Consequences

- External `mp.pdm` / `mp.dn` / `mp.dt` / `mp.alg` reads (~24 sites in priors and uncertainty
  sets) are **untouched** — the fields are unchanged.
- Every docstring/doctest that prints a `MatrixProcessing` (or a `PortfolioOptimisersCovariance`
  / Prior tree containing one) changes in two places: the type name (`MatrixProcessing`) and the
  `order` line, now a symbol tuple instead of `DenoiseDetoneAlg()`; the estimator's fifth type
  parameter prints as the order's type (`Tuple{Symbol, …}`) instead of a marker type. The doctest
  suite is the regression net for the ~39 affected files.
- Constructing a non-default order changes from a marker struct
  (`MatrixProcessing(; order = DetoneDenoiseAlg(), dt = Detone())`) to a symbol tuple
  (`MatrixProcessing(; order = (:pdm, :dt, :dn, :alg), dt = Detone())`).
  [test/test_08_moments.jl](../../test/test_08_moments.jl) is rewritten accordingly.
- Adding a processing step is now a new `:symbol` plus one `matrix_processing_step!(::Val{:symbol}, …)`
  method — never a new order struct + hand-written `matrix_processing!` body.
- A user can now express orders the six types never allowed (omit `:pdm`, repeat a step). The
  warn-fallback covers unrecognised symbols.
