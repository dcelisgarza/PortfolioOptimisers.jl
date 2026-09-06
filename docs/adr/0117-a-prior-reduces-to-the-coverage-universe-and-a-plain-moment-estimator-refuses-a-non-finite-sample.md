---
status: accepted
---

# A prior reduces to the Coverage Universe, and a plain moment estimator refuses a non-finite sample

## Context

The contract of
[ADR 0115](0115-every-optimisation-estimator-reduces-once-at-its-entry-and-its-result-carries-the-investable-mask.md)
puts two duties on the moment layer. A Prior Result lives on the full asset universe, and an asset
the prior could not estimate carries `NaN` in `mu` and on the diagonal of `sigma`. The optimiser
derives the Investable Mask from that frame and reduces once at its entry. So a moment estimator
owes the prior a finite number for every asset the prior can estimate, and the `NaN` frame for
every asset it cannot.

Today that frame is an accident of each estimator's arithmetic. Measured on `dev` with a 60×4
sample whose fourth asset lists at observation 31:

- Eleven estimators, `Covariance` and `SimpleExpectedReturns` among them, return the `NaN` row and
  column in silence. Their finite block equals the estimator on the three complete columns.
- `GerberIQCovariance` returns the same frame, and its finite block differs from the estimator on
  the complete columns by `5.5e-6`. The gap leaks into the other assets.
- Twelve estimators and `EmpiricalPrior()` itself throw `ArgumentError: matrix contains Infs or
  NaNs` from the matrix repair: a Base error, not a library one.
- `MutualInfoCovariance` throws `InexactError: Int64(NaN)`.
- `RegimeAdjustedExpWeightedVariance` handles the gap through its `active_mask` keyword.

The reference implementation refuses a `NaN` in every plain estimator and handles a gap in its
exponentially weighted family alone, through an `active_mask` input, a per-asset observation count,
a freeze on a holiday, a reset on an inactive period, a warm-up `NaN` and a bias correction. Its
prior reduces nowhere. A caller of the reference cannot fit a Gerber or a sample covariance on a
gapped panel at all.

The glossary already says that a Prior Estimator fits on the coverage universe and returns a result
on the full asset universe, and
[ADR 0114](0114-the-asset-panel-travels-as-the-third-positional-argument-of-the-returns-matrix-prior-method.md)
already carries the Asset Panel, with its active mask, into every prior fit.

## Decision

### The prior reduces to the Coverage Universe and expands

A prior that fits a moment on the asset axis reduces its returns matrix to the Coverage Universe
before the fit, fits every plain estimator on that clean block, and expands every block of its
result to the full universe with a `NaN` frame outside the Coverage Universe. Three priors carry the
line: `EmpiricalPrior`, `HighOrderPrior` and `FactorPrior`. A wrapping prior forwards the panel to
the prior it nests, as ADR 0114 states, and reduces nothing itself.

A Prior Result has no mask field, so every block it carries lives on the full universe. That
includes a regression result: its rows outside the Coverage Universe are `NaN`.

### The Coverage Universe is finite and active at every row

An asset is in the Coverage Universe of one fit when its return is finite and the panel's active
mask is `true` at every row of the window. That is the reference's per-cell rule,
`valid = isfinite(X) & active_mask`, taken over the whole window. The estimation mask is not read:
the reference reads it for a regime signal only, never for a moment. With no panel, or a panel
with no masks, the rule is finiteness alone.

A complete window yields `nothing`, and that sentinel skips the slice and the expansion, as the
Investable Mask's `nothing` skips the optimiser's two halves. A window with no asset in the
Coverage Universe throws `IsEmptyError`, as `investable_mask` does.

The rule has one cost, stated here so that no docstring hides it. One non-finite return, or one
inactive row, inside the window puts the asset outside the Coverage Universe for that fit. A caller
with a holiday imputes it in `prices_to_returns`, or uses a mask-aware estimator.

### A plain moment estimator refuses a non-finite sample

Every plain moment verb asserts that its sample is finite, through `assert_all_finite`, and throws
`IsNonFiniteError`. A caller that bypasses the prior gets a named refusal whose message says to fit
through a prior or to use a mask-aware estimator. No plain estimator gains a masked or a pairwise
path, so no covariance is built from pairwise-complete observations and no repair needs a mask.

### The Asset Panel is the third positional argument of the moment verbs

`cov`, `cor`, `mean`, `var`, `std`, `coskewness` and `cokurtosis` gain a third positional
argument, `pnl::Option{<:AssetPanel}`, with `nothing` as the default, as the prior method has.
The root method of each verb is the reduce-and-expand:

```julia
function Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
    cmsk, Xc = coverage_reduction(X, pnl; dims = dims)
    return expand_moment(Statistics.cov(ce, Xc; dims = dims, kwargs...), cmsk)
end
```

A mask-aware estimator overrides the panel method and reads `pnl.amsk` and `pnl.emsk`, so it takes
the whole window and emits its own frame, because it alone knows its warm-up and its resets.
`RegimeAdjustedExpWeightedVariance` forwards them to the two keywords it already has. A plain
estimator needs no edit and no declaration.

### A composite forwards the panel, then repairs the finite block

A composite estimator, `PortfolioOptimisersCovariance` and the seven other files that call the
matrix repair, forwards the panel to its inner estimator and can get a frame back. Its panel method
derives the block from `isfinite.(diag(sigma))`, runs the unchanged `matrix_processing!` on that
block, and writes the block back into the frame. The bare `matrix_processing!` and `posdef!` keep
their whole-matrix refusal, so a `NaN` that reaches a plain path is still refused there.

There is no peel. An off-diagonal `NaN` inside the block is refused with `IsNonFiniteError`. No
estimator of the library, and none of the reference's exponentially weighted family, can make one:
a delisted asset ends the window inactive, so its diagonal is `NaN` and it is outside the block
already. The reference's peel is defensive for a pairwise estimator, which the library does not
have, and a peel here would hide a defect.

### The exponentially weighted family is ported on the same seam

The reference's three exponentially weighted estimators, and the fit that
[#692](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/692) found missing on
`RegimeAdjustedExpWeightedCovariance`, are the mask-aware answer for a young asset. They enter
through the panel override, and the identities the census of
[#670](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/670) pinned are their oracle.
The bias correction is the first power of `1 - λⁿ` for a mean and a variance, and the square root
for a covariance.

## Considered options

| Question | Refused | Why |
| --- | --- | --- |
| Where the gap logic lives | The verb itself reduces and expands, for every caller. | 55 per-type methods renamed to an inner verb, and the three direct callers outside the priors already sit below the optimiser's reduction. |
| Where the gap logic lives | An available-case path per estimator: a masked mean, a pairwise-complete covariance, a masked repair. | About 25 masked paths, a covariance that is not positive semidefinite, and no oracle, because the reference has none. |
| Where the gap logic lives | A write gate over the silent frame. | It cannot help the thirteen estimators that throw before they write, and it cannot correct the Gerber IQ leak. |
| The Coverage Universe | Finite at every row, active at the last row. | A stale finite return during an inactive spell enters the moments as a real return. |
| The Coverage Universe | Finiteness alone. | A delisted asset whose price series continues stays investable. |
| The seam | Two mask keywords, a `Bool` trait per type and a branch in the prior. | A keyword cannot dispatch, so a plain estimator that receives it passes it to `StatsBase` and throws, and a new mask-aware type that forgets the trait fails in silence. |
| The seam | The two masks as positional arguments. | Two arguments where the glossary names one carrier, and every wrapper forwards two things. |
| The block repair | `matrix_processing!` derives the block for every caller. | Every plain path then accepts a frame, which relaxes a finiteness check in advance. |
| The block repair | The reference's greedy peel. | No estimator can make it fire, and a peel hides a defect. |

## Consequences

- One build ticket writes the reduction, the seam, the block repair and the refusals. A second,
  blocked by it, ports the exponentially weighted family and writes the fit of #692.
- The fog item of map [#667](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/667) on
  the repair of a pairwise estimate closes: no estimator computes over a gap.
- Under the exponentially weighted family a young asset is investable while its early scenario
  rows are `NaN`. The reference zero-fills those rows and warns. What the library does with them
  is the measures decision of the map.
- `prices_to_returns` drops every row that still holds a missing entry, so a gapped panel reaches
  the moments only through a hand-built `ReturnsResult` today.
- `cov`, `cor` and `std` overflow the stack on every variance estimator, `SimpleVariance` included.
  That is a defect of the fallback chain, filed separately, and not of this decision.
