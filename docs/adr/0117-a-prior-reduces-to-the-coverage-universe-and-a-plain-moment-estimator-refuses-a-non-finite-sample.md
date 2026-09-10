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

### A plain moment estimator refuses a non-finite sample, unless it carries a Coverage Policy

Every plain moment verb asserts that its sample is finite, through `assert_all_finite`, and throws
`IsNonFiniteError`. A caller that bypasses the prior gets a named refusal whose message says to fit
through a prior or to use a mask-aware estimator.

### A moment estimator may opt into available-case estimation

The all-or-nothing rule above is **monotone non-increasing** over an expanding window. `coverage_mask`
ANDs over every row, and an expanding walk-forward pins `train_start = 1`, so one non-finite or
inactive row anywhere in `[1..t]` puts an asset out and growing the window never puts it back. An
asset that lists after the window's first row is therefore outside the Coverage Universe of every
fold, forever, and a late listing discards the survivors' whole history with it. That is a defect to
route around rather than behaviour to reproduce, because the machinery to be robust to gaps — the
finite-aware renormalising combination, the forced-liquidation fees and the `NaN` moment frame — is
already in the library.

A moment estimator therefore gains one field, `cvg::Option{<:CoveragePolicy}`, defaulting to
`nothing`. The `nothing` arm is read by dispatch and is exactly the reduce-and-expand path above, so
an estimator that carries no policy costs nothing and behaves as it did. With a policy set the
estimator is a **mask-aware** estimator in the sense this ADR already defines: it takes the whole
window, reads `pnl.amsk` itself, and emits its own frame.

Available-case estimation means one rule at every cell. A cell is fitted on the observations at
which every asset of that cell is finite and active, and it carries its own denominator, of the
shape of its accumulator. The frame is then the per-asset half: an asset reaches the answer only
where `admits(alg, share, active, stale, min_coverage)` says so, with `share` that asset's own
observation count over the number of observations fitted. What happens to a delisted asset is a
dispatchable family, `AbstractCoverageAlgorithm`, mirroring `AbstractMomentAlgorithm`:
`DecayCoverage` keeps the history and drops the asset the moment it goes inactive, `ResetCoverage`
zeroes it so a relisting starts cold, and `ExpireCoverage(; after)` holds it in the frame for a
stated staleness. A caller whose rule is none of the three subtypes the root and implements
`fold_inactive!` and `admits`.

The centre of a cell is that cell's own mean, which is what makes the fold exact: a per-pair Welford
recursion carries a per-pair count and a per-pair centre, so an available-case covariance folded
observation by observation is the available-case covariance of the same rows fitted as a block, to
the last bit. The batch arm **is** the fold, so there is one recursion and no second implementation
to drift. A semi-moment has no such recursion, because the clip is taken about a centre the whole
window fixes, so its available-case arm is a two-pass over the block and its centre is each asset's
own available-case mean.

Two costs are stated here so that no docstring hides them. A pair whose two assets are each admitted
but which share no observation is `NaN` on its own, because a covariance of no observations is not a
number; and an available-case correlation may exceed one in absolute value, because its cells are
centred on different observation sets. The correlation is the repair's to handle on the surviving
block. The empty pair is **not**: it is refused by name, under the block rule below, because there is
no number to repair towards and inventing one is the fabrication the plain path exists to refuse.

### The Asset Panel is the third positional argument of the moment verbs

`cov`, `cor`, `mean`, `var`, `std`, `coskewness`, `cokurtosis` and `variance_series` gain a
third positional argument, `pnl::Option{<:AssetPanel}`, as the prior method has.
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

`variance_series` takes the same third positional argument. It refits its estimator once per
observation, so every window meets the refusal on its own, and the root reduces each window to
**its own** Coverage Universe rather than to the sample's. Its two-argument method is unchanged, so
a caller that holds no panel keeps the behaviour it had, and a mask-aware estimator overrides the
panel method as it overrides every other root.

### A composite forwards the panel, then repairs the finite block

A composite estimator, `PortfolioOptimisersCovariance` and the seven other files that call the
matrix repair, forwards the panel to its inner estimator and can get a frame back. Its panel method
derives the block from `isfinite.(diag(sigma))`, runs the unchanged `matrix_processing!` on that
block, and writes the block back into the frame. The bare `matrix_processing!` and `posdef!` keep
their whole-matrix refusal, so a `NaN` that reaches a plain path is still refused there.

There is no peel. An off-diagonal `NaN` inside the block is refused with `IsNonFiniteError`, and the
asset is never dropped until the block is complete. A peel would hide a defect, and it would also be
a second, silent universe rule sitting under the one `admits` already states.

An estimator carrying a Coverage Policy **can** make such a cell, and the refusal is written for it.
Two assets whose observed rows do not intersect — each quoting on rows the other misses, both active
at the last row — are each admitted on their own share and share no observation, so their pair is
`NaN` while both diagonals are finite. The refusal therefore cannot live only in the framed branch:
a matrix whose diagonal is finite everywhere still reaches it, and the caller is told which pair and
why. The reference's own peel is defensive for a pairwise estimator, which the library does not
have.

### The third and fourth order take the policy, and the block rule is one law

A `Coskewness` and a `Cokurtosis` carry `cvg` on the same terms as the second order. Their
available-case arm is the per-cell numerator and denominator of the accumulator's own shape:
`Y` centred on each asset's available-case mean with the masked cells zeroed, `Mi` the mask as
integers, and the pairwise expansion of each; `num = Yᵀz` against `den = Miᵀzc` at the third order,
and `zᵀz` against `zcᵀzc` at the fourth. `coverage_divide` and `coverage_admission` finish both.
The `SemiMoment` arms take the policy too, as the same two-pass the semi-covariance runs: the clip
is taken about each asset's own available-case mean, before the expansion, and an external `mean` is
refused.

What the spectral and matrix-processing steps see is the **block**, and the rule is the one the
second order already keeps. A fit emits a frame; a repair derives its block from that frame's own
diagonal, refuses a non-finite cell inside the block by name, repairs the block, and leaves the
frame around it alone. The diagonal is read at the resolution of the quantity: `sk`'s is
`E[yᵢ³]` at column `(i-1)N + i`, finite where asset `i` has observations of its own, and `kt`'s is
`E[yᵢ²yⱼ²]`, finite where the pair shares one — so `matrix_processing_block!` is already right for a
cokurtosis, and `negative_spectral_coskewness` gains the same shape. That verb is the one place the
rule is written, and the three call sites that reduce a coskewness — the fit, the carrier's
`port_opt_view`, and the high-order factor prior — all reach it.

The refusal matters because the failure it replaces is neither silent nor legible. An
eigendecomposition of a tensor holding one `NaN` does not return `NaN`: Julia's LAPACK wrappers
check first, so `eigen` and `nearest_cor!` both throw `ArgumentError: matrix contains Infs or NaNs`,
which names neither coverage nor the triple that caused it.

**The Investable Mask reads every order the carrier holds.** `investable_mask` derives from `mu` and
the diagonal of `sigma`, and under a policy that no longer implies the higher-order tensors are
finite where it admits. A `HighOrderPrior` therefore ANDs the per-asset diagonals of `sk` and `kt`
into the mask, so an asset the higher orders could not estimate leaves the problem rather than
reaching a spectral step that throws. Where that narrowing drops an asset the low-order moments
held — a policy set on the mean and the covariance and not on `ske` and `kte` — the prior warns
once, naming the assets and the field to set. The configuration is legal and well defined; it is
only silence that is refused, as [ADR 0118](0118-a-fold-zeroes-a-held-gap-once-and-a-value-level-verb-reduces-to-the-investable-mask.md)
refuses it for a scenario fill.

**Neither order folds exactly.** An exact per-cell Welford at the third order needs the pairwise
second co-moments over each triple's own observation set, and at the fourth the second and third
over each quadruple's — three further arrays of `assets² × assets²`. So the online form of an
available-case coskewness or cokurtosis is a buffer refit through `Online`, which is bit-exact with
its own batch arm by construction, and that is the third kind of member the sample-buffer decision
named.

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
| Where the gap logic lives | An available-case path in every estimator, always on. | Every existing fit would change behaviour, and most callers have no gap to pay for. The opt-in `cvg` field gives the path to the caller who asks for it and nothing to the caller who does not. |
| Available-case estimation | Refusing it outright. | The all-or-nothing rule is monotone non-increasing over an expanding window, so a plain estimator could never hold an asset that lists after the first row. Its three original objections are answered: one masked path per family rather than about 25, `posdef!` on the surviving block, and the batch-parity identity of the fold as the oracle. |
| The available-case centre | The whole window's per-asset mean for every cell. | The recursion is then not exact: a cell's count and its centre's count differ, so the Welford identity no longer telescopes and a fold would not equal its own batch fit. |
| The delisting rule | One hard-coded rule. | A delisting has no single right answer, and the maintainer asked for an algorithm per rule, overridable by subtype. |
| The coverage floor | A count of observations. | A count is not comparable across windows of different lengths, and the floor is asked as "how much of this window did the asset quote for". |
| Where the gap logic lives | A write gate over the silent frame. | It cannot help the thirteen estimators that throw before they write, and it cannot correct the Gerber IQ leak. |
| The Coverage Universe | Finite at every row, active at the last row. | A stale finite return during an inactive spell enters the moments as a real return. |
| The Coverage Universe | Finiteness alone. | A delisted asset whose price series continues stays investable. |
| The seam | Two mask keywords, a `Bool` trait per type and a branch in the prior. | A keyword cannot dispatch, so a plain estimator that receives it passes it to `StatsBase` and throws, and a new mask-aware type that forgets the trait fails in silence. |
| The seam | The two masks as positional arguments. | Two arguments where the glossary names one carrier, and every wrapper forwards two things. |
| The third and fourth order | No policy above the second order, the mask intersecting the two universes. | It is the smallest change and it does fix the divergence, but a caller who sets a policy and asks for a high-order prior then gets the Coverage Universe back where the panel is gappiest, and buys nothing. |
| The third and fourth order | `mp` leaves `Coskewness` and `Cokurtosis`, so both become raw fits and every repair lives in a composite. | It is the cleaner shape on architecture, and it re-spells released API well past this map's destination. Held for its own ticket rather than refused. |
| The third and fourth order | Filling an empty cell before the spectral step, with zero or the complete-case value. | It fabricates a number, which is what the plain path refuses to do, and the library has just removed its last silent fill. |
| The mixed configuration | Refusing a policy on the low order without one on `ske` and `kte`. | The configuration is well defined once the mask reads every order, and refusing it would make an opt-in field mandatory in a place the caller did not ask for it. |
| The block repair | `matrix_processing!` derives the block for every caller. | Every plain path then accepts a frame, which relaxes a finiteness check in advance. |
| The block repair | The reference's greedy peel. | No estimator can make it fire, and a peel hides a defect. |

## Consequences

- One build ticket writes the reduction, the seam, the block repair and the refusals. A second,
  blocked by it, ports the exponentially weighted family and writes the fit of #692.
- The fog item of map [#667](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/667) on
  the repair of a pairwise estimate reopens under the `cvg` field: an estimator that carries a
  policy does compute over a gap, and its answer is repaired on the surviving block.
- The `cvg` field lands on `SimpleExpectedReturns`, `SimpleVariance` and `Covariance` first, and on
  `Coskewness` and `Cokurtosis` in the build that
  [#983](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/983) decided. Between the two,
  the orders estimate over different universes and nothing reconciles them: a high-order prior whose
  low-order prior carries a policy has a finite `mu` and `sigma`, a mask admitting every asset, and
  an `sk` and `kt` of `NaN`, and `port_opt_view` throws LAPACK's message. The build closes that gap.
- The block refusal is owed a second reading. `matrix_processing_block!` short-circuits to the plain
  repair when every diagonal is finite, so an empty pair among fully admitted assets bypasses the
  named refusal it was written for and reaches LAPACK instead. The rule is on the matrix, not on the
  frame around it.
- Under the exponentially weighted family a young asset is investable while its early scenario
  rows are `NaN`. The reference zero-fills those rows and warns. What the library does with them
  is the measures decision of the map.
- `prices_to_returns` drops every row that still holds a missing entry, so a gapped panel reaches
  the moments only through a hand-built `ReturnsResult` today.
- `cov`, `cor` and `std` overflow the stack on every variance estimator, `SimpleVariance` included.
  That is a defect of the fallback chain, filed separately, and not of this decision.
