---
status: accepted
---

# The Asset Panel travels as the third positional argument of the returns-matrix prior method

## Context

Every prior estimator implements one method, and
[`AbstractPriorEstimator`](../../src/13_Prior/01_Base_Prior.jl) states it:

```julia
prior(pe::AbstractPriorEstimator, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1,
      kwargs...)
```

The carrier method is written once, in the same file. It reads `rd.X` and `rd.F` off the
[`ReturnsResult`](../../src/03_InputData/03_Preprocessing.jl) and calls the returns-matrix method. A
wrapping prior holds no carrier: it is itself inside that method, so it reaches the estimator it
nests through the matrices alone.

[`CrossSectionalFactorPrior`](../../src/13_Prior/17_CrossSectionalFactorPrior.jl) is the first
estimator that is fitted on something a matrix does not carry. It reads per-asset Panel Fields and
the two universe masks off an [`AssetPanel`](../../src/03_InputData/01_AssetPanel.jl), which travels on
`rd.pnl`. It therefore carried a carrier method of its own that held the fit, and a returns-matrix
method that refused every call.

[Issue #840](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/840) is what that refusal
costs. `prior(EntropyPoolingPrior(; pe = CrossSectionalFactorPrior(…)), rd)` raises even when `rd`
carries the panel, because the wrapper unwraps the carrier and hands its nested estimator a bare
matrix. The same holds for every wrapping prior, and
[ADR 0113](0113-a-factor-attributions-predicted-totals-anchor-on-the-prior-result-and-the-gap-goes-to-the-remainder.md)
is written over exactly that composition: the anchoring rule is stated over a wrapped
cross-sectional fit, and a caller could not build one.

## Decision

### One more positional argument, forwarded like `F`

The interface every prior estimator implements gains a third positional argument:

```julia
prior(pe::AbstractPriorEstimator, X::MatNum, F::Option{<:MatNum} = nothing,
      pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, kwargs...)
```

The carrier method forwards `rd.pnl` there, beside `rd.X` and `rd.F`. The argument defaults to
`nothing`, so a caller that holds matrices alone calls what it called before.

An estimator that reads no panel takes the argument and ignores it. An estimator that declares `F`
as `args...` takes it there and declares nothing.

### A wrapper forwards the panel to the estimator it nests over the assets

A wrapping prior forwards `pnl` unchanged to a nested prior that is fitted on the **assets**, and
does not forward it to one fitted on the **factors**: no panel describes a factor axis. That splits
the wrappers in two.

| Wrapper | The panel reaches |
| --- | --- |
| `HighOrderPriorEstimator`, `BlackLittermanPrior`, `BayesianBlackLittermanPrior`, `HighOrderFactorPriorEstimator`, `EntropyPoolingPrior`, `MeucciEntropyPoolingPrior` | `pe.pe`, the asset prior. |
| `OpinionPoolingPrior` | `pe.pe1`, every opinion in `pe.pes`, and the refit `pe.pe2`. |
| `AugmentedBlackLittermanPrior` | `pe.a_pe` alone. `pe.f_pe` is fitted on the factors. |
| `FactorPrior`, `FactorBlackLittermanPrior` | Nothing. The prior each nests is fitted on the factors. |
| `CrossSectionalFactorPrior` | Nothing. The prior it nests is fitted on the reduced factor returns. |

`ep_prior` takes the argument too, because it is where the entropy pooling estimators fit their
nested prior.

### The Cross-Sectional Factor Prior's fit moves to the returns-matrix method

The refusal method becomes the fit, and the carrier method unwraps onto it. The estimator's own
verbs read a carrier, so the returns-matrix method rebuilds one from `X`, `F`, `pnl`, `iv` and
`ivpa`. A carrier that holds returns holds names for them, and neither a matrix nor a panel states
any, so the rebuild names the columns by their number. No verb of the fit reads a name. The length
of the names is read: it is the asset axis `check_asset_panel` binds the panel to.

Four fields do not survive the round trip, because the carrier method drops them and the rebuild
cannot invent them: `nx`, `ts`, `nb` and `B`. No verb of the fit reads one today. A Descriptor that
comes to read one has to reach it another way, and this ADR is where that will be recorded.

## Alternatives rejected

| Option | How the panel would reach the estimator | Why it was not taken |
| --- | --- | --- |
| **A third positional argument** | The carrier method forwards `rd.pnl`; every wrapper forwards it on. | Taken. It is the shape `F` already has, and a default of `nothing` leaves every existing call working. |
| **Carry the whole `ReturnsResult` through** | A wrapper forwards `rd` rather than the matrices. | It re-writes the interface every prior estimator implements, and the matrices a wrapper hands down are not the carrier's: an opinion pool fits its opinions on the returns its `pe1` produced. |
| **A `prior_input` seam** | A verb per estimator answers the shape its nested estimator takes. | One more family, and one more method per estimator, to carry one field that the carrier already holds. |
| **A keyword argument** | `prior(pe, X, F; pnl = pnl)` | A keyword travels through `kwargs...` to every verb the fit calls, and an estimator that names it would take it twice. The panel is an input of the fit, as `X` and `F` are. |

## Consequences

- Every returns-matrix prior method takes one more positional argument. Every existing caller drops
  it, so no number in the library moves.
- A wrapping prior composes a `CrossSectionalFactorPrior`. `BlackLittermanPrior` and
  `HighOrderPriorEstimator` over one are fitted in
  `test/test_12k_cross_sectional_factor_prior.jl`.
- An estimator outside the library that implements the returns-matrix method with a fixed arity
  raises a `MethodError` on the carrier route until it takes the argument.
- `prior(pe::CrossSectionalFactorPrior, X)` still refuses, because the panel is `nothing`. The
  refusal is an `IsNothingError` against the panel rather than an `ArgumentError` against the
  matrix, and it names both entry points that work.
- A wrapping prior that reweights **observations** still cannot compose this estimator. The fit
  answers on fewer rows than it was given — the window left after the Descriptor warm-up and the
  exposure lag — and entropy pooling builds its prior probabilities over the rows of the matrix it
  was handed, before it fits. That is a second decision about the composition contract, it is not
  taken here, and
  [issue #849](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/849) holds it.

## Amendment (2026-09-06)

The second decision this ADR left untaken is now taken. Issue #849 held it, and
[ADR 0116](0116-a-prior-that-reweights-observations-works-on-the-axis-its-nested-prior-answered.md)
records it: a prior that reweights observations works on the observation axis its nested prior
**answered**, so `ep_prior` fits that prior first and reads its prior probabilities on the rows of
`pr.X`. All three reweighting priors now compose a `CrossSectionalFactorPrior`.

The last consequence above describes the state of the library before that decision. It is correct
history, not a defect.
