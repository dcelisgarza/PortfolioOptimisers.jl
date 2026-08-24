---
status: accepted
---

# A variance estimator's observation weights weight its centre, not its deviations alone

## Context

[#490](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/490), a child of
[#417](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/417), opened on a raise from the
sweep of [#453](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/453). One
[`SimpleVariance`](../../src/08_Moments/04_SimpleVariance.jl) answered the same data with two
numbers:

```julia
w  = [0.2, 0.3, 0.5]
aw = StatsBase.AnalyticWeights(w)
v  = [1.0, 3.0, 4.0]
Xc = reshape(v, 3, 1)
ve = SimpleVariance(; w = aw)

var(ve, v)             # 2.080645161290323   -- centred on the weighted mean
var(ve, Xc; dims = 1)  # 2.383512544802868   -- centred on the unweighted mean
```

The two answers differed by 14.6 %.

### The cause

The two methods of `simple_variance_kernel` resolved the centre in two ways. The matrix method
resolved it before it called `f`:

```julia
mu = isnothing(mean) ? Statistics.mean(me, X; dims = dims, kwargs...) : mean
```

`me` is `ve.me`, and its own `w` field is separate from `ve.w`. The default
`SimpleExpectedReturns()` carries no weights, so `mu` was the **unweighted** column mean and the
weights reached the squared deviations alone.

The vector method passed `mean` through to `Statistics`, which centres a weighted vector on its
**weighted** mean. There is no expected-returns method for a vector, because every
`Statistics.mean(me, X)` method of the library takes a `MatNum`. The vector path therefore had no
`me` to read, and the split was a property of the estimator rather than of the data.

### The two readings

The ticket named both, and both are defensible.

 1. **`w` weights the whole estimate.** A caller who sets `w` on the variance estimator means the
    weighted variance, so the centre is weighted too.
 2. **`me` and `w` are separate knobs on purpose.** `me` is a full estimator, not a mean vector, so
    a caller may want a shrunk or an equilibrium centre with exponentially weighted deviations.

Reading 2 explains why `me` is an estimator field rather than a `mean` keyword, so it is not a straw
man. Three measurements decide between them.

**The library's own weight channel already reads reading 1.** [`factory`](../../src/02_Tools.jl)
replaces a `@wprop`-tagged field at **every level of the tree at once**, so `factory(ve, w)` writes
`w` into `ve.w` and into `ve.me.w` together. Every weighted call site inside the library reaches the
estimator through `factory`, so every one of them already held a weighted centre. Only a hand-built
estimator that never met `factory` saw the split.

**The number that the split produced is not a named estimator.** An unweighted centre with weighted
deviations is neither the weighted variance nor the unweighted one. `Statistics.var(X, w; dims = 1)`
centres on the weighted mean, so the split also cost parity with the function the method wraps.

**Reading 2 survives the fix at the call site.** The `mean` keyword takes any centre, so a caller
who wants a differently weighted centre still writes it:

```julia
var(ve, Xc; dims = 1, mean = Statistics.mean(SimpleExpectedReturns(), Xc; dims = 1))
```

What reading 2 loses is the ability to **store** that combination in one estimator, not the ability
to compute it.

## Decision

**A variance estimator's observation weights weight its centre.** `simple_variance_kernel` resolves
the centre through `factory(me, ve.w)` when the caller supplies no `mean`:

```julia
mu = if !isnothing(mean)
    mean
elseif isnothing(ve.w)
    Statistics.mean(me, X; dims = dims, kwargs...)
else
    Statistics.mean(factory(me, ve.w), X; dims = dims, kwargs...)
end
```

The `isnothing(ve.w)` branch is a performance guard, not a second contract. `ve.w` is a field, so
its type decides the branch and the compiler folds the test away. The guard keeps a windowed loop
from rebuilding the estimator tree of `me` once per window when no weights exist to write.

**`ve.w` wins over `me`'s own weights.** The rule matches `factory`, which has always overwritten
them, so an estimator behaves the same before and after it passes through the weight channel.

**The vector method does not change.** It already centres on the weighted mean, and it has no `me`
to read.

**The `mean` keyword stays the escape hatch.** A caller who wants a centre that the estimator's own
weights do not describe passes it.

## Consequences

- **No internal call site moves.** The standard-deviation risk measures, the moment priors and the
  backtests reach `SimpleVariance` through `factory`, which already weighted `ve.me.w`. The
  measured numbers are the same before and after.
- **A hand-built weighted estimator moves.** `SimpleVariance(; w = aw)` over a matrix answers
  `2.080645161290323` where it answered `2.383512544802868`. That is the defect this ADR fixes, and
  it is the whole size of the behaviour change.
- **A stored estimator can no longer hold a centre that is weighted differently from its
  deviations.** `SimpleVariance(; me = SimpleExpectedReturns(; w = w1), w = w2)` now centres on
  `w2`. The `mean` keyword expresses the old combination at the call site, and `factory` destroyed
  the stored form already.
- **The two paths of one estimator agree.** A vector and its one-column matrix answer the same
  number, which is what [#490](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/490)
  asked for.
- **`me` keeps its reason to be an estimator.** A shrunk or an equilibrium centre still differs
  from a simple mean. It now reads the same observation weights as the deviations it centres.
- **The sibling estimators are not swept by this ADR.**
  [`Covariance`](../../src/08_Moments/03_Covariance.jl),
  [`Coskewness`](../../src/08_Moments/19_Coskewness.jl) and
  [`Cokurtosis`](../../src/08_Moments/20_Cokurtosis.jl) each resolve `mu` from their own `me` in
  the same shape, so a hand-built weighted one carries the same split. The two co-moment estimators
  hold `@wprop w` themselves, so the fix above ports to each in one line. `Covariance` holds no `w`
  at all — its weights live inside `ce`, whose type bound admits any
  `StatsBase.CovarianceEstimator` — so it needs a way to read them first.
  [#492](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/492) holds that question,
  because widening the decision from one estimator to the moment layer is a contract decision rather
  than a defect fix, and #490 scoped its question to `SimpleVariance`.
- **`test_08_moments.jl` holds the contract.** It pins that the vector path and the matrix path
  answer the same number under the same weights, that the unweighted path did not move, and that
  the `mean` keyword still reaches the unweighted centre.
