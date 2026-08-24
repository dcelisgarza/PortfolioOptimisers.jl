---
status: accepted
---

# A moment estimator's observation weights weight its centre, not its deviations alone

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

[#492](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/492) then found the same split
in the three sibling estimators of the moment layer, so this ADR decides the four together.

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

[`Covariance`](../../src/08_Moments/03_Covariance.jl),
[`Coskewness`](../../src/08_Moments/19_Coskewness.jl) and
[`Cokurtosis`](../../src/08_Moments/20_Cokurtosis.jl) resolve `mu` from their own `me` in that same
one line, so each carried the same split.

### The two readings

The ticket named both, and both are defensible.

 1. **`w` weights the whole estimate.** A caller who sets `w` on the estimator means the weighted
    moment, so the centre is weighted too.
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
deviations is neither the weighted moment nor the unweighted one. `Statistics.var(X, w; dims = 1)`
centres on the weighted mean, so the split also cost parity with the function the method wraps.

**Reading 2 survives the fix at the call site.** The `mean` keyword takes any centre, so a caller
who wants a differently weighted centre still writes it:

```julia
var(ve, Xc; dims = 1, mean = Statistics.mean(SimpleExpectedReturns(), Xc; dims = 1))
```

What reading 2 loses is the ability to **store** that combination in one estimator, not the ability
to compute it.

### `Covariance` held no weights to read

`SimpleVariance`, `Coskewness` and `Cokurtosis` each hold `w` as their own `@wprop` field, so the
kernel can read it. `Covariance` held `me`, `ce` and `alg` alone. Its weights lived one level down,
inside whatever `StatsBase.CovarianceEstimator` `ce` happened to be — a `GeneralCovariance` by
default, but the field's type bound admits any estimator, including one from `CovarianceEstimation.jl`
that holds no weights at all. There is no generic way to ask an arbitrary `ce` for its observation
weights, so the fix above had no second argument to pass. #492 named three ways out:

 1. **Give the weights a reader.** Add a verb that answers an estimator's observation weights, and
    let a type opt in.
 2. **Give `Covariance` its own `w`.** A `@wprop w` field beside `me`, `ce` and `alg` makes the fix
    identical to the one the other three take.
 3. **Leave it**, and say so in the docstring.

Option 2 was taken, because it makes the four estimators of the moment layer one shape and one
contract. Option 1 adds a verb to the propagation machinery for one call site. Option 3 keeps
`Covariance` different from its three siblings, which is the inconsistency ADR 0087 spent its
argument on.

## Decision

**A moment estimator's observation weights weight its centre.** The rule holds for
[`SimpleVariance`](../../src/08_Moments/04_SimpleVariance.jl),
[`Covariance`](../../src/08_Moments/03_Covariance.jl),
[`Coskewness`](../../src/08_Moments/19_Coskewness.jl) and
[`Cokurtosis`](../../src/08_Moments/20_Cokurtosis.jl).

`simple_variance_kernel` resolves the centre through `factory(me, ve.w)` when the caller supplies no
`mean`:

```julia
mu = if !isnothing(mean)
    mean
elseif isnothing(ve.w)
    Statistics.mean(me, X; dims = dims, kwargs...)
else
    Statistics.mean(factory(me, ve.w), X; dims = dims, kwargs...)
end
```

`coskewness` and `cokurtosis` take the same expression against `ske.w` and `kte.w`.

The `isnothing(ve.w)` branch is a performance guard, not a second contract. `ve.w` is a field, so
its type decides the branch and the compiler folds the test away. The guard keeps a windowed loop
from rebuilding the estimator tree of `me` once per window when no weights exist to write.

**`Covariance` gains a `@wprop w` field.** It sits last, beside `me`, `ce` and `alg`, and it carries
the same `Option{<:ObsWeights}` bound and the same `assert_nonempty_nonneg_finite_val` check as its
three siblings. `covariance_centre_and_estimator` resolves the centre and the inner estimator
together, and the four methods of `Statistics.cov` and `Statistics.cor` read both from it:

```julia
function covariance_centre_and_estimator(ce::Covariance, X::MatNum; dims::Int = 1,
                                         mean = nothing, kwargs...)
    if isnothing(ce.w)
        mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
        return mu, ce.ce
    end
    me = factory(ce.me, ce.w)
    mu = isnothing(mean) ? Statistics.mean(me, X; dims = dims, kwargs...) : mean
    return mu, factory_child(ce.ce, ce.w)
end
```

`ce.w` therefore reaches the centre through `ce.me` and the deviations through `ce.ce`, so one field
weights the whole estimate. `factory_child` is the verb that already answers this question for a
field of unbounded type: an estimator of the library recurses through `factory`, and a
`StatsBase.CovarianceEstimator` the library does not own passes through unchanged.

**The estimator's own `w` wins over the weights that `me` and `ce` carry.** The rule matches
`factory`, which has always overwritten them, so an estimator behaves the same before and after it
passes through the weight channel.

**The vector method of `SimpleVariance` does not change.** It already centres on the weighted mean,
and it has no `me` to read.

**The `mean` keyword stays the escape hatch.** A caller who wants a centre that the estimator's own
weights do not describe passes it.

## Consequences

- **No internal call site moves.** The standard-deviation risk measures, the moment priors and the
  backtests reach these estimators through `factory`, which already weighted `me.w` and `ce.w`. The
  measured numbers are the same before and after.
- **A hand-built weighted estimator moves.** `SimpleVariance(; w = aw)` over a matrix answers
  `2.080645161290323` where it answered `2.383512544802868`. That is the defect this ADR fixes, and
  it is the whole size of the behaviour change for the three estimators that already held a `w`.
- **`Covariance` gains a fourth field and a fourth type parameter.** `Covariance(me, ce, alg)` is
  now `Covariance(me, ce, alg, w)`. Every construction site in `src/`, `ext/` and `test/` uses the
  keyword constructor, so none of them moves, and `Covariance{<:Any, <:Any, <:FullMoment}` still
  selects on `alg` because a partial parameterisation is a `UnionAll`.
- **`Covariance` gains an `obs_weights_view` method.** The `obs` channel is gated by `@wprop`, so
  before this change `obs_weights_view` returned a `Covariance` unchanged and left the weights
  inside `me` and `ce` at their full length.
  [`ImpliedVolatility`](../../src/08_Moments/24_ImpliedVolatility.jl) is the one caller, and it
  measures a block of rows, so a weighted `Covariance` reached it with the weights of the whole
  sample. The generated method now indexes `me`, `ce` and `w` to the block.
- **Weights buried inside `ce` alone still centre on the unweighted mean.**
  `Covariance(; ce = GeneralCovariance(; w = aw))` is the case, and it is the price of option 2. No
  verb reads the weights of an arbitrary `StatsBase.CovarianceEstimator`, so `Covariance` cannot
  find them. `Covariance(; w = aw)` is the supported spelling, and it weights the centre and the
  deviations both.
- **A stored estimator can no longer hold a centre that is weighted differently from its
  deviations.** `SimpleVariance(; me = SimpleExpectedReturns(; w = w1), w = w2)` now centres on
  `w2`. The `mean` keyword expresses the old combination at the call site, and `factory` destroyed
  the stored form already.
- **The two paths of one estimator agree.** A vector and its one-column matrix answer the same
  number, which is what [#490](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/490)
  asked for.
- **`me` keeps its reason to be an estimator.** A shrunk or an equilibrium centre still differs
  from a simple mean. It now reads the same observation weights as the deviations it centres.
- **`test_08_moments.jl` holds the contract.** It pins that the vector path and the matrix path
  answer the same number under the same weights, that a hand-built weighted estimator and a
  `factory`-built one agree for each of the four, that the unweighted path did not move, and that
  the `mean` keyword still reaches the unweighted centre.
