---
status: accepted
---

# A type is kept by default, and `nothing` spells a skipped procedure or a trivial default

## Context

The map [#1852](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1852) audits the
library for types that it can simplify: wrappers, near duplicates, nested structs that can flatten,
and types that hold no data. Each candidate gets a verdict ticket. The verdicts need one rule to
cite, or each ticket argues the question again
([#1853](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1853)).

One question came back many times: how does a field say "off"? The library had three spellings.

- `nothing`, as in `hac_lags = nothing` or `MaximumRatio(; ohf = nothing)`.
- A singleton type, as in `RawStatistic()`, `NoSlack()` or `IdentityScaling()`.
- A `Bool` or a `Symbol`, as in `PricesToReturns(; padding = false)`.

[#1867](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1867) found a struct with two
spellings. `RegimeAdjustedExpWeightedVariance` turned `regime_method` off with `nothing` and
`hac_floor` off with `NoHacFloor()`. The research of
[#1858](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1858) found more cases of this
kind.

A first ruling kept every singleton that a method dispatched on, and every singleton with a
descriptive name. That ruling removed `NoHacFloor` and `NoPeel` alone. The maintainer then ruled
that the first ruling kept too much. A singleton such as `RawStatistic`, `NoSlack` or
`IdentityScaling` adds a public name, a docstring, an export, a catalogue entry and a test for a
setting that `nothing` states as well. The two names that
[#1878](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1878) added as aliases of
`nothing`, `PlainCholeskySquareRoot` and `PairMeanGerberIQScaler`, were two spellings of one
setting.

## Decision

### `nothing` spells every setting that it can state

A field takes `nothing`, and no singleton, for one of these settings:

1. **A skipped procedure.** The step does not run: no floor, no peel, no slack, no clamp, no
   debias, no decay, no fill, no repair of the systematic block.
2. **A trivial default.** The rule is the identity or the plain form: unit weights, no scaling,
   the raw variance state, the untransformed gradient, the plain Cholesky factor, the pair mean of
   the Gerber IQ scale, or a value sized from the resolved characteristic, as in
   `MaximumRatio(; ohf = nothing, kmin = nothing)`.

The method on `Nothing` holds the behaviour of the setting, and the docstring of the field states
it. The library adds no singleton for such a setting, and no named alias of `nothing`. A method that
dispatches on a singleton is not a reason to keep it, because the method on `Nothing` gives the same
dispatch. A caller with another rule writes a subtype of the family.

### A singleton stays only where `nothing` cannot state its role

A data-less type stays when one of these is true:

1. **It names a rule with content of its own**, a formula or a value that is not the identity or a
   skipped step. `IdentityTarget` is target A of Schäfer and Strimmer, a real shrinkage target.
   `tgt = nothing` would read as "no target". `ZeroUnseenMember` and `ZeroUnknown` state the value
   that they write, zero, and `nothing` does not state a value. `UnitIntervalIntensity` and
   `NonNegativeIntensity` are clamps.
2. **`nothing` already has another meaning in the field.** `NoDefault` of `TimeDependent` is a
   sentinel, because `nothing` is a valid default of a time-dependent field.
3. **It holds data.** `NoRisk` and `NoReturn` hold `settings`, so they are not data-less.

### Keep every other type by default

For a type that is not a data-less setting, such as a wrapper, a near duplicate or a nested struct,
keep it by default. Simplify a candidate only when all four checks fail:

1. No method of the library dispatches on the type.
2. No second site reuses it.
3. It gives no measured runtime or compile benefit, such as specialisation, type stability or a
   zero-cost singleton.
4. No ADR names it deliberate.

The maintainer then rules on each candidate. Nested structs and near duplicates are often features
of the library. They give reuse and dispatch at no extra cost.

An entry on the `kept_types` allow-list of `test/test_70_trivial_unit_census.jl`, under ADR 0168,
meets check 4. Its reason names the type deliberate, so the type stays. A kept type whose docstring
tells a caller to subtype it is finished as a `public` extension point, with an `# Interfaces`
section. Its interface is the verbs of its member that a second member needs, and a method whose
body calls only those verbs dispatches on the abstract type
([#1874](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1874)).

### A step on the output of a covariance estimator is the wrapper

Matrix processing that runs on the output of a covariance estimator has one spelling: the wrapper
`PortfolioOptimisersCovariance(ce, mp)`. A covariance estimator carries no `mp` of its own. The
wrapper takes any `StatsBase.CovarianceEstimator`, also one that another package defines, so a
field on the estimator is a second spelling of one step. `DenoiseCovariance`, `DetoneCovariance`
and `ProcessedCovariance` are functions that build the wrapper, not a third spelling.

An estimator carries `mp` or `pdm` as a field only when the step processes a matrix that the
estimator assembles itself, so no wrapper can reach the matrix:

1. **A step inside the algorithm.** `GerberCovariance`, `SmythBrobyCovariance` and
   `GerberIQCovariance` repair the correlation with `pdm` before they rescale it.
   `GeodesicShrinkageCovariance` repairs a block.
2. **A matrix that a prior assembles.** `FactorPrior`, `CrossSectionalFactorPrior` and the
   Black-Litterman priors process the posterior covariance that they build from parts.
3. **A moment that is not a covariance.** `Coskewness`, `Cokurtosis` and `NegativeSkewness`
   process a matrix of higher order.

The wrapper passes the four checks. Many methods dispatch on it, and it is the default covariance
estimator of the priors and of the uncertainty sets (ADR 0016). So it stays
([#1865](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1865)).

### An abstract type, not a `Union` of its members

A method or a bound names the abstract type where a `Union` would spell all its members. A caller
can subtype an abstract type, but a caller cannot add a member to a `Union`. A `Union` stays in
three cases:

1. **An abstract type cannot state the set.** The set mixes families, as a field that takes an
   estimator or a result does.
2. **The abstract type has a method of its own.** The method is a fallback or a refusal, and the
   `Union` method serves the known members. `cross_sectional_alive_folds` returns `false` on
   `AbstractCrossSectionalRegressionEstimator` and reads the field `alg` of its two members.
3. **It is a named alias for convenience**, or a set that a docstring closes on purpose, as
   `AbstractLowOrderPriorEstimator_A_F_AF` does.

A census of an abstract type with no dispatch reads a `Union` of all its members as a defect, not
as a reason to remove the type
([#1875](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1875)).

The rule holds inside a `Union` that mixes families too. Where some members of the `Union` are all
the members of one abstract type, the `Union` names that type in their place. The online methods
take `Union{JuMPOptimiser, BaseClusteringOptimisationEstimator, InverseVolatility, …}`, not
`Union{JuMPOptimiser, HierarchicalOptimiser, InverseVolatility, …}`
([#1874](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1874)).

### A `Bool` or a `Symbol`

A `Bool` or a `Symbol` is acceptable, mainly where it reads better than `nothing`, as in
`PricesToReturns(; padding = false)`. ADR 0044 still prefers a named type where a flag selects
between two sources.

### Cases ruled under this decision

| Field | Spelling | Removed name |
| --- | --- | --- |
| `hac_floor` of the regime-adjusted estimators | `nothing`, no floor | `NoHacFloor` |
| `CoveragePolicy.peel` | `nothing`, no peel | `NoPeel` |
| `debias` of the regime-adjusted estimators | `nothing`, the raw statistic | `RawStatistic` |
| `RegimeAdjustedExpWeightedCovariance.cor_vol` | `nothing`, the raw variance state | `RawVolatility` |
| `PassiveAggressiveMeanReversion.slack` | `nothing`, PAMR with no slack | `NoSlack` |
| `MirrorDescent.grad` | `nothing`, the untransformed gradient | `PlainGradient` |
| `NumericPanelInput.alg`, and the categorical and tensor inputs | `nothing`, no fill, so a blank cell is refused | `NoPanelFill` |
| `OrthogonalUncertaintySet.scaling` | `nothing`, no scaling | `IdentityScaling` |
| `OrthogonalUncertaintySet.metric`, and the `weighting` keywords | `nothing`, unit weights | `IdentityMetric` |
| `CrossSectionalFactorPrior.srep` | `nothing`, no `:pdm` step on the systematic block | `NoSystematicRepair` |
| `Proximity.decay` | `nothing`, a flat score of one | `NoDecay` |
| `intensity` of `JamesStein` and `BayesStein` | `nothing`, no clamp | `UnclampedIntensity` |
| `mtx_sqrt` | `nothing`, the plain Cholesky factor | `PlainCholeskySquareRoot` |
| `GerberIQCovariance.sc` | `nothing`, the pair mean of the volatilities | `PairMeanGerberIQScaler` |
| `ForecastReversion.scale` | `nothing`, no preconditioning | none |
| `PricesToReturns.gap_return_alg` | `nothing`, the simple default rule | none |
| `PricesToReturns.ret_method`, `padding` | `Symbol`, `Bool` | none |
| `DriftedWeights.wd` | the field is removed, so `DriftedWeights()` is a singleton | the field `wd`. It had one reader, and it had no effect when the scheme set it. `DriftedWeights` and `TargetWeights` stay, because methods dispatch on them and ADR 0110 names them. A new drift of the held weights is a new sibling ([#1860](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1860)). |
| `ImpliedVolatility.mp` | the wrapper `PortfolioOptimisersCovariance(; ce = ImpliedVolatility(; …), mp)` | the field `mp`. It processed the last matrix, the same step as the wrapper ([#1865](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1865)). |

`IdentityTarget`, `NoDefault`, `ZeroUnseenMember`, `ZeroUnknown`, `ZeroStartCentring`,
`UnitIntervalIntensity`, `NonNegativeIntensity`, `NoRisk` and `NoReturn` stay, for the reasons
above. `ZeroStartCentring` is the location of the oracle, a rule with content of its own.

## Considered options

- **Keep every singleton that a method dispatches on, or that has a descriptive name.** This was
  the first ruling of #1867, and the maintainer reversed it. Each such singleton costs a public
  name, a docstring, an export, a catalogue entry and a test. The method on `Nothing` gives the
  same dispatch, and the docstring of the field gives the same description.
- **Simplify by default.** Remove every type that a smaller form can replace. This was rejected
  for types that hold data or a rule with content. It costs dispatch points that callers overload,
  and specialisation that a singleton gives at no cost.
- **A singleton for every "off" value.** This was rejected. `nothing` is the spelling of an
  optional value in the whole library.
- **A named leaf beside `nothing`, as an alias.** This was the ruling of #1878, and the maintainer
  reversed it. Two spellings of one setting is the defect that #1867 found.
- **An `mp` field on every covariance estimator, in place of the wrapper.** This was rejected
  (#1865). It adds a field to each plain covariance estimator, and it cannot reach an estimator
  that another package defines.

## Consequences

- A new field that can skip a procedure, or that has a trivial default, takes
  `Option{<:AbstractX}` with a method on `Nothing`. A family can have `nothing` as its only trivial
  member, beside its rules with content.
- A new covariance estimator takes no `mp` field. A caller wraps it in
  `PortfolioOptimisersCovariance` for matrix processing. A caller that set `mp` on an
  `ImpliedVolatility` moves it to the wrapper.
- A fallback method on the abstract family that refuses a subtype with no method of its own stays
  separate from the method on `Nothing`, as #1878 made it for the Gerber IQ verbs.
- The removed names that v0.32 exported have a line in the migration guide. A caller replaces
  `X()` with `nothing`, and a method of a caller on `X` becomes a method on `Nothing`.
- A verdict ticket of #1852 cites this ADR. For a type that is not a data-less setting, it records
  the four checks.
- Related: ADR 0044 prefers a named type over a flag. ADR 0154 states when an abstract type is
  `public`.
