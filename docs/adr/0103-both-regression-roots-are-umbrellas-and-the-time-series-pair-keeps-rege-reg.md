---
status: accepted
---

# Both regression roots are umbrellas, and the time-series pair keeps `RegE_Reg`

## Context

The library's regression family fits **one model per asset over the observations**.
`AbstractRegressionEstimator` has two members, `StepwiseRegression` and
`DimensionReductionRegression`, and both return a `Regression(M, L, b)` whose loadings matrix `M`
carries one row per asset.

A cross-sectional factor prior needs the transpose of that operation: **one model per observation
across the assets**. The design is a three-dimensional tensor, `observations × assets × factors`,
the target is a matrix of returns, the weights are a matrix of the same shape, and the answer is a
factor-return matrix, a residual matrix and a count of the assets that entered each fit.

[Issue #648](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/648) decided the shape and
[issue #679](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/679) built it. Both sit
under map [#643](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/643), whose governing
rule is that every decision must reproduce the reference implementation, and may only **add**
capability or **simplify** the design.

Three facts about the library decided the shape.

 1. **Thirty-three sites bind the two roots outside their own file**, and every one of the result
    bounds reads `rr.M`. A plain subtype of the old root would have matched them at once, and every
    one of them would have accepted a value it cannot run.
 2. **One asset index cannot serve both results.** `port_opt_view(re::Regression, i)` takes `i` as
    an asset index and slices the **rows** of `M`. A residual matrix is `observations × assets`, so
    the same index slices its **columns**.
 3. **`regression(re::Regression, args...)` is a greedy passthrough.** It returns its first
    argument for any trailing arguments, so a four-argument `regression` method placed beside it
    would return a time-series result silently instead of raising.

## Decision

### Each root gains two children, and the concrete types re-parent

```julia
abstract type AbstractRegressionEstimator <: AbstractEstimator end                    # umbrella
abstract type AbstractTimeSeriesRegressionEstimator     <: AbstractRegressionEstimator end
abstract type AbstractCrossSectionalRegressionEstimator <: AbstractRegressionEstimator end

abstract type AbstractRegressionResult <: AbstractResult end                          # umbrella
abstract type AbstractTimeSeriesRegressionResult     <: AbstractRegressionResult end
abstract type AbstractCrossSectionalRegressionResult <: AbstractRegressionResult end
```

`StepwiseRegression` and `DimensionReductionRegression` become time-series estimators, and
`Regression` becomes a time-series result. The umbrella declares **no** interface of its own: the
`# Interfaces` section moved down to each child, because the two families answer different verbs.

None of the four is exported, per `CLAUDE.md`.

### `RegE_Reg` names the time-series pair

```julia
const RegE_Reg = Union{<:AbstractTimeSeriesRegressionResult,
                       <:AbstractTimeSeriesRegressionEstimator}
```

Every consumer of the alias reads the loadings matrix, which only a time-series result carries. The
33 bounds retighten to the time-series children in the same change, across
`12_ConstraintGeneration/`, `13_Prior/`, `19_RiskMeasures/27_ExpectedRisk.jl` and
`20_Optimisation/`.

### The result is a sibling, and `Regression` is untouched

`CrossSectionalRegression(f, eps, n, b)` holds the factor returns, the residuals, the eligible
asset counts and the optional per-observation intercepts. Its `port_opt_view` slices `eps` on its
**second** axis and leaves the other three alone, which is fact 2 above made concrete.

### The verb is its own, and the weights are an argument

`cross_sectional_regression(cre, Z, X, W)`, on fact 3. `W` is an argument rather than a field,
because a two-pass weighting scheme calls the estimator twice on one design with two different
weight matrices, and a policy stored on the estimator would force a second estimator object or a
mutation.

### Two members, and a four-member rank-deficiency policy

`CrossSectionalLinearRegression(alg, intercept)` runs the closed form on the **weighted design**
rather than on the normal matrix, which halves the condition number in the exponent.
`CrossSectionalTargetRegression(tgt, intercept)` fits one external target per observation and hands
it the cross-sectional weights as observation weights through `factory`.

`alg` is one of `PseudoInverseFallback()` (the default), `RankDeficiencyRefusal()`,
`UncheckedSolve()` and `MinimumNormSolve()`. The rank test reads the `R` diagonal of the pivoted
`QR` inline, because `rank(::QRPivoted)` needs Julia 1.12 while `Project.toml` allows 1.11.

The intercept is a `Bool` and it is fitted by demeaning the cross-section by its weighted centroid,
then recovering `b_t = ybar - f_t . xbar`. The reference implementation's own prior refuses an
intercept, and its regressor is public, so the flag stays: dropping it would remove a mode.

## What the build measured, against the ticket's stated ground truth

Ground truth 3 of #648's resolution states that Julia's `\` returns a **basic** solution for a
rank-deficient non-square design, zeroing the coefficients past the numerical rank. **Measured on
Julia 1.12.7, it does not.** The pivoted-`QR` `ldiv!` follows LAPACK's `xGELSY`: it completes the
orthogonal factorisation and returns the **minimum-norm** solution, which agrees with `pinv` to
round-off.

The four members remain four distinct policies, and the real split is elsewhere:

| Design | `UncheckedSolve` | `PseudoInverseFallback` | `MinimumNormSolve` | `RankDeficiencyRefusal` |
| --- | --- | --- | --- | --- |
| Non-square, rank deficient | minimum norm | minimum norm | minimum norm | refuses, naming the observation |
| Square, exactly singular | throws `SingularException` | minimum norm | minimum norm | refuses, naming the observation |
| Nearly dependent | the `LU` or `QR` answer | the same answer, after a rank test that passes | the pseudo-inverse's truncated answer | the same answer as the fallback |
| Full rank | one factorisation | two factorisations | the pseudo-inverse | two factorisations |

So `UncheckedSolve` buys one factorisation instead of two and pays for it on a **square** singular
design, and `MinimumNormSolve` parts from the other two only where the two tolerances disagree.

## Consequences

- A consumer that reads loadings is now refused at its own signature rather than deep inside a
  factor lift, and the refusal names the type it wanted.
- A cross-sectional estimator cannot reach `regression`, and a time-series one cannot reach
  `cross_sectional_regression`. Neither family names the other's types.
- The umbrellas are extension points. A third regression geometry subtypes a new child of the
  umbrella and inherits no contract it cannot meet.
- A caller who wrote `re::AbstractRegressionEstimator` in their own code and passed a
  `StepwiseRegression` is unaffected, because the umbrella still matches it. A caller who **stored**
  a value under the old root and hands it to a library consumer now meets a `MethodError` at the
  call rather than a silent wrong answer.
