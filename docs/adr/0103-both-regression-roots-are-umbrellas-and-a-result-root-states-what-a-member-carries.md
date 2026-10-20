---
status: accepted
---

# Both regression roots are umbrellas, and a result root states what a member carries

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
rule is that every decision must reproduce the oracle, and may only **add**
capability or **simplify** the design.

Four facts about the library decided the shape.

 1. **Thirty-three sites bind the two roots outside their own file**, and every one of the result
    bounds reads `rr.M`. A plain subtype of the old root would have matched them at once, and every
    one of them would have accepted a value it cannot run.
 2. **One asset index cannot serve both results.** `port_opt_view(re::Regression, i)` takes `i` as
    an asset index and slices the **rows** of `M`. A residual matrix is `observations × assets`, so
    the same index slices its **columns**.
 3. **`regression(re::Regression, args...)` is a greedy passthrough.** It returns its first
    argument for any trailing arguments, so a four-argument `regression` method placed beside it
    would return a loadings result silently instead of raising.
 4. **The fitting geometry and the payload part company.** A root can state how a member was
    fitted, or what a member carries, and the first version of this decision stated both at once:
    a result was "fitted per asset over the observations" **and** "carries the loadings matrix
    `M`". [Issue #649](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/649) found a
    member that meets one criterion and not the other. `CrossSectionalFactorModel` is fitted per
    observation across the assets **and** carries `M`, so under the paired criteria it belonged to
    neither child.

## Decision

### Each root gains two children, and the concrete types re-parent

```julia
abstract type AbstractRegressionEstimator <: AbstractEstimator end                    # umbrella
abstract type AbstractTimeSeriesRegressionEstimator     <: AbstractRegressionEstimator end
abstract type AbstractCrossSectionalRegressionEstimator <: AbstractRegressionEstimator end

abstract type AbstractRegressionResult <: AbstractResult end                          # umbrella
abstract type AbstractLoadingsRegressionResult       <: AbstractRegressionResult end
abstract type AbstractCrossSectionalRegressionResult <: AbstractRegressionResult end
```

`StepwiseRegression` and `DimensionReductionRegression` become time-series estimators, and
`Regression` becomes a loadings result. The umbrella declares **no** interface of its own: the
`# Interfaces` section moved down to each child, because the two families answer different verbs.

None of the four is exported, per `CLAUDE.md`.

### An estimator root states its geometry, and a result root states its payload

Fact 4 forces the two sides apart, and each side takes the criterion its consumers bind.

An estimator carries no payload, so it is named by the verb it answers and the geometry that verb
fits. `AbstractTimeSeriesRegressionEstimator` fits one model per asset over the observations and
answers `regression`. `AbstractCrossSectionalRegressionEstimator` fits one model per observation
across the assets and answers `cross_sectional_regression`.

A result carries a payload, and every consumer of a result binds the payload, so a result root is
named by what a member carries. `AbstractLoadingsRegressionResult` carries the loadings matrix `M`,
one row per asset and one column per factor, whatever geometry fitted it.
`AbstractCrossSectionalRegressionResult` carries no loadings matrix, because the exposures are the
regression's input and an Exposure Estimator produces them.

The cross-sectional result root keeps its name. Its rule already reads on the payload, and
`CrossSectionalRegression` still meets it.

### `RegE_Reg` pairs the loadings result with the time-series estimator

```julia
const RegE_Reg = Union{<:AbstractLoadingsRegressionResult,
                       <:AbstractTimeSeriesRegressionEstimator}
```

Every consumer of the alias reads the loadings matrix. The alias therefore names the two ways a
consumer obtains one: a result that already carries `M`, and an estimator whose verb returns such a
result. The two arms state different criteria, which is fact 4 made concrete, and the alias's
docstring states the asymmetry rather than hiding it.

The estimator arm stays on the time-series root. Renaming it would name an estimator by a payload it
does not hold, and a cross-sectional estimator returns a result that carries no loadings, so the arm
would then have to exclude its own sibling by hand.

The 33 bounds retighten to the two children in the same change, across `12_ConstraintGeneration/`,
`13_Prior/`, `19_RiskMeasures/27_ExpectedRisk.jl` and `20_Optimisation/`.

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
then recovering `b_t = ybar - f_t . xbar`. The oracle's own prior refuses an
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
- A loadings result fitted by any geometry reaches every loadings consumer with no further
  widening. That is what `CrossSectionalFactorModel` needs, and it is why the result roots split on
  the payload rather than on the geometry.

## Amendment (2026-10-09)

[Issue #1625](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1625) gave the
rank-deficiency policy to both members. `CrossSectionalTargetRegression(tgt, alg, intercept, ex)`
now carries `alg` too, and a member means the same thing under both estimators.

Before, the target saw the full design. `GLM` then dropped a collinear column by its own pivot, and
refused a factor set that is dependent to rounding alone with a `PosDefException`.
[#1620](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1620) made the target drop the
dependent columns by the rank test, always. That gave a dropped factor a return of zero, so a
least-squares target and `CrossSectionalLinearRegression` agreed on the fitted values and not on the
factor returns. On the panel of #1620 the asset `sigma` of the two differed by 9 %.

A fifth member, `DependentColumnDrop()`, is the drop of #1620. Under the linear member it solves the
kept columns with `\`. Each member on a target:

| Member | On a rank-deficient design of a target |
| --- | --- |
| `PseudoInverseFallback()`, the default | Fit the kept columns, then project the answer onto the row space of the weighted design |
| `DependentColumnDrop()` | Fit the kept columns, and give each dropped factor zero |
| `RankDeficiencyRefusal()` | Refuse before the target fits, with the message of the linear member |
| `UncheckedSolve()` | Hand the full design to the target, as before #1620 |
| `MinimumNormSolve()` | Refused by the constructor |

Every coefficient vector that differs from the fit by a vector of the null space gives the same
linear predictor, and the projection is the one of least norm among them. For a least-squares target
it is the answer of the pseudo-inverse: measured on #1625, the two agree to 2.2e-16, and the factor
returns of the two estimators agree to 5.6e-16. `MinimumNormSolve()` would be a second name for the
same answer, because the projection changes nothing at full rank. The projection holds only for a
target that reads the design through the linear predictor alone, which is what `is_basis_invariant`
states. So the constructor refuses `PseudoInverseFallback()` for a target that answers `false`, and
its message names the other three members. Both library targets answer `true`.

The carry fold needs no change. A move solves each rank-deficient row again through the estimator,
so the new `alg` reaches it, and the answer of every member that answers such a row reads the basis.
`RankDeficiencyRefusal()` refuses the stream at the step whose batch fit refuses, with the same rank.
The stream names the position of the observation within its step
([#1629](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1629)).

The positional constructor of `CrossSectionalTargetRegression` takes four arguments in place of
three. A caller who builds it by position must add `alg`.
