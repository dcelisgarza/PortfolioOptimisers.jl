---
status: accepted
---

# The elastic net is a penalised regression target

## Context

`KernelTrendPattern` fits the intermediate state of the kernel-based trend pattern tracking of
lai2018ktpt with an elastic net. The options of the fit were a nested struct, `ElasticNetPath`, in
a field bound to that one concrete type. A private function, `elastic_net_path`, read it.

The verdict ticket [#1880](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1880) of
the map [#1852](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1852) applied the four
checks of ADR 0222. All four failed: no dispatch choice, no second site, no measured benefit, and no
ADR. The concrete bound gave the nesting no extensibility, because no other part could replace the
elastic net.

The maintainer ruled that the point of a nested part is that a user can swap it. The model of
lai2018ktpt fits a regression, and the elastic net is the choice of the paper. The library already
has an extension point for a swappable regression, `AbstractRegressionTarget`, with an
`# Interfaces` section and the members `LinearModel` and `GeneralisedLinearModel`. Five estimators
fit a target through `StatsAPI.fit`. An elastic net in that family is available to all of them.

## Decision

1. **The elastic net is a regression target.** `ElasticNet` replaces `ElasticNetPath`. It subtypes
   `AbstractPenalisedRegressionTarget`, and it has the methods of the `# Interfaces` section of
   `AbstractRegressionTarget` that its estimators read: `StatsAPI.fit`, `StatsAPI.coef`,
   `StatsAPI.predict`, `factory` with observation weights, and `regression_target_weights`. Its
   fields are `theta`, `strength`, `ve`, `iters` and `tol`, plus the observation weights and the
   intercept column that `factory` sets.
2. **The family has two branches.** `AbstractUnpenalisedRegressionTarget` holds `LinearModel` and
   `GeneralisedLinearModel`. `AbstractPenalisedRegressionTarget` holds `ElasticNet`. Both are
   `public`, and a user target subtypes one of them.
3. **The strength rule is a nested part.** `strength` is bound to `AbstractPenaltyStrength`.
   `MiddleOfPath(; ratio = 1e-3)` is the rule of lai2018ktpt and the default: the geometric middle
   of the path, `γ_max √ratio`. `FixedStrength(; gamma)` takes the strength from the user.
   `MiddleOfPath` needs `theta > 0`, because a ridge alone has no `γ_max`, so the constructor of
   `ElasticNet` refuses `theta = 0` with it. `FixedStrength` accepts `theta = 0`, which is a ridge.
4. **The standardisation goes through `ve`**, as ADR 0221 states for a moment. `ve = nothing`, the
   default, fits the columns as they are, as lai2018ktpt do. A variance estimator in `ve` divides
   each penalised column by its standard deviation before the fit, and scales the coefficients back
   after it.
5. **The estimator names the intercept column.** An estimator that adds a column of ones to the
   design calls `factory(tgt, InterceptColumn(k))` before the fit. A method on
   `AbstractRegressionTarget` returns the target unchanged, so an unpenalised target ignores the
   call. `ElasticNet` keeps `k`, and does not penalise or standardise that column.
6. **`StepwiseRegression` binds `tgt` to `AbstractUnpenalisedRegressionTarget`.** Its criteria do
   not hold for a penalised fit. `PValue` reads p-values that a penalty makes invalid. `aic`,
   `aicc`, `bic` and `adjr2` count one degree of freedom for each coefficient, which a penalised fit
   does not have. The L1 penalty also selects the factors itself.
7. **`KernelTrendPattern` takes `tgt::AbstractRegressionTarget = ElasticNet()`**, the field name of
   `StepwiseRegression` and `CrossSectionalTargetRegression`. Its fold fits the target and predicts
   with it. `KernelTrendPatternTracking` keeps the flat keywords of the paper, as every constructor
   named for a paper does, and adds `ratio`.

## Considered options

- **Keep `ElasticNetPath` nested and concrete.** Rejected. The concrete bound lets no part replace
  it, so the nesting gives no extensibility, and all four checks of ADR 0222 fail.
- **Flatten the options onto `KernelTrendPattern`**, as `SpatialMedian` holds `iters` and `tol`.
  Rejected. It removes the one place where a different regression can enter.
- **A regression family for `KernelTrendPattern` alone.** Rejected. It makes a second regression
  family beside `AbstractRegressionTarget`, and the regression estimators do not get the elastic
  net.
- **The target detects a constant column and leaves it unpenalised**, as glmnet leaves its
  intercept. Rejected. `KernelTrendPattern` rebuilds its levels with the current level at one, so
  the last column of its design is a column of ones for every asset. Detection would remove the
  penalty from that coefficient, and the fit would no longer be Equation (15) of lai2018ktpt.
- **The user names the unpenalised column in a field.** Rejected. The user must then know where
  each estimator puts its column of ones: first in three estimators, last in
  `TargetReturnForecast`.
- **`StepwiseRegression` refuses a penalised target through a dispatched check** with an
  `ArgumentError`. Rejected by the maintainer in favour of the bound, so the type system states
  which targets the stepwise search accepts.
- **A `Bool` `standardise` field.** Rejected. It fixes the variance rule, and ADR 0221 already
  spells a standardisation with `ve`.

## Consequences

- A clean break with no deprecation, as ADR 0044 and ADR 0213 set. `ElasticNetPath(; theta, ratio,
  iters, tol)` becomes `ElasticNet(; theta, strength = MiddleOfPath(; ratio), iters, tol)`, and
  `KernelTrendPattern(; path)` becomes `KernelTrendPattern(; tgt)`. A user target that subtyped
  `AbstractRegressionTarget` directly now subtypes one of the two branches.
- `ImpliedVolatilityRegression`, `DimensionReductionRegression` and `TargetReturnForecast` with
  `intercept = true` call `factory(tgt, InterceptColumn(k))`. `StepwiseRegression` does not need
  the call, because its bound refuses a penalised target. `CrossSectionalTargetRegression` centres
  `y` and the design and adds no column of ones, so it does not need the call either.
- `ElasticNet` is not basis invariant, so `CrossSectionalTargetRegression` fits it on the design as
  it is. `TargetReturnForecast` refits it at each step, because only `LinearModel` folds.
- The `factory` call with `InterceptColumn` is a direct call on the target. It is not a field
  propagation of ADR 0001.
- A research ticket asks whether the effective degrees of freedom of the elastic net can lift the
  bound of `StepwiseRegression` for `aic`, `aicc`, `bic` and `adjr2`.
- Related: ADR 0221 (`ve` standardises), ADR 0222 (the keep rule), ADR 0158 and ADR 0172
  (`KernelTrendPattern`).
