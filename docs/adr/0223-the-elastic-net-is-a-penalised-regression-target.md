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

The first form of this ADR bound the target of `StepwiseRegression` to the unpenalised branch. The
research ticket [#1887](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1887) found
that three of its criteria can read the effective degrees of freedom of the elastic net. Remark
(The elastic net) after Equation (28) of tibshirani2012 states the count as the trace of the
smoother on the active set. Equation (2.5) of hurvich1998 states AIC and AICc for a linear smoother
in the form that `StatsAPI` uses. Equation (2.16) of zou2007 proposes BIC with the same count. The
verdict ticket [#1898](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1898) ruled on
that form, and items 6 and 8 to 11 state the ruling.

The research also found that `DimensionReductionRegression` and `CrossSectionalTargetRegression`
accept an `ElasticNet` already. Both count one spent parameter for each column, so both overstate
the parameters that a penalised fit spends.

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
6. **`StepwiseRegression` accepts a penalised target under `:aic`, `:aicc` and `:bic` alone.**
   These three criteria read the effective degrees of freedom of item 8. `PValue` reads p-values
   that a penalty makes invalid. `:r2` reads no count, and `:adjr2` needs a residual count that no
   source states for it. The check of the inner constructor, `assert_stepwise_regression`, has two
   methods, so the dispatch states the valid pairs. A `MinValStepwiseRegressionCriterion` takes
   `tgt::AbstractRegressionTarget`. `PValue`, a `MaxValStepwiseRegressionCriterion` and a
   criterion of the user take `tgt::AbstractUnpenalisedRegressionTarget`. A refused pair is a
   `MethodError`. The two methods are on the check and not on the inner constructor, because
   `@concrete` appends a generic positional constructor. A call that matches no inner method goes to
   that constructor with no check, so two inner methods refused no pair (measured on #1899). The docstring of
   `StepwiseRegression` holds the table of the valid and the refused pairs, and it states that
   zou2007 proposes BIC with the effective count and does not derive it. A ridge,
   `ElasticNet(; theta = 0, strength = FixedStrength(; gamma))`, selects no factor, so for it the
   search is the only selection.
7. **`KernelTrendPattern` takes `tgt::AbstractRegressionTarget = ElasticNet()`**, the field name of
   `StepwiseRegression` and `CrossSectionalTargetRegression`. Its fold fits the target and predicts
   with it. `KernelTrendPatternTracking` keeps the flat keywords of the paper, as every constructor
   named for a paper does, and adds `ratio`.
8. **A penalised fit states its effective degrees of freedom.** Let `H` be the smoother of the fit
   on its active set `A`, with the observation weights `V`, the design `P` and the penalty mask
   `D_A`: `H = P_A (P_Aᵀ V P_A + γ(1 − ϑ) D_A)⁻¹ P_Aᵀ V`. `StatsAPI.fit` writes into
   `ElasticNetFit` the model count `df̂ = tr(H)`, the weighted residual sum of squares and `nobs`.
   It reads `H` off the system that `elastic_net_polish` already factors. `ElasticNetFit` gets
   `StatsAPI.dof = df̂ + 1`, because the residual variance counts, as GLM counts it for
   `LinearModel`. It gets `StatsAPI.nobs`, `StatsAPI.deviance` and `StatsAPI.loglikelihood` at
   `σ̂² = RSS / nobs`. `ElasticNetFit` is not a `StatisticalModel`, so it gets `StatsAPI.aic`,
   `StatsAPI.aicc` and `StatsAPI.bic` directly, in the `StatsAPI` forms.
9. **The residual count of a penalised fit is `N − tr(2H − HᵀH)`.** Each estimator writes it into
   `edof` of `Regression` for a penalised target: `StepwiseRegression`,
   `DimensionReductionRegression`, and the per-period count of `CrossSectionalTargetRegression`.
   The count follows from `E[RSS] = σ² tr((I − H)ᵀ(I − H)) + ‖(I − H)μ‖²`, so it is the count
   whose chi-squared law has the correct mean, which `ResidualInflation` reads. For a projection,
   `HᵀH = H`, and the count is `N − p`. So no result of an unpenalised target changes. The
   docstring states the derivation. Section 3.5 of hastie1990 states the count. No copy was found,
   and #1552 lists it.
10. **`ElasticNet` gates the term of a strength that reads the data.** `MiddleOfPath` reads `γ` off
    `y`, so `tr(H)` misses the term `∇γᵀ ∂μ̂/∂γ` of the divergence of the fit. The field
    `sdiv::Option{<:AbstractStrengthDivergence} = nothing` holds the gate. `nothing` keeps
    `df̂ = tr(H)`, and the docstring gives the measured gap: under 0.06 of a degree of freedom at
    `theta = 0.5`, and 0.03 at `theta = 0.99`, on the design of #1887. `StrengthDivergence()` adds
    the term. The strength rule gives `∇γ` through `penalty_strength_gradient`, which the
    `# Interfaces` section of `AbstractPenaltyStrength` names, so it is `public`. It is zero for
    `FixedStrength`, so `StrengthDivergence()` with `FixedStrength` adds zero, and the docstring
    states it. The fit gives `∂μ̂/∂γ` with one more solve on the factorisation that it holds. The
    term changes the model count `df̂` alone. The residual count of item 9 stays, because no
    source states the residual law of a fit whose strength reads the data.
11. **A factor that the penalty sets to zero stays in `included`.** `included` records the factors
    that the search selected, not the active set of the last fit. The row of `M` holds the zero,
    and the docstring of `StepwiseRegression` states it.

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
  which targets the stepwise search accepts. #1898 kept this ruling for the narrower refusal of
  item 6.
- **Keep the bound of `StepwiseRegression` on the unpenalised branch.** Rejected by #1898. Three
  criteria have a source for the effective count, and a ridge has no other way to select factors.
- **Accept a penalised target under `:aic` and `:aicc` alone.** Rejected. zou2007 proposes BIC
  with the effective count when sparsity is the aim, and sparsity is the aim of the search.
- **Accept `:r2` and `:adjr2` too.** Rejected. No source treats either as a criterion for a
  penalised fit.
- **A `MethodError` hint, registered in `__init__`, that names the reason of a refused pair.**
  Rejected. The library has no such hint, and the docstring table states the pairs.
- **The residual count `N − tr(H)`.** Rejected. It overstates the residual count under a ridge
  part, so the bound of `ResidualInflation` is too narrow.
- **The Satterthwaite count `tr(R)² / tr(R²)`, with `R = (I − H)ᵀ(I − H)`.** Rejected. It matches
  the variance of the residual law too, but it also changes the divisor `ediv`, a wider change with
  no measured need.
- **Compute the term of item 10 always, or document the gap alone.** Rejected by the maintainer in
  favour of the gate, so the default costs nothing and the exact count is one field away.
- **The gate on `MiddleOfPath`.** Rejected. The rule cannot compute `∂μ̂/∂γ`, because the fit holds
  the factorisation, and a strength rule of the user would need its own field.
- **Refuse `MiddleOfPath` in the search.** Rejected. It refuses the default `ElasticNet()`.
- **Drop a factor with a zero coefficient from `included`**, after the search or inside it.
  Rejected. `included` records the search, and a removal inside the search changes its path.
- **A `Bool` `standardise` field.** Rejected. It fixes the variance rule, and ADR 0221 already
  spells a standardisation with `ve`.

## Consequences

- A clean break with no deprecation, as ADR 0044 and ADR 0213 set. `ElasticNetPath(; theta, ratio,
  iters, tol)` becomes `ElasticNet(; theta, strength = MiddleOfPath(; ratio), iters, tol)`, and
  `KernelTrendPattern(; path)` becomes `KernelTrendPattern(; tgt)`. A user target that subtyped
  `AbstractRegressionTarget` directly now subtypes one of the two branches.
- `ImpliedVolatilityRegression`, `DimensionReductionRegression` and `TargetReturnForecast` with
  `intercept = true` call `factory(tgt, InterceptColumn(k))`. `StepwiseRegression` calls
  `factory(tgt, InterceptColumn(1))`, because it accepts a penalised target under item 6 and puts
  its column of ones first. `CrossSectionalTargetRegression` centres
  `y` and the design and adds no column of ones, so it does not need the call either.
- `ElasticNet` is not basis invariant, so `CrossSectionalTargetRegression` fits it on the design as
  it is. `TargetReturnForecast` refits it at each step, because only `LinearModel` folds.
- The `factory` call with `InterceptColumn` is a direct call on the target. It is not a field
  propagation of ADR 0001.
- `edof` of `Regression` becomes a real count under a penalised target. `StepwiseRegression` now
  writes it into a `Vector{Int}`, so the build derives the element type from the counts. The
  `edof` of `LinearModel` keeps its value `N − p` in each estimator.
- `DimensionReductionRegression` and `CrossSectionalTargetRegression` with an `ElasticNet` change
  their `edof`, because item 9 corrects a count that overstated the spent parameters.
- [#1899](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1899) builds items 6 and 8
  to 11.
- Related: hastie1990 is not in `References.bib`, and no copy is in the sources folder. #1552
  lists it. tibshirani2012, hurvich1998, zou2007 and hastie2009 are in the sources folder, and the
  build adds them to `References.bib` when a docstring cites them.
- Related: ADR 0221 (`ve` standardises), ADR 0222 (the keep rule), ADR 0158 and ADR 0172
  (`KernelTrendPattern`).
