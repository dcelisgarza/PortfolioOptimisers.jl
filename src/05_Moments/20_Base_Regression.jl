"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of every regression estimator, over both the time-series family and the cross-sectional family.

The type is an umbrella over the two families. It declares no interface of its own, because the two families fit different models and implement different verbs. A time-series estimator fits one model per asset over the observations and implements [`regression`](@ref). A cross-sectional estimator fits one model per observation across the assets and implements [`cross_sectional_regression`](@ref). Subtype the child that names the family, never this root. Then a consumer of one family never receives a value of the other.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractTimeSeriesRegressionEstimator`](@ref)
  - [`AbstractCrossSectionalRegressionEstimator`](@ref)
  - [`AbstractRegressionAlgorithm`](@ref)
  - [`AbstractRegressionResult`](@ref)
"""
abstract type AbstractRegressionEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all time-series regression estimator types.

A subtype fits one model per asset over the observations, and returns a [`Regression`](@ref) that carries the loadings of each asset.

# Interfaces

To add a time-series regression estimator, subtype `AbstractTimeSeriesRegressionEstimator`, keep its parameters in the fields of the struct, and add this method:

## Regression

  - `PortfolioOptimisers.regression(re::AbstractTimeSeriesRegressionEstimator, X::MatNum, F::MatNum) -> Regression`: Returns the regression result of the asset returns `X` on the factor returns `F`.

### Arguments

  - `re`: The regression estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:F])

### Returns

  - `reg::Regression`: The regression result, which carries the loadings matrix and the optional intercept.

# Examples

The example below defines a minimal regression estimator.

```jldoctest
julia> struct MyRegressionEstimator <: PortfolioOptimisers.AbstractTimeSeriesRegressionEstimator end

julia> function PortfolioOptimisers.regression(::MyRegressionEstimator,
                                               X::PortfolioOptimisers.MatNum,
                                               F::PortfolioOptimisers.MatNum)
           return PortfolioOptimisers.Regression(; M = F \\ X)
       end

julia> regression(MyRegressionEstimator(), [1.0 2.0; 3.0 4.0; 5.0 6.0],
                  [1.0 0.0; 0.0 1.0; 0.5 0.5])
Regression
       M ┼ 2×2 Matrix{Float64}
       L ┼ 2×2 Matrix{Float64}
       b ┼ nothing
  esigma ┼ nothing
    edof ┼ nothing
    ediv ┼ nothing
     idx ┼ nothing
      ts ┴ nothing
```

# Related

  - [`AbstractRegressionEstimator`](@ref)
  - [`AbstractCrossSectionalRegressionEstimator`](@ref)
  - [`AbstractLoadingsRegressionResult`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`DimensionReductionRegression`](@ref)
"""
abstract type AbstractTimeSeriesRegressionEstimator <: AbstractRegressionEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cross-sectional regression estimator types.

A subtype fits one model per observation across the assets, and returns a [`CrossSectionalRegression`](@ref) that carries the factor returns of each observation.

# Interfaces

To add a cross-sectional regression estimator, subtype `AbstractCrossSectionalRegressionEstimator`, keep its parameters in the fields of the struct, and add this method:

## Cross-sectional regression

  - `PortfolioOptimisers.cross_sectional_regression(cre::AbstractCrossSectionalRegressionEstimator, Z::Arr3Num, X::MatNum, W::MatNum) -> CrossSectionalRegression`: Returns the cross-sectional regression of the asset returns `X` on the exposure tensor `Z`, under the cross-sectional weights `W`.

### Arguments

  - `cre`: The cross-sectional regression estimator.
  - `Z`: The exposure tensor `observations × assets × factors`.
  - `X`: The asset returns matrix `observations × assets`.
  - `W`: The cross-sectional weights matrix `observations × assets`.

### Returns

  - `csr::CrossSectionalRegression`: The cross-sectional regression result. It carries the factor returns, the residuals, the counts and the optional intercept.

# Examples

The example below defines a minimal cross-sectional regression estimator.

```jldoctest
julia> struct MyCrossSectionalRegressionEstimator <:
              PortfolioOptimisers.AbstractCrossSectionalRegressionEstimator end

julia> function PortfolioOptimisers.cross_sectional_regression(::MyCrossSectionalRegressionEstimator,
                                                               Z::PortfolioOptimisers.Arr3Num,
                                                               X::PortfolioOptimisers.MatNum,
                                                               W::PortfolioOptimisers.MatNum)
           f = permutedims(reduce(hcat, Z[t, :, :] \\ X[t, :] for t in axes(X, 1)))
           eps = X - permutedims(reduce(hcat, Z[t, :, :] * f[t, :] for t in axes(X, 1)))
           return PortfolioOptimisers.CrossSectionalRegression(; f = f, eps = eps,
                                                               n = fill(size(X, 2), size(X, 1)))
       end

julia> cross_sectional_regression(MyCrossSectionalRegressionEstimator(),
                                  reshape([1.0, 0.0, 0.5, 0.0, 1.0, 0.5], 1, 3, 2), [1.0 2.0 1.5],
                                  ones(1, 3))
CrossSectionalRegression
     f ┼ 1×2 Matrix{Float64}
   eps ┼ 1×3 Matrix{Float64}
     n ┼ Vector{Int64}: [3]
  edof ┼ nothing
     b ┼ nothing
    h1 ┼ nothing
    dc ┼ nothing
    lv ┴ nothing
```

# Related

  - [`AbstractRegressionEstimator`](@ref)
  - [`AbstractTimeSeriesRegressionEstimator`](@ref)
  - [`AbstractCrossSectionalRegressionResult`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
  - [`CrossSectionalTargetRegression`](@ref)
"""
abstract type AbstractCrossSectionalRegressionEstimator <: AbstractRegressionEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of every regression result, over both the loadings family and the cross-sectional family.

The type is an umbrella over two children, and an asset index means a different thing in each child. A loadings result holds one row of `M` per asset, so [`port_opt_view`](@ref) slices its rows. A cross-sectional result holds one row per observation and one column per asset, so the same index slices its columns. Subtype the child that states what the result carries, never this root.

# Related

  - [`AbstractResult`](@ref)
  - [`AbstractLoadingsRegressionResult`](@ref)
  - [`AbstractCrossSectionalRegressionResult`](@ref)
  - [`AbstractRegressionEstimator`](@ref)
"""
abstract type AbstractRegressionResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all regression result types that carry a loadings matrix.

A subtype carries a loadings matrix `M`, with one row per asset and one column per factor. The root states what a member carries, and not how the library fitted it. Every consumer that re-bases a constraint or decomposes risk in the factor basis reads `M`, so it binds this type and not the umbrella. The shape of the fit does not decide membership. A result fitted per asset over the observations and a result fitted per observation across the assets both belong here when they carry `M`.

# Related

  - [`AbstractRegressionResult`](@ref)
  - [`AbstractCrossSectionalRegressionResult`](@ref)
  - [`AbstractTimeSeriesRegressionEstimator`](@ref)
  - [`Regression`](@ref)
"""
abstract type AbstractLoadingsRegressionResult <: AbstractRegressionResult end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether a loadings result comes from a fit in a re-based Factor Family.

The fit re-bases a Factor Family whose one-hot exposures are collinear with a global factor. It drops one factor of the family, and writes the family in an equivalent basis of full column rank. A result that carries such a re-basis keeps the raw loadings in `M` and the re-based loadings in `L`. Its raw factor axis is then a linear image of a smaller axis, so a factor covariance on that axis is singular. A consumer that inverts or factorises such a covariance reads this trait, and refuses the result before the factorisation fails.

The method of the root returns `false`, so a member that cannot carry a re-basis needs no method. The value `false` is correct for a [`Regression`](@ref), because a fit per asset over the observations re-bases nothing. Only [`CrossSectionalFactorModel`](@ref) has a method of its own.

# Arguments

  - `rr`: The loadings regression result.

# Returns

  - `val::Bool`: `true` when the result carries a family re-basis, and `false` otherwise. Under a re-basis, the raw factor axis is rank deficient.

# Examples

```jldoctest
julia> PortfolioOptimisers.has_family_rebasis(Regression(; M = [1 2; 3 4], b = [1, 2]))
false
```

# Related

  - [`AbstractLoadingsRegressionResult`](@ref)
  - [`Regression`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function has_family_rebasis(::AbstractLoadingsRegressionResult)::Bool
    return false
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the cross-sectional weightings that name a weight source on a fitted loadings block.

A member of this family selects a weight vector that a fitted loadings block already carries. It computes no weights. The library uses such a family of marker types for a fixed choice among stored quantities, as the `class` field of `NormBallUncertaintySet` does. The sibling family [`AbstractCrossSectionalWeightsAlgorithm`](@ref) computes the cross-sectional weights in the fit of the prior. This family names the output of that fit that a reader reads.

Two readers resolve a member to two different shapes, both from the same field of the block. `orthogonality_weights` reads the weights of the latest observation, one entry per asset. An uncertainty set reads them, because it applies to the next decision. `cs_diagnostic_weights` reads the whole observation history, one row per observation, for a diagnostic that scores every observation and not only the latest.

`nothing` in the place of a member names no weight source, so every asset carries the same weight. It reads nothing from the loadings block, so it serves a block that carries neither a weight history nor an idiosyncratic variance. Both readers return `nothing` for it, and their caller takes `nothing` as a vector of ones.

# Interfaces

## `orthogonality_weights`

  - `orthogonality_weights(metric::AbstractOrthogonalityMetric, rr::AbstractLoadingsRegressionResult) -> VecNum`: Returns the weights of the latest observation.

## `cs_diagnostic_weights`

  - `cs_diagnostic_weights(metric::AbstractOrthogonalityMetric, csfm::CrossSectionalFactorModel) -> MatNum`: Returns the whole observation history the member names.

# Related

  - [`BenchmarkWeightMetric`](@ref)
  - [`RegressionWeightMetric`](@ref)
  - [`InverseIdiosyncraticVarianceMetric`](@ref)
  - [`AbstractCrossSectionalWeightsAlgorithm`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
abstract type AbstractOrthogonalityMetric <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Names the benchmark weights as the cross-sectional weight source.

The `bw` field of a block carries the benchmark weight of every asset at every observation. A reader takes the latest row for an uncertainty set, or the whole history for a diagnostic. A block fitted per asset over the observations has no such field, and the reader refuses it.

The metric needs every investable asset in the latest fit. An asset can be in the Investable Mask and outside the estimation universe of the latest cross-section. Its return is finite, so the prior states a moment for it. But the fit drops the asset, so its benchmark weight is zero there.

A zero weight has no inverse, so an uncertainty set on this metric refuses the block. On a point-in-time Asset Panel whose latest cross-section does not cover the whole Investable Mask, use [`InverseIdiosyncraticVarianceMetric`](@ref) or `nothing`, which read no weight history.

# Examples

```jldoctest
julia> BenchmarkWeightMetric()
BenchmarkWeightMetric()
```

# Related

  - [`AbstractOrthogonalityMetric`](@ref)
  - [`RegressionWeightMetric`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
struct BenchmarkWeightMetric <: AbstractOrthogonalityMetric end
"""
$(DocStringExtensions.TYPEDEF)

Names the regression weights as the cross-sectional weight source.

The `rw` field of a block carries the weight that the Cross-Sectional Regression gave every asset at every observation. A reader takes the latest row for an uncertainty set, or the whole history for a diagnostic. A block fitted per asset over the observations has no such field, and the reader refuses it.

The metric needs every investable asset in the latest fit. An asset can be in the Investable Mask and outside the estimation universe of the latest cross-section. Its return is finite, so the prior states a moment for it. But the fit drops the asset, so its regression weight is zero there.

A zero weight has no inverse, so an uncertainty set on this metric refuses the block. On a point-in-time Asset Panel whose latest cross-section does not cover the whole Investable Mask, use [`InverseIdiosyncraticVarianceMetric`](@ref) or `nothing`, which read no weight history.

# Examples

```jldoctest
julia> RegressionWeightMetric()
RegressionWeightMetric()
```

# Related

  - [`AbstractOrthogonalityMetric`](@ref)
  - [`BenchmarkWeightMetric`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
struct RegressionWeightMetric <: AbstractOrthogonalityMetric end
"""
$(DocStringExtensions.TYPEDEF)

Names the inverse of the idiosyncratic variances as the cross-sectional weight source.

It is the default `metric` of [`OrthogonalUncertaintySet`](@ref). An asset whose returns the factors explain well has a small idiosyncratic variance, and so a large weight. `idiosyncratic_variances` reads the variances from the `esigma` field of the loadings block. `orthogonality_weights` refuses a variance that is not finite or not positive.

# Examples

```jldoctest
julia> InverseIdiosyncraticVarianceMetric()
InverseIdiosyncraticVarianceMetric()
```

# Related

  - [`AbstractOrthogonalityMetric`](@ref)
  - [`idiosyncratic_variances`](@ref)
"""
struct InverseIdiosyncraticVarianceMetric <: AbstractOrthogonalityMetric end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Factor Family Basis result types.

A subtype holds the change of basis of a re-based Factor Family. The root lets [`CrossSectionalFactorModel`](@ref) bind its `fcb` field to a type and not to `Any`. This file declares the root, and not the file of its member, because the library loads the block that has the field before the member that fills it.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`has_family_rebasis`](@ref)
"""
abstract type AbstractFactorFamilyBasis <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cross-sectional regression result types.

A subtype holds the output of a regression fitted per observation across the assets. A member carries no loadings matrix, because the exposures are the input of the regression, and an Exposure Estimator makes them.

# Related

  - [`AbstractRegressionResult`](@ref)
  - [`AbstractLoadingsRegressionResult`](@ref)
  - [`AbstractCrossSectionalRegressionEstimator`](@ref)
  - [`CrossSectionalRegression`](@ref)
"""
abstract type AbstractCrossSectionalRegressionResult <: AbstractRegressionResult end
"""
    const RegE_Reg = Union{AbstractLoadingsRegressionResult,
                           AbstractTimeSeriesRegressionEstimator}

Alias for a loadings regression result or a time-series regression estimator.

The alias matches an [`AbstractLoadingsRegressionResult`](@ref), which is a precomputed result that carries the loadings matrix `M`. It also matches an [`AbstractTimeSeriesRegressionEstimator`](@ref), whose verb returns such a result. The factor models and the regression-based risk code dispatch on it. Every consumer of the alias reads `M`, so the alias names the two ways to get `M`, and not the umbrella.

The two arms of the union use different rules, on purpose. A result carries data, so the result arm names what it carries. It admits each loadings result, whatever the shape of its fit. An estimator carries no data, so the estimator arm names the family whose verb, [`regression`](@ref), returns a loadings result. A cross-sectional estimator implements [`cross_sectional_regression`](@ref), whose result carries no loadings, so the alias excludes it.

# Related

  - [`AbstractLoadingsRegressionResult`](@ref)
  - [`AbstractTimeSeriesRegressionEstimator`](@ref)
  - [`AbstractRegressionResult`](@ref)
  - [`AbstractRegressionEstimator`](@ref)
"""
const RegE_Reg = Union{AbstractLoadingsRegressionResult,
                       AbstractTimeSeriesRegressionEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all regression algorithm types.

A subtype names one choice of a regression estimator, such as the stepwise search, the criterion of the search or the target that fits each model. The estimator holds the subtype in a field.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractStepwiseRegressionAlgorithm`](@ref)
  - [`AbstractStepwiseRegressionCriterion`](@ref)
  - [`AbstractRegressionTarget`](@ref)
"""
abstract type AbstractRegressionAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all stepwise regression algorithm types.

A subtype decides whether the search adds factors to the factor set or removes them. An [`AbstractStepwiseRegressionCriterion`](@ref) decides which move is an improvement.

# Related

  - [`AbstractRegressionAlgorithm`](@ref)
  - [`AbstractStepwiseRegressionCriterion`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`ForwardSelection`](@ref)
  - [`BackwardElimination`](@ref)

# References

  - $(ref_dict[:efroymson1960]) Pages 191-203. $(ref_dict[:no_original_source])
"""
abstract type AbstractStepwiseRegressionAlgorithm <: AbstractRegressionAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all stepwise regression criterion types.

A subtype scores a fitted model, and the stepwise search keeps the move that improves the score.

# Related

  - [`AbstractStepwiseRegressionAlgorithm`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`MinMaxValStepwiseRegressionCriterion`](@ref)
  - [`PValue`](@ref)
"""
abstract type AbstractStepwiseRegressionCriterion <: AbstractRegressionAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all regression target types.

A target is the model that a regression estimator fits. The family has two branches. [`AbstractUnpenalisedRegressionTarget`](@ref) holds the targets that fit the coefficients with no penalty, [`LinearModel`](@ref) and [`GeneralisedLinearModel`](@ref). [`AbstractPenalisedRegressionTarget`](@ref) holds the targets that add a penalty on the coefficients, [`ElasticNet`](@ref). Any other model that fits under observation weights and returns its coefficients can be a target, for example a robust M-estimator that the caller writes on top of `StatsAPI`. The library has no such model. The `# Interfaces` section below states the methods that such a model must have.

# Interfaces

To add a regression target, subtype [`AbstractUnpenalisedRegressionTarget`](@ref) or [`AbstractPenalisedRegressionTarget`](@ref), keep its parameters in the fields of the struct, and add the methods below. Each estimator reads only some of them, so a target needs only the methods of the estimators that it runs under. [`StepwiseRegression`](@ref) accepts a penalised target under `:aic`, `:aicc` and `:bic` alone, and its docstring states the reason of each refused pair.

| Estimator                                | Methods it reads                                                                                                                                                                                               |
|:---------------------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`CrossSectionalTargetRegression`](@ref) | `factory`, `StatsAPI.fit`, `StatsAPI.coef`, `is_basis_invariant`, `residual_count`                                                                                                                             |
| [`TargetReturnForecast`](@ref)           | `StatsAPI.fit`, `StatsAPI.predict`, and `factory` with an [`InterceptColumn`](@ref) under `intercept = true`                                                                                                   |
| [`ImpliedVolatilityRegression`](@ref)    | `factory` with an [`InterceptColumn`](@ref), `StatsAPI.fit`, `StatsAPI.predict`                                                                                                                                |
| [`StepwiseRegression`](@ref)             | `factory` with an [`InterceptColumn`](@ref), `StatsAPI.fit`, `StatsAPI.coef`, `residual_count`, and `StatsAPI.coeftable` under [`PValue`](@ref) or the score verb of its criterion under every other criterion |
| [`DimensionReductionRegression`](@ref)   | `factory` with an [`InterceptColumn`](@ref), `StatsAPI.fit`, `StatsAPI.coef`, `residual_count`, `regression_target_weights`                                                                                    |
| [`KernelTrendPattern`](@ref)             | `StatsAPI.fit`, `StatsAPI.predict`                                                                                                                                                                             |

The three time-series estimators carry their target through [`factory`](@ref), so each of them also reads `factory` when it gets observation weights.

## `factory`

  - `factory(tgt::MyRegressionTarget, w::StatsBase.AbstractWeights) -> MyRegressionTarget`: Return a target that carries the observation weights `w` into its fit.

The cross-sectional regression fits every observation under its cross-sectional weights, so it calls this method on each observation. [`factory(::AbstractRegressionTarget, ::ObsWeights)`](@ref) refuses a target with no such method. Without the refusal, the fit of the target ignores the weights. A target that must also accept a [`DynamicAbstractWeights`](@ref) resolves it against the design with [`get_observation_weights`](@ref) at the fit, as [`LinearModel`](@ref) does.

### Arguments

  - `tgt`: The new regression target.
  - `w`: The observation weights, one per row of the design that the target fits.

### Returns

  - `tgt::MyRegressionTarget`: A target whose fit weighs row `i` by `w[i]`.

## `factory` with an `InterceptColumn`

  - `factory(tgt::MyRegressionTarget, ic::InterceptColumn) -> MyRegressionTarget`: Return a target whose fit leaves column `ic.k` of the design out of the penalty.

An estimator that adds a column of ones to the design calls this method before the fit, with the index of that column. [`factory(::AbstractRegressionTarget, ::InterceptColumn)`](@ref) returns the target unchanged, so an unpenalised target needs no method. A penalised target that must not shrink the intercept adds one, as [`ElasticNet`](@ref) does.

### Arguments

  - `tgt`: The new regression target.
  - `ic`: The intercept column, which holds the index of the column of ones.

### Returns

  - `tgt::MyRegressionTarget`: A target whose fit does not penalise column `ic.k`.

## `StatsAPI.fit`

  - `StatsAPI.fit(tgt::MyRegressionTarget, A::MatNum, y::VecNum) -> MyRegressionFit`: Fit the target to the design `A` and the response `y`.

### Arguments

  - `tgt`: The new regression target.
  - `A`: The design matrix, `samples × regressors`. The library adds no intercept column, because the estimator of the caller states the intercept.
  - `y`: The response vector, one entry per row of `A`.

### Returns

  - `m::MyRegressionFit`: The fitted model.

## `StatsAPI.coef`

  - `StatsAPI.coef(m::MyRegressionFit) -> VecNum`: Return the fitted coefficients.

### Arguments

  - `m`: The fitted model that `StatsAPI.fit` returns.

### Returns

  - `b::VecNum`: The coefficients, one per column of the design.

## `StatsAPI.predict`

  - `StatsAPI.predict(m::MyRegressionFit, A::MatNum) -> VecNum`: Return the fitted response on a new design.

### Arguments

  - `m`: The fitted model that `StatsAPI.fit` returns.
  - `A`: The design matrix, `samples × regressors`, with the same columns as the design of the fit.

### Returns

  - `yhat::VecNum`: The predicted response, one entry per row of `A`.

## `StatsAPI.coeftable`

  - `StatsAPI.coeftable(m::MyRegressionFit) -> StatsBase.CoefTable`: Return the coefficient table of the fit, with the p-value of each coefficient in column 4.

[`StepwiseRegression`](@ref) reads it under [`PValue`](@ref) only. It reads column 4 of the table, which is the column of the p-values in `GLM`. It reads one row per column of the design, in the order of the design.

### Arguments

  - `m`: The fitted model that `StatsAPI.fit` returns.

### Returns

  - `ct::StatsBase.CoefTable`: The coefficient table, one row per column of the design. Column 4 holds the p-values.

## The score verbs

  - `StatsAPI.aic(m::MyRegressionFit) -> Number`, and the same signature for `StatsAPI.aicc`, `StatsAPI.bic`, `StatsAPI.r2` and `StatsAPI.adjr2`: Return the score of the fit that the criterion of the same name in [`STEPWISE_REGRESSION_CRITERIA`](@ref) reads.

[`StepwiseRegression`](@ref) reads the verb of its own criterion and no other, so a target implements only the verbs of the criteria that it runs under. If the verb of the criterion has no method for the fit, the first fit raises a `MethodError`. A penalised fit reads its effective degrees of freedom in `StatsAPI.dof`, as [`ElasticNetFit`](@ref) does.

### Arguments

  - `m`: The fitted model that `StatsAPI.fit` returns.

### Returns

  - `s::Number`: The score of the fit. The criterion states whether a smaller or a larger score is better.

## `residual_count`

  - `residual_count(m::MyRegressionFit, A::MatNum) -> Real`: Return the residual degrees of freedom of the fit, which the estimator writes into `edof` of [`Regression`](@ref).

[`residual_count(::Any, ::MatNum)`](@ref) counts one spent parameter for each column of `A`, which is the count of an unpenalised fit, so such a target needs no method. A penalised fit spends fewer, and adds a method that returns its own count, as [`ElasticNetFit`](@ref) does.

### Arguments

  - `m`: The fitted model that `StatsAPI.fit` returns.
  - `A`: The design of the fit.

### Returns

  - `nu::Real`: The residual degrees of freedom of the fit.

## `regression_target_weights`

  - `regression_target_weights(tgt::MyRegressionTarget) -> Option{ObsWeights}`: Return the observation weights the fit of `tgt` applies, or `nothing` when it applies none.

[`DimensionReductionRegression`](@ref) rebuilds the intercept of each asset from the mean of the response. It weights that mean by these weights, so that the intercept agrees with the fit. The constructor of the estimator calls [`regression_target_weights(::AbstractRegressionTarget)`](@ref), which refuses a target with no such method. Without the refusal, the intercept ignores the weights that `factory` stored in the target.

### Arguments

  - `tgt`: The new regression target.

### Returns

  - `w::Option{ObsWeights}`: The weights that `factory(tgt, w)` stored in `tgt`, or `nothing` when `tgt` carries none.

## `is_basis_invariant`

  - `is_basis_invariant(tgt::MyRegressionTarget) -> Bool`: Return whether the fit of `tgt` gives the same coefficients under an invertible change of the columns of the design, mapped back to the original columns.

The carry fold of [`CrossSectionalFactorPrior`](@ref) reads it when a [`BatchChoice`](@ref) moves the dropped member of a Factor Family. That move is an invertible change of the columns. A target that returns `true` lets the fold keep the factor returns of each past observation, so the move runs no regression again. The default returns `false`, and the move then fits every past observation again.

A fit that minimises a loss of the fitted values alone, such as the weighted squared residuals, can return `true`. An iterative fit of such a loss, such as [`GeneralisedLinearModel`](@ref), returns `true` when each of its iterates is invariant too. The fold then equals the batch fit to rounding, unless rounding stops the two bases at different iterates. A penalty on the coefficients, such as a ridge, makes the answer depend on the basis, so such a target keeps the default.

### Arguments

  - `tgt`: The new regression target.

### Returns

  - `flag::Bool`: `true` when the coefficients of the fit do not depend on the basis of the design.

# Related

  - [`AbstractRegressionAlgorithm`](@ref)
  - [`AbstractUnpenalisedRegressionTarget`](@ref)
  - [`AbstractPenalisedRegressionTarget`](@ref)
  - [`LinearModel`](@ref)
  - [`GeneralisedLinearModel`](@ref)
  - [`ElasticNet`](@ref)
  - [`InterceptColumn`](@ref)
  - [`CrossSectionalTargetRegression`](@ref)
  - [`TargetReturnForecast`](@ref)
  - [`ImpliedVolatilityRegression`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`KernelTrendPattern`](@ref)
  - [`factory`](@ref)
  - [`regression_target_weights`](@ref)
  - [`is_basis_invariant`](@ref)
  - [`residual_count`](@ref)
"""
abstract type AbstractRegressionTarget <: AbstractRegressionAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the regression targets that fit their coefficients with no penalty.

[`LinearModel`](@ref) and [`GeneralisedLinearModel`](@ref) subtype it. A target of this branch fits every coefficient by its loss alone, so the inference of the fit counts one degree of freedom for each coefficient. [`StepwiseRegression`](@ref) binds its target to this branch under [`PValue`](@ref), `:r2` and `:adjr2`, because these criteria read p-values or a residual count that no source states for a penalised fit.

# Interfaces

To add an unpenalised regression target, subtype `AbstractUnpenalisedRegressionTarget`, and add the methods that the `# Interfaces` section of [`AbstractRegressionTarget`](@ref) names for the estimators that the target runs under. The target needs no method for an [`InterceptColumn`](@ref), because the method of [`AbstractRegressionTarget`](@ref) returns the target unchanged.

## `StatsAPI.fit`

  - `StatsAPI.fit(tgt::MyUnpenalisedRegressionTarget, A::MatNum, y::VecNum) -> MyRegressionFit`: Fit the target to the design `A` and the response `y`, with no penalty on the coefficients.

[`StepwiseRegression`](@ref) also reads `StatsAPI.coeftable` or the score verb of its criterion on the fit, as the section of [`AbstractRegressionTarget`](@ref) states.

### Arguments

  - `tgt`: The new regression target.
  - `A`: The design matrix, `samples × regressors`.
  - `y`: The response vector, one entry per row of `A`.

### Returns

  - `m::MyRegressionFit`: The fitted model.

# Related

  - [`AbstractRegressionTarget`](@ref)
  - [`AbstractPenalisedRegressionTarget`](@ref)
  - [`LinearModel`](@ref)
  - [`GeneralisedLinearModel`](@ref)
  - [`StepwiseRegression`](@ref)
"""
abstract type AbstractUnpenalisedRegressionTarget <: AbstractRegressionTarget end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the regression targets that add a penalty on their coefficients.

[`ElasticNet`](@ref) subtypes it. A penalty shrinks the coefficients toward zero, so the fit has fewer degrees of freedom than coefficients, and its p-values are not valid. A penalty also depends on the units of each column, so a target of this branch is not invariant to the basis of the design. An estimator that adds a column of ones names it with [`InterceptColumn`](@ref), so that the target does not shrink the intercept.

# Interfaces

To add a penalised regression target, subtype `AbstractPenalisedRegressionTarget`, and add the methods that the `# Interfaces` section of [`AbstractRegressionTarget`](@ref) names for the estimators that the target runs under. Add the method below too, so that the penalty leaves the intercept alone. Add `residual_count` for the fit too, because its fallback counts one spent parameter for each column.

## `factory`

  - `factory(tgt::MyPenalisedRegressionTarget, ic::InterceptColumn) -> MyPenalisedRegressionTarget`: Return a target whose fit leaves column `ic.k` of the design out of the penalty.

Without this method, the method of [`AbstractRegressionTarget`](@ref) returns the target unchanged, and the fit shrinks the intercept toward zero.

### Arguments

  - `tgt`: The new regression target.
  - `ic`: The intercept column, which holds the index of the column of ones.

### Returns

  - `tgt::MyPenalisedRegressionTarget`: A target whose fit does not penalise column `ic.k`.

# Related

  - [`AbstractRegressionTarget`](@ref)
  - [`AbstractUnpenalisedRegressionTarget`](@ref)
  - [`ElasticNet`](@ref)
  - [`InterceptColumn`](@ref)
  - [`is_basis_invariant`](@ref)
"""
abstract type AbstractPenalisedRegressionTarget <: AbstractRegressionTarget end
"""
$(DocStringExtensions.TYPEDEF)

Names the column of ones that a regression estimator adds to its design.

An estimator that adds an intercept to its design calls `factory(tgt, InterceptColumn(k))` before the fit, with `k` the index of the column of ones. A penalised target, such as [`ElasticNet`](@ref), then leaves that coefficient out of the penalty. An unpenalised target ignores the call. The estimator names the column, because only the estimator knows where it put the column: [`ImpliedVolatilityRegression`](@ref) and [`DimensionReductionRegression`](@ref) put it first, and [`TargetReturnForecast`](@ref) puts it last. The target does not detect a constant column, because [`KernelTrendPattern`](@ref) has a column of ones whose coefficient the penalty must shrink.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    InterceptColumn(k::Integer) -> InterceptColumn

## Validation

  - `k >= 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> InterceptColumn(1)
InterceptColumn
  k ┴ Int64: 1
```

# Related

  - [`AbstractPenalisedRegressionTarget`](@ref)
  - [`ElasticNet`](@ref)
  - [`factory(::AbstractRegressionTarget, ::InterceptColumn)`](@ref)
"""
@concrete struct InterceptColumn <: AbstractAlgorithm
    """
    The index of the column of ones in the design.
    """
    k
    function InterceptColumn(k::Integer)
        @argcheck(k >= 1, DomainError(k, "k must be at least 1"))
        return new{typeof(k)}(k)
    end
end
"""
    factory(tgt::AbstractRegressionTarget, ic::InterceptColumn) -> AbstractRegressionTarget

Return a regression target unchanged when an estimator names its column of ones.

An unpenalised target fits the intercept as it fits every other coefficient, so the index of the column changes nothing. A penalised target that must leave the intercept out of the penalty has a method of its own, as [`ElasticNet`](@ref) has.

# Arguments

  - `tgt`: The regression target.
  - `ic`: The intercept column.

# Returns

  - `tgt::AbstractRegressionTarget`: The input target, unchanged.

# Related

  - [`InterceptColumn`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`factory`](@ref)
"""
function factory(tgt::AbstractRegressionTarget, ::InterceptColumn)
    return tgt
end
"""
    factory(tgt::AbstractRegressionTarget, w::ObsWeights)

Refuse a regression target that has no method to carry observation weights into its fit.

Without this method, a target with no `factory(tgt, w)` method of its own goes to the generic [`factory`](@ref). The generic method returns the target unchanged, and the fit of the target then ignores the weights. A target that the caller fits without weights, as [`TargetReturnForecast`](@ref) fits it, needs no weight method, because only a call that carries weights reaches this method.

# Arguments

  - `tgt`: The regression target, which has no weight method of its own.
  - $(arg_dict[:ow])

# Validation

  - The call always throws an `ArgumentError` that names the method `tgt` lacks. The `# Interfaces` section of [`AbstractRegressionTarget`](@ref) states that method.

# Related

  - [`AbstractRegressionTarget`](@ref)
  - [`CrossSectionalTargetRegression`](@ref)
  - [`factory`](@ref)
"""
function factory(tgt::AbstractRegressionTarget, w::ObsWeights)
    return throw(ArgumentError("$(nameof(typeof(tgt))) has no method factory(::$(nameof(typeof(tgt))), ::$(nameof(typeof(w)))), so its fit cannot carry the observation weights. Define PortfolioOptimisers.factory(tgt::$(nameof(typeof(tgt))), w::StatsBase.AbstractWeights) to return a target whose fit weighs each row by w, as the # Interfaces section of AbstractRegressionTarget states"))
end
"""
$(DocStringExtensions.TYPEDEF)

Fits each response by ordinary least squares through `GLM.LinearModel`.

The target passes its `kwargs` field to `GLM`, so every option that `GLM` accepts reaches the fit, and the observation weights too. The target changes one entry, `weights`. [`glm_weights`](@ref) scales it to the count that `GLM` reads, and the scale does not change the coefficients. `LinearModel()` is the default target of every estimator in the library that takes a regression target.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LinearModel(;
        kwargs::NamedTuple = (;)
    ) -> LinearModel

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> LinearModel()
LinearModel
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractRegressionTarget`](@ref)
  - [`GeneralisedLinearModel`](@ref)
  - [`StatsAPI.fit(::LinearModel, ::MatNum, ::VecNum)`](@ref)
"""
@concrete struct LinearModel <: AbstractUnpenalisedRegressionTarget
    """
    Keyword arguments passed to `fit(GLM.LinearModel, X, y; kwargs...)`.
    """
    kwargs
    function LinearModel(kwargs::NamedTuple)
        return new{typeof(kwargs)}(kwargs)
    end
end
function LinearModel(; kwargs::NamedTuple = (;))::LinearModel
    return LinearModel(kwargs)
end
"""
    factory(re::LinearModel, w::ObsWeights) -> LinearModel

Return a new [`LinearModel`](@ref) regression target with observation weights `w` added to the keyword arguments.

# Algorithm

 1. Merge `w` into `re.kwargs` under the key `weights`. The merge replaces an entry that is already there. The result is the keyword arguments of the new target.
 2. Build a new [`LinearModel`](@ref) from them.

# Arguments

  - `re`: The regression target.
  - $(arg_dict[:ow])

# Returns

  - `re::LinearModel`: A new regression target that holds the weights in `kwargs`.

# Related

  - [`LinearModel`](@ref)
  - [`factory`](@ref)
"""
function factory(re::LinearModel, w::ObsWeights)::LinearModel
    return LinearModel(; kwargs = (; re.kwargs..., weights = w))
end
"""
    glm_weights(w::StatsBase.AbstractWeights) -> StatsBase.FrequencyWeights
    glm_weights(w::Nothing) -> Nothing

Return the frequency weights that `GLM` reads as the observation weights `w`, or `nothing` for a fit with no weights.

`GLM` reads every weight vector as frequency weights, so the number of observations of a weighted fit is ``\\sum_t w_t``. `StatsBase.FrequencyWeights` state a count, so the method returns them unchanged. Every other kind of weights states the relative weight of each observation and no count. For example, `GLM` reads probability weights that sum to one as one observation. The fit then has a residual degree of freedom at or below zero, and a p-value of one for every coefficient. To prevent this, the method scales the other kinds to sum to their Kish effective sample size.

The scale does not change the coefficients, because a weighted fit does not change under a constant scale of its weights. The Kish count is the correct count for observations that are drawn alike and then reweighted. The scenario probabilities and the decay weights of the library are such weights. The count is ``T`` under uniform weights, so the fit equals the unweighted fit. It is ``k`` under equal weights on ``k`` of the rows, so the fit equals the fit of those rows alone.

`docs/adr/0203-a-weighted-regression-reads-the-kish-count-of-its-weights.md` records the choice and the null test that measured it. A caller who wants another count states it with `StatsBase.fweights`.

# Mathematical definition

```math
\\begin{align}
T_{e} &= \\frac{\\left(\\sum_{t} w_{t}\\right)^2}{\\sum_{t} w_{t}^2}\\,,\\\\
\\tilde{w}_{t} &= w_{t} \\frac{T_{e}}{\\sum_{s} w_{s}} = w_{t} \\frac{\\sum_{s} w_{s}}{\\sum_{s} w_{s}^2}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_t_obs])
  - $(math_dict[:cal_T_e])
  - ``\\tilde{w}_{t}``: Frequency weight of observation ``t``. The frequency weights sum to ``T_{e}``.

# Arguments

  - `w`: The resolved observation weights, one per row of the design, or `nothing`.

# Returns

  - `w::Option{StatsBase.FrequencyWeights}`: `w` itself when it is `StatsBase.FrequencyWeights` or `nothing`, otherwise ``\\tilde{w}``.

# Related

  - [`glm_fit_kwargs`](@ref)
  - [`StatsAPI.fit(::LinearModel, ::MatNum, ::VecNum)`](@ref)
  - [`StatsAPI.fit(::GeneralisedLinearModel, ::MatNum, ::VecNum)`](@ref)
"""
function glm_weights(w::StatsBase.FrequencyWeights)
    return w
end
function glm_weights(w::StatsBase.AbstractWeights)
    return StatsBase.fweights(w .* (sum(w) / sum(abs2, w)))
end
function glm_weights(::Nothing)
    return nothing
end
"""
    glm_fit_kwargs(kwargs::NamedTuple, X::MatNum) -> NamedTuple

Return the keyword arguments that a library regression target passes to `GLM`.

[`LinearModel`](@ref) and [`GeneralisedLinearModel`](@ref) keep their observation weights under the `weights` key of their `kwargs`, where [`factory`](@ref) puts them. Both targets resolve and scale the entry here, before the fit. A [`DynamicAbstractWeights`](@ref) holds no values before the method resolves it against the design, and `GLM` reads every weight vector as a count.

# Algorithm

 1. Read the `weights` entry of `kwargs`. When it is not an [`ObsWeights`](@ref), return `kwargs` unchanged.
 2. Resolve the entry against `X` with [`get_observation_weights`](@ref), and scale it with [`glm_weights`](@ref).
 3. Return `kwargs` with the scaled weights under the `weights` key.

# Arguments

  - `kwargs`: The `kwargs` field of the regression target.
  - `X`: The design matrix `observations × factors`.

# Returns

  - `kwargs::NamedTuple`: The keyword arguments of the call to `GLM`.

# Related

  - [`glm_weights`](@ref)
  - [`StatsAPI.fit(::LinearModel, ::MatNum, ::VecNum)`](@ref)
  - [`StatsAPI.fit(::GeneralisedLinearModel, ::MatNum, ::VecNum)`](@ref)
"""
function glm_fit_kwargs(kwargs::NamedTuple, X::MatNum)
    w = get(kwargs, :weights, nothing)
    if !isa(w, ObsWeights)
        return kwargs
    end
    return (; kwargs..., weights = glm_weights(get_observation_weights(w, X)))
end
"""
    StatsAPI.fit(tgt::LinearModel, X::MatNum, y::VecNum)

Fit an ordinary least-squares model with a [`LinearModel`](@ref) target.

# Algorithm

 1. Resolve and scale the `weights` entry of `tgt.kwargs` with [`glm_fit_kwargs`](@ref). Call the result `kwargs`. Weights that are not `StatsBase.FrequencyWeights` then sum to their Kish effective sample size, so the inference of the fit reads that count of observations.
 2. Call `StatsAPI.fit(GLM.LinearModel, X, y; kwargs...)`, and return the fitted model.

# Arguments

  - `tgt`: The regression target, which holds the options of the model.
  - `X`: The design matrix `observations × factors`.
  - `y`: The response vector.

# Returns

  - `model::GLM.LinearModel`: The fitted linear model of the `GLM` package.

# Related

  - [`LinearModel`](@ref)
  - [`GLM.LinearModel`](https://juliastats.org/GLM.jl/stable/api/#GLM.LinearModel)
"""
function StatsAPI.fit(tgt::LinearModel, X::MatNum, y::VecNum)
    return StatsAPI.fit(GLM.LinearModel, X, y; glm_fit_kwargs(tgt.kwargs, X)...)
end
"""
    PSEUDO_R2_VARIANTS

Tuple of the pseudo-``R^2`` variants `StatsAPI.r2` accepts for a fitted [`GeneralisedLinearModel`](@ref).

The members are `:McFadden`, `:CoxSnell`, `:Nagelkerke` and `:devianceratio`. The constructor of [`GeneralisedLinearModel`](@ref) checks its `variant` field against this tuple. A generalised linear model has no classical ``R^2``, so each member scores the fitted model against the intercept-only model of the same family instead.

# Mathematical definition

```math
\\begin{align}
R^2_{\\mathrm{McF}} &= 1 - \\frac{\\ln\\hat{L}}{\\ln\\hat{L}_{0}}\\,,\\\\
R^2_{\\mathrm{CS}} &= 1 - \\left(\\frac{\\hat{L}_{0}}{\\hat{L}}\\right)^{2/T}\\,,\\\\
R^2_{\\mathrm{N}} &= \\frac{R^2_{\\mathrm{CS}}}{1 - \\hat{L}_{0}^{\\,2/T}}\\,,\\\\
R^2_{\\mathrm{dev}} &= 1 - \\frac{D}{D_{0}}\\,.
\\end{align}
```

Where:

  - ``R^2_{\\mathrm{McF}}``: `:McFadden`.
  - ``R^2_{\\mathrm{CS}}``: `:CoxSnell`.
  - ``R^2_{\\mathrm{N}}``: `:Nagelkerke`.
  - ``R^2_{\\mathrm{dev}}``: `:devianceratio`.
  - ``\\hat{L}``: Maximum likelihood of the fitted model.
  - ``\\hat{L}_{0}``: Maximum likelihood of the intercept-only model of the same family.
  - ``D``: Deviance of the fitted model.
  - ``D_{0}``: Deviance of the intercept-only model of the same family.
  - $(math_dict[:T])

The Normal family is the default of [`GeneralisedLinearModel`](@ref), and three facts hold for it. Its deviance is the residual sum of squares, so ``R^2_{\\mathrm{dev}}`` is the classical ``R^2`` of the same fit. Its maximum likelihood carries the fitted dispersion, so ``\\left(\\hat{L}_{0}/\\hat{L}\\right)^{2/T} = D/D_{0}`` and ``R^2_{\\mathrm{CS}}`` equals ``R^2_{\\mathrm{dev}}``.

Its likelihood is a density and not a probability, so ``\\hat{L}_{0}`` can be above or below one. The two forms on the log-likelihood, ``R^2_{\\mathrm{McF}}`` and ``R^2_{\\mathrm{N}}``, can then leave ``[0, 1]`` in either direction. A change of the scale of the response alone moves both from above one to below zero, and ``R^2_{\\mathrm{dev}}`` does not change. Only ``R^2_{\\mathrm{dev}}`` is continuous with the [`LinearModel`](@ref) path, so [`default_regression_criterion_variant`](@ref) returns `:devianceratio`.

# Related

  - [`ADJUSTED_PSEUDO_R2_VARIANTS`](@ref)
  - [`GeneralisedLinearModel`](@ref)
  - [`default_regression_criterion_variant`](@ref)
  - [`regression_criterion_func`](@ref)

# References

  - $(ref_dict[:mcfadden1974]) Equation 30, page 121.
  - $(ref_dict[:coxsnell1989]) Pages 208-209. $(ref_dict[:no_original_source])
  - $(ref_dict[:nagelkerke1991]) Pages 691-692. $(ref_dict[:no_original_source])
  - $(ref_dict[:nelder1972]) Pages 370-384. $(ref_dict[:no_original_source])
"""
const PSEUDO_R2_VARIANTS = (:McFadden, :CoxSnell, :Nagelkerke, :devianceratio)
"""
    ADJUSTED_PSEUDO_R2_VARIANTS

Tuple of the pseudo-``R^2`` variants `StatsAPI.adjr2` accepts for a fitted [`GeneralisedLinearModel`](@ref).

The members are `:McFadden` and `:devianceratio`, a strict subset of [`PSEUDO_R2_VARIANTS`](@ref). [`GeneralisedLinearModel`](@ref) cannot check its `variant` against this tuple, because it does not know which criterion reads the variant. [`StepwiseRegression`](@ref) checks it, because it is the first type that holds the criterion and the target together. `StatsAPI.adjr2` raises an `ArgumentError` on each variant that this tuple omits.

# Mathematical definition

Each member discounts its unadjusted form of [`PSEUDO_R2_VARIANTS`](@ref) by the number of parameters of the model. [mcfadden1974](@cite) states the unadjusted form alone, and the discount of each member is the convention of the `GLM` package. This docstring cites no original source for the discount.

```math
\\begin{align}
\\bar{R}^2_{\\mathrm{McF}} &= 1 - \\frac{\\ln\\hat{L} - k}{\\ln\\hat{L}_{0}}\\,,\\\\
\\bar{R}^2_{\\mathrm{dev}} &= 1 - \\frac{D\\,(T - 1)}{D_{0}\\,(T - k)}\\,.
\\end{align}
```

Where:

  - ``\\bar{R}^2_{\\mathrm{McF}}``: `:McFadden`.
  - ``\\bar{R}^2_{\\mathrm{dev}}``: `:devianceratio`.
  - ``\\hat{L}``: Maximum likelihood of the fitted model.
  - ``\\hat{L}_{0}``: Maximum likelihood of the intercept-only model of the same family.
  - ``D``: Deviance of the fitted model.
  - ``D_{0}``: Deviance of the intercept-only model of the same family.
  - ``k``: Number of estimated parameters, which is `StatsAPI.dof` of the fitted model. It counts the regression coefficients, the intercept, and the dispersion when the family estimates one. The Normal family estimates it, and the binomial and Poisson families do not.
  - $(math_dict[:T])

This ``k`` is the count that `:aic`, `:aicc` and `:bic` read. It is not the predictor count that `:adjr2` reads on a fitted [`LinearModel`](@ref). [`STEPWISE_REGRESSION_CRITERIA`](@ref) states both counts.

# Related

  - [`PSEUDO_R2_VARIANTS`](@ref)
  - [`GeneralisedLinearModel`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`StepwiseRegression`](@ref)

# References

  - $(ref_dict[:mcfadden1974]) Equation 30, page 121.
  - $(ref_dict[:nelder1972]) Pages 370-384. $(ref_dict[:no_original_source])
"""
const ADJUSTED_PSEUDO_R2_VARIANTS = (:McFadden, :devianceratio)
"""
$(DocStringExtensions.TYPEDEF)

Fits each response by a generalised linear model through `GLM.GeneralizedLinearModel`.

The `args` field carries the response distribution and, as an option, the link function. The `kwargs` field carries the other `GLM` options. The default `args = (Normal(),)` with the canonical identity link gives ordinary least squares.

`GLM` defines ``R^2`` for a fitted [`LinearModel`](@ref) only, and `variant` names the pseudo-``R^2`` that the `:r2` and `:adjr2` members of [`STEPWISE_REGRESSION_CRITERIA`](@ref) read in its place. A `nothing` `variant` takes the default of the criterion, which [`default_regression_criterion_variant`](@ref) states. A minimisation criterion reads no variant, so it ignores the field.

The fit is iterative, and `kwargs` sets when it stops. `GLM` stops when the change in the deviance falls below `max(rtol * deviance, atol)`, with `rtol = 1e-6`, `atol = 1e-6` and `maxiter = 30` by default. A tighter tolerance, such as `kwargs = (; rtol = 1e-12, atol = 1e-12)`, brings the fit nearer to the maximum-likelihood answer, and needs more iterations. A tolerance below the rounding of the deviance makes `GLM` throw.

[`is_basis_invariant`](@ref) returns `true`, so the carry fold of [`CrossSectionalFactorPrior`](@ref) folds a move of the dropped member of a Factor Family. The fold keeps the coefficients of the old basis, and the batch fit iterates in the new basis. The two take the same iterates and agree to rounding. When the tolerance is near the rounding of the deviance, rounding can stop the two one iteration apart at an observation. The two then differ by that iteration.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GeneralisedLinearModel(;
        args::Tuple = (Normal(),),
        kwargs::NamedTuple = (;),
        variant::Option{Symbol} = nothing
    ) -> GeneralisedLinearModel

Keywords correspond to the struct's fields.

## Validation

  - If provided, `variant in PSEUDO_R2_VARIANTS`, the wider of the two variant tuples. `StatsAPI.adjr2` accepts [`ADJUSTED_PSEUDO_R2_VARIANTS`](@ref) alone, and [`StepwiseRegression`](@ref) rejects the other two variants when its criterion is `:adjr2`.

# Examples

```jldoctest
julia> GeneralisedLinearModel()
GeneralisedLinearModel
     args ┼ Tuple{Distributions.Normal{Float64}}: (Distributions.Normal{Float64}(μ=0.0, σ=1.0),)
   kwargs ┼ @NamedTuple{}: NamedTuple()
  variant ┴ nothing
```

# Related

  - [`AbstractRegressionTarget`](@ref)
  - [`LinearModel`](@ref)
  - [`PSEUDO_R2_VARIANTS`](@ref)
  - [`ADJUSTED_PSEUDO_R2_VARIANTS`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`default_regression_criterion_variant`](@ref)
  - [`regression_criterion_func`](@ref)
  - [`StatsAPI.fit(::GeneralisedLinearModel, ::MatNum, ::VecNum)`](@ref)
  - [`is_basis_invariant`](@ref)

# References

  - $(ref_dict[:nelder1972]) Pages 370-384. $(ref_dict[:no_original_source])
"""
@concrete struct GeneralisedLinearModel <: AbstractUnpenalisedRegressionTarget
    """
    Positional arguments passed to `fit(GLM.GeneralizedLinearModel, X, y, args...; kwargs...)`.
    """
    args
    """
    Keyword arguments passed to `fit(GLM.GeneralizedLinearModel, X, y, args...; kwargs...)`.
    """
    kwargs
    """
    $(field_dict[:r2variant])
    """
    variant
    function GeneralisedLinearModel(args::Tuple, kwargs::NamedTuple,
                                    variant::Option{Symbol})
        if !isnothing(variant)
            @argcheck(variant in PSEUDO_R2_VARIANTS,
                      "variant must be one of $PSEUDO_R2_VARIANTS. Got\nvariant => $variant")
        end
        return new{typeof(args), typeof(kwargs), typeof(variant)}(args, kwargs, variant)
    end
end
function GeneralisedLinearModel(; args::Tuple = (Distributions.Normal(),),
                                kwargs::NamedTuple = (;),
                                variant::Option{Symbol} = nothing)::GeneralisedLinearModel
    return GeneralisedLinearModel(args, kwargs, variant)
end
"""
    factory(re::GeneralisedLinearModel, w::ObsWeights) -> GeneralisedLinearModel

Return a new [`GeneralisedLinearModel`](@ref) regression target with observation weights `w` added to the keyword arguments.

# Algorithm

 1. Merge `w` into `re.kwargs` under the key `weights`. The merge replaces an entry that is already there. The result is the keyword arguments of the new target.
 2. Build a new [`GeneralisedLinearModel`](@ref) from them, with `re.args` and `re.variant` unchanged.

# Arguments

  - `re`: The regression target.
  - $(arg_dict[:ow])

# Returns

  - `re::GeneralisedLinearModel`: A new regression target that holds the weights in `kwargs`.

# Related

  - [`GeneralisedLinearModel`](@ref)
  - [`factory`](@ref)
"""
function factory(re::GeneralisedLinearModel, w::ObsWeights)::GeneralisedLinearModel
    return GeneralisedLinearModel(; args = re.args, kwargs = (; re.kwargs..., weights = w),
                                  variant = re.variant)
end
"""
    StatsAPI.fit(tgt::GeneralisedLinearModel, X::MatNum, y::VecNum)

Fit a generalised linear model with a [`GeneralisedLinearModel`](@ref) target.

# Algorithm

 1. Resolve and scale the `weights` entry of `tgt.kwargs` with [`glm_fit_kwargs`](@ref). Call the result `kwargs`. Weights that are not `StatsBase.FrequencyWeights` then sum to their Kish effective sample size, so the inference of the fit reads that count of observations.
 2. Call `StatsAPI.fit(GLM.GeneralizedLinearModel, X, y, tgt.args...; kwargs...)`, and return the fitted model.

# Arguments

  - `tgt`: The regression target, which holds the options of the model.
  - `X`: The design matrix `observations × factors`.
  - `y`: The response vector.

# Returns

  - `model::GLM.GeneralizedLinearModel`: The fitted generalised linear model of the `GLM` package.

# Related

  - [`GeneralisedLinearModel`](@ref)
  - [`GLM.GeneralizedLinearModel`](https://juliastats.org/GLM.jl/stable/examples/#Probit-regression)
"""
function StatsAPI.fit(tgt::GeneralisedLinearModel, X::MatNum, y::VecNum)
    return StatsAPI.fit(GLM.GeneralizedLinearModel, X, y, tgt.args...;
                        glm_fit_kwargs(tgt.kwargs, X)...)
end
"""
    regression_target_weights(tgt::AbstractRegressionTarget)

Refuse a regression target that does not state the observation weights its fit applies.

[`DimensionReductionRegression`](@ref) weights the mean of the response by the weights of its target, so it must read them. A target states them with its own method, as the `# Interfaces` section of [`AbstractRegressionTarget`](@ref) states. Only a target with no such method reaches this one. The constructor of the estimator calls it, so the refusal comes before a fit.

# Arguments

  - `tgt`: The regression target, which has no `regression_target_weights` method of its own.

# Validation

  - The call always throws an `ArgumentError` that names the method `tgt` lacks.

# Related

  - [`AbstractRegressionTarget`](@ref)
  - [`regression_target_weights(::Union{LinearModel, GeneralisedLinearModel})`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`factory(::AbstractRegressionTarget, ::ObsWeights)`](@ref)
"""
function regression_target_weights(tgt::AbstractRegressionTarget)
    return throw(ArgumentError("$(nameof(typeof(tgt))) has no method regression_target_weights(::$(nameof(typeof(tgt)))), so the estimator cannot read the observation weights its fit applies. Define PortfolioOptimisers.regression_target_weights(tgt::$(nameof(typeof(tgt)))) to return those weights, or nothing when the fit applies none, as the # Interfaces section of AbstractRegressionTarget states"))
end
"""
    regression_target_weights(tgt::Union{LinearModel, GeneralisedLinearModel})

Return the observation weights that a library target carries into its fit, or `nothing`.

Both targets keep the weights under the `weights` key of `tgt.kwargs`, where [`factory`](@ref) puts them. The method returns the entry as it is, so it does not resolve a [`DynamicAbstractWeights`](@ref) against a design.

# Arguments

  - `tgt`: A [`LinearModel`](@ref) or a [`GeneralisedLinearModel`](@ref).

# Returns

  - `w::Option{ObsWeights}`: `tgt.kwargs.weights`, or `nothing` when `tgt.kwargs` has no `weights` key.

# Related

  - [`AbstractRegressionTarget`](@ref)
  - [`regression_target_weights(::AbstractRegressionTarget)`](@ref)
  - [`factory(::LinearModel, ::ObsWeights)`](@ref)
  - [`factory(::GeneralisedLinearModel, ::ObsWeights)`](@ref)
"""
function regression_target_weights(tgt::Union{LinearModel, GeneralisedLinearModel})
    return get(tgt.kwargs, :weights, nothing)
end
"""
    is_basis_invariant(tgt::Union{LinearModel, GeneralisedLinearModel}) -> Bool
    is_basis_invariant(tgt::AbstractRegressionTarget) -> Bool

Return whether the fit of a regression target gives the same coefficients under an invertible change of the columns of the design, mapped back to the original columns.

Take the columns of the design times an invertible matrix as a new design. A target is invariant to the basis when its coefficients on the new design, times that matrix, are its coefficients on the old design. Then the fitted values do not change either.

A weighted least-squares fit projects the response onto the span of the design, and the change keeps the span. So [`LinearModel`](@ref) returns `true`, on each design of full rank. A maximum-likelihood fit reads the design through the linear predictor alone, and the change keeps the set of linear predictors, so [`GeneralisedLinearModel`](@ref) returns `true` too.

The fit of [`GeneralisedLinearModel`](@ref) is iterative, but each iteration is a weighted least-squares fit, and the start reads the response alone. The two bases then take the same iterates and stop at the same one, and they agree to rounding. `GLM` stops when the change in the deviance falls below `max(rtol * deviance, atol)`, with `rtol = 1e-6`, `atol = 1e-6` and `maxiter = 30` by default. The `kwargs` of the target set the three values. A tolerance near the rounding of the deviance lets rounding stop the two bases one iteration apart, and the two answers then differ by that iteration. A tighter tolerance brings each fit nearer to the maximum-likelihood answer, and it does not bring the two bases nearer to each other.

On a rank-deficient design, `GLM` drops a collinear column by a pivot, so the coefficients depend on the basis. A caller that reads this method, such as `cross_sectional_target_coef`, fits such a design in a pivoted basis of its own. Every other target returns `false`, because this method cannot know whether such a target penalises its coefficients.

The carry fold of [`CrossSectionalFactorPrior`](@ref) reads the method on the target of a [`CrossSectionalTargetRegression`](@ref), when a [`BatchChoice`](@ref) moves the dropped member of a Factor Family. A target of the caller states its answer with its own method, as the `# Interfaces` section of [`AbstractRegressionTarget`](@ref) states.

# Arguments

  - `tgt`: The regression target.

# Returns

  - `flag::Bool`: `true` when the coefficients of the fit do not depend on the basis of a design of full rank.

# Examples

```jldoctest
julia> PortfolioOptimisers.is_basis_invariant(LinearModel())
true

julia> PortfolioOptimisers.is_basis_invariant(GeneralisedLinearModel())
true

julia> struct RidgeTarget <: PortfolioOptimisers.AbstractPenalisedRegressionTarget end

julia> PortfolioOptimisers.is_basis_invariant(RidgeTarget())
false
```

# Related

  - [`AbstractRegressionTarget`](@ref)
  - [`LinearModel`](@ref)
  - [`GeneralisedLinearModel`](@ref)
  - [`CrossSectionalTargetRegression`](@ref)
"""
function is_basis_invariant(::Union{LinearModel, GeneralisedLinearModel})::Bool
    return true
end
function is_basis_invariant(::AbstractRegressionTarget)::Bool
    return false
end
"""
    residual_count(m, A::MatNum) -> Integer

Return the residual degrees of freedom of a fitted regression target that states no count of its own, the number of rows of the design less its number of columns.

A fit with no penalty on a design of full rank spends one parameter for each column, so the count is the count of a projection. [`StepwiseRegression`](@ref) and [`DimensionReductionRegression`](@ref) read it, and write it into `edof` of [`Regression`](@ref). [`CrossSectionalTargetRegression`](@ref) reads it for each observation, on the columns that its solve algorithm keeps, and writes it into `edof` of [`CrossSectionalRegression`](@ref). The method does not read `GLM.dof_residual`. Under weights, `GLM` counts the observations as the sum of the frequency weights, and the count of the library counts the rows. A penalised fit spends fewer parameters than it has columns, so it states its own count, as [`residual_count(::ElasticNetFit, ::MatNum)`](@ref) does.

# Arguments

  - `m`: The fitted model, which the method does not read.
  - `A`: The design of the fit, `observations × columns`.

# Returns

  - `nu::Integer`: `size(A, 1) - size(A, 2)`.

# Related

  - [`AbstractRegressionTarget`](@ref)
  - [`ElasticNetFit`](@ref)
  - [`Regression`](@ref)
"""
function residual_count(::Any, A::MatNum)
    return size(A, 1) - size(A, 2)
end
"""
    residual_count(m::GLM.LinearModel, A::MatNum) -> Integer
    residual_count(m::GLM.GeneralizedLinearModel, A::MatNum) -> Integer

Return the residual degrees of freedom of a `GLM` fit, the number of rows of the design less the rank that the fit identified.

`GLM` fits a rank-deficient design through a pivoted Cholesky factorisation, and gives each column that its pivot drops a coefficient of zero. The fit then spends its rank, not its number of columns, so the count of [`residual_count(::Any, ::MatNum)`](@ref) is too low. `StatsAPI.dof` of a `GLM` fit counts the rank and the dispersion parameter of the family, which a `LinearModel` always has, so the method subtracts the dispersion parameter. On a design of full rank the count is the count of the fallback. The method reads no weights, for the reason that the fallback states. [`CrossSectionalTargetRegression`](@ref) under [`UncheckedSolve`](@ref) gives `GLM` a rank-deficient design, and so can [`StepwiseRegression`](@ref) on a dependent candidate set.

# Arguments

  - `m`: The fitted model of [`LinearModel`](@ref) or of [`GeneralisedLinearModel`](@ref).
  - `A`: The design of the fit, `observations × columns`.

# Returns

  - `nu::Integer`: `size(A, 1)` less the rank of the fit.

# Related

  - [`residual_count`](@ref)
  - [`LinearModel`](@ref)
  - [`GeneralisedLinearModel`](@ref)
"""
function residual_count(m::GLM.LinearModel, A::MatNum)
    return size(A, 1) - (StatsAPI.dof(m) - 1)
end
function residual_count(m::GLM.GeneralizedLinearModel, A::MatNum)
    return size(A, 1) - (StatsAPI.dof(m) - GLM.dispersion_parameter(m.rr.d))
end
"""
    MIN_VAL_STEPWISE_REGRESSION_CRITERIA

Tuple of the symbols that name a stepwise regression criterion for which a lower value is better.

The members are `:aic`, `:aicc` and `:bic`. The library builds [`MinValStepwiseRegressionCriterion`](@ref) from this tuple, and [`STEPWISE_REGRESSION_CRITERIA`](@ref) states what each symbol computes.

# Related

  - [`MAX_VAL_STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
"""
const MIN_VAL_STEPWISE_REGRESSION_CRITERIA = (:aic, :aicc, :bic)
"""
    MAX_VAL_STEPWISE_REGRESSION_CRITERIA

Tuple of the symbols that name a stepwise regression criterion for which a higher value is better.

The members are `:r2` and `:adjr2`. The library builds [`MaxValStepwiseRegressionCriterion`](@ref) from this tuple, and [`STEPWISE_REGRESSION_CRITERIA`](@ref) states what each symbol computes.

# Related

  - [`MIN_VAL_STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
"""
const MAX_VAL_STEPWISE_REGRESSION_CRITERIA = (:r2, :adjr2)
"""
    STEPWISE_REGRESSION_CRITERIA

Tuple of the symbols that name the stepwise regression criteria with one score for a fitted model.

[`StepwiseRegression`](@ref) accepts each symbol of this tuple in its `crit` field, and stores it as a `Val`. [`regression_criterion_func`](@ref), [`regression_polarity`](@ref), [`regression_threshold`](@ref) and the `get_*_reg_incl*!` helpers dispatch on that `Val`. The constructor rejects a symbol outside the tuple.

[`PValue`](@ref) is not a member. It reads the p-values of the coefficients of the fitted model, and not one score, so it is a type with its own stepwise methods. `:aic`, `:aicc` and `:bic` score a fitted [`LinearModel`](@ref) and a fitted [`GeneralisedLinearModel`](@ref) the same way. `GLM` defines `:r2` and `:adjr2` for a fitted [`LinearModel`](@ref) only, so under the other target they read a named pseudo-``R^2`` variant.

# Mathematical definition

## `:aic`, the Akaike information criterion

The criterion trades the fitted likelihood against the number of estimated parameters. A lower value is a better model.

```math
\\begin{align}
\\mathrm{AIC} &= 2k - 2\\ln\\hat{L}\\,.
\\end{align}
```

Where:

  - ``k``: Number of estimated parameters, which is `StatsAPI.dof` of the fitted model. It counts the regression coefficients, the intercept and the residual variance.
  - ``\\hat{L}``: Maximum likelihood of the model.

## `:aicc`, the Akaike information criterion with the small-sample correction

The correction term grows as ``T`` comes near ``k``. On a short sample, `:aicc` then gives a large model a larger penalty than `:aic` does.

```math
\\begin{align}
\\mathrm{AICC} &= \\mathrm{AIC} + \\frac{2k(k+1)}{T - k - 1}\\,.
\\end{align}
```

Where:

  - ``k``: Number of estimated parameters, as for `:aic`.
  - $(math_dict[:T])

The correction term divides by ``T - k - 1``. The criterion has no value when ``T = k + 1``, and the term changes sign below it. So `:aicc` needs a sample longer than the largest model that the search can reach.

## `:bic`, the Bayesian information criterion

The penalty is ``k \\ln T`` and not ``2k``. It is larger than the penalty of `:aic` for ``T \\geq 8``, so the search usually stops with fewer factors. [schwarz1978](@cite) maximises the log-likelihood less ``\\tfrac{1}{2} k \\ln T``. The form below multiplies that score by minus two, which keeps the order of the models and makes a lower value better.

```math
\\begin{align}
\\mathrm{BIC} &= k\\ln T - 2\\ln\\hat{L}\\,.
\\end{align}
```

Where:

  - ``k``: Number of estimated parameters, as for `:aic`.
  - $(math_dict[:T])
  - ``\\hat{L}``: Maximum likelihood of the model.

## `:r2`, the coefficient of determination

The criterion is the share of the variance of the response that the model explains. A higher value is a better fit.

```math
\\begin{align}
R^2 &= 1 - \\frac{\\mathrm{SS}_{\\mathrm{res}}}{\\mathrm{SS}_{\\mathrm{tot}}} = 1 - \\frac{\\sum_t (y_t - \\hat{y}_t)^2}{\\sum_t (y_t - \\bar{y})^2}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{SS}_{\\mathrm{res}}``: Residual sum of squares.
  - ``\\mathrm{SS}_{\\mathrm{tot}}``: Total sum of squares.
  - ``y_t``: Observed response at time ``t``.
  - ``\\hat{y}_t``: Fitted response at time ``t``.
  - ``\\bar{y}``: Mean of the observed responses.

``R^2`` never falls when the model gets one more factor, so it gives no penalty for the size of the model. Under [`ForwardSelection`](@ref) it admits every factor, and under [`BackwardElimination`](@ref) it removes no factor. Use `:adjr2`, `:aic`, `:aicc` or `:bic` when the criterion must penalise the size of the model.

## `:adjr2`, the adjusted coefficient of determination

The adjustment discounts ``R^2`` by the degrees of freedom of the predictors. Unlike the score of `:r2`, the score of `:adjr2` can fall when the model gets one more factor, and the search can stop on it.

```math
\\begin{align}
\\bar{R}^2 &= 1 - (1 - R^2) \\frac{T - 1}{T - k - 1}\\,.
\\end{align}
```

Where:

  - ``k``: Number of predictors, without the intercept. `:aic`, `:aicc` and `:bic` use ``k`` for a different count, which adds the intercept and the residual variance to the predictors.
  - $(math_dict[:T])

# Related

  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`PSEUDO_R2_VARIANTS`](@ref)
  - [`regression_criterion_func`](@ref)
  - [`default_regression_criterion_variant`](@ref)
  - [`regression_threshold`](@ref)
  - [`PValue`](@ref)

# References

  - $(ref_dict[:akaike1974]) Section V, page 719.
  - $(ref_dict[:hurvich1989]) Pages 297-307. $(ref_dict[:no_original_source])
  - $(ref_dict[:schwarz1978]) Section 1, page 461, and the Proposition of Section 3, page 462.
  - $(ref_dict[:theil1961]) Page 213. $(ref_dict[:no_original_source])
"""
const STEPWISE_REGRESSION_CRITERIA = (MIN_VAL_STEPWISE_REGRESSION_CRITERIA...,
                                      MAX_VAL_STEPWISE_REGRESSION_CRITERIA...)
"""
    MinValStepwiseRegressionCriterion

Union of the `Val` types that name a stepwise regression criterion for which a lower value is better.

The library builds the members from [`MIN_VAL_STEPWISE_REGRESSION_CRITERIA`](@ref), so the union is `Union{Val{:aic}, Val{:aicc}, Val{:bic}}`. [`StepwiseRegression`](@ref) stores `Val(crit)` in its first field, `crit`. So a member of this union is the first type parameter of the estimator when the criterion is a minimisation criterion.

# Related

  - [`MIN_VAL_STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`MinMaxValStepwiseRegressionCriterion`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`regression_threshold`](@ref)
"""
const MinValStepwiseRegressionCriterion = Union{map(c -> Val{c},
                                                    MIN_VAL_STEPWISE_REGRESSION_CRITERIA)...}
"""
    MaxValStepwiseRegressionCriterion

Union of the `Val` types that name a stepwise regression criterion for which a higher value is better.

The library builds the members from [`MAX_VAL_STEPWISE_REGRESSION_CRITERIA`](@ref), so the union is `Union{Val{:r2}, Val{:adjr2}}`. Both members read a pseudo-``R^2`` variant under a [`GeneralisedLinearModel`](@ref) target.

# Related

  - [`MAX_VAL_STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`MinMaxValStepwiseRegressionCriterion`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`default_regression_criterion_variant`](@ref)
  - [`regression_threshold`](@ref)
"""
const MaxValStepwiseRegressionCriterion = Union{map(c -> Val{c},
                                                    MAX_VAL_STEPWISE_REGRESSION_CRITERIA)...}
"""
    MinMaxValStepwiseRegressionCriterion

Union of every `Val` type that names a stepwise regression criterion with one score for a fitted model.

This is the union of [`MinValStepwiseRegressionCriterion`](@ref) and [`MaxValStepwiseRegressionCriterion`](@ref). The `crit` field of [`StepwiseRegression`](@ref) holds a member of this union when the criterion is one score, and not a [`PValue`](@ref).

# Related

  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`StepwiseRegression`](@ref)
"""
const MinMaxValStepwiseRegressionCriterion = Union{MinValStepwiseRegressionCriterion,
                                                   MaxValStepwiseRegressionCriterion}
"""
    default_regression_criterion_variant(crit::MaxValStepwiseRegressionCriterion)

Return the pseudo-``R^2`` variant a maximisation criterion reads when the target names none.

`StatsAPI.r2` and `StatsAPI.adjr2` both accept `:McFadden` and `:devianceratio`. Of the two, only `:devianceratio` gives the classical ``R^2`` of a fitted [`LinearModel`](@ref) on a Normal family. With this default, the score stays continuous with the [`LinearModel`](@ref) path under each criterion.

# Arguments

  - `crit`: The maximisation criterion, as the `Val` that the `crit` field of [`StepwiseRegression`](@ref) holds.

# Returns

  - `variant::Symbol`: `:devianceratio`.

# Related

  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`GeneralisedLinearModel`](@ref): A target whose `variant` is not `nothing` overrides this default.
  - [`regression_criterion_func`](@ref): The only caller. It reads the default only when the target names no variant.
  - [`PSEUDO_R2_VARIANTS`](@ref)
  - [`ADJUSTED_PSEUDO_R2_VARIANTS`](@ref)
"""
function default_regression_criterion_variant(::MaxValStepwiseRegressionCriterion)
    return :devianceratio
end
"""
    regression_criterion_func(crit::MinMaxValStepwiseRegressionCriterion,
                              tgt::AbstractRegressionTarget)

Return the function that scores a fitted model under a stepwise regression criterion.

The method dispatches on the `Val` that names the criterion and on the regression target, because the two maximisation criteria read a different quantity under each target. The map is:

| Criterion | [`GeneralisedLinearModel`](@ref)          | Every other target |
|:--------- |:----------------------------------------- |:------------------ |
| `:aic`    | `StatsAPI.aic`                            | `StatsAPI.aic`     |
| `:aicc`   | `StatsAPI.aicc`                           | `StatsAPI.aicc`    |
| `:bic`    | `StatsAPI.bic`                            | `StatsAPI.bic`     |
| `:r2`     | `model -> StatsAPI.r2(model, variant)`    | `StatsAPI.r2`      |
| `:adjr2`  | `model -> StatsAPI.adjr2(model, variant)` | `StatsAPI.adjr2`   |

`StatsAPI.aic`, `StatsAPI.aicc` and `StatsAPI.bic` accept a fitted model of each library target, so each minimisation criterion has one method. `StatsAPI.r2` and `StatsAPI.adjr2` accept a fitted [`LinearModel`](@ref) without a variant. A fitted [`GeneralisedLinearModel`](@ref) needs a named pseudo-``R^2`` variant, so each maximisation criterion has two methods. The method for every other target serves [`LinearModel`](@ref) and a target of the caller alike. It returns the `StatsAPI` verb, and the fit of a target of the caller implements that verb, as the `# Interfaces` section of [`AbstractRegressionTarget`](@ref) states. [`PValue`](@ref) has no method here, because it reads the p-values of the coefficients and not one score.

# Algorithm

The five methods that return a `StatsAPI` function run no steps. The two that close over a variant run these steps:

 1. Read `tgt.variant`. When it is `nothing`, take [`default_regression_criterion_variant`](@ref) of `crit`. Call the result `variant`.
 2. Build a closure over `variant` that calls `StatsAPI.r2(model, variant)`, or `StatsAPI.adjr2(model, variant)` under `:adjr2`.

# Arguments

  - `crit`: The criterion, as the `Val` that the `crit` field of [`StepwiseRegression`](@ref) holds.
  - `tgt`: The regression target that fits the candidate models.

# Returns

  - `f::Function`: The function that computes the criterion value for a fitted model.

# Related

  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`MinMaxValStepwiseRegressionCriterion`](@ref)
  - [`default_regression_criterion_variant`](@ref)
  - [`GeneralisedLinearModel`](@ref)
  - [`StepwiseRegression`](@ref)
"""
function regression_criterion_func(::Val{:aic}, ::AbstractRegressionTarget)
    return StatsAPI.aic
end
function regression_criterion_func(::Val{:aicc}, ::AbstractRegressionTarget)
    return StatsAPI.aicc
end
function regression_criterion_func(::Val{:bic}, ::AbstractRegressionTarget)
    return StatsAPI.bic
end
function regression_criterion_func(::Val{:r2}, ::AbstractRegressionTarget)
    return StatsAPI.r2
end
function regression_criterion_func(::Val{:adjr2}, ::AbstractRegressionTarget)
    return StatsAPI.adjr2
end
function regression_criterion_func(crit::Val{:r2}, tgt::GeneralisedLinearModel)
    variant = ifelse(isnothing(tgt.variant), default_regression_criterion_variant(crit),
                     tgt.variant)
    return model -> StatsAPI.r2(model, variant)
end
function regression_criterion_func(crit::Val{:adjr2}, tgt::GeneralisedLinearModel)
    variant = ifelse(isnothing(tgt.variant), default_regression_criterion_variant(crit),
                     tgt.variant)
    return model -> StatsAPI.adjr2(model, variant)
end
"""
    regression_polarity(crit::MinMaxValStepwiseRegressionCriterion)

Return the three functions that state which direction of a stepwise criterion is better.

A stepwise search asks the same three questions of every criterion: which entry of a score vector is the best one, whether a candidate score improves on the score in hand, and what the worst score of a type is. Each answer is one function under a minimised criterion and its opposite under a maximised one. This method pair is the only place in the library that states the pairing. [`regression_threshold`](@ref), [`get_forward_reg_incl_excl!`](@ref), [`get_backward_reg_incl!`](@ref) and the two `_regression` methods that fill a score vector all read it, and none of them states the pairing again.

# Arguments

  - `crit`: The criterion, as the `Val` that the `crit` field of [`StepwiseRegression`](@ref) holds.

# Returns

  - `polarity::NamedTuple`: Three functions.

      + `best`: `findmin` under a minimised criterion, `findmax` under a maximised one. Returns the best entry of a score vector and its index.
      + `improves`: `<` under a minimised criterion, `>` under a maximised one. Returns whether the first score is better than the second.
      + `worst`: `typemax` under a minimised criterion, `typemin` under a maximised one. Returns the worst score of the type it is given.

# Related

  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`regression_threshold`](@ref)
  - [`get_forward_reg_incl_excl!`](@ref)
  - [`get_backward_reg_incl!`](@ref)
"""
function regression_polarity(::MinValStepwiseRegressionCriterion)
    return (; best = findmin, improves = <, worst = typemax)
end
function regression_polarity(::MaxValStepwiseRegressionCriterion)
    return (; best = findmax, improves = >, worst = typemin)
end
"""
    regression_threshold(crit::MinMaxValStepwiseRegressionCriterion)

Return the threshold that a forward stepwise regression search starts from.

The value is the worst score the criterion can take, so the first candidate model always improves on it. [`regression_polarity`](@ref) states which of `typemax` and `typemin` that is, and `typemax(Float64)` is `Inf`. Only [`ForwardSelection`](@ref) reads it. [`BackwardElimination`](@ref) starts from the score of the full model, because its first move must improve on a model that already exists.

# Arguments

  - `crit`: The criterion, as the `Val` that the `crit` field of [`StepwiseRegression`](@ref) holds.

# Returns

  - `t::Number`: `Inf` for a minimisation criterion, and `-Inf` for a maximisation criterion.

# Related

  - [`regression_polarity`](@ref): The pairing that this method reads.
  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`ForwardSelection`](@ref): The only caller.
  - [`BackwardElimination`](@ref): It starts from the score of the full model, and never from this value.
"""
function regression_threshold(crit::MinMaxValStepwiseRegressionCriterion)
    return regression_polarity(crit).worst(Float64)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the loadings matrix, the intercept vector, the reduced-basis loadings and the idiosyncratic covariance of a fitted factor model.

`M` and `b` are the loadings matrix and the intercept vector of the factor model, with one row per asset. `L` holds the loadings in the reduced basis that a dimension reduction made. It is unset when the estimator regresses on the original factors.

An unset `L` reads back as `M`. A [`@forward_properties`](@ref) `swap(L, M)` rule makes `re.L` return `re.M` when the caller gave no `L`. So a consumer that decomposes risk in the factor basis needs no `Nothing` branch, and `isnothing(re.L)` is never true. Read `getfield(re, :L)` to tell the unset case apart, as [`port_opt_view`](@ref) does. [`StepwiseRegression`](@ref) leaves `L` unset and [`DimensionReductionRegression`](@ref) sets it. `size(L, 2)` is then the width of the basis of the risk decomposition, which is the number of original factors under the first, and the number of retained principal components under the second.

`esigma` holds the idiosyncratic covariance that the fit leaves. It has the same name and the same shapes as the field of [`CrossSectionalFactorModel`](@ref), so one reader reads either block. A regression estimator fits loadings alone and writes nothing here. The prior that lifts the factor moments fills the field, and only when that prior adds a residual block. [`FactorPrior`](@ref) and [`FactorBlackLittermanPrior`](@ref) write the residual variances of [`factor_lift`](@ref) under `rsd = true`, and leave the field unset under `rsd = false`.

`edof` and `ediv` record the count behind each idiosyncratic variance. So a consumer that corrects for the sampling error of `esigma`, such as [`ResidualInflation`](@ref), reads both fields from the block, and does not guess the count. [`StepwiseRegression`](@ref) and [`DimensionReductionRegression`](@ref) write `edof` as the number of observations less the parameters that the fit of each asset spent, which include the intercept. The prior that writes `esigma` restates `edof` in the effective count of its variance estimator. It also writes `ediv`, the divisor of each variance in that count, with [`variance_count`](@ref). A block that the caller builds by hand, or that a variance estimator with no count measures, leaves both unset.

`idx` and `ts` are the row key of the block. The loadings are one fit over the sample, so the block has no rows of its own. The key names the observations of the factor returns, and of the reconstruction of the prior result that carries the block. [`FactorPrior`](@ref) writes the key. A block that the caller builds by hand, or that another prior builds, leaves it unset. A realised [`factor_attribution`](@ref) of a cross-validation reads the key to find the observations of each fold.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{x}_{t} &= \\boldsymbol{b} + \\mathbf{M} \\boldsymbol{f}_{t} + \\boldsymbol{\\varepsilon}_{t}\\,.
\\end{align}
```

Where:

  - $(math_dict[:x_t_obs])
  - ``\\boldsymbol{b}``: Intercept vector ``N \\times 1``, `b`. The term is absent when `b` is unset.
  - ``\\mathbf{M}``: Loadings matrix ``N \\times K`` of the factor model, `M`.
  - ``\\boldsymbol{f}_{t}``: Factor returns for observation ``t``, the ``t``-th row of the factor matrix.
  - ``\\boldsymbol{\\varepsilon}_{t}``: Residual returns for observation ``t``, the part of ``\\boldsymbol{x}_{t}`` that the factors do not explain.
  - $(math_dict[:N])
  - $(math_dict[:K])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Regression(;
        M::MatNum,
        L::Option{<:MatNum} = nothing,
        b::Option{<:VecNum} = nothing,
        esigma::Option{<:VecNum_MatNum} = nothing,
        edof::Option{<:VecNum} = nothing,
        ediv::Option{<:VecNum} = nothing,
        idx::Option{<:VecInt} = nothing,
        ts::Option{<:VecDate} = nothing
    ) -> Regression

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(M)`.
  - If provided, `!isempty(b)`, and `length(b) == size(M, 1)`.
  - If provided, `!isempty(L)`, and `size(L, 1) == size(M, 1)`.
  - If provided, `!isempty(esigma)`, and `esigma` carries `size(M, 1)` entries when it is a vector, or is square with `size(M, 1)` rows when it is a matrix.
  - If provided, `edof` and `ediv` each carry `size(M, 1)` entries.
  - [`assert_row_key_part`](@ref) on `idx` and on `ts`. When both are provided, `length(ts) == length(idx)`.

# Examples

```jldoctest
julia> Regression(; M = [1 2 3; 4 5 6], L = [1 2 3 4; 5 6 7 8], b = [1, 2], esigma = [0.1, 0.2])
Regression
       M ┼ 2×3 Matrix{Int64}
       L ┼ 2×4 Matrix{Int64}
       b ┼ Vector{Int64}: [1, 2]
  esigma ┼ Vector{Float64}: [0.1, 0.2]
    edof ┼ nothing
    ediv ┼ nothing
     idx ┼ nothing
      ts ┴ nothing
```

# Related

  - [`AbstractLoadingsRegressionResult`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 4.1, Equations 4.2-4.3. $(ref_dict[:no_original_source])
"""
@concrete struct Regression <: AbstractLoadingsRegressionResult
    """
    $(arg_dict[:M])
    """
    M
    """
    $(arg_dict[:L])
    """
    L
    """
    $(arg_dict[:b])
    """
    b
    """
    $(arg_dict[:esigma])
    """
    esigma
    """
    $(arg_dict[:edof])
    """
    edof
    """
    $(arg_dict[:ediv])
    """
    ediv
    """
    Position of each observation that the prior result describes, in the returns data that the prior read, or `nothing`. A Scenario Cap keeps the last observations alone, so the positions are then the last positions of the data.
    """
    idx
    """
    Timestamp of each observation that `idx` names, or `nothing` when the returns data that the prior read carries no timestamps. When the folds of a cross-validation carry timestamps too, a [`factor_attribution`](@ref) matches the two by timestamp, so a prior fitted on other returns data still finds its observations.
    """
    ts
    function Regression(M::MatNum, L::Option{<:MatNum}, b::Option{<:VecNum},
                        esigma::Option{<:VecNum_MatNum}, edof::Option{<:VecNum},
                        ediv::Option{<:VecNum}, idx::Option{<:VecInt},
                        ts::Option{<:VecDate})
        @argcheck(!isempty(M), IsEmptyError)
        if isa(b, VecNum)
            @argcheck(!isempty(b), IsEmptyError)
            @argcheck(length(b) == size(M, 1), DimensionMismatch)
        end
        if !isnothing(L)
            @argcheck(!isempty(L), IsEmptyError)
            @argcheck(size(L, 1) == size(M, 1), DimensionMismatch)
        end
        assert_idiosyncratic_covariance(esigma, size(M, 1))
        assert_idiosyncratic_count(edof, size(M, 1), :edof)
        assert_idiosyncratic_count(ediv, size(M, 1), :ediv)
        Tk = row_key_length(idx)
        assert_row_key_part(idx, Tk, :idx)
        assert_row_key_part(ts, Tk, :ts)
        return new{typeof(M), typeof(L), typeof(b), typeof(esigma), typeof(edof),
                   typeof(ediv), typeof(idx), typeof(ts)}(M, L, b, esigma, edof, ediv, idx,
                                                          ts)
    end
end
function Regression(; M::MatNum, L::Option{<:MatNum} = nothing,
                    b::Option{<:VecNum} = nothing,
                    esigma::Option{<:VecNum_MatNum} = nothing,
                    edof::Option{<:VecNum} = nothing, ediv::Option{<:VecNum} = nothing,
                    idx::Option{<:VecInt} = nothing,
                    ts::Option{<:VecDate} = nothing)::Regression
    return Regression(M, L, b, esigma, edof, ediv, idx, ts)
end
# When `L` is unset (`Nothing` type parameter), `:L` falls back to the loadings matrix `M`;
# when `L` is a stored matrix the default field access already returns it, so only the
# `Nothing` specialisation needs a rule (see [`@forward_properties`](@ref)'s `swap`).
@forward_properties Regression{<:Any, Nothing, <:Any, <:Any, <:Any, <:Any} begin
    swap(L, M)
end
"""
    port_opt_view(re::Regression, i)

Return a view of a [`Regression`](@ref) result that keeps the assets that `i` selects.

The new result restricts the loadings matrix `M`, the optional reduced-basis loadings `L`, the intercept vector `b`, the idiosyncratic covariance `esigma` and the counts `edof` and `ediv` to the selected assets.

# Algorithm

 1. Read `L` and `b` with `getfield`, never through property access. The `swap(L, M)` rule of [`Regression`](@ref) makes `re.L` return `re.M` when `L` is unset. So a property read puts a view of `M` in the `L` field of the new result, and the new result no longer shows that `L` was unset.
 2. Take a row view of `M` over `i`. These are the loadings of the selected assets.
 3. Take a row view of `L` over `i` when step 1 found a matrix, and `nothing` otherwise.
 4. Take an element view of `b` over `i` when step 1 found a vector, and `nothing` otherwise.
 5. View `esigma` with [`idiosyncratic_covariance_view`](@ref), which reads its shape: a vector of variances is indexed once, and a full covariance is indexed on both axes.
 6. View `edof` and `ediv` with [`nothing_scalar_array_view`](@ref), which indexes a vector once and passes `nothing` through.
 7. Build a new [`Regression`](@ref) from the six views and the unchanged row key `idx` and `ts`. The constructor runs all its guards again. The key names observations, so an asset selection does not change it.

# Arguments

  - `re`: The regression result.
  - `i`: The indices of the assets to keep.

# Returns

  - `reg::Regression`: A new regression result. Each field with one entry per asset holds the selected assets alone.

# Examples

```jldoctest
julia> re = Regression(; M = [1 2; 3 4; 5 6], L = [10 20; 30 40; 50 60], b = [7, 8, 9])
Regression
       M ┼ 3×2 Matrix{Int64}
       L ┼ 3×2 Matrix{Int64}
       b ┼ Vector{Int64}: [7, 8, 9]
  esigma ┼ nothing
    edof ┼ nothing
    ediv ┼ nothing
     idx ┼ nothing
      ts ┴ nothing

julia> PortfolioOptimisers.port_opt_view(re, [1, 3])
Regression
       M ┼ 2×2 SubArray{Int64, 2, Matrix{Int64}, Tuple{Vector{Int64}, Base.Slice{Base.OneTo{Int64}}}, false}
       L ┼ 2×2 SubArray{Int64, 2, Matrix{Int64}, Tuple{Vector{Int64}, Base.Slice{Base.OneTo{Int64}}}, false}
       b ┼ SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}: [7, 9]
  esigma ┼ nothing
    edof ┼ nothing
    ediv ┼ nothing
     idx ┼ nothing
      ts ┴ nothing
```

# Related

  - [`Regression`](@ref)
"""
function port_opt_view(re::Regression, i, args...)::Regression
    # `L` and `b` are both optional, and `L` must be read with `getfield`: the
    # `swap(L, M)` property rule above makes `re.L` return `re.M` when `L` is unset, so
    # `isnothing(re.L)` is never true and a viewed result would materialise `L` as a copy
    # of `M`, silently losing the unset-ness the rule exists to express.
    L = getfield(re, :L)
    b = getfield(re, :b)
    return Regression(; M = view(re.M, i, :), L = isnothing(L) ? nothing : view(L, i, :),
                      b = isnothing(b) ? nothing : view(b, i),
                      esigma = idiosyncratic_covariance_view(re.esigma, i),
                      edof = nothing_scalar_array_view(re.edof, i),
                      ediv = nothing_scalar_array_view(re.ediv, i), idx = re.idx,
                      ts = re.ts)
end
"""
    set_idiosyncratic_covariance(re::Regression, esigma::Option{<:VecNum_MatNum},
                                 edof::Option{<:VecNum}, ediv::Option{<:VecNum})

Return a [`Regression`](@ref) that carries `esigma`, `edof` and `ediv`, with every other field unchanged, the row key included.

A regression estimator fits loadings alone, so the block that a fit returns carries no idiosyncratic covariance. The prior that lifts the factor moments measures the residual variances in the lift. It writes them here with their counts, so that no consumer computes them again. [`FactorPrior`](@ref) and [`FactorBlackLittermanPrior`](@ref) are the two callers. Each passes what [`factor_lift`](@ref) returns, which is the variances and their counts under `rsd = true`, and `nothing` for the variances and the divisors under `rsd = false`.

This method reads `L` and `b` with `getfield`, as [`port_opt_view`](@ref) does. The `swap(L, M)` rule of [`Regression`](@ref) makes `re.L` return `re.M` when `L` is unset. So a read through the property writes `M` under `L` in the new result, and `isnothing(getfield(re, :L))` is then false.

# Arguments

  - `re`: The regression result to copy.
  - `esigma`: The idiosyncratic covariance to write, or `nothing` to leave the field unset.
  - `edof`: The degrees of freedom of each residual series, or `nothing` to leave the field unset.
  - `ediv`: The divisor of each idiosyncratic variance, or `nothing` to leave the field unset.

# Returns

  - `re::Regression`: A new result that carries `esigma`, `edof` and `ediv`. The constructor runs all its guards again.

# Related

  - [`Regression`](@ref)
  - [`factor_lift`](@ref)
  - [`port_opt_view`](@ref)
"""
function set_idiosyncratic_covariance(re::Regression, esigma::Option{<:VecNum_MatNum},
                                      edof::Option{<:VecNum},
                                      ediv::Option{<:VecNum})::Regression
    return Regression(; M = re.M, L = getfield(re, :L), b = getfield(re, :b),
                      esigma = esigma, edof = edof, ediv = ediv, idx = re.idx, ts = re.ts)
end
"""
    set_row_key(re::Regression, idx::Option{<:VecInt}, ts::Option{<:VecDate})

Return a [`Regression`](@ref) that carries the row key `idx` and `ts`, with every other field unchanged.

A regression estimator fits loadings alone, and does not record the observations that it read. [`FactorPrior`](@ref) writes the key after the fit, because it knows which observations its prior result describes. [`map_row_positions`](@ref) maps the key through a selection of observations, and [`forget_row_positions`](@ref) drops the positions of the key. Both write the new key with this method. The method reads `L` and `b` with `getfield`, for the reason that [`set_idiosyncratic_covariance`](@ref) states.

# Arguments

  - `re`: The regression result to copy.
  - `idx`: The position of each observation in the returns data that the prior read, or `nothing`.
  - `ts`: The timestamp of each observation, or `nothing`.

# Returns

  - `re::Regression`: A new result that carries `idx` and `ts`. The constructor runs all its guards again.

# Related

  - [`Regression`](@ref)
  - [`attribution_row_key`](@ref)
  - [`set_idiosyncratic_covariance`](@ref)
  - [`map_row_positions`](@ref)
  - [`forget_row_positions`](@ref)
"""
function set_row_key(re::Regression, idx::Option{<:VecInt},
                     ts::Option{<:VecDate})::Regression
    return Regression(; M = re.M, L = getfield(re, :L), b = getfield(re, :b),
                      esigma = re.esigma, edof = re.edof, ediv = re.ediv, idx = idx,
                      ts = ts)
end
"""
    regression(re::Regression, args...)

Return the regression result unchanged.

A caller that holds a [`RegE_Reg`](@ref) can call `regression` on it, whether it is an estimator or a result. On a [`Regression`](@ref), the call returns its argument, so the caller needs no branch.

# Arguments

  - `re`: The regression result.
  - `args...`: More arguments, which the method ignores.

# Returns

  - `re::Regression`: The input `re`, unchanged.

# Related

  - [`Regression`](@ref)
"""
function regression(re::Regression, args...)
    return re
end
"""
    regression(re::AbstractTimeSeriesRegressionEstimator, rd::ReturnsResult)

Fit a time-series regression estimator to the asset returns and the factor returns of a [`ReturnsResult`](@ref).

The method calls `regression(re, X, rd.F)` on the observations that it keeps. A [`Regression`](@ref) result is not an estimator, so it does not reach this method. The `regression` method of a result returns the result unchanged.

# Algorithm

 1. Check that `rd` carries both matrices, as `# Validation` states.
 2. Read the source of `re` with [`returns_source`](@ref), and call it `x_src`. A regression that reads the returns data itself states its source in its `x_src` field.
 3. With [`shared_series_view`](@ref), drop the observations at which a factor return is not finite. Under `x_src = ExcessSource()`, also drop the observations of a gap of a vector benchmark. A regression on such an observation has no regressor for any asset.
 4. Read the asset returns with [`benchmark_relative_returns`](@ref) under `x_src`, and call them `X`. `X` is `rd.X`, or `rd.X - rd.B` under `ExcessSource()`.
 5. Call `regression(re, X, rd.F)` on the kept observations, and return its result.

# Arguments

  - `re`: The time-series regression estimator.
  - `rd`: The returns result, which carries the asset returns `X` and the factor returns `F`.

# Validation

  - `!isnothing(rd.X)`. A regression needs the asset returns it explains.
  - `!isnothing(rd.F)`. A regression needs the factor returns it explains them with.
  - The rules of [`benchmark_relative_returns`](@ref). So `x_src = PriorSource()` throws an `ArgumentError`, because the call gives no Prior Result, and `x_src = ExcessSource()` refuses returns data with no benchmark.

# Returns

  - `reg::Regression`: The fitted regression result.

# Related

  - [`Regression`](@ref)
  - [`ReturnsResult`](@ref)
"""
function regression(re::AbstractTimeSeriesRegressionEstimator, rd::ReturnsResult)
    @argcheck(!isnothing(rd.X), IsNothingError)
    @argcheck(!isnothing(rd.F), IsNothingError)
    x_src = returns_source(re)
    rd = shared_series_view(rd, x_src, true)
    return regression(re, benchmark_relative_returns(rd, x_src), rd.F)
end
"""
    assert_finite_factor_rows(F::MatNum) -> Nothing

Refuse a factor returns matrix with a non-finite entry, for a regression on matrices.

A factor return is a regressor of every asset, so a gap at one observation leaves that observation with no regressor for any asset. A fit through a [`ReturnsResult`](@ref) drops such an observation with [`shared_series_view`](@ref) before it reaches a matrix. A caller of a matrix method gives the observations that it chose, and the method refuses a gap in them. It does not return `NaN` loadings.

# Arguments

  - $(arg_dict[:F])

# Validation

  - Every entry of `F` is finite. Otherwise, the method throws an [`IsNonFiniteError`](@ref) that names the first observations with a gap.

# Returns

  - `nothing`.

# Related

  - [`shared_series_view`](@ref)
  - [`regression`](@ref)
"""
function assert_finite_factor_rows(F::MatNum)::Nothing
    if all(isfinite, F)
        return nothing
    end
    rows = findall(t -> !all(isfinite, view(F, t, :)), axes(F, 1))
    return throw(IsNonFiniteError("a factor return is a regressor of every asset, and the factor returns are not finite at observation(s) $(first(rows, 5)) of $(size(F, 1)), so no asset has a regressor there. Fit through a ReturnsResult, which drops those observations, or drop them from X and F before the call."))
end
"""
    pin_regression_choice(re::AbstractTimeSeriesRegressionEstimator, X::MatNum, F::MatNum,
                          w::Option{<:ObsWeights} = nothing)

Return a regression estimator that has no choice to pin, unchanged.

The online step of a prior that fits a regression calls [`pin_regression_choice`](@ref) after the fold, over the rows of its buffer. A regression that makes a choice over every observation, such as the factor set of [`StepwiseRegression`](@ref) or the components of [`DimensionReductionRegression`](@ref), has a method of its own under a [`PinnedChoice`](@ref). That method writes the choice of the first fit into the configuration of the estimator. This method serves every other regression, and a regression under [`BatchChoice`](@ref).

# Arguments

  - `re`: The regression estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:F])
  - $(arg_dict[:w_pin])

# Returns

  - `re::AbstractTimeSeriesRegressionEstimator`: The input estimator, unchanged.

# Related

  - [`StepwiseRegression`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`PinnedChoice`](@ref)
  - [`pin_prior_choice`](@ref)
"""
function pin_regression_choice(re::AbstractTimeSeriesRegressionEstimator, ::MatNum,
                               ::MatNum, ::Option{<:ObsWeights} = nothing)
    return re
end

export regression, Regression, LinearModel, GeneralisedLinearModel, BenchmarkWeightMetric,
       RegressionWeightMetric, InverseIdiosyncraticVarianceMetric
public AbstractTimeSeriesRegressionEstimator, AbstractCrossSectionalRegressionEstimator
# The `# Interfaces`-marked type of #1145 (ADR 0154): AbstractOrthogonalityMetric names the two
# verbs an extension must implement, whose concrete methods live under
# src/11_UncertaintySets/09_OrthogonalUncertaintySets.jl -- the same split #1137 already used.
public AbstractOrthogonalityMetric, orthogonality_weights, cs_diagnostic_weights
# The `# Interfaces`-marked type of #1397 (ADR 0154): a caller's own regression target. The
# verbs its section names are `factory`, which is exported, `StatsAPI` verbs, and
# `regression_target_weights`, which #1441 added for the time-series estimators, and
# `is_basis_invariant`, which #1614 added for the carry fold of the cross-sectional prior.
public AbstractRegressionTarget, regression_target_weights, is_basis_invariant,
       residual_count
# The two branches of #1885 (ADR 0223): a caller's own target subtypes one of them.
public AbstractUnpenalisedRegressionTarget, AbstractPenalisedRegressionTarget
export InterceptColumn
