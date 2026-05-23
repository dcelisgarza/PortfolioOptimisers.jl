"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all regression estimator types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing regression estimation algorithms should be subtypes of `AbstractRegressionEstimator`.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractRegressionAlgorithm`](@ref)
  - [`AbstractRegressionResult`](@ref)
"""
abstract type AbstractRegressionEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all regression result types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types representing the output of regression-based moment estimation should be subtypes of `AbstractRegressionResult`.

# Related

  - [`AbstractResult`](@ref)
  - [`Regression`](@ref)
  - [`AbstractRegressionEstimator`](@ref)
"""
abstract type AbstractRegressionResult <: AbstractResult end
"""
    const RegE_Reg = Union{<:AbstractRegressionResult, <:AbstractRegressionEstimator}

Alias for a regression result or estimator.

Matches either an [`AbstractRegressionResult`](@ref) (pre-computed regression result) or an [`AbstractRegressionEstimator`](@ref) (regression specification). Used for dispatch in factor model and regression-based risk routines.

# Related

  - [`AbstractRegressionResult`](@ref)
  - [`AbstractRegressionEstimator`](@ref)
"""
const RegE_Reg = Union{<:AbstractRegressionResult, <:AbstractRegressionEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all regression algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing specific regression algorithms should be subtypes of `AbstractRegressionAlgorithm`.

These types are used to specify the algorithm when constructing a regression estimator.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractRegressionAlgorithm`](@ref)
  - [`AbstractStepwiseRegressionAlgorithm`](@ref)
  - [`AbstractStepwiseRegressionCriterion`](@ref)
  - [`AbstractRegressionTarget`](@ref)
"""
abstract type AbstractRegressionAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all stepwise regression algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing stepwise regression algorithms should be subtypes of `AbstractStepwiseRegressionAlgorithm`.

# Related

  - [`AbstractRegressionAlgorithm`](@ref)
  - [`AbstractStepwiseRegressionCriterion`](@ref)
  - [`AbstractRegressionTarget`](@ref)
"""
abstract type AbstractStepwiseRegressionAlgorithm <: AbstractRegressionAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all stepwise regression criterion types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types representing criteria for stepwise regression algorithms should be subtypes of `AbstractStepwiseRegressionCriterion`. These criteria are used to evaluate model quality and guide variable selection during stepwise regression, such as AIC, BIC, or R².

# Related

  - [`AbstractStepwiseRegressionAlgorithm`](@ref)
  - [`AbstractRegressionTarget`](@ref)
"""
abstract type AbstractStepwiseRegressionCriterion <: AbstractRegressionAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all regression target types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types representing regression targets (such as linear or generalised linear models) should be subtypes of `AbstractRegressionTarget`.

# Related

  - [`AbstractRegressionAlgorithm`](@ref)
"""
abstract type AbstractRegressionTarget <: AbstractRegressionAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Regression target type for standard linear models in `PortfolioOptimisers.jl`.

`LinearModel` is used to specify a standard linear regression target (i.e., ordinary least squares) when constructing regression estimators. It encapsulates keyword arguments for configuring the underlying linear model fitting routine, enabling flexible extension and dispatch within the regression estimation framework.

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
@concrete struct LinearModel <: AbstractRegressionTarget
    "Keyword arguments passed to `fit(GLM.LinearModel, X, y; kwargs...)`."
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

# Arguments

  - `re`: Linear model regression target.
  - $(arg_dict[:ow])

# Returns

  - `re::LinearModel`: Updated regression target with weights included in `kwargs`.

# Related

  - [`LinearModel`](@ref)
  - [`factory`](@ref)
"""
function factory(re::LinearModel, w::ObsWeights)::LinearModel
    return LinearModel(; kwargs = (; re.kwargs..., weights = w))
end
"""
    StatsAPI.fit(tgt::LinearModel, X::MatNum, y::VecNum)

Fit a standard linear regression model using a [`LinearModel`](@ref) regression target.

This method dispatches to `StatsAPI.fit` with the `GLM.LinearModel` type, passing the design matrix `X`, response vector `y`, and any keyword arguments stored in `tgt.kwargs`. It enables flexible configuration of the underlying linear model fitting routine within the regression estimation framework.

# Arguments

  - `tgt`: Regression target specifying model options.
  - `X`: The design matrix (observations × features).
  - `y`: The response vector.

# Returns

  - `model::GLM.LinearModel`: A fitted linear model object from the GLM.jl package.

# Related

  - [`LinearModel`](@ref)
  - [`GLM.LinearModel`](https://juliastats.org/GLM.jl/stable/api/#GLM.LinearModel)
"""
function StatsAPI.fit(tgt::LinearModel, X::MatNum, y::VecNum)
    kwargs = if haskey(tgt.kwargs, :weights) &&
                isa(tgt.kwargs.weights, DynamicAbstractWeights)
        w = get_observation_weights(tgt.kwargs.weights, X)
        (; tgt.kwargs..., weights = w)
    else
        tgt.kwargs
    end
    return StatsAPI.fit(GLM.LinearModel, X, y; kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Regression target type for generalised linear models (GLMs) in `PortfolioOptimisers.jl`.

`GeneralisedLinearModel` is used to specify a generalised linear regression target (e.g., logistic, Poisson, etc.) when constructing regression estimators. It encapsulates positional and keyword arguments for configuring the underlying GLM fitting routine, enabling flexible extension and dispatch within the regression estimation framework.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GeneralisedLinearModel(;
        args::Tuple = (Normal(),),
        kwargs::NamedTuple = (;)
    ) -> GeneralisedLinearModel

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> GeneralisedLinearModel()
GeneralisedLinearModel
    args ┼ Tuple{Distributions.Normal{Float64}}: (Distributions.Normal{Float64}(μ=0.0, σ=1.0),)
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractRegressionTarget`](@ref)
  - [`LinearModel`](@ref)
  - [`StatsAPI.fit(::GeneralisedLinearModel, ::MatNum, ::VecNum)`](@ref)
"""
@concrete struct GeneralisedLinearModel <: AbstractRegressionTarget
    "Positional arguments passed to `fit(GLM.GeneralizedLinearModel, X, y, args...; kwargs...)`."
    args
    "Keyword arguments passed to `fit(GLM.GeneralizedLinearModel, X, y, args...; kwargs...)`."
    kwargs
    function GeneralisedLinearModel(args::Tuple, kwargs::NamedTuple)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function GeneralisedLinearModel(; args::Tuple = (Distributions.Normal(),),
                                kwargs::NamedTuple = (;))::GeneralisedLinearModel
    return GeneralisedLinearModel(args, kwargs)
end
"""
    factory(re::GeneralisedLinearModel, w::ObsWeights) -> GeneralisedLinearModel

Return a new [`GeneralisedLinearModel`](@ref) regression target with observation weights `w` added to the keyword arguments.

# Arguments

  - `re`: Generalised linear model regression target.
  - $(arg_dict[:ow])

# Returns

  - `re::GeneralisedLinearModel`: Updated regression target with weights included in `kwargs`.

# Related

  - [`GeneralisedLinearModel`](@ref)
  - [`factory`](@ref)
"""
function factory(re::GeneralisedLinearModel, w::ObsWeights)::GeneralisedLinearModel
    return GeneralisedLinearModel(; args = re.args, kwargs = (; re.kwargs..., weights = w))
end
"""
    StatsAPI.fit(tgt::GeneralisedLinearModel, X::MatNum, y::VecNum)

Fit a generalised linear regression model using a [`GeneralisedLinearModel`](@ref) regression target.

This method dispatches to `StatsAPI.fit` with the `GLM.GeneralizedLinearModel` type, passing the design matrix `X`, response vector `y`, any positional arguments in `tgt.args`, and any keyword arguments in `tgt.kwargs`.

# Arguments

  - `tgt`: A [`GeneralisedLinearModel`](@ref) regression target specifying model options.
  - `X`: The design matrix (observations × features).
  - `y`: The response vector.

# Returns

  - `model::GLM.GeneralizedLinearModel`: A fitted generalised linear model object from the GLM.jl package.

# Related

  - [`GeneralisedLinearModel`](@ref)
  - [`GLM.GeneralizedLinearModel`](https://juliastats.org/GLM.jl/stable/examples/#Probit-regression)
"""
function StatsAPI.fit(tgt::GeneralisedLinearModel, X::MatNum, y::VecNum)
    kwargs = if haskey(tgt.kwargs, :weights) &&
                isa(tgt.kwargs.weights, DynamicAbstractWeights)
        w = get_observation_weights(tgt.kwargs.weights, X)
        (; tgt.kwargs..., weights = w)
    else
        tgt.kwargs
    end
    return StatsAPI.fit(GLM.GeneralizedLinearModel, X, y, tgt.args...; kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all stepwise regression criteria in `PortfolioOptimisers.jl` where model fit is evaluated by either minimising or maximising the criterion value.

All concrete and/or abstract types representing stepwise regression criteria (such as AIC, BIC, R², or Adjusted R²) should be subtypes of `AbstractMinMaxValStepwiseRegressionCriterion`.

# Related Types

  - [`AbstractMinValStepwiseRegressionCriterion`](@ref)
  - [`AbstractMaxValStepwiseRegressionCriteria`](@ref)
"""
abstract type AbstractMinMaxValStepwiseRegressionCriterion <:
              AbstractStepwiseRegressionCriterion end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all stepwise regression criteria where lower values indicate better model fit in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing minimisation-based stepwise regression criteria (such as AIC, AICC, or BIC) should be subtypes of `AbstractMinValStepwiseRegressionCriterion`. These criteria are used to guide variable selection in stepwise regression algorithms by minimising the criterion value.

# Related

  - [`AbstractMinMaxValStepwiseRegressionCriterion`](@ref)
  - [`AIC`](@ref)
  - [`AICC`](@ref)
  - [`BIC`](@ref)
"""
abstract type AbstractMinValStepwiseRegressionCriterion <:
              AbstractMinMaxValStepwiseRegressionCriterion end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all stepwise regression criteria where higher values indicate better model fit in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing maximisation-based stepwise regression criteria (such as R² or Adjusted R²) should be subtypes of `AbstractMaxValStepwiseRegressionCriteria`. These criteria are used to guide variable selection in stepwise regression algorithms by maximising the criterion value.

# Related

  - [`AbstractMinMaxValStepwiseRegressionCriterion`](@ref)
  - [`RSquared`](@ref)
  - [`AdjustedRSquared`](@ref)
"""
abstract type AbstractMaxValStepwiseRegressionCriteria <:
              AbstractMinMaxValStepwiseRegressionCriterion end
"""
$(DocStringExtensions.TYPEDEF)

Akaike Information Criterion (AIC) for stepwise regression in `PortfolioOptimisers.jl`.

`AIC` is a minimisation-based criterion used to evaluate model quality in stepwise regression algorithms. Lower values indicate better model fit, penalising model complexity to avoid overfitting.

# Related

  - [`AbstractMinValStepwiseRegressionCriterion`](@ref)
  - [`AICC`](@ref)
  - [`BIC`](@ref)
  - [`regression_criterion_func(::AIC)`](@ref)
"""
struct AIC <: AbstractMinValStepwiseRegressionCriterion end
"""
$(DocStringExtensions.TYPEDEF)

Corrected Akaike Information Criterion (AICC) for stepwise regression in `PortfolioOptimisers.jl`.

`AICC` is a minimisation-based criterion similar to AIC, but includes a correction for small sample sizes. Lower values indicate better model fit, balancing fit and complexity.

# Related

  - [`AbstractMinValStepwiseRegressionCriterion`](@ref)
  - [`AIC`](@ref)
  - [`BIC`](@ref)
  - [`regression_criterion_func(::AICC)`](@ref)
"""
struct AICC <: AbstractMinValStepwiseRegressionCriterion end
"""
$(DocStringExtensions.TYPEDEF)

Bayesian Information Criterion (BIC) for stepwise regression in `PortfolioOptimisers.jl`.

`BIC` is a minimisation-based criterion used to evaluate model quality in stepwise regression algorithms. It penalises model complexity more strongly than AIC. Lower values indicate better model fit.

# Related

  - [`AbstractMinValStepwiseRegressionCriterion`](@ref)
  - [`AIC`](@ref)
  - [`AICC`](@ref)
  - [`regression_criterion_func(::BIC)`](@ref)
"""
struct BIC <: AbstractMinValStepwiseRegressionCriterion end
"""
$(DocStringExtensions.TYPEDEF)

Coefficient of determination (R²) for stepwise regression in `PortfolioOptimisers.jl`.

`RSquared` is a maximisation-based criterion used to evaluate model quality in stepwise regression algorithms. Higher values indicate better model fit, representing the proportion of variance explained by the model.

# Related

  - [`AbstractMaxValStepwiseRegressionCriteria`](@ref)
  - [`AdjustedRSquared`](@ref)
  - [`regression_criterion_func(::RSquared)`](@ref)
"""
struct RSquared <: AbstractMaxValStepwiseRegressionCriteria end
"""
$(DocStringExtensions.TYPEDEF)

Adjusted coefficient of determination (Adjusted R²) for stepwise regression in `PortfolioOptimisers.jl`.

`AdjustedRSquared` is a maximisation-based criterion that adjusts R² for the number of predictors in the model, providing a more accurate measure of model quality when comparing models with different numbers of predictors.

# Related

  - [`AbstractMaxValStepwiseRegressionCriteria`](@ref)
  - [`RSquared`](@ref)
  - [`regression_criterion_func(::AdjustedRSquared)`](@ref)
"""
struct AdjustedRSquared <: AbstractMaxValStepwiseRegressionCriteria end
"""
    regression_criterion_func(::AbstractStepwiseRegressionCriterion)

Return the function used to compute the value of a stepwise regression criterion.

This utility dispatches on the concrete criterion subtype of [`AbstractStepwiseRegressionCriterion`](@ref), returning the corresponding function from [`GLM.jl`](https://juliastats.org/GLM.jl/stable/#Methods-applied-to-fitted-models). Used internally by stepwise regression algorithms to evaluate model quality.

# Arguments

  - `criterion`: A stepwise regression criterion type (e.g., `AIC()`, `BIC()`, `RSquared()`).

# Returns

  - `f::Function`: The function that computes the criterion value for a fitted model.

# Related

  - [`AIC`](@ref)
  - [`AICC`](@ref)
  - [`BIC`](@ref)
  - [`RSquared`](@ref)
  - [`AdjustedRSquared`](@ref)
"""
function regression_criterion_func(::AIC)
    return StatsAPI.aic
end
function regression_criterion_func(::AICC)
    return StatsAPI.aicc
end
function regression_criterion_func(::BIC)
    return StatsAPI.bic
end
function regression_criterion_func(::RSquared)
    return GLM.r2
end
function regression_criterion_func(::AdjustedRSquared)
    return GLM.adjr2
end
"""
    regression_threshold(alg)

Return the threshold value for stepwise regression selection criteria.

Returns the threshold value associated with a stepwise regression criterion. Dispatches on the criterion type to return either a minimum or maximum value.

# Arguments

  - `alg`: Stepwise regression criterion.

# Returns

  - Threshold value.

# Related

  - [`AbstractMinValStepwiseRegressionCriterion`](@ref)
  - [`AbstractMaxValStepwiseRegressionCriteria`](@ref)
"""
function regression_threshold(::AbstractMinValStepwiseRegressionCriterion)
    return Inf
end
function regression_threshold(::AbstractMaxValStepwiseRegressionCriteria)
    return -Inf
end
"""
$(DocStringExtensions.TYPEDEF)

Container type for regression results in `PortfolioOptimisers.jl`.

`Regression` stores the results of a regression-based moment estimation, including the main coefficient matrix, an optional auxiliary matrix, and the intercept vector. This type is used as the standard output for regression estimators, enabling consistent downstream processing and analysis.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Regression(;
        M::MatNum,
        L::Option{<:MatNum} = nothing,
        b::Option{<:VecNum} = nothing
    ) -> Regression

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(M)`.
  - If provided, `!isempty(b)`, and `length(b) == size(M, 1)`.
  - If provided, `!isempty(L)`, and `size(L, 1) == size(M, 1)`.

# Examples

```jldoctest
julia> Regression(; M = [1 2 3; 4 5 6], L = [1 2 3 4; 5 6 7 8], b = [1, 2])
Regression
  M ┼ 2×3 Matrix{Int64}
  L ┼ 2×4 Matrix{Int64}
  b ┴ Vector{Int64}: [1, 2]
```

# Related

  - [`AbstractRegressionResult`](@ref)
"""
@concrete struct Regression <: AbstractRegressionResult
    "$(arg_dict[:M])"
    M
    "$(arg_dict[:L])"
    L
    "$(arg_dict[:b])"
    b
    function Regression(M::MatNum, L::Option{<:MatNum}, b::Option{<:VecNum})
        @argcheck(!isempty(M), IsEmptyError)
        if isa(b, VecNum)
            @argcheck(!isempty(b), IsEmptyError)
            @argcheck(length(b) == size(M, 1), DimensionMismatch)
        end
        if !isnothing(L)
            @argcheck(size(L, 1) == size(M, 1), DimensionMismatch)
        end
        return new{typeof(M), typeof(L), typeof(b)}(M, L, b)
    end
end
function Regression(; M::MatNum, L::Option{<:MatNum} = nothing,
                    b::Option{<:VecNum} = nothing)::Regression
    return Regression(M, L, b)
end
function Base.getproperty(re::Regression{<:Any, Nothing, <:Any}, sym::Symbol)
    return if sym == :L
        getfield(re, :M)
    else
        getfield(re, sym)
    end
end
function Base.getproperty(re::Regression{<:Any, <:MatNum, <:Any}, sym::Symbol)
    return if sym == :L
        getfield(re, :L)
    else
        getfield(re, sym)
    end
end
"""
    regression_view(re::Regression, i)

Return a view of a [`Regression`](@ref) result object, selecting only the rows indexed by `i`.

This function constructs a new `Regression` result, where the coefficient matrix `M`, optional auxiliary matrix `L`, and intercept vector `b` are restricted to the rows specified by the index vector `i`. This is useful for extracting or operating on a subset of regression results, such as for a subset of assets or features.

# Arguments

  - `re`: A regression result object.
  - `i`: Indices of the rows to select.

# Returns

  - `reg::Regression`: A new regression result object with fields restricted to the selected rows.

# Examples

```jldoctest
julia> re = Regression(; M = [1 2; 3 4; 5 6], L = [10 20; 30 40; 50 60], b = [7, 8, 9])
Regression
  M ┼ 3×2 Matrix{Int64}
  L ┼ 3×2 Matrix{Int64}
  b ┴ Vector{Int64}: [7, 8, 9]

julia> PortfolioOptimisers.regression_view(re, [1, 3])
Regression
  M ┼ 2×2 SubArray{Int64, 2, Matrix{Int64}, Tuple{Vector{Int64}, Base.Slice{Base.OneTo{Int64}}}, false}
  L ┼ 2×2 SubArray{Int64, 2, Matrix{Int64}, Tuple{Vector{Int64}, Base.Slice{Base.OneTo{Int64}}}, false}
  b ┴ SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}: [7, 9]
```

# Related

  - [`Regression`](@ref)
"""
function regression_view(re::Regression, i)::Regression
    return Regression(; M = view(re.M, i, :),
                      L = isnothing(re.L) ? nothing : view(re.L, i, :), b = view(re.b, i))
end
"""
    regression_view(re::Option{<:AbstractRegressionEstimator}, args...)

No-op fallback for `regression_view` when the input is `nothing` or an `AbstractRegressionEstimator`.

This method returns the input `re` unchanged. It is used internally to allow generic code to call `regression_view` without needing to check for `nothing` or estimator types, ensuring graceful handling of missing or non-result regression objects.

# Arguments

  - `re`: Either `nothing` or a regression estimator type.
  - `args...`: Additional arguments (ignored).

# Returns

  - The input `re`, unchanged.

# Related

  - [`regression_view(::Regression, ::VecNum)`](@ref)
"""
function regression_view(re::Option{<:AbstractRegressionEstimator}, args...)
    return re
end
"""
    regression(re::Regression, args...)

Return the regression result object unchanged.

This method is a pass-through for [`Regression`](@ref) result objects, allowing generic code to call `regression` on a result and receive the same object. It enables a unified interface for both estimator and result types.

# Arguments

  - `re`: A regression result object.
  - `args...`: Additional arguments (ignored).

# Returns

  - The input `re`, unchanged.

# Related

  - [`Regression`](@ref)
"""
function regression(re::Regression, args...)
    return re
end
"""
    regression(re::AbstractRegressionEstimator, rd::ReturnsResult)

Compute or extract a regression result from an estimator or result and a [`ReturnsResult`](@ref).

This method dispatches to `regression(re, rd.X, rd.F)`, allowing both regression estimators and regression result objects to be used interchangeably in generic workflows. If `re` is an estimator, it computes the regression result using the data in `rd`. If `re` is already a result, it is returned unchanged.

# Arguments

  - `re`: A regression estimator or result object.
  - `rd`: A returns result object containing data matrices `X` and `F`.

# Returns

  - `reg::Regression`: The computed or extracted regression result.

# Related

  - [`Regression`](@ref)
  - [`ReturnsResult`](@ref)
"""
function regression(re::AbstractRegressionEstimator, rd::ReturnsResult)
    @argcheck(!isnothing(rd.X), IsNothingError)
    @argcheck(!isnothing(rd.F), IsNothingError)
    return regression(re, rd.X, rd.F)
end

export regression, Regression, LinearModel, GeneralisedLinearModel, AIC, AICC, BIC,
       RSquared, AdjustedRSquared
