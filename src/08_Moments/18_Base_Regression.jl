"""
    abstract type AbstractRegressionEstimator <: AbstractEstimator end

Abstract supertype for all regression estimator types in PortfolioOptimisers.jl.

All concrete types implementing regression estimation algorithms should subtype `AbstractRegressionEstimator`. This enables a consistent interface for regression-based moment estimation throughout the package.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractRegressionAlgorithm`](@ref)
  - [`AbstractRegressionResult`](@ref)
"""
abstract type AbstractRegressionEstimator <: AbstractEstimator end
"""
    abstract type AbstractRegressionResult <: AbstractResult end

Abstract supertype for all regression result types in PortfolioOptimisers.jl.

All concrete types representing the output of regression-based moment estimation should subtype `AbstractRegressionResult`. This enables a consistent interface for handling regression results, such as fitted parameters, rr, and intercepts, throughout the package.

# Related

  - [`AbstractResult`](@ref)
  - [`Regression`](@ref)
  - [`AbstractRegressionEstimator`](@ref)
"""
abstract type AbstractRegressionResult <: AbstractResult end
"""
    abstract type AbstractRegressionAlgorithm <: AbstractAlgorithm end

Abstract supertype for all regression algorithm types in PortfolioOptimisers.jl.

All concrete types implementing specific regression algorithms should subtype `AbstractRegressionAlgorithm`. This enables flexible extension and dispatch of regression routines for moment estimation.

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
    abstract type AbstractStepwiseRegressionAlgorithm <: AbstractRegressionAlgorithm end

Abstract supertype for all stepwise regression algorithm types in PortfolioOptimisers.jl.

All concrete types implementing stepwise regression algorithms should subtype `AbstractStepwiseRegressionAlgorithm`. This enables modular extension and dispatch for stepwise regression routines, commonly used for variable selection and model refinement in regression-based moment estimation.

# Related

  - [`AbstractRegressionAlgorithm`](@ref)
  - [`AbstractStepwiseRegressionCriterion`](@ref)
  - [`AbstractRegressionTarget`](@ref)
"""
abstract type AbstractStepwiseRegressionAlgorithm <: AbstractRegressionAlgorithm end
"""
    abstract type AbstractStepwiseRegressionCriterion <: AbstractRegressionAlgorithm end

Abstract supertype for all stepwise regression criterion types in PortfolioOptimisers.jl.

All concrete types representing criteria for stepwise regression algorithms should subtype `AbstractStepwiseRegressionCriterion`. These criteria are used to evaluate model quality and guide variable selection during stepwise regression, such as AIC, BIC, or R².

# Related

  - [`AbstractStepwiseRegressionAlgorithm`](@ref)
  - [`AbstractRegressionTarget`](@ref)
"""
abstract type AbstractStepwiseRegressionCriterion <: AbstractRegressionAlgorithm end
"""
    abstract type AbstractRegressionTarget <: AbstractRegressionAlgorithm end

Abstract supertype for all regression target types in PortfolioOptimisers.jl.

All concrete types representing regression targets (such as linear or generalised linear models) should subtype `AbstractRegressionTarget`. This enables flexible specification and dispatch of regression targets when constructing regression estimators.

# Related

  - [`AbstractRegressionAlgorithm`](@ref)
"""
abstract type AbstractRegressionTarget <: AbstractRegressionAlgorithm end
"""
    struct LinearModel{T1} <: AbstractRegressionTarget
        kwargs::T1
    end

Regression target type for standard linear models in PortfolioOptimisers.jl.

`LinearModel` is used to specify a standard linear regression target (i.e., ordinary least squares) when constructing regression estimators. It encapsulates keyword arguments for configuring the underlying linear model fitting routine, enabling flexible extension and dispatch within the regression estimation framework.

# Fields

  - `kwargs`: Keyword arguments to be passed to the linear model fitting routine (e.g., options for the solver or regularisation).

# Constructor

    LinearModel(; kwargs::NamedTuple = (;))

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> LinearModel()
LinearModel
  kwargs | @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractRegressionTarget`](@ref)
  - [`GeneralisedLinearModel`](@ref)
  - [`StatsAPI.fit(::LinearModel, ::AbstractMatrix, ::AbstractVector)`](@ref)
"""
struct LinearModel{T1} <: AbstractRegressionTarget
    kwargs::T1
    function LinearModel(kwargs::NamedTuple)
        return new{typeof(kwargs)}(kwargs)
    end
end
function LinearModel(; kwargs::NamedTuple = (;))
    return LinearModel(kwargs)
end
"""
    StatsAPI.fit(target::LinearModel, X::AbstractMatrix, y::AbstractVector)

Fit a standard linear regression model using a [`LinearModel`](@ref) regression target.

This method dispatches to `StatsAPI.fit` with the `GLM.LinearModel` type, passing the design matrix `X`, response vector `y`, and any keyword arguments stored in `target.kwargs`. It enables flexible configuration of the underlying linear model fitting routine within the PortfolioOptimisers.jl regression estimation framework.

# Arguments

  - `target`: Regression target specifying model options.
  - `X`: The design matrix (observations × features).
  - `y`: The response vector.

# Returns

  - `GLM.LinearModel`: A fitted linear model object from the GLM.jl package.

# Related

  - [`LinearModel`](@ref)
  - [`GLM.LinearModel`](https://juliastats.org/GLM.jl/stable/api/#GLM.LinearModel)
"""
function StatsAPI.fit(target::LinearModel, X::AbstractMatrix, y::AbstractVector)
    return GLM.fit(GLM.LinearModel, X, y; target.kwargs...)
end
"""
    struct GeneralisedLinearModel{T1, T2} <: AbstractRegressionTarget
        args::T1
        kwargs::T2
    end

Regression target type for generalised linear models (GLMs) in PortfolioOptimisers.jl.

`GeneralisedLinearModel` is used to specify a generalised linear regression target (e.g., logistic, Poisson, etc.) when constructing regression estimators. It encapsulates positional and keyword arguments for configuring the underlying GLM fitting routine, enabling flexible extension and dispatch within the regression estimation framework.

# Fields

  - `args`: Positional arguments to be passed to the GLM fitting routine (e.g., distribution and link).
  - `kwargs`: Keyword arguments for the GLM fitting routine (e.g., solver options, regularisation).

# Constructor

    GeneralisedLinearModel(; args::Tuple = (Normal(),), kwargs::NamedTuple = (;))

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> GeneralisedLinearModel()
GeneralisedLinearModel
    args | Tuple{Distributions.Normal{Float64}}: (Distributions.Normal{Float64}(μ=0.0, σ=1.0),)
  kwargs | @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractRegressionTarget`](@ref)
  - [`LinearModel`](@ref)
  - [`StatsAPI.fit(::GeneralisedLinearModel, ::AbstractMatrix, ::AbstractVector)`](@ref)
"""
struct GeneralisedLinearModel{T1, T2} <: AbstractRegressionTarget
    args::T1
    kwargs::T2
    function GeneralisedLinearModel(args::Tuple, kwargs::NamedTuple)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function GeneralisedLinearModel(; args::Tuple = (Normal(),), kwargs::NamedTuple = (;))
    return GeneralisedLinearModel(args, kwargs)
end
"""
    StatsAPI.fit(target::GeneralisedLinearModel, X::AbstractMatrix, y::AbstractVector)

Fit a generalised linear regression model using a [`GeneralisedLinearModel`](@ref) regression target.

This method dispatches to `StatsAPI.fit` with the `GLM.GeneralizedLinearModel` type, passing the design matrix `X`, response vector `y`, any positional arguments in `target.args`, and any keyword arguments in `target.kwargs`. This enables flexible configuration of the underlying GLM fitting routine within the PortfolioOptimisers.jl regression estimation framework.

# Arguments

  - `target`: A [`GeneralisedLinearModel`](@ref) regression target specifying model options.
  - `X`: The design matrix (observations × features).
  - `y`: The response vector.

# Returns

  - `GLM.GeneralizedLinearModel`: A fitted generalised linear model object from the GLM.jl package.

# Related

  - [`GeneralisedLinearModel`](@ref)
  - [`GLM.GeneralizedLinearModel`](https://juliastats.org/GLM.jl/stable/examples/#Probit-regression)
"""
function StatsAPI.fit(target::GeneralisedLinearModel, X::AbstractMatrix, y::AbstractVector)
    return GLM.fit(GLM.GeneralizedLinearModel, X, y, target.args...; target.kwargs...)
end
"""
    abstract type AbstractMinValStepwiseRegressionCriterion <:
                  AbstractStepwiseRegressionCriterion end

Abstract supertype for all stepwise regression criteria where lower values indicate better model fit in PortfolioOptimisers.jl.

All concrete types implementing minimisation-based stepwise regression criteria (such as AIC, AICC, or BIC) should subtype `AbstractMinValStepwiseRegressionCriterion`. These criteria are used to guide variable selection in stepwise regression algorithms by minimising the criterion value.

# Related

  - [`AbstractStepwiseRegressionCriterion`](@ref)
  - [`AIC`](@ref)
  - [`AICC`](@ref)
  - [`BIC`](@ref)
"""
abstract type AbstractMinValStepwiseRegressionCriterion <:
              AbstractStepwiseRegressionCriterion end
"""
    abstract type AbstractMaxValStepwiseRegressionCriteria <:
                  AbstractStepwiseRegressionCriterion end

Abstract supertype for all stepwise regression criteria where higher values indicate better model fit in PortfolioOptimisers.jl.

All concrete types implementing maximisation-based stepwise regression criteria (such as R² or Adjusted R²) should subtype `AbstractMaxValStepwiseRegressionCriteria`. These criteria are used to guide variable selection in stepwise regression algorithms by maximising the criterion value.

# Related

  - [`AbstractStepwiseRegressionCriterion`](@ref)
  - [`RSquared`](@ref)
  - [`AdjustedRSquared`](@ref)
"""
abstract type AbstractMaxValStepwiseRegressionCriteria <:
              AbstractStepwiseRegressionCriterion end
"""
    struct AIC <: AbstractMinValStepwiseRegressionCriterion end

Akaike Information Criterion (AIC) for stepwise regression in PortfolioOptimisers.jl.

`AIC` is a minimisation-based criterion used to evaluate model quality in stepwise regression algorithms. Lower values indicate better model fit, penalising model complexity to avoid overfitting.

# Related

  - [`AbstractMinValStepwiseRegressionCriterion`](@ref)
  - [`AICC`](@ref)
  - [`BIC`](@ref)
  - [`regression_criterion_func(::AIC)`](@ref)
"""
struct AIC <: AbstractMinValStepwiseRegressionCriterion end
"""
    struct AICC <: AbstractMinValStepwiseRegressionCriterion end

Corrected Akaike Information Criterion (AICC) for stepwise regression in PortfolioOptimisers.jl.

`AICC` is a minimisation-based criterion similar to AIC, but includes a correction for small sample sizes. Lower values indicate better model fit, balancing fit and complexity.

# Related

  - [`AbstractMinValStepwiseRegressionCriterion`](@ref)
  - [`AIC`](@ref)
  - [`BIC`](@ref)
  - [`regression_criterion_func(::AICC)`](@ref)
"""
struct AICC <: AbstractMinValStepwiseRegressionCriterion end
"""
    struct BIC <: AbstractMinValStepwiseRegressionCriterion end

Bayesian Information Criterion (BIC) for stepwise regression in PortfolioOptimisers.jl.

`BIC` is a minimisation-based criterion used to evaluate model quality in stepwise regression algorithms. It penalises model complexity more strongly than AIC. Lower values indicate better model fit.

# Related

  - [`AbstractMinValStepwiseRegressionCriterion`](@ref)
  - [`AIC`](@ref)
  - [`AICC`](@ref)
  - [`regression_criterion_func(::BIC)`](@ref)
"""
struct BIC <: AbstractMinValStepwiseRegressionCriterion end
"""
    struct RSquared <: AbstractMaxValStepwiseRegressionCriteria end

Coefficient of determination (R²) for stepwise regression in PortfolioOptimisers.jl.

`RSquared` is a maximisation-based criterion used to evaluate model quality in stepwise regression algorithms. Higher values indicate better model fit, representing the proportion of variance explained by the model.

# Related

  - [`AbstractMaxValStepwiseRegressionCriteria`](@ref)
  - [`AdjustedRSquared`](@ref)
  - [`regression_criterion_func(::RSquared)`](@ref)
"""
struct RSquared <: AbstractMaxValStepwiseRegressionCriteria end
"""
    struct AdjustedRSquared <: AbstractMaxValStepwiseRegressionCriteria end

Adjusted coefficient of determination (Adjusted R²) for stepwise regression in PortfolioOptimisers.jl.

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
function regression_threshold(::AbstractMinValStepwiseRegressionCriterion)
    return Inf
end
function regression_threshold(::AbstractMaxValStepwiseRegressionCriteria)
    return -Inf
end
"""
    struct Regression{T1, T2, T3} <: AbstractRegressionResult
        M::T1
        L::T2
        b::T3
    end

Container type for regression results in PortfolioOptimisers.jl.

`Regression` stores the results of a regression-based moment estimation, including the main coefficient matrix, an optional auxiliary matrix, and the intercept vector. This type is used as the standard output for regression estimators, enabling consistent downstream processing and analysis.

# Fields

  - `M`: Main coefficient matrix (e.g., regression weights or loadings).
  - `L`: Optional auxiliary matrix for recovering lost dimensions in dimensionality reduction regressions.
  - `b`: Optional intercept vector.

# Constructor

    Regression(; M::AbstractMatrix, L::Union{Nothing, <:AbstractMatrix} = nothing,
               b::Union{Nothing, <:AbstractVector} = nothing)

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(M)`.
  - If provided, `!isempty(b)`, and `length(b) == size(M, 1)`.
  - If provided, `!isempty(L)`, and `size(L, 1) == size(M, 1)`.

# Examples

```jldoctest
julia> Regression(; M = [1 2 3; 4 5 6], L = [1 2 3 4; 5 6 7 8], b = [1, 2])
Regression
  M | 2×3 Matrix{Int64}
  L | 2×4 Matrix{Int64}
  b | Vector{Int64}: [1, 2]
```

# Related

  - [`AbstractRegressionResult`](@ref)
"""
struct Regression{T1, T2, T3} <: AbstractRegressionResult
    M::T1
    L::T2
    b::T3
    function Regression(M::AbstractMatrix, L::Union{Nothing, <:AbstractMatrix},
                        b::Union{Nothing, <:AbstractVector})
        @argcheck(!isempty(M))
        if isa(b, AbstractVector)
            @argcheck(!isempty(b))
            @argcheck(length(b) == size(M, 1))
        end
        if !isnothing(L)
            @argcheck(size(L, 1) == size(M, 1))
        end
        return new{typeof(M), typeof(L), typeof(b)}(M, L, b)
    end
end
function Regression(; M::AbstractMatrix, L::Union{Nothing, <:AbstractMatrix} = nothing,
                    b::Union{Nothing, <:AbstractVector} = nothing)
    return Regression(M, L, b)
end
function Base.getproperty(re::Regression{<:Any, Nothing, <:Any}, sym::Symbol)
    return if sym == :L
        getfield(re, :M)
    else
        getfield(re, sym)
    end
end
function Base.getproperty(re::Regression{<:Any, <:AbstractMatrix, <:Any}, sym::Symbol)
    return if sym == :L
        getfield(re, :L)
    else
        getfield(re, sym)
    end
end
"""
    regression_view(re::Regression, i::AbstractVector)

Return a view of a [`Regression`](@ref) result object, selecting only the rows indexed by `i`.

This function constructs a new `Regression` result, where the coefficient matrix `M`, optional auxiliary matrix `L`, and intercept vector `b` are restricted to the rows specified by the index vector `i`. This is useful for extracting or operating on a subset of regression results, such as for a subset of assets or features.

# Arguments

  - `re`: A regression result object.
  - `i`: Indices of the rows to select.

# Returns

  - `Regression`: A new regression result object with fields restricted to the selected rows.

# Examples

```jldoctest
julia> re = Regression(; M = [1 2; 3 4; 5 6], L = [10 20; 30 40; 50 60], b = [7, 8, 9])
Regression
  M | 3×2 Matrix{Int64}
  L | 3×2 Matrix{Int64}
  b | Vector{Int64}: [7, 8, 9]

julia> PortfolioOptimisers.regression_view(re, [1, 3])
Regression
  M | 2×2 SubArray{Int64, 2, Matrix{Int64}, Tuple{Vector{Int64}, Base.Slice{Base.OneTo{Int64}}}, false}
  L | 2×2 SubArray{Int64, 2, Matrix{Int64}, Tuple{Vector{Int64}, Base.Slice{Base.OneTo{Int64}}}, false}
  b | SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}: [7, 9]
```

# Related

  - [`Regression`](@ref)
"""
function regression_view(re::Regression, i::AbstractVector)
    return Regression(; M = view(re.M, i, :),
                      L = isnothing(re.L) ? nothing : view(re.L, i, :), b = view(re.b, i))
end
"""
    regression_view(re::Union{Nothing, <:AbstractRegressionEstimator}, args...)

No-op fallback for `regression_view` when the input is `nothing` or an `AbstractRegressionEstimator`.

This method returns the input `re` unchanged. It is used internally to allow generic code to call `regression_view` without needing to check for `nothing` or estimator types, ensuring graceful handling of missing or non-result regression objects.

# Arguments

  - `re`: Either `nothing` or a regression estimator type.
  - `args...`: Additional arguments (ignored).

# Returns

  - The input `re`, unchanged.

# Related

  - [`regression_view(::Regression, ::AbstractVector)`](@ref)
"""
function regression_view(re::Union{Nothing, <:AbstractRegressionEstimator}, args...)
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

  - `Regression`: The computed or extracted regression result.

# Related

  - [`Regression`](@ref)
  - [`ReturnsResult`](@ref)
"""
function regression(re::AbstractRegressionEstimator, rd::ReturnsResult)
    return regression(re, rd.X, rd.F)
end

export regression, Regression, LinearModel, GeneralisedLinearModel, AIC, AICC, BIC,
       RSquared, AdjustedRSquared
