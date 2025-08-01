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

All concrete types representing the output of regression-based moment estimation should subtype `AbstractRegressionResult`. This enables a consistent interface for handling regression results, such as fitted parameters, loadings, and intercepts, throughout the package.

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
struct LinearModel{T1} <: AbstractRegressionTarget
    kwargs::T1
end
function LinearModel(; kwargs::NamedTuple = (;))
    return LinearModel(kwargs)
end
function GLM.fit(target::LinearModel, X::AbstractMatrix, y::AbstractVector)
    return GLM.fit(GLM.LinearModel, X, y; target.kwargs...)
end
struct GeneralisedLinearModel{T1, T2} <: AbstractRegressionTarget
    args::T1
    kwargs::T2
end
function GeneralisedLinearModel(; args::Tuple = (Normal(),), kwargs::NamedTuple = (;))
    return GeneralisedLinearModel(args, kwargs)
end
function GLM.fit(target::GeneralisedLinearModel, X::AbstractMatrix, y::AbstractVector)
    return GLM.fit(GLM.GeneralizedLinearModel, X, y, target.args...; target.kwargs...)
end
abstract type AbstractMinValStepwiseRegressionCriterion <:
              AbstractStepwiseRegressionCriterion end
abstract type AbstractMaxValStepwiseRegressionCriteria <:
              AbstractStepwiseRegressionCriterion end
struct AIC <: AbstractMinValStepwiseRegressionCriterion end
struct AICC <: AbstractMinValStepwiseRegressionCriterion end
struct BIC <: AbstractMinValStepwiseRegressionCriterion end
struct RSquared <: AbstractMaxValStepwiseRegressionCriteria end
struct AdjustedRSquared <: AbstractMaxValStepwiseRegressionCriteria end
function regression_criterion_func(::AIC)
    return GLM.aic
end
function regression_criterion_func(::AICC)
    return GLM.aicc
end
function regression_criterion_func(::BIC)
    return GLM.bic
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
"""
struct Regression{T1, T2, T3} <: AbstractRegressionResult
    M::T1
    L::T2
    b::T3
end
function Regression(; M::AbstractMatrix, L::Union{Nothing, <:AbstractMatrix} = nothing,
                    b::Union{Nothing, <:AbstractVector})
    @smart_assert(!isempty(M))
    if isa(b, AbstractVector)
        @smart_assert(!isempty(b))
        @smart_assert(length(b) == size(M, 1))
    end
    if !isnothing(L)
        @smart_assert(size(M, 1) == size(L, 1))
    end
    return Regression(M, L, b)
end
function Base.getproperty(r::Regression{<:Any, Nothing, <:Any}, sym::Symbol)
    return if sym == :L
        getfield(r, :M)
    else
        getfield(r, sym)
    end
end
function Base.getproperty(r::Regression{<:Any, <:AbstractMatrix, <:Any}, sym::Symbol)
    return if sym == :L
        getfield(r, :L)
    else
        getfield(r, sym)
    end
end
function regression_view(r::Regression, i::AbstractVector)
    return Regression(; M = view(r.M, i, :), L = isnothing(r.L) ? nothing : view(r.L, i, :),
                      b = view(r.b, i))
end
function regression_view(::Nothing, args...)
    return nothing
end
function regression_view(r::AbstractRegressionEstimator, args...)
    return r
end
function regression(re::Regression, args...)
    return re
end

export regression, Regression
