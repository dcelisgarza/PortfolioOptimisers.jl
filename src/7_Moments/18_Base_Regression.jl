abstract type AbstractRegressionEstimator <: AbstractEstimator end
abstract type AbstractRegressionAlgorithm <: AbstractAlgorithm end
abstract type AbstractRegressionResult <: AbstractResult end
abstract type AbstractStepwiseRegressionAlgorithm <: AbstractRegressionAlgorithm end
abstract type AbstractStepwiseRegressionCriterion <: AbstractRegressionAlgorithm end
abstract type RegressionTarget <: AbstractRegressionAlgorithm end
struct LinearModel{T1 <: NamedTuple} <: RegressionTarget
    kwargs::T1
end
function LinearModel(; kwargs::NamedTuple = (;))
    return LinearModel{typeof(kwargs)}(kwargs)
end
function GLM.fit(target::LinearModel, X::AbstractMatrix, y::AbstractVector)
    return GLM.fit(GLM.LinearModel, X, y; target.kwargs...)
end
struct GeneralisedLinearModel{T1 <: Tuple, T2 <: NamedTuple} <: RegressionTarget
    args::T1
    kwargs::T2
end
function GeneralisedLinearModel(; args::Tuple = (Normal(),), kwargs::NamedTuple = (;))
    return GeneralisedLinearModel{typeof(args), typeof(kwargs)}(args, kwargs)
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
struct RegressionResult{T1 <: AbstractMatrix, T2 <: Union{Nothing, <:AbstractMatrix},
                        T3 <: Union{Nothing, <:AbstractVector}} <: AbstractRegressionResult
    M::T1
    L::T2
    b::T3
end
function RegressionResult(; M::AbstractMatrix,
                          L::Union{Nothing, <:AbstractMatrix} = nothing,
                          b::Union{Nothing, <:AbstractVector})
    @smart_assert(!isempty(M))
    if isa(b, AbstractVector)
        @smart_assert(!isempty(b))
        @smart_assert(length(b) == size(M, 1))
    end
    if !isnothing(L)
        @smart_assert(size(M, 1) == size(L, 1))
    end
    return RegressionResult{typeof(M), typeof(L), typeof(b)}(M, L, b)
end
function Base.getproperty(r::RegressionResult{<:Any, Nothing, <:Any}, sym::Symbol)
    return if sym == :L
        getfield(r, :M)
    else
        getfield(r, sym)
    end
end
function Base.getproperty(r::RegressionResult{<:Any, <:AbstractMatrix, <:Any}, sym::Symbol)
    return if sym == :L
        getfield(r, :L)
    else
        getfield(r, sym)
    end
end
function regression_view(r::RegressionResult, i::AbstractVector)
    return RegressionResult(; M = view(r.M, i, :),
                            L = isnothing(r.L) ? nothing : view(r.L, i, :),
                            b = view(r.b, i))
end
function regression_view(::Nothing, args...)
    return nothing
end
function regression_view(r::AbstractRegressionEstimator, args...)
    return r
end
function regression(re::RegressionResult, args...)
    return re
end

export regression, RegressionResult
