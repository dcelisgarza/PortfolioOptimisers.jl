abstract type AbstractRegressionEstimator <: AbstractEstimator end
abstract type AbstractRegressionAlgorithm <: AbstractAlgorithm end
abstract type AbstractRegressionResult <: AbstractResult end
struct RegressionResult{T1 <: Union{Nothing, <:AbstractVector}, T2 <: AbstractMatrix,
                        T3 <: Union{Nothing, <:AbstractMatrix}} <: AbstractRegressionResult
    b::T1
    M::T2
    L::T3
end
function RegressionResult(; b::Union{Nothing, <:AbstractVector}, M::AbstractMatrix,
                          L::Union{Nothing, <:AbstractMatrix} = nothing)
    @smart_assert(!isempty(M))
    if isa(b, AbstractVector)
        @smart_assert(!isempty(b))
        @smart_assert(length(b) == size(M, 1))
    end
    if !isnothing(L)
        @smart_assert(size(M, 1) == size(L, 1))
    end
    return RegressionResult{typeof(b), typeof(M), typeof(L)}(b, M, L)
end
function Base.getproperty(r::RegressionResult, sym::Symbol)
    return if sym == :frc
        isnothing(r.L) ? r.M : r.L
    else
        getfield(r, sym)
    end
end
function regression_view(r::RegressionResult, i::AbstractVector)
    return RegressionResult(; b = view(r.b, i), M = view(r.M, i, :),
                            L = isnothing(r.L) ? nothing : view(r.L, i, :))
end
function regression_view(r::AbstractRegressionEstimator, args...)
    return r
end
function regression(re::RegressionResult, args...)
    return re
end

export regression, RegressionResult
