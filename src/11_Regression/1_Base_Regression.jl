abstract type AbstractRegressionEstimator <: AbstractEstimator end
abstract type AbstractRegressionAlgorithm <: AbstractAlgorithm end
abstract type AbstractRegressionResult <: AbstractResult end
struct RegressionResult{T1 <: Union{Nothing, <:AbstractVector}, T2 <: AbstractMatrix} <:
       AbstractRegressionResult
    b::T1
    M::T2
end
function RegressionResult(; b::Union{Nothing, <:AbstractVector}, M::AbstractMatrix)
    @smart_assert(!isempty(M))
    if isa(b, AbstractVector)
        @smart_assert(!isempty(b))
        @smart_assert(length(b) == size(M, 1))
    end
    return RegressionResult{typeof(b), typeof(M)}(b, M)
end
function regression_view(r::RegressionResult, i::AbstractVector)
    return RegressionResult(; b = view(r.b, i), M = view(r.M, i, :))
end
function regression_view(r::AbstractRegressionEstimator, args...)
    return r
end
function regression end

export regression, RegressionResult
