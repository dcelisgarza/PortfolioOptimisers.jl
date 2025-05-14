abstract type AbstractRegressionEstimator <: AbstractEstimator end
abstract type AbstractRegressionAlgorithm <: AbstractAlgorithm end
abstract type AbstractRegressionResult <: AbstractResult end
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
function Base.getproperty(r::RegressionResult, sym::Symbol)
    return if sym == :frc
        isnothing(r.L) ? r.M : r.L
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
