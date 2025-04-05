struct SemiCovariance{T1 <: AbstractExpectedReturnsEstimator,
                      T2 <: GeneralWeightedCovariance} <: AbstractPearsonCovarianceEstimator
    me::T1
    ce::T2
end
function SemiCovariance(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                        ce::GeneralWeightedCovariance = GeneralWeightedCovariance())
    return SemiCovariance{typeof(me), typeof(ce)}(me, ce)
end
function StatsBase.cov(ce::SemiCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing,
                       kwargs...)
    mu = isnothing(mean) ? StatsBase.mean(ce.me, X; dims = dims) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return cov(ce.ce, X; dims = dims, mean = zero(eltype(X)))
end
function StatsBase.cor(ce::SemiCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing,
                       kwargs...)
    mu = isnothing(mean) ? mean(ce.me, X; dims = dims) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return robust_cor(ce.ce, X; dims = dims, mean = zero(eltype(X)))
end
function w_moment_factory(ce::SemiCovariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return SemiCovariance(; me = w_moment_factory(ce.me, w),
                          ce = w_moment_factory(ce.ce, w))
end

export SemiCovariance