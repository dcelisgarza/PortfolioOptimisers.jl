struct FullCovariance{T1 <: AbstractExpectedReturnsEstimator,
                      T2 <: GeneralWeightedCovariance} <: AbstractPearsonCovarianceEstimator
    me::T1
    ce::T2
end
function FullCovariance(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                        ce::GeneralWeightedCovariance = GeneralWeightedCovariance())
    return FullCovariance{typeof(me), typeof(ce)}(me, ce)
end
function StatsBase.cov(ce::FullCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing,
                       kwargs...)
    mu = isnothing(mean) ? StatsBase.mean(ce.me, X; dims = dims) : mean
    return cov(ce.ce, X; dims = dims, mean = mu)
end
function StatsBase.cor(ce::FullCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing,
                       kwargs...)
    mu = isnothing(mean) ? StatsBase.mean(ce.me, X; dims = dims) : mean
    return robust_cor(ce.ce, X; dims = dims, mean = mu)
end
function w_moment_factory(ce::FullCovariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return FullCovariance(; me = w_moment_factory(ce.me, w),
                          ce = w_moment_factory(ce.ce, w))
end

export FullCovariance