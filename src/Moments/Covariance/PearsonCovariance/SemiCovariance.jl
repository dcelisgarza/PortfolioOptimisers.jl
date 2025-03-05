struct SemiCovariance{T1 <: ExpectedReturnsEstimator, T2 <: GeneralWeightedCovariance} <:
       PearsonCovarianceEstimator
    me::T1
    ce::T2
end
function SemiCovariance(; me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                        ce::GeneralWeightedCovariance = GeneralWeightedCovariance())
    return SemiCovariance{typeof(me), typeof(ce)}(me, ce)
end
function StatsBase.cov(ce::SemiCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    mu = mean(ce.me, X; dims = dims, kwargs...)
    X = min.(X .- mu, zero(eltype(X)))
    return cov(ce.ce, X; dims = dims, mean = zero(eltype(X)), kwargs...)
end
function StatsBase.cor(ce::SemiCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    mu = mean(ce.me, X; dims = dims, kwargs...)
    X = min.(X .- mu, zero(eltype(X)))
    return robust_cor(ce.ce, X; dims = dims, mean = zero(eltype(X)), kwargs...)
end

export SemiCovariance
