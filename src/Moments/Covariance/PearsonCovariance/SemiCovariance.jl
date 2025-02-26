struct SemiCovariance{T1 <: ExpectedReturnsEstimator, T2 <: GeneralWeightedCovariance} <:
       PearsonCovarianceEstimator
    me::T1
    ce::T2
end
function SemiCovariance(; me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                        ce::GeneralWeightedCovariance = GeneralWeightedCovariance())
    return SemiCovariance{typeof(me), typeof(ce)}(me, ce)
end
function StatsBase.cov(ce::SemiCovariance, X::AbstractMatrix; dims::Int = 1)
    mu = mean(ce.me, X; dims = dims)
    X = min.(X .- mu, zero(eltype(X)))
    return cov(ce.ce, X; dims = dims, mean = zero(eltype(X)))
end
function StatsBase.cor(ce::SemiCovariance, X::AbstractMatrix; dims::Int = 1)
    mu = mean(ce.me, X; dims = dims)
    X = min.(X .- mu, zero(eltype(X)))
    try
        cor(ce.ce, X; dims = dims, mean = zero(eltype(X)))
    catch
        sigma = cov(ce.ce, X; dims = dims, mean = zero(eltype(X)))
        isa(sigma, Matrix) ? StatsBase.cov2cor(sigma) : StatsBase.cov2cor(Matrix(sigma))
    end
end

export SemiCovariance
