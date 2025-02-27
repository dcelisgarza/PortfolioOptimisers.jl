struct FullCovariance{T1 <: ExpectedReturnsEstimator, T2 <: GeneralWeightedCovariance} <:
       PearsonCovarianceEstimator
    me::T1
    ce::T2
end
function FullCovariance(; me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                        ce::GeneralWeightedCovariance = GeneralWeightedCovariance())
    return FullCovariance{typeof(me), typeof(ce)}(me, ce)
end
function StatsBase.cov(ce::FullCovariance, X::AbstractMatrix; dims::Int = 1)
    return cov(ce.ce, X; dims = dims, mean = mean(ce.me, X; dims = dims))
end
function StatsBase.cor(ce::FullCovariance, X::AbstractMatrix; dims::Int = 1)
    mu = mean(ce.me, X; dims = dims)
    try
        cor(ce.ce, X; dims = dims, mean = mu)
    catch
        sigma = cov(ce.ce, X; dims = dims, mean = mu)
        isa(sigma, Matrix) ? StatsBase.cov2cor(sigma) : StatsBase.cov2cor(Matrix(sigma))
    end
end

export FullCovariance
