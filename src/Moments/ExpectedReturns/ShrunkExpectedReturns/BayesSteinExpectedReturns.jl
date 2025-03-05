struct BayesSteinExpectedReturns{T1 <: StatsBase.CovarianceEstimator,
                                 T2 <: ExpectedReturnsEstimator,
                                 T3 <: ShrunkExpectedReturnsTarget} <:
       ShrunkExpectedReturnsEstimator
    ce::T1
    me::T2
    target::T3
end
function BayesSteinExpectedReturns(;
                                   ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                   me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                                   target::ShrunkExpectedReturnsTarget = SERT_GrandMean())
    return BayesSteinExpectedReturns{typeof(ce), typeof(me), typeof(target)}(ce, me, target)
end
function StatsBase.mean(me::BayesSteinExpectedReturns, X::AbstractMatrix; dims::Int = 1)
    mu = mean(me.me, X; dims = dims)
    sigma = cov(me.ce, X; dims = dims)
    T, N = size(X)
    isigma = sigma \ I
    b = if isone(dims)
        transpose(target_mean(me.target, transpose(mu), sigma; isigma = isigma, T = T))
    else
        target_mean(me.target, mu, sigma; isigma = isigma, T = T)
    end
    mb = vec(mu - b)
    alpha = (N + 2) / ((N + 2) + T * dot(mb, isigma, mb))
    return (one(alpha) - alpha) * mu + alpha * b
end

export BayesSteinExpectedReturns
