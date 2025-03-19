struct BayesSteinExpectedReturns{T1 <: ExpectedReturnsEstimator,
                                 T2 <: StatsBase.CovarianceEstimator,
                                 T3 <: ShrunkExpectedReturnsTarget} <:
       ShrunkExpectedReturnsEstimator
    me::T1
    ce::T2
    target::T3
end
function BayesSteinExpectedReturns(; me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                                   ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                   target::ShrunkExpectedReturnsTarget = SERT_GrandMean())
    return BayesSteinExpectedReturns{typeof(me), typeof(ce), typeof(target)}(me, ce, target)
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
function moment_factory_w(ce::BayesSteinExpectedReturns,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return BayesSteinExpectedReturns(; me = moment_factory_w(ce.me, w),
                                     ce = moment_factory_w(ce.ce, w), target = ce.target)
end

export BayesSteinExpectedReturns
