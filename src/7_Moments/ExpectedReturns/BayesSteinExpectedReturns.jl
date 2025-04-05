struct BayesSteinExpectedReturns{T1 <: AbstractExpectedReturnsEstimator,
                                 T2 <: StatsBase.CovarianceEstimator,
                                 T3 <: AbstractShrunkExpectedReturnsTarget} <:
       AbstractShrunkExpectedReturnsEstimator
    me::T1
    ce::T2
    target::T3
end
function BayesSteinExpectedReturns(;
                                   me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                                   ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                   target::AbstractShrunkExpectedReturnsTarget = GrandMean())
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
function w_moment_factory(ce::BayesSteinExpectedReturns,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return BayesSteinExpectedReturns(; me = w_moment_factory(ce.me, w),
                                     ce = w_moment_factory(ce.ce, w), target = ce.target)
end

export BayesSteinExpectedReturns
