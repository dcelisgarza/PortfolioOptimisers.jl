struct JamesSteinExpectedReturns{T1 <: StatsBase.CovarianceEstimator,
                                 T2 <: ExpectedReturnsEstimator,
                                 T3 <: ShrunkExpectedReturnsTarget} <:
       ShrunkExpectedReturnsEstimator
    ce::T1
    me::T2
    target::T3
end
function JamesSteinExpectedReturns(;
                                   ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                   me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                                   target::ShrunkExpectedReturnsTarget = SERT_GrandMean())
    return JamesSteinExpectedReturns{typeof(ce), typeof(me), typeof(target)}(ce, me, target)
end
function StatsBase.mean(me::JamesSteinExpectedReturns, X::AbstractMatrix; dims::Int = 1)
    mu = mean(me.me, X; dims = dims)
    sigma = cov(me.ce, X; dims = dims)
    T, N = size(X)
    b = if isone(dims)
        transpose(target_mean(me.target, transpose(mu), sigma; T = T))
    else
        target_mean(me.target, mu, sigma; T = T)
    end
    evals = eigvals(sigma)
    mb = mu - b
    alpha = (N * mean(evals) - 2 * maximum(evals)) / dot(mb, mb) / T
    return (1 - alpha) * mu + alpha * b
end

export JamesSteinExpectedReturns
