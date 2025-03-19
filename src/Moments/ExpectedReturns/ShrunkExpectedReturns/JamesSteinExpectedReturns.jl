struct JamesSteinExpectedReturns{T1 <: ExpectedReturnsEstimator,
                                 T2 <: StatsBase.CovarianceEstimator,
                                 T3 <: ShrunkExpectedReturnsTarget} <:
       ShrunkExpectedReturnsEstimator
    me::T1
    ce::T2
    target::T3
end
function JamesSteinExpectedReturns(; me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                                   ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                   target::ShrunkExpectedReturnsTarget = SERT_GrandMean())
    return JamesSteinExpectedReturns{typeof(me), typeof(ce), typeof(target)}(me, ce, target)
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
function moment_factory_w(ce::JamesSteinExpectedReturns,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return JamesSteinExpectedReturns(; me = moment_factory_w(ce.me, w),
                                     ce = moment_factory_w(ce.ce, w), target = ce.target)
end

export JamesSteinExpectedReturns
