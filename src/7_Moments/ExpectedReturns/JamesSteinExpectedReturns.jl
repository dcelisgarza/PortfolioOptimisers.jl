struct JamesSteinExpectedReturns{T1 <: AbstractExpectedReturnsEstimator,
                                 T2 <: StatsBase.CovarianceEstimator,
                                 T3 <: AbstractShrunkExpectedReturnsTarget} <:
       AbstractShrunkExpectedReturnsEstimator
    me::T1
    ce::T2
    target::T3
end
function JamesSteinExpectedReturns(;
                                   me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                                   ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                   target::AbstractShrunkExpectedReturnsTarget = GrandMean())
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
function w_moment_factory(ce::JamesSteinExpectedReturns,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return JamesSteinExpectedReturns(; me = w_moment_factory(ce.me, w),
                                     ce = w_moment_factory(ce.ce, w), target = ce.target)
end

export JamesSteinExpectedReturns
