struct BayesSteinExpectedReturns{T1 <: StatsBase.CovarianceEstimator,
                                 T2 <: ShrunkExpectedReturnsTarget,
                                 T3 <: Union{Nothing, <:AbstractWeights}} <:
       ShrunkExpectedReturnsEstimator
    ce::T1
    target::T2
    w::T3
end
function BayesSteinExpectedReturns(;
                                   ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                   target::ShrunkExpectedReturnsTarget = SERT_GrandMean(),
                                   w::Union{Nothing, <:AbstractWeights} = nothing)
    return BayesSteinExpectedReturns{typeof(ce), typeof(target), typeof(w)}(ce, target, w)
end
function StatsBase.mean(me::BayesSteinExpectedReturns, X::AbstractMatrix; dims::Int = 1)
    mu = isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims)
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
