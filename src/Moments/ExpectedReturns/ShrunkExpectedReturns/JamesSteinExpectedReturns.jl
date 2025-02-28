struct JamesSteinExpectedReturns{T1 <: StatsBase.CovarianceEstimator,
                                 T2 <: ShrunkExpectedReturnsTarget,
                                 T3 <: Union{Nothing, <:AbstractWeights}} <:
       ShrunkExpectedReturnsEstimator
    ce::T1
    target::T2
    w::T3
end
function JamesSteinExpectedReturns(;
                                   ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                   target::ShrunkExpectedReturnsTarget = SERT_GrandMean(),
                                   w::Union{Nothing, <:AbstractWeights} = nothing)
    return JamesSteinExpectedReturns{typeof(ce), typeof(target), typeof(w)}(ce, target, w)
end
function StatsBase.mean(me::JamesSteinExpectedReturns, X::AbstractMatrix; dims::Int = 1)
    mu = isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims)
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
