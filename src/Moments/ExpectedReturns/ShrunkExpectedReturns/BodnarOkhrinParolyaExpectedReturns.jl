struct BodnarOkhrinParolyaExpectedReturns{T1 <: StatsBase.CovarianceEstimator,
                                          T2 <: ShrunkExpectedReturnsTarget,
                                          T3 <: Union{Nothing, <:AbstractWeights}} <:
       ShrunkExpectedReturnsEstimator
    ce::T1
    target::T2
    w::T3
end
function BodnarOkhrinParolyaExpectedReturns(;
                                            ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                            target::ShrunkExpectedReturnsTarget = SERT_GrandMean(),
                                            w::Union{Nothing, <:AbstractWeights} = nothing)
    return BodnarOkhrinParolyaExpectedReturns{typeof(ce), typeof(target), typeof(w)}(ce,
                                                                                     target,
                                                                                     w)
end
function StatsBase.mean(me::BodnarOkhrinParolyaExpectedReturns, X::AbstractMatrix;
                        dims::Int = 1)
    mu = isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims)
    sigma = cov(me.ce, X; dims = dims)
    T, N = size(X)
    isigma = sigma \ I
    b = if isone(dims)
        transpose(target_mean(me.target, transpose(mu), sigma; isigma = isigma, T = T))
    else
        target_mean(me.target, mu, sigma; isigma = isigma, T = T)
    end
    u = dot(vec(mu), isigma, vec(mu))
    v = dot(vec(b), isigma, vec(b))
    w = dot(vec(mu), isigma, vec(b))
    alpha = (u - N / (T - N)) * v - w^2
    alpha /= u * v - w^2
    beta = (1 - alpha) * w / u
    return alpha * mu + beta * b
end

export BodnarOkhrinParolyaExpectedReturns
