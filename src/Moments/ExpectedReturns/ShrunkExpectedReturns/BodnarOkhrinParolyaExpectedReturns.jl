struct BodnarOkhrinParolyaExpectedReturns{T1 <: StatsBase.CovarianceEstimator,
                                          T2 <: ExpectedReturnsEstimator,
                                          T3 <: ShrunkExpectedReturnsTarget} <:
       ShrunkExpectedReturnsEstimator
    ce::T1
    me::T2
    target::T3
end
function BodnarOkhrinParolyaExpectedReturns(;
                                            ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                            me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                                            target::ShrunkExpectedReturnsTarget = SERT_GrandMean())
    return BodnarOkhrinParolyaExpectedReturns{typeof(ce), typeof(me), typeof(target)}(ce,
                                                                                      me,
                                                                                      target)
end
function StatsBase.mean(me::BodnarOkhrinParolyaExpectedReturns, X::AbstractMatrix;
                        dims::Int = 1)
    mu = mean(me.me, X; dims = dims)
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
