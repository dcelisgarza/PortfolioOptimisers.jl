struct BodnarOkhrinParolyaExpectedReturns{T1 <: ExpectedReturnsEstimator,
                                          T2 <: StatsBase.CovarianceEstimator,
                                          T3 <: ShrunkExpectedReturnsTarget} <:
       ShrunkExpectedReturnsEstimator
    me::T1
    ce::T2
    target::T3
end
function BodnarOkhrinParolyaExpectedReturns(;
                                            me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                                            ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                            target::ShrunkExpectedReturnsTarget = SERT_GrandMean())
    return BodnarOkhrinParolyaExpectedReturns{typeof(me), typeof(ce), typeof(target)}(me,
                                                                                      ce,
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
function moment_factory_w(ce::BodnarOkhrinParolyaExpectedReturns,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return BodnarOkhrinParolyaExpectedReturns(; me = moment_factory_w(ce.me, w),
                                              ce = moment_factory_w(ce.ce, w),
                                              target = ce.target)
end

export BodnarOkhrinParolyaExpectedReturns
