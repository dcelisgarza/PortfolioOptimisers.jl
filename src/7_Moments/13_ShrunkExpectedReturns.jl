abstract type AbstractShrunkExpectedReturnsEstimator <: AbstractExpectedReturnsEstimator end
abstract type AbstractShrunkExpectedReturnsAlgorithm <: AbstractExpectedReturnsAlgorithm end
abstract type AbstractShrunkExpectedReturnsTarget <: AbstractExpectedReturnsAlgorithm end
struct JamesStein <: AbstractShrunkExpectedReturnsAlgorithm end
struct BayesStein <: AbstractShrunkExpectedReturnsAlgorithm end
struct BodnarOkhrinParolya <: AbstractShrunkExpectedReturnsAlgorithm end
struct GrandMean <: AbstractShrunkExpectedReturnsTarget end
struct VolatilityWeighted <: AbstractShrunkExpectedReturnsTarget end
struct MeanSquareError <: AbstractShrunkExpectedReturnsTarget end

struct ShrunkExpectedReturns{T1 <: AbstractExpectedReturnsEstimator,
                             T2 <: StatsBase.CovarianceEstimator,
                             T3 <: AbstractShrunkExpectedReturnsAlgorithm,
                             T4 <: AbstractShrunkExpectedReturnsTarget} <:
       AbstractShrunkExpectedReturnsEstimator
    me::T1
    ce::T2
    alg::T3
    target::T4
end
function ShrunkExpectedReturns(;
                               me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                               ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                               alg::AbstractShrunkExpectedReturnsAlgorithm = JamesStein(),
                               target::AbstractShrunkExpectedReturnsTarget = GrandMean())
    return ShrunkExpectedReturns{typeof(me), typeof(ce), typeof(alg), typeof(target)}(me,
                                                                                      ce,
                                                                                      alg,
                                                                                      target)
end
function target_mean(::GrandMean, mu::AbstractArray, sigma::AbstractMatrix; kwargs...)
    return fill(mean(mu), length(mu))
end
function target_mean(::VolatilityWeighted, mu::AbstractArray, sigma::AbstractMatrix;
                     isigma = nothing, kwargs...)
    if isnothing(isigma)
        isigma = sigma \ I
    end
    return fill(sum(isigma * mu) / sum(isigma), length(mu))
end
function target_mean(::MeanSquareError, mu::AbstractArray, sigma::AbstractMatrix;
                     T::Integer, kwargs...)
    return fill(tr(sigma) / T, length(mu))
end
function StatsBase.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein, <:Any},
                        X::AbstractMatrix; dims::Int = 1)
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
    return (one(alpha) - alpha) * mu + alpha * b
end
function StatsBase.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein, <:Any},
                        X::AbstractMatrix; dims::Int = 1)
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
function StatsBase.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya,
                                                  <:Any}, X::AbstractMatrix; dims::Int = 1)
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
    beta = (one(alpha) - alpha) * w / u
    return alpha * mu + beta * b
end
function factory(ce::ShrunkExpectedReturns, w::Union{Nothing, <:AbstractWeights} = nothing)
    return ShrunkExpectedReturns(; me = factory(ce.me, w), ce = factory(ce.ce, w),
                                 target = ce.target)
end

export GrandMean, VolatilityWeighted, MeanSquareError, JamesStein, BayesStein,
       BodnarOkhrinParolya, ShrunkExpectedReturns
