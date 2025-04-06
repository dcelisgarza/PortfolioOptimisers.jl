struct GeneralWeightedCovariance{T1 <: StatsBase.CovarianceEstimator,
                                 T2 <: Union{Nothing, <:AbstractWeights}} <:
       AbstractCovarianceEstimator
    ce::T1
    w::T2
end
function GeneralWeightedCovariance(;
                                   ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                                                  corrected = true),
                                   w::Union{Nothing, <:AbstractWeights} = nothing)
    return GeneralWeightedCovariance{typeof(ce), typeof(w)}(ce, w)
end
function StatsBase.cov(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1,
                       mean = nothing, kwargs...)
    return if isnothing(ce.w)
        cov(ce.ce, X; dims = dims, mean = mean)
    else
        cov(ce.ce, X, ce.w; dims = dims, mean = mean)
    end
end
function StatsBase.cor(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1,
                       mean = nothing, kwargs...)
    if isnothing(ce.w)
        robust_cor(ce.ce, X; dims = dims, mean = mean)
    else
        robust_cor(ce.ce, X, ce.w; dims = dims, mean = mean)
    end
end
function factory(ce::GeneralWeightedCovariance,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return GeneralWeightedCovariance(; ce = ce.ce, w = isnothing(w) ? ce.w : w)
end
struct Covariance{T1 <: AbstractMomentAlgorithm, T2 <: AbstractExpectedReturnsEstimator,
                  T3 <: GeneralWeightedCovariance} <: AbstractCovarianceEstimator
    alg::T1
    me::T2
    ce::T3
end
function Covariance(; alg::AbstractMomentAlgorithm = Full(),
                    me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    ce::GeneralWeightedCovariance = GeneralWeightedCovariance())
    return Covariance{typeof(alg), typeof(me), typeof(ce)}(alg, me, ce)
end
function factory(ce::Covariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return Covariance(; alg = ce.alg, me = factory(ce.me, w), ce = factory(ce.ce, w))
end
function StatsBase.cov(ce::Covariance{<:Full, <:Any, <:Any}, X::AbstractMatrix;
                       dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? StatsBase.mean(ce.me, X; dims = dims) : mean
    return cov(ce.ce, X; dims = dims, mean = mu)
end
function StatsBase.cor(ce::Covariance{<:Full, <:Any, <:Any}, X::AbstractMatrix;
                       dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? StatsBase.mean(ce.me, X; dims = dims) : mean
    return robust_cor(ce.ce, X; dims = dims, mean = mu)
end
function StatsBase.cov(ce::Covariance{<:Semi, <:Any, <:Any}, X::AbstractMatrix;
                       dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? StatsBase.mean(ce.me, X; dims = dims) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return cov(ce.ce, X; dims = dims, mean = zero(eltype(X)))
end
function StatsBase.cor(ce::Covariance{<:Semi, <:Any, <:Any}, X::AbstractMatrix;
                       dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? StatsBase.mean(ce.me, X; dims = dims) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return robust_cor(ce.ce, X; dims = dims, mean = zero(eltype(X)))
end

export GeneralWeightedCovariance, Covariance
