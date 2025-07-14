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
function Statistics.cov(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, kwargs...)
    return if isnothing(ce.w)
        robust_cov(ce.ce, X; dims = dims, mean = mean, kwargs...)
    else
        robust_cov(ce.ce, X, ce.w; dims = dims, mean = mean, kwargs...)
    end
end
function Statistics.cor(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, kwargs...)
    if isnothing(ce.w)
        robust_cor(ce.ce, X; dims = dims, mean = mean, kwargs...)
    else
        robust_cor(ce.ce, X, ce.w; dims = dims, mean = mean, kwargs...)
    end
end
function factory(ce::GeneralWeightedCovariance,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return GeneralWeightedCovariance(; ce = ce.ce, w = isnothing(w) ? ce.w : w)
end
struct Covariance{T1 <: AbstractExpectedReturnsEstimator,
                  T2 <: StatsBase.CovarianceEstimator, T3 <: AbstractMomentAlgorithm} <:
       AbstractCovarianceEstimator
    me::T1
    ce::T2
    alg::T3
end
function Covariance(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    ce::StatsBase.CovarianceEstimator = GeneralWeightedCovariance(),
                    alg::AbstractMomentAlgorithm = Full())
    return Covariance{typeof(me), typeof(ce), typeof(alg)}(me, ce, alg)
end
function factory(ce::Covariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return Covariance(; me = factory(ce.me, w), ce = factory(ce.ce, w), alg = ce.alg)
end
function Statistics.cov(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    return cov(ce.ce, X; dims = dims, mean = mu, kwargs...)
end
function Statistics.cor(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    return cor(ce.ce, X; dims = dims, mean = mu, kwargs...)
end
function Statistics.cov(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return cov(ce.ce, X; dims = dims, mean = zero(eltype(X)), kwargs...)
end
function Statistics.cor(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return cor(ce.ce, X; dims = dims, mean = zero(eltype(X)), kwargs...)
end

export GeneralWeightedCovariance, Covariance
