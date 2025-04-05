abstract type AbstractPearsonCovarianceEstimator <: AbstractCovarianceEstimator end
struct GeneralWeightedCovariance{T1 <: StatsBase.CovarianceEstimator,
                                 T2 <: Union{Nothing, <:AbstractWeights}} <:
       AbstractPearsonCovarianceEstimator
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
function w_moment_factory(ce::GeneralWeightedCovariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return GeneralWeightedCovariance(; ce = ce.ce, w = w)
end

export GeneralWeightedCovariance
