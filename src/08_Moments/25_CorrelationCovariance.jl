"""
"""
struct CorrelationCovariance{T1} <: AbstractCovarianceEstimator
    ce::T1
    function CorrelationCovariance(ce::StatsBase.CovarianceEstimator)
        return new{typeof(ce)}(ce)
    end
end
function CorrelationCovariance(; ce::StatsBase.CovarianceEstimator = Covariance())
    return CorrelationCovariance(ce)
end
function factory(ce::CorrelationCovariance,
                 w::Option{<:StatsBase.AbstractWeights} = nothing)
    return CorrelationCovariance(; ce = factory(ce.ce, w))
end
function Statistics.cov(ce::CorrelationCovariance, X::AbstractMatrix{<:Real}; dims::Int = 1,
                        kwargs...)
    return Statistics.cor(ce.ce, X; dims = dims, kwargs...)
end
function Statistics.cor(ce::CorrelationCovariance, X::AbstractMatrix{<:Real}; dims::Int = 1,
                        kwargs...)
    return Statistics.cor(ce.ce, X; dims = dims, kwargs...)
end

export CorrelationCovariance
