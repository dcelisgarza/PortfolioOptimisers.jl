struct CanonicalDistance <: PortfolioOptimisersDistanceMetric end
function distance(::CanonicalDistance, ce::MutualInfoCovariance, X::AbstractMatrix;
                  dims::Int = 1, kwargs...)
    return distance(VariationInfoDistance(; bins = ce.bins, normalise = ce.normalise), ce,
                    X; dims = dims, kwargs...)
end
function distance(::CanonicalDistance, ce::LTDCovariance, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    return distance(LogDistance(), ce, X; dims = dims, kwargs...)
end
function distance(::CanonicalDistance, ce::DistanceCovariance, X::AbstractMatrix;
                  dims::Int = 1, kwargs...)
    return distance(CorrelationDistance(), ce, X; dims = dims, kwargs...)
end
function distance(::CanonicalDistance, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix;
                  dims::Int = 1, kwargs...)
    return distance(SimpleDistance(), ce, X; dims = dims, kwargs...)
end
function distance(::CanonicalDistance, rho::AbstractMatrix, args...; kwargs...)
    return distance(SimpleDistance(), rho; kwargs...)
end
export CanonicalDistance
