struct CanonicalDistance <: PortfolioOptimisersDistanceMetric end
function distance(::CanonicalDistance, ce::MutualInfoCovariance, X::AbstractMatrix;
                  dims::Int = 1)
    return distance(VariationInfoDistance(; bins = ce.bins, normalise = ce.normalise), ce,
                    X; dims = dims)
end
function distance(::CanonicalDistance, ce::LTDCovariance, X::AbstractMatrix; dims::Int = 1)
    return distance(LogDistance(), ce, X; dims = dims)
end
function distance(::CanonicalDistance, ce::DistanceCovariance, X::AbstractMatrix;
                  dims::Int = 1)
    return distance(CorrelationDistance(), ce, X; dims = dims)
end
function distance(::CanonicalDistance, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix;
                  dims::Int = 1)
    return distance(SimpleDistance(), ce, X; dims = dims)
end

export CanonicalDistance
