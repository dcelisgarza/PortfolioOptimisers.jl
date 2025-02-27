struct CanonicalDistanceDistance{T1 <: Distances.Metric, T2 <: Tuple, T3 <: NamedTuple} <:
       PortfolioOptimisersDistanceDistanceMetric
    dist::T1
    args::T2
    kwargs::T3
end
function CanonicalDistanceDistance(; dist::Distances.Metric = Distances.Euclidean(),
                                   args::Tuple = (), kwargs::NamedTuple = (;))
    return CanonicalDistanceDistance{typeof(dist), typeof(args), typeof(kwargs)}(dist, args,
                                                                                 kwargs)
end
function distance(de::CanonicalDistanceDistance, ce::MutualInfoCovariance,
                  X::AbstractMatrix; dims::Int = 1)
    return distance(VariationInfoDistanceDistance(; bins = ce.bins,
                                                  normalise = ce.normalise, dist = de.dist,
                                                  args = de.args, kwargs = de.kwargs), ce,
                    X; dims = dims)
end
function distance(de::CanonicalDistanceDistance, ce::LTDCovariance, X::AbstractMatrix;
                  dims::Int = 1)
    return distance(LogDistanceDistance(; dist = de.dist, args = de.args,
                                        kwargs = de.kwargs), ce, X; dims = dims)
end
function distance(de::CanonicalDistanceDistance, ce::DistanceCovariance, X::AbstractMatrix;
                  dims::Int = 1)
    return distance(CorrelationDistanceDistance(; dist = de.dist, args = de.args,
                                                kwargs = de.kwargs), ce, X; dims = dims)
end
function distance(de::CanonicalDistanceDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    return distance(SimpleDistanceDistance(; dist = de.dist, args = de.args,
                                           kwargs = de.kwargs), ce, X; dims = dims)
end

export CanonicalDistanceDistance
