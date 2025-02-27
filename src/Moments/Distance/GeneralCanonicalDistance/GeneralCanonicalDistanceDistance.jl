struct GeneralCanonicalDistanceDistance{T1 <: Integer, T2 <: Distances.Metric, T3 <: Tuple,
                                        T4 <: NamedTuple} <:
       PortfolioOptimisersDistanceDistanceMetric
    power::T1
    dist::T2
    args::T3
    kwargs::T4
end
function GeneralCanonicalDistanceDistance(; power::Integer = 1,
                                          dist::Distances.Metric = Distances.Euclidean(),
                                          args::Tuple = (), kwargs::NamedTuple = (;))
    @smart_assert(power >= one(power))
    return GeneralCanonicalDistanceDistance{typeof(power), typeof(dist), typeof(args),
                                            typeof(kwargs)}(power, dist, args, kwargs)
end
function distance(de::GeneralCanonicalDistanceDistance, ce::MutualInfoCovariance,
                  X::AbstractMatrix; dims::Int = 1)
    return distance(GeneralVariationInfoDistanceDistance(; power = de.power, bins = ce.bins,
                                                         normalise = ce.normalise,
                                                         dist = de.dist, args = de.args,
                                                         kwargs = de.kwargs), ce, X;
                    dims = dims)
end
function distance(de::GeneralCanonicalDistanceDistance, ce::LTDCovariance,
                  X::AbstractMatrix; dims::Int = 1)
    return distance(GeneralLogDistanceDistance(; power = de.power, dist = de.dist,
                                               args = de.args, kwargs = de.kwargs), ce, X;
                    dims = dims)
end
function distance(de::GeneralCanonicalDistanceDistance, ce::DistanceCovariance,
                  X::AbstractMatrix; dims::Int = 1)
    return distance(GeneralCorrelationDistanceDistance(; power = de.power, dist = de.dist,
                                                       args = de.args, kwargs = de.kwargs),
                    ce, X; dims = dims)
end
function distance(de::GeneralCanonicalDistanceDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    return distance(GeneralDistanceDistance(; power = de.power, dist = de.dist,
                                            args = de.args, kwargs = de.kwargs), ce, X;
                    dims = dims)
end

export GeneralCanonicalDistanceDistance
