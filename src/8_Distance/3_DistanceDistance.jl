struct DistanceDistance{T1 <: Distances.Metric, T2 <: Tuple, T3 <: NamedTuple,
                        T4 <: AbstractDistanceAlgorithm} <: AbstractDistanceEstimator
    dist::T1
    args::T2
    kwargs::T3
    alg::T4
end
function DistanceDistance(; dist::Distances.Metric = Distances.Euclidean(),
                          args::Tuple = (), kwargs::NamedTuple = (;),
                          alg::AbstractDistanceAlgorithm = SimpleDistance())
    return DistanceDistance{typeof(dist), typeof(args), typeof(kwargs), typeof(alg)}(dist,
                                                                                     args,
                                                                                     kwargs,
                                                                                     alg)
end
function distance(de::DistanceDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    dist = distance(Distance(; alg = de.alg), ce, X; dims = dims)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
function distance(de::DistanceDistance, rho::AbstractMatrix, args...; kwargs...)
    dist = distance(Distance(; alg = de.alg), rho, args...; kwargs...)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end

export DistanceDistance
