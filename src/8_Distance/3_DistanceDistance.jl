struct DistanceDistance{T1 <: AbstractDistanceAlgorithm, T2 <: Distances.Metric,
                        T3 <: Tuple, T4 <: NamedTuple} <: AbstractDistanceEstimator
    alg::T1
    dist::T2
    args::T3
    kwargs::T4
end
function DistanceDistance(; alg::AbstractDistanceAlgorithm = SimpleDistance(),
                          dist::Distances.Metric = Distances.Euclidean(), args::Tuple = (),
                          kwargs::NamedTuple = (;))
    return DistanceDistance{typeof(alg), typeof(dist), typeof(args), typeof(kwargs)}(alg,
                                                                                     dist,
                                                                                     args,
                                                                                     kwargs)
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
