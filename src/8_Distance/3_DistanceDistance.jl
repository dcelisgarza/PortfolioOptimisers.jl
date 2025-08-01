struct DistanceDistance{T1, T2, T3, T4} <: AbstractDistanceEstimator
    dist::T1
    args::T2
    kwargs::T3
    alg::T4
end
function DistanceDistance(; dist::Distances.Metric = Distances.Euclidean(),
                          args::Tuple = (), kwargs::NamedTuple = (;),
                          alg::AbstractDistanceAlgorithm = SimpleDistance())
    return DistanceDistance(dist, args, kwargs, alg)
end
function distance(de::DistanceDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    dist = distance(Distance(; alg = de.alg), ce, X; dims = dims, kwargs...)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
function cor_and_dist(de::DistanceDistance, ce::StatsBase.CovarianceEstimator,
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho, dist = cor_and_dist(Distance(; alg = de.alg), ce, X; dims = dims, kwargs...)
    return rho, Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
function distance(de::DistanceDistance, rho::AbstractMatrix, args...; kwargs...)
    dist = distance(Distance(; alg = de.alg), rho, args...; kwargs...)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end

export DistanceDistance
