struct GeneralDistanceDistance{T1, T2, T3, T4, T5} <: AbstractDistanceEstimator
    dist::T1
    args::T2
    kwargs::T3
    power::T4
    alg::T5
end
function GeneralDistanceDistance(; dist::Distances.Metric = Distances.Euclidean(),
                                 args::Tuple = (), kwargs::NamedTuple = (;),
                                 power::Integer = 1,
                                 alg::AbstractDistanceAlgorithm = SimpleDistance())
    return GeneralDistanceDistance(dist, args, kwargs, power, alg)
end
function distance(de::GeneralDistanceDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    dist = distance(GeneralDistance(; power = de.power, alg = de.alg), ce, X; dims = dims,
                    kwargs...)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
function cor_and_dist(de::GeneralDistanceDistance, ce::StatsBase.CovarianceEstimator,
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho, dist = cor_and_dist(GeneralDistance(; power = de.power, alg = de.alg), ce, X;
                             dims = dims, kwargs...)
    return rho, Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
function distance(de::GeneralDistanceDistance, rho::AbstractMatrix, args...; kwargs...)
    dist = distance(GeneralDistance(; power = de.power, alg = de.alg), rho, args...;
                    kwargs...)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end

export GeneralDistanceDistance
