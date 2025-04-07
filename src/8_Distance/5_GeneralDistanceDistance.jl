struct GeneralDistanceDistance{T1 <: AbstractDistanceAlgorithm, T2 <: Integer,
                               T3 <: Distances.Metric, T4 <: Tuple, T5 <: NamedTuple} <:
       AbstractDistanceEstimator
    alg::T1
    power::T2
    dist::T3
    args::T4
    kwargs::T5
end
function GeneralDistanceDistance(; alg::AbstractDistanceAlgorithm = SimpleDistance(),
                                 power::Integer = 1,
                                 dist::Distances.Metric = Distances.Euclidean(),
                                 args::Tuple = (), kwargs::NamedTuple = (;))
    return GeneralDistanceDistance{typeof(alg), typeof(power), typeof(dist), typeof(args),
                                   typeof(kwargs)}(alg, power, dist, args, kwargs)
end
function distance(de::GeneralDistanceDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    dist = distance(GeneralDistance(; alg = de.alg, power = de.power), ce, X; dims = dims)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
function distance(de::GeneralDistanceDistance, rho::AbstractMatrix, args...; kwargs...)
    dist = distance(GeneralDistance(; alg = de.alg, power = de.power), rho, args...;
                    kwargs...)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end

export GeneralDistanceDistance
