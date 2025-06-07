struct GeneralDistanceDistance{T1 <: Distances.Metric, T2 <: Tuple, T3 <: NamedTuple,
                               T4 <: Integer, T5 <: AbstractDistanceAlgorithm} <:
       AbstractDistanceEstimator
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
    return GeneralDistanceDistance{typeof(dist), typeof(args), typeof(kwargs),
                                   typeof(power), typeof(alg)}(dist, args, kwargs, power,
                                                               alg)
end
function distance(de::GeneralDistanceDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    dist = distance(GeneralDistance(; power = de.power, alg = de.alg), ce, X; dims = dims)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
function distance(de::GeneralDistanceDistance, rho::AbstractMatrix, args...; kwargs...)
    dist = distance(GeneralDistance(; power = de.power, alg = de.alg), rho, args...;
                    kwargs...)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end

export GeneralDistanceDistance
