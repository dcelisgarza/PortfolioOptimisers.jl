struct GeneralAbsoluteDistanceDistance{T1 <: Integer, T2 <: Distances.Metric, T3 <: Tuple,
                                       T4 <: NamedTuple} <:
       PortfolioOptimisersDistanceDistanceMetric
    power::T1
    dist::T2
    args::T3
    kwargs::T4
end
function GeneralAbsoluteDistanceDistance(; power::Integer = 1,
                                         dist::Distances.Metric = Distances.Euclidean(),
                                         args::Tuple = (), kwargs::NamedTuple = (;))
    @smart_assert(power > one(power))
    return GeneralAbsoluteDistanceDistance{typeof(power), typeof(dist), typeof(args),
                                           typeof(kwargs)}(power, dist, args, kwargs)
end
function distance(de::GeneralAbsoluteDistanceDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    rho = abs.(robust_cor(ce, X; dims = dims)) .^ de.power
    dist = sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end

export GeneralAbsoluteDistanceDistance
