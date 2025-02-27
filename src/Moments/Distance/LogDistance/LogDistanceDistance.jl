struct LogDistanceDistance{T1 <: Distances.Metric, T2 <: Tuple, T3 <: NamedTuple} <:
       PortfolioOptimisersDistanceDistanceMetric
    dist::T1
    args::T2
    kwargs::T3
end
function LogDistanceDistance(; dist::Distances.Metric = Distances.Euclidean(),
                             args::Tuple = (), kwargs::NamedTuple = (;))
    return LogDistanceDistance{typeof(dist), typeof(args), typeof(kwargs)}(dist, args,
                                                                           kwargs)
end
function distance(de::LogDistanceDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    rho = abs.(robust_cor(ce, X; dims = dims))
    dist = -log.(rho)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
function distance(de::LogDistanceDistance, ce::LTDCovariance, X::AbstractMatrix;
                  dims::Int = 1)
    rho = robust_cor(ce, X; dims = dims)
    dist = -log.(rho)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end

export LogDistanceDistance
