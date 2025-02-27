struct AbsoluteSimpleDistanceDistance{T1 <: Distances.Metric, T2 <: Tuple,
                                      T3 <: NamedTuple} <:
       PortfolioOptimisersDistanceDistanceMetric
    dist::T1
    args::T2
    kwargs::T3
end
function AbsoluteSimpleDistanceDistance(; dist::Distances.Metric = Distances.Euclidean(),
                                        args::Tuple = (), kwargs::NamedTuple = (;))
    return AbsoluteSimpleDistanceDistance{typeof(dist), typeof(args), typeof(kwargs)}(dist,
                                                                                      args,
                                                                                      kwargs)
end
function distance(de::AbsoluteSimpleDistanceDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    rho = abs.(robust_cor(ce, X; dims = dims))
    dist = sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end

export AbsoluteSimpleDistanceDistance
