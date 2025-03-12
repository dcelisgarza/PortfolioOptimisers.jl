struct CorrelationDistanceDistance{T1 <: Distances.Metric, T2 <: Tuple, T3 <: NamedTuple} <:
       PortfolioOptimisersDistanceDistanceMetric
    dist::T1
    args::T2
    kwargs::T3
end
function CorrelationDistanceDistance(; dist::Distances.Metric = Distances.Euclidean(),
                                     args::Tuple = (), kwargs::NamedTuple = (;))
    return CorrelationDistanceDistance{typeof(dist), typeof(args), typeof(kwargs)}(dist,
                                                                                   args,
                                                                                   kwargs)
end
function distance(de::CorrelationDistanceDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    rho = robust_cor(ce, X; dims = dims)
    dist = sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
function distance(de::CorrelationDistanceDistance, rho::AbstractMatrix, args...; kwargs...)
    issquare(rho)
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    dist = sqrt.(clamp!(one(eltype(rho)) .- rho, zero(eltype(rho)), one(eltype(rho))))
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end

export CorrelationDistanceDistance
