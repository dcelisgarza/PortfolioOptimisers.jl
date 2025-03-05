struct SimpleDistanceDistance{T1 <: Distances.Metric, T2 <: Tuple, T3 <: NamedTuple} <:
       PortfolioOptimisersDistanceDistanceMetric
    dist::T1
    args::T2
    kwargs::T3
end
function SimpleDistanceDistance(; dist::Distances.Metric = Distances.Euclidean(),
                                args::Tuple = (), kwargs::NamedTuple = (;))
    return SimpleDistanceDistance{typeof(dist), typeof(args), typeof(kwargs)}(dist, args,
                                                                              kwargs)
end
function distance(de::SimpleDistanceDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = robust_cor(ce, X; dims = dims, kwargs...)
    dist = sqrt.(clamp!((one(eltype(X)) .- rho) * 0.5, zero(eltype(X)), one(eltype(X))))
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
function distance(de::SimpleDistanceDistance, rho::AbstractMatrix, args...; kwargs...)
    @smart_assert(size(rho, 1) == size(rho, 2))
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    dist = sqrt.(clamp!((one(eltype(rho)) .- rho) * 0.5, zero(eltype(rho)),
                        one(eltype(rho))))
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end

export SimpleDistanceDistance
