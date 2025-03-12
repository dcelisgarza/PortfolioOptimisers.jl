struct GeneralDistanceDistance{T1 <: Integer, T2 <: Distances.Metric, T3 <: Tuple,
                               T4 <: NamedTuple} <:
       PortfolioOptimisersDistanceDistanceMetric
    power::T1
    dist::T2
    args::T3
    kwargs::T4
end
function GeneralDistanceDistance(; power::Integer = 1,
                                 dist::Distances.Metric = Distances.Euclidean(),
                                 args::Tuple = (), kwargs::NamedTuple = (;))
    @smart_assert(power >= one(power))
    return GeneralDistanceDistance{typeof(power), typeof(dist), typeof(args),
                                   typeof(kwargs)}(power, dist, args, kwargs)
end
function distance(de::GeneralDistanceDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    scale = isodd(de.power) ? 0.5 : 1.0
    rho = robust_cor(ce, X; dims = dims) .^ de.power
    dist = sqrt.(clamp!((one(eltype(X)) .- rho) * scale, zero(eltype(X)), one(eltype(X))))
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
function distance(de::GeneralDistanceDistance, rho::AbstractMatrix, args...; kwargs...)
    issquare(rho)
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    scale = isodd(de.power) ? 0.5 : 1.0
    dist = sqrt.(clamp!((one(eltype(rho)) .- rho .^ de.power) * scale, zero(eltype(rho)),
                        one(eltype(rho))))
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end

export GeneralDistanceDistance
