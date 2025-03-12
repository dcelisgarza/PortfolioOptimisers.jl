struct GeneralLogDistanceDistance{T1 <: Integer, T2 <: Distances.Metric, T3 <: Tuple,
                                  T4 <: NamedTuple} <:
       PortfolioOptimisersDistanceDistanceMetric
    power::T1
    dist::T2
    args::T3
    kwargs::T4
end
function GeneralLogDistanceDistance(; power::Integer = 1,
                                    dist::Distances.Metric = Distances.Euclidean(),
                                    args::Tuple = (), kwargs::NamedTuple = (;))
    @smart_assert(power >= one(power))
    return GeneralLogDistanceDistance{typeof(power), typeof(dist), typeof(args),
                                      typeof(kwargs)}(power, dist, args, kwargs)
end
function distance(de::GeneralLogDistanceDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    rho = abs.(robust_cor(ce, X; dims = dims)) .^ de.power
    dist = -log.(rho)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
function distance(de::GeneralLogDistanceDistance, ce::LTDCovariance, X::AbstractMatrix;
                  dims::Int = 1)
    rho = robust_cor(ce, X; dims = dims) .^ de.power
    dist = -log.(rho)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
function distance(de::GeneralLogDistanceDistance, rho::AbstractMatrix, args...; kwargs...)
    issquare(rho)
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    dist = -log.(abs.(rho) .^ de.power)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end

export GeneralLogDistanceDistance
