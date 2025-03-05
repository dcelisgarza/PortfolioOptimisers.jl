struct GeneralCanonicalDistance{T1 <: Integer} <: PortfolioOptimisersDistanceMetric
    power::T1
end
function GeneralCanonicalDistance(; power::Integer = 1)
    @smart_assert(power >= one(power))
    return GeneralCanonicalDistance{typeof(power)}(power)
end
function distance(de::GeneralCanonicalDistance, ce::MutualInfoCovariance, X::AbstractMatrix;
                  dims::Int = 1, kwargs...)
    return distance(GeneralVariationInfoDistance(; power = de.power, bins = ce.bins,
                                                 normalise = ce.normalise), ce, X;
                    dims = dims, kwargs...)
end
function distance(de::GeneralCanonicalDistance, ce::LTDCovariance, X::AbstractMatrix;
                  dims::Int = 1, kwargs...)
    return distance(GeneralLogDistance(; power = de.power), ce, X; dims = dims, kwargs...)
end
function distance(de::GeneralCanonicalDistance, ce::DistanceCovariance, X::AbstractMatrix;
                  dims::Int = 1, kwargs...)
    return distance(GeneralCorrelationDistance(; power = de.power), ce, X; dims = dims,
                    kwargs...)
end
function distance(de::GeneralCanonicalDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(GeneralDistance(; power = de.power), ce, X; dims = dims, kwargs...)
end
function distance(de::GeneralCanonicalDistance, rho::AbstractMatrix, args...; kwargs...)
    return distance(GeneralDistance(; power = de.power), rho; kwargs...)
end

export GeneralCanonicalDistance
