struct GeneralDistance{T1 <: Integer} <: PortfolioOptimisersDistanceMetric
    power::T1
end
function GeneralDistance(; power::Integer = 1)
    @smart_assert(power > one(power))
    return GeneralDistance{typeof(power)}(power)
end
function distance(de::GeneralDistance, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix;
                  dims::Int = 1)
    scale = isodd(de.power) ? 0.5 : 1.0
    rho = robust_cor(ce, X; dims = dims) .^ de.power
    return sqrt.(clamp!((one(eltype(X)) .- rho) * scale, zero(eltype(X)), one(eltype(X))))
end

export GeneralDistance
