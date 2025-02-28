struct GeneralAbsoluteDistance{T1 <: Integer} <: PortfolioOptimisersDistanceMetric
    power::T1
end
function GeneralAbsoluteDistance(; power::Integer = 1)
    @smart_assert(power >= one(power))
    return GeneralAbsoluteDistance{typeof(power)}(power)
end
function distance(de::GeneralAbsoluteDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    rho = abs.(robust_cor(ce, X; dims = dims)) .^ de.power
    return sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end

export GeneralAbsoluteDistance
