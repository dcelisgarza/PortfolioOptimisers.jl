struct GeneralSimpleAbsoluteDistance{T1 <: Integer} <: PortfolioOptimisersDistanceMetric
    power::T1
end
function GeneralSimpleAbsoluteDistance(; power::Integer = 1)
    @smart_assert(power > one(power))
    return GeneralSimpleAbsoluteDistance{typeof(power)}(power)
end
function distance(de::GeneralSimpleAbsoluteDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    rho = abs.(robust_cor(ce, X; dims = dims)) .^ de.power
    return clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X)))
end

export GeneralSimpleAbsoluteDistance
