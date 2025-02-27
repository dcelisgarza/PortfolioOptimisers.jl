struct GeneralSimpleDistance{T1 <: Integer} <: PortfolioOptimisersDistanceMetric
    power::T1
end
function GeneralSimpleDistance(; power::Integer = 1)
    @smart_assert(power > one(power))
    return GeneralSimpleDistance{typeof(power)}(power)
end
function distance(de::GeneralSimpleDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    scale = isodd(de.power) ? 0.5 : 1.0
    rho = robust_cor(ce, X; dims = dims) .^ de.power
    return clamp!((one(eltype(X)) .- rho) * scale, zero(eltype(X)), one(eltype(X)))
end

export GeneralSimpleDistance
