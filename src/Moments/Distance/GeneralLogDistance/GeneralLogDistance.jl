struct GeneralLogDistance{T1 <: Integer} <: PortfolioOptimisersDistanceMetric
    power::T1
end
function GeneralLogDistance(; power::Integer = 1)
    @smart_assert(power > one(power))
    return GeneralLogDistance{typeof(power)}(power)
end
function distance(de::GeneralLogDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    rho = abs.(robust_cor(ce, X; dims = dims)) .^ de.power
    return -log.(rho)
end
function distance(de::GeneralLogDistance, ce::LTDCovariance, X::AbstractMatrix;
                  dims::Int = 1)
    rho = robust_cor(ce, X; dims = dims) .^ de.power
    return -log.(rho)
end

export GeneralLogDistance
