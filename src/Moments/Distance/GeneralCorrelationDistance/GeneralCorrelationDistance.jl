struct GeneralCorrelationDistance{T1 <: Integer} <: PortfolioOptimisersDistanceMetric
    power::T1
end
function GeneralCorrelationDistance(; power::Integer = 1)
    @smart_assert(power >= one(power))
    return GeneralCorrelationDistance{typeof(power)}(power)
end
function distance(de::GeneralCorrelationDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    rho = robust_cor(ce, X; dims = dims) .^ de.power
    return sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
function distance(de::GeneralCorrelationDistance, rho::AbstractMatrix, args...; kwargs...)
    issquare(rho)
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- rho .^ de.power, zero(eltype(rho)),
                        one(eltype(rho))))
end

export GeneralCorrelationDistance
