struct GeneralAbsoluteDistance{T1 <: Integer} <: PortfolioOptimisersDistanceMetric
    power::T1
end
function GeneralAbsoluteDistance(; power::Integer = 1)
    @smart_assert(power >= one(power))
    return GeneralAbsoluteDistance{typeof(power)}(power)
end
function distance(de::GeneralAbsoluteDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = abs.(robust_cor(ce, X; dims = dims, kwargs...)) .^ de.power
    return sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end
function distance(de::GeneralAbsoluteDistance, rho::AbstractMatrix, args...; kwargs...)
    @smart_assert(size(rho, 1) == size(rho, 2))
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- abs.(rho) .^ de.power, zero(eltype(rho)),
                        one(eltype(rho))))
end

export GeneralAbsoluteDistance
