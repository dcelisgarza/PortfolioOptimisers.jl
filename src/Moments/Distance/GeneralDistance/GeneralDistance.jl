struct GeneralDistance{T1 <: Integer} <: PortfolioOptimisersDistanceMetric
    power::T1
end
function GeneralDistance(; power::Integer = 1)
    @smart_assert(power >= one(power))
    return GeneralDistance{typeof(power)}(power)
end
function distance(de::GeneralDistance, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix;
                  dims::Int = 1, kwargs...)
    scale = isodd(de.power) ? 0.5 : 1.0
    rho = robust_cor(ce, X; dims = dims, kwargs...) .^ de.power
    return sqrt.(clamp!((one(eltype(X)) .- rho) * scale, zero(eltype(X)), one(eltype(X))))
end
function distance(de::GeneralDistance, rho::AbstractMatrix, args...; kwargs...)
    @smart_assert(size(rho, 1) == size(rho, 2))
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    scale = isodd(de.power) ? 0.5 : 1.0
    return sqrt.(clamp!((one(eltype(rho)) .- rho .^ de.power) * scale, zero(eltype(rho)),
                        one(eltype(rho))))
end

export GeneralDistance
