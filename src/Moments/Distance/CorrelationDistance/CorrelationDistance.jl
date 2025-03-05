struct CorrelationDistance <: PortfolioOptimisersDistanceMetric end
function distance(::CorrelationDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = robust_cor(ce, X; dims = dims, kwargs...)
    return sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
function distance(::CorrelationDistance, rho::AbstractMatrix, args...; kwargs...)
    @smart_assert(size(rho, 1) == size(rho, 2))
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- rho, zero(eltype(rho)), one(eltype(rho))))
end

export CorrelationDistance
