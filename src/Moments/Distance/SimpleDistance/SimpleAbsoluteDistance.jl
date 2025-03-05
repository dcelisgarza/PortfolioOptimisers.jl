struct SimpleAbsoluteDistance <: PortfolioOptimisersAbsoluteDistanceMetric end
function distance(::SimpleAbsoluteDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    rho = abs.(robust_cor(ce, X; dims = dims))
    return sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end
function distance(de::SimpleAbsoluteDistance, rho::AbstractMatrix, args...; kwargs...)
    @smart_assert(size(rho, 1) == size(rho, 2))
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- abs.(rho), zero(eltype(rho)), one(eltype(rho))))
end

export SimpleAbsoluteDistance
