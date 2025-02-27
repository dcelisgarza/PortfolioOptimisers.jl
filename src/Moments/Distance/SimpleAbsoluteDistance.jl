struct AbsoluteSimpleDistance <: PortfolioOptimisersAbsoluteDistanceMetric end
function distance(::AbsoluteSimpleDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    rho = abs.(robust_cor(ce, X; dims = dims))
    return clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X)))
end

export AbsoluteSimpleDistance
