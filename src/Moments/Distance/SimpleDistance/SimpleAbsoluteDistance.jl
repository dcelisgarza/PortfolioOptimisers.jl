struct SimpleAbsoluteDistance <: PortfolioOptimisersAbsoluteDistanceMetric end
function distance(::SimpleAbsoluteDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1)
    rho = abs.(robust_cor(ce, X; dims = dims))
    return sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end

export SimpleAbsoluteDistance
