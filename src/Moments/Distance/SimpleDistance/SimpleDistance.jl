struct SimpleDistance <: PortfolioOptimisersDistanceMetric end
function distance(::SimpleDistance, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix;
                  dims::Int = 1)
    rho = robust_cor(ce, X; dims = dims)
    return sqrt.(clamp!((one(eltype(X)) .- rho) * 0.5, zero(eltype(X)), one(eltype(X))))
end

export SimpleDistance
