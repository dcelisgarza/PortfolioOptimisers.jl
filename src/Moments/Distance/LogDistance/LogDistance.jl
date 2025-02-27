struct LogDistance <: PortfolioOptimisersDistanceMetric end
function distance(::LogDistance, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix;
                  dims::Int = 1)
    rho = abs.(robust_cor(ce, X; dims = dims))
    return -log.(rho)
end
function distance(::LogDistance, ce::LTDCovariance, X::AbstractMatrix; dims::Int = 1)
    rho = robust_cor(ce, X; dims = dims)
    return -log.(rho)
end

export LogDistance
