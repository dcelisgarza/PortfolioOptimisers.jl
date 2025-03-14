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
function distance(de::LogDistance, rho::AbstractMatrix, args...; kwargs...)
    issquare(rho)
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return -log.(abs.(rho))
end

export LogDistance
