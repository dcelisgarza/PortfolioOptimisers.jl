struct LogDistance <: PortfolioOptimisersDistanceMetric end
function distance(::LogDistance, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix;
                  dims::Int = 1, kwargs...)
    rho = abs.(robust_cor(ce, X; dims = dims, kwargs...))
    return -log.(rho)
end
function distance(::LogDistance, ce::LTDCovariance, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    rho = robust_cor(ce, X; dims = dims, kwargs...)
    return -log.(rho)
end
function distance(de::LogDistance, rho::AbstractMatrix, args...; kwargs...)
    @smart_assert(size(rho, 1) == size(rho, 2))
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return -log.(abs.(rho))
end

export LogDistance
