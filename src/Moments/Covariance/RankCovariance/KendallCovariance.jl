struct KendallCovariance{T1 <: PortfolioOptimisersVarianceEstimator} <:
       RankCovarianceEstimator
    ve::T1
end
function KendallCovariance(; ve::PortfolioOptimisersVarianceEstimator = SimpleVariance())
    return KendallCovariance{typeof(ve)}(ve)
end
function StatsBase.cor(::KendallCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return corkendall(X)
end
function StatsBase.cov(ce::KendallCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1)
    return corkendall(X) .* (std_vec ⊗ std_vec)
end

export KendallCovariance
