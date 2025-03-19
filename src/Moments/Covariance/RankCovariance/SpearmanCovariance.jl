struct SpearmanCovariance{T1 <: PortfolioOptimisersVarianceEstimator} <:
       RankCovarianceEstimator
    ve::T1
end
function SpearmanCovariance(; ve::PortfolioOptimisersVarianceEstimator = SimpleVariance())
    return SpearmanCovariance{typeof(ve)}(ve)
end
function StatsBase.cor(::SpearmanCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return corspearman(X)
end
function StatsBase.cov(ce::SpearmanCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1)
    return corspearman(X) .* (std_vec ⊗ std_vec)
end
function moment_factory_w(ce::SpearmanCovariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return SpearmanCovariance(; ve = moment_factory_w(ce.ve, w))
end

export SpearmanCovariance
