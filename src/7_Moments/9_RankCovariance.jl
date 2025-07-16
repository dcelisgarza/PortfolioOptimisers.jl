abstract type RankCovarianceEstimator <: AbstractCovarianceEstimator end
struct KendallCovariance{T1 <: AbstractVarianceEstimator} <: RankCovarianceEstimator
    ve::T1
end
function KendallCovariance(; ve::AbstractVarianceEstimator = SimpleVariance())
    return KendallCovariance{typeof(ve)}(ve)
end
function Statistics.cor(::KendallCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return corkendall(X)
end
function Statistics.cov(ce::KendallCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return corkendall(X) ⊙ (std_vec ⊗ std_vec)
end
function factory(ce::KendallCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return KendallCovariance(; ve = factory(ce.ve, w))
end
struct SpearmanCovariance{T1 <: AbstractVarianceEstimator} <: RankCovarianceEstimator
    ve::T1
end
function SpearmanCovariance(; ve::AbstractVarianceEstimator = SimpleVariance())
    return SpearmanCovariance{typeof(ve)}(ve)
end
function Statistics.cor(::SpearmanCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return corspearman(X)
end
function Statistics.cov(ce::SpearmanCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return corspearman(X) ⊙ (std_vec ⊗ std_vec)
end
function factory(ce::SpearmanCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return SpearmanCovariance(; ve = factory(ce.ve, w))
end

export KendallCovariance, SpearmanCovariance
