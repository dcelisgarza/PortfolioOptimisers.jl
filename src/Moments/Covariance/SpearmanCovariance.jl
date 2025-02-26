struct SpearmanCovariance{T1 <: PortfolioOptimisersVarianceEstimator,
                          T2 <: Union{Nothing, <:AbstractWeights}} <:
       RankCovarianceEstimator
    ve::T1
    w::T2
end
function SpearmanCovariance(; ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                            w::Union{Nothing, <:AbstractWeights} = nothing)
    return SpearmanCovariance{typeof(ve), typeof(w)}(ve, w)
end
function StatsBase.cor(::SpearmanCovariance, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return corspearman(X)
end
function StatsBase.cov(ce::SpearmanCovariance, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end

    std_vec = vec(if isnothing(ce.w)
                      std(ce.ve, X; dims = 1)
                  else
                      std(ce.ve, X, ce.w; dims = 1)
                  end)

    return corspearman(X) .* (std_vec ⊗ std_vec)
end

export SpearmanCovariance
