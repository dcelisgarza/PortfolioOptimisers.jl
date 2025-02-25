struct KendallCovariance{T1 <: POVarianceEstimator,
                         T2 <: Union{Nothing, <:AbstractWeights}} <: RankCovarianceEstimator
    ve::T1
    w::T2
end
function KendallCovariance(; ve::POVarianceEstimator = SimpleVariance(),
                           w::Union{Nothing, <:AbstractWeights} = nothing)
    return KendallCovariance{typeof(ve), typeof(w)}(ve, w)
end
function StatsBase.cor(::KendallCovariance, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return corkendall(X)
end
function StatsBase.cov(ce::KendallCovariance, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end

    std_vec = vec(if isnothing(ce.w)
                      std(ce.ve, X; dims = 1)
                  else
                      std(ce.ve, X, ce.w; dims = 1)
                  end)

    return corkendall(X) .* (std_vec * transpose(std_vec))
end

export KendallCovariance
