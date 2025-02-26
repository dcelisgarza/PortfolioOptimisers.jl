struct MutualInfoCovariance{T1 <: PortfolioOptimisersVarianceEstimator,
                            T2 <: Union{<:Integer, <:AbstractBins}, T3 <: Bool,
                            T4 <: Union{Nothing, <:AbstractWeights}} <:
       PortfolioOptimisersCovarianceEstimator
    ve::T1
    bins::T2
    normalise::T3
    w::T4
end
function MutualInfoCovariance(; ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                              bins::Union{<:Integer, <:AbstractBins} = B_HacineGharbiRavier(),
                              normalise::Bool = true,
                              w::Union{Nothing, <:AbstractWeights} = nothing)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return MutualInfoCovariance{typeof(ve), typeof(bins), typeof(normalise), typeof(w)}(ve,
                                                                                        bins,
                                                                                        normalise,
                                                                                        w)
end
function StatsBase.cor(ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return mutual_info(X, ce.bins, ce.normalise)
end
function StatsBase.cov(ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = vec(if isnothing(ce.w)
                      std(ce.ve, X; dims = 1)
                  else
                      std(ce.ve, X, ce.w; dims = 1)
                  end)
    return mutual_info(X, ce.bins, ce.normalise) .* (std_vec ⊗ std_vec)
end

export MutualInfoCovariance
