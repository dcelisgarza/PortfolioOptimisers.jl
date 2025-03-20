struct MutualInfoCovariance{T1 <: PortfolioOptimisersVarianceEstimator,
                            T2 <: Union{<:Integer, <:AbstractBins}, T3 <: Bool} <:
       PortfolioOptimisersCovarianceEstimator
    ve::T1
    bins::T2
    normalise::T3
end
function MutualInfoCovariance(; ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                              bins::Union{<:Integer, <:AbstractBins} = B_HacineGharbiRavier(),
                              normalise::Bool = true)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return MutualInfoCovariance{typeof(ve), typeof(bins), typeof(normalise)}(ve, bins,
                                                                             normalise)
end
function moment_factory_w(ce::MutualInfoCovariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return MutualInfoCovariance(; ve = moment_factory_w(ce.ve, w), bins = ce.bins,
                                normalise = ce.normalise)
end
function StatsBase.cor(ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1,
                       kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return mutual_info(X, ce.bins, ce.normalise)
end
function StatsBase.cov(ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1,
                       kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1)
    return mutual_info(X, ce.bins, ce.normalise) .* (std_vec ⊗ std_vec)
end

export MutualInfoCovariance
