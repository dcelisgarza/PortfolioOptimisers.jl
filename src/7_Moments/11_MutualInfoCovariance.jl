struct MutualInfoCovariance{T1 <: AbstractVarianceEstimator,
                            T2 <: Union{<:AbstractBins, <:Integer}, T3 <: Bool} <:
       AbstractCovarianceEstimator
    ve::T1
    bins::T2
    normalise::T3
end
function MutualInfoCovariance(; ve::AbstractVarianceEstimator = SimpleVariance(),
                              bins::Union{<:AbstractBins, <:Integer} = HacineGharbiRavier(),
                              normalise::Bool = true)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return MutualInfoCovariance{typeof(ve), typeof(bins), typeof(normalise)}(ve, bins,
                                                                             normalise)
end
function factory(ce::MutualInfoCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return MutualInfoCovariance(; ve = factory(ce.ve, w), bins = ce.bins,
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
