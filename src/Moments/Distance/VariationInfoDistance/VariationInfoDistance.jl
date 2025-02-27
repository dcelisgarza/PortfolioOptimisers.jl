struct VariationInfoDistance{T1 <: PortfolioOptimisersVarianceEstimator,
                             T2 <: Union{<:Integer, <:AbstractBins}, T3 <: Bool} <:
       PortfolioOptimisersDistanceMetric
    ve::T1
    bins::T2
    normalise::T3
end
function VariationInfoDistance(;
                               ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                               bins::Union{<:Integer, <:AbstractBins} = B_HacineGharbiRavier(),
                               normalise::Bool = true)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return VariationInfoDistance{typeof(ve), typeof(bins), typeof(normalise)}(ve, bins,
                                                                              normalise)
end
function distance(de::VariationInfoDistance, ::Any, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return variation_info(X, de.bins, de.normalise)
end

export VariationInfoDistance
