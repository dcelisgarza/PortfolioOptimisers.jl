struct VariationInfoDistance{T1 <: Union{<:Integer, <:AbstractBins}, T2 <: Bool} <:
       PortfolioOptimisersDistanceMetric
    bins::T1
    normalise::T2
end
function VariationInfoDistance(;
                               bins::Union{<:Integer, <:AbstractBins} = B_HacineGharbiRavier(),
                               normalise::Bool = true)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return VariationInfoDistance{typeof(bins), typeof(normalise)}(bins, normalise)
end
function distance(de::VariationInfoDistance, ::Any, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return variation_info(X, de.bins, de.normalise)
end

export VariationInfoDistance
