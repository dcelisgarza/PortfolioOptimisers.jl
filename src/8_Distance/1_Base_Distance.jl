abstract type AbstractDistanceEstimator <: AbstractEstimator end
abstract type AbstractDistanceAlgorithm <: AbstractAlgorithm end
struct SimpleDistance <: AbstractDistanceAlgorithm end
struct SimpleAbsoluteDistance <: AbstractDistanceAlgorithm end
struct LogDistance <: AbstractDistanceAlgorithm end
struct CorrelationDistance <: AbstractDistanceAlgorithm end
struct CanonicalDistance <: AbstractDistanceAlgorithm end
struct VariationInfoDistance{T1, T2} <: AbstractDistanceAlgorithm
    bins::T1
    normalise::T2
end
function VariationInfoDistance(;
                               bins::Union{<:AbstractBins, <:Integer} = HacineGharbiRavier(),
                               normalise::Bool = true)
    if isa(bins, Integer)
        @assert(bins > zero(bins))
    end
    return VariationInfoDistance(bins, normalise)
end
function distance end
function cor_and_dist end

export SimpleDistance, SimpleAbsoluteDistance, LogDistance, CorrelationDistance,
       CanonicalDistance, VariationInfoDistance, distance, cor_and_dist
