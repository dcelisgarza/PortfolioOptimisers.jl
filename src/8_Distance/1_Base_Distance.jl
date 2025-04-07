abstract type AbstractDistanceEstimator <: AbstractEstimator end
abstract type AbstractDistanceAlgorithm <: AbstractAlgorithm end
struct SimpleDistance <: AbstractDistanceAlgorithm end
struct SimpleAbsoluteDistance <: AbstractDistanceAlgorithm end
struct LogDistance <: AbstractDistanceAlgorithm end
struct CorrelationDistance <: AbstractDistanceAlgorithm end
struct CanonicalDistance <: AbstractDistanceAlgorithm end
struct VariationInfoDistance{T1 <: Union{<:Integer, <:AbstractBins}, T2 <: Bool} <:
       AbstractDistanceAlgorithm
    bins::T1
    normalise::T2
end
function VariationInfoDistance(;
                               bins::Union{<:Integer, <:AbstractBins} = HacineGharbiRavier(),
                               normalise::Bool = true)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return VariationInfoDistance{typeof(bins), typeof(normalise)}(bins, normalise)
end

export SimpleDistance, SimpleAbsoluteDistance, LogDistance, CorrelationDistance,
       CanonicalDistance, VariationInfoDistance
