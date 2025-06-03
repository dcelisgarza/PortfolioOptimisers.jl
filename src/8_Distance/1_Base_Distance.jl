abstract type AbstractDistanceEstimator <: AbstractEstimator end
abstract type AbstractDistanceAlgorithm <: AbstractAlgorithm end
struct SimpleDistance <: AbstractDistanceAlgorithm end
struct SimpleAbsoluteDistance <: AbstractDistanceAlgorithm end
struct LogDistance <: AbstractDistanceAlgorithm end
struct CorrelationDistance <: AbstractDistanceAlgorithm end
struct CanonicalDistance <: AbstractDistanceAlgorithm end
struct VariationInfoDistance{T1 <: Union{<:AbstractBins, <:Integer}, T2 <: Bool,
                             T3 <: FLoops.Transducers.Executor} <: AbstractDistanceAlgorithm
    bins::T1
    normalise::T2
    threads::T3
end
function VariationInfoDistance(;
                               bins::Union{<:AbstractBins, <:Integer} = HacineGharbiRavier(),
                               normalise::Bool = true,
                               threads::FLoops.Transducers.Executor = ThreadedEx())
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return VariationInfoDistance{typeof(bins), typeof(normalise), typeof(threads)}(bins,
                                                                                   normalise,
                                                                                   threads)
end
function distance end

export SimpleDistance, SimpleAbsoluteDistance, LogDistance, CorrelationDistance,
       CanonicalDistance, VariationInfoDistance, distance
