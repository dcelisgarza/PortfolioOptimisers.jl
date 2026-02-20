abstract type AbstractCrossValidationScorer <: AbstractEstimator end
const CrossValScorer = Union{<:AbstractCrossValidationScorer, <:Function}
struct NearestQuantilePrediction{T1, T2, T3} <: AbstractCrossValidationScorer
    r::T1
    q::T2
    kwargs::T3
    function NearestQuantilePrediction(r::AbstractBaseRiskMeasure, q::Real,
                                       kwargs::NamedTuple)
        return new{typeof(r), typeof(q), typeof(kwargs)}(r, q, kwargs)
    end
end
function NearestQuantilePrediction(; r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
                                   q::Real = 0.5, kwargs::NamedTuple = (;))
    return NearestQuantilePrediction(r, q, kwargs)
end
function (s::NearestQuantilePrediction)(ppred::PopulationPredictionResult)
    return quantile_by_measure(ppred, s.r, s.q; s.kwargs...)
end

export NearestQuantilePrediction
