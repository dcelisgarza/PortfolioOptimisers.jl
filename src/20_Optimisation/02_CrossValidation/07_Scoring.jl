abstract type AbstractCrossValidationScorer <: AbstractEstimator end
abstract type PredictionScorer <: AbstractCrossValidationScorer end
abstract type PopulationScorer <: AbstractCrossValidationScorer end
const PredictionCrossValScorer = Union{<:PredictionScorer, <:Function}
const PopulationCrossValScorer = Union{<:PopulationScorer, <:Function}
@concrete struct NearestQuantilePrediction <: PredictionScorer
    r
    q
    r_kwargs
    q_kwargs
    function NearestQuantilePrediction(r::AbstractBaseRiskMeasure, q::Real,
                                       r_kwargs::NamedTuple, q_kwargs::NamedTuple)
        return new{typeof(r), typeof(q), typeof(r_kwargs), typeof(q_kwargs)}(r, q, r_kwargs,
                                                                             q_kwargs)
    end
end
function NearestQuantilePrediction(; r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
                                   q::Real = 0.5, r_kwargs::NamedTuple = (;),
                                   q_kwargs::NamedTuple = (;))
    return NearestQuantilePrediction(r, q, r_kwargs, q_kwargs)
end
function (s::NearestQuantilePrediction)(ppred::PopulationPredictionResult)
    return quantile_by_measure(ppred, s.r, s.q; r_kwargs = s.r_kwargs,
                               q_kwargs = s.q_kwargs)
end

export NearestQuantilePrediction
