"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cross-validation scoring strategies.

# Related

  - [`PredictionScorer`](@ref)
  - [`PopulationScorer`](@ref)
"""
abstract type AbstractCrossValidationScorer <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for scoring strategies that operate on single-period prediction
results.

# Related

  - [`AbstractCrossValidationScorer`](@ref)
  - [`NearestQuantilePrediction`](@ref)
  - [`PredictionCrossValScorer`](@ref)
"""
abstract type PredictionScorer <: AbstractCrossValidationScorer end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for scoring strategies that operate on population (multi-path)
prediction results.

# Related

  - [`AbstractCrossValidationScorer`](@ref)
  - [`PopulationCrossValScorer`](@ref)
"""
abstract type PopulationScorer <: AbstractCrossValidationScorer end
"""
    const PredictionCrossValScorer

Union of concrete [`PredictionScorer`](@ref) subtypes and plain functions that score a
[`PopulationPredictionResult`](@ref).
"""
const PredictionCrossValScorer = Union{<:PredictionScorer, <:Function}
"""
    const PopulationCrossValScorer

Union of concrete [`PopulationScorer`](@ref) subtypes and plain functions that score a
population prediction.
"""
const PopulationCrossValScorer = Union{<:PopulationScorer, <:Function}
"""
$(DocStringExtensions.TYPEDEF)

Scoring strategy that selects a prediction by finding the element of a
[`PopulationPredictionResult`](@ref) whose risk measure value is nearest to a target
quantile across the population.

# Fields

  - `r::AbstractBaseRiskMeasure`: Risk measure used to evaluate each prediction.
  - `q::Real`: Target quantile in `[0, 1]`.
  - `r_kwargs::NamedTuple`: Keyword arguments passed to the risk measure.
  - `q_kwargs::NamedTuple`: Keyword arguments passed to `quantile`.

# Constructors

    NearestQuantilePrediction(;
        r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
        q::Real = 0.5,
        r_kwargs::NamedTuple = (;),
        q_kwargs::NamedTuple = (;)
    ) -> NearestQuantilePrediction

# Functor

    (s::NearestQuantilePrediction)(ppred::PopulationPredictionResult)

Evaluates the scorer on a population prediction result and returns the selected
prediction.

# Related

  - [`PredictionScorer`](@ref)
  - [`PopulationPredictionResult`](@ref)
  - [`ConditionalValueatRisk`](@ref)
"""
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
                                   q_kwargs::NamedTuple = (;))::NearestQuantilePrediction
    return NearestQuantilePrediction(r, q, r_kwargs, q_kwargs)
end
function (s::NearestQuantilePrediction)(ppred::PopulationPredictionResult)
    return quantile_by_measure(ppred, s.r, s.q; r_kwargs = s.r_kwargs,
                               q_kwargs = s.q_kwargs)
end

export NearestQuantilePrediction
