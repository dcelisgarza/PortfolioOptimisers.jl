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

$(DocStringExtensions.FIELDS)

# Constructors

    NearestQuantilePrediction(;
        r::BaseRM_VecBaseRM = ConditionalValueatRisk(),
        q::Real = 0.5,
        r_kwargs::NamedTuple = (;),
        q_kwargs::NamedTuple = (;)
    ) -> NearestQuantilePrediction

## Multiplicity

`r` takes one risk measure or a vector of them, and the type gains **no** scalariser field. `r_kwargs` is already a general keyword channel forwarded straight into [`expected_risk`](@ref), so a caller writes `r_kwargs = (sca = MaxScalariser(),)`.

A mixed-polarity vector is admitted here, because [`quantile_by_measure`](@ref) takes an explicit `sign` rather than consulting [`bigger_is_better`](@ref).

## Validation

  - $(val_dict[:q_scorer])

# Functor

    (s::NearestQuantilePrediction)(ppred::PopulationPredictionResult, sign::Integer = 1)

Evaluate the scorer on a population prediction result and return the selected prediction.

`sign` is the orientation of the risk scale, forwarded to [`quantile_by_measure`](@ref). Use `1`
when a larger risk is worse, `-1` when it is better. It negates every risk value before the
quantile is taken, so `sign = -1` selects the same path that `sign = 1` selects at `1 - q`.

# Related

  - [`PredictionScorer`](@ref)
  - [`PopulationPredictionResult`](@ref)
  - [`quantile_by_measure`](@ref)
  - [`ConditionalValueatRisk`](@ref)
"""
@concrete struct NearestQuantilePrediction <: PredictionScorer
    """
    $(field_dict[:r])
    """
    r
    """
    $(field_dict[:q_scorer])
    """
    q
    """
    $(field_dict[:r_kwargs])
    """
    r_kwargs
    """
    $(field_dict[:q_kwargs])
    """
    q_kwargs
    function NearestQuantilePrediction(r::BaseRM_VecBaseRM, q::Real, r_kwargs::NamedTuple,
                                       q_kwargs::NamedTuple)
        @argcheck(zero(q) <= q <= one(q), DomainError(q, "`q` must be in [0, 1]"))
        return new{typeof(r), typeof(q), typeof(r_kwargs), typeof(q_kwargs)}(r, q, r_kwargs,
                                                                             q_kwargs)
    end
end
function NearestQuantilePrediction(; r::BaseRM_VecBaseRM = ConditionalValueatRisk(),
                                   q::Real = 0.5, r_kwargs::NamedTuple = (;),
                                   q_kwargs::NamedTuple = (;))::NearestQuantilePrediction
    return NearestQuantilePrediction(r, q, r_kwargs, q_kwargs)
end
function (s::NearestQuantilePrediction)(ppred::PopulationPredictionResult,
                                        sign::Integer = 1)
    return quantile_by_measure(ppred, s.r, s.q; r_kwargs = s.r_kwargs,
                               q_kwargs = s.q_kwargs, sign = sign)
end

export NearestQuantilePrediction
