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
"""
    successful_members(ppred::PopulationPredictionResult)

The members of `ppred` whose every fold solved.

[`sort_by_measure`](@ref) and [`quantile_by_measure`](@ref) rank only these, so a path with a failed fold is never ranked and never returned.

# Arguments

  - `ppred`: Population prediction result.

# Returns

  - `pred::VecPredRes_MultiPredRes`: The members whose folds all carry an [`OptimisationSuccess`](@ref) return code, in population order.

# Related

  - [`sort_by_measure`](@ref)
  - [`quantile_by_measure`](@ref)
  - [`PopulationPredictionResult`](@ref)
"""
function successful_members(ppred::PopulationPredictionResult)
    return filter(x -> all(y -> isa(y.res.retcode, OptimisationSuccess), x.pred),
                  ppred.pred)
end
"""
    lacks_id(p::PredictionResult)
    lacks_id(p::MultiPeriodPredictionResult)

Say whether the population member `p` needs an `id`.

A [`PredictionResult`](@ref) is one fold and has no `id` field, so it needs none. A [`MultiPeriodPredictionResult`](@ref) needs one when its `id` is `nothing`.

# Arguments

  - `p`: A member of a [`PopulationPredictionResult`](@ref).

# Returns

  - `flag::Bool`: `true` when `p` needs an `id`.

# Related

  - [`with_position_id`](@ref)
  - [`population_ids`](@ref)
"""
function lacks_id(::PredictionResult)
    return false
end
function lacks_id(p::MultiPeriodPredictionResult)
    return isnothing(p.id)
end
"""
    with_position_id(p::PredictionResult, i::Integer)
    with_position_id(p::MultiPeriodPredictionResult, i::Integer)
    with_position_id(p::MultiPeriodPredictionResult{<:Any, <:Any, Nothing}, i::Integer)

Give the population member `p` the `id` `i` when it lacks one.

The method is chosen by the type of the member's `id`, so nothing is tested at run time. A member whose `id` is `nothing` is rebuilt with `id = i`, and its folds and estimator are kept. Every other member is returned as it is.

# Arguments

  - `p`: A member of a [`PopulationPredictionResult`](@ref).
  - `i`: The position of `p` in the population.

# Returns

  - `p::PredRes_MultiPredRes`: The member, with an `id` where it needs one.

# Related

  - [`lacks_id`](@ref)
  - [`population_ids`](@ref)
"""
function with_position_id(p::PredictionResult, ::Integer)
    return p
end
function with_position_id(p::MultiPeriodPredictionResult, ::Integer)
    return p
end
function with_position_id(p::MultiPeriodPredictionResult{<:Any, <:Any, Nothing}, i::Integer)
    return MultiPeriodPredictionResult(; pred = p.pred, id = i, opt = p.opt)
end
"""
    population_ids(pred::VecPredRes_MultiPredRes)

Give every member of a population an `id`, so a selected path can name its place.

[`PopulationPredictionResult`](@ref) calls it on construction. A member whose `id` is `nothing` takes its position in `pred`, which is the numbering [`CombinatorialCrossValidation`](@ref) and [`MultipleRandomised`](@ref) already give their paths. A population in which no member [`lacks_id`](@ref) is returned as the same vector, so a scheme's own population is not copied.

# Arguments

  - `pred`: The members of the population.

# Returns

  - `pred::VecPredRes_MultiPredRes`: The members, each with an `id` where it needs one.

# Related

  - [`PopulationPredictionResult`](@ref)
  - [`with_position_id`](@ref)
  - [`NearestQuantilePrediction`](@ref)
"""
function population_ids(pred::VecPredRes_MultiPredRes)
    return if any(lacks_id, pred)
        [with_position_id(p, i) for (i, p) in enumerate(pred)]
    else
        pred
    end
end
