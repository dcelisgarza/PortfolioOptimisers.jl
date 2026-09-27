"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the cross-validation scorers.

A scorer reads the population of paths that a cross-validation scheme returns. [`PredictionScorer`](@ref) selects one path from it, and [`PopulationScorer`](@ref) scores the population as a whole.

# Related

  - [`PredictionScorer`](@ref)
  - [`PopulationScorer`](@ref)
"""
abstract type AbstractCrossValidationScorer <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the scorers that select one path from a population of predictions.

A subtype is callable. It takes a [`PopulationPredictionResult`](@ref) and returns one of its members.

# Related

  - [`AbstractCrossValidationScorer`](@ref)
  - [`NearestQuantilePrediction`](@ref)
  - [`PredictionCrossValScorer`](@ref)
"""
abstract type PredictionScorer <: AbstractCrossValidationScorer end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the scorers that score a population of predictions as a whole.

The library defines no subtype of it, and no method dispatches on it.

# Related

  - [`AbstractCrossValidationScorer`](@ref)
  - [`PopulationCrossValScorer`](@ref)
"""
abstract type PopulationScorer <: AbstractCrossValidationScorer end
"""
    const PredictionCrossValScorer = Union{<:PredictionScorer, <:Function}

Union of the values that the `scorer` field of [`OptimisationCrossValidation`](@ref) accepts: a [`PredictionScorer`](@ref) or a plain function.

The union exists so that a caller can select a path with a function and write no type. The function takes a [`PopulationPredictionResult`](@ref) and returns one of its members, as a [`PredictionScorer`](@ref) does.

# Related

  - [`PredictionScorer`](@ref)
  - [`NearestQuantilePrediction`](@ref)
  - [`OptimisationCrossValidation`](@ref)
"""
const PredictionCrossValScorer = Union{<:PredictionScorer, <:Function}
"""
    const PopulationCrossValScorer = Union{<:PopulationScorer, <:Function}

Union of a [`PopulationScorer`](@ref) and a plain function that scores a population of predictions as a whole.

It is the population counterpart of [`PredictionCrossValScorer`](@ref). No field or method of the library takes it.

# Related

  - [`PopulationScorer`](@ref)
  - [`PredictionCrossValScorer`](@ref)
"""
const PopulationCrossValScorer = Union{<:PopulationScorer, <:Function}
"""
$(DocStringExtensions.TYPEDEF)

Selects the path of a population whose risk is nearest to a target quantile of the population's risks.

It is the default scorer of [`OptimisationCrossValidation`](@ref) under [`CombinatorialCrossValidation`](@ref). A path with a failed fold, or with a risk that is not finite, takes no part in the quantile, and the scorer never selects it.

# Mathematical definition

```math
\\begin{align}
\\rho_i &= s \\, \\mathcal{R}(P_i)\\,, \\quad i \\in \\mathcal{S}\\,, \\\\
i^{\\star} &= \\underset{i \\in \\mathcal{S}}{\\arg\\min} \\left| \\rho_i - Q_q\\left(\\{\\rho_j\\}_{j \\in \\mathcal{S}}\\right) \\right|\\,.
\\end{align}
```

Where:

  - ``P_i``: Path ``i`` of the population.
  - ``\\mathcal{S}``: Set of the paths whose every fold solved and whose risk ``\\mathcal{R}(P_i)`` is finite.
  - ``\\mathcal{R}(P_i)``: Expected risk of path ``i`` under `r`. The scalariser `r_kwargs.sca` reduces a vector of risk measures to one value, and the default is [`SumScalariser`](@ref).
  - ``s \\in \\{1, -1\\}``: Orientation of the risk scale, the functor's `sign`.
  - ``Q_q``: ``q``-th quantile of a set of values, as `Statistics.quantile` computes it with `q_kwargs`.
  - ``i^{\\star}``: Selected path.

``q = 0`` selects the path of least ``\\rho_i`` and ``q = 1`` the path of greatest ``\\rho_i``, whatever the quantile definition. When `q_kwargs` keeps `alpha` equal to `beta`, as the default does, ``Q_q(\\{-\\rho_j\\}) = -Q_{1-q}(\\{\\rho_j\\})``, so ``s = -1`` selects the path that ``s = 1`` selects at ``1 - q``. A definition with `alpha` different from `beta` breaks this symmetry.

# Algorithm

 1. Keep the members of `ppred` whose every fold solved, through [`successful_members`](@ref), giving `pred`.
 2. Compute `rks`, the vector of ``s \\, \\mathcal{R}(P_i)`` over `pred`, through [`expected_risk`](@ref) with `r_kwargs`.
 3. Drop every entry of `rks` that is not finite, and the member of `pred` at its position.
 4. Compute `rkq`, the `q`-th quantile of `rks`, through `Statistics.quantile` with `q_kwargs`.
 5. Scan `rks` in population order for the entry nearest to `rkq`, giving `idx`. A tie goes to the first entry.
 6. Return `pred[idx]`.

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

`r` takes one risk measure or a vector of them, and the type has no scalariser field. [`expected_risk`](@ref) reads the scalariser from `r_kwargs`, so a caller writes `r_kwargs = (sca = MaxScalariser(),)`.

The scorer admits a vector whose members disagree on polarity, because [`quantile_by_measure`](@ref) takes an explicit `sign` and does not call [`bigger_is_better`](@ref).

## Validation

  - $(val_dict[:q_scorer])

# Functor

    (s::NearestQuantilePrediction)(ppred::PopulationPredictionResult, sign::Integer = 1)

Selects a path of `ppred` through [`quantile_by_measure`](@ref) and returns it.

`sign` is the orientation ``s`` of the risk scale. Use `1` when a larger risk is worse and `-1` when a larger value is better.

When the population is empty, or every member has a failed fold or a risk that is not finite, `Statistics.quantile` receives no value and throws an `ArgumentError`.

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
    member_solved(p::PredictionResult)
    member_solved(p::MultiPeriodPredictionResult)

Says whether every fold of the population member `p` solved.

A [`PredictionResult`](@ref) is one fold, and it solved when its return code is an [`OptimisationSuccess`](@ref). A [`MultiPeriodPredictionResult`](@ref) solved when each of its folds solved.

# Arguments

  - `p`: A member of a [`PopulationPredictionResult`](@ref).

# Returns

  - `flag::Bool`: `true` when every fold of `p` solved.

# Related

  - [`successful_members`](@ref)
"""
function member_solved(p::PredictionResult)
    return isa(p.res.retcode, OptimisationSuccess)
end
function member_solved(p::MultiPeriodPredictionResult)
    return all(member_solved, p.pred)
end
"""
    successful_members(ppred::PopulationPredictionResult)

Gives the members of `ppred` whose every fold solved.

[`sort_by_measure`](@ref) and [`quantile_by_measure`](@ref) rank only these members, so they never rank or return a path with a failed fold.

# Arguments

  - `ppred`: Population prediction result.

# Returns

  - `pred::VecPredRes_MultiPredRes`: The members for which [`member_solved`](@ref) holds, in population order.

# Related

  - [`member_solved`](@ref)
  - [`sort_by_measure`](@ref)
  - [`quantile_by_measure`](@ref)
  - [`PopulationPredictionResult`](@ref)
"""
function successful_members(ppred::PopulationPredictionResult)
    return filter(member_solved, ppred.pred)
end
"""
    lacks_id(p::PredictionResult)
    lacks_id(p::MultiPeriodPredictionResult)

Says whether the population member `p` needs an `id`.

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

Gives the population member `p` the `id` `i` when it lacks one.

The type of the member's `id` selects the method, so the function makes no test at run time. The method for an `id` of `nothing` rebuilds the member with `id = i`, and keeps its folds and its estimator. The other methods return the member unchanged.

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

Gives every member of a population an `id`, so that the `id` of a selected path is its position in the population.

The constructor of [`PopulationPredictionResult`](@ref) calls it. A member whose `id` is `nothing` takes its position in `pred` as its `id`. [`CombinatorialCrossValidation`](@ref) and [`MultipleRandomised`](@ref) number their paths by position too. When no member [`lacks_id`](@ref), the function returns the same vector, so it does not copy a population that a scheme made.

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
