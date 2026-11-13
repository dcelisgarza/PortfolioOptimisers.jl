"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the cross-validation scorers.

A scorer reads the scores of a cross-validation and selects one of the items it scored. The family has two branches:

  - [`PredictionScorer`](@ref) selects one path from the population of paths that a cross-validation scheme returns.
  - [`CrossValidationSearchScorer`](@ref) selects one candidate of a search from the matrix of its cross-validation scores.

# Related

  - [`PredictionScorer`](@ref)
  - [`CrossValidationSearchScorer`](@ref)
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
    const PredictionCrossValScorer = Union{PredictionScorer, Function}

Union of the values that the `scorer` field of [`NestedClustered`](@ref) and [`Stacking`](@ref) accepts: a [`PredictionScorer`](@ref) or a plain function.

The union exists so that a caller can select a path with a function and write no type. The function takes a [`PopulationPredictionResult`](@ref) and returns one of its members, as a [`PredictionScorer`](@ref) does.

# Related

  - [`PredictionScorer`](@ref)
  - [`NearestQuantilePrediction`](@ref)
  - [`NestedClustered`](@ref)
  - [`Stacking`](@ref)
"""
const PredictionCrossValScorer = Union{PredictionScorer, Function}
"""
    oriented_score(sca::Option{<:Sca_VecToScaM}, r::BaseRM_VecBaseRM, p::PredRes_MultiPredRes;
                   kwargs...)

Score a prediction under one risk measure or a vector of them, so that a higher score is always better.

The function orients each measure before it reduces the vector. So a vector whose measures disagree on orientation gives a score with one orientation, and [`bigger_is_better`](@ref) of the vector is never read. Because the values are oriented, [`MaxScalariser`](@ref) and [`MaxValue`](@ref) give the score of the best measure, and [`MinScalariser`](@ref) and [`MinValue`](@ref) give the score of the worst. The function reads every reduction as "higher is better", so [`StdValue`](@ref) gives the highest score to the largest spread.

# Mathematical definition

```math
\\begin{align}
\\mathcal{O}(P) &= o \\, \\mathcal{R}(P)\\,, \\\\
\\mathcal{O}(P) &= S\\left(o_1 s_1 \\mathcal{R}_1(P), \\, \\ldots, \\, o_n s_n \\mathcal{R}_n(P)\\right)\\,, \\\\
o_k &= \\begin{cases} 1 & \\mathcal{R}_k \\ \\mathrm{bigger\\ is\\ better} \\\\ -1 & \\mathrm{otherwise} \\end{cases}\\,.
\\end{align}
```

Where:

  - ``\\mathcal{O}(P)``: Oriented score of the prediction ``P``. The first line is the case of one measure, and the second line the case of a vector.
  - ``\\mathcal{R}(P)``, ``\\mathcal{R}_k(P)``: Expected risk of ``P`` under the measure, or under the ``k``-th measure of the vector.
  - ``o``, ``o_k``: Orientation of the measure, from [`bigger_is_better`](@ref).
  - $(math_dict[:s_k_scale]) A negative scale is a negative weight, so that measure counts against the score.
  - ``S``: The reduction `sca`.
  - $(math_dict[:n_rm_vec])

One measure ignores `sca` and its scale, as [`expected_risk`](@ref) does.

# Algorithm

 1. One measure: return ``o \\, \\mathcal{R}(P)``, with ``\\mathcal{R}(P)`` from [`expected_risk`](@ref) with `kwargs`.
 2. A vector: read the return series of ``P`` once with [`prediction_returns`](@ref), and score that series with [`oriented_returns_score`](@ref) with `kwargs`. That function computes ``o_k s_k \\mathcal{R}_k(P)`` for each measure, and reduces the values by the type of `sca`.

# Arguments

  - `r`: One risk measure, or a vector of them.
  - `p`: A prediction of one fold, a [`PredictionResult`](@ref), or of one path, a [`MultiPeriodPredictionResult`](@ref).
  - `sca`: The reduction of a vector. A [`VectorToScalarMeasure`](@ref) or a function must take a vector of numbers.
  - `kwargs...`: Keyword arguments forwarded to [`expected_risk`](@ref).

# Returns

  - `score::Number`: The oriented score. A vector under `sca = nothing` gives `scores::Vector`, one oriented score for each measure.

# Related

  - [`Sca_VecToScaM`](@ref)
  - [`bigger_is_better`](@ref)
  - [`prediction_returns`](@ref)
  - [`oriented_returns_score`](@ref)
  - [`sort_by_measure`](@ref): sorts the paths of a population by this score.
  - [`quantile_by_measure`](@ref): selects a path by the quantile of the negated score.
"""
function oriented_score(::Option{<:Sca_VecToScaM}, r::AbstractBaseRiskMeasure,
                        p::PredRes_MultiPredRes; kwargs...)
    x = expected_risk(r, p; kwargs...)
    return bigger_is_better(r) ? x : -x
end
#=
The orientation is a trait of the type, so the compiler drops each branch on it, and a
negation in the caller cancels the one of a measure that a smaller value makes better. So one
measure costs what `expected_risk` costs. The members of a mixed vector have an abstract type,
and each call on a member dispatches at run time. A vector method therefore reads the returns
of the prediction once and makes one call on each member, `oriented_member_score`, whose
arithmetic is on concrete types. A call that passed the prediction would box it once for each
member. One measure reads `expected_risk` on the prediction, so that a measure that refuses a
prediction, such as `ExpectedReturn`, still refuses it.
=#
function oriented_score(sca::Option{<:Sca_VecToScaM}, rs::VecBaseRM,
                        p::PredRes_MultiPredRes; kwargs...)
    return oriented_returns_score(sca, rs, prediction_returns(p); kwargs...)
end
"""
    oriented_returns_score(sca::Scalariser, rs::VecBaseRM, X::VecNum; kwargs...)
    oriented_returns_score(sca::Union{VectorToScalarMeasure, Function}, rs::VecBaseRM,
                           X::VecNum; kwargs...)
    oriented_returns_score(sca::Nothing, rs::VecBaseRM, X::VecNum; kwargs...)

Score a return series under a vector of risk measures, so that a higher score is always better.

It is the score ``\\mathcal{O}`` of [`oriented_score`](@ref) for a vector, on a series and not on a prediction. [`oriented_score`](@ref) calls it on the series of a prediction. [`FoldRiskScorer`](@ref) calls it on the returns of one asset over one test window.

# Algorithm

 1. Compute ``o_k s_k \\mathcal{R}_k`` for each measure on `X`, with [`oriented_member_score`](@ref) with `kwargs`.
 2. Reduce the values of step 1 by the type of `sca`:
     1. A [`Scalariser`](@ref): through [`scalarise`](@ref).
     2. A [`VectorToScalarMeasure`](@ref) or a function: collect the values into a vector, and apply [`vec_to_real_measure`](@ref).
     3. `nothing`: return the vector of the values, so each measure keeps its own score.

# Arguments

  - `sca`: The reduction of the vector. A [`VectorToScalarMeasure`](@ref) or a function must take a vector of numbers.
  - `rs`: The vector of risk measures.
  - `X`: The return series.
  - `kwargs...`: Keyword arguments forwarded to [`expected_risk_from_returns`](@ref).

# Returns

  - `score::Number`: The oriented score. Under `sca = nothing`, `scores::Vector`, one oriented score for each measure.

# Related

  - [`oriented_score`](@ref)
  - [`oriented_member_score`](@ref)
"""
function oriented_returns_score(sca::Scalariser, rs::VecBaseRM, X::VecNum; kwargs...)
    return scalarise(r -> oriented_member_score(r, X; kwargs...), sca, rs)
end
function oriented_returns_score(sca::Union{VectorToScalarMeasure, Function}, rs::VecBaseRM,
                                X::VecNum; kwargs...)
    return vec_to_real_measure(sca, oriented_returns_score(nothing, rs, X; kwargs...))
end
function oriented_returns_score(::Nothing, rs::VecBaseRM, X::VecNum; kwargs...)
    return map(r -> oriented_member_score(r, X; kwargs...), rs)
end
"""
    oriented_member_score(r::AbstractBaseRiskMeasure, X::VecNum; kwargs...)

Score one member of a vector of risk measures on a return series, oriented so that a higher value is better and multiplied by its scale.

It is the value ``o_k s_k \\mathcal{R}_k(P)`` of [`oriented_score`](@ref), which reduces these values of a vector. [`oriented_returns_score`](@ref) calls this function once for each member, on one return series.

# Arguments

  - `r`: One member of the vector.
  - `X`: The return series of the prediction.
  - `kwargs...`: Keyword arguments forwarded to [`expected_risk_from_returns`](@ref).

# Returns

  - `score::Number`: The value of `r` on `X`, multiplied by `-1` when [`bigger_is_better`](@ref) does not hold for `r`, and by `r.settings.scale`.

# Related

  - [`oriented_score`](@ref)
  - [`prediction_returns`](@ref)
"""
function oriented_member_score(r::AbstractBaseRiskMeasure, X::VecNum; kwargs...)
    x = expected_risk_from_returns(r, X; kwargs...)
    return (bigger_is_better(r) ? x : -x) * r.settings.scale
end
"""
    prediction_returns(p::PredictionResult)
    prediction_returns(p::MultiPeriodPredictionResult)

Give the realised return series of a prediction, the series that [`expected_risk`](@ref) reads.

A [`PredictionResult`](@ref) is one fold, and its series is `p.rd.X`. A [`MultiPeriodPredictionResult`](@ref) is one path, and its series is `p.mrd.X`, the folds of the path in one series.

# Arguments

  - `p`: A prediction of one fold or of one path.

# Returns

  - `X::VecNum`: The return series of `p`.

# Related

  - [`oriented_score`](@ref)
  - [`expected_risk`](@ref)
"""
function prediction_returns(p::PredictionResult)
    return p.rd.X
end
function prediction_returns(p::MultiPeriodPredictionResult)
    return p.mrd.X
end
"""
    assert_path_reduction(sca::Option{<:Sca_VecToScaM}, r::BaseRM_VecBaseRM, kwargs,
                          name::AbstractString)

Check the reduction of a reader that gives one score for each path of a population.

[`NearestQuantilePrediction`](@ref), [`sort_by_measure`](@ref) and [`quantile_by_measure`](@ref) call it before they score a path. The reduction is the argument `sca`, so a `sca` in the keyword arguments that the reader forwards to [`oriented_score`](@ref) is refused. A reader of a risk takes every keyword that it does not declare, so it would ignore that `sca` without an error. The keys of a `NamedTuple` are part of its type, so the check costs nothing at run time.

# Arguments

  - $(arg_dict[:sca_path])
  - `r`: One risk measure, or a vector of them.
  - `kwargs`: The keyword arguments that the reader forwards to [`oriented_score`](@ref).
  - `name`: The name of `kwargs` in the call of the caller, for the error message.

# Validation

  - `kwargs` must not contain `sca`. The error is a [`ConflictingArgumentError`](@ref).
  - [`assert_one_score_per_path`](@ref) checks `sca` against `r`.

# Returns

  - `nothing`.

# Related

  - [`assert_one_score_per_path`](@ref)
  - [`oriented_score`](@ref)
"""
function assert_path_reduction(sca::Option{<:Sca_VecToScaM}, r::BaseRM_VecBaseRM, kwargs,
                               name::AbstractString)
    @argcheck(!haskey(kwargs, :sca),
              ConflictingArgumentError("`$name` must not contain `sca`: the reduction of a vector of risk measures is the argument `sca`, or the field `sca` of a `NearestQuantilePrediction`. A reader of a risk ignores a `sca` in `$name`. Got\n$name.sca => $(get(kwargs, :sca, nothing))"))
    return assert_one_score_per_path(sca, r)
end
"""
    assert_one_score_per_path(sca::Option{<:Sca_VecToScaM}, r::BaseRM_VecBaseRM)
    assert_one_score_per_path(sca::Nothing, r::VecBaseRM)

Refuse a reduction of `nothing` for a vector of risk measures, where a reader needs one score for each path.

Under `nothing`, [`oriented_score`](@ref) of a vector gives one score for each measure. A sort and a quantile need one number for each path, so the readers of a population refuse it. For one measure the reduction is inert, and `nothing` is admitted. The types of the arguments select the method, so the check makes no test at run time.

# Arguments

  - $(arg_dict[:sca_path])
  - `r`: One risk measure, or a vector of them.

# Validation

  - `sca` must not be `nothing` when `r` is a vector. The error is an [`IsNothingError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`assert_path_reduction`](@ref)
  - [`NearestQuantilePrediction`](@ref)
"""
function assert_one_score_per_path(::Option{<:Sca_VecToScaM}, ::BaseRM_VecBaseRM)
    return nothing
end
function assert_one_score_per_path(::Nothing, ::VecBaseRM)
    return throw(IsNothingError("`sca` must not be `nothing` when `r` is a vector of risk measures: a path needs one score to be sorted or to take part in a quantile, and `nothing` keeps one score for each measure. Pass a reduction such as `SumScalariser()` or `MaxScalariser()`."))
end
"""
$(DocStringExtensions.TYPEDEF)

Selects the path of a population whose risk is nearest to a target quantile of the population's risks.

It is the scorer that [`NestedClustered`](@ref) and [`Stacking`](@ref) use under [`CombinatorialCrossValidation`](@ref) when their `scorer` is `nothing`. A path with a failed fold, or with a risk that is not finite, takes no part in the quantile, and the scorer never selects it.

# Mathematical definition

```math
\\begin{align}
\\rho_i &= -\\mathcal{O}(P_i)\\,, \\quad i \\in \\mathcal{S}\\,, \\\\
i^{\\star} &= \\underset{i \\in \\mathcal{S}}{\\arg\\min} \\left| \\rho_i - Q_q\\left(\\{\\rho_j\\}_{j \\in \\mathcal{S}}\\right) \\right|\\,.
\\end{align}
```

Where:

  - ``P_i``: Path ``i`` of the population.
  - ``\\mathcal{S}``: Set of the paths whose every fold solved and whose ``\\rho_i`` is finite.
  - ``\\mathcal{O}(P_i)``: Oriented score of path ``i`` under `r`, from [`oriented_score`](@ref), so a higher score is better. The field `sca` reduces a vector of risk measures to one value, and the default is [`SumScalariser`](@ref).
  - ``\\rho_i``: Risk of path ``i`` on a scale where a larger value is worse.
  - ``Q_q``: ``q``-th quantile of a set of values, as `Statistics.quantile` computes it with `q_kwargs`.
  - ``i^{\\star}``: Selected path.

``q = 0`` selects the best path and ``q = 1`` the worst path, whatever the quantile definition and whatever the orientation of `r`. For one risk measure ``\\rho_i`` is its expected risk, and for one measure of which a bigger value is better ``\\rho_i`` is the negated value.

# Algorithm

 1. Keep the members of `ppred` whose every fold solved, through [`successful_members`](@ref), giving `pred`.
 2. Compute `rks`, the vector of ``-\\mathcal{O}(P_i)`` over `pred`, through [`oriented_score`](@ref) with `sca` and `r_kwargs`.
 3. Drop every entry of `rks` that is not finite, and the member of `pred` at its position.
 4. Compute `rkq`, the `q`-th quantile of `rks`, through `Statistics.quantile` with `q_kwargs`.
 5. Scan `rks` in population order for the entry nearest to `rkq`, giving `idx`. A tie goes to the first entry.
 6. Return `pred[idx]`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NearestQuantilePrediction(;
        r::BaseRM_VecBaseRM = ConditionalValueatRisk(),
        sca::Option{<:Sca_VecToScaM} = SumScalariser(),
        q::Real = 0.5,
        r_kwargs::NamedTuple = (;),
        q_kwargs::NamedTuple = (;)
    ) -> NearestQuantilePrediction

## Multiplicity

`r` takes one risk measure or a vector of them, and the field `sca` reduces a vector, as in `sca = MinScalariser()` or `sca = MedianValue()`. The reduction reads oriented scores, so [`MinScalariser`](@ref) selects by the worst measure of each path and [`MaxScalariser`](@ref) by the best. For one measure `sca` is inert.

The scorer admits a vector whose members disagree on orientation, because [`oriented_score`](@ref) orients each member before the reduction.

## Validation

  - $(val_dict[:q_scorer])
  - `r_kwargs` must not contain `sca`, through [`assert_path_reduction`](@ref). The reduction is the field `sca`.
  - `sca` must not be `nothing` when `r` is a vector, through [`assert_one_score_per_path`](@ref). A quantile needs one score for each path.

# Functor

    (s::NearestQuantilePrediction)(ppred::PopulationPredictionResult)

Selects a path of `ppred` through [`quantile_by_measure`](@ref) and returns it.

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
    $(field_dict[:sca_path])
    """
    sca
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
    function NearestQuantilePrediction(r::BaseRM_VecBaseRM, sca::Option{<:Sca_VecToScaM},
                                       q::Real, r_kwargs::NamedTuple, q_kwargs::NamedTuple)
        @argcheck(zero(q) <= q <= one(q), DomainError(q, "`q` must be in [0, 1]"))
        assert_path_reduction(sca, r, r_kwargs, "r_kwargs")
        return new{typeof(r), typeof(sca), typeof(q), typeof(r_kwargs), typeof(q_kwargs)}(r,
                                                                                          sca,
                                                                                          q,
                                                                                          r_kwargs,
                                                                                          q_kwargs)
    end
end
function NearestQuantilePrediction(; r::BaseRM_VecBaseRM = ConditionalValueatRisk(),
                                   sca::Option{<:Sca_VecToScaM} = SumScalariser(),
                                   q::Real = 0.5, r_kwargs::NamedTuple = (;),
                                   q_kwargs::NamedTuple = (;))::NearestQuantilePrediction
    return NearestQuantilePrediction(r, sca, q, r_kwargs, q_kwargs)
end
function (s::NearestQuantilePrediction)(ppred::PopulationPredictionResult)
    return quantile_by_measure(s.sca, ppred, s.r, s.q; r_kwargs = s.r_kwargs,
                               q_kwargs = s.q_kwargs)
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
