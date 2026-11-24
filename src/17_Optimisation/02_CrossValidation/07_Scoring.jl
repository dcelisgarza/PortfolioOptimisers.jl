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
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the rules of the slot `r_sca` of a search that a scalariser cannot spell.

A [`Sca_VecToScaM`](@ref) in `r_sca` combines the risk measures of `r` in each fold or path, so the search keeps one score matrix. A member of this family keeps one score matrix for each measure, so the search scores each fold under each measure apart. The members differ in what the search does with the `K` matrices:

  - [`EachValue`](@ref) combines nothing, and gives one ranking for each measure.
  - [`PerCandidate`](@ref) combines the measures once per candidate, after the search scorer reduces the rows of each matrix, and gives one ranking.

So [`oriented_score`](@ref) gives the oriented score of each measure under every member, and the allocator of the scores gives one matrix for each measure.

[`expected_risk`](@ref), [`NearestQuantilePrediction`](@ref) and [`FoldRiskScorer`](@ref) need one value for each sample, so their bounds refuse every member.

# Interfaces

The family is closed. The selection of a search dispatches on [`EachValue`](@ref) and on [`PerCandidate`](@ref), and holds the methods of each for its `K` score matrices. A new member needs those methods, so the library adds no member.

# Related

  - [`EachValue`](@ref)
  - [`PerCandidate`](@ref)
  - [`GridSearchCrossValidation`](@ref)
  - [`Sca_VecToScaM`](@ref)
"""
abstract type AbstractSearchReduction <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The per-value rule of a scalariser slot of a search: keep each value apart, and give one ranking for each value per candidate.

It is the multi-metric form of a search in sklearn. The search does not combine the values. It orders the candidates by each value apart, and it keeps one `opt`, one `idx` and one ranking for each value, in one [`SearchCrossValidationResult`](@ref).

  - In `r_sca` with a vector of `K` risk measures, the search keeps one score matrix for each measure. Each matrix holds the oriented score of its measure, multiplied by the scale of the measure, see [`oriented_score`](@ref).
  - In `s_sca` with a vector of `L` search scorers, each scorer orders the candidates alone. Its values are multiplied by its [`scorer_scale`](@ref).
  - In both, the search keeps a `K × L` matrix of rankings. Row `k` is measure `k`, and column `l` is scorer `l`.

For one measure in `r`, or one scorer in `scorer`, the rule is inert, as a reduction is. The types of `r`, `r_sca`, `scorer` and `s_sca` decide the shape of the Result, never the count at run time. So a vector of one measure under `r_sca = EachValue()` gives a vector of one matrix.

The search takes no refit. A caller who wants the winner of value `v` fitted on the whole sample calls `optimise(res.opt[v], rd)`.

The rule is a rule of [`GridSearchCrossValidation`](@ref) and [`RandomisedSearchCrossValidation`](@ref) alone. It is not a [`Scalariser`](@ref) and not a member of [`Sca_VecToScaM`](@ref). [`expected_risk`](@ref), [`NearestQuantilePrediction`](@ref) and [`FoldRiskScorer`](@ref) need one value for each sample, so their bounds refuse it.

# Constructors

    EachValue() -> EachValue

# Examples

```jldoctest
julia> GridSearchCrossValidation([\"opt.l1\" => [0.0, 0.01]];
                                 r = [ConditionalValueatRisk(), MeanReturn()], r_sca = EachValue()).r_sca
EachValue()
```

# Related

  - [`GridSearchCrossValidation`](@ref)
  - [`SearchCrossValidationResult`](@ref)
  - [`Sca_VecToScaM`](@ref)
  - [`finite_candidate_order`](@ref)
  - [`AbstractSearchReduction`](@ref)
  - [`PerCandidate`](@ref)
"""
struct EachValue <: AbstractSearchReduction end
"""
$(DocStringExtensions.TYPEDEF)

The per-candidate rule of the slot `r_sca` of a search: combine the risk measures once per candidate, after the search scorer reduces the folds or the paths of each measure.

A [`Sca_VecToScaM`](@ref) in `r_sca` combines the `K` measures of `r` in each fold, or in each path of a combinatorial scheme, and then the search scorer reduces the rows. `PerCandidate(sca)` takes the stages in the other order. The search keeps one score matrix for each measure, as under [`EachValue`](@ref). The search scorer reduces the rows of each matrix, and gives the value ``v_{kj}`` of measure ``k`` for candidate ``j``. Then `sca` combines the `K` values of each candidate to one. The search gives one ranking, as for one measure.

For a sum, and a search scorer that is linear in the rows, such as [`HighestMeanScore`](@ref), the two orders give the same value. For [`MaxScalariser`](@ref), [`MinScalariser`](@ref), [`LogSumExpScalariser`](@ref), a [`VectorToScalarMeasure`](@ref) or a function they do not. The mean over the folds of the best measure is not the best measure of the means. For one candidate, two measures and three folds:

```text
            fold 1  fold 2  fold 3   mean
measure 1      3       0       0      1
measure 2      0       2       1      1

r_sca = MaxScalariser():               max per fold [3, 2, 1], then mean = 2
r_sca = PerCandidate(MaxScalariser()): mean per measure [1, 1], then max = 1
```

The first asks how good the best measure is in each fold. The second asks how good the best measure is over all the folds.

# Mathematical definition

```math
\\begin{align}
v_{kj} &= c\\left(\\mathbf{S}^{(k)}_{:,\\,\\mathcal{C}}\\right)_j\\,, \\\\
u_j &= S\\left(v_{1j}, \\, \\ldots, \\, v_{Kj}\\right)\\,.
\\end{align}
```

Where:

  - ``\\mathbf{S}^{(k)}``: Score matrix of measure ``k``. Each entry is the oriented score of the measure, multiplied by its scale, see [`oriented_score`](@ref). So the scale of a measure is its weight in `sca`.
  - ``\\mathcal{C}``: The candidates whose every entry is finite in **every** matrix. A value of fewer than ``K`` measures does not compare with a value of ``K``.
  - ``c``: The search scorer, through [`candidate_values`](@ref).
  - ``v_{kj}``: Value of measure ``k`` for candidate ``j``.
  - ``S``: The reduction `sca`.
  - ``u_j``: Value of candidate ``j`` for the scorer. A vector of search scorers multiplies ``u_j`` of each scorer by its [`scorer_scale`](@ref), and `s_sca` reduces them, as for one measure.
  - ``K``: Count of the risk measures in `r`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PerCandidate(; sca::Option{<:Sca_VecToScaM} = nothing) -> PerCandidate

`PerCandidate(sca)` takes the reduction as the one positional argument.

# Validation

  - When the search runs, a search scorer that gives the index of one winner is refused with an `ArgumentError`, because an index has no value to combine. For one measure in `r` it is admitted.
  - When the search runs, a reduction that gives `NaN` is refused with an [`IsNonFiniteError`](@ref).

For one measure in `r` the rule is inert, as a reduction is. The search scorer of a vector of one measure gives one value, and `sca` reduces that one value.

The rule is a rule of the slot `r_sca` of [`GridSearchCrossValidation`](@ref) and [`RandomisedSearchCrossValidation`](@ref) alone. The slot `s_sca` refuses it, because no stage follows the scorers. [`expected_risk`](@ref), [`NearestQuantilePrediction`](@ref) and [`FoldRiskScorer`](@ref) have no scorer stage, so their bounds refuse it.

# Examples

```jldoctest
julia> PerCandidate(MaxScalariser())
PerCandidate
  sca ┴ MaxScalariser()

julia> GridSearchCrossValidation([\"opt.l1\" => [0.0, 0.01]];
                                 r = [ConditionalValueatRisk(), MeanReturn()],
                                 r_sca = PerCandidate(MaxScalariser())).r_sca.sca
MaxScalariser()
```

# Related

  - [`AbstractSearchReduction`](@ref)
  - [`EachValue`](@ref)
  - [`GridSearchCrossValidation`](@ref)
  - [`Sca_VecToScaM`](@ref)
  - [`candidate_values`](@ref)
  - [`finite_candidate_order`](@ref)
"""
struct PerCandidate{T} <: AbstractSearchReduction
    """
    $(field_dict[:sca_candidate])
    """
    sca::T
    # The struct is written by hand, and not with `@concrete`, because `@concrete` appends a
    # generic positional constructor that skips this one. `PerCandidate(1.0)` would then hold
    # a reduction that is not a `Sca_VecToScaM`. The positional form is the main form of this
    # type.
    function PerCandidate(sca::Option{<:Sca_VecToScaM})
        return new{typeof(sca)}(sca)
    end
end
function PerCandidate(; sca::Option{<:Sca_VecToScaM} = nothing)
    return PerCandidate(sca)
end
"""
    oriented_score(sca::Union{Option{<:Sca_VecToScaM}, AbstractSearchReduction},
                   r::BaseRM_VecBaseRM, p::PredRes_MultiPredRes; kwargs...)
    oriented_score(sca::Option{<:Scalariser}, rs::VecBaseRM, X::VecNum; kwargs...)
    oriented_score(sca::Union{VectorToScalarMeasure, Function}, rs::VecBaseRM, X::VecNum;
                   kwargs...)
    oriented_score(::AbstractSearchReduction, rs::VecBaseRM, X::VecNum; kwargs...)
    oriented_score(sca::Union{Option{<:Sca_VecToScaM}, AbstractSearchReduction},
                   r::AbstractBaseRiskMeasure, res::OptimisationResult,
                   X::Option{<:MatNum} = nothing; kwargs...)
    oriented_score(sca::Option{<:Scalariser}, rs::VecBaseRM, res::OptimisationResult,
                   X::Option{<:MatNum} = nothing; kwargs...)
    oriented_score(sca::Union{VectorToScalarMeasure, Function}, rs::VecBaseRM,
                   res::OptimisationResult, X::Option{<:MatNum} = nothing; kwargs...)
    oriented_score(::AbstractSearchReduction, rs::VecBaseRM, res::OptimisationResult,
                   X::Option{<:MatNum} = nothing; kwargs...)

Score a prediction, or a fitted result over its own training sample, under one risk measure or a vector of them, so that a higher score is always better.

A vector of measures also scores a return series `X`, one sample of a prediction. [`FoldRiskScorer`](@ref) scores the returns of one asset over one test window in this way. The form for a fitted result scores the result. With `X = nothing` it scores the result through its prior result, as `expected_risk(r, res)` does. With a matrix `X` it scores the result over `X`, as `expected_risk(r, res, X)` does. A search calls it through [`candidate_train_score`](@ref) for its train scores.

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

  - ``\\mathcal{O}(P)``: Oriented score of the prediction or the fitted result ``P``. The first line is the case of one measure, and the second line the case of a vector.
  - ``\\mathcal{R}(P)``, ``\\mathcal{R}_k(P)``: Expected risk of ``P`` under the measure, or under the ``k``-th measure of the vector. For a fitted result, it is the risk over its training sample.
  - ``o``, ``o_k``: Orientation of the measure, from [`bigger_is_better`](@ref).
  - $(math_dict[:s_k_scale]) A negative scale is a negative weight, so that measure counts against the score.
  - ``S``: The reduction `sca`.
  - $(math_dict[:n_rm_vec])

One measure ignores `sca` and its scale, as [`expected_risk`](@ref) does. A search calls the function with a rule of [`AbstractSearchReduction`](@ref) in `sca`, such as [`EachValue`](@ref) or [`PerCandidate`](@ref). For a vector, such a rule does not reduce: it gives the vector of the values ``o_k s_k \\mathcal{R}_k(P)``, in the order of `rs`. For one measure it is inert.

# Algorithm

 1. One measure: return ``o \\, \\mathcal{R}(P)``, with ``\\mathcal{R}(P)`` from [`expected_risk`](@ref) with `kwargs`, through [`oriented_value`](@ref).
 2. A vector of a prediction: read the return series of ``P`` once with [`prediction_returns`](@ref), and score the series as in step 3.
 3. A vector of a series or of a fitted result: compute ``o_k s_k \\mathcal{R}_k(P)`` for each measure on the sample, with [`oriented_member_score`](@ref) with `kwargs`. Reduce the values by the type of `sca`:
     1. A [`Scalariser`](@ref) or `nothing`: through [`scalarise`](@ref). `nothing` gives the sum of [`SumScalariser`](@ref).
     2. A [`VectorToScalarMeasure`](@ref) or a function: collect the values into a vector, and apply [`vec_to_real_measure`](@ref).
     3. A rule of [`AbstractSearchReduction`](@ref): collect the values into a vector, and return the vector. The search combines them later, or not at all.

The reduction of step 3 is written twice, once for a series and once for a fitted result, so that each sample is a method of this function and no helper holds a part of the job. The two copies agree: each reads the members through [`oriented_member_score`](@ref), and each reduces by the type of `sca` in the same two ways. The tests of a search hold the copy for a fitted result, and the tests of a scorer of predictions and of [`FoldRiskScorer`](@ref) hold the copy for a series.

# Arguments

  - `r`: One risk measure, or a vector of them.
  - `rs`: A vector of risk measures.
  - `p`: A prediction of one fold, a [`PredictionResult`](@ref), or of one path, a [`MultiPeriodPredictionResult`](@ref).
  - `res`: A fitted result.
  - `X`: In the form for a series, the return series. In the form for a fitted result, the training returns of `res`, or `nothing` to score `res` through its prior result.
  - `sca`: The reduction of a vector. A [`VectorToScalarMeasure`](@ref) or a function must take a vector of numbers. `nothing` gives the sum of [`SumScalariser`](@ref). A rule of [`AbstractSearchReduction`](@ref) keeps each value.
  - `kwargs...`: Keyword arguments forwarded to [`expected_risk`](@ref).

# Returns

  - `score::Number`: The oriented score.
  - `scores::VecNum`: Under a rule of [`AbstractSearchReduction`](@ref) with a vector, the oriented score of each measure, multiplied by its scale.

# Related

  - [`Sca_VecToScaM`](@ref)
  - [`bigger_is_better`](@ref)
  - [`prediction_returns`](@ref)
  - [`oriented_member_score`](@ref)
  - [`sort_by_measure`](@ref): sorts the paths of a population by this score.
  - [`quantile_by_measure`](@ref): selects a path by the quantile of the negated score.
  - [`candidate_train_score`](@ref): the train score of a search, through the form for a fitted result.
"""
function oriented_score(::Union{Option{<:Sca_VecToScaM}, AbstractSearchReduction},
                        r::AbstractBaseRiskMeasure, p::PredRes_MultiPredRes; kwargs...)
    return oriented_value(r, expected_risk(r, p; kwargs...))
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
function oriented_score(sca::Union{Option{<:Sca_VecToScaM}, AbstractSearchReduction},
                        rs::VecBaseRM, p::PredRes_MultiPredRes; kwargs...)
    return oriented_score(sca, rs, prediction_returns(p); kwargs...)
end
function oriented_score(::AbstractSearchReduction, rs::VecBaseRM, X::VecNum; kwargs...)
    return map(r -> oriented_member_score(r, X; kwargs...), rs)
end
function oriented_score(sca::Option{<:Scalariser}, rs::VecBaseRM, X::VecNum; kwargs...)
    return scalarise(r -> oriented_member_score(r, X; kwargs...), sca, rs)
end
function oriented_score(sca::Union{VectorToScalarMeasure, Function}, rs::VecBaseRM,
                        X::VecNum; kwargs...)
    return vec_to_real_measure(sca, map(r -> oriented_member_score(r, X; kwargs...), rs))
end
#=
A fitted result is scored once per fold, for a train score, so the vector form passes the
result to each member. Each member then resolves the result exactly as one measure does,
through `expected_risk(r, res, X)`: the investable view, the prior result and a Deferred
Quantity. The box of the result on each member call is paid by a vector alone.
=#
function oriented_score(::Union{Option{<:Sca_VecToScaM}, AbstractSearchReduction},
                        r::AbstractBaseRiskMeasure, res::OptimisationResult,
                        X::Option{<:MatNum} = nothing; kwargs...)
    return oriented_value(r, expected_risk(r, res, X; kwargs...))
end
function oriented_score(sca::Option{<:Scalariser}, rs::VecBaseRM, res::OptimisationResult,
                        X::Option{<:MatNum} = nothing; kwargs...)
    return scalarise(r -> oriented_member_score(r, res, X; kwargs...), sca, rs)
end
function oriented_score(sca::Union{VectorToScalarMeasure, Function}, rs::VecBaseRM,
                        res::OptimisationResult, X::Option{<:MatNum} = nothing; kwargs...)
    return vec_to_real_measure(sca,
                               map(r -> oriented_member_score(r, res, X; kwargs...), rs))
end
function oriented_score(::AbstractSearchReduction, rs::VecBaseRM, res::OptimisationResult,
                        X::Option{<:MatNum} = nothing; kwargs...)
    return map(r -> oriented_member_score(r, res, X; kwargs...), rs)
end
"""
    oriented_value(r::AbstractBaseRiskMeasure, x)

Orient the value `x` of the risk measure `r`, so that a higher value is better.

# Arguments

  - `r`: The risk measure of `x`.
  - `x`: A value of `r`.

# Returns

  - `x`: `x` when [`bigger_is_better`](@ref) holds for `r`, and `-x` otherwise. The orientation is a trait of the type of `r`, so the compiler drops the branch.

# Related

  - [`oriented_score`](@ref)
  - [`oriented_member_score`](@ref)
"""
function oriented_value(r::AbstractBaseRiskMeasure, x)
    return bigger_is_better(r) ? x : -x
end
"""
    oriented_member_score(r::AbstractBaseRiskMeasure, X::VecNum; kwargs...)
    oriented_member_score(r::AbstractBaseRiskMeasure, res::OptimisationResult,
                          X::Option{<:MatNum}; kwargs...)

Score one member of a vector of risk measures on a return series, or a fitted result over its training sample, oriented so that a higher value is better and multiplied by its scale.

It is the value ``o_k s_k \\mathcal{R}_k(P)`` of [`oriented_score`](@ref), which reduces these values of a vector. The vector methods of [`oriented_score`](@ref) call this function once for each member, on one sample.

# Arguments

  - `r`: One member of the vector.
  - `X`: The return series of the prediction. In the second form, the training returns of `res`, or `nothing` to score `res` through its prior result.
  - `res`: A fitted result.
  - `kwargs...`: Keyword arguments forwarded to [`expected_risk_from_returns`](@ref), or to [`expected_risk`](@ref) for a fitted result.

# Returns

  - `score::Number`: The value of `r`, multiplied by `-1` when [`bigger_is_better`](@ref) does not hold for `r`, and by `r.settings.scale`.

# Related

  - [`oriented_score`](@ref)
  - [`oriented_value`](@ref)
  - [`prediction_returns`](@ref)
"""
function oriented_member_score(r::AbstractBaseRiskMeasure, X::VecNum; kwargs...)
    return oriented_value(r, expected_risk_from_returns(r, X; kwargs...)) * r.settings.scale
end
function oriented_member_score(r::AbstractBaseRiskMeasure, res::OptimisationResult,
                               X::Option{<:MatNum}; kwargs...)
    return oriented_value(r, expected_risk(r, res, X; kwargs...)) * r.settings.scale
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
    assert_path_reduction(kwargs, name::AbstractString)

Check the keyword arguments of a reader that gives one score for each path of a population.

[`NearestQuantilePrediction`](@ref), [`sort_by_measure`](@ref) and [`quantile_by_measure`](@ref) call it before they score a path. The reduction is the argument `sca`, so a `sca` in the keyword arguments that the reader forwards to [`oriented_score`](@ref) is refused. A reader of a risk takes every keyword that it does not declare, so it would ignore that `sca` without an error. The keys of a `NamedTuple` are part of its type, so the check costs nothing at run time.

# Arguments

  - `kwargs`: The keyword arguments that the reader forwards to [`oriented_score`](@ref).
  - `name`: The name of `kwargs` in the call of the caller, for the error message.

# Validation

  - `kwargs` must not contain `sca`. The error is a [`ConflictingArgumentError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`oriented_score`](@ref)
"""
function assert_path_reduction(kwargs, name::AbstractString)
    @argcheck(!haskey(kwargs, :sca),
              ConflictingArgumentError("`$name` must not contain `sca`: the reduction of a vector of risk measures is the argument `sca`, or the field `sca` of a `NearestQuantilePrediction`. A reader of a risk ignores a `sca` in `$name`. Got\n$name.sca => $(get(kwargs, :sca, nothing))"))
    return nothing
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
  - ``\\mathcal{O}(P_i)``: Oriented score of path ``i`` under `r`, from [`oriented_score`](@ref), so a higher score is better. The field `sca` reduces a vector of risk measures to one value, and the default `nothing` gives the sum of [`SumScalariser`](@ref).
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
        sca::Option{<:Sca_VecToScaM} = nothing,
        q::Real = 0.5,
        r_kwargs::NamedTuple = (;),
        q_kwargs::NamedTuple = (;)
    ) -> NearestQuantilePrediction

## Multiplicity

`r` takes one risk measure or a vector of them, and the field `sca` reduces a vector, as in `sca = MinScalariser()` or `sca = MedianValue()`. The default `nothing` gives the sum. The reduction reads oriented scores, so [`MinScalariser`](@ref) selects by the worst measure of each path and [`MaxScalariser`](@ref) by the best. For one measure `sca` is inert.

The scorer admits a vector whose members disagree on orientation, because [`oriented_score`](@ref) orients each member before the reduction.

## Validation

  - $(val_dict[:q_scorer])
  - `r_kwargs` must not contain `sca`, through [`assert_path_reduction`](@ref). The reduction is the field `sca`.

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
        assert_path_reduction(r_kwargs, "r_kwargs")
        return new{typeof(r), typeof(sca), typeof(q), typeof(r_kwargs), typeof(q_kwargs)}(r,
                                                                                          sca,
                                                                                          q,
                                                                                          r_kwargs,
                                                                                          q_kwargs)
    end
end
function NearestQuantilePrediction(; r::BaseRM_VecBaseRM = ConditionalValueatRisk(),
                                   sca::Option{<:Sca_VecToScaM} = nothing, q::Real = 0.5,
                                   r_kwargs::NamedTuple = (;),
                                   q_kwargs::NamedTuple = (;))::NearestQuantilePrediction
    return NearestQuantilePrediction(r, sca, q, r_kwargs, q_kwargs)
end
function (s::NearestQuantilePrediction)(ppred::PopulationPredictionResult)
    return quantile_by_measure(s.sca, ppred, s.r, s.q; r_kwargs = s.r_kwargs,
                               q_kwargs = s.q_kwargs)
end

export NearestQuantilePrediction, EachValue, PerCandidate
# The closed family of the rules of `r_sca` that keep one score matrix per measure, with its
# `# Interfaces` section (ADR 0154, #2002).
public AbstractSearchReduction
"""
    member_solved(p::PredictionResult)
    member_solved(p::MultiPeriodPredictionResult)

Says whether every fold of the population member `p` solved.

A [`PredictionResult`](@ref) is one fold, and it solved when [`fold_solved`](@ref) holds of its return code. A fold that holds a frontier sweep solved when every point did. A [`MultiPeriodPredictionResult`](@ref) solved when each of its folds solved.

# Arguments

  - `p`: A member of a [`PopulationPredictionResult`](@ref).

# Returns

  - `flag::Bool`: `true` when every fold of `p` solved.

# Related

  - [`fold_solved`](@ref)
  - [`successful_members`](@ref)
"""
function member_solved(p::PredictionResult)
    return fold_solved(p.res.retcode)
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

# Validation

  - Every member holds one return series, `ArgumentError` otherwise. A member that holds a frontier sweep has one score for each point, so it has no rank.

# Returns

  - `pred::VecPredRes_MultiPredRes`: The members for which [`member_solved`](@ref) holds, in population order.

# Related

  - [`member_solved`](@ref)
  - [`prediction_returns`](@ref)
  - [`sort_by_measure`](@ref)
  - [`quantile_by_measure`](@ref)
  - [`PopulationPredictionResult`](@ref)
"""
function successful_members(ppred::PopulationPredictionResult)
    @argcheck(all(p -> isa(prediction_returns(p), VecNum), ppred.pred),
              ArgumentError("a path that holds a frontier sweep has one score for each point and no rank; score one point, or optimise without a `Frontier`"))
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
