"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all search-based cross-validation estimators.

Subtypes implement hyperparameter search strategies (e.g. grid search, randomised search) that use cross-validation to select the best estimator configuration.

# Related

  - [`GridSearchCrossValidation`](@ref)
  - [`RandomisedSearchCrossValidation`](@ref)
  - [`AbstractSearchCrossValidationResult`](@ref)
"""
abstract type AbstractSearchCrossValidationEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract result type for search-based cross-validation routines. Serves as the parent for all result types produced by search cross-validation algorithms, encapsulating optimal estimator, score matrices, parameter grid, and selected index.

# Interfaces

  - Subtypes must store the optimal estimator, test and train scores, parameter grid, and selected index.
"""
abstract type AbstractSearchCrossValidationResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all search-based cross-validation algorithm types.

Subtypes define the strategy used to select the best hyperparameter combination from the search results (e.g. selecting by highest mean score).

# Related

  - [`AbstractSearchCrossValidationEstimator`](@ref)
  - [`SearchCrossValidationResult`](@ref)
"""
abstract type AbstractSearchCrossValidationAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Defines the interface for scoring strategies used in search cross-validation. Implementations select the optimal parameter set based on cross-validation scores.

# Interfaces

  - `(::CrossValidationSearchScorer)(scores::AbstractMatrix)`: Returns the index of the optimal parameter set — the column with the best (highest) aggregate score.

## Arguments

  - `scores`: Matrix of scores, where each column corresponds to a parameter set and each row to a cross-validation split. The matrix is orientation-normalised: risk-measure scores are negated so that **higher is always better**, whatever measure `r` is. A scorer therefore returns the index of the column with the best (highest) aggregate score.

## Returns

  - `Int`: Index of the optimal parameter set, **as a position in the matrix the scorer received**.

## A scorer never sees a failed candidate

[`finite_candidate_index`](@ref) hands the scorer the columns whose every entry is finite, and maps the index back to the parameter grid itself. A scorer therefore reads a matrix of finite numbers alone, and it may compute anything on it — a mean, a spread, a rank — without a candidate that failed a fold winning. The matrix a scorer receives is a *view* of the score matrix, and its column count can be smaller than the grid, so a scorer must return a position in that view and must not index the grid itself.

# Examples

```jldoctest
julia> struct MyScore <: PortfolioOptimisers.CrossValidationSearchScorer end

julia> (s::MyScore)(X::Matrix{Float64}) = argmax(dropdims(mean(X; dims = 1); dims = 1))

julia> scores = [0.5 0.6; 0.7 0.8];

julia> scorer = MyScore()
MyScore()

julia> scorer(scores)
2
```

# Related

  - [`CrossValSearchScorer`](@ref)
  - [`HighestMeanScore`](@ref)
  - [`AbstractSearchCrossValidationEstimator`](@ref)
"""
abstract type CrossValidationSearchScorer <: AbstractEstimator end
"""
    const CrossValSearchScorer = Union{<:CrossValidationSearchScorer, <:Function}

Union type for search cross-validation scoring strategies. Accepts either a subtype of `CrossValidationSearchScorer` or a plain function that accepts a matrix and returns an integer.
"""
const CrossValSearchScorer = Union{<:CrossValidationSearchScorer, <:Function}
"""
$(DocStringExtensions.TYPEDEF)

Performs grid search cross-validation for portfolio optimisation estimators. Iterates over parameter grids, applies cross-validation splits, and scores each configuration to select the optimal parameters.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GridSearchCrossValidation(
        p::MultiGSCVValType_VecMultiGSCVValType;
        cv::CrossValidationEstimator = KFold(),
        r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
        scorer::CrossValSearchScorer = HighestMeanScore(),
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        train_score::Bool = false,
        kwargs::NamedTuple = (;),
    ) -> GridSearchCrossValidation

Positional and keyword arguments correspond to fields above.

## Validation

  - `!isempty(p)`.
  - If `p` is a vector of parameter sets: each element must not be empty.
  - All keys in `p` must be of type `GSCVKey` (i.e. `String`, `Symbol`, or `Integer`).

# Examples

```jldoctest
julia> GridSearchCrossValidation(Dict(\"alpha\" => [0.1, 0.2], \"beta\" => [1.0, 2.0]))
GridSearchCrossValidation
            p ┼ Dict{String, Vector{Float64}}: Dict("alpha" => [0.1, 0.2], "beta" => [1.0, 2.0])
           cv ┼ KFold
              │                   n ┼ Int64: 5
              │         purged_size ┼ Int64: 0
              │        embargo_size ┼ Int64: 0
              │                  wd ┼ nothing
              │                  fa ┼ nothing
              │   store_weight_path ┼ Bool: false
              │              strict ┴ Bool: false
            r ┼ ConditionalValueatRisk
              │   settings ┼ RiskMeasureSettings
              │            │   scale ┼ Float64: 1.0
              │            │      ub ┼ nothing
              │            │     rke ┴ Bool: true
              │      alpha ┼ Float64: 0.05
              │          w ┴ nothing
       scorer ┼ HighestMeanScore()
           ex ┼ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
  train_score ┼ Bool: false
       kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`MultiGSCVValType_VecMultiGSCVValType`](@ref)
  - [`AbstractBaseRiskMeasure`](@ref)
  - [`CrossValSearchScorer`](@ref)
  - [`search_cross_validation`](@ref)

# References

  - $(ref_dict[:bergstra2012]) Section 2.
"""
@concrete struct GridSearchCrossValidation <: AbstractSearchCrossValidationEstimator
    """
    $(field_dict[:p_cv])
    """
    p
    """
    $(field_dict[:cv])
    """
    cv
    """
    $(field_dict[:r])
    """
    r
    """
    $(field_dict[:scorer])
    """
    scorer
    """
    $(field_dict[:ex])
    """
    ex
    """
    $(field_dict[:train_score])
    """
    train_score
    """
    $(field_dict[:kwargs])
    """
    kwargs
    function GridSearchCrossValidation(p::Union{<:AbstractVector{<:Pair{<:Any,
                                                                        <:AbstractVector}},
                                                <:AbstractVector{<:AbstractVector{<:Pair{<:Any,
                                                                                         <:AbstractVector}}},
                                                <:AbstractDict{<:Any, <:AbstractVector},
                                                <:AbstractVector{<:AbstractDict{<:Any,
                                                                                <:AbstractVector}}},
                                       cv::CrossValidationEstimator,
                                       r::AbstractBaseRiskMeasure,
                                       scorer::CrossValSearchScorer,
                                       ex::FLoops.Transducers.Executor, train_score::Bool,
                                       kwargs::NamedTuple)
        @argcheck(!isempty(p), IsEmptyError)
        p_flag = isa(p, AbstractVector{<:Pair})
        d_flag = isa(p, AbstractDict)
        vp_flag = isa(p, AbstractVector{<:AbstractVector{<:Pair}})
        vd_flag = isa(p, AbstractVector{<:AbstractDict})
        if p_flag
            @argcheck(all(x -> isa(x[1], GSCVKey), p),
                      ArgumentError("all keys in p must be of type GSCVKey (String, Symbol, or Integer)"))
        elseif d_flag
            @argcheck(all(x -> isa(x, GSCVKey), keys(p)),
                      ArgumentError("all keys in p must be of type GSCVKey (String, Symbol, or Integer)"))
        elseif vp_flag || vd_flag
            @argcheck(all(!isempty, p),
                      IsEmptyError("each parameter set in p cannot be empty"))
            if vp_flag
                for _p in p
                    @argcheck(all(x -> isa(x[1], GSCVKey), _p),
                              ArgumentError("all keys in p must be of type GSCVKey (String, Symbol, or Integer)"))
                end
            end
            if vd_flag
                for _p in p
                    @argcheck(all(x -> isa(x, GSCVKey), keys(_p)),
                              ArgumentError("all keys in p must be of type GSCVKey (String, Symbol, or Integer)"))
                end
            end
        end
        return new{typeof(p), typeof(cv), typeof(r), typeof(scorer), typeof(ex),
                   typeof(train_score), typeof(kwargs)}(p, cv, r, scorer, ex, train_score,
                                                        kwargs)
    end
end
function GridSearchCrossValidation(p::Union{<:AbstractVector{<:Pair{<:Any,
                                                                    <:AbstractVector}},
                                            <:AbstractVector{<:AbstractVector{<:Pair{<:Any,
                                                                                     <:AbstractVector}}},
                                            <:AbstractDict{<:Any, <:AbstractVector},
                                            <:AbstractVector{<:AbstractDict{<:Any,
                                                                            <:AbstractVector}}};
                                   cv::CrossValidationEstimator = KFold(),
                                   r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
                                   scorer::CrossValSearchScorer = HighestMeanScore(),
                                   ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                   train_score::Bool = false, kwargs::NamedTuple = (;))
    return GridSearchCrossValidation(p, cv, r, scorer, ex, train_score, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Randomised search cross-validation estimator for portfolio optimisation. Samples parameter sets from distributions or vectors, applies cross-validation splits, fits and scores each configuration, and selects the optimal parameters using the provided scoring strategy.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RandomisedSearchCrossValidation(
        p::Union{AbstractVector{<:Pair{<:GSCVKey, <:RSCVVal}},
                 AbstractVector{<:AbstractVector{<:Pair{<:GSCVKey,
                                                        <:RSCVVal}}},
                 AbstractDict{<:GSCVKey, <:RSCVVal},
                 AbstractVector{<:AbstractDict{<:GSCVKey,
                                               <:RSCVVal}}};
        cv::CrossValidationEstimator = KFold(),
        r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
        scorer::CrossValSearchScorer = HighestMeanScore(),
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        n_iter::Integer = 10,
        rng::Random.AbstractRNG = Random.default_rng(),
        seed::Option{<:Integer} = nothing,
        train_score::Bool = false,
        kwargs::NamedTuple = (;),
    ) -> RandomisedSearchCrossValidation

Positional and keyword arguments correspond to the struct's fields.

## Validation

  - `!isempty(p)`.
  - If `p` is a vector of parameter sets: each element must not be empty.
  - All keys in `p` must be of type `GSCVKey` (i.e. `String`, `Symbol`, or `Integer`).
  - All values in `p` must be of type `RSCVVal` (i.e. an `AbstractVector` or `Distributions.Distribution`).
  - `n_iter > 0` and finite.

# Examples

```jldoctest
julia> RandomisedSearchCrossValidation(Dict(\"alpha\" => [0.1, 0.2, 0.3],
                                            \"beta\" => Normal(1.0, 0.5)))
RandomisedSearchCrossValidation
            p ┼ Dict{String, Any}: Dict{String, Any}("alpha" => [0.1, 0.2, 0.3], "beta" => Distributions.Normal{Float64}(μ=1.0, σ=0.5))
           cv ┼ KFold
              │                   n ┼ Int64: 5
              │         purged_size ┼ Int64: 0
              │        embargo_size ┼ Int64: 0
              │                  wd ┼ nothing
              │                  fa ┼ nothing
              │   store_weight_path ┼ Bool: false
              │              strict ┴ Bool: false
            r ┼ ConditionalValueatRisk
              │   settings ┼ RiskMeasureSettings
              │            │   scale ┼ Float64: 1.0
              │            │      ub ┼ nothing
              │            │     rke ┴ Bool: true
              │      alpha ┼ Float64: 0.05
              │          w ┴ nothing
       scorer ┼ HighestMeanScore()
           ex ┼ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
       n_iter ┼ Int64: 10
          rng ┼ Random.TaskLocalRNG: Random.TaskLocalRNG()
         seed ┼ nothing
  train_score ┼ Bool: false
       kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractSearchCrossValidationEstimator`](@ref)
  - [`GridSearchCrossValidation`](@ref)
  - [`SearchCrossValidationResult`](@ref)
  - [`CrossValSearchScorer`](@ref)

# References

  - $(ref_dict[:bergstra2012])
"""
@concrete struct RandomisedSearchCrossValidation <: AbstractSearchCrossValidationEstimator
    """
    $(field_dict[:p_cv])
    """
    p
    """
    $(field_dict[:cv])
    """
    cv
    """
    $(field_dict[:r])
    """
    r
    """
    $(field_dict[:scorer])
    """
    scorer
    """
    $(field_dict[:ex])
    """
    ex
    """
    $(field_dict[:n_iter])
    """
    n_iter
    """
    $(field_dict[:rng])
    """
    rng
    """
    $(field_dict[:seed])
    """
    seed
    """
    $(field_dict[:train_score])
    """
    train_score
    """
    $(field_dict[:kwargs])
    """
    kwargs
    function RandomisedSearchCrossValidation(p::Union{<:AbstractVector{<:Pair},
                                                      <:AbstractVector{<:AbstractVector{<:Pair}},
                                                      <:AbstractDict,
                                                      <:AbstractVector{<:AbstractDict}},
                                             cv::CrossValidationEstimator,
                                             r::AbstractBaseRiskMeasure,
                                             scorer::CrossValSearchScorer,
                                             ex::FLoops.Transducers.Executor,
                                             n_iter::Integer, rng::Random.AbstractRNG,
                                             seed::Option{<:Integer}, train_score::Bool,
                                             kwargs::NamedTuple)
        @argcheck(!isempty(p), IsEmptyError)
        p_flag = isa(p, AbstractVector{<:Pair})
        d_flag = isa(p, AbstractDict)
        vp_flag = isa(p, AbstractVector{<:AbstractVector{<:Pair}})
        vd_flag = isa(p, AbstractVector{<:AbstractDict})
        if p_flag
            @argcheck(all(x -> isa(x[1], GSCVKey), p),
                      ArgumentError("all keys in p must be of type GSCVKey (String, Symbol, or Integer)"))
            @argcheck(all(x -> isa(x[2], RSCVVal), p),
                      ArgumentError("all values in p must be of type RSCVVal (AbstractVector or Distribution)"))
        elseif d_flag
            @argcheck(all(x -> isa(x, GSCVKey), keys(p)),
                      ArgumentError("all keys in p must be of type GSCVKey (String, Symbol, or Integer)"))
            @argcheck(all(x -> isa(x, RSCVVal), values(p)),
                      ArgumentError("all values in p must be of type RSCVVal (AbstractVector or Distribution)"))
        elseif vp_flag || vd_flag
            @argcheck(all(!isempty, p),
                      IsEmptyError("each parameter set in p cannot be empty"))
            if vp_flag
                for _p in p
                    @argcheck(all(x -> isa(x[1], GSCVKey), _p),
                              ArgumentError("all keys in p must be of type GSCVKey (String, Symbol, or Integer)"))
                    @argcheck(all(x -> isa(x[2], RSCVVal), _p),
                              ArgumentError("all values in p must be of type RSCVVal (AbstractVector or Distribution)"))
                end
            end
            if vd_flag
                for _p in p
                    @argcheck(all(x -> isa(x, GSCVKey), keys(_p)),
                              ArgumentError("all keys in p must be of type GSCVKey (String, Symbol, or Integer)"))
                    @argcheck(all(x -> isa(x, RSCVVal), values(_p)),
                              ArgumentError("all values in p must be of type RSCVVal (AbstractVector or Distribution)"))
                end
            end
        end
        assert_nonempty_gt0_finite_val(n_iter, :n_iter)
        return new{typeof(p), typeof(cv), typeof(r), typeof(scorer), typeof(ex),
                   typeof(n_iter), typeof(rng), typeof(seed), typeof(train_score),
                   typeof(kwargs)}(p, cv, r, scorer, ex, n_iter, rng, seed, train_score,
                                   kwargs)
    end
end
function RandomisedSearchCrossValidation(p::Union{<:AbstractVector{<:Pair},
                                                  <:AbstractVector{<:AbstractVector{<:Pair}},
                                                  <:AbstractDict,
                                                  <:AbstractVector{<:AbstractDict}};
                                         cv::CrossValidationEstimator = KFold(),
                                         r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
                                         scorer::CrossValSearchScorer = HighestMeanScore(),
                                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                         n_iter::Integer = 10,
                                         rng::Random.AbstractRNG = Random.default_rng(),
                                         seed::Option{<:Integer} = nothing,
                                         train_score::Bool = false,
                                         kwargs::NamedTuple = (;))
    return RandomisedSearchCrossValidation(p, cv, r, scorer, ex, n_iter, rng, seed,
                                           train_score, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

A [`CrossValidationSearchScorer`](@ref) that selects the parameter set with the highest mean score across cross-validation splits.

When called with a score matrix (rows = CV splits, columns = parameter sets), it returns the column index with the largest mean score.

# Examples

```jldoctest
julia> scorer = PortfolioOptimisers.HighestMeanScore();

julia> scores = [0.5 0.8; 0.6 0.7];

julia> scorer(scores)
2
```

# Related

  - [`CrossValidationSearchScorer`](@ref)
  - [`CrossValSearchScorer`](@ref)
  - [`SearchCrossValidationResult`](@ref)
"""
struct HighestMeanScore <: CrossValidationSearchScorer end
function (s::HighestMeanScore)(X::MatNum; dims::Integer = 1)
    return argmax(dropdims(mean(X; dims = dims); dims = dims))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Choose the winning candidate of a search, and never let a failed one win.

A candidate that did not finish every fold carries a non-finite entry in its column of the score matrix, and `argmax` over `[0.80, NaN, 0.50]` returns the position of the `NaN`. So the scorer is handed the columns whose **every** entry is finite, and the index it returns — a position in the matrix it received — is mapped back to the grid through the list of those columns.

The scorer therefore never sees a failed candidate, and it may compute anything on the matrix it receives: a mean, a spread, a rank. A `-Inf` substitution would be safe only for a scorer that reads order alone, because a column holding `-Inf` gives `NaN` for a spread and the `NaN` wins again.

The **raw** matrix stays on the result, so its columns line up with the grid and a reader sees which fold failed.

# Algorithm

 1. Mark the columns of `test_scores` whose every entry is finite.
 2. Throw an `IsNonFiniteError` when no column is finite.
 3. Call `scorer` on the view of `test_scores` at the finite columns.
 4. Return the finite column the index the scorer gave names.

# Arguments

  - `scorer`: The search scorer ([`CrossValSearchScorer`](@ref)).
  - `test_scores`: The `folds × candidates`, or `paths × candidates`, score matrix.

# Validation

  - At least one candidate must have finished every fold.

# Returns

  - `opt_idx::Integer`: The position of the winning candidate in the parameter grid.

# Related

  - [`CrossValidationSearchScorer`](@ref)
  - [`CrossValSearchScorer`](@ref)
  - [`search_cross_validation`](@ref)
  - [`IsNonFiniteError`](@ref)
"""
function finite_candidate_index(scorer::CrossValSearchScorer, test_scores::MatNum)
    finite = vec(all(isfinite, test_scores; dims = 1))
    @argcheck(any(finite),
              IsNonFiniteError("no parameter set finished every fold: every one of the $(size(test_scores, 2)) columns of the score matrix holds a non-finite entry over its $(size(test_scores, 1)) rows. Every candidate failed a fold, so none can be compared; widen the grid, or find why the folds failed."))
    cols = findall(finite)
    return cols[scorer(view(test_scores, :, cols))]
end
"""
$(DocStringExtensions.TYPEDEF)

Result type for search-based cross-validation routines. Stores the optimal estimator, score matrices, parameter grid, and selected index for hyperparameter search.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`AbstractSearchCrossValidationResult`](@ref)
  - [`GridSearchCrossValidation`](@ref)
  - [`RandomisedSearchCrossValidation`](@ref)
  - [`search_cross_validation`](@ref)
"""
@concrete struct SearchCrossValidationResult <: AbstractSearchCrossValidationResult
    """
    $(field_dict[:opt_cv])
    """
    opt
    """
    $(field_dict[:test_scores])
    """
    test_scores
    """
    $(field_dict[:train_scores])
    """
    train_scores
    """
    $(field_dict[:lens_grid])
    """
    lens_grid
    """
    $(field_dict[:val_grid])
    """
    val_grid
    """
    $(field_dict[:idx_cv])
    """
    idx
    function SearchCrossValidationResult(opt::AbstractEstimator, test_scores::MatNum,
                                         train_scores::Option{<:Union{<:MatNum,
                                                                      <:AbstractVector{<:MatNum}}},
                                         lens_grid::AbstractVector,
                                         val_grid::AbstractVector, idx::Integer)
        return new{typeof(opt), typeof(test_scores), typeof(train_scores),
                   typeof(lens_grid), typeof(val_grid), typeof(idx)}(opt, test_scores,
                                                                     train_scores,
                                                                     lens_grid, val_grid,
                                                                     idx)
    end
end
function SearchCrossValidationResult(; opt::AbstractEstimator, test_scores::MatNum,
                                     train_scores::Option{<:Union{<:MatNum,
                                                                  <:AbstractVector{<:MatNum}}},
                                     lens_grid::AbstractVector, val_grid::AbstractVector,
                                     idx::Integer)::SearchCrossValidationResult
    return SearchCrossValidationResult(opt, test_scores, train_scores, lens_grid, val_grid,
                                       idx)
end
"""
    assert_search_entry(est, cv)

Refuse an estimator that is not the configuration alone at the door of a search, once, before any candidate is built.

A search scores every candidate through the one fold loop, and a lens is applied before that loop's warm-up, so a cold estimator seeds one state per candidate and nothing is shared or reset. A warm one is refused by name here rather than inside the candidate loop, where the workers of `gscv.ex` would raise it up to once per candidate. Under every Fold Fit the walk is [`online_entry_state`](@ref), because a search tunes the configuration alone whatever the scheme does with it; under an [`OnlineStep`](@ref) it is the whole [`assert_online_entry`](@ref), so a schedule on a stateful field is refused at the door too.

# Arguments

  - `est`: The estimator the search tunes.
  - `cv`: The scheme the search scores over.

# Validation

  - No `cache` in the tree of `est` holds a state. An `ArgumentError` naming the field is thrown otherwise.
  - Under an [`OnlineStep`](@ref), everything [`assert_online_entry`](@ref) refuses.

# Related

  - [`search_cross_validation`](@ref)
  - [`online_entry_state`](@ref)
  - [`assert_online_entry`](@ref)
  - [`fold_fit`](@ref)
"""
function assert_search_entry(est, cv)
    if !isnothing(fold_fit(cv))
        assert_online_entry(est)
    else
        path = online_entry_state(est)
        @argcheck(isnothing(path),
                  ArgumentError("`$(typeof(est).name.name)` enters `search_cross_validation` carrying a partial-fit state at `$(path)`, and a search tunes the configuration alone: every candidate is built from it through the grid's lenses and scored through the fold loop, which reads its argument as configuration, so the state would reach no fold. Hand the search the estimator with every `cache` at `nothing`."))
    end
    return nothing
end
"""
    pin_draw(cv::MultipleRandomised)
    pin_draw(cv)

Fix the folds a search scores every candidate over.

A search splits its data once and scores every candidate against the same folds, so a scheme whose `split` draws from a random stream must draw the same folds for every candidate. A [`MultipleRandomised`](@ref) with no `seed` draws from its `rng` afresh at every `split`, and the fold loop splits once per candidate, so this draws one seed from that `rng` and returns the scheme with it set: every candidate's `split` then resolves the same stream, and so does the search's own `split` for the row layout. A seeded scheme, and every other scheme, is returned as it is, because its `split` is already a function of the data alone.

# Arguments

  - `cv`: The search's scheme.

# Returns

  - `cv`: The scheme, with a `seed` set when its `split` would otherwise draw afresh.

# Related

  - [`search_cross_validation`](@ref)
  - [`MultipleRandomised`](@ref)
  - [`resolve_rng`](@ref)
"""
function pin_draw(cv::MultipleRandomised)
    return isnothing(cv.seed) ? Accessors.@set(cv.seed = rand(cv.rng, UInt32)) : cv
end
function pin_draw(cv)
    return cv
end
"""
    score_rows(cvr::CrossValidationResult)
    score_rows(cvr::MultipleRandomisedResult)

The rows of a search's score matrix that a candidate's predictions fill, in the order the fold loop returns them.

The score matrix is indexed by `split`: row `j` is the `j`-th fold the scheme enumerates, and a reader lines the rows up with `split`'s enumeration. A contiguous scheme returns one [`MultiPeriodPredictionResult`](@ref) whose predictions are in that order, so its rows are every row in order. A [`MultipleRandomised`](@ref) returns one `MultiPeriodPredictionResult` per path, each sorted by the start of its test window through [`sort_predictions!`](@ref), so its rows are one vector per path: the split rows of that path, in the order the path's predictions come back. [`write_candidate_scores!`](@ref) zips a candidate's predictions against these rows, so the matrix keeps `split`'s order under either shape.

# Arguments

  - `cvr`: The split the search scores over.

# Returns

  - `rows`: The row indices, one range for a contiguous scheme, or one vector of indices per path for a multiple-randomised one.

# Related

  - [`write_candidate_scores!`](@ref)
  - [`search_cross_validation`](@ref)
  - [`sort_predictions!`](@ref)
  - [`MultipleRandomisedResult`](@ref)
"""
function score_rows(cvr::CrossValidationResult)
    return eachindex(cvr.test_idx)
end
function score_rows(cvr::MultipleRandomisedResult)
    return map(unique(cvr.path_ids)) do k
        rows = findall(==(k), cvr.path_ids)
        return rows[sortperm(cvr.test_idx[rows]; by = first)]
    end
end
"""
    write_candidate_scores!(test_scores::MatNum, train_scores::Option{<:MatNum}, i::Integer,
                            predictions::MultiPeriodPredictionResult, rows, r, sgn, kwargs)
    write_candidate_scores!(test_scores::MatNum, train_scores::Option{<:MatNum}, i::Integer,
                            predictions::PopulationPredictionResult, rows, r, sgn, kwargs)

Write the per-fold scores of candidate `i` into column `i` of a search's score matrices.

The candidate's predictions are what [`fit_and_predict`](@ref) returned over the search's scheme, one per fold, and `rows` is [`score_rows`](@ref) of that scheme's split, so prediction and row are zipped and the column reads in `split`'s order. A population is one series per path, and each path is written against its own rows. Each fold's test score is [`expected_risk`](@ref) of the fold's prediction, and its train score, written only when `train_scores` is a matrix, is the risk of the fold's fitted result over its own sample. Both are multiplied by `sgn`, `1` when [`bigger_is_better`](@ref) holds for `r` and `-1` otherwise, so a higher score is always better whatever measure `r` is, which is the orientation a [`CrossValidationSearchScorer`](@ref) relies on. A fold that failed scores `NaN`, and [`finite_candidate_index`](@ref) keeps that column from the scorer.

# Arguments

  - `test_scores`: The `folds × candidates` test score matrix.
  - `train_scores`: The train score matrix of the same shape, or `nothing` when the search records none.
  - `i`: The candidate's column.
  - `predictions`: The candidate's predictions over the scheme.
  - `rows`: The rows the predictions fill, from [`score_rows`](@ref).
  - `r`: The risk measure the search scores with.
  - `sgn`: The sign that orients `r` so that higher is better.
  - `kwargs`: The keyword arguments forwarded to [`expected_risk`](@ref).

# Related

  - [`score_rows`](@ref)
  - [`search_cross_validation`](@ref)
  - [`expected_risk`](@ref)
  - [`bigger_is_better`](@ref)
  - [`finite_candidate_index`](@ref)
"""
function write_candidate_scores!(test_scores::MatNum, train_scores::Option{<:MatNum},
                                 i::Integer, predictions::MultiPeriodPredictionResult, rows,
                                 r, sgn, kwargs)
    for (j, p) in zip(rows, predictions.pred)
        test_scores[j, i] = sgn * expected_risk(r, p; kwargs...)
        if !isnothing(train_scores)
            train_scores[j, i] = sgn * expected_risk(r, p.res; kwargs...)
        end
    end
    return nothing
end
function write_candidate_scores!(test_scores::MatNum, train_scores::Option{<:MatNum},
                                 i::Integer, predictions::PopulationPredictionResult, rows,
                                 r, sgn, kwargs)
    for (path, path_rows) in zip(predictions.pred, rows)
        write_candidate_scores!(test_scores, train_scores, i, path, path_rows, r, sgn,
                                kwargs)
    end
    return nothing
end
"""
    _expr_to_lens(ex::Symbol)

Convert a bare symbol into a `PropertyLens` for field access.

Base case for the lens-building recursion: a bare symbol maps directly to an `Accessors.PropertyLens`.

# Arguments

  - `ex::Symbol`: A field name symbol.

# Returns

  - `Accessors.PropertyLens` for the symbol.

# Related

  - [`expr_to_lens_chain`](@ref)
  - [`parse_lens`](@ref)
"""
_expr_to_lens(ex::Symbol) = Accessors.PropertyLens(ex)
"""
    _eval_index(x)

Evaluate a literal index node in the AST without runtime `eval`.

Converts integer, symbol, or vector expression AST nodes to concrete index values for use in `Accessors.IndexLens`.

# Arguments

  - `x::Integer`: An integer index.
  - `x::Symbol`: A symbolic index.
  - `ex::Expr`: A vector expression (`:vect` head).

# Returns

  - The evaluated index value.

# Related

  - [`expr_to_lens_chain`](@ref)
"""
_eval_index(x::Integer) = x
_eval_index(x::Symbol) = x
_eval_index(ex::Expr)  = ex.head === :vect ? [_eval_index(a) for a in ex.args] : throw(Meta.ParseError("Unsupported index expression: $ex"))
"""
    expr_to_lens_chain(ex)

Convert a Julia expression to a chain of lens accessors.

Internal helper for parsing hyperparameter key strings into composable Accessors.jl lenses.

# Arguments

  - `ex`: Julia expression representing a field access chain.

# Returns

  - Composed lens.

# Related

  - [`parse_lens`](@ref)
  - [`_expr_to_lens`](@ref)
"""
function expr_to_lens_chain(ex)
    optics = Union{Accessors.PropertyLens, Accessors.IndexLens}[]
    while ex isa Expr
        if ex.head === :.
            push!(optics, Accessors.PropertyLens((ex.args[2]::QuoteNode).value))
            ex = ex.args[1]
        elseif ex.head === :ref
            indices = ntuple(i -> _eval_index(ex.args[i + 1]), length(ex.args) - 1)
            push!(optics, Accessors.IndexLens(indices))
            ex = ex.args[1]
        else
            throw(Meta.ParseError("Unsupported expression: $ex"))
        end
    end
    if !(ex isa Symbol)
        throw(Meta.ParseError("Unsupported expression: $ex"))
    end
    push!(optics, Accessors.PropertyLens(ex))  # base case: Symbol
    return foldl(∘, optics)
end
"""
    parse_lens(key::AbstractString)

Parse a hyperparameter key string into an Accessors.jl lens.

Converts a dotted string path (e.g., `"opt.pe.ce"`) into a composable lens for getting and setting nested fields of an estimator object.

# Arguments

  - `key`: Dotted field path string.

# Validation

  - String keys longer than `EQUATION_LIMITS[].max_length` are rejected before `Meta.parse`.
  - Keys deeper than `EQUATION_LIMITS[].max_depth` are rejected before the lens-building walk. A string key is measured after `Meta.parse`, so one depth bound holds for every shape of `key`.

# Returns

  - Composed Accessors.jl lens.

# Related

  - [`expr_to_lens_chain`](@ref)
  - [`GridSearchCrossValidation`](@ref)
  - [`RandomisedSearchCrossValidation`](@ref)
  - [`EQUATION_LIMITS`](@ref)
"""
function parse_lens(key::AbstractString)
    # Trust boundary: cap the untrusted string length before `Meta.parse`, so a deeply
    # nested key cannot exhaust the stack. The length bounds the achievable AST depth at
    # about a third of the character count, which is looser than `max_depth`, so the
    # parsed tree meets the depth cap directly, as the `Expr` form does.
    lim = EQUATION_LIMITS[]
    @argcheck(length(key) <= lim.max_length,
              Meta.ParseError("Lens key string is too long ($(length(key)) > $(lim.max_length) characters)."))
    expr = Meta.parse(key)
    @argcheck(!_expr_depth_exceeds(expr, lim.max_depth),
              Meta.ParseError("Lens key expression is too deeply nested (exceeds depth $(lim.max_depth))."))
    return expr_to_lens_chain(expr)
end
function parse_lens(key::Union{Expr, Symbol})
    # Trust-boundary defence for the pre-built-AST form (no string length cap applies):
    # reject an over-deep tree before `expr_to_lens_chain`'s recursive walk can exhaust the stack.
    lim = EQUATION_LIMITS[]
    @argcheck(!_expr_depth_exceeds(key, lim.max_depth),
              Meta.ParseError("Lens key expression is too deeply nested (exceeds depth $(lim.max_depth))."))
    return expr_to_lens_chain(key)
end
function parse_lens(key::Union{<:ComposedFunction, <:Accessors.PropertyLens,
                               <:Accessors.IndexLens})
    return key
end
"""
    assert_search_grid_cap(total::Integer, detail::AbstractString)
    assert_search_grid_cap(factors::AbstractVector{<:Pair})
    assert_search_grid_cap(estval::AbstractVector{<:Pair{<:Any, <:AbstractVector}})
    assert_search_grid_cap(estval::AbstractDict{<:Any, <:AbstractVector})

Assert the **total** number of search-grid candidates does not exceed the active `max_search_grid` ceiling.

The grid is an `Iterators.product` materialised by `collect`, so `k` tuned parameters of `N` values each cost `N^k` candidates and `N^k` full cross-validated fits. A per-parameter check never sees that product — the same reasoning that gave the frontier sweep [`assert_frontier_sweep_cap`](@ref) — so the cap is asserted where the grid is formed, before the `collect`.

Concatenated parameter sets are a **sum** of products, not a product: each set is capped as it is built, and the concatenated total is capped again on the way out.

# Arguments

  - `total`: Candidate count already computed (a `BigInt` on the product path, since `k` parameters at the ceiling overflow an `Int64` before the check reads it).
  - `detail`: Phrase naming how `total` was made, interpolated into the message.
  - `factors`: `key => value-count` pairs, one per tuned parameter.
  - `estval`: The parameter grid itself, from which the factors are derived.

# Returns

  - `nothing`.

# Throws

  - `DomainError` if the count exceeds `RESOURCE_LIMITS[].max_search_grid`. The message names the count, the factors that made it, and the knob that raises the ceiling.

# Related

  - [`lens_val_grid`](@ref)
  - [`pipeline_lens_val_grid`](@ref)
  - [`assert_frontier_sweep_cap`](@ref)
  - [`RESOURCE_LIMITS`](@ref)
"""
function assert_search_grid_cap(total::Integer, detail::AbstractString)::Nothing
    cap = RESOURCE_LIMITS[].max_search_grid
    @argcheck(total <= cap,
              DomainError(total,
                          "the search grid is $total candidates — $detail — and exceeds RESOURCE_LIMITS[].max_search_grid = $cap. Every candidate runs a full cross-validated fit, so the ceiling is on the whole grid, not on any single parameter's value count. Tune fewer parameters, or shorten a value vector. Raise the ceiling with set_resource_limits!(; max_search_grid) — or with_resource_limits for a single scope, or the \"max_search_grid\" preference for a whole project — for genuinely large machine-authored runs."))
    return nothing
end
function assert_search_grid_cap(factors::AbstractVector{<:Pair})::Nothing
    if isempty(factors)
        return nothing
    end
    return assert_search_grid_cap(prod(big(n) for (_, n) in factors; init = big(1)),
                                  "the product " *
                                  join(("$key = $n" for (key, n) in factors), " × ") *
                                  " across every tuned parameter")
end
function assert_search_grid_cap(estval::AbstractVector{<:Pair{<:Any, <:AbstractVector}})::Nothing
    return assert_search_grid_cap([x[1] => length(x[2]) for x in estval])
end
function assert_search_grid_cap(estval::AbstractDict{<:Any, <:AbstractVector})::Nothing
    return assert_search_grid_cap([key => length(val) for (key, val) in estval])
end

export SearchCrossValidationResult, HighestMeanScore
