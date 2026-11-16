"""
$(DocStringExtensions.TYPEDEF)

Defines the interface for scoring strategies used in search cross-validation. Implementations select the optimal parameter set based on cross-validation scores.

It is the branch of [`AbstractCrossValidationScorer`](@ref) that selects a candidate of a search. Its sibling [`PredictionScorer`](@ref) selects a path of a cross-validation.

# Interfaces

  - [`candidate_scores`](@ref)`(::CrossValidationSearchScorer, scores::AbstractMatrix)`: Returns one value per column of `scores`, bigger is better. The search orders the candidates by this value, so it can keep the `n_best` best ones, and it can combine the values of a vector of scorers through `s_sca`. A subtype that defines no method gives `nothing`, and the search then calls the index form below and keeps one candidate.
  - `(::CrossValidationSearchScorer)(scores::AbstractMatrix)`: Returns the index of the optimal parameter set, the column with the best (highest) aggregate score. It must equal `argmax` of `candidate_scores` when the subtype defines both.
  - [`scorer_scale`](@ref)`(::CrossValidationSearchScorer)`: Optional. Returns the weight of the scorer in a vector of scorers. The fallback gives `1`, so a subtype with a `scale` field defines this method to read it.

## Arguments

  - `scores`: Matrix of scores, where each column corresponds to a parameter set and each row to a cross-validation split. The matrix is orientation-normalised: risk-measure scores are negated so that **higher is always better**, whatever measure `r` is. A scorer therefore returns the index of the column with the best (highest) aggregate score.

## Returns

  - `candidate_scores`: `VecNum`, one value per column of the matrix the scorer received, in the order of the columns, and no value `NaN`. The search refuses a vector of another length with a `DimensionMismatch`, and a `NaN` with an [`IsNonFiniteError`](@ref).
  - The index form: `Int`, the index of the optimal parameter set, **as a position in the matrix the scorer received**.

## A scorer never sees a failed candidate

The search, through `finite_candidate_order`, hands the scorer the columns whose every entry is finite, and maps the order back to the parameter grid itself. A scorer therefore reads a matrix of finite numbers alone, and it may compute anything on it, such as a mean, a spread or a rank, and a candidate that failed a fold cannot win. The matrix a scorer receives is a *view* of the score matrix, and its column count can be smaller than the grid. So a scorer must return values or a position in that view, and must not index the grid itself.

# Examples

```jldoctest
julia> struct MyScore <: PortfolioOptimisers.CrossValidationSearchScorer end

julia> function PortfolioOptimisers.candidate_scores(::MyScore, X::AbstractMatrix)
           return vec(minimum(X; dims = 1))
       end

julia> (s::MyScore)(X::AbstractMatrix) = argmax(PortfolioOptimisers.candidate_scores(s, X))

julia> scores = [0.5 0.6; 0.7 0.8];

julia> scorer = MyScore()
MyScore()

julia> PortfolioOptimisers.candidate_scores(scorer, scores)
2-element Vector{Float64}:
 0.5
 0.6

julia> scorer(scores)
2
```

# Related

  - [`AbstractCrossValidationScorer`](@ref)
  - [`CrossValSearchScorer`](@ref)
  - [`HighestMeanScore`](@ref)
  - [`candidate_scores`](@ref)
  - [`AbstractSearchCrossValidationEstimator`](@ref)
"""
abstract type CrossValidationSearchScorer <: AbstractCrossValidationScorer end
"""
    const CrossValSearchScorer = Union{CrossValidationSearchScorer, Function}

Union type for search cross-validation scoring strategies. Accepts either a subtype of `CrossValidationSearchScorer` or a plain function that gives what a search scorer gives.

A function takes the score matrix and gives one of two forms. A vector holds one value per column, bigger is better, as [`candidate_scores`](@ref) does, so `X -> vec(mean(X; dims = 1))` does the job of [`HighestMeanScore`](@ref). An integer is the index of one winner, as the index form of a subtype is. The search calls the function once and reads the form from the type of the return.

One index cannot order more than one candidate, and it has no value to combine. So a search refuses an index with `n_best > 1`, and a vector of scorers refuses a member that gives an index. Both refusals happen when the search runs, because only the return shows the form. A function has the scale `1`, see [`scorer_scale`](@ref). Write a weight into the function itself.

# Related

  - [`CrossValidationSearchScorer`](@ref)
  - [`candidate_values`](@ref)
"""
const CrossValSearchScorer = Union{CrossValidationSearchScorer, Function}
"""
    candidate_scores(scorer::CrossValSearchScorer, X)
    candidate_scores(scorer::HighestMeanScore, X::MatNum)

Give one value per candidate of a search, bigger is better, so that the search can order the candidates.

The column `j` of `X` holds the scores of candidate `j` over the rows of the scheme, signed so that a bigger score is better. The search sorts the candidates by these values, in descending order, and keeps the first `n_best`. The sort is stable, so a tie keeps the order of the grid.

The method on [`CrossValSearchScorer`](@ref) is the fallback. It gives `nothing`, which states that the scorer gives no value through this method. A function scorer, and a subtype of [`CrossValidationSearchScorer`](@ref) that defines no method, give `nothing`. The search then calls the scorer once, through [`candidate_values`](@ref), and reads the form from the return. A function can give a vector of values or an index, and the index form of a subtype gives an index. The search keeps one candidate for an index, and it refuses `n_best > 1` with an `ArgumentError`.

# Arguments

  - `scorer`: The search scorer.
  - `X`: The `rows × candidates` score matrix that the scorer receives. Every entry is finite.

# Returns

  - `vals::Option{<:VecNum}`: One value per column of `X`, or `nothing` when the scorer gives no value per candidate.

# Examples

```jldoctest
julia> PortfolioOptimisers.candidate_scores(HighestMeanScore(), [0.5 0.75 0.25; 1.0 0.5 0.25])
3-element Vector{Float64}:
 0.75
 0.625
 0.25
```

# Related

  - [`CrossValidationSearchScorer`](@ref)
  - [`HighestMeanScore`](@ref)
  - [`finite_candidate_order`](@ref)
  - [`SearchCrossValidationResult`](@ref)
"""
function candidate_scores(::CrossValSearchScorer, ::Any)
    # The second argument is untyped, so a subtype's own method on any matrix type is more
    # specific in both arguments and never ambiguous with this fallback.
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

A [`CrossValidationSearchScorer`](@ref) that selects the parameter set with the highest mean score across cross-validation splits.

When called with a score matrix (rows = CV splits, columns = parameter sets), it returns the column index with the largest mean score. Its [`candidate_scores`](@ref) are the column means, so a search orders its candidates by the mean score. A tie goes to the first column, in the index form and in the order.

The scale is the weight of the scorer in a vector of search scorers, the same word and mechanism as `settings.scale` of a risk measure. A negative scale is a negative weight. The scale is inert for one scorer, so neither form above reads it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HighestMeanScore(; scale::Number = 1.0) -> HighestMeanScore

Keyword arguments correspond to the fields above.

## Validation

  - `scale` is finite. Otherwise an [`IsNonFiniteError`](@ref).

# Examples

```jldoctest
julia> scorer = PortfolioOptimisers.HighestMeanScore();

julia> scores = [0.5 0.8; 0.6 0.7];

julia> scorer(scores)
2

julia> PortfolioOptimisers.candidate_scores(scorer, scores)
2-element Vector{Float64}:
 0.55
 0.75
```

# Related

  - [`CrossValidationSearchScorer`](@ref)
  - [`CrossValSearchScorer`](@ref)
  - [`candidate_scores`](@ref)
  - [`scorer_scale`](@ref)
  - [`SearchCrossValidationResult`](@ref)
"""
@concrete struct HighestMeanScore <: CrossValidationSearchScorer
    """
    $(field_dict[:scale_scorer])
    """
    scale
    function HighestMeanScore(scale::Number)::HighestMeanScore
        @argcheck(isfinite(scale), IsNonFiniteError("scale must be finite, got $scale"))
        return new{typeof(scale)}(scale)
    end
end
function HighestMeanScore(; scale::Number = 1.0)::HighestMeanScore
    return HighestMeanScore(scale)
end
function (s::HighestMeanScore)(X::MatNum; dims::Integer = 1)
    return argmax(dropdims(mean(X; dims = dims); dims = dims))
end
function candidate_scores(::HighestMeanScore, X::MatNum)
    return vec(mean(X; dims = 1))
end
"""
    scorer_scale(scorer::CrossValSearchScorer)
    scorer_scale(scorer::HighestMeanScore)

Give the weight of a search scorer in a vector of search scorers.

[`candidate_values`](@ref) multiplies the values of each member of a vector by this weight before `s_sca` reduces them. It is the same word and mechanism as `settings.scale` of a risk measure, which [`oriented_member_score`](@ref) reads. A negative scale is a negative weight. The scale is inert for one scorer.

The method on [`CrossValSearchScorer`](@ref) is the fallback, and it gives `1`. So a function, and a subtype of [`CrossValidationSearchScorer`](@ref) with no `scale` field, have the weight one. A subtype with a `scale` field defines a method that reads it. A function writes its weight into its own values.

# Arguments

  - `scorer`: The search scorer.

# Returns

  - `scale::Number`: The weight of the scorer.

# Examples

```jldoctest
julia> PortfolioOptimisers.scorer_scale(HighestMeanScore(; scale = 2.0))
2.0

julia> PortfolioOptimisers.scorer_scale(X -> vec(sum(X; dims = 1)))
1
```

# Related

  - [`HighestMeanScore`](@ref)
  - [`candidate_values`](@ref)
  - [`CrossValidationSearchScorer`](@ref)
"""
function scorer_scale(::CrossValSearchScorer)
    return 1
end
function scorer_scale(s::HighestMeanScore)
    return s.scale
end
"""
    finite_candidate_order(r_sca, s_sca::Union{Option{<:Sca_VecToScaM}, EachValue}, scorer,
                           test_scores::MatNum, n_best::Integer)
    finite_candidate_order(r_sca, ::EachValue, scorers::AbstractVector, test_scores::MatNum,
                           n_best::Integer)
    finite_candidate_order(::EachValue, s_sca, scorer, test_scores::AbstractVector{<:MatNum},
                           n_best::Integer)
    finite_candidate_order(r_sca::PerCandidate,
                           s_sca::Union{Option{<:Sca_VecToScaM}, EachValue}, scorer,
                           test_scores::AbstractVector{<:MatNum}, n_best::Integer)
    finite_candidate_order(r_sca::PerCandidate, ::EachValue, scorers::AbstractVector,
                           test_scores::AbstractVector{<:MatNum}, n_best::Integer)

Order the candidates of a search, best first, keep the first `n_best`, and never let a failed candidate in. Under [`EachValue`](@ref), give one such ranking for each value per candidate. Under [`PerCandidate`](@ref), combine the values of the measures of each candidate first.

A candidate that did not finish every fold carries a non-finite entry in its column of the score matrix, and `argmax` over `[0.80, NaN, 0.50]` returns the position of the `NaN`. So the scorer receives the columns whose **every** entry is finite, and the order that it gives, in positions of the matrix that it received, maps back to the grid through the list of those columns.

The scorer therefore never sees a failed candidate, and it can compute anything on the matrix that it receives: a mean, a spread, a rank. A `-Inf` substitution would be safe only for a scorer that reads order alone, because a column that holds `-Inf` gives `NaN` for a spread and the `NaN` wins again.

The **raw** matrix stays on the result, so its columns agree with the grid and a reader sees which fold failed.

# Mathematical definition

```math
\\begin{align}
\\mathcal{C} &= \\left\\{ i : S_{fi} \\text{ is finite for every } f \\right\\}\\,, \\\\
\\pi &= \\operatorname{sortperm}_{\\downarrow}\\, v\\left(\\mathbf{S}_{:,\\,\\mathcal{C}}\\right)\\,, \\\\
r_k &= c_{\\pi_k}\\,, \\quad k = 1, \\ldots, \\min\\left(n_{b}, |\\mathcal{C}|\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:C_finite_cand])
  - $(math_dict[:S_C_search])
  - $(math_dict[:v_cand_scores])
  - $(math_dict[:pi_cand_order])
  - $(math_dict[:c_j_search])
  - $(math_dict[:r_k_rank])
  - $(math_dict[:n_b_best])

The sort is stable, so a tie keeps the order of the grid, as `argmax` does.

## One ranking for each value

A ranking is the pair `(rank_idx, rank_scores)`. The types of `r_sca`, `s_sca`, `scorer` and `test_scores` select the shape, never the count at run time. `r_sca` is read only when `test_scores` holds one matrix per measure, so for one matrix it is inert.

  - **One matrix, one value.** A reduction or one scorer gives one ranking, by the steps below. This is the path of a search of one value, and [`EachValue`](@ref) with one scorer takes it too.
  - **A vector of `K` matrices, `r_sca = PerCandidate(sca)`.** A candidate enters only when its column is finite in every matrix. The scorer reduces each matrix over these columns, and `sca` combines the `K` values of each candidate, through [`candidate_values`](@ref). The rest is the path of one matrix, so the return is one ranking, or `L` rankings under `s_sca = EachValue()`.
  - **One matrix, `s_sca = EachValue()` with `L` scorers.** Each scorer orders the finite candidates alone, through [`scorer_ranking`](@ref). Its values are multiplied by its [`scorer_scale`](@ref), so a negative scale reverses the order of that scorer alone. A scorer that gives the index of one winner keeps one candidate, whatever `n_best` is, and its entry of `rank_scores` is `nothing`. The return is a vector of `L` entries in each part.
  - **A vector of `K` matrices, `r_sca = EachValue()`.** Each matrix is ranked by the rules above, apart from the others, so a candidate can enter the ranking of one measure and stay out of the ranking of another. A matrix with no finite column gives an empty ranking, through [`empty_ranking`](@ref). Its columns are empty, and its values are empty, or `nothing` where a full ranking of the same scorer holds `nothing`. The return is a vector of `K` entries in each part, or a `K × L` matrix, row `k` for measure `k`, under `s_sca = EachValue()` with `L` scorers, through [`measure_rankings`](@ref).

# Algorithm

 1. Mark the columns of `test_scores` whose every entry is finite, giving `cols`, through [`finite_score_columns`](@ref). Under [`PerCandidate`](@ref), a column must be finite in every matrix.
 2. Throw an `IsNonFiniteError` when no column is finite.
 3. Take the view `X` of `test_scores` at `cols`, or the view of each matrix under [`PerCandidate`](@ref), and call [`candidate_values`](@ref) with `r_sca`, `s_sca` and `scorer` on it, giving `vals`.
 4. If `vals` is a vector: check it through [`assert_candidate_scores`](@ref). Then take the first `min(n_best, length(cols))` positions of the descending order of `vals` through a partial sort, which breaks a tie by the position, and so gives the order of a stable sort. Return the columns of `cols` at those positions and their values.
 5. If `vals` is an integer, the index of one winner: refuse `n_best > 1`, and return its column of `cols` and `nothing`.
 6. Refuse a `vals` of any other type with an `ArgumentError`.

Steps 1 to 3 are written twice, once for one matrix and once for the `K` matrices of [`PerCandidate`](@ref), and each copy has a method for one scorer and a method for `L` scorers under `s_sca = EachValue()`. So each form of the data is a method of this function, and no helper holds a part of the job. The copies agree: each marks the columns through [`finite_score_columns`](@ref), takes the views at them, and reads the values through [`candidate_values`](@ref). The tests of a search of one value hold the copies for one matrix, and the tests of [`PerCandidate`](@ref) hold the copies for the `K` matrices.

# Arguments

  - $(arg_dict[:r_sca])
  - `s_sca`: The reduction of a vector of search scorers, or [`EachValue`](@ref). Inert for one scorer.
  - `scorer`, `scorers`: One search scorer ([`CrossValSearchScorer`](@ref)), or a vector of them.
  - `test_scores`: The `folds × candidates`, or `paths × candidates`, score matrix, or one such matrix per measure under a rule of [`AbstractSearchReduction`](@ref).
  - `n_best`: The count of the candidates that each ranking keeps.

# Validation

  - At least one candidate must have finished every fold. Otherwise an `IsNonFiniteError`.
  - A scorer that gives an index keeps one candidate. Otherwise an `ArgumentError` names the type of the scorer and the form to give.
  - The checks of [`assert_candidate_scores`](@ref) and [`candidate_values`](@ref).
  - A scorer gives a vector or an integer. Otherwise an `ArgumentError`.
  - A vector of matrices under [`EachValue`](@ref): these checks on each matrix that has a finite column, and at least one matrix has a finite column. Otherwise an `IsNonFiniteError` that names the count of the measures.
  - A vector of matrices under [`PerCandidate`](@ref): at least one column is finite in every matrix. Otherwise an `IsNonFiniteError`. A scorer that gives an index is refused with an `ArgumentError`.

# Returns

  - `rank_idx::AbstractVector{<:Integer}`: The positions in the parameter grid of the kept candidates, best first. Under [`EachValue`](@ref), one such vector for each value.
  - `rank_scores::Option{<:VecNum}`: The value of [`candidate_values`](@ref) of each kept candidate, in the order of `rank_idx`, or `nothing` when the scorer gives an index. Under [`EachValue`](@ref), one such entry for each value.

# Related

  - [`candidate_values`](@ref)
  - [`candidate_scores`](@ref)
  - [`CrossValidationSearchScorer`](@ref)
  - [`CrossValSearchScorer`](@ref)
  - [`search_cross_validation`](@ref)
  - [`IsNonFiniteError`](@ref)
  - [`EachValue`](@ref)
  - [`PerCandidate`](@ref)
  - [`finite_score_columns`](@ref)
"""
function finite_candidate_order(r_sca, s_sca::Union{Option{<:Sca_VecToScaM}, EachValue},
                                scorer, test_scores::MatNum, n_best::Integer)
    cols = finite_score_columns(test_scores)
    X = view(test_scores, :, cols)
    return finite_candidate_order(scorer, cols, n_best,
                                  candidate_values(r_sca, s_sca, scorer, X))
end
function finite_candidate_order(r_sca, ::EachValue, scorers::AbstractVector,
                                test_scores::MatNum, n_best::Integer)
    cols = finite_score_columns(test_scores)
    X = view(test_scores, :, cols)
    rankings = [scorer_ranking(s, cols, n_best, candidate_values(r_sca, EachValue(), s, X))
                for s in scorers]
    return [r[1] for r in rankings], [r[2] for r in rankings]
end
function finite_candidate_order(::EachValue, s_sca, scorer,
                                test_scores::AbstractVector{<:MatNum}, n_best::Integer)
    full = [any(c -> all(isfinite, c), eachcol(S)) for S in test_scores]
    @argcheck(any(full),
              IsNonFiniteError("no parameter set finished every fold under any of the $(length(test_scores)) risk measures of `r`: every column of each score matrix holds a non-finite entry. Every candidate failed a fold, so none can be compared; widen the grid, or find why the folds failed."))
    found = [finite_candidate_order(EachValue(), s_sca, scorer, S, n_best)
             for (S, f) in zip(test_scores, full) if f]
    blank = empty_ranking(first(found)...)
    rankings = map(cumsum(full), full) do n, f
        return f ? found[n] : blank
    end
    return measure_rankings(s_sca, scorer, rankings)
end
function finite_candidate_order(r_sca::PerCandidate,
                                s_sca::Union{Option{<:Sca_VecToScaM}, EachValue}, scorer,
                                test_scores::AbstractVector{<:MatNum}, n_best::Integer)
    cols = finite_score_columns(test_scores)
    Xs = [view(S, :, cols) for S in test_scores]
    return finite_candidate_order(scorer, cols, n_best,
                                  candidate_values(r_sca, s_sca, scorer, Xs))
end
function finite_candidate_order(r_sca::PerCandidate, ::EachValue, scorers::AbstractVector,
                                test_scores::AbstractVector{<:MatNum}, n_best::Integer)
    cols = finite_score_columns(test_scores)
    Xs = [view(S, :, cols) for S in test_scores]
    rankings = [scorer_ranking(s, cols, n_best,
                               candidate_values(r_sca, EachValue(), s, Xs))
                for s in scorers]
    return [r[1] for r in rankings], [r[2] for r in rankings]
end
function finite_candidate_order(scorer, cols::AbstractVector{<:Integer}, n_best::Integer,
                                vals::VecNum)
    assert_candidate_scores(scorer, vals, length(cols))
    # A partial sort is not stable, so the order breaks a tie by the position in the grid,
    # which gives the order of a stable descending sort.
    keep = partialsort!(collect(eachindex(vals)), 1:min(n_best, length(cols));
                        lt = (i, j) -> vals[i] > vals[j] || (vals[i] == vals[j] && i < j))
    return cols[keep], vals[keep]
end
function finite_candidate_order(scorer, cols::AbstractVector{<:Integer}, n_best::Integer,
                                idx::Integer)
    @argcheck(isone(n_best),
              ArgumentError("`$(typeof(scorer))` gives the index of one winner and no value for each candidate, so a search can pick one candidate with it but cannot order the n_best = $n_best best ones. Make it give one value per column of the score matrix, bigger is better: a function returns the vector, and a subtype adds a method `PortfolioOptimisers.candidate_scores(::$(typeof(scorer)), X::AbstractMatrix)`. Or set `n_best = 1`."))
    return [cols[idx]], nothing
end
function finite_candidate_order(scorer, ::AbstractVector{<:Integer}, ::Integer, vals)
    return throw(ArgumentError("`$(typeof(scorer))` must give one value per column of the score matrix, a vector, or the index of one winner, an integer. Got a `$(typeof(vals))`."))
end
"""
    assert_candidate_scores(scorer, vals::VecNum, n::Integer)

Refuse the values of a search scorer that cannot order the candidates.

A value per candidate must line up with the columns of the matrix that the scorer received, and `NaN` has no place in an order. A short vector would pick a wrong candidate with no error, and a `NaN` would give an order that depends on the sort. [`finite_candidate_order`](@ref) calls this check on the values of one scorer, and [`candidate_values`](@ref) on the values of each member of a vector, before the reduction.

# Arguments

  - `scorer`: The search scorer that gave `vals`.
  - `vals`: Its values.
  - `n`: The count of the columns that the scorer received.

# Validation

  - `length(vals) == n`. Otherwise a `DimensionMismatch`.
  - No value is `NaN`. Otherwise an [`IsNonFiniteError`](@ref). An infinite value has a place in the order, so it is kept.

# Returns

  - `nothing`.

# Related

  - [`finite_candidate_order`](@ref)
  - [`candidate_values`](@ref)
  - [`candidate_scores`](@ref)
"""
function assert_candidate_scores(scorer, vals::VecNum, n::Integer)::Nothing
    @argcheck(length(vals) == n,
              DimensionMismatch("`$(typeof(scorer))` must give one value per column of the matrix that it receives: got $(length(vals)) values for $n columns."))
    @argcheck(!any(isnan, vals),
              IsNonFiniteError("`$(typeof(scorer))` gave NaN for $(count(isnan, vals)) of the $n candidates that finished every fold, first at position $(findfirst(isnan, vals)). A NaN value cannot be ordered."))
    return nothing
end
"""
    candidate_values(r_sca, s_sca::Union{Option{<:Sca_VecToScaM}, EachValue},
                     scorer::CrossValSearchScorer, X::MatNum)
    candidate_values(r_sca::PerCandidate{<:Option{<:Scalariser}},
                     s_sca::Union{Option{<:Sca_VecToScaM}, EachValue},
                     scorer::CrossValSearchScorer, Xs::AbstractVector{<:MatNum})
    candidate_values(r_sca::PerCandidate{<:Union{VectorToScalarMeasure, Function}},
                     s_sca::Union{Option{<:Sca_VecToScaM}, EachValue},
                     scorer::CrossValSearchScorer, Xs::AbstractVector{<:MatNum})
    candidate_values(r_sca, s_sca::Option{<:Sca_VecToScaM}, scorers::AbstractVector, X)

Give what the search scorer, or the vector of them, gives for the candidates of a search: one value per column of `X`, bigger is better, or the index of one winner.

One scorer gives its [`candidate_scores`](@ref). When these are `nothing`, it is called once on `X`, and the return gives the form: a vector of values from a function, or the index of one winner. `r_sca`, `s_sca` and [`scorer_scale`](@ref) are inert, so a search of one scorer makes the one call of the scorer that it made before.

Under [`PerCandidate`](@ref), `Xs` holds the view of each of the `K` score matrices at the same columns. The scorer gives its values on each view, as one scorer does, and `r_sca.sca` combines the `K` values of each candidate to one, by its type:

  - A [`Scalariser`](@ref) or `nothing` combines through [`scalarise`](@ref), elementwise over the vectors of the measures. `nothing` gives the sum of [`SumScalariser`](@ref).
  - A [`VectorToScalarMeasure`](@ref) or a function combines the vector of the `K` values of each candidate through [`vec_to_real_measure`](@ref).

These are the two reductions of the scorers below, written once for the measures and once for the scorers, as the rule *one owner, one name* states. The two copies agree: each reduces by the type of its reduction in the same two ways. The tests of [`PerCandidate`](@ref) hold the copy for the measures, and the tests of a vector of scorers hold the copy for the scorers. The scale of each measure is in its matrix, so it is the weight of the measure. The scale of a scorer is read after this combination, so a vector of scorers weights the combined values.

A vector of scorers gives one value per candidate. Each member gives its values as one scorer does, and the values of each member are multiplied by its scale. Then `s_sca` reduces the `L` values of each candidate to one, by its type:

  - A [`Scalariser`](@ref) or `nothing` reduces through [`scalarise`](@ref), elementwise over the vectors of the members. `nothing` gives the sum of [`SumScalariser`](@ref).
  - A [`VectorToScalarMeasure`](@ref) or a function reduces the vector of the `L` values of each candidate through [`vec_to_real_measure`](@ref).

Every value is bigger-is-better before the reduction, so [`MaxScalariser`](@ref) and [`MaxValue`](@ref) give the best scorer of each candidate, and [`MinScalariser`](@ref) and [`MinValue`](@ref) the worst. [`StdValue`](@ref) gives the highest value to the largest spread.

# Mathematical definition

```math
\\begin{align}
v_j &= S\\left(s_1 v_{1j}, \\, \\ldots, \\, s_L v_{Lj}\\right)\\,.
\\end{align}
```

Where:

  - ``v_j``: Value of candidate ``j`` that the search orders by.
  - ``v_{lj}``: Value of candidate ``j`` from scorer ``l`` of the vector.
  - ``s_l``: Scale of scorer ``l``, from [`scorer_scale`](@ref).
  - ``S``: The reduction `s_sca`.
  - ``L``: Count of the scorers of the vector.

# Algorithm

 1. One scorer: call [`candidate_scores`](@ref). Return its values, or the return of `scorer(X)` when it gives `nothing`.
 2. One scorer under [`PerCandidate`](@ref): give the return of the scorer on each view as in step 1, and check each one through [`measure_candidate_values`](@ref). Combine the `K` vectors through the type of `r_sca.sca`, and check that no combined value is `NaN`.
 3. A vector: give the return of each member as in step 1, or step 2 under [`PerCandidate`](@ref). Check each one through [`assert_candidate_scores`](@ref), and multiply it by the scale of its member.
 4. Reduce the scaled vectors through the type of `s_sca`, with [`reduce_scorer_values`](@ref), and check that no reduced value is `NaN`.

# Arguments

  - $(arg_dict[:r_sca])
  - `s_sca`: The reduction of a vector of scorers.
  - `scorer`: One search scorer.
  - `scorers`: A vector of search scorers.
  - `X`: The `rows × candidates` score matrix that the scorers receive, or the view of each matrix under [`PerCandidate`](@ref). Every entry is finite.
  - `Xs`: The view of each of the `K` score matrices at the same columns.

# Validation

  - Each member of a vector gives a vector of values. A member that gives an index, or any other form, is refused with an `ArgumentError`, because an index has no value to combine.
  - Under [`PerCandidate`](@ref), the scorer gives a vector of values on each view. A scorer that gives an index, or any other form, is refused with an `ArgumentError`.
  - The checks of [`assert_candidate_scores`](@ref) on the values of each member, and of each measure under [`PerCandidate`](@ref).
  - No reduced value is `NaN`. Otherwise an [`IsNonFiniteError`](@ref) that names `s_sca`, or `r_sca` for the combination of the measures.

# Returns

  - `vals`: One scorer gives its values, `VecNum`, or its return of another type, such as the index of one winner. A vector, and one scorer under [`PerCandidate`](@ref), give `VecNum`, one value per candidate.

# Related

  - [`finite_candidate_order`](@ref)
  - [`candidate_scores`](@ref)
  - [`scorer_scale`](@ref)
  - [`Sca_VecToScaM`](@ref)
  - [`PerCandidate`](@ref)
  - [`measure_candidate_values`](@ref)
  - [`assert_reduced_values`](@ref)
  - [`oriented_score`](@ref): the same two reductions for a vector of risk measures in each fold.
"""
function candidate_values(r_sca, ::Union{Option{<:Sca_VecToScaM}, EachValue},
                          scorer::CrossValSearchScorer, X::MatNum)
    vals = candidate_scores(scorer, X)
    return isnothing(vals) ? scorer(X) : vals
end
function candidate_values(r_sca::PerCandidate{<:Option{<:Scalariser}},
                          s_sca::Union{Option{<:Sca_VecToScaM}, EachValue},
                          scorer::CrossValSearchScorer, Xs::AbstractVector{<:MatNum})
    n = candidate_count(Xs)
    vals = scalarise(X -> measure_candidate_values(scorer, n,
                                                   candidate_values(r_sca, s_sca, scorer,
                                                                    X)), r_sca.sca, Xs)
    assert_reduced_values(:r_sca, r_sca.sca, "$(length(Xs)) risk measures", vals)
    return vals
end
function candidate_values(r_sca::PerCandidate{<:Union{VectorToScalarMeasure, Function}},
                          s_sca::Union{Option{<:Sca_VecToScaM}, EachValue},
                          scorer::CrossValSearchScorer, Xs::AbstractVector{<:MatNum})
    n = candidate_count(Xs)
    V = map(X -> measure_candidate_values(scorer, n,
                                          candidate_values(r_sca, s_sca, scorer, X)), Xs)
    vals = [vec_to_real_measure(r_sca.sca, getindex.(V, j)) for j in 1:n]
    assert_reduced_values(:r_sca, r_sca.sca, "$(length(Xs)) risk measures", vals)
    return vals
end
function candidate_values(r_sca, s_sca::Option{<:Sca_VecToScaM}, scorers::AbstractVector, X)
    vals = reduce_scorer_values(r_sca, s_sca, scorers, X)
    assert_reduced_values(:s_sca, s_sca, "$(length(scorers)) search scorers", vals)
    return vals
end
"""
    assert_reduced_values(slot::Symbol, sca, what::AbstractString, vals::VecNum)

Refuse the values of a reduction in a search that hold `NaN`, which no order can place.

[`candidate_values`](@ref) calls it after `s_sca` reduces a vector of search scorers, and after [`PerCandidate`](@ref) combines the values of the risk measures. Finite values can give `NaN` through a reduction: the sum of `Inf` and `-Inf`, or [`StdValue`](@ref) of one value.

# Arguments

  - `slot`: The slot of the search that holds the reduction, `:s_sca` or `:r_sca`.
  - `sca`: The reduction.
  - `what`: The count and the name of the reduced items, for the message.
  - `vals`: The reduced values, one per candidate.

# Validation

  - No value is `NaN`. Otherwise an [`IsNonFiniteError`](@ref) that names `slot`.

# Returns

  - `nothing`.

# Related

  - [`candidate_values`](@ref)
  - [`assert_candidate_scores`](@ref)
"""
function assert_reduced_values(slot::Symbol, sca, what::AbstractString,
                               vals::VecNum)::Nothing
    @argcheck(!any(isnan, vals),
              IsNonFiniteError("the reduction `$slot` of type `$(typeof(sca))` of the $what gave NaN for $(count(isnan, vals)) of the $(length(vals)) candidates that finished every fold, first at position $(findfirst(isnan, vals)). A NaN value cannot be ordered."))
    return nothing
end
"""
    candidate_count(X::MatNum)
    candidate_count(Xs::AbstractVector{<:MatNum})

Give the count of the candidates of the score matrix that a search scorer receives, or of the views of the `K` matrices under [`PerCandidate`](@ref), which share their columns.

# Arguments

  - `X`: The score matrix, one column per candidate.
  - `Xs`: The view of each score matrix at the same columns.

# Returns

  - `n::Int`: The count of the columns.

# Related

  - [`candidate_values`](@ref)
  - [`reduce_scorer_values`](@ref)
"""
function candidate_count(X::MatNum)
    return size(X, 2)
end
function candidate_count(Xs::AbstractVector{<:MatNum})
    return size(first(Xs), 2)
end
"""
    measure_candidate_values(scorer, n::Integer, vals)

Check the values that a search scorer gives for the score matrix of one risk measure under [`PerCandidate`](@ref).

[`candidate_values`](@ref) calls it once for each measure, before `r_sca.sca` combines the measures. The scale of the scorer is not read here: it weights the combined values, in the stage of the scorers.

# Arguments

  - `scorer`: The search scorer.
  - `n`: The count of the candidates.
  - `vals`: The return of the scorer on the matrix of the measure, from [`candidate_values`](@ref).

# Validation

  - `vals` is a vector. A scorer that gives an index, or any other form, is refused with an `ArgumentError`, because an index has no value to combine.
  - The checks of [`assert_candidate_scores`](@ref).

# Returns

  - `vals::VecNum`: The values of the scorer for the measure.

# Related

  - [`candidate_values`](@ref)
  - [`PerCandidate`](@ref)
  - [`member_candidate_values`](@ref): the same check for a member of a vector of scorers.
"""
function measure_candidate_values(scorer, n::Integer, vals::VecNum)
    assert_candidate_scores(scorer, vals, n)
    return vals
end
function measure_candidate_values(scorer, ::Integer, vals)
    return throw(ArgumentError("`$(typeof(scorer))` gave a `$(typeof(vals))` and no value for each candidate, so it cannot score a search under `r_sca = PerCandidate(sca)`: an index has no value to combine across the risk measures. Make it give one value per column of the score matrix, bigger is better: a function returns the vector, and a subtype adds a method `PortfolioOptimisers.candidate_scores(::$(typeof(scorer)), X::AbstractMatrix)`. Or combine the measures in each fold, with a `Sca_VecToScaM` in `r_sca`."))
end
"""
    member_candidate_values(scorer, n::Integer, vals)

Check the values of one member of a vector of search scorers, and multiply them by its scale.

[`reduce_scorer_values`](@ref) calls it once for each member, with the return of the member, before `s_sca` reduces the vector.

# Arguments

  - `scorer`: The member.
  - `n`: The count of the candidates.
  - `vals`: The return of the member, from [`candidate_values`](@ref).

# Validation

  - `vals` is a vector. A member that gives an index, or any other form, is refused with an `ArgumentError`, because an index has no value to combine.
  - The checks of [`assert_candidate_scores`](@ref).

# Returns

  - `vals::VecNum`: The values of the member, multiplied by [`scorer_scale`](@ref).

# Related

  - [`candidate_values`](@ref)
  - [`reduce_scorer_values`](@ref)
  - [`scorer_scale`](@ref)
  - [`measure_candidate_values`](@ref): the same check for one measure under [`PerCandidate`](@ref).
"""
function member_candidate_values(scorer, n::Integer, vals::VecNum)
    assert_candidate_scores(scorer, vals, n)
    return vals * scorer_scale(scorer)
end
function member_candidate_values(scorer, ::Integer, vals)
    return throw(ArgumentError("`$(typeof(scorer))` gave a `$(typeof(vals))` and no value for each candidate, so it cannot join a vector of search scorers under `s_sca`: an index has no value to combine. Make it give one value per column of the score matrix, bigger is better: a function returns the vector, and a subtype adds a method `PortfolioOptimisers.candidate_scores(::$(typeof(scorer)), X::AbstractMatrix)`. Or use it as the one scorer of the search."))
end
"""
    reduce_scorer_values(r_sca, s_sca::Option{<:Scalariser}, scorers::AbstractVector, X)
    reduce_scorer_values(r_sca, s_sca::Union{VectorToScalarMeasure, Function},
                         scorers::AbstractVector, X)

Reduce the scaled values of a vector of search scorers to one value per candidate.

Each member gives its values on `X` through [`candidate_values`](@ref) with `r_sca`, checked and scaled by [`member_candidate_values`](@ref). Under [`PerCandidate`](@ref), `X` holds one view per measure, and the values of a member are the combination of its values over the measures. Let `v[l][j]` be the scaled value of candidate `j` from member `l`. A [`Scalariser`](@ref) or `nothing` reduces through [`scalarise`](@ref), elementwise over the vectors of the members, and `nothing` gives the sum. A [`VectorToScalarMeasure`](@ref) or a function reduces the vector `[v[l][j] for l]` of each candidate through [`vec_to_real_measure`](@ref). These are the two reductions of [`oriented_score`](@ref) for a vector of risk measures.

# Arguments

  - $(arg_dict[:r_sca])
  - `s_sca`: The reduction.
  - `scorers`: The vector of search scorers.
  - `X`: The score matrix that the members receive, or one view per measure under [`PerCandidate`](@ref).

# Returns

  - `vals::VecNum`: One value per candidate.

# Related

  - [`candidate_values`](@ref)
  - [`member_candidate_values`](@ref)
  - [`scalarise`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
function reduce_scorer_values(r_sca, s_sca::Option{<:Scalariser}, scorers::AbstractVector,
                              X)
    n = candidate_count(X)
    return scalarise(s -> member_candidate_values(s, n,
                                                  candidate_values(r_sca, s_sca, s, X)),
                     s_sca, scorers)
end
function reduce_scorer_values(r_sca, s_sca::Union{VectorToScalarMeasure, Function},
                              scorers::AbstractVector, X)
    n = candidate_count(X)
    V = map(s -> member_candidate_values(s, n, candidate_values(r_sca, s_sca, s, X)),
            scorers)
    return [vec_to_real_measure(s_sca, getindex.(V, j)) for j in 1:n]
end

"""
    finite_score_columns(test_scores::MatNum)
    finite_score_columns(test_scores::AbstractVector{<:MatNum})

Give the columns of a score matrix of a search whose every entry is finite: the candidates that finished every fold.

The second form is the rule of a failed candidate under [`PerCandidate`](@ref): a column enters only when it is finite in every matrix, because a value of fewer than `K` measures does not compare with a value of `K`.

# Arguments

  - `test_scores`: The `folds × candidates`, or `paths × candidates`, score matrix, or one such matrix per measure.

# Validation

  - At least one column is finite, in every matrix for the second form. Otherwise an `IsNonFiniteError`.

# Returns

  - `cols::Vector{Int}`: The finite columns, in the order of the grid.

# Related

  - [`finite_candidate_order`](@ref)
"""
function finite_score_columns(test_scores::MatNum)
    finite = vec(all(isfinite, test_scores; dims = 1))
    @argcheck(any(finite),
              IsNonFiniteError("no parameter set finished every fold: every one of the $(size(test_scores, 2)) columns of the score matrix holds a non-finite entry over its $(size(test_scores, 1)) rows. Every candidate failed a fold, so none can be compared; widen the grid, or find why the folds failed."))
    return findall(finite)
end
function finite_score_columns(test_scores::AbstractVector{<:MatNum})
    finite = mapreduce(S -> vec(all(isfinite, S; dims = 1)), .&, test_scores)
    @argcheck(any(finite),
              IsNonFiniteError("no parameter set finished every fold under every one of the $(length(test_scores)) risk measures of `r`: each of the $(length(finite)) columns holds a non-finite entry in at least one score matrix. Under `r_sca = PerCandidate(sca)` a candidate enters only when it is finite under every measure; widen the grid, or find why the folds failed."))
    return findall(finite)
end
"""
    scorer_ranking(scorer, cols::AbstractVector{<:Integer}, n_best::Integer, vals::VecNum)
    scorer_ranking(scorer, cols::AbstractVector{<:Integer}, n_best::Integer, idx::Integer)

Rank the finite candidates by one member of a vector of search scorers under `s_sca = EachValue()`.

The values of the member are checked through [`assert_candidate_scores`](@ref), multiplied by its [`scorer_scale`](@ref), and ordered through [`finite_candidate_order`](@ref). An index keeps its one candidate with `n_best = 1`, because under [`EachValue`](@ref) a member that gives an index has no value to combine with another member.

# Arguments

  - `scorer`: The member.
  - `cols`: The positions in the grid of the finite columns of the score matrix, which the member received.
  - `n_best`: The count of the candidates to keep.
  - `vals`, `idx`: The return of the member, from [`candidate_values`](@ref).

# Validation

  - The checks of [`assert_candidate_scores`](@ref) and of [`finite_candidate_order`](@ref). A return of another type is refused with an `ArgumentError`.

# Returns

  - `(rank_idx, rank_scores)`: The ranking of the member.

# Related

  - [`finite_candidate_order`](@ref)
  - [`finite_candidate_order`](@ref)
"""
function scorer_ranking(scorer, cols::AbstractVector{<:Integer}, n_best::Integer,
                        vals::VecNum)
    # Check the raw values first, so that an infinite value times a scale of zero is not
    # reported as a NaN of the scorer.
    assert_candidate_scores(scorer, vals, length(cols))
    return finite_candidate_order(scorer, cols, n_best, vals * scorer_scale(scorer))
end
function scorer_ranking(scorer, cols::AbstractVector{<:Integer}, ::Integer, idx::Integer)
    return finite_candidate_order(scorer, cols, 1, idx)
end
function scorer_ranking(scorer, cols::AbstractVector{<:Integer}, n_best::Integer, vals)
    return finite_candidate_order(scorer, cols, n_best, vals)
end
"""
    empty_ranking(rank_idx::AbstractVector{<:Integer}, rank_scores)
    empty_ranking(rank_idx::AbstractVector{<:AbstractVector{<:Integer}},
                  rank_scores::AbstractVector)

Give an empty ranking with the form of a full one, for a matrix with no finite column under `r_sca = EachValue()`.

The columns are empty. The values are empty where the full ranking holds a vector, and `nothing` where it holds `nothing`, the entry of a scorer that gives an index. The second form does this for each of the `L` rankings of `s_sca = EachValue()`.

# Arguments

  - `rank_idx`, `rank_scores`: A full ranking from [`finite_candidate_order`](@ref).

# Returns

  - `(rank_idx, rank_scores)`: The empty ranking.

# Related

  - [`finite_candidate_order`](@ref)
"""
function empty_ranking(rank_idx::AbstractVector{<:Integer}, rank_scores)
    return similar(rank_idx, 0), isnothing(rank_scores) ? nothing : similar(rank_scores, 0)
end
function empty_ranking(rank_idx::AbstractVector{<:AbstractVector{<:Integer}},
                       rank_scores::AbstractVector)
    rankings = map(empty_ranking, rank_idx, rank_scores)
    return [r[1] for r in rankings], [r[2] for r in rankings]
end
"""
    measure_rankings(s_sca, scorer, rankings::AbstractVector)
    measure_rankings(::EachValue, scorers::AbstractVector, rankings::AbstractVector)

Arrange the rankings of the `K` score matrices of `r_sca = EachValue()` into the fields of the Result.

One ranking for each matrix gives a vector of `K` entries in each part. Under `s_sca = EachValue()` with `L` scorers, each matrix gives `L` rankings, and the parts are `K × L` matrices, row `k` for measure `k` and column `l` for scorer `l`.

# Arguments

  - `s_sca`: The reduction of a vector of search scorers, or [`EachValue`](@ref).
  - `scorer`, `scorers`: One search scorer, or a vector of them.
  - `rankings`: The ranking of each score matrix, from [`finite_candidate_order`](@ref).

# Returns

  - `rank_idx`, `rank_scores`: One entry for each value.

# Related

  - [`finite_candidate_order`](@ref)
"""
function measure_rankings(s_sca, scorer, rankings::AbstractVector)
    return [r[1] for r in rankings], [r[2] for r in rankings]
end
function measure_rankings(::EachValue, ::AbstractVector, rankings::AbstractVector)
    # Row `k` of each matrix is the `L` rankings of measure `k`.
    return stack([r[1] for r in rankings]; dims = 1),
           stack([r[2] for r in rankings]; dims = 1)
end

"""
    ranked_candidates(base, lens_grid::AbstractVector, val_grid::AbstractVector,
                      rank_idx::AbstractVector{<:Integer})
    ranked_candidates(base, lens_grid::AbstractVector, val_grid::AbstractVector,
                      rank_idx::AbstractArray{<:AbstractVector{<:Integer}})

Build the kept candidates of a search from their columns, and give the winner of each ranking.

The first form builds one ranking. The second form builds one ranking for each value, the form of [`EachValue`](@ref), and keeps the shape of `rank_idx`. An empty ranking gives `nothing` for its winner and its column, and an empty vector of candidates with the element type of a ranking that is not empty.

# Arguments

  - `base`: The estimator, or the [`Pipeline`](@ref), that the search tunes.
  - `lens_grid`: The lenses of each grid point.
  - `val_grid`: The values of each grid point.
  - `rank_idx`: The columns of the kept candidates, best first, or one vector of them for each value.

# Returns

  - `opt`: The winner, `ranked[1]`, or the winner of each ranking.
  - `idx`: The column of the winner, `rank_idx[1]`, or the column of the winner of each ranking.
  - `ranked`: The kept candidates, built from `base` through [`search_candidate`](@ref), or one vector of them for each value.

# Related

  - [`search_result`](@ref)
  - [`finite_candidate_order`](@ref)
  - [`search_candidate`](@ref)
"""
function ranked_candidates(base, lens_grid::AbstractVector, val_grid::AbstractVector,
                           rank_idx::AbstractVector{<:Integer})
    ranked = [search_candidate(base, lens_grid[k], val_grid[k]) for k in rank_idx]
    return first(ranked), first(rank_idx), ranked
end
function ranked_candidates(base, lens_grid::AbstractVector, val_grid::AbstractVector,
                           rank_idx::AbstractArray{<:AbstractVector{<:Integer}})
    # An empty ranking takes the element type of a full one, so that every entry of
    # `ranked` is a vector of estimators.
    build(ri) = [search_candidate(base, lens_grid[k], val_grid[k]) for k in ri]
    full = build(first(Iterators.filter(!isempty, rank_idx)))
    ranked = [isempty(ri) ? similar(full, 0) : build(ri) for ri in rank_idx]
    return get.(ranked, 1, nothing), get.(rank_idx, 1, nothing), ranked
end
"""
    allocate_search_predictions(gscv, N::Integer)

Allocate the store of the predictions of the `N` candidates of a search, or give `nothing`.

The four methods of [`search_cross_validation`](@ref) on an optimiser and on a [`Pipeline`](@ref), batch and combinatorial, call it before their score loop. With `gscv.rank_prediction = true`, the loop writes the prediction of candidate `i` at entry `i`, and [`ranked_predictions`](@ref) keeps the entries of the kept candidates. With `false`, the loop stores no prediction. The element type is [`AbstractPredictionResult`](@ref), because only the fit gives the type of a prediction. [`ranked_predictions`](@ref) narrows the kept entries to the type of their values.

# Arguments

  - `gscv`: The [`GridSearchCrossValidation`](@ref). It gives `rank_prediction`.
  - `N`: The candidate count.

# Returns

  - `predictions::Option{<:AbstractVector{<:AbstractPredictionResult}}`: A store of `N` entries, uninitialised, or `nothing`.

# Related

  - [`ranked_predictions`](@ref)
  - [`search_result`](@ref)
  - [`allocate_search_scores`](@ref)
"""
function allocate_search_predictions(gscv, N::Integer)
    return gscv.rank_prediction ? Vector{AbstractPredictionResult}(undef, N) : nothing
end
"""
    ranked_predictions(::Nothing, rank_idx::AbstractArray)
    ranked_predictions(predictions::AbstractVector{<:AbstractPredictionResult},
                       rank_idx::AbstractVector{<:Integer})
    ranked_predictions(predictions::AbstractVector{<:AbstractPredictionResult},
                       rank_idx::AbstractArray{<:AbstractVector{<:Integer}})

Keep the predictions of the kept candidates of a search, in the order of `rank_idx`.

The first form is a search with `rank_prediction = false`, and gives `nothing`. The second form keeps one ranking. The third form keeps one ranking for each value, the form of [`EachValue`](@ref), and keeps the shape of `rank_idx`. Each entry is the object of `predictions`, with no copy, so two rankings that keep the same candidate hold the same object. A comprehension narrows each ranking to the type of its predictions. An empty ranking gives an empty vector with the element type of a ranking that is not empty, as [`ranked_candidates`](@ref) does.

# Arguments

  - `predictions`: The prediction of each candidate, in grid order, or `nothing`.
  - `rank_idx`: The columns of the kept candidates, best first, or one vector of them for each value.

# Returns

  - `rank_predictions`: The kept predictions, one vector of them for each value, or `nothing`.

# Related

  - [`allocate_search_predictions`](@ref)
  - [`search_result`](@ref)
  - [`ranked_candidates`](@ref)
"""
function ranked_predictions(::Nothing, ::AbstractArray)
    return nothing
end
function ranked_predictions(predictions::AbstractVector{<:AbstractPredictionResult},
                            rank_idx::AbstractVector{<:Integer})
    return [predictions[k] for k in rank_idx]
end
function ranked_predictions(predictions::AbstractVector{<:AbstractPredictionResult},
                            rank_idx::AbstractArray{<:AbstractVector{<:Integer}})
    full = ranked_predictions(predictions, first(Iterators.filter(!isempty, rank_idx)))
    return [isempty(ri) ? similar(full, 0) : ranked_predictions(predictions, ri)
            for ri in rank_idx]
end
"""
    store_search_prediction!(::Nothing, i::Integer, p::AbstractPredictionResult)
    store_search_prediction!(store::AbstractVector{<:AbstractPredictionResult}, i::Integer,
                             p::AbstractPredictionResult)

Store the prediction of candidate `i` of a search at entry `i` of `store`.

The score loop of each method of [`search_cross_validation`](@ref) calls it after [`write_candidate_scores!`](@ref). The first form is a search with `rank_prediction = false`, whose store is `nothing`, and stores nothing. The prediction is the object that the loop scored, with no copy.

# Arguments

  - `store`: The store of [`allocate_search_predictions`](@ref), or `nothing`.
  - `i`: The column of the candidate.
  - `p`: The prediction of the candidate.

# Returns

  - `nothing`.

# Related

  - [`allocate_search_predictions`](@ref)
  - [`ranked_predictions`](@ref)
  - [`store_search_score!`](@ref)
"""
function store_search_prediction!(::Nothing, ::Integer, ::AbstractPredictionResult)
    return nothing
end
function store_search_prediction!(store::AbstractVector{<:AbstractPredictionResult},
                                  i::Integer, p::AbstractPredictionResult)
    store[i] = p
    return nothing
end
"""
    assert_rank_predictions(rank_idx::AbstractArray, ::Nothing)
    assert_rank_predictions(rank_idx::AbstractVector{<:Integer},
                            rank_predictions::AbstractVector)
    assert_rank_predictions(rank_idx::AbstractArray{<:AbstractVector{<:Integer}},
                            rank_predictions::AbstractArray)

Refuse the kept predictions of a [`SearchCrossValidationResult`](@ref) when they do not agree with its ranking.

The inner constructor of the Result calls it after [`assert_search_ranking`](@ref). The first form is a Result with no predictions, and checks nothing. The second form checks one ranking, and the third form checks one ranking for each value, the form of [`EachValue`](@ref).

# Arguments

  - `rank_idx`: The columns of the kept candidates, or one vector of them for each value.
  - `rank_predictions`: The kept predictions, in the shape of `rank_idx`, or `nothing`.

# Validation

  - The second form: `length(rank_predictions) == length(rank_idx)`. Otherwise a `DimensionMismatch`.
  - The third form: `size(rank_predictions) == size(rank_idx)`, and each entry holds one prediction for each entry of `rank_idx`. Otherwise a `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`assert_search_ranking`](@ref)
  - [`ranked_predictions`](@ref)
"""
function assert_rank_predictions(::AbstractArray, ::Nothing)
    return nothing
end
function assert_rank_predictions(rank_idx::AbstractVector{<:Integer},
                                 rank_predictions::AbstractVector)
    @argcheck(length(rank_predictions) == length(rank_idx),
              DimensionMismatch("`rank_predictions` must hold one prediction per kept candidate, in the order of `ranked`: got $(length(rank_predictions)) predictions for the columns $(rank_idx)."))
    return nothing
end
function assert_rank_predictions(rank_idx::AbstractArray{<:AbstractVector{<:Integer}},
                                 rank_predictions::AbstractArray)
    @argcheck(size(rank_predictions) == size(rank_idx),
              DimensionMismatch("`rank_predictions` must hold one entry for each value, in the shape of `rank_idx`: got the sizes $(size(rank_predictions)) and $(size(rank_idx))."))
    for v in eachindex(rank_idx)
        @argcheck(length(rank_predictions[v]) == length(rank_idx[v]),
                  DimensionMismatch("entry $v of `rank_predictions` must hold one prediction per kept candidate of `rank_idx[$v]`: got $(length(rank_predictions[v])) predictions for the columns $(rank_idx[v])."))
    end
    return nothing
end

export HighestMeanScore
# The `# Interfaces`-marked type of #1138 (ADR 0154). `CrossValidationSearchScorer` names its
# callable shape and the verb `candidate_scores` (#1937), which a subtype implements, and
# `scorer_scale` (#2000), which a subtype with a scale implements.
public CrossValidationSearchScorer, candidate_scores, scorer_scale
