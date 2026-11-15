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
    finite_candidate_order(s_sca::Option{<:Sca_VecToScaM}, scorer, test_scores::MatNum,
                           n_best::Integer)

Order the candidates of a search, best first, keep the first `n_best`, and never let a failed candidate in.

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

# Algorithm

 1. Mark the columns of `test_scores` whose every entry is finite, giving `cols`.
 2. Throw an `IsNonFiniteError` when no column is finite.
 3. Take the view `X` of `test_scores` at `cols`, and call [`candidate_values`](@ref) with `s_sca` and `scorer` on it, giving `vals`.
 4. If `vals` is a vector: check it through [`assert_candidate_scores`](@ref). Then take the first `min(n_best, length(cols))` positions of the descending order of `vals` through a partial sort, which breaks a tie by the position, and so gives the order of a stable sort. Return the columns of `cols` at those positions and their values.
 5. If `vals` is an integer, the index of one winner: refuse `n_best > 1`, and return its column of `cols` and `nothing`.
 6. Refuse a `vals` of any other type with an `ArgumentError`.

# Arguments

  - `s_sca`: The reduction of a vector of search scorers. Inert for one scorer.
  - `scorer`: One search scorer ([`CrossValSearchScorer`](@ref)), or a vector of them.
  - `test_scores`: The `folds × candidates`, or `paths × candidates`, score matrix.
  - `n_best`: The count of the candidates to keep.

# Validation

  - At least one candidate must have finished every fold. Otherwise an `IsNonFiniteError`.
  - A scorer that gives an index keeps one candidate. Otherwise an `ArgumentError` names the type of the scorer and the form to give.
  - The checks of [`assert_candidate_scores`](@ref) and [`candidate_values`](@ref).
  - A scorer gives a vector or an integer. Otherwise an `ArgumentError`.

# Returns

  - `rank_idx::AbstractVector{<:Integer}`: The positions in the parameter grid of the kept candidates, best first.
  - `rank_scores::Option{<:VecNum}`: The value of [`candidate_values`](@ref) of each kept candidate, in the order of `rank_idx`, or `nothing` when the scorer gives an index.

# Related

  - [`candidate_values`](@ref)
  - [`candidate_scores`](@ref)
  - [`CrossValidationSearchScorer`](@ref)
  - [`CrossValSearchScorer`](@ref)
  - [`search_cross_validation`](@ref)
  - [`IsNonFiniteError`](@ref)
"""
function finite_candidate_order(s_sca::Option{<:Sca_VecToScaM}, scorer, test_scores::MatNum,
                                n_best::Integer)
    finite = vec(all(isfinite, test_scores; dims = 1))
    @argcheck(any(finite),
              IsNonFiniteError("no parameter set finished every fold: every one of the $(size(test_scores, 2)) columns of the score matrix holds a non-finite entry over its $(size(test_scores, 1)) rows. Every candidate failed a fold, so none can be compared; widen the grid, or find why the folds failed."))
    cols = findall(finite)
    X = view(test_scores, :, cols)
    return finite_candidate_order(scorer, cols, n_best, candidate_values(s_sca, scorer, X))
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
    candidate_values(s_sca::Option{<:Sca_VecToScaM}, scorer::CrossValSearchScorer, X::MatNum)
    candidate_values(s_sca::Option{<:Sca_VecToScaM}, scorers::AbstractVector, X::MatNum)

Give what the search scorer, or the vector of them, gives for the candidates of a search: one value per column of `X`, bigger is better, or the index of one winner.

One scorer gives its [`candidate_scores`](@ref). When these are `nothing`, it is called once on `X`, and the return gives the form: a vector of values from a function, or the index of one winner. `s_sca` and [`scorer_scale`](@ref) are inert, so a search of one scorer makes the one call of the scorer that it made before.

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
 2. A vector: give the return of each member as in step 1. Check each one through [`assert_candidate_scores`](@ref), and multiply it by the scale of its member.
 3. Reduce the scaled vectors through the type of `s_sca`, with [`reduce_scorer_values`](@ref), and check that no reduced value is `NaN`.

# Arguments

  - `s_sca`: The reduction of a vector of scorers.
  - `scorer`: One search scorer.
  - `scorers`: A vector of search scorers.
  - `X`: The `rows × candidates` score matrix that the scorers receive. Every entry is finite.

# Validation

  - Each member of a vector gives a vector of values. A member that gives an index, or any other form, is refused with an `ArgumentError`, because an index has no value to combine.
  - The checks of [`assert_candidate_scores`](@ref) on the values of each member.
  - No reduced value is `NaN`. Otherwise an [`IsNonFiniteError`](@ref) that names `s_sca`.

# Returns

  - `vals`: One scorer gives its values, `VecNum`, or its return of another type, such as the index of one winner. A vector gives `VecNum`, one value per column of `X`.

# Related

  - [`finite_candidate_order`](@ref)
  - [`candidate_scores`](@ref)
  - [`scorer_scale`](@ref)
  - [`Sca_VecToScaM`](@ref)
  - [`oriented_score`](@ref): the same two reductions for a vector of risk measures.
"""
function candidate_values(::Option{<:Sca_VecToScaM}, scorer::CrossValSearchScorer,
                          X::MatNum)
    vals = candidate_scores(scorer, X)
    return isnothing(vals) ? scorer(X) : vals
end
function candidate_values(s_sca::Option{<:Sca_VecToScaM}, scorers::AbstractVector,
                          X::MatNum)
    vals = reduce_scorer_values(s_sca, scorers, X)
    @argcheck(!any(isnan, vals),
              IsNonFiniteError("the reduction `s_sca` of type `$(typeof(s_sca))` of the $(length(scorers)) search scorers gave NaN for $(count(isnan, vals)) of the $(size(X, 2)) candidates that finished every fold, first at position $(findfirst(isnan, vals)). A NaN value cannot be ordered."))
    return vals
end
"""
    member_candidate_values(scorer, X::MatNum, vals)

Check the values of one member of a vector of search scorers, and multiply them by its scale.

[`reduce_scorer_values`](@ref) calls it once for each member, with the return of the member, before `s_sca` reduces the vector.

# Arguments

  - `scorer`: The member.
  - `X`: The score matrix that the member received.
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
"""
function member_candidate_values(scorer, X::MatNum, vals::VecNum)
    assert_candidate_scores(scorer, vals, size(X, 2))
    return vals * scorer_scale(scorer)
end
function member_candidate_values(scorer, ::MatNum, vals)
    return throw(ArgumentError("`$(typeof(scorer))` gave a `$(typeof(vals))` and no value for each candidate, so it cannot join a vector of search scorers under `s_sca`: an index has no value to combine. Make it give one value per column of the score matrix, bigger is better: a function returns the vector, and a subtype adds a method `PortfolioOptimisers.candidate_scores(::$(typeof(scorer)), X::AbstractMatrix)`. Or use it as the one scorer of the search."))
end
"""
    reduce_scorer_values(s_sca::Option{<:Scalariser}, scorers::AbstractVector, X::MatNum)
    reduce_scorer_values(s_sca::Union{VectorToScalarMeasure, Function},
                         scorers::AbstractVector, X::MatNum)

Reduce the scaled values of a vector of search scorers to one value per candidate.

Each member gives its values on `X` through [`candidate_values`](@ref), checked and scaled by [`member_candidate_values`](@ref). Let `v[l][j]` be the scaled value of candidate `j` from member `l`. A [`Scalariser`](@ref) or `nothing` reduces through [`scalarise`](@ref), elementwise over the vectors of the members, and `nothing` gives the sum. A [`VectorToScalarMeasure`](@ref) or a function reduces the vector `[v[l][j] for l]` of each candidate through [`vec_to_real_measure`](@ref). These are the two reductions of [`oriented_score`](@ref) for a vector of risk measures.

# Arguments

  - `s_sca`: The reduction.
  - `scorers`: The vector of search scorers.
  - `X`: The score matrix that the members receive.

# Returns

  - `vals::VecNum`: One value per column of `X`.

# Related

  - [`candidate_values`](@ref)
  - [`member_candidate_values`](@ref)
  - [`scalarise`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
function reduce_scorer_values(s_sca::Option{<:Scalariser}, scorers::AbstractVector,
                              X::MatNum)
    return scalarise(s -> member_candidate_values(s, X, candidate_values(s_sca, s, X)),
                     s_sca, scorers)
end
function reduce_scorer_values(s_sca::Union{VectorToScalarMeasure, Function},
                              scorers::AbstractVector, X::MatNum)
    V = map(s -> member_candidate_values(s, X, candidate_values(s_sca, s, X)), scorers)
    return [vec_to_real_measure(s_sca, getindex.(V, j)) for j in axes(X, 2)]
end

export HighestMeanScore
# The `# Interfaces`-marked type of #1138 (ADR 0154). `CrossValidationSearchScorer` names its
# callable shape and the verb `candidate_scores` (#1937), which a subtype implements, and
# `scorer_scale` (#2000), which a subtype with a scale implements.
public CrossValidationSearchScorer, candidate_scores, scorer_scale
