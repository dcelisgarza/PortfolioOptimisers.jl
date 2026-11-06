"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the statistics that one test window gives to each asset that it covers.

A [`FoldRiskScorer`](@ref) evaluates a risk measure on each test window. A statistic turns the values of one window into the numbers that an [`AbstractFoldAggregation`](@ref) then reduces over the windows. [`Value`](@ref) keeps the raw value of the measure, and [`Rank`](@ref) replaces it with its rank in the window. A statistic also states its own orientation, because a rank has an orientation of its own, and it is not that of the measure.

# Interfaces

To add a statistic, subtype `AbstractFoldStatistic`, put every parameter of the statistic in the struct, and add the two methods below.

## `fold_statistic`

  - `fold_statistic(alg::MyFoldStatistic, v::VecNum, bib::Bool) -> VecNum`: Turn the values of one test window into one statistic for each asset that the window covers.

### Arguments

  - `alg`: The concrete statistic.
  - `v`: The values of the risk measure on one test window, one entry for each asset that the window covers.
  - `bib`: The [`bigger_is_better`](@ref) flag of the risk measure.

### Returns

  - `stat::VecNum`: One statistic for each entry of `v`, in the same order.

## `statistic_bigger_is_better`

  - `statistic_bigger_is_better(alg::MyFoldStatistic, bib::Bool) -> Bool`: State whether a bigger statistic is better, from the orientation `bib` of the risk measure.

### Arguments

  - `alg`: The concrete statistic.
  - `bib`: The [`bigger_is_better`](@ref) flag of the risk measure.

### Returns

  - `sbib::Bool`: `true` when a bigger statistic is better.

# Related

  - [`Value`](@ref)
  - [`Rank`](@ref)
  - [`AbstractFoldAggregation`](@ref)
  - [`FoldRiskScorer`](@ref)
"""
abstract type AbstractFoldStatistic <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Keeps the raw value of the risk measure on each test window.

The statistic of asset ``i`` on a window is the value of the measure on that window. It has the orientation of the measure, so a fold aggregation of it is in the units of the measure. A window of high volatility then weighs more in a mean than a calm window.

# Constructors

    Value() -> Value

# Examples

```jldoctest
julia> PortfolioOptimisers.fold_statistic(Value(), [0.3, 0.1, 0.2], false)
3-element Vector{Float64}:
 0.3
 0.1
 0.2
```

# Related

  - [`AbstractFoldStatistic`](@ref)
  - [`Rank`](@ref)
"""
struct Value <: AbstractFoldStatistic end
"""
$(DocStringExtensions.TYPEDEF)

Replaces the value of the risk measure on each test window with the percentile rank of the asset in that window.

The rank reads the orientation of the measure, so the best asset of a window takes the lowest rank, whether the measure is a risk or a reward. The rank is free of the scale of the window, so a window of high volatility weighs no more than a calm one. It is also free of the number of assets, so two windows that cover different assets give comparable ranks.

# Mathematical definition

```math
\\begin{align}
p_{i} &= \\frac{n^{<}_{i} + n^{\\leq}_{i}}{2 n}\\,.
\\end{align}
```

Where:

  - ``p_{i}``: The rank of asset ``i`` in the window, strictly inside the open unit interval.
  - ``n``: The number of assets that the window covers.
  - ``n^{<}_{i}``: The number of covered assets that are strictly better than asset ``i`` under the orientation of the measure.
  - ``n^{\\leq}_{i}``: The number of covered assets that are better than asset ``i`` or equal to it, asset ``i`` included.

A tie shares the average of the ranks that its members take with no tie. The best asset of a window with no tie has the rank ``1 / (2 n)``, and the worst has ``1 - 1 / (2 n)``. A lower rank is better, so the statistic has the orientation `false`.

# Constructors

    Rank() -> Rank

# Examples

```jldoctest
julia> PortfolioOptimisers.fold_statistic(Rank(), [0.3, 0.1, 0.1, 0.2], false)
4-element Vector{Float64}:
 0.875
 0.25
 0.25
 0.625
```

# Related

  - [`AbstractFoldStatistic`](@ref)
  - [`Value`](@ref)
  - [`CrossSectionalPercentileRank`](@ref)
"""
struct Rank <: AbstractFoldStatistic end
"""
    fold_statistic(alg::Value, v::VecNum, bib::Bool) -> VecNum
    fold_statistic(alg::Rank, v::VecNum, bib::Bool) -> VecNum

Turn the values of a risk measure on one test window into one statistic for each covered asset.

The [`Value`](@ref) method returns `v`. The [`Rank`](@ref) method returns the percentile rank that the docstring of [`Rank`](@ref) defines.

# Algorithm

The [`Rank`](@ref) method runs these steps:

 1. Orient the values with [`drop_scores`](@ref), so that a bigger entry of `u` is a worse asset.
 2. Sort `u` into `s`.
 3. For each entry of `u`, count with [`cross_sectional_rank_counts`](@ref) the entries of `s` strictly below it and those at or below it.
 4. Divide the sum of the two counts by `2n`, and return the ranks.

# Arguments

  - `alg`: The statistic.
  - `v`: The values of the risk measure on one test window, one entry for each covered asset.
  - `bib`: The [`bigger_is_better`](@ref) flag of the risk measure.

# Returns

  - `stat::VecNum`: One statistic for each entry of `v`, in the same order.

# Related

  - [`AbstractFoldStatistic`](@ref)
  - [`statistic_bigger_is_better`](@ref)
"""
function fold_statistic(::Value, v::VecNum, ::Bool)
    return v
end
function fold_statistic(::Rank, v::VecNum, bib::Bool)
    u = drop_scores(v, bib)
    s = sort(u)
    n = length(u)
    return [sum(cross_sectional_rank_counts(s, x)) / (2 * n) for x in u]
end
"""
    statistic_bigger_is_better(alg::Value, bib::Bool) -> Bool
    statistic_bigger_is_better(alg::Rank, bib::Bool) -> Bool

State whether a bigger statistic is better, from the orientation `bib` of the risk measure.

The [`Value`](@ref) method returns `bib`, because the raw value has the orientation of the measure. The [`Rank`](@ref) method returns `false`, because the best asset of a window takes the lowest rank.

# Arguments

  - `alg`: The statistic.
  - `bib`: The [`bigger_is_better`](@ref) flag of the risk measure.

# Returns

  - `sbib::Bool`: `true` when a bigger statistic is better.

# Related

  - [`AbstractFoldStatistic`](@ref)
  - [`fold_statistic`](@ref)
"""
function statistic_bigger_is_better(::Value, bib::Bool)::Bool
    return bib
end
function statistic_bigger_is_better(::Rank, ::Bool)::Bool
    return false
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that reduce the statistics of one asset over the test windows to one score.

A [`FoldRiskScorer`](@ref) gives each asset one statistic for each test window that covers it. The aggregation reduces those statistics to the score of the asset. Each aggregation holds an [`AbstractFoldStatistic`](@ref) in its `alg` field, so one rule can reduce the raw values with [`Value`](@ref) or the ranks with [`Rank`](@ref). An aggregation keeps the orientation of its statistic. A bigger score is better exactly when a bigger statistic is better.

# Interfaces

To add an aggregation, subtype `AbstractFoldAggregation`, give it an `alg` field that holds an [`AbstractFoldStatistic`](@ref), and add the method below.

## `fold_aggregate`

  - `fold_aggregate(agg::MyFoldAggregation, v::VecNum, sbib::Bool) -> Number`: Reduce the statistics of one asset over the windows that cover it to one score.

### Arguments

  - `agg`: The concrete aggregation.
  - `v`: The statistics of one asset, one entry for each window that covers it.
  - `sbib`: The orientation of the statistic, from [`statistic_bigger_is_better`](@ref).

### Returns

  - `s::Number`: The score of the asset, with the orientation `sbib`.

# Related

  - [`MeanFoldScore`](@ref)
  - [`MedianFoldScore`](@ref)
  - [`WorstFoldScore`](@ref)
  - [`AbstractFoldStatistic`](@ref)
  - [`FoldRiskScorer`](@ref)
"""
abstract type AbstractFoldAggregation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Scores an asset by the mean of its statistic over the test windows that cover it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MeanFoldScore(;
        alg::AbstractFoldStatistic = Value(),
    ) -> MeanFoldScore

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> MeanFoldScore(; alg = Rank())
MeanFoldScore
  alg ┴ Rank()
```

# Related

  - [`AbstractFoldAggregation`](@ref)
  - [`MedianFoldScore`](@ref)
  - [`WorstFoldScore`](@ref)
"""
@concrete struct MeanFoldScore <: AbstractFoldAggregation
    """
    $(field_dict[:fold_alg])
    """
    alg
    function MeanFoldScore(alg::AbstractFoldStatistic)
        return new{typeof(alg)}(alg)
    end
end
function MeanFoldScore(; alg::AbstractFoldStatistic = Value())::MeanFoldScore
    return MeanFoldScore(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Scores an asset by the median of its statistic over the test windows that cover it.

One window of extreme values moves the median less than it moves the mean.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MedianFoldScore(;
        alg::AbstractFoldStatistic = Value(),
    ) -> MedianFoldScore

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> MedianFoldScore()
MedianFoldScore
  alg ┴ Value()
```

# Related

  - [`AbstractFoldAggregation`](@ref)
  - [`MeanFoldScore`](@ref)
  - [`WorstFoldScore`](@ref)
"""
@concrete struct MedianFoldScore <: AbstractFoldAggregation
    """
    $(field_dict[:fold_alg])
    """
    alg
    function MedianFoldScore(alg::AbstractFoldStatistic)
        return new{typeof(alg)}(alg)
    end
end
function MedianFoldScore(; alg::AbstractFoldStatistic = Value())::MedianFoldScore
    return MedianFoldScore(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Scores an asset by its worst statistic over the test windows that cover it.

The worst statistic is the largest one when a smaller statistic is better, and the smallest one when a bigger statistic is better. With [`Value`](@ref) it is the maximum of a risk and the minimum of a reward. With [`Rank`](@ref) it is the largest rank, the window in which the asset placed lowest.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WorstFoldScore(;
        alg::AbstractFoldStatistic = Value(),
    ) -> WorstFoldScore

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> PortfolioOptimisers.fold_aggregate(WorstFoldScore(), [0.1, 0.4, 0.2], false)
0.4

julia> PortfolioOptimisers.fold_aggregate(WorstFoldScore(), [0.1, 0.4, 0.2], true)
0.1
```

# Related

  - [`AbstractFoldAggregation`](@ref)
  - [`MeanFoldScore`](@ref)
  - [`MedianFoldScore`](@ref)
"""
@concrete struct WorstFoldScore <: AbstractFoldAggregation
    """
    $(field_dict[:fold_alg])
    """
    alg
    function WorstFoldScore(alg::AbstractFoldStatistic)
        return new{typeof(alg)}(alg)
    end
end
function WorstFoldScore(; alg::AbstractFoldStatistic = Value())::WorstFoldScore
    return WorstFoldScore(alg)
end
"""
    fold_aggregate(agg::MeanFoldScore, v::VecNum, sbib::Bool) -> Number
    fold_aggregate(agg::MedianFoldScore, v::VecNum, sbib::Bool) -> Number
    fold_aggregate(agg::WorstFoldScore, v::VecNum, sbib::Bool) -> Number

Reduce the statistics of one asset over the test windows that cover it to one score.

The [`MeanFoldScore`](@ref) method returns the mean of `v`, and the [`MedianFoldScore`](@ref) method returns its median. Neither reads `sbib`. The [`WorstFoldScore`](@ref) method returns the minimum of `v` when `sbib` is `true`, and the maximum otherwise.

# Arguments

  - `agg`: The aggregation.
  - `v`: The statistics of one asset, one entry for each window that covers it.
  - `sbib`: The orientation of the statistic, from [`statistic_bigger_is_better`](@ref).

# Returns

  - `s::Number`: The score of the asset, with the orientation `sbib`.

# Related

  - [`AbstractFoldAggregation`](@ref)
  - [`FoldRiskScorer`](@ref)
"""
function fold_aggregate(::MeanFoldScore, v::VecNum, ::Bool)
    return mean(v)
end
function fold_aggregate(::MedianFoldScore, v::VecNum, ::Bool)
    return Statistics.median(v)
end
function fold_aggregate(::WorstFoldScore, v::VecNum, sbib::Bool)
    return sbib ? minimum(v) : maximum(v)
end
"""
    distinct_test_windows(test_idx::VecVecInt) -> VecVecInt
    distinct_test_windows(test_idx::VecVecVecInt) -> VecVecInt

Return each distinct test window of a cross-validation split once.

A [`CombinatorialCrossValidation`](@ref) split holds a vector of test blocks, and one block is in many splits. The method on a `VecVecVecInt` flattens the blocks of every split, and keeps each block once. The method on a `VecVecInt` keeps each window once. An aggregation over the windows then weighs each window once.

# Arguments

  - `test_idx`: The test indices of a split result, one window for each split, or one vector of blocks for each split.

# Returns

  - `windows::VecVecInt`: The distinct test windows, in the order of their first appearance.

# Related

  - [`FoldRiskScorer`](@ref)
"""
function distinct_test_windows(test_idx::VecVecInt)
    return unique(test_idx)
end
function distinct_test_windows(test_idx::VecVecVecInt)
    return unique(Iterators.flatten(test_idx))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the score of each asset that at least `min_folds` folds scored, and `NaN` at every other asset.

An Asset Scorer scores an asset only on a fold that offers it. A fold does not offer an asset outside the Coverage Universe of its window, or outside the asset subset that a [`MultipleRandomised`](@ref) split drew. So the number of folds can differ from asset to asset, and a score from two folds is not as reliable as a score from twenty. An asset with fewer than `min_folds` folds has no score, and the default `min_folds = 1` marks an asset that no fold scored.

A `NaN` score is outside every selection rule and every redundancy algorithm, so the asset leaves the selection and the run does not stop. [`rule_keep`](@ref) states that rule.

# Algorithm

 1. Find `ok`, the positions of the assets whose count is at least `min_folds`.
 2. Evaluate `score_of` at each position of `ok` into `v`. The function calls `score_of` at no other position, so an aggregation never reads an empty set of folds.
 3. Read the number type `S` of `v`, or `T` when `v` is empty.
 4. Return the vector of that type that holds `v` at `ok`, and `NaN` at every other position.

# Arguments

  - `score_of`: The function that returns the score of the asset at a position.
  - `counts`: The number of folds that scored each asset, `assets × 1`.
  - `min_folds`: The smallest number of folds that must score an asset.
  - `T`: The number type of the returns data. The function reads it only when no asset has enough folds.

# Returns

  - `scores::VecNum`: The score vector `assets × 1`. It is `NaN` at each asset with fewer than `min_folds` folds.

# Related

  - [`FoldRiskScorer`](@ref)
  - [`WeightMassScorer`](@ref)
  - [`rule_keep`](@ref)
"""
function min_folds_scores(score_of, counts::VecInt, min_folds::Integer, ::Type{T}) where {T}
    ok = findall(>=(min_folds), counts)
    v = [score_of(i) for i in ok]
    S = isempty(v) ? T : eltype(v)
    scores = fill!(Vector{S}(undef, length(counts)), NaN)
    scores[ok] = v
    return scores
end
"""
$(DocStringExtensions.TYPEDEF)

Scores each asset with a risk measure on each test window of a cross-validation scheme, and aggregates the values over the windows.

One window of bad luck then does not decide the rank of an asset. The score of an asset reads its own column alone, so the training rows do not change it. The scheme gives the windows only. `agg` sets the rule of the aggregation, and its `alg` sets whether it reduces the raw values with [`Value`](@ref) or the ranks with [`Rank`](@ref).

# Mathematical definition

```math
\\begin{align}
v_{i k} &= r\\left(\\boldsymbol{x}_{T_{k},\\, i}\\right)\\,, \\\\
s_{i} &= A\\left(\\left\\{ g_{i k} : k \\in \\mathcal{F}_{i} \\right\\}\\right)\\,.
\\end{align}
```

Where:

  - ``v_{i k}``: The value of the risk measure on the returns of asset ``i`` over the test window ``T_{k}``.
  - ``r``: The risk measure `r`.
  - ``\\boldsymbol{x}_{T_{k},\\, i}``: The returns of asset ``i`` at the rows of the test window ``T_{k}``.
  - ``T_{k}``: The ``k``-th distinct test window of `split(cv, rd)`.
  - ``g_{i k}``: The statistic of asset ``i`` on window ``T_{k}``. It is ``v_{i k}`` under [`Value`](@ref), and the rank of ``v_{i k}`` among the covered assets of the window under [`Rank`](@ref).
  - ``\\mathcal{F}_{i}``: The windows whose Coverage Universe holds asset ``i``.
  - ``A``: The aggregation `agg`.
  - ``s_{i}``: The score of asset ``i``.

The scorer evaluates each distinct window once. A [`CombinatorialCrossValidation`](@ref) split puts each test block in many splits. A count of each split weighs some blocks more than others. A [`MultipleRandomised`](@ref) split draws asset subsets. The scorer ignores them, because the score of an asset does not depend on the other assets.

A window scores only the assets of its own Coverage Universe, the assets whose return is finite and active at every row of the window. An asset with a gap in one window keeps its score from the other windows. Under [`Rank`](@ref), a window ranks only the assets that it covers.

An asset that fewer than `min_folds` windows cover has the score `NaN`. Every selection rule and every redundancy algorithm drops a `NaN` score, so a selector drops that asset and the run does not stop. Inside a selector the training window is already reduced to its Coverage Universe, so the case occurs on a direct call, or with a `min_folds` above one.

# Algorithm

 1. Read the orientation `bib` of `r` with [`bigger_is_better`](@ref).
 2. Take the distinct test windows of `split(cv, rd)` with [`distinct_test_windows`](@ref).
 3. For each window, find the positions `live` of the covered assets with [`cv_live_assets`](@ref).
 4. Evaluate `r` on the covered columns of the window with [`asset_scores`](@ref), and turn the values into statistics with [`fold_statistic`](@ref).
 5. Collect the statistics of each asset over the windows that cover it into `stats`.
 6. Reduce the statistics of each asset that at least `min_folds` windows cover with [`fold_aggregate`](@ref), under the orientation from [`statistic_bigger_is_better`](@ref). Write `NaN` at every other asset. [`min_folds_scores`](@ref) does this step.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FoldRiskScorer(;
        r::AbstractBaseRiskMeasure,
        cv::CVE_Onl = KFold(),
        agg::AbstractFoldAggregation = MeanFoldScore(),
        min_folds::Integer = 1,
    ) -> FoldRiskScorer

Keywords correspond to the struct's fields.

## Validation

  - `supports_precomputed_returns(r)`, through [`assert_scoreable`](@ref).
  - `min_folds >= 1`.

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = [\"A\", \"B\", \"C\"],
                          X = [0.1 0.0 -0.2; -0.1 0.1 0.3; 0.2 0.0 -0.1; 0.0 -0.1 0.2]);

julia> s = FoldRiskScorer(; r = SCM(), cv = KFold(; n = 2));

julia> round.(PortfolioOptimisers.asset_scores(s, rd); digits = 4)
3-element Vector{Float64}:
 0.02
 0.005
 0.085
```

# Related

  - [`AbstractAssetScorer`](@ref)
  - [`AbstractFoldAggregation`](@ref)
  - [`AbstractFoldStatistic`](@ref)
  - [`ScoreSelector`](@ref)
  - [`WeightMassScorer`](@ref)
"""
@concrete struct FoldRiskScorer <: AbstractAssetScorer
    """
    The [`AbstractBaseRiskMeasure`](@ref) that scores the returns of each asset on each test window.
    """
    r
    """
    $(field_dict[:cv])
    """
    cv
    """
    The [`AbstractFoldAggregation`](@ref) that reduces the statistics of each asset over the windows.
    """
    agg
    """
    The smallest number of test windows that must cover an asset. An asset that fewer windows cover has the score `NaN`.
    """
    min_folds
    function FoldRiskScorer(r::AbstractBaseRiskMeasure, cv::CVE_Onl,
                            agg::AbstractFoldAggregation, min_folds::Integer)
        assert_scoreable(r)
        @argcheck(min_folds >= 1,
                  DomainError(min_folds,
                              "min_folds must be at least one, got min_folds = $min_folds"))
        return new{typeof(r), typeof(cv), typeof(agg), typeof(min_folds)}(r, cv, agg,
                                                                          min_folds)
    end
end
function FoldRiskScorer(; r::AbstractBaseRiskMeasure, cv::CVE_Onl = KFold(),
                        agg::AbstractFoldAggregation = MeanFoldScore(),
                        min_folds::Integer = 1)::FoldRiskScorer
    return FoldRiskScorer(r, cv, agg, min_folds)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the statistic of each covered asset on one test window, with the position of each asset in that window.

# Algorithm

 1. Find the positions `live` of the assets in the Coverage Universe of the rows `T` with [`cv_live_assets`](@ref).
 2. Evaluate the risk measure of `s` on the covered columns of the window with [`asset_scores`](@ref).
 3. Turn the values into statistics with [`fold_statistic`](@ref).
 4. Write the position of each covered asset in the statistics into `pos`, and zero at each other asset.

# Arguments

  - `s`: The fold risk scorer.
  - $(arg_dict[:rd])
  - `T`: The rows of the test window.
  - `bib`: The [`bigger_is_better`](@ref) flag of the risk measure of `s`.

# Returns

  - `(; pos, v)::NamedTuple`: `pos` is `assets × 1`, and holds the entry of `v` of each covered asset, or zero. `v` holds one statistic for each covered asset.

# Related

  - [`FoldRiskScorer`](@ref)
"""
function window_statistic(s::FoldRiskScorer, rd::AbstractReturnsResult, T::VecInt,
                          bib::Bool)
    live = cv_live_assets(port_opt_view(rd, T, :))
    v = fold_statistic(s.agg.alg, asset_scores(s.r, view(rd.X, T, live)), bib)
    pos = zeros(Int, size(rd.X, 2))
    pos[live] = eachindex(live)
    return (; pos = pos, v = v)
end
function asset_scores(s::FoldRiskScorer, rd::AbstractReturnsResult)
    bib = bigger_is_better(s.r)
    wins = [window_statistic(s, rd, T, bib)
            for T in distinct_test_windows(split(s.cv, rd).test_idx)]
    stats = [[w.v[w.pos[i]] for w in wins if !iszero(w.pos[i])] for i in axes(rd.X, 2)]
    sbib = statistic_bigger_is_better(s.agg.alg, bib)
    return min_folds_scores(i -> fold_aggregate(s.agg, stats[i], sbib), length.(stats),
                            s.min_folds, eltype(rd.X))
end
function bigger_is_better(s::FoldRiskScorer)::Bool
    return statistic_bigger_is_better(s.agg.alg, bigger_is_better(s.r))
end
"""
$(DocStringExtensions.TYPEDEF)

Scores each asset by the mean weight, or by the selection frequency, that an optimiser gives it over the splits of a cross-validation scheme.

The score reads the whole cross-section of each fold, so no risk measure on one asset column can state it. `opt` is an optimisation estimator or a [`Pipeline`](@ref). `p` is an optional grid of configurations, in the form that [`GridSearchCrossValidation`](@ref) takes, and the scorer pools the folds of every configuration.

# Mathematical definition

```math
\\begin{align}
m_{i} &= \\frac{1}{\\lvert \\mathcal{K}_{i} \\rvert} \\sum_{(c, k) \\in \\mathcal{K}_{i}} w^{(c, k)}_{i}\\,, \\\\
f_{i} &= \\frac{1}{\\lvert \\mathcal{K}_{i} \\rvert} \\sum_{(c, k) \\in \\mathcal{K}_{i}} \\mathbf{1}\\left\\{ \\left\\lvert w^{(c, k)}_{i} \\right\\rvert > \\tau \\right\\}\\,.
\\end{align}
```

Where:

  - ``m_{i}``: The mean weight of asset ``i``, the score when `tol` is `nothing`.
  - ``f_{i}``: The selection frequency of asset ``i``, the score when `tol` is a number.
  - ``\\mathcal{K}_{i}``: The pairs of a configuration ``c`` and a split ``k`` whose optimisation succeeded and whose split offered asset ``i``.
  - ``w^{(c, k)}_{i}``: The weight of asset ``i`` in the portfolio that configuration ``c`` fits on the training window of split ``k``.
  - ``\\tau``: The tolerance `tol`.

A split offers the assets of the Coverage Universe of its training window, and also every asset that the fold made investable. A [`MultipleRandomised`](@ref) split offers only the assets of the subset that it drew. An asset that a selector inside `opt` drops was offered, so it counts with weight zero. An asset outside the drawn subset was not offered, so it does not count. Without this rule, an asset that a random draw seldom picks gets a low score by chance.

The scorer reads the result of each split once. Under [`CombinatorialCrossValidation`](@ref), [`cross_val_predict`](@ref) gives a population of paths, and one split is in many paths. A failed split counts in neither the sum nor the count. The mean weight is signed, so a short position lowers it, and the frequency reads the absolute weight. A bigger score is better for both.

!!! warning

    The scorer runs ``C K`` optimisations, for ``C`` configurations and ``K`` splits. Inside a [`Pipeline`](@ref) under [`search_cross_validation`](@ref), the cost multiplies again by the outer folds and the outer candidates. The scorer keeps no cache.

!!! warning

    An asset that fewer than `min_folds` successful splits offered has the score `NaN`, and every selection rule drops it. Under [`MultipleRandomised`](@ref), a draw of asset subsets can miss an asset completely. A selector then drops that asset only because the random draw did not offer it, and not because of its weight. More subsets in the draw make this less probable.

# Algorithm

 1. Fix the random draw of `cv` with [`pin_draw`](@ref), so that the split and the predictions use the same splits.
 2. Split `rd` once, and find the assets that each split offers with [`split_offered`](@ref).
 3. Build the candidates of the grid with [`weight_mass_candidates`](@ref).
 4. Run [`cross_val_predict`](@ref) on each candidate, and find the prediction of each split with [`split_prediction`](@ref).
 5. Keep each prediction whose optimisation succeeded, and turn it into the record of the split with [`split_weight_record`](@ref).
 6. Count the records that offered each asset.
 7. Return the mean of the record values of each asset that at least `min_folds` records offered, over those records, and `NaN` at every other asset. [`min_folds_scores`](@ref) does this step.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WeightMassScorer(;
        opt::Union{OptE_TD, Pipeline},
        p::Option{<:Union{MultiGSCVValType, AbstractVector}} = nothing,
        cv::CVE_Onl = KFold(),
        tol::Option{<:Real} = nothing,
        min_folds::Integer = 1,
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
    ) -> WeightMassScorer

Keywords correspond to the struct's fields. `p` takes a vector of parameter sets whose types differ, through [`narrow_mixed_vector`](@ref), as [`GridSearchCrossValidation`](@ref) does.

## Validation

  - If `tol` is a number, `tol >= 0`.
  - `min_folds >= 1`.
  - Every candidate of `p` passes the checks of the search: the optimiser checks for an estimator, and the hold-out check for a [`Pipeline`](@ref).

$(val_dict[:count_fraction_grid])

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = [\"A\", \"B\", \"C\"],
                          X = [0.1 0.0 -0.2; -0.1 0.1 0.3; 0.2 0.0 -0.1; 0.0 -0.1 0.2]);

julia> s = WeightMassScorer(; opt = InverseVolatility(), cv = KFold(; n = 2));

julia> round.(PortfolioOptimisers.asset_scores(s, rd); digits = 4)
3-element Vector{Float64}:
 0.2834
 0.5668
 0.1497
```

# Related

  - [`AbstractAssetScorer`](@ref)
  - [`FoldRiskScorer`](@ref)
  - [`ScoreSelector`](@ref)
  - [`cross_val_predict`](@ref)
  - [`GridSearchCrossValidation`](@ref)
"""
@concrete struct WeightMassScorer <: AbstractAssetScorer
    """
    The optimisation estimator or the [`Pipeline`](@ref) whose weights score the assets.
    """
    opt
    """
    The grid of configurations of `opt`, in the form that [`GridSearchCrossValidation`](@ref) takes. `nothing` scores `opt` alone.
    """
    p
    """
    $(field_dict[:cv])
    """
    cv
    """
    The tolerance of the selection frequency. A number scores the share of the splits in which the absolute weight of the asset exceeds it, and `nothing` scores the mean weight.
    """
    tol
    """
    The smallest number of successful splits that must offer an asset. An asset that fewer splits offered has the score `NaN`.
    """
    min_folds
    """
    $(field_dict[:ex])
    """
    ex
    function WeightMassScorer(opt::Union{OptE_TD, Pipeline},
                              p::Option{<:MultiGSCVValType_VecMultiGSCVValType},
                              cv::CVE_Onl, tol::Option{<:Real}, min_folds::Integer,
                              ex::FLoops.Transducers.Executor)
        if !isnothing(tol)
            @argcheck(tol >= zero(tol),
                      DomainError(tol, "tol must be non-negative, got tol = $tol"))
        end
        @argcheck(min_folds >= 1,
                  DomainError(min_folds,
                              "min_folds must be at least one, got min_folds = $min_folds"))
        if !isnothing(p)
            assert_search_candidates(opt, weight_mass_grid(opt, p)...)
        end
        return new{typeof(opt), typeof(p), typeof(cv), typeof(tol), typeof(min_folds),
                   typeof(ex)}(opt, p, cv, tol, min_folds, ex)
    end
end
function WeightMassScorer(; opt::Union{OptE_TD, Pipeline},
                          p::Option{<:Union{MultiGSCVValType, AbstractVector}} = nothing,
                          cv::CVE_Onl = KFold(), tol::Option{<:Real} = nothing,
                          min_folds::Integer = 1,
                          ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())::WeightMassScorer
    return WeightMassScorer(opt,
                            narrow_mixed_vector(p, Union{PairGSCV, MultiGSCVValType}, :p),
                            cv, tol, min_folds, ex)
end
"""
    weight_mass_grid(opt, p) -> Tuple
    weight_mass_grid(opt::Pipeline, p) -> Tuple

Expand the grid `p` of a [`WeightMassScorer`](@ref) into its lenses and its values.

The method on an estimator calls [`lens_val_grid`](@ref), and the method on a [`Pipeline`](@ref) calls [`pipeline_lens_val_grid`](@ref). [`search_cross_validation`](@ref) calls the same two functions, so the scorer and the search read one grid syntax.

# Arguments

  - `opt`: The optimisation estimator or the pipeline.
  - `p`: The grid of configurations.

# Returns

  - `(lenses, vals)::Tuple`: One vector of lenses and one tuple of values for each candidate.

# Related

  - [`weight_mass_candidates`](@ref)
"""
function weight_mass_grid(::Any, p)
    return lens_val_grid(p)
end
function weight_mass_grid(opt::Pipeline, p)
    return pipeline_lens_val_grid(opt, p)
end
"""
    weight_mass_candidates(opt::Union{OptE_TD, Pipeline}, p::Nothing) -> AbstractVector
    weight_mass_candidates(opt::Union{OptE_TD, Pipeline}, p::MultiGSCVValType_VecMultiGSCVValType) -> AbstractVector

Return the candidates that a [`WeightMassScorer`](@ref) runs, one for each point of its grid.

With no grid the candidate is `opt` alone. Otherwise the function sets each lens of a point to its value with [`search_candidate`](@ref).

# Arguments

  - `opt`: The optimisation estimator or the pipeline of the scorer.
  - `p`: The grid of configurations, or `nothing`.

# Returns

  - `cands::AbstractVector`: The candidates, in the order of the grid.

# Related

  - [`weight_mass_grid`](@ref)
"""
function weight_mass_candidates(opt::Union{OptE_TD, Pipeline}, ::Nothing)
    return [opt]
end
function weight_mass_candidates(opt::Union{OptE_TD, Pipeline},
                                p::MultiGSCVValType_VecMultiGSCVValType)
    lens_grid, val_grid = weight_mass_grid(opt, p)
    return map((lenses, vals) -> search_candidate(opt, lenses, vals), lens_grid, val_grid)
end
"""
    split_assets(cvr::MultipleRandomisedResult, k::Integer, N::Integer) -> VecInt
    split_assets(cvr::CrossValidationResult, k::Integer, N::Integer) -> VecInt

Return the asset positions that split `k` gives its optimiser.

A [`MultipleRandomised`](@ref) split gives the subset that it drew. Every other split gives all `N` assets.

# Arguments

  - `cvr`: The split result.
  - `k`: The index of the split.
  - `N`: The number of assets of the returns data.

# Returns

  - `cols::VecInt`: The asset positions of the split.

# Related

  - [`split_offered`](@ref)
"""
function split_assets(cvr::MultipleRandomisedResult, k::Integer, ::Integer)
    return cvr.asset_idx[k]
end
function split_assets(::CrossValidationResult, ::Integer, N::Integer)
    return collect(1:N)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the mask of the assets that split `k` offers its optimiser.

# Algorithm

 1. Read the asset positions `cols` of the split with [`split_assets`](@ref).
 2. Find the positions of `cols` in the Coverage Universe of the training window with [`cv_live_assets`](@ref).
 3. Return the mask that is `true` at those positions.

# Arguments

  - $(arg_dict[:rd])
  - `cvr`: The split result.
  - `k`: The index of the split.

# Returns

  - `offered::BitVector`: The mask `assets × 1`, `true` at each asset that the split offers.

# Related

  - [`WeightMassScorer`](@ref)
  - [`split_weight_record`](@ref)
"""
function split_offered(rd::AbstractReturnsResult, cvr::CrossValidationResult, k::Integer)
    N = size(rd.X, 2)
    cols = split_assets(cvr, k, N)
    live = cv_live_assets(port_opt_view(rd, cvr.train_idx[k], cols))
    offered = falses(N)
    offered[cols[live]] .= true
    return offered
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the prediction of `mp` whose test window starts at row `t`.

# Arguments

  - `mp`: One path of predictions.
  - `t`: The first row of the test window.

# Validation

  - One prediction of `mp` starts at `t`. Otherwise the function throws an `ArgumentError`, because the split and the predictions disagree.

# Returns

  - `pred::PredictionResult`: The prediction of the window.

# Related

  - [`split_prediction`](@ref)
"""
function prediction_at(mp::MultiPeriodPredictionResult, t::Integer)
    for pred in mp.pred
        if first(pred.idx) == t
            return pred
        end
    end
    return throw(ArgumentError("no prediction starts at row $t, so the split and the predictions describe different folds"))
end
"""
    split_prediction(pred::MultiPeriodPredictionResult, cvr::CrossValidationResult, k::Integer) -> PredictionResult
    split_prediction(pred::PopulationPredictionResult, cvr::CombinatorialCrossValidationResult, k::Integer) -> PredictionResult
    split_prediction(pred::PopulationPredictionResult, cvr::MultipleRandomisedResult, k::Integer) -> PredictionResult

Return the prediction that holds the result of split `k`, once for each split.

A sequential or a K-fold scheme gives one prediction for each split. A combinatorial split is in many paths, and every block of the split holds the same result, so the method reads the first block of the split in its path. A randomised split is in one path, and the method reads it there. Each method finds the prediction by the first row of its test window, with [`prediction_at`](@ref).

# Arguments

  - `pred`: The output of [`cross_val_predict`](@ref) over the splits `cvr`.
  - `cvr`: The split result.
  - `k`: The index of the split.

# Returns

  - `p::PredictionResult`: The prediction that holds the result of split `k`.

# Related

  - [`WeightMassScorer`](@ref)
"""
function split_prediction(pred::MultiPeriodPredictionResult, cvr::CrossValidationResult,
                          k::Integer)
    return prediction_at(pred, first(cvr.test_idx[k]))
end
function split_prediction(pred::PopulationPredictionResult,
                          cvr::CombinatorialCrossValidationResult, k::Integer)
    return prediction_at(pred.pred[cvr.path_ids[1, k]], first(first(cvr.test_idx[k])))
end
function split_prediction(pred::PopulationPredictionResult, cvr::MultipleRandomisedResult,
                          k::Integer)
    return prediction_at(pred.pred[cvr.path_ids[k]], first(cvr.test_idx[k]))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the record of one successful split: the assets that it offered, and the value of each.

The weights of a fold are on the universe that its optimiser saw, which a selector or an asset subset can make smaller than `rd`. The function maps them to the assets of `rd` by name, through the names of the investable assets of the prediction.

# Algorithm

 1. Read the weights of the investable assets into `wi`: `w` at the Investable Mask from [`result_investable_mask`](@ref), or all of `w` when the mask is `nothing`.
 2. Write each weight at the position of its asset name in `v`, zero at every other asset, and add each named asset to `offered`.
 3. Turn `v` into the value of each asset with [`weight_mass_value`](@ref).

# Arguments

  - `p`: The prediction of the split.
  - `offered`: The mask of the assets that the split offers, from [`split_offered`](@ref).
  - `idx`: The position of each asset name of `rd`.
  - `tol`: The tolerance of the selection frequency, or `nothing` for the weight.

# Returns

  - `(; offered, v)::NamedTuple`: `offered` is the mask `assets × 1` of the offered assets, and `v` holds the value of each asset.

# Related

  - [`WeightMassScorer`](@ref)
"""
function split_weight_record(p::PredictionResult, offered::BitVector, idx::AbstractDict,
                             tol::Option{<:Real})
    w = p.res.w
    imsk = result_investable_mask(p.res)
    wi = isnothing(imsk) ? w : w[imsk]
    v = zeros(eltype(wi), length(offered))
    offered = copy(offered)
    for (j, n) in enumerate(p.rd.nx)
        v[idx[n]] = wi[j]
        offered[idx[n]] = true
    end
    return (; offered = offered, v = weight_mass_value(tol, v))
end
"""
    weight_mass_value(tol::Nothing, v::VecNum) -> VecNum
    weight_mass_value(tol::Real, v::VecNum) -> VecNum

Return the value that a [`WeightMassScorer`](@ref) averages for each asset of one split.

With `tol = nothing` the value is the weight itself. With a number it is one where the absolute weight exceeds `tol`, and zero otherwise, in the number type of the weights.

# Arguments

  - `tol`: The tolerance of the selection frequency, or `nothing`.
  - `v`: The weights of the split, `assets × 1`.

# Returns

  - `v::VecNum`: The value of each asset.

# Related

  - [`split_weight_record`](@ref)
"""
function weight_mass_value(::Nothing, v::VecNum)
    return v
end
function weight_mass_value(tol::Real, v::VecNum)
    return [ifelse(abs(x) > tol, one(x), zero(x)) for x in v]
end
function asset_scores(s::WeightMassScorer, rd::AbstractReturnsResult)
    cv = pin_draw(s.cv)
    cvr = split(cv, rd)
    ks = eachindex(cvr.train_idx)
    offered = [split_offered(rd, cvr, k) for k in ks]
    idx = Dict(n => i for (i, n) in enumerate(rd.nx))
    preds = [cross_val_predict(opt, rd, cv; ex = s.ex)
             for opt in weight_mass_candidates(s.opt, s.p)]
    splits = [(split_prediction(pred, cvr, k), k) for pred in preds for k in ks]
    recs = [split_weight_record(p, offered[k], idx, s.tol)
            for (p, k) in splits if fold_solved(p.res.retcode)]
    counts = [count(r -> r.offered[i], recs) for i in axes(rd.X, 2)]
    return min_folds_scores(i -> sum(r.v[i] for r in recs if r.offered[i]) / counts[i],
                            counts, s.min_folds, eltype(rd.X))
end
function bigger_is_better(::WeightMassScorer)::Bool
    return true
end
"""
$(DocStringExtensions.TYPEDEF)

Scores each asset by a statistic of its Panel Fields, or of a Descriptor, over the observations of the returns data.

The score reads the Asset Panel `rd.pnl`, not the returns. `fields` gives the value of an asset at each observation, in one of four forms:

 1. A name reads that one Panel Field. With a market capitalisation, a [`ThresholdRule`](@ref) with a bound `lo` is a floor of size.
 2. A vector of `name => coefficient` pairs reads their sum. `[\"a\" => 1, \"b\" => -1]` reads `a - b`.
 3. A vector of names reads their product. With a price and a traded volume, the value is the traded amount, and a [`ThresholdRule`](@ref) on its median is a liquidity floor.
 4. A Descriptor Estimator, such as an [`EWVolumeRatio`](@ref), reads its Descriptor through [`descriptor`](@ref). A smoothed turnover or an illiquidity then becomes a score.

The first three forms are a [`PanelTermType`](@ref). `stat` then reduces the values of the asset over the observations to its score.

Inside a [`ScoreSelector`](@ref), `rd` is the training window, and its panel holds the observations of that window only. So the score moves with the window, and no observation after the window reaches it. A Descriptor also reads the window alone, so its recursion starts again at the first observation of each window.

# Mathematical definition

```math
\\begin{align}
v_{t i} &= \\begin{cases} z^{(1)}_{t i} & \\text{one name} \\\\ \\sum_{k = 1}^{K} c_{k} z^{(k)}_{t i} & \\text{a combination} \\\\ \\prod_{k = 1}^{K} z^{(k)}_{t i} & \\text{a product} \\\\ d_{t i} & \\text{a Descriptor} \\end{cases}\\,, \\\\
s_{i} &= S\\left(\\left\\{ v_{t i} : t \\in \\mathcal{O}_{i} \\right\\}\\right)\\,.
\\end{align}
```

Where:

  - ``v_{t i}``: The value of asset ``i`` at observation ``t``.
  - ``z^{(k)}_{t i}``: The value of the ``k``-th Panel Field of `fields` for asset ``i`` at observation ``t``.
  - ``c_{k}``: The coefficient of the ``k``-th Panel Field of a combination.
  - ``K``: The number of Panel Fields in `fields`.
  - ``d_{t i}``: The Descriptor of `fields` for asset ``i`` at observation ``t``.
  - ``\\mathcal{O}_{i}``: The observations at which ``v_{t i}`` is finite.
  - ``S``: The statistic `stat`.
  - ``s_{i}``: The score of asset ``i``.

A value is `NaN` where the asset is outside the universe, and where a fill policy of the panel wrote one of its Panel Fields. A filled volume is not a trade, so the statistic reads only the observed values. A Descriptor is also `NaN` before its warm-up ends. An asset with fewer than `min_obs` finite values has the score `NaN`, and every selection rule and every redundancy algorithm drops it. An asset with no observed value then leaves the selection, and the run does not stop.

`stat` reads the finite values of one asset in the order of the observations, through [`finite_vec_to_real_measure`](@ref). A measure that holds observation weights, such as `MedianValue(; w)`, holds one weight for each observation of `rd`, and the function cuts them to the finite values, so each value keeps its own weight.

With `bib = nothing`, the orientation of the score is the orientation of its statistic, [`bigger_is_better`](@ref) of `stat`. A statistic of location, such as [`MedianValue`](@ref), answers `true`, so a bigger traded amount is a better score. A statistic of dispersion, [`StdValue`](@ref) or [`VarValue`](@ref), answers `false`, as a risk does. A plain function answers `true`. The orientation of a Panel Field or of a Descriptor can differ from that of the statistic: a bigger median leverage is worse, and so is a bigger median illiquidity. A `Bool` in `bib` then overrides the statistic.

# Algorithm

 1. Read the value of each asset at each observation with [`descriptor_field_values`](@ref). A Descriptor Estimator gives its [`descriptor`](@ref).
 2. Count the finite values of each asset.
 3. Reduce the finite values of each asset that holds at least `min_obs` of them with [`finite_vec_to_real_measure`](@ref) under `stat`. Write `NaN` at every other asset. [`min_folds_scores`](@ref) does this step.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PanelFieldScorer(;
        fields::Union{PanelTermType, AbstractDescriptorEstimator},
        stat::Union{VectorToScalarMeasure, Function} = MedianValue(),
        bib::Option{<:Bool} = nothing,
        min_obs::Integer = 1,
    ) -> PanelFieldScorer

Keywords correspond to the struct's fields.

## Validation

  - The rules of [`assert_panel_terms`](@ref) for a [`PanelTermType`](@ref) in `fields`: a name is not the empty string, a combination or a product holds at least one name, no name in it is empty, and each coefficient is finite. Raises an [`IsEmptyError`](@ref) or a `DomainError`. A Descriptor Estimator checks its own fields in its constructor.
  - `min_obs >= 1`. Raises a `DomainError`.
  - When the scorer scores, the rules of [`descriptor_field_values`](@ref) for each Panel Field of `fields`: `rd.pnl` is a time-varying [`AssetPanel`](@ref), and each name is a [`NumericPanelField`](@ref) of it. A Descriptor Estimator raises the errors of its [`descriptor`](@ref).

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"close\",
                                            vals = [10.0 2.0; 11.0 2.0; 12.0 NaN],
                                            alg = ForwardPanelFill()),
                          NumericPanelInput(; name = \"volume\",
                                            vals = [100.0 50.0; 300.0 70.0; 200.0 60.0])];
                         amsk = trues(3, 2), emsk = trues(3, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(3, 2), pnl = pnl);

julia> s = PanelFieldScorer(; fields = [\"close\", \"volume\"]);

julia> PortfolioOptimisers.asset_scores(s, rd)
2-element Vector{Float64}:
 2400.0
  120.0

julia> PortfolioOptimisers.bigger_is_better(s)
true

julia> PortfolioOptimisers.asset_scores(PanelFieldScorer(; fields = \"volume\"), rd)
2-element Vector{Float64}:
 200.0
  60.0
```

The close of `B` at the third observation is a fill, so the median traded amount of `B` reads its first two traded amounts, `100` and `140`. The median volume of `B` reads all three volumes.

# Related

  - [`AbstractAssetScorer`](@ref)
  - [`ScoreSelector`](@ref)
  - [`ThresholdRule`](@ref)
  - [`VectorToScalarMeasure`](@ref)
  - [`AssetPanel`](@ref)
  - [`PanelTermType`](@ref)
  - [`AbstractDescriptorEstimator`](@ref)
  - [`EWVolumeRatio`](@ref)
  - [`FoldRiskScorer`](@ref)
"""
@concrete struct PanelFieldScorer <: AbstractAssetScorer
    """
    The value of an asset at each observation: a [`PanelTermType`](@ref), which names one Panel Field, a sum or a product of Panel Fields, or a Descriptor Estimator, whose Descriptor is the value.
    """
    fields
    """
    The [`VectorToScalarMeasure`](@ref) or the function that reduces the finite values of one asset over the observations to its score.
    """
    stat
    """
    The orientation of the score: `true` when a bigger score is better. `nothing` takes the [`bigger_is_better`](@ref) of `stat`.
    """
    bib
    """
    The smallest number of finite values that an asset must hold. An asset that holds fewer has the score `NaN`.
    """
    min_obs
    function PanelFieldScorer(fields::Union{PanelTermType, AbstractDescriptorEstimator},
                              stat::Union{VectorToScalarMeasure, Function},
                              bib::Option{<:Bool}, min_obs::Integer)
        assert_panel_terms(fields, :fields)
        @argcheck(min_obs >= 1,
                  DomainError(min_obs,
                              "min_obs must be at least one, got min_obs = $min_obs"))
        return new{typeof(fields), typeof(stat), typeof(bib), typeof(min_obs)}(fields, stat,
                                                                               bib, min_obs)
    end
end
function PanelFieldScorer(; fields::Union{PanelTermType, AbstractDescriptorEstimator},
                          stat::Union{VectorToScalarMeasure, Function} = MedianValue(),
                          bib::Option{<:Bool} = nothing,
                          min_obs::Integer = 1)::PanelFieldScorer
    return PanelFieldScorer(fields, stat, bib, min_obs)
end
function asset_scores(s::PanelFieldScorer, rd::AbstractReturnsResult)
    V = descriptor_field_values(rd, s.fields)
    counts = [count(isfinite, view(V, :, i)) for i in axes(V, 2)]
    return min_folds_scores(i -> finite_vec_to_real_measure(s.stat, view(V, :, i)), counts,
                            s.min_obs, eltype(V))
end
function bigger_is_better(s::PanelFieldScorer)::Bool
    return something(s.bib, bigger_is_better(s.stat))
end
"""
    bigger_is_better(stat::VectorToScalarMeasure) -> Bool
    bigger_is_better(stat::Union{StdValue, VarValue}) -> Bool
    bigger_is_better(stat::Function) -> Bool

Return whether a bigger value of the statistic `stat` is a better score.

A [`PanelFieldScorer`](@ref) with `bib = nothing` takes the orientation of its score from its statistic. A statistic of location or of size, such as the minimum, the mean, the median, the maximum, the sum or the product, answers `true`: a bigger traded amount or a bigger capitalisation is better. A statistic of dispersion, [`StdValue`](@ref) or [`VarValue`](@ref), answers `false`, because a dispersion is a risk, as a variance is for a risk measure. A plain function has no type that can state a trait, and it answers `true`.

A subtype of [`VectorToScalarMeasure`](@ref) whose smaller values are better adds a method that answers `false`.

# Arguments

  - `stat`: The statistic.

# Returns

  - `bib::Bool`: `true` when a bigger value of `stat` is better.

# Related

  - [`PanelFieldScorer`](@ref)
  - [`VectorToScalarMeasure`](@ref)
"""
function bigger_is_better(::VectorToScalarMeasure)::Bool
    return true
end
function bigger_is_better(::Union{StdValue, VarValue})::Bool
    return false
end
function bigger_is_better(::Function)::Bool
    return true
end

export FoldRiskScorer, WeightMassScorer, PanelFieldScorer, MeanFoldScore, MedianFoldScore,
       WorstFoldScore, Value, Rank
public AbstractFoldStatistic, AbstractFoldAggregation, fold_statistic,
       statistic_bigger_is_better, fold_aggregate
