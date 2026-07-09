"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that turn per-asset scores into a keep-mask.

A selection rule is an [`AbstractAlgorithm`](@ref): it is consumed through [`ScoreSelector`](@ref) and never used on its own. Rules split into two kinds.

  - **Literal**: [`ThresholdRule`](@ref) compares a score against absolute bounds and ignores [`bigger_is_better`](@ref). A threshold on a variance means what it says; reinterpreting it as "keep the better ones" would invert the intent of a zero-variance filter.
  - **Ordinal**: [`RankRule`](@ref) and [`QuantileRule`](@ref) sort assets from best to worst — consulting [`bigger_is_better`](@ref), so `:best` is lowest risk for a risk measure and highest value for a return measure — and take counts or fractions from each tail.

# Related

  - [`ScoreSelector`](@ref)
  - [`ThresholdRule`](@ref)
  - [`RankRule`](@ref)
  - [`QuantileRule`](@ref)
"""
abstract type AbstractSelectionRule <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Keep assets whose score falls strictly inside the band `(lo, hi)`.

Both bounds are optional and **literal**: `lo` and `hi` are compared against the raw score, never reinterpreted through [`bigger_is_better`](@ref). Omitting a bound leaves that side unbounded.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ThresholdRule(;
        lo::Option{<:Number} = nothing,
        hi::Option{<:Number} = nothing,
    ) -> ThresholdRule

Keywords correspond to the struct's fields.

## Validation

  - At least one of `lo`, `hi` is not `nothing`.
  - If both are given, `lo < hi`.

# Examples

```jldoctest
julia> ThresholdRule(; lo = 1e-12)   # drop (near-)constant assets
ThresholdRule
  lo ┼ Float64: 1.0e-12
  hi ┴ nothing
```

# Related

  - [`AbstractSelectionRule`](@ref)
  - [`ScoreSelector`](@ref)
"""
@concrete struct ThresholdRule <: AbstractSelectionRule
    """
    Exclusive lower bound on the score; `nothing` leaves the lower side unbounded.
    """
    lo
    """
    Exclusive upper bound on the score; `nothing` leaves the upper side unbounded.
    """
    hi
    function ThresholdRule(lo::Option{<:Number}, hi::Option{<:Number})
        @argcheck(!(isnothing(lo) && isnothing(hi)),
                  IsNothingError("a ThresholdRule needs at least one of lo, hi"))
        if !isnothing(lo) && !isnothing(hi)
            @argcheck(lo < hi,
                      DomainError((lo, hi),
                                  "a ThresholdRule needs lo < hi, got lo = $lo, hi = $hi"))
        end
        return new{typeof(lo), typeof(hi)}(lo, hi)
    end
end
function ThresholdRule(; lo::Option{<:Number} = nothing,
                       hi::Option{<:Number} = nothing)::ThresholdRule
    return ThresholdRule(lo, hi)
end
"""
$(DocStringExtensions.TYPEDEF)

Take `best` and/or `worst` assets from the tails of the score ordering, then keep or drop them.

`best` and `worst` are **counts taken from each end**, not positions: `best = 20` means twenty assets, not rank twenty. Which end is "best" comes from [`bigger_is_better`](@ref) on the score, so `RankRule(; best = 20)` keeps the twenty lowest-risk assets for a risk measure and the twenty highest-return assets for `MeanReturn`. Giving both takes both tails. `action = :drop` complements the whole selection, which is how "drop the five worst" is said without knowing the universe size.

Counts **saturate** at the number of assets: `best = 50` on a 30-asset window keeps all 30 rather than throwing, so a hyperparameter search over `best` is never killed by its largest point.

!!! warning

    Ties at the cut are **excluded entirely**, so a rule may return *fewer* assets than asked. If the 20th and 21st assets have equal scores, `RankRule(; best = 20)` keeps 19 — the tied block is dropped rather than split arbitrarily. This is the library's "if we cannot tell them apart, trust neither" tie policy, shared with `find_uncorrelated_indices`, which removes both assets of an exactly-tied correlated pair. A window whose scores are all equal therefore selects nothing, and [`fit_preprocessing`](@ref) throws.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RankRule(;
        best::Option{<:Integer} = nothing,
        worst::Option{<:Integer} = nothing,
        action::Symbol = :keep,
    ) -> RankRule

Keywords correspond to the struct's fields.

## Validation

  - At least one of `best`, `worst` is not `nothing`.
  - Any given count is `>= 0`, and at least one is `> 0`.
  - `action in (:keep, :drop)`.

# Examples

```jldoctest
julia> RankRule(; worst = 5, action = :drop)   # drop the five worst
RankRule
    best ┼ nothing
   worst ┼ Int64: 5
  action ┴ Symbol: :drop
```

# Related

  - [`AbstractSelectionRule`](@ref)
  - [`QuantileRule`](@ref)
  - [`bigger_is_better`](@ref)
"""
@concrete struct RankRule <: AbstractSelectionRule
    """
    Number of assets to take from the best end; `nothing` takes none.
    """
    best
    """
    Number of assets to take from the worst end; `nothing` takes none.
    """
    worst
    """
    `:keep` retains the taken assets, `:drop` retains everything else.
    """
    action
    function RankRule(best::Option{<:Integer}, worst::Option{<:Integer}, action::Symbol)
        assert_tail_counts(best, worst, :RankRule)
        assert_selection_action(action)
        return new{typeof(best), typeof(worst), typeof(action)}(best, worst, action)
    end
end
function RankRule(; best::Option{<:Integer} = nothing, worst::Option{<:Integer} = nothing,
                  action::Symbol = :keep)::RankRule
    return RankRule(best, worst, action)
end
"""
$(DocStringExtensions.TYPEDEF)

[`RankRule`](@ref) with the tail sizes given as *fractions* of the asset universe.

`best` and `worst` are fractions in `(0, 1)`, converted to counts as `round(Int, fraction * n)` on the window being fitted. Everything else — orientation via [`bigger_is_better`](@ref), the `action` complement, count saturation, and the tie policy that excludes a straddling tied block — is identical to [`RankRule`](@ref).

Fractions and counts are separate types on purpose: `best = 1` (one asset) and `best = 1.0` (the whole universe) would otherwise differ only by a literal's type.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    QuantileRule(;
        best::Option{<:Real} = nothing,
        worst::Option{<:Real} = nothing,
        action::Symbol = :keep,
    ) -> QuantileRule

Keywords correspond to the struct's fields.

## Validation

  - At least one of `best`, `worst` is not `nothing`.
  - Any given fraction lies in `(0, 1)`.
  - `action in (:keep, :drop)`.

# Examples

```jldoctest
julia> QuantileRule(; worst = 0.1, action = :drop)   # drop the worst decile
QuantileRule
    best ┼ nothing
   worst ┼ Float64: 0.1
  action ┴ Symbol: :drop
```

# Related

  - [`AbstractSelectionRule`](@ref)
  - [`RankRule`](@ref)
"""
@concrete struct QuantileRule <: AbstractSelectionRule
    """
    Fraction of the universe to take from the best end; `nothing` takes none.
    """
    best
    """
    Fraction of the universe to take from the worst end; `nothing` takes none.
    """
    worst
    """
    `:keep` retains the taken assets, `:drop` retains everything else.
    """
    action
    function QuantileRule(best::Option{<:Real}, worst::Option{<:Real}, action::Symbol)
        @argcheck(!(isnothing(best) && isnothing(worst)),
                  IsNothingError("a QuantileRule needs at least one of best, worst"))
        for (v, s) in ((best, :best), (worst, :worst))
            if !isnothing(v)
                @argcheck(zero(v) < v < one(v),
                          DomainError(v,
                                      "the $s fraction of a QuantileRule must lie in (0, 1)"))
            end
        end
        assert_selection_action(action)
        return new{typeof(best), typeof(worst), typeof(action)}(best, worst, action)
    end
end
function QuantileRule(; best::Option{<:Real} = nothing, worst::Option{<:Real} = nothing,
                      action::Symbol = :keep)::QuantileRule
    return QuantileRule(best, worst, action)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate the `action` field shared by the ordinal selection rules.

# Related

  - [`RankRule`](@ref)
  - [`QuantileRule`](@ref)
"""
function assert_selection_action(action::Symbol)::Nothing
    @argcheck(action in (:keep, :drop),
              ArgumentError("the action of a selection rule must be :keep or :drop, got $(repr(action))"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate the `best`/`worst` tail sizes shared by the ordinal selection rules.

# Related

  - [`RankRule`](@ref)
  - [`QuantileRule`](@ref)
"""
function assert_tail_counts(best::Option{<:Integer}, worst::Option{<:Integer},
                            name::Symbol)::Nothing
    @argcheck(!(isnothing(best) && isnothing(worst)),
              IsNothingError("a $name needs at least one of best, worst"))
    for (v, s) in ((best, :best), (worst, :worst))
        if !isnothing(v)
            @argcheck(v >= zero(v), DomainError(v, "the $s count of a $name must be >= 0"))
        end
    end
    @argcheck(max(isnothing(best) ? 0 : best, isnothing(worst) ? 0 : worst) > 0,
              DomainError((best, worst), "a $name must take at least one asset"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the mask of the `k` assets furthest into the `tail` end of the score ordering.

`tail` is `:best` or `:worst`; which raw direction that is comes from `bib`, the [`bigger_is_better`](@ref) flag of the score. The count saturates at `length(scores)`.

Assets tied with the `k`-th score are included only when the whole tied block fits within `k`. Otherwise the block straddles the cut and is excluded — the "trust neither" tie policy, which is why the returned mask may hold fewer than `k` assets.

# Related

  - [`RankRule`](@ref)
  - [`bigger_is_better`](@ref)
"""
function tail_mask(scores::VecNum, k::Integer, bib::Bool, tail::Symbol)::BitVector
    n = length(scores)
    if k <= zero(k)
        return falses(n)
    elseif k >= n
        return trues(n)
    end
    ahead = ((tail === :best) == bib) ? (>) : (<)
    perm = sortperm(scores; rev = (ahead === >))
    cut = scores[perm[k]]
    n_ahead = count(s -> ahead(s, cut), scores)
    n_eq = count(==(cut), scores)
    keep_ties = n_ahead + n_eq == k
    return BitVector(ahead(s, cut) || (keep_ties && s == cut) for s in scores)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Turn per-asset `scores` into a keep-mask under a selection rule.

`bib` is the [`bigger_is_better`](@ref) flag of the score that produced them; [`ThresholdRule`](@ref) ignores it.

# Related

  - [`AbstractSelectionRule`](@ref)
  - [`ScoreSelector`](@ref)
"""
function rule_keep(rule::ThresholdRule, scores::VecNum, ::Bool)::BitVector
    lo, hi = rule.lo, rule.hi
    return BitVector((isnothing(lo) || s > lo) && (isnothing(hi) || s < hi) for s in scores)
end
function rule_keep(rule::RankRule, scores::VecNum, bib::Bool)::BitVector
    return tail_action_mask(rule.best, rule.worst, rule.action, scores, bib)
end
function rule_keep(rule::QuantileRule, scores::VecNum, bib::Bool)::BitVector
    n = length(scores)
    to_count(f) = isnothing(f) ? nothing : round(Int, f * n, RoundNearestTiesUp)
    return tail_action_mask(to_count(rule.best), to_count(rule.worst), rule.action, scores,
                            bib)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Union the two tail masks and apply the rule's `action`.

# Related

  - [`tail_mask`](@ref)
  - [`rule_keep`](@ref)
"""
function tail_action_mask(best::Option{<:Integer}, worst::Option{<:Integer}, action::Symbol,
                          scores::VecNum, bib::Bool)::BitVector
    mask = falses(length(scores))
    if !isnothing(best)
        mask .|= tail_mask(scores, best, bib, :best)
    end
    if !isnothing(worst)
        mask .|= tail_mask(scores, worst, bib, :worst)
    end
    return action === :keep ? mask : .!mask
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate that `score` can be evaluated on a single asset's return series.

Scoring asset `i` is `score(X[:, i])`, which is exactly the precomputed-returns path [`supports_precomputed_returns`](@ref) governs. A [`WeightsInput`](@ref) measure — [`Variance`](@ref) and [`StandardDeviation`](@ref) among them — consumes portfolio weights instead, and cannot score an asset.

# Related

  - [`ScoreSelector`](@ref)
  - [`supports_precomputed_returns`](@ref)
"""
function assert_scoreable(score::AbstractBaseRiskMeasure)::Nothing
    if !supports_precomputed_returns(score)
        hint = if isa(score, Union{<:Variance, <:StandardDeviation})
            " `SCM()` computes the same quantity from a return series and is scoreable."
        else
            ""
        end
        throw(ArgumentError("`$(typeof(score))` cannot score a single asset's return series: its `supports_precomputed_returns` is false, so its functor consumes portfolio weights rather than a precomputed return vector.$hint"))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Evaluate `score` on every asset column of `X`.

Columns are passed as views: a risk-measure functor reads its argument and never writes to it.

## Validation

  - Every score is finite. A `NaN` score (a drawdown measure on a constant series, say) would make the ordering meaningless, so it throws rather than sorting arbitrarily.

# Related

  - [`ScoreSelector`](@ref)
  - [`assert_scoreable`](@ref)
"""
function asset_scores(score::AbstractBaseRiskMeasure, X::MatNum)
    scores = [score(view(X, :, i)) for i in axes(X, 2)]
    @argcheck(all(isfinite, scores),
              DomainError(scores,
                          "scoring the asset columns with a $(typeof(score)) produced non-finite values at columns $(findall(!isfinite, scores))"))
    return scores
end
"""
$(DocStringExtensions.TYPEDEF)

Asset selector that scores every asset with a risk measure and keeps the assets a rule admits.

`score` is any [`AbstractBaseRiskMeasure`](@ref) that can be evaluated on a bare return series — asset `i`'s score is `score(X[:, i])`. That reuses the whole risk-measure family: [`ConditionalValueatRisk`](@ref) and the drawdown measures score risk, `SCM()` scores variance, [`MeanReturn`](@ref) scores mean return. [`bigger_is_better`](@ref) tells the ordinal rules which end is "best".

`rule` decides what to do with the scores: an absolute band ([`ThresholdRule`](@ref)) or a count/fraction taken from the tails ([`RankRule`](@ref), [`QuantileRule`](@ref)).

The selected universe is fitted state, so a `ScoreSelector` is safe inside cross-validation: assets are chosen on the training window and the same universe is replayed on test windows.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ScoreSelector(;
        score::AbstractBaseRiskMeasure,
        rule::AbstractSelectionRule,
    ) -> ScoreSelector

Keywords correspond to the struct's fields.

## Validation

  - `supports_precomputed_returns(score)`. [`Variance`](@ref) and [`StandardDeviation`](@ref) are [`WeightsInput`](@ref) measures and are rejected with a pointer to `SCM()`.

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = ["A", "B", "C"], X = [0.1 0.0 -0.2; -0.1 0.0 0.3; 0.2 0.0 -0.1]);

julia> sel = ScoreSelector(; score = SCM(), rule = ThresholdRule(; lo = 1e-12));

julia> PortfolioOptimisers.fit_preprocessing(sel, rd).nx
2-element Vector{String}:
 "A"
 "C"
```

# Related

  - [`AbstractAssetSelector`](@ref)
  - [`AbstractSelectionRule`](@ref)
  - [`ZeroVarianceFilter`](@ref)
  - [`supports_precomputed_returns`](@ref)
"""
@concrete struct ScoreSelector <: AbstractAssetSelector
    """
    Risk measure scoring each asset's return series ([`AbstractBaseRiskMeasure`](@ref)).
    """
    score
    """
    Rule mapping the scores to a keep-mask ([`AbstractSelectionRule`](@ref)).
    """
    rule
    function ScoreSelector(score::AbstractBaseRiskMeasure, rule::AbstractSelectionRule)
        assert_scoreable(score)
        return new{typeof(score), typeof(rule)}(score, rule)
    end
end
function ScoreSelector(; score::AbstractBaseRiskMeasure,
                       rule::AbstractSelectionRule)::ScoreSelector
    return ScoreSelector(score, rule)
end
function select_assets(sel::ScoreSelector, rd::AbstractReturnsResult)::BitVector
    scores = asset_scores(sel.score, rd.X)
    return rule_keep(sel.rule, scores, bigger_is_better(sel.score))
end
"""
$(DocStringExtensions.TYPEDEF)

Asset selector that drops every asset column holding a `missing` or `NaN` observation.

The returns-level counterpart of [`MissingDataFilter`](@ref)'s column threshold, for pipelines fed returns data directly (where the price stages never run). It has no observation-dropping mode: a fitted selector cannot decide which rows of an unseen window to drop without breaking the weights/returns alignment.

# Constructors

    CompleteAssetSelector() -> CompleteAssetSelector

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = ["A", "B"], X = [0.1 0.2; 0.3 NaN]);

julia> PortfolioOptimisers.fit_preprocessing(CompleteAssetSelector(), rd).nx
1-element Vector{String}:
 "A"
```

# Related

  - [`AbstractAssetSelector`](@ref)
  - [`MissingDataFilter`](@ref)
"""
struct CompleteAssetSelector <: AbstractAssetSelector end
function select_assets(::CompleteAssetSelector, rd::AbstractReturnsResult)::BitVector
    keep = falses(size(rd.X, 2))
    keep[find_complete_indices(rd.X; dims = 1)] .= true
    return keep
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the algorithms that decide which assets a [`RedundancySelector`](@ref) discards as redundant.

Each algorithm answers the same question — *given the data and, optionally, a per-asset score, which columns survive?* — and returns a keep-mask. The keep-mask, not a partition into groups, is the seam: [`PairwiseCorrelation`](@ref) drops one asset at a time and may keep two members of the same correlated blob, which "partition, then keep the best of each group" cannot express.

Two algorithms do partition, and share [`groups_argbest`](@ref):

  - [`CorrelationComponents`](@ref) groups by connected component of the over-threshold correlation graph.
  - [`ClusterGroups`](@ref) groups by [`clusterise`](@ref) assignment.

# Interfaces

Concrete redundancy algorithms must implement:

  - `redundancy_keep(alg::MyAlgorithm, rd, scores, bib) -> BitVector`.
  - `requires_score(::MyAlgorithm) -> Bool`, if the algorithm cannot pick a survivor without one.

# Related

  - [`RedundancySelector`](@ref)
  - [`PairwiseCorrelation`](@ref)
  - [`CorrelationComponents`](@ref)
  - [`ClusterGroups`](@ref)
"""
abstract type AbstractRedundancyAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether a redundancy algorithm needs a `score` to pick the survivor of a redundancy group.

Correlation-based algorithms fall back on each asset's summary correlation to the rest of the universe when no score is given, so they return `false`. [`ClusterGroups`](@ref) has no such fallback and returns `true`.

# Related

  - [`AbstractRedundancyAlgorithm`](@ref)
  - [`RedundancySelector`](@ref)
"""
requires_score(::AbstractRedundancyAlgorithm) = true
"""
    redundancy_keep(alg::AbstractRedundancyAlgorithm, rd, scores, bib) -> BitVector

Return the keep-mask a redundancy algorithm admits.

`scores` is `nothing` when the [`RedundancySelector`](@ref) carries no `score`; otherwise it is the per-asset score vector and `bib` is the score's [`bigger_is_better`](@ref) flag.

# Related

  - [`AbstractRedundancyAlgorithm`](@ref)
  - [`RedundancySelector`](@ref)
"""
function redundancy_keep(alg::AbstractRedundancyAlgorithm, ::AbstractReturnsResult,
                         ::Option{<:VecNum}, ::Bool)
    return throw(ArgumentError("redundancy_keep is not implemented for $(typeof(alg))"))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Convert per-asset scores into *drop scores*, where higher means "discard me first".

A risk measure with `bigger_is_better == false` (lower risk is better) already reads as a drop score; one with `bigger_is_better == true` is negated.

# Related

  - [`RedundancySelector`](@ref)
  - [`bigger_is_better`](@ref)
"""
drop_scores(scores::VecNum, bib::Bool) = bib ? -scores : scores
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Keep the single best-scoring member of each redundancy group.

A group whose best score is *tied* keeps nobody: under the library's "if we cannot tell them apart, trust neither" policy, two indistinguishable assets are both discarded — the same stance [`find_uncorrelated_indices`](@ref) takes on an exactly-tied correlated pair. A singleton group is trivially unambiguous and always survives.

# Arguments

  - `groups`: Vector of index vectors partitioning the assets.
  - `scores`: Per-asset scores.
  - `bib`: Whether a larger score is better.

# Returns

  - `keep::BitVector`: One survivor per unambiguous group.

# Related

  - [`CorrelationComponents`](@ref)
  - [`ClusterGroups`](@ref)
"""
function groups_argbest(groups, scores::VecNum, bib::Bool)::BitVector
    keep = falses(length(scores))
    better = bib ? (>) : (<)
    for g in groups
        if isempty(g)
            continue
        end
        best = g[1]
        for i in view(g, 2:length(g))
            if better(scores[i], scores[best])
                best = i
            end
        end
        if count(i -> scores[i] == scores[best], g) == 1
            keep[best] = true
        end
    end
    return keep
end
"""
$(DocStringExtensions.TYPEDEF)

Greedy pairwise correlation pruning: drop assets until no surviving pair exceeds `thr`.

Correlated pairs are visited from most to least correlated, and the worse asset of each pair is removed. "Worse" means the higher drop score: the `RedundancySelector`'s `score` when it has one, otherwise each asset's summary correlation to the rest of the universe — so the asset that is redundant with *most* of the universe goes first.

This algorithm never **chains**. If `ρ(A, B) = 0.97` and `ρ(B, C) = 0.97` but `ρ(A, C) = 0.10`, it drops `B` and keeps both `A` and `C`, honouring the literal promise that no surviving pair exceeds `thr`. [`CorrelationComponents`](@ref) reads the same inputs transitively and keeps only one of the three.

Delegates to [`find_uncorrelated_indices`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PairwiseCorrelation(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        thr::Number = 0.95,
        absolute::Bool = false,
        measure::VectorToScalarMeasure = MeanValue(),
    ) -> PairwiseCorrelation

Keywords correspond to the struct's fields.

# Related

  - [`RedundancySelector`](@ref)
  - [`CorrelationComponents`](@ref)
  - [`find_uncorrelated_indices`](@ref)
"""
@concrete struct PairwiseCorrelation <: AbstractRedundancyAlgorithm
    """
    Covariance estimator supplying the correlation matrix.
    """
    ce
    """
    Correlation at or above which two assets are considered redundant.
    """
    thr
    """
    Whether to compare the absolute value of the correlation.
    """
    absolute
    """
    Reducer producing the fallback drop score from each column of the correlation matrix; ignored when the selector carries a `score`.
    """
    measure
    function PairwiseCorrelation(ce::StatsBase.CovarianceEstimator, thr::Number,
                                 absolute::Bool, measure::VectorToScalarMeasure)
        assert_correlation_threshold(thr)
        return new{typeof(ce), typeof(thr), typeof(absolute), typeof(measure)}(ce, thr,
                                                                               absolute,
                                                                               measure)
    end
end
function PairwiseCorrelation(;
                             ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                             thr::Number = 0.95, absolute::Bool = false,
                             measure::VectorToScalarMeasure = MeanValue())::PairwiseCorrelation
    return PairwiseCorrelation(ce, thr, absolute, measure)
end
requires_score(::PairwiseCorrelation) = false
function redundancy_keep(alg::PairwiseCorrelation, rd::AbstractReturnsResult,
                         scores::Option{<:VecNum}, bib::Bool)::BitVector
    keep = falses(size(rd.X, 2))
    idx = find_uncorrelated_indices(rd.X; ce = alg.ce, t = alg.thr, absolute = alg.absolute,
                                    measure = alg.measure,
                                    scores = if isnothing(scores)
                                        nothing
                                    else
                                        drop_scores(scores, bib)
                                    end)
    keep[idx] .= true
    return keep
end
"""
$(DocStringExtensions.TYPEDEF)

Group assets by connected component of the over-threshold correlation graph, and keep the best-scoring member of each.

Two assets share an edge when their (absolute) correlation is at or above `thr`. Components are transitive, so this reads a chain `A ~ B ~ C` as one redundant blob even when `A` and `C` are uncorrelated, and keeps a single asset from it. That is a stronger claim than [`PairwiseCorrelation`](@ref)'s, and a stronger reduction; choose it when you want one representative per correlated blob rather than a guarantee about surviving pairs.

A component whose best score is tied keeps nobody (see [`groups_argbest`](@ref)).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CorrelationComponents(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        thr::Number = 0.95,
        absolute::Bool = false,
        measure::VectorToScalarMeasure = MeanValue(),
    ) -> CorrelationComponents

Keywords correspond to the struct's fields.

# Related

  - [`RedundancySelector`](@ref)
  - [`PairwiseCorrelation`](@ref)
  - [`groups_argbest`](@ref)
"""
@concrete struct CorrelationComponents <: AbstractRedundancyAlgorithm
    """
    Covariance estimator supplying the correlation matrix.
    """
    ce
    """
    Correlation at or above which two assets share an edge.
    """
    thr
    """
    Whether to compare the absolute value of the correlation.
    """
    absolute
    """
    Reducer producing the fallback score from each column of the correlation matrix; ignored when the selector carries a `score`. Lower is better, so the surviving representative is the least redundant member of its component.
    """
    measure
    function CorrelationComponents(ce::StatsBase.CovarianceEstimator, thr::Number,
                                   absolute::Bool, measure::VectorToScalarMeasure)
        assert_correlation_threshold(thr)
        return new{typeof(ce), typeof(thr), typeof(absolute), typeof(measure)}(ce, thr,
                                                                               absolute,
                                                                               measure)
    end
end
function CorrelationComponents(;
                               ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                               thr::Number = 0.95, absolute::Bool = false,
                               measure::VectorToScalarMeasure = MeanValue())::CorrelationComponents
    return CorrelationComponents(ce, thr, absolute, measure)
end
requires_score(::CorrelationComponents) = false
function redundancy_keep(alg::CorrelationComponents, rd::AbstractReturnsResult,
                         scores::Option{<:VecNum}, bib::Bool)::BitVector
    rho = Statistics.cor(alg.ce, rd.X)
    if alg.absolute
        rho = abs.(rho)
    end
    s, sbib = if isnothing(scores)
        [vec_to_real_measure(alg.measure, x) for x in eachcol(rho)], false
    else
        scores, bib
    end
    return groups_argbest(correlation_components(rho, alg.thr), s, sbib)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the connected components of the graph whose edges are the pairs of `rho` at or above `thr`.

A union-find pass over the strict lower triangle, so components are transitive and every asset lands in exactly one of them (a singleton when it has no over-threshold partner).

# Returns

  - `groups::Vector{Vector{Int}}`: The components, each a vector of asset indices.

# Related

  - [`CorrelationComponents`](@ref)
"""
function correlation_components(rho::MatNum, thr::Number)
    n = size(rho, 1)
    parent = collect(1:n)
    find(i) = parent[i] == i ? i : (parent[i] = find(parent[i]))
    for j in 1:n, i in (j + 1):n
        if rho[i, j] >= thr
            ri, rj = find(i), find(j)
            if ri != rj
                parent[max(ri, rj)] = min(ri, rj)
            end
        end
    end
    groups = Dict{Int, Vector{Int}}()
    for i in 1:n
        push!(get!(groups, find(i), Int[]), i)
    end
    return collect(values(groups))
end
"""
$(DocStringExtensions.TYPEDEF)

Group assets by clustering them, and keep the best-scoring member of each cluster.

Clusters come from [`clusterise`](@ref), so the whole clustering family — hierarchical linkage, DBHT, the non-hierarchical algorithms, and the optimal-number-of-clusters estimators — is available for deciding what "redundant" means. Unlike the correlation algorithms there is no natural fallback survivor rule, so a [`RedundancySelector`](@ref) using `ClusterGroups` must carry a `score`.

A cluster whose best score is tied keeps nobody (see [`groups_argbest`](@ref)).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ClusterGroups(;
        cle::AbstractClustersEstimator = ClustersEstimator(),
    ) -> ClusterGroups

Keywords correspond to the struct's fields.

# Related

  - [`RedundancySelector`](@ref)
  - [`ClustersEstimator`](@ref)
  - [`clusterise`](@ref)
  - [`groups_argbest`](@ref)
"""
@concrete struct ClusterGroups <: AbstractRedundancyAlgorithm
    """
    Clustering estimator partitioning the assets ([`AbstractClustersEstimator`](@ref)).
    """
    cle
    function ClusterGroups(cle::AbstractClustersEstimator)
        return new{typeof(cle)}(cle)
    end
end
function ClusterGroups(;
                       cle::AbstractClustersEstimator = ClustersEstimator())::ClusterGroups
    return ClusterGroups(cle)
end
function redundancy_keep(alg::ClusterGroups, rd::AbstractReturnsResult,
                         scores::Option{<:VecNum}, bib::Bool)::BitVector
    clr = clusterise(alg.cle, rd.X)
    idx = assignments(clr)
    groups = [findall(==(k), idx) for k in 1:(clr.k)]
    return groups_argbest(groups, scores, bib)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate a correlation threshold.

# Related

  - [`PairwiseCorrelation`](@ref)
  - [`CorrelationComponents`](@ref)
"""
function assert_correlation_threshold(thr::Number)::Nothing
    @argcheck(-one(thr) <= thr <= one(thr),
              DomainError(thr, "a correlation threshold must lie in [-1, 1]"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Asset selector that discards assets which duplicate information already carried by others.

`alg` decides what "redundant" means and returns the keep-mask: greedy pairwise correlation pruning ([`PairwiseCorrelation`](@ref)), one representative per correlated blob ([`CorrelationComponents`](@ref)), or one representative per cluster ([`ClusterGroups`](@ref)).

`score` decides *which* asset survives a redundancy group — a risk measure evaluated on each asset's own return series, oriented by [`bigger_is_better`](@ref), exactly as in [`ScoreSelector`](@ref). Leaving it `nothing` falls back to the correlation algorithms' own rule: the asset with the lowest summary correlation to the rest of the universe survives. [`ClusterGroups`](@ref) has no such fallback and requires a `score`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RedundancySelector(;
        alg::AbstractRedundancyAlgorithm = PairwiseCorrelation(),
        score::Option{<:AbstractBaseRiskMeasure} = nothing,
    ) -> RedundancySelector

Keywords correspond to the struct's fields.

## Validation

  - If `score` is given, `supports_precomputed_returns(score)`.
  - If `requires_score(alg)`, `score` is not `nothing`.

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = ["A", "B", "C"],
                          X = [0.10 0.10 -0.05; -0.10 -0.10 0.07; 0.05 0.05 -0.02;
                               0.02 0.02 0.09]);

julia> sel = RedundancySelector(; alg = PairwiseCorrelation(; thr = 0.99), score = SCM());

julia> PortfolioOptimisers.fit_preprocessing(sel, rd).nx
1-element Vector{String}:
 "C"
```

`A` and `B` are identical, so neither survives — the tie policy discards both.

# Related

  - [`AbstractAssetSelector`](@ref)
  - [`AbstractRedundancyAlgorithm`](@ref)
  - [`ScoreSelector`](@ref)
"""
@concrete struct RedundancySelector <: AbstractAssetSelector
    """
    Algorithm deciding which assets are redundant ([`AbstractRedundancyAlgorithm`](@ref)).
    """
    alg
    """
    Risk measure choosing the survivor of each redundancy group; `nothing` uses the algorithm's own rule ([`AbstractBaseRiskMeasure`](@ref)).
    """
    score
    function RedundancySelector(alg::AbstractRedundancyAlgorithm,
                                score::Option{<:AbstractBaseRiskMeasure})
        if !isnothing(score)
            assert_scoreable(score)
        else
            @argcheck(!requires_score(alg),
                      IsNothingError("a $(typeof(alg)) redundancy algorithm cannot choose the survivor of a group on its own; give the RedundancySelector a score"))
        end
        return new{typeof(alg), typeof(score)}(alg, score)
    end
end
function RedundancySelector(; alg::AbstractRedundancyAlgorithm = PairwiseCorrelation(),
                            score::Option{<:AbstractBaseRiskMeasure} = nothing)::RedundancySelector
    return RedundancySelector(alg, score)
end
function select_assets(sel::RedundancySelector, rd::AbstractReturnsResult)::BitVector
    scores, bib = if isnothing(sel.score)
        nothing, false
    else
        asset_scores(sel.score, rd.X), bigger_is_better(sel.score)
    end
    return redundancy_keep(sel.alg, rd, scores, bib)
end

export ThresholdRule, RankRule, QuantileRule, ScoreSelector, CompleteAssetSelector,
       PairwiseCorrelation, CorrelationComponents, ClusterGroups, RedundancySelector
public select_assets, asset_scores, rule_keep, redundancy_keep, requires_score
