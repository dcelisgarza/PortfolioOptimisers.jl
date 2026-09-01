"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that turn per-asset scores into a keep-mask.

A selection rule is an [`AbstractAlgorithm`](@ref): it is consumed through [`ScoreSelector`](@ref) and never used on its own. Rules split into two kinds.

  - **Literal**: [`ThresholdRule`](@ref) compares a score against absolute bounds and ignores [`bigger_is_better`](@ref). A threshold on a variance means what it says; reinterpreting it as "keep the better ones" would invert the intent of a zero-variance filter.
  - **Ordinal**: [`RankRule`](@ref) and [`QuantileRule`](@ref) sort assets from best to worst — consulting [`bigger_is_better`](@ref), so `:best` is lowest risk for a risk measure and highest value for a return measure — and take counts or fractions from each tail.

# Interfaces

In order to implement a new selection rule that works seamlessly with the library, subtype `AbstractSelectionRule` with all necessary parameters as part of the struct, and implement the following method:

  - `rule_keep(rule::AbstractSelectionRule, scores::VecNum, bib::Bool) -> BitVector`: Turn the per-asset scores into a keep-mask.

## Arguments

  - `rule`: The concrete selection rule instance.
  - `scores`: Per-asset score vector `assets × 1`.
  - `bib`: [`bigger_is_better`](@ref) flag of the risk measure that produced `scores`. A literal rule ignores it.

## Returns

  - `keep::BitVector`: Mask `assets × 1` that is `true` for every admitted asset.

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

# Mathematical definition

```math
\\begin{align}
\\mathcal{K} &= \\left\\{ i : l < s_{i} < u \\right\\}\\,.
\\end{align}
```

Where:

  - $(math_dict[:K_keep_set])
  - $(math_dict[:s_i_score])
  - ``l``: `lo`, the lower bound. An omitted bound is ``-\\infty``.
  - ``u``: `hi`, the upper bound. An omitted bound is ``+\\infty``.

Both comparisons are **strict**, so the band is open: an asset whose score equals ``l`` or ``u`` is dropped.

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

# Mathematical definition

```math
\\begin{align}
\\mathcal{T}(k) &= \\left\\{ i : a_{i} + e_{i} \\leq k \\right\\}\\,, \\\\
\\mathcal{S} &= \\mathcal{T}_{\\mathrm{best}}(k_{b}) \\cup \\mathcal{T}_{\\mathrm{worst}}(k_{w})\\,, \\\\
\\mathcal{K} &= \\begin{cases}
    \\mathcal{S} & \\text{keep} \\\\
    \\left\\{ 1,\\, \\dots,\\, N \\right\\} \\setminus \\mathcal{S} & \\text{drop}
\\end{cases}\\,.
\\end{align}
```

Where:

  - $(math_dict[:K_keep_set])
  - $(math_dict[:s_i_score])
  - $(math_dict[:N])
  - $(math_dict[:k_tail_count])
  - ``\\mathcal{T}(k)``: Tail of size ``k``, taken at the end the subscript names.
  - ``a_{i}``: Number of assets whose score is strictly better than ``s_{i}``. At the `best` end a larger score is better when [`bigger_is_better`](@ref) is `true` and a smaller score is better when it is `false`; the `worst` end reverses that.
  - ``e_{i}``: Number of assets whose score equals ``s_{i}``, asset ``i`` included, so ``e_{i} \\geq 1``.
  - ``k_{b}``, ``k_{w}``: `best` and `worst`, each saturated to ``[0,\\, N]``. An omitted count is ``0``.
  - ``\\mathcal{S}``: Union of the two tails, before `action` is applied.

Two consequences follow. A tail admits asset ``i`` only when the whole tied block of ``i`` fits within ``k``, so a block that straddles the cut is excluded and ``\\left| \\mathcal{T}(k) \\right| \\leq k``. A universe whose scores are all equal gives ``a_{i} + e_{i} = N`` for every asset, so every tail of size ``k < N`` is empty.

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
    $(field_dict[:pre_action])
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

`best` and `worst` are fractions in `(0, 1)`, converted to counts as `round(Int, fraction * n, RoundNearestTiesUp)` on the window being fitted. An exact half rounds **up**, so `best = 0.625` on a 4-asset window takes 3 assets, not the 2 that Julia's default banker's rounding would give. Everything else — orientation via [`bigger_is_better`](@ref), the `action` complement, count saturation, and the tie policy that excludes a straddling tied block — is identical to [`RankRule`](@ref).

Fractions and counts are separate types on purpose: `best = 1` (one asset) and `best = 1.0` (the whole universe) would otherwise differ only by a literal's type.

# Mathematical definition

```math
\\begin{align}
k &= \\left\\lfloor f N + \\frac{1}{2} \\right\\rfloor\\,.
\\end{align}
```

Where:

  - $(math_dict[:k_tail_count])
  - $(math_dict[:N])
  - ``f``: `best` or `worst`, a fraction in ``(0,\\, 1)``.

The floor of ``f N + 1/2`` is `RoundNearestTiesUp`: an exact half rounds **up**, and not to the even neighbour that Julia's default `RoundNearest` picks. The two counts are then the ``k_{b}`` and ``k_{w}`` of [`RankRule`](@ref), which states the admitted set.

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
    $(field_dict[:pre_action])
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

# Arguments

  - `action`: The `action` field of an ordinal selection rule.

# Validation

  - `action in (:keep, :drop)`, else an `ArgumentError` is thrown naming the value it got.

# Returns

  - `nothing`.

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

# Arguments

  - `best`: Number of assets to take from the best end, or `nothing`.
  - `worst`: Number of assets to take from the worst end, or `nothing`.
  - `name`: Name of the rule being constructed, written into every exception message.

# Validation

  - At least one of `best`, `worst` is not `nothing`, else an `IsNothingError` is thrown.
  - Every count that is given is `>= 0`, else a `DomainError` is thrown.
  - The larger of the two counts is `> 0`, else a `DomainError` is thrown. A rule that takes no asset is rejected.

# Returns

  - `nothing`.

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

Assets tied with the `k`-th score are included only when the whole tied block fits within `k`. Otherwise the block straddles the cut and is excluded — the "trust neither" tie policy, which is why the returned mask may hold fewer than `k` assets. [`RankRule`](@ref) states the admitted set in closed form.

# Algorithm

 1. Read `n`, the number of scores. Return `n` falses when `k` is not positive, and `n` trues when `k` is at least `n`. The second case is the saturation of the count.
 2. Choose the comparison `ahead`. It is `>` when `tail` and `bib` agree that a larger score lies further into the tail, and `<` otherwise.
 3. Sort the scores under `ahead` with `sortperm`, giving `perm`, and read `cut`, the score at position `k` of that order.
 4. Count the scores strictly `ahead` of `cut`, giving `n_ahead`, and the scores equal to `cut`, giving `n_eq`.
 5. Set `keep_ties` to `n_ahead + n_eq == k`. The tied block at the cut fits inside `k` only then, because `cut` sits at position `k` and so `n_ahead + n_eq >= k` always holds.
 6. Return the mask that is `true` for every score strictly `ahead` of `cut`, and for a score equal to `cut` when `keep_ties` is `true`.

# Arguments

  - `scores`: Per-asset score vector `assets × 1`.
  - `k`: Number of assets to take from the `tail` end.
  - `bib`: [`bigger_is_better`](@ref) flag of the score.
  - `tail`: `:best` or `:worst`, the end to take from.

# Returns

  - `keep::BitVector`: Mask `assets × 1` that is `true` for every asset in the tail. It holds at most `k` assets, and fewer when a tied block straddles the cut.

# Related

  - [`RankRule`](@ref)
  - [`tail_action_mask`](@ref)
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

One method per rule. The [`ThresholdRule`](@ref) method compares each score against the open band and returns the mask directly. The [`RankRule`](@ref) method hands its two counts to [`tail_action_mask`](@ref). The [`QuantileRule`](@ref) method converts each fraction to a count with `round(Int, f * n, RoundNearestTiesUp)` first, then hands those counts to the same function.

# Arguments

  - `rule`: The selection rule.
  - `scores`: Per-asset score vector `assets × 1`.
  - `bib`: [`bigger_is_better`](@ref) flag of the score that produced `scores`.

# Returns

  - `keep::BitVector`: Mask `assets × 1` that is `true` for every asset the rule admits.

# Related

  - [`AbstractSelectionRule`](@ref)
  - [`ScoreSelector`](@ref)
  - [`tail_action_mask`](@ref)
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

# Algorithm

 1. Start from a mask of `length(scores)` falses.
 2. When `best` is not `nothing`, take the `:best` tail of size `best` with [`tail_mask`](@ref) and union it into the mask.
 3. When `worst` is not `nothing`, take the `:worst` tail of size `worst` with [`tail_mask`](@ref) and union it into the mask.
 4. Return the mask when `action` is `:keep`, and its complement when `action` is `:drop`.

# Arguments

  - `best`: Number of assets to take from the best end, or `nothing`.
  - `worst`: Number of assets to take from the worst end, or `nothing`.
  - `action`: `:keep` returns the union, `:drop` returns its complement.
  - `scores`: Per-asset score vector `assets × 1`.
  - `bib`: [`bigger_is_better`](@ref) flag of the score.

# Returns

  - `keep::BitVector`: Mask `assets × 1` that is `true` for every asset the rule admits.

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

# Arguments

  - `score`: The risk measure a selector scores its assets with.

# Validation

  - `supports_precomputed_returns(score)`, else an `ArgumentError` is thrown. The message names the measure, and adds a pointer to `SCM()` when the measure is a [`Variance`](@ref) or a [`StandardDeviation`](@ref).

# Returns

  - `nothing`.

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
        throw(ArgumentError("`$(Base.typename(typeof(score)).wrapper)` cannot score a single asset's return series: its `supports_precomputed_returns` is false, so its functor consumes portfolio weights rather than a precomputed return vector.$hint"))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Evaluate `score` on every asset column of `X`.

Columns are passed as views: a risk-measure functor reads its argument and never writes to it.

# Algorithm

 1. Evaluate `score` on a view of each asset column of `X` in turn, giving `scores`.
 2. Check that every entry of `scores` is finite.
 3. Return `scores`.

# Arguments

  - `score`: The risk measure to evaluate on each asset column.
  - $(arg_dict[:X])

# Validation

  - Every score is finite, else a `DomainError` is thrown naming the offending columns. A `NaN` score ([`Skewness`](@ref) on a constant series, say, which divides by a zero standard deviation) would make the ordering meaningless, so it throws rather than sorting arbitrarily.

# Returns

  - `scores::VecNum`: Score vector `assets × 1`, one entry per asset column of `X`.

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
julia> rd = ReturnsResult(; nx = [\"A\", \"B\", \"C\"], X = [0.1 0.0 -0.2; -0.1 0.0 0.3; 0.2 0.0 -0.1]);

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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Select the assets a [`ScoreSelector`](@ref) keeps: score every asset, then apply the rule.

# Algorithm

 1. Score every asset column of `rd.X` with the selector's `score`, using [`asset_scores`](@ref).
 2. Read the orientation of the score with [`bigger_is_better`](@ref).
 3. Return the keep-mask [`rule_keep`](@ref) admits for those scores under the selector's `rule`.

# Arguments

  - `sel`: The score selector.
  - $(arg_dict[:rd])

# Returns

  - `keep::BitVector`: Mask `assets × 1` that is `true` for every asset the rule admits.

# Related

  - [`ScoreSelector`](@ref)
  - [`asset_scores`](@ref)
  - [`rule_keep`](@ref)
"""
function select_assets(sel::ScoreSelector, rd::AbstractReturnsResult)::BitVector
    scores = asset_scores(sel.score, rd.X)
    return rule_keep(sel.rule, scores, bigger_is_better(sel.score))
end
"""
$(DocStringExtensions.TYPEDEF)

Asset selector that drops every asset column holding a `NaN` observation.

The returns-level counterpart of [`MissingDataFilter`](@ref)'s column threshold, for pipelines fed returns data directly (where the price stages never run). It has no observation-dropping mode: a fitted selector cannot decide which rows of an unseen window to drop without breaking the weights/returns alignment.

A returns carrier binds `X` to a matrix of numbers, so a `missing` never reaches this selector: [`ReturnsResult`](@ref) rejects a `Matrix{Union{Missing, Float64}}` at construction. [`find_complete_indices`](@ref) reads both sentinels, and [`MissingDataFilter`](@ref) removes a `missing` from the price data upstream.

# Constructors

    CompleteAssetSelector() -> CompleteAssetSelector

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; 0.3 NaN]);

julia> PortfolioOptimisers.fit_preprocessing(CompleteAssetSelector(), rd).nx
1-element Vector{String}:
 "A"
```

# Related

  - [`AbstractAssetSelector`](@ref)
  - [`MissingDataFilter`](@ref)
"""
struct CompleteAssetSelector <: AbstractAssetSelector end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Select the assets a [`CompleteAssetSelector`](@ref) keeps: the asset columns that hold no `NaN`.

# Algorithm

 1. Start from a keep-mask of `size(rd.X, 2)` falses.
 2. Find the asset columns of `rd.X` that hold neither a `missing` nor a `NaN`, with [`find_complete_indices`](@ref) along the observation axis.
 3. Set the mask at those columns, and return it.

# Arguments

  - The selector is taken by type alone. It carries no field, so nothing is read from it.
  - $(arg_dict[:rd])

# Returns

  - `keep::BitVector`: Mask `assets × 1` that is `true` for every complete asset column.

# Related

  - [`CompleteAssetSelector`](@ref)
  - [`find_complete_indices`](@ref)
"""
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

## Arguments

  - `alg`: The concrete redundancy algorithm instance.
  - $(arg_dict[:rd])
  - `scores`: Per-asset score vector `assets × 1`, or `nothing` when the selector carries no `score`.
  - `bib`: [`bigger_is_better`](@ref) flag of the score. It is `false` when `scores` is `nothing`.

## Returns

  - `keep::BitVector`: Mask `assets × 1` that is `true` for every surviving asset.

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

The generic method answers `true`, so a new algorithm is asked for a score until it says otherwise. That is the safe default: an algorithm that silently accepted `nothing` and had no fallback would pick a survivor from a `nothing` score vector.

# Arguments

  - The algorithm is taken by type alone. No field of it is read.

# Returns

  - `req::Bool`: `true` when the algorithm cannot pick the survivor of a group without a score.

# Related

  - [`AbstractRedundancyAlgorithm`](@ref)
  - [`RedundancySelector`](@ref)
"""
requires_score(::AbstractRedundancyAlgorithm)::Bool = true
"""
    redundancy_keep(alg::AbstractRedundancyAlgorithm, rd, scores, bib) -> BitVector

Return the keep-mask a redundancy algorithm admits.

`scores` is `nothing` when the [`RedundancySelector`](@ref) carries no `score`; otherwise it is the per-asset score vector and `bib` is the score's [`bigger_is_better`](@ref) flag.

The method shown here is the family's fallback. It runs only for an algorithm that implements none of its own, and it always throws.

# Arguments

  - `alg`: The redundancy algorithm.
  - $(arg_dict[:rd])
  - `scores`: Per-asset score vector `assets × 1`, or `nothing`.
  - `bib`: [`bigger_is_better`](@ref) flag of the score.

# Validation

  - The concrete algorithm implements `redundancy_keep`, else an `ArgumentError` is thrown naming the two methods an extension author must define.

# Returns

  - `keep::BitVector`: Mask `assets × 1` that is `true` for every surviving asset.

# Related

  - [`AbstractRedundancyAlgorithm`](@ref)
  - [`RedundancySelector`](@ref)
"""
function redundancy_keep(alg::AbstractRedundancyAlgorithm, ::AbstractReturnsResult,
                         ::Option{<:VecNum}, ::Bool)
    return throw(ArgumentError("$(typeof(alg)) subtypes AbstractRedundancyAlgorithm but does not implement redundancy_keep. Extension authors: a redundancy algorithm must implement redundancy_keep(alg, rd, scores, bib) -> BitVector returning the keep-mask, and requires_score(alg) -> Bool when it cannot pick a survivor without a score."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Convert per-asset scores into *drop scores*, where higher means "discard me first".

A risk measure with `bigger_is_better == false` (lower risk is better) already reads as a drop score; one with `bigger_is_better == true` is negated. Downstream of this call a **lower** number is always better, whatever the measure's own orientation was.

# Mathematical definition

```math
\\begin{align}
d_{i} &= \\begin{cases}
    -s_{i} & \\text{a larger score is better} \\\\
    s_{i} & \\text{otherwise}
\\end{cases}\\,.
\\end{align}
```

Where:

  - ``d_{i}``: Drop score of asset ``i``. The asset with the higher drop score is discarded first.
  - $(math_dict[:s_i_score])

The map is order-reversing in the first case and order-preserving in the second, so it never changes which of two assets is preferred. It changes only the direction that carries that preference.

# Arguments

  - `scores`: Per-asset score vector `assets × 1`.
  - `bib`: [`bigger_is_better`](@ref) flag of the score that produced `scores`.

# Returns

  - `d::VecNum`: Drop score vector `assets × 1`, in which a higher number means "discard me first".

# Related

  - [`RedundancySelector`](@ref)
  - [`find_uncorrelated_indices`](@ref)
  - [`bigger_is_better`](@ref)
"""
drop_scores(scores::VecNum, bib::Bool) = bib ? -scores : scores
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Keep the single best-scoring member of each redundancy group.

A group whose best score is *tied* keeps nobody: under the library's "if we cannot tell them apart, trust neither" policy, two indistinguishable assets are both discarded — the same stance [`find_uncorrelated_indices`](@ref) takes on an exactly-tied correlated pair. A singleton group is trivially unambiguous and always survives.

# Algorithm

 1. Start from a keep-mask of `length(scores)` falses.
 2. Choose the comparison `better`. It is `>` when `bib` is `true`, and `<` otherwise.
 3. Take the next group `g` of `groups`, and skip it when it is empty.
 4. Walk the members of `g` from the first, holding in `best` the index whose score no later member beats under `better`.
 5. Count the members of `g` whose score equals `scores[best]`. Set the mask at `best` only when that count is one, so a group with a tied best keeps nobody.
 6. Repeat steps 3 to 5 over the remaining groups, then return the mask.

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

Greedy pairwise correlation pruning: drop assets until no surviving pair exceeds `t`.

Correlated pairs are visited from most to least correlated, and the worse asset of each pair is removed. "Worse" means the higher drop score: the `RedundancySelector`'s `score` when it has one, otherwise each asset's summary correlation to the rest of the universe — so the asset that is redundant with *most* of the universe goes first.

This algorithm never **chains**. At `t = 0.7`, a universe with `ρ(A, B) = 0.80`, `ρ(B, C) = 0.81` and `ρ(A, C) = 0.32` loses `B` and keeps both `A` and `C`, honouring the literal promise that no surviving pair exceeds `t`. [`CorrelationComponents`](@ref) reads the same three correlations transitively and keeps only `A`.

How loose the middle correlation can be is bounded by the other two: two edges at `ρ` force the third above `ρ² - (1 - ρ²)`, so a chain of two `0.97` edges cannot have a third correlation below `0.88`. A weakly-connected chain therefore needs weak edges, and the two algorithms diverge most where `t` sits just under them.

Delegates to [`find_uncorrelated_indices`](@ref).

# Mathematical definition

```math
\\begin{align}
\\rho_{i,\\,j} &< t \\quad \\forall\\, i \\neq j \\in \\mathcal{K}\\,.
\\end{align}
```

Where:

  - $(math_dict[:rho_ij]) The absolute value ``\\left| \\rho_{i,\\,j} \\right|`` is read instead when `absolute` is `true`.
  - $(math_dict[:t_corr_threshold])
  - $(math_dict[:K_keep_set])

That is a promise about surviving **pairs** and about nothing else. It does not say that ``\\mathcal{K}`` is the largest such set, and it does not close under transitivity: a chain of two over-threshold edges whose end points are under the threshold satisfies it with both end points kept.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PairwiseCorrelation(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        t::Number = 0.95,
        absolute::Bool = false,
        measure::Num_VecToScaM = MeanValue(),
    ) -> PairwiseCorrelation

Keywords correspond to the struct's fields.

## Validation

  - `-1 <= t <= 1`, checked by [`assert_correlation_threshold`](@ref).

# Related

  - [`RedundancySelector`](@ref)
  - [`CorrelationComponents`](@ref)
  - [`find_uncorrelated_indices`](@ref)
"""
@concrete struct PairwiseCorrelation <: AbstractRedundancyAlgorithm
    """
    $(field_dict[:pre_ce_corr])
    """
    ce
    """
    $(field_dict[:pre_t_corr])
    """
    t
    """
    $(field_dict[:pre_absolute])
    """
    absolute
    """
    $(field_dict[:pre_measure])
    """
    measure
    function PairwiseCorrelation(ce::StatsBase.CovarianceEstimator, t::Number,
                                 absolute::Bool, measure::Num_VecToScaM)
        assert_correlation_threshold(t)
        return new{typeof(ce), typeof(t), typeof(absolute), typeof(measure)}(ce, t,
                                                                             absolute,
                                                                             measure)
    end
end
function PairwiseCorrelation(;
                             ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                             t::Number = 0.95, absolute::Bool = false,
                             measure::Num_VecToScaM = MeanValue())::PairwiseCorrelation
    return PairwiseCorrelation(ce, t, absolute, measure)
end
requires_score(::PairwiseCorrelation)::Bool = false
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Keep a maximally uncorrelated subset under [`PairwiseCorrelation`](@ref), by delegating to [`find_uncorrelated_indices`](@ref).

# Algorithm

 1. Start from a keep-mask of `size(rd.X, 2)` falses.
 2. Turn `scores` into drop scores with [`drop_scores`](@ref), so that downstream a **lower** number is better. Pass `nothing` when `scores` is `nothing`, which leaves [`find_uncorrelated_indices`](@ref) to build its own drop score by collapsing each column of the correlation matrix with `measure`.
 3. Call [`find_uncorrelated_indices`](@ref) on `rd.X` with the algorithm's `ce`, `t`, `absolute` and `measure`, giving `idx`, the surviving asset indices.
 4. Set the mask at `idx`, and return it.

# Arguments

  - `alg`: The pairwise correlation algorithm.
  - $(arg_dict[:rd])
  - `scores`: Per-asset score vector `assets × 1`, or `nothing`.
  - `bib`: [`bigger_is_better`](@ref) flag of the score.

# Returns

  - `keep::BitVector`: Mask `assets × 1` that is `true` for every surviving asset.

# Related

  - [`PairwiseCorrelation`](@ref)
  - [`drop_scores`](@ref)
  - [`find_uncorrelated_indices`](@ref)
"""
function redundancy_keep(alg::PairwiseCorrelation, rd::AbstractReturnsResult,
                         scores::Option{<:VecNum}, bib::Bool)::BitVector
    keep = falses(size(rd.X, 2))
    idx = find_uncorrelated_indices(rd.X; ce = alg.ce, t = alg.t, absolute = alg.absolute,
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

Two assets share an edge when their (absolute) correlation is at or above `t`. Components are transitive, so this reads a chain `A ~ B ~ C` as one redundant blob even when `A` and `C` are uncorrelated, and keeps a single asset from it. That is a stronger claim than [`PairwiseCorrelation`](@ref)'s, and a stronger reduction; choose it when you want one representative per correlated blob rather than a guarantee about surviving pairs.

A component whose best score is tied keeps nobody (see [`groups_argbest`](@ref)).

# Mathematical definition

```math
\\begin{align}
\\mathcal{E} &= \\left\\{ \\left\\{ i,\\, j \\right\\} : i \\neq j\\,,\\; \\rho_{i,\\,j} \\geq t \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathcal{E}``: Edge set of the redundancy graph, whose vertices are the assets.
  - $(math_dict[:rho_ij]) The absolute value ``\\left| \\rho_{i,\\,j} \\right|`` is read instead when `absolute` is `true`.
  - $(math_dict[:t_corr_threshold])

The groups are the connected components of that graph, so every asset lies in exactly one of them and an asset with no over-threshold partner forms a singleton. A component is closed under transitivity even though the edge relation is not, which is the whole difference from [`PairwiseCorrelation`](@ref): a chain ``i \\sim j \\sim k`` is one component whatever ``\\rho_{i,\\,k}`` is.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CorrelationComponents(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        t::Number = 0.95,
        absolute::Bool = false,
        measure::Num_VecToScaM = MeanValue(),
    ) -> CorrelationComponents

Keywords correspond to the struct's fields.

## Validation

  - `-1 <= t <= 1`, checked by [`assert_correlation_threshold`](@ref).

# Related

  - [`RedundancySelector`](@ref)
  - [`PairwiseCorrelation`](@ref)
  - [`groups_argbest`](@ref)
"""
@concrete struct CorrelationComponents <: AbstractRedundancyAlgorithm
    """
    $(field_dict[:pre_ce_corr])
    """
    ce
    """
    $(field_dict[:pre_t_corr])
    """
    t
    """
    $(field_dict[:pre_absolute])
    """
    absolute
    """
    $(field_dict[:pre_measure]) Lower is better, so the surviving representative is the least redundant member of its component.
    """
    measure
    function CorrelationComponents(ce::StatsBase.CovarianceEstimator, t::Number,
                                   absolute::Bool, measure::Num_VecToScaM)
        assert_correlation_threshold(t)
        return new{typeof(ce), typeof(t), typeof(absolute), typeof(measure)}(ce, t,
                                                                             absolute,
                                                                             measure)
    end
end
function CorrelationComponents(;
                               ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                               t::Number = 0.95, absolute::Bool = false,
                               measure::Num_VecToScaM = MeanValue())::CorrelationComponents
    return CorrelationComponents(ce, t, absolute, measure)
end
requires_score(::CorrelationComponents)::Bool = false
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Keep one representative of each correlated blob under [`CorrelationComponents`](@ref).

# Algorithm

 1. Compute the correlation matrix `rho` of `rd.X` with the algorithm's `ce`.
 2. When `absolute` is `true`, replace `rho` with its entrywise absolute value. This happens **before** the threshold is applied, so a strongly negative correlation is an edge.
 3. When `scores` is `nothing`, collapse each column of `rho` with `measure` into the fallback score `s`, and force `sbib` to `false`. A lower summary correlation is then better, so the surviving representative is the asset least correlated with the rest of the universe. Otherwise take `s` from `scores` and `sbib` from `bib`.
 4. Group the assets with [`correlation_components`](@ref) on `rho` and `t`.
 5. Return the mask [`groups_argbest`](@ref) admits for those groups under `s` and `sbib`.

# Arguments

  - `alg`: The correlation components algorithm.
  - $(arg_dict[:rd])
  - `scores`: Per-asset score vector `assets × 1`, or `nothing`.
  - `bib`: [`bigger_is_better`](@ref) flag of the score.

# Returns

  - `keep::BitVector`: Mask `assets × 1` that is `true` for one asset of every component whose best score is not tied.

# Related

  - [`CorrelationComponents`](@ref)
  - [`correlation_components`](@ref)
  - [`groups_argbest`](@ref)
"""
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
    return groups_argbest(correlation_components(rho, alg.t), s, sbib)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the connected components of the graph whose edges are the pairs of `rho` at or above `t`.

A union-find pass over the strict lower triangle, so components are transitive and every asset lands in exactly one of them (a singleton when it has no over-threshold partner).

# Algorithm

 1. Read `n`, the number of rows of `rho`, and set `parent[i]` to `i` for every asset. Each asset starts as its own component.
 2. Take the next entry of the strict lower triangle of `rho`, which holds each pair once.
 3. When that entry is at least `t`, find the root of each member of the pair with `find`, and point the larger root at the smaller one. `find` compresses the path it walks, so a later lookup on that chain is one step.
 4. Repeat steps 2 and 3 over the remaining pairs.
 5. Group the assets by their final root into `groups`, keyed by that root.
 6. Return the values of `groups`.

The threshold is compared against `rho` as it is given, so a caller that wants the absolute value takes it before this call.

# Arguments

  - `rho`: Correlation matrix `assets × assets`.
  - $(arg_dict[:t])

# Returns

  - `groups::Vector{Vector{Int}}`: The components, each a vector of asset indices in ascending order. The components themselves come out of a `Dict`, so their order is not defined. [`groups_argbest`](@ref) reads them as a set and does not depend on it.

# Related

  - [`CorrelationComponents`](@ref)
"""
function correlation_components(rho::MatNum, t::Number)
    n = size(rho, 1)
    parent = collect(1:n)
    find(i) = parent[i] == i ? i : (parent[i] = find(parent[i]))
    for j in 1:n, i in (j + 1):n
        if rho[i, j] >= t
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

## Clustering on a feature matrix

A [`FeatureDistance`](@ref) in the `cle`'s distance slot measures a feature matrix rather than the returns, and `ClusterGroups` supplies it from `rd.Z` — the data carrier the selector is fitted on. There is no `z_src` field here, and that absence is the whole statement: preselection is a *pre-prior* site, so only the data carrier can supply a feature matrix. A selector is fitted by [`fit_preprocessing`](@ref) from the returns data alone and never sees a prior result; in a [`Pipeline`](@ref) it writes `:returns`, which invalidates any `:prior` already computed. An optimiser's `z_src` therefore does not reach here, and setting it changes nothing about this call. Supply `Z` on the [`ReturnsResult`](@ref) — for instance from [`asset_sets_features`](@ref) — or the clustering throws (see [`assert_feature_matrix_supplied`](@ref)).

The selection is decided on the *full* universe and the surviving columns are sliced only afterwards, so `Z` is measured over every asset before any is dropped.

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
  - [`FeatureDistance`](@ref)
  - [`assert_feature_matrix_supplied`](@ref)
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Keep one representative of each cluster under [`ClusterGroups`](@ref).

# Algorithm

 1. Cluster the assets with [`clusterise`](@ref) on `rd.X`, passing the feature matrix `rd.Z` and `z_src = :data_only`, giving the clustering result `clr`.
 2. Read the cluster assignment of every asset into `idx`.
 3. Collect the asset indices of each of the `clr.k` clusters into `groups`.
 4. Return the mask [`groups_argbest`](@ref) admits for those groups under `scores` and `bib`.

`z_src` is fixed at `:data_only` because preselection runs before any prior exists, so the data carrier is the only reachable source of a feature matrix. [`ClusterGroups`](@ref) states why the type carries no field for it.

# Arguments

  - `alg`: The cluster groups algorithm.
  - $(arg_dict[:rd])
  - `scores`: Per-asset score vector `assets × 1`. [`RedundancySelector`](@ref) rejects a `nothing` score for this algorithm at construction, so it is never `nothing` here.
  - `bib`: [`bigger_is_better`](@ref) flag of the score.

# Returns

  - `keep::BitVector`: Mask `assets × 1` that is `true` for one asset of every cluster whose best score is not tied.

# Related

  - [`ClusterGroups`](@ref)
  - [`clusterise`](@ref)
  - [`groups_argbest`](@ref)
"""
function redundancy_keep(alg::ClusterGroups, rd::AbstractReturnsResult,
                         scores::Option{<:VecNum}, bib::Bool)::BitVector
    clr = clusterise(alg.cle, rd.X; Z = rd.Z, z_src = :data_only)
    idx = assignments(clr)
    groups = [findall(==(k), idx) for k in 1:(clr.k)]
    return groups_argbest(groups, scores, bib)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate a correlation threshold.

# Arguments

  - $(arg_dict[:t])

# Validation

  - `-1 <= t <= 1`, else a `DomainError` is thrown. The bound is the range of a correlation coefficient, so a threshold outside it can never be met or can never be missed.

# Returns

  - `nothing`.

# Related

  - [`PairwiseCorrelation`](@ref)
  - [`CorrelationComponents`](@ref)
"""
function assert_correlation_threshold(t::Number)::Nothing
    @argcheck(-one(t) <= t <= one(t),
              DomainError(t, "a correlation threshold must lie in [-1, 1]"))
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
julia> rd = ReturnsResult(; nx = [\"A\", \"B\", \"C\"],
                          X = [0.10 0.10 -0.05; -0.10 -0.10 0.07; 0.05 0.05 -0.02;
                               0.02 0.02 0.09]);

julia> sel = RedundancySelector(; alg = PairwiseCorrelation(; t = 0.99), score = SCM());

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
                      IsNothingError("a $(Base.typename(typeof(alg)).wrapper) redundancy algorithm cannot choose the survivor of a group on its own; give the RedundancySelector a score"))
        end
        return new{typeof(alg), typeof(score)}(alg, score)
    end
end
function RedundancySelector(; alg::AbstractRedundancyAlgorithm = PairwiseCorrelation(),
                            score::Option{<:AbstractBaseRiskMeasure} = nothing)::RedundancySelector
    return RedundancySelector(alg, score)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Select the assets a [`RedundancySelector`](@ref) keeps: score every asset when a score is given, then apply the redundancy algorithm.

# Algorithm

 1. When the selector carries no `score`, set `scores` to `nothing` and `bib` to `false`. The algorithm then uses its own survivor rule.
 2. Otherwise score every asset column of `rd.X` with [`asset_scores`](@ref), and read `bib` from [`bigger_is_better`](@ref) of that score.
 3. Return the keep-mask [`redundancy_keep`](@ref) admits for the selector's `alg`.

# Arguments

  - `sel`: The redundancy selector.
  - $(arg_dict[:rd])

# Returns

  - `keep::BitVector`: Mask `assets × 1` that is `true` for every surviving asset.

# Related

  - [`RedundancySelector`](@ref)
  - [`asset_scores`](@ref)
  - [`redundancy_keep`](@ref)
"""
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
