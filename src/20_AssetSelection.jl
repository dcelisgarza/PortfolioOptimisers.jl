"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that turn the scores of the assets into a keep-mask.

A selection rule is an [`AbstractAlgorithm`](@ref). A [`ScoreSelector`](@ref) uses it, and it has no use on its own. There are two kinds of rule.

  - **Literal.** [`ThresholdRule`](@ref) compares a score with absolute bounds, and it ignores [`bigger_is_better`](@ref). A threshold on a variance is the variance that it states. The rule does not read it as "keep the better assets". That change inverts the purpose of a [`ZeroVarianceFilter`](@ref).
  - **Ordinal.** [`RankRule`](@ref) and [`QuantileRule`](@ref) sort the assets from the best to the worst, and take a count or a fraction from each end. [`bigger_is_better`](@ref) sets the order, so `:best` is the lowest risk for a risk measure and the highest value for a return measure.

# Interfaces

To add a selection rule, subtype `AbstractSelectionRule`, put every parameter of the rule in the struct, and add this method:

  - `rule_keep(rule::AbstractSelectionRule, scores::VecNum, bib::Bool) -> BitVector`: Turn the scores of the assets into a keep-mask.

## Arguments

  - `rule`: The concrete selection rule.
  - `scores`: The score vector `assets × 1`, with one score for each asset.
  - `bib`: The [`bigger_is_better`](@ref) flag of the risk measure that gave `scores`. A literal rule ignores it.

## Returns

  - `keep::BitVector`: The mask `assets × 1`. It is `true` for every asset that the rule admits.

# Related

  - [`ScoreSelector`](@ref)
  - [`ThresholdRule`](@ref)
  - [`RankRule`](@ref)
  - [`QuantileRule`](@ref)
"""
abstract type AbstractSelectionRule <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Keep the assets whose score is strictly inside the band `(lo, hi)`.

Both bounds are optional, and both are literal. The rule compares `lo` and `hi` with the raw score, and [`bigger_is_better`](@ref) does not change them. A bound that the caller omits sets no limit on that side.

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

Both comparisons are **strict**, so the band is open. The rule drops an asset whose score is equal to ``l`` or to ``u``.

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
    The exclusive lower bound on the score. `nothing` sets no lower bound.
    """
    lo
    """
    The exclusive upper bound on the score. `nothing` sets no upper bound.
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

Take a number of assets from each end of the order of the scores, then keep them or drop them.

`best` and `worst` are **counts that the rule takes from each end**, not positions. `best = 20` means twenty assets, not the asset at rank twenty. [`bigger_is_better`](@ref) of the score sets which end is the best. For a risk measure, `RankRule(; best = 20)` keeps the twenty assets with the lowest risk. For `MeanReturn`, it keeps the twenty with the highest return. A rule with both counts takes both ends. `action = :drop` takes the complement of the selection. With it, a rule can drop the five worst assets with no knowledge of the size of the universe.

A count **saturates** at the number of assets. On a window of 30 assets, `best = 50` keeps all 30 and does not throw. A hyperparameter search over `best` then does not fail at its largest value.

!!! warning

    The rule **excludes every asset** of a tied block at the cut, so it can return *fewer* assets than the count. If the 20th and the 21st assets have equal scores, `RankRule(; best = 20)` keeps 19. The rule drops the tied block, and it does not split the block arbitrarily.

    This is the tie policy of the library: if the library cannot tell two assets apart, it keeps neither of them. `find_uncorrelated_indices` has the same policy. It removes both assets of a correlated pair whose drop scores are equal. A window whose scores are all equal selects no asset, and [`fit_preprocessing`](@ref) throws.

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
  - ``\\mathcal{T}(k)``: The tail of size ``k``, at the end that the subscript names.
  - ``a_{i}``: The number of assets whose score is strictly better than ``s_{i}``. At the `best` end, a larger score is better when [`bigger_is_better`](@ref) is `true`, and a smaller score is better when it is `false`. The `worst` end reverses this order.
  - ``e_{i}``: The number of assets whose score is equal to ``s_{i}``. The number includes asset ``i``, so ``e_{i} \\geq 1``.
  - ``k_{b}``, ``k_{w}``: The counts `best` and `worst`, each saturated to ``[0,\\, N]``. An omitted count is ``0``.
  - ``\\mathcal{S}``: The union of the two tails, before the rule applies `action`.

Two facts follow from this definition. A tail admits asset ``i`` only when the whole tied block of ``i`` fits in ``k``. A tail excludes a block that crosses the cut, so ``\\left| \\mathcal{T}(k) \\right| \\leq k``. If all the scores are equal, ``a_{i} + e_{i} = N`` for every asset, so every tail of size ``k < N`` is empty.

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
  - Every given count is `>= 0`, and at least one count is `> 0`.
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
    The number of assets to take from the best end. `nothing` takes no asset.
    """
    best
    """
    The number of assets to take from the worst end. `nothing` takes no asset.
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

A [`RankRule`](@ref) whose tail sizes are *fractions* of the asset universe.

`best` and `worst` are fractions in `(0, 1)`. The rule converts each fraction to a count with `round(Int, fraction * n, RoundNearestTiesUp)` on the window that it fits. An exact half rounds **up**. For example, `best = 0.625` on a window of 4 assets takes 3 assets, and not the 2 that the default `RoundNearest` of Julia gives.

The other behaviour is the same as for [`RankRule`](@ref). [`bigger_is_better`](@ref) sets the order, `action = :drop` takes the complement, a count saturates, and the rule excludes a tied block that crosses the cut.

Fractions and counts have separate types on purpose. If one rule took both, `best = 1` is one asset and `best = 1.0` is the whole universe. Only the type of the literal tells the two apart.

# Mathematical definition

```math
\\begin{align}
k &= \\left\\lfloor f N + \\frac{1}{2} \\right\\rfloor\\,.
\\end{align}
```

Where:

  - $(math_dict[:k_tail_count])
  - $(math_dict[:N])
  - ``f``: The fraction `best` or `worst`, in ``(0,\\, 1)``.

The floor of ``f N + 1/2`` is the rounding mode `RoundNearestTiesUp`. An exact half rounds **up**, and not to the even neighbour that the default `RoundNearest` of Julia selects. The two counts are then the ``k_{b}`` and the ``k_{w}`` of [`RankRule`](@ref), which states the admitted set.

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
  - Every given fraction is in `(0, 1)`.
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
    The fraction of the universe to take from the best end. `nothing` takes no asset.
    """
    best
    """
    The fraction of the universe to take from the worst end. `nothing` takes no asset.
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

Check the `action` field that the ordinal selection rules share.

# Arguments

  - `action`: The `action` field of an ordinal selection rule.

# Validation

  - `action in (:keep, :drop)`. Otherwise the function throws an `ArgumentError` that names the value of `action`.

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

Check the tail counts `best` and `worst` of a [`RankRule`](@ref).

[`QuantileRule`](@ref) takes fractions, and its constructor checks them itself.

# Arguments

  - `best`: The number of assets to take from the best end, or `nothing`.
  - `worst`: The number of assets to take from the worst end, or `nothing`.
  - `name`: The name of the rule that the caller constructs. Every exception message names it.

# Validation

  - At least one of `best` and `worst` is not `nothing`. Otherwise the function throws an `IsNothingError`.
  - Every given count is `>= 0`. Otherwise the function throws a `DomainError`.
  - The larger of the two counts is `> 0`. Otherwise the function throws a `DomainError`, because the rule takes no asset.

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

Return the mask of the `k` assets that are furthest into the `tail` end of the order of the scores.

`tail` is `:best` or `:worst`. `bib`, the [`bigger_is_better`](@ref) flag of the score, sets which raw direction each end is. The count saturates at `length(scores)`.

The mask includes the assets tied with the score at position `k` only when the whole tied block fits in `k`. Otherwise the block crosses the cut, and the mask excludes it. This is the tie policy of the library, and it is the reason that the mask can hold fewer than `k` assets. [`RankRule`](@ref) states the admitted set in closed form.

# Algorithm

 1. Read `n`, the number of scores. Return `n` falses when `k` is not positive, and `n` trues when `k` is at least `n`. The second case is the saturation of the count.
 2. Select the comparison `ahead`. It is `>` when `tail` and `bib` agree that a larger score is further into the tail, and `<` otherwise.
 3. Sort the scores under `ahead` with `sortperm` into `perm`. Read `cut`, the score at position `k` of that order.
 4. Count the scores strictly `ahead` of `cut` into `n_ahead`, and the scores equal to `cut` into `n_eq`.
 5. Set `keep_ties` to `n_ahead + n_eq == k`. The tied block at the cut fits in `k` only then. `cut` is at position `k`, so `n_ahead + n_eq >= k` is always true.
 6. Return the mask that is `true` for every score strictly `ahead` of `cut`, and for a score equal to `cut` when `keep_ties` is `true`.

# Arguments

  - `scores`: The score vector `assets × 1`, with one score for each asset.
  - `k`: The number of assets to take from the `tail` end.
  - `bib`: The [`bigger_is_better`](@ref) flag of the score.
  - `tail`: The end to take from, `:best` or `:worst`.

# Returns

  - `keep::BitVector`: The mask `assets × 1`. It is `true` for every asset in the tail. It holds `k` assets or fewer, and fewer when a tied block crosses the cut.

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

Turn the `scores` of the assets into a keep-mask under a selection rule.

`bib` is the [`bigger_is_better`](@ref) flag of the score that gave `scores`. [`ThresholdRule`](@ref) ignores it.

There is one method for each rule. The [`ThresholdRule`](@ref) method compares each score with the open band, and returns the mask. The [`RankRule`](@ref) method gives its two counts to [`tail_action_mask`](@ref). The [`QuantileRule`](@ref) method first converts each fraction to a count with `round(Int, f * n, RoundNearestTiesUp)`. Then it gives the two counts to the same function.

# Arguments

  - `rule`: The selection rule.
  - `scores`: The score vector `assets × 1`, with one score for each asset.
  - `bib`: The [`bigger_is_better`](@ref) flag of the score that gave `scores`.

# Returns

  - `keep::BitVector`: The mask `assets × 1`. It is `true` for every asset that the rule admits.

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

Make the union of the two tail masks, and apply the `action` of the rule.

# Algorithm

 1. Start from a mask of `length(scores)` falses.
 2. When `best` is not `nothing`, take the `:best` tail of size `best` with [`tail_mask`](@ref), and add it to the mask with a union.
 3. When `worst` is not `nothing`, take the `:worst` tail of size `worst` with [`tail_mask`](@ref), and add it to the mask with a union.
 4. Return the mask when `action` is `:keep`, and its complement when `action` is `:drop`.

# Arguments

  - `best`: The number of assets to take from the best end, or `nothing`.
  - `worst`: The number of assets to take from the worst end, or `nothing`.
  - `action`: `:keep` returns the union, and `:drop` returns its complement.
  - `scores`: The score vector `assets × 1`, with one score for each asset.
  - `bib`: The [`bigger_is_better`](@ref) flag of the score.

# Returns

  - `keep::BitVector`: The mask `assets × 1`. It is `true` for every asset that the rule admits.

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

Check that `score` can take the return series of one asset.

The score of asset `i` is `score(X[:, i])`. This is the route of a precomputed return vector, and [`supports_precomputed_returns`](@ref) controls it. A [`WeightsInput`](@ref) measure, such as [`Variance`](@ref) or [`StandardDeviation`](@ref), takes portfolio weights in place of a return vector. It cannot score an asset.

# Arguments

  - `score`: The risk measure that a selector uses to score its assets.

# Validation

  - `supports_precomputed_returns(score)`. Otherwise the function throws an `ArgumentError` that names the measure. When the measure is a [`Variance`](@ref) or a [`StandardDeviation`](@ref), the message also names `SCM()`, which computes the same quantity from a return series. When it is a [`Skewness`](@ref), the message names [`ThirdCentralMoment`](@ref), and when it is a [`Kurtosis`](@ref), it names [`HighOrderMoment`](@ref) and [`LowOrderMoment`](@ref).

# Returns

  - `nothing`.

# Related

  - [`ScoreSelector`](@ref)
  - [`supports_precomputed_returns`](@ref)
"""
function assert_scoreable(score::AbstractBaseRiskMeasure)::Nothing
    if !supports_precomputed_returns(score)
        hint = if isa(score, Union{Variance, StandardDeviation})
            " `SCM()` computes the same quantity from a return series and is scoreable."
        elseif isa(score, Skewness)
            " `ThirdCentralMoment()` computes the third moment from a return series, and `ThirdCentralMoment(; ve = SimpleVariance())` the standardised skewness. Both are scoreable."
        elseif isa(score, Kurtosis)
            " `HighOrderMoment(; alg = FourthMoment())` computes the fourth moment from a return series, and `LowOrderMoment(; alg = EvenMoment())` its square root. Both are scoreable."
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

The function gives each column to `score` as a view. The functor of a risk measure reads its argument, and does not write to it.

# Algorithm

 1. Evaluate `score` on a view of each asset column of `X`, one column after the other, into `scores`.
 2. Check that every entry of `scores` is finite.
 3. Return `scores`.

# Arguments

  - `score`: The risk measure to evaluate on each asset column.
  - $(arg_dict[:X])

# Validation

  - Every score is finite. Otherwise the function throws a `DomainError` that names the columns with a non-finite score. [`ThirdCentralMoment`](@ref) with a variance estimator in `ve`, on a constant series for example, divides by a zero standard deviation and gives `NaN`. A `NaN` score has no place in an order, so the function throws and does not sort the scores arbitrarily.

# Returns

  - `scores::VecNum`: The score vector `assets × 1`, with one entry for each asset column of `X`.

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

Asset selector that scores every asset with a risk measure, and keeps the assets that a rule admits.

`score` is an [`AbstractBaseRiskMeasure`](@ref) that can take the return series of one asset. The score of asset `i` is `score(X[:, i])`. The selector has no score type of its own, and it uses the risk measures of the library. [`ConditionalValueatRisk`](@ref) and the drawdown measures score risk, `SCM()` scores variance, and [`MeanReturn`](@ref) scores the mean return. [`bigger_is_better`](@ref) tells the ordinal rules which end is the best.

`rule` sets what the selector does with the scores. A [`ThresholdRule`](@ref) keeps an absolute band. A [`RankRule`](@ref) or a [`QuantileRule`](@ref) takes a count or a fraction from the tails.

A `ScoreSelector` is safe in cross-validation. It fits its universe on the training window, and every test window uses that universe.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ScoreSelector(;
        score::AbstractBaseRiskMeasure,
        rule::AbstractSelectionRule,
    ) -> ScoreSelector

Keywords correspond to the struct's fields.

## Validation

  - `supports_precomputed_returns(score)`. [`Variance`](@ref) and [`StandardDeviation`](@ref) are [`WeightsInput`](@ref) measures. The constructor refuses them, and its message names `SCM()`.

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
    The [`AbstractBaseRiskMeasure`](@ref) that scores the return series of each asset.
    """
    score
    """
    The [`AbstractSelectionRule`](@ref) that turns the scores into a keep-mask.
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

Select the assets that a [`ScoreSelector`](@ref) keeps. The method scores every asset, then it applies the rule.

[`fit_preprocessing`](@ref) first reduces the training window to its Coverage Universe. The method ranks only the live assets, and a dead column has no place in the order. [`asset_scores`](@ref) still refuses a non-finite score. A non-finite score from a live column is a defect of the measure, not a gap in the data.

# Algorithm

 1. Score every asset column of `rd.X` with the `score` of the selector, through [`asset_scores`](@ref).
 2. Read the orientation of the score with [`bigger_is_better`](@ref).
 3. Return the keep-mask that [`rule_keep`](@ref) gives for those scores under the `rule` of the selector.

# Arguments

  - `sel`: The score selector.
  - $(arg_dict[:rd])

# Returns

  - `keep::BitVector`: The mask `assets × 1`. It is `true` for every asset that the rule admits.

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

Asset selector that keeps the Coverage Universe of the training window, and drops every other asset column.

It does at the returns level what the column threshold of [`MissingDataFilter`](@ref) does at the price level. A pipeline that starts from returns data, and so runs no price stage, uses it. It cannot drop observations. A fitted selector cannot select the rows of a new window to drop, because the weights and the returns then no longer align.

The library fits every selector of the family on the Coverage Universe. This selector is then the **identity** on the window that it gets. It is the explicit step that asks for the reduction, and for nothing more. The reduction drops an asset whose return is non-finite at a row of the window. It also drops an asset that the [`AssetPanel`](@ref) marks inactive at a row of the window.

A [`ReturnsResult`](@ref) bounds the type of `X` to a matrix of numbers, so a `missing` never gets to this selector. The constructor of the [`ReturnsResult`](@ref) refuses a `Matrix{Union{Missing, Float64}}`. [`MissingDataFilter`](@ref) removes a `missing` from the price data at an earlier stage.

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

Select the assets that a [`CompleteAssetSelector`](@ref) keeps. These are **all** the asset columns of the reduced window.

This selector is the identity on the Coverage Universe. [`fit_preprocessing`](@ref) reduces the training window before it calls this method. Each column that this method gets is finite at every row, and active at every row of the [`AssetPanel`](@ref). `fit_preprocessing` drops the other columns, and this selector is the explicit step that asks for the drop.

The reduction also reads the active mask of the panel. An asset with an old finite price in an inactive period is out of the universe. The released versions of this selector read only the finiteness, and kept that asset.

# Algorithm

 1. Return a keep-mask of `size(rd.X, 2)` trues.

# Arguments

  - The method takes the selector by type alone. The selector has no field, so the method reads nothing from it.
  - $(arg_dict[:rd])

# Returns

  - `keep::BitVector`: The mask `assets × 1`. It is `true` at every column of the reduced window.

# Related

  - [`CompleteAssetSelector`](@ref)
  - [`fit_preprocessing`](@ref)
  - [`coverage_mask`](@ref)
"""
function select_assets(::CompleteAssetSelector, rd::AbstractReturnsResult)::BitVector
    return trues(size(rd.X, 2))
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the algorithms that select which assets a [`RedundancySelector`](@ref) drops as redundant.

Each algorithm takes the data, and a score for each asset when the selector has one. It returns a keep-mask of the columns that survive. The interface returns a keep-mask, not a partition into groups. [`PairwiseCorrelation`](@ref) drops one asset at a time, and it can keep two members of one correlated group. A partition with one survivor for each group cannot show that selection.

Two algorithms do make a partition, and both use [`groups_argbest`](@ref):

  - [`CorrelationComponents`](@ref) makes a group of each connected component of the graph whose edges are the correlations at or above the threshold.
  - [`ClusterGroups`](@ref) makes a group of each cluster that [`clusterise`](@ref) assigns.

# Interfaces

To add a redundancy algorithm, subtype `AbstractRedundancyAlgorithm` and add these methods:

  - `redundancy_keep(alg::MyAlgorithm, rd, scores, bib) -> BitVector`.
  - `requires_score(::MyAlgorithm) -> Bool`, only when the algorithm can select a survivor with no score. The generic method returns `true`, so this method returns `false`.

## Arguments

  - `alg`: The concrete redundancy algorithm.
  - $(arg_dict[:rd])
  - `scores`: The score vector `assets × 1`, with one score for each asset. It is `nothing` when the selector has no `score`.
  - `bib`: The [`bigger_is_better`](@ref) flag of the score. It is `false` when `scores` is `nothing`.

## Returns

  - `keep::BitVector`: The mask `assets × 1`. It is `true` for every asset that survives.

# Related

  - [`RedundancySelector`](@ref)
  - [`PairwiseCorrelation`](@ref)
  - [`CorrelationComponents`](@ref)
  - [`ClusterGroups`](@ref)
"""
abstract type AbstractRedundancyAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether a redundancy algorithm needs a `score` to select the survivor of a redundancy group.

The correlation algorithms return `false`. With no score, they use a summary of the correlation of each asset with the rest of the universe. [`ClusterGroups`](@ref) has no such fallback, and it returns `true`.

The generic method returns `true`, so a [`RedundancySelector`](@ref) asks a new algorithm for a score until the algorithm declares otherwise. This is the safe default. Without it, an algorithm with no fallback can get `nothing` in place of a score vector. It then has no data to select a survivor from.

# Arguments

  - The method takes the algorithm by type alone. It reads no field of it.

# Returns

  - `req::Bool`: `true` when the algorithm cannot select the survivor of a group without a score.

# Related

  - [`AbstractRedundancyAlgorithm`](@ref)
  - [`RedundancySelector`](@ref)
"""
requires_score(::AbstractRedundancyAlgorithm)::Bool = true
"""
    redundancy_keep(alg::AbstractRedundancyAlgorithm, rd, scores, bib) -> BitVector

Return the keep-mask that a redundancy algorithm admits.

`scores` is `nothing` when the [`RedundancySelector`](@ref) has no `score`. Otherwise `scores` holds one score for each asset, and `bib` is the [`bigger_is_better`](@ref) flag of the score.

This method is the fallback of the family. It runs only for an algorithm that has no method of its own, and it always throws.

# Arguments

  - `alg`: The redundancy algorithm.
  - $(arg_dict[:rd])
  - `scores`: The score vector `assets × 1`, or `nothing`.
  - `bib`: The [`bigger_is_better`](@ref) flag of the score.

# Validation

  - The concrete algorithm implements `redundancy_keep`. Otherwise this method throws an `ArgumentError` that names the two methods of the interface.

# Returns

  - `keep::BitVector`: The mask `assets × 1`. It is `true` for every asset that survives.

# Related

  - [`AbstractRedundancyAlgorithm`](@ref)
  - [`RedundancySelector`](@ref)
"""
function redundancy_keep(alg::AbstractRedundancyAlgorithm, ::AbstractReturnsResult,
                         ::Option{<:VecNum}, ::Bool)
    return throw(ArgumentError("$(typeof(alg)) subtypes AbstractRedundancyAlgorithm but does not implement redundancy_keep. Extension authors: a redundancy algorithm must implement redundancy_keep(alg, rd, scores, bib) -> BitVector returning the keep-mask, and requires_score(alg) -> Bool returning false when it can pick a survivor without a score."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Convert the scores of the assets into *drop scores*. Of two assets, the algorithm drops the one with the higher drop score.

When `bib` is `false`, a lower score is better, and the score is already a drop score. When `bib` is `true`, the function negates the score. After this call a **lower** number is always better, whatever the orientation of the measure.

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

  - ``d_{i}``: The drop score of asset ``i``. Of two assets, the algorithm drops the one with the higher drop score.
  - $(math_dict[:s_i_score])

The map reverses the order in the first case, and keeps it in the second. It never changes which of two assets is better. It changes only the direction of the order.

# Arguments

  - `scores`: The score vector `assets × 1`, with one score for each asset.
  - `bib`: The [`bigger_is_better`](@ref) flag of the score that gave `scores`.

# Returns

  - `d::VecNum`: The drop score vector `assets × 1`. Of two assets, the algorithm drops the one with the higher number.

# Related

  - [`RedundancySelector`](@ref)
  - [`find_uncorrelated_indices`](@ref)
  - [`bigger_is_better`](@ref)
"""
drop_scores(scores::VecNum, bib::Bool) = bib ? -scores : scores
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Keep the member with the best score in each redundancy group.

A group whose best score is *tied* keeps no asset. This is the tie policy of the library: if the library cannot tell two assets apart, it keeps neither of them. [`find_uncorrelated_indices`](@ref) does the same with a correlated pair whose drop scores are equal. A group with one member has no tie, so its member always survives.

# Algorithm

 1. Start from a keep-mask of `length(scores)` falses.
 2. Select the comparison `better`. It is `>` when `bib` is `true`, and `<` otherwise.
 3. Take the next group `g` of `groups`. Skip it when it is empty.
 4. Read the members of `g` in order from the first. Keep in `best` the index whose score no later member beats under `better`.
 5. Count the members of `g` whose score is equal to `scores[best]`. Set the mask at `best` only when that count is one, so a group with a tied best keeps no asset.
 6. Do steps 3 to 5 for each of the other groups, then return the mask.

# Arguments

  - `groups`: The vector of index vectors that partition the assets.
  - `scores`: The score vector, with one score for each asset.
  - `bib`: `true` when a larger score is better.

# Returns

  - `keep::BitVector`: The keep-mask, with one survivor for each group whose best score is not tied.

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

Greedy removal of correlated assets. The algorithm drops assets until no pair of the kept assets has a correlation of `t` or more.

The algorithm reads the correlated pairs from the most to the least correlated, and it removes the worse asset of each pair. The worse asset has the higher drop score. The drop score comes from the `score` of the [`RedundancySelector`](@ref) when it has one. Otherwise it is the summary of the correlation of each asset with the rest of the universe. Then the asset that is redundant with *most* of the universe goes first.

This algorithm does not **chain**. Take `t = 0.7`, and a universe with `ρ(A, B) = 0.80`, `ρ(B, C) = 0.81` and `ρ(A, C) = 0.32`. With the default drop score, the algorithm removes `B` and keeps both `A` and `C`, because their correlation is below `t`. [`CorrelationComponents`](@ref) reads the same three correlations as one chain, and keeps only `A`.

The two edges of a chain set a lower bound on the correlation of its end points. Two edges at `ρ` give the end points a correlation of `ρ² - (1 - ρ²)` or more. Thus a chain of two edges at `0.97` cannot have end points with a correlation below `0.88`. A chain whose end points have a low correlation needs weak edges, and the two algorithms differ most when `t` is just below those edges.

The algorithm calls [`find_uncorrelated_indices`](@ref).

# Mathematical definition

```math
\\begin{align}
\\rho_{i,\\,j} &< t \\quad \\forall\\, i \\neq j \\in \\mathcal{K}\\,.
\\end{align}
```

Where:

  - $(math_dict[:rho_ij]) The algorithm reads the absolute value ``\\left| \\rho_{i,\\,j} \\right|`` when `absolute` is `true`.
  - $(math_dict[:t_corr_threshold])
  - $(math_dict[:K_keep_set])

The condition is about the **pairs** of kept assets, and about nothing else. It does not say that ``\\mathcal{K}`` is the largest such set. It is also not transitive. A chain of two edges at or above the threshold, whose end points are below the threshold, meets the condition with both end points kept.

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

  - `-1 <= t <= 1`. [`assert_correlation_threshold`](@ref) checks it.

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

Keep the assets that survive [`PairwiseCorrelation`](@ref). No pair of them has a correlation of `t` or more.

The method calls [`find_uncorrelated_indices`](@ref). A greedy search gives the set of kept assets, so it is not always the largest such set.

# Algorithm

 1. Start from a keep-mask of `size(rd.X, 2)` falses.
 2. Turn `scores` into drop scores with [`drop_scores`](@ref), so that after this step a **lower** number is better. When `scores` is `nothing`, pass `nothing`. [`find_uncorrelated_indices`](@ref) then collapses each column of the correlation matrix with `measure` into its own drop score.
 3. Call [`find_uncorrelated_indices`](@ref) on `rd.X` with the `ce`, `t`, `absolute` and `measure` of the algorithm. It returns `idx`, the indices of the assets that survive.
 4. Set the mask at `idx`, and return it.

# Arguments

  - `alg`: The pairwise correlation algorithm.
  - $(arg_dict[:rd])
  - `scores`: The score vector `assets × 1`, or `nothing`.
  - `bib`: The [`bigger_is_better`](@ref) flag of the score.

# Returns

  - `keep::BitVector`: The mask `assets × 1`. It is `true` for every asset that survives.

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

Group the assets by the connected components of a correlation graph, and keep the member of each group with the best score.

Two assets share an edge when their correlation is at or above `t`. When `absolute` is `true`, the algorithm compares the absolute value of the correlation. A component is transitive, so the algorithm reads a chain `A ~ B ~ C` as one redundant group, also when `A` and `C` are not correlated. It keeps one asset of the group. This claim is stronger than that of [`PairwiseCorrelation`](@ref), and it removes more assets. Use it to keep one representative of each correlated group, and not to bound the correlation of the pairs that survive.

A component whose best score is tied keeps no asset. [`groups_argbest`](@ref) states the tie policy.

# Mathematical definition

```math
\\begin{align}
\\mathcal{E} &= \\left\\{ \\left\\{ i,\\, j \\right\\} : i \\neq j\\,,\\; \\rho_{i,\\,j} \\geq t \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathcal{E}``: The edge set of the redundancy graph, whose vertices are the assets.
  - $(math_dict[:rho_ij]) The algorithm reads the absolute value ``\\left| \\rho_{i,\\,j} \\right|`` when `absolute` is `true`.
  - $(math_dict[:t_corr_threshold])

The groups are the connected components of that graph, and every asset is in exactly one group. An asset with no partner at or above the threshold is a group of one. A component is transitive, but the edge relation is not. This is the difference from [`PairwiseCorrelation`](@ref). A chain ``i \\sim j \\sim k`` is one component, whatever the value of ``\\rho_{i,\\,k}``.

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

  - `-1 <= t <= 1`. [`assert_correlation_threshold`](@ref) checks it.

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
    $(field_dict[:pre_measure]) A lower value is better, so the representative that survives is the least redundant member of its component.
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

Keep one representative of each correlated group under [`CorrelationComponents`](@ref).

# Algorithm

 1. Compute the correlation matrix `rho` of `rd.X` with the `ce` of the algorithm.
 2. When `absolute` is `true`, replace each entry of `rho` with its absolute value. This step comes **before** the threshold, so a strongly negative correlation is an edge.
 3. When `scores` is `nothing`, collapse each column of `rho` with `measure` into the fallback score `s`, and set `sbib` to `false`. A lower summary correlation is then better, so the representative that survives is the asset least correlated with the rest of the universe. Otherwise take `s` from `scores` and `sbib` from `bib`.
 4. Group the assets with [`correlation_components`](@ref) on `rho` and `t`.
 5. Return the mask that [`groups_argbest`](@ref) admits for those groups under `s` and `sbib`.

# Arguments

  - `alg`: The correlation components algorithm.
  - $(arg_dict[:rd])
  - `scores`: The score vector `assets × 1`, or `nothing`.
  - `bib`: The [`bigger_is_better`](@ref) flag of the score.

# Returns

  - `keep::BitVector`: The mask `assets × 1`. It is `true` for one asset of each component whose best score is not tied.

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

The function does one union-find pass over the strict lower triangle. The components are transitive, and every asset is in exactly one of them. An asset with no partner at or above `t` is a component of one.

# Algorithm

 1. Read `n`, the number of rows of `rho`, and set `parent[i]` to `i` for every asset. Each asset starts as its own component.
 2. Take the next entry of the strict lower triangle of `rho`, which holds each pair one time.
 3. When that entry is at least `t`, find the root of each member of the pair with `find`. Point the larger root at the smaller one. `find` compresses the path that it reads, so a later lookup on that chain takes one step.
 4. Do steps 2 and 3 for each of the other pairs.
 5. Group the assets by their last root into `groups`, with that root as the key.
 6. Return the values of `groups`.

The function compares the threshold with `rho` as the caller gives it. A caller that wants the absolute value takes it before this call.

# Arguments

  - `rho`: The correlation matrix `assets × assets`.
  - $(arg_dict[:t])

# Returns

  - `groups::Vector{Vector{Int}}`: The components. Each is a vector of asset indices in ascending order. The components come from a `Dict`, so their order is not defined. [`groups_argbest`](@ref) reads them as a set, and does not depend on the order.

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

Group the assets into clusters, and keep the member of each cluster with the best score.

[`clusterise`](@ref) makes the clusters, so every clustering estimator of the library can set what "redundant" means. This includes hierarchical linkage, DBHT, the non-hierarchical algorithms and the estimators of the optimal number of clusters. The correlation algorithms have a fallback rule for the survivor, but this algorithm has none. A [`RedundancySelector`](@ref) with `ClusterGroups` must have a `score`.

A cluster whose best score is tied keeps no asset. [`groups_argbest`](@ref) states the tie policy.

## Clusters on a Feature Matrix

A [`FeatureDistance`](@ref) in the distance field of `cle` measures a Feature Matrix in place of the returns. It makes that matrix from the [`AssetPanel`](@ref) on `rd`, the returns data that the selector fits on.

Preselection runs before the prior, so it gives `rd` alone. [`fit_preprocessing`](@ref) fits a selector from the returns data alone, and the selector never gets a prior result. In a [`Pipeline`](@ref), the selector writes `:returns`, and this invalidates a `:prior` that an earlier step computed. Thus a panel producer that needs a prior raises an error here, and the error names this site. Put an [`AssetPanel`](@ref) on the [`ReturnsResult`](@ref), for example from [`asset_panel`](@ref). Otherwise [`clusterise`](@ref) throws.

The `cle` reads `rd.X`, the returns as the selector gets them, because a selector fits before a prior or a benchmark frame exists. For this reason, the `cle` states no source, and the constructor refuses a `cle` whose `x_src` is not `nothing`.

The algorithm selects on the *full* universe, and the library slices the columns that survive after the selection. The library measures the Feature Matrix `Z` over every asset, before the selector drops one.

The Feature Matrix takes its column names from the same panel as its values. Thus a [`FeatureDistance`](@ref) with a `sel` selects the same feature columns here as at every other site. Without a `sel`, the Feature Matrix holds the value columns of every Panel Field, with one column for each level of a categorical field. A redundancy selector on all of these columns can drop assets on a distance that the caller did not intend.

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
  - [`asset_panel`](@ref)
"""
@concrete struct ClusterGroups <: AbstractRedundancyAlgorithm
    """
    The [`AbstractClustersEstimator`](@ref) that partitions the assets.
    """
    cle
    function ClusterGroups(cle::AbstractClustersEstimator)
        assert_returns_source(returns_source(cle), (nothing,), :ClusterGroups, "cle.x_src")
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

 1. Cluster the assets with [`clusterise`](@ref) on `rd.X` into the clustering result `clr`. Give `rd` too, so that a [`FeatureDistance`](@ref) reads its panel from `rd`.
 2. Read the cluster assignment of every asset into `idx`.
 3. Collect the asset indices of each of the `clr.k` clusters into `groups`.
 4. Return the mask that [`groups_argbest`](@ref) admits for those groups under `scores` and `bib`.

The method gives `rd` alone, because preselection runs before a prior exists. At this site, `rd` is the only source of a Feature Matrix. [`ClusterGroups`](@ref) states why the type has no source field.

# Arguments

  - `alg`: The cluster groups algorithm.
  - $(arg_dict[:rd])
  - `scores`: The score vector `assets × 1`. The constructor of [`RedundancySelector`](@ref) refuses a `nothing` score for this algorithm, so `scores` is never `nothing` here.
  - `bib`: The [`bigger_is_better`](@ref) flag of the score.

# Returns

  - `keep::BitVector`: The mask `assets × 1`. It is `true` for one asset of each cluster whose best score is not tied.

# Related

  - [`ClusterGroups`](@ref)
  - [`clusterise`](@ref)
  - [`groups_argbest`](@ref)
"""
function redundancy_keep(alg::ClusterGroups, rd::AbstractReturnsResult,
                         scores::Option{<:VecNum}, bib::Bool)::BitVector
    clr = clusterise(alg.cle, rd.X; rd = rd)
    idx = assignments(clr)
    groups = [findall(==(k), idx) for k in 1:(clr.k)]
    return groups_argbest(groups, scores, bib)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Check a correlation threshold.

# Arguments

  - $(arg_dict[:t])

# Validation

  - `-1 <= t <= 1`. Otherwise the function throws a `DomainError`. The bound is the range of a correlation coefficient. A correlation never gets to a threshold above the range, and it is always at or above a threshold below the range.

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

Asset selector that drops the assets whose information other assets already carry.

`alg` sets what "redundant" means, and it returns the keep-mask. [`PairwiseCorrelation`](@ref) prunes correlated pairs greedily. [`CorrelationComponents`](@ref) keeps one representative of each correlated group. [`ClusterGroups`](@ref) keeps one representative of each cluster.

`score` sets *which* asset of a redundancy group survives. It is a risk measure on the return series of each asset, and [`bigger_is_better`](@ref) sets its order, as in [`ScoreSelector`](@ref). When `score` is `nothing`, a correlation algorithm uses its own rule. The asset with the lower summary correlation with the rest of the universe survives. [`ClusterGroups`](@ref) has no such rule, and it needs a `score`.

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

`A` and `B` are identical, so the tie policy drops both of them.

# Related

  - [`AbstractAssetSelector`](@ref)
  - [`AbstractRedundancyAlgorithm`](@ref)
  - [`ScoreSelector`](@ref)
"""
@concrete struct RedundancySelector <: AbstractAssetSelector
    """
    The [`AbstractRedundancyAlgorithm`](@ref) that selects the redundant assets.
    """
    alg
    """
    The [`AbstractBaseRiskMeasure`](@ref) that selects the survivor of each redundancy group. `nothing` uses the rule of the algorithm.
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

Select the assets that a [`RedundancySelector`](@ref) keeps. The method scores every asset when the selector has a score, then it applies the redundancy algorithm.

[`fit_preprocessing`](@ref) first reduces the training window to its Coverage Universe. The correlation, the clusters and the survivor rule then read only the live columns. A dead column is never in a group, and it never survives one.

# Algorithm

 1. When the selector has no `score`, set `scores` to `nothing` and `bib` to `false`. The algorithm then uses its own survivor rule.
 2. Otherwise score every asset column of `rd.X` with [`asset_scores`](@ref), and read `bib` from [`bigger_is_better`](@ref) of that score.
 3. Return the keep-mask that [`redundancy_keep`](@ref) admits for the `alg` of the selector.

# Arguments

  - `sel`: The redundancy selector.
  - $(arg_dict[:rd])

# Returns

  - `keep::BitVector`: The mask `assets × 1`. It is `true` for every asset that survives.

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
public AbstractSelectionRule, AbstractRedundancyAlgorithm, select_assets, asset_scores,
       rule_keep, redundancy_keep, requires_score
