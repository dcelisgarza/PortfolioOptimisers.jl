"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the rules for an entry that the data do not state, in a factor attribution and in a union of graphs.

The family has two jobs. The first job is the rule that a factor attribution applies to an entry the factor model does not state. The second job is the rule that a [`GraphUnion`](@ref) and [`graph_union`](@ref) apply to a part that has no graph in force, before its first key or at a key that holds `nothing`.

  - [`EntrywiseUnknown`](@ref): the union is unknown when any part is unknown, and its keys start at the latest first key of the parts.
  - [`ZeroUnknown`](@ref): a part with no graph in force reads as an empty graph, the union is unknown only when every part is unknown, and its keys start at the earliest first key of the parts.
  - [`KindwiseUnknown`](@ref) has no method of the second job, because its kinds `unstated` and `leverage` do not apply to a graph. It raises a `MethodError` there.

## The first job: a factor attribution

A Prior Estimator states no idiosyncratic variance for an asset in the warm-up of its variance, and no loadings for an asset that delisted. A portfolio that holds such an asset has a variance that reads the unknown entry, so the attribution must say what it does with it. Each concrete rule is one answer, and [`factor_attribution`](@ref) takes it through the keyword `unknown`.

  - [`EntrywiseUnknown`](@ref) keeps every entry the model states, and gives `NaN` for every number that reads an entry it does not state. It is the default.
  - [`ZeroUnknown`](@ref) reads every entry the model does not state as zero, and an absent return as a zero return.
  - [`KindwiseUnknown`](@ref) applies one rule to each kind of unknown entry.

The rule applies on the predicted side to every number, and on both sides to the standalone moments of the asset axis. An entry of an asset the portfolio does not hold has a weight of zero, so it adds nothing to a sum under either rule.

A Leverage-One Pair is a second kind of unknown entry. The fit reproduced the return of the pair, so its total is exact and the split between its systematic and its idiosyncratic part is unknown. This is the opposite of an asset in the warm-up of its variance, whose split is exact up to its unknown idiosyncratic part. The rule applies to this kind on both sides, through [`attribution_leverage_split`](@ref).

# Interfaces

In order to implement a new rule for an unknown entry which will work seamlessly with the library, subtype `AbstractUnknownEntryRule` and implement the following six methods. The first four apply the rule to an entry the factor model does not state, and the last two apply it to the variance split of a Leverage-One Pair. A rule that implements all six applies to every kind, and it can also be a field of [`KindwiseUnknown`](@ref).

## The held assets that keep their unknown entries

  - `attribution_unknown_keep(rule::MyRule, w::VecNum) -> BitVector`: Returns the assets whose unknown entries stay unknown in a predicted attribution.

### Arguments

  - `rule`: The rule.
  - `w`: Portfolio weights.

### Returns

  - `keep::BitVector`: `true` at every asset whose unknown entries stay `NaN`. An asset with a weight of zero adds nothing to a sum, so the attribution reads its unknown entries as zero whatever the rule returns.

## The standalone moments of the predicted asset axis

  - `attribution_standalone(rule::MyRule, A::AbstractArray) -> AbstractArray`: Returns the entries of a block that the standalone volatility and mean of each asset read.

### Arguments

  - `rule`: The rule.
  - `A`: The loadings, the factor-orthogonal mean or the idiosyncratic block.

### Returns

  - `A::AbstractArray`: The entries the standalone moments read, the shape of `A`.

## The observations of the realised standalone moments

  - `attribution_standalone_pairs(rule::MyRule, act::AbstractVector{Bool}) -> AbstractVector{Bool}`: Returns the observations that the realised standalone moments of one asset read.

### Arguments

  - `rule`: The rule.
  - `act`: Whether each aligned pair of the asset is active.

### Returns

  - `a::AbstractVector{Bool}`: The observations the moments read, the length of `act`.

## The note of the report

  - `attribution_unknown_note(rule::MyRule) -> String`: Returns the sentence that a predicted attribution adds to the report of a held asset that is not investable.

### Arguments

  - `rule`: The rule.

### Returns

  - `note::String`: The sentence. It states what the rule does with the unknown entries, so that the reader knows which numbers to trust.

## The marks of a Leverage-One Pair

  - `attribution_leverage_split(rule::MyRule, lev::Option{<:AbstractVector{Bool}}, w::VecNum, M::MatNum) -> Option{<:LeverageSplitResult}`: Returns the marks of a predicted attribution.
  - `attribution_leverage_split(rule::MyRule, h1::Option{<:AbstractMatrix{Bool}}, W::VecNum_MatNum, B::MatNum_Arr3Num) -> Option{<:LeverageSplitResult}`: Returns the marks of a realised attribution.

### Arguments

  - `rule`: The rule.
  - `lev`: Assets the leverage-one mask marks at one row or more, or `nothing` when the model marks no pair.
  - `h1`: The leverage-one mask, `observations × assets`, or `nothing` when the model marks no pair.
  - `w`: Portfolio weights.
  - `W`: Portfolio weights, or the weight history, `observations × assets`.
  - `M`: Loadings, `assets × factors`.
  - `B`: Loadings, or the exposure history, `assets × factors × observations`. Earlier releases held the observations of the history on its first axis. A method written for them reads `B[t, :, :]`, and it must now read `B[:, :, t]`, or the slice that `attribution_slice(B, t)` gives.

### Returns

  - `lv::Option{<:LeverageSplitResult}`: `nothing` when the rule marks nothing. Otherwise a [`LeverageSplitResult`](@ref) with the asset mask `assets` and the factor mask `factors`, whose constructor refuses marks that name no asset. `attribution_leverage_nan` puts `NaN` in every variance number that the marks read.

## The standard errors of a Leverage-One Pair

  - `attribution_leverage_errors(rule::MyRule, h1::Option{<:AbstractMatrix{Bool}}, reg::AbstractMatrix{Bool}, ses::AttributionErrorsResult, pass) -> AttributionErrorsResult`: Returns the standard errors of the mean return contributions after the rule.

### Arguments

  - `rule`: The rule.
  - `h1`: The leverage-one mask, `observations × assets`, or `nothing` when the model marks no pair.
  - `reg`: Whether each pair is in the regression, `observations × assets`.
  - `ses`: The standard errors, an [`AttributionErrorsResult`](@ref) with the fields `sys`, `factor` and `family`.
  - `pass`: The reduction of `attribution_error_pass`, a function of a matrix of variances, `observations × assets`, which returns an [`AttributionErrorsResult`](@ref).

### Returns

  - `ses::AttributionErrorsResult`: The standard errors after the rule. A method that changes an error builds a new [`AttributionErrorsResult`](@ref), whose constructor checks that the three answers agree on what they state.

## The parts of a union of graphs

A rule that implements the following method also applies to a union of graphs, the second job.

  - `union_parts_in_force(rule::MyRule, known::AbstractVector{Bool}) -> Option{<:AbstractVector{<:Integer}}`: Returns the positions of the parts that a union sums, or `nothing` when the union is unknown.

### Arguments

  - `rule`: The rule.
  - `known`: Whether each part has a graph in force at a row.

### Returns

  - `idx::Option{<:AbstractVector{<:Integer}}`: The positions of the parts that the union sums, each of them `known`, or `nothing` when no graph of the union is in force. The first key of a union series is the earliest first key of a part at which the rule returns positions, so the same method states the time rule.

# Related

  - [`EntrywiseUnknown`](@ref)
  - [`ZeroUnknown`](@ref)
  - [`KindwiseUnknown`](@ref)
  - [`factor_attribution`](@ref)
  - [`GraphUnion`](@ref)
  - [`graph_union`](@ref)
"""
abstract type AbstractUnknownEntryRule <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Keeps every entry the factor model states, and gives `NaN` for every number that reads an entry it does not state.

It is the default rule of [`factor_attribution`](@ref). A held asset in the warm-up of its variance keeps its loadings, so the exposures, the systematic variance and the systematic mean are exact. Its idiosyncratic variance is unknown, so the portfolio volatility is unknown, and with it the idiosyncratic part, the total, the remainder and every number divided by the portfolio volatility. A held asset without loadings makes the exposures that read them unknown too. On the asset axis the standalone moments read the entries of the asset alone, and on the realised side its active pairs alone.

A portfolio that holds a Leverage-One Pair has an exact total and an unknown split. The volatility, the volatility contribution, the variance share and the correlation of the systematic and the idiosyncratic component are `NaN`, and so are those of the factor whose only member is the pair, and the systematic and idiosyncratic volatility contributions of the asset. The total, the remainder and every mean keep their values.

In a union of graphs, it is the default rule of [`GraphUnion`](@ref) and [`graph_union`](@ref). The union is in force at a row only when every part has a graph in force there. So the keys of a union series start at the latest first key of its parts, and a row before that key comes before the first key of the union. At a later key, the union is `nothing` when any part has `nothing` in force.

# Related

  - [`AbstractUnknownEntryRule`](@ref)
  - [`ZeroUnknown`](@ref)
  - [`KindwiseUnknown`](@ref)
  - [`factor_attribution`](@ref)
  - [`GraphUnion`](@ref)
"""
struct EntrywiseUnknown <: AbstractUnknownEntryRule end
"""
$(DocStringExtensions.TYPEDEF)

Reads every entry the factor model does not state as zero, and an absent return as a zero return.

The rule is a stated assumption, not a fact of the model. An unknown idiosyncratic variance read as zero makes the portfolio variance a lower bound, so it understates the volatility and every share divided by it. It understates the standard error of a realised mean contribution in the same way. A missing return read as zero dilutes the mean of an asset by the observations it was absent. A total moment the prior does not state reads the model, with every unknown entry of the model read as zero, so the gap of the remainder reads the stated entries alone.

A Leverage-One Pair keeps the split as fitted: its idiosyncratic variance is zero, so the attribution reports a held pair as wholly systematic. This understates the idiosyncratic part by the unknown own variance of the asset, and the factor whose only member is the pair carries that variance.

In a union of graphs, a part with no graph in force reads as an empty graph. The union is unknown only when every part is unknown, so the keys of a union series start at the earliest first key of its parts. The rule is a stated assumption too: a path through the relation of the missing part is lost.

# Related

  - [`AbstractUnknownEntryRule`](@ref)
  - [`EntrywiseUnknown`](@ref)
  - [`KindwiseUnknown`](@ref)
  - [`factor_attribution`](@ref)
  - [`GraphUnion`](@ref)
"""
struct ZeroUnknown <: AbstractUnknownEntryRule end
"""
$(DocStringExtensions.TYPEDEF)

Applies one rule to each kind of unknown entry.

[`EntrywiseUnknown`](@ref) and [`ZeroUnknown`](@ref) are presets that apply one rule to every kind. This rule holds one rule per kind, so a caller can read one kind as zero and keep the other unknown.

  - `unstated`: An entry the factor model does not state. It is the idiosyncratic variance of an asset in the warm-up of its variance, or the loadings of an asset that is not investable.
  - `leverage`: The variance split of a Leverage-One Pair. The fit reproduced the return of the pair, so the data identify its total return and not the split between its systematic and its idiosyncratic part. [`EntrywiseUnknown`](@ref) gives `NaN` for every variance number that reads the split of a held pair. [`ZeroUnknown`](@ref) reads the split as fitted, the idiosyncratic variance of the pair as zero.

The rule applies to a factor attribution alone. Its two kinds do not apply to a graph, so a [`GraphUnion`](@ref) and [`graph_union`](@ref) raise a `MethodError` on it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KindwiseUnknown(;
        unstated::AbstractUnknownEntryRule = EntrywiseUnknown(),
        leverage::AbstractUnknownEntryRule = EntrywiseUnknown()
    ) -> KindwiseUnknown

Keywords correspond to the struct's fields.

## Validation

  - Neither field is a `KindwiseUnknown`, else an `ArgumentError` is raised. A field is the rule of one kind, and a rule per kind has no kind inside it to apply to.

# Related

  - [`AbstractUnknownEntryRule`](@ref)
  - [`EntrywiseUnknown`](@ref)
  - [`ZeroUnknown`](@ref)
  - [`factor_attribution`](@ref)
"""
@concrete struct KindwiseUnknown <: AbstractUnknownEntryRule
    """
    Rule for an entry the factor model does not state.
    """
    unstated
    """
    Rule for the variance split of a Leverage-One Pair.
    """
    leverage
    function KindwiseUnknown(unstated::T1,
                             leverage::T2) where {T1 <: AbstractUnknownEntryRule,
                                                  T2 <: AbstractUnknownEntryRule}
        @argcheck(!isa(unstated, KindwiseUnknown) && !isa(leverage, KindwiseUnknown),
                  ArgumentError("each field of `KindwiseUnknown` is the rule of one kind of unknown entry, so it cannot be a `KindwiseUnknown`"))
        return new{T1, T2}(unstated, leverage)
    end
end
function KindwiseUnknown(; unstated::AbstractUnknownEntryRule = EntrywiseUnknown(),
                         leverage::AbstractUnknownEntryRule = EntrywiseUnknown())::KindwiseUnknown
    return KindwiseUnknown(unstated, leverage)
end

export EntrywiseUnknown, ZeroUnknown, KindwiseUnknown
public AbstractUnknownEntryRule
