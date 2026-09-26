"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all walk-forward cross-validation estimators.

Walk-forward estimators split time series data into sequential training and testing windows, advancing the test window forward at each step. Subtypes implement index-based or date-based walk-forward schemes.

# Related

  - [`IndexWalkForward`](@ref)
  - [`DateWalkForward`](@ref)
  - [`WalkForwardResult`](@ref)
  - [`SequentialCrossValidationEstimator`](@ref)
"""
abstract type WalkForwardEstimator <: SequentialCrossValidationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Result type produced by [`WalkForwardEstimator`](@ref) subtypes after splitting time series data.

Stores the train and test index vectors for each fold of the walk-forward cross-validation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WalkForwardResult(; train_idx::VecVecInt, test_idx::VecVecInt) -> WalkForwardResult

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(train_idx)` (sufficient data to cover training + testing periods).
  - `!isempty(test_idx)` (sufficient data to cover training + testing periods).
  - `length(train_idx) == length(test_idx)`.

# Related

  - [`WalkForwardEstimator`](@ref)
  - [`IndexWalkForward`](@ref)
  - [`DateWalkForward`](@ref)
  - [`SequentialCrossValidationResult`](@ref)
"""
@concrete struct WalkForwardResult <: SequentialCrossValidationResult
    """
    $(field_dict[:train_idx])
    """
    train_idx
    """
    $(field_dict[:test_idx])
    """
    test_idx
    function WalkForwardResult(train_idx::VecVecInt, test_idx::VecVecInt)
        @argcheck(!isempty(train_idx),
                  IsEmptyError("not enough data to cover the training + testing periods, please check your inputs to ensure they are compatible."))
        @argcheck(!isempty(test_idx),
                  IsEmptyError("not enough data to cover the training + testing periods, please check your inputs to ensure they are compatible."))
        @argcheck(length(train_idx) == length(test_idx),
                  DimensionMismatch("train_idx ($(length(train_idx))) must match test_idx ($(length(test_idx)))"))
        return new{typeof(train_idx), typeof(test_idx)}(train_idx, test_idx)
    end
end
function WalkForwardResult(; train_idx::VecVecInt, test_idx::VecVecInt)::WalkForwardResult
    return WalkForwardResult(train_idx, test_idx)
end
"""
    const WalkForward_Onl = Union{<:WalkForwardEstimator, <:Online{<:WalkForwardEstimator}}

Alias for a walk-forward scheme, plain or an Online Scheme.

An Online Scheme is an [`Online`](@ref) around a walk-forward, built by the scheme's function constructor — [`OnlineIndexWalkForward`](@ref), [`OnlineDateWalkForward`](@ref), [`OnlineHindsightSplit`](@ref). It is not a [`WalkForwardEstimator`](@ref) by supertype, because a struct has one, so every bound that admits a walk-forward is written on this alias and the wrapped form reaches the same doors as the plain one. `split`, `n_splits` and [`fold_evaluation`](@ref) forward to the wrapped scheme; [`folds_are_stepped`](@ref) is where the two differ.

# Related

  - [`WalkForwardEstimator`](@ref)
  - [`Online`](@ref)
  - [`CVE_Onl`](@ref)
  - [`WFCVER`](@ref)
"""
const WalkForward_Onl = Union{<:WalkForwardEstimator, <:Online{<:WalkForwardEstimator}}
"""
    const WFCVER = Union{<:WalkForward_Onl, <:WalkForwardResult}

Alias for a walk-forward cross-validation scheme, plain or online, or its result.

# Related

  - [`WalkForward_Onl`](@ref)
  - [`WalkForwardResult`](@ref)
"""
const WFCVER = Union{<:WalkForward_Onl, <:WalkForwardResult}
"""
$(DocStringExtensions.TYPEDEF)

Implements index-based walk-forward cross-validation for time series, supporting purging and flexible train/test windowing.

`purged_size` drops the last `purged_size` rows of each training window. This opens a gap of that many observations before the test window, and removes the training rows whose labels reach into the test period. The test windows do not move, so a purge costs training rows rather than test coverage.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IndexWalkForward(
        train_size::Integer,
        test_size::Integer;
        purged_size::Integer = 0,
        expand_train::Bool = false,
        reduce_test::Bool = false,
        wd::Option{<:AbstractWeightDrift} = nothing,
        pws::Option{<:AbstractPreviousWeightsSource} = nothing,
        fa::Option{<:AbstractFeeAmortisation} = nothing,
        store_weight_path::Bool = false,
        strict::Bool = false,
    ) -> IndexWalkForward

Positional and keyword arguments correspond to the struct's fields.

## Online form

The scheme refits every fold from its training window. Its online form, under which the fold loop warms one estimator up on the first training window and then folds each fold's new observations into it, is the Online Scheme [`OnlineIndexWalkForward`](@ref) builds: the same keywords minus `expand_train`, which it sets `true`, because a fold cannot un-fold an observation.

## Weight drift and previous weights

The two switches are independent, and each one is `nothing` by default, which is the library's original behaviour.

`wd` is the Weight Drift of the scheme. `nothing` reads a fold's return series as `X * w` net of fees, at the target weights of that fold. A [`SelfFinancingDrift`](@ref) reads the series as the wealth ratio of the drifted holdings instead. `store_weight_path` makes the fold store the weight path it computed, which a reader otherwise rebuilds on demand. `strict` decides what a **Held Gap** does: an asset that delists inside a test window carries a non-zero weight and a missing return, and the fold zeroes that pair and warns, or refuses with an `ArgumentError` under `strict`.

`pws` is the Previous-Weights Source. `nothing` threads the target weights of the previous fold into the next one. A [`DriftedWeights`](@ref) threads the weights held after the last observation of the previous fold instead, so a turnover, a tracking or a fee estimator measures the trades a fund places rather than the change in the decision. A fold enumeration of this scheme is a timeline, so the source has a previous fold to read.

## Fee clock

`fa` is the clock the fold's **realised** series charges the two fixed fee terms on, and it overrides the `fa` of the fee itself. `nothing` inherits that fee's clock, which is the library's original behaviour. A [`FirstObservationFees`](@ref) charges the two terms on the first observation of the fold, and an [`AmortisedFees`](@ref) spreads them over the fold. The field reaches the fit not at all, so the optimiser keeps pricing the fee the way its own objective must.

## Validation

  - `train_size` and `purged_size` must be non-empty, non-negative, and finite.
  - `test_size` must be non-empty, greater than zero, and finite.
  - `purged_size < train_size`, because the purge is taken out of the training window.

The rule `train_size < T`, where `T` is the number of observations, belongs to the data rather
than to the estimator, so [`Base.split`](@ref) checks it.

# Examples

```jldoctest
julia> IndexWalkForward(100, 20; purged_size = 5, expand_train = true, reduce_test = false)
IndexWalkForward
         train_size ┼ Int64: 100
          test_size ┼ Int64: 20
        purged_size ┼ Int64: 5
       expand_train ┼ Bool: true
        reduce_test ┼ Bool: false
                 wd ┼ nothing
                pws ┼ nothing
                 fa ┼ nothing
  store_weight_path ┼ Bool: false
             strict ┴ Bool: false
```

# Related

  - [`cross_val_predict`](@ref)
  - [`search_cross_validation`](@ref)
  - [`WalkForwardEstimator`](@ref)
  - [`WalkForwardResult`](@ref)
  - [`OnlineIndexWalkForward`](@ref)
  - [`n_splits`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 15.1.
  - $(ref_dict[:lopezdeprado2018]) Chapter 7.
"""
@concrete struct IndexWalkForward <: WalkForwardEstimator
    """
    $(field_dict[:train_size])
    """
    train_size
    """
    $(field_dict[:test_size])
    """
    test_size
    """
    $(field_dict[:purged_size])
    """
    purged_size
    """
    $(field_dict[:expand_train])
    """
    expand_train
    """
    $(field_dict[:reduce_test])
    """
    reduce_test
    """
    $(field_dict[:wd])
    """
    wd
    """
    $(field_dict[:pws])
    """
    pws
    """
    $(field_dict[:fa_cv])
    """
    fa
    """
    $(field_dict[:store_weight_path])
    """
    store_weight_path
    """
    $(field_dict[:cv_strict])
    """
    strict
    function IndexWalkForward(train_size::Integer, test_size::Integer, purged_size::Integer,
                              expand_train::Bool, reduce_test::Bool,
                              wd::Option{<:AbstractWeightDrift},
                              pws::Option{<:AbstractPreviousWeightsSource},
                              fa::Option{<:AbstractFeeAmortisation},
                              store_weight_path::Bool, strict::Bool)
        assert_nonempty_gt0_finite_val(test_size, :test_size)
        assert_nonempty_nonneg_finite_val(train_size, :train_size)
        assert_nonempty_nonneg_finite_val(purged_size, :purged_size)
        @argcheck(purged_size < train_size,
                  DomainError(purged_size,
                              "purged_size ($purged_size) must be less than train_size ($train_size), because the purge is taken out of the training window"))
        return new{typeof(train_size), typeof(test_size), typeof(purged_size),
                   typeof(expand_train), typeof(reduce_test), typeof(wd), typeof(pws),
                   typeof(fa), typeof(store_weight_path), typeof(strict)}(train_size,
                                                                          test_size,
                                                                          purged_size,
                                                                          expand_train,
                                                                          reduce_test, wd,
                                                                          pws, fa,
                                                                          store_weight_path,
                                                                          strict)
    end
end
function IndexWalkForward(train_size::Integer, test_size::Integer; purged_size::Integer = 0,
                          expand_train::Bool = false, reduce_test::Bool = false,
                          wd::Option{<:AbstractWeightDrift} = nothing,
                          pws::Option{<:AbstractPreviousWeightsSource} = nothing,
                          fa::Option{<:AbstractFeeAmortisation} = nothing,
                          store_weight_path::Bool = false, strict::Bool = false)
    return IndexWalkForward(train_size, test_size, purged_size, expand_train, reduce_test,
                            wd, pws, fa, store_weight_path, strict)
end
"""
    OnlineIndexWalkForward(
        train_size::Integer,
        test_size::Integer;
        purged_size::Integer = 0,
        reduce_test::Bool = false,
        wd::Option{<:AbstractWeightDrift} = nothing,
        pws::Option{<:AbstractPreviousWeightsSource} = nothing,
        fa::Option{<:AbstractFeeAmortisation} = nothing,
        store_weight_path::Bool = false,
        strict::Bool = false,
    ) -> Online{<:IndexWalkForward}

Build the Online Scheme of an [`IndexWalkForward`](@ref): the scheme wrapped in [`Online`](@ref), under which the fold loop fits each fold by the online step instead of a refit.

The keywords are the scheme's, minus `expand_train`, which is set `true`: a fold cannot un-fold an observation, so an online run is expanding by construction, and the mismatch a rolling window would be cannot be written. A rolling window computed online is the estimator's to declare, through `Online(pe; max_history = train_size - purged_size)` on the prior. This constructor is the only way to build the wrapped scheme; `Online(cv)` by hand is refused by name.

Under the Online Scheme the loop takes its online arm, [`online_folds`](@ref). It resolves every [`Online`](@ref) wrapper in the estimator through [`update_online_estimator`](@ref), folds the first training window in with [`partial_fit!`](@ref), and then, per fold, folds only the rows the training window has gained since the last fold and reads the estimator out through `optimise(opt)`. The estimator is threaded from fold to fold, so fold `i` never re-reads the rows fold `i - 1` read, and the run reaches the weights of the batch expanding-window scheme, `IndexWalkForward(train_size, test_size; purged_size, expand_train = true)`, fold for fold.

The loop starts cold: an estimator carrying a partial-fit state at entry is refused by name, and a [`TimeDependent`](@ref) schedule on a field that carries a state — the prior, or the optimiser itself — is refused at warm-up, because a schedule replaces the value a state is threaded through. A schedule on any other field composes with no rule.

# Examples

```jldoctest
julia> OnlineIndexWalkForward(100, 20; purged_size = 5)
Online
          est ┼ IndexWalkForward
              │          train_size ┼ Int64: 100
              │           test_size ┼ Int64: 20
              │         purged_size ┼ Int64: 5
              │        expand_train ┼ Bool: true
              │         reduce_test ┼ Bool: false
              │                  wd ┼ nothing
              │                 pws ┼ nothing
              │                  fa ┼ nothing
              │   store_weight_path ┼ Bool: false
              │              strict ┴ Bool: false
  max_history ┴ nothing
```

# Related

  - [`IndexWalkForward`](@ref)
  - [`Online`](@ref)
  - [`OnlineDateWalkForward`](@ref)
  - [`OnlineHindsightSplit`](@ref)
  - [`online_folds`](@ref)
  - [`folds_are_stepped`](@ref)
  - [`Resume`](@ref)
"""
function OnlineIndexWalkForward(train_size::Integer, test_size::Integer;
                                purged_size::Integer = 0, reduce_test::Bool = false,
                                wd::Option{<:AbstractWeightDrift} = nothing,
                                pws::Option{<:AbstractPreviousWeightsSource} = nothing,
                                fa::Option{<:AbstractFeeAmortisation} = nothing,
                                store_weight_path::Bool = false, strict::Bool = false)
    cv = IndexWalkForward(train_size, test_size, purged_size, true, reduce_test, wd, pws,
                          fa, store_weight_path, strict)
    return Online{typeof(cv), Nothing}(cv, nothing)
end
"""
    Base.split(iwf::IndexWalkForward, rd::Prices_RR) -> WalkForwardResult

Split the returns data `rd` into sequential walk-forward folds using integer observation
indices. Each fold advances the test window by `test_size` observations.

# Arguments

  - `iwf::IndexWalkForward`: Index-based walk-forward cross-validation estimator.
  - `rd`: Returns-level or price-level data to split ([`Prices_RR`](@ref)).

# Validation

  - `train_size < T`, where `T` is the number of observations in `rd`.

# Returns

  - `WalkForwardResult`: Result containing train and test index ranges for each fold.

# Related

  - [`IndexWalkForward`](@ref)
  - [`WalkForwardResult`](@ref)
  - [`n_splits`](@ref)
"""
function Base.split(iwf::IndexWalkForward, rd::Prices_RR)
    (; train_size, test_size, purged_size, expand_train, reduce_test) = iwf
    T = cv_nobs(rd)
    @argcheck(train_size < T,
              DomainError(train_size, "train_size ($train_size) must be less than T ($T)"))
    idx = 1:T
    test_start = train_size
    train_indices = Vector{typeof(idx)}(undef, 0)
    test_indices = Vector{typeof(idx)}(undef, 0)
    while true
        if test_start >= T
            break
        end
        test_end = test_start + test_size
        train_end = test_start - purged_size
        train_start = expand_train ? 1 : test_start - train_size + 1
        if test_end > T
            if !reduce_test
                break
            end
            push!(test_indices, idx[(test_start + 1):end])
        else
            push!(test_indices, idx[(test_start + 1):test_end])
        end
        push!(train_indices, idx[train_start:train_end])
        test_start = test_end
    end

    return WalkForwardResult(; train_idx = train_indices, test_idx = test_indices)
end
"""
    n_splits(cv, rd::Prices_RR)
    n_splits(cv)

Return the number of cross-validation splits (folds) that would be produced by `cv` for the given returns data `rd`.

# Arguments

  - `cv`: A cross-validation estimator or result (e.g. [`KFold`](@ref), [`IndexWalkForward`](@ref), [`DateWalkForward`](@ref), [`CombinatorialCrossValidation`](@ref), [`MultipleRandomised`](@ref), or their corresponding result types).
  - `rd`: Returns-level or price-level data used to determine the number of splits ([`Prices_RR`](@ref)).

# Validation

  - For an [`IndexWalkForward`](@ref), `train_size < T`, where `T` is the number of observations in `rd`. This is the rule [`Base.split`](@ref) checks, so the count and the split agree on data too short to test.

# Returns

  - `Integer`: The number of folds.

# Related

  - [`KFold`](@ref)
  - [`IndexWalkForward`](@ref)
  - [`DateWalkForward`](@ref)
  - [`WalkForwardResult`](@ref)
  - [`CombinatorialCrossValidation`](@ref)
"""
function n_splits(iwf::IndexWalkForward, rd::Prices_RR)
    (; train_size, test_size, reduce_test) = iwf
    T = cv_nobs(rd)
    # The same rule `split` checks, so the two never disagree on short data: without it a
    # `train_size` of `T` or more gives a count of zero or less where `split` refuses.
    @argcheck(train_size < T,
              DomainError(train_size, "train_size ($train_size) must be less than T ($T)"))
    N = T - train_size
    val = div(N, test_size)
    if reduce_test && N % test_size != 0
        val += 1
    end
    return val
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for date adjustment estimators in walk-forward cross-validation.

Subtypes implement specific strategies for adjusting dates used in walk-forward splits. A subtype
reaches a [`DateWalkForward`](@ref) through its `adjuster` field, whose type bound is
[`DateAdjType`](@ref).

# Related

  - [`DateWalkForward`](@ref)
  - [`DateAdjType`](@ref)
  - [`walk_forward_date_range`](@ref)
"""
abstract type DateAdjusterEstimator <: AbstractEstimator end
"""
    const DatesUnionPeriod = Union{<:Dates.Period, <:Dates.CompoundPeriod}

Alias for a Dates period or compound period.

Used internally to accept either simple date periods (e.g., `Dates.Month(1)`) or compound periods (e.g., `Dates.Month(1) + Dates.Day(1)`) as date offsets in walk-forward cross-validation.

# Related

  - [`IntPeriodDateRange`](@ref)
  - [`DateWalkForward`](@ref)
"""
const DatesUnionPeriod = Union{<:Dates.Period, <:Dates.CompoundPeriod}
"""
    const IntPeriodDateRange = Union{<:Integer, <:DatesUnionPeriod}

Alias for an integer or date period used to specify window sizes.

Matches either a plain integer (number of observations) or a date period ([`DatesUnionPeriod`](@ref)) for walk-forward cross-validation splits.

# Related

  - [`DatesUnionPeriod`](@ref)
  - [`DateWalkForward`](@ref)
"""
const IntPeriodDateRange = Union{<:Integer, <:DatesUnionPeriod}
"""
    const DateAdjType = Union{<:Function, <:DateAdjusterEstimator}

Alias for a date adjustment function or estimator.

Matches either a plain `Function` or a [`DateAdjusterEstimator`](@ref) for adjusting dates in walk-forward cross-validation.

# Related

  - [`DateAdjusterEstimator`](@ref)
  - [`DateWalkForward`](@ref)
"""
const DateAdjType = Union{<:Function, <:DateAdjusterEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Implements date-based walk-forward cross-validation for time series, supporting flexible windowing, purging, and custom date adjustment.

`purged_size` drops the last `purged_size` rows of each training window. This opens a gap of that many observations before the test window, and removes the training rows whose labels reach into the test period. The test window itself is not shortened, so `purged_size` must be smaller than the training window.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DateWalkForward(
        train_size::IntPeriodDateRange,
        test_size::Integer;
        period::DatesUnionPeriod = Dates.Day(1),
        period_offset::Option{<:DatesUnionPeriod} = nothing,
        purged_size::Integer = 0,
        adjuster::DateAdjType = identity,
        previous::Bool = false,
        expand_train::Bool = false,
        reduce_test::Bool = false,
        wd::Option{<:AbstractWeightDrift} = nothing,
        pws::Option{<:AbstractPreviousWeightsSource} = nothing,
        fa::Option{<:AbstractFeeAmortisation} = nothing,
        store_weight_path::Bool = false,
        strict::Bool = false,
    ) -> DateWalkForward

Positional and keyword arguments correspond to the struct's fields.

## Online form

The scheme refits every fold from its training window. Its online form, under which the fold loop warms one estimator up on the first training window and then folds each fold's new observations into it, is the Online Scheme [`OnlineDateWalkForward`](@ref) builds: the same keywords minus `expand_train`, which it sets `true`, because a fold cannot un-fold an observation.

## Weight drift and previous weights

The two switches are independent, and each one is `nothing` by default, which is the library's original behaviour.

`wd` is the Weight Drift of the scheme. `nothing` reads a fold's return series as `X * w` net of fees, at the target weights of that fold. A [`SelfFinancingDrift`](@ref) reads the series as the wealth ratio of the drifted holdings instead. `store_weight_path` makes the fold store the weight path it computed, which a reader otherwise rebuilds on demand. `strict` decides what a **Held Gap** does: an asset that delists inside a test window carries a non-zero weight and a missing return, and the fold zeroes that pair and warns, or refuses with an `ArgumentError` under `strict`.

`pws` is the Previous-Weights Source. `nothing` threads the target weights of the previous fold into the next one. A [`DriftedWeights`](@ref) threads the weights held after the last observation of the previous fold instead, so a turnover, a tracking or a fee estimator measures the trades a fund places rather than the change in the decision. A fold enumeration of this scheme is a timeline, so the source has a previous fold to read.

## Fee clock

`fa` is the clock the fold's **realised** series charges the two fixed fee terms on, and it overrides the `fa` of the fee itself. `nothing` inherits that fee's clock, which is the library's original behaviour. A [`FirstObservationFees`](@ref) charges the two terms on the first observation of the fold, and an [`AmortisedFees`](@ref) spreads them over the fold. The field reaches the fit not at all, so the optimiser keeps pricing the fee the way its own objective must.

## Validation

  - `test_size` must be non-empty, greater than zero, and finite.
  - `purged_size` must be non-empty, non-negative, and finite.
  - If `train_size` is an integer, it must be non-empty, non-negative, and finite.

# Examples

```jldoctest
julia> DateWalkForward(252, 21; period = Dates.Day(1), purged_size = 5, expand_train = true)
DateWalkForward
         train_size ┼ Int64: 252
          test_size ┼ Int64: 21
             period ┼ Dates.Day: Dates.Day(1)
      period_offset ┼ nothing
        purged_size ┼ Int64: 5
           adjuster ┼ typeof(identity): identity
           previous ┼ Bool: false
       expand_train ┼ Bool: true
        reduce_test ┼ Bool: false
                 wd ┼ nothing
                pws ┼ nothing
                 fa ┼ nothing
  store_weight_path ┼ Bool: false
             strict ┴ Bool: false
```

# Related

  - [`cross_val_predict`](@ref)
  - [`search_cross_validation`](@ref)
  - [`WalkForwardEstimator`](@ref)
  - [`WalkForwardResult`](@ref)
  - [`OnlineDateWalkForward`](@ref)
  - [`n_splits`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 15.1.
  - $(ref_dict[:lopezdeprado2018]) Chapter 7.
"""
@concrete struct DateWalkForward <: WalkForwardEstimator
    """
    $(field_dict[:train_size])
    """
    train_size
    """
    $(field_dict[:test_size])
    """
    test_size
    """
    $(field_dict[:period])
    """
    period
    """
    $(field_dict[:period_offset])
    """
    period_offset
    """
    $(field_dict[:purged_size])
    """
    purged_size
    """
    $(field_dict[:adjuster])
    """
    adjuster
    """
    $(field_dict[:previous])
    """
    previous
    """
    $(field_dict[:expand_train])
    """
    expand_train
    """
    $(field_dict[:reduce_test])
    """
    reduce_test
    """
    $(field_dict[:wd])
    """
    wd
    """
    $(field_dict[:pws])
    """
    pws
    """
    $(field_dict[:fa_cv])
    """
    fa
    """
    $(field_dict[:store_weight_path])
    """
    store_weight_path
    """
    $(field_dict[:cv_strict])
    """
    strict
    function DateWalkForward(train_size::IntPeriodDateRange, test_size::Integer,
                             period::DatesUnionPeriod,
                             period_offset::Option{<:DatesUnionPeriod},
                             purged_size::Integer, adjuster::DateAdjType, previous::Bool,
                             expand_train::Bool, reduce_test::Bool,
                             wd::Option{<:AbstractWeightDrift},
                             pws::Option{<:AbstractPreviousWeightsSource},
                             fa::Option{<:AbstractFeeAmortisation}, store_weight_path::Bool,
                             strict::Bool)
        assert_nonempty_gt0_finite_val(test_size, :test_size)
        if isa(train_size, Integer)
            assert_nonempty_nonneg_finite_val(train_size, :train_size)
        end
        assert_nonempty_nonneg_finite_val(purged_size, :purged_size)
        return new{typeof(train_size), typeof(test_size), typeof(period),
                   typeof(period_offset), typeof(purged_size), typeof(adjuster),
                   typeof(previous), typeof(expand_train), typeof(reduce_test), typeof(wd),
                   typeof(pws), typeof(fa), typeof(store_weight_path), typeof(strict)}(train_size,
                                                                                       test_size,
                                                                                       period,
                                                                                       period_offset,
                                                                                       purged_size,
                                                                                       adjuster,
                                                                                       previous,
                                                                                       expand_train,
                                                                                       reduce_test,
                                                                                       wd,
                                                                                       pws,
                                                                                       fa,
                                                                                       store_weight_path,
                                                                                       strict)
    end
end
function DateWalkForward(train_size::IntPeriodDateRange, test_size::Integer;
                         period::DatesUnionPeriod = Dates.Day(1),
                         period_offset::Option{<:DatesUnionPeriod} = nothing,
                         purged_size::Integer = 0, adjuster::DateAdjType = identity,
                         previous::Bool = false, expand_train::Bool = false,
                         reduce_test::Bool = false,
                         wd::Option{<:AbstractWeightDrift} = nothing,
                         pws::Option{<:AbstractPreviousWeightsSource} = nothing,
                         fa::Option{<:AbstractFeeAmortisation} = nothing,
                         store_weight_path::Bool = false, strict::Bool = false)
    return DateWalkForward(train_size, test_size, period, period_offset, purged_size,
                           adjuster, previous, expand_train, reduce_test, wd, pws, fa,
                           store_weight_path, strict)
end
"""
    OnlineDateWalkForward(
        train_size::IntPeriodDateRange,
        test_size::Integer;
        period::DatesUnionPeriod = Dates.Day(1),
        period_offset::Option{<:DatesUnionPeriod} = nothing,
        purged_size::Integer = 0,
        adjuster::DateAdjType = identity,
        previous::Bool = false,
        reduce_test::Bool = false,
        wd::Option{<:AbstractWeightDrift} = nothing,
        pws::Option{<:AbstractPreviousWeightsSource} = nothing,
        fa::Option{<:AbstractFeeAmortisation} = nothing,
        store_weight_path::Bool = false,
        strict::Bool = false,
    ) -> Online{<:DateWalkForward}

Build the Online Scheme of a [`DateWalkForward`](@ref): the scheme wrapped in [`Online`](@ref), under which the fold loop fits each fold by the online step instead of a refit.

The keywords are the scheme's, minus `expand_train`, which is set `true`: a fold cannot un-fold an observation, so an online run is expanding by construction, and the mismatch a rolling window would be cannot be written. A rolling window computed online is the estimator's to declare, through `Online(pe; max_history = …)` on the prior, with the cap the window's rows count to. This constructor is the only way to build the wrapped scheme; `Online(cv)` by hand is refused by name. What the loop does under it is what [`OnlineIndexWalkForward`](@ref) describes, over the folds the date scheme cuts.

# Examples

```jldoctest
julia> OnlineDateWalkForward(252, 21; purged_size = 5)
Online
          est ┼ DateWalkForward
              │          train_size ┼ Int64: 252
              │           test_size ┼ Int64: 21
              │              period ┼ Dates.Day: Dates.Day(1)
              │       period_offset ┼ nothing
              │         purged_size ┼ Int64: 5
              │            adjuster ┼ typeof(identity): identity
              │            previous ┼ Bool: false
              │        expand_train ┼ Bool: true
              │         reduce_test ┼ Bool: false
              │                  wd ┼ nothing
              │                 pws ┼ nothing
              │                  fa ┼ nothing
              │   store_weight_path ┼ Bool: false
              │              strict ┴ Bool: false
  max_history ┴ nothing
```

# Related

  - [`DateWalkForward`](@ref)
  - [`Online`](@ref)
  - [`OnlineIndexWalkForward`](@ref)
  - [`OnlineHindsightSplit`](@ref)
  - [`online_folds`](@ref)
  - [`folds_are_stepped`](@ref)
"""
function OnlineDateWalkForward(train_size::IntPeriodDateRange, test_size::Integer;
                               period::DatesUnionPeriod = Dates.Day(1),
                               period_offset::Option{<:DatesUnionPeriod} = nothing,
                               purged_size::Integer = 0, adjuster::DateAdjType = identity,
                               previous::Bool = false, reduce_test::Bool = false,
                               wd::Option{<:AbstractWeightDrift} = nothing,
                               pws::Option{<:AbstractPreviousWeightsSource} = nothing,
                               fa::Option{<:AbstractFeeAmortisation} = nothing,
                               store_weight_path::Bool = false, strict::Bool = false)
    cv = DateWalkForward(train_size, test_size, period, period_offset, purged_size,
                         adjuster, previous, true, reduce_test, wd, pws, fa,
                         store_weight_path, strict)
    return Online{typeof(cv), Nothing}(cv, nothing)
end
"""
    walk_forward_date_range(ts::AbstractVector, period::DatesUnionPeriod,
                            period_offset::Option{<:DatesUnionPeriod}, adjuster::DateAdjType)

Build the date range that a [`DateWalkForward`](@ref) estimator walks over the timestamps `ts`.

The range runs from the first to the last timestamp with a step of `period`. If `period_offset` is
not `nothing`, the start moves to `min(ts[1], ts[1] - period_offset)`, the `adjuster` is applied, and
the offset is added to every date. This shifts each date by the offset without losing the start of
`ts`.

# Arguments

  - `ts`: Sorted timestamp vector ([`cv_timestamps`](@ref)).
  - `period`: Step of the date range.
  - `period_offset`: Offset applied to every date of the range, or `nothing`.
  - `adjuster`: Function applied to the unshifted range, e.g. a business-day adjuster.

# Returns

  - The date range walked by [`Base.split`](@ref) and [`n_splits`](@ref).

# Related

  - [`DateWalkForward`](@ref)
  - [`date_index_positions`](@ref)
"""
function walk_forward_date_range(ts::AbstractVector, period::DatesUnionPeriod,
                                 period_offset::Option{<:DatesUnionPeriod},
                                 adjuster::DateAdjType)
    ti = ts[1]
    tf = ts[end]
    po_flag = !isnothing(period_offset)
    if po_flag
        ti = min(ti, ti - period_offset)
    end
    date_range = adjuster(ti:period:tf)
    if po_flag
        date_range += period_offset
    end
    return date_range
end
"""
    date_index_positions(ts::AbstractVector, date_range, previous::Bool,
                         ::Type{T} = Int) where {T <: Integer}

Map every date of `date_range` to a position in the sorted timestamp vector `ts`.

A date that falls between two timestamps maps to the previous timestamp if `previous` is `true`, and
to the next timestamp if `previous` is `false`. A date before `ts[1]` maps to the first timestamp.
The mapping stops at the first date that falls after `ts[end]`, so the result is not longer than
`date_range`.

# Arguments

  - `ts`: Sorted timestamp vector ([`cv_timestamps`](@ref)).
  - `date_range`: Date range from [`walk_forward_date_range`](@ref).
  - `previous`: If `true`, a date between two timestamps takes the previous timestamp.
  - `T`: Element type of the returned vector.

# Returns

  - `Vector{T}`: Positions in `ts`, in the order of `date_range`.

# Related

  - [`DateWalkForward`](@ref)
  - [`walk_forward_date_range`](@ref)
"""
function date_index_positions(ts::AbstractVector, date_range, previous::Bool,
                              ::Type{T} = Int) where {T <: Integer}
    idx = Vector{T}(undef, 0)
    for date in date_range
        i = searchsortedlast(ts, date)
        if iszero(i) || !previous && ts[i] != date
            i += 1
        end
        if i > length(ts)
            break
        end
        push!(idx, i)
    end
    return idx
end
"""
    Base.split(dwf::DateWalkForward{<:Integer}, rd::Prices_RR) -> WalkForwardResult

Split the returns data `rd` into sequential walk-forward folds using date-aligned indices,
where `train_size` is specified as an integer number of date-range steps.

The timestamp vector ([`cv_timestamps`](@ref)) must not be `nothing`. Training and test windows are aligned
to the `period` date range and advanced by `test_size` steps at a time.

# Arguments

  - `dwf::DateWalkForward{<:Integer}`: Date-based walk-forward estimator with an integer
    `train_size`.
  - `rd`: Returns-level or price-level data with timestamps ([`Prices_RR`](@ref)).

# Returns

  - `WalkForwardResult`: Result containing train and test index ranges for each fold.

# Related

  - [`DateWalkForward`](@ref)
  - [`WalkForwardResult`](@ref)
  - [`n_splits`](@ref)
"""
function Base.split(dwf::DateWalkForward{<:Integer}, rd::Prices_RR)
    ts = cv_timestamps(rd)
    @argcheck(!isnothing(ts), IsNothingError)
    (; train_size, test_size, period, period_offset, purged_size, adjuster, previous, expand_train, reduce_test) = dwf
    T = cv_nobs(rd)
    date_range = walk_forward_date_range(ts, period, period_offset, adjuster)
    tt = typeof(T)
    idx = date_index_positions(ts, date_range, previous, tt)
    N = length(idx)
    i = 1
    train_indices = Vector{UnitRange{tt}}(undef, 0)
    test_indices = Vector{UnitRange{tt}}(undef, 0)
    while true
        if i + train_size > N
            break
        end
        if i + train_size + test_size > N
            if !reduce_test
                break
            end
            push!(test_indices, idx[i + train_size]:T)
        else
            push!(test_indices, idx[i + train_size]:(idx[i + train_size + test_size] - 1))
        end
        train_start = expand_train ? 1 : idx[i]
        push!(train_indices, train_start:(idx[i + train_size] - purged_size - 1))
        i += test_size
    end
    return WalkForwardResult(; train_idx = train_indices, test_idx = test_indices)
end
"""
    special_div(a::Integer, b::Integer)

Return the number of steps of size `b` that fit in the span `1:a`.

The result is the largest `k` for which `1 + k * b <= a`, which is `div(a - 1, b)` for a positive
`a`. A walk-forward fold count uses it this way: the first window starts at the first position of
the span and every later window starts `b` positions after the previous one, so the number of
windows is `special_div(a, b) + 1`. The caller passes the **length** of the span, not the difference
between its last and its first position. `b` must not be zero.

# Arguments

  - `a`: Length of the span of positions at which a window may start.
  - `b`: Step between two window starts.

# Returns

  - `Integer`: The number of steps after the first window, `div(a - 1, b)`.

# Related

  - [`DateWalkForward`](@ref)
  - [`n_splits`](@ref)
"""
function special_div(a::Integer, b::Integer)
    q, r = divrem(a, b)
    return q - ifelse(iszero(r), 1, 0)
end
"""
    n_splits(dwf::DateWalkForward{<:Integer}, rd::Prices_RR) -> Integer

Return the number of walk-forward folds that would be produced by `dwf` for the given
returns data `rd` when the training window size is specified as an integer number of
date-range steps.

# Arguments

  - `dwf::DateWalkForward{<:Integer}`: Date-based walk-forward estimator with an integer
    `train_size`.
  - `rd`: Returns-level or price-level data with timestamps ([`Prices_RR`](@ref)).

# Returns

  - `Integer`: The number of folds.

# Related

  - [`DateWalkForward`](@ref)
  - [`WalkForwardResult`](@ref)
  - [`Base.split(dwf::DateWalkForward{<:Integer}, rd::Prices_RR)`](@ref)
"""
function n_splits(dwf::DateWalkForward{<:Integer}, rd::Prices_RR)
    ts = cv_timestamps(rd)
    @argcheck(!isnothing(ts), IsNothingError)
    (; train_size, test_size, period, period_offset, adjuster, previous, reduce_test) = dwf
    date_range = walk_forward_date_range(ts, period, period_offset, adjuster)
    N = length(date_index_positions(ts, date_range, previous))
    max_start = N - train_size - ifelse(reduce_test, 0, test_size)
    return max_start > 0 ? special_div(max_start, test_size) + 1 : 0
end
"""
    Base.split(dwf::DateWalkForward{<:Any}, rd::Prices_RR) -> WalkForwardResult

Split the returns data `rd` into sequential walk-forward folds using date-aligned indices,
where `train_size` is specified as a date `Period` (e.g., `Dates.Month(6)`).

The timestamp vector ([`cv_timestamps`](@ref)) must not be `nothing`. Training windows are defined by
subtracting `train_size` from the split date, allowing calendar-based window lengths.

# Arguments

  - `dwf::DateWalkForward{<:Any}`: Date-based walk-forward estimator with a `Period`
    `train_size`.
  - `rd`: Returns-level or price-level data with timestamps ([`Prices_RR`](@ref)).

# Returns

  - `WalkForwardResult`: Result containing train and test index ranges for each fold.

# Related

  - [`DateWalkForward`](@ref)
  - [`WalkForwardResult`](@ref)
  - [`n_splits`](@ref)
"""
function Base.split(dwf::DateWalkForward{<:Any}, rd::Prices_RR)
    ts = cv_timestamps(rd)
    @argcheck(!isnothing(ts), IsNothingError)
    (; train_size, test_size, period, period_offset, purged_size, adjuster, previous, expand_train, reduce_test) = dwf
    T = cv_nobs(rd)
    date_range = walk_forward_date_range(ts, period, period_offset, adjuster)
    idx = date_index_positions(ts, date_range, previous, typeof(T))
    train_idx = Vector{typeof(T)}(undef, 0)
    for date in date_range
        date = date - train_size
        i = searchsortedlast(ts, date)
        if i > length(ts)
            break
        end
        push!(train_idx, i)
    end
    N = length(idx)
    i = searchsortedlast(train_idx, 0) + 1
    train_indices = Vector{UnitRange{typeof(T)}}(undef, 0)
    test_indices = Vector{UnitRange{typeof(T)}}(undef, 0)
    while true
        if i > N
            break
        end
        if i + test_size > N
            if !reduce_test
                break
            end
            push!(test_indices, idx[i]:T)
        else
            push!(test_indices, idx[i]:(idx[i + test_size] - 1))
        end
        train_start = expand_train ? 1 : train_idx[i]
        push!(train_indices, train_start:(idx[i] - purged_size - 1))
        i += test_size
    end
    return WalkForwardResult(; train_idx = train_indices, test_idx = test_indices)
end
"""
    n_splits(dwf::DateWalkForward{<:Any}, rd::Prices_RR) -> Integer

Return the number of walk-forward folds that would be produced by `dwf` for the given
returns data `rd` when the training window size is specified as a date `Period`.

# Arguments

  - `dwf::DateWalkForward{<:Any}`: Date-based walk-forward estimator with a `Period`
    `train_size`.
  - `rd`: Returns-level or price-level data with timestamps ([`Prices_RR`](@ref)).

# Returns

  - `Integer`: The number of folds.

# Related

  - [`DateWalkForward`](@ref)
  - [`WalkForwardResult`](@ref)
  - [`Base.split(dwf::DateWalkForward{<:Any}, rd::Prices_RR)`](@ref)
"""
function n_splits(dwf::DateWalkForward{<:Any}, rd::Prices_RR)
    ts = cv_timestamps(rd)
    @argcheck(!isnothing(ts), IsNothingError)
    (; train_size, test_size, period, period_offset, adjuster, previous, reduce_test) = dwf
    date_range = walk_forward_date_range(ts, period, period_offset, adjuster)
    N = length(date_index_positions(ts, date_range, previous))
    M = -1
    for (j, date) in enumerate(date_range)
        date = date - train_size
        i = searchsortedlast(ts, date)
        if i > length(ts)
            break
        end
        M = ifelse(iszero(i), j, M)
    end
    M += 1
    if iszero(M)
        return 0
    end
    last_allowed_start = reduce_test ? N : N - test_size
    if M > last_allowed_start
        return 0
    end
    return special_div(last_allowed_start - M + 1, test_size) + 1
end
"""
$(DocStringExtensions.TYPEDEF)

A cross-validator whose training set contains its test row by design: fold `t` trains on the rows through `t`, or on row `t` alone, and tests on row `t`.

`HindsightSplit` is the Hindsight Comparator rule of [`log_wealth_regret`](@ref) taken per row. An estimator run through it with [`cross_val_predict`](@ref) reads the row it is scored on before it is scored on it, which no causal scheme may do and which this scheme does by declaration, so that the run yields the per-row comparators dynamic regret is stated against. Under `prefix = true` fold `t` trains on rows `1:t`: [`MeanRisk`](@ref) under [`LogarithmicReturn`](@ref) and [`MaximumReturn`](@ref), or [`BestConstantRebalancedPortfolio`](@ref), is then **be-the-leader**, the best constant rebalanced portfolio over the rows through `t` played on row `t`. Under `prefix = false` fold `t` trains on row `t` alone: the top-1 [`ScoreSelector`](@ref) under [`RankRule`](@ref)`(; best = 1)` and [`MeanReturn`](@ref)`(; flag = true)` composed with [`EqualWeighted`](@ref) in a [`Pipeline`](@ref) is then the **per-period minimiser**, one-hot on the row's best asset, whose terminal wealth is ``\\prod_t \\max_i x_{t,i}`` over the price relatives ``x``. A ``K``-switch comparator is any piecewise-constant path the caller builds. The test window is one row, and every row from `start` to the last is a fold, so the scheme's rows are those of an [`IndexWalkForward`](@ref)`(start - 1, 1)` over the same data, and the two prediction results share the timestamps the regret verb demands.

The scheme is a walk-forward in every other respect: its folds are a timeline, it carries the Weight Drift, the Previous-Weights Source, the Fee Clock and the two flags of [`IndexWalkForward`](@ref), and [`fold_evaluation`](@ref) reads them. Its online form is [`OnlineHindsightSplit`](@ref), the prefix split under which the training windows are nested and the loop steps; the row-alone split has none.

A fold can train on one row: every fold of the row-alone split does, and so does the first fold of the prefix split at `start = 1`. The corrected covariance of one row is `NaN`, so a JuMP estimator under the default [`EmpiricalPrior`](@ref) throws there, in the `Posdef` step of its [`MatrixProcessing`](@ref). A comparator whose programme reads no covariance, such as [`MeanRisk`](@ref) under [`LogarithmicReturn`](@ref) and [`MaximumReturn`](@ref), fits one row under a prior that takes the uncorrected covariance and skips that step, `EmpiricalPrior(; ce = PortfolioOptimisersCovariance(; ce = Covariance(; ce = GeneralCovariance(; ce = StatsBase.SimpleCovariance())), mp = MatrixProcessing(; pdm = nothing)))`. The covariance of one row is then the zero matrix, and the log-optimal programme does not read it. [`BestConstantRebalancedPortfolio`](@ref) fits no prior and needs none of this.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HindsightSplit(;
        prefix::Bool = true,
        start::Integer = 1,
        wd::Option{<:AbstractWeightDrift} = nothing,
        pws::Option{<:AbstractPreviousWeightsSource} = nothing,
        fa::Option{<:AbstractFeeAmortisation} = nothing,
        store_weight_path::Bool = false,
        strict::Bool = false,
    ) -> HindsightSplit

Keyword arguments correspond to the struct's fields.

## Validation

  - `start >= 1`.

The rule `start <= T`, where `T` is the number of observations, belongs to the data rather
than to the estimator, so [`Base.split`](@ref) checks it.

# Examples

```jldoctest
julia> HindsightSplit(; prefix = false, start = 21)
HindsightSplit
             prefix ┼ Bool: false
              start ┼ Int64: 21
                 wd ┼ nothing
                pws ┼ nothing
                 fa ┼ nothing
  store_weight_path ┼ Bool: false
             strict ┴ Bool: false
```

# Related

  - [`log_wealth_regret`](@ref)
  - [`LogWealthRegretResult`](@ref)
  - [`cross_val_predict`](@ref)
  - [`IndexWalkForward`](@ref)
  - [`WalkForwardEstimator`](@ref)
  - [`WalkForwardResult`](@ref)
  - [`BestConstantRebalancedPortfolio`](@ref)
  - [`OnlineHindsightSplit`](@ref)
  - [`n_splits`](@ref)
"""
@concrete struct HindsightSplit <: WalkForwardEstimator
    """
    `prefix`: If `true`, fold `t` trains on rows `1:t`, the prefix through its test row; if `false`, on row `t` alone.
    """
    prefix
    """
    `start`: First row scored. Every row from it to the last is a fold, so a scheme over the rows of an [`IndexWalkForward`](@ref)`(train_size, 1)` sets `start = train_size + 1`.
    """
    start
    """
    $(field_dict[:wd])
    """
    wd
    """
    $(field_dict[:pws])
    """
    pws
    """
    $(field_dict[:fa_cv])
    """
    fa
    """
    $(field_dict[:store_weight_path])
    """
    store_weight_path
    """
    $(field_dict[:cv_strict])
    """
    strict
    function HindsightSplit(prefix::Bool, start::Integer, wd::Option{<:AbstractWeightDrift},
                            pws::Option{<:AbstractPreviousWeightsSource},
                            fa::Option{<:AbstractFeeAmortisation}, store_weight_path::Bool,
                            strict::Bool)
        assert_nonempty_gt0_finite_val(start, :start)
        return new{typeof(prefix), typeof(start), typeof(wd), typeof(pws), typeof(fa),
                   typeof(store_weight_path), typeof(strict)}(prefix, start, wd, pws, fa,
                                                              store_weight_path, strict)
    end
end
function HindsightSplit(; prefix::Bool = true, start::Integer = 1,
                        wd::Option{<:AbstractWeightDrift} = nothing,
                        pws::Option{<:AbstractPreviousWeightsSource} = nothing,
                        fa::Option{<:AbstractFeeAmortisation} = nothing,
                        store_weight_path::Bool = false, strict::Bool = false)
    return HindsightSplit(prefix, start, wd, pws, fa, store_weight_path, strict)
end
"""
    OnlineHindsightSplit(;
        start::Integer = 1,
        wd::Option{<:AbstractWeightDrift} = nothing,
        pws::Option{<:AbstractPreviousWeightsSource} = nothing,
        fa::Option{<:AbstractFeeAmortisation} = nothing,
        store_weight_path::Bool = false,
        strict::Bool = false,
    ) -> Online{<:HindsightSplit}

Build the Online Scheme of a [`HindsightSplit`](@ref): the prefix split wrapped in [`Online`](@ref), under which the fold loop fits each fold by the online step instead of a refit.

The keywords are the scheme's, minus `prefix`, which is set `true`: fold `t` then trains on rows `1:t` and tests on row `t`, and those training windows are nested prefixes, so the loop warms up on `1:start`, folds row `t`, reads out, and scores row `t`, exactly as it does under [`OnlineIndexWalkForward`](@ref). The row-alone split, `prefix = false`, is a window of one row that a fold cannot un-fold, so it has no online form and cannot be written here. The read-out after rows `1:t` equals the batch fit over `1:t`, so the comparators this scheme yields are the batch ones fold for fold; the step is faster than the refit only where the estimator's batch fit is itself a row recursion, and a comparator whose step is a solve, or a fixed point over every row it holds, runs no faster.

# Examples

```jldoctest
julia> OnlineHindsightSplit(; start = 21)
Online
          est ┼ HindsightSplit
              │              prefix ┼ Bool: true
              │               start ┼ Int64: 21
              │                  wd ┼ nothing
              │                 pws ┼ nothing
              │                  fa ┼ nothing
              │   store_weight_path ┼ Bool: false
              │              strict ┴ Bool: false
  max_history ┴ nothing
```

# Related

  - [`HindsightSplit`](@ref)
  - [`Online`](@ref)
  - [`OnlineIndexWalkForward`](@ref)
  - [`OnlineDateWalkForward`](@ref)
  - [`log_wealth_regret`](@ref)
  - [`folds_are_stepped`](@ref)
"""
function OnlineHindsightSplit(; start::Integer = 1,
                              wd::Option{<:AbstractWeightDrift} = nothing,
                              pws::Option{<:AbstractPreviousWeightsSource} = nothing,
                              fa::Option{<:AbstractFeeAmortisation} = nothing,
                              store_weight_path::Bool = false, strict::Bool = false)
    cv = HindsightSplit(true, start, wd, pws, fa, store_weight_path, strict)
    return Online{typeof(cv), Nothing}(cv, nothing)
end
"""
    Base.split(hs::HindsightSplit, rd::Prices_RR) -> WalkForwardResult

Split the returns data `rd` into one fold per row from `start` to the last: fold `t` tests on
row `t` and trains on rows `1:t` under `prefix = true`, on row `t` alone otherwise.

# Arguments

  - `hs::HindsightSplit`: Hindsight splitter.
  - `rd`: Returns-level or price-level data to split ([`Prices_RR`](@ref)).

# Validation

  - `start <= T`, where `T` is the number of observations in `rd`.

# Returns

  - `WalkForwardResult`: Result containing train and test index ranges for each fold.

# Related

  - [`HindsightSplit`](@ref)
  - [`WalkForwardResult`](@ref)
  - [`n_splits`](@ref)
"""
function Base.split(hs::HindsightSplit, rd::Prices_RR)
    (; prefix, start) = hs
    T = cv_nobs(rd)
    @argcheck(start <= T, DomainError(start, "start ($start) must not exceed T ($T)"))
    idx = 1:T
    train_idx = [prefix ? idx[1:t] : idx[t:t] for t in start:T]
    test_idx = [idx[t:t] for t in start:T]
    return WalkForwardResult(; train_idx = train_idx, test_idx = test_idx)
end
function n_splits(hs::HindsightSplit, rd::Prices_RR)
    return cv_nobs(rd) - hs.start + 1
end
"""
    Base.split(o::Online{<:WalkForwardEstimator}, rd::Prices_RR) -> WalkForwardResult
    n_splits(o::Online{<:WalkForwardEstimator}, rd::Prices_RR) -> Integer
    fold_evaluation(o::Online{<:WalkForwardEstimator})
    folds_are_stepped(o::Online{<:WalkForwardEstimator}) -> Bool

The scheme verbs of an Online Scheme.

The fold enumeration is the wrapped walk-forward's and does not change, and neither do its evaluation switches, so `split`, `n_splits` and [`fold_evaluation`](@ref) forward to `o.est`. What changes is how each fold is fitted, and [`folds_are_stepped`](@ref) is where the wrapped form answers `true`: the fold loop reads it off the type and takes its online arm.

# Related

  - [`WalkForward_Onl`](@ref)
  - [`OnlineIndexWalkForward`](@ref)
  - [`OnlineDateWalkForward`](@ref)
  - [`OnlineHindsightSplit`](@ref)
  - [`fold_loop`](@ref)
"""
function Base.split(o::Online{<:WalkForwardEstimator}, rd::Prices_RR)
    return split(o.est, rd)
end
function n_splits(o::Online{<:WalkForwardEstimator}, rd::Prices_RR)
    return n_splits(o.est, rd)
end
function fold_evaluation(o::Online{<:WalkForwardEstimator})
    return fold_evaluation(o.est)
end
function folds_are_stepped(::Online{<:WalkForwardEstimator})
    return true
end
function fit_and_predict(opt::OptE_TD, rd::ReturnsResult, cv::WFCVER; cols = :,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                         id = nothing)
    cv_res = split(cv, rd)
    (; train_idx, test_idx) = cv_res
    assert_unshuffled_folds(cv, train_idx)
    (; wd, pws, fa, store_weight_path, strict) = fold_evaluation(cv)
    hwd = held_weights_drift(wd, pws)
    predictions, est = fold_loop(opt, length(train_idx), ex; rd = rd, train_idx = train_idx,
                                 test_idx = test_idx, cv = cv, pws = pws) do fold
        return fit_and_predict(fold.est, fold.rd; train_idx = fold.train,
                               test_idx = fold.test, cols = cols, wd = wd, hwd = hwd,
                               fa = fa, store_weight_path = store_weight_path,
                               strict = strict, w_prev = fold.w_prev)
    end
    return MultiPeriodPredictionResult(; pred = predictions, id = id, opt = est)
end
function fit_and_predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                         cv::WFCVER; ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                         id = nothing)
    cv_res = split(cv, rd)
    test_idx = cv_res.test_idx
    assert_unshuffled_folds(cv, cv_res.train_idx)
    (; wd, pws, fa, store_weight_path, strict) = fold_evaluation(cv)
    hwd = held_weights_drift(wd, pws)
    predictions = parallel_folds(length(test_idx), ex) do i
        return StatsAPI.predict(res, rd, test_idx[i], :; wd = wd, hwd = hwd, fa = fa,
                                store_weight_path = store_weight_path, strict = strict)
    end
    return MultiPeriodPredictionResult(; pred = predictions, id = id)
end

"""
    fold_evaluation(cv::IndexWalkForward)

Read the evaluation switches of a [`IndexWalkForward`](@ref).

The folds of this scheme are a timeline, so it carries both weight switches and states both of them here, beside the Fee Clock of its realised series.

# Returns

  - `(; wd, pws, fa, store_weight_path, strict)`: The Weight Drift, the Previous-Weights Source, the Fee Clock of the fold's realised series, the flag that stores a fold's weight path, and the flag that makes a Held Gap raise rather than warn.

# Related

  - [`fold_evaluation`](@ref)
  - [`IndexWalkForward`](@ref)
  - [`held_weights_drift`](@ref)
  - [`override_fee_amortisation`](@ref)
"""
function fold_evaluation(cv::IndexWalkForward)
    return (; wd = cv.wd, pws = cv.pws, fa = cv.fa,
            store_weight_path = cv.store_weight_path, strict = cv.strict)
end
"""
    fold_evaluation(cv::DateWalkForward)

Read the evaluation switches of a [`DateWalkForward`](@ref).

The folds of this scheme are a timeline, so it carries both weight switches and states both of them here, beside the Fee Clock of its realised series.

# Returns

  - `(; wd, pws, fa, store_weight_path, strict)`: The Weight Drift, the Previous-Weights Source, the Fee Clock of the fold's realised series, the flag that stores a fold's weight path, and the flag that makes a Held Gap raise rather than warn.

# Related

  - [`fold_evaluation`](@ref)
  - [`DateWalkForward`](@ref)
  - [`held_weights_drift`](@ref)
  - [`override_fee_amortisation`](@ref)
"""
function fold_evaluation(cv::DateWalkForward)
    return (; wd = cv.wd, pws = cv.pws, fa = cv.fa,
            store_weight_path = cv.store_weight_path, strict = cv.strict)
end
"""
    fold_evaluation(cv::HindsightSplit)

Read the evaluation switches of a [`HindsightSplit`](@ref).

The folds of this scheme are a timeline, so it carries both weight switches and states both of them here, beside the Fee Clock of its realised series.

# Returns

  - `(; wd, pws, fa, store_weight_path, strict)`: The Weight Drift, the Previous-Weights Source, the Fee Clock of the fold's realised series, the flag that stores a fold's weight path, and the flag that makes a Held Gap raise rather than warn.

# Related

  - [`fold_evaluation`](@ref)
  - [`HindsightSplit`](@ref)
  - [`held_weights_drift`](@ref)
  - [`override_fee_amortisation`](@ref)
"""
function fold_evaluation(cv::HindsightSplit)
    return (; wd = cv.wd, pws = cv.pws, fa = cv.fa,
            store_weight_path = cv.store_weight_path, strict = cv.strict)
end
export WalkForwardResult, IndexWalkForward, DateWalkForward, HindsightSplit,
       OnlineIndexWalkForward, OnlineDateWalkForward, OnlineHindsightSplit, n_splits
