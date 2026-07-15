"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all walk-forward cross-validation estimators in `PortfolioOptimisers.jl`.

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
    const WFCVER = Union{<:WalkForwardEstimator, <:WalkForwardResult}

Alias for a walk-forward cross-validation estimator or result.

Matches either a [`WalkForwardEstimator`](@ref) or a [`WalkForwardResult`](@ref).

# Related

  - [`WalkForwardEstimator`](@ref)
  - [`WalkForwardResult`](@ref)
"""
const WFCVER = Union{<:WalkForwardEstimator, <:WalkForwardResult}
"""
$(DocStringExtensions.TYPEDEF)

Implements index-based walk-forward cross-validation for time series, supporting purging and flexible train/test windowing.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IndexWalkForward(
        train_size::Integer,
        test_size::Integer;
        purged_size::Integer = 0,
        expand_train::Bool = false,
        reduce_test::Bool = false,
    ) -> IndexWalkForward

Positional and keyword arguments correspond to the struct's fields.

## Validation

  - `train_size`, `test_size`, and `purged_size` must be non-empty, non-negative, and finite.
  - Ensures `train_size + purged_size < T` where `T` is the total number of observations.

# Examples

```jldoctest
julia> IndexWalkForward(100, 20; purged_size = 5, expand_train = true, reduce_test = false)
IndexWalkForward
    train_size ┼ Int64: 100
     test_size ┼ Int64: 20
   purged_size ┼ Int64: 5
  expand_train ┼ Bool: true
   reduce_test ┴ Bool: false
```

# Related

  - [`WalkForwardEstimator`](@ref)
  - [`WalkForwardResult`](@ref)
  - [`n_splits`](@ref)
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
    function IndexWalkForward(train_size::Integer, test_size::Integer, purged_size::Integer,
                              expand_train::Bool, reduce_test::Bool)
        assert_nonempty_nonneg_finite_val(test_size, :test_size)
        assert_nonempty_nonneg_finite_val(train_size, :train_size)
        assert_nonempty_nonneg_finite_val(purged_size, :purged_size)
        return new{typeof(train_size), typeof(test_size), typeof(purged_size),
                   typeof(expand_train), typeof(reduce_test)}(train_size, test_size,
                                                              purged_size, expand_train,
                                                              reduce_test)
    end
end
function IndexWalkForward(train_size::Integer, test_size::Integer; purged_size::Integer = 0,
                          expand_train::Bool = false, reduce_test::Bool = false)
    return IndexWalkForward(train_size, test_size, purged_size, expand_train, reduce_test)
end
"""
    Base.split(iwf::IndexWalkForward, rd::Prices_RR) -> WalkForwardResult

Split the returns data `rd` into sequential walk-forward folds using integer observation
indices. Each fold advances the test window by `test_size` observations.

# Arguments

  - `iwf::IndexWalkForward`: Index-based walk-forward cross-validation estimator.
  - `rd`: Returns-level or price-level data to split ([`Prices_RR`](@ref)).

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
    @argcheck(train_size + purged_size < T,
              DomainError(train_size + purged_size,
                          "train_size + purged_size ($(train_size + purged_size)) must be less than T ($T)"))
    idx = 1:T
    test_start = train_size + purged_size
    train_indices = Vector{typeof(idx)}(undef, 0)
    test_indices = Vector{typeof(idx)}(undef, 0)
    while true
        if test_start >= T
            break
        end
        test_end = test_start + test_size
        train_end = test_start - purged_size
        train_start = expand_train ? 1 : train_end - train_size + 1
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
    (; train_size, test_size, purged_size, reduce_test) = iwf
    T = cv_nobs(rd)
    N = T - train_size - purged_size
    val = div(N, test_size)
    if reduce_test && N % test_size != 0
        val += 1
    end
    return val
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for date adjustment estimators in walk-forward cross-validation.

Subtypes implement specific strategies for adjusting dates used in walk-forward splits.

# Related

  - [`DateWalkForward`](@ref)
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
    ) -> DateWalkForward

Positional and keyword arguments correspond to the struct's fields.

## Validation

  - `test_size` and `purged_size` must be non-empty, non-negative, and finite.
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
    reduce_test ┴ Bool: false
```

# Related

  - [`WalkForwardEstimator`](@ref)
  - [`WalkForwardResult`](@ref)
  - [`n_splits`](@ref)
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
    function DateWalkForward(train_size::IntPeriodDateRange, test_size::Integer,
                             period::DatesUnionPeriod,
                             period_offset::Option{<:DatesUnionPeriod},
                             purged_size::Integer, adjuster::DateAdjType, previous::Bool,
                             expand_train::Bool, reduce_test::Bool)
        assert_nonempty_nonneg_finite_val(test_size, :test_size)
        if isa(train_size, Integer)
            assert_nonempty_nonneg_finite_val(train_size, :train_size)
        end
        assert_nonempty_nonneg_finite_val(purged_size, :purged_size)
        return new{typeof(train_size), typeof(test_size), typeof(period),
                   typeof(period_offset), typeof(purged_size), typeof(adjuster),
                   typeof(previous), typeof(expand_train), typeof(reduce_test)}(train_size,
                                                                                test_size,
                                                                                period,
                                                                                period_offset,
                                                                                purged_size,
                                                                                adjuster,
                                                                                previous,
                                                                                expand_train,
                                                                                reduce_test)
    end
end
function DateWalkForward(train_size::IntPeriodDateRange, test_size::Integer;
                         period::DatesUnionPeriod = Dates.Day(1),
                         period_offset::Option{<:DatesUnionPeriod} = nothing,
                         purged_size::Integer = 0, adjuster::DateAdjType = identity,
                         previous::Bool = false, expand_train::Bool = false,
                         reduce_test::Bool = false)
    return DateWalkForward(train_size, test_size, period, period_offset, purged_size,
                           adjuster, previous, expand_train, reduce_test)
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
    tt = typeof(T)
    idx = Vector{tt}(undef, 0)
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

Perform integer division with a special case for zero divisor.

Returns zero if `b` is zero, otherwise computes `div(a, b)`. Used internally in walk-forward date calculations to avoid division by zero.

# Arguments

  - `a`: Dividend.
  - `b`: Divisor.

# Returns

  - `div(a, b)` or zero if `b == 0`.

# Related

  - [`DateWalkForward`](@ref)
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

    N = 0
    for date in date_range
        i = searchsortedlast(ts, date)
        if !previous && ts[i] != date
            i += 1
        end
        if i > length(ts)
            break
        end
        N += 1
    end
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
    idx = Vector{typeof(T)}(undef, 0)
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
            push!(test_indices, idx[i]:(idx[i + test_size] - purged_size - 1))
        end
        train_start = expand_train ? 1 : train_idx[i]
        push!(train_indices, train_start:(idx[i] - 1))
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

    N = 0
    for date in date_range
        i = searchsortedlast(ts, date)
        if !previous && ts[i] != date
            i += 1
        end
        if i > length(ts)
            break
        end
        N += 1
    end

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
    return special_div(last_allowed_start - M, test_size) + 1
end
function fit_and_predict(opt::OptE_TD, rd::ReturnsResult, cv::WFCVER; cols = :,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                         id = nothing)
    cv_res = split(cv, rd)
    (; train_idx, test_idx) = cv_res
    n = length(train_idx)
    td_flag = is_time_dependent(opt)
    if td_flag
        assert_time_dependent_fold_count(opt, n)
    end
    predictions = run_folds(opt, n, ex) do i, prev
        opti = opt
        # Resolve time-dependent constraints first so freshly swapped-in per-fold
        # constraints also receive the previous weights from the factory pass below.
        if td_flag
            ctx = TimeDependentContext(; i = i, n = n, rd = rd, train_idx = train_idx,
                                       test_idx = test_idx,
                                       w_prev = isnothing(prev) ? nothing : prev.res.w)
            opti = update_time_dependent_estimator(opti, ctx)
        end
        if !isnothing(prev) && needs_previous_weights(opt)
            opti = factory(opti, prev.res.w)
        end
        return fit_and_predict(opti, rd; train_idx = train_idx[i], test_idx = test_idx[i],
                               cols = cols)
    end
    return MultiPeriodPredictionResult(; pred = predictions, id = id)
end
function fit_and_predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                         cv::WFCVER; ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                         id = nothing)
    cv_res = split(cv, rd)
    test_idx = cv_res.test_idx
    predictions = parallel_folds(length(test_idx), ex) do i
        return StatsAPI.predict(res, rd, test_idx[i])
    end
    return MultiPeriodPredictionResult(; pred = predictions, id = id)
end

export WalkForwardResult, IndexWalkForward, DateWalkForward, n_splits
