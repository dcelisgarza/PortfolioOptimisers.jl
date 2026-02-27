abstract type WalkForwardEstimator <: SequentialCrossValidationEstimator end
struct WalkForwardResult{T1, T2} <: SequentialCrossValidationResult
    train_idx::T1
    test_idx::T2
    function WalkForwardResult(train_idx::VecVecInt, test_idx::VecVecInt)
        @argcheck(!isempty(train_idx))
        @argcheck(!isempty(test_idx))
        @argcheck(length(train_idx) == length(test_idx))
        return new{typeof(train_idx), typeof(test_idx)}(train_idx, test_idx)
    end
end
function WalkForwardResult(; train_idx::VecVecInt, test_idx::VecVecInt)
    return WalkForwardResult(train_idx, test_idx)
end
struct IndexWalkForward{T1, T2, T3, T4, T5} <: WalkForwardEstimator
    train_size::T1
    test_size::T2
    purged_size::T3
    expend_train::T4
    reduce_test::T5
    function IndexWalkForward(train_size::Integer, test_size::Integer, purged_size::Integer,
                              expend_train::Bool, reduce_test::Bool)
        assert_nonempty_nonneg_finite_val(test_size, :test_size)
        assert_nonempty_nonneg_finite_val(train_size, :train_size)
        assert_nonempty_nonneg_finite_val(purged_size, :purged_size)
        return new{typeof(train_size), typeof(test_size), typeof(purged_size),
                   typeof(expend_train), typeof(reduce_test)}(train_size, test_size,
                                                              purged_size, expend_train,
                                                              reduce_test)
    end
end
function IndexWalkForward(train_size::Integer, test_size::Integer; purged_size::Integer = 0,
                          expend_train::Bool = false, reduce_test::Bool = false)
    return IndexWalkForward(train_size, test_size, purged_size, expend_train, reduce_test)
end
function Base.split(iwf::IndexWalkForward, rd::ReturnsResult)
    (; train_size, test_size, purged_size, expend_train, reduce_test) = iwf
    T = size(rd.X, 1)
    @argcheck(train_size + purged_size < T)
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
        train_start = expend_train ? 1 : train_end - train_size + 1
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
function n_splits(iwf::IndexWalkForward, rd::ReturnsResult)
    (; train_size, test_size, purged_size, reduce_test) = iwf
    T = size(rd.X, 1)
    N = T - train_size - purged_size
    val = div(N, test_size)
    if reduce_test && N % test_size != 0
        val += 1
    end
    return val
end
abstract type DateAdjusterEstimator <: AbstractEstimator end
const DatesUnionPeriod = Union{<:Dates.Period, <:Dates.CompoundPeriod}
const IntPeriodDateRange = Union{<:Integer, <:DatesUnionPeriod}
const DateAdjType = Union{<:Function, <:DateAdjusterEstimator}
struct DateWalkForward{T1, T2, T3, T4, T5, T6, T7, T8, T9} <: WalkForwardEstimator
    train_size::T1
    test_size::T2
    period::T3
    period_offset::T4
    purged_size::T5
    adjuster::T6
    previous::T7
    expend_train::T8
    reduce_test::T9
    function DateWalkForward(train_size::IntPeriodDateRange, test_size::Integer,
                             period::DatesUnionPeriod,
                             period_offset::Option{<:DatesUnionPeriod},
                             purged_size::Integer, adjuster::DateAdjType, previous::Bool,
                             expend_train::Bool, reduce_test::Bool)
        assert_nonempty_nonneg_finite_val(test_size, :test_size)
        if isa(train_size, Integer)
            assert_nonempty_nonneg_finite_val(train_size, :train_size)
        end
        assert_nonempty_nonneg_finite_val(purged_size, :purged_size)
        return new{typeof(train_size), typeof(test_size), typeof(period),
                   typeof(period_offset), typeof(purged_size), typeof(adjuster),
                   typeof(previous), typeof(expend_train), typeof(reduce_test)}(train_size,
                                                                                test_size,
                                                                                period,
                                                                                period_offset,
                                                                                purged_size,
                                                                                adjuster,
                                                                                previous,
                                                                                expend_train,
                                                                                reduce_test)
    end
end
function DateWalkForward(train_size::IntPeriodDateRange, test_size::Integer;
                         period::DatesUnionPeriod = Dates.Day(1),
                         period_offset::Option{<:DatesUnionPeriod} = nothing,
                         purged_size::Integer = 0, adjuster::DateAdjType = identity,
                         previous::Bool = false, expend_train::Bool = false,
                         reduce_test::Bool = false)
    return DateWalkForward(train_size, test_size, period, period_offset, purged_size,
                           adjuster, previous, expend_train, reduce_test)
end
function Base.split(dwf::DateWalkForward{<:Integer}, rd::ReturnsResult)
    @argcheck(!isnothing(rd.ts), IsNothingError)
    (; train_size, test_size, period, period_offset, purged_size, adjuster, previous, expend_train, reduce_test) = dwf
    T = size(rd.X, 1)
    ts = rd.ts
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
        if !previous && ts[i] != date
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
        train_start = expend_train ? 1 : idx[i]
        push!(train_indices, train_start:(idx[i + train_size] - purged_size - 1))
        i += test_size
    end

    return WalkForwardResult(; train_idx = train_indices, test_idx = test_indices)
end
function special_div(a::Integer, b::Integer)
    q, r = divrem(a, b)
    return q - ifelse(iszero(r), 1, 0)
end
function n_splits(dwf::DateWalkForward{<:Integer}, rd::ReturnsResult)
    @argcheck(!isnothing(rd.ts), IsNothingError)
    (; train_size, test_size, period, period_offset, adjuster, previous, reduce_test) = dwf
    ts = rd.ts
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
function Base.split(dwf::DateWalkForward{<:Any}, rd::ReturnsResult)
    @argcheck(!isnothing(rd.ts), IsNothingError)
    (; train_size, test_size, period, period_offset, purged_size, adjuster, previous, expend_train, reduce_test) = dwf
    T = size(rd.X, 1)
    ts = rd.ts
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

    idx = typeof(T)[]
    for date in date_range
        i = searchsortedlast(ts, date)
        if !previous && ts[i] != date
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
        train_start = expend_train ? 1 : train_idx[i]
        push!(train_indices, train_start:(idx[i] - 1))
        i += test_size
    end

    return WalkForwardResult(; train_idx = train_indices, test_idx = test_indices)
end
function n_splits(dwf::DateWalkForward{<:Any}, rd::ReturnsResult)
    @argcheck(!isnothing(rd.ts), IsNothingError)
    (; train_size, test_size, period, period_offset, adjuster, previous, reduce_test) = dwf
    ts = rd.ts
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
function fit_and_predict(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult,
                         cv::WalkForwardEstimator; cols = :,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    cv_res = split(cv, rd)
    (; train_idx, test_idx) = cv_res
    predictions = Vector{PredictionResult}(undef, length(train_idx))
    if needs_previous_weights(opt)
        @info("Running walk forward sequentially because the optimiser must use the previous optimisation's weights. This is because somewhere within the optimisation estimator is contained at least one of the following:\n\t- Turnover and/or TurnoverEstimator,\n\t- WeightsTracking,\n\t- TurnoverRiskMeasure,\n\t- custom constraints which use asset weights,\n\t- custom objective penalties which use asset weights.\nTo enable parallel processing please either mark the weights as fixed or remove the component(s) which use(s) them.")
        for (i, (train, test)) in enumerate(zip(train_idx, test_idx))
            if i > 1
                opt = factory(opt, predictions[i - 1].res.w)
            end
            predictions[i] = fit_and_predict(opt, rd; train_idx = train, test_idx = test,
                                             cols = cols)
        end
    else
        let opt = opt
            FLoops.@floop ex for (i, (train, test)) in enumerate(zip(train_idx, test_idx))
                predictions[i] = fit_and_predict(opt, rd; train_idx = train,
                                                 test_idx = test, cols = cols)
            end
        end
    end
    return MultiPeriodPredictionResult(; pred = predictions)
end
function fit_and_predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                         cv::WalkForwardEstimator;
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    cv_res = split(cv, rd)
    test_idx = cv_res.test_idx
    predictions = Vector{PredictionResult}(undef, length(test_idx))
    FLoops.@floop ex for (i, test) in enumerate(test_idx)
        predictions[i] = predict(res, rd, test)
    end
    return MultiPeriodPredictionResult(; pred = predictions)
end

export WalkForwardResult, IndexWalkForward, DateWalkForward, n_splits
