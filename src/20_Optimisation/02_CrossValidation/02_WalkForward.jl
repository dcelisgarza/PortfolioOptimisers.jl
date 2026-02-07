const IntPeriodDateRange = Union{<:Integer, <:Dates.Period}
struct WalkForward{T1, T2, T3, T4, T5, T6, T7, T8, T9} <: CrossValidationEstimator
    train_size::T1
    test_size::T2
    period::T3
    period_offset::T4
    purged_size::T5
    adjuster::T6
    previous::T7
    expend_train::T8
    reduce_test::T9
    function WalkForward(train_size::IntPeriodDateRange, test_size::Integer,
                         period::Option{<:Union{<:Dates.Period, <:Dates.CompoundPeriod}},
                         period_offset::Option{<:Union{<:Dates.Period,
                                                       <:Dates.CompoundPeriod}},
                         purged_size::Integer, adjuster::Option{<:Function}, previous::Bool,
                         expend_train::Bool, reduce_test::Bool)
        assert_nonempty_nonneg_finite_val(test_size, :test_size)
        if isa(train_size, Integer)
            assert_nonempty_nonneg_finite_val(train_size, :train_size)
        elseif isa(train_size, AbstractVector)
            @argcheck(!isempty(train_size), IsEmptyError)
        end
        if isnothing(period)
            @argcheck(isa(train_size, Integer))
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
function WalkForward(train_size::IntPeriodDateRange, test_size::Integer;
                     period::Option{<:Union{<:Dates.Period, <:Dates.CompoundPeriod}} = nothing,
                     period_offset::Option{<:Union{<:Dates.Period, <:Dates.CompoundPeriod}} = nothing,
                     purged_size::Integer = 0, adjuster::Option{<:Function} = nothing,
                     previous::Bool = false, expend_train::Bool = false,
                     reduce_test::Bool = false)
    return WalkForward(train_size, test_size, period, period_offset, purged_size, adjuster,
                       previous, expend_train, reduce_test)
end
function Base.split(wf::WalkForward{<:Any, <:Any, Nothing}, rd::ReturnsResult)
    (; train_size, test_size, purged_size, expend_train, reduce_test) = wf
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
            push!(test_indices, idx[test_start:end])
        else
            push!(test_indices, idx[test_start:test_end])
        end
        push!(train_indices, idx[train_start:train_end])
        test_start = test_end
    end
    return train_indices, test_indices
end
function split_from_period(wf::WalkForward{<:Integer}, rd::ReturnsResult)
    @argcheck(!isnothing(rd.ts), IsNothingError)
    (; train_size, test_size, period, period_offset, purged_size, previous, expend_train, reduce_test) = wf
    T = size(rd.X, 1)
    ts = rd.ts
    ti = ts[1]
    tf = ts[end]
    po_flag = !isnothing(period_offset)
    if po_flag
        ti = min(ti, ti - period_offset)
    end
    date_range = ti:period:tf
    if po_flag
        date_range = date_range + period_offset
    end
    idx = Int[]
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
    train_indices = Vector{UnitRange{Int}}(undef, 0)
    test_indices = Vector{UnitRange{Int}}(undef, 0)
    while true
        if i + train_size >= N
            break
        end
        if i + train_size + test_size >= N
            if !reduce_test
                break
            end
            push!(test_indices, idx[i + train_size]:T)
        else
            push!(test_indices, idx[i + train_size]:idx[i + train_size + test_size])
        end
        train_start = expend_train ? 1 : idx[i]
        push!(train_indices, train_start:(idx[i + train_size] - purged_size))
        i += test_size
    end

    return train_indices, test_indices
end
function split_from_period(wf::WalkForward{<:Any, <:Any, <:Any}, rd::ReturnsResult)
    @argcheck(!isnothing(rd.ts), IsNothingError)
    (; train_size, test_size, period, period_offset, purged_size, previous, expend_train, reduce_test) = wf
    T = size(rd.X, 1)
    ts = rd.ts
    ti = ts[1]
    tf = ts[end]
    po_flag = !isnothing(period_offset)
    if po_flag
        ti = min(ti, ti - period_offset)
    end
    date_range = ti:period:tf
    if po_flag
        date_range = date_range + period_offset
    end
    idx = Int[]
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

    train_idx = Int[]
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
    train_indices = Vector{UnitRange{Int}}(undef, 0)
    test_indices = Vector{UnitRange{Int}}(undef, 0)
    while true
        if i >= N
            break
        end
        if i + test_size >= N
            if !reduce_test
                break
            end
            push!(test_indices, idx[i]:T)
        else
            push!(test_indices, idx[i]:(idx[i + test_size] - purged_size))
        end
        train_start = expend_train ? 1 : train_idx[i]
        push!(train_indices, train_start:idx[i])
        i += test_size
    end

    return train_indices, test_indices
end
function Base.split(wf::WalkForward, rd::ReturnsResult)
    return split_from_period(wf, rd)
end

export WalkForward
