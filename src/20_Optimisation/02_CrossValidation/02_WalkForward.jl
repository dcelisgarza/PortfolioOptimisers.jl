const IntPeriodDateRange = Union{<:Integer, <:Dates.Period,
                                 <:AbstractVector{<:Dates.TimeType}}
struct WalkForward{T1, T2, T3, T4, T5, T6, T7, T8, T9} <: CrossValidationEstimator
    train_size::T1
    test_size::T2
    freq::T3
    freq_offset::T4
    purged_size::T5
    adjuster::T6
    previous::T7
    expend_train::T8
    reduce_test::T9
    function WalkForward(train_size::IntPeriodDateRange, test_size::Integer,
                         freq::Option{<:Dates.Period}, freq_offset::Option{<:Dates.Period},
                         purged_size::Integer, adjuster::Option{<:Function}, previous::Bool,
                         expend_train::Bool, reduce_test::Bool)
        assert_nonempty_nonneg_finite_val(test_size, :test_size)
        if isa(train_size, Integer)
            assert_nonempty_nonneg_finite_val(train_size, :train_size)
        elseif isa(train_size, AbstractVector)
            @argcheck(!isempty(train_size), IsEmptyError)
        end
        if isnothing(freq)
            @argcheck(isa(train_size, Integer))
        end
        return new{typeof(train_size), typeof(test_size), typeof(freq), typeof(freq_offset),
                   typeof(purged_size), typeof(adjuster), typeof(previous),
                   typeof(expend_train), typeof(reduce_test)}(train_size, test_size, freq,
                                                              freq_offset, purged_size,
                                                              adjuster, previous,
                                                              expend_train, reduce_test)
    end
end
function WalkForward(train_size::IntPeriodDateRange, test_size::Integer;
                     freq::Option{<:Dates.Period} = nothing,
                     freq_offset::Option{<:Dates.Period} = nothing,
                     purged_size::Integer = 0, adjuster::Option{<:Function} = nothing,
                     previous::Bool = false, expend_train::Bool = false,
                     reduce_test::Bool = false)
    return WalkForward(train_size, test_size, freq, freq_offset, purged_size, adjuster,
                       previous, expend_train, reduce_test)
end
function Base.split(wf::WalkForward{<:Any, <:Any, Nothing}, T::Integer)
    (; train_size, test_size, purged_size, expend_train, reduce_test) = wf
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
function split_from_period(wf::WalkForward{<:Integer}, T::Integer) end
function split_from_period(wf::WalkForward, T::Integer) end
function Base.split(wf::WalkForward, T::Integer)
    return split_from_period(wf, T)
end
function Base.split(wf::WalkForward, X::MatNum; dims::Integer = 1)
    return split(wf, size(X, dims))
end
function Base.split(wf::WalkForward, rd::ReturnsResult; dims::Integer = 1)
    return split(wf, rd.X; dims = dims)
end

export WalkForward
