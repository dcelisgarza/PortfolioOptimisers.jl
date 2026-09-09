"""
$(DocStringExtensions.TYPEDEF)

Carries the observations an estimator keeps when its estimate has no exact incremental fold.

The buffer of the [`partial_fit!`](@ref) seam, and a state like any other: it lives in the estimator's `cache` field, it copies, it slices by asset, and [`obs_weights_view`](@ref) drops it. It holds the observations verbatim, `NaN` included, so a read-out over the buffer answers exactly what a batch fit over the same rows answers, and the Coverage Universe of the two agrees by construction rather than by test.

Two kinds of member hold one. A **carry** folds its estimate exactly and keeps the observations because a consumer downstream reads them — [`LowOrderPrior`](@ref) carries `X` for the scenario risk measures — so the buffer is memory rather than arithmetic. A **refit** has no recursion at all, and its read-out runs the batch verb over the buffer. Neither is a windowed estimator: [`Online`](@ref) is the configuration that seeds a buffer, and `max_history` caps what the buffer keeps and nothing else.

The rows are held in a backing matrix with spare capacity, so an append costs amortised `O(1)`: `off` is the number of rows before the valid region and `n` its length, and the region is moved to the front of the backing matrix only when the append would run past its end. [`sample_buffer`](@ref) reads the valid region out.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SampleBufferState(;
        n::Integer = 0,
        off::Integer = 0,
        X::MatNum = Matrix{Float64}(undef, 0, 0),
        max_history::Option{<:Integer} = nothing
    ) -> SampleBufferState

Keywords correspond to the struct's fields. The default is the empty seed [`Online`](@ref) builds: it carries the cap and no observations, and the first append fixes the width and the element type from the observation it is given.

## Validation

  - `n >= 0`. A `DomainError` is thrown otherwise.
  - `off >= 0`. A `DomainError` is thrown otherwise.
  - `off + n <= size(X, 1)`. A `DimensionMismatch` is thrown otherwise.
  - `max_history > 0` when it is not `nothing`. A `DomainError` is thrown otherwise.
  - `n <= max_history` when `max_history` is not `nothing`. A `DimensionMismatch` is thrown otherwise.

## View parameters

When [`port_opt_view`](@ref) is called on this type, its fields are subset to the selected assets:

  - `X`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.SampleBufferState(; max_history = 2)
PortfolioOptimisers.SampleBufferState
            n ┼ Int64: 0
          off ┼ Int64: 0
            X ┼ 0×0 Matrix{Float64}
  max_history ┴ Int64: 2
```

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`Online`](@ref)
  - [`sample_buffer`](@ref)
  - [`fold_buffer`](@ref)
  - [`partial_fit!`](@ref)
"""
@concrete struct SampleBufferState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    Number of rows of `X` that sit before the valid region. It rises when a capped buffer drops its oldest observation, and returns to zero when the valid region is moved to the front of `X`.
    """
    off
    """
    Backing matrix of the buffer, `capacity × assets`. Rows `off + 1` to `off + n` are the observations, in the order they were folded; the rest is spare capacity and holds no meaning. The width and the element type are fixed by the first append.
    """
    X
    """
    $(field_dict[:pf_max_history])
    """
    max_history
end
function SampleBufferState(; n::Integer = 0, off::Integer = 0,
                           X::MatNum = Matrix{Float64}(undef, 0, 0),
                           max_history::Option{<:Integer} = nothing)::SampleBufferState
    assert_sample_buffer_state(n, off, X, max_history)
    return SampleBufferState(n, off, X, max_history)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a sample buffer whose valid region does not sit inside its backing matrix.

# Algorithm

 1. Refuse a negative `n` and a negative `off`.
 2. Refuse a valid region that runs past the last row of `X`.
 3. Return when `max_history` is `nothing`.
 4. Refuse a non-positive `max_history`, and refuse a valid region longer than it.

# Arguments

  - $(arg_dict[:pf_n])
  - `off`: Number of rows of `X` before the valid region.
  - `X`: Backing matrix of the buffer.
  - $(arg_dict[:pf_max_history])

# Validation

  - `n >= 0`. A `DomainError` is thrown otherwise.
  - `off >= 0`. A `DomainError` is thrown otherwise.
  - `off + n <= size(X, 1)`. A `DimensionMismatch` is thrown otherwise.
  - `max_history > 0` when it is not `nothing`. A `DomainError` is thrown otherwise.
  - `n <= max_history` when `max_history` is not `nothing`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
"""
function assert_sample_buffer_state(n::Integer, off::Integer, X::MatNum,
                                    max_history::Option{<:Integer})
    assert_nonneg(n, :n)
    assert_nonneg(off, :off)
    @argcheck(off + n <= size(X, 1),
              DimensionMismatch("the valid region must sit inside the backing matrix, but `off` is $off, `n` is $n and `X` has $(size(X, 1)) rows."))
    if !isnothing(max_history)
        @argcheck(max_history > zero(max_history),
                  DomainError(max_history, "max_history must be positive"))
        @argcheck(n <= max_history,
                  DimensionMismatch("a capped buffer cannot hold more observations than its cap, but `n` is $n and `max_history` is $max_history."))
    end
    return nothing
end
"""
    sample_buffer(state::SampleBufferState)
    sample_buffer(est)

Reads the observations a sample buffer holds, `observations × assets`.

The read-out of [`SampleBufferState`](@ref). It is a view of the valid region of the backing matrix, in the order the observations were folded, so a refit runs over it without copying it. The estimator form reads the state out of the `cache` field, and refuses an estimator that carries no buffer.

# Arguments

  - `state`: The buffer to read.
  - `est`: Estimator whose `cache` field carries the buffer.

# Validation

  - `est.cache` holds a [`SampleBufferState`](@ref). An `ArgumentError` is thrown otherwise.

# Returns

  - `X::SubArray`: The observations the buffer holds, `observations × assets`.

# Related

  - [`SampleBufferState`](@ref)
  - [`assert_sample_buffer`](@ref)
  - [`Online`](@ref)
"""
function sample_buffer(state::SampleBufferState)
    return view(state.X, (state.off + 1):(state.off + state.n), :)
end
function sample_buffer(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator})
    return sample_buffer(assert_sample_buffer(est))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the sample buffer an estimator carries, and refuses an estimator that carries none.

The buffering arm of [`partial_fit!`](@ref) reads its state through this function, so an estimator that was never wrapped in [`Online`](@ref) meets a message naming the wrapper rather than a `MethodError`. An estimator with no `cache` field at all meets the same message. An unresolved [`Online`](@ref) has a method of its own, because the thing it is missing is not a buffer but the warm-up that would have seeded one.

# Arguments

  - `est`: Estimator whose `cache` field carries the buffer.

# Validation

  - `est` has a `cache` field holding a [`SampleBufferState`](@ref). An `ArgumentError` is thrown otherwise.

# Returns

  - `state::SampleBufferState`: The buffer the estimator carries.

# Related

  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`partial_fit_cache`](@ref)
"""
function assert_sample_buffer(est::Union{<:AbstractEstimator,
                                         <:StatsBase.CovarianceEstimator})
    cache = hasfield(typeof(est), :cache) ? getfield(est, :cache) : nothing
    @argcheck(isa(cache, SampleBufferState),
              ArgumentError("`$(typeof(est))` carries no sample buffer, so it has no incremental fold of its own to fall back on. Wrap it in `Online` and resolve the wrapper with `update_online_estimator` before folding, which seeds the buffer its refit reads."))
    return cache::SampleBufferState
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the [`SampleBufferState`](@ref) a carry folds into, seeding an empty one when the estimator carries none.

The seed is written here rather than inside [`partial_fit!`](@ref), so the fold reads as one line and the branch that reads the `cache` field has one home. A family that carries its observations calls [`fold_buffer`](@ref), which is this seed and the fold in one step; the seed it builds is uncapped, because a cap is configuration and [`Online`](@ref) is what carries it.

# Arguments

  - `cache`: The buffer the estimator carries, or `nothing`.

# Returns

  - `state::SampleBufferState`: The buffer `cache` holds, or an empty uncapped buffer.

# Related

  - [`SampleBufferState`](@ref)
  - [`fold_buffer`](@ref)
  - [`Online`](@ref)
"""
function sample_buffer_seed(cache::Option{<:SampleBufferState})
    return isnothing(cache) ? SampleBufferState() : cache
end
"""
    fold_buffer(cache::Option{<:SampleBufferState}, x::VecNum)
    fold_buffer(cache::Option{<:SampleBufferState}, X::MatNum; dims::Int = 1)

Folds observations into a sample buffer, seeding an empty one when the estimator carries none.

The one line a carry writes inside its own [`partial_fit!`](@ref) method: it folds its estimate exactly through its members, and hands its observations here. It is [`sample_buffer_seed`](@ref) followed by [`partial_fit!`](@ref).

# Arguments

  - `cache`: The buffer the estimator carries, or `nothing`.
  - `x`: One observation, whose entries are the assets.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])

# Returns

  - `state::SampleBufferState`: The buffer after the last observation.

# Related

  - [`SampleBufferState`](@ref)
  - [`sample_buffer_seed`](@ref)
  - [`partial_fit!`](@ref)
"""
function fold_buffer(cache::Option{<:SampleBufferState}, x::VecNum)
    return partial_fit!(sample_buffer_seed(cache), x)
end
function fold_buffer(cache::Option{<:SampleBufferState}, X::MatNum; dims::Int = 1)
    return partial_fit!(sample_buffer_seed(cache), X; dims = dims)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds every observation of a block into a [`SampleBufferState`](@ref).

The block arm of the [`partial_fit!`](@ref) interface, and the whole implementation of the buffer's fold: the single-observation arm reshapes its argument and calls this one. A cap truncates the incoming rows before they are copied, so a block longer than the cap never allocates the whole of it. [`reserve_sample_buffer`](@ref) carries the growth, and its docstring states what makes an append amortised `O(1)`.

# Algorithm

 1. Orient `X` to `observations × assets`, transposing it when `dims == 2`, and return the state unchanged when the block is empty.
 2. Truncate the incoming rows to the last `max_history` of them when a cap is set.
 3. Allocate an exact-fit backing matrix when the buffer is the empty seed, which fixes the width and the element type, and refuse a block whose width is not the width already fixed.
 4. Make room for the block with [`reserve_sample_buffer`](@ref), which drops what the cap pushes out and grows the backing matrix.
 5. Copy the block in after the valid region, and rebind `n` with `Accessors.@reset`.

# Arguments

  - `state`: The buffer to fold into.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])

# Validation

  - $(val_dict[:dims])
  - `X` has the width the first append fixed. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `state::SampleBufferState`: The buffer after the last row.

# Related

  - [`SampleBufferState`](@ref)
  - [`reserve_sample_buffer`](@ref)
  - [`partial_fit!`](@ref)
  - [`sample_buffer`](@ref)
"""
function partial_fit!(state::SampleBufferState, X::MatNum; dims::Int = 1)
    Xo = dims_oriented(dims, X)
    t = size(Xo, 1)
    if iszero(t)
        return state
    end
    w = state.max_history
    if !isnothing(w) && t > w
        Xo = view(Xo, (t - w + 1):t, :)
        t = w
    end
    N = size(Xo, 2)
    B = state.X
    if iszero(state.n) && iszero(size(B, 2))
        state = Accessors.@reset state.X = similar(Xo, t, N)
        copyto!(state.X, Xo)
        return Accessors.@reset state.n = t
    end
    @argcheck(size(B, 2) == N,
              DimensionMismatch("the width of a sample buffer is fixed by its first append, but the buffer describes $(size(B, 2)) assets and the block has $N columns."))
    state = reserve_sample_buffer(state, t)
    copyto!(view(state.X, (state.off + state.n + 1):(state.off + state.n + t), :), Xo)
    return Accessors.@reset state.n = state.n + t
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes room for `t` further observations in a [`SampleBufferState`](@ref), and returns the buffer that has it.

The growth half of the buffer's fold, split out of [`partial_fit!`](@ref) so the fold reads as the orientation, the width check and the copy. It writes no observation: it drops what the cap pushes out, moves the valid region to the front of the backing matrix, and allocates a larger matrix when the front is not enough.

The two steps together are what makes an append amortised `O(1)`. Dropping moves `off` rather than rows, so a capped buffer pays nothing for the observation it forgets. Compaction copies the valid region once, and the growth rule leaves the region filling at most half of the backing matrix, so the next compaction is at least `n` appends away. A capped buffer never allocates more than twice its cap, so a cap of `w` costs at most `2w` rows.

# Algorithm

 1. Drop the oldest observations the cap pushes out, by moving `off` forward and `n` back.
 2. Return when the valid region and the block already fit inside the backing matrix.
 3. Allocate a backing matrix of twice the current capacity, or of the room needed if that is larger, when the valid region and the block would fill more than half of the current one. Clamp it to twice the cap when a cap is set.
 4. Copy the valid region to the front of the backing matrix, which is the old one when no allocation happened, and set `off` to zero.

# Arguments

  - `state`: The buffer to make room in.
  - `t`: Number of observations about to be appended.

# Returns

  - `state::SampleBufferState`: The buffer whose backing matrix has room for `t` rows after its valid region.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
"""
function reserve_sample_buffer(state::SampleBufferState, t::Integer)
    n = state.n
    off = state.off
    w = state.max_history
    if !isnothing(w) && n + t > w
        drop = n + t - w
        off += drop
        n -= drop
        state = Accessors.@reset state.n = n
        state = Accessors.@reset state.off = off
    end
    B = state.X
    cap = size(B, 1)
    if off + n + t <= cap
        return state
    end
    if 2 * (n + t) > cap
        newcap = max(n + t, 2 * cap)
        if !isnothing(w)
            newcap = min(newcap, 2 * w)
        end
        state = Accessors.@reset state.X = similar(B, newcap, size(B, 2))
    end
    D = state.X
    for j in axes(B, 2), i in 1:n
        D[i, j] = B[off + i, j]
    end
    return Accessors.@reset state.off = 0
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds one observation into a [`SampleBufferState`](@ref).

The single-observation arm of the [`partial_fit!`](@ref) interface. It reshapes the observation into a one-row block, which costs no copy, and folds it through the block arm.

# Arguments

  - `state`: The buffer to fold into.
  - `x`: One observation, whose entries are the assets.

# Validation

  - `x` has the width the first append fixed. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `state::SampleBufferState`: The buffer after the observation.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(state::SampleBufferState, x::VecNum)
    return partial_fit!(state, reshape(x, 1, length(x)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds two [`SampleBufferState`](@ref) fitted on disjoint blocks into the buffer of the concatenated block.

Concatenation, which is exact: a buffer holds its observations verbatim, so the buffer of two blocks is the buffer of the rows of one followed by the rows of the other. It is the one buffer state that merges. A cap is applied to the result, which keeps the last `max_history` rows of the concatenation.

[`assert_mergeable_states`](@ref) is deliberately not called. Its array-shape rule reads every array field on every axis, and a buffer's backing matrix carries the observation axis as well as the asset axis, so two buffers over the same assets and different numbers of observations would be refused by it. The width and the cap are checked here instead.

# Algorithm

 1. Refuse two buffers of different widths, and two buffers of different caps.
 2. Concatenate the valid region of the first with the valid region of the second.
 3. Keep the last `max_history` rows when a cap is set.

# Arguments

  - `a`: The buffer of the first block of observations.
  - `b`: The buffer of the second block of observations.

# Validation

  - `a` and `b` describe the same number of assets. A `DimensionMismatch` is thrown otherwise.
  - `a` and `b` carry the same cap. An `ArgumentError` is thrown otherwise.

# Returns

  - `state::SampleBufferState`: The buffer the two blocks give when they are folded as one block.

# Related

  - [`SampleBufferState`](@ref)
  - [`merge_states`](@ref)
  - [`sample_buffer`](@ref)
"""
function merge_states(a::SampleBufferState, b::SampleBufferState)
    Xa = sample_buffer(a)
    Xb = sample_buffer(b)
    @argcheck(size(Xa, 2) == size(Xb, 2),
              DimensionMismatch("two sample buffers over different numbers of assets cannot be merged, but `a` describes $(size(Xa, 2)) assets and `b` describes $(size(Xb, 2))."))
    w = a.max_history
    @argcheck(w == b.max_history,
              ArgumentError("two sample buffers of different caps cannot be merged, but `a` has a `max_history` of $(w) and `b` has $(b.max_history)."))
    X = vcat(Xa, Xb)
    t = size(X, 1)
    if !isnothing(w) && t > w
        X = X[(t - w + 1):t, :]
        t = w
    end
    return SampleBufferState(t, 0, X, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`SampleBufferState`](@ref), so the copy shares no array with the original.

The `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref) calls before it folds. The backing matrix is copied whole, spare capacity included, so the copy appends on the same terms as the original.

# Arguments

  - `x`: The buffer to copy.

# Returns

  - `state::SampleBufferState`: A fresh buffer, equal to `x`, whose backing matrix is a fresh matrix.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::SampleBufferState)
    return SampleBufferState(x.n, x.off, copy(x.X), x.max_history)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Slices a [`SampleBufferState`](@ref) to the selected assets.

A buffer holds its observations verbatim, so the slice of the buffer is the buffer of the sliced universe, column for column, and the observation axis passes through. The slice copies by index and does not `view`: a later [`partial_fit!`](@ref) on the viewed estimator would otherwise write through into the backing matrix of the estimator the view was taken from.

# Arguments

  - `x`: The buffer to slice.
  - `i`: Index or indices of the assets to keep.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `state::SampleBufferState`: The buffer of the same observations over the selected assets.

# Related

  - [`SampleBufferState`](@ref)
  - [`port_opt_view`](@ref)
  - [`partial_fit!`](@ref)
"""
function port_opt_view(x::SampleBufferState, i, args...)
    return SampleBufferState(x.n, x.off, x.X[:, i], x.max_history)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds observations into the sample buffer an estimator carries.

The buffering arm of [`partial_fit!`](@ref), and the method every estimator with no exact incremental fold of its own reaches. A family that folds exactly writes methods of its own for its own type, which are more specific and win on dispatch; this one is what remains, so nothing refuses the step.

It is one method over both arms of the interface rather than two, because the families that refuse the step declare one method over both arms too, and a pair of narrower methods here would be ambiguous against each of them. So the arm is chosen by the type of `X` inside the body, which is statically resolved at every call site.

# Algorithm

 1. Read the buffer out of the `cache` field with [`assert_sample_buffer`](@ref), which refuses an estimator that was never wrapped in [`Online`](@ref).
 2. Fold a matrix through the block arm of [`partial_fit!`](@ref), and a vector through the single-observation arm.
 3. Rebind `est.cache` with `Accessors.@reset`, and return the estimator.

# Arguments

  - `est`: Estimator whose buffer is folded forward.
  - `X`: Observations to fold. A matrix holds one observation per row when `dims == 1`, and one per column when `dims == 2`. A vector is a single observation across the assets, and `dims` is ignored.
  - $(arg_dict[:dims])

# Validation

  - `est` carries a [`SampleBufferState`](@ref). An `ArgumentError` is thrown otherwise.
  - $(val_dict[:dims])

# Returns

  - `est`: The estimator, with its `cache` field rebound to the buffer after the last observation.

# Related

  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator},
                      X::VecNum_MatNum; dims::Int = 1)
    state = assert_sample_buffer(est)
    state = isa(X, MatNum) ? partial_fit!(state, X; dims = dims) : partial_fit!(state, X)
    return Accessors.@reset est.cache = state
end
"""
$(DocStringExtensions.TYPEDEF)

Declares that an estimator takes the online step from a buffer of the observations it has seen.

An `Online` is stored *directly in the estimator field it wraps* — e.g. `EmpiricalPrior(; ce = Online(SomeCovariance()))` — and it is **transient**, in the sense [`TimeDependent`](@ref) is: [`update_online_estimator`](@ref) walks the fields that hold one, seeds each wrapped estimator's `cache` with a [`SampleBufferState`](@ref), and rebuilds the host through its keyword constructor. What comes out is an ordinary estimator carrying a state, and no `Online` exists from that point on, so every verb downstream meets a plain estimator and [ADR 0106](https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/main/docs/adr/0106-a-partial-fit-state-is-the-one-result-an-estimator-holds.md) is untouched.

It differs from [`TimeDependent`](@ref) in when it resolves, and the difference is deliberate. A schedule re-resolves every fold, because its value changes every fold. An `Online` resolves **once**, at warm-up, because after warm-up the state is what is threaded from step to step and a second resolution would throw the buffer away.

Wrap an estimator whose estimate has **no** exact incremental fold of its own, or one that folds exactly and carries its observations for a consumer downstream. An estimator that folds exactly and carries nothing needs no wrapper: it already answers [`partial_fit!`](@ref), seeding its own state on the first call, and wrapping it would put a buffer where its own fold expects its own state.

A wrapper and a [`TimeDependent`](@ref) schedule do not wrap each other, and the difference in *when* they resolve is the whole reason. Neither `Online(TimeDependent(…))` nor a schedule whose entry or `default` is an `Online` is admissible: a wrapper reached through a schedule entry would be resolved at no fold at all, or re-seeded at every fold, throwing away the buffer the step threads. They compose the other way round, and both ways are ordinary. An estimator an `Online` wraps may hold schedules of its own, which survive the seeding untouched and resolve per fold afterwards; and one host may hold a wrapper in one field and a schedule in another, each resolving at its own time. The two field scans are disjoint by construction — a field holding one is invisible to the other's candidate list — so neither resolution can reach the other's wrapper.

`max_history` caps that buffer, and it is a memory knob rather than a mode. Capping bounds the memory the buffer holds and changes what a consumer that reads the observations answers — the scenario risk measures, so CVaR, EVaR and CDaR — while an estimate that folds exactly is unaffected and stays fitted over every observation folded so far.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Online(est; max_history::Option{<:Integer} = nothing)
    Online(; est, max_history::Option{<:Integer} = nothing)

## Validation

  - `est` is not an `Online`. An `ArgumentError` is thrown otherwise.
  - `est` has a `cache` field. An `ArgumentError` is thrown otherwise.
  - `max_history > 0` when it is not `nothing`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> Online(Covariance(; alg = SemiMoment()); max_history = 250)
Online
          est ┼ Covariance
              │    me ┼ SimpleExpectedReturns
              │       │   w ┴ nothing
              │    ce ┼ GeneralCovariance
              │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
              │       │    w ┴ nothing
              │   alg ┼ SemiMoment()
              │     w ┴ nothing
  max_history ┴ Int64: 250
```

# Related

  - [`SampleBufferState`](@ref)
  - [`Online_Option`](@ref)
  - [`update_online_estimator`](@ref)
  - [`online_fields`](@ref)
  - [`TimeDependent`](@ref)
"""
struct Online{T1, T2} <: AbstractEstimator
    """
    Estimator the buffer is seeded on. It is the value the field takes once the wrapper has resolved.
    """
    est::T1
    """
    $(field_dict[:pf_max_history])
    """
    max_history::T2
    function Online(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator};
                    max_history::Option{<:Integer} = nothing)
        @argcheck(hasfield(typeof(est), :cache),
                  ArgumentError("`$(typeof(est))` has no `cache` field, so it has nowhere to carry a sample buffer and cannot be wrapped in `Online`."))
        if !isnothing(max_history)
            @argcheck(max_history > zero(max_history),
                      DomainError(max_history, "max_history must be positive"))
        end
        return new{typeof(est), typeof(max_history)}(est, max_history)
    end
end
function Online(::Online, args...; kwargs...)
    return throw(ArgumentError("est cannot be an Online: wrappers do not nest. One wrapper declares one buffer, and an estimator it wraps may carry wrappers of its own — they resolve at the same warm-up — but they belong in its fields, not inside this one."))
end
function Online(; est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator},
                max_history::Option{<:Integer} = nothing)::Online
    return Online(est; max_history = max_history)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses an [`Online`](@ref) that reached a fold without being resolved, naming the warm-up that resolves it.

[`update_online_estimator`](@ref) replaces every wrapper it can reach at warm-up, so a wrapper that reaches a fold is one the warm-up never saw. The case that produces it is a **callable** [`TimeDependent`](@ref): a schedule's value is computed per fold, after the warm-up has run, so a wrapper the callable returns is never seeded. The vector and `default` forms of a schedule are refused at construction; a callable's return cannot be, because it does not exist until the fold does.

Returning the estimator the wrapper would produce is not the fix either, and [#870](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/870) owns what is: a schedule *replaces* the field's value every fold, while the buffer is threaded *through* it, so a schedule that hands back a captured estimator hands back the buffer as it stood when the closure was built and discards every step since.

# Arguments

  - `::Online`: The unresolved wrapper.

# Validation

  - Always throws an `ArgumentError`.

# Returns

  - Never returns.

# Related

  - [`Online`](@ref)
  - [`update_online_estimator`](@ref)
  - [`assert_sample_buffer`](@ref)
"""
function assert_sample_buffer(::Online)
    return throw(ArgumentError("an `Online` is a declaration, not an estimator that folds, and this one reached a fold unresolved. `update_online_estimator` resolves every wrapper it can reach at warm-up; a wrapper that survives to a fold came from a callable `TimeDependent`, whose value is computed per fold and so is never seeded. A schedule and a wrapper do not compose in that direction: a schedule replaces the field's value every fold, while the buffer is threaded through it, so a schedule that returns a captured estimator returns the buffer as it stood when the closure was built."))
end
"""
    const Online_Option{X} = Union{Nothing, <:Online, X}

Alias for a field that accepts `nothing`, a static estimator of type `X`, or an [`Online`](@ref) declaration.

The set of fields whose constructor signatures use this alias is the single source of truth for which estimators may take the online step from a sample buffer, exactly as [`TD_Option`](@ref) is for the fields that may vary over folds.

# Related

  - [`Online`](@ref)
  - [`Option`](@ref)
  - [`online_fields`](@ref)
"""
const Online_Option{X} = Union{Nothing, <:Online, X}
"""
    online_candidate_fields(x)

Field names of `x` whose *type* admits an [`Online`](@ref) — the candidate set [`online_fields`](@ref) narrows by value.

Whether a field holds a wrapper is decidable from `fieldtype` alone, because a `@concrete` host records the value's type in the field's type parameter, so a field holding a static estimator cannot have a type intersecting [`Online`](@ref). The tuple is therefore computed once per host type by a generated function, and a host that carries no wrapper folds to an empty tuple at compile time rather than walking every field at warm-up.

# Related

  - [`Online`](@ref)
  - [`online_fields`](@ref)
"""
@generated function online_candidate_fields(::T) where {T}
    fns = Tuple(f
                for f in fieldnames(T)
                if typeintersect(fieldtype(T, f), Online) !== Union{})
    return :($fns)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the tuple of field names of `est` whose values are [`Online`](@ref).

The scan is generic over the host's fields, so the widened constructor signatures (see [`Online_Option`](@ref)) remain the single source of truth for which estimators may take the step from a buffer — there is no hand-maintained list. Only the fields whose type admits a wrapper are visited (see [`online_candidate_fields`](@ref)); the rest are ruled out at compile time.

# Arguments

  - `est`: Estimator whose fields are scanned.

# Returns

  - `fns::Tuple`: The field names holding an [`Online`](@ref).

# Related

  - [`Online`](@ref)
  - [`online_candidate_fields`](@ref)
  - [`update_online_estimator`](@ref)
"""
function online_fields(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator})
    fns = online_candidate_fields(est)
    return filter(f -> isa(getfield(est, f), Online), fns)
end
"""
    update_online_estimator(est)

Resolves the [`Online`](@ref) declarations of an estimator, seeding the sample buffer each one asks for.

Called once, at warm-up, before the first fold. Each wrapper is replaced by the estimator it wraps, rebuilt through its keyword constructor with `cache` holding an empty [`SampleBufferState`](@ref) carrying the wrapper's cap; the host is then rebuilt through its own keyword constructor, so every construction invariant re-runs. The result holds no `Online`, and the estimators that were wrapped carry the buffer their refit or their carry reads.

A wrapper is recognised in the fields of the estimator handed over, and in the fields of an estimator a wrapper wraps. A host that hands estimators across a boundary of its own — a meta-optimiser, a pipeline — writes a method of its own that recurses, exactly as it does for [`update_time_dependent_estimator`](@ref).

The verb takes no fold context, and that is the difference from [`update_time_dependent_estimator`](@ref) restated: seeding reads nothing from a fold, because the wrapper resolves once rather than per fold.

# Arguments

  - `est`: Estimator, wrapper, or `nothing`.

# Returns

  - Estimator carrying a seeded buffer wherever a wrapper stood, and holding no [`Online`](@ref).

# Related

  - [`Online`](@ref)
  - [`SampleBufferState`](@ref)
  - [`online_fields`](@ref)
  - [`rebuild_estimator`](@ref)
"""
function update_online_estimator(est::Union{<:AbstractEstimator,
                                            <:StatsBase.CovarianceEstimator})
    fns = online_fields(est)
    if isempty(fns)
        return est
    end
    repl = NamedTuple{fns}(map(f -> update_online_estimator(getfield(est, f)), fns))
    return rebuild_estimator(est, repl)
end
function update_online_estimator(::Nothing)
    return nothing
end
function update_online_estimator(o::Online)
    est = rebuild_estimator(o.est,
                            (; cache = SampleBufferState(; max_history = o.max_history)))
    return update_online_estimator(est)
end

export Online
