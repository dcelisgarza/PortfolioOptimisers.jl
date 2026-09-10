"""
$(DocStringExtensions.TYPEDEF)

Carries the observations an estimator keeps when its estimate has no exact incremental fold.

The buffer of the [`partial_fit!`](@ref) seam, and a state like any other: it lives in the estimator's `cache` field, it copies, it slices by asset, and [`obs_weights_view`](@ref) drops it. It holds the observations verbatim, `NaN` included, so a read-out over the buffer answers exactly what a batch fit over the same rows answers, and the Coverage Universe of the two agrees by construction rather than by test.

It holds the **per-observation masks** verbatim on the same terms. A [`CoveragePolicy`](@ref) reads two facts out of an active mask that the rows alone do not carry — a cell that is finite but inactive is excluded, and an asset active at one observation and inactive at the next is a delisting rather than a holiday — so a buffer that kept the rows and dropped the mask would answer a different question from the batch fit, silently. `A` is that mask, `E` is the estimation mask the two regime-adjusted families read, and each is a backing matrix of the shape of `X` or `nothing`. A wrapped estimator folded under a policy therefore matches a batch fit over the same window exactly, as it already does without one.

Two kinds of member hold one. A **carry** folds its estimate exactly and keeps the observations because a consumer downstream reads them — [`LowOrderPrior`](@ref) carries `X` for the scenario risk measures — so the buffer is memory rather than arithmetic. A **refit** has no recursion at all, and its read-out runs the batch verb over the buffer. Neither is a windowed estimator: [`Online`](@ref) is the configuration that seeds a buffer, and `max_history` caps what the buffer keeps and nothing else.

The rows are held in a backing matrix with spare capacity, so an append costs amortised `O(1)`: `off` is the number of rows before the valid region and `n` its length, and the region is moved to the front of the backing matrix only when the append would run past its end. [`sample_buffer`](@ref) reads the valid region out.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SampleBufferState(;
        n::Integer = 0,
        off::Integer = 0,
        X::MatNum = Matrix{Float64}(undef, 0, 0),
        A::Option{<:AbstractMatrix{<:Bool}} = nothing,
        E::Option{<:AbstractMatrix{<:Bool}} = nothing,
        max_history::Option{<:Integer} = nothing
    ) -> SampleBufferState

Keywords correspond to the struct's fields. The default is the empty seed [`Online`](@ref) builds: it carries the cap and no observations, and the first append fixes the width, the element type and the masks from the observations it is given.

## Validation

  - `n >= 0`. A `DomainError` is thrown otherwise.
  - `off >= 0`. A `DomainError` is thrown otherwise.
  - `off + n <= size(X, 1)`. A `DimensionMismatch` is thrown otherwise.
  - `max_history > 0` when it is not `nothing`. A `DomainError` is thrown otherwise.
  - `n <= max_history` when `max_history` is not `nothing`. A `DimensionMismatch` is thrown otherwise.
  - `A` and `E`, when they are not `nothing`, have the shape of `X`. A `DimensionMismatch` is thrown otherwise.

## View parameters

When [`port_opt_view`](@ref) is called on this type, its fields are subset to the selected assets:

  - `X`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `A`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `E`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.SampleBufferState(; max_history = 2)
PortfolioOptimisers.SampleBufferState
            n ┼ Int64: 0
          off ┼ Int64: 0
            X ┼ 0×0 Matrix{Float64}
            A ┼ nothing
            E ┼ nothing
  max_history ┴ Int64: 2
```

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`Online`](@ref)
  - [`sample_buffer`](@ref)
  - [`sample_buffer_kwargs`](@ref)
  - [`fold_buffer`](@ref)
  - [`partial_fit!`](@ref)
  - [`CoveragePolicy`](@ref)
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
    $(field_dict[:pf_buffer_A])
    """
    A
    """
    $(field_dict[:pf_buffer_E])
    """
    E
    """
    $(field_dict[:pf_max_history])
    """
    max_history
end
function SampleBufferState(; n::Integer = 0, off::Integer = 0,
                           X::MatNum = Matrix{Float64}(undef, 0, 0),
                           A::Option{<:AbstractMatrix{<:Bool}} = nothing,
                           E::Option{<:AbstractMatrix{<:Bool}} = nothing,
                           max_history::Option{<:Integer} = nothing)::SampleBufferState
    assert_sample_buffer_state(n, off, X, max_history)
    assert_buffer_mask_shape(X, A, E)
    return SampleBufferState(n, off, X, A, E, max_history)
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
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a per-observation mask whose shape is not the shape of the observations it belongs to.

A mask a buffer holds is cell for cell with the observations it holds, and a mask a fold is given is cell for cell with the block it is given, so one rule covers both: a mask is either absent or of the shape of the matrix beside it. It is the check the batch verbs run on their own arguments, moved to where the buffer stores them.

# Algorithm

 1. Return for a mask that is `nothing`, which is the whole of the no-policy case.
 2. Refuse a mask whose size is not `size(X)`, naming the mask.

# Arguments

  - `X`: The observations the masks belong to.
  - `A`: The active mask, or `nothing`.
  - `E`: The estimation mask, or `nothing`.

# Validation

  - `A` and `E`, when they are not `nothing`, have the shape of `X`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function assert_buffer_mask_shape(X::AbstractMatrix, A::Option{<:AbstractMatrix{<:Bool}},
                                  E::Option{<:AbstractMatrix{<:Bool}})
    if !isnothing(A)
        @argcheck(size(A) == size(X),
                  DimensionMismatch("size(X) ($(size(X))) must match size(active_mask) ($(size(A)))"))
    end
    if !isnothing(E)
        @argcheck(size(E) == size(X),
                  DimensionMismatch("size(X) ($(size(X))) must match size(estimation_mask) ($(size(E)))"))
    end
    return nothing
end
"""
    buffer_mask_view(M::Nothing, rows) -> Nothing
    buffer_mask_view(M::AbstractMatrix{<:Bool}, rows) -> SubArray

Reads a row range out of a per-observation mask, and passes a mask that is not there through.

The one place the buffer slices a mask on the observation axis, so the `nothing` case is written once rather than at each of its callers. [`sample_buffer_kwargs`](@ref) reads the valid region of a stored mask with it, and [`partial_fit!`](@ref) truncates an incoming block with it.

# Arguments

  - `M`: The mask to slice, or `nothing`.
  - `rows`: The rows to keep.

# Returns

  - `M`: The rows of the mask, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`sample_buffer_kwargs`](@ref)
  - [`partial_fit!`](@ref)
"""
function buffer_mask_view(::Nothing, args...)
    return nothing
end
function buffer_mask_view(M::AbstractMatrix{<:Bool}, rows)
    return view(M, rows, :)
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

Reads the per-observation masks a sample buffer holds, as the keywords a batch verb takes.

The second half of the buffer's read-out. [`sample_buffer`](@ref) gives the rows and this gives the masks that explain them, so a read-out arm is the batch verb over the one, splatted with the other, and the wrapped answer is the unwrapped answer over the same window. A buffer that records no mask gives an empty set of keywords, which is why the no-policy case costs nothing: the arm reduces to the call it has always made, and the branch is on a field whose type is concrete, so it is resolved at compile time.

# Algorithm

 1. Slice each stored mask to the valid region with [`buffer_mask_view`](@ref).
 2. Name the masks that are there, and omit the ones that are not.

# Arguments

  - `state`: The buffer to read.

# Returns

  - `kwargs::NamedTuple`: The `active_mask` and `estimation_mask` keywords the buffer holds, of which either or both may be absent.

# Related

  - [`SampleBufferState`](@ref)
  - [`sample_buffer`](@ref)
  - [`buffer_mask_view`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function sample_buffer_kwargs(state::SampleBufferState)
    rows = (state.off + 1):(state.off + state.n)
    A = buffer_mask_view(state.A, rows)
    E = buffer_mask_view(state.E, rows)
    return if isnothing(A)
        isnothing(E) ? (;) : (; estimation_mask = E)
    elseif isnothing(E)
        (; active_mask = A)
    else
        (; active_mask = A, estimation_mask = E)
    end
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
    fold_buffer(cache::Option{<:SampleBufferState}, x::VecNum; kwargs...)
    fold_buffer(cache::Option{<:SampleBufferState}, X::MatNum; dims::Int = 1, kwargs...)

Folds observations into a sample buffer, seeding an empty one when the estimator carries none.

The one line a carry writes inside its own [`partial_fit!`](@ref) method: it folds its estimate exactly through its members, and hands its observations here. It is [`sample_buffer_seed`](@ref) followed by [`partial_fit!`](@ref). The masks of the fold ride with the observations, so a carry hands over the mask it was given and the buffer records it beside the rows it explains.

# Arguments

  - `cache`: The buffer the estimator carries, or `nothing`.
  - `x`: One observation, whose entries are the assets.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: The `active_mask` and `estimation_mask` of the fold, forwarded to [`partial_fit!`](@ref).

# Returns

  - `state::SampleBufferState`: The buffer after the last observation.

# Related

  - [`SampleBufferState`](@ref)
  - [`sample_buffer_seed`](@ref)
  - [`partial_fit!`](@ref)
"""
function fold_buffer(cache::Option{<:SampleBufferState}, x::VecNum; kwargs...)
    return partial_fit!(sample_buffer_seed(cache), x; kwargs...)
end
function fold_buffer(cache::Option{<:SampleBufferState}, X::MatNum; dims::Int = 1,
                     kwargs...)
    return partial_fit!(sample_buffer_seed(cache), X; dims = dims, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds every observation of a block into a [`SampleBufferState`](@ref).

The block arm of the [`partial_fit!`](@ref) interface, and the whole implementation of the buffer's fold: the single-observation arm reshapes its argument and calls this one. A cap truncates the incoming rows before they are copied, so a block longer than the cap never allocates the whole of it. [`reserve_sample_buffer`](@ref) carries the growth, and its docstring states what makes an append amortised `O(1)`.

The masks travel with the rows they explain, cell for cell, through every step: the orientation, the truncation, the growth and the copy. Whether the buffer records a mask at all is fixed by its first append, exactly as the width and the element type are, and a later fold that disagrees is refused by [`assert_buffer_mask_agreement`](@ref) — a buffer whose activity is known for some of its rows and not for others answers neither question. A buffer holding no observations has nothing to disagree with, so [`reset_empty_buffer`](@ref) empties it and the fold seeds it afresh.

# Algorithm

 1. Orient `X` and the masks to `observations × assets`, transposing them when `dims == 2`, refuse a mask that is not of the shape of the block, and return the state unchanged when the block is empty.
 2. Truncate the incoming rows and their masks to the last `max_history` of them when a cap is set.
 3. Empty a buffer that holds no observations and disagrees with the fold about the masks, so that the seed below fixes them afresh.
 4. Seed the buffer with [`seed_sample_buffer`](@ref) when it is the empty seed, which fixes the width, the element type and the masks; and refuse a block whose width is not the width already fixed, or whose masks disagree with the ones the buffer holds.
 5. Make room for the block with [`reserve_sample_buffer`](@ref), which drops what the cap pushes out and grows the backing matrices.
 6. Copy the block and its masks in after the valid region, and rebind `n` with `Accessors.@reset`.

# Arguments

  - `state`: The buffer to fold into.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:pf_active_mask])
  - $(arg_dict[:pf_estimation_mask])

# Validation

  - $(val_dict[:dims])
  - `X` has the width the first append fixed. A `DimensionMismatch` is thrown otherwise.
  - `active_mask` and `estimation_mask`, when they are not `nothing`, have the shape of `X`. A `DimensionMismatch` is thrown otherwise.
  - A buffer holding observations is given the masks it already records. An `ArgumentError` is thrown otherwise.

# Returns

  - `state::SampleBufferState`: The buffer after the last row.

# Related

  - [`SampleBufferState`](@ref)
  - [`reserve_sample_buffer`](@ref)
  - [`seed_sample_buffer`](@ref)
  - [`assert_buffer_mask_agreement`](@ref)
  - [`partial_fit!`](@ref)
  - [`sample_buffer`](@ref)
"""
function partial_fit!(state::SampleBufferState, X::MatNum; dims::Int = 1,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                      estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing)
    Xo, Ao, Eo = dims_oriented(dims, X, active_mask, estimation_mask)
    assert_buffer_mask_shape(Xo, Ao, Eo)
    t = size(Xo, 1)
    if iszero(t)
        return state
    end
    w = state.max_history
    if !isnothing(w) && t > w
        rows = (t - w + 1):t
        Xo = view(Xo, rows, :)
        Ao = buffer_mask_view(Ao, rows)
        Eo = buffer_mask_view(Eo, rows)
        t = w
    end
    N = size(Xo, 2)
    state = reset_empty_buffer(state, Ao, Eo)
    B = state.X
    if iszero(state.n) && iszero(size(B, 2))
        return seed_sample_buffer(state, Xo, Ao, Eo)
    end
    @argcheck(size(B, 2) == N,
              DimensionMismatch("the width of a sample buffer is fixed by its first append, but the buffer describes $(size(B, 2)) assets and the block has $N columns."))
    assert_buffer_mask_agreement(state.A, Ao, "active_mask")
    assert_buffer_mask_agreement(state.E, Eo, "estimation_mask")
    state = reserve_sample_buffer(state, t)
    rows = (state.off + state.n + 1):(state.off + state.n + t)
    copyto!(view(state.X, rows, :), Xo)
    copy_buffer_mask!(state.A, rows, Ao)
    copy_buffer_mask!(state.E, rows, Eo)
    return Accessors.@reset state.n = state.n + t
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a fold whose masks disagree with the ones the buffer already records.

The mixture rule of the buffer, and it refuses rather than resolves because neither resolution is answerable. A buffer that dropped an incoming mask would fit a held gap the policy excludes and would read every delisting as a holiday, which is the silent wrong answer the mask exists to prevent. A buffer that invented a mask for the rows folded before it would claim an activity it never saw. So the mask a buffer records is fixed by its first append, and a caller that changes its mind starts a fresh buffer.

# Arguments

  - `M`: The mask the buffer records, or `nothing`.
  - `Mo`: The mask of the incoming block, or `nothing`.
  - `name`: Name of the mask, for the message.

# Validation

  - `M` and `Mo` are both `nothing`, or neither is. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`reset_empty_buffer`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function assert_buffer_mask_agreement(M::Option{<:AbstractMatrix{<:Bool}},
                                      Mo::Option{<:AbstractMatrix{<:Bool}},
                                      name::AbstractString)
    @argcheck(isnothing(M) == isnothing(Mo),
              ArgumentError("a sample buffer records `$name` for every observation it holds or for none of them, and this fold $(isnothing(Mo) ? "gives none where the buffer records one" : "gives one where the buffer records none"). A buffer whose activity is known for some of its rows and not for others answers neither the question the mask asks nor the question it does not. Fold the whole run with the mask or fold it without, or seed a fresh buffer."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Empties a buffer that holds no observations and does not record the masks the fold is about to give it.

The one case the mixture rule does not bite. A buffer with no rows records nothing about them, so it has no answer to lose and no claim to invent, and emptying it lets [`seed_sample_buffer`](@ref) fix the masks as it fixes the width and the element type. The element type of the backing matrix survives, because `similar` reads it off the matrix it empties.

# Arguments

  - `state`: The buffer to empty.
  - `Ao`: The active mask of the incoming block, or `nothing`.
  - `Eo`: The estimation mask of the incoming block, or `nothing`.

# Returns

  - `state::SampleBufferState`: The buffer unchanged, or the empty seed of the same element type and cap.

# Related

  - [`SampleBufferState`](@ref)
  - [`seed_sample_buffer`](@ref)
  - [`assert_buffer_mask_agreement`](@ref)
  - [`partial_fit!`](@ref)
"""
function reset_empty_buffer(state::SampleBufferState, Ao::Option{<:AbstractMatrix{<:Bool}},
                            Eo::Option{<:AbstractMatrix{<:Bool}})
    if !iszero(state.n) ||
       (isnothing(state.A) == isnothing(Ao) && isnothing(state.E) == isnothing(Eo))
        return state
    end
    state = Accessors.@reset state.X = similar(state.X, 0, 0)
    state = Accessors.@reset state.A = nothing
    state = Accessors.@reset state.E = nothing
    return Accessors.@reset state.off = 0
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fills the empty seed of a buffer from its first block, fixing the width, the element type and the masks.

The first append of a buffer, split out of [`partial_fit!`](@ref) so that the fold reads as the orientation, the truncation and the copy. The backing matrices are an exact fit for the block, so a buffer that is folded once and read once allocates nothing it does not use, and every array is built with `similar` from the argument it holds, so no element type is chosen here.

# Arguments

  - `state`: The empty seed to fill.
  - `Xo`: The oriented block.
  - `Ao`: The oriented active mask of the block, or `nothing`.
  - `Eo`: The oriented estimation mask of the block, or `nothing`.

# Returns

  - `state::SampleBufferState`: The buffer holding the block and its masks.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`reset_empty_buffer`](@ref)
"""
function seed_sample_buffer(state::SampleBufferState, Xo::MatNum,
                            Ao::Option{<:AbstractMatrix{<:Bool}},
                            Eo::Option{<:AbstractMatrix{<:Bool}})
    state = Accessors.@reset state.X = copyto!(similar(Xo, size(Xo)...), Xo)
    state = Accessors.@reset state.A = seed_buffer_mask(Ao)
    state = Accessors.@reset state.E = seed_buffer_mask(Eo)
    return Accessors.@reset state.n = size(Xo, 1)
end
"""
    seed_buffer_mask(Mo::Nothing) -> Nothing
    seed_buffer_mask(Mo::AbstractMatrix{<:Bool}) -> AbstractMatrix{<:Bool}

Copies an incoming mask into a backing matrix of its own, and passes a mask that is not there through.

A mask a fold is given is the caller's array, and the buffer outlives the call, so the buffer takes a copy for the same reason [`Base.copy`](@ref) and [`port_opt_view`](@ref) do: a later fold writes into the backing matrix, and writing through into a caller's mask would be a defect the caller cannot see.

# Arguments

  - `Mo`: The oriented mask of the block, or `nothing`.

# Returns

  - `M`: A backing matrix holding the mask, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`seed_sample_buffer`](@ref)
"""
function seed_buffer_mask(::Nothing)
    return nothing
end
function seed_buffer_mask(Mo::AbstractMatrix{<:Bool})
    return copyto!(similar(Mo, Bool, size(Mo)...), Mo)
end
"""
    copy_buffer_mask!(M::Nothing, rows, Mo) -> Nothing
    copy_buffer_mask!(M::AbstractMatrix{<:Bool}, rows, Mo::AbstractMatrix{<:Bool}) -> AbstractMatrix{<:Bool}

Copies an incoming mask into the rows of the backing matrix that were just made room for.

The mask half of the append, written once so that the two masks and the `nothing` case cost the fold one line each. It is only ever reached with the two arguments agreeing, because [`assert_buffer_mask_agreement`](@ref) has run.

# Arguments

  - `M`: The buffer's backing matrix for this mask, or `nothing`.
  - `rows`: The rows of `M` to write.
  - `Mo`: The oriented mask of the block, or `nothing`.

# Returns

  - `M`: The rows written, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`assert_buffer_mask_agreement`](@ref)
"""
function copy_buffer_mask!(::Nothing, args...)
    return nothing
end
function copy_buffer_mask!(M::AbstractMatrix{<:Bool}, rows, Mo::AbstractMatrix{<:Bool})
    return copyto!(view(M, rows, :), Mo)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes room for `t` further observations in a [`SampleBufferState`](@ref), and returns the buffer that has it.

The growth half of the buffer's fold, split out of [`partial_fit!`](@ref) so the fold reads as the orientation, the width check and the copy. It writes no observation: it drops what the cap pushes out, moves the valid region to the front of the backing matrices, and allocates larger ones when the front is not enough.

The two steps together are what makes an append amortised `O(1)`. Dropping moves `off` rather than rows, so a capped buffer pays nothing for the observation it forgets, and it forgets the masks of that observation on the same terms because they are indexed by the same `off` and `n`. Compaction copies the valid region once, and the growth rule leaves the region filling at most half of the backing matrix, so the next compaction is at least `n` appends away. A capped buffer never allocates more than twice its cap, so a cap of `w` costs at most `2w` rows.

# Algorithm

 1. Drop the oldest observations the cap pushes out, by moving `off` forward and `n` back.
 2. Return when the valid region and the block already fit inside the backing matrix.
 3. Take a capacity of twice the current one, or of the room needed if that is larger, when the valid region and the block would fill more than half of the current one. Clamp it to twice the cap when a cap is set.
 4. Compact the observations and each mask the buffer records to that capacity with [`compact_buffer_array`](@ref), which allocates only when the capacity changes, and set `off` to zero.

# Arguments

  - `state`: The buffer to make room in.
  - `t`: Number of observations about to be appended.

# Returns

  - `state::SampleBufferState`: The buffer whose backing matrices have room for `t` rows after their valid region.

# Related

  - [`SampleBufferState`](@ref)
  - [`compact_buffer_array`](@ref)
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
    newcap = cap
    if 2 * (n + t) > cap
        newcap = max(n + t, 2 * cap)
        if !isnothing(w)
            newcap = min(newcap, 2 * w)
        end
    end
    state = Accessors.@reset state.X = compact_buffer_array(B, off, n, newcap)
    state = Accessors.@reset state.A = compact_buffer_array(state.A, off, n, newcap)
    state = Accessors.@reset state.E = compact_buffer_array(state.E, off, n, newcap)
    return Accessors.@reset state.off = 0
end
"""
    compact_buffer_array(B::Nothing, off, n, newcap) -> Nothing
    compact_buffer_array(B::AbstractMatrix, off::Integer, n::Integer, newcap::Integer) -> AbstractMatrix

Moves the valid region of one backing matrix to the front, allocating a matrix of `newcap` rows when the capacity changes.

The array half of [`reserve_sample_buffer`](@ref), written once because the buffer carries three backing matrices that are indexed alike and compacted alike. Copying forward with an increasing row index is safe in place, because the destination row is never past the source row.

# Arguments

  - `B`: The backing matrix to compact, or `nothing`.
  - `off`: Number of rows before the valid region.
  - `n`: Length of the valid region.
  - `newcap`: Number of rows the compacted matrix has.

# Returns

  - `D`: The compacted matrix, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`reserve_sample_buffer`](@ref)
"""
function compact_buffer_array(::Nothing, args...)
    return nothing
end
function compact_buffer_array(B::AbstractMatrix, off::Integer, n::Integer, newcap::Integer)
    D = size(B, 1) == newcap ? B : similar(B, newcap, size(B, 2))
    for j in axes(B, 2), i in 1:n
        D[i, j] = B[off + i, j]
    end
    return D
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds one observation into a [`SampleBufferState`](@ref).

The single-observation arm of the [`partial_fit!`](@ref) interface. It reshapes the observation into a one-row block, which costs no copy, and folds it through the block arm. The masks of the observation are reshaped on the same terms, so one observation under a policy folds through the same code the block does.

# Arguments

  - `state`: The buffer to fold into.
  - `x`: One observation, whose entries are the assets.
  - `active_mask`: The active mask of the observation, one entry per asset, or `nothing`.
  - `estimation_mask`: The estimation mask of the observation, one entry per asset, or `nothing`.

# Validation

  - `x` has the width the first append fixed. A `DimensionMismatch` is thrown otherwise.
  - `active_mask` and `estimation_mask`, when they are not `nothing`, have one entry per asset. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `state::SampleBufferState`: The buffer after the observation.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(state::SampleBufferState, x::VecNum;
                      active_mask::Option{<:AbstractVector{<:Bool}} = nothing,
                      estimation_mask::Option{<:AbstractVector{<:Bool}} = nothing)
    return partial_fit!(state, reshape(x, 1, length(x));
                        active_mask = observation_mask_row(active_mask),
                        estimation_mask = observation_mask_row(estimation_mask))
end
"""
    observation_mask_row(m::Nothing) -> Nothing
    observation_mask_row(m::AbstractVector{<:Bool}) -> AbstractMatrix{<:Bool}

Reshapes the mask of one observation into a one-row block, and passes a mask that is not there through.

The mask half of the single-observation arm of [`partial_fit!`](@ref). It costs no copy, so one observation folded under a policy allocates no more than one folded without.

# Arguments

  - `m`: The mask of one observation, one entry per asset, or `nothing`.

# Returns

  - `M`: The mask as a one-row block, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
"""
function observation_mask_row(::Nothing)
    return nothing
end
function observation_mask_row(m::AbstractVector{<:Bool})
    return reshape(m, 1, length(m))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds two [`SampleBufferState`](@ref) fitted on disjoint blocks into the buffer of the concatenated block.

Concatenation, which is exact: a buffer holds its observations verbatim, so the buffer of two blocks is the buffer of the rows of one followed by the rows of the other. It is the one buffer state that merges. A cap is applied to the result, which keeps the last `max_history` rows of the concatenation. The masks concatenate with the rows they explain, and two buffers that disagree about which masks they record are refused, on the reasoning [`assert_buffer_mask_agreement`](@ref) states for a fold.

[`assert_mergeable_states`](@ref) is deliberately not called. Its array-shape rule reads every array field on every axis, and a buffer's backing matrix carries the observation axis as well as the asset axis, so two buffers over the same assets and different numbers of observations would be refused by it. The width, the cap and the masks are checked here instead.

# Algorithm

 1. Refuse two buffers of different widths, and two buffers of different caps.
 2. Refuse two buffers that do not record the same masks.
 3. Concatenate the valid region of the first with the valid region of the second, and each mask with its own.
 4. Keep the last `max_history` rows when a cap is set.

# Arguments

  - `a`: The buffer of the first block of observations.
  - `b`: The buffer of the second block of observations.

# Validation

  - `a` and `b` describe the same number of assets. A `DimensionMismatch` is thrown otherwise.
  - `a` and `b` carry the same cap. An `ArgumentError` is thrown otherwise.
  - `a` and `b` record the same masks. An `ArgumentError` is thrown otherwise.

# Returns

  - `state::SampleBufferState`: The buffer the two blocks give when they are folded as one block.

# Related

  - [`SampleBufferState`](@ref)
  - [`merge_states`](@ref)
  - [`assert_buffer_mask_agreement`](@ref)
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
    assert_buffer_mask_agreement(a.A, b.A, "active_mask")
    assert_buffer_mask_agreement(a.E, b.E, "estimation_mask")
    ra = (a.off + 1):(a.off + a.n)
    rb = (b.off + 1):(b.off + b.n)
    X = vcat(Xa, Xb)
    A = merge_buffer_mask(buffer_mask_view(a.A, ra), buffer_mask_view(b.A, rb))
    E = merge_buffer_mask(buffer_mask_view(a.E, ra), buffer_mask_view(b.E, rb))
    t = size(X, 1)
    if !isnothing(w) && t > w
        rows = (t - w + 1):t
        X = X[rows, :]
        A = trim_merged_mask(A, rows)
        E = trim_merged_mask(E, rows)
        t = w
    end
    return SampleBufferState(t, 0, X, A, E, w)
end
"""
    merge_buffer_mask(Ma::Nothing, Mb::Nothing) -> Nothing
    merge_buffer_mask(Ma::AbstractMatrix{<:Bool}, Mb::AbstractMatrix{<:Bool}) -> AbstractMatrix{<:Bool}

Concatenates the same mask of two buffers, and passes a mask neither records through.

The mask half of [`merge_states`](@ref). It is only ever reached with the two arguments agreeing, because [`assert_buffer_mask_agreement`](@ref) has run.

# Arguments

  - `Ma`: The mask of the first buffer, or `nothing`.
  - `Mb`: The mask of the second buffer, or `nothing`.

# Returns

  - `M`: The two masks stacked, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`merge_states`](@ref)
"""
function merge_buffer_mask(::Nothing, ::Any)
    return nothing
end
function merge_buffer_mask(Ma::AbstractMatrix{<:Bool}, Mb::AbstractMatrix{<:Bool})
    return vcat(Ma, Mb)
end
"""
    trim_merged_mask(M::Nothing, rows) -> Nothing
    trim_merged_mask(M::AbstractMatrix{<:Bool}, rows) -> AbstractMatrix{<:Bool}

Keeps the rows a cap admits of a merged mask, by index copy, and passes a mask that is not there through.

The cap half of [`merge_states`](@ref). It copies rather than views for the reason [`port_opt_view`](@ref) does: the merged buffer is appended to afterwards, and a view would write through into the matrix `vcat` built.

# Arguments

  - `M`: The merged mask, or `nothing`.
  - `rows`: The rows the cap admits.

# Returns

  - `M`: The admitted rows, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`merge_states`](@ref)
"""
function trim_merged_mask(::Nothing, ::Any)
    return nothing
end
function trim_merged_mask(M::AbstractMatrix{<:Bool}, rows)
    return M[rows, :]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`SampleBufferState`](@ref), so the copy shares no array with the original.

The `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref) calls before it folds. Each backing matrix is copied whole, spare capacity included, so the copy appends on the same terms as the original, and the masks stay indexed by the same `off` and `n` as the observations.

# Arguments

  - `x`: The buffer to copy.

# Returns

  - `state::SampleBufferState`: A fresh buffer, equal to `x`, whose backing matrices are fresh matrices.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::SampleBufferState)
    return SampleBufferState(x.n, x.off, copy(x.X), copy_buffer_array(x.A),
                             copy_buffer_array(x.E), x.max_history)
end
"""
    copy_buffer_array(M::Nothing) -> Nothing
    copy_buffer_array(M::AbstractMatrix) -> AbstractMatrix

Copies one backing matrix of a buffer, and passes a matrix that is not there through.

The `nothing` arm of `copy` for the two masks, written once so that [`Base.copy`](@ref) reads as one line per field.

# Arguments

  - `M`: The backing matrix to copy, or `nothing`.

# Returns

  - `M`: A fresh matrix equal to it, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`Base.copy`](@ref)
"""
function copy_buffer_array(::Nothing)
    return nothing
end
function copy_buffer_array(M::AbstractMatrix)
    return copy(M)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Slices a [`SampleBufferState`](@ref) to the selected assets.

A buffer holds its observations verbatim, so the slice of the buffer is the buffer of the sliced universe, column for column, and the observation axis passes through. The masks are per cell, so they slice on the same axis and by the same indices, and the viewed buffer answers the same question over the selected assets that the whole one answers over all of them. The slice copies by index and does not `view`: a later [`partial_fit!`](@ref) on the viewed estimator would otherwise write through into the backing matrices of the estimator the view was taken from.

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
    return SampleBufferState(x.n, x.off, x.X[:, i], slice_buffer_mask(x.A, i),
                             slice_buffer_mask(x.E, i), x.max_history)
end
"""
    slice_buffer_mask(M::Nothing, i) -> Nothing
    slice_buffer_mask(M::AbstractMatrix{<:Bool}, i) -> AbstractMatrix{<:Bool}

Slices one backing mask of a buffer to the selected assets, and passes a mask that is not there through.

The mask arm of [`port_opt_view`](@ref) for this state, split out so that the slice reads as one line per field. It copies by index for the reason that method states.

# Arguments

  - `M`: The backing mask to slice, or `nothing`.
  - `i`: Index or indices of the assets to keep.

# Returns

  - `M`: The mask over the selected assets, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`port_opt_view`](@ref)
"""
function slice_buffer_mask(::Nothing, ::Any)
    return nothing
end
function slice_buffer_mask(M::AbstractMatrix{<:Bool}, i)
    return M[:, i]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds observations into the sample buffer an estimator carries.

The buffering arm of [`partial_fit!`](@ref), and the method every estimator carrying a [`SampleBufferState`](@ref) reaches. A family that folds exactly writes methods of its own, and each of them narrows the `cache` type parameter of its own estimator to the state that fold reads, so a buffer never meets them and this method is what remains. The state's type is therefore the whole route, and nothing refuses the step.

It is one method over both arms of the interface rather than two, because the families that refuse the step declare one method over both arms too, and a pair of narrower methods here would be ambiguous against each of them. So the arm is chosen by the type of `X` inside the body, which is statically resolved at every call site.

A buffer carries the per-observation masks beside the observations, so a [`CoveragePolicy`](@ref) mask threads through the wrapper as it does through an estimator's own accumulator, and the read-out hands it back to the batch verb. A wrapped estimator folded under a policy therefore answers what a batch fit over the same window under the same policy answers, and the unwrapped and wrapped paths agree.

# Algorithm

 1. Read the buffer out of the `cache` field with [`assert_sample_buffer`](@ref), which refuses an estimator that was never wrapped in [`Online`](@ref).
 2. Fold a matrix and its masks through the block arm of [`partial_fit!`](@ref), and a vector and its masks through the single-observation arm.
 3. Rebind `est.cache` with `Accessors.@reset`, and return the estimator.

# Arguments

  - `est`: Estimator whose buffer is folded forward.
  - `X`: Observations to fold. A matrix holds one observation per row when `dims == 1`, and one per column when `dims == 2`. A vector is a single observation across the assets, and `dims` is ignored.
  - $(arg_dict[:dims])
  - `active_mask`: The active mask of the block, of the shape of `X`, or of one entry per asset when `X` is one observation, or `nothing`.
  - `estimation_mask`: The estimation mask, on the same terms as `active_mask`.

# Validation

  - `est` carries a [`SampleBufferState`](@ref). An `ArgumentError` is thrown otherwise.
  - The masks, when they are not `nothing`, have the shape of `X`. A `DimensionMismatch` is thrown otherwise.
  - A buffer holding observations is given the masks it already records. An `ArgumentError` is thrown otherwise.
  - $(val_dict[:dims])

# Returns

  - `est`: The estimator, with its `cache` field rebound to the buffer after the last observation.

# Related

  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`sample_buffer_kwargs`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator},
                      X::VecNum_MatNum; dims::Int = 1, active_mask = nothing,
                      estimation_mask = nothing)
    state = assert_sample_buffer(est)
    state = if isa(X, MatNum)
        partial_fit!(state, X; dims = dims, active_mask = active_mask,
                     estimation_mask = estimation_mask)
    else
        partial_fit!(state, X; active_mask = active_mask, estimation_mask = estimation_mask)
    end
    return Accessors.@reset est.cache = state
end
"""
$(DocStringExtensions.TYPEDEF)

Declares that an estimator takes the online step from a buffer of the observations it has seen.

An `Online` is stored *directly in the estimator field it wraps* — e.g. `EmpiricalPrior(; ce = Online(SomeCovariance()))` — and it is **transient**, in the sense [`TimeDependent`](@ref) is: [`update_online_estimator`](@ref) walks the fields that hold one, seeds each wrapped estimator's `cache` with a [`SampleBufferState`](@ref), and rebuilds the host through its keyword constructor. What comes out is an ordinary estimator carrying a state, and no `Online` exists from that point on, so every verb downstream meets a plain estimator and [ADR 0106](https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/main/docs/adr/0106-a-partial-fit-state-is-the-one-result-an-estimator-holds.md) is untouched.

It differs from [`TimeDependent`](@ref) in when it resolves, and the difference is deliberate. A schedule re-resolves every fold, because its value changes every fold. An `Online` resolves **once**, at warm-up, because after warm-up the state is what is threaded from step to step and a second resolution would throw the buffer away.

A wrapper **replaces** an exact fold; it does not add to one. Every family that folds exactly narrows the `cache` type parameter of its own [`partial_fit!`](@ref) methods to the state that fold reads, so a wrapped estimator never meets them: it buffers its observations and answers every read-out verb by running the batch verb over the rows the buffer holds. The state's type is the whole route, and it is decided by whether the caller wrapped the estimator.

So wrap an estimator whose estimate has **no** exact incremental fold of its own, one that folds exactly and carries its observations for a consumer downstream, or one whose estimate you want fitted over a window rather than over every observation. An estimator that folds exactly and carries nothing needs no wrapper: unwrapped it already answers [`partial_fit!`](@ref), seeding its own state on the first call, and it folds in the memory one state costs rather than the memory a buffer costs.

A wrapper and a [`TimeDependent`](@ref) schedule do not wrap each other, and the difference in *when* they resolve is the whole reason. Neither `Online(TimeDependent(…))` nor a schedule whose entry or `default` is an `Online` is admissible: a wrapper reached through a schedule entry would be resolved at no fold at all, or re-seeded at every fold, throwing away the buffer the step threads. They compose the other way round, and both ways are ordinary. An estimator an `Online` wraps may hold schedules of its own, which survive the seeding untouched and resolve per fold afterwards; and one host may hold a wrapper in one field and a schedule in another, each resolving at its own time. The two field scans are disjoint by construction — a field holding one is invisible to the other's candidate list — so neither resolution can reach the other's wrapper.

`max_history` caps that buffer, and the cap **is** the window. An uncapped buffer answers exactly what a batch fit over every observation folded so far answers, and a capped one answers exactly what a batch fit over the last `max_history` observations answers — for the estimate itself and for every consumer that reads the observations, the scenario risk measures among them, so CVaR, EVaR and CDaR. There is one rule and no special case: a buffer means the batch verb over the buffer's rows. An estimator left unwrapped is unaffected, folds exactly, and stays fitted over every observation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Online(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator};
           max_history::Option{<:Integer} = nothing)
    Online(; est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator},
           max_history::Option{<:Integer} = nothing)

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
