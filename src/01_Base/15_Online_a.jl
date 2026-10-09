"""
$(DocStringExtensions.TYPEDEF)

Carries the observations an estimator keeps when its estimate has no exact incremental fold.

This is the buffer of the [`partial_fit!`](@ref) interface. It is a state like any other. It lives in the `cache` field of the estimator, it copies, it slices by asset, and [`obs_weights_view`](@ref) drops it. It holds the observations verbatim, `NaN` included. So the call with no data over the buffer gives the answer of a batch fit over the same rows, and the two have the same Coverage Universe by construction.

The buffer also holds the per-observation masks verbatim. A [`CoveragePolicy`](@ref) reads two facts from an active mask that the rows alone do not carry. A cell that is finite but inactive is excluded, and an asset that is active at one observation and inactive at the next is a delisting, not a holiday. A buffer that kept the rows and dropped the mask would give a different answer from the batch fit, with no warning. `A` is the active mask, and `M` is the estimation mask that the two regime-adjusted families read. Each is a backing matrix of the shape of `X`, or `nothing`. So a wrapped estimator folded under a policy gives the batch fit over the same window, as it does without a policy.

The buffer holds the factor observations on the same terms. A prior whose batch verb is `prior(pe, X, F)` regresses the asset returns on the factor returns, so its refit reads two matrices whose `t`-th rows are contemporaneous. `F` is the second matrix, a backing matrix of `capacity × factors`, or `nothing`. The same `off` and `n` index `F` and `X`, so the rows stay contemporaneous by construction. [`needs_factor_returns`](@ref) answers whether a fit reads `F`. That is a fact of the estimator tree, and the buffer records what the fold gives it. There is one buffer type, and a cap drops the factor rows with the other rows.

The buffer holds the Panel Fields of a time-varying [`AssetPanel`](@ref) on the same terms, for a prior whose estimator tree reads them. [`reads_panel_fields`](@ref) answers whether a tree reads them. Such a prior computes its factors from the Panel Fields at each row, so for it a Panel Field is sample and not fold context. `P` is the vector of the Panel Fields over the valid region alone. The same `n` and the same cap index `P`, and `off` does not. An append joins the rows with `vcat` of two Asset Panels, so it copies the rows that the buffer holds and costs `O(n)`. A refit reads every row at each step, so this cost is smaller than the cost of its call with no data. A panel is a Result and nothing writes into its arrays, so the buffer holds the Panel Fields by reference, and a copy of the buffer shares them. A time-varying panel carries both masks, so a buffer that records `P` records `A` and `M` too.

The buffer holds the Exogenous Series on the same terms, for a prior whose estimator tree reads it. [`reads_exogenous_series`](@ref) answers whether a tree reads it. An observed factor reads its return from the series by name, so for such a prior the series is sample and not fold context. `E` is a backing matrix of `capacity × series`, and `ne` holds the name of each column. The buffer keeps every column, not only the columns that the tree names, so the Fold Context keeps no copy. The same `off` and `n` index `E` and `X`. So `E` means the Exogenous Series on the whole online step, as it does on a [`ReturnsResult`](@ref).

The first append fixes whether the buffer records a mask, whether it records factor rows, whether it records the Exogenous Series and whether it records Panel Fields, as it fixes the width and the element type. It also fixes the names of the series. A later fold that disagrees is refused with an error that names the disagreement, in both directions. A buffer that holds no observations records nothing about them, so the next fold seeds it again from its block.

An estimator whose `cache` holds this buffer is a refit. It has no recursion of its own, and its call with no data runs the batch verb over the rows of the buffer. [`Online`](@ref) seeds the buffer, and a wrapped estimator takes this route also when its statistic has an exact update. A fold-and-carry prior keeps its observations in one of these buffers too, inside a [`PriorCarryState`](@ref). That prior folds its moments exactly. It keeps the rows because [`LowOrderPrior`](@ref) carries `X` for the scenario risk measures, and its buffer has no cap. `max_history` is the window of a refit. A capped buffer holds the last `max_history` observations, and the call with no data gives the batch fit over them.

The backing matrix has spare capacity, so an append costs amortised `O(1)`. `off` is the number of rows before the valid region, and `n` is the length of the region. [`reserve_sample_buffer`](@ref) moves the region to the front of the backing matrix only when an append would go past its end. [`sample_buffer`](@ref) reads the valid region.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SampleBufferState(;
        n::Integer = 0,
        off::Integer = 0,
        X::MatNum = Matrix{Float64}(undef, 0, 0),
        A::Option{<:AbstractMatrix{<:Bool}} = nothing,
        M::Option{<:AbstractMatrix{<:Bool}} = nothing,
        F::Option{<:MatNum} = nothing,
        ne::Option{<:VecStr} = nothing,
        E::Option{<:MatNum} = nothing,
        P::Option{<:AbstractVector} = nothing,
        max_history::Option{<:Integer} = nothing
    ) -> SampleBufferState

Keywords correspond to the struct's fields. The default is the empty seed that [`Online`](@ref) builds. It carries the cap and no observations. The first append replaces every backing matrix, so the width, the element type, the masks, the factor rows, the Exogenous Series and the Panel Fields come from the observations of that append.

## Validation

  - `n >= 0`. A `DomainError` is thrown otherwise.
  - `off >= 0`. A `DomainError` is thrown otherwise.
  - `off + n <= size(X, 1)`. A `DimensionMismatch` is thrown otherwise.
  - `max_history > 0` when it is not `nothing`. A `DomainError` is thrown otherwise.
  - `n <= max_history` when `max_history` is not `nothing`. A `DimensionMismatch` is thrown otherwise.
  - `A` and `M`, when they are not `nothing`, have the shape of `X`. A `DimensionMismatch` is thrown otherwise.
  - `F`, when it is not `nothing`, has as many rows as `X`. A `DimensionMismatch` is thrown otherwise.
  - The rules of [`assert_buffer_series_shape`](@ref) hold for `ne` and `E`.
  - `P`, when it is not `nothing`, holds time-varying Panel Fields over `n` observations and the assets of `X`. The rules of [`assert_buffer_panel_shape`](@ref) apply.

## View parameters

When [`port_opt_view`](@ref) is called on this type, its fields are subset to the selected assets:

  - `X`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `A`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `M`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `F`: Copied unchanged, because the selection indexes assets and this backing describes factors.
  - `ne`, `E`: Copied unchanged, because the selection indexes assets and the Exogenous Series belongs to no asset.
  - `P`: Each Panel Field viewed over the selected assets with [`panel_fields_view`](@ref).

A buffer that holds no observations views to the empty seed with the same cap.

# Examples

```jldoctest
julia> PortfolioOptimisers.SampleBufferState(; max_history = 2)
PortfolioOptimisers.SampleBufferState
            n ┼ Int64: 0
          off ┼ Int64: 0
            X ┼ 0×0 Matrix{Float64}
            A ┼ nothing
            M ┼ nothing
            F ┼ nothing
           ne ┼ nothing
            E ┼ nothing
            P ┼ nothing
  max_history ┴ Int64: 2
```

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`Online`](@ref)
  - [`PriorCarryState`](@ref)
  - [`sample_buffer`](@ref)
  - [`sample_buffer_kwargs`](@ref)
  - [`factor_buffer`](@ref)
  - [`fold_buffer`](@ref)
  - [`partial_fit!`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`needs_factor_returns`](@ref)
  - [`reads_panel_fields`](@ref)
  - [`reads_exogenous_series`](@ref)
  - [`sample_buffer_panel`](@ref)
  - [`exogenous_buffer_kwargs`](@ref)
"""
@concrete struct SampleBufferState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    Number of rows of `X` before the valid region. It rises when a capped buffer drops its oldest observation, and it returns to zero when the valid region moves to the front of `X`.
    """
    off
    """
    Backing matrix of the buffer, `capacity × assets`. Rows `off + 1` to `off + n` are the observations, in the order of the folds. The other rows are spare capacity, and their values have no meaning. The first append fixes the width and the element type.
    """
    X
    """
    $(field_dict[:pf_buffer_A])
    """
    A
    """
    $(field_dict[:pf_buffer_M])
    """
    M
    """
    Backing matrix of the factor observations, `capacity × factors`, or `nothing` when the fold carries none. The same `off` and `n` index it and `X`. The first append that carries it fixes its width and its element type.
    """
    F
    """
    Names of the columns of `E`, or `nothing` when the fold carries no Exogenous Series. The first append that carries the series fixes them.
    """
    ne
    """
    Backing matrix of the Exogenous Series, `capacity × series`, or `nothing` when the fold carries none. The same `off` and `n` index it and `X`. It holds every column of the series that the fold receives.
    """
    E
    """
    The Panel Fields of the observations, or `nothing` when the fold carries none. Each is a time-varying Panel Field over the valid region alone, so `n` and the cap index it and `off` does not. The first append that carries them fixes their names and their kinds.
    """
    P
    """
    $(field_dict[:pf_max_history])
    """
    max_history
end
function SampleBufferState(; n::Integer = 0, off::Integer = 0,
                           X::MatNum = Matrix{Float64}(undef, 0, 0),
                           A::Option{<:AbstractMatrix{<:Bool}} = nothing,
                           M::Option{<:AbstractMatrix{<:Bool}} = nothing,
                           F::Option{<:MatNum} = nothing, ne::Option{<:VecStr} = nothing,
                           E::Option{<:MatNum} = nothing,
                           P::Option{<:AbstractVector} = nothing,
                           max_history::Option{<:Integer} = nothing)::SampleBufferState
    assert_sample_buffer_state(n, off, X, max_history)
    assert_buffer_mask_shape(X, A, M)
    assert_buffer_factor_shape(X, F)
    assert_buffer_series_shape(X, ne, E)
    assert_buffer_panel_shape(X, n, P)
    return SampleBufferState(n, off, X, A, M, F, ne, E, P, max_history)
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

A mask that a buffer holds matches its observations cell for cell, and a mask that a fold receives matches its block cell for cell. So one rule covers both. A mask is absent, or it has the shape of the matrix beside it. The batch verbs run the same check on their own arguments.

# Algorithm

 1. Skip a mask that is `nothing`. With no policy, both masks are `nothing`.
 2. Refuse a mask whose size is not `size(X)`, and name the mask.

# Arguments

  - `X`: The observations the masks belong to.
  - `A`: The active mask, or `nothing`.
  - `M`: The estimation mask, or `nothing`.

# Validation

  - `A` and `M`, when they are not `nothing`, have the shape of `X`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function assert_buffer_mask_shape(X::AbstractMatrix, A::Option{<:AbstractMatrix{<:Bool}},
                                  M::Option{<:AbstractMatrix{<:Bool}})
    if !isnothing(A)
        @argcheck(size(A) == size(X),
                  DimensionMismatch("size(X) ($(size(X))) must match size(active_mask) ($(size(A)))"))
    end
    if !isnothing(M)
        @argcheck(size(M) == size(X),
                  DimensionMismatch("size(X) ($(size(X))) must match size(estimation_mask) ($(size(M)))"))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a factor block whose rows are not the rows of the observations it belongs to.

A factor observation is contemporaneous with the asset observation of the same row. So a factor block that a fold receives has one row for each row of the block beside it, and the factor backing of a buffer has one row for each row of `X`. The check does not compare the two widths, because an asset count and a factor count are unrelated.

# Algorithm

 1. Skip a factor block that is `nothing`. With no factors, the block is `nothing`.
 2. Refuse a factor block whose row count is not `size(X, 1)`.

# Arguments

  - `X`: The observations the factor rows belong to.
  - `F`: The factor rows, or `nothing`.

# Validation

  - `F`, when it is not `nothing`, has as many rows as `X`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`needs_factor_returns`](@ref)
"""
function assert_buffer_factor_shape(X::AbstractMatrix, F::Option{<:AbstractMatrix})
    if !isnothing(F)
        @argcheck(size(F, 1) == size(X, 1),
                  DimensionMismatch("size(X, 1) ($(size(X, 1))) must match size(F, 1) ($(size(F, 1))): a factor observation is contemporaneous with the asset observation of the same row."))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses an Exogenous Series whose rows are not the rows of the observations it belongs to, or whose names do not name its columns.

A row of the series is contemporaneous with the asset observation of the same row, as a factor row is. The names travel with the series, so the two are both absent or both present, and there is one name for each column.

# Algorithm

 1. Refuse names without a series, and a series without names.
 2. Skip a series that is `nothing`.
 3. Refuse a series whose row count is not `size(X, 1)`, and names whose count is not the column count of the series.

# Arguments

  - `X`: The observations the series belongs to.
  - `ne`: The names of the series, or `nothing`.
  - `E`: The series, `observations × series`, or `nothing`.

# Validation

  - `ne` and `E` are both `nothing`, or neither is. An `ArgumentError` is thrown otherwise.
  - `E`, when it is not `nothing`, has as many rows as `X`, and `length(ne) == size(E, 2)`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`reads_exogenous_series`](@ref)
"""
function assert_buffer_series_shape(X::AbstractMatrix, ne::Option{<:VecStr},
                                    E::Option{<:AbstractMatrix})
    @argcheck(isnothing(ne) == isnothing(E),
              ArgumentError("the names of the Exogenous Series travel with the series, so `ne` and `E` are both `nothing` or neither is, but `ne` is $(isnothing(ne) ? "" : "not ")`nothing` and `E` is $(isnothing(E) ? "" : "not ")`nothing`."))
    if !isnothing(E)
        @argcheck(size(E, 1) == size(X, 1),
                  DimensionMismatch("size(X, 1) ($(size(X, 1))) must match size(E, 1) ($(size(E, 1))): a row of the Exogenous Series is contemporaneous with the asset observation of the same row."))
        @argcheck(length(ne) == size(E, 2),
                  DimensionMismatch("`ne` names each column of the Exogenous Series, but it holds $(length(ne)) name(s) and `E` has $(size(E, 2)) column(s)."))
    end
    return nothing
end
"""
    buffer_rows_view(M::Nothing, rows) -> Nothing
    buffer_rows_view(M::AbstractMatrix, rows) -> SubArray

Reads a row range out of a per-observation backing, and passes a backing that is not there through.

The buffer slices a mask or the factor rows on the observation axis only through this function, so the `nothing` case has one method. [`sample_buffer_kwargs`](@ref), [`factor_buffer`](@ref) and [`merge_states`](@ref) read the valid region of a stored backing with it, and [`partial_fit!`](@ref) truncates an incoming block with it.

# Arguments

  - `M`: The backing to slice, or `nothing`.
  - `rows`: The rows to keep.

# Returns

  - `M`: The rows of the backing, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`sample_buffer_kwargs`](@ref)
  - [`factor_buffer`](@ref)
  - [`partial_fit!`](@ref)
"""
function buffer_rows_view(::Nothing, args...)
    return nothing
end
function buffer_rows_view(M::AbstractMatrix, rows)
    return view(M, rows, :)
end
"""
    sample_buffer(state::SampleBufferState)
    sample_buffer(est)

Reads the observations a sample buffer holds, `observations × assets`.

This function reads the rows out of a [`SampleBufferState`](@ref). It is a view of the valid region of the backing matrix, in the order of the folds, so a refit reads it with no copy. The estimator form reads the state from the `cache` field, and it refuses an estimator that carries no buffer.

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

[`sample_buffer`](@ref) gives the rows, and this function gives the masks of those rows. The call with no data runs the batch verb on the rows and splats these keywords into the call, so the wrapped answer is the unwrapped answer over the same window. A buffer that records no mask gives an empty set of keywords, and the arm then makes the same call as with no policy. The branch reads fields of concrete type, so the compiler resolves it.

# Algorithm

 1. Slice each stored mask to the valid region with [`buffer_rows_view`](@ref).
 2. Name the masks that are there, and omit the ones that are not.

# Arguments

  - `state`: The buffer to read.

# Returns

  - `kwargs::NamedTuple`: The `active_mask` and `estimation_mask` keywords the buffer holds, of which either or both may be absent.

# Related

  - [`SampleBufferState`](@ref)
  - [`sample_buffer`](@ref)
  - [`buffer_rows_view`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function sample_buffer_kwargs(state::SampleBufferState)
    rows = (state.off + 1):(state.off + state.n)
    A = buffer_rows_view(state.A, rows)
    M = buffer_rows_view(state.M, rows)
    return if isnothing(A)
        isnothing(M) ? (;) : (; estimation_mask = M)
    elseif isnothing(M)
        (; active_mask = A)
    else
        (; active_mask = A, estimation_mask = M)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the factor observations a sample buffer holds, `observations × factors`, or `nothing` when it holds none.

This is the third function that reads the buffer out, with [`sample_buffer`](@ref) and [`sample_buffer_kwargs`](@ref). A refit whose batch verb reads a factor matrix takes it from here, as the second positional argument of that verb. Its `t`-th row is contemporaneous with the `t`-th row of [`sample_buffer`](@ref), because the same `off` and `n` index the two backings. A buffer that records no factor rows gives `nothing`, which is the value that the batch verb of a prior with no factors receives. The branch reads a field of concrete type, so the compiler resolves it.

# Arguments

  - `state`: The buffer to read.

# Returns

  - `F`: The factor observations the buffer holds, as a view of the valid region, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`sample_buffer`](@ref)
  - [`sample_buffer_kwargs`](@ref)
  - [`buffer_rows_view`](@ref)
  - [`needs_factor_returns`](@ref)
"""
function factor_buffer(state::SampleBufferState)
    return buffer_rows_view(state.F, (state.off + 1):(state.off + state.n))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the Exogenous Series a sample buffer holds, as the keywords a batch verb takes.

This is the fourth function that reads the buffer out. The call with no data of a refit splats these keywords into the batch verb, as `prior(pe, rd)` passes `rd.ne` and `rd.E`. The Fold Context reads them too, when the prior owns the series. Each row of `E` is contemporaneous with the row of [`sample_buffer`](@ref) of the same index, because the same `off` and `n` index the two backings. A buffer that records no series gives an empty set of keywords. The branch reads a field of concrete type, so the compiler resolves it.

# Arguments

  - `state`: The buffer to read.

# Returns

  - `kwargs::NamedTuple`: `ne` and `E`, the names and a view of the valid region of the series, or an empty `NamedTuple`.

# Related

  - [`SampleBufferState`](@ref)
  - [`sample_buffer`](@ref)
  - [`factor_buffer`](@ref)
  - [`reads_exogenous_series`](@ref)
"""
function exogenous_buffer_kwargs(state::SampleBufferState)
    return if isnothing(state.E)
        (;)
    else
        (; ne = state.ne,
         E = buffer_rows_view(state.E, (state.off + 1):(state.off + state.n)))
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the sample buffer an estimator carries, and refuses an estimator that carries none.

The buffering arm of [`partial_fit!`](@ref) reads its state through this function. So an estimator that was never wrapped in [`Online`](@ref) gets a message that names the wrapper, not a `MethodError`. An estimator with no `cache` field gets the same message. An unresolved [`Online`](@ref) has a method of its own, because it does not lack a buffer. It lacks the warm-up that seeds one.

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

The seed has its own function, so the fold is one line and the branch on the `cache` field has one method. A family that carries its observations calls [`fold_buffer`](@ref), which is this seed followed by the fold. The seed has no cap, because a cap is configuration, and [`Online`](@ref) carries it.

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
    fold_buffer(cache::Option{<:SampleBufferState}, x::VecNum, f::Option{<:VecNum} = nothing; kwargs...)
    fold_buffer(cache::Option{<:SampleBufferState}, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Folds observations into a sample buffer, seeding an empty one when the estimator carries none.

A carry calls this function inside its own [`partial_fit!`](@ref) method. The carry folds its estimate exactly through its members, and it gives its observations to this function. The function is [`sample_buffer_seed`](@ref) followed by [`partial_fit!`](@ref). The carry gives the masks and the factor rows of the fold with the observations, and the buffer records them beside their rows.

# Arguments

  - `cache`: The buffer the estimator carries, or `nothing`.
  - `x`: One observation, whose entries are the assets.
  - `f`: The contemporaneous factor observation, whose entries are the factors, or `nothing`.
  - $(arg_dict[:X])
  - `F`: Factor observations to fold, oriented as `X` is, or `nothing`.
  - $(arg_dict[:dims])
  - `kwargs...`: The `active_mask`, the `estimation_mask` and the Exogenous Series `ne` and `E` of the fold, forwarded to [`partial_fit!`](@ref).

# Returns

  - `state::SampleBufferState`: The buffer after the last observation.

# Related

  - [`SampleBufferState`](@ref)
  - [`sample_buffer_seed`](@ref)
  - [`partial_fit!`](@ref)
"""
function fold_buffer(cache::Option{<:SampleBufferState}, x::VecNum,
                     f::Option{<:VecNum} = nothing; kwargs...)
    return partial_fit!(sample_buffer_seed(cache), x, f; kwargs...)
end
function fold_buffer(cache::Option{<:SampleBufferState}, X::MatNum,
                     F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    return partial_fit!(sample_buffer_seed(cache), X, F; dims = dims, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds every observation of a block into a [`SampleBufferState`](@ref).

This is the block arm of the [`partial_fit!`](@ref) interface, and all of the fold of the buffer. The single-observation arm reshapes its argument and calls this arm. A cap truncates the incoming rows before the copy, so the fold never copies the whole of a block longer than the cap. [`reserve_sample_buffer`](@ref) grows the backing matrices, and its docstring states why an append costs amortised `O(1)`.

The masks, the factor rows, the Exogenous Series and the Panel Fields stay with their rows through each step: the orientation, the truncation, the growth and the copy. The first append fixes whether the buffer records a mask, whether it records factor rows, whether it records the Exogenous Series and whether it records Panel Fields, as it fixes the width and the element type. It also fixes the names of the series. [`assert_buffer_presence_agreement`](@ref) refuses a later fold that disagrees. A buffer that knows the activity of some of its rows and not of others answers neither question. A buffer with factor rows for some observations and not for others describes no regression. A buffer that holds no observations has nothing to disagree with, so the fold seeds it again from the block.

The Panel Fields are observations first whatever `dims` is, as the Panel Fields of an [`AssetPanel`](@ref) always are. They keep the rows of the buffer that the cap keeps, and [`panel_fields_append`](@ref) joins them to the rows of the block.

# Algorithm

 1. Orient `X`, the masks, `F` and `E` to `observations × assets`, `observations × factors` and `observations × series`. Transpose them when `dims == 2`.
 2. Refuse a mask that does not have the shape of the block, a factor block whose row count is not the row count of the block, and a series that [`assert_buffer_series_shape`](@ref) refuses.
 3. Return the state unchanged when the block is empty.
 4. Refuse Panel Fields that do not make a time-varying Asset Panel with the masks of the block, over the observations and the assets of the block.
 5. When a cap is set, truncate the incoming rows, their masks, their factor rows, their series and their Panel Fields to the last `max_history` of them.
 6. When the buffer holds no observations, seed it from the block with [`seed_sample_buffer`](@ref) and return it. The seed fixes the width, the element type, the masks, the factor rows, the series with its names, and the Panel Fields.
 7. Refuse a block whose width is not the fixed width. Refuse masks, factor rows, a series or Panel Fields that are absent where the buffer records them or present where it does not, factor rows of a different width, and a series under other names with [`assert_buffer_series_names`](@ref).
 8. Make room for the block with [`reserve_sample_buffer`](@ref), which drops the rows that the cap pushes out and grows the backing matrices.
 9. Copy the block, its masks, its factor rows and its series after the valid region. Join the Panel Fields that the buffer keeps to the Panel Fields of the block with [`panel_fields_append`](@ref), and rebind `P` and `n` with `Accessors.@reset`.

# Arguments

  - `state`: The buffer to fold into.
  - $(arg_dict[:X])
  - `F`: Factor observations to fold, oriented as `X` is, or `nothing`.
  - $(arg_dict[:dims])
  - $(arg_dict[:pf_active_mask])
  - $(arg_dict[:pf_estimation_mask])
  - `ne`: The names of the Exogenous Series of the block, or `nothing`.
  - `E`: The Exogenous Series of the block, oriented as `X` is, or `nothing`.
  - `panel_fields`: The time-varying Panel Fields of the block, observations first, or `nothing`.

# Validation

  - $(val_dict[:dims])
  - `X` has the width the first append fixed. A `DimensionMismatch` is thrown otherwise.
  - `active_mask` and `estimation_mask`, when they are not `nothing`, have the shape of `X`. A `DimensionMismatch` is thrown otherwise.
  - `F`, when it is not `nothing`, has as many rows as `X`, and the width that the first append with factor rows fixed. A `DimensionMismatch` is thrown otherwise.
  - `ne` and `E` obey the rules of [`assert_buffer_series_shape`](@ref), and `ne` equals the names that the first append with a series fixed. An `ArgumentError` is thrown otherwise.
  - `panel_fields`, when it is not `nothing`, obeys the rules of [`assert_buffer_panel_block`](@ref), and it holds the Panel Fields that the first append with Panel Fields fixed, as `vcat` of two Asset Panels states them.
  - A buffer that holds observations receives the masks that it records, and it receives factor rows, a series and Panel Fields if and only if it records them. An `ArgumentError` is thrown otherwise.

# Returns

  - `state::SampleBufferState`: The buffer after the last row.

# Related

  - [`SampleBufferState`](@ref)
  - [`reserve_sample_buffer`](@ref)
  - [`seed_sample_buffer`](@ref)
  - [`assert_buffer_presence_agreement`](@ref)
  - [`partial_fit!`](@ref)
  - [`sample_buffer`](@ref)
  - [`factor_buffer`](@ref)
"""
function partial_fit!(state::SampleBufferState, X::MatNum, F::Option{<:MatNum} = nothing;
                      dims::Int = 1,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                      estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                      ne::Option{<:VecStr} = nothing, E::Option{<:MatNum} = nothing,
                      panel_fields::Option{<:AbstractVector} = nothing)
    Xo, Ao, Mo, Fo, Eo = dims_oriented(dims, X, active_mask, estimation_mask, F, E)
    assert_buffer_mask_shape(Xo, Ao, Mo)
    assert_buffer_factor_shape(Xo, Fo)
    assert_buffer_series_shape(Xo, ne, Eo)
    t = size(Xo, 1)
    if iszero(t)
        return state
    end
    assert_buffer_panel_block(Xo, Ao, Mo, panel_fields)
    Po = panel_fields
    w = state.max_history
    if !isnothing(w) && t > w
        rows = (t - w + 1):t
        Xo = view(Xo, rows, :)
        Ao = buffer_rows_view(Ao, rows)
        Mo = buffer_rows_view(Mo, rows)
        Fo = buffer_rows_view(Fo, rows)
        Eo = buffer_rows_view(Eo, rows)
        Po = panel_fields_view(Po, rows, :)
        t = w
    end
    if iszero(state.n)
        return seed_sample_buffer(state, Xo, Ao, Mo, Fo, ne, Eo, Po)
    end
    N = size(Xo, 2)
    B = state.X
    @argcheck(size(B, 2) == N,
              DimensionMismatch("the width of a sample buffer is fixed by its first append, but the buffer describes $(size(B, 2)) assets and the block has $N columns."))
    assert_buffer_presence_agreement(state.A, Ao, "active_mask")
    assert_buffer_presence_agreement(state.M, Mo, "estimation_mask")
    assert_buffer_presence_agreement(state.F, Fo, "factor returns")
    assert_buffer_presence_agreement(state.E, Eo, "Exogenous Series")
    assert_buffer_presence_agreement(state.P, Po, "Panel Fields")
    assert_buffer_factor_width(state.F, Fo)
    assert_buffer_series_names(state.ne, ne)
    state = reserve_sample_buffer(state, t)
    rows = (state.off + state.n + 1):(state.off + state.n + t)
    copyto!(view(state.X, rows, :), Xo)
    copy_buffer_rows!(state.A, rows, Ao)
    copy_buffer_rows!(state.M, rows, Mo)
    copy_buffer_rows!(state.F, rows, Fo)
    copy_buffer_rows!(state.E, rows, Eo)
    state = Accessors.@reset state.P = panel_fields_append(state.P, Po, state.n + t)
    return Accessors.@reset state.n = state.n + t
end
"""
    assert_buffer_factor_width(B::Nothing, Fo) -> Nothing
    assert_buffer_factor_width(B::AbstractMatrix, Fo::AbstractMatrix) -> Nothing

Refuses a factor block whose width is not the width the first factor append fixed.

This is the width check of `X`, for the factor backing, which can be absent. A buffer that records no factor rows has no width to compare with. [`assert_buffer_presence_agreement`](@ref) runs first and refuses a block that gives factor rows to such a buffer, so the `nothing` method runs only when `Fo` is absent too.

# Arguments

  - `B`: The factor backing the buffer holds, or `nothing`.
  - `Fo`: The oriented factor block, or `nothing`.

# Validation

  - `size(B, 2) == size(Fo, 2)` when the buffer records factor rows. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`assert_buffer_presence_agreement`](@ref)
"""
function assert_buffer_factor_width(::Nothing, ::Any)
    return nothing
end
function assert_buffer_factor_width(B::AbstractMatrix, Fo::AbstractMatrix)
    @argcheck(size(B, 2) == size(Fo, 2),
              DimensionMismatch("the width of a sample buffer's factor rows is fixed by the first append that carries them, but the buffer describes $(size(B, 2)) factors and the block has $(size(Fo, 2)) columns."))
    return nothing
end
"""
    assert_buffer_series_names(ne::Nothing, neo) -> Nothing
    assert_buffer_series_names(ne::VecStr, neo::VecStr) -> Nothing

Refuses an Exogenous Series whose names are not the names the first append fixed.

A consumer reads a column of the series by its name, so a buffer whose columns change their names between two rows gives a column with two meanings. The first append that carries the series fixes the names, as [`ReturnsBufferState`](@ref) pins them. [`assert_buffer_presence_agreement`](@ref) runs first, so the `nothing` method runs only when the block carries no series either.

# Arguments

  - `ne`: The names the buffer records, or `nothing`.
  - `neo`: The names of the incoming block, or `nothing`.

# Validation

  - `neo == ne` when the buffer records a series. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`merge_states`](@ref)
"""
function assert_buffer_series_names(::Nothing, ::Any)
    return nothing
end
function assert_buffer_series_names(ne::VecStr, neo::VecStr)
    @argcheck(ne == neo,
              ArgumentError("the first append fixes the names of the Exogenous Series of a sample buffer, and a consumer reads a column by its name, so a later block must carry the same names. The buffer records $(ne) and the block carries $(neo). Fold the whole run with one set of names, or seed a fresh buffer."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a fold whose masks, factor rows, Exogenous Series or Panel Fields disagree with what the buffer records.

This is the mixture rule of the buffer. It refuses the fold, because neither way to resolve the mixture gives a correct answer. A buffer that dropped an incoming mask would fit a held gap that the policy excludes, and it would read each delisting as a holiday. The mask exists to prevent that wrong answer. A buffer that made up a mask for its earlier rows would state an activity that it never saw. The same rule holds for the factor rows. A buffer with factor rows for some observations and not for others describes no regression, and the buffer cannot make up the missing rows. A prior that reads the Panel Fields computes its factors from them at every row, so a buffer that holds them for some rows and not for others gives it no complete panel. The same holds for the Exogenous Series that an observed factor reads at every fitted row. So the first append fixes what a buffer records, and a caller that wants a change starts a new buffer.

# Arguments

  - `M`: The backing the buffer records, or `nothing`.
  - `Mo`: The same backing of the incoming block, or `nothing`.
  - `name`: Name of the backing, for the message.

# Validation

  - `M` and `Mo` are both `nothing`, or neither is. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`seed_sample_buffer`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`needs_factor_returns`](@ref)
"""
function assert_buffer_presence_agreement(M::Option{<:AbstractArray},
                                          Mo::Option{<:AbstractArray}, name::AbstractString)
    @argcheck(isnothing(M) == isnothing(Mo),
              ArgumentError("a sample buffer records `$name` for every observation it holds or for none of them, and this fold $(isnothing(Mo) ? "gives none where the buffer records one" : "gives one where the buffer records none"). A buffer that records it for some of its rows and not for others answers neither the question it asks nor the question it does not. Fold the whole run with `$name` or fold it without, or seed a fresh buffer."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fills a buffer that holds no observations from a block, which fixes the width, the element type, the masks, the factor rows, the Exogenous Series and the Panel Fields.

[`partial_fit!`](@ref) calls it for the first append, and for each append to a buffer that holds no observations. Each backing matrix is an exact fit for the block, so a buffer that is folded once and read once allocates no spare row. Each array comes from `similar` on the argument that it holds, so the function chooses no element type. The Panel Fields are not copied, because nothing writes into the arrays of a panel.

# Algorithm

 1. Replace `X`, `A`, `M`, `F` and `E` with copies of the block, its masks, its factor rows and its series, made by [`seed_buffer_array`](@ref).
 2. Replace `ne` with the names of the series, and `P` with the Panel Fields of the block.
 3. Set `off` to zero and `n` to the row count of the block.

# Arguments

  - `state`: The buffer to fill. It holds no observations.
  - `Xo`: The oriented block.
  - `Ao`: The oriented active mask of the block, or `nothing`.
  - `Mo`: The oriented estimation mask of the block, or `nothing`.
  - `Fo`: The oriented factor rows of the block, or `nothing`.
  - `ne`: The names of the series of the block, or `nothing`.
  - `Eo`: The oriented Exogenous Series of the block, or `nothing`.
  - `Po`: The Panel Fields of the block, or `nothing`.

# Returns

  - `state::SampleBufferState`: The buffer holding the block, its masks, its factor rows, its series and its Panel Fields.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`seed_buffer_array`](@ref)
"""
function seed_sample_buffer(state::SampleBufferState, Xo::MatNum,
                            Ao::Option{<:AbstractMatrix{<:Bool}},
                            Mo::Option{<:AbstractMatrix{<:Bool}}, Fo::Option{<:MatNum},
                            ne::Option{<:VecStr}, Eo::Option{<:MatNum},
                            Po::Option{<:AbstractVector})
    state = Accessors.@reset state.X = seed_buffer_array(Xo)
    state = Accessors.@reset state.A = seed_buffer_array(Ao)
    state = Accessors.@reset state.M = seed_buffer_array(Mo)
    state = Accessors.@reset state.F = seed_buffer_array(Fo)
    state = Accessors.@reset state.ne = ne
    state = Accessors.@reset state.E = seed_buffer_array(Eo)
    state = Accessors.@reset state.P = Po
    state = Accessors.@reset state.off = 0
    return Accessors.@reset state.n = size(Xo, 1)
end
"""
    seed_buffer_array(Mo::Nothing) -> Nothing
    seed_buffer_array(Mo::AbstractMatrix) -> AbstractMatrix

Copies an incoming block into a backing matrix of its own, and passes a backing that is not there through.

The block of a fold is an array of the caller, and the buffer lives after the call. A later fold writes into the backing matrix, so the buffer takes a copy, as [`Base.copy`](@ref) and [`port_opt_view`](@ref) do. Without the copy, the fold would change the array of the caller. `similar` reads the element type from the block, so an observation, a mask and a factor row each keep their own element type.

# Arguments

  - `Mo`: The oriented block, mask or factor rows, or `nothing`.

# Returns

  - `M`: A backing matrix holding the block, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`seed_sample_buffer`](@ref)
"""
function seed_buffer_array(::Nothing)
    return nothing
end
function seed_buffer_array(Mo::AbstractMatrix)
    return copyto!(similar(Mo, size(Mo)...), Mo)
end
"""
    copy_buffer_rows!(M::Nothing, rows, Mo) -> Nothing
    copy_buffer_rows!(M::AbstractMatrix, rows, Mo::AbstractMatrix) -> AbstractMatrix

Copies an incoming backing into the rows of the buffer's backing matrix that were just made room for.

This is the optional part of the append. The two masks and the factor rows each take one line of the fold, and the `nothing` case has one method. [`assert_buffer_presence_agreement`](@ref) runs first, so the two arguments are both `nothing` or both matrices.

# Arguments

  - `M`: The buffer's backing matrix for this mask or the factor rows, or `nothing`.
  - `rows`: The rows of `M` to write.
  - `Mo`: The oriented mask or factor rows of the block, or `nothing`.

# Returns

  - `M`: The rows written, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`assert_buffer_presence_agreement`](@ref)
"""
function copy_buffer_rows!(::Nothing, args...)
    return nothing
end
function copy_buffer_rows!(M::AbstractMatrix, rows, Mo::AbstractMatrix)
    return copyto!(view(M, rows, :), Mo)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes room for `t` further observations in a [`SampleBufferState`](@ref), and returns the buffer that has it.

This is the growth part of the fold of the buffer. It writes no observation. It drops the rows that the cap pushes out, moves the valid region to the front of the backing matrices, and allocates larger matrices when the front does not have enough room.

These steps make an append cost amortised `O(1)`. A drop moves `off` and copies no row, so a capped buffer pays nothing for the observation that it drops. The masks, the factor rows and the series of that observation drop with it, because the same `off` and `n` index them. A compaction in place happens only when the valid region and the block fill at most half of the backing matrix. So it copies at most half of the matrix, and at least half of the matrix is free after the append. A growth at least doubles the capacity until the clamp at twice the cap stops it, and it copies fewer rows than the old capacity. So all the growths of a buffer copy fewer rows than its last capacity. A block longer than twice the capacity grows the matrix to fit the valid region and the block, and the next growth doubles it. A capped buffer never holds more than twice its cap, so a cap of `w` costs at most `2w` rows.

# Algorithm

 1. Drop the oldest observations the cap pushes out, by moving `off` forward and `n` back.
 2. Return when the valid region and the block already fit inside the backing matrix.
 3. Take a capacity of twice the current one, or of the room needed if that is larger, when the valid region and the block would fill more than half of the current one. Clamp it to twice the cap when a cap is set.
 4. Compact the observations, each mask, the factor rows and the series the buffer records to that capacity with [`compact_buffer_array`](@ref), which allocates only when the capacity changes, and set `off` to zero.

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
    state = Accessors.@reset state.M = compact_buffer_array(state.M, off, n, newcap)
    state = Accessors.@reset state.F = compact_buffer_array(state.F, off, n, newcap)
    state = Accessors.@reset state.E = compact_buffer_array(state.E, off, n, newcap)
    return Accessors.@reset state.off = 0
end
"""
    compact_buffer_array(B::Nothing, off, n, newcap) -> Nothing
    compact_buffer_array(B::AbstractMatrix, off::Integer, n::Integer, newcap::Integer) -> AbstractMatrix

Moves the valid region of one backing matrix to the front, allocating a matrix of `newcap` rows when the capacity changes.

This is the array part of [`reserve_sample_buffer`](@ref). The buffer carries five backing matrices with one index, and this function compacts each of them. A copy in place with an increasing row index is safe, because the destination row `i` is never after the source row `off + i`. So the loop reads each source row before a write can reach it.

# Algorithm

 1. Take `D` as `B` when `B` has `newcap` rows, and as a new matrix of `newcap` rows otherwise.
 2. Copy row `off + i` of `B` to row `i` of `D`, for `i` from 1 to `n`, one column at a time.

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

This is the single-observation arm of the [`partial_fit!`](@ref) interface. It reshapes the observation into a one-row block with no copy, and it folds the block through the block arm. It reshapes the masks, the factor observation and the observation of the Exogenous Series in the same way, so one observation with a policy, with factors or with a series folds through the code of the block.

# Arguments

  - `state`: The buffer to fold into.
  - `x`: One observation, whose entries are the assets.
  - `f`: The contemporaneous factor observation, whose entries are the factors, or `nothing`.
  - `active_mask`: The active mask of the observation, one entry per asset, or `nothing`.
  - `estimation_mask`: The estimation mask of the observation, one entry per asset, or `nothing`.
  - `ne`: The names of the Exogenous Series, or `nothing`.
  - `E`: The observation of the Exogenous Series, one entry per series, or `nothing`.

# Validation

  - `x` has the width the first append fixed. A `DimensionMismatch` is thrown otherwise.
  - `active_mask` and `estimation_mask`, when they are not `nothing`, have one entry per asset. A `DimensionMismatch` is thrown otherwise.
  - `f`, when it is not `nothing`, has the width that the first append with factor rows fixed. A `DimensionMismatch` is thrown otherwise.
  - A buffer that holds observations receives `f` if and only if it records factor rows. An `ArgumentError` is thrown otherwise.

# Returns

  - `state::SampleBufferState`: The buffer after the observation.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(state::SampleBufferState, x::VecNum, f::Option{<:VecNum} = nothing;
                      active_mask::Option{<:AbstractVector{<:Bool}} = nothing,
                      estimation_mask::Option{<:AbstractVector{<:Bool}} = nothing,
                      ne::Option{<:VecStr} = nothing, E::Option{<:VecNum} = nothing)
    return partial_fit!(state, observation_row(x), observation_row(f);
                        active_mask = observation_row(active_mask),
                        estimation_mask = observation_row(estimation_mask), ne = ne,
                        E = observation_row(E))
end
"""
    observation_row(m::Nothing) -> Nothing
    observation_row(m::AbstractVector) -> AbstractMatrix

Reshapes one observation, its mask or its factor observation into a one-row block, and passes one that is not there through.

This is the vector part of the single-observation arm of [`partial_fit!`](@ref). `reshape` makes no copy, so one observation with a policy or with factors allocates no more than one observation without them.

# Arguments

  - `m`: One observation, one mask or one factor observation, or `nothing`.

# Returns

  - `M`: The vector as a one-row block, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
"""
function observation_row(::Nothing)
    return nothing
end
function observation_row(m::AbstractVector)
    return reshape(m, 1, length(m))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds two [`SampleBufferState`](@ref) fitted on disjoint blocks into the buffer of the concatenated block.

The merge is a concatenation, and it is exact. A buffer holds its observations verbatim, so the buffer of two blocks holds the rows of the first block followed by the rows of the second. When a cap is set, the result keeps the last `max_history` rows of the concatenation. The masks, the factor rows, the Exogenous Series and the Panel Fields concatenate with their rows. Two buffers that disagree about what they record, or that name their series differently, are refused, for the reason that [`assert_buffer_presence_agreement`](@ref) states for a fold. A buffer that holds no observations records nothing, so the merge returns a copy of the other buffer.

The method does not call [`assert_mergeable_states`](@ref). That check compares every array field on every axis. The backing matrix of a buffer has the observation axis as well as the asset axis, so the check refuses two buffers over the same assets with different numbers of observations. This method checks the cap, the width, the masks and the factor rows itself.

# Algorithm

 1. Refuse two buffers of different caps.
 2. Return a copy of the other buffer when one buffer holds no observations.
 3. Refuse two buffers of different widths.
 4. Refuse two buffers that do not record the same masks, that disagree about factor rows, a series or Panel Fields, whose factor rows have different widths, or whose series have different names.
 5. Concatenate the valid region of the first with the valid region of the second, and each mask, the factor rows and the series with their own.
 6. Keep the last `max_history` rows when a cap is set.
 7. Join the Panel Fields of the two buffers over the rows that the result keeps, with [`panel_fields_append`](@ref).

# Arguments

  - `a`: The buffer of the first block of observations.
  - `b`: The buffer of the second block of observations.

# Validation

  - `a` and `b` carry the same cap. An `ArgumentError` is thrown otherwise.
  - `a` and `b` describe the same number of assets, when both hold observations. A `DimensionMismatch` is thrown otherwise.
  - `a` and `b` record the same masks, both record factor rows or neither does, both record a series under the same names or neither does, and both record Panel Fields or neither does, when both hold observations. An `ArgumentError` is thrown otherwise.
  - `a` and `b` describe the same number of factors, when they record factor rows. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `state::SampleBufferState`: The buffer the two blocks give when they are folded as one block.

# Related

  - [`SampleBufferState`](@ref)
  - [`merge_states`](@ref)
  - [`assert_buffer_presence_agreement`](@ref)
  - [`sample_buffer`](@ref)
"""
function merge_states(a::SampleBufferState, b::SampleBufferState)
    w = a.max_history
    @argcheck(w == b.max_history,
              ArgumentError("two sample buffers of different caps cannot be merged, but `a` has a `max_history` of $(w) and `b` has $(b.max_history)."))
    if iszero(a.n)
        return copy(b)
    elseif iszero(b.n)
        return copy(a)
    end
    Xa = sample_buffer(a)
    Xb = sample_buffer(b)
    @argcheck(size(Xa, 2) == size(Xb, 2),
              DimensionMismatch("two sample buffers over different numbers of assets cannot be merged, but `a` describes $(size(Xa, 2)) assets and `b` describes $(size(Xb, 2))."))
    assert_buffer_presence_agreement(a.A, b.A, "active_mask")
    assert_buffer_presence_agreement(a.M, b.M, "estimation_mask")
    assert_buffer_presence_agreement(a.F, b.F, "factor returns")
    assert_buffer_presence_agreement(a.E, b.E, "Exogenous Series")
    assert_buffer_presence_agreement(a.P, b.P, "Panel Fields")
    assert_buffer_factor_width(a.F, b.F)
    assert_buffer_series_names(a.ne, b.ne)
    ra = (a.off + 1):(a.off + a.n)
    rb = (b.off + 1):(b.off + b.n)
    X = vcat(Xa, Xb)
    A = merge_buffer_array(buffer_rows_view(a.A, ra), buffer_rows_view(b.A, rb))
    M = merge_buffer_array(buffer_rows_view(a.M, ra), buffer_rows_view(b.M, rb))
    F = merge_buffer_array(buffer_rows_view(a.F, ra), buffer_rows_view(b.F, rb))
    E = merge_buffer_array(buffer_rows_view(a.E, ra), buffer_rows_view(b.E, rb))
    t = size(X, 1)
    if !isnothing(w) && t > w
        rows = (t - w + 1):t
        X = X[rows, :]
        A = trim_merged_array(A, rows)
        M = trim_merged_array(M, rows)
        F = trim_merged_array(F, rows)
        E = trim_merged_array(E, rows)
        t = w
    end
    return SampleBufferState(t, 0, X, A, M, F, a.ne, E, panel_fields_append(a.P, b.P, t), w)
end
"""
    merge_buffer_array(Ma::Nothing, Mb::Nothing) -> Nothing
    merge_buffer_array(Ma::AbstractMatrix, Mb::AbstractMatrix) -> AbstractMatrix

Concatenates the same mask, or the factor rows, of two buffers, and passes a backing neither records through.

This is the optional part of [`merge_states`](@ref). [`assert_buffer_presence_agreement`](@ref) runs first, so the two arguments are both `nothing` or both matrices.

# Arguments

  - `Ma`: The backing of the first buffer, or `nothing`.
  - `Mb`: The backing of the second buffer, or `nothing`.

# Returns

  - `M`: The two backings stacked, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`merge_states`](@ref)
"""
function merge_buffer_array(::Nothing, ::Any)
    return nothing
end
function merge_buffer_array(Ma::AbstractMatrix, Mb::AbstractMatrix)
    return vcat(Ma, Mb)
end
"""
    trim_merged_array(M::Nothing, rows) -> Nothing
    trim_merged_array(M::AbstractMatrix, rows) -> AbstractMatrix

Keeps the rows a cap admits of a merged mask or of the merged factor rows, by index copy, and passes a backing that is not there through.

This is the cap part of [`merge_states`](@ref). It copies the rows and does not take a view. A view would keep the whole concatenation in memory, with the rows that the cap drops, and the merged buffer would carry a backing of a different type from the backing that a fold makes.

# Arguments

  - `M`: The merged backing, or `nothing`.
  - `rows`: The rows the cap admits.

# Returns

  - `M`: The admitted rows, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`merge_states`](@ref)
"""
function trim_merged_array(::Nothing, ::Any)
    return nothing
end
function trim_merged_array(M::AbstractMatrix, rows)
    return M[rows, :]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`SampleBufferState`](@ref), so the copy shares no array with the original.

This is the `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref) calls before it folds. It copies each backing matrix whole, spare capacity included. So the copy appends on the same terms as the original, and the same `off` and `n` index the masks, the factor rows, the series and the observations. The copy shares the Panel Fields and the names of the series with the original. Nothing writes into the arrays of a panel or into the names, and an append makes new arrays.

# Arguments

  - `x`: The buffer to copy.

# Returns

  - `state::SampleBufferState`: A fresh buffer, equal to `x`, whose backing matrices are fresh matrices and whose Panel Fields are the Panel Fields of `x`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::SampleBufferState)
    return SampleBufferState(x.n, x.off, copy(x.X), copy_buffer_array(x.A),
                             copy_buffer_array(x.M), copy_buffer_array(x.F), x.ne,
                             copy_buffer_array(x.E), x.P, x.max_history)
end
"""
    copy_buffer_array(M::Nothing) -> Nothing
    copy_buffer_array(M::AbstractMatrix) -> AbstractMatrix

Copies one backing matrix of a buffer, and passes a matrix that is not there through.

This is the `nothing` case of `copy` for the two masks, the factor rows and the series. [`Base.copy`](@ref) and [`port_opt_view`](@ref) use it to copy each optional field in one line.

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

A buffer holds its observations verbatim, so the slice of the buffer is the buffer of the sliced universe, column for column. The observation axis does not change. The masks are per cell, so they slice on the asset axis by the same indices. The factor rows and the Exogenous Series are copied and not sliced, because the selection indexes assets, a factor and a series belong to no asset, and a fit over a subset of the universe reads the same factors and the same series. The Panel Fields are viewed over the selected assets and not copied, because nothing writes into the arrays of a panel. A tensor Panel Field keeps its whole label axis, because the buffer holds no asset names to find the square case with. The slice copies by index and does not take a `view`. A later [`partial_fit!`](@ref) on the viewed estimator writes into its backing matrices, and with a view it would change the buffer of the original estimator. A buffer that holds no observations has no width to slice, so its view is the empty seed with the same cap.

# Algorithm

 1. Return the empty seed with the same cap when the buffer holds no observations.
 2. Copy the columns `i` of `X` and of each mask, copy `F` and `E` whole, and view each Panel Field over the columns `i`.

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
    if iszero(x.n)
        return SampleBufferState(; max_history = x.max_history)
    end
    return SampleBufferState(x.n, x.off, x.X[:, i], slice_buffer_mask(x.A, i),
                             slice_buffer_mask(x.M, i), copy_buffer_array(x.F), x.ne,
                             copy_buffer_array(x.E), panel_fields_view(x.P, :, i),
                             x.max_history)
end
"""
    slice_buffer_mask(M::Nothing, i) -> Nothing
    slice_buffer_mask(M::AbstractMatrix{<:Bool}, i) -> AbstractMatrix{<:Bool}

Slices one backing mask of a buffer to the selected assets, and passes a mask that is not there through.

This is the mask part of [`port_opt_view`](@ref) for this state, so the slice is one line for each field. It copies by index, for the reason that the method states.

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

This is the buffering arm of [`partial_fit!`](@ref). Each estimator that carries a [`SampleBufferState`](@ref) reaches this method. A family that folds exactly has methods of its own, and each of them narrows the `cache` type parameter of its estimator to the state of that fold. So a buffer never reaches them, and it reaches this method. The type of the state selects the route, and no method refuses the step.

The method covers both arms of the interface. The families that refuse the step also declare one method over both arms, and two narrower methods here would be ambiguous against each of them. So the body selects the arm by the type of `X`, and the compiler resolves that branch at each call site.

A buffer carries the per-observation masks beside the observations. So a [`CoveragePolicy`](@ref) mask goes through the wrapper as it goes through the accumulator of an estimator, and the call with no data gives it back to the batch verb. A wrapped estimator folded under a policy gives the answer of a batch fit over the same window under the same policy.

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
  - A buffer that holds observations receives the masks that it records. An `ArgumentError` is thrown otherwise.
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
    return Accessors.@set est.cache = state
end
"""
$(DocStringExtensions.TYPEDEF)

Declares that an estimator takes the online step from a buffer of the observations it has seen.

An `Online` goes directly in the estimator field that it wraps, for example `HighOrderPriorEstimator(; pe = Online(EmpiricalPrior()))`, or the `pe` of an optimiser, `JuMPOptimiser(; pe = Online(EmpiricalPrior()), slv = …)`. It is transient, as [`TimeDependent`](@ref) is. [`update_online_estimator`](@ref) walks the fields that hold a wrapper, seeds the `cache` of each wrapped estimator with a [`SampleBufferState`](@ref), and rebuilds the outer estimator through its keyword constructor. The result is an ordinary estimator that carries a state. No `Online` exists after that, so each verb downstream receives a plain estimator.

Only a field whose bound is `Onl` admits a wrapper. These are the `pe` field of a wrapping prior and of an optimiser. A moment field inside a prior, such as `ce`, `me`, `ske` or `kte`, admits none, so the bound of the field refuses `EmpiricalPrior(; ce = Online(…))` before any warm-up. The prior owns the rows. It holds its observations once, at the bottom of its chain, and it refits from those rows each member that does not fold. A buffer under one of its moments would hold the same rows a second time, so wrap the prior. Wrap a moment estimator alone only when it is the estimator under study, as in `update_online_estimator(Online(Coskewness()))`.

It differs from [`TimeDependent`](@ref) in when it resolves. A schedule resolves again at each fold, because its value changes at each fold. An `Online` resolves once, at warm-up. After warm-up, the step passes the state from fold to fold, and a second resolution would discard the buffer.

A wrapper replaces an exact fold, and it does not add to one. Each family that folds exactly narrows the `cache` type parameter of its own [`partial_fit!`](@ref) methods to the state of that fold, so a wrapped estimator never reaches them. It buffers its observations, and it answers each call with no data with the batch verb over the rows of the buffer. The type of the state selects the route, and the caller selects the type when it wraps the estimator or not.

Wrap an estimator in one of these cases:

  - Its estimate has no exact incremental fold.
  - It folds exactly, and it carries its observations for a consumer downstream.
  - Its estimate must come from a window, not from every observation.

An estimator that folds exactly and carries nothing needs no wrapper. Unwrapped, it answers [`partial_fit!`](@ref) and seeds its own state at the first call, and it uses the memory of one state, not the memory of a buffer.

A wrapper and a [`TimeDependent`](@ref) schedule do not wrap each other, because they resolve at different times. Neither `Online(TimeDependent(…))` nor a schedule whose entry or `default` is an `Online` is admissible. A wrapper inside a schedule entry would resolve at no fold, or it would seed again at each fold and discard the buffer that the step passes on. They compose the other way, in two ordinary forms. An estimator that an `Online` wraps can hold schedules of its own, which the seed does not change and which resolve at each fold after it. One outer estimator can hold a wrapper in one field and a schedule in another, and each resolves at its own time. The two field scans are disjoint by construction, because a field that holds one kind is not a candidate of the other scan. So neither resolution reaches the wrapper of the other.

On a scheme, the same word declares an Online Scheme. `Online{<:WalkForwardEstimator}` wraps a walk-forward whose folds the loop fits by the online step, not by a refit. The loop warms one estimator up on the first training window, folds the new observations of each fold into it, and reads it out where a refit would run. Only the function constructor of the scheme builds one, [`OnlineIndexWalkForward`](@ref), [`OnlineDateWalkForward`](@ref) or [`OnlineHindsightSplit`](@ref). Each takes the keywords of its scheme except the window knob, and it sets that knob `true`, because a fold cannot remove an observation. `Online(cv)` written by hand is refused with an error that names it. On a scheme the wrapper is not transient. Nothing resolves it, the loop reads it at each fold, and each function that takes the scheme decides on its type. It carries no `max_history`, because the estimator declares a window through the wrapper on the prior.

`max_history` caps the buffer, and the cap is the window. An uncapped buffer gives the batch fit over every observation folded so far. A capped buffer gives the batch fit over the last `max_history` observations. This holds for the estimate and for each consumer that reads the observations, such as the scenario risk measures CVaR, EVaR and CDaR. The rule has no special case. A buffer always means the batch verb over its rows. An unwrapped estimator is not affected. It folds exactly, and it stays fitted over every observation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Online(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator};
           max_history::Option{<:Integer} = nothing)
    Online(; est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator},
           max_history::Option{<:Integer} = nothing)

## Validation

  - `est` is not an `Online`. An `ArgumentError` is thrown otherwise.
  - `est` is not a cross-validation scheme, because the function constructor of a scheme builds its Online Scheme. An `ArgumentError` naming the constructors is thrown otherwise.
  - `est` is not an [`OnlinePortfolioSelection`](@ref) head, because a refit of a head is a batch walk-forward, and the library runs both settings of the wrapper in other forms. An `ArgumentError` naming those forms is thrown otherwise.
  - `est` has a `cache` field. An `ArgumentError` is thrown otherwise.
  - No field of the tree of `est` holds a windowed estimator under [`SeedWindow`](@ref), because the wrapper refits the tree at each step, and a refit has no first fit to remember. [`seed_window_path`](@ref) finds the field, and an `ArgumentError` that names it is thrown otherwise.
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
  - [`OnlineIndexWalkForward`](@ref)
  - [`OnlineDateWalkForward`](@ref)
  - [`OnlineHindsightSplit`](@ref)
"""
@concrete struct Online <: AbstractEstimator
    """
    Estimator that the wrapper seeds a buffer into. The field takes this value after the wrapper resolves.
    """
    est
    """
    $(field_dict[:pf_max_history])
    """
    max_history
    function Online(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator};
                    max_history::Option{<:Integer} = nothing)
        @argcheck(hasfield(typeof(est), :cache),
                  ArgumentError("`$(typeof(est))` has no `cache` field, so it has nowhere to carry a sample buffer and cannot be wrapped in `Online`."))
        path = seed_window_path(est)
        @argcheck(isnothing(path),
                  ArgumentError("`$(typeof(est).name.name)` holds a `SeedWindow` at `$(path)`, and `Online` refits the estimator over its buffer at each step. A refit has no first fit to remember, so it cannot keep the window of the first fit alone. Fold the estimator without `Online` in a prior that folds, such as `EmpiricalPrior`, or set the rule to `RollingWindow()`."))
        if !isnothing(max_history)
            @argcheck(max_history > zero(max_history),
                      DomainError(max_history, "max_history must be positive"))
        end
        return new{typeof(est), typeof(max_history)}(est, max_history)
    end
    # The Online Scheme path. Only the function constructors of the walk-forwards call it,
    # with the window knob already set, so the wrapped scheme is expanding by construction
    # and no value is checked here.
    function Online{T1, T2}(cv::T1,
                            max_history::T2) where {T1 <: CrossValidationEstimator,
                                                    T2 <: Nothing}
        return new{T1, T2}(cv, max_history)
    end
end
function Online(cv::CrossValidationEstimator; kwargs...)
    return throw(ArgumentError("`Online` on a scheme is an Online Scheme, and one is built by its function constructor alone — `OnlineIndexWalkForward`, `OnlineDateWalkForward` or `OnlineHindsightSplit` — never by wrapping a `$(nameof(typeof(cv)))` by hand. Each constructor takes its scheme's keywords minus the window knob and sets that knob `true`, because a fold cannot un-fold an observation, so an online run is expanding by construction. A scheme with no constructor of its own has no online form."))
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

[`update_online_estimator`](@ref) replaces each wrapper that it can reach at warm-up. So a wrapper that reaches a fold is one that the warm-up did not see. A callable [`TimeDependent`](@ref) makes one. The value of a schedule is computed at each fold, after the warm-up, so nothing seeds a wrapper that the callable returns. The constructor of a schedule refuses a wrapper in its vector and `default` forms. It cannot refuse the return value of a callable, because that value does not exist until the fold.

A callable that returns the estimator that the wrapper would make does not solve the problem. A schedule replaces the value of the field at each fold, and the step passes the buffer through that field. So a schedule that returns a captured estimator returns the buffer as it was when the closure was built, and it discards each step since then.

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

The fields whose constructor signatures use this alias are the fields that can take the online step from a sample buffer, as [`TD_Option`](@ref) marks the fields that can change over folds.

# Related

  - [`Online`](@ref)
  - [`Option`](@ref)
  - [`online_fields`](@ref)
"""
const Online_Option{X} = Union{Nothing, <:Online, X}
"""
    const Onl{X} = Union{<:Online, X}

Alias for a required field that accepts a static estimator of type `X` or an [`Online`](@ref) declaration.

This is the form of [`Online_Option`](@ref) for a required field, as [`TD`](@ref) is for [`TD_Option`](@ref). A field that the outer estimator requires takes this alias. A caller can declare the online step there, and the signature refuses `nothing`.

# Related

  - [`Online`](@ref)
  - [`Online_Option`](@ref)
  - [`online_fields`](@ref)
"""
const Onl{X} = Union{<:Online, X}
"""
    const CVE_Onl = Union{<:CrossValidationEstimator, <:Online{<:CrossValidationEstimator}}

Alias for a cross-validation scheme, plain or an Online Scheme.

An Online Scheme is an [`Online`](@ref) around a walk-forward, built by the function constructor of the scheme, [`OnlineIndexWalkForward`](@ref), [`OnlineDateWalkForward`](@ref) or [`OnlineHindsightSplit`](@ref). Its supertype is `AbstractEstimator` and not `CrossValidationEstimator`, because a struct has one supertype. Each function that takes a scheme dispatches on this alias, or on an alias built from it, so the wrapped form reaches the same functions as the plain form. The fold loop then reads [`folds_are_stepped`](@ref) from its type.

# Related

  - [`CrossValidationEstimator`](@ref)
  - [`Online`](@ref)
  - [`CVER`](@ref)
  - [`folds_are_stepped`](@ref)
"""
const CVE_Onl = Union{<:CrossValidationEstimator, <:Online{<:CrossValidationEstimator}}
"""
    online_candidate_fields(x)

Field names of `x` whose type admits an [`Online`](@ref), the candidate set that [`online_fields`](@ref) narrows by value.

`fieldtype` alone decides whether a field can hold a wrapper, because a `@concrete` estimator records the type of the value in the type parameter of the field. The type of a field that holds a static estimator does not intersect [`Online`](@ref). So a generated function computes the tuple once for each estimator type, and for an estimator with no wrapper the tuple is empty at compile time. The warm-up then walks no field.

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

The scan reads the fields of the outer estimator, so the constructor signatures that admit a wrapper, through [`Online_Option`](@ref) and [`Onl`](@ref), decide which estimators can take the step from a buffer. No list is kept by hand. The scan visits only the candidates of [`online_candidate_fields`](@ref), and the compiler removes the other fields.

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

The warm-up calls it once, before the first fold. It replaces each wrapper with the estimator that the wrapper holds, rebuilt through its keyword constructor with the state of [`online_state_seed`](@ref) in `cache`. That state is an empty [`SampleBufferState`](@ref) with the cap of the wrapper, unless the family has a seed of its own. The function then rebuilds the outer estimator through its keyword constructor, so each construction check runs again. The result holds no `Online`, and each estimator that was wrapped carries the state that its refit or its carry reads.

The function finds a wrapper in the fields of the estimator that it receives, and in the fields of an estimator inside a wrapper. An outer estimator that passes estimators across a boundary of its own, such as a meta-optimiser or a pipeline, has a method of its own that recurses, as it has for [`update_time_dependent_estimator`](@ref).

The function takes no fold context, unlike [`update_time_dependent_estimator`](@ref). The seed reads nothing from a fold, because the wrapper resolves once and not at each fold.

# Algorithm

 1. For an [`Online`](@ref), rebuild the wrapped estimator with `cache` set to the state of [`online_state_seed`](@ref), and continue with that estimator.
 2. Find the fields of the estimator that hold a wrapper, with [`online_fields`](@ref). Return the estimator when there is none.
 3. Resolve the value of each of these fields with this function.
 4. Rebuild the estimator with the resolved values through [`rebuild_estimator`](@ref).

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
"""
    estimator_fields(x)

Field names of `x` whose type admits an estimator, the candidate set that [`online_entry_state`](@ref) and [`online_wrapper_path`](@ref) walk.

`fieldtype` alone decides whether a field can hold an estimator, because a `@concrete` estimator records the type of the value in the type parameter of the field. So a generated function computes the tuple once for each estimator type, as for [`online_candidate_fields`](@ref). A field that holds a vector of estimators or a result is not a candidate. A vector is a batch configuration that the step never folds into, and a result carries no state field. A field that holds a [`TimeDependent`](@ref) schedule is a candidate, because a schedule is an estimator. The `TimeDependent` methods of the two walks answer `nothing` for it, because the entries of a schedule resolve at each fold.

# Related

  - [`online_entry_state`](@ref)
  - [`online_candidate_fields`](@ref)
"""
@generated function estimator_fields(::T) where {T}
    fns = Tuple(f
                for f in fieldnames(T)
                if typeintersect(fieldtype(T, f),
                                 Union{<:AbstractEstimator,
                                       <:StatsBase.CovarianceEstimator}) !== Union{})
    return :($fns)
end
"""
    online_entry_state(est)
    online_entry_state(::TimeDependent)

Names the first field of an estimator tree that carries a partial-fit state, or answers `nothing`.

The cold start of the fold loop uses this walk. The online arm of [`fold_loop`](@ref) reads its argument as configuration only, so it refuses by name an estimator that holds a state, and this walk finds the state. The answer is the dotted path from the root, for example `"opt.pe.me.cache"` for the state of a mean estimator under a JuMP head. A [`TimeDependent`](@ref) schedule answers `nothing`, because its entries are batch configuration that resolves at each fold, and the loop passes no state through them. A value that is not an estimator also answers `nothing`.

# Algorithm

 1. Answer `"cache"` when `est` has a `cache` field that is not `nothing`.
 2. Walk the fields of [`estimator_fields`](@ref) in order, and find the path in each field with this function.
 3. Answer the first path found, with the field name and a dot before it, or `nothing` when no field gives one.

# Arguments

  - `est`: The estimator, or any value a field holds.

# Returns

  - `path::Option{<:String}`: The dotted path of the first state found, or `nothing`.

# Related

  - [`estimator_fields`](@ref)
  - [`online_wrapper_path`](@ref)
  - [`update_online_estimator`](@ref)
  - [`partial_fit_cache`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function online_entry_state(est::Union{<:AbstractEstimator,
                                       <:StatsBase.CovarianceEstimator})
    if hasfield(typeof(est), :cache) && !isnothing(getfield(est, :cache))
        return "cache"
    end
    for f in estimator_fields(est)
        path = online_entry_state(getfield(est, f))
        if !isnothing(path)
            return string(f, ".", path)
        end
    end
    return nothing
end
function online_entry_state(::Any)
    return nothing
end
"""
    online_wrapper_path(est)
    online_wrapper_path(::Online)
    online_wrapper_path(::TimeDependent)

Names the first field of an estimator tree that holds an [`Online`](@ref) declaration, or answers `nothing`.

[`assert_batch_entry`](@ref) uses this walk. Only the warm-up of the online arm of the fold loop resolves a wrapper, with [`update_online_estimator`](@ref). Without the check, a batch fit that reaches a wrapper gets a `MethodError` at `prior(pe, X)`, not a refusal. The walk is the walk of [`online_entry_state`](@ref), and the answer is the dotted path from the root. It is `"pe"` for a wrapped prior on a naive optimiser, and `"opt.pe"` for a wrapped prior under a JuMP head. The walk does not enter a field that holds a wrapper, because the warm-up resolves the wrappers below it. A wrapper at the root answers `""`, the empty path. A [`TimeDependent`](@ref) schedule answers `nothing`, because its constructor refuses a wrapper among its entries. A value that is not an estimator also answers `nothing`.

# Algorithm

 1. Answer `""` when `est` is an [`Online`](@ref).
 2. Walk the fields of [`estimator_fields`](@ref) in order, and answer the name of the first field that holds a wrapper.
 3. For a field that holds no wrapper, find the path in the field with this function, and answer it with the field name and a dot before it.
 4. Answer `nothing` when no field gives a path.

# Arguments

  - `est`: The estimator, or any value a field holds.

# Returns

  - `path::Option{<:String}`: The dotted path of the first wrapper found, `""` when `est` is itself one, or `nothing`.

# Related

  - [`assert_batch_entry`](@ref)
  - [`online_entry_state`](@ref)
  - [`estimator_fields`](@ref)
  - [`update_online_estimator`](@ref)
"""
function online_wrapper_path(est::Union{<:AbstractEstimator,
                                        <:StatsBase.CovarianceEstimator})
    for f in estimator_fields(est)
        v = getfield(est, f)
        if isa(v, Online)
            return string(f)
        end
        path = online_wrapper_path(v)
        if !isnothing(path)
            return string(f, ".", path)
        end
    end
    return nothing
end
function online_wrapper_path(::Online)
    return ""
end
function online_wrapper_path(::Any)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the window rule of a windowed moment estimator, the rule that says which observations its window keeps on the online step.

A windowed estimator, such as [`WindowedCovariance`](@ref), fits its inner estimator on a window of observations. In a batch fit every rule keeps the same window, the last `window` observations. The rules differ when the estimator receives its observations one block at a time.

# Interfaces

A rule is a marker for dispatch, and it holds no data. The windowed estimators hold it in their `rule` field, and the methods that fold them dispatch on its type. A new rule needs its own fold methods for the five windowed estimators.

# Related

  - [`RollingWindow`](@ref)
  - [`SeedWindow`](@ref)
  - [`WindowedExpectedReturns`](@ref)
  - [`WindowedCovariance`](@ref)
  - [`WindowedVariance`](@ref)
  - [`WindowedCoskewness`](@ref)
  - [`WindowedCokurtosis`](@ref)
"""
abstract type AbstractWindowRule <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Keeps the last observations of the window at every fit, so the window rolls with each new observation. This is the default rule.

The estimator does not fold. On the online step, a prior that carries the observations, such as the carry of [`EmpiricalPrior`](@ref), refits it over the carried rows at each read, and the window of each refit ends at the last row. So the call with no data after any stream of blocks equals the batch fit over every row received.

# Constructors

    RollingWindow() -> RollingWindow

# Examples

```jldoctest
julia> RollingWindow()
RollingWindow()
```

# Related

  - [`AbstractWindowRule`](@ref)
  - [`SeedWindow`](@ref)
"""
struct RollingWindow <: AbstractWindowRule end
"""
$(DocStringExtensions.TYPEDEF)

Keeps the last observations of the window at the first fit alone. After the first fit, the inner estimator folds every new observation, and no observation leaves the estimate.

In a batch fit the rule keeps the same window as [`RollingWindow`](@ref). On the online step the estimator folds. Its first fold cuts the first block to the window, folds the rows that remain into the inner estimator, and returns the estimator with `window = nothing`, so each later fold passes every row. The call with no data after a stream of blocks then gives the fit over the window of the first block, continued by the fold of every later row, and it equals no batch fit.

The rule needs an inner estimator that folds exactly, such as [`ExpWeightedCovariance`](@ref), and a prior that folds, such as the carry of [`EmpiricalPrior`](@ref). [`Online`](@ref) refuses it, because the wrapper refits its estimator at each step, and a refit has no first fit to remember.

# Constructors

    SeedWindow() -> SeedWindow

# Examples

```jldoctest
julia> SeedWindow()
SeedWindow()
```

# Related

  - [`AbstractWindowRule`](@ref)
  - [`RollingWindow`](@ref)
  - [`seed_window_path`](@ref)
  - [`partial_fit!`](@ref)
"""
struct SeedWindow <: AbstractWindowRule end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the Choice Rule of an estimator, the rule that says how a choice that a fit makes over its whole sample behaves on the online step.

Some fits make a choice that reads every observation, such as the dropped member of a Factor Family in [`CrossSectionalFactorPrior`](@ref), the factor set of each asset in [`StepwiseRegression`](@ref), or the components of [`DimensionReductionRegression`](@ref). In a batch fit every rule makes the same choice. The rules differ when the estimator receives its observations one block at a time.

# Interfaces

A rule is a marker for dispatch, and it holds no data. An estimator that makes such a choice holds the rule in its `choice` field. The step of its online fold dispatches on the type of the rule. A new rule needs its own method of that step for each estimator that holds it.

# Related

  - [`BatchChoice`](@ref)
  - [`PinnedChoice`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`DimensionReductionRegression`](@ref)
"""
abstract type AbstractChoiceRule <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Makes the choice again over every observation at each fit. This is the default rule.

On the online step a refit reads every row of its buffer, so the choice can move from one step to the next, and the call with no data after any stream of blocks equals the batch fit over the same rows. On the carry fold of a [`CrossSectionalFactorPrior`](@ref), a step that moves the dropped member of a Factor Family solves no observation of full rank again. The raw factor returns of such an observation do not depend on the dropped member, so the step selects the columns that the new member keeps, and folds the factor prior again over them.

# Constructors

    BatchChoice() -> BatchChoice

# Examples

```jldoctest
julia> BatchChoice()
BatchChoice()
```

# Related

  - [`AbstractChoiceRule`](@ref)
  - [`PinnedChoice`](@ref)
"""
struct BatchChoice <: AbstractChoiceRule end
"""
$(DocStringExtensions.TYPEDEF)

Keeps the choice of the first fit. After the first fit, every later fit reads the same choice.

In a batch fit the rule makes the same choice as [`BatchChoice`](@ref). On the online step the rule holds no state. After the first step whose rows the fit accepts, the step writes the choice into the configuration of the estimator that it returns, for example the name of the dropped member of each Factor Family into the `families` of a [`CrossSectionalFactorPrior`](@ref), the factor set of each asset into `included` of a [`StepwiseRegression`](@ref), or the components into `proj` of a [`DimensionReductionRegression`](@ref). Each later step then reads that choice. The call with no data after a stream of blocks equals the batch fit over the same rows under the choice of the first fit, which differs from a batch fit when the choice of the whole sample moves.

# Constructors

    PinnedChoice() -> PinnedChoice

# Examples

```jldoctest
julia> PinnedChoice()
PinnedChoice()
```

# Related

  - [`AbstractChoiceRule`](@ref)
  - [`BatchChoice`](@ref)
"""
struct PinnedChoice <: AbstractChoiceRule end
"""
    seed_window_path(est)
    seed_window_path(::Any)

Names the first field of an estimator tree that holds a windowed estimator under [`SeedWindow`](@ref), or answers `nothing`.

The constructor of [`Online`](@ref) uses this walk to refuse a seed window, because the wrapper refits its estimator at each step. The walk is the walk of [`online_wrapper_path`](@ref), and the answer is the dotted path from the root to the `rule` field, such as `"ce.rule"` for the covariance member of a prior. A windowed estimator under a seed window answers `"rule"` through its own method. A value that is not an estimator answers `nothing`.

# Algorithm

 1. Walk the fields of [`estimator_fields`](@ref) in order, and find the path in each field with this function.
 2. Answer the first path found, with the field name and a dot before it.
 3. Answer `nothing` when no field gives a path.

# Arguments

  - `est`: The estimator, or any value a field holds.

# Returns

  - `path::Option{<:String}`: The dotted path of the first seed window found, or `nothing`.

# Related

  - [`SeedWindow`](@ref)
  - [`Online`](@ref)
  - [`online_wrapper_path`](@ref)
  - [`estimator_fields`](@ref)
"""
function seed_window_path(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator})
    for f in estimator_fields(est)
        path = seed_window_path(getfield(est, f))
        if !isnothing(path)
            return string(f, ".", path)
        end
    end
    return nothing
end
function seed_window_path(::Any)
    return nothing
end
"""
    assert_batch_entry(est, entry::AbstractString)

Refuses an estimator that holds an [`Online`](@ref) at the entry of a batch fit, with an error that names the path of the wrapper.

A wrapper is a declaration that the online arm of the fold loop resolves at its warm-up. A batch fit runs no warm-up. A batch fit is a plain [`optimise`](@ref), or a fold of a scheme that is not an Online Scheme. So the wrapper would reach `prior(pe, X)` unresolved and get a `MethodError` that names the whole type. The refusal names the dotted path that [`online_wrapper_path`](@ref) finds, and the two exits. One exit is an Online Scheme, such as [`OnlineIndexWalkForward`](@ref). The other exit is the estimator without the wrapper. Each caller passes an optimiser, which is never a wrapper, so the path at this entry is never the empty path.

# Algorithm

 1. Find the path of the first wrapper in `est` with [`online_wrapper_path`](@ref).
 2. Refuse `est` when the path is not `nothing`, and name the path and the entry function.

# Arguments

  - `est`: The estimator handed to the entry function.
  - `entry`: The name of the entry function, as the message reads it.

# Validation

  - No field in the tree of `est` holds an [`Online`](@ref). An `ArgumentError` naming the field is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`online_wrapper_path`](@ref)
  - [`assert_online_entry`](@ref)
  - [`optimise`](@ref)
  - [`fold_loop`](@ref)
"""
function assert_batch_entry(est, entry::AbstractString)
    path = online_wrapper_path(est)
    @argcheck(isnothing(path),
              ArgumentError("`$(typeof(est).name.name)` enters $(entry) holding an `Online` at `$(path)`, and a batch fit cannot resolve it: `Online` declares the sample buffer the fold loop's online arm seeds at its warm-up and folds the wrapped estimator's rows into, and nothing else seeds one, so the wrapper would reach the batch verb unresolved. Run the estimator through an Online Scheme — `OnlineIndexWalkForward`, `OnlineDateWalkForward` or `OnlineHindsightSplit` — or set `$(path)` to the estimator it wraps and let the batch fit refit it from its rows."))
    return nothing
end

export Online, RollingWindow, SeedWindow, BatchChoice, PinnedChoice
public AbstractWindowRule, AbstractChoiceRule
