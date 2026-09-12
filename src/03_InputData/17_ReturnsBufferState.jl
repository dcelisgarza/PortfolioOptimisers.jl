"""
$(DocStringExtensions.TYPEDEF)

Carries the fold context an optimiser keeps beside its prior, so that a read-out can rebuild the [`ReturnsResult`](@ref) the batch path reads.

The state of the optimiser's online step, decided by [#867](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/867). An optimiser forwards each observation to its prior and to nothing else, and a read-out runs the ordinary batch path over a `ReturnsResult` rebuilt from the state — because every meta-optimiser hands its inner optimisers a view of the caller's carrier, and every [`UniverseSets`](@ref) constraint resolves by asset name, so a read-out that handed the batch path a bare matrix would hand it an empty panel and no names.

The returns are **owned once**, by the prior at the bottom of the chain, which carries them in its own state; this state holds them only where no prior sits beneath the optimiser — [`EqualWeighted`](@ref) and [`RandomWeighted`](@ref), which read the observations and hold no prior. The factor column is owned once on the same terms: the prior's buffer records it wherever the prior's estimator tree reads it, and this state keeps it only where the tree never does — [`needs_factor_returns`](@ref) answering `false` — or where no prior sits beneath. Everything else the carrier holds and the prior does not is here: the benchmark column as a buffer of its own, the timestamps, and the context that is pinned by the first step and checked at every step after it — the asset, factor and benchmark names, and a static [`AssetPanel`](@ref). A time-varying panel is not pinned: its masks ride with the observations they explain into the returns buffer, and the read-out rebuilds the panel from them.

Every column buffer is a [`SampleBufferState`](@ref), so the orientation, the cap, the merge, the copy and the asset slice are inherited per column, and `max_history` is one cap over every column. A single-column benchmark is held as a plain vector, as the timestamps are, because a vector benchmark is not indexed by asset and a slice leaves it alone.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReturnsBufferState(;
        nx::Option{<:VecStr} = nothing,
        X::Option{<:SampleBufferState} = nothing,
        nf::Option{<:VecStr} = nothing,
        F::Option{<:SampleBufferState} = nothing,
        nb::Option{<:VecStr} = nothing,
        B::Option{<:Union{<:SampleBufferState, <:AbstractVector}} = nothing,
        ts::Option{<:AbstractVector} = nothing,
        pnl::Option{<:AssetPanel} = nothing,
        max_history::Option{<:Integer} = nothing
    ) -> ReturnsBufferState

Keywords correspond to the struct's fields. The default is the empty seed a first step fills, and the cap is the one knob a caller sets on it.

## Validation

  - `max_history > 0` when it is not `nothing`. A `DomainError` is thrown otherwise.
  - `pnl` is static when it is not `nothing`. An `ArgumentError` is thrown otherwise, because a time-varying panel's masks belong in the returns buffer.

## View parameters

When [`port_opt_view`](@ref) is called on this type, its fields are subset to the selected assets:

  - `nx`, `X`, `pnl`: Sliced to the selected assets.
  - `nb`, `B`: Sliced when the benchmark is a matrix over the assets, and carried unchanged when it is a single column.
  - `nf`, `F`, `ts`: Carried unchanged, because none of them is indexed by asset.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`SampleBufferState`](@ref)
  - [`ReturnsResult`](@ref)
  - [`returns_result`](@ref)
  - [`partial_fit!`](@ref)
"""
@concrete struct ReturnsBufferState <: AbstractPartialFitState
    """
    Names of the asset columns, pinned by the first step.
    """
    nx
    """
    Buffer of the asset returns, `observations × assets`. `nothing` when a prior beneath the optimiser carries the rows, which is every case but a prior-less head.
    """
    X
    """
    Names of the factor columns, pinned by the first step.
    """
    nf
    """
    Buffer of the factor returns, `observations × factors`. `nothing` when the carrier holds none, and when the prior beneath the optimiser records them in its own buffer, which is every prior whose tree reads them.
    """
    F
    """
    Names of the benchmark columns, pinned by the first step.
    """
    nb
    """
    Buffer of the benchmark, a [`SampleBufferState`](@ref) of `observations × assets` for a matrix benchmark and a plain vector for a single column, or `nothing` when the carrier holds none.
    """
    B
    """
    Timestamps of the observations folded, in order, or `nothing` when the carrier holds none.
    """
    ts
    """
    The static [`AssetPanel`](@ref) of the universe, pinned by the first step, or `nothing`. A time-varying panel is never held here.
    """
    pnl
    """
    $(field_dict[:pf_max_history])
    """
    max_history
end
function ReturnsBufferState(; nx::Option{<:VecStr} = nothing,
                            X::Option{<:SampleBufferState} = nothing,
                            nf::Option{<:VecStr} = nothing,
                            F::Option{<:SampleBufferState} = nothing,
                            nb::Option{<:VecStr} = nothing,
                            B::Option{<:Union{<:SampleBufferState, <:AbstractVector}} = nothing,
                            ts::Option{<:AbstractVector} = nothing,
                            pnl::Option{<:AssetPanel} = nothing,
                            max_history::Option{<:Integer} = nothing)::ReturnsBufferState
    if !isnothing(max_history)
        @argcheck(max_history > 0,
                  DomainError(max_history,
                              "max_history is the number of observations the state keeps, so it must be positive. Pass `nothing`, the default, to keep every observation folded."))
    end
    if !isnothing(pnl)
        @argcheck(panel_is_static(pnl),
                  ArgumentError("a ReturnsBufferState pins a static Asset Panel and never a time-varying one: the masks of a time-varying panel are per-observation, so they travel into the returns buffer beside the rows they explain and the read-out rebuilds the panel from them."))
    end
    return ReturnsBufferState(nx, X, nf, F, nb, B, ts, pnl, max_history)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a step whose carrier disagrees with the context the first step pinned.

The names and the static panel are context, not sample: they are fixed at the first step and every step after it must carry the same, because a state that silently took a new asset axis would fold the next observation onto the wrong column. Two carriers agree when both hold the field and the values are equal, or when neither holds it.

# Arguments

  - `pinned`: The value the first step pinned.
  - `given`: The value this step carries.
  - `name`: The field, for the message.

# Validation

  - `isequal(pinned, given)`. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`partial_fit!`](@ref)
"""
function assert_pinned_context(pinned, given, name::Symbol)::Nothing
    @argcheck(pinned_agree(pinned, given),
              ArgumentError("the online step pins `$name` at the first observation and every later step must carry the same: the state's `$name` is $(pinned_repr(pinned)) and this step's is $(pinned_repr(given)). A change of universe within one run is expressed by the Asset Panel's masks over a fixed axis, not by a new axis."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Answers whether two pinned values are the same value, by content.

`isequal` is not enough for the context a step pins: an [`AssetPanel`](@ref) is an immutable struct holding arrays, and a struct with no equality of its own compares its array fields by identity, so two panels of one universe built by two views of one carrier — which is what every step of a walk-forward hands over — would compare unequal. This walks structs field by field and arrays element by element, and compares the leaves by `isequal`.

# Arguments

  - `a`: The value the first step pinned.
  - `b`: The value this step carries.

# Returns

  - `agree::Bool`: `true` when the two are the same value.

# Related

  - [`assert_pinned_context`](@ref)
  - [`ReturnsBufferState`](@ref)
"""
function pinned_agree(a, b)::Bool
    if isa(a, AbstractArray) && isa(b, AbstractArray)
        return size(a) == size(b) && all(((x, y),) -> pinned_agree(x, y), zip(a, b))
    elseif isa(a, Union{Number, AbstractString, Symbol, Nothing}) ||
           isa(b, Union{Number, AbstractString, Symbol, Nothing})
        return isequal(a, b)
    elseif isstructtype(typeof(a)) &&
           isstructtype(typeof(b)) &&
           fieldnames(typeof(a)) == fieldnames(typeof(b))
        # The two may differ in their type parameters — a view of a panel holds views where
        # the panel holds arrays — so they are walked by field name rather than by type.
        return all(f -> pinned_agree(getfield(a, f), getfield(b, f)), fieldnames(typeof(a)))
    end
    return isequal(a, b)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders a pinned value for a refusal message: a name vector in full, and anything larger by its summary.

# Related

  - [`assert_pinned_context`](@ref)
"""
function pinned_repr(x)
    return if isnothing(x)
        "nothing"
    elseif isa(x, AbstractVector{<:AbstractString})
        repr(collect(x))
    else
        summary(x)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a step that adds a column the first step did not carry, or drops one it did.

Whether the carrier holds a factor matrix, a benchmark or timestamps is fixed by the first step, exactly as a [`SampleBufferState`](@ref) fixes whether it records a mask: a column present for some observations and absent for others cannot be rebuilt into a matrix. A buffer that has folded nothing has nothing to disagree with, so an empty state accepts either.

# Arguments

  - `buffer`: The column's buffer in the state, or `nothing`.
  - `column`: The column this step carries, or `nothing`.
  - `n`: Number of observations the state holds.
  - `name`: The field, for the message.

# Validation

  - `isnothing(buffer) == isnothing(column)`, once `n > 0`. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`partial_fit!`](@ref)
"""
function assert_column_presence(buffer, column, n::Integer, name::Symbol)::Nothing
    @argcheck(iszero(n) || isnothing(buffer) == isnothing(column),
              ArgumentError("the online step fixes at its first observation whether the carrier holds `$name`, and every later step must agree: the state $(isnothing(buffer) ? "holds no" : "holds a") `$name` and this step $(isnothing(column) ? "carries none" : "carries one"). A column is present at every step or at none, because a matrix cannot be rebuilt from rows that sometimes have it."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Number of observations a [`ReturnsBufferState`](@ref) holds.

The count is read off the first column the state keeps, because every column is appended to together and holds the same number of rows; a state that keeps no column at all — a carrier of returns and names alone, whose rows the prior holds — answers zero. The read-out checks the count against the prior's buffer rather than trusting it.

# Arguments

  - `state`: The state to count.

# Returns

  - `n::Int`: The number of observations folded.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`returns_result`](@ref)
"""
function context_count(state::ReturnsBufferState)
    return something(column_count(state.X), column_count(state.F), column_count(state.B),
                     column_count(state.ts), Some(0))
end
"""
    column_count(buffer::Nothing)
    column_count(buffer::SampleBufferState)
    column_count(buffer::AbstractVector)

Number of observations one column of a [`ReturnsBufferState`](@ref) holds, or `nothing` for a column the state does not keep.

# Related

  - [`context_count`](@ref)
  - [`ReturnsBufferState`](@ref)
"""
function column_count(::Nothing)
    return nothing
end
function column_count(buffer::SampleBufferState)
    return buffer.n
end
function column_count(buffer::AbstractVector)
    return length(buffer)
end
"""
    fold_column(buffer::Option{<:SampleBufferState}, column::Option{<:MatNum}, max_history)
    fold_column(buffer::Option{<:AbstractVector}, column::Option{<:AbstractVector}, max_history)

Appends one block of a column to its buffer, seeding the buffer on the first step.

A matrix column is a [`SampleBufferState`](@ref) and takes its block arm; a vector column — the timestamps, a single-column benchmark — is a plain vector, appended to and trimmed to the cap from the front. A column the carrier does not hold stays `nothing`.

# Arguments

  - `buffer`: The column's buffer, or `nothing` before the first step.
  - `column`: The block to append, or `nothing`.
  - `max_history`: The cap every column of the state shares.

# Returns

  - The buffer after the block, or `nothing`.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`SampleBufferState`](@ref)
"""
function fold_column(::Nothing, ::Nothing, ::Option{<:Integer})
    return nothing
end
function fold_column(buffer::Option{<:SampleBufferState}, column::MatNum,
                     max_history::Option{<:Integer})
    return partial_fit!(if isnothing(buffer)
                            SampleBufferState(; max_history = max_history)
                        else
                            buffer
                        end, column)
end
function fold_column(buffer::Option{<:AbstractVector}, column::AbstractVector,
                     max_history::Option{<:Integer})
    buffer = isnothing(buffer) ? similar(column, 0) : buffer
    append!(buffer, column)
    if !isnothing(max_history) && length(buffer) > max_history
        # A capped vector drops from the front, as the buffer drops its oldest rows.
        deleteat!(buffer, 1:(length(buffer) - max_history))
    end
    return buffer
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds the observations of a [`ReturnsResult`](@ref) into a [`ReturnsBufferState`](@ref).

The state's own step: it pins the context on the first observation, checks it on every later one, and appends each column the carrier holds to the buffer of that column. The returns are appended only when the state owns them, which `own_returns` decides once per host; a state whose prior carries the rows appends nothing for `X` and nothing for the masks, because both travel to the prior. The factor column is appended only when the state owns it, which `own_factors` decides once per host from the prior's tree: a prior that reads `F` records it in its own buffer, whose first-append rule then refuses a column that comes and goes, so this state neither keeps it nor checks its presence. The panel is either pinned, when it is static, or left to the returns buffer, when it is time-varying.

# Algorithm

 1. On the first observation, pin `nx`, `nf`, `nb` and a static `pnl`. On every other, refuse a carrier whose pinned context differs, and one that adds or drops a column the state owns.
 2. Append `X` where the state owns the rows, with the active mask of a time-varying panel.
 3. Append `F` where the state owns the factor column, and `B` and `ts` where the carrier holds them.

# Arguments

  - `state`: The state to fold into.
  - `rd`: The carrier holding one or more observations, `observations × assets`.
  - `own_returns`: Whether this state keeps the returns, which is `true` for a head that holds no prior and `false` otherwise.
  - `own_factors`: Whether this state keeps the factor column, which is `true` for a head that holds no prior and for a host whose prior's tree never reads it, and `false` otherwise.

# Validation

  - `rd.X` is not `nothing`. An `IsNothingError` is thrown otherwise.
  - The pinned context and the column presence agree with the first step. An `ArgumentError` is thrown otherwise.

# Returns

  - `state::ReturnsBufferState`: The state after the last observation.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`assert_pinned_context`](@ref)
  - [`assert_column_presence`](@ref)
  - [`fold_column`](@ref)
"""
function partial_fit!(state::ReturnsBufferState, rd::ReturnsResult;
                      own_returns::Bool = false, own_factors::Bool = true)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    n = context_count(state)
    static = isnothing(rd.pnl) || panel_is_static(rd.pnl)
    pnl = static ? rd.pnl : nothing
    if iszero(n) && isnothing(state.nx)
        state = ReturnsBufferState(; nx = rd.nx, X = state.X, nf = rd.nf, F = state.F,
                                   nb = rd.nb, B = state.B, ts = state.ts, pnl = pnl,
                                   max_history = state.max_history)
    else
        assert_pinned_context(state.nx, rd.nx, :nx)
        assert_pinned_context(state.nf, rd.nf, :nf)
        assert_pinned_context(state.nb, rd.nb, :nb)
        assert_pinned_context(state.pnl, pnl, :pnl)
        if own_factors
            assert_column_presence(state.F, rd.F, n, :F)
        end
        assert_column_presence(state.B, rd.B, n, :B)
        assert_column_presence(state.ts, rd.ts, n, :ts)
    end
    X = if own_returns
        amsk = static ? nothing : rd.pnl.amsk
        fold_column_masked(state.X, rd.X, amsk, state.max_history)
    else
        state.X
    end
    F = own_factors ? fold_column(state.F, rd.F, state.max_history) : state.F
    return ReturnsBufferState(; nx = state.nx, X = X, nf = state.nf, F = F, nb = state.nb,
                              B = fold_column(state.B, rd.B, state.max_history),
                              ts = fold_column(state.ts, rd.ts, state.max_history),
                              pnl = state.pnl, max_history = state.max_history)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Appends a block of returns to the buffer a prior-less head owns, with the active mask of its panel.

[`fold_column`](@ref) for the one column that carries a mask: a time-varying panel's active mask rides into the buffer beside the rows, so the read-out rebuilds the panel from it, and a static panel contributes none.

# Arguments

  - `buffer`: The returns buffer, or `nothing` before the first step.
  - `X`: The block to append.
  - `amsk`: The active mask of the block, or `nothing`.
  - `max_history`: The cap of the state.

# Returns

  - `buffer::SampleBufferState`: The buffer after the block.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`fold_column`](@ref)
"""
function fold_column_masked(buffer::Option{<:SampleBufferState}, X::MatNum,
                            amsk::Option{<:AbstractMatrix{<:Bool}},
                            max_history::Option{<:Integer})
    buffer = isnothing(buffer) ? SampleBufferState(; max_history = max_history) : buffer
    return partial_fit!(buffer, X; active_mask = amsk)
end
"""
    returns_result(state::ReturnsBufferState, rows::SampleBufferState)

Rebuilds the [`ReturnsResult`](@ref) the observations folded so far describe.

The reconstitution verb of the optimiser's read-out. `rows` is the buffer that holds the returns — the prior's, or the state's own when the head holds no prior — and the state holds every other column and the pinned context. The factor column comes from the state where it owns one, and from `rows` otherwise, through [`factor_buffer`](@ref): a prior whose tree reads `F` records it beside its rows, so the two are owned once and read from where they live. Every array is **materialised**, not viewed: a buffer's backing matrix is written and reallocated by the next fold, and a carrier holding a view of it would change under its holder.

The panel comes from one of two places. A static panel was pinned by the first step and is returned as it was given. A time-varying panel was never held: its active mask rode into the returns buffer, and it is rebuilt here with the estimation mask equal to the active one, because the estimation mask does not travel the step (see [`partial_fit!`](@ref) on an optimiser).

# Arguments

  - `state`: The fold context.
  - `rows`: The buffer holding the returns.

# Validation

  - `rows` and the state hold the same number of observations, where the state holds any column. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `rd::ReturnsResult`: The carrier, equal field by field to the one a batch fit over the same observations would have read.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`SampleBufferState`](@ref)
  - [`ReturnsResult`](@ref)
  - [`sample_buffer`](@ref)
  - [`sample_buffer_kwargs`](@ref)
  - [`factor_buffer`](@ref)
"""
function returns_result(state::ReturnsBufferState, rows::SampleBufferState)
    n = context_count(state)
    @argcheck(iszero(n) || n == rows.n,
              DimensionMismatch("the fold context and the returns buffer are appended to together, so they hold one number of observations, but the context holds $n and the returns buffer holds $(rows.n)."))
    X = Matrix(sample_buffer(rows))
    msk = sample_buffer_kwargs(rows)
    pnl = if haskey(msk, :active_mask)
        amsk = Matrix(msk.active_mask)
        emsk = haskey(msk, :estimation_mask) ? Matrix(msk.estimation_mask) : copy(amsk)
        AssetPanel(; amsk = amsk, emsk = emsk)
    else
        state.pnl
    end
    F = isnothing(state.F) ? column_matrix(factor_buffer(rows)) : column_matrix(state.F)
    return ReturnsResult(; nx = state.nx, X = X, nf = state.nf, F = F, nb = state.nb,
                         B = column_matrix(state.B), ts = column_matrix(state.ts),
                         pnl = pnl)
end
"""
    column_matrix(buffer::Nothing)
    column_matrix(buffer::SampleBufferState)
    column_matrix(buffer::AbstractMatrix)
    column_matrix(buffer::AbstractVector)

Materialises one column of a [`ReturnsBufferState`](@ref), or the factor rows a prior's buffer holds, for the carrier a read-out rebuilds.

# Arguments

  - `buffer`: The column's buffer, the valid region of the prior's factor rows, or `nothing`.

# Returns

  - The column as a fresh array, or `nothing`.

# Related

  - [`returns_result`](@ref)
  - [`ReturnsBufferState`](@ref)
"""
function column_matrix(::Nothing)
    return nothing
end
function column_matrix(buffer::SampleBufferState)
    return Matrix(sample_buffer(buffer))
end
function column_matrix(buffer::AbstractMatrix)
    return Matrix(buffer)
end
function column_matrix(buffer::AbstractVector)
    return copy(buffer)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds two [`ReturnsBufferState`](@ref) fitted on disjoint blocks into the state of the concatenated block.

Each column merges as its buffer merges — concatenation — and the pinned context must agree, because two runs over different universes describe no single run.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - The pinned context of `a` and `b` agree. An `ArgumentError` is thrown otherwise.

# Returns

  - `state::ReturnsBufferState`: The state the two blocks give when they are folded as one block.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`merge_states`](@ref)
"""
function merge_states(a::ReturnsBufferState, b::ReturnsBufferState)
    assert_pinned_context(a.nx, b.nx, :nx)
    assert_pinned_context(a.nf, b.nf, :nf)
    assert_pinned_context(a.nb, b.nb, :nb)
    assert_pinned_context(a.pnl, b.pnl, :pnl)
    return ReturnsBufferState(; nx = a.nx, X = merge_column(a.X, b.X), nf = a.nf,
                              F = merge_column(a.F, b.F), nb = a.nb,
                              B = merge_column(a.B, b.B), ts = merge_column(a.ts, b.ts),
                              pnl = a.pnl, max_history = a.max_history)
end
"""
    merge_column(a::Nothing, b::Nothing)
    merge_column(a::SampleBufferState, b::SampleBufferState)
    merge_column(a::AbstractVector, b::AbstractVector)

Concatenates one column of two [`ReturnsBufferState`](@ref).

# Related

  - [`merge_states`](@ref)
  - [`ReturnsBufferState`](@ref)
"""
function merge_column(::Nothing, ::Nothing)
    return nothing
end
function merge_column(a::SampleBufferState, b::SampleBufferState)
    return merge_states(a, b)
end
function merge_column(a::AbstractVector, b::AbstractVector)
    return vcat(a, b)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`ReturnsBufferState`](@ref), so the copy shares no array with the original.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::ReturnsBufferState`: A fresh state, equal to `x`.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`partial_fit`](@ref)
"""
function Base.copy(x::ReturnsBufferState)
    return ReturnsBufferState(; nx = copy_column(x.nx), X = copy_column(x.X),
                              nf = copy_column(x.nf), F = copy_column(x.F),
                              nb = copy_column(x.nb), B = copy_column(x.B),
                              ts = copy_column(x.ts), pnl = x.pnl,
                              max_history = x.max_history)
end
"""
    copy_column(x::Nothing)
    copy_column(x)

Copies one column of a [`ReturnsBufferState`](@ref), passing an absent one through.

# Related

  - [`ReturnsBufferState`](@ref)
"""
function copy_column(::Nothing)
    return nothing
end
function copy_column(x)
    return copy(x)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Slices a [`ReturnsBufferState`](@ref) to the selected assets.

The asset axis runs through `nx`, `X`, a matrix benchmark and its names, and the panel; the factors, the timestamps and a single-column benchmark are not indexed by asset and are copied unchanged, for the reason [`port_opt_view`](@ref) on a [`SampleBufferState`](@ref) copies its factor rows — a later fold on the viewed estimator must not write through into the estimator the view was taken from.

# Arguments

  - `x`: The state to slice.
  - `i`: Index or indices of the assets to keep.
  - `args...`: Additional positional arguments, forwarded to the returns buffer.

# Returns

  - `state::ReturnsBufferState`: The state of the same observations over the selected assets.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(x::ReturnsBufferState, i, args...)
    matrix_benchmark = isa(x.B, SampleBufferState)
    return ReturnsBufferState(; nx = nothing_scalar_array_view(x.nx, i),
                              X = isnothing(x.X) ? nothing : port_opt_view(x.X, i, args...),
                              nf = copy_column(x.nf), F = copy_column(x.F),
                              nb = if matrix_benchmark
                                  nothing_scalar_array_view(x.nb, i)
                              else
                                  copy_column(x.nb)
                              end, B = if matrix_benchmark
                                  port_opt_view(x.B, i)
                              else
                                  copy_column(x.B)
                              end, ts = copy_column(x.ts),
                              pnl = isnothing(x.pnl) ? nothing : port_opt_view(x.pnl, i),
                              max_history = x.max_history)
end
