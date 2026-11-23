"""
$(DocStringExtensions.TYPEDEF)

The Fold Context of an optimiser: the columns of a [`ReturnsResult`](@ref) that the prior beneath the optimiser does not keep, and the context that the first step pins.

An optimiser's online step sends each observation to its prior and to nothing else. The call `optimise(opt)` with no data rebuilds a `ReturnsResult` from the prior's buffer and from this state, and runs the batch path over it. The batch path needs the whole `ReturnsResult` and not a bare matrix, because a meta-optimiser hands its inner optimisers a view of the `ReturnsResult`, and a [`UniverseSets`](@ref) constraint finds its assets by name.

Each column has one owner. The prior at the bottom of the chain keeps the returns in its own state. This state keeps them only for a head that holds no prior, which is [`EqualWeighted`](@ref), [`RandomWeighted`](@ref) or [`BestConstantRebalancedPortfolio`](@ref). The prior's buffer keeps the factor column when the prior's estimator tree reads it, and this state keeps it when [`needs_factor_returns`](@ref) answers `false` or when there is no prior. The Exogenous Series follows the factor column. This state always keeps the benchmark column, the Exogenous Graphs and the timestamps.

The first step pins the asset, factor, benchmark, Exogenous Series and Exogenous Graph names and a static [`AssetPanel`](@ref), and every later step must carry the same values. A time-varying panel is not pinned. Its active mask goes into the returns buffer with the rows it describes, and `optimise(opt)` with no data rebuilds the panel from it.

Each matrix column is a [`SampleBufferState`](@ref), so each column takes its orientation, cap, merge, copy and asset slice from that type, and `max_history` caps every column at once. The timestamps and a single-column benchmark are plain vectors, because no asset indexes them.

The Exogenous Graphs have no row for each observation. A step joins its graphs to the graphs of the state with [`graphs_vcat`](@ref), as [`vcat_observations`](@ref) joins two blocks, and the cap keeps the graphs of the rows that stay with [`graphs_view`](@ref). An integer key of an [`ExogenousGraphSeries`](@ref) is a row, so the state counts its rows in `n`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReturnsBufferState(;
        n::Integer = 0,
        nx::Option{<:VecStr} = nothing,
        X::Option{<:SampleBufferState} = nothing,
        nf::Option{<:VecStr} = nothing,
        F::Option{<:SampleBufferState} = nothing,
        nb::Option{<:VecStr} = nothing,
        B::Option{<:Union{SampleBufferState, AbstractVector}} = nothing,
        ne::Option{<:VecStr} = nothing,
        E::Option{<:SampleBufferState} = nothing,
        ng::Option{<:VecStr} = nothing,
        G::Option{<:AbstractVector} = nothing,
        ts::Option{<:AbstractVector} = nothing,
        pnl::Option{<:AssetPanel} = nothing,
        max_history::Option{<:Integer} = nothing
    ) -> ReturnsBufferState

Keywords correspond to the struct's fields. The defaults give the empty seed that a first step fills, and a caller sets only the cap.

## Validation

  - `n >= 0`. A `DomainError` is thrown otherwise.
  - `max_history > 0` when it is not `nothing`. A `DomainError` is thrown otherwise.
  - `pnl` is static when it is not `nothing`. An `ArgumentError` is thrown otherwise, because the masks of a time-varying panel belong in the returns buffer.

## View parameters

[`port_opt_view`](@ref) takes the view of its fields at the selected assets:

  - `nx`, `X`, `pnl`: Sliced to the selected assets.
  - `nb`, `B`: Sliced when the benchmark is a matrix over the assets, and copied unchanged when it is a single column.
  - `nf`, `F`, `ne`, `E`, `ts`: Copied unchanged, because no asset indexes them.
  - `n`, `ng`, `G`: Kept unchanged, because no asset indexes them and no step writes into them.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`SampleBufferState`](@ref)
  - [`ReturnsResult`](@ref)
  - [`returns_result`](@ref)
  - [`partial_fit!`](@ref)
"""
@concrete struct ReturnsBufferState <: AbstractPartialFitState
    """
    Number of the observations that the state covers, after the cap.
    """
    n
    """
    Names of the asset columns, which the first step pins.
    """
    nx
    """
    Buffer of the asset returns, `observations × assets`. It is `nothing` when a prior beneath the optimiser keeps the rows, which is every optimiser except a head that holds no prior.
    """
    X
    """
    Names of the factor columns, which the first step pins.
    """
    nf
    """
    Buffer of the factor returns, `observations × factors`. It is `nothing` when the returns data holds none, and when the prior beneath the optimiser keeps them in its own buffer, which it does whenever its tree reads them.
    """
    F
    """
    Names of the benchmark columns, which the first step pins.
    """
    nb
    """
    Buffer of the benchmark. It is a [`SampleBufferState`](@ref) of `observations × assets` for a matrix benchmark, a plain vector for a single column, and `nothing` when the returns data holds none.
    """
    B
    """
    Names of the Exogenous Series, which the first step pins.
    """
    ne
    """
    Buffer of the Exogenous Series, `observations × series`, or `nothing` when the returns data holds none, and when the prior beneath the optimiser keeps the series in its own buffer, which it does whenever its tree reads it.
    """
    E
    """
    Names of the Exogenous Graphs, which the first step pins.
    """
    ng
    """
    The Exogenous Graphs of the observations that the state covers, one [`ExogenousGraph`](@ref) or [`ExogenousGraphSeries`](@ref) for each name, or `nothing` when the returns data holds none. No prior reads a graph, so this state always keeps them.
    """
    G
    """
    Timestamps of the folded observations, in order, or `nothing` when the returns data holds none.
    """
    ts
    """
    The static [`AssetPanel`](@ref) of the universe, which the first step pins, or `nothing`. This field never holds a time-varying panel.
    """
    pnl
    """
    $(field_dict[:pf_max_history])
    """
    max_history
    function ReturnsBufferState(n::Integer, nx::Option{<:VecStr},
                                X::Option{<:SampleBufferState}, nf::Option{<:VecStr},
                                F::Option{<:SampleBufferState}, nb::Option{<:VecStr},
                                B::Option{<:Union{SampleBufferState, AbstractVector}},
                                ne::Option{<:VecStr}, E::Option{<:SampleBufferState},
                                ng::Option{<:VecStr}, G::Option{<:AbstractVector},
                                ts::Option{<:AbstractVector}, pnl::Option{<:AssetPanel},
                                max_history::Option{<:Integer})
        @argcheck(n >= 0,
                  DomainError(n,
                              "n is the number of observations the state covers, so it must not be negative."))
        if !isnothing(max_history)
            @argcheck(max_history > 0,
                      DomainError(max_history,
                                  "max_history is the number of observations the state keeps, so it must be positive. Pass `nothing`, the default, to keep every observation folded."))
        end
        if !isnothing(pnl)
            @argcheck(panel_is_static(pnl),
                      ArgumentError("a ReturnsBufferState pins a static Asset Panel and never a time-varying one: the masks of a time-varying panel are per-observation, so they travel into the returns buffer beside the rows they explain and optimise(opt) with no data rebuilds the panel from them."))
        end
        return new{typeof(n), typeof(nx), typeof(X), typeof(nf), typeof(F), typeof(nb),
                   typeof(B), typeof(ne), typeof(E), typeof(ng), typeof(G), typeof(ts),
                   typeof(pnl), typeof(max_history)}(n, nx, X, nf, F, nb, B, ne, E, ng, G,
                                                     ts, pnl, max_history)
    end
end
function ReturnsBufferState(; n::Integer = 0, nx::Option{<:VecStr} = nothing,
                            X::Option{<:SampleBufferState} = nothing,
                            nf::Option{<:VecStr} = nothing,
                            F::Option{<:SampleBufferState} = nothing,
                            nb::Option{<:VecStr} = nothing,
                            B::Option{<:Union{SampleBufferState, AbstractVector}} = nothing,
                            ne::Option{<:VecStr} = nothing,
                            E::Option{<:SampleBufferState} = nothing,
                            ng::Option{<:VecStr} = nothing,
                            G::Option{<:AbstractVector} = nothing,
                            ts::Option{<:AbstractVector} = nothing,
                            pnl::Option{<:AssetPanel} = nothing,
                            max_history::Option{<:Integer} = nothing)::ReturnsBufferState
    return ReturnsBufferState(n, nx, X, nf, F, nb, B, ne, E, ng, G, ts, pnl, max_history)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a step whose returns data disagrees with the context that the first step pinned.

The names and the static panel are context and not sample. The first step fixes them, and every later step must carry the same values, because a state that took a new asset axis would fold the next observation onto the wrong column. [`merge_states`](@ref) applies the same check to two states. Two values agree when [`pinned_agree`](@ref) answers `true`, and two `nothing` values agree.

# Arguments

  - `pinned`: The value that the first step pinned.
  - `given`: The value that this step carries.
  - `name`: The field, for the message.

# Validation

  - `pinned_agree(pinned, given)`. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`pinned_agree`](@ref)
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

`isequal` is not enough for pinned context. An [`AssetPanel`](@ref) is an immutable struct that holds arrays, and a struct with no equality method of its own compares its array fields by identity. Two panels of one universe that come from two views of one `ReturnsResult` then compare unequal, and every step of a walk-forward hands over such a view.

# Algorithm

 1. When `a` and `b` are both arrays, answer `true` when their sizes are equal and each pair of elements agrees by this function.
 2. When `a` or `b` is a number, a string, a symbol or `nothing`, answer `isequal(a, b)`.
 3. When `a` and `b` are structs with the same field names, answer `true` when each pair of fields agrees by this function. The two types can have different parameters, because a view of a panel holds views where the panel holds arrays.
 4. Otherwise, answer `isequal(a, b)`.

# Arguments

  - `a`: The value that the first step pinned.
  - `b`: The value that this step carries.

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

Renders a pinned value for a refusal message. A vector of names renders in full, `nothing` renders as `nothing`, and any other value renders as its `summary`.

# Arguments

  - `x`: The pinned value.

# Returns

  - `s::String`: The text for the message.

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

Refuses a step or a merge that adds a column the first step did not carry, or drops a column it did carry.

The first step fixes whether the returns data holds a factor matrix, a benchmark, an Exogenous Series or timestamps, as a [`SampleBufferState`](@ref) fixes whether it records a mask. A column that is present for some observations and absent for others cannot be rebuilt into a matrix. The caller runs this check on every step after the first, also when the state keeps no column at all, and [`merge_states`](@ref) runs it on the columns of two states.

# Arguments

  - `buffer`: The column's buffer in the state, or `nothing`.
  - `column`: The column that the step carries, the buffer of the other state, or `nothing`.
  - `name`: The field, for the message.

# Validation

  - `isnothing(buffer) == isnothing(column)`. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`merge_states`](@ref)
"""
function assert_column_presence(buffer, column, name::Symbol)::Nothing
    @argcheck(isnothing(buffer) == isnothing(column),
              ArgumentError("the online step fixes at its first observation whether the returns data holds `$name`, and every later step and every merged state must agree: the state $(isnothing(buffer) ? "holds no" : "holds a") `$name` and the step or state it meets $(isnothing(column) ? "carries none" : "carries one"). A column is present at every observation or at none, because a matrix cannot be rebuilt from rows that sometimes have it."))
    return nothing
end
"""
    fold_column(buffer::Option{<:SampleBufferState}, column::Option{<:MatNum}, max_history)
    fold_column(buffer::Option{<:AbstractVector}, column::Option{<:AbstractVector}, max_history)

Appends one block of a column to its buffer, and seeds the buffer on the first step.

The vector arm writes into `buffer` and returns it.

# Algorithm

 1. When the returns data does not hold the column, return `nothing`.
 2. For a matrix column, seed a [`SampleBufferState`](@ref) with the cap when `buffer` is `nothing`, and append `column` to it with [`partial_fit!`](@ref).
 3. For a vector column, seed an empty vector with the element type of `column` when `buffer` is `nothing`, and append `column` to it.
 4. When the vector is longer than `max_history`, delete its first entries, so it keeps the last `max_history`, as a capped buffer drops its oldest rows.

# Arguments

  - `buffer`: The column's buffer, or `nothing` before the first step.
  - `column`: The block to append, or `nothing`.
  - `max_history`: The cap that every column of the state shares.

# Returns

  - The buffer after the block, or `nothing`.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`SampleBufferState`](@ref)
  - [`merge_column`](@ref)
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

The step of the state itself. The state always keeps the Exogenous Graphs, because no prior reads one. The optimiser decides once which columns the state owns. `own_returns` is `true` for a head that holds no prior, and for an optimiser whose prior folds the returns in excess of the benchmark, because its buffer then holds no raw row. A state whose prior keeps the raw rows appends no `X` and no mask. `own_factors` is `false` when the prior's tree reads `F`. The prior's buffer then records the factor column, and its own first-append rule refuses a column that comes and goes, so this state does not keep `F` and does not check its presence. `own_exogenous` is `false` when the prior's tree reads the Exogenous Series. The prior's buffer then records every column of the series and its names, so this state keeps neither `E` nor `ne`, and the buffer pins the names.

A head that holds no prior keeps the active mask of a time-varying panel and nothing else of that panel. The estimation mask and the Panel Fields do not reach the state, and no such head reads them. An optimiser with a prior refuses returns data that holds a part its route cannot record, in [`fold_prior`](@ref). The implied volatilities and `ivpa` travel with the returns, so the buffer that holds the rows records them: the prior's buffer, or the state's own `X` for a head that holds no prior.

# Algorithm

 1. Refuse an `rd` with no returns.
 2. Read `pnl`, which is the panel when it is static and `nothing` when it is time-varying.
 3. On the first step, when the state covers no observation, pin `nx`, `nf`, `nb`, `ng` and `pnl`, and pin `ne` when the state owns the series.
 4. On every later step, refuse an `rd` whose names or static panel differ from the pinned ones, and an `rd` that adds or drops a column the state owns.
 5. When the state owns the rows, append `X` with [`fold_column_masked`](@ref), with the active mask of a time-varying panel, and with `rd.iv` and `rd.ivpa`.
 6. When the state owns the factor column, append `F` with [`fold_column`](@ref). When it owns the series, append `E` with [`fold_column`](@ref).
 7. Join the Exogenous Graphs with [`fold_graphs`](@ref), which reads the timestamps of the state before the block joins them.
 8. Append `B` and `ts` with [`fold_column`](@ref), count the rows that stay after the cap, and return the new state.

# Arguments

  - `state`: The state to fold into.
  - `rd`: The returns data of one or more observations, `observations × assets`.
  - `own_returns`: Whether the state keeps the returns. It is `true` for a head that holds no prior, and for an optimiser whose prior states `x_src = ExcessSource()` and so folds the returns in excess of the benchmark. It is `false` otherwise.
  - `own_factors`: Whether the state keeps the factor column. It is `true` for a head that holds no prior and for an optimiser whose prior's tree never reads `F`, and `false` otherwise.
  - `own_exogenous`: Whether the state keeps the Exogenous Series and its names. It is `true` for a head that holds no prior and for an optimiser whose prior's tree never reads the series, and `false` otherwise.

# Validation

  - `rd.X` is not `nothing`. An `IsNothingError` is thrown otherwise.
  - After the first step, the names, the static panel and the presence of each column the state owns agree with the first step. An `ArgumentError` is thrown otherwise.
  - Every refusal of [`partial_fit!`](@ref) on a [`SampleBufferState`](@ref) holds for the matrix columns, and every refusal of [`graph_vcat`](@ref) for the graphs.

# Returns

  - `state::ReturnsBufferState`: The state after the last observation.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`assert_pinned_context`](@ref)
  - [`assert_column_presence`](@ref)
  - [`fold_column`](@ref)
  - [`fold_column_masked`](@ref)
  - [`fold_graphs`](@ref)
  - [`fold_prior`](@ref)
"""
function partial_fit!(state::ReturnsBufferState, rd::ReturnsResult;
                      own_returns::Bool = false, own_factors::Bool = true,
                      own_exogenous::Bool = true)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    static = isnothing(rd.pnl) || panel_is_static(rd.pnl)
    pnl = static ? rd.pnl : nothing
    if iszero(state.n)
        state = ReturnsBufferState(; nx = rd.nx, X = state.X, nf = rd.nf, F = state.F,
                                   nb = rd.nb, B = state.B,
                                   ne = own_exogenous ? rd.ne : nothing, E = state.E,
                                   ng = rd.ng, ts = state.ts, pnl = pnl,
                                   max_history = state.max_history)
    else
        assert_pinned_context(state.nx, rd.nx, :nx)
        assert_pinned_context(state.nf, rd.nf, :nf)
        assert_pinned_context(state.nb, rd.nb, :nb)
        assert_pinned_context(state.ng, rd.ng, :ng)
        assert_pinned_context(state.pnl, pnl, :pnl)
        if own_factors
            assert_column_presence(state.F, rd.F, :F)
        end
        if own_exogenous
            assert_pinned_context(state.ne, rd.ne, :ne)
            assert_column_presence(state.E, rd.E, :E)
        end
        assert_column_presence(state.B, rd.B, :B)
        assert_column_presence(state.ts, rd.ts, :ts)
    end
    X = if own_returns
        amsk = static ? nothing : rd.pnl.amsk
        fold_column_masked(state.X, rd, amsk, state.max_history)
    else
        state.X
    end
    F = own_factors ? fold_column(state.F, rd.F, state.max_history) : state.F
    E = own_exogenous ? fold_column(state.E, rd.E, state.max_history) : state.E
    # The timestamps fold last, because the graphs read the ones of the rows before the block.
    T = size(rd.X, 1)
    G = fold_graphs(state.ng, state.G, rd.G, state.n, T, state.ts, rd.ts, state.max_history)
    return ReturnsBufferState(; n = capped_count(state.n + T, state.max_history),
                              nx = state.nx, X = X, nf = state.nf, F = F, nb = state.nb,
                              B = fold_column(state.B, rd.B, state.max_history),
                              ne = state.ne, E = E, ng = state.ng, G = G,
                              ts = fold_column(state.ts, rd.ts, state.max_history),
                              pnl = state.pnl, max_history = state.max_history)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Number of the observations that stay of `n` folded ones under the cap `max_history`.

# Arguments

  - `n`: The number of observations folded.
  - `max_history`: The cap, or `nothing` for no cap.

# Returns

  - `n::Integer`: The number of observations that stay.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`fold_graphs`](@ref)
"""
function capped_count(n::Integer, max_history::Option{<:Integer})
    return isnothing(max_history) ? n : min(n, max_history)
end
"""
    fold_graphs(ng::Nothing, Ga, Gb, Ta, Tb, tsa, tsb, max_history) -> Nothing
    fold_graphs(ng::VecStr, Ga::Option{<:AbstractVector}, Gb::AbstractVector, Ta::Integer,
                Tb::Integer, tsa::Option{<:AbstractVector}, tsb::Option{<:AbstractVector},
                max_history::Option{<:Integer}) -> AbstractVector

Joins the Exogenous Graphs of a block to the graphs that a [`ReturnsBufferState`](@ref) holds, and keeps the graphs of the rows that stay under the cap.

The graph form of [`fold_column`](@ref). The join is the one of [`vcat_observations`](@ref), so the graphs of the state after any split of a history into blocks have the graph of the whole history in force at each row. A cap drops rows at the front, and the view keeps the graph in force at the first row that stays.

# Algorithm

The method that Julia selects is the algorithm.

 1. No graph: return `nothing`.
 2. Take `Gb` when the state covers no row, `Ta` is zero, and join `Ga` and `Gb` with [`graphs_vcat`](@ref) otherwise.
 3. When the `Ta + Tb` rows are more than the cap, view the joined graphs on the last `max_history` rows with [`graphs_view`](@ref).

# Arguments

  - `ng`: The names of the graphs, or `nothing`.
  - `Ga`: The graphs of the state, or `nothing` before the first step.
  - `Gb`: The graphs of the block.
  - `Ta`, `Tb`: The numbers of rows of the state and of the block.
  - `tsa`, `tsb`: The timestamps of the state and of the block, or `nothing`.
  - `max_history`: The cap that every column of the state shares.

# Validation

  - Every refusal of [`graph_vcat`](@ref).

# Returns

  - The graphs after the block, or `nothing`.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`graphs_vcat`](@ref)
  - [`graphs_view`](@ref)
  - [`fold_column`](@ref)
"""
function fold_graphs(::Nothing, ::Any, ::Any, ::Integer, ::Integer, ::Any, ::Any,
                     ::Option{<:Integer})
    return nothing
end
function fold_graphs(ng::VecStr, Ga::Option{<:AbstractVector}, Gb::AbstractVector,
                     Ta::Integer, Tb::Integer, tsa::Option{<:AbstractVector},
                     tsb::Option{<:AbstractVector}, max_history::Option{<:Integer})
    G = iszero(Ta) ? Gb : graphs_vcat(ng, Ga, Gb, Ta, tsa, Tb, tsb)
    T = Ta + Tb
    n = capped_count(T, max_history)
    if n == T
        return G
    end
    ts = iszero(Ta) || isnothing(tsb) ? tsb : vcat(tsa, tsb)
    return graphs_view(G, (T - n + 1):T, ts)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Appends a block of returns to the buffer that a head with no prior owns, with the active mask of its panel and the implied volatilities of the block.

The form of [`fold_column`](@ref) for the one column that carries a mask. The buffer records the active mask of a time-varying panel with the rows, so `optimise(opt)` with no data can rebuild the panel from it. A static panel gives no mask. The buffer records the implied volatilities with the rows, and pins `ivpa`, so `optimise(opt)` with no data gives them back.

# Algorithm

 1. Seed a [`SampleBufferState`](@ref) with the cap when `buffer` is `nothing`.
 2. Append `rd.X` to the buffer with [`partial_fit!`](@ref), with `amsk` as its active mask, and with `rd.iv` and `rd.ivpa`.

# Arguments

  - `buffer`: The returns buffer, or `nothing` before the first step.
  - `rd`: The returns data of the block. The function reads `X`, `iv` and `ivpa`.
  - `amsk`: The active mask of the block, or `nothing`.
  - `max_history`: The cap of the state.

# Returns

  - `buffer::SampleBufferState`: The buffer after the block.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`fold_column`](@ref)
"""
function fold_column_masked(buffer::Option{<:SampleBufferState}, rd::ReturnsResult,
                            amsk::Option{<:AbstractMatrix{<:Bool}},
                            max_history::Option{<:Integer})
    buffer = isnothing(buffer) ? SampleBufferState(; max_history = max_history) : buffer
    return partial_fit!(buffer, rd.X; active_mask = amsk, iv = rd.iv, ivpa = rd.ivpa)
end
"""
    returns_result(state::ReturnsBufferState, rows::SampleBufferState)

Rebuilds the [`ReturnsResult`](@ref) that the observations folded so far describe.

The reconstitution verb of `optimise(opt)` with no data. `rows` is the buffer that holds the returns, which is the prior's buffer, or the state's own `X` when the head holds no prior. The state holds every other column and the pinned context. The two exceptions are the factor column of a prior whose tree reads it, and the Exogenous Series with its names of a prior whose tree reads it, which `rows` holds. Every array of the result is a new copy and not a view, because the next fold writes into the buffers and can reallocate them, and a `ReturnsResult` that held a view would then change in the hands of its caller. The Panel Fields and the Exogenous Graphs are the exceptions. Nothing writes into the arrays of a panel, and a fold makes new arrays when it appends, so the result shares them with `rows`. The result shares the Exogenous Graphs with the state on the same terms.

# Algorithm

 1. Refuse a state whose count `n` differs from the count of `rows`.
 2. Copy the valid rows of the state's own returns into `X` when the state keeps them, and the valid rows of `rows` otherwise. The state keeps them for a head that holds no prior, where `rows` is the same buffer, and for an optimiser whose prior folds the returns in excess of the benchmark.
 3. When `rows` records an active mask, rebuild a time-varying [`AssetPanel`](@ref) from it with [`buffer_asset_panel`](@ref). Its estimation mask is the one that `rows` records or, when `rows` records none, a copy of the active mask. The step of a refit prior records both masks, and the carry of [`EmpiricalPrior`](@ref) and a head with no prior record the active mask alone. Its Panel Fields are the ones that `rows` records, which it records for a prior that reads them, and none otherwise. When `rows` records no active mask, take the static panel that the state pinned, or `nothing`.
 4. Take `F` from the state when the state keeps the factor column, and from [`factor_buffer`](@ref) of `rows` otherwise. Take `ne` and `E` from the state when the state keeps the series, and from [`exogenous_buffer_kwargs`](@ref) of `rows` otherwise.
 5. Take `iv` and `ivpa` from [`implied_volatility_buffer_kwargs`](@ref) of `rows`, which records them beside the returns.
 6. Copy `B` and `ts` from the state, take `ng` and `G` from it, and build the `ReturnsResult`.

# Arguments

  - `state`: The Fold Context.
  - `rows`: The buffer that holds the returns.

# Validation

  - `rows` and the state hold the same number of observations. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `rd::ReturnsResult`: The rebuilt returns data. Field by field, it is equal to the `ReturnsResult` that a batch fit over the same observations reads, with two exceptions. A key `nothing` at the first row of an Exogenous Graph Series, which states that no graph is in force there, can stay where the batch series has no such key. The graph in force at each row is the same. A time-varying panel comes back with the parts that the route of the prior recorded: both masks and the Panel Fields for a refit prior that reads them, both masks and no Panel Field on the carry fold of a [`CrossSectionalFactorPrior`](@ref), and the active mask alone on the carry of another prior or with no prior. An optimiser with a prior refuses a step that carries a part its route cannot record, and a head with no prior reads none of them. A capped buffer gives back the `iv` of the rows that it keeps, as it gives back `X`.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`SampleBufferState`](@ref)
  - [`ReturnsResult`](@ref)
  - [`sample_buffer`](@ref)
  - [`sample_buffer_kwargs`](@ref)
  - [`factor_buffer`](@ref)
  - [`exogenous_buffer_kwargs`](@ref)
  - [`implied_volatility_buffer_kwargs`](@ref)
  - [`buffer_asset_panel`](@ref)
"""
function returns_result(state::ReturnsBufferState, rows::SampleBufferState)
    @argcheck(state.n == rows.n,
              DimensionMismatch("the fold context and the returns buffer are appended to together, so they hold one number of observations, but the context holds $(state.n) and the returns buffer holds $(rows.n)."))
    # A context that keeps the returns holds the raw rows. Under `x_src = ExcessSource()`
    # the buffer of the prior holds the excess returns, and the rebuilt returns data is raw.
    X = Matrix(sample_buffer(isnothing(state.X) ? rows : state.X))
    pnl = something(buffer_asset_panel(rows), Some(state.pnl))
    F = isnothing(state.F) ? column_matrix(factor_buffer(rows)) : column_matrix(state.F)
    ex = isnothing(state.E) ? exogenous_buffer_kwargs(rows) : (; ne = state.ne, E = state.E)
    vols = implied_volatility_buffer_kwargs(rows)
    return ReturnsResult(; nx = state.nx, X = X, nf = state.nf, F = F, nb = state.nb,
                         B = column_matrix(state.B), ne = get(ex, :ne, nothing),
                         E = column_matrix(get(ex, :E, nothing)), ng = state.ng,
                         G = state.G, ts = column_matrix(state.ts),
                         iv = column_matrix(get(vols, :iv, nothing)),
                         ivpa = copy_column(get(vols, :ivpa, nothing)), pnl = pnl)
end
"""
    column_matrix(buffer::Nothing)
    column_matrix(buffer::SampleBufferState)
    column_matrix(buffer::AbstractMatrix)
    column_matrix(buffer::AbstractVector)

Copies one column of a [`ReturnsBufferState`](@ref), or the factor rows of a prior's buffer, into a new array for the `ReturnsResult` that `optimise(opt)` with no data rebuilds. A buffer gives a matrix of its valid rows, a matrix or a vector gives a copy of itself, and `nothing` gives `nothing`.

# Arguments

  - `buffer`: The column's buffer, the valid region of the prior's factor rows, or `nothing`.

# Returns

  - The column as a new array, or `nothing`.

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

The two states must describe one run, so their pinned context, their cap and the set of columns they keep must agree. Each column then merges as its own buffer merges, by concatenation, and the cap keeps the last `max_history` rows of every column, so the merged columns hold one number of observations.

# Algorithm

 1. Refuse two states whose names or static panels differ.
 2. Refuse two states whose caps differ.
 3. Refuse two states that do not keep the same columns.
 4. Merge each column with [`merge_column`](@ref) and the graphs with [`fold_graphs`](@ref), and return the new state.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - The pinned context of `a` and `b` agrees. An `ArgumentError` is thrown otherwise.
  - `a.max_history == b.max_history`. An `ArgumentError` is thrown otherwise.
  - `a` and `b` keep the same columns. An `ArgumentError` is thrown otherwise.
  - Every refusal of [`merge_states`](@ref) on a [`SampleBufferState`](@ref) holds for the matrix columns.

# Returns

  - `state::ReturnsBufferState`: The state that the two blocks give when they are folded as one block.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`merge_column`](@ref)
  - [`assert_column_presence`](@ref)
  - [`merge_states`](@ref)
"""
function merge_states(a::ReturnsBufferState, b::ReturnsBufferState)
    assert_pinned_context(a.nx, b.nx, :nx)
    assert_pinned_context(a.nf, b.nf, :nf)
    assert_pinned_context(a.nb, b.nb, :nb)
    assert_pinned_context(a.ne, b.ne, :ne)
    assert_pinned_context(a.ng, b.ng, :ng)
    assert_pinned_context(a.pnl, b.pnl, :pnl)
    w = a.max_history
    @argcheck(w == b.max_history,
              ArgumentError("two fold contexts of different caps cannot be merged, but `a` has a `max_history` of $(w) and `b` has $(b.max_history)."))
    assert_column_presence(a.X, b.X, :X)
    assert_column_presence(a.F, b.F, :F)
    assert_column_presence(a.B, b.B, :B)
    assert_column_presence(a.E, b.E, :E)
    assert_column_presence(a.ts, b.ts, :ts)
    return ReturnsBufferState(; n = capped_count(a.n + b.n, w), nx = a.nx,
                              X = merge_column(a.X, b.X, w), nf = a.nf,
                              F = merge_column(a.F, b.F, w), nb = a.nb,
                              B = merge_column(a.B, b.B, w), ne = a.ne,
                              E = merge_column(a.E, b.E, w), ng = a.ng,
                              G = fold_graphs(a.ng, a.G, b.G, a.n, b.n, a.ts, b.ts, w),
                              ts = merge_column(a.ts, b.ts, w), pnl = a.pnl,
                              max_history = w)
end
"""
    merge_column(a::Nothing, b::Nothing, max_history)
    merge_column(a::SampleBufferState, b::SampleBufferState, max_history)
    merge_column(a::AbstractVector, b::AbstractVector, max_history)

Concatenates one column of two [`ReturnsBufferState`](@ref), and keeps the last `max_history` rows. A buffer merges with [`merge_states`](@ref), which applies its own cap. A vector appends `b` to a copy of `a` with [`fold_column`](@ref), which applies the cap, so neither input changes.

# Arguments

  - `a`: The column of the first state, or `nothing`.
  - `b`: The column of the second state, or `nothing`.
  - `max_history`: The cap that both states share.

# Returns

  - The merged column, or `nothing`.

# Related

  - [`merge_states`](@ref)
  - [`fold_column`](@ref)
  - [`ReturnsBufferState`](@ref)
"""
function merge_column(::Nothing, ::Nothing, ::Option{<:Integer})
    return nothing
end
function merge_column(a::SampleBufferState, b::SampleBufferState, ::Option{<:Integer})
    return merge_states(a, b)
end
function merge_column(a::AbstractVector, b::AbstractVector, max_history::Option{<:Integer})
    return fold_column(copy(a), b, max_history)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`ReturnsBufferState`](@ref), so that a fold on the copy leaves the original unchanged.

Every column and every name vector of the copy is a new array. The pinned panel and the Exogenous Graphs are shared and not copied, because no step and no merge writes into them.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::ReturnsBufferState`: A new state, equal to `x`.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`copy_column`](@ref)
  - [`partial_fit`](@ref)
"""
function Base.copy(x::ReturnsBufferState)
    return ReturnsBufferState(; n = x.n, nx = copy_column(x.nx), X = copy_column(x.X),
                              nf = copy_column(x.nf), F = copy_column(x.F),
                              nb = copy_column(x.nb), B = copy_column(x.B),
                              ne = copy_column(x.ne), E = copy_column(x.E), ng = x.ng,
                              G = x.G, ts = copy_column(x.ts), pnl = x.pnl,
                              max_history = x.max_history)
end
"""
    copy_column(x::Nothing)
    copy_column(x)

Copies one field of a partial-fit state with `copy`, and passes a field that is `nothing` through.

# Arguments

  - `x`: The field, or `nothing`.

# Returns

  - A copy of `x`, or `nothing`.

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

The asset axis runs through `nx`, `X`, a matrix benchmark with its names, and the panel. The factors, the Exogenous Series, the timestamps and a single-column benchmark have no asset axis, so the slice copies them unchanged. An Exogenous Graph has no asset axis and no step writes into it, so the slice shares the graphs. A later fold on the sliced state appends to these vectors in place, and a shared vector would change the state that the slice came from, as [`port_opt_view`](@ref) on a [`SampleBufferState`](@ref) explains for its factor rows. The names and the panel come back as views, because no step writes into pinned context.

# Algorithm

 1. Slice `nx` and the panel to `i` as views, and slice `X` with [`port_opt_view`](@ref) on its buffer.
 2. When `B` is a [`SampleBufferState`](@ref), slice `nb` and `B` to `i`. Otherwise, copy both.
 3. Copy `nf`, `F`, `ne`, `E` and `ts`, keep `n`, `ng` and `G`, and return the new state.

# Arguments

  - `x`: The state to slice.
  - `i`: Indices of the assets to keep, as a vector or a range. A single integer is not accepted, because the names must stay a vector.
  - `args...`: More positional arguments, which go to the slice of the returns buffer.

# Returns

  - `state::ReturnsBufferState`: The state of the same observations over the selected assets.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(x::ReturnsBufferState, i, args...)
    matrix_benchmark = isa(x.B, SampleBufferState)
    return ReturnsBufferState(; n = x.n, nx = nothing_scalar_array_view(x.nx, i),
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
                              end, ne = copy_column(x.ne), E = copy_column(x.E), ng = x.ng,
                              G = x.G, ts = copy_column(x.ts),
                              pnl = isnothing(x.pnl) ? nothing : port_opt_view(x.pnl, i),
                              max_history = x.max_history)
end
