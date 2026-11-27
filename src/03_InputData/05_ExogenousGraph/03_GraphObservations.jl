"""
    check_exogenous_graphs(ng::Nothing, G::Nothing, T, ts) -> nothing
    check_exogenous_graphs(ng::Option{<:VecStr}, G::Option{<:AbstractVector},
                           T::Option{<:Integer}, ts::Option{<:AbstractVector}) -> nothing

Check the named Exogenous Graphs of a [`ReturnsResult`](@ref) or a [`PricesResult`](@ref) against their names and against the observation axis.

The graphs share the observation axis with the returns, so an integer key of a series is a row of the returns data, and a date key compares with its timestamps. A graph has no asset axis, and a node needs no asset of the same name, so nothing else constrains the graphs.

`G` takes any vector, and this function checks that each entry is one of the two forms of [`ExG_ExGS`](@ref). A vector literal of a static graph and a series has the element type `AbstractResult`, so a bound on the element type would refuse `G = [g, s]`.

# Algorithm

The method that Julia selects is the algorithm.

 1. `ng` and `G` are both `nothing`: return `nothing`.
 2. Otherwise: check that both are given, check the names with [`assert_graph_names`](@ref), check that there is one graph for each name, and check the keys of each series with [`check_graph_keys`](@ref).

# Arguments

  - `ng`: The names of the graphs, or `nothing`.
  - `G`: One [`ExogenousGraph`](@ref) or [`ExogenousGraphSeries`](@ref) for each name, or `nothing`.
  - `T`: The number of rows of the observation axis, or `nothing` when the returns data holds no returns.
  - `ts`: The timestamps of the rows, or `nothing`.

# Validation

  - `ng` and `G` are both given or both `nothing`. Raises an [`IsNothingError`](@ref).
  - `ng` is not empty, holds no empty name and holds no name twice. Raises an [`IsEmptyError`](@ref) or an `ArgumentError`.
  - `length(ng) == length(G)`. Raises a `DimensionMismatch`.
  - The rules of [`check_graph_keys`](@ref) for each entry of `G`: each entry is an [`ExogenousGraph`](@ref) or an [`ExogenousGraphSeries`](@ref), and the keys of each series fit the observation axis.

# Returns

  - `nothing`.

# Related

  - [`ReturnsResult`](@ref)
  - [`PricesResult`](@ref)
  - [`check_graph_keys`](@ref)
"""
function check_exogenous_graphs(::Nothing, ::Nothing, ::Option{<:Integer},
                                ::Option{<:AbstractVector})::Nothing
    return nothing
end
function check_exogenous_graphs(ng::Option{<:VecStr}, G::Option{<:AbstractVector},
                                T::Option{<:Integer}, ts::Option{<:AbstractVector})::Nothing
    @argcheck(!isnothing(ng),
              IsNothingError("ng cannot be nothing if G is not `nothing`: each Exogenous Graph of G needs a name"))
    @argcheck(!isnothing(G),
              IsNothingError("G cannot be nothing if ng is not `nothing`: each name of ng needs an Exogenous Graph"))
    assert_graph_names(ng, :ng)
    @argcheck(length(ng) == length(G),
              DimensionMismatch("length(ng) == length(G) must hold, because each name of ng names one Exogenous Graph or one Exogenous Graph Series of G. Got $(length(ng)) and $(length(G))"))
    for (name, g) in zip(ng, G)
        check_graph_keys(g, name, T, ts)
    end
    return nothing
end
"""
    check_graph_keys(g::ExogenousGraph, name, T, ts) -> nothing
    check_graph_keys(s::ExogenousGraphSeries{<:VecInt}, name, T, ts) -> nothing
    check_graph_keys(s::ExogenousGraphSeries{<:VecDate}, name, T, ts) -> nothing
    check_graph_keys(x, name, T, ts)

Check the keys of one named graph against the observation axis.

# Algorithm

The method that Julia selects is the algorithm.

 1. A static graph: return `nothing`. It has no key.
 2. Integer keys: check that the returns data has rows, and that the last key is a row.
 3. Date keys: check that the returns data has timestamps, and that the type of the keys compares with them, with [`assert_timestamp_type`](@ref).
 4. Any other value: throw an `ArgumentError`, because an entry of `G` is an [`ExogenousGraph`](@ref) or an [`ExogenousGraphSeries`](@ref).

# Arguments

  - `g`, `s`, `x`: The graph, the series, or another value.
  - `name`: The name of the graph, which the messages quote.
  - `T`: The number of rows, or `nothing`.
  - `ts`: The timestamps of the rows, or `nothing`.

# Validation

  - A series with integer keys needs rows, `T` is not `nothing`. Raises an [`IsNothingError`](@ref).
  - `last(s.ts) <= T`. Raises a `DomainError`.
  - A series with date keys needs timestamps, `ts` is not `nothing`. Raises an [`IsNothingError`](@ref).
  - The keys compare with `ts`. Raises an `ArgumentError`. See [`assert_timestamp_type`](@ref).
  - `x` is a graph or a series. Raises an `ArgumentError` that names its type.

# Returns

  - `nothing`.

# Related

  - [`check_exogenous_graphs`](@ref)
  - [`ExogenousGraphSeries`](@ref)
"""
function check_graph_keys(x, name::AbstractString, ::Any, ::Any)::Nothing
    return throw(ArgumentError("each entry of G is an ExogenousGraph or an ExogenousGraphSeries, and the entry \"$name\" is a $(typeof(x))"))
end
function check_graph_keys(::ExogenousGraph, ::AbstractString, ::Any, ::Any)::Nothing
    return nothing
end
function check_graph_keys(s::ExogenousGraphSeries{<:VecInt}, name::AbstractString,
                          T::Option{<:Integer}, ::Any)::Nothing
    @argcheck(!isnothing(T),
              IsNothingError("the Exogenous Graph Series \"$name\" has row positions as keys, and the returns data holds no rows. Give the returns X or F with the series"))
    @argcheck(last(s.ts) <= T,
              DomainError(last(s.ts),
                          "the Exogenous Graph Series \"$name\" has the key $(last(s.ts)), and the returns data has $T rows. An integer key is a row position, so each key must be a row"))
    return nothing
end
function check_graph_keys(s::ExogenousGraphSeries{<:VecDate}, name::AbstractString, ::Any,
                          ts::Option{<:AbstractVector})::Nothing
    @argcheck(!isnothing(ts),
              IsNothingError("the Exogenous Graph Series \"$name\" has dates as keys, and the returns data holds no timestamps ts. Give ts, or key the series by row positions"))
    assert_timestamp_type(eltype(ts), eltype(s.ts),
                          "the keys of the Exogenous Graph Series \"$name\"")
    return nothing
end
"""
    graph_position(ng::Nothing, name) -> Int
    graph_position(ng::VecStr, name::Nothing) -> Int
    graph_position(ng::VecStr, name::AbstractString) -> Int

Return the position in `ng` of the graph that a consumer names.

A consumer names its graph with a field. Its default `nothing` reads the only graph of the returns data. Two relations that must stay apart, such as the suppliers and the customers, are two named graphs, so the default refuses returns data that holds more than one.

# Algorithm

The method that Julia selects is the algorithm.

 1. `ng` is `nothing`: throw an [`IsNothingError`](@ref), because the returns data holds no graph.
 2. `name` is `nothing`: return 1 when `ng` holds one name, and throw an `ArgumentError` that lists the names otherwise.
 3. `name` is a string: return its position in `ng`, and throw an `ArgumentError` that lists the names when `ng` does not hold it.

# Arguments

  - `ng`: The names of the graphs of the returns data.
  - `name`: The name that the consumer gives, or `nothing`.

# Validation

  - The rules of the three steps above.

# Returns

  - `k::Int`: The position of the graph.

# Related

  - [`graph_in_force`](@ref)
"""
function graph_position(::Nothing, name)::Int
    return throw(IsNothingError("the consumer reads the Exogenous Graph $(isnothing(name) ? "of the returns data" : "\"$name\""), and the returns data holds no Exogenous Graph. Give the graphs as `ng` and `G` of the ReturnsResult"))
end
function graph_position(ng::VecStr, ::Nothing)::Int
    @argcheck(isone(length(ng)),
              ArgumentError("the consumer names no Exogenous Graph, so it reads the only graph of the returns data, and the returns data holds the $(length(ng)) graphs $(pinned_repr(ng)). Name the graph that the consumer reads"))
    return 1
end
function graph_position(ng::VecStr, name::AbstractString)::Int
    k = findfirst(==(name), ng)
    @argcheck(!isnothing(k),
              ArgumentError("the consumer reads the Exogenous Graph \"$name\", and the returns data holds only the graphs $(pinned_repr(ng))"))
    return k
end
"""
    graph_snapshot(g::ExogenousGraph, name, T, ts) -> Tuple{ExogenousGraph, Nothing}
    graph_snapshot(s::ExogenousGraphSeries{<:VecInt}, name, T, ts) -> Tuple{ExogenousGraph, Int}
    graph_snapshot(s::ExogenousGraphSeries{<:VecDate}, name, T, ts) -> Tuple{ExogenousGraph, Any}

Return the graph in force at the last row of the observation axis, with its key.

A fit reads one graph: the graph of the last row of its window. The key of a series is a change point, so the graph in force at a row is the graph of the last key at or before the row, found by `searchsortedlast`. No graph is in force before the first key or at a `nothing` key. The function then refuses, because a consumer of a graph cannot fit with no graph. A stale graph is in the data: a `nothing` key ends the snapshot before it.

# Algorithm

The method that Julia selects is the algorithm.

 1. A static graph: return the graph and `nothing`.
 2. Integer keys: find the last key at or before the row `T`.
 3. Date keys: find the last key at or before the timestamp `ts[end]`.
 4. Refuse when there is no such key, or when its graph is `nothing`. Otherwise return the graph and its key.

# Arguments

  - `g`, `s`: The graph or the series.
  - `name`: The name of the graph, which the message quotes.
  - `T`: The number of rows.
  - `ts`: The timestamps of the rows, or `nothing`.

# Validation

  - A graph is in force at the last row. Raises an `ArgumentError` that names the graph, the last row and the first key.

# Returns

  - `(g, key)`: The graph in force, and its key, or `nothing` for a static graph.

# Related

  - [`graph_in_force`](@ref)
  - [`ExogenousGraphSeries`](@ref)
"""
function graph_snapshot(g::ExogenousGraph, ::AbstractString, ::Any, ::Any)
    return g, nothing
end
function graph_snapshot(s::ExogenousGraphSeries{<:VecInt}, name::AbstractString, T::Integer,
                        ::Any)
    return graph_snapshot_at(s, name, searchsortedlast(s.ts, T), T)
end
function graph_snapshot(s::ExogenousGraphSeries{<:VecDate}, name::AbstractString, ::Any,
                        ts::AbstractVector)
    return graph_snapshot_at(s, name, searchsortedlast(s.ts, last(ts)), last(ts))
end
"""
    graph_snapshot_at(s::ExogenousGraphSeries, name::AbstractString, p::Integer, at) -> Tuple{ExogenousGraph, Any}

Return the graph and the key at position `p` of the series, and refuse when no graph is in force there.

# Arguments

  - `s`: The series.
  - `name`: The name of the graph, which the message quotes.
  - `p`: The position of the key in force, 0 when the row comes before the first key.
  - `at`: The last row or the last timestamp, which the message quotes.

# Validation

  - `p >= 1` and `s.graphs[p]` is not `nothing`. Raises an `ArgumentError` that names the graph, `at` and the first key.

# Returns

  - `(g, key)`: The graph in force and its key.

# Related

  - [`graph_snapshot`](@ref)
"""
function graph_snapshot_at(s::ExogenousGraphSeries, name::AbstractString, p::Integer, at)
    @argcheck(p >= 1 && !isnothing(s.graphs[p]),
              ArgumentError("no graph of the Exogenous Graph Series \"$name\" is in force at the last row $at of the window: $(p < 1 ? "the row comes before the first key $(first(s.ts))" : "the key $(s.ts[p]) holds `nothing`, which ends the graph before it, and the first key is $(first(s.ts))"). A fit reads the graph in force at the last row of its window. Start the window at or after a key that holds a graph"))
    return s.graphs[p], s.ts[p]
end
"""
    graph_in_force(ng::Option{<:VecStr}, G, name, T, ts) -> ExogenousGraph
    graph_in_force(rd::ReturnsResult, name::Option{<:AbstractString} = nothing) -> ExogenousGraph
    graph_in_force(pr::PricesResult, name::Option{<:AbstractString} = nothing) -> ExogenousGraph

Return the Exogenous Graph `name` that is in force at the last row of the returns data or the prices.

A consumer of a graph reads the graph in force at the last row of its window. A cross-validation fold is a view of the returns data on the rows of its window, see [`graph_view`](@ref), so the graph in force at its last row is the graph that the fold knows. A static [`ExogenousGraph`](@ref) is in force at each row.

# Algorithm

 1. Find the position of the graph with [`graph_position`](@ref). `name = nothing` reads the only graph.
 2. Find the graph in force at the last row with [`graph_snapshot`](@ref).

# Arguments

  - `ng`: The names of the graphs, or `nothing`.
  - `G`: The graphs, or `nothing`.
  - `name`: The name of the graph, or `nothing` for the only graph.
  - `T`: The number of rows.
  - `ts`: The timestamps of the rows, or `nothing`.
  - $(arg_dict[:rd])
  - `pr`: The [`PricesResult`](@ref).

# Validation

  - The returns data holds the graph `name`, or holds one graph when `name` is `nothing`. Raises an [`IsNothingError`](@ref) or an `ArgumentError`. See [`graph_position`](@ref).
  - A graph is in force at the last row. Raises an `ArgumentError`. See [`graph_snapshot`](@ref).

# Returns

  - `g::ExogenousGraph`: The graph in force.

# Examples

```jldoctest
julia> g1 = ExogenousGraph(; edges = [(\"S\", \"A\", 0.4)]);

julia> g2 = ExogenousGraph(; edges = [(\"S\", \"A\", 0.5), (\"S\", \"B\", 0.5)]);

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = rand(6, 2), ng = [\"supply\"],
                          G = [ExogenousGraphSeries(; ts = [1, 4], graphs = [g1, g2])]);

julia> graph_in_force(rd) == g2
true

julia> graph_in_force(PortfolioOptimisers.port_opt_view(rd, 1:3, :)) == g1
true
```

# Related

  - [`ExogenousGraph`](@ref)
  - [`ExogenousGraphSeries`](@ref)
  - [`graph_view`](@ref)
  - [`graph_requirements`](@ref)
"""
function graph_in_force(ng::Option{<:VecStr}, G::Option{<:AbstractVector}, name, T, ts)
    k = graph_position(ng, name)
    return first(graph_snapshot(G[k], ng[k], T, ts))
end
"""
    graph_rows(rows::AbstractVector{Bool})
    graph_rows(rows::VecInt) -> VecInt

Check the row positions that a view of an Exogenous Graph Series keeps.

[`timestamp_positions`](@ref) turns a logical mask into its positions before a view reaches this function, so a mask here is a defect of the caller. A `Bool` is an `Integer`, and a sorted mask would read as the positions 0 and 1 with no error.

# Algorithm

The method that Julia selects is the algorithm.

 1. A logical mask: throw an `ArgumentError`.
 2. Positions: check that they are not empty and in clock order, and return them.

# Arguments

  - `rows`: The parent rows that a view keeps.

# Validation

  - `rows` is not a logical mask. Raises an `ArgumentError`.
  - `!isempty(rows)`. Raises an [`IsEmptyError`](@ref).
  - `issorted(rows)`. A series of change points has no graph of a row that comes before an earlier row of the view. Raises an `ArgumentError`.

# Returns

  - `rows::VecInt`: The positions.

# Related

  - [`graph_view`](@ref)
"""
function graph_rows(::AbstractVector{Bool})
    return throw(ArgumentError("a view of an Exogenous Graph Series takes the positions of its rows, and got a logical mask. Pass `findall(mask)`, as `port_opt_view` does"))
end
function graph_rows(rows::VecInt)
    @argcheck(!isempty(rows),
              IsEmptyError("a view of an Exogenous Graph Series keeps one row or more"))
    @argcheck(issorted(rows),
              ArgumentError("a view of an Exogenous Graph Series keeps its rows in clock order, because a graph of a series is in force from its key to the next key. Sort the rows of the view"))
    return rows
end
"""
    graph_view(g::ExogenousGraph, rows, ts) -> ExogenousGraph
    graph_view(s::ExogenousGraphSeries, rows::Colon, ts) -> ExogenousGraphSeries
    graph_view(s::ExogenousGraphSeries{<:VecInt}, rows::AbstractVector, ts) -> ExogenousGraphSeries
    graph_view(s::ExogenousGraphSeries{<:VecDate}, rows::AbstractVector, ts::AbstractVector) -> ExogenousGraphSeries

Return the graph or the series that a view of the returns data on the parent rows `rows` holds.

The view keeps the **anchor**, the last key at or before the first row of the view, which can hold `nothing`. It keeps each key after the first row and at or before the last row, and it drops every other key. So a training fold holds no graph of its test window, and the graph in force at each row of the view is the graph in force at its parent row.

An integer key is a row position, so it moves to the rows of the view: the anchor becomes 1, and a later key becomes `searchsortedfirst(rows, key)`. A key in a gap of `rows` moves to the next row of the view. Of two keys that move to one row, the last stays, because it is the one in force at that row. A date key stays as it is. When no key is in force at the first row and no key comes inside the window, no graph is in force at any row of the view. The view is then a series of one `nothing` key at its first row. The view equals the series that the rows of the view state.

# Algorithm

The method that Julia selects is the algorithm.

 1. A static graph: return `g`. It is in force at each row.
 2. A `Colon`: return `s`.
 3. Integer keys: find the anchor and the keys inside the window with `searchsortedlast`, and move each to its row of the view.
 4. Date keys: find the anchor and the keys inside the window at the timestamps of the first and the last row, and view the two vectors of the series on them.

# Arguments

  - `g`, `s`: The graph or the series.
  - `rows`: The parent rows that the view keeps, as positions in clock order or a range, or a `Colon`.
  - `ts`: The timestamps of the parent rows, or `nothing`. The date method reads them.

# Validation

  - The rules of [`graph_rows`](@ref): `rows` is not a logical mask, is not empty and is in clock order.

# Returns

  - The graph, or the series of the view.

# Examples

```jldoctest
julia> g1 = ExogenousGraph(; edges = [(\"S\", \"A\", 0.4)]);

julia> g2 = ExogenousGraph(; edges = [(\"S\", \"A\", 0.5)]);

julia> s = ExogenousGraphSeries(; ts = [1, 4, 6], graphs = [g1, g2, nothing]);

julia> v = PortfolioOptimisers.graph_view(s, 3:5, nothing);

julia> v.ts
2-element Vector{Int64}:
 1
 2
```

# Related

  - [`ExogenousGraphSeries`](@ref)
  - [`graph_in_force`](@ref)
  - [`graph_vcat`](@ref)
  - [`port_opt_view`](@ref)
"""
function graph_view(g::ExogenousGraph, ::Any, ::Any)
    return g
end
function graph_view(s::ExogenousGraphSeries, ::Colon, ::Any)
    return s
end
function graph_view(s::ExogenousGraphSeries{<:VecInt}, rows::AbstractVector, ::Any)
    rows = graph_rows(rows)
    a = searchsortedlast(s.ts, first(rows))
    b = searchsortedlast(s.ts, last(rows))
    if a == 0 && b == 0
        return ExogenousGraphSeries([1], [nothing])
    end
    ks = Int[]
    gs = similar(s.graphs, 0)
    if a > 0
        push!(ks, 1)
        push!(gs, s.graphs[a])
    end
    for p in (a + 1):b
        k = searchsortedfirst(rows, s.ts[p])
        if !isempty(ks) && ks[end] == k
            gs[end] = s.graphs[p]
        else
            push!(ks, k)
            push!(gs, s.graphs[p])
        end
    end
    return ExogenousGraphSeries(ks, gs)
end
function graph_view(s::ExogenousGraphSeries{<:VecDate}, rows::AbstractVector,
                    ts::AbstractVector)
    rows = graph_rows(rows)
    t1 = ts[first(rows)]
    a = searchsortedlast(s.ts, t1)
    b = searchsortedlast(s.ts, ts[last(rows)])
    if b == 0
        return ExogenousGraphSeries([t1], [nothing])
    end
    lo = max(a, 1)
    return ExogenousGraphSeries(view(s.ts, lo:b), view(s.graphs, lo:b))
end
"""
    graphs_view(G::Nothing, rows, ts) -> Nothing
    graphs_view(G::AbstractVector, rows::Colon, ts) -> AbstractVector
    graphs_view(G::AbstractVector, rows, ts) -> Vector

Return the named graphs that a view of the returns data on the parent rows `rows` holds, with [`graph_view`](@ref) on each.

# Algorithm

The method that Julia selects is the algorithm.

 1. No graph: return `nothing`.
 2. A `Colon`: return `G`. The view keeps each row.
 3. Otherwise: view each graph with [`graph_view`](@ref).

# Arguments

  - `G`: The graphs, or `nothing`.
  - `rows`: The parent rows that the view keeps.
  - `ts`: The timestamps of the parent rows, or `nothing`.

# Returns

  - The graphs of the view.

# Related

  - [`graph_view`](@ref)
  - [`port_opt_view`](@ref)
"""
function graphs_view(::Nothing, ::Any, ::Any)
    return nothing
end
function graphs_view(G::AbstractVector, ::Colon, ::Any)
    return G
end
function graphs_view(G::AbstractVector, rows, ts)
    return graph_view.(G, Ref(rows), Ref(ts))
end
"""
    graphs_clock_view(G::Nothing, ts_new, ts_old) -> Nothing
    graphs_clock_view(G::AbstractVector, ts_new, ts_old) -> Vector

Return the named graphs of a view that keeps the timestamps `ts_new` of the clock `ts_old`, as a view of a [`PricesResult`](@ref) or [`prices_to_returns`](@ref) keeps them.

# Algorithm

The method that Julia selects is the algorithm.

 1. No graph: return `nothing`, and read no timestamp.
 2. Otherwise: find the parent rows of `ts_new` with [`matched_row_indices`](@ref), and view the graphs on them with [`graphs_view`](@ref).

# Arguments

  - `G`: The graphs, or `nothing`.
  - `ts_new`: The timestamps that the view keeps.
  - `ts_old`: The timestamps of the parent clock.

# Validation

  - Each timestamp of `ts_new` is in `ts_old`. Raises an `ArgumentError`. See [`matched_row_indices`](@ref).

# Returns

  - The graphs of the view.

# Related

  - [`graphs_view`](@ref)
  - [`PricesResult`](@ref)
"""
function graphs_clock_view(::Nothing, ::Any, ::Any)
    return nothing
end
function graphs_clock_view(G::AbstractVector, ts_new, ts_old)
    return graphs_view(G, matched_row_indices(ts_new, ts_old), ts_old)
end
"""
    graphs_ingestion_view(G::Nothing, ts_row, ts_old, ts) -> Nothing
    graphs_ingestion_view(G::AbstractVector, ts_row, ts_old, ts) -> Vector

Return the named graphs of the caller on the clock that [`price_ingestion`](@ref) emits, with [`graph_ingestion_view`](@ref) on each.

The join and the collapse of the ingestion move the clock, so a view of the caller's clock cannot state the graphs. An outer join adds a row that the caller's table does not hold, and a collapse whose timestamp function is not `last` names a period by a timestamp that is not the row of its prices.

# Algorithm

The method that Julia selects is the algorithm.

 1. No graph: return `nothing`, and read no timestamp.
 2. Otherwise: put each graph on the emitted clock with [`graph_ingestion_view`](@ref).

# Arguments

  - `G`: The graphs, or `nothing`.
  - `ts_row`: The timestamp of the joined row that each emitted observation takes, from [`ingestion_row_clock`](@ref).
  - `ts_old`: The timestamps of the caller's asset table, the clock of an integer key.
  - `ts`: The timestamps of the emitted clock.

# Returns

  - The graphs on the emitted clock, or `nothing`.

# Related

  - [`graph_ingestion_view`](@ref)
  - [`price_ingestion`](@ref)
  - [`graphs_clock_view`](@ref)
"""
function graphs_ingestion_view(::Nothing, ::Any, ::Any, ::Any)
    return nothing
end
function graphs_ingestion_view(G::AbstractVector, ts_row, ts_old, ts)
    return [graph_ingestion_view(g, ts_row, ts_old, ts) for g in G]
end
"""
    graph_ingestion_view(g::ExogenousGraph, ts_row, ts_old, ts) -> ExogenousGraph
    graph_ingestion_view(s::ExogenousGraphSeries{<:VecInt}, ts_row, ts_old, ts) -> ExogenousGraphSeries
    graph_ingestion_view(s::ExogenousGraphSeries{<:VecDate}, ts_row, ts_old, ts) -> ExogenousGraphSeries

Return one graph of the caller on the clock that [`price_ingestion`](@ref) emits.

The emitted row `i` takes the prices of the joined row at the timestamp `ts_row[i]`, and it carries the timestamp `ts[i]`. So the graph in force at the emitted row `i` is the graph in force at `ts_row[i]`. An integer key is a row of the caller's table, so the graph in force at `ts_row[i]` is the graph in force at the last row of that table at or before `ts_row[i]`. A row that an outer join adds before the first row of the table has no graph in force. A date key compares with `ts_row[i]` directly.

The series on the emitted clock has a key at each emitted row at which another key comes in force, with [`graph_on_rows`](@ref). An integer key becomes the position of that row, and a date key becomes its timestamp `ts[i]`. So a date key that falls in a gap moves to the next emitted row, and a key after the last row is dropped, as [`graph_view`](@ref) drops it.

# Algorithm

The method that Julia selects is the algorithm.

 1. A static graph: return `g`. It is in force at each row.
 2. Integer keys: find the last row `p` of `ts_old` at or before each timestamp of `ts_row` with `searchsortedlast`, then the position of the last key at or before `p`. Build the series on the emitted rows `1:length(ts_row)`.
 3. Date keys: find the position of the last key at or before each timestamp of `ts_row` with `searchsortedlast`. Build the series on the timestamps `ts`.

# Arguments

  - `g`, `s`: The graph or the series.
  - `ts_row`: The timestamp of the joined row that each emitted observation takes.
  - `ts_old`: The timestamps of the caller's asset table.
  - `ts`: The timestamps of the emitted clock, one for each entry of `ts_row`.

# Returns

  - The graph, or the series on the emitted clock.

# Related

  - [`graphs_ingestion_view`](@ref)
  - [`graph_on_rows`](@ref)
  - [`graph_view`](@ref)
"""
function graph_ingestion_view(g::ExogenousGraph, ::Any, ::Any, ::Any)
    return g
end
function graph_ingestion_view(s::ExogenousGraphSeries{<:VecInt}, ts_row::AbstractVector,
                              ts_old::AbstractVector, ::AbstractVector)
    q = [searchsortedlast(s.ts, searchsortedlast(ts_old, t)) for t in ts_row]
    return graph_on_rows(s, q, collect(eachindex(ts_row)))
end
function graph_ingestion_view(s::ExogenousGraphSeries{<:VecDate}, ts_row::AbstractVector,
                              ::AbstractVector, ts::AbstractVector)
    q = [searchsortedlast(s.ts, t) for t in ts_row]
    return graph_on_rows(s, q, ts)
end
"""
    graph_on_rows(s::ExogenousGraphSeries, q::AbstractVector{<:Integer}, ks::AbstractVector) -> ExogenousGraphSeries

Build the series of change points that states the key `q[i]` of `s` in force at each row `i`.

A row with `q[i] == 0` comes before the first key of `s`, so no graph is in force there. The new series has a key at each row at which `q` changes to a positive position, and the key of that row is `ks[i]`. When no row has a graph in force, the series is one `nothing` key at the first row, as [`graph_view`](@ref) states it.

# Arguments

  - `s`: The series.
  - `q`: The position of the key of `s` in force at each row, in clock order, 0 before the first key.
  - `ks`: The key of each row in the new series: its position, or its timestamp.

# Returns

  - `s′::ExogenousGraphSeries`: The series on the rows.

# Related

  - [`graph_ingestion_view`](@ref)
  - [`graph_view`](@ref)
"""
function graph_on_rows(s::ExogenousGraphSeries, q::AbstractVector{<:Integer},
                       ks::AbstractVector)
    i = [j for j in eachindex(q) if q[j] > 0 && (j == firstindex(q) || q[j] != q[j - 1])]
    if isempty(i)
        return ExogenousGraphSeries(ks[1:1], [nothing])
    end
    return ExogenousGraphSeries(ks[i], s.graphs[q[i]])
end
"""
    graph_vcat(a::ExogenousGraph, b::ExogenousGraph, name, Ta, tsa, Tb, tsb) -> ExogenousGraph
    graph_vcat(a::ExogenousGraphSeries{<:VecInt}, b::ExogenousGraphSeries{<:VecInt}, name, Ta, tsa, Tb, tsb) -> ExogenousGraphSeries
    graph_vcat(a::ExogenousGraphSeries{<:VecDate}, b::ExogenousGraphSeries{<:VecDate}, name, Ta, tsa, Tb, tsb) -> ExogenousGraphSeries
    graph_vcat(a::ExG_ExGS, b::ExG_ExGS, name, Ta, tsa, Tb, tsb)

Concatenate one named graph of two blocks of observations, `a` first. It is the inverse of [`graph_view`](@ref).

Two static graphs are one graph, so they must be equal, and the concatenation is `a`. Two series join their change points. Each series is first viewed on its own rows with [`graph_view`](@ref), which drops a key outside them. The integer keys of `b` move by the `Ta` rows of `a`. The anchor of `b`, its key in force at its first row, is dropped when it holds the graph in force at the last row of `a`, because it changes nothing. When it holds another graph, a date anchor at or before the last row of `a` states that graph for a row of `a`, so the concatenation refuses it. When no key of `b` is in force at its first row and a graph is in force at the last row of `a`, a `nothing` key at the first row of `b` keeps the rows of `b` with no graph.

# Algorithm

The method that Julia selects is the algorithm.

 1. Two static graphs: check that they are equal, and return `a`.
 2. Two series with integer keys or two series with date keys: view each on its own rows, apply the rule of the anchor, move the keys of `b`, and build the joined series.
 3. Any other pair: throw an `ArgumentError`, because a static graph and a series, or a series of positions and a series of dates, are not one graph.

# Arguments

  - `a`: The graph of the earlier block.
  - `b`: The graph of the same name of the later block.
  - `name`: The name of the graph, which the messages quote.
  - `Ta`, `Tb`: The numbers of rows of the two blocks.
  - `tsa`, `tsb`: The timestamps of the two blocks, or `nothing`.

# Validation

  - Two static graphs are equal. Raises an `ArgumentError`.
  - A date anchor of `b` at or before the last row of `a` holds the graph in force at that row. Raises an `ArgumentError`.
  - `a` and `b` are of one kind. Raises an `ArgumentError`.

# Returns

  - The joined graph or series.

# Related

  - [`graph_view`](@ref)
  - [`vcat_observations`](@ref)
"""
function graph_vcat(a::ExogenousGraph, b::ExogenousGraph, name::AbstractString, ::Any,
                    ::Any, ::Any, ::Any)
    @argcheck(a == b,
              ArgumentError("a static Exogenous Graph is in force at each row, so a concatenation of two blocks needs the same graph in both, and the graph \"$name\" of the later block differs. Give a series of the two graphs with a key at the first row of the later block"))
    return a
end
function graph_vcat(a::ExogenousGraphSeries{<:VecInt}, b::ExogenousGraphSeries{<:VecInt},
                    name::AbstractString, Ta::Integer, ::Any, Tb::Integer, ::Any)
    a = graph_view(a, 1:Ta, nothing)
    b = graph_view(b, 1:Tb, nothing)
    gA = graph_at_end(a, searchsortedlast(a.ts, Ta)::Int)
    ks = Int[k for k in a.ts]
    gs = Option{ExogenousGraph}[g for g in a.graphs]
    start = 1
    if first(b.ts) == 1
        start = 2
        if !isequal(first(b.graphs), gA)
            push!(ks, Ta + 1)
            push!(gs, first(b.graphs))
        end
    elseif !isnothing(gA)
        push!(ks, Ta + 1)
        push!(gs, nothing)
    end
    for p in start:length(b.ts)
        push!(ks, Ta + b.ts[p])
        push!(gs, b.graphs[p])
    end
    return ExogenousGraphSeries(ks, gs)
end
function graph_vcat(a::ExogenousGraphSeries{<:VecDate}, b::ExogenousGraphSeries{<:VecDate},
                    name::AbstractString, Ta::Integer, tsa::AbstractVector, Tb::Integer,
                    tsb::AbstractVector)
    a = graph_view(a, 1:Ta, tsa)
    b = graph_view(b, 1:Tb, tsb)
    gA = graph_at_end(a, searchsortedlast(a.ts, last(tsa))::Int)
    ks = Dates.AbstractTime[k for k in a.ts]
    gs = Option{ExogenousGraph}[g for g in a.graphs]
    start = 1
    if first(b.ts) <= first(tsb)
        start = 2
        if !isequal(first(b.graphs), gA)
            @argcheck(first(b.ts) > last(tsa),
                      ArgumentError("the later block of the Exogenous Graph Series \"$name\" holds another graph at its key $(first(b.ts)), and that key is at or before the last row $(last(tsa)) of the earlier block, which has a graph of its own there. The two blocks state two graphs for one row"))
            push!(ks, first(b.ts))
            push!(gs, first(b.graphs))
        end
    elseif !isnothing(gA)
        push!(ks, first(tsb))
        push!(gs, nothing)
    end
    for p in start:length(b.ts)
        push!(ks, b.ts[p])
        push!(gs, b.graphs[p])
    end
    # The keys leave in the element type of their values, which the parts share.
    return ExogenousGraphSeries([k for k in ks], gs)
end
function graph_vcat(a::ExG_ExGS, b::ExG_ExGS, name::AbstractString, ::Any, ::Any, ::Any,
                    ::Any)
    return throw(ArgumentError("a concatenation of two blocks joins one kind of Exogenous Graph, and the graph \"$name\" is $(graph_kind(a)) in the earlier block and $(graph_kind(b)) in the later block. Give the same kind in both blocks"))
end
"""
    graph_at_end(s::ExogenousGraphSeries, p::Integer) -> Option{<:ExogenousGraph}

Return the graph of a series at the position `p` of its keys, or `nothing` when `p` is 0.

# Related

  - [`graph_vcat`](@ref)
"""
function graph_at_end(s::ExogenousGraphSeries, p::Integer)
    return p < 1 ? nothing : s.graphs[p]
end
"""
    graph_kind(g::ExogenousGraph) -> String
    graph_kind(s::ExogenousGraphSeries{<:VecInt}) -> String
    graph_kind(s::ExogenousGraphSeries{<:VecDate}) -> String

Name the kind of a graph in a message: a static graph, a series of row positions, or a series of dates.

# Related

  - [`graph_vcat`](@ref)
"""
function graph_kind(::ExogenousGraph)
    return "a static graph"
end
function graph_kind(::ExogenousGraphSeries{<:VecInt})
    return "a series keyed by row positions"
end
function graph_kind(::ExogenousGraphSeries{<:VecDate})
    return "a series keyed by dates"
end
"""
    graphs_vcat(ng::Nothing, Ga::Nothing, Gb::Nothing, Ta, tsa, Tb, tsb) -> Nothing
    graphs_vcat(ng::VecStr, Ga::AbstractVector, Gb::AbstractVector, Ta, tsa, Tb, tsb) -> Vector

Concatenate the named graphs of two blocks of observations, with [`graph_vcat`](@ref) on each name.

The caller checks that the two blocks hold the same names in the same order.

# Arguments

  - `ng`: The names of the graphs, or `nothing`.
  - `Ga`, `Gb`: The graphs of the earlier and the later block, or `nothing`.
  - `Ta`, `Tb`: The numbers of rows of the two blocks.
  - `tsa`, `tsb`: The timestamps of the two blocks, or `nothing`.

# Returns

  - The joined graphs, or `nothing`.

# Related

  - [`graph_vcat`](@ref)
  - [`vcat_observations`](@ref)
"""
function graphs_vcat(::Nothing, ::Nothing, ::Nothing, ::Any, ::Any, ::Any, ::Any)
    return nothing
end
function graphs_vcat(ng::VecStr, Ga::AbstractVector, Gb::AbstractVector, Ta, tsa, Tb, tsb)
    return [graph_vcat(Ga[k], Gb[k], ng[k], Ta, tsa, Tb, tsb)
            for k in eachindex(ng, Ga, Gb)]
end

export graph_in_force
