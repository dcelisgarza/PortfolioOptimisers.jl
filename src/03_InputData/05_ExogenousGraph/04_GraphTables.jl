"""
    graph_snapshots(g::ExogenousGraph) -> Vector{Tuple{Missing, ExogenousGraph}}
    graph_snapshots(s::ExogenousGraphSeries) -> Vector{Tuple}

Return the `(key, graph)` pairs of a graph: one pair with a `missing` key for a static graph, and one pair for each key of a series.

# Related

  - [`graph_dataframe`](@ref)
  - [`graph_manifest`](@ref)
"""
function graph_snapshots(g::ExogenousGraph)
    return [(missing, g)]
end
function graph_snapshots(s::ExogenousGraphSeries)
    return collect(zip(s.ts, s.graphs))
end
"""
    graph_key_type(g::ExogenousGraph) -> Type{Missing}
    graph_key_type(s::ExogenousGraphSeries) -> Type

Return the type of the key column of a graph in a table: `Missing` for a static graph, and the element type of the keys of a series.

# Related

  - [`graph_dataframe`](@ref)
"""
function graph_key_type(::ExogenousGraph)
    return Missing
end
function graph_key_type(s::ExogenousGraphSeries)
    return eltype(s.ts)
end
"""
    graph_storage(g::Nothing) -> String
    graph_storage(At::SparseArrays.AbstractSparseMatrixCSC) -> String
    graph_storage(At::AbstractMatrix) -> String

Name the storage of a snapshot in the manifest: `"none"` for a `nothing` key, `"sparse"` or `"dense"`.

# Related

  - [`graph_manifest`](@ref)
"""
function graph_storage(::Nothing)
    return "none"
end
function graph_storage(::SparseArrays.AbstractSparseMatrixCSC)
    return "sparse"
end
function graph_storage(::AbstractMatrix)
    return "dense"
end
"""
    graph_dataframe(ng::VecStr, G::AbstractVector) -> DataFrames.DataFrame
    graph_dataframe(rd::ReturnsResult) -> DataFrames.DataFrame
    graph_dataframe(pr::PricesResult) -> DataFrames.DataFrame

Write the edges of the named Exogenous Graphs in a table, one row for each edge, so that [`exogenous_graphs`](@ref) can read the graphs back.

The table has five columns: `graph`, `key`, `source`, `target` and `weight`. The key of an edge of a static graph is `missing`. The edges of a graph come in the order of its keys, and in each snapshot in the order of the columns of its adjacency, which is the order of the sources. The edges cannot show a node with no edge, the order of the nodes, a `nothing` key, the storage of an adjacency or whether a graph with one key is a series. [`graph_manifest`](@ref) holds them. Write the two tables with a package that writes a `DataFrame`, for example CSV.jl or Arrow.jl.

# Arguments

  - `ng`: The names of the graphs.
  - `G`: One graph or series for each name.
  - $(arg_dict[:rd])
  - `pr`: The [`PricesResult`](@ref).

# Validation

  - The returns data holds graphs. Raises an [`IsNothingError`](@ref).

# Returns

  - `df::DataFrames.DataFrame`: The edge table.

# Examples

```jldoctest
julia> g = ExogenousGraph(; edges = [(\"S\", \"A\", 0.4), (\"S\", \"B\", 0.6)]);

julia> graph_dataframe([\"supply\"], [g])
2×5 DataFrame
 Row │ graph   key      source  target  weight
     │ String  Missing  String  String  Float64
─────┼──────────────────────────────────────────
   1 │ supply  missing  S       A           0.4
   2 │ supply  missing  S       B           0.6
```

# Related

  - [`graph_manifest`](@ref)
  - [`exogenous_graphs`](@ref)
  - [`ExogenousGraph`](@ref)
  - [`ExogenousGraphSeries`](@ref)
"""
function graph_dataframe(ng::VecStr, G::AbstractVector)
    K = mapreduce(graph_key_type, (a, b) -> Union{a, b}, G; init = Missing)
    W = mapreduce(g -> mapreduce(p -> graph_weight_type(p[2]), promote_type,
                                 graph_snapshots(g); init = Bool), promote_type, G;
                  init = Bool)
    df = DataFrames.DataFrame(; graph = String[], key = K[], source = String[],
                              target = String[], weight = W[])
    for (name, g) in zip(ng, G), (key, s) in graph_snapshots(g)
        graph_edge_rows!(df, name, key, s)
    end
    return df
end
"""
    assert_graphs_supplied(ng::VecStr) -> VecStr
    assert_graphs_supplied(ng::Nothing)

Return the names of the Exogenous Graphs of the returns data, and refuse returns data that holds none.

# Validation

  - `!isnothing(ng)`. Raises an [`IsNothingError`](@ref).

# Related

  - [`graph_dataframe`](@ref)
  - [`graph_manifest`](@ref)
"""
function assert_graphs_supplied(ng::VecStr)
    return ng
end
function assert_graphs_supplied(::Nothing)
    return throw(IsNothingError("the returns data holds no Exogenous Graph, so there is no graph to write. Give the graphs as `ng` and `G`"))
end
"""
    graph_weight_type(g::Nothing) -> Type{Bool}
    graph_weight_type(g::ExogenousGraph) -> Type

Return the element type of the adjacency of a snapshot, and `Bool` for a `nothing` key, which holds no weight.

# Related

  - [`graph_dataframe`](@ref)
"""
function graph_weight_type(::Nothing)
    return Bool
end
function graph_weight_type(g::ExogenousGraph)
    return eltype(g.At)
end
"""
    graph_edge_rows!(df::DataFrames.DataFrame, name::AbstractString, key, g::Nothing) -> DataFrames.DataFrame
    graph_edge_rows!(df::DataFrames.DataFrame, name::AbstractString, key, g::ExogenousGraph) -> DataFrames.DataFrame

Add one row to the edge table for each edge of a snapshot. A `nothing` key has no edge.

# Related

  - [`graph_dataframe`](@ref)
  - [`stored_entries`](@ref)
"""
function graph_edge_rows!(df::DataFrames.DataFrame, ::AbstractString, ::Any, ::Nothing)
    return df
end
function graph_edge_rows!(df::DataFrames.DataFrame, name::AbstractString, key,
                          g::ExogenousGraph)
    I, J, V = stored_entries(g.At)
    for k in eachindex(I, J, V)
        push!(df, (name, key, g.nodes[J[k]], g.nodes[I[k]], V[k]))
    end
    return df
end
"""
    graph_manifest(ng::VecStr, G::AbstractVector) -> DataFrames.DataFrame
    graph_manifest(rd::ReturnsResult) -> DataFrames.DataFrame
    graph_manifest(pr::PricesResult) -> DataFrames.DataFrame

Describe the named Exogenous Graphs in a table, so that [`exogenous_graphs`](@ref) can read them back from a [`graph_dataframe`](@ref) table.

The edge table cannot show a node with no edge, the order of the nodes, a `nothing` key, the storage of an adjacency, or whether a graph is static. The manifest holds them. It has four columns: `kind`, `graph`, `key` and `label`. Its rows are, for each graph in the order of `ng`:

  - One `"graph"` row. Its label is `"static"` or `"series"`.
  - One `"snapshot"` row for each key, with a `missing` key for a static graph. Its label is `"dense"`, `"sparse"`, or `"none"` for a `nothing` key.
  - One `"node"` row for each node of the snapshot, in the order of the nodes. Its label is the name of the node.

# Arguments

  - `ng`: The names of the graphs.
  - `G`: One graph or series for each name.
  - $(arg_dict[:rd])
  - `pr`: The [`PricesResult`](@ref).

# Validation

  - The returns data holds graphs. Raises an [`IsNothingError`](@ref).

# Returns

  - `mf::DataFrames.DataFrame`: The manifest.

# Examples

```jldoctest
julia> g = ExogenousGraph(; nodes = [\"S\", \"A\", \"C\"], edges = [(\"S\", \"A\", 0.4)]);

julia> graph_manifest([\"supply\"], [g])
5×4 DataFrame
 Row │ kind      graph   key      label
     │ String    String  Missing  String
─────┼───────────────────────────────────
   1 │ graph     supply  missing  static
   2 │ snapshot  supply  missing  sparse
   3 │ node      supply  missing  S
   4 │ node      supply  missing  A
   5 │ node      supply  missing  C
```

# Related

  - [`graph_dataframe`](@ref)
  - [`exogenous_graphs`](@ref)
"""
function graph_manifest(ng::VecStr, G::AbstractVector)
    K = mapreduce(graph_key_type, (a, b) -> Union{a, b}, G; init = Missing)
    mf = DataFrames.DataFrame(; kind = String[], graph = String[], key = K[],
                              label = String[])
    for (name, g) in zip(ng, G)
        push!(mf, ("graph", name, missing, isa(g, ExogenousGraph) ? "static" : "series"))
        for (key, s) in graph_snapshots(g)
            push!(mf, ("snapshot", name, key, graph_storage(isnothing(s) ? s : s.At)))
            for node in graph_nodes(s)
                push!(mf, ("node", name, key, node))
            end
        end
    end
    return mf
end
"""
    graph_nodes(g::Nothing) -> Vector{String}
    graph_nodes(g::ExogenousGraph) -> Vector{String}

Return the nodes of a snapshot, and no node for a `nothing` key.

# Related

  - [`graph_manifest`](@ref)
"""
function graph_nodes(::Nothing)
    return String[]
end
function graph_nodes(g::ExogenousGraph)
    return g.nodes
end
"""
    graph_table_key(x::Missing) -> Missing
    graph_table_key(x::Integer) -> Int
    graph_table_key(x::Dates.AbstractTime) -> Dates.AbstractTime
    graph_table_key(x::AbstractString) -> Union{Int, Dates.Date, Dates.DateTime}

Read a key cell of a graph table back as a key.

A reader that guesses the type of a column can give a key as text, so the text method parses an integer, then a `Date`, then a `DateTime`. An empty text is a `missing` key, as a reader of text gives it.

# Validation

  - A text key parses as an integer, a `Date` or a `DateTime`. Raises an `ArgumentError`.

# Related

  - [`exogenous_graphs`](@ref)
"""
function graph_table_key(::Missing)
    return missing
end
function graph_table_key(x::Integer)
    return Int(x)
end
function graph_table_key(x::Dates.AbstractTime)
    return x
end
function graph_table_key(x::AbstractString)
    if isempty(x)
        return missing
    end
    for T in (Int, Dates.Date, Dates.DateTime)
        k = tryparse(T, x)
        if !isnothing(k)
            return k
        end
    end
    return throw(ArgumentError("the key \"$x\" of a graph table is not a row position, a Date or a DateTime"))
end
"""
    graph_table_weight(x::Number) -> Number
    graph_table_weight(x::AbstractString) -> Float64

Read a weight cell of an edge table back as a number. A reader that guesses the type of a column can give it as text.

# Related

  - [`exogenous_graphs`](@ref)
"""
function graph_table_weight(x::Number)
    return x
end
function graph_table_weight(x::AbstractString)
    return parse(Float64, x)
end
"""
    graph_table_edges(df::DataFrames.DataFrame) -> Vector{Tuple{String, Any, String, String, Any}}

Read the rows of an edge table as `(graph, key, source, target, weight)` tuples, with the names as text, see [`panel_text`](@ref), the keys read by [`graph_table_key`](@ref) and the weights by [`graph_table_weight`](@ref).

# Validation

  - The table has the columns `graph`, `key`, `source`, `target` and `weight`. Raises an `ArgumentError`.

# Related

  - [`exogenous_graphs`](@ref)
"""
function graph_table_edges(df::DataFrames.DataFrame)
    missing_cols = setdiff(["graph", "key", "source", "target", "weight"],
                           DataFrames.names(df))
    @argcheck(isempty(missing_cols),
              ArgumentError("an edge table of Exogenous Graphs has the columns graph, key, source, target and weight, and this table has no column $(join(missing_cols, ", "))"))
    return [(panel_text(r.graph), graph_table_key(r.key), panel_text(r.source),
             panel_text(r.target), graph_table_weight(r.weight)) for r in eachrow(df)]
end
"""
    graph_from_edges(rows::AbstractVector, nodes::Option{<:VecStr}, storage::AbstractString) -> Option{<:ExogenousGraph}

Build the snapshot of one key from its rows of the edge table.

# Algorithm

 1. `storage` is `"none"`: return `nothing`.
 2. Build the graph from the edge list, with `nodes` when the manifest gives them.
 3. `storage` is `"dense"`: store the adjacency in a dense matrix.

# Validation

  - A `"none"` snapshot has no edge. Raises an `ArgumentError`.
  - The rules of the edge-list form of [`ExogenousGraph`](@ref).

# Related

  - [`exogenous_graphs`](@ref)
"""
function graph_from_edges(rows::AbstractVector, nodes::Option{<:VecStr},
                          storage::AbstractString)
    if storage == "none"
        @argcheck(isempty(rows),
                  ArgumentError("the manifest gives a `nothing` key to a snapshot that has $(length(rows)) edges in the edge table"))
        return nothing
    end
    g = ExogenousGraph(; nodes = nodes,
                       edges = [(String(r[3]), String(r[4]), r[5]) for r in rows])
    return storage == "dense" ? ExogenousGraph(g.nodes, Matrix(g.At)) : g
end
"""
    exogenous_graphs(df::DataFrames.DataFrame, mf::Nothing = nothing) -> Tuple{Vector{String}, Vector}
    exogenous_graphs(df::DataFrames.DataFrame, mf::DataFrames.DataFrame) -> Tuple{Vector{String}, Vector}

Read named Exogenous Graphs back from an edge table of [`graph_dataframe`](@ref), and from its manifest of [`graph_manifest`](@ref) when it is given.

The two tables give back the `ng` and the `G` that wrote them: the order of the graphs and of the nodes, the nodes with no edge, the `nothing` keys, the static graphs and the storage of each adjacency. The edge table alone gives the graphs in the order of their first row, the nodes of each snapshot in the order of their first use, and a sparse adjacency. A graph whose keys are all `missing` is then static, and any other graph is a series. Pass the two values to a [`ReturnsResult`](@ref) as `ng` and `G`.

# Algorithm

The method that Julia selects is the algorithm.

 1. Read the rows of the edge table with [`graph_table_edges`](@ref).
 2. No manifest: group the rows by graph and then by key, in the order of their first row, and build each snapshot with [`graph_from_edges`](@ref).
 3. A manifest: read its `"graph"`, `"snapshot"` and `"node"` rows in order, take the rows of the edge table of each snapshot, and build it with its nodes and its storage.

# Arguments

  - `df`: The edge table.
  - `mf`: The manifest, or `nothing`.

# Validation

  - The rules of [`graph_table_edges`](@ref) and of [`ExogenousGraph`](@ref) and [`ExogenousGraphSeries`](@ref).
  - With no manifest, the keys of one graph are all `missing` or none of them is. Raises an `ArgumentError`.
  - With a manifest, each row of the edge table belongs to a snapshot of the manifest. Raises an `ArgumentError`.

# Returns

  - `(ng, G)`: The names of the graphs, and one graph or series for each name.

# Examples

```jldoctest
julia> g = ExogenousGraph(; nodes = [\"S\", \"A\", \"C\"], A = [0 0.4 0; 0 0 0; 0 0 0]);

julia> ng, G = exogenous_graphs(graph_dataframe([\"supply\"], [g]), graph_manifest([\"supply\"], [g]));

julia> ng
1-element Vector{String}:
 \"supply\"

julia> only(G) == g
true
```

# Related

  - [`graph_dataframe`](@ref)
  - [`graph_manifest`](@ref)
  - [`ReturnsResult`](@ref)
"""
function exogenous_graphs(df::DataFrames.DataFrame, ::Nothing = nothing)
    rows = graph_table_edges(df)
    ng = unique([r[1] for r in rows])
    G = [graph_from_table_rows(name, filter(r -> r[1] == name, rows)) for name in ng]
    return ng, G
end
function exogenous_graphs(df::DataFrames.DataFrame, mf::DataFrames.DataFrame)
    rows = graph_table_edges(df)
    specs = graph_manifest_specs(mf)
    known = Set((spec.graph, key) for spec in specs for key in spec.keys)
    k = findfirst(r -> !((r[1], r[2]) in known), rows)
    @argcheck(isnothing(k),
              ArgumentError("each row of the edge table belongs to a snapshot of the manifest, and row $k names the graph \"$(isnothing(k) ? "" : rows[k][1])\" at the key $(isnothing(k) ? "" : rows[k][2]), which the manifest does not list"))
    ng = [spec.graph for spec in specs]
    G = [graph_from_spec(spec, rows) for spec in specs]
    return ng, G
end
"""
    graph_from_table_rows(name::AbstractString, rows::AbstractVector) -> ExG_ExGS

Build one graph from its rows of an edge table read with no manifest: a static graph when each key is `missing`, and a series otherwise.

# Validation

  - The keys are all `missing` or none of them is. Raises an `ArgumentError`.

# Related

  - [`exogenous_graphs`](@ref)
"""
function graph_from_table_rows(name::AbstractString, rows::AbstractVector)
    cells = [r[2] for r in rows]
    if all(ismissing, cells)
        return graph_from_edges(rows, nothing, "sparse")
    end
    @argcheck(!any(ismissing, cells),
              ArgumentError("the edge table gives the graph \"$name\" a missing key on some rows and a key on others. A static graph has a missing key on each row, and a series has a key on each row"))
    ks = sort!(unique([k for k in skipmissing(cells)]))
    gs = [graph_from_edges(filter(r -> isequal(r[2], k), rows), nothing, "sparse")
          for k in ks]
    return ExogenousGraphSeries(ks, gs)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds what a manifest of [`graph_manifest`](@ref) states about one graph: its name, whether it is static, and the key, the storage and the nodes of each snapshot.

This type is an implementation detail and is not intended for direct use. [`graph_manifest_specs`](@ref) makes it, and [`exogenous_graphs`](@ref) reads it.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`graph_manifest_specs`](@ref)
  - [`exogenous_graphs`](@ref)
"""
@concrete struct GraphManifestEntry
    """
    `graph`: The name of the graph.
    """
    graph
    """
    `static`: Whether the graph is static.
    """
    static
    """
    `keys`: The key of each snapshot, `missing` for a static graph.
    """
    keys
    """
    `storage`: The storage of each snapshot, `"dense"`, `"sparse"` or `"none"`.
    """
    storage
    """
    `nodes`: The nodes of each snapshot, in order.
    """
    nodes
end
"""
    graph_manifest_specs(mf::DataFrames.DataFrame) -> Vector{GraphManifestEntry}

Read a manifest of [`graph_manifest`](@ref) into one [`GraphManifestEntry`](@ref) for each graph.

# Validation

  - The manifest has the columns `kind`, `graph`, `key` and `label`, and its rows come in the order that [`graph_manifest`](@ref) writes them. Raises an `ArgumentError`.

# Related

  - [`exogenous_graphs`](@ref)
"""
function graph_manifest_specs(mf::DataFrames.DataFrame)
    missing_cols = setdiff(["kind", "graph", "key", "label"], DataFrames.names(mf))
    @argcheck(isempty(missing_cols),
              ArgumentError("a manifest of Exogenous Graphs has the columns kind, graph, key and label, and this table has no column $(join(missing_cols, ", "))"))
    specs = GraphManifestEntry[]
    for r in eachrow(mf)
        graph_manifest_row!(specs, panel_text(r.kind), panel_text(r.graph),
                            graph_table_key(r.key), panel_text(r.label))
    end
    return specs
end
"""
    graph_manifest_row!(specs::AbstractVector, kind::AbstractString, graph::AbstractString, key, label::AbstractString) -> AbstractVector

Add one row of a manifest to the specifications of [`graph_manifest_specs`](@ref).

# Validation

  - The kind is `"graph"`, `"snapshot"` or `"node"`. A `"snapshot"` row follows a `"graph"` row of its graph, and a `"node"` row follows a `"snapshot"` row of its graph and key. Raises an `ArgumentError`.

# Related

  - [`graph_manifest_specs`](@ref)
"""
function graph_manifest_row!(specs::AbstractVector, kind::AbstractString,
                             graph::AbstractString, key, label::AbstractString)
    if kind == "graph"
        push!(specs,
              GraphManifestEntry(graph, label == "static",
                                 Union{Missing, Int, Dates.AbstractTime}[], String[],
                                 Vector{String}[]))
        return specs
    end
    @argcheck(kind in ("snapshot", "node") && !isempty(specs) && specs[end].graph == graph,
              ArgumentError("the manifest row of kind \"$kind\" for the graph \"$graph\" does not follow a graph row of that graph. A manifest lists, for each graph, a graph row, then a snapshot row for each key, each followed by its node rows"))
    spec = specs[end]
    if kind == "snapshot"
        push!(spec.keys, key)
        push!(spec.storage, label)
        push!(spec.nodes, String[])
    else
        @argcheck(!isempty(spec.keys) && isequal(spec.keys[end], key),
                  ArgumentError("the node row \"$label\" of the graph \"$graph\" does not follow the snapshot row of its key $key"))
        push!(spec.nodes[end], label)
    end
    return specs
end
"""
    graph_from_spec(spec::GraphManifestEntry, rows::AbstractVector) -> ExG_ExGS

Build one graph from its [`GraphManifestEntry`](@ref) and the rows of the edge table.

# Validation

  - A static graph has one snapshot. Raises an `ArgumentError`.

# Related

  - [`exogenous_graphs`](@ref)
"""
function graph_from_spec(spec::GraphManifestEntry, rows::AbstractVector)
    # The vectors in the types that `graph_manifest_row!` builds. A vector of entries has an
    # abstract element type, so the fields of one entry are not inferred.
    name = spec.graph::String
    keys = spec.keys::Vector{Union{Missing, Int, Dates.AbstractTime}}
    nodes = spec.nodes::Vector{Vector{String}}
    storage = spec.storage::Vector{String}
    # A snapshot with a graph lists its nodes, and a `"none"` snapshot reads none of them.
    snaps = [graph_from_edges(filter(r -> r[1] == name && isequal(r[2], keys[k]), rows),
                              nodes[k], storage[k]) for k in eachindex(keys)]
    if spec.static::Bool
        @argcheck(isone(length(snaps)) && !isnothing(only(snaps)),
                  ArgumentError("the manifest states that the graph \"$name\" is static, so it holds one snapshot with a graph, and it holds $(length(snaps)) snapshots"))
        return only(snaps)
    end
    return ExogenousGraphSeries([k for k in keys], [g for g in snaps])
end

export graph_dataframe, graph_manifest, exogenous_graphs
