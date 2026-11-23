"""
    assert_graph_names(names::VecStr, sym::Sym_Str) -> nothing

Refuse a list of names of graph nodes, or of graphs, that is empty, that holds an empty name, or that holds a name twice.

# Arguments

  - `names`: The names to check.
  - $(arg_dict[:sym_msg])

# Validation

  - `!isempty(names)`. Raises an [`IsEmptyError`](@ref).
  - No name is the empty string. Raises an `ArgumentError`.
  - `allunique(names)`. Raises an `ArgumentError` that names the first repeated name.

# Returns

  - `nothing`.

# Related

  - [`ExogenousGraph`](@ref)
  - [`check_exogenous_graphs`](@ref)
"""
function assert_graph_names(names::VecStr, sym::Sym_Str)::Nothing
    @argcheck(!isempty(names), IsEmptyError("$sym cannot be empty"))
    @argcheck(all(!isempty, names),
              ArgumentError("$sym holds an empty name at position $(findfirst(isempty, names)), and each name must have one character or more"))
    k = findfirst(i -> names[i] in view(names, 1:(i - 1)), eachindex(names))
    @argcheck(isnothing(k),
              ArgumentError("$sym holds the name \"$(isnothing(k) ? "" : names[k])\" more than one time, and each name must be unique"))
    return nothing
end
"""
    stored_entries(A::SparseArrays.AbstractSparseMatrixCSC) -> Tuple{Vector, Vector, Vector}
    stored_entries(A::AbstractMatrix) -> Tuple{Vector, Vector, Vector}

Return the row, the column and the value of each entry of `A` that is not zero, in the order of the columns.

The sparse method reads the stored entries with `SparseArrays.findnz`, so it costs the number of stored entries. The dense method reads each cell. On the adjacency `At` of an [`ExogenousGraph`](@ref), a row is a target and a column is a source.

# Algorithm

The method that Julia selects is the algorithm.

 1. A sparse matrix: return `SparseArrays.findnz(A)` with the stored zeros removed.
 2. A dense matrix: collect each cell that is not zero, column by column.

# Arguments

  - `A`: The matrix.

# Returns

  - `(I, J, V)`: The rows, the columns and the values of the entries that are not zero.

# Related

  - [`ExogenousGraph`](@ref)
"""
function stored_entries(A::SparseArrays.AbstractSparseMatrixCSC)
    I, J, V = SparseArrays.findnz(A)
    keep = findall(!iszero, V)
    return I[keep], J[keep], V[keep]
end
function stored_entries(A::AbstractMatrix)
    I = Int[]
    J = Int[]
    V = eltype(A)[]
    for j in axes(A, 2), i in axes(A, 1)
        a = A[i, j]
        if !iszero(a)
            push!(I, i)
            push!(J, j)
            push!(V, a)
        end
    end
    return I, J, V
end
"""
    source_columns(A::SparseArrays.AbstractSparseMatrixCSC) -> SparseArrays.SparseMatrixCSC
    source_columns(A::AbstractMatrix) -> AbstractMatrix

Transpose an adjacency with the sources in its rows into the layout of an [`ExogenousGraph`](@ref), with the sources in its columns.

The transpose copies `A` one time and keeps its kind of storage, dense or sparse.

# Algorithm

The method that Julia selects is the algorithm.

 1. A sparse matrix: `permutedims(A)`, a new sparse matrix.
 2. Any other matrix: `permutedims(A)`.

# Arguments

  - `A`: The adjacency with `A[i, j]` the weight of the edge from node `i` to node `j`.

# Returns

  - `At`: The transpose of `A`.

# Related

  - [`ExogenousGraph`](@ref)
"""
function source_columns(A::SparseArrays.AbstractSparseMatrixCSC)
    return permutedims(A)
end
function source_columns(A::AbstractMatrix)
    return permutedims(A)
end
"""
    drop_stored_zeros(At::SparseArrays.AbstractSparseMatrixCSC) -> SparseArrays.AbstractSparseMatrixCSC
    drop_stored_zeros(At::AbstractMatrix) -> AbstractMatrix

Remove the stored zeros of a sparse adjacency, so that each stored entry of an [`ExogenousGraph`](@ref) is an edge.

# Algorithm

The method that Julia selects is the algorithm.

 1. A sparse matrix with a stored zero: return `SparseArrays.dropzeros(At)`, a copy. A sparse matrix with no stored zero comes back with no copy.
 2. Any other matrix: return `At`. A dense adjacency stores each cell.

# Arguments

  - `At`: The adjacency, with the sources in its columns.

# Returns

  - `At`: The adjacency with no stored zero.

# Related

  - [`ExogenousGraph`](@ref)
"""
function drop_stored_zeros(At::SparseArrays.AbstractSparseMatrixCSC)
    return any(iszero, SparseArrays.nonzeros(At)) ? SparseArrays.dropzeros(At) : At
end
function drop_stored_zeros(At::AbstractMatrix)
    return At
end
"""
    assert_finite_weights(nodes::VecStr, At::AbstractMatrix) -> nothing

Refuse an edge of an [`ExogenousGraph`](@ref) whose weight is not finite.

The function reads the entries that are not zero with [`stored_entries`](@ref), so a sparse adjacency costs the number of its edges.

# Arguments

  - `nodes`: The names of the nodes.
  - `At`: The adjacency, with the sources in its columns.

# Validation

  - Each weight is finite. Raises an [`IsNonFiniteError`](@ref) that names the first edge with a weight that is not finite.

# Returns

  - `nothing`.

# Related

  - [`ExogenousGraph`](@ref)
  - [`stored_entries`](@ref)
"""
function assert_finite_weights(nodes::VecStr, At::AbstractMatrix)::Nothing
    I, J, V = stored_entries(At)
    k = findfirst(!isfinite, V)
    @argcheck(isnothing(k),
              IsNonFiniteError("each edge weight of an Exogenous Graph must be finite, and the edge \"$(nodes[J[k]])\" => \"$(nodes[I[k]])\" has the weight $(V[k])"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Holds one directed, weighted graph over named nodes, a relation between assets and other entities that the returns do not show.

An edge from a supplier to a customer, an ownership stake or a flow of an input-output table is such a relation. The nodes have names, so a node can be an asset, an entity that is not an asset, such as a supplier that has no listing, or a label. A consumer finds the assets among the nodes by name. The graph accepts a negative weight, a self-loop and a cycle. A consumer that cannot read one of them states it with [`graph_requirements`](@ref).

The adjacency `At` holds the sources in its **columns**: `At[j, i]` is the weight of the edge from node `i` to node `j`. So the out-edges of a node are one contiguous column, a sum over the out-edges of a node is a column sum, and the exposure of a weight vector `w` to the targets of its assets is `At * w`. [`adjacency`](@ref) gives the matrix with the sources in its rows, with no copy. Each constructor takes its input with the sources in its rows, as an edge list writes them, and transposes it one time.

`At` keeps the kind of storage of its input, dense or sparse, as `ReturnsResult.X` does. A sparse adjacency holds no stored zero, so each stored entry is an edge. The block constructor and the edge-list constructor build a sparse adjacency.

`Graphs.SimpleDiGraph(g)` and `SimpleWeightedGraphs.SimpleWeightedDiGraph(g)` build the graph types of Graphs.jl at the point of use. Their vertex `k` is `g.nodes[k]`. `SimpleWeightedDiGraph` stores its weights with the targets in the rows, as `At` does, so the conversion needs no transpose.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExogenousGraph(;
        nodes::Option{<:VecStr} = nothing,
        A::Option{<:MatNum} = nothing,
        sources::Option{<:VecStr} = nothing,
        targets::Option{<:VecStr} = nothing,
        edges::Option{<:AbstractVector{<:Tuple{AbstractString, AbstractString, Number}}} = nothing,
    ) -> ExogenousGraph

The keywords give one of three forms. Each form takes the sources in its rows.

  - **Square.** `nodes` and `A`, with `size(A) == (length(nodes), length(nodes))` and `A[i, j]` the weight of the edge from `nodes[i]` to `nodes[j]`. `At` is `permutedims(A)`, dense or sparse as `A` is.
  - **Block.** `sources`, `targets` and `A`, with `size(A) == (length(sources), length(targets))` and `A[i, j]` the weight of the edge from `sources[i]` to `targets[j]`. The nodes are the sources in their order, then each target that is not a source in its order. A name in both lists is one node, so an edge between two targets, such as a supplier of a supplier, is a row of the block. When `sources == targets`, the form is the square form, and `At` keeps the kind of `A`. Otherwise `At` is sparse, because a dense embedding of a block of `N × S` cells costs `(N + S)²` cells.
  - **Edge list.** `edges`, a vector of `(source, target, weight)` tuples, and optionally `nodes`. With `nodes`, each name of an edge is a name of `nodes`, and a node with no edge stays in the graph. With no `nodes`, the nodes are the names in the order of their first use. `At` is sparse.

`ExogenousGraph` is an entry point of integer data. An integer or a `Bool` adjacency converts once to its floating-point type through [`float_if_integer`](@ref), because an edge weight is continuous data.

The positional constructor `ExogenousGraph(nodes, At)` takes the adjacency in its stored layout, with the sources in its columns.

## Validation

  - `nodes`, `sources` and `targets` are not empty, hold no empty name and hold no name twice. Raises an [`IsEmptyError`](@ref) or an `ArgumentError`. See [`assert_graph_names`](@ref).
  - The shape of `A` matches its form. Raises a `DimensionMismatch`.
  - Each edge weight is finite. Raises an [`IsNonFiniteError`](@ref) that names the edge. See [`assert_finite_weights`](@ref).
  - An edge list holds each pair of a source and a target one time. Raises an `ArgumentError` that names the pair.
  - With `nodes`, each name of an edge list is in `nodes`. Raises an `ArgumentError` that names the name.
  - The keywords give one of the three forms. Raises an `ArgumentError` that names the three forms.

# Examples

```jldoctest
julia> g = ExogenousGraph(; nodes = [\"A\", \"B\", \"S\"], A = [0 0 0; 0 0 0; 0.4 0.6 0]);

julia> g.At
3×3 Matrix{Float64}:
 0.0  0.0  0.4
 0.0  0.0  0.6
 0.0  0.0  0.0

julia> adjacency(g)[3, 1]
0.4

julia> h = ExogenousGraph(; sources = [\"S\"], targets = [\"A\", \"B\"], A = [0.4 0.6]);

julia> h.nodes
3-element Vector{String}:
 \"S\"
 \"A\"
 \"B\"

julia> ExogenousGraph(; edges = [(\"S\", \"A\", 0.4), (\"S\", \"B\", 0.6)]) == h
true
```

# Related

  - [`ExogenousGraphSeries`](@ref)
  - [`ExG_ExGS`](@ref)
  - [`adjacency`](@ref)
  - [`graph_in_force`](@ref)
  - [`graph_requirements`](@ref)
  - [`ReturnsResult`](@ref)
"""
@concrete struct ExogenousGraph <: AbstractResult
    """
    Names of the nodes (nodes × 1). Node `k` is row `k` and column `k` of `At`.
    """
    nodes
    """
    Adjacency with the sources in its columns (nodes × nodes): `At[j, i]` is the weight of the edge from node `i` to node `j`. A zero is no edge.
    """
    At
    function ExogenousGraph(nodes::VecStr, At::MatNum)
        nodes = convert(Vector{String}, nodes)
        assert_graph_names(nodes, :nodes)
        n = length(nodes)
        @argcheck(size(At) == (n, n),
                  DimensionMismatch("the adjacency of an Exogenous Graph has one row and one column for each node, so its size must be ($n, $n), got $(size(At))"))
        # An edge weight is continuous data, so an integer adjacency is floated here, once.
        At = drop_stored_zeros(float_if_integer(At))
        assert_finite_weights(nodes, At)
        return new{typeof(nodes), typeof(At)}(nodes, At)
    end
end
function ExogenousGraph(; nodes::Option{<:VecStr} = nothing, A::Option{<:MatNum} = nothing,
                        sources::Option{<:VecStr} = nothing,
                        targets::Option{<:VecStr} = nothing,
                        edges::Option{<:AbstractVector{<:Tuple{AbstractString,
                                                               AbstractString, Number}}} = nothing)::ExogenousGraph
    return exogenous_graph(nodes, A, sources, targets, edges)
end
"""
    exogenous_graph(nodes::VecStr, A::MatNum, ::Nothing, ::Nothing, ::Nothing) -> ExogenousGraph
    exogenous_graph(::Nothing, A::MatNum, sources::VecStr, targets::VecStr, ::Nothing) -> ExogenousGraph
    exogenous_graph(nodes::Option{<:VecStr}, ::Nothing, ::Nothing, ::Nothing, edges::AbstractVector) -> ExogenousGraph
    exogenous_graph(nodes, A, sources, targets, edges)

Build an [`ExogenousGraph`](@ref) from the keywords of its constructor, by the form that the given keywords state.

# Algorithm

The method that Julia selects is the algorithm.

 1. `nodes` and `A`: the square form. Check the names and the shape, and transpose `A` with [`source_columns`](@ref).
 2. `sources`, `targets` and `A`: the block form. When `sources == targets`, it is the square form. Otherwise, take the sources and then each target that is not a source as the nodes, and put each entry of `A` that is not zero into a sparse adjacency at its target row and its source column.
 3. `edges`, and optionally `nodes`: the edge-list form. Refuse a repeated pair. Take `nodes`, or the names in the order of their first use, and put each weight into a sparse adjacency at its target row and its source column.
 4. Any other set of keywords: throw an `ArgumentError` that names the three forms.

# Arguments

  - `nodes`: The names of the nodes, or `nothing`.
  - `A`: The adjacency with the sources in its rows, or `nothing`.
  - `sources`: The names of the rows of a block, or `nothing`.
  - `targets`: The names of the columns of a block, or `nothing`.
  - `edges`: The `(source, target, weight)` tuples of an edge list, or `nothing`.

# Validation

  - The rules of the `# Constructors` section of [`ExogenousGraph`](@ref).

# Returns

  - `g::ExogenousGraph`: The graph.

# Related

  - [`ExogenousGraph`](@ref)
  - [`source_columns`](@ref)
  - [`stored_entries`](@ref)
"""
function exogenous_graph(nodes::VecStr, A::MatNum, ::Nothing, ::Nothing, ::Nothing)
    assert_graph_names(nodes, :nodes)
    n = length(nodes)
    @argcheck(size(A) == (n, n),
              DimensionMismatch("the square form of an Exogenous Graph takes one row and one column of A for each node, so size(A) must be ($n, $n), got $(size(A))"))
    return ExogenousGraph(nodes, source_columns(A))
end
function exogenous_graph(::Nothing, A::MatNum, sources::VecStr, targets::VecStr, ::Nothing)
    assert_graph_names(sources, :sources)
    assert_graph_names(targets, :targets)
    @argcheck(size(A) == (length(sources), length(targets)),
              DimensionMismatch("the block form of an Exogenous Graph takes one row of A for each source and one column for each target, so size(A) must be ($(length(sources)), $(length(targets))), got $(size(A))"))
    if sources == targets
        return ExogenousGraph(sources, source_columns(A))
    end
    nodes = vcat(convert(Vector{String}, sources),
                 String[t for t in targets if !(t in sources)])
    pos = Dict(name => k for (k, name) in pairs(nodes))
    I, J, V = stored_entries(A)
    n = length(nodes)
    return ExogenousGraph(nodes,
                          SparseArrays.sparse([pos[targets[j]] for j in J],
                                              [pos[sources[i]] for i in I], V, n, n))
end
function exogenous_graph(nodes::Option{<:VecStr}, ::Nothing, ::Nothing, ::Nothing,
                         edges::AbstractVector)
    pairs_seen = [(String(e[1]), String(e[2])) for e in edges]
    k = findfirst(i -> pairs_seen[i] in view(pairs_seen, 1:(i - 1)), eachindex(pairs_seen))
    @argcheck(isnothing(k),
              ArgumentError("an edge list holds each pair of a source and a target one time, and the edge $(isnothing(k) ? "" : edge_text(pairs_seen[k])) comes more than one time. Add the two weights into one edge, or keep one of them"))
    nodes = edge_list_nodes(nodes, pairs_seen)
    pos = Dict(name => k for (k, name) in pairs(nodes))
    T = mapreduce(e -> typeof(e[3]), promote_type, edges; init = Bool)
    n = length(nodes)
    return ExogenousGraph(nodes,
                          SparseArrays.sparse(Int[pos[p[2]] for p in pairs_seen],
                                              Int[pos[p[1]] for p in pairs_seen],
                                              T[e[3] for e in edges], n, n))
end
function exogenous_graph(nodes, A, sources, targets, edges)
    given = join((string(k)
                  for (k, v) in
                      ((:nodes, nodes), (:A, A), (:sources, sources), (:targets, targets),
                       (:edges, edges)) if !isnothing(v)), ", ")
    return throw(ArgumentError("an Exogenous Graph takes one of three forms: `nodes` and `A` for a square adjacency, `sources`, `targets` and `A` for a block, or `edges` and optionally `nodes` for an edge list. Got the keywords: $(isempty(given) ? "none" : given)"))
end
"""
    edge_list_nodes(nodes::Nothing, pairs::AbstractVector{<:Tuple{String, String}}) -> Vector{String}
    edge_list_nodes(nodes::VecStr, pairs::AbstractVector{<:Tuple{String, String}}) -> Vector{String}

Return the nodes of an edge list: the given names, or the names of the edges in the order of their first use.

# Algorithm

The method that Julia selects is the algorithm.

 1. `nodes` is `nothing`: walk the pairs, and keep the source and then the target of each pair at its first use.
 2. `nodes` is given: check the names, and check that each name of a pair is in `nodes`.

# Arguments

  - `nodes`: The given names, or `nothing`.
  - `pairs`: The `(source, target)` pairs of the edge list.

# Validation

  - Given `nodes` pass [`assert_graph_names`](@ref), and hold each name of a pair. Raises an `ArgumentError` that names the first name that `nodes` does not hold.

# Returns

  - `nodes::Vector{String}`: The names of the nodes.

# Related

  - [`exogenous_graph`](@ref)
"""
function edge_list_nodes(::Nothing, pairs::AbstractVector{<:Tuple{String, String}})
    nodes = String[]
    seen = Set{String}()
    for (s, t) in pairs, name in (s, t)
        if !(name in seen)
            push!(seen, name)
            push!(nodes, name)
        end
    end
    return nodes
end
function edge_list_nodes(nodes::VecStr, pairs::AbstractVector{<:Tuple{String, String}})
    assert_graph_names(nodes, :nodes)
    known = Set{String}(nodes)
    k = findfirst(p -> !(p[1] in known && p[2] in known), pairs)
    @argcheck(isnothing(k),
              ArgumentError("each name of an edge list must be a node, and the edge $(isnothing(k) ? "" : edge_text(pairs[k])) names a node that `nodes` does not hold"))
    return convert(Vector{String}, nodes)
end
"""
    edge_text(pair::Tuple{AbstractString, AbstractString}) -> String

Write a `(source, target)` pair as `"source" => "target"` in an error message.

# Related

  - [`exogenous_graph`](@ref)
  - [`edge_list_nodes`](@ref)
"""
function edge_text(pair::Tuple{AbstractString, AbstractString})
    return string("\"", pair[1], "\" => \"", pair[2], "\"")
end
"""
    isequal(a::ExogenousGraph, b::ExogenousGraph) -> Bool
    ==(a::ExogenousGraph, b::ExogenousGraph) -> Bool
    hash(g::ExogenousGraph, h::UInt) -> UInt

Compare two [`ExogenousGraph`](@ref)s by value: the same nodes in the same order, and the same adjacency.

A dense adjacency and a sparse adjacency with the same entries are equal, because `==` of two matrices compares their entries. The hash reads the same two values, so two equal graphs have one hash.

# Arguments

  - `a`, `b`, `g`: The graphs.
  - `h`: The hash to mix in.

# Returns

  - `tf::Bool`: `true` when the nodes and the adjacencies are equal. `hash` returns a `UInt`.

# Related

  - [`ExogenousGraph`](@ref)
  - [`graph_vcat`](@ref)
"""
function Base.isequal(a::ExogenousGraph, b::ExogenousGraph)
    return isequal(a.nodes, b.nodes) && isequal(a.At, b.At)
end
function Base.:(==)(a::ExogenousGraph, b::ExogenousGraph)
    return a.nodes == b.nodes && a.At == b.At
end
function Base.hash(g::ExogenousGraph, h::UInt)
    return hash(g.At, hash(g.nodes, hash(:ExogenousGraph, h)))
end
"""
    adjacency(g::ExogenousGraph) -> LinearAlgebra.Transpose

Return the adjacency of `g` with the sources in its rows: `adjacency(g)[i, j]` is the weight of the edge from node `i` to node `j`.

The function returns the lazy transpose of `g.At`, so it copies nothing. Read `g.At` in a kernel that walks the out-edges of a node, because a column of `g.At` is contiguous.

# Arguments

  - `g`: The graph.

# Returns

  - `A::LinearAlgebra.Transpose`: The transpose of `g.At`.

# Examples

```jldoctest
julia> g = ExogenousGraph(; edges = [(\"S\", \"A\", 0.4)]);

julia> adjacency(g)[1, 2]
0.4
```

# Related

  - [`ExogenousGraph`](@ref)
"""
function adjacency(g::ExogenousGraph)
    return transpose(g.At)
end
"""
    Graphs.SimpleDiGraph(g::ExogenousGraph) -> Graphs.SimpleDiGraph{Int}
    SimpleWeightedGraphs.SimpleWeightedDiGraph(g::ExogenousGraph) -> SimpleWeightedGraphs.SimpleWeightedDiGraph

Build the directed graph of Graphs.jl, with no weights or with weights, that `g` states.

Vertex `k` is the node `g.nodes[k]`, and each edge that is not zero is an edge, a negative one too. The unweighted graph keeps only the structure. The weighted graph copies the adjacency into a sparse matrix with the targets in its rows, which is the layout that `SimpleWeightedDiGraph` stores, so it builds the graph with `permute = false` and no transpose. The two graph types are mutable, so the conversion copies, and a change to the graph does not change `g`.

# Arguments

  - `g`: The graph.

# Returns

  - `G`: The graph of Graphs.jl.

# Examples

```jldoctest
julia> g = ExogenousGraph(; edges = [(\"S\", \"A\", 0.4), (\"S\", \"B\", 0.6)]);

julia> D = PortfolioOptimisers.Graphs.SimpleDiGraph(g);

julia> PortfolioOptimisers.Graphs.has_edge(D, 1, 3), PortfolioOptimisers.Graphs.has_edge(D, 3, 1)
(true, false)

julia> W = PortfolioOptimisers.SimpleWeightedGraphs.SimpleWeightedDiGraph(g);

julia> PortfolioOptimisers.Graphs.weights(W)[1, 3]
0.6
```

# Related

  - [`ExogenousGraph`](@ref)
  - [`stored_entries`](@ref)
"""
function Graphs.SimpleDiGraph(g::ExogenousGraph)
    I, J, _ = stored_entries(g.At)
    G = Graphs.SimpleDiGraph(length(g.nodes))
    for (t, s) in zip(I, J)
        Graphs.add_edge!(G, s, t)
    end
    return G
end
function SimpleWeightedGraphs.SimpleWeightedDiGraph(g::ExogenousGraph)
    I, J, V = stored_entries(g.At)
    n = length(g.nodes)
    return SimpleWeightedGraphs.SimpleWeightedDiGraph(SparseArrays.sparse(I, J, V, n, n);
                                                      permute = false)
end

export ExogenousGraph, adjacency
