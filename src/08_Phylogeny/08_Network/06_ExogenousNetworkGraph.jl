"""
    single_direction(dir::AbstractEdgeDirection) -> AbstractEdgeDirection
    single_direction(dir::AnyEdges) -> OutEdges

Return the direction along which an Exogenous Network normalises its weights.

A normalisation and a [`GraphUnion`](@ref) each need a single direction, and [`AnyEdges`](@ref) has none, so they act along the out-edges under it.

# Arguments

  - `dir`: The direction of the network, or one part of it.

# Returns

  - `dir::AbstractEdgeDirection`: [`OutEdges`](@ref) for [`AnyEdges`](@ref), and `dir` otherwise.

# Related

  - [`ExogenousNetwork`](@ref)
  - [`network_steps`](@ref)
"""
function single_direction(dir::AbstractEdgeDirection)
    return dir
end
function single_direction(::AnyEdges)
    return OutEdges()
end
"""
    network_adjacency(scope::AbstractNetworkScope, g::ExogenousGraph, nx::VecStr) -> Tuple{SparseMatrixCSC, Vector{Int}}

Return the adjacency of an Exogenous Graph on the vertices of a scope, and the vertex of each asset.

# Algorithm

 1. Place the nodes and the assets on the vertices with [`scope_vertices`](@ref).
 2. Keep each edge that is not a self-loop and whose two nodes have a vertex.
 3. Build the adjacency of the kept edges, with the sources in its columns.

# Arguments

  - `scope`: The scope.
  - `g`: The graph in force.
  - `nx`: The asset names.

# Returns

  - `At::SparseMatrixCSC`: The adjacency, with the sources in its columns.
  - `va::Vector{Int}`: The vertex of each asset.

# Related

  - [`ExogenousNetwork`](@ref)
  - [`exogenous_network_graph`](@ref)
"""
function network_adjacency(scope::AbstractNetworkScope, g::ExogenousGraph, nx::VecStr)
    n, vn, va = scope_vertices(scope, g.nodes, nx)
    I, J, V = stored_entries(g.At)
    keep = findall(k -> I[k] != J[k] && vn[I[k]] > 0 && vn[J[k]] > 0, eachindex(I, J))
    return SparseArrays.sparse(vn[I[keep]], vn[J[keep]], V[keep], n, n), va
end
"""
    network_steps(nte::ExogenousNetwork, d::AbstractEdgeDirection, At::MatNum) -> MatNum

Return the normalised weights of the steps along one part of the direction, with the start of each step in its columns.

# Algorithm

 1. Orient the adjacency along [`single_direction`](@ref) with [`directed_weights`](@ref).
 2. Apply `nte.nrm` with [`normalise_edge_weights`](@ref).

# Arguments

  - $(arg_dict[:nte])
  - `d`: The part of the direction.
  - `At`: The adjacency, with the sources in its columns.

# Returns

  - `M::MatNum`: The weights of the steps.

# Related

  - [`network_lengths`](@ref)
  - [`network_strengths`](@ref)
"""
function network_steps(nte::ExogenousNetwork, d::AbstractEdgeDirection, At::MatNum)
    return normalise_edge_weights(nte.nrm, directed_weights(single_direction(d), At))
end
"""
    step_graph(d::AbstractEdgeDirection, I, J, V, n::Integer, op) -> SimpleWeightedGraphs.SimpleWeightedDiGraph
    step_graph(d::AnyEdges, I, J, V, n::Integer, op) -> SimpleWeightedGraphs.SimpleWeightedGraph

Build the weighted graph of the steps of one part of a direction.

# Algorithm

The method that Julia selects is the algorithm.

 1. A single direction: the directed graph with the weight `V[k]` on the edge from `J[k]` to `I[k]`.
 2. [`AnyEdges`](@ref): the undirected graph. `op` combines the weights of two opposite steps.

# Arguments

  - `d`: The part of the direction.
  - `I`, `J`, `V`: The end, the start and the weight of each step.
  - `n`: The number of vertices.
  - `op`: The function that combines the weights of two opposite steps.

# Returns

  - `g`: The weighted graph.

# Related

  - [`network_lengths`](@ref)
  - [`network_strengths`](@ref)
"""
function step_graph(::AbstractEdgeDirection, I, J, V, n::Integer, ::Any)
    return SimpleWeightedGraphs.SimpleWeightedDiGraph(SparseArrays.sparse(I, J, V, n, n);
                                                      permute = false)
end
function step_graph(::AnyEdges, I, J, V, n::Integer, op)
    return SimpleWeightedGraphs.SimpleWeightedGraph(SparseArrays.sparse([I; J], [J; I],
                                                                        [V; V], n, n, op))
end
"""
    network_hops(nte::ExogenousNetwork, d::AbstractEdgeDirection, At::MatNum) -> Graphs.AbstractGraph
    network_hops(nte::ExogenousNetwork, d::AnyEdges, At::MatNum) -> Graphs.SimpleGraph

Build the graph over which an Exogenous Network counts hops along one part of its direction.

# Algorithm

The method that Julia selects is the algorithm.

 1. A single direction: the directed graph of the steps, from [`walk_graph`](@ref).
 2. [`AnyEdges`](@ref): the undirected graph of the same edges.

# Arguments

  - $(arg_dict[:nte])
  - `d`: The part of the direction.
  - `At`: The adjacency, with the sources in its columns.

# Returns

  - `g::Graphs.AbstractGraph`: A `Graphs.SimpleDiGraph`, or a `Graphs.SimpleGraph` under [`AnyEdges`](@ref).

# Related

  - [`ExogenousNetwork`](@ref)
  - [`separation_graph`](@ref)
"""
function network_hops(::ExogenousNetwork, d::AbstractEdgeDirection, At::MatNum)
    return walk_graph(directed_weights(d, At))
end
function network_hops(::ExogenousNetwork, ::AnyEdges, At::MatNum)
    return Graphs.SimpleGraph(walk_graph(At))
end
"""
    network_lengths(nte::ExogenousNetwork, d::AbstractEdgeDirection, At::MatNum) -> Graphs.AbstractGraph

Build the graph of the lengths of an Exogenous Network along one part of its direction.

# Algorithm

 1. Read the weights of the steps with [`network_steps`](@ref).
 2. Turn each weight into a length with [`edge_length`](@ref). A zero length moves to the smallest positive value of its type, because a weighted graph reads a zero as no edge.
 3. Build the graph with [`step_graph`](@ref). Under [`AnyEdges`](@ref), two opposite steps give the shorter of their two lengths.

# Arguments

  - $(arg_dict[:nte])
  - `d`: The part of the direction.
  - `At`: The adjacency, with the sources in its columns.

# Validation

  - The rules of [`edge_length`](@ref).

# Returns

  - `g::Graphs.AbstractGraph`: The weighted graph.

# Related

  - [`network_strengths`](@ref)
  - [`graph_weight_matrix`](@ref)
"""
function network_lengths(nte::ExogenousNetwork, d::AbstractEdgeDirection, At::MatNum)
    I, J, W = stored_entries(network_steps(nte, d, At))
    V = map(W) do w
        l = edge_length(nte.len, w)
        return iszero(l) ? nextfloat(zero(l)) : l
    end
    return step_graph(d, I, J, V, size(At, 1), min)
end
"""
    network_strengths(nte::ExogenousNetwork, d::AbstractEdgeDirection, At::MatNum) -> Graphs.AbstractGraph

Build the graph of the strengths of an Exogenous Network along one part of its direction.

# Algorithm

 1. Read the weights of the steps with [`network_steps`](@ref).
 2. Build the graph with [`step_graph`](@ref). Under [`AnyEdges`](@ref), two opposite steps give the sum of their two weights.

# Arguments

  - $(arg_dict[:nte])
  - `d`: The part of the direction.
  - `At`: The adjacency, with the sources in its columns.

# Returns

  - `g::Graphs.AbstractGraph`: The weighted graph.

# Related

  - [`network_lengths`](@ref)
  - [`centrality_graph`](@ref)
"""
function network_strengths(nte::ExogenousNetwork, d::AbstractEdgeDirection, At::MatNum)
    I, J, V = stored_entries(network_steps(nte, d, At))
    return step_graph(d, I, J, V, size(At, 1), +)
end
"""
    network_unplaced_assets(nte::ExogenousNetwork, nodes::VecStr, nx::VecStr, gname::AbstractString) -> nothing

Apply the rule of an asset that is not a node of the graph in force: refuse it under `strict`, and warn one time otherwise.

# Arguments

  - $(arg_dict[:nte])
  - `nodes`: The names of the nodes of the graph in force.
  - `nx`: The asset names.
  - `gname`: The name of the graph.

# Validation

  - Under `nte.strict`, each asset is a node. Raises an `ArgumentError` that names the assets and the graph.

# Related

  - [`ExogenousNetwork`](@ref)
"""
function network_unplaced_assets(nte::ExogenousNetwork, nodes::VecStr, nx::VecStr,
                                 gname::AbstractString)::Nothing
    lost = walk_positions(nodes, nx) .== 0
    if !any(lost)
        return nothing
    end
    names = nx[lost]
    msg = "the assets $(pinned_repr(first(names, 5)))$(length(names) > 5 ? " and $(length(names) - 5) more" : "") are not nodes of the Exogenous Graph \"$gname\" in force at the last row"
    @argcheck(!nte.strict,
              ArgumentError("$msg. Add them to the graph, or pass `strict = false`"))
    @warn("$msg, so each is an isolated vertex of the ExogenousNetwork and stays investable. Pass `strict = true` to refuse them.",
          maxlog = log_limit(:warn), _id = @log_id())
    return nothing
end
"""
    exogenous_network_graph(f, nte::ExogenousNetwork, X::MatNum; dims::Int = 1, pr = nothing,
                            rd::Option{<:ReturnsResult} = nothing, kwargs...) -> AssetVertexGraph

Build the graphs of an Exogenous Network from the graph in force at the last row of the returns data.

# Algorithm

 1. Read the returns data with [`consumer_returns`](@ref) and the asset names with [`consumer_asset_names`](@ref). Check that `X` has one series for each asset.
 2. Read the graph of each part of `nte.dir` with [`consumer_graphs`](@ref).
 3. Apply the rule of an asset that is not a node with [`network_unplaced_assets`](@ref).
 4. Place each graph on the vertices of `nte.scope` with [`network_adjacency`](@ref), and build the graph of each part with `f`.

# Arguments

  - `f`: The function that builds the graph of one part: [`network_hops`](@ref), [`network_lengths`](@ref) or [`network_strengths`](@ref).
  - $(arg_dict[:nte])
  - `X`: Returns matrix. The function reads its size alone.
  - $(arg_dict[:dims])
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd])
  - `kwargs...`: Keyword arguments that the function does not read.

# Validation

  - The returns data, its asset names and the graph in force, see [`consumer_returns`](@ref), [`consumer_asset_names`](@ref) and [`consumer_graph`](@ref).
  - `X` has one series for each asset. Raises a `DimensionMismatch`.

# Returns

  - `ag::AssetVertexGraph`: The graphs, with the vertices of the assets.

# Related

  - [`ExogenousNetwork`](@ref)
  - [`separation_graph`](@ref)
  - [`centrality_graph`](@ref)
"""
function exogenous_network_graph(f::F, nte::ExogenousNetwork, X::MatNum; dims::Int = 1,
                                 pr = nothing, rd::Option{<:ReturnsResult} = nothing,
                                 kwargs...) where {F}
    assert_dims(dims)
    r = consumer_returns(nte, pr, rd)
    nx = consumer_asset_names(nte, r)
    @argcheck(size(X, 3 - dims) == length(nx),
              DimensionMismatch("ExogenousNetwork reads one series of X for each asset of the returns data, got $(size(X, 3 - dims)) series and $(length(nx)) asset names"))
    gs, gname = consumer_graphs(nte, r, nte.ng, single_direction(nte.dir))
    network_unplaced_assets(nte, first(gs).nodes, nx, gname)
    parts = map(g -> network_adjacency(nte.scope, g, nx), gs)
    return AssetVertexGraph(map((d, p) -> f(nte, d, first(p)), exposure_parts(nte.dir),
                                parts), last(first(parts)), symmetric_direction(nte.dir))
end
# The methods below are the kernels of the separation family over an Exogenous Network. Their
# docstrings are on the functions of `04_Separation.jl`.
function separation_graph(::HopCount, nte::ExogenousNetwork, X::MatNum; dims::Int = 1,
                          kwargs...)
    return exogenous_network_graph(network_hops, nte, X; dims = dims, kwargs...)
end
function separation_graph(::PathLength, nte::ExogenousNetwork, X::MatNum; dims::Int = 1,
                          kwargs...)
    return exogenous_network_graph(network_lengths, nte, X; dims = dims, kwargs...)
end
"""
    vertex_separations(sep::HopCount, g::Graphs.AbstractGraph, v::Integer) -> Vector
    vertex_separations(sep::PathLength, g::Graphs.AbstractGraph, v::Integer) -> Vector

Measure the separations from one vertex of a graph to each vertex.

An extension of [`AbstractSeparationAlgorithm`](@ref) implements it to measure over an [`ExogenousNetwork`](@ref), whose graph can have more vertices than the assets.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`HopCount`](@ref): one breadth-first traversal, with `Graphs.gdistances`.
 2. [`PathLength`](@ref): the shortest paths of [dijkstra1959](@cite), with `Graphs.dijkstra_shortest_paths`.

# Arguments

  - `sep`: Separation algorithm. The function does not read its budget.
  - `g`: The graph, with lengths as weights under a [`PathLength`](@ref).
  - `v`: The vertex.

# Returns

  - `d::Vector`: The separation to each vertex. An unreachable vertex keeps the sentinel of the routine, `typemax` of the element type.

# Related

  - [`separation_matrix`](@ref)
  - [`is_reachable`](@ref)
  - [`AssetVertexGraph`](@ref)

# References

  - $(ref_dict[:dijkstra1959])
"""
function vertex_separations(::HopCount, g::Graphs.AbstractGraph, v::Integer)
    return Graphs.gdistances(g, v)
end
function vertex_separations(::PathLength, g::Graphs.AbstractGraph, v::Integer)
    return Graphs.dijkstra_shortest_paths(g, v).dists
end
function separation_matrix(sep::AbstractSeparationAlgorithm, ag::AssetVertexGraph)
    # Row `i` holds the separations from asset `i`, so it is the profile of the asset.
    D = mapreduce((a, b) -> min.(a, b), ag.gs) do g
        return stack(v -> view(vertex_separations(sep, g, v), ag.va), ag.va; dims = 1)
    end
    # The sum along a path depends on the end where it starts, so a symmetric separation takes
    # the smaller of the two sums. The two directions of `BothEdges` meet here too.
    return ag.sym ? min.(D, transpose(D)) : D
end
"""
    closure_graph(sep::HopCount, nte::AbstractNetworkEstimator, ag::AssetVertexGraph) -> Graphs.AbstractGraph
    closure_graph(sep::AbstractSeparationAlgorithm, nte::AbstractNetworkEstimator, ag::AssetVertexGraph) -> Graphs.AbstractGraph

Build the graph of the pairs of vertices that a separation relates, over the whole graph of the scope.

A centrality that reads no weight reads this graph, as it reads the [`phylogeny_matrix`](@ref) of a [`NetworkEstimator`](@ref).

# Algorithm

The method that Julia selects is the algorithm.

 1. [`HopCount`](@ref): return the graph itself at one hop, because it is its own closure. Otherwise relate each pair within the hop budget with [`ball_graph`](@ref).
 2. Any other separation: read the budget from the separations of the assets with [`separation_budget`](@ref), and relate each pair within it with [`ball_graph`](@ref).

# Arguments

  - `sep`: Resolved separation algorithm.
  - $(arg_dict[:nte])
  - `ag`: The graph of one part, from [`separation_graph`](@ref).

# Returns

  - `g::Graphs.AbstractGraph`: The plain graph of the related pairs, directed when the graph of `ag` is.

# Related

  - [`centrality_graph`](@ref)
  - [`ball_graph`](@ref)
"""
function closure_graph(sep::HopCount, ::AbstractNetworkEstimator, ag::AssetVertexGraph)
    g = only(ag.gs)
    return isone(sep.n.val) ? g : ball_graph(sep, g, sep.n.val)
end
function closure_graph(sep::AbstractSeparationAlgorithm, nte::AbstractNetworkEstimator,
                       ag::AssetVertexGraph)
    return ball_graph(sep, only(ag.gs),
                      separation_budget(sep, nte, separation_matrix(sep, ag)))
end
"""
    ball_graph(sep::AbstractSeparationAlgorithm, g::Graphs.AbstractGraph, dmax::Number) -> Graphs.AbstractGraph

Build the plain graph with an edge from each vertex to each other vertex within a separation budget.

# Algorithm

 1. Start from a graph with the vertices of `g` and no edge, directed when `g` is.
 2. Measure the separations from each vertex with [`vertex_separations`](@ref), and add an edge to each other vertex that [`is_related`](@ref) admits at `dmax`.

# Arguments

  - `sep`: Separation algorithm.
  - `g`: The graph to measure over.
  - `dmax`: Separation budget.

# Returns

  - `b::Graphs.AbstractGraph`: A `Graphs.SimpleDiGraph` when `g` is directed, and a `Graphs.SimpleGraph` otherwise.

# Related

  - [`closure_graph`](@ref)
  - [`is_related`](@ref)
"""
function ball_graph(sep::AbstractSeparationAlgorithm, g::Graphs.AbstractGraph, dmax::Number)
    n = Graphs.nv(g)
    b = Graphs.is_directed(g) ? Graphs.SimpleDiGraph(n) : Graphs.SimpleGraph(n)
    for v in Graphs.vertices(g)
        for (u, d) in enumerate(vertex_separations(sep, g, v))
            if u != v && is_related(sep, d, dmax)
                Graphs.add_edge!(b, v, u)
            end
        end
    end
    return b
end

public vertex_separations
