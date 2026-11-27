"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for a property of an Exogenous Graph that a consumer of the graph needs.

An [`ExogenousGraph`](@ref) accepts a negative weight, a self-loop and a cycle, because each has a use. A consumer that cannot read one of them states it with [`graph_requirements`](@ref), and the library checks every snapshot of the graph that the consumer names at the entry of the user verb, before the first fold. So a bad snapshot that fold 80 reads fails before fold 1.

# Interfaces

To add a requirement, subtype `AbstractGraphRequirement` and implement the following two methods.

## `graph_requirement_breach`

  - `graph_requirement_breach(req::MyGraphRequirement, g::ExogenousGraph) -> Option{<:Tuple{Integer, Integer}}`: Return the positions of the source and the target of the first edge of `g` that breaks the requirement, or `nothing` when no edge breaks it.

### Arguments

  - `req`: The requirement.
  - `g`: The graph.

### Returns

  - `edge::Option{<:Tuple{Integer, Integer}}`: The `(source, target)` positions in `g.nodes`, or `nothing`.

## `graph_requirement_rule`

  - `graph_requirement_rule(req::MyGraphRequirement) -> String`: Return the rule as a clause that an error message quotes, such as `"each edge weight must be non-negative"`.

### Arguments

  - `req`: The requirement.

### Returns

  - `rule::String`: The rule.

# Related

  - [`NonNegativeWeights`](@ref)
  - [`graph_requirements`](@ref)
  - [`assert_graph_requirements`](@ref)
"""
abstract type AbstractGraphRequirement <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Requires that each edge weight of an Exogenous Graph is non-negative.

A walk that divides a weight by the total weight of a node, and a count of hops, cannot read a negative weight, which states a hedge or a competitor. A self-loop and a cycle pass the requirement.

# Examples

```jldoctest
julia> g = ExogenousGraph(; edges = [(\"S\", \"A\", 0.4), (\"S\", \"B\", -0.1)]);

julia> PortfolioOptimisers.graph_requirement_breach(NonNegativeWeights(), g)
(1, 3)
```

# Related

  - [`AbstractGraphRequirement`](@ref)
  - [`graph_requirements`](@ref)
"""
struct NonNegativeWeights <: AbstractGraphRequirement end
"""
    graph_requirement_breach(req::NonNegativeWeights, g::ExogenousGraph) -> Option{Tuple{Int, Int}}

Return the positions of the source and the target of the first edge of `g` with a negative weight, or `nothing`.

The function reads the edges with [`stored_entries`](@ref), in the order of the sources.

# Arguments

  - `req`: The requirement.
  - `g`: The graph.

# Returns

  - `edge::Option{Tuple{Int, Int}}`: The `(source, target)` positions in `g.nodes`, or `nothing`.

# Related

  - [`NonNegativeWeights`](@ref)
  - [`AbstractGraphRequirement`](@ref)
"""
function graph_requirement_breach(::NonNegativeWeights, g::ExogenousGraph)
    I, J, V = stored_entries(g.At)
    k = findfirst(<(zero(eltype(V))), V)
    return isnothing(k) ? nothing : (J[k], I[k])
end
"""
    graph_requirement_rule(req::NonNegativeWeights) -> String

Return the rule of a requirement as a clause of an error message.

# Related

  - [`NonNegativeWeights`](@ref)
  - [`AbstractGraphRequirement`](@ref)
"""
function graph_requirement_rule(::NonNegativeWeights)
    return "each edge weight must be non-negative"
end
"""
    graph_requirements(consumer) -> Tuple

Return the requirements that a consumer of an Exogenous Graph states for each snapshot that it reads, as a tuple of [`AbstractGraphRequirement`](@ref)s.

The fallback returns `()`: a value that reads no Exogenous Graph needs nothing of one. A consumer of a graph returns its requirements, for example `(NonNegativeWeights(),)`, and implements [`graph_names`](@ref), which names the graphs that it reads. The entry of a user verb, such as `optimise`, a cross-validation or a prior, walks the estimator tree one time, and checks each snapshot of each graph that each consumer names with [`assert_graph_entry`](@ref). A direct call of the consumer checks the one snapshot that it reads, with [`consumer_graph`](@ref).

# Arguments

  - `consumer`: Any value of an estimator tree.

# Returns

  - `reqs::Tuple`: The requirements of the consumer, empty for a value that reads no graph.

# Related

  - [`AbstractGraphRequirement`](@ref)
  - [`graph_names`](@ref)
  - [`assert_graph_entry`](@ref)
  - [`consumer_graph`](@ref)
"""
function graph_requirements(::Any)
    return ()
end
"""
    graph_names(consumer) -> Union{Nothing, AbstractString, VecStr}

Return the names of the Exogenous Graphs that a consumer reads: `nothing` for the only graph of the returns data, one name, or a vector of names.

A consumer that returns requirements from [`graph_requirements`](@ref) implements this function, usually as the value of its field that names its graphs. The fallback refuses, so a consumer that states requirements and names no graph fails at the entry of the user verb with a message that names its type.

# Arguments

  - `consumer`: The consumer of a graph.

# Validation

  - The fallback raises an `ArgumentError` that names the type of `consumer`.

# Returns

  - `names`: The names of the graphs that the consumer reads.

# Related

  - [`graph_requirements`](@ref)
  - [`graph_position`](@ref)
"""
function graph_names(consumer)
    return throw(ArgumentError("the $(nameof(typeof(consumer))) states the requirements $(graph_requirements(consumer)) of a consumer of an Exogenous Graph, and it does not name the graphs that it reads. Implement `PortfolioOptimisers.graph_names(::$(nameof(typeof(consumer))))`"))
end
"""
    graph_name_list(names::Nothing) -> Tuple{Nothing}
    graph_name_list(names::AbstractString) -> Tuple{AbstractString}
    graph_name_list(names::GraphUnion) -> Tuple{GraphUnion}
    graph_name_list(names::AbstractVector) -> AbstractVector

Return the entries of [`graph_names`](@ref) as a collection to walk.

An entry is `nothing` for the only graph, a name, or a [`GraphUnion`](@ref). A vector holds names and unions.

# Related

  - [`assert_graph_entry`](@ref)
"""
function graph_name_list(::Nothing)
    return (nothing,)
end
function graph_name_list(name::AbstractString)
    return (name,)
end
function graph_name_list(names::AbstractVector)
    return names
end
"""
    graph_snapshot_list(g::ExogenousGraph) -> Vector{Tuple{Nothing, ExogenousGraph}}
    graph_snapshot_list(s::ExogenousGraphSeries) -> Vector{Tuple}

Return the `(key, graph)` pairs of each snapshot that holds a graph: the graph and `nothing` for a static graph, and each key of a series that does not hold `nothing`.

# Related

  - [`assert_graph_entry`](@ref)
"""
function graph_snapshot_list(g::ExogenousGraph)
    return [(nothing, g)]
end
function graph_snapshot_list(s::ExogenousGraphSeries)
    return [(s.ts[k], s.graphs[k])
            for k in eachindex(s.ts, s.graphs) if !isnothing(s.graphs[k])]
end
"""
    assert_graph_requirements(consumer, g::ExogenousGraph, name::AbstractString, key, path::AbstractString) -> nothing

Check one snapshot of a graph against each requirement of a consumer.

# Arguments

  - `consumer`: The consumer of the graph.
  - `g`: The snapshot.
  - `name`: The name of the graph.
  - `key`: The key of the snapshot, or `nothing` for a static graph.
  - `path`: The path of the consumer in the estimator tree, which the message quotes.

# Validation

  - No edge of `g` breaks a requirement of [`graph_requirements`](@ref). Raises an `ArgumentError` that names the consumer, the graph, the key, the rule and the edge.

# Returns

  - `nothing`.

# Related

  - [`graph_requirements`](@ref)
  - [`graph_requirement_breach`](@ref)
  - [`assert_graph_entry`](@ref)
"""
function assert_graph_requirements(consumer, g::ExogenousGraph, name::AbstractString, key,
                                   path::AbstractString)::Nothing
    for req in graph_requirements(consumer)
        e = graph_requirement_breach(req, g)
        if isnothing(e)
            continue
        end
        throw(ArgumentError("the $(nameof(typeof(consumer)))$(isempty(path) ? "" : " at `$path`") cannot read the Exogenous Graph \"$name\"$(isnothing(key) ? "" : " at the key $key"): $(graph_requirement_rule(req)), and the edge \"$(g.nodes[e[1]])\" => \"$(g.nodes[e[2]])\" has the weight $(g.At[e[2], e[1]])"))
    end
    return nothing
end
"""
    assert_consumer_graphs(consumer, reqs::Tuple{}, ng, G, path::AbstractString) -> nothing
    assert_consumer_graphs(consumer, reqs::Tuple, ng, G, path::AbstractString) -> nothing

Check each snapshot of each graph that one consumer names.

# Algorithm

The method that Julia selects is the algorithm.

 1. No requirement: return `nothing`. The value reads no graph.
 2. Otherwise: check each entry of [`graph_names`](@ref) with [`assert_consumer_entry`](@ref).

# Validation

  - The rules of [`assert_consumer_entry`](@ref).

# Related

  - [`assert_graph_entry`](@ref)
"""
function assert_consumer_graphs(::Any, ::Tuple{}, ::Any, ::Any, ::AbstractString)::Nothing
    return nothing
end
function assert_consumer_graphs(consumer, ::Tuple, ng, G, path::AbstractString)::Nothing
    for entry in graph_name_list(graph_names(consumer))
        assert_consumer_entry(consumer, entry, ng, G, path)
    end
    return nothing
end
"""
    assert_consumer_entry(consumer, name::Option{<:AbstractString}, ng, G, path::AbstractString) -> nothing
    assert_consumer_entry(consumer, gu::GraphUnion, ng, G, path::AbstractString) -> nothing

Check each snapshot of one graph that a consumer names, or of each part of a union of graphs.

# Algorithm

The method that Julia selects is the algorithm.

 1. A name, or `nothing` for the only graph: find the graph with [`graph_position`](@ref), and check each of its snapshots with [`assert_graph_requirements`](@ref).
 2. A [`GraphUnion`](@ref): check each part by step 1, and check that the keys of the parts are of one kind with [`assert_union_key_kinds`](@ref).

# Arguments

  - `consumer`: The consumer of the graph.
  - `name`, `gu`: The entry of the consumer.
  - `ng`: The names of the graphs of the returns data, or `nothing`.
  - `G`: The graphs, or `nothing`.
  - `path`: The path of the consumer in the estimator tree, which the message quotes.

# Validation

  - The rules of [`graph_position`](@ref), [`assert_graph_requirements`](@ref) and [`assert_union_key_kinds`](@ref).

# Returns

  - `nothing`.

# Related

  - [`assert_consumer_graphs`](@ref)
  - [`GraphUnion`](@ref)
"""
function assert_consumer_entry(consumer, name::Option{<:AbstractString}, ng, G,
                               path::AbstractString)::Nothing
    k = graph_position(ng, name)
    for (key, g) in graph_snapshot_list(G[k])
        assert_graph_requirements(consumer, g, ng[k], key, path)
    end
    return nothing
end
"""
    graph_walk_path(path::AbstractString, f) -> String

Join the path of a value in an estimator tree and the name of one of its fields.

# Related

  - [`assert_graph_walk`](@ref)
"""
function graph_walk_path(path::AbstractString, f)
    return isempty(path) ? string(f) : string(path, ".", f)
end
"""
    assert_graph_walk(est::Union{AbstractEstimator, AbstractAlgorithm, StatsBase.CovarianceEstimator}, ng, G, path::AbstractString) -> nothing
    assert_graph_walk(v::AbstractArray{<:Number}, ng, G, path::AbstractString) -> nothing
    assert_graph_walk(v::AbstractVector{<:Number}, ng, G, path::AbstractString) -> nothing
    assert_graph_walk(v::AbstractVector, ng, G, path::AbstractString) -> nothing
    assert_graph_walk(v::Tuple, ng, G, path::AbstractString) -> nothing
    assert_graph_walk(p::Pair, ng, G, path::AbstractString) -> nothing
    assert_graph_walk(x, ng, G, path::AbstractString) -> nothing

Walk an estimator tree, and check the graphs that each consumer of a graph names.

The walk reads every field, because a consumer can stand in an estimator that a vector or a tuple holds. A Result, a number and an array of numbers hold no consumer, so the walk stops at them.

# Algorithm

The method that Julia selects is the algorithm.

 1. An estimator, an algorithm or a covariance estimator: check its graphs with [`assert_consumer_graphs`](@ref), then walk each field.
 2. An array of numbers: return `nothing`.
 3. A vector or a tuple: walk each element. An undefined element of a vector, such as a row of a log that a callable fills as it runs, holds no consumer.
 4. A Pair: walk its value.
 5. Any other value: return `nothing`.

# Arguments

  - `est`, `v`, `p`, `x`: The value of the tree.
  - `ng`: The names of the graphs of the returns data, or `nothing`.
  - `G`: The graphs, or `nothing`.
  - `path`: The path of the value in the tree, empty at the root.

# Validation

  - The rules of [`assert_consumer_graphs`](@ref) for each consumer.

# Returns

  - `nothing`.

# Related

  - [`assert_graph_entry`](@ref)
  - [`graph_requirements`](@ref)
"""
function assert_graph_walk(est::Union{AbstractEstimator, AbstractAlgorithm,
                                      StatsBase.CovarianceEstimator}, ng, G,
                           path::AbstractString)::Nothing
    assert_consumer_graphs(est, graph_requirements(est), ng, G, path)
    for f in fieldnames(typeof(est))
        assert_graph_walk(getfield(est, f), ng, G, graph_walk_path(path, f))
    end
    return nothing
end
function assert_graph_walk(::AbstractArray{<:Number}, ::Any, ::Any,
                           ::AbstractString)::Nothing
    return nothing
end
function assert_graph_walk(::AbstractVector{<:Number}, ::Any, ::Any,
                           ::AbstractString)::Nothing
    return nothing
end
function assert_graph_walk(v::AbstractVector, ng, G, path::AbstractString)::Nothing
    for i in eachindex(v)
        if isassigned(v, i)
            assert_graph_walk(v[i], ng, G, string(path, "[", i, "]"))
        end
    end
    return nothing
end
function assert_graph_walk(v::Tuple, ng, G, path::AbstractString)::Nothing
    for (i, x) in pairs(v)
        assert_graph_walk(x, ng, G, string(path, "[", i, "]"))
    end
    return nothing
end
function assert_graph_walk(p::Pair, ng, G, path::AbstractString)::Nothing
    return assert_graph_walk(p.second, ng, G, path)
end
function assert_graph_walk(::Any, ::Any, ::Any, ::AbstractString)::Nothing
    return nothing
end
"""
    assert_graph_entry(est, ng::Option{<:VecStr}, G::Option{<:AbstractVector}) -> nothing
    assert_graph_entry(est, rd::ReturnsResult, args...) -> nothing
    assert_graph_entry(est, pr::PricesResult, args...) -> nothing
    assert_graph_entry(est, args...) -> nothing

Check, at the entry of a user verb, every snapshot of each Exogenous Graph that a consumer in the tree of `est` names.

A consumer that cannot read a negative weight or a self-loop states it with [`graph_requirements`](@ref). A check at each fold comes late, because a fold is a view that drops the snapshots of its future, so fold 1 cannot see the bad snapshot that fold 80 reads. So `optimise`, the fold loop of a cross-validation and `prior` call this function one time, before the first view or fold, and it checks every snapshot of the full returns data. The walk also refuses a consumer that names a graph that the returns data does not hold.

# Algorithm

The method that Julia selects is the algorithm.

 1. Names and graphs: walk the tree of `est` with [`assert_graph_walk`](@ref).
 2. A [`ReturnsResult`](@ref) or a [`PricesResult`](@ref): call step 1 with its `ng` and `G`.
 3. Any other data: return `nothing`. It holds no Exogenous Graph.

# Arguments

  - `est`: The estimator of the user verb.
  - `ng`: The names of the graphs, or `nothing`.
  - `G`: The graphs, or `nothing`.
  - $(arg_dict[:rd])
  - `pr`: The [`PricesResult`](@ref).
  - `args`: The other arguments of the user verb, which the function does not read.

# Validation

  - The rules of [`assert_consumer_graphs`](@ref) for each consumer of the tree.

# Returns

  - `nothing`.

# Related

  - [`graph_requirements`](@ref)
  - [`assert_graph_walk`](@ref)
  - [`consumer_graph`](@ref)
"""
function assert_graph_entry(est, ng::Option{<:VecStr}, G::Option{<:AbstractVector})::Nothing
    return assert_graph_walk(est, ng, G, "")
end
function assert_graph_entry(::Any, ::Any...)::Nothing
    return nothing
end
"""
    consumer_graph(consumer, ng, G, name, T, ts) -> ExogenousGraph
    consumer_graph(consumer, ng, G, gu::GraphUnion, T, ts, dir::AbstractEdgeDirection) -> ExogenousGraph
    consumer_graph(consumer, rd::ReturnsResult, name = graph_names(consumer)) -> ExogenousGraph
    consumer_graph(consumer, rd::ReturnsResult, gu::GraphUnion, dir::AbstractEdgeDirection) -> ExogenousGraph

Return the Exogenous Graph `name` in force at the last row of the returns data, after a check against the requirements of `consumer`.

A consumer calls this function in its fit. It reads one snapshot, so a direct call of the consumer, with no user verb around it, checks that snapshot, as the entry of the user verb checks each snapshot. A [`GraphUnion`](@ref) also takes the single direction `dir` of the consumer, along which it normalises its parts. A name takes no direction.

# Algorithm

The method that Julia selects is the algorithm.

 1. A name, or `nothing`: find the graph with [`graph_position`](@ref), and the snapshot at the last row with [`graph_snapshot`](@ref). Check it with [`assert_graph_requirements`](@ref).
 2. A [`GraphUnion`](@ref): read the snapshot of each part in force at the last row, check each with [`assert_graph_requirements`](@ref), choose the parts under the rule `gu.unknown`, and build their union along `dir`, with [`union_in_force`](@ref).

# Arguments

  - `consumer`: The consumer of the graph.
  - `ng`: The names of the graphs, or `nothing`.
  - `G`: The graphs, or `nothing`.
  - `name`: The name of the graph, or `nothing` for the only graph.
  - `gu`: The union of graphs.
  - `T`: The number of rows.
  - `ts`: The timestamps of the rows, or `nothing`.
  - `dir`: The single direction of the consumer, [`OutEdges`](@ref) or [`InEdges`](@ref).
  - $(arg_dict[:rd])

# Validation

  - The rules of [`graph_position`](@ref), [`graph_snapshot`](@ref) and [`assert_graph_requirements`](@ref).
  - A union refuses a row at which its rule finds no graph in force. Raises an `ArgumentError` that names each part with no graph in force. See [`union_in_force`](@ref).

# Returns

  - `g::ExogenousGraph`: The graph in force.

# Related

  - [`graph_in_force`](@ref)
  - [`graph_requirements`](@ref)
  - [`assert_graph_entry`](@ref)
"""
function consumer_graph(consumer, ng::Option{<:VecStr}, G::Option{<:AbstractVector}, name,
                        T, ts)
    k = graph_position(ng, name)
    g, key = graph_snapshot(G[k], ng[k], T, ts)
    assert_graph_requirements(consumer, g, ng[k], key, "")
    return g
end

export NonNegativeWeights
public AbstractGraphRequirement, graph_requirements, graph_names, graph_requirement_breach,
       graph_requirement_rule, consumer_graph, assert_graph_entry
