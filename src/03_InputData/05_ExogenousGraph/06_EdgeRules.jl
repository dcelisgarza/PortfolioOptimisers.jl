"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the direction along which a walk reads the edges of an Exogenous Graph.

An edge of an [`ExogenousGraph`](@ref) goes from a source to a target. A walk along the out-edges of an asset reaches what the asset depends on, and a walk along its in-edges reaches what depends on the asset. [`WalkProfilePanel`](@ref) holds a member in its `dir` field. [`graph_union`](@ref) takes a single direction, along which it normalises each part.

# Interfaces

To add a direction, subtype `AbstractEdgeDirection` and implement the following methods.

## `walk_parts`

  - `walk_parts(dir::MyEdgeDirection) -> Tuple`: Return the single directions that the member walks, one Panel Field each, as a tuple of [`OutEdges`](@ref) and [`InEdges`](@ref).

### Arguments

  - `dir`: The direction.

### Returns

  - `parts::Tuple`: The single directions.

# Related

  - [`OutEdges`](@ref)
  - [`InEdges`](@ref)
  - [`BothEdges`](@ref)
  - [`WalkProfilePanel`](@ref)
"""
abstract type AbstractEdgeDirection <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Walks each edge from its source to its target, so an asset reaches what it depends on.

The walk reads the stored layout of the adjacency, the column of the asset in `At`. Two assets are close when they depend on the same nodes.

# Related

  - [`AbstractEdgeDirection`](@ref)
  - [`InEdges`](@ref)
  - [`BothEdges`](@ref)
"""
struct OutEdges <: AbstractEdgeDirection end
"""
$(DocStringExtensions.TYPEDEF)

Walks each edge from its target to its source, so an asset reaches what depends on it.

The walk reads the transpose of the stored adjacency, the row of the asset in `At`. Two assets are close when the same nodes depend on them, such as two suppliers of one customer.

# Related

  - [`AbstractEdgeDirection`](@ref)
  - [`OutEdges`](@ref)
  - [`BothEdges`](@ref)
"""
struct InEdges <: AbstractEdgeDirection end
"""
$(DocStringExtensions.TYPEDEF)

Walks the out-edges and the in-edges apart, and gives one Panel Field for each.

The two profiles are each scaled to unit length, so a cosine of the stack of the two fields is the mean of the two cosines when neither part is zero. The walk does not add the two directions into one adjacency. A walk of two hops over the sum mixes "shares a supplier" with "is a supplier". A [`GraphUnion`](@ref) in the walk builds one union for each single direction.

# Related

  - [`AbstractEdgeDirection`](@ref)
  - [`OutEdges`](@ref)
  - [`InEdges`](@ref)
"""
struct BothEdges <: AbstractEdgeDirection end
"""
    walk_parts(dir::OutEdges) -> Tuple{OutEdges}
    walk_parts(dir::InEdges) -> Tuple{InEdges}
    walk_parts(dir::BothEdges) -> Tuple{OutEdges, InEdges}

Return the single directions of a walk, one Panel Field for each.

# Algorithm

The method that Julia selects is the algorithm.

# Arguments

  - $(arg_dict[:walk_dir])

# Returns

  - `parts::Tuple`: The single directions.

# Related

  - [`AbstractEdgeDirection`](@ref)
  - [`WalkProfilePanel`](@ref)
"""
function walk_parts(::OutEdges)
    return (OutEdges(),)
end
function walk_parts(::InEdges)
    return (InEdges(),)
end
function walk_parts(::BothEdges)
    return (OutEdges(), InEdges())
end
"""
    edge_direction_label(dir::OutEdges) -> String
    edge_direction_label(dir::InEdges) -> String

Return the part of the name of a Panel Field that names a single direction: `"out"` or `"in"`.

# Algorithm

The method that Julia selects is the algorithm.

# Related

  - [`walk_parts`](@ref)
  - [`WalkProfilePanel`](@ref)
"""
function edge_direction_label(::OutEdges)
    return "out"
end
function edge_direction_label(::InEdges)
    return "in"
end
"""
    directed_weights(dir::OutEdges, At::MatNum) -> MatNum
    directed_weights(dir::InEdges, At::MatNum) -> MatNum

Return the adjacency of a single direction, with the start of each step in its columns.

`At` holds the sources in its columns, so a walk along the out-edges reads it as it is, with no copy. A walk along the in-edges reads its transpose, one copy of the stored entries, dense or sparse as `At` is.

# Algorithm

The method that Julia selects is the algorithm.

# Arguments

  - `dir`: The single direction.
  - `At`: The adjacency of the graph, with the sources in its columns.

# Returns

  - `M::MatNum`: The adjacency of the walk. `M[j, i]` is the weight of the step from node `i` to node `j`.

# Related

  - [`ExogenousGraph`](@ref)
  - [`walk_operator`](@ref)
"""
function directed_weights(::OutEdges, At::MatNum)
    return At
end
function directed_weights(::InEdges, At::MatNum)
    return permutedims(At)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that normalise the edge weights of a walk before it starts.

[`WalkProfilePanel`](@ref) holds a member, or `nothing` for the stored weights, in its `nrm` field. A [`GraphUnion`](@ref) holds one too, and normalises each part before the sum.

# Interfaces

To add a rule, subtype `AbstractEdgeNormalisation` and implement the following method.

## `normalise_edge_weights`

  - `normalise_edge_weights(nrm::MyEdgeNormalisation, M::MatNum) -> MatNum`: Return the adjacency of the walk after the rule, with the start of each step in its columns. It keeps the kind of storage of `M`, dense or sparse.

### Arguments

  - `nrm`: The rule.
  - `M`: The adjacency of the walk, with non-negative weights.

### Returns

  - `M::MatNum`: The normalised adjacency.

A walk under a new member takes the scaled walk that [`WalkProfilePanel`](@ref) states, which is exact for any scale of the weights.

# Related

  - [`WeightShare`](@ref)
  - [`WalkProfilePanel`](@ref)
  - [`normalise_edge_weights`](@ref)
"""
abstract type AbstractEdgeNormalisation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Divides each edge weight by the total weight of its node along the walk direction.

Each step of the walk is then a share of its node, and a walk of `h` hops is the distribution of a random walk after `h` steps. A node with no edge along the direction keeps a zero column, so the adjacency of the walk is substochastic, and the total of each distribution is at most one.

A [`GraphUnion`](@ref) and [`graph_union`](@ref) apply the rule to each part before the sum, along the single direction `dir` of the consumer: a column sum of `At` under [`OutEdges`](@ref), and a row sum under [`InEdges`](@ref). [`BothEdges`](@ref) has no single direction there, so it raises a `MethodError`.

# Mathematical definition

```math
\\begin{align}
\\tilde{M}_{j,\\,i} &= \\begin{cases}
\\dfrac{M_{j,\\,i}}{\\sum_{k} M_{k,\\,i}} & \\sum_{k} M_{k,\\,i} > 0\\,, \\\\
0 & \\text{otherwise}\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``\\mathbf{M}``: Adjacency of the walk before the rule, nodes × nodes, with the start of each step in its columns. Entry ``M_{j,\\,i}`` is the non-negative weight of the step from node ``i`` to node ``j``.
  - $(math_dict[:Mt_walk])

# Related

  - [`AbstractEdgeNormalisation`](@ref)
  - [`WalkProfilePanel`](@ref)
  - [`normalise_edge_weights`](@ref)
"""
struct WeightShare <: AbstractEdgeNormalisation end
"""
    normalise_edge_weights(nrm::Nothing, M::MatNum) -> MatNum
    normalise_edge_weights(nrm::WeightShare, M::MatNum) -> MatNum

Normalise the adjacency of a walk under a rule, or keep it under `nothing`.

# Algorithm

The method that Julia selects is the algorithm.

 1. `nothing`: return `M`.
 2. [`WeightShare`](@ref): take the column sums `s` of `M`, and multiply `M` from the right by the diagonal of the inverse of each sum that is not zero, and of zero for a sum of zero. The product keeps the storage of `M`.

# Arguments

  - `nrm`: The rule, or `nothing`.
  - `M`: The adjacency of the walk, with the start of each step in its columns.

# Returns

  - `M::MatNum`: The normalised adjacency.

# Related

  - [`AbstractEdgeNormalisation`](@ref)
  - [`WeightShare`](@ref)
  - [`walk_operator`](@ref)
"""
function normalise_edge_weights(::Nothing, M::MatNum)
    return M
end
function normalise_edge_weights(::WeightShare, M::MatNum)
    s = vec(sum(M; dims = 1))
    return M * LinearAlgebra.Diagonal([iszero(x) ? zero(inv(x)) : inv(x) for x in s])
end

export OutEdges, InEdges, BothEdges, WeightShare
public AbstractEdgeDirection, AbstractEdgeNormalisation, walk_parts, normalise_edge_weights
