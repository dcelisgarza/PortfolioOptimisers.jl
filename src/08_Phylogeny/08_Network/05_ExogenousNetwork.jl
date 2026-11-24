"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that state which nodes of an Exogenous Graph an Exogenous Network holds.

[`ExogenousNetwork`](@ref) holds a member in its `scope` field. A node of an [`ExogenousGraph`](@ref) can be an asset, an entity that is not an asset, or a label, so a path between two assets can go through a node that is not an asset.

# Interfaces

To add a scope, subtype `AbstractNetworkScope` and implement the following method.

## `scope_vertices`

  - `scope_vertices(scope::MyNetworkScope, nodes::VecStr, nx::VecStr) -> Tuple{Integer, VecInt, VecInt}`: Place the nodes of a graph and the assets on the vertices of the network.

### Arguments

  - `scope`: The scope.
  - `nodes`: The names of the nodes of the graph.
  - `nx`: The asset names.

### Returns

  - `n::Integer`: The number of vertices.
  - `vn::VecInt`: The vertex of each node, or `0` for a node that the network leaves out.
  - `va::VecInt`: The vertex of each asset. Each asset has a vertex, also when it is not a node.

# Related

  - [`AllNodes`](@ref)
  - [`AssetNodes`](@ref)
  - [`ExogenousNetwork`](@ref)
"""
abstract type AbstractNetworkScope <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Keeps every node of the graph, so a path between two assets can go through any node.

A centrality is found in the whole graph, and the result holds the entries of the assets. An asset that the prior drops stays a node, so it changes no path.

# Related

  - [`AbstractNetworkScope`](@ref)
  - [`AssetNodes`](@ref)
  - [`ExogenousNetwork`](@ref)
"""
struct AllNodes <: AbstractNetworkScope end
"""
$(DocStringExtensions.TYPEDEF)

Keeps the subgraph of the assets of the fit, so a path goes through assets alone.

An edge with an end that is not an asset of the fit is not in the network, and a normalisation of the weights reads the subgraph.

# Related

  - [`AbstractNetworkScope`](@ref)
  - [`AllNodes`](@ref)
  - [`ExogenousNetwork`](@ref)
"""
struct AssetNodes <: AbstractNetworkScope end
"""
    scope_vertices(scope::AllNodes, nodes::VecStr, nx::VecStr) -> Tuple{Int, Vector{Int}, Vector{Int}}
    scope_vertices(scope::AssetNodes, nodes::VecStr, nx::VecStr) -> Tuple{Int, Vector{Int}, Vector{Int}}

Place the nodes of a graph and the assets on the vertices of an Exogenous Network.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`AllNodes`](@ref): each node keeps its position as its vertex. Each asset that is not a node takes a vertex of its own after the nodes.
 2. [`AssetNodes`](@ref): the vertices are the assets, in the order of `nx`. A node that is not an asset has no vertex.

# Arguments

  - `scope`: The scope.
  - `nodes`: The names of the nodes of the graph.
  - `nx`: The asset names.

# Returns

  - `n::Int`: The number of vertices.
  - `vn::Vector{Int}`: The vertex of each node, or `0` for a node that the network leaves out.
  - `va::Vector{Int}`: The vertex of each asset.

# Related

  - [`AbstractNetworkScope`](@ref)
  - [`walk_positions`](@ref)
"""
function scope_vertices(::AllNodes, nodes::VecStr, nx::VecStr)
    va = walk_positions(nodes, nx)
    lost = findall(iszero, va)
    va[lost] = length(nodes) .+ eachindex(lost)
    return length(nodes) + length(lost), collect(eachindex(nodes)), va
end
function scope_vertices(::AssetNodes, nodes::VecStr, nx::VecStr)
    return length(nx), walk_positions(nx, nodes), collect(eachindex(nx))
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that turn an edge weight of an Exogenous Graph into a length.

A large edge weight states a strong relation, and a short length states one, so a shortest path over the weights goes through the weakest relations. [`ExogenousNetwork`](@ref) holds a member in its `len` field, or `nothing` when the stored weights are lengths.

# Interfaces

To add a rule, subtype `AbstractEdgeLength` and implement the following method.

## `edge_length`

  - `edge_length(len::MyEdgeLength, w::Number) -> Number`: Return the length of an edge with the weight `w`, after the normalisation of the network. It must not be negative.

### Arguments

  - `len`: The rule.
  - `w`: The positive weight of the edge.

### Returns

  - `l::Number`: The length.

# Related

  - [`ReciprocalLength`](@ref)
  - [`NegativeLogLength`](@ref)
  - [`ExogenousNetwork`](@ref)
"""
abstract type AbstractEdgeLength <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Gives an edge the reciprocal of a power of its weight as its length.

The default `power = 1` is the length of [newman2001](@cite), and the exponent is the tuning parameter of [opsahl2010](@cite). `power = 0` gives each edge a unit length, so a shortest path has the fewest edges. A `power` above one shortens a strong relation more, so a path of many strong edges can beat a path of one weak edge.

# Mathematical definition

```math
\\begin{align}
\\ell &= \\dfrac{1}{w^{\\alpha}}\\,.
\\end{align}
```

Where:

  - ``\\ell``: Length of the edge.
  - ``w``: Weight of the edge, after the normalisation of the network.
  - ``\\alpha``: Exponent, the `power` field.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReciprocalLength(;
        power::Number = 1
    ) -> ReciprocalLength

Keywords correspond to the struct's fields.

## Validation

  - `power` is finite and `power >= 0`. Raises a `DomainError`.

# Examples

```jldoctest
julia> ReciprocalLength(; power = 0.5)
ReciprocalLength
  power ┴ Float64: 0.5
```

# Related

  - [`AbstractEdgeLength`](@ref)
  - [`NegativeLogLength`](@ref)
  - [`ExogenousNetwork`](@ref)

# References

  - $(ref_dict[:newman2001]) Section III.
  - $(ref_dict[:opsahl2010]) Equation 8.
"""
@concrete struct ReciprocalLength <: AbstractEdgeLength
    """
    Exponent of the weight. `0` gives each edge a unit length.
    """
    power
    function ReciprocalLength(power::Number)
        assert_nonempty_nonneg_finite_val(power, :power)
        return new{typeof(power)}(power)
    end
end
function ReciprocalLength(; power::Number = 1)::ReciprocalLength
    return ReciprocalLength(power)
end
"""
$(DocStringExtensions.TYPEDEF)

Gives an edge the negative logarithm of its weight as its length.

The length of a path is the negative logarithm of the product of its weights, so the shortest path has the largest product. With [`WeightShare`](@ref) in the `nrm` field of the network, the product is the probability that a random walk takes the path. The rule refuses a weight above one, because its length is negative.

# Mathematical definition

```math
\\begin{align}
\\ell &= -\\log w\\,, \\\\
\\sum_{e} \\ell_{e} &= -\\log \\prod_{e} w_{e}\\,.
\\end{align}
```

Where:

  - ``\\ell``: Length of the edge.
  - ``w``: Weight of the edge, after the normalisation of the network, ``0 < w \\leq 1``.
  - ``e``: Edge of a path.

# Related

  - [`AbstractEdgeLength`](@ref)
  - [`ReciprocalLength`](@ref)
  - [`WeightShare`](@ref)
  - [`ExogenousNetwork`](@ref)
"""
struct NegativeLogLength <: AbstractEdgeLength end
"""
    edge_length(len::Nothing, w::Number) -> Number
    edge_length(len::ReciprocalLength, w::Number) -> Number
    edge_length(len::NegativeLogLength, w::Number) -> Number

Return the length of an edge with the weight `w` under a rule, or the weight itself under `nothing`.

# Algorithm

The method that Julia selects is the algorithm.

 1. `nothing`: return `w`.
 2. [`ReciprocalLength`](@ref): return the reciprocal of `w` to the power `len.power`.
 3. [`NegativeLogLength`](@ref): refuse a weight above one, and return the negative logarithm of `w`.

# Arguments

  - `len`: The rule, or `nothing`.
  - `w`: The positive weight of the edge.

# Validation

  - Under [`NegativeLogLength`](@ref), `w <= 1`. Raises a `DomainError`.

# Returns

  - `l::Number`: The length. It is zero for a weight of one under [`NegativeLogLength`](@ref), and [`network_lengths`](@ref) repairs it.

# Related

  - [`AbstractEdgeLength`](@ref)
  - [`network_lengths`](@ref)
"""
function edge_length(::Nothing, w::Number)
    return w
end
function edge_length(len::ReciprocalLength, w::Number)
    return inv(w^len.power)
end
function edge_length(::NegativeLogLength, w::Number)
    @argcheck(w <= one(w),
              DomainError(w,
                          "NegativeLogLength gives an edge the length -log(w), which is negative for a weight above one, so it refuses the weight. Set `nrm = WeightShare()` on the network, so that each weight is a share, or use ReciprocalLength()."))
    return -log(w)
end
"""
$(DocStringExtensions.TYPEDEF)

Reads a named Exogenous Graph of the returns data as a network of the assets.

It is the network that the caller states, where [`NetworkEstimator`](@ref) fits a network to the returns. It reads the graph in force at the last row of the returns data at the point of use, so each fold of a cross-validation and each subproblem reads its own graph. It reads no returns, so a gap in the returns does not refuse the fit, and it states no source of returns. A call with no returns data refuses.

Each consumer of a network takes it: [`phylogeny_matrix`](@ref), [`SemiDefinitePhylogenyEstimator`](@ref), [`IntegerPhylogenyEstimator`](@ref), [`asset_phylogeny`](@ref), [`centrality_vector`](@ref), [`CentralityEstimator`](@ref), [`CentralityConstraint`](@ref) and [`PhylogenyPanel`](@ref). To cluster over the graph, put the panel in the distance of a clustering estimator, `FeatureDistance(; ape = PhylogenyPanel(; pl = ExogenousNetwork()))`. [`NetworkClustersEstimator`](@ref) refuses the network, because its kernel reads the similarities and the distances of the returns. The outer optimiser of [`NestedClustered`](@ref) and [`Stacking`](@ref) refuses it, because a synthetic asset is not a node. The factor phylogeny of [`FactorRiskContribution`](@ref) refuses it, because a factor is not a node.

# Direction

The network reads each edge along `dir`. ``d(i,\\,k)`` is the separation from asset ``i`` to asset ``k``, in row ``i`` and column ``k`` of [`separation_matrix`](@ref).

| `dir`               | separation of the pair                                                       |
|:------------------- |:---------------------------------------------------------------------------- |
| [`OutEdges`](@ref)  | the path from ``i`` to ``k`` along the edges                                 |
| [`InEdges`](@ref)   | the path from ``i`` to ``k`` against the edges                               |
| [`BothEdges`](@ref) | the smaller of the two one-way separations of the pair, over both directions |
| [`AnyEdges`](@ref)  | the path in the undirected graph, so a path can turn at each node            |

Under [`AnyEdges`](@ref), two opposite edges between two nodes give the shorter of their two lengths, because a shortest path takes it. They give the sum of their two weights to an algorithm that reads a strength.

A consumer refuses a direction that has no meaning for it. Its constructor raises the error, and a direct call of a function raises it at the call.

| consumer                                                                                      | directions                                                                                                 |
|:--------------------------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------- |
| [`phylogeny_matrix`](@ref), [`asset_phylogeny`](@ref) and the phylogeny constraints           | [`AnyEdges`](@ref) and [`BothEdges`](@ref), because the matrix must be symmetric                           |
| [`centrality_vector`](@ref), [`CentralityEstimator`](@ref) and [`CentralityConstraint`](@ref) | [`AnyEdges`](@ref), [`OutEdges`](@ref) and [`InEdges`](@ref), because [`BothEdges`](@ref) is not one graph |
| [`PhylogenyPanel`](@ref) with [`Proximity`](@ref)                                             | every direction                                                                                            |

# The separations read the assets

The separations are measured from the assets, with one traversal for each asset, and their rows and columns are the assets. So a [`FractionOf`](@ref) budget and the observed diameter of a [`PathLength`](@ref) read the pairs of assets alone, also when the paths go through other nodes. A [`HopCount`](@ref) reads no weight, and a [`PathLength`](@ref) budget has the units of `len`. A budget rule receives the whole graph of `scope`, along the out-edges under [`BothEdges`](@ref).

# Centrality

A centrality is found in the whole graph of `scope`, and the result holds the entries of the assets. An algorithm of shortest paths reads the lengths. [`EigenvectorCentrality`](@ref) reads the weights after `nrm`. When `len` is `nothing`, the weights are lengths and state no strength, so it reads the plain graph, as it does on a tree. An algorithm that reads no weight reads the closure of `sep` over the graph of `scope`.

# Weights

A self-loop is ignored. It is removed before any other step, so a normalisation does not count it. `nrm` applies before `len`. It acts along the single direction of a part, and along the out-edges under [`AnyEdges`](@ref), as a [`GraphUnion`](@ref) with a normalisation of its own does. A zero length cannot be the weight of an edge, so it moves to the smallest positive value of its type, as in [`graph_weight_matrix`](@ref). [`HopCount`](@ref), [`TopologyOnly`](@ref) and `ReciprocalLength(; power = 0)` each give unit weights.

# An asset that is not a node

An asset that is not a node of the graph in force is an isolated vertex, and it stays investable. One warning names these assets and the graph. Under `strict`, it is an error.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExogenousNetwork(;
        ng::Option{<:Union{AbstractString, GraphUnion}} = nothing,
        dir::AbstractEdgeDirection = AnyEdges(),
        scope::AbstractNetworkScope = AllNodes(),
        nrm::Option{<:AbstractEdgeNormalisation} = nothing,
        len::Option{<:AbstractEdgeLength} = ReciprocalLength(),
        sep::AbstractSeparationAlgorithm = HopCount(),
        strict::Bool = false
    ) -> ExogenousNetwork

Keywords correspond to the struct's fields.

## Validation

  - A `String` in `ng` is not empty. Raises an [`IsEmptyError`](@ref). See [`assert_graph_names`](@ref).
  - `dir` has a method of [`exposure_parts`](@ref). Raises a `MethodError`.

The checks against the returns data run at the entry of a user verb or at the call:

  - Each edge weight of each snapshot is non-negative, the requirement [`NonNegativeWeights`](@ref). Raises an `ArgumentError`.
  - A graph is in force at the last row of the returns data. Raises an `ArgumentError`.
  - Under `strict`, each asset is a node of the graph in force. Raises an `ArgumentError`.

# Examples

```jldoctest
julia> ExogenousNetwork(; ng = \"supply\", sep = PathLength())
ExogenousNetwork
      ng ┼ String: \"supply\"
     dir ┼ AnyEdges()
   scope ┼ AllNodes()
     nrm ┼ nothing
     len ┼ ReciprocalLength
         │   power ┴ Int64: 1
     sep ┼ PathLength
         │   dmax ┴ nothing
  strict ┴ Bool: false
```

The graph is read from the returns data. Two suppliers of one customer are two hops apart.

```jldoctest
julia> g = ExogenousGraph(; edges = [(\"A\", \"C\", 2.0), (\"B\", \"C\", 1.0)]);

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; 0.0 0.1], ng = [\"supply\"], G = [g]);

julia> separation_matrix(HopCount(), ExogenousNetwork(), rd.X; rd = rd)
2×2 Matrix{Int64}:
 0  2
 2  0
```

# Related

  - [`AbstractNetworkEstimator`](@ref)
  - [`NetworkEstimator`](@ref)
  - [`ExogenousGraph`](@ref)
  - [`AbstractEdgeDirection`](@ref)
  - [`AbstractNetworkScope`](@ref)
  - [`AbstractEdgeLength`](@ref)
  - [`AbstractSeparationAlgorithm`](@ref)
  - [`DependencyExposure`](@ref)
  - [`WalkProfilePanel`](@ref)
"""
@concrete struct ExogenousNetwork <: AbstractNetworkEstimator
    """
    $(field_dict[:xnet_ng])
    """
    ng
    """
    $(field_dict[:xnet_dir])
    """
    dir
    """
    $(field_dict[:xnet_scope])
    """
    scope
    """
    $(field_dict[:xnet_nrm])
    """
    nrm
    """
    $(field_dict[:xnet_len])
    """
    len
    """
    $(field_dict[:ntsep])
    """
    sep
    """
    $(field_dict[:xnet_strict])
    """
    strict
    function ExogenousNetwork(ng::Option{<:Union{AbstractString, GraphUnion}},
                              dir::AbstractEdgeDirection, scope::AbstractNetworkScope,
                              nrm::Option{<:AbstractEdgeNormalisation},
                              len::Option{<:AbstractEdgeLength},
                              sep::AbstractSeparationAlgorithm, strict::Bool)
        assert_graph_names(ng, :ng)
        # A direction with no part raises its `MethodError` here, before any call.
        exposure_parts(dir)
        return new{typeof(ng), typeof(dir), typeof(scope), typeof(nrm), typeof(len),
                   typeof(sep), typeof(strict)}(ng, dir, scope, nrm, len, sep, strict)
    end
end
function ExogenousNetwork(; ng::Option{<:Union{AbstractString, GraphUnion}} = nothing,
                          dir::AbstractEdgeDirection = AnyEdges(),
                          scope::AbstractNetworkScope = AllNodes(),
                          nrm::Option{<:AbstractEdgeNormalisation} = nothing,
                          len::Option{<:AbstractEdgeLength} = ReciprocalLength(),
                          sep::AbstractSeparationAlgorithm = HopCount(),
                          strict::Bool = false)::ExogenousNetwork
    return ExogenousNetwork(ng, dir, scope, nrm, len, sep, strict)
end
function graph_requirements(::ExogenousNetwork)
    return (NonNegativeWeights(),)
end
function graph_names(nte::ExogenousNetwork)
    return nte.ng
end
"""
    reads_network_returns(pl) -> Bool
    reads_network_returns(nte::ExogenousNetwork) -> Bool

Tell whether a network or clustering source reads the returns, so that the consumer that holds it adds a frame to its head.

An [`ExogenousNetwork`](@ref) reads the graph of the returns data and no returns, so its consumer reads the same values under every source of the returns. [`reader_frames`](@ref) of the consumer reads this answer.

# Arguments

  - `pl`: The network or clustering source.

# Returns

  - `flag::Bool`: `false` for an [`ExogenousNetwork`](@ref), and `true` otherwise.

# Related

  - [`reader_frames`](@ref)
  - [`ExogenousNetwork`](@ref)
"""
function reads_network_returns(::Any)::Bool
    return true
end
function reads_network_returns(::ExogenousNetwork)::Bool
    return false
end
"""
    symmetric_direction(dir::AbstractEdgeDirection) -> Bool
    symmetric_direction(dir::Union{BothEdges, AnyEdges}) -> Bool

Tell whether the separations of an Exogenous Network along a direction are symmetric.

# Arguments

  - `dir`: The direction.

# Returns

  - `flag::Bool`: `true` for [`BothEdges`](@ref) and [`AnyEdges`](@ref), and `false` otherwise.

# Related

  - [`ExogenousNetwork`](@ref)
  - [`assert_symmetric_network`](@ref)
"""
function symmetric_direction(::AbstractEdgeDirection)::Bool
    return false
end
function symmetric_direction(::Union{BothEdges, AnyEdges})::Bool
    return true
end
"""
    assert_symmetric_network(pl::NwE_ClE) -> Nothing
    assert_symmetric_network(nte::ExogenousNetwork) -> Nothing

Refuse a network whose phylogeny matrix is not symmetric.

[`phylogeny_matrix`](@ref) calls it, and the constructors of [`SemiDefinitePhylogenyEstimator`](@ref) and [`IntegerPhylogenyEstimator`](@ref) do too. Only an [`ExogenousNetwork`](@ref) can have separations that are not symmetric.

# Arguments

  - `pl`: The network or clustering source.

# Validation

  - [`symmetric_direction`](@ref) of the `dir` of an [`ExogenousNetwork`](@ref) is `true`. Raises an `ArgumentError`.

# Related

  - [`ExogenousNetwork`](@ref)
  - [`assert_single_network`](@ref)
"""
function assert_symmetric_network(::NwE_ClE)::Nothing
    return nothing
end
function assert_symmetric_network(nte::ExogenousNetwork)::Nothing
    @argcheck(symmetric_direction(nte.dir),
              ArgumentError("a phylogeny matrix is symmetric, and the separations of an ExogenousNetwork along $(nameof(typeof(nte.dir)))() are not. Use `dir = AnyEdges()` for a path that can turn at each node, or `dir = BothEdges()` for the smaller of the two one-way separations."))
    return nothing
end
"""
    assert_single_network(pl::NwE_ClE) -> Nothing
    assert_single_network(nte::ExogenousNetwork) -> Nothing

Refuse a network that is not one graph, so it has no centrality.

[`centrality_graph`](@ref) calls it, and the constructor of [`CentralityEstimator`](@ref) does too.

# Arguments

  - `pl`: The network or clustering source.

# Validation

  - The `dir` of an [`ExogenousNetwork`](@ref) has one part, see [`exposure_parts`](@ref). Raises an `ArgumentError`.

# Related

  - [`ExogenousNetwork`](@ref)
  - [`assert_symmetric_network`](@ref)
"""
function assert_single_network(::NwE_ClE)::Nothing
    return nothing
end
function assert_single_network(nte::ExogenousNetwork)::Nothing
    @argcheck(isone(length(exposure_parts(nte.dir))),
              ArgumentError("a centrality is found in one graph, and an ExogenousNetwork along $(nameof(typeof(nte.dir)))() reads $(length(exposure_parts(nte.dir))) graphs apart. Use `dir = AnyEdges()`, `dir = OutEdges()` or `dir = InEdges()`."))
    return nothing
end
"""
    assert_returns_network(nte::AbstractNetworkEstimator) -> Nothing
    assert_returns_network(nte::ExogenousNetwork) -> Nothing

Refuse a network that has no similarity and no distance matrix of the returns.

The constructor of [`NetworkClustersEstimator`](@ref) calls it, because its kernel reads both matrices.

# Arguments

  - $(arg_dict[:nte])

# Validation

  - `nte` is not an [`ExogenousNetwork`](@ref). Raises an `ArgumentError`.

# Related

  - [`NetworkClustersEstimator`](@ref)
  - [`ExogenousNetwork`](@ref)
"""
function assert_returns_network(::AbstractNetworkEstimator)::Nothing
    return nothing
end
function assert_returns_network(::ExogenousNetwork)::Nothing
    return throw(ArgumentError("NetworkClustersEstimator reads the similarities and the distances of the returns, and an ExogenousNetwork has neither. To cluster over the graph, use `ClustersEstimator(; de = FeatureDistance(; ape = PhylogenyPanel(; pl = ExogenousNetwork())))`."))
end

export ExogenousNetwork, AllNodes, AssetNodes, ReciprocalLength, NegativeLogLength
public AbstractNetworkScope, AbstractEdgeLength, scope_vertices, edge_length
