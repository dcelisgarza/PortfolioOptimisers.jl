"""
$(DocStringExtensions.TYPEDEF)

The basis of the nodes of an Exogenous Graph. It re-bases a constraint written in node names through the reach of the assets over the graph.

A row such as `\"Protocol <= 0.3\"` caps the Dependency Exposure of the portfolio to the node `Protocol`: the weight of `Protocol` itself, when it is an asset, plus the weight of each asset that the reach takes to it. [`DependencyExposure`](@ref) measures the same exposure on a Result, with the same `ng`, `reach` and `dir`.

The cap is on the net exposure, because a re-basis takes a linear form in the weights only. A short weight lowers it. Under non-negative weights the net exposure and the gross exposure are equal. A [`PositionConstraintEstimator`](@ref) around the [`ExposureConstraintEstimator`](@ref) caps the gross, the long or the short exposure that the [`DependencyExposure`](@ref) reports. No entry of the reach is negative, so such a cap holds on the realised weights.

The space reads the graph in force at the last row of the returns, so each fold of a cross-validation and each nested optimisation reads its own graph. The space holds no data, so a view of the assets returns it unchanged.

!!! warning

    The reach marks every asset of the problem as held, because the fit does not know the weights before it solves. A [`FractionOf`](@ref) budget takes its quantile over the pairs of every asset. A [`DependencyExposure`](@ref) takes it over the pairs of the held assets only, so the two budgets can differ. State a [`CountOf`](@ref) budget to measure exactly the exposure that the row caps. Under [`ShortestHops`](@ref) with `decay = nothing`, the default budget reaches every node that a path reaches, and the two agree.

# Mathematical definition

```math
\\begin{align}
  \\boldsymbol{e} &= \\mathbf{R} \\boldsymbol{w} \\\\
  \\boldsymbol{a}^\\intercal \\boldsymbol{e} &= \\left(\\mathbf{R}^\\intercal \\boldsymbol{a}\\right)^\\intercal \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{e}``: ``K \\times 1`` vector of the net Dependency Exposure to each node.
  - ``\\mathbf{R}``: ``K \\times N`` reach of each asset at each node, the own node of the asset included, from [`snapshot_reach`](@ref).
  - $(math_dict[:w_port])
  - ``\\boldsymbol{a}``: ``K \\times 1`` row that the user wrote in node names.

# The names of a row

A name of a row is a node of the graph, and it need not be an asset. The names resolve against the nodes of every snapshot of the graph in the returns data that the fit sees, not against a key of [`UniverseSets`](@ref). A node that the graph in force does not hold has zero exposure. A group of `sets.dict` expands to its members, as in any other constraint, and it keeps a member that left the investable universe, because that member can still be a node. A node with the name of a group reads as the group. A node name that is not a Julia name, such as a name with a space, reaches a row through a group with that one member.

A fold sees no snapshot of its future. So a name that first occurs in a later snapshot does not resolve in an early fold, and the constraint generator drops the row with a warning, or raises under `strict`. The `key` of the [`LinearConstraintEstimator`](@ref) prevents this: it names an entry of `sets.dict` with every node name, and a declared name that the fit does not see has zero exposure. See [`declared_basis`](@ref).

An asset that is not a node of the graph in force adds nothing to any exposure. A row that no asset reaches has zero coefficients, and the constraint generator drops it as [`get_linear_constraints`](@ref) states.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DependencySpace(;
        ng::Option{<:Union{AbstractString, GraphUnion}} = nothing,
        reach::AbstractDependencyReach = ShortestHops(),
        dir::AbstractEdgeDirection = OutEdges()
    ) -> DependencySpace

Keywords correspond to the struct's fields. The defaults are those of [`DependencyExposure`](@ref).

## Validation

  - A `String` in `ng` is not empty. Raises an [`IsEmptyError`](@ref). See [`assert_graph_names`](@ref).
  - `dir` has a method of [`exposure_parts`](@ref). Raises a `MethodError`.

The checks against the returns data run when the library generates the constraint:

  - The returns data is given, and holds the graph. Raises an [`IsNothingError`](@ref) or an `ArgumentError`.
  - A graph is in force at the last row. Raises an `ArgumentError`, because a fit cannot continue with no graph.
  - Each edge weight is non-negative, [`NonNegativeWeights`](@ref). Raises an `ArgumentError`. The entry of the user verb checks every snapshot, see [`assert_graph_entry`](@ref).

An [`ExposureConstraintEstimator`](@ref) over this space refuses a precomputed [`LinearConstraint`](@ref), see [`assert_space_constraint`](@ref). The outer optimiser of a [`NestedClustered`](@ref) or a [`Stacking`](@ref) refuses the space, see [`assert_outer_readers`](@ref).

# Algorithm

[`constraint_space_basis`](@ref) does these steps for this space:

 1. Read the graph of each part of `dir` in force at the last row of `rd` with [`consumer_graphs`](@ref), after the checks above.
 2. Read the node names with [`space_nodes`](@ref), and the assets from `sets.dict[sets.xkey]`.
 3. Find the reach of every asset at every node with [`snapshot_reach`](@ref), and add its three parts.
 4. Return the transpose of the reach as a [`ConstraintBasis`](@ref) over the node names, and `nothing` as the key.

# Examples

```jldoctest
julia> g = ExogenousGraph(; edges = [(\"TokenA\", \"Protocol\", 1), (\"TokenB\", \"TokenA\", 3)],
                          nodes = [\"TokenA\", \"TokenB\", \"Protocol\"]);

julia> rd = ReturnsResult(; nx = [\"TokenA\", \"TokenB\", \"TokenC\"], X = zeros(4, 3), ng = [\"defi\"],
                          G = [g]);

julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"TokenA\", \"TokenB\", \"TokenC\"]));

julia> ece = ExposureConstraintEstimator(;
                                         lce = LinearConstraintEstimator(;
                                                                         val = [\"Protocol <= 0.3\",
                                                                                \"TokenA <= 0.2\"]),
                                         space = DependencySpace(; ng = \"defi\"));

julia> linear_constraints(ece, sets; rd = rd).ineq.A
2×3 transpose(::Matrix{Float64}) with eltype Float64:
 1.0  1.0  0.0
 1.0  1.0  0.0
```

# Related

  - [`AbstractConstraintSpace`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
  - [`DependencyExposure`](@ref)
  - [`ShortestHops`](@ref)
  - [`WalkShares`](@ref)
  - [`GraphUnion`](@ref)
  - [`ConstraintBasis`](@ref)
  - [`PositionConstraintEstimator`](@ref)
"""
@concrete struct DependencySpace <: AbstractConstraintSpace
    """
    $(field_dict[:dsp_ng])
    """
    ng
    """
    $(field_dict[:dsp_reach])
    """
    reach
    """
    $(field_dict[:dexp_dir])
    """
    dir
    function DependencySpace(ng::Option{<:Union{AbstractString, GraphUnion}},
                             reach::AbstractDependencyReach, dir::AbstractEdgeDirection)
        assert_graph_names(ng, :ng)
        # A direction with no part of a reach raises its `MethodError` here, before any call.
        exposure_parts(dir)
        return new{typeof(ng), typeof(reach), typeof(dir)}(ng, reach, dir)
    end
end
function DependencySpace(; ng::Option{<:Union{AbstractString, GraphUnion}} = nothing,
                         reach::AbstractDependencyReach = ShortestHops(),
                         dir::AbstractEdgeDirection = OutEdges())::DependencySpace
    return DependencySpace(ng, reach, dir)
end
function graph_requirements(::DependencySpace)
    return (NonNegativeWeights(),)
end
function graph_names(space::DependencySpace)
    return space.ng
end
"""
    space_nodes(rd::ReturnsResult, ng::Option{<:Union{AbstractString, GraphUnion}}) -> Vector{String}

Return the nodes of every snapshot of the graph in the returns data, or of each part of a union, in the order of their first use.

A node that only an earlier snapshot holds is a name that a row can use, with zero exposure. The returns data of a fold holds no snapshot of its future.

# Arguments

  - $(arg_dict[:rd])
  - `ng`: The graph or the union.

# Returns

  - `nodes::Vector{String}`: The names.

# Related

  - [`DependencySpace`](@ref)
  - [`graph_snapshot_list`](@ref)
"""
function space_nodes(rd::ReturnsResult, ng::Option{<:Union{AbstractString, GraphUnion}})
    names = isa(ng, GraphUnion) ? ng.ng : (ng,)
    Gs = [rd.G[graph_position(rd.ng, name)] for name in names]
    return unique(String[n for G in Gs for (_, g) in graph_snapshot_list(G)
                         for n in g.nodes])
end
# The methods below are the interface of the family. Their docstrings are on the functions of
# `src/09_ConstraintGeneration/08_ExposureConstraintGeneration.jl`.
function constraint_space_basis(space::DependencySpace, sets::UniverseSets, ::Any,
                                rd::Option{<:ReturnsResult} = nothing)
    @argcheck(!isnothing(rd),
              IsNothingError("a constraint over a `DependencySpace` is written in the nodes of an Exogenous Graph, and the graph is in the returns data. This route carries no returns data. Pass it, `linear_constraints(ece, sets; rd = rd)`, or generate the constraint through a `JuMPOptimiser` or a `Pipeline` step"))
    gs, gname = consumer_graphs(space, rd, space.ng, space.dir)
    nodes = space_nodes(rd, space.ng)
    nx = sets.dict[sets.xkey]
    R0, Rd, Ri = snapshot_reach(space.reach, space.dir, gs, nx, nodes, 0, trues(length(nx)))
    return ConstraintBasis(permutedims(R0 .+ Rd .+ Ri), nodes, "node", gname), nothing
end
function assert_space_constraint(::DependencySpace, ::LinearConstraint)::Nothing
    return throw(ArgumentError("a `DependencySpace` re-bases rows that name the nodes of the graph, and a precomputed `LinearConstraint` holds columns with no names. The nodes of a graph change over time, so a column has no meaning in a later fit. Write the rows as a `LinearConstraintEstimator` in node names"))
end
function assert_space_constraint(space::DependencySpace, lce::AbstractVector)::Nothing
    return foreach(x -> assert_space_constraint(space, x), lce)
end
function holds_node_reader(::DependencySpace)::Bool
    return true
end
pipe_reads(::DependencySpace) = (:returns,)

export DependencySpace
