"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the reach of a [`DependencyExposure`](@ref): the rule that gives the part of the weight of a held asset that reaches each node of an Exogenous Graph.

A member gives two matrices for each snapshot of the graph, the part at one hop and the part at two hops or more, `nodes × assets`. The exposure multiplies them by the weights, so the reach is found one time for each snapshot and serves each row of weights that reads the snapshot.

# Interfaces

To add a reach, subtype `AbstractDependencyReach` and implement the following method.

## `dependency_reach`

  - `dependency_reach(reach::MyReach, dirs::Tuple, gs::Tuple, ka::VecInt, held::AbstractVector{Bool}) -> Tuple{MatNum, MatNum}`: Return the part at one hop and the part at two hops or more of the reach from the node of each asset, over the nodes of the snapshot.

### Arguments

  - `reach`: The reach.
  - `dirs`: The parts of the direction, from [`exposure_parts`](@ref).
  - `gs`: The graph of each part, one [`ExogenousGraph`](@ref) each. The graphs have the same nodes.
  - `ka`: The node position of each asset that is a node of the snapshot.
  - `held`: `true` for each asset of `ka` that holds a weight that is not zero at a row that reads the snapshot.

### Returns

  - `(Rd, Ri)`: Two matrices, `nodes × length(ka)`. Each is zero at the node of its own asset.

# Related

  - [`ShortestHops`](@ref)
  - [`WalkShares`](@ref)
  - [`DependencyExposure`](@ref)
  - [`dependency_reach`](@ref)
"""
abstract type AbstractDependencyReach <: AbstractAlgorithm end
"""
    assert_exposure_budget(sep::HopCount{<:Union{CountOf, FractionOf}}) -> nothing
    assert_exposure_budget(sep::HopCount) -> Union{}

Refuse a hop budget that is a rule in a reach of a [`DependencyExposure`](@ref).

A rule reads the returns of a fit, and the exposure describes a Result with no returns to read. A count and a fraction need no returns.

# Algorithm

The method that Julia selects is the algorithm.

# Arguments

  - `sep`: The hop budget.

# Validation

  - `sep.n` is a [`CountOf`](@ref) or a [`FractionOf`](@ref). Raises an `ArgumentError` that names the type of the rule.

# Returns

  - `nothing`.

# Related

  - [`ShortestHops`](@ref)
  - [`WalkShares`](@ref)
"""
function assert_exposure_budget(::HopCount{<:Union{CountOf, FractionOf}})::Nothing
    return nothing
end
function assert_exposure_budget(sep::HopCount)::Nothing
    return throw(ArgumentError("the hop budget of a dependency exposure is a CountOf or a FractionOf, because the exposure reads no returns for a rule, and got the rule $(typeof(sep.n))"))
end
"""
$(DocStringExtensions.TYPEDEF)

Reaches each node along the shortest path from the node of a held asset, the reach of the oracle.

The weight `w_i` of asset `i` reaches node `j` at `h` hops, the unweighted length of the shortest path from `i` to `j` along the direction of the exposure, with the score `f(h)`. A node at one hop is a direct dependency, and a node at two hops or more an indirect one. Any edge that is not zero counts as one hop, whatever its weight, and a self-loop does not change a hop count. Each asset counts one time at each node. A path that returns to the asset counts nothing, because the own weight of an asset is only in the part `weight`.

# Mathematical definition

```math
\\begin{align}
R_{j,\\,i} &= \\begin{cases}
f(h_{i,\\,j}) & 1 \\leq h_{i,\\,j} \\leq n\\,, \\\\
0 & \\text{otherwise}\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``R_{j,\\,i}``: Reach of asset ``i`` at node ``j``.
  - ``h_{i,\\,j}``: Number of hops of the shortest path from the node of asset ``i`` to node ``j``, along the direction of the exposure. A node that the path does not reach has no hop count.
  - ``f``: [`separation_decay`](@ref) of `decay` at the separation ``h`` and the budget ``n``, or one when `decay` is `nothing`.
  - ``n``: Hop budget, the resolved `sep`.

The default `HopCount(; n = FractionOf(1.0))` resolves to the largest hop count of the pairs that the held assets reach, so it reaches every node that a path reaches. `HopCount(k)` stops at `k` hops.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ShortestHops(;
        sep::HopCount = HopCount(; n = FractionOf(1.0)),
        decay::Option{<:AbstractSeparationDecayAlgorithm} = nothing
    ) -> ShortestHops

Keywords correspond to the struct's fields.

## Validation

  - The rules of [`assert_exposure_budget`](@ref): `sep.n` is a [`CountOf`](@ref) or a [`FractionOf`](@ref).

# Examples

```jldoctest
julia> ShortestHops(; sep = HopCount(2))
ShortestHops
    sep ┼ HopCount
        │   n ┼ CountOf
        │     │   val ┴ Int64: 2
  decay ┴ nothing
```

# Related

  - [`AbstractDependencyReach`](@ref)
  - [`WalkShares`](@ref)
  - [`DependencyExposure`](@ref)
  - [`HopCount`](@ref)
  - [`separation_decay`](@ref)
"""
@concrete struct ShortestHops <: AbstractDependencyReach
    """
    $(field_dict[:dexp_sep])
    """
    sep
    """
    $(field_dict[:dexp_decay])
    """
    decay
    function ShortestHops(sep::HopCount, decay::Option{<:AbstractSeparationDecayAlgorithm})
        assert_exposure_budget(sep)
        return new{typeof(sep), typeof(decay)}(sep, decay)
    end
end
function ShortestHops(; sep::HopCount = HopCount(; n = FractionOf(1.0)),
                      decay::Option{<:AbstractSeparationDecayAlgorithm} = nothing)::ShortestHops
    return ShortestHops(sep, decay)
end
"""
$(DocStringExtensions.TYPEDEF)

Reaches each node along every walk from the node of a held asset, at the share of the inputs of the asset that comes from the node.

The weight `w_i` of asset `i` reaches node `j` at the sum over the hops `h` of the weight `(M̃^h)[j, i]` of the walks of `h` hops from `i` to `j`, each with the score `f(h)`. Under [`WeightShare`](@ref) and [`OutEdges`](@ref), `M̃[j, i]` is the share of the inputs of `i` that comes from `j`, so the sum is the input-output table of the dependencies at all depths to the budget. A self-loop is in the total of its node, so the part of the inputs that a firm makes itself lowers its shares to other nodes. The part at one hop is direct, and the part at two hops or more is indirect. A walk that returns to the asset counts nothing.

[`BothEdges`](@ref) adds the walks along the out-edges and along the in-edges, and [`AnyEdges`](@ref) walks the undirected graph.

# Mathematical definition

```math
\\begin{align}
R_{j,\\,i} &= \\sum_{h=1}^{n} f(h)\\, \\left(\\tilde{\\mathbf{M}}^{h}\\right)_{j,\\,i}\\,, \\quad j \\neq i\\,.
\\end{align}
```

Where:

  - ``R_{j,\\,i}``: Reach of asset ``i`` at node ``j``.
  - $(math_dict[:Mt_walk])
  - ``f``: [`separation_decay`](@ref) of `decay` at the separation ``h`` and the budget ``n``, or one when `decay` is `nothing`.
  - ``n``: Hop budget, the resolved `sep`.

The default budget resolves to the largest hop count of the shortest paths that the held assets reach. A walk can be longer than a shortest path, so state a [`CountOf`](@ref) for a table to a deeper level. Under `nrm = nothing` the walk multiplies the stored weights, so a large weight gives a large reach.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WalkShares(;
        sep::HopCount = HopCount(; n = FractionOf(1.0)),
        decay::Option{<:AbstractSeparationDecayAlgorithm} = nothing,
        nrm::Option{<:AbstractEdgeNormalisation} = WeightShare(),
        wgt::Option{<:AbstractEdgeWeightRule} = nothing
    ) -> WalkShares

Keywords correspond to the struct's fields.

## Validation

  - The rules of [`assert_exposure_budget`](@ref): `sep.n` is a [`CountOf`](@ref) or a [`FractionOf`](@ref).

# Examples

```jldoctest
julia> WalkShares(; sep = HopCount(3))
WalkShares
    sep ┼ HopCount
        │   n ┼ CountOf
        │     │   val ┴ Int64: 3
  decay ┼ nothing
    nrm ┼ WeightShare()
    wgt ┴ nothing
```

# Related

  - [`AbstractDependencyReach`](@ref)
  - [`ShortestHops`](@ref)
  - [`DependencyExposure`](@ref)
  - [`WalkProfilePanel`](@ref)
  - [`WeightShare`](@ref)
"""
@concrete struct WalkShares <: AbstractDependencyReach
    """
    $(field_dict[:dexp_sep])
    """
    sep
    """
    $(field_dict[:dexp_decay])
    """
    decay
    """
    $(field_dict[:walk_nrm])
    """
    nrm
    """
    $(field_dict[:walk_wgt])
    """
    wgt
    function WalkShares(sep::HopCount, decay::Option{<:AbstractSeparationDecayAlgorithm},
                        nrm::Option{<:AbstractEdgeNormalisation},
                        wgt::Option{<:AbstractEdgeWeightRule})
        assert_exposure_budget(sep)
        return new{typeof(sep), typeof(decay), typeof(nrm), typeof(wgt)}(sep, decay, nrm,
                                                                         wgt)
    end
end
function WalkShares(; sep::HopCount = HopCount(; n = FractionOf(1.0)),
                    decay::Option{<:AbstractSeparationDecayAlgorithm} = nothing,
                    nrm::Option{<:AbstractEdgeNormalisation} = WeightShare(),
                    wgt::Option{<:AbstractEdgeWeightRule} = nothing)::WalkShares
    return WalkShares(sep, decay, nrm, wgt)
end
"""
$(DocStringExtensions.TYPEDEF)

Configures the Dependency Exposure of a portfolio: the weight of the held assets that depends, directly or indirectly, on each node of an Exogenous Graph.

An edge from `A` to `B` states that `A` depends on `B`, such as a token on its protocol or a firm on its supplier. The exposure to a node is the weight of the node itself, as a held asset, plus the weight of each held asset that the reach takes to the node. [`dependency_exposure`](@ref) applies the configuration to a prediction, to an optimisation result, or to bare weights, and [`portfolio_report`](@ref) reads it in its section "Dependency exposure".

The exposure has three parts: `weight`, `direct` at one hop, and `indirect` at two hops or more. Their sum is `total`. A short weight lowers the exposure, so the Result holds the long part and the short part of each, and the gross exposure is their difference.

The taxonomy section of [`portfolio_report`](@ref) is a case of the exposure: its `net` of a group equals `direct` of a `DependencyExposure` with `ShortestHops(; sep = HopCount(1))` on the bipartite graph from each asset to its label.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DependencyExposure(;
        ng::Option{<:Union{AbstractString, GraphUnion}} = nothing,
        reach::AbstractDependencyReach = ShortestHops(),
        dir::AbstractEdgeDirection = OutEdges(),
        src::AbstractPreviousWeightsSource = TargetWeights(),
        nodes::Option{<:VecStr} = nothing,
        strict::Bool = false
    ) -> DependencyExposure

Keywords correspond to the struct's fields. The defaults are the oracle: every node that a shortest path reaches along the out-edges, at the target weights of each fold.

## Validation

  - A `String` in `ng` is not empty. Raises an [`IsEmptyError`](@ref). See [`assert_graph_names`](@ref).
  - `nodes` is not empty, holds no empty name and holds no name twice. Raises an [`IsEmptyError`](@ref) or an `ArgumentError`.
  - `dir` has a method of [`exposure_parts`](@ref). Raises a `MethodError`.

The checks against the graphs run in [`dependency_exposure`](@ref), before any compute: each snapshot of the graph has non-negative weights, see [`graph_requirements`](@ref).

# Examples

```jldoctest
julia> DependencyExposure(; ng = \"supply\", dir = InEdges())
DependencyExposure
      ng ┼ String: "supply"
   reach ┼ ShortestHops
         │     sep ┼ HopCount
         │         │   n ┼ FractionOf
         │         │     │   val ┴ Float64: 1.0
         │   decay ┴ nothing
     dir ┼ InEdges()
     src ┼ TargetWeights()
   nodes ┼ nothing
  strict ┴ Bool: false
```

# Related

  - [`dependency_exposure`](@ref)
  - [`DependencyExposureResult`](@ref)
  - [`ShortestHops`](@ref)
  - [`WalkShares`](@ref)
  - [`AbstractEdgeDirection`](@ref)
  - [`exposure_weights`](@ref)
  - [`ExogenousGraph`](@ref)
  - [`portfolio_report`](@ref)
"""
@concrete struct DependencyExposure <: AbstractEstimator
    """
    $(field_dict[:dexp_ng])
    """
    ng
    """
    $(field_dict[:dexp_reach])
    """
    reach
    """
    $(field_dict[:dexp_dir])
    """
    dir
    """
    $(field_dict[:dexp_src])
    """
    src
    """
    $(field_dict[:dexp_nodes])
    """
    nodes
    """
    $(field_dict[:dexp_strict])
    """
    strict
    function DependencyExposure(ng::Option{<:Union{AbstractString, GraphUnion}},
                                reach::AbstractDependencyReach, dir::AbstractEdgeDirection,
                                src::AbstractPreviousWeightsSource, nodes::Option{<:VecStr},
                                strict::Bool)
        assert_graph_names(ng, :ng)
        assert_graph_names(nodes, :nodes)
        # A direction with no part of a reach raises its `MethodError` here, before any call.
        exposure_parts(dir)
        return new{typeof(ng), typeof(reach), typeof(dir), typeof(src), typeof(nodes),
                   typeof(strict)}(ng, reach, dir, src, nodes, strict)
    end
end
function DependencyExposure(; ng::Option{<:Union{AbstractString, GraphUnion}} = nothing,
                            reach::AbstractDependencyReach = ShortestHops(),
                            dir::AbstractEdgeDirection = OutEdges(),
                            src::AbstractPreviousWeightsSource = TargetWeights(),
                            nodes::Option{<:VecStr} = nothing,
                            strict::Bool = false)::DependencyExposure
    return DependencyExposure(ng, reach, dir, src, nodes, strict)
end
function graph_requirements(::DependencyExposure)
    return (NonNegativeWeights(),)
end
function graph_names(de::DependencyExposure)
    return de.ng
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the Dependency Exposure of each row of weights of a Result to each node of an Exogenous Graph.

A row is one portfolio: a fold, an observation under a drift, the mean of the rows of a path of a population, or one weight vector. `L` and `S` hold the long and the short parts of `weight`, `direct` and `indirect`, so the net exposure is `L + S` and the gross exposure is `L - S`. [`DataFrames.DataFrame(::DependencyExposureResult)`](@ref) gives the table of the parts. The reach of each distinct snapshot is in `R0`, `Rd` and `Ri`, and [`exposure_contributions`](@ref) gives the part of each asset at each node of one row. The Result holds no contribution for each row, because `rows × nodes × assets` numbers can be too many.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DependencyExposureResult(;
        nodes::VecStr,
        nx::VecStr,
        rows::AbstractVector,
        W::MatNum,
        L::AbstractArray{<:Number, 3},
        S::AbstractArray{<:Number, 3},
        R0::AbstractVector{<:MatNum},
        Rd::AbstractVector{<:MatNum},
        Ri::AbstractVector{<:MatNum},
        ridx::VecInt
    ) -> DependencyExposureResult

Keywords correspond to the struct's fields. [`dependency_exposure`](@ref) builds the Result.

## Validation

  - `size(W) == (length(rows), length(nx))`. Raises a `DimensionMismatch`.
  - `size(L) == size(S) == (length(rows), length(nodes), 3)`. Raises a `DimensionMismatch`.
  - `R0`, `Rd`, `Ri` have one matrix for each snapshot, each `length(nodes) × length(nx)`. Raises a `DimensionMismatch`.
  - `length(ridx) == length(rows)`, and each entry is `0` or a snapshot. Raises a `DimensionMismatch` or a `DomainError`.

# Related

  - [`DependencyExposure`](@ref)
  - [`dependency_exposure`](@ref)
  - [`exposure_contributions`](@ref)
"""
@concrete struct DependencyExposureResult <: AbstractResult
    """
    $(field_dict[:dexp_res_nodes])
    """
    nodes
    """
    $(field_dict[:dexp_res_nx])
    """
    nx
    """
    $(field_dict[:dexp_res_rows])
    """
    rows
    """
    $(field_dict[:dexp_res_W])
    """
    W
    """
    $(field_dict[:dexp_res_L])
    """
    L
    """
    $(field_dict[:dexp_res_S])
    """
    S
    """
    $(field_dict[:dexp_res_R0])
    """
    R0
    """
    $(field_dict[:dexp_res_Rd])
    """
    Rd
    """
    $(field_dict[:dexp_res_Ri])
    """
    Ri
    """
    $(field_dict[:dexp_res_ridx])
    """
    ridx
    function DependencyExposureResult(nodes::VecStr, nx::VecStr, rows::AbstractVector,
                                      W::MatNum, L::AbstractArray{<:Number, 3},
                                      S::AbstractArray{<:Number, 3},
                                      R0::AbstractVector{<:MatNum},
                                      Rd::AbstractVector{<:MatNum},
                                      Ri::AbstractVector{<:MatNum}, ridx::VecInt)
        r, m, n = length(rows), length(nodes), length(nx)
        @argcheck(size(W) == (r, n),
                  DimensionMismatch("W has one row for each label of `rows` and one column for each asset of `nx`, so size(W) must be ($r, $n), got $(size(W))"))
        @argcheck(size(L) == size(S) == (r, m, 3),
                  DimensionMismatch("L and S have one row for each label of `rows`, one column for each node and the three parts, so their size must be ($r, $m, 3), got $(size(L)) and $(size(S))"))
        @argcheck(length(R0) == length(Rd) == length(Ri) &&
                  all(R -> size(R) == (m, n), Iterators.flatten((R0, Rd, Ri))),
                  DimensionMismatch("R0, Rd and Ri hold one matrix of size ($m, $n) for each snapshot"))
        @argcheck(length(ridx) == r,
                  DimensionMismatch("ridx has one entry for each label of `rows`, so length(ridx) must be $r, got $(length(ridx))"))
        @argcheck(all(p -> 0 <= p <= length(Rd), ridx),
                  DomainError(ridx,
                              "each entry of ridx is 0 or the position of a snapshot in 1:$(length(Rd))"))
        return new{typeof(nodes), typeof(nx), typeof(rows), typeof(W), typeof(L), typeof(S),
                   typeof(R0), typeof(Rd), typeof(Ri), typeof(ridx)}(nodes, nx, rows, W, L,
                                                                     S, R0, Rd, Ri, ridx)
    end
end
function DependencyExposureResult(; nodes::VecStr, nx::VecStr, rows::AbstractVector,
                                  W::MatNum, L::AbstractArray{<:Number, 3},
                                  S::AbstractArray{<:Number, 3},
                                  R0::AbstractVector{<:MatNum},
                                  Rd::AbstractVector{<:MatNum},
                                  Ri::AbstractVector{<:MatNum},
                                  ridx::VecInt)::DependencyExposureResult
    return DependencyExposureResult(nodes, nx, rows, W, L, S, R0, Rd, Ri, ridx)
end
"""
    DataFrames.DataFrame(res::DependencyExposureResult) -> DataFrames.DataFrame

Write the Dependency Exposure of a Result as a table.

With one row of weights, the table has one row for each node. With many rows it is long: one row for each row of weights and each node, in that order, after the column `row` that holds the label of the row of weights. This is the index of the oracle, the observation and then the dependency.

The value columns are `weight`, `direct` and `indirect`, the net parts, `total`, their sum, `long` and `short`, the sums of the long and the short parts, and `gross`, the difference of `long` and `short`.

# Arguments

  - `res`: The Result.

# Returns

  - `df::DataFrames.DataFrame`: The table, with the column `node` first or after `row`.

# Examples

```jldoctest
julia> using DataFrames

julia> g = ExogenousGraph(; edges = [(\"Token A\", \"Protocol\", 1), (\"Token B\", \"Token A\", 3)],
                          nodes = [\"Token A\", \"Token B\", \"Protocol\"]);

julia> res = dependency_exposure([0.6, 0.4], [\"Token A\", \"Token B\"], g);

julia> DataFrame(res)[:, [:node, :weight, :direct, :indirect, :total]]
3×5 DataFrame
 Row │ node      weight   direct   indirect  total
     │ String    Float64  Float64  Float64   Float64
─────┼───────────────────────────────────────────────
   1 │ Token A       0.6      0.4       0.0      1.0
   2 │ Token B       0.4      0.0       0.0      0.4
   3 │ Protocol      0.0      0.6       0.4      1.0
```

# Related

  - [`DependencyExposureResult`](@ref)
  - [`dependency_exposure`](@ref)
"""
function DataFrames.DataFrame(res::DependencyExposureResult)
    r, m = length(res.rows), length(res.nodes)
    # Row-major: the nodes of the first row of weights, then the nodes of the second.
    part(A, k) = vec(permutedims(view(A, :, :, k)))
    weight, direct, indirect = (part(res.L, k) .+ part(res.S, k) for k in 1:3)
    long = part(res.L, 1) .+ part(res.L, 2) .+ part(res.L, 3)
    short = part(res.S, 1) .+ part(res.S, 2) .+ part(res.S, 3)
    df = DataFrames.DataFrame(; node = repeat(collect(res.nodes), r), weight = weight,
                              direct = direct, indirect = indirect,
                              total = weight .+ direct .+ indirect, long = long,
                              short = short, gross = long .- short)
    if r > 1
        DataFrames.insertcols!(df, 1, :row => repeat(collect(res.rows); inner = m))
    end
    return df
end
"""
    exposure_contributions(res::DependencyExposureResult, k::Integer) -> Matrix

Return the part of each asset in the exposure of each node, at row `k` of a Dependency Exposure.

The contribution of asset `i` to node `j` is ``C_{j,\\,i} = w_{i}\\, R_{j,\\,i}``, with ``\\mathbf{R} = \\mathbf{R}_{0} + \\mathbf{R}_{d} + \\mathbf{R}_{i}`` the reach of the snapshot of the row, its own node included. So the sum of the row `j` of `C` is the `total` of node `j`.

# Arguments

  - `res`: The Result.
  - `k`: The row of weights.

# Validation

  - Row `k` has one snapshot, `res.ridx[k] > 0`. A row with no graph in force, or the mean of the rows of a path of a population, has none. Raises an `ArgumentError`.

# Returns

  - `C::Matrix`: The contributions, `nodes × assets`, on `res.nodes` and `res.nx`.

# Examples

```jldoctest
julia> g = ExogenousGraph(; edges = [(\"Token A\", \"Protocol\", 1), (\"Token B\", \"Token A\", 3)],
                          nodes = [\"Token A\", \"Token B\", \"Protocol\"]);

julia> exposure_contributions(dependency_exposure([0.6, 0.4], [\"Token A\", \"Token B\"], g), 1)
3×2 Matrix{Float64}:
 0.6  0.4
 0.0  0.4
 0.6  0.4
```

# Related

  - [`DependencyExposureResult`](@ref)
  - [`dependency_exposure`](@ref)
"""
function exposure_contributions(res::DependencyExposureResult, k::Integer)
    p = res.ridx[k]
    @argcheck(p > 0,
              ArgumentError("row $k of the Dependency Exposure has no single snapshot: no graph is in force at it, or it is the mean of the rows of a path. Read the contributions of a row with a graph"))
    return (res.R0[p] .+ res.Rd[p] .+ res.Ri[p]) .* transpose(view(res.W, k, :))
end

export ShortestHops, WalkShares, DependencyExposure, DependencyExposureResult,
       exposure_contributions
public AbstractDependencyReach
