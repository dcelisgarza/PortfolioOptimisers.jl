"""
    assert_union_weights(w::Nothing, n::Integer) -> nothing
    assert_union_weights(w::VecNum, n::Integer) -> nothing

Refuse the weights of a union of `n` graphs that do not give one usable weight for each part.

The checks follow the weights of [`BlendDistance`](@ref). The union does not normalise the weights, because a raw edge weight is data.

# Algorithm

The method that Julia selects is the algorithm.

 1. `nothing`: return `nothing`. Each part has the weight one.
 2. A vector: check its length, and check that each entry is finite and non-negative and that one entry is positive.

# Arguments

  - $(arg_dict[:gu_w])
  - `n`: The number of parts.

# Validation

  - `length(w) == n`. Raises a `DimensionMismatch`.
  - Each entry is finite and non-negative. Raises a `DomainError`.
  - One entry or more is positive. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`GraphUnion`](@ref)
  - [`graph_union`](@ref)
"""
function assert_union_weights(::Nothing, ::Integer)::Nothing
    return nothing
end
function assert_union_weights(w::VecNum, n::Integer)::Nothing
    @argcheck(length(w) == n,
              DimensionMismatch("the weights `w` of a union of graphs have $(length(w)) entries, and the union has $n parts. State one weight for each part"))
    @argcheck(all(x -> isfinite(x) && zero(x) <= x, w),
              DomainError(w,
                          "every weight `w` of a union of graphs must be finite and non-negative"))
    @argcheck(any(x -> zero(x) < x, w),
              DomainError(w,
                          "the weights `w` of a union of graphs have no positive entry, so no part enters the union. State a positive weight for one part or more"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the spec of a union of named Exogenous Graphs, which a consumer of a graph reads in its `ng` field.

A union joins relations before a consumer reads them, so a path can change relation on its way, such as bank → supplier → firm. A sum of what the consumer reads on each graph cannot give that path. A consumer turns the spec into one graph at the point of use, with [`graph_union`](@ref) on the snapshots of the parts in force at the last row of its window and with the direction of the consumer. So a series of parts needs no stored copy of the union. Store the output of [`graph_union`](@ref) under a name of its own in the returns data when many consumers read one union.

A vector of names in the `ng` field of a consumer keeps its own meaning: each graph stays apart. A `GraphUnion` is the one spelling of a union.

# Mathematical definition

```math
\\begin{align}
\\mathbf{A} &= \\sum_{k} w_{k}\\, N(\\mathbf{A}_{k})\\,.
\\end{align}
```

Where:

  - ``\\mathbf{A}``: Adjacency of the union, over the union of the nodes of the parts, in the order of their first use over the parts.
  - ``\\mathbf{A}_{k}``: Adjacency of the part `ng[k]`, embedded in the nodes of the union.
  - ``w_{k}``: Weight of the part, `w[k]`, or one when `w` is `nothing`.
  - ``N``: The normalisation `nrm` along the direction of the consumer, or the identity when `nrm` is `nothing`.

An edge that two relations hold gets the sum of its weights. Under `nrm = WeightShare()`, column `i` of the union, under [`OutEdges`](@ref), is the weighted mix of the shares that each relation gives the dependencies of node `i`. A consumer that applies [`WeightShare`](@ref) again spreads the weights again over the relations in which a node has edges: a firm with credit links and no supply links takes all its share from credit.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GraphUnion(;
        ng::VecStr,
        w::Option{<:VecNum} = nothing,
        nrm::Option{<:AbstractEdgeNormalisation} = nothing,
        unknown::AbstractUnknownEntryRule = EntrywiseUnknown(),
        name::Option{<:AbstractString} = nothing
    ) -> GraphUnion

Keywords correspond to the struct's fields.

## Validation

  - `ng` is not empty, holds no empty name and holds no name twice. Raises an [`IsEmptyError`](@ref) or an `ArgumentError`. See [`assert_graph_names`](@ref).
  - One name with `nrm = nothing` is the graph itself, a second spelling of the name in the consumer. Raises an `ArgumentError` that names `ng = "<name>"`. One name with a normalisation is a normalised graph, and it is valid.
  - The rules of [`assert_union_weights`](@ref) for `w`.
  - A `String` in `name` is not empty. Raises an [`IsEmptyError`](@ref). See [`assert_panel_field_name`](@ref).
  - `unknown` implements [`union_parts_in_force`](@ref). [`KindwiseUnknown`](@ref) raises a `MethodError`.

The checks against the returns data run at the entry of the user verb, before the first fold: each name is a graph of the returns data, each snapshot of each part with a positive weight meets the requirements of the consumer, and the keys of these parts are of one kind. A part with the weight `0` does not enter the union, so these checks skip it, see [`union_kept`](@ref). See [`assert_consumer_entry`](@ref). Non-negative parts and non-negative weights give a non-negative union, so the union needs no check of its own.

# Examples

```jldoctest
julia> gu = GraphUnion(; ng = [\"supply\", \"credit\"], nrm = WeightShare());

julia> PortfolioOptimisers.graph_entry_label(gu)
\"supply+credit\"

julia> WalkProfilePanel(; ng = [\"supply\", gu]).ng[2] === gu
true
```

# Related

  - [`graph_union`](@ref)
  - [`ExogenousGraph`](@ref)
  - [`WalkProfilePanel`](@ref)
  - [`AbstractUnknownEntryRule`](@ref)
  - [`AbstractEdgeNormalisation`](@ref)
"""
@concrete struct GraphUnion <: AbstractAlgorithm
    """
    $(field_dict[:gu_ng])
    """
    ng
    """
    $(field_dict[:gu_w])
    """
    w
    """
    $(field_dict[:gu_nrm])
    """
    nrm
    """
    $(field_dict[:gu_unknown])
    """
    unknown
    """
    $(field_dict[:gu_name])
    """
    name
    function GraphUnion(ng::VecStr, w::Option{<:VecNum},
                        nrm::Option{<:AbstractEdgeNormalisation},
                        unknown::AbstractUnknownEntryRule, name::Option{<:AbstractString})
        assert_graph_names(ng, :ng)
        @argcheck(!(isone(length(ng)) && isnothing(nrm)),
                  ArgumentError("a GraphUnion of the one graph \"$(first(ng))\" with `nrm = nothing` is the graph itself. Write `ng = \"$(first(ng))\"` in the consumer, or give a normalisation such as `nrm = WeightShare()` for a normalised graph"))
        assert_union_weights(w, length(ng))
        assert_panel_field_name(name)
        # The rule must have a method of the second job of its family. A call here raises the
        # `MethodError` of a rule that has none, such as `KindwiseUnknown`, before any fit.
        union_parts_in_force(unknown, trues(length(ng)))
        return new{typeof(ng), typeof(w), typeof(nrm), typeof(unknown), typeof(name)}(ng, w,
                                                                                      nrm,
                                                                                      unknown,
                                                                                      name)
    end
end
function GraphUnion(; ng::VecStr, w::Option{<:VecNum} = nothing,
                    nrm::Option{<:AbstractEdgeNormalisation} = nothing,
                    unknown::AbstractUnknownEntryRule = EntrywiseUnknown(),
                    name::Option{<:AbstractString} = nothing)::GraphUnion
    return GraphUnion(ng, w, nrm, unknown, name)
end
"""
    graph_entry_label(name::AbstractString) -> AbstractString
    graph_entry_label(gu::GraphUnion) -> String
    graph_entry_label(x) -> Union{}

Return the name of one entry of the `ng` field of a consumer, which its Panel Fields and its messages quote.

# Algorithm

The method that Julia selects is the algorithm.

 1. A name: return it.
 2. A [`GraphUnion`](@ref): return `gu.name`, or `join(gu.ng, "+")` when it is `nothing`, through [`resolve_panel_field_name`](@ref).
 3. Any other value: throw an `ArgumentError`.

# Arguments

  - `name`, `gu`, `x`: The entry.

# Validation

  - The entry is a name or a [`GraphUnion`](@ref). Raises an `ArgumentError` that names its type.

# Returns

  - `label::AbstractString`: The name of the entry.

# Related

  - [`GraphUnion`](@ref)
  - [`walk_entry_parts!`](@ref)
"""
function graph_entry_label(name::AbstractString)
    return name
end
function graph_entry_label(gu::GraphUnion{<:VecStr})
    return resolve_panel_field_name(gu.name, join(gu.ng, "+"))
end
function graph_entry_label(x)
    return throw(ArgumentError("each entry of a vector in `ng` is the name of a graph or a GraphUnion, and got a $(typeof(x))"))
end
function assert_graph_names(::GraphUnion, ::Sym_Str)::Nothing
    return nothing
end
function assert_graph_names(names::AbstractVector, sym::Sym_Str)::Nothing
    @argcheck(!isempty(names), IsEmptyError("$sym cannot be empty"))
    return assert_graph_names(String[graph_entry_label(e) for e in names], sym)
end
function graph_name_list(gu::GraphUnion)
    return (gu,)
end
"""
    union_parts_in_force(rule::EntrywiseUnknown, known::AbstractVector{Bool}) -> Option{Vector{Int}}
    union_parts_in_force(rule::ZeroUnknown, known::AbstractVector{Bool}) -> Option{Vector{Int}}

Return the positions of the parts that a union of graphs sums at a row, or `nothing` when no graph of the union is in force there.

It is the method of the second job of [`AbstractUnknownEntryRule`](@ref). The time rule of a union series follows from it: the first key of the union is the earliest first key of a part at which the rule returns positions.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`EntrywiseUnknown`](@ref): every part when every part is `known`, and `nothing` otherwise. So the union starts at the latest first key of its parts.
 2. [`ZeroUnknown`](@ref): the parts that are `known`, and `nothing` when none is. A part with no graph in force reads as an empty graph, so the union starts at the earliest first key of its parts.

# Arguments

  - `rule`: The rule.
  - `known`: Whether each part has a graph in force at the row.

# Returns

  - `idx::Option{Vector{Int}}`: The positions of the parts that the union sums, or `nothing`.

# Related

  - [`AbstractUnknownEntryRule`](@ref)
  - [`GraphUnion`](@ref)
  - [`graph_union`](@ref)
"""
function union_parts_in_force(::EntrywiseUnknown, known::AbstractVector{Bool})
    return all(known) ? collect(eachindex(known)) : nothing
end
function union_parts_in_force(::ZeroUnknown, known::AbstractVector{Bool})
    return any(known) ? findall(known) : nothing
end
"""
    union_part_adjacency(nrm::Nothing, dir, At::MatNum) -> MatNum
    union_part_adjacency(nrm::AbstractEdgeNormalisation, dir::AbstractEdgeDirection, At::MatNum) -> MatNum

Return the adjacency of one part of a union after its normalisation along `dir`, in the stored layout, with the sources in its columns.

# Algorithm

The method that Julia selects is the algorithm.

 1. `nothing`: return `At`, with no copy. The direction does not change the stored weights.
 2. A normalisation: orient `At` along `dir` with [`directed_weights`](@ref), normalise it with [`normalise_edge_weights`](@ref), and orient it back. Under [`OutEdges`](@ref) it is a column sum of `At`, and under [`InEdges`](@ref) a row sum. [`BothEdges`](@ref) has no single direction, so it raises a `MethodError`.

# Arguments

  - `nrm`: The normalisation, or `nothing`.
  - `dir`: The single direction.
  - `At`: The adjacency of the part, with the sources in its columns.

# Returns

  - `N::MatNum`: The normalised adjacency, dense or sparse as `At` is.

# Related

  - [`union_snapshot`](@ref)
  - [`normalise_edge_weights`](@ref)
"""
function union_part_adjacency(::Nothing, ::Any, At::MatNum)
    return At
end
function union_part_adjacency(nrm::AbstractEdgeNormalisation, dir::AbstractEdgeDirection,
                              At::MatNum)
    return directed_weights(dir, normalise_edge_weights(nrm, directed_weights(dir, At)))
end
"""
    union_weight(w::Nothing, k::Integer) -> Bool
    union_weight(w::VecNum, k::Integer) -> Number

Return the weight of the part `k` of a union: `true`, the multiplicative identity, when `w` is `nothing`, and `w[k]` otherwise.

# Related

  - [`union_snapshot`](@ref)
"""
function union_weight(::Nothing, ::Integer)
    return true
end
function union_weight(w::VecNum, k::Integer)
    return w[k]
end
"""
    union_weights(w::Nothing, idx::AbstractVector{<:Integer}) -> Nothing
    union_weights(w::VecNum, idx::AbstractVector{<:Integer}) -> VecNum

Return the weights of the parts `idx` of a union, or `nothing` for unit weights.

# Related

  - [`union_snapshot`](@ref)
"""
function union_weights(::Nothing, ::AbstractVector{<:Integer})
    return nothing
end
function union_weights(w::VecNum, idx::AbstractVector{<:Integer})
    return w[idx]
end
"""
    union_kept(w::Nothing, n::Integer) -> Vector{Int}
    union_kept(w::VecNum, n::Integer) -> Vector{Int}

Return the positions of the parts of a union that enter the sum: every part under unit weights, and each part with a positive weight otherwise. A part with the weight `0` is skipped, so its nodes and its keys do not enter the union. The requirements of a consumer and the check of the key kinds do not read it either. A consumer still finds its name in the returns data, because a name that the data does not hold is a wrong spec.

# Related

  - [`graph_union`](@ref)
  - [`union_in_force`](@ref)
  - [`assert_consumer_entry`](@ref)
"""
function union_kept(::Nothing, n::Integer)
    return collect(1:n)
end
function union_kept(w::VecNum, ::Integer)
    return findall(x -> zero(x) < x, w)
end
"""
    union_nodes(gs::AbstractVector) -> Tuple{Vector{String}, Dict{String, Int}}

Return the union of the nodes of the graphs `gs`, in the order of their first use over the graphs, and the position of each name.

# Related

  - [`union_snapshot`](@ref)
"""
function union_nodes(gs::AbstractVector)
    nodes = String[]
    pos = Dict{String, Int}()
    for g in gs, name in g.nodes
        if !haskey(pos, name)
            push!(nodes, name)
            pos[name] = length(nodes)
        end
    end
    return nodes, pos
end
"""
    union_dense(Ms::AbstractVector, ps::AbstractVector, w, T::Type, n::Integer) -> Matrix

Sum the weighted adjacencies `Ms` of the parts of a union into one dense adjacency of `n` nodes.

The node `i` of the part `k` is the node `ps[k][i]` of the union. A shared edge gets the sum of its weights. [`union_sparse`](@ref) is the sparse form.

# Arguments

  - `Ms`: The normalised adjacency of each part, with the sources in its columns.
  - `ps`: The positions of the nodes of each part in the union.
  - $(arg_dict[:gu_w])
  - `T`: The element type of the union.
  - `n`: The number of nodes of the union.

# Returns

  - `At`: The adjacency of the union, `n × n`, of element type `T`.

# Related

  - [`union_snapshot`](@ref)
"""
function union_dense(Ms::AbstractVector, ps::AbstractVector, w, T::Type, n::Integer)
    At = zeros(T, n, n)
    for k in eachindex(Ms, ps)
        view(At, ps[k], ps[k]) .+= union_weight(w, k) .* Ms[k]
    end
    return At
end
"""
    union_sparse(Ms::AbstractVector, ps::AbstractVector, w, T::Type, n::Integer) -> SparseArrays.SparseMatrixCSC

Sum the weighted adjacencies `Ms` of the parts of a union into one sparse adjacency of `n` nodes.

It reads the stored entries of each part with [`stored_entries`](@ref), and `SparseArrays.sparse` adds the entries at one cell. The arguments are those of [`union_dense`](@ref).

# Related

  - [`union_dense`](@ref)
  - [`union_snapshot`](@ref)
"""
function union_sparse(Ms::AbstractVector, ps::AbstractVector, w, T::Type, n::Integer)
    I = Int[]
    J = Int[]
    V = T[]
    for k in eachindex(Ms, ps)
        Ik, Jk, Vk = stored_entries(Ms[k])
        wk = union_weight(w, k)
        p = ps[k]
        for t in eachindex(Ik, Jk, Vk)
            push!(I, p[Ik[t]])
            push!(J, p[Jk[t]])
            push!(V, wk * Vk[t])
        end
    end
    return SparseArrays.sparse(I, J, V, n, n)
end
"""
    union_snapshot(gs::AbstractVector, w, nrm, dir) -> ExogenousGraph

Build the union of the graphs `gs`, the snapshots of the parts that a union sums at one row: ``\\mathbf{A} = \\sum_{k} w_{k}\\, N(\\mathbf{A}_{k})``.

# Algorithm

 1. Take the union of the nodes with [`union_nodes`](@ref).
 2. Normalise each part along `dir` with [`union_part_adjacency`](@ref).
 3. Derive the element type from the parts and `w`, with `promote_type`.
 4. Sum the parts with [`union_dense`](@ref) when every part is dense, and with [`union_sparse`](@ref) otherwise, because a dense embedding of several node sets costs the square of the union.

# Arguments

  - `gs`: The snapshots, each an [`ExogenousGraph`](@ref).
  - $(arg_dict[:gu_w]) One weight for each snapshot.
  - $(arg_dict[:gu_nrm])
  - $(arg_dict[:gu_dir])

# Returns

  - `g::ExogenousGraph`: The union.

# Related

  - [`graph_union`](@ref)
  - [`GraphUnion`](@ref)
"""
function union_snapshot(gs::AbstractVector, w, nrm, dir)
    nodes, pos = union_nodes(gs)
    Ms = [union_part_adjacency(nrm, dir, g.At) for g in gs]
    ps = [[pos[name] for name in g.nodes] for g in gs]
    T = mapreduce(eltype, promote_type, Ms; init = isnothing(w) ? Bool : eltype(w))
    n = length(nodes)
    At = if any(SparseArrays.issparse, Ms)
        union_sparse(Ms, ps, w, T, n)
    else
        union_dense(Ms, ps, w, T, n)
    end
    return ExogenousGraph(nodes, At)
end
"""
    union_part_at(g::ExogenousGraph, at) -> Tuple{ExogenousGraph, Nothing}
    union_part_at(s::ExogenousGraphSeries, at) -> Tuple{Option{<:ExogenousGraph}, Any}

Return the graph of one part of a union in force at the key `at`, with the key of its snapshot.

It does not refuse, because the rule of the union decides what a part with no graph in force means. A series returns `(nothing, nothing)` for a row before its first key, and `(nothing, key)` at a key that holds `nothing`.

# Algorithm

The method that Julia selects is the algorithm.

 1. A static graph: return the graph and `nothing`. It is in force at every row.
 2. A series: find the last key at or before `at` with `searchsortedlast`.

# Arguments

  - `g`, `s`: The part.
  - `at`: The row position or the timestamp.

# Returns

  - `(g, key)`: The graph in force or `nothing`, and its key or `nothing`.

# Related

  - [`graph_union`](@ref)
  - [`graph_snapshot`](@ref)
"""
function union_part_at(g::ExogenousGraph, ::Any)
    return g, nothing
end
function union_part_at(s::ExogenousGraphSeries{<:AbstractVector}, at)
    p = searchsortedlast(s.ts, at)
    return p < 1 ? (nothing, nothing) : (s.graphs[p], s.ts[p])
end
"""
    union_row_key(g::ExogenousGraph, T, ts) -> Nothing
    union_row_key(s::ExogenousGraphSeries{<:VecInt}, T::Integer, ts) -> Integer
    union_row_key(s::ExogenousGraphSeries{<:VecDate}, T, ts::AbstractVector) -> Dates.AbstractTime

Return the key of the last row of the returns data in the terms of one part: the row `T` for integer keys, and the timestamp `ts[end]` for date keys.

# Algorithm

The method that Julia selects is the algorithm.

# Related

  - [`union_part_at`](@ref)
  - [`consumer_graph`](@ref)
"""
function union_row_key(::ExogenousGraph, ::Any, ::Any)
    return nothing
end
function union_row_key(::ExogenousGraphSeries{<:VecInt}, T::Integer, ::Any)
    return T
end
function union_row_key(::ExogenousGraphSeries{<:VecDate}, ::Any, ts::AbstractVector)
    return last(ts)
end
"""
    union_part_started(g::ExogenousGraph, k) -> Bool
    union_part_started(s::ExogenousGraphSeries, k) -> Bool

Return whether a part of a union has reached its first key at the key `k`. A static graph is in force at every row.

# Related

  - [`union_series_keys`](@ref)
"""
function union_part_started(::ExogenousGraph, ::Any)
    return true
end
function union_part_started(s::ExogenousGraphSeries, k)
    return first(s.ts) <= k
end
"""
    union_first_row(k::Integer) -> Integer
    union_first_row(k::Dates.AbstractTime) -> Dates.AbstractTime

Return a key at or before every row of the returns data, of the kind of `k`: the row `1`, or the earliest value of the type of the timestamp.

A union series takes it as its first key when the union is in force before every key of its parts, as a static part under [`ZeroUnknown`](@ref) is.

# Related

  - [`union_series_keys`](@ref)
"""
function union_first_row(k::Integer)
    return one(k)
end
function union_first_row(k::Dates.AbstractTime)
    return typemin(k)
end
"""
    union_series_keys(gs::AbstractVector, unknown::AbstractUnknownEntryRule) -> AbstractVector

Return the keys of the union series of the parts `gs`, by the time rule of `unknown`.

# Algorithm

 1. Take the sorted union of the keys of the series among the parts.
 2. When the rule finds a graph in force before every key, because the static parts suffice, put the first row of [`union_first_row`](@ref) before the keys.
 3. Otherwise, drop each key before the first key at which the rule finds a graph in force, from the parts that have reached their first key, [`union_part_started`](@ref). Under [`EntrywiseUnknown`](@ref) it is the latest first key of the parts, and under [`ZeroUnknown`](@ref) the earliest.

# Arguments

  - `gs`: The parts of the union, one series or more among them.
  - $(arg_dict[:gu_unknown])

# Validation

  - The rule finds a graph in force at one key or more. Raises an `ArgumentError`.

# Returns

  - `ks::AbstractVector`: The keys of the union, strictly increasing.

# Related

  - [`graph_union`](@ref)
  - [`union_parts_in_force`](@ref)
"""
function union_series_keys(gs::AbstractVector, unknown::AbstractUnknownEntryRule)
    # The keys of the series, of the element type of their values. The assertion keeps a
    # verb of another package on a table out of the dispatch of `unique!` and `sort!`.
    tk = reduce(vcat,
                (g.ts::AbstractVector for g in gs if isa(g, ExogenousGraphSeries)))::AbstractVector
    ks = sort!(unique(tk)::Vector)
    if !isnothing(union_parts_in_force(unknown, [isa(g, ExogenousGraph) for g in gs]))
        return unique!(vcat(union_first_row(first(ks)), ks))
    end
    i = findfirst(k -> !isnothing(union_parts_in_force(unknown,
                                                       [union_part_started(g, k)
                                                        for g in gs])), ks)
    if isnothing(i)
        throw(ArgumentError("the rule $(nameof(typeof(unknown))) finds no key at which a graph of the union is in force"))
    end
    return ks[i:end]
end
"""
    union_at_key(gs::AbstractVector, w, nrm, dir, unknown, k) -> Option{<:ExogenousGraph}

Return the union of the parts `gs` in force at the key `k` of a union series, or `nothing` when the rule `unknown` finds no graph in force there.

# Related

  - [`graph_union`](@ref)
  - [`union_snapshot`](@ref)
"""
function union_at_key(gs::AbstractVector, w, nrm, dir, unknown::AbstractUnknownEntryRule, k)
    snaps = [first(union_part_at(g, k)) for g in gs]
    idx = union_parts_in_force(unknown, [!isnothing(s) for s in snaps])
    return if isnothing(idx)
        nothing
    else
        union_snapshot(snaps[idx], union_weights(w, idx), nrm, dir)
    end
end
"""
    assert_union_key_kinds(parts::AbstractVector, labels::AbstractVector{<:AbstractString}) -> nothing

Refuse the parts of a union that mix a series keyed by row positions and a series keyed by dates.

A static part is in force at every row, so it mixes with either kind.

# Arguments

  - `parts`: The parts, each an [`ExogenousGraph`](@ref) or an [`ExogenousGraphSeries`](@ref).
  - `labels`: The name of each part in the message.

# Validation

  - The series among the parts have keys of one kind. Raises an `ArgumentError` that names two parts of different kinds, with [`graph_kind`](@ref).

# Returns

  - `nothing`.

# Related

  - [`graph_union`](@ref)
  - [`assert_consumer_entry`](@ref)
"""
function assert_union_key_kinds(parts::AbstractVector,
                                labels::AbstractVector{<:AbstractString})::Nothing
    s = findall(p -> isa(p, ExogenousGraphSeries), parts)
    if isempty(s)
        return nothing
    end
    a = first(s)
    b = findfirst(i -> graph_kind(parts[i]) != graph_kind(parts[a]), s)
    @argcheck(isnothing(b),
              ArgumentError("the parts of a union of graphs have keys of one kind, and the part $(labels[a]) is $(graph_kind(parts[a])) while the part $(isnothing(b) ? "" : labels[s[b]]) is $(isnothing(b) ? "" : graph_kind(parts[s[b]])). Key every series of the union by row positions, or every series by dates"))
    return nothing
end
"""
    graph_union(gs::ExG_ExGS...;
                w::Option{<:VecNum} = nothing,
                nrm::Option{<:AbstractEdgeNormalisation} = nothing,
                dir::AbstractEdgeDirection = OutEdges(),
                unknown::AbstractUnknownEntryRule = EntrywiseUnknown()) -> ExG_ExGS

Build the union of Exogenous Graphs, ``\\mathbf{A} = \\sum_{k} w_{k}\\, N(\\mathbf{A}_{k})``, as one [`ExogenousGraph`](@ref) or one [`ExogenousGraphSeries`](@ref).

The nodes are the union of the nodes of the parts, in the order of their first use over the parts. The union is dense when every part is dense, and sparse otherwise. An edge that two parts hold gets the sum of its weights. [`GraphUnion`](@ref) states the same union as a spec in the `ng` field of a consumer, which builds it at the point of use. Store the output of this function under a name of its own in the returns data when many consumers read one union.

A union of static graphs only is static. A union with a series is a series. Its keys are the sorted union of the keys of its parts, from the first key that the rule `unknown` gives, and its graph at a key is the union of the snapshots in force there, by the rule. [`EntrywiseUnknown`](@ref) starts at the latest first key of the parts and holds `nothing` when any part holds `nothing`. [`ZeroUnknown`](@ref) reads a part with no graph in force as an empty graph. A static part is in force at every row.

# Algorithm

 1. Check the parts, the weights, and that `unknown` has a method of [`union_parts_in_force`](@ref).
 2. Keep the parts with a positive weight, [`union_kept`](@ref), and check their key kinds. A part with the weight `0` does not enter the union, so its kind is free.
 3. Static parts only: build the union with [`union_snapshot`](@ref).
 4. A series among the parts: find the keys with [`union_series_keys`](@ref), and build the union at each key with [`union_at_key`](@ref).

# Arguments

  - $(arg_dict[:gu_gs])
  - $(arg_dict[:gu_w])
  - $(arg_dict[:gu_nrm])
  - $(arg_dict[:gu_dir])
  - $(arg_dict[:gu_unknown])

# Validation

  - One part or more. Raises an [`IsEmptyError`](@ref).
  - One part with `nrm = nothing` is the part itself. Raises an `ArgumentError`.
  - The rules of [`assert_union_weights`](@ref) and [`assert_union_key_kinds`](@ref).
  - `unknown` implements [`union_parts_in_force`](@ref). [`KindwiseUnknown`](@ref) raises a `MethodError`.
  - [`BothEdges`](@ref) under a normalisation raises a `MethodError`, because the rule needs a single direction.

# Returns

  - `g::ExG_ExGS`: The union.

# Examples

```jldoctest
julia> g1 = ExogenousGraph(; edges = [(\"S\", \"A\", 2.0)]);

julia> g2 = ExogenousGraph(; edges = [(\"B\", \"A\", 1.0), (\"S\", \"A\", 1.0)]);

julia> u = graph_union(g1, g2);

julia> u.nodes
3-element Vector{String}:
 \"S\"
 \"A\"
 \"B\"

julia> adjacency(u)[1, 2]
3.0

julia> v = graph_union(g1, g2; nrm = WeightShare(), dir = InEdges());

julia> adjacency(v)[1, 2], adjacency(v)[3, 2]
(1.5, 0.5)
```

# Related

  - [`GraphUnion`](@ref)
  - [`ExogenousGraph`](@ref)
  - [`ExogenousGraphSeries`](@ref)
  - [`AbstractUnknownEntryRule`](@ref)
  - [`WeightShare`](@ref)
"""
function graph_union(gs::ExG_ExGS...; w::Option{<:VecNum} = nothing,
                     nrm::Option{<:AbstractEdgeNormalisation} = nothing,
                     dir::AbstractEdgeDirection = OutEdges(),
                     unknown::AbstractUnknownEntryRule = EntrywiseUnknown())
    parts = collect(gs)
    n = length(parts)
    @argcheck(!isempty(parts),
              IsEmptyError("a union of graphs needs one part or more, each an ExogenousGraph or an ExogenousGraphSeries"))
    @argcheck(!(isone(n) && isnothing(nrm)),
              ArgumentError("a union of one graph with `nrm = nothing` is the graph itself. Use the graph, or give a normalisation such as `nrm = WeightShare()` for a normalised graph"))
    assert_union_weights(w, n)
    union_parts_in_force(unknown, trues(n))
    keep = union_kept(w, n)
    kept, wk = parts[keep], union_weights(w, keep)
    assert_union_key_kinds(kept, string.(keep))
    if !any(g -> isa(g, ExogenousGraphSeries), kept)
        return union_snapshot(kept, wk, nrm, dir)
    end
    ks = union_series_keys(kept, unknown)
    return ExogenousGraphSeries(ks,
                                [union_at_key(kept, wk, nrm, dir, unknown, k) for k in ks])
end
"""
    union_in_force(gu::GraphUnion, parts::AbstractVector, snaps::AbstractVector, dir::AbstractEdgeDirection) -> ExogenousGraph

Build the union that a consumer reads at the last row of its window, from the snapshot of each part in force there.

A fit reads one row, so it builds the union of one combination of snapshots, and never the whole series.

# Algorithm

 1. Keep the parts with a positive weight, [`union_kept`](@ref).
 2. Choose the parts to sum under `gu.unknown` with [`union_parts_in_force`](@ref), from the parts that have a graph in force.
 3. Refuse when the rule finds no graph in force. Otherwise build the union with [`union_snapshot`](@ref).

# Arguments

  - `gu`: The union.
  - `parts`: The graph or the series of each name of `gu.ng`.
  - `snaps`: The `(graph, key)` pair of each part at the last row, from [`union_part_at`](@ref).
  - `dir`: The single direction of the consumer.

# Validation

  - The rule finds a graph in force at the last row. Raises an `ArgumentError` that names the union, the rule and each part with no graph in force, with its first key or its key that holds `nothing`.

# Returns

  - `g::ExogenousGraph`: The union.

# Related

  - [`consumer_graph`](@ref)
  - [`GraphUnion`](@ref)
"""
function union_in_force(gu::GraphUnion, parts::AbstractVector, snaps::AbstractVector,
                        dir::AbstractEdgeDirection)
    keep = union_kept(gu.w, length(gu.ng))
    idx = union_parts_in_force(gu.unknown, [!isnothing(first(snaps[p])) for p in keep])
    if isnothing(idx)
        lost = join(("the part \"$(gu.ng[p])\" $(union_part_reason(parts[p], last(snaps[p])))"
                     for p in keep if isnothing(first(snaps[p]))), "; ")
        throw(ArgumentError("no graph of the Graph Union \"$(graph_entry_label(gu))\" is in force at the last row of the window under $(nameof(typeof(gu.unknown)))(): $lost. A fit reads the graph in force at the last row of its window. Start the window at or after a key at which the parts hold a graph"))
    end
    sel = keep[idx]
    return union_snapshot([first(snaps[p]) for p in sel], union_weights(gu.w, sel), gu.nrm,
                          dir)
end
"""
    union_part_reason(s::ExogenousGraphSeries, key) -> String

State why a part of a union has no graph in force at a row, in an error message: the row comes before the first key of the part, or the key in force holds `nothing`.

# Related

  - [`union_in_force`](@ref)
"""
function union_part_reason(s::ExogenousGraphSeries, key)
    return if isnothing(key)
        "comes before its first key $(first(s.ts))"
    else
        "holds `nothing` at its key $key"
    end
end
function consumer_graph(consumer, ng::Option{<:VecStr}, G::Option{<:AbstractVector},
                        gu::GraphUnion, T, ts, dir::AbstractEdgeDirection)
    ks = [graph_position(ng, name) for name in gu.ng]
    parts = [G[k] for k in ks]
    snaps = [union_part_at(p, union_row_key(p, T, ts)) for p in parts]
    for p in union_kept(gu.w, length(ks))
        g, key = snaps[p]
        if !isnothing(g)
            assert_graph_requirements(consumer, g, ng[ks[p]], key, "")
        end
    end
    return union_in_force(gu, parts, snaps, dir)
end
function assert_consumer_entry(consumer, gu::GraphUnion, ng, G,
                               path::AbstractString)::Nothing
    ks = [graph_position(ng, name) for name in gu.ng]
    keep = union_kept(gu.w, length(ks))
    for p in keep
        assert_consumer_entry(consumer, gu.ng[p], ng, G, path)
    end
    assert_union_key_kinds(G[ks[keep]], ["\"$(gu.ng[p])\"" for p in keep])
    return nothing
end

export GraphUnion, graph_union
public union_parts_in_force
