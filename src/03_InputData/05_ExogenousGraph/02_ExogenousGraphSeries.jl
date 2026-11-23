"""
    assert_graph_keys(ts::AbstractVector) -> nothing

Refuse the keys of an [`ExogenousGraphSeries`](@ref) that are not strictly increasing positions or timestamps of one type.

# Arguments

  - `ts`: The keys of the series.

# Validation

  - `!isempty(ts)`. Raises an [`IsEmptyError`](@ref).
  - The keys are of one type: integer positions, or timestamps. A vector whose element type is neither, such as a vector that mixes positions and dates, raises an `ArgumentError`.
  - An integer key is 1 or more. Raises a `DomainError`.
  - Each key is greater than the key before it. Raises an `ArgumentError` that names the first pair out of order.

# Returns

  - `nothing`.

# Related

  - [`ExogenousGraphSeries`](@ref)
"""
function assert_graph_keys(ts::AbstractVector)::Nothing
    @argcheck(!isempty(ts),
              IsEmptyError("an Exogenous Graph Series needs one key or more. A graph that does not change is a static ExogenousGraph"))
    T = eltype(ts)
    @argcheck((T <: Integer && !(T <: Bool)) || T <: Dates.AbstractTime,
              ArgumentError("the keys of an Exogenous Graph Series are 1-based row positions or timestamps of one type, and they cannot mix the two kinds. Got keys of the element type $T"))
    if T <: Integer
        @argcheck(all(>=(1), ts),
                  DomainError(minimum(ts),
                              "an integer key of an Exogenous Graph Series is a 1-based row position, so each key must be 1 or more"))
    end
    k = findfirst(i -> !(ts[i] < ts[i + 1]), firstindex(ts):(lastindex(ts) - 1))
    @argcheck(isnothing(k),
              ArgumentError("the keys of an Exogenous Graph Series must be strictly increasing, because each graph is in force from its key to the next key. Got the key $(isnothing(k) ? "" : ts[k + 1]) after the key $(isnothing(k) ? "" : ts[k])"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Holds a point-in-time series of Exogenous Graphs as a list of change points over the observation axis.

The graph at a key is in force from that key to the next key, and the last graph is in force to the end of the observation axis. A key is a 1-based row position or a timestamp. Without the timestamps `ts` of the [`ReturnsResult`](@ref), only positions are valid. With them, dates and positions are both valid. A key can hold `nothing`, which ends the graph before it, so no graph is in force from that key to the next key. A graph that does not change is a static [`ExogenousGraph`](@ref), not a series.

A consumer reads the graph in force at the last row of its window, with [`graph_in_force`](@ref). A view of the returns data on some of its rows keeps the graph in force at its first row and the keys inside its window, with [`graph_view`](@ref), so a training fold holds no graph of its test window. The snapshots can have different nodes.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExogenousGraphSeries(;
        ts::AbstractVector,
        graphs::AbstractVector{<:Option{<:ExogenousGraph}},
    ) -> ExogenousGraphSeries

Keywords correspond to the struct's fields.

## Validation

  - The rules of [`assert_graph_keys`](@ref): `ts` is not empty, holds positions or timestamps of one type, holds no integer key below 1, and is strictly increasing.
  - `length(ts) == length(graphs)`. Raises a `DimensionMismatch`.

A [`ReturnsResult`](@ref) and a [`PricesResult`](@ref) check the keys against their rows, see [`check_exogenous_graphs`](@ref).

# Examples

```jldoctest
julia> g1 = ExogenousGraph(; edges = [(\"S\", \"A\", 0.4)]);

julia> g2 = ExogenousGraph(; edges = [(\"S\", \"A\", 0.5), (\"S\", \"B\", 0.5)]);

julia> s = ExogenousGraphSeries(; ts = [1, 4, 6], graphs = [g1, g2, nothing]);

julia> s.ts
3-element Vector{Int64}:
 1
 4
 6
```

# Related

  - [`ExogenousGraph`](@ref)
  - [`ExG_ExGS`](@ref)
  - [`graph_in_force`](@ref)
  - [`graph_view`](@ref)
  - [`ReturnsResult`](@ref)
"""
@concrete struct ExogenousGraphSeries <: AbstractResult
    """
    Keys of the change points (keys × 1): strictly increasing 1-based row positions, or strictly increasing timestamps.
    """
    ts
    """
    Graphs of the change points (keys × 1). The graph at `ts[k]` is in force from `ts[k]` to `ts[k + 1]`. `nothing` states that no graph is in force.
    """
    graphs
    function ExogenousGraphSeries(ts::AbstractVector,
                                  graphs::AbstractVector{<:Option{<:ExogenousGraph}})
        assert_graph_keys(ts)
        @argcheck(length(ts) == length(graphs),
                  DimensionMismatch("an Exogenous Graph Series holds one graph or `nothing` for each key, so length(ts) == length(graphs) must hold, got $(length(ts)) and $(length(graphs))"))
        return new{typeof(ts), typeof(graphs)}(ts, graphs)
    end
end
function ExogenousGraphSeries(; ts::AbstractVector,
                              graphs::AbstractVector{<:Option{<:ExogenousGraph}})::ExogenousGraphSeries
    return ExogenousGraphSeries(ts, graphs)
end
"""
    const ExG_ExGS = Union{ExogenousGraph, ExogenousGraphSeries}

Alias for the two forms of an Exogenous Graph that the returns data holds.

Matches a static [`ExogenousGraph`](@ref) or an [`ExogenousGraphSeries`](@ref).

The field `G` of a [`ReturnsResult`](@ref) and of a [`PricesResult`](@ref) holds one of the two forms for each name of `ng`. The family is closed: a graph is in force at each row, or it changes at its keys. [`graph_in_force`](@ref), [`graph_view`](@ref) and [`graph_vcat`](@ref) dispatch on the two members.

# Related

  - [`ExogenousGraph`](@ref)
  - [`ExogenousGraphSeries`](@ref)
  - [`check_exogenous_graphs`](@ref)
"""
const ExG_ExGS = Union{ExogenousGraph, ExogenousGraphSeries}
"""
    isequal(a::ExogenousGraphSeries, b::ExogenousGraphSeries) -> Bool
    ==(a::ExogenousGraphSeries, b::ExogenousGraphSeries) -> Bool
    hash(s::ExogenousGraphSeries, h::UInt) -> UInt

Compare two [`ExogenousGraphSeries`](@ref) by value: the same keys, and the same graph or `nothing` at each key.

# Arguments

  - `a`, `b`, `s`: The series.
  - `h`: The hash to mix in.

# Returns

  - `tf::Bool`: `true` when the keys and the graphs are equal. `hash` returns a `UInt`.

# Related

  - [`ExogenousGraphSeries`](@ref)
  - [`ExogenousGraph`](@ref)
"""
function Base.isequal(a::ExogenousGraphSeries, b::ExogenousGraphSeries)
    return isequal(a.ts, b.ts) && isequal(a.graphs, b.graphs)
end
function Base.:(==)(a::ExogenousGraphSeries, b::ExogenousGraphSeries)
    return a.ts == b.ts &&
           length(a.graphs) == length(b.graphs) &&
           all(k -> isequal(a.graphs[k], b.graphs[k]), eachindex(a.graphs, b.graphs))
end
function Base.hash(s::ExogenousGraphSeries, h::UInt)
    return hash(s.graphs, hash(s.ts, hash(:ExogenousGraphSeries, h)))
end

export ExogenousGraphSeries
