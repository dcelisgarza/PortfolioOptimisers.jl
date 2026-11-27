## plot_exogenous_graph
# A layered layout: an edge of a graph with no cycle goes to the right, one layer or more. A
# cycle keeps one edge of each loop that points back. `graphplot` of GraphRecipes has no
# layered layout, and GraphRecipes invalidates much of the compiled code of a session (#2186).
#
# The nodes in an order with no edge to an earlier node, except the edges that close a cycle.
# It is the reverse of the order in which a depth-first search leaves the nodes.
function graph_node_order(outs::AbstractVector{<:VecInt}, roots::VecInt)
    seen = falses(length(outs))
    left = Int[]
    stack = Tuple{Int, Int}[]
    for r in roots
        if seen[r]
            continue
        end
        seen[r] = true
        push!(stack, (r, 1))
        while !isempty(stack)
            v, k = pop!(stack)
            if k > length(outs[v])
                push!(left, v)
                continue
            end
            push!(stack, (v, k + 1))
            u = outs[v][k]
            if !seen[u]
                seen[u] = true
                push!(stack, (u, 1))
            end
        end
    end
    return reverse!(left)
end
# The layer of a node is the number of nodes of the longest path that ends at it, over the
# edges that close no cycle. A source then moves to the layer before its nearest target, so its
# edges are short.
function graph_layers(outs::AbstractVector{<:VecInt}, ins::AbstractVector{<:VecInt})
    n = length(outs)
    order = graph_node_order(outs, sortperm([!isempty(ins[v]) for v in 1:n]))
    pos = invperm(order)
    layer = ones(Int, n)
    for v in order, u in outs[v]
        if pos[u] > pos[v]
            layer[u] = max(layer[u], layer[v] + 1)
        end
    end
    for v in 1:n
        if isempty(ins[v]) && !isempty(outs[v])
            layer[v] = minimum(view(layer, outs[v])) - 1
        end
    end
    return layer
end
# Order each layer by the mean height of the neighbours of its nodes, forward and back. This
# lowers the number of edges that cross.
function order_layers!(y::AbstractVector{<:Real}, layers::AbstractVector{<:VecInt},
                       outs::AbstractVector{<:VecInt}, ins::AbstractVector{<:VecInt},
                       sweeps::Integer)
    for s in 1:sweeps, l in (isodd(s) ? eachindex(layers) : reverse(eachindex(layers)))
        vs = layers[l]
        if length(vs) > 1
            sort!(vs; by = v -> mean(u -> y[u], Iterators.flatten((ins[v], outs[v]))),
                  rev = true)
        end
        place_layer!(y, vs)
    end
    return y
end
# One layer is centred on height zero, its first node at the top.
function place_layer!(y::AbstractVector{<:Real}, vs::VecInt)
    for (k, v) in enumerate(vs)
        y[v] = (length(vs) + 1) / 2 - k
    end
    return y
end
# The weakly connected components, in the order of their first nodes. A depth-first search over
# the neighbours of both directions leaves the nodes of one component.
function graph_components(nbrs::AbstractVector{<:VecInt})
    seen = falses(length(nbrs))
    comps = Vector{Int}[]
    for r in eachindex(nbrs)
        if !seen[r]
            comp = sort!(graph_node_order(nbrs, r:r))
            foreach(v -> seen[v] = true, comp)
            push!(comps, comp)
        end
    end
    return comps
end
# The nodes of a component in each of its layers.
function component_layers(comp::VecInt, layer::VecInt)
    layers = [Int[] for _ in 1:maximum(view(layer, comp))]
    for v in comp
        push!(layers[layer[v]], v)
    end
    return layers
end
# The coordinates of the nodes. Each weakly connected component has a layout of its own, and
# the components stack from the top, in the order of their first nodes.
function layered_layout(g::ExogenousGraph; sweeps::Integer = 8)
    D = Graphs.SimpleDiGraph(g)
    n = length(g.nodes)
    # A self-loop states no position.
    outs = [filter(!=(v), Graphs.outneighbors(D, v)) for v in 1:n]
    ins = [filter(!=(v), Graphs.inneighbors(D, v)) for v in 1:n]
    layer = graph_layers(outs, ins)
    y = zeros(n)
    top = 0.0
    for comp in graph_components(vcat.(outs, ins))
        layers = component_layers(comp, layer)
        foreach(vs -> place_layer!(y, vs), layers)
        order_layers!(y, layers, outs, ins, sweeps)
        h = maximum(length, layers)
        for v in comp
            y[v] += top - (h - 1) / 2
        end
        top -= h
    end
    return layer, y
end
# The points of one edge. An edge between two neighbouring layers is straight. Any other edge
# bows to its left, so a reciprocal pair splits and an edge over a layer passes its nodes. Both
# ends stop short of the node. A self-loop is a ring above its node.
function graph_edge_path(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, s::Integer,
                         t::Integer, reciprocal::Bool)
    if s == t
        a = range(-pi / 2, 3 * pi / 2; length = 25)
        return x[s] .+ 0.08 .* cos.(a), y[s] .+ 0.16 .* (1 .+ sin.(a))
    end
    dx, dy = x[t] - x[s], y[t] - y[s]
    len = hypot(dx, dy)
    # The top of a bow is at most three quarters of a row from the line of its ends, so every
    # edge stays in the figure.
    bow = dx == 1 && !reciprocal ? 0.0 : min(0.2, 1.5 / len)
    gap = min(0.2, 0.12 / len)
    k = range(gap, 1 - gap; length = iszero(bow) ? 2 : 25)
    return x[s] .+ k .* dx .- (2 * bow * dy) .* k .* (1 .- k),
           y[s] .+ k .* dy .+ (2 * bow * dx) .* k .* (1 .- k)
end
# The edges draw as one series for each sign and each of four widths, so a graph of thousands
# of edges is a few series. A `NaN` splits two edges of one series.
function plot_graph_edges!(plt, g::ExogenousGraph, x::AbstractVector{<:Real},
                           y::AbstractVector{<:Real})
    I, J, V = stored_entries(g.At)
    peak = maximum(abs, V; init = zero(eltype(V)))
    edges = Set(zip(J, I))
    for negative in (false, true), width in 1:4
        ex = Float64[]
        ey = Float64[]
        for (t, s, v) in zip(I, J, V)
            if signbit(v) != negative || clamp(ceil(Int, 4 * abs(v) / peak), 1, 4) != width
                continue
            end
            px, py = graph_edge_path(x, y, s, t, (t, s) in edges)
            append!(ex, px)
            append!(ey, py)
            push!(ex, NaN)
            push!(ey, NaN)
        end
        if !isempty(ex)
            plot!(plt, ex, ey; arrow = true, linewidth = 0.6 * width, linealpha = 0.6,
                  color = negative ? :firebrick : :gray40, label = "")
        end
    end
    return plt
end
# The nodes of each group, in the order of the names of the groups. `nothing` is one group.
function graph_node_groups(g::ExogenousGraph, ::Nothing)
    return ["" => collect(eachindex(g.nodes))]
end
function graph_node_groups(g::ExogenousGraph, groups::AbstractDict)
    labels = [string(get(groups, node, "Ungrouped")) for node in g.nodes]
    return [label => findall(==(label), labels) for label in sort!(unique(labels))]
end
function PortfolioOptimisers.plot_exogenous_graph(g::ExogenousGraph;
                                                  groups::Option{<:AbstractDict} = nothing,
                                                  max_nodes::Integer = 500,
                                                  max_edges::Integer = 5000, kwargs...)
    n = length(g.nodes)
    m = length(last(stored_entries(g.At)))
    @argcheck(n <= max_nodes && m <= max_edges,
              ArgumentError("the figure draws at most $max_nodes nodes and $max_edges edges, and the graph has $n nodes and $m edges. Raise `max_nodes` or `max_edges`, or draw a part of the graph"))
    x, y = layered_layout(g)
    hi = maximum(y) + 0.8
    lo = minimum(y) - 0.8
    plt = plot(; title = "Exogenous Graph", framestyle = :none, grid = false,
               legend = isnothing(groups) ? false : :outerright,
               xlims = (0.5, maximum(x) + 0.5), ylims = (lo, hi),
               size = (700, clamp(round(Int, 60 * (hi - lo)), 220, 1600)), kwargs...)
    plot_graph_edges!(plt, g, x, y)
    for (label, vs) in graph_node_groups(g, groups)
        # The nodes of no group are gray, as the edges are.
        colour = label == "Ungrouped" ? (; color = :gray60) : (;)
        scatter!(plt, x[vs], y[vs]; label = label, markersize = 6, markerstrokewidth = 0.5,
                 colour...)
    end
    # The labels draw last, below the nodes.
    annotate!(plt, [(x[v], y[v] - 0.28, (g.nodes[v], 8, :center)) for v in 1:n])
    return plt
end
# The assets and the other nodes, the default groups of a graph of returns data.
function graph_asset_groups(g::ExogenousGraph, nx::AbstractVector)
    return Dict(node => node in nx ? "Assets" : "Other nodes" for node in g.nodes)
end
function plot_returns_graph(g::ExogenousGraph, rd::ReturnsResult, title::AbstractString;
                            groups::Option{<:AbstractDict} = graph_asset_groups(g, rd.nx),
                            kwargs...)
    return PortfolioOptimisers.plot_exogenous_graph(g; groups = groups, title = title,
                                                    kwargs...)
end
function PortfolioOptimisers.plot_exogenous_graph(rd::ReturnsResult,
                                                  name::Option{<:AbstractString} = nothing;
                                                  kwargs...)
    title = isnothing(name) ? "Exogenous Graph" : "Exogenous Graph: $name"
    return plot_returns_graph(graph_in_force(rd, name), rd, title; kwargs...)
end
function PortfolioOptimisers.plot_exogenous_graph(rd::ReturnsResult, gu::GraphUnion;
                                                  kwargs...)
    return plot_returns_graph(consumer_graph(nothing, rd, gu, OutEdges()), rd,
                              "Exogenous Graph: $(graph_entry_label(gu))"; kwargs...)
end
