#=
function plot_ptf_cumulative_returns(w::ArrNum, X::MatNum,
                                     fees::Option{<:Fees} = nothing;
                                     ts::AbstractVector = 1:size(X, 1),
                                     f::Option{<:Figure} = Figure(),
                                     fpos::Tuple = (1, 1), compound::Bool = false,
                                     ax_kwargs::NamedTuple = (; xlabel = "Date",
                                                              ylabel = "$(compound ? "Compound " : "Simple ") Portfolio Cumulative ReturnsResult"),
                                     line_kwargs::NamedTuple = (; label = "Portfolio"),
                                     legend_kwargs::NamedTuple = (; position = :lt,
                                                                  merge = true))
    ax = Axis(f[fpos...]; ax_kwargs...)
    ret = cumulative_returns(calc_net_returns(w, X, fees), compound)
    lines!(ax, ts, ret; line_kwargs...)
    axislegend(; legend_kwargs...)
    return f
end
function compute_relevant_assets(w::AbstractVector, M::Number, N::Number)
    abs_w = abs.(w)
    idx = sortperm(abs_w; rev = true)
    abs_w /= sum(abs_w)
    N = if one(N) >= N > zero(N)
        cw = cumsum(view(abs_w, idx))
        N = findfirst(x -> one(x) - x < N, cw)
        isnothing(N) ? M : N
    else
        clamp(ceil(Int, N), 1, M)
    end
    return N, idx
end
function plot_asset_cumulative_returns(w::AbstractVector, X::MatNum,
                                       fees::Option{<:Fees} = nothing;
                                       ts::AbstractVector = 1:size(X, 1),
                                       nx::AbstractVector = 1:size(X, 2),
                                       N::Option{<:Number} = nothing,
                                       f::Option{<:Figure} = Figure(),
                                       fpos::Tuple = (1, 1), compound::Bool = false,
                                       ax_kwargs::NamedTuple = (; xlabel = "Date",
                                                                ylabel = "$(compound ? "Compound " : "Simple ") Asset Cumulative ReturnsResult"),
                                       asset_kwargs::NamedTuple = (;),
                                       line_kwargs::NamedTuple = (; label = "Others"),
                                       legend_kwargs::NamedTuple = (; position = :lt,
                                                                    merge = true))
    ax = Axis(f[fpos...]; ax_kwargs...)
    ret = cumulative_returns(calc_net_asset_returns(w, X, fees), compound)
    M = size(X, 2)
    N, idx = compute_relevant_assets(w, M, isnothing(N) ? inv(LinearAlgebra.dot(w, w)) : N)
    ret = view(ret, :, idx)
    nx = view(nx, idx)
    for i in 1:N
        if !haskey(asset_kwargs, :label)
            asset_kwargs_i = (; label = nx[i], asset_kwargs...)
        end
        lines!(ax, ts, view(ret, :, i); asset_kwargs_i...)
    end
    if M > N
        idx = view(idx, (N + 1):M)
        ret = cumulative_returns(calc_net_returns(view(w, idx), view(X, :, idx),
                                                  fees_view(fees, idx)),
                                 compound)
        lines!(ax, ts, ret; line_kwargs...)
    end
    axislegend(; legend_kwargs...)
    return f
end
function plot_composition(w::VecNum, nx::AbstractVector = 1:length(w);
                          N::Option{<:Number} = nothing,
                          f::Option{<:Figure} = Figure(), fpos::Tuple = (1, 1),
                          ax_kwargs::NamedTuple = (; xlabel = "Asset", ylabel = "Weight",
                                                   title = "Portfolio Composition",
                                                   xticklabelrotation = pi / 3),
                          bar_kwargs::NamedTuple = (;))
    M = length(w)
    N, idx = compute_relevant_assets(w, M, isnothing(N) ? inv(LinearAlgebra.dot(w, w)) : N)
    if M > N
        sort!(view(idx, 1:N))
        fidx = view(idx, 1:N)
        ax = Axis(f[fpos...]; xticks = xticks = (1:(N + 1), [nx[fidx]; "Others"]),
                  ax_kwargs...)
        w = [view(w, fidx); sum(view(w, view(idx, (N + 1):M)))]
        barplot!(ax, 1:(N + 1), w; bar_kwargs...)
    else
        ax = Axis(f[fpos...]; xticks = (1:length(w), nx), ax_kwargs...)
        barplot!(ax, 1:length(w), w; bar_kwargs...)
    end
    return f
end
function plot_stacked_bar_composition(w::ArrNum, nx::AbstractVector = 1:size(w, 1);
                                      f::Option{<:Figure} = Figure(),
                                      fpos::Tuple = (1, 1), lpos::Tuple = (1, 2),
                                      ax_kwargs::NamedTuple = (; xlabel = "Portfolios",
                                                               ylabel = "Weight",
                                                               xticks = (1:size(w, 2),
                                                                         string.(1:size(w,
                                                                                        2))),
                                                               title = "Portfolio Composition"),
                                      bar_kwargs::NamedTuple = (; colormap = :viridis))
    if isa(w, VecVecNum)
        w = hcat(w...)
    end
    ax = Axis(f[fpos...]; ax_kwargs...)
    N = size(w, 1)
    M = size(w, 2)
    category = collect(Iterators.flatten([Iterators.repeated(i, N) for i in 1:M]))
    height = vec(w)
    group = collect(Iterators.flatten(Iterators.repeated(1:N, M)))
    cmap = !haskey(bar_kwargs, :colormap) ? :viridis : bar_kwargs.colormap
    colors = try
        resample_cmap(cmap, N)
    catch err
        Makie.categorical_colors(cmap, N)
    end
    barplot!(ax, category, height; stack = group, color = group, bar_kwargs...)
    elements = [PolyElement(; polycolor = colors[i]) for i in 1:length(nx)]
    Legend(f[lpos...], elements, string.(nx), "Assets")
    return f
end
function plot_stacked_area_composition(w::ArrNum, nx::AbstractVector = 1:size(w, 1);
                                       f::Option{<:Figure} = Figure(),
                                       fpos::Tuple = (1, 1), lpos::Tuple = (1, 2),
                                       ax_kwargs::NamedTuple = (; xlabel = "Portfolios",
                                                                ylabel = "Weight",
                                                                xticks = (1:size(w, 2),
                                                                          string.(1:size(w,
                                                                                         2))),
                                                                title = "Portfolio Composition"),
                                       band_kwargs::NamedTuple = (; colormap = :viridis))
    if isa(w, VecVecNum)
        w = hcat(w...)
    end
    cw = cumsum(w; dims = 1)
    ax = Axis(f[fpos...]; ax_kwargs...)
    N = size(w, 1)
    M = size(w, 2)
    cmap = !haskey(band_kwargs, :colormap) ? :viridis : band_kwargs.colormap
    colors = try
        resample_cmap(cmap, N)
    catch err
        Makie.categorical_colors(cmap, N)
    end
    for i in axes(w, 1)
        if i == 1
            band!(ax, 1:M, zeros(M), cw[i, :]; label = nx[i], color = colors[i],
                  band_kwargs...)
        else
            band!(ax, 1:M, cw[i - 1, :], cw[i, :]; label = nx[i], color = colors[i],
                  band_kwargs...)
        end
    end
    elements = [PolyElement(; polycolor = colors[i]) for i in 1:length(nx)]
    Legend(f[lpos...], elements, string.(nx), "Assets")
    return f
end
#! hcl_nodes(hcl; useheight=false), https://github.com/MakieOrg/Makie.jl/pull/2755/files/7a604e1cc11f71bf53c5052482723fd09bc831af#diff-7e892249e5609d2dd4ad2e0b545126c2a68f263da017f54a38883ea8a142d147
function hcl_nodes(hcl; useheight = false)
    nleaves = length(hcl.order)
    nodes = [Makie.DNode(i, Point2d(x, 0), nothing)
             for (i, x) in enumerate(invperm(hcl.order))]

    for ((m1, m2), height) in zip(eachrow(hcl.merges), hcl.heights)
        m1 = ifelse(m1 < 0, -m1, m1 + nleaves)
        m2 = ifelse(m2 < 0, -m2, m2 + nleaves)
        push!(nodes,
              Makie.find_merge(nodes[m1], nodes[m2];
                               height = ifelse(useheight,
                                               height - max(nodes[m1].position[2],
                                                            nodes[m2].position[2]), 1),
                               index = length(nodes) + 1))
    end

    return nodes
end
function plot_dendrogram(clr::AbstractClusteringResult,
                         nx::AbstractVector = 1:length(clr.clustering.order),
                         f::Option{<:Figure} = Figure(), fpos::Tuple = (1, 1),
                         ax_kwargs::NamedTuple = (; title = "Dendrogram"),
                         node_kwargs::NamedTuple = (; useheight = true),
                         dendrogram_kwargs::NamedTuple = (; colormap = :seaborn_colorblind,
                                                          groups = cutree(clr.clustering;
                                                                          k = clr.k),
                                                          origin = Point2d(0,
                                                                           clr.clustering.heights[end])))
    N = length(clr.clustering.order)
    nodes = hcl_nodes(clr.clustering; node_kwargs...)
    ax = Axis(f[fpos...]; ax_kwargs...)
    d = dendrogram!(ax, nodes; dendrogram_kwargs...)
    xpos = getindex.(Makie.dendrogram_node_positions(d)[][1:N], 1)
    xticks = (xpos, string.(view(nx, clr.clustering.order)))
    ax.xticks = xticks
    return f
end
function plot_clusters(clr::AbstractClusteringResult, X::MatNum,
                       nx::AbstractVector = 1:size(X, 1);
                       f::Option{<:Figure} = Figure(),
                       ax_kwargs::NamedTuple = (; yreversed = true,
                                                xticklabelrotation = pi / 2, aspect = 1),
                       color_func = x -> if any(x .< zero(eltype(x)))
                           (-1, 1)
                       else
                           (0, 1)
                       end, heatmap_kwargs::NamedTuple = (; colormap = :viridis),
                       lines_kwargs::NamedTuple = (color = :black, linewidth = 3),
                       node_kwargs::NamedTuple = (; useheight = true),
                       dendrogram_kwargs::NamedTuple = (; colormap = :seaborn_colorblind,
                                                        groups = cutree(clr.clustering;
                                                                        k = clr.k),
                                                        origin = Point2d(0,
                                                                         clr.clustering.heights[end])))
    assert_matrix_issquare(X)
    iscov = any(!isone, LinearAlgebra.diag(X))
    if iscov
        X = cov2cor(X)
    end
    colorrange = color_func(X)
    N = size(X, 1)
    X = view(X, clr.clustering.order, clr.clustering.order)
    nx = view(nx, clr.clustering.order)
    ticks = (1:size(X, 1), string.(nx))
    idx = cutree(clr.clustering; k = clr.k)
    cls = [findall(x -> x == i, idx) for i in 1:(clr.k)]
    ax = Axis(f[2, 2]; yticks = ticks, xticks = ticks, ax_kwargs...)
    heatmap!(ax, 1:N, 1:N, X; colorrange = colorrange, heatmap_kwargs...)
    for (i, cl) in pairs(cls)
        a = [findfirst(x -> x == c, clr.clustering.order) for c in cl]
        a = a[.!isnothing.(a)]
        xmin = minimum(a)
        xmax = xmin + length(cl)
        i1 = [findfirst(x -> x == c, -clr.clustering.merges[:, 1]) for c in cl]
        i1 = i1[.!isnothing.(i1)]
        i2 = [findfirst(x -> x == c, -clr.clustering.merges[:, 2]) for c in cl]
        i2 = i2[.!isnothing.(i2)]
        i3 = unique([i1; i2])
        h = min(maximum(clr.clustering.heights[i3]) * 1.1, 1)
        lines!(ax,
               [xmin - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5, xmin - 0.5,
                xmin - 0.5, xmin - 0.5],
               [xmin - 0.5, xmin - 0.5, xmin - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5,
                xmax - 0.5, xmin - 0.5]; lines_kwargs...)
    end
    Colorbar(f[2, end + 1]; colorrange = colorrange,
             colormap = if haskey(heatmap_kwargs, :colormap)
                 heatmap_kwargs.colormap
             else
                 :viridis
             end)

    nodes = hcl_nodes(clr.clustering; node_kwargs...)
    ax2 = Axis(f[1, 2]; xticklabelsvisible = false)
    d = dendrogram!(ax2, nodes; dendrogram_kwargs...)
    xpos = getindex.(Makie.dendrogram_node_positions(d)[][1:N], 1)

    nodes = hcl_nodes(clr.clustering; node_kwargs...)
    ax3 = Axis(f[2, 1]; xreversed = true, xticklabelsvisible = false)
    d = dendrogram!(ax3, nodes; rotation = :left, dendrogram_kwargs...)
    xpos = getindex.(Makie.dendrogram_node_positions(d)[][1:N], 1)
    return f
end

export plot_asset_cumulative_returns, plot_ptf_cumulative_returns, plot_composition,
       plot_stacked_bar_composition, plot_stacked_area_composition, plot_clusters,
       plot_dendrogram
       =#
