## plot_network
function PortfolioOptimisers.plot_network(pl::NwE_ClE_Cl, X::MatNum,
                                          nx::AbstractVector = 1:size(X, 2),
                                          w::Option{<:VecNum} = nothing;
                                          threshold::Number = 0, kwargs...)
    plr = phylogeny_matrix(pl, X)
    A = copy(plr.X)
    A[abs.(A) .<= threshold] .= 0
    node_size = if isnothing(w)
        fill(1, size(X, 2))
    else
        abs.(w) ./ maximum(abs, w)
    end
    return graphplot(A; names = nx, node_weights = node_size, title = "Asset Network",
                     kwargs...)
end
function PortfolioOptimisers.plot_network(pl::NwE_ClE_Cl, pr::Pr_RR,
                                          w::Option{<:VecNum} = nothing;
                                          nx::AbstractVector = 1:size(pr.X, 2), kwargs...)
    if isa(pr, ReturnsResult) && !isnothing(pr.nx)
        nx = pr.nx
    end
    pr_i, nx_i, w_i = investable_plot_view(pr, nx, w)
    return PortfolioOptimisers.plot_network(pl, pr_i.X, nx_i, w_i; kwargs...)
end
function PortfolioOptimisers.plot_network(pl::NwE_ClE_Cl, res::OptimisationResult;
                                          rd::Option{<:Pr_RR} = nothing,
                                          nx::AbstractVector = 1:length(res.w), kwargs...)
    # The prior arity's own door sees a viewed prior, whose mask is already `nothing`, so
    # the pairing of the result's two universes happens here.
    _, w, pr, _, nx = result_investable_view(res, rd, nothing, nx)
    return PortfolioOptimisers.plot_network(pl, pr, w; nx = nx, kwargs...)
end
## plot_dendrogram
function PortfolioOptimisers.plot_dendrogram(clr::AbstractClusteringResult,
                                             nx::AbstractVector = 1:length(clr.res.order);
                                             dend_theme::Symbol = :Spectral, kwargs...)
    N = length(clr.res.order)
    nx_ord = view(nx, clr.res.order)
    idx = assignments(clr)
    cls = [findall(x -> x == i, idx) for i in 1:(clr.k)]
    colours = palette(dend_theme, clr.k)
    dend = plot(clr.res; normalize = false, ylim = extrema(clr.res.heights),
                xticks = (1:N, nx_ord), xrotation = 90)
    for (i, cl) in pairs(cls)
        a = Iterators.filter(!isnothing, (findfirst(==(c), clr.res.order) for c in cl))
        if isempty(a)
            continue
        end
        xmin = minimum(a)
        xmax = xmin + length(cl)
        i3 = Iterators.filter(!isnothing,
                              (findfirst(==(-c), view(clr.res.merges, :, k)) for k in 1:2
                               for c in cl))
        if isempty(i3)
            continue
        end
        h = min(maximum(j -> clr.res.heights[j], i3) * 1.1, 1)
        plot!(dend,
              [xmin - 0.25, xmax - 0.75, xmax - 0.75, xmax - 0.75, xmax - 0.75, xmin - 0.25,
               xmin - 0.25, xmin - 0.25], [0, 0, 0, h, h, h, h, 0]; color = nothing,
              legend = false, fill = (0, 0.5, colours[mod1(i, clr.k)]))
    end
    return plot(dend; size = (600, 600), kwargs...)
end
function PortfolioOptimisers.plot_dendrogram(cle::HClE_HCl, X::MatNum,
                                             nx::AbstractVector = 1:size(X, 2);
                                             dims::Integer = 1, kwargs...)
    clr = clusterise(cle, X; dims = dims)
    return PortfolioOptimisers.plot_dendrogram(clr, nx; kwargs...)
end
function PortfolioOptimisers.plot_dendrogram(cle::HClE_HCl, pr::PortfolioOptimisers.Pr_RR,
                                             nx::AbstractVector = 1:size(pr.X, 2);
                                             kwargs...)
    if isa(pr, ReturnsResult) && !isnothing(pr.nx)
        nx = pr.nx
    end
    # A clustering is fitted by a plain moment estimator, which refuses a gapped sample, so
    # the prior reduces to the Investable Mask first, as `plot_network` does.
    pr_i, nx_i = investable_plot_view(pr, nx)
    return PortfolioOptimisers.plot_dendrogram(cle, pr_i.X, nx_i; kwargs..., dims = 1)
end
## plot_clusters
function PortfolioOptimisers.plot_clusters(clr::AbstractClusteringResult,
                                           nx::AbstractVector = 1:size(clr.S, 1);
                                           dend_theme::Symbol = :Spectral,
                                           hmap_theme::Symbol = :Spectral,
                                           color_func = x -> if any(<(zero(eltype(x))), x)
                                               (-1, 1)
                                           else
                                               (0, 1)
                                           end, line_color = :black, line_width = 3,
                                           kwargs...)
    S = clr.S
    s = LinearAlgebra.diag(S)
    iscov = any(!isone, s)
    if iscov
        # Copy before rescaling: `cov2cor!` mutates in place, and `clr.S` is the caller's
        # stored result, not ours to rewrite.
        S = copy(S)
        s .= sqrt.(s)
        StatsBase.cov2cor!(S, s)
    end
    clim = color_func(S)
    N = size(S, 1)
    S_ord = view(S, clr.res.order, clr.res.order)
    nx_ord = view(nx, clr.res.order)
    idx = assignments(clr)
    cls = [findall(x -> x == i, idx) for i in 1:(clr.k)]
    colours = palette(dend_theme, clr.k)
    colgrad = cgrad(hmap_theme)
    hmap = plot(S_ord; st = :heatmap, yticks = (1:N, nx_ord), xticks = (1:N, nx_ord),
                xrotation = 90, colorbar = false, clim = clim, xlim = (0.5, N + 0.5),
                ylim = (0.5, N + 0.5), color = colgrad, yflip = true)
    hlim = extrema(clr.res.heights)
    dend1 = plot(clr.res; xticks = false, ylim = hlim)
    dend2 = plot(clr.res; yticks = false, orientation = :horizontal, xrotation = 90,
                 yflip = true, xlim = hlim)
    for (i, cl) in pairs(cls)
        a = Iterators.filter(!isnothing, (findfirst(==(c), clr.res.order) for c in cl))
        if isempty(a)
            continue
        end
        xmin = minimum(a)
        xmax = xmin + length(cl)
        i3 = Iterators.filter(!isnothing,
                              (findfirst(==(-c), view(clr.res.merges, :, k)) for k in 1:2
                               for c in cl))
        if isempty(i3)
            continue
        end
        h = maximum(j -> clr.res.heights[j], i3)
        col_i = colours[mod1(i, clr.k)]
        box_x = [xmin - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5, xmin - 0.5,
                 xmin - 0.5, xmin - 0.5]
        box_y = [xmin - 0.5, xmin - 0.5, xmin - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5,
                 xmax - 0.5, xmin - 0.5]
        dend_rect = [xmin - 0.25, xmax - 0.75, xmax - 0.75, xmax - 0.75, xmax - 0.75,
                     xmin - 0.25, xmin - 0.25, xmin - 0.25]
        dend_h = [0, 0, 0, h, h, h, h, 0]
        plot!(hmap, box_x, box_y; legend = false, color = line_color,
              linewidth = line_width)
        plot!(dend1, dend_rect, dend_h; color = nothing, legend = false,
              fill = (0, 0.5, col_i))
        plot!(dend2, dend_h, dend_rect; color = nothing, legend = false,
              fill = (0, 0.5, col_i))
    end
    l = StatsPlots.grid(2, 2; heights = [0.2, 0.8], widths = [0.8, 0.2])
    return plot(dend1, plot(; ticks = nothing, border = :none, background_color = nothing),
                hmap, dend2; layout = l, size = (600, 600), kwargs...)
end
function PortfolioOptimisers.plot_clusters(cle::HClE_HCl, X::MatNum,
                                           nx::AbstractVector = 1:size(X, 2);
                                           dims::Integer = 1, kwargs...)
    clr = clusterise(cle, X; dims = dims)
    return PortfolioOptimisers.plot_clusters(clr, nx; kwargs...)
end
function PortfolioOptimisers.plot_clusters(cle::HClE_HCl, pr::PortfolioOptimisers.Pr_RR,
                                           nx::AbstractVector = 1:size(pr.X, 2); kwargs...)
    if isa(pr, ReturnsResult) && !isnothing(pr.nx)
        nx = pr.nx
    end
    pr_i, nx_i = investable_plot_view(pr, nx)
    return PortfolioOptimisers.plot_clusters(cle, pr_i.X, nx_i; kwargs..., dims = 1)
end
## plot_centrality
function PortfolioOptimisers.plot_centrality(cte::AbstractCentralityEstimator, X::MatNum,
                                             nx::AbstractVector = 1:size(X, 2);
                                             N::Option{<:Number} = nothing,
                                             percentage::Bool = true, kwargs...)
    plr = centrality_vector(cte, X)
    scores = plr.X
    M = length(scores)
    N, idx = relevant_assets(scores, M, N)
    top_idx = view(idx, 1:N)
    sort!(top_idx; by = i -> scores[i], rev = true)
    if percentage
        scores /= sum(scores)
    end
    return bar(scores[top_idx]; xticks = (1:N, nx[top_idx]), title = "Asset Centrality",
               xlabel = "Asset", ylabel = "Centrality Score", xrotation = 90,
               legend = false, kwargs...)
end
function PortfolioOptimisers.plot_centrality(cte::AbstractCentralityEstimator,
                                             pr::PortfolioOptimisers.AbstractPriorResult,
                                             nx::AbstractVector = 1:size(pr.X, 2);
                                             N::Option{<:Number} = nothing,
                                             percentage::Bool = true, kwargs...)
    pr_i, nx_i = investable_plot_view(pr, nx)
    return PortfolioOptimisers.plot_centrality(cte, pr_i.X, nx_i; N = N,
                                               percentage = percentage, kwargs...)
end
function PortfolioOptimisers.plot_centrality(cte::AbstractCentralityEstimator,
                                             rd::ReturnsResult;
                                             N::Option{<:Number} = nothing,
                                             percentage::Bool = true, kwargs...)
    nx = isnothing(rd.nx) ? (1:size(rd.X, 2)) : rd.nx
    return PortfolioOptimisers.plot_centrality(cte, rd.X, nx; N = N,
                                               percentage = percentage, kwargs...)
end
function PortfolioOptimisers.plot_centrality(cte::AbstractCentralityEstimator,
                                             res::OptimisationResult, rd::ReturnsResult;
                                             N::Option{<:Number} = nothing,
                                             percentage::Bool = true, kwargs...)
    nx = isnothing(rd.nx) ? (1:length(res.w)) : rd.nx
    return PortfolioOptimisers.plot_centrality(cte, rd.X, nx; N = N,
                                               percentage = percentage, kwargs...)
end
