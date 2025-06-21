module PortfolioOptimisersPlotsExt

using PortfolioOptimisers, GraphRecipes, StatsPlots, LinearAlgebra, Statistics, StatsBase,
      Clustering

function PortfolioOptimisers.plot_ptf_cumulative_returns(w::AbstractArray, X::AbstractArray,
                                                         fees::Union{Nothing, <:Fees} = nothing;
                                                         ts::AbstractVector = 1:size(X, 1),
                                                         compound::Bool = false,
                                                         kwargs::NamedTuple = (;
                                                                               title = "Portfolio",
                                                                               xlabel = "Date",
                                                                               ylabel = "$(compound ? "Compound " : "Simple ") Portfolio Cummulative Returns",
                                                                               legend = false))
    ret = cumulative_returns(calc_net_returns(w, X, fees); compound = compound)
    return plot(ts, ret; kwargs...)
end
function compute_relevant_assets(w::AbstractVector, M::Real, N::Real)
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
function PortfolioOptimisers.plot_asset_cumulative_returns(w::AbstractVector,
                                                           X::AbstractArray,
                                                           fees::Union{Nothing, <:Fees} = nothing;
                                                           ts::AbstractVector = 1:size(X,
                                                                                       1),
                                                           nx::AbstractVector = 1:size(X,
                                                                                       2),
                                                           N::Union{Nothing, <:Real} = nothing,
                                                           compound::Bool = false,
                                                           f_kwargs::NamedTuple = (;
                                                                                   xlabel = "Date",
                                                                                   ylabel = "$(compound ? "Compound " : "Simple ") Asset Cummulative Returns"),
                                                           asset_kwargs::NamedTuple = (;),
                                                           summary_kwargs::NamedTuple = (;
                                                                                         label = "Others"),
                                                           legend_kwargs::NamedTuple = (;
                                                                                        position = :auto))
    ret = cumulative_returns(calc_net_asset_returns(w, X, fees); compound = compound)
    M = size(X, 2)
    N, idx = compute_relevant_assets(w, M, isnothing(N) ? inv(dot(w, w)) : N)
    ret = view(ret, :, idx)
    nx = view(nx, idx)
    f = plot(; f_kwargs...)
    for i ∈ 1:N
        if !haskey(asset_kwargs, :label)
            asset_kwargs_i = (; label = nx[i], asset_kwargs...)
        end
        plot!(f, ts, view(ret, :, i); asset_kwargs_i...)
    end
    if M > N
        idx = view(idx, (N + 1):M)
        ret = cumulative_returns(calc_net_returns(view(w, idx), view(X, :, idx),
                                                  PortfolioOptimisers.fees_view(fees, idx));
                                 compound = compound)
        plot!(f, ts, ret; summary_kwargs...)
    end
    plot!(f; legend_kwargs...)
    return f
end
function PortfolioOptimisers.plot_composition(w::AbstractVector{<:Real},
                                              nx::AbstractVector = 1:length(w);
                                              N::Union{Nothing, <:Real} = nothing,
                                              kwargs::NamedTuple = (title = "Portfolio Composition",
                                                                    xlabel = "Asset",
                                                                    ylabel = "Weight",
                                                                    xrotation = 45,
                                                                    legend = false))
    M = length(w)
    N, idx = compute_relevant_assets(w, M, isnothing(N) ? inv(dot(w, w)) : N)
    return if M > N
        sort!(view(idx, 1:N))
        fidx = view(idx, 1:N)
        w = [view(w, fidx); sum(view(w, view(idx, (N + 1):M)))]
        bar(w; xticks = (1:(N + 1), [nx[fidx]; "Others"]), kwargs...)
    else
        bar(w; xticks = (1:M, nx), kwargs...)
    end
end
function PortfolioOptimisers.plot_stacked_bar_composition(w::AbstractArray,
                                                          nx::AbstractVector = 1:size(w, 1);
                                                          kwargs::NamedTuple = (;
                                                                                xlabel = "Portfolios",
                                                                                ylabel = "Weight",
                                                                                title = "Portfolio Composition",
                                                                                legend = :outerright))
    if isa(w, AbstractVector{<:AbstractVector})
        w = hcat(w...)
    end
    M = size(w, 2)
    ctg = repeat(nx; inner = M)
    return groupedbar(transpose(w); xticks = (1:M, 1:M), bar_position = :stack, group = ctg,
                      kwargs...)
end
function PortfolioOptimisers.plot_stacked_area_composition(w::AbstractArray,
                                                           nx::AbstractVector = 1:size(w,
                                                                                       1);
                                                           kwargs::NamedTuple = (;
                                                                                 xlabel = "Portfolios",
                                                                                 ylabel = "Weight",
                                                                                 title = "Portfolio Composition",
                                                                                 legend = :outerright,
                                                                                 xticks = (1:size(w,
                                                                                                  2),
                                                                                           1:size(w,
                                                                                                  2)),))
    if isa(w, AbstractVector{<:AbstractVector})
        w = hcat(w...)
    end
    return areaplot(transpose(w); label = permutedims(nx), kwargs...)
end
function PortfolioOptimisers.plot_clusters(clr::PortfolioOptimisers.AbstractClusteringResult,
                                           X::AbstractMatrix,
                                           nx::AbstractVector = 1:size(X, 1);
                                           color_func = x -> if any(x .<
                                                                    Ref(zero(eltype(x))))
                                               (-1, 1)
                                           else
                                               (0, 1)
                                           end, theme_d = :Spectral, theme = :Spectral,
                                           theme_kwargs = (;), dend1_kwargs = (;),
                                           dend2_kwargs = (;), hmap_kwargs = (;),
                                           line_kwargs = (; color = :black, linewidth = 3),
                                           fig_kwargs = (; size = (600, 600)))
    PortfolioOptimisers.issquare(X)
    iscov = any(!isone, diag(X))
    if iscov
        X = cov2cor(X)
    end
    clim = color_func(X)
    N = size(X, 1)
    X = view(X, clr.clustering.order, clr.clustering.order)
    nx = view(nx, clr.clustering.order)
    idx = cutree(clr.clustering; k = clr.k)
    cls = [findall(x -> x == i, idx) for i ∈ 1:(clr.k)]

    colours = palette(theme_d, clr.k)
    colgrad = cgrad(theme; theme_kwargs...)

    #yticks=(1:nrows,rowlabels)
    hmap = plot(X; st = :heatmap, yticks = (1:N, nx), xticks = (1:N, nx), xrotation = 90,
                colorbar = false, clim = clim, xlim = (0.5, N + 0.5), ylim = (0.5, N + 0.5),
                color = colgrad, yflip = true, hmap_kwargs...)
    dend1 = plot(clr.clustering; xticks = false, ylim = (0, 1), dend1_kwargs...)
    dend2 = plot(clr.clustering; yticks = false, xrotation = 90, orientation = :horizontal,
                 yflip = true, xlim = (0, 1), dend2_kwargs...)

    for (i, cl) ∈ pairs(cls)
        a = [findfirst(x -> x == c, clr.clustering.order) for c ∈ cl]
        a = a[.!isnothing.(a)]
        xmin = minimum(a)
        xmax = xmin + length(cl)

        i1 = [findfirst(x -> x == c, -view(clr.clustering.merges, :, 1)) for c ∈ cl]
        i1 = i1[.!isnothing.(i1)]
        i2 = [findfirst(x -> x == c, -view(clr.clustering.merges, :, 2)) for c ∈ cl]
        i2 = i2[.!isnothing.(i2)]
        i3 = unique([i1; i2])
        h = min(maximum(clr.clustering.heights[i3]) * 1.1, 1)
        plot!(hmap,
              [xmin - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5, xmin - 0.5,
               xmin - 0.5, xmin - 0.5],
              [xmin - 0.5, xmin - 0.5, xmin - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5,
               xmax - 0.5, xmin - 0.5]; legend = false, line_kwargs...)
        plot!(dend1,
              [xmin - 0.25, xmax - 0.75, xmax - 0.75, xmax - 0.75, xmax - 0.75, xmin - 0.25,
               xmin - 0.25, xmin - 0.25], [0, 0, 0, h, h, h, h, 0]; color = nothing,
              legend = false, fill = (0, 0.5, colours[(i - 1) % clr.k + 1]))

        plot!(dend2, [0, 0, 0, h, h, h, h, 0],
              [xmin - 0.25, xmax - 0.75, xmax - 0.75, xmax - 0.75, xmax - 0.75, xmin - 0.25,
               xmin - 0.25, xmin - 0.25]; color = nothing, legend = false,
              fill = (0, 0.5, colours[(i - 1) % clr.k + 1]))
    end

    # https://docs.juliaplots.org/latest/generated/statsplots/#Dendrogram-on-the-right-side
    l = StatsPlots.grid(2, 2; heights = [0.2, 0.8], widths = [0.8, 0.2])
    return plot(dend1, plot(; ticks = nothing, border = :none, background_color = nothing),
                hmap, dend2; layout = l, fig_kwargs...)
end

end
