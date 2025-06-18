function plot_ptf_cumulative_returns(w::AbstractArray, X::AbstractArray,
                                     fees::Union{Nothing, <:Fees} = nothing;
                                     ts::AbstractVector = 1:size(X, 1),
                                     f::Union{Nothing, Figure} = Figure(),
                                     fpos::Tuple = (1, 1), compound::Bool = false,
                                     ax_kwargs::NamedTuple = (; xlabel = "Date",
                                                              ylabel = "$(compound ? "Compound " : "Simple ") Portfolio Cummulative Returns"),
                                     line_kwargs::NamedTuple = (; label = "Portfolio"),
                                     legend_kwargs::NamedTuple = (; position = :lt,
                                                                  merge = true))
    ax = Axis(f[fpos...]; ax_kwargs...)
    ret = cumulative_returns(calc_net_returns(w, X, fees); compound = compound)
    lines!(ax, ts, ret; line_kwargs...)
    axislegend(; legend_kwargs...)
    return f
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
function plot_asset_cumulative_returns(w::AbstractVector, X::AbstractArray,
                                       fees::Union{Nothing, <:Fees} = nothing;
                                       ts::AbstractVector = 1:size(X, 1),
                                       nx::AbstractVector = 1:size(X, 2),
                                       N::Union{Nothing, <:Real} = nothing,
                                       f::Union{Nothing, Figure} = Figure(),
                                       fpos::Tuple = (1, 1), compound::Bool = false,
                                       ax_kwargs::NamedTuple = (; xlabel = "Date",
                                                                ylabel = "$(compound ? "Compound " : "Simple ") Asset Cummulative Returns"),
                                       asset_kwargs::NamedTuple = (;),
                                       line_kwargs::NamedTuple = (; label = "Others"),
                                       legend_kwargs::NamedTuple = (; position = :lt,
                                                                    merge = true))
    ax = Axis(f[fpos...]; ax_kwargs...)
    ret = cumulative_returns(calc_net_asset_returns(w, X, fees); compound = compound)
    M = size(X, 2)
    N, idx = compute_relevant_assets(w, M, isnothing(N) ? inv(dot(w, w)) : N)
    ret = view(ret, :, idx)
    nx = view(nx, idx)
    for i ∈ 1:N
        if !haskey(asset_kwargs, :label)
            asset_kwargs_i = (; label = nx[i], asset_kwargs...)
        end
        lines!(ax, ts, view(ret, :, i); asset_kwargs_i...)
    end
    if M > N
        idx = view(idx, (N + 1):M)
        ret = cumulative_returns(calc_net_returns(view(w, idx), view(X, :, idx),
                                                  fees_view(fees, idx));
                                 compound = compound)
        lines!(ax, ts, ret; line_kwargs...)
    end
    axislegend(; legend_kwargs...)
    return f
end
function plot_composition(w::AbstractVector{<:Real}, nx::AbstractVector = 1:length(w);
                          N::Union{Nothing, <:Real} = nothing,
                          f::Union{Nothing, Figure} = Figure(), fpos::Tuple = (1, 1),
                          ax_kwargs::NamedTuple = (; xlabel = "Asset", ylabel = "Weight",
                                                   title = "Portfolio Composition",
                                                   xticklabelrotation = pi / 3),
                          bar_kwargs::NamedTuple = (;))
    M = length(w)
    N, idx = compute_relevant_assets(w, M, isnothing(N) ? inv(dot(w, w)) : N)
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
function plot_stacked_bar_composition(w::AbstractArray, nx::AbstractVector = 1:size(w, 1);
                                      f::Union{Nothing, Figure} = Figure(),
                                      fpos::Tuple = (1, 1), lpos::Tuple = (1, 2),
                                      ax_kwargs::NamedTuple = (; xlabel = "Portfolios",
                                                               ylabel = "Weight",
                                                               xticks = (1:size(w, 2),
                                                                         string.(1:size(w,
                                                                                        2))),
                                                               title = "Portfolio Composition"),
                                      bar_kwargs::NamedTuple = (; colormap = :viridis))
    if isa(w, AbstractVector{<:AbstractVector})
        w = hcat(w...)
    end
    ax = Axis(f[fpos...]; ax_kwargs...)
    N = size(w, 1)
    M = size(w, 2)
    category = collect(Iterators.flatten([Iterators.repeated(i, N) for i ∈ 1:M]))
    height = vec(w)
    group = collect(Iterators.flatten(Iterators.repeated(1:N, M)))
    cmap = !haskey(bar_kwargs, :colormap) ? :viridis : bar_kwargs.colormap
    colors = try
        resample_cmap(cmap, N)
    catch err
        Makie.categorical_colors(cmap, N)
    end
    barplot!(ax, category, height; stack = group, color = group, bar_kwargs...)
    elements = [PolyElement(; polycolor = colors[i]) for i ∈ 1:length(nx)]
    Legend(f[lpos...], elements, string.(nx), "Assets")
    return f
end
function plot_stacked_area_composition(w::AbstractArray, nx::AbstractVector = 1:size(w, 1);
                                       f::Union{Nothing, Figure} = Figure(),
                                       fpos::Tuple = (1, 1), lpos::Tuple = (1, 2),
                                       ax_kwargs::NamedTuple = (; xlabel = "Portfolios",
                                                                ylabel = "Weight",
                                                                xticks = (1:size(w, 2),
                                                                          string.(1:size(w,
                                                                                         2))),
                                                                title = "Portfolio Composition"),
                                       band_kwargs::NamedTuple = (; colormap = :viridis))
    if isa(w, AbstractVector{<:AbstractVector})
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
    for i ∈ axes(w, 1)
        if i == 1
            band!(ax, 1:M, zeros(M), cw[i, :]; label = nx[i], color = colors[i],
                  band_kwargs...)
        else
            band!(ax, 1:M, cw[i - 1, :], cw[i, :]; label = nx[i], color = colors[i],
                  band_kwargs...)
        end
    end
    elements = [PolyElement(; polycolor = colors[i]) for i ∈ 1:length(nx)]
    Legend(f[lpos...], elements, string.(nx), "Assets")
    return f
end
function plot_dendrogram(hclust::Hclust)
    f = Figure()
    ax = Axis(f[1, 1])
    n = length(hclust.order)

    # Map leaf index to its x-position in the plot
    leaf_positions = Dict(hclust.order[i] => i for i ∈ 1:n)

    # Store the (x, y) coordinates of the top of the vertical line for each cluster
    cluster_coords = Dict{Int, Point2f}()

    # Store all line segments to be plotted
    segments = Point2f[]

    # Iterate through the merges from lowest to highest
    for i ∈ 1:(n - 1)
        height = hclust.height[i]

        # Get the two sub-clusters being merged
        c1_id = hclust.merge[i, 1]
        c2_id = hclust.merge[i, 2]

        # Get coordinates for the first sub-cluster
        x1, y1 = if c1_id < 0
            (leaf_positions[-c1_id], 0.0f0) # It's a leaf
        else
            cluster_coords[c1_id] # It's a previously merged cluster
        end

        # Get coordinates for the second sub-cluster
        x2, y2 = if c2_id < 0
            (leaf_positions[-c2_id], 0.0f0) # It's a leaf
        else
            cluster_coords[c2_id] # It's a previously merged cluster
        end

        # Ensure x1 < x2 for consistent drawing
        if x1 > x2
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        end

        # Add vertical lines from the base of the clusters to the merge height
        push!(segments, Point2f(x1, y1), Point2f(x1, height))
        push!(segments, Point2f(x2, y2), Point2f(x2, height))

        # Add the horizontal line connecting them at the merge height
        push!(segments, Point2f(x1, height), Point2f(x2, height))

        # Store the coordinates for the new cluster formed by this merge
        # The x-position is the midpoint, the y-position is the height
        cluster_coords[i] = Point2f((x1 + x2) / 2, height)
    end

    # Plot all segments at once
    linesegments!(ax, segments; color = :black)

    # Configure axis ticks and labels
    ax.xticks = (1:n, string.(hclust.order))
    ax.xlabel = "Sample"
    ax.ylabel = "Distance"
    return f
end
function plot_clusters(clr::AbstractClusteringResult, X::AbstractMatrix,
                       nx::AbstractVector = 1:size(X, 1);
                       f::Union{Nothing, Figure} = Figure(), fpos::Tuple = (1, 1),
                       ax_kwargs::NamedTuple = (; yreversed = true,
                                                xticklabelrotation = pi / 2, aspect = 1,
                                                title = "Clusters"),
                       color_func = x -> if any(x .< Ref(zero(eltype(x))))
                           (-1, 1)
                       else
                           (0, 1)
                       end, heatmap_kwargs::NamedTuple = (; colormap = :viridis),
                       lines_kwargs::NamedTuple = (color = :black, linewidth = 3))
    issquare(X)
    iscov = any(!isone, diag(X))
    if iscov
        X = cov2cor(X)
    end
    colorrange = color_func(X)
    N = size(X, 1)
    X = view(X, clr.clustering.order, clr.clustering.order)
    nx = view(nx, clr.clustering.order)
    ticks = (1:size(X, 1), string.(nx))
    idx = cutree(clr.clustering; k = clr.k)
    cls = [findall(x -> x == i, idx) for i ∈ 1:(clr.k)]
    ax = Axis(f[fpos...]; yticks = ticks, xticks = ticks, ax_kwargs...)
    heatmap!(ax, 1:N, 1:N, X; colorrange = colorrange, heatmap_kwargs...)
    for (i, cl) ∈ pairs(cls)
        a = [findfirst(x -> x == c, clr.clustering.order) for c ∈ cl]
        a = a[.!isnothing.(a)]
        xmin = minimum(a)
        xmax = xmin + length(cl)
        i1 = [findfirst(x -> x == c, -clr.clustering.merges[:, 1]) for c ∈ cl]
        i1 = i1[.!isnothing.(i1)]
        i2 = [findfirst(x -> x == c, -clr.clustering.merges[:, 2]) for c ∈ cl]
        i2 = i2[.!isnothing.(i2)]
        i3 = unique([i1; i2])
        h = min(maximum(clr.clustering.heights[i3]) * 1.1, 1)
        lines!(ax,
               [xmin - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5, xmin - 0.5,
                xmin - 0.5, xmin - 0.5],
               [xmin - 0.5, xmin - 0.5, xmin - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5,
                xmax - 0.5, xmin - 0.5]; lines_kwargs...)
    end
    Colorbar(f[:, end + 1]; colorrange = colorrange,
             colormap = if haskey(heatmap_kwargs, :colormap)
                 heatmap_kwargs.colormap
             else
                 :viridis
             end)
    return f
end

export plot_asset_cumulative_returns, plot_ptf_cumulative_returns, plot_composition,
       plot_stacked_bar_composition, plot_stacked_area_composition, plot_clusters,
       get_makie_dendrogram_data