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

export plot_asset_cumulative_returns, plot_ptf_cumulative_returns, plot_composition,
       plot_stacked_bar_composition, plot_stacked_area_composition