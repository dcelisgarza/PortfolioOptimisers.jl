function plot_asset_cumulative_returns(w::AbstractArray, X::AbstractArray,
                                       fees::Union{Nothing, <:Fees} = nothing;
                                       ts::AbstractVector = 1:size(X, 1),
                                       nx::AbstractVector = 1:size(X, 2),
                                       N::Real = ceil(Int, log2(size(X, 2))),
                                       f::Union{Nothing, Figure} = Figure(),
                                       fpos::Tuple = (1, 1), compound::Bool = false,
                                       ax_kwargs::NamedTuple = (; xlabel = "Date",
                                                                ylabel = "$(compound ? "Compounded " : "Simple ") Asset Cummulative Returns"),
                                       asset_kwargs::NamedTuple = (;),
                                       line_kwargs::NamedTuple = (; label = "Others"),
                                       legend_kwargs::NamedTuple = (; position = :lt,
                                                                    merge = true))
    ax = Axis(f[fpos...]; ax_kwargs...)
    ret = cumulative_returns(calc_net_asset_returns(w, X, fees); compound = compound)
    M = size(X, 2)
    abs_w = abs.(w)
    idx = sortperm(abs_w; rev = true)
    abs_w /= sum(abs_w)
    N = if one(N) >= N > zero(N)
        cw = cumsum(view(abs_w, idx))
        N = findlast(x -> x <= N, cw)
        isnothing(N) ? M : N
    else
        clamp(ceil(Int, N), 1, M)
    end
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
function plot_ptf_cumulative_returns(w::AbstractArray, X::AbstractArray,
                                     fees::Union{Nothing, <:Fees} = nothing;
                                     ts::AbstractVector = 1:size(X, 1),
                                     f::Union{Nothing, Figure} = Figure(),
                                     fpos::Tuple = (1, 1), compound::Bool = false,
                                     ax_kwargs::NamedTuple = (; xlabel = "Date",
                                                              ylabel = "$(compound ? "Compounded " : "Simple ") Portfolio Cummulative Returns"),
                                     line_kwargs::NamedTuple = (; label = "Portfolio"),
                                     legend_kwargs::NamedTuple = (; position = :lt,
                                                                  merge = true))
    ax = Axis(f[fpos...]; ax_kwargs...)
    ret = cumulative_returns(calc_net_returns(w, X, fees); compound = compound)
    lines!(ax, ts, ret; line_kwargs...)
    axislegend(; legend_kwargs...)
    return f
end
function plot_composition(w::AbstractVector, nx::AbstractVector = 1:length(w);
                          f::Union{Nothing, Figure} = Figure(), fpos::Tuple = (1, 1),
                          ax_kwargs::NamedTuple = (; xlabel = "Asset", ylabel = "Weight",
                                                   xticks = (1:length(w), nx),
                                                   title = "Portfolio Composition",
                                                   xticklabelrotation = pi / 3),
                          bar_kwargs::NamedTuple = (;))
    ax = Axis(f[fpos...]; ax_kwargs...)
    barplot!(ax, 1:length(w), w; bar_kwargs...)
    return f
end
function plot_stacked_composition(w::AbstractArray, nx::AbstractVector = 1:size(w, 1);
                                  f::Union{Nothing, Figure} = Figure(),
                                  fpos::Tuple = (1, 1), lpos::Tuple = (1, 2),
                                  cmap = :viridis,
                                  ax_kwargs::NamedTuple = (; xlabel = "Portfolios",
                                                           ylabel = "Weight",
                                                           xticks = (1:size(w, 2),
                                                                     string.(1:size(w, 2))),
                                                           title = "Portfolio Composition"),
                                  bar_kwargs::NamedTuple = (;))
    ax = Axis(f[fpos...]; ax_kwargs...)
    N = size(w, 1)
    M = size(w, 2)
    category = collect(Iterators.flatten([Iterators.repeated(i, N) for i ∈ 1:M]))
    height = vec(w)
    group = collect(Iterators.flatten(Iterators.repeated(1:N, M)))
    cgrd = cgrad(cmap, length(nx))
    cmap = Makie.to_colormap(cmap)
    if !haskey(bar_kwargs, :color) || !haskey(bar_kwargs, :colormap)
        bar_kwargs = (; colormap = cmap, color = group, bar_kwargs...)
    end
    barplot!(ax, category, height; stack = group, bar_kwargs...)
    elements = [PolyElement(; polycolor = cgrd[i]) for i ∈ 1:length(nx)]
    Legend(f[lpos...], elements, string.(nx), "Assets")
    return f
end

export plot_asset_cumulative_returns, plot_ptf_cumulative_returns, plot_composition,
       plot_stacked_composition