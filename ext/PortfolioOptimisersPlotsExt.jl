module PortfolioOptimisersPlotsExt

using PortfolioOptimisers, GraphRecipes, StatsPlots, LinearAlgebra

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

end
