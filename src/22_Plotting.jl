function plot_cumulative_returns(w::AbstractArray, X::AbstractArray, nx::AbstractVector,
                                 ts::AbstractVector, fees::Union{Nothing, <:Fees} = nothing;
                                 compound::Bool = false, asset::Bool = false,
                                 N::Real = ceil(Int, log2(length(w))),
                                 f::Union{Nothing, Figure} = Figure(),
                                 ax_kwargs::NamedTuple = if asset
                                     (; xlabel = "Date",
                                      ylabel = "$(compound ? "Compounded " : "Simple ") Asset Cummulative Returns")
                                 else
                                     (; xlabel = "Date",
                                      ylabel = "$(compound ? "Compounded " : "Simple ") Portfolio Cummulative Returns")
                                 end, asset_kwargs::NamedTuple = (;),
                                 summary_kwargs::NamedTuple = (; label = "Others"),
                                 ptf_kwargs::NamedTuple = (; label = "Portfolio"),
                                 legend_kwargs::NamedTuple = (; position = :lt,
                                                              merge = true))
    if asset
        ax = Axis(f[1, 1]; ax_kwargs...)
        ret = cumulative_returns(calc_net_asset_returns(w, X, fees); compound = compound)
        M = length(w)
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
            lines!(ax, ts, ret; summary_kwargs...)
        end
    else
        ax = Axis(f[1, 1]; ax_kwargs...)
        ret = cumulative_returns(calc_net_returns(w, X, fees); compound = compound)
        lines!(ax, ts, ret; ptf_kwargs...)
    end
    axislegend(; legend_kwargs...)
    return f
end
function plot_composition end

export plot_cumulative_returns