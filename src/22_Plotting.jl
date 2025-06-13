function plot_cumulative_returns(w::AbstractArray, X::AbstractArray, nx::AbstractVector,
                                 ts::AbstractVector, fees::Union{Nothing, <:Fees} = nothing;
                                 compound::Bool = false, asset::Bool = false, dims::Int = 1)
    cmpd = compound ? "Compounded " : "Simple "
    f = Figure()
    if asset
        ax = Axis(f[1, 1]; xlabel = "Date", ylabel = "$cmpd Asset Cummulative Returns")
        ret = cumulative_returns(calc_net_asset_returns(w, X, fees); compound = compound,
                                 dims = dims)
        for (i, asset) ∈ enumerate(nx)
            lines!(ax, ts, view(ret, :, i); label = asset)
        end
    else
        ax = Axis(f[1, 1]; xlabel = "Date", ylabel = "$cmpd Portfolio Cummulative Returns")
        ret = cumulative_returns(calc_net_returns(w, X, fees); compound = compound,
                                 dims = dims)
        lines!(ax, ts, ret; label = "Portfolio")
    end
    axislegend(; position = :lt, merge = true)
    return f
end
function plot_drawdowns end

export plot_cumulative_returns, plot_drawdowns