module PortfolioOptimisersPlotsExt

using PortfolioOptimisers, StatsPlots, GraphRecipes, LinearAlgebra, Statistics, StatsBase,
      Clustering, Distributions, StatsAPI

import PortfolioOptimisers: ArrNum, VecNum, MatNum, Option, VecNum_VecVecNum, Slv_VecSlv,
                            MatNum_Pr, PrE_Pr, Pr_RR, HClE_HCl, VecVecNum, RegE_Reg,
                            NwE_ClE_Cl, AbstractCentralityEstimator,
                            AbstractClustersEstimator, AbstractClusteringResult,
                            AbstractBaseRiskMeasure, PlottingOptions,
                            NonFiniteAllocationOptimisationResult, _pred_rd_to_matrix,
                            _extract_pr, _relevant_assets, _rolling_window_measure,
                            _extract_fees, OptimisationResult
## plot_ptf_cumulative_returns
function PortfolioOptimisers.plot_ptf_cumulative_returns(w::VecNum_VecVecNum, X::MatNum,
                                                         fees::Option{<:Fees} = nothing;
                                                         ts::AbstractVector = 1:size(X, 1),
                                                         opts::PlottingOptions = PlottingOptions(),
                                                         kwargs...)
    ret = cumulative_returns(calc_net_returns(w, X, fees), opts.compound)
    label = "$(opts.compound ? "Compound" : "Simple") Cumulative Returns"
    return if isa(w, VecNum)
        plot(ts, ret; title = "Portfolio", xlabel = "Date", ylabel = label, legend = false,
             kwargs...)
    else
        plt = plot(ts, ret[1]; title = "Portfolio", xlabel = "Date", ylabel = label,
                   label = "Portfolio 1", legend = true, kwargs...)
        for i in 2:length(w)
            plot!(plt, ts, ret[i]; label = "Portfolio $(i)", kwargs...)
        end
        plt
    end
end
function PortfolioOptimisers.plot_ptf_cumulative_returns(w::VecNum_VecVecNum, pr::Pr_RR,
                                                         fees::Option{<:Fees} = nothing;
                                                         ts::AbstractVector = size(pr.X, 1),
                                                         opts::PlottingOptions = PlottingOptions(),
                                                         kwargs...)
    if isa(pr, ReturnsResult)
        ts = isnothing(pr.ts) ? (1:size(pr.X, 1)) : pr.ts
    end
    return PortfolioOptimisers.plot_ptf_cumulative_returns(w, pr.X, fees; ts = ts,
                                                           opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_ptf_cumulative_returns(res::OptimisationResult, pr::Pr_RR;
                                                         fees::Option{<:Fees} = nothing,
                                                         opts::PlottingOptions = PlottingOptions(),
                                                         kwargs...)
    fees = _extract_fees(res, fees)
    return PortfolioOptimisers.plot_ptf_cumulative_returns(res.w, pr, fees; opts = opts,
                                                           kwargs...)
end
function PortfolioOptimisers.plot_ptf_cumulative_returns(pred::Union{<:PredictionResult,
                                                                     <:MultiPeriodPredictionResult};
                                                         opts::PlottingOptions = PlottingOptions(),
                                                         kwargs...)
    rd = isa(pred, PredictionResult) ? pred.rd : pred.mrd
    ts = isnothing(rd.ts) ? (1:length(isa(rd.X, VecVecNum) ? first(rd.X) : rd.X)) : rd.ts
    w, Xm = _pred_rd_to_matrix(rd)
    return PortfolioOptimisers.plot_ptf_cumulative_returns(w, Xm, nothing; ts = ts,
                                                           opts = opts, kwargs...)
end
# function PortfolioOptimisers.plot_ptf_cumulative_returns(pred::MultiPeriodPredictionResult;
#                                                          opts::PlottingOptions = PlottingOptions(),
#                                                          kwargs...)
#     mrd = pred.mrd
#     # mrd.X already contains net portfolio returns (fees applied during predict())
#     ret = isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
#     ts = isnothing(mrd.ts) ? (1:length(ret)) : mrd.ts
#     cret = cumulative_returns(ret, opts.compound)
#     label_str = "$(opts.compound ? "Compound" : "Simple") Walk-Forward Cumulative Returns"
#     f = plot(ts, cret; title = "Walk-Forward Portfolio", xlabel = "Date",
#              ylabel = label_str, legend = :bottomleft, kwargs...)
#     folds = pred.pred
#     theme_cols = palette(:Dark2_8, length(folds))
#     for (i, p) in enumerate(folds)
#         if isnothing(p.rd.ts)
#             continue
#         end
#         vspan!(f, [p.rd.ts[1], p.rd.ts[end]]; alpha = 0.08,
#                color = theme_cols[mod1(i, length(theme_cols))], label = "")
#     end
#     return f
# end

## plot_asset_cumulative_returns
function PortfolioOptimisers.plot_asset_cumulative_returns(w::VecNum, rd::ReturnsResult,
                                                           fees::Option{<:Fees} = nothing;
                                                           opts::PlottingOptions = PlottingOptions(),
                                                           kwargs...)
    ts = isnothing(rd.ts) ? (1:size(rd.X, 1)) : rd.ts
    nx = isnothing(rd.nx) ? (1:size(rd.X, 2)) : rd.nx
    return PortfolioOptimisers.plot_asset_cumulative_returns(w, rd.X, fees; ts = ts,
                                                             nx = nx, opts = opts,
                                                             kwargs...)
end
function PortfolioOptimisers.plot_asset_cumulative_returns(res::NonFiniteAllocationOptimisationResult,
                                                           rd::ReturnsResult;
                                                           opts::PlottingOptions = PlottingOptions(),
                                                           kwargs...)
    fees = hasproperty(res, :fees) ? res.fees : nothing
    return PortfolioOptimisers.plot_asset_cumulative_returns(res.w, rd, fees; opts = opts,
                                                             kwargs...)
end
function PortfolioOptimisers.plot_asset_cumulative_returns(::PredictionResult; kwargs...)
    throw(ArgumentError("`plot_asset_cumulative_returns(pred::PredictionResult)` is not supported: `PredictionReturnsResult` stores portfolio returns, not per-asset returns. Call `plot_asset_cumulative_returns(pred.res.w, rd::ReturnsResult, ...)` with the original returns data."))
end

## plot_composition
function PortfolioOptimisers.plot_composition(res::NonFiniteAllocationOptimisationResult,
                                              rd::ReturnsResult;
                                              opts::PlottingOptions = PlottingOptions(),
                                              kwargs...)
    nx = isnothing(rd.nx) ? (1:length(res.w)) : rd.nx
    return PortfolioOptimisers.plot_composition(res.w, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_composition(res::NonFiniteAllocationOptimisationResult,
                                              pr::PortfolioOptimisers.AbstractPriorResult;
                                              opts::PlottingOptions = PlottingOptions(),
                                              kwargs...)
    return PortfolioOptimisers.plot_composition(res.w, 1:length(res.w); opts = opts,
                                                kwargs...)
end
function PortfolioOptimisers.plot_composition(pred::PredictionResult;
                                              opts::PlottingOptions = PlottingOptions(),
                                              kwargs...)
    nx = isnothing(pred.rd.nx) ? (1:length(pred.res.w)) : pred.rd.nx
    return PortfolioOptimisers.plot_composition(pred.res.w, nx; opts = opts, kwargs...)
end

## plot_stacked_bar_composition / plot_stacked_area_composition
function PortfolioOptimisers.plot_stacked_bar_composition(res_vec::AbstractVector{<:NonFiniteAllocationOptimisationResult},
                                                          rd::ReturnsResult;
                                                          opts::PlottingOptions = PlottingOptions(),
                                                          kwargs...)
    w = hcat(getproperty.(res_vec, :w)...)
    nx = isnothing(rd.nx) ? (1:size(w, 1)) : rd.nx
    return PortfolioOptimisers.plot_stacked_bar_composition(w, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_stacked_area_composition(res_vec::AbstractVector{<:NonFiniteAllocationOptimisationResult},
                                                           rd::ReturnsResult;
                                                           opts::PlottingOptions = PlottingOptions(),
                                                           kwargs...)
    w = hcat(getproperty.(res_vec, :w)...)
    nx = isnothing(rd.nx) ? (1:size(w, 1)) : rd.nx
    return PortfolioOptimisers.plot_stacked_area_composition(w, nx; opts = opts, kwargs...)
end

## plot_risk_contribution
function PortfolioOptimisers.plot_risk_contribution(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                    w::VecNum, rd::ReturnsResult,
                                                    fees::Option{<:Fees} = nothing;
                                                    opts::PlottingOptions = PlottingOptions(),
                                                    kwargs...)
    nx = isnothing(rd.nx) ? (1:size(rd.X, 2)) : rd.nx
    return PortfolioOptimisers.plot_risk_contribution(r, w, rd.X, fees; nx = nx,
                                                      opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_risk_contribution(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                    res::NonFiniteAllocationOptimisationResult,
                                                    rd::ReturnsResult;
                                                    opts::PlottingOptions = PlottingOptions(),
                                                    kwargs...)
    fees = hasproperty(res, :fees) ? res.fees : nothing
    return PortfolioOptimisers.plot_risk_contribution(r, res.w, rd, fees; opts = opts,
                                                      kwargs...)
end
function PortfolioOptimisers.plot_risk_contribution(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                    res::NonFiniteAllocationOptimisationResult,
                                                    pr::PortfolioOptimisers.AbstractPriorResult;
                                                    opts::PlottingOptions = PlottingOptions(),
                                                    kwargs...)
    fees = hasproperty(res, :fees) ? res.fees : nothing
    return PortfolioOptimisers.plot_risk_contribution(r, res.w, pr.X, fees;
                                                      nx = 1:length(res.w), opts = opts,
                                                      kwargs...)
end
function PortfolioOptimisers.plot_risk_contribution(::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                    ::PredictionResult; kwargs...)
    throw(ArgumentError("`plot_risk_contribution(r, pred::PredictionResult)` is not supported: `PredictionReturnsResult` stores portfolio returns, not raw asset returns. Call `plot_risk_contribution(r, pred.res.w, rd::ReturnsResult, ...)` with the original returns data."))
end

## plot_factor_risk_contribution
function PortfolioOptimisers.plot_factor_risk_contribution(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                           res::NonFiniteAllocationOptimisationResult,
                                                           rd::ReturnsResult;
                                                           re::RegE_Reg = StepwiseRegression(),
                                                           opts::PlottingOptions = PlottingOptions(),
                                                           kwargs...)
    fees = hasproperty(res, :fees) ? res.fees : nothing
    return PortfolioOptimisers.plot_factor_risk_contribution(r, res.w, rd.X, fees; re = re,
                                                             rd = rd, opts = opts,
                                                             kwargs...)
end
function PortfolioOptimisers.plot_factor_risk_contribution(::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                           ::PredictionResult; kwargs...)
    throw(ArgumentError("`plot_factor_risk_contribution(r, pred::PredictionResult)` is not supported: `PredictionReturnsResult` stores portfolio returns, not raw asset returns. Call `plot_factor_risk_contribution(r, pred.res.w, rd::ReturnsResult, ...)` with the original returns data."))
end

## plot_dendrogram
function PortfolioOptimisers.plot_dendrogram(clr::AbstractClusteringResult,
                                             nx::AbstractVector = 1:length(clr.res.order);
                                             dend_theme::Symbol = :Spectral,
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    N = length(clr.res.order)
    nx_ord = view(nx, clr.res.order)
    idx = assignments(clr)
    cls = [findall(x -> x == i, idx) for i in 1:(clr.k)]
    colours = palette(dend_theme, clr.k)
    dend = plot(clr.res; normalize = false, ylim = extrema(clr.res.heights),
                xticks = (1:N, nx_ord), xrotation = 90)
    for (i, cl) in pairs(cls)
        a = filter(!isnothing, [findfirst(x -> x == c, clr.res.order) for c in cl])
        if isempty(a)
            continue
        end
        xmin = minimum(a)
        xmax = xmin + length(cl)
        i1 = filter(!isnothing,
                    [findfirst(x -> x == c, -view(clr.res.merges, :, 1)) for c in cl])
        i2 = filter(!isnothing,
                    [findfirst(x -> x == c, -view(clr.res.merges, :, 2)) for c in cl])
        i3 = unique([i1; i2])
        if isempty(i3)
            continue
        end
        h = min(maximum(clr.res.heights[i3]) * 1.1, 1)
        plot!(dend,
              [xmin - 0.25, xmax - 0.75, xmax - 0.75, xmax - 0.75, xmax - 0.75, xmin - 0.25,
               xmin - 0.25, xmin - 0.25], [0, 0, 0, h, h, h, h, 0]; color = nothing,
              legend = false, fill = (0, 0.5, colours[mod1(i, clr.k)]))
    end
    return plot(dend; size = (600, 600), kwargs...)
end
function PortfolioOptimisers.plot_dendrogram(cle::HClE_HCl, X::MatNum,
                                             nx::AbstractVector = 1:size(X, 2);
                                             dims::Integer = 1,
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    clr = clusterise(cle, X; dims = dims)
    return PortfolioOptimisers.plot_dendrogram(clr, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_dendrogram(cle::HClE_HCl, pr::PortfolioOptimisers.Pr_RR,
                                             nx::AbstractVector = 1:size(pr.X, 2);
                                             dims::Integer = 1,
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    if isa(pr, ReturnsResult) && !isnothing(pr.nx)
        nx = pr.nx
    end
    return PortfolioOptimisers.plot_dendrogram(cle, pr.X, nx; dims = dims, opts = opts,
                                               kwargs...)
end
## plot_clusters
function PortfolioOptimisers.plot_clusters(clr::AbstractClusteringResult,
                                           nx::AbstractVector = 1:size(clr.S, 1);
                                           dend_theme::Symbol = :Spectral,
                                           hmap_theme::Symbol = :Spectral,
                                           color_func = x -> if any(x .< zero(eltype(x)))
                                               (-1, 1)
                                           else
                                               (0, 1)
                                           end, line_color = :black, line_width = 3,
                                           opts::PlottingOptions = PlottingOptions(),
                                           kwargs...)
    S = clr.S
    s = LinearAlgebra.diag(S)
    iscov = any(!isone, s)
    if iscov
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
    dend1 = plot(clr.res; xticks = false, ylim = extrema(clr.res.heights))
    dend2 = plot(clr.res; yticks = false, xrotation = 90, orientation = :horizontal,
                 yflip = true, xlim = extrema(clr.res.heights))
    for (i, cl) in pairs(cls)
        a = filter(!isnothing, [findfirst(x -> x == c, clr.res.order) for c in cl])
        if isempty(a)
            continue
        end
        xmin = minimum(a)
        xmax = xmin + length(cl)
        i1 = filter(!isnothing,
                    [findfirst(x -> x == c, -view(clr.res.merges, :, 1)) for c in cl])
        i2 = filter(!isnothing,
                    [findfirst(x -> x == c, -view(clr.res.merges, :, 2)) for c in cl])
        i3 = unique([i1; i2])
        if isempty(i3)
            continue
        end
        h = min(maximum(clr.res.heights[i3]) * 1.1, 1)
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
                                           dims::Integer = 1,
                                           opts::PlottingOptions = PlottingOptions(),
                                           kwargs...)
    clr = clusterise(cle, X; dims = dims)
    return PortfolioOptimisers.plot_clusters(clr, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_clusters(cle::HClE_HCl, pr::PortfolioOptimisers.Pr_RR,
                                           nx::AbstractVector = 1:size(pr.X, 2);
                                           dims::Integer = 1,
                                           opts::PlottingOptions = PlottingOptions(),
                                           kwargs...)
    if isa(pr, ReturnsResult) && !isnothing(pr.nx)
        nx = pr.nx
    end
    return PortfolioOptimisers.plot_clusters(cle, pr.X, nx; opts = opts, kwargs...)
end

## plot_drawdowns
function PortfolioOptimisers.plot_drawdowns(w::ArrNum, rd::ReturnsResult,
                                            fees::Option{<:Fees} = nothing;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    ts = isnothing(rd.ts) ? (1:size(rd.X, 1)) : rd.ts
    return PortfolioOptimisers.plot_drawdowns(w, rd.X, fees; slv = slv, ts = ts,
                                              opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_drawdowns(res::NonFiniteAllocationOptimisationResult,
                                            rd::ReturnsResult;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    fees = hasproperty(res, :fees) ? res.fees : nothing
    return PortfolioOptimisers.plot_drawdowns(res.w, rd, fees; slv = slv, opts = opts,
                                              kwargs...)
end
function PortfolioOptimisers.plot_drawdowns(pred::PredictionResult;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    rd = pred.rd
    ts = isnothing(rd.ts) ? (1:length(isa(rd.X, VecVecNum) ? first(rd.X) : rd.X)) : rd.ts
    w, Xm = _pred_rd_to_matrix(rd)
    return PortfolioOptimisers.plot_drawdowns(w, Xm, nothing; slv = slv, ts = ts,
                                              opts = opts, kwargs...)
end

## plot_measures
function PortfolioOptimisers.plot_measures(w::VecNum_VecVecNum, pr::Pr_RR,
                                           fees::Option{<:Fees} = nothing;
                                           x::PortfolioOptimisers.AbstractBaseRiskMeasure = Variance(),
                                           y::PortfolioOptimisers.AbstractBaseRiskMeasure = ExpectedReturn(),
                                           z::Option{<:PortfolioOptimisers.AbstractBaseRiskMeasure} = nothing,
                                           c::PortfolioOptimisers.AbstractBaseRiskMeasure = ExpectedReturnRiskRatio(;
                                                                                                                    rk = x,
                                                                                                                    rt = ArithmeticReturn(),
                                                                                                                    rf = 0),
                                           slv::Option{<:Slv_VecSlv} = nothing,
                                           flag::Bool = true,
                                           opts::PlottingOptions = PlottingOptions(),
                                           kwargs...)
    if flag
        x = factory(x, pr, slv)
        y = factory(y, pr, slv)
        z = isnothing(z) ? nothing : factory(z, pr, slv)
        c = factory(c, pr, slv)
    end
    xr = expected_risk(x, w, pr, fees)
    yr = expected_risk(y, w, pr, fees)
    zr = isnothing(z) ? nothing : expected_risk(z, w, pr, fees)
    cr = expected_risk(c, w, pr, fees)
    return if isnothing(zr)
        scatter(xr, yr; zcolor = cr, title = "Pareto Frontier", xlabel = "X", ylabel = "Y",
                colorbar_title = "C", label = nothing, legend = true, kwargs...)
    else
        scatter(xr, yr, zr; zcolor = cr, title = "Pareto Frontier", xlabel = "X",
                ylabel = "Y", zlabel = "Z", colorbar_title = "C", label = nothing,
                legend = true, kwargs...)
    end
end
function PortfolioOptimisers.plot_measures(res_vec::AbstractVector{<:NonFiniteAllocationOptimisationResult},
                                           rd::ReturnsResult;
                                           x::PortfolioOptimisers.AbstractBaseRiskMeasure = Variance(),
                                           y::PortfolioOptimisers.AbstractBaseRiskMeasure = ExpectedReturn(),
                                           z::Option{<:PortfolioOptimisers.AbstractBaseRiskMeasure} = nothing,
                                           c::PortfolioOptimisers.AbstractBaseRiskMeasure = ExpectedReturnRiskRatio(;
                                                                                                                    rk = x,
                                                                                                                    rt = ArithmeticReturn(),
                                                                                                                    rf = 0),
                                           slv::Option{<:Slv_VecSlv} = nothing,
                                           fees::Option{<:Fees} = nothing,
                                           flag::Bool = true,
                                           opts::PlottingOptions = PlottingOptions(),
                                           kwargs...)
    pr = if hasproperty(first(res_vec), :pr)
        first(res_vec).pr
    elseif hasproperty(first(res_vec), :pa) && hasproperty(first(res_vec).pa, :pr)
        first(res_vec).pa.pr
    else
        throw(ArgumentError("result type $(nameof(typeof(first(res_vec)))) has no `.pr` or `.pa.pr`; pass `pr` explicitly"))
    end
    w = getproperty.(res_vec, :w)
    return PortfolioOptimisers.plot_measures(w, pr, fees; x = x, y = y, z = z, c = c,
                                             slv = slv, flag = flag, opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_measures(mpred::MultiPeriodPredictionResult;
                                           x::PortfolioOptimisers.AbstractBaseRiskMeasure = Variance(),
                                           y::PortfolioOptimisers.AbstractBaseRiskMeasure = ExpectedReturn(),
                                           z::Option{<:PortfolioOptimisers.AbstractBaseRiskMeasure} = nothing,
                                           c::PortfolioOptimisers.AbstractBaseRiskMeasure = ExpectedReturnRiskRatio(;
                                                                                                                    rk = x,
                                                                                                                    rt = ArithmeticReturn(),
                                                                                                                    rf = 0),
                                           slv::Option{<:Slv_VecSlv} = nothing,
                                           fees::Option{<:Fees} = nothing,
                                           flag::Bool = true,
                                           opts::PlottingOptions = PlottingOptions(),
                                           kwargs...)
    return PortfolioOptimisers.plot_measures(mpred.res, mpred.mrd; x = x, y = y, z = z,
                                             c = c, slv = slv, fees = fees, flag = flag,
                                             opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_measures(ppred::PopulationPredictionResult;
                                           x::PortfolioOptimisers.AbstractBaseRiskMeasure = Variance(),
                                           y::PortfolioOptimisers.AbstractBaseRiskMeasure = ExpectedReturn(),
                                           z::Option{<:PortfolioOptimisers.AbstractBaseRiskMeasure} = nothing,
                                           c::PortfolioOptimisers.AbstractBaseRiskMeasure = ExpectedReturnRiskRatio(;
                                                                                                                    rk = x,
                                                                                                                    rt = ArithmeticReturn(),
                                                                                                                    rf = 0),
                                           slv::Option{<:Slv_VecSlv} = nothing,
                                           fees::Option{<:Fees} = nothing,
                                           flag::Bool = true,
                                           opts::PlottingOptions = PlottingOptions(),
                                           kwargs...)
    members = ppred.pred
    w = map(members) do m
        if isa(m, PredictionResult)
            m.res.w
        else
            vec(mean(hcat(getproperty.(getproperty.(m.pred, :res), :w)...); dims = 2))
        end
    end
    first_res = isa(members[1], PredictionResult) ? members[1].res : members[1].pred[1].res
    pr = if hasproperty(first_res, :pr)
        first_res.pr
    elseif hasproperty(first_res, :pa) && hasproperty(first_res.pa, :pr)
        first_res.pa.pr
    else
        throw(ArgumentError("population member has no `.pr` or `.pa.pr`; pass `pr` explicitly"))
    end
    return PortfolioOptimisers.plot_measures(w, pr, fees; x = x, y = y, z = z, c = c,
                                             slv = slv, flag = flag, opts = opts, kwargs...)
end

## plot_histogram
function PortfolioOptimisers.plot_histogram(w::ArrNum, rd::ReturnsResult,
                                            fees::Option{<:Fees} = nothing;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    return PortfolioOptimisers.plot_histogram(w, rd.X, fees; slv = slv, opts = opts,
                                              kwargs...)
end
function PortfolioOptimisers.plot_histogram(res::NonFiniteAllocationOptimisationResult,
                                            rd::ReturnsResult;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    fees = hasproperty(res, :fees) ? res.fees : nothing
    return PortfolioOptimisers.plot_histogram(res.w, rd.X, fees; slv = slv, opts = opts,
                                              kwargs...)
end
function PortfolioOptimisers.plot_histogram(pred::PredictionResult;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    rd = pred.rd
    w, Xm = _pred_rd_to_matrix(rd)
    return PortfolioOptimisers.plot_histogram(w, Xm, nothing; slv = slv, opts = opts,
                                              kwargs...)
end

## plot_network
function PortfolioOptimisers.plot_network(pl::NwE_ClE_Cl,
                                          pr::PortfolioOptimisers.AbstractPriorResult,
                                          nx::AbstractVector = 1:size(pr.X, 2),
                                          w::Option{<:VecNum} = nothing;
                                          opts::PlottingOptions = PlottingOptions(),
                                          kwargs...)
    return PortfolioOptimisers.plot_network(pl, pr.X, nx, w; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_network(pl::NwE_ClE_Cl, rd::ReturnsResult,
                                          w::Option{<:VecNum} = nothing;
                                          opts::PlottingOptions = PlottingOptions(),
                                          kwargs...)
    nx = isnothing(rd.nx) ? (1:size(rd.X, 2)) : rd.nx
    return PortfolioOptimisers.plot_network(pl, rd.X, nx, w; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_network(pl::NwE_ClE_Cl,
                                          res::NonFiniteAllocationOptimisationResult,
                                          rd::ReturnsResult;
                                          opts::PlottingOptions = PlottingOptions(),
                                          kwargs...)
    nx = isnothing(rd.nx) ? (1:length(res.w)) : rd.nx
    return PortfolioOptimisers.plot_network(pl, rd.X, nx, res.w; opts = opts, kwargs...)
end

## plot_centrality
function PortfolioOptimisers.plot_centrality(cte::AbstractCentralityEstimator,
                                             pr::PortfolioOptimisers.AbstractPriorResult,
                                             nx::AbstractVector = 1:size(pr.X, 2);
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    return PortfolioOptimisers.plot_centrality(cte, pr.X, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_centrality(cte::AbstractCentralityEstimator,
                                             rd::ReturnsResult;
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    nx = isnothing(rd.nx) ? (1:size(rd.X, 2)) : rd.nx
    return PortfolioOptimisers.plot_centrality(cte, rd.X, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_centrality(cte::AbstractCentralityEstimator,
                                             res::NonFiniteAllocationOptimisationResult,
                                             rd::ReturnsResult;
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    nx = isnothing(rd.nx) ? (1:length(res.w)) : rd.nx
    return PortfolioOptimisers.plot_centrality(cte, rd.X, nx; opts = opts, kwargs...)
end

## plot_correlation
function PortfolioOptimisers.plot_correlation(pr::PortfolioOptimisers.AbstractPriorResult,
                                              nx::AbstractVector = 1:size(pr.sigma, 1);
                                              opts::PlottingOptions = PlottingOptions(),
                                              kwargs...)
    return PortfolioOptimisers.plot_correlation(pr.sigma, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_correlation(pr::PortfolioOptimisers.AbstractPriorResult,
                                              rd::ReturnsResult;
                                              opts::PlottingOptions = PlottingOptions(),
                                              kwargs...)
    nx = isnothing(rd.nx) ? (1:size(pr.sigma, 1)) : rd.nx
    return PortfolioOptimisers.plot_correlation(pr.sigma, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_correlation(res::NonFiniteAllocationOptimisationResult,
                                              rd::ReturnsResult;
                                              opts::PlottingOptions = PlottingOptions(),
                                              kwargs...)
    return PortfolioOptimisers.plot_correlation(_extract_pr(res), rd; opts = opts,
                                                kwargs...)
end
function PortfolioOptimisers.plot_correlation(res::NonFiniteAllocationOptimisationResult;
                                              opts::PlottingOptions = PlottingOptions(),
                                              kwargs...)
    return PortfolioOptimisers.plot_correlation(_extract_pr(res); opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_correlation(pred::PredictionResult, rd::ReturnsResult;
                                              opts::PlottingOptions = PlottingOptions(),
                                              kwargs...)
    return PortfolioOptimisers.plot_correlation(_extract_pr(pred.res), rd; opts = opts,
                                                kwargs...)
end
function PortfolioOptimisers.plot_correlation(pred::PredictionResult;
                                              opts::PlottingOptions = PlottingOptions(),
                                              kwargs...)
    return PortfolioOptimisers.plot_correlation(_extract_pr(pred.res); opts = opts,
                                                kwargs...)
end

## plot_mu
function PortfolioOptimisers.plot_mu(pr::PortfolioOptimisers.AbstractPriorResult,
                                     nx::AbstractVector = 1:length(pr.mu);
                                     opts::PlottingOptions = PlottingOptions(), kwargs...)
    return PortfolioOptimisers.plot_mu(pr.mu, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_mu(pr::PortfolioOptimisers.AbstractPriorResult,
                                     rd::ReturnsResult;
                                     opts::PlottingOptions = PlottingOptions(), kwargs...)
    nx = isnothing(rd.nx) ? (1:length(pr.mu)) : rd.nx
    return PortfolioOptimisers.plot_mu(pr.mu, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_mu(res::NonFiniteAllocationOptimisationResult;
                                     opts::PlottingOptions = PlottingOptions(), kwargs...)
    return PortfolioOptimisers.plot_mu(_extract_pr(res); opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_mu(res::NonFiniteAllocationOptimisationResult,
                                     rd::ReturnsResult;
                                     opts::PlottingOptions = PlottingOptions(), kwargs...)
    return PortfolioOptimisers.plot_mu(_extract_pr(res), rd; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_mu(pred::PredictionResult;
                                     opts::PlottingOptions = PlottingOptions(), kwargs...)
    return PortfolioOptimisers.plot_mu(_extract_pr(pred.res); opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_mu(pred::PredictionResult, rd::ReturnsResult;
                                     opts::PlottingOptions = PlottingOptions(), kwargs...)
    return PortfolioOptimisers.plot_mu(_extract_pr(pred.res), rd; opts = opts, kwargs...)
end

## plot_sigma
function PortfolioOptimisers.plot_sigma(pr::PortfolioOptimisers.AbstractPriorResult,
                                        nx::AbstractVector = 1:size(pr.sigma, 1);
                                        opts::PlottingOptions = PlottingOptions(),
                                        kwargs...)
    return PortfolioOptimisers.plot_sigma(pr.sigma, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_sigma(pr::PortfolioOptimisers.AbstractPriorResult,
                                        rd::ReturnsResult;
                                        opts::PlottingOptions = PlottingOptions(),
                                        kwargs...)
    nx = isnothing(rd.nx) ? (1:size(pr.sigma, 1)) : rd.nx
    return PortfolioOptimisers.plot_sigma(pr.sigma, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_sigma(res::NonFiniteAllocationOptimisationResult,
                                        rd::ReturnsResult;
                                        opts::PlottingOptions = PlottingOptions(),
                                        kwargs...)
    return PortfolioOptimisers.plot_sigma(_extract_pr(res), rd; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_sigma(res::NonFiniteAllocationOptimisationResult;
                                        opts::PlottingOptions = PlottingOptions(),
                                        kwargs...)
    return PortfolioOptimisers.plot_sigma(_extract_pr(res); opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_sigma(pred::PredictionResult;
                                        opts::PlottingOptions = PlottingOptions(),
                                        kwargs...)
    return PortfolioOptimisers.plot_sigma(_extract_pr(pred.res); opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_sigma(pred::PredictionResult, rd::ReturnsResult;
                                        opts::PlottingOptions = PlottingOptions(),
                                        kwargs...)
    return PortfolioOptimisers.plot_sigma(_extract_pr(pred.res), rd; opts = opts, kwargs...)
end

## plot_factor_loadings
function PortfolioOptimisers.plot_factor_loadings(pr::PortfolioOptimisers.AbstractPriorResult,
                                                  nx::AbstractVector = 1:size(pr.rr.M, 1),
                                                  nf::AbstractVector = 1:size(pr.rr.M, 2);
                                                  opts::PlottingOptions = PlottingOptions(),
                                                  kwargs...)
    if isnothing(pr.rr)
        throw(ArgumentError("prior has no factor regression model (rr is nothing); pass M directly"))
    end
    return PortfolioOptimisers.plot_factor_loadings(pr.rr.M, nx, nf; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_factor_loadings(pr::PortfolioOptimisers.AbstractPriorResult,
                                                  rd::ReturnsResult;
                                                  opts::PlottingOptions = PlottingOptions(),
                                                  kwargs...)
    if isnothing(pr.rr)
        throw(ArgumentError("prior has no factor regression model (rr is nothing); pass M directly"))
    end
    nx = isnothing(rd.nx) ? (1:size(pr.rr.M, 1)) : rd.nx
    nf = isnothing(rd.nf) ? (1:size(pr.rr.M, 2)) : rd.nf
    return PortfolioOptimisers.plot_factor_loadings(pr.rr.M, nx, nf; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_factor_loadings(res::NonFiniteAllocationOptimisationResult,
                                                  rd::ReturnsResult;
                                                  opts::PlottingOptions = PlottingOptions(),
                                                  kwargs...)
    return PortfolioOptimisers.plot_factor_loadings(_extract_pr(res), rd; opts = opts,
                                                    kwargs...)
end
function PortfolioOptimisers.plot_factor_loadings(res::NonFiniteAllocationOptimisationResult;
                                                  opts::PlottingOptions = PlottingOptions(),
                                                  kwargs...)
    return PortfolioOptimisers.plot_factor_loadings(_extract_pr(res); opts = opts,
                                                    kwargs...)
end
function PortfolioOptimisers.plot_factor_loadings(pred::PredictionResult, rd::ReturnsResult;
                                                  opts::PlottingOptions = PlottingOptions(),
                                                  kwargs...)
    return PortfolioOptimisers.plot_factor_loadings(_extract_pr(pred.res), rd; opts = opts,
                                                    kwargs...)
end
function PortfolioOptimisers.plot_factor_loadings(pred::PredictionResult;
                                                  opts::PlottingOptions = PlottingOptions(),
                                                  kwargs...)
    return PortfolioOptimisers.plot_factor_loadings(_extract_pr(pred.res); opts = opts,
                                                    kwargs...)
end

## plot_factor_sigma
function PortfolioOptimisers.plot_factor_sigma(pr::PortfolioOptimisers.AbstractPriorResult,
                                               nf::AbstractVector = 1:size(pr.f_sigma, 1);
                                               opts::PlottingOptions = PlottingOptions(),
                                               kwargs...)
    if isnothing(pr.f_sigma)
        throw(ArgumentError("prior has no factor covariance (f_sigma is nothing); pass f_sigma directly"))
    end
    return PortfolioOptimisers.plot_factor_sigma(pr.f_sigma, nf; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_factor_sigma(pr::PortfolioOptimisers.AbstractPriorResult,
                                               rd::ReturnsResult;
                                               opts::PlottingOptions = PlottingOptions(),
                                               kwargs...)
    if isnothing(pr.f_sigma)
        throw(ArgumentError("prior has no factor covariance (f_sigma is nothing); pass f_sigma directly"))
    end
    nf = isnothing(rd.nf) ? (1:size(pr.f_sigma, 1)) : rd.nf
    return PortfolioOptimisers.plot_factor_sigma(pr.f_sigma, nf; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_factor_sigma(res::NonFiniteAllocationOptimisationResult,
                                               rd::ReturnsResult;
                                               opts::PlottingOptions = PlottingOptions(),
                                               kwargs...)
    return PortfolioOptimisers.plot_factor_sigma(_extract_pr(res), rd; opts = opts,
                                                 kwargs...)
end
function PortfolioOptimisers.plot_factor_sigma(res::NonFiniteAllocationOptimisationResult;
                                               opts::PlottingOptions = PlottingOptions(),
                                               kwargs...)
    return PortfolioOptimisers.plot_factor_sigma(_extract_pr(res); opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_factor_sigma(pred::PredictionResult, rd::ReturnsResult;
                                               opts::PlottingOptions = PlottingOptions(),
                                               kwargs...)
    return PortfolioOptimisers.plot_factor_sigma(_extract_pr(pred.res), rd; opts = opts,
                                                 kwargs...)
end
function PortfolioOptimisers.plot_factor_sigma(pred::PredictionResult;
                                               opts::PlottingOptions = PlottingOptions(),
                                               kwargs...)
    return PortfolioOptimisers.plot_factor_sigma(_extract_pr(pred.res); opts = opts,
                                                 kwargs...)
end

## plot_eigenspectrum
function PortfolioOptimisers.plot_eigenspectrum(pr::PortfolioOptimisers.AbstractPriorResult;
                                                opts::PlottingOptions = PlottingOptions(),
                                                kwargs...)
    return PortfolioOptimisers.plot_eigenspectrum(pr.sigma; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_eigenspectrum(pr::PortfolioOptimisers.AbstractPriorResult,
                                                rd::ReturnsResult;
                                                opts::PlottingOptions = PlottingOptions(),
                                                kwargs...)
    T = isnothing(rd.X) ? nothing : size(rd.X, 1)
    return PortfolioOptimisers.plot_eigenspectrum(pr.sigma; N_obs = T, opts = opts,
                                                  kwargs...)
end
function PortfolioOptimisers.plot_eigenspectrum(res::NonFiniteAllocationOptimisationResult;
                                                opts::PlottingOptions = PlottingOptions(),
                                                kwargs...)
    return PortfolioOptimisers.plot_eigenspectrum(_extract_pr(res); opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_eigenspectrum(res::NonFiniteAllocationOptimisationResult,
                                                rd::ReturnsResult;
                                                opts::PlottingOptions = PlottingOptions(),
                                                kwargs...)
    return PortfolioOptimisers.plot_eigenspectrum(_extract_pr(res), rd; opts = opts,
                                                  kwargs...)
end
function PortfolioOptimisers.plot_eigenspectrum(pred::PredictionResult;
                                                opts::PlottingOptions = PlottingOptions(),
                                                kwargs...)
    return PortfolioOptimisers.plot_eigenspectrum(_extract_pr(pred.res); opts = opts,
                                                  kwargs...)
end
function PortfolioOptimisers.plot_eigenspectrum(pred::PredictionResult, rd::ReturnsResult;
                                                opts::PlottingOptions = PlottingOptions(),
                                                kwargs...)
    return PortfolioOptimisers.plot_eigenspectrum(_extract_pr(pred.res), rd; opts = opts,
                                                  kwargs...)
end

## plot_rolling_measure
function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                  w::VecNum, rd::ReturnsResult,
                                                  fees::Option{<:Fees} = nothing;
                                                  opts::PlottingOptions = PlottingOptions(),
                                                  kwargs...)
    ts = isnothing(rd.ts) ? (1:size(rd.X, 1)) : rd.ts
    return PortfolioOptimisers.plot_rolling_measure(r, w, rd.X, fees; ts = ts, opts = opts,
                                                    kwargs...)
end
function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                  res::NonFiniteAllocationOptimisationResult,
                                                  rd::ReturnsResult;
                                                  opts::PlottingOptions = PlottingOptions(),
                                                  kwargs...)
    fees = hasproperty(res, :fees) ? res.fees : nothing
    return PortfolioOptimisers.plot_rolling_measure(r, res.w, rd, fees; opts = opts,
                                                    kwargs...)
end
function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                  pred::PredictionResult;
                                                  opts::PlottingOptions = PlottingOptions(),
                                                  kwargs...)
    rd = pred.rd
    ts = isnothing(rd.ts) ? (1:length(isa(rd.X, VecVecNum) ? first(rd.X) : rd.X)) : rd.ts
    w, Xm = _pred_rd_to_matrix(rd)
    return PortfolioOptimisers.plot_rolling_measure(r, w, Xm, nothing; ts = ts, opts = opts,
                                                    kwargs...)
end

## plot_cv_scores
function PortfolioOptimisers.plot_cv_scores(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                            mpred::MultiPeriodPredictionResult;
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    scores = [expected_risk(r, p) for p in mpred.pred]
    return PortfolioOptimisers.plot_cv_scores(scores, 1:length(scores); opts = opts,
                                              kwargs...)
end
function PortfolioOptimisers.plot_cv_scores(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                            ppred::PopulationPredictionResult;
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    scores = [expected_risk(r, m) for m in ppred.pred]
    return PortfolioOptimisers.plot_cv_scores(scores, 1:length(scores); opts = opts,
                                              kwargs...)
end

## plot_turnover
function PortfolioOptimisers.plot_turnover(mpred::MultiPeriodPredictionResult;
                                           opts::PlottingOptions = PlottingOptions(),
                                           kwargs...)
    folds = mpred.pred
    w_series = getproperty.(getproperty.(folds, :res), :w)
    ts = if isnothing(folds[1].rd.ts)
        1:length(w_series)
    else
        [f.rd.ts[end] for f in folds]
    end
    return PortfolioOptimisers.plot_turnover(w_series; ts = ts, opts = opts, kwargs...)
end

## plot_prior
function PortfolioOptimisers.plot_prior(pr::PortfolioOptimisers.AbstractPriorResult,
                                        rd::ReturnsResult;
                                        opts::PlottingOptions = PlottingOptions(),
                                        kwargs...)
    nx = isnothing(rd.nx) ? (1:length(pr.mu)) : rd.nx
    return PortfolioOptimisers.plot_prior(pr, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_prior(res::NonFiniteAllocationOptimisationResult;
                                        opts::PlottingOptions = PlottingOptions(),
                                        kwargs...)
    return PortfolioOptimisers.plot_prior(_extract_pr(res); opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_prior(res::NonFiniteAllocationOptimisationResult,
                                        rd::ReturnsResult;
                                        opts::PlottingOptions = PlottingOptions(),
                                        kwargs...)
    return PortfolioOptimisers.plot_prior(_extract_pr(res), rd; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_prior(pred::PredictionResult;
                                        opts::PlottingOptions = PlottingOptions(),
                                        kwargs...)
    return PortfolioOptimisers.plot_prior(_extract_pr(pred.res); opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_prior(pred::PredictionResult, rd::ReturnsResult;
                                        opts::PlottingOptions = PlottingOptions(),
                                        kwargs...)
    return PortfolioOptimisers.plot_prior(_extract_pr(pred.res), rd; opts = opts, kwargs...)
end

## plot_factor_mu
function PortfolioOptimisers.plot_factor_mu(pr::PortfolioOptimisers.AbstractPriorResult,
                                            nf::AbstractVector = 1:length(pr.f_mu);
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    if isnothing(pr.f_mu)
        throw(ArgumentError("prior has no factor expected returns (`f_mu` is `nothing`); pass `f_mu` directly"))
    end
    return PortfolioOptimisers.plot_factor_mu(pr.f_mu, nf; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_factor_mu(pr::PortfolioOptimisers.AbstractPriorResult,
                                            rd::ReturnsResult;
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    if isnothing(pr.f_mu)
        throw(ArgumentError("prior has no factor expected returns (`f_mu` is `nothing`); pass `f_mu` directly"))
    end
    nf = isnothing(rd.nf) ? (1:length(pr.f_mu)) : rd.nf
    return PortfolioOptimisers.plot_factor_mu(pr.f_mu, nf; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_factor_mu(res::NonFiniteAllocationOptimisationResult;
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    return PortfolioOptimisers.plot_factor_mu(_extract_pr(res); opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_factor_mu(res::NonFiniteAllocationOptimisationResult,
                                            rd::ReturnsResult;
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    return PortfolioOptimisers.plot_factor_mu(_extract_pr(res), rd; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_factor_mu(pred::PredictionResult;
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    return PortfolioOptimisers.plot_factor_mu(_extract_pr(pred.res); opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_factor_mu(pred::PredictionResult, rd::ReturnsResult;
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    return PortfolioOptimisers.plot_factor_mu(_extract_pr(pred.res), rd; opts = opts,
                                              kwargs...)
end

## plot_benchmark
function PortfolioOptimisers.plot_benchmark(w::ArrNum, rd::ReturnsResult,
                                            fees::Option{<:Fees} = nothing;
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    if isnothing(rd.B)
        throw(ArgumentError("returns data has no benchmark (`B` is `nothing`)"))
    end
    ts = isnothing(rd.ts) ? (1:size(rd.X, 1)) : rd.ts
    nb = rd.nb
    return PortfolioOptimisers.plot_benchmark(w, rd.X, rd.B, fees; ts = ts, nb = nb,
                                              opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_benchmark(res::NonFiniteAllocationOptimisationResult,
                                            rd::ReturnsResult;
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    fees = hasproperty(res, :fees) ? res.fees : nothing
    return PortfolioOptimisers.plot_benchmark(res.w, rd, fees; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_benchmark(pred::PredictionResult;
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    rd = pred.rd
    if isnothing(rd.B)
        throw(ArgumentError("prediction data has no benchmark (`B` is `nothing`)"))
    end
    ts = isnothing(rd.ts) ? (1:length(isa(rd.X, VecVecNum) ? first(rd.X) : rd.X)) : rd.ts
    nb = rd.nb
    w, Xm = _pred_rd_to_matrix(rd)
    B = isa(rd.B, VecVecNum) ? first(rd.B) : rd.B
    return PortfolioOptimisers.plot_benchmark(w, Xm, B, nothing; ts = ts, nb = nb,
                                              opts = opts, kwargs...)
end

## plot_coskewness
function PortfolioOptimisers.plot_coskewness(pr::HighOrderPrior,
                                             nx::AbstractVector = 1:size(pr.sk, 1);
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    if isnothing(pr.sk)
        throw(ArgumentError("prior has no coskewness matrix (`sk` is `nothing`)"))
    end
    return PortfolioOptimisers.plot_coskewness(pr.sk, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_coskewness(pr::HighOrderPrior, rd::ReturnsResult;
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    if isnothing(pr.sk)
        throw(ArgumentError("prior has no coskewness matrix (`sk` is `nothing`)"))
    end
    nx = isnothing(rd.nx) ? (1:size(pr.sk, 1)) : rd.nx
    return PortfolioOptimisers.plot_coskewness(pr.sk, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_coskewness(res::NonFiniteAllocationOptimisationResult;
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    pr = _extract_pr(res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(res)))` prior is not a `HighOrderPrior`; no coskewness available"))
    end
    return PortfolioOptimisers.plot_coskewness(pr; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_coskewness(res::NonFiniteAllocationOptimisationResult,
                                             rd::ReturnsResult;
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    pr = _extract_pr(res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(res)))` prior is not a `HighOrderPrior`; no coskewness available"))
    end
    return PortfolioOptimisers.plot_coskewness(pr, rd; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_coskewness(pred::PredictionResult;
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    pr = _extract_pr(pred.res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(pred.res)))` prior is not a `HighOrderPrior`; no coskewness available"))
    end
    return PortfolioOptimisers.plot_coskewness(pr; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_coskewness(pred::PredictionResult, rd::ReturnsResult;
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    pr = _extract_pr(pred.res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(pred.res)))` prior is not a `HighOrderPrior`; no coskewness available"))
    end
    return PortfolioOptimisers.plot_coskewness(pr, rd; opts = opts, kwargs...)
end

## plot_cokurtosis
function PortfolioOptimisers.plot_cokurtosis(pr::HighOrderPrior,
                                             nx::AbstractVector = 1:isqrt(size(pr.kt, 1));
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    if isnothing(pr.kt)
        throw(ArgumentError("prior has no cokurtosis matrix (`kt` is `nothing`)"))
    end
    return PortfolioOptimisers.plot_cokurtosis(pr.kt, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_cokurtosis(pr::HighOrderPrior, rd::ReturnsResult;
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    if isnothing(pr.kt)
        throw(ArgumentError("prior has no cokurtosis matrix (`kt` is `nothing`)"))
    end
    nx = isnothing(rd.nx) ? (1:isqrt(size(pr.kt, 1))) : rd.nx
    return PortfolioOptimisers.plot_cokurtosis(pr.kt, nx; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_cokurtosis(res::NonFiniteAllocationOptimisationResult;
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    pr = _extract_pr(res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(res)))` prior is not a `HighOrderPrior`; no cokurtosis available"))
    end
    return PortfolioOptimisers.plot_cokurtosis(pr; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_cokurtosis(res::NonFiniteAllocationOptimisationResult,
                                             rd::ReturnsResult;
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    pr = _extract_pr(res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(res)))` prior is not a `HighOrderPrior`; no cokurtosis available"))
    end
    return PortfolioOptimisers.plot_cokurtosis(pr, rd; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_cokurtosis(pred::PredictionResult;
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    pr = _extract_pr(pred.res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(pred.res)))` prior is not a `HighOrderPrior`; no cokurtosis available"))
    end
    return PortfolioOptimisers.plot_cokurtosis(pr; opts = opts, kwargs...)
end
function PortfolioOptimisers.plot_cokurtosis(pred::PredictionResult, rd::ReturnsResult;
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    pr = _extract_pr(pred.res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(pred.res)))` prior is not a `HighOrderPrior`; no cokurtosis available"))
    end
    return PortfolioOptimisers.plot_cokurtosis(pr, rd; opts = opts, kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Asset cumulative returns
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_asset_cumulative_returns(w::VecNum, X::MatNum,
                                                           fees::Option{<:Fees} = nothing;
                                                           ts::AbstractVector = 1:size(X, 1),
                                                           nx::AbstractVector = 1:size(X, 2),
                                                           opts::PlottingOptions = PlottingOptions(),
                                                           kwargs...)
    net_asset_ret = calc_net_asset_returns(w, X, fees)
    ret = cumulative_returns(net_asset_ret, opts.compound)
    M = size(X, 2)
    N, idx = _relevant_assets(w, M, opts.N)
    ret_sorted = view(ret, :, idx)
    nx_sorted = view(nx, idx)

    label_str = "$(opts.compound ? "Compound" : "Simple") Asset Cumulative Returns"
    f = plot(; xlabel = "Date", ylabel = label_str)
    for i in 1:N
        plot!(f, ts, view(ret_sorted, :, i); label = string(nx_sorted[i]))
    end
    if M > N
        rest_idx = view(idx, (N + 1):M)
        rest_ret = cumulative_returns(calc_net_returns(view(w, rest_idx),
                                                       view(X, :, rest_idx),
                                                       PortfolioOptimisers.fees_view(fees,
                                                                                     rest_idx)),
                                      opts.compound)
        plot!(f, ts, rest_ret; label = "Others")
    end
    plot!(f; legend = :outerright, kwargs...)
    return f
end

## ────────────────────────────────────────────────────────────────────────────
## Portfolio composition (bar)
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_composition(w::VecNum, nx::AbstractVector = 1:length(w);
                                              opts::PlottingOptions = PlottingOptions(),
                                              kwargs...)
    M = length(w)
    N, idx = _relevant_assets(w, M, opts.N)
    if M > N
        sort!(view(idx, 1:N))
        fidx = view(idx, 1:N)
        w_plot = [view(w, fidx); sum(view(w, view(idx, (N + 1):M)))]
        nx_plot = [nx[fidx]; "Others"]
    else
        w_plot = w
        nx_plot = nx
    end
    return bar(w_plot; xticks = (1:length(nx_plot), nx_plot),
               title = "Portfolio Composition", xlabel = "Asset", ylabel = "Weight",
               xrotation = 90, legend = false, kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Stacked bar / area compositions (multi-portfolio)
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_stacked_bar_composition(w::VecNum_VecVecNum,
                                                          nx::AbstractVector = 1:size(w, 1);
                                                          opts::PlottingOptions = PlottingOptions(),
                                                          kwargs...)
    wmat = isa(w, VecVecNum) ? hcat(w...) : w
    M = size(wmat, 2)
    ctg = repeat(nx; inner = M)
    return groupedbar(transpose(wmat); xticks = (1:M, 1:M), bar_position = :stack,
                      group = ctg, xlabel = "Portfolios", ylabel = "Weight",
                      title = "Portfolio Composition", legend = :outerright, kwargs...)
end

function PortfolioOptimisers.plot_stacked_area_composition(w::VecNum_VecVecNum,
                                                           nx::AbstractVector = 1:size(w, 1);
                                                           opts::PlottingOptions = PlottingOptions(),
                                                           kwargs...)
    wmat = isa(w, VecVecNum) ? hcat(w...) : w
    M = size(wmat, 2)
    return areaplot(transpose(wmat); xticks = (1:M, 1:M), label = permutedims(nx),
                    xlabel = "Portfolios", ylabel = "Weight",
                    title = "Portfolio Composition", legend = :outerright, kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Risk contribution
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_risk_contribution(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                    w::VecNum, X::MatNum_Pr,
                                                    fees::Option{<:Fees} = nothing;
                                                    nx::AbstractVector = 1:length(w),
                                                    opts::PlottingOptions = PlottingOptions(),
                                                    kwargs...)
    rc = risk_contribution(r, w, X, fees; delta = opts.delta, marginal = opts.marginal)
    if opts.percentage
        rc = rc ./ sum(rc)
    end
    return PortfolioOptimisers.plot_composition(rc, nx; opts = opts, kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Factor risk contribution
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_factor_risk_contribution(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                           w::VecNum, X::MatNum_Pr,
                                                           fees::Option{<:Fees} = nothing;
                                                           re::RegE_Reg = StepwiseRegression(),
                                                           rd::ReturnsResult = ReturnsResult(),
                                                           nf::Option{<:AbstractVector} = nothing,
                                                           opts::PlottingOptions = PlottingOptions(),
                                                           kwargs...)
    rc = factor_risk_contribution(r, w, X, fees; re = re, rd = rd, delta = opts.delta)
    factor_names = if !isnothing(nf) && length(rc) <= length(nf) + 1
        [nf; "Constant"]
    elseif !isnothing(rd.nf) && length(rc) <= length(rd.nf) + 1
        [rd.nf; "Constant"]
    else
        [string.(1:(length(rc) - 1)); "Constant"]
    end
    return PortfolioOptimisers.plot_composition(rc, factor_names; opts = opts,
                                                title = "Factor Risk Contribution",
                                                xlabel = "Factor",
                                                ylabel = "Risk Contribution", kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Drawdowns
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_drawdowns(w::ArrNum, X::MatNum,
                                            fees::Option{<:Fees} = nothing;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            ts::AbstractVector = 1:size(X, 1),
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    cmpd = opts.compound
    α = opts.alpha
    κ = opts.kappa
    rw = opts.rw
    ret = calc_net_returns(w, X, fees)
    cret = cumulative_returns(ret, cmpd)
    dd = drawdowns(cret, cmpd; cX = true) .* 100

    base_risks = 100 * if !cmpd
                       [-AverageDrawdown(; w = rw)(copy(ret)), -UlcerIndex()(copy(ret)),
                        -DrawdownatRisk(; alpha = α)(copy(ret)),
                        -ConditionalDrawdownatRisk(; alpha = α)(copy(ret)), -MaximumDrawdown()(copy(ret))]
                       else
                       [-RelativeAverageDrawdown(; w = rw)(copy(ret)), -RelativeUlcerIndex()(copy(ret)),
                        -RelativeDrawdownatRisk(; alpha = α)(copy(ret)),
                        -RelativeConditionalDrawdownatRisk(; alpha = α)(copy(ret)),
                        -RelativeMaximumDrawdown()(copy(ret))]
                       end

    conf = round((1 - α) * 100; digits = 2)
    base_labels = ["Average Drawdown: $(round(base_risks[1]; digits=2))%",
                   "Ulcer Index: $(round(base_risks[2]; digits=2))%",
                   "$(conf)% DaR: $(round(base_risks[3]; digits=2))%",
                   "$(conf)% CDaR: $(round(base_risks[4]; digits=2))%",
                   "Maximum Drawdown: $(round(base_risks[5]; digits=2))%"]

    risks = copy(base_risks)
    labels = copy(base_labels)
    if !isnothing(slv)
        if !cmpd
            push!(risks, 100 * -EntropicDrawdownatRisk(; slv = slv, alpha = α)(copy(ret)),
                  100 *
                  -RelativisticDrawdownatRisk(; slv = slv, alpha = α, kappa = κ)(copy(ret)))
        else
            push!(risks,
                  100 * -RelativeEntropicDrawdownatRisk(; slv = slv, alpha = α)(copy(ret)),
                  100 *
                  -RelativisticDrawdownatRisk(; slv = slv, alpha = α, kappa = κ)(copy(ret)))
        end
        push!(labels, "$(conf)% EDaR: $(round(risks[6]; digits=2))%",
              "$(conf)% RLDaR ($(round(κ; digits=2))): $(round(risks[7]; digits=2))%")
    end

    theme_cols = palette(:Dark2_5, length(labels) + 1)
    dd_label = "$(cmpd ? "Compounded" : "Uncompounded") Drawdown"
    f_dd = plot(ts, dd; label = dd_label,
                ylabel = "$(cmpd ? "Compounded" : "Uncompounded")\nDrawdown %",
                xlabel = "Date", linewidth = 2, yguidefontsize = 10, color = theme_cols[1],
                ylim = extrema(dd) .* [1.2, 1.01])
    for (i, (risk, lbl)) in enumerate(zip(risks, labels))
        hline!(f_dd, [risk]; label = lbl, color = theme_cols[mod1(i + 1, end)],
               linewidth = 2, legend = :bottomleft)
    end
    f_ret = plot(ts, cret;
                 ylabel = "$(cmpd ? "Compounded" : "Uncompounded")\nCumulative Returns",
                 linewidth = 2, legend = false, yguidefontsize = 10, color = theme_cols[1])
    return plot(f_ret, f_dd; layout = (2, 1), size = (750, ceil(Integer, 750 / 1.618)),
                kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Return distribution histogram
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_histogram(w::ArrNum, X::MatNum,
                                            fees::Option{<:Fees} = nothing;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    α = opts.alpha
    κ = opts.kappa
    rw = opts.rw
    ret = calc_net_returns(w, X, fees)
    T = length(ret)
    npts = opts.points == 0 ? ceil(Int, 4 * sqrt(T)) : opts.points
    mu_r = mean(ret)
    sigma_r = std(ret)
    mir, mar = extrema(ret)
    x_range = range(mir, mar; length = npts)
    mad = LowOrderMoment(; w = rw, alg = MeanAbsoluteDeviation())(w, X, fees)
    gmd = OrderedWeightsArray()(copy(ret))

    base_risks = [mu_r, mu_r - sigma_r, mu_r - mad, mu_r - gmd,
                  -ValueatRisk(; w = rw, alpha = α)(copy(ret)),
                  -ConditionalValueatRisk(; w = rw, alpha = α)(copy(ret)),
                  -OrderedWeightsArray(; w = owa_tg(T))(copy(ret))]

    conf = round((1 - α) * 100; digits = 2)
    base_labels = ["Mean: $(round(100*base_risks[1]; digits=2))%",
                   "Mean - Std ($(round(100*sigma_r; digits=2))%): $(round(100*base_risks[2]; digits=2))%",
                   "Mean - MAD ($(round(100*mad; digits=2))%): $(round(100*base_risks[3]; digits=2))%",
                   "Mean - GMD ($(round(100*gmd; digits=2))%): $(round(100*base_risks[4]; digits=2))%",
                   "$(conf)% VaR: $(round(100*base_risks[5]; digits=2))%",
                   "$(conf)% CVaR: $(round(100*base_risks[6]; digits=2))%",
                   "$(conf)% Tail Gini: $(round(100*base_risks[7]; digits=2))%"]

    risks = copy(base_risks)
    risk_labels = copy(base_labels)
    if !isnothing(slv)
        push!(risks, -EntropicValueatRisk(; w = rw, slv = slv, alpha = α)(copy(ret)),
              -RelativisticValueatRisk(; w = rw, slv = slv, alpha = α, kappa = κ)(copy(ret)))
        push!(risk_labels, "$(conf)% EVaR: $(round(100*risks[8]; digits=2))%",
              "$(conf)% RLVaR ($(round(κ; digits=2))): $(round(100*risks[9]; digits=2))%")
    end
    push!(risks, mir)
    push!(risk_labels, "Worst: $(round(100*mir; digits=2))%")

    colours = palette(:Paired_10, length(risk_labels) + 2)
    plt = histogram(ret; normalize = :pdf, label = "", color = colours[1], alpha = 0.5,
                    ylabel = "Probability Density", xlabel = "Percentage Returns",
                    kwargs...)
    for (i, (risk, lbl)) in enumerate(zip(risks, risk_labels))
        vline!([risk]; label = lbl, color = colours[i + 1], linewidth = 2)
    end
    D = StatsAPI.fit(Normal, ret)
    if opts.flag
        density!(ret;
                 label = "Normal: μ=$(round(100*mean(D); digits=2))%, σ=$(round(100*std(D); digits=2))%",
                 color = colours[end], linewidth = 2)
    else
        plot!(x_range, pdf.(D, x_range);
              label = "Normal: μ=$(round(100*mean(D); digits=2))%, σ=$(round(100*std(D); digits=2))%",
              color = colours[end], linewidth = 2)
    end
    return plt
end

## ────────────────────────────────────────────────────────────────────────────
## Network / phylogeny graph
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_network(pl::NwE_ClE_Cl, X::MatNum,
                                          nx::AbstractVector = 1:size(X, 2),
                                          w::Option{<:VecNum} = nothing;
                                          threshold::Number = 0,
                                          opts::PlottingOptions = PlottingOptions(),
                                          kwargs...)
    plr = phylogeny_matrix(pl, X)
    A = copy(plr.X)
    A[abs.(A) .<= threshold] .= 0
    node_size = if isnothing(w)
        fill(1, size(X, 2))
    else
        abs.(w) ./ maximum(abs.(w))
    end
    return graphplot(A; names = nx, node_weights = node_size, title = "Asset Network",
                     kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Centrality bar chart
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_centrality(cte::AbstractCentralityEstimator, X::MatNum,
                                             nx::AbstractVector = 1:size(X, 2);
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    plr = centrality_vector(cte, X)
    scores = plr.X
    M = length(scores)
    N, idx = _relevant_assets(scores, M, opts.N)
    top_idx = view(idx, 1:N)
    sort!(top_idx; by = i -> scores[i], rev = true)
    return bar(scores[top_idx]; xticks = (1:N, nx[top_idx]), title = "Asset Centrality",
               xlabel = "Asset", ylabel = "Centrality Score", xrotation = 90,
               legend = false, kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Correlation / covariance heatmap
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_correlation(X::MatNum, nx::AbstractVector = 1:size(X, 1);
                                              opts::PlottingOptions = PlottingOptions(),
                                              kwargs...)
    S = copy(X)
    s = LinearAlgebra.diag(S)
    if any(!isone, s)
        s .= sqrt.(s)
        StatsBase.cov2cor!(S, s)
        clim = (-1.0, 1.0)
    else
        clim = extrema(S)
    end
    N = size(S, 1)
    return heatmap(S; xticks = (1:N, nx), yticks = (1:N, nx), xrotation = 90, clim = clim,
                   color = cgrad(:Spectral), yflip = true, title = "Correlation Matrix",
                   colorbar_title = "ρ", kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## MultiPeriodPredictionResult walk-forward composition
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_composition(pred::MultiPeriodPredictionResult;
                                              opts::PlottingOptions = PlottingOptions(),
                                              kwargs...)
    folds = pred.pred
    w_mat = hcat(getproperty.(getproperty.(folds, :res), :w)...)
    nx = let rd1 = folds[1].rd
        isnothing(rd1.nx) ? (1:size(w_mat, 1)) : rd1.nx
    end
    return PortfolioOptimisers.plot_stacked_bar_composition(collect(eachcol(w_mat)), nx;
                                                            opts = opts, xlabel = "Fold",
                                                            title = "Walk-Forward Composition",
                                                            kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## PopulationPredictionResult compositions
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_composition(pred::PopulationPredictionResult;
                                              opts::PlottingOptions = PlottingOptions(),
                                              kwargs...)
    members = pred.pred
    w_mat = hcat(map(m -> if isa(m, PredictionResult)
                         m.res.w
                     else
                         mean(hcat(getproperty.(getproperty.(m.pred, :res), :w)...);
                              dims = 2)[:]
                     end, members)...)
    nx = let rd1 = isa(members[1], PredictionResult) ? members[1].rd : members[1].pred[1].rd
        isnothing(rd1.nx) ? (1:size(w_mat, 1)) : rd1.nx
    end
    return PortfolioOptimisers.plot_stacked_bar_composition(collect(eachcol(w_mat)), nx;
                                                            opts = opts,
                                                            xlabel = "Population Member",
                                                            title = "Population Composition",
                                                            kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Expected returns bar chart
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_mu(mu::VecNum, nx::AbstractVector = 1:length(mu);
                                     opts::PlottingOptions = PlottingOptions(), kwargs...)
    M = length(mu)
    N, idx = _relevant_assets(mu, M, opts.N)
    sort!(view(idx, 1:N))
    top_idx = view(idx, 1:N)
    return bar(mu[top_idx]; xticks = (1:N, string.(nx[top_idx])),
               title = "Expected Returns", xlabel = "Asset", ylabel = "μ", xrotation = 90,
               legend = false, kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Asset volatility bar chart
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_sigma(sigma::MatNum,
                                        nx::AbstractVector = 1:size(sigma, 1);
                                        opts::PlottingOptions = PlottingOptions(),
                                        kwargs...)
    # opts.percentage = true → show variance; false (default) → std dev
    vals = opts.percentage ? LinearAlgebra.diag(sigma) : sqrt.(LinearAlgebra.diag(sigma))
    ylabel_str = opts.percentage ? "Variance (σ²)" : "Volatility (σ)"
    M = length(vals)
    # sort descending by value
    idx = sortperm(vals; rev = true)
    N_show = isnothing(opts.N) ? M : clamp(ceil(Int, opts.N), 1, M)
    top_idx = idx[1:N_show]
    sort!(top_idx)  # restore asset order
    return bar(vals[top_idx]; xticks = (1:N_show, string.(nx[top_idx])),
               title = "Asset Volatility", xlabel = "Asset", ylabel = ylabel_str,
               xrotation = 90, legend = false, kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Factor loadings heatmap
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_factor_loadings(M::MatNum,
                                                  nx::AbstractVector = 1:size(M, 1),
                                                  nf::AbstractVector = 1:size(M, 2);
                                                  opts::PlottingOptions = PlottingOptions(),
                                                  kwargs...)
    Na, Nf = size(M)
    clim_val = maximum(abs, M)
    return heatmap(M; xticks = (1:Nf, string.(nf)), yticks = (1:Na, string.(nx)),
                   xrotation = 90, color = cgrad(:RdBu; rev = true),
                   clim = (-clim_val, clim_val), title = "Factor Loadings",
                   colorbar_title = "β", yflip = true, kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Factor covariance heatmap
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_factor_sigma(f_sigma::MatNum,
                                               nf::AbstractVector = 1:size(f_sigma, 1);
                                               opts::PlottingOptions = PlottingOptions(),
                                               kwargs...)
    return PortfolioOptimisers.plot_correlation(f_sigma, nf; title = "Factor Covariance",
                                                colorbar_title = "ρ_f", opts = opts,
                                                kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Eigenvalue spectrum
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_eigenspectrum(sigma::MatNum;
                                                N_obs::Option{<:Integer} = nothing,
                                                opts::PlottingOptions = PlottingOptions(),
                                                kwargs...)
    ev = eigvals(Symmetric(sigma))
    sorted_ev = sort(real.(ev); rev = true)
    N = length(sorted_ev)

    f = bar(1:N, sorted_ev; title = "Eigenspectrum", xlabel = "Component",
            ylabel = "Eigenvalue", legend = opts.flag && !isnothing(N_obs), kwargs...)

    if opts.flag && !isnothing(N_obs)
        q = N / N_obs          # N/T ratio
        s2 = tr(sigma) / N     # mean variance (≈1 for correlation)
        λ_plus = s2 * (1 + sqrt(q))^2
        hline!(f, [λ_plus]; label = "MP upper bound (λ₊=$(round(λ_plus; digits=4)))",
               linewidth = 2, color = :red, linestyle = :dash)
        if q < 1
            λ_minus = s2 * (1 - sqrt(q))^2
            hline!(f, [λ_minus]; label = "MP lower bound (λ₋=$(round(λ_minus; digits=4)))",
                   linewidth = 2, color = :orange, linestyle = :dash)
        end
    end
    return f
end

## ────────────────────────────────────────────────────────────────────────────
## Rolling risk/return measure
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                  w::VecNum, X::MatNum,
                                                  fees::Option{<:Fees} = nothing;
                                                  ts::AbstractVector = 1:size(X, 1),
                                                  opts::PlottingOptions = PlottingOptions(),
                                                  kwargs...)
    T = size(X, 1)
    window = opts.rolling == 0 ? ceil(Int, sqrt(T)) : opts.rolling
    rolling = _rolling_window_measure(r, w, X, fees, window)
    ts_rolling = ts[window:end]
    rname = string(nameof(typeof(r)))
    return plot(ts_rolling, rolling; title = "Rolling $rname (window=$window)",
                ylabel = rname, xlabel = "Date", legend = false, linewidth = 2, kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Weight stability across folds
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_weight_stability(mpred::MultiPeriodPredictionResult;
                                                   opts::PlottingOptions = PlottingOptions(),
                                                   kwargs...)
    folds = mpred.pred
    w_mat = hcat(getproperty.(getproperty.(folds, :res), :w)...)  # N_assets × N_folds
    nx = let rd1 = folds[1].rd
        isnothing(rd1.nx) ? string.(1:size(w_mat, 1)) : string.(rd1.nx)
    end
    M, K = size(w_mat)
    mean_abs = vec(mean(abs.(w_mat); dims = 2))
    N, idx = _relevant_assets(mean_abs, M, opts.N)
    top_idx = sort(view(idx, 1:N))
    labels = nx[top_idx]
    data = w_mat[top_idx, :]  # N' × K
    # Each column of data' is one asset; K rows are folds.
    return boxplot(data'; xticks = (1:N, labels), title = "Weight Stability",
                   ylabel = "Weight", legend = false, xrotation = 60, kwargs...)
end

function PortfolioOptimisers.plot_weight_stability(ppred::PopulationPredictionResult;
                                                   opts::PlottingOptions = PlottingOptions(),
                                                   kwargs...)
    members = ppred.pred
    # Pool mean weights from all members.
    w_mat = hcat(map(members) do m
                     if isa(m, PredictionResult)
                         m.res.w
                     else
                         vec(mean(hcat(getproperty.(getproperty.(m.pred, :res), :w)...);
                                  dims = 2))
                     end
                 end...)  # N_assets × N_members
    nx = let rd1 = isa(members[1], PredictionResult) ? members[1].rd : members[1].pred[1].rd
        isnothing(rd1.nx) ? string.(1:size(w_mat, 1)) : string.(rd1.nx)
    end
    M, K = size(w_mat)
    mean_abs = vec(mean(abs.(w_mat); dims = 2))
    N, idx = _relevant_assets(mean_abs, M, opts.N)
    top_idx = sort(view(idx, 1:N))
    labels = nx[top_idx]
    data = w_mat[top_idx, :]
    return boxplot(data'; xticks = (1:N, labels), title = "Population Weight Stability",
                   ylabel = "Weight", legend = false, xrotation = 60, kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Cross-validation scores
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_cv_scores(scores::AbstractVector{<:Number},
                                            labels::AbstractVector = 1:length(scores);
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    n = length(scores)
    return bar(scores; xticks = (1:n, string.(labels)), title = "CV Scores",
               ylabel = "Score", xlabel = "Fold / Member", legend = false, xrotation = 45,
               kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Portfolio turnover
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_turnover(w_series::AbstractVector{<:VecNum};
                                           ts::AbstractVector = 1:length(w_series),
                                           opts::PlottingOptions = PlottingOptions(),
                                           kwargs...)
    n = length(w_series)
    turnover = [sum(abs, w_series[t] .- w_series[t - 1]) for t in 2:n]
    return plot(ts[2:end], turnover; title = "Portfolio Turnover",
                ylabel = "Turnover (∑|Δw|)", xlabel = "Date", legend = false, linewidth = 2,
                kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## MultiPeriodPredictionResult: drawdowns, histogram, rolling measure
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_drawdowns(pred::MultiPeriodPredictionResult;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    mrd = pred.mrd
    ret = isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
    ts = isnothing(mrd.ts) ? (1:length(ret)) : mrd.ts
    w = [one(eltype(ret))]
    Xm = reshape(ret, :, 1)
    return PortfolioOptimisers.plot_drawdowns(w, Xm, nothing; slv = slv, ts = ts,
                                              opts = opts, kwargs...)
end

function PortfolioOptimisers.plot_histogram(pred::MultiPeriodPredictionResult;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    mrd = pred.mrd
    ret = isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
    w = [one(eltype(ret))]
    Xm = reshape(ret, :, 1)
    return PortfolioOptimisers.plot_histogram(w, Xm, nothing; slv = slv, opts = opts,
                                              kwargs...)
end

function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                  pred::MultiPeriodPredictionResult;
                                                  opts::PlottingOptions = PlottingOptions(),
                                                  kwargs...)
    mrd = pred.mrd
    ret = isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
    ts = isnothing(mrd.ts) ? (1:length(ret)) : mrd.ts
    w = [one(eltype(ret))]
    Xm = reshape(ret, :, 1)
    return PortfolioOptimisers.plot_rolling_measure(r, w, Xm, nothing; ts = ts, opts = opts,
                                                    kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Factor expected returns bar chart
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_factor_mu(f_mu::VecNum,
                                            nf::AbstractVector = 1:length(f_mu);
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    M = length(f_mu)
    N, idx = _relevant_assets(f_mu, M, opts.N)
    sort!(view(idx, 1:N))
    top_idx = view(idx, 1:N)
    return bar(f_mu[top_idx]; xticks = (1:N, string.(nf[top_idx])),
               title = "Factor Expected Returns", xlabel = "Factor", ylabel = "f_μ",
               xrotation = 90, legend = false, kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Benchmark overlay
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_benchmark(w::ArrNum, X::MatNum, B::VecNum_VecVecNum,
                                            fees::Option{<:Fees} = nothing;
                                            ts::AbstractVector = 1:size(X, 1),
                                            nb::Option{<:AbstractVector} = nothing,
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    ret = cumulative_returns(calc_net_returns(w, X, fees), opts.compound)
    Bmat = isa(B, VecVecNum) ? hcat(B...) : reshape(B, :, 1)
    Nb = size(Bmat, 2)
    bench_labels = isnothing(nb) ? ["Benchmark $i" for i in 1:Nb] : string.(nb[1:Nb])
    f = plot(ts, ret; label = "Portfolio", linewidth = 2, title = "Portfolio vs Benchmark",
             xlabel = "Date",
             ylabel = "$(opts.compound ? "Compound" : "Simple") Cumulative Returns",
             legend = :outerright, kwargs...)
    for i in 1:Nb
        b_ret = cumulative_returns(vec(view(Bmat, :, i)), opts.compound)
        plot!(f, ts, b_ret; label = bench_labels[i], linewidth = 1.5, linestyle = :dash)
    end
    return f
end

function PortfolioOptimisers.plot_benchmark(pred::MultiPeriodPredictionResult;
                                            opts::PlottingOptions = PlottingOptions(),
                                            kwargs...)
    mrd = pred.mrd
    if isnothing(mrd.B)
        throw(ArgumentError("multi-period prediction data has no benchmark (`B` is `nothing`)"))
    end
    ret = isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
    ts = isnothing(mrd.ts) ? (1:length(ret)) : mrd.ts
    w = [one(eltype(ret))]
    Xm = reshape(ret, :, 1)
    return PortfolioOptimisers.plot_benchmark(w, Xm, mrd.B, nothing; ts = ts, nb = mrd.nb,
                                              opts = opts, kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Coskewness heatmap
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_coskewness(sk::MatNum, nx::AbstractVector = 1:size(sk, 1);
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    N = size(sk, 1)
    N2 = size(sk, 2)
    clim_val = maximum(abs, sk)
    tick_step = max(1, div(N2, 20))
    col_ticks = collect(1:tick_step:N2)
    col_labels = if N <= 10
        ["$(nx[j])×$(nx[k])" for j in 1:N for k in 1:N][col_ticks]
    else
        string.(col_ticks)
    end
    return heatmap(sk; yticks = (1:N, string.(nx)), xticks = (col_ticks, col_labels),
                   xrotation = 90, color = cgrad(:RdBu; rev = true),
                   clim = (-clim_val, clim_val), title = "Coskewness Matrix",
                   colorbar_title = "S̃", yflip = true, kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Cokurtosis eigenspectrum / heatmap
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_cokurtosis(kt::MatNum,
                                             nx::AbstractVector = 1:isqrt(size(kt, 1));
                                             opts::PlottingOptions = PlottingOptions(),
                                             kwargs...)
    if opts.percentage
        clim_val = maximum(abs, kt)
        return heatmap(kt; color = cgrad(:RdBu; rev = true), clim = (-clim_val, clim_val),
                       title = "Cokurtosis Matrix", colorbar_title = "K̃", yflip = true,
                       kwargs...)
    else
        ev = sort(real.(eigvals(Symmetric(kt))); rev = true)
        N2 = length(ev)
        f = bar(1:N2, ev; title = "Cokurtosis Eigenspectrum", xlabel = "Component",
                ylabel = "Eigenvalue", legend = opts.flag, kwargs...)
        if opts.flag
            λ_mean = mean(ev)
            hline!(f, [λ_mean]; label = "Mean eigenvalue ($(round(λ_mean; digits=4)))",
                   linewidth = 2, color = :red, linestyle = :dash)
        end
        return f
    end
end

## ────────────────────────────────────────────────────────────────────────────
## Portfolio dashboard (multi-panel composite)
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_portfolio_dashboard(res::NonFiniteAllocationOptimisationResult,
                                                      rd::ReturnsResult;
                                                      r::PortfolioOptimisers.AbstractBaseRiskMeasure = Variance(),
                                                      slv::Option{<:Slv_VecSlv} = nothing,
                                                      opts::PlottingOptions = PlottingOptions(),
                                                      kwargs...)
    fees = hasproperty(res, :fees) ? res.fees : nothing
    w = res.w
    nx = isnothing(rd.nx) ? (1:length(w)) : rd.nx
    ts = isnothing(rd.ts) ? (1:size(rd.X, 1)) : rd.ts
    p1 = PortfolioOptimisers.plot_composition(w, nx; opts = opts)
    p2 = PortfolioOptimisers.plot_ptf_cumulative_returns(w, rd.X, fees; ts = ts,
                                                         opts = opts)
    p3 = PortfolioOptimisers.plot_risk_contribution(r, w, rd.X, fees; nx = nx, opts = opts)
    p4 = PortfolioOptimisers.plot_drawdowns(w, rd.X, fees; slv = slv, ts = ts, opts = opts)
    return plot(p1, p2, p3, p4; layout = (2, 2), size = (1200, 800), kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Cross-validation dashboard (multi-panel composite)
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_cv_dashboard(mpred::MultiPeriodPredictionResult;
                                               opts::PlottingOptions = PlottingOptions(),
                                               kwargs...)
    p1 = PortfolioOptimisers.plot_composition(mpred; opts = opts)
    p2 = PortfolioOptimisers.plot_ptf_cumulative_returns(mpred; opts = opts)
    p3 = PortfolioOptimisers.plot_turnover(mpred; opts = opts)
    p4 = PortfolioOptimisers.plot_weight_stability(mpred; opts = opts)
    return plot(p1, p2, p3, p4; layout = (2, 2), size = (1200, 800), kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Composite prior dashboard
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_prior(pr::PortfolioOptimisers.AbstractPriorResult,
                                        nx::AbstractVector = 1:length(pr.mu);
                                        opts::PlottingOptions = PlottingOptions(),
                                        kwargs...)
    p_mu = PortfolioOptimisers.plot_mu(pr.mu, nx; opts = opts, title = "Expected Returns",
                                       ylabel = "μ")
    p_sigma = PortfolioOptimisers.plot_sigma(pr.sigma, nx; opts = opts,
                                             title = "Asset Volatility", ylabel = "σ")
    p_corr = PortfolioOptimisers.plot_correlation(pr.sigma, nx; opts = opts,
                                                  title = "Correlation Matrix")
    return plot(p_mu, p_sigma, p_corr; layout = (1, 3), size = (1800, 500), kwargs...)
end

end
