module PortfolioOptimisersPlotsExt

using PortfolioOptimisers, StatsPlots, GraphRecipes, LinearAlgebra, Statistics, StatsBase,
      Clustering, Distributions, StatsAPI

import PortfolioOptimisers: ArrNum, VecNum, MatNum, Option, VecNum_VecVecNum, Slv_VecSlv,
                            MatNum_Pr, PrE_Pr, Pr_RR, HClE_HCl, VecVecNum, RegE_Reg,
                            NwE_ClE_Cl, AbstractCentralityEstimator,
                            AbstractClustersEstimator, AbstractClusteringResult,
                            AbstractBaseRiskMeasure, extract_pr, relevant_assets,
                            extract_fees, OptimisationResult

## plot_ptf_cumulative_returns
function PortfolioOptimisers.plot_ptf_cumulative_returns(net_ret::VecNum_VecVecNum;
                                                         ts::AbstractVector,
                                                         compound::Bool = false, kwargs...)
    ret = cumulative_returns(net_ret, compound)
    label = "$(compound ? "Compound" : "Simple") Cumulative Returns"
    return if isa(net_ret, VecNum)
        plot(ts, ret; title = "Portfolio", xlabel = "Date", ylabel = label, legend = false,
             kwargs...)
    else
        plt = plot(ts, ret[1]; title = "Portfolio", xlabel = "Date", ylabel = label,
                   label = "Portfolio 1", legend = true, kwargs...)
        for i in 2:length(net_ret)
            plot!(plt, ts, ret[i]; label = "Portfolio $(i)", kwargs...)
        end
        plt
    end
end
function PortfolioOptimisers.plot_ptf_cumulative_returns(w::VecNum_VecVecNum, X::MatNum,
                                                         fees::Option{<:Fees} = nothing;
                                                         ts::AbstractVector = 1:size(X, 1),
                                                         compound::Bool = false, kwargs...)
    return PortfolioOptimisers.plot_ptf_cumulative_returns(calc_net_returns(w, X, fees);
                                                           ts = ts, compound = compound,
                                                           kwargs...)
end
function PortfolioOptimisers.plot_ptf_cumulative_returns(w::VecNum_VecVecNum, pr::Pr_RR,
                                                         fees::Option{<:Fees} = nothing;
                                                         ts::AbstractVector = 1:size(pr.X,
                                                                                     1),
                                                         compound::Bool = false, kwargs...)
    if isa(pr, ReturnsResult)
        ts = isnothing(pr.ts) ? (1:size(pr.X, 1)) : pr.ts
    end
    return PortfolioOptimisers.plot_ptf_cumulative_returns(w, pr.X, fees; ts = ts,
                                                           compound = compound, kwargs...)
end
function PortfolioOptimisers.plot_ptf_cumulative_returns(res::OptimisationResult, pr::Pr_RR;
                                                         fees::Option{<:Fees} = nothing,
                                                         compound::Bool = false, kwargs...)
    pr = extract_pr(res, pr)
    fees = extract_fees(res, fees)
    return PortfolioOptimisers.plot_ptf_cumulative_returns(res.w, pr, fees;
                                                           compound = compound, kwargs...)
end
function PortfolioOptimisers.plot_ptf_cumulative_returns(res::OptimisationResult;
                                                         fees::Option{<:Fees} = nothing,
                                                         compound::Bool = false, kwargs...)
    pr = extract_pr(res, nothing)
    fees = extract_fees(res, fees)
    return PortfolioOptimisers.plot_ptf_cumulative_returns(res.w, pr, fees;
                                                           compound = compound, kwargs...)
end
function PortfolioOptimisers.plot_ptf_cumulative_returns(pred::Union{<:PredictionResult,
                                                                     <:MultiPeriodPredictionResult};
                                                         compound::Bool = false, kwargs...)
    rd = isa(pred, PredictionResult) ? pred.rd : pred.mrd
    ts = isnothing(rd.ts) ? (1:length(isa(rd.X, VecVecNum) ? first(rd.X) : rd.X)) : rd.ts
    return PortfolioOptimisers.plot_ptf_cumulative_returns(rd.X; ts = ts,
                                                           compound = compound, kwargs...)
end
function PortfolioOptimisers.plot_ptf_cumulative_returns(pred::PopulationPredictionResult;
                                                         compound::Bool = false, kwargs...)
    plt = plot(; kwargs...)
    for p in pred.pred
        plot!(plt,
              PortfolioOptimisers.plot_ptf_cumulative_returns(p; compound = compound,
                                                              kwargs...))
    end
    return plt
end
## plot_asset_cumulative_returns
function PortfolioOptimisers.plot_asset_cumulative_returns(w::VecNum, X::MatNum,
                                                           fees::Option{<:Fees} = nothing;
                                                           ts::AbstractVector = 1:size(X, 1),
                                                           nx::AbstractVector = 1:size(X, 2),
                                                           compound::Bool = false,
                                                           N::Option{<:Integer} = nothing,
                                                           kwargs...)
    net_asset_ret = calc_net_asset_returns(w, X, fees)
    ret = cumulative_returns(net_asset_ret, compound)
    M = size(X, 2)
    N, idx = relevant_assets(w, M, N)
    ret_sorted = view(ret, :, idx)
    nx_sorted = view(nx, idx)

    label_str = "$(compound ? "Compound" : "Simple") Asset Cumulative Returns"
    f = plot(; xlabel = "Date", ylabel = label_str)
    for i in 1:N
        plot!(f, ts, view(ret_sorted, :, i); label = string(nx_sorted[i]))
    end
    if M > N
        rest_idx = view(idx, (N + 1):M)
        rest_ret = cumulative_returns(calc_net_returns(view(w, rest_idx),
                                                       view(X, :, rest_idx),
                                                       PortfolioOptimisers.port_opt_view(fees,
                                                                                         rest_idx)),
                                      compound)
        plot!(f, ts, rest_ret; label = "Others")
    end
    plot!(f; legend = :outerright, kwargs...)
    return f
end
function PortfolioOptimisers.plot_asset_cumulative_returns(w::VecNum, pr::Pr_RR,
                                                           fees::Option{<:Fees} = nothing;
                                                           ts::AbstractVector = 1:size(pr.X,
                                                                                       1),
                                                           nx::AbstractVector = 1:size(pr.X,
                                                                                       2),
                                                           compound::Bool = false,
                                                           N::Option{<:Integer} = nothing,
                                                           kwargs...)
    if isa(pr, ReturnsResult)
        ts = isnothing(pr.ts) ? (1:size(pr.X, 1)) : pr.ts
        nx = isnothing(pr.nx) ? (1:size(pr.X, 2)) : pr.nx
    end
    return PortfolioOptimisers.plot_asset_cumulative_returns(w, pr.X, fees; ts = ts,
                                                             nx = nx, compound = compound,
                                                             N = N, kwargs...)
end
function PortfolioOptimisers.plot_asset_cumulative_returns(res::OptimisationResult,
                                                           pr::Pr_RR;
                                                           fees::Option{<:Fees} = nothing,
                                                           compound::Bool = false,
                                                           N::Option{<:Integer} = nothing,
                                                           kwargs...)
    pr = extract_pr(res, pr)
    fees = extract_fees(res, fees)
    return PortfolioOptimisers.plot_asset_cumulative_returns(res.w, pr, fees;
                                                             compound = compound, N = N,
                                                             kwargs...)
end
function PortfolioOptimisers.plot_asset_cumulative_returns(res::OptimisationResult;
                                                           fees::Option{<:Fees} = nothing,
                                                           compound::Bool = false,
                                                           N::Option{<:Integer} = nothing,
                                                           kwargs...)
    pr = extract_pr(res, nothing)
    fees = extract_fees(res, fees)
    return PortfolioOptimisers.plot_asset_cumulative_returns(res.w, pr, fees;
                                                             compound = compound, N = N,
                                                             kwargs...)
end
function PortfolioOptimisers.plot_asset_cumulative_returns(pred::PredictionResult;
                                                           compound::Bool = false,
                                                           N::Option{<:Integer} = nothing,
                                                           kwargs...)
    return PortfolioOptimisers.plot_asset_cumulative_returns(pred.res; compound = compound,
                                                             N = N, kwargs...)
end
function PortfolioOptimisers.plot_asset_cumulative_returns(pred::MultiPeriodPredictionResult;
                                                           compound::Bool = false,
                                                           N::Option{<:Integer} = nothing,
                                                           kwargs...)
    res_vec = pred.res
    X = Vector{eltype(first(res_vec).rd.X)}[]
    ts = Vector{eltype(first(res_vec).rd.ts)}[]
    nx = first(res_vec).rd.nx
    M = length(nx)
    mean_w = zeros(length(first(res_vec).w), M)
    for res in res_vec
        w = res.w
        pr = extract_pr(res)
        fees = extract_fees(res)
        mean_w .+= w
        net_asset_ret = calc_net_asset_returns(w, pr.X, fees)
        ret = cumulative_returns(net_asset_ret, compound)
        append!(X, vec(ret))
        append!(ts, res.rd.ts)
    end
    mean_w ./= length(res_vec)
    X = reshape(X, length(ts), M)
    N, idx = relevant_assets(mean_w, M, N)
    ret_sorted = view(X, :, idx)
    nx_sorted = view(nx, idx)
    label_str = "$(compound ? "Compound" : "Simple") Asset Cumulative Returns"
    f = plot(; xlabel = "Date", ylabel = label_str)
    for i in 1:N
        plot!(f, ts, view(ret_sorted, :, i); label = string(nx_sorted[i]))
    end
    if M > N
        rest_idx = view(idx, (N + 1):M)
        rest_ret = vec(sum(X; dims = 2) - sum(view(X, :, 1:N); dims = 2))
        plot!(f, ts, rest_ret; label = "Others")
    end
    plot!(f; legend = :outerright, kwargs...)
    return f
end
function PortfolioOptimisers.plot_asset_cumulative_returns(pred::PopulationPredictionResult;
                                                           compound::Bool = false,
                                                           N::Option{<:Integer} = nothing,
                                                           kwargs...)
    plt = plot(; kwargs...)
    for p in pred.pred
        plot!(plt,
              PortfolioOptimisers.plot_asset_cumulative_returns(p; compound = compound,
                                                                N = N, kwargs...))
    end
    return plt
end
## plot_composition
function PortfolioOptimisers.plot_composition(w::VecNum, nx::AbstractVector = 1:length(w);
                                              N::Option{<:Number} = nothing, kwargs...)
    M = length(w)
    N, idx = relevant_assets(w, M, N)
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
function PortfolioOptimisers.plot_composition(res::OptimisationResult, rd::ReturnsResult;
                                              N::Option{<:Number} = nothing, kwargs...)
    nx = isnothing(rd.nx) ? (1:length(res.w)) : rd.nx
    return PortfolioOptimisers.plot_composition(res.w, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_composition(res::OptimisationResult,
                                              pr::PortfolioOptimisers.AbstractPriorResult;
                                              N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_composition(res.w, 1:length(res.w); N = N, kwargs...)
end
function PortfolioOptimisers.plot_composition(pred::PredictionResult;
                                              N::Option{<:Number} = nothing, kwargs...)
    nx = isnothing(pred.rd.nx) ? (1:length(pred.res.w)) : pred.rd.nx
    return PortfolioOptimisers.plot_composition(pred.res.w, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_composition(pred::MultiPeriodPredictionResult;
                                              N::Option{<:Number} = nothing, kwargs...)
    folds = pred.pred
    w_mat = hcat(getproperty.(getproperty.(folds, :res), :w)...)
    nx = let rd1 = folds[1].rd
        isnothing(rd1.nx) ? (1:size(w_mat, 1)) : rd1.nx
    end
    return PortfolioOptimisers.plot_stacked_bar_composition(collect(eachcol(w_mat)), nx;
                                                            xlabel = "Fold",
                                                            title = "Walk-ForwardSelection Composition",
                                                            kwargs...)
end
function PortfolioOptimisers.plot_composition(pred::PopulationPredictionResult;
                                              N::Option{<:Number} = nothing, kwargs...)
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
                                                            xlabel = "Population Member",
                                                            title = "Population Composition",
                                                            kwargs...)
end
## plot_stacked_bar_composition / plot_stacked_area_composition
function PortfolioOptimisers.plot_stacked_bar_composition(w::VecNum_VecVecNum,
                                                          nx::AbstractVector = 1:size(w, 1);
                                                          kwargs...)
    wmat = isa(w, VecVecNum) ? hcat(w...) : w
    M = size(wmat, 2)
    ctg = repeat(nx; inner = M)
    return groupedbar(transpose(wmat); xticks = (1:M, 1:M), bar_position = :stack,
                      group = ctg, xlabel = "Portfolios", ylabel = "Weight",
                      title = "Portfolio Composition", legend = :outerright, kwargs...)
end
function PortfolioOptimisers.plot_stacked_bar_composition(res_vec::AbstractVector{<:OptimisationResult},
                                                          rd::ReturnsResult; kwargs...)
    w = getproperty.(res_vec, :w)
    nx = isnothing(rd.nx) ? (1:size(w, 1)) : rd.nx
    return PortfolioOptimisers.plot_stacked_bar_composition(w, nx; kwargs...)
end
function PortfolioOptimisers.plot_stacked_area_composition(w::VecNum_VecVecNum,
                                                           nx::AbstractVector = 1:size(w, 1);
                                                           kwargs...)
    wmat = isa(w, VecVecNum) ? hcat(w...) : w
    M = size(wmat, 2)
    return areaplot(transpose(wmat); xticks = (1:M, 1:M), label = permutedims(nx),
                    xlabel = "Portfolios", ylabel = "Weight",
                    title = "Portfolio Composition", legend = :outerright, kwargs...)
end
function PortfolioOptimisers.plot_stacked_area_composition(res_vec::AbstractVector{<:OptimisationResult},
                                                           rd::ReturnsResult; kwargs...)
    w = getproperty.(res_vec, :w)
    nx = isnothing(rd.nx) ? (1:size(w, 1)) : rd.nx
    return PortfolioOptimisers.plot_stacked_area_composition(w, nx; kwargs...)
end
## plot_risk_contribution
function PortfolioOptimisers.plot_risk_contribution(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                    w::VecNum, rd::ReturnsResult,
                                                    fees::Option{<:Fees} = nothing;
                                                    delta::Number = 1e-6,
                                                    marginal::Bool = false,
                                                    percentage::Bool = true,
                                                    N::Option{<:Number} = nothing,
                                                    kwargs...)
    nx = isnothing(rd.nx) ? (1:size(rd.X, 2)) : rd.nx
    return PortfolioOptimisers.plot_risk_contribution(r, w, rd.X, fees; nx = nx,
                                                      delta = delta, marginal = marginal,
                                                      percentage = percentage, N = N,
                                                      kwargs...)
end
function PortfolioOptimisers.plot_risk_contribution(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                    res::OptimisationResult,
                                                    rd::ReturnsResult; delta::Number = 1e-6,
                                                    marginal::Bool = false,
                                                    percentage::Bool = true,
                                                    N::Option{<:Number} = nothing,
                                                    kwargs...)
    fees = extract_fees(res, nothing)
    return PortfolioOptimisers.plot_risk_contribution(r, res.w, rd, fees; delta = delta,
                                                      marginal = marginal,
                                                      percentage = percentage, N = N,
                                                      kwargs...)
end
function PortfolioOptimisers.plot_risk_contribution(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                    res::OptimisationResult,
                                                    pr::PortfolioOptimisers.AbstractPriorResult;
                                                    nx::AbstractVector = 1:length(res.w),
                                                    delta::Number = 1e-6,
                                                    marginal::Bool = false,
                                                    percentage::Bool = true,
                                                    N::Option{<:Number} = nothing,
                                                    kwargs...)
    fees = extract_fees(res, nothing)
    return PortfolioOptimisers.plot_risk_contribution(r, res.w, pr.X, fees; nx = nx,
                                                      delta = delta, marginal = marginal,
                                                      percentage = percentage, N = N,
                                                      kwargs...)
end
function PortfolioOptimisers.plot_risk_contribution(::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                    ::PredictionResult; kwargs...)
    return throw(ArgumentError("`plot_risk_contribution(r, pred::PredictionResult)` is not supported: `PredictionReturnsResult` stores portfolio returns, not raw asset returns. Call `plot_risk_contribution(r, pred.res.w, rd::ReturnsResult, ...)` with the original returns data."))
end
## plot_factor_risk_contribution
function PortfolioOptimisers.plot_factor_risk_contribution(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                           res::OptimisationResult,
                                                           rd::ReturnsResult;
                                                           re::RegE_Reg = StepwiseRegression(),
                                                           delta::Number = 1e-6,
                                                           N::Option{<:Number} = nothing,
                                                           kwargs...)
    fees = extract_fees(res, nothing)
    return PortfolioOptimisers.plot_factor_risk_contribution(r, res.w, rd.X, fees; re = re,
                                                             rd = rd, delta = delta, N = N,
                                                             kwargs...)
end
function PortfolioOptimisers.plot_factor_risk_contribution(::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                           ::PredictionResult; kwargs...)
    return throw(ArgumentError("`plot_factor_risk_contribution(r, pred::PredictionResult)` is not supported: `PredictionReturnsResult` stores portfolio returns, not raw asset returns. Call `plot_factor_risk_contribution(r, pred.res.w, rd::ReturnsResult, ...)` with the original returns data."))
end
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
        abs.(w) ./ maximum(abs.(w))
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
    return PortfolioOptimisers.plot_network(pl, pr.X, nx, w; kwargs...)
end
function PortfolioOptimisers.plot_network(pl::NwE_ClE_Cl, res::OptimisationResult;
                                          rd::Option{<:Pr_RR} = nothing,
                                          nx::AbstractVector = 1:length(res.w), kwargs...)
    pr = if isa(rd, ReturnsResult)
        if !isnothing(rd.nx)
            nx = rd.nx
        end
        rd
    elseif isnothing(rd)
        extract_pr(res, pr)
    end
    return PortfolioOptimisers.plot_network(pl, rd.X, nx, res.w; kwargs...)
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
                                             dims::Integer = 1, kwargs...)
    clr = clusterise(cle, X; dims = dims)
    return PortfolioOptimisers.plot_dendrogram(clr, nx; kwargs...)
end
function PortfolioOptimisers.plot_dendrogram(cle::HClE_HCl, pr::PortfolioOptimisers.Pr_RR,
                                             nx::AbstractVector = 1:size(pr.X, 2);
                                             dims::Integer = 1, kwargs...)
    if isa(pr, ReturnsResult) && !isnothing(pr.nx)
        nx = pr.nx
    end
    return PortfolioOptimisers.plot_dendrogram(cle, pr.X, nx; dims = dims, kwargs...)
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
    dend2 = plot(clr.res; yticks = false, orientation = :horizontal, xrotation = 90,
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
        h = maximum(clr.res.heights[i3])
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
                                           nx::AbstractVector = 1:size(pr.X, 2);
                                           dims::Integer = 1, kwargs...)
    if isa(pr, ReturnsResult) && !isnothing(pr.nx)
        nx = pr.nx
    end
    return PortfolioOptimisers.plot_clusters(cle, pr.X, nx; kwargs...)
end
## plot_drawdowns
function PortfolioOptimisers.plot_drawdowns(w::ArrNum, rd::ReturnsResult,
                                            fees::Option{<:Fees} = nothing;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            compound::Bool = false, alpha::Number = 0.05,
                                            kappa::Number = 0.3, rw = nothing, kwargs...)
    ts = isnothing(rd.ts) ? (1:size(rd.X, 1)) : rd.ts
    return PortfolioOptimisers.plot_drawdowns(w, rd.X, fees; slv = slv, ts = ts,
                                              compound = compound, alpha = alpha,
                                              kappa = kappa, rw = rw, kwargs...)
end
function PortfolioOptimisers.plot_drawdowns(res::OptimisationResult, rd::ReturnsResult;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            compound::Bool = false, alpha::Number = 0.05,
                                            kappa::Number = 0.3, rw = nothing, kwargs...)
    fees = extract_fees(res, nothing)
    return PortfolioOptimisers.plot_drawdowns(res.w, rd, fees; slv = slv,
                                              compound = compound, alpha = alpha,
                                              kappa = kappa, rw = rw, kwargs...)
end
function PortfolioOptimisers.plot_drawdowns(pred::PredictionResult;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            compound::Bool = false, alpha::Number = 0.05,
                                            kappa::Number = 0.3, rw = nothing, kwargs...)
    rd = pred.rd
    ret = isa(rd.X, VecVecNum) ? first(rd.X) : rd.X
    ts = isnothing(rd.ts) ? (1:length(ret)) : rd.ts
    return PortfolioOptimisers.plot_drawdowns(ret; slv = slv, ts = ts, compound = compound,
                                              alpha = alpha, kappa = kappa, rw = rw,
                                              kwargs...)
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
                                           factory::Bool = true, kwargs...)
    if factory
        x = PortfolioOptimisers.factory(x, pr, slv)
        y = PortfolioOptimisers.factory(y, pr, slv)
        z = isnothing(z) ? nothing : PortfolioOptimisers.factory(z, pr, slv)
        c = PortfolioOptimisers.factory(c, pr, slv)
    end
    xr = expected_risk(x, w, pr, fees)
    yr = expected_risk(y, w, pr, fees)
    zr = isnothing(z) ? nothing : expected_risk(z, w, pr, fees)
    cr = expected_risk(c, w, pr, fees)
    return if isnothing(zr)
        scatter(xr, yr; zcolor = cr, title = "Pareto Front", xlabel = "X", ylabel = "Y",
                colorbar_title = "C", label = nothing, legend = true, kwargs...)
    else
        scatter(xr, yr, zr; zcolor = cr, title = "Pareto Front", xlabel = "X", ylabel = "Y",
                zlabel = "Z", colorbar_title = "C", label = nothing, legend = true,
                kwargs...)
    end
end
function PortfolioOptimisers.plot_measures(res_vec::AbstractVector{<:OptimisationResult},
                                           pr::Option{<:Pr_RR} = nothing;
                                           x::PortfolioOptimisers.AbstractBaseRiskMeasure = Variance(),
                                           y::PortfolioOptimisers.AbstractBaseRiskMeasure = ExpectedReturn(),
                                           z::Option{<:PortfolioOptimisers.AbstractBaseRiskMeasure} = nothing,
                                           c::PortfolioOptimisers.AbstractBaseRiskMeasure = ExpectedReturnRiskRatio(;
                                                                                                                    rk = x,
                                                                                                                    rt = ArithmeticReturn(),
                                                                                                                    rf = 0),
                                           slv::Option{<:Slv_VecSlv} = nothing,
                                           fees::Option{<:Fees} = nothing,
                                           factory::Bool = true, kwargs...)
    pr = ifelse(isnothing(pr), nothing, pr)
    slv = ifelse(isnothing(slv), nothing, slv)
    pr = extract_pr.(res_vec, pr)
    fees = extract_fees.(res_vec, fees)
    w = getproperty.(res_vec, :w)
    if factory
        x = PortfolioOptimisers.factory.(x, pr, slv)
        y = PortfolioOptimisers.factory.(y, pr, slv)
        z = isnothing(z) ? nothing : PortfolioOptimisers.factory.(z, pr, slv)
        c = PortfolioOptimisers.factory.(c, pr, slv)
    end
    xr = expected_risk.(x, w, pr, fees)
    yr = expected_risk.(y, w, pr, fees)
    zr = isnothing(z) ? nothing : expected_risk.(z, w, pr, fees)
    cr = expected_risk.(c, w, pr, fees)
    return if isnothing(zr)
        scatter(xr, yr; zcolor = cr, title = "Pareto Front", xlabel = "X", ylabel = "Y",
                colorbar_title = "C", label = nothing, legend = true, kwargs...)
    else
        scatter(xr, yr, zr; zcolor = cr, title = "Pareto Front", xlabel = "X", ylabel = "Y",
                zlabel = "Z", colorbar_title = "C", label = nothing, legend = true,
                kwargs...)
    end
end
function PortfolioOptimisers.plot_measures(ppred::Union{<:PredictionResult,
                                                        <:MultiPeriodPredictionResult,
                                                        <:PopulationPredictionResult};
                                           x::PortfolioOptimisers.AbstractBaseRiskMeasure = ConditionalValueatRisk(),
                                           y::PortfolioOptimisers.AbstractBaseRiskMeasure = MeanReturn(),
                                           z::Option{<:PortfolioOptimisers.AbstractBaseRiskMeasure} = nothing,
                                           c::PortfolioOptimisers.AbstractBaseRiskMeasure = MeanReturnRiskRatio(;
                                                                                                                rk = x,
                                                                                                                rt = MeanReturn(),
                                                                                                                rf = 0),
                                           slv::Option{<:Slv_VecSlv} = nothing,
                                           factory::Bool = true, plt = nothing, kwargs...)
    if factory
        x = PortfolioOptimisers.factory(x, nothing, slv)
        y = PortfolioOptimisers.factory(y, nothing, slv)
        z = isnothing(z) ? nothing : PortfolioOptimisers.factory(z, nothing, slv)
        c = PortfolioOptimisers.factory(c, nothing, slv)
    end
    xr = expected_risk(x, ppred)
    yr = expected_risk(y, ppred)
    zr = isnothing(z) ? nothing : expected_risk(z, ppred)
    cr = expected_risk(c, ppred)
    return if isnothing(zr)
        if isnothing(plt)
            scatter(xr, yr; zcolor = cr, title = "Pareto Front", xlabel = "X", ylabel = "Y",
                    colorbar_title = "C", label = nothing, legend = true, kwargs...)
        else
            scatter!(xr, yr; zcolor = cr, title = "Pareto Front", xlabel = "X",
                     ylabel = "Y", colorbar_title = "C", label = nothing, legend = true,
                     kwargs...)
        end
    else
        if isnothing(plt)
            scatter(xr, yr, zr; zcolor = cr, title = "Pareto Front", xlabel = "X",
                    ylabel = "Y", zlabel = "Z", colorbar_title = "C", label = nothing,
                    legend = true, kwargs...)
        else
            scatter!(plt, xr, yr, zr; zcolor = cr, title = "Pareto Front", xlabel = "X",
                     ylabel = "Y", zlabel = "Z", colorbar_title = "C", label = nothing,
                     legend = true, kwargs...)
        end
    end
end
## plot_histogram
function PortfolioOptimisers.plot_histogram(w::ArrNum, rd::ReturnsResult,
                                            fees::Option{<:Fees} = nothing;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            alpha::Number = 0.05, kappa::Number = 0.3,
                                            rw = nothing, points::Integer = 0,
                                            reference::Bool = true, kwargs...)
    return PortfolioOptimisers.plot_histogram(w, rd.X, fees; slv = slv, alpha = alpha,
                                              kappa = kappa, rw = rw, points = points,
                                              reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_histogram(res::OptimisationResult, rd::ReturnsResult;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            alpha::Number = 0.05, kappa::Number = 0.3,
                                            rw = nothing, points::Integer = 0,
                                            reference::Bool = true, kwargs...)
    fees = extract_fees(res, nothing)
    return PortfolioOptimisers.plot_histogram(res.w, rd.X, fees; slv = slv, alpha = alpha,
                                              kappa = kappa, rw = rw, points = points,
                                              reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_histogram(pred::PredictionResult;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            alpha::Number = 0.05, kappa::Number = 0.3,
                                            rw = nothing, points::Integer = 0,
                                            reference::Bool = true, kwargs...)
    rd = pred.rd
    ret = isa(rd.X, VecVecNum) ? first(rd.X) : rd.X
    return PortfolioOptimisers.plot_histogram(ret; slv = slv, alpha = alpha, kappa = kappa,
                                              rw = rw, points = points,
                                              reference = reference, kwargs...)
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
    return PortfolioOptimisers.plot_centrality(cte, pr.X, nx; N = N,
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
## plot_correlation
function PortfolioOptimisers.plot_correlation(pr::PortfolioOptimisers.AbstractPriorResult,
                                              nx::AbstractVector = 1:size(pr.sigma, 1);
                                              kwargs...)
    return PortfolioOptimisers.plot_correlation(pr.sigma, nx; kwargs...)
end
function PortfolioOptimisers.plot_correlation(pr::PortfolioOptimisers.AbstractPriorResult,
                                              rd::ReturnsResult; kwargs...)
    nx = isnothing(rd.nx) ? (1:size(pr.sigma, 1)) : rd.nx
    return PortfolioOptimisers.plot_correlation(pr.sigma, nx; kwargs...)
end
function PortfolioOptimisers.plot_correlation(res::OptimisationResult, rd::ReturnsResult;
                                              kwargs...)
    return PortfolioOptimisers.plot_correlation(extract_pr(res), rd; kwargs...)
end
function PortfolioOptimisers.plot_correlation(res::OptimisationResult; kwargs...)
    return PortfolioOptimisers.plot_correlation(extract_pr(res); kwargs...)
end
function PortfolioOptimisers.plot_correlation(pred::PredictionResult, rd::ReturnsResult;
                                              kwargs...)
    return PortfolioOptimisers.plot_correlation(extract_pr(pred.res), rd; kwargs...)
end
function PortfolioOptimisers.plot_correlation(pred::PredictionResult; kwargs...)
    return PortfolioOptimisers.plot_correlation(extract_pr(pred.res); kwargs...)
end
## plot_mu
function PortfolioOptimisers.plot_mu(pr::PortfolioOptimisers.AbstractPriorResult,
                                     nx::AbstractVector = 1:length(pr.mu);
                                     N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_mu(pr.mu, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_mu(pr::PortfolioOptimisers.AbstractPriorResult,
                                     rd::ReturnsResult; N::Option{<:Number} = nothing,
                                     kwargs...)
    nx = isnothing(rd.nx) ? (1:length(pr.mu)) : rd.nx
    return PortfolioOptimisers.plot_mu(pr.mu, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_mu(res::OptimisationResult; N::Option{<:Number} = nothing,
                                     kwargs...)
    return PortfolioOptimisers.plot_mu(extract_pr(res); N = N, kwargs...)
end
function PortfolioOptimisers.plot_mu(res::OptimisationResult, rd::ReturnsResult;
                                     N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_mu(extract_pr(res), rd; N = N, kwargs...)
end
function PortfolioOptimisers.plot_mu(pred::PredictionResult; N::Option{<:Number} = nothing,
                                     kwargs...)
    return PortfolioOptimisers.plot_mu(extract_pr(pred.res); N = N, kwargs...)
end
function PortfolioOptimisers.plot_mu(pred::PredictionResult, rd::ReturnsResult;
                                     N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_mu(extract_pr(pred.res), rd; N = N, kwargs...)
end
## plot_sigma
function PortfolioOptimisers.plot_sigma(pr::PortfolioOptimisers.AbstractPriorResult,
                                        nx::AbstractVector = 1:size(pr.sigma, 1);
                                        N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_sigma(pr.sigma, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_sigma(pr::PortfolioOptimisers.AbstractPriorResult,
                                        rd::ReturnsResult; N::Option{<:Number} = nothing,
                                        kwargs...)
    nx = isnothing(rd.nx) ? (1:size(pr.sigma, 1)) : rd.nx
    return PortfolioOptimisers.plot_sigma(pr.sigma, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_sigma(res::OptimisationResult, rd::ReturnsResult;
                                        N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_sigma(extract_pr(res), rd; N = N, kwargs...)
end
function PortfolioOptimisers.plot_sigma(res::OptimisationResult;
                                        N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_sigma(extract_pr(res); N = N, kwargs...)
end
function PortfolioOptimisers.plot_sigma(pred::PredictionResult;
                                        N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_sigma(extract_pr(pred.res); N = N, kwargs...)
end
function PortfolioOptimisers.plot_sigma(pred::PredictionResult, rd::ReturnsResult;
                                        N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_sigma(extract_pr(pred.res), rd; N = N, kwargs...)
end
## plot_factor_loadings
function PortfolioOptimisers.plot_factor_loadings(pr::PortfolioOptimisers.AbstractPriorResult,
                                                  nx::Option{<:AbstractVector} = nothing,
                                                  nf::Option{<:AbstractVector} = nothing;
                                                  kwargs...)
    if isnothing(pr.rr)
        throw(ArgumentError("prior has no factor regression model (rr is nothing); pass M directly"))
    end
    nx_use = isnothing(nx) ? (1:size(pr.rr.M, 1)) : nx
    nf_use = isnothing(nf) ? (1:size(pr.rr.M, 2)) : nf
    return PortfolioOptimisers.plot_factor_loadings(pr.rr.M, nx_use, nf_use; kwargs...)
end
function PortfolioOptimisers.plot_factor_loadings(pr::PortfolioOptimisers.AbstractPriorResult,
                                                  rd::ReturnsResult; kwargs...)
    if isnothing(pr.rr)
        throw(ArgumentError("prior has no factor regression model (rr is nothing); pass M directly"))
    end
    nx = isnothing(rd.nx) ? (1:size(pr.rr.M, 1)) : rd.nx
    nf = isnothing(rd.nf) ? (1:size(pr.rr.M, 2)) : rd.nf
    return PortfolioOptimisers.plot_factor_loadings(pr.rr.M, nx, nf; kwargs...)
end
function PortfolioOptimisers.plot_factor_loadings(res::OptimisationResult,
                                                  rd::ReturnsResult; kwargs...)
    return PortfolioOptimisers.plot_factor_loadings(extract_pr(res), rd; kwargs...)
end
function PortfolioOptimisers.plot_factor_loadings(res::OptimisationResult; kwargs...)
    return PortfolioOptimisers.plot_factor_loadings(extract_pr(res); kwargs...)
end
function PortfolioOptimisers.plot_factor_loadings(pred::PredictionResult, rd::ReturnsResult;
                                                  kwargs...)
    return PortfolioOptimisers.plot_factor_loadings(extract_pr(pred.res), rd; kwargs...)
end
function PortfolioOptimisers.plot_factor_loadings(pred::PredictionResult; kwargs...)
    return PortfolioOptimisers.plot_factor_loadings(extract_pr(pred.res); kwargs...)
end
## plot_factor_sigma
function PortfolioOptimisers.plot_factor_sigma(pr::PortfolioOptimisers.AbstractPriorResult,
                                               nf::AbstractVector = 1:size(pr.f_sigma, 1);
                                               kwargs...)
    if isnothing(pr.f_sigma)
        throw(ArgumentError("prior has no factor covariance (f_sigma is nothing); pass f_sigma directly"))
    end
    return PortfolioOptimisers.plot_factor_sigma(pr.f_sigma, nf; kwargs...)
end
function PortfolioOptimisers.plot_factor_sigma(pr::PortfolioOptimisers.AbstractPriorResult,
                                               rd::ReturnsResult; kwargs...)
    if isnothing(pr.f_sigma)
        throw(ArgumentError("prior has no factor covariance (f_sigma is nothing); pass f_sigma directly"))
    end
    nf = isnothing(rd.nf) ? (1:size(pr.f_sigma, 1)) : rd.nf
    return PortfolioOptimisers.plot_factor_sigma(pr.f_sigma, nf; kwargs...)
end
function PortfolioOptimisers.plot_factor_sigma(res::OptimisationResult, rd::ReturnsResult;
                                               kwargs...)
    return PortfolioOptimisers.plot_factor_sigma(extract_pr(res), rd; kwargs...)
end
function PortfolioOptimisers.plot_factor_sigma(res::OptimisationResult; kwargs...)
    return PortfolioOptimisers.plot_factor_sigma(extract_pr(res); kwargs...)
end
function PortfolioOptimisers.plot_factor_sigma(pred::PredictionResult, rd::ReturnsResult;
                                               kwargs...)
    return PortfolioOptimisers.plot_factor_sigma(extract_pr(pred.res), rd; kwargs...)
end
function PortfolioOptimisers.plot_factor_sigma(pred::PredictionResult; kwargs...)
    return PortfolioOptimisers.plot_factor_sigma(extract_pr(pred.res); kwargs...)
end
## plot_eigenspectrum
function PortfolioOptimisers.plot_eigenspectrum(pr::PortfolioOptimisers.AbstractPriorResult;
                                                reference::Bool = true, kwargs...)
    return PortfolioOptimisers.plot_eigenspectrum(pr.sigma; reference = reference,
                                                  kwargs...)
end
function PortfolioOptimisers.plot_eigenspectrum(pr::PortfolioOptimisers.AbstractPriorResult,
                                                rd::ReturnsResult; reference::Bool = true,
                                                kwargs...)
    T = isnothing(rd.X) ? nothing : size(rd.X, 1)
    return PortfolioOptimisers.plot_eigenspectrum(pr.sigma; N_obs = T,
                                                  reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_eigenspectrum(res::OptimisationResult;
                                                reference::Bool = true, kwargs...)
    return PortfolioOptimisers.plot_eigenspectrum(extract_pr(res); reference = reference,
                                                  kwargs...)
end
function PortfolioOptimisers.plot_eigenspectrum(res::OptimisationResult, rd::ReturnsResult;
                                                reference::Bool = true, kwargs...)
    return PortfolioOptimisers.plot_eigenspectrum(extract_pr(res), rd;
                                                  reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_eigenspectrum(pred::PredictionResult;
                                                reference::Bool = true, kwargs...)
    return PortfolioOptimisers.plot_eigenspectrum(extract_pr(pred.res);
                                                  reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_eigenspectrum(pred::PredictionResult, rd::ReturnsResult;
                                                reference::Bool = true, kwargs...)
    return PortfolioOptimisers.plot_eigenspectrum(extract_pr(pred.res), rd;
                                                  reference = reference, kwargs...)
end
## plot_rolling_measure
function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                  w::VecNum, rd::ReturnsResult,
                                                  fees::Option{<:Fees} = nothing;
                                                  rolling::Integer = 0, kwargs...)
    ts = isnothing(rd.ts) ? (1:size(rd.X, 1)) : rd.ts
    return PortfolioOptimisers.plot_rolling_measure(r, w, rd.X, fees; ts = ts,
                                                    rolling = rolling, kwargs...)
end
function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                  res::OptimisationResult,
                                                  rd::ReturnsResult; rolling::Integer = 0,
                                                  kwargs...)
    fees = extract_fees(res, nothing)
    return PortfolioOptimisers.plot_rolling_measure(r, res.w, rd, fees; rolling = rolling,
                                                    kwargs...)
end
function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                  pred::PredictionResult;
                                                  rolling::Integer = 0, kwargs...)
    rd = pred.rd
    ret = isa(rd.X, VecVecNum) ? first(rd.X) : rd.X
    ts = isnothing(rd.ts) ? (1:length(ret)) : rd.ts
    return PortfolioOptimisers.plot_rolling_measure(r, ret; ts = ts, rolling = rolling,
                                                    kwargs...)
end
## plot_cv_scores
function PortfolioOptimisers.plot_cv_scores(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                            mpred::MultiPeriodPredictionResult; kwargs...)
    scores = [expected_risk(r, p) for p in mpred.pred]
    return PortfolioOptimisers.plot_cv_scores(scores, 1:length(scores); kwargs...)
end
function PortfolioOptimisers.plot_cv_scores(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                            ppred::PopulationPredictionResult; kwargs...)
    scores = [expected_risk(r, m) for m in ppred.pred]
    return PortfolioOptimisers.plot_cv_scores(scores, 1:length(scores); kwargs...)
end
## plot_turnover
function PortfolioOptimisers.plot_turnover(mpred::MultiPeriodPredictionResult; kwargs...)
    folds = mpred.pred
    w_series = getproperty.(getproperty.(folds, :res), :w)
    ts = if isnothing(folds[1].rd.ts)
        1:length(w_series)
    else
        [f.rd.ts[end] for f in folds]
    end
    return PortfolioOptimisers.plot_turnover(w_series; ts = ts, kwargs...)
end
## plot_prior
function PortfolioOptimisers.plot_prior(pr::PortfolioOptimisers.AbstractPriorResult,
                                        rd::ReturnsResult; N::Option{<:Number} = nothing,
                                        kwargs...)
    nx = isnothing(rd.nx) ? (1:length(pr.mu)) : rd.nx
    return PortfolioOptimisers.plot_prior(pr, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_prior(res::OptimisationResult;
                                        N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_prior(extract_pr(res); N = N, kwargs...)
end
function PortfolioOptimisers.plot_prior(res::OptimisationResult, rd::ReturnsResult;
                                        N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_prior(extract_pr(res), rd; N = N, kwargs...)
end
function PortfolioOptimisers.plot_prior(pred::PredictionResult;
                                        N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_prior(extract_pr(pred.res); N = N, kwargs...)
end
function PortfolioOptimisers.plot_prior(pred::PredictionResult, rd::ReturnsResult;
                                        N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_prior(extract_pr(pred.res), rd; N = N, kwargs...)
end
## plot_factor_mu
function PortfolioOptimisers.plot_factor_mu(pr::PortfolioOptimisers.AbstractPriorResult,
                                            nf::AbstractVector = 1:length(pr.f_mu);
                                            N::Option{<:Number} = nothing, kwargs...)
    if isnothing(pr.f_mu)
        throw(ArgumentError("prior has no factor expected returns (`f_mu` is `nothing`); pass `f_mu` directly"))
    end
    return PortfolioOptimisers.plot_factor_mu(pr.f_mu, nf; N = N, kwargs...)
end
function PortfolioOptimisers.plot_factor_mu(pr::PortfolioOptimisers.AbstractPriorResult,
                                            rd::ReturnsResult;
                                            N::Option{<:Number} = nothing, kwargs...)
    if isnothing(pr.f_mu)
        throw(ArgumentError("prior has no factor expected returns (`f_mu` is `nothing`); pass `f_mu` directly"))
    end
    nf = isnothing(rd.nf) ? (1:length(pr.f_mu)) : rd.nf
    return PortfolioOptimisers.plot_factor_mu(pr.f_mu, nf; N = N, kwargs...)
end
function PortfolioOptimisers.plot_factor_mu(res::OptimisationResult;
                                            N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_factor_mu(extract_pr(res); N = N, kwargs...)
end
function PortfolioOptimisers.plot_factor_mu(res::OptimisationResult, rd::ReturnsResult;
                                            N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_factor_mu(extract_pr(res), rd; N = N, kwargs...)
end
function PortfolioOptimisers.plot_factor_mu(pred::PredictionResult;
                                            N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_factor_mu(extract_pr(pred.res); N = N, kwargs...)
end
function PortfolioOptimisers.plot_factor_mu(pred::PredictionResult, rd::ReturnsResult;
                                            N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_factor_mu(extract_pr(pred.res), rd; N = N, kwargs...)
end
## plot_benchmark
function PortfolioOptimisers.plot_benchmark(w::ArrNum, rd::ReturnsResult,
                                            fees::Option{<:Fees} = nothing;
                                            compound::Bool = false, kwargs...)
    if isnothing(rd.B)
        throw(ArgumentError("returns data has no benchmark (`B` is `nothing`)"))
    end
    ts = isnothing(rd.ts) ? (1:size(rd.X, 1)) : rd.ts
    nb = rd.nb
    return PortfolioOptimisers.plot_benchmark(w, rd.X, rd.B, fees; ts = ts, nb = nb,
                                              compound = compound, kwargs...)
end
function PortfolioOptimisers.plot_benchmark(res::OptimisationResult, rd::ReturnsResult;
                                            compound::Bool = false, kwargs...)
    fees = extract_fees(res, nothing)
    return PortfolioOptimisers.plot_benchmark(res.w, rd, fees; compound = compound,
                                              kwargs...)
end
function PortfolioOptimisers.plot_benchmark(pred::PredictionResult; compound::Bool = false,
                                            kwargs...)
    rd = pred.rd
    if isnothing(rd.B)
        throw(ArgumentError("prediction data has no benchmark (`B` is `nothing`)"))
    end
    ret = isa(rd.X, VecVecNum) ? first(rd.X) : rd.X
    ts = isnothing(rd.ts) ? (1:length(ret)) : rd.ts
    nb = rd.nb
    B = isa(rd.B, VecVecNum) ? first(rd.B) : rd.B
    return PortfolioOptimisers.plot_benchmark(ret, B; ts = ts, nb = nb, compound = compound,
                                              kwargs...)
end
## plot_coskewness
function PortfolioOptimisers.plot_coskewness(pr::HighOrderPrior,
                                             nx::AbstractVector = 1:size(pr.sk, 1);
                                             kwargs...)
    if isnothing(pr.sk)
        throw(ArgumentError("prior has no coskewness matrix (`sk` is `nothing`)"))
    end
    return PortfolioOptimisers.plot_coskewness(pr.sk, nx; kwargs...)
end
function PortfolioOptimisers.plot_coskewness(pr::HighOrderPrior, rd::ReturnsResult;
                                             kwargs...)
    if isnothing(pr.sk)
        throw(ArgumentError("prior has no coskewness matrix (`sk` is `nothing`)"))
    end
    nx = isnothing(rd.nx) ? (1:size(pr.sk, 1)) : rd.nx
    return PortfolioOptimisers.plot_coskewness(pr.sk, nx; kwargs...)
end
function PortfolioOptimisers.plot_coskewness(res::OptimisationResult; kwargs...)
    pr = extract_pr(res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(res)))` prior is not a `HighOrderPrior`; no coskewness available"))
    end
    return PortfolioOptimisers.plot_coskewness(pr; kwargs...)
end
function PortfolioOptimisers.plot_coskewness(res::OptimisationResult, rd::ReturnsResult;
                                             kwargs...)
    pr = extract_pr(res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(res)))` prior is not a `HighOrderPrior`; no coskewness available"))
    end
    return PortfolioOptimisers.plot_coskewness(pr, rd; kwargs...)
end
function PortfolioOptimisers.plot_coskewness(pred::PredictionResult; kwargs...)
    pr = extract_pr(pred.res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(pred.res)))` prior is not a `HighOrderPrior`; no coskewness available"))
    end
    return PortfolioOptimisers.plot_coskewness(pr; kwargs...)
end
function PortfolioOptimisers.plot_coskewness(pred::PredictionResult, rd::ReturnsResult;
                                             kwargs...)
    pr = extract_pr(pred.res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(pred.res)))` prior is not a `HighOrderPrior`; no coskewness available"))
    end
    return PortfolioOptimisers.plot_coskewness(pr, rd; kwargs...)
end
## plot_cokurtosis
function PortfolioOptimisers.plot_cokurtosis(pr::HighOrderPrior,
                                             nx::AbstractVector = 1:isqrt(size(pr.kt, 1));
                                             reference::Bool = true, kwargs...)
    if isnothing(pr.kt)
        throw(ArgumentError("prior has no cokurtosis matrix (`kt` is `nothing`)"))
    end
    return PortfolioOptimisers.plot_cokurtosis(pr.kt, nx; reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_cokurtosis(pr::HighOrderPrior, rd::ReturnsResult;
                                             reference::Bool = true, kwargs...)
    if isnothing(pr.kt)
        throw(ArgumentError("prior has no cokurtosis matrix (`kt` is `nothing`)"))
    end
    nx = isnothing(rd.nx) ? (1:isqrt(size(pr.kt, 1))) : rd.nx
    return PortfolioOptimisers.plot_cokurtosis(pr.kt, nx; reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_cokurtosis(res::OptimisationResult;
                                             reference::Bool = true, kwargs...)
    pr = extract_pr(res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(res)))` prior is not a `HighOrderPrior`; no cokurtosis available"))
    end
    return PortfolioOptimisers.plot_cokurtosis(pr; reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_cokurtosis(res::OptimisationResult, rd::ReturnsResult;
                                             reference::Bool = true, kwargs...)
    pr = extract_pr(res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(res)))` prior is not a `HighOrderPrior`; no cokurtosis available"))
    end
    return PortfolioOptimisers.plot_cokurtosis(pr, rd; reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_cokurtosis(pred::PredictionResult; reference::Bool = true,
                                             kwargs...)
    pr = extract_pr(pred.res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(pred.res)))` prior is not a `HighOrderPrior`; no cokurtosis available"))
    end
    return PortfolioOptimisers.plot_cokurtosis(pr; reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_cokurtosis(pred::PredictionResult, rd::ReturnsResult;
                                             reference::Bool = true, kwargs...)
    pr = extract_pr(pred.res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(pred.res)))` prior is not a `HighOrderPrior`; no cokurtosis available"))
    end
    return PortfolioOptimisers.plot_cokurtosis(pr, rd; reference = reference, kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Risk contribution
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_risk_contribution(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                    w::VecNum, X::MatNum_Pr,
                                                    fees::Option{<:Fees} = nothing;
                                                    nx::AbstractVector = 1:length(w),
                                                    delta::Number = 1e-6,
                                                    marginal::Bool = false,
                                                    percentage::Bool = true,
                                                    erc::Bool = true,
                                                    N::Option{<:Number} = nothing,
                                                    kwargs...)
    if !(delta > zero(delta))
        throw(DomainError(delta, "delta must be > 0"))
    end
    if !isnothing(N) && !(N > zero(N))
        throw(DomainError(N, "N must be > 0"))
    end
    rc = risk_contribution(r, w, X, fees; delta = delta, marginal = marginal)
    if percentage
        rc = rc / sum(rc)
    end
    plt = PortfolioOptimisers.plot_composition(rc, nx; N = N, ylabel = "Contribution",
                                               title = "Risk Contribution", kwargs...)
    if erc
        plt = hline!(plt, [mean(rc)])
    end
    return plt
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
                                                           delta::Number = 1e-6,
                                                           N::Option{<:Number} = nothing,
                                                           percentage::Bool = true,
                                                           erc::Bool = true, kwargs...)
    if !(delta > zero(delta))
        throw(DomainError(delta, "delta must be > 0"))
    end
    if !isnothing(N) && !(N > zero(N))
        throw(DomainError(N, "N must be > 0"))
    end
    rc = factor_risk_contribution(r, w, X, fees; re = re, rd = rd, delta = delta)
    factor_names = if !isnothing(nf) && length(rc) <= length(nf) + 1
        [nf; "Constant"]
    elseif !isnothing(rd.nf) && length(rc) <= length(rd.nf) + 1
        [rd.nf; "Constant"]
    else
        [string.(1:(length(rc) - 1)); "Constant"]
    end
    if percentage
        rc = rc / sum(rc)
    end
    plt = PortfolioOptimisers.plot_composition(rc, factor_names; N = N,
                                               title = "Factor Risk Contribution",
                                               xlabel = "Factor",
                                               ylabel = "Risk Contribution", kwargs...)
    if erc
        hline!(plt, [mean(rc)])
    end
    return plt
end
## ────────────────────────────────────────────────────────────────────────────
## Drawdowns
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_drawdowns(w::ArrNum, X::MatNum,
                                            fees::Option{<:Fees} = nothing;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            ts::AbstractVector = 1:size(X, 1),
                                            compound::Bool = false, alpha::Number = 0.05,
                                            kappa::Number = 0.3, rw = nothing, kwargs...)
    return PortfolioOptimisers.plot_drawdowns(calc_net_returns(w, X, fees); slv = slv,
                                              ts = ts, compound = compound, alpha = alpha,
                                              kappa = kappa, rw = rw, kwargs...)
end
function PortfolioOptimisers.plot_drawdowns(ret::VecNum;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            ts::AbstractVector, compound::Bool = false,
                                            alpha::Number = 0.05, kappa::Number = 0.3,
                                            rw = nothing, kwargs...)
    if !(zero(alpha) < alpha < one(alpha))
        throw(DomainError(alpha, "alpha must satisfy 0 < alpha < 1"))
    end
    if !(zero(kappa) < kappa < one(kappa))
        throw(DomainError(kappa, "kappa must satisfy 0 < kappa < 1"))
    end
    cret = cumulative_returns(ret, compound)
    dd = drawdowns(cret, compound; cX = true) .* 100

    base_risks = 100 * if !compound
                       [-AverageDrawdown(; w = rw)(ret), -UlcerIndex()(ret),
                        -DrawdownatRisk(; alpha = alpha)(ret),
                        -ConditionalDrawdownatRisk(; alpha = alpha)(ret), -MaximumDrawdown()(ret)]
                       else
                       [-RelativeAverageDrawdown(; w = rw)(ret), -RelativeUlcerIndex()(ret),
                        -RelativeDrawdownatRisk(; alpha = alpha)(ret),
                        -RelativeConditionalDrawdownatRisk(; alpha = alpha)(ret),
                        -RelativeMaximumDrawdown()(ret)]
                       end

    conf = round((1 - alpha) * 100; digits = 2)
    base_labels = ["Average Drawdown: $(round(base_risks[1]; digits=2))%",
                   "Ulcer Index: $(round(base_risks[2]; digits=2))%",
                   "$(conf)% DaR: $(round(base_risks[3]; digits=2))%",
                   "$(conf)% CDaR: $(round(base_risks[4]; digits=2))%",
                   "Maximum Drawdown: $(round(base_risks[5]; digits=2))%"]

    risks = copy(base_risks)
    labels = copy(base_labels)
    if !isnothing(slv)
        if !compound
            push!(risks, 100 * -EntropicDrawdownatRisk(; slv = slv, alpha = alpha)(ret),
                  100 *
                  -RelativisticDrawdownatRisk(; slv = slv, alpha = alpha, kappa = kappa)(ret))
        else
            push!(risks,
                  100 * -RelativeEntropicDrawdownatRisk(; slv = slv, alpha = alpha)(ret),
                  100 *
                  -RelativeRelativisticDrawdownatRisk(; slv = slv, alpha = alpha,
                                                      kappa = kappa)(ret))
        end
        push!(labels, "$(conf)% EDaR: $(round(risks[6]; digits=2))%",
              "$(conf)% RLDaR ($(round(kappa; digits=2))): $(round(risks[7]; digits=2))%")
    end

    theme_cols = palette(:Dark2_5, length(labels) + 1)
    dd_label = "$(compound ? "Compounded" : "Uncompounded") Drawdown"
    f_dd = plot(ts, dd; label = dd_label,
                ylabel = "$(compound ? "Compounded" : "Uncompounded")\nDrawdown %",
                xlabel = "Date", linewidth = 2, yguidefontsize = 10, color = theme_cols[1],
                ylim = extrema(dd) .* [1.2, 1.01])
    for (i, (risk, lbl)) in enumerate(zip(risks, labels))
        hline!(f_dd, [risk]; label = lbl, color = theme_cols[mod1(i + 1, end)],
               linewidth = 2, legend = :bottomleft)
    end
    f_ret = plot(ts, cret;
                 ylabel = "$(compound ? "Compounded" : "Uncompounded")\nCumulative Returns",
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
                                            alpha::Number = 0.05, kappa::Number = 0.3,
                                            rw = nothing, points::Integer = 0,
                                            reference::Bool = true, kwargs...)
    return PortfolioOptimisers.plot_histogram(calc_net_returns(w, X, fees); slv = slv,
                                              alpha = alpha, kappa = kappa, rw = rw,
                                              points = points, reference = reference,
                                              kwargs...)
end
function PortfolioOptimisers.plot_histogram(ret::VecNum;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            alpha::Number = 0.05, kappa::Number = 0.3,
                                            rw = nothing, points::Integer = 0,
                                            reference::Bool = true, kwargs...)
    if !(zero(alpha) < alpha < one(alpha))
        throw(DomainError(alpha, "alpha must satisfy 0 < alpha < 1"))
    end
    if !(zero(kappa) < kappa < one(kappa))
        throw(DomainError(kappa, "kappa must satisfy 0 < kappa < 1"))
    end
    if !(points >= 0)
        throw(DomainError(points, "points must be >= 0"))
    end
    T = length(ret)
    npts = points == 0 ? ceil(Int, 4 * sqrt(T)) : points
    mu_r = mean(ret)
    sigma_r = std(ret)
    mir, mar = extrema(ret)
    x_range = range(mir, mar; length = npts)
    mad = LowOrderMoment(; w = rw, alg = MeanAbsoluteDeviation())(ret)
    gmd = OrderedWeightsArray()(ret)

    base_risks = [mu_r, mu_r - sigma_r, mu_r - mad, mu_r - gmd,
                  -ValueatRisk(; w = rw, alpha = alpha)(ret),
                  -ConditionalValueatRisk(; w = rw, alpha = alpha)(ret),
                  -OrderedWeightsArray(; w = owa_tg(T))(ret)]

    conf = round((1 - alpha) * 100; digits = 2)
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
        push!(risks, -EntropicValueatRisk(; w = rw, slv = slv, alpha = alpha)(ret),
              -RelativisticValueatRisk(; w = rw, slv = slv, alpha = alpha, kappa = kappa)(ret))
        push!(risk_labels, "$(conf)% EVaR: $(round(100*risks[8]; digits=2))%",
              "$(conf)% RLVaR ($(round(kappa; digits=2))): $(round(100*risks[9]; digits=2))%")
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
    if reference
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
## Correlation / covariance heatmap
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_correlation(X::MatNum, nx::AbstractVector = 1:size(X, 1);
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
## Expected returns bar chart
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_mu(mu::VecNum, nx::AbstractVector = 1:length(mu);
                                     N::Option{<:Number} = nothing, kwargs...)
    M = length(mu)
    N, idx = relevant_assets(mu, M, N)
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
                                        variance::Bool = false,
                                        N::Option{<:Number} = nothing, kwargs...)
    vals = variance ? LinearAlgebra.diag(sigma) : sqrt.(LinearAlgebra.diag(sigma))
    ylabel_str = variance ? "Variance (σ²)" : "Volatility (σ)"
    M = length(vals)
    idx = sortperm(vals; rev = true)
    N_show = isnothing(N) ? M : clamp(ceil(Int, N), 1, M)
    top_idx = idx[1:N_show]
    sort!(top_idx)
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
                                               kwargs...)
    return PortfolioOptimisers.plot_correlation(f_sigma, nf; title = "Factor Covariance",
                                                colorbar_title = "ρ_f", kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Eigenvalue spectrum
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_eigenspectrum(sigma::MatNum;
                                                N_obs::Option{<:Integer} = nothing,
                                                reference::Bool = true, kwargs...)
    ev = eigvals(Symmetric(sigma))
    sorted_ev = sort(real.(ev); rev = true)
    N = length(sorted_ev)

    f = bar(1:N, sorted_ev; title = "Eigenspectrum", xlabel = "Component",
            ylabel = "Eigenvalue", legend = reference && !isnothing(N_obs), kwargs...)

    if reference && !isnothing(N_obs)
        q = N / N_obs
        s2 = tr(sigma) / N
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
## Weight stability across folds
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_weight_stability(mpred::MultiPeriodPredictionResult;
                                                   N::Option{<:Number} = nothing, kwargs...)
    folds = mpred.pred
    w_mat = hcat(getproperty.(getproperty.(folds, :res), :w)...)
    nx = let rd1 = folds[1].rd
        isnothing(rd1.nx) ? string.(1:size(w_mat, 1)) : string.(rd1.nx)
    end
    M, K = size(w_mat)
    mean_abs = vec(mean(abs.(w_mat); dims = 2))
    N, idx = relevant_assets(mean_abs, M, N)
    top_idx = sort(view(idx, 1:N))
    labels = nx[top_idx]
    data = w_mat[top_idx, :]
    return boxplot(data'; xticks = (1:N, labels), title = "Weight Stability",
                   ylabel = "Weight", legend = false, xrotation = 60, kwargs...)
end
function PortfolioOptimisers.plot_weight_stability(ppred::PopulationPredictionResult;
                                                   N::Option{<:Number} = nothing, kwargs...)
    members = ppred.pred
    w_mat = hcat(map(members) do m
                     if isa(m, PredictionResult)
                         m.res.w
                     else
                         vec(mean(hcat(getproperty.(getproperty.(m.pred, :res), :w)...);
                                  dims = 2))
                     end
                 end...)
    nx = let rd1 = isa(members[1], PredictionResult) ? members[1].rd : members[1].pred[1].rd
        isnothing(rd1.nx) ? string.(1:size(w_mat, 1)) : string.(rd1.nx)
    end
    M, K = size(w_mat)
    mean_abs = vec(mean(abs.(w_mat); dims = 2))
    N, idx = relevant_assets(mean_abs, M, N)
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
                                            compound::Bool = false, alpha::Number = 0.05,
                                            kappa::Number = 0.3, rw = nothing, kwargs...)
    mrd = pred.mrd
    ret = isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
    ts = isnothing(mrd.ts) ? (1:length(ret)) : mrd.ts
    return PortfolioOptimisers.plot_drawdowns(ret; slv = slv, ts = ts, compound = compound,
                                              alpha = alpha, kappa = kappa, rw = rw,
                                              kwargs...)
end
function PortfolioOptimisers.plot_histogram(pred::MultiPeriodPredictionResult;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            alpha::Number = 0.05, kappa::Number = 0.3,
                                            rw = nothing, points::Integer = 0,
                                            reference::Bool = true, kwargs...)
    mrd = pred.mrd
    ret = isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
    return PortfolioOptimisers.plot_histogram(ret; slv = slv, alpha = alpha, kappa = kappa,
                                              rw = rw, points = points,
                                              reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                  pred::MultiPeriodPredictionResult;
                                                  rolling::Integer = 0, kwargs...)
    mrd = pred.mrd
    ret = isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
    ts = isnothing(mrd.ts) ? (1:length(ret)) : mrd.ts
    return PortfolioOptimisers.plot_rolling_measure(r, ret; ts = ts, rolling = rolling,
                                                    kwargs...)
end
function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                  ppred::PopulationPredictionResult;
                                                  rolling::Integer = 0, kwargs...)
    members = ppred.pred
    rets = map(members) do m
        mrd = isa(m, PredictionResult) ? m.rd : m.mrd
        return isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
    end
    avg_ret = vec(mean(hcat(rets...); dims = 2))
    first_mrd = isa(members[1], PredictionResult) ? members[1].rd : members[1].mrd
    ts = isnothing(first_mrd.ts) ? (1:length(avg_ret)) : first_mrd.ts
    return PortfolioOptimisers.plot_rolling_measure(r, avg_ret; ts = ts, rolling = rolling,
                                                    kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## PopulationPredictionResult: drawdowns, histogram, rolling measure
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_drawdowns(ppred::PopulationPredictionResult;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            compound::Bool = false, alpha::Number = 0.05,
                                            kappa::Number = 0.3, rw = nothing, kwargs...)
    members = ppred.pred
    rets = map(members) do m
        mrd = isa(m, PredictionResult) ? m.rd : m.mrd
        return isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
    end
    avg_ret = vec(mean(hcat(rets...); dims = 2))
    first_mrd = isa(members[1], PredictionResult) ? members[1].rd : members[1].mrd
    ts = isnothing(first_mrd.ts) ? (1:length(avg_ret)) : first_mrd.ts
    return PortfolioOptimisers.plot_drawdowns(avg_ret; slv = slv, ts = ts,
                                              compound = compound, alpha = alpha,
                                              kappa = kappa, rw = rw, kwargs...)
end
function PortfolioOptimisers.plot_histogram(ppred::PopulationPredictionResult;
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            alpha::Number = 0.05, kappa::Number = 0.3,
                                            rw = nothing, points::Integer = 0,
                                            reference::Bool = true, kwargs...)
    members = ppred.pred
    rets = map(members) do m
        mrd = isa(m, PredictionResult) ? m.rd : m.mrd
        return isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
    end
    avg_ret = vec(mean(hcat(rets...); dims = 2))
    return PortfolioOptimisers.plot_histogram(avg_ret; slv = slv, alpha = alpha,
                                              kappa = kappa, rw = rw, points = points,
                                              reference = reference, kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Factor expected returns bar chart
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_factor_mu(f_mu::VecNum,
                                            nf::AbstractVector = 1:length(f_mu);
                                            N::Option{<:Number} = nothing, kwargs...)
    M = length(f_mu)
    N, idx = relevant_assets(f_mu, M, N)
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
                                            compound::Bool = false, kwargs...)
    return PortfolioOptimisers.plot_benchmark(calc_net_returns(w, X, fees), B; ts = ts,
                                              nb = nb, compound = compound, kwargs...)
end
function PortfolioOptimisers.plot_benchmark(net_ret::VecNum, B::VecNum_VecVecNum;
                                            ts::AbstractVector,
                                            nb::Option{<:AbstractVector} = nothing,
                                            compound::Bool = false, kwargs...)
    ret = cumulative_returns(net_ret, compound)
    Bmat = isa(B, VecVecNum) ? hcat(B...) : reshape(B, :, 1)
    Nb = size(Bmat, 2)
    bench_labels = isnothing(nb) ? ["Benchmark $i" for i in 1:Nb] : string.(nb[1:Nb])
    f = plot(ts, ret; label = "Portfolio", linewidth = 2, title = "Portfolio vs Benchmark",
             xlabel = "Date",
             ylabel = "$(compound ? "Compound" : "Simple") Cumulative Returns",
             legend = :outerright, kwargs...)
    for i in 1:Nb
        b_ret = cumulative_returns(vec(view(Bmat, :, i)), compound)
        plot!(f, ts, b_ret; label = bench_labels[i], linewidth = 1.5, linestyle = :dash)
    end
    return f
end
function PortfolioOptimisers.plot_benchmark(pred::MultiPeriodPredictionResult;
                                            compound::Bool = false, kwargs...)
    mrd = pred.mrd
    if isnothing(mrd.B)
        throw(ArgumentError("multi-period prediction data has no benchmark (`B` is `nothing`)"))
    end
    ret = isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
    ts = isnothing(mrd.ts) ? (1:length(ret)) : mrd.ts
    return PortfolioOptimisers.plot_benchmark(ret, mrd.B; ts = ts, nb = mrd.nb,
                                              compound = compound, kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Coskewness heatmap
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_coskewness(sk::MatNum, nx::AbstractVector = 1:size(sk, 1);
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
                                             heatmap::Bool = false, reference::Bool = true,
                                             kwargs...)
    if heatmap
        clim_val = maximum(abs, kt)
        return StatsPlots.heatmap(kt; color = cgrad(:RdBu; rev = true),
                                  clim = (-clim_val, clim_val), title = "Cokurtosis Matrix",
                                  colorbar_title = "K̃", yflip = true, kwargs...)
    else
        ev = sort(real.(eigvals(Symmetric(kt))); rev = true)
        N2 = length(ev)
        f = bar(1:N2, ev; title = "Cokurtosis Eigenspectrum", xlabel = "Component",
                ylabel = "Eigenvalue", legend = reference, kwargs...)
        if reference
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

function PortfolioOptimisers.plot_portfolio_dashboard(res::OptimisationResult, rd::Pr_RR;
                                                      slv::Option{<:Slv_VecSlv} = nothing,
                                                      ts = 1:size(rd.X, 1),
                                                      nx = 1:size(rd.X, 2),
                                                      r::PortfolioOptimisers.AbstractBaseRiskMeasure = Variance(),
                                                      compound::Bool = false,
                                                      N::Option{<:Number} = nothing,
                                                      delta::Number = 1e-6,
                                                      marginal::Bool = false,
                                                      percentage::Bool = true,
                                                      alpha::Number = 0.05,
                                                      kappa::Number = 0.3, rw = nothing,
                                                      kwargs...)
    fees = extract_fees(res, nothing)
    w = res.w
    if isa(rd, ReturnsResult)
        nx = rd.nx
        ts = rd.ts
    end
    p1 = PortfolioOptimisers.plot_composition(w, nx; N = N)
    p2 = PortfolioOptimisers.plot_ptf_cumulative_returns(w, rd.X, fees; ts = ts,
                                                         compound = compound)
    p3 = PortfolioOptimisers.plot_risk_contribution(r, w, rd.X, fees; nx = nx, N = N,
                                                    delta = delta, marginal = marginal,
                                                    percentage = percentage)
    p4 = PortfolioOptimisers.plot_drawdowns(w, rd.X, fees; slv = slv, ts = ts,
                                            compound = compound, alpha = alpha,
                                            kappa = kappa, rw = rw)
    return plot(p1, p2, p3, p4; layout = (2, 2), size = (1200, 800), kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Cross-validation dashboard (multi-panel composite)
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_cv_dashboard(mpred::MultiPeriodPredictionResult;
                                               N::Option{<:Number} = nothing,
                                               compound::Bool = false, kwargs...)
    p1 = PortfolioOptimisers.plot_composition(mpred; N = N)
    p2 = PortfolioOptimisers.plot_ptf_cumulative_returns(mpred; compound = compound)
    p3 = PortfolioOptimisers.plot_turnover(mpred)
    p4 = PortfolioOptimisers.plot_weight_stability(mpred; N = N)
    return plot(p1, p2, p3, p4; layout = (2, 2), size = (1200, 800), kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Composite prior dashboard
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_prior(pr::PortfolioOptimisers.AbstractPriorResult,
                                        nx::AbstractVector = 1:length(pr.mu);
                                        N::Option{<:Number} = nothing, kwargs...)
    p_mu = PortfolioOptimisers.plot_mu(pr.mu, nx; N = N, title = "Expected Returns",
                                       ylabel = "μ")
    p_sigma = PortfolioOptimisers.plot_sigma(pr.sigma, nx; N = N,
                                             title = "Asset Volatility", ylabel = "σ")
    p_corr = PortfolioOptimisers.plot_correlation(pr.sigma, nx;
                                                  title = "Correlation Matrix")
    return plot(p_mu, p_sigma, p_corr; layout = (1, 3), size = (1800, 500), kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Efficient frontier
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_efficient_frontier(res_vec::AbstractVector{<:OptimisationResult},
                                                     pr::Pr_RR;
                                                     x::PortfolioOptimisers.AbstractBaseRiskMeasure = Variance(),
                                                     y::PortfolioOptimisers.AbstractBaseRiskMeasure = ExpectedReturn(),
                                                     c::PortfolioOptimisers.AbstractBaseRiskMeasure = ExpectedReturnRiskRatio(;
                                                                                                                              rk = x,
                                                                                                                              rt = ArithmeticReturn(),
                                                                                                                              rf = 0),
                                                     slv::Option{<:Slv_VecSlv} = nothing,
                                                     fees::Option{<:Fees} = nothing,
                                                     min_risk::Bool = true,
                                                     max_score::Bool = true,
                                                     factory::Bool = true, kwargs...)
    if factory
        x = PortfolioOptimisers.factory(x, pr, slv)
        y = PortfolioOptimisers.factory(y, pr, slv)
        c = PortfolioOptimisers.factory(c, pr, slv)
    end
    w = getproperty.(res_vec, :w)
    xr = expected_risk(x, w, pr, fees)
    yr = expected_risk(y, w, pr, fees)
    cr = expected_risk(c, w, pr, fees)
    order = sortperm(xr)
    xr_s = xr[order]
    yr_s = yr[order]
    cr_s = cr[order]
    xname = string(nameof(typeof(x)))
    yname = string(nameof(typeof(y)))
    cname = string(nameof(typeof(c)))
    plt = plot(xr_s, yr_s; zcolor = cr_s, line_z = cr_s, title = "Efficient Frontier",
               xlabel = xname, ylabel = yname, colorbar_title = cname, label = nothing,
               linewidth = 2, markershape = :circle, markersize = 4, kwargs...)
    if min_risk
        i = argmin(xr_s)
        scatter!(plt, [xr_s[i]], [yr_s[i]]; label = "Min Risk", markershape = :star5,
                 markersize = 12, color = :blue, legend = true)
    end
    if max_score
        i = argmax(cr_s)
        scatter!(plt, [xr_s[i]], [yr_s[i]]; label = "Max $(cname)", markershape = :star5,
                 markersize = 12, color = :red, legend = true)
    end
    return plt
end
function PortfolioOptimisers.plot_efficient_frontier(res_vec::AbstractVector{<:OptimisationResult},
                                                     rd::ReturnsResult; kwargs...)
    return PortfolioOptimisers.plot_efficient_frontier(res_vec, extract_pr(first(res_vec));
                                                       kwargs...)
end
function PortfolioOptimisers.plot_efficient_frontier(w::VecVecNum, pr::Pr_RR;
                                                     x::PortfolioOptimisers.AbstractBaseRiskMeasure = Variance(),
                                                     y::PortfolioOptimisers.AbstractBaseRiskMeasure = ExpectedReturn(),
                                                     c::PortfolioOptimisers.AbstractBaseRiskMeasure = ExpectedReturnRiskRatio(;
                                                                                                                              rk = x,
                                                                                                                              rt = ArithmeticReturn(),
                                                                                                                              rf = 0),
                                                     slv::Option{<:Slv_VecSlv} = nothing,
                                                     fees::Option{<:Fees} = nothing,
                                                     min_risk::Bool = true,
                                                     max_score::Bool = true,
                                                     factory::Bool = true, kwargs...)
    if factory
        x = PortfolioOptimisers.factory(x, pr, slv)
        y = PortfolioOptimisers.factory(y, pr, slv)
        c = PortfolioOptimisers.factory(c, pr, slv)
    end
    xr = expected_risk(x, w, pr, fees)
    yr = expected_risk(y, w, pr, fees)
    cr = expected_risk(c, w, pr, fees)
    order = sortperm(xr)
    xr_s = xr[order]
    yr_s = yr[order]
    cr_s = cr[order]
    xname = string(nameof(typeof(x)))
    yname = string(nameof(typeof(y)))
    cname = string(nameof(typeof(c)))
    plt = plot(xr_s, yr_s; zcolor = cr_s, line_z = cr_s, title = "Efficient Frontier",
               xlabel = xname, ylabel = yname, colorbar_title = cname, label = nothing,
               linewidth = 2, markershape = :circle, markersize = 4, kwargs...)
    if min_risk
        i = argmin(xr_s)
        scatter!(plt, [xr_s[i]], [yr_s[i]]; label = "Min Risk", markershape = :star5,
                 markersize = 12, color = :blue, legend = true)
    end
    if max_score
        i = argmax(cr_s)
        scatter!(plt, [xr_s[i]], [yr_s[i]]; label = "Max $(cname)", markershape = :star5,
                 markersize = 12, color = :red, legend = true)
    end
    return plt
end
function PortfolioOptimisers.plot_efficient_frontier(res::OptimisationResult, pr::Pr_RR;
                                                     fees::Option{<:Fees} = nothing,
                                                     kwargs...)
    fees = extract_fees(res, fees)
    w = res.w
    if isa(w, VecVecNum)
        return PortfolioOptimisers.plot_efficient_frontier(w, pr; fees = fees, kwargs...)
    else
        return PortfolioOptimisers.plot_efficient_frontier([res], pr; fees = fees,
                                                           kwargs...)
    end
end
function PortfolioOptimisers.plot_efficient_frontier(res::OptimisationResult,
                                                     rd::ReturnsResult; kwargs...)
    return PortfolioOptimisers.plot_efficient_frontier(res, extract_pr(res); kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Performance summary
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_performance_summary(w::ArrNum, X::MatNum,
                                                      fees::Option{<:Fees} = nothing;
                                                      periods_per_year::Number = 252,
                                                      alpha::Number = 0.05,
                                                      compound::Bool = false, kwargs...)
    return PortfolioOptimisers.plot_performance_summary(calc_net_returns(w, X, fees);
                                                        periods_per_year = periods_per_year,
                                                        alpha = alpha, compound = compound,
                                                        kwargs...)
end
function PortfolioOptimisers.plot_performance_summary(ret::VecNum;
                                                      periods_per_year::Number = 252,
                                                      alpha::Number = 0.05,
                                                      compound::Bool = false, kwargs...)
    if !(zero(alpha) < alpha < one(alpha))
        throw(DomainError(alpha, "alpha must satisfy 0 < alpha < 1"))
    end
    ann = periods_per_year
    ann_ret = mean(ret) * ann
    ann_vol = std(ret) * sqrt(ann)
    sharpe = ann_vol > 0 ? ann_ret / ann_vol : NaN
    neg_ret = min.(ret, zero(eltype(ret)))
    ddev = sqrt(mean(neg_ret .^ 2) * ann)
    sortino = ddev > 0 ? ann_ret / ddev : NaN
    cret = cumulative_returns(ret, compound)
    dd_series = drawdowns(cret, compound; cX = true)
    max_dd = minimum(dd_series)
    calmar = max_dd < 0 ? ann_ret / abs(max_dd) : NaN
    cvar_val = -ConditionalValueatRisk(; alpha = alpha)(ret)
    conf = round((1 - alpha) * 100; digits = 1)
    vals = [ann_ret * 100, ann_vol * 100, sharpe, sortino, calmar, max_dd * 100,
            cvar_val * 100]
    labels = ["Ann. Return %", "Ann. Vol %", "Sharpe", "Sortino", "Calmar", "Max DD %",
              "$(conf)% CVaR %"]
    colours = [v >= 0 ? :steelblue : :firebrick for v in vals]
    return bar(vals; xticks = (1:length(labels), labels), xrotation = 30,
               title = "Performance Summary", ylabel = "Value", legend = false,
               color = colours, kwargs...)
end
function PortfolioOptimisers.plot_performance_summary(w::ArrNum, rd::ReturnsResult,
                                                      fees::Option{<:Fees} = nothing;
                                                      alpha::Number = 0.05,
                                                      compound::Bool = false, kwargs...)
    return PortfolioOptimisers.plot_performance_summary(w, rd.X, fees; alpha = alpha,
                                                        compound = compound, kwargs...)
end
function PortfolioOptimisers.plot_performance_summary(res::OptimisationResult,
                                                      rd::ReturnsResult;
                                                      alpha::Number = 0.05,
                                                      compound::Bool = false, kwargs...)
    fees = extract_fees(res, nothing)
    return PortfolioOptimisers.plot_performance_summary(res.w, rd.X, fees; alpha = alpha,
                                                        compound = compound, kwargs...)
end
function PortfolioOptimisers.plot_performance_summary(pred::PredictionResult;
                                                      alpha::Number = 0.05,
                                                      compound::Bool = false, kwargs...)
    rd = pred.rd
    ret = isa(rd.X, VecVecNum) ? first(rd.X) : rd.X
    return PortfolioOptimisers.plot_performance_summary(ret; alpha = alpha,
                                                        compound = compound, kwargs...)
end
function PortfolioOptimisers.plot_performance_summary(pred::MultiPeriodPredictionResult;
                                                      alpha::Number = 0.05,
                                                      compound::Bool = false, kwargs...)
    mrd = pred.mrd
    ret = isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
    return PortfolioOptimisers.plot_performance_summary(ret; alpha = alpha,
                                                        compound = compound, kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Rolling maximum drawdown
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_rolling_drawdowns(w::ArrNum, X::MatNum,
                                                    fees::Option{<:Fees} = nothing;
                                                    ts::AbstractVector = 1:size(X, 1),
                                                    rolling::Integer = 0,
                                                    compound::Bool = false, kwargs...)
    return PortfolioOptimisers.plot_rolling_drawdowns(calc_net_returns(w, X, fees); ts = ts,
                                                      rolling = rolling,
                                                      compound = compound, kwargs...)
end
function PortfolioOptimisers.plot_rolling_drawdowns(ret::VecNum; ts::AbstractVector,
                                                    rolling::Integer = 0,
                                                    compound::Bool = false, kwargs...)
    if !(rolling >= 0)
        throw(DomainError(rolling, "rolling must be >= 0"))
    end
    T = length(ret)
    window = rolling == 0 ? ceil(Int, sqrt(T)) : rolling
    cret = cumulative_returns(ret, compound)
    rolling_mdd = [minimum(drawdowns(view(cret, (t - window + 1):t), compound; cX = true)) *
                   100 for t in window:T]
    ts_rolling = ts[window:end]
    label_str = "$(compound ? "Compound" : "Simple") Max Drawdown (window=$window)"
    return plot(ts_rolling, rolling_mdd; title = "Rolling Maximum Drawdown",
                ylabel = "Max Drawdown %", xlabel = "Date", legend = false, linewidth = 2,
                label = label_str, kwargs...)
end
function PortfolioOptimisers.plot_rolling_drawdowns(w::ArrNum, rd::ReturnsResult,
                                                    fees::Option{<:Fees} = nothing;
                                                    rolling::Integer = 0,
                                                    compound::Bool = false, kwargs...)
    ts = isnothing(rd.ts) ? (1:size(rd.X, 1)) : rd.ts
    return PortfolioOptimisers.plot_rolling_drawdowns(w, rd.X, fees; ts = ts,
                                                      rolling = rolling,
                                                      compound = compound, kwargs...)
end
function PortfolioOptimisers.plot_rolling_drawdowns(res::OptimisationResult,
                                                    rd::ReturnsResult; rolling::Integer = 0,
                                                    compound::Bool = false, kwargs...)
    fees = extract_fees(res, nothing)
    return PortfolioOptimisers.plot_rolling_drawdowns(res.w, rd, fees; rolling = rolling,
                                                      compound = compound, kwargs...)
end
function PortfolioOptimisers.plot_rolling_drawdowns(pred::PredictionResult;
                                                    rolling::Integer = 0,
                                                    compound::Bool = false, kwargs...)
    rd = pred.rd
    ret = isa(rd.X, VecVecNum) ? first(rd.X) : rd.X
    ts = isnothing(rd.ts) ? (1:length(ret)) : rd.ts
    return PortfolioOptimisers.plot_rolling_drawdowns(ret; ts = ts, rolling = rolling,
                                                      compound = compound, kwargs...)
end
function PortfolioOptimisers.plot_rolling_drawdowns(pred::MultiPeriodPredictionResult;
                                                    rolling::Integer = 0,
                                                    compound::Bool = false, kwargs...)
    mrd = pred.mrd
    ret = isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
    ts = isnothing(mrd.ts) ? (1:length(ret)) : mrd.ts
    return PortfolioOptimisers.plot_rolling_drawdowns(ret; ts = ts, rolling = rolling,
                                                      compound = compound, kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Rolling measure base
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                  w::VecNum, X::MatNum,
                                                  fees::Option{<:Fees} = nothing;
                                                  ts::AbstractVector = 1:size(X, 1),
                                                  rolling::Integer = 0, kwargs...)
    return PortfolioOptimisers.plot_rolling_measure(r, calc_net_returns(w, X, fees);
                                                    ts = ts, rolling = rolling, kwargs...)
end
function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                  ret::VecNum; ts::AbstractVector,
                                                  rolling::Integer = 0, kwargs...)
    if !(rolling >= 0)
        throw(DomainError(rolling, "rolling must be >= 0"))
    end
    T = length(ret)
    window = rolling == 0 ? ceil(Int, sqrt(T)) : rolling
    rolling_vals = [r(view(ret, (t - window + 1):t)) for t in window:T]
    ts_rolling = ts[window:end]
    rname = string(nameof(typeof(r)))
    return plot(ts_rolling, rolling_vals; title = "Rolling $rname (window=$window)",
                ylabel = rname, xlabel = "Date", legend = false, linewidth = 2, kwargs...)
end

end
