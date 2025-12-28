module PortfolioOptimisersPlotsExt

using PortfolioOptimisers, GraphRecipes, StatsPlots, LinearAlgebra, Statistics, StatsBase,
      Clustering, Distributions

import PortfolioOptimisers: ArrNum, VecNum, MatNum, Option, VecNum_VecVecNum, Slv_VecSlv,
                            MatNum_Pr, PrE_Pr, ClE_Cl, VecVecNum

function PortfolioOptimisers.plot_ptf_cumulative_returns(w::ArrNum, X::MatNum,
                                                         fees::Option{<:Fees} = nothing;
                                                         ts::AbstractVector = 1:size(X, 1),
                                                         compound::Bool = false,
                                                         kwargs::NamedTuple = (;
                                                                               title = "Portfolio",
                                                                               xlabel = "Date",
                                                                               ylabel = "$(compound ? "Compound" : "Simple") Portfolio Cumulative Returns",
                                                                               legend = false),
                                                         ekwargs...)
    ret = cumulative_returns(calc_net_returns(w, X, fees), compound)
    return plot(ts, ret; kwargs..., ekwargs...)
end
function compute_relevant_assets(w::VecNum, M::Number, N::Number)
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
function PortfolioOptimisers.plot_asset_cumulative_returns(w::VecNum, X::MatNum,
                                                           fees::Option{<:Fees} = nothing;
                                                           ts::AbstractVector = 1:size(X, 1),
                                                           nx::AbstractVector = 1:size(X, 2),
                                                           N::Option{<:Number} = nothing,
                                                           compound::Bool = false,
                                                           f_kwargs::NamedTuple = (;
                                                                                   xlabel = "Date",
                                                                                   ylabel = "$(compound ? "Compound" : "Simple") Asset Cumulative Returns"),
                                                           asset_kwargs::NamedTuple = (;),
                                                           summary_kwargs::NamedTuple = (;
                                                                                         label = "Others"),
                                                           legend_kwargs::NamedTuple = (;
                                                                                        position = :auto),
                                                           ekwargs...)
    ret = cumulative_returns(calc_net_asset_returns(w, X, fees), compound)
    M = size(X, 2)
    N, idx = compute_relevant_assets(w, M, isnothing(N) ? inv(LinearAlgebra.dot(w, w)) : N)
    ret = view(ret, :, idx)
    nx = view(nx, idx)
    f = plot(; f_kwargs...)
    for i in 1:N
        if !haskey(asset_kwargs, :label)
            asset_kwargs_i = (; label = nx[i], asset_kwargs...)
        end
        plot!(f, ts, view(ret, :, i); asset_kwargs_i...)
    end
    if M > N
        idx = view(idx, (N + 1):M)
        ret = cumulative_returns(calc_net_returns(view(w, idx), view(X, :, idx),
                                                  PortfolioOptimisers.fees_view(fees, idx)),
                                 compound)
        plot!(f, ts, ret; summary_kwargs...)
    end
    plot!(f; legend_kwargs..., ekwargs...)
    return f
end
function PortfolioOptimisers.plot_composition(w::VecNum, nx::AbstractVector = 1:length(w);
                                              N::Option{<:Number} = nothing,
                                              kwargs::NamedTuple = (title = "Portfolio Composition",
                                                                    xlabel = "Asset",
                                                                    ylabel = "Weight",
                                                                    xrotation = 90,
                                                                    legend = false),
                                              ekwargs...)
    M = length(w)
    N, idx = compute_relevant_assets(w, M, isnothing(N) ? inv(LinearAlgebra.dot(w, w)) : N)
    return if M > N
        sort!(view(idx, 1:N))
        fidx = view(idx, 1:N)
        w = [view(w, fidx); sum(view(w, view(idx, (N + 1):M)))]
        bar(w; xticks = (1:(N + 1), [nx[fidx]; "Others"]), kwargs..., ekwargs...)
    else
        bar(w; xticks = (1:M, nx), kwargs..., ekwargs...)
    end
end
function PortfolioOptimisers.plot_risk_contribution(r::PortfolioOptimisers.AbstractBaseRiskMeasure,
                                                    w::VecNum, X::MatNum_Pr,
                                                    fees::Option{<:Fees} = nothing;
                                                    nx::AbstractVector = 1:length(w),
                                                    N::Option{<:Number} = nothing,
                                                    percentage::Bool = false,
                                                    delta::Number = 1e-6,
                                                    marginal::Bool = false,
                                                    kwargs::NamedTuple = (title = "Risk Contribution",
                                                                          xlabel = "Asset",
                                                                          ylabel = "Risk Contribution",
                                                                          xrotation = 90,
                                                                          legend = false),
                                                    ekwargs...)
    rc = risk_contribution(r, w, X, fees; delta = delta, marginal = marginal)
    if percentage
        rc /= sum(rc)
    end
    return PortfolioOptimisers.plot_composition(rc, nx; N = N, kwargs = kwargs, ekwargs...)
end
function PortfolioOptimisers.plot_stacked_bar_composition(w::VecNum_VecVecNum,
                                                          nx::AbstractVector = 1:size(w, 1);
                                                          kwargs::NamedTuple = (;
                                                                                xlabel = "Portfolios",
                                                                                ylabel = "Weight",
                                                                                title = "Portfolio Composition",
                                                                                legend = :outerright),
                                                          ekwargs...)
    if isa(w, VecVecNum)
        w = hcat(w...)
    end
    M = size(w, 2)
    ctg = repeat(nx; inner = M)
    return groupedbar(transpose(w); xticks = (1:M, 1:M), bar_position = :stack, group = ctg,
                      kwargs..., ekwargs...)
end
function PortfolioOptimisers.plot_stacked_area_composition(w::VecNum_VecVecNum,
                                                           nx::AbstractVector = 1:size(w, 1);
                                                           kwargs::NamedTuple = (;
                                                                                 xlabel = "Portfolios",
                                                                                 ylabel = "Weight",
                                                                                 title = "Portfolio Composition",
                                                                                 legend = :outerright),
                                                           ekwargs...)
    if isa(w, VecVecNum)
        w = hcat(w...)
    end
    M = size(w, 2)
    return areaplot(transpose(w); xticks = (1:M, 1:M), label = permutedims(nx), kwargs...,
                    ekwargs...)
end
function PortfolioOptimisers.plot_dendrogram(clr::PortfolioOptimisers.AbstractClusteringResult,
                                             nx::AbstractVector = 1:length(clr.clustering.order);
                                             dend_theme = :Spectral,
                                             dend_kwargs = (; xrotation = 90),
                                             fig_kwargs = (; size = (600, 600)), ekwargs...)
    N = length(clr.clustering.order)
    nx = view(nx, clr.clustering.order)
    idx = cutree(clr.clustering; k = clr.k)
    cls = [findall(x -> x == i, idx) for i in 1:(clr.k)]
    colours = palette(dend_theme, clr.k)
    dend1 = plot(clr.clustering; normalize = false, ylim = extrema(clr.clustering.heights),
                 xticks = (1:N, nx), dend_kwargs...)
    for (i, cl) in pairs(cls)
        a = [findfirst(x -> x == c, clr.clustering.order) for c in cl]
        a = a[.!isnothing.(a)]
        xmin = minimum(a)
        xmax = xmin + length(cl)
        i1 = [findfirst(x -> x == c, -view(clr.clustering.merges, :, 1)) for c in cl]
        i1 = i1[.!isnothing.(i1)]
        i2 = [findfirst(x -> x == c, -view(clr.clustering.merges, :, 2)) for c in cl]
        i2 = i2[.!isnothing.(i2)]
        i3 = unique([i1; i2])
        h = min(maximum(clr.clustering.heights[i3]) * 1.1, 1)
        plot!(dend1,
              [xmin - 0.25, xmax - 0.75, xmax - 0.75, xmax - 0.75, xmax - 0.75, xmin - 0.25,
               xmin - 0.25, xmin - 0.25], [0, 0, 0, h, h, h, h, 0]; color = nothing,
              legend = false, fill = (0, 0.5, colours[(i - 1) % clr.k + 1]))
    end
    return plot(dend1; fig_kwargs..., ekwargs...)
end
function PortfolioOptimisers.plot_clusters(pe::PrE_Pr, cle::ClE_Cl,
                                           rd::PortfolioOptimisers.ReturnsResult = ReturnsResult();
                                           dims::Integer = 1,
                                           color_func = x -> if any(x .< zero(eltype(x)))
                                               (-1, 1)
                                           else
                                               (0, 1)
                                           end, dend_theme = :Spectral,
                                           hmap_theme = :Spectral, hmap_theme_kwargs = (;),
                                           dend1_kwargs = (;), dend2_kwargs = (;),
                                           hmap_kwargs = (;),
                                           line_kwargs = (; color = :black, linewidth = 3),
                                           fig_kwargs = (; size = (600, 600)), ekwargs...)
    pr = prior(pe, rd; dims = dims)
    clr = clusterise(cle, pr.X; iv = rd.iv, ivpa = rd.ivpa, dims = dims)
    nx = !isnothing(rd.nx) ? rd.nx : (1:size(pr.X, 2))

    X = copy(clr.S)
    s = LinearAlgebra.diag(X)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end
    clim = color_func(X)
    N = size(X, 1)
    X = view(X, clr.clustering.order, clr.clustering.order)
    nx = view(nx, clr.clustering.order)
    idx = cutree(clr.clustering; k = clr.k)
    cls = [findall(x -> x == i, idx) for i in 1:(clr.k)]
    colours = palette(dend_theme, clr.k)
    colgrad = cgrad(hmap_theme; hmap_theme_kwargs...)
    hmap = plot(X; st = :heatmap, yticks = (1:N, nx), xticks = (1:N, nx), xrotation = 90,
                colorbar = false, clim = clim, xlim = (0.5, N + 0.5), ylim = (0.5, N + 0.5),
                color = colgrad, yflip = true, hmap_kwargs...)
    dend1 = plot(clr.clustering; xticks = false, ylim = extrema(clr.clustering.heights),
                 dend1_kwargs...)
    dend2 = plot(clr.clustering; yticks = false, xrotation = 90, orientation = :horizontal,
                 yflip = true, xlim = extrema(clr.clustering.heights), dend2_kwargs...)
    for (i, cl) in pairs(cls)
        a = [findfirst(x -> x == c, clr.clustering.order) for c in cl]
        a = a[.!isnothing.(a)]
        xmin = minimum(a)
        xmax = xmin + length(cl)
        i1 = [findfirst(x -> x == c, -view(clr.clustering.merges, :, 1)) for c in cl]
        i1 = i1[.!isnothing.(i1)]
        i2 = [findfirst(x -> x == c, -view(clr.clustering.merges, :, 2)) for c in cl]
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
                hmap, dend2; layout = l, fig_kwargs..., ekwargs...)
end
function PortfolioOptimisers.plot_drawdowns(w::ArrNum, X::MatNum, slv::Slv_VecSlv,
                                            fees::Option{<:Fees} = nothing;
                                            ts::AbstractVector = 1:size(X, 1),
                                            compound::Bool = false, alpha::Number = 0.05,
                                            kappa::Number = 0.3,
                                            rw::Option{<:StatsBase.AbstractWeights} = nothing,
                                            theme::Symbol = :Dark2_5,
                                            dd_kwargs = (;
                                                         label = "$(compound ? "Compounded" : "Uncompounded") Drawdown",
                                                         ylabel = "$(compound ? "Compounded" : "Uncompounded")\nDrawdown Percentage",
                                                         xlabel = "Date", linewidth = 2,
                                                         yguidefontsize = 10),
                                            dd_func = x -> extrema(x) .* [1.2, 1.01],
                                            l_kwargs::NamedTuple = (; linewidth = 2,
                                                                    legend = :bottomleft),
                                            ret_kwargs::NamedTuple = (;
                                                                      ylabel = "$(compound ? "Compounded" : "Uncompounded")\nCumulative Returns",
                                                                      linewidth = 2,
                                                                      legend = false,
                                                                      yguidefontsize = 10),
                                            f_kwargs::NamedTuple = (;
                                                                    size = (750,
                                                                            ceil(Integer,
                                                                                 750 /
                                                                                 1.618))),
                                            ekwargs...)
    ret = calc_net_returns(w, X, fees)
    cret = cumulative_returns(ret, compound)
    dd = drawdowns(cret, compound; cX = true)
    dd .*= 100
    risks = 100 * if !compound
                  [-AverageDrawdown(; w = rw)(copy(ret)), -UlcerIndex()(copy(ret)),
                   -DrawdownatRisk(; alpha = alpha)(copy(ret)),
                   -ConditionalDrawdownatRisk(; alpha = alpha)(copy(ret)),
                   -EntropicDrawdownatRisk(; slv = slv, alpha = alpha)(copy(ret)),
                   -RelativisticDrawdownatRisk(; slv = slv, alpha = alpha, kappa = kappa)(copy(ret)),
                   -MaximumDrawdown()(copy(ret))]
                  else
                  [-RelativeAverageDrawdown(; w = rw)(copy(ret)), -RelativeUlcerIndex()(copy(ret)),
                   -RelativeDrawdownatRisk(; alpha = alpha)(copy(ret)),
                   -RelativeConditionalDrawdownatRisk(; alpha = alpha)(copy(ret)),
                   -RelativeEntropicDrawdownatRisk(; slv = slv, alpha = alpha)(copy(ret)),
                   -RelativeRelativisticDrawdownatRisk(; slv = slv, alpha = alpha, kappa = kappa)(copy(ret)),
                   -RelativeMaximumDrawdown()(copy(ret))]
                  end
    conf = round((1 - alpha) * 100; digits = 2)
    labels = ("Average Drawdown: $(round(risks[1], digits = 2))%",
              "Ulcer Index: $(round(risks[2], digits = 2))%",
              "$(conf)% Confidence DaR: $(round(risks[3], digits = 2))%",
              "$(conf)% Confidence CDaR: $(round(risks[4], digits = 2))%",
              "$(conf)% Confidence EDaR: $(round(risks[5], digits = 2))%",
              "$(conf)% Confidence RLDaR ($(round(kappa, digits=2))): $(round(risks[6], digits = 2))%",
              "Maximum Drawdown: $(round(risks[7], digits = 2))%")
    colours = palette(theme, length(labels) + 1)
    f_dd = plot(ts, dd; color = colours[1], ylim = dd_func(dd), dd_kwargs...)
    for (i, (risk, label)) in enumerate(zip(risks, labels))
        hline!(f_dd, [risk]; label = label, color = colours[i + 1], l_kwargs...)
    end
    f_ret = plot(ts, cret; color = colours[1], ret_kwargs...)
    return plot(f_ret, f_dd; layout = (2, 1), f_kwargs..., ekwargs...)
end
function PortfolioOptimisers.plot_measures(w::VecNum_VecVecNum,
                                           pr::PortfolioOptimisers.AbstractPriorResult,
                                           fees::Option{<:Fees} = nothing;
                                           x::PortfolioOptimisers.AbstractBaseRiskMeasure = Variance(),
                                           y::PortfolioOptimisers.AbstractBaseRiskMeasure = ReturnRiskMeasure(),
                                           z::Option{<:PortfolioOptimisers.AbstractBaseRiskMeasure} = nothing,
                                           c::PortfolioOptimisers.AbstractBaseRiskMeasure = RatioRiskMeasure(;
                                                                                                             rk = x,
                                                                                                             rt = ArithmeticReturn(),
                                                                                                             rf = 0),
                                           slv::Option{<:Slv_VecSlv} = nothing,
                                           flag::Bool = true,
                                           kwargs::NamedTuple = (title = "Pareto Frontier",
                                                                 xlabel = "X", ylabel = "Y",
                                                                 zlabel = "Z",
                                                                 label = nothing,
                                                                 colorbar_title = "C",
                                                                 legend = true,
                                                                 xrotation = 0,
                                                                 yrotation = 0), ekwargs...)
    if flag
        x = factory(x, pr)
        y = factory(y, pr)
        z = isnothing(z) ? nothing : factory(z, pr, slv)
        c = factory(c, pr)
    end
    xr = expected_risk(x, w, pr, fees)
    yr = expected_risk(y, w, pr, fees)
    zr = isnothing(z) ? nothing : expected_risk(z, w, pr, fees)
    cr = expected_risk(c, w, pr, fees)
    return if isnothing(zr)
        scatter(xr, yr; zcolor = cr, kwargs..., ekwargs...)
    else
        scatter(xr, yr, zr; zcolor = cr, kwargs..., ekwargs...)
    end
end
function PortfolioOptimisers.plot_histogram(w::ArrNum, X::MatNum, slv::Slv_VecSlv,
                                            fees::Option{<:Fees} = nothing; flag = true,
                                            alpha::Number = 0.05, kappa::Number = 0.3,
                                            points::Integer = ceil(Int,
                                                                   4 * sqrt(size(X, 1))),
                                            rw::Option{<:StatsBase.AbstractWeights} = nothing,
                                            theme::Symbol = :Paired_10,
                                            h_kwargs::NamedTuple = (;
                                                                    ylabel = "Probability Density",
                                                                    xlabel = "Percentage Returns",
                                                                    alpha = 0.5),
                                            l_kwargs::NamedTuple = (; linewidth = 2),
                                            pdf_kwargs::NamedTuple = (; linewidth = 2),
                                            e_kwargs...)
    ret = calc_net_returns(w, X, fees)
    mu = Statistics.mean(ret)
    sigma = std(ret)
    mir, mar = extrema(ret)
    x = range(mir, mar; length = points)
    mad = LowOrderMoment(; w = rw, alg = MeanAbsoluteDeviation())(w, X, fees)
    gmd = OrderedWeightsArray()(copy(ret))
    risks = (mu, mu - sigma, mu - mad, mu - gmd,
             -ValueatRisk(; w = rw, alpha = alpha)(copy(ret)),
             -ConditionalValueatRisk(; w = rw, alpha = alpha)(copy(ret)),
             -OrderedWeightsArray(; w = owa_tg(length(ret)))(copy(ret)),
             -EntropicValueatRisk(; w = rw, slv = slv, alpha = alpha)(copy(ret)),
             -RelativisticValueatRisk(; w = rw, slv = slv, alpha = alpha, kappa = kappa)(copy(ret)),
             mir)
    conf = round((1 - alpha) * 100; digits = 2)
    risk_labels = ("Mean: $(round(risks[1], digits=2))%",
                   "Mean - Std. Dev. ($(round(sigma, digits=2))%): $(round(risks[2], digits=2))%",
                   "Mean - MAD ($(round(mad,digits=2))%): $(round(risks[3], digits=2))%",
                   "Mean - GMD ($(round(gmd,digits=2))%): $(round(risks[4], digits=2))%",
                   "$(conf)% Confidence VaR: $(round(risks[5], digits=2))%",
                   "$(conf)% Confidence CVaR: $(round(risks[6], digits=2))%",
                   "$(conf)% Confidence Tail Gini: $(round(risks[7], digits=2))%",
                   "$(conf)% Confidence EVaR: $(round(risks[8], digits=2))%",
                   "$(conf)% Confidence RLVaR ($(round(kappa, digits=2))): $(round(risks[9], digits=2))%",
                   "Worst Realisation: $(round(risks[10], digits=2))%")
    colours = palette(theme, length(risk_labels) + 2)
    plt = histogram(ret; normalize = :pdf, label = "", color = colours[1], h_kwargs...)
    for (i, (risk, label)) in enumerate(zip(risks, risk_labels)) #! Do not change this enumerate to pairs.
        vline!([risk]; label = label, color = colours[i + 1], l_kwargs...)
    end
    D = StatsAPI.fit(Normal, ret)
    if flag
        density!(ret;
                 label = "Normal: μ = $(round(mean(D), digits=2))%, σ = $(round(std(D), digits=2))%",
                 color = colours[end], pdf_kwargs..., e_kwargs...)
    else
        plot!(x, pdf.(D, x);
              label = "Normal: μ = $(round(mean(D), digits=2))%, σ = $(round(std(D), digits=2))%",
              color = colours[end], pdf_kwargs..., e_kwargs...)
    end
    return plt
end

end
