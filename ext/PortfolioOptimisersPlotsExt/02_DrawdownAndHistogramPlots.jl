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
    _, w, rd, fees = result_investable_view(res, rd)
    return PortfolioOptimisers.plot_drawdowns(w, rd, fees; slv = slv, compound = compound,
                                              alpha = alpha, kappa = kappa, rw = rw,
                                              kwargs...)
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
    _, w, rd, fees = result_investable_view(res, rd)
    return PortfolioOptimisers.plot_histogram(w, rd.X, fees; slv = slv, alpha = alpha,
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
                    ylabel = "Probability Density", xlabel = "Returns", kwargs...)
    for (i, (risk, lbl)) in enumerate(zip(risks, risk_labels))
        vline!([risk]; label = lbl, color = colours[i + 1], linewidth = 2)
    end
    if reference
        D = StatsAPI.fit(Normal, ret)
        plot!(x_range, pdf.(D, x_range);
              label = "Normal: μ=$(round(100*mean(D); digits=2))%, σ=$(round(100*std(D); digits=2))%",
              color = colours[end], linewidth = 2)
    end
    return plt
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
    if window > T
        throw(DomainError(rolling,
                          "rolling must be no longer than the return series, which has $T observations"))
    end
    # Each window is drawn down from its own entry value, so hand `drawdowns` the window's
    # returns rather than a slice of the whole-history cumulative series -- the running peak
    # is seeded with the capital at the start of the window, not the capital at inception.
    rolling_mdd = [minimum(drawdowns(view(ret, (t - window + 1):t), compound)) * 100
                   for t in window:T]
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
    _, w, rd, fees = result_investable_view(res, rd)
    return PortfolioOptimisers.plot_rolling_drawdowns(w, rd, fees; rolling = rolling,
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
