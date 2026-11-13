## plot_measures
function PortfolioOptimisers.plot_measures(w::VecNum_VecVecNum, pr::Pr_RR,
                                           fees::Option{<:Fees} = nothing;
                                           x::PortfolioOptimisers.BaseRM_VecBaseRM = Variance(),
                                           y::PortfolioOptimisers.BaseRM_VecBaseRM = ExpectedReturn(),
                                           z::Option{<:PortfolioOptimisers.BaseRM_VecBaseRM} = nothing,
                                           c::PortfolioOptimisers.BaseRM_VecBaseRM = ExpectedReturnRiskRatio(;
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
                                           x::PortfolioOptimisers.BaseRM_VecBaseRM = Variance(),
                                           y::PortfolioOptimisers.BaseRM_VecBaseRM = ExpectedReturn(),
                                           z::Option{<:PortfolioOptimisers.BaseRM_VecBaseRM} = nothing,
                                           c::PortfolioOptimisers.BaseRM_VecBaseRM = ExpectedReturnRiskRatio(;
                                                                                                             rk = x,
                                                                                                             rt = ArithmeticReturn(),
                                                                                                             rf = 0),
                                           slv::Option{<:Slv_VecSlv} = nothing,
                                           fees::Option{<:Fees} = nothing,
                                           factory::Bool = true, kwargs...)
    views = [result_investable_view(res, pr, fees) for res in res_vec]
    w = getindex.(views, 2)
    pr = getindex.(views, 3)
    fees = getindex.(views, 4)
    # Each axis becomes one measure **per result**, so the broadcasts below zip correctly.
    # `Ref` is what makes that safe: a single measure is not iterable and broadcasts as a
    # scalar, but a **vector** of measures is, so without `Ref` it would zip elementwise
    # against `pr` and `slv` — a `DimensionMismatch` when the lengths differ and a silent
    # wrong answer when they happen to match. `fill` gives the un-factoried branch the same
    # per-result shape, so one broadcast serves both.
    n = length(res_vec)
    function per_result(m)
        return factory ? PortfolioOptimisers.factory.(Ref(m), pr, slv) : fill(m, n)
    end
    x = per_result(x)
    y = per_result(y)
    z = isnothing(z) ? nothing : per_result(z)
    c = per_result(c)
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
function PortfolioOptimisers.plot_measures(ppred::PortfolioOptimisers.AbstractPredictionResult;
                                           x::PortfolioOptimisers.BaseRM_VecBaseRM = ConditionalValueatRisk(),
                                           y::PortfolioOptimisers.BaseRM_VecBaseRM = MeanReturn(),
                                           z::Option{<:PortfolioOptimisers.BaseRM_VecBaseRM} = nothing,
                                           c::PortfolioOptimisers.BaseRM_VecBaseRM = MeanReturnRiskRatio(;
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
    # A prediction and a walk-forward path score to one number and a population to one per
    # member, so `vcat` gives the scatter a vector either way: one point, or one per member.
    xr = vcat(expected_risk(x, ppred))
    yr = vcat(expected_risk(y, ppred))
    zr = isnothing(z) ? nothing : vcat(expected_risk(z, ppred))
    cr = vcat(expected_risk(c, ppred))
    return if isnothing(zr)
        if isnothing(plt)
            scatter(xr, yr; zcolor = cr, title = "Pareto Front", xlabel = "X", ylabel = "Y",
                    colorbar_title = "C", label = nothing, legend = true, kwargs...)
        else
            scatter!(plt, xr, yr; zcolor = cr, title = "Pareto Front", xlabel = "X",
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
## plot_rolling_measure
function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                                  w::VecNum, rd::ReturnsResult,
                                                  fees::Option{<:Fees} = nothing;
                                                  rolling::Integer = 0, kwargs...)
    ts = isnothing(rd.ts) ? (1:size(rd.X, 1)) : rd.ts
    return PortfolioOptimisers.plot_rolling_measure(r, w, rd.X, fees; ts = ts,
                                                    rolling = rolling, kwargs...)
end
function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                                  res::OptimisationResult,
                                                  rd::ReturnsResult; rolling::Integer = 0,
                                                  kwargs...)
    _, w, rd, fees = result_investable_view(res, rd)
    return PortfolioOptimisers.plot_rolling_measure(r, w, rd, fees; rolling = rolling,
                                                    kwargs...)
end
function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                                  pred::PredictionResult;
                                                  rolling::Integer = 0, kwargs...)
    rd = pred.rd
    ret = isa(rd.X, VecVecNum) ? first(rd.X) : rd.X
    ts = isnothing(rd.ts) ? (1:length(ret)) : rd.ts
    return PortfolioOptimisers.plot_rolling_measure(r, ret; ts = ts, rolling = rolling,
                                                    kwargs...)
end
function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                                  pred::MultiPeriodPredictionResult;
                                                  rolling::Integer = 0, kwargs...)
    mrd = pred.mrd
    ret = isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
    ts = isnothing(mrd.ts) ? (1:length(ret)) : mrd.ts
    return PortfolioOptimisers.plot_rolling_measure(r, ret; ts = ts, rolling = rolling,
                                                    kwargs...)
end
function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.BaseRM_VecBaseRM,
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
## Performance summary
## ────────────────────────────────────────────────────────────────────────────

# The figures of a summary, scaled for the bars, with their labels. Each figure is a scalar
# for a summary of one entry and a vector for a summary of many. Each extra measure follows
# the released figures, unscaled, under its own name.
function performance_summary_figures(ps::PerformanceSummaryResult)
    conf = round((1 - ps.alpha) * 100; digits = 1)
    # The literal promotes the figures, as the plot does, and takes their element type (#1979).
    vals = [ps.ann_return * 100, ps.ann_volatility * 100, ps.sharpe, ps.sortino, ps.calmar,
            ps.max_drawdown * 100, ps.cvar * 100, ps.excess_ret * 100,
            ps.tracking_error * 100, ps.information_ratio, ps.turnover * 100,
            values(ps.risk)...]
    labels = ["Ann. Return %", "Ann. Vol %", "Sharpe", "Sortino", "Calmar", "Max DD %",
              "$(conf)% CVaR %", "Excess Ret %", "Track. Err %", "Info Ratio", "Turnover %",
              (string(k) for k in keys(ps.risk))...]
    return vals, labels
end
# The label of each entry names the keys that are not `nothing`: the candidate, the path and
# the member. An entry with no key takes its position.
function performance_summary_entry_labels(ps::PerformanceSummaryResult)
    return map(eachindex(ps.n_periods)) do k
        parts = String[]
        if !(isnothing(ps.names[k]))
            push!(parts, string(ps.names[k]))
        end
        if !(isnothing(ps.paths[k]))
            push!(parts, "path $(ps.paths[k])")
        end
        if !(isnothing(ps.members[k]))
            push!(parts, "member $(ps.members[k])")
        end
        return isempty(parts) ? "Entry $(k)" : join(parts, ", ")
    end
end
function performance_summary_bar_colours(v)
    return [x >= 0 ? :steelblue : :firebrick for x in v]
end
# One entry draws one bar chart, with one bar for each figure.
function performance_summary_plot(::Integer, ps::PerformanceSummaryResult; kwargs...)
    vals, labels = performance_summary_figures(ps)
    v = collect(promote(vals...))
    return bar(v; xticks = (1:length(labels), labels), xrotation = 30,
               title = "Performance Summary", ylabel = "Value", legend = false,
               color = performance_summary_bar_colours(v), kwargs...)
end
# Many entries draw one panel for each figure, with one bar for each entry.
function performance_summary_plot(::AbstractVector, ps::PerformanceSummaryResult; kwargs...)
    vals, labels = performance_summary_figures(ps)
    names = performance_summary_entry_labels(ps)
    K = length(names)
    # An index, not a destructure of `zip`: JET reads the element type of `vals` now, and it
    # refused the `getfield` of the destructure (#1979).
    panels = [bar(1:K, vals[i]; xticks = (1:K, names), xrotation = 30, title = labels[i],
                  legend = false, color = performance_summary_bar_colours(vals[i]))
              for i in eachindex(vals, labels)]
    return plot(panels...; layout = length(panels), kwargs...)
end
function PortfolioOptimisers.plot_performance_summary(ps::PerformanceSummaryResult;
                                                      kwargs...)
    return performance_summary_plot(ps.n_periods, ps; kwargs...)
end
# Every other arity computes the summary through `performance_summary`, whose own methods
# validate it, and renders it through the method above. The keywords of the verb go to the
# summary and the rest to the bars. A prediction reads the benchmark of its rows when the
# caller gives none, as its `performance_summary` method does (#1664), because the keyword
# goes to the verb only when the caller gives it.
const PERFORMANCE_SUMMARY_KEYWORDS = (:ppy, :alpha, :compound, :rf, :measures, :benchmark,
                                      :names)
function PortfolioOptimisers.plot_performance_summary(x::Union{AbstractArray,
                                                               OptimisationResult,
                                                               PortfolioOptimisers.AbstractPredictionResult},
                                                      args...; kwargs...)
    skw = (; (p for p in pairs(kwargs) if first(p) in PERFORMANCE_SUMMARY_KEYWORDS)...)
    pkw = (; (p for p in pairs(kwargs) if !(first(p) in PERFORMANCE_SUMMARY_KEYWORDS))...)
    ps = performance_summary(x, args...; skw...)
    return PortfolioOptimisers.plot_performance_summary(ps; pkw...)
end
## ────────────────────────────────────────────────────────────────────────────
## Rolling measure base
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                                  w::VecNum, X::MatNum,
                                                  fees::Option{<:Fees} = nothing;
                                                  ts::AbstractVector = 1:size(X, 1),
                                                  rolling::Integer = 0, kwargs...)
    return PortfolioOptimisers.plot_rolling_measure(r, calc_net_returns(w, X, fees);
                                                    ts = ts, rolling = rolling, kwargs...)
end
function PortfolioOptimisers.plot_rolling_measure(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                                  ret::VecNum; ts::AbstractVector,
                                                  rolling::Integer = 0,
                                                  sca::Option{<:Scalariser} = nothing,
                                                  kwargs...)
    if !(rolling >= 0)
        throw(DomainError(rolling, "rolling must be >= 0"))
    end
    T = length(ret)
    window = rolling == 0 ? ceil(Int, sqrt(T)) : rolling
    # Off the functor, and through the library's own verb rather than a second copy of the
    # rolling loop. `rolling_window_measure` scores each window with
    # `expected_risk_from_returns`, which serves a measure and a vector alike and on a single
    # supported measure returns `r(x)`, so the number is unchanged. It also refuses a `rolling`
    # longer than the sample, which this method's own `rolling >= 0` check never caught.
    rolling_vals = PortfolioOptimisers.rolling_window_measure(r, ret, window; sca = sca)
    ts_rolling = ts[window:end]
    rname = measure_label(r)
    return plot(ts_rolling, rolling_vals; title = "Rolling $rname (window=$window)",
                ylabel = rname, xlabel = "Date", legend = false, linewidth = 2, kwargs...)
end
