## ────────────────────────────────────────────────────────────────────────────
## The forecast evaluation figures
## ────────────────────────────────────────────────────────────────────────────
# Every figure here takes the `ForecastEvaluationResult` and never the block, which is the
# one place this group diverges from the cross-sectional diagnostics above. Those verbs read
# a block that is already fitted, so a figure that calls one costs nothing; here `alpha` can
# cost a rolling refit through `forecast_history`, so the caller pairs once and every figure
# reads that pairing. A weighting still arrives as the second positional argument, exactly
# as it reaches the level-2 verbs of the group.
const FORECAST_IC_LABELS = ["Spearman", "Pearson"]
const FORECAST_WINDOW_BOOK_LABELS = ["Rank Ann. Return", "Rank Sharpe",
                                     "Z-Score Ann. Return", "Z-Score Sharpe"]
function forecast_plot_series(x::AbstractVector, vals::MatNum, labels::AbstractVector,
                              title::AbstractString, xlabel::AbstractString,
                              ylabel::AbstractString; kwargs...)
    plt = plot(x, view(vals, :, 1); title = title, xlabel = xlabel, ylabel = ylabel,
               label = labels[1], legend = true, linewidth = 2, kwargs...)
    for k in 2:size(vals, 2)
        plot!(plt, x, view(vals, :, k); label = labels[k], linewidth = 2, kwargs...)
    end
    return plt
end
# A date whose series carries no number contributes nothing to the running sum, so one
# silenced cross-section breaks no series. The running mean beside it is written here for
# the same reason `cumulative_exposure_ic` is: it transforms what the verb answered for the
# eye, and computes no statistic the library does not already hold.
function forecast_rolling_mean(x::AbstractVector{<:Real}, window::Integer)
    T = length(x)
    # A mean divides, so the type comes from the division and not from the argument: an
    # integer series averages in `Float64` and a `Float32` series stays in `Float32`. The
    # divisor is the finite count, an `Int`. The annotation is what inference reads, because
    # `typeof` alone answers an unbounded `DataType` and `Tf(NaN)` below would then read as a
    # call of every constructor in the world.
    Tf = typeof(one(eltype(x)) / one(Int))::Type{<:Number}
    out = fill(Tf(NaN), T)
    for t in window:T
        s = zero(Tf)
        n = 0
        for j in (t - window + 1):t
            v = x[j]
            if isfinite(v)
                s += v
                n += 1
            end
        end
        if !iszero(n)
            out[t] = s / n
        end
    end
    return out
end
# The cumulative return of one book, with the dates it could not trade held flat, which is
# what `plot_factor_cumulative_returns` does to an absent factor return.
function forecast_cumulative_book(ret::AbstractVector{<:Real}, compound::Bool)
    return cumulative_returns([isfinite(x) ? x : zero(x) for x in ret], compound)
end
function forecast_book_label(kind::Symbol)
    return kind === :rank ? "Rank" : "Z-Score"
end
function PortfolioOptimisers.plot_forecast_cumulative_ic(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                         w::Option{<:MatNum} = nothing;
                                                         min_count::Integer = fe.min_count,
                                                         kwargs...)
    dates::AbstractVector{<:Integer} = fe.dates
    ic = PortfolioOptimisers.forecast_ic(fe, w; min_count = min_count)
    return forecast_plot_series(dates, cumulative_exposure_ic(ic), FORECAST_IC_LABELS,
                                "Cumulative Forecast IC", "Observation", "Cumulative IC";
                                kwargs...)
end
function PortfolioOptimisers.plot_forecast_cumulative_ic(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                         csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                         weighting = PortfolioOptimisers.IdentityMetric(),
                                                         min_count::Integer = fe.min_count,
                                                         kwargs...)
    return PortfolioOptimisers.plot_forecast_cumulative_ic(fe,
                                                           PortfolioOptimisers.cs_diagnostic_weights(weighting,
                                                                                                     csfm);
                                                           min_count = min_count, kwargs...)
end
# The comparison overlays. Each draws one series per evaluation on the dates the
# evaluations share, which `forecast_summary_assert_comparable` makes one axis, and reads the
# same verb the single-forecast figure reads. The threshold is each evaluation's own, for the
# reason `forecast_summary_row` states: the check makes it one number, and a keyword here
# would let the overlay be drawn under a threshold the books were not.
function forecast_overlay_series(fes::AbstractVector{<:PortfolioOptimisers.ForecastEvaluationResult},
                                 names, f, title::AbstractString, ylabel::AbstractString;
                                 kwargs...)
    PortfolioOptimisers.forecast_summary_assert_comparable(fes)
    labels = PortfolioOptimisers.forecast_summary_names(names, length(fes))
    dates::AbstractVector{<:Integer} = first(fes).dates
    # The element is asserted rather than taken from the iterator, for the reason
    # `forecast_evaluation_summary` states.
    series = [f(fes[k]::PortfolioOptimisers.ForecastEvaluationResult)
              for k in eachindex(fes)]
    Tf = promote_type(map(eltype, series)...)
    vals = Matrix{Tf}(undef, length(dates), length(series))
    for k in eachindex(series)
        vals[:, k] = series[k]
    end
    plt = forecast_plot_series(dates, vals, labels, title, "Observation", ylabel; kwargs...)
    return idio_diagnostic_reference!(plt, 0.0)
end
function forecast_overlay_ic(fe::PortfolioOptimisers.ForecastEvaluationResult, w,
                             col::Integer)
    return cumulative_exposure_ic(PortfolioOptimisers.forecast_ic(fe, w))[:, col]
end
function forecast_overlay_book(fe::PortfolioOptimisers.ForecastEvaluationResult,
                               kind::Symbol, compound::Bool)
    ret = PortfolioOptimisers.forecast_portfolio(fe; kind = kind).ret
    return forecast_cumulative_book(ret, compound)
end
function PortfolioOptimisers.plot_forecast_cumulative_ic(fes::AbstractVector{<:PortfolioOptimisers.ForecastEvaluationResult},
                                                         w::Option{<:MatNum} = nothing;
                                                         names = nothing, rank::Bool = true,
                                                         kwargs...)
    col = rank ? 1 : 2
    return forecast_overlay_series(fes, names, fe -> forecast_overlay_ic(fe, w, col),
                                   "Cumulative Forecast IC ($(FORECAST_IC_LABELS[col]))",
                                   "Cumulative IC"; kwargs...)
end
function PortfolioOptimisers.plot_forecast_cumulative_ic(fes::AbstractVector{<:PortfolioOptimisers.ForecastEvaluationResult},
                                                         csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                         weighting = PortfolioOptimisers.IdentityMetric(),
                                                         kwargs...)
    return PortfolioOptimisers.plot_forecast_cumulative_ic(fes,
                                                           PortfolioOptimisers.cs_diagnostic_weights(weighting,
                                                                                                     csfm);
                                                           kwargs...)
end
function PortfolioOptimisers.plot_forecast_rolling_ic(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                      w::Option{<:MatNum} = nothing;
                                                      rolling::Integer = 0,
                                                      min_count::Integer = fe.min_count,
                                                      kwargs...)
    dates::AbstractVector{<:Integer} = fe.dates
    ic = PortfolioOptimisers.forecast_ic(fe, w; min_count = min_count)
    T = size(ic, 1)
    window = iszero(rolling) ? ceil(Int, sqrt(T)) : rolling
    if !(1 <= window <= T)
        throw(DomainError(rolling, "rolling must be in 1:$(T), or 0 for √T"))
    end
    # The same derivation `forecast_rolling_mean` makes, over the same values, so the matrix
    # it fills and the vectors it answers agree on their element type.
    Tf = typeof(one(real(eltype(ic))) / one(Int))::Type{<:Number}
    roll = Matrix{Tf}(undef, T, 2)
    for k in 1:2
        roll[:, k] = forecast_rolling_mean(view(ic, :, k), window)
    end
    return forecast_plot_series(dates, roll, FORECAST_IC_LABELS,
                                "Rolling Forecast IC (window=$window)", "Observation",
                                "Mean IC"; kwargs...)
end
function PortfolioOptimisers.plot_forecast_rolling_ic(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                      csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                      weighting = PortfolioOptimisers.IdentityMetric(),
                                                      rolling::Integer = 0,
                                                      min_count::Integer = fe.min_count,
                                                      kwargs...)
    return PortfolioOptimisers.plot_forecast_rolling_ic(fe,
                                                        PortfolioOptimisers.cs_diagnostic_weights(weighting,
                                                                                                  csfm);
                                                        rolling = rolling,
                                                        min_count = min_count, kwargs...)
end
function PortfolioOptimisers.plot_forecast_cumulative_returns(fe::PortfolioOptimisers.ForecastEvaluationResult;
                                                              kinds = (:rank, :zscore),
                                                              compound::Bool = false,
                                                              kwargs...)
    dates::AbstractVector{<:Integer} = fe.dates
    rets = [PortfolioOptimisers.forecast_portfolio(fe; kind = k).ret for k in kinds]
    Tf = eltype(first(rets))
    cum = Matrix{Tf}(undef, length(dates), length(rets))
    for k in eachindex(rets)
        cum[:, k] = forecast_cumulative_book(rets[k], compound)
    end
    labels = [forecast_book_label(k) for k in kinds]
    kind = compound ? "Compounded" : "Uncompounded"
    return forecast_plot_series(dates, cum, labels,
                                "Forecast Book Cumulative Returns ($kind)", "Observation",
                                "Cumulative Return"; kwargs...)
end
function PortfolioOptimisers.plot_forecast_cumulative_returns(fes::AbstractVector{<:PortfolioOptimisers.ForecastEvaluationResult};
                                                              names = nothing,
                                                              kind::Symbol = :rank,
                                                              compound::Bool = false,
                                                              kwargs...)
    book = forecast_book_label(kind)
    comp = compound ? "Compounded" : "Uncompounded"
    return forecast_overlay_series(fes, names,
                                   fe -> forecast_overlay_book(fe, kind, compound),
                                   "Forecast $(book) Book Cumulative Returns ($comp)",
                                   "Cumulative Return"; kwargs...)
end
function PortfolioOptimisers.plot_forecast_quantile_returns(fe::PortfolioOptimisers.ForecastEvaluationResult;
                                                            quantiles = (0.1,),
                                                            compound::Bool = false,
                                                            kwargs...)
    dates::AbstractVector{<:Integer} = fe.dates
    spread = PortfolioOptimisers.forecast_quantile_spread(fe; quantiles = quantiles).spread
    Tf = eltype(spread)
    cum = Matrix{Tf}(undef, size(spread, 1), size(spread, 2))
    for k in axes(spread, 2)
        cum[:, k] = forecast_cumulative_book(view(spread, :, k), compound)
    end
    labels = ["q = $q" for q in quantiles]
    kind = compound ? "Compounded" : "Uncompounded"
    return forecast_plot_series(dates, cum, labels,
                                "Forecast Quantile Spread Returns ($kind)", "Observation",
                                "Cumulative Return"; kwargs...)
end
function PortfolioOptimisers.plot_forecast_calibration(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                       w::Option{<:MatNum} = nothing;
                                                       bins::Integer = 10, kwargs...)
    c = PortfolioOptimisers.forecast_calibration(fe, w; bins = bins)
    a = c.curve.mean_alpha
    slope = c.slope
    plt = scatter(a, c.curve.mean_y;
                  title = "Forecast Calibration (slope = $(round(slope; digits = 4)))",
                  xlabel = "Mean Forecast", ylabel = "Mean Realised Target",
                  label = "Curve", legend = true, kwargs...)
    plot!(plt, a, slope .* a; label = "Slope", linewidth = 2, linestyle = :dash,
          color = :red)
    return plt
end
function PortfolioOptimisers.plot_forecast_calibration(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                       csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                       weighting = PortfolioOptimisers.IdentityMetric(),
                                                       bins::Integer = 10, kwargs...)
    return PortfolioOptimisers.plot_forecast_calibration(fe,
                                                         PortfolioOptimisers.cs_diagnostic_weights(weighting,
                                                                                                   csfm);
                                                         bins = bins, kwargs...)
end
# The two window tables answer the same eleven figures, so the two pairs of figures share
# one series builder each and neither reads a table's fields in more than one place.
function forecast_window_ic_figure(period::AbstractVector, spearman::VecNum,
                                   pearson::VecNum, title::AbstractString; kwargs...)
    return forecast_plot_series(period, hcat(spearman, pearson), FORECAST_IC_LABELS, title,
                                "Period", "Mean IC"; kwargs...)
end
function forecast_window_book_figure(period::AbstractVector, rank_ret::VecNum,
                                     rank_sharpe::VecNum, zscore_ret::VecNum,
                                     zscore_sharpe::VecNum, title::AbstractString;
                                     kwargs...)
    return forecast_plot_series(period,
                                hcat(rank_ret, rank_sharpe, zscore_ret, zscore_sharpe),
                                FORECAST_WINDOW_BOOK_LABELS, title, "Period", "Value";
                                kwargs...)
end
function PortfolioOptimisers.plot_forecast_ic_by_holding_period(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                                X::MatNum,
                                                                w::Option{<:MatNum} = nothing;
                                                                n::Integer = 10,
                                                                min_count::Integer = fe.min_count,
                                                                kwargs...)
    t = PortfolioOptimisers.forecast_holding_period(fe, X, w; n = n, min_count = min_count)
    return forecast_window_ic_figure(t.period, t.spearman_mean_ic, t.pearson_mean_ic,
                                     "Forecast IC by Holding Period"; kwargs...)
end
function PortfolioOptimisers.plot_forecast_ic_by_holding_period(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                                rd::ReturnsResult,
                                                                csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                                weighting = PortfolioOptimisers.IdentityMetric(),
                                                                n::Integer = 10,
                                                                min_count::Integer = fe.min_count,
                                                                kwargs...)
    return PortfolioOptimisers.plot_forecast_ic_by_holding_period(fe,
                                                                  PortfolioOptimisers.forecast_target_history(fe.target,
                                                                                                              rd,
                                                                                                              csfm),
                                                                  PortfolioOptimisers.cs_diagnostic_weights(weighting,
                                                                                                            csfm);
                                                                  n = n,
                                                                  min_count = min_count,
                                                                  kwargs...)
end
function PortfolioOptimisers.plot_forecast_portfolio_by_holding_period(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                                       X::MatNum,
                                                                       w::Option{<:MatNum} = nothing;
                                                                       n::Integer = 10,
                                                                       min_count::Integer = fe.min_count,
                                                                       kwargs...)
    t = PortfolioOptimisers.forecast_holding_period(fe, X, w; n = n, min_count = min_count)
    return forecast_window_book_figure(t.period, t.rank_ann_return, t.rank_sharpe,
                                       t.zscore_ann_return, t.zscore_sharpe,
                                       "Forecast Books by Holding Period"; kwargs...)
end
function PortfolioOptimisers.plot_forecast_portfolio_by_holding_period(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                                       rd::ReturnsResult,
                                                                       csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                                       weighting = PortfolioOptimisers.IdentityMetric(),
                                                                       n::Integer = 10,
                                                                       min_count::Integer = fe.min_count,
                                                                       kwargs...)
    return PortfolioOptimisers.plot_forecast_portfolio_by_holding_period(fe,
                                                                         PortfolioOptimisers.forecast_target_history(fe.target,
                                                                                                                     rd,
                                                                                                                     csfm),
                                                                         PortfolioOptimisers.cs_diagnostic_weights(weighting,
                                                                                                                   csfm);
                                                                         n = n,
                                                                         min_count = min_count,
                                                                         kwargs...)
end
function PortfolioOptimisers.plot_forecast_ic_decay(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                    X::MatNum,
                                                    w::Option{<:MatNum} = nothing;
                                                    n::Integer = 10,
                                                    min_count::Integer = fe.min_count,
                                                    kwargs...)
    t = PortfolioOptimisers.forecast_decay(fe, X, w; n = n, min_count = min_count)
    return forecast_window_ic_figure(t.period, t.spearman_mean_ic, t.pearson_mean_ic,
                                     "Forecast IC Decay"; kwargs...)
end
function PortfolioOptimisers.plot_forecast_ic_decay(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                    rd::ReturnsResult,
                                                    csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                    weighting = PortfolioOptimisers.IdentityMetric(),
                                                    n::Integer = 10,
                                                    min_count::Integer = fe.min_count,
                                                    kwargs...)
    return PortfolioOptimisers.plot_forecast_ic_decay(fe,
                                                      PortfolioOptimisers.forecast_target_history(fe.target,
                                                                                                  rd,
                                                                                                  csfm),
                                                      PortfolioOptimisers.cs_diagnostic_weights(weighting,
                                                                                                csfm);
                                                      n = n, min_count = min_count,
                                                      kwargs...)
end
function PortfolioOptimisers.plot_forecast_portfolio_decay(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                           X::MatNum,
                                                           w::Option{<:MatNum} = nothing;
                                                           n::Integer = 10,
                                                           min_count::Integer = fe.min_count,
                                                           kwargs...)
    t = PortfolioOptimisers.forecast_decay(fe, X, w; n = n, min_count = min_count)
    return forecast_window_book_figure(t.period, t.rank_ann_return, t.rank_sharpe,
                                       t.zscore_ann_return, t.zscore_sharpe,
                                       "Forecast Books by Decay Window"; kwargs...)
end
function PortfolioOptimisers.plot_forecast_portfolio_decay(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                           rd::ReturnsResult,
                                                           csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                           weighting = PortfolioOptimisers.IdentityMetric(),
                                                           n::Integer = 10,
                                                           min_count::Integer = fe.min_count,
                                                           kwargs...)
    return PortfolioOptimisers.plot_forecast_portfolio_decay(fe,
                                                             PortfolioOptimisers.forecast_target_history(fe.target,
                                                                                                         rd,
                                                                                                         csfm),
                                                             PortfolioOptimisers.cs_diagnostic_weights(weighting,
                                                                                                       csfm);
                                                             n = n, min_count = min_count,
                                                             kwargs...)
end
function PortfolioOptimisers.plot_forecast_factor_correlation(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                              B::Arr3Num,
                                                              w::Option{<:MatNum} = nothing;
                                                              nf::Option{<:AbstractVector} = nothing,
                                                              dates::AbstractVector{<:Integer} = axes(fe.alpha,
                                                                                                      1),
                                                              rank::Bool = false,
                                                              min_count::Integer = fe.min_count,
                                                              kwargs...)
    c = PortfolioOptimisers.forecast_factor_correlation(fe, B, w; dates = dates,
                                                        rank = rank, min_count = min_count)
    labels = isnothing(nf) ? string.(1:size(c, 2)) : string.(nf)
    method = rank ? "Spearman" : "Pearson"
    return forecast_plot_series(dates, c, labels, "Forecast Factor Correlation ($method)",
                                "Observation", "ρ"; kwargs...)
end
function PortfolioOptimisers.plot_forecast_factor_correlation(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                              csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                              nf::Option{<:AbstractVector} = nothing,
                                                              weighting = PortfolioOptimisers.IdentityMetric(),
                                                              kwargs...)
    B = PortfolioOptimisers.cs_diagnostic_exposures(csfm)
    return PortfolioOptimisers.plot_forecast_factor_correlation(fe, B,
                                                                PortfolioOptimisers.cs_diagnostic_weights(weighting,
                                                                                                          csfm);
                                                                nf = exposure_diagnostic_labels(csfm,
                                                                                                nf,
                                                                                                size(B,
                                                                                                     3)),
                                                                kwargs...)
end
# The summary figure follows `plot_factor_model_summary`: the axis is the statistic, the
# series is the forecast, and a block the Result carries as `nothing` is not drawn and the
# title says so. The quantile block is the one such block here, and it contributes one row
# per quantile rather than one row, because its axis is `forecasts × quantiles`.
const FORECAST_SUMMARY_LABELS = ["Spearman IC", "Spearman IR", "Pearson IC", "Pearson IR",
                                 "Rank Ann. Return", "Rank Sharpe", "Z-Score Ann. Return",
                                 "Z-Score Sharpe", "Calibration Slope", "Mean Coverage"]
function forecast_summary_columns(fs::PortfolioOptimisers.ForecastSummaryResult)
    vals = Any[fs.spearman_mean_ic, fs.spearman_ic_ir, fs.pearson_mean_ic, fs.pearson_ic_ir,
               fs.rank_ann_return, fs.rank_sharpe, fs.zscore_ann_return, fs.zscore_sharpe,
               fs.calibration_slope, fs.mean_coverage]
    labels = copy(FORECAST_SUMMARY_LABELS)
    partial = isnothing(fs.spread_sharpe)
    if !partial
        S::MatNum = fs.spread_sharpe
        qs::AbstractVector = fs.quantiles
        for j in eachindex(qs)
            push!(vals, S[:, j])
            push!(labels, "Spread Sharpe q = $(qs[j])")
        end
    end
    K = length(fs.names)
    M = Matrix{Float64}(undef, length(vals), K)
    for r in eachindex(vals)
        v = vals[r]
        for k in 1:K
            M[r, k] = v[k]
        end
    end
    return M, labels, partial
end
function PortfolioOptimisers.plot_forecast_evaluation_summary(fs::PortfolioOptimisers.ForecastSummaryResult;
                                                              kwargs...)
    M, labels, partial = forecast_summary_columns(fs)
    K = size(M, 2)
    names::AbstractVector = fs.names
    title = if partial
        "Forecast Evaluation Summary (no quantile spread)"
    else
        "Forecast Evaluation Summary"
    end
    return groupedbar(M; bar_position = :dodge, xticks = (1:length(labels), labels),
                      label = reshape(string.(names), 1, K), xrotation = 30, title = title,
                      ylabel = "Value", legend = true, kwargs...)
end
function PortfolioOptimisers.plot_forecast_evaluation_summary(fes::AbstractVector{<:PortfolioOptimisers.ForecastEvaluationResult},
                                                              w::Option{<:MatNum} = nothing;
                                                              names = nothing,
                                                              bins::Integer = 10,
                                                              quantiles = nothing,
                                                              kwargs...)
    fs = PortfolioOptimisers.forecast_evaluation_summary(fes, w; names = names, bins = bins,
                                                         quantiles = quantiles)
    return PortfolioOptimisers.plot_forecast_evaluation_summary(fs; kwargs...)
end
function PortfolioOptimisers.plot_forecast_evaluation_summary(fes::AbstractVector{<:PortfolioOptimisers.ForecastEvaluationResult},
                                                              csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                              weighting = PortfolioOptimisers.IdentityMetric(),
                                                              names = nothing,
                                                              bins::Integer = 10,
                                                              quantiles = nothing,
                                                              kwargs...)
    fs = PortfolioOptimisers.forecast_evaluation_summary(fes, csfm; weighting = weighting,
                                                         names = names, bins = bins,
                                                         quantiles = quantiles)
    return PortfolioOptimisers.plot_forecast_evaluation_summary(fs; kwargs...)
end
function PortfolioOptimisers.plot_forecast_evaluation_summary(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                              w::Option{<:MatNum} = nothing;
                                                              kwargs...)
    return PortfolioOptimisers.plot_forecast_evaluation_summary([fe], w; kwargs...)
end
function PortfolioOptimisers.plot_forecast_evaluation_summary(fe::PortfolioOptimisers.ForecastEvaluationResult,
                                                              csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                              kwargs...)
    return PortfolioOptimisers.plot_forecast_evaluation_summary([fe], csfm; kwargs...)
end
