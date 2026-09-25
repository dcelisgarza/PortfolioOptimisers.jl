## ────────────────────────────────────────────────────────────────────────────
## Cross-sectional regression diagnostics
## ────────────────────────────────────────────────────────────────────────────
# Each figure draws one level-2 verb and computes nothing of its own, which is the rule
# `plot_risk_contribution` follows. The prior-result entry points guard through
# `assert_prior_regression` first: `rr` may be absent, and the guard's message names the
# remedy, where reading `pr.rr` straight away would raise on `nothing` with no diagnosis.
# A block that is a time-series `Regression` rather than a `CrossSectionalFactorModel`
# reaches the level-2 verb and raises a `MethodError` naming that type, which is the
# honest report: these diagnostics read a point-in-time exposure history that a
# time-series block does not carry.
const NO_CS_DIAGNOSTIC_LEAD = "a cross-sectional regression diagnostic reads the factor block `rr`. $NO_FACTOR_BLOCK_HINT Pass the block directly as the first argument if you hold it."
function cs_diagnostic_block(pr::PortfolioOptimisers.AbstractPriorResult)
    PortfolioOptimisers.assert_prior_regression(pr, :pr; lead = NO_CS_DIAGNOSTIC_LEAD)
    return pr.rr
end
function cs_diagnostic_labels(csfm, nf::Option{<:AbstractVector}, K::Integer)
    nf_use = isnothing(nf) ? PortfolioOptimisers.cs_diagnostic_factor_names(csfm) : nf
    return isnothing(nf_use) ? string.(1:K) : string.(nf_use)
end
function cs_diagnostic_series(vals::MatNum, labels::AbstractVector, title::AbstractString,
                              ylabel::AbstractString; kwargs...)
    plt = plot(view(vals, :, 1); title = title, xlabel = "Observation", ylabel = ylabel,
               label = labels[1], legend = true, linewidth = 2, kwargs...)
    for k in 2:size(vals, 2)
        plot!(plt, view(vals, :, k); label = labels[k], linewidth = 2, kwargs...)
    end
    return plt
end
function PortfolioOptimisers.plot_cs_regression_r2(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                   kwargs...)
    return plot(PortfolioOptimisers.cs_regression_r2(csfm);
                title = "Cross-Sectional Regression R²", xlabel = "Observation",
                ylabel = "R²", legend = false, linewidth = 2, kwargs...)
end
function PortfolioOptimisers.plot_cs_regression_r2(pr::PortfolioOptimisers.AbstractPriorResult;
                                                   kwargs...)
    return PortfolioOptimisers.plot_cs_regression_r2(cs_diagnostic_block(pr); kwargs...)
end
function PortfolioOptimisers.plot_cs_regression_adjusted_r2(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                            kwargs...)
    return plot(PortfolioOptimisers.cs_regression_adjusted_r2(csfm);
                title = "Cross-Sectional Regression Adjusted R²", xlabel = "Observation",
                ylabel = "Adjusted R²", legend = false, linewidth = 2, kwargs...)
end
function PortfolioOptimisers.plot_cs_regression_adjusted_r2(pr::PortfolioOptimisers.AbstractPriorResult;
                                                            kwargs...)
    return PortfolioOptimisers.plot_cs_regression_adjusted_r2(cs_diagnostic_block(pr);
                                                              kwargs...)
end
function PortfolioOptimisers.plot_cs_regression_aic(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                    kwargs...)
    return plot(PortfolioOptimisers.cs_regression_aic(csfm);
                title = "Cross-Sectional Regression AIC", xlabel = "Observation",
                ylabel = "AIC", legend = false, linewidth = 2, kwargs...)
end
function PortfolioOptimisers.plot_cs_regression_aic(pr::PortfolioOptimisers.AbstractPriorResult;
                                                    kwargs...)
    return PortfolioOptimisers.plot_cs_regression_aic(cs_diagnostic_block(pr); kwargs...)
end
function PortfolioOptimisers.plot_cs_regression_bic(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                    kwargs...)
    return plot(PortfolioOptimisers.cs_regression_bic(csfm);
                title = "Cross-Sectional Regression BIC", xlabel = "Observation",
                ylabel = "BIC", legend = false, linewidth = 2, kwargs...)
end
function PortfolioOptimisers.plot_cs_regression_bic(pr::PortfolioOptimisers.AbstractPriorResult;
                                                    kwargs...)
    return PortfolioOptimisers.plot_cs_regression_bic(cs_diagnostic_block(pr); kwargs...)
end
function PortfolioOptimisers.plot_cs_regression_t_stats(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                        nf::Option{<:AbstractVector} = nothing,
                                                        kwargs...)
    t = PortfolioOptimisers.cs_regression_t_stats(csfm)
    labels = cs_diagnostic_labels(csfm, nf, size(t, 2))
    return cs_diagnostic_series(t, labels, "Cross-Sectional Regression t-Statistics", "t";
                                kwargs...)
end
function PortfolioOptimisers.plot_cs_regression_t_stats(pr::PortfolioOptimisers.AbstractPriorResult;
                                                        nf::Option{<:AbstractVector} = nothing,
                                                        kwargs...)
    return PortfolioOptimisers.plot_cs_regression_t_stats(cs_diagnostic_block(pr); nf = nf,
                                                          kwargs...)
end
function PortfolioOptimisers.plot_cs_regression_t_stat_exceedance_rate(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                                       nf::Option{<:AbstractVector} = nothing,
                                                                       threshold::Number = 2,
                                                                       kwargs...)
    rate = PortfolioOptimisers.cs_regression_t_stat_exceedance_rate(csfm;
                                                                    threshold = threshold)
    labels = cs_diagnostic_labels(csfm, nf, length(rate))
    K = length(rate)
    plt = bar(rate; xticks = (1:K, labels),
              title = "t-Statistic Exceedance Rate (|t| > $threshold)", xlabel = "Factor",
              ylabel = "Fraction of observations", xrotation = 90, legend = false,
              kwargs...)
    # The rate a t-statistic of no explanatory power reaches: |t| > c under a standard
    # normal law.
    hline!(plt, [2 * ccdf(Normal(), threshold)]; label = "", linewidth = 2, color = :red,
           linestyle = :dash)
    return plt
end
function PortfolioOptimisers.plot_cs_regression_t_stat_exceedance_rate(pr::PortfolioOptimisers.AbstractPriorResult;
                                                                       nf::Option{<:AbstractVector} = nothing,
                                                                       threshold::Number = 2,
                                                                       kwargs...)
    return PortfolioOptimisers.plot_cs_regression_t_stat_exceedance_rate(cs_diagnostic_block(pr);
                                                                         nf = nf,
                                                                         threshold = threshold,
                                                                         kwargs...)
end
function PortfolioOptimisers.plot_exposure_vif(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                               nf::Option{<:AbstractVector} = nothing,
                                               kwargs...)
    vif = PortfolioOptimisers.exposure_vif(csfm)
    labels = cs_diagnostic_labels(csfm, nf, size(vif, 2))
    plt = cs_diagnostic_series(vif, labels, "Exposure Variance Inflation Factors", "VIF";
                               kwargs...)
    hline!(plt, [1.0]; label = "", linewidth = 2, color = :red, linestyle = :dash)
    return plt
end
function PortfolioOptimisers.plot_exposure_vif(pr::PortfolioOptimisers.AbstractPriorResult;
                                               nf::Option{<:AbstractVector} = nothing,
                                               kwargs...)
    return PortfolioOptimisers.plot_exposure_vif(cs_diagnostic_block(pr); nf = nf,
                                                 kwargs...)
end
function PortfolioOptimisers.plot_exposure_condition_number(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                            kwargs...)
    return plot(PortfolioOptimisers.exposure_condition_number(csfm);
                title = "Exposure Condition Number", xlabel = "Observation", ylabel = "κ",
                yscale = :log10, legend = false, linewidth = 2, kwargs...)
end
function PortfolioOptimisers.plot_exposure_condition_number(pr::PortfolioOptimisers.AbstractPriorResult;
                                                            kwargs...)
    return PortfolioOptimisers.plot_exposure_condition_number(cs_diagnostic_block(pr);
                                                              kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Cross-sectional exposure diagnostics
## ────────────────────────────────────────────────────────────────────────────
# The exposure group answers on the RAW factor axis, because it reads the exposure
# history as the panel wrote it and never the design of the fit, so these figures label
# their series off `csfm.nf` and not off `cs_diagnostic_factor_names`. The one exception
# is the cumulative information coefficient under `reduced`, which does map the exposures
# through the family re-basis and is then labelled on the reduced axis.
function exposure_diagnostic_labels(csfm, nf::Option{<:AbstractVector}, K::Integer)
    nf_use = isnothing(nf) ? csfm.nf : nf
    return isnothing(nf_use) ? string.(1:K) : string.(nf_use)
end
function PortfolioOptimisers.plot_exposure_correlation(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                       nf::Option{<:AbstractVector} = nothing,
                                                       weighting = PortfolioOptimisers.BenchmarkWeightMetric(),
                                                       kwargs...)
    C = PortfolioOptimisers.exposure_correlation(csfm; weighting = weighting)
    labels = exposure_diagnostic_labels(csfm, nf, size(C, 1))
    K = size(C, 1)
    return heatmap(C; xticks = (1:K, labels), yticks = (1:K, labels), xrotation = 90,
                   clim = (-1.0, 1.0), color = cgrad(:Spectral), yflip = true,
                   title = "Time-Average Exposure Correlation", colorbar_title = "ρ",
                   kwargs...)
end
function PortfolioOptimisers.plot_exposure_correlation(pr::PortfolioOptimisers.AbstractPriorResult;
                                                       nf::Option{<:AbstractVector} = nothing,
                                                       weighting = PortfolioOptimisers.BenchmarkWeightMetric(),
                                                       kwargs...)
    return PortfolioOptimisers.plot_exposure_correlation(cs_diagnostic_block(pr); nf = nf,
                                                         weighting = weighting, kwargs...)
end
function PortfolioOptimisers.plot_cumulative_exposure_ic(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                         nf::Option{<:AbstractVector} = nothing,
                                                         rank::Bool = true,
                                                         reduced::Bool = false, kwargs...)
    ic = PortfolioOptimisers.exposure_ic(csfm; horizon = 1, rank = rank, reduced = reduced)
    cum = cumulative_exposure_ic(ic)
    labels = if reduced
        cs_diagnostic_labels(csfm, nf, size(ic, 2))
    else
        exposure_diagnostic_labels(csfm, nf, size(ic, 2))
    end
    method = rank ? "Spearman" : "Pearson"
    return cs_diagnostic_series(cum, labels, "Cumulative Exposure IC ($method)",
                                "Cumulative IC"; kwargs...)
end
function PortfolioOptimisers.plot_cumulative_exposure_ic(pr::PortfolioOptimisers.AbstractPriorResult;
                                                         nf::Option{<:AbstractVector} = nothing,
                                                         rank::Bool = true,
                                                         reduced::Bool = false, kwargs...)
    return PortfolioOptimisers.plot_cumulative_exposure_ic(cs_diagnostic_block(pr); nf = nf,
                                                           rank = rank, reduced = reduced,
                                                           kwargs...)
end
# An observation whose information coefficient is not defined contributes nothing to the
# running sum, so one missing cross-section breaks no series.
function cumulative_exposure_ic(ic::MatNum)
    P, K = size(ic)
    cum = similar(ic)
    for k in 1:K
        s = zero(eltype(ic))
        for t in 1:P
            v = ic[t, k]
            if isfinite(v)
                s += v
            end
            cum[t, k] = s
        end
    end
    return cum
end
function PortfolioOptimisers.plot_exposure_distribution(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                        factor::Integer = 1,
                                                        observation::Option{<:Integer} = nothing,
                                                        nf::Option{<:AbstractVector} = nothing,
                                                        kwargs...)
    Ms = PortfolioOptimisers.cs_diagnostic_exposures(csfm)
    labels = exposure_diagnostic_labels(csfm, nf, size(Ms, 3))
    slice = if isnothing(observation)
        vec(view(Ms, :, :, factor))
    else
        vec(view(Ms, observation, :, factor))
    end
    values = filter(isfinite, slice)
    span = isnothing(observation) ? "all observations" : "observation $observation"
    return histogram(values; title = "Exposure Distribution: $(labels[factor]) ($span)",
                     xlabel = "Exposure", ylabel = "Count", legend = false, kwargs...)
end
function PortfolioOptimisers.plot_exposure_distribution(pr::PortfolioOptimisers.AbstractPriorResult;
                                                        factor::Integer = 1,
                                                        observation::Option{<:Integer} = nothing,
                                                        nf::Option{<:AbstractVector} = nothing,
                                                        kwargs...)
    return PortfolioOptimisers.plot_exposure_distribution(cs_diagnostic_block(pr);
                                                          factor = factor,
                                                          observation = observation,
                                                          nf = nf, kwargs...)
end
function PortfolioOptimisers.plot_exposure_dispersion(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                      nf::Option{<:AbstractVector} = nothing,
                                                      weighting = PortfolioOptimisers.BenchmarkWeightMetric(),
                                                      kwargs...)
    D = PortfolioOptimisers.exposure_dispersion(csfm; weighting = weighting)
    labels = exposure_diagnostic_labels(csfm, nf, size(D, 2))
    plt = cs_diagnostic_series(D, labels, "Exposure Cross-Sectional Std", "Std"; kwargs...)
    return plt
end
function PortfolioOptimisers.plot_exposure_dispersion(pr::PortfolioOptimisers.AbstractPriorResult;
                                                      nf::Option{<:AbstractVector} = nothing,
                                                      weighting = PortfolioOptimisers.BenchmarkWeightMetric(),
                                                      kwargs...)
    return PortfolioOptimisers.plot_exposure_dispersion(cs_diagnostic_block(pr); nf = nf,
                                                        weighting = weighting, kwargs...)
end
function PortfolioOptimisers.plot_exposure_stability(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                     nf::Option{<:AbstractVector} = nothing,
                                                     step::Integer = 21,
                                                     weighting = PortfolioOptimisers.BenchmarkWeightMetric(),
                                                     kwargs...)
    S = PortfolioOptimisers.exposure_stability(csfm; step = step, weighting = weighting)
    labels = exposure_diagnostic_labels(csfm, nf, size(S, 2))
    plt = cs_diagnostic_series(S, labels, "Exposure Stability (step=$step)", "ρ"; kwargs...)
    hline!(plt, [1.0]; label = "", linewidth = 2, color = :red, linestyle = :dash)
    return plt
end
function PortfolioOptimisers.plot_exposure_stability(pr::PortfolioOptimisers.AbstractPriorResult;
                                                     nf::Option{<:AbstractVector} = nothing,
                                                     step::Integer = 21,
                                                     weighting = PortfolioOptimisers.BenchmarkWeightMetric(),
                                                     kwargs...)
    return PortfolioOptimisers.plot_exposure_stability(cs_diagnostic_block(pr); nf = nf,
                                                       step = step, weighting = weighting,
                                                       kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## The factor model summary and the factor forecast figures
## ────────────────────────────────────────────────────────────────────────────
# The summary answers on the RAW factor axis, so its figure labels its series off
# `csfm.nf`. The two forecast figures read `fpr.sigma`, whose axis is the factor axis of
# the factor prior, so they label off the names the caller passes and fall back to the
# position of the factor.
const FACTOR_SUMMARY_LABELS = ["Ann. Return", "Ann. Vol", "Sharpe", "Autocorr", "Mean |t|",
                               "t Rate", "Mean VIF", "Stability", "Coverage"]
function factor_summary_columns(fs::PortfolioOptimisers.FactorSummaryResult)
    vals = Any[fs.ann_return, fs.ann_volatility, fs.sharpe, fs.autocorr, fs.mean_abs_t,
               fs.t_rate, fs.mean_vif, fs.stability, fs.coverage]
    keep = [i for i in eachindex(vals) if !isnothing(vals[i])]
    K = length(fs.ann_return)
    M = Matrix{Float64}(undef, length(keep), K)
    for (r, i) in enumerate(keep)
        v = vals[i]
        for k in 1:K
            M[r, k] = v[k]
        end
    end
    return M, FACTOR_SUMMARY_LABELS[keep], length(keep) < length(vals)
end
function PortfolioOptimisers.plot_factor_model_summary(fs::PortfolioOptimisers.FactorSummaryResult;
                                                       nf::Option{<:AbstractVector} = nothing,
                                                       kwargs...)
    M, labels, partial = factor_summary_columns(fs)
    K = size(M, 2)
    series = isnothing(nf) ? string.(1:K) : string.(nf)
    title = partial ? "Factor Model Summary (no exposure history)" : "Factor Model Summary"
    return groupedbar(M; bar_position = :dodge, xticks = (1:length(labels), labels),
                      label = reshape(series, 1, K), xrotation = 30, title = title,
                      ylabel = "Value", legend = true, kwargs...)
end
function PortfolioOptimisers.plot_factor_model_summary(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                       nf::Option{<:AbstractVector} = nothing,
                                                       ppy::Number = 1,
                                                       threshold::Number = 2,
                                                       step::Integer = 21,
                                                       weighting = PortfolioOptimisers.BenchmarkWeightMetric(),
                                                       coverage_weighting = PortfolioOptimisers.RegressionWeightMetric(),
                                                       kwargs...)
    fs = PortfolioOptimisers.factor_model_summary(csfm; ppy = ppy, threshold = threshold,
                                                  step = step, weighting = weighting,
                                                  coverage_weighting = coverage_weighting)
    labels = exposure_diagnostic_labels(csfm, nf, length(fs.ann_return))
    return PortfolioOptimisers.plot_factor_model_summary(fs; nf = labels, kwargs...)
end
function PortfolioOptimisers.plot_factor_model_summary(pr::PortfolioOptimisers.AbstractPriorResult;
                                                       nf::Option{<:AbstractVector} = nothing,
                                                       kwargs...)
    return PortfolioOptimisers.plot_factor_model_summary(cs_diagnostic_block(pr); nf = nf,
                                                         kwargs...)
end
const NO_FACTOR_FORECAST_LEAD = "a factor forecast figure draws the factor covariance `fpr.sigma`. $NO_FACTOR_BLOCK_HINT Pass the factor covariance directly as the first argument if you hold it."
function PortfolioOptimisers.plot_factor_forecast_correlation(f_sigma::MatNum,
                                                              nf::AbstractVector = 1:size(f_sigma,
                                                                                          1);
                                                              kwargs...)
    # Copy before rescaling: `cov2cor!` mutates in place, and `f_sigma` is the caller's.
    C = Matrix{real(eltype(f_sigma))}(f_sigma)
    StatsBase.cov2cor!(C, sqrt.(diag(C)))
    K = size(C, 1)
    labels = string.(nf)
    return heatmap(C; xticks = (1:K, labels), yticks = (1:K, labels), xrotation = 90,
                   clim = (-1.0, 1.0), color = cgrad(:Spectral), yflip = true,
                   title = "Factor Forecast Correlation", colorbar_title = "ρ", kwargs...)
end
function PortfolioOptimisers.plot_factor_forecast_correlation(pr::PortfolioOptimisers.AbstractPriorResult,
                                                              nf::Option{<:AbstractVector} = nothing;
                                                              kwargs...)
    PortfolioOptimisers.assert_prior_regression(pr, :pr; lead = NO_FACTOR_FORECAST_LEAD)
    nf_use = isnothing(nf) ? (1:size(factor_plot_prior(pr).sigma, 1)) : nf
    return PortfolioOptimisers.plot_factor_forecast_correlation(factor_plot_prior(pr).sigma,
                                                                nf_use; kwargs...)
end
function PortfolioOptimisers.plot_factor_forecast_volatilities(f_sigma::MatNum,
                                                               nf::AbstractVector = 1:size(f_sigma,
                                                                                           1);
                                                               ppy::Number = 1, kwargs...)
    vol = sqrt.(diag(f_sigma) .* ppy)
    order = sortperm(vol)
    labels = string.(nf)[order]
    K = length(vol)
    return bar(vol[order]; yticks = (1:K, labels), orientation = :h,
               title = "Factor Forecast Volatility", xlabel = "Volatility",
               ylabel = "Factor", legend = false, kwargs...)
end
function PortfolioOptimisers.plot_factor_forecast_volatilities(pr::PortfolioOptimisers.AbstractPriorResult,
                                                               nf::Option{<:AbstractVector} = nothing;
                                                               ppy::Number = 1, kwargs...)
    PortfolioOptimisers.assert_prior_regression(pr, :pr; lead = NO_FACTOR_FORECAST_LEAD)
    nf_use = isnothing(nf) ? (1:size(factor_plot_prior(pr).sigma, 1)) : nf
    return PortfolioOptimisers.plot_factor_forecast_volatilities(factor_plot_prior(pr).sigma,
                                                                 nf_use; ppy = ppy,
                                                                 kwargs...)
end
function PortfolioOptimisers.plot_factor_cumulative_returns(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                            nf::Option{<:AbstractVector} = nothing,
                                                            compound::Bool = false,
                                                            kwargs...)
    f = PortfolioOptimisers.factor_summary_returns(csfm)
    # An observation whose factor return is not finite contributes nothing to the running
    # sum, so one absent cross-section breaks no series.
    g = [isfinite(x) ? x : zero(x) for x in f]
    cum = cumulative_returns(g, compound)
    labels = exposure_diagnostic_labels(csfm, nf, size(cum, 2))
    kind = compound ? "Compounded" : "Uncompounded"
    return cs_diagnostic_series(cum, labels, "Factor Cumulative Returns ($kind)",
                                "Cumulative Return"; kwargs...)
end
function PortfolioOptimisers.plot_factor_cumulative_returns(pr::PortfolioOptimisers.AbstractPriorResult;
                                                            nf::Option{<:AbstractVector} = nothing,
                                                            compound::Bool = false,
                                                            kwargs...)
    return PortfolioOptimisers.plot_factor_cumulative_returns(cs_diagnostic_block(pr);
                                                              nf = nf, compound = compound,
                                                              kwargs...)
end

## ────────────────────────────────────────────────────────────────────────────
## Cross-sectional idiosyncratic diagnostics
## ────────────────────────────────────────────────────────────────────────────
# The idiosyncratic group answers on the asset axis and aggregates it away, so every figure
# here draws one series and none of them labels a factor. Five carry the reference the
# normal law gives the series, and the information coefficient carries none because the
# normal law says nothing about it.
function idio_diagnostic_series(vals::VecNum, title::AbstractString, ylabel::AbstractString;
                                offset::Integer = 0, kwargs...)
    x = (1 + offset):(length(vals) + offset)
    return plot(x, vals; title = title, xlabel = "Observation", ylabel = ylabel,
                legend = false, linewidth = 2, kwargs...)
end
function idio_diagnostic_reference!(plt, value::Real)
    hline!(plt, [value]; label = "", linewidth = 2, color = :red, linestyle = :dash)
    return plt
end
function PortfolioOptimisers.plot_idio_calibration(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                   kwargs...)
    plt = idio_diagnostic_series(PortfolioOptimisers.idio_calibration(csfm),
                                 "Idiosyncratic Calibration",
                                 "Cross-Sectional Std of Standardised Idio Returns";
                                 kwargs...)
    return idio_diagnostic_reference!(plt, 1.0)
end
function PortfolioOptimisers.plot_idio_calibration(pr::PortfolioOptimisers.AbstractPriorResult;
                                                   kwargs...)
    return PortfolioOptimisers.plot_idio_calibration(cs_diagnostic_block(pr); kwargs...)
end
function PortfolioOptimisers.plot_idio_tail_rate(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                 threshold::Real = 3, kwargs...)
    plt = idio_diagnostic_series(PortfolioOptimisers.idio_tail_rate(csfm;
                                                                    threshold = threshold),
                                 "Idiosyncratic Tail Rate (threshold=$threshold)",
                                 "Fraction of Assets"; kwargs...)
    return idio_diagnostic_reference!(plt, 2 * ccdf(Normal(), threshold))
end
function PortfolioOptimisers.plot_idio_tail_rate(pr::PortfolioOptimisers.AbstractPriorResult;
                                                 threshold::Real = 3, kwargs...)
    return PortfolioOptimisers.plot_idio_tail_rate(cs_diagnostic_block(pr);
                                                   threshold = threshold, kwargs...)
end
function PortfolioOptimisers.plot_idio_kurtosis(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                kwargs...)
    plt = idio_diagnostic_series(PortfolioOptimisers.idio_kurtosis(csfm),
                                 "Cross-Sectional Excess Kurtosis", "Excess Kurtosis";
                                 kwargs...)
    return idio_diagnostic_reference!(plt, 0.0)
end
function PortfolioOptimisers.plot_idio_kurtosis(pr::PortfolioOptimisers.AbstractPriorResult;
                                                kwargs...)
    return PortfolioOptimisers.plot_idio_kurtosis(cs_diagnostic_block(pr); kwargs...)
end
function PortfolioOptimisers.plot_idio_skewness(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                kwargs...)
    plt = idio_diagnostic_series(PortfolioOptimisers.idio_skewness(csfm),
                                 "Cross-Sectional Skewness", "Skewness"; kwargs...)
    return idio_diagnostic_reference!(plt, 0.0)
end
function PortfolioOptimisers.plot_idio_skewness(pr::PortfolioOptimisers.AbstractPriorResult;
                                                kwargs...)
    return PortfolioOptimisers.plot_idio_skewness(cs_diagnostic_block(pr); kwargs...)
end
function PortfolioOptimisers.plot_idio_vol_ic(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                              kwargs...)
    return idio_diagnostic_series(PortfolioOptimisers.idio_vol_ic(csfm),
                                  "Idiosyncratic Volatility IC (Spearman)",
                                  "Rank Correlation"; offset = 1, kwargs...)
end
function PortfolioOptimisers.plot_idio_vol_ic(pr::PortfolioOptimisers.AbstractPriorResult;
                                              kwargs...)
    return PortfolioOptimisers.plot_idio_vol_ic(cs_diagnostic_block(pr); kwargs...)
end
function PortfolioOptimisers.plot_idio_vol_residual_dependence(csfm::PortfolioOptimisers.CrossSectionalFactorModel;
                                                               kwargs...)
    plt = idio_diagnostic_series(PortfolioOptimisers.idio_vol_residual_dependence(csfm),
                                 "Idiosyncratic Volatility Residual Dependence (Spearman)",
                                 "Rank Correlation"; offset = 1, kwargs...)
    return idio_diagnostic_reference!(plt, 0.0)
end
function PortfolioOptimisers.plot_idio_vol_residual_dependence(pr::PortfolioOptimisers.AbstractPriorResult;
                                                               kwargs...)
    return PortfolioOptimisers.plot_idio_vol_residual_dependence(cs_diagnostic_block(pr);
                                                                 kwargs...)
end
