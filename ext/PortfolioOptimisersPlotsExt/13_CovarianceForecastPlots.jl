## ────────────────────────────────────────────────────────────────────────────
## The covariance forecast evaluation figures
## ────────────────────────────────────────────────────────────────────────────
# Each figure is the rolling form of `covariance_forecast_summary`. Point t reads the steps
# t - W + 1, …, t that scored, with the weights the summary gives them, so a window as wide
# as the run ends at the number of the summary. A step with no active asset adds nothing to
# a window, and a window with no scored step is `NaN` and drawn blank. The per-step
# statistics come from the library (`target_dof`, `covariance_diagonal_mean`,
# `covariance_exceedance`), so a figure computes the window and the band and nothing else.
const COVARIANCE_CALIBRATION_LABELS = (; mahalanobis = "Mahalanobis Ratio",
                                       diagonal = "Diagonal Ratio", bias = "Bias Statistic")
function covariance_plot_check_window(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult,
                                      window::Integer)
    M = length(cfer.dates)
    if !(1 <= window <= M)
        throw(DomainError(window, "window must be in 1:$(M), the steps of the evaluation"))
    end
    return nothing
end
function covariance_plot_check_diagnostics(diagnostics)
    if isempty(diagnostics) ||
       !all(d -> d isa Symbol && haskey(COVARIANCE_CALIBRATION_LABELS, d), diagnostics)
        throw(DomainError(diagnostics,
                          "diagnostics must name at least one of :mahalanobis, :diagonal and :bias, and nothing else"))
    end
    return nothing
end
function covariance_plot_check_levels(levels)
    if isempty(levels) || !all(q -> zero(q) < q < one(q), levels)
        throw(DomainError(levels, "levels must hold at least one level, each in (0, 1)"))
    end
    return nothing
end
function covariance_plot_check_evaluations(cfers::AbstractVector)
    if isempty(cfers)
        throw(PortfolioOptimisers.IsEmptyError("`cfers` cannot be empty"))
    end
    return nothing
end
# The weighted mean of the scored steps of each window. A ratio weights a step by its
# degrees of freedom, as the summary does, and a loss or a rate, whose `nu` is `nothing`,
# weights each step by one.
function covariance_step_weight(::Nothing, ::Integer)
    return 1
end
function covariance_step_weight(nu::AbstractVector{<:Integer}, j::Integer)
    return nu[j]
end
function covariance_window_mean(v::VecNum, nu::Option{<:AbstractVector{<:Integer}},
                                scored::AbstractVector{Bool}, r::AbstractUnitRange,
                                nan::Real)
    s = zero(nan)
    n = 0
    for j in r
        if !(scored[j])
            continue
        end
        nj = covariance_step_weight(nu, j)
        s += nj * v[j]
        n += nj
    end
    return iszero(n) ? nan : s / n
end
# The sample standard deviation of the scored steps of a window, which needs two.
function covariance_window_std(v::VecNum, scored::AbstractVector{Bool},
                               r::AbstractUnitRange, nan::Real)
    n = count(j -> scored[j], r)
    if n < 2
        return nan
    end
    mu = covariance_window_mean(v, nothing, scored, r, nan)
    ss = zero(nan)
    for j in r
        if scored[j]
            ss += abs2(v[j] - mu)
        end
    end
    return sqrt(ss / (n - 1))
end
# Point t of a rolling series reads the window of the steps t - W + 1, …, t. The first
# W - 1 points have no whole window, so they stay `NaN`.
function covariance_rolling(f, v::VecNum, window::Integer)
    nan = PortfolioOptimisers.float_if_integer(eltype(v))(NaN)
    out = fill(nan, length(v))
    for t in window:length(v)
        out[t] = f((t - window + 1):t, nan)
    end
    return out
end
function covariance_rolling_mean(v::VecNum, nu::Option{<:AbstractVector{<:Integer}},
                                 scored::AbstractVector{Bool}, window::Integer)
    return covariance_rolling((r, nan) -> covariance_window_mean(v, nu, scored, r, nan), v,
                              window)
end
function covariance_rolling_std(v::VecNum, scored::AbstractVector{Bool}, window::Integer)
    return covariance_rolling((r, nan) -> covariance_window_std(v, scored, r, nan), v,
                              window)
end
# One test portfolio is drawn as its own line. Several are drawn as their median, with a
# band from the fifth to the ninety-fifth percentile over the portfolios: the statistics
# the summary gives over its portfolios, read at each window.
function covariance_portfolio_line(f, V::MatNum)
    R = stack(f(view(V, :, k)) for k in axes(V, 2))
    if isone(size(R, 2))
        return R[:, 1], nothing
    end
    med = [Statistics.median(view(R, t, :)) for t in axes(R, 1)]
    lo = [PortfolioOptimisers.summary_quantile(view(R, t, :), 0.05) for t in axes(R, 1)]
    hi = [PortfolioOptimisers.summary_quantile(view(R, t, :), 0.95) for t in axes(R, 1)]
    return med, (lo, hi)
end
function covariance_calibration_line(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult,
                                     d::Symbol, window::Integer)
    scored = cfer.n_valid .> 0
    return if d === :mahalanobis
        nu = PortfolioOptimisers.target_dof.(Ref(cfer.target), cfer.n_valid, cfer.horizon)
        covariance_rolling_mean(cfer.mahalanobis_ratio, nu, scored, window), nothing
    elseif d === :diagonal
        nu = PortfolioOptimisers.target_step_dof.(Ref(cfer.target), cfer.horizon)
        covariance_rolling_mean(PortfolioOptimisers.covariance_diagonal_mean(cfer), nu,
                                scored, window), nothing
    else
        covariance_portfolio_line(v -> covariance_rolling_std(v, scored, window),
                                  cfer.standardised_return)
    end
end
function covariance_qlike_line(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult,
                               window::Integer)
    scored = cfer.n_valid .> 0
    return covariance_portfolio_line(v -> covariance_rolling_mean(v, nothing, scored,
                                                                  window),
                                     cfer.portfolio_qlike)
end
function covariance_exceedance_line(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult,
                                    q::Real, window::Integer)
    e = PortfolioOptimisers.covariance_exceedance(cfer, q)
    return covariance_rolling_mean(e, nothing, cfer.n_valid .> 0, window)
end
# A series of one evaluation is labelled by what it draws. A series of a comparison is
# labelled by its evaluation alone when each evaluation draws one series, and by both
# otherwise.
function covariance_series_label(::Nothing, label::AbstractString, ::Integer)
    return label
end
function covariance_series_label(name, label::AbstractString, n::Integer)
    return isone(n) ? string(name) : "$(name) - $(label)"
end
function covariance_plot_line!(plt, x, y, label, ::Nothing; kwargs...)
    return plot!(plt, x, y; label = label, linewidth = 2, kwargs...)
end
function covariance_plot_line!(plt, x, y, label, band::Tuple; kwargs...)
    lo, hi = band
    return plot!(plt, x, y; ribbon = (y .- lo, hi .- y), fillalpha = 0.15, label = label,
                 linewidth = 2, kwargs...)
end
function covariance_plot_lines(entries, title::AbstractString, ylabel::AbstractString;
                               kwargs...)
    plt = plot(; title = title, xlabel = "Observation", ylabel = ylabel, legend = true,
               kwargs...)
    for e in entries
        covariance_plot_line!(plt, e.x, e.y, e.label, e.band; kwargs...)
    end
    return plt
end
function covariance_calibration_entries(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult,
                                        name, diagnostics, window::Integer)
    covariance_plot_check_window(cfer, window)
    return map(collect(diagnostics)) do d
        y, band = covariance_calibration_line(cfer, d, window)
        label = covariance_series_label(name, COVARIANCE_CALIBRATION_LABELS[d],
                                        length(diagnostics))
        return (; x = cfer.dates, y = y, label = label, band = band)
    end
end
function covariance_qlike_entry(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult,
                                name, window::Integer)
    covariance_plot_check_window(cfer, window)
    y, band = covariance_qlike_line(cfer, window)
    return (; x = cfer.dates, y = y, label = covariance_series_label(name, "QLIKE", 1),
            band = band)
end
function covariance_exceedance_entries(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult,
                                       name, levels, window::Integer)
    covariance_plot_check_window(cfer, window)
    return map(collect(levels)) do q
        y = covariance_exceedance_line(cfer, q, window)
        label = covariance_series_label(name, "q = $(q)", length(levels))
        return (; x = cfer.dates, y = y, label = label, band = nothing)
    end
end
function covariance_calibration_title(window::Integer)
    return "Rolling Covariance Calibration (window=$(window))"
end
function covariance_qlike_title(window::Integer)
    return "Rolling Portfolio QLIKE Loss (window=$(window))"
end
function covariance_exceedance_title(window::Integer)
    return "Rolling Exceedance Rate (window=$(window))"
end
function PortfolioOptimisers.plot_covariance_calibration(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult;
                                                         window::Integer = 50,
                                                         diagnostics = (:mahalanobis,
                                                                        :diagonal, :bias),
                                                         kwargs...)
    covariance_plot_check_diagnostics(diagnostics)
    entries = covariance_calibration_entries(cfer, nothing, diagnostics, window)
    plt = covariance_plot_lines(entries, covariance_calibration_title(window),
                                "Calibration"; kwargs...)
    return idio_diagnostic_reference!(plt, 1.0)
end
function PortfolioOptimisers.plot_covariance_calibration(cfers::AbstractVector{<:PortfolioOptimisers.CovarianceForecastEvaluationResult};
                                                         names = nothing,
                                                         window::Integer = 50,
                                                         diagnostics = (:mahalanobis,
                                                                        :diagonal, :bias),
                                                         kwargs...)
    covariance_plot_check_evaluations(cfers)
    covariance_plot_check_diagnostics(diagnostics)
    nms = PortfolioOptimisers.covariance_forecast_names(names, length(cfers))
    entries = reduce(vcat,
                     [covariance_calibration_entries(cfers[k], nms[k], diagnostics, window)
                      for k in eachindex(cfers, nms)])
    plt = covariance_plot_lines(entries, covariance_calibration_title(window),
                                "Calibration"; kwargs...)
    return idio_diagnostic_reference!(plt, 1.0)
end
function PortfolioOptimisers.plot_covariance_qlike(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult;
                                                   window::Integer = 50, kwargs...)
    return covariance_plot_lines([covariance_qlike_entry(cfer, nothing, window)],
                                 covariance_qlike_title(window), "QLIKE Loss"; kwargs...)
end
function PortfolioOptimisers.plot_covariance_qlike(cfers::AbstractVector{<:PortfolioOptimisers.CovarianceForecastEvaluationResult};
                                                   names = nothing, window::Integer = 50,
                                                   kwargs...)
    covariance_plot_check_evaluations(cfers)
    nms = PortfolioOptimisers.covariance_forecast_names(names, length(cfers))
    entries = [covariance_qlike_entry(cfers[k], nms[k], window)
               for k in eachindex(cfers, nms)]
    return covariance_plot_lines(entries, covariance_qlike_title(window), "QLIKE Loss";
                                 kwargs...)
end
function PortfolioOptimisers.plot_covariance_exceedance(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult;
                                                        levels = (0.95, 0.99),
                                                        window::Integer = 50, kwargs...)
    covariance_plot_check_levels(levels)
    entries = covariance_exceedance_entries(cfer, nothing, levels, window)
    plt = covariance_plot_lines(entries, covariance_exceedance_title(window),
                                "Exceedance Rate"; kwargs...)
    # Each target rate is drawn in the colour of the series of its level.
    for (k, q) in enumerate(levels)
        hline!(plt, [1 - q]; label = "", linewidth = 1, linestyle = :dash,
               color = plt.series_list[k][:seriescolor])
    end
    return plt
end
function PortfolioOptimisers.plot_covariance_exceedance(cfers::AbstractVector{<:PortfolioOptimisers.CovarianceForecastEvaluationResult};
                                                        names = nothing, levels = (0.95,),
                                                        window::Integer = 50, kwargs...)
    covariance_plot_check_evaluations(cfers)
    covariance_plot_check_levels(levels)
    nms = PortfolioOptimisers.covariance_forecast_names(names, length(cfers))
    entries = reduce(vcat,
                     [covariance_exceedance_entries(cfers[k], nms[k], levels, window)
                      for k in eachindex(cfers, nms)])
    plt = covariance_plot_lines(entries, covariance_exceedance_title(window),
                                "Exceedance Rate"; kwargs...)
    # Several evaluations share a level, so each target rate is drawn once, in grey.
    for q in levels
        hline!(plt, [1 - q]; label = "", linewidth = 1, linestyle = :dash, color = :gray)
    end
    return plt
end
