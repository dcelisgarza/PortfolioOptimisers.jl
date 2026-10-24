## ────────────────────────────────────────────────────────────────────────────
## The covariance forecast evaluation figures
## ────────────────────────────────────────────────────────────────────────────
# Each figure is the rolling form of `covariance_forecast_summary`. Point t reads the scored
# steps of its window, with the weights that the summary gives them. So a window as wide as
# the run ends at the value of the summary. A step with no active asset adds nothing to a
# window. A window with no scored step is `NaN`, and the figure leaves it blank. The library
# gives the per-step statistics and their weights through `target_dof`, `target_step_dof`,
# `covariance_step_weights`, `covariance_diagonal_mean` and `covariance_exceedance`. This
# file computes only the windows, the mean or the standard deviation of each window, and the
# band.
"""
    COVARIANCE_CALIBRATION_LABELS

Legend label of each diagnostic of `plot_covariance_calibration`, keyed by the symbol that `diagnostics` takes.

[`plot_covariance_calibration`](@ref) reads the label of each series from this tuple. `covariance_plot_check_diagnostics` reads its keys as the diagnostics that the figure knows.

# Related

  - [`plot_covariance_calibration`](@ref)
  - `covariance_plot_check_diagnostics`
"""
const COVARIANCE_CALIBRATION_LABELS = (; mahalanobis = "Mahalanobis Ratio",
                                       diagonal = "Diagonal Ratio", bias = "Bias Statistic")
"""
    covariance_windows(scored::AbstractVector{Bool}, window::Integer, scored_steps::Bool)

Return the window of each point of a rolling series, as a pair of the point and its range of steps.

Point `t` reads the scored steps in its range `r`. Under `scored_steps = false`, the range of point `t` is `(t - window + 1):t`, so the first `window - 1` points have no window. Under `scored_steps = true`, the range spans the last `window` scored steps up to `t`. So it spans a step with no active asset, and only a scored step has a point.

# Arguments

  - `scored`: Whether each step of the evaluation has at least one active asset.
  - `window`: Number of steps in the window, or number of scored steps under `scored_steps = true`.
  - `scored_steps`: Whether the window counts the scored steps alone.

# Returns

  - `wins::Vector{<:Pair}`: One pair `t => r` for each point that has a window, in the order of the steps.

# Related

  - [`plot_covariance_calibration`](@ref)
  - `covariance_plot_windows`
  - `covariance_rolling`
"""
function covariance_windows(scored::AbstractVector{Bool}, window::Integer,
                            scored_steps::Bool)
    if !scored_steps
        return [t => ((t - window + 1):t) for t in window:length(scored)]
    end
    s = findall(scored)
    return [s[j] => (s[j - window + 1]:s[j]) for j in window:length(s)]
end
"""
    covariance_plot_windows(cfer::CovarianceForecastEvaluationResult, wo::NamedTuple)

Check the window keywords of a figure, and return the windows of an evaluation with the first point that its series draws.

The first point is the first step under `whole_windows = false`. Under `whole_windows = true`, it is the point of the first whole window.

# Arguments

  - `cfer`: The evaluation to draw, from [`covariance_forecast_evaluation`](@ref).
  - `wo`: The three window keywords of a figure, `window`, `scored_steps` and `whole_windows`.

# Validation

  - The evaluation scored at least one step, as `covariance_plot_check_scored` checks.
  - `wo.window` lies in `1:M`. `M` is the number of steps, or the number of scored steps under `scored_steps = true`. The function throws a `DomainError` otherwise.

# Returns

  - `rw::NamedTuple`: The windows `wins` from `covariance_windows`, and the index `from` of the first point to draw.

# Related

  - [`plot_covariance_calibration`](@ref)
  - `covariance_windows`
  - `covariance_plot_entry`
"""
function covariance_plot_windows(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult,
                                 wo::NamedTuple)
    scored = cfer.n_valid .> 0
    covariance_plot_check_scored(scored)
    M = wo.scored_steps ? count(scored) : length(scored)
    if !(1 <= wo.window <= M)
        throw(DomainError(wo.window,
                          "window must be in 1:$(M), the steps that the window counts (the scored steps under `scored_steps = true`)"))
    end
    wins = covariance_windows(scored, wo.window, wo.scored_steps)
    return (; wins = wins, from = wo.whole_windows ? first(first(wins)) : 1)
end
"""
    covariance_plot_check_diagnostics(diagnostics)

Check the `diagnostics` keyword of `plot_covariance_calibration`.

# Arguments

  - `diagnostics`: The diagnostics to draw.

# Validation

  - `diagnostics` is not empty, and each entry is a key of `COVARIANCE_CALIBRATION_LABELS`, which are `:mahalanobis`, `:diagonal` and `:bias`. The function throws a `DomainError` otherwise.

# Returns

  - `nothing`.

# Related

  - [`plot_covariance_calibration`](@ref)
  - `COVARIANCE_CALIBRATION_LABELS`
"""
function covariance_plot_check_diagnostics(diagnostics)
    if isempty(diagnostics) ||
       !all(d -> d isa Symbol && haskey(COVARIANCE_CALIBRATION_LABELS, d), diagnostics)
        throw(DomainError(diagnostics,
                          "diagnostics must name at least one of :mahalanobis, :diagonal and :bias, and nothing else"))
    end
    return nothing
end
"""
    covariance_plot_check_levels(levels)

Check the `levels` keyword of `plot_covariance_exceedance`.

# Arguments

  - `levels`: The confidence levels of the chi-squared quantiles.

# Validation

  - `levels` is not empty, and each level lies in `(0, 1)`. The function throws a `DomainError` otherwise.

# Returns

  - `nothing`.

# Related

  - [`plot_covariance_exceedance`](@ref)
"""
function covariance_plot_check_levels(levels)
    if isempty(levels) || !all(q -> zero(q) < q < one(q), levels)
        throw(DomainError(levels, "levels must hold at least one level, each in (0, 1)"))
    end
    return nothing
end
"""
    covariance_plot_check_scored(scored::AbstractVector{Bool})

Refuse an evaluation with no scored step, because it has nothing to draw.

[`covariance_forecast_summary`](@ref) refuses such an evaluation in the same way, under both settings of `scored_steps`.

# Arguments

  - `scored`: Whether each step of the evaluation has at least one active asset.

# Validation

  - At least one entry of `scored` is `true`. The function throws an `IsEmptyError` otherwise.

# Returns

  - `nothing`.

# Related

  - [`covariance_forecast_summary`](@ref)
  - `covariance_plot_windows`
"""
function covariance_plot_check_scored(scored::AbstractVector{Bool})
    if !any(scored)
        throw(PortfolioOptimisers.IsEmptyError("the evaluation scored no step: every step had no active asset, so there is nothing to draw."))
    end
    return nothing
end
"""
    covariance_plot_check_no_step_weighting(kwargs, figure::AbstractString)

Refuse a `step_weighting` keyword in a figure that does not read it.

Only [`plot_covariance_calibration`](@ref) reads a step weighting. The other figures pass `kwargs` to the plotting backend, which drops a keyword that it does not know. So these figures refuse the keyword, and the caller gets an error in place of a figure that ignores it.

# Arguments

  - `kwargs`: The keyword arguments of the figure.
  - `figure`: The name of the figure, for the message of the error.

# Validation

  - `kwargs` holds no `step_weighting`. The function throws an `ArgumentError` otherwise.

# Returns

  - `nothing`.

# Related

  - [`plot_covariance_qlike`](@ref)
  - [`plot_covariance_exceedance`](@ref)
"""
function covariance_plot_check_no_step_weighting(kwargs, figure::AbstractString)
    if haskey(kwargs, :step_weighting)
        throw(ArgumentError("$(figure) takes no `step_weighting`: a loss and a rate weight each scored step by one. Only `plot_covariance_calibration` reads it."))
    end
    return nothing
end
"""
    covariance_plot_check_evaluations(cfers::AbstractVector)

Check that the comparison method of a figure has at least one evaluation to draw.

# Arguments

  - `cfers`: The evaluations to overlay.

# Validation

  - `cfers` is not empty. The function throws an `IsEmptyError` otherwise.

# Returns

  - `nothing`.

# Related

  - [`plot_covariance_calibration`](@ref)
  - [`plot_covariance_qlike`](@ref)
  - [`plot_covariance_exceedance`](@ref)
"""
function covariance_plot_check_evaluations(cfers::AbstractVector)
    if isempty(cfers)
        throw(PortfolioOptimisers.IsEmptyError("`cfers` cannot be empty"))
    end
    return nothing
end
"""
    covariance_step_weight(omega::Nothing, j::Integer)
    covariance_step_weight(omega::AbstractVector{<:Real}, j::Integer)

Return the weight of step `j` in the mean of a window.

A ratio passes the weights of [`covariance_step_weights`](@ref), as the summary does, and step `j` takes `omega[j]`. A loss or a rate passes `nothing`, and each step takes the weight one.

# Arguments

  - `omega`: The weight of each step, or `nothing`.
  - `j`: The index of the step.

# Returns

  - `w::Real`: The weight of step `j`.

# Related

  - [`covariance_step_weights`](@ref)
  - `covariance_window_mean`
"""
function covariance_step_weight(::Nothing, ::Integer)
    return 1
end
function covariance_step_weight(omega::AbstractVector{<:Real}, j::Integer)
    return omega[j]
end
"""
    covariance_window_mean(v::VecNum, omega::Option{<:AbstractVector{<:Real}},
                           scored::AbstractVector{Bool}, r::AbstractUnitRange, nan::Real)

Return the weighted mean of a per-step statistic over the scored steps of one window.

The mean skips each step of `r` that has no active asset. A window with no scored step has a total weight of zero, and its mean is `nan`. This is the mean that the `# Mathematical definition` of [`plot_covariance_calibration`](@ref) states. Under the weights of the summary, a window as wide as the run gives the mean of [`covariance_forecast_summary`](@ref).

# Arguments

  - `v`: The statistic, one entry per step.
  - `omega`: The weight of each step, or `nothing` for the weight one, as `covariance_step_weight` reads it.
  - `scored`: Whether each step has at least one active asset.
  - `r`: The range of steps of the window.
  - `nan`: The `NaN` of the element type of the series.

# Returns

  - `m::Real`: The weighted mean, or `nan`.

# Related

  - `covariance_step_weight`
  - `covariance_rolling_mean`
  - `covariance_window_std`
"""
function covariance_window_mean(v::VecNum, omega::Option{<:AbstractVector{<:Real}},
                                scored::AbstractVector{Bool}, r::AbstractUnitRange,
                                nan::Real)
    s = zero(nan)
    n = 0
    for j in r
        if !(scored[j])
            continue
        end
        nj = covariance_step_weight(omega, j)
        s += nj * v[j]
        n += nj
    end
    return iszero(n) ? nan : s / n
end
"""
    covariance_window_std(v::VecNum, scored::AbstractVector{Bool}, r::AbstractUnitRange,
                          nan::Real)

Return the sample standard deviation of a per-step statistic over the scored steps of one window.

The bias statistic of [`plot_covariance_calibration`](@ref) is this deviation of the standardised return of a test portfolio. The deviation weights each scored step by one, and it divides by one less than the number of scored steps. So it needs two scored steps, and it is `nan` with fewer.

# Arguments

  - `v`: The statistic, one entry per step.
  - `scored`: Whether each step has at least one active asset.
  - `r`: The range of steps of the window.
  - `nan`: The `NaN` of the element type of the series.

# Returns

  - `s::Real`: The sample standard deviation, or `nan`.

# Related

  - `covariance_window_mean`
  - `covariance_rolling_std`
"""
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
"""
    covariance_rolling(f, v::VecNum, wins::AbstractVector{<:Pair})

Evaluate a statistic of one window at each point of a rolling series.

Point `t` of the series is `f(r, nan)`, where `t => r` is the pair of `wins` for that point. A point with no pair has no window, and it stays `NaN`. `nan` is the `NaN` of the element type of `v`, from [`float_if_integer`](@ref), so an integer series gives a float series.

# Arguments

  - `f`: The statistic of one window, called as `f(r, nan)`.
  - `v`: The series, one entry per step. Its length sets the length of the output.
  - `wins`: The windows, from `covariance_windows`.

# Returns

  - `out::Vector`: The rolling series, one entry per step.

# Related

  - `covariance_windows`
  - `covariance_rolling_mean`
  - `covariance_rolling_std`
"""
function covariance_rolling(f, v::VecNum, wins::AbstractVector{<:Pair})
    nan = PortfolioOptimisers.float_if_integer(eltype(v))(NaN)
    out = fill(nan, length(v))
    for (t, r) in wins
        out[t] = f(r, nan)
    end
    return out
end
"""
    covariance_rolling_mean(v::VecNum, omega::Option{<:AbstractVector{<:Real}},
                            scored::AbstractVector{Bool}, wins::AbstractVector{<:Pair})

Return the rolling weighted mean of a per-step statistic over the scored steps of each window.

Each point is the `covariance_window_mean` of its window, and a point with no window is `NaN`.

# Arguments

  - `v`: The statistic, one entry per step.
  - `omega`: The weight of each step, or `nothing` for the weight one.
  - `scored`: Whether each step has at least one active asset.
  - `wins`: The windows, from `covariance_windows`.

# Returns

  - `out::Vector`: The rolling mean, one entry per step.

# Related

  - `covariance_window_mean`
  - `covariance_rolling`
"""
function covariance_rolling_mean(v::VecNum, omega::Option{<:AbstractVector{<:Real}},
                                 scored::AbstractVector{Bool}, wins::AbstractVector{<:Pair})
    return covariance_rolling((r, nan) -> covariance_window_mean(v, omega, scored, r, nan),
                              v, wins)
end
"""
    covariance_rolling_std(v::VecNum, scored::AbstractVector{Bool},
                           wins::AbstractVector{<:Pair})

Return the rolling sample standard deviation of a per-step statistic over the scored steps of each window.

Each point is the `covariance_window_std` of its window, and a point with no window is `NaN`.

# Arguments

  - `v`: The statistic, one entry per step.
  - `scored`: Whether each step has at least one active asset.
  - `wins`: The windows, from `covariance_windows`.

# Returns

  - `out::Vector`: The rolling standard deviation, one entry per step.

# Related

  - `covariance_window_std`
  - `covariance_rolling`
"""
function covariance_rolling_std(v::VecNum, scored::AbstractVector{Bool},
                                wins::AbstractVector{<:Pair})
    return covariance_rolling((r, nan) -> covariance_window_std(v, scored, r, nan), v, wins)
end
"""
    covariance_portfolio_line(f, V::MatNum)

Reduce the rolling series of each test portfolio to the line and the band of a figure.

`f` maps the column of one test portfolio to its rolling series. One test portfolio gives its own series and no band. Several give their median at each point, with a band from the fifth to the ninety-fifth percentile over the portfolios. These are the statistics that [`covariance_forecast_summary`](@ref) gives over its portfolios. [`summary_quantile`](@ref) takes the percentiles, so a point where one portfolio is `NaN` has a `NaN` median and `NaN` band edges.

# Arguments

  - `f`: The rolling series of one column.
  - `V`: The per-step statistic of each test portfolio, one row per step and one column per portfolio.

# Returns

  - `y::Vector`: The line, one entry per step.
  - `band::Union{Nothing, Tuple}`: `nothing` for one test portfolio, else the lower and the upper edge of the band.

# Related

  - [`covariance_forecast_summary`](@ref)
  - `covariance_calibration_line`
  - `covariance_qlike_line`
"""
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
"""
    covariance_calibration_line(cfer::CovarianceForecastEvaluationResult, d::Symbol,
                                wins::AbstractVector{<:Pair}, sw::AbstractStepWeighting)

Return the rolling series of one diagnostic of `plot_covariance_calibration`, with its band.

`:mahalanobis` gives the weighted mean of the Mahalanobis ratio. Its weights are those that `sw` gives to the degrees of freedom of [`target_dof`](@ref). `:diagonal` gives the weighted mean of [`covariance_diagonal_mean`](@ref). Its weights are those that `sw` gives to the degrees of freedom of [`target_step_dof`](@ref). `:bias` gives the sample standard deviation of the standardised return of each test portfolio, reduced by `covariance_portfolio_line`. Only the bias statistic can have a band.

# Arguments

  - `cfer`: The evaluation to draw, from [`covariance_forecast_evaluation`](@ref).
  - `d`: The diagnostic, `:mahalanobis`, `:diagonal` or `:bias`.
  - `wins`: The windows, from `covariance_windows`.
  - `sw`: The step weighting of the two ratios.

# Returns

  - `y::Vector`: The series, one entry per step.
  - `band::Union{Nothing, Tuple}`: The band of the bias statistic over several test portfolios, else `nothing`.

# Related

  - [`plot_covariance_calibration`](@ref)
  - [`covariance_step_weights`](@ref)
  - `covariance_rolling_mean`
  - `covariance_portfolio_line`
"""
function covariance_calibration_line(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult,
                                     d::Symbol, wins::AbstractVector{<:Pair},
                                     sw::PortfolioOptimisers.AbstractStepWeighting)
    scored = cfer.n_valid .> 0
    return if d === :mahalanobis
        nu = PortfolioOptimisers.target_dof.(Ref(cfer.target), cfer.n_valid, cfer.horizon)
        omega = PortfolioOptimisers.covariance_step_weights(sw, nu)
        covariance_rolling_mean(cfer.mahalanobis_ratio, omega, scored, wins), nothing
    elseif d === :diagonal
        nu = PortfolioOptimisers.target_step_dof.(Ref(cfer.target), cfer.horizon)
        omega = PortfolioOptimisers.covariance_step_weights(sw, nu)
        covariance_rolling_mean(PortfolioOptimisers.covariance_diagonal_mean(cfer), omega,
                                scored, wins), nothing
    else
        covariance_portfolio_line(v -> covariance_rolling_std(v, scored, wins),
                                  cfer.standardised_return)
    end
end
"""
    covariance_qlike_line(cfer::CovarianceForecastEvaluationResult,
                          wins::AbstractVector{<:Pair})

Return the rolling mean of the portfolio QLIKE loss of an evaluation, with its band.

The mean weights each scored step by one. `covariance_portfolio_line` reduces the series of the test portfolios to one line and its band.

# Arguments

  - `cfer`: The evaluation to draw, from [`covariance_forecast_evaluation`](@ref).
  - `wins`: The windows, from `covariance_windows`.

# Returns

  - `y::Vector`: The series, one entry per step.
  - `band::Union{Nothing, Tuple}`: The band over several test portfolios, else `nothing`.

# Related

  - [`plot_covariance_qlike`](@ref)
  - `covariance_rolling_mean`
  - `covariance_portfolio_line`
"""
function covariance_qlike_line(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult,
                               wins::AbstractVector{<:Pair})
    scored = cfer.n_valid .> 0
    return covariance_portfolio_line(v -> covariance_rolling_mean(v, nothing, scored, wins),
                                     cfer.portfolio_qlike)
end
"""
    covariance_exceedance_line(cfer::CovarianceForecastEvaluationResult, q::Real,
                               wins::AbstractVector{<:Pair})

Return the rolling exceedance rate of an evaluation at level `q`.

The rate is the mean of the indicator of [`covariance_exceedance`](@ref) over the scored steps of each window. The mean weights each scored step by one.

# Arguments

  - `cfer`: The evaluation to draw, from [`covariance_forecast_evaluation`](@ref).
  - `q`: The confidence level of the chi-squared quantile.
  - `wins`: The windows, from `covariance_windows`.

# Returns

  - `y::Vector`: The series, one entry per step.

# Related

  - [`plot_covariance_exceedance`](@ref)
  - [`covariance_exceedance`](@ref)
  - `covariance_rolling_mean`
"""
function covariance_exceedance_line(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult,
                                    q::Real, wins::AbstractVector{<:Pair})
    e = PortfolioOptimisers.covariance_exceedance(cfer, q)
    return covariance_rolling_mean(e, nothing, cfer.n_valid .> 0, wins)
end
"""
    covariance_series_label(name::Nothing, label::AbstractString, n::Integer)
    covariance_series_label(name, label::AbstractString, n::Integer)

Return the legend label of one series.

A series of one evaluation has `name = nothing`, and its label is `label`, which names what the series draws. A series of a comparison has the name of its evaluation. Its label is that name alone when each evaluation draws one series, so when `n` is one. Otherwise its label is the name, a hyphen and `label`.

# Arguments

  - `name`: The name of the evaluation, or `nothing` for a figure of one evaluation.
  - `label`: The name of what the series draws.
  - `n`: The number of series that each evaluation draws.

# Returns

  - `lbl::String`: The legend label.

# Related

  - [`covariance_forecast_names`](@ref)
  - `covariance_calibration_entries`
"""
function covariance_series_label(::Nothing, label::AbstractString, ::Integer)
    return label
end
function covariance_series_label(name, label::AbstractString, n::Integer)
    return isone(n) ? string(name) : "$(name) - $(label)"
end
"""
    covariance_plot_line!(plt, x, y, label, band::Nothing; kwargs...)
    covariance_plot_line!(plt, x, y, label, band::Tuple; kwargs...)

Draw one series on a figure, with its band when it has one.

The line has a width of two. A band `(lo, hi)` becomes a ribbon from `lo` to `hi`, with a fill opacity of 0.15. `kwargs` go to the plotting backend last, so a keyword of the caller replaces a default of this function.

# Arguments

  - `plt`: The figure.
  - `x`: The dates of the points.
  - `y`: The values of the points.
  - `label`: The legend label.
  - `band`: The lower and the upper edge of the band, or `nothing`.
  - `kwargs...`: Keyword arguments for the plotting backend.

# Returns

  - `plt::Plots.Plot`: The figure, with the series added.

# Related

  - `covariance_plot_lines`
"""
function covariance_plot_line!(plt, x, y, label, ::Nothing; kwargs...)
    return plot!(plt, x, y; label = label, linewidth = 2, kwargs...)
end
function covariance_plot_line!(plt, x, y, label, band::Tuple; kwargs...)
    lo, hi = band
    return plot!(plt, x, y; ribbon = (y .- lo, hi .- y), fillalpha = 0.15, label = label,
                 linewidth = 2, kwargs...)
end
"""
    covariance_plot_lines(entries, title::AbstractString, ylabel::AbstractString; kwargs...)

Make a figure that draws each entry as one series.

The x axis carries the label "Observation", and the figure shows its legend. `kwargs` go to the figure and to each series.

# Arguments

  - `entries`: The series to draw, each from `covariance_plot_entry`.
  - `title`: The title of the figure.
  - `ylabel`: The label of the y axis.
  - `kwargs...`: Keyword arguments for the plotting backend.

# Returns

  - `plt::Plots.Plot`: The figure.

# Related

  - `covariance_plot_line!`
  - `covariance_plot_entry`
"""
function covariance_plot_lines(entries, title::AbstractString, ylabel::AbstractString;
                               kwargs...)
    plt = plot(; title = title, xlabel = "Observation", ylabel = ylabel, legend = true,
               kwargs...)
    for e in entries
        covariance_plot_line!(plt, e.x, e.y, e.label, e.band; kwargs...)
    end
    return plt
end
"""
    covariance_band_slice(band::Nothing, k::AbstractUnitRange)
    covariance_band_slice(band::Tuple, k::AbstractUnitRange)

Cut a band to the points that a series draws.

# Arguments

  - `band`: The lower and the upper edge of the band, or `nothing`.
  - `k`: The indices of the points to draw.

# Returns

  - `band::Union{Nothing, Tuple}`: Both edges at `k`, or `nothing`.

# Related

  - `covariance_plot_entry`
"""
function covariance_band_slice(::Nothing, ::AbstractUnitRange)
    return nothing
end
function covariance_band_slice(band::Tuple, k::AbstractUnitRange)
    return (band[1][k], band[2][k])
end
"""
    covariance_plot_entry(cfer::CovarianceForecastEvaluationResult, rw::NamedTuple, y::VecNum,
                          label, band)

Make the entry of one series to draw, from the first point that the windows draw.

# Arguments

  - `cfer`: The evaluation, from [`covariance_forecast_evaluation`](@ref). It gives the dates of the points.
  - `rw`: The windows and the first point, from `covariance_plot_windows`.
  - `y`: The series, one entry per step.
  - `label`: The legend label.
  - `band`: The lower and the upper edge of the band, or `nothing`.

# Returns

  - `e::NamedTuple`: The dates `x`, the values `y`, the `label` and the `band`, each from the first point to draw.

# Related

  - `covariance_plot_windows`
  - `covariance_band_slice`
  - `covariance_plot_lines`
"""
function covariance_plot_entry(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult,
                               rw::NamedTuple, y::VecNum, label, band)
    k = (rw.from):length(y)
    return (; x = cfer.dates[k], y = y[k], label = label,
            band = covariance_band_slice(band, k))
end
"""
    covariance_calibration_entries(cfer::CovarianceForecastEvaluationResult, name,
                                   diagnostics, sw::AbstractStepWeighting, wo::NamedTuple)

Make one entry per diagnostic of one evaluation, for `plot_covariance_calibration`.

# Arguments

  - `cfer`: The evaluation to draw, from [`covariance_forecast_evaluation`](@ref).
  - `name`: The name of the evaluation in a comparison, or `nothing`.
  - `diagnostics`: The diagnostics to draw.
  - `sw`: The step weighting of the two ratios.
  - `wo`: The three window keywords of the figure.

# Validation

  - The rules of `covariance_plot_windows`.

# Returns

  - `entries::Vector{<:NamedTuple}`: One entry per diagnostic, from `covariance_plot_entry`.

# Related

  - [`plot_covariance_calibration`](@ref)
  - `covariance_calibration_line`
  - `covariance_series_label`
"""
function covariance_calibration_entries(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult,
                                        name, diagnostics,
                                        sw::PortfolioOptimisers.AbstractStepWeighting,
                                        wo::NamedTuple)
    rw = covariance_plot_windows(cfer, wo)
    return map(collect(diagnostics)) do d
        y, band = covariance_calibration_line(cfer, d, rw.wins, sw)
        label = covariance_series_label(name, COVARIANCE_CALIBRATION_LABELS[d],
                                        length(diagnostics))
        return covariance_plot_entry(cfer, rw, y, label, band)
    end
end
"""
    covariance_qlike_entry(cfer::CovarianceForecastEvaluationResult, name, wo::NamedTuple)

Make the entry of one evaluation, for `plot_covariance_qlike`.

# Arguments

  - `cfer`: The evaluation to draw, from [`covariance_forecast_evaluation`](@ref).
  - `name`: The name of the evaluation in a comparison, or `nothing`.
  - `wo`: The three window keywords of the figure.

# Validation

  - The rules of `covariance_plot_windows`.

# Returns

  - `e::NamedTuple`: The entry, from `covariance_plot_entry`.

# Related

  - [`plot_covariance_qlike`](@ref)
  - `covariance_qlike_line`
"""
function covariance_qlike_entry(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult,
                                name, wo::NamedTuple)
    rw = covariance_plot_windows(cfer, wo)
    y, band = covariance_qlike_line(cfer, rw.wins)
    return covariance_plot_entry(cfer, rw, y, covariance_series_label(name, "QLIKE", 1),
                                 band)
end
"""
    covariance_exceedance_entries(cfer::CovarianceForecastEvaluationResult, name, levels,
                                  wo::NamedTuple)

Make one entry per level of one evaluation, for `plot_covariance_exceedance`.

# Arguments

  - `cfer`: The evaluation to draw, from [`covariance_forecast_evaluation`](@ref).
  - `name`: The name of the evaluation in a comparison, or `nothing`.
  - `levels`: The confidence levels of the chi-squared quantiles.
  - `wo`: The three window keywords of the figure.

# Validation

  - The rules of `covariance_plot_windows`.

# Returns

  - `entries::Vector{<:NamedTuple}`: One entry per level, from `covariance_plot_entry`.

# Related

  - [`plot_covariance_exceedance`](@ref)
  - `covariance_exceedance_line`
  - `covariance_series_label`
"""
function covariance_exceedance_entries(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult,
                                       name, levels, wo::NamedTuple)
    rw = covariance_plot_windows(cfer, wo)
    return map(collect(levels)) do q
        y = covariance_exceedance_line(cfer, q, rw.wins)
        label = covariance_series_label(name, "q = $(q)", length(levels))
        return covariance_plot_entry(cfer, rw, y, label, nothing)
    end
end
"""
    covariance_calibration_title(window::Integer)

Return the title of `plot_covariance_calibration`, which names the width of the window.

# Arguments

  - `window`: Number of steps in the window.

# Returns

  - `title::String`: The title.

# Related

  - [`plot_covariance_calibration`](@ref)
"""
function covariance_calibration_title(window::Integer)
    return "Rolling Covariance Calibration (window=$(window))"
end
"""
    covariance_qlike_title(window::Integer)

Return the title of `plot_covariance_qlike`, which names the width of the window.

# Arguments

  - `window`: Number of steps in the window.

# Returns

  - `title::String`: The title.

# Related

  - [`plot_covariance_qlike`](@ref)
"""
function covariance_qlike_title(window::Integer)
    return "Rolling Portfolio QLIKE Loss (window=$(window))"
end
"""
    covariance_exceedance_title(window::Integer)

Return the title of `plot_covariance_exceedance`, which names the width of the window.

# Arguments

  - `window`: Number of steps in the window.

# Returns

  - `title::String`: The title.

# Related

  - [`plot_covariance_exceedance`](@ref)
"""
function covariance_exceedance_title(window::Integer)
    return "Rolling Exceedance Rate (window=$(window))"
end
function PortfolioOptimisers.plot_covariance_calibration(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult;
                                                         window::Integer = 50,
                                                         diagnostics = (:mahalanobis,
                                                                        :diagonal, :bias),
                                                         step_weighting::PortfolioOptimisers.AbstractStepWeighting = PortfolioOptimisers.DofStepWeighting(),
                                                         scored_steps::Bool = false,
                                                         whole_windows::Bool = false,
                                                         kwargs...)
    covariance_plot_check_diagnostics(diagnostics)
    entries = covariance_calibration_entries(cfer, nothing, diagnostics, step_weighting,
                                             (; window, scored_steps, whole_windows))
    plt = covariance_plot_lines(entries, covariance_calibration_title(window),
                                "Calibration"; kwargs...)
    return idio_diagnostic_reference!(plt, 1.0)
end
function PortfolioOptimisers.plot_covariance_calibration(cfers::AbstractVector{<:PortfolioOptimisers.CovarianceForecastEvaluationResult};
                                                         names = nothing,
                                                         window::Integer = 50,
                                                         diagnostics = (:mahalanobis,
                                                                        :diagonal, :bias),
                                                         step_weighting::PortfolioOptimisers.AbstractStepWeighting = PortfolioOptimisers.DofStepWeighting(),
                                                         scored_steps::Bool = false,
                                                         whole_windows::Bool = false,
                                                         kwargs...)
    covariance_plot_check_evaluations(cfers)
    covariance_plot_check_diagnostics(diagnostics)
    nms = PortfolioOptimisers.covariance_forecast_names(names, length(cfers))
    entries = reduce(vcat,
                     [covariance_calibration_entries(cfers[k], nms[k], diagnostics,
                                                     step_weighting,
                                                     (; window, scored_steps,
                                                      whole_windows))
                      for k in eachindex(cfers, nms)])
    plt = covariance_plot_lines(entries, covariance_calibration_title(window),
                                "Calibration"; kwargs...)
    return idio_diagnostic_reference!(plt, 1.0)
end
function PortfolioOptimisers.plot_covariance_qlike(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult;
                                                   window::Integer = 50,
                                                   scored_steps::Bool = false,
                                                   whole_windows::Bool = false, kwargs...)
    covariance_plot_check_no_step_weighting(kwargs, "plot_covariance_qlike")
    return covariance_plot_lines([covariance_qlike_entry(cfer, nothing,
                                                         (; window, scored_steps,
                                                          whole_windows))],
                                 covariance_qlike_title(window), "QLIKE Loss"; kwargs...)
end
function PortfolioOptimisers.plot_covariance_qlike(cfers::AbstractVector{<:PortfolioOptimisers.CovarianceForecastEvaluationResult};
                                                   names = nothing, window::Integer = 50,
                                                   scored_steps::Bool = false,
                                                   whole_windows::Bool = false, kwargs...)
    covariance_plot_check_evaluations(cfers)
    covariance_plot_check_no_step_weighting(kwargs, "plot_covariance_qlike")
    nms = PortfolioOptimisers.covariance_forecast_names(names, length(cfers))
    entries = [covariance_qlike_entry(cfers[k], nms[k],
                                      (; window, scored_steps, whole_windows))
               for k in eachindex(cfers, nms)]
    return covariance_plot_lines(entries, covariance_qlike_title(window), "QLIKE Loss";
                                 kwargs...)
end
function PortfolioOptimisers.plot_covariance_exceedance(cfer::PortfolioOptimisers.CovarianceForecastEvaluationResult;
                                                        levels = (0.95, 0.99),
                                                        window::Integer = 50,
                                                        scored_steps::Bool = false,
                                                        whole_windows::Bool = false,
                                                        kwargs...)
    covariance_plot_check_levels(levels)
    covariance_plot_check_no_step_weighting(kwargs, "plot_covariance_exceedance")
    entries = covariance_exceedance_entries(cfer, nothing, levels,
                                            (; window, scored_steps, whole_windows))
    plt = covariance_plot_lines(entries, covariance_exceedance_title(window),
                                "Exceedance Rate"; kwargs...)
    # Draw each target rate in the colour of the series of its level.
    for (k, q) in enumerate(levels)
        hline!(plt, [1 - q]; label = "", linewidth = 1, linestyle = :dash,
               color = plt.series_list[k][:seriescolor])
    end
    return plt
end
function PortfolioOptimisers.plot_covariance_exceedance(cfers::AbstractVector{<:PortfolioOptimisers.CovarianceForecastEvaluationResult};
                                                        names = nothing, levels = (0.95,),
                                                        window::Integer = 50,
                                                        scored_steps::Bool = false,
                                                        whole_windows::Bool = false,
                                                        kwargs...)
    covariance_plot_check_evaluations(cfers)
    covariance_plot_check_levels(levels)
    covariance_plot_check_no_step_weighting(kwargs, "plot_covariance_exceedance")
    nms = PortfolioOptimisers.covariance_forecast_names(names, length(cfers))
    entries = reduce(vcat,
                     [covariance_exceedance_entries(cfers[k], nms[k], levels,
                                                    (; window, scored_steps, whole_windows))
                      for k in eachindex(cfers, nms)])
    plt = covariance_plot_lines(entries, covariance_exceedance_title(window),
                                "Exceedance Rate"; kwargs...)
    # Several evaluations share a level, so draw each target rate once, in grey.
    for q in levels
        hline!(plt, [1 - q]; label = "", linewidth = 1, linestyle = :dash, color = :gray)
    end
    return plt
end
