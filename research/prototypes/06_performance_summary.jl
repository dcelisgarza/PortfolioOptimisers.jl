# =============================================================================
# Prototype 6 — A performance summary that is a Result, not a plot.
#
# Purpose
#   The seven headline statistics a caller wants after a backtest are computed
#   at `ext/PortfolioOptimisersPlotsExt.jl:1977-1992`, **inside the plot
#   function**, and are never returned as numbers. A caller without StatsPlots
#   installed cannot get them at all, and a caller with StatsPlots installed
#   gets a bar chart when they wanted a table.
#
#   That inverts the library's own layering. Everywhere else a plot renders a
#   Result that some other function computed. This is the one place a plot
#   computes one.
#
#   This file shows the Result and the function. The plot extension then keeps
#   only the drawing.
#
# Status
#   Standalone. Depends on `Statistics` only. Deliberately no dependency on
#   Distributions, so that it can sit in `src/` beside the other value-level
#   evaluators without adding weight.
#
# Notation used throughout this file
#   T      Number of periods in the return series.
#   r      Vector of periodic portfolio returns, length `T`. Simple returns,
#          not log returns.
#   ppy    Periods per year, used to annualise. 252 for daily, 52 for weekly,
#          12 for monthly.
#   alpha  Tail probability for the tail statistics, in `(0, 1)`.
#
# Sources
#   Sharpe, W. F. (1994). The Sharpe ratio. Journal of Portfolio Management
#     21(1), 49-58.
#   Sortino, F. A. and Price, L. N. (1994). Performance measurement in a
#     downside risk framework. Journal of Investing 3(3), 59-64.
#   Young, T. W. (1991). Calmar ratio: a smoother tool. Futures 20(1), 40.
#   Martin, P. G. and McCann, B. B. (1989). The Investor's Guide to Fidelity
#     Funds. Wiley. The Ulcer index.
#   Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional
#     value-at-risk. Journal of Risk 2(3), 21-41.
#   Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe ratio efficient
#     frontier. Journal of Risk 15(2), 3-44. The standard error of a Sharpe
#     ratio, which is why `sharpe_stderr` is reported beside `sharpe`.
# =============================================================================
module PerformanceSummary

using Statistics

export PerformanceSummaryResult, performance_summary, cumulative_returns, drawdown_series

"""
    PerformanceSummaryResult{T}

The computed summary of a return series.

# Fields

  - `n_periods::Int`: Number of periods, `T`.
  - `periods_per_year::T`: The annualisation factor used.
  - `ann_return::T`: Annualised arithmetic mean return.
  - `ann_volatility::T`: Annualised standard deviation.
  - `sharpe::T`: Annualised Sharpe ratio at a zero risk-free rate.
  - `sharpe_stderr::T`: Standard error of the annualised Sharpe ratio.
  - `sortino::T`: Annualised Sortino ratio.
  - `calmar::T`: Annualised return divided by the absolute maximum drawdown.
  - `max_drawdown::T`: Maximum drawdown, a non-positive number.
  - `ulcer_index::T`: Root mean square drawdown.
  - `cvar::T`: Conditional Value-at-Risk of the periodic loss at `alpha`,
    reported as a positive loss.
  - `var::T`: Value-at-Risk of the periodic loss at `alpha`, positive.
  - `skewness::T`: Sample skewness of the periodic returns.
  - `excess_kurtosis::T`: Sample excess kurtosis, zero for a normal sample.
  - `hit_rate::T`: Fraction of periods with a strictly positive return.
  - `best::T`, `worst::T`: Largest and smallest single-period returns.
  - `alpha::T`: The tail probability used.
  - `compound::Bool`: Whether the drawdown path was compounded.

# Notes

  - **`sharpe_stderr` is the field the plot never had.** A Sharpe ratio without
    its standard error invites a caller to rank two strategies that the data
    cannot separate. Reporting them together is the cheapest honesty available.
"""
struct PerformanceSummaryResult{T <: Real}
    n_periods::Int
    periods_per_year::T
    ann_return::T
    ann_volatility::T
    sharpe::T
    sharpe_stderr::T
    sortino::T
    calmar::T
    max_drawdown::T
    ulcer_index::T
    cvar::T
    var::T
    skewness::T
    excess_kurtosis::T
    hit_rate::T
    best::T
    worst::T
    alpha::T
    compound::Bool
end

"""
    cumulative_returns(r::AbstractVector; compound::Bool = false) -> Vector

Return the cumulative wealth path implied by a periodic return series.

# Arguments

  - `r`: Periodic simple returns, length `T`.
  - `compound`: If `true`, the path is `cumprod(1 .+ r)`. If `false`, it is
    `1 .+ cumsum(r)`.

# Returns

  - A vector of length `T` starting near one.

# Details

The two conventions disagree, and the disagreement grows with the horizon. The
compounded path is what an investor experiences. The additive path is what a
constant-notional strategy experiences. The library's `cumulative_returns` in
`src/17_NetReturnsDrawdowns.jl` carries the same flag, and this prototype
matches its meaning.
"""
function cumulative_returns(r::AbstractVector{<:Real}; compound::Bool = false)
    return compound ? cumprod(one(eltype(r)) .+ r) : one(eltype(r)) .+ cumsum(r)
end

"""
    drawdown_series(cr::AbstractVector; compound::Bool = false) -> Vector

Return the drawdown at every point of a cumulative wealth path.

# Arguments

  - `cr`: Cumulative wealth path, length `T`, as returned by
    [`cumulative_returns`](@ref).
  - `compound`: Must match the flag used to build `cr`.

# Returns

  - A vector of non-positive numbers, length `T`.

# Mathematical definition

Let `peak_t = max(cr_1, ..., cr_t)` be the running maximum. Then

    dd_t = cr_t / peak_t - 1        (compound)
    dd_t = cr_t - peak_t            (additive)

The compounded form is a fraction of the peak. The additive form is in the
units of the return series. **They are not interchangeable**, and a Calmar
ratio built from one is not comparable with a Calmar ratio built from the
other.
"""
function drawdown_series(cr::AbstractVector{<:Real}; compound::Bool = false)
    T = length(cr)
    dd = similar(float.(cr))
    peak = first(cr)
    @inbounds for t in 1:T
        peak = max(peak, cr[t])
        dd[t] = compound ? cr[t] / peak - 1 : cr[t] - peak
    end
    return dd
end

"""
    performance_summary(r::AbstractVector; periods_per_year::Real = 252,
                        alpha::Real = 0.05, compound::Bool = false)
        -> PerformanceSummaryResult

Compute the summary statistics of a periodic return series.

# Arguments

  - `r`: Periodic simple returns, length `T`. Must hold at least two entries.
  - `periods_per_year`: Annualisation factor, `ppy`.
  - `alpha`: Tail probability, in `(0, 1)`.
  - `compound`: Whether to compound the wealth path for the drawdown
    statistics.

# Returns

  - A [`PerformanceSummaryResult`](@ref).

# Mathematical definitions

Let `m` and `s` be the sample mean and the sample standard deviation of `r`.

    ann_return      =  m * ppy
    ann_volatility  =  s * sqrt(ppy)
    sharpe          =  ann_return / ann_volatility
    downside_dev    =  sqrt( mean( min(r, 0)^2 ) * ppy )
    sortino         =  ann_return / downside_dev
    calmar          =  ann_return / abs(max_drawdown)
    ulcer_index     =  sqrt( mean( dd^2 ) )
    VaR_alpha       =  -quantile(r, alpha)
    CVaR_alpha      =  -mean( r[ r <= quantile(r, alpha) ] )

The standard error of the Sharpe ratio uses the Bailey and Lopez de Prado
(2012) expression, which corrects for the skewness `g1` and excess kurtosis
`g2` of the returns:

    se(SR_period)  =  sqrt( (1 - g1 * SR + (g2 / 4) * SR^2) / (T - 1) )

and is annualised by `sqrt(ppy)`. **A non-normal return series makes a Sharpe
ratio less precise than the naive `sqrt((1 + SR^2 / 2) / T)` suggests**, and
negative skew makes it worse, which is the case that matters for a real
portfolio.

# Validation

  - `length(r) >= 2`.
  - `0 < alpha < 1`.
  - `periods_per_year > 0`.

# Notes

  - Every statistic here is already computed somewhere in the library or in its
    plot extension. **The contribution is that they are returned**, so a caller
    can put them in a table, compare two runs, assert on them in a test, or
    feed them to a search.
"""
function performance_summary(r::AbstractVector{<:Real}; periods_per_year::Real = 252,
                             alpha::Real = 0.05, compound::Bool = false)
    T = length(r)
    if T < 2
        throw(ArgumentError("need at least two observations, got $(T)"))
    end
    if !(zero(alpha) < alpha < one(alpha))
        throw(DomainError(alpha, "alpha must satisfy 0 < alpha < 1"))
    end
    if !(periods_per_year > 0)
        throw(DomainError(periods_per_year, "periods_per_year must be > 0"))
    end
    F = float(eltype(r))
    ppy = F(periods_per_year)
    a = F(alpha)

    m = mean(r)
    s = std(r)
    ann_ret = m * ppy
    ann_vol = s * sqrt(ppy)
    sharpe = ann_vol > 0 ? ann_ret / ann_vol : F(NaN)

    # Downside deviation uses a zero minimum acceptable return, and divides by
    # T rather than by the count of negative periods. That is the Sortino and
    # Price convention, and it is the one that makes the ratio comparable
    # across series with different loss frequencies.
    neg = min.(r, zero(F))
    ddev = sqrt(mean(abs2, neg) * ppy)
    sortino = ddev > 0 ? ann_ret / ddev : F(NaN)

    cr = cumulative_returns(r; compound = compound)
    dd = drawdown_series(cr; compound = compound)
    max_dd = minimum(dd)
    ulcer = sqrt(mean(abs2, dd))
    calmar = max_dd < 0 ? ann_ret / abs(max_dd) : F(NaN)

    sorted = sort(r)
    k = max(1, floor(Int, a * T))
    var_val = -sorted[k]
    cvar_val = -mean(view(sorted, 1:k))

    # Sample skewness and excess kurtosis, moment estimators.
    z = (r .- m) ./ s
    g1 = mean(z .^ 3)
    g2 = mean(z .^ 4) - 3

    sr_p = s > 0 ? m / s : F(NaN)
    var_sr = (1 - g1 * sr_p + (g2 / 4) * sr_p^2) / (T - 1)
    sharpe_se = var_sr > 0 ? sqrt(var_sr) * sqrt(ppy) : F(NaN)

    hit = count(>(zero(F)), r) / T

    return PerformanceSummaryResult{F}(T, ppy, ann_ret, ann_vol, sharpe, sharpe_se, sortino,
                                       calmar, max_dd, ulcer, cvar_val, var_val, g1, g2,
                                       F(hit), F(maximum(r)), F(minimum(r)), a, compound)
end

end # module PerformanceSummary
