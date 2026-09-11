"""
$(DocStringExtensions.TYPEDEF)

The headline statistics of one or more covariance forecast evaluations, one entry per evaluation.

`CovarianceForecastSummaryResult` is what [`covariance_forecast_summary`](@ref) returns. It is columnar, on an axis whose entries are the evaluations that were summarised, so a single evaluation is the length-1 case and a side-by-side of two forecasts is the length-2 case, and no comparison class ships beside it — the shape [`ForecastSummaryResult`](@ref) has for the return forecasts. Every column is read off the per-step diagnostics of a [`CovarianceForecastEvaluationResult`](@ref).

# Mathematical definition

```math
\\begin{align}
\\bar{m} &= \\frac{\\sum_{t=1}^{M} \\nu_t\\, m_t}{\\sum_{t=1}^{M} \\nu_t}\\,, \\quad \\nu_t = \\operatorname{dof}(N_t, h_t)\\,, \\\\
\\bar{d} &= \\frac{\\sum_{t=1}^{M} \\nu'_t\\, \\bar{d}_t}{\\sum_{t=1}^{M} \\nu'_t}\\,, \\quad \\nu'_t = \\operatorname{dof}'(h_t)\\,, \\\\
\\bar{m} &\\in 1 \\pm z_{\\alpha/2} \\sqrt{\\frac{2}{\\sum_{t=1}^{M} \\nu_t}}\\,, \\quad \\bar{d} \\in 1 \\pm z_{\\alpha/2} \\sqrt{\\frac{2}{\\sum_{t=1}^{M} \\nu'_t}}\\,, \\\\
B &= \\sqrt{\\frac{1}{M - 1} \\sum_{t=1}^{M} \\left(b_t - \\bar{b}\\right)^2}\\,, \\\\
e_q &= \\frac{1}{M} \\sum_{t=1}^{M} \\mathbb{1}\\left[\\nu_t\\, m_t > \\chi^2_{\\nu_t}(q)\\right]\\,.
\\end{align}
```

Where:

  - ``\\bar{m}``: Mahalanobis ratio over the walk-forward.
  - ``m_t``: Mahalanobis ratio of step ``t``.
  - ``\\nu_t``: Degrees of freedom of the Mahalanobis statistic of step ``t`` under a Gaussian null, ``N_t h_t`` for the realised covariance and ``N_t`` for the horizon return.
  - ``N_t``: Number of active assets at step ``t``.
  - $(math_dict[:h_step])
  - $(math_dict[:M_steps])
  - ``\\bar{d}``: Diagonal ratio over the walk-forward.
  - ``\\bar{d}_t``: Mean of the diagonal ratio over the active assets of step ``t``.
  - ``\\nu'_t``: Degrees of freedom of one asset's step ratio under a Gaussian null, ``h_t`` for the realised covariance and ``1`` for the horizon return.
  - ``z_{\\alpha/2}``: Upper ``\\alpha / 2`` quantile of the standard normal distribution.
  - ``B``: Bias statistic of a test portfolio, the sample standard deviation of its standardised returns.
  - ``b_t``: Standardised return of the test portfolio at step ``t``.
  - ``e_q``: Exceedance rate at level ``q``, the share of steps whose Mahalanobis statistic exceeds the chi-squared quantile of that level.

Under a Gaussian null with the forecast correct and the whitened returns independent, ``\\sum_t \\nu_t\\, \\bar{m} \\sim \\chi^2_{\\sum_t \\nu_t}``, so ``\\mathbb{E}[\\bar{m}] = 1`` and the band holds with probability ``1 - \\alpha``; the ratio-of-sums form weights a step by its degrees of freedom, which is the plain mean when every step has the same ``N_t`` and ``h_t``. The band on ``\\bar{d}`` is the band of one asset's ratio, and it is conservative for a mean over assets whose ratios are correlated. If the whitened coordinates have fourth moment ``\\kappa`` in place of the Gaussian three, the variance of ``\\bar{m}`` is ``(\\kappa - 1) / \\sum_t \\nu_t`` and the band widens by ``\\sqrt{(\\kappa - 1) / 2}``; the correction is stated here and not computed, so the Gaussian band is a reference and not a test. The median of ``\\chi^2_{\\nu} / \\nu`` lies below one, near ``1 - 2 / (9 \\nu)``, so a median is compared with that and never with one. ``B = 1`` under a calibrated forecast, above one when the portfolio's risk is under-predicted. ``e_q`` is ``1 - q`` under the Gaussian null, and it rises with heavy tails as well as with a misspecified forecast.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CovarianceForecastSummaryResult(
        names, mahalanobis_mean, mahalanobis_median, mahalanobis_p5, mahalanobis_p95,
        mahalanobis_band_lo, mahalanobis_band_hi, diagonal_mean, diagonal_median,
        diagonal_p5, diagonal_p95, diagonal_band_lo, diagonal_band_hi, bias_statistic,
        bias_p5, bias_p25, bias_p75, bias_p95, qlike_mean, frobenius_mean,
        portfolio_qlike_mean, levels, exceedance, n_steps, n_portfolios, alpha
    ) -> CovarianceForecastSummaryResult

Arguments correspond to the struct's fields, in the order they are declared. The type is a Result, so [`covariance_forecast_summary`](@ref) builds it and a caller reads it; there is no keyword constructor, and the type validates nothing of its own.

# Related

  - [`covariance_forecast_summary`](@ref)
  - [`CovarianceForecastEvaluationResult`](@ref)
  - [`covariance_forecast_evaluation`](@ref)
  - [`ForecastSummaryResult`](@ref)
"""
@concrete struct CovarianceForecastSummaryResult <: AbstractResult
    """
    Name of each evaluation, one entry per evaluation.
    """
    names
    """
    Mahalanobis ratio over the walk-forward, the degrees-of-freedom-weighted mean of the per-step ratios. One entry per evaluation; the target is one.
    """
    mahalanobis_mean
    """
    Median of the per-step Mahalanobis ratio. One entry per evaluation; its target under a Gaussian null lies below one.
    """
    mahalanobis_median
    """
    Fifth percentile of the per-step Mahalanobis ratio. One entry per evaluation.
    """
    mahalanobis_p5
    """
    Ninety-fifth percentile of the per-step Mahalanobis ratio. One entry per evaluation.
    """
    mahalanobis_p95
    """
    Lower end of the Gaussian band on the Mahalanobis ratio at level `alpha`. One entry per evaluation.
    """
    mahalanobis_band_lo
    """
    Upper end of the Gaussian band on the Mahalanobis ratio at level `alpha`. One entry per evaluation.
    """
    mahalanobis_band_hi
    """
    Diagonal ratio over the walk-forward, the degrees-of-freedom-weighted mean of the per-step mean over active assets. One entry per evaluation; the target is one.
    """
    diagonal_mean
    """
    Median of the per-step diagonal ratio, each step's being its mean over active assets. One entry per evaluation.
    """
    diagonal_median
    """
    Fifth percentile of the per-step diagonal ratio. One entry per evaluation.
    """
    diagonal_p5
    """
    Ninety-fifth percentile of the per-step diagonal ratio. One entry per evaluation.
    """
    diagonal_p95
    """
    Lower end of the Gaussian band on one asset's diagonal ratio at level `alpha`. One entry per evaluation.
    """
    diagonal_band_lo
    """
    Upper end of the Gaussian band on one asset's diagonal ratio at level `alpha`. One entry per evaluation.
    """
    diagonal_band_hi
    """
    Bias statistic, the median over the test portfolios of the sample standard deviation of each one's standardised returns. One entry per evaluation; the target is one.
    """
    bias_statistic
    """
    Fifth percentile of the bias statistic over the test portfolios. One entry per evaluation.
    """
    bias_p5
    """
    Twenty-fifth percentile of the bias statistic over the test portfolios. One entry per evaluation.
    """
    bias_p25
    """
    Seventy-fifth percentile of the bias statistic over the test portfolios. One entry per evaluation.
    """
    bias_p75
    """
    Ninety-fifth percentile of the bias statistic over the test portfolios. One entry per evaluation.
    """
    bias_p95
    """
    Mean QLIKE loss over the steps. One entry per evaluation; only a difference between evaluations is a reading.
    """
    qlike_mean
    """
    Mean Frobenius loss over the steps. One entry per evaluation; only a difference between evaluations is a reading.
    """
    frobenius_mean
    """
    Median over the test portfolios of each one's mean portfolio QLIKE loss. One entry per evaluation.
    """
    portfolio_qlike_mean
    """
    Confidence levels of the exceedance rates, as handed to the summary.
    """
    levels
    """
    Exceedance rate of the per-step Mahalanobis statistic against the chi-squared quantile of each level, `evaluations × levels`. Its target is one less the level.
    """
    exceedance
    """
    Number of steps of each evaluation. One entry per evaluation.
    """
    n_steps
    """
    Number of test portfolios of each evaluation. One entry per evaluation.
    """
    n_portfolios
    """
    Level of the Gaussian bands.
    """
    alpha
end
"""
    covariance_forecast_summary(cfers::AbstractVector{<:CovarianceForecastEvaluationResult};
                                names = nothing, alpha::Real = 0.05,
                                levels = (0.95, 0.99)) -> CovarianceForecastSummaryResult
    covariance_forecast_summary(cfer::CovarianceForecastEvaluationResult; kwargs...)

Summarise one or more covariance forecast evaluations, one entry per evaluation.

The verb over [`CovarianceForecastEvaluationResult`](@ref): the mean, median and tail percentiles of the two calibration ratios with the Gaussian band on each mean, the bias statistic of the test portfolios with its cross-portfolio percentiles, the mean of each loss, and the exceedance rate of the Mahalanobis statistic at each level. The single-Result method is the length-1 case. The mean of a ratio is a ratio of sums, weighted by each step's degrees of freedom under the evaluation's target ([`target_dof`](@ref)), so a date walk-forward whose folds differ in length weights each by its length; under an index walk-forward it is the plain mean, which is what the reference implementation reports.

# Algorithm

 1. Per evaluation, weight the per-step Mahalanobis ratio by its degrees of freedom and reduce the diagonal ratio to a per-step mean over active assets, then take the weighted mean, the median and the fifth and ninety-fifth percentiles of each.
 2. Width the Gaussian band of each mean at `alpha` from the summed degrees of freedom.
 3. Take the sample standard deviation of each test portfolio's standardised returns, and its median and percentiles over the portfolios.
 4. Take the mean of each loss over the steps, and the median over the portfolios of the mean portfolio QLIKE.
 5. Count the steps whose Mahalanobis statistic exceeds the chi-squared quantile of each level.

# Arguments

  - `cfers`: The evaluations to summarise.
  - `names`: A name per evaluation, or `nothing` for `"forecast_1"`, `"forecast_2"`, ….
  - `alpha`: Level of the Gaussian bands, `1 - alpha` coverage.
  - `levels`: Confidence levels of the exceedance rates.

# Validation

  - `cfers` is not empty. An `IsEmptyError` is thrown otherwise.
  - `names` has one entry per evaluation when given. A `DimensionMismatch` is thrown otherwise.
  - `0 < alpha < 1`, and every level lies in `(0, 1)`. A `DomainError` is thrown otherwise.

# Returns

  - `summary::CovarianceForecastSummaryResult`: The columnar summary.

# Related

  - [`CovarianceForecastSummaryResult`](@ref)
  - [`CovarianceForecastEvaluationResult`](@ref)
  - [`covariance_forecast_evaluation`](@ref)
  - [`covariance_forecast_compare`](@ref)
  - [`target_dof`](@ref)
"""
function covariance_forecast_summary(cfers::AbstractVector{<:CovarianceForecastEvaluationResult};
                                     names::Option{<:AbstractVector} = nothing,
                                     alpha::Real = 0.05, levels = (0.95, 0.99))
    @argcheck(!isempty(cfers), IsEmptyError("`cfers` cannot be empty"))
    @argcheck(zero(alpha) < alpha < one(alpha),
              DomainError(alpha, "`alpha` must lie in (0, 1)"))
    @argcheck(all(q -> zero(q) < q < one(q), levels),
              DomainError(levels, "every level must lie in (0, 1)"))
    n = length(cfers)
    nms = isnothing(names) ? ["forecast_$(i)" for i in 1:n] : names
    @argcheck(length(nms) == n,
              DimensionMismatch("`names` has $(length(nms)) entries and `cfers` $(n) evaluations."))
    z = Distributions.cquantile(Distributions.Normal(), alpha / 2)
    Tf = promote_type(eltype(cfers[1].mahalanobis_ratio), typeof(z))
    mm = Vector{Tf}(undef, n)
    mmed = similar(mm)
    m5 = similar(mm)
    m95 = similar(mm)
    mlo = similar(mm)
    mhi = similar(mm)
    dm = similar(mm)
    dmed = similar(mm)
    d5 = similar(mm)
    d95 = similar(mm)
    dlo = similar(mm)
    dhi = similar(mm)
    bs = similar(mm)
    b5 = similar(mm)
    b25 = similar(mm)
    b75 = similar(mm)
    b95 = similar(mm)
    ql = similar(mm)
    fr = similar(mm)
    pql = similar(mm)
    ex = Matrix{Tf}(undef, n, length(levels))
    ns = Vector{Int}(undef, n)
    np = Vector{Int}(undef, n)
    for (i, cfer) in enumerate(cfers)
        m = cfer.mahalanobis_ratio
        dof = target_dof.(Ref(cfer.target), cfer.n_valid, cfer.horizon)
        sdof = target_step_dof.(Ref(cfer.target), cfer.horizon)
        dbar = [Statistics.mean(filter(isfinite, view(cfer.diagonal_ratio, t, :)))
                for t in axes(cfer.diagonal_ratio, 1)]
        mm[i] = LinearAlgebra.dot(dof, m) / sum(dof)
        mmed[i] = Statistics.median(m)
        m5[i] = Statistics.quantile(m, 0.05)
        m95[i] = Statistics.quantile(m, 0.95)
        hw = z * sqrt(2 / sum(dof))
        mlo[i] = 1 - hw
        mhi[i] = 1 + hw
        dm[i] = LinearAlgebra.dot(sdof, dbar) / sum(sdof)
        dmed[i] = Statistics.median(dbar)
        d5[i] = Statistics.quantile(dbar, 0.05)
        d95[i] = Statistics.quantile(dbar, 0.95)
        hwd = z * sqrt(2 / sum(sdof))
        dlo[i] = 1 - hwd
        dhi[i] = 1 + hwd
        b = vec(Statistics.std(cfer.standardised_return; dims = 1))
        bs[i] = Statistics.median(b)
        b5[i] = Statistics.quantile(b, 0.05)
        b25[i] = Statistics.quantile(b, 0.25)
        b75[i] = Statistics.quantile(b, 0.75)
        b95[i] = Statistics.quantile(b, 0.95)
        ql[i] = Statistics.mean(cfer.qlike)
        fr[i] = Statistics.mean(cfer.frobenius)
        pql[i] = Statistics.median(vec(Statistics.mean(cfer.portfolio_qlike; dims = 1)))
        stat = dof .* m
        for (j, q) in enumerate(levels)
            thr = Distributions.quantile.(Distributions.Chisq.(dof), q)
            ex[i, j] = Statistics.mean(stat .> thr)
        end
        ns[i] = length(m)
        np[i] = size(cfer.standardised_return, 2)
    end
    return CovarianceForecastSummaryResult(nms, mm, mmed, m5, m95, mlo, mhi, dm, dmed, d5,
                                           d95, dlo, dhi, bs, b5, b25, b75, b95, ql, fr,
                                           pql, levels, ex, ns, np, alpha)
end
function covariance_forecast_summary(cfer::CovarianceForecastEvaluationResult; kwargs...)
    return covariance_forecast_summary([cfer]; kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

The Diebold–Mariano–West comparison of two covariance forecasts, one row per loss.

`CovarianceForecastComparisonResult` is what [`covariance_forecast_compare`](@ref) returns. It is columnar on an axis whose entries are the losses compared: the whole-matrix QLIKE, the Frobenius loss, and the portfolio QLIKE of each test portfolio. A positive mean difference says the first forecast lost more.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CovarianceForecastComparisonResult(
        names, mean_difference, variance, z, p, lags, n_steps
    ) -> CovarianceForecastComparisonResult

Arguments correspond to the struct's fields, in the order they are declared. The type is a Result, so [`covariance_forecast_compare`](@ref) builds it and a caller reads it; there is no keyword constructor, and the type validates nothing of its own.

# Related

  - [`covariance_forecast_compare`](@ref)
  - [`CovarianceForecastEvaluationResult`](@ref)
"""
@concrete struct CovarianceForecastComparisonResult <: AbstractResult
    """
    Name of each loss compared: `"qlike"`, `"frobenius"`, then `"portfolio_qlike_k"` for each test portfolio `k`.
    """
    names
    """
    Mean over the steps of the first forecast's loss less the second's. One entry per loss.
    """
    mean_difference
    """
    Newey–West long-run variance of the per-step loss difference, at `lags` lags. One entry per loss.
    """
    variance
    """
    Diebold–Mariano–West statistic, asymptotically standard normal under equal expected loss. One entry per loss.
    """
    z
    """
    Two-sided p-value of the statistic against the standard normal. One entry per loss.
    """
    p
    """
    Number of lags of the Bartlett kernel.
    """
    lags
    """
    Number of steps compared.
    """
    n_steps
end
"""
    covariance_forecast_compare(a::CovarianceForecastEvaluationResult, b::CovarianceForecastEvaluationResult;
                                lags::Integer = maximum(a.horizon) - 1) -> CovarianceForecastComparisonResult

Test whether two covariance forecasts differ in expected loss, per loss.

Two mean losses side by side invite exactly the reading a loss on a proxy cannot bear — the level of a loss on a proxy is not a calibration reading — so the comparison tests the difference instead. For each loss the per-step difference is a series whose mean is tested against zero with a long-run variance that absorbs the overlap of consecutive steps: the Bartlett kernel at ``h - 1`` lags by default, the overlap of steps that share observations.

# Mathematical definition

```math
\\begin{align}
\\delta_t &= L^{A}_t - L^{B}_t\\,, \\\\
\\bar{\\delta} &= \\frac{1}{M} \\sum_{t=1}^{M} \\delta_t\\,, \\\\
\\hat{\\omega}^2 &= \\hat{\\gamma}_0 + 2 \\sum_{k=1}^{\\ell} \\left(1 - \\frac{k}{\\ell + 1}\\right) \\hat{\\gamma}_k\\,, \\\\
\\mathrm{DMW} &= \\frac{\\sqrt{M}\\, \\bar{\\delta}}{\\sqrt{\\hat{\\omega}^2}}\\,.
\\end{align}
```

Where:

  - ``\\delta_t``: Difference of the losses of forecasts ``A`` and ``B`` at step ``t``.
  - ``\\bar{\\delta}``: Mean loss difference over the walk-forward.
  - ``\\hat{\\omega}^2``: Long-run variance of ``\\delta_t``, the Bartlett-kernel estimate with ``\\ell`` lags.
  - ``\\hat{\\gamma}_k``: Sample autocovariance of ``\\delta_t`` at lag ``k``.
  - ``\\ell``: Number of lags, ``h - 1`` by default.
  - $(math_dict[:h_step])
  - $(math_dict[:M_steps])
  - ``\\mathrm{DMW}``: Diebold–Mariano–West statistic.

Under equal expected loss, ``\\mathrm{DMW}`` is asymptotically standard normal. A positive value says forecast ``A`` lost more than forecast ``B``. Two identical forecasts have ``\\bar{\\delta} = 0`` and ``\\hat{\\omega}^2 = 0``, so their statistic is not a number.

# Arguments

  - `a`: The first evaluation.
  - `b`: The second evaluation.
  - `lags`: Number of lags of the Bartlett kernel.

# Validation

  - `a.dates == b.dates` and `a.horizon == b.horizon`. An `ArgumentError` is thrown otherwise: two evaluations over different steps compare nothing.
  - `a` and `b` carry the same number of test portfolios. A `DimensionMismatch` is thrown otherwise.
  - `0 <= lags < n_steps`. A `DomainError` is thrown otherwise.

# Returns

  - `cmp::CovarianceForecastComparisonResult`: One row per loss.

# Related

  - [`CovarianceForecastComparisonResult`](@ref)
  - [`CovarianceForecastEvaluationResult`](@ref)
  - [`covariance_forecast_evaluation`](@ref)
  - [`covariance_forecast_summary`](@ref)
  - [`newey_west_variance`](@ref)
"""
function covariance_forecast_compare(a::CovarianceForecastEvaluationResult,
                                     b::CovarianceForecastEvaluationResult;
                                     lags::Integer = maximum(a.horizon) - 1)
    @argcheck(isequal(a.dates, b.dates) && a.horizon == b.horizon,
              ArgumentError("the two evaluations do not share their steps: a comparison reads a per-step loss difference, so both must be run over one walk-forward, with the same dates and the same horizon at every step. Run both forecasts through `covariance_forecast_evaluation` with the same `rd` and `cv`."))
    P = size(a.portfolio_qlike, 2)
    @argcheck(size(b.portfolio_qlike, 2) == P,
              DimensionMismatch("the first evaluation carries $(P) test portfolio(s) and the second $(size(b.portfolio_qlike, 2)); hand both the same `w`."))
    M = length(a.qlike)
    @argcheck(0 <= lags < M, DomainError(lags, "`lags` must lie in [0, n_steps)"))
    names = vcat(["qlike", "frobenius"], ["portfolio_qlike_$(k)" for k in 1:P])
    series = vcat([a.qlike .- b.qlike, a.frobenius .- b.frobenius],
                  [view(a.portfolio_qlike, :, k) .- view(b.portfolio_qlike, :, k)
                   for k in 1:P])
    md = [Statistics.mean(s) for s in series]
    v = [newey_west_variance(s, lags) for s in series]
    z = sqrt(M) .* md ./ sqrt.(v)
    p = 2 .* Distributions.ccdf.(Distributions.Normal(), abs.(z))
    return CovarianceForecastComparisonResult(names, md, v, z, p, lags, M)
end
"""
    newey_west_variance(d::VecNum, lags::Integer)

Bartlett-kernel long-run variance of a series, at `lags` lags.

``\\hat{\\omega}^2 = \\hat{\\gamma}_0 + 2 \\sum_{k=1}^{\\ell} (1 - k / (\\ell + 1))\\, \\hat{\\gamma}_k``, with ``\\hat{\\gamma}_k`` the sample autocovariance at lag ``k`` about the sample mean, divided by the length of the series.

# Arguments

  - `d`: The series.
  - `lags`: Number of lags, ``\\ell``.

# Returns

  - `omega2::Number`: The long-run variance.

# Related

  - [`covariance_forecast_compare`](@ref)
"""
function newey_west_variance(d::VecNum, lags::Integer)
    M = length(d)
    e = d .- Statistics.mean(d)
    gamma(k) = LinearAlgebra.dot(view(e, (k + 1):M), view(e, 1:(M - k))) / M
    omega2 = gamma(0)
    for k in 1:lags
        omega2 += 2 * (1 - k / (lags + 1)) * gamma(k)
    end
    return omega2
end
"""
    covariance_forecast_portfolio(cfer::CovarianceForecastEvaluationResult, rd::ReturnsResult,
                                  w::Option{<:VecNum_VecVecNum}) -> NamedTuple

Re-project the stored forecasts of an evaluation on a new test portfolio.

The verb a request for the forecasts buys: with `store_forecasts = true` the Result holds every step's forecast and the location it was centred on, so a new portfolio is scored without rerunning the loop. Each step reads its stored forecast and location, the test rows `rd.X[cfer.test_idx[i], :]`, and the new `w`, through [`covariance_forecast_step`](@ref), and answers the two portfolio diagnostics. Handing the evaluation's own `w` reproduces its columns.

# Arguments

  - `cfer`: The evaluation, run with `store_forecasts = true`.
  - $(arg_dict[:rd])
  - `w`: The test portfolios on the full universe, as [`covariance_forecast_evaluation`](@ref) takes them.

# Validation

  - `cfer.sigma` is not `nothing`. An `ArgumentError` is thrown otherwise, pointing at `store_forecasts`.
  - `rd.X` is not `nothing`. An `IsNothingError` is thrown otherwise.

# Returns

  - `proj::NamedTuple`: `standardised_return` and `portfolio_qlike`, each `steps × portfolios`.

# Related

  - [`covariance_forecast_evaluation`](@ref)
  - [`CovarianceForecastEvaluationResult`](@ref)
  - [`covariance_forecast_step`](@ref)
"""
function covariance_forecast_portfolio(cfer::CovarianceForecastEvaluationResult,
                                       rd::ReturnsResult, w::Option{<:VecNum_VecVecNum})
    @argcheck(!isnothing(cfer.sigma),
              ArgumentError("the evaluation holds no forecasts to re-project: run `covariance_forecast_evaluation` with `store_forecasts = true`, which keeps every step's forecast and the location it was centred on."))
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    steps = [covariance_forecast_step(cfer.sigma[i], view(rd.X, cfer.test_idx[i], :),
                                      cfer.location[i], w, cfer.target)
             for i in eachindex(cfer.sigma)]
    P = length(steps[1].standardised_return)
    b = Matrix{eltype(steps[1].standardised_return)}(undef, length(steps), P)
    pq = Matrix{eltype(steps[1].portfolio_qlike)}(undef, length(steps), P)
    for (i, s) in enumerate(steps)
        b[i, :] .= s.standardised_return
        pq[i, :] .= s.portfolio_qlike
    end
    return (; standardised_return = b, portfolio_qlike = pq)
end

export covariance_forecast_summary, CovarianceForecastSummaryResult,
       covariance_forecast_compare, CovarianceForecastComparisonResult,
       covariance_forecast_portfolio
