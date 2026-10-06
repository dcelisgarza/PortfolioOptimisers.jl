"""
    exposure_forward_mean_return(R::MatNum, horizon::Integer)

Return the forward mean asset return of every observation but the last `horizon` of them.

The information coefficient scores an exposure against what the asset earned after the exposure was known, and this verb builds that target. An observation whose forward window carries no finite return gets `NaN`, and the verb averages a window that is part finite over the finite part alone.

# Mathematical definition

```math
\\begin{align}
y_{ti} &= \\frac{1}{|\\mathcal{H}_{ti}|} \\sum_{h \\in \\mathcal{H}_{ti}} r_{t+h,i}\\,.
\\end{align}
```

Where:

  - ``y_{ti}``: Forward mean return of asset ``i`` at observation ``t``, `NaN` when ``\\mathcal{H}_{ti}`` is empty.
  - ``r_{ti}``: Return of asset ``i`` at observation ``t``.
  - ``\\mathcal{H}_{ti}``: The offsets ``1 \\le h \\le H`` at which ``r_{t+h,i}`` is finite.
  - ``H``: Forward window, in observations.

# Arguments

  - `R`: Asset return history `observations × assets`.
  - `horizon`: Forward window, in observations.

# Validation

  - `size(R, 1) > horizon`.

# Returns

  - `y::MatNum`: Forward mean return `(observations - horizon) × assets`. Row `t` averages the returns of observations `t + 1` to `t + horizon`.

# Related

  - [`exposure_ic`](@ref)
"""
function exposure_forward_mean_return(R::MatNum, horizon::Integer)
    T, N = size(R)
    @argcheck(T > horizon,
              DimensionMismatch("R ($T observations) must carry more observations than horizon ($horizon)"))
    Tf = float_if_integer(real(eltype(R)))
    P = T - horizon
    y = Matrix{Tf}(undef, P, N)
    for i in 1:N, t in 1:P
        s = zero(Tf)
        c = 0
        for h in 1:horizon
            v = R[t + h, i]
            if isfinite(v)
                s += Tf(v)
                c += 1
            end
        end
        y[t, i] = c > 0 ? s / c : Tf(NaN)
    end
    return y
end
"""
    exposure_ic(B::Arr3Num, R::MatNum, w::Option{<:MatNum} = nothing;
                horizon::Integer = 1, rank::Bool = true,
                ties::Symbol = :average) -> Matrix{<:Real}
    exposure_ic(csfm::CrossSectionalFactorModel, rd::Option{<:ReturnsResult} = nothing;
                horizon::Integer = 1, rank::Bool = true, reduced::Bool = false,
                ties::Symbol = :average) -> FactorDiagnosticResult

Return the information coefficient of every factor exposure, one row per pair of observations.

The information coefficient is the cross-sectional correlation between the exposure known at an observation and the mean asset return over the observations that follow it. It scores the exposure as a forecast of the return.

A risk factor with an information coefficient near zero is not a bad risk factor. The purpose of a risk factor is to explain the covariance and not to predict the mean, so read [`exposure_stability`](@ref) and the variance the factor contributes before you judge one, and never the information coefficient alone.

The block method reads `rw` for the weights of the Pearson form, and answers on the raw factor axis. `reduced` maps the exposures through the family re-basis of the block first, and then the answer is on the reduced axis. Without `rd`, it reconstructs the asset returns as ``\\mathbf{B}_{t-\\ell} \\boldsymbol{f}_{t} + \\boldsymbol{\\varepsilon}_{t}``. The reconstruction reads every factor of an asset, so an exposure that is not finite on one factor makes the return of that asset `NaN`, and the asset leaves the score of every factor. One sparse factor can thus blank the coefficient of every other factor. With `rd`, the method scores each factor against the asset returns `rd.X` on the rows of the block, so each factor reads every asset whose own exposure and forward return are finite. The form with `rd` reads no fit, and it scores the first ``\\ell - 1`` rows of the block too, which the reconstruction trims. On every cell that the fit regressed, the reconstruction gives back `rd.X`, the part the observed factors carry included, unless the [`CrossSectionalFactorPrior`](@ref) read named net returns under its `lx`. So the two forms read the same return wherever the reconstruction is finite, and they differ only by the assets that the reconstruction loses.

# Mathematical definition

```math
\\begin{align}
\\mathrm{IC}_{tk} &= \\rho \\left( \\mathbf{B}_{t \\cdot k}, \\boldsymbol{y}_{t}, \\boldsymbol{u}_{t} \\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{IC}_{tk}``: Information coefficient of factor ``k`` at observation ``t``.
  - $(math_dict[:B_tk_cs])
  - ``\\boldsymbol{y}_{t}``: Forward mean asset return of observation ``t``, over the finite returns of the ``H`` observations that follow it. [`exposure_forward_mean_return`](@ref) defines it.
  - $(math_dict[:u_t_cs])
  - ``\\rho``: The rank correlation of [`cs_spearman_correlation`](@ref) when `rank`, which reads no weight, and the weighted correlation of [`cs_weighted_correlation`](@ref) otherwise.
  - ``H``: Forward window, in observations.

A one-hot exposure, such as an industry, is a block of `0` and a block of `1`, so every asset of it is in a tie. Under the default `ties = :average`, its rank coefficient compares the forward returns of the two blocks. Under `ties = :ordinal`, the order of the assets inside each block sets most of it.

# Algorithm

 1. Resolve the weight history with [`exposure_weights`](@ref).
 2. Build the forward mean return with [`exposure_forward_mean_return`](@ref).
 3. Correlate each exposure against it, one observation at a time, with [`cs_spearman_correlation`](@ref) when `rank` and [`cs_weighted_correlation`](@ref) otherwise.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, unlagged.
  - `R`: Asset return history `observations × assets`, on the observation axis of `B`.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights. The rank form reads no weights.
  - `horizon`: Forward window, in observations.
  - `rank`: Take the rank correlation when `true`, and the weighted correlation otherwise.
  - `csfm`: A cross-sectional factor model block.
  - $(arg_dict[:cs_ic_rd]) [`exposure_ic_returns`](@ref) takes the rows of the block from it.
  - `reduced`: Map the exposures through the family re-basis of the block before the correlation.
  - $(arg_dict[:cs_ties]) The weighted correlation reads no rank, so it ignores `ties`.

# Validation

  - `!isempty(B)`.
  - `size(R) == (size(B, 1), size(B, 2))`.
  - `horizon >= 1` and `size(B, 1) > horizon`.
  - `csfm.Ms` is not `nothing`, and without `rd` neither is `csfm.csr`. Otherwise the verb raises an `IsNothingError` that names the field.
  - With `rd`, the rules of [`exposure_ic_returns`](@ref).
  - The rules of [`cs_ranks`](@ref), when `rank`.

# Returns

  - `ic::Matrix{<:Real}`: `(observations - horizon) × factors`, on the method over histories. Row `t` scores the exposures of observation `t` against the returns that follow it.
  - `r::FactorDiagnosticResult`: On the block method, the same matrix in `X`, with the names and the family labels of its factor axis: the raw axis, or under `reduced` the whole reduced axis that [`cs_diagnostic_factor_names`](@ref) maps with two arguments. With `rd`, its rows are the rows of the block. Without it, they start at row ``\\max(1, \\ell)`` of the block, as [`exposure_ic_data`](@ref) states. The block method returned the bare matrix in earlier releases, so a caller that indexed it now reads `r.X`.

# Related

  - [`exposure_ic_summary`](@ref)
  - [`exposure_forward_mean_return`](@ref)
  - [`cs_spearman_correlation`](@ref)
  - [`cs_weighted_correlation`](@ref)
  - [`exposure_stability`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function exposure_ic(B::Arr3Num, R::MatNum, w::Option{<:MatNum} = nothing;
                     horizon::Integer = 1, rank::Bool = true, ties::Symbol = :average)
    u = exposure_weights(B, w)
    T, N, K = size(B)
    @argcheck(size(R, 1) == T && size(R, 2) == N,
              DimensionMismatch("R ($(size(R, 1))×$(size(R, 2))) must match B ($T×$N on its first two axes)"))
    @argcheck(horizon >= 1, DomainError(horizon, "horizon must be >= 1"))
    y = exposure_forward_mean_return(R, horizon)
    Tf = typeof(sqrt(one(float_if_integer(promote_type(real(eltype(B)), real(eltype(y)),
                                                       real(eltype(u)))))))
    P = T - horizon
    ic = Matrix{Tf}(undef, P, K)
    for k in 1:K, t in 1:P
        a = view(B, t, :, k)
        b = view(y, t, :)
        ic[t, k] = if rank
            Tf(cs_spearman_correlation(a, b; ties = ties))
        else
            Tf(cs_weighted_correlation(a, b, view(u, t, :)))
        end
    end
    return ic
end
function exposure_ic(csfm::CrossSectionalFactorModel, rd::Option{<:ReturnsResult} = nothing;
                     horizon::Integer = 1, rank::Bool = true, reduced::Bool = false,
                     ties::Symbol = :average)
    B, R, w = exposure_ic_data(csfm, rd, reduced)
    ic = exposure_ic(B, R, w; horizon = horizon, rank = rank, ties = ties)
    if reduced
        return FactorDiagnosticResult(ic, cs_diagnostic_factor_names(csfm.fcb, csfm.nf),
                                      cs_diagnostic_factor_names(csfm.fcb, csfm.fam), (2,))
    end
    return FactorDiagnosticResult(ic, csfm.nf, csfm.fam, (2,))
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the summary of an information coefficient series per factor, with the names and the families of the factors.

[`exposure_ic_summary`](@ref) returns it when it reads a [`FactorDiagnosticResult`](@ref) or a block. The method over a bare matrix returns a `NamedTuple` of the same five vectors, and a caller reads a field of either by the same name. [`port_opt_view`](@ref) selects the factors of the summary by position, by name, or by family with a [`LabelGroup`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExposureICSummaryResult(
        mean_ic, std_ic, ic_ir, t_stat, hit_rate, nf, fam
    ) -> ExposureICSummaryResult

The arguments are the fields, in the order of their declaration. The type is a Result, so [`exposure_ic_summary`](@ref) builds it and a caller reads it. It has no keyword constructor, and it checks none of its values.

# Related

  - [`exposure_ic_summary`](@ref)
  - [`exposure_ic`](@ref)
  - [`FactorDiagnosticResult`](@ref)
  - [`port_opt_view`](@ref)
"""
@concrete struct ExposureICSummaryResult <: AbstractResult
    """
    Mean of the information coefficient over the observations at which it is finite, one entry per factor.
    """
    mean_ic
    """
    Standard deviation of the information coefficient over the same observations, one entry per factor.
    """
    std_ic
    """
    Information ratio, the mean over the standard deviation, one entry per factor.
    """
    ic_ir
    """
    T-statistic of the mean, with a long-run standard error, one entry per factor.
    """
    t_stat
    """
    Fraction of the same observations at which the information coefficient is positive, one entry per factor.
    """
    hit_rate
    """
    $(field_dict[:fd_nf])
    """
    nf
    """
    $(field_dict[:fd_fam])
    """
    fam
end
"""
    port_opt_view(s::ExposureICSummaryResult, i, args...)

Return a view of an [`ExposureICSummaryResult`](@ref) that keeps only the factors that `i` selects.

# Algorithm

 1. Cut every field to the selected factors with [`factor_table_fields`](@ref).
 2. Build a new [`ExposureICSummaryResult`](@ref) from the views.

# Arguments

  - `s`: A summary of an information coefficient series.
  - `i`: The factor index: positions, a range, a `Colon`, a vector of names, or a [`LabelGroup`](@ref).
  - `args...`: Additional positional arguments (ignored).

# Validation

  - The rules of [`factor_axis_positions`](@ref).

# Returns

  - `s::ExposureICSummaryResult`: The summary of the selected factors.

# Related

  - [`ExposureICSummaryResult`](@ref)
  - [`factor_table_fields`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(s::ExposureICSummaryResult, i, args...)::ExposureICSummaryResult
    return ExposureICSummaryResult(factor_table_fields(s, i)...)
end
"""
    exposure_ic_summary(ic::MatNum; lags::Integer = 0) -> NamedTuple
    exposure_ic_summary(ic::FactorDiagnosticResult; lags::Integer = 0) -> ExposureICSummaryResult
    exposure_ic_summary(csfm::CrossSectionalFactorModel,
                        rd::Option{<:ReturnsResult} = nothing; horizon::Integer = 1,
                        rank::Bool = true, reduced::Bool = false,
                        ties::Symbol = :average) -> ExposureICSummaryResult

Return the summary of an information coefficient series, one entry per factor.

The mean states the average score, the standard deviation states how much the score moves, and their ratio states the score per unit of movement. The t-statistic states whether the mean is far enough from zero to believe over the observations that carried a score, and the hit rate states how often the score was positive. All five read the observations that carried a score. An observation whose score is `NaN` is one at which nothing was measured, not a miss, so it is in no denominator here, as it is in none of the library's other summaries. The coverage of the series answers the separate question of how often the series had no score.

The standard error of the t-statistic is the long-run one of [`exposure_ic_factor_summary`](@ref), over `lags` autocovariances. [`exposure_ic`](@ref) scores every observation against a window of `horizon` observations, so each row shares returns with the `horizon - 1` rows on each side of it, and the block method derives `lags` as `horizon - 1`. The bare method takes `lags` from the caller. Its default of `0` is the independent case, which is the case of a series scored at a stride of its window.

# Mathematical definition

```math
\\begin{align}
\\overline{\\mathrm{IC}}_{k} &= \\frac{1}{|\\mathcal{T}_{k}|} \\sum_{t \\in \\mathcal{T}_{k}} \\mathrm{IC}_{tk}\\,, \\\\
s_{k}^{2} &= \\frac{1}{|\\mathcal{T}_{k}| - 1} \\sum_{t \\in \\mathcal{T}_{k}} \\left( \\mathrm{IC}_{tk} - \\overline{\\mathrm{IC}}_{k} \\right)^{2}\\,, \\\\
\\mathrm{IR}_{k} &= \\frac{\\overline{\\mathrm{IC}}_{k}}{s_{k}}\\,, \\\\
t_{k} &= \\frac{\\overline{\\mathrm{IC}}_{k}}{\\sigma_{k}} \\sqrt{\\left| \\mathcal{T}_{k} \\right|}\\,, \\\\
\\mathrm{hit}_{k} &= \\frac{1}{|\\mathcal{T}_{k}|} \\sum_{t \\in \\mathcal{T}_{k}} \\mathbb{1}\\left[\\mathrm{IC}_{tk} > 0\\right]\\,.
\\end{align}
```

Where:

  - ``\\mathrm{IC}_{tk}``: Information coefficient of factor ``k`` at observation ``t``.
  - ``\\mathcal{T}_{k}``: The observations at which it is finite.
  - ``\\overline{\\mathrm{IC}}_{k}``: Its mean over ``\\mathcal{T}_{k}``.
  - ``s_{k}``: Its standard deviation over ``\\mathcal{T}_{k}``.
  - ``\\mathrm{IR}_{k}``: Its information ratio.
  - ``\\sigma_{k}``: Its long-run standard deviation over ``\\mathcal{T}_{k}``, which [`exposure_ic_factor_summary`](@ref) defines and which is ``s_{k}`` at `lags = 0`.
  - ``t_{k}``: Its t-statistic.
  - ``\\mathrm{hit}_{k}``: Its hit rate.

# Arguments

  - `ic`: Information coefficient series `pairs × factors`, or the [`FactorDiagnosticResult`](@ref) of [`exposure_ic`](@ref), whose labels the answer keeps.
  - `lags`: Number of autocovariances the t-statistic's standard error reads. It is one less than the number of rows a forward window spans, `horizon - 1` for a series scored at every observation.
  - `csfm`: A cross-sectional factor model block.
  - $(arg_dict[:cs_ic_rd]) [`exposure_ic`](@ref) states how each form reads the returns.
  - `horizon`: Forward window, in observations.
  - `rank`: Take the rank correlation when `true`, and the weighted correlation otherwise.
  - `reduced`: Map the exposures through the family re-basis of the block before the correlation.
  - $(arg_dict[:cs_ties]) The weighted correlation reads no rank, so it ignores `ties`.

# Validation

  - `!isempty(ic)`.
  - `lags >= 0`. Raises a `DomainError`.
  - A [`FactorDiagnosticResult`](@ref) is a series `pairs × factors`: `X` is a matrix and `dims == (2,)`. Raises an `ArgumentError`.

# Returns

  - `summary::NamedTuple`: `(; mean_ic, std_ic, ic_ir, t_stat, hit_rate)`, each one entry per factor, on the method over a matrix.
  - `s::ExposureICSummaryResult`: On the other two methods, the same five vectors, with the names and the family labels of the factor axis of the series. The block method returned the `NamedTuple` in earlier releases, and a field of the Result has the name of the field of the `NamedTuple`.

# Related

  - [`exposure_ic`](@ref)
  - [`exposure_ic_factor_summary`](@ref)
  - [`forecast_factor_correlation`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function exposure_ic_summary(ic::MatNum; lags::Integer = 0)
    @argcheck(!isempty(ic), IsEmptyError("ic cannot be empty"))
    @argcheck(lags >= zero(lags), DomainError(lags, "lags must be >= 0"))
    P, K = size(ic)
    Tm = float_if_integer(real(eltype(ic)))
    Ts = typeof(sqrt(one(Tm)))
    mean_ic = Vector{Tm}(undef, K)
    std_ic = Vector{Ts}(undef, K)
    ic_ir = Vector{Ts}(undef, K)
    t_stat = Vector{Ts}(undef, K)
    hit_rate = Vector{Tm}(undef, K)
    for k in 1:K
        m = exposure_ic_factor_summary(ic, k; lags = lags)
        mean_ic[k] = m.mean_ic
        std_ic[k] = m.std_ic
        ic_ir[k] = m.ic_ir
        t_stat[k] = m.t_stat
        hit_rate[k] = m.hit_rate
    end
    return (; mean_ic = mean_ic, std_ic = std_ic, ic_ir = ic_ir, t_stat = t_stat,
            hit_rate = hit_rate)
end
"""
    exposure_ic_factor_summary(ic::MatNum, k::Integer; lags::Integer = 0)

Return the five summary numbers of one factor's information coefficient series.

The mean, the standard deviation and the hit rate read the observations at which the coefficient is finite, so an observation with no coefficient is in no denominator and the three agree on their sample. A series with no finite coefficient has no hit rate. The ratio has no answer where either of its two terms has none, and none where the standard deviation is zero. The t-statistic is the mean over the standard error of the mean, so it says whether the mean is far enough from zero to believe over the evidence there was.

[`forecast_ic_summary`](@ref) calls this function too, so a coefficient of a factor exposure and a coefficient of a Return Forecast have the same summary.

# The standard error reads the overlap

A coefficient scored against a forward window that is longer than the stride between two scores reads the same returns at consecutive rows. The rows are then not independent, and the standard error of their mean is not ``s / \\sqrt{n}``. The rows that overlap a given row are the ``L`` on each side of it, where ``L`` is one less than the number of strides a window spans. Under no skill, every autocovariance beyond ``L`` is zero.

The autocovariances up to ``L`` are not zero when the scored quantity keeps its ranking of the assets from one row to the next. An exposure is persistent in this sense, so its series is a moving average of order ``L``. A Return Forecast that draws a new ranking at every row gives autocovariances near zero at every lag, and there the plain statistic ``\\mathrm{IR} \\sqrt{n}`` does not overstate the evidence. The long-run variance then adds sampling noise to the standard error and moves it little.

The standard error therefore reads the long-run variance, the variance plus twice the first ``L`` autocovariances, with no taper. The window and the stride fix the order, so the function does not estimate it from the series. A tapered estimate at the same order under-weights autocovariances that the window and the stride imply, and still overstates the statistic. For a persistent ranking under no skill, the autocovariances fall linearly across the window. The Bartlett taper at that order then keeps two thirds of their sum in the limit of a long window, so the statistic is too large by the root of three halves there, and by less at a short window.

The autocovariances read the rows at their positions in `ic`. A row whose coefficient is `NaN` is in no pair, and the lag between two rows is their distance in the series, not in its finite subsequence. Every autocovariance divides by the same ``n - 1`` as the variance, so at `lags = 0` the standard error is exactly ``s / \\sqrt{n}`` and the statistic is ``\\mathrm{IR} \\sqrt{n}``.

The long-run variance is a sum of signed terms, and a short series can sum it to a number that is not positive. The t-statistic is `NaN` there, as it is where the standard deviation is zero. The function does not clamp it, because a clamped value would report a certainty that the series does not carry.

# Mathematical definition

```math
\\begin{align}
\\gamma_{j} &= \\frac{1}{n - 1} \\sum_{t,\\, t + j \\in \\mathcal{T}} \\left( \\mathrm{IC}_{t} - \\overline{\\mathrm{IC}} \\right) \\left( \\mathrm{IC}_{t + j} - \\overline{\\mathrm{IC}} \\right)\\,, \\\\
\\sigma^{2} &= \\gamma_{0} + 2 \\sum_{j = 1}^{L} \\gamma_{j}\\,, \\\\
t &= \\frac{\\overline{\\mathrm{IC}}}{\\sigma} \\sqrt{n}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{IC}_{t}``: Coefficient of factor ``k`` at row ``t``.
  - ``\\mathcal{T}``: The rows at which it is finite.
  - ``n``: The number of rows in ``\\mathcal{T}``.
  - ``\\overline{\\mathrm{IC}}``: Mean of the coefficient over ``\\mathcal{T}``.
  - ``\\gamma_{j}``: Autocovariance at lag ``j``. ``\\gamma_{0}`` is the variance ``s^{2}``.
  - ``\\sigma``: Long-run standard deviation.
  - ``t``: The t-statistic.
  - ``L``: `lags`.

# Arguments

  - `ic`: Information coefficient series `pairs × factors`.
  - `k`: Position of the factor.
  - `lags`: Number of autocovariances the standard error reads, `0` for independent rows. [`forecast_ic_lags`](@ref) derives it from a window and a stride.

# Validation

  - `lags >= 0`. Raises a `DomainError`.

# Returns

  - `m::NamedTuple`: `(; mean_ic, std_ic, ic_ir, t_stat, hit_rate)`, five numbers.

# Related

  - [`exposure_ic_summary`](@ref)
  - [`forecast_ic_summary`](@ref)
  - [`forecast_ic_lags`](@ref)
"""
function exposure_ic_factor_summary(ic::MatNum, k::Integer; lags::Integer = 0)
    @argcheck(lags >= zero(lags), DomainError(lags, "lags must be >= 0"))
    P = size(ic, 1)
    Tm = float_if_integer(real(eltype(ic)))
    Ts = typeof(sqrt(one(Tm)))
    n = 0
    s = zero(Tm)
    h = 0
    for t in 1:P
        v = ic[t, k]
        if isfinite(v)
            n += 1
            s += Tm(v)
            h += v > 0
        end
    end
    m = n > 0 ? s / n : Tm(NaN)
    q = zero(Tm)
    for t in 1:P
        v = ic[t, k]
        if isfinite(v)
            d = Tm(v) - m
            q += d * d
        end
    end
    sd = n > 1 ? sqrt(q / (n - 1)) : Ts(NaN)
    ir = isfinite(m) && isfinite(sd) && sd > zero(Ts) ? m / sd : Ts(NaN)
    return (; mean_ic = m, std_ic = sd, ic_ir = ir,
            t_stat = exposure_ic_t_stat(view(ic, :, k), m, q, n, lags),
            hit_rate = n > 0 ? Tm(h) / Tm(n) : Tm(NaN))
end
"""
    exposure_ic_t_stat(c::VecNum, m::Real, q::Real, n::Integer, lags::Integer) -> Real

Return the t-statistic of one factor's information coefficient series, over its long-run standard error.

[`exposure_ic_factor_summary`](@ref) gives it the mean, the sum of squared deviations and the count, so the function reads the series once more, for the autocovariances alone. The long-run variance is the sum of squared deviations plus twice the first `lags` autocovariance sums, every one over ``n - 1``. Each pair reads its two positions in the series, so a `NaN` row is in no pair and the lag is a distance in the series. A long-run variance that is not positive, or a count of one, has no statistic.

# Arguments

  - `c`: Information coefficient series of one factor, a column of the `pairs × factors` matrix.
  - `m`: Mean of the finite entries of `c`.
  - `q`: Sum of the squared deviations of the finite entries of `c` from `m`.
  - `n`: Number of finite entries of `c`.
  - `lags`: Number of autocovariances the standard error reads.

# Returns

  - `t::Real`: `m` over the long-run standard error of the mean, or `NaN`.

# Related

  - [`exposure_ic_factor_summary`](@ref)
"""
function exposure_ic_t_stat(c::VecNum, m::Real, q::Real, n::Integer, lags::Integer)
    Tf = typeof(q)
    Ts = typeof(sqrt(one(Tf)))
    P = length(c)
    lrv = q
    for j in 1:lags, t in 1:(P - j)
        v = c[t]
        u = c[t + j]
        if isfinite(v) && isfinite(u)
            lrv += 2 * (Tf(v) - m) * (Tf(u) - m)
        end
    end
    lrsd = n > 1 && lrv > zero(Tf) ? sqrt(lrv / (n - 1)) : Ts(NaN)
    return isfinite(m) && isfinite(lrsd) ? m / lrsd * sqrt(Ts(n)) : Ts(NaN)
end
function exposure_ic_summary(ic::FactorDiagnosticResult; lags::Integer = 0)
    @argcheck(ndims(ic.X) == 2 && ic.dims == (2,),
              ArgumentError("exposure_ic_summary reads a series `pairs × factors`, whose factor axis is its second dimension. Got size(ic.X) => $(size(ic.X)) and ic.dims => $(ic.dims)"))
    (; mean_ic, std_ic, ic_ir, t_stat, hit_rate) = exposure_ic_summary(ic.X; lags = lags)
    return ExposureICSummaryResult(mean_ic, std_ic, ic_ir, t_stat, hit_rate, ic.nf, ic.fam)
end
function exposure_ic_summary(csfm::CrossSectionalFactorModel,
                             rd::Option{<:ReturnsResult} = nothing; horizon::Integer = 1,
                             rank::Bool = true, reduced::Bool = false,
                             ties::Symbol = :average)
    return exposure_ic_summary(exposure_ic(csfm, rd; horizon = horizon, rank = rank,
                                           reduced = reduced, ties = ties);
                               lags = horizon - 1)
end
"""
    exposure_ic_data(csfm::CrossSectionalFactorModel, rd::Nothing, reduced::Bool)
    exposure_ic_data(csfm::CrossSectionalFactorModel, rd::ReturnsResult, reduced::Bool)

Return the exposure history, the asset returns and the regression weights the information coefficient reads off a factor model block.

Without `rd`, the function reconstructs the asset returns from the systematic return of the fit and the residual, because the block carries no asset returns. The reconstruction exists from observation ``\\ell + 1``, so the answer starts at observation ``\\max(1, \\ell)``. The return row of that first observation is `NaN`, and no score reads it, because the information coefficient scores an exposure against the returns that follow it. With `rd`, the function reads the asset returns of the rows of the block off `rd` with [`exposure_ic_returns`](@ref). It reads no fit, so the answer covers every row of the block.

# Algorithm

 1. Refuse a block that carries no exposure history, or without `rd` no cross-sectional fit.
 2. With `rd`, read the returns of the rows of the block, and skip step 3.
 3. Reconstruct the return of each observation as ``\\mathbf{B}_{t-\\ell} \\boldsymbol{f}_{t} + \\boldsymbol{\\varepsilon}_{t}``, and write `NaN` where the lag leaves it undefined. The reconstruction runs on the axis the fit ran on: the exposures on the reduced axis when the block carries a family re-basis, and the factor returns of [`cross_sectional_factor_returns`](@ref), which put the observed factors after the estimated ones. So it gives back the returns of every asset, the part the observed factors carry included. Trim all three histories to the observations from ``\\max(1, \\ell)``.
 4. Map the exposures through the family re-basis when `reduced`.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - $(arg_dict[:cs_ic_rd])
  - `reduced`: Map the exposures through the family re-basis of the block.

# Validation

  - `csfm.Ms` is not `nothing`. Otherwise the verb raises an `IsNothingError` that names `Ms`.
  - Without `rd`, `csfm.csr` is not `nothing`. Otherwise the verb raises an `IsNothingError` that names `csr`.
  - Without `rd`, `size(csfm.Ms, 1) > csfm.lag`, and the factor returns carry the observation axis of `csfm.Ms` and the factor axis of its reduced exposures.
  - With `rd`, the rules of [`exposure_ic_returns`](@ref).

# Returns

  - `B::Arr3Num`: Exposure history of the observation axis of the answer.
  - `R::MatNum`: Asset returns of the same axis, reconstructed without `rd`.
  - `w::Option{<:MatNum}`: Regression weights of the same axis, or `nothing`.

# Related

  - [`exposure_ic`](@ref)
  - [`exposure_ic_returns`](@ref)
  - [`cs_regression_lag`](@ref)
  - [`cs_lagged_rows`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function exposure_ic_data(csfm::CrossSectionalFactorModel, ::Nothing, reduced::Bool)
    return exposure_ic_data(csfm, csfm.Ms, csfm.csr, reduced)
end
function exposure_ic_data(csfm::CrossSectionalFactorModel, rd::ReturnsResult, reduced::Bool)
    Ms = cs_diagnostic_exposures(csfm)
    R = exposure_ic_returns(csfm.idx, rd.X, size(Ms, 1))
    return exposure_ic_exposures(csfm.fcb, Ms, reduced), R, csfm.rw
end
function exposure_ic_data(::CrossSectionalFactorModel, ::Nothing,
                          ::Option{<:CrossSectionalRegression}, ::Bool)
    return throw(IsNothingError("Ms cannot be nothing: the exposure information coefficient reads the exposure history of the block"))
end
function exposure_ic_data(::CrossSectionalFactorModel, ::Arr3Num, ::Nothing, ::Bool)
    return throw(IsNothingError("csr cannot be nothing: the exposure information coefficient reads the factor returns and the residuals of the block"))
end
function exposure_ic_data(csfm::CrossSectionalFactorModel, Ms::Arr3Num,
                          csr::CrossSectionalRegression, reduced::Bool)
    lag = cs_regression_lag(csfm.lag)
    T, N = size(Ms, 1), size(Ms, 2)
    @argcheck(T > lag,
              DimensionMismatch("Ms ($T observations) must carry more observations than lag ($lag)"))
    Bd = exposure_ic_exposures(csfm.fcb, Ms, true)
    F = cross_sectional_factor_returns(csfm)
    K = size(Bd, 3)
    @argcheck(size(F, 1) == T && size(F, 2) == K,
              DimensionMismatch("the factor returns ($(size(F, 1))×$(size(F, 2))) must match Ms ($T observations) and the reduced exposures ($K factors)"))
    Tf = float_if_integer(promote_type(real(eltype(Ms)), real(eltype(F)),
                                       real(eltype(csr.eps))))
    start = max(1, lag)
    P = T - start + 1
    R = fill(Tf(NaN), P, N)
    for r in 1:P
        tau = start + r - 1
        j = tau - lag
        if j >= 1
            for i in 1:N
                s = zero(Tf)
                for k in 1:K
                    s += Tf(Bd[j, i, k]) * Tf(F[tau, k])
                end
                R[r, i] = s + Tf(csr.eps[tau, i])
            end
        end
    end
    Bf = exposure_ic_exposures(csfm.fcb, Ms, reduced)
    return Bf[start:T, :, :], R, cs_lagged_rows(csfm.rw, start:T)
end
"""
    exposure_ic_returns(idx, X::Nothing, T::Integer)
    exposure_ic_returns(idx::Nothing, X::MatNum, T::Integer)
    exposure_ic_returns(idx::VecInt, X::MatNum, T::Integer)

Return the asset returns of the rows of a factor model block, read off the returns data that the prior read.

The prior records the position of each row of the block in its returns data in `idx`, so the function reads those rows. A block that records no position, for example one built by hand, states no map from its rows to the returns. Its returns must then be the returns of its own rows, and the function reads `X` whole. It does not align the two at their tail, because a wrong alignment scores each exposure against the returns of another observation, and no check could find it. [`exposure_ic`](@ref) checks the shape of the answer against the exposure history.

# Arguments

  - `idx`: The `idx` field of a [`CrossSectionalFactorModel`](@ref), or `nothing`.
  - `X`: The asset returns of the returns data `observations × assets`, or `nothing`.
  - `T`: Number of rows of the block.

# Validation

  - `X` is not `nothing`. Otherwise the verb raises an `IsNothingError` that names `rd.X`.
  - Without `idx`, `size(X, 1) == T`. Raises a `DimensionMismatch`.
  - With `idx`, `length(idx) == T`, and every entry of `idx` is a row of `X`. Raises a `DimensionMismatch` or a `BoundsError`.

# Returns

  - `R::MatNum`: Asset returns `T × assets`, a view of `X` when `idx` is given.

# Related

  - [`exposure_ic`](@ref)
  - [`exposure_ic_data`](@ref)
  - [`ReturnsResult`](@ref)
"""
function exposure_ic_returns(::Any, ::Nothing, ::Integer)
    return throw(IsNothingError("rd.X cannot be nothing: the exposure information coefficient scores the exposures against the asset returns"))
end
function exposure_ic_returns(::Nothing, X::MatNum, T::Integer)
    @argcheck(size(X, 1) == T,
              DimensionMismatch("rd.X ($(size(X, 1)) observations) must carry the $T rows of the block, because the block records no position idx to read its rows by"))
    return X
end
function exposure_ic_returns(idx::VecInt, X::MatNum, T::Integer)
    @argcheck(length(idx) == T,
              DimensionMismatch("idx ($(length(idx)) entries) must name the $T rows of the block"))
    return view(X, idx, :)
end
"""
    exposure_ic_exposures(fcb::Nothing, Ms::Arr3Num, reduced::Bool)
    exposure_ic_exposures(fcb::FactorFamilyBasis, Ms::Arr3Num, reduced::Bool)

Return the exposure history the information coefficient scores, on the raw axis or on the reduced one.

A block that carries no family re-basis has one axis, so that case returns the history unchanged whatever `reduced` states.

# Arguments

  - `fcb`: The `fcb` field of a [`CrossSectionalFactorModel`](@ref), or `nothing`.
  - `Ms`: Unlagged exposure history `observations × assets × factors`.
  - `reduced`: Map the history through the re-basis.

# Returns

  - `B::Arr3Num`: The history, on the reduced axis when a re-basis is set and `reduced` is `true`.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`reduce_exposures`](@ref)
  - [`exposure_ic_data`](@ref)
"""
function exposure_ic_exposures(::Nothing, Ms::Arr3Num, ::Bool)
    return Ms
end
function exposure_ic_exposures(fcb::FactorFamilyBasis, Ms::Arr3Num, reduced::Bool)
    return reduced ? reduce_exposures(fcb, Ms) : Ms
end

export ExposureICSummaryResult, exposure_ic, exposure_ic_summary
