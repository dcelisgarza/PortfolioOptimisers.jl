"""
    factor_summary_finite_column(A::MatNum, k::Integer)

Return the finite entries of one column of a diagnostic series.

Every column of the summary aggregates a series over the observations, and an observation whose answer is absent contributes nothing rather than poisoning the aggregate. The absent entries are dropped once, here, so each aggregate reads a dense vector and carries no test of its own.

# Arguments

  - `A`: A series `observations × factors`.
  - `k`: Position of the factor.

# Returns

  - `v::Vector{<:Real}`: The entries of column `k` that are not `NaN`, in the order of the observations.

# Related

  - [`factor_summary_column_mean`](@ref)
  - [`factor_summary_column_median`](@ref)
  - [`factor_model_summary`](@ref)
"""
function factor_summary_finite_column(A::MatNum, k::Integer)
    Tf = float(real(eltype(A)))
    v = Vector{Tf}(undef, 0)
    for t in axes(A, 1)
        a = A[t, k]
        if !isnan(a)
            push!(v, Tf(a))
        end
    end
    return v
end
"""
    factor_summary_column_mean(A::MatNum, k::Integer)

Return the mean of the finite entries of one column of a diagnostic series.

# Arguments

  - `A`: A series `observations × factors`.
  - `k`: Position of the factor.

# Returns

  - `m::Real`: The mean, and `NaN` when the column carries no finite entry.

# Related

  - [`factor_summary_finite_column`](@ref)
  - [`factor_model_summary`](@ref)
"""
function factor_summary_column_mean(A::MatNum, k::Integer)
    v = factor_summary_finite_column(A, k)
    return isempty(v) ? convert(eltype(v), NaN) : mean(v)
end
"""
    factor_summary_column_median(A::MatNum, k::Integer)

Return the median of the finite entries of one column of a diagnostic series.

# Arguments

  - `A`: A series `observations × factors`.
  - `k`: Position of the factor.

# Returns

  - `m::Real`: The median, and `NaN` when the column carries no finite entry.

# Related

  - [`factor_summary_finite_column`](@ref)
  - [`factor_model_summary`](@ref)
"""
function factor_summary_column_median(A::MatNum, k::Integer)
    v = factor_summary_finite_column(A, k)
    return isempty(v) ? convert(eltype(v), NaN) : Statistics.median(v)
end
"""
    factor_summary_ratio(m::Real, v::Real)

Return a ratio of two summary statistics, answering `NaN` wherever the ratio is not defined.

A ratio of a summary is a Sharpe ratio, so its denominator is a volatility. A zero volatility, an absent volatility and a division that overflows all say the same thing — the ratio has no value — and this is the one place that says it.

# Arguments

  - `m`: Numerator.
  - `v`: Denominator.

# Returns

  - `r::Real`: The ratio, and `NaN` when the denominator is zero or absent, or when the ratio is not finite.

# Related

  - [`factor_summary_return_stats`](@ref)
"""
function factor_summary_ratio(m::Real, v::Real)
    r = (isnan(v) || iszero(v)) ? oftype(float(m), NaN) : float(m) / v
    return isfinite(r) ? r : oftype(r, NaN)
end
"""
    factor_summary_return_stats(f::MatNum, ppy::Number)

Return the annualised mean, the annualised volatility and the Sharpe ratio of every factor return series.

An observation whose factor return is absent is dropped from the mean and from the volatility, so a series with a gap is still summarised. The volatility is the corrected sample standard deviation, which needs two observations, and a series with fewer reads `NaN`.

# Mathematical definition

Let ``\\boldsymbol{f}_{k}`` be the finite entries of the factor return series of factor ``k``, let ``T_{k}`` be their count, and let ``p`` be `ppy`.

```math
\\begin{align}
\\mathrm{ann\\_return}_{k} &= p \\, \\dfrac{1}{T_{k}} \\sum_{t} f_{tk}\\\\
\\mathrm{ann\\_volatility}_{k} &= \\sqrt{\\dfrac{p}{T_{k} - 1} \\sum_{t} \\left(f_{tk} - \\dfrac{1}{T_{k}} \\sum_{s} f_{sk}\\right)^{2}}\\\\
\\mathrm{sharpe}_{k} &= \\dfrac{\\mathrm{ann\\_return}_{k}}{\\mathrm{ann\\_volatility}_{k}}\\,.
\\end{align}
```

# Arguments

  - `f`: Factor return history `observations × factors`.
  - `ppy`: Periods per year.

# Returns

  - `ann_return::Vector{<:Real}`: One entry per factor.
  - `ann_volatility::Vector{<:Real}`: One entry per factor.
  - `sharpe::Vector{<:Real}`: One entry per factor.

# Related

  - [`factor_summary_ratio`](@ref)
  - [`factor_model_summary`](@ref)
"""
function factor_summary_return_stats(f::MatNum, ppy::Number)
    K = size(f, 2)
    Tf = promote_type(float(real(eltype(f))), float(real(typeof(ppy))))
    ann_return = Vector{Tf}(undef, K)
    ann_volatility = Vector{Tf}(undef, K)
    sharpe = Vector{Tf}(undef, K)
    s = sqrt(Tf(ppy))
    for k in 1:K
        v = factor_summary_finite_column(f, k)
        ann_return[k] = isempty(v) ? Tf(NaN) : Tf(mean(v)) * Tf(ppy)
        ann_volatility[k] = length(v) < 2 ? Tf(NaN) : Tf(std(v)) * s
        sharpe[k] = factor_summary_ratio(ann_return[k], ann_volatility[k])
    end
    return ann_return, ann_volatility, sharpe
end
"""
    factor_summary_autocorrelation(f::MatNum)

Return the lag-one autocorrelation of every factor return series.

The answer is the Pearson correlation of the pair `(f[1:end - 1, k], f[2:end, k])`, each half centred on **its own** mean. That is the reference implementation's definition, and it is not `StatsBase.autocor`, which centres both halves on the mean of the whole series and divides by the sum of squares of the whole series. The two agree in the limit and differ on a short series, so a caller who wants the other definition calls `StatsBase.autocor` on the factor return history itself.

A series with an absent entry reads `NaN`, because the mean of the half that holds it is absent. This too is what the reference implementation answers.

# Mathematical definition

Let ``a_{t} = f_{tk}`` for ``t = 1 \\ldots T - 1``, let ``b_{t} = f_{(t + 1)k}``, and let ``\\bar{a}`` and ``\\bar{b}`` be their means.

```math
\\mathrm{autocorr}_{k} = \\dfrac{\\sum_{t} (a_{t} - \\bar{a})(b_{t} - \\bar{b})}{\\sqrt{\\sum_{t} (a_{t} - \\bar{a})^{2} \\sum_{t} (b_{t} - \\bar{b})^{2}}}\\,.
```

# Arguments

  - `f`: Factor return history `observations × factors`.

# Returns

  - `autocorr::Vector{<:Real}`: One entry per factor, and `NaN` for a series of one observation.

# Related

  - [`factor_model_summary`](@ref)
"""
function factor_summary_autocorrelation(f::MatNum)
    T, K = size(f)
    Tf = float(real(eltype(f)))
    ac = fill(Tf(NaN), K)
    if T < 2
        return ac
    end
    for k in 1:K
        sa = zero(Tf)
        sb = zero(Tf)
        for t in 1:(T - 1)
            sa += Tf(f[t, k])
            sb += Tf(f[t + 1, k])
        end
        ma = sa / (T - 1)
        mb = sb / (T - 1)
        cab = zero(Tf)
        caa = zero(Tf)
        cbb = zero(Tf)
        for t in 1:(T - 1)
            da = Tf(f[t, k]) - ma
            db = Tf(f[t + 1, k]) - mb
            cab += da * db
            caa += da * da
            cbb += db * db
        end
        ac[k] = cab / sqrt(caa * cbb)
    end
    return ac
end
"""
    factor_summary_returns(csfm::CrossSectionalFactorModel)
    factor_summary_returns(csr::Nothing)
    factor_summary_returns(csr::CrossSectionalRegression)

Return the factor return history a summary reads off a factor model block.

The history is on the **raw** factor axis, because it is what the fit produced before any family re-basis. The absent case is the dispatch rather than a branch, and its message names the field the caller must populate.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `csr`: The `csr` field of the block, or `nothing`.

# Validation

  - `csfm.csr` is not `nothing`, else an `IsNothingError` naming `csr` is raised.

# Returns

  - `f::MatNum`: Factor return history `observations × factors`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`factor_model_summary`](@ref)
"""
function factor_summary_returns(csfm::CrossSectionalFactorModel)
    return factor_summary_returns(csfm.csr)
end
function factor_summary_returns(::Nothing)
    return throw(IsNothingError("csr cannot be nothing: a factor model summary reads the factor return history of the block"))
end
function factor_summary_returns(csr::CrossSectionalRegression)
    return csr.f
end
"""
    factor_summary_positions(csfm::CrossSectionalFactorModel)
    factor_summary_positions(fcb::Nothing, nf::Nothing, K::Integer)
    factor_summary_positions(fcb::Nothing, nf::VecStr, K::Integer)
    factor_summary_positions(fcb::AbstractFactorFamilyBasis, nf::Nothing, K::Integer)
    factor_summary_positions(fcb::FactorFamilyBasis, nf::VecStr, K::Integer)

Return the position each raw factor takes on the reduced factor axis.

The regression group answers on the reduced axis and the summary answers on the raw one, so the two are joined **by name**, as the reference implementation joins them. A raw factor the re-basis dropped takes the position `0`, which is what makes its Gram columns read `NaN`.

A block that carries no re-basis needs no join, and the two axes are then the same axis.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `fcb`: The `fcb` field of the block, or `nothing`.
  - `nf`: The `nf` field of the block, or `nothing`.
  - `K`: Number of raw factors.

# Validation

  - `csfm.nf` is not `nothing` when `csfm.fcb` is present, else an `IsNothingError` naming `nf` is raised: a re-based block cannot be joined without the names.

# Returns

  - `pos::Vector{Int}`: One entry per raw factor. Entry `k` is the position of raw factor `k` on the reduced axis, or `0` when the re-basis dropped it.

# Related

  - [`cs_diagnostic_factor_names`](@ref)
  - [`reduce_factor_names`](@ref)
  - [`factor_summary_mapped`](@ref)
"""
function factor_summary_positions(csfm::CrossSectionalFactorModel)
    return factor_summary_positions(csfm.fcb, csfm.nf, size(csfm.M, 2))
end
function factor_summary_positions(::Nothing, ::Nothing, K::Integer)::Vector{Int}
    return collect(1:K)
end
function factor_summary_positions(::Nothing, ::VecStr, K::Integer)::Vector{Int}
    return collect(1:K)
end
function factor_summary_positions(::AbstractFactorFamilyBasis, ::Nothing,
                                  ::Integer)::Vector{Int}
    return throw(IsNothingError("nf cannot be nothing: a summary of a re-based block joins the reduced factor axis to the raw one by name"))
end
function factor_summary_positions(fcb::FactorFamilyBasis, nf::VecStr,
                                  ::Integer)::Vector{Int}
    red = reduce_factor_names(fcb, nf)
    idx = Dict{String, Int}(red[j] => j for j in eachindex(red))
    return Int[get(idx, String(n), 0) for n in nf]
end
"""
    factor_summary_mapped(v::VecNum, pos::AbstractVector{Int})

Return a statistic of the reduced factor axis, written onto the raw factor axis.

# Arguments

  - `v`: A statistic, one entry per reduced factor.
  - `pos`: Position of each raw factor on the reduced axis, `0` where the re-basis dropped it.

# Returns

  - `m::Vector{<:Real}`: One entry per raw factor, and `NaN` at a factor the re-basis dropped.

# Related

  - [`factor_summary_positions`](@ref)
  - [`factor_model_summary`](@ref)
"""
function factor_summary_mapped(v::VecNum, pos::AbstractVector{Int})
    Tf = float(real(eltype(v)))
    m = fill(Tf(NaN), length(pos))
    for k in eachindex(pos)
        j = pos[k]
        if j > 0
            m[k] = Tf(v[j])
        end
    end
    return m
end
"""
    factor_summary_gram(csfm::CrossSectionalFactorModel, threshold::Number)
    factor_summary_gram(Ms::Nothing, csfm::CrossSectionalFactorModel, threshold::Number)
    factor_summary_gram(Ms::Arr3Num, csfm::CrossSectionalFactorModel, threshold::Number)

Return the three regression columns of a factor model summary, on the raw factor axis.

The columns are the mean absolute t-statistic, the rate at which the absolute t-statistic passes `threshold`, and the mean variance inflation factor. Each is the time average of a level-2 series of the regression group, joined back onto the raw factor axis by name.

A block that carries no exposure history carries no regression design either, so the three columns are absent as a whole. The absent case is the dispatch rather than a branch.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `Ms`: The `Ms` field of the block, or `nothing`.
  - `threshold`: Absolute t-statistic the exceedance rate counts against.

# Returns

  - `mean_abs_t::Option{<:Vector{<:Real}}`: One entry per raw factor, or `nothing`.
  - `t_rate::Option{<:Vector{<:Real}}`: One entry per raw factor, or `nothing`.
  - `mean_vif::Option{<:Vector{<:Real}}`: One entry per raw factor, or `nothing`.

# Related

  - [`cs_regression_t_stats`](@ref)
  - [`cs_regression_t_stat_exceedance_rate`](@ref)
  - [`exposure_vif`](@ref)
  - [`factor_summary_positions`](@ref)
"""
function factor_summary_gram(csfm::CrossSectionalFactorModel, threshold::Number)
    return factor_summary_gram(csfm.Ms, csfm, threshold)
end
function factor_summary_gram(::Nothing, ::CrossSectionalFactorModel, ::Number)
    return nothing, nothing, nothing
end
function factor_summary_gram(::Arr3Num, csfm::CrossSectionalFactorModel, threshold::Number)
    pos = factor_summary_positions(csfm)
    t = cs_regression_t_stats(csfm)
    vif = exposure_vif(csfm)
    rate = cs_regression_t_stat_exceedance_rate(csfm; threshold = threshold)
    Kr = size(t, 2)
    Tf = promote_type(float(real(eltype(t))), float(real(eltype(vif))))
    at = abs.(t)
    abs_t = Vector{Tf}(undef, Kr)
    mvif = Vector{Tf}(undef, Kr)
    for j in 1:Kr
        abs_t[j] = factor_summary_column_mean(at, j)
        mvif[j] = factor_summary_column_mean(vif, j)
    end
    return factor_summary_mapped(abs_t, pos), factor_summary_mapped(rate, pos),
           factor_summary_mapped(mvif, pos)
end
"""
    factor_summary_exposure_variance(Ms::Arr3Num, t::Integer, k::Integer)

Return the cross-sectional variance of one factor exposure at one observation.

The variance is taken over the finite exposures of the observation and it is not corrected, which is the reference implementation's convention for the test that finds a constant exposure.

# Arguments

  - `Ms`: Exposure history `observations × assets × factors`, unlagged.
  - `t`: Position of the observation.
  - `k`: Position of the factor.

# Returns

  - `v::Real`: The variance, and `NaN` when the cross-section carries no finite exposure.

# Related

  - [`factor_summary_constant_exposures`](@ref)
"""
function factor_summary_exposure_variance(Ms::Arr3Num, t::Integer, k::Integer)
    Tf = float(real(eltype(Ms)))
    n = 0
    s = zero(Tf)
    for i in axes(Ms, 2)
        a = Ms[t, i, k]
        if !isnan(a)
            n += 1
            s += Tf(a)
        end
    end
    if n == 0
        return Tf(NaN)
    end
    m = s / n
    q = zero(Tf)
    for i in axes(Ms, 2)
        a = Ms[t, i, k]
        if !isnan(a)
            q += (Tf(a) - m)^2
        end
    end
    return q / n
end
"""
    factor_summary_constant_exposures(Ms::Arr3Num)

Return which factor exposures never vary across the cross-section.

A constant exposure is the global intercept and the constant column of a one-hot family. Its cross-section has no spread, so its stability coefficient is not defined and the summary writes `1` in its place: an exposure that never moves is perfectly stable. That patch is the reference implementation's, and it lives in the summary rather than in [`exposure_stability`](@ref), which answers the `NaN` the correlation earns.

# Arguments

  - `Ms`: Exposure history `observations × assets × factors`, unlagged.

# Returns

  - `c::BitVector`: One entry per factor, `true` when the largest cross-sectional variance the factor reaches is under `1e-12`.

# Related

  - [`factor_summary_exposure_variance`](@ref)
  - [`exposure_stability`](@ref)
  - [`factor_model_summary`](@ref)
"""
function factor_summary_constant_exposures(Ms::Arr3Num)
    T = size(Ms, 1)
    K = size(Ms, 3)
    c = falses(K)
    for k in 1:K
        m = -Inf
        for t in 1:T
            v = factor_summary_exposure_variance(Ms, t, k)
            if !isnan(v) && v > m
                m = v
            end
        end
        # A factor with no finite cross-section anywhere leaves `m` at `-Inf`, and it is
        # not constant: the reference's `NaN < 1e-12` is false there too.
        c[k] = m > -Inf && m < 1e-12
    end
    return c
end
"""
    factor_summary_stability(Ms::Arr3Num, csfm::CrossSectionalFactorModel; step::Integer,
                             weighting)

Return the stability column of a factor model summary, on the raw factor axis.

The column is the median over the observations of [`exposure_stability`](@ref), with a constant exposure patched to `1`. A history with no more observations than `step` has no stability series at all, and then every factor reads `NaN` but for the constant ones, which still read `1`.

# Arguments

  - `Ms`: Exposure history `observations × assets × factors`, unlagged.
  - `csfm`: A cross-sectional factor model block.
  - `step`: Number of observations between the two cross-sections the coefficient reads.
  - `weighting`: The [`AbstractOrthogonalityMetric`](@ref) whose weight history the coefficient reads.

# Returns

  - `stability::Vector{<:Real}`: One entry per raw factor.

# Related

  - [`exposure_stability`](@ref)
  - [`factor_summary_constant_exposures`](@ref)
  - [`factor_model_summary`](@ref)
"""
function factor_summary_stability(Ms::Arr3Num, csfm::CrossSectionalFactorModel;
                                  step::Integer = 21, weighting = BenchmarkWeightMetric())
    con = factor_summary_constant_exposures(Ms)
    K = size(Ms, 3)
    Tf = float(real(eltype(Ms)))
    if size(Ms, 1) <= step
        return Tf[con[k] ? one(Tf) : Tf(NaN) for k in 1:K]
    end
    S = exposure_stability(csfm; step = step, weighting = weighting)
    return Tf[con[k] ? one(Tf) : Tf(factor_summary_column_median(S, k)) for k in 1:K]
end
"""
    factor_summary_exposure(csfm::CrossSectionalFactorModel; step, weighting,
                            coverage_weighting)
    factor_summary_exposure(Ms::Nothing, csfm::CrossSectionalFactorModel; kwargs...)
    factor_summary_exposure(Ms::Arr3Num, csfm::CrossSectionalFactorModel; step, weighting,
                            coverage_weighting)

Return the two exposure columns of a factor model summary, on the raw factor axis.

The columns are the median exposure stability and the average coverage. Both read the unlagged exposure history, as the whole exposure group does. A block that carries no exposure history has neither, and the absent case is the dispatch rather than a branch.

The two columns read **different** weight histories, which is what the reference implementation does: the stability reads the history `weighting` names, and the coverage reads the one `coverage_weighting` names.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `Ms`: The `Ms` field of the block, or `nothing`.
  - `step`: Number of observations between the two cross-sections the stability reads.
  - `weighting`: The [`AbstractOrthogonalityMetric`](@ref) the stability reads.
  - `coverage_weighting`: The [`AbstractOrthogonalityMetric`](@ref) whose positive weights are the coverage's universe.

# Returns

  - `stability::Option{<:Vector{<:Real}}`: One entry per raw factor, or `nothing`.
  - `coverage::Option{<:Vector{<:Real}}`: One entry per raw factor, or `nothing`.

# Related

  - [`exposure_stability`](@ref)
  - [`exposure_coverage`](@ref)
  - [`factor_summary_stability`](@ref)
"""
function factor_summary_exposure(csfm::CrossSectionalFactorModel; step::Integer = 21,
                                 weighting = BenchmarkWeightMetric(),
                                 coverage_weighting = RegressionWeightMetric())
    return factor_summary_exposure(csfm.Ms, csfm; step = step, weighting = weighting,
                                   coverage_weighting = coverage_weighting)
end
function factor_summary_exposure(::Nothing, ::CrossSectionalFactorModel; kwargs...)
    return nothing, nothing
end
function factor_summary_exposure(Ms::Arr3Num, csfm::CrossSectionalFactorModel;
                                 step::Integer = 21, weighting = BenchmarkWeightMetric(),
                                 coverage_weighting = RegressionWeightMetric())
    stability = factor_summary_stability(Ms, csfm; step = step, weighting = weighting)
    coverage = exposure_coverage(csfm; weighting = coverage_weighting)
    return stability, coverage
end
"""
$(DocStringExtensions.TYPEDEF)

The headline statistics of every factor of a cross-sectional factor model.

`FactorSummaryResult` is what [`factor_model_summary`](@ref) returns. It carries the nine columns [`plot_factor_model_summary`](@ref) draws, one entry per **raw** factor, and the annualisation factor that produced the first three, so a summary can be tabulated, compared across fits, asserted on in a test, or read without a plotting package installed.

# The five columns that can be absent

`mean_abs_t`, `t_rate` and `mean_vif` read the regression design, and `stability` and `coverage` read the exposure history. A block that carries no exposure history has none of them, and all five read back as `nothing`. A consumer reads that by dispatch rather than by a test of its own.

# The raw factor axis

The three Gram columns are computed on the reduced factor axis of the family re-basis, and are written back onto the raw axis by name. A raw factor the re-basis dropped carries `NaN` in those three columns and a value in the others.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorSummaryResult(
        ann_return, ann_volatility, sharpe, autocorr,
        mean_abs_t, t_rate, mean_vif, stability, coverage, ppy
    ) -> FactorSummaryResult

Arguments correspond to the struct's fields, in the order they are declared. The type is a
Result, so [`factor_model_summary`](@ref) builds it and a caller reads it; there is no
keyword constructor, and the type validates nothing of its own.

# Related

  - [`factor_model_summary`](@ref)
  - [`plot_factor_model_summary`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
@concrete struct FactorSummaryResult <: AbstractResult
    """
    Annualised mean of the factor return series, one entry per raw factor.
    """
    ann_return
    """
    Annualised volatility of the factor return series, one entry per raw factor.
    """
    ann_volatility
    """
    Ratio of the annualised mean to the annualised volatility, one entry per raw factor.
    """
    sharpe
    """
    Lag-one autocorrelation of the factor return series, one entry per raw factor.
    """
    autocorr
    """
    Mean absolute cross-sectional t-statistic, one entry per raw factor, or `nothing` when the block carries no exposure history.
    """
    mean_abs_t
    """
    Fraction of the observations at which the absolute t-statistic passes the threshold, one entry per raw factor, or `nothing` when the block carries no exposure history.
    """
    t_rate
    """
    Mean variance inflation factor, one entry per raw factor, or `nothing` when the block carries no exposure history.
    """
    mean_vif
    """
    Median exposure stability coefficient, one entry per raw factor, or `nothing` when the block carries no exposure history.
    """
    stability
    """
    Average fraction of the universe at which the factor exposure is finite, one entry per raw factor, or `nothing` when the block carries no exposure history.
    """
    coverage
    """
    $(field_dict[:ps_ppy]) It defaults to `1`, which reports the statistics per period.
    """
    ppy
end
"""
    factor_model_summary(csfm::CrossSectionalFactorModel; ppy::Number = 1,
                         threshold::Number = 2, step::Integer = 21,
                         weighting = BenchmarkWeightMetric(),
                         coverage_weighting = RegressionWeightMetric()) -> FactorSummaryResult

Summarise every factor of a cross-sectional factor model as a [`FactorSummaryResult`](@ref).

The summary is the top of the diagnostic hierarchy. It calls one level-2 verb of the regression group and one of the exposure group per column, and it aggregates each series over the observations. It computes no statistic of its own but the factor return statistics and the two aggregates that the reference implementation defines only inside its own summary: the median of the stability, and the patch that reads a constant exposure as perfectly stable.

The answer is on the **raw** factor axis. The regression group answers on the reduced axis of the family re-basis, and the summary joins the two by name.

# Algorithm

 1. Read the factor return history off `csr`, refusing a block that carries none.
 2. Take the annualised mean, the annualised volatility and the Sharpe ratio of each series with [`factor_summary_return_stats`](@ref), dropping the absent observations.
 3. Take the lag-one autocorrelation of each series with [`factor_summary_autocorrelation`](@ref).
 4. Take the mean absolute t-statistic, the exceedance rate and the mean variance inflation factor, and join them onto the raw factor axis. A block with no exposure history gives `nothing` for all three.
 5. Take the median stability and the coverage. A block with no exposure history gives `nothing` for both.
 6. Collect the nine columns and `ppy` into a [`FactorSummaryResult`](@ref).

# Arguments

  - `csfm`: A cross-sectional factor model block. A caller who holds a Prior Result writes `pr.rr`.
  - `ppy`: Periods per year. `252` annualises a daily fit, and the default of `1` reports the statistics per period.
  - `threshold`: Absolute t-statistic the exceedance rate counts against.
  - `step`: Number of observations between the two cross-sections the stability reads.
  - `weighting`: The [`AbstractOrthogonalityMetric`](@ref) whose weight history the stability reads.
  - `coverage_weighting`: The [`AbstractOrthogonalityMetric`](@ref) whose positive weights are the universe of the coverage. Its default of [`RegressionWeightMetric`](@ref) is the estimation universe of the fit, which is the universe the reference implementation reads.

# Validation

  - `ppy > 0`.
  - `csfm.csr` is not `nothing`.
  - `csfm.nf` is not `nothing` when `csfm.fcb` is present.

# Returns

  - `fs::FactorSummaryResult`: The computed summary.

# Related

  - [`FactorSummaryResult`](@ref)
  - [`plot_factor_model_summary`](@ref)
  - [`cs_regression_t_stats`](@ref)
  - [`exposure_stability`](@ref)
  - [`exposure_coverage`](@ref)
"""
function factor_model_summary(csfm::CrossSectionalFactorModel; ppy::Number = 1,
                              threshold::Number = 2, step::Integer = 21,
                              weighting = BenchmarkWeightMetric(),
                              coverage_weighting = RegressionWeightMetric())::FactorSummaryResult
    @argcheck(ppy > zero(ppy), DomainError(ppy, "ppy must be positive"))
    f = factor_summary_returns(csfm)
    ann_return, ann_volatility, sharpe = factor_summary_return_stats(f, ppy)
    autocorr = factor_summary_autocorrelation(f)
    mean_abs_t, t_rate, mean_vif = factor_summary_gram(csfm, threshold)
    stability, coverage = factor_summary_exposure(csfm; step = step, weighting = weighting,
                                                  coverage_weighting = coverage_weighting)
    return FactorSummaryResult(ann_return, ann_volatility, sharpe, autocorr, mean_abs_t,
                               t_rate, mean_vif, stability, coverage, ppy)
end

export FactorSummaryResult, factor_model_summary
