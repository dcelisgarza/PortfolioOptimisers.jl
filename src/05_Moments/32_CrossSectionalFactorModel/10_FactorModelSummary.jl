"""
    factor_summary_finite_column(A::MatNum, k::Integer)

Return the present entries of one column of a diagnostic series.

Every column of the summary aggregates a series over the observations. An observation whose answer is absent holds `NaN`, and it takes no part in the aggregate. This function drops the absent entries once, so each aggregate reads a dense vector and carries no test of its own. An infinite entry is a value and not an absence, so it stays, and the aggregate of its column is infinite too.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{v} &= \\left(a_{tk}\\right)_{t \\in \\mathcal{T}_{k}}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{v}``: Present entries of column ``k``, in the order of the observations.
  - $(math_dict[:a_tk_summary])
  - $(math_dict[:T_k_summary])

# Arguments

  - `A`: A series `observations × factors`.
  - `k`: Position of the factor.

# Returns

  - `v::Vector{<:Real}`: The entries of column `k` that are not `NaN`, in the order of the observations. An integer series gives a floating-point vector.

# Related

  - [`factor_summary_column_mean`](@ref)
  - [`factor_summary_column_median`](@ref)
  - [`factor_model_summary`](@ref)
"""
function factor_summary_finite_column(A::MatNum, k::Integer)
    Tf = float_if_integer(real(eltype(A)))
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

Return the mean of the present entries of one column of a diagnostic series.

# Mathematical definition

```math
\\begin{align}
\\bar{a}_{k} &= \\frac{1}{|\\mathcal{T}_{k}|} \\sum_{t \\in \\mathcal{T}_{k}} a_{tk}\\,.
\\end{align}
```

Where:

  - ``\\bar{a}_{k}``: Mean of the present entries of column ``k``.
  - $(math_dict[:a_tk_summary])
  - $(math_dict[:T_k_summary])

The mean is not defined for an empty ``\\mathcal{T}_{k}``.

# Arguments

  - `A`: A series `observations × factors`.
  - `k`: Position of the factor.

# Returns

  - `m::Real`: The mean, and `NaN` when the column has no present entry.

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

Return the median of the present entries of one column of a diagnostic series.

# Mathematical definition

```math
\\begin{align}
\\tilde{a}_{k} &= \\operatorname{median} \\left\\{ a_{tk} : t \\in \\mathcal{T}_{k} \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\tilde{a}_{k}``: Median of the present entries of column ``k``. For an even count it is the mean of the two middle entries.
  - $(math_dict[:a_tk_summary])
  - $(math_dict[:T_k_summary])

The median is not defined for an empty ``\\mathcal{T}_{k}``.

# Arguments

  - `A`: A series `observations × factors`.
  - `k`: Position of the factor.

# Returns

  - `m::Real`: The median, and `NaN` when the column has no present entry.

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

Return a ratio of two summary statistics, and `NaN` where the ratio has no value.

The summary takes one ratio, the Sharpe ratio, whose denominator is a volatility. A zero volatility, an absent volatility and a division that overflows each leave the ratio without a value, and this function answers `NaN` for all three.

# Mathematical definition

```math
\\begin{align}
r &= \\begin{cases}
m / v & \\text{when } m / v \\text{ is finite}\\,,\\\\
\\mathrm{NaN} & \\text{otherwise}\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``r``: The ratio.
  - ``m``: Numerator.
  - ``v``: Denominator. A zero or `NaN` value of ``v`` leaves ``m / v`` not finite.

# Arguments

  - `m`: Numerator.
  - `v`: Denominator.

# Returns

  - `r::Real`: The ratio, and `NaN` when the denominator is zero or absent, or when the ratio is not finite.

# Related

  - [`factor_summary_return_stats`](@ref)
"""
function factor_summary_ratio(m::Real, v::Real)
    r = m / v
    return isfinite(r) ? r : oftype(r, NaN)
end
"""
    factor_summary_return_stats(f::MatNum, ppy::Number)

Return the annualised mean, the annualised volatility and the Sharpe ratio of every factor return series.

An absent factor return takes no part in the mean or the volatility, so a series with a gap still has a summary. The volatility is the corrected sample standard deviation, which needs two observations, and a series with fewer reads `NaN`. A constant series has a zero volatility, so its Sharpe ratio reads `NaN`.

# Mathematical definition

```math
\\begin{align}
\\hat{\\mu}_{k} &= \\frac{1}{T_{k}} \\sum_{t \\in \\mathcal{T}_{k}} f_{tk}\\,,\\\\
\\mathrm{ann\\_return}_{k} &= p \\, \\hat{\\mu}_{k}\\,,\\\\
\\mathrm{ann\\_volatility}_{k} &= \\sqrt{\\frac{p}{T_{k} - 1} \\sum_{t \\in \\mathcal{T}_{k}} \\left(f_{tk} - \\hat{\\mu}_{k}\\right)^{2}}\\,,\\\\
\\mathrm{sharpe}_{k} &= \\frac{\\mathrm{ann\\_return}_{k}}{\\mathrm{ann\\_volatility}_{k}}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mu}_{k}``: Per-period mean of the present returns of factor ``k``.
  - ``\\mathrm{ann\\_return}_{k}``, ``\\mathrm{ann\\_volatility}_{k}``, ``\\mathrm{sharpe}_{k}``: The three returned entries of factor ``k``.
  - $(math_dict[:f_tk_summary])
  - $(math_dict[:T_k_summary])
  - ``T_{k} = |\\mathcal{T}_{k}|``: Count of the present returns of factor ``k``.
  - ``p``: Periods per year, `ppy`.

The mean is not defined for ``T_{k} = 0``, the volatility is not defined for ``T_{k} < 2``, and the Sharpe ratio is not defined where the volatility is zero.

# Algorithm

For each factor `k`:

 1. Drop the absent returns of column `k` with [`factor_summary_finite_column`](@ref), giving `v`.
 2. Take the mean of `v` and clamp it to the least and the greatest entry of `v`, giving `m`. The clamp changes nothing in exact arithmetic. The rounded mean of a constant `v` can differ from its common value, and the clamp makes `m` equal to that value, so every deviation in step 4 is exactly zero.
 3. Multiply `m` by `ppy`, giving `ann_return[k]`. An empty `v` gives `NaN`.
 4. Take the corrected sample standard deviation of `v` about `m` and multiply it by the square root of `ppy`, giving `ann_volatility[k]`. A `v` with fewer than two entries gives `NaN`.
 5. Divide the two with [`factor_summary_ratio`](@ref), giving `sharpe[k]`.

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
    Tr = promote_type(float_if_integer(real(eltype(f))), real(typeof(ppy)))
    Tv = typeof(sqrt(one(Tr)))
    ann_return = Vector{Tr}(undef, K)
    ann_volatility = Vector{Tv}(undef, K)
    sharpe = Vector{promote_type(Tr, Tv)}(undef, K)
    s = sqrt(Tr(ppy))
    for k in 1:K
        v = factor_summary_finite_column(f, k)
        # The clamp moves nothing in exact arithmetic. On a constant series it makes the mean
        # the common value, so the volatility is exactly zero and the Sharpe ratio is `NaN`.
        m = isempty(v) ? Tr(NaN) : Tr(clamp(mean(v), extrema(v)...))
        ann_return[k] = m * Tr(ppy)
        ann_volatility[k] = length(v) < 2 ? Tv(NaN) : Tv(std(v; mean = m)) * s
        sharpe[k] = factor_summary_ratio(ann_return[k], ann_volatility[k])
    end
    return ann_return, ann_volatility, sharpe
end
"""
    factor_summary_autocorrelation(f::MatNum)

Return the lag-one autocorrelation of every factor return series.

The answer is the Pearson correlation of the pair `(f[1:end - 1, k], f[2:end, k])`, with each half centred on its own mean. `StatsBase.autocor` is a different estimator. It centres both halves on the mean of the whole series and divides by the sum of squares of the whole series. The two agree in the limit and differ on a short series, so a caller who wants the other definition calls `StatsBase.autocor` on the factor return history itself.

# Mathematical definition

```math
\\begin{align}
\\bar{a}_{k} &= \\frac{1}{T - 1} \\sum_{t = 1}^{T - 1} f_{tk}\\,,\\\\
\\bar{b}_{k} &= \\frac{1}{T - 1} \\sum_{t = 2}^{T} f_{tk}\\,,\\\\
\\rho_{k} &= \\frac{\\sum_{t = 1}^{T - 1} \\left(f_{tk} - \\bar{a}_{k}\\right)\\left(f_{(t + 1)k} - \\bar{b}_{k}\\right)}{\\sqrt{\\sum_{t = 1}^{T - 1} \\left(f_{tk} - \\bar{a}_{k}\\right)^{2} \\sum_{t = 1}^{T - 1} \\left(f_{(t + 1)k} - \\bar{b}_{k}\\right)^{2}}}\\,.
\\end{align}
```

Where:

  - ``\\rho_{k}``: Lag-one autocorrelation of factor ``k``.
  - ``\\bar{a}_{k}``: Mean of the leading half, the first ``T - 1`` returns of factor ``k``.
  - ``\\bar{b}_{k}``: Mean of the trailing half, the last ``T - 1`` returns of factor ``k``.
  - $(math_dict[:f_tk_summary])
  - $(math_dict[:T])

The coefficient is not defined for ``T < 2``, nor for a series with a constant half, whose denominator is zero. An absent return makes the mean of its half absent, so the coefficient of that series is absent too.

# Algorithm

 1. Return `NaN` for every factor when `T < 2`.
 2. For each factor `k`, sum each half and track the least and the greatest entry of each half.
 3. Divide each sum by `T - 1` and clamp it to the least and the greatest entry of its half, giving `ma` and `mb`. The clamp changes nothing in exact arithmetic. It makes the mean of a constant half equal to its common value, so the deviations of that half are exactly zero and the coefficient is `NaN`.
 4. Accumulate the cross product `cab` and the two sums of squares `caa` and `cbb` of the deviations.
 5. Divide `cab` by the square root of `caa * cbb`, giving `ac[k]`.

# Arguments

  - `f`: Factor return history `observations × factors`.

# Returns

  - `autocorr::Vector{<:Real}`: One entry per factor, and `NaN` for a series of one observation, a series with a constant half and a series with an absent return.

# Related

  - [`factor_model_summary`](@ref)
"""
function factor_summary_autocorrelation(f::MatNum)
    T, K = size(f)
    Tf = typeof(sqrt(one(float_if_integer(real(eltype(f))))))
    ac = fill(Tf(NaN), K)
    if T < 2
        return ac
    end
    for k in 1:K
        sa = zero(Tf)
        sb = zero(Tf)
        la = ha = Tf(f[1, k])
        lb = hb = Tf(f[2, k])
        for t in 1:(T - 1)
            a = Tf(f[t, k])
            b = Tf(f[t + 1, k])
            sa += a
            sb += b
            la, ha = min(la, a), max(ha, a)
            lb, hb = min(lb, b), max(hb, b)
        end
        # The clamp moves nothing in exact arithmetic. On a constant half it makes the mean
        # the common value, so the deviations of that half are exactly zero.
        ma = clamp(sa / (T - 1), la, ha)
        mb = clamp(sb / (T - 1), lb, hb)
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

Return the factor return history that a summary reads off a factor model block.

The history is on the raw factor axis, because the fit produced it before any family re-basis. A separate method handles a block with no fit, and its message names the field the caller must populate.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `csr`: The `csr` field of the block, or `nothing`.

# Validation

  - `csfm.csr` is not `nothing`, else the function raises an `IsNothingError` that names `csr`.

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

Return the position of each raw factor on the reduced factor axis.

The regression group answers on the reduced axis and the summary answers on the raw one, so the summary joins the two by name. A raw factor that the re-basis dropped takes the position `0`, and its Gram columns read `NaN` for that reason. A block with no re-basis needs no join, because its two axes are the same axis.

# Mathematical definition

```math
\\begin{align}
\\pi_{k} &= \\begin{cases}
j & \\text{when } n_{k} = r_{j}\\,,\\\\
0 & \\text{when } n_{k} \\notin \\left\\{r_{1}, \\ldots, r_{K_{r}}\\right\\}\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:pi_k_summary])
  - ``n_{k}``: Name of raw factor ``k``.
  - ``r_{j}``: Name of reduced factor ``j``.
  - ``K_{r}``: Number of reduced factors.

A block with no re-basis has ``\\pi_{k} = k``.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `fcb`: The `fcb` field of the block, or `nothing`.
  - `nf`: The `nf` field of the block, or `nothing`.
  - `K`: Number of raw factors.

# Validation

  - `csfm.nf` is not `nothing` when `csfm.fcb` is present, else the function raises an `IsNothingError` that names `nf`. The summary cannot join a re-based block to the raw axis without the names.

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

# Mathematical definition

```math
\\begin{align}
m_{k} &= \\begin{cases}
v_{\\pi_{k}} & \\text{when } \\pi_{k} > 0\\,,\\\\
\\mathrm{NaN} & \\text{when } \\pi_{k} = 0\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``m_{k}``: The statistic of raw factor ``k``.
  - ``v_{j}``: The statistic of reduced factor ``j``.
  - $(math_dict[:pi_k_summary])

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
    Tf = float_if_integer(real(eltype(v)))
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

The columns are the mean absolute t-statistic, the rate at which the absolute t-statistic passes `threshold`, and the mean variance inflation factor. Each one is the time average of a series of the regression group, joined back onto the raw factor axis by name.

A block with no exposure history has no regression design either, so all three columns are absent, and a separate method handles that case.

# Algorithm

 1. Take the position of each raw factor on the reduced axis with [`factor_summary_positions`](@ref), giving `pos`.
 2. Take the t-statistic history with [`cs_regression_t_stats`](@ref), giving `t`, and the variance inflation factor history with [`exposure_vif`](@ref), giving `vif`.
 3. Take the exceedance rate at `threshold` with [`cs_regression_t_stat_exceedance_rate`](@ref), giving `rate`.
 4. For each reduced factor, take the mean of the present entries of `abs.(t)` and of `vif` with [`factor_summary_column_mean`](@ref), giving `abs_t` and `mvif`.
 5. Write `abs_t`, `rate` and `mvif` onto the raw axis with [`factor_summary_mapped`](@ref).

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `Ms`: The `Ms` field of the block, or `nothing`.
  - `threshold`: Absolute t-statistic that the exceedance rate counts against.

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
    Tf = promote_type(real(eltype(t)), real(eltype(vif)))
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

The variance is uncorrected, and it reads the exposures that are not `NaN`.

# Mathematical definition

```math
\\begin{align}
\\bar{b}_{tk} &= \\frac{1}{|\\mathcal{N}_{tk}|} \\sum_{i \\in \\mathcal{N}_{tk}} b_{tik}\\,,\\\\
s^{2}_{tk} &= \\frac{1}{|\\mathcal{N}_{tk}|} \\sum_{i \\in \\mathcal{N}_{tk}} \\left(b_{tik} - \\bar{b}_{tk}\\right)^{2}\\,.
\\end{align}
```

Where:

  - $(math_dict[:s2_tk_summary])
  - ``\\bar{b}_{tk}``: Cross-sectional mean of the exposure to factor ``k`` at observation ``t``.
  - ``b_{tik}``: Exposure of asset ``i`` to factor ``k`` at observation ``t``.
  - ``\\mathcal{N}_{tk}``: Assets at which ``b_{tik}`` is not `NaN`.

The variance is not defined for an empty ``\\mathcal{N}_{tk}``. An infinite exposure makes the deviations undefined, so the variance reads `NaN`.

# Algorithm

 1. Sum the exposures of the cross-section that are not `NaN`, count them, and track their least and greatest value.
 2. Return `NaN` when the count is zero.
 3. Divide the sum by the count and clamp it to the least and the greatest value, giving `m`. The clamp changes nothing in exact arithmetic. It makes the mean of a constant cross-section equal to its common value, so the variance of that cross-section is exactly zero.
 4. Average the squared deviations from `m`, giving the variance.

# Arguments

  - `Ms`: Exposure history `observations × assets × factors`, unlagged.
  - `t`: Position of the observation.
  - `k`: Position of the factor.

# Returns

  - `v::Real`: The variance, and `NaN` when the cross-section has no exposure that is not `NaN`.

# Related

  - [`factor_summary_constant_exposures`](@ref)
"""
function factor_summary_exposure_variance(Ms::Arr3Num, t::Integer, k::Integer)
    Tf = float_if_integer(real(eltype(Ms)))
    n = 0
    s = zero(Tf)
    lo = Tf(Inf)
    hi = -Tf(Inf)
    for i in axes(Ms, 2)
        a = Ms[t, i, k]
        if !isnan(a)
            n += 1
            s += Tf(a)
            lo, hi = min(lo, Tf(a)), max(hi, Tf(a))
        end
    end
    if n == 0
        return Tf(NaN)
    end
    # The clamp moves nothing in exact arithmetic. On a constant cross-section it makes the
    # mean the common value, so every deviation is exactly zero.
    m = clamp(s / n, lo, hi)
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

The global intercept and the constant column of a one-hot family are constant exposures. The cross-section of a constant exposure has no spread, so its stability coefficient is not defined, and the summary writes `1` in its place. An exposure that never moves is perfectly stable. The summary applies that patch, and [`exposure_stability`](@ref) keeps the `NaN` that the correlation gives.

# Mathematical definition

```math
\\begin{align}
c_{k} &= \\begin{cases}
\\max_{t \\in \\mathcal{V}_{k}} s^{2}_{tk} < 10^{-12} & \\text{when } \\mathcal{V}_{k} \\neq \\emptyset\\,,\\\\
\\mathrm{false} & \\text{when } \\mathcal{V}_{k} = \\emptyset\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:c_k_summary])
  - $(math_dict[:s2_tk_summary])
  - ``\\mathcal{V}_{k}``: Observations at which ``s^{2}_{tk}`` is not `NaN`.

A factor with no exposure anywhere is not constant.

# Arguments

  - `Ms`: Exposure history `observations × assets × factors`, unlagged.

# Returns

  - `c::BitVector`: One entry per factor, `true` when the largest cross-sectional variance of the factor is under `1e-12`.

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
        # A factor whose every cross-sectional variance is `NaN` leaves `m` at `-Inf`, and it
        # is not constant.
        c[k] = m > -Inf && m < 1e-12
    end
    return c
end
"""
    factor_summary_stability(Ms::Arr3Num, csfm::CrossSectionalFactorModel; step::Integer,
                             weighting)

Return the stability column of a factor model summary, on the raw factor axis.

The column is the median over the observations of [`exposure_stability`](@ref), and a constant exposure reads `1`. A history with no more observations than `step` has no stability series, so every factor reads `NaN` except the constant ones, which still read `1`.

# Mathematical definition

```math
\\begin{align}
\\mathrm{stability}_{k} &= \\begin{cases}
1 & \\text{when } c_{k}\\,,\\\\
\\operatorname{median} \\left\\{ a_{tk} : t \\in \\mathcal{T}_{k} \\right\\} & \\text{otherwise}\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``\\mathrm{stability}_{k}``: The returned entry of factor ``k``.
  - $(math_dict[:c_k_summary])
  - $(math_dict[:a_tk_summary]) Here the series is the stability coefficient that [`exposure_stability`](@ref) returns.
  - $(math_dict[:T_k_summary])

The median is not defined for an empty ``\\mathcal{T}_{k}``, and that is the case for every factor when the history has no more observations than `step`.

# Arguments

  - `Ms`: Exposure history `observations × assets × factors`, unlagged.
  - `csfm`: A cross-sectional factor model block.
  - `step`: Number of observations between the two cross-sections that the coefficient reads.
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
    Tf = float_if_integer(real(eltype(Ms)))
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

The columns are the median exposure stability and the average coverage. Both read the unlagged exposure history, as the whole exposure group does. A block with no exposure history has neither column, and a separate method handles that case.

The two columns read different weight histories. The stability reads the history that `weighting` names, and the coverage reads the history that `coverage_weighting` names.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `Ms`: The `Ms` field of the block, or `nothing`.
  - `step`: Number of observations between the two cross-sections that the stability reads.
  - `weighting`: The [`AbstractOrthogonalityMetric`](@ref) that the stability reads.
  - `coverage_weighting`: The [`AbstractOrthogonalityMetric`](@ref) whose positive weights are the universe of the coverage.

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

[`factor_model_summary`](@ref) returns a `FactorSummaryResult`. It holds the nine columns that [`plot_factor_model_summary`](@ref) draws, one entry per raw factor, and the annualisation factor of the first three. A caller can tabulate a summary, compare it across fits, test it, or read it with no plotting package installed.

# The five columns that can be absent

`mean_abs_t`, `t_rate` and `mean_vif` read the regression design, and `stability` and `coverage` read the exposure history. A block with no exposure history has none of them, and all five are `nothing`. A consumer handles that case by dispatch on `nothing`.

# The raw factor axis

The summary computes the three Gram columns on the reduced factor axis of the family re-basis, and writes them back onto the raw axis by name. A raw factor that the re-basis dropped has `NaN` in those three columns and a value in the others.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorSummaryResult(
        ann_return, ann_volatility, sharpe, autocorr,
        mean_abs_t, t_rate, mean_vif, stability, coverage, ppy
    ) -> FactorSummaryResult

The arguments are the fields, in the order of their declaration. The type is a Result, so [`factor_model_summary`](@ref) builds it and a caller reads it. It has no keyword constructor, and it checks none of its values.

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
    Mean absolute cross-sectional t-statistic, one entry per raw factor, or `nothing` when the block has no exposure history.
    """
    mean_abs_t
    """
    Fraction of the observations at which the absolute t-statistic passes the threshold, one entry per raw factor, or `nothing` when the block has no exposure history.
    """
    t_rate
    """
    Mean variance inflation factor, one entry per raw factor, or `nothing` when the block has no exposure history.
    """
    mean_vif
    """
    Median exposure stability coefficient, one entry per raw factor, or `nothing` when the block has no exposure history.
    """
    stability
    """
    Average fraction of the universe at which the factor exposure is finite, one entry per raw factor, or `nothing` when the block has no exposure history.
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

For each column, the summary calls one function of the regression group or of the exposure group, and it aggregates the series that function returns over the observations. The only statistics it computes itself are the factor return statistics and two aggregates: the median of the stability, and the patch that reads a constant exposure as perfectly stable.

The answer is on the **raw** factor axis. The regression group answers on the reduced axis of the family re-basis, and the summary joins the two by name.

# Algorithm

 1. Read the factor return history off `csr`, and refuse a block that has none.
 2. Take the annualised mean, the annualised volatility and the Sharpe ratio of each series with [`factor_summary_return_stats`](@ref). An absent return takes no part.
 3. Take the lag-one autocorrelation of each series with [`factor_summary_autocorrelation`](@ref).
 4. Take the mean absolute t-statistic, the exceedance rate and the mean variance inflation factor with [`factor_summary_gram`](@ref), on the raw factor axis. A block with no exposure history gives `nothing` for all three.
 5. Take the median stability and the coverage with [`factor_summary_exposure`](@ref). A block with no exposure history gives `nothing` for both.
 6. Collect the nine columns and `ppy` into a [`FactorSummaryResult`](@ref).

# Arguments

  - `csfm`: A cross-sectional factor model block. A caller who holds a Prior Result writes `pr.rr`.
  - `ppy`: Periods per year. `252` annualises a daily fit, and the default of `1` reports the statistics per period.
  - `threshold`: Absolute t-statistic that the exceedance rate counts against.
  - `step`: Number of observations between the two cross-sections that the stability reads.
  - `weighting`: The [`AbstractOrthogonalityMetric`](@ref) whose weight history the stability reads.
  - `coverage_weighting`: The [`AbstractOrthogonalityMetric`](@ref) whose positive weights are the universe of the coverage. Its default of [`RegressionWeightMetric`](@ref) is the estimation universe of the fit.

# Validation

  - `ppy > 0`, else the function raises a `DomainError`.
  - `csfm.csr` is not `nothing`, else the function raises an `IsNothingError`.
  - `csfm.nf` is not `nothing` when `csfm.fcb` is present, else the function raises an `IsNothingError`.

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
