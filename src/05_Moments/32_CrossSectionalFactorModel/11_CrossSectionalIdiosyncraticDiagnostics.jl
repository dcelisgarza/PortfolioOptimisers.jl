"""
    standardised_idio_value(e::Real, v::Real)

Return one standardised idiosyncratic return.

The verb divides the idiosyncratic return by the standard deviation the fit predicted for it. The answer is `NaN` where the division has no meaning. A variance that is not positive, a variance that is not finite, and a return that is not finite each give `NaN`. A negative variance counts as zero, so a variance estimate that undershot gives `NaN` and does not raise.

# Mathematical definition

```math
\\begin{align}
z_{ti} &= \\begin{cases}
\\dfrac{\\varepsilon_{ti}}{\\hat{\\sigma}_{ti}} & \\text{if } \\hat{\\sigma}_{ti} > 0 \\text{ and the ratio is finite}\\,, \\\\
\\mathrm{NaN} & \\text{otherwise}\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:z_ti_idio])
  - $(math_dict[:eps_ti_idio])
  - $(math_dict[:sigma_ti_idio])
  - $(math_dict[:v_ti_idio])

# Arguments

  - `e`: One idiosyncratic return.
  - `v`: The idiosyncratic variance predicted for it.

# Returns

  - `z::Real`: The standardised return, or `NaN`.

# Related

  - [`standardised_idio_returns`](@ref)
  - [`idio_predicted_volatility`](@ref)
"""
function standardised_idio_value(e::Real, v::Real)
    s = sqrt(max(v, zero(v)))
    r = e / s
    return s > zero(s) && isfinite(r) ? r : oftype(r, NaN)
end
"""
    standardised_idio_returns(eps::MatNum, vs::MatNum) -> Matrix{<:Real}

Return the standardised idiosyncratic returns of a cross-sectional fit.

Every calibration series of the idiosyncratic group reads it. The fit predicted a variance for each asset at each observation, and the standardised return states how large the realised return was against that prediction. A fit that is well calibrated leaves cross-sections of standardised returns whose standard deviation is `1`.

# Mathematical definition

```math
\\begin{align}
z_{ti} &= \\frac{\\varepsilon_{ti}}{\\hat{\\sigma}_{ti}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:z_ti_idio])
  - $(math_dict[:eps_ti_idio])
  - $(math_dict[:sigma_ti_idio])
  - $(math_dict[:v_ti_idio])

An entry is `NaN` where the predicted volatility is not positive or the ratio is not finite, as [`standardised_idio_value`](@ref) states.

# Arguments

  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.

# Validation

  - `!isempty(eps)`.
  - `size(vs) == size(eps)`.

# Returns

  - `z::Matrix{<:Real}`: Standardised idiosyncratic returns `observations × assets`, of a floating-point type. An entry whose predicted volatility is zero, or whose return is not finite, is `NaN`.

# Examples

```jldoctest
julia> standardised_idio_returns([1.0 -2.0; 3.0 4.0], [0.25 4.0; 1.0 0.0])
2×2 Matrix{Float64}:
 2.0   -1.0
 3.0  NaN
```

# Related

  - [`standardised_idio_value`](@ref)
  - [`idio_calibration`](@ref)
  - [`idio_tail_rate`](@ref)
  - [`idio_kurtosis`](@ref)
  - [`idio_skewness`](@ref)
"""
function standardised_idio_returns(eps::MatNum, vs::MatNum)
    @argcheck(!isempty(eps), IsEmptyError("eps cannot be empty"))
    @argcheck(size(vs, 1) == size(eps, 1) && size(vs, 2) == size(eps, 2),
              DimensionMismatch("vs ($(size(vs, 1))×$(size(vs, 2))) must match eps ($(size(eps, 1))×$(size(eps, 2)))"))
    Tf = typeof(one(real(eltype(eps))) / sqrt(one(real(eltype(vs)))))
    T, N = size(eps)
    z = Matrix{Tf}(undef, T, N)
    for i in 1:N, t in 1:T
        z[t, i] = standardised_idio_value(Tf(eps[t, i]), Tf(vs[t, i]))
    end
    return z
end
"""
    idio_predicted_volatility(vs::MatNum)

Return the idiosyncratic volatility history the fit predicted.

A negative variance counts as zero, so a variance estimate that undershot gives `0` and does not raise. The two dependence series rank the assets by this quantity.

# Mathematical definition

```math
\\begin{align}
\\hat{\\sigma}_{ti} &= \\sqrt{\\max(v_{ti}, 0)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:sigma_ti_idio])
  - $(math_dict[:v_ti_idio])

# Arguments

  - `vs`: Idiosyncratic variance history `observations × assets`.

# Returns

  - `s::Matrix{<:Real}`: Predicted volatility `observations × assets`, of a floating-point type. A variance that is not finite gives `NaN`.

# Related

  - [`idio_vol_ic`](@ref)
  - [`idio_vol_residual_dependence`](@ref)
  - [`standardised_idio_value`](@ref)
"""
function idio_predicted_volatility(vs::MatNum)
    Tf = typeof(sqrt(one(real(eltype(vs)))))
    T, N = size(vs)
    s = Matrix{Tf}(undef, T, N)
    for i in 1:N, t in 1:T
        s[t, i] = sqrt(max(Tf(vs[t, i]), zero(Tf)))
    end
    return s
end
"""
    idio_row_moments(z::MatNum, t::Integer)

Return the count and the three central moments of one cross-section of standardised idiosyncratic returns.

The calibration, the excess kurtosis and the skewness are each a function of these four numbers alone. So the verb makes its two passes over the cross-section once, and three series read the answer. An entry that is not finite enters neither the count nor a moment.

# Mathematical definition

```math
\\begin{align}
\\bar{z}_{t} &= \\frac{1}{n_{t}} \\sum_{i \\in \\mathcal{F}_{t}} z_{ti}\\,, \\\\
m_{pt} &= \\frac{1}{n_{t}} \\sum_{i \\in \\mathcal{F}_{t}} \\left( z_{ti} - \\bar{z}_{t} \\right)^{p}\\,, \\quad p \\in \\{2, 3, 4\\}\\,.
\\end{align}
```

Where:

  - ``\\bar{z}_{t}``: Mean of the finite standardised returns of observation ``t``.
  - $(math_dict[:m_pt_idio])
  - $(math_dict[:z_ti_idio])
  - $(math_dict[:F_t_idio])
  - $(math_dict[:n_t_idio])

A cross-section whose finite entries are all equal has ``m_{pt} = 0`` for every ``p``.

# Algorithm

 1. Pass over row `t` of `z` once. Over the finite entries, find the count `n`, the sum `s`, the least entry `lo` and the greatest entry `hi`.
 2. When `n` is zero, return `NaN` for the three moments.
 3. Divide `s` by `n` and clamp the answer to `[lo, hi]`, giving the mean `m`. A mean of equal values can carry round-off, so without the clamp a constant cross-section would give a tiny positive `m2` in place of zero. With the clamp, `m` equals the common value exactly, and every deviation is zero.
 4. Pass over row `t` again. Sum the second, third and fourth powers of the deviations from `m`, and divide each sum by `n`, giving `m2`, `m3` and `m4`.

# Arguments

  - `z`: Standardised idiosyncratic returns `observations × assets`.
  - `t`: Position of the observation.

# Returns

  - `n::Int`: Number of assets at which the standardised return is finite.
  - `m2::Real`: Second central moment, or `NaN` when `n` is zero.
  - `m3::Real`: Third central moment, or `NaN` when `n` is zero.
  - `m4::Real`: Fourth central moment, or `NaN` when `n` is zero.

# Related

  - [`idio_calibration`](@ref)
  - [`idio_kurtosis`](@ref)
  - [`idio_skewness`](@ref)
"""
function idio_row_moments(z::MatNum, t::Integer)
    Tf = typeof(zero(real(eltype(z))) / one(Int))
    N = size(z, 2)
    n = 0
    s = zero(Tf)
    lo = typemax(Tf)
    hi = typemin(Tf)
    for i in 1:N
        v = z[t, i]
        if isfinite(v)
            x = Tf(v)
            n += 1
            s += x
            lo = min(lo, x)
            hi = max(hi, x)
        end
    end
    if n == 0
        return (; n = n, m2 = Tf(NaN), m3 = Tf(NaN), m4 = Tf(NaN))
    end
    m = clamp(s / n, lo, hi)
    m2 = zero(Tf)
    m3 = zero(Tf)
    m4 = zero(Tf)
    for i in 1:N
        v = z[t, i]
        if isfinite(v)
            d = Tf(v) - m
            d2 = d * d
            m2 += d2
            m3 += d2 * d
            m4 += d2 * d2
        end
    end
    return (; n = n, m2 = m2 / n, m3 = m3 / n, m4 = m4 / n)
end
"""
    idio_calibration(z::MatNum) -> Vector{<:Real}
    idio_calibration(eps::MatNum, vs::MatNum) -> Vector{<:Real}
    idio_calibration(csfm::CrossSectionalFactorModel) -> Vector{<:Real}

Return the cross-sectional standard deviation of the standardised idiosyncratic returns, one entry per observation.

It is the headline calibration series of a cross-sectional fit. The fit predicted a variance for each asset. When the prediction is right, the realised idiosyncratic returns divided by the predicted volatilities have a standard deviation of `1`. A series above `1` shows that the fit's specific risk is too small, and a series below `1` shows that it is too large.

# Mathematical definition

```math
\\begin{align}
\\varsigma_{t} &= \\sqrt{\\frac{n_{t}}{n_{t} - 1} m_{2t}}\\,.
\\end{align}
```

Where:

  - ``\\varsigma_{t}``: Sample standard deviation of the finite standardised returns of observation ``t``.
  - $(math_dict[:m_pt_idio])
  - $(math_dict[:n_t_idio])
  - $(math_dict[:F_t_idio])
  - $(math_dict[:z_ti_idio])

The deviation is the sample one, so it divides by ``n_{t} - 1``. It is not defined for ``n_{t} < 2``. A constant cross-section gives ``\\varsigma_{t} = 0``.

# Arguments

  - `z`: Standardised idiosyncratic returns `observations × assets`.
  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.
  - `csfm`: A cross-sectional factor model block.

# Validation

  - `!isempty(eps)` and `size(vs) == size(eps)`, on the two-argument form.
  - `csfm.csr` is not `nothing`, else the verb raises an `IsNothingError` that names `csr`.
  - `csfm.vs` is not `nothing`, else the verb raises an `IsNothingError` that names `vs`.

# Returns

  - `c::Vector{<:Real}`: The series, one entry per observation. An observation with fewer than two finite assets is `NaN`.

# Related

  - [`standardised_idio_returns`](@ref)
  - [`idio_row_moments`](@ref)
  - [`idio_calibration_summary`](@ref)
  - [`plot_idio_calibration`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function idio_calibration(z::MatNum)
    Tf = typeof(sqrt(zero(real(eltype(z))) / one(Int)))
    T = size(z, 1)
    c = Vector{Tf}(undef, T)
    for t in 1:T
        m = idio_row_moments(z, t)
        nT = Tf(m.n)
        c[t] = m.n >= 2 ? sqrt(m.m2 * nT / (nT - one(Tf))) : Tf(NaN)
    end
    return c
end
function idio_calibration(eps::MatNum, vs::MatNum)
    return idio_calibration(standardised_idio_returns(eps, vs))
end
function idio_calibration(csfm::CrossSectionalFactorModel)
    ep, vh = idio_diagnostic_data(csfm)
    return idio_calibration(ep, vh)
end
"""
    idio_tail_rate(z::MatNum; threshold::Real = 3) -> Vector{<:Real}
    idio_tail_rate(eps::MatNum, vs::MatNum; threshold::Real = 3) -> Vector{<:Real}
    idio_tail_rate(csfm::CrossSectionalFactorModel; threshold::Real = 3) -> Vector{<:Real}

Return the share of assets whose standardised idiosyncratic return exceeds a threshold, one entry per observation.

Under the normal law the expected rate is ``2 \\Phi(-c)``, which is about `0.0027` for a threshold of three. A series above it shows standardised returns with heavier tails than the normal law gives. The standardised returns of an equity factor model are often fat-tailed, so a rate above the Gaussian reference is common and is not by itself a defect of the fit.

# Mathematical definition

```math
\\begin{align}
r_{t} &= \\frac{1}{n_{t}} \\sum_{i \\in \\mathcal{F}_{t}} \\mathbb{1} \\left\\{ \\left| z_{ti} \\right| > c \\right\\}\\,.
\\end{align}
```

Where:

  - ``r_{t}``: Tail rate of observation ``t``.
  - ``c``: The threshold.
  - ``\\Phi``: Cumulative distribution function of the standard normal law.
  - $(math_dict[:z_ti_idio])
  - $(math_dict[:F_t_idio])
  - $(math_dict[:n_t_idio])

The rate is not defined for ``n_{t} = 0``. An entry that is not finite enters neither the count nor the sum.

# Arguments

  - `z`: Standardised idiosyncratic returns `observations × assets`.
  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.
  - `csfm`: A cross-sectional factor model block.
  - `threshold`: Absolute standardised return above which an asset enters the rate.

# Validation

  - `!isempty(eps)` and `size(vs) == size(eps)`, on the two-argument form.
  - `csfm.csr` is not `nothing`, else the verb raises an `IsNothingError` that names `csr`.
  - `csfm.vs` is not `nothing`, else the verb raises an `IsNothingError` that names `vs`.

# Returns

  - `r::Vector{<:Real}`: The series, one entry per observation. An observation with no finite asset is `NaN`.

# Related

  - [`standardised_idio_returns`](@ref)
  - [`idio_calibration`](@ref)
  - [`idio_calibration_summary`](@ref)
  - [`plot_idio_tail_rate`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function idio_tail_rate(z::MatNum; threshold::Real = 3)
    Tf = typeof(zero(real(eltype(z))) / one(Int))
    T, N = size(z)
    r = Vector{Tf}(undef, T)
    for t in 1:T
        nv = 0
        ne = 0
        for i in 1:N
            v = z[t, i]
            f = isfinite(v)
            nv += f
            ne += f && abs(v) > threshold
        end
        r[t] = nv > 0 ? Tf(ne) / Tf(nv) : Tf(NaN)
    end
    return r
end
function idio_tail_rate(eps::MatNum, vs::MatNum; threshold::Real = 3)
    return idio_tail_rate(standardised_idio_returns(eps, vs); threshold = threshold)
end
function idio_tail_rate(csfm::CrossSectionalFactorModel; threshold::Real = 3)
    ep, vh = idio_diagnostic_data(csfm)
    return idio_tail_rate(ep, vh; threshold = threshold)
end
"""
    idio_kurtosis(z::MatNum) -> Vector{<:Real}
    idio_kurtosis(eps::MatNum, vs::MatNum) -> Vector{<:Real}
    idio_kurtosis(csfm::CrossSectionalFactorModel) -> Vector{<:Real}

Return the cross-sectional excess kurtosis of the standardised idiosyncratic returns, one entry per observation.

The Gaussian reference is `0`, and a positive value shows a cross-section with heavier tails than the normal law gives. Read it beside [`idio_tail_rate`](@ref). The rate counts the assets beyond a threshold, and the kurtosis measures how far beyond it they went.

# Mathematical definition

```math
\\begin{align}
k_{t} &= \\frac{n_{t} - 1}{(n_{t} - 2)(n_{t} - 3)} \\left[ (n_{t} + 1) \\left( \\frac{m_{4t}}{m_{2t}^{2}} - 3 \\right) + 6 \\right]\\,.
\\end{align}
```

Where:

  - ``k_{t}``: Excess kurtosis of the finite standardised returns of observation ``t``.
  - $(math_dict[:m_pt_idio])
  - $(math_dict[:n_t_idio])
  - $(math_dict[:F_t_idio])
  - $(math_dict[:z_ti_idio])

It is the bias-corrected sample excess kurtosis, the estimator ``G_{2}`` of Joanes and Gill. It is not defined for ``n_{t} < 4``, or for a constant cross-section, where ``m_{2t} = 0``.

# Arguments

  - `z`: Standardised idiosyncratic returns `observations × assets`.
  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.
  - `csfm`: A cross-sectional factor model block.

# Validation

  - `!isempty(eps)` and `size(vs) == size(eps)`, on the two-argument form.
  - `csfm.csr` is not `nothing`, else the verb raises an `IsNothingError` that names `csr`.
  - `csfm.vs` is not `nothing`, else the verb raises an `IsNothingError` that names `vs`.

# Returns

  - `k::Vector{<:Real}`: The series, one entry per observation. An observation with fewer than four finite assets, or with a constant cross-section, is `NaN`.

# Related

  - [`standardised_idio_returns`](@ref)
  - [`idio_row_moments`](@ref)
  - [`idio_skewness`](@ref)
  - [`plot_idio_kurtosis`](@ref)
  - [`CrossSectionalFactorModel`](@ref)

# References

  - $(ref_dict[:joanesgill1998])
"""
function idio_kurtosis(z::MatNum)
    Tf = typeof(zero(real(eltype(z))) / one(Int))
    T = size(z, 1)
    k = Vector{Tf}(undef, T)
    for t in 1:T
        m = idio_row_moments(z, t)
        if m.n < 4
            k[t] = Tf(NaN)
        else
            nT = Tf(m.n)
            raw = m.m4 / (m.m2 * m.m2) - Tf(3)
            adj = (nT - one(Tf)) / ((nT - Tf(2)) * (nT - Tf(3)))
            k[t] = ((nT + one(Tf)) * raw + Tf(6)) * adj
        end
    end
    return k
end
function idio_kurtosis(eps::MatNum, vs::MatNum)
    return idio_kurtosis(standardised_idio_returns(eps, vs))
end
function idio_kurtosis(csfm::CrossSectionalFactorModel)
    ep, vh = idio_diagnostic_data(csfm)
    return idio_kurtosis(ep, vh)
end
"""
    idio_skewness(z::MatNum) -> Vector{<:Real}
    idio_skewness(eps::MatNum, vs::MatNum) -> Vector{<:Real}
    idio_skewness(csfm::CrossSectionalFactorModel) -> Vector{<:Real}

Return the cross-sectional skewness of the standardised idiosyncratic returns, one entry per observation.

The Gaussian reference is `0`. A cross-section that stays skewed shows a residual with a direction the factors did not take. A sector or a style that the exposures do not name can cause it.

# Mathematical definition

```math
\\begin{align}
s_{t} &= \\frac{m_{3t}}{m_{2t}^{3/2}} \\frac{\\sqrt{n_{t}(n_{t} - 1)}}{n_{t} - 2}\\,.
\\end{align}
```

Where:

  - ``s_{t}``: Skewness of the finite standardised returns of observation ``t``.
  - $(math_dict[:m_pt_idio])
  - $(math_dict[:n_t_idio])
  - $(math_dict[:F_t_idio])
  - $(math_dict[:z_ti_idio])

It is the bias-corrected sample skewness, the estimator ``G_{1}`` of Joanes and Gill. It is not defined for ``n_{t} < 3``, or for a constant cross-section, where ``m_{2t} = 0``.

# Arguments

  - `z`: Standardised idiosyncratic returns `observations × assets`.
  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.
  - `csfm`: A cross-sectional factor model block.

# Validation

  - `!isempty(eps)` and `size(vs) == size(eps)`, on the two-argument form.
  - `csfm.csr` is not `nothing`, else the verb raises an `IsNothingError` that names `csr`.
  - `csfm.vs` is not `nothing`, else the verb raises an `IsNothingError` that names `vs`.

# Returns

  - `s::Vector{<:Real}`: The series, one entry per observation. An observation with fewer than three finite assets, or with a constant cross-section, is `NaN`.

# Related

  - [`standardised_idio_returns`](@ref)
  - [`idio_row_moments`](@ref)
  - [`idio_kurtosis`](@ref)
  - [`plot_idio_skewness`](@ref)
  - [`CrossSectionalFactorModel`](@ref)

# References

  - $(ref_dict[:joanesgill1998])
"""
function idio_skewness(z::MatNum)
    Tf = typeof(sqrt(zero(real(eltype(z))) / one(Int)))
    T = size(z, 1)
    s = Vector{Tf}(undef, T)
    for t in 1:T
        m = idio_row_moments(z, t)
        if m.n < 3
            s[t] = Tf(NaN)
        else
            nT = Tf(m.n)
            raw = m.m3 / m.m2^Tf(1.5)
            s[t] = raw * sqrt(nT * (nT - one(Tf))) / (nT - Tf(2))
        end
    end
    return s
end
function idio_skewness(eps::MatNum, vs::MatNum)
    return idio_skewness(standardised_idio_returns(eps, vs))
end
function idio_skewness(csfm::CrossSectionalFactorModel)
    ep, vh = idio_diagnostic_data(csfm)
    return idio_skewness(ep, vh)
end
"""
    idio_vol_dependence(eps::MatNum, vs::MatNum, standardise::Bool, ties::Symbol)

Return the rank correlation of the predicted idiosyncratic volatility against the next observation's absolute idiosyncratic return, standardised or not.

It does the work of [`idio_vol_ic`](@ref) and of [`idio_vol_residual_dependence`](@ref). The two read the same two cross-sections, and differ only in whether they divide the target by the predicted volatility. Both score the volatility the fit predicted at observation ``t`` against what the asset realised at observation ``t + 1``, so the series has one entry fewer than the history. Those two docstrings state the mathematics.

# Algorithm

 1. Check that `eps` is not empty, that `vs` has its size, and that it has more than one observation.
 2. Compute the predicted volatility `sig` with [`idio_predicted_volatility`](@ref).
 3. For each observation `t` but the last, fill `a` with row `t` of `sig`. Fill `b` with the absolute idiosyncratic returns of observation `t + 1`. When `standardise` is `true`, divide each entry of `b` by the volatility predicted at observation `t`, through [`standardised_idio_value`](@ref).
 4. Correlate `a` and `b` with [`cs_spearman_correlation`](@ref) under `min_count = 5` and `ties`, giving entry `t` of `c`.

# Arguments

  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.
  - `standardise`: Divide the absolute return of observation ``t + 1`` by the volatility predicted at observation ``t``.
  - $(arg_dict[:cs_ties])

# Validation

  - `!isempty(eps)`.
  - `size(vs) == size(eps)`.
  - `size(eps, 1) > 1`.
  - The rules of [`cs_ranks`](@ref).

# Returns

  - `c::Vector{<:Real}`: The series, `observations - 1` entries. Entry `t` reads observations `t` and `t + 1`. An observation that shares fewer than five finite assets is `NaN`.

# Related

  - [`idio_vol_ic`](@ref)
  - [`idio_vol_residual_dependence`](@ref)
  - [`cs_spearman_correlation`](@ref)
  - [`idio_predicted_volatility`](@ref)
"""
function idio_vol_dependence(eps::MatNum, vs::MatNum, standardise::Bool, ties::Symbol)
    @argcheck(!isempty(eps), IsEmptyError("eps cannot be empty"))
    @argcheck(size(vs, 1) == size(eps, 1) && size(vs, 2) == size(eps, 2),
              DimensionMismatch("vs ($(size(vs, 1))×$(size(vs, 2))) must match eps ($(size(eps, 1))×$(size(eps, 2)))"))
    T, N = size(eps)
    @argcheck(T > 1,
              DimensionMismatch("eps ($T observations) must carry more than one observation"))
    Tf = typeof(one(real(eltype(eps))) / sqrt(one(real(eltype(vs)))))
    sig = idio_predicted_volatility(vs)
    c = Vector{Tf}(undef, T - 1)
    a = Vector{Tf}(undef, N)
    b = Vector{Tf}(undef, N)
    for t in 1:(T - 1)
        for i in 1:N
            a[i] = Tf(sig[t, i])
            m = abs(Tf(eps[t + 1, i]))
            b[i] = standardise ? standardised_idio_value(m, Tf(vs[t, i])) : m
        end
        c[t] = cs_spearman_correlation(a, b; min_count = 5, ties = ties)
    end
    return c
end
"""
    idio_vol_ic(eps::MatNum, vs::MatNum; ties::Symbol = :average) -> Vector{<:Real}
    idio_vol_ic(csfm::CrossSectionalFactorModel; ties::Symbol = :average) -> Vector{<:Real}

Return the information coefficient of the predicted idiosyncratic volatility, one entry per pair of observations.

The fit predicted a volatility for each asset at observation ``t``. The assets it called the most volatile should be the assets that moved the most at observation ``t + 1``. This series scores that ranking.

A high value shows a fit that ranks specific risk across the assets well. The series also responds to a broad cross-sectional effect such as size or liquidity. So read it beside [`idio_vol_residual_dependence`](@ref), which shows whether the level of the prediction still leaks into the standardised returns.

# Mathematical definition

```math
\\begin{align}
\\mathrm{IC}_{t} &= \\rho_{S} \\left( \\hat{\\sigma}_{t \\cdot}, \\left| \\varepsilon_{t + 1, \\cdot} \\right| \\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{IC}_{t}``: Information coefficient of observation ``t``, read against observation ``t + 1``.
  - ``\\hat{\\sigma}_{t \\cdot}``, ``\\varepsilon_{t + 1, \\cdot}``: The cross-sections of observations ``t`` and ``t + 1``, one entry per asset.
  - $(math_dict[:rho_S_cs])
  - $(math_dict[:sigma_ti_idio])
  - $(math_dict[:v_ti_idio])
  - $(math_dict[:eps_ti_idio])

The correlation is not defined over fewer than five common finite assets.

# Arguments

  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.
  - `csfm`: A cross-sectional factor model block.
  - $(arg_dict[:cs_ties])

# Validation

  - `!isempty(eps)`, `size(vs) == size(eps)` and `size(eps, 1) > 1`.
  - `csfm.csr` is not `nothing`, else the verb raises an `IsNothingError` that names `csr`.
  - `csfm.vs` is not `nothing`, else the verb raises an `IsNothingError` that names `vs`.
  - The rules of [`cs_ranks`](@ref).

# Returns

  - `c::Vector{<:Real}`: The series, `observations - 1` entries. An observation that shares fewer than five finite assets is `NaN`.

# Related

  - [`idio_vol_dependence`](@ref)
  - [`idio_vol_residual_dependence`](@ref)
  - [`plot_idio_vol_ic`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function idio_vol_ic(eps::MatNum, vs::MatNum; ties::Symbol = :average)
    return idio_vol_dependence(eps, vs, false, ties)
end
function idio_vol_ic(csfm::CrossSectionalFactorModel; ties::Symbol = :average)
    ep, vh = idio_diagnostic_data(csfm)
    return idio_vol_ic(ep, vh; ties = ties)
end
"""
    idio_vol_residual_dependence(eps::MatNum, vs::MatNum;
                                 ties::Symbol = :average) -> Vector{<:Real}
    idio_vol_residual_dependence(csfm::CrossSectionalFactorModel;
                                 ties::Symbol = :average) -> Vector{<:Real}

Return the rank correlation of the predicted idiosyncratic volatility against the next observation's standardised absolute idiosyncratic return, one entry per pair of observations.

Division of the realised move by the predicted volatility should remove the level of the prediction. So a fit that is well calibrated leaves a series near `0`, because the size of an asset's standardised move does not depend on how volatile the fit said the asset would be. A series that stays positive shows a fit that under-predicts the volatile assets, and a series that stays negative shows a fit that over-predicts them.

The target divides the absolute return of observation ``t + 1`` by the volatility predicted at observation ``t``, not at observation ``t + 1``. The notation ``|z_{t+1}|`` can name either quantity, and the two differ wherever the prediction moved between the two observations.

Read it beside [`idio_vol_ic`](@ref). A fit that ranks well and leaves no residual dependence has a high information coefficient and a dependence near `0`.

# Mathematical definition

```math
\\begin{align}
d_{t} &= \\rho_{S} \\left( \\hat{\\sigma}_{t \\cdot}, \\frac{\\left| \\varepsilon_{t + 1, \\cdot} \\right|}{\\hat{\\sigma}_{t \\cdot}} \\right)\\,.
\\end{align}
```

Where:

  - ``d_{t}``: Residual dependence of observation ``t``, read against observation ``t + 1``.
  - ``\\hat{\\sigma}_{t \\cdot}``, ``\\varepsilon_{t + 1, \\cdot}``: The cross-sections of observations ``t`` and ``t + 1``, one entry per asset. The division is per asset, and it is `NaN` where ``\\hat{\\sigma}_{ti}`` is zero.
  - $(math_dict[:rho_S_cs])
  - $(math_dict[:sigma_ti_idio])
  - $(math_dict[:v_ti_idio])
  - $(math_dict[:eps_ti_idio])

The correlation is not defined over fewer than five common finite assets.

# Arguments

  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.
  - `csfm`: A cross-sectional factor model block.
  - $(arg_dict[:cs_ties])

# Validation

  - `!isempty(eps)`, `size(vs) == size(eps)` and `size(eps, 1) > 1`.
  - `csfm.csr` is not `nothing`, else the verb raises an `IsNothingError` that names `csr`.
  - `csfm.vs` is not `nothing`, else the verb raises an `IsNothingError` that names `vs`.
  - The rules of [`cs_ranks`](@ref).

# Returns

  - `d::Vector{<:Real}`: The series, `observations - 1` entries. An observation that shares fewer than five finite assets is `NaN`.

# Related

  - [`idio_vol_dependence`](@ref)
  - [`idio_vol_ic`](@ref)
  - [`plot_idio_vol_residual_dependence`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function idio_vol_residual_dependence(eps::MatNum, vs::MatNum; ties::Symbol = :average)
    return idio_vol_dependence(eps, vs, true, ties)
end
function idio_vol_residual_dependence(csfm::CrossSectionalFactorModel;
                                      ties::Symbol = :average)
    ep, vh = idio_diagnostic_data(csfm)
    return idio_vol_residual_dependence(ep, vh; ties = ties)
end
"""
    idio_nan_mean(v::VecNum)

Return the mean of the finite entries of a series.

A diagnostic series is `NaN` at an observation that had too few assets, and the summary reads the observations that answered. This verb states that rule once for the four means the summary takes.

# Mathematical definition

```math
\\begin{align}
\\bar{a} &= \\frac{1}{|\\mathcal{A}|} \\sum_{t \\in \\mathcal{A}} a_{t}\\,.
\\end{align}
```

Where:

  - ``\\bar{a}``: Mean of the finite entries.
  - $(math_dict[:a_t_series])
  - $(math_dict[:A_series_fin])

The mean is not defined for an empty ``\\mathcal{A}``.

# Arguments

  - `v`: The series.

# Returns

  - `m::Real`: The mean of the finite entries, or `NaN` when the series has none.

# Related

  - [`idio_calibration_summary`](@ref)
  - [`idio_nan_median`](@ref)
"""
function idio_nan_mean(v::VecNum)
    Tf = typeof(zero(real(eltype(v))) / one(Int))
    s = zero(Tf)
    n = 0
    for x in v
        if isfinite(x)
            s += Tf(x)
            n += 1
        end
    end
    return n > 0 ? s / n : Tf(NaN)
end
"""
    idio_nan_median(v::VecNum)

Return the median of the finite entries of a series.

The summary reads the median of the calibration series beside its mean. One observation with a cross-section that is almost constant moves the mean, but not the median.

# Mathematical definition

```math
\\begin{align}
\\tilde{a} &= \\operatorname{median} \\left\\{ a_{t} : t \\in \\mathcal{A} \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\tilde{a}``: Median of the finite entries. For an even count it is the mean of the two middle entries.
  - $(math_dict[:a_t_series])
  - $(math_dict[:A_series_fin])

The median is not defined for an empty ``\\mathcal{A}``.

# Arguments

  - `v`: The series.

# Returns

  - `m::Real`: The median of the finite entries, or `NaN` when the series has none.

# Related

  - [`idio_calibration_summary`](@ref)
  - [`idio_nan_mean`](@ref)
"""
function idio_nan_median(v::VecNum)
    Tf = typeof(zero(real(eltype(v))) / one(Int))
    f = Vector{Tf}(undef, 0)
    sizehint!(f, length(v))
    for x in v
        if isfinite(x)
            push!(f, Tf(x))
        end
    end
    return isempty(f) ? Tf(NaN) : Tf(Statistics.median(f))
end
"""
    idio_calibration_summary(eps::MatNum, vs::MatNum; threshold::Real = 3)
    idio_calibration_summary(csfm::CrossSectionalFactorModel; threshold::Real = 3)

Return the five time-aggregated numbers of the calibration of a cross-sectional fit.

The four calibration series each answer per observation. To judge a whole fit, read their time aggregates. The five numbers are the mean and the median of [`idio_calibration`](@ref), and the means of [`idio_kurtosis`](@ref), [`idio_skewness`](@ref) and [`idio_tail_rate`](@ref). Each aggregate skips the observations that had too few assets to answer.

Under the normal law the expected values are `1`, `1`, `0`, `0` and ``2 \\Phi(-c)``, where ``\\Phi`` is the cumulative distribution function of the standard normal law and ``c`` is the threshold. A fit of an equity universe usually has a positive excess kurtosis and a tail rate above the Gaussian reference. Read the first two numbers for the scale of the specific risk, and the last three for the shape of its tails.

# Algorithm

 1. Compute the standardised returns `z` once with [`standardised_idio_returns`](@ref).
 2. Compute the calibration series `cs` from `z` with [`idio_calibration`](@ref).
 3. Take the mean of `cs` with [`idio_nan_mean`](@ref) and its median with [`idio_nan_median`](@ref), giving `mean_cs_std` and `median_cs_std`.
 4. Compute the kurtosis, skewness and tail rate series from `z`, and take the mean of each with [`idio_nan_mean`](@ref), giving `mean_kurtosis`, `mean_skewness` and `mean_tail_rate`.

# Arguments

  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.
  - `csfm`: A cross-sectional factor model block.
  - `threshold`: Absolute standardised return above which an asset enters the tail rate.

# Validation

  - `!isempty(eps)` and `size(vs) == size(eps)`, on the two-argument form.
  - `csfm.csr` is not `nothing`, else the verb raises an `IsNothingError` that names `csr`.
  - `csfm.vs` is not `nothing`, else the verb raises an `IsNothingError` that names `vs`.

# Returns

  - `mean_cs_std::Real`: Mean of the calibration series.
  - `median_cs_std::Real`: Median of the calibration series.
  - `mean_kurtosis::Real`: Mean of the excess kurtosis series.
  - `mean_skewness::Real`: Mean of the skewness series.
  - `mean_tail_rate::Real`: Mean of the tail rate series.

# Related

  - [`idio_calibration`](@ref)
  - [`idio_kurtosis`](@ref)
  - [`idio_skewness`](@ref)
  - [`idio_tail_rate`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function idio_calibration_summary(eps::MatNum, vs::MatNum; threshold::Real = 3)
    z = standardised_idio_returns(eps, vs)
    cs = idio_calibration(z)
    return (; mean_cs_std = idio_nan_mean(cs), median_cs_std = idio_nan_median(cs),
            mean_kurtosis = idio_nan_mean(idio_kurtosis(z)),
            mean_skewness = idio_nan_mean(idio_skewness(z)),
            mean_tail_rate = idio_nan_mean(idio_tail_rate(z; threshold = threshold)))
end
function idio_calibration_summary(csfm::CrossSectionalFactorModel; threshold::Real = 3)
    ep, vh = idio_diagnostic_data(csfm)
    return idio_calibration_summary(ep, vh; threshold = threshold)
end
"""
    idio_diagnostic_data(csfm::CrossSectionalFactorModel)
    idio_diagnostic_data(csr::Nothing, vs::Option{<:MatNum})
    idio_diagnostic_data(csr::CrossSectionalRegression, vs::Nothing)
    idio_diagnostic_data(csr::CrossSectionalRegression, vs::MatNum)

Return the idiosyncratic return history and the idiosyncratic variance history that an idiosyncratic diagnostic reads off a factor model block.

The group reads the residual of the fit against the variance the fit predicted for it. Neither history has a factor axis, so the group takes no lag and no family re-basis, and it reads the two histories as the block wrote them. Dispatch, not a branch, selects the absent case, and its message names the field the caller must populate.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `csr`: The `csr` field of the block, or `nothing`.
  - `vs`: The `vs` field of the block, or `nothing`.

# Validation

  - `csfm.csr` is not `nothing`, else the verb raises an `IsNothingError` that names `csr`.
  - `csfm.vs` is not `nothing`, else the verb raises an `IsNothingError` that names `vs`.

# Returns

  - `eps::MatNum`: Idiosyncratic return history `observations × assets`.
  - `vs::MatNum`: Idiosyncratic variance history `observations × assets`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`CrossSectionalRegression`](@ref)
  - [`idio_calibration`](@ref)
  - [`idio_vol_ic`](@ref)
"""
function idio_diagnostic_data(csfm::CrossSectionalFactorModel)
    return idio_diagnostic_data(csfm.csr, csfm.vs)
end
function idio_diagnostic_data(::Nothing, ::Option{<:MatNum})
    return throw(IsNothingError("csr cannot be nothing: an idiosyncratic diagnostic reads the idiosyncratic returns of the block"))
end
function idio_diagnostic_data(::CrossSectionalRegression, ::Nothing)
    return throw(IsNothingError("vs cannot be nothing: an idiosyncratic diagnostic reads the idiosyncratic variance history of the block"))
end
function idio_diagnostic_data(csr::CrossSectionalRegression, vs::MatNum)
    return csr.eps, vs
end

export standardised_idio_returns, idio_calibration, idio_tail_rate, idio_kurtosis,
       idio_skewness, idio_vol_ic, idio_vol_residual_dependence, idio_calibration_summary
