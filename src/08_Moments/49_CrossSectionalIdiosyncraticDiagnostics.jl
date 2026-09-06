"""
    standardised_idio_value(e::Real, v::Real)

Return one standardised idiosyncratic return.

The idiosyncratic return is divided by the standard deviation the fit predicted for it, and the answer is `NaN` wherever that division has no meaning: a variance that is not positive, a variance that is not finite, and a return that is not finite each read `NaN`. A negative variance is clamped to zero before the square root, so the verb answers rather than raises on a variance estimate that undershot.

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

It is the level-0 kernel of the idiosyncratic group. Every calibration series of this file reads it: the fit predicted a variance for each asset at each observation, and the standardised return states how large the realised return was against that prediction. A well calibrated fit leaves a cross-section of standardised returns whose standard deviation is `1`.

# Mathematical definition

```math
z_{ti} = \\frac{\\varepsilon_{ti}}{\\sqrt{\\max(v_{ti}, 0)}}
```

Where:

  - ``\\varepsilon_{ti}``: Idiosyncratic return of asset ``i`` at observation ``t``.
  - ``v_{ti}``: Idiosyncratic variance predicted for asset ``i`` at observation ``t``.

# Arguments

  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.

# Validation

  - `!isempty(eps)`.
  - `size(vs) == size(eps)`.

# Returns

  - `z::Matrix{<:Real}`: Standardised idiosyncratic returns `observations × assets`. An entry whose predicted volatility is zero, or whose return is not finite, is `NaN`.

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
    Tf = promote_type(float(real(eltype(eps))), float(real(eltype(vs))))
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

A negative variance is clamped to zero before the square root, so a variance estimate that undershot answers `0` rather than raising. The two dependence series read it as the quantity whose ranking they score.

# Arguments

  - `vs`: Idiosyncratic variance history `observations × assets`.

# Returns

  - `s::Matrix{<:Real}`: Predicted volatility `observations × assets`.

# Related

  - [`idio_vol_ic`](@ref)
  - [`idio_vol_residual_dependence`](@ref)
  - [`standardised_idio_value`](@ref)
"""
function idio_predicted_volatility(vs::MatNum)
    Tf = float(real(eltype(vs)))
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

The calibration, the excess kurtosis and the skewness are each a function of these four numbers alone, so the two passes over the cross-section are made once and read three times. An entry that is not finite enters neither the count nor a moment, which is the rule the reference implementation applies.

# Mathematical definition

```math
m_{p} = \\frac{1}{n} \\sum_{i \\in \\mathcal{F}} \\left( z_{i} - \\bar{z} \\right)^{p}, \\qquad \\bar{z} = \\frac{1}{n} \\sum_{i \\in \\mathcal{F}} z_{i}
```

Where:

  - ``\\mathcal{F}``: The assets at which the standardised return is finite.
  - ``n``: Size of ``\\mathcal{F}``.

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
    Tf = float(real(eltype(z)))
    N = size(z, 2)
    n = 0
    s = zero(Tf)
    for i in 1:N
        v = z[t, i]
        if isfinite(v)
            n += 1
            s += Tf(v)
        end
    end
    if n == 0
        return (; n = n, m2 = Tf(NaN), m3 = Tf(NaN), m4 = Tf(NaN))
    end
    m = s / n
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

It is the headline calibration series of a cross-sectional fit. The fit predicted a variance for each asset, and dividing the realised idiosyncratic return by the predicted volatility leaves a cross-section whose standard deviation is `1` when the prediction was right. A series that sits above `1` is a fit whose specific risk is too small, and one that sits below `1` is a fit whose specific risk is too large.

The deviation is the sample one, so it divides by ``n - 1``. An observation at which fewer than two assets carry a finite standardised return reads `NaN`.

# Mathematical definition

```math
c_{t} = \\sqrt{\\frac{n_{t}}{n_{t} - 1} m_{2t}}
```

Where:

  - ``m_{2t}``: Second central moment of the cross-section of observation ``t``.
  - ``n_{t}``: Number of assets at which the standardised return is finite.

# Arguments

  - `z`: Standardised idiosyncratic returns `observations × assets`.
  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.
  - `csfm`: A cross-sectional factor model block.

# Validation

  - `!isempty(eps)` and `size(vs) == size(eps)`, on the two-argument form.
  - `csfm.csr` is not `nothing`, else an `IsNothingError` naming `csr` is raised.
  - `csfm.vs` is not `nothing`, else an `IsNothingError` naming `vs` is raised.

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
    Tf = float(real(eltype(z)))
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

The Gaussian reference of a threshold of three is ``2 \\Phi(-3) \\approx 0.0027``, so a series that sits above it is a fit whose standardised returns carry heavier tails than the normal law implies. A rate of one to three percent is ordinary for an equity factor model, and it is not by itself a defect of the fit.

An asset enters the denominator when its standardised return is finite, and the numerator when the absolute value of that return exceeds the threshold. An observation at which no asset carries a finite standardised return reads `NaN`.

# Mathematical definition

```math
r_{t} = \\frac{1}{n_{t}} \\sum_{i \\in \\mathcal{F}_{t}} \\mathbb{1} \\left\\{ \\left| z_{ti} \\right| > c \\right\\}
```

Where:

  - ``\\mathcal{F}_{t}``: The assets at which the standardised return is finite.
  - ``n_{t}``: Size of ``\\mathcal{F}_{t}``.
  - ``c``: The threshold.

# Arguments

  - `z`: Standardised idiosyncratic returns `observations × assets`.
  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.
  - `csfm`: A cross-sectional factor model block.
  - `threshold`: Absolute standardised return above which an asset enters the rate.

# Validation

  - `!isempty(eps)` and `size(vs) == size(eps)`, on the two-argument form.
  - `csfm.csr` is not `nothing`, else an `IsNothingError` naming `csr` is raised.
  - `csfm.vs` is not `nothing`, else an `IsNothingError` naming `vs` is raised.

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
    Tf = float(real(eltype(z)))
    T, N = size(z)
    r = Vector{Tf}(undef, T)
    for t in 1:T
        nv = 0
        ne = 0
        for i in 1:N
            v = z[t, i]
            nv += isfinite(v)
            ne += abs(v) > threshold
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

The Gaussian reference is `0`, and a positive value is a cross-section whose tails are heavier than the normal law implies. Read it beside [`idio_tail_rate`](@ref): the rate counts the assets beyond a threshold, and the kurtosis weighs how far beyond it they went.

The estimate is bias corrected, so it matches the ordinary sample estimator of the excess kurtosis. An observation at which fewer than four assets carry a finite standardised return reads `NaN`, and so does one whose cross-section is constant.

# Mathematical definition

```math
k_{t} = \\frac{n_{t} - 1}{(n_{t} - 2)(n_{t} - 3)} \\left[ (n_{t} + 1) \\left( \\frac{m_{4t}}{m_{2t}^{2}} - 3 \\right) + 6 \\right]
```

Where:

  - ``m_{2t}``, ``m_{4t}``: Second and fourth central moments of the cross-section of observation ``t``.
  - ``n_{t}``: Number of assets at which the standardised return is finite.

# Arguments

  - `z`: Standardised idiosyncratic returns `observations × assets`.
  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.
  - `csfm`: A cross-sectional factor model block.

# Validation

  - `!isempty(eps)` and `size(vs) == size(eps)`, on the two-argument form.
  - `csfm.csr` is not `nothing`, else an `IsNothingError` naming `csr` is raised.
  - `csfm.vs` is not `nothing`, else an `IsNothingError` naming `vs` is raised.

# Returns

  - `k::Vector{<:Real}`: The series, one entry per observation. An observation with fewer than four finite assets is `NaN`.

# Related

  - [`standardised_idio_returns`](@ref)
  - [`idio_row_moments`](@ref)
  - [`idio_skewness`](@ref)
  - [`plot_idio_kurtosis`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function idio_kurtosis(z::MatNum)
    Tf = float(real(eltype(z)))
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

The Gaussian reference is `0`. A cross-section that is persistently skewed is a fit whose residual carries a direction the factors did not take, which a sector or a style the exposures do not name can produce.

The estimate is bias corrected, so it matches the ordinary sample estimator of the skewness. An observation at which fewer than three assets carry a finite standardised return reads `NaN`, and so does one whose cross-section is constant.

# Mathematical definition

```math
s_{t} = \\frac{m_{3t}}{m_{2t}^{3/2}} \\frac{\\sqrt{n_{t}(n_{t} - 1)}}{n_{t} - 2}
```

Where:

  - ``m_{2t}``, ``m_{3t}``: Second and third central moments of the cross-section of observation ``t``.
  - ``n_{t}``: Number of assets at which the standardised return is finite.

# Arguments

  - `z`: Standardised idiosyncratic returns `observations × assets`.
  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.
  - `csfm`: A cross-sectional factor model block.

# Validation

  - `!isempty(eps)` and `size(vs) == size(eps)`, on the two-argument form.
  - `csfm.csr` is not `nothing`, else an `IsNothingError` naming `csr` is raised.
  - `csfm.vs` is not `nothing`, else an `IsNothingError` naming `vs` is raised.

# Returns

  - `s::Vector{<:Real}`: The series, one entry per observation. An observation with fewer than three finite assets is `NaN`.

# Related

  - [`standardised_idio_returns`](@ref)
  - [`idio_row_moments`](@ref)
  - [`idio_kurtosis`](@ref)
  - [`plot_idio_skewness`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function idio_skewness(z::MatNum)
    Tf = float(real(eltype(z)))
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
    idio_vol_dependence(eps::MatNum, vs::MatNum, standardise::Bool)

Return the rank correlation of the predicted idiosyncratic volatility against the next observation's absolute idiosyncratic return, standardised or not.

It is the worker of [`idio_vol_ic`](@ref) and of [`idio_vol_residual_dependence`](@ref), which read the same two cross-sections and differ only in whether the target is divided by the predicted volatility. Both score the volatility the fit predicted at observation ``t`` against what the asset realised at observation ``t + 1``, so the series is one entry shorter than the history.

The correlation is taken over the assets at which both cross-sections are finite, and an observation that shares fewer than five such assets reads `NaN`.

# Arguments

  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.
  - `standardise`: Divide the absolute return of observation ``t + 1`` by the volatility predicted at observation ``t``.

# Validation

  - `!isempty(eps)`.
  - `size(vs) == size(eps)`.
  - `size(eps, 1) > 1`.

# Returns

  - `c::Vector{<:Real}`: The series, `observations - 1` entries. Entry `t` reads observations `t` and `t + 1`.

# Related

  - [`idio_vol_ic`](@ref)
  - [`idio_vol_residual_dependence`](@ref)
  - [`cs_spearman_correlation`](@ref)
  - [`idio_predicted_volatility`](@ref)
"""
function idio_vol_dependence(eps::MatNum, vs::MatNum, standardise::Bool)
    @argcheck(!isempty(eps), IsEmptyError("eps cannot be empty"))
    @argcheck(size(vs, 1) == size(eps, 1) && size(vs, 2) == size(eps, 2),
              DimensionMismatch("vs ($(size(vs, 1))×$(size(vs, 2))) must match eps ($(size(eps, 1))×$(size(eps, 2)))"))
    T, N = size(eps)
    @argcheck(T > 1,
              DimensionMismatch("eps ($T observations) must carry more than one observation"))
    Tf = promote_type(float(real(eltype(eps))), float(real(eltype(vs))))
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
        c[t] = cs_spearman_correlation(a, b; min_count = 5)
    end
    return c
end
"""
    idio_vol_ic(eps::MatNum, vs::MatNum) -> Vector{<:Real}
    idio_vol_ic(csfm::CrossSectionalFactorModel) -> Vector{<:Real}

Return the information coefficient of the predicted idiosyncratic volatility, one entry per pair of observations.

The fit predicted a volatility for each asset at observation ``t``, and the assets it called the most volatile should be the assets that moved the most at observation ``t + 1``. This series scores that ranking with the rank correlation of the predicted volatility against the absolute idiosyncratic return of the next observation.

A high value is a fit that ranks specific risk across the assets well. The series also picks up a broad cross-sectional effect such as size or liquidity, so read it beside [`idio_vol_residual_dependence`](@ref), which states whether the level of the prediction still leaks into what the fit standardised.

# Mathematical definition

```math
\\mathrm{IC}_{t} = \\rho_{S} \\left( \\hat{\\sigma}_{t \\cdot}, \\left| \\boldsymbol{\\varepsilon}_{t + 1, \\cdot} \\right| \\right)
```

Where:

  - ``\\hat{\\sigma}_{ti} = \\sqrt{\\max(v_{ti}, 0)}``: Predicted idiosyncratic volatility.
  - ``\\rho_{S}``: The cross-sectional rank correlation, over the assets at which both cross-sections are finite.

# Arguments

  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.
  - `csfm`: A cross-sectional factor model block.

# Validation

  - `!isempty(eps)`, `size(vs) == size(eps)` and `size(eps, 1) > 1`.
  - `csfm.csr` is not `nothing`, else an `IsNothingError` naming `csr` is raised.
  - `csfm.vs` is not `nothing`, else an `IsNothingError` naming `vs` is raised.

# Returns

  - `c::Vector{<:Real}`: The series, `observations - 1` entries. An observation sharing fewer than five finite assets is `NaN`.

# Related

  - [`idio_vol_dependence`](@ref)
  - [`idio_vol_residual_dependence`](@ref)
  - [`plot_idio_vol_ic`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function idio_vol_ic(eps::MatNum, vs::MatNum)
    return idio_vol_dependence(eps, vs, false)
end
function idio_vol_ic(csfm::CrossSectionalFactorModel)
    ep, vh = idio_diagnostic_data(csfm)
    return idio_vol_ic(ep, vh)
end
"""
    idio_vol_residual_dependence(eps::MatNum, vs::MatNum) -> Vector{<:Real}
    idio_vol_residual_dependence(csfm::CrossSectionalFactorModel) -> Vector{<:Real}

Return the rank correlation of the predicted idiosyncratic volatility against the next observation's standardised absolute idiosyncratic return, one entry per pair of observations.

Dividing the realised move by the volatility the fit predicted should remove the level of that prediction. So a well calibrated fit leaves a series near `0`: how large an asset's standardised move was should not depend on how volatile the fit said the asset would be. A series that stays positive is a fit that under-predicts the volatile assets, and one that stays negative is a fit that over-predicts them.

The target divides the absolute return of observation ``t + 1`` by the volatility predicted at observation ``t``, and not by the volatility predicted at observation ``t + 1``. Both quantities are written ``z_{t+1}`` in the literature, and they differ wherever the prediction moved between the two observations. This verb reproduces the reference implementation, which divides by the volatility of observation ``t``.

Read it beside [`idio_vol_ic`](@ref). A fit that ranks well and leaves no residual dependence carries a high information coefficient and a dependence near `0`.

# Mathematical definition

```math
d_{t} = \\rho_{S} \\left( \\hat{\\sigma}_{t \\cdot}, \\frac{\\left| \\boldsymbol{\\varepsilon}_{t + 1, \\cdot} \\right|}{\\hat{\\sigma}_{t \\cdot}} \\right)
```

Where:

  - ``\\hat{\\sigma}_{ti} = \\sqrt{\\max(v_{ti}, 0)}``: Predicted idiosyncratic volatility.
  - ``\\rho_{S}``: The cross-sectional rank correlation, over the assets at which both cross-sections are finite.

# Arguments

  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.
  - `csfm`: A cross-sectional factor model block.

# Validation

  - `!isempty(eps)`, `size(vs) == size(eps)` and `size(eps, 1) > 1`.
  - `csfm.csr` is not `nothing`, else an `IsNothingError` naming `csr` is raised.
  - `csfm.vs` is not `nothing`, else an `IsNothingError` naming `vs` is raised.

# Returns

  - `d::Vector{<:Real}`: The series, `observations - 1` entries. An observation sharing fewer than five finite assets is `NaN`.

# Related

  - [`idio_vol_dependence`](@ref)
  - [`idio_vol_ic`](@ref)
  - [`plot_idio_vol_residual_dependence`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function idio_vol_residual_dependence(eps::MatNum, vs::MatNum)
    return idio_vol_dependence(eps, vs, true)
end
function idio_vol_residual_dependence(csfm::CrossSectionalFactorModel)
    ep, vh = idio_diagnostic_data(csfm)
    return idio_vol_residual_dependence(ep, vh)
end
"""
    idio_nan_mean(v::VecNum)

Return the mean of the finite entries of a series.

A diagnostic series carries `NaN` at an observation that had too few assets, and the summary reads the observations that answered. This verb states that rule once for the four means the summary takes.

# Arguments

  - `v`: The series.

# Returns

  - `m::Real`: The mean of the finite entries, or `NaN` when the series carries none.

# Related

  - [`idio_calibration_summary`](@ref)
  - [`idio_nan_median`](@ref)
"""
function idio_nan_mean(v::VecNum)
    Tf = float(real(eltype(v)))
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

The summary reads the median of the calibration series beside its mean, because one observation whose cross-section was nearly constant moves the mean and not the median.

# Arguments

  - `v`: The series.

# Returns

  - `m::Real`: The median of the finite entries, or `NaN` when the series carries none.

# Related

  - [`idio_calibration_summary`](@ref)
  - [`idio_nan_mean`](@ref)
"""
function idio_nan_median(v::VecNum)
    Tf = float(real(eltype(v)))
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

The four calibration series each answer per observation, and a caller who judges a whole fit reads their time aggregate instead. The five numbers are the mean and the median of [`idio_calibration`](@ref), and the means of [`idio_kurtosis`](@ref), of [`idio_skewness`](@ref) and of [`idio_tail_rate`](@ref). Every aggregate skips the observations that had too few assets to answer.

Under the normal law the expected values are `1`, `1`, `0`, `0` and ``2 \\Phi(-c)``. A fit of an equity universe ordinarily carries a positive excess kurtosis and a tail rate above the Gaussian reference, so read the first two numbers for the scale of the specific risk and the last three for the shape of its tails.

# Arguments

  - `eps`: Idiosyncratic return history `observations × assets`.
  - `vs`: Idiosyncratic variance history `observations × assets`.
  - `csfm`: A cross-sectional factor model block.
  - `threshold`: Absolute standardised return above which an asset enters the tail rate.

# Validation

  - `!isempty(eps)` and `size(vs) == size(eps)`, on the two-argument form.
  - `csfm.csr` is not `nothing`, else an `IsNothingError` naming `csr` is raised.
  - `csfm.vs` is not `nothing`, else an `IsNothingError` naming `vs` is raised.

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

Return the idiosyncratic return history and the idiosyncratic variance history an idiosyncratic diagnostic reads off a factor model block.

The group reads the residual of the fit against the variance the fit predicted for it. Neither history carries a factor axis, so the group takes no lag and no family re-basis, and the two histories are read as the block wrote them. The absent case is the dispatch rather than a branch, and its message names the field the caller must populate.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `csr`: The `csr` field of the block, or `nothing`.
  - `vs`: The `vs` field of the block, or `nothing`.

# Validation

  - `csfm.csr` is not `nothing`, else an `IsNothingError` naming `csr` is raised.
  - `csfm.vs` is not `nothing`, else an `IsNothingError` naming `vs` is raised.

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
