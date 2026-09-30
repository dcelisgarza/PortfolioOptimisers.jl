"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the weight of each lag of a HAC estimate, in the number type of `o`.

A count of lags names the Bartlett weights ``k_{i} = 1 - i / (L + 1)``. A vector is already the
weight of each lag, such as the kernel of [`hac_row_kernel`](@ref), and it is returned as it is.
Without a HAC adjustment there is no lag.

# Arguments

  - `hac_lags::Option{<:Union{<:Integer, <:VecNum}}`: Count of HAC lags, the weight of each lag, or
    `nothing`.
  - `o::Number`: A number whose type the Bartlett weights take.

# Returns

  - `k::VecNum`: The weight of each lag, empty without a HAC adjustment.

# Related

  - [`regime_bias_table`](@ref)
  - [`hac_row_kernel`](@ref)
"""
function hac_lag_weights(::Nothing, o::Number)
    return typeof(o)[]
end
function hac_lag_weights(hac_lags::Integer, o::Number)
    return [one(o) - i * one(o) / (hac_lags + 1) for i in 1:hac_lags]
end
function hac_lag_weights(k::VecNum, ::Number)
    return k
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Adds one point of the grid to the moments of the damped lag terms: the log of the Laplace
transform of a HAC variance estimate, and one column of the inverse on the newest rows.

The estimate is ``x^{\\top} M x`` of iid standard Normal returns, newest first, with
``M_{jj} = c \\mu_{j}`` and ``M_{j, j+i} = c \\mu_{j} w_{i}``, ``c = 1 - \\lambda``: the newer
return of each lagged product carries the weight. The row weights mix two variances, ``\\mu_{j} = a \\lambda^{j} + b \\lambda^{j - d}`` for ``j \\geq d``. The banded LDLᵀ factorisation of ``I + s M`` adds one row at a
time, and the rows of ``L^{-1}`` on the newest columns follow it on the same band, so

```math
\\begin{align}
\\left[(I + s M)^{-1}\\right]_{a c} &= \\sum_{j} \\frac{(L^{-1})_{j a} (L^{-1})_{j c}}{d_{j}}\\,.
\\end{align}
```

A row whose weight ``s c \\mu_{j}`` falls below the machine epsilon enters the log-determinant as its
weight, so the loop stops there, and the geometric sum of the rest closes it: the history is
infinite.

# Arguments

  - `buf::NamedTuple`: Buffers (mutated): `lam`, the decay; `w`, the Bartlett weight of each lag;
    `lrow`, `lpr`, `dpr`, `upr` and `mpr`, the current row and the last rows of the factor, their
    pivots, their rows of ``L^{-1}`` and their weights; `u`, scratch; and `acc`, the column of the
    inverse on the first rows.
  - `sc::Number`: The point ``s`` times the normalisation ``c``.
  - `mix::Tuple`: ``(a, b, d)``.
  - `col::Integer`: Column of the inverse, from zero for the newest return.

# Returns

  - `lG::Number`: ``-\\frac{1}{2} \\ln \\det(I + s M)``, or ``-\\infty`` where a pivot is not
    positive: the determinant has crossed zero.

# Related

  - [`hac_damped_row!`](@ref)
  - [`hac_damped_integral!`](@ref)
  - [`hac_row_kernel`](@ref)
"""
function hac_damped_point!(buf::NamedTuple, sc::Number, mix::Tuple, col::Integer)
    (; lam, acc) = buf
    (a, b, d) = mix
    fill!(acc, zero(eltype(acc)))
    e1 = a * sc
    e2 = b * sc
    lG = zero(sc)
    j = 0
    while true
        mj = e1 + (j >= d ? e2 : zero(e2))
        if mj <= eps(typeof(sc)) && j >= length(buf.u)
            return lG - (e1 + e2) / (1 - lam) / 2
        end
        dj = hac_damped_row!(buf, mj, j, col)
        if !(dj > zero(dj))
            return -convert(typeof(lG), Inf)
        end
        lG -= log(dj) / 2
        e1 *= lam
        e2 = j >= d ? e2 * lam : e2
        j += 1
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Adds row ``j`` to the banded LDLᵀ factorisation of [`hac_damped_point!`](@ref), and its row of
``L^{-1}`` on the newest columns to the column of the inverse.

The entry of the row at lag ``i`` is ``s c \\mu_{j-i} w_{i}``, the weight of the newer return, so the
row reads the weights of the last rows from `mpr`. The row of ``L^{-1}`` is
``u_{j} = e_{j} - \\sum_{i} L_{j, j-i}\\, u_{j-i}`` on the band, and it adds
``u_{j a}\\, u_{j c} / d_{j}`` to the column. A pivot that is not positive adds nothing.

# Arguments

  - `buf::NamedTuple`: Buffers of [`hac_damped_point!`](@ref) (mutated).
  - `mj::Number`: The weight ``s c \\mu_{j}`` of the row.
  - `j::Integer`: Index of the row, from zero for the newest return.
  - `col::Integer`: Column of the inverse, from zero for the newest return.

# Returns

  - `dj::Number`: The pivot of the row.

# Related

  - [`hac_damped_point!`](@ref)
  - [`hac_damped_shift!`](@ref)
"""
function hac_damped_row!(buf::NamedTuple, mj::Number, j::Integer, col::Integer)
    (; w, lrow, lpr, dpr, upr, mpr, u, acc) = buf
    p = min(length(w), j)
    for i in p:-1:1
        num = mpr[i] * w[i] -
              sum(k -> lrow[k] * lpr[i, k - i] * dpr[k], (i + 1):p; init = zero(mj))
        lrow[i] = num / dpr[i]
    end
    dj = one(mj) + mj - sum(i -> lrow[i]^2 * dpr[i], 1:p; init = zero(mj))
    for c in eachindex(u)
        u[c] = (j == c - 1 ? one(mj) : zero(mj)) -
               sum(i -> lrow[i] * upr[i, c], 1:p; init = zero(mj))
    end
    if dj > zero(dj)
        acc .+= view(u, eachindex(acc)) .* (u[col + 1] / dj)
    end
    hac_damped_shift!(buf, dj, mj)

    return dj
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Moves the last rows of the factorisation of [`hac_damped_point!`](@ref) one place back, and keeps
the new row first: its pivot, its weight, its entries and its row of ``L^{-1}``.

# Arguments

  - `buf::NamedTuple`: Buffers of [`hac_damped_point!`](@ref) (mutated).
  - `dj::Number`: The pivot of the new row.
  - `mj::Number`: The weight of the new row.

# Returns

  - `buf::NamedTuple`: The buffers after the move.

# Related

  - [`hac_damped_row!`](@ref)
"""
function hac_damped_shift!(buf::NamedTuple, dj::Number, mj::Number)
    (; lrow, lpr, dpr, upr, mpr, u) = buf
    for r in reverse(eachindex(dpr))
        dpr[r] = r == 1 ? dj : dpr[r - 1]
        mpr[r] = r == 1 ? mj : mpr[r - 1]
        lpr[r, :] .= r == 1 ? lrow : view(lpr, r - 1, :)
        upr[r, :] .= r == 1 ? u : view(upr, r - 1, :)
    end

    return buf
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Integrates the moments of [`hac_damped_point!`](@ref) over the grid, and returns
``\\mathbb{E}[x_{a} x_{c} / Q]`` for each ``a`` on the first rows, with ``\\mathbb{E}[1 / Q]``.

With ``Q = x^{\\top} M x``, ``\\mathbb{E}[x_{a} x_{c} / Q] = \\frac{1}{2} \\int_{0}^{\\infty} [(I + s M)^{-1}]_{a c} \\det(I + s M)^{-1/2}\\, ds``. The trapezoid rule on ``\\ln s`` is spectrally
accurate on this integrand. Where ``M`` has a negative eigenvalue the transform falls to a minimum
and then rises, and the integral stops at the minimum with half weight, as
[`hac_laplace!`](@ref) does: it is the moment of the positive part.

# Arguments

  - `buf::NamedTuple`: Buffers of [`hac_damped_point!`](@ref) (mutated).
  - `sig::VecNum`: Grid of ``s (1 - \\lambda)``, uniform in its log.
  - `mix::Tuple`: ``(a, b, d)`` of [`hac_damped_point!`](@ref).
  - `col::Integer`: Column of the inverse, from zero for the newest return.

# Returns

  - `(m, e)::Tuple{<:VecNum, <:Number}`: ``\\mathbb{E}[x_{a} x_{c} / Q]`` for each ``a`` on the
    first rows, and ``\\mathbb{E}[1 / Q]``.

# Related

  - [`hac_damped_point!`](@ref)
  - [`hac_row_kernel`](@ref)
"""
function hac_damped_integral!(buf::NamedTuple, sig::VecNum, mix::Tuple, col::Integer)
    L = length(buf.w)
    lG = similar(sig)
    F = similar(sig, length(sig), L)
    for g in eachindex(sig)
        lG[g] = hac_damped_point!(buf, sig[g], mix, col)
        F[g, :] .= buf.acc
    end
    i = argmin(g -> isinf(lG[g]) ? typemax(lG[g]) : lG[g], eachindex(lG))
    wt = [if g < i || i == lastindex(lG)
              exp(lG[g]) * sig[g]
          elseif g == i
              exp(lG[g]) * sig[g] / 2
          else
              zero(lG[g])
          end
          for g in eachindex(lG)]
    h = log(sig[2] / sig[1]) / (2 * (1 - buf.lam))
    return vec(sum(F .* wt; dims = 1)) .* h, sum(wt) * h
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the HAC kernel whose raw estimate has the noise of the separate correlation path, where
each row divides by the volatility before its update.

Row ``p`` of the correlation recursion reads ``y_{p} = x_{p} / \\sigma_{p-1}`` and the lagged
terms ``\\tilde{y}_{p,l} = x_{p-l} / \\sigma_{p-1}``. The variance ``V_{p-1}`` already holds
``x_{p-l}^{2}`` and the HAC products of ``x_{p-l}``, so each lagged term is damped and correlated
with the returns near it, while its mean product with ``y_{p}`` stays zero. The kernel of
[`regime_bias_table`](@ref) at `cor_decay` assumes the raw noise, so the factor over-corrects:
at 12 assets, half-lives of 10 and 20 and two lags, the truth reads 2.9 % below it.

Regress each damped term on the returns divided by their own volatility before their own update,
``\\tilde{y}_{p,l} = \\sum_{d=1}^{2L} \\beta_{l,d}\\, y_{p-d} + e_{p,l}``. The ``y`` are
uncorrelated, each of variance ``\\mathbb{E}[1/V]``, and the residual is small. A row of the
correlation state is then a raw HAC product of the ``y`` with the kernel below, whose width is
``2L``, and the residual variance enters in quadrature at ``d = l``. Both moments come from the
law of the steady-state HAC variance, the second through
``1 / \\sqrt{ab} = \\frac{2}{\\pi} \\int_{0}^{\\pi/2} d\\theta / (a \\sin^{2}\\theta + b \\cos^{2}\\theta)``, which turns the product of two volatilities into one quadratic form.

# Mathematical definition

```math
\\begin{align}
\\beta_{l,d} &= \\frac{\\mathbb{E}\\left[x_{p-l}\\, x_{p-d} / (\\sigma_{p-1}\\, \\sigma_{p-d-1})\\right]}{\\mathbb{E}[1 / V]}\\,, \\qquad
\\Gamma_{l} = \\frac{\\mathbb{E}\\left[x_{p-l}^{2} / V_{p-1}\\right]}{\\mathbb{E}[1 / V]}\\,, \\\\
\\omega_{d} &= \\sum_{l=1}^{L} k_{l}\\, \\beta_{l,d}\\,, \\qquad
\\omega_{l} \\leftarrow \\operatorname{sign}(\\omega_{l}) \\sqrt{\\omega_{l}^{2} + k_{l}^{2} \\left(\\Gamma_{l} - \\sum_{d} \\beta_{l,d}^{2}\\right)}\\,.
\\end{align}
```

Where:

  - ``x``: iid standard Normal returns.
  - ``V_{t}``: The HAC variance at `decay` after observation ``t``, ``\\sigma_{t}^{2} = V_{t}``.
  - ``k_{l}``: Bartlett weight ``1 - l / (L + 1)`` of lag ``l``.
  - ``\\omega_{d}``: Weight of lag ``d`` of the kernel, ``d = 1, \\ldots, 2L``.

At a half-life of 10 and two lags the kernel is 0.613, 0.290, −0.022 and −0.003, against the
Bartlett 0.667 and 0.333. With it the mean of the debiased statistic reads 0.2 % to 2.2 % high at
``R = I`` over 4 to 24 assets, half-lives of 5 to 20, correlation half-lives of 20 and 40 and one
to four lags, where it read 0.8 % to 9.5 % low. The kernel is that of the steady state, and the
rows of the warm-up read it too. What remains is the part that every row rule has:
the time modulation of the rows and their coupling with ``1 / \\hat{V}``, which the factor of
[`variance_noise_bias!`](@ref) does not read.

# Algorithm

 1. Lay the grid of ``\\ln s`` on ``[-24, 8]`` with step ``1/4``, and the composite 4-point
    Gauss–Legendre rule of [`gauss_legendre_rule`](@ref) on two halves of ``[0, \\pi/2]``.
 2. With [`hac_damped_integral!`](@ref), find ``\\mathbb{E}[1/V]`` and ``\\Gamma_{l}`` from one
    variance, and each column ``d`` of ``\\beta`` from the mixed variance at each node of the rule.
 3. Sum the kernel, and add the residual in quadrature.

Against a fine grid (step ``1/20`` on ``[-40, 30]``, 32 nodes) the moments agree to ``2 \\times 10^{-7}`` at a half-life of 10, and to ``3 \\times 10^{-5}`` at a half-life of 5 and four lags, where
the estimate is negative in some draws and the cut of the positive part moves with the grid. Against
a chain of ``2 \\times 10^{7}`` rows they agree to its noise, ``3 \\times 10^{-4}``. The kernel costs
0.1 s at a half-life of 10 and two lags, and 4 s at a half-life of 250.

# Arguments

  - `decay::Number`: Decay of the variance.
  - `hac_lags::Integer`: Count of HAC lags.

# Returns

  - `kernel::VecNum`: The weight of each of the ``2L`` lags.

# Related

  - [`hac_damped_integral!`](@ref)
  - [`correlation_hac_lags!`](@ref)
  - [`regime_bias_table`](@ref)
  - [`mahalanobis_regime_bias!`](@ref)
"""
function hac_row_kernel(decay::Number, hac_lags::Integer)
    h = one(decay) / 4
    sig = exp.(range(-24 * one(h), 8 * one(h); step = h)) .* (1 - decay)
    T = eltype(sig)
    w = hac_lag_weights(hac_lags, one(T))
    L = hac_lags
    S = 2 * L
    buf = (; lam = decay, w, lrow = zeros(T, L), lpr = zeros(T, L, L), dpr = ones(T, L),
           upr = zeros(T, L, S), mpr = zeros(T, L), u = zeros(T, S), acc = zeros(T, L))
    gam = zeros(T, L)
    einv = zero(T)
    for l in 1:L
        m, einv = hac_damped_integral!(buf, sig, (one(T), zero(T), 0), l - 1)
        gam[l] = m[l]
    end
    rule = gauss_legendre_rule(one(T))
    nodes = [(pi * (2 * q - 1 + t) / 8, wt * pi / 8) for q in 1:2 for (t, wt) in rule]
    beta = zeros(T, L, S)
    for d in 1:S, (th, wt) in nodes
        m, _ = hac_damped_integral!(buf, sig, (sin(th)^2, cos(th)^2, d), d - 1)
        beta[:, d] .+= wt .* m
    end
    beta .*= 2 / (pi * einv)
    gam ./= einv
    kernel = vec(sum(w .* beta; dims = 1))
    for l in 1:L
        r = max(gam[l] - sum(abs2, view(beta, l, :)), zero(T))
        kernel[l] = sign(kernel[l]) * sqrt(kernel[l]^2 + w[l]^2 * r)
    end

    return kernel
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the HAC weights that the bias factor of the correlation reads: the kernel of
[`hac_row_kernel`](@ref) on the separate correlation path where each row divides by the volatility
before its update, else `ce.hac_lags`. The kernel is made the first time the state needs it.

# Arguments

  - `store::NamedTuple`: Store of the state, whose `kernel` fills on the first call (mutated).
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.

# Returns

  - `hac::Option{<:Union{<:Integer, <:VecNum}}`: The kernel, or `ce.hac_lags`.

# Related

  - [`hac_row_kernel`](@ref)
  - [`regime_bias_store`](@ref)
  - [`variance_noise_bias!`](@ref)
  - [`diagonal_law_correlation`](@ref)
  - [`update_var_cor!`](@ref)
"""
function correlation_hac_lags!(store::NamedTuple, ce::RegimeAdjustedExpWeightedCovariance)
    if isnothing(ce.hac_lags) || !has_separate_cor_decay(ce) || !ce.hac_vol_before
        return ce.hac_lags
    end
    if isempty(store.kernel)
        append!(store.kernel, hac_row_kernel(ce.decay, ce.hac_lags))
    end

    return store.kernel
end
