"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes ``\\ln \\det(I + s B)`` and its derivative in ``s`` at one point, for each count of
observations up to the length of `g`, where ``B`` is the banded matrix ``A / c`` of
[`regime_bias_table`](@ref).

The rows of the banded LDLᵀ factorisation of [`hac_ldl_taylor_row!`](@ref) carry each pivot with
its derivative, so ``\\ln \\det`` is the sum of the logs of the pivots, and its derivative the sum
of ``d_{m}' / d_{m}``. ``B`` at one count is the leading block of ``B`` at the next, so a pivot
that is not positive marks that count and every larger one. At zero lags the buffers keep one
row, which no lag reads, so the path is that of the plain weights.

# Arguments

  - `g::VecNum`: Output: ``\\ln \\det(I + s B)`` at each count, ``\\infty`` past the first pivot
    that is not positive (mutated).
  - `r::VecNum`: Output: the derivative at each count, zero where `g` is ``\\infty`` (mutated).
  - `s::Number`: Point of the lattice.
  - `decay::Number`: Decay of the weights.
  - `hac_lags::Union{Integer, VecNum}`: Count of HAC lags, or the weight of each lag.

# Returns

  - `(g, r)::Tuple{VecNum, VecNum}`: The filled vectors.

# Related

  - [`mahalanobis_lattice`](@ref)
  - [`hac_ldl_taylor_row!`](@ref)
"""
function hac_log_det_path!(g::VecNum, r::VecNum, s::Number, decay::Number,
                           hac_lags::Union{Integer, VecNum})
    z = ntuple(_ -> zero(s), 2)
    k = hac_lag_weights(hac_lags, decay)
    Lb = max(length(k), 1)
    buf = (; l = fill(z, Lb, Lb), d = fill(z, Lb), row = fill(z, Lb), k)
    lam = one(decay)
    gk = zero(s)
    rk = zero(s)
    for m in eachindex(g, r)
        dn = hac_ldl_taylor_row!(buf, s * lam, lam, decay, min(length(k), m - 1))
        if !(dn[1] > zero(dn[1]))
            g[m:end] .= convert(eltype(g), Inf)
            r[m:end] .= zero(eltype(r))
            break
        end
        gk += log(dn[1])
        rk += dn[2] / dn[1]
        g[m] = gk
        r[m] = rk
        lam *= decay
    end

    return g, r
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Finds the cut of the HAC transform of one count, the point ``\\sigma^{*}`` where
``\\ln \\det(I + \\sigma B)`` has its maximum, and the value there.

The derivative ``\\rho_{0}`` falls in ``\\sigma``, and it falls to ``-\\infty`` at the first zero of
the determinant where ``B`` has a negative eigenvalue. The lattice brackets its zero, and a
bisection on the exact derivative of [`hac_log_det_slope`](@ref) finds it. A shift of the cut by
one sixteenth in ``\\ln \\sigma`` moves the factor of the first moment by 0.3 % at two assets, a
half-life of 5 and five lags, so the step of the lattice is too coarse. The value at the cut is the
lattice value below it plus the integral of the exact derivative, by the 4-point Gauss–Legendre
rule.

# Arguments

  - `gh::VecNum`: ``\\ln \\det(I + \\sigma B)`` on the lattice, ``\\infty`` past the zero of the
    determinant.
  - `rh::VecNum`: Its derivative on the lattice.
  - `xf::AbstractRange`: Lattice of ``\\ln \\sigma``.
  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of observations.
  - `hac_lags::Union{Integer, VecNum}`: Count of HAC lags, or the weight of each lag.
  - `slope`: In place of `decay`, `K` and `hac_lags`, a function of ``\\sigma`` that gives the exact
    derivative, or a negative number past the first zero of the determinant. The first method
    reads it from [`hac_log_det_slope`](@ref) on the banded matrix.

# Returns

  - `peak::Option{<:Tuple{Number, Number}}`: ``(\\sigma^{*}, \\ln \\det(I + \\sigma^{*} B))``, or
    `nothing` where the derivative is positive on the whole lattice, because ``B`` is positive
    definite.

# Related

  - [`mahalanobis_lattice`](@ref)
  - [`mahalanobis_cut!`](@ref)
  - [`hac_log_det_slope`](@ref)
"""
function hac_log_det_peak(gh::VecNum, rh::VecNum, xf::AbstractRange, decay::Number,
                          K::Integer, hac_lags::Union{Integer, VecNum})
    c = (one(decay) - decay) / (one(decay) - decay^K)
    return hac_log_det_peak(gh, rh, xf,
                            function (sigma)
                                sl = hac_log_det_slope(decay, K, sigma / c, hac_lags)
                                return isnothing(sl) ? -one(sigma) : sl / c
                            end)
end
function hac_log_det_peak(gh::VecNum, rh::VecNum, xf::AbstractRange, slope)
    p = findfirst(i -> !(rh[i] > zero(rh[i])) || isinf(gh[i]), eachindex(gh, rh))
    if isnothing(p)
        return nothing
    end
    a = exp(xf[p - 1])
    lo, hi = a, exp(xf[p])
    for _ in 1:64
        mid = (lo + hi) / 2
        if slope(mid) > zero(mid)
            lo = mid
        else
            hi = mid
        end
    end
    rule = gauss_legendre_rule(one(a))

    return lo,
           gh[p - 1] +
           (lo - a) / 2 * sum(((t, wt),) -> wt * slope(a + (lo - a) * (1 + t) / 2), rule)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the slope ``\\ell'(t)`` of ``\\ell(t) = \\ln \\det(I + t A)`` from the banded LDLᵀ
factorisation of ``I + t A``, with the first derivative in ``t`` carried through it.

``\\ell(t) = \\sum_{m} \\ln d_{m}(t)`` over the pivots, so ``\\ell' = \\sum_{m} d_{m}' / d_{m}``.
[`hac_log_det_peak`](@ref) bisects on it to find the cut of a HAC transform. A row whose diagonal
``t\\, c\\, \\lambda^{m}`` falls below the machine epsilon of `decay` times ``1 - \\lambda`` enters
the sum as its weight, so the loop stops there and adds the remaining mass, as
[`mahalanobis_bias_sums`](@ref) does.

# Arguments

  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of observations in the estimate.
  - `t::Number`: Argument of the log-determinant.
  - `hac_lags::Union{Integer, VecNum}`: Count of HAC lags, or the weight of each lag.

# Returns

  - `slope::Option{<:Number}`: ``\\ell'(t) = \\sum_{k} \\mu_{k} / (1 + t \\mu_{k})`` over the
    eigenvalues ``\\mu`` of ``A``, or `nothing` where ``I + t A`` is not positive definite.

# Related

  - [`hac_ldl_taylor_row!`](@ref)
  - [`hac_log_det_peak`](@ref)
"""
function hac_log_det_slope(decay::Number, K::Integer, t::Number,
                           hac_lags::Union{Integer, VecNum})
    c = (one(decay) - decay) / (one(decay) - decay^K)
    z = ntuple(_ -> zero(c * t), 2)
    k = hac_lag_weights(hac_lags, decay)
    L = length(k)
    buf = (; l = fill(z, L, L), d = fill(z, L), row = fill(z, L), k)
    g1 = zero(c * t)
    lam = one(decay)
    for m in 0:(K - 1)
        if t * c * lam <= eps(decay) * (one(decay) - decay)
            return g1 + c * lam * (one(decay) - decay^(K - m)) / (one(decay) - decay)
        end
        dn = hac_ldl_taylor_row!(buf, t * c * lam, c * lam, decay, min(L, m))
        if !(dn[1] > zero(dn[1]))
            return nothing
        end
        g1 += dn[2] / dn[1]
        lam *= decay
    end

    return g1
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Adds one row to the banded LDLᵀ factorisation of ``I + t A``, with each entry carried as its value
and its first derivative in ``t``.

The entries of ``I + t A`` are linear in ``t``, so the diagonal of the row is ``(1 + t a, a)`` and
its entry at lag ``i`` is ``(t a_{i}, a_{i})``, with ``a_{i} = k_{i}\\, a / \\lambda^{i}``.
The recursion is that of [`hac_ldl_row!`](@ref), in [`taylor_mul`](@ref) and
[`taylor_div`](@ref).

# Arguments

  - `buf::NamedTuple`: Buffers of the factorisation (mutated): `l[r, i]`, the entry of row
    ``m - r`` at column ``m - r - i``; `d[r]`, the pivot of row ``m - r``; `row`, scratch; and
    `k`, the weight ``k_{i}`` of each lag, from [`hac_lag_weights`](@ref).
  - `ta::Number`: ``t\\, c\\, \\lambda^{m}``, the diagonal of ``t A`` in the row.
  - `a::Number`: ``c\\, \\lambda^{m}``, the diagonal of ``A`` in the row.
  - `decay::Number`: Decay of the weights.
  - `p::Integer`: Count of lags that reach an observation of the estimate, ``\\min(L, m)``.

# Returns

  - `pivot::NTuple{2, <:Number}`: The new pivot and its derivative.

# Related

  - [`hac_log_det_slope`](@ref)
  - [`hac_log_det_path!`](@ref)
"""
function hac_ldl_taylor_row!(buf::NamedTuple, ta::Number, a::Number, decay::Number,
                             p::Integer)
    (; l, d, row, k) = buf
    L = length(d)
    for i in p:-1:1
        ki = k[i] / decay^i
        num = (ta * ki, a * ki)
        for j in (i + 1):p
            num = num .- taylor_mul(taylor_mul(row[j], l[i, j - i]), d[j])
        end
        row[i] = taylor_div(num, d[i])
    end
    pivot = (one(ta) + ta, a)
    for i in 1:p
        pivot = pivot .- taylor_mul(taylor_mul(row[i], row[i]), d[i])
    end
    for r in L:-1:2
        d[r] = d[r - 1]
        l[r, :] .= view(l, r - 1, :)
    end
    d[1] = pivot
    l[1, :] .= row

    return pivot
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the nodes and weights of the 4-point Gauss–Legendre rule on ``[-1, 1]``, in the number
type of `o`.

# Arguments

  - `o::Number`: One, in the number type of the rule.

# Returns

  - `rule::NTuple{4, <:Tuple{Number, Number}}`: The pair of node and weight of each point.

# Related

  - [`mahalanobis_q_integral!`](@ref)
  - [`hac_log_det_peak`](@ref)
"""
function gauss_legendre_rule(o::Number)
    a = sqrt(3 * o / 7 - 2 * o / 7 * sqrt(6 * o / 5))
    b = sqrt(3 * o / 7 + 2 * o / 7 * sqrt(6 * o / 5))
    return ((-b, (18 - sqrt(30 * o)) / 36), (-a, (18 + sqrt(30 * o)) / 36),
            (a, (18 + sqrt(30 * o)) / 36), (b, (18 - sqrt(30 * o)) / 36))
end
