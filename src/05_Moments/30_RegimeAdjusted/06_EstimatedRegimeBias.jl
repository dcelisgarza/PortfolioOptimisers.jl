"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the exact factor of one term of an exponentially weighted variance estimate under
[`EstimatedCentring`](@ref), for an asset whose returns are all valid.

The term of the newest return is its squared deviation plus, with HAC lags, twice the weighted
products of the deviation with the lagged deviations. The location of each deviation holds the
returns before it, so the mean of the term is the variance times this factor, and the estimator
divides each term by it. The factor is [`centring_factor`](@ref) plus the part of the lagged
products of [`centring_lag_factor`](@ref), on a history with no gap.

# Mathematical definition

```math
\\begin{align}
F_{n} &= 1 + \\frac{(1 - \\lambda)(1 + \\lambda^{n})}{(1 + \\lambda)(1 - \\lambda^{n})} + \\sum_{l = 1}^{L} 2 k_{l} \\left(\\frac{\\lambda^{l} S_{2}(n - l)}{S_{1}(n) S_{1}(n - l)} - \\frac{(1 - \\lambda) \\lambda^{l - 1}}{S_{1}(n)}\\right)\\,, \\\\
S_{1}(n) &= 1 - \\lambda^{n}\\,, \\quad S_{2}(n) = \\frac{(1 - \\lambda)^{2} (1 - \\lambda^{2 n})}{1 - \\lambda^{2}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:lambda_ew])
  - ``n``: Count of the returns that the location holds.
  - ``k_{l}``: Weight of the lag ``l``. A lag with ``n - l < 1`` has no deviation and adds nothing.

# Arguments

  - `decay::Number`: Decay of the weights.
  - `n::Integer`: Count of the returns that the location holds, at least one.
  - `k::VecNum`: Weight of each lag, from [`hac_lag_weights`](@ref), empty without HAC.

# Returns

  - `F::Number`: The factor.

# Related

  - [`centring_factor`](@ref)
  - [`centring_lag_factor`](@ref)
  - [`estimated_bias_count!`](@ref)
"""
function estimated_term_factor(decay::Number, n::Integer, k::VecNum)
    F = centring_factor(EstimatedCentring(), decay, n)
    s1t = one(decay) - decay^n
    for l in eachindex(k)
        nr = n - l
        if nr >= 1
            s2r = (one(decay) - decay)^2 * (one(decay) - decay^(2 * nr)) /
                  (one(decay) - decay^2)
            F += 2 *
                 k[l] *
                 (decay^l / s1t * s2r / (one(decay) - decay^nr) -
                  (one(decay) - decay) * decay^(l - 1) / s1t)
        end
    end
    return F
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes the lattice of ``\\ln s`` and the state of [`estimated_bias_count!`](@ref) before the first
count.

The lattice is ascending, with the step ``h = |\\ln \\lambda| / m``, where ``m`` is the smallest
count of interleaved lattices that keeps ``h \\le 1/10``. So ``\\lambda s`` of each point is the
point ``m`` places below it, and a count reads no interpolation. The lattice runs from the seed
``\\ln s_{0}`` to ``x_{\\mathrm{top}}``. Below the seed the transform is ``e^{-s \\kappa_{1} / 2}``
to the first order in ``s``, and the state is the state at ``s = 0``. The default seed
``s_{0} = \\sqrt{\\epsilon (1 - \\lambda)}`` makes the error of that order ``\\epsilon``.

The state of each point is the covariance ``P`` of the location and the last ``L`` deviations
under the tilt, and the log of the absolute value of ``\\det(I + s M)`` with its sign. Before the
first count the location holds one return, so ``P_{11} = 1`` and the deviations are zero. With a
complex state type, each point takes ``s (1 + i \\varepsilon)``: the imaginary part of each value is
then ``\\varepsilon`` times its derivative in ``\\ln s``, exact to the round-off, and `dZ` holds the
derivative of the log of the determinant.

# Arguments

  - `decay::Number`: Decay of the weights.
  - `hac_lags::Option{<:Integer}`: Count of HAC lags, or `nothing`.
  - `x0::Number`: ``\\ln s_{0}``, the seed.
  - `xtop::Number`: The top of the lattice in ``\\ln s``.
  - `::Type{S}`: Element type of the state: real, or complex for the derivatives.

# Returns

  - `lat::NamedTuple`: `k`, the weight of each lag; `m`; the step `h`; `x`, the lattice of
    ``\\ln s``; `s` and `es`, ``s`` and ``e^{-s}`` on it; `sc`, the point that the state reads, and
    `ep`, ``\\varepsilon``; `tail0` and `tail1`, the sums of ``h e^{-s}`` and
    ``h \\ln s\\, e^{-s}`` from each point to the top; `P`, the covariance of each point; `P0`, the
    covariance at ``s = 0``; `lZ` and `sZ`, the log and the sign of the determinant; `dZ`, the
    derivative of `lZ` in ``\\ln s``; `pb`, scratch; and `top`, the index of the highest point that
    still moves.

# Related

  - [`estimated_bias_count!`](@ref)
  - [`estimated_bias_integrals!`](@ref)
"""
function estimated_bias_lattice(decay::Number, hac_lags::Option{<:Integer},
                                x0::Number = (log(eps(typeof(log(decay)))) + log1p(-decay)) /
                                             2, xtop::Number = 50 * one(x0),
                                ::Type{S} = typeof(log(decay))) where {S}
    del = -log(decay)
    T = typeof(del)
    k = hac_lag_weights(hac_lags, one(del))
    d = length(k) + 1
    m = max(ceil(Int, 10 * del), 1)
    h = del / m
    N = ceil(Int, (xtop - x0) / h) + 1
    x = x0 .+ h .* (0:(N - 1))
    s = exp.(x)
    es = exp.(-s)
    ep = eps(T)^2
    P = zeros(S, d, d, N)
    view(P, 1, 1, :) .= one(S)
    P0 = zeros(S, d, d)
    P0[1, 1] = one(S)
    return (; k, m, h, x, s, es, sc = s .* (one(S) + ep * (S <: Complex ? im : 0)), ep,
            tail0 = reverse(cumsum(reverse(h .* es))),
            tail1 = reverse(cumsum(reverse(h .* x .* es))), P, P0, lZ = zeros(T, N),
            sZ = ones(T, N), dZ = zeros(T, N), pb = zeros(S, d), top = Ref(N))
end
"""
    lattice_det_step(D::Real, ::Number)
    lattice_det_step(D::Complex, ep::Number)

Splits the factor of the determinant that one count adds into the log of its absolute value, its
sign, and the derivative of the log in ``\\ln s``.

A real factor carries no derivative. A complex factor at the point ``s (1 + i \\varepsilon)`` is
its real value plus ``i \\varepsilon`` times its derivative in ``\\ln s``.

# Arguments

  - `D`: The factor, from [`estimated_bias_point!`](@ref).
  - `ep::Number`: The step ``\\varepsilon`` of the complex point.

# Returns

  - `(l, sg, dl)::Tuple`: ``\\ln |D|``, the sign of ``D``, and ``D' / D``.

# Related

  - [`estimated_bias_count!`](@ref)
"""
function lattice_det_step(D::Real, ::Number)
    return log(abs(D)), sign(D), zero(D)
end
function lattice_det_step(D::Complex, ep::Number)
    r = real(D)
    return log(abs(r)), sign(r), imag(D) / (ep * r)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Advances the state of one point of the lattice by one count, and returns the factor of the
determinant that the count adds.

The state ``y = (m, d_{1}, \\ldots, d_{L})`` holds the location and the last ``L`` deviations,
with covariance ``P`` under the tilt. A new return ``x \\sim N(0, 1)`` gives the deviation
``u = x - m`` and the term ``q = (u^{2} + 2 u\\, h) / F``, with ``h = \\sum_{l} k_{l} d_{l}``. The
tilt ``e^{-s q / 2}`` is a Gaussian factor in ``w = (u, h)``, so it multiplies the determinant by
``\\det(I + s J S)`` and conditions every covariance on ``w``. The new state is
``(m + c\\, u, u, d_{1}, \\ldots, d_{L - 1})``.

# Mathematical definition

```math
\\begin{align}
S &= \\operatorname{Cov}(w)\\,, \\quad J = \\frac{1}{F} \\begin{pmatrix} 1 & 1 \\\\ 1 & 0 \\end{pmatrix}\\,, \\quad
K_{2} = s\\, (I + s J S)^{-1} J\\,, \\\\
P'_{\\alpha \\beta} &= \\operatorname{Cov}(y'_{\\alpha}, y'_{\\beta}) - g_{\\alpha}^{\\top} K_{2}\\, g_{\\beta}\\,, \\quad g_{\\alpha} = \\operatorname{Cov}(y'_{\\alpha}, w)\\,.
\\end{align}
```

Where the covariances on the right are taken before the tilt, from ``P`` and the unit variance of
``x``.

# Arguments

  - `Pn::MatNum`: Covariance of the point after the count (mutated).
  - `Pp::MatNum`: Covariance of the point ``\\lambda s`` before the count.
  - `pb::VecNum`: Scratch of length ``L + 1`` (mutated).
  - `s::Number`: Point of the lattice.
  - `k::VecNum`: Weight of each lag.
  - `F::Number`: Exact factor of the term, from [`estimated_term_factor`](@ref).
  - `c::Number`: Weight of the new return in the location.

# Returns

  - `D::Number`: ``\\det(I + s J S)``. It can be negative where the estimate is indefinite.

# Related

  - [`estimated_bias_count!`](@ref)
"""
function estimated_bias_point!(Pn::MatNum, Pp::MatNum, pb::VecNum, s::Number, k::VecNum,
                               F::Number, c::Number)
    d = size(Pp, 1)
    @inbounds begin
        for a in 1:d
            acc = zero(eltype(pb))
            for l in 1:(d - 1)
                acc += k[l] * Pp[a, 1 + l]
            end
            pb[a] = acc
        end
        P11 = Pp[1, 1]
        S11 = P11 + 1
        S12 = -pb[1]
        S22 = zero(P11)
        for l in 1:(d - 1)
            S22 += k[l] * pb[1 + l]
        end
        sF = s / F
        A11 = 1 + sF * (S11 + S12)
        A12 = sF * (S12 + S22)
        A21 = sF * S11
        A22 = 1 + sF * S12
        D = A11 * A22 - A12 * A21
        q = sF / D
        K11 = q * (A22 - A12)
        K12 = q * A22
        K22 = -q * A21
        # The location m + c u, then the newest deviation u, then the older deviations.
        u1 = c - (1 - c) * P11
        h1 = (1 - c) * pb[1]
        Pn[1, 1] = (1 - c)^2 * P11 + c^2 - (K11 * u1^2 + 2 * K12 * u1 * h1 + K22 * h1^2)
        if d > 1
            v = K11 * S11 + K12 * S12
            w = K12 * S11 + K22 * S12
            Pn[1, 2] = Pn[2, 1] = u1 - (u1 * v + h1 * w)
            Pn[2, 2] = S11 - (S11 * v + S12 * w)
        end
        for b in 3:d
            ub = -Pp[1, b - 1]
            vb = K11 * ub + K12 * pb[b - 1]
            wb = K12 * ub + K22 * pb[b - 1]
            Pn[1, b] = Pn[b, 1] = (1 - c) * Pp[1, b - 1] - (u1 * vb + h1 * wb)
            Pn[2, b] = Pn[b, 2] = ub - (S11 * vb + S12 * wb)
            for a in 3:b
                Pn[a, b] = Pn[b, a] = Pp[a - 1, b - 1] -
                                      (pb[a - 1] * wb - Pp[1, a - 1] * vb)
            end
        end
    end
    return D
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Advances every point of the lattice that still moves by one count: the term of the return after
the `n` returns that the location holds.

The point ``i`` reads the point ``i - m`` of the last count, which is ``\\lambda s``, so the points
are taken from the top down and the state is overwritten in place. A point within ``m`` of the
bottom reads the seed: the state at ``s = 0``, and ``\\ln \\det(I + \\lambda s M) = \\lambda s \\kappa_{1}``
with the exact mean ``\\kappa_{1}`` of the estimate of the last count. Each term has the mean one,
so ``\\kappa_{1} = (1 - \\lambda^{n - 1}) / (1 - \\lambda)``. Each point adds the factor of
[`lattice_det_step`](@ref) to `lZ`, `sZ` and `dZ`.

# Arguments

  - `lat::NamedTuple`: Lattice of [`estimated_bias_lattice`](@ref) (mutated).
  - `decay::Number`: Decay of the weights.
  - `n::Integer`: Count of the returns that the location holds before the term.

# Returns

  - `lat::NamedTuple`: The lattice after the count.

# Related

  - [`estimated_bias_point!`](@ref)
  - [`estimated_term_factor`](@ref)
"""
function estimated_bias_count!(lat::NamedTuple, decay::Number, n::Integer)
    (; k, m, s, sc, ep, P, P0, lZ, sZ, dZ, pb, top) = lat
    F = estimated_term_factor(decay, n, k)
    c = (one(decay) - decay) / (one(decay) - decay^(n + 1))
    k1 = (one(decay) - decay^(n - 1)) / (one(decay) - decay)
    for i in top[]:-1:1
        src = i > m ? view(P, :, :, i - m) : P0
        l, sg, dl = lattice_det_step(estimated_bias_point!(view(P, :, :, i), src, pb, sc[i],
                                                           k, F, c), ep)
        if i > m
            lZ[i], sZ[i], dZ[i] = lZ[i - m] + l, sZ[i - m] * sg, dZ[i - m] + dl
        else
            lZ[i], sZ[i], dZ[i] = decay * s[i] * k1 + l, sg, decay * s[i] * k1 + dl
        end
    end
    estimated_bias_point!(P0, copy(P0), pb, zero(eltype(s)), k, F, c)
    return lat
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the four integrals of the Laplace transform of an estimate of `K` terms that the bias
factors read, with the dependence of the next deviation, and stops the points at the top that no
longer count.

The transform is ``G(s) = |\\det(I + s M)|^{-1/2}``, and the tilted variance of the next deviation
is ``V(s) = 1 + P_{11}``, so ``\\mathbb{E}[u^{2} e^{-s R / 2}] = G V``. As in
[`hac_laplace!`](@ref), the transform is cut at its minimum below the first point whose determinant
is not positive, and the last point takes the weight one half. Below the seed ``s_{0}`` the
integrals take ``G = e^{-a s}`` with ``a = \\kappa_{1} / 2`` and ``V = f``, and the half weight of
the seed takes its Euler–Maclaurin term. A point at the top whose ``|G| \\max(s, 1)`` falls below
``\\epsilon^{3}`` adds nothing, so it stops moving, and the points above it with it.

# Mathematical definition

```math
\\begin{align}
I_{R} &= \\int_{0}^{\\infty} G(s) \\frac{V(s)}{f}\\, \\mathrm{d}s\\,, \\quad
I_{F} = \\int_{0}^{\\infty} G(s) \\sqrt{\\frac{V(s)}{f}}\\, s^{-1/2}\\, \\mathrm{d}s\\,, \\\\
I_{0} &= \\int_{0}^{\\infty} \\left(e^{-s} - G(s)\\right) s^{-1}\\, \\mathrm{d}s\\,, \\quad
I_{1} = \\int_{0}^{\\infty} \\ln s \\left(G(s) - e^{-s}\\right) s^{-1}\\, \\mathrm{d}s\\,.
\\end{align}
```

Where ``f`` is [`centring_factor`](@ref) of the next deviation, whose location holds ``K + 1``
returns.

# Arguments

  - `lat::NamedTuple`: Lattice of [`estimated_bias_lattice`](@ref) after the count `K` (mutated
    where its top falls).
  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of the terms of the estimate.
  - `dep::Bool`: Whether ``I_{R}`` and ``I_{F}`` read the dependence ``V / f``. Without it they
    read one, the law of the estimate alone.

# Returns

  - `integrals::NamedTuple`: `rms`, `fm`, `l0` and `l1`, the integrals ``I_{R}``, ``I_{F}``,
    ``I_{0}`` and ``I_{1}``.

# Related

  - [`estimated_bias_factor`](@ref)
  - [`estimated_bias_cut`](@ref)
  - [`estimated_bias_seed`](@ref)
  - [`estimated_bias_freeze!`](@ref)
  - [`hac_laplace!`](@ref)
"""
function estimated_bias_integrals!(lat::NamedTuple, decay::Number, K::Integer,
                                   dep::Bool = true)
    (; h, x, s, es, tail0, tail1, P, lZ) = lat
    f = centring_factor(EstimatedCentring(), decay, K + 1)
    icut, wl = estimated_bias_cut(lat)
    acc = estimated_bias_seed(lat, decay, K)
    for i in 1:icut
        G = exp(-lZ[i] / 2)
        r = dep ? (1 + P[1, 1, i]) / f : one(f)
        w = i == 1 ? h / 2 : (i == icut ? wl : h)
        acc = acc .+ w .* (G * r * s[i], G * sqrt(r * s[i]), es[i] - G, x[i] * (G - es[i]))
    end
    # The part of `e^{-s}` past the cut.
    if icut < length(s)
        acc = acc .+ (0, 0, tail0[icut + 1] + (h - wl) * es[icut],
                      -tail1[icut + 1] - (h - wl) * x[icut] * es[icut])
    end
    estimated_bias_freeze!(lat)
    return NamedTuple{(:rms, :fm, :l0, :l1)}(acc)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Finds the last point of the lattice that the integrals of one count read, and its weight.

The transform falls to a minimum and rises to the first point whose determinant is not positive
where the estimate is indefinite. As in [`hac_laplace!`](@ref), the integrals stop at the minimum
below that point, which takes the weight one half. Where the transform falls on every point that
still moves, the last of them takes the full weight.

# Arguments

  - `lat::NamedTuple`: Lattice of [`estimated_bias_lattice`](@ref).

# Returns

  - `(icut, w)::Tuple`: The index of the last point, and its weight in the trapezoid rule.

# Related

  - [`estimated_bias_integrals!`](@ref)
"""
function estimated_bias_cut(lat::NamedTuple)
    (; h, lZ, sZ, top) = lat
    imin, last = 1, top[]
    for i in 1:top[]
        if !(sZ[i] > 0)
            last = i - 1
            break
        end
        if lZ[i] > lZ[imin]
            imin = i
        end
    end
    return min(imin, last), imin < last || last < top[] ? h / 2 : h
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the four integrals of [`estimated_bias_integrals!`](@ref) below the seed of the lattice,
with the Euler–Maclaurin term of the half weight of the seed.

Below the seed ``s_{0}`` the transform is ``G = e^{-a s}`` to the first order, with
``a = \\kappa_{1} / 2`` and the exact mean ``\\kappa_{1}`` of the estimate, and the deviation has its
untilted variance, so ``V = f``. The seed point takes the weight one half, so each integral above
it gains ``h^{2} F'(\\ln s_{0}) / 12``, with ``F`` its integrand in ``\\ln s``.

# Arguments

  - `lat::NamedTuple`: Lattice of [`estimated_bias_lattice`](@ref).
  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of the terms of the estimate.

# Returns

  - `tails::NTuple{4, <:Number}`: The parts of ``I_{R}``, ``I_{F}``, ``I_{0}`` and ``I_{1}``.

# Related

  - [`estimated_bias_integrals!`](@ref)
"""
function estimated_bias_seed(lat::NamedTuple, decay::Number, K::Integer)
    (; h, x, s) = lat
    a = (one(decay) - decay^K) / (one(decay) - decay) / 2
    s0, x0 = s[1], x[1]
    return (s0 - a * s0^2 / 2 + h^2 / 12 * s0,
            2 * sqrt(s0) - 2 * a * s0 * sqrt(s0) / 3 + h^2 / 24 * sqrt(s0),
            (a - 1) * s0 * (1 + h^2 / 12), (1 - a) * s0 * (x0 - 1 + h^2 / 12 * (1 + x0)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Stops the points at the top of the lattice whose transform no longer counts.

A point whose ``|G| \\max(s, 1)`` is below ``\\epsilon^{3}`` adds nothing to an integral, and the
points above it only read it. So the top falls to the highest point above that bound, and no later
count moves the points past it.

# Arguments

  - `lat::NamedTuple`: Lattice of [`estimated_bias_lattice`](@ref) (mutated where its top falls).

# Returns

  - `lat::NamedTuple`: The lattice.

# Related

  - [`estimated_bias_integrals!`](@ref)
"""
function estimated_bias_freeze!(lat::NamedTuple)
    (; s, lZ, top) = lat
    while top[] > 1 && exp(-lZ[top[]] / 2) * max(s[top[]], 1) < eps(eltype(s))^3
        top[] -= 1
    end
    return lat
end
"""
    estimated_bias_factor(::RootMeanSquaredAdjusted, I::NamedTuple, c::Number)
    estimated_bias_factor(::FirstMomentRegimeAdjusted, I::NamedTuple, c::Number)
    estimated_bias_factor(::LogRegimeAdjusted, I::NamedTuple, c::Number)
    estimated_bias_factor(m::RegimeTermMoments, I::NamedTuple, c::Number)

Computes the bias factor of a regime method, or the moments of one term, from the integrals of
[`estimated_bias_integrals!`](@ref).

The factor of each method is the moment of [`regime_bias_factor`](@ref), with the dependence of the
deviation on the estimate where the method reads it. [`RootMeanSquaredAdjusted`](@ref) reads
``\\mathbb{E}[u^{2} / (f Q)]`` and [`FirstMomentRegimeAdjusted`](@ref) reads
``\\mathbb{E}[|u| / \\sqrt{f Q}]^{2} / \\mathbb{E}[|z|]^{2}``, so both read ``V``.
[`LogRegimeAdjusted`](@ref) reads ``\\mathbb{E}[\\ln u^{2}] - \\mathbb{E}[\\ln Q]``, and the law of
``u`` does not depend on the estimate, so it reads the law of ``Q`` alone. The moments of
[`RegimeTermMoments`](@ref) take the same factors.

# Mathematical definition

```math
\\begin{align}
b_{R} &= \\frac{I_{R}}{2 c}\\,, \\quad b_{F} = \\frac{I_{F}^{2}}{2 \\pi c}\\,, \\quad
b_{L} = \\exp\\left(-I_{0} - \\ln(2 c)\\right)\\,, \\\\
\\mathrm{Var}[\\ln Q] &= 2 (I_{1} - \\gamma I_{0}) - I_{0}^{2}\\,.
\\end{align}
```

Where ``c`` is the normalised weight of the newest term, and ``\\gamma`` the Euler-Mascheroni
constant.

# Arguments

  - `method`: Regime adjustment method, or the table of the moments of one term.
  - `I::NamedTuple`: Integrals of [`estimated_bias_integrals!`](@ref).
  - `c::Number`: Normalised weight of the newest term.

# Returns

  - `factor`: The factor of the method, or the triple ``(f, b, v)`` of
    [`RegimeTermMoments`](@ref).

# Related

  - [`estimated_bias_integrals!`](@ref)
  - [`regime_bias_factor`](@ref)
  - [`RegimeTermMoments`](@ref)
"""
function estimated_bias_factor(::RootMeanSquaredAdjusted, I::NamedTuple, c::Number)
    return I.rms / (2 * c)
end
function estimated_bias_factor(::FirstMomentRegimeAdjusted, I::NamedTuple, c::Number)
    return I.fm^2 / (2 * c * pi)
end
function estimated_bias_factor(::LogRegimeAdjusted, I::NamedTuple, c::Number)
    return exp(-(I.l0 + log(2 * c)))
end
function estimated_bias_factor(::RegimeTermMoments, I::NamedTuple, c::Number)
    f = estimated_bias_factor(RootMeanSquaredAdjusted(), I, c)
    return (f, one(f), zero(f))
end
function estimated_bias_factor(m::RegimeTermMoments{<:FirstMomentRegimeAdjusted},
                               I::NamedTuple, c::Number)
    f = estimated_bias_factor(RootMeanSquaredAdjusted(), I, c)
    q = estimated_bias_factor(m.method, I, c)
    return (f, q / f, f / q - one(f))
end
function estimated_bias_factor(m::RegimeTermMoments{<:LogRegimeAdjusted}, I::NamedTuple,
                               c::Number)
    f = estimated_bias_factor(RootMeanSquaredAdjusted(), I, c)
    v = 2 * (I.l1 - Base.MathConstants.eulergamma * I.l0) - I.l0^2
    return (f, estimated_bias_factor(m.method, I, c) / f, v)
end
"""
    regime_bias_table(method, decay::Number, K::Integer, hac_lags, ::PreCentred, debias)
    regime_bias_table(method, decay::Number, K::Integer, hac_lags, ::ZeroStartCentring, debias)

Computes the bias factor of a regime statistic for every count from one to `K`, on the law of a
pre-centred estimate.

A pre-centred deviation is the return itself, so the terms do not share a location, and the table
of [`regime_bias_table`](@ref) without a centring is exact: on the plain weights, or on the banded
weight matrix of the HAC lags. A location that starts at zero is not divided by its weight, and
the table takes the same law.

# Arguments

  - `method::Union{<:RegimeAdjustedMethod, <:RegimeTermMoments}`: Regime adjustment method, or the
    table of the moments of one term.
  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Largest count of terms in the table.
  - `hac_lags::Option{<:Union{<:Integer, <:VecNum}}`: Count of HAC lags, the weight of each lag, or
    `nothing`.
  - `::AbstractRegimeDebias`: Ignored debias rule: a pre-centred deviation does not depend on the
    estimate.

# Returns

  - `table::AbstractVector`: The factor for each count.

# Related

  - [`regime_bias_table`](@ref)
  - [`regime_bias!`](@ref)
"""
function regime_bias_table(method::Union{<:RegimeAdjustedMethod, <:RegimeTermMoments},
                           decay::Number, K::Integer, ::Nothing,
                           ::Union{PreCentred, ZeroStartCentring},
                           ::AbstractRegimeDebias = ExactDebias())
    return regime_bias_table(method, decay, K)
end
function regime_bias_table(method::Union{<:RegimeAdjustedMethod, <:RegimeTermMoments},
                           decay::Number, K::Integer, hac_lags::Union{<:Integer, <:VecNum},
                           ::Union{PreCentred, ZeroStartCentring},
                           ::AbstractRegimeDebias = ExactDebias())
    return regime_bias_table(method, decay, K, hac_lags)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the exact bias factor of a regime statistic for every count from one to `K`, where each
deviation is taken from an estimated location.

The location of each deviation holds the returns before it, so the terms of the estimate share
their returns, and the estimate is a quadratic form ``Q = c\\, x^{\\top} M x`` in the returns with a
dense ``M``. The law of ``Q`` is then not the law of the pre-centred table. The next deviation
``u`` reads the same location, so ``u`` and ``Q`` are not independent either. Each part moves the
factor. Without HAC the two parts nearly cancel: at a half-life of 10 the pre-centred factors are
within ``4 \\times 10^{-4}`` of this table. With HAC they add: there the pre-centred
root-mean-squared factor is 0.26 %, 0.58 % and 1.39 % too large at one, two and four lags, as a
Monte Carlo of the estimator confirms. This table is exact for both parts.

The estimate after ``K`` terms is ``R_{K} = \\lambda R_{K-1} + q_{K}`` before its normalisation, so
the Laplace transform at ``s`` after ``K`` terms reads the transform at ``\\lambda s`` after ``K - 1``
terms. On a lattice of step ``|\\ln \\lambda|`` in ``\\ln s`` that is the next point below, and a
Gaussian state carries the location and the last deviations at each point, so each count costs
one step at each point.

# Algorithm

 1. Make the lattice with [`estimated_bias_lattice`](@ref).
 2. For each count up to ``K_{h} = \\lceil \\ln \\sqrt{\\epsilon} / \\ln \\lambda \\rceil``, advance the
    lattice with [`estimated_bias_count!`](@ref), take the integrals with
    [`estimated_bias_integrals!`](@ref), and the factor with [`estimated_bias_factor`](@ref).
 3. Past ``K_{h}`` the factor changes as ``\\lambda^{K}``, and so does the pre-centred one. Each
    count takes the pre-centred factor times the ratio of the two at ``K_{h}``.

Where the estimate is positive definite, the table agrees with the eigenvalues of a dense ``M``
on a grid of step ``1/200`` to ``10^{-12}``. Where it is not, the cut moves with the lattice, as in
[`regime_bias_table`](@ref), and the table agrees with that grid to ``5 \\times 10^{-5}`` at the
first indefinite counts of a half-life of 10 and two lags, and to ``10^{-9}`` from 20 counts. The
rule of step 3 is within ``2 \\times 10^{-11}`` of the recursion at a half-life of 40 and two lags,
and within ``4 \\times 10^{-9}`` at a half-life of 5 and four lags. The table to the saturation
count costs 0.1 s at a half-life of 40 and two lags.

# Arguments

  - `method::Union{<:RegimeAdjustedMethod, <:RegimeTermMoments}`: Regime adjustment method, or the
    table of the moments of one term.
  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Largest count of terms in the table.
  - `hac_lags::Option{<:Integer}`: Count of HAC lags, or `nothing`.
  - `::EstimatedCentring`: The centring of the estimator.
  - `debias::AbstractRegimeDebias`: The debias rule. [`ExactDebias`](@ref), the default, reads the
    dependence of the deviation on the estimate, and [`LawDebias`](@ref) the law of the
    estimate alone.

# Returns

  - `table::AbstractVector`: The factor for each count, or the moments of
    [`RegimeTermMoments`](@ref).

# Related

  - [`estimated_bias_lattice`](@ref)
  - [`estimated_bias_count!`](@ref)
  - [`estimated_bias_integrals!`](@ref)
  - [`estimated_bias_factor`](@ref)
  - [`regime_bias!`](@ref)
  - [`EstimatedCentring`](@ref)
"""
function regime_bias_table(method::Union{<:RegimeAdjustedMethod, <:RegimeTermMoments},
                           decay::Number, K::Integer, hac_lags::Option{<:Integer},
                           ::EstimatedCentring,
                           debias::AbstractRegimeDebias = ExactDebias())
    base = regime_bias_table(method, decay, K, hac_lags, PreCentred())
    lat = estimated_bias_lattice(decay, hac_lags)
    Kh = min(K, ceil(Int, log(sqrt(eps(eltype(lat.s)))) / log(decay)))
    table = similar(base)
    for k in 1:Kh
        estimated_bias_count!(lat, decay, k)
        table[k] = estimated_bias_factor(method,
                                         estimated_bias_integrals!(lat, decay, k,
                                                                   reads_dependence(debias)),
                                         (one(decay) - decay) / (one(decay) - decay^k))
    end
    r = table[Kh] ./ base[Kh]
    for k in (Kh + 1):K
        table[k] = base[k] .* r
    end
    return table
end
