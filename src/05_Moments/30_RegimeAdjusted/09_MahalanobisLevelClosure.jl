"""
$(DocStringExtensions.TYPEDEF)

Computes the Mahalanobis bias factor with the level recursion of [`mahalanobis_level_bias`](@ref)
alone, which takes the mean log-determinant of each level inside the exponential. This is the
default rule of [`MahalanobisTarget`](@ref).

Near the gate the recursion reads 0.2 % to 1.3 % below a Monte Carlo of the statistic, and at the
steady state up to about 0.15 % below it, as [`GaussianLevelRecursion`](@ref) states. It costs one
recursion at each node of [`mahalanobis_bias_nodes`](@ref).

# Constructors

    MeanLevelRecursion() -> MeanLevelRecursion

# Examples

```jldoctest
julia> MahalanobisTarget().recursion
MeanLevelRecursion()
```

# Related

  - [`AbstractLevelRecursion`](@ref)
  - [`GaussianLevelRecursion`](@ref)
  - [`mahalanobis_recursion_ratio`](@ref)
"""
struct MeanLevelRecursion <: AbstractLevelRecursion end
"""
$(DocStringExtensions.TYPEDEF)

Computes the Mahalanobis bias factor with a second-order closure of the level recursion, which
carries the covariance of the log-determinant of each level as well as its mean.

The level recursion of [`mahalanobis_level_bias`](@ref) removes one direction at a time, and it
takes the deflated weights as fixed: the mean log-determinant ``D_{k}`` of each level enters the
exponential. Two errors of the second order follow, with opposite signs. The last transform
``\\mathbb{E}[e^{-F/2}]`` misses ``\\operatorname{Var}[F] / 8`` in its exponent, and the step of each
level reads the mean of a function of a random matrix at the mean of the matrix. At five assets,
a half-life of 2 and 10 observations they move the factor of the root mean square by −0.95 % and
+0.35 %. The closure takes the log-determinant ``F_{k}`` of each level as a Gaussian process with
mean ``D_{k}`` and covariance ``C_{k}``. Each step averages its integrand over that law, and the
covariance takes the linear response of the step, which holds the dependence of one level on the
levels before it, and the new variance of the removed direction, ``K_{\\mathrm{int}}``, which is the
covariance of the two log-ratios of quadratic forms in one Gaussian direction, in closed form
through ``D``, ``\\rho`` and ``\\rho'`` at the two roots of a quadratic. The last transform reads
``e^{-D/2 + C/8}``.

Against a Monte Carlo of 2 million draws of the law of the statistic, at half-lives of 2 to 40,
no lag to two lags and two to 12 assets, the factors near the gate read within about two standard
errors, where [`MeanLevelRecursion`](@ref) read 0.2 % to 1.3 % low: at 12 assets, a half-life of 5
and 16 observations, −0.06 %, −0.01 % and +0.02 % against −0.43 %, −0.26 % and −0.12 %, each with a
standard error of 0.03 % to 0.06 %. The rest is largest at two assets under HAC at the first count,
−0.33 % for the root mean square against −1.2 %, where the law is furthest from a Gaussian.

The closure costs 0.2 s to 0.7 s for each count at 5 to 12 assets, so
[`mahalanobis_recursion_ratio`](@ref) runs it at eight counts and interpolates its ratio to the
recursion. The nodes of one count of assets then cost 2 s to 5.4 s, where the recursion alone
costs 0.3 s to 0.9 s. Under [`EstimatedCentring`](@ref) the factor takes the ratio of the
pre-centred law. Against a Monte Carlo of a million draws of the statistic, that removes most of
the gap of the root mean square and the log under HAC, but the first moment, which reads 0.1 % to
0.2 % high under HAC at 8 and 12 assets, then reads 0.26 % to 0.40 % high. The two rules are under comparison on the forecasts of the
estimator, and the closure may be removed if it does not pay its cost.

# Constructors

    GaussianLevelRecursion() -> GaussianLevelRecursion

# Examples

```jldoctest
julia> MahalanobisTarget(; recursion = GaussianLevelRecursion()).recursion
GaussianLevelRecursion()
```

# Related

  - [`AbstractLevelRecursion`](@ref)
  - [`MeanLevelRecursion`](@ref)
  - [`mahalanobis_recursion_ratio`](@ref)
  - [`level_closure_factors`](@ref)
"""
struct GaussianLevelRecursion <: AbstractLevelRecursion end
"""
    mahalanobis_recursion_ratio(::MeanLevelRecursion, method, decay, Ks, n, hac_lags, start, Ksat)
    mahalanobis_recursion_ratio(::GaussianLevelRecursion, method, decay, Ks, n, hac_lags, start, Ksat)

The ratio of the Mahalanobis bias factor of a level recursion to the factor of
[`MeanLevelRecursion`](@ref), at each count of `Ks`.

The ratio of [`GaussianLevelRecursion`](@ref) is smooth in ``\\lambda^{K}``, and it tends to a
constant past the saturation: at five assets, a half-life of 10 and two lags it falls from
1.0066 at the gate to 1.0008, and at 12 assets and a half-life of 5 from 1.0037 to 1.0015. Eight
Chebyshev–Lobatto points in ``\\lambda^{K - K_{0}}`` from the gate ``K_{0}`` to the saturation hold
it, and a barycentric interpolant reads it at each count.

# Arguments

  - `method::RegimeAdjustedMethod`: Regime adjustment method, which names the moment.
  - `decay::Number`: Decay of the weights.
  - `Ks::AbstractVector{<:Integer}`: Counts of observations.
  - `n::Integer`: Count of assets that contribute to the statistic, at least two.
  - `hac_lags::Option{<:Union{Integer, VecNum}}`: Count of HAC lags, the weight of each lag, or
    `nothing`.
  - `start::Integer`: The first count, [`mahalanobis_bias_start`](@ref).
  - `Ksat::Integer`: The count of saturation, [`mahalanobis_bias_saturation`](@ref).

# Returns

  - `ratio::Option{<:VecNum}`: `nothing` for [`MeanLevelRecursion`](@ref), so the factor is not
    touched, and the ratio at each count of `Ks` for [`GaussianLevelRecursion`](@ref).

# Related

  - [`AbstractLevelRecursion`](@ref)
  - [`mahalanobis_bias_nodes`](@ref)
  - [`level_closure_factors`](@ref)
"""
function mahalanobis_recursion_ratio(::MeanLevelRecursion, ::RegimeAdjustedMethod, ::Number,
                                     ::AbstractVector{<:Integer}, ::Integer, ::Any,
                                     ::Integer, ::Integer)
    return nothing
end
function mahalanobis_recursion_ratio(::GaussianLevelRecursion, method::RegimeAdjustedMethod,
                                     decay::Number, Ks::AbstractVector{<:Integer},
                                     n::Integer, hac_lags::Option{<:Union{Integer, VecNum}},
                                     start::Integer, Ksat::Integer)
    o = one(decay)
    t = (o .- cos.(pi .* (0:7) ./ 7 .* o)) ./ 2
    Kr = unique([if t[i] <= decay^(Ksat - start)
                     Ksat
                 else
                     clamp(round(Int, start + log(t[i]) / log(decay)), start, Ksat)
                 end
                 for i in eachindex(t)])
    closure = [level_closure_factors((method,), decay, K, n, hac_lags)[1] for K in Kr]
    lr = log.(closure ./ mahalanobis_level_bias(method, decay, Kr, n, hac_lags))
    sigma = decay .^ (Kr .- start)
    bw = [inv(prod(sigma[a] - sigma[b] for b in eachindex(sigma) if b != a; init = o))
          for a in eachindex(sigma)]
    return map(Ks) do K
        x = decay^(min(K, Ksat) - start)
        a = findfirst(==(x), sigma)
        if !(isnothing(a))
            return exp(lr[a])
        end
        return exp(sum(bw[k] * lr[k] / (x - sigma[k]) for k in eachindex(bw)) /
                   sum(bw[k] / (x - sigma[k]) for k in eachindex(bw)))
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes the linear maps of the closure on the grid of [`mahalanobis_bias_grid`](@ref).

The state of the closure is the removed part ``q`` on the nodes, the eight Chebyshev nodes and the
grid past the junction. The vector `v` of [`mahalanobis_level_step!`](@ref) holds the Chebyshev
values, the five grid values at the junction, which the Chebyshev values fix, and the grid past
it. The vector `e` of [`mahalanobis_q_integral!`](@ref) is linear in `v`. The closure reads both
through one vector ``y = (v; e)``.

# Arguments

  - `grid::NamedTuple`: Grids of [`mahalanobis_bias_grid`](@ref).

# Returns

  - `maps::NamedTuple`: `By`, the map from the nodes to ``y``; and `NV`, the length of `v`.

# Related

  - [`level_closure_factors`](@ref)
"""
function level_closure_maps(grid::NamedTuple)
    NN = length(grid.nodes)
    NV = 8 + grid.NG
    T = eltype(grid.nodes)
    P = zeros(T, NV, NN)
    for l in 1:8
        P[l, l] = one(T)
        for g in 1:5
            P[8 + g, l] = grid.junction[g][l]
        end
    end
    for a in 9:NN
        P[a + 5, a] = one(T)
    end
    J = zeros(T, 10 + grid.NG, NV)
    v = zeros(T, NV)
    e = zeros(T, 10 + grid.NG)
    for k in 1:NV
        fill!(v, zero(T))
        v[k] = one(T)
        mahalanobis_q_integral!(e, v, grid)
        J[:, k] .= e
    end
    return (; By = vcat(P, J * P), NV)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Evaluates the mean field of one level at a point: the log-determinant ``D = g - Q`` and its
derivative ``\\rho = \\rho_{0} - q``.

# Arguments

  - `ctx::NamedTuple`: The context of [`level_closure_factors`](@ref).
  - `y::VecNum`: The mean of ``y = (v; e)``.
  - `z::Number`: Point to evaluate at.

# Returns

  - `(D, rho)::Tuple{Number, Number}`: The two values.

# Related

  - [`level_closure_table`](@ref)
"""
function level_closure_point(ctx::NamedTuple, y::VecNum, z::Number)
    (; grid, gh, rh, m, ct, NV) = ctx
    xf = grid.xf
    st = lagrange_stencil(first(xf), step(xf), length(xf), log(z))
    qs = mahalanobis_value_stencil(z, grid.geo)
    Qs = mahalanobis_integral_stencil(z, grid.geo)
    q = stencil_dot(qs, view(y, 1:NV))
    Q = stencil_dot(Qs, view(y, (NV + 1):length(y)))
    return stencil_dot(st, gh, m) - Q, ct * stencil_dot(st, rh, m) - q
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the mean field of one level of the closure on a uniform grid of ``\\ln z``, as
[`level_closure_table`](@ref) fills it.

# Fields

  - `x0`: First point of the grid of ``\\ln z``.
  - `h`: Step of the grid.
  - `D`: ``D`` at each point.
  - `R`: ``\\rho`` at each point.
  - `Rx`: ``z \\rho'`` at each point.

# Related

  - [`level_closure_table`](@ref)
  - [`level_closure_read`](@ref)
"""
@concrete struct LevelClosureTable
    x0
    h
    D
    R
    Rx
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Tabulates the mean field of one level, ``D``, ``\\rho`` and ``z \\rho'``, on a uniform grid of
``\\ln z`` of step ``1/16`` from ``-40`` to ``40``, for [`level_closure_kernel`](@ref), which reads
them at about a million points. The derivative is a central difference of ``\\rho`` in ``\\ln z`` of
step ``10^{-4}``, on the interpolants of the grid, which are polynomials.

# Arguments

  - `ctx::NamedTuple`: The context of [`level_closure_factors`](@ref).
  - `y::VecNum`: The mean of ``y = (v; e)``.

# Returns

  - `table::LevelClosureTable`: The grid and the three columns.

# Related

  - [`level_closure_read`](@ref)
"""
function level_closure_table(ctx::NamedTuple, y::VecNum)
    T = eltype(y)
    h = one(T) / 16
    xs = range(-40 * one(T), 40 * one(T); step = h)
    d = one(T) / 10^4
    D = zeros(T, length(xs))
    R = zeros(T, length(xs))
    Rx = zeros(T, length(xs))
    for (i, x) in enumerate(xs)
        D[i], R[i] = level_closure_point(ctx, y, exp(x))
        Rx[i] = (level_closure_point(ctx, y, exp(x + d))[2] -
                 level_closure_point(ctx, y, exp(x - d))[2]) / (2 * sinh(d))
    end
    return LevelClosureTable(first(xs), h, D, R, Rx)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the table of [`level_closure_table`](@ref) at one point with the 4-point Lagrange stencil.

# Arguments

  - `tab::LevelClosureTable`: The table.
  - `z::Number`: Point to read at.

# Returns

  - `(D, rho, drho)::NTuple{3, <:Number}`: ``D``, ``\\rho`` and ``\\rho'`` at `z`.

# Related

  - [`level_closure_kernel`](@ref)
"""
function level_closure_read(tab::LevelClosureTable, z::Number)
    p = (log(z) - tab.x0) / tab.h + 1
    i = clamp(floor(Int, p) - 1, 1, length(tab.D) - 3)
    u = p - i
    w = (-(u - 1) * (u - 2) * (u - 3) / 6, u * (u - 2) * (u - 3) / 2,
         -u * (u - 1) * (u - 3) / 2, u * (u - 1) * (u - 2) / 6)
    D = sum(k -> w[k] * tab.D[i + k - 1], 1:4)
    R = sum(k -> w[k] * tab.R[i + k - 1], 1:4)
    Rx = sum(k -> w[k] * tab.Rx[i + k - 1], 1:4)
    return D, R, Rx / z
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the integrand of [`level_closure_kernel`](@ref) at one point ``(a, b)``.

``V(s) = \\ln(x^{\\top} M_{s} x) - \\ln(x^{\\top} M x)``, with ``M_{s} = M (I + s M)^{-1}``, is the
log-ratio that one removed direction ``x`` adds to the log-determinant, and the kernel is the
covariance of ``V'(s)`` and ``V'(t)``. The joint transform of the two quadratic forms of
``\\ln(x^{\\top} M_{s} x)`` and ``\\ln(x^{\\top} M_{t} x)`` at ``(a, b)`` is ``e^{-E/2}``, with
``E = D(p) + D(q) - D(s) - D(t)`` and ``p``, ``q`` the roots of
``(1 + s\\mu)(1 + t\\mu) + 2 a \\mu (1 + t\\mu) + 2 b \\mu (1 + s\\mu)``, so the mixed derivative in
``s`` and ``t`` reads ``D``, ``\\rho`` and ``\\rho'`` at the roots. Where the roots are close next to
their size, the term ``p_{st} (\\rho(p) - \\rho(q))``, whose factor is ``1 / (q - p)``, takes the
divided difference of ``\\rho``, so the round-off of the two values is not amplified.

# Arguments

  - `tab::LevelClosureTable`: The table of [`level_closure_table`](@ref).
  - `s::Number`, `t::Number`: The two points of the kernel.
  - `a::Number`, `b::Number`: The point of the quadrature.
  - `fs::Tuple`, `ft::Tuple`: ``(D, \\rho)`` at `s` and at `t`.

# Returns

  - `val::Number`: ``e^{-E/2} (E_{s} E_{t} / 4 - E_{st} / 2)``.

# Related

  - [`level_closure_kernel`](@ref)
"""
function level_closure_integrand(tab::LevelClosureTable, s::Number, t::Number, a::Number,
                                 b::Number, fs::Tuple, ft::Tuple)
    disc = sqrt((s - t + 2 * a - 2 * b)^2 + 16 * a * b)
    p = (s + t + 2 * a + 2 * b + disc) / 2
    q = (s * t + 2 * a * t + 2 * b * s) / p
    ps = (s - t + 2 * a - 2 * b + disc) / (2 * disc)
    pt = (t - s + 2 * b - 2 * a + disc) / (2 * disc)
    Dp, Rp, dRp = level_closure_read(tab, p)
    Dq, Rq, dRq = level_closure_read(tab, q)
    dd = disc > p / 10^6 ? (Rp - Rq) / disc : level_closure_read(tab, (p + q) / 2)[3]
    Es = Rp * ps + Rq * (1 - ps) - fs[2]
    Et = Rp * pt + Rq * (1 - pt) - ft[2]
    Est = dRp * ps * pt + dRq * (1 - ps) * (1 - pt) -
          (1 - (1 - pt) * ps - pt * (1 - ps)) * dd
    return exp(-(Dp + Dq - fs[1] - ft[1]) / 2) * (Es * Et / 4 - Est / 2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the covariance that one removed direction adds to the removed part of the next level,
``(1 + s)(1 + t) K_{\\mathrm{int}}(s, t)``, on a grid of ``\\ln s`` of step one.

``K_{\\mathrm{int}}(s, t) = \\partial_{s} \\partial_{t} \\operatorname{Cov}(\\ln A_{s}, \\ln A_{t})`` with
``A_{s} = x^{\\top} M_{s} x``, for the mean field of the level, and
``\\operatorname{Cov}(\\ln A_{s}, \\ln A_{t}) = \\iint \\operatorname{Cov}(e^{-a A_{s}}, e^{-b A_{t}})\\, \\mathrm{d}\\ln a\\, \\mathrm{d}\\ln b``.
The trapezoid rule of step one in ``\\ln a`` and ``\\ln b`` on ``[-16, 26]`` holds it. The scaled
kernel is bounded: it tends to ``\\operatorname{Var}[q(0)]`` at zero and to zero past the saturation,
where ``s q(s)`` is the count of the removed directions, so its interpolant has no error that
``s t`` multiplies. The grid stops three steps past the point where the saturated count
``s \\rho(s)`` reaches its limit to ``10^{-4}``, and the kernel is constant past it.

# Arguments

  - `tab::LevelClosureTable`: The table of [`level_closure_table`](@ref).
  - `T::Type`: The number type.
  - `tol::Number`: The relative floor of [`level_closure_span`](@ref), ``10^{-7}`` by default.
    Zero reads every point.

# Returns

  - `(xc, K)::Tuple`: The grid of ``\\ln s`` and the scaled kernel on it.

# Related

  - [`level_closure_step`](@ref)
"""
function level_closure_kernel(tab::LevelClosureTable, ::Type{T},
                              tol::Number = one(T) / 10^7) where {T}
    A = exp.(range(-16 * one(T), 26 * one(T); step = one(T)))
    lim = tab.R[end] * exp(tab.x0 + tab.h * (length(tab.R) - 1))
    xs = tab.x0 .+ tab.h .* (0:(length(tab.R) - 1))
    isat = findfirst(i -> tab.R[i] * exp(xs[i]) >= lim - lim / 10^4 && xs[i] > -6,
                     eachindex(xs))
    xhi = clamp(ceil(isnothing(isat) ? 30 * one(T) : xs[isat]) + 3, one(T), 30 * one(T))
    xc = range(-6 * one(T), xhi; step = one(T))
    sc = exp.(xc)
    # Each pair is built by hand: a slice of a tuple has a length that the analysis cannot bound.
    f = map(sc) do s
        D, R, _ = level_closure_read(tab, s)
        return (D, R)
    end
    dm = zeros(T, length(sc), length(A))
    for i in eachindex(sc), ia in eachindex(A)
        Da, Ra, _ = level_closure_read(tab, sc[i] + 2 * A[ia])
        dm[i, ia] = -(Ra - f[i][2]) / 2 * exp(-(Da - f[i][1]) / 2)
    end
    span = [level_closure_span(view(dm, i, :), tol) for i in eachindex(sc)]
    K = zeros(T, length(sc), length(sc))
    for i in eachindex(sc), j in i:length(sc)
        acc = zero(T)
        for ia in span[i], ib in span[j]
            acc += level_closure_integrand(tab, sc[i], sc[j], A[ia], A[ib], f[i], f[j]) -
                   dm[i, ia] * dm[j, ib]
        end
        K[i, j] = K[j, i] = acc * (1 + sc[i]) * (1 + sc[j])
    end
    return xc, K
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the points of the quadrature in ``\\ln a`` where one node of the kernel holds weight: the
span where ``|\\partial_{s} m_{1}(s, a)|`` is at least `tol` times its largest value. The
covariance of the two exponentials falls with each of the two factors, so the points outside the
span add nothing that the rule resolves.

# Arguments

  - `dm::VecNum`: ``\\partial_{s} m_{1}(s, a)`` at each point.
  - `tol::Number`: The relative floor.

# Returns

  - `span::UnitRange{Int}`: The points.

# Related

  - [`level_closure_kernel`](@ref)
"""
function level_closure_span(dm::VecNum, tol::Number)
    top = maximum(abs, dm)
    keep = findall(x -> abs(x) >= tol * top, dm)
    return isempty(keep) ? (1:0) : (first(keep):last(keep))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the quadratic form ``w^{\\top} Y w`` of one 8-point stencil of ``y``, the variance of the
value that the stencil reads.

# Arguments

  - `Y::MatNum`: Covariance of ``y``.
  - `i0::Integer`: Index of the first point of the stencil in ``y``.
  - `w::NTuple{8, <:Number}`: Its weights.

# Returns

  - `val::Number`: The form.

# Related

  - [`level_closure_transform`](@ref)
"""
function level_closure_form(Y::MatNum, i0::Integer, w::NTuple{8, <:Number})
    acc = zero(eltype(Y))
    for k in 1:8, l in 1:8
        acc += w[k] * w[l] * Y[i0 + k - 1, i0 + l - 1]
    end
    return acc
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the stencils of the removed part ``q`` and of its integral ``Q`` at one node, with their
products with the covariance of ``y``, which every step of the node reads.

A Chebyshev node reads its own value of `v`, and its integral through the Chebyshev weights; a
grid node reads `v` and `e` at its own index.

# Arguments

  - `grid::NamedTuple`: Grids of [`mahalanobis_bias_grid`](@ref).
  - `NV::Integer`: The length of `v`.
  - `a::Integer`: The node.
  - `Y::MatNum`: The covariance of ``y = (v; e)``.

# Returns

  - `node::NamedTuple`: The indices `qi`, `Qi` and weights `qw`, `Qw` of the two stencils,
    `Yq = Y q` and `YQ = Y Q`, ``\\operatorname{Var}[Q]`` as `vQ` and ``\\operatorname{Cov}(Q, q)`` as
    `cQq`.

# Related

  - [`level_closure_step`](@ref)
  - [`level_closure_pair`](@ref)
"""
function level_closure_node(grid::NamedTuple, NV::Integer, a::Integer, Y::MatNum)
    T = eltype(Y)
    qi, qw, Qi, Qw = if a <= 8
        [a], [one(T)], collect(1:8), collect(T, grid.Qcheb[a])
    else
        [a + 5], [one(T)], [NV + a + 5], [one(T)]
    end
    Yq = view(Y, :, qi) * qw
    YQ = view(Y, :, Qi) * Qw
    return (; qi, qw, Qi, Qw, Yq, YQ, vQ = LinearAlgebra.dot(Qw, view(YQ, Qi)),
            cQq = LinearAlgebra.dot(Qw, view(Yq, Qi)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads one step of one node of the closure: the values of the two stencils of the point
``s + u`` on the mean, and the variance of ``\\Delta Q`` and its covariance with ``\\Delta q``, the
differences between ``s + u`` and the node.

# Arguments

  - `Y::MatNum`: The covariance of ``y = (v; e)``.
  - `y::VecNum`: The mean of ``y``.
  - `node::NamedTuple`: The node, from [`level_closure_node`](@ref).
  - `qs::Tuple`, `Qs::Tuple`: The stencils of the point on `v` and on `e`, from the pairs of
    [`mahalanobis_bias_grid`](@ref).
  - `NV::Integer`: The length of `v`.

# Returns

  - `(yq, yQ, var, cov)::NTuple{4, <:Number}`: ``q`` and ``Q`` at the point,
    ``\\operatorname{Var}[\\Delta Q]`` and ``\\operatorname{Cov}(\\Delta Q, \\Delta q)``.

# Related

  - [`level_closure_step`](@ref)
"""
function level_closure_pair(Y::MatNum, y::VecNum, node::NamedTuple, qs::Tuple, Qs::Tuple,
                            NV::Integer)
    iq, wq = qs
    iQ, wQ = NV + Qs[1], Qs[2]
    T = eltype(Y)
    yq = yQ = QQ = Qq = QsQa = Qsqa = qsQa = zero(T)
    for k in 1:8
        yq += wq[k] * y[iq + k - 1]
        yQ += wQ[k] * y[iQ + k - 1]
        QsQa += wQ[k] * node.YQ[iQ + k - 1]
        Qsqa += wQ[k] * node.Yq[iQ + k - 1]
        qsQa += wq[k] * node.YQ[iq + k - 1]
        for l in 1:8
            QQ += wQ[k] * wQ[l] * Y[iQ + k - 1, iQ + l - 1]
            Qq += wQ[k] * wq[l] * Y[iQ + k - 1, iq + l - 1]
        end
    end
    return yq, yQ, QQ - 2 * QsQa + node.vQ, Qq - Qsqa - qsQa + node.cQq
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the scaled kernel of [`level_closure_kernel`](@ref) at every pair of nodes, and removes its
scale: ``K_{\\mathrm{int}}(s, t)`` is the kernel read with a clamped 8-point Lagrange stencil in
``\\ln s`` and in ``\\ln t``, over ``(1 + s)(1 + t)``. The clamp holds the kernel constant past the
ends of its grid.

# Arguments

  - `xc::AbstractRange`: The grid of ``\\ln s`` of the kernel.
  - `Kc::MatNum`: The scaled kernel on it.
  - `sn::AbstractVector{T}`: The nodes.

# Returns

  - `Kint::Matrix{T}`: The kernel at each pair of nodes.

# Related

  - [`level_closure_step`](@ref)
"""
function level_closure_spread(xc::AbstractRange, Kc::MatNum,
                              sn::AbstractVector{T}) where {T}
    N = length(sn)
    i0 = zeros(Int, N)
    W = zeros(T, 8, N)
    for i in 1:N
        j0, w = lagrange_stencil(first(xc), step(xc), length(xc),
                                 clamp(log(sn[i]), first(xc), last(xc)))
        i0[i] = j0
        for k in 1:8
            W[k, i] = w[k]
        end
    end
    Kint = zeros(T, N, N)
    for i in 1:N, j in 1:N
        acc = zero(T)
        for k in 1:8, l in 1:8
            acc += W[k, i] * W[l, j] * Kc[i0[i] + k - 1, i0[j] + l - 1]
        end
        Kint[i, j] = acc / ((1 + sn[i]) * (1 + sn[j]))
    end
    return Kint
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Runs one level of the closure: the mean of the removed part on the nodes and its covariance.

# Algorithm

 1. Map the mean and the covariance of the nodes to ``y = (v; e)``.
 2. At each node ``s`` and step ``u``, average the integrand of [`mahalanobis_level_step!`](@ref)
    over the Gaussian law of the level: with ``\\Delta Q`` and ``\\Delta q`` the differences
    between ``s + u`` and ``s``, the integrand is
    ``e^{-\\Delta D / 2 + \\operatorname{Var}[\\Delta Q] / 8} (\\Delta \\rho - \\operatorname{Cov}(\\Delta Q, \\Delta q) / 2)``.
 3. Accumulate the linear response of the step: a perturbation of the level moves the new removed
    part by ``\\frac{h}{2} \\sum_{u} \\psi (\\Delta \\delta q - \\frac{1}{2} \\Delta \\rho\\, \\Delta \\delta Q)``.
    Every term of it vanishes as ``u \\to 0``.
 4. The new covariance is the response of the old one plus the kernel of
    [`level_closure_kernel`](@ref) at the nodes, read with a clamped 8-point Lagrange stencil.

# Arguments

  - `ctx::NamedTuple`: The context of [`level_closure_factors`](@ref).
  - `nq::VecNum`: The mean of the removed part on the nodes.
  - `Kq::MatNum`: Its covariance.

# Returns

  - `(nq, Kq)::Tuple`: The mean and the covariance of the next level.

# Related

  - [`level_closure_factors`](@ref)
  - [`level_closure_node`](@ref)
  - [`level_closure_pair`](@ref)
  - [`mahalanobis_level_step!`](@ref)
"""
function level_closure_step(ctx::NamedTuple, nq::VecNum, Kq::MatNum)
    (; grid, By, NV, dg, dr) = ctx
    y = By * nq
    Y = By * Kq * transpose(By)
    Ay = zeros(eltype(Kq), length(nq), length(y))
    new = similar(nq)
    for a in eachindex(nq)
        node = level_closure_node(grid, NV, a, Y)
        qn = LinearAlgebra.dot(node.qw, view(y, node.qi))
        Qn = LinearAlgebra.dot(node.Qw, view(y, node.Qi))
        acc = fsum = gsum = zero(eltype(Kq))
        for b in axes(dg, 2)
            qs, Qs = grid.pairs[a, b]
            yq, yQ, var, cov = level_closure_pair(Y, y, node, qs, Qs, NV)
            dR = dr[a, b] - (yq - qn)
            psi = exp(-(dg[a, b] - (yQ - Qn)) / 2 + var / 8)
            acc += psi * (dR - cov / 2)
            f = grid.h * psi / 2
            for k in 1:8
                Ay[a, qs[1] + k - 1] += f * qs[2][k]
                Ay[a, NV + Qs[1] + k - 1] -= f * dR / 2 * Qs[2][k]
            end
            fsum += f
            gsum += f * dR / 2
        end
        view(Ay, a, node.qi) .-= fsum .* node.qw
        view(Ay, a, node.Qi) .+= gsum .* node.Qw
        new[a] = qn - grid.h / 2 * acc
    end
    xc, Kc = level_closure_kernel(level_closure_table(ctx, y), eltype(Kq))
    Kint = level_closure_spread(xc, Kc, convert(Vector{eltype(Kq)}, grid.nodes))
    A = LinearAlgebra.I + Ay * By
    Kn = A * Kq * transpose(A) + Kint
    return new, (Kn + transpose(Kn)) / 2
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Evaluates the last transform of the closure, ``e^{-D/2 + \\operatorname{Var}[Q]/8}``, on the output
grid, with the cut of [`mahalanobis_transform`](@ref) where the count has one.

# Arguments

  - `ctx::NamedTuple`: The context of [`level_closure_factors`](@ref).
  - `y::VecNum`: The mean of ``y = (v; e)``.
  - `Y::MatNum`: Its covariance.

# Returns

  - `(so, L)::Tuple{VecNum, VecNum}`: The grid and the transform on it.

# Related

  - [`level_closure_factors`](@ref)
"""
function level_closure_transform(ctx::NamedTuple, y::VecNum, Y::MatNum)
    (; grid, gh, m, cut, NV) = ctx
    so = grid.so
    k = length(so) - 1
    gt = gh
    if !isnothing(cut)
        ls = log(cut.sstar)
        k = clamp(round(Int, (ls - log(first(so))) / grid.ho), 0, length(so) - 1)
        so = so .* exp(ls - log(so[k + 1]))
        gt = cut.gt
    end
    xf = grid.xf
    L = zeros(eltype(y), length(so))
    for a in 1:(k + 1)
        st = lagrange_stencil(first(xf), step(xf), length(xf), log(so[a]))
        Qs = mahalanobis_integral_stencil(so[a], grid.geo)
        L[a] = exp(-(stencil_dot(st, gt, m) - stencil_dot(Qs, view(y, (NV + 1):length(y)))) /
                   2 + level_closure_form(Y, NV + Qs[1], Qs[2]) / 8)
    end
    if !isnothing(cut)
        L[k + 1] /= 2
    end
    return so, L
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the Mahalanobis bias factor of each method at one count of observations with the closure
of [`GaussianLevelRecursion`](@ref).

# Algorithm

 1. Tabulate the lattice of the count with [`mahalanobis_lattice`](@ref), round the normalisation
    to it, continue it past the cut with [`mahalanobis_cut!`](@ref), and fill the differences of
    [`mahalanobis_pair_differences!`](@ref), as [`mahalanobis_level_bias`](@ref) does.
 2. Run ``n - 1`` levels of [`level_closure_step`](@ref) from a removed part of zero with no
    variance.
 3. Read each factor off [`level_closure_transform`](@ref) with [`regime_bias_factor`](@ref).

# Arguments

  - `methods::Tuple`: Regime adjustment methods.
  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of observations.
  - `n::Integer`: Count of assets that contribute to the statistic, at least two.
  - `hac_lags::Option{<:Union{Integer, VecNum}}`: Count of HAC lags, the weight of each lag, or
    `nothing`.

# Returns

  - `factors::VecNum`: The factor of each method.

# Related

  - [`GaussianLevelRecursion`](@ref)
  - [`mahalanobis_recursion_ratio`](@ref)
"""
function level_closure_factors(methods::Tuple, decay::Number, K::Integer, n::Integer,
                               hac_lags::Option{<:Union{Integer, VecNum}})
    grid = mahalanobis_bias_grid(decay)
    G, Rh, peaks = mahalanobis_lattice(decay, [K], grid.xf, hac_lags)
    c = (one(decay) - decay) / (one(decay) - decay^K)
    m = round(Int, log(c) / step(grid.xf))
    ct = exp(m * step(grid.xf))
    gh = G[:, 1]
    rh = Rh[:, 1]
    cut = mahalanobis_cut!(gh, rh, grid.xf, ct, peaks[1])
    NN = length(grid.nodes)
    dg = zeros(eltype(G), NN, length(grid.su))
    dr = similar(dg)
    mahalanobis_pair_differences!(dg, dr, grid, gh, rh, (m, ct))
    ctx = merge((; grid, gh, rh, m, ct, cut, dg, dr), level_closure_maps(grid))
    nq = zeros(eltype(G), NN)
    Kq = zeros(eltype(G), NN, NN)
    for _ in 1:(n - 1)
        nq, Kq = level_closure_step(ctx, nq, Kq)
    end
    so, L = level_closure_transform(ctx, ctx.By * nq, ctx.By * Kq * transpose(ctx.By))
    return [regime_bias_factor(method, so, L, one(c), grid.ho) * ct / c
            for method in methods]
end
export MeanLevelRecursion, GaussianLevelRecursion
public mahalanobis_recursion_ratio
