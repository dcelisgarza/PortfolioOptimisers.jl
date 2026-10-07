"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes ``\\ln \\det(I + \\sigma M_{K})`` of an estimate of `K` terms under
[`EstimatedCentring`](@ref), its derivative in ``\\sigma``, and the tilted variance of the location,
at one point ``\\sigma``.

The chain of [`estimated_bias_count!`](@ref) for the point ``\\sigma`` after `K` terms reads the
points ``\\sigma \\lambda^{K - j}`` at the count ``j``. This function runs that chain alone, at the
complex point ``\\sigma (1 + i \\varepsilon)``, so each derivative is exact to the round-off. A count
whose point is below the seed ``\\ln s_{0} = \\ln \\epsilon + \\ln(1 - \\lambda)`` reads the state at
``s = 0``, as the lattice does.

# Arguments

  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of the terms of the estimate.
  - `sigma::Number`: The point ``\\sigma``.
  - `k::VecNum`: Weight of each lag, from [`hac_lag_weights`](@ref).

# Returns

  - `(alive, rho, g, vg, dvg)::Tuple`: Whether the determinant is positive along the chain, the
    derivative ``\\rho_{0} = \\mathrm{d} g / \\mathrm{d} \\sigma``, ``g``, the tilted variance
    ``P_{11}`` of the location, and its derivative in ``\\ln \\sigma``.

# Related

  - [`estimated_bias_count!`](@ref)
  - [`hac_log_det_peak`](@ref)
  - [`estimated_mahalanobis_columns!`](@ref)
"""
function estimated_log_det_chain(decay::Number, K::Integer, sigma::Number, k::VecNum)
    T = typeof(log(decay))
    d = length(k) + 1
    ep = eps(T)^2
    x0 = log(eps(T)) + log1p(-decay)
    P0 = zeros(T, d, d)
    P0[1, 1] = one(T)
    P = zeros(Complex{T}, d, d)
    Pt = similar(P)
    P0t = similar(P0)
    pb = zeros(Complex{T}, d)
    pb0 = zeros(T, d)
    started = false
    lz, sz, dz = zero(T), one(T), zero(T)
    for j in 1:K
        F = estimated_term_factor(decay, j, k)
        c = (one(decay) - decay) / (one(decay) - decay^(j + 1))
        sj = sigma * decay^(K - j)
        if !started && log(sj) >= x0
            lz = dz = decay * sj * (one(decay) - decay^(j - 1)) / (one(decay) - decay)
            P .= P0
            started = true
        end
        if started
            Pt .= P
            l, sg, dl = lattice_det_step(estimated_bias_point!(P, Pt, pb,
                                                               sj * (1 + im * ep), k, F, c),
                                         ep)
            lz, sz, dz = lz + l, sz * sg, dz + dl
        end
        P0t .= P0
        estimated_bias_point!(P0, P0t, pb0, zero(T), k, F, c)
    end
    if !started
        k1 = (one(decay) - decay^K) / (one(decay) - decay)
        return true, k1, sigma * k1, P0[1, 1], zero(T)
    end
    return sz > 0, dz / sigma, lz, real(P[1, 1]), imag(P[1, 1]) / ep
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fills the columns of the Mahalanobis level recursion at one count from the lattice of
[`estimated_bias_lattice`](@ref), on the lattice `xf` of [`mahalanobis_bias_grid`](@ref) that
shares its points.

The columns are ``g = \\ln \\det(I + \\sigma M_{K})`` and ``\\rho_{0} = g'``, which the law reads, and
``\\sigma V_{g}(\\sigma)`` and its derivative, which the dependence of the deviation reads: with
``V_{g} = g^{\\top} (I + \\sigma M_{K})^{-1} g = P_{11}``, the location moves ``g`` by
``t\\, \\sigma V_{g}`` to the first order where ``M_{K}`` takes ``t\\, g g^{\\top}``.

  - Below the seed: ``g = \\sigma \\kappa_{1}``, ``\\rho_{0} = \\kappa_{1}``, and ``V_{g}`` at ``s = 0``.
  - From the first point whose determinant is not positive: ``g = \\infty`` and zero elsewhere, as
    [`hac_log_det_path!`](@ref) marks such a point.
  - Above the top of the lattice: the continuation of [`mahalanobis_cut!`](@ref),
    ``g = g_{t} + N_{t} \\ln(\\sigma / \\sigma_{t})`` with ``N_{t} = \\sigma_{t} \\rho_{0}(\\sigma_{t})``, and
    ``\\sigma V_{g}`` held. There ``e^{-g / 2} < \\epsilon^{3}``.

# Arguments

  - `cols::NamedTuple`: `g`, `r`, `vt` and `rt` on `xf` (mutated).
  - `lat::NamedTuple`: Complex lattice after the count `K`.
  - `xf::AbstractRange`: Lattice of ``\\ln \\sigma``, whose point `j0 + i` is the point `i` of `lat`.
  - `j0::Integer`: Count of the points of `xf` below the seed.
  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of the terms of the estimate.

# Returns

  - `cols::NamedTuple`: The filled columns.

# Related

  - [`mahalanobis_level_bias`](@ref)
  - [`estimated_bias_lattice`](@ref)
"""
function estimated_mahalanobis_columns!(cols::NamedTuple, lat::NamedTuple,
                                        xf::AbstractRange, j0::Integer, decay::Number,
                                        K::Integer)
    (; g, r, vt, rt) = cols
    (; s, P, P0, lZ, sZ, dZ, ep, top) = lat
    k1 = (one(decay) - decay^K) / (one(decay) - decay)
    v0 = real(P0[1, 1])
    for j in 1:j0
        sig = exp(xf[j])
        g[j], r[j], vt[j], rt[j] = sig * k1, k1, sig * v0, v0
    end
    dead = false
    it = j0
    for i in 1:min(top[], length(xf) - j0)
        dead |= !(sZ[i] > 0)
        p = P[1, 1, i]
        g[j0 + i], r[j0 + i], vt[j0 + i], rt[j0 + i] = if dead
            (convert(eltype(g), Inf), zero(eltype(r)), zero(eltype(vt)), zero(eltype(rt)))
        else
            (lZ[i], dZ[i] / s[i], s[i] * real(p), real(p) + imag(p) / ep)
        end
        it = j0 + i
    end
    nt = exp(xf[it]) * r[it]
    for j in (it + 1):length(xf)
        g[j], r[j], vt[j], rt[j] = if dead
            (convert(eltype(g), Inf), zero(eltype(r)), zero(eltype(vt)), zero(eltype(rt)))
        else
            (g[it] + nt * (xf[j] - xf[it]), nt / exp(xf[j]), vt[it], zero(eltype(rt)))
        end
    end
    return cols
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Runs the level recursion of [`mahalanobis_level_bias`](@ref) on one pair of columns, and returns
the transform of the last level with the sum over the levels of the integral of each transform.

Each level ``k`` of the recursion gives the Laplace transform ``L_{k}`` of the Schur complement
``S_{k}`` of one direction in ``k + 1`` directions. The determinant of the estimate is the product
of the ``S_{k}``, so ``\\mathbb{E}[\\ln \\det W] = \\sum_{k} \\mathbb{E}[\\ln S_{k}]``, and each term is
``\\int_{0}^{\\infty} (e^{-s} - L_{k}(s))\\, s^{-1}\\, \\mathrm{d}s`` up to a constant. The function returns
``J = \\sum_{k = 0}^{n - 1} \\int L_{k}(s)\\, s^{-1}\\, \\mathrm{d}s`` on the output grid, so a change of
the columns moves ``\\mathbb{E}[\\ln \\det W]`` by minus the change of ``J``. With a cut, each level
reads the transform of [`mahalanobis_transform`](@ref) at the cut.

# Arguments

  - `grid::NamedTuple`: Grids of [`mahalanobis_bias_grid`](@ref).
  - `gh::VecNum`: ``g`` on the lattice (mutated by [`mahalanobis_cut!`](@ref)).
  - `rh::VecNum`: ``\\rho_{0}`` on the lattice (mutated by [`mahalanobis_cut!`](@ref)).
  - `peak::Option{<:Tuple{<:Number, <:Number}}`: The peak of [`hac_log_det_peak`](@ref), or
    `nothing`.
  - `shift::Tuple{<:Integer, <:Number}`: The shift ``m`` of the lattice and the normalisation
    ``\\tilde{c}``.
  - `n::Integer`: Count of assets that contribute to the statistic.
  - `bufs::NamedTuple`: Buffers `dg`, `dr`, `v`, `e` and `lv` of the recursion (mutated).

# Returns

  - `(so, L, J)::Tuple`: The output grid and the transform of the last level, and ``J``.

# Related

  - [`mahalanobis_level_bias`](@ref)
  - [`mahalanobis_level_step!`](@ref)
  - [`mahalanobis_transform`](@ref)
"""
function mahalanobis_level_sum!(grid::NamedTuple, gh::VecNum, rh::VecNum,
                                peak::Option{<:Tuple}, shift::Tuple, n::Integer,
                                bufs::NamedTuple)
    (; dg, dr, v, e, lv) = bufs
    m = shift[1]
    cut = mahalanobis_cut!(gh, rh, grid.xf, shift[2], peak)
    mahalanobis_pair_differences!(dg, dr, grid, gh, rh, shift)
    fill!(v, zero(eltype(v)))
    J = zero(eltype(v))
    for k in 0:(n - 1)
        if k > 0
            mahalanobis_level_step!(v, e, lv, dg, dr, grid)
        end
        mahalanobis_q_integral!(e, v, grid)
        so, L = mahalanobis_transform(merge(grid, (; cut)), gh, e, m)
        J += grid.ho * sum(L)
        if k == n - 1
            return so, L, J
        end
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the Mahalanobis bias factor at each count of `Ks` where each deviation is taken from an
estimated location: the law of the estimate on its dense form and, where the debias rule reads
it, the dependence of the deviation on the estimate.

The terms share their location, so the estimate of a pair with one shared history is
``W = Z^{\\top} M_{K} Z`` with the dense ``M_{K}`` of [`regime_bias_table`](@ref), and the recursion
starts from ``g = \\ln \\det(I + \\sigma M_{K})`` in place of the banded form. The lattice of
[`estimated_bias_lattice`](@ref) gives ``g`` and its derivative on the points of the grid, at the
step ``|\\ln \\lambda| / m`` in place of ``1/8``, and the peak of a HAC estimate takes the exact
slope of [`estimated_log_det_chain`](@ref). The deviation ``u = z - Z^{\\top} g`` reads the same
returns as ``W``, and [`estimated_mahalanobis_dependence`](@ref) gives the part of each method that
this dependence moves.

The columns agree with the eigenvalues of a dense ``M_{K}`` to ``10^{-11}``. Against a Monte Carlo
of a million draws of the statistic at half-lives of 5, 10 and 40, no lag to four lags and two to
12 assets, the three factors of [`ExactDebias`](@ref) are within 0.3 % at the steady state and
0.6 % at the first scored count, the error of the level recursion. The factors of the law alone,
[`LawDebias`](@ref), are 0.2 % to 1.4 % high. The pre-centred factors are 1.3 % to 3.1 % high
under HAC, and 7.5 % at the first scored count of a half-life of 40.

# Algorithm

 1. Make the complex lattice from ``\\ln \\epsilon + \\ln(1 - \\lambda)`` to 51, and the grid whose
    `xf` shares its points, up to the top of the output grid and one stencil past it.
 2. Advance the lattice to each count of `Ks` in order, and fill the columns with
    [`estimated_mahalanobis_columns!`](@ref).
 3. Read the factor of the count with [`estimated_mahalanobis_factor`](@ref).

# Arguments

  - `method::RegimeAdjustedMethod`: Regime adjustment method, which names the moment.
  - `decay::Number`: Decay of the weights.
  - `Ks::AbstractVector{<:Integer}`: Counts of observations, each above `n + 1`.
  - `n::Integer`: Count of assets that contribute to the statistic, at least two.
  - `hac_lags::Option{<:Integer}`: Count of HAC lags, or `nothing`.
  - `debias::AbstractRegimeDebias`: The debias rule, read by [`reads_dependence`](@ref).

# Returns

  - `factors::VecNum`: The factor at each count of `Ks`.

# Related

  - [`mahalanobis_level_bias`](@ref)
  - [`estimated_mahalanobis_columns!`](@ref)
  - [`estimated_mahalanobis_factor`](@ref)
  - [`mahalanobis_level_sum!`](@ref)
"""
function estimated_mahalanobis_level_bias(method::RegimeAdjustedMethod, decay::Number,
                                          Ks::AbstractVector{<:Integer}, n::Integer,
                                          hac_lags::Option{<:Integer},
                                          debias::AbstractRegimeDebias)
    T = typeof(log(decay))
    x0 = log(eps(T)) + log1p(-decay)
    # The output grid reaches 50 at a normalisation of one, so `xf` does too, with one stencil.
    lat = estimated_bias_lattice(decay, hac_lags, x0, 51 * one(T), Complex{T})
    hf = lat.h
    j0 = ceil(Int, (x0 - (-61 + log1p(-decay))) / hf)
    grid = mahalanobis_bias_grid(decay, hf, x0 - j0 * hf, 50 + 8 * hf)
    NF = length(grid.xf)
    cols = (; g = zeros(T, NF), r = zeros(T, NF), vt = zeros(T, NF), rt = zeros(T, NF))
    NN = length(grid.nodes)
    bufs = (; dg = Matrix{T}(undef, NN, length(grid.su)),
            dr = Matrix{T}(undef, NN, length(grid.su)), v = zeros(T, 8 + grid.NG),
            e = zeros(T, 10 + grid.NG), lv = (zeros(T, NN), zeros(T, NN), zeros(T, NN)))
    ctx = (; grid, cols, bufs, j0, n, k = lat.k, debias)
    out = zeros(T, length(Ks))
    K = 0
    for i in sortperm(Ks)
        while K < Ks[i]
            K += 1
            estimated_bias_count!(lat, decay, K)
            estimated_bias_freeze!(lat)
        end
        estimated_mahalanobis_columns!(cols, lat, grid.xf, j0, decay, K)
        out[i] = estimated_mahalanobis_factor(method, ctx, decay, K)
    end
    return out
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the Mahalanobis bias factor of one count under [`EstimatedCentring`](@ref) from the
columns of [`estimated_mahalanobis_columns!`](@ref).

# Algorithm

 1. Round the normalisation ``c`` to the lattice, ``\\tilde{c} = e^{m h_{f}}``.
 2. Find the peak with [`hac_log_det_peak`](@ref) on the slope of
    [`estimated_log_det_chain`](@ref), and take the value of ``g`` there from the chain too.
 3. Run [`mahalanobis_level_sum!`](@ref), read the factor of the law with
    [`regime_bias_factor`](@ref), and scale it by ``\\tilde{c} / c``.
 4. Where [`reads_dependence`](@ref) holds for the debias rule, replace it with the factor of
    [`estimated_mahalanobis_dependence`](@ref).

# Arguments

  - `method::RegimeAdjustedMethod`: Regime adjustment method, which names the moment.
  - `ctx::NamedTuple`: The grid, the columns, the buffers of the recursion, `j0`, the count of
    assets `n`, the weight `k` of each lag, and the debias rule.
  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of the terms of the estimate.

# Returns

  - `factor::Number`: The factor at `K`.

# Related

  - [`estimated_mahalanobis_level_bias`](@ref)
  - [`estimated_mahalanobis_dependence`](@ref)
"""
function estimated_mahalanobis_factor(method::RegimeAdjustedMethod, ctx::NamedTuple,
                                      decay::Number, K::Integer)
    (; grid, cols, bufs, n, k) = ctx
    hf = step(grid.xf)
    c = (one(decay) - decay) / (one(decay) - decay^K)
    m = round(Int, log(c) / hf)
    shift = (m, exp(m * hf))
    chain = sigma -> estimated_log_det_chain(decay, K, sigma, k)
    peak = hac_log_det_peak(cols.g, cols.r, grid.xf, function (sigma)
                                ch = chain(sigma)
                                return ch[1] ? ch[2] : -one(sigma)
                            end)
    peak = isnothing(peak) ? nothing : (peak[1], chain(peak[1])...)
    law = mahalanobis_level_sum!(grid, copy(cols.g), copy(cols.r),
                                 isnothing(peak) ? nothing : (peak[1], peak[4]), shift, n,
                                 bufs)
    b = regime_bias_factor(method, law[1], law[2], one(c), grid.ho) * shift[2] / c
    if !reads_dependence(ctx.debias)
        return b
    end
    return estimated_mahalanobis_dependence(method, b, ctx, law, peak, shift, decay, K)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Runs the level recursion on the columns of the estimate with the direction of the deviation taken
out, ``W_{\\perp}``, the part of [`estimated_mahalanobis_dependence`](@ref) that the first moment and
the log read.

In the frame where the deviation is the first direction, ``\\ln \\det W = \\ln \\det W_{\\perp} + \\ln S``,
with ``S`` the Schur complement of that direction, and ``d^{2} = \\|u\\|^{2} / S``. ``W_{\\perp}`` is a
plain weighted Wishart of ``n - 1`` directions, whose weights are those of the history and the new
return, ``\\tilde{M} = \\operatorname{diag}(M_{K}, 0)``, compressed onto the complement of
``e = (-g; 1)``. By the identity of Jacobi its log-determinant is that of ``\\tilde{M}`` plus
``\\ln (V / f)``, with ``V = e^{\\top} (I + \\sigma \\tilde{M})^{-1} e = 1 + V_{g}`` and ``f = e^{\\top} e``.

# Mathematical definition

```math
\\begin{align}
g_{\\perp}(\\sigma) &= g(\\sigma) + \\ln \\frac{V(\\sigma)}{f}\\,, \\quad
\\rho_{\\perp}(\\sigma) = \\rho_{0}(\\sigma) + \\frac{V'(\\sigma)}{V(\\sigma)}\\,, \\quad
V(\\sigma) = 1 + \\frac{\\sigma V_{g}(\\sigma)}{\\sigma}\\,.
\\end{align}
```

# Arguments

  - `ctx::NamedTuple`: The context of [`estimated_mahalanobis_factor`](@ref).
  - `peak::Option{<:Tuple}`: The peak and the values of [`estimated_log_det_chain`](@ref) there,
    or `nothing`. The cut stays at the peak of ``g``.
  - `shift::Tuple{<:Integer, <:Number}`: The shift ``m`` of the lattice and ``\\tilde{c}``.
  - `f::Number`: ``f = 1 + g^{\\top} g``, [`centring_factor`](@ref) of the deviation.

# Returns

  - `(so, L, J)::Tuple`: The run of [`mahalanobis_level_sum!`](@ref) over ``n`` levels.

# Related

  - [`estimated_mahalanobis_dependence`](@ref)
  - [`mahalanobis_level_sum!`](@ref)
"""
function estimated_perp_run(ctx::NamedTuple, peak::Option{<:Tuple}, shift::Tuple, f::Number)
    (; grid, cols, bufs, n) = ctx
    sig = exp.(grid.xf)
    V = one(f) .+ cols.vt ./ sig
    gp = cols.g .+ log.(V ./ f)
    rp = cols.r .+ (cols.rt ./ sig .- cols.vt ./ sig .^ 2) ./ V
    pk = isnothing(peak) ? nothing : (peak[1], peak[4] + log((1 + peak[5]) / f))
    return mahalanobis_level_sum!(grid, gp, rp, pk, shift, n, bufs)
end
"""
    estimated_mahalanobis_dependence(::RegimeAdjustedMethod, b::Number, ctx, law, peak, shift, decay, K)
    estimated_mahalanobis_dependence(::RootMeanSquaredAdjusted, b::Number, ctx, law, peak, shift, decay, K)
    estimated_mahalanobis_dependence(::FirstMomentRegimeAdjusted, b::Number, ctx, law, peak, shift, decay, K)
    estimated_mahalanobis_dependence(::LogRegimeAdjusted, b::Number, ctx, law, peak, shift, decay, K)

Computes the Mahalanobis factor of a method with the dependence of the deviation on the estimate,
from the factor of the law.

The deviation is ``u = z - Z^{\\top} g`` and the statistic ``d^{2} = u^{\\top} (c W)^{-1} u / f``. In the
frame where ``u`` is the first direction, ``d^{2} \\propto \\|u\\|^{2} / S`` with ``S`` the Schur
complement of that direction, and ``\\|u\\|^{2} / f`` is a ``\\chi^{2}_{n}``.

  - The root mean square reads ``\\mathbb{E}[u^{\\top} W^{-1} u] = \\mathbb{E}[\\operatorname{tr} W^{-1}] + \\mathbb{E}[y^{\\top} W^{-1} y]``,
    ``y = Z^{\\top} g``, and the second term is the derivative in ``t`` of
    ``\\mathbb{E}[\\ln \\det Z^{\\top}(M_{K} + t\\, g g^{\\top}) Z]``: the columns move by ``t\\, \\sigma V_{g}``,
    and a central difference of ``J`` at ``t = \\pm \\epsilon^{1/3}`` gives it, at a fixed cut.
  - The log reads ``\\mathbb{E}[\\ln d^{2}]``, and
    ``\\ln d^{2} = \\ln \\|u\\|^{2} + \\ln \\det W_{\\perp} - \\ln \\det W`` up to the scale. Each
    log-determinant is the sum over the levels of ``\\mathbb{E}[\\ln S_{k}] = \\int (e^{-s/2} - L_{k}) s^{-1} \\mathrm{d}s``,
    from [`mahalanobis_level_sum!`](@ref) and [`estimated_perp_run`](@ref).
  - The first moment reads ``\\mathbb{E}[d]``. Given the other directions, ``S`` is a quadratic form
    in the first, whose coordinate along ``e`` is ``\\|u\\| / \\sqrt{f}``, so the mean over that
    coordinate is in closed form. With the mean log-determinant of each level inside the
    exponential, as the recursion takes it, the transform of the last level ``L`` becomes
    ``L^{n + 1} L_{\\perp}^{-n}``.

Any other method returns the factor of the law.

# Mathematical definition

```math
\\begin{align}
b_{R} &= \\frac{1}{f} \\left(b - \\frac{J(t) - J(-t)}{2 t\\, n c}\\right)\\,, \\\\
b_{L} &= \\frac{\\tilde{c}}{c} \\exp\\left(J - J_{\\perp} - h \\sum_{s} e^{-s / 2}\\right)\\,, \\\\
b_{F} &= \\frac{\\tilde{c}}{c} \\frac{1}{2 \\pi} \\left(h \\sum_{s} s^{1/2} L(s)^{n + 1} L_{\\perp}(s)^{-n}\\right)^{2}\\,.
\\end{align}
```

Where ``J`` sums the levels of ``W`` and ``J_{\\perp}`` the ``n - 1`` levels of ``W_{\\perp}``, and the
sums run over the output grid of the last level.

# Arguments

  - `method::RegimeAdjustedMethod`: Regime adjustment method.
  - `b::Number`: The factor of the law.
  - `ctx::NamedTuple`: The context of [`estimated_mahalanobis_factor`](@ref).
  - `law::Tuple`: The run of [`mahalanobis_level_sum!`](@ref) of the law: the output grid, the
    transform of the last level and ``J``.
  - `peak::Option{<:Tuple}`: The peak and the values of [`estimated_log_det_chain`](@ref) there,
    or `nothing`.
  - `shift::Tuple{<:Integer, <:Number}`: The shift ``m`` of the lattice and ``\\tilde{c}``.
  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of the terms of the estimate.

# Returns

  - `factor::Number`: The factor of the method.

# Related

  - [`estimated_mahalanobis_factor`](@ref)
  - [`estimated_perp_run`](@ref)
  - [`mahalanobis_level_sum!`](@ref)
"""
function estimated_mahalanobis_dependence(::RegimeAdjustedMethod, b::Number, ::NamedTuple,
                                          ::Tuple, ::Any, ::Tuple, ::Number, ::Integer)
    return b
end
function estimated_mahalanobis_dependence(::RootMeanSquaredAdjusted, b::Number,
                                          ctx::NamedTuple, ::Tuple, peak::Option{<:Tuple},
                                          shift::Tuple, decay::Number, K::Integer)
    (; grid, cols, bufs, n) = ctx
    ep = cbrt(eps(typeof(b)))
    Jt = map((ep, -ep)) do t
        pk = isnothing(peak) ? nothing : (peak[1], peak[4] + t * peak[1] * peak[5])
        return mahalanobis_level_sum!(grid, cols.g .+ t .* cols.vt, cols.r .+ t .* cols.rt,
                                      pk, shift, n, bufs)[3]
    end
    c = (one(decay) - decay) / (one(decay) - decay^K)
    return (b - (Jt[1] - Jt[2]) / (2 * ep) / (n * c)) /
           centring_factor(EstimatedCentring(), decay, K + 1)
end
function estimated_mahalanobis_dependence(::LogRegimeAdjusted, ::Number, ctx::NamedTuple,
                                          law::Tuple, peak::Option{<:Tuple}, shift::Tuple,
                                          decay::Number, K::Integer)
    so, _, J = law
    _, Lp, Jp = estimated_perp_run(ctx, peak, shift,
                                   centring_factor(EstimatedCentring(), decay, K + 1))
    ho = ctx.grid.ho
    # W⊥ has n - 1 directions: drop the last level of the run.
    Jp -= ho * sum(Lp)
    c = (one(decay) - decay) / (one(decay) - decay^K)
    return shift[2] / c * exp(J - Jp - ho * sum(s -> exp(-s / 2), so))
end
function estimated_mahalanobis_dependence(::FirstMomentRegimeAdjusted, ::Number,
                                          ctx::NamedTuple, law::Tuple,
                                          peak::Option{<:Tuple}, shift::Tuple,
                                          decay::Number, K::Integer)
    so, L, _ = law
    _, Lp, _ = estimated_perp_run(ctx, peak, shift,
                                  centring_factor(EstimatedCentring(), decay, K + 1))
    n = ctx.n
    acc = zero(eltype(L))
    for a in eachindex(so, L, Lp)
        if L[a] > 0 && Lp[a] > 0
            acc += sqrt(so[a]) * exp((n + 1) * log(L[a]) - n * log(Lp[a]))
        end
    end
    c = (one(decay) - decay) / (one(decay) - decay^K)
    return shift[2] / c * (ctx.grid.ho * acc)^2 / (2 * pi)
end
"""
    mahalanobis_level_bias(method, decay::Number, Ks, n::Integer, hac_lags, ::PreCentred, debias)
    mahalanobis_level_bias(method, decay::Number, Ks, n::Integer, hac_lags, ::ZeroStartCentring, debias)
    mahalanobis_level_bias(method, decay::Number, Ks, n::Integer, hac_lags, ::EstimatedCentring, debias)

Computes the Mahalanobis bias factor at each count of `Ks` on the law of the centring of the
estimator.

A pre-centred estimate, and an estimate whose location starts at zero, read
[`mahalanobis_level_bias`](@ref) without a centring: their deviation does not share the terms of
the estimate, so the debias rule changes nothing. Under [`EstimatedCentring`](@ref) a count up to
``K_{h} = \\lceil \\ln \\sqrt{\\epsilon} / \\ln \\lambda \\rceil`` reads
[`estimated_mahalanobis_level_bias`](@ref). Past ``K_{h}`` both factors change as ``\\lambda^{K}``,
so a count takes the pre-centred factor times the ratio of the two at ``K_{h}``, as
[`regime_bias_table`](@ref) does. The forward lattice then stops at ``K_{h}``.

# Arguments

  - `method::RegimeAdjustedMethod`: Regime adjustment method, which names the moment.
  - `decay::Number`: Decay of the weights.
  - `Ks::AbstractVector{<:Integer}`: Counts of observations, each above `n + 1`.
  - `n::Integer`: Count of assets that contribute to the statistic, at least two.
  - `hac_lags::Option{<:Union{<:Integer, <:VecNum}}`: Count of HAC lags, the weight of each lag
    (pre-centred law only), or `nothing`.
  - `centring`: The centring of the estimator.
  - `debias::AbstractRegimeDebias`: The debias rule, [`ExactDebias`](@ref) by default.

# Returns

  - `factors::VecNum`: The factor at each count of `Ks`.

# Related

  - [`mahalanobis_level_bias`](@ref)
  - [`estimated_mahalanobis_level_bias`](@ref)
  - [`mahalanobis_bias_nodes`](@ref)
"""
function mahalanobis_level_bias(method::RegimeAdjustedMethod, decay::Number,
                                Ks::AbstractVector{<:Integer}, n::Integer,
                                hac_lags::Option{<:Union{<:Integer, <:VecNum}},
                                ::Union{PreCentred, ZeroStartCentring},
                                ::AbstractRegimeDebias = ExactDebias())
    return mahalanobis_level_bias(method, decay, Ks, n, hac_lags)
end
function mahalanobis_level_bias(method::RegimeAdjustedMethod, decay::Number,
                                Ks::AbstractVector{<:Integer}, n::Integer,
                                hac_lags::Option{<:Integer}, ::EstimatedCentring,
                                debias::AbstractRegimeDebias = ExactDebias())
    Kh = ceil(Int, log(sqrt(eps(typeof(log(decay))))) / log(decay))
    lo = findall(<=(Kh), Ks)
    hi = findall(>(Kh), Ks)
    out = similar(Ks, typeof(log(decay)))
    fe = estimated_mahalanobis_level_bias(method, decay,
                                          vcat(Ks[lo], isempty(hi) ? Int[] : [Kh]), n,
                                          hac_lags, debias)
    out[lo] .= view(fe, eachindex(lo))
    if !isempty(hi)
        fp = mahalanobis_level_bias(method, decay, vcat(Ks[hi], Kh), n, hac_lags)
        out[hi] .= view(fp, eachindex(hi)) .* (fe[end] / fp[end])
    end
    return out
end
