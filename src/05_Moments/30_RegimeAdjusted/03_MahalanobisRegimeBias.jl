"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the weights of the 8-point Lagrange interpolant on a uniform grid at one point.

The stencil is the eight grid points nearest to `x`, moved inside the grid at its two ends.

# Arguments

  - `x0::Number`: First point of the grid.
  - `h::Number`: Step of the grid.
  - `N::Integer`: Count of points in the grid, at least eight.
  - `x::Number`: Point to interpolate at.

# Returns

  - `(i0, wts)::Tuple{Int, NTuple{8, <:Number}}`: Index of the first point of the stencil, and the
    weight of each of its eight points.

# Related

  - [`mahalanobis_bias_grid`](@ref)
"""
function lagrange_stencil(x0::Number, h::Number, N::Integer, x::Number)
    p = (x - x0) / h + 1
    i0 = clamp(floor(Int, p) - 3, 1, N - 7)
    wts = ntuple(a -> prod(b -> b == a ? one(p) : (p - (i0 + b - 1)) / (a - b), 1:8), 8)
    return i0, wts
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the weights of the degree-7 Chebyshev interpolant on ``[0, s_{c}]``, and of its integral
from zero, at one point.

The eight nodes are ``s_{l} = s_{c} (1 + \\cos\\theta_{l}) / 2``, ``\\theta_{l} = \\pi (l - 1/2) / 8``.

# Arguments

  - `sc::Number`: Right end ``s_{c}`` of the interval.
  - `s::Number`: Point in ``[0, s_{c}]``.

# Returns

  - `(val, int)::Tuple{NTuple{8, <:Number}, NTuple{8, <:Number}}`: Weight of each node in the
    value at `s`, and in the integral from zero to `s`.

# Related

  - [`mahalanobis_bias_grid`](@ref)
"""
function chebyshev_stencil(sc::Number, s::Number)
    t = clamp(2 * s / sc - 1, -one(sc), one(sc))
    th = acos(t)
    # Integral of T_j from -1 to t.
    IT = ntuple(8) do j1
        j = j1 - 1
        if j == 0
            t + 1
        elseif j == 1
            (t^2 - 1) / 2
        else
            (cos((j + 1) * th) / (j + 1) - cos((j - 1) * th) / (j - 1)) / 2 -
            ((-one(sc))^(j + 1) / (j + 1) - (-one(sc))^(j - 1) / (j - 1)) / 2
        end
    end
    thl = ntuple(l -> pi * (l - one(sc) / 2) / 8, 8)
    cj(j, l) = (j == 0 ? one(sc) : 2 * one(sc)) / 8 * cos(j * thl[l])
    val = ntuple(l -> sum(j -> cj(j, l) * cos(j * th), 0:7), 8)
    int = ntuple(l -> sum(j -> cj(j, l) * IT[j + 1], 0:7) * sc / 2, 8)
    return val, int
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes the fixed grids and interpolation weights of the level recursion of
[`mahalanobis_level_bias`](@ref), in the number type of `decay`.

The removed part ``q`` of the recursion is held on ``[0, s_{c}]`` by eight Chebyshev values and on
``\\ln s \\in [-1, 30]`` by a grid of step ``1/4``. The two parts overlap on the first four steps of
the grid, and the grid values there are always the Chebyshev values, so the function has one value
at every point. A Chebyshev interpolant of more nodes, or a grid that starts lower, amplifies
round-off from one level to the next: the recursion multiplies an error of frequency ``\\omega``
by about ``1 - \\ln(\\omega) / 2``.

# Arguments

  - `decay::Number`: Decay of the weights. Its number type is the type of every weight.
  - `Klat::Integer`: Largest count of observations the lattice of `g` must serve.

# Returns

  - `grid::NamedTuple`: The step `h`, the grid `sg`, the node points `nodes`, the steps `su` of the
    integral, the pair stencils, the output grid `so` of [`regime_bias_table`](@ref), the
    lattice `xf` on which `g` and ``\\rho_{0}`` are tabulated, the bounds `geo` of the grid, and
    `cut = nothing`, which [`mahalanobis_level_bias`](@ref) replaces where the transform is cut.

# Related

  - [`mahalanobis_level_bias`](@ref)
  - [`lagrange_stencil`](@ref)
  - [`chebyshev_stencil`](@ref)
"""
function mahalanobis_bias_grid(decay::Number)
    o = one(decay)
    h = o / 4
    xlo = -o
    xhi = 30 * o
    sc = exp(xlo + 4 * h)
    sg = exp.(range(xlo, xhi; step = h))
    NG = length(sg)
    cheb = [sc * (1 + cos(pi * (l - o / 2) / 8)) / 2 for l in 1:8]
    nodes = vcat(cheb, view(sg, 6:NG))
    su = exp.(range(-30 * o, 40 * o; step = h))
    hf = o / 8
    # The lattice serves every shift `m hf` of the normalisation, down to `log(1 - decay)`.
    xf = range(-61 * o + log1p(-decay), 42 * o; step = hf)
    NF = length(xf)
    geo = (; sc, xlo, xhi, h, NG, send = sg[end])
    fstencil(s) = lagrange_stencil(first(xf), hf, NF, log(s))
    pairs = stencil_table(nodes .+ permutedims(su)) do s
        return (mahalanobis_value_stencil(s, geo), mahalanobis_integral_stencil(s, geo),
                fstencil(s))
    end
    ho = o / 10
    so = exp.(range(-60 * o, 50 * o; step = ho))
    return (; h, sc, sg, NG, nodes, su, xf, pairs, ho, geo, cut = nothing,
            junction = stencil_table(s -> chebyshev_stencil(sc, s)[1], view(sg, 1:5)),
            Qjunction = stencil_table(s -> chebyshev_stencil(sc, s)[2], view(sg, 1:5)),
            Qcheb = stencil_table(s -> chebyshev_stencil(sc, s)[2], cheb),
            fnodes = stencil_table(fstencil, nodes), so,
            Qout = stencil_table(s -> mahalanobis_integral_stencil(s, geo), so),
            fout = stencil_table(fstencil, so))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Evaluates a stencil function at every point of an array, and stores the results in an array of
the element type of the first result.

A loop fills the table, because a comprehension over points of an abstract number type leaves
its element type to `collect`, and the analysis of the package then meets a generic method that
does not apply.

# Arguments

  - `f`: Function from a point to its stencil.
  - `pts::AbstractArray`: Points, at least one.

# Returns

  - `table::AbstractArray`: `f` at each point, with the shape of `pts`.

# Related

  - [`mahalanobis_bias_grid`](@ref)
"""
function stencil_table(f, pts::AbstractArray)
    table = Array{typeof(f(first(pts)))}(undef, size(pts))
    for i in eachindex(pts, table)
        table[i] = f(pts[i])
    end
    return table
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the stencil of the removed part ``q`` at one point, as it acts on the Chebyshev values of
``q`` followed by ``q`` on the grid.

The Chebyshev interpolant serves ``[0, s_{c}]``, the grid Lagrange interpolant serves the grid,
and past the grid ``q(s) = s_{\\mathrm{end}}\\, q(s_{\\mathrm{end}}) / s``, because every eigenvalue of
the removed part then has ``s \\mu \\gg 1``.

# Arguments

  - `s::Number`: Point to evaluate at.
  - `geo::NamedTuple`: `sc`, `xlo`, `xhi`, `h`, `NG` and `send` of [`mahalanobis_bias_grid`](@ref).

# Returns

  - `st::Tuple{Int, NTuple{8, <:Number}}`: First index and weights, for [`stencil_dot`](@ref).

# Related

  - [`mahalanobis_integral_stencil`](@ref)
  - [`mahalanobis_bias_grid`](@ref)
"""
function mahalanobis_value_stencil(s::Number, geo::NamedTuple)
    if s <= geo.sc
        return 1, chebyshev_stencil(geo.sc, s)[1]
    elseif log(s) <= geo.xhi
        i0, w = lagrange_stencil(geo.xlo, geo.h, geo.NG, log(s))
        return 8 + i0, w
    end
    return geo.NG + 1, ntuple(a -> a == 8 ? geo.send / s : zero(s), 8)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the stencil of the integral ``Q(s) = \\int_{0}^{s} q`` at one point, as it acts on the
vector that [`mahalanobis_q_integral!`](@ref) fills.

The Chebyshev integral serves ``[0, s_{c}]``, the grid Lagrange interpolant of ``Q`` serves the
grid, and past the grid
``Q(s) = Q(s_{\\mathrm{end}}) + s_{\\mathrm{end}}\\, q(s_{\\mathrm{end}}) \\ln(s / s_{\\mathrm{end}})``.

# Arguments

  - `s::Number`: Point to evaluate at.
  - `geo::NamedTuple`: `sc`, `xlo`, `xhi`, `h`, `NG` and `send` of [`mahalanobis_bias_grid`](@ref).

# Returns

  - `st::Tuple{Int, NTuple{8, <:Number}}`: First index and weights, for [`stencil_dot`](@ref).

# Related

  - [`mahalanobis_value_stencil`](@ref)
  - [`mahalanobis_bias_grid`](@ref)
"""
function mahalanobis_integral_stencil(s::Number, geo::NamedTuple)
    if s <= geo.sc
        return 1, chebyshev_stencil(geo.sc, s)[2]
    elseif log(s) <= geo.xhi
        i0, w = lagrange_stencil(geo.xlo, geo.h, geo.NG, log(s))
        return 8 + i0, w
    end
    return geo.NG + 2, ntuple(a -> a == 7 ? one(s) : a == 8 ? log(s) - geo.xhi : zero(s),
                              8)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Evaluates an 8-point stencil on a vector.

# Arguments

  - `st::Tuple{Integer, NTuple{8, <:Number}}`: First index and weights of the stencil.
  - `v::VecNum`: Vector of values.
  - `m::Integer`: Shift of the first index.

# Returns

  - `val::Number`: ``\\sum_{a} w_{a} v_{i_{0} + m + a - 1}``.

# Related

  - [`mahalanobis_level_bias`](@ref)
"""
function stencil_dot(st::Tuple{<:Integer, <:NTuple{8, <:Number}}, v::VecNum, m::Integer = 0)
    i0, w = st
    return sum(a -> w[a] * v[i0 + m + a - 1], 1:8)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the bias factor of the squared Mahalanobis distance that a regime method reads, at each
count of observations in `Ks`, by the level recursion.

The squared distance of a correctly calibrated return ``u``, independent of the estimate, is
``u^{\\top} \\hat{C}^{-1} u = \\chi^{2}_{n} R``, where ``R = 1 / S`` and ``S`` is the Schur
complement of one direction in ``W = \\sum_{j} w_{j} z_{j} z_{j}^{\\top}``,
``z_{j} \\sim N(0, I_{n})``. Each method reads its own moment of ``R``, as
[`regime_bias_factor`](@ref) states, so the factor comes from the Laplace transform of ``S``.
Given the other directions, ``S`` is a weighted sum of ``\\chi^{2}_{1}`` terms. The exact step
from ``k`` directions to ``k + 1`` then reads the mean log-determinant of the deflated weights,
and the recursion takes the mean inside the exponential. That is the one approximation. It is
exact at one asset and at equal weights, where the spectrum is not random, and on exponential
weights at 12 assets and a half-life of 10 the three factors are 5e-4 to 7e-4 below a Monte Carlo
of a million draws, against 0.55 %, 2.5 % and 4.5 % above it for the factor of the mean alone.

A HAC estimate is ``Z^{\\top} A Z`` with the banded weight matrix ``A`` of
[`regime_bias_table`](@ref), so it has the law of a plain estimate whose weights are the
eigenvalues ``\\mu`` of ``A``, and the recursion starts from ``D_{0}(s) = \\ln \\det(I + s A)``.
``A`` has negative eigenvalues, so ``D_{0}`` is real only below the first zero of the
determinant, and no moment of ``R`` is finite: the Schur complement has a positive density at
zero. The transform is cut at the maximum ``s^{*}`` of ``D_{0}``, where ``\\rho_{0} = 0``, as the
table of one direction is. For the recursion, [`mahalanobis_cut!`](@ref) continues ``D_{0}`` as
the transform of its saturated weights, so the recursion runs as on the plain weights, and
[`mahalanobis_transform`](@ref) cuts the last transform at ``s^{*}``. The factor is then exact at
one asset on that cut and where ``A`` is positive definite. At 12 assets, a half-life of 10 and
two lags the three factors are 1.2e-3, 8e-4 and 3e-4 below a Monte Carlo of 4 million draws of the
HAC estimate itself, against 1.8 %, 5.3 % and 8.9 % above it for the factor of the mean. Where the
effective count of observations is near the gate the recursion is 0.2 % to 0.7 % below the Monte
Carlo, with or without HAC: at 3.6 effective observations and three assets the plain weights give
0.9 % and 0.6 %, and the HAC weights 0.7 % and 0.3 %. The first scored rows of five assets at a
half-life of 5 and one lag, where 4e-4 of draws are not positive definite, read 0.3 % and 0.03 %
below it.

# Mathematical definition

```math
\\begin{align}
D_{0}(s) &= \\sum_{j} \\ln(1 + s w_{j})\\,, \\qquad \\rho_{k} = D_{k}'\\,, \\qquad D_{k}(0) = 0\\,, \\\\
\\rho_{k+1}(s) &= \\rho_{k}(s) + \\frac{1}{2} \\int_{0}^{\\infty}
e^{-(D_{k}(s + u) - D_{k}(s)) / 2}\\, \\bigl(\\rho_{k}(s + u) - \\rho_{k}(s)\\bigr)\\,
\\frac{\\mathrm{d}u}{u}\\,, \\\\
\\mathbb{E}\\bigl[e^{-t S}\\bigr] &\\approx e^{-D_{n-1}(2 t) / 2}\\,, \\qquad 2 t \\leq s^{*}\\,.
\\end{align}
```

Where:

  - ``w_{j}``: Normalised weight of the observation ``j`` steps before the newest, or the
    eigenvalue ``\\mu_{j}`` of ``A`` with a HAC adjustment.
  - ``D_{k}``: Log-determinant of the Laplace transform after ``k`` directions are removed.
  - ``s^{*}``: The cut, ``\\infty`` without a HAC adjustment or where ``A`` is positive definite.
    The recursion reads ``D_{0}`` with the continuation of [`mahalanobis_cut!`](@ref).

# Algorithm

 1. Tabulate ``D_{0}`` and its derivative on the lattice for the weights ``\\lambda^{j}``, or for
    the banded matrix ``A / c``, with [`mahalanobis_lattice`](@ref), and keep a copy at each count
    of `Ks`.
 2. For each count, round the normalisation ``c = (1 - \\lambda) / (1 - \\lambda^{K})`` to the
    lattice, ``\\tilde{c} = e^{m h_{f}}``, so the weights ``\\tilde{c} \\lambda^{j}`` read the
    lattice without interpolation.
 3. Where the count has a cut, continue the lattice with [`mahalanobis_cut!`](@ref).
 4. Run the recursion for ``n - 1`` levels on the removed part ``q_{k} = \\rho_{0} - \\rho_{k}``,
    with the trapezoid rule on ``\\ln u``. The result at ``s`` reads ``q_{k}`` only on
    ``[s, \\infty)``, so an error moves only toward smaller ``s``.
 5. Read the factor of the method off the transform of [`mahalanobis_transform`](@ref) with
    [`regime_bias_factor`](@ref), and scale it by ``\\tilde{c} / c``.

# Arguments

  - `method::RegimeAdjustedMethod`: Regime adjustment method, which names the moment.
  - `decay::Number`: Decay of the weights.
  - `Ks::AbstractVector{<:Integer}`: Counts of observations, each above `n + 1`.
  - `n::Integer`: Count of assets that contribute to the statistic, at least two.
  - `hac_lags::Option{<:Integer}`: Count of HAC lags, or `nothing`.

# Returns

  - `factors::VecNum`: The factor at each count of `Ks`.

# Related

  - [`mahalanobis_bias_nodes`](@ref)
  - [`mahalanobis_bias_grid`](@ref)
  - [`mahalanobis_lattice`](@ref)
  - [`mahalanobis_cut!`](@ref)
  - [`mahalanobis_transform`](@ref)
  - [`regime_bias_factor`](@ref)
  - [`MahalanobisTarget`](@ref)
"""
function mahalanobis_level_bias(method::RegimeAdjustedMethod, decay::Number,
                                Ks::AbstractVector{<:Integer}, n::Integer,
                                hac_lags::Option{<:Integer} = nothing)
    grid = mahalanobis_bias_grid(decay)
    hf = step(grid.xf)
    G, Rh, peaks = mahalanobis_lattice(decay, Ks, grid.xf, hac_lags)
    T = eltype(G)
    NN = length(grid.nodes)
    dg = Matrix{T}(undef, NN, length(grid.su))
    dr = similar(dg)
    v = zeros(T, 8 + grid.NG)
    e = zeros(T, 10 + grid.NG)
    bufs = (zeros(T, NN), zeros(T, NN), zeros(T, NN))
    return map(eachindex(Ks)) do i
        c = (one(decay) - decay) / (one(decay) - decay^Ks[i])
        m = round(Int, log(c) / hf)
        shift = (m, exp(m * hf))
        gh = view(G, :, i)
        rh = view(Rh, :, i)
        cut = mahalanobis_cut!(gh, rh, grid.xf, shift[2], peaks[i])
        mahalanobis_pair_differences!(dg, dr, grid, gh, rh, shift)
        fill!(v, zero(T))
        for _ in 1:(n - 1)
            mahalanobis_level_step!(v, e, bufs, dg, dr, grid)
        end
        mahalanobis_q_integral!(e, v, grid)
        so, L = mahalanobis_transform(merge(grid, (; cut)), gh, e, m)
        return regime_bias_factor(method, so, L, one(c), grid.ho) * shift[2] / c
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Tabulates the log-determinant ``g`` of the Laplace transform of one direction and its derivative
``\\rho_{0}`` on the lattice of [`mahalanobis_bias_grid`](@ref), at each count of `Ks`, before the
normalisation of the weights.

Without a HAC adjustment the weights are ``\\lambda^{j}``, so ``g(s) = \\sum_{j} \\ln(1 + s \\lambda^{j})``
and ``\\rho_{0}(s) = \\sum_{j} \\lambda^{j} / (1 + s \\lambda^{j})``, and the table adds one
observation at a time. The weights are positive, so no count has a cut.

With one, [`hac_log_det_path!`](@ref) gives ``g(s) = \\ln \\det(I + s B)`` for the banded matrix
``B = A / c`` at every count, one lattice point at a time, and [`hac_log_det_peak`](@ref) finds
the cut of each count. A point past the first zero of the determinant holds ``g = \\infty``.

# Arguments

  - `decay::Number`: Decay of the weights.
  - `Ks::AbstractVector{<:Integer}`: Counts of observations.
  - `xf::AbstractRange`: Lattice of ``\\ln s``.
  - `hac_lags::Option{<:Integer}`: Count of HAC lags, or `nothing`.

# Returns

  - `(G, Rh, peaks)::Tuple`: ``g`` and ``\\rho_{0}`` with one column for each count, and for each
    count the peak of [`hac_log_det_peak`](@ref), or `nothing` where the transform takes no cut.

# Related

  - [`mahalanobis_level_bias`](@ref)
  - [`hac_log_det_path!`](@ref)
  - [`hac_log_det_peak`](@ref)
"""
function mahalanobis_lattice(decay::Number, Ks::AbstractVector{<:Integer},
                             xf::AbstractRange, ::Nothing)
    ef = exp.(xf)
    ghat = zero(ef)
    rhat = zero(ef)
    G = Matrix{eltype(ef)}(undef, length(ef), length(Ks))
    Rh = similar(G)
    lam = one(decay)
    j = 0
    for i in sortperm(Ks)
        while j < Ks[i]
            ghat .+= log1p.(lam .* ef)
            rhat .+= lam ./ (1 .+ lam .* ef)
            lam *= decay
            j += 1
        end
        G[:, i] .= ghat
        Rh[:, i] .= rhat
    end

    return G, Rh, fill(nothing, length(Ks))
end
function mahalanobis_lattice(decay::Number, Ks::AbstractVector{<:Integer},
                             xf::AbstractRange, hac_lags::Integer)
    ef = exp.(xf)
    G = Matrix{eltype(ef)}(undef, length(ef), length(Ks))
    Rh = similar(G)
    g = Vector{eltype(ef)}(undef, maximum(Ks))
    r = similar(g)
    for p in eachindex(ef)
        hac_log_det_path!(g, r, ef[p], decay, hac_lags)
        for i in eachindex(Ks)
            G[p, i] = g[Ks[i]]
            Rh[p, i] = r[Ks[i]]
        end
    end
    peaks = map(i -> hac_log_det_peak(view(G, :, i), view(Rh, :, i), xf, decay, Ks[i],
                                      hac_lags), eachindex(Ks))

    return G, Rh, peaks
end
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
  - `hac_lags::Integer`: Count of HAC lags.

# Returns

  - `(g, r)::Tuple{<:VecNum, <:VecNum}`: The filled vectors.

# Related

  - [`mahalanobis_lattice`](@ref)
  - [`hac_ldl_taylor_row!`](@ref)
"""
function hac_log_det_path!(g::VecNum, r::VecNum, s::Number, decay::Number,
                           hac_lags::Integer)
    z = ntuple(_ -> zero(s), 3)
    Lb = max(hac_lags, 1)
    buf = (; l = fill(z, Lb, Lb), d = fill(z, Lb), row = fill(z, Lb))
    lam = one(decay)
    gk = zero(s)
    rk = zero(s)
    for m in eachindex(g, r)
        dn = hac_ldl_taylor_row!(buf, s * lam, lam, decay, min(hac_lags, m - 1))
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
bisection on the exact derivative of [`hac_log_det_slopes`](@ref) finds it. A shift of the cut by
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
  - `hac_lags::Integer`: Count of HAC lags.

# Returns

  - `peak::Option{<:Tuple{<:Number, <:Number}}`: ``(\\sigma^{*}, \\ln \\det(I + \\sigma^{*} B))``, or
    `nothing` where the derivative is positive on the whole lattice, because ``B`` is positive
    definite.

# Related

  - [`mahalanobis_lattice`](@ref)
  - [`mahalanobis_cut!`](@ref)
  - [`hac_log_det_slopes`](@ref)
"""
function hac_log_det_peak(gh::VecNum, rh::VecNum, xf::AbstractRange, decay::Number,
                          K::Integer, hac_lags::Integer)
    p = findfirst(i -> !(rh[i] > zero(rh[i])) || isinf(gh[i]), eachindex(gh, rh))
    if isnothing(p)
        return nothing
    end
    c = (one(decay) - decay) / (one(decay) - decay^K)
    function slope(sigma)
        sl = hac_log_det_slopes(decay, K, sigma / c, hac_lags)
        return isnothing(sl) ? -one(sigma) : sl[1] / c
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

Returns the nodes and weights of the 4-point Gauss–Legendre rule on ``[-1, 1]``, in the number
type of `o`.

# Arguments

  - `o::Number`: One, in the number type of the rule.

# Returns

  - `rule::NTuple{4, <:Tuple{<:Number, <:Number}}`: The pair of node and weight of each point.

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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Continues the lattice of one count past the peak of its saturated count, and returns the cut and
the lattice that the last transform reads.

A Laplace transform of a positive variable is log-convex, so ``\\rho_{0}`` falls and stays
positive. With negative eigenvalues ``\\rho_{0}`` falls to zero at the cut ``\\sigma^{*}`` and to
``-\\infty`` at the first zero of the determinant, so no continuation from ``\\sigma^{*}`` is the
transform of a law, and the level recursion is not stable on one. The saturated count
``\\sigma \\rho_{0}(\\sigma)`` counts the weights with ``\\sigma \\mu \\gg 1``: it rises to ``K`` for a
positive spectrum, and the negative eigenvalues pull it down before ``\\sigma^{*}``. Past its
maximum ``N_{m}`` at ``\\sigma_{m}`` the lattice takes ``\\rho_{0} = N_{m} / \\sigma``, the transform of
``N_{m}`` saturated weights. The value and the slope of the count agree at ``\\sigma_{m}``, so the
join is smooth, ``\\rho_{0}`` falls and stays positive, and the recursion runs on the whole grid, as
it does on the plain weights. The last transform reads the true ``g`` up to ``\\sigma^{*}``, so the
factor stays exact at one asset. A hard cut, where the transform is zero past ``\\sigma^{*}``, puts
a pole ``k / (s^{*} - s)`` into the level ``k``, and a continuation from ``\\sigma^{*}`` makes
``\\rho_{0}`` rise: the grid holds neither, and both give factors that are not finite near the
gate.

# Mathematical definition

```math
\\begin{align}
N_{m} &= \\max_{\\sigma \\leq \\sigma^{*}} \\sigma \\rho_{0}(\\sigma)\\,, \\qquad g(\\sigma) = g(\\sigma_{m}) + N_{m} \\ln\\frac{\\sigma}{\\sigma_{m}}\\,, \\qquad \\rho_{0}(\\sigma) = \\frac{N_{m}}{\\sigma}\\,, \\qquad \\sigma > \\sigma_{m}\\,.
\\end{align}
```

Where:

  - ``\\sigma``: Point of the lattice, before the normalisation.
  - ``\\sigma_{m}``: The point of the lattice where the saturated count is largest.
  - ``\\sigma^{*}``: The peak of [`hac_log_det_peak`](@ref), and ``s^{*} = \\sigma^{*} / \\tilde{c}``.

# Arguments

  - `gh::VecNum`: ``g`` on the lattice (mutated past ``\\sigma_{m}``).
  - `rh::VecNum`: ``\\rho_{0}`` on the lattice (mutated past ``\\sigma_{m}``).
  - `xf::AbstractRange`: Lattice of ``\\ln \\sigma``.
  - `ct::Number`: The normalisation ``\\tilde{c}`` of the lattice.
  - `peak::Option{<:Tuple{<:Number, <:Number}}`: The peak of [`hac_log_det_peak`](@ref), or
    `nothing`.

# Returns

  - `cut::Option{<:NamedTuple}`: `nothing` where `peak` is `nothing`. Else the cut ``s^{*}`` as
    `sstar`, and as `gt` the true ``g`` up to ``\\sigma^{*}``, with ``g(\\sigma^{*})`` past it.

# Related

  - [`mahalanobis_level_bias`](@ref)
  - [`hac_log_det_peak`](@ref)
  - [`mahalanobis_transform`](@ref)
"""
function mahalanobis_cut!(::VecNum, ::VecNum, ::AbstractRange, ::Number, ::Nothing)
    return nothing
end
function mahalanobis_cut!(gh::VecNum, rh::VecNum, xf::AbstractRange, ct::Number,
                          peak::Tuple)
    sigma, gs = peak
    p = searchsortedlast(xf, log(sigma))
    gt = copy(gh)
    gt[(p + 1):end] .= gs
    pm = argmax(q -> exp(xf[q]) * rh[q], firstindex(xf):p)
    nm = exp(xf[pm]) * rh[pm]
    for q in (pm + 1):lastindex(xf)
        gh[q] = gh[pm] + nm * (xf[q] - xf[pm])
        rh[q] = nm / exp(xf[q])
    end

    return (; sstar = sigma / ct, gt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fills the differences of ``g`` and of ``\\rho_{0}`` between each node ``s`` and each point
``s + u`` of the level recursion, for one count of observations.

# Arguments

  - `dg::MatNum`: Output: ``g(s + u) - g(s)`` for each node and step (mutated).
  - `dr::MatNum`: Output: ``\\rho_{0}(s + u) - \\rho_{0}(s)`` for each node and step (mutated).
  - `grid::NamedTuple`: Grids of [`mahalanobis_bias_grid`](@ref).
  - `gh::VecNum`: ``g`` on the lattice.
  - `rh::VecNum`: ``\\rho_{0}`` on the lattice.
  - `shift::Tuple{<:Integer, <:Number}`: The shift ``m`` of the lattice and the normalisation
    ``\\tilde{c}``.

# Returns

  - `(dg, dr)::Tuple{<:MatNum, <:MatNum}`: The filled matrices.

# Related

  - [`mahalanobis_level_bias`](@ref)
"""
function mahalanobis_pair_differences!(dg::MatNum, dr::MatNum, grid::NamedTuple, gh::VecNum,
                                       rh::VecNum, shift::Tuple)
    m, ct = shift
    for a in eachindex(grid.fnodes)
        gn = stencil_dot(grid.fnodes[a], gh, m)
        rn = ct * stencil_dot(grid.fnodes[a], rh, m)
        for b in axes(dg, 2)
            st = grid.pairs[a, b][3]
            dg[a, b] = stencil_dot(st, gh, m) - gn
            dr[a, b] = ct * stencil_dot(st, rh, m) - rn
        end
    end

    return dg, dr
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Evaluates the Laplace transform ``e^{-D_{n-1}(s) / 2}`` of the last level on the output grid of
[`mahalanobis_bias_grid`](@ref).

Without a cut the grid is the output grid. With one, the grid moves by less than half a step so
that one point falls on ``s^{*}``. That point takes the weight one half of a closed trapezoid, and
every point past it takes zero, so the cut adds no error of the first order in the step. Below
the cut the transform reads the true ``g`` of the cut, not the continuation that the recursion
read.

# Arguments

  - `grid::NamedTuple`: Grids of [`mahalanobis_bias_grid`](@ref), with the cut of the count.
  - `gh::VecNum`: ``g`` on the lattice, which the transform reads without a cut.
  - `e::VecNum`: Integral of the removed part, from [`mahalanobis_q_integral!`](@ref).
  - `m::Integer`: Shift of the lattice.

# Returns

  - `(so, L)::Tuple{<:VecNum, <:VecNum}`: The grid and the transform on it.

# Related

  - [`mahalanobis_level_bias`](@ref)
  - [`regime_bias_factor`](@ref)
"""
function mahalanobis_transform(grid::NamedTuple, gh::VecNum, e::VecNum, m::Integer)
    if isnothing(grid.cut)
        return grid.so,
               [exp(-(stencil_dot(grid.fout[a], gh, m) - stencil_dot(grid.Qout[a], e)) / 2)
                for a in eachindex(grid.so)]
    end
    xf = grid.xf
    ls = log(grid.cut.sstar)
    k = clamp(round(Int, (ls - log(first(grid.so))) / grid.ho), 0, length(grid.so) - 1)
    so = grid.so .* exp(ls - log(grid.so[k + 1]))
    L = zero(so)
    for a in 1:(k + 1)
        st = lagrange_stencil(first(xf), step(xf), length(xf), log(so[a]))
        L[a] = exp(-(stencil_dot(st, grid.cut.gt, m) -
                     stencil_dot(mahalanobis_integral_stencil(so[a], grid.geo), e)) / 2)
    end
    L[k + 1] /= 2

    return so, L
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Runs one level of the recursion of [`mahalanobis_level_bias`](@ref) in place.

# Algorithm

 1. Fill the integral ``Q`` of the removed part with [`mahalanobis_q_integral!`](@ref).
 2. Read ``q`` and ``Q`` at each node: the Chebyshev nodes, then the grid past the junction.
 3. At each node ``s``, sum the trapezoid rule over the steps ``u`` of
    ``e^{-(D(s + u) - D(s)) / 2} (\\rho(s + u) - \\rho(s))``, with ``D = g - Q`` and
    ``\\rho = \\rho_{0} - q``, and take the new ``q`` as ``q - h \\sum / 2``.
 4. Write the new values into `v`, and set the grid values at the junction to the Chebyshev
    values.

# Arguments

  - `v::VecNum`: The Chebyshev values of ``q``, then ``q`` on the grid (mutated).
  - `e::VecNum`: Buffer of [`mahalanobis_q_integral!`](@ref) (mutated).
  - `bufs::Tuple{<:VecNum, <:VecNum, <:VecNum}`: Buffers for ``q`` and ``Q`` at the nodes and for
    the new ``q`` (mutated).
  - `dg::MatNum`: ``g(s + u) - g(s)`` for each node and step.
  - `dr::MatNum`: ``\\rho_{0}(s + u) - \\rho_{0}(s)`` for each node and step.
  - `grid::NamedTuple`: Grids of [`mahalanobis_bias_grid`](@ref).

# Returns

  - `v::VecNum`: The values of the next level.

# Related

  - [`mahalanobis_level_bias`](@ref)
"""
function mahalanobis_level_step!(v::VecNum, e::VecNum, bufs::Tuple, dg::MatNum, dr::MatNum,
                                 grid::NamedTuple)
    qn, Qn, new = bufs
    mahalanobis_q_integral!(e, v, grid)
    for l in 1:8
        qn[l] = v[l]
        Qn[l] = sum(grid.Qcheb[l][k] * v[k] for k in 1:8)
    end
    for a in 9:length(qn)
        qn[a] = v[a + 5]
        Qn[a] = e[a + 5]
    end
    for a in eachindex(qn)
        acc = zero(eltype(v))
        for b in axes(dg, 2)
            qs, Qs = grid.pairs[a, b]
            dD = dg[a, b] - (stencil_dot(Qs, e) - Qn[a])
            acc += exp(-dD / 2) * (dr[a, b] - (stencil_dot(qs, v) - qn[a]))
        end
        new[a] = qn[a] - grid.h / 2 * acc
    end
    for l in 1:8
        v[l] = new[l]
    end
    for a in 9:length(new)
        v[a + 5] = new[a]
    end
    for g in 1:5
        v[8 + g] = sum(grid.junction[g][l] * new[l] for l in 1:8)
    end
    return v
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fills the vector that the integral stencils of [`mahalanobis_bias_grid`](@ref) read, from the
values of the removed part ``q``.

The integral ``Q(s) = \\int_{0}^{s} q`` is the Chebyshev integral on the first five grid points,
and then a 4-point Gauss–Legendre rule on each step of the grid, with ``q`` interpolated on the
grid.

# Arguments

  - `e::VecNum`: Output: the Chebyshev values of ``q``, then ``Q`` on the grid, then
    ``s\\, q(s)`` at the end of the grid (mutated).
  - `v::VecNum`: The Chebyshev values of ``q``, then ``q`` on the grid.
  - `grid::NamedTuple`: Grids of [`mahalanobis_bias_grid`](@ref).

# Returns

  - `e::VecNum`: The filled vector.

# Related

  - [`mahalanobis_level_bias`](@ref)
"""
function mahalanobis_q_integral!(e::VecNum, v::VecNum, grid::NamedTuple)
    NG, h, sg = grid.NG, grid.h, grid.sg
    gl = gauss_legendre_rule(one(h))
    xlo = log(sg[1])
    for l in 1:8
        e[l] = v[l]
    end
    for i in 1:5
        e[8 + i] = sum(grid.Qjunction[i][l] * v[l] for l in 1:8)
    end
    qg = view(v, 9:(8 + NG))
    for i in 5:(NG - 1)
        x0 = xlo + (i - 1) * h
        e[9 + i] = e[8 + i] +
                   h / 2 * sum(gl) do (t, wt)
        x = x0 + h * (1 + t) / 2
        return wt * exp(x) * stencil_dot(lagrange_stencil(xlo, h, NG, x), qg)
    end
    end
    e[9 + NG] = qg[end] * sg[end]
    e[10 + NG] = zero(eltype(e))
    return e
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the factor of `method` for the inverse-Wishart law ``\\nu / \\chi^{2}_{\\nu - n + 1}``
whose mean is `b`.

At equal weights this is the exact law of ``R``, with ``\\nu = K``. On other weights it has the
right pole at ``K = n + 1``, so the ratio of the exact factor to it is smooth in ``\\lambda^{K}``,
and [`mahalanobis_bias_nodes`](@ref) interpolates that ratio.

# Arguments

  - `method::RegimeAdjustedMethod`: Regime adjustment method, which names the moment.
  - `b::Number`: Mean of ``R``, from [`mahalanobis_bias`](@ref).
  - `n::Integer`: Count of assets that contribute to the statistic.

# Returns

  - `factor::Number`: `b` for the root mean square,
    ``(\\nu / 2) (\\Gamma((\\nu - n) / 2) / \\Gamma((\\nu - n + 1) / 2))^{2}`` for the first moment,
    and ``(\\nu / 2) e^{-\\psi((\\nu - n + 1) / 2)}`` for the log, with
    ``\\nu = b (n + 1) / (b - 1)``.

# Related

  - [`mahalanobis_bias_nodes`](@ref)
  - [`mahalanobis_bias`](@ref)
"""
function inverse_wishart_bias(::RootMeanSquaredAdjusted, b::Number, ::Integer)
    return b
end
function inverse_wishart_bias(::FirstMomentRegimeAdjusted, b::Number, n::Integer)
    nu = b * (n + 1) / (b - 1)
    return nu / 2 * exp(2 * (SpecialFunctions.loggamma((nu - n) / 2) -
                             SpecialFunctions.loggamma((nu - n + 1) / 2)))
end
function inverse_wishart_bias(::LogRegimeAdjusted, b::Number, n::Integer)
    nu = b * (n + 1) / (b - 1)
    return nu / 2 * exp(-SpecialFunctions.digamma((nu - n + 1) / 2))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the count of observations past which the Mahalanobis bias factor no longer changes: the
count where ``\\lambda^{K}`` falls below the machine epsilon, and at least the gate `n + 4`.

# Arguments

  - `decay::Number`: Decay of the weights.
  - `n::Integer`: Count of assets that contribute to the statistic.

# Returns

  - `Ksat::Int`: The count.

# Related

  - [`mahalanobis_bias_nodes`](@ref)
  - [`regime_bias!`](@ref)
"""
function mahalanobis_bias_saturation(decay::Number, n::Integer)
    return max(ceil(Int, log(eps(one(decay))) / log(decay)), n + 4)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the first count of observations that the Mahalanobis statistic scores, where the
interpolation of [`mahalanobis_bias_nodes`](@ref) starts.

Without a HAC adjustment that is `n + 4`. With one, [`regime_bias_open`](@ref) also asks that the
effective count ``1 / \\operatorname{tr}(A^{2})`` pass `n + 1`, which the banded weight matrix
reaches later: at 12 assets, a half-life of 10 and two lags, at 54 observations. The search stops
at [`mahalanobis_bias_saturation`](@ref).

# Arguments

  - `decay::Number`: Decay of the weights.
  - `n::Integer`: Count of assets that contribute to the statistic.
  - `hac_lags::Option{<:Integer}`: Count of HAC lags, or `nothing`.

# Returns

  - `K1::Int`: The count.

# Related

  - [`mahalanobis_bias_nodes`](@ref)
  - [`regime_bias_open`](@ref)
"""
function mahalanobis_bias_start(::Number, n::Integer, ::Nothing)
    return n + 4
end
function mahalanobis_bias_start(decay::Number, n::Integer, hac_lags::Integer)
    K = n + 4
    Ksat = mahalanobis_bias_saturation(decay, n)
    while K < Ksat && !regime_bias_open(true, n, decay, K, hac_lags)
        K += 1
    end

    return K
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the nodes of the interpolation of the Mahalanobis bias factor over the count of
observations, at one count of assets.

The ratio of the factor of [`mahalanobis_level_bias`](@ref) to the factor of
[`inverse_wishart_bias`](@ref) is smooth in ``\\lambda^{K}``, from the gate ``K = n + 4`` to the
count where ``\\lambda^{K}`` falls below the machine epsilon. Sixteen Chebyshev–Lobatto points in
``\\lambda^{K}``, rounded to counts, interpolate it to within ``10^{-6}`` at every count, at
half-lives from 2 to 250 and from 5 to 30 assets. The nodes cost about 0.3 s at 12 assets and
0.65 s at 30 assets, once for each count of assets that a fit meets.

With a HAC adjustment the first count, [`mahalanobis_bias_start`](@ref), is where the gate opens,
and there the factor is steep: a HAC estimate has about half the effective observations of a plain
one. The ratio to the law of the plain fixed point, whose pole at ``K = n + 1`` is the pole of the
factor, then varies too fast near the gate for a polynomial in ``\\lambda^{K}`` at a long
half-life: sixteen points left ``7 \\times 10^{-3}`` at a half-life of 250. So the first 32
counts take the factor of the recursion itself, and sixteen points interpolate the rest. Over one to
five lags and 2 to 12 assets that is within ``10^{-9}`` at half-lives of 5, 10 and 40, and within
``5 \\times 10^{-6}`` at a half-life of 250. The nodes cost 0.3 s to 1.5 s at half-lives up to 40,
and up to 3.7 s at a half-life of 250, five lags and 12 assets.

# Arguments

  - `method::RegimeAdjustedMethod`: Regime adjustment method, which names the moment.
  - `decay::Number`: Decay of the weights.
  - `n::Integer`: Count of assets that contribute to the statistic, at least two.
  - `hac_lags::Option{<:Integer}`: Count of HAC lags, or `nothing`.

# Returns

  - `nodes::NamedTuple`: `sigma`, the nodes ``\\lambda^{K - n - 4}``; `bw`, their barycentric
    weights; `ratio`, the ratio at each node; `exact`, the factor at each count from `start` that
    the interpolation does not serve, empty without HAC; and `start`, the first count.

# Related

  - [`mahalanobis_regime_bias!`](@ref)
  - [`mahalanobis_level_bias`](@ref)
  - [`mahalanobis_bias_start`](@ref)
  - [`inverse_wishart_bias`](@ref)
"""
function mahalanobis_bias_nodes(method::RegimeAdjustedMethod, decay::Number, n::Integer,
                                hac_lags::Option{<:Integer} = nothing)
    start = mahalanobis_bias_start(decay, n, hac_lags)
    Ksat = max(mahalanobis_bias_saturation(decay, n), start)
    K1 = min(start + (isnothing(hac_lags) ? 0 : 32), Ksat)
    Ks = unique([if t <= decay^(Ksat - K1)
                     Ksat
                 else
                     clamp(round(Int, K1 + log(t) / log(decay)), K1, Ksat)
                 end
                 for t in (1 .- cos.(pi .* (0:15) ./ 15 .* one(decay))) ./ 2])
    f = mahalanobis_level_bias(method, decay, vcat(Ks, start:(K1 - 1)), n, hac_lags)
    sigma = decay .^ (Ks .- (n + 4))
    ratio = view(f, eachindex(Ks)) ./
            [inverse_wishart_bias(method, mahalanobis_bias(decay, K, n), n) for K in Ks]
    bw = [inv(prod(sigma[a] - sigma[b] for b in eachindex(sigma) if b != a;
                   init = one(decay))) for a in eachindex(sigma)]
    return (; sigma, bw, ratio, exact = f[(length(Ks) + 1):end], start)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the Mahalanobis bias factor that a regime method reads, and makes the nodes of its
interpolation the first time the state meets a count of assets.

# Arguments

  - `store::AbstractDict`: Nodes of the state, by count of assets (mutated where a count is new).
  - `method::RegimeAdjustedMethod`: Regime adjustment method, which names the moment.
  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of observations in the estimate, that [`regime_bias_open`](@ref) scores.
  - `n::Integer`: Count of assets that contribute to the statistic.
  - `hac_lags::Option{<:Integer}`: Count of HAC lags, or `nothing`.

# Returns

  - `factor::Number`: The factor of the recursion where the nodes hold it. Else
    [`inverse_wishart_bias`](@ref) at `K`, times the interpolated ratio. A count past
    [`mahalanobis_bias_saturation`](@ref) reads the saturation count.

# Related

  - [`mahalanobis_bias_nodes`](@ref)
  - [`regime_target_statistic`](@ref)
  - [`MahalanobisTarget`](@ref)
"""
function mahalanobis_regime_bias!(store::AbstractDict, method::RegimeAdjustedMethod,
                                  decay::Number, K::Integer, n::Integer,
                                  hac_lags::Option{<:Integer} = nothing)
    nodes = get!(() -> mahalanobis_bias_nodes(method, decay, n, hac_lags), store, n)
    i = K - nodes.start + 1
    if i in eachindex(nodes.exact)
        return nodes.exact[i]
    end
    K = min(K, mahalanobis_bias_saturation(decay, n))
    x = decay^(K - n - 4)
    a = findfirst(==(x), nodes.sigma)
    r = if !isnothing(a)
        nodes.ratio[a]
    else
        sum(nodes.bw[k] * nodes.ratio[k] / (x - nodes.sigma[k])
            for k in eachindex(nodes.bw)) /
        sum(nodes.bw[k] / (x - nodes.sigma[k]) for k in eachindex(nodes.bw))
    end
    return inverse_wishart_bias(method, mahalanobis_bias(decay, K, n), n) * r
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes the empty table of bias factors of a target that reads one estimated variance per
direction.

# Arguments

  - `::RegimeAdjustedTarget`: Regime-adjustment target.
  - `::Number`: Ignored decay.
  - `::Type{T}`: Element type of the state.

# Returns

  - `bias::Vector{T}`: An empty vector.

# Related

  - [`regime_bias!`](@ref)
"""
function regime_bias_store(::RegimeAdjustedTarget, ::Number, ::Type{T}) where {T}
    return T[]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes the empty table of the diagonal target: the moments of one term of
[`RegimeTermMoments`](@ref), a triple for each count of observations.

# Arguments

  - `::DiagonalTarget`: Diagonal regime-adjustment target.
  - `::Number`: Ignored decay.
  - `::Type{T}`: Element type of the state.

# Returns

  - `bias::Vector{NTuple{3, T}}`: An empty vector.

# Related

  - [`regime_bias_state`](@ref)
  - [`RegimeTermMoments`](@ref)
"""
function regime_bias_store(::DiagonalTarget, ::Number, ::Type{T}) where {T}
    return NTuple{3, T}[]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes the empty store of the Mahalanobis bias factor: its interpolation nodes, keyed by the count
of assets, and the two tables of the variance factor of the separate correlation path.

# Arguments

  - `::MahalanobisTarget`: Mahalanobis regime-adjustment target.
  - `decay::Number`: Decay of the weights, whose type is the type of the nodes.
  - `::Type`: Ignored element type of the state.

# Returns

  - `bias::NamedTuple`: `nodes`, an empty dictionary from a count of assets to the nodes of
    [`mahalanobis_bias_nodes`](@ref); and `kappa` and `spread`, the empty tables of
    [`variance_noise_bias!`](@ref), which only the separate path fills.

# Related

  - [`mahalanobis_regime_bias!`](@ref)
  - [`variance_noise_bias!`](@ref)
"""
function regime_bias_store(::MahalanobisTarget, decay::Number, ::Type)
    return (;
            nodes = Dict{Int,
                         NamedTuple{(:sigma, :bw, :ratio, :exact, :start),
                                    Tuple{Vector{typeof(decay)}, Vector{typeof(decay)},
                                          Vector{typeof(decay)}, Vector{typeof(decay)},
                                          Int}}}(), kappa = typeof(decay)[],
            spread = typeof(decay)[])
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the factor by which the noise of the variance at `decay` moves the bias of the squared
Mahalanobis distance on the separate correlation path, and grows the tables of the state where the
count is past their end.

On the separate path the block is ``\\hat{C} = D \\hat{R} D``, with the volatilities ``D`` from the
variance at `decay` and the correlation ``\\hat{R}`` from the recursion at `cor_decay`. The factor
of [`mahalanobis_regime_bias!`](@ref) at `cor_decay` is the bias of an estimate whose diagonal
carries the noise of `cor_decay`. At ``R = I`` the statistic is
``\\sum_{i} (\\hat{R}^{-1})_{ii} / \\hat{V}_{i}``, and at equal weights ``\\hat{R}`` is independent of
the variances, so the diagonal enters the mean through ``\\mathbb{E}[1/Q]`` alone: ``\\kappa`` is the
ratio of that moment at the two decays, from the exact tables of [`regime_bias_table`](@ref), plain
or HAC.

The noisier variance also spreads the statistic, which lowers the root and the log that
`FirstMomentRegimeAdjusted` and `LogRegimeAdjusted` read. The spread is the sum over the assets of
the noise of each ``1 / \\hat{V}_{i}``, so it averages over the assets. At one asset the block is
the variance alone, and the exact factor is the method's own table at `decay`; the ratio ``\\rho``
of the method's moment to the mean's carries that step. For ``n`` assets, the second-order change
of the root and the log of a quadratic form whose ``n`` terms carry independent noise is
``3 / (n + 2)`` of the change at one asset, exactly on a sphere, so the power is ``3 / (n + 2)``. It
is one at one asset, and the factor tends to ``\\kappa`` alone as the count of assets grows. For
`RootMeanSquaredAdjusted`, ``\\rho = 1``.

# Mathematical definition

```math
\\begin{align}
f_{K,n} &= \\kappa_{K} \\left(\\frac{\\rho_{\\lambda,K}}{\\rho_{\\lambda_{c},K}}\\right)^{3 / (n + 2)}\\,, \\qquad
\\kappa_{K} = \\frac{\\mathbb{E}_{\\lambda}\\left[Q_{K}^{-1}\\right]}{\\mathbb{E}_{\\lambda_{c}}\\left[Q_{K}^{-1}\\right]}\\,, \\qquad
\\rho_{\\lambda,K} = \\frac{m_{\\lambda}(Q_{K})}{\\mathbb{E}_{\\lambda}\\left[Q_{K}^{-1}\\right]}\\,.
\\end{align}
```

Where:

  - ``Q_{K}``: An estimated variance of ``K`` observations over the true one.
  - ``\\lambda``, ``\\lambda_{c}``: `decay` and `cor_decay`.
  - ``m_{\\lambda}``: The moment of ``Q^{-1}`` that the regime method reads, its table at ``\\lambda``.
  - ``n``: Count of assets that contribute to the statistic.

At 12 assets, a half-life of 10 and a correlation half-life of 20, ``\\kappa = 1.0347``, and the
power of the ratio is 0.9981 for `FirstMomentRegimeAdjusted` and 0.9963 for `LogRegimeAdjusted`;
at two HAC lags, 1.0694, 0.9964 and 0.9928. Against the true factor of the separate path, measured
in the steady state over 64 seeds, the three methods then read 1.0018, 1.0018 and 1.0017 at
``R = I``, where the spread had put them at 1.0018, 0.9999 and 0.9980. At 4 assets and half-lives
of 5 and 40 the power leaves 0.02 % and 0.06 % of a gap of 1.5 % and 3.0 %. It holds the noise of
the variances with the correlation fixed. The noise of ``\\hat{R}`` amplifies the spread, the data
that the two estimates share reduces it, and the division of each correlation row by the volatility
after its own update adds to it. Without HAC the three cancel at ``R = I``; at two lags the power
carries about three quarters of the spread. The spread grows with the correlation of the assets,
which the power does not read: at an equicorrelation of 0.8 it carries 42 % of it, so the methods read
0.9969, 0.9943 and 0.9917. The mean keeps parts of the next order, within 0.3 % without HAC and
up to 3.1 % high at two lags, where the division of each correlation row by the volatility after
its own update dominates.

# Arguments

  - `store::NamedTuple`: Store of the state, whose tables `kappa` and `spread` grow (mutated).
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `K::Integer`: Count of observations in the estimate.
  - `n::Integer`: Count of assets that contribute to the statistic.

# Returns

  - `factor::Number`: The factor at `K`, one where the estimator runs one decay.

# Related

  - [`regime_bias_store`](@ref)
  - [`regime_bias_table`](@ref)
  - [`mahalanobis_regime_bias!`](@ref)
  - [`regime_target_statistic`](@ref)
"""
function variance_noise_bias!(store::NamedTuple, ce::RegimeAdjustedExpWeightedCovariance,
                              K::Integer, n::Integer)
    if !has_separate_cor_decay(ce)
        return one(ce.decay)
    end
    (; kappa, spread) = store
    Ksat = ceil(Int, log(eps(eltype(kappa))) / log(max(ce.decay, ce.cor_decay)))
    if K > length(kappa) && length(kappa) < Ksat
        len = min(max(2 * K, 64), Ksat)
        (tv, mv), (tc, mc) = map((ce.decay, ce.cor_decay)) do decay
            map((RootMeanSquaredAdjusted(), ce.regime_method)) do method
                if isnothing(ce.hac_lags)
                    regime_bias_table(method, decay, len)
                else
                    regime_bias_table(method, decay, len, ce.hac_lags)
                end
            end
        end
        resize!(kappa, len)
        resize!(spread, len)
        kappa .= tv ./ tc
        spread .= (mv ./ tv) ./ (mc ./ tc)
    end
    k = min(K, length(kappa))

    return kappa[k] * spread[k]^(3 // (n + 2))
end
