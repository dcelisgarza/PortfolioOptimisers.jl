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
    integral, the pair stencils, the output grid `so` of [`regime_bias_table`](@ref), and the
    lattice `xf` on which `g` and ``\\rho_{0}`` are tabulated.

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
    return (; h, sc, sg, NG, nodes, su, xf, pairs, ho,
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

# Mathematical definition

```math
\\begin{align}
D_{0}(s) &= \\sum_{j} \\ln(1 + s w_{j})\\,, \\qquad \\rho_{k} = D_{k}'\\,, \\qquad D_{k}(0) = 0\\,, \\\\
\\rho_{k+1}(s) &= \\rho_{k}(s) + \\frac{1}{2} \\int_{0}^{\\infty}
e^{-(D_{k}(s + u) - D_{k}(s)) / 2}\\, \\bigl(\\rho_{k}(s + u) - \\rho_{k}(s)\\bigr)\\,
\\frac{\\mathrm{d}u}{u}\\,, \\\\
\\mathbb{E}\\bigl[e^{-t S}\\bigr] &\\approx e^{-D_{n-1}(2 t) / 2}\\,.
\\end{align}
```

Where:

  - ``w_{j}``: Normalised weight of the observation ``j`` steps before the newest.
  - ``D_{k}``: Log-determinant of the Laplace transform after ``k`` directions are removed.

# Algorithm

 1. Tabulate ``\\sum_{j} \\ln(1 + s \\lambda^{j})`` and its derivative on the lattice, adding one
    observation at a time, and keep a copy at each count of `Ks`.
 2. For each count, round the normalisation ``c = (1 - \\lambda) / (1 - \\lambda^{K})`` to the
    lattice, ``\\tilde{c} = e^{m h_{f}}``, so the weights ``\\tilde{c} \\lambda^{j}`` read the
    lattice without interpolation.
 3. Run the recursion for ``n - 1`` levels on the removed part ``q_{k} = \\rho_{0} - \\rho_{k}``,
    with the trapezoid rule on ``\\ln u``. The result at ``s`` reads ``q_{k}`` only on
    ``[s, \\infty)``, so an error moves only toward smaller ``s``.
 4. Read the factor of the method off the transform with [`regime_bias_factor`](@ref), and scale
    it by ``\\tilde{c} / c``.

# Arguments

  - `method::RegimeAdjustedMethod`: Regime adjustment method, which names the moment.
  - `decay::Number`: Decay of the weights.
  - `Ks::AbstractVector{<:Integer}`: Counts of observations, each above `n + 1`.
  - `n::Integer`: Count of assets that contribute to the statistic, at least two.

# Returns

  - `factors::VecNum`: The factor at each count of `Ks`.

# Related

  - [`mahalanobis_bias_nodes`](@ref)
  - [`mahalanobis_bias_grid`](@ref)
  - [`regime_bias_factor`](@ref)
  - [`MahalanobisTarget`](@ref)
"""
function mahalanobis_level_bias(method::RegimeAdjustedMethod, decay::Number,
                                Ks::AbstractVector{<:Integer}, n::Integer)
    grid = mahalanobis_bias_grid(decay)
    xf = grid.xf
    hf = step(xf)
    ef = exp.(xf)
    T = eltype(ef)
    ghat = zero(ef)
    rhat = zero(ef)
    G = Matrix{T}(undef, length(ef), length(Ks))
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
    NN = length(grid.nodes)
    dg = Matrix{T}(undef, NN, length(grid.su))
    dr = similar(dg)
    v = zeros(T, 8 + grid.NG)
    e = zeros(T, 10 + grid.NG)
    bufs = (zeros(T, NN), zeros(T, NN), zeros(T, NN))
    return map(eachindex(Ks)) do i
        c = (one(decay) - decay) / (one(decay) - decay^Ks[i])
        m = round(Int, log(c) / hf)
        ct = exp(m * hf)
        gh = view(G, :, i)
        rh = view(Rh, :, i)
        for a in 1:NN
            gn = stencil_dot(grid.fnodes[a], gh, m)
            rn = ct * stencil_dot(grid.fnodes[a], rh, m)
            for b in axes(dg, 2)
                st = grid.pairs[a, b][3]
                dg[a, b] = stencil_dot(st, gh, m) - gn
                dr[a, b] = ct * stencil_dot(st, rh, m) - rn
            end
        end
        fill!(v, zero(T))
        for _ in 1:(n - 1)
            mahalanobis_level_step!(v, e, bufs, dg, dr, grid)
        end
        mahalanobis_q_integral!(e, v, grid)
        L = [exp(-(stencil_dot(grid.fout[a], gh, m) - stencil_dot(grid.Qout[a], e)) / 2)
             for a in eachindex(grid.so)]
        return regime_bias_factor(method, grid.so, L, one(c), grid.ho) * ct / c
    end
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
    o = one(h)
    a = sqrt(3 * o / 7 - 2 * o / 7 * sqrt(6 * o / 5))
    b = sqrt(3 * o / 7 + 2 * o / 7 * sqrt(6 * o / 5))
    gl = ((-b, (18 - sqrt(30 * o)) / 36), (-a, (18 + sqrt(30 * o)) / 36),
          (a, (18 + sqrt(30 * o)) / 36), (b, (18 - sqrt(30 * o)) / 36))
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

Computes the nodes of the interpolation of the Mahalanobis bias factor over the count of
observations, at one count of assets.

The ratio of the factor of [`mahalanobis_level_bias`](@ref) to the factor of
[`inverse_wishart_bias`](@ref) is smooth in ``\\lambda^{K}``, from the gate ``K = n + 4`` to the
count where ``\\lambda^{K}`` falls below the machine epsilon. Sixteen Chebyshev–Lobatto points in
``\\lambda^{K}``, rounded to counts, interpolate it to within ``10^{-6}`` at every count, at
half-lives from 2 to 250 and from 5 to 30 assets. The nodes cost about 0.3 s at 12 assets and
0.65 s at 30 assets, once for each count of assets that a fit meets.

# Arguments

  - `method::RegimeAdjustedMethod`: Regime adjustment method, which names the moment.
  - `decay::Number`: Decay of the weights.
  - `n::Integer`: Count of assets that contribute to the statistic, at least two.

# Returns

  - `nodes::NamedTuple`: `sigma`, the nodes ``\\lambda^{K - n - 4}``; `bw`, their barycentric
    weights; and `ratio`, the ratio at each node.

# Related

  - [`mahalanobis_regime_bias!`](@ref)
  - [`mahalanobis_level_bias`](@ref)
  - [`inverse_wishart_bias`](@ref)
"""
function mahalanobis_bias_nodes(method::RegimeAdjustedMethod, decay::Number, n::Integer)
    K1 = n + 4
    Ksat = mahalanobis_bias_saturation(decay, n)
    Ks = unique([if t <= decay^(Ksat - K1)
                     Ksat
                 else
                     clamp(round(Int, K1 + log(t) / log(decay)), K1, Ksat)
                 end
                 for t in (1 .- cos.(pi .* (0:15) ./ 15 .* one(decay))) ./ 2])
    f = mahalanobis_level_bias(method, decay, Ks, n)
    sigma = decay .^ (Ks .- K1)
    ratio = f ./
            [inverse_wishart_bias(method, mahalanobis_bias(decay, K, n), n) for K in Ks]
    bw = [inv(prod(sigma[a] - sigma[b] for b in eachindex(sigma) if b != a;
                   init = one(decay))) for a in eachindex(sigma)]
    return (; sigma, bw, ratio)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the Mahalanobis bias factor that a regime method reads, and makes the nodes of its
interpolation the first time the state meets a count of assets.

# Arguments

  - `store::AbstractDict`: Nodes of the state, by count of assets (mutated where a count is new).
  - `method::RegimeAdjustedMethod`: Regime adjustment method, which names the moment.
  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of observations in the estimate, above `n + 3`.
  - `n::Integer`: Count of assets that contribute to the statistic.

# Returns

  - `factor::Number`: [`inverse_wishart_bias`](@ref) at `K`, times the interpolated ratio. A
    count past [`mahalanobis_bias_saturation`](@ref) reads the saturation count.

# Related

  - [`mahalanobis_bias_nodes`](@ref)
  - [`regime_target_statistic`](@ref)
  - [`MahalanobisTarget`](@ref)
"""
function mahalanobis_regime_bias!(store::AbstractDict, method::RegimeAdjustedMethod,
                                  decay::Number, K::Integer, n::Integer)
    nodes = get!(() -> mahalanobis_bias_nodes(method, decay, n), store, n)
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
of assets, and the table of the variance factor of the separate correlation path.

# Arguments

  - `::MahalanobisTarget`: Mahalanobis regime-adjustment target.
  - `decay::Number`: Decay of the weights, whose type is the type of the nodes.
  - `::Type`: Ignored element type of the state.

# Returns

  - `bias::NamedTuple`: `nodes`, an empty dictionary from a count of assets to the nodes of
    [`mahalanobis_bias_nodes`](@ref); and `kappa`, the empty table of
    [`variance_noise_bias!`](@ref), which only the separate path fills.

# Related

  - [`mahalanobis_regime_bias!`](@ref)
  - [`variance_noise_bias!`](@ref)
"""
function regime_bias_store(::MahalanobisTarget, decay::Number, ::Type)
    return (;
            nodes = Dict{Int,
                         NamedTuple{(:sigma, :bw, :ratio),
                                    NTuple{3, Vector{typeof(decay)}}}}(),
            kappa = typeof(decay)[])
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the factor by which the noise of the variance at `decay` moves the bias of the squared
Mahalanobis distance on the separate correlation path, and grows the table of the state where the
count is past its end.

On the separate path the block is ``\\hat{C} = D \\hat{R} D``, with the volatilities ``D`` from the
variance at `decay` and the correlation ``\\hat{R}`` from the recursion at `cor_decay`. The factor
of [`mahalanobis_regime_bias!`](@ref) at `cor_decay` is the bias of an estimate whose diagonal
carries the noise of `cor_decay`. At ``R = I`` the statistic is
``\\sum_{i} (\\hat{R}^{-1})_{ii} / \\hat{V}_{i}``, and at equal weights ``\\hat{R}`` is independent of
the variances, so the diagonal enters through ``\\mathbb{E}[1/Q]`` alone: the factor is the ratio
of that moment at the two decays, from the exact tables of [`regime_bias_table`](@ref), plain or
HAC. The one factor serves the three regime methods: the noise of the variances averages over the
assets, so it moves the statistic through its mean.

# Mathematical definition

```math
\\kappa_{K} = \\frac{\\mathbb{E}_{\\lambda}\\left[Q_{K}^{-1}\\right]}{\\mathbb{E}_{\\lambda_{c}}\\left[Q_{K}^{-1}\\right]}\\,.
```

Where:

  - ``Q_{K}``: An estimated variance of ``K`` observations over the true one.
  - ``\\lambda``, ``\\lambda_{c}``: `decay` and `cor_decay`.

At 12 assets, a half-life of 10 and a correlation half-life of 20, ``\\kappa = 1.0347``, and
1.0694 at two HAC lags. Against the true factor of the separate path, measured in the steady state
over 64 seeds, the product with the factor at `cor_decay` is within 0.3 % without HAC, at a
correlation of zero, at the correlation of a random factor model and at an equicorrelation of 0.8;
at two HAC lags it is 1.7 % to 2.3 % below. The rest has three parts of the next order: the
Schur complement that the inverse reads down-weights the newest rows, which the variance at
`decay` reads most; the correlation of the assets; and the division of each row of the correlation
recursion by the volatility after its own update.

# Arguments

  - `store::NamedTuple`: Store of the state, whose table `kappa` grows (mutated).
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `K::Integer`: Count of observations in the estimate.

# Returns

  - `kappa::Number`: The factor at `K`, one where the estimator runs one decay.

# Related

  - [`regime_bias_store`](@ref)
  - [`regime_bias_table`](@ref)
  - [`mahalanobis_regime_bias!`](@ref)
  - [`regime_target_statistic`](@ref)
"""
function variance_noise_bias!(store::NamedTuple, ce::RegimeAdjustedExpWeightedCovariance,
                              K::Integer)
    if !has_separate_cor_decay(ce)
        return one(ce.decay)
    end
    kappa = store.kappa
    Ksat = ceil(Int, log(eps(eltype(kappa))) / log(max(ce.decay, ce.cor_decay)))
    if K > length(kappa) && length(kappa) < Ksat
        n = min(max(2 * K, 64), Ksat)
        tv, tc = map((ce.decay, ce.cor_decay)) do decay
            if isnothing(ce.hac_lags)
                regime_bias_table(RootMeanSquaredAdjusted(), decay, n)
            else
                regime_bias_table(RootMeanSquaredAdjusted(), decay, n, ce.hac_lags)
            end
        end
        resize!(kappa, n)
        kappa .= tv ./ tc
    end

    return kappa[min(K, length(kappa))]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns one, the variance factor of a state that takes no bias correction.

# Arguments

  - `::Nothing`: The store of a state whose estimator has `debias = false` or no regime method.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `::Integer`: Ignored count of observations.

# Returns

  - `one(ce.decay)`.

# Related

  - [`variance_noise_bias!`](@ref)
"""
function variance_noise_bias!(::Nothing, ce::RegimeAdjustedExpWeightedCovariance, ::Integer)
    return one(ce.decay)
end
