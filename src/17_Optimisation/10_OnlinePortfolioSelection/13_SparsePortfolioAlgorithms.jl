"""
$(DocStringExtensions.TYPEDEF)

Puts the whole budget on the asset with the largest Price Relative Forecast, and splits it in equal parts between tied assets.

This is the optimum of the programme that the paper of the short-term sparse portfolio states, its equation (12). The optimum does not depend on ``\\lambda``, so the algorithm takes no parameter. Where the programme has no minimum, the algorithm returns the same split.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{b}^\\star &= \\underset{\\boldsymbol{b}}{\\arg\\min} \\; \\langle \\boldsymbol{b}, \\boldsymbol{\\phi} \\rangle + \\lambda \\lVert \\boldsymbol{b} \\rVert_1 \\quad \\text{s.t.} \\quad \\boldsymbol{1}^\\intercal \\boldsymbol{b} = 1\\,, \\\\
\\boldsymbol{b}^\\star &= \\frac{1}{\\lvert \\mathcal{K} \\rvert} \\sum_{k \\in \\mathcal{K}} \\boldsymbol{e}_k\\,, \\quad \\mathcal{K} = \\underset{i}{\\arg\\min} \\; \\phi_i\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{b}^\\star``: Optimum of the programme.
  - $(math_dict[:b_sspo])
  - $(math_dict[:phi_sspo])
  - $(math_dict[:lambda_l1])
  - ``\\mathcal{K}``: Set of the assets with the smallest entry of ``\\boldsymbol{\\phi}``.
  - ``\\boldsymbol{e}_k``: Unit vector of asset ``k``.

Over ``\\boldsymbol{b} \\geq \\boldsymbol{0}`` the penalty is the constant ``\\lambda``, so the optimum there is a vertex ``\\boldsymbol{e}_k`` with ``k \\in \\mathcal{K}``. A short of ``s`` on asset ``j`` that funds more of asset ``k`` changes the objective by ``s (\\phi_k - \\phi_j + 2 \\lambda)``. That change is not negative while ``\\max \\boldsymbol{\\phi} - \\min \\boldsymbol{\\phi} \\leq 2 \\lambda``, so no short improves on the vertex. Past that bound the programme has no minimum, and the objective falls without end along that short from the vertex. Tied assets share an optimal face, and the equal split is its centre.

# Examples

```jldoctest
julia> L1Optimum()
L1Optimum()
```

# Related

  - [`ShortTermSparsePortfolio`](@ref)
  - [`HuberOptimum`](@ref)
  - [`AlternatingDirectionMethod`](@ref)
  - [`largest_forecast_split`](@ref)

# References

  - $(ref_dict[:lai2018sspo])
"""
struct L1Optimum <: AbstractSparsePortfolioAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Finds the fixed point of the alternating direction iteration of the short-term sparse portfolio in closed form.

The paper's iteration converges to this point, so it is the default algorithm of [`ShortTermSparsePortfolio`](@ref). Where the programme has no minimum, the algorithm returns the split of [`L1Optimum`](@ref).

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{b}, \\boldsymbol{g}}{\\min} &\\; \\langle \\boldsymbol{b}, \\boldsymbol{\\phi} \\rangle + \\lambda \\lVert \\boldsymbol{g} \\rVert_1 + \\frac{a}{2} \\lVert \\boldsymbol{b} - \\boldsymbol{g} \\rVert_2^2 \\quad \\text{s.t.} \\quad \\boldsymbol{1}^\\intercal \\boldsymbol{b} = 1\\,, \\\\
h(b) &= \\underset{g}{\\min} \\; \\lambda \\lvert g \\rvert + \\frac{a}{2} (b - g)^2 = \\begin{cases} \\tfrac{a}{2} b^2 & \\lvert b \\rvert \\leq \\gamma\\,, \\\\ \\lambda \\lvert b \\rvert - \\tfrac{\\lambda \\gamma}{2} & \\text{otherwise}\\,, \\end{cases} \\\\
b_i(\\nu) &= \\mathrm{clamp}\\left( -\\frac{\\phi_i + \\nu}{a}, -\\gamma, \\gamma \\right)\\,, \\quad \\nu \\in \\left[ -\\lambda - \\min \\boldsymbol{\\phi}, \\lambda - \\max \\boldsymbol{\\phi} \\right]\\,.
\\end{align}
```

Where:

  - $(math_dict[:b_sspo])
  - $(math_dict[:g_aux])
  - $(math_dict[:phi_sspo])
  - $(math_dict[:lambda_l1])
  - $(math_dict[:gamma_st])
  - $(math_dict[:a_huber])
  - ``h``: Huber penalty of one coordinate, the minimum of the programme over the auxiliary coordinate ``g``.
  - $(math_dict[:nu_budget])
  - ``b_i(\\nu)``: Coordinate of asset ``i`` at the multiplier ``\\nu``, where ``\\lvert \\phi_i + \\nu \\rvert < \\lambda``.
  - $(math_dict[:N])

The programme is equation (20) of the paper without its penalty on the budget residual, which is zero on the budget. The minimum over ``\\boldsymbol{g}`` is the soft threshold of ``\\boldsymbol{b}`` at ``\\gamma``, and it leaves the penalty ``h`` on each coordinate. The paper's equation (19) is ``h / \\lambda``.

The derivative ``h'(b) = \\mathrm{clamp}(a b, -\\lambda, \\lambda)`` is bounded, so a multiplier ``\\nu`` exists only in the interval above. An asset with ``\\lvert \\phi_i + \\nu \\rvert < \\lambda`` holds ``b_i(\\nu)``. An asset with ``\\lvert \\phi_i + \\nu \\rvert = \\lambda`` can hold any amount past ``\\pm \\gamma``, with the sign of ``-(\\phi_i + \\nu)``, because ``h`` is linear there. The sum of the ``b_i(\\nu)`` falls piecewise linearly in ``\\nu``, with knots at ``-\\phi_i \\pm \\lambda``. The multiplier is the root of that sum at one, or the end of the interval where the sum cannot reach one. At the lower end the assets with the smallest ``\\phi_i`` take the remainder of the budget, and at the upper end the assets with the largest ``\\phi_i`` take it.

When ``N \\gamma \\leq 1`` the sum is at most one everywhere. An asset ``k`` with the smallest ``\\phi_k`` then takes ``1 - \\sum_{i \\neq k} b_i``, and every other asset holds at most ``\\gamma``. When ``\\max \\boldsymbol{\\phi} - \\min \\boldsymbol{\\phi} > 2 \\lambda`` the interval is empty, and the programme has no minimum.

# Algorithm

 1. Compute the ends of the interval of the multiplier, `lo = -lambda - minimum(phi)` and `hi = lambda - maximum(phi)`.
 2. When `lo > hi`, the programme has no minimum. Return the split of [`largest_forecast_split`](@ref).
 3. Find the multiplier `nu` in `[lo, hi]` with [`huber_multiplier`](@ref).
 4. Compute the coordinates `b` at `nu` with [`huber_coordinates`](@ref).
 5. Add the remainder of the budget to `b` with [`huber_remainder!`](@ref), in equal parts between tied assets, and return `b`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HuberOptimum(; lambda::Real = 0.5, gamma::Real = 0.01) -> HuberOptimum

Keywords correspond to the struct's fields, and the defaults are the paper's.

## Validation

  - `lambda > 0`, `gamma > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> HuberOptimum()
HuberOptimum
  lambda ┼ Float64: 0.5
   gamma ┴ Float64: 0.01
```

# Related

  - [`ShortTermSparsePortfolio`](@ref)
  - [`L1Optimum`](@ref)
  - [`AlternatingDirectionMethod`](@ref)
  - [`huber_multiplier`](@ref)

# References

  - $(ref_dict[:lai2018sspo])
"""
struct HuberOptimum{T1 <: Real, T2 <: Real} <: AbstractSparsePortfolioAlgorithm
    """
    $(field_dict[:lambda_sspo])
    """
    lambda::T1
    """
    $(field_dict[:gamma_sspo])
    """
    gamma::T2
    function HuberOptimum(lambda::Real, gamma::Real)
        @argcheck(lambda > zero(lambda), DomainError(lambda, "lambda must be positive"))
        @argcheck(gamma > zero(gamma), DomainError(gamma, "gamma must be positive"))
        return new{typeof(lambda), typeof(gamma)}(lambda, gamma)
    end
end
function HuberOptimum(; lambda::Real = 0.5, gamma::Real = 0.01)::HuberOptimum
    return HuberOptimum(lambda, gamma)
end
"""
$(DocStringExtensions.TYPEDEF)

Runs the alternating direction iteration of the short-term sparse portfolio, and stops where the paper stops it.

Take this algorithm to reproduce the paper's loop and its stop. Its fixed point is the point that [`HuberOptimum`](@ref) finds in closed form, but the stop comes long before the iterate reaches that point. The stop reads the budget residual, which changes sign as the dual variable adapts, and the iteration meets `tol = 1e-4` at one of those sign changes. The scaled projection of the stopped iterate is usually the fixed point's projection, but it can hold other assets. A tolerance that the sign changes never meet runs all `iters` iterations.

The tests measure these numbers from the uniform seed. On `0.02 .* randn(StableRNG(11), 40, 4)`, over the 36 windows of five rows, the stop comes after 359 to 4619 iterations and 2 to 27 sign changes. At the stop the iterate is up to about 0.67 from the fixed point in one coordinate, and it comes within ``10^{-6}`` of the fixed point after 29461 to 104386 iterations. On `0.02 .* randn(StableRNG(1), 60, 4)`, rows 2 to 6 give two assets almost equal forecasts. At `zeta = 500` the stopped iterate projects to about `[0.46, 0, 0, 0.54]`, and the fixed point projects to `[0, 0, 0, 1]`.

# Mathematical definition

```math
\\begin{align}
L(\\boldsymbol{b}, \\boldsymbol{g}, \\rho) &= \\langle \\boldsymbol{b}, \\boldsymbol{\\phi} \\rangle + \\frac{a}{2} \\lVert \\boldsymbol{b} - \\boldsymbol{g} \\rVert_2^2 + \\lambda \\lVert \\boldsymbol{g} \\rVert_1 + \\frac{\\eta}{2} (\\boldsymbol{1}^\\intercal \\boldsymbol{b} - 1)^2 + \\rho (\\boldsymbol{1}^\\intercal \\boldsymbol{b} - 1)\\,, \\\\
\\boldsymbol{b}^{(o+1)} &= \\underset{\\boldsymbol{b}}{\\arg\\min} \\; L(\\boldsymbol{b}, \\boldsymbol{g}^{(o)}, \\rho^{(o)}) = \\left( a \\mathbf{I} + \\eta \\boldsymbol{1} \\boldsymbol{1}^\\intercal \\right)^{-1} \\left( a \\boldsymbol{g}^{(o)} + (\\eta - \\rho^{(o)}) \\boldsymbol{1} - \\boldsymbol{\\phi} \\right)\\,, \\\\
\\boldsymbol{g}^{(o+1)} &= \\underset{\\boldsymbol{g}}{\\arg\\min} \\; L(\\boldsymbol{b}^{(o+1)}, \\boldsymbol{g}, \\rho^{(o)}) = \\operatorname{sign}(\\boldsymbol{b}^{(o+1)}) \\odot \\max(\\lvert \\boldsymbol{b}^{(o+1)} \\rvert - \\gamma, 0)\\,, \\\\
\\rho^{(o+1)} &= \\rho^{(o)} + \\eta (\\boldsymbol{1}^\\intercal \\boldsymbol{b}^{(o+1)} - 1)\\,, \\\\
\\left( a \\mathbf{I} + \\eta \\boldsymbol{1} \\boldsymbol{1}^\\intercal \\right)^{-1} \\boldsymbol{v} &= \\frac{\\boldsymbol{v}}{a} - \\frac{\\eta \\, \\boldsymbol{1}^\\intercal \\boldsymbol{v}}{a (a + \\eta N)} \\boldsymbol{1}\\,.
\\end{align}
```

Where:

  - ``L``: Augmented Lagrangian of the iteration.
  - $(math_dict[:b_sspo])
  - $(math_dict[:g_aux])
  - ``\\rho``: Dual variable of the budget constraint.
  - ``o``: Iteration index, counted from one.
  - $(math_dict[:phi_sspo])
  - $(math_dict[:lambda_l1])
  - $(math_dict[:gamma_st])
  - $(math_dict[:a_huber])
  - ``\\eta``: Weight of the penalty on the budget residual, and the step of the dual variable.
  - ``\\mathbf{I}``: ``N \\times N`` identity matrix.
  - ``\\boldsymbol{v}``: Any ``N \\times 1`` vector.
  - $(math_dict[:N])

``L`` is the paper's equation (13), and the three updates are its equations (29), (32) and (27). The last line is the Sherman-Morrison identity. A fixed point of the updates has a zero budget residual, and it solves the programme of [`HuberOptimum`](@ref).

# Algorithm

 1. Seed `b` and `g` at the held allocation `w`, and `rho` at zero, as step 3 of the paper's Algorithm 1 does. Compute `a = lambda / gamma`.
 2. Compute the right-hand side `rhs` of the update of ``\\boldsymbol{b}`` from `g` and `rho`.
 3. Compute `b` from `rhs` with the inverse in closed form, in ``O(N)`` operations.
 4. Compute `g`, the soft threshold of `b` at `gamma`.
 5. Compute the budget residual `res = sum(b) - 1`, and add `eta * res` to `rho`.
 6. When `abs(res) < tol`, or after `iters` iterations, return `b`. Otherwise go back to step 2.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AlternatingDirectionMethod(;
        lambda::Real = 0.5,
        gamma::Real = 0.01,
        eta::Real = 0.005,
        iters::Integer = 10_000,
        tol::Real = 1e-4
    ) -> AlternatingDirectionMethod

Keywords correspond to the struct's fields, and the defaults are the paper's.

## Validation

  - `lambda > 0`, `gamma > 0`, `eta > 0`, `tol > 0`. A `DomainError` is thrown otherwise.
  - `iters >= 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> AlternatingDirectionMethod()
AlternatingDirectionMethod
  lambda ┼ Float64: 0.5
   gamma ┼ Float64: 0.01
     eta ┼ Float64: 0.005
   iters ┼ Int64: 10000
     tol ┴ Float64: 0.0001
```

# Related

  - [`ShortTermSparsePortfolio`](@ref)
  - [`HuberOptimum`](@ref)
  - [`L1Optimum`](@ref)

# References

  - $(ref_dict[:lai2018sspo])
"""
struct AlternatingDirectionMethod{T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Integer,
                                  T5 <: Real} <: AbstractSparsePortfolioAlgorithm
    """
    $(field_dict[:lambda_sspo])
    """
    lambda::T1
    """
    $(field_dict[:gamma_sspo])
    """
    gamma::T2
    """
    The weight of the penalty on the budget residual, which is also the step of the dual variable.
    """
    eta::T3
    """
    $(field_dict[:iter])
    """
    iters::T4
    """
    The convergence tolerance on the budget residual.
    """
    tol::T5
    function AlternatingDirectionMethod(lambda::Real, gamma::Real, eta::Real,
                                        iters::Integer, tol::Real)
        @argcheck(lambda > zero(lambda), DomainError(lambda, "lambda must be positive"))
        @argcheck(gamma > zero(gamma), DomainError(gamma, "gamma must be positive"))
        @argcheck(eta > zero(eta), DomainError(eta, "eta must be positive"))
        @argcheck(iters >= 1, DomainError(iters, "iters must be at least 1"))
        @argcheck(tol > zero(tol), DomainError(tol, "tol must be positive"))
        return new{typeof(lambda), typeof(gamma), typeof(eta), typeof(iters), typeof(tol)}(lambda,
                                                                                           gamma,
                                                                                           eta,
                                                                                           iters,
                                                                                           tol)
    end
end
function AlternatingDirectionMethod(; lambda::Real = 0.5, gamma::Real = 0.01,
                                    eta::Real = 0.005, iters::Integer = 10_000,
                                    tol::Real = 1e-4)::AlternatingDirectionMethod
    return AlternatingDirectionMethod(lambda, gamma, eta, iters, tol)
end
"""
    sparse_portfolio_iterate(alg::L1Optimum, phi::AbstractVector, w::AbstractVector)
    sparse_portfolio_iterate(alg::HuberOptimum, phi::AbstractVector, w::AbstractVector)
    sparse_portfolio_iterate(alg::AlternatingDirectionMethod, phi::AbstractVector, w::AbstractVector)

Finds the unscaled target of a [`ShortTermSparsePortfolio`](@ref) step for the objective vector `phi`.

[`L1Optimum`](@ref) returns the optimum of the programme that the paper states. [`HuberOptimum`](@ref) returns the fixed point of the paper's iteration in closed form. [`AlternatingDirectionMethod`](@ref) runs the paper's iteration from the held allocation `w`, and it is the only method that reads `w`. The type docstrings state the formulas.

# Returns

  - `b::AbstractVector`: The unscaled target. The two closed forms sum to one. The iteration sums to one within `tol` when it stops at `tol`, and it can be further from one when it runs all `iters` iterations.

# Related

  - [`AbstractSparsePortfolioAlgorithm`](@ref)
  - [`ShortTermSparsePortfolio`](@ref)
  - [`largest_forecast_split`](@ref)
  - [`huber_multiplier`](@ref)
"""
function sparse_portfolio_iterate(::L1Optimum, phi::AbstractVector, ::AbstractVector)
    return largest_forecast_split(phi)
end
function sparse_portfolio_iterate(alg::HuberOptimum, phi::AbstractVector, ::AbstractVector)
    lo = -alg.lambda - minimum(phi)
    hi = alg.lambda - maximum(phi)
    if lo > hi
        return largest_forecast_split(phi)
    end
    b = huber_coordinates(alg, phi, huber_multiplier(alg, phi, lo, hi))
    return huber_remainder!(b, phi)
end
function sparse_portfolio_iterate(alg::AlternatingDirectionMethod, phi::AbstractVector,
                                  w::AbstractVector)
    b = collect(w)
    g = copy(b)
    rho = zero(eltype(b)) * alg.eta
    a = alg.lambda / alg.gamma
    N = length(w)
    for _ in 1:(alg.iters)
        rhs = a .* g .+ (alg.eta - rho) .- phi
        # `(a I + η 1 1ᵀ)⁻¹ v = v / a − η (1ᵀ v) / (a (a + η N)) 1` by Sherman–Morrison.
        b = rhs ./ a .- alg.eta * sum(rhs) / (a * (a + alg.eta * N))
        g = sign.(b) .* max.(abs.(b) .- alg.gamma, zero(alg.gamma))
        res = sum(b) - one(eltype(b))
        rho += alg.eta * res
        if abs(res) < alg.tol
            break
        end
    end
    return b
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Splits the whole budget in equal parts between the smallest entries of `phi`, which are the assets with the largest forecast.

The split is the optimum of [`L1Optimum`](@ref). [`HuberOptimum`](@ref) returns it where its programme has no minimum.

# Related

  - [`L1Optimum`](@ref)
  - [`HuberOptimum`](@ref)
"""
function largest_forecast_split(phi::AbstractVector)
    K = phi .== minimum(phi)
    return ifelse.(K, one(eltype(phi)) / count(K), zero(eltype(phi)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the coordinates of [`HuberOptimum`](@ref) at the multiplier `nu`, each clamped to `[-gamma, gamma]`.

The docstring of [`HuberOptimum`](@ref) states the formula of the coordinates.

# Related

  - [`HuberOptimum`](@ref)
  - [`huber_multiplier`](@ref)
"""
function huber_coordinates(alg::HuberOptimum, phi::AbstractVector, nu::Real)
    a = alg.lambda / alg.gamma
    return clamp.((phi .+ nu) ./ -a, -alg.gamma, alg.gamma)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Finds the multiplier of [`HuberOptimum`](@ref) in `[lo, hi]`.

The sum of the coordinates is linear between two knots, so the linear interpolation between the two knots that bracket the root gives the root.

# Algorithm

 1. Compute `budget(nu)`, the sum of the coordinates of [`huber_coordinates`](@ref) at a multiplier `nu`.
 2. When `budget(lo) <= 1`, return `lo`. When `budget(hi) >= 1`, return `hi`.
 3. Collect the knots `-phi .- lambda` and `-phi .+ lambda` that lie strictly between `lo` and `hi`, giving `inner`.
 4. Sort `inner`, and put `lo` before it and `hi` after it, giving `knots`.
 5. Compute `sums`, the budget at each knot.
 6. Find the first knot `i` whose sum is at most one.
 7. Interpolate linearly between the knots `i - 1` and `i`, and return the multiplier where the budget is one.

# Related

  - [`HuberOptimum`](@ref)
  - [`huber_coordinates`](@ref)
  - [`huber_remainder!`](@ref)
"""
function huber_multiplier(alg::HuberOptimum, phi::AbstractVector, lo::Real, hi::Real)
    budget(nu) = sum(huber_coordinates(alg, phi, nu))
    if budget(lo) <= one(lo)
        return lo
    elseif budget(hi) >= one(hi)
        return hi
    end
    inner = filter(v -> lo < v < hi, vcat(-phi .- alg.lambda, -phi .+ alg.lambda))
    knots = vcat(lo, sort!(inner), hi)
    sums = map(budget, knots)
    i = findfirst(v -> v <= one(v), sums)
    return knots[i - 1] +
           (sums[i - 1] - one(eltype(sums))) * (knots[i] - knots[i - 1]) /
           (sums[i - 1] - sums[i])
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Adds the remainder of the budget to `b` in place, in equal parts between tied assets.

At an end of the interval of the multiplier, the assets that take the remainder are the assets where the Huber penalty is linear, so they can hold any amount past `gamma`. Inside the interval the remainder is rounding.

# Algorithm

 1. Compute the remainder `r = 1 - sum(b)`.
 2. Find the assets `K`, which have the smallest entry of `phi` when `r` is not negative, and the largest entry when `r` is negative.
 3. Add `r / count(K)` to each asset of `K`, and return `b`.

# Related

  - [`HuberOptimum`](@ref)
  - [`huber_multiplier`](@ref)
"""
function huber_remainder!(b::AbstractVector, phi::AbstractVector)
    r = one(eltype(b)) - sum(b)
    K = phi .== (r >= zero(r) ? minimum(phi) : maximum(phi))
    b[K] .+= r / count(K)
    return b
end
export L1Optimum, HuberOptimum, AlternatingDirectionMethod
public AbstractSparsePortfolioAlgorithm, sparse_portfolio_iterate
