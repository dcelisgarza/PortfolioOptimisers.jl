"""
$(DocStringExtensions.TYPEDEF)

The optimum of the programme the short-term sparse portfolio's paper states: the whole budget on the asset with the largest Price Relative Forecast, split in equal parts between tied assets.

# Mathematical definition

The programme is ``\\min_{\\boldsymbol{b}} \\langle \\boldsymbol{b}, \\boldsymbol{\\phi} \\rangle + \\lambda \\lVert \\boldsymbol{b} \\rVert_1`` subject to ``\\boldsymbol{1}^\\intercal \\boldsymbol{b} = 1``. Over ``\\boldsymbol{b} \\geq \\boldsymbol{0}`` the penalty is the constant ``\\lambda``, so the optimum there is the vertex ``\\boldsymbol{e}_k`` with ``k = \\arg\\min_i \\phi_i``. A short of ``t`` on asset ``j`` that funds more of asset ``k`` changes the objective by ``t (\\phi_k - \\phi_j + 2 \\lambda)``, which is not negative while ``\\max \\boldsymbol{\\phi} - \\min \\boldsymbol{\\phi} \\leq 2 \\lambda``, so no short improves on the vertex. Past that bound the programme has no minimum, and the objective falls without end along the same direction, so the vertex is also its limit. The optimum does not depend on ``\\lambda``, and the algorithm takes no parameter. Tied assets share an optimal face, and the equal split is its centre.

# Examples

```jldoctest
julia> L1Optimum()
L1Optimum()
```

# Related

  - [`ShortTermSparsePortfolio`](@ref)
  - [`HuberOptimum`](@ref)
  - [`AlternatingDirectionMethod`](@ref)

# References

  - $(ref_dict[:lai2018sspo])
"""
struct L1Optimum <: AbstractSparsePortfolioAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The fixed point of the short-term sparse portfolio's alternating direction iteration, solved in closed form.

# Mathematical definition

The fixed point solves ``\\min_{\\boldsymbol{b}, \\boldsymbol{g}} \\langle \\boldsymbol{b}, \\boldsymbol{\\phi} \\rangle + \\lambda \\lVert \\boldsymbol{g} \\rVert_1 + \\tfrac{a}{2} \\lVert \\boldsymbol{b} - \\boldsymbol{g} \\rVert^2`` subject to ``\\boldsymbol{1}^\\intercal \\boldsymbol{b} = 1``, with ``a = \\lambda / \\gamma``. The minimum over ``\\boldsymbol{g}`` is the soft threshold of ``\\boldsymbol{b}`` at ``\\gamma``, and it leaves a Huber penalty on each coordinate,

```math
\\begin{align}
h(b) &= \\begin{cases} \\tfrac{a}{2} b^2 & \\lvert b \\rvert \\leq \\gamma\\,, \\\\ \\lambda \\lvert b \\rvert - \\tfrac{\\lambda \\gamma}{2} & \\text{otherwise}\\,, \\end{cases}
\\end{align}
```

so the programme is ``\\min_{\\boldsymbol{b}} \\langle \\boldsymbol{b}, \\boldsymbol{\\phi} \\rangle + \\sum_i h(b_i)`` subject to the budget. Its derivative ``h'(b) = \\mathrm{clamp}(a b, -\\lambda, \\lambda)`` is bounded, so a multiplier ``\\nu`` exists only in ``[-\\lambda - \\min \\boldsymbol{\\phi}, \\lambda - \\max \\boldsymbol{\\phi}]``, and every asset inside the clamp holds

```math
\\begin{align}
b_i(\\nu) &= \\mathrm{clamp}\\left( -\\frac{\\phi_i + \\nu}{a}, -\\gamma, \\gamma \\right)\\,.
\\end{align}
```

The sum of the ``b_i(\\nu)`` falls piecewise linearly in ``\\nu`` with knots at ``-\\phi_i \\pm \\lambda``, so ``\\nu`` is its root at one, read off the knots, or the end of the interval where the sum cannot reach one. At the lower end the assets with the smallest ``\\phi_i`` take the remainder of the budget in equal parts, and at the upper end the assets with the largest take it. When ``N \\gamma \\leq 1`` the sum is at most one everywhere, so the largest forecast takes ``1 - \\sum_{i \\neq k} b_i`` and every other asset holds at most ``\\gamma``. Past the bound ``\\max \\boldsymbol{\\phi} - \\min \\boldsymbol{\\phi} \\leq 2 \\lambda`` the interval is empty and the programme has no minimum; the algorithm then takes the limit the objective falls towards, the whole budget on the largest forecast, as [`L1Optimum`](@ref) does.

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

The alternating direction iteration the short-term sparse portfolio's paper runs, stopped where the paper stops it.

# Mathematical definition

With ``a = \\lambda / \\gamma``, the iteration is seeded at ``\\boldsymbol{b} = \\boldsymbol{g} = \\boldsymbol{w}_t`` and ``\\rho = 0``, and repeats

```math
\\begin{align}
\\boldsymbol{b} &\\leftarrow \\left( a \\boldsymbol{I} + \\eta \\boldsymbol{1} \\boldsymbol{1}^\\intercal \\right)^{-1} \\left( a \\boldsymbol{g} + (\\eta - \\rho) \\boldsymbol{1} - \\boldsymbol{\\phi} \\right)\\,,\\quad
\\boldsymbol{g} \\leftarrow \\operatorname{sign}(\\boldsymbol{b}) \\odot \\max(\\lvert \\boldsymbol{b} \\rvert - \\gamma, 0)\\,,\\quad
\\rho \\leftarrow \\rho + \\eta (\\boldsymbol{1}^\\intercal \\boldsymbol{b} - 1)\\,,
\\end{align}
```

until ``\\lvert \\boldsymbol{1}^\\intercal \\boldsymbol{b} - 1 \\rvert < \\texttt{tol}`` or `iters` iterations. The fixed matrix is inverted once in closed form through the Sherman–Morrison identity, so every iteration is ``O(N)``. The fixed point is [`HuberOptimum`](@ref)'s. The dual step ``\\eta`` is small, so the iterate needs thousands to hundreds of thousands of iterations to reach it, and the budget residual changes sign as the dual variable adapts. The paper's `tol = 1e-4` is therefore met at a zero crossing after a few hundred to a few thousand iterations, while the iterate is still moving by tenths, and the scaled projection of that point and of the fixed point can land on different assets. Take this algorithm to reproduce the paper's loop; a tolerance the crossings never reach runs every one of the `iters` iterations.

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
    The penalty on the budget equality and the dual step.
    """
    eta::T3
    """
    Maximum number of iterations.
    """
    iters::T4
    """
    Convergence tolerance on the budget residual.
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

The iterate ``\\boldsymbol{b}`` of a [`ShortTermSparsePortfolio`](@ref) step for the objective `phi`, which sums to one: the optimum of the stated programme, the closed form of the iteration's fixed point, or the paper's iteration seeded at the held allocation `w`. Only the iteration reads `w`.

# Related

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

The whole budget on the smallest entries of `phi`, the largest forecasts, in equal parts between ties.

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

The coordinates ``\\mathrm{clamp}(-(\\phi_i + \\nu) / a, -\\gamma, \\gamma)`` of [`HuberOptimum`](@ref) at the multiplier `nu`, with ``a = \\lambda / \\gamma``.

# Related

  - [`huber_multiplier`](@ref)
"""
function huber_coordinates(alg::HuberOptimum, phi::AbstractVector, nu::Real)
    a = alg.lambda / alg.gamma
    return clamp.(-(phi .+ nu) ./ a, -alg.gamma, alg.gamma)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The multiplier of [`HuberOptimum`](@ref) in `[lo, hi]`: `lo` when the coordinates sum to at most one there, `hi` when they sum to at least one there, and otherwise the root of their sum at one. The sum is linear between the knots ``-\\phi_i \\pm \\lambda``, so the root is read off the two knots that bracket it.

# Related

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

Adds the remainder of the budget to `b` in place, in equal parts: to the smallest entries of `phi` when the remainder is not negative, and to the largest when it is. At an end of the multiplier's interval those assets are the ones the clamp does not bind, and inside it the remainder is rounding.

# Related

  - [`huber_multiplier`](@ref)
"""
function huber_remainder!(b::AbstractVector, phi::AbstractVector)
    r = one(eltype(b)) - sum(b)
    K = phi .== (r >= zero(r) ? minimum(phi) : maximum(phi))
    b[K] .+= r / count(K)
    return b
end
export L1Optimum, HuberOptimum, AlternatingDirectionMethod
