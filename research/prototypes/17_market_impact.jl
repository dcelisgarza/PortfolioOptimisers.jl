# =============================================================================
# Prototype 17 — Market impact and optimal execution.
#
# Purpose
#   Reports 2, 3, 4 and 7 all name this gap. The library models fees and
#   turnover, and `BudgetMarketImpact` in
#   `src/20_Optimisation/09_JuMPConstraints/03_BudgetConstraints.jl` is the only
#   place impact appears. Fees and impact are different things:
#
#     * A **fee** is proportional to the value traded. Trading twice as much
#       costs twice as much.
#     * **Impact** is the price move the trade itself causes. Trading twice as
#       much costs roughly `2^(3/2)`, so about 2.8 times as much.
#
#   The consequence is structural, not cosmetic. A proportional cost leaves the
#   optimal portfolio unchanged in shape and merely creates a no-trade band. A
#   convex cost changes the *answer*, because it makes concentration expensive
#   in a way variance does not capture, and it makes the trade schedule a
#   decision in its own right.
#
# Status
#   Standalone. Depends on `LinearAlgebra` and `Statistics`. The convexity and
#   conic-representability notes are checked numerically by the driver.
#
# Notation used throughout this file
#   Q        Order size, in shares or in currency. Always non-negative here.
#   V        Average daily volume, same units as `Q`.
#   sigma    Volatility of the asset, per period, as a fraction.
#   X0       Total position to liquidate.
#   n        Number of trading intervals.
#   tau      Length of one interval.
#   x_k      Shares remaining after interval `k`. `x_0 = X0`, `x_n = 0`.
#   n_k      Shares traded in interval `k`, equal to `x_{k-1} - x_k`.
#   eta      Temporary impact coefficient.
#   gam      Permanent impact coefficient.
#   lam      Risk aversion of the execution problem.
#
# Sources
#   Almgren, R. and Chriss, N. (2001). Optimal execution of portfolio
#     transactions. Journal of Risk 3(2), 5-39. The closed-form trajectory
#     implemented here.
#   Almgren, R., Thum, C., Hauptmann, E. and Li, H. (2005). Direct estimation
#     of equity market impact. Risk 18(7), 58-62. The empirical exponent, close
#     to `3/5` for temporary impact rather than the textbook `1/2`.
#   Torre, N. and Ferrari, M. (1997). Market Impact Model Handbook. BARRA. The
#     origin of the square-root rule in practice.
#   Kyle, A. S. (1985). Continuous auctions and insider trading. Econometrica
#     53(6), 1315-1335. The linear permanent-impact model.
#   Gatheral, J. (2010). No-dynamic-arbitrage and market impact. Quantitative
#     Finance 10(7), 749-759. Why permanent impact must be linear.
#   Frazzini, A., Israel, R. and Moskowitz, T. J. (2018). Trading costs.
#     Working paper, AQR Capital Management. Live-execution evidence that
#     realised costs are far below the square-root rule for real traders.
# =============================================================================
module MarketImpact

using LinearAlgebra, Statistics

export AbstractImpactModel, LinearImpact, SquareRootImpact, PowerLawImpact, impact_cost,
       marginal_impact, almgren_chriss_schedule, execution_cost_moments, socp_exponent_note

"""
    AbstractImpactModel

Supertype of the cost models. Each answers one question: what does it cost to
trade `Q` units?
"""
abstract type AbstractImpactModel end

"""
    LinearImpact <: AbstractImpactModel

Cost proportional to the size traded.

# Fields

  - `bps::Float64`: Cost in basis points of the value traded.

# Mathematical definition

    cost(Q)  =  (bps / 10_000) * Q

# Notes

  - This is a **fee**, not impact, and it is included so the others have
    something to be compared against. Its marginal cost is constant, so it
    creates a no-trade band and otherwise leaves the optimal portfolio's shape
    untouched.
"""
struct LinearImpact <: AbstractImpactModel
    bps::Float64
    LinearImpact(; bps::Real = 5.0) = new(float(bps))
end

"""
    SquareRootImpact <: AbstractImpactModel

The square-root law, the industry standard.

# Fields

  - `coefficient::Float64`: Dimensionless constant, typically near one.
  - `sigma::Float64`: Asset volatility per period, as a fraction.
  - `adv::Float64`: Average daily volume, in the same units as the order.

# Mathematical definition

Cost **per unit traded** grows with the square root of participation:

    price move  =  coefficient * sigma * sqrt( Q / ADV )

so the **total** cost is

    cost(Q)  =  coefficient * sigma * Q * sqrt( Q / ADV )
             =  ( coefficient * sigma / sqrt(ADV) ) * Q^(3/2)

# Notes

  - **The exponent `3/2` on total cost is the whole content of the model.**
    Doubling the order raises total cost by `2^(3/2)`, about 2.83.
  - Almgren and co-authors (2005) estimate the exponent on the price move
    empirically at about `3/5` rather than `1/2`, giving a total exponent near
    `1.6`. Use [`PowerLawImpact`](@ref) to test the sensitivity.
  - **The cost is convex in `Q`, so it is optimiser friendly.** A `Q^(3/2)`
    term is representable in a second-order cone, so it can enter a JuMP model
    with no loss of tractability. See [`socp_exponent_note`](@ref).
"""
struct SquareRootImpact <: AbstractImpactModel
    coefficient::Float64
    sigma::Float64
    adv::Float64
    function SquareRootImpact(; coefficient::Real = 1.0, sigma::Real = 0.02,
                              adv::Real = 1.0e6)
        if adv <= 0
            throw(DomainError(adv, "adv must be > 0"))
        end
        if sigma < 0
            throw(DomainError(sigma, "sigma must be >= 0"))
        end
        return new(float(coefficient), float(sigma), float(adv))
    end
end

"""
    PowerLawImpact <: AbstractImpactModel

A general power law, with the exponent exposed.

# Fields

  - `coefficient::Float64`, `sigma::Float64`, `adv::Float64`: As for
    [`SquareRootImpact`](@ref).
  - `delta::Float64`: Exponent on participation in the **price move**. The
    square-root model is `delta = 0.5`; Almgren and co-authors (2005) estimate
    about `0.6`.

# Mathematical definition

    cost(Q)  =  coefficient * sigma * Q * ( Q / ADV )^delta

The total cost exponent is `1 + delta`.

# Notes

  - The cost is convex whenever `delta >= 0`, and it is **conic representable
    for rational `delta`** through a chain of power cones, which JuMP supports
    directly with `MOI.PowerCone`. The library already uses that cone in
    `src/20_Optimisation/09_JuMPConstraints/13_WeightNormConstraints.jl`, so
    the machinery is present.
"""
struct PowerLawImpact <: AbstractImpactModel
    coefficient::Float64
    sigma::Float64
    adv::Float64
    delta::Float64
    function PowerLawImpact(; coefficient::Real = 1.0, sigma::Real = 0.02,
                            adv::Real = 1.0e6, delta::Real = 0.6)
        if adv <= 0
            throw(DomainError(adv, "adv must be > 0"))
        end
        if delta < 0
            throw(DomainError(delta, "delta must be >= 0"))
        end
        return new(float(coefficient), float(sigma), float(adv), float(delta))
    end
end

"""
    impact_cost(model::AbstractImpactModel, Q::Real) -> Real

Return the total cost of trading `Q` units. Sign is ignored, because buying and
selling cost the same in every model here.

# Notes

  - **Cost is a function of `abs(Q)`.** Any model that returns a negative cost
    for a sale is wrong, and the driver asserts symmetry.
"""
impact_cost(m::LinearImpact, Q::Real) = (m.bps / 10_000) * abs(Q)
function impact_cost(m::SquareRootImpact, Q::Real)
    q = abs(Q)
    return m.coefficient * m.sigma * q * sqrt(q / m.adv)
end
function impact_cost(m::PowerLawImpact, Q::Real)
    q = abs(Q)
    return m.coefficient * m.sigma * q * (q / m.adv)^m.delta
end

"""
    marginal_impact(model::AbstractImpactModel, Q::Real) -> Real

Return `d cost / d Q` at `Q > 0`.

# Mathematical definition

  - Linear: `bps / 10_000`, constant.
  - Square root: `(3/2) * coefficient * sigma * sqrt(Q / ADV)`.
  - Power law: `(1 + delta) * coefficient * sigma * (Q / ADV)^delta`.

# Notes

  - **The factor `3/2` is the one practitioners get wrong.** The marginal cost
    of the last share is one and a half times the average cost per share, not
    equal to it. A trader who prices a block at the average square-root cost
    under-charges the marginal decision by fifty per cent.
"""
marginal_impact(m::LinearImpact, Q::Real) = m.bps / 10_000
function marginal_impact(m::SquareRootImpact, Q::Real)
    q = abs(Q)
    return 1.5 * m.coefficient * m.sigma * sqrt(q / m.adv)
end
function marginal_impact(m::PowerLawImpact, Q::Real)
    q = abs(Q)
    return (1 + m.delta) * m.coefficient * m.sigma * (q / m.adv)^m.delta
end

"""
    almgren_chriss_schedule(X0::Real, n::Integer; sigma::Real, eta::Real,
                            gam::Real = 0.0, lam::Real = 1.0e-6, tau::Real = 1.0)
        -> NamedTuple

Return the optimal liquidation trajectory of Almgren and Chriss (2001).

# Arguments

  - `X0`: Total position to liquidate.
  - `n`: Number of trading intervals.
  - `sigma`: Volatility per unit time, in price units per square root of time.
  - `eta`: Temporary impact coefficient, price move per unit trading rate.
  - `gam`: Permanent impact coefficient. Affects the cost but **not** the
    optimal path.
  - `lam`: Risk aversion. Zero gives the minimum-cost path; large gives fast
    liquidation.
  - `tau`: Length of one interval.

# Returns

A `NamedTuple` with `x` (holdings after each interval, length `n + 1`), `n_k`
(shares traded in each interval, length `n`), `kappa`, `half_life`,
`expected_cost` and `variance`.

# Mathematical definition

The trader minimises `E[cost] + lam * Var[cost]`. With
`eta_tilde = eta - gam * tau / 2`, define

    kappa_tilde^2  =  lam sigma^2 / eta_tilde,
    cosh(kappa tau) =  1 + kappa_tilde^2 tau^2 / 2

The optimal holdings follow a hyperbolic decay:

    x_k  =  X0 * sinh( kappa (T - t_k) ) / sinh( kappa T ),     T = n tau

and the trade list is `n_k = x_{k-1} - x_k`.

# The three regimes

  - `lam -> 0`: `kappa -> 0`, the path becomes **linear**, and the strategy is
    a uniform schedule. This is time-weighted average price, and it minimises
    expected cost while accepting maximum risk.
  - `lam` large: `kappa` large, the position is liquidated almost immediately,
    paying impact to remove risk.
  - In between: an exponential-like decay with **half-life `log(2) / kappa`**,
    which is the number a trader actually reasons with.

# Notes

  - **Permanent impact drops out of the optimal path.** Gatheral (2010) shows
    permanent impact must be linear to exclude dynamic arbitrage, and a linear
    permanent cost depends only on the *total* traded, not on the schedule. So
    it changes the bill and not the plan. That is a genuinely
    counter-intuitive result and it is worth stating in any documentation.
  - The trajectory is deterministic. It does not react to the price path. That
    is the model's main limitation and the reason adaptive execution exists.
"""
function almgren_chriss_schedule(X0::Real, n::Integer; sigma::Real, eta::Real,
                                 gam::Real = 0.0, lam::Real = 1.0e-6, tau::Real = 1.0)
    if n < 1
        throw(DomainError(n, "n must be >= 1"))
    end
    if eta <= 0
        throw(DomainError(eta, "eta must be > 0"))
    end
    if lam < 0
        throw(DomainError(lam, "lam must be >= 0"))
    end
    T = n * tau
    eta_t = eta - gam * tau / 2
    if eta_t <= 0
        throw(ArgumentError("eta - gam * tau / 2 = $(eta_t) must be > 0; permanent impact is too large relative to temporary"))
    end
    x = Vector{Float64}(undef, n + 1)
    kappa = 0.0
    if iszero(lam) || iszero(sigma)
        # Risk-neutral limit: uniform liquidation.
        for k in 0:n
            x[k + 1] = X0 * (1 - k / n)
        end
    else
        k2 = lam * sigma^2 / eta_t
        kappa = acosh(1 + k2 * tau^2 / 2) / tau
        s = sinh(kappa * T)
        for k in 0:n
            x[k + 1] = X0 * sinh(kappa * (T - k * tau)) / s
        end
    end
    x[end] = 0.0
    nk = [x[k] - x[k + 1] for k in 1:n]
    # Expected cost: permanent on the total, temporary on each rate.
    ec = gam * X0^2 / 2 + eta_t * sum(abs2, nk) / tau
    # Variance of the execution cost, from the holdings still exposed.
    va = sigma^2 * tau * sum(abs2, view(x, 2:n))
    return (; x = x, n_k = nk, kappa = kappa,
            half_life = iszero(kappa) ? Inf : log(2) / kappa, expected_cost = ec,
            variance = va)
end

"""
    execution_cost_moments(schedule, sigma::Real, eta::Real, gam::Real, tau::Real)
        -> NamedTuple

Recompute the mean and variance of a schedule's implementation shortfall, from
the trajectory alone.

# Arguments

  - `schedule`: The output of [`almgren_chriss_schedule`](@ref), or any
    `NamedTuple` with `x` and `n_k`.
  - `sigma`, `eta`, `gam`, `tau`: The model parameters.

# Returns

A `NamedTuple` with `expected_cost`, `variance` and `std`.

# Mathematical definition

    E[C]   =  (gam / 2) X0^2  +  (eta - gam tau / 2) sum_k n_k^2 / tau
    Var[C] =  sigma^2 tau sum_{k=1}^{n-1} x_k^2

The variance depends on the **holdings still outstanding**, which is why
liquidating faster reduces risk and raises cost. The two terms are the whole
trade-off.

# Notes

  - Used to verify that the closed-form trajectory really is optimal: perturb
    it and the objective `E + lam Var` must rise. The driver does exactly that.
"""
function execution_cost_moments(schedule, sigma::Real, eta::Real, gam::Real, tau::Real)
    x = schedule.x
    nk = schedule.n_k
    X0 = first(x)
    n = length(nk)
    eta_t = eta - gam * tau / 2
    ec = gam * X0^2 / 2 + eta_t * sum(abs2, nk) / tau
    va = sigma^2 * tau * sum(abs2, view(x, 2:n))
    return (; expected_cost = ec, variance = va, std = sqrt(max(va, 0.0)))
end

"""
    socp_exponent_note() -> String

Return the note on how a power-law impact term enters a conic optimiser.

# Details

A term `t >= Q^(3/2)` with `Q >= 0` is representable with two second-order
cones by introducing `s` with

    s^2 <= t * Q       (rotated second-order cone)
    Q^2 <= s * 1       (rotated second-order cone)

Eliminating `s` gives `Q^3 <= t^2 * Q`, that is `t >= Q^(3/2)`.

For a general rational exponent `1 + delta` the term is a single
`MOI.PowerCone(1 / (1 + delta))` constraint, which JuMP passes straight to
Clarabel, SCS or Mosek.

**So a convex impact model costs nothing in tractability.** The reason the
library does not have one is not that it is hard.
"""
function socp_exponent_note()
    return """
    t >= Q^(3/2), Q >= 0  is two rotated second-order cones:
        [t, Q, s] with s^2 <= t * Q  and  Q^2 <= s
    A general exponent 1 + delta is one MOI.PowerCone(1 / (1 + delta)).
    """
end

end # module MarketImpact
