# =============================================================================
# Prototype 8 — The Critical Line Algorithm: the exact efficient frontier.
#
# Purpose
#   Every mean-variance problem in the library goes to a generic conic solver.
#   A solver returns one portfolio for one risk aversion. Sweeping the frontier
#   therefore costs one solve per point, and the points in between are
#   guessed by interpolation.
#
#   The Critical Line Algorithm of Markowitz (1956) returns the frontier
#   **exactly**, as a finite list of turning points. Between two consecutive
#   turning points the optimal weights are an affine function of the risk
#   aversion, so the whole frontier is known in closed form after a number of
#   steps equal to the number of times an asset enters or leaves a bound.
#
#   What that buys, beyond speed:
#
#     1. The exact set of corner portfolios, which is what a caller needs to
#        answer "at which risk aversion does asset seven enter the portfolio?".
#     2. A frontier with no interpolation error, so a plot cannot mislead.
#     3. An audit trail: the active set at every point on the frontier.
#
# Status
#   Standalone. Depends on `LinearAlgebra` only. The verification driver
#   compares every turning point against a conic solver.
#
# Notation used throughout this file
#   N       Number of assets.
#   mu      Expected returns, length `N`.
#   sigma   Covariance matrix, `N x N`, symmetric positive definite.
#   lb, ub  Lower and upper weight bounds, each length `N`.
#   lambda  Risk-aversion parameter, non-negative. Large means return seeking.
#           `lambda = 0` gives the global minimum-variance portfolio.
#   F       The free set: indices whose weight is strictly inside its bounds.
#   B       The bounded set: indices resting on a bound.
#   w_F     The free weights. `w_B` are the bounded ones, held fixed.
#   gam     The multiplier of the budget constraint.
#
# Sources
#   Markowitz, H. M. (1956). The optimization of a quadratic function subject
#     to linear constraints. Naval Research Logistics Quarterly 3(1-2),
#     111-133. The original algorithm.
#   Markowitz, H. M. (1959). Portfolio Selection: Efficient Diversification of
#     Investments. Wiley.
#   Niedermayer, A. and Niedermayer, D. (2010). Applying Markowitz's critical
#     line algorithm. In: Handbook of Portfolio Construction, Springer,
#     383-400. The modern exposition, and the speed comparison against a
#     general quadratic-programme solver.
#   Bailey, D. H. and Lopez de Prado, M. (2013). An open-source implementation
#     of the critical-line algorithm for portfolio optimization. Algorithms
#     6(1), 169-196. The reference implementation this file follows.
# =============================================================================
module CriticalLine

using LinearAlgebra

export CLAFrontier, critical_line, weights_at_lambda, frontier_points,
       max_sharpe_on_frontier

"""
    CLAFrontier{T}

The exact efficient frontier, as a list of turning points.

# Fields

  - `lambdas::Vector{T}`: Risk-aversion values at the turning points, in
    **decreasing** order. The first is the return-seeking end and the last is
    zero, the global minimum-variance portfolio.
  - `weights::Vector{Vector{T}}`: The portfolio at each turning point.
  - `free_sets::Vector{Vector{Int}}`: The free set at each turning point. This
    is the audit trail. An index that appears in one entry and not the next
    hit a bound at that value of `lambda`.
  - `gammas::Vector{T}`: The budget multiplier at each turning point.

# Notes

  - **The frontier is affine between consecutive turning points.** For
    `lambda` between `lambdas[k+1]` and `lambdas[k]`, the optimum is the
    linear interpolation of `weights[k+1]` and `weights[k]`. That is the whole
    content of the algorithm, and it is why [`weights_at_lambda`](@ref) needs
    no solver.
"""
struct CLAFrontier{T <: Real}
    lambdas::Vector{T}
    weights::Vector{Vector{T}}
    free_sets::Vector{Vector{Int}}
    gammas::Vector{T}
end

"""
    _affine_coefficients(sigma, mu, lb, ub, F, w, budget) -> (alpha, beta, C)

Return the affine representation of the free weights as a function of `lambda`.

# Arguments

  - `sigma`, `mu`: The problem data.
  - `lb`, `ub`: Bounds (unused here, kept for signature symmetry).
  - `F`: The free set, a vector of indices.
  - `w`: The current full weight vector. Only its bounded entries are read.
  - `budget`: The total weight the portfolio must sum to, normally one.

# Returns

  - `alpha`, `beta`: Vectors of length `length(F)` with
    `w_F(lambda) = alpha + lambda * beta`.
  - `C`: The inverse of `sigma[F, F]`, reused by the caller.

# Mathematical definition

For the problem

    minimise  (1/2) w' sigma w  -  lambda mu' w    subject to  1' w = budget

the stationarity condition on the free block, with the bounded block held at
`w_B`, is

    sigma_FF w_F  +  sigma_FB w_B  -  lambda mu_F  -  gam 1  =  0

and the budget gives `1' w_F = budget - 1' w_B =: s`. Writing `C = sigma_FF^(-1)`
and

    a0 = 1' C mu_F,   a1 = 1' C 1,   a2 = 1' C sigma_FB w_B

the budget multiplier is

    gam(lambda) = ( s + a2 - lambda a0 ) / a1

and substitution gives the affine form

    beta   =  C mu_F  -  (a0 / a1) C 1
    alpha  =  -C sigma_FB w_B  +  ( (s + a2) / a1 ) C 1

**Both are constant while the free set is constant**, which is exactly the
definition of a segment of the critical line.
"""
function _affine_coefficients(sigma::AbstractMatrix{T}, mu::AbstractVector{T},
                              lb::AbstractVector{T}, ub::AbstractVector{T}, F::Vector{Int},
                              w::AbstractVector{T}, budget::T) where {T <: Real}
    B = setdiff(1:length(mu), F)
    C = inv(Symmetric(sigma[F, F]))
    onesF = ones(T, length(F))
    Cones = C * onesF
    Cmu = C * view(mu, F)
    a0 = dot(onesF, Cmu)
    a1 = dot(onesF, Cones)
    wB = view(w, B)
    s = budget - sum(wB)
    if isempty(B)
        a2 = zero(T)
        corr = zeros(T, length(F))
    else
        sfb_wb = sigma[F, B] * wB
        corr = C * sfb_wb
        a2 = dot(onesF, corr)
    end
    beta = Cmu .- (a0 / a1) .* Cones
    alpha = -corr .+ ((s + a2) / a1) .* Cones
    return alpha, beta, C
end

"""
    _initial_turning_point(mu, lb, ub, budget) -> (w, free_index)

Return the return-maximising feasible portfolio, and the single index that is
free there.

# Details

Start every asset at its lower bound. Walk the assets in decreasing order of
expected return and raise each to its upper bound until the budget is met. The
asset that absorbs the remainder is the only one strictly inside its bounds,
so it is the free set at `lambda = infinity`.

This is the `initAlgo` step of Bailey and Lopez de Prado (2013).

# Validation

  - `sum(lb) <= budget <= sum(ub)`, otherwise the problem is infeasible.
"""
function _initial_turning_point(mu::AbstractVector{T}, lb::AbstractVector{T},
                                ub::AbstractVector{T}, budget::T) where {T <: Real}
    if !(sum(lb) <= budget <= sum(ub))
        throw(ArgumentError("infeasible bounds: sum(lb) = $(sum(lb)), sum(ub) = $(sum(ub)), budget = $(budget)"))
    end
    order = sortperm(mu; rev = true)
    w = collect(lb)
    remaining = budget - sum(lb)
    free = last(order)
    for i in order
        room = ub[i] - lb[i]
        if remaining <= room
            w[i] += remaining
            remaining = zero(T)
            free = i
            break
        else
            w[i] = ub[i]
            remaining -= room
        end
    end
    return w, free
end

"""
    critical_line(mu::AbstractVector, sigma::AbstractMatrix;
                  lb = zeros(length(mu)), ub = ones(length(mu)),
                  budget::Real = 1.0, tol::Real = 1e-10,
                  max_iter::Integer = 10_000) -> CLAFrontier

Trace the exact efficient frontier.

# Arguments

  - `mu`: Expected returns, length `N`.
  - `sigma`: Covariance, `N x N`, symmetric positive definite.
  - `lb`, `ub`: Weight bounds, each length `N`. Defaults give a long-only
    portfolio with no single-name cap.
  - `budget`: The required total weight.
  - `tol`: Numerical tolerance for bound tests and for the free-set update.
  - `max_iter`: Safety cap on the number of turning points.

# Returns

  - A [`CLAFrontier`](@ref), with `lambdas` decreasing and ending at zero.

# Algorithm

 1. Start at the return-maximising corner, with one free asset.
 2. At each step compute two candidate values of `lambda`:
    **Case A**, the largest `lambda` below the current one at which some free
    asset reaches one of its own bounds; and **Case B**, the largest `lambda`
    below the current one at which some bounded asset would leave its bound.
 3. Take whichever candidate is larger, update the free set, and record the
    turning point.
 4. Stop when no candidate is positive. Append the `lambda = 0` point, which
    is the global minimum-variance portfolio subject to the bounds.

# Notes

  - **The number of turning points is not bounded by `N`.** An asset may leave
    a bound and return to it, so the count is the number of active-set changes
    along the frontier. `max_iter` guards against a cycle caused by numerical
    noise.
  - `sigma[F, F]` must be invertible at every step. For a singular or
    near-singular covariance, process it first. The library's `PosdefEstimator`
    and denoising machinery exist for exactly this.
"""
function critical_line(mu::AbstractVector{<:Real}, sigma::AbstractMatrix{<:Real};
                       lb::AbstractVector{<:Real} = zeros(length(mu)),
                       ub::AbstractVector{<:Real} = ones(length(mu)), budget::Real = 1.0,
                       tol::Real = 1e-10, max_iter::Integer = 10_000)
    N = length(mu)
    if size(sigma) != (N, N)
        throw(DimensionMismatch("sigma must be $(N) x $(N), got $(size(sigma))"))
    end
    if length(lb) != N || length(ub) != N
        throw(DimensionMismatch("lb and ub must both have length $(N)"))
    end
    T = float(promote_type(eltype(mu), eltype(sigma), eltype(lb), eltype(ub)))
    mu_ = collect(T, mu)
    sig = Matrix{T}(sigma)
    lb_ = collect(T, lb)
    ub_ = collect(T, ub)
    bud = T(budget)
    tol_ = T(tol)

    w, first_free = _initial_turning_point(mu_, lb_, ub_, bud)
    F = [first_free]
    lam_current = T(Inf)

    lambdas = T[]
    weights = Vector{T}[]
    free_sets = Vector{Int}[]
    gammas = T[]

    for _ in 1:max_iter
        # ---- Case A: a free asset reaches one of its bounds -----------------
        lam_a = T(-Inf)
        i_a = 0
        bound_a = zero(T)
        if length(F) > 1
            alpha, beta, _ = _affine_coefficients(sig, mu_, lb_, ub_, F, w, bud)
            for (k, i) in enumerate(F)
                if abs(beta[k]) <= tol_
                    continue
                end
                for v in (lb_[i], ub_[i])
                    lam = (v - alpha[k]) / beta[k]
                    if lam < lam_current - tol_ && lam > lam_a
                        lam_a = lam
                        i_a = i
                        bound_a = v
                    end
                end
            end
        end

        # ---- Case B: a bounded asset leaves its bound -----------------------
        lam_b = T(-Inf)
        i_b = 0
        for j in setdiff(1:N, F)
            Fj = sort(vcat(F, j))
            # `sigma[Fj, Fj]` must stay invertible; skip the candidate if not.
            local alpha, beta
            try
                alpha, beta, _ = _affine_coefficients(sig, mu_, lb_, ub_, Fj, w, bud)
            catch
                continue
            end
            k = findfirst(==(j), Fj)
            if abs(beta[k]) <= tol_
                continue
            end
            lam = (w[j] - alpha[k]) / beta[k]
            if lam < lam_current - tol_ && lam > lam_b
                lam_b = lam
                i_b = j
            end
        end

        # ---- Choose the transition -----------------------------------------
        if max(lam_a, lam_b) <= tol_
            lam_current = zero(T)
        elseif lam_a > lam_b
            lam_current = lam_a
            w[i_a] = bound_a
            deleteat!(F, findfirst(==(i_a), F))
        else
            lam_current = lam_b
            push!(F, i_b)
            sort!(F)
        end

        # ---- Record the turning point ---------------------------------------
        alpha, beta, C = _affine_coefficients(sig, mu_, lb_, ub_, F, w, bud)
        wF = alpha .+ lam_current .* beta
        for (k, i) in enumerate(F)
            w[i] = wF[k]
        end
        onesF = ones(T, length(F))
        a1 = dot(onesF, C * onesF)
        a0 = dot(onesF, C * view(mu_, F))
        Bset = setdiff(1:N, F)
        s = bud - sum(view(w, Bset))
        a2 = isempty(Bset) ? zero(T) : dot(onesF, C * (sig[F, Bset] * view(w, Bset)))
        gam = (s + a2 - lam_current * a0) / a1

        push!(lambdas, lam_current)
        push!(weights, copy(w))
        push!(free_sets, copy(F))
        push!(gammas, gam)

        lam_current <= tol_ && break
    end
    return CLAFrontier{T}(lambdas, weights, free_sets, gammas)
end

"""
    weights_at_lambda(f::CLAFrontier, lambda::Real) -> Vector

Return the exact optimal portfolio at any risk aversion, without a solver.

# Arguments

  - `f`: A traced frontier.
  - `lambda`: Risk aversion, non-negative.

# Returns

  - The optimal weights, length `N`.

# Details

The optimum is affine in `lambda` between consecutive turning points, so the
answer is the linear interpolation of the two turning points that bracket the
request. A `lambda` above the largest turning point returns the
return-maximising corner, because the frontier is constant beyond it.

**This is the payoff of the algorithm.** After one trace, any point on the
frontier costs one interpolation.
"""
function weights_at_lambda(f::CLAFrontier{T}, lambda::Real) where {T}
    lam = T(lambda)
    if lam < 0
        throw(DomainError(lambda, "lambda must be >= 0"))
    end
    lams = f.lambdas
    if lam >= first(lams)
        return copy(first(f.weights))
    end
    if lam <= last(lams)
        return copy(last(f.weights))
    end
    for k in 1:(length(lams) - 1)
        hi, lo = lams[k], lams[k + 1]
        if lo <= lam <= hi
            t = hi ≈ lo ? zero(T) : (lam - lo) / (hi - lo)
            return (1 - t) .* f.weights[k + 1] .+ t .* f.weights[k]
        end
    end
    return copy(last(f.weights))
end

"""
    frontier_points(f::CLAFrontier, mu::AbstractVector, sigma::AbstractMatrix;
                    n::Integer = 0) -> NamedTuple

Return the risk and return coordinates of the frontier.

# Arguments

  - `f`: A traced frontier.
  - `mu`, `sigma`: The problem data.
  - `n`: If zero, report the turning points themselves. If positive, report `n`
    points evenly spaced in `lambda` between the ends.

# Returns

A `NamedTuple` with `lambdas`, `risk` (standard deviation), `ret` (expected
return) and `weights`.
"""
function frontier_points(f::CLAFrontier{T}, mu::AbstractVector{<:Real},
                         sigma::AbstractMatrix{<:Real}; n::Integer = 0) where {T}
    lams = if n <= 0
        f.lambdas
    else
        collect(range(first(f.lambdas), last(f.lambdas); length = n))
    end
    ws = [weights_at_lambda(f, l) for l in lams]
    risk = [sqrt(max(dot(w, sigma, w), zero(T))) for w in ws]
    ret = [dot(w, mu) for w in ws]
    return (; lambdas = lams, risk = risk, ret = ret, weights = ws)
end

"""
    max_sharpe_on_frontier(f::CLAFrontier, mu::AbstractVector,
                           sigma::AbstractMatrix; rf::Real = 0.0,
                           n::Integer = 2000) -> NamedTuple

Return the maximum-Sharpe portfolio on the traced frontier.

# Arguments

  - `f`: A traced frontier.
  - `mu`, `sigma`: The problem data.
  - `rf`: Risk-free rate, in the same period units as `mu`.
  - `n`: Number of interpolation points used in the golden-section-free scan.

# Returns

A `NamedTuple` with `w`, `sharpe`, `lambda`, `risk` and `ret`.

# Details

The Sharpe ratio is quasi-concave along the frontier, so a dense scan of the
affine segments finds the maximum reliably. **A dense scan is cheap here
precisely because evaluation needs no solver**, which is not true of the
frontier the library builds today.
"""
function max_sharpe_on_frontier(f::CLAFrontier{T}, mu::AbstractVector{<:Real},
                                sigma::AbstractMatrix{<:Real}; rf::Real = 0.0,
                                n::Integer = 2000) where {T}
    pts = frontier_points(f, mu, sigma; n = n)
    best = -Inf
    k_best = 1
    for k in eachindex(pts.lambdas)
        sr = pts.risk[k] > 0 ? (pts.ret[k] - rf) / pts.risk[k] : -Inf
        if sr > best
            best = sr
            k_best = k
        end
    end
    return (; w = pts.weights[k_best], sharpe = best, lambda = pts.lambdas[k_best],
            risk = pts.risk[k_best], ret = pts.ret[k_best])
end

end # module CriticalLine
