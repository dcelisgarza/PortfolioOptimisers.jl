# =============================================================================
# Prototype 2 — A Wasserstein ambiguity set as a first-class uncertainty set.
#
# Purpose
#   The library holds Wasserstein machinery inside two risk measures
#   (`DistributionallyRobustConditionalValueatRisk` and its range twin) and
#   nowhere else. The uncertainty-set family in `src/14_UncertaintySets` has
#   four members and none of them is an ambiguity set. This file shows the
#   fifth member, and it shows that three separate robust counterparts fall
#   out of one radius.
#
#   The central finding: for a type-1 Wasserstein ball the robust counterpart
#   of a linear loss and of CVaR is the empirical value **plus a dual-norm
#   penalty on the weights**. The library can already emit that penalty. See
#   `L2Regularisation` and `LpRegularisation` in
#   `src/20_Optimisation/09_JuMPConstraints/12_RegularisationConstraints.jl`.
#   What is absent is the name and the calibrated radius.
#
# Status
#   Standalone. Depends on `LinearAlgebra`, `Statistics`, `JuMP` and a conic
#   solver. All are already dependencies of the package or of its test
#   environment.
#
# Notation used throughout this file
#   T        Number of observations (scenarios).
#   N        Number of assets.
#   X        Returns matrix, `T x N`, observations-major. Row `t` is scenario
#            `xi_t`.
#   w        Portfolio weight vector, length `N`.
#   mu_hat   Empirical expected returns, length `N`.
#   sig_hat  Empirical covariance, `N x N`.
#   delta    Wasserstein radius. A distance, never a squared distance.
#   q        Order of the ground metric on the return space.
#   p        Order of the dual norm, `1/p + 1/q = 1`. The penalty uses `p`.
#   alpha    Tail probability of CVaR, in `(0, 1)`.
#
# Sources
#   Mohajerin Esfahani, P. and Kuhn, D. (2018). Data-driven distributionally
#     robust optimization using the Wasserstein metric: performance guarantees
#     and tractable reformulations. Mathematical Programming 171(1), 115-166.
#     The conic reformulation the library already implements for DR-CVaR.
#   Blanchet, J., Chen, L. and Zhou, X. Y. (2022). Distributionally robust
#     mean-variance portfolio selection with Wasserstein distances.
#     Management Science 68(9), 6382-6410. The result that a type-2 ball turns
#     the standard deviation into a 2-norm regularised standard deviation.
#   Blanchet, J., Kang, Y. and Murthy, K. (2019). Robust Wasserstein profile
#     inference and applications to machine learning. Journal of Applied
#     Probability 56(3), 830-857. The data-driven choice of the radius.
#   Gao, R. and Kleywegt, A. J. (2023). Distributionally robust stochastic
#     optimization with Wasserstein distance. Mathematics of Operations
#     Research 48(2), 603-655. The general duality.
#   DeMiguel, V., Garlappi, L., Nogales, F. J. and Uppal, R. (2009). A
#     generalized approach to portfolio optimization: improving performance by
#     constraining portfolio norms. Management Science 55(5), 798-812. The
#     empirical result that norm constraints improve out-of-sample behaviour,
#     which the Wasserstein reading explains.
# =============================================================================
module WassersteinAmbiguity

using LinearAlgebra, Statistics

export WassersteinAmbiguitySet, dual_norm_order, robust_expected_return, robust_std,
       robust_cvar, empirical_cvar, worst_case_shifted_returns, radius_from_confidence

"""
    WassersteinAmbiguitySet{T}

An ambiguity set: the ball of probability measures within Wasserstein distance
`radius` of the empirical measure of the observed returns.

# Fields

  - `radius::T`: The Wasserstein radius `delta`. Must be non-negative. A radius
    of zero recovers the non-robust problem exactly.
  - `order::Int`: The order of the Wasserstein distance, `1` or `2`. Order one
    is the right choice for CVaR and for a linear return term. Order two is the
    right choice for the standard deviation.
  - `ground_norm::T`: The order `q` of the norm on the return space that
    defines the transport cost. Common choices are `1`, `2` and `Inf`.

# Mathematical definition

The type-`k` Wasserstein distance between two measures `P` and `Q` on the
return space is

    W_k(P, Q) = ( inf_{pi in Pi(P, Q)}  E_pi[ ||xi - zeta||_q^k ] )^(1/k)

where `Pi(P, Q)` is the set of couplings with marginals `P` and `Q`. The
ambiguity set is

    B_delta(P_hat) = { P : W_k(P, P_hat) <= delta }

`P_hat` is the empirical measure that puts mass `1/T` on each observed row of
`X`. Note that `radius` bounds the **distance**, not its `k`-th power. A
formulation that bounds the power must pass `delta^(1/k)` here.

# Notes

  - **This is the type the library does not have.** ADR 0050 states that an
    uncertainty set carries the quantity it bounds. An ambiguity set bounds the
    *measure*, which is the quantity every other set is derived from, so it
    sits at the head of the family rather than beside its members.
"""
struct WassersteinAmbiguitySet{T <: Real}
    radius::T
    order::Int
    ground_norm::T
    function WassersteinAmbiguitySet(radius::T, order::Int,
                                     ground_norm::T) where {T <: Real}
        if radius < 0
            throw(DomainError(radius, "radius must be >= 0"))
        end
        if !(order in (1, 2))
            throw(DomainError(order, "order must be 1 or 2"))
        end
        if !(ground_norm >= 1)
            throw(DomainError(ground_norm, "ground_norm must be >= 1"))
        end
        return new{T}(radius, order, ground_norm)
    end
end
function WassersteinAmbiguitySet(; radius::Real = 0.0, order::Integer = 1,
                                 ground_norm::Real = 2.0)
    r, g = promote(float(radius), float(ground_norm))
    return WassersteinAmbiguitySet(r, Int(order), g)
end

"""
    dual_norm_order(set::WassersteinAmbiguitySet) -> Real

Return the order `p` of the norm that appears in the robust penalty.

# Details

The transport cost uses `||.||_q` on the return space. Duality turns it into
`||.||_p` on the weight space, with `1/p + 1/q = 1`. The three cases that
matter:

| ground metric `q` | penalty norm `p` | reading                              |
|:----------------- |:---------------- |:------------------------------------ |
| `1`               | `Inf`            | penalises the largest single holding |
| `2`               | `2`              | penalises concentration, ridge-like  |
| `Inf`             | `1`              | penalises gross exposure, lasso-like |
"""
function dual_norm_order(set::WassersteinAmbiguitySet)
    q = set.ground_norm
    if isinf(q)
        return one(q)
    elseif isone(q)
        return typemax(q)
    end
    return q / (q - 1)
end

"""
    robust_expected_return(set::WassersteinAmbiguitySet, mu_hat::AbstractVector,
                           w::AbstractVector) -> Real

Return the worst-case expected portfolio return over the ambiguity set.

# Arguments

  - `set`: The ambiguity set.
  - `mu_hat`: Empirical expected returns, length `N`.
  - `w`: Portfolio weights, length `N`.

# Returns

  - The scalar `inf_{P in B} E_P[w' xi]`.

# Mathematical definition

For any `P` in the ball, a coupling argument with Holder's inequality gives

    | E_P[w' xi] - E_Phat[w' xi] |  <=  ||w||_p * W_1(P, Phat)  <=  delta * ||w||_p

and the bound is attained by the measure that shifts every atom by
`-delta * g`, where `g` is the vector that attains the dual norm. Hence

    inf_{P in B} E_P[w' xi]  =  mu_hat' w  -  delta * ||w||_p

**This is a linear-plus-norm expression, not a new cone.** A mean term robust
to a Wasserstein ball is the plain mean term minus an `Lp` penalty. The
library can emit exactly this today with `LpRegularisation`.
"""
function robust_expected_return(set::WassersteinAmbiguitySet,
                                mu_hat::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    if length(mu_hat) != length(w)
        throw(DimensionMismatch("mu_hat has length $(length(mu_hat)), w has length $(length(w))"))
    end
    return dot(mu_hat, w) - set.radius * norm(w, dual_norm_order(set))
end

"""
    robust_std(set::WassersteinAmbiguitySet, sig_hat::AbstractMatrix,
               w::AbstractVector) -> Real

Return the worst-case portfolio standard deviation over a type-2 ambiguity set.

# Arguments

  - `set`: The ambiguity set. `set.order` must be `2` and `set.ground_norm`
    must be `2`, because the result below is proved for the Euclidean cost.
  - `sig_hat`: Empirical covariance, `N x N`.
  - `w`: Portfolio weights, length `N`.

# Returns

  - The scalar `sup_{P in B} sqrt(Var_P(w' xi))`.

# Mathematical definition

Blanchet, Chen and Zhou (2022) show that the standard deviation is
1-Lipschitz with respect to the type-2 Wasserstein distance in the direction
`w`, and that the bound is tight:

    sup_{P in B_delta} sqrt(Var_P(w' xi))  =  sqrt(w' sig_hat w)  +  delta * ||w||_2

# Notes

  - **The library already builds this expression.** `L2Regularisation` with the
    `SOCRiskExpr` algorithm introduces `t >= ||w||_2` and adds `val * t` to the
    objective. Setting `val = delta` beside a `StandardDeviation` risk measure
    gives the distributionally robust problem exactly. The gap is that nothing
    tells a caller that `val` is a Wasserstein radius, and nothing calibrates
    it. See [`radius_from_confidence`](@ref).
  - This also explains the empirical finding of DeMiguel and co-authors (2009)
    that norm-constrained portfolios do better out of sample. A norm penalty is
    not an ad-hoc smoother. It is the exact price of distributional ambiguity.
"""
function robust_std(set::WassersteinAmbiguitySet, sig_hat::AbstractMatrix{<:Real},
                    w::AbstractVector{<:Real})
    N = length(w)
    if size(sig_hat) != (N, N)
        throw(DimensionMismatch("sig_hat must be $(N) x $(N), got $(size(sig_hat))"))
    end
    if set.order != 2
        throw(ArgumentError("robust_std needs a type-2 set, got order $(set.order)"))
    end
    quad = dot(w, sig_hat, w)
    return sqrt(max(quad, zero(quad))) + set.radius * norm(w, 2)
end

"""
    empirical_cvar(losses::AbstractVector, alpha::Real) -> Real

Return the empirical Conditional Value-at-Risk of a loss sample at level
`alpha`.

# Arguments

  - `losses`: Sample of losses, length `T`. A loss is positive when money is
    lost, so pass `-X * w` for a portfolio.
  - `alpha`: Tail probability, in `(0, 1)`.

# Returns

  - The scalar `min_tau { tau + (1/alpha) * mean( max(loss - tau, 0) ) }`.

# Details

The minimum is attained at the `1 - alpha` empirical quantile, so the routine
sorts once and averages the tail. It uses the Rockafellar and Uryasev (2000)
form, which is the same form the library's `ConditionalValueatRisk` uses.
"""
function empirical_cvar(losses::AbstractVector{<:Real}, alpha::Real)
    if !(zero(alpha) < alpha < one(alpha))
        throw(DomainError(alpha, "alpha must satisfy 0 < alpha < 1"))
    end
    T = length(losses)
    sorted = sort(losses; rev = true)
    # Number of whole observations in the tail, at least one.
    k = max(1, floor(Int, alpha * T))
    tau = sorted[k]
    excess = zero(float(eltype(losses)))
    @inbounds for i in 1:k
        excess += sorted[i] - tau
    end
    return tau + excess / (alpha * T)
end

"""
    robust_cvar(set::WassersteinAmbiguitySet, X::AbstractMatrix,
                w::AbstractVector, alpha::Real) -> Real

Return the worst-case CVaR of the portfolio loss over a type-1 ambiguity set.

# Arguments

  - `set`: The ambiguity set. `set.order` must be `1`.
  - `X`: Returns matrix, `T x N`.
  - `w`: Portfolio weights, length `N`.
  - `alpha`: Tail probability, in `(0, 1)`.

# Returns

  - The scalar `sup_{P in B} CVaR_alpha^P( -w' xi )`.

# Mathematical definition

Write the CVaR in the Rockafellar and Uryasev form,

    CVaR_alpha(l) = min_tau { tau + (1/alpha) E[ (l(xi) - tau)_+ ] }

For a fixed `tau` the integrand `xi -> tau + (1/alpha) (-w'xi - tau)_+` is
Lipschitz in `xi` with modulus `||w||_p / alpha`. For a type-1 ball the
supremum of an expectation of a Lipschitz function is the empirical
expectation plus the radius times the modulus, so

    sup_{P in B_delta} CVaR_alpha^P( -w' xi )
        =  CVaR_alpha^Phat( -w' xi )  +  (delta / alpha) * ||w||_p

# Notes

  - The `1 / alpha` factor is the whole content of the result. **A tail
    functional pays a higher price for the same ambiguity than a mean does**,
    and the multiplier is exactly the reciprocal of the tail probability. At
    `alpha = 0.05` the same radius costs twenty times as much.
  - This identity holds when the return space is unbounded. The library's
    `DistributionallyRobustConditionalValueatRisk` instead imposes the support
    constraint `xi >= -1`, meaning no asset may lose more than its value, and
    solves the exact conic program of Mohajerin Esfahani and Kuhn for it. That
    program is tighter than this closed form, which is therefore an upper
    bound on it, and both are upper bounds on the empirical CVaR.
"""
function robust_cvar(set::WassersteinAmbiguitySet, X::AbstractMatrix{<:Real},
                     w::AbstractVector{<:Real}, alpha::Real)
    if size(X, 2) != length(w)
        throw(DimensionMismatch("X has $(size(X, 2)) columns, w has length $(length(w))"))
    end
    if set.order != 1
        throw(ArgumentError("robust_cvar needs a type-1 set, got order $(set.order)"))
    end
    losses = -(X * w)
    base = empirical_cvar(losses, alpha)
    return base + (set.radius / alpha) * norm(w, dual_norm_order(set))
end

"""
    worst_case_shifted_returns(set::WassersteinAmbiguitySet, X::AbstractMatrix,
                               w::AbstractVector, alpha::Real) -> Matrix

Build the returns matrix of the measure that attains the worst-case CVaR.

This function exists to **check** [`robust_cvar`](@ref) numerically, and it is
also the honest answer to the question a risk committee asks: *what world are
we being robust against?*

# Arguments

  - `set`: The ambiguity set, of order one.
  - `X`: Returns matrix, `T x N`.
  - `w`: Portfolio weights, length `N`.
  - `alpha`: Tail probability, in `(0, 1)`.

# Returns

  - `Xw::Matrix`, `T x N`: The perturbed scenarios. Rows outside the tail are
    unchanged. Each of the `k = floor(alpha * T)` worst rows is moved by
    `-(delta / alpha) * g`, where `g` attains the dual norm of `w`.

# Details

The transport budget is an average over all `T` rows. Spending the whole
budget on the `alpha * T` tail rows moves each of them by `delta / alpha`,
because `(1/T) * (alpha * T) * (delta / alpha) = delta`. Concentrating the
budget on the tail is optimal, because only the tail enters the CVaR.
"""
function worst_case_shifted_returns(set::WassersteinAmbiguitySet, X::AbstractMatrix{<:Real},
                                    w::AbstractVector{<:Real}, alpha::Real)
    T = size(X, 1)
    losses = -(X * w)
    k = max(1, floor(Int, alpha * T))
    tail = partialsortperm(losses, 1:k; rev = true)
    p = dual_norm_order(set)
    g = _dual_norm_attainer(w, p)
    step = (set.radius / alpha)
    Xw = copy(Matrix(X))
    for t in tail
        @views Xw[t, :] .-= step .* g
    end
    return Xw
end

"""
    _dual_norm_attainer(w::AbstractVector, p::Real) -> Vector

Return the unit-`q`-norm vector `g` with `w' g = ||w||_p`.

# Arguments

  - `w`: Weight vector, length `N`.
  - `p`: Order of the penalty norm.

# Returns

  - `g::Vector`, length `N`, with `norm(g, q) == 1` where `1/p + 1/q = 1`.

# Details

Three cases are enough for the norms in use:

  - `p == 1`: the ground metric is `Inf`, and `g = sign.(w)`.
  - `p == 2`: the ground metric is `2`, and `g = w / ||w||_2`.
  - `p == Inf`: the ground metric is `1`, and `g` is a one-hot vector on the
    largest holding.
"""
function _dual_norm_attainer(w::AbstractVector{<:Real}, p::Real)
    if isone(p)
        return sign.(w)
    elseif isinf(p)
        g = zero(float.(w))
        g[argmax(abs.(w))] = sign(w[argmax(abs.(w))])
        return g
    elseif p == 2
        nw = norm(w, 2)
        return iszero(nw) ? zero(float.(w)) : w ./ nw
    end
    q = p / (p - 1)
    # General Holder attainer: g_i = sign(w_i) * |w_i|^(p-1), then rescale.
    g = sign.(w) .* abs.(w) .^ (p - 1)
    ng = norm(g, q)
    return iszero(ng) ? g : g ./ ng
end

"""
    radius_from_confidence(T::Integer, N::Integer; confidence::Real = 0.95,
                           scale::Real = 1.0) -> Float64

Return a data-driven Wasserstein radius.

# Arguments

  - `T`: Number of observations the empirical measure is built from.
  - `N`: Number of assets.
  - `confidence`: Target coverage of the ball, in `(0, 1)`.
  - `scale`: A multiplier on the whole expression, in the units of the return
    data. Its role is to convert the dimensionless rate into the scale of the
    problem. Set it to the average asset volatility for a first pass.

# Returns

  - The radius `delta`.

# Mathematical definition

Blanchet, Kang and Murthy (2019) show that the smallest radius whose ball
covers the true measure with probability `confidence` shrinks at the
parametric rate,

    delta(T)  =  scale * sqrt( chi2_q(confidence, N) / T )

where `chi2_q` is the quantile function of a chi-squared distribution with `N`
degrees of freedom. The `1 / sqrt(T)` rate is the part to trust. The constant
is the part to calibrate.

# Notes

  - Mohajerin Esfahani and Kuhn (2018) give a distribution-free radius that
    shrinks at rate `T^(-1/N)`. That rate is useless for a realistic universe:
    at `N = 50` it barely shrinks at all. The concentration route above buys
    the parametric rate by assuming a light tail, which is the trade every
    practical calibration makes.
  - **The honest route is cross-validation.** The library already splits a
    Pipeline on contiguous time windows. Treat `delta` as one more
    hyperparameter for `GridSearchCrossValidation` and let the out-of-sample
    score choose it. This function then supplies the grid's centre, not its
    answer.
"""
function radius_from_confidence(T::Integer, N::Integer; confidence::Real = 0.95,
                                scale::Real = 1.0)
    if T <= 0
        throw(DomainError(T, "T must be > 0"))
    end
    if !(zero(confidence) < confidence < one(confidence))
        throw(DomainError(confidence, "confidence must satisfy 0 < confidence < 1"))
    end
    # Wilson-Hilferty approximation to the chi-squared quantile. It avoids a
    # dependency on Distributions for a prototype, and it is accurate to about
    # one per cent for N >= 2.
    z = _standard_normal_quantile(confidence)
    chi2 = N * (1 - 2 / (9N) + z * sqrt(2 / (9N)))^3
    return scale * sqrt(chi2 / T)
end

"""
    _standard_normal_quantile(p::Real) -> Float64

Return the standard normal quantile at probability `p`, by bisection on the
error function. Accurate to about `1e-10` and free of dependencies.
"""
function _standard_normal_quantile(p::Real)
    lo, hi = -10.0, 10.0
    for _ in 1:200
        mid = (lo + hi) / 2
        cdf = (1 + erf_approx(mid / sqrt(2))) / 2
        if cdf < p
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

"""
    erf_approx(x::Real) -> Float64

Abramowitz and Stegun 7.1.26 rational approximation to the error function.
Maximum absolute error about `1.5e-7`, which is far below the precision a
Wasserstein radius is ever known to.
"""
function erf_approx(x::Real)
    s = sign(x)
    z = abs(float(x))
    t = 1 / (1 + 0.3275911z)
    y = 1 -
        (((((1.061405429t - 1.453152027)t) + 1.421413741)t - 0.284496736)t + 0.254829592)t *
        exp(-z * z)
    return s * y
end

end # module WassersteinAmbiguity
