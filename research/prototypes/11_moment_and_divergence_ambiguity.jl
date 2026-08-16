# =============================================================================
# Prototype 11 — Moment and divergence ambiguity sets.
#
# Purpose
#   Prototype 2 supplies the Wasserstein ball, which bounds the *measure*.
#   Reports 3 and 4 also ask for two other ambiguity families, and they answer
#   genuinely different questions:
#
#     * A **Gelbrich** (moment) ball bounds the mean and the covariance and
#       says nothing about anything else. It is the right set when a caller
#       trusts the shape of the distribution but not its two moments, which is
#       the usual situation with a factor model.
#     * A **divergence** ball bounds the likelihood ratio against a reference
#       measure. It can only reweight scenarios the caller already has, so it
#       can never invent a new event. That makes it the conservative choice
#       when the scenario set is rich, and useless when it is not.
#
#   The three sets answer, respectively: what if the data moves, what if the
#   moments are wrong, and what if the probabilities are wrong. **A caller
#   should be able to say which they mean.** Today the library offers none of
#   the three under those names.
#
# Status
#   Standalone. Depends on `LinearAlgebra`, `Statistics` and `LogExpFunctions`.
#
# Notation used throughout this file
#   N        Number of assets.
#   T        Number of scenarios.
#   mu, sig  A moment pair: mean (length `N`) and covariance (`N x N`).
#   rho      Radius of a Gelbrich ball, in return units.
#   w        Portfolio weights, length `N`.
#   L        Per-scenario losses, length `T`. Positive means money lost.
#   p        Reference scenario probabilities, length `T`.
#   eta      Radius of a divergence ball, in nats.
#   theta    The dual variable of a divergence ball. It has units of loss and
#            is the risk-aversion parameter of an entropic risk measure.
#
# Sources
#   Gelbrich, M. (1990). On a formula for the L2 Wasserstein metric between
#     measures on Euclidean and Hilbert spaces. Mathematische Nachrichten
#     147(1), 185-203. The distance implemented here.
#   Bhatia, R., Jain, T. and Lim, Y. (2019). On the Bures-Wasserstein distance
#     between positive definite matrices. Expositiones Mathematicae 37(2),
#     165-191.
#   Nguyen, V. A., Kuhn, D. and Mohajerin Esfahani, P. (2022).
#     Distributionally robust inverse covariance estimation: the Wasserstein
#     shrinkage estimator. Operations Research 70(1), 490-515. The moment-ball
#     reading used here.
#   Delage, E. and Ye, Y. (2010). Distributionally robust optimization under
#     moment uncertainty with application to data-driven problems. Operations
#     Research 58(3), 595-612. The original moment-ambiguity portfolio result.
#   Ben-Tal, A., den Hertog, D., De Waegenaere, A., Melenberg, B. and Rennen,
#     G. (2013). Robust solutions of optimization problems affected by
#     uncertain probabilities. Management Science 59(2), 341-357. The
#     phi-divergence family.
#   Hansen, L. P. and Sargent, T. J. (2008). Robustness. Princeton University
#     Press. The economic reading of the entropic dual.
#   Follmer, H. and Schied, A. (2011). Stochastic Finance: An Introduction in
#     Discrete Time, 3rd edition. De Gruyter. The entropic risk measure and its
#     convex-duality representation.
# =============================================================================
module MomentDivergenceAmbiguity

using LinearAlgebra, Statistics, LogExpFunctions

export gelbrich_distance, bures_wasserstein, worst_case_moments, gelbrich_worst_case_std,
       gelbrich_worst_case_mean, entropic_risk, kl_worst_case_expectation,
       kl_worst_case_measure, divergence_of_tilt

# -----------------------------------------------------------------------------
# Moment ambiguity: the Gelbrich ball
# -----------------------------------------------------------------------------
"""
    bures_wasserstein(S1::AbstractMatrix, S2::AbstractMatrix) -> Real

Return the Bures-Wasserstein distance between two covariance matrices.

# Arguments

  - `S1`, `S2`: Symmetric positive semi-definite matrices, `N x N`.

# Returns

  - The non-negative scalar `d(S1, S2)`.

# Mathematical definition

    d(S1, S2)^2  =  trace( S1 + S2 - 2 ( S1^{1/2} S2 S1^{1/2} )^{1/2} )

This is the squared 2-Wasserstein distance between two centred Gaussians with
these covariances. It is a genuine metric on the cone of positive
semi-definite matrices.

# Notes

  - **When the two matrices commute it collapses to a Frobenius norm**,
    `d = norm(S1^{1/2} - S2^{1/2}, 2)`. Diagonal matrices always commute, so
    for uncorrelated assets the distance is just the Euclidean distance
    between the vectors of standard deviations. That special case is the one
    to build intuition on, and the routine's test asserts it.
  - The matrix square roots are computed by eigen-decomposition with negative
    eigenvalues clipped to zero, so a numerically indefinite input is
    tolerated rather than rejected.
  - **The formula loses half its digits near zero.** The trace expression
    cancels to about `1e-14` when the two matrices are equal, and the square
    root turns that into about `1e-7`. Measured: `d(S, S)` returns `1.2e-7`
    rather than zero. Compare the **squared** distance against a squared
    tolerance when testing for equality, and never treat a value below `1e-6`
    as meaningfully non-zero.
"""
function bures_wasserstein(S1::AbstractMatrix{<:Real}, S2::AbstractMatrix{<:Real})
    N = size(S1, 1)
    if size(S1) != size(S2)
        throw(DimensionMismatch("S1 is $(size(S1)), S2 is $(size(S2))"))
    end
    A = _psd_sqrt(S1)
    inner = _psd_sqrt(Symmetric(A * S2 * A))
    val = tr(S1) + tr(S2) - 2 * tr(inner)
    return sqrt(max(val, zero(val)))
end

"""
    _psd_sqrt(S::AbstractMatrix) -> Matrix

Return the symmetric positive semi-definite square root of `S`, with negative
eigenvalues clipped to zero.
"""
function _psd_sqrt(S::AbstractMatrix{<:Real})
    Ssym = Symmetric((S .+ transpose(S)) ./ 2)
    vals, vecs = eigen(Ssym)
    return vecs * Diagonal(sqrt.(max.(vals, zero(eltype(vals))))) * transpose(vecs)
end

"""
    gelbrich_distance(mu1::AbstractVector, S1::AbstractMatrix,
                      mu2::AbstractVector, S2::AbstractMatrix) -> Real

Return the Gelbrich distance between two moment pairs.

# Arguments

  - `mu1`, `S1`: The first mean and covariance.
  - `mu2`, `S2`: The second.

# Returns

  - The non-negative scalar `G`.

# Mathematical definition

    G^2  =  || mu1 - mu2 ||_2^2  +  d_BW(S1, S2)^2

Gelbrich (1990) proved that this is a **lower bound** on the 2-Wasserstein
distance between *any* two distributions with these moments, and that it is
attained by the Gaussians. So the ball

    { P : G( moments(P), (mu_hat, sig_hat) ) <= rho }

contains the Wasserstein ball of the same radius. It is the larger, and
therefore more conservative, set.

# Notes

  - **The two radii are not interchangeable.** A Gelbrich radius and a
    Wasserstein radius of the same number describe different sets, and the
    Gelbrich one is weaker. Report which was used.
  - `G` is a metric: symmetric, zero only when the moment pairs agree, and it
    satisfies the triangle inequality. The verification driver asserts all
    three.
"""
function gelbrich_distance(mu1::AbstractVector{<:Real}, S1::AbstractMatrix{<:Real},
                           mu2::AbstractVector{<:Real}, S2::AbstractMatrix{<:Real})
    if length(mu1) != length(mu2)
        throw(DimensionMismatch("mu1 has length $(length(mu1)), mu2 has length $(length(mu2))"))
    end
    dm = sum(abs2, collect(mu1) .- collect(mu2))
    db = bures_wasserstein(S1, S2)
    return sqrt(dm + db^2)
end

"""
    gelbrich_worst_case_std(S::AbstractMatrix, w::AbstractVector, rho::Real) -> Real

Return the largest portfolio standard deviation over a Gelbrich ball.

# Arguments

  - `S`: Nominal covariance, `N x N`.
  - `w`: Portfolio weights, length `N`.
  - `rho`: Ball radius.

# Returns

  - `sqrt(w' S w) + rho * norm(w, 2)`.

# Mathematical definition

    sup { sqrt( w' S' w ) : d_BW(S', S) <= rho }  =  sqrt(w' S w)  +  rho ||w||_2

The bound follows because `S -> sqrt(w' S w)` is 1-Lipschitz with respect to
the Bures-Wasserstein metric in the direction `w`, and it is attained.

# Notes

  - **This is numerically identical to the type-2 Wasserstein result in
    prototype 2**, which is not a coincidence: for elliptical families the two
    balls give the same worst-case second moment. The sets differ, the answer
    for this particular functional does not. Say "moment ambiguity" when the
    doubt is about the estimator and "Wasserstein" when it is about the data.
  - So the same conclusion applies: `L2Regularisation` already emits this
    term. See prototype 2, Finding A.
"""
function gelbrich_worst_case_std(S::AbstractMatrix{<:Real}, w::AbstractVector{<:Real},
                                 rho::Real)
    if rho < 0
        throw(DomainError(rho, "rho must be >= 0"))
    end
    q = dot(w, S, w)
    return sqrt(max(q, zero(q))) + rho * norm(w, 2)
end

"""
    gelbrich_worst_case_mean(mu::AbstractVector, w::AbstractVector, rho::Real) -> Real

Return the smallest portfolio expected return over a Gelbrich ball.

# Mathematical definition

    inf { mu'' w : || mu'' - mu ||_2 <= rho }  =  mu' w  -  rho ||w||_2

by Cauchy-Schwarz, attained at `mu'' = mu - rho w / ||w||_2`.

# Notes

  - A Gelbrich ball of radius `rho` spends its budget between the mean and the
    covariance, because `G^2` adds the two. Bounding both at the full radius
    simultaneously, as a caller who calls both functions with the same `rho`
    does, is **conservative but not tight**. State that when reporting it.
"""
function gelbrich_worst_case_mean(mu::AbstractVector{<:Real}, w::AbstractVector{<:Real},
                                  rho::Real)
    if rho < 0
        throw(DomainError(rho, "rho must be >= 0"))
    end
    return dot(mu, w) - rho * norm(w, 2)
end

"""
    worst_case_moments(mu::AbstractVector, S::AbstractMatrix, w::AbstractVector,
                       rho::Real; split::Real = 0.5) -> NamedTuple

Return an explicit moment pair inside the Gelbrich ball that is adverse for
the portfolio `w`.

# Arguments

  - `mu`, `S`: The nominal moments.
  - `w`: The portfolio the adversary is targeting.
  - `rho`: Ball radius.
  - `split`: Fraction of the squared radius spent on the mean, in `[0, 1]`.
    The remainder goes to the covariance.

# Returns

A `NamedTuple` with `mu_bad`, `S_bad`, `distance` (which must not exceed
`rho`), and the resulting `mean` and `std` of the portfolio.

# Details

The mean moves against the portfolio along `-w / ||w||`, spending radius
`rho_m = rho sqrt(split)`. The covariance is inflated along the same
direction, `S_bad = (A + rho_c u u')^2` with `A = S^{1/2}` and
`u = A w / ||A w||`, spending radius `rho_c = rho sqrt(1 - split)`.

**This is the answer to "what world are we protecting against?"** A radius is
abstract. A covariance matrix a caller can look at is not.

# Notes

  - The construction is *a* point in the ball that hurts, not necessarily the
    exact maximiser of a chosen functional. The routine reports its own
    `distance`, so the claim can be checked rather than assumed.
"""
function worst_case_moments(mu::AbstractVector{<:Real}, S::AbstractMatrix{<:Real},
                            w::AbstractVector{<:Real}, rho::Real; split::Real = 0.5)
    if !(0 <= split <= 1)
        throw(DomainError(split, "split must lie in [0, 1]"))
    end
    if rho < 0
        throw(DomainError(rho, "rho must be >= 0"))
    end
    nw = norm(w, 2)
    rho_m = rho * sqrt(split)
    rho_c = rho * sqrt(1 - split)
    mu_bad = iszero(nw) ? collect(float.(mu)) : collect(float.(mu)) .- rho_m .* (w ./ nw)
    A = _psd_sqrt(S)
    Aw = A * w
    nAw = norm(Aw)
    B = iszero(nAw) ? A : A .+ rho_c .* ((Aw ./ nAw) * transpose(Aw ./ nAw))
    S_bad = Symmetric(B * B)
    return (; mu_bad = mu_bad, S_bad = Matrix(S_bad),
            distance = gelbrich_distance(mu_bad, S_bad, mu, S), mean = dot(mu_bad, w),
            std = sqrt(max(dot(w, S_bad, w), 0.0)))
end

# -----------------------------------------------------------------------------
# Divergence ambiguity: the Kullback-Leibler ball
# -----------------------------------------------------------------------------
"""
    entropic_risk(L::AbstractVector, theta::Real; p = nothing) -> Real

Return the entropic risk measure of a loss sample.

# Arguments

  - `L`: Losses, length `T`. Positive means money lost.
  - `theta`: Risk tolerance, positive. Large means nearly risk neutral.
  - `p`: Reference probabilities, length `T`. Uniform if absent.

# Returns

  - The scalar `theta * log( E_p[ exp(L / theta) ] )`.

# Mathematical definition

    rho_theta(L)  =  theta * log  sum_t p_t exp( L_t / theta )

# Notes

  - As `theta -> infinity` this tends to `E_p[L]`, the risk-neutral value. As
    `theta -> 0` it tends to `max_t L_t`, the worst case. **The parameter
    interpolates the whole range between the mean and the maximum**, which is
    what makes it the natural dual variable for a divergence ball.
  - It is computed in log space, so an exponent of a large loss does not
    overflow. That matters: `exp(L / theta)` overflows for small `theta` in
    any naive implementation.
  - The measure is convex and translation invariant but **not** coherent: it
    fails positive homogeneity. That is a deliberate property, not a defect,
    and it is why it prices a doubling of position size at more than double
    the risk.
"""
function entropic_risk(L::AbstractVector{<:Real}, theta::Real;
                       p::Union{Nothing, AbstractVector{<:Real}} = nothing)
    if !(theta > 0)
        throw(DomainError(theta, "theta must be > 0"))
    end
    T = length(L)
    logp = isnothing(p) ? fill(-log(float(T)), T) : log.(p)
    return theta * logsumexp(logp .+ L ./ theta)
end

"""
    kl_worst_case_expectation(L::AbstractVector, eta::Real;
                              p = nothing, theta_lo::Real = 1e-8,
                              theta_hi::Real = 1e8, iters::Integer = 300)
        -> NamedTuple

Return the largest expected loss over a Kullback-Leibler ball.

# Arguments

  - `L`: Losses, length `T`.
  - `eta`: Ball radius in nats. Zero returns the reference expectation.
  - `p`: Reference probabilities. Uniform if absent.
  - `theta_lo`, `theta_hi`, `iters`: Controls for the one-dimensional search.

# Returns

A `NamedTuple` with `value`, `theta` (the minimiser), and `q` (the worst-case
probabilities).

# Mathematical definition

The primal problem is

    sup { E_q[L]  :  D(q || p) <= eta,  sum q = 1,  q >= 0 }

By the Donsker-Varadhan variational formula and Lagrangian duality it equals a
**one-dimensional** convex programme:

    sup_q E_q[L]  =  min_{theta > 0}  { theta log E_p[ exp(L / theta) ]  +  theta * eta }
                  =  min_{theta > 0}  { rho_theta(L)  +  theta * eta }

and the maximising measure is the exponential tilt

    q_t  proportional to  p_t exp( L_t / theta* )

# Notes

  - **The whole problem collapses to a scalar minimisation**, whatever `T` is.
    That is the same collapse prototype 5 exploits for the breakeven view, and
    for the same reason: a Kullback-Leibler constraint has an exponential
    family as its solution set.
  - The objective is convex in `theta`, so a golden-section or bisection search
    on the derivative is reliable. This implementation uses a golden-section
    search on the value, which needs no derivative.
  - **A divergence ball can only reweight scenarios that exist.** If every
    scenario in `L` is mild, no `eta` produces a severe worst case. That is the
    structural difference from a Wasserstein ball, which can move a scenario to
    a value never observed. Choose accordingly: divergence for "my
    probabilities are wrong", Wasserstein for "my data is incomplete".
"""
function kl_worst_case_expectation(L::AbstractVector{<:Real}, eta::Real;
                                   p::Union{Nothing, AbstractVector{<:Real}} = nothing,
                                   theta_lo::Real = 1e-8, theta_hi::Real = 1e8,
                                   iters::Integer = 300)
    if eta < 0
        throw(DomainError(eta, "eta must be >= 0"))
    end
    T = length(L)
    pv = isnothing(p) ? fill(one(float(eltype(L))) / T, T) : collect(float.(p))
    if iszero(eta)
        return (; value = dot(pv, L), theta = Inf, q = pv)
    end
    obj(th) = entropic_risk(L, th; p = pv) + th * eta
    # Golden-section search on a convex univariate function.
    gr = (sqrt(5.0) - 1) / 2
    a, b = float(theta_lo), float(theta_hi)
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    for _ in 1:iters
        if obj(c) < obj(d)
            b = d
        else
            a = c
        end
        c = b - gr * (b - a)
        d = a + gr * (b - a)
    end
    th = (a + b) / 2
    q = kl_worst_case_measure(L, th; p = pv)
    return (; value = obj(th), theta = th, q = q)
end

"""
    kl_worst_case_measure(L::AbstractVector, theta::Real; p = nothing) -> Vector

Return the exponential tilt `q_t ∝ p_t exp(L_t / theta)`.

# Notes

  - Computed in log space for the same overflow reason as
    [`entropic_risk`](@ref).
"""
function kl_worst_case_measure(L::AbstractVector{<:Real}, theta::Real;
                               p::Union{Nothing, AbstractVector{<:Real}} = nothing)
    if !(theta > 0)
        throw(DomainError(theta, "theta must be > 0"))
    end
    T = length(L)
    logp = isnothing(p) ? fill(-log(float(T)), T) : log.(p)
    s = logp .+ L ./ theta
    return exp.(s .- logsumexp(s))
end

"""
    divergence_of_tilt(L::AbstractVector, theta::Real; p = nothing) -> Real

Return `D(q_theta || p)` for the exponential tilt at `theta`.

# Details

Useful as a consistency check: at the optimal `theta` returned by
[`kl_worst_case_expectation`](@ref), this must equal the radius `eta`, because
the constraint binds whenever `eta` is small enough to matter. The
verification driver asserts it.
"""
function divergence_of_tilt(L::AbstractVector{<:Real}, theta::Real;
                            p::Union{Nothing, AbstractVector{<:Real}} = nothing)
    T = length(L)
    pv = isnothing(p) ? fill(one(float(eltype(L))) / T, T) : collect(float.(p))
    q = kl_worst_case_measure(L, theta; p = pv)
    s = zero(float(eltype(L)))
    @inbounds for t in eachindex(q)
        if q[t] > 0
            s += q[t] * log(q[t] / pv[t])
        end
    end
    return s
end

end # module MomentDivergenceAmbiguity
