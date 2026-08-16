# =============================================================================
# Prototype 4 — Calibration against a known truth.
#
# Purpose
#   Cross-validation asks: would this method have worked on the history I have?
#   It cannot ask: does this method recover the right answer when I know what
#   the right answer is? The second question needs a simulated world, and the
#   library cannot build one. This file builds it.
#
#   The protocol is a second evaluation protocol beside the Holdout Split and
#   Cross-Validation, and it reuses the same scoring idea:
#
#     1. Fix a true `(mu, sigma)`. This is the world.
#     2. Draw `T` observations from it. This is the sample a caller would see.
#     3. Fit the moment estimator under test on the sample.
#     4. Optimise on the fitted moments. This is the portfolio.
#     5. Score the portfolio under the TRUE moments, against the portfolio the
#        truth itself implies.
#     6. Repeat, and report the distribution.
#
#   Step 5 is what history cannot give. It separates the error of the method
#   from the noise of one realised path.
#
# Status
#   Standalone. Depends on `LinearAlgebra`, `Statistics` and `Random`.
#
# Notation used throughout this file
#   N        Number of assets.
#   T        Number of observations drawn per trial.
#   K        Number of Monte Carlo trials.
#   mu_t     True expected returns, length `N`.
#   sig_t    True covariance, `N x N`.
#   mu_h     Estimated expected returns from one sample.
#   sig_h    Estimated covariance from one sample.
#   w_star   Oracle portfolio, computed from the true moments.
#   w_hat    Estimated portfolio, computed from the fitted moments.
#
# Sources
#   Jorion, P. (1986). Bayes-Stein estimation for portfolio analysis. Journal
#     of Financial and Quantitative Analysis 21(3), 279-292. The mean shrinkage
#     estimator implemented below.
#   Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for
#     large-dimensional covariance matrices. Journal of Multivariate Analysis
#     88(2), 365-411. The covariance shrinkage estimator implemented below.
#   Kan, R. and Zhou, G. (2007). Optimal portfolio choice with parameter
#     uncertainty. Journal of Financial and Quantitative Analysis 42(3),
#     621-656. The expected utility loss from estimation error.
#   DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal versus naive
#     diversification: how inefficient is the 1/N portfolio strategy? Review of
#     Financial Studies 22(5), 1915-1953. The benchmark this protocol exists to
#     make reproducible.
#   Michaud, R. O. (1989). The Markowitz optimization enigma: is optimized
#     optimal? Financial Analysts Journal 45(1), 31-42. Error maximisation.
# =============================================================================
module SimulatedTruthCalibration

using LinearAlgebra, Statistics, Random

export oracle_max_sharpe, oracle_min_variance, sharpe_ratio, sample_moments,
       ledoit_wolf_identity, bayes_stein_mean, calibration_study

"""
    sharpe_ratio(w::AbstractVector, mu::AbstractVector, sig::AbstractMatrix) -> Real

Return the Sharpe ratio of a portfolio, with a zero risk-free rate.

# Arguments

  - `w`: Weights, length `N`.
  - `mu`: Expected returns, length `N`.
  - `sig`: Covariance, `N x N`.

# Returns

  - The scalar `w' mu / sqrt(w' sig w)`, or `NaN` when the variance is zero.

# Mathematical definition

    SR(w) = (w' mu) / sqrt(w' sig w)

The ratio is scale invariant in `w`, so a normalisation of the weights does not
change it. That property is what lets an unconstrained oracle be compared
against a budget-constrained estimate.
"""
function sharpe_ratio(w::AbstractVector{<:Real}, mu::AbstractVector{<:Real},
                      sig::AbstractMatrix{<:Real})
    var = dot(w, sig, w)
    return var > 0 ? dot(w, mu) / sqrt(var) : NaN
end

"""
    oracle_max_sharpe(mu::AbstractVector, sig::AbstractMatrix) -> Vector

Return the maximum-Sharpe portfolio, normalised to a unit budget.

# Arguments

  - `mu`: Expected returns, length `N`.
  - `sig`: Covariance, `N x N`, positive definite.

# Returns

  - `w::Vector`, length `N`, with `sum(w) == 1`.

# Mathematical definition

The tangency portfolio solves `max_w (w' mu) / sqrt(w' sig w)` and, because the
objective is scale free, has the closed form

    w  proportional to  sig^(-1) mu

The normalisation divides by `1' sig^(-1) mu`. **The result may hold short
positions and may be extremely levered.** That is not a defect of the code: it
is the property that makes the mean-variance problem an error maximiser, and
this protocol exists to measure it.
"""
function oracle_max_sharpe(mu::AbstractVector{<:Real}, sig::AbstractMatrix{<:Real})
    raw = Symmetric(sig) \ collect(mu)
    s = sum(raw)
    return iszero(s) ? raw : raw ./ s
end

"""
    oracle_min_variance(sig::AbstractMatrix) -> Vector

Return the global minimum-variance portfolio, normalised to a unit budget.

# Arguments

  - `sig`: Covariance, `N x N`, positive definite.

# Returns

  - `w::Vector`, length `N`, with `sum(w) == 1`.

# Mathematical definition

    w  =  sig^(-1) 1  /  (1' sig^(-1) 1)

This portfolio uses no expected return at all, which is why it is the standard
control in an estimation-error study. Any gap between it and a mean-variance
portfolio is the value the mean estimate added, net of the error it introduced.
"""
function oracle_min_variance(sig::AbstractMatrix{<:Real})
    N = size(sig, 1)
    raw = Symmetric(sig) \ ones(eltype(sig), N)
    return raw ./ sum(raw)
end

"""
    sample_moments(X::AbstractMatrix) -> Tuple{Vector, Matrix}

Return the plain sample mean and sample covariance of `X`.

# Arguments

  - `X`: Returns, `T x N`.

# Returns

  - `(mu, sig)`: The sample mean, length `N`, and the unbiased sample
    covariance, `N x N`.
"""
function sample_moments(X::AbstractMatrix{<:Real})
    return vec(mean(X; dims = 1)), Matrix(cov(X))
end

"""
    ledoit_wolf_identity(X::AbstractMatrix) -> Tuple{Vector, Matrix}

Return the sample mean, and the covariance shrunk towards a scaled identity
with the Ledoit and Wolf (2004) optimal intensity.

# Arguments

  - `X`: Returns, `T x N`.

# Returns

  - `(mu, sig)`: The sample mean, and the shrunk covariance, `N x N`.

# Mathematical definition

The target is the scaled identity `F = m * I` with `m = trace(S) / N`, the
average eigenvalue of the sample covariance `S`. The estimator is the convex
combination

    sig  =  beta * F  +  (1 - beta) * S,        beta = min(1, b2 / d2)

with

    d2  =  || S - F ||_F^2  /  N
    b2  =  min( d2 ,  (1 / (N * T^2)) * sum_t || x_t x_t' - S ||_F^2 )

where `|| . ||_F` is the Frobenius norm and `x_t` is the `t`-th centred
observation. `d2` measures how far the sample covariance is from the target.
`b2` measures how noisy the sample covariance is. The intensity is the ratio,
so a noisy estimate in a small sample shrinks hard and a precise one in a large
sample barely moves.

# Notes

  - The library already ships this family. See
    `src/08_Moments/03_Covariance.jl` and the `CovarianceEstimation` package it
    interoperates with. The estimator is reproduced here so the prototype
    stays standalone, not because it is absent.
"""
function ledoit_wolf_identity(X::AbstractMatrix{<:Real})
    T, N = size(X)
    mu = vec(mean(X; dims = 1))
    Xc = X .- transpose(mu)
    S = (transpose(Xc) * Xc) ./ T
    m = tr(S) / N
    F = m * Matrix{eltype(S)}(I, N, N)
    d2 = sum(abs2, S .- F) / N
    b2_sum = zero(float(eltype(S)))
    for t in 1:T
        xt = view(Xc, t, :)
        b2_sum += sum(abs2, xt * transpose(xt) .- S)
    end
    b2 = min(d2, b2_sum / (N * T^2))
    beta = d2 > 0 ? min(one(d2), b2 / d2) : zero(d2)
    return mu, beta .* F .+ (1 - beta) .* S
end

"""
    bayes_stein_mean(X::AbstractMatrix) -> Tuple{Vector, Matrix}

Return the Jorion (1986) Bayes-Stein shrunk mean, with the sample covariance.

# Arguments

  - `X`: Returns, `T x N`.

# Returns

  - `(mu, sig)`: The shrunk mean, length `N`, and the sample covariance.

# Mathematical definition

Shrink the sample mean `mu_h` towards the scalar `mu_0`, the mean of the global
minimum-variance portfolio:

    mu_0  =  (1' S^(-1) mu_h) / (1' S^(-1) 1)

    nu    =  (N + 2) / ( (N + 2) + T * (mu_h - mu_0 1)' S^(-1) (mu_h - mu_0 1) )

    mu    =  (1 - nu) * mu_h  +  nu * mu_0 * 1

The intensity `nu` falls as the sample grows and as the cross-sectional spread
of the means grows. It rises with the number of assets, which is the whole
content of the Stein effect: **the more means a caller estimates at once, the
less each one should be trusted.**

# Notes

  - The library has this family too, as `ShrunkExpectedReturns` in
    `src/08_Moments/16_ShrunkExpectedReturns.jl`.
"""
function bayes_stein_mean(X::AbstractMatrix{<:Real})
    T, N = size(X)
    mu_h = vec(mean(X; dims = 1))
    S = Matrix(cov(X))
    Sinv_one = Symmetric(S) \ ones(eltype(S), N)
    mu_0 = dot(ones(eltype(S), N), Symmetric(S) \ mu_h) / sum(Sinv_one)
    d = mu_h .- mu_0
    quad = dot(d, Symmetric(S) \ d)
    nu = (N + 2) / ((N + 2) + T * max(quad, zero(quad)))
    return (1 - nu) .* mu_h .+ nu .* mu_0, S
end

"""
    calibration_study(mu_t::AbstractVector, sig_t::AbstractMatrix;
                      estimators, rule = oracle_max_sharpe, T::Integer = 250,
                      n_trials::Integer = 500,
                      rng::Random.AbstractRNG = Random.default_rng()) -> Vector

Run the Monte Carlo protocol and return one summary per estimator.

# Arguments

  - `mu_t`: True expected returns, length `N`. The world.
  - `sig_t`: True covariance, `N x N`. The world.
  - `estimators`: A vector of `name => f` pairs, where `f(X)` returns the tuple
    `(mu_h, sig_h)` fitted on a sample `X` of size `T x N`.
  - `rule`: The optimiser under test. It takes `(mu_h, sig_h)` and returns
    weights. Default is [`oracle_max_sharpe`](@ref), applied to the *estimated*
    moments, which makes it the plain mean-variance rule.
  - `T`: Observations per trial.
  - `n_trials`: Number of Monte Carlo trials, `K`.
  - `rng`: Random number generator.

# Returns

A vector of `NamedTuple`s, one per estimator, each with:

  - `name`: The estimator label.
  - `sharpe_mean`, `sharpe_sd`: The realised Sharpe ratio **under the true
    moments**, averaged over trials.
  - `sharpe_loss`: `SR(w_star) - mean(SR(w_hat))`, the price of not knowing the
    truth. It cannot be negative in expectation, because `w_star` maximises the
    Sharpe ratio under the true moments by construction.
  - `risk_inflation`: `mean( sqrt(w_hat' sig_t w_hat) ) / sqrt(w_star' sig_t w_star)`. How much more risk the estimated portfolio actually runs than it
    was asked to.
  - `active_share_from_oracle`: `mean( 0.5 * ||w_hat - w_star||_1 )`. How far
    the estimated portfolio sits from the right answer.
  - `gross_leverage`: `mean( ||w_hat||_1 )`. One for a long-only fully invested
    portfolio, and much larger when the rule maximises error.

# Notes

  - The oracle Sharpe ratio is the ceiling of the study. **No estimator can
    beat it, and an estimator that appears to has a bug**, most often a leak of
    the true moments into the fit. Assert it in a test.
"""
function calibration_study(mu_t::AbstractVector{<:Real}, sig_t::AbstractMatrix{<:Real};
                           estimators::AbstractVector, rule::Function = oracle_max_sharpe,
                           T::Integer = 250, n_trials::Integer = 500,
                           rng::Random.AbstractRNG = Random.default_rng())
    N = length(mu_t)
    if size(sig_t) != (N, N)
        throw(DimensionMismatch("sig_t must be $(N) x $(N), got $(size(sig_t))"))
    end
    w_star = rule(mu_t, sig_t)
    sr_star = sharpe_ratio(w_star, mu_t, sig_t)
    risk_star = sqrt(dot(w_star, sig_t, w_star))
    L = cholesky(Symmetric(sig_t)).L
    out = Vector{NamedTuple}(undef, length(estimators))
    for (k, (name, fit)) in enumerate(estimators)
        srs = Vector{Float64}(undef, n_trials)
        risks = Vector{Float64}(undef, n_trials)
        as = Vector{Float64}(undef, n_trials)
        lev = Vector{Float64}(undef, n_trials)
        for trial in 1:n_trials
            X = randn(rng, T, N) * transpose(L) .+ transpose(mu_t)
            mu_h, sig_h = fit(X)
            w_hat = rule(mu_h, sig_h)
            srs[trial] = sharpe_ratio(w_hat, mu_t, sig_t)
            risks[trial] = sqrt(max(dot(w_hat, sig_t, w_hat), 0.0))
            as[trial] = sum(abs, w_hat .- w_star) / 2
            lev[trial] = sum(abs, w_hat)
        end
        out[k] = (; name = String(name), sharpe_mean = mean(srs), sharpe_sd = std(srs),
                  sharpe_loss = sr_star - mean(srs),
                  risk_inflation = mean(risks) / risk_star,
                  active_share_from_oracle = mean(as), gross_leverage = mean(lev),
                  oracle_sharpe = sr_star)
    end
    return out
end

end # module SimulatedTruthCalibration
