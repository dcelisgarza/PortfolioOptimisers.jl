# =============================================================================
# Prototype 21 — Regime models: a hidden Markov driver for the whole panel.
#
# Purpose
#   Reports 2, 4 and 7 all ask for regimes. The library already has
#   *regime-adjusted* moments, in `src/08_Moments/36_*` and `37_*`. What it
#   does not have is a **regime model**: an object that infers the state, gives
#   the probability of being in each state, and lets every downstream quantity
#   be conditioned on it.
#
#   The design that works is asymmetric, and that asymmetry is the main idea
#   here:
#
#     * The **state** is inferred from one low-dimensional driver series, such
#       as the market return or a volatility index. A hidden Markov model on
#       one series is well identified with a few hundred observations.
#     * The **moments** are then estimated for the whole panel, weighted by the
#       smoothed state probabilities. A separate model per asset would need far
#       more data than exists.
#
#   Fitting a `K`-state model directly on `N` assets needs `K N (N + 3) / 2`
#   parameters and never converges to anything meaningful. Fitting it on one
#   series needs `3K + K^2` and does.
#
# Status
#   Standalone. Depends on `LinearAlgebra`, `Statistics`, `Random` and
#   `LogExpFunctions`.
#
# Notation used throughout this file
#   T       Number of observations.
#   K       Number of hidden states.
#   N       Number of assets.
#   y       The driver series, length `T`.
#   X       The asset panel, `T x N`.
#   P       Transition matrix, `K x K`. `P[i, j]` is `Pr(s_t = j | s_{t-1} = i)`.
#   pi0     Initial state distribution, length `K`.
#   mu_k    Driver mean in state `k`. `sd_k` its standard deviation.
#   gam     Smoothed state probabilities, `T x K`. `gam[t, k] = Pr(s_t = k | y)`.
#
# Sources
#   Baum, L. E., Petrie, T., Soules, G. and Weiss, N. (1970). A maximization
#     technique occurring in the statistical analysis of probabilistic
#     functions of Markov chains. Annals of Mathematical Statistics 41(1),
#     164-171. The expectation-maximisation algorithm implemented here.
#   Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected
#     applications in speech recognition. Proceedings of the IEEE 77(2),
#     257-286. The forward-backward recursions, in the notation used here.
#   Hamilton, J. D. (1989). A new approach to the economic analysis of
#     nonstationary time series and the business cycle. Econometrica 57(2),
#     357-384. The regime-switching model in economics.
#   Ang, A. and Bekaert, G. (2002). International asset allocation with regime
#     shifts. Review of Financial Studies 15(4), 1137-1187. Why regimes matter
#     for allocation: correlations rise in the bad state.
#   Nystrup, P., Madsen, H. and Lindstrom, E. (2018). Dynamic portfolio
#     optimization across hidden market regimes. Quantitative Finance 18(1),
#     83-95.
# =============================================================================
module RegimeModels

using LinearAlgebra, Statistics, Random, LogExpFunctions

export HMMFit, fit_hmm, forward_backward, regime_conditional_moments, mixture_moments,
       worst_case_regime_weights, regime_persistence

"""
    HMMFit{T}

A fitted Gaussian hidden Markov model on a univariate driver.

# Fields

  - `pi0::Vector{T}`: Initial distribution, length `K`.
  - `P::Matrix{T}`: Transition matrix, `K x K`, rows summing to one.
  - `mu::Vector{T}`: State means, length `K`.
  - `sd::Vector{T}`: State standard deviations, length `K`.
  - `gam::Matrix{T}`: Smoothed state probabilities, `T x K`.
  - `loglik::Vector{T}`: Log likelihood at each iteration. **Must be
    non-decreasing**, which is the defining property of the algorithm and the
    driver's main assertion.
  - `converged::Bool`.

# Notes

  - States are returned **sorted by mean**, ascending, so state one is always
    the worst. Without that convention the labels permute between runs and no
    downstream code can rely on them. This is the label-switching problem, and
    sorting is the standard fix.
"""
struct HMMFit{T <: Real}
    pi0::Vector{T}
    P::Matrix{T}
    mu::Vector{T}
    sd::Vector{T}
    gam::Matrix{T}
    loglik::Vector{T}
    converged::Bool
end

"""
    _log_emission(y::AbstractVector, mu::AbstractVector, sd::AbstractVector) -> Matrix

Return the `T x K` matrix of log Gaussian densities `log p(y_t | s_t = k)`.
"""
function _log_emission(y::AbstractVector{T}, mu::AbstractVector{T},
                       sd::AbstractVector{T}) where {T <: Real}
    Tn = length(y)
    K = length(mu)
    B = Matrix{T}(undef, Tn, K)
    @inbounds for k in 1:K, t in 1:Tn
        z = (y[t] - mu[k]) / sd[k]
        B[t, k] = -log(sd[k]) - T(0.5) * log(2 * T(pi)) - z * z / 2
    end
    return B
end

"""
    forward_backward(logB::AbstractMatrix, logP::AbstractMatrix,
                     logpi0::AbstractVector) -> NamedTuple

Run the forward-backward recursions entirely in log space.

# Arguments

  - `logB`: Log emission densities, `T x K`.
  - `logP`: Log transition matrix, `K x K`.
  - `logpi0`: Log initial distribution, length `K`.

# Returns

A `NamedTuple` with `gam` (`T x K` smoothed probabilities), `xi_sum`
(`K x K` expected transition counts) and `loglik`.

# Mathematical definition

    alpha_t(k)  =  log p( y_1..t , s_t = k )
    beta_t(k)   =  log p( y_{t+1}..T | s_t = k )
    gam_t(k)    =  exp( alpha_t(k) + beta_t(k) - loglik )
    xi_t(i,j)   =  exp( alpha_t(i) + logP(i,j) + logB_{t+1}(j) + beta_{t+1}(j) - loglik )

# Notes

  - **Every step is in log space, using `logsumexp`.** The naive recursion
    underflows to zero after a few hundred observations, because it multiplies
    densities. Scaling factors are the classical alternative; log space is
    simpler and has no failure mode.
"""
function forward_backward(logB::AbstractMatrix{T}, logP::AbstractMatrix{T},
                          logpi0::AbstractVector{T}) where {T <: Real}
    Tn, K = size(logB)
    a = Matrix{T}(undef, Tn, K)
    b = zeros(T, Tn, K)
    @views a[1, :] .= logpi0 .+ logB[1, :]
    for t in 2:Tn
        for j in 1:K
            a[t, j] = logsumexp(view(a, t - 1, :) .+ view(logP, :, j)) + logB[t, j]
        end
    end
    for t in (Tn - 1):-1:1
        for i in 1:K
            b[t, i] = logsumexp(view(logP, i, :) .+ view(logB, t + 1, :) .+
                                view(b, t + 1, :))
        end
    end
    ll = logsumexp(view(a, Tn, :))
    gam = exp.(a .+ b .- ll)
    xi_sum = zeros(T, K, K)
    for t in 1:(Tn - 1), i in 1:K, j in 1:K
        xi_sum[i, j] += exp(a[t, i] + logP[i, j] + logB[t + 1, j] + b[t + 1, j] - ll)
    end
    return (; gam = gam, xi_sum = xi_sum, loglik = ll)
end

"""
    fit_hmm(y::AbstractVector, K::Integer; max_iter::Integer = 300,
            tol::Real = 1e-8, rng::Random.AbstractRNG = Random.default_rng())
        -> HMMFit

Fit a `K`-state Gaussian hidden Markov model by Baum-Welch.

# Arguments

  - `y`: The driver series, length `T`.
  - `K`: Number of states. Two or three in practice.
  - `max_iter`, `tol`: Convergence controls on the log likelihood.
  - `rng`: Used only for the initialisation.

# Returns

  - An [`HMMFit`](@ref), with states sorted by mean.

# The algorithm

Alternate until the log likelihood stops rising:

 1. **Expectation.** Run [`forward_backward`](@ref) to get `gam` and the
    expected transition counts.
 2. **Maximisation.** Set each state's mean and variance to the `gam`-weighted
    sample moments, and each transition row to the normalised expected counts.

# Notes

  - **The log likelihood is guaranteed non-decreasing.** That is what
    expectation-maximisation buys, and it is the only reliable check that the
    implementation is right. The driver asserts it at every iteration.
  - **It converges to a local optimum.** The initialisation here spreads the
    state means across the empirical quantiles of `y`, which is stable for
    financial data because the states really do differ in mean and variance.
    For a production version, restart from several initialisations and keep
    the best likelihood.
  - Two states on daily equity returns reliably recover a calm state with
    positive drift and a turbulent state with negative drift and two to three
    times the volatility. That is Ang and Bekaert's (2002) finding, and it is
    the reason the model earns its place.
"""
function fit_hmm(y::AbstractVector{<:Real}, K::Integer; max_iter::Integer = 300,
                 tol::Real = 1e-8, rng::Random.AbstractRNG = Random.default_rng())
    if K < 1
        throw(DomainError(K, "K must be >= 1"))
    end
    T = float(eltype(y))
    yv = collect(T, y)
    Tn = length(yv)
    if Tn < 2K
        throw(ArgumentError("need at least $(2K) observations for $(K) states, got $(Tn)"))
    end
    # Spread the initial means across the empirical quantiles.
    qs = [quantile(yv, (k - T(0.5)) / K) for k in 1:K]
    mu = collect(T, qs)
    sd = fill(std(yv), K)
    P = fill(T(0.1) / max(K - 1, 1), K, K)
    for k in 1:K
        P[k, k] = T(0.9)
    end
    if K == 1
        (P = ones(T, 1, 1))
    end
    pi0 = fill(one(T) / K, K)
    lls = T[]
    converged = false
    gam = zeros(T, Tn, K)
    for it in 1:max_iter
        logB = _log_emission(yv, mu, sd)
        fb = forward_backward(logB, log.(P), log.(pi0))
        gam = fb.gam
        push!(lls, fb.loglik)
        if it > 1 && abs(lls[end] - lls[end - 1]) <= tol * max(one(T), abs(lls[end]))
            converged = true
            break
        end
        # Maximisation.
        pi0 = vec(gam[1, :])
        pi0 ./= sum(pi0)
        for i in 1:K
            s = sum(view(fb.xi_sum, i, :))
            if s > 0
                @views P[i, :] .= fb.xi_sum[i, :] ./ s
            end
        end
        for k in 1:K
            wk = view(gam, :, k)
            sw = sum(wk)
            if sw <= 0
                continue
            end
            mu[k] = dot(wk, yv) / sw
            v = dot(wk, (yv .- mu[k]) .^ 2) / sw
            sd[k] = sqrt(max(v, eps(T)))
        end
    end
    ord = sortperm(mu)
    return HMMFit{T}(pi0[ord], P[ord, ord], mu[ord], sd[ord], gam[:, ord], lls, converged)
end

"""
    regime_conditional_moments(X::AbstractMatrix, gam::AbstractMatrix)
        -> NamedTuple

Estimate the mean and covariance of the whole panel within each regime.

# Arguments

  - `X`: Asset returns, `T x N`.
  - `gam`: Smoothed state probabilities, `T x K`, rows summing to one.

# Returns

A `NamedTuple` with `mu` (a `K`-vector of length-`N` means), `sigma` (a
`K`-vector of `N x N` covariances), and `weight` (the total probability mass
of each state).

# Mathematical definition

For state `k` with weights `g_t = gam[t, k]`,

    mu_k     =  sum_t g_t x_t / sum_t g_t
    sigma_k  =  sum_t g_t (x_t - mu_k)(x_t - mu_k)' / sum_t g_t

These are the **soft-assignment** moments: every observation contributes to
every state in proportion to its posterior probability. A hard assignment,
which allocates each observation to its most likely state, discards that
information and produces noisier estimates when the states overlap.

# Notes

  - **The effective sample size of state `k` is `sum_t g_t`, not `T`.** A state
    that occupies ten per cent of the sample has one tenth of the data, so its
    covariance is ten times noisier. Report the weight beside the moments, and
    shrink hard in the rare state.
  - This is the asymmetric design: `gam` comes from a one-dimensional model,
    and the `N`-dimensional moments are conditioned on it.
"""
function regime_conditional_moments(X::AbstractMatrix{<:Real}, gam::AbstractMatrix{<:Real})
    Tn, N = size(X)
    if size(gam, 1) != Tn
        throw(DimensionMismatch("X has $(Tn) rows, gam has $(size(gam, 1))"))
    end
    K = size(gam, 2)
    Tv = float(eltype(X))
    mus = Vector{Vector{Tv}}(undef, K)
    sigs = Vector{Matrix{Tv}}(undef, K)
    wts = Vector{Tv}(undef, K)
    for k in 1:K
        g = view(gam, :, k)
        sw = sum(g)
        wts[k] = sw
        if sw <= 0
            mus[k] = zeros(Tv, N)
            sigs[k] = Matrix{Tv}(I, N, N)
            continue
        end
        m = vec(transpose(X) * g) ./ sw
        Xc = X .- transpose(m)
        S = transpose(Xc) * (Xc .* g) ./ sw
        mus[k] = m
        sigs[k] = Matrix((S .+ transpose(S)) ./ 2)
    end
    return (; mu = mus, sigma = sigs, weight = wts)
end

"""
    mixture_moments(mus::AbstractVector, sigmas::AbstractVector,
                    p::AbstractVector) -> NamedTuple

Return the unconditional mean and covariance of a mixture over regimes.

# Arguments

  - `mus`: One mean vector per state.
  - `sigmas`: One covariance per state.
  - `p`: State probabilities, summing to one.

# Returns

A `NamedTuple` with `mu`, `sigma`, `within` and `between`.

# Mathematical definition

    mu     =  sum_k p_k mu_k

    sigma  =  sum_k p_k sigma_k                      (within)
            + sum_k p_k ( mu_k - mu )( mu_k - mu )'  (between)

**The second term is the one that gets forgotten.** It is the law of total
covariance: the unconditional variance is the average conditional variance
*plus* the variance of the conditional means. A caller who averages the regime
covariances and stops has understated the risk, and understated it most
exactly when the regimes differ most in mean, which is when it matters.

# Notes

  - The between term is a sum of outer products of the same vectors, so it is
    positive semi-definite. Dropping it therefore always understates, never
    overstates.
  - The driver checks the identity against a direct simulation from the
    mixture.
"""
function mixture_moments(mus::AbstractVector, sigmas::AbstractVector,
                         p::AbstractVector{<:Real})
    K = length(mus)
    if length(sigmas) != K || length(p) != K
        throw(DimensionMismatch("mus, sigmas and p must all have length $(K)"))
    end
    if !isapprox(sum(p), 1; atol = 1e-8)
        throw(DomainError(sum(p), "state probabilities must sum to one"))
    end
    N = length(first(mus))
    Tv = float(eltype(first(mus)))
    mu = zeros(Tv, N)
    for k in 1:K
        mu .+= p[k] .* mus[k]
    end
    within = zeros(Tv, N, N)
    between = zeros(Tv, N, N)
    for k in 1:K
        within .+= p[k] .* sigmas[k]
        d = mus[k] .- mu
        between .+= p[k] .* (d * transpose(d))
    end
    return (; mu = mu, sigma = within .+ between, within = within, between = between)
end

"""
    worst_case_regime_weights(mus::AbstractVector, sigmas::AbstractVector;
                              gamma::Real = 1.0, budget::Real = 1.0,
                              iters::Integer = 400, step::Real = 0.5)
        -> NamedTuple

Return the portfolio that maximises the **worst** regime's utility.

# Arguments

  - `mus`, `sigmas`: The regime-conditional moments.
  - `gamma`: Risk aversion.
  - `budget`: Total weight.
  - `iters`, `step`: Controls for the subgradient ascent.

# Returns

A `NamedTuple` with `w`, `utilities` (one per regime), `worst_regime` and
`worst_utility`.

# Mathematical definition

    maximise over w    min over k   [ mu_k' w  -  (gamma / 2) w' sigma_k w ]
    subject to         1' w = budget

The inner minimum of concave functions is concave, so the problem is a concave
maximisation and a projected subgradient method converges. At each step the
subgradient is that of the currently worst regime.

# Notes

  - **This is not the same as optimising the mixture.** Optimising the mixture
    accepts a bad outcome in the rare state if the common state is good enough.
    This refuses to. Which is right depends on whether the caller can survive
    the bad state, which is a question about them and not about the data.
  - The answer is usually much closer to the bad regime's portfolio than the
    state probabilities suggest, because the bad regime has both the lower mean
    and the higher variance and therefore dominates the minimum almost always.
"""
function worst_case_regime_weights(mus::AbstractVector, sigmas::AbstractVector;
                                   gamma::Real = 1.0, budget::Real = 1.0,
                                   iters::Integer = 400, step::Real = 0.5)
    K = length(mus)
    N = length(first(mus))
    Tv = float(eltype(first(mus)))
    w = fill(Tv(budget) / N, N)
    util(k, w) = dot(mus[k], w) - (gamma / 2) * dot(w, sigmas[k], w)
    for it in 1:iters
        us = [util(k, w) for k in 1:K]
        k = argmin(us)
        g = mus[k] .- gamma .* (sigmas[k] * w)
        w .+= (Tv(step) / sqrt(it)) .* g
        # Project onto the budget hyperplane.
        w .+= (Tv(budget) - sum(w)) / N
    end
    us = [util(k, w) for k in 1:K]
    return (; w = w, utilities = us, worst_regime = argmin(us), worst_utility = minimum(us))
end

"""
    regime_persistence(P::AbstractMatrix) -> NamedTuple

Return the expected duration of each state and the stationary distribution.

# Arguments

  - `P`: Transition matrix, `K x K`.

# Returns

A `NamedTuple` with `expected_duration` and `stationary`.

# Mathematical definition

The time spent in state `k` before leaving is geometric with success
probability `1 - P[k, k]`, so

    expected duration  =  1 / ( 1 - P[k, k] )

The stationary distribution is the left eigenvector of `P` for eigenvalue one,
normalised to sum to one.

# Notes

  - **Report the duration, never the transition probability.** `P[1, 1] = 0.98`
    means nothing to a reader. "The bad state lasts fifty days on average"
    means something, and it is the number that decides whether a regime model
    is usable for allocation at all: a state that lasts two days cannot be
    traded.
"""
function regime_persistence(P::AbstractMatrix{<:Real})
    K = size(P, 1)
    dur = [P[k, k] < 1 ? 1 / (1 - P[k, k]) : Inf for k in 1:K]
    vals, vecs = eigen(transpose(Matrix(float.(P))))
    i = argmin(abs.(vals .- 1))
    v = real.(view(vecs, :, i))
    v = abs.(v)
    return (; expected_duration = dur, stationary = v ./ sum(v))
end

end # module RegimeModels
