# =============================================================================
# Prototype 19 — Online moments: `partial_fit` for an estimator.
#
# Purpose
#   Reports 3 and 4 both ask for it, and report 4 poses the question exactly:
#
#       Can a fitted estimator carry state forward through time and be
#       evaluated without refitting from scratch?
#
#   Today every walk-forward step recomputes the covariance over the whole
#   window. With `T` observations and `S` steps that is `O(S T N^2)` work, of
#   which almost all is repeated. An incremental update is `O(N^2)` per new
#   observation, so a walk-forward becomes `O(S N^2)`.
#
#   The interesting part is not the speed. It is that **an exact incremental
#   estimator and a batch estimator must agree to machine precision**, and the
#   naive incremental formula does not. Welford's algorithm does, and the
#   difference is the whole content of this file.
#
# Status
#   Standalone. Depends on `LinearAlgebra` and `Statistics`.
#
# Notation used throughout this file
#   N       Number of assets.
#   n       Number of observations absorbed so far.
#   mu      Running mean, length `N`.
#   M       Running matrix of co-moments, `N x N`. The covariance is
#           `M / (n - 1)`.
#   lam     Exponential decay weight in `(0, 1]`. Larger forgets faster.
#   x       One new observation, length `N`.
#
# Sources
#   Welford, B. P. (1962). Note on a method for calculating corrected sums of
#     squares and products. Technometrics 4(3), 419-420. The exact update.
#   Chan, T. F., Golub, G. H. and LeVeque, R. J. (1983). Algorithms for
#     computing the sample variance: analysis and recommendations. The American
#     Statistician 37(3), 242-247. The merge formula, and the numerical
#     analysis of why the textbook formula fails.
#   Riskmetrics Group (1996). RiskMetrics Technical Document, 4th edition.
#     J. P. Morgan. The exponentially weighted covariance and the origin of the
#     `0.94` daily decay.
#   Fleming, J., Kirby, C. and Ostdiek, B. (2001). The economic value of
#     volatility timing. Journal of Finance 56(1), 329-352. Why a decaying
#     covariance is worth the trouble.
# =============================================================================
module OnlineMoments

using LinearAlgebra, Statistics

export WelfordMoments, EWMAMoments, partial_fit!, current_mean, current_cov, merge_moments,
       half_life, decay_from_half_life

"""
    WelfordMoments{T}

An exactly incremental mean and covariance.

# Fields

  - `n::Int`: Observations absorbed.
  - `mu::Vector{T}`: Running mean, length `N`.
  - `M::Matrix{T}`: Running co-moment matrix, `N x N`.

# Notes

  - **This is a mutable estimator, which the library's design rules forbid in
    `src/`.** An adapted version must return a new Result from
    `partial_fit`, not mutate one, so that an Estimator stays configuration and
    a Result stays data. The mutable form is used here because it makes the
    algorithm legible.
"""
mutable struct WelfordMoments{T <: Real}
    n::Int
    mu::Vector{T}
    M::Matrix{T}
end
function WelfordMoments(N::Integer; T::Type{<:Real} = Float64)
    return WelfordMoments{T}(0, zeros(T, N), zeros(T, N, N))
end

"""
    partial_fit!(w::WelfordMoments, x::AbstractVector) -> WelfordMoments

Absorb one observation.

# Arguments

  - `w`: The running state.
  - `x`: A new observation, length `N`.

# Mathematical definition

    n     <- n + 1
    d     <- x - mu_old
    mu    <- mu_old + d / n
    M     <- M + d * (x - mu_new)'

The covariance is then `M / (n - 1)`.

# Notes

  - **The asymmetry in the last line is the point.** The first factor uses the
    *old* mean and the second uses the *new* one. Using the same mean in both
    gives the textbook formula, which loses catastrophic precision when the
    mean is large relative to the spread. Chan and co-authors (1983) show the
    textbook form can return a **negative** variance on real data. Welford's
    cannot.
  - Cost is `O(N^2)` per observation, with no allocation beyond the difference
    vector.
"""
function partial_fit!(w::WelfordMoments{T}, x::AbstractVector{<:Real}) where {T}
    N = length(w.mu)
    if length(x) != N
        throw(DimensionMismatch("x has length $(length(x)), state has $(N) assets"))
    end
    w.n += 1
    d = collect(T, x) .- w.mu
    w.mu .+= d ./ w.n
    d2 = collect(T, x) .- w.mu
    # M += d * d2', the outer product of the pre- and post-update deviations.
    @inbounds for j in 1:N, i in 1:N
        w.M[i, j] += d[i] * d2[j]
    end
    return w
end

"""
    partial_fit!(w::WelfordMoments, X::AbstractMatrix) -> WelfordMoments

Absorb every row of `X`, in order.
"""
function partial_fit!(w::WelfordMoments, X::AbstractMatrix{<:Real})
    for t in axes(X, 1)
        partial_fit!(w, view(X, t, :))
    end
    return w
end

"""
    current_mean(w::WelfordMoments) -> Vector

Return the running mean. Equal to `vec(mean(X; dims = 1))` to machine
precision.
"""
current_mean(w::WelfordMoments) = copy(w.mu)

"""
    current_cov(w::WelfordMoments; corrected::Bool = true) -> Matrix

Return the running covariance.

# Arguments

  - `w`: The running state.
  - `corrected`: Divide by `n - 1` (the unbiased estimator) or by `n`.

# Returns

  - An `N x N` symmetric matrix. Equal to `cov(X)` to machine precision.

# Notes

  - The result is symmetrised on the way out, because the accumulated `M` can
    drift from symmetry by a rounding error over millions of updates.
"""
function current_cov(w::WelfordMoments; corrected::Bool = true)
    d = corrected ? max(w.n - 1, 1) : max(w.n, 1)
    C = w.M ./ d
    return (C .+ transpose(C)) ./ 2
end

"""
    merge_moments(a::WelfordMoments, b::WelfordMoments) -> WelfordMoments

Combine two independently accumulated states into one.

# Arguments

  - `a`, `b`: States accumulated over **disjoint** observation sets.

# Returns

  - A new state identical to one built by absorbing both sets in sequence.

# Mathematical definition

Chan, Golub and LeVeque (1983):

    n   =  n_a + n_b
    d   =  mu_b - mu_a
    mu  =  mu_a  +  d * n_b / n
    M   =  M_a + M_b  +  d d' * ( n_a n_b / n )

The last term is the co-moment the two halves cannot see, because each was
centred on its own mean.

# Notes

  - **This is what makes the estimator parallel.** Split the history across
    threads, accumulate independently, and merge. The library's `FLoops`
    dependency already provides the parallel loop; what is absent is an
    estimator with an associative merge.
  - The operation is associative and commutative, so the merge order does not
    change the answer. The driver asserts that.
"""
function merge_moments(a::WelfordMoments{T}, b::WelfordMoments{T}) where {T}
    if length(a.mu) != length(b.mu)
        throw(DimensionMismatch("states have $(length(a.mu)) and $(length(b.mu)) assets"))
    end
    if a.n == 0
        return WelfordMoments{T}(b.n, copy(b.mu), copy(b.M))
    end
    if b.n == 0
        return WelfordMoments{T}(a.n, copy(a.mu), copy(a.M))
    end
    n = a.n + b.n
    d = b.mu .- a.mu
    mu = a.mu .+ d .* (b.n / n)
    M = a.M .+ b.M .+ (d * transpose(d)) .* (a.n * b.n / n)
    return WelfordMoments{T}(n, mu, M)
end

"""
    EWMAMoments{T}

An exponentially weighted mean and covariance.

# Fields

  - `lam::T`: Decay weight in `(0, 1]`. The weight on the newest observation.
  - `mu::Vector{T}`: Running mean.
  - `S::Matrix{T}`: Running covariance.
  - `n::Int`: Observations absorbed, used for the initialisation bias
    correction.
  - `initialised::Bool`: Whether the first observation has been seen.

# Mathematical definition

    mu_t  =  (1 - lam) mu_{t-1}  +  lam x_t
    S_t   =  (1 - lam) S_{t-1}   +  lam ( x_t - mu_{t-1} ) ( x_t - mu_{t-1} )'

# Notes

  - **The deviation uses the *previous* mean, not the updated one.** Using the
    updated mean shrinks the estimated covariance towards zero, because the
    mean has already moved towards the observation. RiskMetrics uses the
    previous mean, and so does this.
  - RiskMetrics' daily decay is `lam = 0.06`, a half life of about 11 days.
    Their monthly figure is `lam = 0.03`.
"""
mutable struct EWMAMoments{T <: Real}
    lam::T
    mu::Vector{T}
    S::Matrix{T}
    n::Int
    initialised::Bool
end
function EWMAMoments(N::Integer; lam::Real = 0.06, T::Type{<:Real} = Float64)
    if !(0 < lam <= 1)
        throw(DomainError(lam, "lam must satisfy 0 < lam <= 1"))
    end
    return EWMAMoments{T}(T(lam), zeros(T, N), zeros(T, N, N), 0, false)
end

"""
    partial_fit!(e::EWMAMoments, x::AbstractVector) -> EWMAMoments

Absorb one observation into an exponentially weighted state.
"""
function partial_fit!(e::EWMAMoments{T}, x::AbstractVector{<:Real}) where {T}
    N = length(e.mu)
    if length(x) != N
        throw(DimensionMismatch("x has length $(length(x)), state has $(N) assets"))
    end
    xv = collect(T, x)
    if !e.initialised
        e.mu .= xv
        e.initialised = true
        e.n = 1
        return e
    end
    d = xv .- e.mu
    e.S .= (1 - e.lam) .* e.S .+ e.lam .* (d * transpose(d))
    e.mu .= (1 - e.lam) .* e.mu .+ e.lam .* xv
    e.n += 1
    return e
end

function partial_fit!(e::EWMAMoments, X::AbstractMatrix{<:Real})
    for t in axes(X, 1)
        partial_fit!(e, view(X, t, :))
    end
    return e
end

current_mean(e::EWMAMoments) = copy(e.mu)

"""
    current_cov(e::EWMAMoments) -> Matrix

Return the exponentially weighted covariance, symmetrised.
"""
function current_cov(e::EWMAMoments)
    return (e.S .+ transpose(e.S)) ./ 2
end

"""
    half_life(lam::Real) -> Real

Return the half life in periods of an exponential weight `lam`.

# Mathematical definition

    half_life  =  log(2) / ( -log(1 - lam) )

The weight on an observation `k` periods old is `lam (1 - lam)^k`, so the
weight halves every `half_life` periods.

# Notes

  - **Report the half life, never the decay.** `lam = 0.06` means nothing to a
    reader. "Eleven days" means something.
"""
function half_life(lam::Real)
    if !(0 < lam < 1)
        throw(DomainError(lam, "lam must satisfy 0 < lam < 1"))
    end
    return log(2) / (-log(1 - lam))
end

"""
    decay_from_half_life(h::Real) -> Real

Return the decay weight whose half life is `h` periods. The inverse of
[`half_life`](@ref).

# Mathematical definition

    lam  =  1 - 2^(-1 / h)
"""
function decay_from_half_life(h::Real)
    if !(h > 0)
        throw(DomainError(h, "half life must be > 0"))
    end
    return 1 - 2^(-1 / h)
end

end # module OnlineMoments
