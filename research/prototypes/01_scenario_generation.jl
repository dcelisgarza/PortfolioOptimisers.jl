# =============================================================================
# Prototype 1 — Scenario generation from a fitted prior.
#
# Purpose
#   A Prior asserts a distribution over asset returns. Nothing in `src/` can
#   draw from it. This file gives the smallest seam that closes the gap:
#   one verb, `simulate_returns`, with one method per generative model.
#
# Status
#   Standalone. It depends only on `LinearAlgebra`, `Random`, `Statistics`,
#   `StatsBase` and `Distributions`, all of which are already direct
#   dependencies of PortfolioOptimisers.jl. It does not load the library.
#
# Notation used throughout this file
#   T      Number of observations (rows) in the historical returns matrix.
#   N      Number of assets (columns).
#   S      Number of simulated observations to produce.
#   X      Historical returns matrix, `T x N`, observations-major.
#   mu     Vector of expected returns, length `N`.
#   sigma  Covariance matrix of returns, `N x N`, symmetric positive definite.
#   R      A simulated returns matrix, `S x N`, with the same column order
#          as `X`.
#
# Sources
#   Sklar, A. (1959). Fonctions de repartition a n dimensions et leurs marges.
#     Publications de l'Institut de Statistique de l'Universite de Paris 8,
#     229-231. The theorem that separates marginals from dependence.
#   Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap.
#     Journal of the American Statistical Association 89(428), 1303-1313.
#   Kruskal, W. H. (1958). Ordinal measures of association. Journal of the
#     American Statistical Association 53(284), 814-861. The
#     `rho = 2 sin(pi * rho_s / 6)` identity for the Gaussian copula.
#   Cont, R. (2001). Empirical properties of asset returns: stylized facts and
#     statistical issues. Quantitative Finance 1(2), 223-236. The reason a
#     Gaussian generator is not enough.
# =============================================================================
module ScenarioGeneration

using LinearAlgebra, Random, Statistics, StatsBase, Distributions

export AbstractScenarioAlgorithm, GaussianScenarios, StudentTScenarios,
       GaussianCopulaScenarios, StationaryBootstrapScenarios, simulate_returns

"""
    AbstractScenarioAlgorithm

Supertype of the generative models that produce a simulated returns matrix.

A member of this family is an **Algorithm** in the library's vocabulary. It
selects the generative behaviour and it carries no data. The data arrives as
an argument to [`simulate_returns`](@ref).
"""
abstract type AbstractScenarioAlgorithm end

"""
    GaussianScenarios <: AbstractScenarioAlgorithm

Draw from the multivariate normal distribution `N(mu, sigma)`.

# Mathematical definition

Let `sigma = L * L'` be the Cholesky factor of the covariance matrix. Draw a
matrix `Z` of `S x N` independent standard normal variates. Then

    R = 1 * mu' + Z * L'

has mean `mu` and covariance `sigma` by construction.

# Notes

  - This generator reproduces the first two moments exactly in expectation and
    nothing else. It has no excess kurtosis and no tail dependence, so it
    understates the loss of a diversified portfolio in a crash. Use it as the
    baseline against which the other generators are read.
"""
struct GaussianScenarios <: AbstractScenarioAlgorithm end

"""
    StudentTScenarios{T} <: AbstractScenarioAlgorithm

Draw from the multivariate Student-t distribution with `nu` degrees of freedom,
scaled so that the covariance of the draw equals `sigma`.

# Fields

  - `nu::T`: Degrees of freedom. Must satisfy `nu > 2`, because the covariance
    of a multivariate t is undefined at or below two degrees of freedom.

# Mathematical definition

A multivariate t variate with scale matrix `S_c` and `nu` degrees of freedom is

    Y = mu + Z * L' / sqrt(W / nu),    W ~ ChiSquared(nu),   S_c = L * L'

and it has covariance `nu / (nu - 2) * S_c`. To make the draw match a target
covariance `sigma`, set the scale matrix to

    S_c = (nu - 2) / nu * sigma

which is what this generator does. Every asset then carries the same tail
index `nu`, and the assets share one radial shock `W`, so the draw has tail
dependence that [`GaussianScenarios`](@ref) cannot produce.
"""
struct StudentTScenarios{T <: Real} <: AbstractScenarioAlgorithm
    nu::T
    function StudentTScenarios(nu::T) where {T <: Real}
        if !(nu > 2)
            throw(DomainError(nu,
                              "nu must be > 2, otherwise the covariance of the multivariate t is undefined"))
        end
        return new{T}(nu)
    end
end
StudentTScenarios(; nu::Real = 5.0) = StudentTScenarios(nu)

"""
    GaussianCopulaScenarios <: AbstractScenarioAlgorithm

Draw with a Gaussian dependence structure and **empirical marginals**.

# Fields

  - `use_spearman::Bool`: If `true`, estimate the copula correlation from
    Spearman rank correlation through the identity below. If `false`, use the
    Pearson correlation of the data directly. The rank route is robust to the
    marginal shape, which is the whole point of a copula, so it is the default.

# Mathematical definition

By Sklar's theorem any joint distribution `F` factorises into its marginals
`F_1, ..., F_N` and a copula `C`:

    F(x_1, ..., x_N) = C(F_1(x_1), ..., F_N(x_N))

The Gaussian copula takes `C` to be the copula of a normal with correlation
matrix `P`. The sampler is:

 1. Draw `Z ~ N(0, P)`, of size `S x N`.
 2. Map to uniforms with the normal CDF: `U[:, j] = Phi(Z[:, j])`.
 3. Map to the data scale with the empirical quantile of each column:
    `R[:, j] = quantile(X[:, j], U[:, j])`.

Step 3 keeps every marginal exactly as observed, skew and fat tails included.
Step 1 supplies the dependence. The rank transform in step 2 distorts a
Pearson correlation, so `P` is recovered from the Spearman rank correlation
`rho_s` through the Gaussian-copula identity

    P_ij = 2 * sin(pi * rho_s_ij / 6)

# Notes

  - The Gaussian copula has **zero tail dependence**. Two assets simulated this
    way become independent in the extreme, however high their correlation. If
    the joint tail is the object of study, this generator is the wrong one, and
    a t-copula or a vine copula is the correct next step. See Aas, K., Czado,
    C., Frigessi, A. and Bakken, H. (2009), Pair-copula constructions of
    multiple dependence, Insurance: Mathematics and Economics 44(2), 182-198.
"""
struct GaussianCopulaScenarios <: AbstractScenarioAlgorithm
    use_spearman::Bool
end
GaussianCopulaScenarios(; use_spearman::Bool = true) = GaussianCopulaScenarios(use_spearman)

"""
    StationaryBootstrapScenarios{T} <: AbstractScenarioAlgorithm

Resample blocks of consecutive historical observations, with geometrically
distributed block lengths that wrap around the end of the sample.

# Fields

  - `block_size::T`: Mean block length. Must be positive. The probability that
    a block ends at any step is `p = 1 / block_size`.

# Mathematical definition

Pick a start index uniformly on `1:T`. At each step, with probability
`1 - p` take the next observation, and with probability `p` jump to a fresh
uniform start index. Indices wrap with modular arithmetic. The result is
stationary, which is the property the fixed-block bootstrap lacks.

# Notes

  - This generator makes no distributional assumption at all. It preserves
    every stylised fact that survives inside a block, volatility clusters
    included, and it can never produce a return the market has not already
    produced. That last property is a limit, not a feature: a stress test
    built on it cannot go beyond the worst day in the sample.
  - The library already uses this algorithm for uncertainty sets. See
    `src/14_UncertaintySets/04_BootstrapUncertaintySets.jl`. What is absent is
    the use of it as a returns generator.
"""
struct StationaryBootstrapScenarios{T <: Real} <: AbstractScenarioAlgorithm
    block_size::T
    function StationaryBootstrapScenarios(block_size::T) where {T <: Real}
        if !(block_size > 0)
            throw(DomainError(block_size, "block_size must be > 0"))
        end
        return new{T}(block_size)
    end
end
function StationaryBootstrapScenarios(; block_size::Real = 10.0)
    return StationaryBootstrapScenarios(block_size)
end

# -----------------------------------------------------------------------------
# Helper: a Cholesky factor that survives a matrix that is only almost positive
# definite. The library has `posdef` machinery for this; a prototype must not
# depend on it, so the fallback is an eigenvalue clip.
# -----------------------------------------------------------------------------
"""
    _safe_cholesky_factor(sigma::AbstractMatrix{<:Real}) -> Matrix

Return a lower-triangular `L` with `L * L'` equal to `sigma`, or to the nearest
positive definite matrix to `sigma` when the Cholesky factorisation fails.

# Arguments

  - `sigma`: Symmetric matrix, `N x N`.

# Returns

  - `L::Matrix`: Lower-triangular factor, `N x N`.

# Details

The repair clips every eigenvalue below `eps` up to `eps` and rebuilds the
matrix. In the library this job belongs to `PosdefEstimator`, and an adapted
version must call that instead of this function.
"""
function _safe_cholesky_factor(sigma::AbstractMatrix{<:Real})
    sym = Symmetric((sigma + sigma') / 2)
    fact = cholesky(sym; check = false)
    if issuccess(fact)
        return Matrix(fact.L)
    end
    vals, vecs = eigen(sym)
    floor_val = eps(eltype(vals)) * maximum(abs, vals)
    clipped = vecs * Diagonal(max.(vals, floor_val)) * vecs'
    return Matrix(cholesky(Symmetric((clipped + clipped') / 2)).L)
end

"""
    simulate_returns(alg::GaussianScenarios, mu::AbstractVector,
                     sigma::AbstractMatrix; n_obs::Integer,
                     rng::Random.AbstractRNG = Random.default_rng()) -> Matrix

Draw `n_obs` observations from `N(mu, sigma)`.

# Arguments

  - `alg`: The generator. Carries no parameters.
  - `mu`: Expected returns, length `N`.
  - `sigma`: Covariance matrix, `N x N`.
  - `n_obs`: Number of simulated observations, `S`.
  - `rng`: Random number generator. Pass a seeded one for a reproducible draw.

# Returns

  - `R::Matrix`: Simulated returns, `S x N`.

# Validation

  - `length(mu) == size(sigma, 1) == size(sigma, 2)`.
  - `n_obs > 0`.
"""
function simulate_returns(alg::GaussianScenarios, mu::AbstractVector{<:Real},
                          sigma::AbstractMatrix{<:Real}; n_obs::Integer,
                          rng::Random.AbstractRNG = Random.default_rng())
    _check_moments(mu, sigma, n_obs)
    L = _safe_cholesky_factor(sigma)
    Z = randn(rng, n_obs, length(mu))
    return Z * transpose(L) .+ transpose(mu)
end

"""
    simulate_returns(alg::StudentTScenarios, mu::AbstractVector,
                     sigma::AbstractMatrix; n_obs::Integer,
                     rng::Random.AbstractRNG = Random.default_rng()) -> Matrix

Draw `n_obs` observations from a multivariate Student-t whose covariance is
`sigma` and whose mean is `mu`.

# Arguments

  - `alg`: The generator. `alg.nu` is the degrees of freedom.
  - `mu`: Expected returns, length `N`.
  - `sigma`: Target covariance matrix, `N x N`. Note that this is the
    covariance of the draw, not the scale matrix of the t.
  - `n_obs`: Number of simulated observations, `S`.
  - `rng`: Random number generator.

# Returns

  - `R::Matrix`: Simulated returns, `S x N`.
"""
function simulate_returns(alg::StudentTScenarios, mu::AbstractVector{<:Real},
                          sigma::AbstractMatrix{<:Real}; n_obs::Integer,
                          rng::Random.AbstractRNG = Random.default_rng())
    _check_moments(mu, sigma, n_obs)
    nu = alg.nu
    # Scale down so that (nu / (nu - 2)) * scale == sigma.
    scale = ((nu - 2) / nu) .* sigma
    L = _safe_cholesky_factor(scale)
    N = length(mu)
    Z = randn(rng, n_obs, N)
    # One radial shock per observation, shared by every asset. This is the
    # source of the tail dependence.
    W = rand(rng, Distributions.Chisq(nu), n_obs)
    radial = sqrt.(nu ./ W)
    return (Z .* radial) * transpose(L) .+ transpose(mu)
end

"""
    simulate_returns(alg::GaussianCopulaScenarios, X::AbstractMatrix;
                     n_obs::Integer,
                     rng::Random.AbstractRNG = Random.default_rng()) -> Matrix

Draw `n_obs` observations with a Gaussian copula and the empirical marginals
of `X`.

# Arguments

  - `alg`: The generator. `alg.use_spearman` selects the correlation route.
  - `X`: Historical returns, `T x N`.
  - `n_obs`: Number of simulated observations, `S`.
  - `rng`: Random number generator.

# Returns

  - `R::Matrix`: Simulated returns, `S x N`. Every column takes values in the
    convex hull of the corresponding column of `X`, because the empirical
    quantile interpolates between order statistics and never extrapolates.
"""
function simulate_returns(alg::GaussianCopulaScenarios, X::AbstractMatrix{<:Real};
                          n_obs::Integer, rng::Random.AbstractRNG = Random.default_rng())
    if n_obs <= 0
        throw(DomainError(n_obs, "n_obs must be > 0"))
    end
    N = size(X, 2)
    corr_mat = if alg.use_spearman
        rho_s = StatsBase.corspearman(X)
        # Gaussian-copula identity. `clamp` guards the numerical drift that
        # puts a rank correlation a hair outside [-1, 1].
        2 .* sin.((pi / 6) .* clamp.(rho_s, -1, 1))
    else
        Statistics.cor(X)
    end
    # A correlation matrix must have a unit diagonal after the transform.
    corr_mat = (corr_mat + transpose(corr_mat)) / 2
    for i in 1:N
        corr_mat[i, i] = one(eltype(corr_mat))
    end
    L = _safe_cholesky_factor(corr_mat)
    Z = randn(rng, n_obs, N) * transpose(L)
    normal = Distributions.Normal()
    R = Matrix{float(eltype(X))}(undef, n_obs, N)
    for j in 1:N
        col = view(X, :, j)
        u = Distributions.cdf.(normal, view(Z, :, j))
        # `clamp` keeps the probability strictly inside (0, 1) so that the
        # empirical quantile stays defined at the boundary.
        u = clamp.(u, eps(eltype(u)), one(eltype(u)) - eps(eltype(u)))
        R[:, j] = Statistics.quantile(col, u)
    end
    return R
end

"""
    simulate_returns(alg::StationaryBootstrapScenarios, X::AbstractMatrix;
                     n_obs::Integer,
                     rng::Random.AbstractRNG = Random.default_rng()) -> Matrix

Resample `n_obs` observations from `X` with the stationary bootstrap.

# Arguments

  - `alg`: The generator. `alg.block_size` is the mean block length.
  - `X`: Historical returns, `T x N`.
  - `n_obs`: Number of simulated observations, `S`.
  - `rng`: Random number generator.

# Returns

  - `R::Matrix`: Resampled returns, `S x N`. Every row is a row of `X`.
"""
function simulate_returns(alg::StationaryBootstrapScenarios, X::AbstractMatrix{<:Real};
                          n_obs::Integer, rng::Random.AbstractRNG = Random.default_rng())
    if n_obs <= 0
        throw(DomainError(n_obs, "n_obs must be > 0"))
    end
    T = size(X, 1)
    p = min(one(float(alg.block_size)), inv(float(alg.block_size)))
    idx = Vector{Int}(undef, n_obs)
    current = rand(rng, 1:T)
    for t in 1:n_obs
        idx[t] = current
        if rand(rng) < p
            current = rand(rng, 1:T)
        else
            # Wrap with modular arithmetic so the sample is stationary.
            current = current == T ? 1 : current + 1
        end
    end
    return X[idx, :]
end

"""
    _check_moments(mu, sigma, n_obs)

Refuse a moment pair whose shapes disagree, or a non-positive draw count.
"""
function _check_moments(mu::AbstractVector, sigma::AbstractMatrix, n_obs::Integer)
    N = length(mu)
    if size(sigma, 1) != N || size(sigma, 2) != N
        throw(DimensionMismatch("sigma must be $(N) x $(N) to match mu, got $(size(sigma))"))
    end
    if n_obs <= 0
        throw(DomainError(n_obs, "n_obs must be > 0"))
    end
    return nothing
end

end # module ScenarioGeneration
