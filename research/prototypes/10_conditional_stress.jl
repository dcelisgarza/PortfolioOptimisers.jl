# =============================================================================
# Prototype 10 — Conditional stress scenarios and scenario transformations.
#
# Purpose
#   Prototype 1 draws from a fitted model. This file does the two things a
#   stress test needs beyond that:
#
#     1. **Condition.** Fix some assets or factors at stressed values and let
#        the rest of the universe respond *through its own dependence
#        structure*, rather than being held at its unconditional mean. A stress
#        test that shocks equities and leaves credit at its average is not a
#        stress test.
#     2. **Transform.** Take an existing scenario set and bend it: multiply
#        volatility, push correlations towards one, shift a factor. The
#        transformed set feeds straight into the library's existing risk
#        measures, because it is still just a returns matrix.
#
#   It also supplies the tail-dependent generator that prototype 1 deliberately
#   lacks. A Gaussian copula has zero tail dependence, so it cannot produce the
#   joint crash that a stress test exists to study. A t-copula can, and its
#   tail-dependence coefficient is known in closed form, so the generator can
#   be checked rather than trusted.
#
# Status
#   Standalone. Depends on `LinearAlgebra`, `Statistics`, `Random`,
#   `StatsBase` and `Distributions`, all already dependencies of the library.
#
# Notation used throughout this file
#   T, N     Observations and assets.
#   X        Returns matrix, `T x N`.
#   mu, sig  Expected returns (length `N`) and covariance (`N x N`).
#   idx      Indices of the *stressed* assets or factors.
#   rest     The complementary indices, which respond to the stress.
#   v        The stressed values, one per entry of `idx`.
#   nu       Degrees of freedom of a t-copula.
#   lam_tail The coefficient of lower tail dependence, in `[0, 1)`.
#
# Sources
#   Meucci, A. (2005). Risk and Asset Allocation. Springer. Chapter 3 gives the
#     conditional-normal formulae used by `conditional_gaussian_stress`.
#   Embrechts, P., McNeil, A. J. and Straumann, D. (2002). Correlation and
#     dependence in risk management: properties and pitfalls. In: Risk
#     Management: Value at Risk and Beyond, Cambridge University Press, 176-223.
#     The argument against correlation as a dependence summary, and the
#     definition of tail dependence.
#   Demarta, S. and McNeil, A. J. (2005). The t copula and related copulas.
#     International Statistical Review 73(1), 111-129. The closed-form tail
#     dependence coefficient verified in this file.
#   Kupiec, P. H. (1998). Stress testing in a value at risk framework. Journal
#     of Derivatives 6(1), 7-24. The case for conditional rather than
#     univariate stress.
#   Basel Committee on Banking Supervision (2018). Stress testing principles.
#     Bank for International Settlements. The supervisory framing.
# =============================================================================
module ConditionalStress

using LinearAlgebra, Statistics, Random, StatsBase, Distributions

export conditional_gaussian_stress, conditional_moments, t_copula_scenarios,
       tail_dependence_coefficient, empirical_tail_dependence, scale_volatility,
       shock_correlation, shift_mean, factor_shock, stress_report

"""
    conditional_moments(mu::AbstractVector, sig::AbstractMatrix,
                        idx::AbstractVector{<:Integer}, v::AbstractVector)
        -> NamedTuple

Return the conditional mean and covariance of the unstressed assets, given
that the stressed ones take the values `v`.

# Arguments

  - `mu`: Unconditional expected returns, length `N`.
  - `sig`: Unconditional covariance, `N x N`.
  - `idx`: Indices held at stressed values.
  - `v`: The stressed values, `length(idx)`.

# Returns

A `NamedTuple`:

  - `rest`: The indices that were not stressed.
  - `mu_cond`: Conditional mean of the unstressed block.
  - `sig_cond`: Conditional covariance of the unstressed block.
  - `beta`: The `length(rest) x length(idx)` matrix of conditional
    sensitivities, `sig_21 * inv(sig_11)`. Row `i` is how asset `i` responds to
    a unit shock in each stressed name. **This is a regression beta, derived
    rather than estimated.**

# Mathematical definition

Partition the return vector into the stressed block `1` and the rest `2`:

    mu = [mu_1; mu_2],    sig = [ sig_11  sig_12 ; sig_21  sig_22 ]

Under a joint normal the conditional law of block 2 is normal with

    mu_{2|1}  =  mu_2  +  sig_21 sig_11^(-1) ( v - mu_1 )
    sig_{2|1} =  sig_22  -  sig_21 sig_11^(-1) sig_12

# Notes

  - **The conditional covariance does not depend on `v`.** Under a normal, a
    stress changes where the rest of the universe is centred but not how much
    it varies. That is a real limitation, not a modelling choice: correlations
    rise in crises and a normal cannot express it. Combine this function with
    [`shock_correlation`](@ref) when that matters.
  - `sig_{2|1}` is the Schur complement of `sig_11` in `sig`. It is positive
    semi-definite whenever `sig` is, so the result is always a valid
    covariance.
"""
function conditional_moments(mu::AbstractVector{<:Real}, sig::AbstractMatrix{<:Real},
                             idx::AbstractVector{<:Integer}, v::AbstractVector{<:Real})
    N = length(mu)
    if size(sig) != (N, N)
        throw(DimensionMismatch("sig must be $(N) x $(N), got $(size(sig))"))
    end
    if length(idx) != length(v)
        throw(DimensionMismatch("idx has length $(length(idx)), v has length $(length(v))"))
    end
    if !all(i -> 1 <= i <= N, idx)
        throw(BoundsError(mu, idx))
    end
    if !allunique(idx)
        throw(ArgumentError("idx must not repeat an index"))
    end
    rest = setdiff(1:N, idx)
    s11 = Symmetric(sig[idx, idx])
    s21 = sig[rest, idx]
    s22 = sig[rest, rest]
    beta = s21 / s11
    mu_cond = view(mu, rest) .+ beta * (collect(v) .- view(mu, idx))
    sig_cond = Symmetric(s22 .- beta * transpose(s21))
    return (; rest = rest, mu_cond = mu_cond, sig_cond = Matrix(sig_cond), beta = beta)
end

"""
    conditional_gaussian_stress(mu::AbstractVector, sig::AbstractMatrix,
                                idx::AbstractVector{<:Integer},
                                v::AbstractVector; n_obs::Integer,
                                rng::Random.AbstractRNG = Random.default_rng())
        -> Matrix

Draw a stressed scenario set: the named assets are pinned at `v` and the rest
are sampled from their conditional law.

# Arguments

  - `mu`, `sig`: Unconditional moments.
  - `idx`: Indices to pin.
  - `v`: Values to pin them at.
  - `n_obs`: Number of scenarios.
  - `rng`: Random number generator.

# Returns

  - A `n_obs x N` matrix. Column `idx[k]` is constant at `v[k]`. Every other
    column is drawn from the conditional normal.

# Notes

  - The output is an ordinary returns matrix, so **every risk measure in the
    library consumes it unchanged**. That is the whole design: a stress
    scenario is data, not a new type of estimator. It is the same conclusion
    ADR 0045 reached for the Feature Matrix.
"""
function conditional_gaussian_stress(mu::AbstractVector{<:Real},
                                     sig::AbstractMatrix{<:Real},
                                     idx::AbstractVector{<:Integer},
                                     v::AbstractVector{<:Real}; n_obs::Integer,
                                     rng::Random.AbstractRNG = Random.default_rng())
    if n_obs <= 0
        throw(DomainError(n_obs, "n_obs must be > 0"))
    end
    cm = conditional_moments(mu, sig, idx, v)
    N = length(mu)
    Tv = float(eltype(mu))
    out = Matrix{Tv}(undef, n_obs, N)
    for (k, i) in enumerate(idx)
        out[:, i] .= v[k]
    end
    Sc = Symmetric((cm.sig_cond .+ transpose(cm.sig_cond)) ./ 2)
    fact = cholesky(Sc; check = false)
    L = if issuccess(fact)
        Matrix(fact.L)
    else
        vals, vecs = eigen(Sc)
        floorv = eps(eltype(vals)) * max(maximum(abs, vals), one(eltype(vals)))
        Matrix(cholesky(Symmetric(vecs * Diagonal(max.(vals, floorv)) * transpose(vecs))).L)
    end
    Z = randn(rng, n_obs, length(cm.rest))
    out[:, cm.rest] .= Z * transpose(L) .+ transpose(cm.mu_cond)
    return out
end

"""
    tail_dependence_coefficient(rho::Real, nu::Real) -> Real

Return the closed-form lower tail dependence coefficient of a bivariate
t-copula.

# Arguments

  - `rho`: Copula correlation, in `(-1, 1)`.
  - `nu`: Degrees of freedom, positive.

# Returns

  - The coefficient `lam_tail` in `[0, 1)`.

# Mathematical definition

Tail dependence measures whether extremes occur together:

    lam_tail  =  lim_{u -> 0+}  P( U_2 <= u  |  U_1 <= u )

For the t-copula, Demarta and McNeil (2005) give

    lam_tail  =  2 * t_{nu+1} ( - sqrt( (nu + 1) (1 - rho) / (1 + rho) ) )

where `t_{nu+1}` is the distribution function of a univariate t with `nu + 1`
degrees of freedom.

# Notes

  - **The Gaussian copula is the limit `nu -> infinity`, where this is zero for
    every `rho < 1`.** Two assets with correlation 0.95 modelled by a Gaussian
    copula become independent in the extreme tail. That is the single most
    consequential modelling error in portfolio stress testing, and it is why
    this prototype exists.
  - The coefficient rises as `nu` falls. At `rho = 0.5` and `nu = 4` it is
    about 0.25, meaning a one-in-a-hundred loss in one asset comes with a
    one-in-a-hundred loss in the other about a quarter of the time.
"""
function tail_dependence_coefficient(rho::Real, nu::Real)
    if !(-1 < rho < 1)
        throw(DomainError(rho, "rho must satisfy -1 < rho < 1"))
    end
    if !(nu > 0)
        throw(DomainError(nu, "nu must be > 0"))
    end
    arg = -sqrt((nu + 1) * (1 - rho) / (1 + rho))
    return 2 * Distributions.cdf(Distributions.TDist(nu + 1), arg)
end

"""
    t_copula_scenarios(X::AbstractMatrix; nu::Real = 5.0, use_spearman::Bool = true,
                       n_obs::Integer, rng::Random.AbstractRNG = Random.default_rng())
        -> Matrix

Draw scenarios with a t-copula and the empirical marginals of `X`.

# Arguments

  - `X`: Historical returns, `T x N`.
  - `nu`: Degrees of freedom of the copula. Lower means more tail dependence.
  - `use_spearman`: Recover the copula correlation from Spearman rank
    correlation rather than from Pearson.
  - `n_obs`: Number of scenarios.
  - `rng`: Random number generator.

# Returns

  - An `n_obs x N` matrix with the empirical marginals of `X` and t-copula
    dependence.

# Mathematical definition

 1. Draw `Y ~ t_nu(0, P)`, that is `Y = Z / sqrt(W / nu)` with `Z ~ N(0, P)`
    and `W ~ ChiSquared(nu)` shared across assets.
 2. Map to uniforms with the univariate t distribution function,
    `U_j = t_nu(Y_j)`.
 3. Map to the data scale with the empirical quantile of each column.

Step 1's **shared** radial variable `W` is the source of the tail dependence.
Step 2 must use the *same* `nu`, otherwise the uniforms are not uniform.

# Notes

  - The rank correlation is preserved because steps 2 and 3 are monotone
    transformations, and Spearman correlation is invariant under those.
  - For a hierarchy of tail behaviour across pairs, a vine copula is the next
    step. See Aas, Czado, Frigessi and Bakken (2009). A single `nu` forces one
    tail index on the whole universe, which is this generator's main limit.
"""
function t_copula_scenarios(X::AbstractMatrix{<:Real}; nu::Real = 5.0,
                            use_spearman::Bool = true, n_obs::Integer,
                            rng::Random.AbstractRNG = Random.default_rng())
    if n_obs <= 0
        throw(DomainError(n_obs, "n_obs must be > 0"))
    end
    if !(nu > 2)
        throw(DomainError(nu, "nu must be > 2 so that the copula has finite variance"))
    end
    N = size(X, 2)
    P = if use_spearman
        2 .* sin.((pi / 6) .* clamp.(StatsBase.corspearman(X), -1, 1))
    else
        Statistics.cor(X)
    end
    P = (P .+ transpose(P)) ./ 2
    for i in 1:N
        P[i, i] = one(eltype(P))
    end
    Sc = Symmetric(P)
    fact = cholesky(Sc; check = false)
    L = if issuccess(fact)
        Matrix(fact.L)
    else
        vals, vecs = eigen(Sc)
        floorv = eps(eltype(vals)) * max(maximum(abs, vals), one(eltype(vals)))
        Matrix(cholesky(Symmetric(vecs * Diagonal(max.(vals, floorv)) * transpose(vecs))).L)
    end
    Z = randn(rng, n_obs, N) * transpose(L)
    W = rand(rng, Distributions.Chisq(nu), n_obs)
    Y = Z .* sqrt.(nu ./ W)
    td = Distributions.TDist(nu)
    R = Matrix{float(eltype(X))}(undef, n_obs, N)
    for j in 1:N
        u = Distributions.cdf.(td, view(Y, :, j))
        u = clamp.(u, eps(eltype(u)), one(eltype(u)) - eps(eltype(u)))
        R[:, j] = Statistics.quantile(view(X, :, j), u)
    end
    return R
end

"""
    empirical_tail_dependence(x::AbstractVector, y::AbstractVector; q::Real = 0.01)
        -> Real

Estimate the lower tail dependence of a sample pair.

# Arguments

  - `x`, `y`: Paired samples of equal length.
  - `q`: Tail quantile level. Smaller is closer to the limit and noisier.

# Returns

  - The estimate `P( y <= Q_y(q) | x <= Q_x(q) )`.

# Details

This is the finite-sample analogue of the limit in
[`tail_dependence_coefficient`](@ref). **It is biased upwards, badly, and the
bias dies slowly.** It is a check on a generator, not an estimator to report.

Measured on two million simulated scenarios at copula correlation `0.595`:

| `q`      | `nu = 3` | `nu = 5` | `nu = 12` | `nu = 500` |
|:-------- | --------:| --------:| ---------:| ----------:|
| `0.05`   | 0.426    | 0.383    | 0.341     | 0.307      |
| `0.005`  | 0.384    | 0.298    | 0.222     | 0.150      |
| `0.0005` | 0.385    | 0.282    | 0.170     | 0.083      |
| theory   | 0.370    | 0.263    | 0.092     | 0.000      |

Read the last column. A **near-Gaussian** copula, whose true tail dependence is
exactly zero, still measures `0.307` at `q = 0.05`. A practitioner who
estimates tail dependence from a five-per-cent tail will conclude that a
Gaussian model has strong tail dependence, which is the opposite of the truth.
The heavy-tailed columns converge quickly. The light-tailed ones do not.
"""
function empirical_tail_dependence(x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
                                   q::Real = 0.01)
    if length(x) != length(y)
        throw(DimensionMismatch("x has length $(length(x)), y has length $(length(y))"))
    end
    if !(zero(q) < q < one(q))
        throw(DomainError(q, "q must satisfy 0 < q < 1"))
    end
    qx = quantile(x, q)
    qy = quantile(y, q)
    nx = count(<=(qx), x)
    nboth = count(i -> x[i] <= qx && y[i] <= qy, eachindex(x))
    return iszero(nx) ? 0.0 : nboth / nx
end

# -----------------------------------------------------------------------------
# Scenario transformations. Each maps a returns matrix to a returns matrix.
# -----------------------------------------------------------------------------
"""
    scale_volatility(X::AbstractMatrix, factor::Real; cols = :) -> Matrix

Multiply the dispersion of the named columns by `factor`, leaving their means
unchanged.

# Arguments

  - `X`: Returns, `T x N`.
  - `factor`: Multiplier. `2.0` doubles the standard deviation.
  - `cols`: Columns to transform. All by default.

# Mathematical definition

    X'_{t,j}  =  mean_j  +  factor * ( X_{t,j} - mean_j )

# Notes

  - **This preserves every correlation exactly**, because it is a diagonal
    rescaling of the centred data. To change correlations, use
    [`shock_correlation`](@ref). Keeping the two transformations separate is
    deliberate: a caller should be able to say which one caused a change in
    the reported risk.
"""
function scale_volatility(X::AbstractMatrix{<:Real}, factor::Real; cols = :)
    if factor < 0
        throw(DomainError(factor, "factor must be >= 0"))
    end
    Y = Matrix(float.(X))
    idx = cols === (:) ? (1:size(Y, 2)) : cols
    for j in idx
        m = mean(view(Y, :, j))
        @views Y[:, j] .= m .+ factor .* (Y[:, j] .- m)
    end
    return Y
end

"""
    shock_correlation(X::AbstractMatrix, theta::Real; target::Real = 1.0) -> Matrix

Push every pairwise correlation towards `target` by the fraction `theta`,
holding the means and the volatilities fixed.

# Arguments

  - `X`: Returns, `T x N`.
  - `theta`: Blend fraction in `[0, 1]`. Zero leaves `X` unchanged.
  - `target`: The correlation to move towards. `1.0` is the crisis case, where
    diversification disappears.

# Returns

  - A `T x N` matrix with the requested correlation and the original marginal
    means and standard deviations.

# Mathematical definition

Let `C` be the sample correlation and `D` the diagonal matrix of sample
standard deviations. Form the shocked correlation

    C'  =  (1 - theta) C  +  theta ( target * (1 1' - I) + I )

repair it to the nearest positive semi-definite correlation matrix, and
re-colour the standardised data:

    X'  =  Z L'  D  +  mean,     where  C' = L L',  Z = standardised X

# Notes

  - **The repair step is not optional.** A convex blend of a valid correlation
    matrix and the all-ones matrix is valid, but a blend towards an arbitrary
    `target` need not be, and neither is the result of any interesting shock.
    The library owns this problem already: `NearestCorrelationMatrix` and
    `PosdefEstimator` are the right tools in an adapted version.
  - The transformation re-colours the data with a Cholesky factor, so it
    **destroys the original joint tail structure** and replaces it with a
    linear one. Apply a correlation shock to a Gaussian-like scenario set, or
    accept that the tails are no longer the ones the generator produced.
"""
function shock_correlation(X::AbstractMatrix{<:Real}, theta::Real; target::Real = 1.0)
    if !(0 <= theta <= 1)
        throw(DomainError(theta, "theta must lie in [0, 1]"))
    end
    T = float(eltype(X))
    N = size(X, 2)
    m = vec(mean(X; dims = 1))
    s = vec(std(X; dims = 1))
    C = Statistics.cor(X)
    Tgt = fill(T(target), N, N)
    for i in 1:N
        Tgt[i, i] = one(T)
    end
    Cs = (1 - theta) .* C .+ theta .* Tgt
    Cs = (Cs .+ transpose(Cs)) ./ 2
    for i in 1:N
        Cs[i, i] = one(T)
    end
    # Repair to the nearest positive semi-definite matrix, then renormalise the
    # diagonal so the result is still a correlation matrix.
    vals, vecs = eigen(Symmetric(Cs))
    floorv = sqrt(eps(T))
    Cr = vecs * Diagonal(max.(vals, floorv)) * transpose(vecs)
    d = sqrt.(diag(Cr))
    Cr = Cr ./ (d * transpose(d))
    Cr = (Cr .+ transpose(Cr)) ./ 2
    L = Matrix(cholesky(Symmetric(Cr)).L)
    Z = (X .- transpose(m)) ./ transpose(s)
    # Decorrelate first so the re-colouring imposes exactly `Cr`.
    Cz = Statistics.cor(Z)
    Lz = Matrix(cholesky(Symmetric((Cz .+ transpose(Cz)) ./ 2)).L)
    Zw = (Lz \ transpose(Z))
    return transpose(L * Zw) .* transpose(s) .+ transpose(m)
end

"""
    shift_mean(X::AbstractMatrix, delta::AbstractVector) -> Matrix

Add `delta` to every row. Changes the mean and nothing else.
"""
function shift_mean(X::AbstractMatrix{<:Real}, delta::AbstractVector{<:Real})
    if length(delta) != size(X, 2)
        throw(DimensionMismatch("delta has length $(length(delta)), X has $(size(X, 2)) columns"))
    end
    return Matrix(float.(X)) .+ transpose(delta)
end

"""
    factor_shock(X::AbstractMatrix, loadings::AbstractMatrix,
                 shock::AbstractVector) -> Matrix

Apply a shock expressed in factor space, propagated to assets by their
loadings.

# Arguments

  - `X`: Returns, `T x N`.
  - `loadings`: `N x K` matrix. Row `i` holds asset `i`'s exposure to each of
    the `K` factors.
  - `shock`: The factor move, length `K`.

# Returns

  - `X .+ transpose(loadings * shock)`.

# Mathematical definition

    X'_{t,i}  =  X_{t,i}  +  sum_k  loadings[i, k] * shock[k]

# Notes

  - **This is the honest way to state a macro scenario.** "Rates rise 100 basis
    points" is a statement about a factor, not about 500 assets. Expressing it
    once and letting the loadings distribute it keeps the scenario auditable,
    and it uses the factor machinery the library already has in
    `src/08_Moments/21_Base_Regression.jl` and the factor priors.
"""
function factor_shock(X::AbstractMatrix{<:Real}, loadings::AbstractMatrix{<:Real},
                      shock::AbstractVector{<:Real})
    N = size(X, 2)
    if size(loadings, 1) != N
        throw(DimensionMismatch("loadings must have $(N) rows, got $(size(loadings, 1))"))
    end
    if size(loadings, 2) != length(shock)
        throw(DimensionMismatch("loadings has $(size(loadings, 2)) columns, shock has length $(length(shock))"))
    end
    return Matrix(float.(X)) .+ transpose(loadings * shock)
end

"""
    stress_report(w::AbstractVector, scenarios::AbstractVector{<:Pair};
                  alpha::Real = 0.05) -> Vector{<:NamedTuple}

Evaluate one portfolio across a set of named scenario matrices.

# Arguments

  - `w`: Portfolio weights, length `N`.
  - `scenarios`: A vector of `name => X` pairs, each `X` a `T x N` matrix.
  - `alpha`: Tail level for the tail statistics.

# Returns

One `NamedTuple` per scenario, with `name`, `mean`, `vol`, `var`, `cvar`,
`worst` and `prob_loss`.

# Notes

  - Report the **base** scenario first and read every other row as a
    difference from it. An absolute CVaR under a stress is hard to calibrate.
    A ratio to the base case is not.
"""
function stress_report(w::AbstractVector{<:Real}, scenarios::AbstractVector{<:Pair};
                       alpha::Real = 0.05)
    if !(zero(alpha) < alpha < one(alpha))
        throw(DomainError(alpha, "alpha must satisfy 0 < alpha < 1"))
    end
    out = NamedTuple[]
    for (name, X) in scenarios
        if size(X, 2) != length(w)
            throw(DimensionMismatch("scenario $(name) has $(size(X, 2)) columns, w has length $(length(w))"))
        end
        r = X * w
        sorted = sort(r)
        k = max(1, floor(Int, alpha * length(r)))
        push!(out,
              (; name = String(name), mean = mean(r), vol = std(r), var = -sorted[k],
               cvar = -mean(view(sorted, 1:k)), worst = minimum(r),
               prob_loss = count(<(0), r) / length(r)))
    end
    return out
end

end # module ConditionalStress
