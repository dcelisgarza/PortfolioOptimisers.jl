#=
The checks of this subject are split over the files `test_08y_regime_adjusted_covariance*.jl`, so that each file of the
suite runs in about the same time. This file holds one part of them.

`RegimeAdjustedExpWeightedCovariance` answers the verbs its Choice Surface promises.

The type entered the library with a docstring that stated the mathematics and no code behind it.
Issue #637 held that gap: `cov` and `cor` reached the covariance surface's own fallback, which
reads `cor`, which reads `cov` back through `StatsBase`'s generic method, so a caller met a
`StackOverflowError` rather than a matrix.

The recursion is a port of the oracle, so this file is checked against the oracle
itself. `ORACLE_X` and the five matrices below were measured by fitting the oracle
on that fixture, one matrix per configuration, and pasted here as literals.

`oracle_estimator` names three keywords of the oracle: it clamps the multiplier to
`(0.7, 1.6)`, it assumes centred returns, and its numerical floor is `1e-12`. Since #1383 the
clamp and the floor are this estimator's defaults too, and only the centring differs. Every
other keyword is the same value on both sides.

Three of the five multipliers land inside the clamp and two are clamped at its upper bound, so
the file covers both the free and the clamped branch of the read-out.

The remaining testsets pin what the oracle cannot: that an incremental fit is exact, that the
two masks blank what they should, that the state answers the same as the sample, and that the
estimator refuses what it cannot fit.
=#
using Test, PortfolioOptimisers, Statistics, LinearAlgebra, StableRNGs
include(joinpath(@__DIR__, "parity_harness.jl"))

const PO = PortfolioOptimisers

# ---------------------------------------------------------------------- the oracle

const ORACLE_X = [0.000684 0.027195 0.024494;
                  -0.010206 -0.005959 -0.010548;
                  0.011395 -0.001121 0.014938;
                  -0.036946 0.031331 -0.001929;
                  0.013608 -0.002731 -0.007582;
                  0.009262 0.01649 -0.004051;
                  -0.003056 0.013714 -0.017407;
                  -0.030288 0.0079 -0.013411;
                  -0.038407 -0.016281 -0.009352;
                  -0.023864 -0.029849 0.000733;
                  0.017945 -0.004663 -0.014872;
                  0.0077 0.014345 -0.006;
                  0.01525 0.029201 -0.005795;
                  -0.022778 0.009734 0.006931;
                  0.030766 -0.035969 -0.018525;
                  -0.023468 -0.048552 0.003541]

# Regime multiplier 1.2740841202371018, inside the clamp.
const ORACLE_DEFAULTS = [0.0008286145038843979 0.00011937441370453566 -0.00011402629309657688;
                         0.00011937441370453566 0.0011164047529160048 7.572490331233897e-05;
                         -0.00011402629309657688 7.572490331233897e-05 0.00020053608580165147]

# Regime multiplier 1.6, clamped at the upper bound.
const ORACLE_MAHALANOBIS_LOG = [0.001306761378765317 0.00018825868085891418 -0.00017982446032970844;
                                0.00018825868085891418 0.001760619211155158 0.00011942149044630743;
                                -0.00017982446032970844 0.00011942149044630743 0.00031625419389343155]

# Regime multiplier 1.6, clamped at the upper bound.
const ORACLE_DIAGONAL_RMS_HAC = [0.0011021927661016959 0.00052240520185323 0.00020430938799968898;
                                 0.00052240520185323 0.002344356715749304 0.00013186063811999454;
                                 0.00020430938799968898 0.00013186063811999454 0.00039051388831200663]

# Regime multiplier 1.0183628958663742, inside the clamp.
const ORACLE_SEPARATE_CORR = [0.0005293726014233321 0.00011202774816653493 8.601801777603576e-05;
                              0.00011202774816653493 0.0007132316481573059 7.834237598024428e-05;
                              8.601801777603576e-05 7.834237598024428e-05 0.00012811543718148962]

# Regime multiplier 1.206989861102959, inside the clamp.
const ORACLE_UNCENTRED = [0.000801971157898267 8.962017349379358e-05 -0.00014077995961418242;
                          8.962017349379358e-05 0.0010776296766476458 4.272920057362547e-05;
                          -0.00014077995961418242 4.272920057362547e-05 0.00016979430933553749]

function oracle_estimator(; kwargs...)
    return RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 3,
                                               regime_decay = exp2(-2 / inv(log2(inv(0.9)))),
                                               regime_min_obs = 2,
                                               regime_lohi_mult = (0.7, 1.6),
                                               centring = PreCentred(), min_val = 1e-12,
                                               debias = RawStatistic(), kwargs...)
end

#=
Issue #1430. The default `PortfolioTarget()` builds its inverse-volatility direction from the
estimate that it divides by, so the direction and the error of the estimate are correlated, and
the factor of a fixed direction leaves 1.058 at 12 assets and a half-life of 10. `debias = ExactDebias()`
also divides that direction by `1 + Δ`, its second-order excess, read on the estimated
correlation of the block (ADR 0190).
=#
@testset "the inverse-volatility direction divides by its second-order excess" begin
    # The sums of the products of the normalised weights, against the sums written out.
    lambda, slow = 2.0^(-1 / 10), 2.0^(-1 / 20)
    wts(d, K) = (w = [(1 - d) * d^j for j in 0:(K - 1)]; w ./ sum(w))
    for K in (5, 17, 400)
        @test isapprox(PO.exp_weight_cross_sum(lambda, lambda, K),
                       sum(abs2, wts(lambda, K)); rtol = 1e-12)
        @test isapprox(PO.exp_weight_cross_sum(lambda, slow, K),
                       dot(wts(lambda, K), wts(slow, K)); rtol = 1e-12)
    end
    @test isapprox(PO.exp_weight_cross_sum(lambda, lambda, 10^6),
                   (1 - lambda) / (1 + lambda); rtol = 1e-12)

    # The excess against the formula written with matrices, on a correlation that is not
    # uniform. It reads the correlation alone, so a scale of the volatilities does not move it.
    rng = StableRNG(1430)
    na = 12
    F = randn(rng, na, 3)
    S = F * transpose(F) + Diagonal(0.3 .+ rand(rng, na))
    Rho = S ./ sqrt.(diag(S) .* transpose(diag(S)))
    vol = 0.5 .+ rand(rng, na)
    sv, svc = 0.03, 0.02
    c = vec(sum(Rho; dims = 2))
    A, T = sum(c), sum(c .^ 3)
    B, S3 = dot(c, (Rho .^ 2) * c), sum(Rho .^ 3)
    delta = sv * (1 + S3 / A - 2 * B / A^2) + svc * (1 - S3 / A - 2 * T / A^2 + 2 * B / A^2)
    @test isapprox(PO.inverse_volatility_bias(vol .* Rho .* transpose(vol), sv, svc, 1e-12),
                   delta; rtol = 1e-12)
    # On one decay it is 2 s (1 - T / A²): 2 s (1 - 1/n) at R = I, and zero at a correlation of
    # one, where the estimated direction is proportional to the true one.
    @test isapprox(PO.inverse_volatility_bias(Rho, sv, sv, 1e-12), 2 * sv * (1 - T / A^2);
                   rtol = 1e-12)
    @test isapprox(PO.inverse_volatility_bias(Matrix(1.0I, na, na), sv, sv, 1e-12),
                   2 * sv * (1 - 1 / na); rtol = 1e-12)
    @test abs(PO.inverse_volatility_bias(vol .* transpose(vol), sv, sv, 1e-12)) < 1e-14
    # A block whose inverse-volatility portfolio has no variance takes no excess.
    @test iszero(PO.inverse_volatility_bias([1.0 -1.0; -1.0 1.0], sv, sv, 1e-12))

    # On iid Normal returns the squared multiplier is one when the statistic is correct.
    # Measured on this seed, half-life 10: 0.999 (RMS) and 0.982 (first moment) debiased, 1.110
    # and 1.073 raw, and 1.013 (RMS) on the separate path. Eight seeds of this fixture: 1.0009 ±
    # 0.0013 (RMS), 1.0042 ± 0.0051 (first moment) and 1.013 ± 0.019 (log, too noisy to pin).
    U = cholesky(Symmetric(S)).U
    R = randn(rng, 12000, na) * U .* 0.01
    base = (; decay = lambda, min_obs = 5, regime_lohi_mult = nothing,
            regime_decay = 2.0^(-1 / 500), regime_min_obs = 1, centring = PreCentred())
    function covariance_multiplier(method; extra...)
        on = RegimeAdjustedExpWeightedCovariance(; base..., extra...,
                                                 regime_method = method)
        off = RegimeAdjustedExpWeightedCovariance(; base..., extra...,
                                                  regime_method = nothing)
        return mean(cov(on, R) ./ cov(off, R))
    end
    @test 0.99 < covariance_multiplier(PO.RootMeanSquaredAdjusted()) < 1.01
    @test 0.97 < covariance_multiplier(PO.FirstMomentRegimeAdjusted()) < 1.03
    @test covariance_multiplier(PO.RootMeanSquaredAdjusted(); debias = RawStatistic()) >
          1.08
    @test 0.98 <
          covariance_multiplier(PO.RootMeanSquaredAdjusted(); cor_decay = 2.0^(-1 / 20)) <
          1.02
end

#=
Issue #1432. The diagonal sum `S = Σ_i z_i²` of returns with the correlation `R` has the law
`Σ_k μ_k χ²_k(1)` on the eigenvalues of `R`, so its root and its log read the correlation. The
constants `E[χ(n)]` and `ψ(x n) + ln y` do not, so `DiagonalTarget` divides the sum by the factor of its
law, on the estimated spectrum shrunk until its dispersion is unbiased (ADR 0190).
=#
@testset "the diagonal statistic divides by the factor of its law" begin
    SF = PO.SpecialFunctions
    FM, LG = PO.FirstMomentRegimeAdjusted(), PO.LogRegimeAdjusted()
    chi_root(n) = sqrt(2) * exp(SF.loggamma((n + 1) / 2) - SF.loggamma(n / 2))
    # The law is known at independent squares, a χ²(n) variate, and at one eigenvalue n, n times
    # a χ²(1) variate. Both factors are one at μ = 1, because each constant is the expectation at
    # independent squares: the root of a χ²(n) variate for the first moment, and the Log's
    # whatever its parameters, because it takes each square as the Gamma(x, y) variate that its
    # constant assumes. Measured: 1e-13 or less.
    for n in (1, 2, 12, 200)
        spike = [n; zeros(n - 1)]
        @test isapprox(PO.regime_denom(FM, PO.DiagonalTarget(), n), chi_root(n);
                       rtol = 1e-14)
        @test isapprox(PO.regime_law_factor(FM, ones(n)), 1; rtol = 1e-12)
        @test isapprox(PO.regime_law_factor(FM, spike), 2n / (pi * chi_root(n)^2);
                       rtol = 1e-12)
        @test isapprox(PO.regime_law_factor(LG, ones(n)), 1; rtol = 1e-12)
        @test isapprox(PO.regime_law_factor(PO.LogRegimeAdjusted(; x = 1.3, y = 0.7),
                                            ones(n)), 1; rtol = 1e-12)
        @test isapprox(PO.regime_law_factor(LG, spike),
                       exp(log(n) + SF.digamma(0.5) - SF.digamma(n / 2)); rtol = 1e-12)
    end

    rng = StableRNG(1432)
    X = randn(rng, 200, 4) *
        [1.0 0.5 0.3 0.0; 0.0 1.0 0.4 0.2; 0.0 0.0 1.0 0.6; 0.0 0.0 0.0 1.0] .* 0.01
    idx = 1:4
    lambda = 2.0^(-1 / 10)
    for (lam, extra) in ((lambda, (;)), (2.0^(-1 / 20), (; cor_decay = 2.0^(-1 / 20))))
        ce = RegimeAdjustedExpWeightedCovariance(; decay = lambda, min_obs = 5, extra...,
                                                 regime_target = PO.DiagonalTarget(),
                                                 regime_method = FM)
        cache = partial_fit!(ce, X).cache
        C = PO.regime_covariance_block(cache, ce, idx)
        # Each pair read all 200 rows, so `1/K` is the sum of the squared normalised weights of
        # its terms, and the shrunk spectrum keeps the trace and has the unbiased dispersion
        # `Σ_{i≠j} r²`. On the separate path the correlation reads `cor_decay`. The estimated
        # location takes no term from the first row, so the terms are 199.
        nterms = 200 - PO.centring_lag(ce.centring)
        w = [(1 - lam) * lam^j for j in 0:(nterms - 1)]
        w ./= sum(w)
        rho = C ./ sqrt.(diag(C) * transpose(diag(C)))
        q = sum(rho[i, j]^2 - (1 - rho[i, j]^2)^2 * sum(abs2, w)
                for i in idx, j in idx if i != j)
        Rs = PO.diagonal_law_correlation(cache, ce, C, idx)
        mu = eigvals(Symmetric(Rs))
        @test isapprox(sum(mu), 4; rtol = 1e-12) &&
              isapprox(diag(Rs), ones(4); rtol = 1e-14)
        @test isapprox(sum(abs2, mu .- 1), q; rtol = 1e-10)
        @test sum(abs2, mu .- 1) < sum(abs2, eigvals(Symmetric(rho)) .- 1)
        # A block without correlation keeps the identity, and a block of correlation one keeps
        # its single eigenvalue, because a sample correlation of one has no variance.
        @test PO.diagonal_law_correlation(cache, ce, Matrix(Diagonal([1.0, 2.0, 3.0, 4.0])),
                                          idx) == I(4)
        v = [1.0, 2.0, 3.0, 4.0]
        @test isapprox(eigvals(Symmetric(PO.diagonal_law_correlation(cache, ce,
                                                                     v * transpose(v), idx))),
                       [0, 0, 0, 4]; atol = 1e-12)

        # The mean's method needs no factor, and the raw statistic takes none. The debiased
        # factor reads the moments of each term at the count of its asset (#1434).
        raw = RegimeAdjustedExpWeightedCovariance(; decay = lambda, min_obs = 5, extra...,
                                                  regime_target = PO.DiagonalTarget(),
                                                  regime_method = FM,
                                                  debias = RawStatistic())
        m = [PO.regime_bias!(cache.bias.moments, PO.RegimeTermMoments(FM), lambda, k)
             for k in cache.obs_count[idx]]
        @test PO.diagonal_law_factor(PO.RootMeanSquaredAdjusted(), cache, ce, C, idx, m) ===
              1.0
        @test PO.diagonal_law_factor(FM, cache, raw, C, idx, m) === 1.0
        sb, sv = sqrt.(getindex.(m, 2)), sqrt.(getindex.(m, 3))
        E = eigen(Symmetric(sb .* Rs .* transpose(sb)))
        @test PO.diagonal_law_factor(FM, cache, ce, C, idx, m) ==
              PO.regime_law_factor(FM, max.(E.values, 0), E.vectors,
                                   sv .* Rs .^ 2 .* transpose(sv))
        @test PO.diagonal_law_factor(LG, cache, ce, C, idx, m) < 1
    end
end

#=
Issue #1434. Each term of the diagonal sum carries the noise `a_i = 1 / (f Q_i)` of its estimated
variance, and the correlation of the assets correlates that noise, so the per-term factor of the
mean left 0.986 and 0.971 at a correlation of 0.9. The factor now expands the root and the log in
`a^(1/2)` and `ln a`, the variables in which the statistic is linear along the direction where
every term carries the same noise (ADR 0190).
=#
@testset "the diagonal factor reads the noise of each estimate" begin
    SF = PO.SpecialFunctions
    FM, LG = PO.FirstMomentRegimeAdjusted(), PO.LogRegimeAdjusted()
    # At equal weights `Q` is χ²(K)/K, so the moments of one term have closed forms. A half-life of
    # 1e6 makes the 30 weights equal to within 2e-5.
    K = 30
    eq = 2.0^(-1 / 1e6)
    f = K / (K - 2)
    qf = (sqrt(K / 2) * SF.gamma((K - 1) / 2) / SF.gamma(K / 2))^2
    ql = exp(log(K / 2) - SF.digamma(K / 2))
    tfm = PO.regime_bias_table(PO.RegimeTermMoments(FM), eq, K)[K]
    tlg = PO.regime_bias_table(PO.RegimeTermMoments(LG), eq, K)[K]
    trm = PO.regime_bias_table(PO.RegimeTermMoments(PO.RootMeanSquaredAdjusted()), eq, K)[K]
    @test all(isapprox.(tfm, (f, qf / f, f / qf - 1); rtol = 1e-10))
    @test all(isapprox.(tlg, (f, ql / f, SF.trigamma(K / 2)); rtol = 1e-10))
    @test isapprox(trm[1], f; rtol = 1e-11) && trm[2:3] == (1.0, 0.0)
    # A diagonal state holds the triples, and grows them as a table of factors grows.
    bias = NTuple{3, Float64}[]
    @test PO.regime_bias!(bias, PO.RegimeTermMoments(FM), 0.9, 10) == bias[10]
    @test length(bias) == 64

    # The correction vanishes where every term carries the same noise: at one asset the factor is
    # the scalar method's, and at a correlation of one it is the law of one eigenvalue at the scale
    # b. Measured: 2e-15 or less.
    b, v = 0.93, 0.07
    @test isapprox(PO.regime_law_factor(FM, [b], ones(1, 1), fill(v, 1, 1)), b;
                   rtol = 1e-12)
    @test isapprox(PO.regime_law_factor(LG, [b], ones(1, 1), fill(v, 1, 1)), b;
                   rtol = 1e-12)
    n = 6
    Es = eigen(Symmetric(fill(b, n, n)))
    for method in (FM, LG, PO.LogRegimeAdjusted(; x = 1.3, y = 0.7))
        @test abs(PO.regime_law_correction(method, max.(Es.values, 0), Es.vectors,
                                           fill(v, n, n))) < 1e-14
    end

    # The reduction of the quadratic sum against the sum over every node written with matrices,
    # on a correlation that is not uniform and on unequal scales.
    rng = StableRNG(1434)
    F = randn(rng, n, 2)
    S = F * transpose(F) + Diagonal(0.5 .+ rand(rng, n))
    R = S ./ sqrt.(diag(S) .* transpose(diag(S)))
    sb, sv = sqrt.(0.9 .+ 0.05 .* rand(rng, n)), sqrt.(0.05 .+ 0.03 .* rand(rng, n))
    Rb = sb .* R .* transpose(sb)
    Cm = sv .* R .^ 2 .* transpose(sv)
    E = eigen(Symmetric(Rb))
    for method in (FM, LG, PO.LogRegimeAdjusted(; x = 1.3, y = 0.7))
        a, nu = PO.regime_law_shape(method)
        x = PO.regime_law_grid(Float64)
        lin, quad = 0.0, 0.0
        for xi in x
            tau = exp(xi) / n
            M = Rb / (I + 2tau * Rb)
            L = det(I + 2tau * Rb)^(-a)
            wl, wq = PO.regime_law_weights(method, tau)
            lin += wl * L * dot(diag(Cm), diag(M))
            quad += wq *
                    L *
                    sum(Cm .* (nu^2 .* diag(M) .* transpose(diag(M)) .+ 2nu .* M .^ 2))
        end
        @test isapprox(PO.regime_law_correction(method, E.values, E.vectors, Cm),
                       step(x) * (lin - quad) / 2; rtol = 1e-10)
    end

    # Through the library: on iid Normal returns the squared multiplier is one when the statistic
    # is correct. The #1428 fixture read 0.995 (first moment) and 0.989 (log) over 8 seeds after
    # #1432, and reads 1.0001 and 0.9997 now (0.9995 and 1.0004 on this seed, 0.9993 and 1.0001
    # on the separate path). Under two HAC lags the moments of each term come from the table of
    # the banded weight matrix (#1433): 0.990 and 0.991 on this seed, and 1.002 and 0.999 over 8
    # seeds of the #1433 fixture, whose standard error is 0.004.
    rng = StableRNG(1428)
    na, nr = 12, 12000
    A = randn(rng, na, na)
    U = cholesky(Symmetric(A * transpose(A) / na + Diagonal(rand(rng, na)))).U
    Rt = randn(rng, nr, na) * U .* 0.01
    base = (; decay = 2.0^(-1 / 10), min_obs = 5, regime_lohi_mult = nothing,
            regime_decay = 2.0^(-1 / 500), regime_min_obs = 1, centring = PreCentred(),
            regime_target = PO.DiagonalTarget())
    function multiplier(method; extra...)
        on = RegimeAdjustedExpWeightedCovariance(; base..., extra...,
                                                 regime_method = method)
        off = RegimeAdjustedExpWeightedCovariance(; base..., extra...,
                                                  regime_method = nothing)
        return mean(cov(on, Rt) ./ cov(off, Rt))
    end
    for method in (FM, LG), extra in ((;), (; cor_decay = 2.0^(-1 / 20)))
        @test 0.99 < multiplier(method; extra...) < 1.01
    end
    for method in (FM, LG)
        @test 0.97 < multiplier(method; hac_lags = 2) < 1.03
    end
end

#=
Issue #1433. A HAC estimate over `K` observations is the quadratic form `z' A z` in the returns,
with the banded weight matrix `A`, so it has the law of a plain estimate whose weights are the
eigenvalues of `A`. Every bias factor reads that spectrum, without an eigen-decomposition: the
tables and the Mahalanobis fixed point from a banded LDLᵀ factorisation of `I + s A`, and the
effective count from `tr(A²)`. `A` is indefinite, so the gate also needs `1 / tr(A²) > n + 1`.
The per-term floor at zero made the variance 16 % too large at two lags, so it is off by default,
and `hac_floor = PerTermHacFloor()` keeps it (ADR 0190).
=#
@testset "a HAC estimate reads the spectrum of its weight matrix" begin
    function hac_A(lam, L, K)
        B = diagm(lam .^ (0:(K - 1)))
        for i in 1:L, j in 0:(K - i - 1)
            B[j + 1, j + i + 1] = B[j + i + 1, j + 1] = lam^j * (1 - i / (L + 1))
        end
        return (1 - lam) / (1 - lam^K) * B
    end
    # The table, against the eigenvalues on a grid twenty times finer. Where `A` is positive
    # definite the two agree to 6e-13; where it is not, to 5e-7 (measured 5.4e-7).
    function eigen_factor(method, lam, L, K)
        A = hac_A(lam, L, K)
        mu = eigvals(Symmetric(A)) ./ ((1 - lam) / (1 - lam^K))
        s = exp.(range(-75, 60; step = 1 / 200))
        lG = map(x -> (v = 1 .+ x .* mu; any(<=(0), v) ? -Inf : -sum(log, v) / 2), s)
        return PO.regime_bias_factor(method, s, PO.hac_laplace!(similar(s), lG),
                                     (1 - lam) / (1 - lam^K), 1 / 200)
    end
    for (hl, L) in ((10, 1), (10, 2), (40, 5)),
        method in (PO.RootMeanSquaredAdjusted(), PO.FirstMomentRegimeAdjusted(),
                   PO.LogRegimeAdjusted())

        lam = 2.0^(-1 / hl)
        table = PO.regime_bias_table(method, lam, 200, L)
        for K in (5, 20, 200)
            definite = minimum(eigvals(Symmetric(hac_A(lam, L, K)))) > 0
            @test isapprox(table[K], eigen_factor(method, lam, L, K);
                           rtol = definite ? 1e-11 : 1e-6)
        end
    end
    # At two lags and a half-life of 10 the steady-state factor is 1.148, twice the excess of the
    # plain weights' 1.071, which a Monte Carlo of 400 000 draws reads as 1.146.
    lam = 2.0^(-1 / 10)
    @test isapprox(PO.regime_bias_table(PO.RootMeanSquaredAdjusted(), lam, 200, 2)[200],
                   1.1476; atol = 1e-4)
    # The state grows the HAC table where the estimator has HAC lags.
    bias = Float64[]
    @test PO.regime_bias!(bias, PO.RootMeanSquaredAdjusted(), lam, 30, 2) ==
          PO.regime_bias_table(PO.RootMeanSquaredAdjusted(), lam, 64, 2)[30]

    # The traces of the products of the weight matrices, and of the square of a pair's, against
    # the matrices written out.
    for L in (1, 2, 5), K in (3, 17, 300)
        @test isapprox(PO.exp_weight_cross_sum(0.9, 0.95, K, L),
                       tr(hac_A(0.9, L, K) * hac_A(0.95, L, K)); rtol = 1e-12)
        @test isapprox(PO.pair_weight_square_sum(0.9, 1 - 0.9^K, L),
                       sum(abs2, hac_A(0.9, L, K)); rtol = 1e-12)
    end
    @test PO.exp_weight_cross_sum(0.9, 0.95, 17, nothing) ==
          PO.exp_weight_cross_sum(0.9, 0.95, 17)

    # The two sums of the Mahalanobis fixed point, against the eigenvalues, and the fixed point.
    for K in (20, 600), L in (1, 5), t in (1.0, 30.0)
        mu = eigvals(Symmetric(hac_A(lam, L, K)))
        slopes = PO.hac_log_det_slopes(lam, K, t, L)
        @test isapprox(slopes[1], sum(mu ./ (1 .+ t .* mu)); rtol = 1e-12)
        @test isapprox(slopes[2], sum(mu ./ (1 .+ t .* mu) .^ 2); rtol = 1e-12)
    end
    mu = eigvals(Symmetric(hac_A(lam, 2, 200)))
    b = PO.mahalanobis_bias(lam, 200, 12, 2)
    @test isapprox(inv(b), sum(mu ./ (1 .+ 13 .* mu .* b)); rtol = 1e-12)
    # A Monte Carlo reads 2.479 there, and the plain fixed point 1.623. At 17 observations the
    # estimate is not positive definite in 11 % of draws, and no root exists.
    @test isapprox(b, 2.523; atol = 1e-3)
    @test isnothing(PO.mahalanobis_bias(lam, 17, 12, 2))

    # The gate: 12 assets at a half-life of 10 and two lags first score at 54 observations;
    # at a half-life of 5 the steady state has 6.8 effective observations, and never scores.
    first_open(n, d, L) = findfirst(K -> PO.regime_bias_open(ExactDebias(), n, d, K, L),
                                    1:2000)
    @test first_open(12, lam, 2) == 54
    @test first_open(1, lam, 2) == 5
    @test isnothing(first_open(12, 2.0^(-1 / 5), 2))
    @test first_open(12, lam, nothing) == 16

    # Without the floor the scalar HAC variance is the diagonal of the HAC covariance of one
    # decay, as the plain variance is. The floor breaks that identity.
    rng = StableRNG(1433)
    na, nr = 12, 12000
    A = randn(rng, na, na)
    U = cholesky(Symmetric(A * transpose(A) / na + Diagonal(rand(rng, na)))).U
    R = randn(rng, nr, na) * U .* 0.01
    plain = (; decay = lam, min_obs = 5, regime_method = nothing, centring = PreCentred(),
             hac_lags = 2)
    @test isapprox(var(RegimeAdjustedExpWeightedVariance(; plain...), R),
                   diag(cov(RegimeAdjustedExpWeightedCovariance(; plain...), R));
                   rtol = 1e-12)
    @test !isapprox(var(RegimeAdjustedExpWeightedVariance(; plain...,
                                                          hac_floor = PerTermHacFloor()),
                        R), diag(cov(RegimeAdjustedExpWeightedCovariance(; plain...), R));
                    rtol = 1e-3)
    # The separate path floors its variance only under `hac_floor = PerTermHacFloor()` too.
    sep = (; plain..., cor_decay = 2.0^(-1 / 20))
    @test cov(RegimeAdjustedExpWeightedCovariance(; sep...), R) !=
          cov(RegimeAdjustedExpWeightedCovariance(; sep..., hac_floor = PerTermHacFloor()),
              R)

    # On iid Normal returns the squared multiplier is one when the statistic is correct.
    base = (; decay = lam, min_obs = 5, regime_lohi_mult = nothing, hac_lags = 2,
            regime_decay = 2.0^(-1 / 500), regime_min_obs = 1, centring = PreCentred())
    function covariance_multiplier(target, method; extra...)
        on = RegimeAdjustedExpWeightedCovariance(; base..., extra...,
                                                 regime_target = target,
                                                 regime_method = method)
        off = RegimeAdjustedExpWeightedCovariance(; base..., extra...,
                                                  regime_method = nothing)
        return mean(cov(on, R) ./ cov(off, R))
    end
    function variance_multiplier(method; extra...)
        on = RegimeAdjustedExpWeightedVariance(; base..., extra..., regime_method = method)
        off = RegimeAdjustedExpWeightedVariance(; base..., extra...,
                                                regime_method = nothing)
        return mean(var(on, R) ./ var(off, R))
    end
    fixed = PO.PortfolioTarget(; w = fill(1 / na, na))
    # Measured on this seed: scalar 1.000 (RMS) and 0.987 (first moment), 1.147 raw, and 0.837
    # with the floor, which the factor then over-corrects; the equal-weight direction 1.013, from
    # 1.162 raw; Mahalanobis 1.007, from 2.49 raw, since #1438 gave it the recursion on the
    # spectrum of `A` (0.988 with the mean's fixed point). Over 8 seeds: 1.002, 1.003, 1.015 and
    # 1.000 (0.981 before #1438).
    RMS, FM = PO.RootMeanSquaredAdjusted(), PO.FirstMomentRegimeAdjusted()
    for method in (RMS, FM)
        @test 0.97 < variance_multiplier(method) < 1.03
    end
    @test variance_multiplier(RMS; debias = RawStatistic()) > 1.1
    @test variance_multiplier(RMS; hac_floor = PerTermHacFloor()) < 0.9
    @test 0.97 < covariance_multiplier(fixed, RMS) < 1.03
    @test covariance_multiplier(fixed, RMS; debias = RawStatistic()) > 1.1
    @test 0.97 < covariance_multiplier(PO.MahalanobisTarget(), RMS; min_obs = 17) < 1.03
    @test covariance_multiplier(PO.MahalanobisTarget(), RMS; min_obs = 17,
                                debias = RawStatistic()) > 2
end

#=
A row with no valid asset is one lag of the HAC products. The variance estimator returned before
it pushed the row into its buffer, so the returns on the two sides of a holiday read as one lag
apart, and its variance left the diagonal of the covariance, which kept the row.
=#
@testset "a row with no valid asset is one lag of the HAC products" begin
    # Two finite returns two observations apart give no product at one lag, so the variance is
    # (λ r₁² + r₃²) / (1 + λ). The defect read 2 ω₁ r₁ r₃ more, 4.33e-4 for 3e-4.
    lam = 0.5
    kw = (; decay = lam, min_obs = 1, regime_method = nothing, centring = PreCentred(),
          hac_lags = 1)
    X = reshape([0.01, NaN, 0.02], :, 1)
    expected = (lam * 0.01^2 + 0.02^2) / (1 + lam)
    @test isapprox(var(RegimeAdjustedExpWeightedVariance(; kw...), X)[1], expected;
                   rtol = 1e-14)
    online = partial_fit!(RegimeAdjustedExpWeightedVariance(; kw...), X[1:1, :])
    v1 = copy(online.cache.variance)
    online = partial_fit!(online, X[2:2, :])
    # The holiday holds every estimate and every count, and enters the buffer as one lag.
    @test online.cache.variance == v1 && online.cache.obs_count == [1]
    @test all(isnan, only(online.cache.ret_buffer))
    online = partial_fit!(online, X[3:3, :])
    @test isapprox(var(online)[1], expected; rtol = 1e-14)

    # An asset reads the same lags whether a neighbour is valid on its holidays or not, under
    # both centrings, at one and three lags, and across one and three holidays.
    for centring in (PreCentred(), EstimatedCentring()), L in (1, 3), gap in (1, 3)
        x = [0.01; fill(NaN, gap); -0.02; 0.015; fill(NaN, gap); 0.03; 0.005; -0.01]
        kw = (; decay = 2.0^(-1 / 5), min_obs = 1, regime_method = nothing, centring,
              hac_lags = L)
        alone = var(RegimeAdjustedExpWeightedVariance(; kw...), reshape(x, :, 1))[1]
        mixed = var(RegimeAdjustedExpWeightedVariance(; kw...),
                    hcat(x, 0.01 .* sin.(axes(x, 1))))[1]
        @test isapprox(alone, mixed; rtol = 1e-13)
    end

    # On a panel with holidays the scalar HAC variance is still the diagonal of the HAC
    # covariance of one decay.
    R = randn(StableRNG(1795), 400, 4) .* 0.01
    R[[30, 31, 120, 250, 251, 252], :] .= NaN
    R[[60, 61], 2] .= NaN
    plain = (; decay = 2.0^(-1 / 20), min_obs = 5, regime_method = nothing,
             centring = PreCentred(), hac_lags = 2)
    @test isapprox(var(RegimeAdjustedExpWeightedVariance(; plain...), R),
                   diag(cov(RegimeAdjustedExpWeightedCovariance(; plain...), R));
                   rtol = 1e-12)
end
