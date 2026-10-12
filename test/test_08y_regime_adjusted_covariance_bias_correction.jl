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
Issue #1415. The squared Mahalanobis distance against an estimated block is too large on average,
because the inverse of an estimate is (Jensen's inequality). While the block had fewer
observations than assets it was singular, the ridge of `safe_regime_cholesky` made the distance
about 1e12, and that one value held the regime state for many half-lives. The estimator keyword
`debias = ExactDebias()` skips each row whose block has `n + 3` observations or fewer, where the variance
of the statistic is not finite, and divides the rest by the bias factor of the regime method
(#1431; the fixed point `mahalanobis_bias` was the factor of the mean alone).
`debias = RawStatistic()` is the oracle's raw statistic, which the first testset pins. The keyword was a
field of the target until #1428 moved it to the estimator.
=#
@testset "the Mahalanobis statistic is divided by the bias of its estimated block" begin
    # At equal weights the fixed point is the exact inverse-Wishart mean K / (K - n - 1). A half-life
    # of 1e6 makes the 30 weights equal to within 3e-5.
    @test isapprox(PO.mahalanobis_bias(2.0^(-1 / 1e6), 30, 5), 30 / 24; rtol = 1e-9)
    # The fixed point has no solution where the mean of the statistic is not finite.
    @test isnothing(PO.mahalanobis_bias(0.9, 6, 5))
    @test !isnothing(PO.mahalanobis_bias(0.9, 7, 5))
    # The factor solves the fixed point on the full sum, and the weights below the machine
    # epsilon, which the loop adds as mass, move it by no more than round-off.
    for (lambda, nobs, nassets) in
        ((2.0^(-1 / 10), 200, 12), (2.0^(-1 / 40), 5000, 100), (0.9, 14, 12))
        b = PO.mahalanobis_bias(lambda, nobs, nassets)
        wts = [(1 - lambda) * lambda^j for j in 0:(nobs - 1)]
        wts ./= sum(wts)
        @test isapprox(1 / b, sum(wts ./ (1 .+ (nassets + 1) .* wts .* b)); rtol = 1e-13)
    end

    # On iid Normal returns, the squared multiplier of `RootMeanSquaredAdjusted` is the mean of
    # the statistic over its dimension, so a correct statistic gives one. Measured on this seed
    # after #1431: 0.957 with `debias = ExactDebias()` and 1.010 on the separate path, where the factor of
    # the mean alone gave 0.951 and 1.009, and 3.8e10 with `debias = RawStatistic()`, from the ridge of
    # the singular rows. The testset of #1431 measures the calibration over 12 000 rows and a
    # regime half-life of 500.
    rng = StableRNG(1415)
    na, nr = 12, 4000
    A = randn(rng, na, na)
    U = cholesky(Symmetric(A * transpose(A) / na + Diagonal(rand(rng, na)))).U
    R = randn(rng, nr, na) * U .* 0.01
    base = (; decay = 2.0^(-1 / 10), min_obs = 5, regime_lohi_mult = nothing,
            regime_decay = 2.0^(-1 / 1000), regime_min_obs = 1, centring = PreCentred())
    function squared_multiplier(target; extra...)
        rms = RegimeAdjustedExpWeightedCovariance(; base..., extra...,
                                                  regime_target = target,
                                                  regime_method = PO.RootMeanSquaredAdjusted())
        off = RegimeAdjustedExpWeightedCovariance(; base..., extra...,
                                                  regime_method = nothing)
        return mean(cov(rms, R) ./ cov(off, R))
    end
    @test 0.9 < squared_multiplier(PO.MahalanobisTarget()) < 1.1
    @test 0.9 < squared_multiplier(PO.MahalanobisTarget(); cor_decay = 2.0^(-1 / 20)) < 1.1
    @test squared_multiplier(PO.MahalanobisTarget(); debias = RawStatistic()) > 1e6

    # The gate skips each row whose block has n + 3 observations or fewer, and does not count it.
    # The raw statistic scores every row after `min_obs`.
    fit(target; extra...) = partial_fit!(RegimeAdjustedExpWeightedCovariance(; base...,
                                                                             extra...,
                                                                             regime_target = target),
                                         R[1:60, :]).cache
    @test fit(PO.MahalanobisTarget()).n_regime_obs == 60 - (na + 4)
    @test fit(PO.MahalanobisTarget(); debias = RawStatistic()).n_regime_obs == 60 - 5
end

#=
Issue #1428. A statistic of one direction reads one estimated variance `v̂ = σ² Q`, with
`Q = Σ w_j z_j²`, and each regime method reads its own moment of `Q`: `E[1/Q]` for the root mean
square, `E[Q^(-1/2)]²` for the first moment, `exp(-E[ln Q])` for the log. `debias = ExactDebias()` divides
by that moment, from the exact table `regime_bias_table`. `DiagonalTarget` divides each term by
`E[1/Q_i]`, which keeps the mean of its sum at `n` at every correlation (ADR 0190).
=#
@testset "a one-direction regime statistic divides by the bias its method reads" begin
    SF = PO.SpecialFunctions
    # At equal weights `Q` is χ²(K)/K, so the three moments have closed forms. A half-life of 1e6
    # makes the 30 weights equal to within 2e-5. Measured: 5.9e-13, 7.5e-13 and 5.4e-13.
    K = 30
    tables = [PO.regime_bias_table(m, 2.0^(-1 / 1e6), K)[K]
              for m in (PO.RootMeanSquaredAdjusted(), PO.FirstMomentRegimeAdjusted(),
                        PO.LogRegimeAdjusted())]
    @test isapprox(tables[1], K / (K - 2); rtol = 1e-11)
    @test isapprox(tables[2], (sqrt(K / 2) * SF.gamma((K - 1) / 2) / SF.gamma(K / 2))^2;
                   rtol = 1e-11)
    @test isapprox(tables[3], exp(log(K / 2) - SF.digamma(K / 2)); rtol = 1e-11)
    # The moment of the mean is the fixed point of #1415 at n = 1 to within its 0.2 %.
    @test isapprox(PO.regime_bias_table(PO.RootMeanSquaredAdjusted(), 2.0^(-1 / 10), 200)[200],
                   PO.mahalanobis_bias(2.0^(-1 / 10), 200, 1); rtol = 2e-3)

    # The table grows to twice the count, and past the count where λ^K is below the machine
    # epsilon it no longer changes, so the last entry serves every larger count.
    bias = Float64[]
    f10 = PO.regime_bias!(bias, PO.RootMeanSquaredAdjusted(), 0.9, 10)
    @test length(bias) == 64 && f10 == bias[10]
    f_far = PO.regime_bias!(bias, PO.RootMeanSquaredAdjusted(), 0.9, 10^6)
    @test length(bias) == ceil(Int, log(eps()) / log(0.9)) && f_far == bias[end]
    @test PO.regime_bias!(nothing, PO.RootMeanSquaredAdjusted(), 0.9, 10) === one(0.9)
    @test !PO.regime_bias_open(ExactDebias(), 1, 0.9, 4, nothing) &&
          PO.regime_bias_open(ExactDebias(), 1, 0.9, 5, nothing) &&
          PO.regime_bias_open(RawStatistic(), 1, 0.9, 1, nothing)

    # On iid Normal returns the squared multiplier is the mean of the transformed statistic,
    # which is one when the statistic is correct. Measured on this seed, half-life 10:
    # scalar variance 1.07 raw and 1.00 debiased (8 seeds: 1.0004 RMS, 0.9994 first moment,
    # 0.9993 log); the fixed-weight portfolio and the diagonal target the same.
    rng = StableRNG(1428)
    na, nr = 12, 12000
    A = randn(rng, na, na)
    U = cholesky(Symmetric(A * transpose(A) / na + Diagonal(rand(rng, na)))).U
    R = randn(rng, nr, na) * U .* 0.01
    base = (; decay = 2.0^(-1 / 10), min_obs = 5, regime_lohi_mult = nothing,
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
    for method in (PO.RootMeanSquaredAdjusted(), PO.FirstMomentRegimeAdjusted())
        @test 0.98 < variance_multiplier(method) < 1.02
        @test 0.98 < covariance_multiplier(fixed, method) < 1.02
        @test variance_multiplier(method; debias = RawStatistic()) > 1.04
        @test covariance_multiplier(fixed, method; debias = RawStatistic()) > 1.04
    end
    # The diagonal target also divides the sum by the factor of its law for the root and the log
    # (#1432). Measured on this seed: 0.996 and 0.992 on both paths; 8 seeds read 0.941 and 0.962
    # before, and 0.995 and 0.989 after.
    methods = (PO.RootMeanSquaredAdjusted(), PO.FirstMomentRegimeAdjusted(),
               PO.LogRegimeAdjusted())
    for method in methods, extra in ((;), (; cor_decay = 2.0^(-1 / 20)))
        @test 0.98 < covariance_multiplier(PO.DiagonalTarget(), method; extra...) < 1.02
    end
    @test covariance_multiplier(PO.DiagonalTarget(), PO.RootMeanSquaredAdjusted();
                                debias = RawStatistic()) > 1.05

    # The gate skips an estimate of four observations or fewer. With `min_obs = 1` the raw
    # statistic scores every row after the first, and the debiased one every row after the fifth.
    short = (; base..., min_obs = 1)
    for target in (PO.DiagonalTarget(), fixed)
        state(debias) = partial_fit!(RegimeAdjustedExpWeightedCovariance(; short...,
                                                                         regime_target = target,
                                                                         debias = if debias
                                                                             ExactDebias()
                                                                         else
                                                                             RawStatistic()
                                                                         end), R[1:40, :]).cache
        @test state(true).n_regime_obs == 40 - 5
        @test state(false).n_regime_obs == 40 - 1
        @test isnothing(state(false).bias)
    end
    vstate(debias) = partial_fit!(RegimeAdjustedExpWeightedVariance(; short...,
                                                                    debias = if debias
                                                                        ExactDebias()
                                                                    else
                                                                        RawStatistic()
                                                                    end), R[1:40, :]).cache
    @test vstate(true).n_regime_obs == 40 - 5
    @test vstate(false).n_regime_obs == 40 - 1

    # The state holds its own table, so a copy grows a table of its own. A regime method of
    # `nothing` holds none.
    st = vstate(true)
    @test !isempty(st.bias) && copy(st).bias !== st.bias && copy(st).bias == st.bias
    @test isnothing(partial_fit!(RegimeAdjustedExpWeightedVariance(; short...,
                                                                   regime_method = nothing),
                                 R[1:40, :]).cache.bias)
    # A direction that keeps no weight over the contributing assets takes no update.
    Xn = R[1:40, 1:3]
    Xn[:, 3] .= NaN
    none = PO.PortfolioTarget(; w = [0.0, 0.0, 1.0])
    @test partial_fit!(RegimeAdjustedExpWeightedCovariance(; short...,
                                                           regime_target = none), Xn).cache.n_regime_obs ==
          0
    # On the separate path the portfolio reads the factor at `cor_decay`, because the
    # correlations carry most of the variance of a direction (8 seeds: 1.004 with it, 0.971 with
    # `decay`, at a correlation half-life of 20).
    sst = partial_fit!(RegimeAdjustedExpWeightedCovariance(; short...,
                                                           regime_target = fixed,
                                                           cor_decay = 2.0^(-1 / 20)),
                       R[1:40, :]).cache
    @test sst.bias[5:40] ==
          PO.regime_bias_table(PO.FirstMomentRegimeAdjusted(), 2.0^(-1 / 20),
                               length(sst.bias))[5:40]
    # An incremental fit reads the same table as a fit over the sample.
    ce = RegimeAdjustedExpWeightedCovariance(; short..., regime_target = fixed)
    @test isequal(cov(partial_fit!(partial_fit!(ce, R[1:200, :]), R[201:400, :])),
                  cov(ce, R[1:400, :]))
end
