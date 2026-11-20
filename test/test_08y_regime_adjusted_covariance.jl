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
                                               debias = nothing, kwargs...)
end

@testset "the port reproduces the oracle" begin
    for (c, ce, oracle) in (("defaults", oracle_estimator(), ORACLE_DEFAULTS),
                            # The oracle scores the raw statistic (#1415, #1428).
                            ("mahalanobis log",
                             oracle_estimator(; regime_target = PO.MahalanobisTarget(),
                                              regime_method = PO.LogRegimeAdjusted()), ORACLE_MAHALANOBIS_LOG),
                            ("diagonal rms hac",
                             oracle_estimator(; regime_target = PO.DiagonalTarget(),
                                              regime_method = PO.RootMeanSquaredAdjusted(), hac_lags = 2),
                             ORACLE_DIAGONAL_RMS_HAC),
                            # The oracle divides each product by the raw variance state (#1729).
                            ("separate corr", oracle_estimator(; cor_decay = 0.97, cor_vol = nothing),
                             ORACLE_SEPARATE_CORR),
                            # The oracle's estimated location, `ZeroStartCentring()` since #1507.
                            ("uncentred", oracle_estimator(; centring = ZeroStartCentring()), ORACLE_UNCENTRED))
        # The two implementations run the same operations in a different language and sum in
        # another order, so they agree to round-off rather than bit for bit. Measured on
        # 2026-10-07, cell by cell: the covariance to maxrel 2.5e-15 (`separate corr`; the
        # other four 1.7e-15 or less), the correlation to 2.5e-15. That is about ten ulps
        # after the recursion over 16 rows and the multiplier, so `1e-14` leaves a factor four.
        @test parity_compare(cov(ce, ORACLE_X), oracle; rtol = 1e-14, name = "$c cov").ok
        # The correlation is the same matrix rescaled, so it needs no oracle of its own.
        @test parity_compare(cor(ce, ORACLE_X), PO.regime_adjusted_correlation(oracle);
                             rtol = 1e-14, name = "$c cor").ok
    end

    # The five configurations are distinct answers, so no pair of them is a vacuous pass.
    oracles = (ORACLE_DEFAULTS, ORACLE_MAHALANOBIS_LOG, ORACLE_DIAGONAL_RMS_HAC,
               ORACLE_SEPARATE_CORR, ORACLE_UNCENTRED)
    for i in eachindex(oracles), j in eachindex(oracles)
        if i < j
            @test !isapprox(oracles[i], oracles[j]; rtol = 1e-6)
        end
    end
end

@testset "the verbs answer, and the answer is a covariance" begin
    rng = StableRNG(8901)
    X = randn(rng, 90, 4) .* 0.02
    X[70:end, :] .*= 4.0

    for ce in (RegimeAdjustedExpWeightedCovariance(; decay = 0.94, min_obs = 5,
                                                   regime_min_obs = 3),
               RegimeAdjustedExpWeightedCovariance(; decay = 0.94, min_obs = 5,
                                                   regime_min_obs = 3, hac_lags = 2),
               RegimeAdjustedExpWeightedCovariance(; decay = 0.94, cor_decay = 0.97, min_obs = 5,
                                                   regime_min_obs = 3),
               RegimeAdjustedExpWeightedCovariance(; decay = 0.94, min_obs = 5,
                                                   regime_min_obs = 3,
                                                   centring = EstimatedCentring()),
               RegimeAdjustedExpWeightedCovariance(; decay = 0.94, min_obs = 5,
                                                   regime_min_obs = 3,
                                                   regime_target = PO.MahalanobisTarget()),
               RegimeAdjustedExpWeightedCovariance(; decay = 0.94, min_obs = 5,
                                                   regime_min_obs = 3,
                                                   regime_target = PO.DiagonalTarget()),
               RegimeAdjustedExpWeightedCovariance(; decay = 0.94, min_obs = 5,
                                                   regime_min_obs = 3,
                                                   regime_method = PO.LogRegimeAdjusted()),
               RegimeAdjustedExpWeightedCovariance(; decay = 0.94, min_obs = 5,
                                                   regime_min_obs = 3,
                                                   regime_method = PO.RootMeanSquaredAdjusted()))
        sigma = cov(ce, X)
        rho = cor(ce, X)
        @test size(sigma) == (4, 4)
        @test all(isfinite, sigma)
        @test issymmetric(sigma)
        @test all(>(0), LinearAlgebra.diag(sigma))
        @test isapprox(LinearAlgebra.diag(rho), ones(4))
        @test all(x -> -1 <= x <= 1, rho)
        @test issymmetric(rho)
        # The correlation is the covariance rescaled, and the regime multiplier cancels.
        vol = sqrt.(LinearAlgebra.diag(sigma))
        @test isapprox(rho, sigma ./ (vol * transpose(vol)))
        # The observations may lie along either dimension.
        @test isapprox(cov(ce, permutedims(X); dims = 2), sigma)
        @test isapprox(cor(ce, permutedims(X); dims = 2), rho)
    end
end

@testset "an incremental fit is exact" begin
    rng = StableRNG(8902)
    X = randn(rng, 40, 3) .* 0.02
    X[30:end, :] .*= 3.0

    # The keywords are held rather than the estimator, because the copy test rebuilds the same
    # estimator around a copied state and a rebuild that dropped `hac_lags` would read a buffer
    # its own configuration does not admit.
    for kwargs in ((decay = 0.9, min_obs = 3, regime_min_obs = 2),
                   (decay = 0.9, min_obs = 3, regime_min_obs = 2, hac_lags = 2),
                   (decay = 0.9, cor_decay = 0.95, min_obs = 3, regime_min_obs = 2),
                   (decay = 0.9, cor_decay = 0.95, min_obs = 3, regime_min_obs = 2, hac_lags = 2),
                   (decay = 0.9, min_obs = 3, regime_min_obs = 2, centring = EstimatedCentring()))
        ce = RegimeAdjustedExpWeightedCovariance(; kwargs...)
        batch = cov(ce, X)

        # One block folded onto a cold estimator is the whole-sample fit.
        @test isequal(cov(partial_fit!(ce, X)), batch)
        @test isequal(cor(partial_fit!(ce, X)), cor(ce, X))

        # Two blocks, one after the other, are one call over the whole sample.
        halves = partial_fit!(partial_fit!(ce, X[1:17, :]), X[18:end, :])
        @test isequal(cov(halves), batch)

        # One observation at a time is the same recursion.
        one_at_a_time = foldl((c, i) -> partial_fit!(c, view(X, i, :)), axes(X, 1);
                              init = ce)
        @test isequal(cov(one_at_a_time), batch)

        # A state held by hand answers as the estimator's own does.
        fitted = partial_fit!(ce, X)
        @test isequal(cov(fitted, fitted.cache), cov(fitted))
        @test isequal(cor(fitted, fitted.cache), cor(fitted))

        # A fold on a copy leaves the original alone.
        original = fitted.cache
        copied = copy(original)
        partial_fit!(RegimeAdjustedExpWeightedCovariance(; kwargs..., cache = copied), X)
        @test isequal(cov(fitted, original), batch)

        # The two blocks do not add, and the refusal says so.
        @test_throws ArgumentError PO.merge_states(original, copy(original))
        # The generic refusals run first, as they do for the variance twin: a pair over
        # different numbers of assets is a `DimensionMismatch`, not this family's refusal.
        narrow = partial_fit!(RegimeAdjustedExpWeightedCovariance(; kwargs...), X[:, 1:2])
        @test_throws DimensionMismatch PO.merge_states(original, narrow.cache)
    end

    # An estimator that was given no observation carries no state, so the read is refused.
    @test_throws ArgumentError cov(RegimeAdjustedExpWeightedCovariance())
    @test_throws ArgumentError cor(RegimeAdjustedExpWeightedCovariance())
    @test isnothing(RegimeAdjustedExpWeightedCovariance().cache)
end
