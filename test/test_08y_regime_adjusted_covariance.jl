#=
`RegimeAdjustedExpWeightedCovariance` answers the verbs its Choice Surface promises.

The type entered the library with a docstring that stated the mathematics and no code behind it.
Issue #637 held that gap: `cov` and `cor` reached the covariance surface's own fallback, which
reads `cor`, which reads `cov` back through `StatsBase`'s generic method, so a caller met a
`StackOverflowError` rather than a matrix.

The recursion is a port of the reference implementation, so the oracle of this file is the
reference itself. `ORACLE_X` and the five matrices below were measured by fitting the reference
on that fixture, one matrix per configuration, and pasted here as literals.

`oracle_estimator` names three keywords of the reference: it clamps the multiplier to
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

const PO = PortfolioOptimisers

# ---------------------------------------------------------------- the reference oracle

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

@testset "the port reproduces the reference implementation" begin
    for (ce, oracle) in ((oracle_estimator(), ORACLE_DEFAULTS),
                         # The oracle scores the raw statistic (#1415, #1428).
                         (oracle_estimator(; regime_target = PO.MahalanobisTarget(),
                                           regime_method = PO.LogRegimeAdjusted()), ORACLE_MAHALANOBIS_LOG),
                         (oracle_estimator(; regime_target = PO.DiagonalTarget(),
                                           regime_method = PO.RootMeanSquaredAdjusted(), hac_lags = 2),
                          ORACLE_DIAGONAL_RMS_HAC),
                         (oracle_estimator(; cor_decay = 0.97), ORACLE_SEPARATE_CORR),
                         # The reference's estimated location, `ZeroStartCentring()` since #1507.
                         (oracle_estimator(; centring = ZeroStartCentring()), ORACLE_UNCENTRED))
        # The two implementations run the same operations in a different language, so they
        # agree to round-off rather than bit for bit.
        @test isapprox(cov(ce, ORACLE_X), oracle; rtol = 1e-12)
        # The correlation is the same matrix rescaled, so it needs no oracle of its own.
        @test isapprox(cor(ce, ORACLE_X), PO.regime_adjusted_correlation(oracle);
                       rtol = 1e-12)
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

@testset "the two masks blank what they should" begin
    rng = StableRNG(8903)
    X = randn(rng, 40, 4) .* 0.02
    ce = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 3, regime_min_obs = 2)

    # An asset that never lists is NaN in its own row and column, and nowhere else.
    active = trues(size(X))
    active[:, 4] .= false
    sigma = cov(ce, X; active_mask = active)
    @test all(isnan, sigma[4, :])
    @test all(isnan, sigma[:, 4])
    @test all(isfinite, sigma[1:3, 1:3])
    @test all(isnan, cor(ce, X; active_mask = active)[4, :])

    # An asset below the warm-up is blanked the same way.
    short = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 200,
                                                regime_min_obs = 2)
    @test all(isnan, cov(short, X))

    # A whole column of NaN returns is the same case as an asset that never lists.
    Xn = copy(X)
    Xn[:, 2] .= NaN
    @test all(isnan, cov(ce, Xn)[2, :])
    @test all(isfinite, cov(ce, Xn)[[1, 3, 4], [1, 3, 4]])

    # The estimation mask moves the regime multiplier and leaves the recursion alone.
    est = trues(size(X))
    est[:, 1] .= false
    masked = cov(ce, X; estimation_mask = est)
    @test !isapprox(masked, cov(ce, X))
    # Both are the same matrix up to the scalar multiplier the mask changes.
    ratio = masked ./ cov(ce, X)
    @test isapprox(ratio, fill(ratio[1, 1], size(ratio)))
end

@testset "the regime adjustment turns off, and it clamps" begin
    rng = StableRNG(8904)
    X = randn(rng, 60, 3) .* 0.02
    X[45:end, :] .*= 5.0

    plain = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 3,
                                                regime_method = nothing)
    # No regime state advances, so the answer is the plain exponentially weighted recursion,
    # which is what an unreachable `regime_min_obs` also gives.
    unreachable = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 3,
                                                      regime_min_obs = 10_000)
    @test isequal(cov(plain, X), cov(unreachable, X))

    # The clamp bites on a loud sample, and it is the only difference between the two.
    unclamped = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 3,
                                                    regime_min_obs = 2,
                                                    regime_lohi_mult = nothing)
    clamped = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 3,
                                                  regime_min_obs = 2,
                                                  regime_lohi_mult = (0.7, 1.6))
    factor = sqrt(cov(unclamped, X)[1, 1] / cov(plain, X)[1, 1])
    expected = cov(plain, X) .* clamp(factor, 0.7, 1.6)^2
    @test isapprox(cov(clamped, X), expected; rtol = 1e-12)
    # The correlation does not read the multiplier at all.
    @test isapprox(cor(clamped, X), cor(unclamped, X))
end

@testset "the portfolio target reads its weights" begin
    rng = StableRNG(8905)
    X = randn(rng, 40, 3) .* 0.02
    X[30:end, :] .*= 3.0
    base = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 3,
                                               regime_min_obs = 2)

    one_row = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 3,
                                                  regime_min_obs = 2,
                                                  regime_target = PO.PortfolioTarget(;
                                                                                     w = [0.2,
                                                                                          0.3,
                                                                                          0.5]))
    two_rows = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 3,
                                                   regime_min_obs = 2,
                                                   regime_target = PO.PortfolioTarget(;
                                                                                      w = [0.2 0.3 0.5;
                                                                                           0.5 0.3 0.2]))
    # Named weights answer, and they are not the inverse-volatility direction.
    @test all(isfinite, cov(one_row, X))
    @test !isapprox(cov(one_row, X), cov(base, X))
    @test !isapprox(cov(two_rows, X), cov(one_row, X))

    # A row scaled by a constant names the same direction, because the rows are normalised.
    scaled = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 3,
                                                 regime_min_obs = 2,
                                                 regime_target = PO.PortfolioTarget(;
                                                                                    w = [2.0,
                                                                                         3.0,
                                                                                         5.0]))
    @test isapprox(cov(scaled, X), cov(one_row, X))

    # The weights meet the universe at the fit, so the fit is what refuses a wrong shape.
    wrong = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 3,
                                                regime_min_obs = 2,
                                                regime_target = PO.PortfolioTarget(;
                                                                                   w = [0.5,
                                                                                        0.5]))
    @test_throws DimensionMismatch cov(wrong, X)
    negative = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 3,
                                                   regime_min_obs = 2,
                                                   regime_target = PO.PortfolioTarget(;
                                                                                      w = [-0.5,
                                                                                           0.5,
                                                                                           1.0]))
    @test_throws DomainError cov(negative, X)

    # Each target states how many active assets its statistic needs.
    @test PO.min_active_assets(PO.MahalanobisTarget()) == 2
    @test PO.min_active_assets(PO.DiagonalTarget()) == 1
    @test PO.min_active_assets(PO.PortfolioTarget()) == 1
end

@testset "the estimator refuses what it cannot fit" begin
    rng = StableRNG(8906)
    X = randn(rng, 20, 3) .* 0.02
    ce = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 3, regime_min_obs = 2)

    # The guard is the family's, not this verb's.
    @test_throws DomainError cov(ce, X; dims = 3)
    @test_throws DomainError cov(ce, X; dims = 0)
    @test_throws DomainError cor(ce, X; dims = 3)

    # A mask that does not cover the sample is refused rather than recycled.
    @test_throws DimensionMismatch cov(ce, X; active_mask = trues(19, 3))
    @test_throws DimensionMismatch cov(ce, X; estimation_mask = trues(20, 2))

    # A state fitted on another universe is refused.
    other = partial_fit!(RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 3,
                                                             regime_min_obs = 2),
                         randn(rng, 20, 2) .* 0.02)
    @test_throws DimensionMismatch partial_fit!(other, X)

    # The constructor's own domain.
    @test_throws DomainError RegimeAdjustedExpWeightedCovariance(; decay = 0.0)
    @test_throws DomainError RegimeAdjustedExpWeightedCovariance(; cor_decay = -0.5)
    @test_throws DomainError RegimeAdjustedExpWeightedCovariance(; hac_lags = 0)
    @test_throws DomainError RegimeAdjustedExpWeightedCovariance(; min_obs = 0)
    @test_throws DomainError RegimeAdjustedExpWeightedCovariance(; regime_min_obs = 0)
    @test_throws DomainError RegimeAdjustedExpWeightedCovariance(;
                                                                 regime_lohi_mult = (1.6,
                                                                                     0.7))
    @test_throws DomainError RegimeAdjustedExpWeightedCovariance(;
                                                                 regime_lohi_mult = (0.0,
                                                                                     1.6))

    # A `cor_decay` equal to `decay` states the same recursion in two places, so it takes the
    # single covariance path and answers what `nothing` answers.
    same = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, cor_decay = 0.9, min_obs = 3,
                                               regime_min_obs = 2)
    @test !PO.has_separate_cor_decay(same)
    @test isequal(cov(same, X), cov(ce, X))
    @test PO.has_separate_cor_decay(RegimeAdjustedExpWeightedCovariance(; decay = 0.9,
                                                                        cor_decay = 0.95))
end

#=
The branches a well-behaved sample never reaches.

Every line below is a refusal, a reset or a shape the ordinary fits above walk past: a block that
does not factorise, a direction that keeps no weight, a variance that never leaves zero, an asset
that delists while the correlation runs at its own decay, and the two mask paths that only a
`dims = 2` call or a one-observation fold takes. They are the whole of this file's uncovered
lines, and each one is driven here on purpose.
=#
@testset "the refusals and the resets" begin
    # ------------------------------------------------ the guarded factorisation

    # A singular block does not factorise plainly, and the first ridge repairs it.
    singular = [1.0 1.0; 1.0 1.0]
    @test isnothing(PO.safe_regime_cholesky(singular, 1e-12)) == false
    @test LinearAlgebra.issuccess(PO.safe_regime_cholesky(singular, 1e-12))

    # A zero diagonal sends the scale to the largest absolute entry, and an indefinite block
    # survives no ridge, so the helper refuses rather than throws.
    indefinite = [0.0 1.0; 1.0 0.0]
    @test isnothing(PO.safe_regime_cholesky(indefinite, 1e-12))

    # The Mahalanobis statistic passes that refusal on rather than raising.
    @test isnothing(PO.regime_statistic(PO.MahalanobisTarget(), [0.1, -0.2], indefinite,
                                        [1, 2], 1e-12))

    # A portfolio row that keeps no weight over the contributing assets is dropped, and a target
    # whose every row is dropped refuses.
    C = [4.0e-4 1.0e-4; 1.0e-4 9.0e-4]
    empty_rows = PO.PortfolioTarget(; w = [0.0 0.0 1.0])
    @test isnothing(PO.regime_statistic(empty_rows, [0.01, -0.02], C, [1, 2], 1e-12))

    # ------------------------------------------------ the separate correlation path

    rng = StableRNG(8907)
    sep(; kwargs...) = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, cor_decay = 0.95,
                                                           min_obs = 2, regime_min_obs = 2,
                                                           kwargs...)

    # A sample of zeros never lifts a variance off zero, so the correlation rebuild finds no
    # active block and the covariance stays at its seed.
    @test all(iszero, cov(sep(), zeros(6, 3)))

    # A sample below the numerical floor lifts the variance off zero and leaves the correlation
    # state's diagonal at it, so the rebuild stops one step later than the case above. The
    # read-out then reports the variance alone, with no correlation to carry off the diagonal.
    # The returns are taken as deviations from zero: an estimated location would remove the
    # constant sample whole.
    tiny = cov(sep(; centring = PreCentred()), fill(1.0e-9, 6, 3))
    @test all(>(0), LinearAlgebra.diag(tiny))
    @test all(iszero, tiny - LinearAlgebra.Diagonal(tiny))

    # A sample of nothing but `NaN` leaves every asset uncounted, so the read-out has no active
    # block at all and blanks the whole matrix.
    @test all(isnan, cov(sep(), fill(NaN, 6, 3)))

    # An asset that delists mid-sample has its variance and its correlation state reset, which
    # is the branch the single-recursion path does not carry.
    X = randn(rng, 30, 3) .* 0.02
    active = trues(size(X))
    active[20:end, 3] .= false
    delisted = cov(sep(), X; active_mask = active)
    @test all(isnan, delisted[3, :])
    @test all(isfinite, delisted[1:2, 1:2])

    # ------------------------------------------------ the regime statistic refuses in a fit

    # The estimation mask keeps one asset, and the weights put every unit on the other, so no
    # row of the direction survives and no observation advances the regime state.
    est = trues(size(X))
    est[:, 1] .= false
    est[:, 3] .= false
    zeroed = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2,
                                                 regime_min_obs = 2,
                                                 regime_target = PO.PortfolioTarget(;
                                                                                    w = [1.0,
                                                                                         0.0,
                                                                                         0.0]))
    inert = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2,
                                                regime_min_obs = 10_000,
                                                regime_target = PO.PortfolioTarget(;
                                                                                   w = [1.0,
                                                                                        0.0,
                                                                                        0.0]))
    @test isequal(cov(zeroed, X; estimation_mask = est),
                  cov(inert, X; estimation_mask = est))

    # ------------------------------------------------ the two mask paths

    ce = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2, regime_min_obs = 2)

    # A mask along `dims = 2` is sliced by column, which is the orientation the row path skips.
    @test isequal(cov(ce, permutedims(X); dims = 2, active_mask = permutedims(active),
                      estimation_mask = permutedims(est)),
                  cov(ce, X; active_mask = active, estimation_mask = est))

    # A one-observation fold carries its own masks, one vector at a time.
    folded = foldl((c, i) -> partial_fit!(c, view(X, i, :);
                                          active_mask = view(active, i, :),
                                          estimation_mask = view(est, i, :)), axes(X, 1);
                   init = ce)
    @test isequal(cov(folded), cov(ce, X; active_mask = active, estimation_mask = est))
end

@testset "the pass without a callback runs the same recursion" begin
    rng = StableRNG(8877)
    X = randn(rng, 60, 3) .* 0.02
    X[45:end, :] .*= 4.0

    # Issue #877. The callback of `regime_adjusted_covariance_pass!` is what makes
    # `variance_series` one forward pass. `cov` and `partial_fit!` want the last cache alone, so
    # they read the method that takes no callback, and the two caches read the same covariance.
    for ce in (oracle_estimator(), oracle_estimator(; hac_lags = 2),
               oracle_estimator(; regime_method = PO.LogRegimeAdjusted()))
        with_f = PO.regime_adjusted_covariance_pass!((args...) -> nothing, ce, X, 1,
                                                     nothing, nothing)
        without_f = PO.regime_adjusted_covariance_pass!(ce, X, 1, nothing, nothing)
        @test isequal(PO.regime_adjusted_covariance(with_f, ce),
                      PO.regime_adjusted_covariance(without_f, ce))
    end
end

@testset "a holiday holds every entry of its asset on the path with one decay" begin
    # Issue #1420 and ADR 0181 (amendment of 2026-09-29). A holiday carries no information about
    # the entries of its asset, so the state and the weight of each pair that contains it hold.
    # Two equal assets and five holidays of the second, while the first returns zero: the pair
    # holds its covariance and the variance of asset 1 falls, so the entries imply a
    # correlation above one. The report restores the nearest positive semidefinite
    # correlation, which is one, and keeps the variances.
    r = [0.02, -0.01, 0.015, -0.02, 0.01, 0.03, -0.025, 0.02]
    Xh = vcat(hcat(r, r), [zeros(5) fill(NaN, 5)])
    ce = RegimeAdjustedExpWeightedCovariance(; decay = 0.7, min_obs = 1,
                                             centring = PreCentred())
    st = partial_fit!(ce, Xh).cache
    short = partial_fit!(RegimeAdjustedExpWeightedCovariance(; decay = 0.7, min_obs = 1,
                                                             centring = PreCentred()),
                         Xh[1:8, :]).cache
    @test st.covariance[:, 2] == short.covariance[:, 2]
    @test st.weight[:, 2] == short.weight[:, 2]
    P = st.covariance ./ st.weight
    @test P[1, 2] / sqrt(P[1, 1] * P[2, 2]) > 2
    sh = cov(ce, Xh)
    @test isapprox(sh[1, 2] / sqrt(sh[1, 1] * sh[2, 2]), 1; rtol = 1e-12)
    @test minimum(eigvals(Symmetric(sh))) > -1e-14 * maximum(abs, sh)

    # The report is positive semidefinite for every pattern of holidays.
    rng = StableRNG(1343)
    worst = Inf
    for _ in 1:100
        T, N = rand(rng, 10:40), rand(rng, 2:5)
        X = randn(rng, T, N) / 100
        X[rand(rng, T, N) .< 0.3] .= NaN
        S = cov(RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 1), X)
        f = findall(isfinite, LinearAlgebra.diag(S))
        length(f) < 2 && continue
        m = maximum(abs, S[f, f])
        iszero(m) || (worst = min(worst, minimum(eigvals(Symmetric(S[f, f]))) / m))
    end
    @test worst > -1e-14
end

@testset "a holiday holds every entry of its asset on the separate path" begin
    # Issue #1420 and ADR 0181 (amendment of 2026-09-29). The separate `cor_decay` path takes
    # the same step on its correlation state and its weight. Two equal assets and five holidays
    # of the second, at which the first has a zero deviation: every entry of the second asset
    # holds, and the report restores a correlation of one.
    r = [0.02, -0.01, 0.015, -0.02, 0.01, 0.03, -0.025, 0.02]
    Xh = vcat(hcat(r, r), [zeros(5) fill(NaN, 5)])
    mk() = RegimeAdjustedExpWeightedCovariance(; decay = 0.7, cor_decay = 0.9, min_obs = 1,
                                               centring = PreCentred())
    ce = mk()
    @test PO.has_separate_cor_decay(ce)
    st = partial_fit!(ce, Xh).cache
    short = partial_fit!(mk(), Xh[1:8, :]).cache
    @test st.cor_state[:, 2] == short.cor_state[:, 2]
    @test st.cor_weight[:, 2] == short.cor_weight[:, 2]
    @test isnothing(st.weight)
    @test isapprox(cor(ce, Xh)[1, 2], 1; rtol = 1e-12)

    # The report is positive semidefinite for every pattern of holidays. Before #1346 these
    # panels gave a smallest eigenvalue of -0.1075 times the largest entry.
    rng = StableRNG(7)
    worst_cov = Inf
    for _ in 1:200
        T, N = rand(rng, 20:60), rand(rng, 3:5)
        X = randn(rng, T, N) / 100
        X[rand(rng, T, N) .< 0.3] .= NaN
        ce = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, cor_decay = 0.97,
                                                 min_obs = 1)
        s = cov(ce, X)
        if all(isfinite, s)
            worst_cov = min(worst_cov, minimum(eigvals(Symmetric(s))) / maximum(abs, s))
        end
    end
    @test worst_cov > -1e-14
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
constants `√n` and `ψ(x n) + ln y` do not, so `DiagonalTarget` divides the sum by the factor of its
law, on the estimated spectrum shrunk until its dispersion is unbiased (ADR 0190).
=#
@testset "the diagonal statistic divides by the factor of its law" begin
    SF = PO.SpecialFunctions
    FM, LG = PO.FirstMomentRegimeAdjusted(), PO.LogRegimeAdjusted()
    chi_root(n) = sqrt(2) * exp(SF.loggamma((n + 1) / 2) - SF.loggamma(n / 2))
    # The law is known at independent squares, a χ²(n) variate, and at one eigenvalue n, n times
    # a χ²(1) variate. The Log factor is one at μ = 1 whatever its parameters, because it takes each
    # square as the Gamma(x, y) variate that its constant assumes. Measured: 1e-13 or less.
    for n in (1, 2, 12, 200)
        spike = [n; zeros(n - 1)]
        @test isapprox(PO.regime_law_factor(FM, ones(n)), chi_root(n)^2 / n; rtol = 1e-12)
        @test isapprox(PO.regime_law_factor(FM, spike), 2 / pi; rtol = 1e-12)
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
    @test isapprox(PO.regime_law_factor(FM, [b], ones(1, 1), fill(v, 1, 1)), 2b / pi;
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
Issue #1431. The squared Mahalanobis distance of a correctly calibrated return is `χ²_n R`, with
`R = 1 / S` and `S` the Schur complement of one direction in the weighted Wishart `W`. Each regime
method reads its own moment of `R`: `E[R]` for the root mean square, `E[√R]²` for the first moment
and `exp(E[ln R])` for the log. The fixed point `b` of #1415 is a deterministic equivalent of the
first moment alone, 0.55 % high at 12 assets and a half-life of 10, and dividing every method by
it gave squared multipliers of 0.994, 0.975 and 0.956. `mahalanobis_level_bias` computes each
moment by a recursion over the `n - 1` other directions, which is exact at one asset and at equal
weights, and `mahalanobis_regime_bias!` interpolates it over the count of observations
(ADR 0190).
=#
@testset "the Mahalanobis statistic divides by the bias its method reads" begin
    methods = (PO.RootMeanSquaredAdjusted(), PO.FirstMomentRegimeAdjusted(),
               PO.LogRegimeAdjusted())
    # At equal weights `R = K / χ²(K - n + 1)`, the law of `inverse_wishart_bias` at `ν = K`. A
    # half-life of 1e6 makes the 30 weights equal to within 2e-5. Measured at 12 assets: 2.4e-8,
    # 1.7e-8 and 1.1e-8.
    for n in (2, 12), m in methods
        @test isapprox(PO.mahalanobis_level_bias(m, 2.0^(-1 / 1e6), [30], n)[1],
                       PO.inverse_wishart_bias(m, 30 / (30 - n - 1), n); rtol = 1e-6)
    end
    # At one asset the recursion takes no step, and the factor is the exact table of the scalar
    # estimator.
    lam = 2.0^(-1 / 10)
    for m in methods
        @test isapprox(PO.mahalanobis_level_bias(m, lam, [200], 1)[1],
                       PO.regime_bias_table(m, lam, 200)[200]; rtol = 1e-9)
    end
    # A Monte Carlo of one million draws at 12 assets, a half-life of 10 and 400 observations gave
    # 1.6138, 1.5827 and 1.5524, each with a relative standard error of 3e-4. The recursion gives
    # 1.6132, 1.5821 and 1.5519, below the truth by the sign its one approximation predicts. The
    # moments are ordered by Jensen's inequality, and the fixed point is 0.55 % above the mean.
    f = [PO.mahalanobis_level_bias(m, lam, [400], 12)[1] for m in methods]
    for (fi, mc) in zip(f, (1.6138, 1.5827, 1.5524))
        @test isapprox(fi, mc; rtol = 1.5e-3)
    end
    @test f[1] > f[2] > f[3]
    @test PO.mahalanobis_bias(lam, 400, 12) > 1.004 * f[1]

    # The interpolation over the count of observations reproduces the recursion between its
    # nodes, and it reads the node exactly at the gate. A count past saturation reads the last node.
    for (m, K) in zip(methods, (23, 61, 150))
        store = PO.regime_bias_store(PO.MahalanobisTarget(), lam, Float64).nodes
        @test isapprox(PO.mahalanobis_regime_bias!(store, m, lam, K, 12),
                       PO.mahalanobis_level_bias(m, lam, [K], 12)[1]; rtol = 1e-7)
        @test collect(keys(store)) == [12]
        @test PO.mahalanobis_regime_bias!(store, m, lam, 16, 12) ≈
              PO.mahalanobis_level_bias(m, lam, [16], 12)[1]
        Ksat = PO.mahalanobis_bias_saturation(lam, 12)
        @test PO.mahalanobis_regime_bias!(store, m, lam, 10 * Ksat, 12) ==
              PO.mahalanobis_regime_bias!(store, m, lam, Ksat, 12)
    end

    # On iid Normal returns the squared multiplier is the mean of the transformed statistic,
    # which is one when the statistic is correct. Eight seeds gave 1.0000 ± 0.0020, 0.9999 ±
    # 0.0011 and 0.9997 ± 0.0010, where the fixed point gave 0.994, 0.975 and 0.956.
    rng = StableRNG(1431)
    na, nr = 12, 12000
    A = randn(rng, na, na)
    U = cholesky(Symmetric(A * transpose(A) / na + Diagonal(rand(rng, na)))).U
    R = randn(rng, nr, na) * U .* 0.01
    base = (; decay = lam, min_obs = 5, regime_lohi_mult = nothing,
            regime_decay = 2.0^(-1 / 500), regime_min_obs = 1, centring = PreCentred())
    function multiplier(method)
        on = RegimeAdjustedExpWeightedCovariance(; base...,
                                                 regime_target = PO.MahalanobisTarget(),
                                                 regime_method = method)
        off = RegimeAdjustedExpWeightedCovariance(; base..., regime_method = nothing)
        return mean(cov(on, R) ./ cov(off, R))
    end
    for m in methods
        @test 0.98 < multiplier(m) < 1.02
    end

    # The state keeps one set of nodes for each count of assets it meets, and a copy keeps its
    # own. Without the correction the state keeps no store.
    fit(; extra...) = partial_fit!(RegimeAdjustedExpWeightedCovariance(; base..., extra...,
                                                                       regime_target = PO.MahalanobisTarget()),
                                   R[1:60, :]).cache
    st = fit()
    @test st.bias.nodes isa AbstractDict && collect(keys(st.bias.nodes)) == [na]
    @test copy(st).bias !== st.bias && copy(st).bias == st.bias
    @test isnothing(fit(; debias = RawStatistic()).bias)
end

#=
Issue #1437. On the separate correlation path the block is `D R̂ D`, with the volatilities from
the variance at `decay` and the correlation at `cor_decay`. The Mahalanobis factor at `cor_decay`
holds the noise of the diagonal at `cor_decay`, so the noisier variance at `decay` left the
statistic 3 % too large, and 9 % at two HAC lags. At `R = I` and equal weights the correlation is
independent of the variances, so the diagonal enters through `E[1/Q]` alone, and
`variance_noise_bias!` multiplies the factor by the ratio of that moment at the two decays. The
maintainer ruled that the one factor serves the three methods, and that the next-order parts go to
a child issue of map #1375; #1439 then added the spread of each method (the testset below).
=#
@testset "the separate path divides by the noise of its variance" begin
    lam, lamc = 2.0^(-1 / 10), 2.0^(-1 / 20)
    RMS = PO.RootMeanSquaredAdjusted()
    sep = RegimeAdjustedExpWeightedCovariance(; decay = lam, cor_decay = lamc,
                                              regime_target = PO.MahalanobisTarget(),
                                              regime_method = RMS)
    # For the mean the factor is the ratio of the two exact tables, plain or HAC, and one on one
    # decay. Under HAC the table at `cor_decay` reads the kernel of the damped rows (#1445).
    for (ce, lags) in ((sep, nothing),
                       (RegimeAdjustedExpWeightedCovariance(; decay = lam, cor_decay = lamc, hac_lags = 2,
                                                            regime_target = PO.MahalanobisTarget(),
                                                            regime_method = RMS), 2))
        store = PO.regime_bias_store(PO.MahalanobisTarget(), lamc, Float64)
        for K in (5, 40, 3000)
            tv = PO.regime_bias!(Float64[], RMS, lam, K, lags)
            tc = PO.regime_bias!(Float64[], RMS, lamc, K,
                                 PO.correlation_hac_lags!(store, ce))
            @test PO.variance_noise_bias!(store, ce, K, 12) ≈ tv / tc rtol = 1e-14
        end
    end
    store = PO.regime_bias_store(PO.MahalanobisTarget(), lamc, Float64)
    one_decay = RegimeAdjustedExpWeightedCovariance(; decay = lam,
                                                    regime_target = PO.MahalanobisTarget())
    @test PO.variance_noise_bias!(store, one_decay, 40, 12) == 1 && isempty(store.kappa)
    # At 12 assets, a half-life of 10 and a correlation half-life of 20 the steady-state factor
    # is 1.0347, and 1.0694 at two lags.
    @test PO.variance_noise_bias!(store, sep, 10^5, 12) ≈ 1.0347 atol = 1e-4

    # On iid Normal returns the squared multiplier is one when the statistic is correct.
    # Measured on this seed: 1.0036, 1.0023 and 0.9994, and 1.0036, 1.0004 and 0.9957 before the
    # per-method term of #1439. Eight seeds gave 0.9975 ± 0.0013, 0.9970 ± 0.0007 and
    # 0.9964 ± 0.0012, from 0.9975, 0.9952 and 0.9927 with one factor for the three methods and
    # 1.0321, 1.0297 and 1.0272 with the factor at `cor_decay` alone; the true factor of the
    # block, measured without return noise over 64 seeds, is within 0.9 % of the corrected one at
    # correlations of 0, of a random factor model and of 0.8 (#1447).
    rng = StableRNG(1437)
    na, nr = 12, 12000
    A = randn(rng, na, na)
    U = cholesky(Symmetric(A * transpose(A) / na + Diagonal(rand(rng, na)))).U
    R = randn(rng, nr, na) * U .* 0.01
    base = (; decay = lam, cor_decay = lamc, min_obs = 5, regime_lohi_mult = nothing,
            regime_decay = 2.0^(-1 / 500), regime_min_obs = 1, centring = PreCentred())
    function multiplier(method; extra...)
        on = RegimeAdjustedExpWeightedCovariance(; base..., extra...,
                                                 regime_target = PO.MahalanobisTarget(),
                                                 regime_method = method)
        off = RegimeAdjustedExpWeightedCovariance(; base..., extra...,
                                                  regime_method = nothing)
        return mean(cov(on, R) ./ cov(off, R))
    end
    mults = map(multiplier, (RMS, PO.FirstMomentRegimeAdjusted(), PO.LogRegimeAdjusted()))
    @test all(m -> 0.98 < m < 1.02, mults)
    # The per-method term of #1439 brings Log to 0.0042 of RMS on this seed, from 0.0079.
    @test mults[1] - mults[3] < 0.006
    # At two lags, measured on this seed: 1.0111, and 1.0337 where each row divides by the
    # volatility after its update (`hac_vol_before = VolatilityAfterUpdate()`, #1448). Eight seeds gave
    # 1.0009 ± 0.0040, 1.0006 ± 0.0040 and 1.0001 ± 0.0039 for the three methods, where the
    # Bartlett kernel at `cor_decay` gave 0.9659 ± 0.0050 for RMS (#1445), and 1.0260 ± 0.0059 with
    # the rule after the update, where the factor at `cor_decay` alone gave 1.0898. Before #1438 the
    # factor at `cor_decay` was the mean's fixed point, 1.8 % too large, and the eight seeds gave
    # 1.0184.
    @test 0.99 < multiplier(RMS; hac_lags = 2, min_obs = 17) < 1.03

    # The state of the separate path fills the table, and a copy keeps its own.
    st = partial_fit!(RegimeAdjustedExpWeightedCovariance(; base...,
                                                          regime_target = PO.MahalanobisTarget()),
                      R[1:60, :]).cache
    @test !isempty(st.bias.kappa) && copy(st).bias.kappa !== st.bias.kappa
end

#=
Issue #1439. The noisier variance at `decay` also spreads the statistic, which lowers the root and
the log that `FirstMomentRegimeAdjusted` and `LogRegimeAdjusted` read. At one asset the block is the
variance alone, so the exact factor is the method's own table at `decay`. The factor of the
separate path multiplies `κ` by the ratio of the method's moment to the mean's at the two decays,
to the power `3 / (n + 2)`: exact at one asset, of the second order on a sphere of `n` independent
terms, and one for the mean. The maintainer ruled on #1439 that this term lands at its strength at
`R = I`, and that the mean at a correlated `R` and under HAC go to child issues of map #1375.
=#
@testset "the separate path divides each method by the spread of its variance" begin
    lam, lamc = 2.0^(-1 / 10), 2.0^(-1 / 20)
    RMS, FM, LG = PO.RootMeanSquaredAdjusted(), PO.FirstMomentRegimeAdjusted(),
                  PO.LogRegimeAdjusted()
    sep(m, lags) = RegimeAdjustedExpWeightedCovariance(; decay = lam, cor_decay = lamc,
                                                       hac_lags = lags, regime_method = m,
                                                       regime_target = PO.MahalanobisTarget())
    function table(m, d, lags)
        return if isnothing(lags)
            PO.regime_bias_table(m, d, 3000)
        else
            PO.regime_bias_table(m, d, 3000, lags)
        end
    end
    for lags in (nothing, 2)
        # Under HAC the tables at `cor_decay` read the kernel of the damped rows (#1445).
        kc = isnothing(lags) ? nothing : PO.hac_row_kernel(lam, lags)
        tv, tc = table(RMS, lam, lags), table(RMS, lamc, kc)
        for m in (RMS, FM, LG)
            mv, mc = table(m, lam, lags), table(m, lamc, kc)
            for K in (5, 40, 3000), n in (1, 12)
                store = PO.regime_bias_store(PO.MahalanobisTarget(), lamc, Float64)
                spread = (mv[K] / tv[K]) / (mc[K] / tc[K])
                @test PO.variance_noise_bias!(store, sep(m, lags), K, n) ≈
                      tv[K] / tc[K] * spread^(3 / (n + 2)) rtol = 1e-14
            end
        end
        # The mean does not read the count of assets.
        store = PO.regime_bias_store(PO.MahalanobisTarget(), lamc, Float64)
        @test PO.variance_noise_bias!(store, sep(RMS, lags), 40, 1) ==
              PO.variance_noise_bias!(store, sep(RMS, lags), 40, 12)
    end

    # At one asset the factor at `cor_decay` times this factor is the method's table at `decay`.
    for m in (RMS, FM, LG)
        store = PO.regime_bias_store(PO.MahalanobisTarget(), lamc, Float64)
        b = PO.mahalanobis_regime_bias!(store.nodes, m, lamc, 40, 1, nothing)
        @test b * PO.variance_noise_bias!(store, sep(m, nothing), 40, 1) ≈
              PO.regime_bias_table(m, lam, 40)[40] rtol = 1e-9
    end

    # At 12 assets, a half-life of 10 and a correlation half-life of 20 the power of the ratio is
    # 0.9981 and 0.9963 in the steady state, and 0.9960 and 0.9921 at two lags on the kernel of the
    # damped rows (0.9964 and 0.9928 on the Bartlett kernel before #1445). Against the true
    # factor of the block, measured without return noise over 64 seeds at `R = I`, the three
    # methods then read 1.0018, 1.0018 and 1.0017, from 1.0018, 0.9999 and 0.9980.
    # The table of the ratio belongs to the method of the state, so each method takes a store.
    steady(m, lags) = PO.variance_noise_bias!(PO.regime_bias_store(PO.MahalanobisTarget(),
                                                                   lamc, Float64),
                                              sep(m, lags), 10^5, 12)
    for (lags, fm, lg) in ((nothing, 0.9981, 0.9963), (2, 0.9960, 0.9921))
        @test steady(FM, lags) / steady(RMS, lags) ≈ fm atol = 1e-4
        @test steady(LG, lags) / steady(RMS, lags) ≈ lg atol = 1e-4
    end

    # A `Float32` state keeps its type.
    ce32 = RegimeAdjustedExpWeightedCovariance(; decay = 2.0f0^(-1.0f0 / 10),
                                               cor_decay = 2.0f0^(-1.0f0 / 20),
                                               regime_target = PO.MahalanobisTarget())
    store32 = PO.regime_bias_store(PO.MahalanobisTarget(), ce32.cor_decay, Float32)
    @test PO.variance_noise_bias!(store32, ce32, 40, 12) isa Float32
end

#=
Issue #1438. A HAC estimate is `Z'AZ` with the banded weight matrix `A` of #1433, so the level
recursion of #1431 holds on the spectrum of `A`, from `D₀(s) = ln det(I + sA)`. `A` has negative
eigenvalues, so the transform is cut at the maximum `s*` of `D₀`, as the table of one direction is.
The recursion reads `D₀` continued past the peak of its saturated count `σ ρ₀(σ)` as the transform
of the saturated weights, and the last transform reads the true `D₀` up to `s*`. Before this issue
the HAC path divided every method by the mean's fixed point, 1.8 %, 5.3 % and 8.9 % above the
three factors at 12 assets, a half-life of 10 and two lags. The first 32 counts after the gate take
the recursion itself, and the interpolation serves the rest (ADR 0190).
=#
@testset "under HAC the Mahalanobis statistic divides by the bias its method reads" begin
    methods = (PO.RootMeanSquaredAdjusted(), PO.FirstMomentRegimeAdjusted(),
               PO.LogRegimeAdjusted())
    lam = 2.0^(-1 / 10)
    function hac_B(lam, L, K)
        B = diagm(lam .^ (0:(K - 1)))
        for i in 1:L, j in 0:(K - i - 1)
            B[j + 1, j + i + 1] = B[j + i + 1, j + 1] = lam^j * (1 - i / (L + 1))
        end
        return B
    end
    # At no lag the banded path is the plain one: the logs of the pivots against `log1p`
    # (measured 2e-14).
    for m in methods
        @test isapprox(PO.mahalanobis_level_bias(m, lam, [16, 400], 12, 0),
                       PO.mahalanobis_level_bias(m, lam, [16, 400], 12); rtol = 1e-12)
    end
    # At one asset the factor is the table of one direction on the same cut, and at five
    # observations `A` is positive definite and nothing is cut (measured 7e-12).
    for m in methods, (d, K, L) in ((lam, 200, 2), (2.0^(-1 / 40), 200, 5), (lam, 5, 1))
        @test isapprox(PO.mahalanobis_level_bias(m, d, [K], 1, L)[1],
                       PO.regime_bias_table(m, d, K, L)[K]; rtol = 1e-10)
    end

    # The peak: the slope of the log-determinant is zero there, and its value is the log-determinant
    # of the eigenvalues.
    K = 300
    d5 = 2.0^(-1 / 5)
    grid = PO.mahalanobis_bias_grid(d5)
    G, Rh, peaks = PO.mahalanobis_lattice(d5, [K], grid.xf, 5)
    sigma, gs = peaks[1]
    c = (1 - d5) / (1 - d5^K)
    mu = eigvals(Symmetric(hac_B(d5, 5, K)))
    @test abs(PO.hac_log_det_slopes(d5, K, sigma / c, 5)[1] / c) <
          1e-9 * sum(abs.(mu) ./ abs.(1 .+ sigma .* mu))
    @test isapprox(gs, sum(log1p.(sigma .* mu)); rtol = 1e-10)
    # The continuation keeps the true lattice up to the cut for the last transform, and the lattice
    # of the recursion falls as `N / σ` past the peak of the saturated count.
    gh, rh = G[:, 1], Rh[:, 1]
    cut = PO.mahalanobis_cut!(gh, rh, grid.xf, 1.0, peaks[1])
    p = searchsortedlast(grid.xf, log(sigma))
    @test cut.sstar == sigma && cut.gt[1:p] == G[1:p, 1] && all(==(gs), cut.gt[(p + 1):end])
    sat = exp.(grid.xf) .* rh
    pm = argmax(sat[1:p])
    @test all(>(0), rh) && all(x -> x ≈ sat[pm], sat[pm:end])
    @test isnothing(PO.mahalanobis_cut!(gh, rh, grid.xf, 1.0, nothing))

    # A Monte Carlo of four million draws of the Schur complement of the HAC estimate itself, at 12
    # assets, a half-life of 10, two lags and 520 observations, gave 2.4777, 2.3958 and 2.3165,
    # each with a standard error of 5e-4. The recursion is 1.2e-3, 8e-4 and 3e-4 below, and the
    # mean's fixed point 1.8 % above the first.
    f = [PO.mahalanobis_level_bias(m, lam, [520], 12, 2)[1] for m in methods]
    for (fi, mc) in zip(f, (2.4777, 2.3958, 2.3165))
        @test isapprox(fi, mc; rtol = 2e-3)
    end
    @test f[1] > f[2] > f[3]
    @test PO.mahalanobis_bias(lam, 520, 12, 2) > 1.015 * f[1]
    # Near the gate, at five assets, a half-life of 5, one lag and 12 observations, where 4e-4 of
    # draws are not positive definite, the Monte Carlo gave 2.9710 and 2.5881; the recursion is
    # 0.25 % and 0.03 % below. A hard cut gave infinite factors here, and at 390 observations and a
    # half-life of 40.
    for (m, mc) in zip(methods[2:3], (2.9710, 2.5881))
        @test isapprox(PO.mahalanobis_level_bias(m, d5, [12], 5, 1)[1], mc; rtol = 5e-3)
    end
    @test all(isfinite,
              PO.mahalanobis_level_bias(methods[2], 2.0^(-1 / 40), [380, 390], 12, 1))

    # The nodes start where the gate opens, and the first 32 counts take the recursion itself.
    @test PO.mahalanobis_bias_start(lam, 12, 2) == 54
    @test PO.mahalanobis_bias_start(lam, 12, nothing) == 16
    for (m, d, n, L) in ((methods[1], 2.0^(-1 / 40), 5, 1), (methods[2], lam, 12, 2),
                         (methods[3], 2.0^(-1 / 250), 2, 2))
        store = PO.regime_bias_store(PO.MahalanobisTarget(), d, Float64).nodes
        K1 = PO.mahalanobis_bias_start(d, n, L)
        Ks = [K1, K1 + 3, K1 + 31, K1 + 32, K1 + 45, 3 * K1 + 100]
        @test isapprox([PO.mahalanobis_regime_bias!(store, m, d, K, n, L) for K in Ks],
                       PO.mahalanobis_level_bias(m, d, Ks, n, L); rtol = 1e-5)
        @test store[n].start == K1 && length(store[n].exact) == 32
    end
    plain = PO.regime_bias_store(PO.MahalanobisTarget(), lam, Float64).nodes
    PO.mahalanobis_regime_bias!(plain, methods[1], lam, 30, 12)
    @test isempty(plain[12].exact) && plain[12].start == 16

    # On iid Normal returns the squared multiplier is one when the statistic is correct. Eight
    # seeds gave 1.0035 ± 0.0039, 1.0031 ± 0.0038 and 1.0023 ± 0.0040, where the mean's fixed point
    # gave 0.984, 0.952 and 0.920.
    rng = StableRNG(1438)
    na, nr = 12, 12000
    A = randn(rng, na, na)
    U = cholesky(Symmetric(A * transpose(A) / na + Diagonal(rand(rng, na)))).U
    R = randn(rng, nr, na) * U .* 0.01
    base = (; decay = lam, min_obs = 5, regime_lohi_mult = nothing, hac_lags = 2,
            regime_decay = 2.0^(-1 / 500), regime_min_obs = 1, centring = PreCentred())
    function multiplier(method)
        on = RegimeAdjustedExpWeightedCovariance(; base...,
                                                 regime_target = PO.MahalanobisTarget(),
                                                 regime_method = method)
        off = RegimeAdjustedExpWeightedCovariance(; base..., regime_method = nothing)
        return mean(cov(on, R) ./ cov(off, R))
    end
    for m in methods
        @test 0.98 < multiplier(m) < 1.02
    end
end

#=
Issue #1448, ruled on #1444. Under HAC the diagonal of a row, `x_t² + 2 Σ w_l x_t x_{t-l}`, can be
negative. A volatility that holds the row damps a positive row and amplifies a negative one, so
the diagonal of the correlation state is skewed down and falls on the guards of `update_var_cor!`
and `pair_weighted_correlation`. So under HAC the separate path divides each row by the volatility
before its update, and `hac_vol_before = VolatilityAfterUpdate()` keeps the rule after it. Without HAC the diagonal
is `x_t² ≥ 0`, and the rule after the update stays (#1440).
=#
@testset "under HAC the separate path divides each row by the volatility before its update" begin
    # The recursion by hand on a two-asset case of six rows and one lag.
    function by_hand(X, lam, lamc, before)
        v, Q, prev = zeros(2), zeros(2, 2), nothing
        for t in axes(X, 1)
            x = X[t, :]
            A = x * transpose(x)
            if !isnothing(prev)
                A += (x * transpose(prev) + prev * transpose(x)) / 2
            end
            s = before ? v : lam * v + (1 - lam) * diag(A)
            v = lam * v + (1 - lam) * diag(A)
            is = [si > 1e-12 ? 1 / sqrt(si) : 0.0 for si in s]
            Q = lamc * Q + (1 - lamc) * A .* (is * transpose(is))
            prev = x
        end
        return v, Q
    end
    X = randn(StableRNG(1448), 6, 2) .* [0.02 0.01]
    rule(b; extra...) = RegimeAdjustedExpWeightedCovariance(; decay = 0.8, cor_decay = 0.9,
                                                            hac_lags = 1,
                                                            hac_vol_before = if b
                                                                VolatilityBeforeUpdate()
                                                            else
                                                                VolatilityAfterUpdate()
                                                            end, regime_method = nothing,
                                                            centring = PreCentred(),
                                                            min_obs = 1, extra...)
    states = map((true, false)) do b
        return PO.regime_adjusted_covariance_pass!(rule(b), X, 1, nothing, nothing)
    end
    for (b, st) in zip((true, false), states)
        v, Q = by_hand(X, 0.8, 0.9, b)
        @test st.variance ≈ v rtol = 1e-14
        @test st.cor_state ≈ Q rtol = 1e-14
    end
    # The two rules differ, and the default is the rule before the update.
    @test !isapprox(states[1].cor_state, states[2].cor_state; rtol = 1e-2)
    @test isa(RegimeAdjustedExpWeightedCovariance().hac_vol_before, VolatilityBeforeUpdate)

    # Without HAC, or on the path with one decay, the keyword changes nothing.
    Y = randn(StableRNG(1449), 200, 3) .* 0.01
    for extra in ((; hac_lags = nothing), (; cor_decay = nothing))
        @test isequal(cov(rule(true; extra...), Y), cov(rule(false; extra...), Y))
    end

    # At eight lags, on iid returns at a correlation of 0.3 and half-lives of 10 and 20, the rule
    # after the update holds the covariance, or clamps the correlation, on 529 to 772 of 20 000
    # rows over three seeds (1.1 % to 3.0 % over 16 seeds of 10⁶ rows on #1444). The rule before
    # the update does neither.
    function guard_rows(b, X)
        ce = RegimeAdjustedExpWeightedCovariance(; decay = 2.0^(-1 / 10),
                                                 cor_decay = 2.0^(-1 / 20), hac_lags = 8,
                                                 hac_vol_before = if b
                                                     VolatilityBeforeUpdate()
                                                 else
                                                     VolatilityAfterUpdate()
                                                 end, regime_method = nothing,
                                                 centring = PreCentred())
        rows = 0
        PO.regime_adjusted_covariance_pass!(ce, X, 1, nothing, nothing) do i, cache
            if i > 500
                Q = PO.pair_weighted_block(cache.cor_state, cache.cor_weight, 1:2)
                rows += any(<=(ce.min_val), diag(cache.cor_state)) ||
                        abs(Q[1, 2]) > sqrt(Q[1, 1] * Q[2, 2])
            end
            return nothing
        end
        return rows
    end
    Z = randn(StableRNG(1448), 20_000, 2) * cholesky([1 0.3; 0.3 1]).U
    @test guard_rows(true, Z) == 0
    @test guard_rows(false, Z) > 100
end

#=
Issue #1445. Under the rule before the update, row `p` divides `x_{p-l}` by `σ_{p-1}`, and
`V_{p-1}` already holds `x_{p-l}²` and the HAC products of `x_{p-l}`. So each lagged term of the
row is damped, and correlated with the returns near it, while its mean product with `x_p` stays
zero: the correlation carries less noise than the Bartlett kernel of `cor_decay` assumes, and the
factor over-corrected by 0.8 % to 9.5 % at `R = I`. Regressed on the returns divided by their own
volatility within `2L` of its row, each damped term gives a raw HAC kernel of width `2L`, which the
tables at `cor_decay` read (ADR 0190).
=#
@testset "under HAC the separate path reads the kernel of its damped rows" begin
    lam, lamc = 2.0^(-1 / 10), 2.0^(-1 / 20)
    RMS, FM, LG = PO.RootMeanSquaredAdjusted(), PO.FirstMomentRegimeAdjusted(),
                  PO.LogRegimeAdjusted()
    # A count of lags names the Bartlett weights and a vector is its own weights, so the tables,
    # the slopes and the cross sum agree bit for bit.
    k2 = PO.hac_lag_weights(2, 1.0)
    @test k2 == [1 - 1 / 3, 1 - 2 / 3] && isempty(PO.hac_lag_weights(nothing, 1.0))
    @test PO.hac_lag_weights(k2, 1.0) === k2
    for m in (RMS, FM, LG)
        @test PO.regime_bias_table(m, lamc, 300, 2) ==
              PO.regime_bias_table(m, lamc, 300, k2)
    end
    @test PO.hac_log_det_slopes(lam, 200, 3.0, 2) ==
          PO.hac_log_det_slopes(lam, 200, 3.0, k2)
    @test PO.exp_weight_cross_sum(lam, lamc, 40, 2) ==
          PO.exp_weight_cross_sum(lam, lamc, 40, PO.hac_lag_weights(2, lam * lamc))

    # The kernel against its definition: on a chain of 2 × 10⁶ rows, regress each damped term
    # `x_{p-l} / σ_{p-1}` on `y_{p-d} = x_{p-d} / σ_{p-d-1}`, and add the residual in quadrature.
    # Measured 4e-4 at a half-life of 5, where the damping moves the kernel by 0.1 from Bartlett.
    function chain_kernel(lam, L, T, rng)
        w = [1 - l / (L + 1) for l in 1:L]
        xs, Vh = zeros(2L), ones(2L + 1)
        G, B, iv = zeros(L), zeros(L, 2L), 0.0
        for t in 1:T
            x = randn(rng)
            if t > 500
                for l in 1:L
                    yt = xs[l] / sqrt(Vh[1])
                    G[l] += yt^2
                    B[l, :] .+= yt .* xs ./ sqrt.(Vh[2:end])
                end
                iv += 1 / Vh[1]
            end
            h = x^2 + 2 * sum(w[l] * x * xs[l] for l in 1:L)
            Vh[2:end] .= Vh[1:(end - 1)]
            Vh[1] = lam * Vh[1] + (1 - lam) * h
            xs[2:end] .= xs[1:(end - 1)]
            xs[1] = x
        end
        G ./= iv
        B ./= iv
        k = vec(sum(w .* B; dims = 1))
        for l in 1:L
            k[l] = sign(k[l]) * sqrt(k[l]^2 + w[l]^2 * max(G[l] - sum(abs2, B[l, :]), 0))
        end
        return k
    end
    d5 = 2.0^(-1 / 5)
    k5 = PO.hac_row_kernel(d5, 2)
    @test isapprox(k5, chain_kernel(d5, 2, 2_000_000, StableRNG(1445)); atol = 2e-3)
    @test k5[1] < 0.6 && k5[2] < 0.3 && k5[3] < 0
    # At a half-life of 10 and two lags the kernel is 0.613, 0.290, −0.022 and −0.003.
    @test isapprox(PO.hac_row_kernel(lam, 2), [0.6131, 0.2898, -0.0223, -0.0030];
                   atol = 1e-4)
    @test PO.hac_row_kernel(2.0f0^(-1.0f0 / 10), 2) isa Vector{Float32}

    # The tables at `cor_decay` read the kernel on the separate path under HAC with the rule before
    # the update, once for each state, and the Bartlett weights elsewhere.
    sep(b; extra...) = RegimeAdjustedExpWeightedCovariance(; decay = lam, cor_decay = lamc,
                                                           hac_lags = 2,
                                                           hac_vol_before = if b
                                                               VolatilityBeforeUpdate()
                                                           else
                                                               VolatilityAfterUpdate()
                                                           end, regime_method = RMS,
                                                           regime_target = PO.MahalanobisTarget(),
                                                           centring = PreCentred(),
                                                           min_obs = 1, extra...)
    store = PO.regime_bias_store(PO.MahalanobisTarget(), lamc, Float64)
    kern = PO.correlation_hac_lags!(store, sep(true))
    @test kern == PO.hac_row_kernel(lam, 2) &&
          PO.correlation_hac_lags!(store, sep(true)) === kern
    for ce in (sep(false), sep(true; hac_lags = nothing), sep(true; cor_decay = nothing))
        @test PO.correlation_hac_lags!(PO.regime_bias_store(PO.MahalanobisTarget(), lamc,
                                                            Float64), ce) === ce.hac_lags
    end

    # Against the true factor of the block, measured without return noise at 12 assets and
    # `R = I`: 1.0085 ± 0.0015 over 8 seeds of 20 000 rows (1.0076 over 32 seeds of 50 000), where
    # the Bartlett kernel of the rule after the update read 0.9717 (0.9709). Over 4 to 24 assets,
    # half-lives of 5 to 20, correlation half-lives of 20 and 40 and one to four lags the kernel
    # reads 0.2 % to 2.2 % high at `R = I`, and −2.6 % to +3.4 % at every correlation, from 0.8 % to
    # 10.2 % low. Four seeds keep the test short; the gap is more than ten standard errors.
    function block_truth(ce, n, seeds, rows; burn = 1000)
        vals = map(seeds) do s
            X = randn(StableRNG(s), burn + rows + 1, n)
            acc, cnt = 0.0, 0
            PO.regime_adjusted_covariance_pass!(ce, X, 1, nothing, nothing) do i, cache
                if burn < i
                    C = PO.regime_covariance_block(cache, ce, 1:n)
                    acc += tr(inv(cholesky(Symmetric(C)))) / n
                    cnt += 1
                end
                return nothing
            end
            return acc / cnt
        end
        return mean(vals)
    end
    function factor(ce, n)
        st = PO.regime_bias_state(ce, Float64)
        return PO.mahalanobis_regime_bias!(st.nodes, RMS, lamc, 10^5, n,
                                           PO.correlation_hac_lags!(st, ce)) *
               PO.variance_noise_bias!(st, ce, 10^5, n)
    end
    truth = block_truth(sep(true), 12, 1445:1448, 12_000)
    @test 0.995 < truth / factor(sep(true), 12) < 1.025
    @test truth / factor(sep(false), 12) < 0.985
end

#=
Issue #1461, found by #1445. The Diagonal target shrinks its estimated correlation towards the
identity until the dispersion of its eigenvalues is unbiased, which subtracts the noise of each
`r̂` from `Σ r̂²`. Under the rule before the update the lagged terms of each row are damped, so the
noise is less than the Bartlett kernel says, and the shrink subtracted too much. The shrink reads
the kernel of `hac_row_kernel`, as the Mahalanobis tables do (ADR 0190).
=#
@testset "under HAC the diagonal shrink reads the kernel of its damped rows" begin
    lam, lamc = 2.0^(-1 / 10), 2.0^(-1 / 20)
    kern = PO.hac_row_kernel(lam, 2)
    # The sum of a pair's squared weights reads any weights of its lags, and it is the trace of the
    # square of its banded weight matrix, `exp_weight_cross_sum` at one decay.
    for K in (3, 17, 300), hac in (1, 2, 5, kern)
        @test isapprox(PO.pair_weight_square_sum(lamc, 1 - lamc^K, hac),
                       PO.exp_weight_cross_sum(lamc, lamc, K, hac); rtol = 1e-12)
    end
    @test PO.pair_weight_square_sum(lamc, 0.7, 2) ===
          PO.pair_weight_square_sum(lamc, 0.7, PO.hac_lag_weights(2, 1.0))
    @test PO.pair_weight_square_sum(0.9f0, 0.7f0, 2) isa Float32

    # The state of a Diagonal target makes the kernel on the separate path under HAC with the
    # rule before the update, and keeps the Bartlett weights elsewhere.
    diag_ce(b; extra...) = RegimeAdjustedExpWeightedCovariance(; decay = lam,
                                                               cor_decay = lamc,
                                                               hac_lags = 2,
                                                               hac_vol_before = if b
                                                                   VolatilityBeforeUpdate()
                                                               else
                                                                   VolatilityAfterUpdate()
                                                               end,
                                                               regime_method = PO.FirstMomentRegimeAdjusted(),
                                                               regime_target = PO.DiagonalTarget(),
                                                               centring = PreCentred(),
                                                               min_obs = 1, extra...)
    X = randn(StableRNG(1461), 300, 4) *
        [1.0 0.5 0.3 0.0; 0.0 1.0 0.4 0.2; 0.0 0.0 1.0 0.6; 0.0 0.0 0.0 1.0]
    for ce in (diag_ce(false), diag_ce(true; cor_decay = nothing))
        @test isempty(partial_fit!(ce, X).cache.bias.kernel)
    end
    ce = diag_ce(true)
    cache = partial_fit!(ce, X).cache
    @test cache.bias.kernel == kern
    # The shrunk spectrum has the dispersion that the kernel leaves, more than the Bartlett one.
    idx = 1:4
    C = PO.regime_covariance_block(cache, ce, idx)
    rho = C ./ sqrt.(diag(C) * transpose(diag(C)))
    q(hac) = sum(rho[i, j]^2 -
                 (1 - rho[i, j]^2)^2 *
                 PO.pair_weight_square_sum(lamc, cache.cor_weight[i, j], hac)
                 for i in idx, j in idx if i != j)
    mu = eigvals(Symmetric(PO.diagonal_law_correlation(cache, ce, C, idx)))
    @test isapprox(sum(abs2, mu .- 1), q(kern); rtol = 1e-10) && q(kern) > q(2)

    # At `R = I` the mean of `r̂²` is the noise alone. Over 4 seeds of 100 000 rows it reads
    # 0.0320 ± 0.0003, which is 0.960 of the kernel's model and 0.874 of the Bartlett one (16
    # seeds of 50 000 rows: 0.964 and 0.876).
    r2 = mean(1461:1464) do s
        Z = randn(StableRNG(s), 101_000, 2)
        ce2 = diag_ce(true; regime_method = nothing)
        acc, cnt = 0.0, 0
        PO.regime_adjusted_covariance_pass!(ce2, Z, 1, nothing, nothing) do i, cache
            if i > 1000
                B = PO.regime_covariance_block(cache, ce2, 1:2)
                acc += B[1, 2]^2 / (B[1, 1] * B[2, 2])
                cnt += 1
            end
            return nothing
        end
        return acc / cnt
    end
    @test 0.93 < r2 / PO.pair_weight_square_sum(lamc, 1.0, kern) < 1
    @test r2 / PO.pair_weight_square_sum(lamc, 1.0, 2) < 0.9
end

@testset "under the estimated location the tables read the law of the shared deviations" begin
    # The estimate as the variance pass forms it: after K + 1 returns it is (1 - λ) xᵀ M x, and
    # the location is gᵀx, both read off the pass by polarisation on unit returns.
    function estimated_form(lam, L, K)
        ce = RegimeAdjustedExpWeightedVariance(; decay = lam, hac_lags = L,
                                               regime_method = nothing, min_obs = 1)
        state(x) = PO.regime_adjusted_variance_pass!(ce, reshape(x, :, 1), 1, nothing,
                                                     nothing)
        E = Matrix(1.0I, K + 1, K + 1)
        R(x) = state(x).variance[1] / (1 - lam)
        r = [R(E[:, i]) for i in 1:(K + 1)]
        M = [i == j ? r[i] : (R(E[:, i] + E[:, j]) - r[i] - r[j]) / 2
             for i in 1:(K + 1), j in 1:(K + 1)]
        g = [state(E[:, i]).location[1] for i in 1:(K + 1)]
        return M, g
    end
    # Each factor from the eigenvalues on a grid of step 1/200, with the tilted variance
    # V = 1 + gᵀ(I + sM)⁻¹g of the next deviation: the root mean square and the first moment
    # read it, and the log reads the law of the estimate alone.
    function dense_factors(lam, L, K)
        M, g = estimated_form(lam, L, K)
        F = eigen(Symmetric(M))
        mu = F.values
        gq = F.vectors' * g
        s = exp.(range(-75, 60; step = 1 / 200))
        lG = map(x -> (v = 1 .+ x .* mu; any(<=(0), v) ? -Inf : -sum(log, v) / 2), s)
        G = PO.hac_laplace!(similar(s), lG)
        V = [max(1 + sum(gq .^ 2 ./ (1 .+ x .* mu)), 0) for x in s]
        c = (1 - lam) / (1 - lam^K)
        f = PO.centring_factor(EstimatedCentring(), lam, K + 1)
        return (PO.regime_bias_factor(PO.RootMeanSquaredAdjusted(), s, G .* V ./ f, c,
                                      1 / 200),
                PO.regime_bias_factor(PO.FirstMomentRegimeAdjusted(), s, G .* sqrt.(V ./ f),
                                      c, 1 / 200),
                PO.regime_bias_factor(PO.LogRegimeAdjusted(), s, G, c, 1 / 200),
                PO.regime_log_variance(s, G, 1 / 200),
                PO.regime_bias_factor(PO.RootMeanSquaredAdjusted(), s, G, c, 1 / 200),
                PO.regime_bias_factor(PO.FirstMomentRegimeAdjusted(), s, G, c, 1 / 200)),
               minimum(mu) > 0
    end
    lam = 2.0^(-1 / 10)
    methods = (PO.RootMeanSquaredAdjusted(), PO.FirstMomentRegimeAdjusted(),
               PO.LogRegimeAdjusted())
    for L in (nothing, 2)
        tables = [PO.regime_bias_table(m, lam, 64, L, EstimatedCentring()) for m in methods]
        logm = PO.regime_bias_table(PO.RegimeTermMoments(PO.LogRegimeAdjusted()), lam, 64,
                                    L, EstimatedCentring())
        # The law of the estimate alone: the deviation taken as independent of it.
        laws = [PO.regime_bias_table(m, lam, 64, L, EstimatedCentring(), LawDebias())
                for m in methods[1:2]]
        for K in (5, 20, 40)
            d, definite = dense_factors(lam, L, K)
            # Measured: 7e-13 where the estimate is positive definite, and 1.3e-9 where it
            # is not, where the cut moves with the grid.
            rtol = definite ? 1e-11 : 1e-8
            for i in 1:3
                @test isapprox(tables[i][K], d[i]; rtol)
            end
            @test isapprox(logm[K][3], d[4]; rtol)
            @test logm[K][1] == tables[1][K]
            @test isapprox(laws[1][K], d[5]; rtol)
            @test isapprox(laws[2][K], d[6]; rtol)
        end
    end
    # The pre-centred table over the exact one at the steady state, a half-life of 10, and one,
    # two and four lags. A Monte Carlo of the estimator (64 000 paths) reads 1.0024, 1.0058 and
    # 1.0140 for the root mean square.
    for (L, ratio) in ((1, 1.00256), (2, 1.00583), (4, 1.01388))
        @test isapprox(PO.regime_bias_table(PO.RootMeanSquaredAdjusted(), lam, 400, L)[400] /
                       PO.regime_bias_table(PO.RootMeanSquaredAdjusted(), lam, 400, L,
                                            EstimatedCentring())[400], ratio; atol = 1e-5)
    end
    # Past λ^K = √ε the table holds its ratio to the pre-centred one: at a count of 400 that is
    # within 1e-9 of the recursion run to that count.
    lat = PO.estimated_bias_lattice(lam, 2)
    for k in 1:400
        PO.estimated_bias_count!(lat, lam, k)
        k == 400 &&
            @test isapprox(PO.regime_bias_table(PO.RootMeanSquaredAdjusted(), lam, 400, 2,
                                                EstimatedCentring())[400],
                           PO.estimated_bias_factor(PO.RootMeanSquaredAdjusted(),
                                                    PO.estimated_bias_integrals!(lat, lam,
                                                                                 k),
                                                    (1 - lam) / (1 - lam^k)); rtol = 1e-9)
        k < 400 && PO.estimated_bias_integrals!(lat, lam, k)
    end
    # The number type of the decay carries through, and a pre-centred or zero-start estimate
    # reads the table without a centring.
    @test eltype(PO.regime_bias_table(PO.FirstMomentRegimeAdjusted(), Float32(lam), 64, 2,
                                      EstimatedCentring())) === Float32
    @test PO.regime_bias_table(PO.LogRegimeAdjusted(), lam, 64, 2, PreCentred()) ==
          PO.regime_bias_table(PO.LogRegimeAdjusted(), lam, 64, 2)
    @test PO.regime_bias_table(PO.LogRegimeAdjusted(), lam, 64, nothing,
                               ZeroStartCentring()) ==
          PO.regime_bias_table(PO.LogRegimeAdjusted(), lam, 64)
    # The state of an estimator under the estimated location holds the exact table.
    @test PO.regime_bias!(Float64[], PO.RootMeanSquaredAdjusted(), lam, 30, 2,
                          EstimatedCentring()) ==
          PO.regime_bias_table(PO.RootMeanSquaredAdjusted(), lam, 64, 2,
                               EstimatedCentring())[30]
    ce = RegimeAdjustedExpWeightedVariance(; decay = lam, hac_lags = 2, min_obs = 1)
    st = PO.regime_adjusted_variance_pass!(ce, randn(StableRNG(1548), 40, 2), 1, nothing,
                                           nothing)
    @test st.bias == PO.regime_bias_table(ce.regime_method, lam, length(st.bias), 2,
                                          EstimatedCentring())

    # The Mahalanobis columns against the eigenvalues of the form: g = ln det(I + σM), its
    # derivative, and σ gᵀ(I + σM)⁻¹g, below the peak.
    K = 60
    M, g = estimated_form(lam, 2, K)
    F = eigen(Symmetric(M))
    mu = F.values
    gq = F.vectors' * g
    x0 = log(eps()) + log1p(-lam)
    lat = PO.estimated_bias_lattice(lam, 2, x0, 51.0, ComplexF64)
    j0 = ceil(Int, (x0 - (-61 + log1p(-lam))) / lat.h)
    xf = range(x0 - j0 * lat.h, 50 + 8 * lat.h; step = lat.h)
    cols = (; g = zeros(length(xf)), r = zeros(length(xf)), vt = zeros(length(xf)),
            rt = zeros(length(xf)))
    for k in 1:K
        PO.estimated_bias_count!(lat, lam, k)
        PO.estimated_bias_freeze!(lat)
    end
    PO.estimated_mahalanobis_columns!(cols, lat, xf, j0, lam, K)
    for j in (j0 - 50, j0 + 5, j0 + 300, j0 + 560)
        sig = exp(xf[j])
        @test isapprox(cols.g[j], sum(log1p.(sig .* mu)); rtol = 1e-11, atol = 1e-14)
        @test isapprox(cols.r[j], sum(mu ./ (1 .+ sig .* mu)); rtol = 1e-11)
        @test isapprox(cols.vt[j], sig * sum(gq .^ 2 ./ (1 .+ sig .* mu)); rtol = 1e-11)
    end
    # The chain at one point gives the same slope and value.
    ch = PO.estimated_log_det_chain(lam, K, exp(xf[j0 + 300]), lat.k)
    @test ch[1]
    @test isapprox(ch[2], cols.r[j0 + 300]; rtol = 1e-11)
    @test isapprox(ch[3], cols.g[j0 + 300]; rtol = 1e-11)
    # Below the seed the chain reads the mean of the estimate.
    ch = PO.estimated_log_det_chain(lam, K, 1e-30, lat.k)
    @test ch[1] && ch[2] == (1 - lam^K) / (1 - lam) && ch[3] == 1e-30 * ch[2]
    # At 200 counts, a half-life of 10, two lags and five assets, a Monte Carlo of a million
    # draws of the statistic reads 1.4994, 1.4513 and 1.4046 (standard error 0.08 %): the three
    # factors of the exact rule are within 0.07 %. The law alone reads 0.5 % above, and the
    # pre-centred factors 1.6 %, 1.4 % and 1.3 % above.
    f = [PO.mahalanobis_level_bias(m, lam, [200], 5, 2, EstimatedCentring())[1]
         for m in methods]
    @test isapprox(f, [1.49839, 1.45126, 1.40387]; rtol = 1e-5)
    f = [PO.mahalanobis_level_bias(m, lam, [200], 5, 2, EstimatedCentring(), LawDebias())[1]
         for m in methods]
    @test isapprox(f, [1.50738, 1.45779, 1.40984]; rtol = 1e-5)
    # Without HAC the three factors are within 0.05 % of a Monte Carlo of the statistic.
    f = [PO.mahalanobis_level_bias(m, lam, [200], 5, nothing, EstimatedCentring())[1]
         for m in methods]
    @test isapprox(f, [1.23597, 1.21405, 1.19265]; rtol = 1e-5)
    # Past λ^K = √ε the factor holds its ratio to the pre-centred one.
    Kh = ceil(Int, log(sqrt(eps())) / log(lam))
    f2 = PO.mahalanobis_level_bias(methods[2], lam, [Kh, Kh + 40], 2, 2,
                                   EstimatedCentring())
    p2 = PO.mahalanobis_level_bias(methods[2], lam, [Kh, Kh + 40], 2, 2)
    @test f2[2] / p2[2] ≈ f2[1] / p2[1]
    # A pre-centred estimate has no dependence to read, so the rule changes nothing there.
    @test PO.mahalanobis_level_bias(methods[2], lam, [40, 200], 5, 2, PreCentred(),
                                    LawDebias()) ==
          PO.mahalanobis_level_bias(methods[2], lam, [40, 200], 5, 2)
    @test PO.debiases(LawDebias()) &&
          PO.reads_dependence(ExactDebias()) &&
          !PO.reads_dependence(LawDebias()) &&
          !PO.reads_dependence(RawStatistic())
    # The state of an estimator holds the table of its rule.
    ce = RegimeAdjustedExpWeightedVariance(; decay = lam, hac_lags = 2, min_obs = 1,
                                           debias = LawDebias())
    st = PO.regime_adjusted_variance_pass!(ce, randn(StableRNG(1549), 40, 2), 1, nothing,
                                           nothing)
    @test st.bias == PO.regime_bias_table(ce.regime_method, lam, length(st.bias), 2,
                                          EstimatedCentring(), LawDebias())
end
