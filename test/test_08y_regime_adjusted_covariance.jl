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
                                               centred = true, min_val = 1e-12,
                                               debias = false, kwargs...)
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
                         (oracle_estimator(; centred = false), ORACLE_UNCENTRED))
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
                                                   regime_min_obs = 3, centred = false),
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
                   (decay = 0.9, min_obs = 3, regime_min_obs = 2, centred = false))
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
    tiny = cov(sep(), fill(1.0e-9, 6, 3))
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
    ce = RegimeAdjustedExpWeightedCovariance(; decay = 0.7, min_obs = 1, centred = true)
    st = partial_fit!(ce, Xh).cache
    short = partial_fit!(RegimeAdjustedExpWeightedCovariance(; decay = 0.7, min_obs = 1,
                                                             centred = true), Xh[1:8, :]).cache
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
                                               centred = true)
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
`debias = true` skips each row whose block has `n + 3` observations or fewer, where the variance
of the statistic is not finite, and divides the rest by the fixed point `mahalanobis_bias`.
`debias = false` is the oracle's raw statistic, which the first testset pins. The keyword was a
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
    # the statistic over its dimension, so a correct statistic gives one. Measured on this seed:
    # 0.998 with `debias = true`, 1.021 on the separate path, and 3.8e10 with `debias = false`,
    # from the ridge of the singular rows. Six other seeds gave 0.95 to 1.03.
    rng = StableRNG(1415)
    na, nr = 12, 4000
    A = randn(rng, na, na)
    U = cholesky(Symmetric(A * transpose(A) / na + Diagonal(rand(rng, na)))).U
    R = randn(rng, nr, na) * U .* 0.01
    base = (; decay = 2.0^(-1 / 10), min_obs = 5, regime_lohi_mult = nothing,
            regime_decay = 2.0^(-1 / 1000), regime_min_obs = 1, centred = true)
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
    @test squared_multiplier(PO.MahalanobisTarget(); debias = false) > 1e6

    # The gate skips each row whose block has n + 3 observations or fewer, and does not count it.
    # The raw statistic scores every row after `min_obs`.
    fit(target; extra...) = partial_fit!(RegimeAdjustedExpWeightedCovariance(; base...,
                                                                             extra...,
                                                                             regime_target = target),
                                         R[1:60, :]).cache
    @test fit(PO.MahalanobisTarget()).n_regime_obs == 60 - (na + 4)
    @test fit(PO.MahalanobisTarget(); debias = false).n_regime_obs == 60 - 5
end

#=
Issue #1428. A statistic of one direction reads one estimated variance `v̂ = σ² Q`, with
`Q = Σ w_j z_j²`, and each regime method reads its own moment of `Q`: `E[1/Q]` for the root mean
square, `E[Q^(-1/2)]²` for the first moment, `exp(-E[ln Q])` for the log. `debias = true` divides
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
    @test PO.regime_bias_gate(true, 1) == 4 && PO.regime_bias_gate(false, 1) == 0

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
            regime_decay = 2.0^(-1 / 500), regime_min_obs = 1, centred = true)
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
        @test variance_multiplier(method; debias = false) > 1.04
        @test covariance_multiplier(fixed, method; debias = false) > 1.04
    end
    @test 0.98 <
          covariance_multiplier(PO.DiagonalTarget(), PO.RootMeanSquaredAdjusted()) <
          1.02
    @test covariance_multiplier(PO.DiagonalTarget(), PO.RootMeanSquaredAdjusted();
                                debias = false) > 1.05

    # The gate skips an estimate of four observations or fewer. With `min_obs = 1` the raw
    # statistic scores every row after the first, and the debiased one every row after the fifth.
    short = (; base..., min_obs = 1)
    for target in (PO.DiagonalTarget(), fixed)
        state(debias) = partial_fit!(RegimeAdjustedExpWeightedCovariance(; short...,
                                                                         regime_target = target,
                                                                         debias = debias),
                                     R[1:40, :]).cache
        @test state(true).n_regime_obs == 40 - 5
        @test state(false).n_regime_obs == 40 - 1
        @test isnothing(state(false).bias)
    end
    vstate(debias) = partial_fit!(RegimeAdjustedExpWeightedVariance(; short...,
                                                                    debias = debias),
                                  R[1:40, :]).cache
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
