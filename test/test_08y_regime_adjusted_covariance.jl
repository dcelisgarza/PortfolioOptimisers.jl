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
of the statistic is not finite, and divides the rest by the bias factor of the regime method
(#1431; the fixed point `mahalanobis_bias` was the factor of the mean alone).
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
    # the statistic over its dimension, so a correct statistic gives one. Measured on this seed
    # after #1431: 0.957 with `debias = true` and 1.010 on the separate path, where the factor of
    # the mean alone gave 0.951 and 1.009, and 3.8e10 with `debias = false`, from the ridge of
    # the singular rows. The testset of #1431 measures the calibration over 12 000 rows and a
    # regime half-life of 500.
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
    @test !PO.regime_bias_open(true, 1, 0.9, 4, nothing) &&
          PO.regime_bias_open(true, 1, 0.9, 5, nothing) &&
          PO.regime_bias_open(false, 1, 0.9, 1, nothing)

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
    # The diagonal target also divides the sum by the factor of its law for the root and the log
    # (#1432). Measured on this seed: 0.996 and 0.992 on both paths; 8 seeds read 0.941 and 0.962
    # before, and 0.995 and 0.989 after.
    methods = (PO.RootMeanSquaredAdjusted(), PO.FirstMomentRegimeAdjusted(),
               PO.LogRegimeAdjusted())
    for method in methods, extra in ((;), (; cor_decay = 2.0^(-1 / 20)))
        @test 0.98 < covariance_multiplier(PO.DiagonalTarget(), method; extra...) < 1.02
    end
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

#=
Issue #1430. The default `PortfolioTarget()` builds its inverse-volatility direction from the
estimate that it divides by, so the direction and the error of the estimate are correlated, and
the factor of a fixed direction leaves 1.058 at 12 assets and a half-life of 10. `debias = true`
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
            regime_decay = 2.0^(-1 / 500), regime_min_obs = 1, centred = true)
    function covariance_multiplier(method; extra...)
        on = RegimeAdjustedExpWeightedCovariance(; base..., extra...,
                                                 regime_method = method)
        off = RegimeAdjustedExpWeightedCovariance(; base..., extra...,
                                                  regime_method = nothing)
        return mean(cov(on, R) ./ cov(off, R))
    end
    @test 0.99 < covariance_multiplier(PO.RootMeanSquaredAdjusted()) < 1.01
    @test 0.97 < covariance_multiplier(PO.FirstMomentRegimeAdjusted()) < 1.03
    @test covariance_multiplier(PO.RootMeanSquaredAdjusted(); debias = false) > 1.08
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
        # Each pair read all 200 rows, so `1/K` is the sum of the squared normalised weights, and
        # the shrunk spectrum keeps the trace and has the unbiased dispersion `Σ_{i≠j} r²`. On the
        # separate path the correlation reads `cor_decay`.
        w = [(1 - lam) * lam^j for j in 0:199]
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
                                                  regime_method = FM, debias = false)
        m = [PO.regime_bias!(cache.bias, PO.RegimeTermMoments(FM), lambda, k)
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
            regime_decay = 2.0^(-1 / 500), regime_min_obs = 1, centred = true,
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
and `hac_floor = true` keeps it (ADR 0190).
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
    first_open(n, d, L) = findfirst(K -> PO.regime_bias_open(true, n, d, K, L), 1:2000)
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
    plain = (; decay = lam, min_obs = 5, regime_method = nothing, centred = true,
             hac_lags = 2)
    @test isapprox(var(RegimeAdjustedExpWeightedVariance(; plain...), R),
                   diag(cov(RegimeAdjustedExpWeightedCovariance(; plain...), R));
                   rtol = 1e-12)
    @test !isapprox(var(RegimeAdjustedExpWeightedVariance(; plain..., hac_floor = true), R),
                    diag(cov(RegimeAdjustedExpWeightedCovariance(; plain...), R));
                    rtol = 1e-3)
    # The separate path floors its variance only under `hac_floor = true` too.
    sep = (; plain..., cor_decay = 2.0^(-1 / 20))
    @test cov(RegimeAdjustedExpWeightedCovariance(; sep...), R) !=
          cov(RegimeAdjustedExpWeightedCovariance(; sep..., hac_floor = true), R)

    # On iid Normal returns the squared multiplier is one when the statistic is correct.
    base = (; decay = lam, min_obs = 5, regime_lohi_mult = nothing, hac_lags = 2,
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
    # Measured on this seed: scalar 1.000 (RMS) and 0.987 (first moment), 1.147 raw, and 0.837
    # with the floor, which the factor then over-corrects; the equal-weight direction 1.013, from
    # 1.162 raw; Mahalanobis 0.988, from 2.49 raw. Over 8 seeds: 1.002, 1.003, 1.015 and 0.981.
    RMS, FM = PO.RootMeanSquaredAdjusted(), PO.FirstMomentRegimeAdjusted()
    for method in (RMS, FM)
        @test 0.97 < variance_multiplier(method) < 1.03
    end
    @test variance_multiplier(RMS; debias = false) > 1.1
    @test variance_multiplier(RMS; hac_floor = true) < 0.9
    @test 0.97 < covariance_multiplier(fixed, RMS) < 1.03
    @test covariance_multiplier(fixed, RMS; debias = false) > 1.1
    @test 0.97 < covariance_multiplier(PO.MahalanobisTarget(), RMS; min_obs = 17) < 1.03
    @test covariance_multiplier(PO.MahalanobisTarget(), RMS; min_obs = 17, debias = false) >
          2
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
        store = PO.regime_bias_store(PO.MahalanobisTarget(), lam, Float64)
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
            regime_decay = 2.0^(-1 / 500), regime_min_obs = 1, centred = true)
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
    @test st.bias isa AbstractDict && collect(keys(st.bias)) == [na]
    @test copy(st).bias !== st.bias && copy(st).bias == st.bias
    @test isnothing(fit(; debias = false).bias)
end
