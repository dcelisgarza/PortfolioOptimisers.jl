#=
Check `src/08_Moments/39_CrossSectionalWeights.jl` against the mathematics its docstrings
state, and against the reference implementation the map of issue #643 ports. Issue #681.

THREE FACTS SHAPE THE PROBES.

1. THE WEIGHT PAIRED WITH AN OBSERVATION READS NO RESIDUAL OF THAT OBSERVATION. The refine
   step reads row `t - 1` of the variance series, so row 1 carries no estimate at all and
   row 2 reads a fit on one observation. `The warm-up rows fall back to the cap weights`
   pins both rows. `A later residual moves no earlier weight` pins the direction twice: a
   change to the last residual moves no weight at all, and a change to an early one moves
   every later weight and no earlier one.

2. THE REALISED BLEND MUST EQUAL THE NOMINAL SHRINKAGE. Both components are normalised over
   the eligible set before they are blended, so every eligible observation sums to one and
   the share of the inverse variance component is exactly `lambda`. A blend that normalised
   over different universes would drift away from the number the caller wrote.
   `The realised blend equals the nominal shrinkage` measures both halves.

3. THE STORED WEIGHTS COME FROM THE REFERENCE IMPLEMENTATION. `REFERENCE_W1_A` and
   `REFERENCE_W1_B` were produced by the reference implementation's own winsoriser and
   blend, on the inputs written beside them, and they are the oracle of the port. The plain
   Julia `reference_blended_weights` below re-derives the same quantity a second way, so a
   testset never compares the file against itself.
=#

using Statistics

# The blend written out one observation at a time, from the variance series alone. It shares
# no code with the implementation: the winsorisation, the median cap and the two
# normalisations are written here in full.
function reference_blended_weights(V, W0, mask, lambda, ratio, wins)
    T, N = size(W0)
    IV = fill(NaN, T, N)
    for t in 2:T, i in 1:N
        IV[t, i] = 1 / V[t - 1, i]
    end
    for t in 1:T, i in 1:N
        if !mask[t, i]
            IV[t, i] = NaN
        end
    end
    ready = falses(T)
    for t in 1:T
        u = [IV[t, i] for i in 1:N if W0[t, i] > 0 && !isnan(IV[t, i])]
        if any(isfinite, u)
            qlo, qhi = quantile(u, wins[1]), quantile(u, wins[2])
            for i in 1:N
                IV[t, i] = clamp(IV[t, i], qlo, qhi)
            end
        else
            IV[t, :] .= NaN
        end
        ready[t] = any(isfinite, view(IV, t, :))
    end
    for t in 1:T
        if !ready[t]
            continue
        end
        cap = median([IV[t, i] for i in 1:N if !isnan(IV[t, i])]) * ratio
        IV[t, :] .= min.(view(IV, t, :), cap)
        IV[t, :] ./= sum(x for x in view(IV, t, :) if !isnan(x))
    end
    Wm = W0 ./ sum(W0; dims = 2)
    W1 = zeros(T, N)
    for t in 1:T, i in 1:N
        u = !ready[t] ? Wm[t, i] : (isnan(IV[t, i]) ? 0.0 : IV[t, i])
        W1[t, i] = mask[t, i] ? lambda * u + (1 - lambda) * Wm[t, i] : 0.0
    end
    return W1
end

# The expanding-window variance series, written out with `var` so the probe never reads the
# library's own `variance_series`.
function reference_variance_series(X)
    T, N = size(X)
    V = fill(NaN, T, N)
    for t in 2:T, i in 1:N
        V[t, i] = var(view(X, 1:t, i))
    end
    return V
end

# Four observations of three assets, every pair eligible, market caps 100, 200 and 300.
const EPS_A = [0.010 -0.020 0.030
               0.020 0.010 -0.010
               -0.030 0.020 0.010
               0.040 -0.010 0.020]
const MASK_A = trues(4, 3)
const MCAP_A = [100.0 200.0 300.0
                100.0 200.0 300.0
                100.0 200.0 300.0
                100.0 200.0 300.0]
# `MarketCapWeights(; p = 1.0)` and `BlendedInverseVarianceWeights(; p = 1.0, lambda = 0.5,
# ratio = 20.0, wins = (0.025, 0.975))`, as the reference implementation answers them.
const REFERENCE_W1_A = [0.16666666666666666 0.3333333333333333 0.5
                        0.16666666666666666 0.3333333333333333 0.5
                        0.5055487368313388 0.21576148102689985 0.27868978214176127
                        0.20074844833880978 0.350675428988682 0.4485761226725082]

# Five observations of four assets, one pair excluded per observation.
const EPS_B = [0.010 -0.020 0.030 0.005
               0.020 0.010 -0.010 0.015
               -0.030 0.020 0.010 -0.005
               0.040 -0.010 0.020 0.025
               0.005 0.030 -0.020 0.010]
const MASK_B = [true true true false
                true true false true
                true true true true
                false true true true
                true false true true]
const MCAP_B = [400.0 100.0 900.0 1600.0
                400.0 100.0 900.0 1600.0
                400.0 100.0 900.0 1600.0
                400.0 100.0 900.0 1600.0
                400.0 100.0 900.0 1600.0]
# `BlendedInverseVarianceWeights(; p = 0.5, lambda = 1.0, ratio = 2.0, wins = (0.1, 0.9))`,
# as the reference implementation answers it.
const REFERENCE_W1_B = [0.3333333333333333 0.16666666666666666 0.5 0.0
                        0.2857142857142857 0.14285714285714285 0.0 0.5714285714285714
                        0.45699777848302126 0.050777530942557915 0.035226912091399554 0.45699777848302126
                        0.0 0.23828125 0.25390625 0.5078125
                        0.15288220551378442 0.0 0.3258145363408521 0.5213032581453634]

@testset "Cross-sectional weights" begin
    @testset "The type tree" begin
        @test MarketCapWeights <: PortfolioOptimisers.AbstractCrossSectionalWeightsAlgorithm
        @test BlendedInverseVarianceWeights <:
              PortfolioOptimisers.AbstractCrossSectionalWeightsAlgorithm
        @test PortfolioOptimisers.AbstractCrossSectionalWeightsAlgorithm <:
              PortfolioOptimisers.AbstractAlgorithm
        # The root answers `false`, so a one-pass member needs no method of its own.
        @test !PortfolioOptimisers.needs_second_pass(MarketCapWeights())
        @test PortfolioOptimisers.needs_second_pass(BlendedInverseVarianceWeights(;
                                                                                  lambda = 0.5))
    end

    @testset "The cap weights raise a power on the eligible pairs alone" begin
        mask = [true true false
                false true true]
        mcap = [4.0 9.0 16.0
                25.0 36.0 49.0]
        @test PortfolioOptimisers.cross_sectional_cap_weights(0.5, mcap, mask) ==
              [2.0 3.0 0.0; 0.0 6.0 7.0]
        @test PortfolioOptimisers.cross_sectional_cap_weights(1.0, mcap, mask) ==
              [4.0 9.0 0.0; 0.0 36.0 49.0]
        # A power of zero needs no capitalisation matrix, and gives every eligible pair one.
        @test PortfolioOptimisers.cross_sectional_cap_weights(0.0, nothing, mask) ==
              [1.0 1.0 0.0; 0.0 1.0 1.0]
        @test PortfolioOptimisers.cross_sectional_cap_weights(0.0, mcap, mask) ==
              [1.0 1.0 0.0; 0.0 1.0 1.0]
        # Both members read the capitalisation the same way.
        @test PortfolioOptimisers.cs_weights_initial(MarketCapWeights(; p = 0.5), mcap,
                                                     mask) ==
              PortfolioOptimisers.cs_weights_initial(BlendedInverseVarianceWeights(;
                                                                                   p = 0.5,
                                                                                   lambda = 0.3),
                                                     mcap, mask)
        # An integer panel promotes to a float result.
        @test eltype(PortfolioOptimisers.cross_sectional_cap_weights(1, [1 2; 3 4],
                                                                     trues(2, 2))) <:
              AbstractFloat
    end

    @testset "The cap weights refuse a malformed design" begin
        mask = [true true false]
        @test_throws PortfolioOptimisers.IsEmptyError PortfolioOptimisers.cross_sectional_cap_weights(0.5,
                                                                                                      nothing,
                                                                                                      falses(0,
                                                                                                             0))
        @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.cross_sectional_cap_weights(0.5,
                                                                                                        nothing,
                                                                                                        mask)
        @test_throws DimensionMismatch PortfolioOptimisers.cross_sectional_cap_weights(0.5,
                                                                                       [1.0 2.0],
                                                                                       mask)
        # An eligible pair must carry a finite, non-negative capitalisation. An ineligible
        # one may carry anything, because the power is never applied to it.
        @test_throws DomainError PortfolioOptimisers.cross_sectional_cap_weights(0.5,
                                                                                 [1.0 -2.0 3.0],
                                                                                 mask)
        @test_throws DomainError PortfolioOptimisers.cross_sectional_cap_weights(0.5,
                                                                                 [1.0 NaN 3.0],
                                                                                 mask)
        @test PortfolioOptimisers.cross_sectional_cap_weights(0.5, [1.0 4.0 -9.0], mask) ==
              [1.0 2.0 0.0]
    end

    @testset "The constructors refuse a malformed policy" begin
        @test_throws DomainError MarketCapWeights(; p = NaN)
        @test_throws DomainError MarketCapWeights(; p = -0.5)
        @test_throws DomainError BlendedInverseVarianceWeights(; p = Inf, lambda = 0.5)
        @test_throws DomainError BlendedInverseVarianceWeights(; lambda = -0.1)
        @test_throws DomainError BlendedInverseVarianceWeights(; lambda = 1.1)
        @test_throws DomainError BlendedInverseVarianceWeights(; lambda = 0.5, ratio = 0.0)
        @test_throws DomainError BlendedInverseVarianceWeights(; lambda = 0.5, ratio = Inf)
        @test_throws DomainError BlendedInverseVarianceWeights(; lambda = 0.5,
                                                               wins = (0.9, 0.1))
        @test_throws DomainError BlendedInverseVarianceWeights(; lambda = 0.5,
                                                               wins = (-0.1, 0.9))
        @test_throws DomainError BlendedInverseVarianceWeights(; lambda = 0.5,
                                                               wins = (0.1, 1.1))
        # The two endpoints and a zero blend are all admissible.
        @test BlendedInverseVarianceWeights(; lambda = 0.0, wins = (0.0, 1.0)) isa
              BlendedInverseVarianceWeights
        @test MarketCapWeights(; p = 0.0) isa MarketCapWeights
    end

    @testset "The lagged inverse variance reads no later observation" begin
        eps = [1.0 2.0
               3.0 6.0
               2.0 4.0]
        IV = PortfolioOptimisers.cross_sectional_lagged_inverse_variance(SimpleVariance(),
                                                                         eps, trues(3, 2))
        # Row 1 carries no estimate, and row 2 reads a fit on one observation.
        @test all(isnan, view(IV, 1, :))
        @test all(isnan, view(IV, 2, :))
        @test IV[3, :] ≈ [0.5, 0.125]
        # Row 3 reads rows 1 and 2 alone, so row 3's own residual moves nothing.
        eps2 = copy(eps)
        eps2[3, :] = [100.0, -100.0]
        IV2 = PortfolioOptimisers.cross_sectional_lagged_inverse_variance(SimpleVariance(),
                                                                          eps2, trues(3, 2))
        @test IV2[3, :] == IV[3, :]
        # A pair outside the mask carries no estimate, whatever its variance.
        IV3 = PortfolioOptimisers.cross_sectional_lagged_inverse_variance(SimpleVariance(),
                                                                          eps,
                                                                          [true false
                                                                           true true
                                                                           false true])
        @test isnan(IV3[3, 1])
        @test IV3[3, 2] ≈ 0.125
        @test_throws PortfolioOptimisers.IsEmptyError PortfolioOptimisers.cross_sectional_lagged_inverse_variance(SimpleVariance(),
                                                                                                                  zeros(0,
                                                                                                                        0),
                                                                                                                  falses(0,
                                                                                                                         0))
        @test_throws DimensionMismatch PortfolioOptimisers.cross_sectional_lagged_inverse_variance(SimpleVariance(),
                                                                                                   eps,
                                                                                                   trues(3,
                                                                                                         3))
    end

    @testset "The bounds cap an extreme inverse variance" begin
        # No winsorisation, a median of 3 and a ratio of 2 cap the row at 6.
        IV = [1.0 3.0 100.0]
        PortfolioOptimisers.cross_sectional_winsorise!(IV, ones(1, 3), (0.0, 1.0))
        @test IV == [1.0 3.0 100.0]
        ready = PortfolioOptimisers.cross_sectional_median_cap!(IV, 2.0)
        @test ready == [true]
        @test IV ≈ [0.1 0.3 0.6]
        # The winsorisation reads the estimation universe, which is the set of pairs with a
        # positive first-pass weight. The third asset leaves it, so the quantiles come from
        # the first two alone and the third is clipped to them.
        IV = [1.0 3.0 100.0]
        PortfolioOptimisers.cross_sectional_winsorise!(IV, [1.0 1.0 0.0], (0.0, 1.0))
        @test IV == [1.0 3.0 3.0]
        PortfolioOptimisers.cross_sectional_median_cap!(IV, 1e6)
        @test IV ≈ [1.0 3.0 3.0] ./ 7.0
        # An observation with no estimate at all takes an all-NaN row and answers `false`.
        IV = [NaN NaN NaN]
        PortfolioOptimisers.cross_sectional_winsorise!(IV, ones(1, 3), (0.0, 1.0))
        @test all(isnan, IV)
        ready = PortfolioOptimisers.cross_sectional_median_cap!(IV, 2.0)
        @test ready == [false]
        @test all(isnan, IV)
        @test_throws DimensionMismatch PortfolioOptimisers.cross_sectional_winsorise!([1.0 2.0],
                                                                                      ones(1,
                                                                                           3),
                                                                                      (0.0,
                                                                                       1.0))
    end

    @testset "The port reproduces the reference implementation" begin
        rwA = BlendedInverseVarianceWeights(; p = 1.0, lambda = 0.5, ratio = 20.0,
                                            wins = (0.025, 0.975))
        W0A = PortfolioOptimisers.cs_weights_initial(rwA, MCAP_A, MASK_A)
        @test W0A == MCAP_A
        W1A = PortfolioOptimisers.cs_weights_refine(rwA, W0A, EPS_A, SimpleVariance(),
                                                    MASK_A)
        @test W1A ≈ REFERENCE_W1_A

        rwB = BlendedInverseVarianceWeights(; p = 0.5, lambda = 1.0, ratio = 2.0,
                                            wins = (0.1, 0.9))
        W0B = PortfolioOptimisers.cs_weights_initial(rwB, MCAP_B, MASK_B)
        @test W0B == [20.0 10.0 30.0 0.0
                      20.0 10.0 0.0 40.0
                      20.0 10.0 30.0 40.0
                      0.0 10.0 30.0 40.0
                      20.0 0.0 30.0 40.0]
        W1B = PortfolioOptimisers.cs_weights_refine(rwB, W0B, EPS_B, SimpleVariance(),
                                                    MASK_B)
        @test W1B ≈ REFERENCE_W1_B
        # An ineligible pair carries no weight at all.
        @test all(iszero, W1B[.!MASK_B])
    end

    @testset "The blend agrees with the mathematics written out" begin
        for (eps, mask, mcap, p, lambda, ratio, wins) in
            ((EPS_A, MASK_A, MCAP_A, 1.0, 0.5, 20.0, (0.025, 0.975)),
             (EPS_B, MASK_B, MCAP_B, 0.5, 1.0, 2.0, (0.1, 0.9)),
             (EPS_B, MASK_B, MCAP_B, 0.0, 0.25, 20.0, (0.05, 0.95)))
            alg = BlendedInverseVarianceWeights(; p = p, lambda = lambda, ratio = ratio,
                                                wins = wins)
            W0 = PortfolioOptimisers.cs_weights_initial(alg, iszero(p) ? nothing : mcap,
                                                        mask)
            W1 = PortfolioOptimisers.cs_weights_refine(alg, W0, eps, SimpleVariance(), mask)
            V = reference_variance_series(eps)
            @test W1 ≈ reference_blended_weights(V, W0, mask, lambda, ratio, wins)
        end
    end

    @testset "The warm-up rows fall back to the cap weights" begin
        alg = BlendedInverseVarianceWeights(; p = 1.0, lambda = 1.0, ratio = 20.0)
        W0 = PortfolioOptimisers.cs_weights_initial(alg, MCAP_A, MASK_A)
        W1 = PortfolioOptimisers.cs_weights_refine(alg, W0, EPS_A, SimpleVariance(), MASK_A)
        # Rows 1 and 2 carry no variance estimate, so they take the normalised cap weights
        # even at a blend of one, which asks for the inverse variance component alone.
        cap = MCAP_A ./ sum(MCAP_A; dims = 2)
        @test W1[1, :] ≈ cap[1, :]
        @test W1[2, :] ≈ cap[2, :]
        @test !isapprox(W1[3, :], cap[3, :])
    end

    @testset "A later residual moves no earlier weight" begin
        alg = BlendedInverseVarianceWeights(; p = 1.0, lambda = 0.5)
        W0 = PortfolioOptimisers.cs_weights_initial(alg, MCAP_A, MASK_A)
        W1 = PortfolioOptimisers.cs_weights_refine(alg, W0, EPS_A, SimpleVariance(), MASK_A)
        # The weight of the last observation reads every earlier residual and none of its
        # own, so changing the last residual moves no weight at all.
        late = copy(EPS_A)
        late[4, :] = [5.0, -5.0, 5.0]
        @test PortfolioOptimisers.cs_weights_refine(alg, W0, late, SimpleVariance(),
                                                    MASK_A) == W1
        # Changing an earlier residual moves every weight that reads it, and no other.
        early = copy(EPS_A)
        early[2, :] = [5.0, -5.0, 5.0]
        W1b = PortfolioOptimisers.cs_weights_refine(alg, W0, early, SimpleVariance(),
                                                    MASK_A)
        @test W1b[1:2, :] == W1[1:2, :]
        @test W1b[3, :] != W1[3, :]
        @test W1b[4, :] != W1[4, :]
    end

    @testset "The realised blend equals the nominal shrinkage" begin
        rwcap = MarketCapWeights(; p = 0.5)
        W0 = PortfolioOptimisers.cs_weights_initial(rwcap, MCAP_B, MASK_B)
        cap = W0 ./ sum(W0; dims = 2)
        pure = BlendedInverseVarianceWeights(; p = 0.5, lambda = 1.0)
        Wiv = PortfolioOptimisers.cs_weights_refine(pure, W0, EPS_B, SimpleVariance(),
                                                    MASK_B)
        for lambda in (0.0, 0.25, 0.5, 0.75, 1.0)
            alg = BlendedInverseVarianceWeights(; p = 0.5, lambda = lambda)
            W1 = PortfolioOptimisers.cs_weights_refine(alg, W0, EPS_B, SimpleVariance(),
                                                       MASK_B)
            # Every eligible observation sums to one, so the two components enter in the
            # proportion the caller wrote.
            @test all(x -> isapprox(x, 1.0), sum(W1; dims = 2))
            @test W1 ≈ lambda .* Wiv .+ (1 - lambda) .* cap
        end
    end

    @testset "An observation with no eligible asset carries no weight" begin
        mask = [true true true
                false false false
                true true true
                true true true]
        alg = BlendedInverseVarianceWeights(; p = 0.5, lambda = 0.5)
        W0 = PortfolioOptimisers.cs_weights_initial(alg, MCAP_A, mask)
        @test all(iszero, W0[2, :])
        W1 = PortfolioOptimisers.cs_weights_refine(alg, W0, EPS_A, SimpleVariance(), mask)
        @test all(iszero, W1[2, :])
        @test all(isfinite, W1)
        @test isapprox(sum(W1[4, :]), 1.0)
    end

    @testset "A zero variance leaves at the median cap" begin
        # The second asset's residual never moves, so its estimated variance is zero and its
        # inverse is infinite. The median cap is what answers it.
        eps = [0.01 0.05 0.03
               0.02 0.05 -0.01
               -0.03 0.05 0.01
               0.04 0.05 0.02]
        alg = BlendedInverseVarianceWeights(; p = 0.0, lambda = 1.0, ratio = 20.0)
        W0 = PortfolioOptimisers.cs_weights_initial(alg, nothing, trues(4, 3))
        W1 = PortfolioOptimisers.cs_weights_refine(alg, W0, eps, SimpleVariance(),
                                                   trues(4, 3))
        @test all(isfinite, W1)
        @test all(x -> isapprox(x, 1.0), sum(W1; dims = 2))
        # The capped asset carries exactly `ratio` times the median of the row.
        for t in 3:4
            @test W1[t, 2] ≈ 20.0 * median(W1[t, :])
        end
    end

    @testset "The refine step refuses a malformed design" begin
        alg = BlendedInverseVarianceWeights(; lambda = 0.5)
        @test_throws DimensionMismatch PortfolioOptimisers.cs_weights_refine(alg,
                                                                             ones(4, 3),
                                                                             EPS_B,
                                                                             SimpleVariance(),
                                                                             MASK_A)
        @test_throws PortfolioOptimisers.IsNonFiniteError PortfolioOptimisers.cs_weights_refine(alg,
                                                                                                fill(NaN,
                                                                                                     4,
                                                                                                     3),
                                                                                                EPS_A,
                                                                                                SimpleVariance(),
                                                                                                MASK_A)
        @test_throws DomainError PortfolioOptimisers.cs_weights_refine(alg, -ones(4, 3),
                                                                       EPS_A,
                                                                       SimpleVariance(),
                                                                       MASK_A)
    end
end
