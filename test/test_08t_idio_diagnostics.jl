#=
The idiosyncratic group of the cross-sectional diagnostics, decided by #709 and built by #800.

The oracle is the reference implementation. The block of `# the reference oracle` below was
built from the very arrays this file builds, run in the reference implementation's own
environment, and its answers are written out as literals. Rebuild them by generating the
same fixture, exporting it, and running the reference's factor model block on it.
=#
using Statistics

@testset "Cross-sectional idiosyncratic diagnostics" begin
    # The fixture the reference oracle was measured on. Every mutation below drives one
    # branch, so a change to any of them invalidates the literals in `# the reference
    # oracle`. The order matters: `eps` is drawn from the unmutated `vs`, so a mutation of
    # `vs` that moved above the draw would change every entry of both arrays.
    rng = StableRNG(13579)
    T, N = 12, 8
    vs = abs.(randn(rng, T, N)) .+ 0.5
    eps = sqrt.(vs) .* randn(rng, T, N)
    # A standardised return of ten drives the tail rate above zero.
    eps[3, 2] = 10 * sqrt(vs[3, 2])
    # A predicted variance of zero leaves the asset with no standardised return.
    vs[2, 3] = 0.0
    # A negative variance estimate is clamped to zero, so it reads the same way.
    vs[4, 5] = -0.25
    # An idiosyncratic return that is not finite costs its own asset alone.
    eps[6, 1] = NaN
    # A variance that is not finite costs the asset in every series of the group.
    vs[7, 4:8] .= NaN
    # A cross-section that is constant has no kurtosis and no skewness.
    eps[8, :] .= 0.0
    # An observation with one finite asset answers no moment at all.
    vs[10, 2:8] .= 0.0
    # An observation with no finite asset answers nothing, not even the tail rate.
    vs[11, :] .= 0.0
    # An observation with three finite assets answers the skewness and not the kurtosis.
    vs[12, 4:8] .= 0.0

    csr = CrossSectionalRegression(; f = 0.02 * randn(rng, T, 3), eps = eps, n = fill(N, T))
    csfm = CrossSectionalFactorModel(; M = randn(rng, N, 3), b = zeros(N), csr = csr,
                                     vs = vs, lag = 1)

    # A `NaN` compares unequal to itself, so the pattern of the absent answers is asserted
    # separately from the values.
    function idio_agrees(a, b; rtol = 1e-10)
        if isnan.(a) != isnan.(b)
            return false
        end
        m = .!isnan.(a)
        return isapprox(a[m], b[m]; rtol = rtol)
    end

    @testset "the reference oracle" begin
        ref_cal = [0.6576024139953216, 0.9566856355619364, 3.5819946565338796,
                   1.0519333239956812, 0.9497821692436044, 1.2977713493444354,
                   0.6959235710960231, 0.0, 0.8998539717557806, NaN, NaN,
                   0.5846367760017855]
        ref_tr3 = [0.0, 0.0, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, NaN, 0.0]
        ref_tr2 = [0.0, 0.0, 0.125, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0, 0.125, 0.0,
                   NaN, 0.0]
        ref_kur = [-0.15254705014793474, -0.8040388814186846, 5.876859489839838,
                   0.184494728400335, -0.7683124676705245, -2.3086251082994793, NaN, NaN,
                   1.408003196921393, NaN, NaN, NaN]
        ref_skw = [-0.13260664664663818, -0.5295931972573044, 2.2683821115641236,
                   0.6761337170660473, -0.387749898137298, 0.26469296344660576,
                   0.03572329744216094, NaN, 0.6348304054154105, NaN, NaN,
                   1.0182452423606356]
        # The reference ranks a cross-section with an unstable sort, so a tie block takes an
        # order the sort chose and not one a rule states. Observation ten carries seven
        # predicted volatilities that are exactly zero, and the reference answers
        # `0.21428571428571427` there. `cs_ordinal_ranks` breaks a tie by the order of the
        # asset axis, which is the deterministic reading, and the reference reproduces this
        # series to the last bit once its own ranks are made stable. The entry below is the
        # library's answer, and it is the only one of the ninety-eight this file asserts
        # that parts from the reference as it runs.
        ref_ic = [-0.09523809523809523, -0.5714285714285714, -0.047619047619047616,
                  -0.3333333333333333, -0.07142857142857142, -0.023809523809523808, NaN,
                  0.2857142857142857, -0.8095238095238095, 0.19047619047619047,
                  -0.19047619047619047]
        ref_rd = [-0.35714285714285715, -0.6071428571428571, -0.30952380952380953,
                  -0.10714285714285714, -0.17857142857142858, -0.11904761904761904, NaN,
                  0.19047619047619047, -0.9047619047619048, NaN, NaN]

        @test idio_agrees(idio_calibration(eps, vs), ref_cal)
        @test idio_agrees(idio_tail_rate(eps, vs), ref_tr3)
        @test idio_agrees(idio_tail_rate(eps, vs; threshold = 2), ref_tr2)
        @test idio_agrees(idio_kurtosis(eps, vs), ref_kur)
        @test idio_agrees(idio_skewness(eps, vs), ref_skw)
        @test idio_agrees(idio_vol_ic(eps, vs), ref_ic)
        @test idio_agrees(idio_vol_residual_dependence(eps, vs), ref_rd)

        # The reference's own summary, under its own key names.
        s = idio_calibration_summary(eps, vs)
        @test isapprox(s.mean_cs_std, 1.0676183867528448; rtol = 1e-12)
        @test isapprox(s.median_cs_std, 0.9248180704996924; rtol = 1e-12)
        @test isapprox(s.mean_kurtosis, 0.4908334153749917; rtol = 1e-12)
        @test isapprox(s.mean_skewness, 0.4275619994726381; rtol = 1e-12)
        @test isapprox(s.mean_tail_rate, 0.011363636363636364; rtol = 1e-12)
    end

    @testset "the level-0 kernel" begin
        z = standardised_idio_returns(eps, vs)
        @test size(z) == (T, N)
        # A zero variance, a negative variance and a return that is not finite each read
        # `NaN`, and a variance that is not finite reads `NaN` too.
        @test isnan(z[2, 3])
        @test isnan(z[4, 5])
        @test isnan(z[6, 1])
        @test all(isnan, z[7, 4:8])
        @test all(isnan, z[11, :])
        # Every other entry is the return over the predicted volatility.
        @test z[1, 1] ≈ eps[1, 1] / sqrt(vs[1, 1])
        @test z[3, 2] ≈ 10.0
        # The predicted volatility clamps a negative variance to zero.
        s = PortfolioOptimisers.idio_predicted_volatility(vs)
        @test s[4, 5] == 0.0
        @test s[1, 1] ≈ sqrt(vs[1, 1])
        @test isnan(s[7, 4])
        # The four `z`-verbs answer the same over `z` as over the two histories.
        @test idio_agrees(idio_calibration(z), idio_calibration(eps, vs))
        @test idio_agrees(idio_tail_rate(z), idio_tail_rate(eps, vs))
        @test idio_agrees(idio_kurtosis(z), idio_kurtosis(eps, vs))
        @test idio_agrees(idio_skewness(z), idio_skewness(eps, vs))
    end

    @testset "the level-2 methods read the block" begin
        @test idio_agrees(idio_calibration(csfm), idio_calibration(eps, vs))
        @test idio_agrees(idio_tail_rate(csfm), idio_tail_rate(eps, vs))
        @test idio_agrees(idio_tail_rate(csfm; threshold = 2),
                          idio_tail_rate(eps, vs; threshold = 2))
        @test idio_agrees(idio_kurtosis(csfm), idio_kurtosis(eps, vs))
        @test idio_agrees(idio_skewness(csfm), idio_skewness(eps, vs))
        @test idio_agrees(idio_vol_ic(csfm), idio_vol_ic(eps, vs))
        @test idio_agrees(idio_vol_residual_dependence(csfm),
                          idio_vol_residual_dependence(eps, vs))
        @test idio_calibration_summary(csfm) == idio_calibration_summary(eps, vs)
        @test idio_calibration_summary(csfm; threshold = 2) ==
              idio_calibration_summary(eps, vs; threshold = 2)
    end

    @testset "the analytic cases" begin
        # A panel whose predicted variance is the true one, constant in time and ranking
        # the assets. The standardised returns are then standard normal by construction.
        rng2 = StableRNG(2468)
        T2, N2 = 400, 80
        vs2 = repeat(collect(range(0.5, 4.0; length = N2))', T2, 1)
        eps2 = sqrt.(vs2) .* randn(rng2, T2, N2)
        s2 = idio_calibration_summary(eps2, vs2)
        # The calibration is near one, the tail rate near the Gaussian reference, and the
        # kurtosis and the skewness near zero.
        @test isapprox(s2.mean_cs_std, 1.0; atol = 0.02)
        @test isapprox(s2.median_cs_std, 1.0; atol = 0.02)
        # The Gaussian reference of a threshold of three, `2Φ(-3)`.
        @test isapprox(s2.mean_tail_rate, 0.002699796063260207; atol = 0.0015)
        @test isapprox(s2.mean_kurtosis, 0.0; atol = 0.15)
        @test isapprox(s2.mean_skewness, 0.0; atol = 0.08)
        # The volatility ranks the truth, so the information coefficient is high, and the
        # standardised move no longer depends on it, so the residual dependence is near zero.
        @test mean(idio_vol_ic(eps2, vs2)) > 0.2
        @test isapprox(mean(idio_vol_residual_dependence(eps2, vs2)), 0.0; atol = 0.03)
        # A scaled variance history moves the calibration by the inverse scale.
        @test idio_calibration(eps2, 4 .* vs2) ≈ idio_calibration(eps2, vs2) ./ 2
        # It moves neither dependence series: a positive scale is monotone, so it leaves
        # every rank of both cross-sections where it was.
        @test idio_vol_ic(eps2, 4 .* vs2) == idio_vol_ic(eps2, vs2)
        @test idio_vol_residual_dependence(eps2, 4 .* vs2) ==
              idio_vol_residual_dependence(eps2, vs2)
        # The five entries of the summary are the means and the median of the series. The
        # summary sums the entries in order and `Statistics.mean` sums them pairwise, so
        # the two agree to the last bits of the answer and not bit for bit.
        z2 = standardised_idio_returns(eps2, vs2)
        @test s2.mean_cs_std ≈ mean(idio_calibration(z2)) rtol = 1e-12
        @test s2.median_cs_std == median(idio_calibration(z2))
        @test s2.mean_kurtosis ≈ mean(idio_kurtosis(z2)) rtol = 1e-12
        @test s2.mean_skewness ≈ mean(idio_skewness(z2)) rtol = 1e-12
        @test s2.mean_tail_rate ≈ mean(idio_tail_rate(z2)) rtol = 1e-12
    end

    @testset "an aggregate over a series that never answered" begin
        # Every predicted variance is zero, so no observation carries a standardised
        # return, every series is `NaN` throughout, and every aggregate answers `NaN`.
        eps0 = ones(3, 4)
        vs0 = zeros(3, 4)
        @test all(isnan, standardised_idio_returns(eps0, vs0))
        @test all(isnan, idio_calibration(eps0, vs0))
        @test all(isnan, idio_tail_rate(eps0, vs0))
        s0 = idio_calibration_summary(eps0, vs0)
        @test all(isnan,
                  (s0.mean_cs_std, s0.median_cs_std, s0.mean_kurtosis, s0.mean_skewness,
                   s0.mean_tail_rate))
        @test isnan(PortfolioOptimisers.idio_nan_mean([NaN, NaN]))
        @test isnan(PortfolioOptimisers.idio_nan_median([NaN, NaN]))
        @test PortfolioOptimisers.idio_nan_mean([1.0, NaN, 3.0]) == 2.0
        @test PortfolioOptimisers.idio_nan_median([1.0, NaN, 3.0, 5.0]) == 3.0
        # The kernel of the moments answers `NaN` at an observation with no finite asset.
        m = PortfolioOptimisers.idio_row_moments(standardised_idio_returns(eps0, vs0), 1)
        @test m.n == 0
        @test all(isnan, (m.m2, m.m3, m.m4))
    end

    @testset "the refusals" begin
        @test_throws PortfolioOptimisers.IsEmptyError standardised_idio_returns(zeros(0, 0),
                                                                                zeros(0, 0))
        @test_throws DimensionMismatch standardised_idio_returns(eps, vs[:, 1:4])
        @test_throws PortfolioOptimisers.IsEmptyError idio_vol_ic(zeros(0, 0), zeros(0, 0))
        @test_throws DimensionMismatch idio_vol_ic(eps, vs[:, 1:4])
        # The two dependence series read two observations at a time, so one is not enough.
        @test_throws DimensionMismatch idio_vol_ic(eps[1:1, :], vs[1:1, :])
        @test_throws DimensionMismatch idio_vol_residual_dependence(eps[1:1, :], vs[1:1, :])
        # A block that carries no cross-sectional fit, and one that carries no variance
        # history, each name the field the caller must populate.
        no_csr = CrossSectionalFactorModel(; M = randn(rng, N, 3), b = zeros(N), vs = vs,
                                           lag = 1)
        no_vs = CrossSectionalFactorModel(; M = randn(rng, N, 3), b = zeros(N), csr = csr,
                                          lag = 1)
        for f in
            (idio_calibration, idio_tail_rate, idio_kurtosis, idio_skewness, idio_vol_ic,
             idio_vol_residual_dependence, idio_calibration_summary)
            @test_throws PortfolioOptimisers.IsNothingError f(no_csr)
            @test_throws PortfolioOptimisers.IsNothingError f(no_vs)
        end
        @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.idio_diagnostic_data(nothing,
                                                                                                 nothing)
    end
end
