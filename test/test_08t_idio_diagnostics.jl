#=
The idiosyncratic group of the cross-sectional diagnostics, decided by #709 and built by #800.

The testset `the stored oracle` compares with literals that the oracle gave on the very
arrays this file builds. Rebuild them by an export of the same fixture and a run of the
oracle's factor model block on it. Each comparison is cell by cell (`parity_compare`), and
states the worst value measured beside its tolerance.
=#
using Statistics
include(joinpath(@__DIR__, "parity_harness.jl"))

@testset "Cross-sectional idiosyncratic diagnostics" begin
    # The fixture the oracle was measured on. Every mutation below drives one branch, so a
    # change to any of them invalidates the literals of `the stored oracle`. The order
    # matters: `eps` is drawn from the unmutated `vs`, so a mutation of `vs` that moved
    # above the draw would change every entry of both arrays.
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

    # Cell by cell: the pattern of the absent answers agrees exactly, and each finite cell
    # lies within `rtol` of its own size.
    # Against the oracle, the calibration measures maxrel 1.7e-16 and the rest is
    # bit-equal, except the two higher moments, which state their own tolerance.
    function idio_agrees(a, b, label = ""; rtol = 1e-14)
        return parity_compare(a, b; rtol = rtol, name = label).ok
    end

    @testset "the stored oracle" begin
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
        # The oracle ranks a cross-section with an unstable sort, so a tie block takes an
        # order the sort chose and not one a rule states. Observation ten carries seven
        # predicted volatilities that are exactly zero, and the oracle answers
        # `0.21428571428571427` there. Under `ties = :ordinal`, `cs_ranks` breaks a tie by the
        # order of the asset axis, which is the deterministic reading, and the oracle
        # reproduces this series to the last bit once its own ranks are made stable. The
        # entry below is the library's answer under that rule, and it is the only one of the
        # ninety-eight this file asserts that parts from the oracle as it runs.
        ref_ic = [-0.09523809523809523, -0.5714285714285714, -0.047619047619047616,
                  -0.3333333333333333, -0.07142857142857142, -0.023809523809523808, NaN,
                  0.2857142857142857, -0.8095238095238095, 0.19047619047619047,
                  -0.19047619047619047]
        ref_rd = [-0.35714285714285715, -0.6071428571428571, -0.30952380952380953,
                  -0.10714285714285714, -0.17857142857142858, -0.11904761904761904, NaN,
                  0.19047619047619047, -0.9047619047619048, NaN, NaN]

        # The oracle divides each return by the volatility of its own observation, which
        # `ahead = false` states (#1387).
        @test idio_agrees(idio_calibration(eps, vs; ahead = false), ref_cal)
        @test idio_agrees(idio_tail_rate(eps, vs; ahead = false), ref_tr3)
        @test idio_agrees(idio_tail_rate(eps, vs; threshold = 2, ahead = false), ref_tr2)
        # Measured maxrel 1.9e-14 and 3.6e-15, maxscaled 4.9e-16 and 2.1e-16. An excess
        # kurtosis subtracts 3 from a ratio of moments, and a skewness near zero cancels its
        # centred cubes, so a small cell carries the round-off of the largest one.
        @test idio_agrees(idio_kurtosis(eps, vs; ahead = false), ref_kur, "kurtosis";
                          rtol = 1e-13)
        @test idio_agrees(idio_skewness(eps, vs; ahead = false), ref_skw, "skewness";
                          rtol = 5e-14)
        @test idio_agrees(idio_vol_ic(eps, vs; ties = :ordinal), ref_ic)
        @test idio_agrees(idio_vol_residual_dependence(eps, vs; ties = :ordinal), ref_rd)
        # Under the default `ties = :average`, the two observations whose predictions tie
        # move. Observation eleven predicts zero for every asset, so its ranks are constant
        # and it has no coefficient. Observation ten gives its seven zeros their mean rank,
        # which `tiedrank` states. No other entry has a tie, so each keeps the oracle's
        # number. The residual dependence divides by the zeros, so it has no tie to move.
        ica = idio_vol_ic(eps, vs)
        sig = PortfolioOptimisers.idio_predicted_volatility(vs)
        @test idio_agrees(ica[1:9], ref_ic[1:9])
        @test ica[10] ≈ cor(tiedrank(sig[10, :]), tiedrank(abs.(eps[11, :])))
        @test isnan(ica[11])
        @test idio_agrees(idio_vol_residual_dependence(eps, vs), ref_rd)
        @test isequal(idio_vol_ic(csfm), ica)
        @test isequal(idio_vol_ic(csfm; ties = :ordinal),
                      idio_vol_ic(eps, vs; ties = :ordinal))

        # The oracle's own summary, under its own key names. Measured maxrel 1.0e-15.
        s = idio_calibration_summary(eps, vs; ahead = false)
        @test idio_agrees([s.mean_cs_std, s.median_cs_std, s.mean_kurtosis, s.mean_skewness,
                           s.mean_tail_rate],
                          [1.0676183867528448, 0.9248180704996924, 0.4908334153749917,
                           0.4275619994726381, 0.011363636363636364], "summary")
    end

    @testset "the level-0 kernel" begin
        z = standardised_idio_returns(eps, vs; ahead = false)
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
        @test idio_agrees(idio_calibration(z), idio_calibration(eps, vs; ahead = false))
        @test idio_agrees(idio_tail_rate(z), idio_tail_rate(eps, vs; ahead = false))
        @test idio_agrees(idio_kurtosis(z), idio_kurtosis(eps, vs; ahead = false))
        @test idio_agrees(idio_skewness(z), idio_skewness(eps, vs; ahead = false))
        # The default divides the return of row `t` by the volatility of row `t - 1`, so the
        # first row has no forecast and every other row is the same-row kernel of the
        # shifted variance history.
        za = standardised_idio_returns(eps, vs)
        @test all(isnan, za[1, :])
        @test isequal(za[2:end, :],
                      standardised_idio_returns(eps[2:end, :], vs[1:(end - 1), :];
                                                ahead = false))
        @test za[4, 2] ≈ eps[4, 2] / sqrt(vs[3, 2])
        # The zero variance of row 2 costs the return of row 3, not of row 2.
        @test isfinite(za[2, 3]) && isnan(za[3, 3])
        @test idio_agrees(idio_calibration(za), idio_calibration(eps, vs))
        @test idio_agrees(idio_tail_rate(za), idio_tail_rate(eps, vs))
        @test idio_agrees(idio_kurtosis(za), idio_kurtosis(eps, vs))
        @test idio_agrees(idio_skewness(za), idio_skewness(eps, vs))
        @test idio_calibration_summary(eps, vs) ==
              idio_calibration_summary(eps[2:end, :], vs[1:(end - 1), :]; ahead = false)
    end

    @testset "a variance that has read its own return bounds the standardised return" begin
        # An exponentially weighted variance puts the weight `1 - λ` on the latest squared
        # return, so `v_t ≥ (1 - λ) ε_t²` and the same-row ratio can never pass
        # `1 / sqrt(1 - λ)`, however large the shock. The variance of the previous row has not
        # read the shock, so the default reports it at its full size (#1387).
        lam = 0.9
        e = [0.01 0.01; -0.01 0.01; 0.01 -0.01; 0.5 0.01; 0.01 0.01]
        v = similar(e)
        v[1, :] .= 1e-4
        for t in 2:size(e, 1)
            v[t, :] = lam .* v[t - 1, :] .+ (1 - lam) .* e[t, :] .^ 2
        end
        zs = standardised_idio_returns(e, v; ahead = false)
        za = standardised_idio_returns(e, v)
        @test maximum(abs, filter(isfinite, zs)) <= 1 / sqrt(1 - lam)
        @test za[4, 1] ≈ 0.5 / sqrt(v[3, 1])
        @test za[4, 1] > 45
        @test idio_tail_rate(e, v; threshold = 4, ahead = false)[4] == 0.0
        @test idio_tail_rate(e, v; threshold = 4)[4] == 0.5
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
        for f in (idio_calibration, idio_kurtosis, idio_skewness, idio_tail_rate)
            @test idio_agrees(f(csfm; ahead = false), f(eps, vs; ahead = false))
        end
        @test idio_calibration_summary(csfm; threshold = 2, ahead = false) ==
              idio_calibration_summary(eps, vs; threshold = 2, ahead = false)
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
        @test idio_agrees(idio_calibration(eps2, 4 .* vs2),
                          idio_calibration(eps2, vs2) ./ 2)
        # It moves neither dependence series: a positive scale is monotone, so it leaves
        # every rank of both cross-sections where it was.
        @test idio_vol_ic(eps2, 4 .* vs2) == idio_vol_ic(eps2, vs2)
        @test idio_vol_residual_dependence(eps2, 4 .* vs2) ==
              idio_vol_residual_dependence(eps2, vs2)
        # The five entries of the summary are the means and the median of the series. The
        # summary sums the entries in order and `Statistics.mean` sums them pairwise, so
        # the two agree to the last bits of the answer and not bit for bit. The first row
        # has no forecast, so every series is `NaN` there and the summary skips it.
        z2 = standardised_idio_returns(eps2, vs2)
        @test all(isnan, z2[1, :])
        @test s2.mean_cs_std ≈ mean(idio_calibration(z2)[2:end]) rtol = 1e-12
        @test s2.median_cs_std == median(idio_calibration(z2)[2:end])
        @test s2.mean_kurtosis ≈ mean(idio_kurtosis(z2)[2:end]) rtol = 1e-12
        @test s2.mean_skewness ≈ mean(idio_skewness(z2)[2:end]) rtol = 1e-12
        @test s2.mean_tail_rate ≈ mean(idio_tail_rate(z2)[2:end]) rtol = 1e-12
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

    @testset "a constant cross-section, an infinite entry and an integer input" begin
        # A cross-section of equal non-zero values has no skewness and no kurtosis. Its mean
        # carries round-off, so a centred moment computed from it is often a tiny positive
        # number and not zero: before the exact test, 4195 of 20000 such rows of 3 to 7
        # assets gave a finite skewness, up to 2.45. The counts avoid 4, whose mean is exact.
        for nc in (3, 5, 6, 7), x in (0.1, -0.7, 1.3, 2.9)
            zc = fill(x, 2, nc)
            @test all(isnan, idio_skewness(zc))
            @test nc < 4 || all(isnan, idio_kurtosis(zc))
            @test idio_calibration(zc) == zeros(2)
            mc = PortfolioOptimisers.idio_row_moments(zc, 1)
            @test (mc.n, mc.m2, mc.m3, mc.m4) == (nc, 0.0, 0.0, 0.0)
        end
        # An infinite entry enters neither the count nor the sum of the tail rate, so the
        # rate over the finite entries 0.5, 4 and -5 is two in three, never one.
        @test idio_tail_rate([Inf 0.5 4.0 NaN -5.0]) == [2 / 3]
        # An integer history answers in floating point rather than raising `InexactError`.
        e_int = [1 -2 3 0; 3 4 -1 2; 0 1 2 -3]
        v_int = [1 4 1 0; 1 0 4 1; 1 1 1 1]
        z_int = standardised_idio_returns(e_int, v_int; ahead = false)
        @test eltype(z_int) == Float64
        @test isequal(z_int, [1.0 -1.0 3.0 NaN; 3.0 NaN -0.5 2.0; 0.0 1.0 2.0 -3.0])
        @test idio_calibration([1 2 3; 4 5 7]) ≈ [1.0, std([4, 5, 7])]
        # A Float32 history keeps its element type through every series.
        rng32 = StableRNG(7)
        e32 = randn(rng32, Float32, 6, 9)
        v32 = rand(rng32, Float32, 6, 9) .+ 0.5f0
        for f in
            (idio_calibration, idio_tail_rate, idio_kurtosis, idio_skewness, idio_vol_ic,
             idio_vol_residual_dependence)
            @test eltype(f(e32, v32)) == Float32
        end
        @test idio_calibration_summary(e32, v32).mean_cs_std isa Float32
        # Each series takes its type from its own operation, never from `float`. A square
        # root is inexact, so a `Rational` history answers in the type `sqrt` lands in. A
        # rate and the moment ratio of the kurtosis only divide, so an exact `z` stays exact.
        er = Rational{Int}.(round.(Int, 8 .* e32)) .// 8
        vr = Rational{Int}.(round.(Int, 8 .* v32)) .// 8
        @test eltype(standardised_idio_returns(er, vr)) == typeof(sqrt(1 // 1))
        @test eltype(idio_vol_ic(er, vr)) == typeof(sqrt(1 // 1))
        zr = [1//2 -3//2 2//1 1//4 -1//1; 3//1 -1//3 1//5 2//1 -2//1]
        @test idio_tail_rate(zr; threshold = 2) == [0 // 1, 1 // 5]
        @test eltype(idio_kurtosis(zr)) == Rational{Int}
        @test eltype(idio_skewness(zr)) == typeof(sqrt(1 // 1))
        @test PortfolioOptimisers.idio_nan_mean([1 // 2, 1 // 3]) === 5 // 12
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
