#=
The summary of the cross-sectional diagnostics, decided by #709 and built by #801.

The oracle is the reference implementation. The blocks of `# the reference oracle` below were
built from the very arrays this file builds, run in the reference implementation's own
environment, and its answers are written out as literals. Rebuild them by generating the
same fixtures, exporting them, and running the reference's factor model block on them.

The first fixture is the one `test_08r_cs_regression_diagnostics.jl` uses, whose design is
well conditioned, so its Gram columns carry values. The second is the one
`test_08s_exposure_diagnostics.jl` uses, whose third factor is a constant intercept, so it
drives the patch that reads a constant exposure as perfectly stable.
=#
using Statistics

@testset "Factor model summary" begin
    # A `NaN` compares unequal to itself, so the pattern of the absent answers is asserted
    # separately from the values.
    function summary_agrees(a, b; rtol = 1e-10)
        if isnan.(a) != isnan.(b)
            return false
        end
        m = .!isnan.(a)
        return isapprox(a[m], b[m]; rtol = rtol)
    end

    # The first fixture. Every mutation drives one branch of the regression group, so a
    # change to any of them invalidates the literals in `# the reference oracle`.
    rngA = StableRNG(987654321)
    TA, NA, KA = 8, 6, 3
    MsA = randn(rngA, TA, NA, KA)
    fA = 0.02 * randn(rngA, TA, KA)
    epsA = 0.01 * randn(rngA, TA, NA)
    rwA = abs.(randn(rngA, TA, NA)) .+ 0.1
    bwA = fill(1 / NA, TA, NA)
    rwA[3, 2] = 0.0
    rwA[6, 5] = 0.0
    rwA[8, 3] = 0.0
    rwA[8, 4] = 0.0
    rwA[8, 5] = 0.0
    rwA[8, 6] = 0.0
    epsA[4, 3] = NaN
    MsA[2, 4, 1] = NaN
    # The factor return of one observation is absent, which costs that observation its
    # t-statistic and costs the whole series its autocorrelation.
    fA[5, 2] = NaN
    MsA[6, :, 3] = MsA[6, :, 1]

    csrA = CrossSectionalRegression(; f = fA, eps = epsA, n = fill(NA, TA))
    csfmA = CrossSectionalFactorModel(; M = MsA[TA, :, :], b = zeros(NA), csr = csrA,
                                      Ms = MsA, rw = rwA, bw = bwA, lag = 1)

    @testset "the reference oracle, the well-conditioned fixture" begin
        ref_ann_return = [1.7688020592276463, -4.704611043276863, 0.06102501991895286]
        ref_ann_vol = [0.2500971159903252, 0.30449966488269103, 0.2801176875443828]
        ref_sharpe = [7.07246084075624, -15.450299576156583, 0.21785493252468693]
        ref_autocorr = [-0.5425646747475837, NaN, -0.39620089475439907]
        ref_mean_abs_t = [1.3761850065112538, 4.764397545417589, 3.9180561817454196]
        ref_t_rate = [0.4, 0.6, 0.4]
        ref_mean_vif = [2.0598549319598916, 1.8428351056716823, 1.652400329922471]
        ref_stability = [-0.4257338194977316, 0.11924492275222029, 0.1814524979202049]
        ref_coverage = [0.9791666666666667, 1.0, 1.0]

        fs = factor_model_summary(csfmA; ppy = 252, step = 3)
        @test summary_agrees(fs.ann_return, ref_ann_return)
        @test summary_agrees(fs.ann_volatility, ref_ann_vol)
        @test summary_agrees(fs.sharpe, ref_sharpe)
        @test summary_agrees(fs.autocorr, ref_autocorr)
        @test summary_agrees(fs.mean_abs_t, ref_mean_abs_t)
        @test summary_agrees(fs.t_rate, ref_t_rate)
        @test summary_agrees(fs.mean_vif, ref_mean_vif)
        @test summary_agrees(fs.stability, ref_stability)
        @test summary_agrees(fs.coverage, ref_coverage)
        @test fs.ppy == 252

        # `ppy = 1` reports the per-period numbers, which is the #708 default.
        ref_ann_return_1 = [0.007019055790585898, -0.018669091441574852,
                            0.00024216277745616214]
        ref_ann_vol_1 = [0.015754637441068472, 0.0191816758948121, 0.01764575569221267]
        ref_sharpe_1 = [0.4455231557591381, -0.973277389522785, 0.013723570794025676]
        fs1 = factor_model_summary(csfmA; ppy = 1, step = 3)
        @test summary_agrees(fs1.ann_return, ref_ann_return_1)
        @test summary_agrees(fs1.ann_volatility, ref_ann_vol_1)
        @test summary_agrees(fs1.sharpe, ref_sharpe_1)
        # Only the first three columns carry the annualisation.
        @test summary_agrees(fs1.autocorr, ref_autocorr)
        @test summary_agrees(fs1.mean_abs_t, ref_mean_abs_t)
        @test summary_agrees(fs1.stability, ref_stability)
        @test summary_agrees(fs1.coverage, ref_coverage)

        # The stability reads the weight history the caller names, and the coverage does
        # not: the reference wires its universe to the regression weights alone.
        ref_stability_id = [-0.4257338194977314, 0.11924492275222023, 0.18145249792020487]
        ref_stability_rw = [-0.25229769514274375, 0.0032308107079838436,
                            0.26713295230303086]
        fs_id = factor_model_summary(csfmA; ppy = 252, step = 3,
                                     weighting = IdentityMetric())
        fs_rw = factor_model_summary(csfmA; ppy = 252, step = 3,
                                     weighting = RegressionWeightMetric())
        @test summary_agrees(fs_id.stability, ref_stability_id)
        @test summary_agrees(fs_rw.stability, ref_stability_rw)
        @test summary_agrees(fs_id.coverage, ref_coverage)
        @test summary_agrees(fs_rw.coverage, ref_coverage)

        # A history with no more observations than `step` has no stability series at all.
        fs21 = factor_model_summary(csfmA; ppy = 252, step = 21)
        @test all(isnan, fs21.stability)
        @test summary_agrees(fs21.coverage, ref_coverage)

        # The threshold steers the exceedance rate alone.
        ref_t_rate_1 = [0.4, 0.8, 0.6]
        fst = factor_model_summary(csfmA; ppy = 252, step = 3, threshold = 1)
        @test summary_agrees(fst.t_rate, ref_t_rate_1)
        @test summary_agrees(fst.mean_abs_t, ref_mean_abs_t)
    end

    # The second fixture. Its third factor is a constant intercept and its fourth is finite
    # on two assets alone.
    rngB = StableRNG(24680)
    TB, NB, KB = 10, 6, 4
    MsB = randn(rngB, TB, NB, KB)
    fB = 0.02 * randn(rngB, TB, KB)
    epsB = 0.01 * randn(rngB, TB, NB)
    rwB = abs.(randn(rngB, TB, NB)) .+ 0.1
    bwB = fill(1 / NB, TB, NB)
    vsB = abs.(randn(rngB, TB, NB)) .+ 0.5
    MsB[:, :, 3] .= 1.0
    MsB[:, 3:6, 4] .= NaN
    rwB[3, 2] = 0.0
    bwB[7, 4] = 0.0
    MsB[2, 4, 1] = NaN
    MsB[5, 1, 2] = NaN
    MsB[5, 2, 2] = NaN
    MsB[5, 3, 2] = NaN
    MsB[5, 4, 2] = NaN

    csrB = CrossSectionalRegression(; f = fB, eps = epsB, n = fill(NB, TB))
    csfmB = CrossSectionalFactorModel(; M = MsB[TB, :, :], b = zeros(NB), csr = csrB,
                                      Ms = MsB, vs = vsB, rw = rwB, bw = bwB, lag = 1)

    @testset "the reference oracle, the constant-exposure fixture" begin
        ref_ann_return = [-0.4907810529445153, 1.7641281876800439, 3.287865325954306,
                          2.112811248657294]
        ref_ann_vol = [0.33405767428380584, 0.3481792051822995, 0.27711562516097665,
                       0.3038473320095795]
        ref_sharpe = [-1.4691506608752887, 5.06672472514946, 11.86459740061349,
                      6.953529045931142]
        ref_autocorr = [-0.1731707888686296, -0.11429924268878187, -0.16733266080439935,
                        -0.009766919079024838]
        ref_t_rate = [0.0, 0.0, 0.0, 0.0]
        ref_stability = [-0.15454019826790003, -0.041871561766556646, 1.0, NaN]
        ref_coverage = [0.9833333333333334, 0.9333333333333332, 1.0, 0.32]

        fs = factor_model_summary(csfmB; ppy = 252, step = 3)
        @test summary_agrees(fs.ann_return, ref_ann_return)
        @test summary_agrees(fs.ann_volatility, ref_ann_vol)
        @test summary_agrees(fs.sharpe, ref_sharpe)
        @test summary_agrees(fs.autocorr, ref_autocorr)
        @test summary_agrees(fs.t_rate, ref_t_rate)
        @test summary_agrees(fs.stability, ref_stability)
        @test summary_agrees(fs.coverage, ref_coverage)
        # No cross-section of this fixture carries enough eligible assets for a Gram
        # answer, so the two Gram averages are absent at every factor.
        @test all(isnan, fs.mean_abs_t)
        @test all(isnan, fs.mean_vif)

        # A constant exposure keeps its `1` when the history is too short for a series.
        ref_stability_21 = [NaN, NaN, 1.0, NaN]
        fs21 = factor_model_summary(csfmB; ppy = 252, step = 21)
        @test summary_agrees(fs21.stability, ref_stability_21)
    end

    @testset "the factor return columns are a performance summary" begin
        # The first and the third series carry no absent observation, so the summary's
        # first three columns are `performance_summary` on the series itself.
        fs = factor_model_summary(csfmA; ppy = 252, step = 3)
        for k in (1, 3)
            ps = performance_summary(fA[:, k]; periods_per_year = 252)
            @test isapprox(fs.ann_return[k], ps.ann_return)
            @test isapprox(fs.ann_volatility[k], ps.ann_volatility)
            @test isapprox(fs.sharpe[k], ps.sharpe)
        end
        # The second series carries an absent observation. The summary drops it and
        # `performance_summary` does not, so only the summary answers there.
        ps2 = performance_summary(fA[:, 2]; periods_per_year = 252)
        @test isnan(ps2.ann_return)
        @test isfinite(fs.ann_return[2])
    end

    @testset "the Gram columns are the means of the level-2 series" begin
        fs = factor_model_summary(csfmA; ppy = 252, step = 3)
        t = cs_regression_t_stats(csfmA)
        vif = exposure_vif(csfmA)
        rate = cs_regression_t_stat_exceedance_rate(csfmA; threshold = 2)
        for k in 1:KA
            at = filter(!isnan, abs.(t[:, k]))
            av = filter(!isnan, vif[:, k])
            @test isapprox(fs.mean_abs_t[k], Statistics.mean(at))
            @test isapprox(fs.mean_vif[k], Statistics.mean(av))
        end
        @test fs.t_rate == rate
        # The stability column is the MEDIAN of the series, and not its mean.
        S = exposure_stability(csfmA; step = 3)
        for k in 1:KA
            @test isapprox(fs.stability[k], Statistics.median(filter(!isnan, S[:, k])))
        end
        @test fs.coverage == exposure_coverage(csfmA; weighting = RegressionWeightMetric())
    end

    @testset "a block with no exposure history carries five absent columns" begin
        blk = CrossSectionalFactorModel(; M = MsA[TA, :, :], b = zeros(NA), csr = csrA,
                                        rw = rwA, bw = bwA, lag = 1)
        fs = factor_model_summary(blk; ppy = 252, step = 3)
        @test isnothing(fs.mean_abs_t)
        @test isnothing(fs.t_rate)
        @test isnothing(fs.mean_vif)
        @test isnothing(fs.stability)
        @test isnothing(fs.coverage)
        # The four factor return columns are unchanged by the absence.
        full = factor_model_summary(csfmA; ppy = 252, step = 3)
        @test summary_agrees(fs.ann_return, full.ann_return)
        @test summary_agrees(fs.ann_volatility, full.ann_volatility)
        @test summary_agrees(fs.sharpe, full.sharpe)
        @test summary_agrees(fs.autocorr, full.autocorr)
    end

    @testset "a re-based block writes NaN on the factor the basis dropped" begin
        rng5 = StableRNG(19283746)
        Tr, Nr, Kr = 6, 5, 3
        Msr = randn(rng5, Tr, Nr, Kr)
        fr = 0.02 * randn(rng5, Tr, Kr)
        epsr = 0.01 * randn(rng5, Tr, Nr)
        rwr = abs.(randn(rng5, Tr, Nr)) .+ 0.1
        # One constrained family holds factors 1 and 2, and drops the second of them.
        fcb = FactorFamilyBasis(; fnm = ["industry"], fi = [[1, 2]], di = [2],
                                ratios = reshape(collect(range(0.4, 0.9; length = Tr)), Tr,
                                                 1), K = Kr)
        csr_r = CrossSectionalRegression(; f = fr, eps = epsr, n = fill(Nr, Tr))
        L = PortfolioOptimisers.reduce_loadings(fcb, Msr[Tr, :, :])
        blk = CrossSectionalFactorModel(; M = Msr[Tr, :, :], L = L, b = zeros(Nr),
                                        csr = csr_r, Ms = Msr, rw = rwr, fcb = fcb, lag = 1,
                                        nf = ["value", "size", "momentum"])
        fs = factor_model_summary(blk; ppy = 1, step = 2,
                                  weighting = RegressionWeightMetric())
        # The answer is on the raw axis, which is wider than the reduced one.
        @test length(fs.ann_return) == Kr
        @test length(fs.mean_abs_t) == Kr
        # "size" is the dropped member, so it carries no Gram answer.
        @test isnan(fs.mean_abs_t[2])
        @test isnan(fs.t_rate[2])
        @test isnan(fs.mean_vif[2])
        # The retained members carry the reduced answers, in the reduced order.
        t = cs_regression_t_stats(blk)
        vif = exposure_vif(blk)
        rate = cs_regression_t_stat_exceedance_rate(blk)
        @test size(t, 2) == 2
        for (raw, red) in ((1, 1), (3, 2))
            @test isapprox(fs.mean_abs_t[raw],
                           Statistics.mean(filter(!isnan, abs.(t[:, red]))))
            @test isapprox(fs.mean_vif[raw], Statistics.mean(filter(!isnan, vif[:, red])))
            @test fs.t_rate[raw] == rate[red]
        end
        # The exposure columns read the raw axis, so they carry an answer at every factor.
        @test !any(isnothing, (fs.stability, fs.coverage))
        @test length(fs.coverage) == Kr
    end

    @testset "the refusals" begin
        # A block with no cross-sectional fit carries no factor return history.
        blk = CrossSectionalFactorModel(; M = MsA[TA, :, :], b = zeros(NA), Ms = MsA,
                                        rw = rwA, bw = bwA, lag = 1)
        @test_throws PortfolioOptimisers.IsNothingError factor_model_summary(blk)
        # A non-positive annualisation factor has no meaning.
        @test_throws DomainError factor_model_summary(csfmA; ppy = 0)
        @test_throws DomainError factor_model_summary(csfmA; ppy = -252)
        # A re-based block that names no factor cannot be joined back to the raw axis.
        rng6 = StableRNG(5647382)
        Tn, Nn, Kn = 6, 5, 3
        Msn = randn(rng6, Tn, Nn, Kn)
        fn = 0.02 * randn(rng6, Tn, Kn)
        epsn = 0.01 * randn(rng6, Tn, Nn)
        rwn = abs.(randn(rng6, Tn, Nn)) .+ 0.1
        fcbn = FactorFamilyBasis(; fnm = ["industry"], fi = [[1, 2]], di = [2],
                                 ratios = reshape(collect(range(0.4, 0.9; length = Tn)), Tn,
                                                  1), K = Kn)
        csr_n = CrossSectionalRegression(; f = fn, eps = epsn, n = fill(Nn, Tn))
        Ln = PortfolioOptimisers.reduce_loadings(fcbn, Msn[Tn, :, :])
        blkn = CrossSectionalFactorModel(; M = Msn[Tn, :, :], L = Ln, b = zeros(Nn),
                                         csr = csr_n, Ms = Msn, rw = rwn, fcb = fcbn,
                                         lag = 1)
        @test_throws PortfolioOptimisers.IsNothingError factor_model_summary(blkn; step = 2,
                                                                             weighting = RegressionWeightMetric())
    end

    @testset "the kernels of the summary" begin
        # An empty column has no mean and no median, and neither reduction throws.
        A = [NaN 1.0; NaN 2.0; NaN 4.0]
        @test isnan(PortfolioOptimisers.factor_summary_column_mean(A, 1))
        @test isnan(PortfolioOptimisers.factor_summary_column_median(A, 1))
        @test PortfolioOptimisers.factor_summary_column_mean(A, 2) ≈ 7 / 3
        @test PortfolioOptimisers.factor_summary_column_median(A, 2) == 2.0
        @test PortfolioOptimisers.factor_summary_finite_column(A, 1) == Float64[]
        # A ratio has no value on a zero, an absent or an overflowing denominator.
        @test isnan(PortfolioOptimisers.factor_summary_ratio(1.0, 0.0))
        @test isnan(PortfolioOptimisers.factor_summary_ratio(1.0, NaN))
        @test isnan(PortfolioOptimisers.factor_summary_ratio(Inf, 1.0))
        @test PortfolioOptimisers.factor_summary_ratio(3.0, 1.5) == 2.0
        # A series of one observation has no autocorrelation.
        @test all(isnan,
                  PortfolioOptimisers.factor_summary_autocorrelation(reshape([1.0], 1, 1)))
        # A series of one observation has no volatility either, and so no Sharpe ratio.
        r, v, s = PortfolioOptimisers.factor_summary_return_stats(reshape([0.5], 1, 1), 4)
        @test r == [2.0]
        @test isnan(v[1])
        @test isnan(s[1])
        # A cross-section with no finite exposure has no variance, and a factor with no
        # finite cross-section anywhere is not constant.
        E = fill(NaN, 2, 3, 1)
        @test isnan(PortfolioOptimisers.factor_summary_exposure_variance(E, 1, 1))
        @test PortfolioOptimisers.factor_summary_constant_exposures(E) == falses(1)
        @test PortfolioOptimisers.factor_summary_constant_exposures(ones(2, 3, 1)) ==
              trues(1)
        # A statistic of the reduced axis maps onto the raw axis, and a dropped factor
        # takes a `NaN`.
        @test summary_agrees(PortfolioOptimisers.factor_summary_mapped([5.0, 6.0],
                                                                       [2, 0, 1]),
                             [6.0, NaN, 5.0])
    end
end
