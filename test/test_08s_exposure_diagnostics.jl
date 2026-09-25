#=
The exposure group of the cross-sectional diagnostics, decided by #709 and built by #799.

The oracle is the reference implementation. The block of `# the reference oracle` below was
built from the very arrays this file builds, run in the reference implementation's own
environment, and its answers are written out as literals. Rebuild them by generating the
same fixture, exporting it, and running the reference's factor model block on it.
=#
using Statistics

@testset "Cross-sectional exposure diagnostics" begin
    # The fixture the reference oracle was measured on. Every mutation below drives one
    # branch, so a change to any of them invalidates the literals in `# the reference
    # oracle`.
    rng = StableRNG(24680)
    T, N, K = 10, 6, 4
    Ms = randn(rng, T, N, K)
    f = 0.02 * randn(rng, T, K)
    eps = 0.01 * randn(rng, T, N)
    rw = abs.(randn(rng, T, N)) .+ 0.1
    bw = fill(1 / N, T, N)
    vs = abs.(randn(rng, T, N)) .+ 0.5
    # The third factor is the global intercept. Its cross-section is constant, so every
    # pair that holds it is degenerate and reads `0` by convention.
    Ms[:, :, 3] .= 1.0
    # The fourth factor is finite on two assets alone, so no pair that holds it ever
    # reaches the three common assets an estimate needs, and every such pair reads `NaN`.
    Ms[:, 3:6, 4] .= NaN
    # An asset that carries no weight leaves the universe of that observation.
    rw[3, 2] = 0.0
    bw[7, 4] = 0.0
    # A missing exposure costs its own pair and not the observation.
    Ms[2, 4, 1] = NaN
    # An observation at which a factor keeps two finite assets is insufficient there alone.
    Ms[5, 1, 2] = NaN
    Ms[5, 2, 2] = NaN
    Ms[5, 3, 2] = NaN
    Ms[5, 4, 2] = NaN

    csr = CrossSectionalRegression(; f = f, eps = eps, n = fill(N, T))
    csfm = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = csr, Ms = Ms,
                                     vs = vs, rw = rw, bw = bw, lag = 1)
    # The fourth factor is not finite on four of the six assets, so the reconstructed asset
    # return of those assets is not finite either, and the information coefficient of every
    # factor would be `NaN`. It is read on the three dense factors of the same fixture.
    csr3 = CrossSectionalRegression(; f = f[:, 1:3], eps = eps, n = fill(N, T))
    csfm3 = CrossSectionalFactorModel(; M = Ms[T, :, 1:3], b = zeros(N), csr = csr3,
                                      Ms = Ms[:, :, 1:3], vs = vs, rw = rw, bw = bw,
                                      lag = 1)

    # A `NaN` compares unequal to itself, so the pattern of the absent answers is asserted
    # separately from the values.
    function exposure_agrees(a, b; rtol = 1e-10)
        if isnan.(a) != isnan.(b)
            return false
        end
        m = .!isnan.(a)
        return isapprox(a[m], b[m]; rtol = rtol)
    end

    @testset "the reference oracle" begin
        ref_corr_bw = [ 1.0 -0.11910774310012738 0.0 NaN;
                       -0.11910774310012745 1.0 0.0 NaN;
                       0.0 0.0 1.0 NaN;
                       NaN NaN NaN 1.0]
        ref_corr_id = [ 1.0 -0.06318259646864377 0.0 NaN;
                       -0.06318259646864377 1.0 0.0 NaN;
                       0.0 0.0 1.0 NaN;
                       NaN NaN NaN 1.0]
        ref_corr_rw = [ 1.0 -0.12423074669435247 0.0 NaN;
                       -0.12423074669435247 1.0 0.0 NaN;
                       0.0 0.0 1.0 NaN;
                       NaN NaN NaN 1.0]
        ref_ic_spearman = [ 0.2571428571428571 0.8285714285714286 0.08571428571428572;
                           1.0 0.4 -0.6;
                           0.2 -0.6571428571428571 -0.08571428571428572;
                           1.0 0.14285714285714285 -0.42857142857142855;
                           NaN NaN NaN;
                           0.2 0.7142857142857143 -0.08571428571428572;
                           -0.8857142857142857 0.14285714285714285 -0.14285714285714285;
                           -0.6571428571428571 0.02857142857142857 -0.14285714285714285;
                           -0.6571428571428571 0.3142857142857143 -0.4857142857142857]
        ref_ic_pearson = [ 0.017660125172323477 0.8554536111961 NaN;
                          0.9704452550730513 0.3235269729843106 NaN;
                          0.6358545350022913 -0.7769061129248623 NaN;
                          0.987622896128322 0.021171816164635107 NaN;
                          NaN NaN NaN;
                          -0.14622469419159947 0.5839881757529729 NaN;
                          -0.7618984163053351 0.6376018710833449 NaN;
                          -0.4310263691775938 0.12206610944522596 NaN;
                          -0.6113039548526288 0.5481220484702234 NaN]
        ref_ic_h2 = [ 0.14285714285714285 0.8285714285714286 -0.6571428571428571;
                     0.2 0.9428571428571428 -0.3142857142857143;
                     -0.14285714285714285 -0.2 -0.42857142857142855;
                     1.0 0.14285714285714285 -0.42857142857142855;
                     -0.14285714285714285 NaN -0.2571428571428571;
                     0.6571428571428571 -0.6 -0.37142857142857144;
                     -0.7714285714285715 0.37142857142857144 -0.02857142857142857;
                     -0.2 0.2571428571428571 -0.37142857142857144]
        ref_mean_ic = [0.057142857142857204, 0.2392857142857143, -0.23571428571428568]
        ref_std_ic = [0.733695466730492, 0.4584006929917067, 0.23843513154179433]
        ref_ic_ir = [0.07788361756887816, 0.5220012053734906, -0.9885887377002087]
        # The reference's exposure summary reads `nanmean(ic > 0)`, and the comparison turns
        # a `NaN` into `false` before the mean sees it, so it counts the one silenced date of
        # nine as a miss: 5/9, 7/9, 1/9. Its evaluation summary drops a `NaN`, as every
        # summary of this library does, so the port reads the eight dates that carried a
        # coefficient and this is a deliberate divergence, not a missed one (2026-09-14).
        ref_hit_rate = [0.5555555555555556, 0.7777777777777778, 0.1111111111111111]
        port_hit_rate = [0.625, 0.875, 0.125]
        ref_stability = [ -0.5461756157848291 0.13772753623689454 NaN NaN;
                         -0.4619931721037595 0.7008326890567194 NaN NaN;
                         -0.8047640404112031 NaN NaN NaN;
                         -0.6946808326022188 0.01790866776878912 NaN NaN;
                         0.0779317550879532 NaN NaN NaN;
                         0.42596729822258317 0.004471339771113092 NaN NaN;
                         -0.5135794708889005 -0.03310005030782944 NaN NaN;
                         -0.5065206111895351 0.9103284212148111 NaN NaN]
        ref_dispersion = [ 0.8727358662984228 0.6268674463921016 2.220446049250313e-16 0.5606120216910173;
                          0.7351065923556201 0.8648558927554777 2.220446049250313e-16 1.0423815034982549;
                          0.7448846636964156 0.6002230796625391 2.220446049250313e-16 1.1554390259263865;
                          1.7166240612882144 0.8779819496662131 2.220446049250313e-16 0.05788537538701428;
                          0.9003385783471725 0.04365284049156737 2.220446049250313e-16 0.019149574064470987;
                          1.4050487363573019 0.9902060412128743 2.220446049250313e-16 0.3669213565103253;
                          0.7763831805507717 0.543290572945996 0.0 0.2348554998332349;
                          1.089040170020986 0.4291012199829881 2.220446049250313e-16 0.18499707402029192;
                          0.36515357013408956 0.4882494000681807 2.220446049250313e-16 0.2326031925443415;
                          0.6332267854890944 0.3970126344357655 2.220446049250313e-16 0.01635061891197087]
        ref_coverage = [0.9833333333333334, 0.9333333333333332, 1.0, 0.32]

        @test exposure_agrees(exposure_correlation(csfm), ref_corr_bw)
        @test exposure_agrees(exposure_correlation(csfm; weighting = IdentityMetric()),
                              ref_corr_id)
        @test exposure_agrees(exposure_correlation(csfm;
                                                   weighting = RegressionWeightMetric()),
                              ref_corr_rw)
        # The third factor is the constant intercept, so every asset of it is in one tie. The
        # reference breaks a tie by its sort, which on this fixture is the order of the asset
        # axis, and it scores a rank coefficient there. `ties = :ordinal` reproduces it.
        @test exposure_agrees(exposure_ic(csfm3; ties = :ordinal), ref_ic_spearman)
        @test exposure_agrees(exposure_ic(csfm3; rank = false), ref_ic_pearson)
        @test exposure_agrees(exposure_ic(csfm3; horizon = 2, ties = :ordinal), ref_ic_h2)
        # Under the default `ties = :average`, a constant exposure has constant ranks, so it
        # has no rank coefficient, as it has no Pearson one. The two factors with no tie keep
        # the reference's numbers.
        ica = exposure_ic(csfm3)
        @test all(isnan, ica[:, 3])
        @test exposure_agrees(ica[:, 1:2], ref_ic_spearman[:, 1:2])
        ica2 = exposure_ic(csfm3; horizon = 2)
        @test all(isnan, ica2[:, 3])
        @test exposure_agrees(ica2[:, 1:2], ref_ic_h2[:, 1:2])
        @test isequal(exposure_ic_summary(csfm3).mean_ic[1:2],
                      exposure_ic_summary(csfm3; ties = :ordinal).mean_ic[1:2])
        @test isnan(exposure_ic_summary(csfm3).mean_ic[3])
        s = exposure_ic_summary(csfm3; ties = :ordinal)
        @test exposure_agrees(s.mean_ic, ref_mean_ic)
        @test exposure_agrees(s.std_ic, ref_std_ic)
        @test exposure_agrees(s.ic_ir, ref_ic_ir)
        @test exposure_agrees(s.hit_rate, port_hit_rate)
        @test count(isfinite, exposure_ic(csfm3)[:, 1]) == 8
        @test size(exposure_ic(csfm3), 1) == 9
        @test exposure_agrees(port_hit_rate .* (8 / 9), ref_hit_rate)
        @test exposure_agrees(exposure_stability(csfm; step = 2), ref_stability)
        @test exposure_agrees(exposure_dispersion(csfm), ref_dispersion)
        # The reference's summary reads the regression weights for the universe of the
        # coverage, where this verb reads the weighting the caller names. The benchmark
        # weighting is the default, so the reference's answer is the one under
        # `RegressionWeightMetric()`.
        @test exposure_agrees(exposure_coverage(csfm; weighting = RegressionWeightMetric()),
                              ref_coverage)
        @test exposure_coverage(csfm)[4] == 0.34
    end

    @testset "the answers carry the axes the verbs state" begin
        @test size(exposure_correlation(csfm)) == (K, K)
        @test size(exposure_ic(csfm3)) == (T - 1, 3)
        @test size(exposure_ic(csfm3; horizon = 3)) == (T - 3, 3)
        @test size(exposure_stability(csfm; step = 4)) == (T - 4, K)
        @test size(exposure_dispersion(csfm)) == (T, K)
        @test length(exposure_coverage(csfm)) == K
        for v in exposure_ic_summary(csfm3)
            @test length(v) == 3
        end
    end

    @testset "the block summary reads the overlap of its forward window" begin
        # `exposure_ic` scores every observation, so a window of `h` observations overlaps
        # its `h - 1` neighbours and the block method hands `h - 1` lags to the kernel. At
        # the default window there is no overlap and the bare method's default agrees.
        ic2 = exposure_ic(csfm3; horizon = 2)
        s2 = exposure_ic_summary(csfm3; horizon = 2)
        # The constant third factor has no rank coefficient, so its column is `NaN`, and
        # the comparisons read `isequal`.
        @test isequal(s2.t_stat, exposure_ic_summary(ic2; lags = 1).t_stat)
        @test !isequal(s2.t_stat, exposure_ic_summary(ic2).t_stat)
        @test isequal(s2.mean_ic, exposure_ic_summary(ic2).mean_ic)
        @test isequal(s2.ic_ir, exposure_ic_summary(ic2).ic_ir)
        @test isequal(exposure_ic_summary(csfm3).t_stat,
                      exposure_ic_summary(exposure_ic(csfm3)).t_stat)
        @test_throws DomainError exposure_ic_summary(ic2; lags = -1)
    end

    @testset "the correlation of a constant exposure and of a sparse one" begin
        # The third factor is constant across the assets, so its correlation against any
        # other is `0` by convention. The fourth is never finite on three common assets,
        # so its correlation is `NaN`. The diagonal is `1` in both cases.
        C = exposure_correlation(csfm)
        @test all(isone, LinearAlgebra.diag(C))
        @test C[1, 3] == C[3, 1] == 0
        @test C[2, 3] == C[3, 2] == 0
        @test all(isnan, C[1:3, 4])
        @test all(isnan, C[4, 1:3])
        @test exposure_agrees(C, Matrix(transpose(C)))
    end

    @testset "the information coefficient of a perfect forecast is one" begin
        rng2 = StableRNG(13571113)
        Ti, Ni = 5, 6
        R = randn(rng2, Ti, Ni)
        B = Array{Float64, 3}(undef, Ti, Ni, 1)
        # The exposure of an observation is the return of the observation that follows it,
        # so the exposure forecasts the return without error.
        for t in 1:(Ti - 1)
            B[t, :, 1] = R[t + 1, :]
        end
        B[Ti, :, 1] = R[1, :]
        @test isapprox(exposure_ic(B, R)[:, 1], ones(Ti - 1); rtol = 1e-12)
        @test isapprox(exposure_ic(B, R; rank = false)[:, 1], ones(Ti - 1); rtol = 1e-12)
        # The forward mean return of a horizon of one observation is the next return.
        @test PortfolioOptimisers.exposure_forward_mean_return(R, 1) == R[2:Ti, :]
        # A window with no finite return has no mean.
        Rn = copy(R)
        Rn[2, :] .= NaN
        @test all(isnan, PortfolioOptimisers.exposure_forward_mean_return(Rn, 1)[1, :])
        @test_throws DimensionMismatch PortfolioOptimisers.exposure_forward_mean_return(R,
                                                                                        Ti)
    end

    @testset "the stability of a constant exposure and of a reshuffled one" begin
        rng3 = StableRNG(97531)
        Ts, Ns = 4, 8
        B = Array{Float64, 3}(undef, Ts, Ns, 3)
        col = randn(rng3, Ns)
        for t in 1:Ts
            # The first factor keeps its ordering, so its stability is `1`.
            B[t, :, 1] = col
            # The second factor is constant across the assets, so it has no correlation and
            # its stability is `NaN`. The summary of #709 is what reads such a factor as
            # `1`.
            B[t, :, 2] .= 2.0
            # The third factor is drawn afresh at each observation, so its stability is the
            # correlation of two independent cross-sections.
            B[t, :, 3] = randn(rng3, Ns)
        end
        S = exposure_stability(B; step = 1)
        @test size(S) == (Ts - 1, 3)
        @test all(isapprox(1.0), S[:, 1])
        @test all(isnan, S[:, 2])
        @test all(abs.(S[:, 3]) .< 1)
        @test_throws DomainError exposure_stability(B; step = 0)
        @test_throws DimensionMismatch exposure_stability(B; step = Ts)
    end

    @testset "the dispersion of a standardised exposure is one" begin
        rng4 = StableRNG(2468013)
        Td, Nd = 3, 7
        B = Array{Float64, 3}(undef, Td, Nd, 2)
        for t in 1:Td
            x = randn(rng4, Nd)
            # A cross-section of zero mean and unit population standard deviation has a
            # dispersion of exactly one under equal weights.
            B[t, :, 1] = (x .- Statistics.mean(x)) ./ Statistics.std(x; corrected = false)
            B[t, :, 2] = 3.0 * B[t, :, 1]
        end
        D = exposure_dispersion(B)
        @test isapprox(D[:, 1], ones(Td); rtol = 1e-12)
        @test isapprox(D[:, 2], fill(3.0, Td); rtol = 1e-12)
        # A factor whose whole cross-section is absent has no dispersion.
        Bn = copy(B)
        Bn[1, :, 1] .= NaN
        @test isnan(exposure_dispersion(Bn)[1, 1])
        @test !isnan(exposure_dispersion(Bn)[1, 2])
    end

    @testset "the coverage counts the universe of each observation" begin
        B = Array{Float64, 3}(undef, 2, 4, 2)
        B[:, :, 1] .= 1.0
        B[:, :, 2] .= 1.0
        # One asset of the universe carries no exposure, so the coverage is `1 - 1/n`.
        B[1, 2, 2] = NaN
        @test exposure_coverage(B) == [1.0, 1 - 1 / 8]
        # An asset outside the universe counts in neither the numerator nor the
        # denominator, so a missing exposure there costs no coverage.
        w = ones(2, 4)
        w[1, 2] = 0.0
        @test exposure_coverage(B, w) == [1.0, 1.0]
        # An observation whose universe is empty contributes nothing.
        w0 = ones(2, 4)
        w0[1, :] .= 0.0
        @test exposure_coverage(B, w0) == [0.5, 0.5]
    end

    @testset "a weighting member reads the field it names" begin
        # Each member reads one field of the block over the whole observation axis.
        @test PortfolioOptimisers.cs_diagnostic_weights(BenchmarkWeightMetric(), csfm) ===
              bw
        @test PortfolioOptimisers.cs_diagnostic_weights(RegressionWeightMetric(), csfm) ===
              rw
        @test PortfolioOptimisers.cs_diagnostic_weights(InverseIdiosyncraticVarianceMetric(),
                                                        csfm) == 1 ./ vs
        @test isnothing(PortfolioOptimisers.cs_diagnostic_weights(IdentityMetric(), csfm))
        # The answer under a member is the answer under the history it names.
        @test exposure_agrees(exposure_dispersion(csfm;
                                                  weighting = RegressionWeightMetric()),
                              exposure_dispersion(Ms, rw))
        @test exposure_agrees(exposure_stability(csfm; step = 2,
                                                 weighting = IdentityMetric()),
                              exposure_stability(Ms; step = 2))
        # A block that carries no such field refuses, and the refusal names the field.
        bare = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = csr,
                                         Ms = Ms, lag = 1)
        for metric in (BenchmarkWeightMetric(), RegressionWeightMetric(),
                       InverseIdiosyncraticVarianceMetric())
            @test_throws PortfolioOptimisers.IsNothingError exposure_correlation(bare;
                                                                                 weighting = metric)
        end
        @test_throws PortfolioOptimisers.IsNothingError exposure_stability(bare; step = 2)
        @test_throws PortfolioOptimisers.IsNothingError exposure_dispersion(bare)
        @test_throws PortfolioOptimisers.IsNothingError exposure_coverage(bare)
    end

    @testset "a block that carries no history refuses by name" begin
        empty_block = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N))
        @test_throws PortfolioOptimisers.IsNothingError exposure_correlation(empty_block;
                                                                             weighting = IdentityMetric())
        @test_throws PortfolioOptimisers.IsNothingError exposure_ic(empty_block)
        no_fit = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), Ms = Ms,
                                           lag = 1)
        @test_throws PortfolioOptimisers.IsNothingError exposure_ic(no_fit)
        @test_throws PortfolioOptimisers.IsNothingError exposure_ic_summary(no_fit)
        # The exposure history and the fit must carry the same axes.
        short = CrossSectionalRegression(; f = f[1:(T - 1), :], eps = eps[1:(T - 1), :],
                                         n = fill(N, T - 1))
        mismatched = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = short,
                                               Ms = Ms, lag = 1)
        @test_throws DimensionMismatch exposure_ic(mismatched)
        long_lag = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = csr,
                                             Ms = Ms, lag = T)
        @test_throws DimensionMismatch exposure_ic(long_lag)
    end

    @testset "the level-1 verbs check their arguments" begin
        B = randn(StableRNG(5), 3, 4, 2)
        @test_throws DimensionMismatch exposure_dispersion(B, ones(2, 4))
        @test_throws DomainError exposure_dispersion(B, fill(-1.0, 3, 4))
        @test_throws PortfolioOptimisers.IsEmptyError exposure_dispersion(zeros(0, 0, 0))
        @test_throws DimensionMismatch exposure_ic(B, randn(StableRNG(6), 2, 4))
        @test_throws DomainError exposure_ic(B, randn(StableRNG(6), 3, 4); horizon = 0)
        @test_throws DimensionMismatch exposure_ic(B, randn(StableRNG(6), 3, 4);
                                                   horizon = 3)
        @test_throws PortfolioOptimisers.IsEmptyError exposure_ic_summary(zeros(0, 0))
        # A history of ones is what an absent weight history resolves to.
        @test PortfolioOptimisers.exposure_weights(B, nothing) == ones(3, 4)
        @test PortfolioOptimisers.exposure_weights(B, ones(3, 4)) == ones(3, 4)
    end

    @testset "the correlation kernels" begin
        # The weighted correlation of two cross-sections, and the cases with no answer.
        a = [1.0, 2.0, 3.0, 4.0]
        b = [2.0, 4.0, 6.0, 8.0]
        u = ones(4)
        @test PortfolioOptimisers.cs_weighted_correlation(a, b, u) ≈ 1.0
        @test PortfolioOptimisers.cs_weighted_correlation(a, -b, u) ≈ -1.0
        # Fewer than three assets enter, so there is no answer.
        @test isnan(PortfolioOptimisers.cs_weighted_correlation(a, b, [1.0, 1.0, 0.0, 0.0]))
        # One cross-section is constant, so the denominator vanishes.
        @test isnan(PortfolioOptimisers.cs_weighted_correlation(a, ones(4), u))
        # A weight that is not finite excludes the asset.
        @test isnan(PortfolioOptimisers.cs_weighted_correlation(a, b, [1.0, 1.0, NaN, NaN]))
        # The rank correlation reads the order and not the level, so a monotone map of one
        # cross-section leaves it unchanged.
        @test PortfolioOptimisers.cs_spearman_correlation(a, exp.(b)) ≈ 1.0
        @test PortfolioOptimisers.cs_spearman_correlation(a, [1.0, 2.0, NaN, 4.0]) ≈ 1.0
        # Under the default, equal values share the mean of their positions, so a constant
        # cross-section has constant ranks and no rank correlation (#1332).
        ra = PortfolioOptimisers.cs_ranks([1.0, 1.0, 1.0], fill(true, 3), :average)
        @test ra == [2.0, 2.0, 2.0]
        @test isnan(PortfolioOptimisers.cs_spearman_correlation(ones(5), collect(1.0:5.0)))
        @test isnan(PortfolioOptimisers.cs_spearman_correlation(ones(5),
                                                                collect(5.0:-1:1.0)))
        # Under `:ordinal`, a tie takes the order of the asset axis, so the ranks are a
        # permutation, and the asset order sets the answer of a constant cross-section.
        ro = PortfolioOptimisers.cs_ranks([1.0, 1.0, 1.0], fill(true, 3), :ordinal)
        @test ro == [1.0, 2.0, 3.0]
        @test PortfolioOptimisers.cs_spearman_correlation(ones(5), collect(1.0:5.0);
                                                          ties = :ordinal) ≈ 1.0
        @test PortfolioOptimisers.cs_spearman_correlation(ones(5), collect(5.0:-1:1.0);
                                                          ties = :ordinal) ≈ -1.0
        # Two targets that differ only inside the ties of the first cross-section get one
        # answer under the default, and two under `:ordinal`.
        tb = [1.0, 1.0, 2.0, 2.0]
        for (yb, ordv) in (([1.0, 2.0, 3.0, 4.0], 1.0), ([2.0, 1.0, 4.0, 3.0], 0.6))
            @test PortfolioOptimisers.cs_spearman_correlation(tb, yb) ≈ 4 / sqrt(20)
            @test PortfolioOptimisers.cs_spearman_correlation(tb, yb; ties = :ordinal) ≈
                  ordv
        end
        # A run of two and a run of three, with a masked entry sorted to the end.
        rm = PortfolioOptimisers.cs_ranks([2.0, 1.0, 2.0, Inf, 2.0, 1.0],
                                          [true, true, true, false, true, true], :average)
        @test rm[[1, 2, 3, 5, 6]] == [4.0, 1.5, 4.0, 4.0, 1.5]
        @test isnan(rm[4])
        # The two rules give the same ranks to a cross-section with no tie.
        xr = randn(StableRNG(1332), 20)
        @test PortfolioOptimisers.cs_ranks(xr, trues(20), :average) ==
              PortfolioOptimisers.cs_ranks(xr, trues(20), :ordinal)
        for tr in (:average, :ordinal)
            rn = PortfolioOptimisers.cs_ranks([1.0, 2.0, Inf], [true, true, false], tr)
            @test rn[1:2] == [1.0, 2.0]
            @test isnan(rn[3])
        end
        @test_throws PortfolioOptimisers.ConflictingArgumentError PortfolioOptimisers.cs_ranks([1.0],
                                                                                               [true],
                                                                                               :dense)
        @test_throws PortfolioOptimisers.ConflictingArgumentError PortfolioOptimisers.cs_spearman_correlation(a,
                                                                                                              b;
                                                                                                              ties = :min)
    end

    @testset "a re-based block answers on the reduced axis when it is asked to" begin
        rng6 = StableRNG(864213579)
        Tr, Nr, Kr = 6, 5, 3
        Msr = randn(rng6, Tr, Nr, Kr)
        fr = 0.02 * randn(rng6, Tr, Kr)
        epsr = 0.01 * randn(rng6, Tr, Nr)
        rwr = abs.(randn(rng6, Tr, Nr)) .+ 0.1
        fcb = FactorFamilyBasis(; fnm = ["industry"], fi = [[1, 2]], di = [2],
                                ratios = reshape(collect(range(0.4, 0.9; length = Tr)), Tr,
                                                 1), K = Kr)
        csr_r = CrossSectionalRegression(; f = fr, eps = epsr, n = fill(Nr, Tr))
        L = PortfolioOptimisers.reduce_loadings(fcb, Msr[Tr, :, :])
        blk = CrossSectionalFactorModel(; M = Msr[Tr, :, :], L = L, b = zeros(Nr),
                                        csr = csr_r, Ms = Msr, rw = rwr, fcb = fcb, lag = 1)
        Kred = PortfolioOptimisers.reduced_factor_count(fcb)
        @test Kred == 2
        # The exposure group answers on the raw axis, because it reads the exposure history
        # and never the design of the fit.
        @test size(exposure_correlation(blk; weighting = IdentityMetric())) == (Kr, Kr)
        @test size(exposure_dispersion(blk; weighting = IdentityMetric())) == (Tr, Kr)
        @test size(exposure_ic(blk)) == (Tr - 1, Kr)
        # `reduced` is the one mode that maps the exposures through the re-basis first.
        @test size(exposure_ic(blk; reduced = true)) == (Tr - 1, Kred)
        @test size(exposure_ic_summary(blk; reduced = true).mean_ic) == (Kred,)
        # A block that carries no re-basis has one axis, whatever `reduced` states.
        @test exposure_agrees(exposure_ic(csfm3; reduced = true), exposure_ic(csfm3))
    end
end
