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
