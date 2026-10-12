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
