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
Issue #1448, ruled on #1444. Under HAC the diagonal of a row, `x_t² + 2 Σ w_l x_t x_{t-l}`, can be
negative. A volatility that holds the row damps a positive row and amplifies a negative one, so
the diagonal of the correlation state is skewed down and falls on the guards of `update_var_cor!`
and `pair_weighted_correlation`. So under HAC the separate path divides each row by the volatility
before its update, and `hac_vol_before = VolatilityAfterUpdate()` keeps the rule after it. Without HAC the diagonal
is `x_t² ≥ 0`, and the rule after the update stays (#1440).
=#
@testset "under HAC the separate path divides each row by the volatility before its update" begin
    # The recursion by hand on a two-asset case of six rows and one lag.
    # Under the default `cor_vol` each variance is divided by its weight first (#1729).
    function by_hand(X, lam, lamc, before, corrected = true)
        v, Q, prev = zeros(2), zeros(2, 2), nothing
        for t in axes(X, 1)
            x = X[t, :]
            A = x * transpose(x)
            if !isnothing(prev)
                A += (x * transpose(prev) + prev * transpose(x)) / 2
            end
            s = before ? v : lam * v + (1 - lam) * diag(A)
            if corrected
                s = s / max(1 - lam^(before ? t - 1 : t), eps())
            end
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
        # The oracle's rule reads the raw variance state.
        raw = PO.regime_adjusted_covariance_pass!(rule(b; cor_vol = RawVolatility()), X, 1,
                                                  nothing, nothing)
        @test raw.cor_state ≈ by_hand(X, 0.8, 0.9, b, false)[2] rtol = 1e-14
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

#=
Issues #1548 and #1549, found by #1527. Under `EstimatedCentring()` the terms of the estimate
share their location, and the next deviation reads it too. `ExactDebias()` reads the law of the
estimate and the dependence of the deviation on it; `LawDebias()` reads the law alone (ADR 0190).
=#
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
    # #1724: at a small block and a small count the root-mean-squared factor was NaN, because a
    # central difference in `t` stepped far past the range where its columns are linear. The
    # complex step has no such range.
    f = PO.mahalanobis_level_bias(methods[1], lam, [6, 7, 30], 2, nothing,
                                  EstimatedCentring())
    @test isapprox(f, [2.01020, 1.76105, 1.14720]; rtol = 1e-5)
    # Under two lags at two assets the lattice marks a point dead by round-off where the exact
    # slope is still positive, and the cut fell on it: every factor at the first count was Inf.
    for m in methods, d in (ExactDebias(), LawDebias())
        @test all(isfinite,
                  PO.mahalanobis_level_bias(m, lam, [6, 7], 2, 2, EstimatedCentring(), d))
    end
    # Every exact count and every node of the interpolation is finite, up to eight assets.
    for (hl, lags, n) in
        [(10, nothing, 2), (10, nothing, 3), (10, nothing, 8), (10, 2, 2), (10, 2, 3),
         (10, 2, 8), (40, 2, 2)]
        nd = PO.mahalanobis_bias_nodes(methods[1], 2.0^(-1 / hl), n, lags,
                                       EstimatedCentring(), ExactDebias())
        @test all(isfinite, nd.exact) && all(isfinite, nd.ratio)
    end
    # The grid of #1724 (#1773). The round-off of the cause grows as the count of assets and
    # the half-life fall, so at a half-life of 5 every count of 2 to 12 assets, every method
    # and every debias rule is checked, with and without lags (52 s). At half-lives of 10 and
    # 40 the six pairs of a method and a rule rotate over the counts (61 s). The whole grid
    # took 417 s and every factor was finite.
    rules = [(m, d) for m in methods for d in (ExactDebias(), LawDebias())]
    for lags in (nothing, 2), n in 2:12, (m, d) in rules
        nd = PO.mahalanobis_bias_nodes(m, 2.0^(-1 / 5), n, lags, EstimatedCentring(), d)
        @test all(isfinite, nd.exact) && all(isfinite, nd.ratio)
    end
    for hl in (10, 40), lags in (nothing, 2), n in 2:12
        m, d = rules[mod1(n + (isnothing(lags) ? 0 : 3) + (hl == 40 ? 1 : 0), 6)]
        nd = PO.mahalanobis_bias_nodes(m, 2.0^(-1 / hl), n, lags, EstimatedCentring(), d)
        @test all(isfinite, nd.exact) && all(isfinite, nd.ratio)
    end
end
