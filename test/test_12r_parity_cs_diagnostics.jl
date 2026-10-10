#=
Parity of the cross-sectional diagnostics and the factor model summary on the block of a real fit
(#1387, map #1375).

Every stored case under `test/assets/Parity_CrossSectionalFactorModel_Diag<Panel><Case>_<Output>.csv.gz`
is the oracle's diagnostics of its own fit of a configuration of `parity_grid.jl`, on the exchange
`parity_write(dir, grid_fixture(fx); filled = true)` writes. The fit of each configuration is at
parity (#1385), so each case measures the diagnostics of two blocks that agree. The panels list
late, delist, list again, skip a holiday and keep an asset outside the estimation universe, and the
cases carry an exposure lag of two, a constrained family with a level that empties, two constrained
families, the Currency Factors and an observed factor.

The oracle ranks a tie in the order of an unstable sort, which no rule states. Its rank is replaced
by a stable one, which `ties = :ordinal` reproduces, and a case whose name ends in `Average` replaces
it by the midrank, which is the default `ties = :average`.

A group of outputs is stored side by side in one file:

  - `Regression`: the t-statistics, the variance inflation factors, the condition number and the
    four scores `r2`, `adjusted_r2`, `aic`, `bic`. `TRate`: the exceedance rate at the thresholds
    `2` and `1.5`, for the cases without a constrained family. In a family case a singular row
    changes the rate (#1421), and the factor model summary below checks that difference.
    `Gram`: the Gram of `DiagSmallBase`, one row per observation.
  - `Correlation`, `Stability`, `Dispersion`: the benchmark, identity and regression weightings.
    `Stability5`: the stability at a step of five.
  - `IC1`, `IC3`: the rank and the linear coefficient at the horizons one and three. `ICSummary`:
    their mean, deviation, ratio and hit rate, in that order. `ICRed`: the same on the reduced axis.
  - `Summary`: the nine columns of the summary at `ppy = 252`, and again at `threshold = 1.5` and
    `step = 5`.
  - `Idio`: the calibration, the tail rates at three and two, the excess kurtosis and the skewness,
    with each return divided by the volatility of its own observation. `IdioDependence`: the
    information coefficient and the residual dependence of the volatility. `AheadIdio`: the oracle's
    own kernels on each return divided by the volatility of the previous observation.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))
include(joinpath(@__DIR__, "parity_grid.jl"))

@testset "The cross-sectional diagnostics on a fitted block, at parity (#1387)" begin
    PO = PortfolioOptimisers
    U = "CrossSectionalFactorModel"
    load(c, o) = parity_load(U, c, o)
    loadv(c, o) = vec(load(c, o))
    pc(a, b, name; rtol = 1e-12, scale = :cell) = parity_compare(a, b; rtol = rtol,
                                                                 scale = scale, name = name).ok
    fxs = parity_small_panel()
    fxl = parity_large_panel()
    rds = grid_fixture(fxs)
    rdl = grid_fixture(fxl)
    blocks = Dict{String, Any}()
    function block(fix, c)
        return get!(blocks, "$(fix)$(c)") do
            rd = fix == "Small" ? rds : rdl
            return prior(grid_prior(c, rd), rd).rr
        end
    end
    fam = ("FamOne", "FamTwo")
    cases = [("Small", "Base"), ("Small", "Lag2"), ("Small", "FamOne"), ("Small", "FamTwo"),
             ("Small", "Currency"), ("Small", "Macro"), ("Large", "Base"),
             ("Large", "FamOne")]
    weightings = (BenchmarkWeightMetric(), IdentityMetric(), RegressionWeightMetric())

    @testset "The regression group, $(fix) $(c)" for (fix, c) in cases
        csfm = block(fix, c)
        n = "Diag$(fix)$(c)"
        R = load(n, "Regression")
        t = cs_regression_t_stats(csfm).X
        K = size(t, 2)
        kap = exposure_condition_number(csfm)
        okap = R[:, 2K + 1]
        # The constrained family's level that empties makes rows whose design is singular,
        # and both sides flag them. Such a row identifies only the coefficients whose unit
        # vector lies in the row space of its Gram. The oracle's t-statistic of another
        # coefficient is a ratio of round-off (7.87 for a level with no member), and it
        # divides every residual variance by `n - K`. Ours is `NaN` for an unidentified
        # coefficient, its variance inflation factor is `Inf` (`R_k^2 = 1`) or `NaN` (a
        # zero column), and the unbiased variance divides by `n - rank` (#1421), Better.
        idf = (kap .< 1e12) .& (okap .< 1e12)
        @test all(>=(1e12), kap[.!idf]) && all(>=(1e12), okap[.!idf])
        @test c in fam || all(idf)
        # A t-statistic near zero carries the relative round-off of the fit, so the check
        # reads the array scale. Measured maxrel 1.6e-12 on the large panel on one host, 2.1e-12
        # on the CI host, and maxscaled 7.4e-15 on both.
        @test pc(t[idf, :], R[idf, 1:K], "$(n) t"; scale = :array)
        vif = exposure_vif(csfm).X
        @test pc(vif[idf, :], R[idf, (K + 1):(2K)], "$(n) vif")
        @test pc(kap[idf], okap[idf], "$(n) kappa")
        # An identified coefficient has one variance whatever basis of the column space the
        # fit reads, so on a singular row its t-statistic equals that of the fit on a
        # pivoted basis of the weighted design, which has full rank. Every coefficient
        # outside the basis is unidentified. Measured maxrel 3.3e-15 over 233 coefficients.
        d = PO.cs_regression_data(csfm)
        mask, u = PO.cs_diagnostic_mask_weights(d.B, d.eps, d.w)
        for r in findall(.!idf)
            A = sqrt.(u[r, mask[r, :]]) .* d.B[r, mask[r, :], :]
            kb = sort(qr(A, ColumnNorm()).p[1:rank(A)])
            wr = isnothing(d.w) ? nothing : d.w[r:r, :]
            tb = vec(cs_regression_t_stats(d.B[r:r, :, kb], d.f[r:r, kb], d.eps[r:r, :],
                                           wr))
            ib = .!isnan.(t[r, kb])
            @test any(ib) && pc(t[r, kb][ib], tb[ib], "$(n) basis t $(r)"; rtol = 1e-13)
            nb = setdiff(1:K, kb)
            @test all(isnan, t[r, nb]) && all(v -> isinf(v) || isnan(v), vif[r, nb])
            @test all(v -> isnan(v) || v >= 1, vif[r, :])
        end
        # `1 - rss / tss` of a small score loses digits. Measured maxrel 2.5e-12 at a score
        # near 0.02 under the exposure lag of two, and maxscaled 6.5e-16.
        @test pc(cs_regression_r2(csfm), R[:, 2K + 2], "$(n) r2"; rtol = 3e-12)
        # A score charges the rank of each row, which the oracle's column count `k = K`
        # exceeds on a singular row. So the default is at parity on the rows of full rank,
        # `k = K` is at parity on every row, and on a singular row the AIC is the oracle's
        # less `2 (K - rank)`. Measured maxrel 4.3e-14 on the adjusted score.
        rk = PO.cs_score_regressors(nothing, d.B,
                                    PO.cs_regression_score_parts(d.B, d.f, d.eps, d.w)[4])
        @test all(==(K), rk[idf]) && all(<(K), rk[.!idf])
        aic = cs_regression_aic(csfm)
        @test pc(aic[.!idf], R[.!idf, 2K + 4] .- 2 .* (K .- rk[.!idf]), "$(n) aic rank")
        for (j, verb) in ((3, cs_regression_adjusted_r2), (4, cs_regression_aic),
                          (5, cs_regression_bic))
            @test pc(verb(csfm)[idf], R[idf, 2K + j], "$(n) score $(j)")
            @test pc(verb(csfm; k = K), R[:, 2K + j], "$(n) score $(j) k")
        end
        if !(c in fam)
            tr = load(n, "TRate")
            @test pc(cs_regression_t_stat_exceedance_rate(csfm).X, tr[:, 1], "$(n) rate")
            @test pc(cs_regression_t_stat_exceedance_rate(csfm; threshold = 1.5).X,
                     tr[:, 2], "$(n) rate 1.5")
        end
    end

    @testset "The Gram of the regression group" begin
        # A cross-product of a centred exposure comes from a cancellation, so the Gram
        # compares against its largest entry. Measured maxscaled 4.1e-16, and maxrel 1.1e-10
        # cell by cell on the market-by-style entries.
        csfm = block("Small", "Base")
        d = PO.cs_regression_data(csfm)
        u = ifelse.(isfinite.(d.eps), d.w, 0.0)
        G = cs_gram(d.B, u)
        K = size(G, 2)
        @test pc(reshape(permutedims(G, (1, 3, 2)), size(G, 1), K * K),
                 load("DiagSmallBase", "Gram"), "gram"; scale = :array)
    end

    @testset "The exposure group, $(fix) $(c)" for (fix, c) in cases
        csfm = block(fix, c)
        n = "Diag$(fix)$(c)"
        B = csfm.Ms
        K = size(B, 3)
        # The oracle's benchmark weight needs a finite return. Its observed-factor path reads
        # the returns net of the observed factor, `X_t - z_(t-1) m_t`, which the stored case
        # builds for it, and that return is `NaN` at the first active observation of a listing,
        # where `z_(t-1)` is absent. So the oracle's weight is zero on two cells where ours is
        # the capitalisation weight of an asset whose base-currency return is finite. The
        # stored `BenchmarkWeightChanges` holds those cells, and every benchmark-weighted
        # output of this case reads the oracle's weights through the verb's array form, which
        # measures the kernel alone.
        bw = csfm.bw
        if c == "Macro"
            ch = load(n, "BenchmarkWeightChanges")
            @test size(ch, 1) == 2 && all(iszero, ch[:, 3])
            bw = copy(bw)
            for r in eachrow(ch)
                bw[Int(r[1]), Int(r[2])] = r[3]
            end
        end
        C = load(n, "Correlation")
        S = load(n, "Stability")
        D = load(n, "Dispersion")
        for (j, (wn, wt)) in
            enumerate(zip(("bench", "identity", "regression"), weightings))
            u = j == 1 ? bw : PO.cs_diagnostic_weights(wt, csfm)
            cols = ((j - 1) * K + 1):(j * K)
            @test pc(exposure_correlation(B, u), C[cols, :], "$(n) correlation $(wn)")
            @test pc(exposure_stability(B, u), S[:, cols], "$(n) stability $(wn)")
            # A constant exposure has a dispersion of exactly zero: the mean of the entering
            # values is clamped to their range, so the deviations are zero. The oracle reads
            # the round-off of the mean, at most 4.5e-16. Every other entry is at parity.
            dp = exposure_dispersion(B, u)
            od = D[:, cols]
            z = dp .== 0
            @test all(<=(1e-15), abs.(od[z]))
            @test pc(dp[.!z], od[.!z], "$(n) dispersion $(wn)")
            if j == 1
                @test pc(exposure_stability(B, u; step = 5), load(n, "Stability5"),
                         "$(n) stability 5")
            end
        end
        # The verbs over the block read the block's own weights through the same kernels.
        u = PO.cs_diagnostic_weights(BenchmarkWeightMetric(), csfm)
        @test isequal(exposure_correlation(csfm).X, exposure_correlation(B, u))
        @test isequal(exposure_stability(csfm).X, exposure_stability(B, u))
        @test isequal(exposure_dispersion(csfm).X, exposure_dispersion(B, u))
    end

    @testset "The exposure information coefficient, $(fix) $(c)" for (fix, c) in cases
        csfm = block(fix, c)
        n = "Diag$(fix)$(c)"
        K = size(csfm.Ms, 3)
        Q = load(n, "ICSummary")
        for (h, o) in ((1, load(n, "IC1")), (3, load(n, "IC3"))),
            (j, rank) in enumerate((true, false))

            ic = exposure_ic(csfm; horizon = h, rank = rank, ties = :ordinal).X
            # A linear coefficient near zero carries the round-off of the fit's returns.
            # Measured maxrel 1.4e-12 on the large panel, and maxscaled 7.4e-16.
            @test pc(ic, o[:, ((j - 1) * K + 1):(j * K)], "$(n) ic $(h) $(rank)";
                     rtol = 1.5e-12)
            s = exposure_ic_summary(csfm; horizon = h, rank = rank, ties = :ordinal)
            q = Q[:, (8 * (h == 3) + 4 * (j - 1) + 1):(8 * (h == 3) + 4 * j)]
            @test pc(s.mean_ic, q[:, 1], "$(n) mean ic $(h) $(rank)")
            @test pc(s.std_ic, q[:, 2], "$(n) std ic $(h) $(rank)")
            @test pc(s.ic_ir, q[:, 3], "$(n) ic ir $(h) $(rank)")
            # The oracle counts an observation whose coefficient is `NaN` as a miss, so its hit
            # rate is a share of every observation. Ours is a share of the observations that
            # have a coefficient (ADR 0149). The two agree on a factor with no `NaN`.
            P = size(ic, 1)
            @test q[:, 4] == [count(>(0), view(ic, :, k)) / P for k in 1:K]
            nn = [count(!isnan, view(ic, :, k)) for k in 1:K]
            @test isequal(s.hit_rate,
                          [nn[k] > 0 ? count(>(0), view(ic, :, k)) / nn[k] : NaN
                           for k in 1:K])
        end
        if c in fam
            # The rank coefficient on the reduced axis is bit-equal on the three panels. The
            # linear one carries the round-off of the fit's returns, as on the raw axis:
            # measured maxrel 1.1e-12 on the large panel, and maxscaled 2.3e-15.
            o = load(n, "ICRed")
            Kr = size(o, 2) ÷ 2
            @test pc(exposure_ic(csfm; ties = :ordinal, reduced = true).X, o[:, 1:Kr],
                     "$(n) reduced rank ic")
            @test pc(exposure_ic(csfm; rank = false, reduced = true).X, o[:, (Kr + 1):end],
                     "$(n) reduced linear ic"; rtol = 1.5e-12)
        end
    end

    @testset "The factor model summary, $(fix) $(c)" for (fix, c) in cases
        csfm = block(fix, c)
        n = "Diag$(fix)$(c)"
        O = load(n, "Summary")
        nf = csfm.nf
        cols = (:ann_return, :ann_volatility, :sharpe, :autocorr, :mean_abs_t, :t_rate,
                :mean_vif, :coverage, :stability)
        keep = trues(length(nf), length(cols))
        if c in fam
            # The factor return of a level the basis drops is fixed by the constraint of its
            # family. The block carries it on the raw axis in `fr`, so the summary states the
            # four return statistics of that level too, at parity (#1422).
            red = PO.reduce_factor_names(csfm.fcb, nf)
            drop = [!(x in red) for x in nf]
            @test any(drop)
            @test all(isfinite, factor_model_summary(csfm).ann_return[drop])
            # A level that empties leaves rows where the design is singular. There a
            # coefficient the design does not identify has no t-statistic and an infinite
            # VIF, and the others divide by `n - rank`, where the oracle reads round-off and
            # `n - K` (#1421, see the regression group). So a singular row changes the
            # t-statistic of every factor, and the oracle's mean absolute t and exceedance
            # rate differ for each one, by 0.33 and 0.26 relative on the large panel. A factor
            # that a singular row does not identify has a mean VIF of `Inf`, where the oracle's
            # is finite. The three columns compare instead against the same statistics of the
            # oracle's t-statistics and VIFs on the rows of full rank and ours on the singular
            # rows, which the regression group pins to the fit on a pivoted basis. A factor
            # that no singular row identifies reads the oracle's rows of full rank alone
            # (1.7500 for the level of one member, where the round-off rows gave 1.7755).
            # Every row of `FamTwo` is singular, so there the check reads ours alone and tests
            # the statistics. Measured maxrel 8.2e-16 on the mean absolute t, 2.9e-16 on the
            # mean VIF, and an exact rate.
            t = cs_regression_t_stats(csfm).X
            vif = exposure_vif(csfm).X
            R = load(n, "Regression")
            K = size(t, 2)
            sing = .!((exposure_condition_number(csfm) .< 1e12) .& (R[:, 2K + 1] .< 1e12))
            @test any(sing) && (c == "FamTwo" || !all(sing))
            th = R[:, 1:K]
            th[sing, :] .= t[sing, :]
            vh = R[:, (K + 1):(2K)]
            vh[sing, :] .= vif[sing, :]
            pos = PO.factor_summary_positions(csfm)
            colmean(A) = PO.factor_summary_mapped([PO.factor_summary_column_mean(A, j)
                                                   for j in 1:K], pos)
            mabs = colmean(abs.(th))
            mvif = colmean(vh)
            @test any(isinf, mvif)
            for (thr, step) in ((2, 1), (1.5, 5))
                fs = factor_model_summary(csfm; ppy = 252, threshold = thr, step = step)
                rate = PO.factor_summary_mapped(cs_regression_t_stat_exceedance_rate(th;
                                                                                     threshold = thr),
                                                pos)
                @test pc(fs.mean_abs_t, mabs, "$(n) mean abs t $(thr)")
                @test pc(fs.t_rate, rate, "$(n) t rate $(thr)")
                @test pc(fs.mean_vif, mvif, "$(n) mean vif $(thr)")
            end
            keep[:, 5:7] .= false
        end
        if c == "Macro"
            # The stability reads the benchmark weights of the fit, which differ on two cells
            # (see the exposure group).
            keep[:, 9] .= false
        end
        for (off, fs) in ((0, factor_model_summary(csfm; ppy = 252)),
                          (9, factor_model_summary(csfm; ppy = 252, threshold = 1.5, step = 5)))
            for (j, cn) in enumerate(cols)
                k = keep[:, j]
                @test pc(getfield(fs, cn)[k], O[k, off + j], "$(n) $(cn) $(off)")
            end
        end
    end

    @testset "The idiosyncratic group, $(fix) $(c)" for (fix, c) in cases
        csfm = block(fix, c)
        if c in fam
            # Better (#1423). Asset 3 is the only member of its industry until it delists,
            # so the design gives it a direction of its own and the fit reproduces its
            # return. Its residual is zero by construction, and so is its variance under the
            # design: its standardised return is 0 / 0. The oracle divides the round-off of
            # one by the round-off of the other, near 1e-18 / 1e-17, and enters an order-one
            # number into every cross-section, so no stored case compares here. The fit
            # marks the pair, writes an exact zero, and the group leaves the asset out, so
            # each verb equals the same verb on the panel without asset 3.
            h = csfm.csr.h1
            act = .!isnan.(csfm.csr.eps)
            @test findall(vec(any(h; dims = 1))) == [3]
            @test h[:, 3] == act[:, 3]
            @test all(iszero, csfm.csr.eps[h])
            @test all(x -> iszero(x) || isnan(x), csfm.vs[:, 3])
            @test all(isnan,
                      standardised_idio_returns(PO.idio_diagnostic_data(csfm)...)[:, 3])
            k = setdiff(axes(csfm.csr.eps, 2), 3)
            ek = csfm.csr.eps[:, k]
            vk = csfm.vs[:, k]
            for ahead in (true, false)
                @test isequal(idio_calibration(csfm; ahead = ahead),
                              idio_calibration(ek, vk; ahead = ahead))
                @test isequal(idio_tail_rate(csfm; ahead = ahead),
                              idio_tail_rate(ek, vk; ahead = ahead))
                @test isequal(idio_kurtosis(csfm; ahead = ahead),
                              idio_kurtosis(ek, vk; ahead = ahead))
                @test isequal(idio_skewness(csfm; ahead = ahead),
                              idio_skewness(ek, vk; ahead = ahead))
                @test isequal(idio_calibration_summary(csfm; ahead = ahead),
                              idio_calibration_summary(ek, vk; ahead = ahead))
            end
            @test isequal(idio_vol_ic(csfm), idio_vol_ic(ek, vk))
            continue
        end
        n = "Diag$(fix)$(c)"
        I = load(n, "Idio")
        @test pc(idio_calibration(csfm; ahead = false), I[:, 1], "$(n) calibration")
        @test pc(idio_tail_rate(csfm; ahead = false), I[:, 2], "$(n) tail 3")
        @test pc(idio_tail_rate(csfm; threshold = 2, ahead = false), I[:, 3], "$(n) tail 2")
        # The excess kurtosis subtracts three from a ratio near three, so an entry near zero
        # comes from a cancellation and compares against the largest entry. Measured maxrel
        # 2.4e-11 on the large panel, and maxscaled 1.4e-15.
        @test pc(idio_kurtosis(csfm; ahead = false), I[:, 4], "$(n) kurtosis";
                 scale = :array)
        @test pc(idio_skewness(csfm; ahead = false), I[:, 5], "$(n) skewness")
        @test pc(collect(values(idio_calibration_summary(csfm; ahead = false))),
                 loadv(n, "IdioSummary"), "$(n) summary")
        Dp = load(n, "IdioDependence")
        @test pc(idio_vol_ic(csfm; ties = :ordinal), Dp[:, 1], "$(n) vol ic")
        @test pc(idio_vol_residual_dependence(csfm; ties = :ordinal), Dp[:, 2],
                 "$(n) residual dependence")
        # Better: the default divides each return by the volatility of the previous row, whose
        # estimate has not read it. The oracle's own kernels on that standardisation give the
        # stored `Ahead` outputs.
        A = load(n, "AheadIdio")
        @test pc(idio_calibration(csfm), A[:, 1], "$(n) ahead calibration")
        @test pc(idio_tail_rate(csfm), A[:, 2], "$(n) ahead tail")
        # An excess kurtosis near zero is a difference of near-equal terms: measured cell
        # maxrel 2.4e-11 and maxscaled 2.6e-15, so the check is `:array`.
        @test pc(idio_kurtosis(csfm), A[:, 3], "$(n) ahead kurtosis"; scale = :array)
        @test pc(idio_skewness(csfm), A[:, 4], "$(n) ahead skewness")
        @test pc(collect(values(idio_calibration_summary(csfm))),
                 loadv(n, "AheadIdioSummary"), "$(n) ahead summary")
    end

    @testset "The same-row standardisation understates the tails" begin
        # The panels draw Gaussian residuals. The variance of a row has read the return of
        # that row, so the same-row tail rate sits below the Gaussian reference of 0.27 %, and
        # the one-step-ahead rate sits above it, as the error of an estimated variance makes
        # it. Measured on the large panel: 0.12 % against 0.55 %, and a mean deviation of 0.988
        # against 1.009.
        csfm = block("Large", "Base")
        sr = idio_calibration_summary(csfm; ahead = false)
        sa = idio_calibration_summary(csfm)
        # The Gaussian reference of a threshold of three, `2Φ(-3)`.
        @test sr.mean_tail_rate < 0.002699796063260207 < sa.mean_tail_rate
        @test sr.mean_cs_std < 1 < sa.mean_cs_std
    end

    @testset "The midrank of a tie" begin
        # The default `ties = :average` against the oracle's kernels with the midrank. The
        # constant market exposure has constant midranks, so it has no coefficient at any
        # observation, on either side.
        csfm = block("Small", "Base")
        n = "DiagSmallBaseAverage"
        Q = load(n, "ICSummary")
        for (h, o) in ((1, load(n, "IC1")), (3, load(n, "IC3")))
            ic = exposure_ic(csfm; horizon = h).X
            @test pc(ic, o, "average ic $(h)")
            @test all(isnan, ic[:, 1])
            s = exposure_ic_summary(csfm; horizon = h)
            q = Q[:, (4 * (h == 3) + 1):(4 * (h == 3) + 4)]
            @test pc(s.mean_ic, q[:, 1], "average mean ic $(h)")
            @test pc(s.std_ic, q[:, 2], "average std ic $(h)")
            @test pc(s.ic_ir, q[:, 3], "average ic ir $(h)")
            # The market factor has no coefficient, so it has no hit rate. The oracle counts
            # each `NaN` as a miss and reports zero.
            @test isnan(s.hit_rate[1]) && q[1, 4] == 0
            @test s.hit_rate[2:end] == q[2:end, 4]
        end
        Dp = load(n, "IdioDependence")
        @test pc(idio_vol_ic(csfm), Dp[:, 1], "average vol ic")
        @test pc(idio_vol_residual_dependence(csfm), Dp[:, 2],
                 "average residual dependence")
    end
end
