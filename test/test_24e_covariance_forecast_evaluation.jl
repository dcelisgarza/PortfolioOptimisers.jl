#=
The covariance forecast evaluation, issue #1023, against the decision of #873 (ADR 0143).

One verb through the one fold loop: `covariance_forecast_evaluation(est, rd, cv)` scores a
covariance estimator's, or a prior's, forecast on the test rows of every fold, in batch when
the walk-forward refits and online when it declares `ff = OnlineStep()`. Around the verb sit
the per-step kernel and its parity with the reference implementation's, the two realised
targets, the location the test rows are centred on, the online and rolling identities over a
panel with a listing and a delisting, the date form, the summary against the reference's
summaries, the Diebold–Mariano–West comparison, the re-projection, and every refusal.

The kernel's parity literals were produced by the reference's per-step kernel on the fixed
forecast and test rows below, with two test portfolios, once with every cell finite and once
with one cell missing; the summary literals by its summary verbs on a run of seven such
steps. Both sides read the same formula-built inputs, so no random stream is shared.
=#
# A caller's covariance estimator with no location of its own, for the fallback. A struct is
# declared at the top level because a testset body is a local scope.
struct NoLocationCovariance <: PortfolioOptimisers.AbstractCovarianceEstimator end
@testset "Covariance forecast evaluation: one verb through the fold loop" begin
    using Test, PortfolioOptimisers, StableRNGs, Statistics, StatsBase, Dates, LinearAlgebra
    po = PortfolioOptimisers
    cfe = covariance_forecast_evaluation
    rng = StableRNG(20260911)
    T, N = 160, 8
    X = randn(rng, T, N) ./ 100 .+ 0.0003
    nx = ["A$i" for i in 1:N]
    ts = Date(2024, 1, 1) .+ Day.(0:(T - 1))
    rd = ReturnsResult(; nx = nx, X = X, ts = ts)
    # A listing at row 31 and a delisting at row 131, expressed by the panel's masks.
    amsk = trues(T, N)
    amsk[1:30, 3] .= false
    amsk[131:end, 5] .= false
    Xg = copy(X)
    Xg[.!amsk] .= NaN
    rdg = ReturnsResult(; nx = nx, X = Xg, ts = ts,
                        pnl = AssetPanel(; amsk = amsk, emsk = copy(amsk)))
    w, h, p = 60, 5, 2
    batch_cv = IndexWalkForward(w, h; purged_size = p, expand_train = true)
    online_cv = IndexWalkForward(w, h; purged_size = p, ff = OnlineStep())
    finite_max(a, b) = maximum(abs, filter(!isnan, a .- b))
    columns = (:mahalanobis_ratio, :diagonal_ratio, :qlike, :frobenius,
               :standardised_return, :portfolio_qlike)

    @testset "The kernel against the reference, with and without a missing cell" begin
        A = [1.0 0.3 -0.2 0.1; 0.2 1.1 0.4 -0.3; -0.1 0.5 0.9 0.2; 0.3 -0.2 0.1 1.2]
        sig = (A * A') / 1e4 + Diagonal([1e-4, 2e-4, 1.5e-4, 1.2e-4])
        Zr = [sin(0.7 * t + 0.29 * i) / 100 + 0.004 * cos(0.11 * t * i)
              for t in 1:3, i in 1:4]
        Zn = copy(Zr)
        Zn[2, 3] = NaN
        W = [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]]
        # The reference centres nothing, so its numbers are ours at a zero location under
        # its own target, the horizon return.
        k(Z, wt) = covariance_forecast_step(sig, Z, zeros(4), wt, HorizonReturn())
        s = k(Zr, nothing)
        @test s.n_valid == 4
        @test isapprox(s.mahalanobis_ratio, 0.978852966148661; atol = 1e-14)
        @test isapprox(mean(s.diagonal_ratio), 1.1925335096638512; atol = 1e-14)
        @test isapprox(s.standardised_return, [1.9211917469757487]; atol = 1e-14)
        @test isapprox(s.portfolio_qlike, [-6.949593962426171]; atol = 1e-13)
        s = k(Zr, W)
        @test isapprox(s.standardised_return, [1.894133508777383, 1.6253620587183681];
                       atol = 1e-14)
        @test isapprox(s.portfolio_qlike, [-6.961786146396911, -7.110698408378284];
                       atol = 1e-13)
        # One missing cell: the pairwise count scales the forecast, `H ⊙ Σ̂`, and the
        # reference's numbers move with it.
        s = k(Zn, nothing)
        @test s.n_valid == 4
        @test isapprox(s.mahalanobis_ratio, 0.9039713006494878; atol = 1e-14)
        @test isapprox(mean(s.diagonal_ratio), 1.0830084753657339; atol = 1e-14)
        @test isapprox(s.standardised_return, [1.850608226250147]; atol = 1e-14)
        @test isapprox(s.portfolio_qlike, [-7.149655216935746]; atol = 1e-13)
        s = k(Zn, W)
        @test isapprox(s.standardised_return, [1.8276572778164961, 1.5382386137061885];
                       atol = 1e-14)
        @test isapprox(s.portfolio_qlike, [-7.156129366719016, -7.322586405112045];
                       atol = 1e-13)
        # An independent re-derivation of the missing-cell case in plain Julia, so the
        # literals above are not the only oracle: with `H` the pairwise finite count,
        # `m = R' (H ⊙ Σ̂)⁻¹ R / N`, `d_i = R_i² / (H_ii Σ̂_ii)`, `b = w'R / √(w'(H ⊙ Σ̂)w)`.
        fin = isfinite.(Zn)
        Zf = ifelse.(fin, Zn, 0.0)
        R = vec(sum(Zf; dims = 1))
        H = fin' * fin
        seff = H .* sig
        @test isapprox(s.mahalanobis_ratio, dot(R, seff \ R) / 4; rtol = 1e-12)
        @test isapprox(s.diagonal_ratio, R .^ 2 ./ (diag(H) .* diag(sig)); rtol = 1e-12)
        wn = W[2] ./ sum(W[2])
        v = dot(wn, seff, wn)
        @test isapprox(s.standardised_return[2], dot(wn, R) / sqrt(v); rtol = 1e-12)
        @test isapprox(s.portfolio_qlike[2], log(v) + sum(abs2, Zf * wn) / v; rtol = 1e-12)
        # The whole-matrix losses, which the reference does not compute, against their
        # closed forms on the finite case: `h log|Σ̂| + tr(Σ̂⁻¹ S)` and `‖S / h − Σ̂‖²_F`.
        sr = covariance_forecast_step(sig, Zr, zeros(4), nothing, RealisedCovariance())
        S = Zr' * Zr
        @test isapprox(sr.qlike, 3 * logdet(sig) + tr(sig \ S); rtol = 1e-12)
        @test isapprox(sr.frobenius, sum(abs2, S ./ 3 .- sig); rtol = 1e-12)
        @test isapprox(sr.mahalanobis_ratio, tr(sig \ S) / 12; rtol = 1e-12)
        # The identity `L^QLIKE = h log|Σ̂| + N h m` holds under a gap too.
        sn = covariance_forecast_step(sig, Zn, zeros(4), nothing, RealisedCovariance())
        @test isapprox(sn.qlike, 3 * logdet(sig) + 4 * 3 * sn.mahalanobis_ratio;
                       rtol = 1e-12)
    end

    @testset "The two targets coincide at h = 1, and their degrees of freedom" begin
        sig = cov(Covariance(), X[1:60, :])
        Z = X[61:61, :]
        c = vec(mean(X[1:60, :]; dims = 1))
        a = covariance_forecast_step(sig, Z, c, nothing, RealisedCovariance())
        b = covariance_forecast_step(sig, Z, c, nothing, HorizonReturn())
        for f in columns
            @test getfield(a, f) == getfield(b, f)
        end
        Z5 = X[61:65, :]
        a = covariance_forecast_step(sig, Z5, c, nothing, RealisedCovariance())
        b = covariance_forecast_step(sig, Z5, c, nothing, HorizonReturn())
        @test a.mahalanobis_ratio != b.mahalanobis_ratio
        # The portfolio QLIKE reads the per-row portfolio returns whatever the target.
        @test a.portfolio_qlike == b.portfolio_qlike
        @test a.standardised_return == b.standardised_return
        @test po.target_dof(RealisedCovariance(), 8, 5) == 40
        @test po.target_dof(HorizonReturn(), 8, 5) == 8
        @test po.target_step_dof(RealisedCovariance(), 5) == 5
        @test po.target_step_dof(HorizonReturn(), 5) == 1
        S, H = po.realised_target(RealisedCovariance(), Z5)
        @test S ≈ Z5' * Z5
        @test H == fill(5, N, N)
        S, H = po.realised_target(HorizonReturn(), Z5)
        r = vec(sum(Z5; dims = 1))
        @test S ≈ r * r'
    end

    @testset "The centring term: the raw ratio less the centred one" begin
        # With `R` the raw column sums of the test rows, `S_raw − S_c = R c' + c R' − h c c'`,
        # so the raw Mahalanobis ratio exceeds the centred one by
        # `(2 R' Σ̂⁻¹ c − h c' Σ̂⁻¹ c) / (N h)` at every step, exactly; its expectation when
        # `E[z] = c` is the bias `c' Σ̂⁻¹ c / N` the reference's raw ratio carries.
        Xm = X .+ 0.02
        sig = cov(Covariance(), Xm[1:60, :])
        c = vec(mean(Xm[1:60, :]; dims = 1))
        Z = Xm[61:65, :]
        raw = covariance_forecast_step(sig, Z, zeros(N), nothing, RealisedCovariance())
        cen = covariance_forecast_step(sig, Z, c, nothing, RealisedCovariance())
        R = vec(sum(Z; dims = 1))
        term = (2 * dot(R, sig \ c) - 5 * dot(c, sig \ c)) / (N * 5)
        @test isapprox(raw.mahalanobis_ratio - cen.mahalanobis_ratio, term; rtol = 1e-10)
        @test raw.mahalanobis_ratio > cen.mahalanobis_ratio
        # The location is the estimator's: a covariance estimator centres on its own
        # mean, a centred exponentially weighted one on zero, a prior on its `mu`.
        @test po.forecast_location(Covariance(), Xm[1:60, :]) ≈ c
        @test po.forecast_location(GeneralCovariance(), Xm[1:60, :]) ≈ c
        ow = StatsBase.pweights(collect(1.0:60))
        @test po.forecast_location(GeneralCovariance(; w = ow), Xm[1:60, :]) ≈
              vec(mean(Xm[1:60, :], ow; dims = 1))
        @test po.forecast_location(PortfolioOptimisersCovariance(), Xm[1:60, :]) ≈ c
        @test po.forecast_location(ExpWeightedCovariance(; centred = true), Xm[1:60, :]) ==
              zeros(N)
        @test all(!iszero, po.forecast_location(ExpWeightedCovariance(), Xm[1:60, :]))
        @test po.forecast_location(EmpiricalPrior(), po.port_opt_view(rd, 1:60, :)) ≈
              vec(mean(X[1:60, :]; dims = 1))
        ce = po.partial_fit!(Covariance(), Xm[1:60, :])
        @test po.forecast_location(ce) ≈ c
        # Under a policy the batch centre is the fold's: each asset's own available-case
        # mean, the diagonal of the per-pair centre.
        cp = Covariance(; cvg = CoveragePolicy())
        cb = po.forecast_location(cp, Xg[1:60, :]; active_mask = amsk[1:60, :])
        co = po.forecast_location(po.partial_fit!(cp, Xg[1:60, :];
                                                  active_mask = amsk[1:60, :]))
        @test cb == co
        @test cb[3] ≈ mean(Xg[31:60, 3])
        @test cb[1] ≈ mean(Xg[1:60, 1])
        # The fallback for an estimator with no location of its own is the finite column
        # mean of the window.
        @test po.forecast_location(NoLocationCovariance(), Xg[1:60, :])[3] ≈
              mean(Xg[31:60, 3])
    end

    @testset "The online identity over a panel with a listing and a delisting" begin
        # (estimator, carrier, tolerance). The exponentially weighted families and the
        # policy fold are exact, because their batch fit is the pass; the plain Welford
        # families and the prior differ by the rounding of a row-by-row fold.
        cases = ((GeneralCovariance(), rd, 1e-12), (Covariance(), rd, 1e-12),
                 (PortfolioOptimisersCovariance(), rd, 1e-12),
                 (Covariance(; cvg = CoveragePolicy()), rdg, 0.0),
                 (EmpiricalPrior(), rd, 1e-12), (EmpiricalPrior(), rdg, 1e-12),
                 (ExpWeightedCovariance(), rd, 0.0),
                 (ExpWeightedCovariance(; centred = true), rdg, 0.0),
                 (RegimeAdjustedExpWeightedCovariance(), rd, 0.0))
        for (est, r, tol) in cases
            b = cfe(est, r, batch_cv)
            o = cfe(est, r, online_cv)
            @test b.dates == o.dates
            @test b.horizon == o.horizon == fill(h, length(b.dates))
            @test b.n_valid == o.n_valid
            @test isequal(isnan.(b.diagonal_ratio), isnan.(o.diagonal_ratio))
            for f in columns
                @test finite_max(getfield(b, f), getfield(o, f)) <= tol
            end
        end
        # The gapped panel: an expanding window from the first row never holds the young
        # asset in its Coverage Universe, so the prior's frame drops it at every step, and
        # it drops the delisted asset once it leaves.
        b = cfe(EmpiricalPrior(), rdg, batch_cv)
        @test b.n_valid[1] == 7
        @test b.n_valid[end] == 6
        @test all(isnan, b.diagonal_ratio[:, 3])
        @test isnan(b.diagonal_ratio[end, 5])
        # A policy admits the young asset from the rows it has, so its frame holds all
        # eight once the asset has listed.
        bp = cfe(Covariance(; cvg = CoveragePolicy()), rdg, batch_cv)
        @test 8 in bp.n_valid
        @test !isnan(bp.diagonal_ratio[end, 3])
        # The realised target and the test portfolios ride into both arms alike.
        wt = [fill(1 / N, N), rand(StableRNG(1), N)]
        b = cfe(EmpiricalPrior(), rd, batch_cv; target = HorizonReturn(), w = wt)
        o = cfe(EmpiricalPrior(), rd, online_cv; target = HorizonReturn(), w = wt)
        @test size(b.standardised_return) == (length(b.dates), 2)
        @test b.target == HorizonReturn()
        for f in columns
            @test finite_max(getfield(b, f), getfield(o, f)) <= 1e-12
        end
    end

    @testset "The rolling identity through Online(est; max_history = w)" begin
        rolling = IndexWalkForward(w + p, h; purged_size = p)
        stepped = IndexWalkForward(w + p, h; purged_size = p, ff = OnlineStep())
        @test all(length.(split(rolling, rd).train_idx) .== w)
        for (est, r) in
            ((Covariance(), rd), (GeneralCovariance(), rd), (EmpiricalPrior(), rd),
             (Covariance(; cvg = CoveragePolicy()), rdg), (EmpiricalPrior(), rdg),
             (ExpWeightedCovariance(), rdg))
            b = cfe(est, r, rolling)
            o = cfe(po.Online(est; max_history = w), r, stepped)
            @test b.n_valid == o.n_valid
            for f in columns
                # A capped buffer reads out over exactly the rolling window's rows, so the
                # two fits see one sample.
                @test finite_max(getfield(b, f), getfield(o, f)) <= 1e-12
            end
        end
    end

    @testset "The date form" begin
        bd = DateWalkForward(w, h; period = Day(1), purged_size = p, expand_train = true)
        od = DateWalkForward(w, h; period = Day(1), purged_size = p, ff = OnlineStep())
        bi = cfe(Covariance(), rdg, batch_cv)
        b = cfe(Covariance(), rdg, bd)
        o = cfe(Covariance(), rdg, od)
        # The date form reaches the index form's rows, fold for fold, on the folds it cuts;
        # a day period whose last window would run past the sample cuts one fold fewer.
        n = length(b.test_idx)
        @test n >= length(bi.test_idx) - 1
        @test b.test_idx == bi.test_idx[1:n]
        @test b.dates == bi.dates[1:n]
        @test b.mahalanobis_ratio == bi.mahalanobis_ratio[1:n]
        @test finite_max(b.mahalanobis_ratio, o.mahalanobis_ratio) <= 1e-12
        # A calendar period cuts folds of unequal length, and the horizon records each.
        bm = DateWalkForward(2, 1; period = Month(1))
        r = cfe(Covariance(), rd, bm)
        @test r.horizon == length.(split(bm, rd).test_idx)
        @test length(unique(r.horizon)) > 1
        s = covariance_forecast_summary(r)
        @test s.n_steps == [length(r.horizon)]
        # A carrier with no timestamps labels a step by its first test row.
        rn = cfe(Covariance(), ReturnsResult(; nx = nx, X = X), batch_cv)
        @test rn.dates == first.(split(batch_cv, rd).test_idx)
    end

    @testset "The summary against the reference's summaries" begin
        A = [1.0 0.3 -0.2 0.1; 0.2 1.1 0.4 -0.3; -0.1 0.5 0.9 0.2; 0.3 -0.2 0.1 1.2]
        sig = (A * A') / 1e4 + Diagonal([1e-4, 2e-4, 1.5e-4, 1.2e-4])
        W = [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]]
        M = 7
        steps = map(0:(M - 1)) do k
            s = sig .* (1 + 0.1 * k)
            Z = [sin(0.7 * (t + 3 * k) + 0.29 * i) / 100 +
                 0.004 * cos(0.11 * (t + 3 * k) * i) for t in 1:3, i in 1:4]
            if k == 4
                Z[1, 2] = NaN
            end
            return covariance_forecast_step(s, Z, zeros(4), W, HorizonReturn())
        end
        stack(f) = reduce(vcat, [permutedims(getfield(s, f)) for s in steps])
        cfer = CovarianceForecastEvaluationResult(collect(0:(M - 1)),
                                                  [(1:3) .+ 3k for k in 0:(M - 1)],
                                                  fill(3, M), HorizonReturn(),
                                                  [s.n_valid for s in steps],
                                                  [s.mahalanobis_ratio for s in steps],
                                                  stack(:diagonal_ratio),
                                                  [s.qlike for s in steps],
                                                  [s.frobenius for s in steps],
                                                  stack(:standardised_return),
                                                  stack(:portfolio_qlike), W, nothing,
                                                  nothing)
        @test isapprox(cfer.mahalanobis_ratio,
                       [0.978852966148661, 0.4653960403261437, 0.011680925757934219,
                        0.315530814251327, 0.34876711552388395, 0.15922909197779334,
                        0.2726172060164678]; atol = 1e-14)
        s = covariance_forecast_summary(cfer)
        @test s.names == ["forecast_1"]
        # The reference's `summary()` rows, to the six digits it prints.
        @test isapprox(s.mahalanobis_mean[1], 0.364582; atol = 5e-7)
        @test isapprox(s.mahalanobis_median[1], 0.315531; atol = 5e-7)
        @test isapprox(s.mahalanobis_p5[1], 0.055945; atol = 5e-7)
        @test isapprox(s.mahalanobis_p95[1], 0.824816; atol = 5e-7)
        @test isapprox(s.diagonal_mean[1], 0.425925; atol = 5e-7)
        @test isapprox(s.diagonal_median[1], 0.36462; atol = 5e-6)
        @test isapprox(s.diagonal_p5[1], 0.057891; atol = 5e-7)
        @test isapprox(s.diagonal_p95[1], 0.988609; atol = 5e-7)
        @test isapprox(s.portfolio_qlike_mean[1], -7.520415; atol = 5e-7)
        # Its `bias_statistic_summary()`.
        @test isapprox(s.bias_statistic[1], 1.090341; atol = 5e-7)
        @test isapprox(s.bias_p5[1], 1.054794; atol = 5e-7)
        @test isapprox(s.bias_p25[1], 1.070593; atol = 5e-7)
        @test isapprox(s.bias_p75[1], 1.110090; atol = 5e-7)
        @test isapprox(s.bias_p95[1], 1.125889; atol = 5e-7)
        # Its `exceedance_summary()` at (0.95, 0.99): no step exceeds.
        @test s.levels == (0.95, 0.99)
        @test s.exceedance == [0.0 0.0]
        @test s.n_steps == [M]
        @test s.n_portfolios == [2]
        # The Gaussian bands: `1 ± z √(2 / Σ dof)`, with `Σ dof = 4 M` under the horizon
        # return and `M` for one asset's ratio.
        z = 1.959963984540054
        @test isapprox(s.mahalanobis_band_hi[1], 1 + z * sqrt(2 / (4M)); rtol = 1e-12)
        @test isapprox(s.mahalanobis_band_lo[1], 1 - z * sqrt(2 / (4M)); rtol = 1e-12)
        @test isapprox(s.diagonal_band_hi[1], 1 + z * sqrt(2 / M); rtol = 1e-12)
        @test s.alpha == 0.05
        # The realised covariance has `N h` degrees of freedom per step, so its band is
        # narrower by `√h` and its exceedance threshold sits at `χ²_{N h}`.
        cfr = CovarianceForecastEvaluationResult(cfer.dates, cfer.test_idx, cfer.horizon,
                                                 RealisedCovariance(), cfer.n_valid,
                                                 cfer.mahalanobis_ratio,
                                                 cfer.diagonal_ratio, cfer.qlike,
                                                 cfer.frobenius, cfer.standardised_return,
                                                 cfer.portfolio_qlike, W, nothing, nothing)
        sr = covariance_forecast_summary(cfr; alpha = 0.1, levels = (0.5,))
        @test isapprox(sr.mahalanobis_band_hi[1], 1 + 1.6448536269514722 * sqrt(2 / (12M));
                       rtol = 1e-12)
        @test sr.levels == (0.5,)
        @test size(sr.exceedance) == (1, 1)
        # Two evaluations are one Result of length two, and the length-1 method agrees.
        s2 = covariance_forecast_summary([cfer, cfr]; names = ["a", "b"])
        @test s2.names == ["a", "b"]
        @test s2.mahalanobis_mean[1] == s.mahalanobis_mean[1]
        @test s2.qlike_mean == fill(mean(cfer.qlike), 2)
        # Every column has one entry per evaluation.
        for f in fieldnames(CovarianceForecastSummaryResult)
            v = getfield(s2, f)
            if isa(v, AbstractVector)
                @test length(v) == 2
            end
        end
    end

    @testset "The comparison: Diebold–Mariano–West on the loss difference" begin
        bcv3 = IndexWalkForward(w, 3; expand_train = true)
        a = cfe(Covariance(), rd, bcv3; store_forecasts = true)
        b = cfe(ExpWeightedCovariance(; decay = 0.8), rd, bcv3)
        cmp = covariance_forecast_compare(a, b)
        @test cmp.names == ["qlike", "frobenius", "portfolio_qlike_1"]
        @test cmp.lags == 2
        @test cmp.n_steps == length(a.dates)
        # On an i.i.d. panel the sample covariance is close to the truth and a fast decay
        # is noisy, so the first loses less: a negative mean difference, and a small `p`.
        @test all(cmp.mean_difference .< 0)
        @test all(cmp.z .< 0)
        @test all(cmp.p .< 0.05)
        @test cmp.p ≈ 2 .* po.Distributions.ccdf.(po.Distributions.Normal(), abs.(cmp.z))
        # The same forecast twice: no difference, and no statistic.
        same = covariance_forecast_compare(a, a)
        @test all(iszero, same.mean_difference)
        @test all(iszero, same.variance)
        @test all(isnan, same.z)
        # The Newey–West variance at zero lags is the plain sample variance about the
        # mean, and a lag adds Bartlett-weighted autocovariances.
        d = a.qlike .- b.qlike
        M = length(d)
        @test po.newey_west_variance(d, 0) ≈ var(d; corrected = false)
        e = d .- mean(d)
        g1 = dot(e[2:end], e[1:(end - 1)]) / M
        @test po.newey_west_variance(d, 1) ≈
              var(d; corrected = false) + 2 * (1 - 1 / 2) * g1
        @test covariance_forecast_compare(a, b; lags = 0).variance ≈
              [var(a.qlike .- b.qlike; corrected = false),
               var(a.frobenius .- b.frobenius; corrected = false),
               var(a.portfolio_qlike[:, 1] .- b.portfolio_qlike[:, 1]; corrected = false)]
        # The sign: swapping the arguments negates the statistic.
        rev = covariance_forecast_compare(b, a)
        @test rev.z ≈ -cmp.z
    end

    @testset "The re-projection equals a rerun with that portfolio" begin
        wt = [fill(1 / N, N), collect(1.0:N) ./ sum(1:N)]
        a = cfe(Covariance(), rdg, batch_cv; store_forecasts = true)
        @test length(a.sigma) == length(a.dates)
        @test all(s -> size(s) == (N, N), a.sigma)
        @test all(c -> length(c) == N, a.location)
        pj = covariance_forecast_portfolio(a, rdg, wt)
        rerun = cfe(Covariance(), rdg, batch_cv; w = wt)
        @test pj.standardised_return == rerun.standardised_return
        @test pj.portfolio_qlike == rerun.portfolio_qlike
        # The evaluation's own `w` reproduces its columns.
        own = covariance_forecast_portfolio(a, rdg, nothing)
        @test own.standardised_return == a.standardised_return
        @test own.portfolio_qlike == a.portfolio_qlike
        # Without the forecasts there is nothing to re-project.
        @test isnothing(rerun.sigma)
        @test isnothing(rerun.location)
    end

    @testset "Every refusal by name" begin
        sig = cov(Covariance(), X[1:60, :])
        Z = X[61:65, :]
        c = zeros(N)
        # The kernel.
        @test_throws DimensionMismatch covariance_forecast_step(sig, Z[:, 1:3], c, nothing,
                                                                RealisedCovariance())
        @test_throws DimensionMismatch covariance_forecast_step(sig, Z, c[1:3], nothing,
                                                                RealisedCovariance())
        @test_throws DimensionMismatch covariance_forecast_step(sig, Z, c, ones(3),
                                                                RealisedCovariance())
        @test_throws DimensionMismatch covariance_forecast_step(sig, Z, c,
                                                                [ones(N), ones(3)],
                                                                RealisedCovariance())
        e = @test_throws ArgumentError covariance_forecast_step(sig, fill(NaN, 2, N), c,
                                                                nothing,
                                                                RealisedCovariance())
        @test occursin("no asset is active", e.value.msg)
        nan_sig = fill(NaN, N, N)
        e = @test_throws ArgumentError covariance_forecast_step(nan_sig, Z, c, nothing,
                                                                RealisedCovariance())
        @test occursin("no asset is active", e.value.msg)
        # A non-finite location drops the asset from the active subset.
        cn = copy(c)
        cn[2] = NaN
        @test covariance_forecast_step(sig, Z, cn, nothing, RealisedCovariance()).n_valid ==
              N - 1
        # The verb.
        @test_throws po.IsNothingError cfe(Covariance(), ReturnsResult(; nx = nx), batch_cv)
        e = @test_throws ArgumentError cfe(po.Online(Covariance()), rd, batch_cv)
        @test occursin("declares no Fold Fit", e.value.msg)
        # The loop starts cold: a state at entry is refused by name.
        e = @test_throws ArgumentError cfe(po.partial_fit!(Covariance(), X[1:10, :]), rd,
                                           online_cv)
        @test occursin("`cache`", e.value.msg)
        # A wrapper at the root is walked through: the state it wraps is named by its path.
        e = @test_throws ArgumentError cfe(po.Online(po.partial_fit!(Covariance(),
                                                                     X[1:10, :])), rd,
                                           online_cv)
        @test occursin("`est.cache`", e.value.msg)
        @test isnothing(po.online_entry_state(po.Online(Covariance())))
        @test_throws po.IsNothingError po.partial_fit!(Covariance(),
                                                       ReturnsResult(; nx = nx))
        # The location.
        e = @test_throws ArgumentError po.forecast_location(Covariance())
        @test occursin("no partial-fit state", e.value.msg)
        e = @test_throws ArgumentError po.forecast_location(PortfolioOptimisersCovariance())
        @test occursin("no partial-fit state", e.value.msg)
        # The summary.
        r = cfe(Covariance(), rd, batch_cv)
        @test_throws po.IsEmptyError covariance_forecast_summary(CovarianceForecastEvaluationResult[])
        @test_throws DimensionMismatch covariance_forecast_summary([r]; names = ["a", "b"])
        @test_throws DomainError covariance_forecast_summary(r; alpha = 1.5)
        @test_throws DomainError covariance_forecast_summary(r; levels = (0.95, 1.0))
        # The comparison.
        r2 = cfe(Covariance(), rd, IndexWalkForward(w, 3; expand_train = true))
        e = @test_throws ArgumentError covariance_forecast_compare(r, r2)
        @test occursin("do not share their steps", e.value.msg)
        r3 = cfe(Covariance(), rd, batch_cv; w = [fill(1 / N, N), fill(1 / N, N)])
        @test_throws DimensionMismatch covariance_forecast_compare(r, r3)
        @test_throws DomainError covariance_forecast_compare(r, r; lags = length(r.dates))
        @test_throws DomainError covariance_forecast_compare(r, r; lags = -1)
        # The re-projection.
        e = @test_throws ArgumentError covariance_forecast_portfolio(r, rd, nothing)
        @test occursin("store_forecasts", e.value.msg)
        rs = cfe(Covariance(), rd, batch_cv; store_forecasts = true)
        @test_throws po.IsNothingError covariance_forecast_portfolio(rs,
                                                                     ReturnsResult(;
                                                                                   nx = nx),
                                                                     nothing)
    end

    @testset "The loop's traits and the exports" begin
        @test !po.is_time_dependent(Covariance())
        @test !po.is_time_dependent(EmpiricalPrior())
        @test !po.is_time_dependent(po.Online(Covariance()))
        @test !po.needs_previous_weights(Covariance())
        @test !po.needs_previous_weights(EmpiricalPrior())
        @test !po.needs_previous_weights(po.Online(Covariance()))
        @test po.advance_previous_fold(nothing, :prev, (; a = 1)) == :prev
        @test isdefined(PortfolioOptimisers, :AbstractRealisedTarget)
        @test !(:AbstractRealisedTarget in names(PortfolioOptimisers))
        for nm in (:RealisedCovariance, :HorizonReturn, :covariance_forecast_step,
                   :covariance_forecast_evaluation, :CovarianceForecastEvaluationResult,
                   :covariance_forecast_summary, :CovarianceForecastSummaryResult,
                   :covariance_forecast_compare, :CovarianceForecastComparisonResult,
                   :covariance_forecast_portfolio)
            @test nm in names(PortfolioOptimisers)
        end
        @test RealisedCovariance() isa po.AbstractRealisedTarget
        @test HorizonReturn() isa po.AbstractRealisedTarget
    end
end
