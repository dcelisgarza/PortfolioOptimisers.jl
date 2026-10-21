#=
The covariance forecast evaluation, issue #1023, against the decision of #873 (ADR 0143).

One verb through the one fold loop: `covariance_forecast_evaluation(est, rd, cv)` scores a
covariance estimator's, or a prior's, forecast on the test rows of every fold, in batch when
the walk-forward refits and online when it is an Online Scheme. Around the verb sit
the per-step kernel and its parity with the oracle's, the two realised
targets, the location the test rows are centred on, the online and rolling identities over a
panel with a listing and a delisting, the date form, the summary against the oracle's
summaries, the Diebold–Mariano–West comparison, the re-projection, and every refusal.

The kernel's parity literals were produced by the oracle's per-step kernel on the fixed
forecast and test rows below, with two test portfolios, once with every cell finite and once
with one cell missing; the summary literals by its summary verbs on a run of seven such
steps. Both sides read the same formula-built inputs, so no random stream is shared.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))
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
    online_cv = OnlineIndexWalkForward(w, h; purged_size = p)
    finite_max(a, b) = maximum(abs, filter(!isnan, a .- b))
    columns = (:mahalanobis_ratio, :diagonal_ratio, :qlike, :frobenius,
               :standardised_return, :portfolio_qlike)

    @testset "The kernel against the oracle, with and without a missing cell" begin
        A = [1.0 0.3 -0.2 0.1; 0.2 1.1 0.4 -0.3; -0.1 0.5 0.9 0.2; 0.3 -0.2 0.1 1.2]
        sig = (A * A') / 1e4 + Diagonal([1e-4, 2e-4, 1.5e-4, 1.2e-4])
        Zr = [sin(0.7 * t + 0.29 * i) / 100 + 0.004 * cos(0.11 * t * i)
              for t in 1:3, i in 1:4]
        Zn = copy(Zr)
        Zn[2, 3] = NaN
        W = [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]]
        # The oracle centres nothing, so its numbers are ours at a zero location under
        # its own target, the horizon return.
        k(Z, wt) = covariance_forecast_step(sig, Z, zeros(4), wt, HorizonReturn())
        # Measured maxrel 2.4e-16 at most, on every statistic of the four steps below.
        s = k(Zr, nothing)
        @test s.n_valid == 4
        @test parity_compare([s.mahalanobis_ratio], [0.978852966148661]; rtol = 1e-14,
                             name = "mahalanobis Zr").ok
        @test parity_compare([mean(s.diagonal_ratio)], [1.1925335096638512]; rtol = 1e-14,
                             name = "diagonal mean Zr").ok
        @test parity_compare(s.standardised_return, [1.9211917469757487]; rtol = 1e-14,
                             name = "stdret Zr").ok
        @test parity_compare(s.portfolio_qlike, [-6.949593962426171]; rtol = 1e-14,
                             name = "pqlike Zr").ok
        s = k(Zr, W)
        @test parity_compare(s.standardised_return, [1.894133508777383, 1.6253620587183681];
                             rtol = 1e-14, name = "stdret Zr W").ok
        @test parity_compare(s.portfolio_qlike, [-6.961786146396911, -7.110698408378284];
                             rtol = 1e-14, name = "pqlike Zr W").ok
        # One missing cell: the pairwise count scales the forecast, `H ⊙ Σ̂`, and the
        # oracle's numbers move with it.
        s = k(Zn, nothing)
        @test s.n_valid == 4
        @test parity_compare([s.mahalanobis_ratio], [0.9039713006494878]; rtol = 1e-14,
                             name = "mahalanobis Zn").ok
        @test parity_compare([mean(s.diagonal_ratio)], [1.0830084753657339]; rtol = 1e-14,
                             name = "diagonal mean Zn").ok
        @test parity_compare(s.standardised_return, [1.850608226250147]; rtol = 1e-14,
                             name = "stdret Zn").ok
        @test parity_compare(s.portfolio_qlike, [-7.149655216935746]; rtol = 1e-14,
                             name = "pqlike Zn").ok
        s = k(Zn, W)
        @test parity_compare(s.standardised_return,
                             [1.8276572778164961, 1.5382386137061885]; rtol = 1e-14,
                             name = "stdret Zn W").ok
        @test parity_compare(s.portfolio_qlike, [-7.156129366719016, -7.322586405112045];
                             rtol = 1e-14, name = "pqlike Zn W").ok
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
        # The whole-matrix losses, which the oracle does not compute, against their
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
        # `E[z] = c` is the bias `c' Σ̂⁻¹ c / N` the oracle's raw ratio carries.
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
        # mean, a centred exponentially weighted one on zero, a prior on the centre of its
        # `sigma`, which the default empirical prior publishes as its `mu`.
        @test po.forecast_location(Covariance(), Xm[1:60, :]) ≈ c
        @test po.forecast_location(GeneralCovariance(), Xm[1:60, :]) ≈ c
        ow = StatsBase.pweights(collect(1.0:60))
        @test po.forecast_location(GeneralCovariance(; w = ow), Xm[1:60, :]) ≈
              vec(mean(Xm[1:60, :], ow; dims = 1))
        @test po.forecast_location(PortfolioOptimisersCovariance(), Xm[1:60, :]) ≈ c
        @test po.forecast_location(ExpWeightedCovariance(; centring = PreCentred()),
                                   Xm[1:60, :]) == zeros(N)
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
        # The panel arm of the location turns the mask where the caller turns the sample.
        # A mask-aware estimator reads a mask shaped as its sample is, so an arm that
        # handed the panel's mask through unturned threw at `dims = 2` alone (issue #1249).
        # The panel spans the whole sample, so the window here is the whole sample too.
        Xgt = permutedims(Xg)
        for ce in (Covariance(; cvg = CoveragePolicy()),
                   ExpWeightedCovariance(; decay = 0.94, min_obs = 4),
                   RegimeAdjustedExpWeightedCovariance(; decay = 0.94, min_obs = 4,
                                                       regime_min_obs = 2))
            @test isequal(vec(po.forecast_location(ce, Xgt, rdg.pnl; dims = 2)),
                          vec(po.forecast_location(ce, Xg, rdg.pnl)))
        end
        # The fallback for an estimator with no location of its own is the finite column
        # mean of the window.
        @test po.forecast_location(NoLocationCovariance(), Xg[1:60, :])[3] ≈
              mean(Xg[31:60, 3])
    end

    @testset "A prior centres on the centre of its sigma, not on the mu it publishes" begin
        # Issue #1327. A shrunk mean moves the `mu` of an empirical prior and leaves its
        # `sigma`, which `ce` forms about the sample mean. The location follows `sigma`.
        sh = ShrunkExpectedReturns()
        rdv = po.port_opt_view(rd, 1:60, :)
        pe = EmpiricalPrior(; me = sh)
        c = po.forecast_location(pe, rdv)
        @test c ≈ vec(mean(X[1:60, :]; dims = 1))
        @test maximum(abs, c - prior(pe, rdv).mu) > 1e-5
        @test last(po.forecast_moments(pe, rd, 1:60)) == c
        @test po.forecast_location(EmpiricalPrior(), rdv) == prior(EmpiricalPrior(), rdv).mu
        ps = EmpiricalPrior(; ce = StatsBase.SimpleCovariance(), me = sh)
        @test po.forecast_location(ps, rdv) ≈ c
        @test po.forecast_location(HighOrderPriorEstimator(; pe = pe), rdv) ≈ c
        # The horizon arm maps the log location as `mu` maps the log mean.
        Xl = log1p.(X[1:60, :])
        ch = expm1.(5 * vec(mean(Xl; dims = 1)) + 5 * diag(cov(Xl)) / 2)
        @test po.forecast_location(EmpiricalPrior(; me = sh, horizon = 5), rdv) ≈ ch
        @test po.forecast_location(EmpiricalPrior(; horizon = 5), rdv) ≈
              prior(EmpiricalPrior(; horizon = 5), rdv).mu
        # A factor prior lifts the location of its factor prior.
        Fm = randn(StableRNG(1327), T, 2) ./ 100 .+ 0.0002
        rdf = ReturnsResult(; nx = nx, X = X, nf = ["F1", "F2"], F = Fm, ts = ts)
        rdfv = po.port_opt_view(rdf, 1:60, :)
        # Two factors give a James-Stein intensity that is not positive, and the default
        # clamp then returns the sample mean (#1784). Bayes-Stein shrinks two factors.
        fp = FactorPrior(;
                         pe = EmpiricalPrior(;
                                             me = ShrunkExpectedReturns(;
                                                                        alg = BayesStein())))
        fpr = prior(fp, rdfv)
        cf = fpr.rr.M * vec(mean(Fm[1:60, :]; dims = 1)) + fpr.rr.b
        @test po.forecast_location(fp, rdfv) ≈ cf
        @test maximum(abs, cf - fpr.mu) > 1e-7
        @test po.forecast_location(HighOrderFactorPriorEstimator(; pe = fp), rdfv) ≈ cf
        @test po.forecast_location(FactorPrior(), rdfv) == prior(FactorPrior(), rdfv).mu
        # A posterior forms its `sigma` about its `mu`.
        bl = BlackLittermanPrior(; pe = pe,
                                 views = LinearConstraintEstimator(; val = ["A1 == 0.001"]),
                                 sets = UniverseSets(; dict = Dict("nx" => nx)))
        @test po.forecast_location(bl, rdv) == prior(bl, rdv).mu
        # The data-less form reads the fold: a folded `ce`, a `ce` refitted over the
        # carried rows, the horizon arm, and a host whose wrapped posterior answers `mu`.
        @test po.forecast_location(po.partial_fit!(pe, X[1:60, :])) ≈ c
        @test po.forecast_location(po.partial_fit!(ps, X[1:60, :])) ≈ c
        @test po.forecast_location(po.partial_fit!(EmpiricalPrior(; me = sh, horizon = 5),
                                                   X[1:60, :])) ≈ ch
        hb = po.partial_fit!(HighOrderPriorEstimator(; pe = bl), X[1:60, :])
        @test po.forecast_location(hb) == prior(hb).mu
        # So the mean estimator no longer moves the evaluation of a `sigma` it never read.
        for cv in (batch_cv, online_cv)
            a = cfe(pe, rd, cv)
            b = cfe(EmpiricalPrior(), rd, cv)
            for f in columns
                @test finite_max(getfield(a, f), getfield(b, f)) == 0
            end
        end
        a = cfe(fp, rdf, batch_cv)
        b = cfe(FactorPrior(), rdf, batch_cv)
        for f in columns
            @test finite_max(getfield(a, f), getfield(b, f)) <= 1e-12
        end
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
                 (ExpWeightedCovariance(; centring = PreCentred()), rdg, 0.0),
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
        stepped = OnlineIndexWalkForward(w + p, h; purged_size = p)
        @test all(length.(split(rolling, rd).train_idx) .== w)
        for (est, r) in
            ((Covariance(), rd), (GeneralCovariance(), rd), (EmpiricalPrior(), rd),
             (Covariance(; cvg = CoveragePolicy()), rdg), (EmpiricalPrior(), rdg),
             (ExpWeightedCovariance(), rdg),
             (EmpiricalPrior(; me = ShrunkExpectedReturns(), horizon = 2), rd),
             (FactorPrior(; pe = EmpiricalPrior(; me = ShrunkExpectedReturns())),
              ReturnsResult(; nx = nx, X = X, nf = ["F1", "F2"],
                            F = randn(StableRNG(1327), T, 2) ./ 100, ts = ts)))
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

    @testset "A bare covariance estimator reads the implied volatilities (#1643)" begin
        # The regression reads windows of 20 rows and needs more than two of them, so the
        # first training window is longer than the one above.
        wv = 90
        cvb = IndexWalkForward(wv, h; purged_size = p, expand_train = true)
        cvo = OnlineIndexWalkForward(wv, h; purged_size = p)
        iv = 0.2 .+ abs.(randn(StableRNG(1643), T, N)) ./ 50
        ivpas = (1.1, collect(range(1.05, 1.2; length = N)))
        algs = (ImpliedVolatilityRegression(), ImpliedVolatilityPremium())
        poc(ce) = PortfolioOptimisersCovariance(; ce = ce)
        for alg in algs, ivpa in ivpas
            ce = ImpliedVolatility(; alg = alg, ppy = 252)
            r = ReturnsResult(; nx = nx, X = X, iv = iv, ivpa = ivpa, ts = ts)
            # Batch: the bare estimator answers what the prior that holds it answers.
            b = cfe(ce, r, cvb)
            @test length(b.dates) == 14
            for est in (EmpiricalPrior(; ce = ce), poc(ce))
                bb = cfe(est, r, cvb)
                for f in columns
                    @test finite_max(getfield(b, f), getfield(bb, f)) == 0
                end
            end
            # Online: the buffer of the composite records the surface beside the rows, so
            # the stepped run is the batch run.
            o = cfe(po.Online(poc(ce)), r, cvo)
            @test b.n_valid == o.n_valid
            for f in columns
                @test finite_max(getfield(b, f), getfield(o, f)) == 0
            end
            # The bare estimator carries a buffer of its own, so it steps with no composite
            # (#1653).
            ob = cfe(po.Online(ce), r, cvo)
            @test b.n_valid == ob.n_valid
            for f in columns
                @test finite_max(getfield(b, f), getfield(ob, f)) == 0
            end
        end
        r = ReturnsResult(; nx = nx, X = X, iv = iv, ivpa = 1.1, ts = ts)
        ce = ImpliedVolatility(; ppy = 252)
        # The rolling identity, through a capped buffer.
        b = cfe(ce, r, IndexWalkForward(wv + p, h; purged_size = p))
        o = cfe(po.Online(poc(ce); max_history = wv), r,
                OnlineIndexWalkForward(wv + p, h; purged_size = p))
        for f in columns
            @test finite_max(getfield(b, f), getfield(o, f)) == 0
        end
        # An estimator that reads no surface ignores it, in batch, in an exact fold and in a
        # buffer alike.
        for (est, cv) in
            ((Covariance(), cvb), (Covariance(), cvo), (po.Online(Covariance()), cvo))
            a = cfe(est, r, cv)
            z = cfe(est, rd, cv)
            for f in columns
                @test finite_max(getfield(a, f), getfield(z, f)) == 0
            end
        end
        # The buffer pins `ivpa` and the presence of `iv` at its first append.
        ob = po.partial_fit!(po.update_online_estimator(po.Online(poc(ce))),
                             po.port_opt_view(r, 1:wv, :))
        @test ob.cache.iv == iv[1:wv, :] && ob.cache.ivpa == 1.1
        @test_throws ArgumentError po.partial_fit!(ob, X[(wv + 1):(wv + 2), :];
                                                   iv = iv[(wv + 1):(wv + 2), :],
                                                   ivpa = 2.0)
        @test_throws ArgumentError po.partial_fit!(ob, X[(wv + 1):(wv + 2), :])
        o1 = po.partial_fit!(ob, X[wv + 1, :]; iv = iv[wv + 1, :], ivpa = 1.1)
        @test o1.cache.n == wv + 1 && po.sample_buffer(o1.cache) == X[1:(wv + 1), :]
        # The read-out runs over a view of the buffer, so it differs from a fit over a
        # contiguous copy by rounding alone.
        @test isapprox(cov(o1),
                       cov(poc(ce), X[1:(wv + 1), :]; iv = iv[1:(wv + 1), :], ivpa = 1.1);
                       rtol = 1e-12)
        # A covariance estimator of another package carries no buffer, and the refusal names
        # the composite.
        e = @test_throws ArgumentError po.Online(StatsBase.SimpleCovariance())
        @test occursin("PortfolioOptimisersCovariance", e.value.msg)
    end

    @testset "x_src = ExcessSource() evaluates a forecast of the returns in excess of the benchmark (#1658, #1704)" begin
        # A vector `B` is one benchmark for every asset, and a matrix `B` is one per asset.
        Bv = randn(StableRNG(1658), T) ./ 100
        Bm = randn(StableRNG(16580), T, N) ./ 100
        wt = [fill(1 / N, N), collect(1.0:N) ./ sum(1:N)]
        runs = ((Covariance(), batch_cv), (Covariance(), online_cv),
                (EmpiricalPrior(), batch_cv), (EmpiricalPrior(), online_cv),
                (po.Online(Covariance()), online_cv))
        for (nb, B) in ((["bm"], Bv), (["bm$i" for i in 1:N], Bm))
            rb = ReturnsResult(; nx = nx, X = X, nb = nb, B = B, ts = ts)
            # The evaluation of a hand-built `X - B`, which loses `B`.
            rx = ReturnsResult(; nx = nx, X = X .- B, ts = ts)
            for (est, cv) in runs
                a = cfe(est, rb, cv; x_src = ExcessSource(), store_forecasts = true)
                z = cfe(est, rx, cv; store_forecasts = true)
                # The Result records the frame that the verb read, `DataSource()` for `nothing`.
                @test a.x_src === ExcessSource() && z.x_src === DataSource()
                @test a.n_valid == z.n_valid
                for f in columns
                    @test finite_max(getfield(a, f), getfield(z, f)) == 0
                end
                @test a.sigma == z.sigma && a.location == z.location
                # The re-projection reads the test rows in the frame of the evaluation.
                @test isequal(covariance_forecast_portfolio(a, rb, wt),
                              covariance_forecast_portfolio(z, rx, wt))
            end
            # Under `x_src = nothing` and `DataSource()` the benchmark is not read.
            a = cfe(Covariance(), rb, batch_cv)
            z = cfe(Covariance(), rd, batch_cv)
            @test a.x_src === DataSource()
            @test cfe(Covariance(), rb, batch_cv; x_src = DataSource()).qlike == a.qlike
            for f in columns
                @test finite_max(getfield(a, f), getfield(z, f)) == 0
            end
        end
        # An `ImpliedVolatility` reads the surface of the assets beside `X - B`.
        wv = 90
        cvb = IndexWalkForward(wv, h; purged_size = p, expand_train = true)
        iv = 0.2 .+ abs.(randn(StableRNG(1643), T, N)) ./ 50
        ce = ImpliedVolatility(; ppy = 252)
        a = cfe(ce,
                ReturnsResult(; nx = nx, X = X, nb = ["bm"], B = Bv, iv = iv, ivpa = 1.1,
                              ts = ts), cvb; x_src = ExcessSource())
        z = cfe(ce, ReturnsResult(; nx = nx, X = X .- Bv, iv = iv, ivpa = 1.1, ts = ts),
                cvb)
        for f in columns
            @test finite_max(getfield(a, f), getfield(z, f)) == 0
        end
        # No excess return exists without a benchmark.
        e = @test_throws ArgumentError cfe(Covariance(), rd, batch_cv;
                                           x_src = ExcessSource())
        @test occursin("carries no benchmark", e.value.msg)
        rb = ReturnsResult(; nx = nx, X = X, nb = ["bm"], B = Bv, ts = ts)
        a = cfe(Covariance(), rb, batch_cv; x_src = ExcessSource(), store_forecasts = true)
        e = @test_throws ArgumentError covariance_forecast_portfolio(a, rd, nothing)
        @test occursin("carries no benchmark", e.value.msg)
        # Two evaluations in different frames do not compare.
        e = @test_throws ArgumentError covariance_forecast_compare(a,
                                                                   cfe(Covariance(), rb,
                                                                       batch_cv))
        @test occursin("read different returns", e.value.msg)
        @test occursin("x_src = ExcessSource()", e.value.msg)
        # `nothing` and `DataSource()` read one frame, so their evaluations compare.
        cmp = covariance_forecast_compare(cfe(Covariance(), rb, batch_cv),
                                          cfe(GeneralCovariance(), rb, batch_cv;
                                              x_src = DataSource()))
        @test cmp isa CovarianceForecastComparisonResult
        # The verb reads no prior, so it refuses `PriorSource()`.
        e = @test_throws ArgumentError cfe(Covariance(), rb, batch_cv;
                                           x_src = PriorSource())
        @test occursin("covariance_forecast_evaluation", e.value.msg)
        # A positional `Bool` in the place of the source builds no Result.
        vals = Any[getfield(a, f) for f in fieldnames(typeof(a))]
        @test fieldnames(typeof(a))[5] === :x_src
        vals[5] = true
        @test_throws MethodError CovarianceForecastEvaluationResult(vals...)
        # The verb picks the frame, so the estimator is an inner reader and states no source.
        # A prior in the Excess Frame would subtract `B` from the `X - B` that the verb holds.
        for est in (EmpiricalPrior(; x_src = ExcessSource()),
                    EmpiricalPrior(; x_src = DataSource()),
                    po.Online(EmpiricalPrior(; x_src = ExcessSource())))
            e = @test_throws ArgumentError cfe(est, rb, online_cv; x_src = ExcessSource())
            @test occursin("the `est` of `covariance_forecast_evaluation`", e.value.msg)
        end
    end

    @testset "A prior that states a source reads its location in its own frame (#1704)" begin
        # A shrunk mean makes the location read the returns through `ce`, so a location read
        # off the raw `X` would differ from the location in the Excess Frame.
        Bv = randn(StableRNG(1704), T) ./ 100
        rb = ReturnsResult(; nx = nx, X = X, nb = ["bm"], B = Bv, ts = ts)
        rx = ReturnsResult(; nx = nx, X = X .- Bv, ts = ts)
        sh = ShrunkExpectedReturns()
        c = po.forecast_location(EmpiricalPrior(; me = sh, x_src = ExcessSource()), rb)
        @test c ≈ po.forecast_location(EmpiricalPrior(; me = sh), rx)
        @test maximum(abs, c - po.forecast_location(EmpiricalPrior(; me = sh), rb)) > 1e-5
        # A gap in the vector benchmark drops its observation from the fit and the location.
        Bg = copy(Bv)
        Bg[[40, 97]] .= NaN
        keep = setdiff(1:T, [40, 97])
        rg = ReturnsResult(; nx = nx, X = X, nb = ["bm"], B = Bg, ts = ts)
        rk = ReturnsResult(; nx = nx, X = X[keep, :] .- Bv[keep], ts = ts[keep])
        @test po.forecast_location(EmpiricalPrior(; me = sh, x_src = ExcessSource()), rg) ≈
              po.forecast_location(EmpiricalPrior(; me = sh), rk)
    end

    @testset "A gap in a vector benchmark re-projects on the kept rows (#1691)" begin
        # The evaluation drops the observations with a gap in `B` before it splits the folds,
        # so `test_idx` counts the kept rows, and the re-projection must drop the same rows.
        Bv = randn(StableRNG(1658), T) ./ 100
        Bg = copy(Bv)
        Bg[[40, 97]] .= NaN
        keep = setdiff(1:T, [40, 97])
        wt = [fill(1 / N, N), collect(1.0:N) ./ sum(1:N)]
        rb = ReturnsResult(; nx = nx, X = X, nb = ["bm"], B = Bg, ts = ts)
        rx = ReturnsResult(; nx = nx, X = X[keep, :] .- Bv[keep], ts = ts[keep])
        for cv in (batch_cv, online_cv)
            a = cfe(Covariance(), rb, cv; x_src = ExcessSource(), store_forecasts = true,
                    w = wt)
            z = cfe(Covariance(), rx, cv; store_forecasts = true, w = wt)
            @test isequal(a.standardised_return, z.standardised_return)
            pa = covariance_forecast_portfolio(a, rb, wt)
            @test isequal(pa.standardised_return, a.standardised_return)
            @test isequal(pa.portfolio_qlike, a.portfolio_qlike)
        end
    end

    @testset "The date form" begin
        bd = DateWalkForward(w, h; period = Day(1), purged_size = p, expand_train = true)
        od = OnlineDateWalkForward(w, h; period = Day(1), purged_size = p)
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

    @testset "The summary against the oracle's summaries" begin
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
                                                  fill(3, M), HorizonReturn(), DataSource(),
                                                  [s.n_valid for s in steps],
                                                  [s.mahalanobis_ratio for s in steps],
                                                  stack(:diagonal_ratio),
                                                  [s.qlike for s in steps],
                                                  [s.frobenius for s in steps],
                                                  stack(:standardised_return),
                                                  stack(:portfolio_qlike), W, nothing,
                                                  nothing)
        # Measured maxrel 1.8e-16.
        @test parity_compare(cfer.mahalanobis_ratio,
                             [0.978852966148661, 0.4653960403261437, 0.011680925757934219,
                              0.315530814251327, 0.34876711552388395, 0.15922909197779334,
                              0.2726172060164678]; rtol = 1e-14,
                             name = "cfer mahalanobis ratio").ok
        s = covariance_forecast_summary(cfer)
        @test s.names == ["forecast_1"]
        # The oracle's `summary()` rows and its `bias_statistic_summary()`, at full digits:
        # its own step kernel and summaries, run on this closed-form fixture.
        ours = [s.mahalanobis_mean[1], s.mahalanobis_median[1], s.mahalanobis_p5[1],
                s.mahalanobis_p95[1], s.diagonal_mean[1], s.diagonal_median[1],
                s.diagonal_p5[1], s.diagonal_p95[1], s.portfolio_qlike_mean[1],
                s.bias_statistic[1], s.bias_p5[1], s.bias_p25[1], s.bias_p75[1],
                s.bias_p95[1]]
        oracle = [0.36458202285745867, 0.315530814251327, 0.05594537562389196,
                  0.8248158884019055, 0.42592521496695807, 0.36461964662163726,
                  0.05789099926930382, 0.9886089390858581, -7.520415207825884,
                  1.09034120664011, 1.0547936163414997, 1.0705925453631042,
                  1.1100898679171156, 1.1258887969387201]
        # Measured maxrel 2.4e-16.
        @test parity_compare(ours, oracle; name = "cfe summary").ok
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
                                                 RealisedCovariance(), DataSource(),
                                                 cfer.n_valid, cfer.mahalanobis_ratio,
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

    @testset "The comparison: a Mean Test on the loss difference (#1787)" begin
        bcv3 = IndexWalkForward(w, 3; expand_train = true)
        a = cfe(Covariance(), rd, bcv3; store_forecasts = true)
        b = cfe(ExpWeightedCovariance(; decay = 0.8), rd, bcv3)
        mt = WildBootstrapMeanTest(; seed = 1787)
        cmp = covariance_forecast_compare(a, b; mt = mt)
        @test cmp.names == ["qlike", "frobenius", "portfolio_qlike_1"]
        @test cmp.n_steps == length(a.dates)
        # On an i.i.d. panel the sample covariance is close to the truth and a fast decay
        # is noisy, so the first loses less: a negative mean difference, and a small `p`.
        @test all(cmp.mean_difference .< 0)
        @test all(cmp.t_stat .< 0)
        @test all(cmp.p_value .< 0.05)
        @test all(cmp.mean_ci_upper .< 0)
        # Each loss is the Mean Test of its difference at a window of one step, and the
        # generator is resolved once for the call, so the losses take different draws.
        ds = [a.qlike .- b.qlike, a.frobenius .- b.frobenius,
              a.portfolio_qlike[:, 1] .- b.portfolio_qlike[:, 1]]
        rng = po.mean_test_rng(mt)
        for (i, d) in enumerate(ds)
            @test (cmp.t_stat[i], cmp.p_value[i], cmp.mean_ci_lower[i],
                   cmp.mean_ci_upper[i]) == values(po.mean_test(mt, d, 1, rng))
        end
        @test covariance_forecast_compare(a, b; mt = mt).p_value == cmp.p_value
        # The rule of zero lags of Diebold and Mariano: the mean over its standard error.
        ov = covariance_forecast_compare(a, b; mt = OverlapMeanTest())
        @test ov.t_stat ≈ [mean(d) / (std(d) / sqrt(length(d))) for d in ds]
        @test ov.mean_difference == cmp.mean_difference
        # The same forecast twice: no difference, and no statistic.
        same = covariance_forecast_compare(a, a)
        @test all(iszero, same.mean_difference)
        @test all(isnan, same.t_stat) && all(isnan, same.p_value)
        @test all(isnan, same.mean_ci_lower) && all(isnan, same.mean_ci_upper)
        # The sign: swapping the arguments negates the statistic, and keeps the p-value.
        rev = covariance_forecast_compare(b, a; mt = mt)
        @test rev.t_stat ≈ -cmp.t_stat
        @test rev.p_value ≈ cmp.p_value
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
        # An online state updates its location in place, so each step stores its own copy
        # and the re-projection of an online run reproduces its columns too (#1028).
        o = cfe(Covariance(), rdg, online_cv; store_forecasts = true)
        @test o.location[1] !== o.location[2]
        @test o.location[1] != o.location[end]
        op = covariance_forecast_portfolio(o, rdg, nothing)
        @test op.standardised_return == o.standardised_return
        @test op.portfolio_qlike == o.portfolio_qlike
        # The stored forecasts of an online run are the batch run's, fold for fold, for
        # every family state that the fold writes in place.
        for est in (Covariance(), GeneralCovariance(), ExpWeightedCovariance(),
                    Covariance(; cvg = CoveragePolicy()))
            o = cfe(est, rd, online_cv; store_forecasts = true)
            b = cfe(est, rd, batch_cv; store_forecasts = true)
            @test length(unique(o.location)) == length(o.dates)
            @test maximum(maximum(abs, x - y) for (x, y) in zip(o.location, b.location)) <=
                  1e-12
            @test maximum(maximum(abs, x - y) for (x, y) in zip(o.sigma, b.sigma)) <= 1e-12
            pjo = covariance_forecast_portfolio(o, rd, wt)
            ro = cfe(est, rd, online_cv; w = wt)
            @test finite_max(pjo.standardised_return, ro.standardised_return) <= 1e-12
        end
    end

    @testset "The summary and the comparison at their edges (#1028)" begin
        # One step: the Mahalanobis columns are that step's ratio, and the bias statistic,
        # a sample standard deviation over the steps, is NaN with its percentiles.
        one = cfe(Covariance(), rd, IndexWalkForward(T - h, h))
        @test length(one.dates) == 1
        s1 = covariance_forecast_summary(one)
        @test s1.mahalanobis_mean == one.mahalanobis_ratio
        @test s1.mahalanobis_p5 == one.mahalanobis_ratio
        @test all(isnan,
                  (s1.bias_statistic[1], s1.bias_p5[1], s1.bias_p25[1], s1.bias_p75[1],
                   s1.bias_p95[1]))
        @test isfinite(s1.portfolio_qlike_mean[1])
        @test isnan(po.summary_quantile([1.0, NaN, 3.0], 0.5))
        # The `NaN` is read off the column, so it keeps the element type of the data.
        @test po.summary_quantile(Float32[1, NaN, 3], 0.5) isa Float32
        @test po.summary_quantile([1.0, 2.0, 3.0], 0.25) == quantile([1.0, 2.0, 3.0], 0.25)
        # A listing and a delisting change the active count under an index walk-forward,
        # so the Mahalanobis mean weights each step by N_t h and is not the plain mean. The
        # diagonal mean weights each step by h alone, so it is the plain mean.
        g = cfe(Covariance(), rdg, batch_cv)
        @test length(unique(g.n_valid)) > 1
        sg = covariance_forecast_summary(g)
        nu = g.n_valid .* g.horizon
        @test sg.mahalanobis_mean[1] ≈ dot(nu, g.mahalanobis_ratio) / sum(nu)
        @test !isapprox(sg.mahalanobis_mean[1], mean(g.mahalanobis_ratio); rtol = 1e-4)
        dbar = [mean(filter(isfinite, r)) for r in eachrow(g.diagonal_ratio)]
        @test sg.diagonal_mean[1] ≈ mean(dbar)
        # One step has no long-run variance, so its test is `NaN`; two steps answer.
        one_b = cfe(ExpWeightedCovariance(; decay = 0.8), rd, IndexWalkForward(T - h, h))
        c1 = covariance_forecast_compare(one, one_b)
        @test c1.n_steps == 1 && all(isnan, c1.p_value) && all(isfinite, c1.mean_difference)
        two = cfe(Covariance(), rd, IndexWalkForward(T - 2h, h))
        two_b = cfe(ExpWeightedCovariance(; decay = 0.8), rd, IndexWalkForward(T - 2h, h))
        @test length(two.dates) == 2
        @test all(isfinite, covariance_forecast_compare(two, two_b).t_stat)
        # A scored loss that is not finite is a defect of the data, so its test is `NaN`,
        # not a test of the other steps; the other losses keep their tests.
        bad = deepcopy(two)
        bad.qlike[1] = Inf
        cb = covariance_forecast_compare(bad, two_b)
        @test isinf(cb.mean_difference[1]) && isnan(cb.t_stat[1]) && isnan(cb.p_value[1])
        @test all(isfinite, cb.t_stat[2:end])
        # A step that a side did not score is missing at its position: the test reads the
        # series with a `NaN` there, and a `NaN` outside the scored steps is no defect.
        mt = WildBootstrapMeanTest(; seed = 3)
        d = [0.3, -0.1, NaN, 0.5, -0.4, 0.1, 0.2]
        k = .!isnan.(d)
        @test po.loss_difference_test(mt, d, k, po.mean_test_rng(mt)) ==
              po.mean_test(mt, d, 1, po.mean_test_rng(mt))
        @test po.loss_difference_test(mt, d, k, po.mean_test_rng(mt)) !=
              po.mean_test(mt, d[k], 1, po.mean_test_rng(mt))
        @test all(isnan,
                  values(po.loss_difference_test(mt, [0.1, Inf, 0.2], trues(3),
                                                 po.mean_test_rng(mt))))
        # The number type of the losses is kept, on the answer and on the `NaN` path.
        d32 = Float32[0.3, -0.1, 0.2, 0.5, -0.4]
        @test po.loss_difference_test(OverlapMeanTest(), d32, trues(5), nothing).t_stat isa
              Float32
        d32[2] = Inf32
        @test po.loss_difference_test(OverlapMeanTest(), d32, trues(5), nothing).t_stat isa
              Float32
    end

    @testset "The step weighting: the plain mean is one keyword away (#1513)" begin
        # A listing and a delisting move N_t, and a calendar walk-forward moves h_t, so the
        # two rules differ on both ratios.
        z = po.Distributions.cquantile(po.Distributions.Normal(), 0.025)
        for r in (cfe(Covariance(), rdg, batch_cv),
                  cfe(Covariance(), rd, DateWalkForward(2, 1; period = Month(1))))
            k = r.n_valid .> 0
            sd = covariance_forecast_summary(r)
            se = covariance_forecast_summary(r; step_weighting = EqualStepWeighting())
            @test sd.step_weighting === DofStepWeighting()
            @test se.step_weighting === EqualStepWeighting()
            sx = covariance_forecast_summary(r; step_weighting = DofStepWeighting())
            @test all(f -> isequal(getfield(sd, f), getfield(sx, f)),
                      fieldnames(typeof(sd)))
            m = r.mahalanobis_ratio[k]
            nu = po.target_dof.(Ref(r.target), r.n_valid[k], r.horizon[k])
            dbar = po.covariance_diagonal_mean(r)[k]
            snu = po.target_step_dof.(Ref(r.target), r.horizon[k])
            # The plain mean, and the band of the plain mean: Var = 2 sum(1 / nu_t) / M^2.
            @test se.mahalanobis_mean[1] ≈ mean(m) rtol = 1e-14
            @test se.diagonal_mean[1] ≈ mean(dbar) rtol = 1e-14
            M = length(m)
            @test se.mahalanobis_band_hi[1] - 1 ≈ z * sqrt(2 * sum(inv, nu)) / M rtol = 1e-14
            @test 1 - se.diagonal_band_lo[1] ≈ z * sqrt(2 * sum(inv, snu)) / M rtol = 1e-14
            # The default band is the chi-squared variance 2 / sum(nu), to the last bit.
            @test sd.mahalanobis_band_hi[1] == 1 + z * sqrt(2 / sum(nu))
            @test sd.diagonal_band_lo[1] == 1 - z * sqrt(2 / sum(snu))
            @test sd.mahalanobis_mean[1] == dot(nu, m) / sum(nu)
            # The plain mean has the wider band: sum(1/nu) / M^2 >= 1 / sum(nu).
            @test se.mahalanobis_band_hi[1] > sd.mahalanobis_band_hi[1]
            # Every other column reads no weight.
            for f in (:mahalanobis_median, :mahalanobis_p5, :diagonal_p95, :bias_statistic,
                      :qlike_mean, :exceedance, :n_steps)
                @test isequal(getfield(sd, f), getfield(se, f))
            end
        end
        # The two rules agree when every step has the same degrees of freedom.
        r = cfe(Covariance(), rd, batch_cv)
        @test allequal(r.n_valid) && allequal(r.horizon)
        @test covariance_forecast_summary(r; step_weighting = EqualStepWeighting()).mahalanobis_mean[1] ≈
              covariance_forecast_summary(r).mahalanobis_mean[1] rtol = 1e-14
        @test po.covariance_step_weights(DofStepWeighting(), [8, 8, 7]) == [8, 8, 7]
        @test po.covariance_step_weights(EqualStepWeighting(), [8, 8, 7]) == [1, 1, 1]
        @test DofStepWeighting() isa po.AbstractStepWeighting
        @test EqualStepWeighting() isa po.AbstractStepWeighting
    end

    @testset "A wrapper centres where the estimator it holds centres" begin
        # The panel arm of the wrapper's `cov` hands the panel to the estimator it holds, so a
        # mask-aware estimator inside it admits the young asset; the location follows.
        for inner in (Covariance(; cvg = CoveragePolicy()),
                      ExpWeightedCovariance(; decay = 0.94, min_obs = 4))
            poc = PortfolioOptimisersCovariance(; ce = inner)
            pw = po.port_opt_view(rdg, 1:60, :).pnl
            @test isequal(po.forecast_location(poc, Xg[1:60, :], pw),
                          po.forecast_location(inner, Xg[1:60, :], pw))
            @test cfe(poc, rdg, batch_cv).n_valid == cfe(inner, rdg, batch_cv).n_valid
        end
        # A StatsBase estimator inside a wrapper is read as the library reads it in `cov`.
        for ce in (PortfolioOptimisersCovariance(; ce = StatsBase.SimpleCovariance()),
                   po.CorrelationCovariance(; ce = StatsBase.SimpleCovariance()))
            @test po.forecast_location(ce, X[1:60, :]) ≈ vec(mean(X[1:60, :]; dims = 1))
            @test length(cfe(ce, rd, batch_cv).dates) ==
                  length(split(batch_cv, rd).test_idx)
        end
        @test isequal(po.forecast_location(PortfolioOptimisersCovariance(;
                                                                         ce = StatsBase.SimpleCovariance()),
                                           Xg[1:60, :], po.port_opt_view(rdg, 1:60, :).pnl),
                      po.forecast_location(GeneralCovariance(), Xg[1:60, :],
                                           po.port_opt_view(rdg, 1:60, :).pnl))
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
        # A step with no active asset is valid input with nothing to score (#1389): no test
        # return, or no forecast. It is unscored, and it still checks the portfolios.
        for (sg, Zs) in ((sig, fill(NaN, 2, N)), (fill(NaN, N, N), Z))
            u = covariance_forecast_step(sg, Zs, c, [ones(N), ones(N)],
                                         RealisedCovariance())
            @test u.n_valid == 0
            @test isnan(u.mahalanobis_ratio) && isnan(u.qlike) && isnan(u.frobenius)
            @test all(isnan, u.diagonal_ratio) && length(u.diagonal_ratio) == N
            @test all(isnan, u.standardised_return) && length(u.portfolio_qlike) == 2
            @test_throws DimensionMismatch covariance_forecast_step(sg, Zs, c, ones(3),
                                                                    RealisedCovariance())
        end
        # A non-finite location drops the asset from the active subset.
        cn = copy(c)
        cn[2] = NaN
        @test covariance_forecast_step(sig, Z, cn, nothing, RealisedCovariance()).n_valid ==
              N - 1
        # A singular forecast has no Cholesky factor.
        v1 = randn(StableRNG(3), N)
        @test_throws PosDefException covariance_forecast_step(v1 * v1', Z, c, nothing,
                                                              RealisedCovariance())
        # The verb.
        @test_throws po.IsNothingError cfe(Covariance(), ReturnsResult(; nx = nx), batch_cv)
        e = @test_throws ArgumentError cfe(po.Online(Covariance()), rd, batch_cv)
        @test occursin("not an Online Scheme", e.value.msg)
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
        # Two evaluations that share no scored step have no difference to test.
        r0 = deepcopy(r)
        r0.n_valid .= 0
        e = @test_throws po.IsEmptyError covariance_forecast_compare(r0, r)
        @test occursin("share no scored step", e.value.msg)
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
        @test !Base.isexported(PortfolioOptimisers, :AbstractRealisedTarget)
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
