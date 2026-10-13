using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, Dates, TimeSeries,
      Clarabel

# Gaps in the benchmark, the implied volatilities and the factor returns, issue #1672, under
# ADR 0117 amended for #1672.
#
# A gap in a Shared Series, a vector benchmark under `brt` or a factor return that the fit
# reads, drops the observation at the entry of the fit, and keeps every asset. A gap in a
# per-asset series, a matrix benchmark or the implied volatilities, acts as a gap in `X` at
# the same cell.
#
# The oracle of every testset is the same fit on the same panel with the gapped cells removed
# by hand: the observation for a Shared Series, the asset for a per-asset series.

const PO = PortfolioOptimisers

rng = StableRNG(16720001)
T, N, K = 60, 3, 2
X0 = randn(rng, T, N) ./ 100 .+ 0.0005
F0 = randn(rng, T, K) ./ 100
Bv0 = randn(rng, T) ./ 100
Bm0 = randn(rng, T, N) ./ 100
iv0 = 0.2 .+ 0.05 .* rand(rng, T, N)
ts0 = Date(2024, 1, 1) .+ Day.(0:(T - 1))
nx = ["a", "b", "c"]
nf = ["f1", "f2"]
gap = 17
keep = setdiff(1:T, gap)

@testset "A gap in a vector benchmark under brt drops the observation" begin
    Bg = copy(Bv0)
    Bg[gap] = NaN
    rdg = ReturnsResult(; nx = nx, X = X0, nb = ["BM"], B = Bg, ts = ts0)
    rdh = ReturnsResult(; nx = nx, X = X0[keep, :], nb = ["BM"], B = Bv0[keep],
                        ts = ts0[keep])
    # Before #1672 every asset left the Coverage Universe and the prior threw.
    p1 = prior(EmpiricalPrior(), rdg; brt = true)
    p2 = prior(EmpiricalPrior(), rdh; brt = true)
    @test all(isfinite, p1.sigma)
    @test p1.mu ≈ p2.mu
    @test p1.sigma ≈ p2.sigma
    # A fit that does not read the benchmark keeps every observation.
    @test size(prior(EmpiricalPrior(), rdg).X, 1) == T
    # The optimiser drops the observation once, before every reader.
    opt = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; brt = true))
    @test optimise(opt, rdg).w ≈ optimise(opt, rdh).w
    @test optimise(InverseVolatility(; brt = true), rdg).w ≈
          optimise(InverseVolatility(; brt = true), rdh).w
    # The view cuts every field at the same observations, and is idempotent.
    v = PO.shared_series_view(rdg, true, false)
    @test v.ts == ts0[keep]
    @test PO.shared_series_view(v, true, false) === v
    @test PO.shared_series_view(rdg, false, false) === rdg
    # A benchmark with a gap at every observation leaves the fit nothing.
    rde = ReturnsResult(; nx = nx, X = X0, nb = ["BM"], B = fill(NaN, T))
    @test_throws PO.IsEmptyError prior(EmpiricalPrior(), rde; brt = true)
end

@testset "A head with no fallback drops a vector benchmark gap under x_src = :data" begin
    # Before #1688 the direct method of a head with `fb = nothing` solved on the raw panel,
    # so a reader of the data under `x_src = :data` met the gap rows and refused them.
    rng = StableRNG(1679)
    Tg, Ng = 120, 6
    Fg = 0.01 .* randn(rng, Tg, 2)
    Lg = 0.5 .+ rand(rng, Ng, 2)
    Xg = Fg * transpose(Lg) .+ 0.008 .* randn(rng, Tg, Ng) .+ 0.0005
    Bg = 0.9 .* Fg[:, 1] .+ 0.002 .* randn(rng, Tg)
    kg = setdiff(1:Tg, [10, 47])
    nxg = ["A$i" for i in 1:Ng]
    tsg = Date(2020, 1, 1) .+ Day.(0:(Tg - 1))
    Bn = copy(Bg)
    Bn[[10, 47]] .= NaN
    rdg = ReturnsResult(; nx = nxg, X = Xg, nf = nf, F = Fg, nb = ["BM"], B = Bn, ts = tsg)
    rdh = ReturnsResult(; nx = nxg, X = Xg[kg, :], nf = nf, F = Fg[kg, :], nb = ["BM"],
                        B = Bg[kg], ts = tsg[kg])
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))
    ho(xs, pe) = HierarchicalOptimiser(; pe = pe, x_src = xs, brt = true)
    jo(xs; kw...) = JuMPOptimiser(; slv = slv, x_src = xs, brt = true, kw...)
    heads = (("HierarchicalRiskParity, EmpiricalPrior",
              (xs, fb) -> HierarchicalRiskParity(; opt = ho(xs, EmpiricalPrior()), fb = fb)),
             ("HierarchicalRiskParity, FactorPrior",
              (xs, fb) -> HierarchicalRiskParity(; opt = ho(xs, FactorPrior()), fb = fb)),
             ("HierarchicalEqualRiskContribution",
              (xs, fb) -> HierarchicalEqualRiskContribution(;
                                                            opt = ho(xs, EmpiricalPrior()),
                                                            fb = fb)),
             ("MeanRisk, a centrality constraint",
              (xs, fb) -> MeanRisk(; opt = jo(xs; cte = CentralityConstraint()), fb = fb)),
             ("NestedClustered",
              (xs, fb) -> NestedClustered(; x_src = xs, brt = true,
                                          opti = MeanRisk(; opt = jo(xs)),
                                          opto = MeanRisk(; opt = jo(xs)), fb = fb)))
    @testset "$name, x_src = :$xs, fb = $(typeof(fb))" for (name, f) in heads,
                                                           xs in (:prior, :data),
                                                           fb in
                                                           (nothing, InverseVolatility())

        res = optimise(f(xs, fb), rdg)
        @test isa(res.retcode, OptimisationSuccess)
        @test isnothing(res.fb)
        @test res.w ≈ optimise(f(xs, fb), rdh).w
    end
end

@testset "A covariance forecast evaluation under brt drops the observation" begin
    Bg = copy(Bv0)
    Bg[gap] = NaN
    rdg = ReturnsResult(; nx = nx, X = X0, nb = ["BM"], B = Bg, ts = ts0)
    rdh = ReturnsResult(; nx = nx, X = X0[keep, :], nb = ["BM"], B = Bv0[keep],
                        ts = ts0[keep])
    cv = IndexWalkForward(20, 5)
    a = covariance_forecast_evaluation(Covariance(), rdg, cv; brt = true)
    b = covariance_forecast_evaluation(Covariance(), rdh, cv; brt = true)
    @test all(isfinite, a.mahalanobis_ratio)
    @test isequal(a.mahalanobis_ratio, b.mahalanobis_ratio)
    @test a.dates == b.dates
end

@testset "An online step refuses an observation with a Shared Series gap" begin
    # A batch fit drops the observation. A step cannot, because the buffer of the prior and
    # the Fold Context hold one row each for every observation (#1675).
    Bg = copy(Bv0)
    Bg[gap] = NaN
    rdg = ReturnsResult(; nx = nx, X = X0, nb = ["BM"], B = Bg, ts = ts0)
    rows(r, i) = PO.port_opt_view(r, i, :)
    step(opt, r, t) = foldl((o, i) -> partial_fit!(o, rows(r, i:i)), 1:t; init = opt)
    opt = InverseVolatility(; brt = true)
    o = step(opt, rdg, gap - 1)
    @test isapprox(optimise(o).w, optimise(opt, rows(rdg, 1:(gap - 1))).w; atol = 1e-10)
    @test_throws PO.IsNonFiniteError partial_fit!(o, rows(rdg, gap:gap))
    # Without brt the step does not read the benchmark, so it folds the observation.
    @test !isnothing(partial_fit!(step(InverseVolatility(), rdg, gap - 1),
                                  rows(rdg, gap:gap)).cache)
    # A prior that reads the factor returns refuses a gap in them at its own step.
    Fg = copy(F0)
    Fg[gap, 1] = NaN
    rdf = ReturnsResult(; nx = nx, X = X0, nf = nf, F = Fg, ts = ts0)
    pe = PO.update_online_estimator(PO.Online(FactorPrior()))
    @test_throws PO.IsNonFiniteError partial_fit!(pe, rdf)
end

@testset "A gap in the factor returns drops the observation" begin
    Fg = copy(F0)
    Fg[gap, 2] = NaN
    rdg = ReturnsResult(; nx = nx, X = X0, nf = nf, F = Fg, ts = ts0)
    rdh = ReturnsResult(; nx = nx, X = X0[keep, :], nf = nf, F = F0[keep, :],
                        ts = ts0[keep])
    # Before #1672 the factor took `NaN` moments and the row reached the regression.
    re = FactorPrior().re
    q1 = prior(FactorPrior(; re = re), rdg)
    q2 = prior(FactorPrior(; re = re), rdh)
    @test all(isfinite, q1.sigma)
    @test q1.mu ≈ q2.mu
    @test q1.sigma ≈ q2.sigma
    # The row key names the observations that the fit read.
    @test q1.rr.ts == ts0[keep]
    # The regression over a ReturnsResult drops the same observation, and the regression
    # over matrices refuses it by name, because its caller chose the observations.
    @test regression(re, rdg).M ≈ regression(re, rdh).M
    @test_throws PO.IsNonFiniteError regression(re, X0, Fg)
    # A prior that does not read the factor returns keeps every observation.
    @test size(prior(EmpiricalPrior(), rdg).X, 1) == T
    # The optimiser reads the tree of its prior.
    opt = HierarchicalRiskParity(;
                                 opt = HierarchicalOptimiser(; pe = FactorPrior(; re = re)))
    @test optimise(opt, rdg).w ≈ optimise(opt, rdh).w
end

@testset "A gap in a matrix benchmark acts as a gap in X" begin
    Bg = copy(Bm0)
    Bg[gap, 3] = NaN
    amsk = trues(T, N)
    pnl = AssetPanel(; amsk = amsk, emsk = copy(amsk))
    rdg = ReturnsResult(; nx = nx, X = X0, nb = nx, B = Bg, ts = ts0, pnl = pnl)
    # With brt = false no fit reads the benchmark, so it changes nothing.
    @test prior(EmpiricalPrior(), rdg).sigma ≈
          prior(EmpiricalPrior(), ReturnsResult(; nx = nx, X = X0)).sigma
    # Under brt the cell leaves the Coverage Universe as a gap in X does: the asset is out,
    # and the other assets equal the fit without it.
    p1 = prior(EmpiricalPrior(), rdg; brt = true)
    rdh = ReturnsResult(; nx = nx[1:2], X = X0[:, 1:2], nb = nx[1:2], B = Bm0[:, 1:2])
    p2 = prior(EmpiricalPrior(), rdh; brt = true)
    @test all(isnan, p1.sigma[3, :])
    @test p1.sigma[1:2, 1:2] ≈ p2.sigma
    # Under brt the estimation mask marks the cell, for a cross-sectional reader.
    v = PO.shared_series_view(rdg, true, false)
    @test !v.pnl.emsk[gap, 3]
    @test count(!, v.pnl.emsk) == 1
    @test v.pnl.amsk == amsk
    # A product with weights reads the cells that the portfolio reads. Before #1672 a gap
    # at an asset of zero weight gave a `NaN` outer benchmark, because `NaN * 0` is `NaN`.
    wi = [0.5 0.0; 0.5 0.4; 0.0 0.6]
    _, Bo = PO.prepare_outer_rd(rdg, wi, RenormaliseActive())
    Bz = copy(Bg)
    Bz[gap, 3] = 0
    @test all(isfinite, Bo)
    @test Bo ≈ Bz * wi
    @test PO.benchmark_cells(rdg) == Bz
    # A gap in X at the cell is read the same way, and an inactive cell too.
    Xg = copy(X0)
    Xg[gap + 1, 1] = NaN
    amsk2 = trues(T, N)
    amsk2[gap + 2, 2] = false
    rdx = ReturnsResult(; nx = nx, X = Xg, nb = nx, B = Bm0,
                        pnl = AssetPanel(; amsk = amsk2, emsk = amsk2 .& isfinite.(Xg)))
    Bx = copy(Bm0)
    Bx[gap + 1, 1] = 0
    Bx[gap + 2, 2] = 0
    @test PO.benchmark_cells(rdx) == Bx
    # A vector benchmark is a series of the whole portfolio, so it passes through.
    rdv = ReturnsResult(; nx = nx, X = X0, nb = ["BM"], B = Bv0)
    @test PO.benchmark_cells(rdv) === rdv.B
end

@testset "A wrapped ImpliedVolatility on a gapped panel" begin
    # `CorrelationCovariance` has no panel method of its own, so it reaches the generic root.
    ce = CorrelationCovariance(;
                               ce = ImpliedVolatility(; ppy = 252,
                                                      alg = ImpliedVolatilityPremium()))
    pe = EmpiricalPrior(; ce = ce)
    # A gap in X: the generic root reduces `iv` with `X`. Before #1672 the wrapped estimator
    # read three columns of `iv` against two of `X`, and threw a DimensionMismatch.
    Xg = copy(X0)
    Xg[gap, 3] = NaN
    s1 = Statistics.cov(ce, Xg, nothing; iv = iv0, ivpa = 1.2)
    s2 = Statistics.cov(ce, X0[:, 1:2]; iv = iv0[:, 1:2], ivpa = 1.2)
    @test all(isnan, s1[3, :])
    @test s1[1:2, 1:2] ≈ s2
    p1 = prior(pe, ReturnsResult(; nx = nx, X = Xg, iv = iv0, ivpa = 1.2))
    p2 = prior(pe,
               ReturnsResult(; nx = nx[1:2], X = X0[:, 1:2], iv = iv0[:, 1:2], ivpa = 1.2))
    @test p1.sigma[1:2, 1:2] ≈ p2.sigma
    # A gap in iv: the asset leaves the Coverage Universe of a fit that reads the surface,
    # and a vector premium takes the same slice. Before #1672 the fit threw.
    ivg = copy(iv0)
    ivg[gap, 2] = NaN
    s3 = Statistics.cov(ce, X0, nothing; iv = ivg, ivpa = [1.1, 1.2, 1.3])
    s4 = Statistics.cov(ce, X0[:, [1, 3]]; iv = iv0[:, [1, 3]], ivpa = [1.1, 1.3])
    @test all(isnan, s3[2, :])
    @test s3[[1, 3], [1, 3]] ≈ s4
    p3 = prior(pe, ReturnsResult(; nx = nx, X = X0, iv = ivg, ivpa = [1.1, 1.2, 1.3]))
    @test all(isnan, p3.sigma[2, :])
    # A fit that does not read the surface keeps the asset.
    @test all(isfinite, Statistics.cov(Covariance(), X0, nothing; iv = ivg))
    @test PO.reads_implied_volatility(pe)
    @test !PO.reads_implied_volatility(EmpiricalPrior())
end

@testset "A collapse of the implied volatilities reads the finite active cells" begin
    w = [0.5, 0.25, 0.25]
    ivg = copy(iv0)
    ivg[gap, 2] = NaN
    r = PO.collapse_fold_rate(ivg, w, nothing)
    @test all(isfinite, r)
    # The gap counts as an inactive member of its row, and the row renormalises.
    @test r[gap] ≈ (0.5 * ivg[gap, 1] + 0.25 * ivg[gap, 3]) / 0.75
    @test r[keep] ≈ ivg[keep, :] * w
    # The meta-optimiser collapse and the fold collapse agree.
    W = reshape(w, :, 1)
    @test vec(PO.collapse_rate(ivg, W, nothing, RenormaliseActive())) ≈ r
    amsk = trues(T, N)
    amsk[gap + 1, 1] = false
    pnl = AssetPanel(; amsk = amsk, emsk = copy(amsk))
    r2 = PO.collapse_fold_rate(iv0, w, pnl)
    @test r2[gap + 1] ≈ (0.25 * iv0[gap + 1, 2] + 0.25 * iv0[gap + 1, 3]) / 0.5
end

@testset "MissingDataFilter counts the gaps of the benchmark, the factors and iv" begin
    d = Date(2020, 1, 1):Day(1):Date(2020, 1, 6)
    Xp = TimeArray(d, [100.0 50; 101 51; 102 52; 103 53; 104 54; 105 55], ["A", "B"])
    Bp = TimeArray(d, [10.0, 11, NaN, 13, 14, 15], ["BM"])
    Fp = TimeArray(d, [1.0, 2, 3, 4, NaN, 6], ["f"])
    ivp = TimeArray(d, [0.2 0.3; 0.2 NaN; 0.2 0.3; 0.2 0.3; 0.2 0.3; 0.2 0.3], ["A", "B"])
    pr = PricesResult(; X = Xp, B = Bp, F = Fp, iv = ivp)
    # A gap in a one-column benchmark or a factor misses every asset of its observation.
    @test PO.shared_gap_rows(pr) == [false, false, true, false, true, false]
    res = fit_preprocessing(MissingDataFilter(; row_thr = 0.5), pr)
    out = apply_preprocessing(res, pr)
    @test timestamp(out.X) == d[[1, 2, 4, 6]]
    # The default threshold drops no observation, so the default output does not change.
    @test timestamp(apply_preprocessing(fit_preprocessing(MissingDataFilter(), pr), pr).X) ==
          collect(d)
    # A gap in the implied volatility of an asset is a missing observation of that asset.
    @test fit_preprocessing(MissingDataFilter(; col_thr = 0.1), pr).nx == [:A]
    @test fit_preprocessing(MissingDataFilter(; col_thr = 0.1), PricesResult(; X = Xp)).nx ==
          [:A, :B]
    # So does a gap in the own column of a per-asset benchmark.
    Bm = TimeArray(d, [10.0 20; 11 21; 12 22; 13 23; 14 NaN; 15 25], ["BA", "BB"])
    @test fit_preprocessing(MissingDataFilter(; col_thr = 0.1),
                            PricesResult(; X = Xp, B = Bm)).nx == [:A]
    @test !any(PO.shared_gap_rows(PricesResult(; X = Xp, B = Bm)))
    # A series that holds no entry at a timestamp of X has a gap there.
    Fs = TimeArray(d[[1, 2, 3, 5, 6]], [1.0, 2, 3, 5, 6], ["f"])
    @test PO.shared_gap_rows(PricesResult(; X = Xp, F = Fs)) ==
          [false, false, false, true, false, false]
    # The stepped filter counts the same cells.
    mdf, _ = PO.partial_fit_transform(MissingDataFilter(; col_thr = 0.1), pr)
    @test fit_preprocessing(mdf).nx == [:A]
end
