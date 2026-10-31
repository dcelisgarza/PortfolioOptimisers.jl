using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, Dates, TimeSeries,
      Clarabel

# Gaps in the benchmark, the implied volatilities and the factor returns, issue #1672, under
# ADR 0117 amended for #1672. The online step drops the same observations, issue #1675.
#
# A gap in a Shared Series, a vector benchmark in the Excess Frame or a factor return that the fit
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

@testset "A gap in a vector benchmark in the Excess Frame drops the observation" begin
    Bg = copy(Bv0)
    Bg[gap] = NaN
    rdg = ReturnsResult(; nx = nx, X = X0, nb = ["BM"], B = Bg, ts = ts0)
    rdh = ReturnsResult(; nx = nx, X = X0[keep, :], nb = ["BM"], B = Bv0[keep],
                        ts = ts0[keep])
    # Before #1672 every asset left the Coverage Universe and the prior threw.
    epx = EmpiricalPrior(; x_src = ExcessSource())
    p1 = prior(epx, rdg)
    p2 = prior(epx, rdh)
    @test all(isfinite, p1.sigma)
    @test p1.mu ≈ p2.mu
    @test p1.sigma ≈ p2.sigma
    # A fit that does not read the benchmark keeps every observation.
    @test size(prior(EmpiricalPrior(), rdg).X, 1) == T
    # The optimiser drops the observation once, before every reader.
    opt = HierarchicalRiskParity(;
                                 opt = HierarchicalOptimiser(; pe = epx,
                                                             cle = ClustersEstimator(;
                                                                                     x_src = ExcessSource())))
    @test optimise(opt, rdg).w ≈ optimise(opt, rdh).w
    @test optimise(InverseVolatility(; pe = epx), rdg).w ≈
          optimise(InverseVolatility(; pe = epx), rdh).w
    # The view cuts every field at the same observations, and is idempotent.
    v = PO.shared_series_view(rdg, ExcessSource(), false)
    @test v.ts == ts0[keep]
    @test PO.shared_series_view(v, ExcessSource(), false) === v
    @test PO.shared_series_view(rdg, nothing, false) === rdg
    # A benchmark with a gap at every observation leaves the fit nothing.
    rde = ReturnsResult(; nx = nx, X = X0, nb = ["BM"], B = fill(NaN, T))
    @test_throws PO.IsEmptyError prior(epx, rde)
end

@testset "A head with no fallback drops a vector benchmark gap for a graph reader" begin
    # Before #1688 the direct method of a head with `fb = nothing` solved on the raw panel,
    # so a graph reader of the data met the gap rows and refused them. Since #1701 the
    # graph reader states its own source: `ExcessSource()` reads `X - B` from the data, and
    # `PriorSource()` reads the prior, which states `ExcessSource()`.
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
    epx = EmpiricalPrior(; x_src = ExcessSource())
    ho(xs, pe) = HierarchicalOptimiser(; pe = pe, cle = ClustersEstimator(; x_src = xs))
    jo(; kw...) = JuMPOptimiser(; pe = epx, slv = slv, kw...)
    heads = (("HierarchicalRiskParity, EmpiricalPrior",
              (xs, fb) -> HierarchicalRiskParity(; opt = ho(xs, epx), fb = fb)),
             ("HierarchicalRiskParity, FactorPrior",
              (xs, fb) -> HierarchicalRiskParity(;
                                                 opt = ho(xs,
                                                          FactorPrior(;
                                                                      x_src = ExcessSource())),
                                                 fb = fb)),
             ("HierarchicalEqualRiskContribution",
              (xs, fb) -> HierarchicalEqualRiskContribution(; opt = ho(xs, epx), fb = fb)),
             ("MeanRisk, a centrality constraint",
              (xs, fb) -> MeanRisk(; opt = jo(; cte = CentralityConstraint(; x_src = xs)),
                                   fb = fb)),
             ("NestedClustered",
              (xs, fb) -> NestedClustered(; pe = epx, cle = ClustersEstimator(; x_src = xs),
                                          opti = MeanRisk(; opt = jo()),
                                          opto = MeanRisk(; opt = jo()), fb = fb)))
    @testset "$name, x_src = $xs, fb = $(typeof(fb))" for (name, f) in heads,
                                                          xs in
                                                          (PriorSource(), ExcessSource()),
                                                          fb in
                                                          (nothing, InverseVolatility())

        res = optimise(f(xs, fb), rdg)
        @test isa(res.retcode, OptimisationSuccess)
        @test isnothing(res.fb)
        @test res.w ≈ optimise(f(xs, fb), rdh).w
    end
end

@testset "A covariance forecast evaluation in the Excess Frame drops the observation" begin
    Bg = copy(Bv0)
    Bg[gap] = NaN
    rdg = ReturnsResult(; nx = nx, X = X0, nb = ["BM"], B = Bg, ts = ts0)
    rdh = ReturnsResult(; nx = nx, X = X0[keep, :], nb = ["BM"], B = Bv0[keep],
                        ts = ts0[keep])
    cv = IndexWalkForward(20, 5)
    a = covariance_forecast_evaluation(Covariance(), rdg, cv; x_src = ExcessSource())
    b = covariance_forecast_evaluation(Covariance(), rdh, cv; x_src = ExcessSource())
    @test all(isfinite, a.mahalanobis_ratio)
    @test isequal(a.mahalanobis_ratio, b.mahalanobis_ratio)
    @test a.dates == b.dates
end

@testset "An online step drops an observation with a Shared Series gap (#1675)" begin
    # The step keeps the row in the buffer of the prior and in the Fold Context, and marks
    # it, so every fit through the buffer drops it. The call with no data equals the batch
    # fit over the rows folded so far, and the batch fit equals the fit with the gapped row
    # removed by hand.
    using Clarabel
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false))
    rows(r, i) = PO.port_opt_view(r, i, :)
    step(opt, r, t) = foldl((o, i) -> partial_fit!(o, rows(r, i:i)), 1:t; init = opt)
    online(e; kwargs...) = PO.update_online_estimator(PO.Online(e; kwargs...))
    t = 30
    k30 = setdiff(1:t, gap)
    Bg = copy(Bv0)
    Bg[gap] = NaN
    rdg = ReturnsResult(; nx = nx, X = X0, nb = ["BM"], B = Bg, ts = ts0)
    rdh = ReturnsResult(; nx = nx, X = X0[k30, :], nb = ["BM"], B = Bv0[k30], ts = ts0[k30])

    @testset "A gap of a vector benchmark in the Excess Frame, per family" begin
        epx = EmpiricalPrior(; x_src = ExcessSource())
        clx = ClustersEstimator(; x_src = ExcessSource())
        jopt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv)
        jexc = JuMPOptimiser(; pe = epx, slv = slv)
        hexc = HierarchicalOptimiser(; pe = epx, cle = clx)
        fams = (MeanRisk(; opt = jexc), HierarchicalRiskParity(; opt = hexc),
                HierarchicalEqualRiskContribution(; opt = hexc),
                InverseVolatility(; pe = epx),
                NestedClustered(; pe = epx, cle = clx, opti = MeanRisk(; opt = jopt),
                                opto = MeanRisk(; opt = jopt)),
                SubsetResampling(; pe = epx, opt = MeanRisk(; opt = jopt), seed = 3,
                                 n_subsets = 3, subset_size = 2))
        for opt in fams
            o = step(opt, rdg, t)
            solved = !isa(opt,
                          Union{HierarchicalRiskParity, HierarchicalEqualRiskContribution,
                                InverseVolatility})
            atol = solved ? 1e-8 : 1e-12
            @test isapprox(optimise(o).w, optimise(opt, rows(rdg, 1:t)).w; atol = atol)
            # The batch entry of an estimator with no fallback drops the row too (#1672).
            @test isapprox(optimise(opt, rows(rdg, 1:t)).w, optimise(opt, rdh).w;
                           atol = atol)
            # The Fold Context keeps every row, so the rebuilt data is the data folded.
            @test PO.returns_result(o).ts == ts0[1:t]
        end
        # The refit route marks the row in its buffer, and a rolling window counts every
        # row, as the training window of a batch walk-forward does.
        o = step(PO.update_online_estimator(InverseVolatility(; pe = PO.Online(epx))), rdg,
                 t)
        @test PO.buffer_gap_marks(o.pe.cache) == ((1:t) .== gap)
        @test PO.buffer_fit_count(o.pe.cache) == t - 1
        @test isapprox(optimise(o).w, optimise(InverseVolatility(; pe = epx), rdh).w;
                       atol = 1e-12)
        for w in (20, 10)
            iw = PO.update_online_estimator(InverseVolatility(;
                                                              pe = PO.Online(epx;
                                                                             max_history = w)))
            @test isapprox(optimise(step(iw, rdg, t)).w,
                           optimise(InverseVolatility(; pe = epx),
                                    rows(rdg, (t - w + 1):t)).w; atol = 1e-12)
        end
        # A scenario risk measure pairs the rows of the prior with the last rows of the
        # returns data, and a high order carry folds its co-moments.
        mk() = MeanRisk(;
                        opt = JuMPOptimiser(;
                                            pe = HighOrderPriorEstimator(;
                                                                         pe = EmpiricalPrior(),
                                                                         x_src = ExcessSource()),
                                            slv = slv), r = ConditionalValueatRisk())
        @test isapprox(optimise(step(mk(), rdg, t)).w, optimise(mk(), rdh).w; atol = 1e-8)
        # A first step whose every row has a gap folds nothing into the moments.
        B1 = copy(Bv0)
        B1[1] = NaN
        rd1 = ReturnsResult(; nx = nx, X = X0, nb = ["BM"], B = B1, ts = ts0)
        rh1 = ReturnsResult(; nx = nx, X = X0[2:t, :], nb = ["BM"], B = Bv0[2:t],
                            ts = ts0[2:t])
        @test isapprox(optimise(step(InverseVolatility(; pe = epx), rd1, t)).w,
                       optimise(InverseVolatility(; pe = epx), rh1).w; atol = 1e-12)
        # A group with no reader in the Excess Frame does not read the benchmark, so the
        # step marks nothing.
        o = step(PO.update_online_estimator(InverseVolatility(;
                                                              pe = PO.Online(EmpiricalPrior()))),
                 rdg, t)
        @test isnothing(o.pe.cache.G)
    end

    @testset "A gap of the factor returns, for a prior that reads them" begin
        Fg = copy(F0)
        Fg[gap, 1] = NaN
        rdf = ReturnsResult(; nx = nx, X = X0, nf = nf, F = Fg, ts = ts0)
        pstep(pe, r, t) = foldl((p, i) -> partial_fit!(p, rows(r, i:i)), 1:t; init = pe)
        p = prior(pstep(online(FactorPrior()), rdf, t))
        q = prior(FactorPrior(), rows(rdf, 1:t))
        @test p.sigma ≈ q.sigma
        @test p.mu ≈ q.mu
        @test size(p.X, 1) == t - 1
        @test p.rr.idx == q.rr.idx
        # A capped buffer counts every row.
        @test prior(pstep(online(FactorPrior(); max_history = 12), rdf, t)).sigma ≈
              prior(FactorPrior(), rows(rdf, (t - 11):t)).sigma
        # The optimiser over the prior, and a host whose co-moments fold.
        iv = PO.update_online_estimator(InverseVolatility(; pe = PO.Online(FactorPrior())))
        @test isapprox(optimise(step(iv, rdf, t)).w,
                       optimise(InverseVolatility(; pe = FactorPrior()), rows(rdf, 1:t)).w;
                       atol = 1e-12)
        h = prior(pstep(HighOrderPriorEstimator(; pe = online(FactorPrior())), rdf, t))
        hb = prior(HighOrderPriorEstimator(; pe = FactorPrior()), rows(rdf, 1:t))
        @test isapprox(h.sk, hb.sk; atol = 1e-15)
        @test isapprox(h.kt, hb.kt; atol = 1e-15)
        # A gap in the first row leaves the first step nothing to fit.
        F1 = copy(F0)
        F1[1, 2] = NaN
        rd1 = ReturnsResult(; nx = nx, X = X0, nf = nf, F = F1, ts = ts0)
        @test prior(pstep(online(FactorPrior()), rd1, t)).sigma ≈
              prior(FactorPrior(), rows(rd1, 1:t)).sigma
        # A buffer whose every row has a gap keeps no observation.
        Fe = fill(NaN, 3, K)
        rde = ReturnsResult(; nx = nx, X = X0[1:3, :], nf = nf, F = Fe, ts = ts0[1:3])
        @test_throws PO.IsEmptyError prior(partial_fit!(online(FactorPrior()), rde))
    end

    @testset "The marks of a buffer travel with its rows" begin
        s = partial_fit!(PO.SampleBufferState(; max_history = 4), [1.0 2; 3 4])
        @test isnothing(s.G)
        # A buffer that never met a gap costs what it cost before the marks existed.
        @test PO.shared_series_buffer(s) === s
        @test !PO.buffer_fits_none(s)
        # Measured through a function barrier, so the call itself is what is counted.
        allocs(f, x) = (f(x); @allocated f(x))
        @test allocs(PO.shared_series_buffer, s) == 0
        @test allocs(PO.buffer_fits_none, s) == 0
        @test allocs(PO.buffer_fit_count, s) == 0
        s = partial_fit!(s, [5.0 6; 7 8; 9 10]; gaps = [false, true, false])
        @test PO.buffer_gap_marks(s) == [false, false, true, false]
        @test PO.sample_buffer(PO.shared_series_buffer(s)) == [3.0 4; 5 6; 9 10]
        @test PO.shared_series_buffer(PO.shared_series_buffer(s)).n == 3
        m = PO.merge_states(s,
                            partial_fit!(PO.SampleBufferState(; max_history = 4), [1.0 1]))
        @test PO.buffer_gap_marks(m) == [false, true, false, false]
        @test PO.buffer_gap_marks(copy(s)) == PO.buffer_gap_marks(s)
        @test PO.buffer_gap_marks(PO.port_opt_view(s, [2])) == PO.buffer_gap_marks(s)
        @test_throws DimensionMismatch partial_fit!(s, [1.0 1]; gaps = [true, false])
        # A block whose marks are all false records none in a buffer that holds none.
        e = partial_fit!(PO.SampleBufferState(; max_history = 4), [1.0 2; 3 4])
        e = partial_fit!(e, [5.0 6; 7 8]; gaps = [false, false])
        @test isnothing(e.G) && e.n == 4
        # A block longer than the cap keeps the marks of the rows it keeps.
        cap = partial_fit!(PO.SampleBufferState(; max_history = 2), [1.0 2; 3 4; 5 6; 7 8];
                           gaps = [true, false, true, false])
        @test PO.buffer_gap_marks(cap) == [true, false]
        @test PO.sample_buffer(cap) == [5.0 6; 7 8]
        # A block with no marks, appended to a buffer that holds them, marks its rows as no gap.
        s = partial_fit!(s, [11.0 12])
        @test PO.buffer_gap_marks(s) == [false, true, false, false]
        @test PO.sample_buffer(s) == [5.0 6; 7 8; 9 10; 11 12]
    end

    @testset "The member rows of a gapped block carry the active mask" begin
        # The members of a carry fold the unmarked rows of the block and the same rows of the
        # active mask, so a block with a gap equals the block without it.
        n = 20
        Xg = X0[1:n, :]
        gp = falses(n)
        gp[7] = true
        rk = findall(!, gp)
        Mg = trues(n, N)
        Mg[3, 2] = false
        Mg[7, 1] = false
        Mg[12, 3] = false
        a = prior(partial_fit!(EmpiricalPrior(), Xg; gaps = gp, active_mask = Mg))
        b = prior(partial_fit!(EmpiricalPrior(), Xg[rk, :]; active_mask = Mg[rk, :]))
        # The asset whose only hole sits on the dropped row stays in the Coverage Universe,
        # and the two with a hole on a kept row leave it.
        @test isequal(isnan.(a.mu), [false, true, true])
        @test isapprox(a.mu, b.mu; nans = true)
        @test isapprox(a.sigma, b.sigma; nans = true)
        # With no mask the keyword is absent from what the members get, and the rows still go.
        c = prior(partial_fit!(EmpiricalPrior(), Xg; gaps = gp))
        d = prior(partial_fit!(EmpiricalPrior(), Xg[rk, :]))
        @test c.mu ≈ d.mu
        @test c.sigma ≈ d.sigma
        # The mask is read: the same rows without it keep every asset.
        @test all(isfinite, c.mu)
    end

    @testset "A fold of an empty block leaves a co-moment unchanged" begin
        for e in (Coskewness(), Cokurtosis())
            a = partial_fit!(e, X0[1:10, :])
            b = partial_fit!(a, zeros(0, N))
            @test b.cache === a.cache
        end
    end
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
    # A raw prior reads no benchmark, so it changes nothing.
    @test prior(EmpiricalPrior(), rdg).sigma ≈
          prior(EmpiricalPrior(), ReturnsResult(; nx = nx, X = X0)).sigma
    # In the Excess Frame the cell leaves the Coverage Universe as a gap in X does: the
    # asset is out, and the other assets equal the fit without it.
    epx = EmpiricalPrior(; x_src = ExcessSource())
    p1 = prior(epx, rdg)
    rdh = ReturnsResult(; nx = nx[1:2], X = X0[:, 1:2], nb = nx[1:2], B = Bm0[:, 1:2])
    p2 = prior(epx, rdh)
    @test all(isnan, p1.sigma[3, :])
    @test p1.sigma[1:2, 1:2] ≈ p2.sigma
    # In the Excess Frame the estimation mask marks the cell, for a cross-sectional reader.
    v = PO.shared_series_view(rdg, ExcessSource(), false)
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
    # The same holds for a prior over the plain default: its surface has a gap, and the
    # asset stays, with the moments of the fit on the data without the surface.
    rdg = ReturnsResult(; nx = nx, X = X0, iv = ivg, ivpa = [1.1, 1.2, 1.3])
    p4 = prior(EmpiricalPrior(), rdg)
    @test all(isfinite, p4.sigma)
    @test p4.sigma ≈ prior(EmpiricalPrior(), ReturnsResult(; nx = nx, X = X0)).sigma
    # The wrapped surface reads the gap in the surface through every wrapper above it.
    ce2 = PortfolioOptimisersCovariance(; ce = ce)
    p5 = prior(EmpiricalPrior(; ce = ce2), rdg)
    @test all(isnan, p5.sigma[2, :])
    @test p5.sigma[[1, 3], [1, 3]] ≈ p3.sigma[[1, 3], [1, 3]]
    # No estimator of the library holds a pair, so no verb reaches the method for a pair
    # (#1919). It reads the value the pair holds.
    @test PO.reads_implied_volatility("iv" => ImpliedVolatility(; ppy = 252))
    @test !PO.reads_implied_volatility("iv" => Covariance())
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

@testset "A Cross-Sectional Factor Prior carry drops a benchmark gap in the Excess Frame (#1675)" begin
    # The carry computes its exposures, its histories and its lags from the observations
    # that the batch fit keeps, and its buffer keeps every row with its mark.
    include(joinpath(@__DIR__, "parity_harness.jl"))
    rows(r, i) = PO.port_opt_view(r, i, :)
    rdp = parity_panel(; T = 120, N = 15, seed = 1377).rd
    mpass(f) = CompositeExposure(; descriptors = [Passthrough(; field = f)],
                                 outlier = nothing, scoring = nothing, family = "style")
    factors = ["market" => ConstantExposure(), "style1" => mpass("style1"),
               "style2" => mpass("style2")]
    est() = CrossSectionalFactorPrior(; lambda = 1, factors = factors, minra = 5,
                                      x_src = ExcessSource())
    Bp = randn(StableRNG(16750001), 120) ./ 200
    # One gap inside a block and one at the first row of a block.
    Bp[60] = NaN
    Bp[91] = NaN
    rdb = ReturnsResult(; nx = rdp.nx, X = rdp.X, nf = rdp.nf, F = rdp.F, nb = ["BM"],
                        B = Bp, ne = rdp.ne, E = rdp.E, ts = rdp.ts, iv = rdp.iv,
                        ivpa = rdp.ivpa, pnl = rdp.pnl)
    pe = foldl((p, (a, b)) -> PO.fold_prior(p, rows(rdb, a:b)),
               ((1, 50), (51, 90), (91, 120)); init = est())
    @test PO.buffer_fit_count(pe.cache.buf) == 118
    p = prior(pe)
    q = prior(est(), rdb)
    @test isequal(isnan.(p.sigma), isnan.(q.sigma))
    @test isapprox(filter(isfinite, p.sigma), filter(isfinite, q.sigma); atol = 1e-12)
    @test isapprox(filter(isfinite, p.mu), filter(isfinite, q.mu); atol = 1e-12)
    @test p.rr.idx == q.rr.idx
end

@testset "A prior records the observations of a Shared Series cut (#1744)" begin
    # After a cut, no part of a prior stated which observation of the returns data each row
    # of its `X` is. Three readers paired the wrong rows: a graph reader of the prior returns
    # with the raw returns data, an online fallback, and the row key of the prior of an
    # optimiser. The prior now records the dropped observations in `ss_gaps`.
    using Clarabel
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false))
    rows(r, i) = PO.port_opt_view(r, i, :)
    step(opt, r) = foldl((o, i) -> partial_fit!(o, rows(r, i:i)), 1:size(r.X, 1);
                         init = opt)
    Fg = copy(F0)
    Fg[gap, 1] = NaN
    # The surface is `NaN` at the dropped observation alone, so a reader that pairs it with
    # a row of the prior refuses it.
    ivg = copy(iv0)
    ivg[gap, 1] = NaN
    rdg = ReturnsResult(; nx = nx, X = X0, nf = nf, F = Fg, iv = ivg, ivpa = 1.1)
    pr = prior(FactorPrior(), rdg)

    @testset "The marks and the row key of a prior" begin
        @test findall(pr.ss_gaps) == [gap]
        @test length(pr.ss_gaps) == T
        @test pr.rr.idx == keep
        # A fit that drops nothing records nothing.
        @test isnothing(prior(FactorPrior(),
                              ReturnsResult(; nx = nx, X = X0, nf = nf, F = F0)).ss_gaps)
        @test isnothing(prior(EmpiricalPrior(), rdg).ss_gaps)
        # A high order prior answers the marks of the prior that it wraps.
        hp = prior(HighOrderPriorEstimator(; pe = FactorPrior()), rdg)
        @test hp.ss_gaps == pr.ss_gaps
        @test hp.rr.idx == keep
        # A view on the assets keeps the marks, which live on the observations.
        @test PO.port_opt_view(pr, 1:2).ss_gaps == pr.ss_gaps
        # A second cut composes with the first: the prior read the `T - 1` observations
        # that a first cut kept, and dropped one more.
        rd2 = rows(rdg, keep)
        F2 = copy(rd2.F)
        F2[3, 2] = NaN
        p2 = prior(FactorPrior(), ReturnsResult(; nx = nx, X = rd2.X, nf = nf, F = F2))
        @test findall(p2.ss_gaps) == [3]
        p3 = PO.shared_series_record(p2, keep, T)
        @test findall(p3.ss_gaps) == [3, gap]
        @test p3.rr.idx == setdiff(keep, 3)
    end

    @testset "The constructor checks the marks" begin
        Xs = X0[1:4, 1:2]
        mu = vec(Statistics.mean(Xs; dims = 1))
        sigma = Statistics.cov(Xs)
        @test isnothing(LowOrderPrior(; X = Xs, mu = mu, sigma = sigma).ss_gaps)
        @test LowOrderPrior(; X = Xs, mu = mu, sigma = sigma,
                            ss_gaps = [false, true, false, false, false]).ss_gaps ==
              [false, true, false, false, false]
        # Marks with no gap are refused, so the cut data is always shorter than the marks.
        @test_throws ArgumentError LowOrderPrior(; X = Xs, mu = mu, sigma = sigma,
                                                 ss_gaps = falses(5))
        # The rows of `X` are the last kept observations, so the marks keep at least that many.
        @test_throws DimensionMismatch LowOrderPrior(; X = Xs, mu = mu, sigma = sigma,
                                                     ss_gaps = [true, true, false, false])
    end

    @testset "A graph reader pairs iv by the marks" begin
        # The window is short enough that the regression of the 59 kept rows has more than
        # four windows.
        ive = ImpliedVolatility(; ppy = 252, alg = ImpliedVolatilityRegression(; ws = 10))
        clp = ClustersEstimator(; ce = ive, x_src = PriorSource())
        ref = clusterise(clp, pr; iv = ivg[keep, :], ivpa = 1.1)
        # Before #1744 the last rows of iv paired row `gap` of iv with row `gap - 1` of
        # `pr.X`, and the call raised an `IsNonFiniteError`.
        @test clusterise(clp, pr; rd = rdg, iv = rdg.iv, ivpa = 1.1).res.order ==
              ref.res.order
        pick = PO.implied_volatility_picker
        @test pick(pr, pr.X, ivg) == ivg[keep, :]
        n = 20
        Xn = pr.X[(end - n + 1):end, :]
        @test pick(pr, Xn, ivg) == ivg[keep[(end - n + 1):end], :]
        # The data after the cut has fewer rows than the marks, so its last rows pair.
        @test pick(pr, Xn, ivg[keep, :]) == ivg[keep[(end - n + 1):end], :]
        @test PO.prior_data_rows(rdg, T, n) == (T - n + 1):T
        # A head that holds the result cuts the marked observations, as a head that fits
        # the prior does.
        hrp(pe) = optimise(HierarchicalRiskParity(;
                                                  opt = HierarchicalOptimiser(; pe = pe,
                                                                              cle = clp)),
                           rdg)
        @test hrp(pr).w ≈ hrp(FactorPrior()).w
    end

    @testset "An optimiser maps the prior that it fits back to the data" begin
        # Before #1744 the optimiser cut `rd` before its prior, so the key counted the
        # positions of the cut data, `1:(T - 1)`.
        heads = (InverseVolatility(; pe = FactorPrior()),
                 MeanRisk(; opt = JuMPOptimiser(; pe = FactorPrior(), slv = slv)),
                 HierarchicalRiskParity(;
                                        opt = HierarchicalOptimiser(; pe = FactorPrior())),
                 HierarchicalEqualRiskContribution(;
                                                   opt = HierarchicalOptimiser(;
                                                                               pe = FactorPrior())))
        @testset "$(nameof(typeof(h)))" for h in heads
            res = optimise(h, rdg)
            @test res.pr.ss_gaps == pr.ss_gaps
            @test res.pr.rr.idx == keep
            @test res.pr.sigma ≈ pr.sigma
        end
        # A prior result keeps the record of the data that it read.
        res = optimise(InverseVolatility(; pe = pr), rdg)
        @test res.pr.ss_gaps == pr.ss_gaps
        @test res.pr.rr.idx == keep
        @test PO.intersect_kept_rows([1, 2, 4], [2, 3, 4]) == [2, 4]
    end

    @testset "An online fallback reads the rows of its own rule" begin
        rdf = ReturnsResult(; nx = nx, X = X0, nf = nf, F = Fg)
        # Infeasible bounds, so the fallback answers.
        bad = WeightBounds(; lb = fill(0.5, N), ub = ones(N))
        mk(pe) = MeanRisk(; opt = JuMPOptimiser(; pe = pe, slv = slv, wb = bad),
                          fb = InverseVolatility())
        b = optimise(mk(FactorPrior()), rdf)
        o = optimise(step(PO.update_online_estimator(mk(PO.Online(FactorPrior()))), rdf))
        # The fallback holds an EmpiricalPrior, which reads no factor return, so it reads
        # every observation in batch and online.
        @test b.w ≈ optimise(InverseVolatility(), rdf).w
        @test o.w ≈ b.w
        # The read-out of the head itself still cuts the marked observation.
        ok(pe) = MeanRisk(; opt = JuMPOptimiser(; pe = pe, slv = slv))
        ro = optimise(step(PO.update_online_estimator(ok(PO.Online(FactorPrior()))), rdf))
        rb = optimise(ok(FactorPrior()), rdf)
        @test isapprox(ro.w, rb.w; rtol = 1e-6)
        @test ro.pr.ss_gaps == rb.pr.ss_gaps
        @test ro.pr.rr.idx == rb.pr.rr.idx == keep
    end
end
