#=
The online step of the optimiser layer, issue #1007, against the decision of #867.

Two verbs and one forward: `partial_fit!(opt, rd₁)` folds an observation into the prior and
records the rest of the carrier in a `ReturnsBufferState`; `optimise(opt)` rebuilds the
carrier from the state, swaps the folded prior for its read-out and runs the ordinary batch
path. #867 §10 names the three identities this file pins, at three levels, so a failure
names which broke: the reconstitution is exact, the read-out is pure, and the weights agree
with batch per family.

The fixture is synthetic, because the identities are structural and a solver run over eight
assets is what keeps the JuMP families cheap.
=#
@testset "Optimiser partial fit: the step folds, the read-out runs the batch path" begin
    using Test, PortfolioOptimisers, Clarabel, StableRNGs, Statistics, Dates, LinearAlgebra
    po = PortfolioOptimisers
    rng = StableRNG(20260911)
    T, N, K = 120, 8, 3
    X = randn(rng, T, N) ./ 100 .+ 0.0003
    F = randn(rng, T, K) ./ 100
    Bm = randn(rng, T, N) ./ 100
    Bv = randn(rng, T) ./ 100
    nx = ["A$i" for i in 1:N]
    nf = ["F$i" for i in 1:K]
    ts = Date(2024, 1, 1) .+ Day.(0:(T - 1))
    rd = ReturnsResult(; nx = nx, X = X, ts = ts)
    rdf = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, nb = nx, B = Bm, ts = ts)
    rdv = ReturnsResult(; nx = nx, X = X, nb = ["BM"], B = Bv, ts = ts)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = false, allow_almost = false),
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                 "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10))
    rows(r, i) = po.port_opt_view(r, i, :)
    step(opt, r, t) = foldl((o, i) -> partial_fit!(o, rows(r, i:i)), 1:t; init = opt)
    same_fields(a, b) = all(f -> isequal(getfield(a, f), getfield(b, f)),
                            (:nx, :X, :nf, :F, :nb, :B, :ts, :iv, :ivpa))

    @testset "The reconstitution is exact, field by field" begin
        for r in (rd, rdf, rdv), t in (12, 40, 80)
            o = step(MeanRisk(; opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv)), r,
                     t)
            rdp = po.returns_result(o)
            @test same_fields(rdp, rows(r, 1:t))
            @test isnothing(rdp.pnl)
            # A block warm-up is the same state as a row-by-row one.
            blk = partial_fit!(MeanRisk(;
                                        opt = JuMPOptimiser(; pe = EmpiricalPrior(),
                                                            slv = slv)), rows(r, 1:t))
            @test same_fields(po.returns_result(blk), rdp)
            # The carrier is materialised: a later step leaves it alone.
            o2 = partial_fit!(o, rows(r, (t + 1):(t + 1)))
            @test same_fields(rdp, rows(r, 1:t))
            @test same_fields(po.returns_result(o2), rows(r, 1:(t + 1)))
        end
        # The prior-less heads own the rows themselves.
        for opt in (EqualWeighted(), RandomWeighted(; seed = 1))
            o = step(opt, rdf, 40)
            @test !isnothing(o.cache.X)
            @test same_fields(po.returns_result(o), rows(rdf, 1:40))
        end
        # A host with a prior owns none: the rows live in the prior's buffer.
        o = step(InverseVolatility(), rd, 40)
        @test isnothing(o.cache.X)
        @test po.returns_result(o).X == X[1:40, :]
        # A single-column benchmark beside a prior: the context counts by the benchmark.
        o = step(InverseVolatility(), rdv, 10)
        @test po.context_count(o.cache) == 10
        @test same_fields(po.returns_result(o), rows(rdv, 1:10))
    end

    @testset "The read-out is pure" begin
        o = step(MeanRisk(; opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv)), rd,
                 60)
        before = copy(po.partial_fit_cache(o.opt.pe))
        r1 = optimise(o)
        r2 = optimise(o)
        @test r1.w == r2.w
        after = po.partial_fit_cache(o.opt.pe)
        @test po.sample_buffer(after) == po.sample_buffer(before)
        @test after.buf.n == before.buf.n
        # The read-out hands back a fresh estimator and leaves the folded one folded.
        @test !isnothing(o.opt.cache)
        o3 = partial_fit!(o, rows(rd, 61:61))
        @test po.partial_fit_cache(o3.opt.pe).buf.n == 61
    end

    @testset "The weights agree with batch, per family" begin
        t = 80
        b = rows(rd, 1:t)
        jopt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv)
        hopt = HierarchicalOptimiser(; pe = EmpiricalPrior())
        fams = (MeanRisk(; opt = jopt), RiskBudgeting(; opt = jopt),
                HierarchicalRiskParity(; opt = hopt),
                HierarchicalEqualRiskContribution(; opt = hopt), InverseVolatility(),
                EqualWeighted(), RandomWeighted(; seed = 7),
                NestedClustered(; opti = MeanRisk(; opt = jopt),
                                opto = MeanRisk(; opt = jopt)),
                SubsetResampling(; opt = MeanRisk(; opt = jopt), seed = 3, n_subsets = 3,
                                 subset_size = 5))
        for opt in fams
            o = step(opt, rd, t)
            ro = optimise(o)
            rb = optimise(opt, b)
            # A solver reads the folded moments through its own tolerance, so the JuMP
            # families agree to a solver's precision and the rest to the moments' own.
            # `RiskBudgeting` is the knife edge of the family: two solves of the same
            # programme differ by 1.3e-5 in one weight on Julia 1.13 whatever Clarabel
            # tolerance is set, so it takes the width it measures.
            solved = isa(opt, po.JuMPOptimisationEstimator) ||
                     isa(opt, NestedClustered) ||
                     isa(opt, SubsetResampling)
            atol = isa(opt, RiskBudgeting) ? 5e-5 : (solved ? 1e-5 : 1e-10)
            @test isapprox(ro.w, rb.w; atol = atol)
        end
        # A block warm-up followed by single steps reaches the same weights.
        o = partial_fit!(MeanRisk(; opt = jopt), rows(rd, 1:50))
        o = step(o, rows(rd, 51:t), t - 50)
        @test isapprox(optimise(o).w, optimise(MeanRisk(; opt = jopt), b).w; atol = 1e-5)
    end

    @testset "A factor prior folds both matrices, and a wrapping prior forwards" begin
        t = 80
        b = rows(rdf, 1:t)
        # A factor prior refits from a pair of buffers, so it is wrapped; the two wrapping
        # priors fold by forwarding and own no buffer, so they are not.
        bl = BlackLittermanPrior(;
                                 views = BlackLittermanViews(; P = [1.0 zeros(1, N - 1)],
                                                             Q = [0.01]))
        for (pe, online) in ((FactorPrior(), po.Online(FactorPrior())),
                             (HighOrderPriorEstimator(), HighOrderPriorEstimator()), (bl, bl))
            opt = InverseVolatility(; pe = pe)
            o = step(po.update_online_estimator(InverseVolatility(; pe = online)), rdf, t)
            @test isapprox(optimise(o).w, optimise(opt, b).w; rtol = 1e-8)
        end
        # `F` is owned once (#1013): a prior whose tree reads it records it in its own
        # buffer and the context keeps none, while a tree that never reads it leaves the
        # column to the context. The read-out's carrier is the same either way.
        ep_f = EntropyPoolingPrior(; pe = FactorPrior())
        o = step(po.update_online_estimator(InverseVolatility(; pe = po.Online(ep_f))), rdf,
                 t)
        @test isnothing(o.cache.F)
        @test po.factor_buffer(po.prior_returns_buffer(o.pe)) == F[1:t, :]
        @test same_fields(po.returns_result(o), b)
        e = step(po.update_online_estimator(InverseVolatility(;
                                                              pe = po.Online(EntropyPoolingPrior()))),
                 rdf, t)
        @test !isnothing(e.cache.F)
        @test isnothing(po.factor_buffer(po.prior_returns_buffer(e.pe)))
        @test same_fields(po.returns_result(e), b)
        # A factor leaf under an optional-argument host, through a JuMP optimiser on a
        # walk-forward with `rd.F` present, against batch.
        mr(pe) = MeanRisk(; opt = JuMPOptimiser(; pe = pe, slv = slv))
        o = step(po.update_online_estimator(mr(po.Online(ep_f))), rdf, t)
        @test isapprox(optimise(o).w, optimise(mr(ep_f), b).w; atol = 1e-5)
        # And the same tree with `rd.F` absent is refused by name at the step, not at the
        # leaf.
        @test_throws po.IsNothingError partial_fit!(po.update_online_estimator(mr(po.Online(ep_f))),
                                                    rows(rd, 1:1))
        # A wrapper two levels down is resolved through the optimiser's warm-up: the host
        # holds a wrapping prior, which holds another, which holds the wrapper.
        nested(pe) = HighOrderPriorEstimator(;
                                             pe = BlackLittermanPrior(; pe = pe,
                                                                      views = bl.views))
        o = step(po.update_online_estimator(InverseVolatility(;
                                                              pe = nested(po.Online(EntropyPoolingPrior())))),
                 rd, t)
        @test isapprox(optimise(o).w,
                       optimise(InverseVolatility(; pe = nested(EntropyPoolingPrior())),
                                rows(rd, 1:t)).w; rtol = 1e-8)
    end

    @testset "A wrapper's cap is the window, and the context follows it" begin
        w = 30
        o = step(po.update_online_estimator(InverseVolatility(;
                                                              pe = po.Online(EmpiricalPrior();
                                                                             max_history = w))),
                 rdf, 80)
        rdp = po.returns_result(o)
        @test same_fields(rdp, rows(rdf, 51:80))
        @test isapprox(optimise(o).w, optimise(InverseVolatility(), rows(rdf, 51:80)).w;
                       rtol = 1e-8)
        # And on a prior-less head the wrapper caps the rows it owns.
        e = step(po.update_online_estimator(po.Online(EqualWeighted(); max_history = w)),
                 rdv, 80)
        @test same_fields(po.returns_result(e), rows(rdv, 51:80))
    end

    @testset "A time-varying panel travels through its masks" begin
        Xg = copy(X)
        amsk = trues(T, N)
        amsk[1:30, 3] .= false          # lists at row 31
        amsk[91:end, 5] .= false        # delists at row 91
        Xg[.!amsk] .= NaN
        pnl = AssetPanel(; amsk = amsk, emsk = copy(amsk))
        rdg = ReturnsResult(; nx = nx, X = Xg, pnl = pnl)
        for t in (60, 100)
            b = rows(rdg, 1:t)
            o = step(EqualWeighted(), rdg, t)
            rdp = po.returns_result(o)
            @test isequal(rdp.X, b.X)
            @test rdp.pnl.amsk == b.pnl.amsk
            @test rdp.pnl.emsk == b.pnl.emsk
            @test optimise(o).w == optimise(EqualWeighted(), b).w
        end
        # A NaN gap folds through a prior to the batch Coverage Universe.
        o = step(MeanRisk(; opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv)), rdg,
                 100)
        ro = optimise(o)
        rb = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv)),
                      rows(rdg, 1:100))
        @test ro.imsk == rb.imsk
        @test isapprox(ro.w, rb.w; atol = 1e-5)
        # A masks-only panel reaches a meta-optimiser's outer fit: `collapse_asset_panel` used
        # to rebuild it from an empty comprehension, which the panel's constructor refuses.
        nco = NestedClustered(;
                              opti = MeanRisk(;
                                              opt = JuMPOptimiser(; pe = EmpiricalPrior(),
                                                                  slv = slv)),
                              opto = MeanRisk(;
                                              opt = JuMPOptimiser(; pe = EmpiricalPrior(),
                                                                  slv = slv)))
        @test isapprox(optimise(step(nco, rdg, 100)).w, optimise(nco, rows(rdg, 1:100)).w;
                       atol = 1e-5)
        # A static panel is pinned and returned as it was given.
        spnl = AssetPanel(; pf = [NumericPanelField(; name = "cap", vals = collect(1.0:N))])
        rds = ReturnsResult(; nx = nx, X = X, pnl = spnl)
        o = step(EqualWeighted(), rds, 20)
        @test po.returns_result(o).pnl.pf[1].vals == spnl.pf[1].vals
    end

    @testset "What the step refuses, by name" begin
        r1 = rows(rd, 1:1)
        opt = MeanRisk(; opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv))
        # A finite allocation has no step and no read-out.
        @test_throws ArgumentError partial_fit!(DiscreteAllocation(; slv = slv), r1)
        @test_throws ArgumentError optimise(DiscreteAllocation(; slv = slv))
        # A schedule on the prior, and a schedule of optimisers.
        @test_throws ArgumentError partial_fit!(MeanRisk(;
                                                         opt = JuMPOptimiser(;
                                                                             pe = TimeDependent([EmpiricalPrior()];
                                                                                                default = EmpiricalPrior()),
                                                                             slv = slv)),
                                                r1)
        @test_throws ArgumentError partial_fit!(TimeDependent([opt]; default = opt), r1)
        # A fitted prior is batch configuration.
        pr = prior(EmpiricalPrior(), rd)
        @test_throws ArgumentError partial_fit!(InverseVolatility(; pe = pr), r1)
        # And it still answers the fold-less entry it always had.
        @test optimise(InverseVolatility(; pe = pr)).w ==
              optimise(InverseVolatility(; pe = pr), rd).w
        # An implied-volatility surface does not travel the step.
        @test_throws ArgumentError partial_fit!(opt,
                                                ReturnsResult(; nx = nx, X = X[1:1, :],
                                                              iv = fill(0.2, 1, N),
                                                              ivpa = 1.0))
        # A time-varying Panel Field, and an estimation mask narrower than the active one.
        tv = AssetPanel(; pf = [NumericPanelField(; name = "cap", vals = ones(1, N))],
                        amsk = trues(1, N), emsk = trues(1, N))
        @test_throws ArgumentError partial_fit!(opt,
                                                ReturnsResult(; nx = nx, X = X[1:1, :],
                                                              pnl = tv))
        em = trues(1, N)
        em[1, 1] = false
        @test_throws ArgumentError partial_fit!(opt,
                                                ReturnsResult(; nx = nx, X = X[1:1, :],
                                                              pnl = AssetPanel(;
                                                                               amsk = trues(1,
                                                                                            N),
                                                                               emsk = em)))
        # The pinned context, and the column presence.
        o = partial_fit!(opt, r1)
        @test_throws ArgumentError partial_fit!(o,
                                                ReturnsResult(; nx = reverse(nx),
                                                              X = X[2:2, :]))
        @test_throws ArgumentError partial_fit!(o, rows(rdf, 2:2))
        @test_throws ArgumentError partial_fit!(partial_fit!(opt, rows(rdf, 1:1)),
                                                rows(rd, 2:2))
        # `Online` wraps the prior, never the host that holds it.
        @test_throws ArgumentError po.update_online_estimator(po.Online(JuMPOptimiser(;
                                                                                      slv = slv)))
        # An optimiser that has folded nothing has nothing to read out.
        @test_throws ArgumentError optimise(opt)
        @test_throws ArgumentError optimise(EqualWeighted())
        @test_throws ArgumentError po.returns_result(opt)
        # A pinned static panel refuses a later step that carries none, and the message
        # names the panel by its summary rather than by its content.
        spnl = AssetPanel(; pf = [NumericPanelField(; name = "cap", vals = collect(1.0:N))])
        os = partial_fit!(EqualWeighted(),
                          ReturnsResult(; nx = nx, X = X[1:1, :], pnl = spnl))
        err = try
            partial_fit!(os, rows(rd, 2:2))
            nothing
        catch e
            e
        end
        @test isa(err, ArgumentError) && occursin("AssetPanel", err.msg)
        @test !po.pinned_agree(spnl, spnl.pf[1])
        # A state with no rows is refused where the prior's buffer is read.
        @test_throws ArgumentError po.returns_buffer(po.SimpleExpectedReturnsState(; n = 1,
                                                                                   mu = zeros(N)))
    end

    @testset "The state merges, copies and slices" begin
        a = step(EqualWeighted(), rdf, 30).cache
        b = step(EqualWeighted(), rows(rdf, 31:60), 30).cache
        m = po.merge_states(a, b)
        @test same_fields(po.returns_result(m, m.X), rows(rdf, 1:60))
        c = copy(a)
        @test same_fields(po.returns_result(c, c.X), po.returns_result(a, a.X))
        @test c.X.X !== a.X.X && c.ts !== a.ts
        v = po.port_opt_view(a, 2:4)
        rv = po.returns_result(v, v.X)
        @test same_fields(rv, po.port_opt_view(rows(rdf, 1:30), 2:4))
        # A single-column benchmark is not indexed by asset, so a slice leaves it alone.
        s = po.port_opt_view(step(EqualWeighted(), rdv, 30).cache, 2:4)
        @test same_fields(po.returns_result(s, s.X), po.port_opt_view(rows(rdv, 1:30), 2:4))
        # A merge of two universes is refused.
        @test_throws ArgumentError po.merge_states(a,
                                                   step(EqualWeighted(),
                                                        ReturnsResult(; nx = reverse(nx),
                                                                      X = X), 5).cache)
        # And the context is rendered under the host only on request.
        @test !(:cache in po.show_fields(EqualWeighted()))
        @test :cache in fieldnames(EqualWeighted)
    end
end
