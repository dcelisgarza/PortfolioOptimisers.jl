@testset "Pipeline fit" begin
    using Test, PortfolioOptimisers, TimeSeries, Dates, StableRNGs, Statistics, Clarabel

    function make_prices(; T = 60, N = 5)
        rng = StableRNG(123456789)
        ts = range(Date(2020, 1, 1); step = Day(1), length = T)
        return TimeArray(ts, 100 .+ cumsum(randn(rng, T, N) / 10; dims = 1),
                         string.("A", 1:N))
    end
    function make_returns(; T = 60, N = 5)
        rng = StableRNG(987654321)
        return ReturnsResult(; nx = string.("A", 1:N), X = randn(rng, T, N) / 100)
    end
    jopt() = JuMPOptimiser(;
                           slv = Solver(; name = :Clarabel, solver = Clarabel.Optimizer,
                                        settings = Dict("verbose" => false)))

    @testset "construction and naming" begin
        pipe = Pipeline(; steps = (PricesToReturns(), EmpiricalPrior(), EqualWeighted()))
        @test pipe.names == ("returns", "prior", "opt")

        # repeated slots are suffixed in order; explicit names pass through
        pipe = Pipeline(;
                        steps = (MissingDataFilter(), "impute" => Imputer(),
                                 PricesToReturns()))
        @test pipe.names == ("prices_1", "impute", "returns")
        pipe = Pipeline(; steps = (MissingDataFilter(), Imputer()))
        @test pipe.names == ("prices_1", "prices_2")

        # steps cannot be empty
        @test_throws PortfolioOptimisers.IsEmptyError Pipeline(; steps = ())

        # duplicate names are rejected
        @test_throws ArgumentError Pipeline(;
                                            steps = ("a" => Imputer(),
                                                     "a" => MissingDataFilter()))

        # non-steppable estimators are rejected at construction
        @test_throws ArgumentError Pipeline(; steps = (Covariance(),))

        # declared reads must be writable by an earlier step or the input
        ps = PipelineStep(; est = c -> c.prior, reads = (:prior,), writes = :prior)
        @test_throws ArgumentError Pipeline(; steps = (ps,))
        pipe = Pipeline(; steps = (EmpiricalPrior(), ps))
        @test pipe.names == ("prior_1", "prior_2")
    end

    @testset "fit at returns level" begin
        rd = make_returns()
        pipe = Pipeline(; steps = (EmpiricalPrior(), EqualWeighted()))
        res = PortfolioOptimisers.fit(pipe, rd)
        @test res isa PipelineResult
        @test res["prior"] === res.results[1]
        @test res.ctx.prior.X == rd.X
        @test res.ctx.returns === rd
        @test res.w ≈ fill(0.2, 5)
        @test res["opt"].w ≈ fill(0.2, 5)
        @test_throws ArgumentError res["nope"]

        # a pipeline without an optimisation step has no weights
        res = PortfolioOptimisers.fit(Pipeline(; steps = (EmpiricalPrior(),)), rd)
        @test isnothing(res.ctx.opt)
        @test_throws PortfolioOptimisers.PropertyPathError res.w
    end

    @testset "fit at prices level with injection" begin
        X = make_prices()
        pr = PricesResult(; X = X)
        pipe = Pipeline(;
                        steps = (MissingDataFilter(; col_thr = 0.5), Imputer(),
                                 PricesToReturns(), EmpiricalPrior(),
                                 HierarchicalRiskParity()))
        res = PortfolioOptimisers.fit(pipe, pr)
        @test size(res.ctx.returns.X) == (59, 5)
        @test length(res.w) == 5
        @test sum(res.w) ≈ 1

        # the shared prior is injected: identical to manually configuring the optimiser
        rd = prices_to_returns(X)
        pr_manual = prior(EmpiricalPrior(), rd)
        w_manual = optimise(HierarchicalRiskParity(;
                                                   opt = HierarchicalOptimiser(;
                                                                               pe = pr_manual)),
                            rd).w
        @test res.w ≈ w_manual
    end

    @testset "inject_context: prior and phylogeny" begin
        rd = make_returns()
        pr = prior(EmpiricalPrior(), rd)
        cl = clusterise(ClustersEstimator(), rd)
        ctx = PortfolioOptimisers.PipelineContext(; returns = rd, prior = pr,
                                                  phylogeny = cl)

        # hierarchical: pe and cle both overridden
        hrp2 = PortfolioOptimisers.inject_context(HierarchicalRiskParity(), ctx)
        @test hrp2.opt.pe === pr
        @test hrp2.opt.cle === cl

        # JuMP: pe overridden, phylogeny ignored (enters via constraint results)
        mr2 = PortfolioOptimisers.inject_context(MeanRisk(; opt = jopt()), ctx)
        @test mr2.opt.pe === pr

        # PipelineStep-wrapped optimisers are injected too
        ps = PipelineStep(; est = HierarchicalRiskParity(), reads = (:returns,),
                          writes = :opt)
        ps2 = PortfolioOptimisers.maybe_inject_step(ps, ctx)
        @test ps2.est.opt.pe === pr

        # naive optimisers have nothing to inject and ignore prior/phylogeny
        ew2 = PortfolioOptimisers.inject_context(EqualWeighted(), ctx)
        @test ew2 === EqualWeighted()
    end

    @testset "uncertainty steps and routing" begin
        rd = make_returns()
        ps_mu = PipelineStep(; est = DeltaUncertaintySet(), reads = (:returns,),
                             writes = :uncertainty, target = :mu)
        ps_sigma = PipelineStep(; est = DeltaUncertaintySet(), reads = (:returns,),
                                writes = :uncertainty, target = :sigma)
        pipe = Pipeline(; steps = (ps_mu, ps_sigma))
        @test pipe.names == ("uncertainty_1", "uncertainty_2")
        res = PortfolioOptimisers.fit(pipe, rd)
        unc = res.ctx.uncertainty
        @test unc isa PortfolioOptimisers.PipelineUncertaintySets
        mu_ref = mu_ucs(DeltaUncertaintySet(), rd.X)
        sigma_ref = sigma_ucs(DeltaUncertaintySet(), rd.X)
        @test unc.mu.lb == mu_ref.lb
        @test unc.mu.ub == mu_ref.ub
        @test unc.sigma.lb == sigma_ref.lb
        @test unc.sigma.ub == sigma_ref.ub

        # routing: mu -> ret.ucs, sigma -> UncertaintySetVariance.ucs
        mr = MeanRisk(; opt = jopt(), r = UncertaintySetVariance())
        mr2 = PortfolioOptimisers.inject_context(mr, res.ctx)
        @test mr2.opt.ret.ucs === unc.mu
        @test mr2.r.ucs === unc.sigma

        # vector risk measures are searched for UncertaintySetVariance
        mr = MeanRisk(; opt = jopt(), r = [Variance(), UncertaintySetVariance()])
        mr2 = PortfolioOptimisers.inject_context(mr, res.ctx)
        @test mr2.r[2].ucs === unc.sigma
        @test mr2.r[1] === mr.r[1]

        # bare uncertainty steps must declare a target
        ctx = PortfolioOptimisers.PipelineContext(; returns = rd)
        @test_throws ArgumentError PortfolioOptimisers.run_step(DeltaUncertaintySet(), ctx)
        ps_bad = PipelineStep(; est = DeltaUncertaintySet(), writes = :uncertainty)
        @test_throws ArgumentError PortfolioOptimisers.run_step(ps_bad, ctx)

        # unroutable targets fail loudly
        @test_throws ArgumentError PortfolioOptimisers.fit(Pipeline(;
                                                                    steps = (ps_sigma,
                                                                             EqualWeighted())),
                                                           rd)
        @test_throws ArgumentError PortfolioOptimisers.fit(Pipeline(;
                                                                    steps = (ps_mu,
                                                                             HierarchicalRiskParity())),
                                                           rd)
        # sigma with no UncertaintySetVariance in r
        @test_throws ArgumentError PortfolioOptimisers.fit(Pipeline(;
                                                                    steps = (ps_sigma,
                                                                             HierarchicalRiskParity())),
                                                           rd)
    end

    @testset "constraint steps and routing" begin
        rd = make_returns()
        pipe = Pipeline(;
                        steps = (WeightBoundsEstimator(; ub = 0.3),
                                 LinearConstraintEstimator(; val = "A1 <= 0.25"),
                                 EmpiricalPrior()))
        @test pipe.names == ("constraints_1", "constraints_2", "prior")
        res = PortfolioOptimisers.fit(pipe, rd)
        cons = res.ctx.constraints
        @test cons isa Vector{PortfolioOptimisers.AbstractConstraintResult}
        @test cons[1] isa WeightBounds
        @test cons[2] isa LinearConstraint

        # routing into a JuMP optimiser
        mr2 = PortfolioOptimisers.inject_context(MeanRisk(; opt = jopt()), res.ctx)
        @test mr2.opt.wb === cons[1]
        @test mr2.opt.lcse === cons[2]

        # hierarchical optimisers accept weight bounds but not linear constraints
        ctx_wb = PortfolioOptimisers.PipelineContext(; returns = rd, constraints = cons[1])
        hrp2 = PortfolioOptimisers.inject_context(HierarchicalRiskParity(), ctx_wb)
        @test hrp2.opt.wb === cons[1]
        @test_throws ArgumentError PortfolioOptimisers.inject_context(HierarchicalRiskParity(),
                                                                      res.ctx)

        # unroutable constraint results fail loudly
        rb = PortfolioOptimisers.risk_budget_constraints(nothing; N = 5)
        ctx_rb = PortfolioOptimisers.PipelineContext(; returns = rd, constraints = rb)
        @test_throws ArgumentError PortfolioOptimisers.inject_context(MeanRisk(;
                                                                               opt = jopt()),
                                                                      ctx_rb)

        # optimisers without an injectable config reject computed constraints
        @test_throws ArgumentError PortfolioOptimisers.inject_context(EqualWeighted(),
                                                                      ctx_wb)
    end

    @testset "nested pipelines and guards" begin
        X = make_prices()
        pr = PricesResult(; X = X)
        sub = Pipeline(; steps = (MissingDataFilter(), PricesToReturns()))
        @test PortfolioOptimisers.pipe_reads(sub) == (:prices,)
        @test PortfolioOptimisers.pipe_writes(sub) == :returns
        pipe = Pipeline(; steps = (sub, EmpiricalPrior(), EqualWeighted()))
        @test pipe.names == ("returns", "prior", "opt")
        res = PortfolioOptimisers.fit(pipe, pr)
        @test res.results[1] isa PipelineResult
        @test size(res.ctx.returns.X) == (59, 5)
        @test res.w ≈ fill(0.2, 5)

        # a pipeline is not an optimiser and cannot be wrapped in one
        @test_throws ArgumentError optimise(pipe, prices_to_returns(X))
    end
end
