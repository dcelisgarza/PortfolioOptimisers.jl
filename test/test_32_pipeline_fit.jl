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
    # assets built from a known 5×2 loading matrix, so a factor mandate has something
    # to bite on and the regression recovers a well-conditioned `M`
    function make_factor_returns(; T = 60, N = 5)
        rng = StableRNG(987654321)
        F = randn(rng, T, 2) / 100
        M = [1.0 0.0; 0.5 0.5; 0.0 1.0; 0.8 0.2; 0.2 0.8]
        return ReturnsResult(; nx = string.("A", 1:N),
                             X = F * transpose(M) + randn(rng, T, N) / 1000,
                             nf = ["MTUM", "VLUE"], F = F)
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
        res = fit(pipe, rd)
        @test res isa PipelineResult
        @test res["prior"] === res.results[1]
        @test res.ctx.prior.X == rd.X
        @test res.ctx.returns === rd
        @test res.w ≈ fill(0.2, 5)
        @test res["opt"].w ≈ fill(0.2, 5)
        @test_throws ArgumentError res["nope"]

        # a pipeline without an optimisation step has no weights
        res = fit(Pipeline(; steps = (EmpiricalPrior(),)), rd)
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
        res = fit(pipe, pr)
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
        res = fit(pipe, rd)
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

        # target = :both derives both halves from a single `ucs` call
        ps_both = PipelineStep(; est = DeltaUncertaintySet(), reads = (:returns,),
                               writes = :uncertainty, target = :both)
        res_both = fit(Pipeline(; steps = (ps_both,)), rd)
        unc_both = res_both.ctx.uncertainty
        @test unc_both isa PortfolioOptimisers.PipelineUncertaintySets
        @test unc_both.mu.lb == mu_ref.lb
        @test unc_both.mu.ub == mu_ref.ub
        @test unc_both.sigma.lb == sigma_ref.lb
        @test unc_both.sigma.ub == sigma_ref.ub

        # ... and agrees with the two narrowed steps run separately
        @test unc_both.mu.ub == unc.mu.ub
        @test unc_both.sigma.ub == unc.sigma.ub

        # the fitted object of a :both step is the pair itself
        @test res_both.results[1] === unc_both

        # :both requires an optimiser that can take *both* halves
        mr_both = PortfolioOptimisers.inject_context(MeanRisk(; opt = jopt(),
                                                              r = UncertaintySetVariance()),
                                                     res_both.ctx)
        @test mr_both.opt.ret.ucs === unc_both.mu
        @test mr_both.r.ucs === unc_both.sigma
        @test_throws ArgumentError PortfolioOptimisers.inject_context(MeanRisk(;
                                                                               opt = jopt(),
                                                                               r = Variance()),
                                                                      res_both.ctx)

        # bare uncertainty steps must declare a target
        ctx = PortfolioOptimisers.PipelineContext(; returns = rd)
        @test_throws ArgumentError PortfolioOptimisers.run_step(DeltaUncertaintySet(), ctx)
        ps_bad = PipelineStep(; est = DeltaUncertaintySet(), writes = :uncertainty)
        @test_throws ArgumentError PortfolioOptimisers.run_step(ps_bad, ctx)

        # an unknown target is rejected
        ps_wrong = PipelineStep(; est = DeltaUncertaintySet(), reads = (:returns,),
                                writes = :uncertainty, target = :nonsense)
        @test_throws ArgumentError PortfolioOptimisers.run_step(ps_wrong, ctx)

        # unroutable targets fail loudly
        @test_throws ArgumentError fit(Pipeline(; steps = (ps_sigma, EqualWeighted())), rd)
        @test_throws ArgumentError fit(Pipeline(;
                                                steps = (ps_mu, HierarchicalRiskParity())),
                                       rd)
        # sigma with no UncertaintySetVariance in r
        @test_throws ArgumentError fit(Pipeline(;
                                                steps = (ps_sigma,
                                                         HierarchicalRiskParity())), rd)
    end

    @testset "constraint steps and routing" begin
        rd = make_returns()
        pipe = Pipeline(;
                        steps = (WeightBoundsEstimator(; ub = 0.3),
                                 LinearConstraintEstimator(; val = "A1 <= 0.25"),
                                 EmpiricalPrior()))
        @test pipe.names == ("constraints_1", "constraints_2", "prior")
        res = fit(pipe, rd)
        cons = res.ctx.constraints
        @test cons isa Vector{PortfolioOptimisers.AbstractConstraintResult}
        @test cons[1] isa WeightBounds
        @test cons[2] isa LinearConstraint

        # routing into a JuMP optimiser
        mr2 = PortfolioOptimisers.inject_context(MeanRisk(; opt = jopt()), res.ctx)
        @test mr2.opt.wb === cons[1]
        @test mr2.opt.lcse === cons[2]

        # phylogeny-constraint estimators are steppable: they compute their own
        # phylogeny from :returns and route to the JuMP optimiser's `ple` field
        pipe_ph = Pipeline(; steps = (SemiDefinitePhylogenyEstimator(), EmpiricalPrior()))
        @test pipe_ph.names == ("constraints", "prior")
        res_ph = fit(pipe_ph, rd)
        @test res_ph.ctx.constraints isa SemiDefinitePhylogeny
        mr_ph = PortfolioOptimisers.inject_context(MeanRisk(; opt = jopt()), res_ph.ctx)
        @test mr_ph.opt.ple === res_ph.ctx.constraints

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

        # naive optimisers carry an asset-dimensioned `wb` of their own, so a computed
        # weight bound reaches them directly rather than being rejected
        ew2 = PortfolioOptimisers.inject_context(EqualWeighted(), ctx_wb)
        @test ew2.wb === cons[1]
        # they still reject a target they have no field for
        ctx_lc = PortfolioOptimisers.PipelineContext(; returns = rd, constraints = cons[2])
        @test_throws ArgumentError PortfolioOptimisers.inject_context(EqualWeighted(),
                                                                      ctx_lc)
    end

    @testset "exposure constraint steps" begin
        #=
        A factor mandate as a bare pipeline step. The returns carry a factor block, so
        `pipeline_asset_sets` declares the `nf` axis and the step's names resolve
        against it; the basis is the pipeline prior's loadings, and what lands in the
        `constraints` slot is an ordinary asset-space LinearConstraint.
        =#
        rd = make_factor_returns()
        ece = ExposureConstraintEstimator(;
                                          lce = LinearConstraintEstimator(;
                                                                          val = "MTUM <= 0.3"),
                                          space = FactorSpace())
        pipe = Pipeline(; steps = (FactorPrior(), ece, MeanRisk(; opt = jopt())))
        @test pipe.names == ("prior", "constraints", "opt")
        res = fit(pipe, rd)
        lcr = res.ctx.constraints
        @test lcr isa LinearConstraint
        # projected into asset space: one row, one column per asset, not per factor
        @test size(lcr.ineq.A) == (1, length(rd.nx))

        #=
        The criterion is that the step buys nothing but convenience: generating the
        same constraint by hand against the same prior and passing it through `lcse`
        must give the same weights. It does, bit for bit, because the step calls the
        same generator with the same basis.
        =#
        pr = prior(FactorPrior(), rd)
        sets = UniverseSets(; dict = Dict("nx" => rd.nx, "nf" => rd.nf))
        lcr_direct = linear_constraints(ece, sets; rr = pr.rr)
        w_direct = optimise(MeanRisk(;
                                     opt = JuMPOptimiser(; pe = pr, lcse = lcr_direct,
                                                         slv = jopt().slv)), rd).w
        @test res.w == w_direct
        # and the mandate actually binds
        @test (transpose(pr.rr.M)*res.w)[1] ≈ 0.3 atol = 1e-6

        #=
        The `:prior` read is a construction-time declaration, so a pipeline with no
        prior step is rejected before any data is touched — the reads check fires,
        not `require_slot`.
        =#
        @test_throws ArgumentError Pipeline(; steps = (ece, MeanRisk(; opt = jopt())))

        #=
        A prior that carries no regression is a different failure: the slot is
        populated, so the pipeline constructs, and generation throws the
        missing-loadings error. It is deliberately not governed by `strict`.
        =#
        pipe_nolo = Pipeline(; steps = (EmpiricalPrior(), ece, MeanRisk(; opt = jopt())))
        @test_throws PortfolioOptimisers.IsNothingError fit(pipe_nolo, rd)
        # same error when the returns carry no factor block at all
        rd_nof = ReturnsResult(; nx = rd.nx, X = rd.X)
        @test_throws PortfolioOptimisers.IsNothingError fit(pipe_nolo, rd_nof)

        #=
        A wrapped vector produces sibling constraints rather than a nested one, so
        `constraint_targets` sees them alongside every other step's result and packs
        them into `lcse` as a vector.
        =#
        ece_vec = ExposureConstraintEstimator(;
                                              lce = [LinearConstraintEstimator(;
                                                                               val = "MTUM <= 0.3"),
                                                     LinearConstraintEstimator(;
                                                                               val = "VLUE >= 0.1")],
                                              space = FactorSpace())
        res_vec = fit(Pipeline(;
                               steps = (FactorPrior(), ece_vec, MeanRisk(; opt = jopt()))),
                      rd)
        cons_vec = res_vec.ctx.constraints
        @test cons_vec isa Vector{PortfolioOptimisers.AbstractConstraintResult}
        @test length(cons_vec) == 2
        @test all(c -> isa(c, LinearConstraint), cons_vec)
        f_w = transpose(pr.rr.M) * res_vec.w
        @test f_w[1] <= 0.3 + 1e-6
        @test f_w[2] >= 0.1 - 1e-6

        #=
        A row whose names all resolve away leaves the slot untouched instead of
        crashing: generation already decided the condition was recoverable and
        returned nothing, and the slot's job is to carry constraints, not to
        re-diagnose that.
        =#
        ece_none = ExposureConstraintEstimator(;
                                               lce = LinearConstraintEstimator(;
                                                                               val = "NOPE <= 0.3"),
                                               space = FactorSpace())
        res_none = (@test_logs (:warn,) (:warn,) fit(Pipeline(;
                                                              steps = (FactorPrior(),
                                                                       ece_none,
                                                                       EqualWeighted())),
                                                     rd))
        @test isnothing(res_none.ctx.constraints)
        @test res_none.w ≈ fill(1 / length(rd.nx), length(rd.nx))

        #=
        Two ordinary linear-constraint steps route into `lcse` as a vector of
        precomputed constraints, which the optimiser must pass through untouched.
        This is not exposure-specific — it was reachable before there was a
        re-basis — but the wrapped-vector step above lands on the same seam.
        =#
        res_two = fit(Pipeline(;
                               steps = (LinearConstraintEstimator(; val = "A1 <= 0.3"),
                                        LinearConstraintEstimator(; val = "A2 >= 0.1"),
                                        FactorPrior(), MeanRisk(; opt = jopt()))), rd)
        @test res_two.w[1] <= 0.3 + 1e-6
        @test res_two.w[2] >= 0.1 - 1e-6
    end

    @testset "nested pipelines and guards" begin
        X = make_prices()
        pr = PricesResult(; X = X)
        sub = Pipeline(; steps = (MissingDataFilter(), PricesToReturns()))
        @test PortfolioOptimisers.pipe_reads(sub) == (:prices,)
        @test PortfolioOptimisers.pipe_writes(sub) == :returns
        pipe = Pipeline(; steps = (sub, EmpiricalPrior(), EqualWeighted()))
        @test pipe.names == ("returns", "prior", "opt")
        res = fit(pipe, pr)
        @test res.results[1] isa PipelineResult
        @test size(res.ctx.returns.X) == (59, 5)
        @test res.w ≈ fill(0.2, 5)

        # a pipeline is not an optimiser and cannot be wrapped in one
        @test_throws ArgumentError optimise(pipe, prices_to_returns(X))
    end
end
