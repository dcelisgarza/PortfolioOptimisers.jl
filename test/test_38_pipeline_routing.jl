@testset "Pipeline routing targets" begin
    using Test, PortfolioOptimisers

    #=
    Solver-free by construction: every assertion here is a struct rebuild, so the seam
    between the Pipeline and the optimisers is exercised without building a JuMP model.
    The `Solver` below is a placeholder needed only to construct a `JuMPOptimiser`.
    =#
    PO = PortfolioOptimisers
    slv = Solver(; name = :placeholder, solver = nothing)
    jo() = JuMPOptimiser(; slv = slv)

    optimisers() = ("JuMPOptimiser" => jo(),
                    "HierarchicalOptimiser" => HierarchicalOptimiser(),
                    "MeanRisk" => MeanRisk(; opt = jo()),
                    "RiskBudgeting" => RiskBudgeting(; opt = jo()),
                    "NearOptimalCentering" => NearOptimalCentering(; opt = jo()),
                    "FactorRiskContribution" => FactorRiskContribution(; opt = jo()),
                    "RelaxedRiskBudgeting" => RelaxedRiskBudgeting(; opt = jo()),
                    "HierarchicalRiskParity" => HierarchicalRiskParity(),
                    "NestedClustered" =>
                        NestedClustered(; opti = EqualWeighted(), opto = EqualWeighted()),
                    "Stacking" =>
                        Stacking(; opti = [EqualWeighted()], opto = EqualWeighted()),
                    "SubsetResampling" => SubsetResampling(; opt = EqualWeighted()),
                    "EqualWeighted" => EqualWeighted(),
                    "InverseVolatility" => InverseVolatility(),
                    "RandomWeighted" => RandomWeighted())

    @testset "accepted targets are exactly the expected table" begin
        #=
        The routing rule is derived, not declared: a target lands in the like-named field
        of whichever optimiser has one. This table is therefore a lock on the *consequences*
        of the field layouts, and it fails in both directions — adding a field named after a
        target silently enrols that optimiser in routing, and removing one silently
        withdraws it. Either way the change should be deliberate, so it should land here.

        Reading the rows: the two configurations route into their own fields; the six
        estimators that hold a configuration forward to it (and the five carrying risk
        measures additionally accept :sigma_ucs into their own `r`); the naive and meta
        optimisers carry the target fields themselves.
        =#
        expected = Dict("JuMPOptimiser" => [:pe, :wb, :lcse, :ple, :mu_ucs],
                        "HierarchicalOptimiser" => [:pe, :cle, :wb],
                        "MeanRisk" => [:pe, :wb, :lcse, :ple, :mu_ucs, :sigma_ucs],
                        "RiskBudgeting" => [:pe, :wb, :lcse, :ple, :mu_ucs, :sigma_ucs],
                        "NearOptimalCentering" =>
                            [:pe, :wb, :lcse, :ple, :mu_ucs, :sigma_ucs],
                        "FactorRiskContribution" =>
                            [:pe, :wb, :lcse, :ple, :mu_ucs, :sigma_ucs],
                        # No `r` field, so no covariance uncertainty set.
                        "RelaxedRiskBudgeting" => [:pe, :wb, :lcse, :ple, :mu_ucs],
                        "HierarchicalRiskParity" => [:pe, :cle, :wb, :sigma_ucs],
                        # Meta-optimisers carry `pe`/`cle`/`wb` themselves.
                        "NestedClustered" => [:pe, :cle, :wb], "Stacking" => [:pe, :wb],
                        "SubsetResampling" => [:pe, :wb], "EqualWeighted" => [:wb],
                        "InverseVolatility" => [:pe, :wb], "RandomWeighted" => [:wb])
        for (name, opt) in optimisers()
            accepted = [t
                        for t in PO.PIPELINE_ROUTING_TARGETS
                        if PO.pipe_accepts(opt, Val(t))]
            @test accepted == expected[name]
        end
        @test length(expected) == length(optimisers())
    end

    @testset "delegation, declared rather than probed" begin
        #=
        `MeanRisk` has no `pe` field of its own, so the derived hasfield rule alone would
        reject `:pe`; it accepts only because it declares where its configuration lives.
        =#
        mr = MeanRisk(; opt = jo())
        @test !hasfield(typeof(mr), :pe)
        @test PO.pipe_accepts(mr, Val(:pe))
        @test PO.pipe_config_field(mr) === :opt
        @test PO.pipe_config_field(EqualWeighted()) === nothing
        #=
        `SubsetResampling` has a field literally named `opt`, holding an inner *estimator*
        rather than a configuration. Declaring rather than probing for `:opt` is what keeps
        that name collision from being mistaken for a configuration to route through.
        =#
        sr = SubsetResampling(; opt = EqualWeighted())
        @test hasfield(typeof(sr), :opt)
        @test PO.pipe_config_field(sr) === nothing
    end

    @testset "pipe_route places the value and changes nothing else" begin
        wb = WeightBounds(; lb = fill(0.1, 3), ub = fill(0.9, 3))
        for (name, opt) in optimisers()
            PO.pipe_accepts(opt, Val(:wb)) || continue
            routed = PO.pipe_route(opt, Val(:wb), wb)
            # Type parameters change with the new field value; the wrapper must not.
            @test Base.typename(typeof(routed)).wrapper ===
                  Base.typename(typeof(opt)).wrapper
            target = if isnothing(PO.pipe_config_field(opt))
                routed
            else
                getfield(routed, PO.pipe_config_field(opt))
            end
            @test target.wb === wb
        end
        # Delegation rebuilds through the configuration rather than replacing it.
        mr = MeanRisk(; opt = jo())
        routed = PO.pipe_route(mr, Val(:wb), wb)
        @test routed.opt.wb === wb
        @test routed.r === mr.r
        @test routed.obj === mr.obj
    end

    @testset "unroutable targets: optional pass by, the rest fail closed" begin
        @test PO.PIPELINE_OPTIONAL_TARGETS == (:pe, :cle)
        @test all(t -> t in PO.PIPELINE_ROUTING_TARGETS, PO.PIPELINE_OPTIONAL_TARGETS)

        ew = EqualWeighted()
        # `EqualWeighted` has no prior to override and computes nothing from one.
        @test PO.pipe_route(ew, Val(:pe), nothing) === ew
        # A `JuMPOptimiser` takes phylogeny as constraint results, never as a structure.
        j = jo()
        @test PO.pipe_route(j, Val(:cle), nothing) === j
        # Everything else would silently change the solved portfolio, so it throws.
        for t in (:lcse, :ple, :mu_ucs, :sigma_ucs)
            @test_throws ArgumentError PO.pipe_route(ew, Val(t), nothing)
        end
        @test_throws ArgumentError PO.pipe_route(HierarchicalOptimiser(), Val(:lcse),
                                                 nothing)
    end

    @testset "the two policy-carrying targets" begin
        #=
        `:mu_ucs` and `:sigma_ucs` name no field: they carry the validation the seam would
        otherwise lose. Both fail closed rather than dropping a computed set.
        =#
        mu_ucs = MuEllipsoidalUncertaintySet()
        sigma_ucs = SigmaEllipsoidalUncertaintySet()
        # Only an ArithmeticReturn can carry a bound on expected returns.
        @test PO.pipe_route(jo(), Val(:mu_ucs), mu_ucs).ret.ucs === mu_ucs
        @test_throws ArgumentError PO.pipe_route(JuMPOptimiser(; slv = slv,
                                                               ret = LogarithmicReturn()),
                                                 Val(:mu_ucs), mu_ucs)
        # The covariance half lands in the estimator's own risk measures, not its config.
        routed = PO.pipe_route(MeanRisk(; opt = jo(), r = UncertaintySetVariance()),
                               Val(:sigma_ucs), sigma_ucs)
        @test routed.r.ucs === sigma_ucs
        # An `r` field with no UncertaintySetVariance has nowhere to put the set.
        @test_throws ArgumentError PO.pipe_route(MeanRisk(; opt = jo(), r = Variance()),
                                                 Val(:sigma_ucs), sigma_ucs)
    end

    @testset "unroutable uncertainty is rejected at construction" begin
        #=
        Without this the failure surfaces at injection, which under cross_val_predict is
        after the first fold has already fitted every earlier step.
        =#
        ucs_step(t) = PipelineStep(; est = NormalUncertaintySet(), reads = (:returns,),
                                   writes = :uncertainty, target = t)

        # HierarchicalRiskParity can take a covariance set into its `r`, but has nowhere
        # to put a mean set.
        @test_throws ArgumentError Pipeline(;
                                            steps = (EmpiricalPrior(), ucs_step(:mu),
                                                     HierarchicalRiskParity()))
        @test_throws ArgumentError Pipeline(;
                                            steps = (EmpiricalPrior(), ucs_step(:both),
                                                     HierarchicalRiskParity()))
        hrp = Pipeline(;
                       steps = (EmpiricalPrior(), ucs_step(:sigma),
                                HierarchicalRiskParity()))
        @test length(hrp.steps) == 3

        # A JuMP-based optimiser accepts both halves.
        mr = Pipeline(;
                      steps = (EmpiricalPrior(), ucs_step(:both),
                               MeanRisk(; opt = jo(), r = UncertaintySetVariance())))
        @test length(mr.steps) == 3

        #=
        The check is structural: it establishes the optimiser family can receive the
        target, not that this configuration accepts the value. A JuMPOptimiser carrying a
        non-ArithmeticReturn still constructs, and still fails at injection.
        =#
        @test PO.pipe_accepts(MeanRisk(;
                                       opt = JuMPOptimiser(; slv = slv,
                                                           ret = LogarithmicReturn())),
                              Val(:mu_ucs))

        # Targets are only known statically for uncertainty steps.
        @test PO.pipe_required_targets(ucs_step(:both)) == (:mu_ucs, :sigma_ucs)
        @test PO.pipe_required_targets(ucs_step(:sigma)) == (:sigma_ucs,)
        @test PO.pipe_required_targets(EmpiricalPrior()) == ()
    end

    @testset "constraint slot fans out by result type" begin
        wb = WeightBounds(; lb = fill(0.1, 3), ub = fill(0.9, 3))
        @test PO.constraint_targets(nothing) == []
        @test PO.constraint_targets(wb) == [:wb => wb]
        #=
        A constraint result with no target at all is rejected at the fan-out, before any
        optimiser sees it — there is no field it could ever land in.
        =#
        @test_throws ArgumentError PO.constraint_targets(PO.risk_budget_constraints(nothing;
                                                                                    N = 5))
    end
end
