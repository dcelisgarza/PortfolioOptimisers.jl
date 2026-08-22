@testset "Pipeline routing targets" begin
    using Test, PortfolioOptimisers, InteractiveUtils

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
        optimisers carry the target fields themselves. `:rkb` is the one target that is
        neither derived nor universal — it lands at `rba.rkb`, so only the two optimisers
        carrying a risk-budgeting algorithm accept it.
        =#
        jump = [:pe, :wb, :lcse, :cte, :ple, :lt, :st, :slt, :sst, :sglt, :sgst, :smtx,
                :sgmtx]
        expected = Dict("JuMPOptimiser" => [jump; :mu_ucs],
                        "HierarchicalOptimiser" => [:pe, :cle, :wb],
                        "MeanRisk" => [jump; :mu_ucs; :sigma_ucs],
                        "RiskBudgeting" => [jump; :rkb; :mu_ucs; :sigma_ucs],
                        "NearOptimalCentering" => [jump; :mu_ucs; :sigma_ucs],
                        "FactorRiskContribution" => [jump; :mu_ucs; :sigma_ucs],
                        # No `r` field, so no covariance uncertainty set.
                        "RelaxedRiskBudgeting" => [jump; :rkb; :mu_ucs],
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
        for t in
            (:lcse, :cte, :ple, :lt, :st, :slt, :sst, :sglt, :sgst, :smtx, :sgmtx, :rkb,
             :mu_ucs, :sigma_ucs)
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

        # Targets are known statically for uncertainty steps and for constraint steps.
        @test PO.pipe_required_targets(ucs_step(:both)) == (:mu_ucs, :sigma_ucs)
        @test PO.pipe_required_targets(ucs_step(:sigma)) == (:sigma_ucs,)
        @test PO.pipe_required_targets(EmpiricalPrior()) == ()
        @test PO.pipe_required_targets(WeightBoundsEstimator()) == (:wb,)
        @test PO.pipe_required_targets(CentralityConstraint()) == (:cte,)
    end

    @testset "constraint slot fans out by carried target, then by result type" begin
        wb = WeightBounds(; lb = fill(0.1, 3), ub = fill(0.9, 3))
        lc = LinearConstraint(;
                              ineq = PartialLinearConstraint(; A = [1.0 0.0 0.0],
                                                             B = [0.5]))
        thr = Threshold(0.05)
        rkb = PO.risk_budget_constraints(nothing; N = 5)
        @test PO.constraint_targets(nothing) == []
        @test PO.constraint_targets(wb) == [:wb => wb]
        # A RiskBudget names exactly one field, so its type alone places it.
        @test PO.constraint_targets(rkb) == [:rkb => rkb]
        #=
        A Threshold names six fields, so its type cannot place it. A step declares the
        target and the declaration travels with the value.
        =#
        @test_throws ArgumentError PO.constraint_targets(thr)
        @test PO.constraint_targets(PO.TargetedConstraint(:sglt, thr)) == [:sglt => thr]
        #=
        An accumulating target packs; a single-valued one refuses the second write rather
        than letting it overwrite the first.
        =#
        pair = PO.constraint_targets(PO.AbstractConstraintResult[lc, lc])
        @test only(pair).first === :lcse
        @test only(pair).second == [lc, lc]
        @test_throws ArgumentError PO.constraint_targets(PO.AbstractConstraintResult[wb,
                                                                                     wb])
        @test_throws ArgumentError PO.constraint_targets(PO.AbstractConstraintResult[PO.TargetedConstraint(:lt,
                                                                                                           thr),
                                                                                     PO.TargetedConstraint(:lt,
                                                                                                           thr)])
        # A result of a type no target exists for is still rejected at the fan-out.
        @test_throws ArgumentError PO.constraint_targets(PartialLinearConstraint(;
                                                                                 A = [1.0 0.0 0.0],
                                                                                 B = [0.5]))
    end

    @testset "a target accumulates exactly when its field takes a vector of results" begin
        #=
        `PIPELINE_ACCUMULATING_TARGETS` is a claim about the *receiving fields*, so it is
        checked against them rather than restated. Each row puts a two-element vector of
        computed results in one target and asserts the optimiser takes it; the scenario and
        group rows carry the membership matrices their lengths are validated against, which
        is the check that makes packing them safe.
        =#
        wb = WeightBounds(; lb = fill(0.1, 3), ub = fill(0.9, 3))
        lc = LinearConstraint(;
                              ineq = PartialLinearConstraint(; A = [1.0 0.0 0.0],
                                                             B = [0.5]))
        thr = Threshold(0.05)
        ple = SemiDefinitePhylogeny(; A = [0 1 0; 1 0 0; 0 0 0], p = 0.05)
        mtx = [1 0 0; 0 1 1]
        packed = (:lcse => (; lcse = [lc, lc]), :ple => (; ple = [ple, ple]),
                  :slt => (; slt = [thr, thr], smtx = [mtx, mtx]),
                  :sst => (; sst = [thr, thr], smtx = [mtx, mtx]),
                  :sglt => (; sglt = [thr, thr], sgmtx = [mtx, mtx]),
                  :sgst => (; sgst = [thr, thr], sgmtx = [mtx, mtx]),
                  :smtx => (; smtx = [mtx, mtx]), :sgmtx => (; sgmtx = [mtx, mtx]))
        for (target, kwargs) in packed
            @test JuMPOptimiser(; slv = slv, kwargs...) isa JuMPOptimiser
            @test target in PO.PIPELINE_ACCUMULATING_TARGETS
        end
        #=
        `:cte` accumulates by *folding*, not packing, so it is the one member whose field
        does not take a vector of results. Its vector form is a vector of estimators, and
        generation appends their rows into one constraint — which is the value the fold
        produces, so that is what the field must accept.
        =#
        @test :cte in PO.PIPELINE_ACCUMULATING_TARGETS
        @test_throws TypeError JuMPOptimiser(; slv = slv, cte = [lc, lc])
        @test JuMPOptimiser(; slv = slv, cte = PO.merge_linear_constraints([lc, lc])) isa
              JuMPOptimiser
        @test Set([first.(packed)...; :cte]) == Set(PO.PIPELINE_ACCUMULATING_TARGETS)

        # The complement: fields that hold exactly one value, and refuse a vector.
        single = (:wb => (; wb = [wb, wb]), :lt => (; lt = [thr, thr]),
                  :st => (; st = [thr, thr]))
        for (target, kwargs) in single
            @test_throws TypeError JuMPOptimiser(; slv = slv, kwargs...)
            @test !(target in PO.PIPELINE_ACCUMULATING_TARGETS)
        end
        # `:rkb` lives on the risk-budgeting algorithm, and is single-valued there.
        rkb = PO.risk_budget_constraints(nothing; N = 3)
        @test_throws TypeError AssetRiskBudgeting(; rkb = [rkb, rkb])
        @test !(:rkb in PO.PIPELINE_ACCUMULATING_TARGETS)

        #=
        The fan-out follows the table in both directions: two writes combine for an
        accumulating target and throw for every other one.
        =#
        for target in PO.PIPELINE_ACCUMULATING_TARGETS
            if target === :cte
                continue
            end
            two = PO.AbstractConstraintResult[PO.TargetedConstraint(target, thr),
                                              PO.TargetedConstraint(target, thr)]
            @test only(PO.constraint_targets(two)).second == [thr, thr]
        end
        for target in (:wb, :lt, :st, :rkb)
            two = PO.AbstractConstraintResult[PO.TargetedConstraint(target, thr),
                                              PO.TargetedConstraint(target, thr)]
            @test_throws ArgumentError PO.constraint_targets(two)
        end
        # Two writes to `:cte` fold into one constraint carrying both blocks of rows.
        two_cte = PO.AbstractConstraintResult[PO.TargetedConstraint(:cte, lc),
                                              PO.TargetedConstraint(:cte, lc)]
        folded = only(PO.constraint_targets(two_cte)).second
        @test folded isa LinearConstraint
        @test folded.ineq.A == [lc.ineq.A; lc.ineq.A]
        @test folded.ineq.B == [lc.ineq.B; lc.ineq.B]
        # A value that is not a LinearConstraint cannot be folded, and says so.
        @test_throws ArgumentError PO.constraint_targets(PO.AbstractConstraintResult[PO.TargetedConstraint(:cte,
                                                                                                           thr),
                                                                                     PO.TargetedConstraint(:cte,
                                                                                                           thr)])
    end

    @testset "n centrality steps equal one cte field holding n estimators" begin
        #=
        The equivalence the fold exists to preserve. `centrality_constraints` over a vector
        of estimators appends every row into one result, so a pipeline running one step per
        estimator must reach the optimiser with that same constraint — otherwise the two
        ways of writing the same mandate would build different models.
        =#
        rd = ReturnsResult(; nx = string.("A", 1:5),
                           X = randn(StableRNG(987654321), 60, 5) / 100)
        cc1 = CentralityConstraint(; B = MinValue(), comp = <=)
        cc2 = CentralityConstraint(; B = MaxValue(), comp = >=)
        cc3 = CentralityConstraint(; B = MinValue(), comp = ==)
        manual = PO.centrality_constraints([cc1, cc2, cc3], rd)

        ctx = PO.PipelineContext(; returns = rd)
        for cc in (cc1, cc2, cc3)
            _, ctx = PO.run_step(cc, ctx)
        end
        target, stepped = only(PO.constraint_targets(ctx.constraints))
        @test target === :cte
        @test stepped isa LinearConstraint
        @test stepped.ineq.A == manual.ineq.A
        @test stepped.ineq.B == manual.ineq.B
        @test stepped.eq.A == manual.eq.A
        @test stepped.eq.B == manual.eq.B
        #=
        And the merged constraint is what the optimiser actually receives. The fold builds
        a fresh constraint per call, so this is an equality of rows rather than of objects
        — unlike a packed target, which routes the very results the steps produced.
        =#
        routed = PO.inject_context(MeanRisk(; opt = jo()), ctx)
        @test routed.opt.cte.ineq.A == manual.ineq.A
        @test routed.opt.cte.ineq.B == manual.ineq.B
        @test routed.opt.cte.eq.A == manual.eq.A
        @test routed.opt.cte.eq.B == manual.eq.B
    end

    @testset "every constraint family declares where its step lands" begin
        #=
        The finding this locks: a family with a run_step but no routing target computed a
        result the injection seam could not place, so the Pipeline fitted every earlier
        step and then always threw — under cross_val_predict, after the first fold. One
        declaration now drives the step, the construction check and the fan-out, so a new
        family cannot be half-wired.
        =#
        function concrete_subtypes(T)
            out = Any[]
            for S in subtypes(T)
                isabstracttype(S) ? append!(out, concrete_subtypes(S)) : push!(out, S)
            end
            return out
        end
        families = [S
                    for S in concrete_subtypes(PO.AbstractConstraintEstimator)
                    if parentmodule(S) === PO && !(S <: PO.JuMPConstraintEstimator)]
        @test !isempty(families)
        #=
        Asked of the type rather than of an instance: a family declares by defining a
        method more specific than the empty fallback, and that is exactly what the
        fallback's own signature tests for.
        =#
        fallback = which(PO.pipe_constraint_targets, Tuple{PO.AbstractConstraintEstimator})
        for S in families
            @test which(PO.pipe_constraint_targets, Tuple{S}) !== fallback
        end
        # Every declared target is a real one, and every choice is a legal annotation.
        @test all(t -> t in PO.PIPELINE_ROUTING_TARGETS, PO.PIPELINE_THRESHOLD_TARGETS)
        @test all(t -> t in PO.PIPELINE_STEP_TARGETS, PO.PIPELINE_THRESHOLD_TARGETS)
        @test all(t -> t in PO.PIPELINE_ROUTING_TARGETS,
                  PO.PIPELINE_ASSET_SETS_MATRIX_TARGETS)
        @test all(t -> t in PO.PIPELINE_STEP_TARGETS, PO.PIPELINE_ASSET_SETS_MATRIX_TARGETS)
        @test all(t -> t in PO.PIPELINE_ROUTING_TARGETS, PO.PIPELINE_ACCUMULATING_TARGETS)
    end

    @testset "each family computes a value and pairs it with its target" begin
        #=
        The end-to-end half of the finding: a family with a step but no target computed a
        result and then always threw at injection. Running each family's step here and
        routing what it produced is what shows the two halves meet.
        =#
        rd = ReturnsResult(; nx = string.("A", 1:5),
                           X = randn(StableRNG(987654321), 60, 5) / 100)
        ctx0 = PO.PipelineContext(; returns = rd)
        steps = ["WeightBoundsEstimator" =>
                     (WeightBoundsEstimator(; lb = 0.05, ub = 0.5), nothing, :wb),
                 "LinearConstraintEstimator" =>
                     (LinearConstraintEstimator(; val = "A1 <= 0.3"), nothing, :lcse),
                 "ThresholdEstimator" => (ThresholdEstimator(; val = 0.05), :st, :st),
                 # The scalar resolves to `RiskBudget(1.0)`, which `test_01_structs.jl`
                 # asserts. This row asserts the routing target only.
                 "RiskBudgetEstimator" => (RiskBudgetEstimator(; val = 0.2), nothing, :rkb),
                 "CentralityConstraint" => (CentralityConstraint(), nothing, :cte),
                 "IntegerPhylogenyEstimator" =>
                     (IntegerPhylogenyEstimator(; B = 1), nothing, :ple)]
        for (name, (est, declared, target)) in steps
            step = if isnothing(declared)
                est
            else
                PipelineStep(; est = est, reads = (:returns,), writes = :constraints,
                             target = declared)
            end
            _, ctx = PO.run_step(step, ctx0)
            @test PO.constraint_targets(ctx.constraints)[1].first === target
        end
        # An unwrapped uncertainty step still refuses; the constraint rule did not widen it.
        @test_throws ArgumentError PO.run_step(NormalUncertaintySet(), ctx0)
    end

    @testset "a constraint step is refused when the optimiser cannot receive it" begin
        thr_step(t) = PipelineStep(; est = ThresholdEstimator(; val = 0.05),
                                   reads = (:returns,), writes = :constraints, target = t)
        #=
        A ThresholdEstimator names six fields, so an unwrapped step cannot be placed and
        the Pipeline says so at construction rather than at injection.
        =#
        @test_throws ArgumentError Pipeline(;
                                            steps = (ThresholdEstimator(; val = 0.05),
                                                     MeanRisk(; opt = jo())))
        # A target that belongs to another family is refused where it is written.
        @test_throws ArgumentError Pipeline(;
                                            steps = (thr_step(:smtx),
                                                     MeanRisk(; opt = jo())))
        # Declared correctly, it constructs; in front of an optimiser with no such field,
        # it does not.
        @test length(Pipeline(; steps = (thr_step(:lt), MeanRisk(; opt = jo()))).steps) == 2
        @test_throws ArgumentError Pipeline(; steps = (thr_step(:lt), EqualWeighted()))
        #=
        A risk budget names one field, so it needs no annotation — and only the two
        optimisers carrying a risk-budgeting algorithm can receive it.
        =#
        rkb_est = RiskBudgetEstimator(; val = 0.2)
        @test length(Pipeline(; steps = (rkb_est, RiskBudgeting(; opt = jo()))).steps) == 2
        @test_throws ArgumentError Pipeline(; steps = (rkb_est, MeanRisk(; opt = jo())))
        #=
        A CentralityConstraint produces a LinearConstraint, which would fan out to :lcse
        by type; the family declares :cte instead, so the constraint keeps its own model
        keys rather than being merged into the general linear block.
        =#
        @test PO.pipe_constraint_targets(CentralityConstraint()) == (:cte,)
        @test length(Pipeline(; steps = (CentralityConstraint(), MeanRisk(; opt = jo()))).steps) ==
              2
        # A JuMP-model constraint is configuration, not a step, and is refused as one.
        @test_throws ArgumentError Pipeline(;
                                            steps = (BudgetRange(; lb = 0.9, ub = 1.1),
                                                     MeanRisk(; opt = jo())))
    end
end
