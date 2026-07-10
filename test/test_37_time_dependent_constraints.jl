# TimeDependentCallable subtypes must be defined at top level. `TDCap` is a plain functor;
# `TDPrevWCap` declares its previous-weights requirement directly.
struct TDCap <: PortfolioOptimisers.TimeDependentCallable
    hi::Float64
    lo::Float64
end
function (c::TDCap)(ctx::TimeDependentContext)
    return WeightBounds(; lb = 0.0,
                        ub = c.hi - (c.hi - c.lo) * (ctx.i - 1) / max(ctx.n - 1, 1))
end
struct TDPrevWCap <: PortfolioOptimisers.TimeDependentCallable end
function (c::TDPrevWCap)(ctx::TimeDependentContext)
    return Threshold(; val = isnothing(ctx.w_prev) ? 0.01 : 0.05)
end
function PortfolioOptimisers.needs_previous_weights(::TDPrevWCap)
    return true
end
@testset "Time-dependent constraints" begin
    using Test, PortfolioOptimisers, Clarabel, StableRNGs
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5) * 0.01 .+ 0.001
    rd = ReturnsResult(; nx = ["A", "B", "C", "D", "E"], X = X)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    @testset "TimeDependent construction" begin
        td = TimeDependent([Threshold(; val = 0.01), Threshold(; val = 0.02)])
        @test td.val[1].val == 0.01
        # Keyword form.
        @test TimeDependent(; val = [Fees(; l = 0.001)]) isa TimeDependent
        # Empty entry vectors are rejected.
        @test_throws PortfolioOptimisers.IsEmptyError TimeDependent(Union{}[])
        # Callables are stored as-is.
        @test TimeDependent(ctx -> Threshold(; val = 0.01)) isa TimeDependent
        @test TimeDependent(TDCap(0.35, 0.2)) isa TimeDependent
        @test TimeDependent(PreviousWeightsFunction(ctx -> WeightBounds())) isa
              TimeDependent
    end
    @testset "Host validation" begin
        td = TimeDependent([Threshold(; val = 0.01), Threshold(; val = 0.02)])
        opt = JuMPOptimiser(; slv = slv, lt = td)
        @test PortfolioOptimisers.is_time_dependent(opt)
        @test PortfolioOptimisers.time_dependent_fields(opt) == (:lt,)
        @test !PortfolioOptimisers.needs_previous_weights(opt)
        # A schedule in a field that cannot hold it is an ordinary construction error.
        @test_throws TypeError JuMPOptimiser(; slv = slv, ret = td)
        # Test-substitution surfaces type-incompatible entries at construction: a
        # Threshold entry cannot be substituted into `card`.
        @test_throws TypeError JuMPOptimiser(; slv = slv,
                                             card = TimeDependent([Threshold(; val = 0.01)]))
        # Static validation still runs on substituted entries (card must be > 0).
        @test_throws DomainError JuMPOptimiser(; slv = slv, card = TimeDependent([0]))
        @test JuMPOptimiser(; slv = slv, card = TimeDependent([2, 3])) isa JuMPOptimiser
        # Hierarchical host.
        htd = TimeDependent([Fees(; l = 0.001), Fees(; l = 0.002)])
        hopt = HierarchicalOptimiser(; slv = slv, fees = htd)
        @test PortfolioOptimisers.is_time_dependent(hopt)
        @test PortfolioOptimisers.time_dependent_fields(hopt) == (:fees,)
        @test_throws TypeError HierarchicalOptimiser(; slv = slv,
                                                     fees = TimeDependent([Threshold(;
                                                                                     val = 0.01)]))
        # Meta hosts accept schedules in their own wb/fees.
        mr0 = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        sr = SubsetResampling(; opt = mr0,
                              wb = TimeDependent([WeightBounds(), WeightBounds()]))
        @test PortfolioOptimisers.is_time_dependent(sr)
        @test PortfolioOptimisers.time_dependent_fields(sr) == (:wb,)
    end
    @testset "Per-fold resolution and wrapper recursion" begin
        td = TimeDependent([Threshold(; val = 0.01), Threshold(; val = 0.02)])
        opt = JuMPOptimiser(; slv = slv, lt = td)
        ctx1 = TimeDependentContext(; i = 1, n = 2, rd = rd, train_idx = [1:100, 1:150],
                                    test_idx = [101:150, 151:200])
        ctx2 = TimeDependentContext(; i = 2, n = 2, rd = rd, train_idx = [1:100, 1:150],
                                    test_idx = [101:150, 151:200])
        o1 = PortfolioOptimisers.update_time_dependent_estimator(opt, ctx1)
        o2 = PortfolioOptimisers.update_time_dependent_estimator(opt, ctx2)
        @test o1.lt.val == 0.01
        @test o2.lt.val == 0.02
        @test !PortfolioOptimisers.is_time_dependent(o1)
        @test !PortfolioOptimisers.is_time_dependent(o2)
        # Wrapper recursion rebuilds the inner optimiser.
        mr = MeanRisk(; opt = opt)
        @test PortfolioOptimisers.is_time_dependent(mr)
        mr2 = PortfolioOptimisers.update_time_dependent_estimator(mr, ctx2)
        @test mr2.opt.lt.val == 0.02
        @test !PortfolioOptimisers.is_time_dependent(mr2)
        # Function form receives the fold context.
        optwb = JuMPOptimiser(; slv = slv,
                              wb = TimeDependent(ctx -> WeightBounds(; lb = 0.0,
                                                                     ub = 1.0 / ctx.i)))
        @test PortfolioOptimisers.update_time_dependent_estimator(optwb, ctx2).wb.ub == 0.5
        # PreviousWeightsFunction declares the prev-weights requirement as data.
        pwf = TimeDependent(PreviousWeightsFunction(TDPrevWCap()))
        optp = JuMPOptimiser(; slv = slv, lt = pwf)
        @test PortfolioOptimisers.needs_previous_weights(optp)
        @test PortfolioOptimisers.update_time_dependent_estimator(optp, ctx1).lt.val == 0.01
        # Fold-count validation.
        @test_throws DimensionMismatch PortfolioOptimisers.assert_time_dependent_fold_count(opt,
                                                                                            5)
        @test isnothing(PortfolioOptimisers.assert_time_dependent_fold_count(opt, 2))
        @test isnothing(PortfolioOptimisers.assert_time_dependent_fold_count(optwb, 5))
        # Callable structs: functor over ctx with direct needs_previous_weights
        # declaration; equivalent to the bare-function form.
        optcap = JuMPOptimiser(; slv = slv, wb = TimeDependent(TDCap(0.35, 0.2)))
        @test !PortfolioOptimisers.needs_previous_weights(optcap)
        optfn = JuMPOptimiser(; slv = slv,
                              wb = TimeDependent(ctx -> WeightBounds(; lb = 0.0,
                                                                     ub = 0.35 -
                                                                          0.15 *
                                                                          (ctx.i - 1) /
                                                                          max(ctx.n - 1, 1))))
        for ctx in (ctx1, ctx2)
            oc = PortfolioOptimisers.update_time_dependent_estimator(optcap, ctx)
            of = PortfolioOptimisers.update_time_dependent_estimator(optfn, ctx)
            @test oc.wb.ub ≈ of.wb.ub
        end
        optpw = JuMPOptimiser(; slv = slv, lt = TimeDependent(TDPrevWCap()))
        @test PortfolioOptimisers.needs_previous_weights(optpw)
        @test PortfolioOptimisers.update_time_dependent_estimator(optpw, ctx1).lt.val ==
              0.01
        # Meta-optimisers resolve their own fields and forward the per-fold update to
        # their inner estimators.
        sr = SubsetResampling(; opt = mr,
                              wb = TimeDependent([WeightBounds(; lb = 0.0, ub = 0.9),
                                                  WeightBounds(; lb = 0.0, ub = 0.8)]))
        @test PortfolioOptimisers.is_time_dependent(sr)
        sr2 = PortfolioOptimisers.update_time_dependent_estimator(sr, ctx2)
        @test sr2.wb.ub == 0.8
        @test sr2.opt.opt.lt.val == 0.02
        @test !PortfolioOptimisers.is_time_dependent(sr2)
        st = Stacking(; opti = [mr, MeanRisk(; opt = JuMPOptimiser(; slv = slv))],
                      opto = MeanRisk(; opt = JuMPOptimiser(; slv = slv)))
        @test PortfolioOptimisers.is_time_dependent(st)
        st2 = PortfolioOptimisers.update_time_dependent_estimator(st, ctx1)
        @test st2.opti[1].opt.lt.val == 0.01
        @test !PortfolioOptimisers.is_time_dependent(st2)
        @test_throws DimensionMismatch PortfolioOptimisers.assert_time_dependent_fold_count(st,
                                                                                            9)
        # Asset views map through vector entries.
        tdv = TimeDependent([WeightBounds(; lb = 0, ub = fill(0.8, 5)),
                             WeightBounds(; lb = zeros(5), ub = 0.9)])
        tdv2 = PortfolioOptimisers.port_opt_view(tdv, 1:3)
        @test length(tdv2.val[1].ub) == 3
        @test length(tdv2.val[1].lb) == 1
        @test length(tdv2.val[2].lb) == 3
        @test length(tdv2.val[2].ub) == 1
    end
    @testset "Shared schedule across hosts" begin
        # One TimeDependent object may be shared between hosts: construction and
        # per-fold updates never mutate it.
        td = TimeDependent([WeightBounds(; lb = 0.0, ub = 0.9),
                            WeightBounds(; lb = 0.0, ub = 0.8)])
        entries = copy(td.val)
        mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = td),
                      fb = HierarchicalRiskParity(;
                                                  opt = HierarchicalOptimiser(; slv = slv,
                                                                              wb = td)))
        @test mr.opt.wb === td
        @test mr.fb.opt.wb === td
        @test td.val == entries
        ctx = TimeDependentContext(; i = 2, n = 2, rd = rd, train_idx = [1:100, 1:150],
                                   test_idx = [101:150, 151:200])
        mr2 = PortfolioOptimisers.update_time_dependent_estimator(mr, ctx)
        @test mr2.opt.wb.ub == 0.8
        @test mr2.fb.opt.wb.ub == 0.8
        @test td.val == entries
        @test mr.opt.wb === td
    end
    @testset "Inert outside fold loops" begin
        td = TimeDependent([Threshold(; val = 0.01), Threshold(; val = 0.02)])
        mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv, lt = td))
        # Reset replaces schedules with the fields' static defaults.
        mrr = PortfolioOptimisers.reset_time_dependent_estimator(mr)
        @test isnothing(mrr.opt.lt)
        @test !PortfolioOptimisers.is_time_dependent(mrr)
        mrw = MeanRisk(;
                       opt = JuMPOptimiser(; slv = slv,
                                           wb = TimeDependent([WeightBounds()]),
                                           bgt = TimeDependent([1.0])))
        mrwr = PortfolioOptimisers.reset_time_dependent_estimator(mrw)
        @test mrwr.opt.wb == WeightBounds()
        @test mrwr.opt.bgt == 1.0
        # A fold-less optimise runs with the affected field at its default.
        res = optimise(mr, rd)
        @test isa(res.retcode, OptimisationSuccess)
        mr0 = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        res0 = optimise(mr0, rd)
        @test res.w == res0.w
        # Hierarchical host is equally inert.
        htd = TimeDependent([Fees(; l = 0.001), Fees(; l = 0.002)])
        hrp = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; slv = slv, fees = htd))
        hres = optimise(hrp, rd)
        @test isa(hres.retcode, OptimisationSuccess)
        hrp0 = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; slv = slv))
        @test hres.w == optimise(hrp0, rd).w
    end
    @testset "Walk-forward end to end" begin
        cv = IndexWalkForward(100, 50)
        n = n_splits(cv, rd)
        @test n == 2
        # Tight fold-2 bound forces different weights across folds.
        tdwb = TimeDependent([WeightBounds(; lb = 0.0, ub = 1.0),
                              WeightBounds(; lb = 0.15, ub = 0.25)])
        mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = tdwb))
        preds = cross_val_predict(mr, rd, cv)
        @test length(preds.pred) == 2
        w1 = preds.pred[1].res.w
        w2 = preds.pred[2].res.w
        @test all(x -> 0.15 - 1e-6 <= x <= 0.25 + 1e-6, w2)
        @test !isapprox(w1, w2)
        # Mis-sized entries fail at split time, before any fold runs.
        td3 = TimeDependent([WeightBounds(), WeightBounds(), WeightBounds()])
        mr3 = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = td3))
        @test_throws DimensionMismatch cross_val_predict(mr3, rd, cv)
        # Baseline without a schedule differs on fold 2.
        mrb = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        predsb = cross_val_predict(mrb, rd, cv)
        @test isapprox(w1, predsb.pred[1].res.w)
        @test !isapprox(w2, predsb.pred[2].res.w)
        # Function form: same bounds computed on the fly.
        tdf = TimeDependent(ctx -> if ctx.i == 1
                                WeightBounds()
                            else
                                WeightBounds(; lb = 0.15, ub = 0.25)
                            end)
        mrf = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = tdf))
        predsf = cross_val_predict(mrf, rd, cv)
        @test isapprox(w2, predsf.pred[2].res.w)
        # PreviousWeightsFunction runs sequentially and sees fold-1 weights.
        seen = Union{Nothing, Vector{Float64}}[]
        pwf = TimeDependent(PreviousWeightsFunction(ctx -> begin
                                                        push!(seen,
                                                              if isnothing(ctx.w_prev)
                                                                  nothing
                                                              else
                                                                  copy(ctx.w_prev)
                                                              end)
                                                        WeightBounds()
                                                    end))
        mrp = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = pwf))
        predsp = cross_val_predict(mrp, rd, cv)
        @test length(seen) == 2
        @test isnothing(seen[1])
        @test isapprox(seen[2], predsp.pred[1].res.w)
    end
    @testset "KFold and hierarchical end to end" begin
        cv = KFold(; n = 4)
        tdwb = TimeDependent([WeightBounds(; lb = 0.15, ub = 0.25), WeightBounds(),
                              WeightBounds(), WeightBounds()])
        mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = tdwb))
        preds = cross_val_predict(mr, rd, cv)
        @test length(preds.pred) == 4
        @test all(x -> 0.15 - 1e-6 <= x <= 0.25 + 1e-6, preds.pred[1].res.w)
        # Hierarchical host under walk-forward.
        cvw = IndexWalkForward(100, 50)
        htd = TimeDependent([Fees(; l = 0.0), Fees(; l = 0.01)])
        hrp = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; slv = slv, fees = htd))
        hpreds = cross_val_predict(hrp, rd, cvw)
        @test length(hpreds.pred) == 2
        @test isnothing(hpreds.pred[1].res.fees) || hpreds.pred[1].res.fees.l == 0.0
        @test hpreds.pred[2].res.fees.l == 0.01
    end
end
