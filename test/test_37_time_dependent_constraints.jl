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
# `TDOptCap` declares that its per-fold value is an optimiser, which is what makes a schedule
# holding it statically admissible in an optimiser-valued field.
struct TDOptCap <: PortfolioOptimisers.TimeDependentOptimiserCallable end
function (c::TDOptCap)(ctx::TimeDependentContext)
    return isodd(ctx.i) ? EqualWeighted() : InverseVolatility()
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
        # bind defaults to :outermost and only accepts :outermost or :nearest.
        @test td.bind == :outermost
        @test TimeDependent([1.0], :nearest).bind == :nearest
        @test TimeDependent(; val = [1.0], bind = :nearest).bind == :nearest
        @test_throws ArgumentError TimeDependent([1.0], :sideways)
        @test_throws ArgumentError TimeDependent(; val = [1.0], bind = :sideways)
        # port_opt_view preserves bind.
        @test PortfolioOptimisers.port_opt_view(TimeDependent([1.0, 2.0], :nearest), 1).bind ==
              :nearest
    end
    @testset "Schedules do not nest" begin
        inner = TimeDependent([1.0, 2.0])
        @test_throws ArgumentError TimeDependent(inner)
        @test_throws ArgumentError TimeDependent(; val = inner)
        @test_throws ArgumentError TimeDependent([inner, 2.0])
        @test_throws ArgumentError TimeDependent([1.0, 2.0]; default = inner)
    end
    @testset "Fold-less defaults" begin
        # No default: the field resets to its host's static default (`wb → WeightBounds()`).
        td = TimeDependent([WeightBounds(; lb = 0.0, ub = 0.3),
                            WeightBounds(; lb = 0.0, ub = 0.4)])
        @test td.default === NoDefault()
        wb0 = PortfolioOptimisers.reset_time_dependent_estimator(JuMPOptimiser(; slv = slv,
                                                                               wb = td)).wb
        @test wb0.lb == 0.0 && wb0.ub == 1.0
        # An explicit default overrides it.
        tdd = TimeDependent([WeightBounds(; lb = 0.0, ub = 0.3),
                             WeightBounds(; lb = 0.0, ub = 0.4)];
                            default = WeightBounds(; lb = 0.0, ub = 0.9))
        @test PortfolioOptimisers.reset_time_dependent_estimator(JuMPOptimiser(; slv = slv,
                                                                               wb = tdd)).wb.ub ==
              0.9
        # The default is test-substituted through the host constructor at construction time,
        # so a type-incompatible or invalid default fails there, not mid-backtest.
        @test_throws TypeError JuMPOptimiser(; slv = slv,
                                             card = TimeDependent([2, 3];
                                                                  default = Threshold(;
                                                                                      val = 0.1)))
        @test_throws DomainError JuMPOptimiser(; slv = slv,
                                               card = TimeDependent([2, 3]; default = 0))
        @test JuMPOptimiser(; slv = slv, card = TimeDependent([2, 3]; default = 5)).card.default ==
              5
        # A required field — one whose host declares `NoDefault()` because it has no static
        # default — has no fold-less value unless the schedule supplies one.
        ew, iv = EqualWeighted(), InverseVolatility()
        @test_throws PortfolioOptimisers.TimeDependentDefaultError PortfolioOptimisers.time_dependent_reset_value(TimeDependent([ew,
                                                                                                                                 iv]),
                                                                                                                  (;
                                                                                                                   opti = NoDefault()),
                                                                                                                  :opti,
                                                                                                                  ew)
        @test PortfolioOptimisers.time_dependent_reset_value(TimeDependent([ew, iv];
                                                                           default = ew),
                                                             (; opti = NoDefault()), :opti,
                                                             ew) === ew
        # port_opt_view carries the default through alongside the entries.
        v = PortfolioOptimisers.port_opt_view(tdd, 1)
        @test v.bind == :outermost && v.default.ub == 0.9
    end
    @testset "Optimiser-position bound" begin
        ew, iv = EqualWeighted(), InverseVolatility()
        res = optimise(ew, rd)
        # Statically admissible: a schedule of optimisers, a mixed schedule of an optimiser
        # and a precomputed result, and a declared optimiser callable.
        @test TimeDependent([ew, iv]) isa PortfolioOptimisers.TD_OptE_Opt
        @test TimeDependent([ew, res]) isa PortfolioOptimisers.TD_OptE_Opt
        @test TimeDependent(TDOptCap()) isa PortfolioOptimisers.TD_OptE_Opt
        # Admitted, and checked only when the fold loop swaps the value in.
        @test TimeDependent(ctx -> ew) isa PortfolioOptimisers.TD_OptE_Opt
        @test TimeDependent(PreviousWeightsFunction(ctx -> ew)) isa
              PortfolioOptimisers.TD_OptE_Opt
        # A schedule of constraint values is not an optimiser schedule.
        @test !(TimeDependent([Fees(; l = 0.001)]) isa PortfolioOptimisers.TD_OptE_Opt)
        # The optional form additionally admits `nothing` and the static value.
        TDO = PortfolioOptimisers.TDO_Option{<:PortfolioOptimisers.OptE_Opt}
        @test nothing isa TDO
        @test ew isa TDO
        @test TimeDependent([ew, iv]) isa TDO
        @test !(TimeDependent([Fees(; l = 0.001)]) isa TDO)
        # A declared optimiser callable resolves per fold like any other callable.
        ctx = TimeDependentContext(; i = 2, n = 2, rd = rd, train_idx = [1:100, 1:150],
                                   test_idx = [101:150, 151:200])
        @test PortfolioOptimisers.time_dependent_value(TimeDependent(TDOptCap()), ctx) isa
              InverseVolatility
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
    @testset "Naive optimisers" begin
        sched = TimeDependent([WeightBounds(; lb = 0.0, ub = 0.9),
                               WeightBounds(; lb = 0.0, ub = 0.8)])
        ew = EqualWeighted(; wb = sched)
        @test PortfolioOptimisers.is_time_dependent(ew)
        @test PortfolioOptimisers.time_dependent_fields(ew) == (:wb,)
        @test !PortfolioOptimisers.needs_previous_weights(ew)
        # The fallback participates in the trait, the fold-count assertion, and the
        # per-fold update.
        ewfb = EqualWeighted(; fb = EqualWeighted(; wb = sched))
        @test PortfolioOptimisers.is_time_dependent(ewfb)
        @test isempty(PortfolioOptimisers.time_dependent_fields(ewfb))
        @test_throws DimensionMismatch PortfolioOptimisers.assert_time_dependent_fold_count(ewfb,
                                                                                            5)
        @test isnothing(PortfolioOptimisers.assert_time_dependent_fold_count(ewfb, 2))
        ctx2 = TimeDependentContext(; i = 2, n = 2, rd = rd, train_idx = [1:100, 1:150],
                                    test_idx = [101:150, 151:200])
        ew2 = PortfolioOptimisers.update_time_dependent_estimator(ew, ctx2)
        @test ew2.wb.ub == 0.8
        ewfb2 = PortfolioOptimisers.update_time_dependent_estimator(ewfb, ctx2)
        @test ewfb2.fb.wb.ub == 0.8
        @test !PortfolioOptimisers.is_time_dependent(ewfb2)
        # A previous-weights callable in a naive field forces sequential execution.
        @test PortfolioOptimisers.needs_previous_weights(EqualWeighted(;
                                                                       wb = TimeDependent(PreviousWeightsFunction(ctx -> WeightBounds()))))
        @test PortfolioOptimisers.needs_previous_weights(EqualWeighted(;
                                                                       fb = EqualWeighted(;
                                                                                          wb = TimeDependent(PreviousWeightsFunction(ctx -> WeightBounds())))))
        # Reset restores each type's static default: WeightBounds() for EqualWeighted
        # and InverseVolatility, nothing for RandomWeighted.
        @test PortfolioOptimisers.reset_time_dependent_estimator(ew).wb == WeightBounds()
        @test PortfolioOptimisers.reset_time_dependent_estimator(InverseVolatility(;
                                                                                   wb = sched)).wb ==
              WeightBounds()
        @test isnothing(PortfolioOptimisers.reset_time_dependent_estimator(RandomWeighted(;
                                                                                          wb = sched)).wb)
        @test PortfolioOptimisers.reset_time_dependent_estimator(ewfb).fb.wb ==
              WeightBounds()
        # Inert outside fold loops.
        res = optimise(ew, rd)
        @test isa(res.retcode, OptimisationSuccess)
        @test res.w == optimise(EqualWeighted(), rd).w
        ivsched = InverseVolatility(; wb = sched)
        @test optimise(ivsched, rd).w == optimise(InverseVolatility(), rd).w
        # Walk-forward end to end: fold 2 caps the first asset.
        cvw = IndexWalkForward(100, 50)
        tdcap = TimeDependent([WeightBounds(),
                               WeightBounds(; lb = 0.0, ub = [0.1, 1.0, 1.0, 1.0, 1.0])])
        ewp = cross_val_predict(EqualWeighted(; wb = tdcap), rd, cvw)
        @test length(ewp.pred) == 2
        @test all(isapprox(0.2), ewp.pred[1].res.w)
        @test ewp.pred[2].res.w[1] <= 0.1 + 1e-6
        @test !isapprox(ewp.pred[1].res.w, ewp.pred[2].res.w)
    end
    @testset "Meta-optimisers: outer and inner cross-validation" begin
        mr0 = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        # Standalone meta: the inner CV leg consumes the inner estimator's schedule
        # against the INNER folds, while the fold-less full-window inner solves reset
        # themselves and never evaluate it.
        seen3 = zeros(Int, 3)
        obs3 = TimeDependent(ctx -> begin
                                 seen3[ctx.i] += 1
                                 WeightBounds(; lb = 0.0, ub = 0.9)
                             end)
        mr_obs = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = obs3))
        st = Stacking(; opti = [mr_obs, mr0], opto = mr0,
                      cv = OptimisationCrossValidation(; cv = KFold(; n = 3)))
        stres = optimise(st, rd)
        @test isa(stres.retcode, OptimisationSuccess)
        @test seen3 == [1, 1, 1]
        # A vector schedule sized to the inner CV works standalone; a mis-sized one
        # fails at the inner split.
        l2_3 = TimeDependent([1e-4, 2e-4, 3e-4])
        st3 = Stacking(;
                       opti = [MeanRisk(; opt = JuMPOptimiser(; slv = slv, l2 = l2_3)),
                               mr0], opto = mr0,
                       cv = OptimisationCrossValidation(; cv = KFold(; n = 3)))
        @test isa(optimise(st3, rd).retcode, OptimisationSuccess)
        l2_2 = TimeDependent([1e-4, 2e-4])
        st2 = Stacking(;
                       opti = [MeanRisk(; opt = JuMPOptimiser(; slv = slv, l2 = l2_2)),
                               mr0], opto = mr0,
                       cv = OptimisationCrossValidation(; cv = KFold(; n = 3)))
        @test_throws DimensionMismatch optimise(st2, rd)
        # Under an outer fold loop the OUTERMOST CV wins: the same 2-entry schedule now
        # matches the outer walk-forward (2 folds), the callable sees only outer fold
        # indices, and the inner KFold(3) receives already-static estimators.
        cvw = IndexWalkForward(100, 50)
        seen2 = zeros(Int, 2)
        obs2 = TimeDependent(ctx -> begin
                                 seen2[ctx.i] += 1
                                 WeightBounds(; lb = 0.0, ub = 0.9)
                             end)
        st_outer = Stacking(;
                            opti = [MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = obs2)),
                                    mr0], opto = mr0,
                            cv = OptimisationCrossValidation(; cv = KFold(; n = 3)))
        opreds = cross_val_predict(st_outer, rd, cvw)
        @test length(opreds.pred) == 2
        @test seen2 == [1, 1]
        @test isa(optimise(PortfolioOptimisers.update_time_dependent_estimator(st_outer,
                                                                               TimeDependentContext(;
                                                                                                    i = 1,
                                                                                                    n = 2,
                                                                                                    rd = rd,
                                                                                                    train_idx = [1:100],
                                                                                                    test_idx = [101:150])),
                           rd).retcode, OptimisationSuccess)
        # A schedule sized to the inner CV fails at the OUTER split when an outer fold
        # loop is present.
        st_outer3 = Stacking(;
                             opti = [MeanRisk(;
                                              opt = JuMPOptimiser(; slv = slv, l2 = l2_3)),
                                     mr0], opto = mr0,
                             cv = OptimisationCrossValidation(; cv = KFold(; n = 3)))
        @test_throws DimensionMismatch cross_val_predict(st_outer3, rd, cvw)
        # NestedClustered under an outer walk-forward with a different inner CV.
        nco = NestedClustered(;
                              opti = MeanRisk(;
                                              opt = JuMPOptimiser(; slv = slv, l2 = l2_2)),
                              opto = mr0,
                              cv = OptimisationCrossValidation(; cv = KFold(; n = 3)))
        npreds = cross_val_predict(nco, rd, cvw)
        @test length(npreds.pred) == 2
        # The meta's own wb schedule is resolved against the outer folds and caps the
        # combined weights; standalone it is inert.
        srsched = TimeDependent([WeightBounds(), WeightBounds(; lb = 0.15, ub = 0.25)])
        sr = SubsetResampling(; opt = mr0, wb = srsched, subset_size = 3, n_subsets = 2,
                              rng = StableRNG(7), seed = 11)
        spreds = cross_val_predict(sr, rd, cvw)
        @test length(spreds.pred) == 2
        @test all(x -> 0.15 - 1e-6 <= x <= 0.25 + 1e-6, spreds.pred[2].res.w)
        sr0 = SubsetResampling(; opt = mr0, subset_size = 3, n_subsets = 2,
                               rng = StableRNG(7), seed = 11)
        @test optimise(sr, rd).w == optimise(sr0, rd).w
    end
    @testset "Bind: nearest vs outermost fold loops" begin
        mr0 = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        cvw = IndexWalkForward(100, 50)  # 2 outer folds over the 200-row sample
        # On a plain (non-meta) estimator the single fold loop is BOTH outermost and
        # nearest, so a :nearest schedule is consumed by it exactly like :outermost:
        # it must be sized to that loop's folds (2), and a mis-sized one still fails.
        mr_near2 = MeanRisk(;
                            opt = JuMPOptimiser(; slv = slv,
                                                l2 = TimeDependent([1e-4, 2e-4], :nearest)))
        @test length(cross_val_predict(mr_near2, rd, cvw).pred) == 2
        mr_near3 = MeanRisk(;
                            opt = JuMPOptimiser(; slv = slv,
                                                l2 = TimeDependent([1e-4, 2e-4, 3e-4],
                                                                   :nearest)))
        @test_throws DimensionMismatch cross_val_predict(mr_near3, rd, cvw)
        # The discriminating case: an inner estimator's schedule sized to the INNER
        # KFold(3) under an outer walk-forward (2 folds). With the default :outermost
        # the outer loop tries to consume it and the size-3 schedule fails at the outer
        # split; with :nearest the outer loop skips it and the inner KFold(3) consumes it.
        l2_3 = [1e-4, 2e-4, 3e-4]
        st_out = Stacking(;
                          opti = [MeanRisk(;
                                           opt = JuMPOptimiser(; slv = slv,
                                                               l2 = TimeDependent(l2_3))),
                                  mr0], opto = mr0,
                          cv = OptimisationCrossValidation(; cv = KFold(; n = 3)))
        @test_throws DimensionMismatch cross_val_predict(st_out, rd, cvw)
        st_near = Stacking(;
                           opti = [MeanRisk(;
                                            opt = JuMPOptimiser(; slv = slv,
                                                                l2 = TimeDependent(l2_3,
                                                                                   :nearest))),
                                   mr0], opto = mr0,
                           cv = OptimisationCrossValidation(; cv = KFold(; n = 3)))
        npreds = cross_val_predict(st_near, rd, cvw)
        @test length(npreds.pred) == 2
        @test isa(npreds.pred[1].res.retcode, OptimisationSuccess)
        # Standalone, the meta's inner CV is the nearest (and only) loop, so a :nearest
        # schedule sized to the inner KFold(3) is consumed there.
        st_solo = Stacking(;
                           opti = [MeanRisk(;
                                            opt = JuMPOptimiser(; slv = slv,
                                                                l2 = TimeDependent(l2_3,
                                                                                   :nearest))),
                                   mr0], opto = mr0,
                           cv = OptimisationCrossValidation(; cv = KFold(; n = 3)))
        @test isa(optimise(st_solo, rd).retcode, OptimisationSuccess)
        # Mixed binds coexist: the meta's own wb is :outermost (sized to the 2 outer
        # folds) while the inner estimator's l2 is :nearest (sized to the inner KFold(3)),
        # each validated at its own split.
        st_mix = Stacking(;
                          opti = [MeanRisk(;
                                           opt = JuMPOptimiser(; slv = slv,
                                                               l2 = TimeDependent(l2_3,
                                                                                  :nearest))),
                                  mr0], opto = mr0,
                          wb = TimeDependent([WeightBounds(),
                                              WeightBounds(; lb = 0.0, ub = 0.9)]),
                          cv = OptimisationCrossValidation(; cv = KFold(; n = 3)))
        @test length(cross_val_predict(st_mix, rd, cvw).pred) == 2
    end
    @testset "Per-fold vectors of constraints" begin
        mr0 = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        cvw = IndexWalkForward(100, 50)
        # A field that statically accepts a vector of constraints holds a per-fold vector
        # of such vectors: entry i is fold i's whole constraint vector. Here fold 1 gets
        # one linear constraint and fold 2 gets two.
        sets = AssetSets(; dict = Dict("nx" => ["A", "B", "C", "D", "E"]))
        lce1 = LinearConstraintEstimator(; val = "A <= 0.5")
        lce2 = LinearConstraintEstimator(; val = "B <= 0.5")
        mr_vv = MeanRisk(;
                         opt = JuMPOptimiser(; slv = slv, sets = sets,
                                             lcse = TimeDependent([[lce1], [lce1, lce2]])))
        pvv = cross_val_predict(mr_vv, rd, cvw)
        @test length(pvv.pred) == 2
        @test isa(pvv.pred[1].res.retcode, OptimisationSuccess)
        @test isa(pvv.pred[2].res.retcode, OptimisationSuccess)
        # The whole per-fold vector is returned by time_dependent_value.
        ctx1 = TimeDependentContext(; i = 1, n = 2, rd = rd, train_idx = [1:100, 1:150],
                                    test_idx = [101:150, 151:200])
        ctx2 = TimeDependentContext(; i = 2, n = 2, rd = rd, train_idx = [1:100, 1:150],
                                    test_idx = [101:150, 151:200])
        td_vv = TimeDependent([[lce1], [lce1, lce2]])
        @test length(PortfolioOptimisers.time_dependent_value(td_vv, ctx1)) == 1
        @test length(PortfolioOptimisers.time_dependent_value(td_vv, ctx2)) == 2
        # A mis-sized schedule still fails the fold-count check.
        mr_bad = MeanRisk(;
                          opt = JuMPOptimiser(; slv = slv, sets = sets,
                                              lcse = TimeDependent([[lce1]])))
        @test_throws DimensionMismatch cross_val_predict(mr_bad, rd, cvw)
        # needs_previous_weights descends into per-fold vector entries.
        w0 = fill(0.2, 5)
        tds = TimeDependent([[Turnover(; val = 0.1, w = w0)],
                             [Turnover(; val = 0.2, w = w0)]])
        @test PortfolioOptimisers.needs_previous_weights(tds)
        # The callable form assembles the fold's whole constraint vector, keeping shared
        # static parts in one place — here a per-fold dynamic cap plus a fixed one.
        mr_fn = MeanRisk(;
                         opt = JuMPOptimiser(; slv = slv, sets = sets,
                                             lcse = TimeDependent(ctx -> [LinearConstraintEstimator(;
                                                                                                    val = "A <= $(0.5 - 0.05 * (ctx.i - 1))"),
                                                                          lce2])))
        pfn = cross_val_predict(mr_fn, rd, cvw)
        @test length(pfn.pred) == 2
        @test all(p -> isa(p.res.retcode, OptimisationSuccess), pfn.pred)
        # The callable may also return vectors that differ in length per fold (the
        # vector-of-vectors shape, computed rather than listed).
        struct LC_Callable <: PortfolioOptimisers.TimeDependentCallable end
        function (c::LC_Callable)(ctx::TimeDependentContext)
            return if ctx.i == 1
                [LinearConstraintEstimator(; val = "A <= $(0.5 - 0.05 * (ctx.i - 1))")]
            else
                [LinearConstraintEstimator(; val = "A <= 0.4"), lce2]
            end
        end
        mr_fn_vv = MeanRisk(;
                            opt = JuMPOptimiser(; slv = slv, sets = sets,
                                                lcse = TimeDependent(ctx -> if ctx.i == 1
                                                                         [LinearConstraintEstimator(;
                                                                                                    val = "A <= $(0.5 - 0.05 * (ctx.i - 1))")]
                                                                     else
                                                                         [LinearConstraintEstimator(;
                                                                                                    val = "A <= 0.4"),
                                                                          lce2]
                                                                     end)))
        pfn_vv = cross_val_predict(mr_fn_vv, rd, cvw)
        @test length(pfn_vv.pred) == 2
        @test all(p -> isa(p.res.retcode, OptimisationSuccess), pfn_vv.pred)
        # time_dependent_value invokes the callable and returns its per-fold vector.
        td_fn_vv = TimeDependent(ctx -> ctx.i == 1 ? [lce2] : [lce1, lce2])
        @test length(PortfolioOptimisers.time_dependent_value(td_fn_vv, ctx1)) == 1
        @test length(PortfolioOptimisers.time_dependent_value(td_fn_vv, ctx2)) == 2

        mr_fn_vv2 = MeanRisk(;
                             opt = JuMPOptimiser(; slv = slv, sets = sets,
                                                 lcse = TimeDependent(LC_Callable())))
        pfn_vv = cross_val_predict(mr_fn_vv2, rd, cvw)
        @test length(pfn_vv.pred) == 2
        @test all(p -> isa(p.res.retcode, OptimisationSuccess), pfn_vv.pred)
        # time_dependent_value invokes the callable and returns its per-fold vector.
        td_fn_vv = TimeDependent(ctx -> ctx.i == 1 ? [lce2] : [lce1, lce2])
        @test length(PortfolioOptimisers.time_dependent_value(td_fn_vv, ctx1)) == 1
        @test length(PortfolioOptimisers.time_dependent_value(td_fn_vv, ctx2)) == 2
    end
    @testset "Problem-definition fields of the base optimisers" begin
        # The whole problem definition varies over folds, not just the constraints: the
        # prior estimator, returns model, scalariser and asset sets of a `JuMPOptimiser`,
        # and the prior, clustering estimator, sets and weight finaliser of a
        # `HierarchicalOptimiser`. Execution control (`slv`, `sc`, `so`, `brt`, `cle_pr`,
        # `strict`) stays static.
        cvw = IndexWalkForward(100, 50)
        pe_semi = EmpiricalPrior(;
                                 ce = PortfolioOptimisersCovariance(;
                                                                    ce = Covariance(;
                                                                                    alg = SemiMoment())))
        tdpe = TimeDependent([EmpiricalPrior(), pe_semi])
        tdret = TimeDependent([ArithmeticReturn(), ArithmeticReturn(; lb = 0.0005)])
        tdsca = TimeDependent([SumScalariser(), MaxScalariser()])
        setsA = AssetSets(; dict = Dict("nx" => rd.nx, "g1" => ["A", "B"]))
        setsB = AssetSets(; dict = Dict("nx" => rd.nx, "g1" => ["D", "E"]))
        tdsets = TimeDependent([setsA, setsB])
        @testset "Widened fields, defaults and required-field types" begin
            opt = JuMPOptimiser(; slv = slv, pe = tdpe, ret = tdret, sca = tdsca,
                                sets = tdsets)
            @test PortfolioOptimisers.time_dependent_fields(opt) == (:pe, :sets, :ret, :sca)
            # Fold-less: each field falls back to its registered static default.
            r0 = PortfolioOptimisers.reset_time_dependent_estimator(opt)
            @test r0.pe == EmpiricalPrior()
            @test r0.ret == ArithmeticReturn()
            @test r0.sca == SumScalariser()
            @test isnothing(r0.sets)
            # Per fold: the schedule's entry is swapped in.
            ctx2 = TimeDependentContext(; i = 2, n = 2, rd = rd, train_idx = 1:150,
                                        test_idx = 151:200)
            r2 = PortfolioOptimisers.update_time_dependent_estimator(opt, ctx2)
            @test r2.pe == pe_semi
            @test r2.ret.lb == 0.0005
            @test isa(r2.sca, MaxScalariser)
            @test r2.sets === setsB
            # These fields are typed `TD` rather than `TD_Option`: they always carry a
            # value, so `nothing` stays inadmissible.
            @test_throws TypeError JuMPOptimiser(; slv = slv, pe = nothing)
            @test_throws TypeError JuMPOptimiser(; slv = slv, ret = nothing)
            @test_throws TypeError JuMPOptimiser(; slv = slv, sca = nothing)
            @test_throws TypeError HierarchicalOptimiser(; cle = nothing)
            @test_throws TypeError HierarchicalOptimiser(; wf = nothing)
            # The constructor test-substitutes each entry, so a wrongly-typed one throws
            # here rather than mid-backtest.
            @test_throws TypeError JuMPOptimiser(; slv = slv,
                                                 pe = TimeDependent([EmpiricalPrior(), 1.0]))
            hopt = HierarchicalOptimiser(; slv = slv, pe = tdpe, sets = tdsets,
                                         cle = TimeDependent([ClustersEstimator(),
                                                              ClustersEstimator(;
                                                                                alg = HClustAlgorithm(;
                                                                                                      linkage = :single))]),
                                         wf = TimeDependent([IterativeWeightFinaliser(),
                                                             IterativeWeightFinaliser(;
                                                                                      iter = 7)]))
            @test PortfolioOptimisers.time_dependent_fields(hopt) == (:pe, :cle, :sets, :wf)
            h0 = PortfolioOptimisers.reset_time_dependent_estimator(hopt)
            @test h0.pe == EmpiricalPrior()
            @test h0.cle == ClustersEstimator()
            @test h0.wf == IterativeWeightFinaliser()
            @test isnothing(h0.sets)
            h2 = PortfolioOptimisers.update_time_dependent_estimator(hopt, ctx2)
            @test h2.pe == pe_semi
            @test h2.cle.alg.linkage == :single
            @test h2.wf.iter == 7
            @test h2.sets === setsB
        end
        @testset "Per-fold prior estimator" begin
            p = cross_val_predict(MeanRisk(; opt = JuMPOptimiser(; slv = slv, pe = tdpe)),
                                  rd, cvw)
            p0 = cross_val_predict(MeanRisk(; opt = JuMPOptimiser(; slv = slv)), rd, cvw)
            p1 = cross_val_predict(MeanRisk(;
                                            opt = JuMPOptimiser(; slv = slv, pe = pe_semi)),
                                   rd, cvw)
            @test isapprox(p.pred[1].res.w, p0.pred[1].res.w)
            @test isapprox(p.pred[2].res.w, p1.pred[2].res.w)
            @test !isapprox(p0.pred[2].res.w, p1.pred[2].res.w)
        end
        @testset "Per-fold returns model" begin
            p = cross_val_predict(MeanRisk(; opt = JuMPOptimiser(; slv = slv, ret = tdret)),
                                  rd, cvw)
            p0 = cross_val_predict(MeanRisk(; opt = JuMPOptimiser(; slv = slv)), rd, cvw)
            p1 = cross_val_predict(MeanRisk(;
                                            opt = JuMPOptimiser(; slv = slv,
                                                                ret = ArithmeticReturn(;
                                                                                       lb = 0.0005))),
                                   rd, cvw)
            @test isapprox(p.pred[1].res.w, p0.pred[1].res.w)
            @test isapprox(p.pred[2].res.w, p1.pred[2].res.w)
            # The fold-2 minimum-return constraint binds, so it is not the fold-2 baseline.
            @test !isapprox(p0.pred[2].res.w, p1.pred[2].res.w)
        end
        @testset "Per-fold scalariser" begin
            r = [Variance(), ConditionalValueatRisk()]
            p = cross_val_predict(MeanRisk(; r = r,
                                           opt = JuMPOptimiser(; slv = slv, sca = tdsca)),
                                  rd, cvw)
            psum = cross_val_predict(MeanRisk(; r = r,
                                              opt = JuMPOptimiser(; slv = slv,
                                                                  sca = SumScalariser())),
                                     rd, cvw)
            pmax = cross_val_predict(MeanRisk(; r = r,
                                              opt = JuMPOptimiser(; slv = slv,
                                                                  sca = MaxScalariser())),
                                     rd, cvw)
            @test isapprox(p.pred[1].res.w, psum.pred[1].res.w)
            @test isapprox(p.pred[2].res.w, pmax.pred[2].res.w)
            @test !isapprox(psum.pred[2].res.w, pmax.pred[2].res.w)
        end
        @testset "Per-fold asset sets" begin
            # The same group-keyed bound estimator caps a different group of assets each
            # fold, because the sets it resolves against are scheduled.
            wbe = WeightBoundsEstimator(; ub = ["g1" => 0.15])
            p = cross_val_predict(MeanRisk(;
                                           opt = JuMPOptimiser(; slv = slv, wb = wbe,
                                                               sets = tdsets)), rd, cvw)
            w1 = p.pred[1].res.w
            w2 = p.pred[2].res.w
            @test all(x -> x <= 0.15 + 1e-6, w1[1:2])
            @test all(x -> x > 0.15 + 1e-6, w1[4:5])
            @test all(x -> x <= 0.15 + 1e-6, w2[4:5])
            @test all(x -> x > 0.15 + 1e-6, w2[1:2])
        end
        @testset "Per-fold clustering estimator" begin
            single = ClustersEstimator(; alg = HClustAlgorithm(; linkage = :single))
            tdcle = TimeDependent([single, ClustersEstimator()])
            p = cross_val_predict(HierarchicalRiskParity(;
                                                         opt = HierarchicalOptimiser(;
                                                                                     slv = slv,
                                                                                     cle = tdcle)),
                                  rd, cvw)
            p0 = cross_val_predict(HierarchicalRiskParity(;
                                                          opt = HierarchicalOptimiser(;
                                                                                      slv = slv)),
                                   rd, cvw)
            p1 = cross_val_predict(HierarchicalRiskParity(;
                                                          opt = HierarchicalOptimiser(;
                                                                                      slv = slv,
                                                                                      cle = single)),
                                   rd, cvw)
            @test isapprox(p.pred[1].res.w, p1.pred[1].res.w)
            @test isapprox(p.pred[2].res.w, p0.pred[2].res.w)
            @test !isapprox(p0.pred[1].res.w, p1.pred[1].res.w)
        end
    end
    @testset "A schedule as the optimiser itself" begin
        ew, iv = EqualWeighted(), InverseVolatility()
        cvw = IndexWalkForward(100, 50)  # 2 folds over the 200-row sample
        @testset "Each fold runs its own optimiser" begin
            sched = TimeDependent([ew, iv])
            p = cross_val_predict(sched, rd, cvw)
            pew = cross_val_predict(ew, rd, cvw)
            piv = cross_val_predict(iv, rd, cvw)
            @test length(p.pred) == 2
            # Fold 1 is the EqualWeighted run, fold 2 the InverseVolatility one.
            @test isapprox(p.pred[1].res.w, pew.pred[1].res.w)
            @test isapprox(p.pred[2].res.w, piv.pred[2].res.w)
            @test !isapprox(pew.pred[2].res.w, piv.pred[2].res.w)
            # A declared optimiser callable schedules the same way (odd folds EqualWeighted).
            pc = cross_val_predict(TimeDependent(TDOptCap()), rd, cvw)
            @test isapprox(pc.pred[1].res.w, pew.pred[1].res.w)
            @test isapprox(pc.pred[2].res.w, piv.pred[2].res.w)
        end
        @testset "A mis-sized schedule fails at the split" begin
            @test_throws DimensionMismatch cross_val_predict(TimeDependent([ew, iv, ew]),
                                                             rd, cvw)
        end
        @testset "A schedule of results replays per-fold weights" begin
            r1, r2 = optimise(ew, rd), optimise(iv, rd)
            p = cross_val_predict(TimeDependent([r1, r2]), rd, cvw)
            @test length(p.pred) == 2
            # Each entry is predicted on its fold's test window, not re-optimised.
            @test isapprox(p.pred[1].res.w, r1.w)
            @test isapprox(p.pred[2].res.w, r2.w)
        end
        @testset "A mixed schedule optimises or predicts per entry" begin
            res = optimise(iv, rd)
            p = cross_val_predict(TimeDependent([ew, res]), rd, cvw)
            pew = cross_val_predict(ew, rd, cvw)
            @test isapprox(p.pred[1].res.w, pew.pred[1].res.w)
            # Fold 2's entry is a precomputed result: replayed, not refitted, so its weights
            # are the full-sample ones rather than the fold-2 training-window ones.
            @test isapprox(p.pred[2].res.w, res.w)
            @test !isapprox(res.w, cross_val_predict(iv, rd, cvw).pred[2].res.w)
        end
        @testset "Post-swap recursion binds the entry's own schedules to this loop" begin
            # The swapped-in estimator's :outermost schedule is sized to the SAME 2 outer
            # folds and resolved against the same context, so fold 2 gets l2 = 2e-4.
            l2 = TimeDependent([1e-4, 2e-4])
            mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv, l2 = l2))
            p = cross_val_predict(TimeDependent([ew, mr]), rd, cvw)
            mr2 = MeanRisk(; opt = JuMPOptimiser(; slv = slv, l2 = 2e-4))
            @test isapprox(p.pred[2].res.w, cross_val_predict(mr2, rd, cvw).pred[2].res.w)
            # A schedule inside an entry is still sized to this loop, so a mis-sized one
            # fails even though the top-level schedule is correctly sized.
            mr_bad = MeanRisk(;
                              opt = JuMPOptimiser(; slv = slv,
                                                  l2 = TimeDependent([1e-4, 2e-4, 3e-4])))
            @test_throws DimensionMismatch cross_val_predict(TimeDependent([ew, mr_bad]),
                                                             rd, cvw)
        end
        @testset "Fold-less solves need a default" begin
            # A schedule is defined only over folds; a fold-less solve has none.
            @test_throws PortfolioOptimisers.TimeDependentDefaultError optimise(TimeDependent([ew,
                                                                                               iv]),
                                                                                rd)
            # The `default` is the fold-less optimiser.
            @test isapprox(optimise(TimeDependent([ew, iv]; default = iv), rd).w,
                           optimise(iv, rd).w)
            # It is reset in turn, so its own schedules take their defaults too.
            mr = MeanRisk(;
                          opt = JuMPOptimiser(; slv = slv,
                                              l2 = TimeDependent([1e-4, 2e-4];
                                                                 default = 5e-4)))
            mr0 = MeanRisk(; opt = JuMPOptimiser(; slv = slv, l2 = 5e-4))
            @test isapprox(optimise(TimeDependent([ew, mr]; default = mr), rd).w,
                           optimise(mr0, rd).w)
        end
        @testset "A callable's output is checked at the swap" begin
            # Bare callables cannot be checked statically, so a non-optimiser output is
            # caught when the fold loop swaps it in.
            @test_throws ArgumentError cross_val_predict(TimeDependent(ctx -> Fees(;
                                                                                   l = 0.001)),
                                                         rd, cvw)
        end
        @testset "Other cross-validation schemes" begin
            sched = TimeDependent([ew, iv, ew])
            # KFold: 3 non-sequential folds.
            @test length(cross_val_predict(sched, rd, KFold(; n = 3)).pred) == 3
            # Combinatorial: the schedule is indexed by the fold's position in the split
            # enumeration, and predictions are regrouped into paths.
            cvc = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
            @test !isempty(cross_val_predict(TimeDependent(fill(ew, n_splits(cvc))), rd,
                                             cvc).pred)
        end
    end
end
