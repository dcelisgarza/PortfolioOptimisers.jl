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
        l2_3 = TimeDependent([L2Regularisation(; val = 1e-4),
                              L2Regularisation(; val = 2e-4),
                              L2Regularisation(; val = 3e-4)])
        st3 = Stacking(;
                       opti = [MeanRisk(; opt = JuMPOptimiser(; slv = slv, l2 = l2_3)),
                               mr0], opto = mr0,
                       cv = OptimisationCrossValidation(; cv = KFold(; n = 3)))
        @test isa(optimise(st3, rd).retcode, OptimisationSuccess)
        l2_2 = TimeDependent([L2Regularisation(; val = 1e-4),
                              L2Regularisation(; val = 2e-4)])
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
                                                l2 = TimeDependent([L2Regularisation(;
                                                                                     val = 1e-4),
                                                                    L2Regularisation(;
                                                                                     val = 2e-4)],
                                                                   :nearest)))
        @test length(cross_val_predict(mr_near2, rd, cvw).pred) == 2
        mr_near3 = MeanRisk(;
                            opt = JuMPOptimiser(; slv = slv,
                                                l2 = TimeDependent([L2Regularisation(;
                                                                                     val = 1e-4),
                                                                    L2Regularisation(;
                                                                                     val = 2e-4),
                                                                    L2Regularisation(;
                                                                                     val = 3e-4)],
                                                                   :nearest)))
        @test_throws DimensionMismatch cross_val_predict(mr_near3, rd, cvw)
        # The discriminating case: an inner estimator's schedule sized to the INNER
        # KFold(3) under an outer walk-forward (2 folds). With the default :outermost
        # the outer loop tries to consume it and the size-3 schedule fails at the outer
        # split; with :nearest the outer loop skips it and the inner KFold(3) consumes it.
        l2_3 = [L2Regularisation(; val = 1e-4), L2Regularisation(; val = 2e-4),
                L2Regularisation(; val = 3e-4)]
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
    @testset "Problem-definition fields of the concrete estimators" begin
        # The concrete estimators' own problem definition varies over folds too: the risk
        # measure, objective, budgeting algorithm, warm starts and the fallback itself.
        # Execution control (`bins`, `ucs_flag`, `alg` variants, `flag`, `ex`, `sq`, `rng`,
        # `seed`, `brt`, `strict`) stays static.
        cvw = IndexWalkForward(100, 50)
        jopt = JuMPOptimiser(; slv = slv)
        ctx2 = TimeDependentContext(; i = 2, n = 2, rd = rd, train_idx = 1:150,
                                    test_idx = 151:200)
        @testset "MeanRisk: widened fields, defaults and the scheduled fallback" begin
            tdr = TimeDependent([Variance(), ConditionalValueatRisk()])
            tdobj = TimeDependent([MinimumRisk(), MaximumUtility(; l = 4.0)])
            tdwi = TimeDependent([fill(0.2, 5), fill(0.2, 5)])
            tdfb = TimeDependent([EqualWeighted(), InverseVolatility()])
            mr = MeanRisk(; opt = jopt, r = tdr, obj = tdobj, wi = tdwi, fb = tdfb)
            @test PortfolioOptimisers.is_time_dependent(mr)
            @test PortfolioOptimisers.time_dependent_fields(mr) == (:r, :obj, :wi, :fb)
            # Fold-less: r/obj reset to their registered static defaults, wi/fb to nothing.
            mr0 = PortfolioOptimisers.reset_time_dependent_estimator(mr)
            @test mr0.r == Variance()
            @test mr0.obj == MinimumRisk()
            @test isnothing(mr0.wi)
            @test isnothing(mr0.fb)
            @test !PortfolioOptimisers.is_time_dependent(mr0)
            # Per fold: entry i is swapped into each field.
            mr2 = PortfolioOptimisers.update_time_dependent_estimator(mr, ctx2)
            @test isa(mr2.r, ConditionalValueatRisk)
            @test isa(mr2.obj, MaximumUtility)
            @test mr2.wi == fill(0.2, 5)
            @test isa(mr2.fb, InverseVolatility)
            @test !PortfolioOptimisers.is_time_dependent(mr2)
            # A scheduled fallback with a `default` survives a fold-less reset as that
            # default.
            mrd = MeanRisk(; opt = jopt,
                           fb = TimeDependent([EqualWeighted(), InverseVolatility()];
                                              default = EqualWeighted()))
            @test PortfolioOptimisers.reset_time_dependent_estimator(mrd).fb ==
                  EqualWeighted()
            # Required fields stay non-nothing (typed TD, not TD_Option).
            @test_throws TypeError MeanRisk(; opt = jopt, r = nothing)
            @test_throws TypeError MeanRisk(; opt = jopt, obj = nothing)
            # The constructor test-substitutes each entry, so a wrongly-typed or invalid
            # one throws at construction rather than mid-backtest.
            @test_throws TypeError MeanRisk(; opt = jopt,
                                            r = TimeDependent([Variance(), 1.0]))
            @test_throws TypeError MeanRisk(; opt = jopt,
                                            obj = TimeDependent([MinimumRisk(), Variance()]))
            @test_throws PortfolioOptimisers.IsEmptyError MeanRisk(; opt = jopt,
                                                                   wi = TimeDependent([Float64[],
                                                                                       fill(0.2,
                                                                                            5)]))
            # The schedules participate in the fold-count assertion, entries included.
            @test_throws DimensionMismatch PortfolioOptimisers.assert_time_dependent_fold_count(mr,
                                                                                                5)
            @test isnothing(PortfolioOptimisers.assert_time_dependent_fold_count(mr, 2))
            # An asset-subset view slices the warm-start schedule's entries.
            mrv = PortfolioOptimisers.port_opt_view(MeanRisk(; opt = jopt, wi = tdwi), 1:3,
                                                    X)
            @test all(x -> length(x) == 3, mrv.wi.val)
        end
        @testset "Per-fold risk measure under walk-forward" begin
            tdr = TimeDependent([Variance(), ConditionalValueatRisk()])
            p = cross_val_predict(MeanRisk(; opt = jopt, r = tdr), rd, cvw)
            p0 = cross_val_predict(MeanRisk(; opt = jopt), rd, cvw)
            p1 = cross_val_predict(MeanRisk(; opt = jopt, r = ConditionalValueatRisk()), rd,
                                   cvw)
            @test isapprox(p.pred[1].res.w, p0.pred[1].res.w)
            @test isapprox(p.pred[2].res.w, p1.pred[2].res.w)
            @test !isapprox(p0.pred[2].res.w, p1.pred[2].res.w)
        end
        @testset "Per-fold objective under walk-forward" begin
            tdobj = TimeDependent([MinimumRisk(), MaximumUtility(; l = 4.0)])
            p = cross_val_predict(MeanRisk(; opt = jopt, obj = tdobj), rd, cvw)
            p0 = cross_val_predict(MeanRisk(; opt = jopt), rd, cvw)
            p1 = cross_val_predict(MeanRisk(; opt = jopt, obj = MaximumUtility(; l = 4.0)),
                                   rd, cvw)
            @test isapprox(p.pred[1].res.w, p0.pred[1].res.w)
            @test isapprox(p.pred[2].res.w, p1.pred[2].res.w)
            @test !isapprox(p0.pred[2].res.w, p1.pred[2].res.w)
        end
        @testset "Scheduled fallback: consumed per fold, inert without one" begin
            # An infeasible primary (lower bounds sum past the budget) forces the fallback
            # walk. Fold-less, a defaulted schedule resets to its default and that
            # optimiser runs; a defaultless one resets to nothing and the failure stands.
            bad = JuMPOptimiser(; slv = slv, wb = WeightBounds(; lb = 0.5, ub = 0.55))
            mrd = MeanRisk(; opt = bad,
                           fb = TimeDependent([EqualWeighted(), InverseVolatility()];
                                              default = EqualWeighted()))
            res = optimise(mrd, rd)
            @test isa(res.retcode, OptimisationSuccess)
            @test isapprox(res.w, optimise(EqualWeighted(), rd).w)
            mrn = MeanRisk(; opt = bad,
                           fb = TimeDependent([EqualWeighted(), InverseVolatility()]))
            resn = optimise(mrn, rd)
            @test !isa(resn.retcode, OptimisationSuccess)
            # Under the fold loop, fold 2's entry is the fallback that runs.
            p = cross_val_predict(MeanRisk(; opt = bad,
                                           fb = TimeDependent([EqualWeighted(),
                                                               InverseVolatility()])), rd,
                                  cvw)
            piv = cross_val_predict(InverseVolatility(), rd, cvw)
            @test isapprox(p.pred[2].res.w, piv.pred[2].res.w)
        end
        @testset "RiskBudgeting and RelaxedRiskBudgeting" begin
            rba1, rba2 = AssetRiskBudgeting(), AssetRiskBudgeting()
            rb = RiskBudgeting(; opt = jopt, r = TimeDependent([Variance(), Variance()]),
                               rba = TimeDependent([rba1, rba2]))
            @test PortfolioOptimisers.time_dependent_fields(rb) == (:r, :rba)
            rb0 = PortfolioOptimisers.reset_time_dependent_estimator(rb)
            @test rb0.r == Variance()
            @test rb0.rba == AssetRiskBudgeting()
            @test PortfolioOptimisers.update_time_dependent_estimator(rb, ctx2).rba === rba2
            @test_throws TypeError RiskBudgeting(; opt = jopt, rba = nothing)
            rrb = RelaxedRiskBudgeting(; opt = jopt, rba = TimeDependent([rba1, rba2]))
            @test PortfolioOptimisers.time_dependent_fields(rrb) == (:rba,)
            @test PortfolioOptimisers.reset_time_dependent_estimator(rrb).rba ==
                  AssetRiskBudgeting()
            @test PortfolioOptimisers.update_time_dependent_estimator(rrb, ctx2).rba ===
                  rba2
        end
        @testset "NearOptimalCentering and FactorRiskContribution" begin
            noc = NearOptimalCentering(; opt = jopt,
                                       r = TimeDependent([StandardDeviation(),
                                                          ConditionalValueatRisk()]),
                                       obj = TimeDependent([MinimumRisk(),
                                                            MaximumUtility(; l = 4.0)]),
                                       w_min = TimeDependent([fill(0.0, 5), fill(0.01, 5)]))
            @test PortfolioOptimisers.time_dependent_fields(noc) == (:r, :obj, :w_min)
            noc0 = PortfolioOptimisers.reset_time_dependent_estimator(noc)
            @test noc0.r == StandardDeviation()
            @test noc0.obj == MinimumRisk()
            @test isnothing(noc0.w_min)
            noc2 = PortfolioOptimisers.update_time_dependent_estimator(noc, ctx2)
            @test isa(noc2.r, ConditionalValueatRisk)
            @test noc2.w_min == fill(0.01, 5)
            frc = FactorRiskContribution(; opt = jopt,
                                         re = TimeDependent([StepwiseRegression(),
                                                             StepwiseRegression()]),
                                         r = TimeDependent([Variance(), Variance()]))
            @test PortfolioOptimisers.time_dependent_fields(frc) == (:re, :r)
            frc0 = PortfolioOptimisers.reset_time_dependent_estimator(frc)
            @test frc0.re == StepwiseRegression()
            @test frc0.r == Variance()
            @test isa(PortfolioOptimisers.update_time_dependent_estimator(frc, ctx2).re,
                      StepwiseRegression)
        end
        @testset "Hierarchical estimators: HRP, HERC and Schur" begin
            hopt = HierarchicalOptimiser(; slv = slv)
            tdr = TimeDependent([Variance(), ConditionalValueatRisk()])
            hrp = HierarchicalRiskParity(; opt = hopt, r = tdr,
                                         sca = TimeDependent([SumScalariser(),
                                                              MaxScalariser()]))
            @test PortfolioOptimisers.time_dependent_fields(hrp) == (:r, :sca)
            hrp0 = PortfolioOptimisers.reset_time_dependent_estimator(hrp)
            @test hrp0.r == Variance()
            @test hrp0.sca == SumScalariser()
            hrp2 = PortfolioOptimisers.update_time_dependent_estimator(hrp, ctx2)
            @test isa(hrp2.r, ConditionalValueatRisk)
            @test isa(hrp2.sca, MaxScalariser)
            # End to end: fold 2 allocates with the scheduled risk measure.
            p = cross_val_predict(HierarchicalRiskParity(; opt = hopt, r = tdr), rd, cvw)
            p0 = cross_val_predict(HierarchicalRiskParity(; opt = hopt), rd, cvw)
            p1 = cross_val_predict(HierarchicalRiskParity(; opt = hopt,
                                                          r = ConditionalValueatRisk()), rd,
                                   cvw)
            @test isapprox(p.pred[1].res.w, p0.pred[1].res.w)
            @test isapprox(p.pred[2].res.w, p1.pred[2].res.w)
            @test !isapprox(p0.pred[2].res.w, p1.pred[2].res.w)
            herc = HierarchicalEqualRiskContribution(; opt = hopt, ri = tdr,
                                                     ro = TimeDependent([Variance(),
                                                                         StandardDeviation()]),
                                                     scai = TimeDependent([SumScalariser(),
                                                                           MaxScalariser()]))
            # `scao` defaults to `scai`, so an explicit `scai` schedule aliases into
            # `scao` — the same flow-through the static default has always had.
            @test PortfolioOptimisers.time_dependent_fields(herc) ==
                  (:ri, :ro, :scai, :scao)
            @test herc.scao === herc.scai
            herc0 = PortfolioOptimisers.reset_time_dependent_estimator(herc)
            @test herc0.ri == Variance()
            @test herc0.ro == Variance()
            @test herc0.scai == SumScalariser()
            @test herc0.scao == SumScalariser()
            herc2 = PortfolioOptimisers.update_time_dependent_estimator(herc, ctx2)
            @test isa(herc2.ri, ConditionalValueatRisk)
            @test isa(herc2.ro, StandardDeviation)
            @test isa(herc2.scao, MaxScalariser)
            # An explicit static `scao` breaks the alias: no schedule reaches it, so it
            # stays put across folds while `scai` varies.
            herc_static = HierarchicalEqualRiskContribution(; opt = hopt,
                                                            scai = TimeDependent([SumScalariser(),
                                                                                  MaxScalariser()]),
                                                            scao = SumScalariser())
            @test PortfolioOptimisers.time_dependent_fields(herc_static) == (:scai,)
            hst0 = PortfolioOptimisers.reset_time_dependent_estimator(herc_static)
            @test hst0.scao == SumScalariser()
            hst2 = PortfolioOptimisers.update_time_dependent_estimator(herc_static, ctx2)
            @test isa(hst2.scai, MaxScalariser)
            @test hst2.scao == SumScalariser()
            sch = SchurComplementHierarchicalRiskParity(; opt = hopt,
                                                        params = TimeDependent([SchurComplementParams(;
                                                                                                      gamma = 0.2),
                                                                                SchurComplementParams(;
                                                                                                      gamma = 0.8)]))
            @test PortfolioOptimisers.time_dependent_fields(sch) == (:params,)
            @test PortfolioOptimisers.reset_time_dependent_estimator(sch).params ==
                  SchurComplementParams()
            @test PortfolioOptimisers.update_time_dependent_estimator(sch, ctx2).params.gamma ==
                  0.8
        end
        @testset "Naive optimisers: weight finaliser, sets and prior" begin
            tdwf = TimeDependent([IterativeWeightFinaliser(),
                                  IterativeWeightFinaliser(; iter = 7)])
            ew = EqualWeighted(; wf = tdwf)
            @test PortfolioOptimisers.time_dependent_fields(ew) == (:wf,)
            @test PortfolioOptimisers.reset_time_dependent_estimator(ew).wf ==
                  IterativeWeightFinaliser()
            @test PortfolioOptimisers.update_time_dependent_estimator(ew, ctx2).wf.iter == 7
            @test_throws TypeError EqualWeighted(; wf = nothing)
            # A fold-less solve runs with the finaliser at its static default.
            @test isapprox(optimise(ew, rd).w, optimise(EqualWeighted(), rd).w)
            iv = InverseVolatility(;
                                   pe = TimeDependent([EmpiricalPrior(),
                                                       EmpiricalPrior(;
                                                                      ce = PortfolioOptimisersCovariance(;
                                                                                                         ce = Covariance(;
                                                                                                                         alg = SemiMoment())))]))
            @test PortfolioOptimisers.time_dependent_fields(iv) == (:pe,)
            @test PortfolioOptimisers.reset_time_dependent_estimator(iv).pe ==
                  EmpiricalPrior()
            @test isa(PortfolioOptimisers.update_time_dependent_estimator(iv, ctx2).pe.ce.ce.alg,
                      SemiMoment)
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
            l2 = TimeDependent([L2Regularisation(; val = 1e-4),
                                L2Regularisation(; val = 2e-4)])
            mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv, l2 = l2))
            p = cross_val_predict(TimeDependent([ew, mr]), rd, cvw)
            mr2 = MeanRisk(;
                           opt = JuMPOptimiser(; slv = slv,
                                               l2 = L2Regularisation(; val = 2e-4)))
            @test isapprox(p.pred[2].res.w, cross_val_predict(mr2, rd, cvw).pred[2].res.w)
            # A schedule inside an entry is still sized to this loop, so a mis-sized one
            # fails even though the top-level schedule is correctly sized.
            mr_bad = MeanRisk(;
                              opt = JuMPOptimiser(; slv = slv,
                                                  l2 = TimeDependent([L2Regularisation(;
                                                                                       val = 1e-4),
                                                                      L2Regularisation(;
                                                                                       val = 2e-4),
                                                                      L2Regularisation(;
                                                                                       val = 3e-4)])))
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
                                              l2 = TimeDependent([L2Regularisation(;
                                                                                   val = 1e-4),
                                                                  L2Regularisation(;
                                                                                   val = 2e-4)];
                                                                 default = L2Regularisation(;
                                                                                            val = 5e-4))))
            mr0 = MeanRisk(;
                           opt = JuMPOptimiser(; slv = slv,
                                               l2 = L2Regularisation(; val = 5e-4)))
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
    @testset "Traits and views over schedules of optimisers" begin
        ew, iv = EqualWeighted(), InverseVolatility()
        w0 = fill(1 / 5, 5)
        res = optimise(iv, rd)
        mrtn = MeanRisk(;
                        opt = JuMPOptimiser(; slv = slv,
                                            tn = Turnover(; val = 0.1, w = w0)))
        mrv = MeanRisk(;
                       opt = JuMPOptimiser(; slv = slv,
                                           wb = WeightBounds(; lb = zeros(5),
                                                             ub = fill(0.8, 5))))
        cvw = IndexWalkForward(100, 50)  # 2 folds over the 200-row sample
        @testset "needs_previous_weights recurses into entries" begin
            # An estimator entry with an unfixed turnover needs the previous weights.
            @test PortfolioOptimisers.needs_previous_weights(TimeDependent([mrtn, ew]))
            # A precomputed result never does; nor do plain estimators.
            @test !PortfolioOptimisers.needs_previous_weights(TimeDependent([res, res]))
            @test !PortfolioOptimisers.needs_previous_weights(TimeDependent([ew, iv]))
            # A mixed schedule takes the disjunction over its entries.
            @test PortfolioOptimisers.needs_previous_weights(TimeDependent([mrtn, res]))
            # The trait is scope-blind: it never consults bind.
            @test PortfolioOptimisers.needs_previous_weights(TimeDependent([mrtn, ew],
                                                                           :nearest))
        end
        @testset "Fold-count assertion over optimiser schedules" begin
            # A mixed schedule is length-checked exactly like a schedule of values.
            sched = TimeDependent([mrtn, res])
            @test isnothing(PortfolioOptimisers.assert_time_dependent_fold_count(sched, 2))
            @test_throws DimensionMismatch PortfolioOptimisers.assert_time_dependent_fold_count(sched,
                                                                                                3)
            # The schedules inside each entry are sized to the same loop...
            mr_bad = MeanRisk(;
                              opt = JuMPOptimiser(; slv = slv,
                                                  l2 = TimeDependent([L2Regularisation(;
                                                                                       val = 1e-4),
                                                                      L2Regularisation(;
                                                                                       val = 2e-4),
                                                                      L2Regularisation(;
                                                                                       val = 3e-4)])))
            @test_throws DimensionMismatch PortfolioOptimisers.assert_time_dependent_fold_count(TimeDependent([ew,
                                                                                                               mr_bad]),
                                                                                                2)
            # ...but the default is not: it only ever runs outside a fold loop.
            @test isnothing(PortfolioOptimisers.assert_time_dependent_fold_count(TimeDependent([ew,
                                                                                                iv];
                                                                                               default = mr_bad),
                                                                                 2))
            # A :nearest schedule is skipped by a loop scanning with all_binds = false —
            # the fold loop that owns it validates it against its own fold count.
            near = TimeDependent([ew, iv, ew], :nearest; default = ew)
            @test isnothing(PortfolioOptimisers.assert_time_dependent_fold_count(near, 2,
                                                                                 false))
            @test_throws DimensionMismatch PortfolioOptimisers.assert_time_dependent_fold_count(near,
                                                                                                2,
                                                                                                true)
        end
        @testset "is_time_dependent sees the schedule; the swap resolves it fully" begin
            @test PortfolioOptimisers.is_time_dependent(TimeDependent([ew, iv]))
            @test PortfolioOptimisers.is_time_dependent(TimeDependent([res, res]))
            @test PortfolioOptimisers.is_time_dependent(TimeDependent([mrtn, res]))
            # The swap recurses into the resolved entry with the same context, so nothing
            # time-dependent survives it — even when the entry carries its own schedules.
            mrl = MeanRisk(;
                           opt = JuMPOptimiser(; slv = slv,
                                               l2 = TimeDependent([L2Regularisation(;
                                                                                    val = 1e-4),
                                                                   L2Regularisation(;
                                                                                    val = 2e-4)])))
            sched = TimeDependent([ew, mrl])
            ctx1 = TimeDependentContext(; i = 1, n = 2, rd = rd, train_idx = [1:100, 1:150],
                                        test_idx = [101:150, 151:200])
            ctx2 = TimeDependentContext(; i = 2, n = 2, rd = rd, train_idx = [1:100, 1:150],
                                        test_idx = [101:150, 151:200])
            o1 = PortfolioOptimisers.update_time_dependent_estimator(sched, ctx1)
            o2 = PortfolioOptimisers.update_time_dependent_estimator(sched, ctx2)
            @test o1 isa EqualWeighted
            @test o2.opt.l2.val == 2e-4
            @test !PortfolioOptimisers.is_time_dependent(o1)
            @test !PortfolioOptimisers.is_time_dependent(o2)
        end
        @testset "port_opt_view slices estimator entries and the default" begin
            tdo = TimeDependent([mrv, ew]; default = mrv)
            tdo2 = PortfolioOptimisers.port_opt_view(tdo, 1:3, rd.X)
            @test tdo2.val[1].opt.wb.ub == fill(0.8, 3)
            @test tdo2.val[1].opt.wb.lb == zeros(3)
            @test tdo2.default.opt.wb.ub == fill(0.8, 3)
            # Under multiple-randomised CV each fold's optimiser is viewed to the fold's
            # random asset subset, so the per-fold weights live on the subset.
            cvm = MultipleRandomised(IndexWalkForward(100, 50); subset_size = 3,
                                     rng = StableRNG(42), seed = 7)
            p = cross_val_predict(TimeDependent([mrv, ew]), rd, cvm)
            @test all(path -> all(x -> length(x.res.w) == 3, path.pred), p.pred)
            # A precomputed result has no asset-subset view: its full-universe weights
            # have no sub-portfolio meaning, so a schedule holding one is rejected under
            # asset subsampling...
            @test_throws ArgumentError PortfolioOptimisers.port_opt_view(TimeDependent([mrv,
                                                                                        res]),
                                                                         1:3, rd.X)
            @test_throws ArgumentError cross_val_predict(TimeDependent([mrv, res]), rd, cvm)
            # ...but passes through the trivial all-assets view unchanged.
            @test PortfolioOptimisers.port_opt_view(res, :, rd.X) === res
        end
        @testset "Swap-then-factory: previous weights reach the swapped-in optimiser" begin
            p_direct = cross_val_predict(mrtn, rd, cvw)
            p_sched = cross_val_predict(TimeDependent([mrtn, mrtn]), rd, cvw)
            # The factory pass runs after the swap, so fold 2's turnover anchors to fold
            # 1's weights exactly as in the unscheduled run...
            @test isapprox(p_sched.pred[2].res.w, p_direct.pred[2].res.w)
            # ...and the anchor is material: fold 1's weights are not the constructor's w0.
            @test !isapprox(p_direct.pred[1].res.w, w0)
        end
    end
end
