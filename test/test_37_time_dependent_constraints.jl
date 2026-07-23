# TimeDependentCallable subtypes must be defined at top level. `_test_TDCap` is a plain functor;
# `TDPrevWCap` declares its previous-weights requirement directly.
struct _test_TDCap <: PortfolioOptimisers.TimeDependentCallable
    hi::Float64
    lo::Float64
end
function (c::_test_TDCap)(ctx::TimeDependentContext)
    return WeightBounds(; lb = 0.0,
                        ub = c.hi - (c.hi - c.lo) * (ctx.i - 1) / max(ctx.n - 1, 1))
end
struct _test_TDPrevWCap <: PortfolioOptimisers.TimeDependentCallable end
function (c::TDPrevWCap)(ctx::TimeDependentContext)
    return Threshold(; val = isnothing(ctx.w_prev) ? 0.01 : 0.05)
end
function PortfolioOptimisers.needs_previous_weights(::TDPrevWCap)
    return true
end
# `TDOptCap` declares that its per-fold value is an optimiser, which is what makes a schedule
# holding it statically admissible in an optimiser-valued field.
struct _test_TDOptCap <: PortfolioOptimisers.TimeDependentOptimiserCallable end
function (c::TDOptCap)(ctx::TimeDependentContext)
    return isodd(ctx.i) ? EqualWeighted() : InverseVolatility()
end
# `TDLog` records the optimiser it picks per fold in a mutable field, keyed by the coordinates
# recovery needs — `(path_id, i)`, so it works under multi-path schemes too. A callable
# schedule selects nothing by index, so its per-fold decision is recoverable only if it logs
# it; the `TimeDependentCallable` struct interface is where that logging lives (#148).
struct _test_TDLog{T, U} <: PortfolioOptimisers.TimeDependentOptimiserCallable
    odd::T
    even::U
    log::Vector{Tuple{Union{Int, Nothing}, Int, Symbol}}
end
function (c::TDLog)(ctx::TimeDependentContext)
    pick = isodd(ctx.i) ? :odd : :even
    push!(c.log, (ctx.path_id, ctx.i, pick))
    return pick === :odd ? c.odd : c.even
end
@testset "Time-dependent constraints" begin
    using Test, PortfolioOptimisers, Clarabel, StableRNGs, Dates, TimeSeries
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
        @test TimeDependent(_test_TDCap(0.35, 0.2)) isa TimeDependent
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
    @testset "Candidate fields narrow the scan without changing it" begin
        # The scan only visits the fields whose type admits a schedule, so it must agree with
        # a scan of every field of the host, on static and scheduled hosts alike.
        function full_scan(opt, all_binds::Bool = true)
            return filter(fieldnames(typeof(opt))) do f
                x = getfield(opt, f)
                return isa(x, TimeDependent) &&
                       (PortfolioOptimisers.entitled(opt, f, all_binds) ||
                        x.bind === :outermost)
            end
        end
        hosts = [JuMPOptimiser(; slv = slv),
                 JuMPOptimiser(; slv = slv,
                               wb = TimeDependent([WeightBounds(; ub = 0.3),
                                                   WeightBounds(; ub = 0.5)])),
                 JuMPOptimiser(; slv = slv, bgt = TimeDependent([1.0, 0.9]),
                               fees = TimeDependent([Fees(; l = 0.001), Fees(; l = 0.002)],
                                                    :nearest; default = Fees())),
                 MeanRisk(; opt = JuMPOptimiser(; slv = slv)),
                 MeanRisk(; r = TimeDependent([Variance(), StandardDeviation()]),
                          opt = JuMPOptimiser(; slv = slv)),
                 NestedClustered(;
                                 opti = TimeDependent([EqualWeighted(),
                                                       InverseVolatility()], :nearest;
                                                      default = EqualWeighted()),
                                 opto = TimeDependent([EqualWeighted(),
                                                       InverseVolatility()]),
                                 cv = OptimisationCrossValidation(; cv = KFold(; n = 2)))]
        for host in hosts, all_binds in (true, false)
            @test PortfolioOptimisers.time_dependent_fields(host, all_binds) ==
                  full_scan(host, all_binds)
        end
        # A static host has no schedule-admitting field at all, so it scans to nothing.
        static = JuMPOptimiser(; slv = slv)
        @test PortfolioOptimisers.time_dependent_candidate_fields(static) == ()
        @test isempty(PortfolioOptimisers.time_dependent_fields(static))
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
        _test_TDO = PortfolioOptimisers.TDO_Option{<:PortfolioOptimisers.OptE_Opt}
        @test nothing isa _test_TDO
        @test ew isa _test_TDO
        @test TimeDependent([ew, iv]) isa _test_TDO
        @test !(TimeDependent([Fees(; l = 0.001)]) isa _test_TDO)
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
        optcap = JuMPOptimiser(; slv = slv, wb = TimeDependent(_test_TDCap(0.35, 0.2)))
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
        struct _test_TDLC_Callable <: PortfolioOptimisers.TimeDependentCallable end
        function (c::TDLC_Callable)(ctx::TimeDependentContext)
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
                                                 lcse = TimeDependent(TDLC_Callable())))
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
            # Required fields stay non-nothing (typed _test_TD, not _test_TD_Option).
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
    @testset "Schedules in the meta-optimisers' optimiser-valued fields" begin
        ew, iv = EqualWeighted(), InverseVolatility()
        mr0 = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        mrl = MeanRisk(;
                       opt = JuMPOptimiser(; slv = slv,
                                           l2 = L2Regularisation(; val = 1e-3)))
        res = optimise(iv, rd)
        cvw = IndexWalkForward(100, 50)  # 2 outer folds over the 200-row sample
        cv3 = OptimisationCrossValidation(; cv = KFold(; n = 3))
        sex = PortfolioOptimisers.FLoops.SequentialEx()
        @testset "Construction: bind legality per position" begin
            # :nearest is legal on NestedClustered.opti — its inner CV is entered per
            # cluster, so the field is the inner fold loop's entry point — but the
            # position's double consumer forces an explicit default (the per-cluster
            # full-sample leg always resolves the schedule fold-lessly)...
            @test_throws PortfolioOptimisers.TimeDependentDefaultError NestedClustered(;
                                                                                       opti = TimeDependent([mr0,
                                                                                                             mrl,
                                                                                                             mr0],
                                                                                                            :nearest),
                                                                                       opto = mr0,
                                                                                       cv = cv3)
            # ...and an inner CV to bind to; without one the schedule would be inert.
            @test_throws ArgumentError NestedClustered(;
                                                       opti = TimeDependent([mr0, mrl, mr0],
                                                                            :nearest;
                                                                            default = mr0),
                                                       opto = mr0)
            @test isa(NestedClustered(;
                                      opti = TimeDependent([mr0, mrl, mr0], :nearest;
                                                           default = mr0), opto = mr0,
                                      cv = cv3), NestedClustered)
            # Stacking's inner CV is entered per candidate, so :nearest is legal on an
            # element (same two checks)...
            @test_throws PortfolioOptimisers.TimeDependentDefaultError Stacking(;
                                                                                opti = [TimeDependent([ew,
                                                                                                       iv,
                                                                                                       ew],
                                                                                                      :nearest),
                                                                                        mr0],
                                                                                opto = mr0,
                                                                                cv = cv3)
            @test_throws ArgumentError Stacking(;
                                                opti = [TimeDependent([ew, iv, ew],
                                                                      :nearest;
                                                                      default = ew), mr0],
                                                opto = mr0)
            @test isa(Stacking(;
                               opti = [TimeDependent([ew, iv, ew], :nearest; default = ew),
                                       mr0], opto = mr0, cv = cv3), Stacking)
            # ...and rejected on the field: the fold loop is handed the elements, and a
            # per-fold candidate vector would break the identity of opto's columns.
            @test_throws ArgumentError Stacking(;
                                                opti = TimeDependent([[ew, mr0], [iv, mr0]],
                                                                     :nearest;
                                                                     default = [ew, mr0]),
                                                opto = mr0, cv = cv3)
            # No inner fold loop owns the outer optimisers, SubsetResampling.opt or any
            # fallback, so :nearest is rejected there outright.
            @test_throws ArgumentError Stacking(; opti = [mr0],
                                                opto = TimeDependent([ew, iv], :nearest;
                                                                     default = ew),
                                                cv = cv3)
            @test_throws ArgumentError NestedClustered(; opti = mr0,
                                                       opto = TimeDependent([ew, iv],
                                                                            :nearest;
                                                                            default = ew),
                                                       cv = cv3)
            @test_throws ArgumentError SubsetResampling(;
                                                        opt = TimeDependent([ew, iv],
                                                                            :nearest;
                                                                            default = ew),
                                                        subset_size = 3, n_subsets = 2)
            @test_throws ArgumentError Stacking(; opti = [mr0], opto = mr0,
                                                fb = TimeDependent([ew, iv], :nearest))
            @test_throws ArgumentError NestedClustered(; opti = mr0, opto = mr0,
                                                       fb = TimeDependent([ew, iv],
                                                                          :nearest))
            @test_throws ArgumentError SubsetResampling(; opt = mr0,
                                                        fb = TimeDependent([ew, iv],
                                                                           :nearest),
                                                        subset_size = 3, n_subsets = 2)
            # NCO's opti/opto schedule entries must be estimators, like the static
            # fields: the constructor's entry substitution rejects a precomputed result.
            @test_throws TypeError NestedClustered(; opti = TimeDependent([mr0, res]),
                                                   opto = mr0)
            @test_throws TypeError NestedClustered(; opti = mr0,
                                                   opto = TimeDependent([mr0, res]))
            # cv stays static: it IS the inner fold loop, not per-fold problem definition.
            @test_throws TypeError NestedClustered(; opti = mr0, opto = mr0,
                                                   cv = TimeDependent([cv3, cv3]))
            # A literal mixing static optimisers and schedules narrows through the
            # keyword constructor even though its inferred eltype is too wide.
            @test isa(Stacking(; opti = [mr0, TimeDependent([ew, iv]; default = ew)],
                               opto = mr0).opti, PortfolioOptimisers.VecOptE_Opt_TD)
            @test_throws ArgumentError Stacking(; opti = [mr0, Variance()], opto = mr0)
            # The hosts declare which fields their own inner fold loop consumes.
            @test PortfolioOptimisers.inner_fold_fields(Stacking(; opti = [mr0],
                                                                 opto = mr0)) == (:opti,)
            @test PortfolioOptimisers.inner_fold_fields(NestedClustered(; opti = mr0,
                                                                        opto = mr0)) ==
                  (:opti,)
            @test PortfolioOptimisers.inner_fold_fields(SubsetResampling(; opt = mr0)) == ()
        end
        @testset "Per-field entitlement: the scan leaves :nearest opti for the inner CV" begin
            ncoX = NestedClustered(;
                                   opti = TimeDependent([mr0, mrl, mr0], :nearest;
                                                        default = mr0),
                                   opto = TimeDependent([ew, iv]; default = ew), cv = cv3)
            # Even at all_binds = true the pass skips the :nearest opti — NCO's own inner
            # CV, not the scanning loop, is nearest for that field — while the
            # :outermost opto is taken either way.
            @test PortfolioOptimisers.time_dependent_fields(ncoX, true) == (:opto,)
            @test PortfolioOptimisers.time_dependent_fields(ncoX, false) == (:opto,)
            # An :outermost opti is not the inner loop's to consume, so the scan takes it.
            ncoO = NestedClustered(; opti = TimeDependent([mr0, mrl]; default = mr0),
                                   opto = mr0)
            @test PortfolioOptimisers.time_dependent_fields(ncoO, true) == (:opti,)
        end
        @testset "NestedClustered: reset trap and the double consumer" begin
            # An observable :nearest schedule: the callable may only run in the inner CV
            # leg, never in the per-cluster full-sample leg (which takes the default).
            seen = zeros(Int, 3)
            obs = TimeDependent(ctx -> begin
                                    seen[ctx.i] += 1
                                    mrl
                                end, :nearest; default = mr0)
            nco = NestedClustered(; opti = obs, opto = mr0, cv = cv3, ex = sex)
            # The fold-less reset at the top of _optimise must leave the :nearest
            # schedule in place — resetting it would swap in the default before the
            # inner CV ever saw it.
            nres = PortfolioOptimisers.reset_time_dependent_estimator(nco)
            @test isa(nres.opti, TimeDependent)
            # One standalone solve exercises both consumers: the callable fires once per
            # inner fold per cluster; the per-cluster optimise leg runs the default.
            nr = optimise(nco, rd)
            @test isa(nr.retcode, OptimisationSuccess)
            @test seen == fill(nr.clr.k, 3)
            # Under an outer walk-forward the inner-sized :nearest schedule is skipped by
            # the outer split's count check and consumed by the inner KFold(3).
            near3 = TimeDependent([mr0, mrl, mr0], :nearest; default = ew)
            nco3 = NestedClustered(; opti = near3, opto = mr0, cv = cv3)
            @test length(cross_val_predict(nco3, rd, cvw).pred) == 2
            @test isa(optimise(nco3, rd).retcode, OptimisationSuccess)
            # :outermost resolves per outer fold before recursion; an inner-sized
            # :outermost schedule therefore fails at the outer split.
            nco2 = NestedClustered(; opti = TimeDependent([mr0, mrl]), opto = mr0)
            @test length(cross_val_predict(nco2, rd, cvw).pred) == 2
            nco_bad = NestedClustered(; opti = TimeDependent([mr0, mrl, mr0]), opto = mr0)
            @test_throws DimensionMismatch cross_val_predict(nco_bad, rd, cvw)
            # Fold-less, an :outermost schedule resolves to its default; without one
            # there is no optimiser to run.
            @test isa(optimise(NestedClustered(;
                                               opti = TimeDependent([mr0, mrl];
                                                                    default = mr0),
                                               opto = mr0), rd).retcode,
                      OptimisationSuccess)
            @test_throws PortfolioOptimisers.TimeDependentDefaultError optimise(nco_bad, rd)
        end
        @testset "Stacking: element schedules and the field-level vector of vectors" begin
            # An element schedule is the inner CV's per-candidate entry point; the
            # full-sample wi fit resolves it to its default within the same solve.
            seen = zeros(Int, 3)
            el = TimeDependent(ctx -> begin
                                   seen[ctx.i] += 1
                                   mrl
                               end, :nearest; default = mr0)
            st = Stacking(; opti = [el, mr0], opto = mr0, cv = cv3, ex = sex)
            sres = optimise(st, rd)
            @test isa(sres.retcode, OptimisationSuccess)
            @test seen == [1, 1, 1]
            @test length(sres.resi) == 2
            # Under an outer loop the recursion leaves the :nearest element alone; an
            # outer-sized :outermost element is resolved by the outer loop instead.
            @test length(cross_val_predict(st, rd, cvw).pred) == 2
            st2 = Stacking(; opti = [TimeDependent([ew, iv]), mr0], opto = mr0)
            @test length(cross_val_predict(st2, rd, cvw).pred) == 2
            # Fold-less, a defaultless :outermost element hits the wi fit and throws.
            @test_throws PortfolioOptimisers.TimeDependentDefaultError optimise(Stacking(;
                                                                                         opti = [TimeDependent([ew,
                                                                                                                iv]),
                                                                                                 mr0],
                                                                                         opto = mr0,
                                                                                         ex = sex),
                                                                                rd)
            # A mixed element schedule optimises or predicts per outer fold.
            stm = Stacking(; opti = [TimeDependent([mr0, res]), mr0], opto = mr0)
            @test length(cross_val_predict(stm, rd, cvw).pred) == 2
            # Field-level: entry i is fold i's complete candidate vector; the default is
            # the fold-less vector, and a defaultless schedule cannot solve fold-lessly.
            stf = Stacking(;
                           opti = TimeDependent([[mr0, ew], [mrl, iv]];
                                                default = [mr0, ew]), opto = mr0)
            @test length(cross_val_predict(stf, rd, cvw).pred) == 2
            @test isa(optimise(stf, rd).retcode, OptimisationSuccess)
            @test_throws PortfolioOptimisers.TimeDependentDefaultError optimise(Stacking(;
                                                                                         opti = TimeDependent([[mr0,
                                                                                                                ew],
                                                                                                               [mrl,
                                                                                                                iv]]),
                                                                                         opto = mr0),
                                                                                rd)
            @test_throws DimensionMismatch cross_val_predict(Stacking(;
                                                                      opti = TimeDependent([[mr0,
                                                                                             ew]]),
                                                                      opto = mr0), rd, cvw)
        end
        @testset "Stacking: opto and scale schedules" begin
            # opto resolves per outer fold; fold-less it resets to its default.
            sto = Stacking(; opti = [mr0], opto = TimeDependent([ew, iv]; default = ew))
            @test length(cross_val_predict(sto, rd, cvw).pred) == 2
            @test isa(optimise(sto, rd).retcode, OptimisationSuccess)
            @test_throws PortfolioOptimisers.TimeDependentDefaultError optimise(Stacking(;
                                                                                         opti = [mr0],
                                                                                         opto = TimeDependent([ew,
                                                                                                               iv])),
                                                                                rd)
            # scale is ordinary problem definition: per-fold under CV, its static
            # default (nothing) fold-lessly.
            sts = Stacking(; opti = [mr0, mrl], opto = mr0,
                           scale = TimeDependent([[1.0, 1.0], [2.0, 1.0]]))
            @test length(cross_val_predict(sts, rd, cvw).pred) == 2
            @test optimise(sts, rd).w ==
                  optimise(Stacking(; opti = [mr0, mrl], opto = mr0), rd).w
            # Entry substitution still runs the cross-field checks per fold.
            @test_throws DimensionMismatch Stacking(; opti = [mr0, mrl], opto = mr0,
                                                    scale = TimeDependent([[1.0],
                                                                           [2.0, 1.0]]))
        end
        @testset "SubsetResampling: opt, sizes and the nothing-entry fallback" begin
            sr = SubsetResampling(; opt = TimeDependent([mr0, mrl]; default = mr0),
                                  subset_size = 3, n_subsets = 2, rng = StableRNG(7),
                                  seed = 11)
            @test length(cross_val_predict(sr, rd, cvw).pred) == 2
            sr0 = SubsetResampling(; opt = mr0, subset_size = 3, n_subsets = 2,
                                   rng = StableRNG(7), seed = 11)
            @test optimise(sr, rd).w == optimise(sr0, rd).w
            @test_throws PortfolioOptimisers.TimeDependentDefaultError optimise(SubsetResampling(;
                                                                                                 opt = TimeDependent([mr0,
                                                                                                                      mrl]),
                                                                                                 subset_size = 3,
                                                                                                 n_subsets = 2),
                                                                                rd)
            # subset_size and n_subsets are per-fold problem definition with static
            # defaults to reset to.
            srs = SubsetResampling(; opt = mr0, subset_size = TimeDependent([3, 4]),
                                   n_subsets = TimeDependent([2, 3]), rng = StableRNG(7),
                                   seed = 11)
            @test length(cross_val_predict(srs, rd, cvw).pred) == 2
            srr = PortfolioOptimisers.reset_time_dependent_estimator(srs)
            @test srr.subset_size == 0.8
            @test srr.n_subsets == 2
            @test_throws TypeError SubsetResampling(; opt = mr0,
                                                    subset_size = TimeDependent([3, "a"]))
            # An optional optimiser field admits nothing entries: the fallback is
            # switched off on fold 2 and the failure stands there.
            bad = JuMPOptimiser(; slv = slv, wb = WeightBounds(; lb = 0.5, ub = 0.55))
            mrbad = MeanRisk(; opt = bad)
            fbsched = TimeDependent([iv, nothing])
            @test isnothing(PortfolioOptimisers.assert_time_dependent_fold_count(fbsched, 2,
                                                                                 true))
            @test_throws DimensionMismatch PortfolioOptimisers.assert_time_dependent_fold_count(fbsched,
                                                                                                3,
                                                                                                true)
            srf = SubsetResampling(; opt = mrbad, fb = fbsched, subset_size = 3,
                                   n_subsets = 2, rng = StableRNG(7), seed = 11)
            @test isnothing(PortfolioOptimisers.reset_time_dependent_estimator(srf).fb)
            pf = cross_val_predict(srf, rd, cvw)
            @test isa(pf.pred[1].res.retcode, OptimisationSuccess)
            @test !isa(pf.pred[2].res.retcode, OptimisationSuccess)
            # Regression: the fold-less fast path dispatches on fb, not seed — a static
            # fallback must be walked even when seed is nothing.
            sr_fb = SubsetResampling(; opt = mrbad, fb = iv, subset_size = 3, n_subsets = 2,
                                     rng = StableRNG(7))
            @test isa(optimise(sr_fb, rd).retcode, OptimisationSuccess)
        end
        @testset "factory and previous weights through surviving schedules" begin
            w0 = fill(1 / 5, 5)
            mrtn = MeanRisk(;
                            opt = JuMPOptimiser(; slv = slv,
                                                tn = Turnover(; val = 0.1, w = w0)))
            # The factory pass sees through a schedule, rebuilding entries and default.
            f = PortfolioOptimisers.factory(TimeDependent([mrtn, ew], :nearest;
                                                          default = mrtn), fill(0.2, 5))
            @test isa(f, TimeDependent)
            @test f.bind === :nearest
            # A schedule inside a field-level entry vector counts toward the trait...
            @test PortfolioOptimisers.time_dependent_entry_needs_previous_weights(TimeDependent([mrtn,
                                                                                                 ew]))
            @test PortfolioOptimisers.needs_previous_weights(TimeDependent([[mrtn, ew],
                                                                            [ew, ew]]))
            # ...and so does an element schedule of a Stacking, driving sequential outer
            # folds; the surviving :nearest element rides the post-resolution factory
            # pass without being consumed.
            stn = Stacking(;
                           opti = [TimeDependent([mrtn, mrtn, mrtn], :nearest;
                                                 default = mrtn), mr0], opto = mr0,
                           cv = cv3)
            @test PortfolioOptimisers.needs_previous_weights(stn)
            @test length(cross_val_predict(stn, rd, cvw).pred) == 2
        end
        @testset "Asset-subset views through meta fields" begin
            mrv = MeanRisk(;
                           opt = JuMPOptimiser(; slv = slv,
                                               wb = WeightBounds(; lb = zeros(5),
                                                                 ub = fill(0.8, 5))))
            stv = Stacking(; opti = [TimeDependent([mrv, ew]; default = mrv), mr0],
                           opto = mr0)
            sv = PortfolioOptimisers.port_opt_view(stv, 1:3, rd.X)
            @test sv.opti[1].val[1].opt.wb.ub == fill(0.8, 3)
            @test sv.opti[1].default.opt.wb.ub == fill(0.8, 3)
            ncov = NestedClustered(;
                                   opti = TimeDependent([mrv, mrv, ew], :nearest;
                                                        default = mrv), opto = mr0,
                                   cv = cv3)
            nv = PortfolioOptimisers.port_opt_view(ncov, 1:3, rd.X)
            @test isa(nv.opti, TimeDependent)
            @test nv.opti.bind === :nearest
            @test nv.opti.val[1].opt.wb.ub == fill(0.8, 3)
        end
    end
    @testset "Problem-definition fields of the meta-optimisers" begin
        # The metas' own non-optimiser problem definition — prior, sets, weight finaliser,
        # and NestedClustered's clustering estimator — takes schedules like any other
        # host's; `cv`, `ex`, `brt`, `cle_pr` and `strict` stay static.
        ew, iv = EqualWeighted(), InverseVolatility()
        mr0 = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        cvw = IndexWalkForward(100, 50)
        pe_semi = EmpiricalPrior(;
                                 ce = PortfolioOptimisersCovariance(;
                                                                    ce = Covariance(;
                                                                                    alg = SemiMoment())))
        tdpe = TimeDependent([EmpiricalPrior(), pe_semi])
        setsA = AssetSets(; dict = Dict("nx" => rd.nx, "g1" => ["A", "B"]))
        setsB = AssetSets(; dict = Dict("nx" => rd.nx, "g1" => ["D", "E"]))
        tdsets = TimeDependent([setsA, setsB])
        tdwf = TimeDependent([IterativeWeightFinaliser(),
                              IterativeWeightFinaliser(; iter = 7)])
        tdcle = TimeDependent([ClustersEstimator(),
                               ClustersEstimator(;
                                                 alg = HClustAlgorithm(; linkage = :single))])
        ctx2 = TimeDependentContext(; i = 2, n = 2, rd = rd, train_idx = 1:150,
                                    test_idx = 151:200)
        @testset "Widened fields, defaults and per-fold resolution" begin
            nco = NestedClustered(; pe = tdpe, cle = tdcle, sets = tdsets, wf = tdwf,
                                  opti = mr0, opto = mr0)
            @test PortfolioOptimisers.time_dependent_fields(nco) == (:pe, :cle, :sets, :wf)
            n0 = PortfolioOptimisers.reset_time_dependent_estimator(nco)
            @test n0.pe == EmpiricalPrior()
            @test n0.cle == ClustersEstimator()
            @test isnothing(n0.sets)
            @test n0.wf == IterativeWeightFinaliser()
            n2 = PortfolioOptimisers.update_time_dependent_estimator(nco, ctx2)
            @test n2.pe == pe_semi
            @test n2.cle.alg.linkage == :single
            @test n2.sets === setsB
            @test n2.wf.iter == 7
            st = Stacking(; pe = tdpe, sets = tdsets, wf = tdwf, opti = [mr0], opto = mr0)
            @test PortfolioOptimisers.time_dependent_fields(st) == (:pe, :sets, :wf)
            s0 = PortfolioOptimisers.reset_time_dependent_estimator(st)
            @test s0.pe == EmpiricalPrior() && isnothing(s0.sets)
            @test s0.wf == IterativeWeightFinaliser()
            s2 = PortfolioOptimisers.update_time_dependent_estimator(st, ctx2)
            @test s2.pe == pe_semi && s2.sets === setsB && s2.wf.iter == 7
            sr = SubsetResampling(; pe = tdpe, sets = tdsets, wf = tdwf, opt = ew,
                                  subset_size = 3, n_subsets = 2)
            @test PortfolioOptimisers.time_dependent_fields(sr) == (:pe, :sets, :wf)
            sr0 = PortfolioOptimisers.reset_time_dependent_estimator(sr)
            @test sr0.pe == EmpiricalPrior() && isnothing(sr0.sets)
            @test sr0.wf == IterativeWeightFinaliser()
            sr2 = PortfolioOptimisers.update_time_dependent_estimator(sr, ctx2)
            @test sr2.pe == pe_semi && sr2.sets === setsB && sr2.wf.iter == 7
            # The always-carrying fields stay `TD`, not `TD_Option`.
            @test_throws TypeError NestedClustered(; opti = mr0, opto = mr0, pe = nothing)
            @test_throws TypeError Stacking(; opti = [mr0], opto = mr0, wf = nothing)
            @test_throws TypeError SubsetResampling(; opt = ew, pe = nothing)
            # Entry substitution rejects a wrongly-typed entry at construction.
            @test_throws TypeError NestedClustered(; opti = mr0, opto = mr0,
                                                   pe = TimeDependent([EmpiricalPrior(),
                                                                       1.0]))
        end
        @testset "Per-fold resolution under walk-forward" begin
            # A scheduled meta field resolves against the fold loop that reaches the
            # meta: the callable fires once per outer fold with that fold's context.
            seen = zeros(Int, 2)
            obspe = TimeDependent(ctx -> begin
                                      seen[ctx.i] += 1
                                      EmpiricalPrior()
                                  end)
            p = cross_val_predict(NestedClustered(; pe = obspe, opti = mr0, opto = mr0), rd,
                                  cvw)
            @test length(p.pred) == 2 && seen == [1, 1]
            fill!(seen, 0)
            obssets = TimeDependent(ctx -> begin
                                        seen[ctx.i] += 1
                                        setsA
                                    end)
            p = cross_val_predict(Stacking(; sets = obssets, opti = [ew], opto = ew), rd,
                                  cvw)
            @test length(p.pred) == 2 && seen == [1, 1]
            fill!(seen, 0)
            obswf = TimeDependent(ctx -> begin
                                      seen[ctx.i] += 1
                                      IterativeWeightFinaliser()
                                  end)
            p = cross_val_predict(SubsetResampling(; wf = obswf, opt = ew, subset_size = 3,
                                                   n_subsets = 2), rd, cvw)
            @test length(p.pred) == 2 && seen == [1, 1]
            # A two-entry vector schedule in every widened field swaps entry i in per
            # fold, end to end.
            pv = cross_val_predict(NestedClustered(; pe = tdpe, cle = tdcle, sets = tdsets,
                                                   wf = tdwf, opti = mr0, opto = mr0), rd,
                                   cvw)
            @test length(pv.pred) == 2
            @test all(x -> isa(x.res.retcode, OptimisationSuccess), pv.pred)
        end
        @testset "A :nearest fallback is rejected at construction on every host" begin
            # The #138 bind decision: no fold loop ever consumes a fallback — the
            # fallback walk is a retry chain within one fold's solve — so :nearest has
            # nothing to bind to and is rejected outright, as on the metas.
            fbn = TimeDependent([ew, iv], :nearest)
            jopt = JuMPOptimiser(; slv = slv)
            @test_throws ArgumentError MeanRisk(; opt = jopt, fb = fbn)
            @test_throws ArgumentError NearOptimalCentering(; opt = jopt, fb = fbn)
            @test_throws ArgumentError RiskBudgeting(; opt = jopt, fb = fbn)
            @test_throws ArgumentError RelaxedRiskBudgeting(; opt = jopt, fb = fbn)
            @test_throws ArgumentError FactorRiskContribution(; opt = jopt, fb = fbn)
            @test_throws ArgumentError HierarchicalRiskParity(; fb = fbn)
            @test_throws ArgumentError HierarchicalEqualRiskContribution(; fb = fbn)
            @test_throws ArgumentError SchurComplementHierarchicalRiskParity(; fb = fbn)
            @test_throws ArgumentError InverseVolatility(; fb = fbn)
            @test_throws ArgumentError EqualWeighted(; fb = fbn)
            @test_throws ArgumentError RandomWeighted(; fb = fbn)
            # :outermost fallbacks still construct.
            @test isa(MeanRisk(; opt = jopt, fb = TimeDependent([ew, iv])), MeanRisk)
            @test isa(InverseVolatility(; fb = TimeDependent([ew, iv])), InverseVolatility)
        end
    end
    @testset "Recovering which entry produced a fold (#148)" begin
        # No provenance is stored on predictions; the docs claim it is recoverable across every
        # cross-validation form. These tests pin the contracts recovery leans on, once per
        # scheme. EqualWeighted/InverseVolatility are deterministic and distinguishable, so a
        # fold's weights identify the entry that ran. The single-thread executor keeps the
        # callable-logging runs race-free and reproducible.
        ew, iv = EqualWeighted(), InverseVolatility()
        seqex = PortfolioOptimisers.FLoops.SequentialEx()
        @testset "A vector schedule is keyed by the fold index" begin
            # The scheme-independent linchpin: entry i is fold i, resolved purely from ctx.i —
            # this is what makes `val[i]` the recovery answer for every vector schedule.
            sched = TimeDependent([ew, iv, ew])
            mkctx = i -> TimeDependentContext(; i = i, n = 3, rd = rd,
                                              train_idx = [[1], [2], [3]],
                                              test_idx = [[1], [2], [3]])
            for i in 1:3
                @test PortfolioOptimisers.time_dependent_value(sched, mkctx(i)) ===
                      sched.val[i]
            end
        end
        # Time-ordered single-path schemes: predictions come back in fold order, so fold i is
        # pred[i] and `sched.val[i]` names the optimiser behind it, end to end.
        function assert_fold_is_pred(cv, mkopt = identity)
            n = PortfolioOptimisers.n_splits(cv, rd)
            sched = TimeDependent([iseven(i) ? iv : ew for i in 1:n])
            mp = cross_val_predict(mkopt(sched), rd, cv)
            pe = cross_val_predict(mkopt(ew), rd, cv)
            pv = cross_val_predict(mkopt(iv), rd, cv)
            @test length(mp.pred) == n
            for i in 1:n
                expected = (sched.val[i] === iv ? pv : pe).pred[i].res.w
                @test isapprox(mp.pred[i].res.w, expected)
            end
            return n
        end
        @testset "KFold" begin
            @test assert_fold_is_pred(KFold(; n = 4)) == 4
        end
        @testset "WalkForward" begin
            # All walk-forward variants share one fold loop; IndexWalkForward stands in.
            @test assert_fold_is_pred(IndexWalkForward(100, 50)) == 2
        end
        @testset "Pipeline" begin
            # The pipeline fold loop resolves a schedule step per fold; recovery is the same
            # fold-order alignment for the time-ordered schemes. A returns-level pipeline
            # also runs combinatorial and MultipleRandomised (recovery identical to the plain
            # optimiser — same split, same recombination). A price-starting pipeline runs them
            # too: MR keeps rows contiguous, combinatorial accepts the boundary-return
            # approximation over its non-contiguous training rows.
            mkpipe = step -> Pipeline(steps = (EmpiricalPrior(), step))
            @test assert_fold_is_pred(KFold(; n = 3), mkpipe) == 3
            @test assert_fold_is_pred(IndexWalkForward(100, 50), mkpipe) == 2
            # Combinatorial: split->entry recovery holds for a pipeline schedule too.
            ccv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
            n_c = PortfolioOptimisers.n_splits(ccv)
            homog = cross_val_predict(mkpipe(TimeDependent(fill(iv, n_c))), rd, ccv)
            pv = cross_val_predict(mkpipe(iv), rd, ccv)
            @test [length(p.pred) for p in homog.pred] == [length(p.pred) for p in pv.pred]
            @test all(isapprox(a.res.w, b.res.w) for (pa, pb) in zip(homog.pred, pv.pred)
                      for (a, b) in zip(pa.pred, pb.pred))
            # MultipleRandomised: each path runs the inner walk-forward over its asset subset.
            mkmr = () -> MultipleRandomised(IndexWalkForward(100, 50); subset_size = 3,
                                            n_subsets = 2, rng = StableRNG(42), seed = 1)
            n_pp = PortfolioOptimisers.n_splits(IndexWalkForward(100, 50), rd)
            pm = cross_val_predict(mkpipe(TimeDependent([iseven(i) ? iv : ew
                                                         for i in 1:n_pp])), rd, mkmr())
            @test length(pm.pred) == 2
            @test all(path -> length(path.pred) == n_pp, pm.pred)
            # A price-starting pipeline now runs BOTH at the price level: combinatorial accepts
            # the boundary-return approximation (non-contiguous training rows), MR keeps rows
            # contiguous by drawing over assets.
            ts = range(; start = Date(2021, 1, 1), step = Day(1),
                       length = size(rd.X, 1) + 1)
            Xc = 100 .* cumprod(1 .+ vcat(zeros(1, size(rd.X, 2)), rd.X); dims = 1)
            pr = PricesResult(; X = TimeArray(collect(ts), Xc, rd.nx))
            ppipe = Pipeline(steps = (PricesToReturns(), EmpiricalPrior(), iv))
            cc = cross_val_predict(ppipe, pr, ccv)
            @test isa(cc, PortfolioOptimisers.PopulationPredictionResult)
            @test length(cc.pred) == maximum(split(ccv, pr).path_ids)
            @test all(isa(p.res.retcode, PortfolioOptimisers.OptimisationSuccess)
                      for pa in cc.pred for p in pa.pred)
            mr = cross_val_predict(ppipe, pr, mkmr())
            @test length(mr.pred) == 2
        end
        @testset "Combinatorial: split->path map recovers the entry behind each path" begin
            # Predictions are recombined into paths, but the schedule is keyed by split index
            # and `split` returns the split->path map, so the entry feeding a path is
            # recoverable without touching a prediction — and it matches what actually ran.
            ccv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
            n_c = PortfolioOptimisers.n_splits(ccv)
            sched = TimeDependent([iseven(j) ? iv : ew for j in 1:n_c])
            paths = split(ccv, rd).path_ids
            @test size(paths) == (ccv.n_test_folds, n_c)
            pp = cross_val_predict(sched, rd, ccv)
            pe = cross_val_predict(ew, rd, ccv)
            pv = cross_val_predict(iv, rd, ccv)
            # Every prediction in a path came from a split whose entry `paths` attributes to
            # that path; that entry's static backtest holds a matching prediction.
            for (path_id, path) in enumerate(pp.pred)
                entries = unique(sched.val[j] for j in 1:n_c if path_id in paths[:, j])
                pool = [p.res.w for e in entries for src in ((e === iv ? pv : pe),)
                        for pa in src.pred for p in pa.pred]
                for p in path.pred
                    @test any(w -> isapprox(w, p.res.w), pool)
                end
            end
        end
        @testset "MultipleRandomised: schedule keyed per-path, coordinates via ctx.path_id" begin
            # Each path runs the inner scheme's folds, so the schedule is keyed by the fold's
            # position *within a path*; a wrong length is rejected up front.
            inner = IndexWalkForward(100, 50)
            n_pp = PortfolioOptimisers.n_splits(inner, rd)
            # A fresh, same-seed estimator each call, so mixed and pure runs share one split.
            mkmr = () -> MultipleRandomised(inner; subset_size = 3, n_subsets = 2,
                                            rng = StableRNG(42), seed = 1)
            pp = cross_val_predict(TimeDependent([iseven(i) ? iv : ew for i in 1:n_pp]), rd,
                                   mkmr())
            @test length(pp.pred) == 2                       # n_subsets paths
            @test all(path -> length(path.pred) == n_pp, pp.pred)
            @test_throws DimensionMismatch cross_val_predict(TimeDependent([ew]), rd,
                                                             mkmr())
            # Vector schedule, observable: a homogeneous schedule reproduces the pure run
            # exactly, and a mixed one matches one of the two pure strategies at every
            # position — the schedule really is consumed per fold within each path.
            pe = cross_val_predict(ew, rd, mkmr())
            pv = cross_val_predict(iv, rd, mkmr())
            homog = cross_val_predict(TimeDependent(fill(iv, n_pp)), rd, mkmr())
            for p in 1:2, k in 1:n_pp
                @test isapprox(homog.pred[p].pred[k].res.w, pv.pred[p].pred[k].res.w)
                w = pp.pred[p].pred[k].res.w
                @test isapprox(w, pe.pred[p].pred[k].res.w) ||
                      isapprox(w, pv.pred[p].pred[k].res.w)
            end
            # A callable recovers its per-fold decision from `(ctx.path_id, ctx.i)`: run
            # sequentially so the shared log is race-free, and every (path, fold) is recorded.
            log = Tuple{Union{Int, Nothing}, Int, Symbol}[]
            cross_val_predict(TimeDependent(TDLog(ew, iv, log); default = ew), rd, mkmr();
                              ex = seqex)
            for p in 1:2
                picks = sort([(e[2], e[3]) for e in log if e[1] == p])
                @test picks == [(i, isodd(i) ? :odd : :even) for i in 1:n_pp]
            end
        end
        @testset "A callable logs its own per-fold decision (single path)" begin
            cvw = IndexWalkForward(100, 50)
            n = PortfolioOptimisers.n_splits(cvw, rd)
            log = Tuple{Union{Int, Nothing}, Int, Symbol}[]
            mp = cross_val_predict(TimeDependent(TDLog(ew, iv, log); default = ew), rd, cvw;
                                   ex = seqex)
            @test Set(log) == Set([(nothing, i, isodd(i) ? :odd : :even) for i in 1:n])
            pe = cross_val_predict(ew, rd, cvw)
            pv = cross_val_predict(iv, rd, cvw)
            for i in 1:n
                picked = log[findfirst(e -> e[2] == i, log)][3]
                @test isapprox(mp.pred[i].res.w, (picked === :odd ? pe : pv).pred[i].res.w)
            end
        end
    end
end
