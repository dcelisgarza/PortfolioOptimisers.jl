# TimeDependentCallable subtypes must be defined at top level. `TDCap` declares its
# target via the trait; `TDPrevWCap` declares its previous-weights requirement directly.
struct TDCap <: PortfolioOptimisers.TimeDependentCallable
    hi::Float64
    lo::Float64
end
function (c::TDCap)(ctx::TimeDependentContext)
    return WeightBounds(; lb = 0.0,
                        ub = c.hi - (c.hi - c.lo) * (ctx.i - 1) / max(ctx.n - 1, 1))
end
function PortfolioOptimisers.default_time_dependent_target(::TDCap)
    return :wb
end
struct TDPrevWCap <: PortfolioOptimisers.TimeDependentCallable end
function (c::TDPrevWCap)(ctx::TimeDependentContext)
    return Threshold(; val = isnothing(ctx.w_prev) ? 0.01 : 0.05)
end
function PortfolioOptimisers.needs_previous_weights(::TDPrevWCap)
    return true
end
struct TDNoTrait <: PortfolioOptimisers.TimeDependentCallable end
function (c::TDNoTrait)(ctx::TimeDependentContext)
    return nothing
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
        # Target inference from unambiguous value types.
        w0 = fill(0.2, 5)
        td_tn = TimeDependent([Turnover(; w = w0, val = 0.5),
                               Turnover(; w = w0, val = 0.3)])
        @test td_tn.field == :tn
        td_fees = TimeDependent([Fees(; l = 0.001), Fees(; l = 0.002)])
        @test td_fees.field == :fees
        td_wb = TimeDependent([WeightBounds(; lb = 0.0, ub = 0.5),
                               WeightBounds(; lb = 0.0, ub = 0.6)])
        @test td_wb.field == :wb
        # nothing entries participate in inference without breaking it.
        @test TimeDependent([nothing, Fees(; l = 0.001)]).field == :fees
        # Ambiguous value types demand an explicit target.
        @test_throws ArgumentError TimeDependent([Threshold(; val = 0.01)])
        @test TimeDependent([Threshold(; val = 0.01)], :lt).field == :lt
        @test TimeDependent([Threshold(; val = 0.01)], :st).field == :st
        # Callables demand an explicit target.
        @test_throws ArgumentError TimeDependent(ctx -> nothing)
        @test TimeDependent(ctx -> Threshold(; val = 0.01), :lt).field == :lt
        # Empty entry vectors are rejected.
        @test_throws PortfolioOptimisers.IsEmptyError TimeDependent(Union{}[], :lt)
        # Keyword form.
        @test TimeDependent(; val = [Fees(; l = 0.001)]).field == :fees
    end
    @testset "Host validation" begin
        td = TimeDependent([Threshold(; val = 0.01), Threshold(; val = 0.02)], :lt)
        opt = JuMPOptimiser(; slv = slv, tdc = td)
        @test PortfolioOptimisers.is_time_dependent(opt)
        @test isnothing(opt.lt)
        @test !PortfolioOptimisers.needs_previous_weights(opt)
        # Sole-source rule: targeted field must be left at its default.
        @test_throws ArgumentError JuMPOptimiser(; slv = slv, lt = Threshold(; val = 0.5),
                                                 tdc = td)
        @test_throws ArgumentError JuMPOptimiser(; slv = slv,
                                                 wb = WeightBounds(; lb = 0.1, ub = 0.9),
                                                 tdc = TimeDependent(ctx -> WeightBounds(),
                                                                     :wb))
        # wb left at its constructor default is fine.
        @test JuMPOptimiser(; slv = slv,
                            tdc = TimeDependent(ctx -> WeightBounds(), :wb)) isa
              JuMPOptimiser
        # Unknown and duplicate targets.
        @test_throws ArgumentError JuMPOptimiser(; slv = slv,
                                                 tdc = TimeDependent([Threshold(;
                                                                                val = 0.01)],
                                                                     :nope))
        @test_throws ArgumentError JuMPOptimiser(; slv = slv,
                                                 tdc = [td,
                                                        TimeDependent([Threshold(;
                                                                                 val = 0.3)],
                                                                      :lt)])
        # Test-substitution surfaces type-incompatible entries at construction.
        @test_throws TypeError JuMPOptimiser(; slv = slv,
                                             tdc = TimeDependent([Threshold(; val = 0.01)],
                                                                 :card))
        # Hierarchical host.
        htd = TimeDependent([Fees(; l = 0.001), Fees(; l = 0.002)])
        hopt = HierarchicalOptimiser(; slv = slv, tdc = htd)
        @test PortfolioOptimisers.is_time_dependent(hopt)
        @test_throws ArgumentError HierarchicalOptimiser(; slv = slv,
                                                         fees = Fees(; l = 0.1), tdc = htd)
        @test_throws ArgumentError HierarchicalOptimiser(; slv = slv,
                                                         tdc = TimeDependent([Threshold(;
                                                                                        val = 0.01)],
                                                                             :lt))
    end
    @testset "Per-fold resolution and wrapper recursion" begin
        td = TimeDependent([Threshold(; val = 0.01), Threshold(; val = 0.02)], :lt)
        opt = JuMPOptimiser(; slv = slv, tdc = td)
        ctx1 = TimeDependentContext(; i = 1, n = 2, rd = rd, train_idx = [1:100, 1:150],
                                    test_idx = [101:150, 151:200])
        ctx2 = TimeDependentContext(; i = 2, n = 2, rd = rd, train_idx = [1:100, 1:150],
                                    test_idx = [101:150, 151:200])
        o1 = PortfolioOptimisers.update_time_dependent_estimator(opt, ctx1)
        o2 = PortfolioOptimisers.update_time_dependent_estimator(opt, ctx2)
        @test o1.lt.val == 0.01
        @test o2.lt.val == 0.02
        @test isnothing(o1.tdc) && isnothing(o2.tdc)
        # Wrapper recursion rebuilds the inner optimiser and clears its tdc.
        mr = MeanRisk(; opt = opt)
        @test PortfolioOptimisers.is_time_dependent(mr)
        mr2 = PortfolioOptimisers.update_time_dependent_estimator(mr, ctx2)
        @test mr2.opt.lt.val == 0.02
        @test isnothing(mr2.opt.tdc)
        # Function form receives the fold context.
        tdwb = TimeDependent(ctx -> WeightBounds(; lb = 0.0, ub = 1.0 / ctx.i), :wb)
        optwb = JuMPOptimiser(; slv = slv, tdc = tdwb)
        @test PortfolioOptimisers.update_time_dependent_estimator(optwb, ctx2).wb.ub == 0.5
        # PreviousWeightsFunction declares the prev-weights requirement as data.
        pwf = TimeDependent(PreviousWeightsFunction(ctx -> Threshold(;
                                                                     val = if isnothing(ctx.w_prev)
                                                                         0.01
                                                                     else
                                                                         0.05
                                                                     end)), :lt)
        optp = JuMPOptimiser(; slv = slv, tdc = pwf)
        @test PortfolioOptimisers.needs_previous_weights(optp)
        @test PortfolioOptimisers.update_time_dependent_estimator(optp, ctx1).lt.val == 0.01
        # Fold-count validation.
        @test_throws DimensionMismatch PortfolioOptimisers.assert_time_dependent_fold_count(opt,
                                                                                            5)
        @test isnothing(PortfolioOptimisers.assert_time_dependent_fold_count(opt, 2))
        @test isnothing(PortfolioOptimisers.assert_time_dependent_fold_count(optwb, 5))
        # Callable structs: functor over ctx, trait-inferred target, direct
        # needs_previous_weights declaration; equivalent to the bare-function form.
        tdcap = TimeDependent(TDCap(0.35, 0.2))
        @test tdcap.field == :wb
        optcap = JuMPOptimiser(; slv = slv, tdc = tdcap)
        @test !PortfolioOptimisers.needs_previous_weights(optcap)
        tdfn = TimeDependent(ctx -> WeightBounds(; lb = 0.0,
                                                 ub = 0.35 -
                                                      0.15 * (ctx.i - 1) /
                                                      max(ctx.n - 1, 1)), :wb)
        optfn = JuMPOptimiser(; slv = slv, tdc = tdfn)
        for ctx in (ctx1, ctx2)
            oc = PortfolioOptimisers.update_time_dependent_estimator(optcap, ctx)
            of = PortfolioOptimisers.update_time_dependent_estimator(optfn, ctx)
            @test oc.wb.ub ≈ of.wb.ub
        end
        optpw = JuMPOptimiser(; slv = slv, tdc = TimeDependent(TDPrevWCap(), :lt))
        @test PortfolioOptimisers.needs_previous_weights(optpw)
        @test PortfolioOptimisers.update_time_dependent_estimator(optpw, ctx1).lt.val ==
              0.01
        # A subtype without the target trait still requires an explicit field.
        @test_throws ArgumentError TimeDependent(TDNoTrait())
        @test TimeDependent(TDNoTrait(), :lt) isa TimeDependent
        # Meta-optimisers forward the traits and the per-fold update to their inner
        # estimators.
        sr = SubsetResampling(; opt = mr)
        @test PortfolioOptimisers.is_time_dependent(sr)
        sr2 = PortfolioOptimisers.update_time_dependent_estimator(sr, ctx2)
        @test sr2.opt.opt.lt.val == 0.02
        @test isnothing(sr2.opt.opt.tdc)
        st = Stacking(; opti = [mr, MeanRisk(; opt = JuMPOptimiser(; slv = slv))],
                      opto = MeanRisk(; opt = JuMPOptimiser(; slv = slv)))
        @test PortfolioOptimisers.is_time_dependent(st)
        st2 = PortfolioOptimisers.update_time_dependent_estimator(st, ctx1)
        @test st2.opti[1].opt.lt.val == 0.01
        @test isnothing(st2.opti[1].opt.tdc)
        @test_throws DimensionMismatch PortfolioOptimisers.assert_time_dependent_fold_count(st,
                                                                                            9)
        # Asset views map through vector entries.
        tdv = TimeDependent([WeightBounds(; lb = zeros(5), ub = fill(0.8, 5)),
                             WeightBounds(; lb = zeros(5), ub = fill(0.9, 5))])
        tdv2 = PortfolioOptimisers.port_opt_view(tdv, 1:3)
        @test length(tdv2.val[1].ub) == 3
        @test tdv2.field == :wb
    end
    @testset "Inert outside fold loops" begin
        td = TimeDependent([Threshold(; val = 0.01), Threshold(; val = 0.02)], :lt)
        mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv, tdc = td))
        # A fold-less optimise runs with the targeted field at its default.
        res = optimise(mr, rd)
        @test isa(res.retcode, OptimisationSuccess)
        mr0 = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        res0 = optimise(mr0, rd)
        @test res.w == res0.w
    end
    @testset "Walk-forward end to end" begin
        cv = IndexWalkForward(100, 50)
        n = n_splits(cv, rd)
        @test n == 2
        # Tight fold-2 bound forces different weights across folds.
        tdwb = TimeDependent([WeightBounds(; lb = 0.0, ub = 1.0),
                              WeightBounds(; lb = 0.15, ub = 0.25)])
        mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv, tdc = tdwb))
        preds = cross_val_predict(mr, rd, cv)
        @test length(preds.pred) == 2
        w1 = preds.pred[1].res.w
        w2 = preds.pred[2].res.w
        @test all(x -> 0.15 - 1e-6 <= x <= 0.25 + 1e-6, w2)
        @test !isapprox(w1, w2)
        # Mis-sized entries fail at split time, before any fold runs.
        td3 = TimeDependent([WeightBounds(), WeightBounds(), WeightBounds()], :wb)
        mr3 = MeanRisk(; opt = JuMPOptimiser(; slv = slv, tdc = td3))
        @test_throws DimensionMismatch cross_val_predict(mr3, rd, cv)
        # Baseline without tdc differs on fold 2.
        mrb = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        predsb = cross_val_predict(mrb, rd, cv)
        @test isapprox(w1, predsb.pred[1].res.w)
        @test !isapprox(w2, predsb.pred[2].res.w)
        # Function form: same bounds computed on the fly.
        tdf = TimeDependent(ctx -> if ctx.i == 1
                                WeightBounds()
                            else
                                WeightBounds(; lb = 0.15, ub = 0.25)
                            end, :wb)
        mrf = MeanRisk(; opt = JuMPOptimiser(; slv = slv, tdc = tdf))
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
                                                    end), :wb)
        mrp = MeanRisk(; opt = JuMPOptimiser(; slv = slv, tdc = pwf))
        predsp = cross_val_predict(mrp, rd, cv)
        @test length(seen) == 2
        @test isnothing(seen[1])
        @test isapprox(seen[2], predsp.pred[1].res.w)
    end
    @testset "KFold and hierarchical end to end" begin
        cv = KFold(; n = 4)
        tdwb = TimeDependent([WeightBounds(; lb = 0.15, ub = 0.25), WeightBounds(),
                              WeightBounds(), WeightBounds()], :wb)
        mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv, tdc = tdwb))
        preds = cross_val_predict(mr, rd, cv)
        @test length(preds.pred) == 4
        @test all(x -> 0.15 - 1e-6 <= x <= 0.25 + 1e-6, preds.pred[1].res.w)
        # Hierarchical host under walk-forward.
        cvw = IndexWalkForward(100, 50)
        htd = TimeDependent([Fees(; l = 0.0), Fees(; l = 0.01)])
        hrp = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; slv = slv, tdc = htd))
        hpreds = cross_val_predict(hrp, rd, cvw)
        @test length(hpreds.pred) == 2
        @test isnothing(hpreds.pred[1].res.fees) || hpreds.pred[1].res.fees.l == 0.0
        @test hpreds.pred[2].res.fees.l == 0.01
    end
end
